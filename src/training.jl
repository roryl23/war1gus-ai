# All feature extraction and trajectory bookkeeping happen before the epoch loop.
# Columns retain catalog order; offsets are one-based and include a final sentinel.
struct PpoBatch
 headers::Matrix{Float32}
 entities::Matrix{Float32}
 candidates::Matrix{Float32}
 entity_offsets::Vector{Int}
 candidate_offsets::Vector{Int}
 first_waits::Vector{Bool}
 exploration_weights::Vector{Float32}
 exploration_weight_sums::Vector{Float32}
 candidate_states::Vector{Int}
 actors::Vector{Int}
 targets_entity::Vector{Int}
 actions::Vector{Int}
 bootstrap_scores::Vector{Float32}
 old_log_probabilities::Vector{Float32}
 old_values::Vector{Float32}
 targets::Vector{Float32}
 advantages::Vector{Float32}
end

function _pack_ppo_batch(
 steps::Vector{TrajectoryStep}, targets::Vector{Float32}, advantages::Vector{Float32},
)::PpoBatch
 count = length(steps)
 total_entities = sum(length(step.observation.entities) for step in steps)
 total_candidates = sum(length(step.observation.candidates) for step in steps)
 headers = Matrix{Float32}(undef, STATE_HEADER_WORDS, count)
 entities = Matrix{Float32}(undef, ENTITY_WORDS, total_entities)
 candidates = Matrix{Float32}(undef, CANDIDATE_WORDS, total_candidates)
 entity_offsets = Vector{Int}(undef, count + 1)
 candidate_offsets = Vector{Int}(undef, count + 1)
 first_waits = Vector{Bool}(undef, count)
 exploration_weights = Vector{Float32}(undef, total_candidates)
 exploration_weight_sums = Vector{Float32}(undef, count)
 candidate_states = Vector{Int}(undef, total_candidates)
 actors = Vector{Int}(undef, total_candidates)
 targets_entity = Vector{Int}(undef, total_candidates)
 actions = Vector{Int}(undef, count)
 bootstrap_scores = Vector{Float32}(undef, total_candidates)
 old_log_probabilities = Vector{Float32}(undef, count)
 old_values = Vector{Float32}(undef, count)

 next_entity = 1
 next_candidate = 1
 for index in eachindex(steps)
  step = steps[index]
  observation = step.observation
  isempty(observation.candidates) &&
   throw(ArgumentError("terminal states have no selectable candidates"))
  0 <= step.action < length(observation.candidates) ||
   throw(ArgumentError("candidate index $(step.action) is outside the supplied candidate sequence"))
  entity_offsets[index] = next_entity
  first_waits[index] = candidate_kind(first(observation.candidates)) == 0
  exploration_weight_sums[index] = _training_exploration_weight_sum(
   observation.candidates, first_waits[index] && length(observation.candidates) > 1,
  )
  candidate_offsets[index] = next_candidate
  _feature_column!(headers, index, _header_features(observation.header))
  for entity in observation.entities
   _feature_column!(entities, next_entity, _entity_features(entity))
   next_entity += 1
  end
  for candidate in observation.candidates
   _feature_column!(candidates, next_candidate, _candidate_features(candidate))
   exploration_weights[next_candidate] = _training_exploration_weight(candidate)
   candidate_states[next_candidate] = index
   actor = candidate_actor(candidate)
   target = candidate_target(candidate)
   # Global indices are fixed here; one past the final entity is a zero reference.
   actors[next_candidate] = iszero(actor) ? total_entities + 1 : entity_offsets[index] + actor - 1
   targets_entity[next_candidate] = iszero(target) ? total_entities + 1 : entity_offsets[index] + target - 1
   bootstrap_scores[next_candidate] = candidate_bootstrap_score(candidate)
   next_candidate += 1
  end
  actions[index] = candidate_offsets[index] + step.action
  old_log_probabilities[index] = step.old_log_probability
  old_values[index] = step.old_value
 end
 entity_offsets[end] = next_entity
 candidate_offsets[end] = next_candidate
 return PpoBatch(
  headers, entities, candidates, entity_offsets, candidate_offsets, first_waits,
  exploration_weights, exploration_weight_sums, candidate_states,
  actors, targets_entity, actions, bootstrap_scores,
  old_log_probabilities, old_values, targets, advantages,
 )
end

# Pool the entities belonging to each observation and pack actor and target
# embeddings in score-head order. This function's rule accumulates all three
# paths into one embedding gradient instead of allocating a full E-column
# gradient for each independent gather.
function _entity_context_and_references(
 embeddings::Matrix{Float32}, offsets::Vector{Int},
 actors::Vector{Int}, targets::Vector{Int},
)
 dimensions, entity_count = size(embeddings)
 context = zeros(Float32, dimensions, length(offsets) - 1)
 references = zeros(Float32, 2 * dimensions, length(actors))
 for state in axes(context, 2)
  first_entity = offsets[state]
  last_entity = offsets[state+1] - 1
  if first_entity <= last_entity
   for entity in first_entity:last_entity, dimension in 1:dimensions
    context[dimension, state] += embeddings[dimension, entity]
   end
   scale = 1.0f0 / Float32(last_entity - first_entity + 1)
   for dimension in 1:dimensions
    context[dimension, state] *= scale
   end
  end
 end
 for candidate in eachindex(actors)
  actor = actors[candidate]
  target = targets[candidate]
  if actor <= entity_count
   for dimension in 1:dimensions
    references[dimension, candidate] = embeddings[dimension, actor]
   end
  end
  if target <= entity_count
   for dimension in 1:dimensions
    references[dimensions+dimension, candidate] = embeddings[dimension, target]
   end
  end
 end
 return context, references
end

function ChainRulesCore.rrule(
 ::typeof(_entity_context_and_references),
 embeddings::Matrix{Float32}, offsets::Vector{Int},
 actors::Vector{Int}, targets::Vector{Int},
)
 outputs = _entity_context_and_references(embeddings, offsets, actors, targets)
 function entity_pullback(cotangent)
  cotangent = ChainRulesCore.unthunk(cotangent)
  if cotangent isa ChainRulesCore.AbstractZero
   return (
    ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(),
    ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
   )
  end
  context_gradient = ChainRulesCore.unthunk(cotangent[1])
  reference_gradient = ChainRulesCore.unthunk(cotangent[2])
  dimensions, entity_count = size(embeddings)
  gradient = zeros(Float32, dimensions, entity_count)
  if !(context_gradient isa ChainRulesCore.AbstractZero)
   for state in axes(context_gradient, 2)
    first_entity = offsets[state]
    last_entity = offsets[state+1] - 1
    if first_entity <= last_entity
     scale = 1.0f0 / Float32(last_entity - first_entity + 1)
     for entity in first_entity:last_entity, dimension in 1:dimensions
      gradient[dimension, entity] += context_gradient[dimension, state] * scale
     end
    end
   end
  end
  if !(reference_gradient isa ChainRulesCore.AbstractZero)
   for candidate in eachindex(actors)
    actor = actors[candidate]
    target = targets[candidate]
    if actor <= entity_count
     for dimension in 1:dimensions
      gradient[dimension, actor] += reference_gradient[dimension, candidate]
     end
    end
    if target <= entity_count
     for dimension in 1:dimensions
      gradient[dimension, target] += reference_gradient[dimensions+dimension, candidate]
     end
    end
   end
  end
  return (
   ChainRulesCore.NoTangent(), gradient,
   ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
  )
 end
 return outputs, entity_pullback
end

# The segmented catalog derivative avoids a full-catalog tangent per state.
function _segmented_policy_statistics_with_cache(
 scores::Vector{Float32}, offsets::Vector{Int}, actions::Vector{Int}, first_waits::Vector{Bool},
 exploration_weights::Vector{Float32}, exploration_weight_sums::Vector{Float32},
)
 count = length(actions)
 statistics = Matrix{Float32}(undef, 2, count)
 log_probabilities = similar(scores)
 probabilities = similar(scores)
 for index in eachindex(actions)
  first_candidate = offsets[index]
  last_candidate = offsets[index+1] - 1
  skip_wait = first_waits[index] && last_candidate > first_candidate
  weight_sum = exploration_weight_sums[index]
  segment_log_probabilities = Flux.logsoftmax(@view scores[first_candidate:last_candidate])
  entropy = 0.0f0
  for candidate in first_candidate:last_candidate
   local_index = candidate - first_candidate + 1
   policy_log_probability = segment_log_probabilities[local_index]
   log_probability = _training_mixture_log_probability(
    policy_log_probability, local_index, skip_wait,
    TRAIN_EXPLORATION_FRACTION * exploration_weights[candidate] / weight_sum,
   )
   log_probabilities[candidate] = log_probability
   probabilities[candidate] = exp(policy_log_probability)
   entropy -= exp(log_probability) * log_probability
  end
  statistics[1, index] = log_probabilities[actions[index]]
  statistics[2, index] = entropy
 end
 return statistics, log_probabilities, probabilities
end

_segmented_policy_statistics(
 scores::Vector{Float32}, offsets::Vector{Int}, actions::Vector{Int}, first_waits::Vector{Bool},
 exploration_weights::Vector{Float32}, exploration_weight_sums::Vector{Float32},
) = first(_segmented_policy_statistics_with_cache(
 scores, offsets, actions, first_waits, exploration_weights, exploration_weight_sums,
))

function ChainRulesCore.rrule(
 ::typeof(_segmented_policy_statistics),
 scores::Vector{Float32}, offsets::Vector{Int}, actions::Vector{Int}, first_waits::Vector{Bool},
 exploration_weights::Vector{Float32}, exploration_weight_sums::Vector{Float32},
)
 statistics, log_probabilities, probabilities =
  _segmented_policy_statistics_with_cache(
   scores, offsets, actions, first_waits, exploration_weights, exploration_weight_sums,
  )
 function segmented_pullback(cotangent)
  cotangent = ChainRulesCore.unthunk(cotangent)
  if cotangent isa ChainRulesCore.AbstractZero
   return (
    ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(),
    ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
    ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
   )
  end
  score_gradient = similar(scores)
  for index in eachindex(actions)
   log_probability_gradient = cotangent[1, index]
   entropy_gradient = cotangent[2, index]
   first_candidate = offsets[index]
   last_candidate = offsets[index+1] - 1
   action = actions[index]
   # The policy's softmax p is the only score-dependent part of q.
   # Summing p_i log(q_i) (not q_i log(q_i)) is required by dH(q)/ds.
   policy_weighted_log_probability = 0.0f0
   for candidate in first_candidate:last_candidate
    policy_weighted_log_probability += probabilities[candidate] * log_probabilities[candidate]
   end
   skip_wait = first_waits[index] && last_candidate > first_candidate
   action_factor = skip_wait && action == first_candidate ? 1.0f0 :
                   TRAIN_POLICY_FRACTION *
                   probabilities[action] / exp(log_probabilities[action])
   for candidate in first_candidate:last_candidate
    probability = probabilities[candidate]
    score_gradient[candidate] =
     log_probability_gradient * action_factor * ((candidate == action ? 1.0f0 : 0.0f0) - probability) -
     entropy_gradient * TRAIN_POLICY_FRACTION * probability *
     (log_probabilities[candidate] - policy_weighted_log_probability)
   end
  end
  return (
   ChainRulesCore.NoTangent(), score_gradient,
   ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
   ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
  )
 end
 return statistics, segmented_pullback
end

function _ppo_loss(
 policy::AiPolicy, batch::PpoBatch, clip_epsilon::Float32,
 value_coefficient::Float32, entropy_coefficient::Float32,
)
 count = length(batch.actions)
 header_embeddings = policy.header_encoder(batch.headers)
 if isempty(batch.entities)
  # Calling the encoder with empty columns would produce a zero gradient in
  # place of the old path's `nothing`, advancing Adam state on empty batches.
  entity_context = zeros(Float32, EMBED_DIM, count)
  references = zeros(Float32, 2 * EMBED_DIM, length(batch.actors))
 else
  entity_embeddings = policy.entity_encoder(batch.entities)
  entity_context, references = _entity_context_and_references(
   entity_embeddings, batch.entity_offsets, batch.actors, batch.targets_entity,
  )
 end
 global_context = header_embeddings .+ entity_context
 candidate_embeddings = policy.candidate_encoder(batch.candidates)
 score_inputs = vcat(
  global_context[:, batch.candidate_states], candidate_embeddings, references,
 )
 scores = vec(policy.score_head(score_inputs)) .+ batch.bootstrap_scores
 values = vec(policy.value_head(vcat(global_context, entity_context)))

 # Only the segmented reductions use custom rules; clipping retains ordinary
 # Flux AD semantics, including min/max/clamp tie behavior.
 statistics = _segmented_policy_statistics(
  scores, batch.candidate_offsets, batch.actions, batch.first_waits,
  batch.exploration_weights, batch.exploration_weight_sums,
 )
 log_probabilities = @view statistics[1, :]
 entropy = @view statistics[2, :]
 ratio = exp.(log_probabilities .- batch.old_log_probabilities)
 unclipped_policy = ratio .* batch.advantages
 clipped_policy = clamp.(ratio, 1.0f0 - clip_epsilon, 1.0f0 + clip_epsilon) .* batch.advantages
 actor_loss = .-min.(unclipped_policy, clipped_policy)
 unclipped_value_loss = (values .- batch.targets) .^ 2
 clipped_values = batch.old_values .+
                  clamp.(values .- batch.old_values, -clip_epsilon, clip_epsilon)
 clipped_value_loss = (clipped_values .- batch.targets) .^ 2
 critic_loss = 0.5f0 .* max.(unclipped_value_loss, clipped_value_loss)
 return sum(actor_loss .+ value_coefficient .* critic_loss .-
            entropy_coefficient .* entropy) / Float32(count)
end
