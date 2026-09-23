import ChainRulesCore
using Flux
using LinearAlgebra
using Random
using Serialization
using Statistics

const STATE_VERSION = UInt32(3)
const STATE_HEADER_WORDS = 22
const ENTITY_WORDS = 14
const CANDIDATE_WORDS = 12
const MAX_CANDIDATES = 512
const MAX_STATE_WORDS = 65_536
const MAX_ENTITY_COUNT = div(MAX_STATE_WORDS - STATE_HEADER_WORDS, ENTITY_WORDS)
const CANDIDATE_KIND_COUNT = 13
const EMBED_DIM = 48
const HIDDEN_DIM = 96
const CHECKPOINT_VERSION = 4
const CATALOG_VERSION = 4
const REWARD_VERSION = 3
const POLICY_VERSION = 3
const PPO_VERSION = 1
const DEFAULT_RANDOM_SEED = 0x574131
const DEFAULT_BATCH_SIZE = 128
const DEFAULT_ROLLOUT_FRAGMENT = 32
const DEFAULT_PPO_EPOCHS = 4
const DEFAULT_CHECKPOINT_EVERY = 16
const DEFAULT_LEAGUE_SNAPSHOT_EVERY = 8
const DEFAULT_LEAGUE_MAX_SNAPSHOTS = 8
const PPO_ALGORITHM = "trajectory-ppo-gae-v1"


function training_seed_from_environment()::Int
 raw = get(ENV, "WAR1GUS_AI_SEED", "")
 isempty(raw) && return DEFAULT_RANDOM_SEED
 seed = tryparse(Int, raw)
 !isnothing(seed) || throw(ArgumentError("WAR1GUS_AI_SEED must be an integer"))
 return seed
end
const MODE_INFERENCE = :inference
const MODE_TRAIN = :train
const MODE_RESET_TRAIN = :reset_train
const MODE_LEAGUE = :league
const MODE_LEAGUE_EVALUATE = :league_evaluate

const CANDIDATE_NAMES = (
 "wait",
 "gather-gold",
 "gather-wood",
 "build",
 "train",
 "research",
 "attack-entity",
 "move-group",
 "explore",
 "repair",
 "formation",
 "defend",
 "cast-spell",
)

"""Return the stable name for a v3 candidate kind."""
function candidate_name(kind::Integer)::String
 0 <= kind < CANDIDATE_KIND_COUNT || throw(ArgumentError("candidate kind $kind is outside the v3 catalog"))
 return CANDIDATE_NAMES[Int(kind)+1]
end

struct EntityObservation
 words::NTuple{ENTITY_WORDS,UInt32}
end

struct CandidateObservation
 words::NTuple{CANDIDATE_WORDS,UInt32}
end

struct StateObservation
 header::NTuple{STATE_HEADER_WORDS,UInt32}
 entities::Vector{EntityObservation}
 candidates::Vector{CandidateObservation}
end

@inline player_of(state::AbstractVector{<:Integer}) = UInt32(state[2])
@inline state_entity_count(state::AbstractVector{<:Integer}) = Int(UInt32(state[11]))
@inline state_candidate_count(state::AbstractVector{<:Integer}) = Int(UInt32(state[12]))
@inline candidate_kind(candidate::CandidateObservation) = Int(candidate.words[1])
@inline candidate_actor(candidate::CandidateObservation) = Int(candidate.words[2])
@inline candidate_target(candidate::CandidateObservation) = Int(candidate.words[3])
@inline candidate_bootstrap_score(candidate::CandidateObservation) = _signed_word(candidate.words[12]) / 1_000.0f0

@inline function _signed_word(word::UInt32)::Float32
 return word <= UInt32(typemax(Int32)) ? Float32(word) :
        Float32(Int64(word) - 4_294_967_296)
end

"""Return separately logged shaping components carried in a v3 state header."""
function reward_components(state::AbstractVector{<:Integer})
 length(state) >= STATE_HEADER_WORDS || throw(ArgumentError("AI state has $(length(state)) words, shorter than its v3 header"))
 return (
  enemy_progress=_signed_word(UInt32(state[19])),
  own_loss=_signed_word(UInt32(state[20])),
  time=_signed_word(UInt32(state[21])),
  terminal_component=_signed_word(UInt32(state[22])),
 )
end

"""Validate a complete variable-length v3 state and its authoritative candidate catalog."""
function validate_state(
 state::AbstractVector{<:Integer};
 terminal::Bool=false,
 frame_candidate_count::Union{Nothing,Integer}=nothing,
)::Nothing
 length(state) >= STATE_HEADER_WORDS ||
  throw(ArgumentError("AI state has $(length(state)) words, shorter than its v3 header"))
 UInt32(state[1]) == STATE_VERSION ||
  throw(ArgumentError("unsupported AI state protocol version $(state[1])"))

 entity_count = state_entity_count(state)
 candidate_count = state_candidate_count(state)
 entity_count <= MAX_ENTITY_COUNT ||
  throw(ArgumentError("AI state has $entity_count entities, exceeding the defensive cap of $MAX_ENTITY_COUNT"))
 candidate_count <= MAX_CANDIDATES ||
  throw(ArgumentError("AI state has $candidate_count candidates, exceeding the defensive cap of $MAX_CANDIDATES"))
 if terminal
  candidate_count == 0 || throw(ArgumentError("AI terminal state must not contain candidates"))
 else
  candidate_count >= 1 || throw(ArgumentError("AI step state must contain at least the wait candidate"))
 end
 if !isnothing(frame_candidate_count)
  Int(frame_candidate_count) == candidate_count ||
   throw(ArgumentError("AI frame candidate count $(frame_candidate_count) disagrees with state candidate count $candidate_count"))
 end

 expected_words = STATE_HEADER_WORDS + ENTITY_WORDS * entity_count + CANDIDATE_WORDS * candidate_count
 expected_words <= MAX_STATE_WORDS ||
  throw(ArgumentError("AI state has $expected_words words, exceeding the protocol cap of $MAX_STATE_WORDS"))
 length(state) == expected_words ||
  throw(ArgumentError("AI state has $(length(state)) words, expected $expected_words from its v3 counts"))

 candidate_count == 0 && return nothing
 first_candidate = STATE_HEADER_WORDS + ENTITY_WORDS * entity_count + 1
 UInt32(state[first_candidate]) == 0 ||
  throw(ArgumentError("AI candidate 1 must be the wait candidate"))
 for candidate_index in 1:candidate_count
  offset = first_candidate + (candidate_index - 1) * CANDIDATE_WORDS
  kind = Int(UInt32(state[offset]))
  0 <= kind < CANDIDATE_KIND_COUNT ||
   throw(ArgumentError("AI candidate $candidate_index has unknown kind $kind"))
  for field in (2, 3)
   entity_index = Int(UInt32(state[offset+field-1]))
   0 <= entity_index <= entity_count ||
    throw(ArgumentError("AI candidate $candidate_index references entity $entity_index outside 0:$entity_count"))
  end
 end
 return nothing
end

# A separate call keeps the record offset immutable in the ntuple closure. Capturing
# parse_state's incremented offset would box it once per record.
@inline function _record_words(
 state::AbstractVector{<:Integer},
 offset::Int,
 ::Val{N},
)::NTuple{N,UInt32} where N
 return ntuple(field -> UInt32(@inbounds state[offset+field-1]), Val(N))
end

"""Parse a complete v3 state into header, entity, and candidate records."""
function parse_state(
 state::AbstractVector{<:Integer};
 terminal::Bool=false,
 validated::Bool=false,
)::StateObservation
 Base.require_one_based_indexing(state)
 validated || validate_state(state; terminal)
 header = _record_words(state, 1, Val(STATE_HEADER_WORDS))
 entity_count = Int(header[11])
 candidate_count = Int(header[12])
 entities = Vector{EntityObservation}(undef, entity_count)
 offset = STATE_HEADER_WORDS + 1
 for index in eachindex(entities)
  entities[index] = EntityObservation(_record_words(state, offset, Val(ENTITY_WORDS)))
  offset += ENTITY_WORDS
 end
 candidates = Vector{CandidateObservation}(undef, candidate_count)
 for index in eachindex(candidates)
  candidates[index] = CandidateObservation(_record_words(state, offset, Val(CANDIDATE_WORDS)))
  offset += CANDIDATE_WORDS
 end
 return StateObservation(header, entities, candidates)
end

"""Variable candidate-scoring policy with entity-aware context and a scalar value head."""
struct AiPolicy{H<:Dense,E<:Dense,C<:Dense,S<:Chain,V<:Chain}
 header_encoder::H
 entity_encoder::E
 candidate_encoder::C
 score_head::S
 value_head::V
end

Flux.@layer AiPolicy

function create_policy(; seed::Integer=DEFAULT_RANDOM_SEED)
 rng = MersenneTwister(seed)
 init = (dimensions...) -> Flux.glorot_uniform(rng, dimensions...)
 return AiPolicy(
  Dense(STATE_HEADER_WORDS => EMBED_DIM, tanh; init),
  Dense(ENTITY_WORDS => EMBED_DIM, tanh; init),
  Dense(CANDIDATE_WORDS => EMBED_DIM, tanh; init),
  Chain(Dense(4 * EMBED_DIM => HIDDEN_DIM, tanh; init), Dense(HIDDEN_DIM => 1; init)),
  Chain(Dense(2 * EMBED_DIM => HIDDEN_DIM, tanh; init), Dense(HIDDEN_DIM => 1; init)),
 )
end

@inline _bounded(value::Real, scale::Real) = clamp(Float32(value) / Float32(scale), -4.0f0, 4.0f0)
@inline _hashed_feature(word::UInt32) = Float32(word % UInt32(65_537)) / 65_537.0f0

@inline function _header_features(header::NTuple{STATE_HEADER_WORDS,UInt32})
 return (
  _bounded(header[1], 3), _bounded(header[2], 8), _bounded(header[3], 8), _bounded(header[4], 300),
  _bounded(header[5], 20_000), _bounded(header[6], 20_000), _bounded(header[7], 200), _bounded(header[8], 200),
  _bounded(header[9], 256), _bounded(header[10], 256), _bounded(header[11], 256), _bounded(header[12], MAX_CANDIDATES),
  _bounded(header[13], 50_000), _bounded(header[14], 50_000), _bounded(header[15], 50_000), _bounded(header[16], 50_000),
  _bounded(header[17], 1_000), _bounded(header[18], 1_000), _bounded(_signed_word(header[19]), 1_000),
  _bounded(_signed_word(header[20]), 1_000), _bounded(_signed_word(header[21]), 100), _bounded(_signed_word(header[22]), 1_000),
 )
end

encode_header(header::NTuple{STATE_HEADER_WORDS,UInt32})::Vector{Float32} =
 collect(_header_features(header))

@inline function _entity_features(entity::EntityObservation)
 words = entity.words
 return (
  _bounded(words[1], MAX_ENTITY_COUNT), _hashed_feature(words[2]), _bounded(words[3], 4), _bounded(words[4], 16),
  _bounded(words[5], 256), _bounded(words[6], 256), _bounded(words[7], 1_000), _bounded(words[8], 1_000),
  _bounded(words[9], 20_000), _bounded(words[10], 20_000), _hashed_feature(words[11]), _bounded(words[12], 8),
  _bounded(words[13], 32), _bounded(words[14], 32),
 )
end

encode_entity(entity::EntityObservation)::Vector{Float32} = collect(_entity_features(entity))

@inline function _candidate_features(candidate::CandidateObservation)
 words = candidate.words
 return (
  _bounded(words[1], CANDIDATE_KIND_COUNT), _bounded(words[2], MAX_ENTITY_COUNT), _bounded(words[3], MAX_ENTITY_COUNT),
  _hashed_feature(words[4]), _bounded(words[5], 256), _bounded(words[6], 256), _bounded(words[7], 128),
  _bounded(words[8], 32), _bounded(words[9], 300), _bounded(words[10], 512), _bounded(words[11], 32),
  _bounded(_signed_word(words[12]), 1_000),
 )
end

encode_candidate(candidate::CandidateObservation)::Vector{Float32} = collect(_candidate_features(candidate))

# The workspace owns all batch storage. It is not part of Flux.@layer (or any checkpoint),
# and a session never shares it with another session or the PPO worker.
mutable struct InferenceWorkspace
 header_input::Vector{Float32}
 header_embedding::Vector{Float32}
 entity_input::Matrix{Float32}
 entity_embedding::Matrix{Float32}
 entity_context::Vector{Float32}
 global_context::Vector{Float32}
 candidate_input::Matrix{Float32}
 candidate_embedding::Matrix{Float32}
 score_input::Matrix{Float32}
 score_base::Vector{Float32}
 hidden::Matrix{Float32}
 scores::Vector{Float32}
 value_input::Vector{Float32}
 value_hidden::Vector{Float32}
end

InferenceWorkspace() = InferenceWorkspace(
 Vector{Float32}(undef, STATE_HEADER_WORDS), Vector{Float32}(undef, EMBED_DIM),
 Matrix{Float32}(undef, ENTITY_WORDS, 0), Matrix{Float32}(undef, EMBED_DIM, 0),
 zeros(Float32, EMBED_DIM), Vector{Float32}(undef, EMBED_DIM),
 Matrix{Float32}(undef, CANDIDATE_WORDS, 0), Matrix{Float32}(undef, EMBED_DIM, 0),
 Matrix{Float32}(undef, 3 * EMBED_DIM, 0), Vector{Float32}(undef, HIDDEN_DIM),
 Matrix{Float32}(undef, HIDDEN_DIM, 0),
 Float32[], Vector{Float32}(undef, 2 * EMBED_DIM), Vector{Float32}(undef, HIDDEN_DIM),
)

function _size_workspace!(workspace::InferenceWorkspace, entity_count::Int, candidate_count::Int)
 if entity_count > size(workspace.entity_input, 2)
  capacity = max(entity_count, max(16, 2 * size(workspace.entity_input, 2)))
  workspace.entity_input = Matrix{Float32}(undef, ENTITY_WORDS, capacity)
  workspace.entity_embedding = Matrix{Float32}(undef, EMBED_DIM, capacity)
 end
 if candidate_count > size(workspace.candidate_input, 2)
  capacity = max(candidate_count, max(16, 2 * size(workspace.candidate_input, 2)))
  workspace.candidate_input = Matrix{Float32}(undef, CANDIDATE_WORDS, capacity)
  workspace.candidate_embedding = Matrix{Float32}(undef, EMBED_DIM, capacity)
  workspace.score_input = Matrix{Float32}(undef, 3 * EMBED_DIM, capacity)
  workspace.hidden = Matrix{Float32}(undef, HIDDEN_DIM, capacity)
  resize!(workspace.scores, capacity)
 end
 return nothing
end

@inline function _feature_column!(matrix::Matrix{Float32}, column::Int, features::NTuple{N,Float32}) where N
 @inbounds for row in 1:N
  matrix[row, column] = features[row]
 end
 return nothing
end

function _activate_columns!(output::AbstractMatrix{Float32}, layer::Dense, columns::Int)
 activation = Flux.NNlib.fast_act(layer.σ, output)
 @inbounds for column in 1:columns, row in axes(output, 1)
  output[row, column] = activation(output[row, column] + layer.bias[row])
 end
 return nothing
end

"""Evaluate the unchanged Flux weights without allocating a Dense result for each record."""
function _inference_forward!(
 workspace::InferenceWorkspace,
 policy::AiPolicy,
 observation::StateObservation,
)
 entity_count = length(observation.entities)
 candidate_count = length(observation.candidates)
 candidate_count > 0 || throw(ArgumentError("terminal states have no selectable candidates"))
 _size_workspace!(workspace, entity_count, candidate_count)

 header_features = _header_features(observation.header)
 @inbounds for row in 1:STATE_HEADER_WORDS
  workspace.header_input[row] = header_features[row]
 end
 header_layer = policy.header_encoder
 mul!(workspace.header_embedding, header_layer.weight, workspace.header_input)
 header_activation = Flux.NNlib.fast_act(header_layer.σ, workspace.header_embedding)
 @inbounds for row in 1:EMBED_DIM
  workspace.header_embedding[row] = header_activation(workspace.header_embedding[row] + header_layer.bias[row])
 end

 fill!(workspace.entity_context, 0.0f0)
 if entity_count > 0
  for (column, entity) in enumerate(observation.entities)
   _feature_column!(workspace.entity_input, column, _entity_features(entity))
  end
  @views mul!(
   workspace.entity_embedding[:, 1:entity_count],
   policy.entity_encoder.weight,
   workspace.entity_input[:, 1:entity_count],
  )
  _activate_columns!(workspace.entity_embedding, policy.entity_encoder, entity_count)
  @inbounds for column in 1:entity_count, row in 1:EMBED_DIM
   workspace.entity_context[row] += workspace.entity_embedding[row, column]
  end
  @inbounds for row in 1:EMBED_DIM
   workspace.entity_context[row] /= Float32(entity_count)
  end
 end
 @inbounds for row in 1:EMBED_DIM
  workspace.global_context[row] = workspace.header_embedding[row] + workspace.entity_context[row]
 end

 for (column, candidate) in enumerate(observation.candidates)
  _feature_column!(workspace.candidate_input, column, _candidate_features(candidate))
 end
 @views mul!(
  workspace.candidate_embedding[:, 1:candidate_count],
  policy.candidate_encoder.weight,
  workspace.candidate_input[:, 1:candidate_count],
 )
 _activate_columns!(workspace.candidate_embedding, policy.candidate_encoder, candidate_count)

 @inbounds for (column, candidate) in enumerate(observation.candidates)
  actor = candidate_actor(candidate)
  target = candidate_target(candidate)
  for row in 1:EMBED_DIM
   workspace.score_input[row, column] = workspace.candidate_embedding[row, column]
   workspace.score_input[EMBED_DIM+row, column] =
    actor == 0 ? 0.0f0 : workspace.entity_embedding[row, actor]
   workspace.score_input[2*EMBED_DIM+row, column] =
    target == 0 ? 0.0f0 : workspace.entity_embedding[row, target]
  end
 end
 score_hidden, score_output = policy.score_head.layers
 @views mul!(
  workspace.score_base,
  score_hidden.weight[:, 1:EMBED_DIM],
  workspace.global_context,
 )
 @views mul!(
  workspace.hidden[:, 1:candidate_count],
  score_hidden.weight[:, EMBED_DIM+1:4*EMBED_DIM],
  workspace.score_input[:, 1:candidate_count],
 )
 score_activation = Flux.NNlib.fast_act(score_hidden.σ, workspace.hidden)
 @inbounds for column in 1:candidate_count, row in 1:HIDDEN_DIM
  workspace.hidden[row, column] = score_activation(
   workspace.hidden[row, column] + workspace.score_base[row] + score_hidden.bias[row],
  )
 end
 @inbounds for (column, candidate) in enumerate(observation.candidates)
  score = 0.0f0
  for row in 1:HIDDEN_DIM
   score += score_output.weight[1, row] * workspace.hidden[row, column]
  end
  workspace.scores[column] = score + score_output.bias[1] + candidate_bootstrap_score(candidate)
 end

 @inbounds for row in 1:EMBED_DIM
  workspace.value_input[row] = workspace.global_context[row]
  workspace.value_input[EMBED_DIM+row] = workspace.entity_context[row]
 end
 value_hidden, value_output = policy.value_head.layers
 mul!(workspace.value_hidden, value_hidden.weight, workspace.value_input)
 value_activation = Flux.NNlib.fast_act(value_hidden.σ, workspace.value_hidden)
 @inbounds for row in 1:HIDDEN_DIM
  workspace.value_hidden[row] = value_activation(workspace.value_hidden[row] + value_hidden.bias[row])
 end
 value = 0.0f0
 @inbounds for row in 1:HIDDEN_DIM
  value += value_output.weight[1, row] * workspace.value_hidden[row]
 end
 value += value_output.bias[1]
 return @view(workspace.scores[1:candidate_count]), value
end

function _entity_embeddings(policy::AiPolicy, observation::StateObservation)
 return map(entity -> policy.entity_encoder(encode_entity(entity)), observation.entities)
end

function _encoded_context(policy::AiPolicy, observation::StateObservation)
 header_embedding = policy.header_encoder(encode_header(observation.header))
 entity_embeddings = _entity_embeddings(policy, observation)
 return header_embedding, entity_embeddings
end

function _entity_context(entity_embeddings)
 return isempty(entity_embeddings) ? zeros(Float32, EMBED_DIM) :
        reduce(+, entity_embeddings) ./ Float32(length(entity_embeddings))
end

function _global_context(policy::AiPolicy, observation::StateObservation)
 header_embedding, entity_embeddings = _encoded_context(policy, observation)
 isempty(entity_embeddings) && return header_embedding, entity_embeddings
 return header_embedding .+ _entity_context(entity_embeddings), entity_embeddings
end

function _entity_embedding_or_zero(entity_embeddings, index::Integer)
 return index == 0 ? zeros(Float32, EMBED_DIM) : entity_embeddings[index]
end

function _candidate_score(
 policy::AiPolicy,
 candidate::CandidateObservation,
 global_context,
 entity_embeddings,
)
 candidate_embedding = policy.candidate_encoder(encode_candidate(candidate))
 actor_embedding = _entity_embedding_or_zero(entity_embeddings, candidate_actor(candidate))
 target_embedding = _entity_embedding_or_zero(entity_embeddings, candidate_target(candidate))
 return only(policy.score_head(vcat(global_context, candidate_embedding, actor_embedding, target_embedding))) +
        candidate_bootstrap_score(candidate)
end

"""Compute all actor and critic outputs from one shared state encoding."""
function _policy_forward(policy::AiPolicy, observation::StateObservation)
 isempty(observation.candidates) && throw(ArgumentError("terminal states have no selectable candidates"))
 header_embedding, entity_embeddings = _encoded_context(policy, observation)
 entity_context = _entity_context(entity_embeddings)
 global_context = header_embedding .+ entity_context
 scores = map(
  candidate -> _candidate_score(policy, candidate, global_context, entity_embeddings),
  observation.candidates,
 )
 value = only(policy.value_head(vcat(global_context, entity_context)))
 return scores, value
end

"""Score exactly the producer-supplied candidate sequence; no fixed action mask is used."""
function candidate_scores(policy::AiPolicy, observation::StateObservation)
 isempty(observation.candidates) && throw(ArgumentError("terminal states have no selectable candidates"))
 global_context, entity_embeddings = _global_context(policy, observation)
 return map(candidate -> _candidate_score(policy, candidate, global_context, entity_embeddings), observation.candidates)
end

function candidate_scores(policy::AiPolicy, state::AbstractVector{<:Integer})
 return candidate_scores(policy, parse_state(state))
end

function value_estimate(policy::AiPolicy, observation::StateObservation)
 global_context, entity_embeddings = _global_context(policy, observation)
 return only(policy.value_head(vcat(global_context, _entity_context(entity_embeddings))))
end

function value_estimate(policy::AiPolicy, state::AbstractVector{<:Integer})
 return value_estimate(policy, parse_state(state))
end

function candidate_distribution(policy::AiPolicy, observation::StateObservation)
 scores = candidate_scores(policy, observation)
 return scores, exp.(Flux.logsoftmax(scores))
end

function _candidate_log_probability_and_entropy(scores::AbstractVector{<:Real}, action::Integer)
 0 <= action < length(scores) ||
  throw(ArgumentError("candidate index $action is outside the supplied candidate sequence"))
 log_probabilities = Flux.logsoftmax(scores)
 return log_probabilities[Int(action)+1], -sum(exp.(log_probabilities) .* log_probabilities)
end

function candidate_log_probability_and_entropy(
 policy::AiPolicy,
 observation::StateObservation,
 action::Integer,
)
 return _candidate_log_probability_and_entropy(candidate_scores(policy, observation), action)
end

function _sample_from_log_probabilities(log_probabilities::AbstractVector{<:Real}, rng::AbstractRNG)::Int
 isempty(log_probabilities) && throw(ArgumentError("cannot sample an empty candidate sequence"))
 threshold = rand(rng)
 cumulative = zero(eltype(log_probabilities))
 for (index, log_probability) in enumerate(log_probabilities)
  cumulative += exp(log_probability)
  threshold <= cumulative && return index - 1
 end
 return length(log_probabilities) - 1
end

function sample_candidate(scores::AbstractVector{<:Real}, rng::AbstractRNG)::Int
 return _sample_from_log_probabilities(Flux.logsoftmax(scores), rng)
end

function _select_action(
 scores::AbstractVector{<:Real},
 log_probabilities::AbstractVector{<:Real};
 training::Bool,
 rng::AbstractRNG,
)::Int
 return training ? _sample_from_log_probabilities(log_probabilities, rng) : argmax(scores) - 1
end

function _select_action(scores::AbstractVector{<:Real}; training::Bool, rng::AbstractRNG)::Int
 return training ? sample_candidate(scores, rng) : argmax(scores) - 1
end

"""Choose greedily for inference and stochastically from supplied candidates during training."""
function select_action(
 policy::AiPolicy,
 observation::StateObservation;
 training::Bool=false,
 rng::AbstractRNG=Random.default_rng(),
)::Int
 scores = if training
  candidate_scores(policy, observation)
 else
  first(_inference_forward!(InferenceWorkspace(), policy, observation))
 end
 return _select_action(scores; training, rng)
end

function select_action(policy::AiPolicy, state::AbstractVector{<:Integer}; kwargs...)::Int
 return select_action(policy, parse_state(state); kwargs...)
end

function default_checkpoint_path()::String
 configured = strip(get(ENV, "WAR1GUS_AI_CHECKPOINT", ""))
 !isempty(configured) && return configured
 state_home = get(ENV, "XDG_STATE_HOME", "")
 root = isempty(state_home) ? joinpath(homedir(), ".local", "state") : state_home
 return joinpath(root, "war1gus", "trajectory_ppo.jls")
end

function checkpoint_payload(policy::AiPolicy, optimizer_state, update_count::Integer)
 return (
  version=CHECKPOINT_VERSION,
  protocol_version=Int(STATE_VERSION),
  header_words=STATE_HEADER_WORDS,
  entity_words=ENTITY_WORDS,
  candidate_words=CANDIDATE_WORDS,
  max_candidates=MAX_CANDIDATES,
  catalog_version=CATALOG_VERSION,
  reward_version=REWARD_VERSION,
  policy_version=POLICY_VERSION,
  ppo_version=PPO_VERSION,
  algorithm=PPO_ALGORITHM,
  update_count=Int(update_count),
  model_state=Flux.state(policy),
  optimizer_state=optimizer_state,
 )
end

function save_policy_checkpoint!(
 path::AbstractString,
 policy::AiPolicy,
 optimizer_state,
 update_count::Integer,
)::Nothing
 update_count >= 0 || throw(ArgumentError("checkpoint update count must not be negative"))
 directory = dirname(path)
 mkpath(directory)
 temporary = joinpath(directory, ".$(basename(path)).$(getpid()).$(time_ns()).tmp")
 try
  open(temporary, "w") do io
   serialize(io, checkpoint_payload(policy, optimizer_state, update_count))
   flush(io)
  end
  mv(temporary, path; force=true)
 catch
  ispath(temporary) && rm(temporary; force=true)
  rethrow()
 end
 return nothing
end

function _validate_checkpoint_payload(payload, fresh_optimizer_state)::Nothing
 payload isa NamedTuple || throw(ArgumentError("AI checkpoint has an invalid format"))
 required = (
  :version, :protocol_version, :header_words, :entity_words, :candidate_words, :max_candidates,
  :catalog_version, :reward_version, :policy_version, :ppo_version, :algorithm, :update_count,
  :model_state, :optimizer_state,
 )
 all(field -> hasproperty(payload, field), required) ||
  throw(ArgumentError("AI checkpoint is missing v4 PPO compatibility metadata"))
 payload.version == CHECKPOINT_VERSION || throw(ArgumentError("AI checkpoint version $(payload.version) is unsupported"))
 Int(payload.protocol_version) == Int(STATE_VERSION) || throw(ArgumentError("AI checkpoint protocol is incompatible"))
 Int(payload.header_words) == STATE_HEADER_WORDS || throw(ArgumentError("AI checkpoint header shape is incompatible"))
 Int(payload.entity_words) == ENTITY_WORDS || throw(ArgumentError("AI checkpoint entity shape is incompatible"))
 Int(payload.candidate_words) == CANDIDATE_WORDS || throw(ArgumentError("AI checkpoint candidate shape is incompatible"))
 Int(payload.max_candidates) == MAX_CANDIDATES || throw(ArgumentError("AI checkpoint candidate cap is incompatible"))
 Int(payload.catalog_version) == CATALOG_VERSION || throw(ArgumentError("AI checkpoint catalog version is incompatible"))
 Int(payload.reward_version) == REWARD_VERSION || throw(ArgumentError("AI checkpoint reward version is incompatible"))
 Int(payload.policy_version) == POLICY_VERSION || throw(ArgumentError("AI checkpoint policy version is incompatible"))
 Int(payload.ppo_version) == PPO_VERSION || throw(ArgumentError("AI checkpoint PPO version is incompatible"))
 payload.algorithm == PPO_ALGORITHM || throw(ArgumentError("AI checkpoint algorithm is incompatible"))
 payload.update_count isa Integer && payload.update_count >= 0 ||
  throw(ArgumentError("AI checkpoint has an invalid update count"))
 typeof(payload.optimizer_state) == typeof(fresh_optimizer_state) ||
  throw(ArgumentError("AI checkpoint optimizer state is incompatible"))
 return nothing
end

function load_policy_checkpoint!(path::AbstractString, policy::AiPolicy, fresh_optimizer_state)
 isfile(path) || return (update_count=0, optimizer_state=fresh_optimizer_state)
 payload = open(deserialize, path)
 _validate_checkpoint_payload(payload, fresh_optimizer_state)
 try
  Flux.loadmodel!(policy, payload.model_state)
 catch error
  throw(ArgumentError("AI checkpoint model parameters are incompatible: $(sprint(showerror, error))"))
 end
 return (update_count=Int(payload.update_count), optimizer_state=payload.optimizer_state)
end

function reset_policy_checkpoint!(
 path::AbstractString;
 league_path::AbstractString=joinpath(dirname(path), "league"),
)::Nothing
 ispath(path) && rm(path; force=true)
 isdir(league_path) && rm(league_path; recursive=true, force=true)
 return nothing
end

struct TrajectoryStep
 observation::StateObservation
 action::Int
 reward::Float32
 old_log_probability::Float32
 old_value::Float32
 terminal::Bool
end

struct TrajectoryFragment
 steps::Vector{TrajectoryStep}
 bootstrap_value::Float32
 terminal::Bool
end

struct Decision
 state::Vector{UInt32}
 observation::StateObservation
 action::Int
 log_probability::Float32
 value::Float32
 policy_generation::Int
 collectable::Bool
end

mutable struct ClientSession
 previous::Union{Nothing,Decision}
 fragment::Vector{TrajectoryStep}
 fragment_generation::Union{Nothing,Int}
 last_sequence::Union{Nothing,UInt32}
 player::Union{Nothing,UInt32}
 trainable::Bool
 frozen_policy::Union{Nothing,AiPolicy}
 finalized::Bool
 workspace::InferenceWorkspace
end

ClientSession() = ClientSession(
 nothing, TrajectoryStep[], nothing, nothing, nothing, true, nothing, false, InferenceWorkspace(),
)

struct PpoWorkerFailure
 error
 backtrace
end

mutable struct OnlineTrainer{O,R<:AbstractRNG}
 policy::AiPolicy
 optimizer_state::O
 lock::ReentrantLock
 checkpoint_lock::ReentrantLock
 mode::Symbol
 read_only::Bool
 gamma::Float32
 gae_lambda::Float32
 clip_epsilon::Float32
 value_coefficient::Float32
 entropy_coefficient::Float32
 update_count::Int
 policy_generation::Int
 checkpoint_path::String
 batch_size::Int
 rollout_fragment::Int
 ppo_epochs::Int
 checkpoint_every::Int
 pending_fragments::Vector{TrajectoryFragment}
 worker_active::Bool
 worker_task::Union{Nothing,Task}
 worker_failure::Union{Nothing,PpoWorkerFailure}
 last_update_loss::Union{Nothing,Float32}
 rng::R
 train_player::UInt32
 league_snapshot_every::Int
 league_max_snapshots::Int
 league_path::String
 league_snapshot_override::Union{Nothing,String}
 league_snapshots::Vector{String}
end

is_training(trainer::OnlineTrainer)::Bool =
 !trainer.read_only && trainer.mode in (MODE_TRAIN, MODE_RESET_TRAIN, MODE_LEAGUE)
is_league(trainer::OnlineTrainer)::Bool =
 trainer.mode in (MODE_LEAGUE, MODE_LEAGUE_EVALUATE)
is_league_evaluation(trainer::OnlineTrainer)::Bool =
 trainer.mode == MODE_LEAGUE_EVALUATE

function _train_player_from_environment()::UInt32
 raw = get(ENV, "WAR1GUS_AI_TRAIN_PLAYER", "0")
 parsed = tryparse(Int, raw)
 !isnothing(parsed) && 0 <= parsed <= typemax(UInt32) ||
  throw(ArgumentError("WAR1GUS_AI_TRAIN_PLAYER must be a nonnegative UInt32 player index"))
 return UInt32(parsed)
end

function read_only_from_environment()::Bool
 value = lowercase(strip(get(ENV, "WAR1GUS_AI_READ_ONLY", "")))
 value in ("", "0", "false", "no", "off") && return false
 value in ("1", "true", "yes", "on") && return true
 throw(ArgumentError("WAR1GUS_AI_READ_ONLY must be a boolean value"))
end

function league_directory_from_environment(checkpoint_path::AbstractString)::String
 configured = strip(get(ENV, "WAR1GUS_AI_LEAGUE_DIR", ""))
 return isempty(configured) ? joinpath(dirname(checkpoint_path), "league") : configured
end

function league_snapshot_from_environment()::Union{Nothing,String}
 configured = strip(get(ENV, "WAR1GUS_AI_SNAPSHOT", ""))
 return isempty(configured) ? nothing : configured
end

league_directory(trainer::OnlineTrainer) = trainer.league_path

function _refresh_league_snapshots!(trainer::OnlineTrainer)::Nothing
 directory = league_directory(trainer)
 if !isdir(directory)
  trainer.league_snapshots = String[]
  return nothing
 end
 trainer.league_snapshots = sort(
  [joinpath(directory, name) for name in readdir(directory) if startswith(name, "snapshot-") && endswith(name, ".jls") && isfile(joinpath(directory, name))],
 )
 return nothing
end

function _league_snapshot_paths(directory::AbstractString)::Vector{String}
 !isdir(directory) && return String[]
 return sort(
  [joinpath(directory, name) for name in readdir(directory) if startswith(name, "snapshot-") && endswith(name, ".jls") && isfile(joinpath(directory, name))],
 )
end

function _write_league_snapshot!(
 trainer::OnlineTrainer,
 policy::AiPolicy,
 optimizer_state,
 update_count::Int,
)::String
 is_training(trainer) || throw(ArgumentError("read-only or evaluation trainers cannot save league snapshots"))
 directory = league_directory(trainer)
 mkpath(directory)
 path = joinpath(directory, "snapshot-$(lpad(update_count, 10, '0')).jls")
 save_policy_checkpoint!(path, policy, optimizer_state, update_count)
 return path
end

function _publish_league_snapshot_locked!(
 trainer::OnlineTrainer,
 path::String,
)::Tuple{Vector{String},Int}
 snapshots = sort(unique([trainer.league_snapshots; path]))
 overflow = length(snapshots) - trainer.league_max_snapshots
 protected = trainer.league_snapshot_override
 removable = isnothing(protected) ? snapshots : filter(snapshot -> snapshot != protected, snapshots)
 stale = removable[1:min(max(overflow, 0), length(removable))]
 trainer.league_snapshots = filter(snapshot -> !(snapshot in stale), snapshots)
 return stale, length(trainer.league_snapshots)
end

function _remove_league_snapshots!(snapshots::Vector{String})::Nothing
 foreach(snapshot -> rm(snapshot; force=true), snapshots)
 return nothing
end

function _save_league_snapshot_locked!(trainer::OnlineTrainer)::String
 path = _write_league_snapshot!(trainer, trainer.policy, trainer.optimizer_state, trainer.update_count)
 stale, retained = _publish_league_snapshot_locked!(trainer, path)
 _remove_league_snapshots!(stale)
 log_event("league_snapshot"; update_count=trainer.update_count, path, retained)
 return path
end

function _load_frozen_policy(trainer::OnlineTrainer, path::AbstractString)::AiPolicy
 isfile(path) || throw(ArgumentError("frozen league snapshot does not exist: $path"))
 frozen = create_policy(seed=0xF00D)
 optimizer = Flux.OptimiserChain(Flux.ClipNorm(1.0f0), Flux.Adam(1.0f-3))
 fresh_optimizer_state = Flux.setup(optimizer, frozen)
 load_policy_checkpoint!(path, frozen, fresh_optimizer_state)
 return frozen
end

function _frozen_snapshot_path_locked!(trainer::OnlineTrainer)::String
 !isnothing(trainer.league_snapshot_override) && return trainer.league_snapshot_override::String
 if isempty(trainer.league_snapshots)
  is_training(trainer) ||
   throw(ArgumentError("league evaluation requires WAR1GUS_AI_SNAPSHOT or an existing league snapshot"))
  return _save_league_snapshot_locked!(trainer)
 end
 return rand(trainer.rng, trainer.league_snapshots)
end

function _assign_session_locked!(trainer::OnlineTrainer, session::ClientSession, player::UInt32)::Nothing
 if isnothing(session.player)
  session.player = player
  session.trainable = !is_league(trainer) || player == trainer.train_player
  if is_league(trainer) && !session.trainable
   path = _frozen_snapshot_path_locked!(trainer)
   session.frozen_policy = _load_frozen_policy(trainer, path)
   log_event("league_assignment"; player=Int(player), train_player=Int(trainer.train_player), trainable=false, snapshot=path)
  elseif is_league(trainer)
   log_event("league_assignment"; player=Int(player), train_player=Int(trainer.train_player), trainable=true, snapshot=nothing)
  end
 elseif session.player != player
  throw(ArgumentError("AI session player changed from $(session.player) to $player"))
 end
 return nothing
end

function create_trainer(
 ;
 mode::Symbol=MODE_INFERENCE,
 checkpoint_path::AbstractString=default_checkpoint_path(),
 seed::Integer=training_seed_from_environment(),
 gamma::Real=0.9995f0,
 gae_lambda::Real=0.95f0,
 clip_epsilon::Real=0.2f0,
 value_coefficient::Real=0.5f0,
 entropy_coefficient::Real=0.01f0,
 batch_size::Integer=DEFAULT_BATCH_SIZE,
 rollout_fragment::Integer=DEFAULT_ROLLOUT_FRAGMENT,
 ppo_epochs::Integer=DEFAULT_PPO_EPOCHS,
 checkpoint_every::Integer=DEFAULT_CHECKPOINT_EVERY,
 train_player::UInt32=_train_player_from_environment(),
 league_snapshot_every::Integer=DEFAULT_LEAGUE_SNAPSHOT_EVERY,
 league_max_snapshots::Integer=DEFAULT_LEAGUE_MAX_SNAPSHOTS,
 read_only::Bool=read_only_from_environment(),
 league_path::AbstractString=league_directory_from_environment(checkpoint_path),
 league_snapshot_override::Union{Nothing,AbstractString}=league_snapshot_from_environment(),
 policy::Union{Nothing,AiPolicy}=nothing,
)
 mode in (MODE_INFERENCE, MODE_TRAIN, MODE_RESET_TRAIN, MODE_LEAGUE, MODE_LEAGUE_EVALUATE) ||
  throw(ArgumentError("unknown AI mode: $mode"))
 batch_size > 0 || throw(ArgumentError("PPO batch size must be positive"))
 rollout_fragment > 0 || throw(ArgumentError("PPO rollout fragment length must be positive"))
 ppo_epochs > 0 || throw(ArgumentError("PPO epoch count must be positive"))
 checkpoint_every > 0 || throw(ArgumentError("checkpoint interval must be positive"))
 league_snapshot_every > 0 || throw(ArgumentError("league snapshot interval must be positive"))
 league_max_snapshots > 0 || throw(ArgumentError("league snapshot bound must be positive"))
 0.0 <= gamma <= 1.0 || throw(ArgumentError("GAE discount factor must be in [0, 1]"))
 0.0 <= gae_lambda <= 1.0 || throw(ArgumentError("GAE lambda must be in [0, 1]"))
 0.0 < clip_epsilon < 1.0 || throw(ArgumentError("PPO clipping epsilon must be in (0, 1)"))
 value_coefficient >= 0 || throw(ArgumentError("value coefficient must not be negative"))
 entropy_coefficient >= 0 || throw(ArgumentError("entropy coefficient must not be negative"))

 selected_policy = isnothing(policy) ? create_policy(seed=seed) : policy
 selected_path = String(checkpoint_path)
 selected_league_path = abspath(String(league_path))
 selected_override = isnothing(league_snapshot_override) ? nothing : abspath(String(league_snapshot_override))
 optimizer = Flux.OptimiserChain(Flux.ClipNorm(1.0f0), Flux.Adam(1.0f-3))
 fresh_optimizer_state = Flux.setup(optimizer, selected_policy)
 restored = if mode == MODE_RESET_TRAIN && !read_only
  reset_policy_checkpoint!(selected_path; league_path=selected_league_path)
  (update_count=0, optimizer_state=fresh_optimizer_state)
 else
  load_policy_checkpoint!(selected_path, selected_policy, fresh_optimizer_state)
 end
 trainer = OnlineTrainer(
  selected_policy, restored.optimizer_state, ReentrantLock(), ReentrantLock(), mode, read_only,
  Float32(gamma), Float32(gae_lambda), Float32(clip_epsilon), Float32(value_coefficient),
  Float32(entropy_coefficient), restored.update_count, 0, selected_path, Int(batch_size),
  Int(rollout_fragment), Int(ppo_epochs), Int(checkpoint_every), TrajectoryFragment[], false,
  nothing, nothing, nothing, MersenneTwister(seed), train_player, Int(league_snapshot_every),
  Int(league_max_snapshots), selected_league_path, selected_override, String[],
 )
 if is_league(trainer)
  !isnothing(selected_override) && !isfile(selected_override) &&
   throw(ArgumentError("WAR1GUS_AI_SNAPSHOT does not name a file: $selected_override"))
  _refresh_league_snapshots!(trainer)
  !isnothing(selected_override) && _load_frozen_policy(trainer, selected_override)
  if isempty(trainer.league_snapshots) && isnothing(selected_override)
   is_training(trainer) ? _save_league_snapshot_locked!(trainer) :
   throw(ArgumentError("league evaluation requires WAR1GUS_AI_SNAPSHOT or an existing league snapshot"))
  end
  log_event(
   "league_configuration";
   mode=String(mode),
   train_player=Int(train_player),
   directory=selected_league_path,
   snapshot_override=selected_override,
   read_only,
  )
 end
 log_event("trainer_seed"; seed=Int(seed), mode=String(mode), train_player=Int(train_player), read_only)
 return trainer
end

function _rethrow_worker_failure_locked!(trainer::OnlineTrainer)::Nothing
 failure = trainer.worker_failure
 isnothing(failure) && return nothing
 throw((failure::PpoWorkerFailure).error)
end

function save_checkpoint!(trainer::OnlineTrainer)::Nothing
 is_training(trainer) || return nothing
 lock(trainer.checkpoint_lock) do
  policy, optimizer_state, update_count = lock(trainer.lock) do
   _rethrow_worker_failure_locked!(trainer)
   snapshot_policy, snapshot_optimizer = deepcopy((trainer.policy, trainer.optimizer_state))
   return snapshot_policy, snapshot_optimizer, trainer.update_count
  end
  save_policy_checkpoint!(trainer.checkpoint_path, policy, optimizer_state, update_count)
 end
 return nothing
end

function _fragment_gae(trainer::OnlineTrainer, fragment::TrajectoryFragment)
 count = length(fragment.steps)
 advantages = Vector{Float32}(undef, count)
 targets = Vector{Float32}(undef, count)
 advantage = 0.0f0
 for index in count:-1:1
  step = fragment.steps[index]
  next_value = index == count ? fragment.bootstrap_value : fragment.steps[index+1].old_value
  continuation = step.terminal ? 0.0f0 : 1.0f0
  delta = step.reward + trainer.gamma * continuation * next_value - step.old_value
  advantage = delta + trainer.gamma * trainer.gae_lambda * continuation * advantage
  advantages[index] = advantage
  targets[index] = step.old_value + advantage
 end
 return targets, advantages
end

"""Compute GAE targets for complete terminal or bootstrapped trajectory fragments."""
function trajectory_targets_and_advantages(trainer::OnlineTrainer, fragments::Vector{TrajectoryFragment})
 steps = TrajectoryStep[]
 targets = Float32[]
 advantages = Float32[]
 for fragment in fragments
  isempty(fragment.steps) && continue
  fragment_targets, fragment_advantages = _fragment_gae(trainer, fragment)
  append!(steps, fragment.steps)
  append!(targets, fragment_targets)
  append!(advantages, fragment_advantages)
 end
 isempty(steps) && return steps, targets, advantages
 scale = std(advantages; corrected=false)
 normalized = scale > eps(Float32) ? (advantages .- mean(advantages)) ./ scale : advantages .- mean(advantages)
 return steps, targets, Float32.(normalized)
end

include("training.jl")

function _pending_step_count(fragments::Vector{TrajectoryFragment})::Int
 return sum(length(fragment.steps) for fragment in fragments)
end

_pending_step_count(trainer::OnlineTrainer)::Int = _pending_step_count(trainer.pending_fragments)

function _train_ppo_snapshot!(
 trainer::OnlineTrainer,
 policy::AiPolicy,
 optimizer_state,
 fragments::Vector{TrajectoryFragment},
)::Tuple{Float32,Int}
 steps, targets, advantages = trajectory_targets_and_advantages(trainer, fragments)
 isempty(steps) && throw(ArgumentError("PPO update has no trajectory steps"))
 loss = 0.0f0
 batch = _pack_ppo_batch(steps, targets, advantages)
 for _ in 1:trainer.ppo_epochs
  result = Flux.withgradient(policy) do trained_policy
   _ppo_loss(
    trained_policy, batch, trainer.clip_epsilon,
    trainer.value_coefficient, trainer.entropy_coefficient,
   )
  end
  loss = Float32(result.val)
  isfinite(loss) || throw(ArgumentError("PPO update produced a non-finite loss"))
  Flux.update!(optimizer_state, policy, result.grad[1])
 end
 return loss, length(steps)
end

function _record_worker_failure!(trainer::OnlineTrainer, error, backtrace)::Nothing
 lock(trainer.lock) do
  trainer.worker_active = false
  trainer.worker_failure = PpoWorkerFailure(error, backtrace)
 end
 log_event("ppo_update_failed"; error=sprint(showerror, error))
 return nothing
end

function _run_ppo_update_worker!(
 trainer::OnlineTrainer,
 policy::AiPolicy,
 optimizer_state,
 fragments::Vector{TrajectoryFragment},
 dispatched_generation::Int,
)::Nothing
 started_at = time_ns()
 try
  loss, step_count = _train_ppo_snapshot!(trainer, policy, optimizer_state, fragments)
  update_count, checkpoint_due, league_snapshot_due = lock(trainer.lock) do
   trainer.policy_generation == dispatched_generation ||
    throw(ArgumentError("PPO worker published against an unexpected policy generation"))
   trainer.policy = policy
   trainer.optimizer_state = optimizer_state
   trainer.update_count += 1
   trainer.policy_generation += 1
   trainer.last_update_loss = loss
   return (
    trainer.update_count,
    trainer.update_count % trainer.checkpoint_every == 0,
    is_league(trainer) && trainer.update_count % trainer.league_snapshot_every == 0,
   )
  end

  if checkpoint_due
   lock(trainer.checkpoint_lock) do
    save_policy_checkpoint!(trainer.checkpoint_path, policy, optimizer_state, update_count)
   end
  end
  snapshot_path = nothing
  if league_snapshot_due
   snapshot_path = _write_league_snapshot!(trainer, policy, optimizer_state, update_count)
   stale, retained = lock(trainer.lock) do
    return _publish_league_snapshot_locked!(trainer, snapshot_path)
   end
   _remove_league_snapshots!(stale)
   log_event("league_snapshot"; update_count, path=snapshot_path, retained)
  end

  duration_ms = (time_ns() - started_at) / 1_000_000
  lock(trainer.lock) do
   trainer.worker_active = false
  end

  log_event(
   "ppo_update";
   update_count,
   policy_generation=dispatched_generation + 1,
   fragments=length(fragments),
   steps=step_count,
   epochs=trainer.ppo_epochs,
   loss,
   duration_ms,
   checkpoint_due,
   league_snapshot=snapshot_path,
   gamma=trainer.gamma,
   gae_lambda=trainer.gae_lambda,
   clip_epsilon=trainer.clip_epsilon,
  )
 catch error
  _record_worker_failure!(trainer, error, catch_backtrace())
 end
 return nothing
end

function _worker_busy_locked(trainer::OnlineTrainer)::Bool
 task = trainer.worker_task
 return trainer.worker_active || (!isnothing(task) && !istaskdone(task))
end

function _schedule_pending_update_locked!(trainer::OnlineTrainer; force::Bool=false)::Bool
 is_training(trainer) || return false
 _rethrow_worker_failure_locked!(trainer)
 _worker_busy_locked(trainer) && return false
 isempty(trainer.pending_fragments) && return false
 !force && _pending_step_count(trainer) < trainer.batch_size && return false

 fragments = trainer.pending_fragments
 trainer.pending_fragments = TrajectoryFragment[]
 policy, optimizer_state = deepcopy((trainer.policy, trainer.optimizer_state))
 dispatched_generation = trainer.policy_generation
 trainer.worker_active = true
 log_event(
  "ppo_update_scheduled";
  update_count=trainer.update_count + 1,
  policy_generation=dispatched_generation,
  fragments=length(fragments),
  steps=_pending_step_count(fragments),
  forced=force,
 )
 trainer.worker_task = Threads.@spawn begin
  _run_ppo_update_worker!(trainer, policy, optimizer_state, fragments, dispatched_generation)
 end
 return true
end

function _discard_stale_fragment_locked!(trainer::OnlineTrainer, session::ClientSession)::Nothing
 if !isnothing(session.fragment_generation) &&
    session.fragment_generation != trainer.policy_generation
  empty!(session.fragment)
  session.fragment_generation = nothing
 end
 return nothing
end

function _submit_fragment_locked!(
 trainer::OnlineTrainer,
 session::ClientSession,
 bootstrap_value::Float32,
 terminal::Bool,
)::Nothing
 _discard_stale_fragment_locked!(trainer, session)
 isempty(session.fragment) && return nothing
 session.fragment_generation == trainer.policy_generation ||
  throw(ArgumentError("trajectory fragment crossed a PPO policy generation"))
 push!(trainer.pending_fragments, TrajectoryFragment(copy(session.fragment), bootstrap_value, terminal))
 empty!(session.fragment)
 session.fragment_generation = nothing
 _schedule_pending_update_locked!(trainer)
 return nothing
end

function _log_training_step(
 decision::Decision,
 reward::Int32,
 sequence::UInt32,
 next_state::Union{Nothing,Vector{UInt32}},
 terminal::Bool,
)::Nothing
 verbose_logging_enabled() || return nothing
 log_event(
  "training_sample";
  session_id=Int(player_of(decision.state)),
  sequence,
  state=decision.state,
  action=decision.action,
  candidate_kind=candidate_name(candidate_kind(decision.observation.candidates[decision.action+1])),
  reward,
  next_state,
  terminal,
 )
 return nothing
end

function _log_reward_decomposition(
 state::AbstractVector{<:Integer},
 sequence::UInt32,
 reward::Int32,
 terminal::Bool,
)::Nothing
 verbose_logging_enabled() || return nothing
 components = reward_components(state)
 log_event(
  "reward_decomposition";
  session_id=Int(player_of(state)),
  sequence,
  reward,
  enemy_progress=components.enemy_progress,
  own_loss=components.own_loss,
  time=components.time,
  terminal_component=components.terminal_component,
  terminal,
 )
 return nothing
end

function _training_session(trainer::OnlineTrainer, session::ClientSession)::Bool
 return is_training(trainer) && session.trainable
end

"""Credit the prior decision, append a trajectory step, then select the current candidate."""
function process_step!(
 trainer::OnlineTrainer,
 session::ClientSession,
 sequence::UInt32,
 reward::Int32,
 state::Vector{UInt32};
 validated_state::Bool=false,
)::Int
 observation = parse_state(state; validated=validated_state)
 lock(trainer.lock) do
  _rethrow_worker_failure_locked!(trainer)
  session.finalized && throw(ArgumentError("AI session for player $(session.player) is already finalized"))
  _assign_session_locked!(trainer, session, player_of(state))
  if !isnothing(session.last_sequence)
   if sequence == session.last_sequence
    return (session.previous::Decision).action
   end
   sequence == session.last_sequence + one(UInt32) ||
    throw(ArgumentError("out-of-order AI step sequence $sequence"))
  end

  previous = session.previous
  training = _training_session(trainer, session)
  active_policy = session.trainable ? trainer.policy : (session.frozen_policy::AiPolicy)
  scores, value = _inference_forward!(session.workspace, active_policy, observation)
  if !isnothing(previous)
   if training && previous.collectable &&
      previous.policy_generation == trainer.policy_generation &&
      !_worker_busy_locked(trainer)
    _discard_stale_fragment_locked!(trainer, session)
    push!(
     session.fragment,
     TrajectoryStep(
      previous.observation, previous.action, Float32(reward),
      previous.log_probability, previous.value, false,
     ),
    )
    session.fragment_generation = trainer.policy_generation
    if length(session.fragment) >= trainer.rollout_fragment
     _submit_fragment_locked!(trainer, session, Float32(value), false)
    end
    _log_training_step(previous, reward, sequence, state, false)
   elseif training
    _discard_stale_fragment_locked!(trainer, session)
   end
   _log_reward_decomposition(state, sequence, reward, false)
  end

  log_probabilities = Flux.logsoftmax(scores)
  action = _select_action(scores, log_probabilities; training, rng=trainer.rng)
  log_probability = log_probabilities[action+1]
  collectable = training && !_worker_busy_locked(trainer)
  session.previous = Decision(
   state, observation, action, Float32(log_probability), Float32(value),
   trainer.policy_generation, collectable,
  )
  session.last_sequence = sequence
  return action
 end
end

"""Attach terminal reward to the last live candidate, submit its fragment, and finalize once."""
function process_terminal!(
 trainer::OnlineTrainer,
 session::ClientSession,
 sequence::UInt32,
 reward::Int32,
 state::Vector{UInt32};
 validated_state::Bool=false,
)::Bool
 parse_state(state; terminal=true, validated=validated_state)
 lock(trainer.lock) do
  _rethrow_worker_failure_locked!(trainer)
  session.finalized && return false
  _assign_session_locked!(trainer, session, player_of(state))
  if !isnothing(session.last_sequence)
   sequence == session.last_sequence + one(UInt32) ||
    throw(ArgumentError("out-of-order AI terminal sequence $sequence"))
  end
  previous = session.previous
  training = _training_session(trainer, session)
  if !isnothing(previous) && training && previous.collectable &&
     previous.policy_generation == trainer.policy_generation &&
     !_worker_busy_locked(trainer)
   _discard_stale_fragment_locked!(trainer, session)
   push!(
    session.fragment,
    TrajectoryStep(
     previous.observation, previous.action, Float32(reward),
     previous.log_probability, previous.value, true,
    ),
   )
   session.fragment_generation = trainer.policy_generation
   _submit_fragment_locked!(trainer, session, 0.0f0, true)
   _log_training_step(previous, reward, sequence, nothing, true)
  elseif training
   empty!(session.fragment)
   session.fragment_generation = nothing
  end
  _log_reward_decomposition(state, sequence, reward, true)
  session.previous = nothing
  session.last_sequence = nothing
  session.finalized = true
  log_event("episode_finalized"; player=Int(player_of(state)), sequence, reward, trained=training)
  return true
 end
end

"""Force a partial update when needed, then wait until the PPO worker is fully idle."""
function flush_trajectories!(trainer::OnlineTrainer)::Union{Nothing,Float32}
 is_training(trainer) || return nothing
 loss = nothing
 while true
  task = lock(trainer.lock) do
   if _worker_busy_locked(trainer)
    return trainer.worker_task
   end
   _rethrow_worker_failure_locked!(trainer)
   if _schedule_pending_update_locked!(trainer; force=true)
    return trainer.worker_task
   end
   return nothing
  end
  isnothing(task) && return loss
  wait(task)
  loss = lock(trainer.lock) do
   _rethrow_worker_failure_locked!(trainer)
   return trainer.last_update_loss
  end
 end
end

function _warmup_state(; terminal::Bool=false)::Vector{UInt32}
 header = UInt32[
  STATE_VERSION, 0, 0, 300, 500, 500, 20, 4, 128, 128, 1, terminal ? 0 : 2,
  1_000, 1_000, 500, 500, 0, 0, 0, 0, 0, 0,
 ]
 entity = UInt32[1, 0x1234, 1, 1, 20, 20, 60, 60, 400, 0, 0, 0, 1, 4]
 wait = UInt32[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 gather = UInt32[1, 1, 0, 0, 20, 20, 1, 0, 5, 10, 1, 100]
 return terminal ? vcat(header, entity) : vcat(header, entity, wait, gather)
end

"""Compile the trajectory PPO gradient path on a disposable trainer before accepting training clients."""
function warmup_training_runtime!()::Nothing
 trainer = create_trainer(
  mode=MODE_TRAIN,
  checkpoint_path=tempname(),
  batch_size=2,
  rollout_fragment=2,
  ppo_epochs=1,
  checkpoint_every=typemax(Int),
 )
 state = _warmup_state()
 session = ClientSession()
 process_step!(trainer, session, UInt32(0), Int32(0), state)
 process_step!(trainer, session, UInt32(1), Int32(1), state)
 process_terminal!(trainer, session, UInt32(2), Int32(1), _warmup_state(terminal=true))
 flush_trajectories!(trainer)
 return nothing
end

const DEFAULT_POLICY = create_policy(seed=DEFAULT_RANDOM_SEED)
select_action(state::AbstractVector{<:Integer}) = select_action(DEFAULT_POLICY, state)
