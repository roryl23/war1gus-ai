using Flux
using Random
using Serialization
using Statistics

model_state_snapshot(policy) = deepcopy(Flux.state(policy))

# The pre-packing PPO objective, kept here as an independent scalar oracle.
function scalar_ppo_reference(policy, steps, targets, advantages, clip_epsilon, value_coefficient, entropy_coefficient)
  total_loss = 0.0f0
  for index in eachindex(steps)
    step = steps[index]
    scores, value = War1gusAI._policy_forward(policy, step.observation)
    log_probabilities = War1gusAI.training_action_log_probabilities(scores, step.observation.candidates)
    log_probability = log_probabilities[step.action+1]
    entropy = -sum(exp.(log_probabilities) .* log_probabilities)
    ratio = exp(log_probability - step.old_log_probability)
    actor_loss = -min(
      ratio * advantages[index],
      clamp(ratio, 1.0f0 - clip_epsilon, 1.0f0 + clip_epsilon) * advantages[index],
    )
    unclipped_value_loss = (value - targets[index])^2
    clipped_value = step.old_value + clamp(value - step.old_value, -clip_epsilon, clip_epsilon)
    clipped_value_loss = (clipped_value - targets[index])^2
    total_loss += actor_loss +
                  value_coefficient * 0.5f0 * max(unclipped_value_loss, clipped_value_loss) -
                  entropy_coefficient * entropy
  end
  return total_loss / Float32(length(steps))
end

function policy_gradient_arrays(gradient)
  return (
    gradient.header_encoder.weight, gradient.header_encoder.bias,
    gradient.entity_encoder.weight, gradient.entity_encoder.bias,
    gradient.candidate_encoder.weight, gradient.candidate_encoder.bias,
    gradient.score_head.layers[1].weight, gradient.score_head.layers[1].bias,
    gradient.score_head.layers[2].weight, gradient.score_head.layers[2].bias,
    gradient.value_head.layers[1].weight, gradient.value_head.layers[1].bias,
    gradient.value_head.layers[2].weight, gradient.value_head.layers[2].bias,
  )
end

@testset "packed PPO matches scalar objective and every trainable gradient" begin
  policy = War1gusAI.create_policy(seed=43)
  observations = [
    War1gusAI.parse_state(v3_state(
      cycle=101,
      entities=[v3_entity(slot=1, hp=20), v3_entity(slot=2, role=3), v3_entity(slot=3, relation=2)],
      candidates=[
        v3_candidate(),
        v3_candidate(kind=6, actor=3, target=1, bootstrap=400),
        v3_candidate(kind=7, actor=0, target=2, x=40, group_size=2),
        v3_candidate(kind=6, actor=1, target=3, bootstrap=-300, group_size=1),
        v3_candidate(kind=8, actor=3, target=3, bootstrap=200),
      ],
    )),
    War1gusAI.parse_state(v3_state(
      cycle=202, entities=[v3_entity(slot=9, role=2, hp=37)],
      candidates=[
        v3_candidate(),
        v3_candidate(kind=8, actor=1, target=1, bootstrap=-450),
        v3_candidate(kind=7, actor=0, target=1, x=85),
      ],
    )),
    War1gusAI.parse_state(v3_state(
      cycle=303, entities=Vector{UInt32}[],
      candidates=[
        v3_candidate(),
        v3_candidate(kind=7, x=14),
        v3_candidate(kind=0, bootstrap=12_000),
        v3_candidate(kind=7, x=90, bootstrap=-100),
      ],
    )),
    War1gusAI.parse_state(v3_state(
      cycle=404, entities=[v3_entity(slot=4, hp=53), v3_entity(slot=5, relation=3, role=2)],
      candidates=[
        v3_candidate(),
        v3_candidate(kind=6, actor=2, target=1),
        v3_candidate(kind=6, actor=1, target=2, bootstrap=550),
      ],
    )),
  ]
  actions = [3, 2, 2, 1] # Zero-based catalog indices, including the concentrated logit.
  ratios = Float32[1.6, 0.5, 1.05, 0.95]
  advantages = Float32[1.0, -0.8, 0.6, -0.7]
  targets = Float32[]
  steps = War1gusAI.TrajectoryStep[]
  for index in eachindex(observations)
    scores, value = War1gusAI._policy_forward(policy, observations[index])
    log_probability = War1gusAI.training_action_log_probabilities(scores, observations[index].candidates)[actions[index]+1]
    old_value = value - 1.0f0
    # Both max branches: target above the current value selects the clipped
    # critic; target at the old value selects the unclipped critic.
    push!(targets, index in (1, 3) ? value + 1.0f0 : old_value)
    push!(steps, War1gusAI.TrajectoryStep(
      observations[index], actions[index], 0.0f0,
      log_probability - log(ratios[index]), old_value, false,
    ))
  end
  clip_epsilon, value_coefficient, entropy_coefficient = 0.2f0, 0.5f0, 0.01f0
  batch = War1gusAI._pack_ppo_batch(steps, targets, advantages)
  scalar = Flux.withgradient(policy) do trained_policy
    scalar_ppo_reference(
      trained_policy, steps, targets, advantages, clip_epsilon,
      value_coefficient, entropy_coefficient,
    )
  end
  packed = Flux.withgradient(policy) do trained_policy
    War1gusAI._ppo_loss(
      trained_policy, batch, clip_epsilon, value_coefficient, entropy_coefficient,
    )
  end
  @test packed.val ≈ scalar.val atol = 2f-5 rtol = 2f-5
  for (actual, expected) in zip(policy_gradient_arrays(packed.grad[1]), policy_gradient_arrays(scalar.grad[1]))
    # Matmul over packed columns changes Float32 reduction order from the
    # scalar path, particularly through the segmented entity mean.
    @test actual ≈ expected atol = 3f-5 rtol = 3f-4
  end
end

@testset "opening hints bias exploration without removing other choices" begin
  candidates = [
    v3_candidate(),
    v3_candidate(kind=14, group_size=1),
    v3_candidate(kind=14, group_size=2),
    v3_candidate(kind=14),
  ]
  observation = War1gusAI.parse_state(v3_state(candidates=candidates))
  scores = zeros(Float32, length(candidates))
  log_probabilities = War1gusAI.training_action_log_probabilities(scores, observation.candidates)
  probabilities = exp.(log_probabilities)
  @test all(>(0), probabilities)
  @test sum(probabilities) ≈ 1.0f0 atol = 1f-6
  @test probabilities ≈ Float32[0.05, 0.05+0.8*8/41, 0.05+0.8*32/41, 0.05+0.8/41] atol = 1f-6
  @test probabilities[3] > probabilities[2] > probabilities[4] > probabilities[1]

  plain = War1gusAI.parse_state(v3_state(candidates=[
    v3_candidate(), v3_candidate(kind=14), v3_candidate(kind=14), v3_candidate(kind=14),
  ]))
  plain_probabilities = exp.(War1gusAI.training_action_log_probabilities(scores, plain.candidates))
  @test plain_probabilities ≈ Float32[0.05, 0.05+0.8/3, 0.05+0.8/3, 0.05+0.8/3] atol = 1f-6
  @test probabilities[3] > plain_probabilities[3]

  step = War1gusAI.TrajectoryStep(observation, 2, 0.0f0, log_probabilities[3], 0.0f0, false)
  batch = War1gusAI._pack_ppo_batch([step], Float32[0], Float32[1])
  packed_statistics = War1gusAI._segmented_policy_statistics(
    scores, batch.candidate_offsets, batch.actions, batch.first_waits,
    batch.exploration_weights, batch.exploration_weight_sums,
  )
  @test packed_statistics[1, 1] ≈ log_probabilities[3] atol = 1f-6
  @test packed_statistics[2, 1] ≈ -sum(probabilities .* log_probabilities) atol = 1f-6
end

@testset "entity-free PPO does not advance Adam's entity momentum" begin
  policy = War1gusAI.create_policy(seed=47)
  optimizer_state = Flux.setup(Flux.Adam(1.0f-3), policy)
  with_entity = War1gusAI.parse_state(v3_state(
    entities=[v3_entity(slot=1, hp=35)],
    candidates=[v3_candidate(), v3_candidate(kind=7, actor=1, x=51)],
  ))
  without_entity = War1gusAI.parse_state(v3_state(
    entities=Vector{UInt32}[],
    candidates=[v3_candidate(), v3_candidate(kind=7, x=51)],
  ))
  for observation in (with_entity, without_entity)
    scores, value = War1gusAI._policy_forward(policy, observation)
    step = War1gusAI.TrajectoryStep(
      observation, 1, 0.0f0, War1gusAI.training_action_log_probabilities(scores, observation.candidates)[2], value, false,
    )
    batch = War1gusAI._pack_ppo_batch([step], Float32[value+1], Float32[0.7])
    result = Flux.withgradient(policy) do trained_policy
      War1gusAI._ppo_loss(trained_policy, batch, 0.2f0, 0.5f0, 0.01f0)
    end
    if observation === without_entity
      @test isnothing(result.grad[1].entity_encoder)
      before = deepcopy(Flux.state(policy).entity_encoder)
      Flux.update!(optimizer_state, policy, result.grad[1])
      @test Flux.state(policy).entity_encoder == before
    else
      Flux.update!(optimizer_state, policy, result.grad[1])
    end
  end
end

@testset "variable entity and candidate policy" begin
  entities = Vector{UInt32}[v3_entity(slot=1, relation=1, role=1), v3_entity(slot=2, relation=2, role=3, hp=45)]
  candidates = Vector{UInt32}[
    v3_candidate(),
    v3_candidate(kind=6, actor=1, target=2, distance=10, bootstrap=50),
    v3_candidate(kind=10, actor=1, group_size=1, formation=2, bootstrap=100),
  ]
  state = v3_state(entities=entities, candidates=candidates)
  observation = War1gusAI.parse_state(state)
  @test length(observation.entities) == 2
  @test length(observation.candidates) == 3
  @test [War1gusAI.candidate_name(kind) for kind in 12:18] == [
    "cast-spell", "actor-choice", "action-choice", "entity-choice",
    "x-choice", "y-choice", "page-choice",
  ]
  @test_throws ArgumentError War1gusAI.candidate_name(19)
  far_edge = War1gusAI.parse_state(v3_state(candidates=[
    v3_candidate(), v3_candidate(kind=16, x=1022, y=1022),
    v3_candidate(kind=16, x=1023, y=1023),
  ]))
  near_edge_features = War1gusAI.encode_candidate(far_edge.candidates[2])
  far_edge_features = War1gusAI.encode_candidate(far_edge.candidates[3])
  @test near_edge_features[5] < far_edge_features[5] < 4
  @test near_edge_features[6] < far_edge_features[6] < 4

  policy = War1gusAI.create_policy(seed=17)
  scores = War1gusAI.candidate_scores(policy, observation)
  @test length(scores) == 3
  @test all(isfinite, scores)
  @test isfinite(War1gusAI.value_estimate(policy, observation))
  peaked = War1gusAI.parse_state(v3_state(candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1, bootstrap=1_000_000)]))
  peaked_log_probability, peaked_entropy = War1gusAI.candidate_log_probability_and_entropy(policy, peaked, 1)
  @test isfinite(peaked_log_probability)
  @test isfinite(peaked_entropy)
  @test War1gusAI.select_action(policy, observation) in 0:2
  @test War1gusAI.select_action(policy, observation; training=false, rng=MersenneTwister(1)) ==
        War1gusAI.select_action(policy, observation; training=false, rng=MersenneTwister(2))
  @test all(
    War1gusAI.select_action(policy, observation; training=true, rng=MersenneTwister(seed)) in 0:2
    for seed in 1:32
  )

  bad_wait = copy(state)
  bad_wait[War1gusAI.STATE_HEADER_WORDS+War1gusAI.ENTITY_WORDS*2+1] = 7
  @test_throws ArgumentError War1gusAI.parse_state(bad_wait)
  bad_actor = copy(state)
  bad_actor[War1gusAI.STATE_HEADER_WORDS+War1gusAI.ENTITY_WORDS*2+War1gusAI.CANDIDATE_WORDS+2] = 3
  @test_throws ArgumentError War1gusAI.parse_state(bad_actor)
  @test_throws ArgumentError War1gusAI.parse_state(v3_state(candidates=Vector{UInt32}[]))
end

@testset "reused inference storage preserves catalogs and current weights" begin
  policy = War1gusAI.create_policy(seed=29)
  workspace = War1gusAI.InferenceWorkspace()
  entities = [v3_entity(slot=index, relation=index % 3 + 1, hp=index * 7) for index in 1:48]
  candidates = Vector{UInt32}[v3_candidate()]
  append!(candidates, [
    v3_candidate(
      kind=index % 18 + 1, actor=index % 49, target=(index * 7) % 49,
      x=index % 128, bootstrap=index - 256,
    ) for index in 1:511
  ])
  large = War1gusAI.parse_state(v3_state(; entities, candidates))
  small = War1gusAI.parse_state(v3_state(candidates=[
    v3_candidate(), v3_candidate(kind=6, actor=1, target=1, bootstrap=100),
  ]))
  empty = War1gusAI.parse_state(v3_state(
    entities=Vector{UInt32}[], candidates=[v3_candidate(), v3_candidate()],
  ))
  # Grow, shrink, clear entity references, then grow again without stale columns.
  for observation in (large, small, empty, large)
    expected_scores, expected_value = War1gusAI._policy_forward(policy, observation)
    scores, value = War1gusAI._inference_forward!(workspace, policy, observation)
    @test scores ≈ expected_scores atol = 2f-5 rtol = 2f-5
    @test value ≈ expected_value atol = 2f-5 rtol = 2f-5
    @test argmax(scores) == argmax(expected_scores)
  end
  @test War1gusAI.select_action(policy, empty) == 0
  tied = War1gusAI.parse_state(v3_state(
    entities=Vector{UInt32}[], candidates=[v3_candidate() for _ in 1:511],
  ))
  @test War1gusAI.select_action(policy, tied) == 0

  # PPO publication and frozen opponents must use their current weights.
  replacement = War1gusAI.create_policy(seed=41)
  for active_policy in (replacement, policy)
    expected_scores, expected_value = War1gusAI._policy_forward(active_policy, small)
    scores, value = War1gusAI._inference_forward!(workspace, active_policy, small)
    @test scores ≈ expected_scores atol = 2f-5 rtol = 2f-5
    @test value ≈ expected_value atol = 2f-5 rtol = 2f-5
  end
end

@testset "GAE targets and normalized trajectory PPO" begin
  state = v3_state(candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
  observation = War1gusAI.parse_state(state)
  trainer = War1gusAI.create_trainer(
    mode=War1gusAI.MODE_TRAIN,
    checkpoint_path=tempname(),
    seed=19,
    gamma=0.5f0,
    gae_lambda=0.5f0,
    batch_size=8,
    ppo_epochs=1,
  )
  fragment = War1gusAI.TrajectoryFragment(
    War1gusAI.TrajectoryStep[
      War1gusAI.TrajectoryStep(observation, 0, 1.0f0, 0.0f0, 0.1f0, false),
      War1gusAI.TrajectoryStep(observation, 1, 2.0f0, 0.0f0, 0.2f0, true),
    ],
    0.0f0,
    true,
  )
  targets, advantages = War1gusAI._fragment_gae(trainer, fragment)
  @test targets ≈ Float32[1.55, 2.0]
  @test advantages ≈ Float32[1.45, 1.8]
  steps, normalized_targets, normalized_advantages =
    War1gusAI.trajectory_targets_and_advantages(trainer, War1gusAI.TrajectoryFragment[fragment])
  @test length(steps) == 2
  @test normalized_targets == targets
  @test isapprox(mean(normalized_advantages), 0.0f0; atol=1f-6)
  @test all(isfinite, normalized_advantages)

  mktempdir() do directory
    checkpoint = joinpath(directory, "ppo.jls")
    learner = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=checkpoint,
      seed=23,
      batch_size=6,
      rollout_fragment=16,
      ppo_epochs=2,
      checkpoint_every=1,
    )
    stages = [
      v3_state(player=4, cycle=100, candidates=[v3_candidate(), v3_candidate(kind=kind, actor=1, target=1, x=30, y=60)])
      for kind in 13:18
    ]
    terminal = v3_state(player=4, cycle=100, candidates=Vector{UInt32}[], terminal_reward=1_000)
    before = model_state_snapshot(learner.policy)

    session = War1gusAI.ClientSession()
    for (index, stage) in enumerate(stages)
      War1gusAI.process_step!(learner, session, UInt32(index - 1), Int32(0), stage)
    end
    @test length(session.fragment) == 5
    @test all(step -> step.reward == 0.0f0, session.fragment)
    @test War1gusAI.process_terminal!(learner, session, UInt32(6), Int32(1_000), terminal)
    task = learner.worker_task
    @test !isnothing(task)
    War1gusAI.flush_trajectories!(learner)
    @test istaskdone(task::Task)
    @test learner.update_count == 1
    @test learner.policy_generation == 1
    @test isempty(learner.pending_fragments)
    @test Flux.state(learner.policy) != before
    payload = open(deserialize, checkpoint)
    @test payload.update_count == 1
    @test payload.model_state == Flux.state(learner.policy)
    restored = War1gusAI.create_trainer(checkpoint_path=checkpoint, seed=99)
    @test restored.update_count == learner.update_count
    @test Flux.state(restored.policy) == Flux.state(learner.policy)
    legacy = joinpath(directory, "legacy.jls")
    open(legacy, "w") do io
      serialize(io, merge(payload, (catalog_version=4, reward_version=3)))
    end
    migrated = War1gusAI.create_trainer(checkpoint_path=legacy, seed=99)
    @test migrated.update_count == learner.update_count
    @test Flux.state(migrated.policy) == payload.model_state
    @test typeof(migrated.optimizer_state) == typeof(learner.optimizer_state)
    War1gusAI.save_policy_checkpoint!(legacy, migrated.policy, migrated.optimizer_state, migrated.update_count)
    migrated_payload = open(deserialize, legacy)
    @test migrated_payload.model_state == payload.model_state
    incompatible = joinpath(directory, "incompatible.jls")
    open(incompatible, "w") do io
      serialize(io, merge(payload, (catalog_version=3,)))
    end
    failure = try
      War1gusAI.create_trainer(checkpoint_path=incompatible)
      nothing
    catch error
      error
    end
    @test failure isa ArgumentError
    War1gusAI.flush_trajectories!(learner)
    @test learner.update_count == 1
  end
end

@testset "terminal finalization only submits trajectory work" begin
  mktempdir() do directory
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=joinpath(directory, "terminal.jls"),
      seed=31,
      batch_size=32,
      rollout_fragment=32,
      ppo_epochs=1,
      checkpoint_every=100,
    )
    session = War1gusAI.ClientSession()
    live = v3_state(player=9, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
    terminal = v3_state(player=9, candidates=Vector{UInt32}[], terminal_reward=-1_000)
    War1gusAI.process_step!(trainer, session, UInt32(0), Int32(0), live)
    @test War1gusAI.process_terminal!(trainer, session, UInt32(1), Int32(-1_000), terminal)
    @test trainer.update_count == 0
    @test length(trainer.pending_fragments) == 1
    @test isnothing(trainer.worker_task)
    @test !trainer.worker_active
    @test !War1gusAI.process_terminal!(trainer, session, UInt32(1), Int32(-1_000), terminal)
    @test_throws ArgumentError War1gusAI.process_step!(trainer, session, UInt32(2), Int32(0), live)
    @test !isnothing(War1gusAI.flush_trajectories!(trainer))
    @test trainer.update_count == 1
  end
end

@testset "single-flight PPO preserves request servicing and generations" begin
  mktempdir() do directory
    checkpoint = joinpath(directory, "single-flight.jls")
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=checkpoint,
      seed=33,
      batch_size=1,
      rollout_fragment=32,
      ppo_epochs=1,
      checkpoint_every=1,
    )
    training = War1gusAI.ClientSession()
    partial = War1gusAI.ClientSession()
    during_update = War1gusAI.ClientSession()
    live = v3_state(player=13, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
    terminal = v3_state(player=13, candidates=Vector{UInt32}[], terminal_reward=99)
    partial_live = v3_state(player=14, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=1, actor=1)])
    partial_terminal = v3_state(player=14, candidates=Vector{UInt32}[], terminal_reward=7)
    concurrent_live = v3_state(player=15, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=1, actor=1)])
    concurrent_terminal = v3_state(player=15, candidates=Vector{UInt32}[], terminal_reward=8)

    War1gusAI.process_step!(trainer, partial, UInt32(0), Int32(0), partial_live)
    War1gusAI.process_step!(trainer, partial, UInt32(1), Int32(1), partial_live)
    @test length(partial.fragment) == 1
    @test partial.fragment_generation == 0

    actor_policy = trainer.policy
    actor_optimizer = trainer.optimizer_state
    actor_state = model_state_snapshot(actor_policy)
    task = lock(trainer.lock) do
      War1gusAI.process_step!(trainer, training, UInt32(0), Int32(0), live)
      @test War1gusAI.process_terminal!(trainer, training, UInt32(1), Int32(99), terminal)
      scheduled = trainer.worker_task
      @test !isnothing(scheduled)
      @test trainer.worker_active
      @test !istaskdone(scheduled::Task)
      @test trainer.update_count == 0
      @test trainer.policy === actor_policy
      @test trainer.optimizer_state === actor_optimizer
      @test Flux.state(trainer.policy) == actor_state
      @test War1gusAI.process_step!(trainer, during_update, UInt32(0), Int32(0), concurrent_live) in 0:1
      @test !(during_update.previous::War1gusAI.Decision).collectable
      return scheduled
    end

    War1gusAI.flush_trajectories!(trainer)
    @test istaskdone(task::Task)
    @test trainer.update_count == 1
    @test trainer.policy !== actor_policy
    @test trainer.optimizer_state !== actor_optimizer
    @test War1gusAI.process_terminal!(trainer, partial, UInt32(2), Int32(7), partial_terminal)
    @test isempty(partial.fragment)
    @test isnothing(partial.fragment_generation)
    @test War1gusAI.process_terminal!(trainer, during_update, UInt32(1), Int32(8), concurrent_terminal)
    War1gusAI.flush_trajectories!(trainer)
    @test trainer.update_count == 1
    @test trainer.policy_generation == 1
    @test !trainer.worker_active
    payload = open(deserialize, checkpoint)
    @test payload.update_count == 1
  end
end

@testset "PPO worker failures reach flushes and later requests" begin
  mktempdir() do directory
    log_path = joinpath(directory, "failed-update.jsonl")
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=joinpath(directory, "failed-update.jls"),
      seed=34,
      batch_size=1,
      rollout_fragment=1,
      ppo_epochs=1,
    )
    trainer.entropy_coefficient = Float32(NaN)
    session = War1gusAI.ClientSession()
    live = v3_state(player=15, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
    terminal = v3_state(player=15, candidates=Vector{UInt32}[], terminal_reward=-5)

    War1gusAI.start_event_logger!(path=log_path, stdout_io=devnull, mirror_to_stdout=false)
    try
      War1gusAI.process_step!(trainer, session, UInt32(0), Int32(0), live)
      @test War1gusAI.process_terminal!(trainer, session, UInt32(1), Int32(-5), terminal)
      task = trainer.worker_task
      @test !isnothing(task)
      @test_throws ArgumentError War1gusAI.flush_trajectories!(trainer)
      @test istaskdone(task::Task)
      @test !isnothing(trainer.worker_failure)
      @test_throws ArgumentError War1gusAI.process_step!(
        trainer, War1gusAI.ClientSession(), UInt32(0), Int32(0), live,
      )
    finally
      War1gusAI.stop_event_logger!()
    end
    @test any(record -> occursin("\"type\":\"ppo_update_failed\"", record), readlines(log_path))
  end
end

@testset "interleaved players finalize independently" begin
  mktempdir() do directory
    path = joinpath(directory, "interleaved.jls")
    log_path = joinpath(directory, "events.jsonl")
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=path,
      seed=35,
      batch_size=32,
      rollout_fragment=32,
      ppo_epochs=1,
      checkpoint_every=100,
    )
    first = War1gusAI.ClientSession()
    second = War1gusAI.ClientSession()
    first_live = v3_state(player=10, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
    second_live = v3_state(player=11, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=1, actor=1)])
    first_terminal = v3_state(player=10, candidates=Vector{UInt32}[], terminal_reward=101)
    second_terminal = v3_state(player=11, candidates=Vector{UInt32}[], terminal_reward=202)

    withenv("WAR1GUS_AI_VERBOSE_LOG" => "1") do
      War1gusAI.start_event_logger!(; path=log_path, stdout_io=devnull)
      try
        War1gusAI.process_step!(trainer, first, UInt32(0), Int32(0), first_live)
        War1gusAI.process_step!(trainer, second, UInt32(0), Int32(0), second_live)
        @test War1gusAI.process_terminal!(trainer, first, UInt32(1), Int32(101), first_terminal)
        @test !War1gusAI.process_terminal!(trainer, first, UInt32(1), Int32(101), first_terminal)
        @test !second.finalized
        War1gusAI.process_step!(trainer, second, UInt32(1), Int32(31), second_live)
        @test only(second.fragment).reward == 31.0f0
        @test War1gusAI.process_terminal!(trainer, second, UInt32(2), Int32(202), second_terminal)
        @test !isnothing(War1gusAI.flush_trajectories!(trainer))
      finally
        War1gusAI.stop_event_logger!()
      end

      samples = filter(record -> occursin("\"type\":\"training_sample\"", record), readlines(log_path))
      @test count(record -> occursin("\"session_id\":10", record) && occursin("\"reward\":101", record) && occursin("\"terminal\":true", record), samples) == 1
      @test count(record -> occursin("\"session_id\":11", record) && occursin("\"reward\":202", record) && occursin("\"terminal\":true", record), samples) == 1
      @test count(record -> occursin("\"session_id\":11", record), samples) == 2
      @test all(record -> occursin("\"state\":[", record), samples)
      @test any(record -> occursin("\"next_state\":[", record), samples)
    end
  end
end

@testset "training samples are verbose-only diagnostics" begin
  mktempdir() do directory
    checkpoint_path = joinpath(directory, "compact.jls")
    log_path = joinpath(directory, "events.jsonl")
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=checkpoint_path,
      seed=41,
      batch_size=32,
      rollout_fragment=32,
      ppo_epochs=1,
      checkpoint_every=100,
    )
    session = War1gusAI.ClientSession()
    live = v3_state(player=12, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
    terminal = v3_state(player=12, candidates=Vector{UInt32}[], terminal_reward=17)

    withenv("WAR1GUS_AI_VERBOSE_LOG" => nothing) do
      War1gusAI.start_event_logger!(path=log_path, stdout_io=devnull, mirror_to_stdout=false)
      try
        War1gusAI.process_step!(trainer, session, UInt32(0), Int32(0), live)
        @test War1gusAI.process_terminal!(trainer, session, UInt32(1), Int32(17), terminal)
        @test !isnothing(War1gusAI.flush_trajectories!(trainer))
      finally
        War1gusAI.stop_event_logger!()
      end
    end

    records = readlines(log_path)
    @test !any(record -> occursin("\"type\":\"training_sample\"", record), records)
    @test !any(record -> occursin("\"type\":\"reward_decomposition\"", record), records)
    @test !any(record -> occursin("\"state\":[", record), records)
    @test any(record -> occursin("\"type\":\"episode_finalized\"", record), records)
  end
end

@testset "league assigns only the configured seat to PPO" begin
  mktempdir() do directory
    checkpoint = joinpath(directory, "checkpoint.jls")
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_LEAGUE,
      checkpoint_path=checkpoint,
      seed=37,
      train_player=UInt32(0),
      batch_size=32,
      league_snapshot_every=1,
      league_max_snapshots=2,
    )
    @test length(trainer.league_snapshots) == 1
    @test dirname(only(trainer.league_snapshots)) == joinpath(directory, "league")

    train_session = War1gusAI.ClientSession()
    opponent_session = War1gusAI.ClientSession()
    train_state = v3_state(player=0, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=1, actor=1)])
    opponent_state = v3_state(player=1, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
    @test War1gusAI.process_step!(trainer, train_session, UInt32(0), Int32(0), train_state) in 0:1
    @test War1gusAI.process_step!(trainer, opponent_session, UInt32(0), Int32(0), opponent_state) in 0:1
    @test train_session.trainable
    @test !opponent_session.trainable
    @test !isnothing(opponent_session.frozen_policy)
    @test opponent_session.frozen_policy !== trainer.policy
    # Frozen league opponents still sample actions, but never contribute PPO data.
    frozen = opponent_session.frozen_policy::War1gusAI.AiPolicy
    fill!(frozen.score_head.layers[2].weight, 0.0f0)
    fill!(frozen.score_head.layers[2].bias, 0.0f0)
    actor_state = v3_state(
      player=1, entities=[v3_entity(slot=1)],
      candidates=[v3_candidate(), v3_candidate(kind=13, actor=1)],
    )
    choices = Set(War1gusAI.process_step!(
      trainer, opponent_session, UInt32(sequence), Int32(0), actor_state,
    ) for sequence in 1:32)
    @test choices == Set(0:1)
    @test isempty(opponent_session.fragment)
  end
end

@testset "league training explores despite a wait-biased checkpoint" begin
  mktempdir() do directory
    checkpoint = joinpath(directory, "wait-biased.jls")
    policy = War1gusAI.create_policy(seed=61)
    fill!(policy.score_head.layers[2].weight, 0.0f0)
    fill!(policy.score_head.layers[2].bias, 0.0f0)
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_LEAGUE,
      checkpoint_path=checkpoint,
      policy=policy,
      seed=61,
      train_player=UInt32(0),
      batch_size=128,
      rollout_fragment=128,
    )
    War1gusAI.save_checkpoint!(trainer)
    snapshot = only(trainer.league_snapshots)
    train_session = War1gusAI.ClientSession()
    opponent_session = War1gusAI.ClientSession()
    biased_state = player -> v3_state(
      player=player, entities=[v3_entity(slot=1)],
      candidates=[v3_candidate(bootstrap=12_000), v3_candidate(kind=13, actor=1)],
    )
    train_state, opponent_state = biased_state(0), biased_state(1)
    @test War1gusAI.candidate_scores(policy, War1gusAI.parse_state(train_state))[1] -
          War1gusAI.candidate_scores(policy, War1gusAI.parse_state(train_state))[2] ≈ 12.0f0
    train_choices = Int[]
    opponent_choices = Int[]
    for sequence in 0:31
      push!(train_choices, War1gusAI.process_step!(
        trainer, train_session, UInt32(sequence), Int32(0), train_state,
      ))
      push!(opponent_choices, War1gusAI.process_step!(
        trainer, opponent_session, UInt32(sequence), Int32(0), opponent_state,
      ))
    end
    @test 1 in train_choices
    @test 1 in opponent_choices
    scores = War1gusAI.candidate_scores(policy, War1gusAI.parse_state(train_state))
    log_probabilities = War1gusAI.training_action_log_probabilities(
      scores, War1gusAI.parse_state(train_state).candidates,
    )
    @test exp(log_probabilities[1]) ≈ 0.2f0 atol = 1f-4
    @test exp(log_probabilities[2]) ≈ 0.8f0 atol = 1f-4
    @test train_session.previous.log_probability ≈ log_probabilities[last(train_choices)+1] atol = 1f-6
    @test !isempty(train_session.fragment)
    @test isempty(opponent_session.fragment)

    single_wait = player -> v3_state(player=player, candidates=[v3_candidate(bootstrap=12_000)])
    @test War1gusAI.process_step!(
      trainer, train_session, UInt32(32), Int32(0), single_wait(0),
    ) == 0
    @test War1gusAI.process_step!(
      trainer, opponent_session, UInt32(32), Int32(0), single_wait(1),
    ) == 0
    @test isempty(opponent_session.fragment)

    evaluation = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_LEAGUE_EVALUATE,
      checkpoint_path=checkpoint,
      league_snapshot_override=snapshot,
      seed=67,
      train_player=UInt32(0),
    )
    for state in (train_state, opponent_state)
      session = War1gusAI.ClientSession()
      @test all(
        War1gusAI.process_step!(evaluation, session, UInt32(sequence), Int32(0), state) == 0
        for sequence in 0:15
      )
      @test isempty(session.fragment)
    end
  end
end

@testset "league evaluation and read-only guard never update or save" begin
  mktempdir() do directory
    checkpoint = joinpath(directory, "current.jls")
    league_pool = joinpath(directory, "heldout-league")
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_LEAGUE,
      checkpoint_path=checkpoint,
      league_path=league_pool,
      seed=47,
      train_player=UInt32(0),
    )
    War1gusAI.save_checkpoint!(trainer)
    snapshot = only(trainer.league_snapshots)
    forced = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_LEAGUE,
      checkpoint_path=checkpoint,
      league_path=league_pool,
      league_snapshot_override=snapshot,
      league_max_snapshots=1,
      seed=49,
      train_player=UInt32(0),
    )
    forced.update_count = 1
    lock(forced.lock) do
      War1gusAI._save_league_snapshot_locked!(forced)
    end
    @test ispath(snapshot)
    @test forced.league_snapshots == [snapshot]

    evaluation = withenv(
      "WAR1GUS_AI_LEAGUE_DIR" => league_pool,
      "WAR1GUS_AI_SNAPSHOT" => snapshot,
    ) do
      War1gusAI.create_trainer(
        mode=War1gusAI.MODE_LEAGUE_EVALUATE,
        checkpoint_path=checkpoint,
        seed=53,
        train_player=UInt32(0),
      )
    end
    @test War1gusAI.is_league(evaluation)
    @test War1gusAI.is_league_evaluation(evaluation)
    @test !War1gusAI.is_training(evaluation)
    @test War1gusAI.league_directory(evaluation) == league_pool
    @test evaluation.league_snapshot_override == snapshot
    @test Flux.state(evaluation.policy) == Flux.state(trainer.policy)

    train_session = War1gusAI.ClientSession()
    opponent_session = War1gusAI.ClientSession()
    train_state = v3_state(player=0, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=1, actor=1)])
    opponent_state = v3_state(player=1, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
    terminal = v3_state(player=0, candidates=Vector{UInt32}[], terminal_reward=1_000)
    @test War1gusAI.process_step!(evaluation, train_session, UInt32(0), Int32(0), train_state) in 0:1
    @test War1gusAI.process_step!(evaluation, opponent_session, UInt32(0), Int32(0), opponent_state) in 0:1
    @test train_session.trainable
    @test !opponent_session.trainable
    @test !isnothing(opponent_session.frozen_policy)
    @test War1gusAI.process_terminal!(evaluation, train_session, UInt32(1), Int32(1_000), terminal)
    @test evaluation.update_count == 0
    @test isempty(evaluation.pending_fragments)
    @test !evaluation.worker_active
    @test isnothing(evaluation.worker_task)

    read_only_path = joinpath(directory, "read-only.jls")
    readonly = withenv("WAR1GUS_AI_READ_ONLY" => "true") do
      War1gusAI.create_trainer(mode=War1gusAI.MODE_TRAIN, checkpoint_path=read_only_path, seed=59)
    end
    @test !War1gusAI.is_training(readonly)
    War1gusAI.save_checkpoint!(readonly)
    @test !ispath(read_only_path)
    readonly_session = War1gusAI.ClientSession()
    readonly_live = v3_state(player=8, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1)])
    readonly_terminal = v3_state(player=8, candidates=Vector{UInt32}[], terminal_reward=-1_000)
    War1gusAI.process_step!(readonly, readonly_session, UInt32(0), Int32(0), readonly_live)
    War1gusAI.process_terminal!(readonly, readonly_session, UInt32(1), Int32(-1_000), readonly_terminal)
    @test readonly.update_count == 0
    @test !ispath(read_only_path)
    @test !readonly.worker_active
    @test isnothing(readonly.worker_task)
  end
end

@testset "v4 checkpoints reject legacy metadata" begin
  mktempdir() do directory
    path = joinpath(directory, "trajectory.jls")
    trainer = War1gusAI.create_trainer(mode=War1gusAI.MODE_TRAIN, checkpoint_path=path, seed=41)
    War1gusAI.save_checkpoint!(trainer)
    payload = open(deserialize, path)
    @test payload.version == War1gusAI.CHECKPOINT_VERSION
    @test payload.protocol_version == 3
    @test payload.header_words == 22
    @test payload.entity_words == 14
    @test payload.candidate_words == 12
    @test payload.algorithm == War1gusAI.PPO_ALGORITHM
    @test payload.catalog_version == 7
    for catalog_version in (4, 5, 6)
      open(path, "w") do io
        serialize(io, merge(payload, (catalog_version=catalog_version,)))
      end
      restored = War1gusAI.create_trainer(mode=War1gusAI.MODE_INFERENCE, checkpoint_path=path)
      @test restored.update_count == trainer.update_count
    end

    open(path, "w") do io
      serialize(io, (version=3, obsolete_checkpoint=true, update_count=0))
    end
    @test_throws ArgumentError War1gusAI.create_trainer(mode=War1gusAI.MODE_INFERENCE, checkpoint_path=path)
    reset = War1gusAI.create_trainer(mode=War1gusAI.MODE_RESET_TRAIN, checkpoint_path=path, seed=43)
    @test reset.update_count == 0
    @test !ispath(path)
  end
end
