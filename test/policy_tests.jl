using Flux
using Random
using Serialization
using Statistics

model_state_snapshot(policy) = deepcopy(Flux.state(policy))

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
  @test War1gusAI.candidate_name(10) == "formation"
  @test_throws ArgumentError War1gusAI.candidate_name(12)

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
      batch_size=2,
      rollout_fragment=16,
      ppo_epochs=2,
      checkpoint_every=1,
    )
    first = v3_state(player=4, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=1, actor=1, bootstrap=50)])
    second = v3_state(player=4, cycle=101, candidates=Vector{UInt32}[v3_candidate(), v3_candidate(kind=7, actor=1, x=30, bootstrap=70)])
    terminal = v3_state(player=4, cycle=102, candidates=Vector{UInt32}[], terminal_reward=1_000)
    before = model_state_snapshot(learner.policy)

    session = War1gusAI.ClientSession()
    War1gusAI.process_step!(learner, session, UInt32(0), Int32(0), first)
    War1gusAI.process_step!(learner, session, UInt32(1), Int32(20), second)
    @test War1gusAI.process_terminal!(learner, session, UInt32(2), Int32(1_000), terminal)
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

    open(path, "w") do io
      serialize(io, (version=3, obsolete_checkpoint=true, update_count=0))
    end
    @test_throws ArgumentError War1gusAI.create_trainer(mode=War1gusAI.MODE_INFERENCE, checkpoint_path=path)
    reset = War1gusAI.create_trainer(mode=War1gusAI.MODE_RESET_TRAIN, checkpoint_path=path, seed=43)
    @test reset.update_count == 0
    @test !ispath(path)
  end
end
