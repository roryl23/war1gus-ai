using Flux
using Random
using Serialization

function representative_state(; gold=500, wood=500, supply=20, demand=4, workers=8,
  barracks=1, lumber_mills=1, blacksmiths=1, stables=1,
  soldiers=2, shooters=2, cavalry=2, catapults=1)
  return UInt32[
    1, 0, 0, 1_000, gold, wood, supply, demand, workers, 1,
    barracks, lumber_mills, blacksmiths, stables, soldiers, shooters, cavalry, catapults,
  ]
end

model_state_snapshot(policy) = deepcopy(Flux.state(policy))

@testset "state-conditioned transformer policy" begin
  policy = War1gusAI.create_policy()
  early = representative_state(workers=2)
  supply_pressure = representative_state(supply=10, demand=9)
  missing_infrastructure = representative_state(barracks=0, lumber_mills=0, stables=0)
  army_building = representative_state(soldiers=2, shooters=1, cavalry=0, catapults=0)
  blacksmith_building = representative_state(blacksmiths=0, cavalry=0, catapults=0)
  strategic_choice = representative_state(cavalry=0, catapults=0)
  ready_to_attack = representative_state()
  barracks_stage = representative_state(barracks=0, lumber_mills=0, blacksmiths=0, stables=0)
  lumber_stage = representative_state(barracks=1, lumber_mills=0, blacksmiths=0, stables=0)
  blacksmith_stage = representative_state(barracks=1, lumber_mills=1, blacksmiths=0, stables=0)
  stable_stage = representative_state(barracks=1, lumber_mills=1, blacksmiths=1, stables=0)

  progression_actions = War1gusAI.select_action.(Ref(policy), [
    early,
    supply_pressure,
    missing_infrastructure,
    army_building,
    blacksmith_building,
  ])
  @test progression_actions == [0, 1, 2, 4, 3]
  dependency_actions = War1gusAI.select_action.(Ref(policy), [
    barracks_stage,
    lumber_stage,
    blacksmith_stage,
    stable_stage,
  ])
  @test dependency_actions == [2, 2, 3, 2]

  strategic_action = War1gusAI.select_action(policy, strategic_choice)
  prior_only_action = argmax(War1gusAI.action_priors(strategic_choice)) - 1
  @test prior_only_action == 5
  @test strategic_action == 6
  @test strategic_action != prior_only_action
  @test War1gusAI.select_action(policy, ready_to_attack) == 7

  actions = [progression_actions; strategic_action; War1gusAI.select_action(policy, ready_to_attack)]
  @test all(action -> 0 <= action < 10, actions)
  @test all(abs.(War1gusAI.action_logits(policy, strategic_choice)) .<= 1.0f0)
  @test War1gusAI.action_logits(policy, early) != War1gusAI.action_logits(policy, ready_to_attack)
  @test_throws ArgumentError War1gusAI.select_action(policy, early[1:end-1])
end

@testset "legal stochastic training actions" begin
  policy = War1gusAI.create_policy()
  state = representative_state()
  rng = MersenneTwister(12)
  actions = [War1gusAI.select_action(policy, state; training=true, rng) for _ in 1:128]
  priors = War1gusAI.action_priors(state)
  @test all(action -> isfinite(priors[action+1]), actions)
  @test length(unique(actions)) > 1
  @test War1gusAI.select_action(policy, state; training=false, rng=MersenneTwister(1)) ==
        War1gusAI.select_action(policy, state; training=false, rng=MersenneTwister(2))
end

@testset "ordered online trainer batches" begin
  mktempdir() do directory
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=joinpath(directory, "batch.jls"),
      seed=19,
      batch_size=3,
      checkpoint_every=100,
    )
    state = representative_state()
    next_state = representative_state(gold=650)
    action = War1gusAI.select_action(trainer.policy, state)

    @test isnothing(War1gusAI.enqueue_transition!(trainer, state, action, Int32(11), next_state))
    @test isnothing(War1gusAI.enqueue_transition!(trainer, next_state, action, Int32(-7), state))
    @test getfield.(trainer.pending_transitions, :reward) == Int32[11, -7]
    before = model_state_snapshot(trainer.policy)
    loss = War1gusAI.enqueue_transition!(trainer, state, action, Int32(23), next_state)

    @test loss isa Float32
    @test trainer.update_count == 1
    @test isempty(trainer.pending_transitions)
    @test Flux.state(trainer.policy) != before

    @test isnothing(War1gusAI.enqueue_transition!(trainer, state, action, Int32(31), next_state))
    @test trainer.update_count == 1
    @test War1gusAI.update_terminal!(trainer, next_state, action, Int32(-100)) isa Float32
    @test trainer.update_count == 2
    @test isempty(trainer.pending_transitions)
  end
end

@testset "client reward attribution and shared trainer" begin
  mktempdir() do directory
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=joinpath(directory, "shared.jls"),
      seed=23,
      batch_size=32,
      checkpoint_every=100,
    )
    session = War1gusAI.ClientSession()
    first = representative_state()
    second = representative_state(gold=700)
    action = War1gusAI.process_step!(trainer, session, UInt32(0), Int32(0), first)
    War1gusAI.process_step!(trainer, session, UInt32(1), Int32(41), second)

    transition = only(trainer.pending_transitions)
    @test transition.state === first
    @test transition.action == action
    @test transition.reward == Int32(41)
    @test transition.next_state === second

    concurrent = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=joinpath(directory, "concurrent.jls"),
      seed=29,
      batch_size=4,
      checkpoint_every=100,
    )
    @sync for index in 1:4
      @async begin
        client = War1gusAI.ClientSession()
        state = representative_state(gold=500 + index)
        action = War1gusAI.process_step!(concurrent, client, UInt32(0), Int32(0), state)
        yield()
        War1gusAI.process_step!(concurrent, client, UInt32(1), Int32(index), state)
        @test 0 <= action < War1gusAI.ACTION_DIM
      end
    end
    @test concurrent.update_count == 1
    @test isempty(concurrent.pending_transitions)
  end
end

@testset "inference does not queue or update" begin
  mktempdir() do directory
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_INFERENCE,
      checkpoint_path=joinpath(directory, "inference.jls"),
      seed=31,
    )
    before = model_state_snapshot(trainer.policy)
    session = War1gusAI.ClientSession()
    War1gusAI.process_step!(trainer, session, UInt32(0), Int32(0), representative_state())
    War1gusAI.process_step!(trainer, session, UInt32(1), Int32(25), representative_state(gold=900))
    War1gusAI.process_terminal!(trainer, session, UInt32(2), Int32(-100))

    @test trainer.update_count == 0
    @test isempty(trainer.pending_transitions)
    @test Flux.state(trainer.policy) == before
  end
end

@testset "checkpoint load and reset modes" begin
  mktempdir() do directory
    path = joinpath(directory, "actor_critic.jls")
    trainer = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_TRAIN,
      checkpoint_path=path,
      seed=37,
      batch_size=1,
      checkpoint_every=100,
    )
    state = representative_state()
    action = War1gusAI.select_action(trainer.policy, state)
    War1gusAI.enqueue_transition!(trainer, state, action, Int32(5), representative_state(gold=600))
    War1gusAI.save_checkpoint!(trainer)

    restored = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_INFERENCE,
      checkpoint_path=path,
      seed=999,
    )
    @test restored.update_count == trainer.update_count
    @test Flux.state(restored.policy) == Flux.state(trainer.policy)
    @test restored.optimizer_state == trainer.optimizer_state
    @test isfile(path)
    @test !any(name -> endswith(name, ".tmp"), readdir(directory))

    reset = War1gusAI.create_trainer(
      mode=War1gusAI.MODE_RESET_TRAIN,
      checkpoint_path=path,
      seed=41,
    )
    @test reset.update_count == 0
    @test !ispath(path)
    @test Flux.state(reset.policy) == Flux.state(War1gusAI.create_policy(seed=41))

    open(path, "w") do io
      serialize(io, (version=War1gusAI.CHECKPOINT_VERSION + 1, state_dim=18, action_dim=10, update_count=0, model_state=Flux.state(reset.policy)))
    end
    @test_throws ArgumentError War1gusAI.create_trainer(mode=War1gusAI.MODE_INFERENCE, checkpoint_path=path)
  end
end
