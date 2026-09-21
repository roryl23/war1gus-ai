using Flux
using Random
using Serialization

function representative_state(
 ;
 player=0,
 gold=500,
 wood=500,
 supply=20,
 demand=4,
 workers=8,
 town_halls=1,
 barracks=1,
 lumber_mills=1,
 blacksmiths=1,
 stables=1,
 soldiers=2,
 shooters=2,
 cavalry=2,
 catapults=1,
 enemy_units=8,
 enemy_buildings=5,
 legal_actions=(0, 1, 2, 4, 5, 10, 16, 17, 20, 21, 22, 23),
)
 mask = foldl((bits, action) -> bits | (UInt64(1) << action), legal_actions; init=UInt64(0))
 return UInt32[
  2, player, 0, 1_000, gold, wood, supply, demand, workers, town_halls,
  barracks, lumber_mills, blacksmiths, stables, soldiers, shooters, cavalry, catapults,
  workers+soldiers+shooters+cavalry+catapults,
  town_halls+barracks+lumber_mills+blacksmiths+stables,
  enemy_units, enemy_buildings, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1,
  UInt32(mask & UInt64(0xffffffff)), UInt32(mask >> 32),
 ]
end

model_state_snapshot(policy) = deepcopy(Flux.state(policy))

function json_scalar_field_matches(record::AbstractString, field::AbstractString, value::AbstractString)::Bool
 return occursin(Regex("\\\"" * field * "\\\"\\s*:\\s*" * value * "(?:\\s*[,}])"), record)
end

function json_array_field_matches(record::AbstractString, field::AbstractString, values)::Bool
 contents = join(string.(values), "\\s*,\\s*")
 return occursin(Regex("\\\"" * field * "\\\"\\s*:\\s*\\[" * contents * "\\s*\\]"), record)
end

@testset "v2 primitive action catalog and masked policy" begin
 expected_names = (
  "wait", "gather-gold", "gather-wood", "build-town-hall", "train-worker", "build-farm",
  "build-barracks", "build-lumber-mill", "build-blacksmith", "build-stables", "train-soldier",
  "train-shooter", "train-cavalry", "train-catapult", "research-weapon", "research-armor",
  "attack-nearest-unit", "attack-nearest-building", "attack-weakest-unit", "attack-weakest-building",
  "defend-base", "explore", "prepare-building-space", "repair-building",
 )
 @test War1gusAI.STATE_DIM == 34
 @test War1gusAI.FEATURE_DIM == 32
 @test War1gusAI.ACTION_DIM == 24
 @test War1gusAI.ACTION_NAMES == expected_names
 @test [War1gusAI.action_name(action) for action in 0:23] == collect(expected_names)
 @test_throws ArgumentError War1gusAI.action_name(24)

 policy = War1gusAI.create_policy()
 wait_only = representative_state(legal_actions=(0,))
 @test War1gusAI.legal_mask(wait_only) == UInt64(1)
 @test War1gusAI.select_action(policy, wait_only) == War1gusAI.ACTION_WAIT
 @test all(
  War1gusAI.select_action(policy, wait_only; training=true, rng=MersenneTwister(seed)) == War1gusAI.ACTION_WAIT
  for seed in 1:16
 )

 state = representative_state(legal_actions=(0, 4))
 high_illegal = fill(-20.0f0, War1gusAI.ACTION_DIM)
 high_illegal[War1gusAI.ACTION_ATTACK_NEAREST_UNIT+1] = 1_000.0f0
 high_illegal[War1gusAI.ACTION_TRAIN_WORKER+1] = 2.0f0
 masked = War1gusAI.mask_action_scores(high_illegal, state)
 @test masked[War1gusAI.ACTION_ATTACK_NEAREST_UNIT+1] == -Inf32
 @test argmax(masked) - 1 == War1gusAI.ACTION_TRAIN_WORKER
 @test all(
  War1gusAI.sample_legal_action(masked, MersenneTwister(seed)) in (0, 4)
  for seed in 1:64
 )
 @test all(isfinite, War1gusAI.action_scores(policy, state)[[1, 5]])
 @test all(!isfinite, War1gusAI.action_scores(policy, state)[setdiff(1:24, [1, 5])])
 @test_throws ArgumentError War1gusAI.validate_state(representative_state(legal_actions=()))
 @test_throws ArgumentError War1gusAI.validate_state(representative_state(legal_actions=(4,)))
 @test_throws ArgumentError War1gusAI.encode_state(state[1:end-1])
 @test length(War1gusAI.encode_state(state)) == 32
end

@testset "masked stochastic training actions" begin
 policy = War1gusAI.create_policy()
 state = representative_state(legal_actions=(0, 1, 2, 16, 22))
 actions = [War1gusAI.select_action(policy, state; training=true, rng=MersenneTwister(seed)) for seed in 1:128]
 @test all(action -> War1gusAI.is_legal_action(state, action), actions)
 @test War1gusAI.select_action(policy, state; training=false, rng=MersenneTwister(1)) ==
       War1gusAI.select_action(policy, state; training=false, rng=MersenneTwister(2))
end

@testset "ordered online trainer batches retain masks" begin
 mktempdir() do directory
  trainer = War1gusAI.create_trainer(
   mode=War1gusAI.MODE_TRAIN,
   checkpoint_path=joinpath(directory, "batch.jls"),
   seed=19,
   batch_size=3,
   checkpoint_every=100,
  )
  state = representative_state(legal_actions=(0, 1, 4, 16))
  next_state = representative_state(gold=650, legal_actions=(0, 2, 5, 17))
  action = War1gusAI.select_action(trainer.policy, state)

  @test isnothing(War1gusAI.enqueue_transition!(trainer, state, action, Int32(11), next_state))
  transition = only(trainer.pending_transitions)
  @test transition.legal_mask == War1gusAI.legal_mask(state)
  @test transition.next_legal_mask == War1gusAI.legal_mask(next_state)
  @test_throws ArgumentError War1gusAI.enqueue_transition!(trainer, state, 23, Int32(1), next_state)
  @test isnothing(War1gusAI.enqueue_transition!(trainer, next_state, War1gusAI.ACTION_WAIT, Int32(-7), state))
  before = model_state_snapshot(trainer.policy)
  loss = War1gusAI.enqueue_transition!(trainer, state, action, Int32(23), next_state)

  @test loss isa Float32
  @test trainer.update_count == 1
  @test isempty(trainer.pending_transitions)
  @test Flux.state(trainer.policy) != before
  @test isnothing(War1gusAI.enqueue_transition!(trainer, state, action, Int32(31), next_state))
  @test War1gusAI.update_terminal!(trainer, next_state, War1gusAI.ACTION_WAIT, Int32(-100)) isa Float32
  @test trainer.update_count == 2
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
  first = representative_state(legal_actions=(0, 1, 4))
  second = representative_state(gold=700, legal_actions=(0, 2, 5))
  action = War1gusAI.process_step!(trainer, session, UInt32(0), Int32(0), first)
  War1gusAI.process_step!(trainer, session, UInt32(1), Int32(41), second)

  transition = only(trainer.pending_transitions)
  @test transition.state === first
  @test transition.action == action
  @test transition.legal_mask == War1gusAI.legal_mask(first)
  @test transition.reward == Int32(41)
  @test transition.next_state === second
  @test transition.next_legal_mask == War1gusAI.legal_mask(second)

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
    state = representative_state(gold=500 + index, legal_actions=(0, 1, 4, 16))
    action = War1gusAI.process_step!(concurrent, client, UInt32(0), Int32(0), state)
    yield()
    War1gusAI.process_step!(concurrent, client, UInt32(1), Int32(index), state)
    @test War1gusAI.is_legal_action(state, action)
   end
  end
  @test concurrent.update_count == 1
  @test isempty(concurrent.pending_transitions)
 end
end

@testset "inference does not queue or update" begin
 mktempdir() do directory
  trainer = War1gusAI.create_trainer(mode=War1gusAI.MODE_INFERENCE, checkpoint_path=joinpath(directory, "inference.jls"), seed=31)
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

@testset "training transition events include masks and action names" begin
 mktempdir() do directory
  path = joinpath(directory, "events.jsonl")
  trainer = War1gusAI.create_trainer(mode=War1gusAI.MODE_TRAIN, checkpoint_path=joinpath(directory, "events.jls"), seed=43, batch_size=32, checkpoint_every=100)
  first = representative_state(legal_actions=(0, 1, 4))
  first[2] = UInt32(71)
  second = representative_state(legal_actions=(0, 2, 5))
  second[2] = UInt32(72)
  first_action = 0
  second_action = 0

  War1gusAI.start_event_logger!(; path, stdout_io=devnull)
  try
   session = War1gusAI.ClientSession()
   first_action = War1gusAI.process_step!(trainer, session, UInt32(7), Int32(0), first)
   second_action = War1gusAI.process_step!(trainer, session, UInt32(8), Int32(41), second)
   @test War1gusAI.process_step!(trainer, session, UInt32(8), Int32(999), representative_state()) == second_action
   War1gusAI.process_terminal!(trainer, session, UInt32(9), Int32(-100))
  finally
   War1gusAI.stop_event_logger!()
  end

  records = filter(record -> json_scalar_field_matches(record, "type", "\"training_sample\""), readlines(path))
  @test length(records) == 2
  nonterminal, terminal = records
  @test json_scalar_field_matches(nonterminal, "session_id", "71")
  @test json_array_field_matches(nonterminal, "state", first)
  @test json_scalar_field_matches(nonterminal, "legal_mask", string(War1gusAI.legal_mask(first)))
  @test json_scalar_field_matches(nonterminal, "action", string(first_action))
  @test json_scalar_field_matches(nonterminal, "action_name", "\"" * War1gusAI.action_name(first_action) * "\"")
  @test json_array_field_matches(nonterminal, "next_state", second)
  @test json_scalar_field_matches(nonterminal, "next_legal_mask", string(War1gusAI.legal_mask(second)))
  @test json_scalar_field_matches(nonterminal, "terminal", "false")
  @test json_scalar_field_matches(terminal, "legal_mask", string(War1gusAI.legal_mask(second)))
  @test json_scalar_field_matches(terminal, "action_name", "\"" * War1gusAI.action_name(second_action) * "\"")
  @test json_scalar_field_matches(terminal, "next_legal_mask", "null")
  @test json_scalar_field_matches(terminal, "terminal", "true")
 end
end

@testset "checkpoint v3 metadata and old checkpoint rejection" begin
 mktempdir() do directory
  path = joinpath(directory, "actor_critic.jls")
  trainer = War1gusAI.create_trainer(mode=War1gusAI.MODE_TRAIN, checkpoint_path=path, seed=37, batch_size=1, checkpoint_every=100)
  state = representative_state(legal_actions=(0, 1, 4))
  action = War1gusAI.select_action(trainer.policy, state)
  War1gusAI.enqueue_transition!(trainer, state, action, Int32(5), representative_state(gold=600, legal_actions=(0, 2, 5)))
  War1gusAI.save_checkpoint!(trainer)

  payload = open(deserialize, path)
  @test payload.version == 3
  @test payload.state_dim == 34
  @test payload.feature_dim == 32
  @test payload.action_dim == 24
  @test payload.catalog_version == 2
  @test payload.reward_version == 2
  @test payload.mask_encoding == War1gusAI.LEGAL_MASK_ENCODING
  restored = War1gusAI.create_trainer(mode=War1gusAI.MODE_INFERENCE, checkpoint_path=path, seed=999)
  @test restored.update_count == trainer.update_count
  @test Flux.state(restored.policy) == Flux.state(trainer.policy)

  open(path, "w") do io
   serialize(io, (version=2, state_dim=18, action_dim=10, update_count=0, model_state=Flux.state(trainer.policy), optimizer_state=trainer.optimizer_state))
  end
  @test_throws ArgumentError War1gusAI.create_trainer(mode=War1gusAI.MODE_INFERENCE, checkpoint_path=path)
  reset = War1gusAI.create_trainer(mode=War1gusAI.MODE_RESET_TRAIN, checkpoint_path=path, seed=41)
  @test reset.update_count == 0
  @test !ispath(path)
 end
end
