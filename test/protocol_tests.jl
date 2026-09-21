using Sockets

function append_u32_be!(bytes::Vector{UInt8}, value::UInt32)
  append!(bytes, UInt8[(value>>24)&0xff, (value>>16)&0xff, (value>>8)&0xff, value&0xff])
  return bytes
end

function v3_entity(; slot=1, relation=1, role=1, x=20, y=20, hp=60, max_hp=60, gold_cost=400, wood_cost=0)
  return UInt32[slot, 0x1234+slot, relation, role, x, y, hp, max_hp, gold_cost, wood_cost, 0, 0, 1, 4]
end

function v3_candidate(; kind=0, actor=0, target=0, auxiliary=0, x=0, y=0, group_size=0, formation=0, cadence=0, distance=0, producers=0, bootstrap=0)
  return UInt32[kind, actor, target, auxiliary, x, y, group_size, formation, cadence, distance, producers, reinterpret(UInt32, Int32(bootstrap))]
end

function v3_state(
  ;
  player=0,
  cycle=100,
  entities=Vector{UInt32}[v3_entity()],
  candidates=Vector{UInt32}[v3_candidate()],
  enemy_progress=0,
  own_loss=0,
  time_reward=0,
  terminal_reward=0,
)
  entity_records = isempty(entities) ? UInt32[] : reduce(vcat, entities)
  candidate_records = isempty(candidates) ? UInt32[] : reduce(vcat, candidates)
  header = UInt32[
    3, player, 0, cycle, 500, 500, 20, 4, 128, 128, length(entities), length(candidates),
    1_000, 1_000, 500, 500, 0, 0,
    reinterpret(UInt32, Int32(enemy_progress)), reinterpret(UInt32, Int32(own_loss)),
    reinterpret(UInt32, Int32(time_reward)), reinterpret(UInt32, Int32(terminal_reward)),
  ]
  return vcat(header, entity_records, candidate_records)
end

function processor_frame(
  prefix::Char,
  sequence::UInt32,
  reward::Int32,
  state::Vector{UInt32};
  candidate_count=Int(state[12]),
)
  bytes = UInt8[prefix]
  append_u32_be!(bytes, sequence)
  append_u32_be!(bytes, reinterpret(UInt32, reward))
  append_u32_be!(bytes, UInt32(length(state)))
  append_u32_be!(bytes, UInt32(candidate_count))
  foreach(value -> append_u32_be!(bytes, value), state)
  return bytes
end

@testset "v3 variable binary framing" begin
  @test isnothing(War1gusAI.validate_init_frame(UInt8['I', 3]))
  @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['I', 2])
  @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['I', 3, 0])
  @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['S', 3])

  entities = Vector{UInt32}[v3_entity(slot=1), v3_entity(slot=2, relation=2, role=3, hp=45)]
  candidates = Vector{UInt32}[v3_candidate(), v3_candidate(kind=6, actor=1, target=2, distance=12, bootstrap=250)]
  state = v3_state(player=2, cycle=0x01020304, entities=entities, candidates=candidates, enemy_progress=90, own_loss=-20, time_reward=-1)
  bytes = processor_frame('S', UInt32(0x01020304), Int32(-100), state)[2:end]
  frame = War1gusAI.decode_step_frame(IOBuffer(bytes), UInt8('S'))
  @test frame.sequence == UInt32(0x01020304)
  @test frame.reward == Int32(-100)
  @test frame.state == state
  @test frame.candidate_count == 2
  @test War1gusAI.reward_components(state) == (enemy_progress=90.0f0, own_loss=-20.0f0, time=-1.0f0, terminal_component=0.0f0)

  many_candidates = Vector{UInt32}[v3_candidate()]
  append!(many_candidates, [v3_candidate(kind=7, actor=1, x=index, bootstrap=index) for index in 1:299])
  many = v3_state(entities=Vector{UInt32}[v3_entity()], candidates=many_candidates)
  @test War1gusAI.decode_step_frame(IOBuffer(processor_frame('S', UInt32(1), Int32(0), many)[2:end]), UInt8('S')).candidate_count == 300
  large_entities = Vector{UInt32}[v3_entity(slot=index) for index in 1:4_100]
  large_state = v3_state(entities=large_entities, candidates=Vector{UInt32}[v3_candidate()])
  @test War1gusAI.decode_step_frame(IOBuffer(processor_frame('S', UInt32(2), Int32(0), large_state)[2:end]), UInt8('S')).candidate_count == 1

  terminal = v3_state(entities=entities, candidates=Vector{UInt32}[], terminal_reward=1_000)
  terminal_frame = War1gusAI.decode_step_frame(IOBuffer(processor_frame('E', UInt32(2), Int32(1_000), terminal)[2:end]), UInt8('E'))
  @test terminal_frame.candidate_count == 0
  @test War1gusAI.parse_state(terminal; terminal=true).candidates == War1gusAI.CandidateObservation[]

  @test_throws ArgumentError War1gusAI.decode_step_frame(IOBuffer(processor_frame('S', UInt32(0), Int32(0), state; candidate_count=1)[2:end]), UInt8('S'))
  @test_throws ArgumentError War1gusAI.decode_step_frame(IOBuffer(processor_frame('S', UInt32(0), Int32(0), terminal)[2:end]), UInt8('S'))
  @test_throws ArgumentError War1gusAI.decode_step_frame(IOBuffer(processor_frame('E', UInt32(0), Int32(0), state)[2:end]), UInt8('E'))
  @test War1gusAI.decode_step_frame(IOBuffer(bytes[1:end-1]), UInt8('S')) === nothing
  @test_throws ArgumentError War1gusAI.decode_step_frame(IOBuffer(bytes), UInt8('X'))

  malformed = copy(state)
  malformed[1] = 2
  @test_throws ArgumentError War1gusAI.decode_step_frame(IOBuffer(processor_frame('S', UInt32(0), Int32(0), malformed)[2:end]), UInt8('S'))
  oversized = copy(state)
  oversized[12] = UInt32(513)
  @test_throws ArgumentError War1gusAI.parse_state(oversized)
end

@testset "v3 server emits a u32 candidate index" begin
  server = listen(ip"127.0.0.1", 0)
  port = Int(getsockname(server)[2])
  trainer = War1gusAI.create_trainer(checkpoint_path=tempname(), seed=7)
  sessions = Dict{UInt32,War1gusAI.ClientSession}()
  sessions_lock = ReentrantLock()
  completed = Set{UInt32}()
  handler = @async War1gusAI.handle_client(accept(server), trainer, sessions, sessions_lock, completed)
  client = connect(ip"127.0.0.1", port)
  candidates = Vector{UInt32}[v3_candidate()]
  append!(candidates, [v3_candidate(kind=7, actor=1, x=index, bootstrap=index == 299 ? 1_000_000 : 0) for index in 1:299])
  state = v3_state(player=5, candidates=candidates)
  try
    write(client, UInt8['I', 3])
    write(client, processor_frame('S', UInt32(0), Int32(0), state))
    flush(client)
    response = War1gusAI.decode_u32_be(War1gusAI.read_exact(client, 4)::Vector{UInt8}, 1)
    @test response == UInt32(299)
    @test response > UInt32(255)

    terminal = v3_state(player=5, candidates=Vector{UInt32}[], terminal_reward=1_000)
    write(client, processor_frame('E', UInt32(1), Int32(1_000), terminal))
    flush(client)
    wait(handler)
    @test UInt32(5) in completed
  finally
    isopen(client) && close(client)
    isopen(server) && close(server)
  end
end

@testset "server modes include launcher league environment" begin
  @test War1gusAI.parse_server_args(["--host", "0.0.0.0", "--port", "49100", "--train"]) ==
        ("0.0.0.0", 49100, War1gusAI.MODE_TRAIN)
  @test War1gusAI.parse_server_args(["--league-train"]) ==
        (War1gusAI.DEFAULT_HOST, War1gusAI.DEFAULT_PORT, War1gusAI.MODE_LEAGUE)
  @test War1gusAI.parse_server_args(["--league-evaluate"]) ==
        (War1gusAI.DEFAULT_HOST, War1gusAI.DEFAULT_PORT, War1gusAI.MODE_LEAGUE_EVALUATE)
  configured = withenv("WAR1GUS_AI_MODE" => "league-train") do
    War1gusAI.parse_server_args(String[])
  end
  evaluation = withenv("WAR1GUS_AI_MODE" => "league-evaluate") do
    War1gusAI.parse_server_args(String[])
  end
  command_override = withenv("WAR1GUS_AI_MODE" => "league-train") do
    War1gusAI.parse_server_args(["--league-evaluate"])
  end
  restart_override = withenv("WAR1GUS_AI_MODE" => "reset-train") do
    War1gusAI.parse_server_args(["--train"])
  end
  selected_seed = withenv("WAR1GUS_AI_SEED" => "73") do
    War1gusAI.training_seed_from_environment()
  end
  configured_checkpoint = withenv("WAR1GUS_AI_CHECKPOINT" => "/tmp/shared-ppo.jls") do
    War1gusAI.default_checkpoint_path()
  end
  @test selected_seed == 73
  @test configured == (War1gusAI.DEFAULT_HOST, War1gusAI.DEFAULT_PORT, War1gusAI.MODE_LEAGUE)
  @test evaluation == (War1gusAI.DEFAULT_HOST, War1gusAI.DEFAULT_PORT, War1gusAI.MODE_LEAGUE_EVALUATE)
  @test configured_checkpoint == "/tmp/shared-ppo.jls"
  @test command_override == (War1gusAI.DEFAULT_HOST, War1gusAI.DEFAULT_PORT, War1gusAI.MODE_LEAGUE_EVALUATE)
  @test restart_override == (War1gusAI.DEFAULT_HOST, War1gusAI.DEFAULT_PORT, War1gusAI.MODE_TRAIN)
end
