using Sockets

function append_u32_be!(bytes::Vector{UInt8}, value::UInt32)
 append!(bytes, UInt8[(value>>24)&0xff, (value>>16)&0xff, (value>>8)&0xff, value&0xff])
 return bytes
end

function processor_frame(prefix::Char, sequence::UInt32, reward::Int32, state::Vector{UInt32})
 bytes = UInt8[prefix]
 append_u32_be!(bytes, sequence)
 append_u32_be!(bytes, reinterpret(UInt32, reward))
 foreach(value -> append_u32_be!(bytes, value), state)
 return bytes
end

function protocol_state(; player=0, cycle=100, legal_actions=(0,), enemy_units=8, enemy_buildings=5)
 mask = foldl((bits, action) -> bits | (UInt64(1) << action), legal_actions; init=UInt64(0))
 return UInt32[
  2, player, 0, cycle, 500, 500, 20, 4, 8, 1, 1, 1, 1, 1, 2, 2,
  1, 1, 14, 5, enemy_units, enemy_buildings, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1,
  UInt32(mask & UInt64(0xffffffff)), UInt32(mask >> 32),
 ]
end

@testset "v2 AI processor binary frames and legal mask" begin
 @test isnothing(War1gusAI.validate_init_frame(UInt8['I', 34, 24]))
 @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['I', 33, 24])
 @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['I', 34, 23])
 @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['S', 34, 24])
 @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['I', 34])

 state = protocol_state(player=2, cycle=0x01020304, legal_actions=(0, 4, 16))
 bytes = processor_frame('S', UInt32(0x01020304), Int32(0x11223344), state)[2:end]
 frame = War1gusAI.decode_step_frame(IOBuffer(bytes), UInt8('S'))
 @test frame.sequence == UInt32(0x01020304)
 @test frame.reward == Int32(0x11223344)
 @test frame.state == state
 @test War1gusAI.legal_mask(frame.state) == UInt64(0x0000000000010011)
 negative = processor_frame('S', UInt32(9), Int32(-100), state)[2:end]
 @test War1gusAI.decode_step_frame(IOBuffer(negative), UInt8('S')).reward == Int32(-100)
 @test War1gusAI.decode_step_frame(IOBuffer(bytes[1:end-1]), UInt8('E')) === nothing
 @test_throws ArgumentError War1gusAI.decode_step_frame(IOBuffer(bytes), UInt8('X'))
 @test_throws ArgumentError War1gusAI.decode_step_frame(
  IOBuffer(processor_frame('S', UInt32(0), Int32(0), protocol_state(legal_actions=()))[2:end]),
  UInt8('S'),
 )
 old_state = copy(state)
 old_state[1] = 1
 @test_throws ArgumentError War1gusAI.decode_step_frame(IOBuffer(processor_frame('S', UInt32(0), Int32(0), old_state)[2:end]), UInt8('S'))
end

@testset "server training mode arguments" begin
 @test War1gusAI.parse_server_args(["--host", "0.0.0.0", "--port", "49100", "--train"]) ==
       ("0.0.0.0", 49100, War1gusAI.MODE_TRAIN)
 @test War1gusAI.parse_server_args(["--reset-train"]) ==
       (War1gusAI.DEFAULT_HOST, War1gusAI.DEFAULT_PORT, War1gusAI.MODE_RESET_TRAIN)
 @test War1gusAI.parse_server_args(String[]) ==
       (War1gusAI.DEFAULT_HOST, War1gusAI.DEFAULT_PORT, War1gusAI.MODE_INFERENCE)
 @test_throws ArgumentError War1gusAI.parse_server_args(["--train", "--reset-train"])
end

@testset "AI processor TCP clients emit masked primitive responses" begin
 mktemp() do log_path, log_io
  close(log_io)
  stdout_log = IOBuffer()
  War1gusAI.start_event_logger!(; path=log_path, stdout_io=stdout_log)
  server = listen(ip"127.0.0.1", 0)
  port = Int(getsockname(server)[2])
  trainer = War1gusAI.create_trainer(checkpoint_path=tempname())
  sessions = Dict{UInt32,War1gusAI.ClientSession}()
  sessions_lock = ReentrantLock()
  handlers = Task[]
  accepter = @async for _ in 1:3
   socket = accept(server)
   push!(handlers, @async War1gusAI.handle_client(socket, trainer, sessions, sessions_lock))
  end
  wait_only = protocol_state(player=1, legal_actions=(0,))
  restricted = protocol_state(player=2, legal_actions=(0, 4, 16))
  clients = TCPSocket[]
  try
   first_client = connect(ip"127.0.0.1", port)
   second_client = connect(ip"127.0.0.1", port)
   malformed_client = connect(ip"127.0.0.1", port)
   append!(clients, [first_client, second_client, malformed_client])

   for (client, state) in ((first_client, wait_only), (second_client, restricted))
    write(client, UInt8['I', 34, 24])
    write(client, processor_frame('S', UInt32(0), Int32(0), state))
    flush(client)
    action = Int(read(client, UInt8))
    @test War1gusAI.is_legal_action(state, action)
    if state === wait_only
     @test action == War1gusAI.ACTION_WAIT
    end

    write(client, processor_frame('S', UInt32(1), Int32(0), state))
    flush(client)
    @test Int(read(client, UInt8)) == action
    write(client, processor_frame('E', UInt32(2), Int32(0), state))
    flush(client)
    @test_throws EOFError read(client, UInt8)
   end
   write(malformed_client, UInt8['I', 34, 23])
   flush(malformed_client)
  finally
   foreach(client -> isopen(client) && close(client), clients)
  end
  wait(accepter)
  foreach(wait, handlers)
  close(server)
  War1gusAI.stop_event_logger!()

  log_lines = filter(line -> !isempty(line), split(read(log_path, String), '\n'))
  stdout_lines = filter(line -> !isempty(line), split(String(take!(stdout_log)), '\n'))
  event_types = [only(match(r"\"type\":\"([^\"]+)\"", line).captures) for line in log_lines]
  request_lines = [line for (line, event_type) in zip(log_lines, event_types) if event_type == "network_request"]
  response_lines = [line for (line, event_type) in zip(log_lines, event_types) if event_type == "network_response"]
  client_error_lines = [line for (line, event_type) in zip(log_lines, event_types) if event_type == "client_error"]
  @test log_lines == stdout_lines
  @test length(request_lines) == 8
  @test length(response_lines) == 4
  @test length(client_error_lines) == 1
  @test any(line -> occursin("\"state_dim\":34", line) && occursin("\"action_dim\":24", line), request_lines)
  @test all(line -> occursin("\"legal_mask\":", line), filter(line -> occursin("\"request_kind\":\"step\"", line), request_lines))
  @test all(line -> occursin("\"action_name\":", line) && occursin("\"legal_mask\":", line), response_lines)
 end
end

@testset "stdin lifecycle shutdown" begin
 stopped = listen(ip"127.0.0.1", 0)
 War1gusAI.monitor_lifecycle_input(IOBuffer("ignored\nstop\n"), stopped)
 @test !isopen(stopped)
 ended = listen(ip"127.0.0.1", 0)
 War1gusAI.monitor_lifecycle_input(IOBuffer("ignored\n"), ended)
 @test !isopen(ended)
end

@testset "lifecycle shutdown closes active clients" begin
 game_listener = listen(ip"127.0.0.1", 0)
 game_port = Int(getsockname(game_listener)[2])
 control_listener = listen(ip"127.0.0.1", 0)
 control_port = Int(getsockname(control_listener)[2])
 control_writer = connect(ip"127.0.0.1", control_port)
 control_reader = accept(control_listener)
 game_client = connect(ip"127.0.0.1", game_port)
 trainer = War1gusAI.create_trainer(checkpoint_path=tempname())
 server_task = @async War1gusAI.serve("127.0.0.1", game_port; trainer=trainer, listener=game_listener, lifecycle_input=control_reader)
 try
  write(game_client, UInt8['I', 34, 24])
  state = protocol_state(legal_actions=(0,))
  write(game_client, processor_frame('S', UInt32(0), Int32(0), state))
  flush(game_client)
  @test Int(read(game_client, UInt8)) == 0
  write(control_writer, "stop\n")
  flush(control_writer)
  wait(server_task)
  @test_throws EOFError read(game_client, UInt8)
 finally
  isopen(game_client) && close(game_client)
  isopen(control_writer) && close(control_writer)
  isopen(control_reader) && close(control_reader)
  isopen(control_listener) && close(control_listener)
 end
end

@testset "reconnect preserves and deduplicates client session" begin
 mktempdir() do directory
  server = listen(ip"127.0.0.1", 0)
  port = Int(getsockname(server)[2])
  trainer = War1gusAI.create_trainer(mode=War1gusAI.MODE_TRAIN, checkpoint_path=joinpath(directory, "reconnect.jls"), batch_size=32)
  sessions = Dict{UInt32,War1gusAI.ClientSession}()
  sessions_lock = ReentrantLock()
  handlers = Task[]
  accepter = @async for _ in 1:3
   socket = accept(server)
   push!(handlers, @async War1gusAI.handle_client(socket, trainer, sessions, sessions_lock))
  end
  first_state = protocol_state(player=2, legal_actions=(0, 1, 4))
  second_state = protocol_state(player=2, cycle=101, legal_actions=(0, 2, 5))
  clients = TCPSocket[]
  try
   first = connect(ip"127.0.0.1", port)
   push!(clients, first)
   write(first, UInt8['I', 34, 24])
   write(first, processor_frame('S', UInt32(0), Int32(0), first_state))
   flush(first)
   read(first, UInt8)
   close(first)

   second = connect(ip"127.0.0.1", port)
   push!(clients, second)
   write(second, UInt8['I', 34, 24])
   write(second, processor_frame('S', UInt32(1), Int32(41), second_state))
   flush(second)
   second_action = read(second, UInt8)
   @test length(trainer.pending_transitions) == 1
   close(second)

   retry = connect(ip"127.0.0.1", port)
   push!(clients, retry)
   write(retry, UInt8['I', 34, 24])
   write(retry, processor_frame('S', UInt32(1), Int32(41), second_state))
   flush(retry)
   @test read(retry, UInt8) == second_action
   @test length(trainer.pending_transitions) == 1
   write(retry, processor_frame('E', UInt32(2), Int32(100), second_state))
   flush(retry)
   @test_throws EOFError read(retry, UInt8)
  finally
   foreach(client -> isopen(client) && close(client), clients)
   wait(accepter)
   foreach(wait, handlers)
   close(server)
  end
  @test trainer.update_count == 1
  @test isempty(trainer.pending_transitions)
 end
end

@testset "lifecycle shutdown drains terminal frame" begin
 mktempdir() do directory
  game_listener = listen(ip"127.0.0.1", 0)
  game_port = Int(getsockname(game_listener)[2])
  control_listener = listen(ip"127.0.0.1", 0)
  control_port = Int(getsockname(control_listener)[2])
  control_writer = connect(ip"127.0.0.1", control_port)
  control_reader = accept(control_listener)
  game_client = connect(ip"127.0.0.1", game_port)
  trainer = War1gusAI.create_trainer(mode=War1gusAI.MODE_TRAIN, checkpoint_path=joinpath(directory, "terminal.jls"), batch_size=32)
  server_task = @async War1gusAI.serve("127.0.0.1", game_port; trainer=trainer, listener=game_listener, lifecycle_input=control_reader)
  state = protocol_state(player=3, legal_actions=(0, 1, 4))
  try
   write(game_client, UInt8['I', 34, 24])
   write(game_client, processor_frame('S', UInt32(0), Int32(0), state))
   flush(game_client)
   read(game_client, UInt8)
   write(game_client, processor_frame('E', UInt32(1), Int32(100), state))
   flush(game_client)
   write(control_writer, "stop\n")
   flush(control_writer)
   wait(server_task)
   @test trainer.update_count == 1
   @test isempty(trainer.pending_transitions)
   @test_throws EOFError read(game_client, UInt8)
  finally
   isopen(game_client) && close(game_client)
   isopen(control_writer) && close(control_writer)
   isopen(control_reader) && close(control_reader)
   isopen(control_listener) && close(control_listener)
  end
 end
end
