using Sockets

function append_u32_be!(bytes::Vector{UInt8}, value::UInt32)
  append!(bytes, UInt8[
    (value>>24)&0xff,
    (value>>16)&0xff,
    (value>>8)&0xff,
    value&0xff,
  ])
  return bytes
end

@testset "AI processor binary frames" begin
  @test isnothing(War1gusAI.validate_init_frame(UInt8['I', 18, 10]))
  @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['I', 17, 10])
  @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['S', 18, 10])
  @test_throws ArgumentError War1gusAI.validate_init_frame(UInt8['I', 18])

  state = UInt32[1, 2, 1, 0x01020304, 500, 700, 30, 12, 8, 1, 1, 1, 1, 1, 4, 3, 2, 1]
  bytes = UInt8[]
  append_u32_be!(bytes, UInt32(0x11223344))
  foreach(value -> append_u32_be!(bytes, value), state)

  frame = War1gusAI.decode_step_frame(IOBuffer(bytes), UInt8('S'))
  @test frame.reward == UInt32(0x11223344)
  @test frame.state == state
  @test War1gusAI.decode_step_frame(IOBuffer(bytes[1:end-1]), UInt8('E')) === nothing
  @test_throws ArgumentError War1gusAI.decode_step_frame(IOBuffer(bytes), UInt8('X'))
end

@testset "AI processor TCP clients" begin
  server = listen(ip"127.0.0.1", 0)
  port = Int(getsockname(server)[2])
  handlers = Task[]
  accepter = @async for _ in 1:3
    socket = accept(server)
    push!(handlers, @async War1gusAI.handle_client(socket))
  end
  early = UInt32[1, 0, 0, 100, 500, 500, 20, 4, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  ready = UInt32[1, 0, 0, 100, 500, 500, 20, 4, 8, 1, 1, 1, 1, 1, 8, 0, 4, 2]
  clients = TCPSocket[]
  try
    first_client = connect(ip"127.0.0.1", port)
    second_client = connect(ip"127.0.0.1", port)
    malformed_client = connect(ip"127.0.0.1", port)
    append!(clients, [first_client, second_client, malformed_client])

    for (client, state, expected_action) in ((first_client, early, 0), (second_client, ready, 7))
      setup = UInt8['I', 18, 10]
      step = UInt8['S']
      append_u32_be!(step, UInt32(0))
      foreach(value -> append_u32_be!(step, value), state)
      write(client, setup)
      write(client, step)
      flush(client)
      @test Int(read(client, UInt8)) == expected_action

      write(client, step)
      flush(client)
      @test Int(read(client, UInt8)) == expected_action

      terminal = copy(step)
      terminal[1] = UInt8('E')
      write(client, terminal)
      flush(client)
      @test_throws EOFError read(client, UInt8)
    end
    write(malformed_client, UInt8['I'])
  finally
    foreach(client -> isopen(client) && close(client), clients)
  end
  wait(accepter)
  foreach(wait, handlers)
  close(server)
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
  server_task = @async War1gusAI.serve(
    "127.0.0.1",
    game_port;
    listener=game_listener,
    lifecycle_input=control_reader,
  )
  try
    request = UInt8['I', 18, 10, 'S']
    append_u32_be!(request, UInt32(0))
    foreach(value -> append_u32_be!(request, value), UInt32[
      1, 0, 0, 100, 500, 500, 20, 4, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    ])
    write(game_client, request)
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
