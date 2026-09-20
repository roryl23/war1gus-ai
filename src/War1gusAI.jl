module War1gusAI

using Sockets

include("model.jl")

const DEFAULT_HOST = "127.0.0.1"
const DEFAULT_PORT = 48721
const INIT_PREFIX = UInt8('I')
const STEP_PREFIX = UInt8('S')
const END_PREFIX = UInt8('E')
const PROTOCOL_VERSION = UInt32(1)

struct StepFrame
 reward::UInt32
 state::Vector{UInt32}
end

"""Validate the fixed three-byte setup frame sent by Stratagus."""
function validate_init_frame(frame::AbstractVector{UInt8})::Nothing
 length(frame) == 3 || throw(ArgumentError("AI setup frame must contain exactly three bytes"))
 frame[1] == INIT_PREFIX || throw(ArgumentError("AI setup frame has an invalid prefix"))
 Int(frame[2]) == STATE_DIM || throw(ArgumentError("AI setup frame has state dimension $(frame[2]), expected $STATE_DIM"))
 Int(frame[3]) == ACTION_DIM || throw(ArgumentError("AI setup frame has action dimension $(frame[3]), expected $ACTION_DIM"))
 return nothing
end

function read_exact(io::IO, count::Integer)::Union{Vector{UInt8},Nothing}
 bytes = Vector{UInt8}(undef, count)
 for index in eachindex(bytes)
  try
   bytes[index] = read(io, UInt8)
  catch error
   error isa EOFError && return nothing
   rethrow()
  end
 end
 return bytes
end

@inline function decode_u32_be(bytes::AbstractVector{UInt8}, offset::Integer)::UInt32
 return (UInt32(bytes[offset]) << 24) |
        (UInt32(bytes[offset+1]) << 16) |
        (UInt32(bytes[offset+2]) << 8) |
        UInt32(bytes[offset+3])
end

"""Read one fixed-size step or end payload after its prefix has been consumed."""
function decode_step_frame(io::IO, prefix::UInt8)::Union{StepFrame,Nothing}
 (prefix == STEP_PREFIX || prefix == END_PREFIX) || throw(ArgumentError("AI frame has an invalid prefix"))
 bytes = read_exact(io, 4 * (STATE_DIM + 1))
 isnothing(bytes) && return nothing

 state = Vector{UInt32}(undef, STATE_DIM)
 for index in eachindex(state)
  state[index] = decode_u32_be(bytes, 5 + 4 * (index - 1))
 end
 state[1] == PROTOCOL_VERSION || throw(ArgumentError("unsupported AI state protocol version $(state[1])"))
 return StepFrame(decode_u32_be(bytes, 1), state)
end

function handle_client(socket::TCPSocket, policy::AiPolicy=DEFAULT_POLICY)::Nothing
 try
  init = read_exact(socket, 3)
  isnothing(init) && return nothing
  validate_init_frame(init)

  while true
   prefix_bytes = read_exact(socket, 1)
   isnothing(prefix_bytes) && return nothing
   prefix = prefix_bytes[1]
   frame = decode_step_frame(socket, prefix)
   isnothing(frame) && return nothing

   if prefix == STEP_PREFIX
    write(socket, UInt8(select_action(policy, frame.state)))
    flush(socket)
   else
    return nothing
   end
  end
 catch error
  error isa InterruptException && rethrow()
  return nothing
 finally
  isopen(socket) && close(socket)
 end
end

"""Close the listener after a lifecycle `stop` line or end-of-input."""
function monitor_lifecycle_input(input::IO, server::Sockets.TCPServer)::Nothing
 try
  while !eof(input)
   readline(input) == "stop" && break
  end
 catch error
  error isa EOFError || rethrow()
 finally
  isopen(server) && close(server)
 end
 return nothing
end

"""Serve independent Stratagus AI processor connections until lifecycle shutdown."""
function serve(
 host::AbstractString=DEFAULT_HOST,
 port::Integer=DEFAULT_PORT;
 policy::AiPolicy=DEFAULT_POLICY,
 monitor_stdin::Bool=true,
 lifecycle_input::IO=stdin,
 listener::Union{Nothing,Sockets.TCPServer}=nothing,
)::Nothing
 1 <= port <= typemax(UInt16) || throw(ArgumentError("port must be between 1 and 65535"))
 server = isnothing(listener) ? listen(getaddrinfo(host), port) : listener
 active_lock = ReentrantLock()
 active_sockets = Set{TCPSocket}()
 active_tasks = Set{Task}()
 monitor_stdin && errormonitor(@async monitor_lifecycle_input(lifecycle_input, server))
 try
  while isopen(server)
   socket = try
    accept(server)
   catch
    isopen(server) && rethrow()
    break
   end
   lock(active_lock) do
    push!(active_sockets, socket)
   end
   task = @async try
    handle_client(socket, policy)
   finally
    lock(active_lock) do
     delete!(active_sockets, socket)
     delete!(active_tasks, current_task())
    end
   end
   lock(active_lock) do
    push!(active_tasks, task)
   end
   errormonitor(task)
  end
 finally
  isopen(server) && close(server)
  sockets, tasks = lock(active_lock) do
   collect(active_sockets), collect(active_tasks)
  end
  foreach(socket -> isopen(socket) && close(socket), sockets)
  foreach(wait, tasks)
 end
 return nothing
end

function parse_server_args(args::AbstractVector{<:AbstractString}=ARGS)::Tuple{String,Int}
 host = DEFAULT_HOST
 port = DEFAULT_PORT
 index = 1
 while index <= length(args)
  argument = args[index]
  if argument == "--host"
   index == length(args) && throw(ArgumentError("--host requires a value"))
   index += 1
   host = args[index]
  elseif startswith(argument, "--host=")
   host = argument[8:end]
   isempty(host) && throw(ArgumentError("--host requires a value"))
  elseif argument == "--port"
   index == length(args) && throw(ArgumentError("--port requires a value"))
   index += 1
   port = tryparse(Int, args[index])
   isnothing(port) && throw(ArgumentError("--port must be an integer"))
  elseif startswith(argument, "--port=")
   port = tryparse(Int, argument[8:end])
   isnothing(port) && throw(ArgumentError("--port must be an integer"))
  else
   throw(ArgumentError("unknown argument: $argument"))
  end
  index += 1
 end
 1 <= port <= typemax(UInt16) || throw(ArgumentError("port must be between 1 and 65535"))
 return host, port
end

function real_main(args::AbstractVector{<:AbstractString}=ARGS)::Nothing
 host, port = parse_server_args(args)
 serve(host, port)
 return nothing
end

function julia_main()::Cint
 try
  real_main()
 catch
  Base.invokelatest(Base.display_error, Base.catch_stack())
  return 1
 end
 return 0
end

end  # module
