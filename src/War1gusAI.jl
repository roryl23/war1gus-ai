module War1gusAI

using Sockets

include("model.jl")

const DEFAULT_HOST = "127.0.0.1"
const DEFAULT_PORT = 48721
const INIT_PREFIX = UInt8('I')
const STEP_PREFIX = UInt8('S')
const END_PREFIX = UInt8('E')
const PROTOCOL_VERSION = UInt32(1)
const CLIENT_SHUTDOWN_GRACE_SECONDS = 1.0

struct StepFrame
 sequence::UInt32
 reward::Int32
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
 bytes = read_exact(io, 4 * (STATE_DIM + 2))
 isnothing(bytes) && return nothing

 state = Vector{UInt32}(undef, STATE_DIM)
 for index in eachindex(state)
  state[index] = decode_u32_be(bytes, 9 + 4 * (index - 1))
 end
 state[1] == PROTOCOL_VERSION || throw(ArgumentError("unsupported AI state protocol version $(state[1])"))
 return StepFrame(
  decode_u32_be(bytes, 1),
  reinterpret(Int32, decode_u32_be(bytes, 5)),
  state,
 )
end

function handle_client(
 socket::TCPSocket,
 trainer::OnlineTrainer,
 sessions::Dict{UInt32,ClientSession},
 sessions_lock::ReentrantLock,
)::Nothing
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
   session_id = frame.state[2]
   session = lock(sessions_lock) do
    get!(ClientSession, sessions, session_id)
   end

   if prefix == STEP_PREFIX
    action = process_step!(trainer, session, frame.sequence, frame.reward, frame.state)
    write(socket, UInt8(action))
    flush(socket)
   else
    process_terminal!(trainer, session, frame.sequence, frame.reward)
    lock(sessions_lock) do
     get(sessions, session_id, nothing) === session && delete!(sessions, session_id)
    end
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
 host::AbstractString,
 port::Integer;
 trainer::OnlineTrainer,
 monitor_stdin::Bool=true,
 lifecycle_input::IO=stdin,
 listener::Union{Nothing,Sockets.TCPServer}=nothing,
)::Nothing
 1 <= port <= typemax(UInt16) || throw(ArgumentError("port must be between 1 and 65535"))
 server = isnothing(listener) ? listen(getaddrinfo(host), port) : listener
 active_lock = ReentrantLock()
 active_sockets = Set{TCPSocket}()
 active_tasks = Set{Task}()
 sessions_lock = ReentrantLock()
 sessions = Dict{UInt32,ClientSession}()
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
    handle_client(socket, trainer, sessions, sessions_lock)
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
  tasks = lock(active_lock) do
   collect(active_tasks)
  end
  timedwait(() -> all(istaskdone, tasks), CLIENT_SHUTDOWN_GRACE_SECONDS)
  sockets = lock(active_lock) do
   collect(active_sockets)
  end
  foreach(socket -> isopen(socket) && close(socket), sockets)
  foreach(wait, tasks)
  if is_training(trainer)
   flush_transitions!(trainer)
   save_checkpoint!(trainer)
  end
 end
 return nothing
end

function parse_server_args(args::AbstractVector{<:AbstractString}=ARGS)::Tuple{String,Int,Symbol}
 host = DEFAULT_HOST
 port = DEFAULT_PORT
 mode = MODE_INFERENCE
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
  elseif argument == "--train"
   mode == MODE_INFERENCE || throw(ArgumentError("only one training mode may be selected"))
   mode = MODE_TRAIN
  elseif argument == "--reset-train"
   mode == MODE_INFERENCE || throw(ArgumentError("only one training mode may be selected"))
   mode = MODE_RESET_TRAIN
  else
   throw(ArgumentError("unknown argument: $argument"))
  end
  index += 1
 end
 1 <= port <= typemax(UInt16) || throw(ArgumentError("port must be between 1 and 65535"))
 return host, port, mode
end

function real_main(args::AbstractVector{<:AbstractString}=ARGS)::Nothing
 host, port, mode = parse_server_args(args)
 trainer = create_trainer(mode=mode)
 is_training(trainer) && warmup_training_runtime!()
 serve(host, port; trainer=trainer)
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
