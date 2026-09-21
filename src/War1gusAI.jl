module War1gusAI

using Sockets

include("logging.jl")
include("model.jl")

const DEFAULT_HOST = "127.0.0.1"
const DEFAULT_PORT = 48721
const INIT_PREFIX = UInt8('I')
const STEP_PREFIX = UInt8('S')
const END_PREFIX = UInt8('E')
const PROTOCOL_VERSION = UInt8(3)
const CLIENT_SHUTDOWN_GRACE_SECONDS = 1.0

struct StepFrame
 sequence::UInt32
 reward::Int32
 state::Vector{UInt32}
 candidate_count::Int
end

"""Validate the exact two-byte v3 setup frame sent by Stratagus."""
function validate_init_frame(frame::AbstractVector{UInt8})::Nothing
 length(frame) == 2 || throw(ArgumentError("AI setup frame must contain exactly two bytes"))
 frame[1] == INIT_PREFIX || throw(ArgumentError("AI setup frame has an invalid prefix"))
 frame[2] == PROTOCOL_VERSION ||
  throw(ArgumentError("AI setup frame has protocol version $(frame[2]), expected $PROTOCOL_VERSION"))
 return nothing
end

function read_exact(io::IO, count::Integer)::Union{Vector{UInt8},Nothing}
 count >= 0 || throw(ArgumentError("cannot read a negative byte count"))
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

function _decode_words(bytes::AbstractVector{UInt8})::Vector{UInt32}
 length(bytes) % 4 == 0 || throw(ArgumentError("AI frame word payload is not u32 aligned"))
 words = Vector{UInt32}(undef, div(length(bytes), 4))
 for index in eachindex(words)
  words[index] = decode_u32_be(bytes, 4 * (index - 1) + 1)
 end
 return words
end

function write_u32_be(io::IO, value::UInt32)::Nothing
 write(io, UInt8((value >> 24) & 0xff))
 write(io, UInt8((value >> 16) & 0xff))
 write(io, UInt8((value >> 8) & 0xff))
 write(io, UInt8(value & 0xff))
 return nothing
end

"""Read one variable-length v3 step or terminal frame after its prefix has been consumed."""
function decode_step_frame(io::IO, prefix::UInt8)::Union{StepFrame,Nothing}
 (prefix == STEP_PREFIX || prefix == END_PREFIX) || throw(ArgumentError("AI frame has an invalid prefix"))
 envelope = read_exact(io, 16)
 isnothing(envelope) && return nothing
 sequence = decode_u32_be(envelope, 1)
 reward = reinterpret(Int32, decode_u32_be(envelope, 5))
 word_count = Int(decode_u32_be(envelope, 9))
 frame_candidate_count = Int(decode_u32_be(envelope, 13))
 frame_candidate_count <= MAX_CANDIDATES ||
  throw(ArgumentError("AI frame has $frame_candidate_count candidates, exceeding the defensive cap of $MAX_CANDIDATES"))
 prefix == STEP_PREFIX && frame_candidate_count < 1 &&
  throw(ArgumentError("AI step frame must contain at least the wait candidate"))
 prefix == END_PREFIX && frame_candidate_count != 0 &&
  throw(ArgumentError("AI terminal frame must not contain candidates"))
 word_count >= STATE_HEADER_WORDS || throw(ArgumentError("AI frame word count $word_count is shorter than the v3 header"))
 word_count <= MAX_STATE_WORDS ||
  throw(ArgumentError("AI frame word count $word_count exceeds the protocol cap of $MAX_STATE_WORDS"))

 header_bytes = read_exact(io, 4 * STATE_HEADER_WORDS)
 isnothing(header_bytes) && return nothing
 header = _decode_words(header_bytes)
 UInt32(header[1]) == STATE_VERSION ||
  throw(ArgumentError("unsupported AI state protocol version $(header[1])"))
 entity_count = Int(header[11])
 state_candidate_count = Int(header[12])
 entity_count <= MAX_ENTITY_COUNT ||
  throw(ArgumentError("AI state has $entity_count entities, exceeding the defensive cap of $MAX_ENTITY_COUNT"))
 state_candidate_count <= MAX_CANDIDATES ||
  throw(ArgumentError("AI state has $state_candidate_count candidates, exceeding the defensive cap of $MAX_CANDIDATES"))
 expected_words = STATE_HEADER_WORDS + ENTITY_WORDS * entity_count + CANDIDATE_WORDS * state_candidate_count
 expected_words <= MAX_STATE_WORDS ||
  throw(ArgumentError("AI frame requires $expected_words words, exceeding the protocol cap of $MAX_STATE_WORDS"))
 word_count == expected_words ||
  throw(ArgumentError("AI frame declares $word_count words, but its header requires $expected_words"))

 tail_bytes = read_exact(io, 4 * (word_count - STATE_HEADER_WORDS))
 isnothing(tail_bytes) && return nothing
 state = vcat(header, _decode_words(tail_bytes))
 terminal = prefix == END_PREFIX
 validate_state(state; terminal, frame_candidate_count)
 return StepFrame(sequence, reward, state, frame_candidate_count)
end

function _candidate_kind_for_log(frame::StepFrame, action::Int)::String
 observation = parse_state(frame.state)
 return candidate_name(candidate_kind(observation.candidates[action+1]))
end

function handle_client(
 socket::TCPSocket,
 trainer::OnlineTrainer,
 sessions::Dict{UInt32,ClientSession},
 sessions_lock::ReentrantLock,
 completed_players::Set{UInt32}=Set{UInt32}(),
)::Nothing
 try
  init = read_exact(socket, 2)
  isnothing(init) && return nothing
  validate_init_frame(init)
  log_event("network_request"; request_kind="init", protocol_version=Int(PROTOCOL_VERSION))

  while true
   prefix_bytes = read_exact(socket, 1)
   isnothing(prefix_bytes) && return nothing
   prefix = prefix_bytes[1]
   frame = decode_step_frame(socket, prefix)
   isnothing(frame) && return nothing
   player = player_of(frame.state)
   session = lock(sessions_lock) do
    player in completed_players && throw(ArgumentError("received a frame for finalized player $player"))
    get!(ClientSession, sessions, player)
   end

   if prefix == STEP_PREFIX
    log_event(
     "network_request";
     request_kind="step",
     player=Int(player),
     sequence=frame.sequence,
     reward=frame.reward,
     word_count=length(frame.state),
     candidate_count=frame.candidate_count,
     reward_components=reward_components(frame.state),
    )
    action = process_step!(trainer, session, frame.sequence, frame.reward, frame.state)
    0 <= action < frame.candidate_count ||
     throw(ArgumentError("policy returned candidate index $action outside 0:$(frame.candidate_count-1)"))
    write_u32_be(socket, UInt32(action))
    flush(socket)
    log_event(
     "network_response";
     request_kind="step",
     player=Int(player),
     sequence=frame.sequence,
     candidate_index=action,
     candidate_kind=_candidate_kind_for_log(frame, action),
     candidate_count=frame.candidate_count,
    )
   else
    log_event(
     "network_request";
     request_kind="end",
     player=Int(player),
     sequence=frame.sequence,
     reward=frame.reward,
     word_count=length(frame.state),
     candidate_count=frame.candidate_count,
     reward_components=reward_components(frame.state),
    )
    finalized = process_terminal!(trainer, session, frame.sequence, frame.reward, frame.state)
    lock(sessions_lock) do
     finalized && push!(completed_players, player)
    end
    return nothing
   end
  end
 catch error
  error isa InterruptException && rethrow()
  log_event("client_error"; error=sprint(showerror, error))
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
  if error isa EOFError
   nothing
  else
   log_event("error"; context="lifecycle_input", error=sprint(showerror, error))
  end
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
 actual_port = Int(getsockname(server)[2])
 log_event("server_start"; host=String(host), port=actual_port, protocol_version=Int(PROTOCOL_VERSION))
 active_lock = ReentrantLock()
 active_sockets = Set{TCPSocket}()
 active_tasks = Set{Task}()
 sessions_lock = ReentrantLock()
 sessions = Dict{UInt32,ClientSession}()
 completed_players = Set{UInt32}()
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
    handle_client(socket, trainer, sessions, sessions_lock, completed_players)
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
  try
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
    flush_trajectories!(trainer)
    save_checkpoint!(trainer)
   end
  finally
   log_event("server_stop"; host=String(host), port=actual_port)
  end
 end
 return nothing
end

function _mode_from_environment()::Symbol
 value = get(ENV, "WAR1GUS_AI_MODE", "")
 isempty(value) && return MODE_INFERENCE
 value == "train" && return MODE_TRAIN
 value == "reset-train" && return MODE_RESET_TRAIN
 value == "league-train" && return MODE_LEAGUE
 value == "league-evaluate" && return MODE_LEAGUE_EVALUATE
 throw(ArgumentError("WAR1GUS_AI_MODE has an unknown value: $value"))
end

function parse_server_args(args::AbstractVector{<:AbstractString}=ARGS)::Tuple{String,Int,Symbol}
 host = DEFAULT_HOST
 port = DEFAULT_PORT
 mode = _mode_from_environment()
 explicit_mode = false
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
  elseif argument in ("--train", "--reset-train", "--league-train", "--league-evaluate")
   explicit_mode && throw(ArgumentError("only one training mode may be selected"))
   explicit_mode = true
   mode = if argument == "--train"
    MODE_TRAIN
   elseif argument == "--reset-train"
    MODE_RESET_TRAIN
   elseif argument == "--league-train"
    MODE_LEAGUE
   else
    MODE_LEAGUE_EVALUATE
   end
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
 trainer = create_trainer(mode=mode, seed=training_seed_from_environment())
 serve(host, port; trainer=trainer)
 return nothing
end

function julia_main()::Cint
 start_event_logger!()
 try
  real_main()
 catch error
  log_event("error"; error=sprint(showerror, error))
  return 1
 finally
  stop_event_logger!()
 end
 return 0
end

end  # module
