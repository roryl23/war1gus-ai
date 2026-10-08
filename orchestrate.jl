#!/usr/bin/env julia
# Standalone rollout coordinator.  It deliberately uses only Julia's standard library.

using Random
using Sockets

const DEFAULT_WALL_TIMEOUT_SECONDS = 7200
const REQUIRED_OPTIONS = (:launcher, :data_dir, :matches, :workers, :timeout_cycles, :seed,
 :output, :state_root, :rollout_config)

"""Parse the orchestration command line without touching the filesystem."""
function parse_options(arguments::Vector{String})
 isempty(arguments) && throw(ArgumentError("expected mode: train or evaluate"))
 mode = first(arguments)
 mode in ("train", "evaluate") || throw(ArgumentError("mode must be train or evaluate"))
 values = Dict{Symbol,Any}(
  :maps => String[], :held_out_maps => String[], :reset => false, :fast_forward => false,
  :training_mixture => false, :checkpoint => nothing, :league_snapshot => nothing,
  :schedule => nothing, :wall_timeout_seconds => DEFAULT_WALL_TIMEOUT_SECONDS,
 )
 option_names = Dict(
  "--launcher" => :launcher, "--data-dir" => :data_dir, "--matches" => :matches,
  "--workers" => :workers, "--timeout-cycles" => :timeout_cycles, "--seed" => :seed,
  "--wall-timeout-seconds" => :wall_timeout_seconds,
  "--output" => :output, "--state-root" => :state_root,
  "--checkpoint" => :checkpoint, "--league-snapshot" => :league_snapshot,
  "--rollout-config" => :rollout_config, "--schedule" => :schedule,
 )
 specified = Set{Symbol}()
 index = 2
 while index <= length(arguments)
  argument = arguments[index]
  if argument == "--map" || argument == "--held-out-map"
   index == length(arguments) && throw(ArgumentError("$argument requires a value"))
   target = argument == "--map" ? :maps : :held_out_maps
   push!(values[target], arguments[index+1])
   index += 2
  elseif argument in ("--reset", "--fast-forward", "--training-mixture")
   key = Symbol(replace(argument[3:end], '-' => '_'))
   values[key] && throw(ArgumentError("$argument may be supplied once"))
   values[key] = true
   index += 1
  elseif haskey(option_names, argument)
   index == length(arguments) && throw(ArgumentError("$argument requires a value"))
   key = option_names[argument]
   key in specified && throw(ArgumentError("$argument may be supplied once"))
   push!(specified, key)
   raw = arguments[index+1]
   values[key] = key in (:matches, :workers, :timeout_cycles, :seed, :wall_timeout_seconds) ? parse(Int, raw) : raw
   index += 2
  else
   throw(ArgumentError("unknown option: $argument"))
  end
 end
 for key in REQUIRED_OPTIONS
  haskey(values, key) || throw(ArgumentError("missing required option --$(replace(String(key), '_' => '-'))"))
 end
 for key in REQUIRED_OPTIONS
  value = values[key]
  value isa AbstractString && !isempty(value) || value isa Integer ||
   throw(ArgumentError("--$(replace(String(key), '_' => '-')) must not be empty"))
 end
 for key in (:checkpoint, :league_snapshot, :schedule)
  values[key] === nothing || !isempty(values[key]) ||
   throw(ArgumentError("--$(replace(String(key), '_' => '-')) must not be empty"))
 end
 values[:matches] > 0 || throw(ArgumentError("--matches must be positive"))
 values[:matches] <= 20_000 || throw(ArgumentError("--matches exceeds available isolated port range"))
 values[:workers] > 0 || throw(ArgumentError("--workers must be positive"))
 values[:timeout_cycles] > 0 || throw(ArgumentError("--timeout-cycles must be positive"))
 values[:wall_timeout_seconds] > 0 || throw(ArgumentError("--wall-timeout-seconds must be positive"))
 values[:seed] >= 0 || throw(ArgumentError("--seed must be nonnegative"))
 all(value -> !isempty(value), values[:maps]) || throw(ArgumentError("--map must not be empty"))
 all(value -> !isempty(value), values[:held_out_maps]) || throw(ArgumentError("--held-out-map must not be empty"))
 if mode == "train"
  values[:training_mixture] && throw(ArgumentError("--training-mixture is evaluate-only"))
  isempty(values[:maps]) && throw(ArgumentError("train requires at least one --map"))
  values[:reset] && values[:checkpoint] === nothing &&
   throw(ArgumentError("train --reset requires an explicit --checkpoint"))
  values[:workers] == 1 || throw(ArgumentError("train requires --workers 1 because it updates shared state"))
 else
  isempty(values[:held_out_maps]) && throw(ArgumentError("evaluate requires at least one --held-out-map"))
  values[:checkpoint] === nothing && throw(ArgumentError("evaluate requires --checkpoint"))
  values[:reset] && throw(ArgumentError("evaluate cannot reset training state"))
 end
 return (; mode, launcher=values[:launcher], data_dir=values[:data_dir], maps=copy(values[:maps]),
  held_out_maps=copy(values[:held_out_maps]), matches=values[:matches], workers=values[:workers],
  timeout_cycles=values[:timeout_cycles], wall_timeout_seconds=values[:wall_timeout_seconds],
  seed=values[:seed], output=values[:output],
  state_root=values[:state_root], checkpoint=values[:checkpoint],
  league_snapshot=values[:league_snapshot], schedule=values[:schedule], reset=values[:reset],
  fast_forward=values[:fast_forward], training_mixture=values[:training_mixture],
  rollout_config=values[:rollout_config])
end

"""Read the selected map's literal player roster; never execute map Lua to discover seats."""
function map_trainable_seats(options, map::AbstractString)
 path = rollout_map_path(options, map)
 isfile(path) || throw(ArgumentError("selected map is not a file: $path"))
 endswith(lowercase(path), ".smp") ||
  throw(ArgumentError("selected map must be a .smp presentation: $path"))
 presentation = read(path, String)
 declarations = collect(eachmatch(
  r"(?m)^[ \t]*DefinePlayerTypes[ \t]*\(([^\r\n()]*)\)[ \t]*(?:--[^\r\n]*)?\r?$",
  presentation))
 length(declarations) == 1 ||
  throw(ArgumentError("map has no unique literal DefinePlayerTypes roster: $path"))
 roster = declarations[1].captures[1]
 occursin(r"^\s*\"[A-Za-z]+\"(?:\s*,\s*\"[A-Za-z]+\")*\s*$", roster) ||
  throw(ArgumentError("map has unsupported DefinePlayerTypes roster: $path"))
 player_types = [entry.captures[1] for entry in eachmatch(r"\"([A-Za-z]+)\"", roster)]
 all(type -> type in ("computer", "person", "nobody", "neutral"), player_types) ||
  throw(ArgumentError("map has unsupported player types: $path"))
 length(player_types) <= 16 ||
  throw(ArgumentError("map has more than 16 player slots: $path"))

 setup_path = path[1:end-4] * ".sms"
 isfile(setup_path) || throw(ArgumentError("selected map has no .sms setup: $setup_path"))
 # Direct, literal assignments identify a different AI when present. Other
 # computer seats may receive their AI through Load() or setup code, so the
 # presentation roster remains authoritative when no direct assignment exists.
 ai_types = Dict{Int,String}()
 for line in eachline(setup_path)
  assignment = match(r"^[ \t]*SetAiType[ \t]*\([ \t]*(\d+)[ \t]*,[ \t]*\"([A-Za-z0-9_-]+)\"[ \t]*\)[ \t]*(?:--.*)?$", line)
  assignment === nothing && continue
  seat = parse(Int, assignment.captures[1])
  if haskey(ai_types, seat) && ai_types[seat] != assignment.captures[2]
   throw(ArgumentError("map has conflicting AI assignments for seat $seat: $setup_path"))
  end
  ai_types[seat] = assignment.captures[2]
 end
 seats = [index - 1 for (index, player_type) in enumerate(player_types)
          if player_type == "computer" && get(ai_types, index - 1, "war1gus-ai") == "war1gus-ai"]
 isempty(seats) &&
  throw(ArgumentError("map has no eligible computer seats for war1gus-ai: $path"))
 return seats
end

"""Select a distinct non-ephemeral port per match without consuming the schedule RNG."""
function safe_match_ports(ephemeral_low::Int, ephemeral_high::Int)
 1 <= ephemeral_low <= ephemeral_high <= 65535 ||
  throw(ArgumentError("invalid Linux ephemeral port range: $ephemeral_low $ephemeral_high"))
 # Prefer unprivileged low ports when the host uses the usual 32768..60999 range.
 candidates = (10_001:65_535, 1024:10_000)
 return [port for range in candidates for port in range
         if !(ephemeral_low <= port <= ephemeral_high)]
end

function match_port_schedule(matches::Int, state_root::AbstractString, ephemeral_low::Int, ephemeral_high::Int)
 ports = safe_match_ports(ephemeral_low, ephemeral_high)
 length(ports) >= matches ||
  throw(ArgumentError("only $(length(ports)) non-ephemeral ports available for $matches matches; Linux ephemeral range is $ephemeral_low..$ephemeral_high"))
 # Separate run roots normally select different ports, without changing seed/seat draws.
 rotation = Int(mod(hash(abspath(state_root), UInt(0)), UInt(length(ports))))
 return [ports[mod1(rotation + ordinal, length(ports))] for ordinal in 1:matches]
end

function linux_ephemeral_port_range()
 path = "/proc/sys/net/ipv4/ip_local_port_range"
 isfile(path) || throw(ArgumentError("Linux ephemeral port range unavailable: $path"))
 fields = split(strip(read(path, String)))
 length(fields) == 2 || throw(ArgumentError("invalid Linux ephemeral port range in $path"))
 return parse(Int, fields[1]), parse(Int, fields[2])
end

"""Read an explicit schedule in file order, rejecting malformed or ineligible rows."""
function read_explicit_schedule(options, maps, eligible_seats)
 path = options.schedule
 isfile(path) || throw(ArgumentError("--schedule must name a file: $path"))
 text = read(path, String)
 isvalid(text) || throw(ArgumentError("--schedule must be UTF-8: $path"))
 lines = split(text, '\n'; keepempty=true)
 !isempty(lines) && isempty(last(lines)) && pop!(lines)
 !isempty(lines) && first(lines) == "map\ttrain_player\tseed" ||
  throw(ArgumentError("--schedule requires header map<TAB>train_player<TAB>seed"))
 length(lines) - 1 == options.matches ||
  throw(ArgumentError("--schedule row count must equal --matches ($(options.matches))"))
 rows = NamedTuple[]
 seeds = Set{Int}()
 for (line_number, line) in enumerate(@view lines[2:end])
  fields = split(line, '\t'; keepempty=true)
  length(fields) == 3 || throw(ArgumentError("--schedule line $(line_number + 1) must have three TSV columns"))
  map_index = findfirst(==(fields[1]), maps)
  map_index === nothing && throw(ArgumentError("--schedule line $(line_number + 1) has an unselected map"))
  occursin(r"^(0|[1-9][0-9]*)$", fields[2]) ||
   throw(ArgumentError("--schedule line $(line_number + 1) has an invalid train_player"))
  seat = tryparse(Int, fields[2])
  seat !== nothing && seat in eligible_seats[map_index] ||
   throw(ArgumentError("--schedule line $(line_number + 1) has an ineligible train_player"))
  occursin(r"^[1-9][0-9]*$", fields[3]) ||
   throw(ArgumentError("--schedule line $(line_number + 1) has an invalid seed"))
  seed = tryparse(Int32, fields[3])
  seed === nothing && throw(ArgumentError("--schedule line $(line_number + 1) seed exceeds positive Int32"))
  Int(seed) in seeds && throw(ArgumentError("--schedule line $(line_number + 1) repeats a seed"))
  push!(seeds, Int(seed))
  push!(rows, (; map=String(fields[1]), train_player=seat, seed=Int(seed)))
 end
 return rows
end

"""Construct a reproducible, precomputed match schedule so worker ordering cannot affect it."""
function build_schedule(options)
 ephemeral_low, ephemeral_high = linux_ephemeral_port_range()
 ports = match_port_schedule(options.matches, options.state_root, ephemeral_low, ephemeral_high)
 rng = MersenneTwister(options.seed)
 maps = options.mode == "train" ? options.maps : options.held_out_maps
 eligible_seats = [map_trainable_seats(options, map) for map in maps]
 explicit_rows = options.schedule === nothing ? nothing : read_explicit_schedule(options, maps, eligible_seats)
 map_order = options.mode == "train" ? shuffle(rng, collect(eachindex(maps))) : Int[]
 seat_orders = options.mode == "train" ? [shuffle(rng, seats) for seats in eligible_seats] : Vector{Int}[]
 seat_counts = zeros(Int, length(maps))
 schedule = NamedTuple[]
 child_seeds = Set{Int}()
 for ordinal in 1:options.matches
  if explicit_rows === nothing
   map_index = options.mode == "train" ? map_order[mod1(ordinal, length(map_order))] : rand(rng, eachindex(maps))
   map = maps[map_index]
   seat_counts[map_index] += 1
   seats = options.mode == "train" ? seat_orders[map_index] : eligible_seats[map_index]
   seat = options.mode == "train" ? seats[mod1(seat_counts[map_index], length(seats))] :
          seats[Int(mod(UInt(options.seed), UInt(length(seats))))+1]
   child_seed = rand(rng, 1:typemax(Int32))
   while child_seed in child_seeds
    child_seed = rand(rng, 1:typemax(Int32))
   end
   push!(child_seeds, child_seed)
  else
   row = explicit_rows[ordinal]
   map, seat, child_seed = row.map, row.train_player, row.seed
  end
  match_id = string(options.mode, "-", lpad(ordinal, 6, '0'), "-", string(child_seed, base=16))
  port = ports[ordinal]
  push!(schedule, (; ordinal, map, train_player=seat, seed=child_seed, port,
   match_id, mode=options.mode))
 end
 return schedule
end

function match_paths(options, match)
 root = joinpath(options.state_root, "matches", match.match_id)
 return (; root, xdg_state=joinpath(root, "xdg"), ai_log=joinpath(root, "ai.jsonl"),
  child_log=joinpath(root, "launcher.log"))
end

function rollout_map_path(options, map::AbstractString)::String
 return isabspath(map) ? String(map) : abspath(joinpath(options.data_dir, map))
end


function training_checkpoint(options)
 options.mode == "train" || throw(ArgumentError("only training has a mutable checkpoint"))
 return something(options.checkpoint, joinpath(options.state_root, "checkpoint.jls"))
end

training_league_dir(options) = joinpath(dirname(training_checkpoint(options)), "league")

ai_binary_path(options) = joinpath(dirname(options.rollout_config), "build", "bin", "War1gusAI")

function reset_training_state!(options)
 options.mode == "train" && options.reset || return nothing
 checkpoint = options.checkpoint
 checkpoint === nothing && throw(ArgumentError("train --reset requires an explicit --checkpoint"))
 rm(checkpoint; force=true)
 rm(joinpath(dirname(checkpoint), "league"); force=true, recursive=true)
 return nothing
end

"""Build, but do not run, one launcher command and its isolated environment."""

function build_match_command(options, match)
 paths = match_paths(options, match)
 command = String[options.launcher]
 options.mode == "train" && push!(command, "--league-train")
 options.mode == "evaluate" && push!(command, "--league-evaluate")
 push!(command, "-b")
 append!(command, ["-d", options.data_dir, "-c", options.rollout_config])
 checkpoint = options.mode == "train" ? training_checkpoint(options) : something(options.checkpoint, "")
 league_dir = joinpath(dirname(checkpoint), "league")
 environment = Dict{String,String}(
  "WAR1GUS_ROLLOUT_MAP" => rollout_map_path(options, match.map),
  "WAR1GUS_ROLLOUT_SEED" => string(match.seed),
  "WAR1GUS_ROLLOUT_TRAIN_PLAYER" => string(match.train_player),
  "WAR1GUS_ROLLOUT_TIMEOUT_CYCLES" => string(options.timeout_cycles),
  "WAR1GUS_ROLLOUT_FAST_FORWARD" => options.fast_forward ? "1" : "0",
  "WAR1GUS_ROLLOUT_MATCH_ID" => match.match_id,
  "WAR1GUS_ROLLOUT_MODE" => match.mode,
  "WAR1GUS_AI_TRAIN_PLAYER" => string(match.train_player),
  "WAR1GUS_AI_SEED" => string(match.seed),
  "WAR1GUS_AI_BINARY" => ai_binary_path(options),
  "WAR1GUS_AI_CHECKPOINT" => checkpoint,
  "WAR1GUS_AI_LEAGUE_DIR" => league_dir,
  "WAR1GUS_AI_SNAPSHOT" => something(options.league_snapshot, ""),
  "WAR1GUS_AI_LOG_PATH" => paths.ai_log,
  "XDG_STATE_HOME" => paths.xdg_state,
  "STRATAGUS_UNBUFFERED_STDIO" => "1",
  "WAR1GUS_AI_PORT" => string(match.port),
 )
 if options.mode == "evaluate"
  environment["WAR1GUS_AI_READ_ONLY"] = "1"
 end
 environment["WAR1GUS_AI_EVAL_TRAINING_MIXTURE"] = options.training_mixture ? "1" : "0"
 return Cmd(command), environment, paths
end

function json_escape(value::AbstractString)
 output = IOBuffer()
 for character in value
  if character == '"'
   write(output, "\\\"")
  elseif character == '\\'
   write(output, "\\\\")
  elseif character == '\n'
   write(output, "\\n")
  elseif character == '\r'
   write(output, "\\r")
  elseif character == '\t'
   write(output, "\\t")
  elseif UInt32(character) < 0x20
   print(output, "\\u", lpad(string(UInt32(character), base=16), 4, '0'))
  else
   write(output, character)
  end
 end
 return String(take!(output))
end

function json_value(value)
 value === nothing && return "null"
 value isa Bool && return value ? "true" : "false"
 value isa Integer && return string(value)
 value isa AbstractFloat && return isfinite(value) ? string(value) : "null"
 value isa AbstractString && return "\"$(json_escape(value))\""
 if value isa AbstractDict
  return "{" * join((json_value(String(key)) * ":" * json_value(item) for (key, item) in pairs(value)), ",") * "}"
 elseif value isa NamedTuple
  return "{" * join((json_value(String(key)) * ":" * json_value(item) for (key, item) in pairs(value)), ",") * "}"
 elseif value isa AbstractVector || value isa Tuple
  return "[" * join((json_value(item) for item in value), ",") * "]"
 end
 return "null"
end
json_line(type::AbstractString; fields...) = json_value((; type=type, fields...))

# Minimal complete JSON parser for child lines. It accepts objects, arrays, strings, numbers, booleans and null.
mutable struct JsonCursor
 text::String
 index::Int
end
function json_skip!(cursor::JsonCursor)
 while cursor.index <= lastindex(cursor.text) && cursor.text[cursor.index] in (' ', '\t', '\r', '\n')
  cursor.index = nextind(cursor.text, cursor.index)
 end
end
function json_string!(cursor::JsonCursor)
 cursor.text[cursor.index] == '"' || throw(ArgumentError("expected JSON string"))
 cursor.index = nextind(cursor.text, cursor.index)
 output = IOBuffer()
 while cursor.index <= lastindex(cursor.text)
  character = cursor.text[cursor.index]
  cursor.index = nextind(cursor.text, cursor.index)
  character == '"' && return String(take!(output))
  if character == '\\'
   cursor.index > lastindex(cursor.text) && throw(ArgumentError("unterminated escape"))
   escape = cursor.text[cursor.index]
   cursor.index = nextind(cursor.text, cursor.index)
   if escape in ('"', '\\', '/')
    write(output, escape)
   elseif escape == 'b'
    write(output, '\b')
   elseif escape == 'f'
    write(output, '\f')
   elseif escape == 'n'
    write(output, '\n')
   elseif escape == 'r'
    write(output, '\r')
   elseif escape == 't'
    write(output, '\t')
   elseif escape == 'u'
    start = cursor.index
    for _ in 1:4
     cursor.index <= lastindex(cursor.text) && isxdigit(cursor.text[cursor.index]) ||
      throw(ArgumentError("invalid Unicode escape"))
     cursor.index = nextind(cursor.text, cursor.index)
    end
    write(output, "\\u", SubString(cursor.text, start, prevind(cursor.text, cursor.index)))
   else
    throw(ArgumentError("invalid JSON escape"))
   end
  elseif UInt32(character) < 0x20
   throw(ArgumentError("control character in JSON string"))
  else
   write(output, character)
  end
 end
 throw(ArgumentError("unterminated JSON string"))
end
function json_literal!(cursor::JsonCursor, literal::String, value)
 startswith(SubString(cursor.text, cursor.index), literal) || throw(ArgumentError("invalid JSON literal"))
 for _ in literal
  cursor.index = nextind(cursor.text, cursor.index)
 end
 return value
end
function json_value!(cursor::JsonCursor)
 json_skip!(cursor)
 cursor.index > lastindex(cursor.text) && throw(ArgumentError("missing JSON value"))
 character = cursor.text[cursor.index]
 character == '"' && return json_string!(cursor)
 character == '{' && return json_object!(cursor)
 character == '[' && return json_array!(cursor)
 character == 't' && return json_literal!(cursor, "true", true)
 character == 'f' && return json_literal!(cursor, "false", false)
 character == 'n' && return json_literal!(cursor, "null", nothing)
 start = cursor.index
 while cursor.index <= lastindex(cursor.text) && !(cursor.text[cursor.index] in (' ', '\t', '\r', '\n', ',', ']', '}'))
  cursor.index = nextind(cursor.text, cursor.index)
 end
 token = SubString(cursor.text, start, prevind(cursor.text, cursor.index))
 occursin(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$", token) || throw(ArgumentError("invalid JSON number"))
 return occursin(r"[.eE]", token) ? parse(Float64, token) : parse(Int, token)
end
function json_array!(cursor::JsonCursor)
 cursor.index = nextind(cursor.text, cursor.index)
 values = Any[]
 json_skip!(cursor)
 cursor.index <= lastindex(cursor.text) && cursor.text[cursor.index] == ']' && (cursor.index = nextind(cursor.text, cursor.index); return values)
 while true
  push!(values, json_value!(cursor))
  json_skip!(cursor)
  cursor.index <= lastindex(cursor.text) || throw(ArgumentError("unterminated JSON array"))
  cursor.text[cursor.index] == ']' && (cursor.index = nextind(cursor.text, cursor.index); return values)
  cursor.text[cursor.index] == ',' || throw(ArgumentError("expected JSON array separator"))
  cursor.index = nextind(cursor.text, cursor.index)
 end
end
function json_object!(cursor::JsonCursor)
 cursor.index = nextind(cursor.text, cursor.index)
 object = Dict{String,Any}()
 json_skip!(cursor)
 cursor.index <= lastindex(cursor.text) && cursor.text[cursor.index] == '}' && (cursor.index = nextind(cursor.text, cursor.index); return object)
 while true
  json_skip!(cursor)
  key = json_string!(cursor)
  json_skip!(cursor)
  cursor.index <= lastindex(cursor.text) && cursor.text[cursor.index] == ':' || throw(ArgumentError("expected JSON object colon"))
  cursor.index = nextind(cursor.text, cursor.index)
  object[key] = json_value!(cursor)
  json_skip!(cursor)
  cursor.index <= lastindex(cursor.text) || throw(ArgumentError("unterminated JSON object"))
  cursor.text[cursor.index] == '}' && (cursor.index = nextind(cursor.text, cursor.index); return object)
  cursor.text[cursor.index] == ',' || throw(ArgumentError("expected JSON object separator"))
  cursor.index = nextind(cursor.text, cursor.index)
 end
end
function parse_json_object(line::AbstractString)
 cursor = JsonCursor(String(line), firstindex(line))
 json_skip!(cursor)
 cursor.index <= lastindex(cursor.text) && cursor.text[cursor.index] == '{' || throw(ArgumentError("JSON line is not an object"))
 object = json_object!(cursor)
 json_skip!(cursor)
 cursor.index > lastindex(cursor.text) || throw(ArgumentError("trailing JSON data"))
 return object
end
function child_json_object(line::AbstractString)
 try
  object = parse_json_object(line)
  return get(object, "type", nothing) isa AbstractString ? object : nothing
 catch
  return nothing
 end
end

const TERMINAL_NUMERIC_FIELDS = (
 "trainable_player", "seed", "cycles", "kills", "razings", "gold", "wood",
 "units", "buildings", "total_units", "total_buildings",
 "produced_units", "completed_buildings", "lost_units", "lost_buildings",
)

function collect_events_of_type(lines, event_type::AbstractString)
 events = Dict{String,Any}[]
 for line in lines
  event = child_json_object(line)
  event === nothing && continue
  event["type"] == event_type && push!(events, event)
 end
 return events
end

function collect_result_events(paths)
 events = collect_events_of_type(eachline(paths.child_log), "rollout_terminal")
 isfile(paths.ai_log) || return events
 append!(events, collect_events_of_type(eachline(paths.ai_log), "error"))
 return events
end

function child_environment(overrides::AbstractDict{String,String})::Dict{String,String}
 environment = Dict{String,String}(ENV)
 delete!(environment, "LD_LIBRARY_PATH")
 delete!(environment, "LD_PRELOAD")
 merge!(environment, overrides)
 return environment
end

function require_rollout_terminal(events, match; expected_map::AbstractString=match.map)
 any(event -> get(event, "type", nothing) == "error", events) &&
  throw(ArgumentError("AI server reported an error"))

 terminals = [
  event for event in events
  if get(event, "type", nothing) == "rollout_terminal" &&
  get(event, "match_id", nothing) == match.match_id
 ]
 length(terminals) == 1 ||
  throw(ArgumentError("expected exactly one rollout_terminal for $(match.match_id), found $(length(terminals))"))
 terminal = only(terminals)
 get(terminal, "map", nothing) == expected_map ||
  throw(ArgumentError("rollout_terminal map does not match $(match.match_id)"))
 get(terminal, "mode", nothing) == match.mode ||
  throw(ArgumentError("rollout_terminal mode does not match $(match.match_id)"))
 get(terminal, "trainable_player", nothing) == match.train_player ||
  throw(ArgumentError("rollout_terminal player does not match $(match.match_id)"))
 get(terminal, "seed", nothing) == match.seed ||
  throw(ArgumentError("rollout_terminal seed does not match $(match.match_id)"))
 get(terminal, "outcome", nothing) in ("win", "loss", "draw", "timeout") ||
  throw(ArgumentError("rollout_terminal has an invalid outcome"))
 all(field -> get(terminal, field, nothing) isa Number, TERMINAL_NUMERIC_FIELDS) ||
  throw(ArgumentError("rollout_terminal is missing a numeric metric"))
 return terminal
end

number_field(record, field::String) = begin
 value = get(record, field, 0)
 value isa Number ? Float64(value) : 0.0
end
function outcome_for(record)
 outcome = lowercase(string(get(record, "outcome", get(record, "result", ""))))
 outcome in ("win", "loss", "draw", "timeout") && return outcome
 get(record, "timeout", false) === true && return "timeout"
 winner = get(record, "winner", nothing)
 player = get(record, "train_player", get(record, "player", nothing))
 winner isa Number && player isa Number && return winner == player ? "win" : "loss"
 return "draw"
end

"""Aggregate rollout terminal records into evaluation metrics with zero-safe efficiencies."""
function aggregate_evaluation(records::AbstractVector)
 matches = length(records)
 wins = 0
 losses = 0
 draws = 0
 timeouts = 0
 cycles = 0.0
 win_cycles = 0.0
 kills = 0.0
 razings = 0.0
 own_losses = 0.0
 total_production = 0.0
 gold = 0.0
 wood = 0.0
 per_map = Dict{String,Dict{String,Any}}()
 for record in records
  outcome = outcome_for(record)
  outcome == "win" && (wins += 1)
  outcome == "loss" && (losses += 1)
  outcome == "draw" && (draws += 1)
  outcome == "timeout" && (timeouts += 1)
  match_cycles = number_field(record, "cycles")
  cycles += match_cycles
  outcome == "win" && (win_cycles += match_cycles)
  kills += number_field(record, "kills")
  razings += number_field(record, "razings")
  total_production += number_field(record, "produced_units") + number_field(record, "completed_buildings")
  own_losses += number_field(record, "lost_units") + number_field(record, "lost_buildings")
  gold += number_field(record, "gold")
  wood += number_field(record, "wood")
  map = string(get(record, "map", "unknown"))
  entry = get!(per_map, map, Dict{String,Any}("matches" => 0, "wins" => 0, "losses" => 0, "draws" => 0, "timeouts" => 0))
  entry["matches"] += 1
  entry[outcome == "win" ? "wins" : outcome == "loss" ? "losses" : outcome == "timeout" ? "timeouts" : "draws"] += 1
 end
 destroyed_assets = kills + razings
 return Dict{String,Any}(
  "matches" => matches, "wins" => wins, "losses" => losses, "draws" => draws, "timeouts" => timeouts,
  "win_rate" => matches == 0 ? 0.0 : wins / matches,
  "mean_cycles" => matches == 0 ? 0.0 : cycles / matches,
  "mean_elimination_time" => wins == 0 ? 0.0 : win_cycles / wins,
  "destroyed_assets" => destroyed_assets,
  "own_losses" => own_losses,
  "total_production" => total_production,
  "asset_efficiency" => own_losses == 0 ? 0.0 : destroyed_assets / own_losses,
  "combat_efficiency" => own_losses == 0 ? 0.0 : kills / own_losses,
  "total_gold" => gold, "total_wood" => wood, "per_map" => per_map,
 )
end

function emit!(io::IO, event_lock::ReentrantLock, type::AbstractString; fields...)
 line = json_line(type; fields...)
 lock(event_lock) do
  write(io, line, '\n')
  flush(io)
 end
 return nothing
end

"""Check the address the AI binds before starting a match; the AI remains responsible for bind races."""
function require_available_port(port::Int)
 try
  socket = listen(ip"0.0.0.0", port)
  close(socket)
 catch error
  throw(ErrorException("AI port $port is unavailable: $(sprint(showerror, error))"))
 end
 return nothing
end

"""Run one launcher in its own Linux process group and reap it on every exit path."""
function run_match_process(command, environment, child_log, timeout_seconds::Int)
 spawned = run(pipeline(detach(setenv(command, child_environment(environment))),
   stdout=child_log, stderr=child_log); wait=false)
 leader = spawned isa Base.Process ? spawned : first(spawned.processes)
 group_id = getpid(leader)
 timed_out = false
 try
  timed_out = timedwait(() -> !process_running(leader), timeout_seconds; pollint=0.1) == :timed_out
 finally
  # Detached processes have a private process group. Kill it even if the launcher
  # exited first, since the Julia server may otherwise outlive its parent.
  # Preserve the group ID: getpid(leader) fails once the launcher has exited.
  result = ccall(:kill, Cint, (Cint, Cint), -group_id, Base.SIGKILL)
  kill_errno = result == 0 ? 0 : Base.Libc.errno()
  wait(spawned)
  (result == 0 || kill_errno == Base.Libc.ESRCH) ||
   throw(SystemError("kill launcher process group $group_id", kill_errno))
 end
 return timed_out, leader.exitcode
end

function run_one_match(options, match, output, output_lock)
 command, environment, paths = build_match_command(options, match)
 mkpath(dirname(paths.root))
 mkdir(paths.root) # Never overwrite a prior invocation's match logs.
 mkpath(paths.xdg_state)
 if options.mode == "train"
  mkpath(dirname(environment["WAR1GUS_AI_CHECKPOINT"]))
  mkpath(environment["WAR1GUS_AI_LEAGUE_DIR"])
 end
 emit!(output, output_lock, "match_start"; match_id=match.match_id, map=match.map, mode=match.mode, command=command.exec)
 timed_out = false
 status = 1
 launch_error = nothing
 open(paths.child_log, "w") do child_log
  try
   require_available_port(match.port)
   timed_out, status = run_match_process(command, environment, child_log, options.wall_timeout_seconds)
  catch error
   launch_error = sprint(showerror, error)
  end
 end
 if timed_out
  reason = "launcher and descendants exceeded $(options.wall_timeout_seconds)s wall-clock limit"
  emit!(output, output_lock, "child_timeout"; match_id=match.match_id, timeout_seconds=options.wall_timeout_seconds,
   error=reason, log_path=paths.child_log)
  throw(ErrorException("$reason for $(match.match_id)"))
 end
 events = collect_result_events(paths)
 if status != 0 || launch_error !== nothing
  ai_errors = [string(get(event, "error", event)) for event in events if get(event, "type", nothing) == "error"]
  reason = something(launch_error, isempty(ai_errors) ? "launcher exited with code $status" : join(ai_errors, "; "))
  emit!(output, output_lock, "child_failure"; match_id=match.match_id, exit_code=status,
   error=reason, log_path=paths.child_log)
  throw(ErrorException("launcher failed for $(match.match_id): $reason"))
 end
 terminal = try
  require_rollout_terminal(events, match; expected_map=rollout_map_path(options, match.map))
 catch error
  emit!(output, output_lock, "result_failure"; match_id=match.match_id, error=sprint(showerror, error), log_path=paths.child_log)
  rethrow()
 end
 terminal["map"] = match.map
 evaluation = Dict{String,Any}[]
 if options.mode == "evaluate"
  push!(evaluation, terminal)
  emit!(output, output_lock, "evaluation_record"; match_id=match.match_id, record=terminal)
 end
 emit!(output, output_lock, "match_complete"; match_id=match.match_id, log_path=paths.child_log)
 return evaluation
end

function validate_runtime_paths(options)
 isfile(options.launcher) && isexecutable(options.launcher) || throw(ArgumentError("--launcher must name an executable file"))
 isdir(options.data_dir) || throw(ArgumentError("--data-dir must name a directory"))
 isfile(options.rollout_config) || throw(ArgumentError("--rollout-config must name a file"))
 binary = ai_binary_path(options)
 isfile(binary) && isexecutable(binary) || throw(ArgumentError("compiled War1gus AI is not executable: $binary"))
 if options.mode == "evaluate"
  isfile(options.checkpoint) || throw(ArgumentError("--checkpoint must name a file for evaluation"))
  options.league_snapshot === nothing || isfile(options.league_snapshot) ||
   throw(ArgumentError("--league-snapshot must name a file"))
 end
 return nothing
end

function sync_runtime_files(options)
 source_dir = abspath(joinpath(@__DIR__, "..", "..", ".."))
 script = joinpath(source_dir, "cmake", "sync-runtime.cmake")
 isfile(script) || throw(ArgumentError("runtime sync script is missing: $script"))
 data_dir = abspath(options.data_dir)
 try
  run(addenv(`cmake -DWAR1GUS_SOURCE_DIR=$source_dir -DWAR1GUS_DATA_DIR=$data_dir -P $script`,
   "LD_LIBRARY_PATH" => nothing))
 catch error
  throw(ErrorException("runtime sync failed for --data-dir $(options.data_dir): $(sprint(showerror, error))"))
 end
 return nothing
end

function validate_selected_maps(options)
 maps = options.mode == "train" ? options.maps : options.held_out_maps
 for map in maps
  path = rollout_map_path(options, map)
  isfile(path) || throw(ArgumentError("selected map is not a file: $path"))
 end
 return nothing
end
function run_orchestration(options; output_created=Ref(false))
 ispath(options.output) && throw(ArgumentError("output already exists; use a fresh run path: $(options.output)"))
 validate_runtime_paths(options)
 sync_runtime_files(options)
 validate_selected_maps(options)
 schedule = build_schedule(options)
 for match in schedule
  ispath(match_paths(options, match).root) &&
   throw(ArgumentError("match state already exists; use a fresh --state-root: $(match_paths(options, match).root)"))
 end
 mkpath(dirname(options.output))
 mkpath(options.state_root)
 mkdir(joinpath(options.state_root, ".orchestrator-run"))
 failures = Any[]
 evaluation = Dict{String,Any}[]
 output_lock = ReentrantLock()
 result_lock = ReentrantLock()
 flags = Base.JL_O_WRONLY | Base.JL_O_CREAT | Base.JL_O_EXCL
 fd = ccall(:open, Cint, (Cstring, Cint, Cint), options.output, flags, 0o666)
 fd >= 0 || throw(SystemError("open coordinator output $(options.output)", Base.Libc.errno()))
 open(Base.RawFD(fd)) do output
  output_created[] = true
  options.mode == "train" && options.reset && reset_training_state!(options)
  for match in schedule
   emit!(output, output_lock, "schedule"; match_id=match.match_id, map=match.map, train_player=match.train_player, seed=string(match.seed), port=match.port, mode=match.mode, action_selection=options.mode == "evaluate" ? (options.training_mixture ? "training_mixture" : "greedy") : "training_mixture")
  end
  queue = Channel{Any}(length(schedule))
  foreach(match -> put!(queue, match), schedule)
  close(queue)
  workers = Task[]
  for _ in 1:min(options.workers, length(schedule))
   push!(workers, @async begin
    for match in queue
     try
      records = run_one_match(options, match, output, output_lock)
      lock(result_lock) do
       append!(evaluation, records)
      end
     catch error
      lock(result_lock) do
       push!(failures, error)
      end
     end
    end
   end)
  end
  foreach(wait, workers)
  if options.mode == "evaluate"
   metrics = aggregate_evaluation(evaluation)
   emit!(output, output_lock, "evaluation_aggregate"; metrics=metrics, action_selection=options.training_mixture ? "training_mixture" : "greedy")
  end
  emit!(output, output_lock, "orchestration_complete"; mode=options.mode, matches=length(schedule), failures=length(failures))
 end
 isempty(failures) || throw(first(failures))
 return nothing
end

function main(arguments=copy(ARGS))
 output = nothing
 output_created = Ref(false)
 try
  options = parse_options(arguments)
  output = options.output
  run_orchestration(options; output_created)
  return 0
 catch error
  line = json_line("orchestration_error"; error=sprint(showerror, error))
  if output === nothing || !output_created[]
   println(stderr, line)
  else
   try
    open(output, "a") do io
     write(io, line, '\n')
    end
   catch
    println(stderr, line)
   end
  end
  return 1
 end
end

if abspath(PROGRAM_FILE) == @__FILE__
 exit(main())
end
