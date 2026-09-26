function json_string_end(bytes::Vector{UInt8}, position::Int)::Int
  bytes[position] == UInt8('"') || return 0
  position += 1
  while position <= length(bytes)
    byte = bytes[position]
    if byte == UInt8('"')
      return position + 1
    elseif byte == UInt8('\\')
      position += 1
      position <= length(bytes) || return 0
      escape = bytes[position]
      if escape == UInt8('u')
        position + 4 <= length(bytes) || return 0
        for hex in bytes[position+1:position+4]
          (UInt8('0') <= hex <= UInt8('9') || UInt8('a') <= hex <= UInt8('f') || UInt8('A') <= hex <= UInt8('F')) || return 0
        end
        position += 4
      elseif !(escape in UInt8['"', '\\', '/', 'b', 'f', 'n', 'r', 't'])
        return 0
      end
    elseif byte < 0x20
      return 0
    end
    position += 1
  end
  return 0
end

function json_number_end(bytes::Vector{UInt8}, position::Int)::Int
  start = position
  position <= length(bytes) && bytes[position] == UInt8('-') && (position += 1)
  position <= length(bytes) || return 0
  if bytes[position] == UInt8('0')
    position += 1
  elseif UInt8('1') <= bytes[position] <= UInt8('9')
    while position <= length(bytes) && UInt8('0') <= bytes[position] <= UInt8('9')
      position += 1
    end
  else
    return 0
  end
  if position <= length(bytes) && bytes[position] == UInt8('.')
    position += 1
    decimal_start = position
    while position <= length(bytes) && UInt8('0') <= bytes[position] <= UInt8('9')
      position += 1
    end
    position == decimal_start && return 0
  end
  if position <= length(bytes) && bytes[position] in UInt8['e', 'E']
    position += 1
    position <= length(bytes) && bytes[position] in UInt8['+', '-'] && (position += 1)
    exponent_start = position
    while position <= length(bytes) && UInt8('0') <= bytes[position] <= UInt8('9')
      position += 1
    end
    position == exponent_start && return 0
  end
  return position == start ? 0 : position
end

function json_value_end(bytes::Vector{UInt8}, position::Int)::Int
  position > length(bytes) && return 0
  byte = bytes[position]
  if byte == UInt8('"')
    return json_string_end(bytes, position)
  elseif byte == UInt8('{')
    position += 1
    position <= length(bytes) && bytes[position] == UInt8('}') && return position + 1
    while true
      position = json_string_end(bytes, position)
      if position == 0 || position > length(bytes) || bytes[position] != UInt8(':')
        return 0
      end
      position = json_value_end(bytes, position + 1)
      if position == 0 || position > length(bytes)
        return 0
      end
      if bytes[position] == UInt8('}')
        return position + 1
      elseif bytes[position] != UInt8(',')
        return 0
      end
      position += 1
    end
  elseif byte == UInt8('[')
    position += 1
    position <= length(bytes) && bytes[position] == UInt8(']') && return position + 1
    while true
      position = json_value_end(bytes, position)
      if position == 0 || position > length(bytes)
        return 0
      end
      if bytes[position] == UInt8(']')
        return position + 1
      elseif bytes[position] != UInt8(',')
        return 0
      end
      position += 1
    end
  elseif position + 3 <= length(bytes) && bytes[position:position+3] == codeunits("true")
    return position + 4
  elseif position + 4 <= length(bytes) && bytes[position:position+4] == codeunits("false")
    return position + 5
  elseif position + 3 <= length(bytes) && bytes[position:position+3] == codeunits("null")
    return position + 4
  end
  return json_number_end(bytes, position)
end

function is_json_line(line::AbstractString)::Bool
  isvalid(line) || return false
  bytes = collect(codeunits(line))
  return json_value_end(bytes, 1) == length(bytes) + 1
end
const JSON_STRING_ESCAPES = Dict{UInt8,UInt8}(
  UInt8('"') => UInt8('"'), UInt8('\\') => UInt8('\\'), UInt8('/') => UInt8('/'),
  UInt8('b') => 0x08, UInt8('f') => 0x0c, UInt8('n') => 0x0a,
  UInt8('r') => 0x0d, UInt8('t') => 0x09,
)

function decoded_json_string(bytes::Vector{UInt8}, start::Int)
  stop = json_string_end(bytes, start)
  stop != 0 || throw(ArgumentError("invalid JSON string"))
  value = IOBuffer()
  position = start + 1
  while position < stop - 1
    byte = bytes[position]
    if byte != UInt8('\\')
      write(value, byte)
    else
      position += 1
      escape = bytes[position]
      if escape == UInt8('u')
        codepoint = parse(Int, String(bytes[position+1:position+4]); base=16)
        position += 4
        if 0xd800 <= codepoint <= 0xdbff
          position + 6 < stop && bytes[position+1:position+2] == UInt8['\\', 'u'] ||
            throw(ArgumentError("invalid JSON surrogate pair"))
          low = parse(Int, String(bytes[position+3:position+6]); base=16)
          0xdc00 <= low <= 0xdfff || throw(ArgumentError("invalid JSON surrogate pair"))
          codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00
          position += 6
        elseif 0xdc00 <= codepoint <= 0xdfff
          throw(ArgumentError("invalid JSON surrogate pair"))
        end
        write(value, Char(codepoint))
      else
        escaped = get(JSON_STRING_ESCAPES, escape, nothing)
        isnothing(escaped) && throw(ArgumentError("invalid JSON escape"))
        write(value, escaped)
      end
    end
    position += 1
  end
  return String(take!(value)), stop
end

function decoded_json_field(line::AbstractString, field::AbstractString)::String
  bytes = collect(codeunits(line))
  is_json_line(line) && bytes[1] == UInt8('{') || throw(ArgumentError("invalid JSON object"))
  position = 2
  while bytes[position] != UInt8('}')
    key, position = decoded_json_string(bytes, position)
    bytes[position] == UInt8(':') || throw(ArgumentError("invalid JSON field"))
    position += 1
    if key == field
      value, _ = decoded_json_string(bytes, position)
      return value
    end
    position = json_value_end(bytes, position)
    bytes[position] == UInt8('}') && break
    position += 1
  end
  throw(ArgumentError("missing JSON field: $field"))
end

mutable struct BlockingLoggerIO <: IO
  entered::Channel{Nothing}
  release::Base.Event
end

function Base.write(io::BlockingLoggerIO, value::String)::Int
  put!(io.entered, nothing)
  wait(io.release)
  return ncodeunits(value)
end

Base.write(::BlockingLoggerIO, ::UInt8)::Int = 1
Base.flush(::BlockingLoggerIO)::Nothing = nothing

@testset "event logger default path" begin
  configured_path = joinpath(tempdir(), "custom-war1gus-ai.log")
  fallback_path = joinpath(Sys.BINDIR, "War1gusAI.log")
  withenv("WAR1GUS_AI_LOG_PATH" => nothing) do
    @test War1gusAI.default_log_path() == fallback_path
  end
  withenv("WAR1GUS_AI_LOG_PATH" => "") do
    @test War1gusAI.default_log_path() == fallback_path
  end
  withenv("WAR1GUS_AI_LOG_PATH" => configured_path) do
    @test War1gusAI.default_log_path() == configured_path
  end
end

@testset "verbose logging requires an explicit opt-in" begin
  for value in (nothing, "", "0", "true", "01", "1 ")
    withenv("WAR1GUS_AI_VERBOSE_LOG" => value) do
      @test !War1gusAI.verbose_logging_enabled()
    end
  end
  withenv("WAR1GUS_AI_VERBOSE_LOG" => "1") do
    @test War1gusAI.verbose_logging_enabled()
  end
end


@testset "asynchronous JSON-lines event logger" begin
  War1gusAI.stop_event_logger!()
  path = tempname()
  stdout_buffer = IOBuffer()
  message = "quoted \" slash \\ controls \b\f\n\r\t$(Char(1)) snowman ☃ emoji 😀"

  try
    @test War1gusAI.start_event_logger!(path=path, stdout_io=stdout_buffer, mirror_to_stdout=true) === nothing
    @test War1gusAI.log_event(
      "test_event";
      message=message,
      values=Any[UInt32(7), nothing, (true, "ok")],
      nested=(flag=false, label=:unit),
      bad=Inf,
      unsupported=Dict(:ignored => 1),
      type="must_not_override",
      timestamp=0,
    ) === nothing
    for index in 1:3
      War1gusAI.log_event("queued"; index=index)
    end
    @test War1gusAI.stop_event_logger!() === nothing

    stdout_lines = filter(!isempty, split(String(take!(stdout_buffer)), '\n'))
    file_lines = filter(!isempty, split(read(path, String), '\n'))
    @test stdout_lines == file_lines
    @test length(file_lines) == 4
    @test all(is_json_line, file_lines)

    event = only(filter(line -> occursin("\"type\":\"test_event\"", line), file_lines))
    @test startswith(event, "{\"type\":\"test_event\",\"timestamp\":")
    @test occursin("\"message\":\"quoted \\\" slash \\\\ controls \\b\\f\\n\\r\\t\\u0001 snowman \\u2603 emoji \\ud83d\\ude00\"", event)
    @test occursin("\"values\":[7,null,[true,\"ok\"]]", event)
    @test occursin("\"nested\":{\"flag\":false,\"label\":\"unit\"}", event)
    @test occursin("\"bad\":null", event)
    @test occursin("\"unsupported\":null", event)
    @test !occursin("must_not_override", event)
    @test !occursin("\"timestamp\":0", event)

    restart_path = tempname()
    restart_stdout = IOBuffer()
    @test War1gusAI.start_event_logger!(path=restart_path, stdout_io=restart_stdout, mirror_to_stdout=true) === nothing
    War1gusAI.log_event("restarted"; payload=nothing)
    War1gusAI.stop_event_logger!()
    restart_stdout_lines = filter(!isempty, split(String(take!(restart_stdout)), '\n'))
    restart_file_lines = filter(!isempty, split(read(restart_path, String), '\n'))
    @test restart_stdout_lines == restart_file_lines
    @test length(restart_file_lines) == 1
    @test is_json_line(only(restart_file_lines))
    @test occursin("\"type\":\"restarted\"", only(restart_file_lines))
    @test occursin("\"payload\":null", only(restart_file_lines))
  finally
    War1gusAI.stop_event_logger!()
    rm(path; force=true)
    @isdefined(restart_path) && rm(restart_path; force=true)
  end
end

@testset "configured event log is not mirrored to stdout" begin
  War1gusAI.stop_event_logger!()
  path = tempname()
  stdout_buffer = IOBuffer()
  mirror_buffer = IOBuffer()
  try
    withenv("WAR1GUS_AI_LOG_PATH" => path) do
      @test !War1gusAI.event_logger_mirrors_to_stdout()
      War1gusAI.start_event_logger!(stdout_io=stdout_buffer)
      War1gusAI.log_event("file_only"; payload="kept")
      War1gusAI.stop_event_logger!()

      @test War1gusAI.start_event_logger!(stdout_io=mirror_buffer, mirror_to_stdout=true) === nothing
      War1gusAI.log_event("explicit_mirror")
      War1gusAI.stop_event_logger!()
    end

    @test isempty(String(take!(stdout_buffer)))
    file_lines = filter(!isempty, split(read(path, String), '\n'))
    @test length(file_lines) == 2
    @test all(is_json_line, file_lines)
    @test any(line -> occursin("\"type\":\"file_only\"", line), file_lines)
    @test any(line -> occursin("\"type\":\"explicit_mirror\"", line), file_lines)
    mirror_lines = filter(!isempty, split(String(take!(mirror_buffer)), '\n'))
    @test length(mirror_lines) == 1
    @test occursin("\"type\":\"explicit_mirror\"", only(mirror_lines))
  finally
    War1gusAI.stop_event_logger!()
    rm(path; force=true)
  end
end

@testset "event logger reports unavailable file" begin
  War1gusAI.stop_event_logger!()
  mktempdir() do directory
    stdout_buffer = IOBuffer()
    try
      withenv("WAR1GUS_AI_LOG_PATH" => directory) do
        War1gusAI.start_event_logger!(stdout_io=stdout_buffer)
        War1gusAI.log_event("survives_file_error")
        War1gusAI.stop_event_logger!()
      end

      lines = filter(!isempty, split(String(take!(stdout_buffer)), '\n'))
      @test length(lines) == 1
      @test all(is_json_line, lines)
      error_line = only(filter(line -> occursin("\"type\":\"logger_error\"", line), lines))
      @test occursin("\"operation\":\"open\"", error_line)
      @test decoded_json_field(error_line, "path") == directory
      @test occursin("\"error\":\"", error_line)
    finally
      War1gusAI.stop_event_logger!()
    end
  end
end

@testset "event logger bounded shutdown" begin
  War1gusAI.stop_event_logger!()
  path = tempname()
  stdout_io = BlockingLoggerIO(Channel{Nothing}(1), Base.Event())
  logger = nothing

  try
    @test War1gusAI.start_event_logger!(path=path, stdout_io=stdout_io, mirror_to_stdout=true) === nothing
    logger = War1gusAI.EVENT_LOGGER[]
    @test !isnothing(logger)
    War1gusAI.log_event("blocked_writer")
    @test timedwait(() -> isready(stdout_io.entered), 5.0) == :ok

    elapsed = @elapsed War1gusAI.stop_event_logger!()
    @test elapsed < War1gusAI.EVENT_LOGGER_SHUTDOWN_TIMEOUT_SECONDS + 2.0
    @test War1gusAI.EVENT_LOGGER[] === nothing
  finally
    notify(stdout_io.release)
    War1gusAI.stop_event_logger!()
    if !isnothing(logger)
      @test timedwait(() -> istaskdone(logger.task), 5.0) == :ok
    end
    rm(path; force=true)
  end
end
