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
  fallback_path = joinpath(Sys.BINDIR, "War1gusAI.log")
  withenv("WAR1GUS_AI_LOG_PATH" => nothing) do
    @test War1gusAI.default_log_path() == fallback_path
  end
  withenv("WAR1GUS_AI_LOG_PATH" => "") do
    @test War1gusAI.default_log_path() == fallback_path
  end
  withenv("WAR1GUS_AI_LOG_PATH" => "/tmp/custom-war1gus-ai.log") do
    @test War1gusAI.default_log_path() == "/tmp/custom-war1gus-ai.log"
  end
end


@testset "asynchronous JSON-lines event logger" begin
  War1gusAI.stop_event_logger!()
  path = tempname()
  stdout_buffer = IOBuffer()
  message = "quoted \" slash \\ controls \b\f\n\r\t$(Char(1)) snowman ☃ emoji 😀"

  try
    @test War1gusAI.start_event_logger!(path=path, stdout_io=stdout_buffer) === nothing
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
    @test War1gusAI.start_event_logger!(path=restart_path, stdout_io=restart_stdout) === nothing
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

@testset "event logger reports unavailable file" begin
  War1gusAI.stop_event_logger!()
  mktempdir() do directory
    stdout_buffer = IOBuffer()
    try
      War1gusAI.start_event_logger!(path=directory, stdout_io=stdout_buffer)
      War1gusAI.log_event("survives_file_error")
      War1gusAI.stop_event_logger!()
      +
      lines = filter(!isempty, split(String(take!(stdout_buffer)), '\n'))
      @test length(lines) == 2
      @test all(is_json_line, lines)
      error_line = only(filter(line -> occursin("\"type\":\"logger_error\"", line), lines))
      @test occursin("\"operation\":\"open\"", error_line)
      @test occursin("\"path\":\"$(directory)\"", error_line)
      @test occursin("\"error\":\"", error_line)
      @test any(line -> occursin("\"type\":\"survives_file_error\"", line), lines)
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
    @test War1gusAI.start_event_logger!(path=path, stdout_io=stdout_io) === nothing
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
