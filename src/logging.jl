const EVENT_LOGGER_LOCK = ReentrantLock()
const EVENT_LOGGER_LIFECYCLE_LOCK = ReentrantLock()
const EVENT_LOGGER = Ref{Any}(nothing)
const EVENT_LOGGER_SHUTDOWN_TIMEOUT_SECONDS = 1.0

mutable struct EventLogger
  lock::ReentrantLock
  wakeup::Threads.Condition
  queue::Vector{String}
  active::Bool
  stopping::Bool
  path::String
  stdout_io::IO
  task::Union{Nothing,Task}
end

"""Return the JSON-lines log path, honoring an explicit environment override."""
function default_log_path()::String
  path = get(ENV, "WAR1GUS_AI_LOG_PATH", "")
  return isempty(path) ? joinpath(Sys.BINDIR, "War1gusAI.log") : path
end

@inline function _json_hex_digit(value::UInt32)::UInt8
  return value < 10 ? UInt8('0') + UInt8(value) : UInt8('a') + UInt8(value - 10)
end
function _json_unicode_escape!(output::IO, codepoint::UInt32)::Nothing
  write(output, "\\u")
  write(output, _json_hex_digit((codepoint >> 12) & 0x0f))
  write(output, _json_hex_digit((codepoint >> 8) & 0x0f))
  write(output, _json_hex_digit((codepoint >> 4) & 0x0f))
  write(output, _json_hex_digit(codepoint & 0x0f))
  return nothing
end


function _json_string!(output::IO, value::AbstractString)::Nothing
  write(output, UInt8('"'))
  for character in value
    if character == '"'
      write(output, "\\\"")
    elseif character == '\\'
      write(output, "\\\\")
    elseif character == '\b'
      write(output, "\\b")
    elseif character == '\f'
      write(output, "\\f")
    elseif character == '\n'
      write(output, "\\n")
    elseif character == '\r'
      write(output, "\\r")
    elseif character == '\t'
      write(output, "\\t")
    elseif !isvalid(character)
      write(output, "\\ufffd")
    else
      codepoint = UInt32(character)
      if codepoint < 0x20
        write(output, "\\u00")
        write(output, _json_hex_digit(codepoint >> 4))
        write(output, _json_hex_digit(codepoint & 0x0f))
      elseif codepoint <= 0x7f
        write(output, character)
      elseif codepoint <= 0xffff
        _json_unicode_escape!(output, codepoint)
      else
        surrogate = codepoint - 0x10000
        _json_unicode_escape!(output, 0xd800 + (surrogate >> 10))
        _json_unicode_escape!(output, 0xdc00 + (surrogate & 0x03ff))
      end
    end
  end
  write(output, UInt8('"'))
  return nothing
end

function _json_value!(output::IO, value, depth::Int=0)::Nothing
  depth > 64 && (write(output, "null"); return nothing)

  if value === nothing
    write(output, "null")
  elseif value isa Bool
    write(output, value ? "true" : "false")
  elseif value isa Integer
    write(output, string(value))
  elseif value isa AbstractFloat
    isfinite(value) ? write(output, string(value)) : write(output, "null")
  elseif value isa AbstractString
    _json_string!(output, value)
  elseif value isa Symbol
    _json_string!(output, String(value))
  elseif value isa NamedTuple
    write(output, UInt8('{'))
    first_field = true
    for (name, field_value) in pairs(value)
      first_field || write(output, UInt8(','))
      _json_string!(output, String(name))
      write(output, UInt8(':'))
      _json_value!(output, field_value, depth + 1)
      first_field = false
    end
    write(output, UInt8('}'))
  elseif value isa AbstractVector || value isa Tuple
    write(output, UInt8('['))
    first_item = true
    for item in value
      first_item || write(output, UInt8(','))
      _json_value!(output, item, depth + 1)
      first_item = false
    end
    write(output, UInt8(']'))
  else
    write(output, "null")
  end
  return nothing
end

function _event_line(type::AbstractString, timestamp::Float64, fields)::String
  output = IOBuffer()
  try
    write(output, "{\"type\":")
    _json_string!(output, type)
    write(output, ",\"timestamp\":")
    _json_value!(output, timestamp)
    for (name, value) in pairs(fields)
      (name === :type || name === :timestamp) && continue
      write(output, UInt8(','))
      _json_string!(output, String(name))
      write(output, UInt8(':'))
      _json_value!(output, value)
    end
    write(output, UInt8('}'))
    return String(take!(output))
  catch
    return "{\"type\":\"logger_error\",\"timestamp\":$(repr(timestamp))}"
  end
end

function _take_event_batch!(logger::EventLogger)::Union{Vector{String},Nothing}
  lock(logger.lock)
  try
    while isempty(logger.queue) && !logger.stopping
      wait(logger.wakeup)
    end
    isempty(logger.queue) && return nothing
    batch = logger.queue
    logger.queue = String[]
    return batch
  finally
    unlock(logger.lock)
  end
end
function _report_file_error!(logger::EventLogger, operation::AbstractString, error)::Nothing
  line = _event_line(
    "logger_error",
    time(),
    (operation=operation, path=logger.path, error=sprint(showerror, error)),
  )
  try
    write(logger.stdout_io, line)
    write(logger.stdout_io, UInt8('\n'))
    flush(logger.stdout_io)
  catch
  end
  return nothing
end


function _write_event_lines!(logger::EventLogger)::Nothing
  file_io = try
    open(logger.path, "a")
  catch error
    _report_file_error!(logger, "open", error)
    nothing
  end
  try
    while true
      batch = _take_event_batch!(logger)
      isnothing(batch) && break
      for line in batch
        if !isnothing(file_io)
          try
            write(file_io, line)
            write(file_io, UInt8('\n'))
          catch error
            _report_file_error!(logger, "write", error)
            try
              close(file_io)
            catch
            end
            file_io = nothing
          end
        end
        try
          write(logger.stdout_io, line)
          write(logger.stdout_io, UInt8('\n'))
        catch
        end
      end
      if !isnothing(file_io)
        try
          flush(file_io)
        catch error
          _report_file_error!(logger, "flush", error)
          try
            close(file_io)
          catch
          end
          file_io = nothing
        end
      end
      try
        flush(logger.stdout_io)
      catch
      end
    end
  finally
    if !isnothing(file_io)
      try
        flush(file_io)
      catch error
        _report_file_error!(logger, "flush", error)
      end
      try
        close(file_io)
      catch error
        _report_file_error!(logger, "close", error)
      end
    end
    try
      flush(logger.stdout_io)
    catch
    end
  end
  return nothing
end

function _stop_event_logger!(logger::EventLogger)::Nothing
  lock(logger.lock)
  try
    logger.active = false
    logger.stopping = true
    notify(logger.wakeup)
  finally
    unlock(logger.lock)
  end
  task = logger.task
  !isnothing(task) && timedwait(() -> istaskdone(task), EVENT_LOGGER_SHUTDOWN_TIMEOUT_SECONDS)
  return nothing
end

"""Start a background JSON-lines logger, replacing and draining any prior logger."""
function start_event_logger!(; path::AbstractString=default_log_path(), stdout_io::IO=stdout)::Nothing
  lock(EVENT_LOGGER_LIFECYCLE_LOCK)
  try
    lock(EVENT_LOGGER_LOCK)
    current = try
      EVENT_LOGGER[]
    finally
      EVENT_LOGGER[] = nothing
      unlock(EVENT_LOGGER_LOCK)
    end
    !isnothing(current) && _stop_event_logger!(current)

    logger_lock = ReentrantLock()
    logger = EventLogger(
      logger_lock,
      Threads.Condition(logger_lock),
      String[],
      true,
      false,
      String(path),
      stdout_io,
      nothing,
    )
    lock(EVENT_LOGGER_LOCK)
    try
      EVENT_LOGGER[] = logger
    finally
      unlock(EVENT_LOGGER_LOCK)
    end
    logger.task = Threads.@spawn _write_event_lines!(logger)
  finally
    unlock(EVENT_LOGGER_LIFECYCLE_LOCK)
  end
  return nothing
end

"""Queue one JSON event. Before startup and after shutdown this is a no-op."""
function log_event(type::AbstractString; kwargs...)::Nothing
  lock(EVENT_LOGGER_LOCK)
  logger = try
    EVENT_LOGGER[]
  finally
    unlock(EVENT_LOGGER_LOCK)
  end
  isnothing(logger) && return nothing
  lock(logger.lock)
  active = try
    logger.active
  finally
    unlock(logger.lock)
  end
  active || return nothing

  line = _event_line(type, time(), kwargs)
  lock(logger.lock)
  try
    logger.active || return nothing
    push!(logger.queue, line)
    notify(logger.wakeup)
  finally
    unlock(logger.lock)
  end
  return nothing
end

"""Drain queued events and stop the active background logger."""
function stop_event_logger!()::Nothing
  lock(EVENT_LOGGER_LIFECYCLE_LOCK)
  try
    lock(EVENT_LOGGER_LOCK)
    logger = try
      EVENT_LOGGER[]
    finally
      EVENT_LOGGER[] = nothing
      unlock(EVENT_LOGGER_LOCK)
    end
    !isnothing(logger) && _stop_event_logger!(logger)
  finally
    unlock(EVENT_LOGGER_LIFECYCLE_LOCK)
  end
  return nothing
end
