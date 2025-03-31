module war1gus

using Base.Threads
using Format

function async_stdin_reader(
  c::Channel{String}  # stdin channel
)
  while true
    line = readline(stdin)
    put!(c, line)
    yield()
  end
end

@kwdef mutable struct Game
  state::String
  function Game(state::String)
    new(state)
  end
end

function process_position(
  tokens::Vector{SubString{String}}
)::Game
  tokens = ""  # default state
  return Game(state)
end

function process_go(
  tokens::Vector{SubString{String}},
  g::Game,
  c::Channel{Bool},  # control channel
  o::Channel{String}  # output channel
)::Task
  args = Dict()
  for i in 2:2:length(tokens)
    key = tokens[i]
    value = i + 1 <= length(tokens) ? tokens[i + 1] : nothing
    args[key] = value
  end
  return @spawn evaluator(
    Game(g),
    args,
    c,
    o,
  )
end

function process_setoption(
  tokens::Vector{SubString{String}},
  options::Dict,
)::Dict
  if length(tokens) > 2 && tokens[2] == "name"
    option_name = tokens[3]
    option_value = nothing
    if length(tokens) > 4 && tokens[4] == "value"
      option_value = join(tokens[5:end], " ")
    end
    options[option_name] = option_value
  else
    println("Invalid setoption command")
  end
  return options
end

function process_command(
  line::String,
  op::Dict,  # options
  g::Game,  # game state
  t::Task,  # engine task
  c::Channel{Bool},  # control channel
  o::Channel{String}  # output channel
)::Tuple{Game, Any}
  tokens = split(line)
  if length(tokens) == 0
    return tokens, nothing
  end
  cmd_type = tokens[1]

  if cmd_type == "position"
    return tokens, process_position(tokens)
  elseif cmd_type == "go"
    return tokens, process_go(tokens, g, c, o)
  elseif cmd_type == "setoption"
    return tokens, process_setoption(tokens, op)
  elseif cmd_type == "stratagus"
    println("id name War1gusAI 0.1.0")
    println("id author roryl23")
    println("stratagusok")
    return tokens, nothing
  elseif cmd_type == "isready"
    println("readyok")
    return tokens, nothing
  elseif cmd_type == "stop"
    put!(c, false)
    wait(t)
  elseif cmd_type == "stratagusnewgame"
    return tokens, nothing
  elseif cmd_type == "quit"
    put!(c, false)
    println("quitting as soon as possible...")
    exit(0)
  else
    println("Unknown Stratagus command: $cmd_type")
  end
  return tokens, nothing
end

function real_main()
  # initializations
  stdin_channel = Channel{String}(1)
  control_channel = Channel{Bool}(1)
  output_channel = Channel{String}(Inf)
  engine_task = Task(())

  options = Dict()
  game = Game()

  @spawn async_stdin_reader(stdin_channel)
  # main loop
  while true
    if isready(stdin_channel)
      line = take!(stdin_channel)
      command, result = process_command(
        line,
        options,
        game,
        engine_task,
        control_channel,
        output_channel,
      )
      if ==(command[1], "go") && typeof(result) == Task
        engine_task = result
      elseif ==(command[1], "position") && typeof(result) == Game
        game = result
      end
    end

    if isready(output_channel)
      println(take!(output_channel))
    end

    yield()
  end
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