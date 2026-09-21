using Test

include(joinpath(@__DIR__, "..", "orchestrate.jl"))

function orchestration_args(mode; extras=String[])
 base = String[
  mode, "--launcher", "/tmp/war1gus-launcher", "--data-dir", "/tmp/war1gus-data",
  "--matches", "8", "--workers", mode == "train" ? "1" : "3",
  "--timeout-cycles", "12000", "--seed", "73", "--output", "/tmp/rollouts.jsonl",
  "--state-root", "/tmp/war1gus-state", "--rollout-config", "/tmp/rollout.lua",
 ]
 append!(base, extras)
 return base
end

@testset "orchestration options reject invalid mode and incomplete evaluation" begin
 @test_throws ArgumentError parse_options(String[])
 @test_throws ArgumentError parse_options(["play"])
 @test_throws ArgumentError parse_options(orchestration_args("train"))
 @test_throws ArgumentError parse_options(orchestration_args("train"; extras=["--map", "goldrush", "--workers", "2"]))
 @test_throws ArgumentError parse_options(orchestration_args("train"; extras=["--map", "goldrush", "--reset"]))
 @test_throws ArgumentError parse_options(orchestration_args("evaluate"; extras=["--held-out-map", "ice"]))
 @test_throws ArgumentError parse_options(orchestration_args("evaluate"; extras=["--held-out-map", "ice", "--league-snapshot", "/tmp/frozen-opponent"]))
 @test_throws ArgumentError parse_options(orchestration_args("evaluate"; extras=["--held-out-map", "ice", "--checkpoint", "/tmp/policy", "--reset"]))
end

@testset "schedules are deterministic and respect training and held-out boundaries" begin
 train = parse_options(orchestration_args("train"; extras=["--map", "goldrush", "--map", "forest"]))
 first_schedule = build_schedule(train)
 second_schedule = build_schedule(train)
 @test first_schedule == second_schedule
 @test length(unique(match.seed for match in first_schedule)) == length(first_schedule)
 @test all(match -> 1 <= match.seed <= typemax(Int32), first_schedule)
 @test Set(match.map for match in first_schedule) == Set(["goldrush", "forest"])
 @test length(unique(match.train_player for match in first_schedule)) > 1
 @test length(unique(match.port for match in first_schedule)) == length(first_schedule)

 evaluate = parse_options(orchestration_args("evaluate"; extras=["--map", "training-only", "--held-out-map", "ice", "--held-out-map", "island", "--checkpoint", "/tmp/policy"]))
 evaluation_schedule = build_schedule(evaluate)
 @test all(match -> match.map in ("ice", "island"), evaluation_schedule)
 @test length(unique(match.train_player for match in evaluation_schedule)) == 1
 @test all(match -> match.mode == "evaluate", evaluation_schedule)
end

@testset "evaluation aggregate reports outcomes and zero-safe efficiencies" begin
 metrics = aggregate_evaluation([
  Dict("map" => "ice", "outcome" => "win", "cycles" => 100, "units" => 4, "buildings" => 1, "total_units" => 7, "total_buildings" => 2, "kills" => 6, "razings" => 2, "gold" => 500, "wood" => 200),
  Dict("map" => "ice", "outcome" => "loss", "cycles" => 200, "units" => 2, "buildings" => 3, "total_units" => 4, "total_buildings" => 6, "kills" => 1, "razings" => 3, "gold" => 300, "wood" => 400),
  Dict("map" => "island", "outcome" => "timeout", "cycles" => 300),
 ])
 @test metrics["matches"] == 3
 @test (metrics["wins"], metrics["losses"], metrics["draws"], metrics["timeouts"]) == (1, 1, 0, 1)
 @test metrics["win_rate"] == 1 / 3
 @test metrics["mean_cycles"] == 200.0
 @test metrics["mean_elimination_time"] == 100.0
 @test metrics["destroyed_assets"] == 12.0
 @test metrics["own_loss_proxy"] == 9.0
 @test metrics["total_production"] == 19.0
 @test metrics["asset_efficiency"] == 4 / 3
 @test metrics["combat_efficiency"] == 7 / 9
 @test (metrics["total_gold"], metrics["total_wood"]) == (800.0, 600.0)
 @test metrics["per_map"]["ice"]["wins"] == 1
 @test aggregate_evaluation(Dict{String,Any}[])["combat_efficiency"] == 0.0
end

@testset "rollout terminal records are complete and unique" begin
 match = (; match_id="evaluate-000001", map="held-out", mode="evaluate", train_player=2, seed=31)
 terminal_line = json_line(
  "rollout_terminal";
  match_id=match.match_id,
  map=match.map,
  mode=match.mode,
  trainable_player=match.train_player,
  seed=match.seed,
  cycles=100,
  kills=3,
  razings=1,
  gold=500,
  wood=200,
  units=4,
  buildings=2,
  total_units=7,
  total_buildings=3,
  outcome="win",
 )
 large_diagnostic_line = json_line("training_sample"; payload=repeat("diagnostic", 10_000))
 fatal_line = json_line("error"; error="incompatible checkpoint")
 incomplete_line = json_line(
  "rollout_terminal";
  match_id=match.match_id,
  map=match.map,
  mode=match.mode,
  trainable_player=match.train_player,
  seed=match.seed,
  cycles=100,
  kills=3,
  razings=1,
  gold=500,
  wood=200,
  units=4,
  buildings=2,
  total_units=7,
  outcome="win",
 )
 write_log = function (path, lines)
  open(path, "w") do io
   for line in lines
    println(io, line)
   end
  end
 end
 mktempdir() do root
  paths = (; child_log=joinpath(root, "launcher.log"), ai_log=joinpath(root, "ai.jsonl"))
  write_log(paths.child_log, ["engine noise", large_diagnostic_line, terminal_line])
  write_log(paths.ai_log, [large_diagnostic_line, terminal_line])
  events = collect_result_events(paths)
  @test [event["type"] for event in events] == ["rollout_terminal"]
  @test require_rollout_terminal(events, match)["outcome"] == "win"
  @test_throws ArgumentError require_rollout_terminal(Dict{String,Any}[], match)

  write_log(paths.ai_log, [large_diagnostic_line, terminal_line, fatal_line])
  retained = collect_result_events(paths)
  @test [event["type"] for event in retained] == ["rollout_terminal", "error"]
  @test_throws ArgumentError require_rollout_terminal(retained, match)

  write_log(paths.ai_log, String[])
  write_log(paths.child_log, [terminal_line, terminal_line])
  @test_throws ArgumentError require_rollout_terminal(collect_result_events(paths), match)

  write_log(paths.child_log, [terminal_line])
  rm(paths.ai_log)
  @test [event["type"] for event in collect_result_events(paths)] == ["rollout_terminal"]

  write_log(paths.child_log, [incomplete_line])
  @test_throws ArgumentError require_rollout_terminal(collect_result_events(paths), match)
 end
end

@testset "child environment preserves runtime variables without loader overrides" begin
 withenv(
  "LD_LIBRARY_PATH" => "/nix/store/incompatible",
  "LD_PRELOAD" => "/nix/store/incompatible.so",
  "WAR1GUS_ORCHESTRATOR_TEST_KEEP" => "kept",
 ) do
  child = child_environment(Dict("WAR1GUS_ORCHESTRATOR_TEST_OVERRIDE" => "set"))
  @test !haskey(child, "LD_LIBRARY_PATH")
  @test !haskey(child, "LD_PRELOAD")
  @test child["WAR1GUS_ORCHESTRATOR_TEST_KEEP"] == "kept"
  @test child["WAR1GUS_ORCHESTRATOR_TEST_OVERRIDE"] == "set"
 end
end

@testset "match commands isolate state and preserve train versus evaluation contracts" begin
 train = parse_options(orchestration_args("train"; extras=["--map", "goldrush", "--checkpoint", "/tmp/checkpoint", "--league-snapshot", "/tmp/snapshot", "--reset"]))
 @test train.workers == 1
 schedule = build_schedule(train)
 first_command, first_environment, first_paths = build_match_command(train, schedule[1])
 next_command, next_environment, next_paths = build_match_command(train, schedule[2])
 @test first_command.exec[1] == train.launcher
 @test "--league-train" in first_command.exec
 @test !in("--reset-train", first_command.exec)
 @test "-b" in first_command.exec
 @test "--league-train" in next_command.exec
 @test !in("--reset-train", next_command.exec)
 @test "-b" in next_command.exec
 data_index = findfirst(==("-d"), first_command.exec)
 config_index = findfirst(==("-c"), first_command.exec)
 @test first_command.exec[data_index+1] == train.data_dir
 @test first_command.exec[config_index+1] == train.rollout_config
 @test first_environment["WAR1GUS_ROLLOUT_MAP"] == abspath(joinpath(train.data_dir, schedule[1].map))
 @test rollout_map_path(train, "/tmp/already-absolute.smp") == "/tmp/already-absolute.smp"
 @test Set(["WAR1GUS_ROLLOUT_MAP", "WAR1GUS_ROLLOUT_SEED", "WAR1GUS_ROLLOUT_TRAIN_PLAYER",
  "WAR1GUS_ROLLOUT_TIMEOUT_CYCLES", "WAR1GUS_ROLLOUT_MATCH_ID", "WAR1GUS_ROLLOUT_MODE",
  "WAR1GUS_AI_TRAIN_PLAYER", "WAR1GUS_AI_SEED", "WAR1GUS_AI_BINARY", "WAR1GUS_AI_CHECKPOINT",
  "WAR1GUS_AI_LEAGUE_DIR", "WAR1GUS_AI_SNAPSHOT", "WAR1GUS_AI_LOG_PATH", "XDG_STATE_HOME",
  "STRATAGUS_UNBUFFERED_STDIO", "WAR1GUS_AI_PORT"]) ⊆
       Set(keys(first_environment))
 @test first_environment["WAR1GUS_AI_SEED"] == string(schedule[1].seed)
 @test first_environment["WAR1GUS_AI_BINARY"] == joinpath(dirname(train.rollout_config), "build", "bin", "War1gusAI")
 @test first_environment["WAR1GUS_AI_CHECKPOINT"] == "/tmp/checkpoint"
 @test first_environment["WAR1GUS_AI_LEAGUE_DIR"] == "/tmp/league"
 @test first_environment["WAR1GUS_AI_LEAGUE_DIR"] == next_environment["WAR1GUS_AI_LEAGUE_DIR"]
 @test first_environment["WAR1GUS_AI_CHECKPOINT"] == next_environment["WAR1GUS_AI_CHECKPOINT"]
 @test first_environment["WAR1GUS_AI_SNAPSHOT"] == next_environment["WAR1GUS_AI_SNAPSHOT"]
 @test first_environment["XDG_STATE_HOME"] != next_environment["XDG_STATE_HOME"]
 @test first_environment["WAR1GUS_AI_LOG_PATH"] != next_environment["WAR1GUS_AI_LOG_PATH"]
 @test first_environment["WAR1GUS_AI_PORT"] != next_environment["WAR1GUS_AI_PORT"]
 @test first_environment["STRATAGUS_UNBUFFERED_STDIO"] == "1"
 @test first_paths.root != next_paths.root

 default_train = parse_options(orchestration_args("train"; extras=["--map", "goldrush"]))
 default_schedule = build_schedule(default_train)
 _, default_first_environment, _ = build_match_command(default_train, default_schedule[1])
 _, default_next_environment, _ = build_match_command(default_train, default_schedule[2])
 shared_checkpoint = default_first_environment["WAR1GUS_AI_CHECKPOINT"]
 @test !isempty(shared_checkpoint)
 @test shared_checkpoint == joinpath(default_train.state_root, "checkpoint.jls")
 @test shared_checkpoint == default_next_environment["WAR1GUS_AI_CHECKPOINT"]
 @test default_first_environment["WAR1GUS_AI_LEAGUE_DIR"] == joinpath(dirname(shared_checkpoint), "league")

 evaluate = parse_options(orchestration_args("evaluate"; extras=["--held-out-map", "ice", "--checkpoint", "/tmp/frozen-policy", "--league-snapshot", "/tmp/frozen-snapshot"]))
 evaluation_command, evaluation_environment, _ = build_match_command(evaluate, first(build_schedule(evaluate)))
 @test "--league-evaluate" in evaluation_command.exec
 @test !in("--league-train", evaluation_command.exec)
 @test !in("--reset-train", evaluation_command.exec)
 @test "-b" in evaluation_command.exec
 @test evaluation_environment["WAR1GUS_AI_READ_ONLY"] == "1"
 @test evaluation_environment["WAR1GUS_AI_CHECKPOINT"] == "/tmp/frozen-policy"
 @test evaluation_environment["WAR1GUS_AI_SNAPSHOT"] == "/tmp/frozen-snapshot"
 @test evaluation_environment["WAR1GUS_AI_LEAGUE_DIR"] == "/tmp/league"
end

@testset "reset removes only explicit training artifacts" begin
 mktempdir() do directory
  checkpoint = joinpath(directory, "configured", "policy.jls")
  league = joinpath(dirname(checkpoint), "league")
  retained_log = joinpath(directory, "matches", "preserved.jsonl")
  mkpath(league)
  mkpath(dirname(retained_log))
  write(checkpoint, "checkpoint")
  write(joinpath(league, "snapshot.jls"), "snapshot")
  write(retained_log, "log")

  train = parse_options(orchestration_args("train"; extras=["--map", "goldrush", "--checkpoint", checkpoint, "--reset"]))
  reset_training_state!(train)
  @test !ispath(checkpoint)
  @test !ispath(league)
  @test ispath(retained_log)
 end
end
