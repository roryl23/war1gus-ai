# War1gus AI

`war1gus-ai` is the experimental Julia/Flux opponent used by War1gus. This
repository is intended to be checked out at `scripts/ai/war1gus` in a War1gus
checkout. All commands below start from this AI-project directory, so generated
state remains owned by this submodule.

## Architecture

Stratagus starts the compiled application on demand and communicates with it on
localhost through its `AiProcessor*` TCP API. Protocol v3 carries a
variable-length observation containing economy totals, reward components, and
one record for every visible own, enemy, or resource entity. Lua supplies the
complete catalog of legal choices for that decision; a choice includes its
actor, optional target entity and map position, group/formation metadata,
cadence, and production context.

The Flux policy scores that producer-supplied catalog directly rather than
selecting from a fixed action list. The catalog covers waiting, gathering gold
or wood, legal building placement, per-structure training and research,
entity-targeted attacks, group movement, exploration, repair, formations, and
defence. Stratagus validates and executes the selected primitive command;
Julia never emits Lua source.

Normal games use deterministic inference from the saved policy. Training uses
stochastic trajectory PPO with a value head. A checkpoint includes model and
optimizer state plus protocol, observation shape, catalog, reward, policy, and
PPO compatibility metadata.

## Build and play

The integration is Linux-focused. Build and extract War1gus, then enter this
AI-project directory. From here, install the Julia dependencies, compile the AI
application, and derive the enclosing War1gus root:

```sh
cd "$(git rev-parse --show-toplevel)"
AI_ROOT=$PWD
WAR1GUS_ROOT=$(git rev-parse --show-superproject-working-tree)
test -n "$WAR1GUS_ROOT" || {
  printf '%s\n' 'war1gus-ai must be checked out as a War1gus submodule' >&2
  exit 1
}

julia --project="$AI_ROOT" -e 'using Pkg; Pkg.instantiate()'
bash "$AI_ROOT/build.sh"
```

To build the War1gus executable from this directory, run its root build entry
point in a subshell:

```sh
(cd "$WAR1GUS_ROOT" && bash build.sh War1gus)
```

Launch the game from its root so the Lua integration can resolve the submodule
binary:

```sh
(cd "$WAR1GUS_ROOT" && ./build/war1gus)
```

Select `war1gus-ai` for a computer player. Lua launches
`scripts/ai/war1gus/build/bin/War1gusAI` when it is available, then tries the
installed game-data copy. Set `WAR1GUS_AI_BINARY` to use a different executable.
The child process starts only when an AI player needs it and closes when the
game ends. `--train` enables online training; `--reset-train` deliberately
removes the selected checkpoint and its sibling league before training.

## Headless training

The rollout coordinator runs repeatable headless matches. It trains one mutable
policy seat, samples frozen opponents from a bounded snapshot league, and
randomizes map, seat, and seed schedules. Training must use `--workers 1`,
because its checkpoint and league are shared mutable state.

Set the War1gus data directory for your installation, then resume or start
training with the Forest observer map. This example intentionally omits
`--reset`: if `$PWD/ai-training/checkpoint.jls` already exists, training
resumes it and its `$PWD/ai-training/league/` snapshots.

```sh
WAR1GUS_DATA_DIR="$HOME/.local/share/stratagus/data.War1gus"

julia --project="$AI_ROOT" "$AI_ROOT/orchestrate.jl" train \
  --launcher "$WAR1GUS_ROOT/build/war1gus" \
  --data-dir "$WAR1GUS_DATA_DIR" \
  --rollout-config "$AI_ROOT/rollout.lua" \
  --map 'maps/Forest1AI-Observer(5).smp' \
  --matches 100 --workers 1 --timeout-cycles 18000 --seed 1 \
  --state-root "$PWD/ai-training" \
  --checkpoint "$PWD/ai-training/checkpoint.jls" \
  --output "$PWD/ai-training/train.jsonl"
```

Use `--reset` only when intentionally discarding the selected checkpoint and
league; it requires an explicit `--checkpoint` and removes that checkpoint and
its sibling `league/` directory.

## Evaluation

Evaluation is read-only. Use a map excluded from the training schedule and the
immutable training checkpoint. It reports per-map outcomes, win rate,
elimination time, production, resource totals, and combat/asset-efficiency
proxies as JSON Lines.

```sh
julia --project="$AI_ROOT" "$AI_ROOT/orchestrate.jl" evaluate \
  --launcher "$WAR1GUS_ROOT/build/war1gus" \
  --data-dir "$WAR1GUS_DATA_DIR" \
  --rollout-config "$AI_ROOT/rollout.lua" \
  --held-out-map 'maps/GoldRushAI-Max-Observer(5).smp' \
  --matches 20 --workers 4 --timeout-cycles 18000 --seed 2 \
  --state-root "$PWD/ai-evaluation" \
  --checkpoint "$PWD/ai-training/checkpoint.jls" \
  --output "$PWD/ai-evaluation/evaluate.jsonl"
```

Pass `--league-snapshot PATH` to pin every frozen opponent to one compatible
snapshot instead of sampling the checkpoint's sibling `league/` directory.

## Rewards and logs

Asset value is each non-wall unit or building's gold-plus-wood cost multiplied
by remaining-health fraction. Enemy asset loss produces positive progress,
own asset loss and elapsed-time buckets are negative, and victory or defeat
adds the terminal reward component. Lua logs these components; the AI logs
network requests and responses, selected candidates, training samples, PPO
updates, league assignments and snapshots, and episode finalization as JSON
records with a `type` field.

For coordinator runs, live files are under the state root:

- The coordinator stream is `$PWD/ai-training/train.jsonl` or
  `$PWD/ai-evaluation/evaluate.jsonl`.
- Each match has `$PWD/ai-training/matches/<match-id>/ai.jsonl` (or the
  equivalent evaluation path) for AI JSON events.
- The coordinator redirects **both child stdout and stderr** to that match's
  `launcher.log`; this includes engine output and the AI's stdout JSON events.

To follow logs for match directories that already exist, run:

```sh
tail -q -f "$PWD"/ai-training/matches/*/ai.jsonl \
  "$PWD"/ai-training/matches/*/launcher.log
```

The shell expands this glob before `tail` starts, so it sees only match
directories that exist when invoked; re-run it to include later matches. For a
direct, non-coordinated AI launch, set `WAR1GUS_AI_LOG_PATH` to choose the JSON
log file; otherwise the application chooses its default log location.
