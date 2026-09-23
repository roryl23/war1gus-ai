# War1gus AI

`war1gus-ai` is the experimental Julia/Flux opponent used by War1gus. This
repository is intended to be checked out at `scripts/ai/war1gus` in a War1gus
checkout. All commands below start from this AI-project directory, so generated
state remains owned by this submodule.

## Architecture

Stratagus starts the compiled application on demand and communicates with it on
localhost through its `AiProcessor*` TCP API. Protocol v3 carries a
variable-length observation containing economy totals, reward components, and
one record for every strategically relevant own, enemy, or resource entity.
Neutral roads are counted for construction limits but omitted from policy
observations. Lua supplies the complete catalog of legal choices for that
decision; a choice includes its actor, optional target entity and map position,
group/formation metadata, cadence, and production context.

The Flux policy scores that producer-supplied catalog directly rather than
selecting from a fixed action list. The catalog covers waiting, gathering gold
or wood, bounded legal construction for every race building, roads, and walls,
every trainable multiplayer unit, the complete base and rebalanced research
trees, researched auto-targeted and position-targeted spell use, entity-targeted attacks, group movement,
exploration, repair, formations, and defence. Stratagus validates the selected
commands as one bounded batch before execution; Julia never emits Lua source.

Runtime evaluation batches the entity and candidate encoders into reusable
per-player CPU buffers and projects the shared context once per request. PPO
differentiation retains the Flux forward path with the same Float32 weights and
feature encodings; checkpoints require no migration. Batched arithmetic can
differ from scalar evaluation by Float32 rounding. TCP frames are bulk-decoded,
and each selected index is sent in one four-byte write with `TCP_NODELAY`.

Normal games use deterministic inference from the saved policy. Training uses
stochastic on-policy PPO with a value head: rewarded trajectories improve the
policy that generated them. A single background worker trains each PPO batch on
an isolated policy-and-optimizer copy while request handlers continue selecting
actions with the last published inference policy. Collection pauses while that
update is in flight, so no partial fragment crosses policy generations or
becomes a stale backlog. When training completes, the worker atomically
publishes the new policy. Shutdown waits for submitted work to finish and saves
the final published state. A checkpoint includes model and optimizer state plus
protocol, observation shape, catalog, reward, policy, and PPO compatibility
metadata.

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

The full build and `build.sh War1gus` refresh source-owned Lua scripts and maps
in the extracted data directory after installation. Set `WAR1GUS_DATA_DIR` to
override the default `$XDG_DATA_HOME/stratagus/data.War1gus` (or
`~/.local/share/stratagus/data.War1gus`); builds skip this step if extraction
has not yet created `war1data`. Generated configuration, extracted Warcraft
assets, and training state are not replaced.

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

Normal play sends one nonblocking request per AI player while the game keeps
advancing. A late response is accepted only while its original candidate is
still legal in the current world; responses older than 499 cycles are discarded.
Only the multiplayer host runs the Julia policy. It queues each accepted
decision as an atomic network command batch; clients execute the same batch
without opening an AI connection. Replays execute recorded batches without
starting Julia. `--train`, `--reset-train`, `--league-train`, and
`--league-evaluate` retain synchronous decisions and terminal reward delivery.
For a command-line network lobby with two humans and one AI, pass `ai=1`
alongside `numplayers=2` in the server's `-G` options.

### Julia threads

The packaged `War1gusAI` launcher exports `JULIA_NUM_THREADS=2` when the
variable is unset, allowing network-facing inference and background PPO work
to make progress concurrently. An explicit environment value takes precedence;
for example:

```sh
JULIA_NUM_THREADS=4 "$AI_ROOT/build/bin/War1gusAI" --train
```

The packaged launcher and `run.sh` also default `OPENBLAS_NUM_THREADS=1` when
unset. The policy's small matrix operations otherwise incur unnecessary BLAS
thread-pool overhead. Set `OPENBLAS_NUM_THREADS` explicitly to override this
default; it is independent of the Julia threads used for request handling and
background PPO updates.

## Headless training

The rollout coordinator runs repeatable headless matches. It trains one mutable
policy seat, samples frozen opponents from a bounded snapshot league, and
randomizes map, seat, and seed schedules. Training must use `--workers 1`,
because its checkpoint and league are shared mutable state.

The coordinator also refreshes those files in `--data-dir` before validating
selected maps or resetting a checkpoint. Direct training and evaluation runs
therefore do not require a separate build solely to sync changed scripts or
maps. This step requires CMake and an extracted `war1data` directory.

Set the War1gus data directory for your installation, then start a fresh
coordinator run with the Forest observer map. The checkpoint and its sibling
league are persistent mutable training state; `mktemp` creates a fresh unique
`<run-id>` state and output directory beneath `ai-training/runs/` for each
coordinator invocation.

```sh
WAR1GUS_DATA_DIR="$HOME/.local/share/stratagus/data.War1gus"
mkdir -p "$AI_ROOT/ai-training/runs"
TRAIN_RUN_DIR="$(mktemp -d "$AI_ROOT/ai-training/runs/$(date -u +%Y%m%dT%H%M%S)-XXXXXX")"

julia --project="$AI_ROOT" "$AI_ROOT/orchestrate.jl" train \
  --launcher "$WAR1GUS_ROOT/build/war1gus" \
  --data-dir "$WAR1GUS_DATA_DIR" \
  --rollout-config "$AI_ROOT/rollout.lua" \
  --map 'maps/Forest1AI-Observer(3).smp' \
  --matches 100 --workers 1 --timeout-cycles 18000 --seed 1 \
  --state-root "$TRAIN_RUN_DIR" \
  --checkpoint "$AI_ROOT/ai-training/checkpoint.jls" \
  --output "$TRAIN_RUN_DIR/train.jsonl"
```

This example intentionally omits `--reset`. That continues the model,
optimizer, and checkpoint-sibling `league/` state from
`$AI_ROOT/ai-training`, but it does **not** resume an interrupted coordinator
schedule or its individual matches. A fresh run directory preserves every
prior coordinator stream and match log. Reusing a state root and output path
instead truncates its `train.jsonl` and launcher logs.

Use `--reset` only when intentionally discarding the selected checkpoint and
league; it requires an explicit `--checkpoint` and removes that checkpoint and
its sibling `league/` directory.

## Evaluation

Evaluation is read-only. Use a map excluded from the training schedule and the
immutable training checkpoint. It reports per-map outcomes, win rate,
elimination time, production, resource totals, and combat/asset-efficiency
proxies as JSON Lines. The example creates a fresh unique `<run-id>` directory
under `ai-evaluation/runs/` for its state and output.

```sh
mkdir -p "$AI_ROOT/ai-evaluation/runs"
EVALUATION_RUN_DIR="$(mktemp -d "$AI_ROOT/ai-evaluation/runs/$(date -u +%Y%m%dT%H%M%S)-XXXXXX")"

julia --project="$AI_ROOT" "$AI_ROOT/orchestrate.jl" evaluate \
  --launcher "$WAR1GUS_ROOT/build/war1gus" \
  --data-dir "$WAR1GUS_DATA_DIR" \
  --rollout-config "$AI_ROOT/rollout.lua" \
  --held-out-map 'maps/GoldRushAI-Max-Observer(5).smp' \
  --matches 20 --workers 4 --timeout-cycles 18000 --seed 2 \
  --state-root "$EVALUATION_RUN_DIR" \
  --checkpoint "$AI_ROOT/ai-training/checkpoint.jls" \
  --output "$EVALUATION_RUN_DIR/evaluate.jsonl"
```

Pass `--league-snapshot PATH` to pin every frozen opponent to one compatible
snapshot instead of sampling the checkpoint's sibling `league/` directory.

## Rewards and logs

Asset value is each non-wall unit or building's gold-plus-wood cost multiplied
by remaining-health fraction. Enemy asset loss produces positive progress,
own asset loss and elapsed-time buckets are negative, and victory or defeat
adds the terminal reward component. Compact logs report errors, server
lifecycle, trainer and league configuration, league assignments and snapshots,
episode finalization, and PPO worker scheduling/completion information,
including low-frequency queue and timing fields. Gradient work and periodic
checkpoint or league-snapshot file I/O run in that worker rather than on the
request-response path; gameplay continues with the most recently published
policy during an update. `WAR1GUS_AI_VERBOSE_LOG=1` also emits high-frequency
diagnostics: Julia `network_request`, `network_response`,
`reward_decomposition`, and complete `training_sample` events, plus Lua reward
and action records.

Full `training_sample` records are diagnostics, not a training-data format.
PPO trains from bounded in-memory trajectory batches; exactly one update may be
in flight, rather than a general producer-consumer trajectory queue. Checkpoints
persist PPO state, including the final worker-published policy during orderly
shutdown; JSONL logs are not replayed for training.

For coordinator runs, live files are under that invocation's run directory:

- The coordinator stream is
  `$AI_ROOT/ai-training/runs/<run-id>/train.jsonl` or
  `$AI_ROOT/ai-evaluation/runs/<run-id>/evaluate.jsonl`. It contains the
  schedule and match results, not copied child event streams.
- Each match has
  `$AI_ROOT/ai-training/runs/<run-id>/matches/<match-id>/ai.jsonl` (or the
  equivalent evaluation path) for AI JSON events.
- Each match's `launcher.log` receives child launcher output. The coordinator
  scans it for `rollout_terminal` records and that match's `ai.jsonl` for fatal
  `error` records required to validate the match; it never re-emits parsed child
  events as `child_stdout`.

To follow logs for training match directories that already exist, run:

```sh
tail -q -f "$AI_ROOT"/ai-training/runs/*/matches/*/ai.jsonl \
  "$AI_ROOT"/ai-training/runs/*/matches/*/launcher.log
```

The shell expands this glob before `tail` starts, so it sees only match
directories that exist when invoked; re-run it to include later matches. For a
direct, non-coordinated AI launch, set `WAR1GUS_AI_LOG_PATH` to choose the JSON
log file. When it is set, Julia writes events only to that file and does not
mirror them to stdout; otherwise the application chooses its default log
location.
