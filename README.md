# War1gus AI

`war1gus-ai` is the experimental Julia/Flux opponent used by War1gus. This
repository is intended to be checked out at `scripts/ai/war1gus` in a War1gus
checkout. All commands below start from this AI-project directory, so generated
state remains owned by this submodule.

## Architecture

Stratagus starts the compiled application on demand and communicates with it on
localhost through its `AiProcessor*` TCP API. Protocol v3 carries economy and
reward totals plus records for on-map units, buildings, resources, and neutral
roads. The server selects an owned actor, an order, and then any required target
entity or exact map coordinates in successive requests. Candidate pages keep
every actor, target, and coordinate reachable without truncating the catalog
at 512 choices per request.

The engine derives build, train, upgrade, research, and spell choices from its
registered producer and caster definitions. Lua also offers movement, combat,
resource, transport, stop, hold, and cancellation orders to every owned actor;
it does not mask choices by affordability, dependencies, supply, idle state,
construction counts, or road placement. Stratagus validates and executes the
selected order. A rejected publication incurs a bounded training penalty, so
the policy can learn placement requirements such as building beside a road
instead of following a scripted road-first rule. Intermediate actor, action,
and coordinate selections receive zero immediate reward and later PPO credit.
The engine suppresses native AI decision managers and unsolicited unit orders
for `war1gus-ai`; pathfinding and execution of explicit orders still run.

Runtime evaluation batches the entity and candidate encoders into reusable
per-player CPU buffers and projects the shared context once per request. PPO
packs trajectory features once per batch and evaluates all observations through
batched Flux layers. Fused entity-pooling/reference and categorical-reduction
derivatives avoid allocating a full-batch gradient for each observation.
Activations are recomputed after every optimizer update; GAE, clipping,
optimizer behavior, and the default four full-batch epochs are unchanged.
Weights and features remain Float32. Existing catalog-v4/reward-v3 checkpoints
load with a logged contract migration; the next save writes catalog-v5/reward-v4
metadata without resetting the learned weights or optimizer.
Batched arithmetic can differ from scalar evaluation by Float32 rounding.
TCP frames are bulk-decoded, and each selected index is sent in one four-byte
write with `TCP_NODELAY`.

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

Rebuild both Stratagus and War1gus from this directory. The coordinator requires
the current engine's `SetFastForwardCycle` binding and synchronous socket
readiness support; compiling Julia alone does not update either. These root
build stages install system-wide and may request `sudo`:

```sh
(cd "$WAR1GUS_ROOT" && bash build.sh Stratagus && bash build.sh War1gus)
```

For a local CMake build instead, configure War1gus with `STRATAGUS` pointing to
the rebuilt engine executable and `STRATAGUS_INCLUDE_DIR` pointing to
`"$WAR1GUS_ROOT/stratagus/gameheaders"`. An existing launcher can otherwise keep
using an older installed engine even after the local engine is rebuilt.

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

Normal play sends one nonblocking request per AI player at a time, advancing
the staged choice across responses while gameplay continues. A late response
is accepted only while its actor and target still exist; responses older than
499 cycles are discarded.
Only the multiplayer host runs the Julia policy. It queues each accepted
decision as an atomic network command batch; clients execute the same batch
without opening an AI connection. Replays execute recorded batches without
starting Julia. `--train`, `--reset-train`, `--league-train`, and
`--league-evaluate` retain synchronous decisions and terminal reward delivery.
The synchronous engine adapter waits for socket readiness instead of sleeping
between successful protocol stages; reconnect and terminal-delivery deadlines
remain bounded. This does not enable asynchronous decisions during training.
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

## Rollout training

The rollout coordinator runs repeatable unattended benchmark matches. By
default, training and evaluation render normally. Add `--fast-forward` to opt
into the engine's fast-forward display/event throttling after simulation starts:
this favors simulation throughput at the cost of visible frame rate and input
responsiveness. It still initializes SDL and is not a renderer-free headless
engine. Neither mode changes game type, diplomacy, random seeds, or simulation
steps.

Each match trains one mutable policy seat and samples frozen opponents from a
bounded snapshot league. Map and child-seed schedules are reproducible. Each
map rotates through its own shuffled computer-seat roster from the `.smp`
`DefinePlayerTypes` declaration, excluding explicit non-`war1gus-ai` assignments
in its `.sms` setup. Observer and absent seats are never scheduled. Unsupported
rosters and maps without eligible computer seats fail before checkpoint reset.
Training must use `--workers 1`, because its checkpoint and league are shared
mutable state.
When alternatives exist, training samples each AI seat from a 20% policy and
80% uniform non-wait mixture. `wait` remains selectable; every non-wait
candidate retains probability even under a strongly wait-biased checkpoint.
Only the mutable seat contributes PPO trajectories, whose likelihoods and
entropy use the same mixture. Inference and league evaluation remain greedy.

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

Both examples use normal rendering. To opt in, add `--fast-forward` immediately
after `train` or `evaluate` in the respective command (for example,
`orchestrate.jl train --fast-forward` with the remaining arguments unchanged).
This choice is independent of checkpoint state: an existing training run does
not need `--reset` to change rendering behavior.

Pass `--league-snapshot PATH` to pin every frozen opponent to one compatible
snapshot instead of sampling the checkpoint's sibling `league/` directory.

## Rewards and logs

Asset value is each non-wall unit or building's gold-plus-wood cost multiplied
by remaining-health fraction. Enemy asset loss produces positive progress,
own asset loss and elapsed-time buckets are negative, and victory or defeat
adds the terminal reward component. An engine-rejected published order adds
`-5` to the next actor-stage reward; expired responses do not incur this penalty.
Compact logs report errors, server
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
