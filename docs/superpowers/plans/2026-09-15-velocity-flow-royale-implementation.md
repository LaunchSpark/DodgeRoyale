# VelocityFlow for DodgeRoyale implementation plan

**Date:** 2026-09-15  
**Status:** Tasks 1-13 implemented on `velocity-flow-royale`; the manual
graphical check is the one item still outstanding. Ownership revised 2026-09-16; the task
checkboxes below are acceptance criteria, not an execution log.  
**Design:** [VelocityFlow for DodgeRoyale](../specs/2026-09-15-velocity-flow-royale-design.md)

## Outcome and boundaries

Deliver a deterministic headless arena, one shared observation encoder, a
native batch environment process, and a repository-local Python training package that
works through both the CLI and dashboard. Finish with an end-to-end PPO update,
checkpoint reload, reproducibility checks, and measured throughput.

All paths below are relative to the active DodgeRoyale checkout. Both languages
ship from the same repository and revision:

* **Rust:** `src/`, `tests/`, and `benches/` at the repository root.
* **Python:** `training/`, with the `dodge_royale` package and its own tests.
* **Shared golden fixtures:** `tests/fixtures/gym-v1/`, consumed by both sides.

DodgeAI stays independent and unchanged. Adapt only the needed VelocityFlow,
PPO session, telemetry, reward, and dashboard code into this repository, recording
source revision/path and preserving applicable notices. Do not import `dodge`,
install DodgeAI, use sibling-checkout paths or symlinks, or add a shared package.
Do not copy the PICO-8 runtime, cartridges, old checkpoints, or other architectures.

The observation remains an intentional **256 x 256 reference-pixel local
window**, 64 x 64 cells, seven channels, and nine velocity-conditioned paths.
Enemies outside that window are unobserved. Do not enlarge it or introduce a
global map in this implementation.

Execute one action per 60 Hz simulation step. Predict each candidate by holding
its direction for 24 frames by default, then releasing and coasting; prediction
does not advance the live arena. Retain the current acceleration, diagonal
normalization, collision rules, and wrapping. Do not implement instantaneous
velocity or copy the old cart's path offsets.

In-game inference/export, additional gameplay, observation compression, and a
long training campaign are outside this plan. No new Rust crates are needed.
Keep the existing native database/demo path working. Gym branches before native
database or Tokio initialization. Python dependencies remain optional for game
builds, and the trainer must install and run with DodgeAI absent.

## Execution order and checkpoints

| Stage | Tasks | Reviewable result |
|---|---|---|
| Shared simulation | 1-3 | Graphical movement and deterministic headless steps use the same systems |
| Rust environment | 4-7 | A real native process accepts batches and emits correct observations |
| Python training | 8-11 | VecEnv, policy, CLI, and dashboard work with the new layout |
| Acceptance | 12-13 | End-to-end checks, documented measurements, and aligned developer docs |

Task dependencies are explicit below. Keep changes in focused, buildable
checkpoints; record the single repository revision in benchmark results.
Run focused checks while implementing, and the complete gates at the end.
Do not mark a task complete based only on its new files existing.

## Task 1 - Extract player simulation and establish scheduling

**Dependencies:** none.  
**Rust files:** create `src/simulation.rs`; modify `src/lib.rs`,
`src/game/mod.rs`, `src/game/player.rs`, `src/game/enemy.rs`,
`src/game/camera.rs`, and `src/game/player_art.rs`.

- [ ] Add renderer-independent `Player`, `PlayerIntent`, `PlayerSet::Move`,
  `SimulationSet`, `SimulationPlugin`, and `spawn_player_body`. Supply the
  collider, velocity, target marker, intent, and transform without art assets.
- [ ] Move movement into `move_players`, calling `motion::advance_motion`.
  Preserve transform Z and the existing `Defeated` behavior. Keep intent as a
  direction, not a speed; diagonal normalization stays in `advance_motion`.
- [ ] Install enemy, population, reference-enemy, and seeded-RNG plugins once.
  Move RNG plugin ownership out of `game::build_app` if it becomes owned by
  `SimulationPlugin`. Inserting `GameSeed` alone does not seed the spawn queue:
  `SeededRngPlugin` must be installed and its `PreStartup` system must run.
- [ ] Retain the enemy plugin's existing internal set chain. Nest its sets and
  `PlayerSet::Move` in `SimulationSet`; order movement before Prepare. Configure
  world bounds and preserve the catalog's `BoundaryMode::Wrap` settings.
- [ ] Gate the simulation set on Playing in the graphical app. Explicitly
  order keyboard intent writing before movement. Replace camera/trail
  `.after(move_player)` dependencies with `.after(PlayerSet::Move)`.
- [ ] Leave decoration, menu transitions, visual cleanup, and queue clearing
  in the graphical adapters. Ensure death handling is not installed twice.

**Acceptance:** headless system tests prove same-update intent consumption and
defeated-player immobility. Existing enemy tests pass. Both feature sets
compile; a graphical smoke check covers movement, camera/trails, seam crossing,
death, menu return, and starting another game.

## Task 2 - Implement arena lifecycle, stepping, snapshots, and counters

**Dependencies:** 1.  
**Rust files:** extend `src/simulation.rs`; add
`src/simulation/tests.rs`; modify `src/enemy_population.rs` and `src/enemy.rs`
only where shared initialization or collision-death reporting requires it.

- [ ] Define validated `ArenaConfig`, `ArenaView`, player/enemy/blast view
  records, and a step result containing frame, hit, death count, terminated,
  and truncated. Include an initialization-pass bound with a documented
  default. Reject zero max frames; allow zero enemies for controlled tests.
- [ ] Keep each `HeadlessArena` as an owned `App` with `MinimalPlugins`, manual
  time, no run loop, renderer, global logger, or terminal signal plugin. Finish
  plugin setup once and keep each worker arena's schedules single-threaded.
  Do not create an executor per frame or per entity.
- [ ] Initialize seed and player before placement. Reuse the existing
  replenishment system in a small initialization schedule, including deferred
  spawn application. Do not duplicate the placement algorithm. Prime normal
  app startup/time with simulation disabled, then run bounded spawn-only
  passes. No player/enemy movement, blast aging, contacts, or episode counters
  advance during initialization. Frame zero has a full population or returns
  an error; it never silently returns a partially populated episode.
- [ ] Use one shared timestep constant for manual time and prediction. Treat
  frame count as authoritative; test time using the representable Duration
  chosen for 1/60 second, not an impossible exact rational nanosecond value.
- [ ] Advance exactly one update per `step`, observe hits after Effects, and
  snapshot after Cleanup/Replenish and deferred commands. Latch termination
  on the controlled player's hit; truncate on reaching the frame budget.
  Both flags may be true on the same frame. Reject subsequent steps until
  reset rather than silently advancing a completed episode.
- [ ] Count unique enemies killed by enemy/enemy collisions at their existing
  death marking point, before cleanup; exclude player-contact deaths. Clear
  per-step counters before each step. Do not infer deaths from population size.
- [ ] Store snapshots in world units. Include entity identity for deterministic
  cell tie-breaking, complete colliders (offset/enabled included), blast age,
  and access to the configured blast duration for phase normalization.
- [ ] `reset(seed)` creates a fresh app and clears counters, messages, queued
  spawns, hazards, and prior termination state. Internal test helpers can
  place actors directly without exposing mutable world access in the public
  production interface.

**Acceptance:** test first-step movement/time, frame-zero population, impossible
placement, reset equivalence, direct-contact and blast death, truncation,
simultaneous death/timeout, post-completion rejection, and collision clusters
counted once. Compare same-seed/action snapshots throughout a deterministic
run and after reset. Use a controlled nonterminal scenario for a full 600-step
test, and a separate seeded-spawn test to prove different seeds change state.
Do not use a hash containing only the seed as evidence of simulation variance.

## Task 3 - Add action mapping and pure path prediction

**Dependencies:** 2.  
**Rust files:** extend `src/simulation.rs` and its unit tests; reuse
`src/motion.rs`, `src/torus.rs`, and `src/scale.rs`.

- [ ] Define action order exactly once: idle, left, right, up, down, up-left,
  up-right, down-left, down-right. Validate action bytes before stepping.
- [ ] Implement `predict_path` by repeatedly calling `advance_motion` on copied
  position/velocity at the shared timestep. Hold input for `hold_frames`, then
  use zero input. Sample after frames 4, 12, 24, 48, 72, and 108.
- [ ] Permit hold zero (immediate coasting) and holds beyond the final horizon;
  cap computational work at the largest horizon, not at the hold duration.
- [ ] Preserve world wrapping and input normalization. Invalid non-finite
  public inputs fail explicitly rather than entering observation tensors.

**Acceptance:** compare all nine paths against a real enemy-free arena from
rest and moving starts. Cover idle braking, reversal, diagonals, both wrap
axes, and hold values 0, 1, 24, and 108. Prediction must leave live state and
RNG unchanged.

## Task 4 - Implement the shared observation encoder

**Dependencies:** 2-3.  
**Rust files:** create `src/observation.rs` and
`src/observation/tests.rs`; export through `src/lib.rs`.

- [ ] Define a serializable layout description with protocol/layout version,
  dtype, section offsets/lengths, channel names/semantics, units, axis direction,
  grid dimensions, action directions, horizons, hold duration, and sample
  normalization. Derive lengths from constants and assert 28,782 values.
- [ ] Convert world displacements using `wrapped_delta / scale::PIXEL` and
  negate Y. Convert velocity using `world_units_per_second / (6.25 * 60 * 4)`
  with the same Y flip; clamp velocity components to [-1, 1].
- [ ] Rasterize enabled, valid collider rectangles using their world offsets,
  nearest periodic image, and positive-area cell overlap. Clip footprint
  coverage to the window; do not clamp a completely outside hazard onto an
  edge cell. The grid is intentionally local.
- [ ] Resolve each cell's hazard owner by type priority, collider area, then
  entity bits. Write the owner's occupancy, velocity, and blast phase together.
  Empty/non-blast phase is zero; blast velocity is zero. Player footprint is
  independent of hazard ownership. Clamp phase for float boundary noise.
- [ ] Encode path samples as player-relative, Y-flipped pixel displacements
  divided by 128, not absolute window coordinates. Do not clamp path values.
  The Python sampler adds the window center exactly once.
- [ ] Compute a conservative maximum path-distance bound including initial
  momentum, sustained input, and coasting. Warn on stderr when configured
  paths may exceed the window. Do not change hold duration or window size.
- [ ] Fill a caller-owned buffer to avoid avoidable per-cell allocations.

**Acceptance:** test unit conversions, bounds, offsets, corner/edge coverage,
disabled hazards, same-type ties, blast phase, seam-straddling hazards, and
path layout. Check Y with a stationary player. For a moving player, test
relative displacement using integrated actor displacements; quantized cell
indices need not move every frame, and end-of-frame velocity is not average
frame velocity. Keep golden numeric observations for the Python decoder.

## Task 5 - Freeze and implement protocol version 1

**Dependencies:** 4.  
**Rust files:** create native-only `src/gym/mod.rs` and
`src/gym/protocol.rs`; add a protocol document at
`docs/superpowers/specs/velocity-flow-royale-protocol-v1.md`.

- [ ] Write the byte schema before the Python client: eight-byte magic,
  version, tagged messages with explicit payload lengths, fixed-width
  little-endian integers/f32, and length-prefixed UTF-8 strings. Never emit
  Rust struct memory, `usize`, or implicit alignment padding. Use existing
  std/Serde facilities; no new transport crate.
- [ ] Specify handshake layout fields from Task 4, including action ordering
  and every section offset. Make incompatible versions/layouts fail before
  constructing a policy.
- [ ] Define STEP request count, action validation, response ordering, and
  terminal-observation indexing. Each env record contains the transition's
  frame count, flags, enemy-death count, old episode seed, and optional reset
  seed. Return post-reset observations in the main batch and pre-reset final
  observations in the terminal section. Exactly one step contributes reward.
- [ ] Define RESET responses: initial observations plus reset metadata, no
  fabricated transition reward. Explicit RESET(seed) restarts episode indices
  at zero; support an unseeded reset variant that advances episode indices
  without restarting the root seed. Specify this with an explicit presence
  flag so seed zero remains a valid seed. Auto-reset advances only that env.
- [ ] Specify `mix(root_seed, env_index, episode_index)` with domain separation
  and fixed test vectors; avoid language-default hashes. Rust owns episode
  seed selection; any Python seed preview uses the same published function.
- [ ] Define CLOSE acknowledgement, EOF shutdown, and a bounded ERROR record
  containing a code and message. Reject invalid sizes/actions and truncated
  input rather than partially stepping a batch. A failed reset/step ends the
  session explicitly; do not return a mixed partial-success batch.
- [ ] Implement codecs against `Read`/`Write` for in-memory tests. Document
  checked payload-size limits derived from the configured env count.

**Acceptance:** known byte fixtures, short reads/writes, truncation, invalid
lengths/opcodes/actions, seed zero, reset stream semantics, and strict response
lengths. Include episode frame count in the fixture so dashboard metadata is
part of the protocol contract.

## Task 6 - Add persistent arena workers and batch coordination

**Dependencies:** 2, 4-5.  
**Rust files:** create `src/gym/workers.rs`; extend `src/gym/mod.rs`.

- [ ] Start a bounded number of named std threads (default two, explicit
  `--threads` override). Each creates, updates, resets, and drops its own apps.
  Never move an `App` across threads or use unsafe trait implementations.
- [ ] Give each worker a stable shard of env indices. Use bounded channels
  with one in-flight batch per worker; send only owned actions/config/results.
  Gather by env index, never by completion order.
- [ ] Keep each arena's schedules single-threaded; account for Bevy's shared
  task-pool initialization separately so `--threads` does not multiply pools
  per arena. Verify multiple apps can be created/reset concurrently.
- [ ] Auto-reset after copying final observation and episode metadata. Preserve
  per-env seed counters across other envs finishing and across worker counts.
- [ ] Propagate initialization errors and channel disconnection to the
  coordinator. On CLOSE, EOF, or failure, stop issuing work, unblock pending
  sends/receives, drop arenas on their owner thread, and join workers. Avoid
  joining a worker that is blocked sending into an undrained result channel.

**Acceptance:** compare output bytes with one versus two workers, including
asymmetric episode lengths and auto-resets. Test initialization failure, worker
disconnection, repeated resets, and shutdown while a bounded batch is pending.
No busy waiting or per-step thread creation.

## Task 7 - Wire the native CLI and test the real process

**Dependencies:** 5-6.  
**Rust files:** modify `src/native.rs`, `src/main.rs`, and gym modules;
create `tests/gym_protocol.rs`; update `run.sh` if exposing a helper command.

- [ ] Add the `gym` subcommand with env count, root seed, enemies, max frames,
  hold frames, and worker count. Dispatch before inspecting DATABASE_URL,
  building Tokio, or running startup demos. Keep normal launch/seed arguments
  and the existing smoke-test behavior compatible.
- [ ] Validate env/worker counts and frame limits before allocating arenas.
  Use a locked binary stdin reader/stdout BufWriter and flush every response.
  Emit diagnostics only to stderr; do not install LogPlugin per arena.
- [ ] Serve handshake and request loop using the shared coordinator. Ensure
  errors have a nonzero exit and EOF/CLOSE have a clean shutdown.
- [ ] Carry forward the earlier idle-player headless entry point: use a finite
  HeadlessArena episode and report seed, elapsed simulation frames/time, and
  ending reason. Keep gym's binary stdout separate from human-readable output
  and retain the existing native startup contract for ordinary launch modes.
- [ ] Run subprocess tests with timeouts, pipe readers that drain concurrently,
  and guaranteed child cleanup. Cover RESET, STEP, terminal observation,
  auto-reset seed, CLOSE, EOF, and malformed requests. Set an invalid configured
  database URL in the gym test to prove the command bypasses database startup.

**Acceptance:** native subprocess tests pass in both feature configurations;
protocol stdout contains only valid records. Restore the graphics development
binary after headless commands as the existing task runner requires.

## Task 8 - Build the Python protocol client

**Dependencies:** 5 and 7 for live acceptance.  
**Files:** create `training/pyproject.toml`, a dependency lock,
`training/dodge_royale/__init__.py`, `training/dodge_royale/protocol.py`, and
`training/tests/test_protocol.py`; extend `training/README.md`. Put shared
fixtures under root `tests/fixtures/gym-v1/` and their Rust emitter in `examples/`.

- [ ] Package the trainer independently under `training/`, with declared Python
  support and dependencies for NumPy, Gymnasium, SB3, PyTorch, the dashboard,
  and tests. Record reproducible dependency versions and installation commands;
  no dependency on DodgeAI, Lua, or a cartridge. Use a local virtual environment.
- [ ] Emit committed protocol-v1 `.bin` fixtures plus a JSON manifest containing
  seed/config, expected headers/shapes, and selected numeric values with offsets.
  Cover asymmetric positions, moving hazards, blast phase, seam wrapping, and
  auto-reset with terminal observations. Keep handwritten wire-image tests as
  independent checks. Python consumes fixtures without building or running Rust.
- [ ] Add a versioned `Layout` dataclass and strict validation of offsets,
  sizes, semantics, action/horizon counts, dtype, and total observation length.
  Store a plain serializable representation for policy/checkpoint kwargs.
- [ ] Locate the binary through DODGE_ROYALE_BIN or this checkout's
  `target/release/` path resolved from the module location, with platform suffix.
  An installation outside the checkout requires an explicit binary path; document
  the override for custom Cargo target directories as well.
  Launch using an argument list, binary pipes, and no shell.
- [ ] Implement exact-length reads/writes, clean EOF errors, and response
  decoding from reusable scratch buffers. Returned observations, terminal
  arrays, and retained metadata must own their data.
- [ ] Drain stderr on a managed reader thread into a bounded diagnostic ring.
  Include its tail in startup/protocol failures. Close is idempotent; close
  stdin/handles, await graceful exit, then terminate/kill on timeout and reap
  the child. Interrupt pending reads so dashboard shutdown cannot hang.
- [ ] Consume the Rust fixtures without duplicating handwritten expectations
  for layout offsets. Add mocked partial-read, EOF, and buffer-alias tests.

**Acceptance:** mock client tests run with no binary; live fixture/handshake
tests skip only when no binary is configured. If DODGE_ROYALE_BIN is explicitly
set but broken, fail rather than silently skipping.

## Task 9 - Implement RoyaleVecEnv and reward/telemetry contracts

**Dependencies:** 8.  
**Python files:** create `training/dodge_royale/vec_env.py`,
`training/dodge_royale/rewards.py`, `training/dodge_royale/telemetry.py`, and
`training/tests/test_vec_env.py`; adapt needed reward/telemetry logic locally.

- [ ] Implement reset, deferred seed application, step_async/step_wait, close,
  reset_infos, get_attr/set_attr/env_method, and env_is_wrapped. Support indices
  correctly; get_attr(render_mode) returns None per requested env. Unsupported
  attributes raise AttributeError; unsupported mutation/method calls fail
  clearly. Do not wrap this already batched env in SubprocVecEnv.
- [ ] Expose float32 Box bounds from Layout and Discrete(9). Track pending
  operations; validate the entire action batch before serializing uint8 values.
  RESET(seed) reproduces frame zero; unseeded resets continue the seed stream.
- [ ] Convert flags into dones; set TimeLimit.truncated to truncated and not
  terminated. Put final observations in infos and new-episode seeds in
  reset_infos. Do not attach stale terminal observations to live transitions.
- [ ] Compute survival reward only on a surviving step; apply death penalty
  once on termination and the configured enemy-death reward for that step.
  Truncation alone is not a death. Auto-reset initialization earns no reward.
- [ ] Populate `frames` and `run_frames` from the transition episode, and
  `run_over` on completion. Provide `training_events` with survival_frames,
  deaths, lives_spent=0, and enemies_destroyed. Use per-step event counts, not
  cumulative totals. Track episode return for standard episode summary infos.
- [ ] Expose only applicable Royale reward controls. Store reward configuration
  and training artifacts locally, without reading DodgeAI's user settings.

**Acceptance:** test rewards, same-frame death/timeout, metadata before and
after reset, seed replay, indices, and retained arrays using a fake client.
Feed a synthetic 60-frame episode into the local DashboardCallback and
assert one second of survival plus correct kills/deaths. Test timeout value
bootstrapping uses the terminal observation, not the reset observation.

## Task 10 - Implement the Royale extractor and checkpoint registration

**Dependencies:** 4, 8-9.  
**Python files:** create `training/dodge_royale/velocity.py`,
`training/dodge_royale/policies.py`, and `training/tests/test_velocity.py`.

- [ ] Decode player values, seven 64 x 64 channels, and [9,6,2] paths from
  Layout. Keep the stem at full resolution: Conv 7->32, average pool to 32 x 32,
  body 32->64->64 with dilation 2->32, then bilinear upsample to 64 x 64.
  Adapt v2 activation/padding choices into the local implementation.
- [ ] Implement the 4 x 4 context branch, 48->6 danger head, and critic's
  32-channel mean plus 128 summary values plus 2 player/6 senses (168 inputs).
  Use geometric cell centers, relative hazard/player velocity, and a specified
  finite no-threat sentinel. Handle zero-distance threats without NaNs.
- [ ] Decode `points = (128,128) + paths*128`, then normalize to grid_sample
  coordinates with align_corners=False and border padding. This makes the
  stored relative path coordinates equal to the normalized sample coordinates
  for this layout; test that equivalence to catch double offsets/Y flips.
- [ ] Sample each horizon's own field slice, weight by normalized
  `0.985 ** horizon`, negate danger, and append 64 critic features. Return 73
  features. Port VelocityFlowPolicy/_FieldLogits locally and use pi=[]/vf=[64].
- [ ] Register `velocity-flow-royale` in the local policy module. Pass Layout
  through features_extractor_kwargs. Validate full checkpoint/handshake
  compatibility before model construction. New checkpoints use stable
  `dodge_royale` module paths. Reject foreign checkpoints; do not port PICO-8
  checkpoint migration or require DodgeAI imports to load Royale checkpoints.

**Acceptance:** test exact controller sampling/action ranking with synthetic
fields, horizon pairing, no-threat/overlap cases, finite logits/gradients, and
73-feature shape. Check different initial velocities produce the encoded
different paths. Save/reload preserves outputs and layout; incompatible layouts
with identical obs_len fail. Port relevant numerical policy tests locally.

## Task 11 - Integrate CLI, session lifecycle, and dashboard

**Dependencies:** 9-10.  
**Python files:** create `training/dodge_royale/train.py`,
`training/dodge_royale/dashboard.py`, `training/dodge_royale/training.py`, and
local session/GUI tests including `training/tests/test_training.py`.

- [x] Provide `python -m dodge_royale.train` and
  `python -m dodge_royale.dashboard`. Both run Royale; no `--game` switch or
  PICO-8 environment factory. Default to 8 envs, 1024 rollout
  steps, 100 enemies, 3600 max frames, hold 24, and two Rust worker threads.
  Expose overrides in the CLI; do not reinterpret action-repeat as hold frames.
- [x] Centralize env construction and use RoyaleVecEnv directly in
  both entry points. Route --check-env to a VecEnv smoke check for Royale,
  rather than passing it to Gymnasium's scalar-env checker.
- [x] Register GPU architecture selection and per-architecture hyperparameters:
  gamma=0.99**0.25, gae_lambda=0.95**0.25, with the remaining starting settings
  inherited explicitly from v2. Apply them on new models and checkpoint loads,
  using one local configuration for both entry points.
- [x] Bound Royale minibatches independently of total rollout samples. Start
  with a cap of 128, allow an explicit override, and choose a valid size for
  small smoke rollouts. Record the measured memory limit; a cap is not a
  guarantee against OOM on every device.
- [x] Thread the live Layout through new-model creation. On reload, compare
  the stored Layout before attaching the env. Changing enemy count/max frames
  can restart envs without changing model shape; changing hold duration must
  follow the spec's compatibility check, not silently reinterpret a checkpoint.
- [x] Make Game Config show only its supported knobs, disable Watch
  Agent with the stated autopilot explanation, and use the frame-count infos
  from Task 9 for the local survival charts. Audit field-overlay/diagnostic
  paths for hard-coded PICO-8 observation sizes before enabling them.
- [x] Route staged config, stop/restart, initialization failure, and checkpoint
  failure through cleanup that closes the Rust process. Port checkpoint save
  behavior and relevant lifecycle tests without cart migration machinery.

**Status:** complete. `training.py` and `train.py` landed in f1edaa8; `metrics.py`, `worker.py` and `dashboard.py` here. Verified against the real binary: `--dry-run`, `--check-env`, a full train-and-save, `--resume-latest`, the notebook in script mode, and `marimo run` serving HTTP 200 with no gym started until Start is pressed.

**Acceptance:** mocked CLI/session/GUI tests cover Royale defaults, invalid
pairings, initial creation, reload, config restart, and child cleanup. Verify
lambda/gamma on both creation and load. Install, import, train, and launch the
dashboard from this repository with DodgeAI absent from paths and dependencies.

## Task 12 - Run cross-language acceptance and final regression gates

**Dependencies:** 7-11.  
**Files:** create `training/tests/test_integration.py`; extend Rust
subprocess tests and numeric fixtures as needed.

- [x] Build the native release binary and point DODGE_ROYALE_BIN to its exact
  absolute path. Test actual handshake, RESET, STEP, CLOSE, and error paths.
- [x] Replay seed/action fixtures across processes and worker counts. Compare
  exact observations/metadata on the same build/platform. Do not promise
  cross-platform floating-point bit identity or full graphical/headless entity
  allocation identity.
- [x] Run one real PPO update with two envs, n_steps=16, a compatible small
  minibatch, and one epoch. Assert finite metrics and at least one trainable
  parameter changes. Save, reload, and perform another prediction/step.
- [x] Test death and timeout terminal observations, reset seeds, frozen retained
  arrays, dashboard duration, and ordinary checkpoint compatibility. Ensure
  external-process tests always have deadlines and clean up on assertion failure.
- [x] Check both Rust feature configurations and the WASM library build. Manually
  verify graphical controls and screen lifecycle after the extraction.

**Status:** complete except the manual graphical check, which is recorded as unverified below.

Run the required Rust gates from the Rust repository:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo clippy --locked --all-targets --no-default-features -- -D warnings
cargo test --locked --no-default-features
cargo run --locked --no-default-features -- --smoke-test
docker compose --profile tools config --quiet
cargo check --locked --lib --target wasm32-unknown-unknown --no-default-features
cargo build --locked --release --no-default-features
```

Install the pinned toolchain's wasm target if absent. Preserve the existing CI
database check; no new persistence/container behavior is introduced here.
Restore the default-feature development binary with `cargo build --locked`
after the headless gates, or use the existing run.sh restoration path.

From the repository root, after installing the trainer into its own environment
(PowerShell example; resolve the binary in the active checkout):

```powershell
$env:DODGE_ROYALE_BIN = (Resolve-Path target/release/dodge-royale.exe).Path
python -m pytest training/tests -q
```

CI builds the gym and runs Python against the same checkout/revision, with an
explicit binary path. Unit-only CI may skip live integration, but the final
acceptance run must not. Verify a clean checkout can run without DodgeAI present.

## Task 13 - Measure performance and finish developer documentation

**Dependencies:** 12.  
**Status:** complete. Results in [benchmark results](../results/2026-09-17-benchmark-royale.md).  
**Files:** add `training/tools/benchmark_royale.py` and a benchmark-results
document alongside this plan; update root and training READMEs, `AGENTS.md`, and
relevant task-runner/CI commands.

- [x] Benchmark release simulation, encoding, transport/decoding, policy
  inference, and PPO optimization separately at 8 and 64 envs. Add opt-in
  native timing summaries on stderr for simulation versus encoding; avoid
  per-step logging. Report env-steps/second separately from batch round trips.
- [x] Record hardware, repository revision, dependency versions, seed suite,
  enemy count, holds, worker count, rollout/minibatch sizes, precision/device,
  wall time, and peak host/device memory. Synchronize accelerator work when
  timing it. Use short rollouts for the 64-env pipeline benchmark until its
  memory budget is measured; do not launch an accidental 7 GiB rollout.
- [x] Evaluate idle and seeded-random policies over the same bounded seed suite.
  Report survival frames and termination/truncation rates. Treat maximum-length
  timeouts as censored survival, not proof of learned skill. These are baseline
  checks, not a long policy-training run.
- [x] Document launch commands for both CLI and GUI, the intentional local
  observation, seed/reset semantics, checkpoint layout checks, reward knobs,
  and diagnostics. Record persistent worker ownership and single-threaded arena
  schedules in AGENTS.md, distinguishing this path from existing Tokio/Rayon
  startup behavior.
- [x] Keep compression and further batching as measured follow-up work. If a
  benchmark fails a memory/throughput target, report the result and recommend
  a specific next experiment; do not silently change the observation contract.

## Completion checklist

- [ ] Shared movement and scheduling preserve playable graphical behavior.
      **Not verified.** Needs a human at a window; automated coverage
      reaches the headless simulation only.
- [x] Frame-zero initialization is seeded, bounded, and contains no hidden play.
- [x] The first public step advances one simulation frame; completed episodes
  remain frozen until reset.
- [x] Rust observations and Python sampling agree numerically across seams,
  Y orientation, units, cell centers, and horizon ordering.
- [x] Worker count changes do not change seeded output; failures and shutdown
  do not strand workers or child processes.
- [x] Auto-reset preserves terminal observation, frame count, flags, rewards,
  and old/new seeds correctly.
- [x] Training installs and runs with no DodgeAI checkout or package available;
  all fixtures, tests, entry points, dependencies, and artifacts are local.
- [x] Native database smoke paths remain compatible; gym has no database/runtime
  startup dependency.
- [x] A real PPO update and save/reload pass; dashboard survival reports seconds.
- [x] Required checks and benchmark results are recorded with actual outcomes,
  including any unavailable tool or platform limitation.
