# VelocityFlow for DodgeRoyale — design

**Status:** approved design, ready for an implementation plan.
**Date:** 2026-09-15.
**Repositories:** `DodgeRoyale` (Rust/Bevy, simulation and encoder) and
`DodgeAI` (Python/SB3, training stack).

## 1. Goal

Train a VelocityFlow v2 style policy to play DodgeRoyale's player, using
DodgeAI's existing PPO stack, and later run it inside the game on native and
web builds.

The architecture is ported. The trained weights are not: arena scale, hazards
and the movement model all differ, so a DodgeRoyale policy is trained from
scratch.

### Deliverables in this spec

1. A headless, deterministic DodgeRoyale simulation.
2. A Rust observation encoder and a `gym` protocol the Python trainer drives.
3. A DodgeAI vectorised environment, a `velocity-flow-royale` architecture, and
   dashboard wiring.

### Out of scope

* **In-game autopilot** (weight export, a Rust forward pass, a toggle key,
  parity tests against PyTorch). Its own spec, written after a policy trains.
* Difficulty, powerups, pickups, score and pattern hazards. DodgeRoyale does
  not have them; their observation channels are not reserved either, because
  training starts fresh.
* Multi-agent or opponent AI.

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| What the AI drives | The player | Matches DodgeAI; one agent, one arena |
| Training stack | DodgeAI, Python, SB3 | Proven over 31.9M steps; only the environment is new |
| Bridge | `dodge-royale gym`, binary over stdin/stdout | No new crates; one encoder serves training and, later, in-game play; PyO3 would link MSVC Python against a GNU toolchain |
| Observation | 256 px window, 64x64 cells of 4 px, player-centred, wrapping | 4 px cells keep 3-6 px hitboxes legible; ±128 px covers the 144-176 px chase lock-on and every threat that can reach the player within the 108-frame horizon |
| Decision interval | One action per simulation frame | The policy corrects more often; look ahead, execute one frame, reconsider |
| Predicted paths | Hold the action `hold_frames` (default 24), then coast | Holding 1 frame makes the nine paths differ by under one cell; holding all 108 sends samples 259 px out, past the window |
| Path origin | The player's actual position and velocity | Momentum carries about 4x current velocity, up to ~10 px; a rest-start path is misplaced by ~2.5 cells |
| Initialisation | Fresh | Lets the layout fit DodgeRoyale and lets paths be velocity-conditioned |

## 3. Section 1 — headless simulation core

### 3.1 Moving the simulation out of `src/game/`

Enemies, population, kamikazes, blasts, collision, wrapping, motion and the
seeded RNG are already renderer-free. Three things are not, and move into the
library:

* Player movement (`game/player.rs` today reads the keyboard directly).
* Enemy system ordering and the wrapping `EnemyWorld` (`game/enemy.rs`).
* Death handling (the game keeps its own screen transition).

**New module `src/simulation.rs`:**

* `SimulationPlugin` adds `EnemyPlugin`, `EnemyPopulationPlugin`,
  `ReferenceEnemyPlugin`, a wrapping `EnemyWorld`, and the ordering
  `PlayerSet::Move -> EnemySet::Prepare .. Replenish`, all inside a
  `SimulationSet`. The game gates that set on `Screen::Playing`; headless runs
  it every frame.
* `PlayerIntent(Vec2)` holds the requested direction. The library system
  `move_players` applies `advance_motion`. `game/player.rs` maps keyboard to
  intent, and the autopilot will write the same component, so a human and the
  AI share one movement path.
* `spawn_player_body` adds the simulation components (`Player`, `EnemyTarget`,
  collider, velocity, intent). The game adds sprite, trail, shadow, ghosting.

Unchanged and still graphics-only: shrapnel, trails, camera, ghosting, menus,
themes, the config screen.

### 3.2 `HeadlessArena`

* `ArenaConfig { seed, enemy_count, max_frames, hold_frames }`.
* Built on `MinimalPlugins` with no run loop and
  `TimeUpdateStrategy::ManualDuration(1/60 s)`. `GameSeed` is inserted before
  `finish()`, so `SeededRngPlugin` reseeds the spawn queue.
* `step()` advances exactly one 60 Hz frame. A test asserts the first update
  advances a full 1/60 s, not zero.
* `set_intent(Vec2)`, `view() -> ArenaView`, `reset(seed)` (a fresh `App`; if
  measurement shows that is slow, despawn-and-respawn instead).
* **Population initialisation is bounded.** An episode starts only once the
  arena holds `enemy_count` enemies. Initialisation runs spawn-only passes:
  enemies are placed but do not move, collide or detonate before frame 0, and
  time is primed separately. Placement can fail, so the number of passes is
  bounded (`max_init_passes`) and exhausting it is an error reported to the
  caller rather than a silent short population. Frame 0 is the first observed
  frame.
* Wrapping comes from each archetype's `BoundaryMode::Wrap` plus the
  `EnemyWorld` half extents; a test asserts both.

### 3.3 `ArenaView`

World units throughout; the encoder converts. Contents:

* Player: position, velocity, collider.
* Enemies: kind, position, velocity, collider. `Dying` enemies are excluded,
  because they can no longer kill.
* Blasts: position, collider, age.
* Arena half extents, frame number, `player_hit`, `enemy_deaths`.

`enemy_deaths` is a per-step count of enemies marked `Dying` in
`EnemySet::Deaths` (enemy-on-enemy), excluding the enemy that hit the player,
recorded before `Cleanup`. Population differences cannot measure it, because
replacements spawn in the same update.

### 3.4 `predict_path`

```rust
predict_path(position, velocity, direction, hold_frames, horizons) -> [Vec2; 6]
```

Pure; steps `advance_motion` at 1/60 s with `direction` held for `hold_frames`
frames and zero input after that; returns wrapped positions at frames
4, 12, 24, 48, 72, 108.

`hold_frames` is configuration, so holding for the whole horizon can be A/B
tested without touching the simulation.

### 3.5 Tests

* Determinism: same seed and intents give an identical state hash after 600
  frames; a different seed differs.
* A hit sets `terminated`; `max_frames` sets `truncated`.
* `ArenaView` unit conversions against a hand-placed enemy.
* `predict_path` against real arena steps: from rest, from a moving start, idle
  braking, diagonal speed, a seam crossing.
* First-frame timing; wrapping configuration; an impossible initialisation
  configuration reports failure.
* Existing enemy tests and `./run.sh check` still pass.

## 4. Section 2a — encoder and protocol (Rust)

### 4.1 Encoder (`src/observation.rs`, wasm-safe)

Window: 256 px across, 64x64 cells of 4 px, centred on the player, y down.

| Part | Contents | Values |
|---|---|---|
| Player | `dpx, dpy`, reference px per frame / 4 | 2 |
| Channels 0-2 | Normal, kamikaze, blast footprints | 3 x 4096 |
| Channel 3 | Blast phase `1 - 2 * progress`: +1 growing, 0 at peak, -1 expiring | 4096 |
| Channel 4 | Player collider footprint | 4096 |
| Channels 5-6 | vx, vy of the hazard that won the cell | 2 x 4096 |
| Paths | 9 actions x 6 horizons x (dx, dy), window px / 128 | 108 |

Total 28,782 f32 (115,128 bytes).

**Coordinate contract**

* Displacement from the player uses `torus::wrapped_delta`, never plain
  subtraction. Then `/ 6.25` to reference px, then `dy = -dy`.
* Window origin is the player minus 128 px. Cell `(col, row)` covers
  `[4*col, 4*col+4)` on each axis, centre `4*col+2`. A rectangle paints every
  cell it overlaps with positive area (the `_paint` rule).
* The player sits at window px (128, 128), the corner shared by cells 31 and
  32, which matches `grid_sample` with `align_corners=False`. There is no
  centre cell: cell-centre displacement is `(col + 0.5 - 32, row + 0.5 - 32)`.
* Flattening: `[dpx, dpy]`, then channels 0..6 row-major (top to bottom, left
  to right), then paths ordered action, horizon, `(dx, dy)`.
* Actions: 0 idle, 1 left, 2 right, 3 up, 4 down, 5 up-left, 6 up-right,
  7 down-left, 8 down-right. The direction reaches `PlayerIntent`
  unnormalised; `advance_motion` normalises. "Up" is +y in world, -dy in the
  observation.
* Velocities are reference px per frame / 4, clipped to [-1, 1]. Enemies peak
  near 0.12 and the player at 0.625, so nothing clips in normal play. The
  encoded velocity is **world** velocity; the senses subtract the player's.
* Paths are not clipped; the model samples with border padding. At startup the
  gym computes the largest possible path extent for `hold_frames` and warns if
  it leaves the window.

**Cell ownership.** Sort key: type priority (blast > kamikaze > normal), then
larger footprint area, then lower entity bits (deterministic within a seeded
run). The winner writes type, velocity and phase together, so no cell mixes two
hazards.

### 4.2 `dodge-royale gym`

Arguments: `--envs N --seed S --enemies 100 --max-frames 3600 --hold-frames 24
--threads T`. No window, no Tokio, no database.

Binary, little-endian, stdin and stdout. Stdout is one locked `BufWriter`,
flushed after every response; all logs go to stderr.

1. **Handshake (Rust -> Python):** magic, protocol version, `n_envs`, grid
   size, cell size, channel list and semantics, `n_actions`, horizons,
   `hold_frames`, `obs_len`.
2. **`STEP`:** one action byte per env; each env advances exactly one frame.
   Response: observations, then per env `terminated`, `truncated`,
   `enemy_deaths`, seed; then terminal observations for finished envs.
3. **`RESET(seed)`** resets every env. **`CLOSE`** exits.
4. **Auto-reset ownership.** A finished env returns its new episode's first
   observation in the observation batch; the previous episode's final
   observation, flags, death count and seed as transition metadata; and the new
   seed as reset metadata. Env `i`'s `k`-th episode seed is `mix(seed, i, k)`,
   counter-based, so scheduling cannot affect seed selection.
5. **Rewards are computed in Python**, so `rewards.json` and the dashboard
   sliders keep working.

### 4.3 Parallelism

`App` is `!Send` in bevy_app 0.19.1 (`RunnerFn = Box<dyn FnOnce(App) ->
AppExit>`, no `Send` bound), so arenas cannot be stepped through Rayon's
parallel iterators.

`T` persistent worker threads each create and keep their own shard of arenas,
with bounded channels to a coordinator that scatters actions and gathers
results in env-index order. The same `SimulationPlugin` runs in the game, the
tests and the workers. The `World`-plus-schedule alternative is rejected: it
would re-implement plugin build, messages, time and deferred commands.

### 4.4 Performance

No bottleneck is assumed. The plan benchmarks simulation, encoding, pipe
transfer, inference and optimisation separately, at 8 and 64 envs.

If transfer dominates, the fallback is packing channels 0-2 and 4 as
bitfields: 115,128 -> 51,640 bytes, about 2.23x. Only if measured.

**Measured**, 2026-09-16, `cargo bench --no-default-features --bench
gym_throughput` on 8 workers. Inference and optimisation are DodgeAI's and are
not covered here.

| Stage | Per env step | 8 envs / step | 64 envs / step |
|---|---|---|---|
| Simulate | 487 us | | |
| Encode | 61 us | | |
| Batch step (simulate + encode, on 8 workers) | 166-201 us | 1.61 ms | 10.60 ms |
| Transfer (write, pipe, parse) | 262-289 us | 2.31 ms | 16.79 ms |
| The same bytes, one `write_all` | 19-20 us | 0.16 ms | 1.21 ms |

Transfer does dominate: 1.4x the batch step at 8 envs, 1.6x at 64. **But
packing is the wrong fix.** The pipe moves these bytes at about 5,500 MiB/s
and the protocol gets 380-420 MiB/s, so roughly nine tenths of transfer is the
codec, not the kernel: `write_floats` and `read_floats` move four bytes per
call, which is 1.8M calls for a 64-env batch. Copying whole slices saves up to
15.6 ms per step at 64 envs against packing's 9.3 ms, and it does not change a
byte on the wire, so `PROTOCOL_VERSION` stays at 1. Do that first and measure
again; packing is worth revisiting only if transfer still dominates after it.

Two things the table is not. The batch step is 2.5x the single-threaded work
divided by the workers, so there is coordination overhead to look at
separately. And `simulate` at 487 us a frame is the largest single cost in the
system; nothing here says whether that is the 4,950 pairwise collision checks
or Bevy's per-`update` overhead.

### 4.5 Tests

* Encoder: priority overlaps, same-type tie-breaks, a seam-straddling hazard,
  cell-boundary rectangles, blast phase sign.
* Y flip with a stationary player; a separate moving-player test asserting that
  grid displacement equals enemy velocity minus player velocity.
* Paths in the observation match `predict_path`.
* Protocol: in-memory round trip, byte-identical output for the same seed and
  actions, the auto-reset path.

## 5. Section 2b — DodgeAI (Python)

New package `dodge/royale/`. No existing v2 code changes, so current
checkpoints keep loading.

### 5.1 `protocol.py`

* Launches the binary from `DODGE_ROYALE_BIN`, defaulting to DodgeRoyale's
  release executable.
* Parses the handshake into a `Layout` dataclass.
* **Exact-length reads:** a loop until the frame is complete; one `readinto`
  need not fill it.
* **No buffer aliasing.** Reads land in preallocated buffers, but the arrays
  returned to SB3 are owned copies, terminal observations included. SB3 calls
  `step()` before storing the previous observation in its rollout buffer, so a
  reused buffer would silently corrupt training.
* A thread drains stderr into a ring buffer, so protocol errors carry its tail.
  `close()` sends `CLOSE`, then kills on timeout.

### 5.2 `vec_env.py` — `RoyaleVecEnv(VecEnv)`

* Spaces: `Box` with per-element bounds (±1 for grid and player values,
  ±inf for paths), `Discrete(9)`.
* **Full `VecEnv` contract:** `step_async`, `step_wait`, `reset`, `seed`,
  `reset_infos`, `env_is_wrapped`, `close`, `get_attr`, `set_attr`,
  `env_method`. `VecEnv.__init__` calls `self.get_attr("render_mode")` and
  catches only `AttributeError`, so `get_attr` returns `[None] * n_envs` for
  `render_mode` and raises `AttributeError` for anything unsupported.
* Reward from `rewards.json`: `survival_per_frame` per surviving frame, minus
  `death_penalty` on a hit, plus `uncontrolled_score_weight * 0.5 *
  enemy_deaths`. `edge_penalty` and `score_weight` do not apply (no walls, no
  score).
* Infos: `terminal_observation`, `TimeLimit.truncated = truncated and not
  terminated`, seeds, and `training_events` with the keys telemetry charts.

### 5.3 `velocity.py` — `VelocityFlowRoyaleExtractor`

Built from `Layout`, reusing v2's structure:

* Trunk: half-resolution stem 7->32, body 32->64->64 (dilation 2)->32,
  `context_net`, `field_head` 48->6.
* Critic: `value_pool` 32->8 to a 4x4 summary; `value_net` takes
  32 means + 128 summary + 2 player + 6 senses = 168 inputs.
* Senses: computed around the geometric centre, channels 0-2 lethal,
  5-6 velocity.
* Controller: `points = centre + paths * 128` from each observation, replacing
  the fixed `offsets` buffer; `grid_sample` with border padding; diagonal
  pairing; `0.985^frame` weights; negated danger concatenated with value
  features, giving 73 outputs.
* `VelocityFlowPolicy` and `_FieldLogits` are reused unchanged.

### 5.4 Integration checklist (DodgeAI)

* `ARCHITECTURES` gains `velocity-flow-royale`; `policy_class`,
  `policy_kwargs`, and checkpoint architecture detection all handle it.
* `GPU_ARCHITECTURES` includes it.
* Hyperparameters apply to **new models and checkpoint loads**. v2's values,
  converted from 4-frame to 1-frame decisions: `gamma = 0.99^(1/4) ~= 0.9975`,
  `gae_lambda = 0.95^(1/4) ~= 0.987`. These preserve time scales; they do not
  make one-frame PPO equivalent to four-frame PPO. Starting values, to tune.
* `dodge/training.py:608` hard-codes `gae_lambda=0.95` and the load path omits
  lambda entirely. Both take the value from the architecture table.
* The layout is persisted through `features_extractor_kwargs`. A loaded
  checkpoint's **full** layout is validated against the handshake — channel
  semantics, action ordering, horizons, hold duration — not only `obs_len`.
  Save and reload are covered by tests.
* Game/architecture compatibility: a royale architecture refuses a PICO-8 env
  and the reverse.
* `train_ppo.py --game royale` builds `RoyaleVecEnv`.
* `training_gui.py --game royale` branches the env factory
  (`dodge/training.py:684`). Game Config shows enemy count, max frames and hold
  frames only. Watch Agent is disabled until the autopilot spec.

### 5.5 Rollout memory

At `n_steps=1024`, observation storage alone:

| Envs | Rollout observations |
|---|---|
| 8 | 0.88 GiB |
| 64 | 7.03 GiB |

Excludes training tensors and temporary copies. **8 envs is the training
default; 64 is a benchmark configuration only.** Minibatch size is bounded
independently of rollout size, so it cannot exhaust GPU memory.

### 5.6 Tests

Unit tests that always run: reward arithmetic against a fixture
`rewards.json`; extractor behaviour; mocked protocol parsing; layout
validation; save and reload.

Integration tests that skip without the binary: handshake, shapes and dtypes,
determinism, auto-reset placement of `terminal_observation` and
`TimeLimit.truncated`, retained observations unchanged after another step, and
one PPO update at `n_steps=16`.

Numerical, not only shape: a known field sampled along known paths, asserting
the horizon pairing (the value judging frame 48 comes from the frame-48 slice)
and the resulting action ranking. A shape-correct controller can still choose
the wrong action.

## 6. Risks

* **Throughput is unknown.** 60 decisions per game second and 115 KB per env
  step; the benchmark decides whether packing is needed.
* **`hold_frames` is empirical.** 24 is a starting point.
* **One-frame PPO is not four-frame PPO.** Converted gamma and lambda keep the
  time scales, nothing more.
* **A fresh policy has no baseline.** Survival time against an idle player and
  against a random policy are the first comparisons.
