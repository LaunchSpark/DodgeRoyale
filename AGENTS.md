# DodgeRoyale architecture and working agreements

## Scope and starting point

This is a browser-rendered Bevy game, with native graphics and a headless Docker target. Keep the
foundation small until gameplay requires more. It was initialized with Bevy CLI
`0.1.0-alpha.2` using `bevy new dodge-royale --template minimal`, backed by
`cargo-generate 0.23.14`, then updated to Bevy `0.19.1`. The existing Git repository
was preserved. Rust `1.95.0` is pinned in `rust-toolchain.toml`, Docker, and CI.

Record significant architectural decisions here when changing ownership,
concurrency, crate boundaries, persistence, features, or deployment. Keep README
commands and CI aligned with those decisions. Do not introduce networking,
authentication, gameplay frameworks, or additional crates speculatively.

## Ownership and concurrency

- **Bevy owns the main thread and ECS world.** `src/main.rs` selects browser or
  native startup; `src/game/mod.rs` builds the game. Express gameplay through components,
  resources, systems, and focused plugins as it grows. Let Bevy schedule independent
  systems and use ECS parallel queries for work that operates on world data.
- **Tokio owns asynchronous I/O.** One explicit runtime has two worker threads.
  Use it for SQLx and future network/file I/O. Keep blocking work off its async
  workers. `src/startup.rs` uses `try_join!` to concurrently prepare PostgreSQL and
  run a small serialization/scoring example. Startup may wait before the Bevy
  loop begins; frame systems must never call `block_on`, perform database queries
  synchronously, or wait for worker results.
- **Rayon owns independent CPU batches.** `src/compute.rs` creates an explicit,
  bounded two-thread pool and uses parallel iterators with stable output order.
  The finite startup demo bridges into Rayon through `tokio::task::spawn_blocking`.
  Reuse a pool for recurring jobs; do not construct pools per entity/frame, use
  unbounded task spawning, or size every executor to all available cores. Measure
  real workloads before changing these initial limits.
- **Serde defines owned data at boundaries.** `src/model.rs` holds snapshots and
  score DTOs. Derive serialization on boundary data rather than the entire ECS
  world. Move owned snapshots into workers; return owned results. If recurring
  background work is introduced, use bounded channels with explicit backpressure,
  cancellation, error handling, and shutdown; apply results in Bevy systems.
- The demo is deliberately finite: it deserializes a small round, computes scores
  in parallel, serializes them, and optionally migrates the database. It does not
  save fabricated scores on normal startup. The app retains the Tokio runtime
  until exit, while the demo's Rayon and database pools close after startup.

## Royale training ownership

All VelocityFlow-for-Royale work belongs in this repository. Rust owns the shared
renderer-free simulation (`src/simulation.rs`), observation encoder
(`src/observation.rs`), and native gym server (`src/gym/`). Player intent feeds the
same movement systems for human and policy control. Observations use the intentional
256-pixel local window; the gym protocol is the Rust/Python boundary.

The gym bypasses database and Tokio startup. Persistent bounded std workers create
and retain their own Bevy apps with single-threaded schedules; apps never cross
threads. The coordinator gathers owned results in environment-index order. This
worker model is separate from the finite startup demo's Rayon pool.

The wire format is frozen at protocol v1 and is guarded two ways that do not
depend on each other: hand-written wire images in `src/gym/protocol/tests.rs` say
what the bytes must be from first principles, and committed golden fixtures under
`tests/fixtures/gym-v1/` are read back by `tests/gym_fixtures.rs` and by Python.
Changing how the codec moves bytes is allowed; changing which bytes it moves is a
version bump and a spec edit. Floats travel a block at a time through a reusable
buffer rather than four bytes per call — measured, not assumed, by
`benches/gym_throughput.rs`, which is also where a claim about transfer cost
belongs. `serde_json` enables `float_roundtrip` because both the handshake layout
and the fixture manifest carry floats as JSON, and its default parser is not
correctly rounded.

The Python trainer lives under `training/dodge_royale/`, with its own package
metadata, tests, CLI, dashboard, and local training artifacts. Golden protocol
fixtures live at root `tests/fixtures/gym-v1/` for both languages. Build and
validate both sides from one repository revision. Python dependencies remain
optional for playing or building the game.

`protocol.py` owns the wire format and nothing else: it launches the gym, decodes
messages and returns arrays, and must not import SB3, Gymnasium or PyTorch.
Training semantics belong in `vec_env.py` above it. Those learner dependencies are
the `train` extra, so the client stays installable and testable without them, and
its decoding is exercised against the committed fixtures with no Rust toolchain,
no build and no child process.

Every array handed to a caller is a copy. Reads may use scratch buffers, but SB3
stores an observation and reads it back only after the next `step`, so a returned
view onto a reused buffer would rewrite a rollout underneath the learner and look
like a training problem rather than a decoding one.

`vec_env.py` owns the training semantics the client refuses to. The gym is
already a batch behind one pipe, so `RoyaleVecEnv` is an adapter and must never
be wrapped in `SubprocVecEnv` or `DummyVecEnv`, which would launch N gyms of N
envs. Because the gym auto-resets, one message describes two episodes: batch
observations are the replacement, `terminal_observation` and every other info
value are the episode that ended. `TimeLimit.truncated` is truncation without
termination, so a death on the frame the budget expires is a death and is not
bootstrapped from. Reward is computed in Python, in `rewards.py`, so tuning it
is a config edit rather than a rebuild; a surviving frame earns survival, a
death earns none and pays the penalty, and a reset is not a transition and earns
nothing. Telemetry counts per step, never cumulatively, and one action per 60 Hz
frame means sixty frames is one second. `get_attr` answers `render_mode` and
raises `AttributeError` for anything else, because `VecEnv.__init__` probes it
and catches only that.

`velocity.py` and `policies.py` are adapted from DodgeAI revision e32f222, with
the source recorded in each module. Nothing imports DodgeAI, and a Royale
checkpoint never needs it to load. The architecture is ported; no weights are.
The network's only output is a danger field -- the controller that reads it is
arithmetic and is not learned, so all the learning pressure lands on the field.
Paths come from the observation rather than a fixed rest-start table, because
Royale's player carries momentum that a rest-start path misplaces by about two
and a half cells. Because the window is player-centred and `path_scale` is its
half width, a stored path already *is* its own `grid_sample` coordinate;
`sample_points` is the identity and is tested as such, since a double offset or
a flipped Y would otherwise keep every tensor the right shape while sampling
cells the player never reaches. Each horizon is read from its own field slice --
the diagonal -- which is what lets "lethal now, clear in two seconds" be
expressed at all. The layout travels into the checkpoint through
`features_extractor_kwargs`, and `require_loadable` refuses both a foreign
checkpoint and a Royale one trained against a different layout before a model is
built: equal lengths with reordered channels would load, run, and point every
trained filter somewhere else.

Shutdown never depends on the child cooperating. `close` is idempotent, writes
CLOSE, then closes stdin only -- closing stdout first would break the server's
one-byte acknowledgement and turn a clean exit into a failed one -- then waits,
terminates, and kills, reaping at every stage. The binary is found through
`DODGE_ROYALE_BIN` or the checkout the module was imported from, never the
working directory; an override that is set but wrong is an error rather than a
silent fall back to a different build.

DodgeAI remains independent. Adapt needed policy/training code locally with source
revision/path attribution and applicable notices; do not modify or import DodgeAI,
add it as a dependency, or require sibling checkouts, symlinks, cartridges, or old
checkpoints. New checkpoints use stable `dodge_royale` module paths. The local
trainer has Royale-only entry points and no PICO-8 environment or migration path.

## Player appearance

The reference's player is drawn by `circfill` particles, not a sprite-sheet image
(`reference/code.lua` drawgame/updateparts). `src/game/player_art.rs` recreates
the filled pixel circle as one shared 9×9 nearest-filtered mask, tinted with the
existing foreground and theme shadow colors. The visual radius follows the
reference's four pixels using `art::PIXEL`; collision bounds remain independent.
All shadows draw behind the trail bodies. Trail stamps sample movement at 60 Hz,
shrink by `0.9` per reference frame, and expire after ten frames. They are visual
`GameEntity` entities, cleaned up on screen exit, with no colliders or enemy-target
markers. Keep visual effects separate from player movement and camera transforms.

## Camera following

The browser/native game camera is independent of the player entity. Its target is
the player's position plus velocity times 0.15 seconds of lookahead. It follows
that target through our reusable `tween::exponential` in `src/tween.rs`, with a
decay rate of 12 per second. This implements `1 - exp(-rate * dt)` using `exp_m1`
for precision at small frame times, then interpolates from the current value.
Keep both velocity lookahead and smoothing. Do not restart a timed tween on each
direction change. See
[the exponential smoothing explanation](https://lisyarus.github.io/blog/posts/exponential-smoothing.html).
Run camera following after player movement in `Update`,
before Bevy propagates transforms. Preserve camera Z, and clamp both the target
and the resulting position using the viewport extents. `R` explicitly recenters.
If movement later uses `FixedUpdate`, interpolate the rendered player position
before camera following; smoothing alone does not fix mismatched update timing.

## Enemy foundation and collision

Use ECS composition for the reusable enemy base, following Bevy's
[required components](https://bevy.org/news/bevy-0-15/) model. Public `Enemy` in
`src/enemy.rs` requires `EnemySettings`, `EnemyState`, `EnemyMotion`, `Velocity2d`, `Collider`,
and `Transform`. Concrete types add their own marker/components; avoid a boxed
abstract-class hierarchy. The core plugin has no renderer, database, or native
runtime dependency and is tested with a headless Bevy app. The player shares
`Velocity2d` and is marked `EnemyTarget`.

`EnemySettings` exposes speed, exponential response, detection/retention ranges,
stopping distance, arrival radius, lookahead, and chase/manual steering. It uses
Serde with defaults for data-driven type definitions. Validate settings with
`is_valid`; invalid settings freeze movement. All distances use world units and
times use seconds. Acquisition chooses the nearest eligible actor, retains it
within a larger radius to avoid target switching, and reacquires after target
loss. Missing targets produce smooth braking. Movement uses `tween::exponential`
and integrates its velocity curve, caps background-tab time, preserves Z, and
clamps the full offset collider to `EnemyWorld` bounds.

The ordered public sets are `EnemySet::Prepare`, `Steer`, `Modifiers`, `Move`,
`Deaths`, `Contacts`, `Effects`, `Cleanup`, and `Replenish`. In the game,
they run only during `Screen::Playing` and after player movement. A custom enemy
can set `SteeringMode::Manual` and write `Velocity2d` between Steer and Move;
shared movement still enforces its speed limit and bounds. Chase consumers can
override `EnemyState.target` with an eligible entity between those same sets.
`src/game/enemy.rs` decorates newly spawned types with sprites and shadows and tags
them `GameEntity` for normal screen cleanup. It clears pending spawns on screen exit.

`src/collision.rs` supplies configurable axis-aligned rectangular sensor hitboxes
(half extents, world-space offset, enabled flag). Actors are unparented; rotation
and transform scale do not resize hitboxes. Touching edges count as contact.
`EnemyContact` reports each overlapping enemy/target pair once per update after
movement, independently of the selected pursuit target. Consumers read messages
after Contacts and implement their own hit cooldowns/damage/knockback. This is
discrete overlap detection, with no penetration response, swept collision,
or pathfinding. Add those when gameplay requires them;
high-speed projectiles will need swept collision. Enemy logic stays in Bevy
systems, not Tokio/Rayon jobs. No physics crate is needed for this foundation.

Enemy/enemy contact kills both actors. The Deaths system checks each unordered
pair after movement, collects the complete set of victims before mutation, and
marks each victim `Dying`. Effects run before Cleanup despawns each victim once
(including visual children). This makes simultaneous
collision clusters independent of query order. Disabled/invalid colliders cannot
kill. Dying actors cannot move or report player contacts. Deferred cleanup finishes
before population counts are read. For this small population, pairwise overlap checks
are sufficient; add spatial indexing if measured counts require it.

## Enemy population and spawn queue

`src/enemy_population.rs` provides an optional renderer-independent
`EnemyPopulationPlugin`. The game's default population target is 100, with weighted
Normal and Kamikaze definitions in `EnemyPopulation.types`. Each `EnemyArchetype`
supplies a name, typed `EnemyKind`, relative weight, settings, and collider. The
spawned `EnemyType` preserves its display name and `EnemyKind` selects behavior.
Zero-weight definitions are disabled for future selection. The controller counts all
live `Enemy` entities and fills only the deficit; lowering the target does not
kill existing actors. Replacements are created after deaths in the same update
when placement succeeds and the work budget permits.

First fill `EnemySpawnQueue` with selected **type snapshots**, then sample random
positions for those exact settings. Positions must be strictly beyond that type's
detection range plus a configurable margin (32 units initially) from every
`EnemyTarget`. Fit the whole offset collider inside `EnemyWorld.half_extents` and
reject actor overlaps. Reserve each accepted placement immediately so deferred
spawns in the same batch cannot overlap. Randomness uses one `fastrand::Rng` owned
by the queue, with browser `js` seeding enabled only on WASM; `with_seed` supports
reproducible tests.

Bound work to 64 queued requests and 64 candidates per request per update by
default. Failed requests retain their selected type and retry on later updates;
never reroll to an easier type or relax detection/collider constraints. No targets,
unbounded/invalid arena geometry, or impossible placement defers spawning, so the
live count may temporarily be below the target. Catalog changes affect new
selections; clear the queue explicitly to discard old snapshots. Queue and active
actors are cleared through the game's normal screen lifecycle. Newly spawned
chasers remain idle until a player comes within their detection range.

## Reference enemy personalities

`src/enemy_types.rs` implements the hostile personalities from `reference/code.lua`:
Normal (`p = 0`) is a filled square; Kamikaze (`p = 1`) is outlined. The default
catalog preserves their nominal 76.5:17.5 relative weights, renormalized without
power-ups, and each type's 3/4/5/6-pixel size distribution (20/50/20/10). Sizes are
selected as part of the queued type snapshot before placement. Shared chase
smoothing, detection ranges, and constant-population spawning remain the movement
foundation. Both reference types opt into `BoundaryMode::Bounce`; other consumers
retain the default clamp behavior.

Kamikazes reduce their signed speed multiplier by 0.6 per second within 156.25
world units of their tracked player, and recover outside that distance. They can
reverse as in the reference; the multiplier is bounded to [-1, 1]. This modifies
`EnemyMotion` without overwriting the archetype's speed. There is no proximity fuse.
Every dying kamikaze creates one stationary `KamikazeBlast` (`p = -1`) before cleanup.
Its square grows to 187.5 units over half a second, shrinks, and expires at one
second. `KamikazeSettings` exposes these distances, rates, and timings.

Blasts are transient hazards, not `Enemy` entities: they do not occupy population
slots or collide with enemies, matching the reference's exclusion. Their entire
filled hitbox is lethal even though their art is an outline. Normal/kamikaze
contact and blast contact emit one `PlayerHit`, disable the player's collider,
zero velocity, and mark it `Defeated`. The game pauses that player's movement and
returns through the existing screen wipe to the menu. A new game creates a fresh
player. Enemies, blast roots, and their child visuals use normal `GameEntity`
cleanup. The shared mask/trail player art is unaffected.

Power-up personalities 2/3/4 (arena clear, freeze, shrink) are deliberately absent
from the catalog and have no pickup/effect systems, per the user's instruction.

## Persistence

PostgreSQL 17 is the development database. SQLx `0.9` uses Tokio and Rustls with
only PostgreSQL, macros, and migrations enabled. `src/database.rs` owns pool
creation, migrations, and parameter-bound queries; ECS systems do not own SQL.
The initial pool permits five connections with a five-second acquisition timeout.

`DATABASE_URL` is optional: absent means the game runs without PostgreSQL; present
means startup must connect and migrate successfully or report an error. An invalid
or unreachable configured database must not silently become an in-memory fallback.
`.env.example` documents local values; the application reads environment variables
and does not implicitly load `.env`. Never commit real credentials or log URLs.

Migrations in `migrations/` are embedded with `sqlx::migrate!`. `build.rs` watches
the directory so adding a migration triggers a rebuild. Keep SQL line endings LF
for stable checksums. Add new migrations rather than rewriting applied ones.
Queries use `.bind()` and runtime checks so ordinary builds need neither a live
database nor `.sqlx` metadata. If adopting `query!`, add an offline metadata
generation/check workflow at the same time. Use transactions for future writes
that must succeed together. The ignored database test verifies the real schema,
bound values, and constraints against PostgreSQL.

## Features and Docker

The default `graphics` feature enables the browser/native game and 2D renderer. With
`--no-default-features`, the same binary uses `MinimalPlugins` and a paced 60 Hz
runner, without graphics dependencies. `--headless` bypasses windows in a graphical
build; `--smoke-test` performs startup and one headless frame, then exits. Keep
headless tests independent of a GPU/display. Browser builds target
`wasm32-unknown-unknown` and WebGL 2 through Bevy CLI. SQLx/PostgreSQL and the
Tokio/Rayon setup target native processes only.

Docker's `web` target serves the WASM bundle through non-root nginx; rendering
happens in the browser and needs no database. The `runtime` target contains a
non-root headless release binary. Compose
provides a healthy PostgreSQL dependency and a persistent named volume. Its DB
port binds only to localhost, and example credentials are for local development.
Native runs connect to localhost; containers connect to service `db`. Graphics
run on the host; display/GPU forwarding into Docker is not configured. The optional
`development` target supplies Rust tools and Linux Bevy libraries. Bevy's built-in
terminal handler turns Ctrl-C into `AppExit`; Docker's normal SIGTERM ends
the headless process. Add coordinated SIGTERM draining when ongoing jobs exist.

## Rust practices and checks

Follow [Namtao's Rust practices](https://www.namtao.com/rust/) in the context of this
small project: strict Clippy, Serde/Rayon, readable error reports via `color-eyre`,
and useful developer feedback. The manifest denies pedantic/nursery lints and
panic-prone operations; `unsafe` is forbidden. Propagate errors with `Result` and
`?`, prefer checked conversions and arithmetic, and avoid production
`unwrap`/`expect`, indexing, `todo!`, and `unimplemented!`. Test-only allowances
live in `clippy.toml`. Use narrow, reasoned lint expectations for Bevy injection
when needed. Do not disable strict lints across the crate to silence a system.

Prefer explicit types/enums for meaningful states; use typestate only when it
prevents an actual invalid transition. Stable Rust plus Docker is the reproducible
environment here; Nix/nightly are not prerequisites. Optional tools are `bacon`
and `cargo-nextest`; Cargo's built-in commands remain sufficient.

Before finishing relevant changes, run:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo clippy --locked --all-targets --no-default-features -- -D warnings
cargo test --locked --no-default-features
cargo run --locked --no-default-features -- --smoke-test
docker compose --profile tools config --quiet
```

For persistence changes, start the Compose database and run
`cargo test --locked --no-default-features --test database -- --ignored` with
`DATABASE_URL` set. For container changes, build the image and run
`docker compose run --rm app --smoke-test`. CI covers both feature configurations
and the real database test. Keep tests focused on behavior and boundaries.
