# DodgeRoyale

A browser-rendered Bevy prototype: move a white pixelated orb through a 6,000 × 4,000
world with smooth acceleration and a camera that follows your movement. The blue
arena, circular player, and shrinking trail take inspiration from artridge's
[**Dodge**](https://www.lexaloffle.com/bbs/?tid=34986). Its Lua source is kept locally
in `reference/code.lua`, which is gitignored and not distributed with this repository.

Use **WASD** or the **arrow keys** to move. **R** recenters the camera on its current
follow target. Click the canvas or **Focus game** to restore keyboard focus.
The arena maintains a target population of 100 enemies. Each spawn selects a type
first, then finds a random position outside that type's player detection range.
Enemies that collide with each other both die and are replaced. Enemy or blast
contact defeats the player and returns to the menu.

## AI training

The VelocityFlow trainer lives in this repository under
[`training/`](training/README.md), alongside the Rust simulation, observation
encoder, and gym protocol. It is a uv project: the protocol client, policy, PPO
session and CLI are implemented and tested; the marimo dashboard is still
pending. DodgeAI remains independent, and no sibling checkout is required.

Python is optional for building and playing the game. To train:

```sh
cargo build --release --no-default-features   # the gym binary the trainer drives
cd training
uv sync --extra dashboard --extra cu126       # or --extra cpu without an NVIDIA GPU
uv run python -m dodge_royale.train --check-env
```

[`training/README.md`](training/README.md) has the full setup, including what to
install first and how to point at a binary somewhere else. See the
[implementation plan](docs/superpowers/plans/2026-09-15-velocity-flow-royale-implementation.md)
for what remains.

## Building

### Prerequisites

1. Install [rustup](https://rustup.rs). Rust 1.95.0 is pinned in
   `rust-toolchain.toml`, so the first `cargo` command in this directory downloads
   the right toolchain, Clippy, and rustfmt automatically.
2. On Linux, install [Bevy's OS dependencies](https://github.com/bevyengine/bevy/blob/main/docs/linux_dependencies.md)
   (ALSA, udev, X11/Wayland headers). macOS and Windows need nothing extra.
3. For the browser build only, add the WebAssembly target and the pinned web tools:

   ```sh
   rustup target add wasm32-unknown-unknown
   cargo install --locked \
     --git https://github.com/TheBevyFlock/bevy_cli \
     --rev 53fea37954e71b872df815ad81a3809d620db856 bevy_cli
   cargo install --locked --version 0.2.128 wasm-bindgen-cli
   ```

   The `wasm-bindgen-cli` version must exactly match `wasm-bindgen` in `Cargo.lock`.

If `cargo` is not on your `PATH` after installing rustup, run `. "$HOME/.cargo/env"`.
No database is required to build or play.

### Native (desktop window)

```sh
./run.sh                          # Debug build, then launch the game
cargo build --locked              # Debug build → target/debug/dodge-royale
cargo build --locked --release    # Optimized build → target/release/dodge-royale
cargo run --locked --release      # Build and launch the optimized game
```

The first build compiles Bevy and takes several minutes; later builds are
incremental. Release builds use thin LTO and a single codegen unit, so they compile
slower but run faster.

Headless builds (`--no-default-features`, `cargo headless`, `cargo smoke`) write to
the same `target/debug/dodge-royale` path and replace the playable binary with a
windowless one. `./run.sh test`, `./run.sh check`, and `./run.sh smoke` rebuild the
graphics binary afterwards; if you run those commands through `cargo` directly, run
`cargo build --locked` before launching. `./run.sh help` lists every task.

### Browser

```sh
bevy run --locked web --open
```

The game opens at <http://127.0.0.1:4000>. Keep the command running while playing;
stop it with Ctrl-C and rerun it after Rust changes. The browser needs WebGL 2,
WebAssembly, and a keyboard. [Bevy CLI](https://thebevyflock.github.io/bevy_cli/cli/web.html)
compiles Rust to WebAssembly, generates the JavaScript bindings, and serves
`web/index.html`.

To produce a static release bundle:

```sh
bevy build --locked --release web --bundle --wasm-opt false
```

Serve the generated `target/bevy_web/web-release/dodge-royale/` directory over
HTTP. `wasm-opt` is disabled for this foundation; Cargo still applies its optimized
web release profile.

To build without installing any Rust tools, use Docker (next section).

## Docker

```sh
docker compose up --build web
```

Open <http://localhost:8080>. This builds the WebAssembly bundle and serves it
through a non-root nginx container. Rendering happens in your browser. The `web`
service runs independently of PostgreSQL and the native app. Set `WEB_PORT` to use
another host port. The first build installs the pinned Rust/Bevy tools and compiles
dependencies; later builds reuse Docker layers and Cargo caches.

The native foundations remain available as separate services:

```sh
docker compose up --build app
docker compose run --rm app --smoke-test
docker compose down
```

`app` starts its PostgreSQL dependency and runs Bevy headlessly. It has no HTTP or
multiplayer API. PostgreSQL persists in a named volume and binds to
`127.0.0.1:5432`; set `POSTGRES_PORT` if needed. `down` preserves database data.
Only use `docker compose down -v` when you intend to delete that data.

An optional tools container provides Rust, Clippy, rustfmt, and Linux Bevy libraries:

```sh
docker compose run --rm dev cargo check --locked --all-targets
docker compose run --rm dev cargo test --locked --no-default-features
```

## Native app and database

The same game also runs in a native window with `cargo run --locked` (see
[Building](#building)).

Native startup exercises Tokio async I/O, Rayon parallel compute, and Serde JSON.
SQLx connects and migrates PostgreSQL only when `DATABASE_URL` is exported. These
native services are excluded from the browser build.

```sh
cp .env.example .env
docker compose up -d --wait db
set -a
. ./.env
set +a
cargo run --locked
```

The executable reads exported variables and does not load `.env` automatically.
Compose reads `.env` itself and supplies the container hostname. Example credentials
are for local development. If you change them, update the native `DATABASE_URL` too.
An invalid configured database produces an error rather than silently falling back.

```sh
cargo headless                  # Native headless loop; Ctrl-C exits
cargo smoke                     # One headless frame, then exit
cargo run --locked -- --help
```

The default `graphics` feature enables the game renderer. Native
`--no-default-features` builds remove graphics; `--headless` also bypasses windows
in graphical native builds. The smoke test exercises native startup and one Bevy
frame without a GPU. It does not insert demo records.

## Enemy population

Tune `EnemyPopulation` in `src/enemy_population.rs`: `target_count` defaults to 100,
with weighted Normal and Kamikaze definitions in `types`. Each queued type brings its
own detection range and collider. Spawns stay inside the arena, outside detection
range plus a 32-unit margin, and clear of other actors. Failed placements stay
queued for later attempts. The population can temporarily fall below its target
when no valid position is available; existing enemies are not culled when lowering
the configured target.

Enemy collisions remove both actors, including their shadows, before replacements
are placed. Enemies spawn idle; move around the arena to enter their detection range.

Normal enemies are filled squares; kamikazes are outlined squares that slow and
can reverse near the player. A dying kamikaze leaves an expanding, shrinking square
blast that lasts one second. The whole blast area is lethal, and it does not count
toward the enemy population. Both types bounce off arena edges. Spawn weights and
the four size variants follow the reference. Power-ups are not connected.

## Adding an enemy type

The shared base is `enemy::Enemy`: Bevy inserts its required settings, tracking
state, velocity, collider, and transform. Add your type's marker and override the
components you need:

```rust
use bevy::prelude::*;
use dodge_royale::{collision::Collider, enemy::{Enemy, EnemySettings}};

#[derive(Component)]
struct FastChaser;

fn spawn(mut commands: Commands) {
    commands.spawn((
        FastChaser,
        Enemy,
        EnemySettings {
            speed: 280.0,
            response: 10.0,
            detection_range: 1_000.0,
            retention_range: 1_400.0,
            lookahead_seconds: 0.2,
            ..default()
        },
        Collider::rectangle(Vec2::splat(16.0)),
        Transform::from_xyz(500.0, 200.0, 10.0),
        Sprite::from_color(Color::WHITE, Vec2::splat(32.0)),
    ));
}
```

The game already installs `EnemyPlugin` and `EnemyPopulationPlugin`. Add an
`EnemyArchetype` (name, `EnemyKind`, weight, settings, collider) to
`EnemyPopulation.types` for automatic replenishment, or register
your own spawn system. Use
`src/game/enemy.rs` as the example for screen cleanup and themed visuals. Mark
eligible targets with `EnemyTarget`; the player already has it. Other settings
include stopping distance and arrival slowing. Collider size, offset, and enabled
state are independent of the sprite.

For custom steering, set `SteeringMode::Manual` and write `Velocity2d` in a system
ordered after `EnemySet::Steer` and before `EnemySet::Move`. Read `EnemyContact`
messages after `EnemySet::Contacts` to implement hit behavior. Each overlap emits
one message per update: consumers own hit cooldowns. These are rectangular sensors,
with no physical pushing. `ReferenceEnemyPlugin` makes hostile contact lethal;
enemy/enemy contact kills both actors. Use `EnemySet::Effects` for on-death behavior
before Cleanup removes `Dying` actors. The core also works without graphics.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo clippy --locked --all-targets --no-default-features -- -D warnings
cargo check --locked --target wasm32-unknown-unknown
cargo clippy --locked --target wasm32-unknown-unknown -- -D warnings
cargo test --locked --no-default-features
cargo run --locked --no-default-features -- --smoke-test
docker compose --profile tools config --quiet
```

Headless tests cover normalized movement, acceleration and braking, arena bounds,
background-tab timing, camera viewport bounds, exponential following, and the
serialization/compute boundaries. CI checks native graphics, headless, and WASM
configurations and runs the PostgreSQL integration test. Browser rendering still
needs an actual browser check after renderer or shell changes.

The gym serves headless arenas to a trainer over stdin and stdout
(`./run.sh gym --envs 8`). Its throughput is measured rather than assumed:

```sh
./run.sh bench          # 8 and 64 envs, the sizes the design names
./run.sh bench 4        # any other env count
```

Golden protocol fixtures live in
[`tests/fixtures/gym-v1/`](tests/fixtures/gym-v1/README.md): real protocol bytes,
committed, so the Python client can be tested without a Rust toolchain. Two Rust
test sets guard the wire format independently — hand-written wire images, and the
fixtures read back with the shipped codec. Regenerate after an intentional
protocol or encoder change:

```sh
cargo run --locked --no-default-features --example gym_fixtures
```

The benchmark times simulation, encoding and pipe transfer separately, per env
step, and prices the design’s packing fallback against them. It must run
optimised; a debug build makes the simulation look like the bottleneck.

With the exported environment and Compose DB from above, run the persistence check:

```sh
cargo test --locked --no-default-features --test database -- --ignored
```

The test inserts, reads, and removes its own records and verifies a database
constraint. Embedded migrations run automatically on configured native startup;
ordinary builds need neither SQLx CLI nor a database.

Optional tools are `bacon` and `cargo-nextest`; Cargo's built-in commands suffice:

```sh
cargo install --locked bacon cargo-nextest
bacon clippy
bacon headless
cargo nextest run --locked --no-default-features
```

The project was initialized with Bevy CLI `0.1.0-alpha.2`, its minimal template,
and `cargo-generate 0.23.14`, then updated to Bevy `0.19.1`. Architecture and the
concurrency boundaries live in [AGENTS.md](AGENTS.md), following
[Namtao's Rust guidance](https://www.namtao.com/rust/). The camera retains velocity
lookahead and uses our reusable `tween::exponential` in `src/tween.rs`, following
the approach in the [official top-down camera example](https://bevy.org/examples/camera/2d-top-down-camera/)
and [lisyarus's exponential smoothing explanation](https://lisyarus.github.io/blog/posts/exponential-smoothing.html).
