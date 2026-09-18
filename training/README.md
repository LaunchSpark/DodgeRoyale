# DodgeRoyale training

The Python trainer for DodgeRoyale: the gym protocol client, the SB3 vector
environment, the VelocityFlow policy, the PPO session, the CLI, and the
dashboard. It drives the Rust simulation as a child process, so a working
setup needs both halves of the repository.

**Implemented so far:** `protocol.py` (the protocol-v1 client), `vec_env.py`
(`RoyaleVecEnv`), `rewards.py`, `telemetry.py`, `velocity.py` (the extractor),
`policies.py`, `training.py` (the PPO session) and `train.py` (the CLI), with
their tests, plus `metrics.py`, `worker.py` and `dashboard.py` (the marimo
dashboard). Tasks 8-13 of the
[implementation plan](../docs/superpowers/plans/2026-09-15-velocity-flow-royale-implementation.md)
are complete, with [measured results](../docs/superpowers/results/2026-09-17-benchmark-royale.md).
The one item left is a manual check of the graphical game, which needs a human
at a window.

## Setting up

### 1. Install the tools

| Tool | Why | Install |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python environment and dependency manager | `winget install astral-sh.uv`, `brew install uv`, or the [install script](https://docs.astral.sh/uv/getting-started/installation/) |
| [Rust](https://rustup.rs/) | Builds the gym binary the trainer drives | `rustup`, which picks up the toolchain pinned in `rust-toolchain.toml` |

You do not need to install Python yourself. uv reads `.python-version` and
fetches CPython 3.12 if the machine does not already have it.

### 2. Build the gym binary

From the **repository root**, not this directory:

```sh
cargo build --release --no-default-features
```

`--no-default-features` leaves out the renderer, which the trainer never uses
and which drags in graphics libraries. This writes
`target/release/dodge-royale` (`.exe` on Windows), which is where the client
looks for it.

The first build takes a few minutes. A debug build works for smoke checks,
but train against the release one: `benches/gym_throughput.rs` is written
against release, and the debug binary has never been benchmarked here.

### 3. Create the Python environment

From **this** directory, pick the line that matches what you are doing:

```sh
cd training

uv sync --extra dashboard --extra cu126      # everything, PyTorch with CUDA
uv sync --extra dashboard --extra cpu        # everything, PyTorch without CUDA
uv sync --extra train                        # learner and tests, no dashboard
uv sync                                      # protocol client only (see below)
```

Pick one of the first two unless you know you want less. Any of them creates
`training/.venv`, installs the exact versions pinned in `uv.lock`, and installs
`dodge_royale` itself in editable mode. Two collaborators running the same line
get the same environment.

**`uv sync` on its own is not a smaller version of the others.** It installs
the protocol client and pytest and nothing else, because `protocol.py` knows
nothing about SB3, Gymnasium or PyTorch and a wire-format change should be
testable without a multi-gigabyte download. What it does *not* give you is the
rest of the suite: `test_vec_env.py`, `test_velocity.py` and `test_training.py`
import the learner, so collecting the whole `tests/` directory fails at import.
Run the subset that matches the install:

```sh
uv run pytest tests/test_protocol.py tests/test_lifecycle.py   # 67, or 64 with no binary
```

uv syncs make the environment match the flags you passed, so running plain
`uv sync` in a checkout that already has the learner **uninstalls it**, torch
included. Pass the same extras every time, or add them
(`uv sync --extra dashboard --extra cu126`) rather than dropping them.

**On Windows, put the cache on the same drive as the checkout.** uv hardlinks
from its cache into `.venv` when both are on one filesystem and copies when
they are not. A sync that changes nothing does no work either way, but any
sync that installs or swaps PyTorch copies it — 2.4 GB, about a minute
here. If `uv cache dir` prints a `C:` path and you work on another drive, set
`UV_CACHE_DIR` alongside your projects once:

```sh
setx UV_CACHE_DIR D:\.uv-cache
```

> **`uv sync --all-extras` does not work here, on purpose.** `cpu` and `cu126`
> are declared as conflicting extras, because they are two builds of the same
> package, so asking for every extra at once asks for both. uv says so rather
> than silently picking one. Name the extras you want instead.

**Which torch build?** If you have an NVIDIA GPU, use `cu126`. Without the
extras PyPI decides, and on Windows PyPI ships a CPU-only wheel, so the card
would sit idle. Training runs either way. Check what you got:

```sh
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.14.0+cu126 True` is a GPU install. `2.14.0+cpu False` or `2.14.0 False` is
not. `cu126` with `False` means the build is right but torch cannot reach a
card — driver, toolkit, or no NVIDIA GPU present. `--device cpu` forces CPU
whatever is installed.

### 4. Check it worked

```sh
uv run pytest                 # 185 tests with a binary built, 180 + 5 skipped without
uv run python -m dodge_royale.train --check-env
```

180 of those need neither Rust nor a gym process. The other five are marked
`live` and drive the binary from step 2; they skip when there is none.

The last one launches a real gym, resets and steps the batch, prints the
layout the two sides negotiated, and exits. If it prints a layout and "the
batch resets and steps", both halves are talking to each other.

`uv run` runs a command inside the project environment without activating
anything. Activating works too — `.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` elsewhere — and then `uv run` can be dropped from
every command below.

### If the binary is somewhere else

The client looks for `target/release/dodge-royale` relative to the checkout it
was imported from, not the shell's working directory. Point it elsewhere with
an environment variable:

```sh
DODGE_ROYALE_BIN=/path/to/dodge-royale uv run pytest -m live
```

A custom `CARGO_TARGET_DIR`, or an install outside the checkout, needs that
override. If it is set but wrong that is an error rather than a quiet fallback
to a different build, so a typo cannot train you against the wrong binary.

## Running the tests

```sh
uv run pytest                    # everything, browser tests included
uv run pytest -m "not browser"   # skip the slow Playwright suite
uv run pytest -m live            # only the tests that drive a real gym
uv run pytest -m "not live"      # only the ones that need no binary
```

Three groups. Most tests need neither Rust nor a gym. `live` tests drive the
real binary and skip when there is none. `browser` tests additionally launch a
marimo server and Chromium, which needs a one-time
`uv run playwright install chromium`; they are the only coverage of the
dashboard's controls, since a button press only reaches the worker through
marimo's reactive graph.

Fixture tests read the committed messages in `../tests/fixtures/gym-v1/` and
never build or run Rust, which is why most of the suite passes on a machine
with no Rust toolchain at all. Live tests skip when no binary is built, but
fail rather than skip when `DODGE_ROYALE_BIN` is set and broken.

## Training

```sh
uv run python -m dodge_royale.train --dry-run     # what would run; launches no gym
uv run python -m dodge_royale.train --check-env   # reset and step the batch, then exit
uv run python -m dodge_royale.train               # 8 envs, 1024 steps, 100 enemies
uv run python -m dodge_royale.train --resume-latest
uv run python -m dodge_royale.train --help        # every override
```

Royale only: there is no `--game` switch. `--hold-frames` is the prediction
hold, not an action repeat; the policy decides every frame either way.

Checkpoints land in `training/checkpoints/`, ignored by Git along with
`training/runs/` and `training/.venv/`. `rewards.json` holds the reward
controls that apply to Royale; `--rewards` points at a different file.

## Benchmarking

```sh
uv run python tools/benchmark_royale.py                  # 8 and 64 envs, plus baselines
uv run python tools/benchmark_royale.py --envs 8 --skip-baselines
uv run python tools/benchmark_royale.py --out results.md --json results.json
```

Times the round trip, inference, optimization and the whole `learn` separately,
and records the revision, hardware, versions, device, rollout shape and peak
memory beside them. It reports what a full 1,024-step rollout *would* cost
rather than allocating one: at 64 envs that is 7.03 GiB of observations before
any training tensors.

`cargo bench --locked --no-default-features --bench gym_throughput`, from the
repository root, splits the Rust side into simulation, encoding and transfer.

## Training history

Every finished episode is appended to `<checkpoint-dir>/<run-name>.history.jsonl`
as it happens, and copied beside each checkpoint as
`<checkpoint>.history.jsonl`, so a model carries its whole past rather than
only the rolling averages a run prints while it goes.

Each record holds the survival time **and the enemy count it was played
against**, because those only mean something together: three seconds against
twelve enemies is not three seconds against a hundred, and the count can change
between runs. It also records the timestep the episode ended on, its seed, the
prediction hold, the reward, the enemy-on-enemy kills, and whether the player
died or the clock ran out — a timeout is a censored survival time, not a longer
one.

```python
from dodge_royale.history import read_history
from collections import Counter

episodes = read_history("checkpoints/royale-final.history.jsonl")
print(len(episodes), Counter(e.enemies for e in episodes))
print(max(e.seconds for e in episodes))
```

Resuming continues the history rather than starting a new one: a resumed
checkpoint's records are inherited when the new run has none of its own. JSON
Lines, flushed per episode, so a run that is killed keeps everything it
actually finished.

## The dashboard

A [marimo](https://marimo.io/) notebook over a live training session: start,
pause, resume, save, stop, and the run's metrics and charts as it goes.

```sh
uv run marimo run  dodge_royale/dashboard.py --no-sandbox   # use it
uv run marimo edit dodge_royale/dashboard.py --no-sandbox   # change it
uv run python dodge_royale/dashboard.py                     # a short real run, no browser
```

From the repository root, `./run.sh web` starts this server and the browser game
together; `./run.sh web-docker` does the same with the Docker web service. The
game's **AI dashboard** link opens `http://127.0.0.1:2718/` in a new tab. The
runner uses the installed `.venv` directly, so launching the web game does not
trigger an `uv sync` or swap the PyTorch build.

`--no-sandbox` because the notebook carries a PEP 723 header, and without the
flag marimo offers to build a separate environment from it. The header is
there so the notebook is self-describing and `marimo edit --sandbox` works;
the project environment you already synced has everything.

The last line is script mode. marimo runs every cell once with a small
configuration, trains for a few hundred steps against a real gym, prints the
metrics and exits -- the end-to-end smoke test for the dashboard, needing no
browser.

Training runs on a background thread so the page stays responsive. Only one
run exists at a time: marimo re-runs a cell whenever its inputs change, so a
cell that *created* a run would create another on every rerun, and the worker
lives outside the notebook to prevent exactly that. Stopping, failing, closing
the tab or killing the kernel all close the gym process.

### Watch the agent

The dashboard can run the newest policy and draw **what it sees** — the
256-pixel observation window, the nine paths it is choosing between, and which
one it took. Not the arena: protocol v1 carries the observation, not the world.
That is the more useful picture anyway, because it is exactly the information
the policy had. A dodge into a threat is a bug in the field; a dodge into empty
space is a bug in the paths.

A snapshot is published automatically after every completed update, to
`<checkpoint-dir>/<run-name>-live/update-XXXXXXXX.zip`. The viewer picks up the
newest one **between episodes, never during one**, so every episode is
attributable to a single update. Writes are atomic — a hidden name renamed into
place — so the viewer can never open a half-written zip, and only the newest few
are kept.

Watching the agent play the real game, with the game's own art, is the in-game
autopilot: weight export and a Rust forward pass, which is its own spec.

## Layout

```text
training/
  pyproject.toml          # Package, dependencies, extras, test configuration
  uv.lock                 # Pinned versions. Committed; never edited by hand.
  .python-version         # The interpreter uv fetches
  rewards.json            # Reward controls that apply to Royale
  dodge_royale/
    protocol.py           # Protocol-v1 client and Layout  [done]
    vec_env.py            # RoyaleVecEnv  [done]
    velocity.py           # VelocityFlowRoyaleExtractor  [done]
    policies.py           # Policy classes and checkpoint registration  [done]
    rewards.py            # Royale reward configuration  [done]
    telemetry.py          # Training events and duration reporting  [done]
    training.py           # PPO session and process lifecycle  [done]
    train.py              # CLI entry point  [done]
    history.py            # Per-episode training history  [done]
    snapshots.py          # A policy snapshot per update  [done]
    viewer.py             # Renders what the policy sees  [done]
    watching.py           # The one watch session a kernel owns  [done]
    metrics.py            # Metric definitions shared by CLI and dashboard  [done]
    worker.py             # Background training thread and its controls  [done]
    dashboard.py          # marimo dashboard  [done]
  tests/                  # Unit tests and live integration tests
```

Rust and Python share root `tests/fixtures/gym-v1/`: committed binary messages
plus JSON expectations. Both sides read the same bytes.

### Changing dependencies

```sh
uv add some-package              # a runtime dependency
uv add --optional train …        # into an extra
uv add --dev pytest-xdist        # into the dev group
uv lock --upgrade                # refresh every pin
```

Each of those rewrites `uv.lock`. Commit it with the change, or the next
person resolves something different from you.

## Ownership and reuse

DodgeAI remains a separate PICO-8 project. Adapt only the needed model and
training code locally, keeping source revision and path attribution and any
applicable notices. The Royale trainer must not import `dodge`, install
DodgeAI, reference a sibling checkout, or require a cartridge or a
pre-existing checkpoint.

Dependencies and the virtual environment belong here. Training configuration,
checkpoints and logs also live here by default, with generated artifacts
ignored by Git. Python is optional for the native and browser game builds: the
game neither imports nor needs any of this.

Defaults are eight environments, 1,024 rollout steps, 100 enemies, 3,600
maximum frames, a 24-frame prediction hold, and two Rust worker threads. The
policy executes one action per simulation frame; the prediction hold does not
change that.

See the [design](../docs/superpowers/specs/2026-09-15-velocity-flow-royale-design.md)
and [protocol v1](../docs/superpowers/specs/velocity-flow-royale-protocol-v1.md).
