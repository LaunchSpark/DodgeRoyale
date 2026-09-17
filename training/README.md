# DodgeRoyale training

The Python trainer for DodgeRoyale: the gym protocol client, the SB3 vector
environment, the VelocityFlow policy, the PPO session, the CLI, and the
dashboard. It drives the Rust simulation as a child process, so a working
setup needs both halves of the repository.

**Implemented so far:** `protocol.py` (the protocol-v1 client), `vec_env.py`
(`RoyaleVecEnv`), `rewards.py`, `telemetry.py`, `velocity.py` (the extractor),
`policies.py`, `training.py` (the PPO session) and `train.py` (the CLI), with
their tests. `dashboard.py` is still pending, from Task 11 of the
[implementation plan](../docs/superpowers/plans/2026-09-15-velocity-flow-royale-implementation.md);
marimo is already installed for it.

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

The first build takes a few minutes. A debug build works too, but it is
roughly an order of magnitude slower per simulated frame, so the wait pays for
itself in the first training run.

### 3. Create the Python environment

From **this** directory, pick the line that matches what you are doing:

```sh
cd training

uv sync                                      # protocol client + pytest only
uv sync --extra train                        # add the learner (SB3, Gymnasium, torch)
uv sync --extra dashboard --extra cu126      # everything, PyTorch with CUDA
uv sync --extra dashboard --extra cpu        # everything, PyTorch without CUDA
```

Any of these creates `training/.venv`, installs the exact versions pinned in
`uv.lock`, and installs `dodge_royale` itself in editable mode. Two
collaborators running the same line get the same environment.

The first line is deliberately small. `protocol.py` knows nothing about SB3,
Gymnasium or PyTorch, so a change to the wire format can be tested without a
multi-gigabyte download.

**On Windows, put the cache on the same drive as the checkout.** uv hardlinks
from its cache into `.venv` when both are on one filesystem and copies when
they are not, which for PyTorch is 2.4 GB copied on every sync. If `uv cache
dir` prints a `C:` path and you work on another drive, set `UV_CACHE_DIR`
alongside your projects once and installs become near-instant:

```sh
setx UV_CACHE_DIR D:\.uv-cache
```

> **`uv sync --all-extras` does not work here, on purpose.** `cpu` and `cu126`
> are declared as conflicting extras, because they are two builds of the same
> package, so asking for every extra at once asks for both. uv says so rather
> than silently picking one. Name the extras you want instead.

**Which torch build?** If you have an NVIDIA GPU, use `cu126`. Without the
extras, PyPI decides, and on Windows PyPI ships a CPU-only wheel — training
still runs, about an order of magnitude slower. Check what you got:

```sh
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.14.0+cu126 True` is a GPU install. `2.14.0+cpu False` or `2.14.0 False` is
not. If `cu126` prints `False`, the graphics driver is the problem rather than
this project.

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
uv run pytest                 # everything; live tests skip without a binary
uv run pytest -m live         # only the tests that drive a real gym
uv run pytest -m "not live"   # explicitly skip those
```

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

## The dashboard

Not written yet. marimo is in the `dashboard` extra and installed, so when
`dodge_royale/dashboard.py` arrives it will run with:

```sh
uv run marimo edit dodge_royale/dashboard.py
```

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
    dashboard.py          # marimo dashboard  [pending]
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
