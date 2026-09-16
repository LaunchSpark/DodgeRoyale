# DodgeRoyale training

This directory owns the Python trainer for DodgeRoyale: the gym protocol client,
SB3 vector environment, VelocityFlow policy, PPO session, CLI, and dashboard.

**Implemented so far:** `protocol.py` (the protocol-v1 client), `vec_env.py`
(`RoyaleVecEnv`), `rewards.py`, `telemetry.py`, `velocity.py` (the extractor),
`policies.py`, `training.py` (the PPO session) and `train.py` (the CLI), with
their tests. Only `dashboard.py` is still pending, from Task 11 of the
[implementation plan](../docs/superpowers/plans/2026-09-15-velocity-flow-royale-implementation.md);
the layout and entry points below are the contract for that work.

## Running the tests

```sh
cd training
python -m pytest              # no Rust, no gym process, no PyTorch
python -m pytest -m live      # also drives a built gym binary
```

The client depends on NumPy and the standard library, and nothing else. SB3,
Gymnasium and PyTorch are the `train` extra, so a wire-format change can be
tested without a multi-gigabyte download. Fixture tests read the committed
messages in `../tests/fixtures/gym-v1/` and never build or run Rust. Live tests
skip when no binary is built, but fail rather than skip when `DODGE_ROYALE_BIN`
is set and broken.

## Training

```sh
python -m dodge_royale.train --dry-run      # what would run, no gym launched
python -m dodge_royale.train --check-env    # reset and step the batch, then exit
python -m dodge_royale.train                # 8 envs, 1024 steps, 100 enemies
python -m dodge_royale.train --resume-latest
```

Royale only: there is no `--game` switch. `--hold-frames` is the prediction
hold, not an action repeat; the policy decides every frame either way.

## Planned layout

```text
training/
  pyproject.toml           # Package, Python version, dependencies, test configuration
  rewards.json            # Reward controls that apply to Royale
  dodge_royale/
    protocol.py           # Protocol-v1 client and Layout  [done]
    vec_env.py            # RoyaleVecEnv  [done]
    velocity.py           # VelocityFlowRoyaleExtractor  [done]
    policies.py           # Local policy classes and checkpoint registration  [done]
    rewards.py            # Royale reward configuration  [done]
    telemetry.py          # Training events and duration reporting  [done]
    training.py           # PPO session and process lifecycle  [done]
    train.py              # CLI entry point  [done]
    dashboard.py          # Dashboard entry point
  tests/                  # Unit tests and live integration tests
  tools/                  # End-to-end benchmark
```

Rust and Python share root `tests/fixtures/gym-v1/`: committed binary messages
plus JSON expectations. Fixture tests run without compiling Rust. Live integration
tests launch the gym binary built from the same checkout.

## Ownership and reuse

DodgeAI remains a separate PICO-8 project. Adapt only the needed model and training
code locally, keeping source revision/path attribution and applicable notices.
The Royale trainer must not import `dodge`, install DodgeAI, reference a sibling
checkout, or require a cartridge or pre-existing checkpoint. Do not move or alter
DodgeAI's pending work.

Dependencies and the virtual environment belong here. Training configuration,
checkpoints, and logs also live here by default, with generated artifacts ignored
by Git. Python is optional for the native and browser game builds.

## Entry points to implement

After Task 8 defines installation and dependencies, the installed package exposes
`python -m dodge_royale.train` and `python -m dodge_royale.dashboard`. Both train
Royale directly; no game selector is needed. Task 11 supplies the commands.

The client honors `DODGE_ROYALE_BIN`. In a source checkout its default is the
repository's `target/release/dodge-royale` executable, with the platform suffix.
An external installation or custom Cargo target directory uses the explicit
override. Resolve paths from the installed module/check-out location, not the
shell's working directory. The trainer launches the binary directly with `gym`.

Initial defaults remain eight environments, 1,024 rollout steps, 100 enemies,
3,600 maximum frames, a 24-frame prediction hold, and two Rust worker threads.
The policy executes one action per simulation frame. Prediction hold duration
does not change the decision interval.

See the [design](../docs/superpowers/specs/2026-09-15-velocity-flow-royale-design.md)
and [protocol v1](../docs/superpowers/specs/velocity-flow-royale-protocol-v1.md).
