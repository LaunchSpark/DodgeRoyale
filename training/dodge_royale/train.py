"""`python -m dodge_royale.train` -- the training CLI.

Royale only. There is no `--game` switch and no PICO-8 environment factory:
this trainer builds one thing, so a selector would be a knob with one setting
and a second code path to keep correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .history import history_path, read_history
from .metrics import MetricsCollector
from .policies import ARCHITECTURES, ROYALE_ARCHITECTURE
from .protocol import GymError, ProtocolError
from .training import (
    DEFAULTS,
    CheckpointWriter,
    SessionConfig,
    check_env,
    describe,
    session,
    train,
)

__all__ = ["build_parser", "config_from_args", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dodge_royale.train",
        description="Train a VelocityFlow policy to play DodgeRoyale.",
    )
    arena = parser.add_argument_group("arena")
    arena.add_argument("--envs", type=int, default=DEFAULTS.envs)
    arena.add_argument("--seed", type=int, default=DEFAULTS.seed)
    arena.add_argument("--enemies", type=int, default=DEFAULTS.enemies)
    arena.add_argument("--max-frames", type=int, default=DEFAULTS.max_frames)
    arena.add_argument(
        "--hold-frames",
        type=int,
        default=DEFAULTS.hold_frames,
        help=(
            "frames a predicted path holds its action before coasting. "
            "Not an action repeat: the policy still decides every frame."
        ),
    )
    arena.add_argument("--threads", type=int, default=DEFAULTS.threads,
                       help="Rust worker threads inside the gym")
    arena.add_argument("--binary", default=None,
                       help="gym executable; defaults to DODGE_ROYALE_BIN or this checkout")

    learner = parser.add_argument_group("learner")
    learner.add_argument("--n-steps", type=int, default=DEFAULTS.n_steps)
    learner.add_argument("--minibatch-cap", type=int, default=DEFAULTS.minibatch_cap,
                         help="upper bound on a minibatch, independent of rollout size")
    learner.add_argument("--n-epochs", type=int, default=DEFAULTS.n_epochs)
    learner.add_argument("--learning-rate", type=float, default=DEFAULTS.learning_rate)
    learner.add_argument("--total-timesteps", type=int, default=DEFAULTS.total_timesteps)
    learner.add_argument("--device", default=DEFAULTS.device)
    learner.add_argument("--architecture", default=ROYALE_ARCHITECTURE,
                         choices=sorted(ARCHITECTURES))

    run = parser.add_argument_group("run")
    run.add_argument("--rewards", default=None, help="path to a rewards.json")
    run.add_argument("--checkpoint-dir", default=DEFAULTS.checkpoint_dir)
    run.add_argument("--run-name", default=DEFAULTS.run_name)
    run.add_argument("--resume", default=None, help="a checkpoint to continue from")
    run.add_argument("--resume-latest", action="store_true",
                     help="continue from the newest checkpoint of this run name")

    parser.add_argument("--check-env", action="store_true",
                        help="reset and step the batch, report, and exit without training")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the configuration and exit without launching a gym")
    return parser


def config_from_args(args: argparse.Namespace) -> SessionConfig:
    return SessionConfig(
        envs=args.envs,
        seed=args.seed,
        enemies=args.enemies,
        max_frames=args.max_frames,
        hold_frames=args.hold_frames,
        threads=args.threads,
        n_steps=args.n_steps,
        minibatch_cap=args.minibatch_cap,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        total_timesteps=args.total_timesteps,
        device=args.device,
        binary=args.binary,
        rewards_path=args.rewards,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        architecture=args.architecture,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        config.validate()
    except ValueError as error:
        parser.error(str(error))
        return 2  # pragma: no cover - parser.error exits

    if args.resume and args.resume_latest:
        parser.error("--resume and --resume-latest ask for different checkpoints")

    resume: str | Path | None = args.resume
    if args.resume_latest:
        resume = CheckpointWriter.for_config(config).latest()
        if resume is None:
            parser.error(
                f"no checkpoint named {config.run_name}-* in {config.checkpoint_dir}"
            )

    if args.dry_run:
        print(describe(config))
        return 0

    try:
        if args.check_env:
            # A batched env, so this is a VecEnv smoke check rather than
            # Gymnasium's scalar checker, which would reject it for its shape
            # and tell us nothing about whether it works.
            with session(config) as env:
                check_env(env)
                print(describe(config, env.layout))
                print("\nthe batch resets and steps")
            return 0

        # The same collector the dashboard reads, so a metric added for a
        # chart is reported here too and neither can define one differently.
        metrics = MetricsCollector()
        path = train(config, resume=resume, callback=metrics)
    except (GymError, ProtocolError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"saved {path}")
    for definition, reading in metrics.snapshot().rows():
        print(f"  {definition.label:<14} {reading}")

    # Where the model's own past now lives. Printed because a history nobody
    # can find is not a history.
    episodes = read_history(history_path(path))
    if episodes:
        counts = sorted({record.enemies for record in episodes})
        print(
            f"  history        {len(episodes):,} episodes beside the checkpoint, "
            f"enemy counts {counts}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
