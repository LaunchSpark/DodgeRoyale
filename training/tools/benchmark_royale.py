"""Time each stage of a training step separately, and the baselines beside it.

    uv run python tools/benchmark_royale.py                 # 8 and 64 envs
    uv run python tools/benchmark_royale.py --envs 8        # one size
    uv run python tools/benchmark_royale.py --skip-baselines

Five stages, measured apart so a slow run can be attributed rather than
guessed at:

* **simulate + encode** -- the Rust side, reported by `benches/gym_throughput.rs`
  and summarised here rather than re-measured, because the gym does not expose
  the two separately over the protocol.
* **transport** -- a full STEP round trip through the pipe, with the arenas'
  own time subtracted out.
* **inference** -- a forward pass over one rollout's observations.
* **optimization** -- one PPO update over a filled rollout buffer.
* **end to end** -- `learn`, which is all of the above plus coordination.

The 64-env pipeline is deliberately short. A 1,024-step rollout at 64 envs is
about 7 GiB of observations before any training tensors, so the rollout here
is bounded and the memory it *would* need is reported rather than allocated.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from dodge_royale.metrics import MetricsCollector
from dodge_royale.policies import ARCHITECTURES, ROYALE_ARCHITECTURE
from dodge_royale.protocol import find_binary
from dodge_royale.training import SessionConfig, build_model, minibatch_size, session

#: Rollout steps used for the pipeline stages. Short on purpose: this measures
#: rate, and a long rollout would only buy precision at the cost of memory.
BENCH_STEPS = 16

#: Seeds each baseline policy is evaluated over.
BASELINE_SEEDS = (11, 12, 13, 14)


def observation_bytes(values: int, envs: int, steps: int) -> int:
    """What a rollout buffer of this shape costs, in bytes, before training."""
    return values * envs * steps * 4


@dataclass
class Environment:
    """Everything needed to tell whether a later number is comparable."""

    revision: str
    dirty: bool
    python: str
    torch: str
    cuda_available: bool
    device_name: str | None
    cpu: str
    logical_cores: int
    ram_gib: float
    os: str
    rustc: str
    binary: str

    @classmethod
    def capture(cls) -> "Environment":
        def run(*command: str) -> str:
            try:
                return subprocess.run(
                    command, capture_output=True, text=True, timeout=60
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):  # pragma: no cover
                return "unavailable"

        root = Path(__file__).resolve().parent.parent.parent
        revision = run("git", "-C", str(root), "rev-parse", "HEAD")
        status = run("git", "-C", str(root), "status", "--porcelain")
        try:
            import psutil  # noqa: F401

            ram = 0.0
        except ImportError:
            ram = 0.0
        if sys.platform == "win32":
            ram_text = run(
                "powershell", "-NoProfile", "-Command",
                "[math]::Round((Get-CimInstance Win32_OperatingSystem)"
                ".TotalVisibleMemorySize/1MB,1)",
            )
            try:
                ram = float(ram_text)
            except ValueError:  # pragma: no cover
                ram = 0.0

        return cls(
            revision=revision or "unknown",
            dirty=bool(status),
            python=sys.version.split()[0],
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            device_name=(
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            cpu=platform.processor() or platform.machine(),
            logical_cores=__import__("os").cpu_count() or 0,
            ram_gib=ram,
            os=f"{platform.system()} {platform.release()}",
            rustc=rustc_version(run),
            binary=str(find_binary()),
        )


def rustc_version(run) -> str:
    """`rustc --version`, looking where rustup puts it if PATH lacks it."""
    version = run("rustc", "--version")
    if version and version != "unavailable":
        return version
    candidate = Path.home() / ".cargo" / "bin" / (
        "rustc.exe" if sys.platform == "win32" else "rustc"
    )
    return run(str(candidate), "--version") if candidate.exists() else "unavailable"


@dataclass
class StageResult:
    name: str
    seconds_per_batch: float
    env_steps_per_second: float
    note: str = ""


@dataclass
class SizeResult:
    envs: int
    workers: int
    enemies: int
    rollout_steps: int
    minibatch: int
    device: str
    stages: list[StageResult] = field(default_factory=list)
    rollout_observation_bytes: int = 0
    full_rollout_observation_bytes: int = 0
    peak_host_mib: float = 0.0
    peak_device_mib: float = 0.0


def synchronize(device: str) -> None:
    """Make accelerator work actually finish before the clock is read."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_transport(env, steps: int) -> tuple[float, float]:
    """Seconds per STEP round trip, and the share that is not the arenas.

    The gym simulates and encodes before it answers, so a round trip includes
    both. Reporting the whole thing as "transport" would be wrong, which is
    why the Rust benchmark's own split is quoted beside it.
    """
    actions = np.zeros(env.num_envs, dtype=np.int64)
    for _ in range(4):  # warm the pipe and the arenas
        env.step(actions)
    start = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    elapsed = time.perf_counter() - start
    return elapsed / steps, env.num_envs * steps / elapsed


def measure_inference(model, observations: np.ndarray, repeats: int = 20) -> float:
    """Seconds for one forward pass over a rollout's worth of observations."""
    device = model.device
    tensor = torch.as_tensor(observations).to(device)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for _ in range(3):
            model.policy(tensor)
        synchronize(str(device))
        start = time.perf_counter()
        for _ in range(repeats):
            model.policy(tensor)
        synchronize(str(device))
        return (time.perf_counter() - start) / repeats


def measure_optimization(model, env, steps: int) -> float:
    """Seconds for one PPO update over a filled rollout buffer.

    `train` is timed on its own, after `collect_rollouts` has filled the
    buffer, so the number is the optimizer's and not the simulation's.
    """
    # SB3's own setup, rather than hand-wiring `_last_obs`: `collect_rollouts`
    # also needs the episode-info buffer and the callback bound to the model,
    # and `_setup_learn` is the only thing that builds all of it consistently.
    total, callback = model._setup_learn(
        total_timesteps=env.num_envs * steps,
        callback=MetricsCollector(),
        reset_num_timesteps=True,
    )
    callback.on_training_start(locals(), globals())

    def fill() -> None:
        model.rollout_buffer.reset()
        model.collect_rollouts(
            env, callback, model.rollout_buffer, n_rollout_steps=steps
        )

    # The first `train` on CUDA pays for kernel selection and autotuning, which
    # measured 15.6 s here against a steady state near 0.1 s. Timing it would
    # report the warm-up as the cost of an update.
    fill()
    model.train()
    synchronize(str(model.device))

    fill()
    synchronize(str(model.device))
    start = time.perf_counter()
    model.train()
    synchronize(str(model.device))
    return time.perf_counter() - start


def peak_device_mib() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def peak_host_mib() -> float:
    """Peak resident set of this process, in MiB."""
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss
    return peak / (1024**2)


def benchmark_size(envs: int, *, enemies: int, threads: int, device: str) -> SizeResult:
    config = SessionConfig(
        envs=envs,
        enemies=enemies,
        threads=threads,
        max_frames=3600,
        n_steps=BENCH_STEPS,
        n_epochs=1,
        total_timesteps=envs * BENCH_STEPS * 2,
        device=device,
    )
    batch = minibatch_size(config.rollout_samples(), config.minibatch_cap)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = SizeResult(
        envs=envs,
        workers=threads,
        enemies=enemies,
        rollout_steps=BENCH_STEPS,
        minibatch=batch,
        device=device,
    )

    with session(config) as env:
        values = env.layout.observation_values
        result.rollout_observation_bytes = observation_bytes(values, envs, BENCH_STEPS)
        result.full_rollout_observation_bytes = observation_bytes(values, envs, 1024)

        seconds, rate = measure_transport(env, BENCH_STEPS)
        result.stages.append(
            StageResult(
                "round trip (simulate + encode + transport)",
                seconds,
                rate,
                "one STEP answered by the gym; the Rust benchmark splits it further",
            )
        )

        model = build_model(config, env)
        actual_device = str(next(model.policy.parameters()).device)
        result.device = actual_device

        observations = env.reset()
        forward = measure_inference(model, observations)
        result.stages.append(
            StageResult("inference (one batch forward)", forward, envs / forward)
        )

        update = measure_optimization(model, env, BENCH_STEPS)
        result.stages.append(
            StageResult(
                "optimization (one PPO update)",
                update,
                envs * BENCH_STEPS / update,
                f"{BENCH_STEPS} steps x {envs} envs, minibatch {batch}, 1 epoch",
            )
        )

        metrics = MetricsCollector()
        start = time.perf_counter()
        model.learn(total_timesteps=config.total_timesteps, callback=metrics)
        synchronize(actual_device)
        elapsed = time.perf_counter() - start
        snapshot = metrics.snapshot()
        # Per batch, like every other row: `learn` covers many batch steps, so
        # reporting its total in a per-batch column would make it look ~30x
        # more expensive than the round trip it is mostly made of.
        batches = max(snapshot.timesteps // envs, 1)
        result.stages.append(
            StageResult(
                "end to end (learn)",
                elapsed / batches,
                snapshot.timesteps / elapsed,
                f"{snapshot.timesteps} env steps over {batches} batch steps and "
                f"{snapshot.updates} updates; {elapsed:.2f} s total",
            )
        )

    result.peak_device_mib = peak_device_mib()
    result.peak_host_mib = peak_host_mib()
    return result


# --- baselines -------------------------------------------------------------


@dataclass
class Baseline:
    policy: str
    episodes: int
    mean_frames: float
    median_frames: float
    deaths: int
    timeouts: int

    @property
    def censored(self) -> float:
        """Share of episodes that hit the budget rather than ending.

        A timeout is a censored observation of survival, not evidence the
        policy is good: it only says the episode outlasted the clock.
        """
        return self.timeouts / self.episodes if self.episodes else 0.0


def evaluate_baseline(
    name: str, *, enemies: int, max_frames: int, seeds=BASELINE_SEEDS
) -> Baseline:
    """Run a fixed policy over a bounded seed suite."""
    from dodge_royale.rewards import Rewards
    from dodge_royale.vec_env import RoyaleVecEnv

    lengths: list[int] = []
    deaths = timeouts = 0
    generator = np.random.default_rng(0)

    for seed in seeds:
        with RoyaleVecEnv(
            envs=2, enemies=enemies, max_frames=max_frames, threads=2,
            seed=seed, rewards=Rewards(),
        ) as env:
            wanted = 4  # episodes per seed
            finished = 0
            while finished < wanted:
                if name == "idle":
                    actions = np.zeros(env.num_envs, dtype=np.int64)
                else:
                    actions = generator.integers(
                        0, env.action_space.n, size=env.num_envs
                    )
                _, _, _, infos = env.step(actions)
                for info in infos:
                    summary = info.get("episode_summary")
                    if summary is None:
                        continue
                    finished += 1
                    lengths.append(int(summary["frames"]))
                    if summary["died"]:
                        deaths += 1
                    else:
                        timeouts += 1

    return Baseline(
        policy=name,
        episodes=len(lengths),
        mean_frames=float(np.mean(lengths)) if lengths else 0.0,
        median_frames=float(np.median(lengths)) if lengths else 0.0,
        deaths=deaths,
        timeouts=timeouts,
    )


# --- reporting -------------------------------------------------------------


def report(environment: Environment, sizes: list[SizeResult], baselines: list[Baseline]) -> str:
    lines = ["# Benchmark results", ""]
    lines.append(f"Revision `{environment.revision[:12]}`"
                 + (" (working tree dirty)" if environment.dirty else "")
                 + f", {environment.os}.")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    for key, value in asdict(environment).items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines.append("")

    for size in sizes:
        lines.append(f"## {size.envs} envs")
        lines.append("")
        lines.append(
            f"{size.workers} gym workers, {size.enemies} enemies, rollout "
            f"{size.rollout_steps} steps, minibatch {size.minibatch}, device "
            f"`{size.device}`."
        )
        lines.append("")
        lines.append("| Stage | Per batch | Env steps/s | Note |")
        lines.append("|---|---|---|---|")
        for stage in size.stages:
            lines.append(
                f"| {stage.name} | {stage.seconds_per_batch * 1e3:.2f} ms | "
                f"{stage.env_steps_per_second:,.0f} | {stage.note} |"
            )
        lines.append("")
        lines.append(
            f"Rollout observations at {size.rollout_steps} steps: "
            f"{size.rollout_observation_bytes / 1024**2:,.0f} MiB. "
            f"At 1,024 steps they would be "
            f"{size.full_rollout_observation_bytes / 1024**3:,.2f} GiB, which is "
            f"why this benchmark does not allocate one."
        )
        lines.append(
            f"Peak host working set {size.peak_host_mib:,.0f} MiB; "
            f"peak device allocation {size.peak_device_mib:,.0f} MiB."
        )
        lines.append("")

    if baselines:
        lines.append("## Baselines")
        lines.append("")
        lines.append(
            "Fixed policies over the same bounded seed suite "
            f"{list(BASELINE_SEEDS)}. A timeout is a censored survival "
            "observation, not evidence of skill."
        )
        lines.append("")
        lines.append("| Policy | Episodes | Mean frames | Median | Deaths | Timeouts | Censored |")
        lines.append("|---|---|---|---|---|---|---|")
        for baseline in baselines:
            lines.append(
                f"| {baseline.policy} | {baseline.episodes} | "
                f"{baseline.mean_frames:,.1f} | {baseline.median_frames:,.1f} | "
                f"{baseline.deaths} | {baseline.timeouts} | {baseline.censored:.0%} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, nargs="*", default=[8, 64])
    parser.add_argument("--enemies", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline-enemies", type=int, default=100)
    parser.add_argument("--baseline-max-frames", type=int, default=3600)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    environment = Environment.capture()
    print(f"revision {environment.revision[:12]}"
          + (" (dirty)" if environment.dirty else ""), file=sys.stderr)

    sizes = []
    for envs in args.envs:
        print(f"measuring {envs} envs...", file=sys.stderr)
        sizes.append(
            benchmark_size(
                envs, enemies=args.enemies, threads=args.threads, device=args.device
            )
        )

    baselines = []
    if not args.skip_baselines:
        for name in ("idle", "random"):
            print(f"evaluating the {name} baseline...", file=sys.stderr)
            baselines.append(
                evaluate_baseline(
                    name,
                    enemies=args.baseline_enemies,
                    max_frames=args.baseline_max_frames,
                )
            )

    text = report(environment, sizes, baselines)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "environment": asdict(environment),
                    "sizes": [asdict(size) for size in sizes],
                    "baselines": [asdict(base) for base in baselines],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
