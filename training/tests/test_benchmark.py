"""Benchmark provenance must describe the measured code, not its own output."""

import hashlib
import subprocess
from pathlib import Path

from tools import benchmark_royale


def test_benchmark_ignores_its_output_files(monkeypatch):
    root = Path(benchmark_royale.__file__).resolve().parent.parent.parent
    report = root / "docs/superpowers/results/benchmark.md"
    data = report.with_suffix(".json")
    diff = report.with_suffix(".diff")
    outputs = (report, data, diff)

    def run(command, **_kwargs):
        if "rev-parse" in command:
            value = "revision"
        elif "status" in command:
            value = "\n".join(f" M {p.relative_to(root).as_posix()}" for p in outputs)
        else:
            value = ""
        return subprocess.CompletedProcess(command, 0, stdout=value)

    monkeypatch.setattr(benchmark_royale.subprocess, "run", run)
    monkeypatch.setattr(benchmark_royale, "find_binary", lambda: Path("gym.exe"))

    environment = benchmark_royale.Environment.capture(outputs)
    assert environment.revision == "revision"
    assert not environment.dirty
    assert environment.dirty_files == ()


def test_benchmark_diff_hash_matches_file_bytes_on_windows(monkeypatch, tmp_path):
    diff = "diff --git a/code b/code\n+line\n"
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=diff)

    monkeypatch.setattr(benchmark_royale.subprocess, "run", run)
    environment = benchmark_royale.Environment(
        revision="revision", dirty=True, python="3", torch="2", cuda_available=False,
        device_name=None, cpu="cpu", logical_cores=1, ram_gib=1.0, os="Windows",
        rustc="rustc", binary="gym.exe",
    )
    root = Path(benchmark_royale.__file__).resolve().parent.parent.parent
    output = root / "docs/superpowers/results/benchmark.md"
    environment.preserve_diff(tmp_path / "benchmark.md", (output,))

    saved = (tmp_path / "benchmark.diff").read_bytes()
    assert saved == diff.encode("utf-8")
    assert environment.diff_sha256 == hashlib.sha256(saved).hexdigest()
    assert ":(exclude)docs/superpowers/results/benchmark.md" in commands[0]
