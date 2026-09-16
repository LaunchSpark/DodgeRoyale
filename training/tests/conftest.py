"""Shared fixtures: the committed golden messages, and where to find them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# training/tests/conftest.py -> the repository root.
REPOSITORY = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPOSITORY / "tests" / "fixtures" / "gym-v1"


@pytest.fixture(scope="session")
def manifest() -> dict:
    """What the Rust emitter recorded beside the binary fixtures."""
    path = FIXTURES / "manifest.json"
    if not path.exists():
        pytest.fail(
            f"no fixture manifest at {path}. Regenerate with "
            "`cargo run --locked --no-default-features --example gym_fixtures`."
        )
    # Python's json is correctly rounded, so the manifest's decimals come back
    # as exactly the doubles the emitter widened its f32 values to.
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fixture_bytes() -> "callable[[str], bytes]":
    def read(name: str) -> bytes:
        return (FIXTURES / name).read_bytes()

    return read


def fixtures_of(manifest: dict, kind: str) -> list[dict]:
    return [entry for entry in manifest["fixtures"] if entry["kind"] == kind]
