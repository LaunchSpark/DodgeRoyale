"""Shared fixtures: the committed golden messages, and where to find them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Imported here, before any test runs, for its side effect: on Windows marimo
# installs a selector event loop policy, and a selector loop cannot spawn a
# subprocess, which is how Playwright starts a browser. Importing it during a
# test would flip the policy out from under a later browser test; importing it
# now means `pytest_collection_finish` can put the policy back once and nothing
# changes it again. Optional, because a protocol-only install has no marimo.
try:  # noqa: SIM105
    import marimo  # noqa: F401
except ImportError:
    pass

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


def pytest_collection_finish(session):
    """Give Playwright back an event loop that can start a browser.

    Importing marimo switches Windows to a selector event loop policy, and a
    selector loop cannot spawn subprocesses, so Playwright fails with a bare
    `NotImplementedError`. It only happens when some earlier module has
    imported the notebook, so the browser suite passes on its own and fails in
    a full run -- exactly the shape that gets mistaken for flakiness.

    A collection hook rather than a fixture: this has to happen after every
    test module has been imported and before any fixture runs, and the
    `playwright` session fixture is not something an autouse fixture is
    ordered ahead of.
    """
    if sys.platform != "win32":
        return
    import asyncio

    if isinstance(
        asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy
    ):
        return
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    # The policy alone is not enough: marimo also installs a selector *loop*,
    # and Playwright uses whatever loop is already current. Replace it with one
    # the new policy built, which is the part that can spawn a browser.
    asyncio.set_event_loop(asyncio.new_event_loop())
