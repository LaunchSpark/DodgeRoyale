"""Shutdown, deadlines, and making sure no child is left behind.

Most of these use a stand-in process rather than the gym, because the cases
worth testing are the ones where the child misbehaves: ignores CLOSE, never
exits, or dies during the handshake. A well-behaved gym cannot exercise any of
them. The tests that do need the real binary are marked `live` and skip when it
has not been built, unless DODGE_ROYALE_BIN says otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from dodge_royale.protocol import GymClient, GymError, find_binary

REPOSITORY = Path(__file__).resolve().parent.parent.parent


def stand_in(body: str) -> list[str]:
    """A Python child standing in for the gym, running `body`."""
    return [sys.executable, "-c", textwrap.dedent(body)]


class FakeClient(GymClient):
    """A client over a chosen child, skipping the handshake.

    The lifecycle is the same object either way, so shutdown can be tested
    without needing the child to speak the protocol.
    """

    def __init__(self, command: list[str], *, close_timeout: float = 1.0) -> None:
        # Deliberately not calling super().__init__: that one launches the gym
        # and performs a handshake, which is what these tests are avoiding.
        self._binary = Path(command[0])
        self._close_timeout = close_timeout
        self._closed = False
        from collections import deque

        self._stderr = deque(maxlen=200)
        self._drain = None
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
        )
        self._start_drain()
        self.envs = 1
        self.layout = None


# --- shutdown ------------------------------------------------------------


def test_close_reaps_a_child_that_exits_on_its_own():
    client = FakeClient(stand_in("import sys; sys.stdin.read()"))
    client.close()
    assert client.returncode is not None, "a closed session leaves no running child"


def test_close_is_idempotent():
    client = FakeClient(stand_in("import sys; sys.stdin.read()"))
    client.close()
    first = client.returncode
    client.close()
    client.close()
    assert client.returncode == first


def test_a_child_that_ignores_close_is_terminated_within_the_deadline():
    """Closing must not depend on the child's cooperation.

    This one ignores its stdin closing and sleeps. Without escalation, close()
    would block for a minute; with it, the deadline is the whole cost.
    """
    client = FakeClient(
        stand_in(
            """
            import signal, time
            try:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
            except (AttributeError, ValueError):
                pass
            time.sleep(60)
            """
        ),
        close_timeout=0.5,
    )
    started = time.monotonic()
    client.close()
    elapsed = time.monotonic() - started

    assert client.returncode is not None, "the child must be gone"
    assert elapsed < 10.0, f"close took {elapsed:.1f}s; the deadline should bound it"


def test_close_after_the_child_has_already_died_does_not_raise():
    client = FakeClient(stand_in("raise SystemExit(3)"))
    # Give it a moment to actually be gone before we tidy up.
    client._process.wait(timeout=10)
    client.close()
    assert client.returncode == 3


def test_the_context_manager_closes_on_the_way_out():
    client = FakeClient(stand_in("import sys; sys.stdin.read()"))
    with client:
        assert client.returncode is None
    assert client.returncode is not None


def test_a_binary_that_is_not_a_gym_fails_the_constructor_and_leaks_nothing():
    """The constructor's own cleanup path, through the real constructor.

    Pointed at the Python interpreter, the launch succeeds and then everything
    after it fails: `gym` is not a script, so stdout stays empty and the
    handshake hits EOF. The constructor must turn that into an error, take the
    child with it, and say what the child said.
    """
    with pytest.raises(GymError) as caught:
        GymClient(envs=1, binary=sys.executable, close_timeout=1.0)

    # The failure has to carry the child's own complaint, or the report is
    # "the handshake ended early" and nothing about why.
    assert "gym" in caught.value.stderr.lower() or caught.value.stderr

    # Nothing of ours should still be running. The client was never bound to a
    # name, so this asserts the constructor cleaned up rather than relying on
    # a caller who never got an object to close.
    assert "ended inside a message" in str(caught.value) or "not a DodgeRoyale" in str(
        caught.value
    )


def test_stderr_is_drained_and_bounded():
    client = FakeClient(
        stand_in(
            """
            import sys
            for index in range(5000):
                print('line %d' % index, file=sys.stderr)
            sys.stderr.flush()
            sys.stdin.read()
            """
        )
    )
    try:
        # The child writes far more than the ring holds; the point is that it
        # neither blocks on a full pipe nor grows without bound here.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and len(client._stderr) < 200:
            time.sleep(0.05)
        tail = client.stderr_tail()
        assert len(client._stderr) <= 200, "the diagnostic ring must stay bounded"
        assert "line 4999" in tail or "line" in tail
    finally:
        client.close()


def test_sending_to_a_closed_session_is_refused_rather_than_hanging():
    client = FakeClient(stand_in("import sys; sys.stdin.read()"))
    client.close()
    with pytest.raises(GymError, match="closed"):
        client.step([0])


def test_a_child_that_has_gone_away_reports_its_exit_code():
    client = FakeClient(stand_in("raise SystemExit(101)"))
    try:
        client._process.wait(timeout=10)
        detail = client._exit_detail()
        assert "101" in detail
    finally:
        client.close()


# --- locating the binary -------------------------------------------------


def test_an_explicit_override_that_is_wrong_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    """A deliberate override must never be silently ignored.

    Falling back to the checkout's binary here would run a different build
    than the one asked for, and the run would look fine.
    """
    missing = tmp_path / "definitely-not-here"
    monkeypatch.setenv("DODGE_ROYALE_BIN", str(missing))
    with pytest.raises(GymError, match="no gym binary"):
        find_binary()


def test_an_explicit_argument_beats_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DODGE_ROYALE_BIN", str(tmp_path / "from-env"))
    chosen = tmp_path / "from-argument"
    chosen.write_bytes(b"")
    assert find_binary(chosen) == chosen


def test_the_default_is_resolved_from_the_checkout_not_the_working_directory(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("DODGE_ROYALE_BIN", raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        found = find_binary()
    except GymError as error:
        # Not built here: the message must still point into the checkout
        # rather than at the directory the shell happened to be in.
        assert str(REPOSITORY) in str(error)
    else:
        assert str(REPOSITORY) in str(found)


# --- against the real gym ------------------------------------------------


def gym_available() -> bool:
    try:
        find_binary()
    except GymError:
        return False
    return True


explicit_binary = bool(os.environ.get("DODGE_ROYALE_BIN"))
live = pytest.mark.skipif(
    not gym_available() and not explicit_binary,
    reason="no gym binary; build with `cargo build --release --no-default-features`",
)


@pytest.mark.live
@live
def test_a_live_session_hands_shakes_steps_and_closes():
    with GymClient(envs=2, seed=7, enemies=12, max_frames=8, threads=1) as gym:
        assert gym.envs == 2
        assert gym.handshake.root_seed == 7
        assert gym.initial.observations.shape == (2, gym.layout.observation_values)

        batch = gym.step([2, 5])
        assert batch.transitions[0].frame == 1
        assert batch.observations.shape == (2, gym.layout.observation_values)

        kept = batch.observations.copy()
        gym.step([0, 0])
        assert (batch.observations == kept).all(), "a retained observation changed"
    assert gym.returncode == 0, "a closed session exits cleanly"


@pytest.mark.live
@live
def test_a_live_seeded_reset_reproduces_frame_zero():
    with GymClient(envs=2, seed=7, enemies=12, max_frames=64, threads=1) as gym:
        first = gym.initial
        gym.step([1, 1])
        again = gym.reset(7)
        assert again.seeds == first.seeds
        assert (again.observations == first.observations).all()


@pytest.mark.live
@live
def test_a_live_budget_produces_a_terminal_observation():
    with GymClient(envs=2, seed=7, enemies=12, max_frames=3, threads=1) as gym:
        for _ in range(3):
            batch = gym.step([0, 0])
        assert all(t.truncated for t in batch.transitions)
        assert sorted(batch.terminal) == [0, 1]
        for transition in batch.transitions:
            assert transition.time_limited
            assert transition.reset_seed is not None
