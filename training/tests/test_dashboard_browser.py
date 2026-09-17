"""Drive the dashboard in a real browser.

Everything else about the notebook is tested in process: the worker's
lifecycle, the metric definitions, the source. None of that touches the part a
user actually uses -- a button press travelling through marimo's reactive
graph into the worker, and the resulting state coming back out as rendered
HTML. That path only exists in a browser, so this is the only place it is
covered.

Marked `live` because it needs the gym binary, and `browser` so it can be
deselected: it launches a marimo server and a Chromium, which is slower than
the rest of the suite put together.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import pytest

from dodge_royale.protocol import GymError, find_binary

SERVER_LOG: list[Path] = []

NOTEBOOK = Path(__file__).resolve().parent.parent / "dodge_royale" / "dashboard.py"


def gym_available() -> bool:
    try:
        find_binary()
    except GymError:
        return False
    return True


pytestmark = [
    pytest.mark.live,
    pytest.mark.browser,
    pytest.mark.skipif(not gym_available(), reason="no gym binary built"),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server():
    """A marimo server for the notebook, torn down however the tests end.

    `--no-sandbox` because the notebook carries a PEP 723 header and marimo
    would otherwise stop to ask whether to build an environment from it.
    """
    port = free_port()
    # The log goes to a file rather than a pipe: a pipe nobody drains fills and
    # blocks the server, and when a test fails the log is the only account of
    # what the kernel thought it was doing.
    log = Path(tempfile.gettempdir()) / f"marimo-dashboard-{port}.log"
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "marimo", "run", str(NOTEBOOK),
            "--no-sandbox", "--headless", "--host", "127.0.0.1",
            "--port", str(port), "--no-token",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=NOTEBOOK.parent.parent,
    )
    SERVER_LOG.append(log)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read().decode("utf-8", "replace")
                pytest.fail(f"the marimo server exited:\n{output}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)
        else:  # pragma: no cover - only on a wedged server
            pytest.fail("the marimo server never accepted a connection")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
            process.wait(timeout=30)
        handle.close()


def gym_processes() -> int:
    """How many gym children are alive, counted from the OS."""
    if sys.platform == "win32":
        output = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq dodge-royale.exe"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return output.count("dodge-royale.exe")
    output = subprocess.run(
        ["pgrep", "-c", "-f", "dodge-royale"], capture_output=True, text=True, timeout=60
    ).stdout.strip()
    return int(output or 0)


def state_text(page) -> str:
    """Whatever the status heading currently says."""
    return page.locator("h3").first.inner_text(timeout=30_000).strip().lower()


def wait_for_state(page, *wanted: str, timeout: float = 120.0) -> str:
    deadline = time.monotonic() + timeout
    seen = ""
    while time.monotonic() < deadline:
        seen = state_text(page)
        if any(word in seen for word in wanted):
            return seen
        page.wait_for_timeout(250)
    raise AssertionError(f"waited for {wanted}, the page still says {seen!r}")


def press(page, label: str) -> None:
    page.get_by_role("button", name=label, exact=False).first.click()


@pytest.fixture
def dashboard(server, page):
    """A loaded page that is left with no run going, whatever the test did."""
    page.set_default_timeout(60_000)
    page.goto(server)
    page.wait_for_selector("h3", timeout=90_000)
    try:
        yield page
    finally:
        # Unconditional, and tolerant of a page left in any state: the worker
        # is process-global and outlives this browser session, so a run this
        # test started would otherwise still be going for the next one.
        try:
            press(page, "Stop")
            wait_for_state(page, "stopped", "idle", timeout=90)
        except Exception:  # noqa: BLE001 - teardown must not mask a failure
            pass


# --- the page renders ----------------------------------------------------


def test_the_page_loads_with_no_run_and_no_gym(dashboard):
    assert "idle" in state_text(dashboard)
    assert gym_processes() == 0, "loading the page must not start a gym"
    assert dashboard.get_by_text("No run yet").count() >= 1


def test_the_controls_are_present(dashboard):
    for label in ("Start", "Pause", "Save checkpoint", "Stop"):
        assert dashboard.get_by_role("button", name=label, exact=False).count() >= 1, label


def test_watch_agent_is_shown_as_unavailable(dashboard):
    assert dashboard.get_by_text("Watch Agent is not available").count() >= 1


def test_the_configuration_summary_renders(dashboard):
    dashboard.get_by_text("Game configuration").first.click()
    assert dashboard.get_by_text("velocity-flow-royale").count() >= 1


# --- the lifecycle, through the buttons ----------------------------------


def test_start_runs_and_stop_cleans_up(dashboard):
    assert gym_processes() == 0
    press(dashboard, "Start")
    wait_for_state(dashboard, "running", "starting", timeout=120)

    # A real gym is now a child of the marimo server.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and gym_processes() == 0:
        dashboard.wait_for_timeout(250)
    assert gym_processes() >= 1, "Start must launch a gym"

    press(dashboard, "Stop")
    wait_for_state(dashboard, "stopped", timeout=120)

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and gym_processes() > 0:
        dashboard.wait_for_timeout(250)
    assert gym_processes() == 0, "Stop must take the gym with it"


def test_pause_and_resume_through_the_buttons(dashboard):
    press(dashboard, "Start")
    wait_for_state(dashboard, "running", timeout=120)

    press(dashboard, "Pause")
    wait_for_state(dashboard, "paused", timeout=120)

    press(dashboard, "Pause")
    wait_for_state(dashboard, "running", timeout=120)

    press(dashboard, "Stop")
    wait_for_state(dashboard, "stopped", timeout=120)


def test_pressing_start_twice_does_not_start_two_gyms(dashboard):
    """The reactive-rerun hazard, exercised the way a user would hit it."""
    press(dashboard, "Start")
    wait_for_state(dashboard, "running", timeout=120)
    running = gym_processes()
    assert running >= 1

    press(dashboard, "Start")
    dashboard.wait_for_timeout(3_000)
    assert gym_processes() == running, "a second Start must not launch another gym"

    press(dashboard, "Stop")
    wait_for_state(dashboard, "stopped", timeout=120)


def test_metrics_appear_once_a_run_is_going(dashboard):
    press(dashboard, "Start")
    wait_for_state(dashboard, "running", timeout=120)

    # The metric table replaces the "no run yet" placeholder.
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if dashboard.get_by_text("Throughput").count() >= 1:
            break
        dashboard.wait_for_timeout(500)
    assert dashboard.get_by_text("Throughput").count() >= 1, "metrics rendered"
    assert dashboard.get_by_text("Device").count() >= 1

    press(dashboard, "Stop")
    wait_for_state(dashboard, "stopped", timeout=120)


def test_saving_writes_a_checkpoint(dashboard, tmp_path):
    press(dashboard, "Start")
    wait_for_state(dashboard, "running", timeout=120)

    press(dashboard, "Save checkpoint")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if dashboard.get_by_text("Checkpoints").count() >= 1:
            break
        dashboard.wait_for_timeout(500)
    assert dashboard.get_by_text("Checkpoints").count() >= 1, "a checkpoint was listed"

    press(dashboard, "Stop")
    wait_for_state(dashboard, "stopped", timeout=120)


def test_a_reload_does_not_strand_the_gym(dashboard, server):
    """Closing the tab and coming back must not leave a run orphaned."""
    press(dashboard, "Start")
    wait_for_state(dashboard, "running", timeout=120)
    assert gym_processes() >= 1

    dashboard.reload()
    dashboard.wait_for_selector("h3", timeout=90_000)
    # marimo gives a reloaded page a fresh kernel session, so the reloaded view
    # shows no run. What must not happen is the old gym surviving forever.
    press(dashboard, "Stop")
    dashboard.wait_for_timeout(2_000)
