"""The actual WASM game can ask Python for actions and keep rendering."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import numpy as np
import pytest

from dodge_royale.browser_watch import GameWatch

pytestmark = [pytest.mark.browser, pytest.mark.live]

BUNDLE = (
    Path(__file__).resolve().parent.parent.parent
    / "target" / "bevy_web" / "web-release" / "dodge-royale"
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class RampField:
    """An extractor with the one thing that matters here: a field."""

    def field(self, observations):
        import torch

        # Row-major counting order, so a transposed or flipped field shows up
        # as the wrong numbers rather than as a plausible blur.
        return torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)


class FieldPolicy:
    features_extractor = RampField()

    def obs_to_tensor(self, observation):
        import torch

        return torch.as_tensor(observation.copy()), None


class RightPolicy:
    policy = FieldPolicy()

    def predict(self, observation, *, deterministic):
        assert observation.shape == (1, 28782)
        assert deterministic
        # Fifteen degrees: a heading the old nine-way action space could not
        # express, so the page arriving at it proves the whole path is a
        # vector and not an index somewhere in the middle.
        return np.array([[0.966, 0.259]], dtype=np.float32), None


def test_real_web_game_uses_the_policy_service(page, tmp_path):
    index = BUNDLE / "index.html"
    if not index.exists() or "dodgeObserve" not in index.read_text(encoding="utf-8"):
        pytest.skip("build the current web bundle with `bevy build --release web --bundle`")
    handler = partial(QuietHandler, directory=str(BUNDLE))
    web = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=web.serve_forever, daemon=True)
    thread.start()
    watch = GameWatch(tmp_path, web_url=f"http://127.0.0.1:{web.server_port}/")
    watch.model = RightPolicy()
    watch.update = 1
    try:
        page.goto(watch.url)
        if not page.evaluate("!!document.createElement('canvas').getContext('webgl2')"):
            pytest.skip("this browser has no WebGL 2")
        # A heading, not an index, and one the old nine-way space could not
        # express: proof the whole path is a vector end to end.
        page.wait_for_function(
            "window.dodgeAction && Math.abs(window.dodgeAction()[0] - 0.966) < 1e-3"
            " && Math.abs(window.dodgeAction()[1] - 0.259) < 1e-3",
            timeout=90_000,
        )
        page.wait_for_function("document.querySelector('#loading').hidden", timeout=90_000)
        assert watch.requests >= 1
        assert page.locator("#game").evaluate("canvas => canvas.width > 1 && canvas.height > 1")
        assert page.locator("body").evaluate("body => body.classList.contains('watch-mode')")
        assert page.locator("#game").evaluate("canvas => canvas.tabIndex") == -1
        assert page.locator(".topbar").is_hidden()
        assert page.locator(".footer").is_hidden()
        assert "update 1" in page.locator("#watch-status").inner_text()

        # The field the action came from reaches the page whole, on the
        # observation's own 64x64 lattice. Waited for rather than sampled: a
        # reply is asynchronous, and an episode ending clears the last one
        # until the next decision is made.
        page.wait_for_function(
            "window.dodgeField() && window.dodgeField().edge === 64", timeout=60_000
        )
        field = page.evaluate(
            "() => { const f = window.dodgeField();"
            " return f && { frame: f.frame, edge: f.edge, values: Array.from(f.values) }; }"
        )
        assert field is not None, "the reply carried no field"
        assert field["edge"] == 64
        assert len(field["values"]) == 64 * 64
        assert field["values"][:64] == list(range(64)), "the first row is the first row"
        assert field["values"][64] == 64, "and rows follow each other, not columns"

        # And the game consumed it: anchored to the frame whose observation
        # produced it, and uploaded. A field that arrives but never anchors
        # leaves this null, which is exactly the failure worth catching.
        page.wait_for_function("window.dodgeDrawnFrame !== null", timeout=30_000)
        drawn = page.evaluate("window.dodgeDrawnFrame")
        assert isinstance(drawn, int) and drawn >= 1
    finally:
        watch.close()
        web.shutdown()
        web.server_close()
        thread.join()
