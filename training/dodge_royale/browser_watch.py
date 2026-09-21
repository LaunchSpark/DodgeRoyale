"""Local inference bridge for the real browser game in the watch iframe.

The Bevy game owns the world and rendering. It sends its own encoded
observation here; this service returns the chosen action and the danger field
the policy chose it from. It is loopback-only, and never creates a second arena
that could drift away from the picture the user sees.

The field is what makes this worth watching. An action on its own says what the
policy did; the field says what it believed, cell by cell, about the 256-pixel
window it was looking at. A dodge into a bright cell is a bug in the field; a
dodge away from a dark one is a bug in the paths.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
import threading
from typing import Any
from urllib.parse import urlencode, urlsplit

import numpy as np

from .policies import checkpoint_architecture, checkpoint_layout
from .protocol import Layout, ProtocolError
from .snapshots import newest_snapshot

#: The field's edge in cells, two reserved bytes, then the direction the
#: policy chose, as x and y. Twelve bytes, so the field that follows starts on
#: a four-byte boundary and the browser can read it as a Float32Array over the
#: response buffer instead of copying 16 KB out of it sixty times a second.
#:
#: An edge of zero means this policy has no field to show: a checkpoint without
#: one still plays, it just has no thoughts to draw.
REPLY_HEADER = struct.Struct("<HHff")


class GameWatch:
    """Serve policy decisions to one browser game instance."""

    def __init__(self, snapshots: str | Path, *, web_url: str) -> None:
        self.snapshots = Path(snapshots)
        self.web_url = web_url.rstrip("/") + "/"
        parsed_url = urlsplit(self.web_url)
        if parsed_url.scheme != "http" or parsed_url.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("the watch game URL must be a local HTTP address")
        self.web_origin = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.layout: Layout | None = None
        self.model: Any = None
        self.update: int | None = None
        self.error: str | None = None
        self.requests = 0
        self._lock = threading.Lock()
        watch = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_OPTIONS(self) -> None:  # noqa: N802 - http.server API
                self.send_response(204)
                self._cors()
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Dodge-Layout")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Max-Age", "600")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                self.connection.settimeout(5.0)
                if self.path == "/episode":
                    with watch._lock:
                        watch._reload()
                    self._reply(204, b"")
                    return
                if self.path != "/infer":
                    self._reply(404, b"not found")
                    return
                try:
                    raw_layout = self.headers.get("X-Dodge-Layout", "")
                    if len(raw_layout) > 4096:
                        raise ValueError("layout header is too large")
                    layout = Layout.from_json(json.loads(raw_layout))
                    length = int(self.headers.get("Content-Length", "-1"))
                    expected = layout.observation_values * 4
                    if length != expected:
                        raise ValueError(f"observation has {length} bytes, expected {expected}")
                    observation = np.frombuffer(self.rfile.read(length), dtype="<f4")
                    if observation.size != layout.observation_values or not np.isfinite(observation).all():
                        raise ValueError("observation is truncated or non-finite")
                    with watch._lock:
                        if watch.layout is None:
                            watch.layout = layout
                            watch._reload()
                        else:
                            watch.layout.require_compatible(layout)
                        body = watch._decide(observation)
                        watch.requests += 1
                        update = watch.update
                        error = watch.error
                    self._reply(200, body, update=update, error=error)
                except (ValueError, TypeError, TimeoutError, ProtocolError) as error:
                    self._reply(400, str(error).encode("utf-8", "replace")[:1024])
                except Exception as error:  # noqa: BLE001 - keep the game responsive
                    watch.error = str(error)
                    self._reply(500, b"policy inference failed")

            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", watch.web_origin)
                self.send_header("Cache-Control", "no-store")

            def _reply(
                self, code: int, body: bytes, *, update: int | None = None,
                error: str | None = None,
            ) -> None:
                self.send_response(code)
                self._cors()
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/octet-stream")
                if update is not None:
                    self.send_header("X-Watch-Update", f"update {update}")
                if error is not None:
                    self.send_header("X-Watch-Error", json.dumps(error[:200], ensure_ascii=True))
                self.send_header("Access-Control-Expose-Headers", "X-Watch-Update, X-Watch-Error")
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def log_message(self, *_: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="watch-policy", daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        service = f"http://127.0.0.1:{self._server.server_port}/"
        return f"{self.web_url}?{urlencode({'watch': '1', 'policy_url': service})}"

    def set_scale(self, _scale: int) -> None:
        """The Bevy camera fits its iframe; observation zoom does not apply."""

    def _reload(self) -> None:
        if self.layout is None:
            return
        found = newest_snapshot(self.snapshots, after=self.update if self.update else -1)
        if found is None:
            return
        path, update = found
        try:
            # The architecture first. A checkpoint from before actions became
            # directions has the same layout and the same extractor, so every
            # other gate waves it through and torch is left to report a missing
            # state-dict key -- which says nothing about why it cannot be used.
            checkpoint_architecture(path)
            checkpoint_layout(path).require_compatible(self.layout)
            from stable_baselines3 import PPO

            model = PPO.load(path, device="cpu")
            self.model = model
            self.update = update
            self.error = None
        except Exception as error:  # noqa: BLE001 - keep the previous working policy
            self.error = f"could not load {path.name}: {error}"

    def _decide(self, observation: np.ndarray) -> bytes:
        """The direction, and the field it was chosen from, as one reply.

        A field that cannot be produced costs the viewer a picture; a direction
        that cannot be produced costs it the game. So a failure in the second
        forward pass is reported and dropped rather than raised: the agent keeps
        playing, and the reason the field went dark is on screen.
        """
        x, y = self._predict(observation)
        try:
            field = self._field(observation)
        except Exception as error:  # noqa: BLE001 - the picture, not the game
            self.error = f"no danger field: {error}"
            field = None
        if field is None:
            return REPLY_HEADER.pack(0, 0, x, y)
        return REPLY_HEADER.pack(field.shape[0], 0, x, y) + field.tobytes()

    def _predict(self, observation: np.ndarray) -> tuple[float, float]:
        """The heading to travel on. Standing still is `(0, 0)`.

        Deterministic: watching should show the policy's own choice, not a
        sample from the exploration noise around it. A sampled heading would
        wander either side of the decision and look like indecision the agent
        does not actually have.
        """
        if self.model is None:
            return 0.0, 0.0
        actions, _ = self.model.predict(observation.reshape(1, -1), deterministic=True)
        direction = np.asarray(actions, dtype=np.float64).reshape(-1)
        if direction.size != 2 or not np.isfinite(direction).all():
            raise ValueError(f"policy returned {direction} rather than a direction")
        return float(direction[0]), float(direction[1])

    def _field(self, observation: np.ndarray) -> np.ndarray | None:
        """The nearest-future danger slice, or None if this policy has none.

        Detected by asking whether the extractor can produce a field rather
        than by architecture name, so anything exposing one is drawable and a
        policy without one is skipped rather than crashed.

        The field has one slice per horizon -- "how dangerous is this cell at
        frame h" -- and slice zero is the nearest future, which is the one that
        matches what is on screen now. The later slices are predictions about a
        world the viewer cannot see yet.

        A second forward pass, not a reuse of the one `predict` ran: SB3 gives
        no way back to the features of a completed prediction, and one extra
        pass per decision on a single observation is cheaper than the plumbing
        to avoid it.
        """
        policy = getattr(self.model, "policy", None)
        extractor = getattr(policy, "features_extractor", None)
        if policy is None or not hasattr(extractor, "field"):
            return None
        import torch

        with torch.no_grad():
            tensor, _ = policy.obs_to_tensor(observation.reshape(1, -1))
            field = extractor.field(tensor)
        values = field[0, 0].cpu().numpy()
        if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] > 0xFFFF:
            return None
        return np.ascontiguousarray(values, dtype="<f4")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
