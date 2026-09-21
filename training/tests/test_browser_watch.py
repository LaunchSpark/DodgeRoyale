"""The browser game's bridge: a valid encoded frame in, a decision out."""

from __future__ import annotations

import json
import socket

import pytest
from dataclasses import asdict
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from dodge_royale.browser_watch import REPLY_HEADER, GameWatch


def unpack(body: bytes):
    """A reply as (direction, field), the field None or a square grid."""
    edge, _, x, y = REPLY_HEADER.unpack_from(body)
    if edge == 0:
        assert len(body) == REPLY_HEADER.size, "no field means no payload"
        return (x, y), None
    field = np.frombuffer(body, dtype="<f4", offset=REPLY_HEADER.size)
    assert field.size == edge * edge
    return (x, y), field.reshape(edge, edge)


class FakeModel:
    def predict(self, observation, *, deterministic):
        assert observation.shape == (1, 28782)
        assert deterministic
        # Fifteen degrees: off the old nine-way compass on purpose, so a
        # bridge that rounded to one of them would send something else.
        return np.array([[0.966, 0.259]], dtype=np.float32), None


def test_stopping_watch_closes_its_policy_port(tmp_path):
    watch = GameWatch(tmp_path, web_url="http://127.0.0.1:4000/")
    port = watch._server.server_port
    watch.close()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            pass
    except OSError:
        pass
    else:
        raise AssertionError("stopping Watch left its policy server listening")


def test_browser_game_receives_policy_actions_from_its_own_observation(manifest, tmp_path):
    watch = GameWatch(tmp_path, web_url="http://127.0.0.1:4000/")
    try:
        assert "watch=1" in watch.url
        assert "policy_url=" in watch.url
        watch.model = FakeModel()
        observation = np.zeros(manifest["layout"]["observation_values"], dtype="<f4")
        endpoint = f"http://127.0.0.1:{watch._server.server_port}/infer"
        request = Request(
            endpoint,
            data=observation.tobytes(),
            headers={"Content-Type": "application/octet-stream",
                     "X-Dodge-Layout": json.dumps(manifest["layout"])},
        )
        with urlopen(request, timeout=5) as response:
            direction, field = unpack(response.read())
            # The heading survives as a heading. Fifteen degrees is not one of
            # the nine the old action space had, so a bridge that rounded
            # anywhere would come back with something else.
            assert direction == pytest.approx((0.966, 0.259), abs=1e-6)
            assert field is None, "a policy with no danger field still plays"
            assert response.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:4000"

        preflight = Request(endpoint, method="OPTIONS")
        with urlopen(preflight, timeout=5) as response:
            assert response.status == 204
            assert "X-Dodge-Layout" in response.headers["Access-Control-Allow-Headers"]

        wrong = Request(endpoint, data=b"short", headers=request.headers)
        try:
            urlopen(wrong, timeout=5)
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("a malformed observation must be refused")

        changed = dict(manifest["layout"], hold_frames=25)
        wrong_layout = Request(
            endpoint,
            data=observation.tobytes(),
            headers={"Content-Type": "application/octet-stream",
                     "X-Dodge-Layout": json.dumps(changed)},
        )
        try:
            urlopen(wrong_layout, timeout=5)
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("the server must refuse a changed layout")
    finally:
        watch.close()


def test_browser_bridge_loads_a_real_royale_checkpoint(tmp_path):
    import pytest

    from dodge_royale.protocol import GymError, find_binary
    from dodge_royale.training import SessionConfig, build_model, make_env

    try:
        find_binary()
    except GymError:
        pytest.skip("no gym binary built")
    config = SessionConfig(envs=1, n_steps=8, n_epochs=1, total_timesteps=8, device="cpu")
    with make_env(config) as env:
        build_model(config, env).save(tmp_path / "update-00000001.zip")
        observation = env.reset()[0].astype("<f4", copy=False)
        layout = json.dumps(asdict(env.layout))

    watch = GameWatch(tmp_path, web_url="http://127.0.0.1:4000/")
    try:
        request = Request(
            f"http://127.0.0.1:{watch._server.server_port}/infer",
            data=observation.tobytes(),
            headers={"Content-Type": "application/octet-stream", "X-Dodge-Layout": layout},
        )
        with urlopen(request, timeout=15) as response:
            direction, field = unpack(response.read())
            assert response.headers["X-Watch-Update"] == "update 1"
        assert all(np.isfinite(direction))
        assert watch.update == 1

        # The whole point of watching: the reply carries the field the action
        # was chosen from, on the same 64x64 lattice as the observation window,
        # so the game can draw it over the region it describes.
        assert field is not None, "a VelocityFlow policy must publish its field"
        assert field.shape == (64, 64)
        assert np.isfinite(field).all()
        assert field.max() > field.min(), "a field with no gradient says nothing"
    finally:
        watch.close()
