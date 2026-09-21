"""Decode the committed fixtures, and refuse everything malformed.

None of these build or run Rust. They read bytes that are already in the
repository, which is the point of committing them: a wire-format regression
should be visible here without a toolchain.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

from conftest import fixtures_of
from dodge_royale.protocol import (
    ACTIONS,
    MAGIC,
    PROTOCOL_VERSION,
    GymError,
    Layout,
    ProtocolError,
    decode_handshake,
    decode_reset,
    decode_step,
    encode_close,
    encode_reset,
    encode_step,
    max_payload,
)


def stream(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)


# --- every committed fixture --------------------------------------------


def test_the_handshake_fixture_announces_a_layout_we_accept(manifest, fixture_bytes):
    handshake = decode_handshake(stream(fixture_bytes("handshake.bin")))
    assert handshake.protocol_version == PROTOCOL_VERSION
    assert handshake.layout.as_dict() == manifest["layout"]
    assert handshake.layout.observation_values == manifest["observation_values"]
    assert handshake.layout.actions == ACTIONS


def test_every_reset_fixture_decodes(manifest, fixture_bytes):
    width = manifest["observation_values"]
    for entry in fixtures_of(manifest, "reset"):
        batch = decode_reset(stream(fixture_bytes(entry["file"])), entry["envs"], width)
        assert list(batch.seeds) == entry["episode_seeds"], entry["file"]
        assert batch.observations.shape == (entry["envs"], width), entry["file"]
        assert batch.observations.dtype == np.float32
        assert np.isfinite(batch.observations).all(), entry["file"]


def test_every_step_fixture_decodes(manifest, fixture_bytes):
    width = manifest["observation_values"]
    seen = 0
    for entry in fixtures_of(manifest, "step"):
        batch = decode_step(stream(fixture_bytes(entry["file"])), entry["envs"], width)
        assert batch.observations.shape == (entry["envs"], width), entry["file"]
        assert len(batch.transitions) == entry["envs"], entry["file"]

        for expected in entry["transitions"]:
            got = batch.transitions[expected["env"]]
            assert got.frame == expected["frame"], entry["file"]
            assert got.terminated == expected["terminated"], entry["file"]
            assert got.truncated == expected["truncated"], entry["file"]
            assert got.enemy_deaths == expected["enemy_deaths"], entry["file"]
            assert got.episode_seed == expected["episode_seed"], entry["file"]
            assert got.reset_seed == expected["reset_seed"], entry["file"]

        assert sorted(batch.terminal) == entry.get("terminal_envs", []), entry["file"]
        for env, observation in batch.terminal.items():
            assert observation.shape == (width,), entry["file"]
            # A finished episode is replaced in the same message, so the
            # terminal observation and the batch row for that env are two
            # different frames and must not be the same numbers.
            assert not np.array_equal(observation, batch.observations[env]), entry["file"]
        seen += 1
    assert seen >= 4


def test_fixture_values_match_the_manifest_exactly(manifest, fixture_bytes):
    """Both forms the manifest records, on every sample it records.

    The decimal and the bit pattern, because a decoder that agreed with one
    and not the other would be reading the file correctly and the numbers
    wrongly, or the reverse.
    """
    width = manifest["observation_values"]
    checked = 0
    for entry in fixtures_of(manifest, "step") + fixtures_of(manifest, "reset"):
        raw = fixture_bytes(entry["file"])
        if entry["kind"] == "step":
            batch = decode_step(stream(raw), entry["envs"], width)
            terminal = batch.terminal
        else:
            batch = decode_reset(stream(raw), entry["envs"], width)
            terminal = {}

        for sample in entry.get("samples", []):
            value = batch.observations[sample["env"], sample["index"]]
            assert float(value) == sample["value"], (entry["file"], sample["index"])
            assert struct.unpack("<I", struct.pack("<f", value))[0] == sample["bits"]
            checked += 1
        for sample in entry.get("terminal_samples", []):
            value = terminal[sample["env"]][sample["index"]]
            assert float(value) == sample["value"], (entry["file"], sample["index"])
            assert struct.unpack("<I", struct.pack("<f", value))[0] == sample["bits"]
            checked += 1
    assert checked > 1000, "the manifest should pin a lot more than a handful of values"


# --- both ways an episode ends ------------------------------------------


def test_a_timeout_is_truncated_and_bootstrappable(manifest, fixture_bytes):
    """The frame budget ran out, so the state was still perfectly alive."""
    width = manifest["observation_values"]
    batch = decode_step(stream(fixture_bytes("step-auto-reset.bin")), 2, width)
    for transition in batch.transitions:
        assert transition.truncated
        assert not transition.terminated
        assert transition.done
        assert transition.time_limited, "a learner must bootstrap from a timeout"
        assert transition.reset_seed is not None
    assert sorted(batch.terminal) == [0, 1]


def test_a_death_is_terminated_and_must_not_be_bootstrapped(manifest, fixture_bytes):
    """The player was hit, so there is no future to bootstrap towards.

    This is the case `step-auto-reset.bin` cannot cover: it has both envs
    timing out together, and a learner that treated the two endings alike
    would still pass every assertion there.
    """
    width = manifest["observation_values"]
    batch = decode_step(stream(fixture_bytes("step-death.bin")), 2, width)

    dead = [index for index, t in enumerate(batch.transitions) if t.terminated]
    assert dead, "the death fixture should contain a death"
    for index in dead:
        transition = batch.transitions[index]
        assert not transition.truncated, "a death is not a timeout"
        assert transition.done
        assert not transition.time_limited, "a learner must not bootstrap from a death"
        assert transition.reset_seed is not None, "a finished episode is replaced"
        assert index in batch.terminal

    # And the same message carries an env that did not finish: a batch is not
    # all-or-nothing, and a client that assumed it was would mis-handle this.
    alive = [index for index, t in enumerate(batch.transitions) if not t.done]
    assert alive, "the envs do not die together; this is the mixed batch"
    for index in alive:
        assert batch.transitions[index].reset_seed is None
        assert index not in batch.terminal


def test_a_running_step_ends_nothing(manifest, fixture_bytes):
    width = manifest["observation_values"]
    batch = decode_step(stream(fixture_bytes("step-running.bin")), 2, width)
    assert not batch.terminal
    for transition in batch.transitions:
        assert not transition.done
        assert transition.reset_seed is None


# --- retained arrays own their memory ------------------------------------


def test_a_retained_observation_is_unchanged_by_later_reads(manifest, fixture_bytes):
    """The corruption this client exists to prevent.

    SB3 stores the observation it was handed and only reads it after calling
    step again. If a decode reused the buffer behind it, the rollout would
    quietly fill with the wrong frames, and it would look like a training
    problem rather than a decoding one.
    """
    width = manifest["observation_values"]
    first = decode_step(stream(fixture_bytes("step-running.bin")), 2, width)
    kept = first.observations
    kept_copy = kept.copy()
    kept_terminal = decode_step(stream(fixture_bytes("step-death.bin")), 2, width)
    terminal_kept = kept_terminal.terminal[
        next(iter(kept_terminal.terminal))
    ]
    terminal_copy = terminal_kept.copy()

    # Several more messages through the same decoder.
    for _ in range(3):
        decode_step(stream(fixture_bytes("step-auto-reset.bin")), 2, width)
        decode_reset(stream(fixture_bytes("reset-frame-zero.bin")), 2, width)
        decode_step(stream(fixture_bytes("scenes.bin")), 3, width)

    assert np.array_equal(kept, kept_copy), "a retained batch changed under us"
    assert np.array_equal(terminal_kept, terminal_copy), "a retained terminal changed"


def test_decoded_arrays_share_memory_with_nothing_else(manifest, fixture_bytes):
    """Owned, writable, and not a window onto bytes someone else holds.

    `owndata` is the wrong question: reshaping a decoded vector makes a view
    whose base owns the data, which is fine. What must be true is that no two
    decodes share storage, and that nothing hands back a read-only window onto
    the received buffer, which is what `np.frombuffer` alone would do.
    """
    width = manifest["observation_values"]
    raw = fixture_bytes("step-running.bin")
    first = decode_step(stream(raw), 2, width)
    second = decode_step(stream(raw), 2, width)

    assert not np.shares_memory(first.observations, second.observations)
    for batch in (first, second):
        assert batch.observations.flags.writeable
        base = batch.observations.base
        assert base is None or base.flags.owndata

    death = decode_step(stream(fixture_bytes("step-death.bin")), 2, width)
    for observation in death.terminal.values():
        assert observation.flags.writeable
        assert not np.shares_memory(observation, death.observations)

    # Writing into one must not be visible in the other.
    first.observations[0, 0] = -12.5
    assert second.observations[0, 0] != -12.5


# --- malformed input -----------------------------------------------------


def test_another_programs_output_is_rejected_rather_than_parsed():
    with pytest.raises(ProtocolError, match="not a DodgeRoyale gym stream"):
        decode_handshake(stream(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64))


def test_a_future_protocol_version_is_refused_before_a_policy_is_built():
    later = MAGIC + struct.pack("<I", PROTOCOL_VERSION + 1) + b"\x81\x00\x00\x00\x00"
    with pytest.raises(ProtocolError, match="speaks protocol v"):
        decode_handshake(stream(later))


def test_a_handshake_that_disagrees_with_its_own_envelope_is_refused(fixture_bytes):
    raw = bytearray(fixture_bytes("handshake.bin"))
    length = struct.unpack_from("<I", raw, 13)[0]
    body = json.loads(bytes(raw[17 : 17 + length]))
    body["protocol_version"] = 99
    rebuilt = json.dumps(body).encode()
    forged = bytes(raw[:13]) + struct.pack("<I", len(rebuilt)) + rebuilt
    with pytest.raises(ProtocolError, match="announces protocol v99"):
        decode_handshake(stream(forged))


@pytest.mark.parametrize("cut", [0, 1, 7, 8, 12, 13, 40, 400])
def test_a_message_cut_short_is_truncation_not_a_short_batch(fixture_bytes, cut):
    """Every prefix must fail, and fail as truncation.

    A decoder that returned a short batch here would hand the learner fewer
    envs than it asked for, which nothing downstream checks.
    """
    raw = fixture_bytes("handshake.bin")[:cut]
    with pytest.raises(ProtocolError):
        decode_handshake(stream(raw))


@pytest.mark.parametrize("keep", [1, 5, 9, 25, 1000, 200_000])
def test_a_step_cut_short_is_refused(manifest, fixture_bytes, keep):
    width = manifest["observation_values"]
    raw = fixture_bytes("step-running.bin")[:keep]
    with pytest.raises(ProtocolError):
        decode_step(stream(raw), 2, width)


def test_a_short_read_is_looped_over_rather_than_truncating():
    """One read need not fill the request; a pipe hands over what has arrived.

    A decoder that trusted a single read would work against a file and corrupt
    large messages against a pipe, which is the only place it matters.
    """

    class Dribble(io.RawIOBase):
        """A stream that never returns more than three bytes at a time."""

        def __init__(self, data: bytes) -> None:
            self._data = data
            self._at = 0

        def read(self, size: int = -1) -> bytes:
            take = min(3 if size < 0 else min(size, 3), len(self._data) - self._at)
            chunk = self._data[self._at : self._at + take]
            self._at += take
            return chunk

    envs, width = 2, 4
    message = bytearray([0x82])
    message += struct.pack("<I", envs)
    for env in range(envs):
        message += struct.pack("<IBBIQBQ", env, 0, 0, 0, env, 0, 0)
    message += struct.pack("<I", envs * width)
    for index in range(envs * width):
        message += struct.pack("<f", index * 0.5)
    message += struct.pack("<I", 0)

    batch = decode_step(Dribble(bytes(message)), envs, width)
    assert batch.observations.shape == (envs, width)
    assert batch.observations[1, 3] == pytest.approx(3.5)


def test_a_declared_length_past_the_maximum_is_refused_before_allocating():
    """A hostile count must not become a multi-gigabyte allocation."""
    envs, width = 2, 8
    message = bytearray([0x82])
    message += struct.pack("<I", envs)
    for env in range(envs):
        message += struct.pack("<IBBIQBQ", 0, 0, 0, 0, env, 0, 0)
    message += struct.pack("<I", 0xFFFF_FFFF)
    with pytest.raises(ProtocolError, match="exceeds"):
        decode_step(stream(bytes(message)), envs, width)


def test_a_float_array_of_the_wrong_length_is_refused(manifest):
    envs, width = 2, 8
    message = bytearray([0x82])
    message += struct.pack("<I", envs)
    for env in range(envs):
        message += struct.pack("<IBBIQBQ", 0, 0, 0, 0, env, 0, 0)
    # One value short of a batch: long enough to read, wrong enough to reject.
    message += struct.pack("<I", envs * width - 1)
    message += b"\x00\x00\x00\x00" * (envs * width - 1)
    message += struct.pack("<I", 0)
    with pytest.raises(ProtocolError, match="not 2 x 8"):
        decode_step(stream(bytes(message)), envs, width)


def test_an_empty_stream_is_an_eof_not_a_hang():
    with pytest.raises(ProtocolError, match="ended inside a message"):
        decode_handshake(stream(b""))


def test_a_response_read_as_the_wrong_kind_names_the_opcode(manifest, fixture_bytes):
    width = manifest["observation_values"]
    with pytest.raises(ProtocolError, match="opcode"):
        decode_step(stream(fixture_bytes("reset-frame-zero.bin")), 2, width)


def test_an_error_record_becomes_an_exception_carrying_its_code():
    message = b"placement failed"
    record = bytes([0xFF]) + struct.pack("<I", 2) + struct.pack("<I", len(message)) + message
    with pytest.raises(GymError, match="placement failed") as caught:
        decode_step(stream(record), 2, 8)
    assert caught.value.code == 2


def test_a_terminal_observation_for_an_env_that_does_not_exist_is_refused():
    envs, width = 2, 4
    message = bytearray([0x82])
    message += struct.pack("<I", envs)
    for env in range(envs):
        message += struct.pack("<IBBIQBQ", 1, 0, 0, 0, env, 0, 0)
    message += struct.pack("<I", envs * width) + b"\x00\x00\x00\x00" * (envs * width)
    message += struct.pack("<I", 1) + struct.pack("<I", 7)  # env 7 of 2
    message += struct.pack("<I", width) + b"\x00\x00\x00\x00" * width
    with pytest.raises(ProtocolError, match="env 7"):
        decode_step(stream(bytes(message)), envs, width)


# --- requests ------------------------------------------------------------


def test_a_step_request_is_bytes_we_can_write_out_by_hand():
    assert encode_step([(0.0, 0.0), (1.0, -0.5)]) == bytes(
        [
            0x01,  # STEP
            2, 0, 0, 0,  # two envs, not two floats
            0, 0, 0, 0,  # env 0 x = 0.0
            0, 0, 0, 0,  # env 0 y = 0.0
            0, 0, 0x80, 0x3F,  # env 1 x = 1.0
            0, 0, 0, 0xBF,  # env 1 y = -0.5
        ]
    )


def test_a_direction_is_written_as_little_endian_f32():
    """Not a round number in binary: a server reading these bytes as anything
    else would put the player on a different heading."""
    direction = (0.123456789, -0.987654321)
    written = encode_step([direction])[5:]
    assert written == struct.pack("<2f", *direction)
    assert struct.unpack("<2f", written) == struct.unpack(
        "<2f", struct.pack("<2f", *direction)
    )


def test_a_reset_carries_a_presence_flag_so_seed_zero_stays_a_seed():
    assert encode_reset(None) == bytes([0x02, 0]) + b"\x00" * 8
    assert encode_reset(0) == bytes([0x02, 1]) + b"\x00" * 8
    assert encode_reset(7) == bytes([0x02, 1]) + struct.pack("<Q", 7)
    assert encode_reset(None) != encode_reset(0)


def test_a_close_is_one_byte():
    assert encode_close() == b"\x03"


@pytest.mark.parametrize(
    "action", [(float("nan"), 0.0), (0.0, float("inf")), (float("-inf"), 0.0)]
)
def test_a_direction_that_is_not_finite_is_refused_before_it_is_sent(action):
    """The gym answers a malformed STEP by ending the session, so a direction
    that is not a number must never reach the pipe."""
    with pytest.raises(ValueError, match="not finite"):
        encode_step([action])


@pytest.mark.parametrize("action", [3, None, (1.0,), (1.0, 2.0, 3.0)])
def test_an_action_that_is_not_a_pair_is_refused_before_it_is_sent(action):
    with pytest.raises(ValueError, match="an .x, y. direction"):
        encode_step([action])


def test_a_long_direction_is_sent_rather_than_normalised():
    """Length sets no speed, so nothing here rescales it. Normalising would
    quietly discard the one thing a short vector says: stand still."""
    assert encode_step([(3.0, 4.0)])[5:] == struct.pack("<2f", 3.0, 4.0)


def test_the_payload_bound_admits_a_real_batch_and_refuses_far_more(manifest):
    width = manifest["observation_values"]
    bound = max_payload(8, width)
    assert bound > 8 * width * 4
    assert bound < 8 * width * 4 * 4


# --- layout validation ---------------------------------------------------


def valid_layout(manifest) -> dict:
    return json.loads(json.dumps(manifest["layout"]))


def test_a_layout_round_trips_through_its_plain_form(manifest):
    layout = Layout.from_json(valid_layout(manifest))
    assert Layout.from_json(layout.as_dict()) == layout


@pytest.mark.parametrize(
    ("field", "value", "complaint"),
    [
        ("dtype", "f16", "must be f32"),
        ("y_axis", "up", "y axis"),
        ("grid", 0, "must both be positive"),
        ("window_pixels", 255, "not 255px across"),
        ("path_scale", 0.0, "must be positive"),
    ],
)
def test_a_layout_that_does_not_add_up_is_refused(manifest, field, value, complaint):
    raw = valid_layout(manifest)
    raw[field] = value
    with pytest.raises(ProtocolError, match=complaint):
        Layout.from_json(raw)


def test_a_grid_section_that_does_not_fit_its_channels_is_refused(manifest):
    raw = valid_layout(manifest)
    raw["channels"] = raw["channels"][:-1]
    with pytest.raises(ProtocolError, match="channels of"):
        Layout.from_json(raw)


def test_sections_that_leave_a_gap_are_refused(manifest):
    raw = valid_layout(manifest)
    raw["grid_section"]["offset"] += 1
    with pytest.raises(ProtocolError, match="gap or an overlap"):
        Layout.from_json(raw)


def test_a_reordered_channel_list_is_a_different_layout(manifest):
    """Same length, same shapes, every trained filter pointed elsewhere."""
    trained = Layout.from_json(valid_layout(manifest))
    raw = valid_layout(manifest)
    raw["channels"] = [raw["channels"][1], raw["channels"][0], *raw["channels"][2:]]
    served = Layout.from_json(raw)

    assert served.observation_values == trained.observation_values
    with pytest.raises(ProtocolError, match="not the one this policy was trained"):
        trained.require_compatible(served)


def test_a_layout_compatible_with_itself_is_accepted(manifest):
    layout = Layout.from_json(valid_layout(manifest))
    layout.require_compatible(Layout.from_json(valid_layout(manifest)))


def test_a_different_hold_duration_is_refused_even_at_the_same_length(manifest):
    trained = Layout.from_json(valid_layout(manifest))
    raw = valid_layout(manifest)
    raw["hold_frames"] += 1
    with pytest.raises(ProtocolError, match="hold_frames"):
        trained.require_compatible(Layout.from_json(raw))


def test_sections_address_the_parts_they_name(manifest, fixture_bytes):
    layout = Layout.from_json(valid_layout(manifest))
    batch = decode_step(
        stream(fixture_bytes("scenes.bin")), 3, manifest["observation_values"]
    )
    player = layout.player_section.slice(batch.observations)
    paths = layout.path_section.slice(batch.observations)
    grid = layout.grid_section.slice(batch.observations)
    assert player.shape == (3, 2)
    assert paths.shape == (3, len(ACTIONS) * len(layout.horizons) * 2)
    assert grid.shape == (3, len(layout.channels) * layout.grid**2)
    # The player's own velocity is scaled into [-1, 1] by the encoder.
    assert np.all(np.abs(player) <= 1.0)
