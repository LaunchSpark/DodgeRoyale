"""Client for the DodgeRoyale gym protocol, version 1.

Speaks the format frozen in ``docs/superpowers/specs/velocity-flow-royale-protocol-v1.md``
and implemented by ``src/gym/protocol.rs``. Nothing here knows about SB3,
Gymnasium or PyTorch: this module launches the gym, reads messages, and hands
back NumPy arrays. Training semantics belong a layer up, in the vector
environment.

Two entry points, on purpose:

``decode_*``
    Pure functions over bytes. They are what the committed fixtures in
    ``tests/fixtures/gym-v1/`` are read with, so the wire format can be tested
    with no Rust toolchain, no build and no child process.

:class:`GymClient`
    The live session: launch, handshake, step, reset, close.

**Returned arrays own their memory.** Reads land in reusable scratch buffers,
but every array handed out is a copy. A learner stores the observation it was
given and asks for the next one before it looks at the stored one again; a
reused buffer would rewrite history under it, and the corruption would look
like a training problem rather than a decoding one.
"""

from __future__ import annotations

import io
import json
import os
import platform
import struct
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Final, Iterable, Sequence

import numpy as np

__all__ = [
    "ACTIONS",
    "MAGIC",
    "PROTOCOL_VERSION",
    "GymClient",
    "GymError",
    "Layout",
    "ProtocolError",
    "ResetBatch",
    "Section",
    "StepBatch",
    "Transition",
    "decode_handshake",
    "decode_reset",
    "decode_step",
    "find_binary",
]

MAGIC: Final = b"DRGYM\x00\x00\x01"
PROTOCOL_VERSION: Final = 1

# Request opcodes.
_REQ_STEP: Final = 0x01
_REQ_RESET: Final = 0x02
_REQ_CLOSE: Final = 0x03

# Response opcodes.
_RES_HANDSHAKE: Final = 0x81
_RES_STEP: Final = 0x82
_RES_RESET: Final = 0x83
_RES_CLOSED: Final = 0x84
_RES_ERROR: Final = 0xFF

ACTIONS: Final = (
    "idle",
    "left",
    "right",
    "up",
    "down",
    "up-left",
    "up-right",
    "down-left",
    "down-right",
)

# Longest ERROR message the server will send, per the spec.
_ERROR_MESSAGE_CAP: Final = 1024

# Lines of the child's stderr kept for diagnostics.
_STDERR_LINES: Final = 200

#: Read buffer for the gym's stdout. One megabyte holds a whole 8-env batch
#: and a useful slice of a 64-env one, and costs a megabyte.
READ_BUFFER_BYTES: Final = 1024 * 1024


class ProtocolError(Exception):
    """The bytes on the wire were not what protocol v1 says they should be."""


class GymError(Exception):
    """The gym process failed, or reported an ERROR record.

    Carries the tail of the child's stderr, because a protocol failure is
    almost always explained by something the server already said out loud.
    """

    def __init__(self, message: str, *, code: int | None = None, stderr: str = "") -> None:
        detail = message
        if code is not None:
            detail = f"{detail} (code {code})"
        if stderr:
            detail = f"{detail}\n--- gym stderr ---\n{stderr}"
        super().__init__(detail)
        self.code = code
        self.stderr = stderr


# --- the layout ---------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """Where one part of the flat observation starts, and how long it is."""

    offset: int
    length: int

    @property
    def stop(self) -> int:
        return self.offset + self.length

    def slice(self, observation: np.ndarray) -> np.ndarray:
        """This section of one observation, as a view."""
        return observation[..., self.offset : self.stop]


@dataclass(frozen=True)
class Layout:
    """How to read an observation, exactly as the handshake announced it.

    A checkpoint stores this, so a policy trained against one layout cannot be
    fed another. :meth:`require_compatible` is the check that refuses it.
    """

    version: int
    dtype: str
    grid: int
    cell_pixels: int
    window_pixels: int
    world_units_per_pixel: float
    velocity_scale: float
    y_axis: str
    channels: tuple[str, ...]
    actions: tuple[str, ...]
    horizons: tuple[int, ...]
    hold_frames: int
    path_scale: float
    player_section: Section
    grid_section: Section
    path_section: Section
    observation_values: int

    @classmethod
    def from_json(cls, raw: dict) -> "Layout":
        """Build a layout from handshake JSON, checking it describes itself.

        Validation happens here rather than at first use because the point of
        the handshake is to fail before a policy is built, not after a few
        thousand steps produce nonsense.
        """
        try:
            layout = cls(
                version=int(raw["version"]),
                dtype=str(raw["dtype"]),
                grid=int(raw["grid"]),
                cell_pixels=int(raw["cell_pixels"]),
                window_pixels=int(raw["window_pixels"]),
                world_units_per_pixel=float(raw["world_units_per_pixel"]),
                velocity_scale=float(raw["velocity_scale"]),
                y_axis=str(raw["y_axis"]),
                channels=tuple(str(name) for name in raw["channels"]),
                actions=tuple(str(name) for name in raw["actions"]),
                horizons=tuple(int(frame) for frame in raw["horizons"]),
                hold_frames=int(raw["hold_frames"]),
                path_scale=float(raw["path_scale"]),
                player_section=Section(**raw["player_section"]),
                grid_section=Section(**raw["grid_section"]),
                path_section=Section(**raw["path_section"]),
                observation_values=int(raw["observation_values"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError(f"the handshake layout is malformed: {error}") from error
        layout.validate()
        return layout

    def validate(self) -> None:
        """Refuse a layout that does not add up.

        Every one of these is a silent disaster if it is wrong: a grid section
        that is not ``channels * cells`` long would misalign every channel
        after the first, and a dtype that is not ``f32`` would misread every
        value while parsing cleanly.
        """
        if self.dtype != "f32":
            raise ProtocolError(f"observations must be f32, not {self.dtype!r}")
        if self.y_axis != "down":
            raise ProtocolError(f"the observation's y axis must point down, not {self.y_axis!r}")
        if self.grid <= 0 or self.cell_pixels <= 0:
            raise ProtocolError("the grid and its cells must both be positive")
        if self.grid * self.cell_pixels != self.window_pixels:
            raise ProtocolError(
                f"a {self.grid}x{self.grid} grid of {self.cell_pixels}px cells is not "
                f"{self.window_pixels}px across"
            )
        if self.player_section.offset != 0:
            raise ProtocolError("the player's own values must come first")

        cells = self.grid * self.grid
        if self.grid_section.length != len(self.channels) * cells:
            raise ProtocolError(
                f"{len(self.channels)} channels of {cells} cells is "
                f"{len(self.channels) * cells} values, not {self.grid_section.length}"
            )
        expected_paths = len(self.actions) * len(self.horizons) * 2
        if self.path_section.length != expected_paths:
            raise ProtocolError(
                f"{len(self.actions)} actions x {len(self.horizons)} horizons x 2 is "
                f"{expected_paths} values, not {self.path_section.length}"
            )

        # The three sections must tile the observation exactly: no gap a
        # decoder would read as data, no overlap two readers would disagree on.
        ordered = sorted(
            (self.player_section, self.grid_section, self.path_section),
            key=lambda section: section.offset,
        )
        at = 0
        for section in ordered:
            if section.offset != at:
                raise ProtocolError(
                    f"the observation has a gap or an overlap at value {at}"
                )
            at = section.stop
        if at != self.observation_values:
            raise ProtocolError(
                f"the sections cover {at} values but the layout declares "
                f"{self.observation_values}"
            )
        if tuple(self.actions) != ACTIONS:
            raise ProtocolError(
                f"this client knows the actions {ACTIONS}, the server sent {self.actions}"
            )
        if self.path_scale <= 0 or self.velocity_scale <= 0:
            raise ProtocolError("the path and velocity scales must be positive")

    def require_compatible(self, other: "Layout") -> None:
        """Refuse a layout a policy trained against `self` cannot be fed.

        Every field matters, not only the total length: two layouts of equal
        size that order their channels differently would point every trained
        filter at a different thing while every shape still checked out.
        """
        if self == other:
            return
        differences = [
            f"{name}: trained against {getattr(self, name)!r}, server sent {getattr(other, name)!r}"
            for name in self.__dataclass_fields__
            if getattr(self, name) != getattr(other, name)
        ]
        raise ProtocolError(
            "the gym's observation layout is not the one this policy was trained "
            "against:\n  " + "\n  ".join(differences)
        )

    def as_dict(self) -> dict:
        """A plain serialisable form, for checkpoint kwargs."""
        return {
            "version": self.version,
            "dtype": self.dtype,
            "grid": self.grid,
            "cell_pixels": self.cell_pixels,
            "window_pixels": self.window_pixels,
            "world_units_per_pixel": self.world_units_per_pixel,
            "velocity_scale": self.velocity_scale,
            "y_axis": self.y_axis,
            "channels": list(self.channels),
            "actions": list(self.actions),
            "horizons": list(self.horizons),
            "hold_frames": self.hold_frames,
            "path_scale": self.path_scale,
            "player_section": {
                "offset": self.player_section.offset,
                "length": self.player_section.length,
            },
            "grid_section": {
                "offset": self.grid_section.offset,
                "length": self.grid_section.length,
            },
            "path_section": {
                "offset": self.path_section.offset,
                "length": self.path_section.length,
            },
            "observation_values": self.observation_values,
        }


@dataclass(frozen=True)
class Handshake:
    """What the server announces before any stepping."""

    protocol_version: int
    envs: int
    enemy_count: int
    max_frames: int
    root_seed: int
    workers: int
    layout: Layout


# --- messages -----------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """What happened to one env during one step."""

    frame: int
    terminated: bool
    truncated: bool
    enemy_deaths: int
    episode_seed: int
    reset_seed: int | None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def time_limited(self) -> bool:
        """Truncation is the only reason this episode ended.

        The distinction a learner acts on: bootstrap from the terminal
        observation when the clock ran out, never when the player died. Both
        flags can be set on the same frame, and then the death wins.
        """
        return self.truncated and not self.terminated


@dataclass(frozen=True)
class StepBatch:
    """A decoded ``0x82`` STEP response."""

    transitions: tuple[Transition, ...]
    #: ``(envs, observation_values)``. What each env is on *now*: for an env
    #: that finished, its replacement episode's first frame.
    observations: np.ndarray
    #: Env index -> that finished episode's last observation.
    terminal: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class ResetBatch:
    """A decoded ``0x83`` RESET response."""

    seeds: tuple[int, ...]
    observations: np.ndarray


# --- reading bytes ------------------------------------------------------


def _read_exactly(stream: BinaryIO, count: int) -> bytes:
    """Read exactly `count` bytes, or say the stream ended inside a message.

    One ``read`` need not fill the request: a pipe hands over whatever has
    arrived. Looping is the whole point of this function, and getting it wrong
    produces a decoder that works on small messages and corrupts large ones.
    """
    if count == 0:
        return b""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            got = count - remaining
            raise ProtocolError(
                f"the stream ended inside a message: wanted {count} bytes, got {got}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]


class _Reader:
    """A cursor over a stream, with the payload bound the spec requires."""

    def __init__(self, stream: BinaryIO, maximum: int) -> None:
        self._stream = stream
        self._maximum = maximum

    def u8(self) -> int:
        return _read_exactly(self._stream, 1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", _read_exactly(self._stream, 4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", _read_exactly(self._stream, 8))[0]

    def optional_u64(self) -> int | None:
        """A presence flag, then the value, always both.

        Seed zero is a perfectly good seed, so absence is a flag and never a
        sentinel. The value is on the wire either way and must be consumed.
        """
        present = self.u8()
        value = self.u64()
        return value if present else None

    def length(self) -> int:
        """A declared count, refused before anything is allocated for it."""
        declared = self.u32()
        if declared > self._maximum:
            raise ProtocolError(
                f"a declared length of {declared} exceeds the {self._maximum} maximum "
                "for this session"
            )
        return declared

    def raw(self, count: int) -> bytes:
        return _read_exactly(self._stream, count)

    def byte_string(self) -> bytes:
        return _read_exactly(self._stream, self.length())

    def floats(self) -> np.ndarray:
        """A float array, as an owned little-endian ``float32`` vector."""
        count = self.length()
        raw = _read_exactly(self._stream, count * 4)
        # frombuffer is a view of `raw`; the copy is what makes it the
        # caller's to keep. numpy handles the byte order, so this decodes the
        # same on a big-endian host.
        return np.frombuffer(raw, dtype="<f4", count=count).astype(np.float32, copy=True)


def _error_record(reader: _Reader) -> GymError:
    code = reader.u32()
    message = reader.byte_string()[:_ERROR_MESSAGE_CAP]
    return GymError(message.decode("utf-8", errors="replace"), code=code)


def _expect_opcode(reader: _Reader, wanted: int, what: str) -> None:
    opcode = reader.u8()
    if opcode == wanted:
        return
    if opcode == _RES_ERROR:
        raise _error_record(reader)
    raise ProtocolError(f"expected {what} (opcode {wanted:#04x}), got opcode {opcode:#04x}")


def decode_handshake(stream: BinaryIO) -> Handshake:
    """Read the stream's opening: magic, version, then the layout as JSON."""
    magic = _read_exactly(stream, 8)
    if magic != MAGIC:
        raise ProtocolError(
            f"this is not a DodgeRoyale gym stream: it starts {magic.hex(' ')}, "
            f"not {MAGIC.hex(' ')}"
        )
    version = struct.unpack("<I", _read_exactly(stream, 4))[0]
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"this client speaks protocol v{PROTOCOL_VERSION}, the gym speaks v{version}"
        )

    # The handshake arrives before any env count is known, so its own bound is
    # a fixed one: a layout is a few kilobytes of JSON.
    reader = _Reader(stream, maximum=1 << 20)
    _expect_opcode(reader, _RES_HANDSHAKE, "a handshake")
    try:
        raw = json.loads(reader.byte_string())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"the handshake is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ProtocolError("the handshake is not a JSON object")

    announced = int(raw.get("protocol_version", -1))
    if announced != PROTOCOL_VERSION:
        raise ProtocolError(
            f"the handshake announces protocol v{announced}, not v{PROTOCOL_VERSION}"
        )
    return Handshake(
        protocol_version=announced,
        envs=int(raw["envs"]),
        enemy_count=int(raw["enemy_count"]),
        max_frames=int(raw["max_frames"]),
        root_seed=int(raw["root_seed"]),
        workers=int(raw["workers"]),
        layout=Layout.from_json(raw["layout"]),
    )


def max_payload(envs: int, observation_values: int) -> int:
    """The largest declared length this session will accept.

    Mirrors ``max_payload`` in ``src/gym/protocol.rs``. A length past this is a
    protocol error rather than a large allocation.
    """
    per_env = observation_values * 4 + 64
    return (envs + 1) * per_env * 2 + 4096


def _reshape(values: np.ndarray, envs: int, width: int, what: str) -> np.ndarray:
    if values.size != envs * width:
        raise ProtocolError(
            f"{what} carries {values.size} values, not {envs} x {width}"
        )
    return values.reshape(envs, width)


def decode_step(stream: BinaryIO, envs: int, observation_values: int) -> StepBatch:
    """Read one ``0x82`` STEP response."""
    reader = _Reader(stream, max_payload(envs, observation_values))
    _expect_opcode(reader, _RES_STEP, "a step response")

    count = reader.u32()
    if count != envs:
        raise ProtocolError(f"a step response for {count} envs, but the session has {envs}")
    transitions = tuple(
        Transition(
            frame=reader.u32(),
            terminated=bool(reader.u8()),
            truncated=bool(reader.u8()),
            enemy_deaths=reader.u32(),
            episode_seed=reader.u64(),
            reset_seed=reader.optional_u64(),
        )
        for _ in range(envs)
    )

    observations = _reshape(reader.floats(), envs, observation_values, "a step response")

    terminal: dict[int, np.ndarray] = {}
    for _ in range(reader.u32()):
        env = reader.u32()
        if env >= envs:
            raise ProtocolError(f"a terminal observation for env {env}, out of {envs}")
        if env in terminal:
            raise ProtocolError(f"two terminal observations for env {env}")
        values = reader.floats()
        if values.size != observation_values:
            raise ProtocolError(
                f"a terminal observation of {values.size} values, not {observation_values}"
            )
        terminal[env] = values
    return StepBatch(transitions=transitions, observations=observations, terminal=terminal)


def decode_reset(stream: BinaryIO, envs: int, observation_values: int) -> ResetBatch:
    """Read one ``0x83`` RESET response."""
    reader = _Reader(stream, max_payload(envs, observation_values))
    _expect_opcode(reader, _RES_RESET, "a reset response")

    count = reader.u32()
    if count != envs:
        raise ProtocolError(f"a reset response for {count} envs, but the session has {envs}")
    seeds = tuple(reader.u64() for _ in range(envs))
    observations = _reshape(reader.floats(), envs, observation_values, "a reset response")
    return ResetBatch(seeds=seeds, observations=observations)


# --- writing requests ---------------------------------------------------


def encode_step(actions: Sequence[int]) -> bytes:
    """A STEP request: one action byte per env, in env order."""
    payload = bytes(_checked_action(action) for action in actions)
    return bytes([_REQ_STEP]) + struct.pack("<I", len(payload)) + payload


def _checked_action(action: int) -> int:
    value = int(action)
    if not 0 <= value < len(ACTIONS):
        raise ValueError(f"action {value} is not one of the {len(ACTIONS)} actions")
    return value


def encode_reset(seed: int | None) -> bytes:
    """A RESET request. ``None`` continues the seed stream; a seed restarts it."""
    if seed is None:
        return bytes([_REQ_RESET, 0]) + struct.pack("<Q", 0)
    if not 0 <= seed < (1 << 64):
        raise ValueError(f"a seed must fit in a u64, got {seed}")
    return bytes([_REQ_RESET, 1]) + struct.pack("<Q", seed)


def encode_close() -> bytes:
    return bytes([_REQ_CLOSE])


# --- finding the binary -------------------------------------------------


def find_binary(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate the gym binary.

    ``DODGE_ROYALE_BIN`` wins, and if it is set but wrong that is an error
    rather than a silent fall back to a different binary: an override the user
    took the trouble to set should not be ignored. Otherwise this looks for the
    release build in the checkout this module was imported from -- not the
    shell's working directory, which is not where the code lives.
    """
    suffix = ".exe" if platform.system() == "Windows" else ""
    if explicit is not None:
        path = Path(explicit)
    else:
        override = os.environ.get("DODGE_ROYALE_BIN")
        if override:
            path = Path(override)
        else:
            # training/dodge_royale/protocol.py -> the repository root.
            root = Path(__file__).resolve().parent.parent.parent
            path = root / "target" / "release" / f"dodge-royale{suffix}"
            if not path.exists():
                raise GymError(
                    f"no gym binary at {path}. Build it with `cargo build --release "
                    "--no-default-features`, or set DODGE_ROYALE_BIN. An installation "
                    "outside the checkout, or a custom CARGO_TARGET_DIR, needs the "
                    "explicit override."
                )
            return path
    if not path.exists():
        raise GymError(f"no gym binary at {path}")
    return path


# --- the live session ---------------------------------------------------


class GymClient:
    """One ``dodge-royale gym`` child process, and the conversation with it.

    Not thread safe: one owner sends a request and reads its response. The
    stderr drain is the only other thread, and it never touches the protocol
    stream.
    """

    def __init__(
        self,
        *,
        envs: int = 8,
        seed: int = 0,
        enemies: int = 100,
        max_frames: int = 3600,
        hold_frames: int = 24,
        threads: int = 2,
        binary: str | os.PathLike[str] | None = None,
        close_timeout: float = 5.0,
    ) -> None:
        self._binary = find_binary(binary)
        self._close_timeout = close_timeout
        self._closed = False
        self._stderr: deque[str] = deque(maxlen=_STDERR_LINES)
        self._drain: threading.Thread | None = None

        command = [
            str(self._binary),
            "gym",
            "--envs",
            str(envs),
            "--seed",
            str(seed),
            "--enemies",
            str(enemies),
            "--max-frames",
            str(max_frames),
            "--hold-frames",
            str(hold_frames),
            "--threads",
            str(threads),
        ]
        # An argument list and binary pipes, never a shell: nothing here should
        # depend on how a shell would split or expand these.
        self._process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
        )
        # Read through a buffer, write without one.
        #
        # A raw pipe read returns only what has arrived, so a 7 MB batch at 64
        # envs becomes thousands of small reads: measured at 545 ms a step and
        # 13 MiB/s, against 45 ms and 154 MiB/s once buffered -- and the
        # penalty grows with the batch, so it barely shows at 8 envs. Requests
        # stay unbuffered, because a request the server never sees is a
        # deadlock and they are far too small for buffering to pay.
        if self._process.stdout is not None:
            self._process.stdout = io.BufferedReader(
                self._process.stdout, READ_BUFFER_BYTES
            )
        self._start_drain()

        try:
            self.handshake = decode_handshake(self._stdout)
            self.layout = self.handshake.layout
            self.envs = self.handshake.envs
            # Frame zero arrives unasked, so a caller can step immediately.
            self.initial = self._read_reset()
        except BaseException as error:
            # A handshake that fails leaves a child holding pipes open.
            self.close()
            if isinstance(error, GymError):
                raise
            raise GymError(str(error), stderr=self.stderr_tail()) from error

    # -- plumbing --

    @property
    def _stdin(self) -> BinaryIO:
        stream = self._process.stdin
        if stream is None:
            raise GymError("the gym's stdin is not open", stderr=self.stderr_tail())
        return stream

    @property
    def _stdout(self) -> BinaryIO:
        stream = self._process.stdout
        if stream is None:
            raise GymError("the gym's stdout is not open", stderr=self.stderr_tail())
        return stream

    def _start_drain(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return

        def pump() -> None:
            # Bounded: a chatty or looping child must not become unbounded
            # memory in the trainer.
            for line in iter(stream.readline, b""):
                self._stderr.append(line.decode("utf-8", errors="replace").rstrip("\n"))
            stream.close()

        self._drain = threading.Thread(target=pump, name="dodge-royale-stderr", daemon=True)
        self._drain.start()

    def stderr_tail(self) -> str:
        """The last lines the child wrote, for a failure report."""
        return "\n".join(self._stderr)

    def _send(self, request: bytes) -> None:
        if self._closed:
            raise GymError("this session is closed")
        try:
            self._stdin.write(request)
            self._stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise GymError(
                f"the gym stopped reading: {error}", stderr=self._exit_detail()
            ) from error

    def _exit_detail(self) -> str:
        """Stderr, plus how the child exited if it has.

        Waited on briefly rather than not at all: a child that has just died
        usually has not been reaped yet, and "exit code 101" is most of the
        diagnosis.
        """
        try:
            code = self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            code = None
        tail = self.stderr_tail()
        if code is None:
            return tail
        return f"{tail}\n--- gym exited with code {code} ---" if tail else f"exit code {code}"

    def _read_step(self) -> StepBatch:
        try:
            return decode_step(self._stdout, self.envs, self.layout.observation_values)
        except GymError as error:
            raise GymError(str(error), code=error.code, stderr=self._exit_detail()) from error
        except ProtocolError as error:
            raise GymError(str(error), stderr=self._exit_detail()) from error

    def _read_reset(self) -> ResetBatch:
        try:
            return decode_reset(self._stdout, self.envs, self.layout.observation_values)
        except GymError as error:
            raise GymError(str(error), code=error.code, stderr=self._exit_detail()) from error
        except ProtocolError as error:
            raise GymError(str(error), stderr=self._exit_detail()) from error

    # -- the protocol --

    def step(self, actions: Sequence[int] | Iterable[int]) -> StepBatch:
        """Advance every env one frame."""
        chosen = list(actions)
        if len(chosen) != self.envs:
            raise ValueError(f"this session has {self.envs} envs, got {len(chosen)} actions")
        self._send(encode_step(chosen))
        return self._read_step()

    def reset(self, seed: int | None = None) -> ResetBatch:
        """Restart every env.

        A seed restarts the stream from that root and returns the episode
        counters to zero. Without one, each env's counter advances, continuing
        the stream from the same root.
        """
        self._send(encode_reset(seed))
        return self._read_reset()

    def close(self) -> None:
        """End the session and make sure the child is gone.

        Idempotent, and safe to call on a half-built client: it is what the
        constructor calls when the handshake fails. The child is asked to
        close, then waited for, then terminated, then killed -- and reaped at
        every stage, because a killed child whose status is never collected is
        a zombie.
        """
        if self._closed:
            return
        self._closed = True

        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write(encode_close())
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

        # Close stdin but *not* stdout. A CLOSE is answered with a one-byte
        # CLOSED before the server exits, and closing the read end first would
        # break that write and turn an ordinary shutdown into a failed one. An
        # EOF on stdin is the other clean ending, and it also unblocks a server
        # parked in a read. The acknowledgement is left in the pipe: it is not
        # worth a blocking read that has no portable deadline, and the exit
        # status already says how the session ended.
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        if not self._reap(self._close_timeout):
            self._process.terminate()
            if not self._reap(self._close_timeout):
                self._process.kill()
                # No timeout: after a kill the wait is the reap, and a deadline
                # here would leave a zombie to avoid a hang that cannot happen.
                self._process.wait()

        # Safe now: nothing is left to write into it.
        if self._process.stdout is not None:
            try:
                self._process.stdout.close()
            except (BrokenPipeError, OSError):
                pass

        if self._drain is not None:
            self._drain.join(timeout=self._close_timeout)
            self._drain = None

    def _reap(self, timeout: float) -> bool:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    def __enter__(self) -> "GymClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best effort: an interpreter shutdown that drops the client should not
        # leave a gym running.
        try:
            self.close()
        except Exception:  # noqa: BLE001 - nothing useful can be done here
            pass


if __name__ == "__main__":  # pragma: no cover - a smoke check, not a test
    with GymClient(envs=2, max_frames=64) as gym:
        print(f"layout v{gym.layout.version}, {gym.envs} envs", file=sys.stderr)
        batch = gym.step([0] * gym.envs)
        print(f"frame {batch.transitions[0].frame}", file=sys.stderr)
