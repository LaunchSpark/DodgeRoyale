# DodgeRoyale gym protocol, version 2

**Status:** frozen. Any change to a message's meaning bumps
`PROTOCOL_VERSION` and this document together.
**Implemented by:** `src/gym/protocol.rs` (Rust) and
`training/dodge_royale/protocol.py` (Python), in this same repository.

## What changed from version 1

One thing: a STEP request carried one action **byte** per env, naming one of
nine compass headings. It now carries one direction **vector** per env, so the
agent can travel on any heading rather than the nine.

Nothing else moved. Every response is byte-for-byte what version 1 sent, the
observation layout is unchanged, and the handshake differs only in the version
it announces. The nine action names still appear in the layout, because the
observation still carries nine velocity-conditioned candidate paths for a
policy to score — they are the paths, not the choices.

There is no negotiation and no compatibility mode. A version 1 client and a
version 2 server disagree about the meaning of a STEP's bytes, so they refuse
each other at the magic rather than at the first step.

The trainer runs `dodge-royale gym` as a child process and speaks this protocol
over its stdin and stdout. Stdout carries protocol bytes and nothing else;
every diagnostic goes to stderr.

## Conventions

* Little endian, fixed width. `u8`, `u32`, `u64`, `f32`.
* No Rust struct is ever written to the wire, so padding, alignment and `usize`
  cannot leak into the format.
* A byte string is a `u32` length followed by that many bytes. Text is UTF-8.
* A float array is a `u32` count followed by that many `f32`.
* An optional `u64` is a `u8` presence flag followed by the value, always.
  Seed zero is a legitimate seed, so absence is a flag and never a sentinel.

## Opening

The server writes, once, before anything else:

| Field | Type | Value |
|---|---|---|
| magic | 8 bytes | `44 52 47 59 4D 00 00 02` (`DRGYM\0\0\x02`) |
| version | u32 | 2 |
| opcode | u8 | `0x81` HANDSHAKE |
| handshake | byte string | JSON |

The JSON carries `protocol_version`, `envs`, `enemy_count`, `max_frames`,
`root_seed`, `workers`, and `layout`. The layout describes the observation
completely: version, dtype, grid size, cell size, window size, world units per
pixel, velocity scale, Y direction, channel names in order, action names in
order, horizons, hold frames, path scale, the offset and length of each
section, and the total value count.

A client that cannot match the layout it was trained against must fail here,
before it builds a policy.

The server then writes one `0x83` RESET response carrying frame zero: every
arena is already built and filled by the time the handshake is sent, so a
client can start stepping without asking for a reset it did not want. A later
explicit RESET still restarts the arenas.

## Requests

| Opcode | Name | Payload |
|---|---|---|
| `0x01` | STEP | u32 env count, then `x` and `y` as `f32` per env, in env order |
| `0x02` | RESET | optional u64 seed (presence flag, then value) |
| `0x03` | CLOSE | none |

An action is a direction in world coordinates, `+y` up. Its **length is not
speed**: only the heading is read, and the simulation normalises it. A
direction shorter than the simulation's idle floor means standing still, which
is how an agent says "nowhere" when every heading looks equally bad — and the
only way it can, since a continuous policy never samples exactly zero.

The length prefix counts **envs, not floats**, so a payload that lost half its
bytes is a length error rather than half a batch of plausible directions
applied to the wrong arenas. Each env costs eight bytes, and the payload cap is
checked against that byte cost.

A STEP is read in full before any complaint about its contents, so a rejected
message does not leave unread bytes in the pipe for the next read to trip over.
It is then validated completely before any env moves: the direction count must
equal the env count, and every component must be finite. A NaN would reach the
player's position and from there every value of every observation, so it is
refused rather than repaired. A failure leaves the batch untouched.

## Responses

### `0x82` STEP

| Field | Type |
|---|---|
| env count | u32 |
| per env: frame | u32 |
| per env: terminated | u8 |
| per env: truncated | u8 |
| per env: enemy deaths | u32 |
| per env: episode seed | u64 |
| per env: reset seed | optional u64 |
| observations | float array, `envs * observation_values` |
| terminal count | u32 |
| per terminal: env index | u32 |
| per terminal: observation | float array |

`observations` is what each env is on **now**. For an env that finished, that
is the first frame of its replacement episode. The finished episode's last
observation is in the terminal section, under that env's index.

`frame` counts the frames of the episode this transition belongs to, including
this one. Exactly one step contributes reward, and an auto-reset's first
observation contributes none.

`terminated` and `truncated` may both be set on the same frame: a player can
die on the frame the budget runs out. A learner bootstraps from the terminal
observation when truncation is the only reason the episode ended.

### `0x83` RESET

| Field | Type |
|---|---|
| env count | u32 |
| per env: episode seed | u64 |
| observations | float array |

A reset produces no transition, so it carries no flags, no death count and no
reward.

### `0x84` CLOSED

No payload. The server exits after writing it.

### `0xFF` ERROR

`u32` code, then a byte string message capped at 1,024 bytes. An error ends the
session; the server never answers with a partially stepped batch.

## Episode seeds

```
episode_seed(root, env, episode)
```

is counter-based with domain separation, so an env's seed stream depends only
on its own index and its own episode count -- never on how many workers there
are, on which env finished first, or on the order results came back in.

Fixed vectors, checked by both implementations:

| root | env | episode | seed |
|---|---|---|---|
| 0 | 0 | 0 | `0x03c945febf14bb41` |
| 42 | 0 | 0 | `0xf180f60c32205505` |
| 42 | 1 | 0 | `0x9669bf1bfeabf0a8` |
| 42 | 0 | 1 | `0x3f1497312d1e30a8` |

`RESET` with a seed restarts every env and returns the episode counters to
zero. `RESET` without one advances each env's episode counter, continuing the
stream from the same root. An auto-reset advances only the env that finished.

## Sizes

Every declared length is checked against a maximum derived from the session's
env count and observation size before anything is allocated. A length past that
maximum is a protocol error, not a large allocation.
