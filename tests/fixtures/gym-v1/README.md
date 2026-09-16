# Golden fixtures, gym protocol v1

Real protocol bytes, committed, so the Python client can be tested without a
Rust toolchain, a build, or a running gym. Nothing here is generated at test
time: `manifest.json` describes files that are already in the repository, and
both languages read the same bytes.

Regenerate after an intentional protocol or encoder change:

```sh
cargo run --locked --no-default-features --example gym_fixtures
```

That rewrites every file here. It is a maintenance action, not a test step —
see [`examples/gym_fixtures.rs`](../../../examples/gym_fixtures.rs).

## Files

| File | Message | What it is for |
|---|---|---|
| `handshake.bin` | `0x81` | Magic, version, and the layout as JSON. |
| `reset-frame-zero.bin` | `0x83` | The unasked frame zero that follows the handshake. |
| `step-running.bin` | `0x82` | Two ordinary transitions: no terminal section, no replacement seed. |
| `step-auto-reset.bin` | `0x82` | Both envs hit the frame budget: terminal observations, replacement seeds, `truncated` set. |
| `scenes.bin` | `0x82` | Three hand-placed arenas that pin the encoder's axes, signs and wrapping. |

The first four come from a real `ArenaBatch` at root seed 7, so they are
exactly what the server sends. `scenes.bin` is encoded from arenas built by
hand, because a real episode cannot be relied on to contain a half-grown blast
or a hazard straddling the seam on any chosen frame:

* **env 0** — asymmetric positions and moving hazards. No two hazards share an
  offset, a size or a velocity, so a decoder that transposes or flips an axis
  cannot reproduce it.
* **env 1** — blast phase either side of its peak. A blast passes through every
  width twice, so size alone cannot say which half of its life it is in; the
  two blasts here must come back with opposite signs (+0.84 and -0.84).
* **env 2** — hazards over both wrap seams. Displacement uses
  `torus::wrapped_delta`, so a hazard a few pixels past the edge is a few
  pixels away. A decoder that subtracts positions sees an empty window.

## Comparing values

Each sample in `manifest.json` carries the same number twice:

* `value` — the `f32` widened to `f64`. That widening is exact, and the JSON is
  written with enough digits to round-trip, so a reader whose JSON parser is
  correctly rounded can compare with `==`. Python's `json` is. Rust's
  `serde_json` needs its `float_roundtrip` feature, which this crate enables.
* `bits` — the same value's little-endian `u32`, for a reader that would rather
  compare integers than trust a decimal.

`index` is an offset into one env's observation, not into the batch. For the
batch arrays, add `env * observation_values`; terminal observations are already
one env's own, so the index applies directly.

## What checks these

* [`tests/gym_fixtures.rs`](../../gym_fixtures.rs) reads every file back with
  the shipped codec and checks it against the manifest, so a codec change
  cannot quietly leave Python testing against bytes no Rust test has read.
* `src/gym/protocol/tests.rs` holds hand-written wire images that say what the
  bytes must be from first principles. Those are the independent check: they
  would catch a codec and a fixture set that drifted together.
