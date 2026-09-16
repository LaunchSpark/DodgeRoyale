//! The byte schema the trainer speaks, version 1.
//!
//! Frozen before the Python client exists, so both sides are written against
//! the document rather than against each other. Everything is explicit: fixed
//! width, little endian, length-prefixed. No Rust struct is ever handed to the
//! wire, so layout, padding and `usize` cannot leak into the format.
//!
//! See `docs/superpowers/specs/velocity-flow-royale-protocol-v1.md`.

use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};

use crate::observation::Layout;

/// Identifies the stream, and catches a client pointed at the wrong binary.
pub const MAGIC: [u8; 8] = *b"DRGYM\0\0\x01";

/// Bumped whenever a message's meaning changes.
pub const PROTOCOL_VERSION: u32 = 1;

/// Opcodes a client may send.
pub mod request {
    pub const STEP: u8 = 0x01;
    pub const RESET: u8 = 0x02;
    pub const CLOSE: u8 = 0x03;
}

/// Opcodes the arena server sends.
pub mod response {
    pub const HANDSHAKE: u8 = 0x81;
    pub const STEP: u8 = 0x82;
    pub const RESET: u8 = 0x83;
    pub const CLOSED: u8 = 0x84;
    pub const ERROR: u8 = 0xFF;
}

/// Why a message could not be read or written.
#[derive(Debug)]
pub enum ProtocolError {
    Io(io::Error),
    /// The stream did not start with [`MAGIC`].
    BadMagic,
    /// A version this build does not speak.
    UnsupportedVersion(u32),
    /// An opcode outside the two tables.
    UnknownOpcode(u8),
    /// A payload larger than the configured maximum.
    PayloadTooLarge {
        declared: u64,
        maximum: u64,
    },
    /// The stream ended inside a message.
    Truncated,
    /// A STEP carrying the wrong number of actions.
    ActionCount {
        got: usize,
        expected: usize,
    },
    /// An action byte outside the nine actions.
    InvalidAction(u8),
    /// A string field that was not UTF-8, or JSON that did not parse.
    Malformed(String),
}

impl From<io::Error> for ProtocolError {
    fn from(error: io::Error) -> Self {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            Self::Truncated
        } else {
            Self::Io(error)
        }
    }
}

impl core::fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::BadMagic => write!(formatter, "the stream is not a DodgeRoyale gym stream"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "protocol version {version} is not supported")
            }
            Self::UnknownOpcode(code) => write!(formatter, "unknown opcode {code:#04x}"),
            Self::PayloadTooLarge { declared, maximum } => write!(
                formatter,
                "payload of {declared} bytes exceeds the {maximum} byte maximum"
            ),
            Self::Truncated => write!(formatter, "the stream ended inside a message"),
            Self::ActionCount { got, expected } => {
                write!(formatter, "expected {expected} actions, got {got}")
            }
            Self::InvalidAction(byte) => write!(formatter, "action {byte} is not one of the nine"),
            Self::Malformed(reason) => write!(formatter, "malformed message: {reason}"),
        }
    }
}

impl core::error::Error for ProtocolError {}

/// What the server announces before any stepping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Handshake {
    pub protocol_version: u32,
    pub envs: u32,
    pub enemy_count: u32,
    pub max_frames: u32,
    pub root_seed: u64,
    pub workers: u32,
    /// How the observation is laid out, including channel names and sections.
    pub layout: Layout,
}

/// What a client asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// One action per env, in env order.
    Step(Vec<u8>),
    /// Restart every env. `Some` restarts the seed stream from that root;
    /// `None` continues it, which is why the seed carries a presence flag --
    /// zero is a perfectly good seed.
    Reset(Option<u64>),
    /// Finish, acknowledge, and exit.
    Close,
}

/// What happened to one env during one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EnvTransition {
    /// Frames the finished-or-continuing episode had run, counting this one.
    pub frame: u32,
    pub terminated: bool,
    pub truncated: bool,
    pub enemy_deaths: u32,
    /// The seed of the episode this transition belongs to.
    pub episode_seed: u64,
    /// The seed of the episode that replaced it, if this one ended.
    pub reset_seed: Option<u64>,
}

impl EnvTransition {
    /// Whether the episode ended on this frame.
    #[must_use]
    pub const fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// The response to a STEP.
///
/// `observations` holds the observation each env is now on: for an env that
/// finished, that is its new episode's first frame. The final observation of
/// the episode that ended travels separately, in `terminal`, because a learner
/// needs both.
#[derive(Debug, Clone, PartialEq)]
pub struct StepBatch {
    pub transitions: Vec<EnvTransition>,
    pub observations: Vec<f32>,
    pub terminal: Vec<TerminalObservation>,
}

/// One finished episode's last observation.
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalObservation {
    pub env: u32,
    pub observation: Vec<f32>,
}

/// The response to a RESET.
#[derive(Debug, Clone, PartialEq)]
pub struct ResetBatch {
    pub seeds: Vec<u64>,
    pub observations: Vec<f32>,
}

/// Domain separation for the episode seed stream.
const SEED_DOMAIN: u64 = 0x444f_4447_4559_4d00;

/// The seed for one env's `episode_index`-th episode.
///
/// Counter-based on purpose: an env's seeds do not depend on when it finished,
/// how many workers there are, or which order results came back in.
#[must_use]
#[expect(
    clippy::as_conversions,
    reason = "Widening u32 to u64 is exact, and From is not const yet"
)]
pub const fn episode_seed(root: u64, env: u32, episode: u64) -> u64 {
    let mut value = root ^ SEED_DOMAIN;
    value = mix(value ^ (env as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    mix(value ^ episode.wrapping_mul(0xbf58_476d_1ce4_e5b9))
}

/// The splitmix64 finaliser, which spreads neighbouring inputs across the range.
const fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// The largest payload this session will accept, derived from its env count.
///
/// A step's payload is bounded by every env reporting a full observation plus a
/// terminal one, so anything beyond that is a malformed length rather than a
/// message worth allocating for.
#[must_use]
pub fn max_payload(envs: u32, observation_values: usize) -> u64 {
    let per_env =
        u64::try_from(observation_values.saturating_mul(4).saturating_add(64)).unwrap_or(u64::MAX);
    let envs = u64::from(envs).saturating_add(1);
    envs.saturating_mul(per_env)
        .saturating_mul(2)
        .saturating_add(4_096)
}

// --- writing ------------------------------------------------------------

fn write_u8<W: Write>(writer: &mut W, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// An optional seed: a presence flag, then the value, always both.
fn write_optional_u64<W: Write>(writer: &mut W, value: Option<u64>) -> io::Result<()> {
    write_u8(writer, u8::from(value.is_some()))?;
    write_u64(writer, value.unwrap_or(0))
}

fn write_bytes<W: Write>(writer: &mut W, bytes: &[u8]) -> io::Result<()> {
    write_u32(writer, u32::try_from(bytes.len()).unwrap_or(u32::MAX))?;
    writer.write_all(bytes)
}

/// Floats converted per pass, in each direction.
///
/// A whole observation is 28,782 values, and a 64-env batch is 1.8M of them.
/// Converting one at a time costs a call into the writer for every four bytes,
/// which measured about a fourteenth of the pipe's own speed. A block amortises
/// that without holding a second copy of the batch: 4 KiB, which stays in L1
/// and never scales with the env count.
const FLOAT_BLOCK: usize = 1_024;

/// A float array: a `u32` count, then that many little-endian `f32`.
///
/// Byte for byte what writing them one at a time produced; only the number of
/// calls changed, which is why `PROTOCOL_VERSION` does not move.
fn write_floats<W: Write>(writer: &mut W, values: &[f32]) -> io::Result<()> {
    write_u32(writer, u32::try_from(values.len()).unwrap_or(u32::MAX))?;
    // One buffer for the whole array, refilled per block and never regrown.
    let mut block = Vec::with_capacity(FLOAT_BLOCK.saturating_mul(4));
    for values in values.chunks(FLOAT_BLOCK) {
        block.clear();
        for value in values {
            block.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&block)?;
    }
    Ok(())
}

/// Announce the stream: magic, version, then the handshake as JSON.
///
/// # Errors
///
/// Any write failure, or a handshake that cannot be serialised.
pub fn write_handshake<W: Write>(
    writer: &mut W,
    handshake: &Handshake,
) -> Result<(), ProtocolError> {
    writer.write_all(&MAGIC)?;
    write_u32(writer, PROTOCOL_VERSION)?;
    write_u8(writer, response::HANDSHAKE)?;
    let json = serde_json::to_vec(handshake)
        .map_err(|error| ProtocolError::Malformed(error.to_string()))?;
    write_bytes(writer, &json)?;
    writer.flush()?;
    Ok(())
}

/// Send one step's results.
///
/// # Errors
///
/// Any write failure.
pub fn write_step<W: Write>(writer: &mut W, batch: &StepBatch) -> Result<(), ProtocolError> {
    write_u8(writer, response::STEP)?;
    write_u32(
        writer,
        u32::try_from(batch.transitions.len()).unwrap_or(u32::MAX),
    )?;
    for transition in &batch.transitions {
        write_u32(writer, transition.frame)?;
        write_u8(writer, u8::from(transition.terminated))?;
        write_u8(writer, u8::from(transition.truncated))?;
        write_u32(writer, transition.enemy_deaths)?;
        write_u64(writer, transition.episode_seed)?;
        // A presence flag, so that seed zero stays a usable seed.
        write_optional_u64(writer, transition.reset_seed)?;
    }
    write_floats(writer, &batch.observations)?;
    write_u32(
        writer,
        u32::try_from(batch.terminal.len()).unwrap_or(u32::MAX),
    )?;
    for terminal in &batch.terminal {
        write_u32(writer, terminal.env)?;
        write_floats(writer, &terminal.observation)?;
    }
    writer.flush()?;
    Ok(())
}

/// Send a reset's results: no transition, and so no reward, ever.
///
/// # Errors
///
/// Any write failure.
pub fn write_reset<W: Write>(writer: &mut W, batch: &ResetBatch) -> Result<(), ProtocolError> {
    write_u8(writer, response::RESET)?;
    write_u32(writer, u32::try_from(batch.seeds.len()).unwrap_or(u32::MAX))?;
    for seed in &batch.seeds {
        write_u64(writer, *seed)?;
    }
    write_floats(writer, &batch.observations)?;
    writer.flush()?;
    Ok(())
}

/// Acknowledge a CLOSE.
///
/// # Errors
///
/// Any write failure.
pub fn write_closed<W: Write>(writer: &mut W) -> Result<(), ProtocolError> {
    write_u8(writer, response::CLOSED)?;
    writer.flush()?;
    Ok(())
}

/// Report a failure and end the session; never a partial batch.
///
/// # Errors
///
/// Any write failure.
pub fn write_error<W: Write>(
    writer: &mut W,
    code: u32,
    message: &str,
) -> Result<(), ProtocolError> {
    write_u8(writer, response::ERROR)?;
    write_u32(writer, code)?;
    // Bounded: a diagnostic must never become a denial of service.
    let bytes = message.as_bytes();
    let capped = bytes.get(..bytes.len().min(1_024)).unwrap_or(bytes);
    write_bytes(writer, capped)?;
    writer.flush()?;
    Ok(())
}

/// Send one request.
///
/// # Errors
///
/// Any write failure.
pub fn write_request<W: Write>(writer: &mut W, request: &Request) -> Result<(), ProtocolError> {
    match request {
        Request::Step(actions) => {
            write_u8(writer, request::STEP)?;
            write_bytes(writer, actions)?;
        }
        Request::Reset(seed) => {
            write_u8(writer, request::RESET)?;
            write_optional_u64(writer, *seed)?;
        }
        Request::Close => write_u8(writer, request::CLOSE)?,
    }
    writer.flush()?;
    Ok(())
}

// --- reading ------------------------------------------------------------

fn read_exact<R: Read>(reader: &mut R, buffer: &mut [u8]) -> Result<(), ProtocolError> {
    reader.read_exact(buffer).map_err(ProtocolError::from)
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8, ProtocolError> {
    let mut byte = [0_u8; 1];
    read_exact(reader, &mut byte)?;
    byte.first().copied().ok_or(ProtocolError::Truncated)
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, ProtocolError> {
    let mut bytes = [0_u8; 4];
    read_exact(reader, &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, ProtocolError> {
    let mut bytes = [0_u8; 8];
    read_exact(reader, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_length<R: Read>(reader: &mut R, maximum: u64) -> Result<usize, ProtocolError> {
    let declared = u64::from(read_u32(reader)?);
    if declared > maximum {
        return Err(ProtocolError::PayloadTooLarge { declared, maximum });
    }
    usize::try_from(declared).map_err(|_| ProtocolError::PayloadTooLarge { declared, maximum })
}

fn read_bytes<R: Read>(reader: &mut R, maximum: u64) -> Result<Vec<u8>, ProtocolError> {
    let length = read_length(reader, maximum)?;
    let mut bytes = vec![0_u8; length];
    read_exact(reader, &mut bytes)?;
    Ok(bytes)
}

/// Read a float array, in blocks rather than four bytes at a time.
///
/// The declared count is still checked against `maximum` before anything is
/// allocated, and the block is a fixed 4 KiB regardless of what was declared,
/// so a hostile length cannot turn into a large read buffer.
fn read_floats<R: Read>(reader: &mut R, maximum: u64) -> Result<Vec<f32>, ProtocolError> {
    let count = read_length(reader, maximum)?;
    let mut values = Vec::with_capacity(count);
    let mut block = vec![0_u8; FLOAT_BLOCK.saturating_mul(4)];

    let mut remaining = count;
    while remaining > 0 {
        let taking = remaining.min(FLOAT_BLOCK);
        let Some(bytes) = block.get_mut(..taking.saturating_mul(4)) else {
            // Unreachable: `taking` is capped at the block's own length.
            return Err(ProtocolError::Malformed(
                "a float block outgrew its buffer".to_owned(),
            ));
        };
        read_exact(reader, bytes)?;
        for word in bytes.chunks_exact(4) {
            let quad: [u8; 4] = word.try_into().map_err(|_| ProtocolError::Truncated)?;
            values.push(f32::from_le_bytes(quad));
        }
        remaining = remaining.saturating_sub(taking);
    }
    Ok(values)
}

/// Read the stream's opening.
///
/// # Errors
///
/// [`ProtocolError::BadMagic`] for another program's output,
/// [`ProtocolError::UnsupportedVersion`] for a version this build cannot read.
pub fn read_handshake<R: Read>(reader: &mut R) -> Result<Handshake, ProtocolError> {
    let mut magic = [0_u8; 8];
    read_exact(reader, &mut magic)?;
    if magic != MAGIC {
        return Err(ProtocolError::BadMagic);
    }
    let version = read_u32(reader)?;
    if version != PROTOCOL_VERSION {
        return Err(ProtocolError::UnsupportedVersion(version));
    }
    let opcode = read_u8(reader)?;
    if opcode != response::HANDSHAKE {
        return Err(ProtocolError::UnknownOpcode(opcode));
    }
    let json = read_bytes(reader, 1 << 20)?;
    serde_json::from_slice(&json).map_err(|error| ProtocolError::Malformed(error.to_string()))
}

/// Read one request, or `None` if the stream ended cleanly between messages.
///
/// A client that simply goes away is a normal shutdown; a client that stops
/// halfway through a message is not, and that difference is only visible here,
/// at the message boundary.
///
/// # Errors
///
/// As [`read_request`].
pub fn read_request_or_eof<R: Read>(
    reader: &mut R,
    envs: usize,
    maximum: u64,
) -> Result<Option<Request>, ProtocolError> {
    let mut opcode = [0_u8; 1];
    match reader.read(&mut opcode) {
        Ok(0) => return Ok(None),
        Ok(_) => {}
        Err(error) => return Err(ProtocolError::from(error)),
    }
    let opcode = opcode.first().copied().ok_or(ProtocolError::Truncated)?;
    read_request_body(reader, opcode, envs, maximum).map(Some)
}

/// Read one request, validating it completely before it can act on anything.
///
/// # Errors
///
/// A malformed or truncated request, a wrong action count, or an action byte
/// outside the nine. None of these step any env.
pub fn read_request<R: Read>(
    reader: &mut R,
    envs: usize,
    maximum: u64,
) -> Result<Request, ProtocolError> {
    let opcode = read_u8(reader)?;
    read_request_body(reader, opcode, envs, maximum)
}

fn read_request_body<R: Read>(
    reader: &mut R,
    opcode: u8,
    envs: usize,
    maximum: u64,
) -> Result<Request, ProtocolError> {
    match opcode {
        request::STEP => {
            let actions = read_bytes(reader, maximum)?;
            if actions.len() != envs {
                return Err(ProtocolError::ActionCount {
                    got: actions.len(),
                    expected: envs,
                });
            }
            if let Some(bad) = actions
                .iter()
                .copied()
                .find(|byte| crate::simulation::Action::from_byte(*byte).is_none())
            {
                return Err(ProtocolError::InvalidAction(bad));
            }
            Ok(Request::Step(actions))
        }
        request::RESET => {
            let present = read_u8(reader)?;
            let seed = read_u64(reader)?;
            Ok(Request::Reset((present != 0).then_some(seed)))
        }
        request::CLOSE => Ok(Request::Close),
        other => Err(ProtocolError::UnknownOpcode(other)),
    }
}

/// Read a step response.
///
/// # Errors
///
/// A malformed or truncated response.
pub fn read_step<R: Read>(reader: &mut R, maximum: u64) -> Result<StepBatch, ProtocolError> {
    let opcode = read_u8(reader)?;
    if opcode != response::STEP {
        return Err(ProtocolError::UnknownOpcode(opcode));
    }
    let count = read_length(reader, maximum)?;
    let mut transitions = Vec::with_capacity(count);
    for _ in 0..count {
        let frame = read_u32(reader)?;
        let terminated = read_u8(reader)? != 0;
        let truncated = read_u8(reader)? != 0;
        let enemy_deaths = read_u32(reader)?;
        let episode_seed = read_u64(reader)?;
        let has_reset = read_u8(reader)? != 0;
        let reset_seed = read_u64(reader)?;
        transitions.push(EnvTransition {
            frame,
            terminated,
            truncated,
            enemy_deaths,
            episode_seed,
            reset_seed: has_reset.then_some(reset_seed),
        });
    }
    let observations = read_floats(reader, maximum)?;
    let terminal_count = read_length(reader, maximum)?;
    let mut terminal = Vec::with_capacity(terminal_count);
    for _ in 0..terminal_count {
        let env = read_u32(reader)?;
        let observation = read_floats(reader, maximum)?;
        terminal.push(TerminalObservation { env, observation });
    }
    Ok(StepBatch {
        transitions,
        observations,
        terminal,
    })
}

/// Read a reset response.
///
/// # Errors
///
/// A malformed or truncated response.
pub fn read_reset<R: Read>(reader: &mut R, maximum: u64) -> Result<ResetBatch, ProtocolError> {
    let opcode = read_u8(reader)?;
    if opcode != response::RESET {
        return Err(ProtocolError::UnknownOpcode(opcode));
    }
    let count = read_length(reader, maximum)?;
    let mut seeds = Vec::with_capacity(count);
    for _ in 0..count {
        seeds.push(read_u64(reader)?);
    }
    let observations = read_floats(reader, maximum)?;
    Ok(ResetBatch {
        seeds,
        observations,
    })
}

#[cfg(test)]
mod tests;
