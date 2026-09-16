//! The request loop: a handshake, then step and reset until told to stop.
//!
//! Stdout carries protocol bytes and nothing else. Everything a human might
//! want to read goes to stderr, because a stray `println!` here would be
//! indistinguishable from an observation.

use std::io::{Read, Write};

use crate::observation::{OBSERVATION_VALUES, layout, max_path_pixels};

use super::protocol::{
    Handshake, PROTOCOL_VERSION, ProtocolError, Request, max_payload, read_request_or_eof,
    write_closed, write_error, write_handshake, write_reset, write_step,
};
use super::workers::{ArenaBatch, BatchConfig, BatchError};

/// Error codes carried by an ERROR record.
pub mod code {
    /// The request could not be parsed, or asked for something impossible.
    pub const PROTOCOL: u32 = 1;
    /// An arena or a worker failed.
    pub const ARENA: u32 = 2;
}

/// Why the session ended badly.
#[derive(Debug)]
pub enum ServerError {
    Protocol(ProtocolError),
    Batch(BatchError),
}

impl core::fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Protocol(error) => write!(formatter, "{error}"),
            Self::Batch(error) => write!(formatter, "{error}"),
        }
    }
}

impl core::error::Error for ServerError {}

impl From<ProtocolError> for ServerError {
    fn from(error: ProtocolError) -> Self {
        Self::Protocol(error)
    }
}

impl From<BatchError> for ServerError {
    fn from(error: BatchError) -> Self {
        Self::Batch(error)
    }
}

/// Run one session to its end.
///
/// The opening is the handshake and then frame zero, so a client can start
/// stepping without asking for a reset it does not want.
///
/// # Errors
///
/// A malformed request or a failing arena. Either is reported as an ERROR
/// record first, then returned, because a half-stepped batch is worse than no
/// batch at all.
pub fn serve<R: Read, W: Write>(
    config: BatchConfig,
    reader: &mut R,
    writer: &mut W,
) -> Result<(), ServerError> {
    // Paths are allowed to leave the window -- the reader samples them with
    // border padding -- but a hold long enough to make that usual is worth
    // saying out loud, once, where a human will see it.
    let reach = max_path_pixels(config.hold_frames);
    if reach > crate::observation::WINDOW_HALF {
        eprintln!(
            "warning: with hold-frames {}, a path can reach {reach:.0} reference pixels, \
             past the {:.0} pixel half window; far samples will read the window's border",
            config.hold_frames,
            crate::observation::WINDOW_HALF
        );
    }

    let handshake = Handshake {
        protocol_version: PROTOCOL_VERSION,
        envs: config.envs,
        enemy_count: u32::try_from(config.enemy_count).unwrap_or(u32::MAX),
        max_frames: config.max_frames,
        root_seed: config.root_seed,
        workers: config.workers,
        layout: layout(config.hold_frames),
    };
    write_handshake(writer, &handshake)?;

    let (mut batch, initial) = match ArenaBatch::start(config) {
        Ok(started) => started,
        Err(error) => {
            write_error(writer, code::ARENA, &error.to_string())?;
            return Err(ServerError::Batch(error));
        }
    };
    write_reset(writer, &initial)?;

    let envs = usize::try_from(config.envs).unwrap_or(usize::MAX);
    let maximum = max_payload(config.envs, OBSERVATION_VALUES);

    loop {
        let request = match read_request_or_eof(reader, envs, maximum) {
            // A client that goes away between messages is an ordinary exit.
            Ok(None) => return Ok(()),
            Ok(Some(request)) => request,
            Err(error) => {
                write_error(writer, code::PROTOCOL, &error.to_string())?;
                return Err(ServerError::Protocol(error));
            }
        };

        match request {
            Request::Step(actions) => match batch.step(&actions) {
                Ok(result) => write_step(writer, &result)?,
                Err(error) => {
                    write_error(writer, code::ARENA, &error.to_string())?;
                    return Err(ServerError::Batch(error));
                }
            },
            Request::Reset(seed) => match batch.reset(seed) {
                Ok(result) => write_reset(writer, &result)?,
                Err(error) => {
                    write_error(writer, code::ARENA, &error.to_string())?;
                    return Err(ServerError::Batch(error));
                }
            },
            Request::Close => {
                write_closed(writer)?;
                return Ok(());
            }
        }
    }
}
