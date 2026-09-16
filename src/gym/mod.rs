//! The batch environment the trainer drives.
//!
//! Native only: it speaks a binary protocol over stdin and stdout, owns the
//! arenas, and never opens a window, a database or an async runtime.

pub mod protocol;
pub mod workers;
