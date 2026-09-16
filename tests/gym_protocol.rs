//! Drives the real `dodge-royale gym` process.
//!
//! The in-memory codec tests cannot see stdout contamination, a missing flush,
//! or a startup path that reaches for a database, so these spawn the binary.
//! Every test has a deadline and kills its child on the way out, including
//! when an assertion fails.

// clippy.toml allows these inside tests, but its detection only reaches
// #[test] functions; the helpers below are ordinary functions in a test-only
// binary, where a failed expectation is exactly the intended failure.
#![expect(
    clippy::expect_used,
    reason = "Test-only binary: a failed expectation is how a test reports"
)]

use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use dodge_royale::gym::protocol::{
    ProtocolError, Request, read_handshake, read_reset, read_step, write_request,
};
use dodge_royale::observation::OBSERVATION_VALUES;

/// Long enough for a debug build to fill a small arena, short enough that a
/// hung child fails the test rather than the suite.
const DEADLINE: Duration = Duration::from_secs(90);

/// A running gym, killed when the test ends however it ends.
///
/// The child is shared with a watchdog thread, so a deadline can kill it from
/// the outside. That is what unblocks a test parked in a read on a server that
/// has stopped answering: without it, the reader waits for a pipe that will
/// never close, and no destructor runs while it does.
struct Gym {
    child: Arc<Mutex<Child>>,
    /// Optional so a test can drop it, which is the EOF the server exits on.
    stdin: Option<std::process::ChildStdin>,
    stdout: std::process::ChildStdout,
    /// Tells the watchdog the test finished in time.
    finished: Option<mpsc::Sender<()>>,
    watchdog: Option<thread::JoinHandle<()>>,
}

impl Gym {
    fn start(args: &[&str]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_dodge-royale"));
        command
            .arg("gym")
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // A database URL that could never connect: the gym must not touch
            // it, so its presence proves the command bypasses startup.
            .env("DATABASE_URL", "postgres://nobody@127.0.0.1:1/never");
        let mut child = command.spawn().expect("the gym binary starts");
        let stdin = child.stdin.take().expect("a stdin pipe");
        let stdout = child.stdout.take().expect("a stdout pipe");
        // Stderr is drained continuously; a full pipe would otherwise block the
        // child forever while we wait for stdout.
        if let Some(mut stderr) = child.stderr.take() {
            thread::spawn(move || {
                let mut sink = Vec::new();
                let _ = stderr.read_to_end(&mut sink);
            });
        }
        let child = Arc::new(Mutex::new(child));
        let (finished, done) = mpsc::channel();
        let watched = Arc::clone(&child);
        let watchdog = thread::spawn(move || {
            // Either the test says it is done, or the deadline passes and the
            // child is killed so every pipe read returns.
            if done.recv_timeout(DEADLINE) == Err(RecvTimeoutError::Timeout)
                && let Ok(mut child) = watched.lock()
            {
                let _ = child.kill();
                let _ = child.wait();
            }
        });

        Self {
            child,
            stdin: Some(stdin),
            stdout,
            finished: Some(finished),
            watchdog: Some(watchdog),
        }
    }

    fn send(&mut self, request: &Request) {
        let stdin = self.stdin.as_mut().expect("stdin is still open");
        write_request(stdin, request).expect("the request is written");
        stdin.flush().expect("and flushed");
    }

    /// Hang up, which is what a trainer that simply exits looks like.
    fn hang_up(&mut self) {
        self.stdin.take();
    }

    fn wait(&self) -> std::process::ExitStatus {
        let mut child = self.child.lock().expect("the child is not poisoned");
        child.wait().expect("the child exits")
    }
}

impl Drop for Gym {
    fn drop(&mut self) {
        // Stop the watchdog first, then reap: a killed child whose status is
        // never collected is a zombie.
        self.finished.take();
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
    }
}

const SMALL: [&str; 10] = [
    "--envs",
    "2",
    "--threads",
    "2",
    "--enemies",
    "4",
    "--max-frames",
    "3",
    "--seed",
    "7",
];

#[test]
fn the_opening_is_a_handshake_and_frame_zero() {
    let mut gym = Gym::start(&SMALL);
    let handshake = read_handshake(&mut gym.stdout).expect("a handshake arrives");
    assert_eq!(handshake.envs, 2);
    assert_eq!(handshake.root_seed, 7);
    assert_eq!(handshake.layout.observation_values, OBSERVATION_VALUES);
    assert_eq!(handshake.layout.channels.len(), 7);
    assert_eq!(handshake.layout.actions.len(), 9);

    let initial = read_reset(&mut gym.stdout, 1 << 30).expect("frame zero arrives");
    assert_eq!(initial.seeds.len(), 2);
    assert_eq!(initial.observations.len(), 2 * OBSERVATION_VALUES);
    assert!(
        initial.observations.iter().all(|value| value.is_finite()),
        "an observation is all finite numbers"
    );

    gym.send(&Request::Close);
    assert!(gym.wait().success(), "a closed session exits cleanly");
}

#[test]
fn stepping_reports_transitions_and_auto_resets_at_the_budget() {
    let mut gym = Gym::start(&SMALL);
    read_handshake(&mut gym.stdout).expect("a handshake");
    read_reset(&mut gym.stdout, 1 << 30).expect("frame zero");

    // The budget is three frames, so the third step ends both episodes.
    for frame in 1..=3_u32 {
        gym.send(&Request::Step(vec![2, 5]));
        let batch = read_step(&mut gym.stdout, 1 << 30).expect("a step response");
        assert_eq!(batch.transitions.len(), 2);
        assert_eq!(batch.observations.len(), 2 * OBSERVATION_VALUES);
        for transition in &batch.transitions {
            assert_eq!(transition.frame, frame);
            assert_eq!(transition.truncated, frame == 3);
        }
        if frame == 3 {
            assert_eq!(batch.terminal.len(), 2, "both episodes ended");
            for terminal in &batch.terminal {
                assert_eq!(terminal.observation.len(), OBSERVATION_VALUES);
            }
            for transition in &batch.transitions {
                assert!(
                    transition.reset_seed.is_some(),
                    "and each carried a new episode's seed"
                );
            }
        } else {
            assert!(batch.terminal.is_empty());
        }
    }

    gym.send(&Request::Close);
    assert!(gym.wait().success());
}

#[test]
fn a_seeded_reset_reproduces_frame_zero() {
    let mut gym = Gym::start(&SMALL);
    read_handshake(&mut gym.stdout).expect("a handshake");
    let first = read_reset(&mut gym.stdout, 1 << 30).expect("frame zero");

    gym.send(&Request::Step(vec![1, 1]));
    read_step(&mut gym.stdout, 1 << 30).expect("a step");

    gym.send(&Request::Reset(Some(7)));
    let again = read_reset(&mut gym.stdout, 1 << 30).expect("a reset");
    assert_eq!(again.seeds, first.seeds);
    assert_eq!(
        again.observations, first.observations,
        "the same root seed is the same arena"
    );

    gym.send(&Request::Close);
    assert!(gym.wait().success());
}

#[test]
fn the_session_ends_cleanly_when_the_client_goes_away() {
    let mut gym = Gym::start(&SMALL);
    read_handshake(&mut gym.stdout).expect("a handshake");
    read_reset(&mut gym.stdout, 1 << 30).expect("frame zero");
    gym.hang_up();
    assert!(
        gym.wait().success(),
        "an EOF between messages is an ordinary exit"
    );
}

#[test]
fn a_malformed_request_is_an_error_record_and_a_failed_exit() {
    let mut gym = Gym::start(&SMALL);
    read_handshake(&mut gym.stdout).expect("a handshake");
    read_reset(&mut gym.stdout, 1 << 30).expect("frame zero");

    // An action outside the nine: the batch must not step at all.
    gym.send(&Request::Step(vec![0, 200]));
    let error = read_step(&mut gym.stdout, 1 << 30);
    assert!(
        matches!(error, Err(ProtocolError::UnknownOpcode(0xFF))),
        "the answer is an ERROR record, not a batch: {error:?}"
    );
    assert!(
        !gym.wait().success(),
        "and the process exits non-zero rather than carrying on"
    );
}

#[test]
fn a_step_with_the_wrong_action_count_never_steps_an_env() {
    let mut gym = Gym::start(&SMALL);
    read_handshake(&mut gym.stdout).expect("a handshake");
    read_reset(&mut gym.stdout, 1 << 30).expect("frame zero");

    gym.send(&Request::Step(vec![0]));
    let error = read_step(&mut gym.stdout, 1 << 30);
    assert!(matches!(error, Err(ProtocolError::UnknownOpcode(0xFF))));
    assert!(!gym.wait().success());
}

#[test]
fn protocol_output_carries_nothing_but_protocol() {
    let mut gym = Gym::start(&SMALL);
    read_handshake(&mut gym.stdout).expect("a handshake");
    read_reset(&mut gym.stdout, 1 << 30).expect("frame zero");
    gym.send(&Request::Step(vec![0, 0]));
    read_step(&mut gym.stdout, 1 << 30).expect("a step");
    gym.send(&Request::Close);

    // Everything after the acknowledgement must be end of stream: a stray
    // log line on stdout would show up here.
    let mut rest = Vec::new();
    gym.stdout.read_to_end(&mut rest).expect("stdout drains");
    assert_eq!(
        rest,
        vec![0x84],
        "the close acknowledgement, and not one byte more"
    );
    assert!(gym.wait().success());
}
