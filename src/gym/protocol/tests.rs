//! Byte-level tests for protocol v2.
//!
//! These pin the format: a change that breaks one of these breaks every
//! trained checkpoint's ability to be fed, so it needs a version bump.

#![expect(
    clippy::as_conversions,
    reason = "Fixed byte offsets in hand-written fixtures"
)]

use crate::gym::protocol::*;
use crate::observation::{OBSERVATION_VALUES, layout};
use crate::simulation::DEFAULT_HOLD_FRAMES;

fn handshake() -> Handshake {
    Handshake {
        protocol_version: PROTOCOL_VERSION,
        envs: 2,
        enemy_count: 100,
        max_frames: 3_600,
        root_seed: 42,
        workers: 2,
        layout: layout(DEFAULT_HOLD_FRAMES),
    }
}

#[test]
fn a_handshake_round_trips_with_its_whole_layout() {
    let mut bytes = Vec::new();
    write_handshake(&mut bytes, &handshake()).expect("writing a handshake");

    assert_eq!(
        bytes.get(..8).expect("the magic"),
        b"DRGYM\0\0\x02",
        "the stream identifies itself first, and says which version it speaks"
    );
    let read = read_handshake(&mut bytes.as_slice()).expect("reading it back");
    assert_eq!(read, handshake());
    assert_eq!(read.layout.observation_values, OBSERVATION_VALUES);
    assert_eq!(
        read.layout.channels.first().map(String::as_str),
        Some("normal-enemy"),
        "channel names travel with the data"
    );
}

#[test]
fn another_programs_output_is_rejected_rather_than_parsed() {
    let mut noise = b"hello there, this is not a gym".to_vec();
    noise.extend_from_slice(&[0_u8; 64]);
    assert!(matches!(
        read_handshake(&mut noise.as_slice()),
        Err(ProtocolError::BadMagic)
    ));
}

#[test]
fn a_future_version_is_refused_before_anything_is_built() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&99_u32.to_le_bytes());
    match read_handshake(&mut bytes.as_slice()) {
        Err(ProtocolError::UnsupportedVersion(99)) => {}
        other => panic!("expected an unsupported version, got {other:?}"),
    }
}

#[test]
fn a_step_request_is_bytes_we_can_write_out_by_hand() {
    let mut bytes = Vec::new();
    let actions = vec![Vec2::ZERO, Vec2::new(1.0, -0.5)];
    write_request(&mut bytes, &Request::Step(actions.clone())).expect("writing a step");
    assert_eq!(
        bytes,
        vec![
            0x01, // STEP
            2, 0, 0, 0, // two envs, not two floats
            0, 0, 0, 0, // env 0 x = 0.0
            0, 0, 0, 0, // env 0 y = 0.0
            0, 0, 0x80, 0x3F, // env 1 x = 1.0
            0, 0, 0, 0xBF, // env 1 y = -0.5
        ],
        "opcode, env count, then x and y per env"
    );

    let request = read_request(&mut bytes.as_slice(), 2, 1_024).expect("reading it back");
    assert_eq!(request, Request::Step(actions));
}

#[test]
fn a_direction_survives_the_wire_exactly() {
    // Not a round number in binary: a client that read the payload as
    // anything but little-endian f32 would come back with a different angle.
    let actions = vec![Vec2::new(0.123_456_79, -0.987_654_3)];
    let mut bytes = Vec::new();
    write_request(&mut bytes, &Request::Step(actions.clone())).expect("writing a step");
    assert_eq!(
        read_request(&mut bytes.as_slice(), 1, 1_024).expect("reading it back"),
        Request::Step(actions)
    );
}

#[test]
fn a_reset_carries_a_presence_flag_so_seed_zero_stays_a_seed() {
    let mut seeded = Vec::new();
    write_request(&mut seeded, &Request::Reset(Some(0))).expect("writing a reset");
    assert_eq!(seeded, vec![0x02, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        read_request(&mut seeded.as_slice(), 2, 1_024).expect("reading it"),
        Request::Reset(Some(0)),
        "an explicit zero seed is not an absent seed"
    );

    let mut unseeded = Vec::new();
    write_request(&mut unseeded, &Request::Reset(None)).expect("writing a reset");
    assert_eq!(
        read_request(&mut unseeded.as_slice(), 2, 1_024).expect("reading it"),
        Request::Reset(None)
    );
}

#[test]
fn a_close_is_one_byte() {
    let mut bytes = Vec::new();
    write_request(&mut bytes, &Request::Close).expect("writing a close");
    assert_eq!(bytes, vec![0x03]);
    assert_eq!(
        read_request(&mut bytes.as_slice(), 2, 1_024).expect("reading it"),
        Request::Close
    );
}

#[test]
fn a_step_with_the_wrong_number_of_actions_steps_nothing() {
    let mut bytes = Vec::new();
    write_request(&mut bytes, &Request::Step(vec![Vec2::ZERO; 3])).expect("writing a step");
    match read_request(&mut bytes.as_slice(), 2, 1_024) {
        Err(ProtocolError::ActionCount {
            got: 3,
            expected: 2,
        }) => {}
        other => panic!("expected an action-count error, got {other:?}"),
    }
}

#[test]
fn a_direction_that_is_not_finite_is_refused() {
    // A NaN would reach the player's position and from there every value in
    // every observation, so it must not get past the reader.
    for bad in [
        Vec2::new(f32::NAN, 0.0),
        Vec2::new(0.0, f32::INFINITY),
        Vec2::splat(f32::NEG_INFINITY),
    ] {
        let mut bytes = Vec::new();
        write_request(&mut bytes, &Request::Step(vec![Vec2::ZERO, bad])).expect("writing a step");
        match read_request(&mut bytes.as_slice(), 2, 1_024) {
            Err(ProtocolError::InvalidAction { env: 1 }) => {}
            other => panic!("expected env 1 refused for {bad}, got {other:?}"),
        }
    }
}

#[test]
fn a_long_direction_is_carried_rather_than_refused() {
    // Only finiteness is the reader's business. Length does not set speed, and
    // clamping here would hide a client bug the arena would otherwise ignore.
    let actions = vec![Vec2::splat(1_000.0)];
    let mut bytes = Vec::new();
    write_request(&mut bytes, &Request::Step(actions.clone())).expect("writing a step");
    assert_eq!(
        read_request(&mut bytes.as_slice(), 1, 1_024).expect("reading it back"),
        Request::Step(actions)
    );
}

#[test]
fn an_unknown_opcode_is_named_rather_than_guessed() {
    match read_request(&mut [0x7f_u8].as_slice(), 1, 1_024) {
        Err(ProtocolError::UnknownOpcode(0x7f)) => {}
        other => panic!("expected an unknown opcode, got {other:?}"),
    }
}

#[test]
fn a_declared_length_beyond_the_maximum_is_refused_before_allocating() {
    // Opcode, then four billion envs. The count names envs and each costs
    // eight bytes, so what is refused is the eight-times-larger byte cost --
    // and the refusal quotes the cap that really exists, not a scaled one.
    let bytes = vec![0x01, 0xff, 0xff, 0xff, 0xff];
    match read_request(&mut bytes.as_slice(), 2, 64) {
        Err(ProtocolError::PayloadTooLarge { declared, maximum }) => {
            assert_eq!(declared, u64::from(u32::MAX) * 8);
            assert_eq!(maximum, 64);
        }
        other => panic!("expected a size refusal, got {other:?}"),
    }
}

#[test]
fn a_message_cut_in_half_is_truncation_not_a_short_batch() {
    let mut bytes = Vec::new();
    write_request(&mut bytes, &Request::Step(vec![Vec2::ZERO; 2])).expect("writing a step");
    bytes.truncate(bytes.len() - 1);
    assert!(matches!(
        read_request(&mut bytes.as_slice(), 2, 1_024),
        Err(ProtocolError::Truncated)
    ));
}

fn sample_batch() -> StepBatch {
    StepBatch {
        transitions: vec![
            EnvTransition {
                frame: 17,
                terminated: false,
                truncated: false,
                enemy_deaths: 2,
                episode_seed: 7,
                reset_seed: None,
            },
            EnvTransition {
                frame: 3_600,
                terminated: true,
                truncated: true,
                enemy_deaths: 0,
                episode_seed: 8,
                reset_seed: Some(0),
            },
        ],
        observations: vec![0.5, -0.25, 1.0, 0.0],
        terminal: vec![TerminalObservation {
            env: 1,
            observation: vec![0.125, 0.25],
        }],
    }
}

#[test]
fn a_step_response_round_trips_every_field() {
    let batch = sample_batch();
    let mut bytes = Vec::new();
    write_step(&mut bytes, &batch).expect("writing a batch");
    let read = read_step(&mut bytes.as_slice(), 1 << 20).expect("reading it back");
    assert_eq!(read, batch);

    let ended = read.transitions.get(1).expect("the second env");
    assert!(
        ended.done(),
        "death and timeout on the same frame both land"
    );
    assert_eq!(
        ended.reset_seed,
        Some(0),
        "a replacement episode's seed of zero survives the round trip"
    );
    assert_eq!(read.transitions.first().expect("the first env").frame, 17);
}

#[test]
fn a_reset_response_carries_seeds_and_observations_and_no_transition() {
    let batch = ResetBatch {
        seeds: vec![11, 12],
        observations: vec![0.0, 1.0, 2.0, 3.0],
    };
    let mut bytes = Vec::new();
    write_reset(&mut bytes, &batch).expect("writing a reset");
    assert_eq!(
        read_reset(&mut bytes.as_slice(), 1 << 20).expect("reading it back"),
        batch
    );
}

#[test]
fn a_response_read_as_the_wrong_kind_is_an_opcode_error() {
    let mut bytes = Vec::new();
    write_reset(
        &mut bytes,
        &ResetBatch {
            seeds: vec![1],
            observations: vec![0.0],
        },
    )
    .expect("writing a reset");
    assert!(matches!(
        read_step(&mut bytes.as_slice(), 1 << 20),
        Err(ProtocolError::UnknownOpcode(_))
    ));
}

#[test]
fn an_error_record_is_bounded() {
    let mut bytes = Vec::new();
    let long = "x".repeat(10_000);
    write_error(&mut bytes, 7, &long).expect("writing an error");
    assert!(
        bytes.len() < 2_048,
        "a diagnostic must not become a denial of service: {} bytes",
        bytes.len()
    );
}

#[test]
fn a_close_acknowledgement_is_its_own_opcode() {
    let mut bytes = Vec::new();
    write_closed(&mut bytes).expect("writing an ack");
    assert_eq!(bytes, vec![0x84]);
}

#[test]
fn episode_seeds_are_fixed_vectors_both_sides_can_check() {
    // Pinned so Python can preview a seed with the same published function.
    assert_eq!(episode_seed(0, 0, 0), 0x03c9_45fe_bf14_bb41);
    assert_eq!(episode_seed(42, 0, 0), 0xf180_f60c_3220_5505);
    assert_eq!(episode_seed(42, 1, 0), 0x9669_bf1b_feab_f0a8);
    assert_eq!(episode_seed(42, 0, 1), 0x3f14_9731_2d1e_30a8);
}

#[test]
fn an_envs_seed_stream_does_not_depend_on_other_envs() {
    let first: Vec<u64> = (0..4).map(|episode| episode_seed(5, 0, episode)).collect();
    let second: Vec<u64> = (0..4).map(|episode| episode_seed(5, 1, episode)).collect();
    assert_ne!(first, second, "two envs are two streams");
    assert_eq!(
        first,
        (0..4)
            .map(|episode| episode_seed(5, 0, episode))
            .collect::<Vec<_>>(),
        "and a stream is a pure function of its counters"
    );
}

#[test]
fn the_payload_bound_covers_a_full_batch_and_refuses_more() {
    let maximum = max_payload(8, OBSERVATION_VALUES);
    let one_batch = (8 * OBSERVATION_VALUES * 4) as u64;
    assert!(maximum > one_batch, "a real batch fits: {maximum}");
    assert!(
        maximum < one_batch * 4,
        "but the bound still bounds something: {maximum}"
    );
}

/// The exact bytes `sample_batch` must occupy on the wire.
///
/// Written out by hand rather than captured from the writer, so it is a
/// statement about the format and not a photograph of the implementation. A
/// codec that changes how many calls it makes must still produce this.
fn sample_batch_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(response::STEP);
    bytes.extend_from_slice(&[2, 0, 0, 0]); // two envs

    // Env 0: frame 17, running, two enemy deaths, episode seed 7, no reset.
    bytes.extend_from_slice(&[17, 0, 0, 0]);
    bytes.extend_from_slice(&[0, 0]); // terminated, truncated
    bytes.extend_from_slice(&[2, 0, 0, 0]);
    bytes.extend_from_slice(&[7, 0, 0, 0, 0, 0, 0, 0]);
    bytes.extend_from_slice(&[0]); // no replacement seed
    bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // written even so

    // Env 1: frame 3600, died as the budget ran out, replaced on seed zero.
    bytes.extend_from_slice(&[0x10, 0x0E, 0, 0]);
    bytes.extend_from_slice(&[1, 1]);
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes.extend_from_slice(&[8, 0, 0, 0, 0, 0, 0, 0]);
    bytes.extend_from_slice(&[1]); // present, and the value is zero
    bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);

    // Observations: 0.5, -0.25, 1.0, 0.0.
    bytes.extend_from_slice(&[4, 0, 0, 0]);
    bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x3F]);
    bytes.extend_from_slice(&[0x00, 0x00, 0x80, 0xBE]);
    bytes.extend_from_slice(&[0x00, 0x00, 0x80, 0x3F]);
    bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    // One terminal observation, for env 1: 0.125, 0.25.
    bytes.extend_from_slice(&[1, 0, 0, 0]);
    bytes.extend_from_slice(&[1, 0, 0, 0]);
    bytes.extend_from_slice(&[2, 0, 0, 0]);
    bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x3E]);
    bytes.extend_from_slice(&[0x00, 0x00, 0x80, 0x3E]);
    bytes
}

#[test]
fn an_auto_reset_step_response_is_bytes_we_can_write_out_by_hand() {
    let mut written = Vec::new();
    write_step(&mut written, &sample_batch()).expect("writing a batch");
    assert_eq!(
        written,
        sample_batch_bytes(),
        "the wire format is frozen: little endian, length prefixed, in this order"
    );
}

#[test]
fn a_reset_response_is_bytes_we_can_write_out_by_hand() {
    let mut written = Vec::new();
    write_reset(
        &mut written,
        &ResetBatch {
            seeds: vec![11, 12],
            observations: vec![0.5, -0.25],
        },
    )
    .expect("writing a reset");

    let mut expected = Vec::new();
    expected.push(response::RESET);
    expected.extend_from_slice(&[2, 0, 0, 0]);
    expected.extend_from_slice(&[11, 0, 0, 0, 0, 0, 0, 0]);
    expected.extend_from_slice(&[12, 0, 0, 0, 0, 0, 0, 0]);
    expected.extend_from_slice(&[2, 0, 0, 0]);
    expected.extend_from_slice(&[0x00, 0x00, 0x00, 0x3F]);
    expected.extend_from_slice(&[0x00, 0x00, 0x80, 0xBE]);
    assert_eq!(written, expected);
}

#[test]
fn float_arrays_survive_any_length_a_chunked_codec_might_split() {
    // A codec that moves floats in blocks has a seam. These lengths sit on
    // both sides of every power-of-two block size it might pick, so a batch
    // that ends mid-block is covered whatever the block turns out to be.
    for count in [
        0_usize, 1, 2, 255, 256, 257, 511, 512, 1_023, 1_024, 1_025, 4_097,
    ] {
        let observations: Vec<f32> = (0..count)
            .map(|index| {
                // Values that are not round: a byte swap or a dropped block
                // has to show up as a different number, not a similar one.
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "The index is small and only needs to be distinct"
                )]
                let scaled = index as f32;
                scaled.mul_add(0.001_25, -3.5)
            })
            .collect();
        let batch = ResetBatch {
            seeds: vec![1],
            observations,
        };

        let mut bytes = Vec::new();
        write_reset(&mut bytes, &batch).expect("writing a reset");
        assert_eq!(
            bytes.len(),
            1 + 4 + 8 + 4 + count * 4,
            "a float array is a count and that many four-byte values: {count}"
        );
        assert_eq!(
            read_reset(&mut bytes.as_slice(), 1 << 20).expect("reading it back"),
            batch,
            "every value comes back bit for bit at length {count}"
        );
    }
}
