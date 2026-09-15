//! CPU work on owned snapshots, using a separate bounded Rayon pool.

use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder, prelude::*};

use crate::model::{PlayerScore, RoundResult};

/// Build a reusable two-worker pool without changing Rayon's global pool.
///
/// Create this once at startup. Bevy and Tokio have their own worker pools, so
/// keeping the CPU pool small avoids creating a full CPU-sized pool for each.
///
/// # Errors
///
/// Returns an error if Rayon cannot start its worker threads.
pub fn build_pool() -> Result<ThreadPool, ThreadPoolBuildError> {
    ThreadPoolBuilder::new()
        .num_threads(2)
        .thread_name(|index| format!("dodge-cpu-{index}"))
        .build()
}

/// Compute ten points per dodge in parallel, preserving the input order.
///
/// This synchronous function belongs on a background worker for substantial
/// batches: calling it from a Bevy system or Tokio async task still blocks that
/// caller until the batch finishes.
#[must_use]
pub fn score_round(pool: &ThreadPool, results: &[RoundResult]) -> Vec<PlayerScore> {
    pool.install(|| {
        results
            .par_iter()
            .map(|result| PlayerScore {
                player_name: result.player_name.clone(),
                score: i64::from(result.dodges).saturating_mul(10),
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::{build_pool, score_round};
    use crate::model::{PlayerScore, RoundResult};

    #[test]
    fn scoring_preserves_order_and_handles_boundaries() {
        let pool = build_pool().expect("test CPU pool should start");
        let results = [
            RoundResult {
                player_name: "Zero".to_owned(),
                dodges: 0,
            },
            RoundResult {
                player_name: "Maximum".to_owned(),
                dodges: u32::MAX,
            },
            RoundResult {
                player_name: "One".to_owned(),
                dodges: 1,
            },
        ];
        let expected = vec![
            PlayerScore {
                player_name: "Zero".to_owned(),
                score: 0,
            },
            PlayerScore {
                player_name: "Maximum".to_owned(),
                score: 42_949_672_950,
            },
            PlayerScore {
                player_name: "One".to_owned(),
                score: 10,
            },
        ];

        assert_eq!(pool.current_num_threads(), 2);
        assert_eq!(score_round(&pool, &results), expected);
        assert!(score_round(&pool, &[]).is_empty());
    }
}
