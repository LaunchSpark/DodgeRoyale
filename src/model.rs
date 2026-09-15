//! Serializable data passed between gameplay, background work, and storage.

use serde::{Deserialize, Serialize};

/// A small, owned snapshot that can leave Bevy's world for background processing.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RoundResult {
    /// The player represented by this snapshot.
    pub player_name: String,
    /// The number of successful dodges in the round.
    pub dodges: u32,
}

/// A computed score ready to serialize or persist.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlayerScore {
    /// The player who earned the score.
    pub player_name: String,
    /// A nonnegative score; `PostgreSQL` stores this as a `BIGINT`.
    pub score: i64,
}

#[cfg(test)]
mod tests {
    use super::{PlayerScore, RoundResult};

    #[test]
    fn json_preserves_names_and_large_scores() {
        let score = PlayerScore {
            player_name: "Player \"一\"".to_owned(),
            score: i64::from(u32::MAX).saturating_mul(10),
        };
        let json = serde_json::to_string(&score).expect("score should serialize to JSON");
        let decoded: PlayerScore =
            serde_json::from_str(&json).expect("serialized score should deserialize");

        assert_eq!(decoded, score);
        assert!(
            serde_json::from_str::<RoundResult>(r#"{"player_name":"Player","dodges":-1}"#).is_err()
        );
    }
}
