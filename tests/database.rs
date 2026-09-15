//! Opt-in persistence check against the local development `PostgreSQL` instance.

#![cfg(not(target_arch = "wasm32"))]

use dodge_royale::{database, model::PlayerScore};

#[tokio::test]
#[ignore = "requires PostgreSQL and DATABASE_URL; run with --test database -- --ignored"]
async fn scores_round_trip_and_reject_negative_values() {
    let database_url =
        std::env::var("DATABASE_URL").expect("explicit database test requires DATABASE_URL");
    let pool = database::connect(&database_url)
        .await
        .expect("test PostgreSQL connection should succeed");
    database::migrate(&pool)
        .await
        .expect("test database migrations should succeed");

    let score = PlayerScore {
        player_name: "O'Reilly \"一\"; -- integration test".to_owned(),
        score: 120,
    };
    let id = database::save_score(&pool, &score)
        .await
        .expect("valid test score should save");
    let stored = sqlx::query_as::<_, (String, i64)>(
        "SELECT player_name, score FROM player_scores WHERE id = $1",
    )
    .bind(id)
    .fetch_one(&pool)
    .await;

    let invalid_score = PlayerScore {
        player_name: score.player_name.clone(),
        score: -1,
    };
    let invalid_result = database::save_score(&pool, &invalid_score).await;

    // Clean up before asserting, including an unexpected successful invalid
    // insert, so a failed check does not leave test scores in the database.
    let cleanup = sqlx::query("DELETE FROM player_scores WHERE id = $1 OR id = $2")
        .bind(id)
        .bind(invalid_result.as_ref().ok().copied())
        .execute(&pool)
        .await;
    pool.close().await;
    cleanup.expect("test rows should be removed before assertions");

    assert_eq!(
        stored.expect("saved test score should be readable"),
        (score.player_name, score.score)
    );
    assert!(matches!(
        invalid_result,
        Err(sqlx::Error::Database(error)) if error.is_check_violation()
    ));
}
