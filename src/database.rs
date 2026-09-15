//! Optional `PostgreSQL` persistence; invoke these async functions on Tokio.

use std::time::Duration;

use sqlx::{PgPool, migrate::MigrateError, postgres::PgPoolOptions};

use crate::model::PlayerScore;

/// Connect using a small pool and a finite connection acquisition timeout.
///
/// # Errors
///
/// Returns an error if the URL is invalid, `PostgreSQL` cannot be reached, or
/// authentication or connection acquisition fails.
pub async fn connect(database_url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(5))
        .connect(database_url)
        .await
}

/// Apply the migrations embedded in the executable.
///
/// # Errors
///
/// Returns an error if the database cannot apply or validate the migrations.
pub async fn migrate(pool: &PgPool) -> Result<(), MigrateError> {
    sqlx::migrate!("./migrations").run(pool).await
}

/// Save a score with bound parameters and return its generated database ID.
///
/// This uses a runtime query, so building the project never requires a live
/// database or a pre-generated `SQLx` query cache.
///
/// # Errors
///
/// Returns an error if the connection, query, or database constraints fail.
pub async fn save_score(pool: &PgPool, score: &PlayerScore) -> Result<i64, sqlx::Error> {
    sqlx::query_scalar(
        "INSERT INTO player_scores (player_name, score) VALUES ($1, $2) RETURNING id",
    )
    .bind(&score.player_name)
    .bind(score.score)
    .fetch_one(pool)
    .await
}
