//! A finite startup demo; frame systems never wait for I/O or parallel batches.

use color_eyre::eyre::{Result, WrapErr};
use dodge_royale::{compute, database, model::RoundResult};

pub async fn initialize(database_url: Option<&str>) -> Result<String> {
    // These independent tasks make progress concurrently. CPU work runs on a
    // blocking bridge into Rayon so it cannot occupy a Tokio async worker.
    let (json, ()) = tokio::try_join!(score_demo(), prepare_database(database_url))?;
    Ok(json)
}

async fn score_demo() -> Result<String> {
    tokio::task::spawn_blocking(|| {
        let pool = compute::build_pool()?;
        let results: Vec<RoundResult> = serde_json::from_str(
            r#"[{"player_name":"Player One","dodges":12},{"player_name":"Player Two","dodges":7}]"#,
        )?;
        let scores = compute::score_round(&pool, &results);
        Ok(serde_json::to_string(&scores)?)
    })
    .await
    .wrap_err("The parallel scoring task failed")?
}

async fn prepare_database(database_url: Option<&str>) -> Result<()> {
    if let Some(url) = database_url {
        let pool = database::connect(url).await.wrap_err(
            "Could not connect to PostgreSQL; check DATABASE_URL and docker compose up -d db",
        )?;
        let migration = database::migrate(&pool).await;
        pool.close().await;
        migration.wrap_err("Could not apply database migrations")?;
    }
    Ok(())
}
