# /// script
# requires-python = ">=3.11"
# dependencies = ["dodge-royale-training[dashboard]"]
#
# [tool.uv.sources]
# dodge-royale-training = { path = "../", editable = true }
# ///
"""The training dashboard, as a marimo notebook.

    uv run marimo edit dodge_royale/dashboard.py     # change it
    uv run marimo run dodge_royale/dashboard.py      # use it
    uv run dodge_royale/dashboard.py                 # script mode: a short real run

marimo re-runs a cell whenever anything it reads changes, which is what keeps
the page live without a refresh loop. It is also the hazard: a cell that
*created* a training run would create another on every rerun, each one
launching its own gym. So no cell here owns a run. The worker lives in
`worker.py` behind a module-level singleton, and these cells only read
snapshots of it and send it commands.

Script mode runs a small real session end to end and prints the metrics, which
is the smoke test for everything below.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="DodgeRoyale training")


@app.cell
def _():
    import marimo as mo

    from dodge_royale.metrics import METRICS
    from dodge_royale.training import DEFAULTS, SessionConfig, describe
    from dodge_royale.worker import WorkerState, active_worker, start_worker

    return (
        DEFAULTS,
        SessionConfig,
        WorkerState,
        active_worker,
        describe,
        mo,
        start_worker,
    )


@app.cell
def _(mo):
    script_mode = mo.app_meta().mode == "script"
    return (script_mode,)


@app.cell
def _(mo):
    mo.md(r"""
    # DodgeRoyale training

    A PPO run against the Rust gym. The run itself lives on a background thread
    so this page stays responsive; everything below is a snapshot of it.
    """)
    return


@app.cell
def _(mo):
    # The page's clock. Each tick re-runs the cells that read it, which is how
    # the numbers update without anything polling in a loop.
    # Every option has to be a real interval. "off" is not a duration, and an
    # invalid option leaves the element without a working clock: the page then
    # only updates when a button is pressed, which looks exactly like a run
    # that never leaves "starting".
    refresh = mo.ui.refresh(
        options=["1s", "2s", "5s", "10s"], default_interval="1s", label="Refresh"
    )
    refresh
    return (refresh,)


@app.cell
def _(DEFAULTS, mo):
    # Royale only: there is no game selector, because there is one game and a
    # selector with one setting is a second code path to keep correct.
    envs = mo.ui.slider(1, 64, value=DEFAULTS.envs, label="Environments", show_value=True)
    enemies = mo.ui.slider(0, 200, value=DEFAULTS.enemies, label="Enemies", show_value=True)
    max_frames = mo.ui.number(60, 60_000, value=DEFAULTS.max_frames, label="Frame budget")
    hold_frames = mo.ui.slider(
        1, 108, value=DEFAULTS.hold_frames, label="Prediction hold (frames)", show_value=True
    )
    threads = mo.ui.slider(1, 16, value=DEFAULTS.threads, label="Gym threads", show_value=True)
    seed = mo.ui.number(0, 2**31 - 1, value=DEFAULTS.seed, label="Root seed")
    return enemies, envs, hold_frames, max_frames, seed, threads


@app.cell
def _(DEFAULTS, mo):
    n_steps = mo.ui.number(8, 8192, value=DEFAULTS.n_steps, label="Rollout steps")
    minibatch_cap = mo.ui.number(8, 4096, value=DEFAULTS.minibatch_cap, label="Minibatch cap")
    total_timesteps = mo.ui.number(
        1_000, 100_000_000, value=DEFAULTS.total_timesteps, label="Total steps"
    )
    device = mo.ui.dropdown(["auto", "cpu", "cuda"], value=DEFAULTS.device, label="Device")
    run_name = mo.ui.text(value=DEFAULTS.run_name, label="Run name")
    resume_path = mo.ui.text(
        value="", label="Resume from checkpoint", placeholder="empty starts fresh"
    )
    return (
        device,
        minibatch_cap,
        n_steps,
        resume_path,
        run_name,
        total_timesteps,
    )


@app.cell
def _(
    device,
    enemies,
    envs,
    hold_frames,
    max_frames,
    minibatch_cap,
    mo,
    n_steps,
    resume_path,
    run_name,
    seed,
    threads,
    total_timesteps,
):
    mo.accordion(
        {
            "Game configuration": mo.vstack(
                [envs, enemies, max_frames, hold_frames, threads, seed]
            ),
            "Learner": mo.vstack(
                [n_steps, minibatch_cap, total_timesteps, device, run_name, resume_path]
            ),
        }
    )
    return


@app.cell
def _(
    SessionConfig,
    device,
    enemies,
    envs,
    hold_frames,
    max_frames,
    minibatch_cap,
    n_steps,
    run_name,
    script_mode,
    seed,
    threads,
    total_timesteps,
):
    # One config, built from the form and read by everything below, so the
    # summary and the run cannot describe different things. Script mode keeps
    # every widget but shrinks the run to something that finishes.
    staged = SessionConfig(
        envs=2 if script_mode else int(envs.value),
        seed=int(seed.value),
        enemies=8 if script_mode else int(enemies.value),
        # Short enough that episodes actually finish, so the smoke run
        # exercises the episode metrics and not only the optimizer ones.
        max_frames=16 if script_mode else int(max_frames.value),
        hold_frames=int(hold_frames.value),
        threads=1 if script_mode else int(threads.value),
        n_steps=16 if script_mode else int(n_steps.value),
        minibatch_cap=int(minibatch_cap.value),
        total_timesteps=128 if script_mode else int(total_timesteps.value),
        device="cpu" if script_mode else device.value,
        run_name=run_name.value or "royale",
    )
    return (staged,)


@app.cell
def _(describe, mo, staged):
    # A specific, expected failure with a meaningful recovery: an impossible
    # configuration should be shown as a message rather than a traceback, and
    # must block Start.
    try:
        summary, problem = describe(staged), None
    except ValueError as invalid:
        summary, problem = "", str(invalid)

    mo.md(f"**Configuration is invalid:** {problem}") if problem else mo.md(
        f"```text\n{summary}\n```"
    )
    return (problem,)


@app.cell
def _(mo):
    start_button = mo.ui.run_button(label="Start")
    pause_button = mo.ui.run_button(label="Pause / resume")
    save_button = mo.ui.run_button(label="Save checkpoint")
    stop_button = mo.ui.run_button(label="Stop")
    mo.hstack([start_button, pause_button, save_button, stop_button], justify="start")
    return pause_button, save_button, start_button, stop_button


@app.cell
def _(
    WorkerState,
    active_worker,
    mo,
    pause_button,
    problem,
    resume_path,
    save_button,
    script_mode,
    staged,
    start_button,
    start_worker,
    stop_button,
):
    # Every control acts on the single worker rather than on anything this
    # cell owns. A rerun re-reads it; it never creates one. `start_worker`
    # refuses while a run is active, so a double click cannot make two gyms.
    running = active_worker()
    action = ""

    if script_mode:
        start_worker(staged)
        action = "script mode: started a short run"
    elif start_button.value and problem:
        action = f"refused: {problem}"
    elif start_button.value and running is not None and running.state.is_active:
        # The worker outlives a browser session on purpose: reloading the page
        # should find the run still going rather than orphan it. That makes
        # "already running" an ordinary thing to say, not an error to raise,
        # and asking rather than catching keeps it off the exception path.
        action = f"a run is already {running.state.value}; stop it first"
    elif start_button.value:
        start_worker(staged, resume=resume_path.value.strip() or None)
        action = "started"
    elif pause_button.value and running is None:
        action = "nothing is running"
    elif pause_button.value and running.state is WorkerState.PAUSED:
        running.resume_training()
        action = "resumed"
    elif pause_button.value:
        running.pause()
        action = "pausing"
    elif save_button.value and running is None:
        action = "nothing is running"
    elif save_button.value:
        running.request_save()
        action = "a checkpoint was requested"
    elif stop_button.value and running is None:
        action = "nothing is running"
    elif stop_button.value:
        running.stop()
        action = "stopped"

    mo.md(f"*{action}*")
    return (action,)


@app.cell
def _(action, active_worker, mo, refresh, script_mode):
    # `refresh.value`, not `refresh`. marimo's reactivity is over variable
    # assignments, so depending on the element itself re-runs this cell only
    # when the cell that *created* it re-runs -- never on a tick. Reading the
    # value is what subscribes to the clock, and without it the page freezes on
    # whatever state it happened to see when a button was last pressed.
    refresh.value, action

    # In script mode there is no clock, so wait for the run rather than
    # snapshotting an empty one.
    worker = active_worker()
    if script_mode and worker is not None:
        worker.wait(300)

    status = worker.status() if worker is not None else None
    mo.md(f"### {status.state.value if status else 'idle'}")
    return (status,)


@app.cell
def _(mo, status):
    # `status` is None only before the first run, which is a value to render
    # rather than a reason to skip the cell.
    groups: dict[str, list[str]] = {}
    for metric, shown in (status.metrics.rows() if status else []):
        groups.setdefault(metric.group, []).append(
            f"| {metric.label} | {shown} | {metric.description} |"
        )
    blocks = [
        f"**{name.title()}**\n\n| Metric | Value | What it is |\n|---|---|---|\n"
        + "\n".join(groups[name])
        for name in ("episode", "throughput", "optimizer", "run")
        if name in groups
    ]
    mo.md(
        "\n\n".join(blocks)
        if blocks
        else "*No run yet. Set the configuration above and press Start.*"
    )
    return


@app.cell
def _(mo, status):
    import altair as alt
    import polars as pl

    # Charts come from the same bounded history the worker publishes, so the
    # page never holds more points than the worker kept.
    points = pl.DataFrame(
        {
            "steps": [snap.timesteps for snap in (status.history if status else [])],
            "survival": [
                snap.survival_seconds for snap in (status.history if status else [])
            ],
            "return": [
                snap.episode_return for snap in (status.history if status else [])
            ],
        },
        schema={"steps": pl.Int64, "survival": pl.Float64, "return": pl.Float64},
    ).drop_nulls()

    survival_chart = (
        alt.Chart(points)
        .mark_line()
        .encode(x="steps:Q", y=alt.Y("survival:Q", title="survival (s)"))
        .properties(height=180, title="Survival")
    )
    return_chart = (
        alt.Chart(points)
        .mark_line(color="#d97706")
        .encode(x="steps:Q", y=alt.Y("return:Q", title="return"))
        .properties(height=180, title="Episode return")
    )
    mo.ui.altair_chart(survival_chart & return_chart) if len(points) > 1 else mo.md(
        "*Charts appear once the run has finished a few episodes.*"
    )
    return


@app.cell
def _(mo, status):
    failure = status.error if status else None
    saved = list(status.checkpoints[-5:]) if status else []
    mo.md(
        f"**The run failed.**\n\n```text\n{failure}\n```"
        if failure
        else ("**Checkpoints**\n\n" + "\n".join(f"- `{p}`" for p in saved) if saved else "")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Watch Agent is not available.** Watching the policy play needs the trained
    weights running inside the game: exporting them, and a forward pass in Rust.
    That is the in-game autopilot, deliberately out of scope for this spec and
    written after a policy trains. Until then the numbers above are how a run is
    judged.
    """)
    return


@app.cell
def _(script_mode, status):
    # Script mode renders nothing, so the smoke test has to say its piece on
    # stdout. This is the one place the two modes genuinely differ in output.
    if script_mode:
        print(f"state: {status.state.value}")
        if status.error:
            print(f"error: {status.error}")
        for definition, reading in status.metrics.rows():
            print(f"  {definition.label:<14} {reading}")
        for written in status.checkpoints:
            print(f"  checkpoint     {written}")
    return


if __name__ == "__main__":
    app.run()
