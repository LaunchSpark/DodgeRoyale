#!/usr/bin/env bash
# Task runner for DodgeRoyale.
#
# Headless commands (--no-default-features) write to the SAME path as the
# graphics build, target/debug/dodge-royale, and silently replace the playable
# binary with a windowless one. Every headless task here restores the graphics
# build afterwards so the next launch always opens a window.
set -euo pipefail
cd "$(dirname "$0")"

# rustup edits the saved PATH, but shells opened before the install miss it.
# The web task also needs Bevy CLI from the same directory.
if { ! command -v cargo >/dev/null 2>&1 || ! command -v bevy >/dev/null 2>&1; } \
    && [ -x "$HOME/.cargo/bin/cargo" ]; then
    PATH="$HOME/.cargo/bin:$PATH"
fi

BIN=target/debug/dodge-royale
# Windows appends .exe; nm does not resolve the bare name the way exec does.
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*) BIN="$BIN.exe" ;;
esac

restore_graphics() {
    echo "--> restoring graphics build"
    cargo build --locked
}

# A failing check must not leave a headless binary in place, so restore on any
# exit path once a headless task has run.
restore_on_exit() {
    trap 'restore_graphics >/dev/null 2>&1 || true' EXIT
}

# Fail loudly rather than opening no window, if the binary has no renderer.
# grep -c reads all input; grep -q would exit early, kill nm with SIGPIPE, and
# trip pipefail, making a healthy binary look headless.
assert_graphics() {
    local symbols
    # --defined-only, not -U: GNU nm reads -U as --unicode and eats the path.
    symbols=$(nm --defined-only "$BIN" 2>/dev/null | grep -c bevy_render || true)
    if [ "${symbols:-0}" -eq 0 ]; then
        echo "ERROR: $BIN has no renderer (headless build). Run: ./run.sh build" >&2
        exit 1
    fi
}

dashboard_python=""
dashboard_pid=""

dashboard_ready() {
    "$dashboard_python" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:2718/health", timeout=0.5).close()' \
        >/dev/null 2>&1
}

stop_dashboard() {
    if [ -n "$dashboard_pid" ]; then
        kill "$dashboard_pid" 2>/dev/null || true
        wait "$dashboard_pid" 2>/dev/null || true
    fi
}

start_dashboard() {
    if [ -x "training/.venv/Scripts/python.exe" ]; then
        dashboard_python="$PWD/training/.venv/Scripts/python.exe"
    elif [ -x "training/.venv/bin/python" ]; then
        dashboard_python="$PWD/training/.venv/bin/python"
    else
        echo "ERROR: install the dashboard first: cd training && uv sync --extra dashboard --extra cpu (or cu126)" >&2
        return 1
    fi

    if dashboard_ready; then
        echo "--> using the dashboard already running at http://127.0.0.1:2718/"
        return 0
    fi

    echo "--> starting marimo at http://127.0.0.1:2718/"
    (
        cd training
        "$dashboard_python" -m marimo run dodge_royale/dashboard.py \
            --no-sandbox --headless --host 127.0.0.1 --port 2718 --no-token
    ) &
    dashboard_pid=$!
    trap stop_dashboard EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    local attempt
    for ((attempt = 0; attempt < 100; attempt++)); do
        if dashboard_ready; then
            return 0
        fi
        if ! kill -0 "$dashboard_pid" 2>/dev/null; then
            echo "ERROR: marimo exited before it became ready" >&2
            wait "$dashboard_pid" || true
            return 1
        fi
        sleep 0.2
    done
    echo "ERROR: marimo did not become ready at http://127.0.0.1:2718/" >&2
    return 1
}

run_web_with_dashboard() {
    start_dashboard
    "$@"
}

case "${1:-buildAndRun}" in
buildAndRun | run)
    cargo build --locked
    assert_graphics
    echo "--> launching"
    exec "$BIN" "${@:2}"
    ;;
build)
    cargo build --locked
    ;;
test)
    restore_on_exit
    cargo test --locked --no-default-features
    restore_graphics
    ;;
check)
    restore_on_exit
    echo "--> fmt"
    cargo fmt --all -- --check
    echo "--> clippy (graphics)"
    cargo clippy --locked --all-targets --all-features -- -D warnings
    echo "--> clippy (headless)"
    cargo clippy --locked --all-targets --no-default-features -- -D warnings
    echo "--> tests"
    cargo test --locked --no-default-features
    echo "--> smoke"
    cargo run --locked --no-default-features -- --smoke-test
    restore_graphics
    echo "ALL CHECKS PASSED"
    ;;
smoke)
    restore_on_exit
    cargo run --locked --no-default-features -- --smoke-test
    restore_graphics
    ;;
headless)
    cargo run --locked --no-default-features
    restore_graphics
    ;;
fmt)
    cargo fmt --all
    ;;
gym)
    # Serves the trainer on stdout, so diagnostics stay on stderr. Arguments
    # after the task name are passed through, e.g. ./run.sh gym --envs 4.
    restore_on_exit
    cargo run --locked --no-default-features -- gym "${@:2}"
    # Stdout belongs to the protocol, so this task's own chatter goes to stderr:
    # a client reading the pipe would otherwise find "--> restoring graphics
    # build" appended to the byte stream.
    restore_graphics >&2
    ;;
bench)
    # The bench profile builds into target/release, so this never replaces the
    # playable target/debug binary and needs no restore. Optimised on purpose:
    # debug numbers would say the simulation is the bottleneck when it is not.
    # Arguments after the task name are env counts, e.g. ./run.sh bench 4.
    cargo bench --locked --no-default-features --bench gym_throughput -- "${@:2}"
    ;;
web)
    run_web_with_dashboard bevy run --locked web --open
    ;;
web-docker)
    run_web_with_dashboard docker compose up --build web
    ;;
*)
    cat <<'USAGE'
usage: ./run.sh [command]

  buildAndRun   build the graphics binary and launch it (default)
  build         build only
  run           alias for buildAndRun
  test          headless tests, then restore the graphics build
  check         full gate: fmt, both clippy configs, tests, smoke
  smoke         one headless frame, then restore the graphics build
  headless      one idle-player episode, then report how it ended
  gym           serve headless arenas to a trainer over stdin/stdout
  bench         time simulation, encoding and pipe transfer per env step
  fmt           format the workspace
  web           browser game and marimo dashboard, needs Bevy CLI and training env
  web-docker    Docker web game with the same host dashboard
USAGE
    exit 1
    ;;
esac
