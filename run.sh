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
if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
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
    restore_graphics
    ;;
web)
    bevy run --locked web --open
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
  fmt           format the workspace
  web           browser build, needs the Bevy CLI
USAGE
    exit 1
    ;;
esac
