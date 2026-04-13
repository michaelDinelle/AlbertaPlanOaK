#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/carla_whisky_env.sh"

OAK_BINARY="${OAK_BINARY:-$ROOT_DIR/.build/Release/OaK}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/miniforge3/envs/oak}"
PYTHON_VERSION="${OAK_PYTHON_VERSION:-3.11}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
PYTHON_SITE_PACKAGES="$CONDA_ENV_DIR/lib/python$PYTHON_VERSION/site-packages"
PYTHON_LIB="$CONDA_ENV_DIR/lib/libpython$PYTHON_VERSION.dylib"
REQUESTED_ENV="${OAK_ENV:-${1:-all}}"
# EnvPool: OAK_ENVPOOL_TASK_ID=<one gym id> or omit for suite; OAK_ENVPOOL_SUITE_PREFIX overrides default (MiniGrid-) for logical name minigrid.
REQUESTED_POLICY="$(printf '%s' "${OAK_POLICY:-oak}" | tr '[:upper:]' '[:lower:]')"
CARLA_RENDER_MODE="${OAK_CARLA_RENDER_MODE:-offscreen}"
CARLA_MODE_FILE="${OAK_CARLA_MODE_FILE:-$ROOT_DIR/.oak-carla-mode}"
CARLA_STARTED_SERVER=0
CARLA_PREEXISTING_PIDS=""
CARLA_STARTED_PIDS=""

die() {
    echo "$*" >&2
    exit 1
}

validate_release_binary() {
    [ -x "$OAK_BINARY" ] || die "Missing release binary at $OAK_BINARY. Run 'xcodebuild -scheme OaK -configuration Release -destination 'platform=macOS' SYMROOT=\"\$(pwd)/.build\"' or './doctor.sh metal'."
}

requires_python_prefix() {
    case "$REQUESTED_ENV" in
        carla)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

python_import_snippet() {
    case "$REQUESTED_ENV" in
        pyflyt)
            printf '%s\n' 'import gymnasium, numpy, PyFlyt, PyFlyt.gym_envs'
            ;;
        minigrid|cheetah|walker|humanoid)
            printf '%s\n' 'import numpy, envpool'
            ;;
        all)
            printf '%s\n' 'import gymnasium, numpy, PyFlyt, PyFlyt.gym_envs, envpool'
            ;;
        *)
            printf '%s\n' ''
            ;;
    esac
}

validate_python_prefix() {
    [ -d "$CONDA_ENV_DIR" ] || die "Missing conda environment at $CONDA_ENV_DIR. Run './doctor.sh python' or './setup.sh'."
    [ -f "$CONDA_ENV_DIR/conda-meta/history" ] || die "Invalid conda environment at $CONDA_ENV_DIR: missing conda-meta/history. Run './doctor.sh python' or './setup.sh'."
    [ -x "$PYTHON_BIN" ] || die "Missing Python executable at $PYTHON_BIN. Run './doctor.sh python' or './setup.sh'."
    [ -f "$PYTHON_LIB" ] || die "Missing Python library at $PYTHON_LIB. Run './doctor.sh python' or './setup.sh'."
}

export_python_runtime() {
    export PYTHONHOME="$CONDA_ENV_DIR"
    export PYTHONPATH="$PYTHON_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHON_LIBRARY="$PYTHON_LIB"
}

validate_python_imports() {
    local snippet
    snippet="$(python_import_snippet)"
    [ -n "$snippet" ] || return 0

    if ! "$PYTHON_BIN" -c "$snippet" >/dev/null 2>&1; then
        die "Python preflight failed for '$REQUESTED_ENV'. Run './doctor.sh python' for details."
    fi
}

should_preflight_metal() {
    [ "$REQUESTED_POLICY" = "random" ] && return 1
    return 0
}

validate_metal_runtime() {
    if ! "$OAK_BINARY" doctor-metal --quiet >/dev/null 2>&1; then
        die "Metal preflight failed for this runtime. Run './doctor.sh metal' for details."
    fi
}

carla_server_listening() {
    lsof -nP -iTCP:"$OAK_CARLA_PORT" -sTCP:LISTEN >/dev/null 2>&1
}

carla_process_ids() {
    pgrep -f 'CARLA_0\.9\.15|CarlaUE4\.exe|CarlaUE4-Win64-Shipping\.exe|Kegworks' 2>/dev/null || true
}

wait_for_carla_server() {
    local attempts="${OAK_CARLA_START_TIMEOUT:-60}"

    while [ "$attempts" -gt 0 ]; do
        if carla_server_listening; then
            return 0
        fi
        sleep 1
        attempts=$((attempts - 1))
    done

    return 1
}

start_carla_server() {
    local quality="${OAK_CARLA_QUALITY:-Low}"
    local log_file="${OAK_CARLA_LOG:-$ROOT_DIR/.oak-carla.log}"
    local -a args

    [ -x "$OAK_WHISKY_CMD" ] || {
        echo "Missing Whisky CLI at $OAK_WHISKY_CMD" >&2
        return 1
    }
    [ -n "${OAK_WHISKY_BOTTLE:-}" ] || {
        echo "Missing Whisky bottle name" >&2
        return 1
    }
    [ -n "${OAK_CARLA_WIN_EXE:-}" ] || {
        echo "Missing CARLA Windows executable path" >&2
        return 1
    }

    args=("$OAK_WHISKY_BOTTLE" "$OAK_CARLA_WIN_EXE" "-quality-level=$quality" "-carla-rpc-port=$OAK_CARLA_PORT")
    if [ "$CARLA_RENDER_MODE" = "offscreen" ]; then
        args+=("-RenderOffScreen")
    fi

    nohup "$OAK_WHISKY_CMD" run "${args[@]}" >"$log_file" 2>&1 &
}

validate_carla_autostart_policy() {
    if [ "${OAK_CARLA_AUTOSTART:-1}" != "1" ]; then
        return 0
    fi

    if [ "$CARLA_RENDER_MODE" != "offscreen" ]; then
        return 0
    fi

    if [ "${OAK_CARLA_ALLOW_VISIBLE_AUTOSTART:-0}" = "1" ]; then
        return 0
    fi

    cat >&2 <<EOF
Whisky CARLA autostart is not truly headless. The Windows Unreal process still launches under Whisky even with -RenderOffScreen.
Refusing to autostart by default.

Start a CARLA server yourself and rerun, or explicitly opt into a visible autostart with:
  OAK_CARLA_ALLOW_VISIBLE_AUTOSTART=1 ./run.sh carla
EOF
    return 1
}

record_carla_launch_mode() {
    printf '%s\n' "$CARLA_RENDER_MODE" >"$CARLA_MODE_FILE"
}

capture_started_carla_pids() {
    local current_pids pid existing_pid seen
    current_pids="$(carla_process_ids | tr '\n' ' ')"
    CARLA_STARTED_PIDS=""

    for pid in $current_pids; do
        seen=0
        for existing_pid in $CARLA_PREEXISTING_PIDS; do
            if [ "$pid" = "$existing_pid" ]; then
                seen=1
                break
            fi
        done
        if [ "$seen" -eq 0 ]; then
            CARLA_STARTED_PIDS="${CARLA_STARTED_PIDS}${CARLA_STARTED_PIDS:+ }$pid"
        fi
    done
}

shutdown_started_carla_server() {
    local pid lingering_pids

    if [ "$CARLA_STARTED_SERVER" != "1" ]; then
        return 0
    fi

    if [ "${OAK_CARLA_KEEP_ALIVE:-0}" = "1" ]; then
        return 0
    fi

    if [ -z "$CARLA_STARTED_PIDS" ]; then
        CARLA_STARTED_PIDS="$(carla_process_ids | tr '\n' ' ')"
    fi

    if [ -n "$CARLA_STARTED_PIDS" ]; then
        kill $CARLA_STARTED_PIDS >/dev/null 2>&1 || true
        sleep 2

        lingering_pids=""
        for pid in $CARLA_STARTED_PIDS; do
            if kill -0 "$pid" >/dev/null 2>&1; then
                lingering_pids="${lingering_pids}${lingering_pids:+ }$pid"
            fi
        done

        if [ -n "$lingering_pids" ]; then
            kill -9 $lingering_pids >/dev/null 2>&1 || true
        fi
    fi

    rm -f "$CARLA_MODE_FILE"
}

validate_existing_carla_mode() {
    if [ "$CARLA_RENDER_MODE" != "offscreen" ]; then
        return 0
    fi

    if [ ! -f "$CARLA_MODE_FILE" ]; then
        echo "CARLA is already listening on $OAK_CARLA_HOST:$OAK_CARLA_PORT, but its launch mode is unknown. Stop it and rerun so OaK can attach to an offscreen server, or explicitly opt into a visible server with OAK_CARLA_RENDER_MODE=windowed." >&2
        return 1
    fi

    if [ "$(cat "$CARLA_MODE_FILE")" != "offscreen" ]; then
        echo "CARLA is already listening in windowed mode. Stop it and rerun so OaK can attach to an offscreen server, or explicitly opt into the GUI with OAK_CARLA_RENDER_MODE=windowed." >&2
        return 1
    fi
}

trap shutdown_started_carla_server EXIT

validate_release_binary
export OAK_ROOT_DIR="$ROOT_DIR"
export OAK_CARLA_HOST="${OAK_CARLA_HOST:-127.0.0.1}"
export OAK_CARLA_PORT="${OAK_CARLA_PORT:-2000}"

if requires_python_prefix; then
    validate_python_prefix
    export_python_runtime
    validate_python_imports
fi

if should_preflight_metal; then
    validate_metal_runtime
fi

if [ "$REQUESTED_ENV" = "carla" ] || [ "$REQUESTED_ENV" = "all" ]; then
    if ! oak_export_carla_whisky_env "$ROOT_DIR"; then
        die "Missing Whisky CARLA configuration. Run './doctor.sh carla' or './setup.sh' after installing the matching Windows Python version for the packaged CARLA client inside the Whisky bottle."
    fi

    if carla_server_listening; then
        if ! validate_existing_carla_mode; then
            exit 1
        fi
    else
        CARLA_PREEXISTING_PIDS="$(carla_process_ids | tr '\n' ' ')"

        if [ "${OAK_CARLA_AUTOSTART:-1}" != "1" ]; then
            die "CARLA is not listening on $OAK_CARLA_HOST:$OAK_CARLA_PORT. Start it in Whisky first or rerun with OAK_CARLA_AUTOSTART=1. For diagnostics, run './doctor.sh carla'."
        fi

        if ! validate_carla_autostart_policy; then
            exit 1
        fi

        echo "Starting CARLA via Whisky..." >&2
        if ! start_carla_server; then
            echo "Failed to launch CARLA via Whisky. Check ${OAK_CARLA_LOG:-$ROOT_DIR/.oak-carla.log}." >&2
            exit 1
        fi

        if ! wait_for_carla_server; then
            echo "CARLA did not start listening on $OAK_CARLA_HOST:$OAK_CARLA_PORT. Check ${OAK_CARLA_LOG:-$ROOT_DIR/.oak-carla.log}." >&2
            exit 1
        fi

        record_carla_launch_mode
        CARLA_STARTED_SERVER=1
        capture_started_carla_pids
    fi
fi

set +e
"$OAK_BINARY" "$@"
OAK_EXIT_CODE=$?
set -e
exit "$OAK_EXIT_CODE"
