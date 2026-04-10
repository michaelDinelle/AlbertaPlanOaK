#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/carla_whisky_env.sh"

OAK_BINARY="${OAK_BINARY:-$ROOT_DIR/.build/release/OaK}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
CONDA_BIN="${CONDA_BIN:-$CONDA_DIR/bin/conda}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/miniforge3/envs/oak}"
PYTHON_VERSION="${OAK_PYTHON_VERSION:-3.11}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
PYTHON_LIB="$CONDA_ENV_DIR/lib/libpython$PYTHON_VERSION.dylib"
PYTHON_IMPORT_CHECK='import gymnasium, minigrid, PyFlyt, dm_control.suite, mujoco'
CHECK="${1:-all}"

die() {
    echo "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: ./doctor.sh [python|metal|envs|carla|all]

Checks:
  python  Verify the oak Conda prefix, direct imports, and conda run -p imports.
  metal   Verify the release binary sees a Metal device and can submit a compute kernel.
  envs    Run bounded launcher smoke tests for minigrid, pyflyt, cheetah, walker, and humanoid.
  carla   Verify Whisky CARLA paths and the bridge healthcheck without autostarting the server.
  all     Run python, metal, envs, and carla in order.
EOF
}

build_release_if_needed() {
    if [ "${OAK_DOCTOR_SKIP_BUILD:-0}" = "1" ]; then
        return
    fi

    if [ ! -x "$OAK_BINARY" ]; then
        (
            cd "$ROOT_DIR"
            swift build -c release
        )
    fi
}

doctor_python() {
    [ -x "$CONDA_BIN" ] || die "Missing conda binary at $CONDA_BIN"
    [ -d "$CONDA_ENV_DIR" ] || die "Missing conda environment at $CONDA_ENV_DIR"
    [ -f "$CONDA_ENV_DIR/conda-meta/history" ] || die "Invalid conda environment at $CONDA_ENV_DIR: missing conda-meta/history"
    [ -x "$PYTHON_BIN" ] || die "Missing Python executable at $PYTHON_BIN"
    [ -f "$PYTHON_LIB" ] || die "Missing Python library at $PYTHON_LIB"

    echo "[doctor] python: validating direct prefix imports"
    "$PYTHON_BIN" -c "$PYTHON_IMPORT_CHECK" >/dev/null

    echo "[doctor] python: validating conda run -p imports"
    "$CONDA_BIN" run -p "$CONDA_ENV_DIR" python -c "$PYTHON_IMPORT_CHECK" >/dev/null

    echo "[doctor] python: ok"
}

doctor_metal() {
    build_release_if_needed
    echo "[doctor] metal: probing release runtime"
    "$OAK_BINARY" doctor-metal
    echo "[doctor] metal: ok"
}

doctor_envs() {
    build_release_if_needed
    for env_name in minigrid pyflyt cheetah walker humanoid; do
        echo "[doctor] envs: $env_name"
        (
            cd "$ROOT_DIR"
            OAK_ALLOW_CPU_FALLBACK=0 \
            OAK_MAX_STEPS="${OAK_DOCTOR_ENV_STEPS:-1}" \
            OAK_METRICS_STDOUT_SUMMARIES=0 \
            ./run.sh "$env_name"
        )
    done

    echo "[doctor] envs: ok"
}

doctor_carla() {
    local bridge_script_win

    [ -f "$ROOT_DIR/scripts/carla_whisky_bridge.py" ] || die "Missing CARLA bridge script at $ROOT_DIR/scripts/carla_whisky_bridge.py"
    if ! oak_export_carla_whisky_env "$ROOT_DIR"; then
        die "Missing Whisky CARLA configuration. Run './setup.sh' after installing the matching Windows Python inside the Whisky bottle."
    fi

    bridge_script_win="$(oak_to_windows_path "$ROOT_DIR/scripts/carla_whisky_bridge.py")"

    echo "[doctor] carla: validating Whisky bridge healthcheck"
    (
        eval "$("$OAK_WHISKY_CMD" shellenv "$OAK_WHISKY_BOTTLE")"
        wine64 "$OAK_CARLA_WIN_PYTHON" "$bridge_script_win" healthcheck "$OAK_CARLA_WIN_EGG" "$OAK_CARLA_WIN_PYTHONAPI"
    ) >/dev/null

    echo "[doctor] carla: ok"
}

case "$CHECK" in
    python)
        doctor_python
        ;;
    metal)
        doctor_metal
        ;;
    envs)
        doctor_envs
        ;;
    carla)
        doctor_carla
        ;;
    all)
        doctor_python
        doctor_metal
        doctor_envs
        doctor_carla
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
