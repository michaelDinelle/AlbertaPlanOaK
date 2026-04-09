#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/carla_whisky_env.sh"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

echo "OaK 2026 Setup - 7 Environments"

# 1. Install Miniforge if missing
CONDA_DIR="$HOME/miniforge3"
CONDA_BIN="$CONDA_DIR/bin/conda"
CONDA_SH="$CONDA_DIR/etc/profile.d/conda.sh"
OAK_ENV_DIR="$CONDA_DIR/envs/oak"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
DIRECT_PYTHON="$OAK_ENV_DIR/bin/python"
PYTHON_IMPORT_CHECK='import gymnasium, minigrid, PyFlyt, dm_control, mujoco'

if [ ! -f "$CONDA_BIN" ]; then
    echo "Installing Miniforge3..."
    rm -rf "$CONDA_DIR"
    curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
    rm /tmp/miniforge.sh
fi

[ -x "$CONDA_BIN" ] || fail "Conda binary not found at $CONDA_BIN"
[ -f "$CONDA_SH" ] || fail "Conda activation script not found at $CONDA_SH"

# 2. Setup conda for this script
export PATH="$CONDA_DIR/bin:$PATH"
source "$CONDA_SH"

# 3. Create clean environment
echo "Creating oak environment..."
if [ -d "$OAK_ENV_DIR" ]; then
    echo "Removing existing oak environment..."
    if [ -f "$OAK_ENV_DIR/conda-meta/history" ]; then
        conda env remove -y -p "$OAK_ENV_DIR"
    else
        rm -rf "$OAK_ENV_DIR"
    fi
fi

conda create -y -p "$OAK_ENV_DIR" python=3.11
[ -f "$OAK_ENV_DIR/conda-meta/history" ] || fail "Conda environment metadata missing at $OAK_ENV_DIR/conda-meta/history"

conda activate "$OAK_ENV_DIR"
[ "${CONDA_PREFIX:-}" = "$OAK_ENV_DIR" ] || fail "Expected CONDA_PREFIX=$OAK_ENV_DIR, got ${CONDA_PREFIX:-<unset>}"
[ -x "$DIRECT_PYTHON" ] || fail "Python executable not found at $DIRECT_PYTHON"
[ -f "$REQUIREMENTS_FILE" ] || fail "Missing requirements file at $REQUIREMENTS_FILE"

# 4. Install pybullet from conda-forge (pre-built ARM64)
echo "Installing pybullet..."
conda install -c conda-forge pybullet -y

# 5. Install all Python packages in one go
echo "Installing Python packages..."
"$DIRECT_PYTHON" -m pip install --upgrade pip setuptools wheel
"$DIRECT_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

echo "Validating oak environment..."
[ -f "$OAK_ENV_DIR/conda-meta/history" ] || fail "Conda environment metadata missing after install at $OAK_ENV_DIR/conda-meta/history"
"$DIRECT_PYTHON" -c "$PYTHON_IMPORT_CHECK" >/dev/null || fail "Direct prefix Python import validation failed"
"$CONDA_BIN" run -p "$OAK_ENV_DIR" python -c "$PYTHON_IMPORT_CHECK" >/dev/null || fail "conda run -p validation failed"
echo "✓ Python environment validated"

# 6. Configure CARLA
echo ""
echo "Configuring CARLA..."
[ -f "$ROOT_DIR/scripts/carla_whisky_bridge.py" ] || fail "Missing CARLA Whisky bridge script at $ROOT_DIR/scripts/carla_whisky_bridge.py"

WHISKY_PREFIX="$(oak_whisky_prefix)" || fail "Whisky bottle '$(oak_whisky_bottle_name)' not found. Create or import it in Whisky first."
WHISKY_CMD="$(oak_whisky_cmd_path)"
WHISKY_BOTTLE="$(oak_whisky_bottle_name)"
CARLA_EGG_HOST="$(oak_carla_egg_host "$WHISKY_PREFIX")"
CARLA_PYTHON_VERSION="$(oak_carla_python_version "$CARLA_EGG_HOST")"
WHISKY_PYTHON_HOST="$(oak_whisky_python_host "$WHISKY_PREFIX" "$CARLA_PYTHON_VERSION")"
CARLA_PY_HOST="$(oak_carla_python_api_host "$CARLA_EGG_HOST")"
CARLA_APP_HOST="$(oak_carla_app_host "$WHISKY_PREFIX")"
CARLA_ROOT="$(oak_carla_root_host "$CARLA_PY_HOST")"
CARLA_EXE="$(oak_carla_exe_host "$CARLA_ROOT")"
CARLA_LINK="$HOME/Applications/CARLA_0.9.15"
CARLA_WORKTREE_LINK="$ROOT_DIR/CARLA_0.9.15"
BRIDGE_SCRIPT_WIN="$(oak_to_windows_path "$ROOT_DIR/scripts/carla_whisky_bridge.py")"
WHISKY_PYTHON_WIN="$(oak_to_windows_path "$WHISKY_PYTHON_HOST")"
CARLA_EGG_WIN="$(oak_to_windows_path "$CARLA_EGG_HOST")"
CARLA_PY_WIN="$(oak_to_windows_path "$CARLA_PY_HOST")"

[ -x "$WHISKY_CMD" ] || fail "Whisky CLI not found at $WHISKY_CMD"
[ -n "$WHISKY_PYTHON_HOST" ] || fail "Windows Python $CARLA_PYTHON_VERSION not found in Whisky bottle '$WHISKY_BOTTLE'. Install a matching Windows Python for $CARLA_EGG_HOST."
[ -f "$WHISKY_PYTHON_HOST" ] || fail "Windows Python executable not found: $WHISKY_PYTHON_HOST"
[ -n "$CARLA_EGG_HOST" ] || fail "CARLA Python egg not found in Whisky bottle '$WHISKY_BOTTLE'"
[ -f "$CARLA_EGG_HOST" ] || fail "CARLA Python egg not found: $CARLA_EGG_HOST"
[ -d "$CARLA_PY_HOST" ] || fail "CARLA Python API directory not found: $CARLA_PY_HOST"
[ -d "$CARLA_APP_HOST" ] || fail "CARLA.app wrapper not found: $CARLA_APP_HOST"
[ -f "$CARLA_EXE" ] || fail "CARLA executable not found: $CARLA_EXE"

if xattr -p com.apple.quarantine "$CARLA_APP_HOST" >/dev/null 2>&1; then
    echo "Removing Gatekeeper quarantine from CARLA.app..."
    xattr -d com.apple.quarantine "$CARLA_APP_HOST" || fail "Failed to clear com.apple.quarantine on $CARLA_APP_HOST"
fi

mkdir -p "$HOME/Applications"
ln -sfn "$CARLA_ROOT" "$CARLA_LINK"
ln -sfn "$CARLA_LINK" "$CARLA_WORKTREE_LINK"

(
    eval "$("$WHISKY_CMD" shellenv "$WHISKY_BOTTLE")"
    wine64 "$WHISKY_PYTHON_WIN" "$BRIDGE_SCRIPT_WIN" healthcheck "$CARLA_EGG_WIN" "$CARLA_PY_WIN"
) >/dev/null
echo "✓ CARLA configured"

# 7. Build OaK
echo ""
echo "Building OaK..."
swift build -c release

echo ""
echo "✓ Setup complete"
echo ""
echo "To use:"
echo " source $CONDA_DIR/etc/profile.d/conda.sh"
echo " conda activate $OAK_ENV_DIR"
echo "./run.sh"
