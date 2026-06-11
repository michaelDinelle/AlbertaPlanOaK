#!/bin/bash
set -euo pipefail

# OaK setup: the environment is GPU-resident (Sources/OakPhysics.metal, an
# fp32 port of MuJoCo HalfCheetah verified against fp64 — see
# docs/gpu-physics-port-spec.md). The only native dependencies are the Metal
# toolchain and Swift. Python + mujoco are used solely by validation and
# replay tooling (tools/physics_parity.py oracle, tools/replay_render.py).

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${OAK_VENV_DIR:-$ROOT/.venv}"
PYBOOT="$(command -v python3.11 || command -v python3 || true)"

[ -n "$PYBOOT" ] || { echo "Python 3 not found." >&2; exit 1; }
[ -x "$VENV/bin/python" ] || "$PYBOOT" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements.txt"

mkdir -p "$ROOT/.build"
xcrun -sdk macosx metal -fno-fast-math -c "$ROOT/Sources/OakKernel.metal" -o "$ROOT/.build/OakKernel.air"
xcrun -sdk macosx metallib "$ROOT/.build/OakKernel.air" -o "$ROOT/.build/OakKernel.metallib"
swift build -c release
echo "Setup complete. ./benchmark.sh"
