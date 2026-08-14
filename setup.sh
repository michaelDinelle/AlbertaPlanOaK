#!/bin/bash
set -euo pipefail

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
echo "Setup complete. Run: ./fleet.py run <steps>"
