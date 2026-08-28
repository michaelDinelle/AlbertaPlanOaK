#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
command -v python3.11 >/dev/null || { echo "python3.11 not found." >&2; exit 1; }

[ -x "$VENV/bin/python" ] || python3.11 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements.txt"

mkdir -p "$ROOT/.build"
xcrun -sdk macosx metal -fno-fast-math -c "$ROOT/Sources/Kernels.metal" -o "$ROOT/.build/OakKernel.air"
xcrun -sdk macosx metallib "$ROOT/.build/OakKernel.air" -o "$ROOT/.build/OakKernel.metallib"
swift build -c release
echo "Setup complete. Run: ./fleet.py run"
