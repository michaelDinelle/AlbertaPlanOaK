#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PRESET_FILE="$ROOT/.build/OakKernel.preset"
PRESET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --preset) PRESET="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
if [ -z "$PRESET" ] && [ -f "$PRESET_FILE" ]; then PRESET="$(cat "$PRESET_FILE")"; fi
case "$PRESET" in
  base|pro|max|ultra) ;;
  *) echo "choose a geometry preset: ./setup.sh --preset base|pro|max|ultra" >&2; exit 1 ;;
esac
command -v python3.11 >/dev/null || { echo "python3.11 not found." >&2; exit 1; }

[ -x "$VENV/bin/python" ] || python3.11 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements.txt"

mkdir -p "$ROOT/.build"
xcrun -sdk macosx metal -fno-fast-math "-DOAK_PRESET_$(printf '%s' "$PRESET" | tr a-z A-Z)" -c "$ROOT/Sources/Kernels.metal" -o "$ROOT/.build/OakKernel.air"
xcrun -sdk macosx metallib "$ROOT/.build/OakKernel.air" -o "$ROOT/.build/OakKernel.metallib"
printf '%s\n' "$PRESET" > "$PRESET_FILE"
swift build -c release
echo "Setup complete: OaK $PRESET. Run: ./fleet.py run"
