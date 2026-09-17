#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
STORED="$ROOT/.build/OakKernel.geometry"
GEOMETRY="${1:-}"
if [ -z "$GEOMETRY" ] && [ -f "$STORED" ]; then GEOMETRY="$(cat "$STORED")"; fi
if [[ ! "$GEOMETRY" =~ ^p([0-9]+)_s([0-9]+)_c([0-9]+)_k([0-9]+)_u([0-9]+)_t([0-9]+)$ ]]; then
  echo "declare the geometry: ./setup.sh p<percepts>_s<slots>_c<candidates>_k<options>_u<updates>_t<traces>" >&2
  exit 1
fi
PERCEPTS="${BASH_REMATCH[1]}"; SLOTS="${BASH_REMATCH[2]}"
CANDIDATES="${BASH_REMATCH[3]}"; OPTIONS="${BASH_REMATCH[4]}"; UPDATES="${BASH_REMATCH[5]}"; TRACES="${BASH_REMATCH[6]}"

mkdir -p "$ROOT/.build"
xcrun -sdk macosx metal -fno-fast-math \
  -DOAK_PERCEPTS="$PERCEPTS" -DOAK_SLOTS="$SLOTS" \
  -DOAK_CANDIDATES="$CANDIDATES" -DOAK_OPTIONS="$OPTIONS" -DOAK_UPDATES="$UPDATES" -DOAK_TRACES="$TRACES" \
  -DOAK_GEOMETRY_MARKER="oak_geometry_$GEOMETRY" \
  -c "$ROOT/Sources/Kernels.metal" -o "$ROOT/.build/OakKernel.air"
xcrun -sdk macosx metallib "$ROOT/.build/OakKernel.air" -o "$ROOT/.build/OakKernel.metallib"
printf '%s\n' "$GEOMETRY" > "$STORED"
swift build -c release
echo "Setup complete: $PERCEPTS percepts, $SLOTS slots x $CANDIDATES candidates, $OPTIONS options, $UPDATES updated per step, $TRACES traces"
