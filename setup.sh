#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PRESET_FILE="$ROOT/.build/preset"
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

mkdir -p "$ROOT/.build"
python3 "$ROOT/catalog.py"
printf '%s\n' _oak_act _constructFeatures _predictAndControl _replaceWorstFeature _emitPolicy _sampleAction _sampleDiscreteAction _actorCoefficients _updateUtilities _worker _env_step > "$ROOT/.build/hot.order"
CACHE_LINE="$(sysctl -n hw.cachelinesize)"
FLAGS=(-O3 -mcpu=apple-m4 -flto=full "-falign-functions=$CACHE_LINE" -fno-fast-math -ffp-contract=off -Werror=double-promotion -Wall -Wextra
  "-DOAK_PRESET_$(printf '%s' "$PRESET" | tr a-z A-Z)" -I"$ROOT/Sources")
SRCS=("$ROOT"/Sources/*.c "$ROOT"/Sources/envs/*.c "$ROOT"/.build/catalog/*.c)
OBJ="$ROOT/.build/obj"
rm -rf "$OBJ"
mkdir -p "$OBJ$ROOT/Sources/envs" "$OBJ$ROOT/.build/catalog"
printf '%s\n' "${SRCS[@]}" | xargs -P "$(sysctl -n hw.ncpu)" -I {} clang "${FLAGS[@]}" -c {} -o "$OBJ{}.o"
OBJS=("${SRCS[@]/#/$OBJ}")
clang "${FLAGS[@]}" "${OBJS[@]/%/.o}" -lm -lpthread -Wl,-order_file,"$ROOT/.build/hot.order" -Wl,-object_path_lto,"$ROOT/.build/lto.o" -o "$ROOT/.build/oak"
first="$(nm -n "$ROOT/.build/oak" | awk '$2 ~ /^[tT]$/ && $3 != "__mh_execute_header" { print $3; exit }')"
[ "$first" = _oak_act ] || { echo "the linker placed $first first, not _oak_act" >&2; exit 1; }
printf '%s\n' "$PRESET" > "$PRESET_FILE"
echo "Setup complete: OaK $PRESET. Record with ./fleet.py; watch with: cd frontend && bun install && bun run dev"
