#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
[ $# -eq 0 ] || { echo "setup.sh takes no arguments; geometry is declared in Sources/geometry.h" >&2; exit 1; }

mkdir -p "$ROOT/.build"
python3 "$ROOT/catalog.py"
printf '%s\n' _oak_act _constructFeatures _predictAndControl _replaceWorstFeature _emitPolicy _sampleAction _sampleDiscreteAction _actorCoefficients _updateUtilities _worker _env_step > "$ROOT/.build/hot.order"
CACHE_LINE="$(sysctl -n hw.cachelinesize)"
FLAGS=(-O3 -mcpu=apple-m4 -flto=full "-falign-functions=$CACHE_LINE" -fno-fast-math -ffp-contract=off -Werror=double-promotion -Wall -Wextra
  -I"$ROOT/Sources")
SRCS=("$ROOT"/Sources/*.c "$ROOT"/Sources/envs/*.c "$ROOT"/.build/catalog/*.c)
OBJ="$ROOT/.build/obj"
rm -rf "$OBJ"
mkdir -p "$OBJ$ROOT/Sources/envs" "$OBJ$ROOT/.build/catalog"
printf '%s\n' "${SRCS[@]}" | xargs -P "$(sysctl -n hw.ncpu)" -I {} clang "${FLAGS[@]}" -c {} -o "$OBJ{}.o"
OBJS=("${SRCS[@]/#/$OBJ}")
clang "${FLAGS[@]}" "${OBJS[@]/%/.o}" -lm -lpthread -Wl,-order_file,"$ROOT/.build/hot.order" -Wl,-object_path_lto,"$ROOT/.build/lto.o" -o "$ROOT/.build/oak"
first="$(nm -n "$ROOT/.build/oak" | awk '!found && $2 ~ /^[tT]$/ && $3 != "__mh_execute_header" { print $3; found = 1 }')"
[ "$first" = _oak_act ] || { echo "the linker placed $first first, not _oak_act" >&2; exit 1; }
echo "Setup complete. Record with ./fleet.py; watch with: cd frontend && bun install && bun run dev"
