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
printf '_oak_act\n_worker\n_env_step\n' > "$ROOT/.build/hot.order"
CACHE_LINE="$(sysctl -n hw.cachelinesize)"
clang -O3 -mcpu=apple-m4 -flto=full "-falign-functions=$CACHE_LINE" -fno-fast-math -ffp-contract=off -Werror=double-promotion -Wall -Wextra \
  "-DOAK_PRESET_$(printf '%s' "$PRESET" | tr a-z A-Z)" -I"$ROOT/Sources" \
  "$ROOT"/Sources/*.c "$ROOT"/Sources/envs/*.c -lm -lpthread -Wl,-order_file,"$ROOT/.build/hot.order" -o "$ROOT/.build/oak"
first="$(nm -n "$ROOT/.build/oak" | awk '$2 ~ /^[tT]$/ && $3 != "__mh_execute_header" { print $3; exit }')"
[ "$first" = _oak_act ] || { echo "the linker placed $first first, not _oak_act" >&2; exit 1; }
printf '%s\n' "$PRESET" > "$PRESET_FILE"
echo "Setup complete: OaK $PRESET. Record with ./fleet.py; watch with: cd frontend && bun install && bun run dev"
