#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${OAK_BINARY:-$ROOT/.build/Release/OaK}"
[ -x "$BIN" ] || { printf 'Missing release binary at %s. Run ./setup.sh first.\n' "$BIN" >&2; exit 1; }
export OS_ACTIVITY_MODE=disable
exec "$BIN" benchmark
