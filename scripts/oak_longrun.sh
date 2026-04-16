#!/usr/bin/env bash
# Long focused `OaK run` — requires OAK_STOP_AFTER_STEPS (aggregate measured steps).
# Per-env depth ≈ OAK_STOP_AFTER_STEPS / OAK_NUM_ENVS. See AGENTS.md (Long runs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${OAK_STOP_AFTER_STEPS:?set OAK_STOP_AFTER_STEPS (aggregate measured step target)}"
ENV_ID="${1:-${OAK_ENV:?set OAK_ENV or pass env-id as first argument}}"
exec ./.build/release/OaK run "$ENV_ID"
