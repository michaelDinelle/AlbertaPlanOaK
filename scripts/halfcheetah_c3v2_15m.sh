#!/usr/bin/env bash
# Tier C3 v2 paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
# Validates the variance-balanced recurrent-encoder fix (W_state at sqrt(baseO/R) so the
# state-region contribution to z matches the obs-region contribution; obs/trace cols stay
# at RFF bandwidth-1) vs the C3 v1 regression (.oak-benchmarks/halfcheetah-c3-15m, RFF-1
# across all three regions → state dominated z-variance, chatter collapsed 0.21→0.026,
# lift halved to +52k/seed). Acceptance: lift not worse than C1 baseline (+111k/seed) by
# more than 5%; walking-touch count ideally up vs both C1 and C3 v1.
# Uses `OaK run` — hot-path mode, no per-kernel waitUntilCompleted overhead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f "$ROOT/.build/release/OaK" ]; then
    echo "Building OaK..."
    swift build -c release
fi

export OAK_ENV="HalfCheetah-v4"
export OAK_NUM_ENVS=1
export OAK_WARMUP_STEPS=1000
export OAK_HEARTBEAT_STEPS=50000

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-c3v2-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier C3 v2 bench (1.5M oak + 1.5M random, 10 seeds)..."

for seed in {0..9}; do
  echo "Starting seed $seed random..."
  OAK_METRICS_STORAGE="$RUN_DIR/seed${seed}-random" \
  OAK_SEED=$seed \
  OAK_POLICY=random \
  OAK_STOP_AFTER_STEPS=1500000 \
  ./.build/release/OaK run "$OAK_ENV" > /dev/null 2>&1 &

  echo "Starting seed $seed oak..."
  OAK_METRICS_STORAGE="$RUN_DIR/seed${seed}-oak" \
  OAK_SEED=$seed \
  OAK_POLICY=oak \
  OAK_STOP_AFTER_STEPS=1500000 \
  ./.build/release/OaK run "$OAK_ENV" > /dev/null 2>&1 &
done

echo "All 20 processes are running in the background."
wait
echo "All done."
