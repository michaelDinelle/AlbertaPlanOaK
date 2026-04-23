#!/usr/bin/env bash
# Tier C2 retry paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
# Validates the third C2 attempt — true Sutton-Precup-Singh option termination
# (termW + closed-form geometric Q^k + hold-or-refresh chord), this time on top of the
# C1 substrate (encoder reads [std_obs ‖ obsTrace] so β^k(s) conditions on slow-frequency
# dynamics the option-critic gradient can actually learn from).
#
# HARD-GATED acceptance (per docs/roadmap.md): 10/10 wins, mean lift >= +105k/seed (C1
# baseline +111k, tolerate -5%), 0 NF div, chatter median <= 0.30, AND (peak walking
# touches >= 6/10 seeds OR max peak rewardRate > +0.20). Kill on any of: mean lift
# < +100k, more than 2 seeds with negative lift, any NF, chatter median > 0.40, or
# median rewardRate < -0.40.
#
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
export OAK_HEARTBEAT_STEPS=50000

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-c2retry-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier C2 retry bench (1.5M oak + 1.5M random, 10 seeds)..."

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
