#!/usr/bin/env bash
# Tier C1 deep paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
# Validates whether the C1 transient peak walking (6/10 seeds reach rewardRate > 0 at
# 1.5M) consolidates into sustained walking by 10M, or stays transient. ETA ~2.8 hr wall
# at the C1 1541 sps/stream concurrent throughput.
#
# Uses `OaK run` — hot-path mode, no per-kernel waitUntilCompleted overhead.
# Random streams finish first (~6 min); oak streams hold the GPU for the full 10M.
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
export OAK_HEARTBEAT_STEPS=200000   # 50 heartbeats over 10M (4× sparser than 1.5M run)

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-c1-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier C1 10M bench (10M oak + 10M random, 10 seeds)..."
echo "  start: $(date -u +%FT%TZ)"

for seed in {0..9}; do
  echo "Starting seed $seed random..."
  OAK_METRICS_STORAGE="$RUN_DIR/seed${seed}-random" \
  OAK_SEED=$seed \
  OAK_POLICY=random \
  OAK_STOP_AFTER_STEPS=10000000 \
  ./.build/release/OaK run "$OAK_ENV" > /dev/null 2>&1 &

  echo "Starting seed $seed oak..."
  OAK_METRICS_STORAGE="$RUN_DIR/seed${seed}-oak" \
  OAK_SEED=$seed \
  OAK_POLICY=oak \
  OAK_STOP_AFTER_STEPS=10000000 \
  ./.build/release/OaK run "$OAK_ENV" > /dev/null 2>&1 &
done

echo "All 20 processes are running in the background."
wait
echo "  end:   $(date -u +%FT%TZ)"
echo "All done."
