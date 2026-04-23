#!/usr/bin/env bash
# Tier D2 + PERFOPT + PERFOPT2 deep paired bench: HalfCheetah-v4, 10 seeds × {oak, random},
# 10M steps. Validates whether D2's +28% max peak at 1.5M (+0.162 vs C1 +0.127) compounds
# into a true Step-9 win at 10M scale — i.e. max peak rewardRate should exceed the C1 10M
# trajectory (+0.310 at 10M) if the prioritized-sweeping mechanism is actually locking in
# transient walking cells better than the uniform sweep.
#
# Current agent stack:
#   C1 encoder (std_obs ‖ obsTrace Fourier features, Plan Step 3)
#   Tier B dense optTransition + RR-Subtasks + MetaOptimize per-feature α
#   Tier D2 prioritized sweeping (Moore-Atkeson 1993, Plan footnote 13)
#   PERFOPT oak_post_learn action region parallelization (bit-id with D2 algorithm)
#   PERFOPT2 kernel fusion: oak_bk_encode + oak_meta_predict (bit-id)
#
# ETA ~1h10m wall at PERFOPT2's 2534 sps/stream concurrent throughput (vs ~2h48m at C1).
#
# Uses `OaK run` — hot-path mode, no per-kernel waitUntilCompleted overhead.
# Random streams finish first (~4 min); oak streams hold the GPU for the full 10M.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f "$ROOT/.build/release/OaK" ]; then
    echo "Building OaK..."
    swift build -c release
fi

export OAK_ENV="HalfCheetah-v4"
export OAK_NUM_ENVS=1
export OAK_HEARTBEAT_STEPS=200000   # 50 heartbeats over 10M (matches C1 10M cadence)

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-d2-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 D2+PERFOPT2 10M bench (10M oak + 10M random, 10 seeds)..."
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
