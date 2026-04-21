#!/usr/bin/env bash
# PERF-OPT paired bench (oak_post_learn action region parallelization, post-D2):
# HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps. Validates two things:
#
#   1. Bit-identity parity vs D2 commit 23d28cdb9 — the algorithm is unchanged. A 10k
#      single-process diagnostic showed chatter / rewardRate / rewardSum / scoreGradL2
#      diff = 0.0e+00 exact. Per-seed lifts on this 1.5M bench should match the D2
#      bench exactly (within tiny scheduling-jitter noise on SPS but ZERO on
#      algorithmic outputs). Any per-seed lift drift > ±0.1% would indicate a hidden
#      bit-id break we missed.
#
#   2. Concurrent-mode agentMs improvement. The single-process diagnostic showed
#      oak_post_learn drop 138.66 → 14.71 µs (-89.4%) and total GPU compute drop
#      244.09 → 120.27 µs (-50.7%) — saved ~124 µs/step on the GPU. In concurrent
#      bench (20 processes), the wall-clock improvement should manifest as agentMs
#      noticeably below D2's 0.6018 ms median.
#
# The change: oak_post_learn region 3 (action mu_p computation) was 6 threads each
# doing K·R = 2048 sequential fma in 1 simdgroup with 26 idle lanes (~110 µs / step,
# 55% of total GPU). Now D·K threads, dispatched as D simdgroups of K=32 lanes each.
# Lane k of TG p computes its own dot in parallel; lane 0 reads each lane's dot via
# simd_shuffle and applies chord with fma in the original kk=0..K-1 order — bit-id
# preserved by construction (Inv-10).
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
export OAK_WARMUP_STEPS=1000
export OAK_HEARTBEAT_STEPS=50000

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-perfopt-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 PERF-OPT bench (1.5M oak + 1.5M random, 10 seeds)..."

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
