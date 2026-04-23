#!/usr/bin/env bash
# Step 11 sharpening deep paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
# THE plateau-past-2M test — does G&T picking by planUtil (Q^k − V EMA) instead of utilOpt
# (|tdPsi|² EMA) restore continual improvement past the F1 convergence point?
#
# Distributional acceptance (key signal is "5+ of 10 seeds show running-max-peak growth
# between 2M and 10M heartbeats"):
#   10/10 wins AND 0 NF AND mean lift ≥ +1.5M AND ≥5 of 10 seeds grow post-2M
#   AND chatter ≤ 0.15.
#
# Baseline at 10M:
#   F1 10M: mean lift +1.67M, 0/10 seeds grew post-2M (that's the bug Step 11 fixes).
#
# Kill: any NF, mean lift < +1M, fewer than 3 seeds show post-2M growth, chatter > 0.30.
#
# ETA ~1h10m wall at PERFOPT2's 2534 SPS/stream.
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
export OAK_HEARTBEAT_STEPS=200000

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-step11-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Step 11 sharpening 10M bench (10M oak + 10M random, 10 seeds)..."
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
