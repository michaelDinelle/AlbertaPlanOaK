#!/usr/bin/env bash
# Tier G1 eligibility traces deep paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
# The real test — does TD(λ) multi-step credit assignment raise per-step learning velocity
# vs F1's λ=0 baseline (+1.67M mean lift at 10M)?
#
# Distributional acceptance (the signal is "≥ 7 of 10 seeds beat F1 counterpart"):
#   10/10 wins AND 0 NF AND mean lift ≥ +1.9M (F1 +1.67M; +15% target)
#   AND chatter median ≤ 0.08 AND ≥ 7 seeds beat F1 counterpart
#   AND traceValueMeanAbs stable throughout run.
#
# Kill: any NF, mean lift < F1's +1.67M (regression), chatter > 0.20,
#   < 4 seeds beat F1 counterpart, traceValueMeanAbs > 200 (divergence).
#
# ETA ~1h10m wall at PERFOPT2's ~2500 SPS/stream.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-traces-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier G1 eligibility-traces 10M bench (10M oak + 10M random, 10 seeds)..."
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
