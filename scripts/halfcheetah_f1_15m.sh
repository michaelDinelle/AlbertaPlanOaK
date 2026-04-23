#!/usr/bin/env bash
# Tier F1 paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
# Validates the plateau-breaker σ = 1/√A structural policy variance change in
# sampleAndScoreActions. Prior σ=1 gave per-step Gaussian noise magnitude √6 ≈ 2.45
# on HC (~100% of the [-1, 1] action range), drowning any committed option policy.
# F1 scales per-dim σ to 0.41 on HC so committed gaits can dominate the signal over
# multi-step walking patterns.
#
# Williams 1992 REINFORCE for N(μ, σ²):
#   raw = μ + σ·z,  z ~ N(0, 1)
#   scoreGrad = d log π/dμ = (raw−μ)/σ² = z/σ
#
# On HC (A=6): σ=0.41, invSigma=√6=2.45. Kim implicit alphaPi auto-adjusts: ∥∇log π∥²
# scales by 1/σ² = A, so alphaPi shrinks by A and Head 2 (optPolicy REINFORCE) update
# magnitude scales by σ — slower, more committed policy adaptation (intended).
#
# 1.5M screen acceptance: 10/10 wins AND 0 NF AND mean lift ≥ +90k AND chatter median
# below D3's 0.11 AND ≤4 negative seeds. Kill: any NF, lift < +40k, chatter > 0.30
# (unexpected rise), >4 negative.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-f1-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier F1 bench (1.5M oak + 1.5M random, 10 seeds)..."

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
