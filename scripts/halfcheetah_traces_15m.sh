#!/usr/bin/env bash
# Tier G1 eligibility traces paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
# Plan Step 1 completion: TD(λ) on valueW + Emphatic TD(λ) on per-option ψ with structural
# λ = 1 − 1/baseO (0.941 on HC, τ≈17 steps — classical TD(0.94) range).
#
# Screen acceptance (hard-gated):
#   10/10 wins AND 0 NF AND mean lift ≥ +240k (F1 baseline +237k, parity-or-better)
#   AND chatter median ≤ 0.08 AND ≤ 2 negative seeds.
# Kill: any NF, lift < +150k, chatter > 0.25, > 4 negative seeds.
# If screen passes, proceed to 10×10M (the learning-velocity test vs F1 +1.67M baseline).
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-traces-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier G1 eligibility-traces bench (1.5M oak + 1.5M random, 10 seeds)..."

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
