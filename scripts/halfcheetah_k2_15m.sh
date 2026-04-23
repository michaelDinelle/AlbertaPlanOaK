#!/usr/bin/env bash
# Tier K2 (Plan Step 2 "feature combinations": stochastic compositional reseed)
# paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
#
# K2 replaces K1's greedy top-2 parent choice with deterministic stochastic sampling
# without replacement from paired utility mass. This keeps Step-2 composition while
# preserving coverage, so the 1.5M screen is treated only as a safety check.
#
# Primary signal: mean lift >= +220k (screen only; 10M remains the acceptance gate).
# Secondary: chatter median <= 0.08. Tertiary: chordEffectiveK trend vs K1's uniform 31.5.
#
# Screen acceptance (hard-gated):
#   10/10 wins AND 0 NF AND mean lift >= +220k AND chatter median <= 0.08
#   AND <= 1 negative seed.
# Kill: any NF, lift < +150k, chatter > 0.25, > 3 negatives, actL2 > 1.5
# (bandwidth drift or unstable parent-selection fallback).
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-k2-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier K2 stochastic-composition bench (1.5M oak + 1.5M random, 10 seeds)..."

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
