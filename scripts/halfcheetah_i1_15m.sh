#!/usr/bin/env bash
# Tier I1 (Plan Step 3 partial: multi-timescale obsTrace spectrum) paired bench:
# HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
#
# Replaces Tier-C1's single obsTrace (rate 1/baseO) with M = floor(log2(baseO))+1 trace
# levels at structural rates 1/2^m. HC baseO=17 → M=5 → 4 stored trace levels at
# τ ∈ {2, 4, 8, 16}. Encoder input grows from 2*baseO=34 to M*baseO=85 dims per
# pre-activation. Plan Step 3 literal "incorporation of older signals and traces".
#
# Screen acceptance (hard-gated):
#   10/10 wins AND 0 NF AND mean lift ≥ +220k (F1 baseline +237k; parity-or-better)
#   AND chatter median ≤ 0.08 AND ≤ 1 negative seed.
# Kill: any NF, lift < +150k, chatter > 0.25, > 3 negative seeds.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-i1-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier I1 multi-timescale-traces bench (1.5M oak + 1.5M random, 10 seeds)..."

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
