#!/usr/bin/env bash
# Tier D2 paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps. Validates
# Plan Step 9 (Moore-Atkeson 1993 prioritized sweeping) on the C1 substrate. The change
# adds a chord-weighted L2 norm of each option's model TD error as the per-option priority
# signal (computed in-line in oak_plan, fused with the existing tg_nHatCurr barrier),
# meanPrio computed once per step in oak_plan_reduce via simd_sum and broadcast-cached
# in KernelScalars, and oak_learn_heads Head 4 gated by `prioOpt[k] >= meanPrio` so only
# above-mean-priority options refresh their dense optTransition this step. ~half of K
# options skip per step on average → ~65k FMA saved per step out of K·R² = 131k.
#
# Acceptance vs C1 baseline (+111k/seed @ 1.5M): 10/10 wins, mean lift >= +90k, 0 NF,
# chatter median <= 0.30, walking touches >= 4/10, agentMs <= 0.32 ms.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-d2-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier D2 bench (1.5M oak + 1.5M random, 10 seeds)..."

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
