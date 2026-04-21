#!/usr/bin/env bash
# Tier C4 deep paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
# Validates Plan Step 10 STOMP completion — structural termination from ψ:
#
#   β_k(s') = clamp((1 + curr[k]) / 2, 0, 1)     [replaces utilOpt-based stability cap]
#   δ_k     = (r − ρ̄) + (1 − β_k)·curr[k] + β_k·ψ_k(s') − ψ_k(s)    [RR-Subtasks eq. 17]
#
# Distributional acceptance (per post-D3 methodology lesson — max peak is outlier-
# sensitive, use distributional metrics):
#   10/10 wins AND 0 NF AND median per-seed max peak ≥ +0.07 AND walking coverage
#   ≥ 4.5% of HBs>0 AND mean lift ≥ +550k/seed
# Baselines for comparison:
#   D3 10M: median peak +0.047, walking coverage 3.2%, mean lift +620k
#   C1 10M: median peak +0.026, walking coverage 1.2%, mean lift +668k
#
# Kill: any NF, mean lift < +400k, median peak < +0.020, walking coverage < 2%.
#
# Current agent stack:
#   C1 encoder (obsTrace, Plan Step 3)
#   Tier B RR-Subtasks ψ-GVFs + dense optTransition + MetaOptimize per-feature α
#   Tier D2 Head 4 prioritized sweeping (Moore-Atkeson 1993, Plan Step 9)
#   Tier D3 Head 1 chord-weighted ψ updates (Plan Step 10 off-policy attribution)
#   Tier C4 structural β_k(s') = (1+curr[k])/2 (Plan Step 10 STOMP termination)
#   PERFOPT + PERFOPT2 kernel optimizations (bit-id, +64% SPS vs C1)
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-c4-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier C4 10M bench (10M oak + 10M random, 10 seeds)..."
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
