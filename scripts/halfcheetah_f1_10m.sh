#!/usr/bin/env bash
# Tier F1 deep paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
# Validates whether F1 structural σ = 1/√A breaks the +0.3 max peak / +0.05 median
# peak plateau that held across C1 / D2 / D3 / C4 10M characterizations.
#
# Current agent stack:
#   C1 encoder (obsTrace, Plan Step 3)
#   Tier B RR-Subtasks ψ-GVFs + dense optTransition + MetaOptimize per-feature α
#   Tier D2 Head 4 prioritized sweeping (Moore-Atkeson 1993, Plan Step 9)
#   Tier D3 Head 1 chord-weighted ψ updates (Plan Step 10 off-policy attribution)
#   Tier F1 structural policy σ = 1/√A (plateau-breaker, Williams 1992 σ-parameterized)
#   PERFOPT + PERFOPT2 kernel optimizations (bit-id, +64% SPS vs C1)
#
# Distributional acceptance (per D3 methodology):
#   10/10 wins AND 0 NF AND median per-seed max peak ≥ +0.10 (D3: +0.047)
#   AND walking coverage ≥ 6% of HBs>0 (D3: 3.2%) AND mean lift ≥ +600k (D3: +620k).
#
# Kill: any NF, mean lift < +400k, median peak < +0.030 (below D3), walking < 3%.
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
export OAK_HEARTBEAT_STEPS=200000

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-f1-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier F1 10M bench (10M oak + 10M random, 10 seeds)..."
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
