#!/usr/bin/env bash
# Tier D3 deep paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
# Validates whether D3's 1.5M specialization signature (chatter -49%, walking breadth up,
# aggregate lift within noise, max peak lower at 1.5M) compounds into real peak depth at
# 10M. The 1.5M peak is known to be unreliable (D2 methodology lesson); real verdict is
# at 10M scale.
#
# Current agent stack:
#   C1 encoder (std_obs ‖ obsTrace Fourier features, Plan Step 3)
#   Tier B dense optTransition + RR-Subtasks + MetaOptimize per-feature α
#   Tier D2 prioritized sweeping (Moore-Atkeson 1993, Plan footnote 13)
#   Tier D3 chord-weighted Head 1 psi updates (Plan Step 10 off-policy attribution)
#   PERFOPT + PERFOPT2 kernel optimizations (bit-id with D2 algorithm)
#
# Acceptance at 10M: max peak > +0.35 (above C1's +0.310 and D2's +0.135) AND mean
# lift >= +600k AND 10/10 wins AND 0 NF.
# Kill: max peak < +0.25 OR mean lift < +500k.
#
# ETA ~1h10m wall at PERFOPT2's 2534 sps/stream concurrent throughput.
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
export OAK_HEARTBEAT_STEPS=200000   # 50 heartbeats over 10M (matches D2 10M + C1 10M cadence)

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-d3-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier D3 10M bench (10M oak + 10M random, 10 seeds)..."
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
