#!/usr/bin/env bash
# Tier J1 (Plan Step 11 "eventually" literal: G&T mean-utility gate) deep paired bench:
# HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
#
# KEY signals to watch:
#   PRIMARY: oakOptionReinitCountFinal and oakFeatureReinitCountFinal drop << stepCount.
#   SECONDARY: chordEffectiveK median < 30 (chord concentrates → H1 activates retroactively).
#   TERTIARY: mean lift ≥ F1 +1.67M (parity; performance gain hoped-for but not required).
#
# Acceptance: 10/10 wins AND 0 NF AND mean lift ≥ F1's +1.67M (parity) AND chatter median ≤ 0.08
#   AND ≥ 5 of 10 seeds beat F1 counterpart AND reinit counts data-driven.
#
# Kill: any NF, mean lift < F1 −10%, chatter > 0.20, < 3 seeds beat F1,
#   reinit == stepCount (gate inactive) or reinit == 0 (gate frozen).
#
# ETA ~75min wall at I1 substrate SPS.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-j1-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier J1 mean-utility-gate 10M bench (10M oak + 10M random, 10 seeds)..."
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
