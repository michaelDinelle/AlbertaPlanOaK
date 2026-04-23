#!/usr/bin/env bash
# Tier K1 (Plan Step 2 "feature combinations") deep paired bench:
# HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
#
# KEY signals to watch:
#   PRIMARY (the exciting one): chordEffectiveK median < 30 — would confirm
#     Plan Step 2 compositional features break uniform-chord symmetry observed
#     across H1 10M (30.86) and I1 10M (31.42). If effK < 20, H1 hysteresis
#     meaningfully active (compound tier win — K1 retroactively unlocks H1).
#   SECONDARY: mean lift >= F1 +1.67M (parity or better).
#   TERTIARY: >= 5 of 10 seeds beat F1; chatter median <= 0.08; actL2 stationary.
#
# Acceptance: 10/10 wins AND 0 NF AND mean lift >= F1 +1.67M AND chatter <= 0.08
#   AND >= 5 seeds beat F1 AND chordEffectiveK median < 30.
#
# Kill: any NF, mean lift < F1 −10%, chatter > 0.20, < 3 seeds beat F1,
#   actL2 monotonic growth (bandwidth not stationary).
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
export OAK_WARMUP_STEPS=1000
export OAK_HEARTBEAT_STEPS=200000

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-k1-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier K1 feature-combinations 10M bench (10M oak + 10M random, 10 seeds)..."
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
