#!/usr/bin/env bash
# Tier H1 (Plan Step 10 STOMP) deep paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 10M steps.
# THE Plan-Step-10-orthodoxy test — does chord hysteresis via model-confidence β give
# multi-step options without regressing F1's +1.67M mean lift baseline?
#
# Acceptance (structural orthodoxy win; mean-lift gain hoped-for but not required):
#   10/10 wins AND 0 NF AND mean lift ≥ F1's +1.67M (parity) AND chatter median ≤ 0.08
#   AND ≥ 5 of 10 seeds beat F1 counterpart
#   AND chordPersistenceMeanAbs median in [0.3, 0.9] (genuinely multi-step, not over-sticky).
#
# Kill: any NF, mean lift < F1 −10% (major regression), chatter > 0.20, < 3 seeds beat F1,
#   persistence < 0.15 or > 0.99.
#
# ETA ~1h10m wall at PERFOPT2's ~2500 SPS/stream.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-h1-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier H1 chord-hysteresis 10M bench (10M oak + 10M random, 10 seeds)..."
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
