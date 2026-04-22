#!/usr/bin/env bash
# Tier H1 (Plan Step 10 STOMP multi-step options via chord hysteresis) paired bench:
# HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
#
# β^k = 1/(1 + ntdSqSum[k]) (optTransition confidence) drives chord hysteresis:
#   chord_k ← β^k · chordPrev[k] + (1 − β^k) · softmax_k(ψ·s'); renormalize via simd_sum.
# Sidesteps the ψ-coupled termination family exhausted by C2/C4/Step-11 (β here derives
# from forward-model-prediction error, a different signal class).
#
# Screen acceptance (hard-gated):
#   10/10 wins AND 0 NF AND mean lift ≥ +220k (F1 baseline +237k; parity-or-better)
#   AND chatter median ≤ 0.08 AND ≤ 1 negative seed
#   AND chordPersistenceMeanAbs median > 0.3 (validates multi-step options emerge).
# Kill: any NF, lift < +150k, chatter > 0.25, > 3 negative seeds, persistence < 0.1.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-h1-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier H1 chord-hysteresis bench (1.5M oak + 1.5M random, 10 seeds)..."

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
