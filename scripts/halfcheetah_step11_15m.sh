#!/usr/bin/env bash
# Step 11 sharpening paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
# Validates Plan Step 11 "useful in planning" literal — G&T reseeding picks argmin planUtil
# instead of argmin utilOpt. planUtil[k] = EMA(Q^k(s) − V(s)) is computed per-option in
# oak_plan_reduce. Options with consistently low Q^k − V get reseeded by G&T; options with
# consistently high Q^k − V (planning-critical) are kept.
#
# Screen acceptance (hard-gated): 10/10 wins AND 0 NF AND mean lift ≥ +200k AND
# chatter ≤ 0.10 AND ≤3 negative seeds.
# Kill: any NF, lift < +100k, chatter > 0.25, >4 negative seeds.
# If screen passes, proceed to 10×10M paired bench (the plateau-past-2M test).
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-step11-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Step 11 sharpening bench (1.5M oak + 1.5M random, 10 seeds)..."

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
