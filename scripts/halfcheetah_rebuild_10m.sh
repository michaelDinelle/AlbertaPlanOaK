#!/usr/bin/env bash
# Step 1-11 rebuild deep bench: HalfCheetah-v4, 10 seeds x {oak, random}.
# Rebuilds Metal explicitly so metrics cannot come from a stale metallib.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/.build"
xcrun -sdk macosx metal -fno-fast-math -c Sources/OakKernel.metal -o .build/OakKernel.air
xcrun -sdk macosx metallib .build/OakKernel.air -o .build/OakKernel.metallib
swift build -c release

export OAK_ENV="HalfCheetah-v4"
export OAK_NUM_ENVS=1
export OAK_HEARTBEAT_STEPS="${OAK_HEARTBEAT_STEPS:-200000}"

STEPS="${OAK_REBUILD_STEPS:-10000000}"
RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-rebuild-10m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Step 1-11 rebuild bench (${STEPS} oak + ${STEPS} random, 10 seeds)..."
echo "  start: $(date -u +%FT%TZ)"

for seed in {0..9}; do
  echo "Starting seed $seed random..."
  OAK_METRICS_STORAGE="$RUN_DIR/seed${seed}-random" \
  OAK_SEED=$seed \
  OAK_POLICY=random \
  OAK_STOP_AFTER_STEPS="$STEPS" \
  ./.build/release/OaK run "$OAK_ENV" > /dev/null 2>&1 &

  echo "Starting seed $seed oak..."
  OAK_METRICS_STORAGE="$RUN_DIR/seed${seed}-oak" \
  OAK_SEED=$seed \
  OAK_POLICY=oak \
  OAK_STOP_AFTER_STEPS="$STEPS" \
  ./.build/release/OaK run "$OAK_ENV" > /dev/null 2>&1 &
done

echo "All 20 processes are running in the background."
wait
echo "  end:   $(date -u +%FT%TZ)"
echo "All done."
