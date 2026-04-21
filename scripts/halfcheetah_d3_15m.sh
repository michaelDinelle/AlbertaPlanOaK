#!/usr/bin/env bash
# Tier D3 paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
# Validates Plan Step 10 off-policy attribution — Head 1 (psi) now chord-weighted:
#
#   Δψ[k, i] = α[i] · tdPsi[k] · prev[i] · max(chordPrev[k], 1/K)
#
# Before D3, psi was the only head without chord weighting, so all K ψ-GVFs
# converged to the chord's value rather than per-option values. With chord-
# weighted updates, option k's psi specializes to what happens when k is active.
# The 1/K floor is a structural minimum that prevents self-reinforcing starvation
# when chord[k]→0. K is the Plan-defined option count (host-capped at simdgroup
# width 32), so 1/K is not a tunable.
#
# Motivation: D2's 10M signature was "more walking breadth, less walking depth"
# (11 heartbeats > 0 vs C1's 6; max peak +0.135 vs C1's +0.310). D3 addresses
# the depth side directly — options that find walking can lock in their psi
# aggressively without cross-option interference.
#
# Acceptance vs PERFOPT2 baseline: 10/10 wins, mean lift ≥ +90k, 0 NF,
# chatter ≤ 0.30, max peak ≥ +0.17, walking ≥ 4/10, no chord collapse.
#
# Uses `OaK run` — hot-path mode, no per-kernel waitUntilCompleted overhead.
# ETA ~12-15 min at PERFOPT2's 2534 sps/stream concurrent throughput.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-d3-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier D3 bench (1.5M oak + 1.5M random, 10 seeds)..."

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
