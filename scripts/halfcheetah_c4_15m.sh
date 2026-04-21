#!/usr/bin/env bash
# Tier C4 paired bench: HalfCheetah-v4, 10 seeds × {oak, random}, 1.5M steps.
# Validates Plan Step 10 STOMP completion — structural termination from ψ:
#
#   β_k(s') = clamp((1 + curr[k]) / 2, 0, 1)     [replaces utilOpt-based stability cap]
#   δ_k     = (r − ρ̄) + (1 − β_k)·curr[k] + β_k·ψ_k(s') − ψ_k(s)    [unchanged, RR-Subtasks eq. 17]
#
# curr[k] is a Fourier feature (cos/sin) ∈ [-1, 1], so β ∈ [0, 1] with no learnable
# parameter. When feature_k is achieved (curr[k]→+1), β→1, option terminates and TD
# bootstraps from ψ(s'). When feature_k anti-achieved (curr[k]→-1), β→0, option
# continues and TD uses curr[k] as cumulant.
#
# The three prior C2 attempts (v1 β₀=1/R, v2 β₀=0.5, retry-on-C1) all regressed ~99%
# because they compounded three new mechanisms: free termW parameter, closed-form
# Q^k = (r̂-ρ̄)/β + V, hold-or-refresh chord Bernoulli. C4 adds NONE of those — just
# replaces the β_k formula in the existing TD slot. Builds on D3's chord-weighted
# Head 1 which already provides per-option ψ specialization.
#
# 1.5M screen acceptance (hard-gated): 10/10 wins AND 0 NF AND mean lift ≥ +90k
# AND chatter ≤ 0.30 AND ≤4 negative seeds. Kill on any NF, lift < +40k, chatter
# > 0.45. If screen passes, proceed to 10x10M bench.
#
# Uses `OaK run` — hot-path mode. ETA ~12-15 min at PERFOPT2's 2534 SPS/stream.
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

RUN_DIR="$ROOT/.oak-benchmarks/halfcheetah-c4-15m"
mkdir -p "$RUN_DIR"
echo "Launching HalfCheetah-v4 Tier C4 bench (1.5M oak + 1.5M random, 10 seeds)..."

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
