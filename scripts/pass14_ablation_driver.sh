#!/bin/bash
# Pass 14: per-tier ablation driver. Hard-rule infrastructure for the >=2-of-5-channels
# strict gate. Replaces Pass 12's lenient >=1-channel rule.
#
# Per-tier usage (touched kernels + canary set):
#   scripts/pass14_ablation_driver.sh \
#       --tier=0 \
#       --kernels=oak_learn_encoder_spr,oak_meta_action,oak_encode,oak_predict_outcome,\
#                 oak_meta_policy,oak_predict,oak_plan
#
# Final-pass usage (all 20 kernels):
#   scripts/pass14_ablation_driver.sh --tier=7 --full
#
# Outputs to .oak-benchmarks/pass14-tier${TIER}-ablation/{baseline, <kernel>}/ for the
# per-kernel decision script (scripts/pass14_kernel_decision.py) to consume.
#
# Default scope: 84 envs × 10 seeds × 50K steps × 2048 envs (matches per-tier strict gate).
# Override via OAK_BENCH_STEPS / OAK_BENCH_SEED_COUNT / OAK_BENCH_NUM_ENVS.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${OAK_BINARY:-$ROOT/.build/release/OaK}"
[ -x "$BIN" ] || { echo "Missing release binary at $BIN. Run swift build -c release first." >&2; exit 1; }

# Pass 13 / Pass 14 default 20-kernel pipeline. Updated by tiers (Tier 1 fuses to 19; Tier 3
# adds EVE; Tier 6 adds oak_sample_actions; etc.). Use --kernels for per-tier subset.
ALL_KERNELS=(
    oak_encode oak_predict_outcome oak_meta_policy oak_predict_next_latent oak_predict
    oak_plan oak_reduce_alpha oak_learn_value oak_learn_encoder oak_learn_encoder_spr
    oak_learn_outcome oak_learn_psi oak_learn_pi oak_learn_beta oak_learn_omega
    oak_meta_action oak_bk_reduce oak_bk_feature oak_bk_option oak_bk_admit_commit
)

TIER=""
KERNELS_ARG=""
FULL=0
SKIP_BASELINE=0
for arg in "$@"; do
    case "$arg" in
        --tier=*) TIER="${arg#--tier=}" ;;
        --kernels=*) KERNELS_ARG="${arg#--kernels=}" ;;
        --full) FULL=1 ;;
        --skip-baseline) SKIP_BASELINE=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [ -z "$TIER" ]; then
    echo "missing --tier=<N>" >&2
    exit 2
fi

if [ "$FULL" = 1 ]; then
    KERNELS=("${ALL_KERNELS[@]}")
elif [ -n "$KERNELS_ARG" ]; then
    IFS=',' read -ra KERNELS <<< "$KERNELS_ARG"
else
    echo "must specify --kernels=K1,K2,... or --full" >&2
    exit 2
fi

BASE_DIR="$ROOT/.oak-benchmarks/pass14-tier${TIER}-ablation"
mkdir -p "$BASE_DIR"

# Default scope: full 10-seed × 50K × 2048 envs per the per-tier strict gate. Pass 12 used
# 2 seeds (lenient); Pass 14 uses 10 (proper statistical power).
export OAK_BENCH_STEPS="${OAK_BENCH_STEPS:-50000}"
export OAK_BENCH_SEED_COUNT="${OAK_BENCH_SEED_COUNT:-10}"
export OAK_BENCH_NUM_ENVS="${OAK_BENCH_NUM_ENVS:-2048}"
export OS_ACTIVITY_MODE=disable

run_one() {
    local label="$1"; shift
    local out_dir="$BASE_DIR/$label"
    if [ -d "$out_dir" ] && [ -n "$(ls -A "$out_dir" 2>/dev/null)" ]; then
        echo "[$label] already populated; skip (delete $out_dir to re-run)" >&2
        return 0
    fi
    mkdir -p "$out_dir"
    echo "[$label] starting bench: steps=$OAK_BENCH_STEPS seeds=$OAK_BENCH_SEED_COUNT envs=$OAK_BENCH_NUM_ENVS" >&2
    local t0; t0=$(date +%s)
    OAK_BENCH_DIR="$out_dir" "$@" "$BIN" benchmark > "$out_dir/_driver.log" 2>&1
    local t1; t1=$(date +%s)
    echo "[$label] done in $((t1 - t0))s — results in $out_dir" >&2
}

# 1. Baseline: no kernel ablated, BOTH oak + random (random is the env-vs-random comparison
# baseline for the strict report). Skip with --skip-baseline if you already have a fresh
# main bench from the per-tier acceptance run.
if [ "$SKIP_BASELINE" = 0 ]; then
    run_one "baseline" env
fi

# 2. One bench per ablated kernel (oak ONLY — kernel decision compares oak-ablated to
# oak-baseline; random is unaffected by kernel ablation, halve compute by skipping it).
for k in "${KERNELS[@]}"; do
    run_one "$k" env "OAK_ABLATE_KERNEL=$k" "OAK_BENCH_POLICIES=oak"
done

echo "" >&2
echo "All Tier-$TIER ablations complete. Decide kernel keep/remove with:" >&2
echo "  python3 scripts/pass14_kernel_decision.py \\" >&2
echo "    --ablation-dir=$BASE_DIR \\" >&2
echo "    --baseline-dir=$BASE_DIR/baseline \\" >&2
echo "    --output=docs/pass-14-tier${TIER}-ablation.md" >&2
