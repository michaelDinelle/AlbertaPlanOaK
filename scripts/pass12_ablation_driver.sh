#!/bin/bash
# Pass 12 Tier 2: per-kernel ablation driver.
# Runs the bench at full scope with each kernel in turn ablated, plus a baseline run with no
# ablation. Writes results to .oak-benchmarks/pass12-ablation/{baseline, <kernel>}/ for the
# multi-signal report (scripts/pass12_multi_signal_report.py --baseline=...) to consume.
#
# Per the multi-signal Pareto Hard Rule (AGENTS.md), a kernel is removed in Tier 3 iff its
# ablation does NOT hurt at p < 0.05 on at least one of the four agent-side channels
# (rewardRate, LPM, utilFeat, pEff). agent_ms is the implicit fifth channel — keeping a
# kernel COSTS FLOPs.
#
# Usage:
#   scripts/pass12_ablation_driver.sh                           # full bench scope
#   OAK_BENCH_STEPS=20000 OAK_BENCH_SEED_COUNT=3 \
#       scripts/pass12_ablation_driver.sh                       # quick-iter scope
#   scripts/pass12_ablation_driver.sh oak_meta_policy oak_learn_omega   # subset of kernels
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${OAK_BINARY:-$ROOT/.build/release/OaK}"
[ -x "$BIN" ] || { echo "Missing release binary at $BIN. Run swift build -c release first." >&2; exit 1; }

# Pass-11 18 kernels (post-Tier-3 phase ordering). If subset args given, use those instead.
ALL_KERNELS=(
    oak_encode oak_predict_outcome oak_meta_policy oak_predict oak_plan
    oak_reduce_alpha oak_learn_value oak_learn_encoder oak_learn_outcome
    oak_learn_psi oak_learn_pi oak_learn_beta oak_learn_omega oak_meta_action
    oak_bk_reduce oak_bk_feature oak_bk_option oak_bk_admit_commit
)
if [ $# -gt 0 ]; then
    KERNELS=("$@")
else
    KERNELS=("${ALL_KERNELS[@]}")
fi

BASE_DIR="$ROOT/.oak-benchmarks/pass12-ablation"
mkdir -p "$BASE_DIR"

# Default scope mirrors AGENTS.md Pareto Hard Rule: 84 envs × ≥10 seeds × ≥50K steps × 2048 envs.
export OAK_BENCH_STEPS="${OAK_BENCH_STEPS:-50000}"
export OAK_BENCH_SEED_COUNT="${OAK_BENCH_SEED_COUNT:-10}"
export OAK_BENCH_NUM_ENVS="${OAK_BENCH_NUM_ENVS:-2048}"
# Only oak runs are affected by ablation; random is the same baseline. Skip random in the
# ablation runs to halve compute (we re-use the baseline's random results in the report).
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

# 1. Baseline: no kernel ablated.
run_one "baseline" env

# 2. One bench per ablated kernel.
for k in "${KERNELS[@]}"; do
    run_one "$k" env "OAK_ABLATE_KERNEL=$k"
done

echo "All ablations complete. Generate the report with:" >&2
echo "  python3 scripts/pass12_multi_signal_report.py $BASE_DIR/baseline --output=docs/pass-12-bench-report.md" >&2
echo "  for k in ${KERNELS[*]}; do" >&2
echo "    python3 scripts/pass12_multi_signal_report.py $BASE_DIR/\$k --baseline=$BASE_DIR/baseline --output=docs/pass-12-ablation-\$k.md" >&2
echo "  done" >&2
