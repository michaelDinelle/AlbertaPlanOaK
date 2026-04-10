#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_ENVS=(pyflyt cheetah walker humanoid minigrid)
VALID_ENVS=(pyflyt carla cheetah walker humanoid minigrid)
POLICIES=(oak random)
MODE="${OAK_BENCH_MODE:-quality}"
SEED_START="${OAK_BENCH_SEED_START:-0}"
SEED_COUNT="${OAK_BENCH_SEED_COUNT:-10}"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
BENCH_DIR="${OAK_BENCH_DIR:-$ROOT_DIR/.oak-benchmarks/$TIMESTAMP}"
RUNS_DIR="$BENCH_DIR/runs"
MANIFEST_PATH="$BENCH_DIR/manifest.tsv"
SUMMARY_JSON="$BENCH_DIR/summary.json"
SUMMARY_MD="$BENCH_DIR/summary.md"

usage() {
    cat <<'EOF'
Usage: ./benchmark.sh [env ...]

Runs seeded OaK-vs-random benchmarks and writes artifacts to .oak-benchmarks/.

Defaults:
  envs: pyflyt cheetah walker humanoid minigrid
  mode: quality
  seeds: 0..9

Environment variables:
  OAK_BENCH_MODE=quality|throughput|both
  OAK_BENCH_SEED_START=<int>
  OAK_BENCH_SEED_COUNT=<int>
  OAK_BENCH_DIR=<path>
  OAK_BENCH_SKIP_BUILD=1
  OAK_BENCH_WARMUP_STEPS=<int>
  OAK_BENCH_QUALITY_STEPS=<int>
  OAK_BENCH_THROUGHPUT_SECONDS=<float>

Notes:
  - CARLA is benchmarked only when you pass `carla` explicitly.
  - CARLA autostart is forced off unless you override OAK_CARLA_AUTOSTART.
EOF
}

is_valid_env() {
    local candidate="$1"
    local env
    for env in "${VALID_ENVS[@]}"; do
        if [ "$env" = "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

DEFAULT_WARMUP_STEPS="${OAK_BENCH_DEFAULT_WARMUP_STEPS:-1000}"
DEFAULT_QUALITY_STEPS="${OAK_BENCH_DEFAULT_QUALITY_STEPS:-10000}"
DEFAULT_THROUGHPUT_SECONDS="${OAK_BENCH_DEFAULT_THROUGHPUT_SECONDS:-10}"

validate_positive_integer() {
    local label="$1"
    local value="$2"
    case "$value" in
        ''|*[!0-9]*)
            printf '%s must be a positive integer, got %s\n' "$label" "$value" >&2
            exit 1
            ;;
        *)
            if [ "$value" -le 0 ]; then
                printf '%s must be a positive integer, got %s\n' "$label" "$value" >&2
                exit 1
            fi
            ;;
    esac
}

validate_nonnegative_integer() {
    local label="$1"
    local value="$2"
    case "$value" in
        ''|*[!0-9]*)
            printf '%s must be a non-negative integer, got %s\n' "$label" "$value" >&2
            exit 1
            ;;
        *)
            ;;
    esac
}

build_if_needed() {
    if [ "${OAK_BENCH_SKIP_BUILD:-0}" = "1" ]; then
        return
    fi

    (
        cd "$ROOT_DIR"
        swift build -c release
    )
}

bench_requires_python() {
    local env_name
    for env_name in "${ENVS[@]}"; do
        case "$env_name" in
            carla)
                ;;
            *)
                return 0
                ;;
        esac
    done
    return 1
}

run_doctor_gates() {
    if [ "${OAK_BENCH_SKIP_DOCTOR:-0}" = "1" ]; then
        return
    fi

    if [ "${OAK_ALLOW_CPU_FALLBACK:-0}" = "1" ]; then
        printf 'Normal benchmark mode requires Metal. Unset OAK_ALLOW_CPU_FALLBACK or set OAK_BENCH_SKIP_DOCTOR=1 for diagnosis-only runs.\n' >&2
        exit 1
    fi

    (
        cd "$ROOT_DIR"
        OAK_DOCTOR_SKIP_BUILD=1 ./doctor.sh metal
    )

    if bench_requires_python; then
        (
            cd "$ROOT_DIR"
            OAK_DOCTOR_SKIP_BUILD=1 ./doctor.sh python
        )
    fi
}

run_one() {
    local mode="$1"
    local env="$2"
    local policy="$3"
    local seed="$4"
    local warmup_steps="$5"
    local stop_steps="$6"
    local stop_seconds="$7"
    local metrics_path="$RUNS_DIR/${env}-${policy}-${mode}-seed${seed}.jsonl"
    local log_path="$RUNS_DIR/${env}-${policy}-${mode}-seed${seed}.log"

    printf '[bench] mode=%s env=%s policy=%s seed=%s warmup=%s stop_steps=%s stop_seconds=%s\n' \
        "$mode" "$env" "$policy" "$seed" "$warmup_steps" "${stop_steps:-na}" "${stop_seconds:-na}"

    if ! (
        cd "$ROOT_DIR"
        OAK_ALLOW_CPU_FALLBACK=0 \
        OAK_NUM_ENVS="${OAK_BENCH_NUM_ENVS:-8}" \
        OAK_POLICY="$policy" \
        OAK_SEED="$seed" \
        OAK_WARMUP_STEPS="$warmup_steps" \
        OAK_STOP_AFTER_STEPS="$stop_steps" \
        OAK_STOP_AFTER_SECONDS="$stop_seconds" \
        OAK_METRICS_JSONL_PATH="$metrics_path" \
        OAK_METRICS_STEP_RECORDS=0 \
        OAK_METRICS_SUMMARY_RECORDS=0 \
        OAK_METRICS_STDOUT_SUMMARIES=0 \
        OAK_CARLA_AUTOSTART="${OAK_CARLA_AUTOSTART:-0}" \
        ./run.sh "$env" >"$log_path" 2>&1
    ); then
        printf '[bench] run failed; see %s\n' "$log_path" >&2
        exit 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$env" "$policy" "$seed" "$warmup_steps" "$stop_steps" "$stop_seconds" "$metrics_path" "$log_path" >>"$MANIFEST_PATH"
}

mkdir -p "$RUNS_DIR"
printf 'mode\tenv\tpolicy\tseed\twarmup_steps\tstop_steps\tstop_seconds\tmetrics_path\tlog_path\n' >"$MANIFEST_PATH"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

ENVS=()
if [ "$#" -gt 0 ]; then
    for requested_env in "$@"; do
        if [ "$requested_env" = "all" ]; then
            ENVS+=("${DEFAULT_ENVS[@]}")
            continue
        fi

        if ! is_valid_env "$requested_env"; then
            printf 'Unknown environment: %s\n' "$requested_env" >&2
            usage >&2
            exit 1
        fi

        ENVS+=("$requested_env")
    done
else
    ENVS=("${DEFAULT_ENVS[@]}")
fi

MODES=()
case "$MODE" in
    quality)
        MODES=(quality)
        ;;
    throughput)
        MODES=(throughput)
        ;;
    both)
        MODES=(quality throughput)
        ;;
    *)
        printf 'Unknown OAK_BENCH_MODE: %s\n' "$MODE" >&2
        exit 1
        ;;
esac

validate_positive_integer "OAK_BENCH_SEED_COUNT" "$SEED_COUNT"

printf '[bench] output_dir=%s\n' "$BENCH_DIR"
build_if_needed
run_doctor_gates

pids=()
for mode in "${MODES[@]}"; do
    for env in "${ENVS[@]}"; do
        warmup_steps="${OAK_BENCH_WARMUP_STEPS:-$DEFAULT_WARMUP_STEPS}"
        validate_nonnegative_integer "warmup steps for $env" "$warmup_steps"

        stop_steps=""
        stop_seconds=""
        if [ "$mode" = "quality" ]; then
            stop_steps="${OAK_BENCH_QUALITY_STEPS:-$DEFAULT_QUALITY_STEPS}"
            validate_positive_integer "quality steps for $env" "$stop_steps"
        else
            stop_seconds="${OAK_BENCH_THROUGHPUT_SECONDS:-$DEFAULT_THROUGHPUT_SECONDS}"
        fi

        seed="$SEED_START"
        seed_limit=$((SEED_START + SEED_COUNT))
        pids=()
        while [ "$seed" -lt "$seed_limit" ]; do
            for policy in "${POLICIES[@]}"; do
                run_one "$mode" "$env" "$policy" "$seed" "$warmup_steps" "$stop_steps" "$stop_seconds" &
                pids+=($!)
            done
            seed=$((seed + 1))
        done
        
        # Wait for all seeds and policies for this environment to finish before moving to the next
        if [ ${#pids[@]} -gt 0 ]; then
            for pid in "${pids[@]}"; do
                wait "$pid" || true
            done
        fi
        pids=()
    done
done

python3 - "$BENCH_DIR" "$MANIFEST_PATH" "$SUMMARY_JSON" "$SUMMARY_MD" <<'PY'
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

bench_dir, manifest_path, summary_json_path, summary_md_path = sys.argv[1:5]
t_critical_95_df9 = 2.262157


def summarize(values):
    values = list(values)
    if not values:
        return None
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    ci95 = t_critical_95_df9 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "median": median,
        "min": min(values),
        "max": max(values),
        "ci95": ci95,
    }


def read_run_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise RuntimeError(f"metrics file is empty: {path}")
    run_start = next((record for record in records if record.get("type") == "run_start"), None)
    run_end = next((record for record in reversed(records) if record.get("type") == "run_end"), None)
    if run_start is None or run_end is None:
        raise RuntimeError(f"metrics file is missing run_start/run_end: {path}")
    return run_start, run_end


def format_stat(stat, digits=5):
    if stat is None:
        return "n/a"
    return f"{stat['mean']:.{digits}f} +/- {stat['std']:.{digits}f}"


def format_delta(stat, digits=5):
    if stat is None:
        return "n/a"
    return f"{stat['mean']:.{digits}f} +/- {stat['ci95']:.{digits}f}"


runs = []
with open(manifest_path, "r", encoding="utf-8") as manifest_handle:
    reader = csv.DictReader(manifest_handle, delimiter="\t")
    for row in reader:
        run_start, run_end = read_run_records(row["metrics_path"])
        runs.append({
            "mode": row["mode"],
            "env": row["env"],
            "policy": row["policy"],
            "seed": int(row["seed"]),
            "metrics_path": row["metrics_path"],
            "log_path": row["log_path"],
            "warmup_steps": int(row["warmup_steps"]),
            "configured_stop_steps": int(row["stop_steps"]) if row["stop_steps"] else None,
            "configured_stop_seconds": float(row["stop_seconds"]) if row["stop_seconds"] else None,
            "run_id": run_end.get("runId"),
            "reward_sum": float(run_end.get("rewardSum", 0.0)),
            "reward_mean": float(run_end.get("rewardMean", 0.0)),
            "reward_std": float(run_end.get("rewardStd", 0.0)),
            "done_count": int(run_end.get("doneCount", 0)),
            "reset_count": int(run_end.get("resetCount", 0)),
            "sps": float(run_end.get("stepsPerSecond", 0.0)),
            "agent_ms": float(run_end.get("agentStepMsMean", 0.0)),
            "env_ms": float(run_end.get("envStepMsMean", 0.0)),
            "total_ms": float(run_end.get("totalStepMsMean", 0.0)),
            "action_l2": float(run_end.get("actionL2Mean", 0.0)),
            "state_l2": float(run_end.get("stateL2Mean", 0.0)),
            "measured_steps": int(run_end.get("measuredStepCount", run_end.get("step", 0))),
            "measured_seconds": float(run_end.get("measuredSeconds", 0.0)),
            "stop_reason": run_end.get("stopReason"),
            "run_start": run_start,
            "run_end": run_end,
        })

grouped_runs = defaultdict(list)
for run in runs:
    grouped_runs[(run["mode"], run["env"], run["policy"])].append(run)

summary = {
    "generatedAt": datetime.now(timezone.utc).isoformat(),
    "benchDir": bench_dir,
    "runs": len(runs),
    "groups": {},
    "comparisons": {},
}

for (mode, env, policy), entries in sorted(grouped_runs.items()):
    entries = sorted(entries, key=lambda entry: entry["seed"])
    policy_summary = {
        "n": len(entries),
        "seeds": [entry["seed"] for entry in entries],
        "warmupSteps": entries[0]["warmup_steps"] if entries else 0,
        "configuredStopSteps": entries[0]["configured_stop_steps"] if entries else None,
        "configuredStopSeconds": entries[0]["configured_stop_seconds"] if entries else None,
        "stopReasons": sorted({entry["stop_reason"] for entry in entries}),
        "rewardSum": summarize(entry["reward_sum"] for entry in entries),
        "rewardMean": summarize(entry["reward_mean"] for entry in entries),
        "rewardStd": summarize(entry["reward_std"] for entry in entries),
        "stepsPerSecond": summarize(entry["sps"] for entry in entries),
        "doneCount": summarize(entry["done_count"] for entry in entries),
        "resetCount": summarize(entry["reset_count"] for entry in entries),
        "measuredSteps": summarize(entry["measured_steps"] for entry in entries),
        "measuredSeconds": summarize(entry["measured_seconds"] for entry in entries),
        "agentStepMs": summarize(entry["agent_ms"] for entry in entries),
        "envStepMs": summarize(entry["env_ms"] for entry in entries),
        "totalStepMs": summarize(entry["total_ms"] for entry in entries),
        "actionL2": summarize(entry["action_l2"] for entry in entries),
        "stateL2": summarize(entry["state_l2"] for entry in entries),
        "runArtifacts": [
            {
                "seed": entry["seed"],
                "metricsPath": entry["metrics_path"],
                "logPath": entry["log_path"],
                "stopReason": entry["stop_reason"],
            }
            for entry in entries
        ],
    }
    summary["groups"].setdefault(mode, {}).setdefault(env, {})[policy] = policy_summary

for mode in sorted({run["mode"] for run in runs}):
    mode_runs = [run for run in runs if run["mode"] == mode]
    for env in sorted({run["env"] for run in mode_runs}):
        oak_runs = {run["seed"]: run for run in mode_runs if run["env"] == env and run["policy"] == "oak"}
        random_runs = {run["seed"]: run for run in mode_runs if run["env"] == env and run["policy"] == "random"}
        shared_seeds = sorted(set(oak_runs) & set(random_runs))
        if not shared_seeds:
            continue

        reward_sum_delta = [oak_runs[seed]["reward_sum"] - random_runs[seed]["reward_sum"] for seed in shared_seeds]
        reward_mean_delta = [oak_runs[seed]["reward_mean"] - random_runs[seed]["reward_mean"] for seed in shared_seeds]
        sps_delta = [oak_runs[seed]["sps"] - random_runs[seed]["sps"] for seed in shared_seeds]
        wins = sum(1 for seed in shared_seeds if oak_runs[seed]["reward_sum"] > random_runs[seed]["reward_sum"])
        losses = sum(1 for seed in shared_seeds if oak_runs[seed]["reward_sum"] < random_runs[seed]["reward_sum"])
        ties = len(shared_seeds) - wins - losses

        summary["comparisons"].setdefault(mode, {})[env] = {
            "pairedSeeds": shared_seeds,
            "rewardSumDelta": summarize(reward_sum_delta),
            "rewardMeanDelta": summarize(reward_mean_delta),
            "stepsPerSecondDelta": summarize(sps_delta),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }

with open(summary_json_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")

lines = [
    "# OaK Benchmark",
    "",
    f"- Generated: {summary['generatedAt']}",
    f"- Output dir: `{bench_dir}`",
    f"- Runs: {summary['runs']}",
]

for mode in sorted(summary["groups"]):
    lines.extend([
        "",
        f"## {mode.title()}",
        "",
        "| env | policy | n | reward_sum | reward_mean | sps | resets | measured_steps | stop |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for env in sorted(summary["groups"][mode]):
        for policy in ("oak", "random"):
            policy_summary = summary["groups"][mode][env].get(policy)
            if policy_summary is None:
                continue
            lines.append(
                f"| {env} | {policy} | {policy_summary['n']} | "
                f"{format_stat(policy_summary['rewardSum'])} | "
                f"{format_stat(policy_summary['rewardMean'])} | "
                f"{format_stat(policy_summary['stepsPerSecond'], digits=2)} | "
                f"{format_stat(policy_summary['resetCount'], digits=2)} | "
                f"{format_stat(policy_summary['measuredSteps'], digits=2)} | "
                f"{', '.join(policy_summary['stopReasons'])} |"
            )

    comparisons = summary["comparisons"].get(mode, {})
    if comparisons:
        lines.extend([
            "",
            f"### {mode.title()} vs Random",
            "",
            "| env | paired_seeds | reward_sum_delta | reward_mean_delta | sps_delta | win/loss/tie |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for env in sorted(comparisons):
            comparison = comparisons[env]
            lines.append(
                f"| {env} | {len(comparison['pairedSeeds'])} | "
                f"{format_delta(comparison['rewardSumDelta'])} | "
                f"{format_delta(comparison['rewardMeanDelta'])} | "
                f"{format_delta(comparison['stepsPerSecondDelta'], digits=2)} | "
                f"{comparison['wins']}/{comparison['losses']}/{comparison['ties']} |"
            )

with open(summary_md_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines))
    handle.write("\n")
PY

cat "$SUMMARY_MD"
