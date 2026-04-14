#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OAK_PYTHON_PREFIX="${OAK_PYTHON_PREFIX:-$ROOT_DIR/.venv}"
PYTHON_BIN="$OAK_PYTHON_PREFIX/bin/python"
PYTHON_LIB=""
PYTHONHOME_EMBED=""
OAK_PYTHON_SITE_PACKAGES=""
OAK_BINARY="${OAK_BINARY:-$ROOT_DIR/.build/Release/OaK}"

BENCH_REQUEST="${1:-core}"
RESOLVE_ONLY=0
if [ "${1:-}" = "--resolve-only" ]; then
    RESOLVE_ONLY=1
    BENCH_REQUEST="${2:-core}"
fi

if [ "$BENCH_REQUEST" != "core" ]; then
    printf 'The public benchmark surface only supports the full core suite.\n' >&2
    exit 1
fi

resolve_suite() {
    "$PYTHON_BIN" - <<'PY'
import envpool
import envpool.minigrid.registration  # noqa: F401
import envpool.mujoco.gym.registration  # noqa: F401

envs = {"HalfCheetah-v4", "Humanoid-v4", "Walker2d-v4"}
envs.update(env_id for env_id in envpool.list_all_envs() if env_id.startswith("MiniGrid-"))
print("\n".join(sorted(envs)))
PY
}

if [ "${OAK_BENCH_SKIP_BUILD:-0}" != "1" ]; then
    cd "$ROOT_DIR"
    echo "[info] Building OaK via swift build..."
    swift build -c release
fi
[ -x "$OAK_BINARY" ] || {
    printf 'Missing release binary at %s. Run ./setup.sh or build Release, or set OAK_BINARY.\n' "$OAK_BINARY" >&2
    exit 1
}
[ -d "$OAK_PYTHON_PREFIX" ] || {
    printf 'Missing Python environment at %s. Run ./setup.sh.\n' "$OAK_PYTHON_PREFIX" >&2
    exit 1
}
[ -x "$PYTHON_BIN" ] || {
    printf 'Missing Python executable at %s. Run ./setup.sh.\n' "$PYTHON_BIN" >&2
    exit 1
}
if [ -n "${OAK_PYTHON_LIBRARY:-}" ]; then
    PYTHON_LIB="$OAK_PYTHON_LIBRARY"
else
    PYTHON_LIB="$("$PYTHON_BIN" -c 'import os, sys, sysconfig
ldl = sysconfig.get_config_var("LDLIBRARY") or ""
libdir = sysconfig.get_config_var("LIBDIR") or ""
if not ldl:
    raise SystemExit("LDLIBRARY unset")
for path in (
    os.path.normpath(os.path.join(libdir, ldl)),
    os.path.normpath(os.path.join(sys.base_prefix, ldl)),
    os.path.normpath(os.path.join(sys.base_prefix, "Python")),
):
    if path and os.path.isfile(path):
        print(path)
        break
else:
    raise SystemExit("could not resolve embedded Python library path from sysconfig")
')"
fi
[ -f "$PYTHON_LIB" ] || {
    printf 'Missing Python library at %s.\n' "$PYTHON_LIB" >&2
    exit 1
}
PYTHONHOME_EMBED="$("$PYTHON_BIN" -c 'import sys; print(sys.base_prefix)')"
OAK_PYTHON_SITE_PACKAGES="$("$PYTHON_BIN" -c 'import site; p = site.getsitepackages(); print(p[0] if p else "")')"
export OAK_ROOT_DIR="${OAK_ROOT_DIR:-$ROOT_DIR}"
export PYTHONHOME="$PYTHONHOME_EMBED"
export PYTHONPATH="$OAK_PYTHON_SITE_PACKAGES"
export PYTHON_LIBRARY="$PYTHON_LIB"
export PYGAME_HIDE_SUPPORT_PROMPT=1

if [ "$RESOLVE_ONLY" = "1" ]; then
    resolve_suite
    exit 0
fi

SEED_START="${OAK_BENCH_SEED_START:-0}"
SEED_COUNT="${OAK_BENCH_SEED_COUNT:-10}"
WARMUP_STEPS="${OAK_BENCH_WARMUP_STEPS:-${OAK_BENCH_DEFAULT_WARMUP_STEPS:-1000}}"
STOP_STEPS="${OAK_BENCH_STEPS:-${OAK_BENCH_DEFAULT_STEPS:-50000}}"
POLICIES_RAW="${OAK_BENCH_POLICIES:-oak,random}"
NUM_ENVS="${OAK_BENCH_NUM_ENVS:-${OAK_NUM_ENVS:-1}}"
BENCH_DIR="${OAK_BENCH_DIR:-${OAK_ROOT_DIR:-$ROOT_DIR}/.oak-benchmarks/$(date -u +%Y%m%d-%H%M%S)-$$}"
RUNS_DIR="$BENCH_DIR/runs"
BENCHMARK_JSON="$BENCH_DIR/benchmark.json"

IFS=',' read -r -a POLICIES <<< "$POLICIES_RAW"

mkdir -p "$RUNS_DIR"

printf '[bench] output_dir=%s\n' "$BENCH_DIR"
printf '[bench] execution_policy=serial and policy-blocked; seeds and policies do not run concurrently on the shared device\n'
printf '[bench] measured_step_rounding=global measured env steps advance in multiples of numEnvs and may exceed requested stop steps by up to numEnvs - 1\n'
printf '[bench] measurement_bootstrap=measured benchmarking begins only after warmup is satisfied and at least one prior vector update has established a derived step budget\n'
printf '[bench] resolved_suite_begin\n'
SUITE_ROWS="$(resolve_suite)"
printf '%s\n' "$SUITE_ROWS" | while IFS= read -r concrete_env; do
    printf '[bench] resolved task=%s\n' "$concrete_env"
done
printf '[bench] resolved_suite_end\n'

for ((seed=SEED_START; seed<SEED_START+SEED_COUNT; seed++)); do
    while IFS= read -r concrete_env; do
        for policy in "${POLICIES[@]}"; do
            [ -n "$policy" ] || continue
            stem="$(printf '%s' "${concrete_env}-${policy}-seed${seed}" | tr -cs '[:alnum:]' '-')"
            metrics_path="$RUNS_DIR/${stem}.jsonl"
            log_path="$RUNS_DIR/${stem}.log"

            printf '[bench] start task=%s policy=%s seed=%s warmup=%s stop_steps=%s batch_size=%s\n' \
                "$concrete_env" "$policy" "$seed" "$WARMUP_STEPS" "$STOP_STEPS" "$NUM_ENVS"

            (
                export OAK_ROOT_DIR="$ROOT_DIR"
                export PYTHONHOME="$PYTHONHOME_EMBED"
                export PYTHONPATH="$OAK_PYTHON_SITE_PACKAGES"
                export PYTHON_LIBRARY="$PYTHON_LIB"
                export OAK_NUM_ENVS="$NUM_ENVS"
                export OAK_POLICY="$policy"
                export OAK_SEED="$seed"
                export OAK_WARMUP_STEPS="$WARMUP_STEPS"
                export OAK_STOP_AFTER_STEPS="$STOP_STEPS"
                unset OAK_STOP_AFTER_SECONDS
                export OAK_METRICS_JSONL_PATH="$metrics_path"
                "$OAK_BINARY" "$concrete_env"
            ) >"$log_path" 2>&1
            printf '[bench] done task=%s policy=%s seed=%s\n' "$concrete_env" "$policy" "$seed"
        done
    done <<< "$SUITE_ROWS"
done

"$PYTHON_BIN" - "$BENCH_DIR" "$BENCHMARK_JSON" <<'PY'
from __future__ import annotations
import json, math, subprocess, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

bench_dir = Path(sys.argv[1])
benchmark_json = Path(sys.argv[2])
runs_dir = bench_dir / "runs"
schema = "oak.benchmark.truth_first.v3"

def load_jsonl(path: Path):
    start = end = None
    summaries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if any(token in raw for token in ("NaN", "Infinity", "-Infinity")):
            continue
        record = json.loads(raw)
        if record["type"] == "run_start" and start is None:
            start = record
        elif record["type"] == "summary":
            summaries.append(record)
        elif record["type"] == "run_end":
            end = record
    log_path = path.with_suffix(".log")
    return {
        "start": start,
        "end": end,
        "summaries": summaries,
        "metricsPath": str(path),
        "logPath": str(log_path) if log_path.exists() else "",
    }

def numeric_summary(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": mean,
        "std": std,
    }

def reward_mean(run):
    end = run["end"]
    if not end:
        return None
    steps = int(end.get("measuredStepCount") or 0)
    if steps <= 0:
        return None
    return float(end.get("rewardSum") or 0.0) / float(steps)

def steps_per_second(run):
    end = run["end"]
    if not end:
        return None
    seconds = float(end.get("measuredSeconds") or 0.0)
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return float(end.get("measuredStepCount") or 0.0) / seconds

def termination_rate(run):
    end = run["end"]
    if not end:
        return None
    steps = max(float(end.get("measuredStepCount") or 0), 1.0)
    return float(end.get("doneCount") or 0) / steps

def flatten_health(health, prefix=""):
    out = {}
    if isinstance(health, dict):
        for key, value in health.items():
            child = f"{prefix}.{key}" if prefix else key
            out.update(flatten_health(value, child))
    elif isinstance(health, (int, float)):
        out[prefix] = float(health)
    return out

def validate_run(run):
    start = run["start"]
    end = run["end"]
    problems = []
    env_id = (start or end or {}).get("envId", "")
    descriptor = f'{env_id}/{(start or end or {}).get("policy","")}/seed{(start or end or {}).get("seed",0)}'
    if start is None:
        problems.append(f"{descriptor}: missing run_start in {run['metricsPath']}")
    if end is None:
        problems.append(f"{descriptor}: missing run_end in {run['metricsPath']}")
    if not run["summaries"]:
        problems.append(f"{descriptor}: missing rolling summary records in {run['metricsPath']}")
    if start and end:
        if start.get("envId") != end.get("envId"):
            problems.append(f"{descriptor}: task resolution mismatch in run_end")
        for label, value in flatten_health(end.get("health") or {}).items():
            if not math.isfinite(value):
                problems.append(f"{descriptor}: invalid model-health field {label} in run_end")
                break
    return problems

def exact_sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return {"test": "exact_sign_test", "pValue": 1.0, "significant": False}
    tail = min(wins, losses)
    def choose(n, k):
        if k < 0 or k > n:
            return 0
        k = min(k, n - k)
        value = 1
        for i in range(k):
            value = value * (n - i) // (i + 1)
        return value
    cumulative = sum(choose(n, k) / (2 ** n) for k in range(tail + 1))
    p_value = min(1.0, 2 * cumulative)
    return {"test": "exact_sign_test", "pValue": p_value, "significant": p_value < 0.05}

jsonl_paths = sorted(runs_dir.glob("*.jsonl"))
if not jsonl_paths:
    raise SystemExit(f"No metrics JSONL files in {runs_dir}")

runs = [load_jsonl(path) for path in jsonl_paths]
problems = []
for run in runs:
    problems.extend(validate_run(run))
if problems:
    raise SystemExit("\n".join(sorted(set(problems))))

task_buckets = defaultdict(list)
for run in runs:
    start = run["start"]
    task_buckets[start.get("envId","")].append(run)

tasks = []
for key in sorted(task_buckets):
    entries = task_buckets[key]
    first = entries[0]["start"]
    policy_buckets = defaultdict(list)
    for entry in entries:
        policy_buckets[entry["start"]["policy"]].append(entry)

    policies = {}
    for policy in sorted(policy_buckets):
        bucket = policy_buckets[policy]
        first_start = bucket[0]["start"]
        template_health = None
        for run in bucket:
            health = (run["end"] or {}).get("health") or (run["summaries"][-1] if run["summaries"] else {}).get("health")
            if health:
                template_health = health
                break

        modules = {}
        if template_health:
            for module, fields in template_health.items():
                if not isinstance(fields, dict):
                    continue
                if module == "compute":
                    continue
                modules[module] = {
                    field: numeric_summary([((run["end"] or {}).get("health") or {}).get(module, {}).get(field) for run in bucket])
                    for field in sorted(fields)
                }

        compute = {}
        if template_health and "compute" in template_health:
            compute = {
                field: numeric_summary([((run["end"] or {}).get("health") or {}).get("compute", {}).get(field) for run in bucket])
                for field in sorted(template_health["compute"])
            }

        policies[policy] = {
            "schema": {
                "numEnvs": first_start.get("numEnvs"),
                "warmupSteps": first_start.get("warmupSteps"),
                "stopSteps": first_start.get("stopAfterMeasuredSteps"),
                "summaryWindowSize": first_start.get("summaryWindowSize"),
                "observation": {
                    "flatDim": first_start.get("observationDim"),
                },
                "action": {
                    "flatDim": first_start.get("actionDim"),
                    "discreteCount": first_start.get("actionDiscreteCount"),
                },
            },
            "reward": {
                "sum": numeric_summary([(run["end"] or {}).get("rewardSum") for run in bucket]),
                "mean": numeric_summary([reward_mean(run) for run in bucket]),
            },
            "termination": {
                "doneCount": numeric_summary([(run["end"] or {}).get("doneCount") for run in bucket]),
                "rate": numeric_summary([termination_rate(run) for run in bucket]),
            },
            "speed": {
                "measuredEnvSteps": numeric_summary([(run["end"] or {}).get("measuredStepCount") for run in bucket]),
                "vectorUpdates": numeric_summary([(run["end"] or {}).get("vectorUpdateCount") for run in bucket]),
                "measuredSeconds": numeric_summary([(run["end"] or {}).get("measuredSeconds") for run in bucket]),
                "stepsPerSecond": numeric_summary([steps_per_second(run) for run in bucket]),
            },
            "runtime": {
                "agentStepMs": numeric_summary([(run["end"] or {}).get("agentStepMsMean") for run in bucket]),
                "envStepMs": numeric_summary([(run["end"] or {}).get("envStepMsMean") for run in bucket]),
                "totalStepMs": numeric_summary([(run["end"] or {}).get("totalStepMsMean") for run in bucket]),
            },
            "compute": compute,
            "modules": modules,
            "artifacts": [
                {
                    "seed": run["start"]["seed"],
                    "metricsPath": run["metricsPath"],
                    "logPath": run["logPath"],
                    "runId": run["start"]["runId"],
                    "stopReason": (run["end"] or {}).get("stopReason"),
                }
                for run in sorted(bucket, key=lambda r: r["start"]["seed"])
            ],
        }

    comparisons = {}
    policy_names = sorted(policy_buckets)
    for i, left in enumerate(policy_names[:-1]):
        for right in policy_names[i + 1:]:
            left_by_seed = {run["start"]["seed"]: run for run in policy_buckets[left]}
            right_by_seed = {run["start"]["seed"]: run for run in policy_buckets[right]}
            shared = sorted(set(left_by_seed) & set(right_by_seed))
            reward_sum_deltas = []
            reward_mean_deltas = []
            sps_deltas = []
            termination_deltas = []
            wins = losses = ties = 0
            for seed in shared:
                left_end = left_by_seed[seed]["end"]
                right_end = right_by_seed[seed]["end"]
                reward_sum_delta = float(left_end.get("rewardSum") or 0) - float(right_end.get("rewardSum") or 0)
                reward_sum_deltas.append(reward_sum_delta)
                reward_mean_deltas.append((reward_mean(left_by_seed[seed]) or 0.0) - (reward_mean(right_by_seed[seed]) or 0.0))
                sps_deltas.append((steps_per_second(left_by_seed[seed]) or 0.0) - (steps_per_second(right_by_seed[seed]) or 0.0))
                termination_deltas.append((termination_rate(left_by_seed[seed]) or 0.0) - (termination_rate(right_by_seed[seed]) or 0.0))
                if reward_sum_delta > 0:
                    wins += 1
                elif reward_sum_delta < 0:
                    losses += 1
                else:
                    ties += 1
            comparisons[f"{left}__vs__{right}"] = {
                "leftPolicy": left,
                "rightPolicy": right,
                "pairedSeeds": shared,
                "rewardSumDelta": numeric_summary(reward_sum_deltas),
                "rewardMeanDelta": numeric_summary(reward_mean_deltas),
                "stepsPerSecondDelta": numeric_summary(sps_deltas),
                "terminationRateDelta": numeric_summary(termination_deltas),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "significance": exact_sign_test(wins, losses),
            }

    tasks.append({
        "envId": key,
        "observation": {
            "flatDim": first.get("observationDim"),
        },
        "action": {
            "flatDim": first.get("actionDim"),
            "discreteCount": first.get("actionDiscreteCount"),
        },
        "policies": policies,
        "comparisons": comparisons,
    })

try:
    git_revision = subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"], cwd=bench_dir, text=True).strip()
except Exception:
    git_revision = "unknown"

benchmark = {
    "schema": schema,
    "generatedAt": datetime.now(timezone.utc).isoformat(),
    "benchDir": str(bench_dir),
    "gitRevision": git_revision,
    "taskCount": len(tasks),
    "runCount": len(runs),
    "stepAccounting": {
        "measuredEnvSteps": "global environment steps",
        "vectorUpdates": "batched agent-env update iterations",
        "rounding": "actual measured env steps can exceed the requested stop steps by up to numEnvs - 1",
    },
    "tasks": tasks,
}
benchmark_json.write_text(json.dumps(benchmark, indent=2, sort_keys=True), encoding="utf-8")
print(f"[bench] --- results ---")
print(f"[bench] bench_dir={bench_dir}")
print(f"[bench] benchmark_json={benchmark_json}")
print(f"[bench] task_count={len(tasks)}")
print(f"[bench] run_count={len(runs)}")
print(f"[bench] per-run metrics: {runs_dir}/*.jsonl")
print(f"[bench] per-run logs: {runs_dir}/*.log")
PY
