#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_SCRIPT="$ROOT_DIR/scripts/benchmark_report.py"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/miniforge3/envs/oak}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"

DEFAULT_REQUESTS=(core)
VALID_REQUESTS=(pyflyt carla cheetah walker humanoid minigrid smoke core)
HARDWARE_OPTIMAL_ENVS=(cheetah walker humanoid minigrid)

SEED_START="${OAK_BENCH_SEED_START:-0}"
SEED_COUNT="${OAK_BENCH_SEED_COUNT:-10}"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
BENCH_DIR="${OAK_BENCH_DIR:-$ROOT_DIR/.oak-benchmarks/$TIMESTAMP}"
RUNS_DIR="$BENCH_DIR/runs"
MANIFEST_PATH="$RUNS_DIR/.benchmark_manifest.tsv"
BENCHMARK_JSON="$BENCH_DIR/benchmark.json"
DEFAULT_WARMUP_STEPS="${OAK_BENCH_DEFAULT_WARMUP_STEPS:-1000}"
DEFAULT_BENCHMARK_STEPS="${OAK_BENCH_DEFAULT_STEPS:-50000}"
RESOLVE_ONLY=0

usage() {
    cat <<'EOF'
Usage: ./benchmark.sh [--resolve-only] [request ...]

Single benchmark harness for OaK.

Requests:
  smoke             Resolve the fast iteration league.
  core              Resolve the scientific promotion league.
  pyflyt            Run the PyFlyt task directly.
  carla             Run the CARLA task directly.
  cheetah           Run HalfCheetah-v4 directly.
  walker            Run Walker2d-v4 directly.
  humanoid          Run Humanoid-v4 directly.
  minigrid          Run the MiniGrid family (or one task if OAK_ENVPOOL_TASK_ID is set).

Defaults:
  requests: core
  seeds: 0..9

Environment variables:
  OAK_BENCH_POLICIES=oak,random
  OAK_BENCH_BASELINE_POLICY=<policy>
  OAK_BENCH_SEED_START=<int>
  OAK_BENCH_SEED_COUNT=<int>
  OAK_BENCH_DIR=<path>
  OAK_BENCH_SKIP_BUILD=1
  OAK_BENCH_SKIP_DOCTOR=1
  OAK_BENCH_WARMUP_STEPS=<int>
  OAK_BENCH_STEPS=<int>
  OAK_BENCH_NUM_ENVS=<int>
  OAK_BENCH_SMOKE_MINIGRID_LIMIT=<int>        default 3
  OAK_ENVPOOL_TASK_ID=<gym-id>                single-task ad hoc mode only
  OAK_MINIGRID_ENV_ID=<gym-id>                legacy single-task ad hoc mode only
  OAK_ENVPOOL_SUITE_PREFIX=<prefix>           ad hoc family mode only

Artifacts:
  benchmark.json
  runs/*.jsonl
  runs/*.log

Notes:
  - League runs ignore inherited task-selection env vars and resolve concrete tasks explicitly.
  - --resolve-only prints the resolved task triples (league, logical env, concrete env id) and exits.
EOF
}

is_valid_request() {
    local candidate="$1"
    local request
    for request in "${VALID_REQUESTS[@]}"; do
        if [ "$request" = "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

is_hardware_optimal() {
    local candidate="$1"
    local env
    for env in "${HARDWARE_OPTIMAL_ENVS[@]}"; do
        if [ "$env" = "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

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

validate_python_prefix() {
    [ -d "$CONDA_ENV_DIR" ] || {
        printf 'Missing conda environment at %s. Run ./doctor.sh python or ./setup.sh.\n' "$CONDA_ENV_DIR" >&2
        exit 1
    }
    [ -x "$PYTHON_BIN" ] || {
        printf 'Missing Python executable at %s. Run ./doctor.sh python or ./setup.sh.\n' "$PYTHON_BIN" >&2
        exit 1
    }
}

parse_policies() {
    python3 - "$1" <<'PY'
import sys

raw = sys.argv[1]
policies = [token.strip().lower() for token in raw.split(",") if token.strip()]
if not policies:
    raise SystemExit("OAK_BENCH_POLICIES resolved to an empty list")

seen = set()
for policy in policies:
    if policy in seen:
        continue
    seen.add(policy)
    print(policy)
PY
}

resolve_requested_runs() {
    validate_python_prefix
    PYGAME_HIDE_SUPPORT_PROMPT=1 "$PYTHON_BIN" - "$@" <<'PY'
import os
import sys

FIXED = {
    "cheetah": "HalfCheetah-v4",
    "walker": "Walker2d-v4",
    "humanoid": "Humanoid-v4",
}
SMOKE_MINIGRID_PREFERRED = [
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-MemoryS9-v0",
]
SMOKE_MINIGRID_LIMIT = max(int(os.environ.get("OAK_BENCH_SMOKE_MINIGRID_LIMIT", "3")), 1)
DEFAULT_PREFIX = "MiniGrid-"


def load_registered_envs():
    import envpool.registration as registration

    return sorted(str(env_id) for env_id in registration.list_all_envs())


registered_envs = None


def registered_with_prefix(prefix):
    global registered_envs
    if registered_envs is None:
        registered_envs = load_registered_envs()
    return [env_id for env_id in registered_envs if env_id.startswith(prefix)]


def resolve_minigrid_family(prefix):
    env_ids = registered_with_prefix(prefix)
    if not env_ids:
        raise SystemExit(
            f"EnvPool reported no environment ids with prefix {prefix!r}. "
            "Check the configured Python environment."
        )
    return env_ids


def resolve_smoke_minigrid():
    minigrids = resolve_minigrid_family(DEFAULT_PREFIX)
    selected = [env_id for env_id in SMOKE_MINIGRID_PREFERRED if env_id in minigrids]
    for env_id in minigrids:
        if env_id not in selected:
            selected.append(env_id)
        if len(selected) >= SMOKE_MINIGRID_LIMIT:
            break
    return selected[:SMOKE_MINIGRID_LIMIT]


def append(rows, seen, league, logical_env, concrete_env_id):
    key = (league, logical_env, concrete_env_id)
    if key in seen:
        return
    seen.add(key)
    rows.append(key)


def resolve_ad_hoc_minigrid():
    explicit = (
        os.environ.get("OAK_ENVPOOL_TASK_ID", "").strip()
        or os.environ.get("OAK_MINIGRID_ENV_ID", "").strip()
    )
    if explicit and explicit.lower() != "suite":
        return [explicit]
    prefix = os.environ.get("OAK_ENVPOOL_SUITE_PREFIX", "").strip() or DEFAULT_PREFIX
    return resolve_minigrid_family(prefix)


def expand_request(request, rows, seen):
    request = request.lower()
    if request == "core":
        for logical_env, concrete_env_id in FIXED.items():
            append(rows, seen, "core", logical_env, concrete_env_id)
        for concrete_env_id in resolve_minigrid_family(DEFAULT_PREFIX):
            append(rows, seen, "core", "minigrid", concrete_env_id)
        return

    if request == "smoke":
        for logical_env, concrete_env_id in FIXED.items():
            append(rows, seen, "smoke", logical_env, concrete_env_id)
        for concrete_env_id in resolve_smoke_minigrid():
            append(rows, seen, "smoke", "minigrid", concrete_env_id)
        return

    if request in FIXED:
        append(rows, seen, "adhoc", request, FIXED[request])
        return

    if request == "minigrid":
        for concrete_env_id in resolve_ad_hoc_minigrid():
            append(rows, seen, "adhoc", "minigrid", concrete_env_id)
        return

    if request in {"pyflyt", "carla"}:
        append(rows, seen, "adhoc", request, request)
        return

    raise SystemExit(f"Unknown benchmark request: {request}")


rows = []
seen = set()
for request in sys.argv[1:]:
    expand_request(request, rows, seen)

for row in rows:
    print("\t".join(row))
PY
}

build_if_needed() {
    if [ "${OAK_BENCH_SKIP_BUILD:-0}" = "1" ]; then
        return
    fi

    (
        cd "$ROOT_DIR"
        echo "[info] Building OaK and MLX shaders via xcodebuild..."
        xcodebuild -scheme OaK -configuration Release -destination 'platform=macOS' SYMROOT="$(pwd)/.build" -quiet
    )
}

bench_requires_python() {
    local manifest_rows="$1"
    while IFS=$'\t' read -r _league logical_env _concrete_env; do
        if [ "$logical_env" != "carla" ]; then
            return 0
        fi
    done <<<"$manifest_rows"
    return 1
}

run_doctor_gates() {
    local resolved_rows="$1"

    if [ "${OAK_BENCH_SKIP_DOCTOR:-0}" = "1" ]; then
        return
    fi

    (
        cd "$ROOT_DIR"
        OAK_DOCTOR_SKIP_BUILD=1 ./doctor.sh metal
    )

    if bench_requires_python "$resolved_rows"; then
        (
            cd "$ROOT_DIR"
            OAK_DOCTOR_SKIP_BUILD=1 ./doctor.sh python
        )
    fi
}

sanitize_filename() {
    printf '%s' "$1" | tr -cs '[:alnum:]' '-'
}

validate_and_write_row() {
    local row_path="$1"
    local metrics_path="$2"
    local league="$3"
    local logical_env="$4"
    local concrete_env_id="$5"
    local policy="$6"
    local seed="$7"
    local num_envs="$8"
    local warmup_steps="$9"
    local stop_steps="${10}"
    local log_path="${11}"

    python3 - "$metrics_path" "$row_path" "$league" "$logical_env" "$concrete_env_id" "$policy" "$seed" "$num_envs" "$warmup_steps" "$stop_steps" "$log_path" <<'PY'
import json
import sys
from pathlib import Path

(
    metrics_path,
    row_path,
    league,
    logical_env,
    concrete_env_id,
    policy,
    seed,
    num_envs,
    warmup_steps,
    stop_steps,
    log_path,
) = sys.argv[1:]

records = [json.loads(line) for line in Path(metrics_path).read_text(encoding="utf-8").splitlines() if line.strip()]
run_start = next((record for record in records if record.get("type") == "run_start"), None)
run_end = next((record for record in reversed(records) if record.get("type") == "run_end"), None)
if run_start is None or run_end is None:
    raise SystemExit(f"Metrics file {metrics_path} is missing run_start or run_end")

if run_start.get("env") != concrete_env_id:
    raise SystemExit(
        f"Task-resolution mismatch for {metrics_path}: run_start env={run_start.get('env')!r}, expected {concrete_env_id!r}"
    )
if run_end.get("env") != concrete_env_id:
    raise SystemExit(
        f"Run-end mismatch for {metrics_path}: run_end env={run_end.get('env')!r}, expected {concrete_env_id!r}"
    )
if run_start.get("logicalEnv") != logical_env:
    raise SystemExit(
        f"Logical env mismatch for {metrics_path}: run_start logicalEnv={run_start.get('logicalEnv')!r}, expected {logical_env!r}"
    )
if run_start.get("concreteEnvId") != concrete_env_id:
    raise SystemExit(
        f"Concrete env id mismatch for {metrics_path}: run_start concreteEnvId={run_start.get('concreteEnvId')!r}, expected {concrete_env_id!r}"
    )
if run_start.get("schemaFingerprint") in {None, ""}:
    raise SystemExit(f"Missing schema fingerprint in run_start for {metrics_path}")
backend = run_start.get("backend") or run_end.get("backend") or ""
if policy == "oak" and backend != "metal":
    raise SystemExit(
        f"OaK benchmark must run on Metal, but {metrics_path} reported backend={backend!r}"
    )

row = [
    league,
    logical_env,
    concrete_env_id,
    run_start.get("backend") or run_end.get("backend") or "",
    policy,
    seed,
    num_envs,
    warmup_steps,
    stop_steps,
    str(run_end.get("runId") or run_start.get("runId") or ""),
    metrics_path,
    log_path,
    str(run_start.get("schemaFingerprint") or ""),
    str(run_start.get("observationSpaceKind") or ""),
    str(run_start.get("observationFlatDim") or ""),
    str(run_start.get("actionSpaceKind") or ""),
    str(run_start.get("actionFlatDim") or ""),
    str(run_start.get("actionDiscreteCount") or ""),
    json.dumps(run_start.get("actionShape") or []),
]
Path(row_path).write_text("\t".join(row) + "\n", encoding="utf-8")
PY
}

print_run_completion() {
    local metrics_path="$1"
    local logical_env="$2"
    local concrete_env_id="$3"
    local policy="$4"
    local seed="$5"

    python3 - "$metrics_path" "$logical_env" "$concrete_env_id" "$policy" "$seed" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, logical_env, concrete_env_id, policy, seed = sys.argv[1:]
records = [json.loads(line) for line in Path(metrics_path).read_text(encoding="utf-8").splitlines() if line.strip()]
run_end = next((record for record in reversed(records) if record.get("type") == "run_end"), None)
if run_end is None:
    raise SystemExit(f"Metrics file {metrics_path} is missing run_end")

def fmt(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)

print(
    "[bench] done env={env} task={task} policy={policy} seed={seed} reward_sum={reward} done_count={done} steps_per_second={sps} stop_reason={stop}".format(
        env=logical_env,
        task=concrete_env_id,
        policy=policy,
        seed=seed,
        reward=fmt(float(run_end.get("rewardSum", 0.0) or 0.0)),
        done=fmt(float(run_end.get("doneCount", 0.0) or 0.0)),
        sps=fmt(float(run_end.get("stepsPerSecond", 0.0) or 0.0)),
        stop=run_end.get("stopReason") or "",
    )
)
PY
}

run_one() {
    local league="$1"
    local logical_env="$2"
    local concrete_env_id="$3"
    local policy="$4"
    local seed="$5"
    local warmup_steps="$6"
    local stop_steps="$7"

    local num_envs="${OAK_BENCH_NUM_ENVS:-}"
    if [ -z "$num_envs" ]; then
        if is_hardware_optimal "$logical_env"; then
            num_envs=1024
        else
            num_envs=8
        fi
    fi

    local safe_concrete
    safe_concrete="$(sanitize_filename "$concrete_env_id")"
    local stem="${logical_env}-${safe_concrete}-${policy}-seed${seed}"
    local metrics_path="$RUNS_DIR/${stem}.jsonl"
    local log_path="$RUNS_DIR/${stem}.log"
    local row_path="$RUNS_DIR/${stem}.row.tsv"

    printf '[bench] start league=%s env=%s task=%s policy=%s seed=%s warmup=%s stop_steps=%s batch_size=%s\n' \
        "$league" "$logical_env" "$concrete_env_id" "$policy" "$seed" "$warmup_steps" "${stop_steps:-na}" "$num_envs"

    local task_envpool_id=""
    if [ "$logical_env" = "minigrid" ]; then
        task_envpool_id="$concrete_env_id"
    fi

    if ! (
        cd "$ROOT_DIR"
        PYGAME_HIDE_SUPPORT_PROMPT=1 \
        OAK_NUM_ENVS="$num_envs" \
        OAK_POLICY="$policy" \
        OAK_SEED="$seed" \
        OAK_WARMUP_STEPS="$warmup_steps" \
        OAK_STOP_AFTER_STEPS="$stop_steps" \
        OAK_STOP_AFTER_SECONDS= \
        OAK_METRICS_JSONL_PATH="$metrics_path" \
        OAK_CARLA_AUTOSTART="${OAK_CARLA_AUTOSTART:-0}" \
        OAK_BENCH_LEAGUE="$league" \
        OAK_ENVPOOL_TASK_ID="$task_envpool_id" \
        OAK_MINIGRID_ENV_ID= \
        OAK_ENVPOOL_SUITE_PREFIX= \
        ./run.sh "$logical_env" >"$log_path" 2>&1
    ); then
        printf '[bench] run failed; see %s\n' "$log_path" >&2
        return 1
    fi

    validate_and_write_row \
        "$row_path" \
        "$metrics_path" \
        "$league" \
        "$logical_env" \
        "$concrete_env_id" \
        "$policy" \
        "$seed" \
        "$num_envs" \
        "$warmup_steps" \
        "$stop_steps" \
        "$log_path"

    print_run_completion "$metrics_path" "$logical_env" "$concrete_env_id" "$policy" "$seed"
}

ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --resolve-only)
            RESOLVE_ONLY=1
            ;;
        *)
            ARGS+=("$1")
            ;;
    esac
    shift
done

if [ "${#ARGS[@]}" -eq 0 ]; then
    ARGS=("${DEFAULT_REQUESTS[@]}")
fi

for request in "${ARGS[@]}"; do
    if ! is_valid_request "$request"; then
        printf 'Unknown benchmark request: %s\n' "$request" >&2
        usage >&2
        exit 1
    fi
done

validate_positive_integer "OAK_BENCH_SEED_COUNT" "$SEED_COUNT"
POLICIES=()
while IFS= read -r policy; do
    POLICIES+=("$policy")
done < <(parse_policies "${OAK_BENCH_POLICIES:-oak,random}")

if [ "${#POLICIES[@]}" -eq 0 ]; then
    printf 'No benchmark policies were configured.\n' >&2
    exit 1
fi

resolved_rows="$(resolve_requested_runs "${ARGS[@]}")"
if [ -z "$resolved_rows" ]; then
    printf 'No concrete tasks were resolved for requests: %s\n' "${ARGS[*]}" >&2
    exit 1
fi

if [ "$RESOLVE_ONLY" = "1" ]; then
    printf '%s\n' "$resolved_rows"
    exit 0
fi

mkdir -p "$RUNS_DIR"

cleanup_benchmark_intermediates() {
    find "$RUNS_DIR" -name '*.row.tsv' -delete 2>/dev/null || true
    rm -f "$MANIFEST_PATH"
}

trap 'cleanup_benchmark_intermediates' EXIT

printf '[bench] output_dir=%s\n' "$BENCH_DIR"
build_if_needed
run_doctor_gates "$resolved_rows"

while IFS=$'\t' read -r league logical_env concrete_env_id; do
    warmup_steps="${OAK_BENCH_WARMUP_STEPS:-$DEFAULT_WARMUP_STEPS}"
    validate_nonnegative_integer "warmup steps for $logical_env/$concrete_env_id" "$warmup_steps"

    stop_steps="${OAK_BENCH_STEPS:-$DEFAULT_BENCHMARK_STEPS}"
    validate_positive_integer "benchmark steps for $logical_env/$concrete_env_id" "$stop_steps"

    seed="$SEED_START"
    seed_limit=$((SEED_START + SEED_COUNT))
    pids=()
    while [ "$seed" -lt "$seed_limit" ]; do
        for policy in "${POLICIES[@]}"; do
            run_one "$league" "$logical_env" "$concrete_env_id" "$policy" "$seed" "$warmup_steps" "$stop_steps" &
            pids+=($!)
        done
        seed=$((seed + 1))
    done

    failed=0
    if [ ${#pids[@]} -gt 0 ]; then
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done
    fi
    if [ "$failed" -ne 0 ]; then
        printf 'One or more benchmark runs failed for %s/%s.\n' "$logical_env" "$concrete_env_id" >&2
        exit 1
    fi
done <<<"$resolved_rows"

printf 'league\tlogical_env\tconcrete_env_id\tbackend\tpolicy\tseed\tnum_envs\twarmup_steps\tstop_steps\trun_id\tmetrics_path\tlog_path\tschema_fingerprint\tobservation_space_kind\tobservation_flat_dim\taction_space_kind\taction_flat_dim\taction_discrete_count\taction_shape\n' >"$MANIFEST_PATH"

row_files=()
while IFS= read -r row_file; do
    row_files+=("$row_file")
done < <(find "$RUNS_DIR" -name '*.row.tsv' | sort)

if [ "${#row_files[@]}" -eq 0 ]; then
    printf 'No benchmark manifest rows were generated in %s\n' "$RUNS_DIR" >&2
    exit 1
fi

for row_file in "${row_files[@]}"; do
    cat "$row_file" >>"$MANIFEST_PATH"
done

GIT_REVISION="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown')"
BASELINE_POLICY="${OAK_BENCH_BASELINE_POLICY:-}"
if [ -z "$BASELINE_POLICY" ]; then
    for policy in "${POLICIES[@]}"; do
        if [ "$policy" = "random" ]; then
            BASELINE_POLICY="random"
            break
        fi
    done
fi
if [ -z "$BASELINE_POLICY" ]; then
    BASELINE_POLICY="${POLICIES[0]}"
fi

python3 "$REPORT_SCRIPT" \
    --bench-dir "$BENCH_DIR" \
    --manifest "$MANIFEST_PATH" \
    --benchmark-json "$BENCHMARK_JSON" \
    --git-revision "$GIT_REVISION" \
    --python-bin "$PYTHON_BIN" \
    --baseline-policy "$BASELINE_POLICY"

task_count="$(python3 - "$BENCHMARK_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    benchmark = json.load(handle)
print(benchmark["config"]["resolvedTaskCount"])
PY
)"

run_count="$(python3 - "$BENCHMARK_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    benchmark = json.load(handle)
print(benchmark["config"]["runCount"])
PY
)"

printf '[bench] task_count=%s\n' "$task_count"
printf '[bench] run_count=%s\n' "$run_count"
printf '[bench] benchmark_json=%s\n' "$BENCHMARK_JSON"
