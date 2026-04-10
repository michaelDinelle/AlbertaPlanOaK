#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_SCRIPT="$ROOT_DIR/scripts/benchmark_report.py"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/miniforge3/envs/oak}"
PYTHON_VERSION="${OAK_PYTHON_VERSION:-3.11}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"

DEFAULT_REQUESTS=("${OAK_BENCH_DEFAULT_REQUEST:-smoke}")
VALID_REQUESTS=(pyflyt carla cheetah walker humanoid minigrid all hardware-optimal slow smoke core)
HARDWARE_OPTIMAL_ENVS=(cheetah walker humanoid minigrid)
SLOW_ENVS=(pyflyt carla)

MODE="${OAK_BENCH_MODE:-quality}"
SEED_START="${OAK_BENCH_SEED_START:-0}"
SEED_COUNT="${OAK_BENCH_SEED_COUNT:-10}"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
BENCH_DIR="${OAK_BENCH_DIR:-$ROOT_DIR/.oak-benchmarks/$TIMESTAMP}"
RUNS_DIR="$BENCH_DIR/runs"
RESOLVED_MANIFEST_PATH="$BENCH_DIR/resolved_manifest.tsv"
LEGACY_MANIFEST_PATH="$BENCH_DIR/manifest.tsv"
SUMMARY_JSON="$BENCH_DIR/summary.json"
TASK_SUMMARY_JSON="$BENCH_DIR/task_summary.json"
FAMILY_SUMMARY_JSON="$BENCH_DIR/family_summary.json"
RESEARCH_VERDICT_JSON="$BENCH_DIR/research_verdict.json"
SUMMARY_MD="$BENCH_DIR/summary.md"
REPORT_MD="$BENCH_DIR/REPORT.md"

DEFAULT_WARMUP_STEPS="${OAK_BENCH_DEFAULT_WARMUP_STEPS:-1000}"
DEFAULT_QUALITY_STEPS="${OAK_BENCH_DEFAULT_QUALITY_STEPS:-50000}"
DEFAULT_THROUGHPUT_SECONDS="${OAK_BENCH_DEFAULT_THROUGHPUT_SECONDS:-10}"
RESOLVE_ONLY=0

usage() {
    cat <<'EOF'
Usage: ./benchmark.sh [--resolve-only] [request ...]

Scientific benchmark harness for OaK.

Requests:
  smoke             Resolve the fast iteration league.
  core              Resolve the scientific promotion league.
  pyflyt            Run the PyFlyt task directly.
  carla             Run the CARLA task directly.
  cheetah           Run HalfCheetah-v4 directly.
  walker            Run Walker2d-v4 directly.
  humanoid          Run Humanoid-v4 directly.
  minigrid          Run the MiniGrid family (or one task if OAK_ENVPOOL_TASK_ID is set).
  hardware-optimal  Expand to cheetah walker humanoid minigrid.
  slow              Expand to pyflyt carla.
  all               Expand to pyflyt cheetah walker humanoid minigrid.

Defaults:
  requests: smoke
  mode: quality
  seeds: 0..9

Environment variables:
  OAK_BENCH_MODE=quality|throughput|both
  OAK_BENCH_POLICIES=oak,random
  OAK_BENCH_BASELINE_POLICY=<policy>
  OAK_BENCH_SEED_START=<int>
  OAK_BENCH_SEED_COUNT=<int>
  OAK_BENCH_DIR=<path>
  OAK_BENCH_SKIP_BUILD=1
  OAK_BENCH_SKIP_DOCTOR=1
  OAK_BENCH_WARMUP_STEPS=<int>
  OAK_BENCH_QUALITY_STEPS=<int>
  OAK_BENCH_THROUGHPUT_SECONDS=<float>
  OAK_BENCH_NUM_ENVS=<int>
  OAK_BENCH_SMOKE_MINIGRID_LIMIT=<int>        default 3
  OAK_ENVPOOL_TASK_ID=<gym-id>                single-task ad hoc mode only
  OAK_MINIGRID_ENV_ID=<gym-id>                legacy single-task ad hoc mode only
  OAK_ENVPOOL_SUITE_PREFIX=<prefix>           ad hoc family mode only

Artifacts:
  resolved_manifest.tsv
  task_summary.json
  family_summary.json
  research_verdict.json
  summary.json
  summary.md
  REPORT.md

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
DEFAULT_ENVS = ["pyflyt", "cheetah", "walker", "humanoid", "minigrid"]
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

    if request == "hardware-optimal":
        for env_name in ["cheetah", "walker", "humanoid", "minigrid"]:
            expand_request(env_name, rows, seen)
        return

    if request == "slow":
        for env_name in ["pyflyt", "carla"]:
            expand_request(env_name, rows, seen)
        return

    if request == "all":
        for env_name in DEFAULT_ENVS:
            expand_request(env_name, rows, seen)
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

    if [ "${OAK_ALLOW_CPU_FALLBACK:-0}" = "1" ]; then
        printf 'Normal benchmark mode requires Metal. Unset OAK_ALLOW_CPU_FALLBACK or set OAK_BENCH_SKIP_DOCTOR=1 for diagnosis-only runs.\n' >&2
        exit 1
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
    local mode="$6"
    local policy="$7"
    local seed="$8"
    local num_envs="$9"
    local warmup_steps="${10}"
    local stop_steps="${11}"
    local stop_seconds="${12}"
    local log_path="${13}"

    python3 - "$metrics_path" "$row_path" "$league" "$logical_env" "$concrete_env_id" "$mode" "$policy" "$seed" "$num_envs" "$warmup_steps" "$stop_steps" "$stop_seconds" "$log_path" <<'PY'
import json
import sys
from pathlib import Path

(
    metrics_path,
    row_path,
    league,
    logical_env,
    concrete_env_id,
    mode,
    policy,
    seed,
    num_envs,
    warmup_steps,
    stop_steps,
    stop_seconds,
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

row = [
    league,
    mode,
    logical_env,
    concrete_env_id,
    run_start.get("backend") or run_end.get("backend") or "",
    policy,
    seed,
    num_envs,
    warmup_steps,
    stop_steps,
    stop_seconds,
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

run_one() {
    local mode="$1"
    local league="$2"
    local logical_env="$3"
    local concrete_env_id="$4"
    local policy="$5"
    local seed="$6"
    local warmup_steps="$7"
    local stop_steps="$8"
    local stop_seconds="$9"

    local num_envs="${OAK_BENCH_NUM_ENVS:-}"
    local allow_cpu_fallback="${OAK_ALLOW_CPU_FALLBACK:-0}"
    if [ -z "$num_envs" ]; then
        if is_hardware_optimal "$logical_env"; then
            num_envs=1024
        else
            num_envs=8
        fi
    fi

    local safe_concrete
    safe_concrete="$(sanitize_filename "$concrete_env_id")"
    local stem="${logical_env}-${safe_concrete}-${policy}-${mode}-seed${seed}"
    local metrics_path="$RUNS_DIR/${stem}.jsonl"
    local log_path="$RUNS_DIR/${stem}.log"
    local row_path="$RUNS_DIR/${stem}.row.tsv"

    printf '[bench] league=%s mode=%s env=%s task=%s policy=%s seed=%s warmup=%s stop_steps=%s stop_seconds=%s batch_size=%s\n' \
        "$league" "$mode" "$logical_env" "$concrete_env_id" "$policy" "$seed" "$warmup_steps" "${stop_steps:-na}" "${stop_seconds:-na}" "$num_envs"

    local task_envpool_id=""
    if [ "$logical_env" = "minigrid" ]; then
        task_envpool_id="$concrete_env_id"
    fi

    if ! (
        cd "$ROOT_DIR"
        PYGAME_HIDE_SUPPORT_PROMPT=1 \
        OAK_ALLOW_CPU_FALLBACK="$allow_cpu_fallback" \
        OAK_NUM_ENVS="$num_envs" \
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
        "$mode" \
        "$policy" \
        "$seed" \
        "$num_envs" \
        "$warmup_steps" \
        "$stop_steps" \
        "$stop_seconds" \
        "$log_path"
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

printf '[bench] output_dir=%s\n' "$BENCH_DIR"
build_if_needed
run_doctor_gates "$resolved_rows"

while IFS=$'\t' read -r league logical_env concrete_env_id; do
    for mode in "${MODES[@]}"; do
        warmup_steps="${OAK_BENCH_WARMUP_STEPS:-$DEFAULT_WARMUP_STEPS}"
        validate_nonnegative_integer "warmup steps for $logical_env/$concrete_env_id" "$warmup_steps"

        stop_steps=""
        stop_seconds=""
        if [ "$mode" = "quality" ]; then
            stop_steps="${OAK_BENCH_QUALITY_STEPS:-$DEFAULT_QUALITY_STEPS}"
            validate_positive_integer "quality steps for $logical_env/$concrete_env_id" "$stop_steps"
        else
            stop_seconds="${OAK_BENCH_THROUGHPUT_SECONDS:-$DEFAULT_THROUGHPUT_SECONDS}"
        fi

        seed="$SEED_START"
        seed_limit=$((SEED_START + SEED_COUNT))
        pids=()
        while [ "$seed" -lt "$seed_limit" ]; do
            for policy in "${POLICIES[@]}"; do
                run_one "$mode" "$league" "$logical_env" "$concrete_env_id" "$policy" "$seed" "$warmup_steps" "$stop_steps" "$stop_seconds" &
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
            printf 'One or more benchmark runs failed for %s/%s in %s mode.\n' "$logical_env" "$concrete_env_id" "$mode" >&2
            exit 1
        fi
    done
done <<<"$resolved_rows"

printf 'league\tmode\tlogical_env\tconcrete_env_id\tbackend\tpolicy\tseed\tnum_envs\twarmup_steps\tstop_steps\tstop_seconds\trun_id\tmetrics_path\tlog_path\tschema_fingerprint\tobservation_space_kind\tobservation_flat_dim\taction_space_kind\taction_flat_dim\taction_discrete_count\taction_shape\n' >"$RESOLVED_MANIFEST_PATH"

row_files=()
while IFS= read -r row_file; do
    row_files+=("$row_file")
done < <(find "$RUNS_DIR" -name '*.row.tsv' | sort)

if [ "${#row_files[@]}" -eq 0 ]; then
    printf 'No resolved manifest rows were generated in %s\n' "$RUNS_DIR" >&2
    exit 1
fi

for row_file in "${row_files[@]}"; do
    cat "$row_file" >>"$RESOLVED_MANIFEST_PATH"
done
cp "$RESOLVED_MANIFEST_PATH" "$LEGACY_MANIFEST_PATH"

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
    --manifest "$RESOLVED_MANIFEST_PATH" \
    --summary-json "$SUMMARY_JSON" \
    --task-summary-json "$TASK_SUMMARY_JSON" \
    --family-summary-json "$FAMILY_SUMMARY_JSON" \
    --research-verdict-json "$RESEARCH_VERDICT_JSON" \
    --summary-md "$SUMMARY_MD" \
    --report-md "$REPORT_MD" \
    --git-revision "$GIT_REVISION" \
    --python-bin "$PYTHON_BIN" \
    --baseline-policy "$BASELINE_POLICY"

cat "$SUMMARY_MD"
printf '\n[bench] Wrote scientific benchmark artifacts to %s\n' "$BENCH_DIR"
