#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_SCRIPT="$ROOT_DIR/scripts/benchmark_report.py"
OAK_PYTHON_PREFIX="${OAK_PYTHON_PREFIX:-$ROOT_DIR/.venv}"
PYTHON_BIN="$OAK_PYTHON_PREFIX/bin/python"
PYTHON_LIB=""
# Embedding must see the real stdlib (encodings, …) plus venv site-packages; venvs do not duplicate stdlib under .venv.
PYTHONHOME_EMBED=""
OAK_PYTHON_SITE_PACKAGES=""
OAK_BINARY="${OAK_BINARY:-$ROOT_DIR/.build/Release/OaK}"

resolve_oak_python_library() {
    if [ -n "${OAK_PYTHON_LIBRARY:-}" ]; then
        PYTHON_LIB="$OAK_PYTHON_LIBRARY"
        return 0
    fi
    PYTHON_LIB="$("$PYTHON_BIN" -c 'import os, sys, sysconfig
ldl = sysconfig.get_config_var("LDLIBRARY") or ""
libdir = sysconfig.get_config_var("LIBDIR") or ""
if not ldl:
    raise SystemExit("LDLIBRARY unset")
candidates = [
    os.path.normpath(os.path.join(libdir, ldl)),
    os.path.normpath(os.path.join(sys.base_prefix, ldl)),
    os.path.normpath(os.path.join(sys.base_prefix, "Python")),
]
for path in candidates:
    if path and os.path.isfile(path):
        print(path)
        break
else:
    raise SystemExit("could not resolve embedded Python library path from sysconfig")
')" || return 1
}

DEFAULT_REQUESTS=(core)
VALID_REQUESTS=(pyflyt cheetah walker humanoid minigrid smoke core)
HARDWARE_OPTIMAL_ENVS=(cheetah walker humanoid minigrid)

SEED_START="${OAK_BENCH_SEED_START:-0}"
SEED_COUNT="${OAK_BENCH_SEED_COUNT:-10}"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
BENCH_DIR="${OAK_BENCH_DIR:-$ROOT_DIR/.oak-benchmarks/$TIMESTAMP}"
RUNS_DIR="$BENCH_DIR/runs"
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
  cheetah           Run HalfCheetah-v4 directly.
  walker            Run Walker2d-v4 directly.
  humanoid          Run Humanoid-v4 directly.
  minigrid          Run the MiniGrid family (or one task if OAK_ENVPOOL_TASK_ID is set).

Defaults:
  requests: core
  seeds: 0..9

Environment variables:
  OAK_PYTHON_PREFIX=<path>                  default <repo>/.venv
  OAK_PYTHON_LIBRARY=<path>                 optional; auto from sysconfig if unset
  OAK_BENCH_POLICIES=oak,random
  OAK_BENCH_BASELINE_POLICY=<policy>
  OAK_BENCH_SEED_START=<int>
  OAK_BENCH_SEED_COUNT=<int>
  OAK_BENCH_DIR=<path>
  OAK_BENCH_SKIP_BUILD=1
  OAK_BENCH_WARMUP_STEPS=<int>
  OAK_BENCH_STEPS=<int>
  OAK_BENCH_NUM_ENVS=<int>
  OAK_BENCH_SMOKE_MINIGRID_LIMIT=<int>        default 3
  OAK_ENVPOOL_TASK_ID=<gym-id>                single-task ad hoc mode only
  OAK_MINIGRID_ENV_ID=<gym-id>                legacy single-task ad hoc mode only
  OAK_ENVPOOL_SUITE_PREFIX=<prefix>           ad hoc family mode only

Artifacts:
  benchmark.json                 Schema oak.benchmark: dense ordered stats, provenance
  runs/*.jsonl                   Per-run metrics (single source of truth for benchmark_report)
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
    [ -d "$OAK_PYTHON_PREFIX" ] || {
        printf 'Missing Python environment at %s. Run ./setup.sh.\n' "$OAK_PYTHON_PREFIX" >&2
        exit 1
    }
    [ -x "$PYTHON_BIN" ] || {
        printf 'Missing Python executable at %s. Run ./setup.sh.\n' "$PYTHON_BIN" >&2
        exit 1
    }
    resolve_oak_python_library || {
        printf 'Could not resolve libpython for embedding. Set OAK_PYTHON_LIBRARY to the libpython dylib/framework path.\n' >&2
        exit 1
    }
    [ -f "$PYTHON_LIB" ] || {
        printf 'Missing Python library at %s. Run ./setup.sh or set OAK_PYTHON_LIBRARY.\n' "$PYTHON_LIB" >&2
        exit 1
    }
    PYTHONHOME_EMBED="$("$PYTHON_BIN" -c 'import sys; print(sys.base_prefix)')"
    OAK_PYTHON_SITE_PACKAGES="$("$PYTHON_BIN" -c 'import site; p = site.getsitepackages(); print(p[0] if p else "")')"
    [ -n "$PYTHONHOME_EMBED" ] || {
        printf 'Could not resolve sys.base_prefix for embedding.\n' >&2
        exit 1
    }
    [ -n "$OAK_PYTHON_SITE_PACKAGES" ] || {
        printf 'Could not resolve venv site-packages path.\n' >&2
        exit 1
    }
}

# OaK embedding: PYTHONHOME = stdlib prefix (base_prefix); PYTHONPATH = venv site-packages only.
# Exporting these in the parent shell breaks plain `python3`; keep them inside the subshell around OaK.
validate_oak_release_binary() {
    [ -x "$OAK_BINARY" ] || {
        printf 'Missing release binary at %s. Run ./setup.sh or xcodebuild -scheme OaK -configuration Release, or set OAK_BINARY.\n' "$OAK_BINARY" >&2
        exit 1
    }
}

with_oak_python_embedding() {
    (
        export OAK_ROOT_DIR="${OAK_ROOT_DIR:-$ROOT_DIR}"
        export PYTHONHOME="$PYTHONHOME_EMBED"
        export PYTHONPATH="$OAK_PYTHON_SITE_PACKAGES"
        export PYTHON_LIBRARY="$PYTHON_LIB"
        "$@"
    )
}

validate_oak_python_imports_for_benchmark() {
    # Covers every logical env this harness may run (union of per-env import checks).
    PYGAME_HIDE_SUPPORT_PROMPT=1 "$PYTHON_BIN" -c 'import gymnasium, numpy, PyFlyt, PyFlyt.gym_envs, envpool' >/dev/null 2>&1 || {
        printf 'Python import preflight failed for benchmark suite. Run ./setup.sh.\n' >&2
        exit 1
    }
}

validate_oak_metal_if_needed() {
    local policy
    for policy in "${POLICIES[@]}"; do
        if [ "$policy" = "oak" ]; then
            if ! with_oak_python_embedding "$OAK_BINARY" doctor-metal --quiet >/dev/null 2>&1; then
                printf 'Metal preflight failed. Run ./setup.sh or ensure a Metal-capable GPU is available.\n' >&2
                exit 1
            fi
            return 0
        fi
    done
}

prepare_oak_benchmark_runtime() {
    validate_oak_release_binary
    validate_python_prefix
    export OAK_ROOT_DIR="$ROOT_DIR"
    validate_oak_python_imports_for_benchmark
    validate_oak_metal_if_needed
}

parse_policies() {
    "$PYTHON_BIN" - "$1" <<'PY'
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

    if request == "pyflyt":
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

sanitize_filename() {
    printf '%s' "$1" | tr -cs '[:alnum:]' '-'
}

print_run_completion() {
    local metrics_path="$1"
    local logical_env="$2"
    local concrete_env_id="$3"
    local policy="$4"
    local seed="$5"

    "$PYTHON_BIN" - "$metrics_path" "$logical_env" "$concrete_env_id" "$policy" "$seed" <<'PY'
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

    printf '[bench] start league=%s env=%s task=%s policy=%s seed=%s warmup=%s stop_steps=%s batch_size=%s\n' \
        "$league" "$logical_env" "$concrete_env_id" "$policy" "$seed" "$warmup_steps" "${stop_steps:-na}" "$num_envs"

    local task_envpool_id=""
    if [ "$logical_env" = "minigrid" ]; then
        task_envpool_id="$concrete_env_id"
    fi

    if ! (
        cd "$ROOT_DIR"
        export OAK_ROOT_DIR="$ROOT_DIR"
        export PYTHONHOME="$PYTHONHOME_EMBED"
        export PYTHONPATH="$OAK_PYTHON_SITE_PACKAGES"
        export PYTHON_LIBRARY="$PYTHON_LIB"
        PYGAME_HIDE_SUPPORT_PROMPT=1 \
        OAK_NUM_ENVS="$num_envs" \
        OAK_POLICY="$policy" \
        OAK_SEED="$seed" \
        OAK_WARMUP_STEPS="$warmup_steps" \
        OAK_STOP_AFTER_STEPS="$stop_steps" \
        OAK_STOP_AFTER_SECONDS= \
        OAK_METRICS_JSONL_PATH="$metrics_path" \
        OAK_BENCH_LEAGUE="$league" \
        OAK_ENVPOOL_TASK_ID="$task_envpool_id" \
        OAK_MINIGRID_ENV_ID= \
        OAK_ENVPOOL_SUITE_PREFIX= \
        "$OAK_BINARY" "$logical_env" >"$log_path" 2>&1
    ); then
        printf '[bench] run failed; see %s\n' "$log_path" >&2
        return 1
    fi

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

printf '[bench] output_dir=%s\n' "$BENCH_DIR"
build_if_needed
prepare_oak_benchmark_runtime

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

(cd "$BENCH_DIR" && "$PYTHON_BIN" "$REPORT_SCRIPT")

policies_csv=""
for policy in "${POLICIES[@]}"; do
    if [ -n "$policies_csv" ]; then
        policies_csv+=","
    fi
    policies_csv+="$policy"
done

printf '\n[bench] ─── results (paths and headline metrics) ───\n'
printf '[bench] bench_dir=%s\n' "$BENCH_DIR"
printf '[bench] benchmark_json=%s\n' "$BENCHMARK_JSON"
printf '[bench] requests=%s\n' "${ARGS[*]}"
printf '[bench] seed_range=%s..%s\n' "$SEED_START" "$((SEED_START + SEED_COUNT - 1))"
printf '[bench] policies=%s\n' "$policies_csv"

"$PYTHON_BIN" - "$BENCHMARK_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    b = json.load(handle)

tasks = b["tasks"]
stat_columns = b["statColumns"]
comparison_delta_order = b["comparisonDeltaOrder"]
mean_idx = stat_columns.index("mean")
reward_sum_delta_idx = comparison_delta_order.index("rewardSumDelta")

run_count = sum(
    len(pol["artifacts"])
    for t in tasks
    for pol in t["policies"].values()
)

print(f"[bench] task_count={len(tasks)}")
print(f"[bench] run_count={run_count}")
print(f"[bench] schema={b.get('schema', '')}")
print(f"[bench] generated_at={b['generatedAt']}")
print(f"[bench] git_revision={b['gitRevision']}")
print(f"[bench] config_hash={b['configHash']}")
print(f"[bench] baseline_policy={b['baselinePolicy']}")

pv = b.get("pythonVersions") or {}
if pv:
    bits = " ".join(f"{k}={v}" for k, v in sorted(pv.items()))
    print(f"[bench] python_versions {bits}")

for t in tasks:
    league = t["league"]
    logical = t["logicalEnv"]
    cid = t["concreteEnvId"]
    print(f"[bench] task {league}/{logical}/{cid}")
    comps = t.get("comparisons") or {}
    for pair_name in sorted(comps.keys()):
        c = comps[pair_name]
        rs = c["deltas"][reward_sum_delta_idx]
        sig = c["sig"]
        mean = rs[mean_idx] if len(rs) > mean_idx else None
        p = sig[0] if sig else None
        parts = [pair_name]
        if mean is not None:
            parts.append(f"reward_sum_delta_mean={mean:.6g}")
        if p is not None:
            parts.append(f"sign_test_p={p:.6g}")
        print(f"[bench]   compare {' | '.join(parts)}")
PY

printf '[bench] per-run metrics: %s/runs/*.jsonl\n' "$BENCH_DIR"
printf '[bench] per-run logs: %s/runs/*.log\n' "$BENCH_DIR"
