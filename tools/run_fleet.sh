#!/usr/bin/env bash
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

if [ "${1:-}" = "--job" ]; then
  shift
  STEPS="$1"; NAME="$2"; KEEPER="${3:-}"
  ROOT="$REPO/.oak-benchmarks/$NAME"
  ABLATE="${OAK_ABLATE:-none}"
  if [ -n "$KEEPER" ] && [ -f "$KEEPER/meta.json" ]; then
    KABL="$(.venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1])).get('ablationLabel') or 'none')" "$KEEPER/meta.json")"
    [ "$KABL" = "model-to-keyboard" ] && KABL="none"
    if [ "$KABL" != "$ABLATE" ] && [ "${OAK_ABLATE_FORCE:-0}" != "1" ]; then
      echo "[run_fleet] CONFIG ERROR: candidate OAK_ABLATE='$ABLATE' != best-kernel ablation '$KABL'."
      echo "[run_fleet] The gate would compare different configs. Set OAK_ABLATE='$KABL' or OAK_ABLATE_FORCE=1."
      exit 2
    fi
  fi
  echo "[run_fleet] $NAME — B=${OAK_BENCH_SEED_COUNT:-1408} x $STEPS steps, OAK_ABLATE=$ABLATE, caffeinate -dimsu, DETACHED."
  OAK_ABLATE="$ABLATE" \
  OAK_BENCH_ENVS="${OAK_BENCH_ENVS:-}" \
  OAK_BENCH_POLICIES="${OAK_BENCH_POLICIES:-oak}" \
  OAK_BENCH_STEPS="$STEPS" \
  OAK_BENCH_SEED_COUNT="${OAK_BENCH_SEED_COUNT:-1408}" \
  OAK_RECORD_ACTIONS="${OAK_RECORD_ACTIONS:-1}" \
  OAK_RECORD_FORMAT="${OAK_RECORD_FORMAT:-4}" \
  OAK_FLEET_TIMING=1 \
  OAK_METRICS_STORAGE="$ROOT" \
  caffeinate -dimsu .build/release/OaK benchmark
  echo "[run_fleet] benchmark exit=$?"
  echo "[run_fleet] ===== population report (flags silent death + whole-population view) ====="
  .venv/bin/python tools/fleet.py report "$ROOT" --policy oak || true
  if [ -n "$KEEPER" ]; then
    echo "[run_fleet] ===== gate vs best kernel $KEEPER (REFUSES if either run is incomplete) ====="
    .venv/bin/python tools/fleet.py gate "$KEEPER" "$ROOT" --policy oak || true
  fi
  touch "$ROOT/.done"
  echo "[run_fleet] DONE — .done written."
  exit 0
fi

STEPS="${1:?usage: run_fleet.sh <steps> <out-name> [best-kernel-root]}"
NAME="${2:?usage: run_fleet.sh <steps> <out-name> [best-kernel-root]}"
KEEPER="${3:-}"
ROOT="$REPO/.oak-benchmarks/$NAME"
LOG="$REPO/.oak-benchmarks/$NAME.log"
mkdir -p "$REPO/.oak-benchmarks"
rm -f "$ROOT/.done" 2>/dev/null || true
.venv/bin/python - "$LOG" "$REPO/tools/run_fleet.sh" "$STEPS" "$NAME" "$KEEPER" <<'PY'
import subprocess, sys
log_path, script, steps, name, keeper = sys.argv[1:6]
with open(log_path, "w") as log:
    p = subprocess.Popen(["bash", script, "--job", steps, name, keeper],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
print(f"[run_fleet] DETACHED job PID {p.pid} (new session) — survives Claude Code compaction/idle.")
PY
echo "[run_fleet]   log : $LOG"
echo "[run_fleet]   done: $ROOT/.done   (touched after benchmark + report + gate finish)"
echo "[run_fleet]   poll: tail \"$LOG\"  ;  ls \"$ROOT/.done\""
