#!/usr/bin/env bash
# The standard, reliable way to run a fleet benchmark.
#
# WHY FULLY DETACHED (setsid): Claude Code `run_in_background:true` tasks are SIGTERM'd on context
# compaction and on session idle-timeout (anthropics/claude-code GitHub #25188, #32050). That is what
# silently killed our 64-minute runs — short (~6 min) runs finish before a compaction, long ones do
# not. `setsid` puts the job in its OWN session, fully decoupled from the conversation, so it survives
# compaction / idle / session-end. The launcher returns IMMEDIATELY; the detached job writes
# everything to a log and `touch`es <root>/.done when finished (benchmark -> population report ->
# gate). Poll the log + .done — do NOT wrap this script in run_in_background (that reintroduces the
# bug). caffeinate -dimsu is kept as cheap insurance (it is NOT the fix; sleep was never the cause).
#
# Usage:  tools/run_fleet.sh <steps> <out-name> [keeper-root]
# Check:  tail -f .oak-benchmarks/<out-name>.log   ;   ls .oak-benchmarks/<out-name>/.done
# Env:    OAK_BENCH_ENV, OAK_BENCH_POLICIES, OAK_BENCH_SEED_COUNT
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

if [ "${1:-}" = "--job" ]; then
  # ---- the detached job, running inside its own (setsid) session ----
  shift
  STEPS="$1"; NAME="$2"; KEEPER="${3:-}"
  ROOT="$REPO/.oak-benchmarks/$NAME"
  echo "[run_fleet] $NAME — B=${OAK_BENCH_SEED_COUNT:-1024} x $STEPS steps, caffeinate -dimsu, DETACHED."
  OAK_BENCH_ENV="${OAK_BENCH_ENV:-HalfCheetah-v4}" \
  OAK_BENCH_POLICIES="${OAK_BENCH_POLICIES:-oak}" \
  OAK_BENCH_STEPS="$STEPS" \
  OAK_BENCH_SEED_COUNT="${OAK_BENCH_SEED_COUNT:-1024}" \
  OAK_FLEET_TIMING=1 \
  OAK_METRICS_STORAGE="$ROOT" \
  caffeinate -dimsu .build/release/OaK benchmark
  echo "[run_fleet] benchmark exit=$?"
  echo "[run_fleet] ===== population report (flags silent death + whole-population view) ====="
  .venv/bin/python tools/fleet.py report "$ROOT" --policy oak || true
  if [ -n "$KEEPER" ]; then
    echo "[run_fleet] ===== gate vs $KEEPER (REFUSES if either run is incomplete) ====="
    .venv/bin/python tools/fleet.py gate "$KEEPER" "$ROOT" --policy oak || true
  fi
  touch "$ROOT/.done"
  echo "[run_fleet] DONE — .done written."
  exit 0
fi

# ---- launcher: detach a copy of self into its own session and return immediately ----
STEPS="${1:?usage: run_fleet.sh <steps> <out-name> [keeper-root]}"
NAME="${2:?usage: run_fleet.sh <steps> <out-name> [keeper-root]}"
KEEPER="${3:-}"
ROOT="$REPO/.oak-benchmarks/$NAME"
LOG="$REPO/.oak-benchmarks/$NAME.log"
mkdir -p "$REPO/.oak-benchmarks"
rm -f "$ROOT/.done" 2>/dev/null || true
# macOS has no `setsid`; Python's start_new_session=True calls setsid() in the child — same effect:
# a new session, decoupled from Claude Code's process group, surviving compaction/idle/session-end.
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
