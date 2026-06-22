#!/usr/bin/env bash
# The standard, reliable way to run a fleet benchmark. Always use this — never a bare
# `OaK benchmark` — because it bakes in the three things that kept biting us:
#   (1) caffeinate -dimsu : max sleep-prevention. Long GPU runs die when the laptop sleeps (the
#       Metal/GPU context is lost), producing a SILENT partial run. -dimsu is the strongest assertion,
#       but it still cannot override a closed lid on battery — keep the lid OPEN and stay on power.
#   (2) auto population report : on exit, `fleet.py report` prints the WHOLE-population view (rate
#       distribution + %lanes above each reward tier + sustaining fraction) and a LOUD banner if the
#       run did not reach its horizon. Report the population, never a single hand-picked lane.
#   (3) optional auto-gate vs a keeper (refuses if either run is incomplete).
# Launch it via the harness as a background task (run_in_background:true) so it notifies on completion;
# on the notification, read the log — the population report + completeness verdict are already in it.
#
# Usage: tools/run_fleet.sh <steps> <out-name> [keeper-root]
#   env overrides: OAK_BENCH_ENV, OAK_BENCH_POLICIES, OAK_BENCH_SEED_COUNT
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
STEPS="${1:?usage: run_fleet.sh <steps> <out-name> [keeper-root]}"
NAME="${2:?usage: run_fleet.sh <steps> <out-name> [keeper-root]}"
KEEPER="${3:-}"
ROOT="$PWD/.oak-benchmarks/$NAME"

echo "[run_fleet] $NAME — B=${OAK_BENCH_SEED_COUNT:-1024} x $STEPS steps, caffeinate -dimsu."
echo "[run_fleet] keep the laptop LID OPEN and on POWER — caffeinate cannot stop a lid-close sleep."
OAK_BENCH_ENV="${OAK_BENCH_ENV:-HalfCheetah-v4}" \
OAK_BENCH_POLICIES="${OAK_BENCH_POLICIES:-oak}" \
OAK_BENCH_STEPS="$STEPS" \
OAK_BENCH_SEED_COUNT="${OAK_BENCH_SEED_COUNT:-1024}" \
OAK_FLEET_TIMING=1 \
OAK_METRICS_STORAGE="$ROOT" \
caffeinate -dimsu .build/release/OaK benchmark
RC=$?

echo "[run_fleet] benchmark exit=$RC. Population report (flags silent death + whole-population view):"
.venv/bin/python tools/fleet.py report "$ROOT" --policy oak
if [ -n "$KEEPER" ]; then
  echo "[run_fleet] gate vs $KEEPER (the gate REFUSES if either run is incomplete):"
  .venv/bin/python tools/fleet.py gate "$KEEPER" "$ROOT" --policy oak
fi
