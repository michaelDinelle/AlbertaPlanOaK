#!/usr/bin/env bash
# Standard visible verification run: B=1024, fixed CB, heartbeat table.
# Usage: tools/check.sh <name> [steps=100000] [ablate=none]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?usage: check.sh <name> [steps] [ablate] [role]}"; STEPS="${2:-100000}"; ABL="${3:-none}"; ROLE="${4:-candidate}"
ROOT=".oak-benchmarks/$NAME"
rm -rf "$ROOT"
# Role registry: a candidate launch IS the current candidate (the run the
# user's tables track). Scaffold/baseline runs pass role=scaffold and never
# touch the registry.
if [ "$ROLE" = "candidate" ]; then
  .venv/bin/python - "$NAME" <<'PY'
import json, sys
p = '.oak-benchmarks/ROLES.json'
r = json.load(open(p))
r['candidate'] = sys.argv[1]
json.dump(r, open(p, 'w'), indent=1)
PY
fi
# 100k CADENCE, PERIOD (user law 2026-07-05): every run >= 300k heartbeats
# at exactly 100k — one cadence for all screens, Tier-Ls, and keepers, so
# every milestone is a legal comparison point everywhere. Short checks
# (< 300k) keep steps/10 for early kill resolution; their FINAL row lands on
# the horizon (100k check -> 100k row) and is the only comparison point.
HB=100000   # 100k cadence PERIOD — every run, every horizon, one window everywhere
OAK_BENCH_HEARTBEAT_STEPS="$HB" \
OAK_ABLATE="$ABL" OAK_BENCH_POLICIES=oak \
OAK_BENCH_STEPS="$STEPS" OAK_BENCH_SEED_COUNT=1408 OAK_STEPS_PER_CB=1024 \
OAK_FLEET_TIMING=1 OAK_METRICS_STORAGE="$ROOT" .build/release/OaK benchmark 2>&1 | tail -1
.venv/bin/python - "$ROOT" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]+'/aggregate.jsonl')]
print('step | pos | P50 | recentP50 | P10 | best | recentMax | dead | SPS')
for r in rows:
    print(r['step'],'| %.3f'%r['positiveRateFraction'],'| %+.4f'%r['deliveredRateP50'],
          '| %+.4f'%r['recentRateP50'],'| %+.4f'%r['deliveredRateP10'],
          '| %.3f'%r['bestLaneDeliveredRate'],'| %.3f'%r['recentRateMax'],
          '|',r['dead'],'| %d'%r['aggStepsPerSecond'])
PY
