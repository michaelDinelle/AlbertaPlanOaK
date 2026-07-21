#!/usr/bin/env python3
"""THE canonical heartbeat table. Usage: tools/table.py <root1> [root2 ...] <keeper-root> <step>
Any number of run roots, one column each. THE LAST ROOT MUST BE THE CURRENT
KEEPER (user law 2026-07-05: every table shows the keeper, no exceptions).
Refuses (hard error) if ANY root lacks a row at the requested step —
apples-to-apples is structural.
User law 2026-07-19: one full table PER ENVIRONMENT under the same call —
fleet total and SPS only make sense per env (reward scales and lane costs
differ). Single-env runs print a single table. Rows fixed: config | deaths |
median cum | median live | P10 | best lane | live max | positivity | SPS |
reward/GFLOP (median) | fleet total."""
import json, sys, os
BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.oak-benchmarks')
if len(sys.argv) >= 4:
    roots, step = sys.argv[1:-1], int(sys.argv[-1])
else:
    # DEFAULT MODE (user law 2026-07-06): "the table" = CANDIDATE vs parent vs
    # KEEPER from .oak-benchmarks/ROLES.json — no discretion about which roots
    # to show. Optional single arg = step; omitted = latest common 100k
    # milestone across the role roots.
    R = json.load(open(f'{BENCH}/ROLES.json'))
    roots = [f'{BENCH}/{R[k]}' for k in ('candidate', 'parent', 'keeper') if R.get(k)]
    if len(sys.argv) == 2:
        step = int(sys.argv[1])
    else:
        common = None
        for rt in roots:
            steps = {json.loads(l)['step'] for l in open(f'{rt}/aggregate.jsonl') if l.strip()}
            steps = {t for t in steps if t % 100000 == 0}
            common = steps if common is None else (common & steps)
        if not common:
            raise SystemExit("TABLE ERROR: no common 100k milestone across candidate/parent/keeper yet.")
        step = max(common)

def lane_envs(root):
    """seed -> envId from fleet.jsonl rows (mixed) or meta.envId (legacy)."""
    meta = json.load(open(f'{root}/meta.json'))
    default = meta.get('envId')
    out = {}
    for l in open(f'{root}/fleet.jsonl'):
        x = json.loads(l)
        s = x.get('seed')
        if s is not None and s not in out:
            out[s] = x.get('envId', default)
    return out, meta

def heartbeat_window(root, s):
    aggs = [json.loads(l)['step'] for l in open(f'{root}/aggregate.jsonl')]
    prior = [t for t in aggs if t < s]
    return s - (max(prior) if prior else 0)

def agg_rows(root, s):
    """Per-env aggregate at step s from fleet.jsonl lanes."""
    envs, meta = lane_envs(root)
    flops = meta.get('flopsPerLaneStep', 105296)
    # Per-env lane cost: mixed runs stamp flopsPerLaneStep per envSchedule
    # entry (env physics differs per env; the agent part is env-independent).
    env_flops = {e['envId']: e.get('flopsPerLaneStep', flops)
                 for e in (meta.get('envSchedule') or [])}
    by_env = {}
    for l in open(f'{root}/fleet.jsonl'):
        x = json.loads(l)
        if x['step'] != s or x['kind'] not in ('hb', 'final'):
            continue
        e = envs.get(x['seed'])
        if e is None:
            continue
        by_env.setdefault(e, []).append(x)
    out = {}
    for e, rows in by_env.items():
        rates = sorted(x['rewardSum'] / s for x in rows)
        recent = [x.get('health', {}).get('rewardRate', 0.0) for x in rows]
        recent_s = sorted(recent)
        n = len(rates)
        dead = sum(1 for x in rows if x.get('firstNonFiniteStep'))
        pos = sum(1 for r in recent if r > 0)
        sps = sum(x.get('stepsPerSecond', 0.0) for x in rows)
        def q(p):
            return rates[min(n - 1, int(p * n))]
        out[e] = {
            'dead': dead, 'median_cum': q(0.5),
            'median_live': recent_s[n // 2] if recent_s else 0.0,
            'p10': q(0.1), 'best': rates[-1], 'live_max': max(recent) if recent else 0.0,
            'pos': 100.0 * pos / n, 'sps': sps, 'flops': env_flops.get(e, flops),
            'total': sum(x['rewardSum'] for x in rows), 'n': n,
        }
    return out, meta

data = []
for rt in roots:
    # Step presence check (apples-to-apples structural law).
    steps = {json.loads(l)['step'] for l in open(f'{rt}/aggregate.jsonl') if l.strip()}
    if step not in steps:
        raise SystemExit(
            f"TABLE ERROR: {rt} has no row at step {step}.\n"
            f"  available steps: {sorted(steps)}\n"
            f"  Compare only at steps present in ALL roots.")
    env_agg, meta = agg_rows(rt, step)
    data.append((env_agg, meta))
# envSchedule comparability (mixed-env fleets): schedules must match exactly,
# EXCEPT when one root is single-env (null schedule) and the other is mixed —
# then we render only the single env they share (the overlapping slice).
# The per-entry flopsPerLaneStep is measurement metadata (like the whitelisted
# top-level key — it changes with organCostVersion), so the comparison keys on
# the lane ASSIGNMENT (envId + seed ranges) only.
def _sched_sig(m):
    sig = [{k: v for k, v in e.items() if k != 'flopsPerLaneStep'}
           for e in (m.get('envSchedule') or [])]
    return json.dumps(sig or None, sort_keys=True)
scheds = [_sched_sig(m) for _, m in data]
if len(set(scheds)) > 1:
    null_sched = [i for i, s in enumerate(scheds) if s == 'null']
    if len(null_sched) == len(scheds) - 1:
        # One mixed root, rest single-env: restrict to the shared env.
        shared = set()
        for env_agg, _ in data:
            shared |= set(env_agg.keys())
        common = [e for e in shared if all(e in env_agg for env_agg, _ in data)]
        if not common:
            raise SystemExit(
                "TABLE ERROR: mixed-vs-single roots share no env — refusing to render.")
        # Filter all roots to the shared env(s).
        data = [({e: agg[e] for e in common if e in agg}, m) for agg, m in data]
    else:
        raise SystemExit(
            "TABLE ERROR: envSchedule differs across roots — mixed-env fleets are not "
            "apples-to-apples; refusing to render.\n  " + "\n  ".join(scheds))
# WHITELIST comparability gate (user law 2026-07-05).
metas = [m for _, m in data]
# organFlops/organCostVersion are measurement metadata (Phase 8), not run
# config: pre-organ roots (version 0) MUST stay comparable with v1 roots —
# the keeper-comparison law outranks the version gate. A version split is
# flagged below the table instead of refusing (cross-version ABSOLUTE
# reward/GFLOP claims remain refused; the row is a ratio-share instrument).
ALLOWED_DIFF = {'startedAt', 'ablation', 'ablationLabel', 'steps', 'flopsPerLaneStep',
                'organFlops', 'organCostVersion',
                # lane-assignment equality already enforced by the _sched_sig
                # gate above; raw entries may differ only in per-entry flops.
                'envSchedule'}
# Mixed-vs-single: batch/seeds/envSchedule differ by construction (the mixed
# root has more lanes on other envs); the shared-env slice is apples-to-apples.
if len(set(_sched_sig(m) for m in metas)) > 1:
    ALLOWED_DIFF |= {'batch', 'seeds', 'envSchedule'}
allkeys = set().union(*(m.keys() for m in metas))
mismatches = []
for kf in sorted(allkeys - ALLOWED_DIFF):
    vals = [json.dumps(m.get(kf), sort_keys=True) for m in metas]
    if len(set(vals)) > 1:
        mismatches.append(f"  {kf}: " + " vs ".join(vals))
windows_chk = [heartbeat_window(rt, step) for rt in roots]
if len(set(windows_chk)) > 1:
    mismatches.append(f"  heartbeatWindow@step: {windows_chk}")
if mismatches:
    raise SystemExit(
        "TABLE ERROR: runs are not apples-to-apples — refusing to render.\n"
        "Fields differing (whitelist: only startedAt may differ):\n"
        + "\n".join(mismatches))

def fmt(e):
    if e is None:
        return ['—'] * 10
    return [str(e['dead']), '%+.4f' % e['median_cum'], '%+.4f' % e['median_live'],
            '%+.4f' % e['p10'], '%.3f' % e['best'], '%.3f' % e['live_max'],
            '%.1f%%' % e['pos'], '%dK' % (e['sps'] / 1000),
            '%.0f' % (e['median_cum'] / e['flops'] * 1e9) if e['flops'] else '—',
            '%.1fM' % (e['total'] / 1e6)]

names = ['deaths', 'median (cumulative)', 'median (live)', 'P10', 'best lane',
         'live max', 'positivity', 'SPS', 'reward/GFLOP (median)', 'fleet total']
hdrs = [os.path.basename(os.path.normpath(rt)) for rt in roots]
abls = [(m.get('ablationLabel') or 'none') for _, m in data]
cfgs = [f'ablation={a}, B={len(m.get("seeds", [])) or m.get("batch")}, {m.get("steps", 0)//1000}k, {m.get("startedAt", "?")}'
        for a, (_, m) in zip(abls, data)]
env_ids = sorted({e for env_agg, _ in data for e in env_agg})
for env_id in env_ids:
    print('| @%dk [%s] | %s |' % (step // 1000, env_id, ' | '.join(hdrs)))
    print('|---|' + '---|' * len(roots))
    print('| config | %s |' % ' | '.join(cfgs))
    cols = [fmt(env_agg.get(env_id)) for env_agg, _ in data]
    for i, n in enumerate(names):
        print('| %s | %s |' % (n, ' | '.join(c[i] for c in cols)))
    print()
ocv = {m.get('organCostVersion', 0) for _, m in data}
if len(ocv) > 1:
    print('⚠️ organCostVersion differs across roots (%s): the FLOP model changed '
          '(Phase 8 re-bin + twin-dot/FeatureConstruction/ActionEmission corrections). '
          'reward/GFLOP rows are computed from each run\'s own model — compare them as '
          'ratios; cross-version ABSOLUTE FLOP claims are refused (design §1).'
          % ' vs '.join(str(v) for v in sorted(ocv)))
print('⤷ DISPLAY CONTRACT: paste this table VERBATIM into the reply text — tool output is not user-visible; keeper column mandatory; never summarize or say "table above".')
