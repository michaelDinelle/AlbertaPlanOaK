#!/usr/bin/env python3
"""THE canonical heartbeat table. Usage: tools/table.py <root1> [root2 ...] <keeper-root> <step>
Any number of run roots, one column each. THE LAST ROOT MUST BE THE CURRENT
KEEPER (user law 2026-07-05: every table shows the keeper, no exceptions).
Refuses (hard error) if ANY root lacks a row at the requested step —
apples-to-apples is structural. Rows fixed: config | deaths | median cum |
median live | P10 | best lane | live max | positivity | SPS |
reward/GFLOP (median) | fleet total (all lanes).
Mixed-env runs (envSchedule): the table renders the POOLED block followed by a
per-env block for every env in the schedule — user law 2026-07-19: the Tiers
must include all environments in the table."""
import json, sys, os
from collections import defaultdict
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

def row(root, s):
    aggs = [json.loads(l) for l in open(f'{root}/aggregate.jsonl')]
    r = [x for x in aggs if x['step'] == s]
    if not r:
        raise SystemExit(
            f"TABLE ERROR: {root} has no row at step {s}.\n"
            f"  available steps: {sorted(x['step'] for x in aggs)}\n"
            f"  Compare only at steps present in ALL roots.")
    r = r[-1]
    # heartbeat window at this step: the 'live' rows average over exactly this
    # many steps — comparable across roots ONLY if the windows match.
    prior = [x['step'] for x in aggs if x['step'] < s]
    r['_window'] = s - (max(prior) if prior else 0)
    # fleet total at the EXACT step, from hb AND final rows (a run's last
    # step is written as kind=final; the old nearest-step fallback silently
    # substituted the previous heartbeat's total — every 1M table showed the
    # 900k sum. Exact-step or hard error, same law as the quantile rows.)
    tot, nrows = 0.0, 0
    for l in open(f'{root}/fleet.jsonl'):
        x = json.loads(l)
        if x['step'] == s and x['kind'] in ('hb', 'final'):
            tot += x['rewardSum']; nrows += 1
    if nrows == 0:
        raise SystemExit(f"TABLE ERROR: {root} has no per-lane rows at step {s} — cannot compute fleet total.")
    meta = json.load(open(f'{root}/meta.json'))
    f = meta.get('flopsPerLaneStep', 105296)
    return r, tot, f, meta

def env_row(root, s, env_id, envs):
    """Per-env aggregate at step s from fleet.jsonl lanes."""
    rates, rewards, pos, dead = [], 0.0, 0, 0
    best, live_max = 0.0, 0.0
    recent = []
    for l in open(f'{root}/fleet.jsonl'):
        x = json.loads(l)
        if x['step'] != s or x['kind'] not in ('hb', 'final'):
            continue
        if envs.get(x['seed']) != env_id:
            continue
        rewards += x['rewardSum']
        rates.append(x['rewardSum'] / s)
        if x.get('firstNonFiniteStep'):
            dead += 1
        h = x.get('health', {})
        rr = h.get('rewardRate', 0.0)
        recent.append(rr)
        if rr > 0:
            pos += 1
        best = max(best, x['rewardSum'] / s)
        live_max = max(live_max, rr)
    if not rates:
        return None
    rates.sort()
    n = len(rates)
    def q(p):
        return rates[min(n - 1, int(p * n))]
    return {
        'dead': dead, 'median_cum': q(0.5), 'median_live': sorted(recent)[len(recent)//2] if recent else 0.0,
        'p10': q(0.1), 'best': best, 'live_max': live_max,
        'pos': 100.0 * pos / n, 'n': n, 'total': rewards,
    }

data = [row(rt, step) for rt in roots]
env_maps = {}
for rt in roots:
    env_maps[rt], _ = lane_envs(rt)
# envSchedule (mixed-env fleets): when EITHER root carries it, both must match
# exactly — lanes on different env schedules are never apples-to-apples. Checked
# explicitly ahead of the generic whitelist loop for an actionable error; roots
# without the field (all legacy single-env runs) compare equal ('null') and pass.
scheds = [json.dumps(m.get('envSchedule'), sort_keys=True) for _, _, _, m in data]
if len(set(scheds)) > 1:
    raise SystemExit(
        "TABLE ERROR: envSchedule differs across roots — mixed-env fleets are not "
        "apples-to-apples; refusing to render.\n  " + "\n  ".join(scheds)
        + "\nGate per env (fleet.py gate --env-slice) instead of tabling pooled lanes.")
# WHITELIST comparability gate (user law 2026-07-05): a table renders ONLY
# when every field below is IDENTICAL across all roots. Anything not on the
# allowed-to-differ list (run name, startedAt) that differs = hard refusal.
# This is a whitelist by construction: new meta fields are compared by
# default; a field must be explicitly added to ALLOWED_DIFF to be exempt.
# ablation is the TREATMENT variable — candidate vs keeper legitimately
# differ in it (user ruling 2026-07-05); it stays loudly visible in the
# config row. Everything else fail-closed.
# 'steps' (run horizon) may differ ONLY because the separate heartbeat-window
# equality check below enforces the REAL apples-to-apples criterion: identical
# cadence at the compared step (100k everywhere by law). A row at step S with
# equal windows is the same measurement regardless of how long the runs
# ultimately go — this is what makes keeper tables possible at every 100k
# milestone of a 300k screen (keeper-column-mandatory law).
# 'flopsPerLaneStep' is a derived accounting constant (feeds only the
# reward/GFLOP row), not a behavior config: the 2026-07-19 slimming corrected
# the hand formula (105296 -> structural 220768), so keeper roots predating the
# fix legitimately differ. Exempt like startedAt.
ALLOWED_DIFF = {'startedAt', 'ablation', 'ablationLabel', 'steps', 'flopsPerLaneStep'}
metas = [m for _, _, _, m in data]
allkeys = set().union(*(m.keys() for m in metas))
mismatches = []
for kf in sorted(allkeys - ALLOWED_DIFF):
    vals = [json.dumps(m.get(kf), sort_keys=True) for m in metas]
    if len(set(vals)) > 1:
        mismatches.append(f"  {kf}: " + " vs ".join(vals))
windows_chk = [r['_window'] for r, _, _, _ in data]
if len(set(windows_chk)) > 1:
    mismatches.append(f"  heartbeatWindow@step: {windows_chk}")
if mismatches:
    raise SystemExit(
        "TABLE ERROR: runs are not apples-to-apples — refusing to render.\n"
        "Fields differing (whitelist: only startedAt may differ):\n"
        + "\n".join(mismatches)
        + "\nRun the candidate with the IDENTICAL config (same horizon, seeds, "
        "ablation, env, cadence) before comparing.")

def fmt(r, tot, f):
    return [str(r['dead']), '%+.4f' % r['deliveredRateP50'], '%+.4f' % r['recentRateP50'],
            '%+.4f' % r['deliveredRateP10'], '%.3f' % r['bestLaneDeliveredRate'],
            '%.3f' % r['recentRateMax'], '%.1f%%' % (100 * r['positiveRateFraction']),
            '%dK' % (r['aggStepsPerSecond'] / 1000), '%.0f' % (r['deliveredRateP50'] / f * 1e9),
            '%.1fM' % (tot / 1e6)]

def fmt_env(e):
    if e is None:
        return ['—'] * 8
    return [str(e['dead']), '%+.4f' % e['median_cum'], '%+.4f' % e['median_live'],
            '%+.4f' % e['p10'], '%.3f' % e['best'], '%.3f' % e['live_max'],
            '%.1f%%' % e['pos'], '%.1fM' % (e['total'] / 1e6)]

names = ['deaths', 'median (cumulative)', 'median (live)', 'P10', 'best lane',
         'live max', 'positivity', 'SPS', 'reward/GFLOP (median)', 'fleet total (all lanes)']
env_names = ['deaths', 'median (cumulative)', 'median (live)', 'P10', 'best lane',
             'live max', 'positivity', 'fleet total']
cols = [fmt(r, t, f) for r, t, f, _ in data]
hdrs = [os.path.basename(os.path.normpath(rt)) for rt in roots]
abls = [(m.get('ablationLabel') or 'none') for _, _, _, m in data]
cfgs = [f'ablation={a}, B={len(m.get("seeds", [])) or m.get("batch")}, {m.get("steps", 0)//1000}k, {m.get("startedAt", "?")}'
        for a, (_, _, _, m) in zip(abls, data)]
print('| @%dk | %s |' % (step // 1000, ' | '.join(hdrs)))
print('|---|' + '---|' * len(roots))
print('| config | %s |' % ' | '.join(cfgs))
for i, n in enumerate(names):
    print('| %s | %s |' % (n, ' | '.join(c[i] for c in cols)))
# Per-env blocks (user law 2026-07-19: the Tiers must include all environments).
env_ids = sorted({e for em in env_maps.values() for e in em.values() if e})
if len(env_ids) > 1:
    for env_id in env_ids:
        ecols = []
        for rt in roots:
            e = env_row(rt, step, env_id, env_maps[rt])
            ecols.append(fmt_env(e))
        print('| [%s] | %s |' % (env_id, ' | '.join([''] * len(roots))))
        for i, n in enumerate(env_names):
            print('|  %s %s | %s |' % (' ', n, ' | '.join(c[i] for c in ecols)))
print('⤷ DISPLAY CONTRACT: paste this table VERBATIM into the reply text — tool output is not user-visible; keeper column mandatory; never summarize or say "table above".')
