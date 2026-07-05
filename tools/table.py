#!/usr/bin/env python3
"""THE canonical heartbeat table. Usage: tools/table.py <candidate-root> <keeper-root> <step>
Rows (fixed order, never edit casually — the user reads these across runs):
deaths | median cum | median live | P10 | best lane | live max | positivity |
SPS | reward/GFLOP (median) | fleet total (all lanes)."""
import json, sys, os
cand, keep, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
def row(root, s):
    aggs = [json.loads(l) for l in open(f'{root}/aggregate.jsonl')]
    r = [x for x in aggs if x['step'] == s]
    if not r:
        # HARD REFUSAL (user law 2026-07-05): a step-mismatched table is
        # apples-to-oranges and must be impossible to produce, not warned about.
        raise SystemExit(
            f"TABLE ERROR: {root} has no row at step {s}.\n"
            f"  available steps: {sorted(x['step'] for x in aggs)}\n"
            f"  Compare only at steps present in BOTH roots.")
    r = r[-1]
    # fleet total: nearest heartbeat step with rows (an exact-step miss used
    # to print a misleading 0.0M).
    hbs = {}
    for l in open(f'{root}/fleet.jsonl'):
        x = json.loads(l)
        if x['kind'] == 'hb':
            hbs.setdefault(x['step'], 0.0)
            hbs[x['step']] += x['rewardSum']
    tstep = min(hbs, key=lambda t: abs(t - r['step'])) if hbs else r['step']
    tot = hbs.get(tstep, 0.0)
    meta = json.load(open(f'{root}/meta.json'))
    f = meta.get('flopsPerLaneStep', 105296)
    return r, tot, f, meta
(c, ct, cf, cm), (k, kt, kf, km) = row(cand, step), row(keep, step)
def fmt(r, tot, f):
    return [str(r['dead']), '%+.4f' % r['deliveredRateP50'], '%+.4f' % r['recentRateP50'],
            '%+.4f' % r['deliveredRateP10'], '%.3f' % r['bestLaneDeliveredRate'],
            '%.3f' % r['recentRateMax'], '%.1f%%' % (100 * r['positiveRateFraction']),
            '%dK' % (r['aggStepsPerSecond'] / 1000), '%.0f' % (r['deliveredRateP50'] / f * 1e9),
            '%.1fM' % (tot / 1e6)]
names = ['deaths', 'median (cumulative)', 'median (live)', 'P10', 'best lane',
         'live max', 'positivity', 'SPS', 'reward/GFLOP (median)', 'fleet total (all lanes)']
cv, kv = fmt(c, ct, cf), fmt(k, kt, kf)
cn, kn = os.path.basename(os.path.normpath(cand)), os.path.basename(os.path.normpath(keep))
ca, ka = cm.get('ablationLabel') or 'none', km.get('ablationLabel') or 'none'
print(f'| @{step//1000}k | {cn} | {kn} |')
print('|---|---|---|')
print(f'| config | ablation={ca}, B={len(cm.get("seeds", [])) or cm.get("batch")}, {cm.get("steps", 0)//1000}k, {cm.get("startedAt", "?")} | ablation={ka}, B={len(km.get("seeds", [])) or km.get("batch")}, {km.get("steps", 0)//1000}k, {km.get("startedAt", "?")} |')
for n, a, b in zip(names, cv, kv):
    print(f'| {n} | {a} | {b} |')
if ca != ka:
    print(f'⚠️  CONFIG MISMATCH: candidate ablation={ca} vs keeper ablation={ka} — comparison invalid.')

