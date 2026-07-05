#!/usr/bin/env python3
"""THE canonical heartbeat table. Usage: tools/table.py <root1> [root2 ...] <keeper-root> <step>
Any number of run roots, one column each. THE LAST ROOT MUST BE THE CURRENT
KEEPER (user law 2026-07-05: every table shows the keeper, no exceptions).
Refuses (hard error) if ANY root lacks a row at the requested step —
apples-to-apples is structural. Rows fixed: config | deaths | median cum |
median live | P10 | best lane | live max | positivity | SPS |
reward/GFLOP (median) | fleet total (all lanes)."""
import json, sys, os
if len(sys.argv) < 4:
    raise SystemExit("usage: table.py <root1> [root2 ...] <keeper-root> <step>  (>=2 roots: keeper column is MANDATORY)")
roots, step = sys.argv[1:-1], int(sys.argv[-1])
def row(root, s):
    aggs = [json.loads(l) for l in open(f'{root}/aggregate.jsonl')]
    r = [x for x in aggs if x['step'] == s]
    if not r:
        raise SystemExit(
            f"TABLE ERROR: {root} has no row at step {s}.\n"
            f"  available steps: {sorted(x['step'] for x in aggs)}\n"
            f"  Compare only at steps present in ALL roots.")
    r = r[-1]
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
data = [row(rt, step) for rt in roots]
def fmt(r, tot, f):
    return [str(r['dead']), '%+.4f' % r['deliveredRateP50'], '%+.4f' % r['recentRateP50'],
            '%+.4f' % r['deliveredRateP10'], '%.3f' % r['bestLaneDeliveredRate'],
            '%.3f' % r['recentRateMax'], '%.1f%%' % (100 * r['positiveRateFraction']),
            '%dK' % (r['aggStepsPerSecond'] / 1000), '%.0f' % (r['deliveredRateP50'] / f * 1e9),
            '%.1fM' % (tot / 1e6)]
names = ['deaths', 'median (cumulative)', 'median (live)', 'P10', 'best lane',
         'live max', 'positivity', 'SPS', 'reward/GFLOP (median)', 'fleet total (all lanes)']
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
if len(set(abls)) > 1:
    print(f'⚠️  CONFIG MISMATCH: ablations differ across columns ({abls}) — comparison invalid.')
