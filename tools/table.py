#!/usr/bin/env python3
"""THE canonical heartbeat table. Usage: tools/table.py <candidate-root> <keeper-root> <step>
Rows (fixed order, never edit casually — the user reads these across runs):
deaths | median cum | median live | P10 | best lane | live max | positivity |
SPS | reward/GFLOP (median) | fleet total (all lanes)."""
import json, sys
cand, keep, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
def row(root, s):
    aggs = [json.loads(l) for l in open(f'{root}/aggregate.jsonl')]
    r = [x for x in aggs if x['step'] == s]
    r = r[-1] if r else aggs[-1]
    tot = sum(x['rewardSum'] for x in (json.loads(l) for l in open(f'{root}/fleet.jsonl'))
              if x['step'] == r['step'] and x['kind'] == 'hb')
    f = json.load(open(f'{root}/meta.json')).get('flopsPerLaneStep', 105296)
    return r, tot, f
(c, ct, cf), (k, kt, kf) = row(cand, step), row(keep, step)
def fmt(r, tot, f):
    return [str(r['dead']), '%+.4f' % r['deliveredRateP50'], '%+.4f' % r['recentRateP50'],
            '%+.4f' % r['deliveredRateP10'], '%.3f' % r['bestLaneDeliveredRate'],
            '%.3f' % r['recentRateMax'], '%.1f%%' % (100 * r['positiveRateFraction']),
            '%dK' % (r['aggStepsPerSecond'] / 1000), '%.0f' % (r['deliveredRateP50'] / f * 1e9),
            '%.1fM' % (tot / 1e6)]
names = ['deaths', 'median (cumulative)', 'median (live)', 'P10', 'best lane',
         'live max', 'positivity', 'SPS', 'reward/GFLOP (median)', 'fleet total (all lanes)']
cv, kv = fmt(c, ct, cf), fmt(k, kt, kf)
print(f'| @{step//1000}k | candidate | keeper |')
print('|---|---|---|')
for n, a, b in zip(names, cv, kv):
    print(f'| {n} | {a} | {b} |')
