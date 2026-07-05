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
data = [row(rt, step) for rt in roots]
# WHITELIST comparability gate (user law 2026-07-05): a table renders ONLY
# when every field below is IDENTICAL across all roots. Anything not on the
# allowed-to-differ list (run name, startedAt) that differs = hard refusal.
# This is a whitelist by construction: new meta fields are compared by
# default; a field must be explicitly added to ALLOWED_DIFF to be exempt.
# ablation is the TREATMENT variable — candidate vs keeper legitimately
# differ in it (user ruling 2026-07-05); it stays loudly visible in the
# config row. Everything else fail-closed.
ALLOWED_DIFF = {'startedAt', 'ablation', 'ablationLabel'}
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
print('⤷ DISPLAY CONTRACT: paste this table VERBATIM into the reply text — tool output is not user-visible; keeper column mandatory; never summarize or say "table above".')
