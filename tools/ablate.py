#!/usr/bin/env python3
"""Organ-bisect harness: run an ablation matrix at one depth, emit the
conviction table. (Named ablate.py — bisect.py shadows the stdlib module.)
Usage:
  tools/ablate.py --steps 300000 --base planning generate-and-test option-keyboard
Each organ is ablated ON TOP of --base; a baseline (--base only) is included."""
import argparse, json, os, subprocess

def run_one(label, ablate, steps):
    root = f'.oak-benchmarks/ablate-{label}-{steps // 1000}k'
    subprocess.run(['rm', '-rf', root])
    env = dict(os.environ, OAK_ABLATE=ablate, OAK_BENCH_ENV='HalfCheetah-v4',
               OAK_BENCH_POLICIES='oak', OAK_BENCH_STEPS=str(steps),
               OAK_BENCH_SEED_COUNT='1024', OAK_STEPS_PER_CB='1024',
               OAK_METRICS_STORAGE=root)
    print(f'== {label} (OAK_ABLATE={ablate}) ==', flush=True)
    subprocess.run(['.build/release/OaK', 'benchmark'], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = [json.loads(l) for l in open(root + '/aggregate.jsonl')]
    return rows[-1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('organs', nargs='+')
    p.add_argument('--steps', type=int, default=300000)
    p.add_argument('--base', default='none')
    a = p.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    res = {'BASELINE': run_one('base', a.base, a.steps)}
    for o in a.organs:
        abl = o if a.base == 'none' else f'{a.base},{o}'
        res[o] = run_one(o, abl, a.steps)
    print('\nconfig | pos | P50 | recentP50 | best | dead')
    for k, r in res.items():
        print(f"{k:>18s} | {r['positiveRateFraction']:.3f} | {r['deliveredRateP50']:+.4f}"
              f" | {r['recentRateP50']:+.4f} | {r['bestLaneDeliveredRate']:.3f} | {r['dead']}")

if __name__ == '__main__':
    main()
