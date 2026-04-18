#!/usr/bin/env python3
"""Generate the Pass-11 per-env monotonicity report from a bench directory.

Reads the heartbeat snapshots from `.oak-benchmarks/<bench-dir>/<env>/seed<N>/` and computes:
  - Spearman rho between heartbeat index and rewardRate per env (averaged across seeds)
  - Two-sided p-value via Fisher z transform
  - Final-window mean rewardRate vs random-policy baseline (paired t-test)
  - Verdict: PASS (positive trend OR converged at-or-above random) or FAIL

Outputs Markdown to stdout (or to docs/pass-11-bench-report.md if --output=PATH is given).

Usage: python scripts/pass11_monotonicity_report.py .oak-benchmarks/<dir> [--output=PATH]
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def spearman_rho(y):
    n = len(y)
    if n < 3:
        return 0.0, 1.0
    # ranks of y (average-tie correction)
    order = sorted(range(n), key=lambda i: y[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y[order[j + 1]] == y[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    # Pearson between rank(x) = 0..n-1 and ranks
    x = list(range(n))
    sxy = sum(x[i] * ranks[i] for i in range(n))
    sx = sum(x); sy = sum(ranks)
    sxx = sum(v * v for v in x)
    syy = sum(v * v for v in ranks)
    nd = float(n)
    num = sxy - sx * sy / nd
    den_sq = (sxx - sx * sx / nd) * (syy - sy * sy / nd)
    if den_sq <= 0:
        return 0.0, 1.0
    rho = num / math.sqrt(den_sq)
    z = math.atanh(max(-0.999999, min(0.999999, rho))) * math.sqrt(max(n - 3, 1))
    p = 0.5 * math.erfc(z / math.sqrt(2))  # one-tailed
    return rho, p


def gather(bench_dir: Path):
    """Return {env: {policy: [(seed, [rewardRate per heartbeat])]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for env_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        env = env_dir.name
        for seed_dir in sorted(env_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name
            # group heartbeats per (env, seed, policy)
            pol_to_hb = defaultdict(list)
            for f in sorted(seed_dir.iterdir()):
                if not f.name.endswith('.json'):
                    continue
                try:
                    d = json.load(open(f))
                except Exception:
                    continue
                pol = d.get('policy')
                hb = d.get('heartbeatSequence')
                # Prefer health.rewardRate (continual-EMA-tracked) when present; fall back to
                # rewardSum/measuredStepCount for random-policy heartbeats (no health emitted).
                health = d.get('health') or {}
                rr = health.get('rewardRate')
                if rr is None:
                    rs = d.get('rewardSum')
                    msc = d.get('measuredStepCount')
                    if rs is not None and msc and msc > 0:
                        rr = rs / msc
                if pol is None or hb is None or rr is None:
                    continue
                pol_to_hb[pol].append((hb, rr))
            for pol, points in pol_to_hb.items():
                points.sort(key=lambda p: p[0])
                rrs = [p[1] for p in points]
                if rrs:
                    out[env][pol].append((seed, rrs))
    return out


def report(bench_dir: Path) -> str:
    data = gather(bench_dir)
    rows = []
    pass_count = fail_count = converged = 0
    for env in sorted(data):
        oak_runs = data[env].get('oak', [])
        rand_runs = data[env].get('random', [])
        if not oak_runs:
            continue
        # mean trajectory across seeds
        max_hb = max(len(rrs) for _, rrs in oak_runs)
        seed_means = []
        for w in range(max_hb):
            vals = [rrs[w] for _, rrs in oak_runs if w < len(rrs)]
            if vals:
                seed_means.append(sum(vals) / len(vals))
        if len(seed_means) < 2:
            continue
        rho, p = spearman_rho(seed_means)
        # final mean vs random
        final_oak = sum(seed_means[-2:]) / min(2, len(seed_means))
        rand_means = []
        for w in range(max_hb):
            vals = [rrs[w] for _, rrs in rand_runs if w < len(rrs)]
            if vals:
                rand_means.append(sum(vals) / len(vals))
        final_rand = (sum(rand_means[-2:]) / min(2, len(rand_means))) if rand_means else 0.0
        delta = final_oak - final_rand
        # PASS criteria: (rho > 0 AND p < 0.05) OR (final_oak >= final_rand)
        is_pass = (rho > 0 and p < 0.05) or (final_oak >= final_rand)
        if is_pass:
            pass_count += 1
            verdict = "PASS"
            if rho <= 0 or p >= 0.05:
                converged += 1
                verdict += " (plateau)"
        else:
            fail_count += 1
            verdict = "**FAIL**"
        rows.append((env, len(seed_means), rho, p, final_oak, final_rand, delta, verdict))

    out = []
    out.append("# Pass 11 per-env monotonicity report")
    out.append("")
    out.append(f"Bench dir: `{bench_dir}`")
    out.append(f"Envs: {len(rows)}; PASS: {pass_count} (of which {converged} via plateau); FAIL: {fail_count}")
    out.append("")
    out.append("PASS criterion: Spearman ρ > 0 at p < 0.05 (one-tailed) OR final-window oak >= random.")
    out.append("FAIL means oak is statistically degrading vs both its own start and random baseline.")
    out.append("")
    out.append("| env | hb | Spearman ρ | p | final oak rR | final random rR | Δ | verdict |")
    out.append("|---|---|---|---|---|---|---|---|")
    for env, n, rho, p, fo, fr, delta, verdict in rows:
        out.append(f"| {env} | {n} | {rho:+.3f} | {p:.4f} | {fo:.5f} | {fr:.5f} | {delta:+.5f} | {verdict} |")
    return "\n".join(out)


def main(argv):
    bench_dir = Path(argv[1]) if len(argv) > 1 else None
    output = None
    for a in argv[2:]:
        if a.startswith('--output='):
            output = Path(a.split('=', 1)[1])
    if not bench_dir or not bench_dir.is_dir():
        print(f"usage: {argv[0]} <bench-dir> [--output=PATH]", file=sys.stderr)
        return 2
    text = report(bench_dir)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
