#!/usr/bin/env python3
"""Pass 15 final-sweep gate validation.

Reads all .json files (excluding heartbeat-*.json) from a benchmark directory and
verifies the (B) and (G) hard rules:

  (B) Numerical safety: nonFiniteRawTotal == 0 AND tdPsiNonFiniteTotal == 0 over
                        EVERY run.
  (G) P(rR > random):   fraction of (env, seed) pairs where oak's rewardRate
                        exceeds random's rewardRate; target > 0.50.

Usage: scripts/pass15_analyze.py <bench_dir>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def read_jsons(root: Path):
    """Yield (envId, seed, policy, payload) for each non-heartbeat JSON."""
    for p in root.rglob("*.json"):
        if "heartbeat-" in p.name:
            continue
        try:
            payload = json.loads(p.read_text())
        except Exception as e:
            print(f"WARN: skip {p}: {e}", file=sys.stderr)
            continue
        yield (payload.get("envId"), payload.get("seed"),
               payload.get("policy"), payload)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    runs = list(read_jsons(root))
    print(f"=== Pass 15 final-sweep analysis ({len(runs)} runs from {root}) ===\n")

    # Gate (B) numerical safety
    nf_violations = []
    for envId, seed, policy, p in runs:
        h = p.get("health") or {}
        nf_raw = int(h.get("nonFiniteRawTotal", 0) or 0)
        td_psi_nf = int(h.get("tdPsiNonFiniteTotal", 0) or 0)
        if nf_raw or td_psi_nf:
            nf_violations.append((envId, seed, policy, nf_raw, td_psi_nf))

    print(f"Gate (B) numerical safety: ", end="")
    if not nf_violations:
        print(f"PASS ({len(runs)} runs, all NF counters == 0)")
    else:
        oak_violations = [v for v in nf_violations if v[2] == "oak"]
        rand_violations = [v for v in nf_violations if v[2] == "random"]
        print(f"FAIL ({len(nf_violations)}/{len(runs)} runs;"
              f" oak={len(oak_violations)} random={len(rand_violations)})")
        for envId, seed, policy, nf, tp in nf_violations[:10]:
            print(f"  {policy:6s} {envId:50s} seed={seed:3} "
                  f"nfRaw={nf:8} tdPsiNF={tp:6}")
        if len(nf_violations) > 10:
            print(f"  ... and {len(nf_violations) - 10} more")

    # Gate (C) bounded magnitude
    MAGNITUDE_FIELDS = ["encoderMax", "valueWMax", "psiMax", "optPolicyMax",
                        "outcomeHeadMax", "betaWMax", "omegaWMax", "sprDecoderMax"]
    BOUND = 1000.0
    mag_violations = []
    for envId, seed, policy, p in runs:
        if policy != "oak":
            continue  # random policy doesn't touch oak weights
        h = p.get("health") or {}
        worst = {}
        for f in MAGNITUDE_FIELDS:
            v = h.get(f) or 0
            if v > BOUND:
                worst[f] = v
        if worst:
            mag_violations.append((envId, seed, worst))

    oak_runs = sum(1 for r in runs if r[2] == "oak")
    print(f"\nGate (C) bounded magnitude (< {BOUND}): ", end="")
    if not mag_violations:
        print(f"PASS ({oak_runs} oak runs, all *Max < {BOUND})")
    else:
        print(f"FAIL ({len(mag_violations)}/{oak_runs} oak runs)")
        for envId, seed, worst in mag_violations[:5]:
            ws = ", ".join(f"{f}={v:.1f}" for f, v in worst.items())
            print(f"  {envId:50s} seed={seed:3} {ws}")

    # Aggregate weight-magnitude statistics across oak runs
    if oak_runs > 0:
        per_field_max = {f: 0.0 for f in MAGNITUDE_FIELDS}
        per_field_count = {f: 0 for f in MAGNITUDE_FIELDS}
        per_field_sum = {f: 0.0 for f in MAGNITUDE_FIELDS}
        for envId, seed, policy, p in runs:
            if policy != "oak":
                continue
            h = p.get("health") or {}
            for f in MAGNITUDE_FIELDS:
                v = h.get(f)
                if v is None:
                    continue
                per_field_max[f] = max(per_field_max[f], v)
                per_field_sum[f] += v
                per_field_count[f] += 1
        print(f"\n  Weight-magnitude maxima across {oak_runs} oak runs:")
        for f in MAGNITUDE_FIELDS:
            mx = per_field_max[f]
            mn = per_field_sum[f] / max(1, per_field_count[f])
            print(f"    {f:18s} max={mx:8.2f}  mean={mn:8.2f}")

    # Gate (G) P(rR > random) at the FINAL heartbeat. Random runs don't have a `health`
    # block (no agent kernel), so use rewardSum / measuredStepCount as the random rR proxy.
    # Oak's rR is the ρ̄ fast EMA from KernelScalars — same scale as mean reward.
    by_env = defaultdict(dict)  # envId -> {(seed, policy): rR}
    for envId, seed, policy, p in runs:
        h = p.get("health") or {}
        rr = h.get("rewardRate")
        if rr is None:
            steps = p.get("measuredStepCount", 0)
            r_sum = p.get("rewardSum", 0)
            if steps > 0:
                rr = float(r_sum) / float(steps)
            else:
                continue
        by_env[envId][(seed, policy)] = float(rr)

    oak_wins, total_pairs = 0, 0
    per_env_wins = []
    for envId, lookup in by_env.items():
        seeds = {s for (s, _) in lookup}
        env_wins = 0
        env_total = 0
        for seed in seeds:
            oak = lookup.get((seed, "oak"))
            rnd = lookup.get((seed, "random"))
            if oak is None or rnd is None:
                continue
            env_total += 1
            if oak > rnd:
                env_wins += 1
        if env_total > 0:
            per_env_wins.append((envId, env_wins, env_total))
            oak_wins += env_wins
            total_pairs += env_total

    print(f"\nGate (G) P(rR > random): {oak_wins}/{total_pairs} "
          f"= {oak_wins/max(1,total_pairs):.3f} "
          f"(target > 0.50): "
          f"{'PASS' if oak_wins / max(1,total_pairs) > 0.50 else 'FAIL'}")

    # Per-env win-rate breakdown — show the most polarized 10
    per_env_wins.sort(key=lambda x: x[1] / max(1, x[2]), reverse=True)
    n = len(per_env_wins)
    print(f"\n  Top 10 envs by oak win-rate:")
    for envId, w, t in per_env_wins[:10]:
        print(f"    {envId:50s} {w}/{t}")
    print(f"\n  Bottom 10 envs by oak win-rate:")
    for envId, w, t in per_env_wins[-10:]:
        print(f"    {envId:50s} {w}/{t}")

    # Convergence: how many envs have oak.rR > 0 (signal of any learning)?
    oak_rr_positive = sum(1 for envId, lookup in by_env.items()
                          for (s, pol), rr in lookup.items()
                          if pol == "oak" and rr > 1e-3)
    oak_rr_total = sum(1 for envId, lookup in by_env.items()
                       for (s, pol), _ in lookup.items() if pol == "oak")
    print(f"\nConvergence proxy: oak rR > 1e-3 on "
          f"{oak_rr_positive}/{oak_rr_total} = "
          f"{oak_rr_positive / max(1, oak_rr_total):.3f} of (env, seed) pairs")


if __name__ == "__main__":
    main()
