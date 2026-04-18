#!/usr/bin/env python3
"""Pass-14: per-kernel keep/remove decision under the >=2-of-5-channels strict rule.

Tightens Pass-12's lenient >=1-channel rule. Forces every kernel to contribute on at least
TWO of {rewardRate, LPM, utilFeat, pEff, agent_ms} at p<0.05 — kernels that only show harm
on a single channel must be re-designed (or removed).

For each kernel that has a corresponding ablation bench dir, compute per-channel paired-t Δ
vs the baseline bench and apply the rule:

  channels = {rewardRate, LPM_traj_mean, util_traj_mean, rank_min, agent_ms}
  harmed_count(kernel) = | { ch | ablation arm strictly worse than baseline at p<0.05 } |
  KEEP iff harmed_count >= 2

For agent_ms, "ablation worse" means "ablation was FASTER" (so the kernel costs FLOPs but
doesn't earn enough — agent_ms harm is a sign the kernel is doing real GPU work, which is
necessary but not sufficient for KEEP).

Usage:
  python3 scripts/pass14_kernel_decision.py [--ablation-dir=DIR] [--baseline-dir=DIR]
                                            [--output=PATH] [--p=0.05]

Outputs to stdout (or --output=PATH): a markdown table of per-kernel verdict + per-channel
harm + the >=2-channel decision.
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def welch_t(xs, ys):
    """Welch's t-test (unequal variance), one-tailed p (xs > ys)."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return 0.0, 0.0, 1.0
    mx = sum(xs) / nx; my = sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1) if nx > 1 else 0.0
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1) if ny > 1 else 0.0
    se = math.sqrt(vx / nx + vy / ny)
    if se <= 0:
        return mx - my, 0.0, 0.5 if mx == my else (0.0 if mx > my else 1.0)
    t = (mx - my) / se
    p = 0.5 * math.erfc(t / math.sqrt(2))
    return mx - my, t, p


def gather_oak(bench_dir: Path):
    out = defaultdict(dict)
    for env_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        env = env_dir.name
        for seed_dir in sorted(env_dir.iterdir()):
            if not seed_dir.is_dir(): continue
            seed = seed_dir.name
            hbs = []
            for f in sorted(seed_dir.iterdir()):
                if not f.name.endswith('.json'): continue
                try: d = json.load(open(f))
                except Exception: continue
                if d.get('policy') != 'oak': continue
                hb = d.get('heartbeatSequence')
                if hb is None: continue
                health = d.get('health') or {}
                rr = health.get('rewardRate')
                if rr is None:
                    rs = d.get('rewardSum'); msc = d.get('measuredStepCount')
                    if rs is not None and msc and msc > 0: rr = rs / msc
                hbs.append({
                    'hb': hb, 'rr': rr,
                    'lpm': health.get('optionModelPredictionMSE'),
                    'util': health.get('utilityFeatureMean'),
                    'rank': health.get('activeRank'),
                    'agent_ms': d.get('agentStepMsMean'),
                })
            if not hbs: continue
            hbs.sort(key=lambda x: x['hb'])
            rrs = [x['rr'] for x in hbs if x['rr'] is not None]
            lpms = [x['lpm'] for x in hbs if x['lpm'] is not None]
            uts = [x['util'] for x in hbs if x['util'] is not None]
            rks = [x['rank'] for x in hbs if x['rank'] is not None]
            ams = [x['agent_ms'] for x in hbs if x['agent_ms'] is not None]
            if not rrs: continue
            out[env][seed] = {
                'mean_rewardRate': sum(rrs) / len(rrs),
                'final_rewardRate': rrs[-1],
                'agent_ms': ams[-1] if ams else None,
                'outcome_error_traj': lpms,
                'util_traj': uts,
                'rank_min': min(rks) if rks else None,
            }
    return out


def per_channel_harm(baseline_data, ablated_data, p_threshold):
    """For each of 5 channels, count how many envs show ablation strictly worse than baseline."""
    channels = ('rewardRate', 'LPM', 'utilFeat', 'pEff', 'agent_ms')
    harmed = {ch: [] for ch in channels}
    common_envs = sorted(set(baseline_data) & set(ablated_data))
    for env in common_envs:
        b = baseline_data[env]; a = ablated_data[env]
        common_seeds = sorted(set(b) & set(a))
        if len(common_seeds) < 2: continue

        # rewardRate: higher is better; ablation worse = lower → t test b > a
        bx = [b[s]['mean_rewardRate'] for s in common_seeds]
        ax = [a[s]['mean_rewardRate'] for s in common_seeds]
        d, t, p = welch_t(bx, ax)
        if d > 0 and p < p_threshold: harmed['rewardRate'].append(env)

        # LPM: lower outcomeError = better. Ablation worse = higher outcomeError.
        bx = [-(sum(b[s]['outcome_error_traj']) / len(b[s]['outcome_error_traj']))
              for s in common_seeds if b[s]['outcome_error_traj']]
        ax = [-(sum(a[s]['outcome_error_traj']) / len(a[s]['outcome_error_traj']))
              for s in common_seeds if a[s]['outcome_error_traj']]
        if len(bx) >= 2 and len(ax) >= 2:
            d, t, p = welch_t(bx, ax)
            if d > 0 and p < p_threshold: harmed['LPM'].append(env)

        # utilFeat: higher = better.
        bx = [sum(b[s]['util_traj']) / len(b[s]['util_traj'])
              for s in common_seeds if b[s]['util_traj']]
        ax = [sum(a[s]['util_traj']) / len(a[s]['util_traj'])
              for s in common_seeds if a[s]['util_traj']]
        if len(bx) >= 2 and len(ax) >= 2:
            d, t, p = welch_t(bx, ax)
            if d > 0 and p < p_threshold: harmed['utilFeat'].append(env)

        # pEff: higher = better. Ablation worse = lower rank_min.
        bx = [b[s]['rank_min'] for s in common_seeds if b[s]['rank_min'] is not None]
        ax = [a[s]['rank_min'] for s in common_seeds if a[s]['rank_min'] is not None]
        if len(bx) >= 2 and len(ax) >= 2:
            d, t, p = welch_t(bx, ax)
            if d > 0 and p < p_threshold: harmed['pEff'].append(env)

        # agent_ms: lower = better. Ablation worse = ablation FASTER (lower agent_ms) which means
        # the kernel was doing real GPU work. We test b > a (baseline slower with kernel on).
        bx = [b[s]['agent_ms'] for s in common_seeds if b[s]['agent_ms'] is not None]
        ax = [a[s]['agent_ms'] for s in common_seeds if a[s]['agent_ms'] is not None]
        if len(bx) >= 2 and len(ax) >= 2:
            d, t, p = welch_t(bx, ax)
            if d > 0 and p < p_threshold: harmed['agent_ms'].append(env)

    return harmed


def kernel_earns_flops(harmed, min_channels=2):
    """A kernel earns its FLOPs iff it harms >=min_channels of 5 channels on >=1 env."""
    channels_with_any_harm = sum(1 for ch, envs in harmed.items() if len(envs) > 0)
    return channels_with_any_harm >= min_channels


def main(argv):
    ablation_root = Path('.oak-benchmarks/pass14-tier0-ablation')
    baseline_dir = ablation_root / 'baseline'
    output = None
    p_threshold = 0.05
    min_channels = 2
    for a in argv[1:]:
        if a.startswith('--ablation-dir='): ablation_root = Path(a.split('=', 1)[1])
        elif a.startswith('--baseline-dir='): baseline_dir = Path(a.split('=', 1)[1])
        elif a.startswith('--output='): output = Path(a.split('=', 1)[1])
        elif a.startswith('--p='): p_threshold = float(a.split('=', 1)[1])
        elif a.startswith('--min-channels='): min_channels = int(a.split('=', 1)[1])
    if not baseline_dir.is_dir():
        print(f"baseline dir not found: {baseline_dir}", file=sys.stderr); return 2
    base_data = gather_oak(baseline_dir)

    rows = []
    keep_list = []
    remove_list = []
    for kdir in sorted(p for p in ablation_root.iterdir() if p.is_dir()):
        kname = kdir.name
        if kname == 'baseline': continue
        a_data = gather_oak(kdir)
        if not a_data: continue
        harmed = per_channel_harm(base_data, a_data, p_threshold)
        keep = kernel_earns_flops(harmed, min_channels)
        rows.append((kname, keep, harmed))
        (keep_list if keep else remove_list).append(kname)

    # Sort: kept kernels first (most harm), then removed.
    rows.sort(key=lambda r: (-int(r[1]), -sum(len(envs) for envs in r[2].values())))

    out = []
    out.append(f"# Pass 14: per-kernel keep/remove decision (>={min_channels}-of-5-channels rule)")
    out.append("")
    out.append(f"Baseline: `{baseline_dir}`")
    out.append(f"Ablation root: `{ablation_root}`")
    out.append(f"Significance: p < {p_threshold} | Min harmed channels: {min_channels}/5")
    out.append("")
    out.append("## Decision rule")
    out.append("")
    out.append(f"A kernel is KEPT iff its ablation HURTS at least **{min_channels} of 5** channels")
    out.append("(rewardRate, LPM, utilFeat, pEff, agent_ms) at p<0.05 on at least one env. ")
    out.append("Otherwise REMOVED — single-channel kernels do not earn their FLOPs under Pass 14's")
    out.append("strict rule (Pass 12 used a lenient >=1 rule that let SPR-like kernels through")
    out.append("with only LPM contribution).")
    out.append("")
    out.append("## Per-kernel decision table")
    out.append("")
    out.append("| kernel | decision | channels harmed | rR envs | LPM envs | utilFeat envs | pEff envs | agent_ms envs |")
    out.append("|---|---|---|---|---|---|---|---|")
    for kname, keep, harmed in rows:
        decision = "**KEEP**" if keep else "**REMOVE**"
        ch_count = sum(1 for envs in harmed.values() if len(envs) > 0)
        out.append(f"| {kname} | {decision} | {ch_count}/5 | "
                   f"{len(harmed['rewardRate'])} | {len(harmed['LPM'])} | "
                   f"{len(harmed['utilFeat'])} | {len(harmed['pEff'])} | "
                   f"{len(harmed['agent_ms'])} |")

    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"- KEEP ({len(keep_list)}): {', '.join(sorted(keep_list))}")
    out.append(f"- REMOVE ({len(remove_list)}): {', '.join(sorted(remove_list))}")
    text = "\n".join(out)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)
    # Exit code: 0 iff all kernels in the bench KEEP. Useful for CI.
    return 0 if not remove_list else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
