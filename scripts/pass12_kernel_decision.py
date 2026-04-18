#!/usr/bin/env python3
"""Pass-12 Tier 3: per-kernel keep/remove decision under the multi-signal Pareto rule.

For each kernel that has a corresponding ablation bench dir under
.oak-benchmarks/pass12-ablation/<kernel>/, compute the per-channel paired-t Δ vs the baseline
bench and apply the AGENTS.md Hard Rule: a kernel is REMOVED iff its ablation does NOT hurt
at p < 0.05 on at least one of the four agent-side channels (rewardRate, LPM, utilFeat, pEff).
agent_ms doesn't count as an EARNING vote since keeping a kernel COSTS FLOPs.

Usage:
  python3 scripts/pass12_kernel_decision.py [--ablation-dir=DIR] [--baseline-dir=DIR]

Outputs to stdout (or --output=PATH): a markdown table ranking kernels by total earned
channels, plus a final list of REMOVE candidates and KEEP candidates.
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def welch_t(xs, ys):
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
    """Return {env: {seed: {final_rewardRate, mean_rewardRate, agent_ms,
                            outcome_error_traj, util_traj, rank_min}}}."""
    out = defaultdict(dict)
    for env_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        env = env_dir.name
        for seed_dir in sorted(env_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
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


def per_kernel_signals(baseline_data, ablated_data):
    """Compute per-channel paired-t aggregated across envs:
       returns dict{channel -> dict{env -> (delta, p)}}.
       Δ = baseline_oak - ablated_oak (positive = baseline-with-kernel does BETTER → kernel earns).
    """
    out = {ch: {} for ch in ('rewardRate', 'LPM_traj_mean', 'util_traj_mean', 'rank_min', 'agent_ms')}
    common_envs = sorted(set(baseline_data) & set(ablated_data))
    for env in common_envs:
        b = baseline_data[env]; a = ablated_data[env]
        common_seeds = sorted(set(b) & set(a))
        if len(common_seeds) < 2:
            continue
        # rewardRate channel
        bx = [b[s]['mean_rewardRate'] for s in common_seeds]
        ax = [a[s]['mean_rewardRate'] for s in common_seeds]
        d, t, p = welch_t(bx, ax)
        out['rewardRate'][env] = (d, p)
        # LPM channel: lower outcomeError trajectory mean = better
        bx = [-(sum(b[s]['outcome_error_traj']) / len(b[s]['outcome_error_traj']))
              for s in common_seeds if b[s]['outcome_error_traj']]
        ax = [-(sum(a[s]['outcome_error_traj']) / len(a[s]['outcome_error_traj']))
              for s in common_seeds if a[s]['outcome_error_traj']]
        if len(bx) >= 2 and len(ax) >= 2:
            d, t, p = welch_t(bx, ax)
            out['LPM_traj_mean'][env] = (d, p)
        # utilFeat channel: higher = better
        bx = [sum(b[s]['util_traj']) / len(b[s]['util_traj'])
              for s in common_seeds if b[s]['util_traj']]
        ax = [sum(a[s]['util_traj']) / len(a[s]['util_traj'])
              for s in common_seeds if a[s]['util_traj']]
        if len(bx) >= 2 and len(ax) >= 2:
            d, t, p = welch_t(bx, ax)
            out['util_traj_mean'][env] = (d, p)
        # rank_min channel: higher = better
        bx = [b[s]['rank_min'] for s in common_seeds if b[s]['rank_min'] is not None]
        ax = [a[s]['rank_min'] for s in common_seeds if a[s]['rank_min'] is not None]
        if len(bx) >= 2 and len(ax) >= 2:
            d, t, p = welch_t(bx, ax)
            out['rank_min'][env] = (d, p)
        # agent_ms: lower with-kernel costs MORE — but for keep/remove, we want to know if
        # ablating the kernel saves FLOPs (agent_ms drops without it).
        bx = [b[s]['agent_ms'] for s in common_seeds if b[s]['agent_ms'] is not None]
        ax = [a[s]['agent_ms'] for s in common_seeds if a[s]['agent_ms'] is not None]
        if len(bx) >= 2 and len(ax) >= 2:
            d, t, p = welch_t(bx, ax)
            out['agent_ms'][env] = (d, p)
    return out


def keep_decision(signals, p_threshold=0.05):
    """A kernel is KEPT iff it earns ≥1 of the 4 agent-side channels significantly on ≥1 env
    (i.e., baseline_with_kernel > ablated_without_kernel, p < threshold)."""
    earning_envs = {ch: 0 for ch in ('rewardRate', 'LPM_traj_mean', 'util_traj_mean', 'rank_min')}
    hurt_envs = {ch: 0 for ch in earning_envs}
    for ch in earning_envs:
        for env, (d, p) in signals.get(ch, {}).items():
            if p < p_threshold and d > 0:
                earning_envs[ch] += 1
            elif p < p_threshold and d < 0:
                hurt_envs[ch] += 1
    earned_total = sum(earning_envs.values())
    hurt_total = sum(hurt_envs.values())
    keep = earned_total >= 1
    return keep, earning_envs, hurt_envs, earned_total, hurt_total


def main(argv):
    ablation_root = Path('.oak-benchmarks/pass12-ablation')
    baseline_dir = ablation_root / 'baseline'
    output = None
    p_threshold = 0.05
    for a in argv[1:]:
        if a.startswith('--ablation-dir='): ablation_root = Path(a.split('=', 1)[1])
        elif a.startswith('--baseline-dir='): baseline_dir = Path(a.split('=', 1)[1])
        elif a.startswith('--output='): output = Path(a.split('=', 1)[1])
        elif a.startswith('--p='): p_threshold = float(a.split('=', 1)[1])
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
        if not a_data:
            continue
        sigs = per_kernel_signals(base_data, a_data)
        keep, earned, hurt, et, ht = keep_decision(sigs, p_threshold)
        # agent_ms cost: positive Δ = baseline (with kernel) is SLOWER → ablation saves FLOPs
        ams_envs = sigs.get('agent_ms', {})
        ms_savings = sum(d for d, _ in ams_envs.values()) / max(len(ams_envs), 1) if ams_envs else 0.0
        rows.append((kname, keep, et, ht, earned, hurt, ms_savings))
        (keep_list if keep else remove_list).append(kname)

    # Sort by net earning (earned - hurt), descending
    rows.sort(key=lambda r: -(r[2] - r[3]))

    out = []
    out.append("# Pass 12 Tier 3: per-kernel keep/remove decision (multi-signal Pareto)")
    out.append("")
    out.append(f"Baseline: `{baseline_dir}`")
    out.append(f"Ablation root: `{ablation_root}`")
    out.append(f"Significance threshold: p < {p_threshold}")
    out.append("")
    out.append("## Decision rule (AGENTS.md Multi-signal Pareto Hard Rule)")
    out.append("")
    out.append("A kernel is KEPT iff its ablation HURTS at least one of the four agent-side channels")
    out.append("(rewardRate, LPM_traj_mean, util_traj_mean, rank_min) at p < threshold on at least one")
    out.append("env. Otherwise REMOVED — the kernel is not earning its FLOPs.")
    out.append("")
    out.append("## Per-kernel decision table")
    out.append("")
    out.append("| kernel | decision | net earned envs | earned (rR/LPM/util/rank) | hurt (rR/LPM/util/rank) | mean Δagent_ms (ms; positive = ablation saves FLOPs) |")
    out.append("|---|---|---|---|---|---|")
    for kname, keep, et, ht, earned, hurt, ms in rows:
        decision = "**KEEP**" if keep else "**REMOVE**"
        e_str = f"{earned['rewardRate']}/{earned['LPM_traj_mean']}/{earned['util_traj_mean']}/{earned['rank_min']}"
        h_str = f"{hurt['rewardRate']}/{hurt['LPM_traj_mean']}/{hurt['util_traj_mean']}/{hurt['rank_min']}"
        out.append(f"| {kname} | {decision} | {et - ht:+d} | {e_str} | {h_str} | {ms:+.4f} |")

    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"- KEEP: {len(keep_list)} kernels: {', '.join(sorted(keep_list))}")
    out.append(f"- REMOVE: {len(remove_list)} kernels: {', '.join(sorted(remove_list))}")
    text = "\n".join(out)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
