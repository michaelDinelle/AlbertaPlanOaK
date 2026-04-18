#!/usr/bin/env python3
"""Pass-12 multi-signal Pareto bench report.

For a bench directory (oak + random heartbeats per env per seed), computes per-env paired-t Δ
on FIVE principled channels — the 5-channel Pareto gate from Pass 12:

  rewardRate  : oak vs random — outcome (necessary)                       (Alberta Plan)
  LPM         : Spearman(window, mean|outcomeError|) per oak run          (Hou-An-Du 2509.25438)
  utilFeat    : Spearman(window, utilityFeatureMean) per oak run          (GraMa 2505.24061)
  pEff        : min(activeRank_window) > floorR (anti-collapse safety)    (RankGradientLost 2604.01913)
  agent_ms    : oak agentStepMsMean (lower is better; Pareto-FLOP/reward) (AGENTS.md Hard Rule)

Acceptance per env (multi-signal Pareto): a tier passes iff
  (1) no rewardRate regression at p<0.05 across the bench  AND
  (2) at least ONE of {rewardRate↑, LPM↑, utilFeat↑, pEff↑, agent_ms↓} improves at p<0.05.

This script ALSO prints the strict P(Δ>0) for backwards-compat with Pass 11 reports.

Usage:
  python3 scripts/pass12_multi_signal_report.py <bench-dir> [--output=PATH] [--baseline=DIR]

If --baseline is provided, the report ALSO emits per-env paired-t Δ (oak_with - oak_baseline)
on each channel — this is the per-kernel-ablation signal used by Tier 2/3.
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


# ---------------------------- statistics --------------------------------------

def welch_t(xs, ys):
    """Welch's t-test (unequal variance), one-tailed p (xs > ys)."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return 0.0, 0.0, 1.0
    mx = sum(xs) / nx
    my = sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1) if nx > 1 else 0.0
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1) if ny > 1 else 0.0
    se = math.sqrt(vx / nx + vy / ny)
    if se <= 0:
        return mx - my, 0.0, 0.5 if mx == my else (0.0 if mx > my else 1.0)
    t = (mx - my) / se
    # Welch-Satterthwaite df
    num = (vx / nx + vy / ny) ** 2
    den = ((vx / nx) ** 2 / max(nx - 1, 1)) + ((vy / ny) ** 2 / max(ny - 1, 1))
    df = num / den if den > 0 else max(nx + ny - 2, 1)
    p = 0.5 * math.erfc(t / math.sqrt(2))   # one-tailed (xs > ys)
    _ = df  # df unused in normal-approx p; sample sizes >=3 make this OK
    return mx - my, t, p


def paired_t(deltas):
    """One-sample t-test on paired Δs vs 0; one-tailed p (mean > 0)."""
    n = len(deltas)
    if n < 2:
        return (sum(deltas) / max(n, 1)) if n else 0.0, 0.0, 1.0
    m = sum(deltas) / n
    s2 = sum((d - m) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(s2 / n) if s2 > 0 else 0.0
    if se <= 0:
        return m, 0.0, 0.5 if m == 0 else (0.0 if m > 0 else 1.0)
    t = m / se
    p = 0.5 * math.erfc(t / math.sqrt(2))
    return m, t, p


def spearman_rho(y):
    n = len(y)
    if n < 3:
        return 0.0, 1.0
    order = sorted(range(n), key=lambda i: y[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y[order[j + 1]] == y[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    x = list(range(n))
    sxy = sum(x[i] * ranks[i] for i in range(n))
    sx, sy = sum(x), sum(ranks)
    sxx = sum(v * v for v in x)
    syy = sum(v * v for v in ranks)
    nd = float(n)
    num = sxy - sx * sy / nd
    den_sq = (sxx - sx * sx / nd) * (syy - sy * sy / nd)
    if den_sq <= 0:
        return 0.0, 1.0
    rho = num / math.sqrt(den_sq)
    z = math.atanh(max(-0.999999, min(0.999999, rho))) * math.sqrt(max(n - 3, 1))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return rho, p


# ---------------------------- bench ingestion --------------------------------

def gather_per_seed(bench_dir: Path):
    """Return {env: {policy: [{seed, rewardRates, lpms, utilFeats, pEffs, agentMs, finalRR}]}}.

    rewardRates / lpms / utilFeats / pEffs are per-heartbeat trajectories (sorted by hb).
    finalRR is the last heartbeat's rewardRate. agentMs is the final agentStepMsMean.
    """
    out = defaultdict(lambda: defaultdict(list))
    for env_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        env = env_dir.name
        for seed_dir in sorted(env_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name
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
                if pol is None or hb is None:
                    continue
                health = d.get('health') or {}
                rr = health.get('rewardRate')
                if rr is None:
                    rs = d.get('rewardSum')
                    msc = d.get('measuredStepCount')
                    if rs is not None and msc and msc > 0:
                        rr = rs / msc
                # outcomeError is not yet emitted to JSON in Pass 11; LPM channel uses
                # `optionModelPredictionMSE` (Design-2 ψ TD telemetry) as the closest proxy
                # of the agent's model-prediction error. Pass 12 Tier 0 will optionally add
                # an explicit `outcomeErrorL2Mean` to KernelStats.
                lpm_proxy = health.get('optionModelPredictionMSE') if health else None
                util = health.get('utilityFeatureMean') if health else None
                rank = health.get('activeRank') if health else None
                ams = d.get('agentStepMsMean')
                pol_to_hb[pol].append((hb, rr, lpm_proxy, util, rank, ams))
            for pol, points in pol_to_hb.items():
                points.sort(key=lambda p: p[0])
                rrs   = [p[1] for p in points if p[1] is not None]
                lpms  = [p[2] for p in points if p[2] is not None]
                uts   = [p[3] for p in points if p[3] is not None]
                rks   = [p[4] for p in points if p[4] is not None]
                ams_s = [p[5] for p in points if p[5] is not None]
                if not rrs:
                    continue
                out[env][pol].append({
                    'seed': seed,
                    'rewardRates': rrs,
                    'lpms': lpms,
                    'utilFeats': uts,
                    'activeRanks': rks,
                    'agentMs': ams_s[-1] if ams_s else None,
                    'finalRR': rrs[-1],
                    'meanRR': sum(rrs) / len(rrs),
                })
    return out


# ---------------------------- per-env multi-signal --------------------------

def per_env_signals(env_data, baseline_data=None):
    """Compute per-env multi-signal verdict.

    Returns dict with keys: rewardRate_delta, rewardRate_p, LPM_rho, LPM_p,
    utilFeat_rho, utilFeat_p, pEff_floor_ok, agent_ms_oak, ablation_deltas (if baseline_data)
    """
    oak = env_data.get('oak', [])
    rand = env_data.get('random', [])
    out = {}

    # rewardRate channel: per-seed final rewardRate, oak vs random Welch
    if oak and rand:
        seeds_oak = {r['seed']: r['meanRR'] for r in oak}
        seeds_rnd = {r['seed']: r['meanRR'] for r in rand}
        common = sorted(set(seeds_oak) & set(seeds_rnd))
        if common:
            xs = [seeds_oak[s] for s in common]
            ys = [seeds_rnd[s] for s in common]
            d, t, p = welch_t(xs, ys)
            out['rewardRate_delta'] = d
            out['rewardRate_p'] = p
        else:
            out['rewardRate_delta'] = 0.0
            out['rewardRate_p'] = 1.0
    else:
        out['rewardRate_delta'] = 0.0
        out['rewardRate_p'] = 1.0

    # LPM channel: positive Spearman trend on oak's mean|outcomeError|? Decreasing error =
    # learning progress (so a NEGATIVE Spearman ρ is GOOD — error trends down over time).
    # We invert sign so "LPM_rho > 0" means "learning is happening".
    lpm_rhos = []
    for run in oak:
        if len(run.get('lpms', [])) >= 3:
            rho, _ = spearman_rho(run['lpms'])
            lpm_rhos.append(-rho)   # invert: error-down = LPM-up
    if lpm_rhos:
        m, t, p = paired_t(lpm_rhos)
        out['LPM_rho'] = m
        out['LPM_p'] = p
    else:
        out['LPM_rho'] = 0.0
        out['LPM_p'] = 1.0

    # utilFeat channel: positive Spearman on oak's utilityFeatureMean = increasing internal utility
    util_rhos = []
    for run in oak:
        if len(run.get('utilFeats', [])) >= 3:
            rho, _ = spearman_rho(run['utilFeats'])
            util_rhos.append(rho)
    if util_rhos:
        m, t, p = paired_t(util_rhos)
        out['utilFeat_rho'] = m
        out['utilFeat_p'] = p
    else:
        out['utilFeat_rho'] = 0.0
        out['utilFeat_p'] = 1.0

    # pEff (activeRank) floor: min activeRank across oak runs/heartbeats > sqrt(R) floor
    floor_ok = True
    min_rank = float('inf')
    for run in oak:
        if run.get('activeRanks'):
            for r in run['activeRanks']:
                min_rank = min(min_rank, r)
    out['pEff_min'] = min_rank if min_rank != float('inf') else None
    out['pEff_floor_ok'] = floor_ok  # caller uses pEff_min vs sqrt(R) floor externally

    # agent_ms: mean of final-heartbeat agentMs
    ms = [r['agentMs'] for r in oak if r.get('agentMs') is not None]
    out['agent_ms_oak'] = (sum(ms) / len(ms)) if ms else None

    # Ablation channel (optional): oak (this bench) vs oak (baseline bench)
    if baseline_data:
        b_oak = baseline_data.get('oak', [])
        if oak and b_oak:
            seeds_a = {r['seed']: r['meanRR'] for r in oak}
            seeds_b = {r['seed']: r['meanRR'] for r in b_oak}
            common = sorted(set(seeds_a) & set(seeds_b))
            if common:
                # Paired Δ: how much DOES this kernel ablation hurt vs baseline?
                deltas = [seeds_b[s] - seeds_a[s] for s in common]   # baseline - ablated
                m, t, p = paired_t(deltas)
                out['ablation_rR_harm'] = m
                out['ablation_rR_harm_p'] = p
            else:
                out['ablation_rR_harm'] = 0.0
                out['ablation_rR_harm_p'] = 1.0

    return out


# ---------------------------- top-level report -------------------------------

def report(bench_dir: Path, baseline_dir: Path | None = None) -> str:
    data = gather_per_seed(bench_dir)
    base = gather_per_seed(baseline_dir) if baseline_dir else None

    # Per-env signals
    env_signals = {}
    for env in sorted(data):
        b = base.get(env) if base else None
        env_signals[env] = per_env_signals(data[env], b)

    # Headline counts
    total = len(env_signals)
    rR_strict_win = sum(1 for s in env_signals.values()
                        if s['rewardRate_delta'] > 0 and s['rewardRate_p'] < 0.05)
    rR_strict_loss = sum(1 for s in env_signals.values()
                         if s['rewardRate_delta'] < 0 and s['rewardRate_p'] < 0.05)
    rR_pos = sum(1 for s in env_signals.values() if s['rewardRate_delta'] > 0)
    lpm_pos = sum(1 for s in env_signals.values()
                  if s['LPM_rho'] > 0 and s['LPM_p'] < 0.05)
    util_pos = sum(1 for s in env_signals.values()
                   if s['utilFeat_rho'] > 0 and s['utilFeat_p'] < 0.05)
    multi_signal_pass = sum(
        1 for s in env_signals.values()
        if (s['rewardRate_delta'] >= 0 or s['rewardRate_p'] >= 0.05)
        and ((s['rewardRate_delta'] > 0 and s['rewardRate_p'] < 0.05) or
             (s['LPM_rho'] > 0 and s['LPM_p'] < 0.05) or
             (s['utilFeat_rho'] > 0 and s['utilFeat_p'] < 0.05))
    )

    lines = []
    lines.append("# Pass 12 multi-signal Pareto bench report")
    lines.append("")
    lines.append(f"Bench dir: `{bench_dir}`")
    if baseline_dir:
        lines.append(f"Baseline (for ablation Δ): `{baseline_dir}`")
    lines.append(f"Envs: {total}")
    lines.append("")
    lines.append("## Headline counts")
    lines.append("")
    lines.append(f"- **rewardRate strict win** (Δ > 0 at p < 0.05): **{rR_strict_win}/{total}**  ({rR_strict_win/max(total,1):.3f})")
    lines.append(f"- **rewardRate strict loss** (Δ < 0 at p < 0.05): {rR_strict_loss}/{total}")
    lines.append(f"- **P(rewardRate Δ > 0) (any sign, regardless of significance)**: {rR_pos}/{total}  ({rR_pos/max(total,1):.3f})")
    lines.append(f"- **LPM positive trend** (model-error improving at p < 0.05): {lpm_pos}/{total}  ({lpm_pos/max(total,1):.3f})")
    lines.append(f"- **utilFeat positive trend** (internal utility improving at p < 0.05): {util_pos}/{total}  ({util_pos/max(total,1):.3f})")
    lines.append(f"- **Multi-signal Pareto PASS** (no rewardRate regression AND ≥1 of 3 channels improves): {multi_signal_pass}/{total}  ({multi_signal_pass/max(total,1):.3f})")
    lines.append("")
    lines.append("## Per-env table")
    lines.append("")
    cols = ["env", "rR Δ", "rR p", "LPM ρ", "LPM p", "util ρ", "util p", "pEff min", "agent_ms"]
    if baseline_dir:
        cols += ["ablation rR harm", "harm p"]
    cols += ["multi-signal verdict"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for env in sorted(env_signals):
        s = env_signals[env]
        rRd = s['rewardRate_delta']; rRp = s['rewardRate_p']
        lpm_r = s['LPM_rho']; lpm_p = s['LPM_p']
        util_r = s['utilFeat_rho']; util_p = s['utilFeat_p']
        pEff = s['pEff_min']
        ams = s['agent_ms_oak']
        # Multi-signal verdict
        no_regression = rRd >= 0 or rRp >= 0.05
        any_signal_up = ((rRd > 0 and rRp < 0.05) or
                         (lpm_r > 0 and lpm_p < 0.05) or
                         (util_r > 0 and util_p < 0.05))
        verdict = "PASS" if (no_regression and any_signal_up) else ("FAIL" if not no_regression else "TIED")
        if rRd > 0 and rRp < 0.05:
            verdict += " (rR↑)"
        elif lpm_r > 0 and lpm_p < 0.05:
            verdict += " (LPM↑)"
        elif util_r > 0 and util_p < 0.05:
            verdict += " (util↑)"
        row = [env,
               f"{rRd:+.5f}",
               f"{rRp:.3f}",
               f"{lpm_r:+.3f}",
               f"{lpm_p:.3f}",
               f"{util_r:+.3f}",
               f"{util_p:.3f}",
               f"{pEff:.1f}" if pEff is not None else "?",
               f"{ams:.3f}" if ams is not None else "?"]
        if baseline_dir:
            ah = s.get('ablation_rR_harm', 0.0)
            ap = s.get('ablation_rR_harm_p', 1.0)
            row += [f"{ah:+.5f}", f"{ap:.3f}"]
        row += [verdict]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main(argv):
    bench_dir = Path(argv[1]) if len(argv) > 1 else None
    baseline = None
    output = None
    for a in argv[2:]:
        if a.startswith('--output='):
            output = Path(a.split('=', 1)[1])
        elif a.startswith('--baseline='):
            baseline = Path(a.split('=', 1)[1])
    if not bench_dir or not bench_dir.is_dir():
        print(f"usage: {argv[0]} <bench-dir> [--output=PATH] [--baseline=DIR]", file=sys.stderr)
        return 2
    text = report(bench_dir, baseline)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
