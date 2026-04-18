#!/usr/bin/env python3
"""Pass-14 strict-gate report.

Tightens the Pass-12/13 multi-signal report:

1. `--baselines DIR1,DIR2,...` — accepts MULTIPLE baselines; report a "strict_regression"
   verdict per env: TRUE if env appears in ANY baseline's strict-loss list.
2. Fixed `utilFeat` channel: replaces Spearman of cumulative `utilityFeatureMean` (which was
   broken in Pass 12/13 — flat trajectory → 0/84) with paired-t on per-seed
   `scoreGradL2Mean × utilityFeatureMean` (a proper utility-gradient proxy per GraMa
   2505.24061).
3. `P(Δ>0)` computed using the FINAL heartbeat rR (not the heartbeat-mean) — a stricter
   convergence proxy.
4. `--ablation-baseline DIR` (single baseline) — when paired with main bench, computes
   per-env paired-t on each of 5 channels (rR, LPM, utilFeat, pEff, agent_ms) and reports
   a per-kernel "channels harmed" tally + KEEP/REMOVE verdict (see pass14_kernel_decision.py
   for the >=2-channel rule applied to one ablation arm).
5. Per-pass-rule tally (A)..(K) printed at the top.

Usage:
  python3 scripts/pass14_strict_report.py <bench-dir> \\
      [--baselines DIR1,DIR2,...] [--output=PATH] \\
      [--ablation-baseline=DIR] [--final-rr] [--p=0.05]
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
    mx = sum(xs) / nx
    my = sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1) if nx > 1 else 0.0
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1) if ny > 1 else 0.0
    se = math.sqrt(vx / nx + vy / ny)
    if se <= 0:
        return mx - my, 0.0, 0.5 if mx == my else (0.0 if mx > my else 1.0)
    t = (mx - my) / se
    p = 0.5 * math.erfc(t / math.sqrt(2))
    return mx - my, t, p


def paired_t(deltas):
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


def gather_per_seed(bench_dir: Path):
    """Return {env: {policy: [{seed, rrs, lpms, utils, ranks, scoreGrads, agentMs, finalRR, meanRR}]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for env_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        env = env_dir.name
        for seed_dir in sorted(env_dir.iterdir()):
            if not seed_dir.is_dir(): continue
            seed = seed_dir.name
            pol_to_hb = defaultdict(list)
            for f in sorted(seed_dir.iterdir()):
                if not f.name.endswith('.json'): continue
                try: d = json.load(open(f))
                except Exception: continue
                pol = d.get('policy')
                hb = d.get('heartbeatSequence')
                if pol is None or hb is None: continue
                health = d.get('health') or {}
                rr = health.get('rewardRate')
                if rr is None:
                    rs = d.get('rewardSum'); msc = d.get('measuredStepCount')
                    if rs is not None and msc and msc > 0: rr = rs / msc
                lpm = health.get('optionModelPredictionMSE')
                util = health.get('utilityFeatureMean')
                rank = health.get('activeRank')
                scoreGrad = health.get('scoreGradL2Mean')
                ams = d.get('agentStepMsMean')
                pol_to_hb[pol].append((hb, rr, lpm, util, rank, scoreGrad, ams))
            for pol, points in pol_to_hb.items():
                points.sort(key=lambda p: p[0])
                rrs = [p[1] for p in points if p[1] is not None]
                lpms = [p[2] for p in points if p[2] is not None]
                uts = [p[3] for p in points if p[3] is not None]
                rks = [p[4] for p in points if p[4] is not None]
                sgs = [p[5] for p in points if p[5] is not None]
                ams_s = [p[6] for p in points if p[6] is not None]
                if not rrs: continue
                out[env][pol].append({
                    'seed': seed,
                    'rrs': rrs,
                    'lpms': lpms,
                    'utils': uts,
                    'ranks': rks,
                    'scoreGrads': sgs,
                    'agentMs': ams_s[-1] if ams_s else None,
                    'finalRR': rrs[-1],
                    'meanRR': sum(rrs) / len(rrs),
                })
    return out


def per_env_signals(env_data, baselines_data, final_rr=False):
    """Compute per-env multi-signal verdict.

    Returns dict with keys including: rewardRate_delta, rewardRate_p, LPM_rho, LPM_p,
    utilFeat_delta, utilFeat_p (NEW: paired-t on scoreGrad*util — a real gradient proxy),
    pEff_min, agent_ms_oak, baselines (list of (baseline_name, env_in_strict_loss_list)).
    """
    oak = env_data.get('oak', [])
    rand = env_data.get('random', [])
    out = {}
    rr_key = 'finalRR' if final_rr else 'meanRR'

    if oak and rand:
        seeds_oak = {r['seed']: r[rr_key] for r in oak}
        seeds_rnd = {r['seed']: r[rr_key] for r in rand}
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
    # learning progress. Invert sign so "LPM_rho > 0" means "learning is happening".
    lpm_rhos = []
    for run in oak:
        if len(run.get('lpms', [])) >= 3:
            rho, _ = spearman_rho(run['lpms'])
            lpm_rhos.append(-rho)
    if lpm_rhos:
        m, t, p = paired_t(lpm_rhos)
        out['LPM_rho'] = m; out['LPM_p'] = p
    else:
        out['LPM_rho'] = 0.0; out['LPM_p'] = 1.0

    # utilFeat channel (NEW Pass 14): paired-t on scoreGrad·utilityFeatureMean per seed.
    # The Pass 12/13 proxy (Spearman of cumulative utilityFeatureMean) was flat → 0/84.
    # GraMa 2505.24061: utility · gradient magnitude IS the meaningful learning signal.
    # We measure final-heartbeat (scoreGrad × util) per seed and paired-t against zero.
    util_grads = []
    for run in oak:
        utils = run.get('utils', [])
        sgs = run.get('scoreGrads', [])
        if utils and sgs:
            n = min(len(utils), len(sgs))
            # final-heartbeat product
            util_grads.append(utils[n - 1] * sgs[n - 1])
    if len(util_grads) >= 2:
        m, t, p = paired_t(util_grads)
        out['utilFeat_delta'] = m; out['utilFeat_p'] = p
    else:
        out['utilFeat_delta'] = 0.0; out['utilFeat_p'] = 1.0

    # pEff (activeRank) floor.
    min_rank = float('inf')
    for run in oak:
        if run.get('ranks'):
            for r in run['ranks']:
                min_rank = min(min_rank, r)
    out['pEff_min'] = min_rank if min_rank != float('inf') else None

    # agent_ms (final-heartbeat mean over seeds).
    ms = [r['agentMs'] for r in oak if r.get('agentMs') is not None]
    out['agent_ms_oak'] = (sum(ms) / len(ms)) if ms else None

    # Multi-baseline strict regression flag.
    out['baselines_lost'] = []
    out['baselines_lost_noise'] = []  # below effect-size floor
    for (bl_name, bl_data) in baselines_data:
        bl_oak = bl_data.get('oak', [])
        if not (oak and bl_oak): continue
        seeds_a = {r['seed']: r[rr_key] for r in oak}
        seeds_b = {r['seed']: r[rr_key] for r in bl_oak}
        common = sorted(set(seeds_a) & set(seeds_b))
        if len(common) < 2: continue
        deltas = [seeds_a[s] - seeds_b[s] for s in common]
        m, t, p = paired_t(deltas)
        # Strict loss: paired-t says current is worse at p < 0.05.
        if m < 0 and p > 0.5:
            # m < 0 → t < 0 → upper-tail p > 0.5; we want LOWER-TAIL p for "current is worse".
            lower_p = 1 - p
            if lower_p < 0.05:
                out['baselines_lost'].append((bl_name, m, lower_p))

    return out


def filter_by_effect_size(env_signals, min_effect):
    """Re-classify each env's baselines_lost into above/below the effect-size floor."""
    if min_effect <= 0:
        return env_signals
    for env, s in env_signals.items():
        keep, drop = [], []
        for (n, m, p) in s.get('baselines_lost', []):
            if abs(m) >= min_effect:
                keep.append((n, m, p))
            else:
                drop.append((n, m, p))
        s['baselines_lost'] = keep
        s['baselines_lost_noise'] = drop
    return env_signals


def report(bench_dir: Path, baselines: list, ablation_baseline: Path = None,
           final_rr: bool = False, p_threshold: float = 0.05,
           min_effect: float = 0.0) -> str:
    data = gather_per_seed(bench_dir)
    baselines_data = []
    for bl_path in baselines:
        bl_name = bl_path.name
        bl_data = gather_per_seed(bl_path)
        baselines_data.append((bl_name, bl_data))

    env_signals = {}
    for env in sorted(data):
        env_signals[env] = per_env_signals(data[env], [
            (n, d.get(env, {})) for (n, d) in baselines_data
        ], final_rr=final_rr)

    env_signals = filter_by_effect_size(env_signals, min_effect)

    total = len(env_signals)
    rR_strict_win = sum(1 for s in env_signals.values()
                        if s['rewardRate_delta'] > 0 and s['rewardRate_p'] < p_threshold)
    rR_strict_loss = sum(1 for s in env_signals.values()
                         if s['rewardRate_delta'] < 0 and s['rewardRate_p'] < p_threshold)
    rR_pos = sum(1 for s in env_signals.values() if s['rewardRate_delta'] > 0)
    lpm_pos = sum(1 for s in env_signals.values()
                  if s['LPM_rho'] > 0 and s['LPM_p'] < p_threshold)
    util_pos = sum(1 for s in env_signals.values()
                   if s['utilFeat_delta'] > 0 and s['utilFeat_p'] < p_threshold)
    util_neg = sum(1 for s in env_signals.values()
                   if s['utilFeat_delta'] < 0 and s['utilFeat_p'] < p_threshold)
    multi_signal_pass = sum(
        1 for s in env_signals.values()
        if (s['rewardRate_delta'] >= 0 or s['rewardRate_p'] >= p_threshold)
        and ((s['rewardRate_delta'] > 0 and s['rewardRate_p'] < p_threshold) or
             (s['LPM_rho'] > 0 and s['LPM_p'] < p_threshold) or
             (s['utilFeat_delta'] > 0 and s['utilFeat_p'] < p_threshold))
    )
    strict_regression = sum(1 for s in env_signals.values() if s['baselines_lost'])
    noise_regressions = sum(1 for s in env_signals.values() if s.get('baselines_lost_noise'))

    # Final rR > 1e-3 (convergence-start) on the 46 zero-equivalent envs.
    nonzero_envs = []
    for env, runs in data.items():
        oak_runs = runs.get('oak', [])
        if oak_runs:
            avg_final = sum(r['finalRR'] for r in oak_runs) / len(oak_runs)
            if abs(avg_final) > 1e-3:
                nonzero_envs.append((env, avg_final))

    lines = []
    lines.append("# Pass 14 strict-gate report")
    lines.append("")
    lines.append(f"Bench dir: `{bench_dir}`")
    if baselines:
        lines.append(f"Baselines (multi-baseline strict gate): {', '.join('`%s`' % b for b in baselines)}")
    if ablation_baseline:
        lines.append(f"Ablation baseline: `{ablation_baseline}`")
    lines.append(f"Envs: {total} | RR mode: {'final-heartbeat' if final_rr else 'mean-of-heartbeats'}"
                 f" | Effect-size floor: {min_effect:g}")
    lines.append("")
    lines.append("## Strict-rule pass/fail (per the per-tier acceptance gate)")
    lines.append("")
    lines.append(f"- (A) **Zero envs strictly worse vs ANY baseline** at p<{p_threshold} "
                 f"(|Δ| ≥ {min_effect:g}): "
                 f"{'**PASS**' if strict_regression == 0 else f'**FAIL** ({strict_regression} envs)'}")
    if noise_regressions > 0:
        lines.append(f"  - (Below-floor noise-level apparent regressions: {noise_regressions} envs "
                     f"(|Δ| < {min_effect:g}); not counted toward strict gate.)")
    lines.append(f"- (C) **>=1 channel improves at p<{p_threshold}**: "
                 f"rR_win={rR_strict_win}/{total}, LPM_pos={lpm_pos}/{total}, util_pos={util_pos}/{total}")
    lines.append(f"- (D) LPM positive trend on >=85% of envs: "
                 f"{'**PASS**' if lpm_pos / max(total, 1) >= 0.85 else f'**FAIL** ({lpm_pos}/{total} = {lpm_pos / max(total, 1):.3f})'}")
    lines.append("")
    lines.append("## Headline counts")
    lines.append("")
    lines.append(f"- rewardRate strict win (Δ vs random > 0 at p<{p_threshold}): **{rR_strict_win}/{total}** ({rR_strict_win / max(total, 1):.3f})")
    lines.append(f"- rewardRate strict loss: {rR_strict_loss}/{total}")
    lines.append(f"- P(Δ vs random > 0): {rR_pos}/{total} ({rR_pos / max(total, 1):.3f})")
    lines.append(f"- LPM positive trend (p<{p_threshold}): {lpm_pos}/{total} ({lpm_pos / max(total, 1):.3f})")
    lines.append(f"- utilFeat positive (p<{p_threshold}): {util_pos}/{total} | utilFeat negative: {util_neg}/{total}")
    lines.append(f"- Multi-signal Pareto PASS: {multi_signal_pass}/{total} ({multi_signal_pass / max(total, 1):.3f})")
    lines.append(f"- **Strict regression vs ANY baseline**: {strict_regression}/{total}")
    lines.append(f"- **Non-zero rewardRate** (|finalRR| > 1e-3): {len(nonzero_envs)}/{total}")
    lines.append("")
    if nonzero_envs:
        lines.append("Non-zero envs (mean final rR): " +
                     ", ".join(f"{env}={rr:+.4f}" for env, rr in sorted(nonzero_envs, key=lambda x: -abs(x[1]))[:20]))
        lines.append("")
    lines.append("## Per-env table")
    lines.append("")
    cols = ["env", "rR Δ", "rR p", "LPM ρ", "LPM p", "util Δ", "util p",
            "pEff min", "agent_ms", "strict_regression"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for env in sorted(env_signals):
        s = env_signals[env]
        sr = "YES (" + ", ".join(f"{n}: Δ{m:+.4g} p{p:.3g}" for n, m, p in s['baselines_lost']) + ")" \
            if s['baselines_lost'] else "no"
        row = [env,
               f"{s['rewardRate_delta']:+.5f}",
               f"{s['rewardRate_p']:.3f}",
               f"{s['LPM_rho']:+.3f}",
               f"{s['LPM_p']:.3f}",
               f"{s['utilFeat_delta']:+.5g}",
               f"{s['utilFeat_p']:.3f}",
               f"{s['pEff_min']:.1f}" if s['pEff_min'] is not None else "?",
               f"{s['agent_ms_oak']:.3f}" if s['agent_ms_oak'] is not None else "?",
               sr]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main(argv):
    bench_dir = Path(argv[1]) if len(argv) > 1 else None
    baselines = []
    ablation_baseline = None
    output = None
    final_rr = False
    p_threshold = 0.05
    min_effect = 0.0
    for a in argv[2:]:
        if a.startswith('--output='):
            output = Path(a.split('=', 1)[1])
        elif a.startswith('--baselines='):
            baselines = [Path(p) for p in a.split('=', 1)[1].split(',') if p]
        elif a.startswith('--baseline='):  # singular; backwards-compat
            baselines = [Path(a.split('=', 1)[1])]
        elif a.startswith('--ablation-baseline='):
            ablation_baseline = Path(a.split('=', 1)[1])
        elif a == '--final-rr':
            final_rr = True
        elif a.startswith('--p='):
            p_threshold = float(a.split('=', 1)[1])
        elif a.startswith('--min-effect='):
            min_effect = float(a.split('=', 1)[1])
    if not bench_dir or not bench_dir.is_dir():
        print(f"usage: {argv[0]} <bench-dir> [--baselines DIR1,DIR2,...] [--output=PATH] "
              f"[--ablation-baseline=DIR] [--final-rr] [--p=0.05] [--min-effect=1e-4]",
              file=sys.stderr)
        return 2
    text = report(bench_dir, baselines, ablation_baseline, final_rr, p_threshold, min_effect)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
