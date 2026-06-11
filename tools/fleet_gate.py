#!/usr/bin/env python3
"""Population promotion gate for OaK fleet runs.

Compares a candidate fleet run against the keeper fleet run on the SAME seed
set and horizon (hard error otherwise). All resampling is PAIRED by seed.
The candidate PASSES iff ALL four criteria hold (intersection-union test:
each component runs one-sided at alpha=0.05; no multiplicity correction is
needed for an AND-gate):

  1. Divergence non-inferiority: 95th pct of paired-bootstrap
     (p_cand - p_keep) <= delta            (delta defaults to 0.64/sqrt(N):
                                            0.02 @ N=1024, 0.01 @ N=4096)
  2. IQM delivered-rate non-inferiority: 5th pct of paired-bootstrap
     (IQM_cand - IQM_keep) >= -0.05*|IQM_keep|, over ALL lanes.
     A non-blocking IMPROVEMENT flag is set when the 5th pct > 0.
  3. Survival non-inferiority: 5th pct of paired-bootstrap
     (RMST_cand - RMST_keep) >= -0.05*horizon.
  4. No new non-finite phases: the candidate's set of phases with nonzero
     cumulative NF counts must be a subset of the keeper's, and each phase's
     fleet total <= max(2x keeper total, 100*N/10).

Usage: fleet_gate.py <keeper-root> <candidate-root> [--policy oak]
                     [--delta D] [--rate-rel 0.05] [--rmst-rel 0.05]
Exit code 0 = PASS, 1 = FAIL, 2 = usage/shape error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fleet_stats import death_times, finals, iqm, load_run, paired_bootstrap, rmst  # noqa: E402


def unpaired_bootstrap(keeper, candidate, stat, n_boot=10_000, seed=0):
    """Bootstrap distribution of stat(candidate*) - stat(keeper*) with each side
    resampled INDEPENDENTLY. Correct for physics-swap / disjoint-population
    comparisons where lanes are not paired twins (pairing by seed is meaningless
    across physics backends: different RNG streams, different trajectories)."""
    rng = np.random.default_rng(seed)
    k, c = np.asarray(keeper, dtype=float), np.asarray(candidate, dtype=float)
    ik = rng.integers(0, len(k), size=(n_boot, len(k)))
    ic = rng.integers(0, len(c), size=(n_boot, len(c)))
    return np.array([stat(c[rc]) - stat(k[rk]) for rk, rc in zip(ik, ic)])

NF_PHASES = ("encode", "learn", "plan", "tdPsi")


def nf_totals(fin: dict[int, dict]) -> dict[str, int]:
    tot = dict.fromkeys(NF_PHASES, 0)
    for row in fin.values():
        nf = (row.get("health") or {}).get("nonFiniteByPhase") or {}
        for ph in NF_PHASES:
            tot[ph] += int(nf.get(ph, 0))
    return tot


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic + asymptotic p-value (numpy only)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    d = float(np.abs(cdf_a - cdf_b).max())
    n = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    if lam < 0.3:  # asymptotic series diverges as lam -> 0; p is 1 here
        return d, 1.0
    j = np.arange(1, 101)
    p = 2 * np.sum((-1) ** (j - 1) * np.exp(-2 * (lam * j) ** 2))
    return d, float(min(max(p, 0.0), 1.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("keeper", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--policy", default="oak")
    ap.add_argument("--delta", type=float, default=None,
                    help="divergence non-inferiority margin (default 0.64/sqrt(N))")
    ap.add_argument("--rate-rel", type=float, default=0.05)
    ap.add_argument("--rmst-rel", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--equivalence", action="store_true",
                    help="two-sided physics-swap equivalence: run all criteria in BOTH "
                         "directions (swap keeper/candidate) and add KS tests; PASS only "
                         "if both directions PROMOTE and KS p >= 0.01")
    args = ap.parse_args()

    keep_run, cand_run = load_run(args.keeper), load_run(args.candidate)
    keep_fin, cand_fin = (finals(r, args.policy) for r in (keep_run, cand_run))
    keep_seeds, cand_seeds = sorted(keep_fin), sorted(cand_fin)
    if args.equivalence:
        # Physics-swap / disjoint-population comparison: lanes are NOT paired
        # twins (different trajectories even at equal seeds), so only equal N
        # is required and all statistics are unpaired.
        if len(keep_seeds) != len(cand_seeds):
            print(f"GATE ERROR: population sizes differ "
                  f"({len(keep_seeds)} vs {len(cand_seeds)}).")
            return 2
    elif keep_seeds != cand_seeds:
        print(f"GATE ERROR: seed sets differ (keeper {len(keep_seeds)}, "
              f"candidate {len(cand_seeds)}); paired comparison requires identical seeds.")
        return 2
    kh, ch = keep_run["meta"]["steps"], cand_run["meta"]["steps"]
    if kh != ch:
        print(f"GATE ERROR: horizons differ (keeper {kh}, candidate {ch}).")
        return 2
    n, horizon = len(keep_seeds), kh

    def run_gate(keep_fin, cand_fin, keep_seeds, cand_seeds, label):
        return gate_verdicts(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args, label)

    if args.equivalence:
        return equivalence_gate(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args)

    ok = run_gate(keep_fin, cand_fin, keep_seeds, cand_seeds, None)
    print("PROMOTE" if ok else "REJECT")
    return 0 if ok else 1


def equivalence_gate(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args) -> int:
    """TOST-style two-sided equivalence with three-way verdicts per criterion:
    EQUIVALENT   — 90% CI of the unpaired bootstrap difference lies inside the band
    DIFFERENT    — 90% CI lies entirely outside the band
    INCONCLUSIVE — CI spans a band edge: underpowered at this N/horizon; the
                   printed factor estimates how much more N is needed.
    Overall: EQUIVALENT iff all numeric criteria EQUIVALENT and KS p >= 0.01;
    DIFFERENT if any criterion DIFFERENT or KS fails; else INCONCLUSIVE (exit 3)."""
    k_t, k_e, _ = death_times(keep_fin, horizon)
    c_t, c_e, _ = death_times(cand_fin, horizon)
    k_del = np.array([keep_fin[s]["rewardSum"] / max(keep_fin[s]["step"], 1) for s in keep_seeds])
    c_del = np.array([cand_fin[s]["rewardSum"] / max(cand_fin[s]["step"], 1) for s in cand_seeds])
    print(f"fleet_gate --equivalence: N={n} horizon={horizon} policy={args.policy} "
          f"boot={args.boot} (unpaired TOST)")

    def tost(name: str, draws: np.ndarray, band: float, fmt: str = "{:+.4f}") -> str:
        lo, hi = np.percentile(draws, [5, 95])
        point = float(np.median(draws))
        if -band <= lo and hi <= band:
            verdict = "EQUIVALENT"
        elif hi < -band or lo > band:
            verdict = "DIFFERENT"
        else:
            verdict = "INCONCLUSIVE"
        half = (hi - lo) / 2
        need = (half / band) ** 2 if band > 0 else float("inf")
        extra = f" needs ~{need:.1f}x N" if verdict == "INCONCLUSIVE" and need > 1 else ""
        print(f"  [{verdict:12s}] {name:14s} Δ={fmt.format(point)} "
              f"CI90=[{fmt.format(lo)}, {fmt.format(hi)}] band=±{fmt.format(band).lstrip('+')}{extra}")
        return verdict

    verdicts = []
    dp = unpaired_bootstrap(k_e.astype(float), c_e.astype(float), np.mean,
                            n_boot=args.boot, seed=11)
    # Fixed practical-irrelevance band for instrument swaps (NOT N-dependent:
    # the band encodes what difference we don't care about; N buys resolution).
    band_p = args.delta if args.delta is not None else 0.03
    verdicts.append(tost("divergence", dp, band_p))
    di = unpaired_bootstrap(k_del, c_del, iqm, n_boot=args.boot, seed=12)
    verdicts.append(tost("iqmDelivered", di, args.rate_rel * abs(iqm(k_del)), "{:+.5g}"))
    rng = np.random.default_rng(13)
    ik = rng.integers(0, n, size=(args.boot, n))
    ic = rng.integers(0, n, size=(args.boot, n))
    dr = np.array([
        rmst(c_t[rc], c_e[rc], horizon) - rmst(k_t[rk], k_e[rk], horizon)
        for rk, rc in zip(ik, ic)])
    verdicts.append(tost("rmstSurvival", dr, args.rmst_rel * horizon, "{:+,.0f}"))
    k_nf, c_nf = nf_totals(keep_fin), nf_totals(cand_fin)
    nf_ok = (not [ph for ph in NF_PHASES if c_nf[ph] > 0 and k_nf[ph] == 0]
             and not [ph for ph in NF_PHASES if c_nf[ph] > max(2 * k_nf[ph], 100 * n / 10)]
             and not [ph for ph in NF_PHASES if k_nf[ph] > max(2 * c_nf[ph], 100 * n / 10)])
    print(f"  [{'EQUIVALENT' if nf_ok else 'DIFFERENT':12s}] nfPhases       "
          f"keeper={k_nf} cand={c_nf}")
    verdicts.append("EQUIVALENT" if nf_ok else "DIFFERENT")
    d_del, p_del = ks_test(k_del, c_del)
    d_srv, p_srv = ks_test(k_t[k_e == 1], c_t[c_e == 1]) if k_e.sum() and c_e.sum() else (0.0, 1.0)
    ks_ok = p_del >= 0.01 and p_srv >= 0.01
    print(f"  [{'EQUIVALENT' if ks_ok else 'DIFFERENT':12s}] ksDistribution deliveredRate "
          f"D={d_del:.4f} p={p_del:.4f}; deathTimes D={d_srv:.4f} p={p_srv:.4f}")
    verdicts.append("EQUIVALENT" if ks_ok else "DIFFERENT")
    if all(v == "EQUIVALENT" for v in verdicts):
        print("EQUIVALENT")
        return 0
    if "DIFFERENT" in verdicts:
        print("DIFFERENT")
        return 1
    print("INCONCLUSIVE — increase N or horizon and rerun")
    return 3


def gate_verdicts(keep_fin, cand_fin, keep_seeds, cand_seeds, n, horizon, args, label):
    delta = args.delta if args.delta is not None else 0.64 / np.sqrt(n)
    # Unpaired statistics for physics-swap/disjoint-seed comparisons; paired
    # (variance-reduced) for same-physics same-seed candidate gating.
    boot = unpaired_bootstrap if args.equivalence else paired_bootstrap
    verdicts: list[tuple[str, bool, str]] = []

    # 1. divergence non-inferiority
    k_times, k_events, _ = death_times(keep_fin, horizon)
    c_times, c_events, _ = death_times(cand_fin, horizon)
    dp = boot(k_events.astype(float), c_events.astype(float),
              np.mean, n_boot=args.boot, seed=11)
    dp95 = float(np.percentile(dp, 95))
    p_keep, p_cand = k_events.mean(), c_events.mean()
    verdicts.append((
        "divergence", dp95 <= delta,
        f"p keeper={p_keep:.3f} cand={p_cand:.3f} Δp95={dp95:+.4f} δ={delta:.4f}"))

    # 2. IQM delivered-rate non-inferiority (paired, all lanes)
    k_del = np.array([keep_fin[s]["rewardSum"] / max(keep_fin[s]["step"], 1)
                      for s in keep_seeds])
    c_del = np.array([cand_fin[s]["rewardSum"] / max(cand_fin[s]["step"], 1)
                      for s in cand_seeds])
    di = boot(k_del, c_del, iqm, n_boot=args.boot, seed=12)
    di5 = float(np.percentile(di, 5))
    iqm_keep = iqm(k_del)
    margin = args.rate_rel * abs(iqm_keep)
    improved = di5 > 0
    verdicts.append((
        "iqmDelivered", di5 >= -margin,
        f"IQM keeper={iqm_keep:.5g} cand={iqm(c_del):.5g} ΔIQM5={di5:+.5g} "
        f"margin={-margin:.5g}{' IMPROVEMENT' if improved else ''}"))

    # 3. survival non-inferiority via RMST (paired)
    def rmst_stat_factory(times):
        def stat_on(events_resampled, times_resampled):
            return rmst(times_resampled, events_resampled, horizon)
        return stat_on

    rng = np.random.default_rng(13)
    idx = rng.integers(0, n, size=(args.boot, n))
    if args.equivalence:
        idx_c = rng.integers(0, n, size=(args.boot, n))
        dr = np.array([
            rmst(c_times[rc], c_events[rc], horizon) - rmst(k_times[rk], k_events[rk], horizon)
            for rk, rc in zip(idx, idx_c)
        ])
    else:
        dr = np.array([
            rmst(c_times[row], c_events[row], horizon) - rmst(k_times[row], k_events[row], horizon)
            for row in idx
        ])
    dr5 = float(np.percentile(dr, 5))
    rmst_margin = args.rmst_rel * horizon
    verdicts.append((
        "rmstSurvival", dr5 >= -rmst_margin,
        f"RMST keeper={rmst(k_times, k_events, horizon):,.0f} "
        f"cand={rmst(c_times, c_events, horizon):,.0f} ΔRMST5={dr5:+,.0f} "
        f"margin={-rmst_margin:,.0f}"))

    # 4. no new NF phases + ceiling
    k_nf, c_nf = nf_totals(keep_fin), nf_totals(cand_fin)
    new_phases = [ph for ph in NF_PHASES if c_nf[ph] > 0 and k_nf[ph] == 0]
    over = [ph for ph in NF_PHASES
            if c_nf[ph] > max(2 * k_nf[ph], 100 * n / 10)]
    verdicts.append((
        "nfPhases", not new_phases and not over,
        f"keeper={k_nf} cand={c_nf}"
        + (f" NEW={new_phases}" if new_phases else "")
        + (f" OVER={over}" if over else "")))

    hdr = f"fleet_gate: N={n} horizon={horizon} policy={args.policy} boot={args.boot}"
    if label:
        hdr += f" [{label}]"
    print(hdr)
    ok = True
    for name, passed, detail in verdicts:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:14s} {detail}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
