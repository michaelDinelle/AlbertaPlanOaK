#!/usr/bin/env python3
"""Tables A–D: oak vs random reward + per-env SPS. Stdout is pipe tables + headings/bullets (GFM).

Expects one or more **benchmark run roots** (e.g. `.oak-benchmarks/<run-id>/`): for each env and seed,
snapshots live at `<run-root>/<envId>/seed<N>/<stem>.json` with optional pulses
`<stem>.heartbeat-NNNNNN.json`. For each stem the loader picks the latest snapshot: final `<stem>.json`
if present, else the highest heartbeat index.

Use `--run LABEL DIR` (repeat for multiple runs). Aliases: `--metrics-run`, `--partial-run`.

Emits a **Snapshot coverage** table: `numEnvs` range/distinct values and `measuredStepCount` stats so cross-run
comparisons stay honest when parallel width or measured length differ.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

EPS = 1e-9
# env, oak, rnd, Δ, W/L/T, sps_oak, sps_rnd
Row = tuple[str, float, float, float, str, float, float]


@dataclass(frozen=True)
class RunSnapshotSummary:
    """Aggregates over every metrics JSON read under the run root (oak + random, all seeds)."""

    num_envs_min: int | None
    num_envs_max: int | None
    num_envs_distinct: tuple[int, ...]
    measured_step_min: int | None
    measured_step_max: int | None
    measured_step_mean: float | None
    snapshot_count: int


def _tag(d: float) -> str:
    if d > EPS:
        return "W"
    if d < -EPS:
        return "L"
    return "T"


def _latest_snapshot_per_stem(seed_dir: Path) -> list[Path]:
    """Prefer final `<stem>.json`, else highest `<stem>.heartbeat-NNNNNN.json`."""
    final: dict[str, Path] = {}
    pulses: dict[str, tuple[int, Path]] = {}
    for fp in seed_dir.glob("*.json"):
        name = fp.name
        if ".heartbeat-" in name:
            stem, _, rest = name.partition(".heartbeat-")
            try:
                seq = int(rest.split(".json", 1)[0])
            except ValueError:
                continue
            prev = pulses.get(stem)
            if prev is None or seq > prev[0]:
                pulses[stem] = (seq, fp)
        else:
            final[name[: -len(".json")]] = fp
    picks: list[Path] = []
    for stem, fp in final.items():
        picks.append(fp)
    for stem, (_, fp) in pulses.items():
        if stem not in final:
            picks.append(fp)
    return picks


def load_run_root(run_root: Path) -> tuple[list[Row], RunSnapshotSummary]:
    """Load rows from `<run-root>/<envId>/seed<N>/*.json` metrics snapshots."""
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"oak": [], "random": []})
    sps: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"oak": [], "random": []})
    snapshots: list[Path] = []
    num_envs_vals: list[int] = []
    measured_steps_vals: list[int] = []
    if run_root.is_dir():
        for env_dir in sorted(run_root.iterdir()):
            if not env_dir.is_dir():
                continue
            for seed_dir in sorted(env_dir.iterdir()):
                if seed_dir.is_dir():
                    snapshots.extend(_latest_snapshot_per_stem(seed_dir))
    for fp in sorted(snapshots):
        j = json.loads(fp.read_text())
        ne_raw = j.get("numEnvs")
        if isinstance(ne_raw, (int, float)) and ne_raw > 0:
            num_envs_vals.append(int(ne_raw))
        msc = j.get("measuredStepCount")
        if isinstance(msc, (int, float)) and msc == msc and msc >= 0:
            measured_steps_vals.append(int(msc))
        pol = j.get("policy")
        if pol not in ("oak", "random"):
            continue
        eid, rs = j.get("envId"), j.get("rewardSum")
        if eid is None or rs is None:
            continue
        agg[eid][pol].append(rs)
        v = j.get("stepsPerSecond")
        ne = j.get("numEnvs")
        if isinstance(v, (int, float)) and v == v:
            if isinstance(ne, (int, float)) and ne > 0:
                sps[eid][pol].append(float(v) / float(ne))
            else:
                sps[eid][pol].append(float("nan"))
    out: list[Row] = []
    for eid in sorted(agg):
        a = agg[eid]
        if not a["oak"] or not a["random"]:
            continue
        mo, mr = mean(a["oak"]), mean(a["random"])
        d = mo - mr
        so = [x for x in sps.get(eid, {}).get("oak", []) if x == x]
        sr = [x for x in sps.get(eid, {}).get("random", []) if x == x]
        mso = mean(so) if so else float("nan")
        msr = mean(sr) if sr else float("nan")
        out.append((eid, mo, mr, d, _tag(d), mso, msr))
    ne_dist = tuple(sorted(set(num_envs_vals)))
    summary = RunSnapshotSummary(
        num_envs_min=min(num_envs_vals) if num_envs_vals else None,
        num_envs_max=max(num_envs_vals) if num_envs_vals else None,
        num_envs_distinct=ne_dist,
        measured_step_min=min(measured_steps_vals) if measured_steps_vals else None,
        measured_step_max=max(measured_steps_vals) if measured_steps_vals else None,
        measured_step_mean=mean(measured_steps_vals) if measured_steps_vals else None,
        snapshot_count=len(snapshots),
    )
    return out, summary


def _pick(rows: list[Row], envs: set[str] | None) -> list[Row]:
    return rows if not envs else [x for x in rows if x[0] in envs]


def stats(rows: list[Row]) -> dict:
    n = len(rows)
    o = [x[1] for x in rows]
    r = [x[2] for x in rows]
    d = [x[3] for x in rows]
    w = sum(1 for x in rows if x[4] == "W")
    return {
        "n": n,
        "mu_oak": mean(o) if n else float("nan"),
        "sd_oak": pstdev(o) if n > 1 else 0.0,
        "mu_rand": mean(r) if n else float("nan"),
        "sd_rand": pstdev(r) if n > 1 else 0.0,
        "mu_delta": mean(d) if n else float("nan"),
        "sd_delta": pstdev(d) if n > 1 else 0.0,
        "wins": w,
        "losses": sum(1 for x in rows if x[4] == "L"),
        "ties": sum(1 for x in rows if x[4] == "T"),
        "p_delta_gt_0": (w / n) if n else float("nan"),
    }


def sps_stats(rows: list[Row]) -> dict:
    n = len(rows)
    if not n:
        z = float("nan")
        return {"n": 0, "mu_oak": z, "sd_oak": z, "mu_rand": z, "sd_rand": z, "mu_delta": z, "sd_delta": z}

    def moments(xs: list[float]) -> tuple[float, float]:
        xs = [x for x in xs if x == x]
        if not xs:
            return float("nan"), float("nan")
        m = mean(xs)
        s = pstdev(xs) if len(xs) > 1 else 0.0
        return m, s

    so = [x[5] for x in rows]
    sr = [x[6] for x in rows]
    dd = [x[5] - x[6] for x in rows if x[5] == x[5] and x[6] == x[6]]
    mo, sdo = moments(so)
    mr, sdr = moments(sr)
    md, sdd = moments(dd)
    return {"n": n, "mu_oak": mo, "sd_oak": sdo, "mu_rand": mr, "sd_rand": sdr, "mu_delta": md, "sd_delta": sdd}


def _f(x: float, p: bool = False) -> str:
    if x != x:
        return "—"
    return f"{x:.3f}" if p else f"{x:.3e}"


def _fmt_ne_dist(t: tuple[int, ...]) -> str:
    if not t:
        return "—"
    if len(t) == 1:
        return str(t[0])
    return ", ".join(str(x) for x in t)


def emit(
    runs: list[tuple[str, list[Row]]],
    summaries: list[tuple[str, RunSnapshotSummary]],
    note: str | None,
    env_note: str | None,
) -> None:
    pre = [(lb, stats(rs), sps_stats(rs)) for lb, rs in runs]
    print("### Definitions")
    if note:
        print(f"- {note}")
    if env_note:
        print(f"- Env filter: {env_note}")
    print(
        "- **Reward (TABLE A–C):** `rewardSum` is the sum of rewards over every measured env-step in the window "
        "(one reward per parallel env per batch). Totals are comparable across different `numEnvs` when "
        "`measuredStepCount` matches; if runs used different measured lengths or unfinished heartbeats, compare "
        "the **Measured steps** row below."
    )
    print("- Δ = mean oak reward − mean random per env (mean over seeds from snapshot JSON in each env folder).")
    print("- W / L / T from sign of Δ; P(Δ>0) = W/n; μ, σ over env rows.")
    print(
        "- **TABLE D:** steps/s **per parallel env**: `stepsPerSecond / numEnvs` (aggregate throughput divided by "
        "parallel width, so different `numEnvs` does not double-count width in this row)."
    )
    print(
        "- **Snapshot coverage table:** min/max/mean over **all** snapshot JSON files under each run root (not restricted "
        "to the aligned env subset used in TABLE A–D)."
    )
    print()
    print("### Snapshot coverage (parallel width + measured length)")
    print(
        "| Run | JSON files | numEnvs min | numEnvs max | numEnvs (distinct) | measuredStepCount min | max | mean |"
    )
    print("|-----|------------|-------------|-------------|--------------------|-------------------------|-----|------|")
    for lb, s in summaries:
        mn = "—" if s.num_envs_min is None else str(s.num_envs_min)
        mx = "—" if s.num_envs_max is None else str(s.num_envs_max)
        ms_min = "—" if s.measured_step_min is None else str(s.measured_step_min)
        ms_max = "—" if s.measured_step_max is None else str(s.measured_step_max)
        ms_mean = "—" if s.measured_step_mean is None else f"{s.measured_step_mean:.1f}"
        print(
            f"| {lb} | {s.snapshot_count} | {mn} | {mx} | {_fmt_ne_dist(s.num_envs_distinct)} | "
            f"{ms_min} | {ms_max} | {ms_mean} |"
        )
    print()
    print("### TABLE A — Summary")
    print("| Run | n | μ_oak | σ_oak | μ_rand | σ_rand | μ_Δ | σ_Δ | P(Δ>0) |")
    print("|-----|---|-------|-------|--------|--------|-----|-----|--------|")
    for lb, s, _ in pre:
        print(
            f"| {lb} | {s['n']} | {_f(s['mu_oak'])} | {_f(s['sd_oak'])} | {_f(s['mu_rand'])} | {_f(s['sd_rand'])} | "
            f"{_f(s['mu_delta'])} | {_f(s['sd_delta'])} | {_f(s['p_delta_gt_0'], True)} |"
        )
    print()
    print("### TABLE B — W/L/T")
    print("| Run | n | W | L | T |")
    print("|-----|---|---|---|---|")
    for lb, s, _ in pre:
        if s["n"] == 0:
            print(f"| {lb} | 0 | — | — | — |")
        else:
            print(f"| {lb} | {s['n']} | {s['wins']} | {s['losses']} | {s['ties']} |")
    print()
    print("### TABLE C — Mean reward")
    print("| Run | n | oak | random | Δ |")
    print("|-----|---|-----|--------|---|")
    for lb, s, _ in pre:
        print(f"| {lb} | {s['n']} | {_f(s['mu_oak'])} | {_f(s['mu_rand'])} | {_f(s['mu_delta'])} |")
    print()
    print("### TABLE D — SPS per parallel env (mean over env rows)")
    print("| Run | n | μ_SPS_oak | σ_SPS_oak | μ_SPS_rand | σ_SPS_rand | μ_ΔSPS | σ_ΔSPS |")
    print("|-----|---|-----------|-----------|------------|------------|--------|--------|")
    for lb, _, sp in pre:
        if sp["n"] == 0:
            print(f"| {lb} | 0 | — | — | — | — | — | — |")
        else:
            print(
                f"| {lb} | {sp['n']} | {_f(sp['mu_oak'])} | {_f(sp['sd_oak'])} | {_f(sp['mu_rand'])} | {_f(sp['sd_rand'])} | "
                f"{_f(sp['mu_delta'])} | {_f(sp['sd_delta'])} |"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark tables A–D from .oak-benchmarks run roots.")
    ap.add_argument(
        "--run",
        "--metrics-run",
        "--partial-run",
        nargs=2,
        action="append",
        dest="runs",
        metavar=("LABEL", "DIR"),
        help="Benchmark run root (e.g. .oak-benchmarks/<run-id>/). Repeat for multiple runs.",
    )
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-align-to-partial", "--no-align", action="store_true", dest="na")
    a = ap.parse_args()

    only = {x.strip() for x in a.only.split(",") if x.strip()} if a.only else None
    env_note = ", ".join(sorted(only)) if only else None

    dirs: list[tuple[str, Path]] = []
    for lb, ps in a.runs or []:
        p = Path(ps).expanduser()
        if p.is_dir():
            dirs.append((lb, p))
        else:
            print(f"(skip) {lb} -> {p}", flush=True)

    loaded: list[tuple[str, list[Row], RunSnapshotSummary]] = []
    for lb, d in dirs:
        rows, summary = load_run_root(d)
        loaded.append((lb, rows, summary))

    align = bool(loaded) and not a.na
    align_ids: set[str] | None = None
    if align:
        sets = [{r[0] for r in rows} for _, rows, _ in loaded]
        align_ids = set.intersection(*sets) if sets else set()
        if only:
            align_ids &= only

    def filt(rows: list[Row]) -> list[Row]:
        r = _pick(rows, only)
        return _pick(r, align_ids) if align_ids is not None else r

    runs_out: list[tuple[str, list[Row]]] = []
    summaries_out: list[tuple[str, RunSnapshotSummary]] = []
    for lb, rows, summary in loaded:
        fr = filt(rows)
        runs_out.append((lb, fr))
        # Summaries reflect the same files as load_run_root; reward rows may be filtered by env alignment.
        summaries_out.append((lb, summary))

    if not runs_out:
        print("No inputs. Use --run LABEL DIR (repeat). Example: --run myrun .oak-benchmarks/20260417-073830-69143", flush=True)
        raise SystemExit(1)

    note = None
    if align and align_ids is not None:
        n = len(align_ids)
        if len(loaded) > 1:
            note = f"Aligned: intersection of paired envs across run roots, n={n}."
    elif loaded and a.na:
        note = "Unaligned: each run uses all envs with paired oak+random; row counts may differ."

    emit(runs_out, summaries_out, note, env_note)


if __name__ == "__main__":
    main()
