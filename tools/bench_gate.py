#!/usr/bin/env python3
"""Gate an OaK candidate root against a keeper benchmark root."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as stats
from pathlib import Path


def load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def metric(record: dict, name: str) -> float:
    if name == "rewardSum":
        value = record.get("rewardSum")
    elif name == "nonFiniteTotal":
        phases = record.get("health", {}).get("nonFiniteByPhase", {})
        value = sum(v for v in phases.values() if isinstance(v, (int, float))) if isinstance(phases, dict) else 0
    else:
        value = record.get("health", {}).get("envInfo", {}).get(name, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def collect(root: Path, policy: str, at_step: int | None) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in glob.glob(str(root / "**" / "*.json"), recursive=True):
        item = load(path)
        if not item or item.get("command") != "benchmark" or item.get("policy") != policy:
            continue
        step = int(item.get("measuredStepCount", 0))
        if at_step is not None and step != at_step:
            continue
        if at_step is None and item.get("stopReason") != "measured_steps":
            continue
        seed = int(item.get("seed", -1))
        if seed >= 0 and step >= int(out.get(seed, {}).get("measuredStepCount", -1)):
            out[seed] = item
    return out


def fmt(x: float, precision: int = 3) -> str:
    text = f"{x:.{precision}f}"
    return f"{x:.2e}" if len(text) > 11 else text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--policy", default="oak")
    parser.add_argument("--rocket-seed", type=int, default=3)
    parser.add_argument("--rocket-min-ratio", type=float, default=0.97)
    parser.add_argument("--backward-seeds", default="8")
    parser.add_argument("--focus-seeds", default="0,1,3,8")
    parser.add_argument("--focus-max-drop-ratio", type=float, default=0.03)
    parser.add_argument("--at-step", type=int)
    parser.add_argument("--allow-nf-regression", action="store_true")
    args = parser.parse_args()

    base = collect(args.baseline, args.policy, args.at_step)
    cand = collect(args.candidate, args.policy, args.at_step)
    focus = [int(x) for x in args.focus_seeds.split(",") if x.strip()]
    backward = {int(x) for x in args.backward_seeds.split(",") if x.strip()}
    seeds = [s for s in focus if s in base and s in cand]
    if not seeds:
        raise SystemExit("no common final benchmark files found for requested focus seeds")

    failures: list[str] = []
    rows: list[str] = ["seed      baseR      candR        dR    baseX    candX       dX  dNF  verdict"]
    d_rewards: list[float] = []
    d_xvels: list[float] = []
    for seed in seeds:
        br = metric(base[seed], "rewardSum")
        cr = metric(cand[seed], "rewardSum")
        bx = metric(base[seed], "info:x_velocity")
        cx = metric(cand[seed], "info:x_velocity")
        bnf = int(metric(base[seed], "nonFiniteTotal"))
        cnf = int(metric(cand[seed], "nonFiniteTotal"))
        d_rewards.append(cr - br)
        d_xvels.append(cx - bx)
        verdict = "ok"
        if seed == args.rocket_seed and br > 0.0:
            floor = br * args.rocket_min_ratio
            if cr < floor:
                verdict = "rocket-loss"
                failures.append(f"seed {seed} rocket reward {fmt(cr)} < {fmt(floor)}")
        reward_floor = br - max(1.0, abs(br) * args.focus_max_drop_ratio)
        if cr < reward_floor:
            verdict = "focus-regression" if verdict == "ok" else f"{verdict}+focus"
            failures.append(f"seed {seed} reward {fmt(cr)} < focus floor {fmt(reward_floor)}")
        if seed in backward and (cr < br or cx < bx):
            verdict = "backward-regression" if verdict == "ok" else f"{verdict}+backward"
            failures.append(
                f"seed {seed} regressed backward attractor: dReward {fmt(cr - br)}, dXVel {fmt(cx - bx, 5)}"
            )
        if not args.allow_nf_regression and cnf > bnf:
            verdict = "nf-regression" if verdict == "ok" else f"{verdict}+nf"
            failures.append(f"seed {seed} non-finite count {bnf}->{cnf}")
        rows.append(
            f"{seed:4d} {fmt(br):>10} {fmt(cr):>10} {fmt(cr - br):>9}"
            f" {fmt(bx, 5):>8} {fmt(cx, 5):>8} {fmt(cx - bx, 5):>8}"
            f" {cnf - bnf:4d}  {verdict}"
        )

    pos_reward = sum(metric(cand[s], "rewardSum") > 0.0 for s in seeds)
    pos_x = sum(metric(cand[s], "info:x_velocity") > 0.0 for s in seeds)
    rows.append(
        "mean "
        f"{fmt(stats.mean(d_rewards)):>31} {fmt(stats.mean(d_xvels), 5):>27}"
        f"  posReward {pos_reward}/{len(seeds)} posX {pos_x}/{len(seeds)}"
    )
    print("\n".join(rows))
    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"- {item}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
