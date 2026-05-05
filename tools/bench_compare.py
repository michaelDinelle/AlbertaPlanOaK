#!/usr/bin/env python3
"""Compare two OaK benchmark roots by seed and heartbeat/final metrics."""

from __future__ import annotations

import argparse
import glob
import json
import statistics as stats
from pathlib import Path


def load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def env_mean(record: dict, name: str) -> float:
    value = record.get("health", {}).get("envInfo", {}).get(name, {}).get("mean")
    return value if isinstance(value, (int, float)) else 0.0


def nonfinite(record: dict) -> int:
    phases = record.get("health", {}).get("nonFiniteByPhase", {})
    if not isinstance(phases, dict):
        return 0
    return sum(int(v) for v in phases.values() if isinstance(v, (int, float)))


def collect(root: Path, policy: str, final_only: bool) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for path in glob.glob(str(root / "**" / "*.json"), recursive=True):
        item = load(path)
        if not item or item.get("command") != "benchmark" or item.get("policy") != policy:
            continue
        if final_only and item.get("stopReason") != "measured_steps":
            continue
        seed = int(item.get("seed", -1))
        out.setdefault(seed, []).append(item)
    for items in out.values():
        items.sort(key=lambda r: (
            int(r.get("measuredStepCount", 0)),
            1 if r.get("stopReason") == "measured_steps" else 0,
        ))
    return out


def fmt(x: float, width: int = 10, precision: int = 3) -> str:
    text = f"{x:.{precision}f}"
    if len(text) > width:
        text = f"{x:.2e}"
    return text.rjust(width)


def row_metrics(record: dict) -> tuple[float, float, int, float, float, float]:
    h = record.get("health", {})
    return (
        float(record.get("rewardSum", 0.0)),
        env_mean(record, "info:x_velocity"),
        nonfinite(record),
        float(record.get("stepsPerSecond", 0.0)),
        float(h.get("actorUpdateNormMean", 0.0) or 0.0),
        float(h.get("modelPriorityMax", 0.0) or 0.0),
    )


def final_report(base: dict[int, list[dict]], cand: dict[int, list[dict]]) -> str:
    seeds = sorted(set(base) & set(cand))
    lines = ["seed      dReward      dXVel  dNF      baseR      candR      candX        actor         prio"]
    deltas: list[tuple[int, float, float, int]] = []
    for seed in seeds:
        b, c = base[seed][-1], cand[seed][-1]
        br, bx, bnf, *_ = row_metrics(b)
        cr, cx, cnf, _, actor, prio = row_metrics(c)
        deltas.append((seed, cr - br, cx - bx, cnf - bnf))
        lines.append(
            f"{seed:4d} {fmt(cr - br)} {fmt(cx - bx, 8, 5)} {cnf - bnf:4d}"
            f" {fmt(br)} {fmt(cr)} {fmt(cx, 8, 5)} {actor:12.3g} {prio:12.3g}"
        )
    if deltas:
        dr = [x[1] for x in deltas]
        dx = [x[2] for x in deltas]
        pos_reward = sum(cand[s][-1].get("rewardSum", 0.0) > 0 for s, *_ in deltas)
        pos_x = sum(env_mean(cand[s][-1], "info:x_velocity") > 0 for s, *_ in deltas)
        lines.append(
            "mean "
            f"{fmt(stats.mean(dr))} {fmt(stats.mean(dx), 8, 5)}"
            f" posReward {pos_reward}/{len(deltas)} posX {pos_x}/{len(deltas)}"
            f" nfDelta {sum(x[3] for x in deltas)}"
        )
    return "\n".join(lines)


def heartbeat_report(base: dict[int, list[dict]], cand: dict[int, list[dict]]) -> str:
    lines = ["seed   firstStep    dReward      dXVel  note"]
    for seed in sorted(set(base) & set(cand)):
        by_step = {int(r.get("measuredStepCount", 0)): r for r in base[seed]}
        found = None
        note = "no divergence"
        for c in cand[seed]:
            step = int(c.get("measuredStepCount", 0))
            b = by_step.get(step)
            if not b:
                continue
            br, bx, bnf, *_ = row_metrics(b)
            cr, cx, cnf, *_ = row_metrics(c)
            if abs(cr - br) > 1e-6 or abs(cx - bx) > 1e-9 or cnf != bnf:
                found = (step, cr - br, cx - bx)
                note = f"nf {bnf}->{cnf}"
                break
        if found is None:
            lines.append(f"{seed:4d}          -          -        -  {note}")
        else:
            lines.append(f"{seed:4d} {found[0]:10d} {fmt(found[1])} {fmt(found[2], 8, 5)}  {note}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--policy", default="oak")
    parser.add_argument("--heartbeats", action="store_true")
    args = parser.parse_args()
    base = collect(args.baseline, args.policy, final_only=not args.heartbeats)
    cand = collect(args.candidate, args.policy, final_only=not args.heartbeats)
    print(final_report(base, cand))
    if args.heartbeats:
        print()
        print(heartbeat_report(base, cand))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
