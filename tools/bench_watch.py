#!/usr/bin/env python3
"""Continuously summarize OaK benchmark heartbeat JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def latest_root() -> Path | None:
    base = Path(".oak-benchmarks")
    if not base.exists():
        return None
    dirs = [
        p for p in base.iterdir()
        if p.is_dir() and p.name != ".oak-benchmark.lock"
    ]
    return max(dirs, key=lambda p: p.stat().st_mtime, default=None)


def load(path: Path) -> dict | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def info(record: dict, name: str, field: str) -> float | None:
    value = record.get("health", {}).get("envInfo", {}).get(name, {}).get(field)
    return value if isinstance(value, (int, float)) else None


def health(record: dict, name: str) -> float | int | None:
    value = record.get("health", {}).get(name)
    return value if isinstance(value, (int, float)) else None


def nonfinite(record: dict) -> int:
    phases = record.get("health", {}).get("nonFiniteByPhase", {})
    if not isinstance(phases, dict):
        return 0
    return sum(int(v) for v in phases.values() if isinstance(v, (int, float)))


def fmt(value: float | int | None, width: int, precision: int = 3) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    if isinstance(value, int):
        text = str(value)
    else:
        text = f"{value:.{precision}g}"
        if len(text) > width:
            text = f"{value:.1e}"
    if len(text) > width:
        text = text[:width]
    return text.rjust(width)


def records(root: Path, policy_filter: set[str] | None) -> list[tuple[dict, dict | None]]:
    by_key: dict[tuple[str, str, int], list[dict]] = {}
    for path in root.rglob("*.json"):
        item = load(path)
        if not item or item.get("command") != "benchmark":
            continue
        policy = item.get("policy")
        if policy_filter and policy not in policy_filter:
            continue
        key = (item.get("envId", ""), policy, int(item.get("seed", -1)))
        by_key.setdefault(key, []).append(item)

    out: list[tuple[dict, dict | None]] = []
    for items in by_key.values():
        items.sort(key=lambda r: (
            int(r.get("measuredStepCount", 0)),
            int(r.get("heartbeatSequence", 1_000_000_000)),
        ))
        latest = items[-1]
        latest_step = int(latest.get("measuredStepCount", 0))
        prev = next(
            (r for r in reversed(items[:-1]) if int(r.get("measuredStepCount", 0)) < latest_step),
            None,
        )
        out.append((latest, prev))
    out.sort(key=lambda pair: (
        pair[0].get("policy", ""),
        int(pair[0].get("seed", -1)),
    ))
    return out


def window_mean(latest: dict, prev: dict | None, name: str) -> float | None:
    if prev is None:
        return None
    a_sum = info(latest, name, "sum")
    b_sum = info(prev, name, "sum")
    a_count = info(latest, name, "finiteCount")
    b_count = info(prev, name, "finiteCount")
    if None in (a_sum, b_sum, a_count, b_count) or a_count == b_count:
        return None
    return (a_sum - b_sum) / (a_count - b_count)


def render(root: Path, policy_filter: set[str] | None) -> str:
    rows = records(root, policy_filter)
    lines = [
        f"root: {root}",
        "pol    seed   step     reward      dR       xvel    vWin     run      ctrl     nf  term   prio     actor    dAct   reinit  selAge   selDur    keyL1    advL1",
    ]
    for latest, prev in rows:
        step = int(latest.get("measuredStepCount", 0))
        reward = latest.get("rewardSum")
        d_reward = None if prev is None else reward - prev.get("rewardSum", 0)
        xvel = info(latest, "info:x_velocity", "mean")
        v_win = window_mean(latest, prev, "info:x_velocity")
        run = info(latest, "info:reward_run", "sum")
        ctrl = info(latest, "info:reward_ctrl", "sum")
        term = health(latest, "terminationRate")
        prio = health(latest, "modelPriorityMax")
        actor = health(latest, "actorUpdateNormMean")
        action_delta = health(latest, "actionDeltaL2Mean")
        reinit = health(latest, "oakOptionReinitCountFinal")
        sel_age = health(latest, "selectedOptionAge")
        sel_dur = health(latest, "selectedOptionDuration")
        key_l1 = health(latest, "keyboardL1")
        adv_l1 = health(latest, "actorAdvantageL1")
        cols = [
            f"{str(latest.get('policy', '-'))[:6]:6}",
            f"{int(latest.get('seed', -1)):4d}",
            f"{step:7d}",
            fmt(reward, 11),
            fmt(d_reward, 9),
            fmt(xvel, 9),
            fmt(v_win, 8),
            fmt(run, 9),
            fmt(ctrl, 9),
            f"{nonfinite(latest):4d}",
            fmt(term, 6),
            fmt(prio, 8),
            fmt(actor, 9),
            fmt(action_delta, 7),
            fmt(reinit, 8, 0),
            fmt(sel_age, 7, 0),
            fmt(sel_dur, 8),
            fmt(key_l1, 8),
            fmt(adv_l1, 8),
        ]
        lines.append(" ".join(cols))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", help="Benchmark root; defaults to newest .oak-benchmarks run.")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh seconds.")
    parser.add_argument("--policy", action="append", help="Policy to show; repeatable.")
    parser.add_argument("--once", action="store_true", help="Print one table and exit.")
    parser.add_argument("--no-clear", action="store_true", help="Append tables instead of refreshing the terminal.")
    args = parser.parse_args()

    root = Path(args.root) if args.root else latest_root()
    if root is None:
        print("No .oak-benchmarks run found.", file=sys.stderr)
        return 1
    policy_filter = set(args.policy) if args.policy else None
    while True:
        text = render(root, policy_filter)
        if not args.no_clear and sys.stdout.isatty():
            sys.stdout.write("\033[2J\033[H")
        print(text, flush=True)
        if args.once:
            return 0
        time.sleep(max(args.interval, 0.2))


if __name__ == "__main__":
    raise SystemExit(main())
