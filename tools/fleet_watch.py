#!/usr/bin/env python3
"""Live view of a running fleet: tails aggregate.jsonl and prints, per policy,
the latest heartbeat (alive/dead, delivered-rate quantiles, aggregate steps/s).

Usage: fleet_watch.py <run-root> [--follow] [--interval 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def latest_rows(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    path = root / "aggregate.jsonl"
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial last line of a live run
            rows[row["policy"]] = row
    return rows


def render(rows: dict[str, dict]) -> str:
    if not rows:
        return "(no aggregate.jsonl rows yet)"
    out = [f"{'policy':8s} {'step':>10s} {'alive':>11s} "
           f"{'P10':>10s} {'P50':>10s} {'P90':>10s} {'agg sps':>10s} {'wall':>8s}"]
    for pol, r in sorted(rows.items()):
        total = r["alive"] + r["dead"]
        out.append(
            f"{pol:8s} {r['step']:>10,d} {r['alive']:>6d}/{total:<4d} "
            f"{r['deliveredRateP10']:>10.4g} {r['deliveredRateP50']:>10.4g} "
            f"{r['deliveredRateP90']:>10.4g} {r['aggStepsPerSecond']:>10,.0f} "
            f"{r['wallSeconds']:>7.0f}s")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()
    while True:
        print(render(latest_rows(args.root)), flush=True)
        if not args.follow:
            return 0
        time.sleep(args.interval)
        print()


if __name__ == "__main__":
    sys.exit(main())
