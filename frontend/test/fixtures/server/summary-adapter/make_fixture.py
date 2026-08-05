#!/usr/bin/env python3
"""Build the hermetic mini fleet root used by the summaryAdapter golden test.

The adapter contract is "adapt(v1 summary) == the v2 summary fleet.py now writes
for the same root", so the fixture pair must come from ONE root run through both
writers. This script only creates the root (meta.json + fleet.jsonl); the two
summaries are produced by `tools/fleet.py report` before and after the v2 rename
and checked in as summary-v1.json / summary-v2.json.

Deterministic: no RNG, no clock. Re-running it reproduces the root byte for byte.

    .venv/bin/python frontend/test/fixtures/server/summary-adapter/make_fixture.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "run"
STEPS = 1000
HEARTBEATS = [200, 400, 600, 800, 1000]
ENVS = [("HalfCheetah-v4", 0, 7, 230244), ("FourRooms-cont-v0", 8, 15, 210284)]
# lane 5 dies at 600 (survival curve + a death detector + an NF phase),
# every other lane runs to the horizon.
DEAD_LANE, DEAD_STEP = 5, 600


def env_of(seed: int) -> tuple[str, int]:
    for env_id, lo, hi, flops in ENVS:
        if lo <= seed <= hi:
            return env_id, flops
    raise AssertionError(seed)


def reward_sum(seed: int, step: int) -> float:
    """A per-lane rate that peaks mid-run then decays, so plasticity/peak, the
    solved-threshold count and retention are all exercised with real spread."""
    rate0 = (seed - 6) * 0.05           # -0.30 .. +0.45 across the 16 lanes
    peak = 0.6 + 0.05 * (seed % 4)
    shape = math.sin(math.pi * min(step, STEPS) / STEPS) ** 2
    return round(step * (rate0 + peak * shape * (0.4 if seed % 3 else 1.0)), 6)


def health(seed: int, step: int, dead: bool) -> dict:
    return {
        "rewardRate": round(0.001 * ((seed % 7) - 3) + 0.0002 * (step / 100), 6),
        "actionDeltaL2Mean": round(0.01 * (seed % 5) + 0.001 * (step / 100), 6),
        "actionDeltaPeak": round(0.05 * (seed % 5), 6),
        "continuingRestartCount": step // 250,
        "actorSupportEntropy": 0.0 if seed == 2 else round(0.5 + 0.01 * (seed % 9), 6),
        "terminationRate": round(0.02 * (seed % 6), 6),
        "optionDurationMean": round(3.0 + (seed % 4), 6),
        "keyboardL1": round(1.5 + 0.1 * (seed % 8), 6),
        "mainTd": round(0.01 * (seed % 3), 6),
        "planTd": 0.0,
        "nonFiniteByPhase": {
            "encode": 0,
            "learn": 0,
            "plan": 3 if dead else 0,
            "tdPsi": 0,
        },
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(16))
    meta = {
        "fleetSchemaVersion": 1,
        "command": "benchmark",
        "envId": "HalfCheetah-v4",
        "batch": len(seeds),
        "seeds": seeds,
        "steps": STEPS,
        "policies": ["oak"],
        "ablation": [],
        "ablationLabel": "none",
        "flopsPerLaneStep": 230244,
        "organCostVersion": 1,
        "startedAt": "2026-08-04T00:00:00Z",
        "envSchedule": [
            {"envId": e, "seedStart": lo, "seedEnd": hi, "flopsPerLaneStep": fl}
            for e, lo, hi, fl in ENVS
        ],
    }
    (ROOT / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")

    fleet_lines: list[str] = []
    death_lines: list[str] = []
    for seed in seeds:
        env_id, _ = env_of(seed)
        for step in HEARTBEATS:
            dead = seed == DEAD_LANE and step >= DEAD_STEP
            row = {
                "v": 1,
                "kind": "final" if step == STEPS else "hb",
                "envId": env_id,
                "policy": "oak",
                "ablationLabel": "none",
                "lane": seed,
                "seed": seed,
                "step": step,
                "wallSeconds": step / 100.0,
                "stepsPerSecond": 1000.0 + seed,
                "rewardSum": reward_sum(seed, step),
                "doneCount": step // 200,
                "kernelRho": round(0.001 * (seed % 5), 6),
                "paramAbsMax": 1000.0 + seed,
                "firstNonFiniteStep": DEAD_STEP if dead else None,
                "firstNonFiniteDetector": "paramExplosion" if dead else None,
                "health": health(seed, step, dead),
            }
            fleet_lines.append(json.dumps(row, sort_keys=True))
    death_lines.append(json.dumps({
        "seed": DEAD_LANE, "lane": DEAD_LANE, "step": DEAD_STEP,
        "detector": "paramExplosion", "policy": "oak",
        "rewardSumAtDeath": reward_sum(DEAD_LANE, DEAD_STEP),
        "lastGoodRewardRate": 0.01,
        "nfByPhase": {"encode": 0, "learn": 0, "plan": 3, "tdPsi": 0},
        "rhoAtDeath": 0.004, "paramAbsMax": 1005.0,
    }, sort_keys=True))
    (ROOT / "fleet.jsonl").write_text("\n".join(fleet_lines) + "\n")
    (ROOT / "deaths.jsonl").write_text("\n".join(death_lines) + "\n")

    agg_lines = []
    for step in HEARTBEATS:
        rates = sorted(reward_sum(s, step) / step for s in seeds)
        agg_lines.append(json.dumps({
            "step": step, "kind": "final" if step == STEPS else "hb", "policy": "oak",
            "alive": len(seeds) - (1 if step >= DEAD_STEP else 0),
            "dead": 1 if step >= DEAD_STEP else 0,
            "deliveredRateP10": rates[1], "deliveredRateP50": rates[8],
            "deliveredRateP90": rates[14],
            "recentRateP10": rates[1], "recentRateP50": rates[8], "recentRateP90": rates[14],
            "recentRateMax": rates[-1],
            "bestLaneSeed": 15, "bestLaneDeliveredRate": rates[-1],
            "bestLaneRewardSum": reward_sum(15, step),
            "positiveRateLaneCount": sum(1 for r in rates if r > 0),
            "positiveRateFraction": sum(1 for r in rates if r > 0) / len(rates),
            "aggStepsPerSecond": 16000.0, "wallSeconds": step / 100.0,
        }, sort_keys=True))
    (ROOT / "aggregate.jsonl").write_text("\n".join(agg_lines) + "\n")
    print(f"wrote {ROOT}")


if __name__ == "__main__":
    main()
