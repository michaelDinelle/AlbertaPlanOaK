#!/usr/bin/env python3
"""Generate scientific OaK benchmark summaries from a resolved manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any


CRITICAL_FLAG_NAMES = {
    "task_resolution_mismatch",
    "missing_run_start",
    "missing_run_end",
    "schema_drift_within_task",
    "nan_or_inf_detected",
    "incomplete_seed_coverage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scientific OaK benchmark reports")
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--task-summary-json", required=True)
    parser.add_argument("--family-summary-json", required=True)
    parser.add_argument("--research-verdict-json", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--baseline-policy", required=True)
    return parser.parse_args()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_python_package_versions(python_bin: str) -> dict[str, str]:
    try:
        version_probe = """
import importlib.metadata as m
import json
import platform

pkgs = ["envpool", "gymnasium", "numpy", "pygame"]
versions = {"python": platform.python_version()}
for pkg in pkgs:
    try:
        versions[pkg] = m.version(pkg)
    except Exception:
        versions[pkg] = "unavailable"
print(json.dumps(versions, sort_keys=True))
"""
        output = subprocess.check_output(
            [
                python_bin,
                "-c",
                version_probe,
            ],
            text=True,
        )
        return json.loads(output)
    except Exception as exc:  # pragma: no cover - best effort provenance
        return {"python": "unavailable", "error": str(exc)}


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def summarize(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_mean_ci(values: list[float], seed_key: str, samples: int = 2000) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    if len(values) == 1:
        return {"low": values[0], "high": values[0]}
    rng = random.Random(stable_json([seed_key, values]))
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(draw) / len(draw))
    means.sort()
    low_index = max(int(0.025 * (samples - 1)), 0)
    high_index = min(int(0.975 * (samples - 1)), samples - 1)
    return {"low": means[low_index], "high": means[high_index]}


def exact_sign_test(wins: int, losses: int) -> dict[str, Any]:
    n = wins + losses
    if n == 0:
        return {"test": "exact_sign_test", "pValue": 1.0, "significant": False}
    tail = min(wins, losses)
    cumulative = sum(math.comb(n, k) for k in range(0, tail + 1)) / (2**n)
    p_value = min(1.0, 2 * cumulative)
    return {"test": "exact_sign_test", "pValue": p_value, "significant": p_value < 0.05}


def read_records(metrics_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    flags: list[str] = []
    try:
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return None, None, ["missing_run_start", "missing_run_end"]
    run_start = next((record for record in records if record.get("type") == "run_start"), None)
    run_end = next((record for record in reversed(records) if record.get("type") == "run_end"), None)
    if run_start is None:
        flags.append("missing_run_start")
    if run_end is None:
        flags.append("missing_run_end")
    return run_start, run_end, flags


def numeric_health_flags(record: dict[str, Any] | None) -> list[str]:
    if record is None:
        return []
    for key, value in record.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return ["nan_or_inf_detected"]
    return []


def parse_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def group_mean(entries: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for entry in entries:
        value = entry["run_end"].get(key)
        if is_finite_number(value):
            values.append(float(value))
    return values


def task_health_flags(entries: list[dict[str, Any]], expected_seeds: set[int]) -> tuple[list[str], list[str]]:
    critical: set[str] = set()
    warnings: set[str] = set()

    observed_seeds = {entry["seed"] for entry in entries if entry.get("run_end") is not None}
    if observed_seeds != expected_seeds:
        critical.add("incomplete_seed_coverage")

    concrete_ids = {entry["run_start"].get("concreteEnvId") for entry in entries if entry.get("run_start")}
    env_names = {entry["run_start"].get("env") for entry in entries if entry.get("run_start")}
    manifest_ids = {entry["concrete_env_id"] for entry in entries}
    if concrete_ids - manifest_ids or env_names - manifest_ids:
        critical.add("task_resolution_mismatch")

    schema_fingerprints = {entry["run_start"].get("schemaFingerprint") for entry in entries if entry.get("run_start")}
    schema_fingerprints.discard(None)
    if len(schema_fingerprints) > 1:
        critical.add("schema_drift_within_task")

    if entries and all(float((entry["run_end"] or {}).get("plannerStepMsMean", 0.0) or 0.0) <= 0 for entry in entries):
        warnings.add("planner_never_engaged")

    if entries and all(float((entry["run_end"] or {}).get("activeRankMax", 0.0) or 0.0) <= 1.0 for entry in entries):
        warnings.add("active_rank_stuck_at_minimum")

    if entries and entries[0]["logical_env"] == "minigrid":
        total_completions = sum(int((entry["run_end"] or {}).get("doneCount", 0) or 0) for entry in entries)
        if total_completions == 0:
            warnings.add("zero_completions_sparse_suite")

    for entry in entries:
        critical.update(entry["flags"])

    return sorted(critical), sorted(warnings)


def build_policy_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    reward_sum_values = group_mean(entries, "rewardSum")
    reward_mean_values = group_mean(entries, "rewardMean")
    reward_std_values = group_mean(entries, "rewardStd")
    sps_values = group_mean(entries, "stepsPerSecond")
    done_values = group_mean(entries, "doneCount")
    reset_values = group_mean(entries, "resetCount")
    measured_step_values = group_mean(entries, "measuredStepCount")
    vector_update_values = group_mean(entries, "vectorUpdateCount")
    planner_active_values = group_mean(entries, "plannerActiveFraction")
    active_rank_values = group_mean(entries, "activeRankMean")
    active_rank_min_values = group_mean(entries, "activeRankMin")
    active_rank_max_values = group_mean(entries, "activeRankMax")
    completion_values = [
        (float(entry["run_end"].get("doneCount", 0) or 0) / max(float(entry["run_end"].get("measuredStepCount", 0) or 0), 1.0))
        for entry in entries
        if entry.get("run_end")
    ]

    return {
        "n": len(entries),
        "seeds": sorted(entry["seed"] for entry in entries),
        "backend": entries[0]["backend"] if entries else "",
        "league": entries[0]["league"] if entries else "",
        "logicalEnv": entries[0]["logical_env"] if entries else "",
        "concreteEnvId": entries[0]["concrete_env_id"] if entries else "",
        "schemaFingerprint": entries[0]["schema_fingerprint"] if entries else "",
        "observationSpaceKind": entries[0]["observation_space_kind"] if entries else "",
        "observationFlatDim": entries[0]["observation_flat_dim"] if entries else "",
        "actionSpaceKind": entries[0]["action_space_kind"] if entries else "",
        "actionFlatDim": entries[0]["action_flat_dim"] if entries else "",
        "actionDiscreteCount": entries[0]["action_discrete_count"] if entries else "",
        "actionShape": json.loads(entries[0]["action_shape"]) if entries and entries[0]["action_shape"] else [],
        "rewardSum": summarize(reward_sum_values),
        "rewardMean": summarize(reward_mean_values),
        "rewardStd": summarize(reward_std_values),
        "stepsPerSecond": summarize(sps_values),
        "doneCount": summarize(done_values),
        "resetCount": summarize(reset_values),
        "globalEnvStepCount": summarize(measured_step_values),
        "vectorUpdateCount": summarize(vector_update_values),
        "completionRate": summarize(completion_values),
        "plannerActiveFraction": summarize(planner_active_values),
        "activeRankMean": summarize(active_rank_values),
        "activeRankMin": summarize(active_rank_min_values),
        "activeRankMax": summarize(active_rank_max_values),
        "runArtifacts": [
            {
                "seed": entry["seed"],
                "metricsPath": entry["metrics_path"],
                "logPath": entry["log_path"],
                "runId": (entry["run_end"] or {}).get("runId") or "",
                "stopReason": (entry["run_end"] or {}).get("stopReason"),
                "flags": entry["flags"],
            }
            for entry in sorted(entries, key=lambda item: item["seed"])
        ],
    }


def build_comparison(pair_key: str, left_entries: list[dict[str, Any]], right_entries: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_seed = {entry["seed"]: entry for entry in left_entries}
    right_by_seed = {entry["seed"]: entry for entry in right_entries}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    reward_sum_deltas: list[float] = []
    reward_mean_deltas: list[float] = []
    sps_deltas: list[float] = []
    wins = losses = ties = 0

    for seed in shared_seeds:
        left = left_by_seed[seed]["run_end"]
        right = right_by_seed[seed]["run_end"]
        reward_sum_delta = float(left.get("rewardSum", 0.0) or 0.0) - float(right.get("rewardSum", 0.0) or 0.0)
        reward_mean_delta = float(left.get("rewardMean", 0.0) or 0.0) - float(right.get("rewardMean", 0.0) or 0.0)
        sps_delta = float(left.get("stepsPerSecond", 0.0) or 0.0) - float(right.get("stepsPerSecond", 0.0) or 0.0)
        reward_sum_deltas.append(reward_sum_delta)
        reward_mean_deltas.append(reward_mean_delta)
        sps_deltas.append(sps_delta)
        if reward_sum_delta > 0:
            wins += 1
        elif reward_sum_delta < 0:
            losses += 1
        else:
            ties += 1

    reward_sum_summary = summarize(reward_sum_deltas)
    reward_mean_summary = summarize(reward_mean_deltas)
    sps_summary = summarize(sps_deltas)
    sign_test = exact_sign_test(wins, losses)

    return {
        "pairedSeeds": shared_seeds,
        "rewardSumDelta": {
            **(reward_sum_summary or {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}),
            "bootstrapCI95": bootstrap_mean_ci(reward_sum_deltas, pair_key) if reward_sum_deltas else {"low": 0.0, "high": 0.0},
        },
        "rewardMeanDelta": {
            **(reward_mean_summary or {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}),
            "bootstrapCI95": bootstrap_mean_ci(reward_mean_deltas, pair_key + "::reward_mean") if reward_mean_deltas else {"low": 0.0, "high": 0.0},
        },
        "stepsPerSecondDelta": {
            **(sps_summary or {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}),
            "bootstrapCI95": bootstrap_mean_ci(sps_deltas, pair_key + "::sps") if sps_deltas else {"low": 0.0, "high": 0.0},
        },
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "significance": sign_test,
    }


def family_macro_summary(task_summaries: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    reward_sum_means = []
    reward_mean_means = []
    sps_means = []
    completion_means = []
    planner_fraction_means = []
    active_rank_means = []
    for task in task_summaries:
        policy_summary = task["policySummaries"].get(policy_name)
        if not policy_summary:
            continue
        reward_sum_means.append(policy_summary["rewardSum"]["mean"])
        reward_mean_means.append(policy_summary["rewardMean"]["mean"])
        sps_means.append(policy_summary["stepsPerSecond"]["mean"])
        completion_means.append(policy_summary["completionRate"]["mean"])
        planner_fraction_means.append(policy_summary["plannerActiveFraction"]["mean"])
        active_rank_means.append(policy_summary["activeRankMean"]["mean"])

    return {
        "taskCount": len(reward_sum_means),
        "rewardSumMacroMean": summarize(reward_sum_means),
        "rewardMeanMacroMean": summarize(reward_mean_means),
        "stepsPerSecondMacroMean": summarize(sps_means),
        "completionRateMacroMean": summarize(completion_means),
        "plannerActiveFractionMacroMean": summarize(planner_fraction_means),
        "activeRankMacroMean": summarize(active_rank_means),
    }


def build_family_comparison(task_summaries: list[dict[str, Any]], pair_name: str) -> dict[str, Any]:
    task_deltas = []
    improved = regressed = tied = 0
    for task in task_summaries:
        comparison = task["comparisons"].get(pair_name)
        if not comparison:
            continue
        mean_delta = comparison["rewardSumDelta"]["mean"]
        task_deltas.append(mean_delta)
        if mean_delta > 0:
            improved += 1
        elif mean_delta < 0:
            regressed += 1
        else:
            tied += 1

    summary = summarize(task_deltas)
    return {
        "taskCount": len(task_deltas),
        "rewardSumDeltaMacroMean": summary,
        "taskWinLossTie": {"wins": improved, "losses": regressed, "ties": tied},
        "significance": exact_sign_test(improved, regressed),
    }


def format_stat(stat: dict[str, Any] | None, digits: int = 5) -> str:
    if not stat:
        return "n/a"
    return f"{stat['mean']:.{digits}f} +/- {stat['std']:.{digits}f}"


def format_ci(stat: dict[str, Any] | None, digits: int = 5) -> str:
    if not stat:
        return "n/a"
    ci = stat["bootstrapCI95"]
    return f"{stat['mean']:.{digits}f} [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]"


def main() -> int:
    args = parse_args()
    bench_dir = Path(args.bench_dir)
    manifest_path = Path(args.manifest)
    manifest_rows = parse_manifest(manifest_path)
    python_versions = load_python_package_versions(args.python_bin)
    generated_at = datetime.now(timezone.utc).isoformat()

    run_entries: list[dict[str, Any]] = []
    for row in manifest_rows:
        metrics_path = Path(row["metrics_path"])
        run_start, run_end, flags = read_records(metrics_path)
        flags.extend(numeric_health_flags(run_start))
        flags.extend(numeric_health_flags(run_end))
        run_entries.append(
            {
                **row,
                "seed": int(row["seed"]),
                "num_envs": int(row["num_envs"]),
                "warmup_steps": int(row["warmup_steps"]),
                "stop_steps": int(row["stop_steps"]) if row["stop_steps"] else None,
                "stop_seconds": float(row["stop_seconds"]) if row["stop_seconds"] else None,
                "observation_flat_dim": int(row["observation_flat_dim"]) if row["observation_flat_dim"] else None,
                "action_flat_dim": int(row["action_flat_dim"]) if row["action_flat_dim"] else None,
                "action_discrete_count": int(row["action_discrete_count"]) if row["action_discrete_count"] else None,
                "run_start": run_start,
                "run_end": run_end,
                "flags": sorted(set(flags)),
            }
        )

    manifest_signature = [
        {
            "league": entry["league"],
            "mode": entry["mode"],
            "logicalEnv": entry["logical_env"],
            "concreteEnvId": entry["concrete_env_id"],
            "policy": entry["policy"],
            "seed": entry["seed"],
            "numEnvs": entry["num_envs"],
            "warmupSteps": entry["warmup_steps"],
            "stopSteps": entry["stop_steps"],
            "stopSeconds": entry["stop_seconds"],
            "schemaFingerprint": entry["schema_fingerprint"],
        }
        for entry in run_entries
    ]
    config_hash = hashlib.sha256(stable_json(manifest_signature).encode("utf-8")).hexdigest()
    policies = sorted({entry["policy"] for entry in run_entries})
    modes = sorted({entry["mode"] for entry in run_entries})
    expected_seeds = {entry["seed"] for entry in run_entries}
    resolved_tasks = sorted(
        {
            (entry["league"], entry["logical_env"], entry["concrete_env_id"])
            for entry in run_entries
        }
    )

    task_buckets: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for entry in run_entries:
        task_buckets[(entry["mode"], entry["logical_env"], entry["concrete_env_id"])][entry["policy"]].append(entry)

    task_summaries: list[dict[str, Any]] = []
    critical_flags: set[str] = set()
    warning_flags: set[str] = set()
    for (mode, logical_env, concrete_env_id), policy_map in sorted(task_buckets.items()):
        policy_summaries: dict[str, Any] = {}
        task_entries = [entry for entries in policy_map.values() for entry in entries]
        task_critical, task_warnings = task_health_flags(task_entries, expected_seeds)
        critical_flags.update(task_critical)
        warning_flags.update(task_warnings)
        for policy, entries in sorted(policy_map.items()):
            policy_summaries[policy] = build_policy_summary(entries)

        comparisons: dict[str, Any] = {}
        for left_policy, right_policy in combinations(sorted(policy_map), 2):
            pair_name = f"{left_policy}__vs__{right_policy}"
            comparisons[pair_name] = build_comparison(pair_name, policy_map[left_policy], policy_map[right_policy])

        task_summaries.append(
            {
                "mode": mode,
                "league": task_entries[0]["league"],
                "logicalEnv": logical_env,
                "concreteEnvId": concrete_env_id,
                "policySummaries": policy_summaries,
                "comparisons": comparisons,
                "health": {"critical": task_critical, "warnings": task_warnings},
            }
        )

    family_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task_summary in task_summaries:
        family_buckets[(task_summary["mode"], task_summary["logicalEnv"])].append(task_summary)

    family_summaries: list[dict[str, Any]] = []
    for (mode, logical_env), family_tasks in sorted(family_buckets.items()):
        policy_summaries: dict[str, Any] = {}
        family_comparisons: dict[str, Any] = {}
        family_critical = sorted({flag for task in family_tasks for flag in task["health"]["critical"]})
        family_warnings = sorted({flag for task in family_tasks for flag in task["health"]["warnings"]})
        for policy in policies:
            policy_summaries[policy] = family_macro_summary(family_tasks, policy)
        for left_policy, right_policy in combinations(policies, 2):
            pair_name = f"{left_policy}__vs__{right_policy}"
            family_comparisons[pair_name] = build_family_comparison(family_tasks, pair_name)
        family_summaries.append(
            {
                "mode": mode,
                "logicalEnv": logical_env,
                "taskCount": len(family_tasks),
                "policySummaries": policy_summaries,
                "comparisons": family_comparisons,
                "health": {"critical": family_critical, "warnings": family_warnings},
            }
        )

    task_lookup = {(task["mode"], task["concreteEnvId"]): task for task in task_summaries}
    dual_axis_complete = {"quality", "throughput"}.issubset(set(modes))
    pareto_by_policy: dict[str, dict[str, list[dict[str, Any]]]] = {
        policy: {"improvements": [], "regressions": [], "mixed": []}
        for policy in policies
        if policy != args.baseline_policy
    }

    for policy in list(pareto_by_policy):
        for _league, _logical_env, concrete_env_id in resolved_tasks:
            quality_task = task_lookup.get(("quality", concrete_env_id))
            throughput_task = task_lookup.get(("throughput", concrete_env_id))
            quality_pair = None
            throughput_pair = None
            ordered_pair = tuple(sorted([policy, args.baseline_policy]))
            pair_name = f"{ordered_pair[0]}__vs__{ordered_pair[1]}"
            direction = 1 if ordered_pair[0] == policy else -1
            if quality_task and pair_name in quality_task["comparisons"]:
                quality_pair = quality_task["comparisons"][pair_name]["rewardSumDelta"]["mean"] * direction
            if throughput_task and pair_name in throughput_task["comparisons"]:
                throughput_pair = throughput_task["comparisons"][pair_name]["rewardSumDelta"]["mean"] * direction
            if quality_pair is None and throughput_pair is None:
                continue

            verdict = {
                "concreteEnvId": concrete_env_id,
                "qualityRewardDelta": quality_pair,
                "throughputRewardDelta": throughput_pair,
            }
            if dual_axis_complete and quality_pair is not None and throughput_pair is not None:
                if quality_pair >= 0 and throughput_pair >= 0 and (quality_pair > 0 or throughput_pair > 0):
                    pareto_by_policy[policy]["improvements"].append(verdict)
                elif quality_pair <= 0 and throughput_pair <= 0 and (quality_pair < 0 or throughput_pair < 0):
                    pareto_by_policy[policy]["regressions"].append(verdict)
                else:
                    pareto_by_policy[policy]["mixed"].append(verdict)
            else:
                target = "improvements" if (quality_pair or 0) > 0 else "regressions" if (quality_pair or 0) < 0 else "mixed"
                pareto_by_policy[policy][target].append(verdict)

    top_improvements = []
    top_regressions = []
    for policy, verdicts in pareto_by_policy.items():
        top_improvements.extend(
            sorted(
                (
                    {
                        "policy": policy,
                        **item,
                        "score": (item.get("qualityRewardDelta") or 0) + (item.get("throughputRewardDelta") or 0),
                    }
                    for item in verdicts["improvements"]
                ),
                key=lambda item: item["score"],
                reverse=True,
            )
        )
        top_regressions.extend(
            sorted(
                (
                    {
                        "policy": policy,
                        **item,
                        "score": (item.get("qualityRewardDelta") or 0) + (item.get("throughputRewardDelta") or 0),
                    }
                    for item in verdicts["regressions"]
                ),
                key=lambda item: item["score"],
            )
        )

    research_verdict = {
        "generatedAt": generated_at,
        "benchDir": str(bench_dir),
        "gitRevision": args.git_revision,
        "baselinePolicy": args.baseline_policy,
        "leagueNames": sorted({league for league, _logical_env, _concrete_env_id in resolved_tasks}),
        "dualAxisComplete": dual_axis_complete,
        "promotionEligible": dual_axis_complete and not critical_flags,
        "criticalFlags": sorted(critical_flags),
        "warningFlags": sorted(warning_flags),
        "paretoByPolicy": pareto_by_policy,
        "topImprovements": top_improvements[:20],
        "topRegressions": top_regressions[:20],
    }

    task_summary = {
        "generatedAt": generated_at,
        "benchDir": str(bench_dir),
        "gitRevision": args.git_revision,
        "manifestPath": str(manifest_path),
        "configHash": config_hash,
        "pythonVersions": python_versions,
        "policies": policies,
        "modes": modes,
        "resolvedTasks": [
            {"league": league, "logicalEnv": logical_env, "concreteEnvId": concrete_env_id}
            for league, logical_env, concrete_env_id in resolved_tasks
        ],
        "taskSummaries": task_summaries,
    }

    family_summary = {
        "generatedAt": generated_at,
        "benchDir": str(bench_dir),
        "gitRevision": args.git_revision,
        "configHash": config_hash,
        "policies": policies,
        "modes": modes,
        "familySummaries": family_summaries,
    }

    combined_summary = {
        "generatedAt": generated_at,
        "benchDir": str(bench_dir),
        "gitRevision": args.git_revision,
        "manifestPath": str(manifest_path),
        "configHash": config_hash,
        "pythonVersions": python_versions,
        "baselinePolicy": args.baseline_policy,
        "taskSummary": task_summary,
        "familySummary": family_summary,
        "researchVerdict": research_verdict,
    }

    lines = [
        "# Scientific OaK Benchmark",
        "",
        f"- Generated: {generated_at}",
        f"- Output dir: `{bench_dir}`",
        f"- Git revision: `{args.git_revision}`",
        f"- Baseline policy: `{args.baseline_policy}`",
        f"- Policies: {', '.join(policies)}",
        f"- Modes: {', '.join(modes)}",
        f"- Resolved tasks: {len(resolved_tasks)}",
        f"- Config hash: `{config_hash}`",
        "",
        "## Provenance",
        "",
        f"- Python packages: `{stable_json(python_versions)}`",
        f"- Manifest: `{manifest_path}`",
        "",
        "## Family Macro Summary",
        "",
        "| mode | family | policy | tasks | reward_sum | reward_mean | sps | completion | planner_active | health |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for family in family_summaries:
        health = family["health"]
        health_label = ", ".join(health["critical"] + health["warnings"]) if health["critical"] or health["warnings"] else "ok"
        for policy in policies:
            policy_summary = family["policySummaries"][policy]
            lines.append(
                f"| {family['mode']} | {family['logicalEnv']} | {policy} | {policy_summary['taskCount']} | "
                f"{format_stat(policy_summary['rewardSumMacroMean'])} | "
                f"{format_stat(policy_summary['rewardMeanMacroMean'])} | "
                f"{format_stat(policy_summary['stepsPerSecondMacroMean'], digits=2)} | "
                f"{format_stat(policy_summary['completionRateMacroMean'], digits=5)} | "
                f"{format_stat(policy_summary['plannerActiveFractionMacroMean'], digits=5)} | "
                f"{health_label} |"
            )

    lines.extend(
        [
            "",
            f"## Comparisons vs `{args.baseline_policy}`",
            "",
            "| mode | family | candidate | reward_sum_delta | task wins/losses/ties | p_value | health |",
            "| --- | --- | --- | ---: | --- | ---: | --- |",
        ]
    )

    for family in family_summaries:
        health = family["health"]
        health_label = ", ".join(health["critical"] + health["warnings"]) if health["critical"] or health["warnings"] else "ok"
        for policy in policies:
            if policy == args.baseline_policy:
                continue
            pair_name = "__vs__".join(sorted([policy, args.baseline_policy]))
            comparison = family["comparisons"].get(pair_name)
            if not comparison:
                continue
            task_wlt = comparison["taskWinLossTie"]
            lines.append(
                f"| {family['mode']} | {family['logicalEnv']} | {policy} | "
                f"{format_stat(comparison['rewardSumDeltaMacroMean'])} | "
                f"{task_wlt['wins']}/{task_wlt['losses']}/{task_wlt['ties']} | "
                f"{comparison['significance']['pValue']:.5f} | {health_label} |"
            )

    lines.extend(
        [
            "",
            "## Harness Health",
            "",
            f"- Critical flags: {', '.join(sorted(critical_flags)) if critical_flags else 'none'}",
            f"- Warning flags: {', '.join(sorted(warning_flags)) if warning_flags else 'none'}",
            f"- Promotion eligible: {'yes' if research_verdict['promotionEligible'] else 'no'}",
            "",
            "## Top Improvements",
            "",
            "| policy | task | quality_delta | throughput_delta |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for improvement in research_verdict["topImprovements"][:10]:
        lines.append(
            f"| {improvement['policy']} | {improvement['concreteEnvId']} | "
            f"{(improvement.get('qualityRewardDelta') or 0):.5f} | "
            f"{(improvement.get('throughputRewardDelta') or 0):.5f} |"
        )
    if not research_verdict["topImprovements"]:
        lines.append("| n/a | n/a | 0.00000 | 0.00000 |")

    lines.extend(
        [
            "",
            "## Top Regressions",
            "",
            "| policy | task | quality_delta | throughput_delta |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for regression in research_verdict["topRegressions"][:10]:
        lines.append(
            f"| {regression['policy']} | {regression['concreteEnvId']} | "
            f"{(regression.get('qualityRewardDelta') or 0):.5f} | "
            f"{(regression.get('throughputRewardDelta') or 0):.5f} |"
        )
    if not research_verdict["topRegressions"]:
        lines.append("| n/a | n/a | 0.00000 | 0.00000 |")

    Path(args.task_summary_json).write_text(json.dumps(task_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.family_summary_json).write_text(json.dumps(family_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.research_verdict_json).write_text(json.dumps(research_verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.summary_json).write_text(json.dumps(combined_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 1 if critical_flags else 0


if __name__ == "__main__":
    sys.exit(main())
