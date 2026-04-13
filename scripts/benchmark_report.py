#!/usr/bin/env python3
"""Generate a single canonical benchmark.json from benchmark run artifacts."""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a canonical OaK benchmark JSON artifact")
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark-json", required=True)
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
        output = subprocess.check_output([python_bin, "-c", version_probe], text=True)
        return json.loads(output)
    except Exception as exc:  # pragma: no cover - best effort provenance
        _ = exc
        return {
            "python": "unavailable",
            "envpool": "unavailable",
            "gymnasium": "unavailable",
            "numpy": "unavailable",
            "pygame": "unavailable",
        }


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def summarize(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
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


def parse_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def read_records(metrics_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return None, None
    run_start = next((record for record in records if record.get("type") == "run_start"), None)
    run_end = next((record for record in reversed(records) if record.get("type") == "run_end"), None)
    return run_start, run_end


def extract_dynamic_schema(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if record is None:
        return []
    schema = record.get("dynamicTelemetrySchema")
    if isinstance(schema, dict):
        descriptors = schema.get("descriptors")
        if isinstance(descriptors, list):
            return [descriptor for descriptor in descriptors if isinstance(descriptor, dict)]
    return []


def extract_dynamic_values(record: dict[str, Any] | None) -> list[float]:
    if record is None:
        return []
    values = record.get("dynamicTelemetryValues")
    if not isinstance(values, list):
        return []
    output: list[float] = []
    for value in values:
        if is_finite_number(value):
            output.append(float(value))
        else:
            output.append(0.0)
    return output


def metric_values(entries: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for entry in entries:
        run_end = entry.get("run_end")
        if run_end is None:
            continue
        value = run_end.get(key)
        if is_finite_number(value):
            values.append(float(value))
    return values


def summarize_metric(entries: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return summarize(metric_values(entries, key))


def completion_values(entries: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for entry in entries:
        run_end = entry.get("run_end")
        if run_end is None:
            continue
        done_count = float(run_end.get("doneCount", 0.0) or 0.0)
        measured_step_count = max(float(run_end.get("measuredStepCount", 0.0) or 0.0), 1.0)
        values.append(done_count / measured_step_count)
    return values


def stop_reason_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in entries:
        run_end = entry.get("run_end")
        if not run_end:
            continue
        stop_reason = run_end.get("stopReason")
        if stop_reason:
            counts[str(stop_reason)] += 1
    return dict(sorted(counts.items()))


def summarize_dynamic_telemetry(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not entries:
        return [], {}

    schema = extract_dynamic_schema(entries[0].get("run_start"))
    if not schema:
        return [], {}

    summaries: dict[str, Any] = {}
    for index, descriptor in enumerate(schema):
        key = str(descriptor.get("key") or f"metric_{index}")
        values: list[float] = []
        for entry in entries:
            run_end_values = extract_dynamic_values(entry.get("run_end"))
            if index < len(run_end_values) and is_finite_number(run_end_values[index]):
                values.append(float(run_end_values[index]))
        if values:
            summaries[key] = summarize(values)
    return schema, summaries


def build_policy_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    first = entries[0] if entries else {}
    dynamic_schema, dynamic_telemetry = summarize_dynamic_telemetry(entries)
    return {
        "runCount": len(entries),
        "seeds": sorted(entry["seed"] for entry in entries),
        "backend": first.get("backend", ""),
        "league": first.get("league", ""),
        "logicalEnv": first.get("logical_env", ""),
        "concreteEnvId": first.get("concrete_env_id", ""),
        "numEnvs": first.get("num_envs"),
        "warmupSteps": first.get("warmup_steps"),
        "stopSteps": first.get("stop_steps"),
        "schemaFingerprint": first.get("schema_fingerprint", ""),
        "observationSpaceKind": first.get("observation_space_kind", ""),
        "observationFlatDim": first.get("observation_flat_dim"),
        "actionSpaceKind": first.get("action_space_kind", ""),
        "actionFlatDim": first.get("action_flat_dim"),
        "actionDiscreteCount": first.get("action_discrete_count"),
        "actionShape": first.get("action_shape", []),
        "rewardSum": summarize_metric(entries, "rewardSum"),
        "rewardMean": summarize_metric(entries, "rewardMean"),
        "rewardStd": summarize_metric(entries, "rewardStd"),
        "doneCount": summarize_metric(entries, "doneCount"),
        "resetCount": summarize_metric(entries, "resetCount"),
        "measuredStepCount": summarize_metric(entries, "measuredStepCount"),
        "vectorUpdateCount": summarize_metric(entries, "vectorUpdateCount"),
        "measuredSeconds": summarize_metric(entries, "measuredSeconds"),
        "stepsPerSecond": summarize_metric(entries, "stepsPerSecond"),
        "completionRate": summarize(completion_values(entries)),
        "plannerStepMsMean": summarize_metric(entries, "plannerStepMsMean"),
        "plannerActiveFraction": summarize_metric(entries, "plannerActiveFraction"),
        "activeRankMean": summarize_metric(entries, "activeRankMean"),
        "activeRankMin": summarize_metric(entries, "activeRankMin"),
        "activeRankMax": summarize_metric(entries, "activeRankMax"),
        "activeGVFsMean": summarize_metric(entries, "activeGVFsMean"),
        "activeOptionsMean": summarize_metric(entries, "activeOptionsMean"),
        "actionL2Mean": summarize_metric(entries, "actionL2Mean"),
        "stateL2Mean": summarize_metric(entries, "stateL2Mean"),
        "agentStepMsMean": summarize_metric(entries, "agentStepMsMean"),
        "envStepMsMean": summarize_metric(entries, "envStepMsMean"),
        "totalStepMsMean": summarize_metric(entries, "totalStepMsMean"),
        "dynamicTelemetrySchema": dynamic_schema,
        "dynamicTelemetry": dynamic_telemetry,
        "stopReasonCounts": stop_reason_counts(entries),
        "runArtifacts": [
            {
                "seed": entry["seed"],
                "metricsPath": entry["metrics_path"],
                "logPath": entry["log_path"],
                "runId": (entry["run_end"] or {}).get("runId") or (entry["run_start"] or {}).get("runId") or "",
                "stopReason": (entry["run_end"] or {}).get("stopReason"),
            }
            for entry in sorted(entries, key=lambda item: item["seed"])
        ],
    }


def build_comparison(pair_key: str, left_entries: list[dict[str, Any]], right_entries: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_seed = {entry["seed"]: entry for entry in left_entries if entry.get("run_end") is not None}
    right_by_seed = {entry["seed"]: entry for entry in right_entries if entry.get("run_end") is not None}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    reward_sum_deltas: list[float] = []
    reward_mean_deltas: list[float] = []
    steps_per_second_deltas: list[float] = []
    completion_rate_deltas: list[float] = []
    wins = losses = ties = 0

    for seed in shared_seeds:
        left_run_end = left_by_seed[seed]["run_end"]
        right_run_end = right_by_seed[seed]["run_end"]
        left_reward_sum = float(left_run_end.get("rewardSum", 0.0) or 0.0)
        right_reward_sum = float(right_run_end.get("rewardSum", 0.0) or 0.0)
        left_reward_mean = float(left_run_end.get("rewardMean", 0.0) or 0.0)
        right_reward_mean = float(right_run_end.get("rewardMean", 0.0) or 0.0)
        left_steps_per_second = float(left_run_end.get("stepsPerSecond", 0.0) or 0.0)
        right_steps_per_second = float(right_run_end.get("stepsPerSecond", 0.0) or 0.0)
        left_completion_rate = float(left_run_end.get("doneCount", 0.0) or 0.0) / max(float(left_run_end.get("measuredStepCount", 0.0) or 0.0), 1.0)
        right_completion_rate = float(right_run_end.get("doneCount", 0.0) or 0.0) / max(float(right_run_end.get("measuredStepCount", 0.0) or 0.0), 1.0)

        reward_sum_delta = left_reward_sum - right_reward_sum
        reward_mean_delta = left_reward_mean - right_reward_mean
        steps_per_second_delta = left_steps_per_second - right_steps_per_second
        completion_rate_delta = left_completion_rate - right_completion_rate

        reward_sum_deltas.append(reward_sum_delta)
        reward_mean_deltas.append(reward_mean_delta)
        steps_per_second_deltas.append(steps_per_second_delta)
        completion_rate_deltas.append(completion_rate_delta)

        if reward_sum_delta > 0:
            wins += 1
        elif reward_sum_delta < 0:
            losses += 1
        else:
            ties += 1

    def comparison_summary(values: list[float], seed_suffix: str) -> dict[str, Any]:
        base = summarize(values) or {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            **base,
            "bootstrapCI95": bootstrap_mean_ci(values, f"{pair_key}::{seed_suffix}") if values else {"low": 0.0, "high": 0.0},
        }

    return {
        "pairedSeeds": shared_seeds,
        "rewardSumDelta": comparison_summary(reward_sum_deltas, "reward_sum"),
        "rewardMeanDelta": comparison_summary(reward_mean_deltas, "reward_mean"),
        "stepsPerSecondDelta": comparison_summary(steps_per_second_deltas, "steps_per_second"),
        "completionRateDelta": comparison_summary(completion_rate_deltas, "completion_rate"),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "significance": exact_sign_test(wins, losses),
    }


def build_runs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: (item["league"], item["logical_env"], item["concrete_env_id"], item["policy"], item["seed"])):
        run_end = entry.get("run_end") or {}
        run_start = entry.get("run_start") or {}
        runs.append(
            {
                "league": entry["league"],
                "logicalEnv": entry["logical_env"],
                "concreteEnvId": entry["concrete_env_id"],
                "backend": entry["backend"],
                "policy": entry["policy"],
                "seed": entry["seed"],
                "numEnvs": entry["num_envs"],
                "warmupSteps": entry["warmup_steps"],
                "stopSteps": entry["stop_steps"],
                "runId": run_end.get("runId") or run_start.get("runId") or "",
                "metricsPath": entry["metrics_path"],
                "logPath": entry["log_path"],
                "schemaFingerprint": entry["schema_fingerprint"],
                "observationSpaceKind": entry["observation_space_kind"],
                "observationFlatDim": entry["observation_flat_dim"],
                "actionSpaceKind": entry["action_space_kind"],
                "actionFlatDim": entry["action_flat_dim"],
                "actionDiscreteCount": entry["action_discrete_count"],
                "actionShape": entry["action_shape"],
                "stopReason": run_end.get("stopReason"),
                "dynamicTelemetrySchema": extract_dynamic_schema(run_start),
                "dynamicTelemetryValues": extract_dynamic_values(run_end),
            }
        )
    return runs


def validate_numeric_record(record: dict[str, Any] | None, label: str) -> list[str]:
    if record is None:
        return []

    messages: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                messages.append(f"{label}: non-finite numeric value for {path}")
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    for key, value in record.items():
        walk(value, key)
    return messages


def validate_entry(entry: dict[str, Any]) -> list[str]:
    descriptor = (
        f"{entry['league']}/{entry['logical_env']}/{entry['concrete_env_id']}/"
        f"{entry['policy']}/seed{entry['seed']}"
    )
    messages: list[str] = []
    run_start = entry.get("run_start")
    run_end = entry.get("run_end")

    if run_start is None:
        messages.append(f"{descriptor}: missing run_start in {entry['metrics_path']}")
    if run_end is None:
        messages.append(f"{descriptor}: missing run_end in {entry['metrics_path']}")

    if run_start is not None:
        start_concrete = str(run_start.get("concreteEnvId") or run_start.get("env") or "")
        start_logical = str(run_start.get("logicalEnv") or "")
        if start_concrete != entry["concrete_env_id"]:
            messages.append(
                f"{descriptor}: task resolution mismatch in run_start; "
                f"recorded {start_concrete!r}, expected {entry['concrete_env_id']!r}"
            )
        if start_logical != entry["logical_env"]:
            messages.append(
                f"{descriptor}: logical env mismatch in run_start; "
                f"recorded {start_logical!r}, expected {entry['logical_env']!r}"
            )
        if str(run_start.get("schemaFingerprint") or "") == "":
            messages.append(f"{descriptor}: missing schema fingerprint in run_start")

    if run_end is not None:
        end_concrete = str(run_end.get("concreteEnvId") or run_end.get("env") or "")
        end_logical = str(run_end.get("logicalEnv") or "")
        if end_concrete != entry["concrete_env_id"]:
            messages.append(
                f"{descriptor}: task resolution mismatch in run_end; "
                f"recorded {end_concrete!r}, expected {entry['concrete_env_id']!r}"
            )
        if end_logical != entry["logical_env"]:
            messages.append(
                f"{descriptor}: logical env mismatch in run_end; "
                f"recorded {end_logical!r}, expected {entry['logical_env']!r}"
            )

    backend = str(
        (run_start or {}).get("backend")
        or (run_end or {}).get("backend")
        or entry.get("backend")
        or ""
    )
    if entry["policy"] == "oak" and backend != "metal":
        messages.append(f"{descriptor}: OaK benchmark must run on Metal, recorded backend={backend!r}")

    messages.extend(validate_numeric_record(run_start, f"{descriptor}: run_start"))
    messages.extend(validate_numeric_record(run_end, f"{descriptor}: run_end"))
    return messages


def validate_task(entries: list[dict[str, Any]]) -> list[str]:
    first = entries[0]
    descriptor = f"{first['league']}/{first['logical_env']}/{first['concrete_env_id']}"
    messages: list[str] = []

    expected_pairs = {(entry["policy"], entry["seed"]) for entry in entries}
    observed_pairs = {(entry["policy"], entry["seed"]) for entry in entries if entry.get("run_end") is not None}
    if observed_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - observed_pairs)
        messages.append(f"{descriptor}: incomplete run coverage; missing {missing_pairs}")

    schema_fingerprints = {
        str(fingerprint)
        for entry in entries
        for fingerprint in [
            (entry.get("run_start") or {}).get("schemaFingerprint"),
            (entry.get("run_end") or {}).get("schemaFingerprint"),
        ]
        if fingerprint not in {None, ""}
    }
    if len(schema_fingerprints) > 1:
        messages.append(f"{descriptor}: schema drift within task; fingerprints={sorted(schema_fingerprints)}")

    return messages


def validate_policy_group(entries: list[dict[str, Any]]) -> list[str]:
    first = entries[0]
    descriptor = (
        f"{first['league']}/{first['logical_env']}/{first['concrete_env_id']}/"
        f"{first['policy']}"
    )
    schema_signatures = {
        stable_json(extract_dynamic_schema(entry.get("run_start")))
        for entry in entries
    }
    if len(schema_signatures) > 1:
        return [f"{descriptor}: dynamic telemetry schema drift within policy group"]
    return []


def main() -> int:
    args = parse_args()
    bench_dir = Path(args.bench_dir)
    manifest_path = Path(args.manifest)
    benchmark_json_path = Path(args.benchmark_json)
    if benchmark_json_path.exists():
        benchmark_json_path.unlink()

    manifest_rows = parse_manifest(manifest_path)
    python_versions = load_python_package_versions(args.python_bin)
    generated_at = datetime.now(timezone.utc).isoformat()

    run_entries: list[dict[str, Any]] = []
    validation_messages: list[str] = []
    for row in manifest_rows:
        metrics_path = Path(row["metrics_path"])
        run_start, run_end = read_records(metrics_path)
        action_shape = json.loads(row["action_shape"]) if row.get("action_shape") else []
        entry = {
            **row,
            "seed": int(row["seed"]),
            "num_envs": int(row["num_envs"]),
            "warmup_steps": int(row["warmup_steps"]),
            "stop_steps": int(row["stop_steps"]) if row["stop_steps"] else None,
            "observation_flat_dim": int(row["observation_flat_dim"]) if row["observation_flat_dim"] else None,
            "action_flat_dim": int(row["action_flat_dim"]) if row["action_flat_dim"] else None,
            "action_discrete_count": int(row["action_discrete_count"]) if row["action_discrete_count"] else None,
            "action_shape": action_shape,
            "run_start": run_start,
            "run_end": run_end,
        }
        run_entries.append(entry)
        validation_messages.extend(validate_entry(entry))

    task_buckets: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for entry in run_entries:
        task_buckets[(entry["league"], entry["logical_env"], entry["concrete_env_id"])][entry["policy"]].append(entry)

    for policy_map in task_buckets.values():
        task_entries = [entry for entries in policy_map.values() for entry in entries]
        validation_messages.extend(validate_task(task_entries))
        for entries in policy_map.values():
            validation_messages.extend(validate_policy_group(entries))

    if validation_messages:
        for message in sorted(set(validation_messages)):
            print(message, file=sys.stderr)
        return 1

    config_signature = [
        {
            "league": entry["league"],
            "logicalEnv": entry["logical_env"],
            "concreteEnvId": entry["concrete_env_id"],
            "policy": entry["policy"],
            "seed": entry["seed"],
            "numEnvs": entry["num_envs"],
            "warmupSteps": entry["warmup_steps"],
            "stopSteps": entry["stop_steps"],
            "schemaFingerprint": entry["schema_fingerprint"],
        }
        for entry in run_entries
    ]
    config_hash = hashlib.sha256(stable_json(config_signature).encode("utf-8")).hexdigest()
    policies = sorted({entry["policy"] for entry in run_entries})

    tasks: list[dict[str, Any]] = []
    for (league, logical_env, concrete_env_id), policy_map in sorted(task_buckets.items()):
        task_entries = [entry for entries in policy_map.values() for entry in entries]
        policy_summaries = {
            policy: build_policy_summary(entries)
            for policy, entries in sorted(policy_map.items())
        }
        comparisons = {
            f"{left_policy}__vs__{right_policy}": build_comparison(
                f"{left_policy}__vs__{right_policy}",
                policy_map[left_policy],
                policy_map[right_policy],
            )
            for left_policy, right_policy in combinations(sorted(policy_map), 2)
        }

        first_entry = task_entries[0]
        tasks.append(
            {
                "league": league,
                "logicalEnv": logical_env,
                "concreteEnvId": concrete_env_id,
                "schemaFingerprint": first_entry["schema_fingerprint"],
                "observationSpaceKind": first_entry["observation_space_kind"],
                "observationFlatDim": first_entry["observation_flat_dim"],
                "actionSpaceKind": first_entry["action_space_kind"],
                "actionFlatDim": first_entry["action_flat_dim"],
                "actionDiscreteCount": first_entry["action_discrete_count"],
                "actionShape": first_entry["action_shape"],
                "policies": policy_summaries,
                "comparisons": comparisons,
            }
        )

    runs = build_runs(run_entries)
    benchmark = {
        "generatedAt": generated_at,
        "benchDir": str(bench_dir),
        "gitRevision": args.git_revision,
        "configHash": config_hash,
        "baselinePolicy": args.baseline_policy,
        "policies": policies,
        "pythonVersions": python_versions,
        "config": {
            "resolvedTaskCount": len(tasks),
            "runCount": len(runs),
        },
        "runs": runs,
        "tasks": tasks,
    }

    benchmark_json_path.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
