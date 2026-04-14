#!/usr/bin/env python3
"""Generate benchmark.json (schema oak.benchmark).

Root defines column orders once: policyStatOrder, statColumns, comparisonDeltaOrder, deltaStatColumns,
artifactColumns. Policy stats are a dense array parallel to policyStatOrder; dt/dsw/dts/dtmn/dtmx/dtfc are
dense arrays parallel to dynamicTelemetrySchema rows (dt/dsw: temporal mean summaries; dts/dtmn/dtmx:
within-run temporal std/min/max per seed; dtfc: non-finite sample counts per channel). Comparisons use a deltas matrix
parallel to comparisonDeltaOrder plus seeds, record, and sig [pValue, significant]."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

BENCHMARK_SCHEMA = "oak.benchmark"
STAT_COLUMNS = ["n", "mean", "std", "median", "min", "max"]
DELTA_STAT_COLUMNS = ["n", "mean", "std", "median", "min", "max", "ci95Low", "ci95High"]
ARTIFACT_COLUMNS = ["seed", "metricsPath", "logPath", "runId", "stopReason"]

# Order matches build_policy_summary; each value is a summarize() dict before compaction.
POLICY_STAT_KEYS = (
    "rewardSum",
    "rewardMean",
    "rewardStd",
    "doneCount",
    "resetCount",
    "measuredStepCount",
    "vectorUpdateCount",
    "measuredSeconds",
    "stepsPerSecond",
    "completionRate",
    "plannerStepMsMean",
    "plannerActiveFraction",
    "activeRankMean",
    "activeRankMin",
    "activeRankMax",
    "activeGVFsMean",
    "activeOptionsMean",
    "actionL2Mean",
    "stateL2Mean",
    "agentStepMsMean",
    "envStepMsMean",
    "totalStepMsMean",
)

COMPARISON_DELTA_ORDER = (
    "rewardSumDelta",
    "rewardMeanDelta",
    "stepsPerSecondDelta",
    "completionRateDelta",
)


def parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="Generate oak.benchmark benchmark.json (dense ordered arrays, compact telemetry). "
        "Run with cwd set to the bench directory (writes benchmark.json; discovers runs/*.jsonl)."
    )
    parser.parse_args()


def resolve_git_revision(bench_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=bench_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def infer_baseline_policy(entries: list[dict[str, Any]]) -> str:
    order: list[str] = []
    seen: set[str] = set()
    for row in entries:
        p = str(row.get("policy", ""))
        if p and p not in seen:
            seen.add(p)
            order.append(p)
    if "random" in seen:
        return "random"
    return order[0] if order else "oak"


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
    n = len(values)
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if n > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def compact_stat(summary: dict[str, Any] | None) -> list[Any] | None:
    if summary is None:
        return None
    return [summary["n"], summary["mean"], summary["std"], summary["median"], summary["min"], summary["max"]]


def compact_dt_schema(schema: list[dict[str, Any]]) -> list[list[str]]:
    return [[str(d.get("key", "")), str(d.get("unit", ""))] for d in schema]


def compact_delta(delta: dict[str, Any]) -> list[Any]:
    ci = delta.get("bootstrapCI95") or {"low": 0.0, "high": 0.0}
    return [
        delta["n"],
        delta["mean"],
        delta["std"],
        delta["median"],
        delta["min"],
        delta["max"],
        float(ci["low"]),
        float(ci["high"]),
    ]


def artifact_matrix(run_artifacts: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in run_artifacts:
        rows.append(
            [
                row["seed"],
                row["metricsPath"],
                row["logPath"],
                row["runId"],
                row.get("stopReason"),
            ]
        )
    return rows


def policy_stats_row(policy: dict[str, Any]) -> list[Any]:
    return [compact_stat(policy.get(key)) for key in POLICY_STAT_KEYS]


def dense_dynamic_rows(schema: list[dict[str, Any]], summaries: dict[str, Any]) -> list[Any]:
    rows: list[Any] = []
    for index, descriptor in enumerate(schema):
        key = str(descriptor.get("key") or f"metric_{index}")
        rows.append(compact_stat(summaries.get(key)))
    return rows


def to_compact_policy(policy: dict[str, Any]) -> dict[str, Any]:
    schema = policy.get("dynamicTelemetrySchema") or []
    return {
        "runCount": policy["runCount"],
        "seeds": policy["seeds"],
        "backend": policy["backend"],
        "numEnvs": policy["numEnvs"],
        "warmupSteps": policy["warmupSteps"],
        "stopSteps": policy["stopSteps"],
        "stats": policy_stats_row(policy),
        "dynamicTelemetrySchema": compact_dt_schema(schema),
        "dt": dense_dynamic_rows(schema, policy.get("dynamicTelemetry") or {}),
        "dsw": dense_dynamic_rows(schema, policy.get("dynamicTelemetrySummaryWindows") or {}),
        "dts": dense_dynamic_rows(schema, policy.get("dynamicTelemetryTemporalStd") or {}),
        "dtmn": dense_dynamic_rows(schema, policy.get("dynamicTelemetryTemporalMin") or {}),
        "dtmx": dense_dynamic_rows(schema, policy.get("dynamicTelemetryTemporalMax") or {}),
        "dtfc": dense_dynamic_rows(schema, policy.get("dynamicTelemetryNonFinite") or {}),
        "stopReasonCounts": policy.get("stopReasonCounts") or {},
        "artifacts": artifact_matrix(policy.get("runArtifacts") or []),
    }


def to_compact_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    sig = comparison["significance"]
    deltas = [compact_delta(comparison[name]) for name in COMPARISON_DELTA_ORDER]
    return {
        "seeds": comparison["pairedSeeds"],
        "deltas": deltas,
        "wins": comparison["wins"],
        "losses": comparison["losses"],
        "ties": comparison["ties"],
        "sig": [sig["pValue"], sig["significant"]],
    }


def validate_benchmark_shapes(benchmark: dict[str, Any]) -> None:
    """Fail fast if layout invariants are broken (alignment of parallel arrays)."""
    porder = benchmark["policyStatOrder"]
    delta_order = benchmark["comparisonDeltaOrder"]
    stat_len = len(benchmark["statColumns"])
    delta_len = len(benchmark["deltaStatColumns"])
    if len(porder) != len(POLICY_STAT_KEYS) or porder != list(POLICY_STAT_KEYS):
        raise ValueError("policyStatOrder must match POLICY_STAT_KEYS")
    if len(delta_order) != len(COMPARISON_DELTA_ORDER) or delta_order != list(COMPARISON_DELTA_ORDER):
        raise ValueError("comparisonDeltaOrder must match COMPARISON_DELTA_ORDER")
    for task in benchmark["tasks"]:
        for policy in task["policies"].values():
            stats_row = policy["stats"]
            if len(stats_row) != len(porder):
                raise ValueError("policy stats row length must match policyStatOrder")
            for cell in stats_row:
                if cell is None:
                    continue
                if not isinstance(cell, list) or len(cell) != stat_len:
                    raise ValueError("each non-null policy stat must be statColumns-sized")
            schema = policy["dynamicTelemetrySchema"]
            dt_group = ("dt", "dsw", "dts", "dtmn", "dtmx", "dtfc")
            for name in dt_group:
                if len(policy[name]) != len(schema):
                    raise ValueError(f"{name} must match dynamicTelemetrySchema length")
            for vec in (*policy["dt"], *policy["dsw"], *policy["dts"], *policy["dtmn"], *policy["dtmx"], *policy["dtfc"]):
                if vec is None:
                    continue
                if not isinstance(vec, list) or len(vec) != stat_len:
                    raise ValueError("each non-null dt/dsw/dts/dtmn/dtmx/dtfc entry must be statColumns-sized")
        for comp in task["comparisons"].values():
            if len(comp["deltas"]) != len(delta_order):
                raise ValueError("comparison deltas must match comparisonDeltaOrder length")
            for row in comp["deltas"]:
                if not isinstance(row, list) or len(row) != delta_len:
                    raise ValueError("each comparison delta row must be deltaStatColumns-sized")
            sig = comp["sig"]
            if not isinstance(sig, list) or len(sig) != 2:
                raise ValueError("sig must be [pValue, significant]")


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


def discover_metrics_jsonl(runs_dir: Path) -> list[Path]:
    return sorted(runs_dir.glob("*.jsonl"))


def _opt_int_field(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def entry_from_metrics_path(metrics_path: Path) -> dict[str, Any]:
    """Build report entry fields plus run_start/run_end from a single metrics JSONL."""
    run_start, run_end = read_records(metrics_path)
    rs = run_start or {}
    action_shape = rs.get("actionShape")
    if not isinstance(action_shape, list):
        action_shape = []
    warmup_steps = int(rs.get("warmupSteps") or 0)
    stop_steps_val = _opt_int_field(rs.get("stopAfterMeasuredSteps"))
    raw_num = rs.get("numEnvs")
    num_envs = int(raw_num) if raw_num not in (None, "") else 1

    log_candidate = metrics_path.with_suffix(".log")
    log_path = str(log_candidate.resolve()) if log_candidate.is_file() else ""

    ofd = rs.get("observationFlatDim")
    afd = rs.get("actionFlatDim")
    adc = rs.get("actionDiscreteCount")

    return {
        "league": str(rs.get("league") or ""),
        "logical_env": str(rs.get("logicalEnv") or ""),
        "concrete_env_id": str(rs.get("concreteEnvId") or rs.get("env") or ""),
        "backend": str(rs.get("backend") or ""),
        "policy": str(rs.get("policy") or ""),
        "seed": int(rs["seed"]) if rs.get("seed") is not None else 0,
        "num_envs": num_envs,
        "warmup_steps": warmup_steps,
        "stop_steps": stop_steps_val,
        "run_id": str(rs.get("runId") or ""),
        "metrics_path": str(metrics_path.resolve()),
        "log_path": log_path,
        "schema_fingerprint": str(rs.get("schemaFingerprint") or ""),
        "observation_space_kind": str(rs.get("observationSpaceKind") or ""),
        "observation_flat_dim": int(ofd) if ofd not in (None, "") else None,
        "action_space_kind": str(rs.get("actionSpaceKind") or ""),
        "action_flat_dim": int(afd) if afd not in (None, "") else None,
        "action_discrete_count": int(adc) if adc not in (None, "") else None,
        "action_shape": action_shape,
        "run_start": run_start,
        "run_end": run_end,
    }


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


def extract_dynamic_values(record: dict[str, Any] | None) -> list[float | None]:
    """Parallel to schema; `None` preserves JSON null (no finite value — not a substitute zero)."""
    if record is None:
        return []
    values = record.get("dynamicTelemetryValues")
    if values is None:
        return []
    if not isinstance(values, list):
        return []
    output: list[float | None] = []
    for value in values:
        if value is None:
            output.append(None)
        elif is_finite_number(value):
            output.append(float(value))
        else:
            output.append(None)
    return output


def extract_dynamic_non_finite_counts(record: dict[str, Any] | None) -> list[int]:
    if record is None:
        return []
    values = record.get("dynamicTelemetryNonFiniteCounts")
    if values is None:
        return []
    if not isinstance(values, list):
        return []
    out: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            out.append(value)
        elif isinstance(value, float) and value.is_integer():
            out.append(int(value))
    return out


def extract_dynamic_field(record: dict[str, Any] | None, field: str) -> list[float | None]:
    """Same layout as dynamicTelemetryValues for optional parallel arrays (std/min/max)."""
    if record is None:
        return []
    values = record.get(field)
    if not isinstance(values, list):
        return []
    output: list[float | None] = []
    for value in values:
        if value is None:
            output.append(None)
        elif is_finite_number(value):
            output.append(float(value))
        else:
            output.append(None)
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


def summarize_dynamic_telemetry(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not entries:
        return [], {}, {}, {}, {}, {}

    schema = extract_dynamic_schema(entries[0].get("run_start"))
    if not schema:
        return [], {}, {}, {}, {}, {}

    summaries_mean: dict[str, Any] = {}
    summaries_std: dict[str, Any] = {}
    summaries_tmin: dict[str, Any] = {}
    summaries_tmax: dict[str, Any] = {}
    summaries_nonfinite: dict[str, Any] = {}
    for index, descriptor in enumerate(schema):
        key = str(descriptor.get("key") or f"metric_{index}")
        values_mean: list[float] = []
        values_std: list[float] = []
        values_tmin: list[float] = []
        values_tmax: list[float] = []
        values_nf: list[float] = []
        for entry in entries:
            run_end = entry.get("run_end")
            run_end_values = extract_dynamic_values(run_end)
            if index < len(run_end_values) and is_finite_number(run_end_values[index]):
                values_mean.append(float(run_end_values[index]))
            std_list = extract_dynamic_field(run_end, "dynamicTelemetryStd")
            if index < len(std_list) and is_finite_number(std_list[index]):
                values_std.append(float(std_list[index]))
            mn_list = extract_dynamic_field(run_end, "dynamicTelemetryMin")
            if index < len(mn_list) and is_finite_number(mn_list[index]):
                values_tmin.append(float(mn_list[index]))
            mx_list = extract_dynamic_field(run_end, "dynamicTelemetryMax")
            if index < len(mx_list) and is_finite_number(mx_list[index]):
                values_tmax.append(float(mx_list[index]))
            nf_list = extract_dynamic_non_finite_counts(run_end)
            if index < len(nf_list):
                values_nf.append(float(nf_list[index]))
        if values_mean:
            summaries_mean[key] = summarize(values_mean)
        if values_std:
            summaries_std[key] = summarize(values_std)
        if values_tmin:
            summaries_tmin[key] = summarize(values_tmin)
        if values_tmax:
            summaries_tmax[key] = summarize(values_tmax)
        if values_nf:
            summaries_nonfinite[key] = summarize(values_nf)
    return schema, summaries_mean, summaries_std, summaries_tmin, summaries_tmax, summaries_nonfinite


def summarize_dynamic_telemetry_from_summary_windows(
    entries: list[dict[str, Any]],
    schema: list[dict[str, Any]],
) -> dict[str, Any]:
    """Exact n/mean/sample-std/median/min/max over every rolling `summary` line (finite values only).

    One pass over JSONL per metrics file; memory O(len(schema) * N) for N summary rows.
    Uses the same ``summarize`` helper as seed-aggregated ``run_end`` telemetry.
    """
    if not schema or not entries:
        return {}

    nkey = len(schema)
    buckets: list[list[float]] = [[] for _ in range(nkey)]
    loads = json.loads

    for entry in entries:
        metrics_path = Path(str(entry.get("metrics_path") or ""))
        if not metrics_path.is_file():
            continue
        try:
            with metrics_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("type") != "summary":
                        continue
                    vals = extract_dynamic_values(record)
                    nvals = len(vals)
                    if nvals < nkey:
                        upper = nvals
                    else:
                        upper = nkey
                    for i in range(upper):
                        v = vals[i]
                        if not is_finite_number(v):
                            continue
                        buckets[i].append(v)
        except OSError:
            continue

    out: dict[str, Any] = {}
    for index, descriptor in enumerate(schema):
        key = str(descriptor.get("key") or f"metric_{index}")
        stats = summarize(buckets[index])
        if stats is not None:
            out[key] = stats
    return out


def build_policy_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    first = entries[0] if entries else {}
    (
        dynamic_schema,
        dynamic_telemetry,
        dynamic_telemetry_std,
        dynamic_telemetry_tmin,
        dynamic_telemetry_tmax,
        dynamic_telemetry_nonfinite,
    ) = summarize_dynamic_telemetry(entries)
    summary_window_telemetry: dict[str, Any] = {}
    if dynamic_schema and str(first.get("policy") or "") == "oak":
        summary_window_telemetry = summarize_dynamic_telemetry_from_summary_windows(entries, dynamic_schema)
    return {
        "runCount": len(entries),
        "seeds": sorted(entry["seed"] for entry in entries),
        "backend": first.get("backend", ""),
        "numEnvs": first.get("num_envs"),
        "warmupSteps": first.get("warmup_steps"),
        "stopSteps": first.get("stop_steps"),
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
        "dynamicTelemetryTemporalStd": dynamic_telemetry_std,
        "dynamicTelemetryTemporalMin": dynamic_telemetry_tmin,
        "dynamicTelemetryTemporalMax": dynamic_telemetry_tmax,
        "dynamicTelemetryNonFinite": dynamic_telemetry_nonfinite,
        "dynamicTelemetrySummaryWindows": summary_window_telemetry,
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
    metrics_path = entry["metrics_path"]

    if run_start is None:
        messages.append(f"{descriptor}: missing run_start in {metrics_path}")
    if run_end is None:
        messages.append(f"{descriptor}: missing run_end in {metrics_path}")

    if run_start is not None:
        env_field = str(run_start.get("env") or "")
        conc_field = str(run_start.get("concreteEnvId") or "")
        if env_field and conc_field and env_field != conc_field:
            messages.append(
                f"{descriptor}: task resolution mismatch in run_start; "
                f"recorded {env_field!r}, expected {conc_field!r}"
            )
        if str(run_start.get("schemaFingerprint") or "") == "":
            messages.append(f"{descriptor}: missing schema fingerprint in run_start")

    if run_start is not None and run_end is not None:
        start_concrete = str(run_start.get("concreteEnvId") or run_start.get("env") or "")
        start_logical = str(run_start.get("logicalEnv") or "")
        end_concrete = str(run_end.get("concreteEnvId") or run_end.get("env") or "")
        end_logical = str(run_end.get("logicalEnv") or "")
        if start_concrete and end_concrete and start_concrete != end_concrete:
            messages.append(
                f"{descriptor}: task resolution mismatch in run_end; "
                f"recorded {end_concrete!r}, expected {start_concrete!r}"
            )
        if start_logical and end_logical and start_logical != end_logical:
            messages.append(
                f"{descriptor}: logical env mismatch in run_end; "
                f"recorded {end_logical!r}, expected {start_logical!r}"
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
    parse_args()
    bench_dir = Path.cwd().resolve()
    runs_dir = bench_dir / "runs"
    benchmark_json_path = bench_dir / "benchmark.json"
    if benchmark_json_path.exists():
        benchmark_json_path.unlink()

    jsonl_paths = discover_metrics_jsonl(runs_dir)
    if not jsonl_paths:
        print(f"No metrics JSONL files in {runs_dir}", file=sys.stderr)
        return 1

    run_entries: list[dict[str, Any]] = [entry_from_metrics_path(p) for p in jsonl_paths]
    baseline_env = (os.environ.get("OAK_BENCH_BASELINE_POLICY") or "").strip()
    baseline_policy = baseline_env if baseline_env else infer_baseline_policy(run_entries)
    git_revision = resolve_git_revision(bench_dir)
    python_versions = load_python_package_versions(sys.executable)
    generated_at = datetime.now(timezone.utc).isoformat()

    validation_messages: list[str] = []
    for entry in run_entries:
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

    tasks: list[dict[str, Any]] = []
    for (league, logical_env, concrete_env_id), policy_map in sorted(task_buckets.items()):
        task_entries = [entry for entries in policy_map.values() for entry in entries]
        policy_summaries = {
            policy: to_compact_policy(build_policy_summary(entries))
            for policy, entries in sorted(policy_map.items())
        }
        comparisons = {
            f"{left_policy}__vs__{right_policy}": to_compact_comparison(
                build_comparison(
                    f"{left_policy}__vs__{right_policy}",
                    policy_map[left_policy],
                    policy_map[right_policy],
                )
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

    benchmark = {
        "schema": BENCHMARK_SCHEMA,
        "policyStatOrder": list(POLICY_STAT_KEYS),
        "comparisonDeltaOrder": list(COMPARISON_DELTA_ORDER),
        "statColumns": STAT_COLUMNS,
        "deltaStatColumns": DELTA_STAT_COLUMNS,
        "artifactColumns": ARTIFACT_COLUMNS,
        "significanceTest": "exact_sign_test",
        "generatedAt": generated_at,
        "benchDir": str(bench_dir),
        "gitRevision": git_revision,
        "configHash": config_hash,
        "baselinePolicy": baseline_policy,
        "pythonVersions": python_versions,
        "tasks": tasks,
    }

    validate_benchmark_shapes(benchmark)

    benchmark_json_path.write_text(stable_json(benchmark) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
