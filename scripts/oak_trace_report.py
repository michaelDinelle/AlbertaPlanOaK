#!/usr/bin/env python3
"""
Summarize an Instruments .trace from `xctrace export` (Metal System Trace–class).

Requires: Xcode command-line tools (`xctrace` on PATH).

**Primary GPU phase times for automation:** use `OaK diagnose` JSON field
`diagnostic.metalPhaseGpuMsMean` (host-measured per phase). This script only
parses Apple's trace export XML — useful for template sanity, row counts, and
cross-checking `xctrace` output; it does not run OaK or read diagnose JSON.

Usage:
  python3 scripts/oak_trace_report.py /path/to/capture.trace
  python3 scripts/oak_trace_report.py /path/to/capture.trace --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any

SCHEMAS = (
    "metal-application-intervals",
    "metal-driver-intervals",
    "metal-shader-profiler-shader-list",
    "metal-shader-profiler-intervals",
    "os-signpost",
)


def _xctrace_export(trace: str, xpath: str) -> str:
    return subprocess.check_output(
        ["xctrace", "export", "--input", trace, "--xpath", xpath],
        text=True,
        stderr=subprocess.DEVNULL,
    )


def _xctrace_toc(trace: str) -> str:
    return subprocess.check_output(
        ["xctrace", "export", "--input", trace, "--toc"],
        text=True,
        stderr=subprocess.DEVNULL,
    )


def _parse_rows(xml: str) -> list[ET.Element]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    return root.findall(".//row")


def _row_duration_ns(row: ET.Element) -> float:
    for el in row.iter():
        if el.tag == "duration" and el.text and str(el.text).strip().isdigit():
            return float(el.text)
    return 0.0


def _is_oak(blob: str, pid: str | None) -> bool:
    if not pid:
        return False
    return (
        f'pid="{pid}"' in blob
        or f"OaK ({pid})" in blob
        or f", pid: {pid})" in blob
    )


def _shader_timeline_enabled(trace: str) -> bool | None:
    path = os.path.join(trace, "form.template")
    if not os.path.isfile(path):
        return None
    try:
        data = open(path, "rb").read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    if "Shader Timeline: Enabled" in data:
        return True
    if "Shader Timeline: Disabled" in data:
        return False
    return None


def analyze(trace: str) -> dict[str, Any]:
    toc = _xctrace_toc(trace)
    m = re.search(r'name="OaK"[^>]*pid="(\d+)"', toc)
    pid = m.group(1) if m else None
    wall_m = re.search(r"<duration>([\d.]+)</duration>", toc)
    wall_s = float(wall_m.group(1)) if wall_m else None
    tmpl_m = re.search(r"<template-name>([^<]+)</template-name>", toc)

    out: dict[str, Any] = {
        "trace": os.path.abspath(trace),
        "template_name": tmpl_m.group(1) if tmpl_m else None,
        "oak_pid": pid,
        "wall_seconds": wall_s,
        "shader_timeline_enabled_in_form_template": _shader_timeline_enabled(trace),
        "tables": {},
    }

    for schema in SCHEMAS:
        try:
            xml = _xctrace_export(
                trace, f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
            )
        except subprocess.CalledProcessError:
            out["tables"][schema] = {"error": "export_failed", "row_count": 0}
            continue
        rows = _parse_rows(xml)
        info: dict[str, Any] = {"row_count": len(rows)}

        if schema == "metal-application-intervals" and pid:
            oak_ns = 0.0
            oak_rows = 0
            by_evt: dict[str, float] = defaultdict(float)
            for row in rows:
                blob = ET.tostring(row, encoding="unicode")
                if not _is_oak(blob, pid):
                    continue
                oak_rows += 1
                ns = _row_duration_ns(row)
                oak_ns += ns
                ml = re.findall(r'<metal-object-label[^>]*fmt="([^"]*)"', blob)
                evt = ml[0] if ml else "unknown"
                by_evt[evt] += ns
            info["oak_row_count"] = oak_rows
            info["oak_duration_sum_ms"] = oak_ns / 1e6
            tot = oak_ns if oak_ns else 1.0
            info["oak_top_event_types_by_duration_ms"] = [
                {"label": k[:120], "ms": v / 1e6, "pct_of_oak_sum": round(100 * v / tot, 2)}
                for k, v in sorted(by_evt.items(), key=lambda x: -x[1])[:15]
            ]

        if schema == "metal-driver-intervals" and pid:
            oak_ns = 0.0
            oak_rows = 0
            by_drv: dict[str, float] = defaultdict(float)
            for row in rows:
                blob = ET.tostring(row, encoding="unicode")
                if not _is_oak(blob, pid):
                    continue
                oak_rows += 1
                ns = _row_duration_ns(row)
                oak_ns += ns
                g = re.findall(r'<gpu-driver-name[^>]*fmt="([^"]*)"', blob)
                key = g[0] if g else "unknown"
                by_drv[key] += ns
            info["oak_row_count"] = oak_rows
            info["oak_duration_sum_ms"] = oak_ns / 1e6
            info["oak_by_driver_name_ms"] = {
                k[:80]: round(v / 1e6, 6) for k, v in sorted(by_drv.items(), key=lambda x: -x[1])[:12]
            }

        if schema == "os-signpost" and pid:
            oak_rows = 0
            phases: dict[str, float] = defaultdict(float)
            sp_ms = 0.0
            for row in rows:
                blob = ET.tostring(row, encoding="unicode")
                if not _is_oak(blob, pid):
                    continue
                oak_rows += 1
                ns = _row_duration_ns(row)
                sp_ms += ns
                matched = False
                for p in (
                    "encode",
                    "prediction",
                    "learn",
                    "planning",
                    "forwardModel",
                    "metaPolicy",
                    "bookkeeping",
                ):
                    if p in blob:
                        phases[p] += ns
                        matched = True
                        break
                if not matched:
                    phases["_other"] += ns
            info["oak_row_count"] = oak_rows
            info["oak_duration_sum_ms"] = sp_ms / 1e6
            info["oak_phases_ms"] = {k: round(v / 1e6, 6) for k, v in phases.items()}

        out["tables"][schema] = info

    out["notes"] = [
        "oak_duration_sum_ms in metal-application-intervals sums short labeled API intervals; not total GPU time per step.",
        "metal-shader-profiler-intervals is often empty in xctrace export even when Shader Timeline is on.",
        "Agent vs env and per-phase GPU ms: use `OaK diagnose` JSON (agentShare, envShare, metalPhaseGpuMsMean). This tool only reads .trace exports.",
    ]
    return out


def _print_human(report: dict[str, Any]) -> None:
    print("OaK trace report:", report["trace"])
    print(f"  Template: {report.get('template_name')}")
    print(f"  OaK pid: {report.get('oak_pid')}  Wall (trace): {report.get('wall_seconds')} s")
    st = report.get("shader_timeline_enabled_in_form_template")
    print(f"  Shader Timeline (form.template): {st}")
    print()
    for schema, info in report["tables"].items():
        print(f"  [{schema}]")
        if "error" in info:
            print(f"    error: {info['error']}")
            continue
        print(f"    rows: {info.get('row_count', 0)}")
        for k in ("oak_row_count", "oak_duration_sum_ms"):
            if k in info:
                print(f"    {k}: {info[k]}")
        if schema == "metal-application-intervals" and "oak_top_event_types_by_duration_ms" in info:
            print("    top event types (OaK):")
            for row in info["oak_top_event_types_by_duration_ms"][:8]:
                print(f"      {row['label'][:60]:60} {row['ms']:.4f} ms  ({row['pct_of_oak_sum']}%)")
        if schema == "metal-driver-intervals" and info.get("oak_by_driver_name_ms"):
            print("    by driver name (ms):", info["oak_by_driver_name_ms"])
        if schema == "os-signpost" and info.get("oak_phases_ms"):
            print("    signpost phases (ms):", info["oak_phases_ms"])
        print()
    print("Notes:")
    for n in report.get("notes", []):
        print(f"  - {n}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize OaK Metal .trace via xctrace export.")
    ap.add_argument("trace", help="Path to .trace bundle")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = ap.parse_args()

    trace = args.trace
    if not os.path.isdir(trace):
        print(f"Not a trace bundle: {trace}", file=sys.stderr)
        return 1

    try:
        report = analyze(trace)
    except FileNotFoundError:
        print("xctrace not found; install Xcode command line tools.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"xctrace failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
