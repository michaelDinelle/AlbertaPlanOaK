#!/usr/bin/env python3
import json
import re
import subprocess
from pathlib import Path

import fleet

AGENTS = ("OaK", "Oracle", "PPO", "Random")
BASELINES = (("random", "Random"), ("oracle", "Oracle"), ("ppo", "PPO"))
COLORS = {"OaK": "#5548b4", "Oracle": "#52a96a", "PPO": "#68a8cc", "Random": "#e88152"}
CATEGORIES = ("control", "navigation", "memory", "decision", "prediction", "world")
README = fleet.REPO / "README.md"
CHART = fleet.REPO / "oak-results.svg"


def load():
    best = fleet.BEST.read_text().strip()
    oak = fleet.Recording(fleet.BENCH / best)
    links = json.loads((oak.root / "baselines.json").read_text())
    loaded = {}

    def recording(name):
        if name not in loaded:
            loaded[name] = fleet.Recording(fleet.BENCH / name)
        return loaded[name]

    catalog = json.loads(subprocess.run([str(fleet.ENGINE), "envs"], capture_output=True, text=True, check=True).stdout)
    category = {entry["envId"]: entry["category"] for entry in catalog}
    rows = []
    for env in oak.envs:
        means = {"OaK": fleet.mean(oak.scores(env))}
        for policy, agent in BASELINES:
            means[agent] = fleet.mean(recording(links[policy][env]).scores(env))
        rows.append({"env": env, "category": category[env], "means": means})
    sps = {"OaK": oak.meta["stepsPerSecond"]}
    for policy, agent in BASELINES:
        sps[agent] = recording(links[policy][oak.envs[0]]).meta["stepsPerSecond"]
    start = max(i for i, end in enumerate(oak.ends) if end <= oak.steps - fleet.TERMINAL)
    return best, oak, oak.ends[-1] - oak.ends[start], rows, sps


def normalized(row, agent):
    means = row["means"]
    return (means[agent] - means["Random"]) / (means["Oracle"] - means["Random"])


def axis_table(rows):
    groups = [(name, [r for r in rows if r["category"] == name]) for name in CATEGORIES] + [("all", rows)]
    return [(name, len(members), {agent: fleet.mean([normalized(r, agent) for r in members]) for agent in AGENTS}) for name, members in groups]


def chart(best, steps, window, rows, sps):
    axes = axis_table(rows)
    lo = min(-0.25, min(v for _, _, means in axes for v in means.values()))
    hi = max(1.0, max(v for _, _, means in axes for v in means.values()))
    lo, hi = lo - 0.05, hi + 0.05
    x0, x1, top, pitch = 250, 1010, 150, 62
    height = top + pitch * len(axes) + 210
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="{height}" viewBox="0 0 1250 {height}" role="img" aria-labelledby="title desc">',
         '<title id="title">OaK and its baselines by environment category</title>',
         f'<desc id="desc">Mean normalized reward per step over the final {window:,} steps of the {steps:,}-step best recording, Random at 0 and Oracle at 1, one row per category and one for all {len(rows)} environments.</desc>',
         f'<rect width="1250" height="{height}" fill="#ffffff"/>', '<g font-family="Arial, Helvetica, sans-serif" fill="#243044">']

    def text(x, y, t, size=17, fill="#243044", anchor="start", weight="400"):
        s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{t}</text>')

    def xat(v):
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    text(54, 55, "Reward by category", 30, weight="600")
    text(54, 86, f"Mean normalized reward per step, final {window:,} of {steps:,} steps, best recording {best[:9]}", 16, fill="#637084")
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = xat(v)
        s.append(f'<path d="M {x:.1f} {top - 28} V {top + pitch * len(axes) - 20}" stroke="{"#9aa3b2" if v in (0.0, 1.0) else "#e2e6ec"}" stroke-width="1"/>')
        text(x, top - 36, f"{v:g}", 13, fill="#647084", anchor="middle")
    text(xat(0.0), top - 54, "Random", 13, fill=COLORS["Random"], anchor="middle", weight="600")
    text(xat(1.0), top - 54, "Oracle", 13, fill=COLORS["Oracle"], anchor="middle", weight="600")
    for i, (name, count, means) in enumerate(axes):
        y = top + pitch * i
        weight = "600" if name == "all" else "400"
        text(54, y + 6, name if name != "all" else f"all", 19, weight=weight)
        text(54, y + 26, f"n = {count}", 13, fill="#687489")
        s.append(f'<path d="M {x0} {y} H {x1}" stroke="#d0d6df" stroke-width="1"/>')
        px, ox = xat(means["PPO"]), xat(means["OaK"])
        s.append(f'<rect x="{px - 6:.1f}" y="{y - 6}" width="12" height="12" fill="{COLORS["PPO"]}"/>')
        s.append(f'<circle cx="{ox:.1f}" cy="{y}" r="8" fill="{COLORS["OaK"]}"/>')
        text(px, y + 27, f"{means['PPO']:.3f}", 12, fill=COLORS["PPO"], anchor="middle")
        text(ox, y - 15, f"{means['OaK']:.3f}", 12, fill=COLORS["OaK"], anchor="middle", weight="600")
    legend_y = top + pitch * len(axes) + 20
    s.append(f'<circle cx="{x0 + 8}" cy="{legend_y}" r="8" fill="{COLORS["OaK"]}"/>')
    text(x0 + 24, legend_y + 6, "OaK", 15)
    s.append(f'<rect x="{x0 + 94}" y="{legend_y - 6}" width="12" height="12" fill="{COLORS["PPO"]}"/>')
    text(x0 + 116, legend_y + 6, "PPO", 15)
    overall = axes[-1][2]
    ranking = sorted(AGENTS, key=overall.get, reverse=True)
    text(x0, legend_y + 50, "Overall, equal weight over every environment", 17, weight="600")
    for j, agent in enumerate(ranking):
        y = legend_y + 80 + j * 26
        text(x0, y, f"{j + 1}. {agent}", 16, fill=COLORS[agent], weight="600")
        text(x0 + 160, y, f"{overall[agent]:.3f}", 16, anchor="end")
        text(x0 + 200, y, f"{sps[agent]:,.0f} steps/s", 14, fill="#637084")
    s.append("</g></svg>")
    CHART.write_text("\n".join(s) + "\n")


def readme(best, oak, window, rows, sps):
    text = README.read_text()
    text = re.sub(r"[\d,]+ SPS on the", f"{sps['OaK']:,.0f} SPS on the", text, count=1)
    lines = text.split("\n")
    first = next(i for i, line in enumerate(lines) if line.startswith("Reward per step over the final"))
    last = next(i for i, line in enumerate(lines) if line.startswith("Baseline throughput on the same schedule"))
    block = [f"Reward per step over the final {window:,} steps of the {oak.steps:,}-step best recording (`{best}`), all {len(rows)} environments with their declared seed counts ({oak.lanes:,} lanes), with the Oracle, Random and PPO baselines on the same seeds. Mean over seeds; Score = (OaK − Random) / (Oracle − Random), unclipped.", "",
             "| Environment | Category | OaK | Oracle | Random | PPO | Score |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["means"]
        block.append(f"| {row['env']} | {row['category']} | {fleet.fmt(m['OaK'])} | {fleet.fmt(m['Oracle'])} | {fleet.fmt(m['Random'])} | {fleet.fmt(m['PPO'])} | {fleet.fmt(normalized(row, 'OaK'))} |")
    block += ["", f"Baseline throughput on the same schedule: PPO {sps['PPO']:,.0f} steps/s, Random {sps['Random']:,.0f} steps/s, Oracle {sps['Oracle']:,.0f} steps/s."]
    README.write_text("\n".join(lines[:first] + block + lines[last + 1:]))


if __name__ == "__main__":
    best, oak, window, rows, sps = load()
    chart(best, oak.steps, window, rows, sps)
    readme(best, oak, window, rows, sps)
    for name, count, means in axis_table(rows):
        print(f"{name:11s} n={count:3d}  " + "  ".join(f"{agent} {means[agent]:+.3f}" for agent in AGENTS))
