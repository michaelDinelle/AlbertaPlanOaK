#!/usr/bin/env python3
import json
import math
import re
import subprocess
from pathlib import Path

import fleet

AGENTS = ("OaK", "Oracle", "Stream", "PPO", "Constant", "Random")
BASELINES = (("random", "Random"), ("constant", "Constant"), ("oracle", "Oracle"), ("ppo", "PPO"), ("stream", "Stream"))
COLORS = {"OaK": "#5548b4", "Oracle": "#52a96a", "Stream": "#b0508e", "PPO": "#68a8cc", "Constant": "#9a8f6a", "Random": "#e88152"}
CATEGORIES = ("control", "navigation", "memory", "decision", "prediction", "world", "games", "puzzles")
README = fleet.REPO / "README.md"
CHART = fleet.REPO / "oak-performance-radar.svg"


def load():
    best = fleet.BEST.read_text().strip()
    oak = fleet.Recording(fleet.BENCH / best)
    base = fleet.Baselines(oak)
    catalog = json.loads(subprocess.run([str(oak.root / fleet.RECORDED_ENGINE), "envs"], capture_output=True, text=True, check=True).stdout)
    category = {entry["envId"]: entry["category"] for entry in catalog}
    rows = []
    for env in oak.envs:
        means = {"OaK": fleet.mean(oak.scores(env))}
        for policy, agent in BASELINES:
            means[agent] = base.mean(policy, env)
        rows.append({"env": env, "category": category[env], "means": means, "scores": {agent: base.norm(env, means[agent]) for agent in AGENTS}})
    sps = {"OaK": oak.meta["stepsPerSecond"]}
    for policy, agent in BASELINES:
        lanes = {}
        for env in oak.envs:
            lanes[base.links[policy][env]] = lanes.get(base.links[policy][env], 0) + len(oak.seeds[env])
        sps[agent] = sum(lanes.values()) / sum(count / base.loaded[name].meta["stepsPerSecond"] for name, count in lanes.items())
    start = max(i for i, end in enumerate(oak.ends) if end <= oak.steps - fleet.TERMINAL)
    return best, oak, oak.ends[-1] - oak.ends[start], rows, sps


def normalized(row, agent):
    return row["scores"][agent]


def axis_table(rows):
    assert all(r["category"] in CATEGORIES for r in rows), sorted({r["category"] for r in rows} - set(CATEGORIES))
    groups = [(name, members) for name in CATEGORIES for members in [[r for r in rows if r["category"] == name]] if members] + [("all", rows)]
    return [(name, len(members), {agent: fleet.average([normalized(r, agent) for r in members]) for agent in AGENTS}) for name, members in groups]


def chart(best, steps, window, rows, sps):
    axes = axis_table(rows)[:-1]
    ahead = {name: sum(1 for r in rows if r["category"] == name and normalized(r, "OaK") > normalized(r, "PPO")) for name, _, _ in axes}
    shown = [means[agent] for _, _, means in axes for agent in ("OaK", "PPO", "Stream")]
    lo, hi = min([-0.25] + [v - 0.05 for v in shown]), max([1.0] + [v + 0.05 for v in shown])
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="930" viewBox="0 0 1250 930" role="img" aria-labelledby="title desc">',
         f'<title id="title">OaK and PPO across the {len(axes)} environment categories</title>',
         f'<desc id="desc">Mean normalized reward per step over the final {window:,} steps of the {steps:,}-step best recording, one axis per declared category with its environment count, the better of Random and the constant do-nothing policy at 0 and Oracle at 1. Outlines only: the polygon area is not the score. The overall ranking is the equal-weight mean over all {len(rows)} environments.</desc>',
         '<rect width="1250" height="930" fill="#ffffff"/>', '<g font-family="Arial, Helvetica, sans-serif" fill="#243044">']

    def text(x, y, t, size=18, fill="#243044", anchor="start", weight="400"):
        s.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{t}</text>')

    cx, cy, R = 430, 390, 240
    angles = [-math.pi / 2 + 2 * math.pi * i / len(axes) for i in range(len(axes))]

    def radius(value):
        return R * (value - lo) / (hi - lo)

    def xy(i, value):
        return cx + radius(value) * math.cos(angles[i]), cy + radius(value) * math.sin(angles[i])

    def points(agent):
        return " ".join(f"{x:.2f},{y:.2f}" for i, (_, _, means) in enumerate(axes) for x, y in [xy(i, means[agent])])

    text(54, 55, "Policy comparison", 30, weight="600")
    text(54, 86, f"Mean normalized reward per step \u00b7 {len(rows)} environments \u00b7 best recording {best[:9]}", 17, fill="#637084")
    for value in (0.25, 0.5, 0.75):
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{radius(value):.1f}" fill="none" stroke="#cbd2dc" stroke-width="1" stroke-dasharray="2 4"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{radius(0.0):.1f}" fill="none" stroke="{COLORS["Random"]}" stroke-width="2.5" stroke-dasharray="3 5"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{radius(1.0):.1f}" fill="none" stroke="{COLORS["Oracle"]}" stroke-width="2.5" stroke-dasharray="9 6"/>')
    for i in range(len(axes)):
        x, y = xy(i, hi)
        s.append(f'<path d="M {cx} {cy} L {x:.2f} {y:.2f}" stroke="#d0d6df" stroke-width="1"/>')
    for agent in ("PPO", "Stream", "OaK"):
        s.append(f'<polygon points="{points(agent)}" fill="none" stroke="{COLORS[agent]}" stroke-width="3.5" stroke-linejoin="round"/>')
        for i, (_, _, means) in enumerate(axes):
            x, y = xy(i, means[agent])
            if agent != "OaK":
                s.append(f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" fill="{COLORS[agent]}"/>')
            else:
                s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{COLORS[agent]}"/>')
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        text(cx + radius(value) - 2, cy + 21, f"{value:g}", 13, fill="#647084", anchor="middle")
    text(cx - 8, cy + 22, f"{lo:.2f}".replace("-", "\u2212"), 12, fill="#647084", anchor="end")
    text(cx + radius(0.0) + 6, cy - 8, "Floor", 12, fill=COLORS["Random"], weight="600")
    text(cx + radius(1.0) + 6, cy - 8, "Oracle", 12, fill=COLORS["Oracle"], weight="600")
    for i, (name, count, means) in enumerate(axes):
        x, y = cx + (R + 40) * math.cos(angles[i]), cy + (R + 40) * math.sin(angles[i])
        anchor = "start" if math.cos(angles[i]) > 0.35 else "end" if math.cos(angles[i]) < -0.35 else "middle"
        text(x, y, name, 20, anchor=anchor)
        text(x, y + 21, f"n = {count}", 14, fill="#687489", anchor=anchor)
    overall = axis_table(rows)[-1][2]
    ranking = sorted(AGENTS, key=overall.get, reverse=True)
    scale = 280 / max([1.0] + [v for v in overall.values() if math.isfinite(v)])
    text(900, 222, "Overall ranking", 24, weight="600")
    text(900, 248, f"equal weight over all {len(rows)} environments", 15, fill="#637084")
    for j, agent in enumerate(ranking):
        y = 290 + j * 56
        text(900, y, f"{j + 1}. {agent}", 20, fill=COLORS[agent], weight="600")
        text(1180, y, f"{overall[agent]:.3f}", 20, anchor="end")
        if overall[agent] > 0:
            s.append(f'<rect x="900" y="{y + 12}" width="{scale * overall[agent]:.2f}" height="13" fill="{COLORS[agent]}"/>')
        else:
            s.append(f'<path d="M 900 {y + 10} v 17" stroke="{COLORS[agent]}" stroke-width="2"/>')
        text(1180, y - 22, f"{sps[agent]:,.0f} steps/s", 13, fill="#637084", anchor="end")
    text(900, 640, "steps per second; baselines lane-weighted over their recordings", 15, fill="#637084")
    text(54, 745, "Axis scores", 17, weight="600")
    columns = [330 + i * 750 / (len(axes) - 1) for i in range(len(axes))]
    for i, (name, _, _) in enumerate(axes):
        text(columns[i], 745, name, 15, anchor="middle")
    for j, agent in enumerate(("OaK", "Stream", "PPO")):
        y = 775 + j * 22
        text(54, y, agent, 17, fill=COLORS[agent], weight="600")
        for i, (_, _, means) in enumerate(axes):
            text(columns[i], y, f"{means[agent]:.3f}", 16, anchor="middle")
    text(54, 849, "OaK ahead of PPO on", 15, fill="#637084")
    for i, (name, count, _) in enumerate(axes):
        text(columns[i], 849, f"{ahead[name]} of {count}", 15, anchor="middle", fill="#637084")
    text(54, 904, f"Equal weight per environment \u00b7 outlines, not areas \u00b7 floor = max(Random, Constant) = 0 \u00b7 Oracle = 1 \u00b7 final {window:,} of {steps:,} steps", 15, fill="#637084")
    s.append("</g></svg>")
    CHART.write_text("\n".join(s) + "\n")


def readme(best, oak, window, rows, sps):
    text = README.read_text()
    text = re.sub(r"[\d,]+ SPS on the", f"{sps['OaK']:,.0f} SPS on the", text, count=1)
    lines = text.split("\n")
    first = next(i for i, line in enumerate(lines) if line.startswith("Reward per step over the final"))
    last = next(i for i, line in enumerate(lines) if line.startswith("Baseline throughput"))
    block = [f"Reward per step over the final {window:,} steps of the {oak.steps:,}-step best recording (`{best}`), all {len(rows)} environments with {oak.lanes // len(rows)} seeds each ({oak.lanes:,} lanes), with the Oracle, streaming actor-critic, PPO, constant do-nothing and Random baselines on the same seeds. Mean over seeds; Score = (OaK − Floor) / (Oracle − Floor) with Floor = max(Random, Constant), unclipped; a dash where the Oracle does not rise above the floor.", "",
             "| Environment | Category | OaK | Oracle | Stream | PPO | Constant | Random | Score |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["means"]
        score = normalized(row, "OaK")
        block.append(f"| {row['env']} | {row['category']} | {fleet.fmt(m['OaK'])} | {fleet.fmt(m['Oracle'])} | {fleet.fmt(m['Stream'])} | {fleet.fmt(m['PPO'])} | {fleet.fmt(m['Constant'])} | {fleet.fmt(m['Random'])} | {fleet.fmt(score) if math.isfinite(score) else '—'} |")
    block += ["", f"Baseline throughput, lane-weighted over the recordings that hold each environment's baseline: streaming actor-critic {sps['Stream']:,.0f} steps/s, PPO {sps['PPO']:,.0f} steps/s, Random {sps['Random']:,.0f} steps/s, Constant {sps['Constant']:,.0f} steps/s, Oracle {sps['Oracle']:,.0f} steps/s."]
    README.write_text("\n".join(lines[:first] + block + lines[last + 1:]))


if __name__ == "__main__":
    best, oak, window, rows, sps = load()
    chart(best, oak.steps, window, rows, sps)
    readme(best, oak, window, rows, sps)
    for name, count, means in axis_table(rows):
        print(f"{name:11s} n={count:3d}  " + "  ".join(f"{agent} {means[agent]:+.3f}" for agent in AGENTS))
