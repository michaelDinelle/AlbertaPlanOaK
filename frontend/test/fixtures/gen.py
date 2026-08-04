#!/usr/bin/env python3
"""Fixture driver for the OaK frontend sim-core golden tests.

Runs every per-domain generator that is present next to this file:

    gen_core.py     Philox vectors, u01 edges, cheetah FK      (sim-core track)
    gen_grid.py     gridworld decoder goldens                  (grid track)
    gen_classic.py  classic-control decoder goldens            (classic track)

Each generator module exposes `generate(outdir: Path) -> list[str]` and returns
the filenames it wrote. Generators own disjoint files so tracks never collide.

Run with the repo venv (numpy + the tools/ reference modules):

    /Users/studio/OaKMeta/.venv/bin/python frontend/test/fixtures/gen.py

Options:
    --only core,grid    run a subset of generators
    --list              print the generators that would run, then exit
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOOLS = REPO / "tools"

# generator module -> short name used by --only
GENERATORS = [
    ("gen_core", "core"),
    ("gen_grid", "grid"),
    ("gen_classic", "classic"),
]


def _prepare_path() -> None:
    for p in (str(HERE), str(TOOLS)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated subset: core,grid,classic")
    ap.add_argument("--list", action="store_true", help="list available generators and exit")
    args = ap.parse_args(argv)

    _prepare_path()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    present = [(mod, short) for mod, short in GENERATORS if (HERE / f"{mod}.py").exists()]
    if args.list:
        for mod, short in present:
            print(f"{short:8s} {mod}.py")
        missing = [m for m, _ in GENERATORS if not (HERE / f"{m}.py").exists()]
        for m in missing:
            print(f"{'-':8s} {m}.py (absent)")
        return 0

    if wanted:
        unknown = wanted - {short for _, short in GENERATORS}
        if unknown:
            ap.error(f"unknown generator(s): {', '.join(sorted(unknown))}")

    ran = 0
    written: list[str] = []
    failed: list[str] = []
    for mod, short in present:
        if wanted and short not in wanted:
            continue
        print(f"== {mod} ==")
        try:
            m = importlib.import_module(mod)
            files = m.generate(HERE)
        except Exception as e:  # a broken sibling generator must not kill the rest
            print(f"   FAILED: {type(e).__name__}: {e}")
            failed.append(mod)
            continue
        ran += 1
        for f in files or []:
            written.append(f)
            print(f"   wrote {f}")

    print(f"\n{ran} generator(s), {len(written)} file(s) written to {HERE}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
