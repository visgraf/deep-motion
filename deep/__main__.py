"""python -m deep <command>: deep-motion's commands.

    selftest [--out DIR]    stage 2 (#2): the primitives' geometry, numpy only
    figures --out DIR       stage 2's figure
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PREFIX = "[deep]"


def _git(path: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "clean": run("status", "--porcelain", "--untracked-files=no") == ""}


def selftest(a) -> int:
    from . import primitives
    res, num = primitives.selftest()
    for name, ok in res:
        print(f"{PREFIX} stage 2 {'PASS' if ok else 'FAIL'} {name}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rec = {"command": "selftest", "checks": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num,
               "numpy": np.__version__, "python": sys.version.split()[0], "computed_by": {"deep-motion": _git(REPO)}}
        (out / "stage2_selftest.json").write_text(json.dumps(rec, indent=1, default=float))
    return 0 if all(ok for _, ok in res) else 1


def figures(a) -> int:
    from . import figures as fg
    for f in fg.stage2(a.out):
        print(f"{PREFIX} figure {f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m deep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest", help="stage 2: the primitives' geometry (numpy)")
    p.add_argument("--out", default=None)
    p = sub.add_parser("figures", help="stage 2's figure")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"selftest": selftest, "figures": figures}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
