"""python -m deep <command>: deep-motion's commands.

    selftest [--stage 2|3] [--out DIR]   stage 2 (#2) the primitives' geometry; stage 3 (#3) the atlas; numpy only
    figures [--stage 2|3] --out DIR      a stage's figures
    selftest --stage 4 [--run DIR]       stage 4 (#4) the slit fan on the Panorama carrier; S4.2 needs the run
    stage4-render --run DIR              stage 4's three looks on an arc in the calibration room (CPU)
    stage5a-render --kind K --run DIR    stage 5a (#5): a short Panorama run, line / arc / pivot (CPU)
    stage5a --line D --arc D --pivot D   stage 5a: the oracle evaluation and checks O5.1-O5.3
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
    from . import atlas, primitives, sensor
    if a.stage == "4":
        res, num = sensor.selftest(run_dir=a.run)
    else:
        res, num = {"2": primitives, "3": atlas}[a.stage].selftest()
    for name, ok in res:
        print(f"{PREFIX} stage {a.stage} {'PASS' if ok else 'FAIL'} {name}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rec = {"command": "selftest", "checks": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num,
               "numpy": np.__version__, "python": sys.version.split()[0], "computed_by": {"deep-motion": _git(REPO)}}
        (out / f"stage{a.stage}_selftest.json").write_text(json.dumps(rec, indent=1, default=float))
    return 0 if all(ok for _, ok in res) else 1


def figures(a) -> int:
    from . import figures as fg
    fn = {"2": fg.stage2, "3": fg.stage3, "4": lambda o: fg.stage4(a.run, o)}[a.stage]
    for f in fn(a.out):
        print(f"{PREFIX} figure {f}")
    return 0


def stage4_render(a) -> int:
    from . import sensor
    fan = sensor.SlitFan(width=a.width)
    return sensor.render(a.run, fan, sensor.arc_run_poses(), spp=a.spp, seed_base=a.seed_base, device=a.device,
                         git={"deep-motion": _git(REPO)})


def stage5a_render(a) -> int:
    from . import oracle, sensor
    poses = oracle.run_poses(a.kind)
    return sensor.render(a.run, sensor.SlitFan(width=a.width), poses, spp=a.spp, seed_base=a.seed_base, device=a.device,
                         git={"deep-motion": _git(REPO)})


def stage5a(a) -> int:
    from . import oracle
    res, num, ev = oracle.selftest({"line": a.line, "arc": a.arc, "pivot": a.pivot})
    for name, ok in res:
        print(f"{PREFIX} stage 5a {'PASS' if ok else 'FAIL'} {name}")
    for k, t in num["tables"].items():
        print(f"{PREFIX} {k}: " + json.dumps(t))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rec = {"command": "stage5a", "runs": {"line": a.line, "arc": a.arc, "pivot": a.pivot},
               "checks": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num,
               "computed_by": {"deep-motion": _git(REPO)}}
        (out / "stage5a.json").write_text(json.dumps(rec, indent=1, default=float))
        np.savez_compressed(out / "stage5a_maps.npz", **{f"{k}_{f}": v for k, e in ev.items()
                            for f, v in (("pix", e["pix"]), ("rtrue", e["rtrue"]), ("I", e["I"]), ("nvis", e["nvis"]))})
    return 0 if all(ok for _, ok in res) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m deep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest", help="a stage's checks: 2 the primitives' geometry, 3 the atlas (numpy)")
    p.add_argument("--stage", default="2", choices=["2", "3", "4"])
    p.add_argument("--run", default=None, help="stage 4: the rendered run for S4.2")
    p.add_argument("--out", default=None)
    p = sub.add_parser("figures", help="a stage's figures")
    p.add_argument("--stage", default="2", choices=["2", "3", "4"])
    p.add_argument("--run", default=None, help="stage 4: the rendered run")
    p.add_argument("--out", required=True)
    p = sub.add_parser("stage4-render", help="stage 4: three Panorama looks on an arc in the calibration room (CPU)")
    p.add_argument("--run", required=True)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--seed-base", type=int, default=9000)
    p.add_argument("--device", default="CPU")
    p = sub.add_parser("stage5a-render", help="stage 5a: a short Panorama run (line, arc or pivot) in the calibration room")
    p.add_argument("--kind", required=True, choices=["line", "arc", "pivot"])
    p.add_argument("--run", required=True)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--seed-base", type=int, default=9100)
    p.add_argument("--device", default="CPU")
    p = sub.add_parser("stage5a", help="stage 5a: the oracle evaluation and its checks on the three runs")
    p.add_argument("--line", required=True)
    p.add_argument("--arc", required=True)
    p.add_argument("--pivot", required=True)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return {"selftest": selftest, "figures": figures, "stage4-render": stage4_render, "stage5a-render": stage5a_render,
            "stage5a": stage5a}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
