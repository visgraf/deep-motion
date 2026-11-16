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
    fn = {"2": fg.stage2, "3": fg.stage3, "4": lambda o: fg.stage4(a.run, o),
          "5a": lambda o: fg.stage5a(dict(r.split("=", 1) for r in a.runs), o, stride=a.stride),
          "5b2": lambda o: fg.stage5b2({k: v.split(",") for k, v in (r.split("=", 1) for r in a.runs)}, o)}[a.stage]
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
    poses = oracle.run_poses(a.kind, n=a.n, ds=a.ds)
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
                            for f, v in (("pix", e["pix"]), ("rtrue", e["rtrue"]), ("I", e["I"]), ("nvis", e["nvis"]),
                                         ("b_ref", e["b_ref"]), ("lat_ref", e["lat_ref"]))})
    return 0 if all(ok for _, ok in res) else 1


def stage5b2_render(a) -> int:
    from . import fusion, sensor
    poses = fusion.path_poses(kind=a.path)[a.first:a.last + 1]
    return sensor.render(a.run, sensor.SlitFan(width=a.width), [(s, c, R) for s, c, R, _ in poses], spp=a.spp,
                         seed_base=a.seed_base + a.first, device=a.device, git={"deep-motion": _git(REPO)})


def stage5b2(a) -> int:
    from . import fusion
    res, num, ex = fusion.selftest(a.runs, kind=a.path)
    for name, ok in res:
        print(f"{PREFIX} stage 5b.2 {'PASS' if ok else 'FAIL'} {name}")
    print(f"{PREFIX} " + json.dumps({k: v for k, v in num.items() if k != "progress_0.5px"}))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "stage5b2", "runs": a.runs, "checks": [{"name": n, "ok": bool(ok)} for n, ok in res],
               "numbers": num, "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / ("stage5b2.json" if a.path == "composed" else f"stage5b2_{a.path}.json")).write_text(json.dumps(rec, indent=1, default=float))
    return 0 if all(ok for _, ok in res) else 1


def stage5a_table(a) -> int:
    """The oracle's sector tables on any runs, no checks (the addendum to #5: longer windows)."""
    from . import oracle
    out = {}
    for spec in a.runs:
        name, d = spec.split("=", 1)
        ev = oracle.evaluate(d, stride=a.stride)
        out[name] = {"run": d, "anchors": int(len(ev["rtrue"])), "seen_by_another": float(np.mean(ev["nvis"] >= 1)),
                     "table": oracle.sector_table(ev), "profile_0.25px": oracle.profile(ev)}
        print(f"{PREFIX} {name}: " + json.dumps(out[name]["table"], ensure_ascii=False))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "stage5a-table", "runs": out, "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / "stage5a_table.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m deep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest", help="a stage's checks: 2 the primitives' geometry, 3 the atlas (numpy)")
    p.add_argument("--stage", default="2", choices=["2", "3", "4"])
    p.add_argument("--run", default=None, help="stage 4: the rendered run for S4.2")
    p.add_argument("--out", default=None)
    p = sub.add_parser("figures", help="a stage's figures")
    p.add_argument("--stage", default="2", choices=["2", "3", "4", "5a", "5b2"])
    p.add_argument("--run", default=None, help="stage 4: the rendered run")
    p.add_argument("--runs", nargs="*", default=[], help="stage 5a: name=DIR ...")
    p.add_argument("--stride", type=int, default=1, help="stage 5a: keep every n-th reference anchor")
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
    p.add_argument("--n", type=int, default=21, help="looks")
    p.add_argument("--ds", type=float, default=0.01, help="step, m (line, arc)")
    p.add_argument("--device", default="CPU")
    p = sub.add_parser("stage5a", help="stage 5a: the oracle evaluation and its checks on the three runs")
    p.add_argument("--line", required=True)
    p.add_argument("--arc", required=True)
    p.add_argument("--pivot", required=True)
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage5a-table", help="stage 5a: the oracle's sector tables on runs, no checks")
    p.add_argument("runs", nargs="+", help="name=DIR")
    p.add_argument("--stride", type=int, default=1, help="keep every n-th reference anchor (memory at 3600 wide)")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage5b2-render", help="stage 5b.2: looks first..last of the composed path (CPU)")
    p.add_argument("--run", required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--last", type=int, required=True)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--seed-base", type=int, default=10000)
    p.add_argument("--path", default="composed", choices=["composed", "straight"])
    p.add_argument("--device", default="CPU")
    p = sub.add_parser("stage5b2", help="stage 5b.2: the fusion and checks F1-F3 on the composed path's runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--path", default="composed", choices=["composed", "straight"])
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return {"stage5b2-render": stage5b2_render, "stage5b2": stage5b2, "stage5a-table": stage5a_table, "selftest": selftest, "figures": figures, "stage4-render": stage4_render, "stage5a-render": stage5a_render,
            "stage5a": stage5a}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
