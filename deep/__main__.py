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
          "5b2": lambda o: fg.stage5b2({k: v.split(",") for k, v in (r.split("=", 1) for r in a.runs)}, o,
                                       hit_stride=a.stride)}[a.stage]
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
                         git={"deep-motion": _git(REPO)}, drop_exr=a.drop_exr)


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
    start = dict(fusion.START)
    if a.start:
        x, y, hd = (float(v) for v in a.start.split(","))
        start.update({"p0": (x, y), "heading_deg": hd})
    if a.height is not None:
        start["h"] = a.height
    allp = fusion.path_poses(kind=a.path, step=a.step, start=start)
    last = len(allp) - 1 if a.last is None else a.last
    poses = allp[a.first:last + 1]
    meta = {"experiment": "stage 5b.2: the composed path and the fusion (#7)", "path": a.path, "step_m": a.step,
            "start": {"p0": list(start["p0"]), "heading_deg": start["heading_deg"], "h": start["h"]},
            "looks_of_path": [a.first, last, len(allp)]}
    return sensor.render(a.run, sensor.SlitFan(width=a.width), [(s, c, R) for s, c, R, _ in poses], spp=a.spp,
                         seed_base=a.seed_base + a.first, device=a.device, git={"deep-motion": _git(REPO)},
                         blend=a.blend, meta=meta, drop_exr=a.drop_exr)


def stage5b2(a) -> int:
    from . import fusion
    res, num, ex = fusion.selftest(a.runs, kind=a.path, cell=a.cell, hit_stride=a.hit_stride, max_cells=a.max_cells)
    for name, ok in res:
        print(f"{PREFIX} stage 5b.2 {'PASS' if ok else 'FAIL'} {name}")
    print(f"{PREFIX} " + json.dumps({k: v for k, v in num.items() if k != "progress_0.5px"}))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "stage5b2", "runs": a.runs, "checks": [{"name": n, "ok": bool(ok)} for n, ok in res],
               "numbers": num, "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / ("stage5b2.json" if num["path"] == "composed" else f"stage5b2_{num['path']}.json")).write_text(json.dumps(rec, indent=1, default=float))
    return 0 if all(ok for _, ok in res) else 1


def stage7a(a) -> int:
    from . import match as mt
    if a.b2:
        res, num = mt.selftest_b2(dict(kv.split("=", 1) for kv in a.b2), rung=a.rung, max_n=a.max_anchors)
    elif a.r2:
        res, num = mt.selftest_r2(dict(kv.split("=", 1) for kv in a.r2), max_n=a.max_anchors)
    elif a.k14:
        res, num = mt.selftest_k14(max_n=a.max_anchors)
    elif a.k13:
        res, num = mt.selftest_k13(max_n=a.max_anchors)
    elif a.k12:
        res, num = mt.selftest_k12(max_n=a.max_anchors)
    elif a.k11:
        res, num = mt.selftest_k11(max_n=a.max_anchors)
    elif a.k10:
        res, num = mt.selftest_k10(max_n=a.max_anchors)
    elif a.k9:
        res, num = mt.selftest_k9(max_n=a.max_anchors)
    elif a.k8:
        res, num = mt.selftest_k8(max_n=a.max_anchors)
    elif a.a:
        res, num = mt.selftest_a(rung=a.rung, kinds=tuple(a.kinds), max_n=a.max_anchors)
    else:
        res, num = (mt.selftest_k3 if a.k3 else mt.selftest_b)(max_n=a.max_anchors)
    for name, ok in res:
        print(f"{PREFIX} stage 7a {'PASS' if ok else 'FAIL'} {name}")
    print(f"{PREFIX} " + json.dumps(num))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "stage7a", "checks": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num,
               "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / (f"stage7a_b2_{a.rung}.json" if a.b2 else "stage7a_r2.json" if a.r2 else "stage7a_k14.json" if a.k14 else "stage7a_k13.json" if a.k13 else "stage7a_k12.json" if a.k12 else "stage7a_k11.json" if a.k11 else "stage7a_k10.json" if a.k10 else "stage7a_k9.json" if a.k9 else "stage7a_k8.json" if a.k8 else f"stage7a_a_{a.rung}.json" if a.a else "stage7a_k3.json" if a.k3 else "stage7a_b.json")).write_text(json.dumps(rec, indent=1))
    return 0 if all(ok for _, ok in res) else 1


def stage8(a) -> int:
    from . import e2e
    if a.r0:
        paths = {"turn": e2e.r0_path("turn90"), "straight": e2e.r0_path("straight_long")}
    else:
        paths = {}
        for kv in a.runs:
            name, run_dir = kv.split("=", 1)
            paths[name] = e2e.render_path(run_dir)
    res, num = e2e.selftest(paths, max_n=a.max_anchors)
    for name, ok in res:
        print(f"{PREFIX} stage 8 {'PASS' if ok else 'FAIL'} {name}")
    print(f"{PREFIX} " + json.dumps(num))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "stage8", "rung": "r0" if a.r0 else "render", "max_anchors": a.max_anchors,
               "checks": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num,
               "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / ("stage8_r0.json" if a.r0 else "stage8.json")).write_text(json.dumps(rec, indent=1))
    return 0 if all(ok for _, ok in res) else 1


def demo2(a) -> int:
    from . import demo2 as d2, demo as dm
    import numpy as np
    if a.r0:
        T, looks, truth, vis, fan, refs = d2.r0_tour()
        v2 = (d2.check_continuity_r0(T), d2.check_continuity_r0(T, shift=np.array([0.01, 0.0, 0.0])))
    else:
        T, looks, truth, vis, fan, refs = d2.render_tour(a.straight, a.turn)
        step = float(np.median(np.diff([t[3] for t in T])))
        v2 = (dm.check_continuity(T, step, [a.straight], [a.turn]),
              dm.check_continuity(T, step, [a.straight], [a.turn], shift=np.array([0.01, 0.0, 0.0])))
    return d2.run(T, looks, truth, vis, fan, refs, a.out, v2, limit=a.limit, best=a.best, git=_git(REPO),
                  videos=not a.no_video)


def assess(a) -> int:
    from . import assess as asx, e2e
    paths = {}
    if a.r0:
        for name, kind in (("turn", "turn90"), ("straight", "straight_long")):
            looks, truth, fan, _ = e2e.r0_path(kind, refs=asx.references(218))
            paths[name] = (looks, truth, fan, asx.references(len(looks)), asx.r0_vis(kind, fan))
    else:
        for kv in a.runs:
            name, run_dir = kv.split("=", 1)
            from . import oracle as orc
            _, L = orc.load(run_dir)
            refs = asx.references(len(L))
            looks, truth, fan, _ = e2e.render_path(run_dir, refs=refs)
            paths[name] = (looks, truth, fan, refs, asx.render_vis(run_dir, fan))
    res, num = asx.selftest(paths)
    for name, ok in res:
        print(f"{PREFIX} assess {'PASS' if ok else 'FAIL'} {name}")
    print(f"{PREFIX} " + json.dumps(num))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "assess", "rung": "r0" if a.r0 else "render",
               "checks": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num,
               "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / ("assess_r0.json" if a.r0 else "assess.json")).write_text(json.dumps(rec, indent=1))
    return 0 if all(ok for _, ok in res) else 1


def _torch_threads():
    import os
    import torch
    if os.environ.get("DEEP_TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["DEEP_TORCH_THREADS"]))
    return torch


def stage11_selftest(a) -> int:
    _torch_threads()
    from . import ntrace as nt
    res = nt.selftest()
    for name, ok, val, cname, cfail, cval in res:
        print(f"{PREFIX} stage11 {'PASS' if ok else 'FAIL'} {name}: {val}; {cname}: {'fails' if cfail else 'DOES NOT FAIL'} ({cval})")
    ok_all = all(r[1] and r[4] for r in res)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "stage11-selftest", "checks": [{"name": r[0], "ok": bool(r[1]), "value": r[2], "control": r[3],
                                                          "control_fails": bool(r[4]), "control_value": r[5]} for r in res],
               "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / "n_selftest.json").write_text(json.dumps(rec, indent=1, default=float))
    return 0 if ok_all else 1


def stage11_fixtures(a) -> int:
    _torch_threads()
    from . import nfixtures as nf
    res = nf.main(Path(a.out), names=a.only)
    res["computed_by"] = {"deep-motion": _git(REPO)}
    (Path(a.out) / "c1_fixtures.json").write_text(json.dumps(res, indent=1, default=float))
    for g in res["gates"]:
        print(f"{PREFIX} C1 {'PASS' if g['ok'] else 'FAIL'} {g['fixture']} {g['label']}: {g}")
    for c in res["controls"]:
        print(f"{PREFIX} C1 control {'fails' if c['fails'] else 'DOES NOT FAIL'}: {c['control']}")
    return 0 if res["ok"] and res["controls_fail"] else 1


def note_figs(a) -> int:
    from . import note_figs as nf
    for f in nf.run(a.out):
        print(f"{PREFIX} wrote {f}")
    return 0


def moebius(a) -> int:
    from . import moebius as mb
    res, num = mb.selftest()
    for name, ok in res:
        print(f"{PREFIX} moebius {'PASS' if ok else 'FAIL'} {name}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "moebius", "checks": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num,
               "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / "moebius_selftest.json").write_text(json.dumps(rec, indent=1))
    return 0 if all(ok for _, ok in res) else 1


def compare(a) -> int:
    from . import compare as cp
    for f in cp.run(a.out):
        print(f"{PREFIX} wrote {f}")
    return 0


def demo(a) -> int:
    from . import demo as dm
    return dm.run(a.straight, a.turn, a.out, sigma_px=a.sigma, cell=a.cell, hit_stride=a.hit_stride,
                  max_cells=a.max_cells, limit=a.limit, best=a.best, git={"deep-motion": _git(REPO)})


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
    p.add_argument("--stride", type=int, default=1, help="stage 5a: keep every n-th reference anchor; stage 5b2: bin a random 1/n of each look's hits")
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
    p.add_argument("--drop-exr", action="store_true", help="delete each look's EXR once its record exists (disk)")
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
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=None, help="default: the path's last look")
    p.add_argument("--step", type=float, default=0.025, help="m of arc length between looks")
    p.add_argument("--blend", default=None, help="a scene .blend (default: the calibration room)")
    p.add_argument("--start", default=None, help="x,y,heading_deg of the path's start (default: the room's EYE, 0,0,90)")
    p.add_argument("--height", type=float, default=None, help="head height, m (default 1.6)")
    p.add_argument("--drop-exr", action="store_true", help="delete each look's EXR once its record exists (disk)")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--seed-base", type=int, default=10000)
    p.add_argument("--path", default="composed", choices=["composed", "straight", "turn90", "straight_long"])
    p.add_argument("--device", default="CPU")
    p = sub.add_parser("stage5b2", help="stage 5b.2: the fusion and checks F1-F3 on the composed path's runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--path", default=None, choices=["composed", "straight", "turn90", "straight_long"],
                   help="default: from run.json (composed for runs before it was recorded)")
    p.add_argument("--cell", type=float, default=0.02)
    p.add_argument("--hit-stride", type=int, default=1, help="bin a random 1/n of each look's hits (memory at 3600 wide)")
    p.add_argument("--max-cells", type=int, default=None, help="a random subset of the cells (time)")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage7a", help="#14: the trace sweep B on R0 (line, arc), checks K1-K2")
    p.add_argument("--max-anchors", type=int, default=20000)
    p.add_argument("--k3", action="store_true", help="check K3: B's reliability marking")
    p.add_argument("--a", action="store_true", help="checks K4-K5 (rung r0) or K6-K7 (rung r1): A, and both on R1")
    p.add_argument("--rung", choices=["r0", "r1", "r2"], default="r0")
    p.add_argument("--b2", nargs="+", default=None, metavar="KIND=RUN",
                   help="B2, the patch cost (#17): K18 with --rung r1, K19 with --rung r2, on render runs")
    p.add_argument("--r2", nargs="+", default=None, metavar="KIND=RUN",
                   help="checks K15-K17 (#17) on render runs, e.g. line=runs/r2-line arc=runs/r2-arc")
    p.add_argument("--k14", action="store_true", help="check K14: the planar prior, method P (R1)")
    p.add_argument("--k13", action="store_true", help="check K13: method H2, grown in rounds (R1)")
    p.add_argument("--k12", action="store_true", help="check K12: method H, seeding the anchors without texture (R1)")
    p.add_argument("--k11", action="store_true", help="check K11: method G, two layers per contour anchor (R1)")
    p.add_argument("--k10", action="store_true", help="check K10: the surface metric for contour anchors (R1)")
    p.add_argument("--k9", action="store_true", help="check K9: method F, occlusion ordering in time (R1)")
    p.add_argument("--k8", action="store_true", help="check K8: method E at depth edges (R1; R0 reported)")
    p.add_argument("--kinds", nargs="+", default=["line", "arc"])
    p.add_argument("--out", default=None)
    p = sub.add_parser("demo2", help="#23: demo D2, the perfect and the real matcher side by side along #9's tour")
    p.add_argument("--straight", default=None, help="the straight path's run (R3)")
    p.add_argument("--turn", default=None, help="the turn's run (R3)")
    p.add_argument("--r0", action="store_true", help="the dry run: R0's calibration room")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="the first n looks only, no orbit (a preview)")
    p.add_argument("--no-video", action="store_true", help="the record and the last frames only")
    p.add_argument("--best", action="store_true", help="x264 preset slow")
    p = sub.add_parser("assess", help="#22: the perfect and the real matcher on the same runs, same pipeline (A1-A3)")
    p.add_argument("--runs", nargs="*", default=[], help="turn=DIR straight=DIR (render runs)")
    p.add_argument("--r0", action="store_true", help="the dry run: R0's calibration room, turn90 and straight_long")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage8", help="#19: end to end, B's reliable estimates fused along a path (E1-E3)")
    p.add_argument("--runs", nargs="*", default=[], help="turn=DIR straight=DIR (render runs)")
    p.add_argument("--r0", action="store_true", help="the dry run: R0's calibration room, turn90 and straight_long")
    p.add_argument("--max-anchors", type=int, default=20000)
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage11-selftest", help="#26: N's geometry (N.1-N.5) with controls; numpy and torch, no renders")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage11-fixtures", help="#26, C1: M4's fixtures rebuilt for the panorama, N's gates and controls")
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", default=None, help="fixture names (a preview; C1 is the full set)")
    p = sub.add_parser("note-figs", help="the Moebius Traces note's schematic figures (formulas only)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("moebius", help="#11: the primitives on the Riemann sphere, checks M1-M4 with controls")
    p.add_argument("--out", default=None)
    p = sub.add_parser("compare", help="#11: the representations side by side (figures, Chat sandbox)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("demo", help="#9: the tour (the straight to the turn's start, then the turn) as two videos")
    p.add_argument("--straight", nargs="+", required=True, help="the straight path's run(s)")
    p.add_argument("--turn", nargs="+", required=True, help="the turn's run(s)")
    p.add_argument("--out", required=True)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--cell", type=float, default=0.02)
    p.add_argument("--hit-stride", type=int, default=1)
    p.add_argument("--max-cells", type=int, default=60000)
    p.add_argument("--limit", type=int, default=None, help="the first n looks only, no orbit, no V1 (a preview)")
    p.add_argument("--best", action="store_true", help="x264 preset slow")
    a = ap.parse_args(argv)
    return {"stage11-selftest": stage11_selftest, "stage11-fixtures": stage11_fixtures, "demo2": demo2, "assess": assess, "stage8": stage8, "stage7a": stage7a, "note-figs": note_figs, "moebius": moebius, "compare": compare, "demo": demo, "stage5b2-render": stage5b2_render, "stage5b2": stage5b2, "stage5a-table": stage5a_table, "selftest": selftest, "figures": figures, "stage4-render": stage4_render, "stage5a-render": stage5a_render,
            "stage5a": stage5a}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
