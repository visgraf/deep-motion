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


def demo2_ply(a) -> int:
    from . import demo2 as d2
    import numpy as np
    if a.r0:
        T, looks, truth, vis, fan, refs = d2.r0_tour()
        rgb_of = lambda j: np.repeat(looks[j][3][..., None].astype(np.float64), 3, axis=2)
        name = "calibration_room"
    else:
        T, looks, truth, vis, fan, refs = d2.render_tour(a.straight, a.turn)
        rgb_of = d2.render_rgb(T)
        name = "classroom"
    return d2.export_ply(T, looks, truth, vis, fan, refs, rgb_of, a.ply_dir, a.out, git=_git(REPO), name=name)


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
    git = {"deep-motion": _git(REPO)}                           # read before the run
    res = nf.main(Path(a.out), names=a.only)
    res["computed_by"] = git
    (Path(a.out) / "c1_fixtures.json").write_text(json.dumps(res, indent=1, default=float))
    for g in res["gates"]:
        print(f"{PREFIX} C1 {'PASS' if g['ok'] else 'FAIL'} {g['fixture']} {g['label']}: {g}")
    for c in res["controls"]:
        print(f"{PREFIX} C1 control {'fails' if c['fails'] else 'DOES NOT FAIL'}: {c['control']}")
    return 0 if res["ok"] and res["controls_fail"] else 1


def _stage11_inputs(a):
    """The rows' inputs for stage11 and stage11b: the real runs, or the dry run's synthetic room."""
    from . import assess as asx, e2e, straight as st
    from . import oracle as orc
    if a.dry_run:
        paths, m4in = st.synthetic_room(cache=Path(a.cache) / "room.npz" if a.cache else None)
        refs = paths["128"][3]
        m4 = {}
        for lvl in ("1024", "128"):
            f = Path(a.cache) / f"m4_{lvl}.npz" if a.cache else None
            if f is not None and f.exists():
                m4[lvl] = list(np.load(f, allow_pickle=True)["obj"])
            else:
                m4[lvl] = st.m4_dry(m4in, paths["128"][0], refs, lvl)
                if f is not None:
                    np.savez_compressed(f, obj=np.array(m4[lvl], dtype=object))
        rec_assess = None
        src = {"dry_run": "a synthetic room of M4's fixture planes (deep/straight.py synthetic_room), Chat sandbox, not gated",
               "noise": st.DRY_NOISE}
    else:
        _, L = orc.load(a.r128)
        refs = asx.references(len(L))
        looks, truth, fan, _ = e2e.render_path(a.r128, refs=refs)
        paths = {"128": (looks, truth, fan, refs, asx.render_vis(a.r128, fan))}
        if a.r1024:
            l2, _, _, _ = e2e.render_path(a.r1024, refs=refs)
            paths["1024"] = (l2, truth, fan, refs, None)
        m4 = {}
        for kv in a.m4:
            lvl, dirs = kv.split("=", 1)
            res_dir, run_dir = dirs.split(",", 1)
            m4[lvl] = st.m4_points(res_dir, run_dir)
        rec_assess = json.loads((REPO / "docs/runs/dm-issue-22/assess.json").read_text())
        src = {"r128": a.r128, "r1024": a.r1024, "m4": a.m4}
    return paths, m4, rec_assess, src


def stage11(a) -> int:
    _torch_threads()
    import time as _t
    from . import straight as st
    t0 = _t.perf_counter()
    git = {"deep-motion": _git(REPO)}                           # read before the run
    paths, m4, rec_assess, src = _stage11_inputs(a)
    out = st.run(paths, m4=m4, cache=a.cache, rows=a.rows, assess_record=rec_assess,
                 log=lambda m: print(f"{PREFIX} {m}", flush=True))
    out.update({"command": "stage11", "sources": src, "computed_by": git,
                "seconds_total": round(_t.perf_counter() - t0, 1)})
    for c in out["checks"]:
        cf = c["control_fails"]
        print(f"{PREFIX} {'PASS' if c['ok'] else 'FAIL'} {c['check']}; control {c['control']}: "
              f"{'not evaluated' if cf is None else ('fails' if cf else 'DOES NOT FAIL')}")
    for line in st.table(out):
        print(line)
    print(f"{PREFIX} gap: " + json.dumps(out["gap"], default=float))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        name = "stage11_dry.json" if a.dry_run else "stage11.json"
        (Path(a.out) / name).write_text(json.dumps(out, indent=1, default=float))
        (Path(a.out) / name.replace(".json", "_table.md")).write_text("\n".join(st.table(out)) + "\n")
    return 0 if all(c["ok"] and c["control_fails"] is not False for c in out["checks"]) else 1


def stage11b(a) -> int:
    _torch_threads()
    import time as _t
    from . import straight as st
    t0 = _t.perf_counter()
    git = {"deep-motion": _git(REPO)}                           # read before the run
    paths, m4, rec_assess, src = _stage11_inputs(a)
    out = st.breakdown(paths, m4=m4, cache=a.cache, assess_record=rec_assess,
                       log=lambda m: print(f"{PREFIX} {m}", flush=True))
    out.update({"command": "stage11b", "sources": src, "computed_by": git,
                "seconds_total": round(_t.perf_counter() - t0, 1)})
    lines = st.breakdown_tables(out)
    for line in lines:
        print(line)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        name = "stage11b_dry" if a.dry_run else "stage11b"
        (Path(a.out) / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        (Path(a.out) / f"{name}_tables.md").write_text("\n".join(lines) + "\n")
    ok = all(v["equal_to_stage9"] for v in out["consistency"].values()) if out["consistency"] else True
    return 0 if ok else 1


def stage11c(a) -> int:
    _torch_threads()
    import time as _t
    from . import straight as st
    t0 = _t.perf_counter()
    git = {"deep-motion": _git(REPO)}                           # read before the run
    paths, m4, rec_assess, src = _stage11_inputs(a)
    out = st.diagnose(paths, m4=m4, cache=a.cache, assess_record=rec_assess,
                      log=lambda m: print(f"{PREFIX} {m}", flush=True))
    out.update({"command": "stage11c", "sources": src, "computed_by": git,
                "seconds_total": round(_t.perf_counter() - t0, 1)})
    lines = st.diagnose_tables(out)
    for line in lines:
        print(line)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        name = "stage11c_dry" if a.dry_run else "stage11c"
        (Path(a.out) / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        (Path(a.out) / f"{name}_tables.md").write_text("\n".join(lines) + "\n")
    ok = all(v.get("equal_to_stage9", v.get("equal", True)) for v in out["consistency"].values())
    return 0 if ok else 1


def standard_selftest(a) -> int:
    """S1's code (deep/standard.py) and stage 12's span logic (deep/confirm.py), each check with its control."""
    from . import standard as sd
    res = [("standard", r) for r in sd.selftest()]
    try:
        from . import confirm as cf, coverage as cv
        res += [("stage12", r) for r in cf.selftest()]
        res += [("stage13a", r) for r in cv.selftest()]
        from . import arcfixtures as af
        res += [("stage14", r) for r in af.selftest()]
    except ImportError as e:                                    # deep/confirm.py needs torch (via ntrace)
        print(f"{PREFIX} stage12 span logic not run: {e}")
    for who, (name, ok, val, cname, cfail, cval) in res:
        print(f"{PREFIX} {who} {'PASS' if ok else 'FAIL'} {name}: {val}; control {cname}: "
              f"{'fails' if cfail else 'DOES NOT FAIL'} ({cval})")
    ok_all = all(r[1] and r[4] for _, r in res)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"command": "standard-selftest", "standard": sd.VERSION,
               "checks": [{"module": w, "name": r[0], "ok": bool(r[1]), "value": r[2], "control": r[3],
                           "control_fails": bool(r[4]), "control_value": r[5]} for w, r in res],
               "computed_by": {"deep-motion": _git(REPO)}}
        (Path(a.out) / "standard_selftest.json").write_text(json.dumps(rec, indent=1, default=str))
    return 0 if ok_all else 1


def stage12(a) -> int:
    """#28: the confirmation distance, the first run under S1."""
    _torch_threads()
    import time as _t
    from . import confirm as cf, ntrace as nt, standard as sd
    t0 = _t.perf_counter()
    git = {"deep-motion": _git(REPO)}                           # read before the run
    paths, m4, _, src = _stage11_inputs(a)
    rec_dir = REPO / "docs/runs/dm-issue-26"
    suffix = "_dry" if a.dry_run else ""
    rec11 = json.loads((rec_dir / f"stage11{suffix}.json").read_text())
    rec11c = json.loads((rec_dir / f"stage11c{suffix}.json").read_text())
    out = cf.run(paths, m4=m4, cache=a.cache, rec11=rec11, rec11c=rec11c, log=lambda m: print(f"{PREFIX} {m}", flush=True))
    controls = ["mirrored poses", "points moved 20 %"]
    if a.dry_run:
        std = sd.stamp("dry", {"scene": "synthetic room of M4's fixture planes, ray-cast", "gated": False},
                       controls=controls)
    else:
        fan = paths["128"][2]
        values = dict(sd.PROFILES["ablation"]["values"], width=int(fan.width),
                      psi_deg=list(nt.SETTINGS["psi_anchor_deg"]), lat_max_deg=float(nt.SETTINGS["lat_max_deg"]),
                      confirm_span_m=list(cf.SPANS_M))
        std = sd.stamp("ablation", values, departures={"confirm_span_m": "the value under study (#28): 0, 0.2 and 0.5 m"},
                       controls=controls)
    out.update(std)
    out.update({"command": "stage12", "sources": src, "computed_by": git,
                "seconds_total": round(_t.perf_counter() - t0, 1)})
    print(f"{PREFIX} S1 record: {std['profile']}, problems: {std['check_record'] or 'none'}")
    for c in out["checks"]:
        ok = c.get("ok")
        head = "control only" if ok is None and "control_fails" in c and c["check"].startswith("the standard") else (
            "not evaluated" if ok is None else ("PASS" if ok else "FAIL"))
        cf_ = c.get("control_fails")
        print(f"{PREFIX} {head} {c['check']}" + (f"; control {c['control']}: "
              f"{'not evaluated' if cf_ is None else ('fails' if cf_ else 'DOES NOT FAIL')} ({c.get('control_value')})"
              if "control" in c else ""))
    for line in cf.table(out):
        print(line)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        name = "stage12_dry" if a.dry_run else "stage12"
        (Path(a.out) / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        (Path(a.out) / f"{name}_table.md").write_text("\n".join(cf.table(out)) + "\n")
    bad = [c for c in out["checks"] if c.get("ok") is False or c.get("control_fails") is False]
    return 1 if bad or std["check_record"] else 0


def stage12b(a) -> int:
    """#28, P-D5: two confirmation rules (same cell; within 2 cm) at spans 0-1.0 m, under S2."""
    _torch_threads()
    import time as _t
    from . import confirm as cf, ntrace as nt, standard as sd
    t0 = _t.perf_counter()
    git = {"deep-motion": _git(REPO)}                           # read before the run
    paths, m4, _, src = _stage11_inputs(a)
    rec12 = json.loads((REPO / "docs/runs/dm-issue-28" / ("stage12_dry.json" if a.dry_run else "stage12.json")).read_text())
    out = cf.run_b(paths, m4=m4, cache=a.cache, rec12=rec12, log=lambda m: print(f"{PREFIX} {m}", flush=True))
    controls = ["mirrored poses", "points moved 20 %"]
    if a.dry_run:
        std = sd.stamp("dry", {"scene": "synthetic room of M4's fixture planes, ray-cast", "gated": False}, controls=controls)
    else:
        fan = paths["128"][2]
        values = dict(sd.PROFILES["ablation"]["values"], width=int(fan.width),
                      psi_deg=list(nt.SETTINGS["psi_anchor_deg"]), lat_max_deg=float(nt.SETTINGS["lat_max_deg"]),
                      confirm_span_m=list(cf.SPANS_12B), confirm_rule=list(cf.RULES))
        std = sd.stamp("ablation", values, departures={
            "confirm_span_m": "a value under study (#28, P-D5): 0, 0.2, 0.5, 0.8 and 1.0 m",
            "confirm_rule": "a value under study (#28, P-D5): the same cell, and an estimate within 2 cm"}, controls=controls)
    out.update(std)
    out.update({"command": "stage12b", "sources": src, "computed_by": git,
                "seconds_total": round(_t.perf_counter() - t0, 1)})
    print(f"{PREFIX} {std['standard']} record: {std['profile']}, problems: {std['check_record'] or 'none'}")
    for c in out["checks"]:
        ok = c.get("ok")
        head = "control only" if c["check"].startswith("the standard") else (
            "not evaluated" if ok is None else ("PASS" if ok else "FAIL"))
        cf_ = c.get("control_fails")
        print(f"{PREFIX} {head} {c['check']}" + (f"; control {c['control']}: "
              f"{'not evaluated' if cf_ is None else ('fails' if cf_ else 'DOES NOT FAIL')} ({c.get('control_value')})"
              if "control" in c else ""))
    for line in cf.table_b(out):
        print(line)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        name = "stage12b_dry" if a.dry_run else "stage12b"
        (Path(a.out) / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        (Path(a.out) / f"{name}_table.md").write_text("\n".join(cf.table_b(out)) + "\n")
    bad = [c for c in out["checks"] if c.get("ok") is False or c.get("control_fails") is False]
    return 1 if bad or std["check_record"] else 0


def demo3(a) -> int:
    """#29, P-D5: demo D3, N1024 along the Classroom's straight path as S2's model grows (video and PLY)."""
    _torch_threads()
    import numpy as np
    from . import assess as asx, confirm as cf, demo2 as d2, demo3 as d3, e2e, fusion as fu, ntrace as nt, oracle as orc
    from . import standard as sd, straight as st
    git = {"deep-motion": _git(REPO)}                           # read before the run
    if a.dry_run:
        paths, _ = st.synthetic_room(K=a.dry_looks, y0=a.dry_y0, fov_frames=False,
                                     cache=Path(a.cache) / "room.npz" if a.cache else None)
        looks = paths["1024"][0]
        T = [(None, lk[1], lk[2], float(lk[0]), 0) for lk in looks]
        rgb_of = lambda j: np.repeat(np.asarray(looks[j][3], float)[..., None], 3, axis=2)
        cache = a.cache                                         # as stage 11's dry run: the room and the rows under one folder
        f12d = REPO / "docs/runs/dm-issue-28/stage12b_dry.json"   # stage 12b's dry run is on #26's room (K = 111)
        rec12b = json.loads(f12d.read_text()) if (f12d.exists() and a.dry_looks == 111) else None
        f13d = REPO / "docs/runs/dm-issue-31/stage13a_dry.json"   # and so is stage 13a's
        rec13a = json.loads(f13d.read_text()) if (f13d.exists() and a.dry_looks == 111) else None
        name, label = "dry_room_straight", "synthetic room (dry run), straight path"
        src = {"dry_run": f"a synthetic room of M4's fixture planes (deep/straight.py synthetic_room, K={a.dry_looks}, "
                          f"y0={a.dry_y0}), Chat sandbox, not gated", "noise": st.DRY_NOISE}
    else:
        _, L = orc.load(a.r128)
        refs = asx.references(len(L))
        looks, truth, fan, _ = e2e.render_path(a.r128, refs=refs)
        paths = {"128": (looks, truth, fan, refs, asx.render_vis(a.r128, fan))}
        l2, _, _, _ = e2e.render_path(a.r1024, refs=refs)
        paths["1024"] = (l2, truth, fan, refs, None)
        T = [(p, c, R, float(s), 0) for p, c, R, s in fu.load_looks([a.r1024])]
        rgb_of = d2.render_rgb(T)
        cache = a.cache
        f12b = REPO / "docs/runs/dm-issue-28/stage12b.json"
        rec12b = json.loads(f12b.read_text()) if f12b.exists() else None
        f13a = REPO / "docs/runs/dm-issue-31/stage13a.json"
        rec13a = json.loads(f13a.read_text()) if f13a.exists() else None
        name, label = "classroom_straight", "Classroom, straight path"
        src = {"r128": a.r128, "r1024": a.r1024, "stage12b": str(f12b.relative_to(REPO)) if rec12b else None}
    if d3.is_map(a.rule):
        name += "_coverage"
    rec = d3.run(paths, cache, a.out, a.ply_dir, T, rgb_of, rule=a.rule, span=a.span, rec12b=rec12b, name=name,
                 label=label, limit=a.limit, best=a.best, video=not a.no_video, log=lambda m: print(f"{PREFIX} {m}", flush=True),
                 psi_max=a.psi_max, join_short=a.join_short, rec13a=rec13a, floor=a.floor)
    if a.dry_run:
        std = sd.stamp("dry", {"scene": "synthetic room of M4's fixture planes, ray-cast", "gated": False},
                       controls=["mirrored poses"])
    else:
        fan = paths["128"][2]
        values = dict(sd.PROFILES["build"]["values"], path="straight_long", width=int(fan.width),
                      psi_deg=[float(nt.SETTINGS["psi_anchor_deg"][0]), float(min(a.psi_max, nt.SETTINGS["psi_anchor_deg"][1]))],
                      lat_max_deg=float(nt.SETTINGS["lat_max_deg"]), confirm_rule=a.rule, confirm_span_m=a.span)
        dep = {"path": "one primitive, R3's straight_long: the demo's path (#29, P-D5)"}
        if d3.is_map(a.rule):                                   # coverage first (#32, P-D8)
            values.update(gate_right=a.floor, confirm_span_m=0.0)
            dep.update({"confirm_rule": "coverage first (Luiz, P-D8): the neighbour filter in place of S2's confirmation",
                        "confirm_span_m": "no span: the neighbour filter",
                        "gate_right": "coverage first: Luiz's usable floor (P-D8); S2's gate reported only"})
        if a.join_short:
            values["frames"] = [11, 5]
            dep["frames"] = "two window sets: 11 frames over +/-0.5 m, and 5 over +/-0.2 m (#26's short windows; P-D8)"
        if a.psi_max < nt.SETTINGS["psi_anchor_deg"][1]:
            dep["psi_deg"] = "the ray set chosen from stage 12b's error map (#28, P-D6): the backward sectors dropped"
        if a.rule != sd.PROFILES["build"]["values"]["confirm_rule"]:
            dep.setdefault("confirm_rule", "chosen after stage 12b (#28, P-D6)")
        if abs(values["confirm_span_m"] - sd.CONFIRM_SPAN_M) > 1e-9:
            dep.setdefault("confirm_span_m", "chosen after stage 12b (#28, P-D6)")
        std = sd.stamp("build", values, departures=dep, controls=["mirrored poses"])
    rec.update(std)
    rec.update({"sources": src, "computed_by": git})
    print(f"{PREFIX} {std['standard']} record: {std['profile']}, problems: {std['check_record'] or 'none'}")
    for c in rec["checks"]:
        ok = c.get("ok")
        head = "control only" if c["name"].startswith("the standard") else (
            "not evaluated" if ok is None else ("PASS" if ok else "FAIL"))
        cf_ = c.get("control_fails")
        print(f"{PREFIX} {head} {c['name']}" + (f"; control {c['control']}: "
              f"{'not evaluated' if cf_ is None else ('fails' if cf_ else 'DOES NOT FAIL')}" if "control" in c else ""))
    e = rec["end"]
    print(f"{PREFIX} end: model {e['model']['cells']:,} cells, {e['model']['right']}, coverage {e['model']['coverage']:.4f}; "
          f"all {e['all']['cells']:,}; gate {e['gate']['result']} (unverified {e['gate']['unverified']})")
    for w, f in rec["files"].items():
        print(f"{PREFIX} {w}: {f['path']}, {f['points']:,} points, {f['mb']} MB")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    base = "demo4" if d3.is_map(a.rule) else "demo3"           # D4 (#32): the coverage-first map
    rec["demo"] = base.upper()
    (Path(a.out) / (f"{base}_dry.json" if a.dry_run else f"{base}.json")).write_text(json.dumps(rec, indent=1, default=float))
    bad = [c for c in rec["checks"] if c.get("ok") is False or c.get("control_fails") is False]
    return 1 if bad or std["check_record"] else 0


def stage13a(a) -> int:
    """#31: coverage first. Anchor sources x ray sets x truth-free filters, from stage 11's caches; the frontier."""
    _torch_threads()
    import time as _t
    from . import coverage as cv, standard as sd
    t0 = _t.perf_counter()
    git = {"deep-motion": _git(REPO)}                           # read before the run
    paths, _, _, src = _stage11_inputs(a)
    rec_demo = json.loads((REPO / "docs/runs/dm-issue-29" / ("demo3_dry.json" if a.dry_run else "demo3.json")).read_text())
    out = cv.run(paths, cache=a.cache, rec_demo=rec_demo, log=lambda m: print(f"{PREFIX} {m}", flush=True))
    controls = ["mirrored poses"]
    if a.dry_run:
        std = sd.stamp("dry", {"scene": "synthetic room of M4's fixture planes, ray-cast", "gated": False}, controls=controls)
    else:
        values = dict(sd.PROFILES["ablation"]["values"], psi_deg=[list(v) for v in cv.RAYS.values()],
                      confirm_rule=list(cv.FILTERS), confirm_span_m=0.0, gate_right=list(cv.FLOORS))
        std = sd.stamp("ablation", values, departures={
            "psi_deg": "two ray sets under study (#31)",
            "confirm_rule": "coverage first (Luiz, 2026-11-21, #31): truth-free filters in place of S2's confirmation",
            "confirm_span_m": "no span: a second reference anywhere (the 2 cm filter)",
            "gate_right": "coverage first: the frontier at 98, 97 and 95 % right, S2's gate reported only"}, controls=controls)
    out.update(std)
    out.update({"command": "stage13a", "sources_runs": src, "computed_by": git,
                "seconds_total": round(_t.perf_counter() - t0, 1)})
    print(f"{PREFIX} {std['standard']} record: {std['profile']}, problems: {std['check_record'] or 'none'}")
    for c in out["checks"]:
        ok = c.get("ok")
        head = "control only" if c["check"].startswith("the standard") else (
            "not evaluated" if ok is None else ("PASS" if ok else "FAIL"))
        cf_ = c.get("control_fails")
        print(f"{PREFIX} {head} {c['check']}" + (f"; control {c['control']}: "
              f"{'not evaluated' if cf_ is None else ('fails' if cf_ else 'DOES NOT FAIL')}" if "control" in c else ""))
    lines = cv.table(out)
    for line in lines:
        print(line)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        name = "stage13a_dry" if a.dry_run else "stage13a"
        (Path(a.out) / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        (Path(a.out) / f"{name}_table.md").write_text("\n".join(lines) + "\n")
    bad = [c for c in out["checks"] if c.get("ok") is False or c.get("control_fails") is False]
    return 1 if bad or std["check_record"] else 0


def stage14_fixtures(a) -> int:
    """#34, G1: M4's fixtures along an arc, N's per-frame path, M4's gates and two controls."""
    _torch_threads()
    import time as _t
    from . import arcfixtures as af, standard as sd
    t0 = _t.perf_counter()
    git = {"deep-motion": _git(REPO)}                           # read before the run
    res = af.run_all(log=lambda m: print(f"{PREFIX} {m}", flush=True), names=a.only)
    res.update(sd.stamp("fixture", {"scene": "M4's fixtures (deep/nfixtures.py)", "step_m": 0.10, "frames_each_side": 5,
                                    "gates": "M4's fixture gates"},
                        departures={"scene": "M4's fixtures along an arc of radius 1.0 m (deep/arcfixtures.py, #34)",
                                    "gates": "M4's fixture gates without the radial wall (#34)"},
                        controls=["steps 20 % long", "scene scaled 1.5x", "poses of a radius 20 % too large"]))
    res["computed_by"], res["seconds_total"] = git, round(_t.perf_counter() - t0, 1)
    import json as _j
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "g1_fixtures.json").write_text(_j.dumps(res, indent=1, default=float))
    for g in res["gates"]:
        print(f"{PREFIX} G1 {'PASS' if g['ok'] else 'FAIL'} {g['fixture']} {g['label']}: {g}")
    for c in res["controls"]:
        print(f"{PREFIX} G1 control {'fails' if c['fails'] else 'DOES NOT FAIL'}: {c['control']}")
    print(f"{PREFIX} {res['standard']} record: {res['profile']}, problems: {res['check_record'] or 'none'}")
    return 0 if res["ok"] and res["controls_fail"] and not res["check_record"] else 1


def stage14(a) -> int:
    """#34: G0 (N's per-frame path forced on the straight path) and G2 (N on the turn, C3's form), with C0 on the turn."""
    _torch_threads()
    import time as _t
    from types import SimpleNamespace
    from . import arcs, assess as asx, e2e, oracle as orc, standard as sd
    t0 = _t.perf_counter()
    git = {"deep-motion": _git(REPO)}                           # read before the run
    log = lambda m: print(f"{PREFIX} {m}", flush=True)
    out = {"command": "stage14", "checks": []}
    if not a.skip_g0:
        paths, _, _, src = _stage11_inputs(SimpleNamespace(dry_run=a.dry_run, cache=a.cache, r128=a.r128, r1024=a.r1024, m4=[]))
        out["g0"] = arcs.g0(paths, a.cache, log=log)
        out["checks"].append({k: v for k, v in out["g0"].items() if k in ("check", "ok", "control", "control_fails", "control_value")})
        out["sources_straight"] = src
    if a.dry_run:
        pt = arcs.synthetic_arc_room(cache=Path(a.turn_cache) / "arc_room.npz", log=log)
        rec_assess, rec11 = None, None
        out["sources_turn"] = {"dry_run": "#26's synthetic room along an arc of radius 1.0 m (deep/arcs.py), Chat sandbox, not gated"}
    else:
        _, L = orc.load(a.turn128)
        refs = asx.references(len(L))
        looks, truth, fan, _ = e2e.render_path(a.turn128, refs=refs)
        pt = {"128": (looks, truth, fan, refs, asx.render_vis(a.turn128, fan))}
        l2, _, _, _ = e2e.render_path(a.turn1024, refs=refs)
        pt["1024"] = (l2, truth, fan, refs, None)
        rec_assess = json.loads((REPO / "docs/runs/dm-issue-22/assess.json").read_text())
        rec11 = json.loads((REPO / "docs/runs/dm-issue-26/stage11.json").read_text())
        out["sources_turn"] = {"turn128": a.turn128, "turn1024": a.turn1024}
    out["g2"] = arcs.g2(pt, a.turn_cache, assess_record=rec_assess, rec11=rec11, log=log)
    out["checks"] += out["g2"]["checks"]
    if a.dry_run:
        std = sd.stamp("dry", {"scene": "synthetic room of M4's fixture planes, ray-cast", "gated": False},
                       controls=["mirrored poses"])
    else:
        std = sd.stamp("ablation", dict(sd.PROFILES["ablation"]["values"], confirm_span_m=0.0, confirm_rule="none"),
                       departures={"confirm_rule": "C3's form: every cell, no confirmation (#34)",
                                   "confirm_span_m": "no confirmation (#34)"}, controls=["mirrored poses"])
    out.update(std)
    out.update({"computed_by": git, "seconds_total": round(_t.perf_counter() - t0, 1)})
    print(f"{PREFIX} {std['standard']} record: {std['profile']}, problems: {std['check_record'] or 'none'}")
    for c in out["checks"]:
        ok, cf_ = c.get("ok"), c.get("control_fails")
        print(f"{PREFIX} {'not evaluated' if ok is None else ('PASS' if ok else 'FAIL')} {c['check']}; control {c.get('control')}: "
              f"{'not evaluated' if cf_ is None else ('fails' if cf_ else 'DOES NOT FAIL')} ({c.get('control_value')})")
    if "g0" in out:
        g = out["g0"]
        print(f"{PREFIX} G0: {g['anchors']:,} anchors, same decision {g['same_decision']}, max rel zeta {g['max_rel_zeta']:.3g}; "
              f"line {g['line']}; per-frame {g['per_frame']}")
    for name, r in out["g2"]["rows"].items():
        print(f"{PREFIX} turn {name}: {r['cells']:,} cells, right {r['right']}, coverage {r['coverage']:.4f}")
    print(f"{PREFIX} turn N by sector: " + "; ".join(f"{k} {p['cells']:,} at {p['right']}" for k, p in out["g2"]["n_by_sector"]["parts"].items()))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / ("stage14_dry.json" if a.dry_run else "stage14.json")).write_text(json.dumps(out, indent=1, default=float))
    bad = [c for c in out["checks"] if c.get("ok") is False or c.get("control_fails") is False]
    return 1 if bad or std["check_record"] else 0


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
    p = sub.add_parser("demo2-ply", help="#25: demo D2's final maps as coloured PLY point clouds (perfect, real)")
    p.add_argument("--straight", default=None)
    p.add_argument("--turn", default=None)
    p.add_argument("--r0", action="store_true", help="the dry run: R0's calibration room (grey)")
    p.add_argument("--ply-dir", required=True, help="where the PLY files go (keep it under runs/, not committed)")
    p.add_argument("--out", required=True, help="the record's folder")
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
    p = sub.add_parser("stage11", help="#26: the ablation on the straight path, C0/C2/C3 and the decomposition")
    p.add_argument("--r128", default=None, help="R3's straight_long run (128 spp): the truth and B128")
    p.add_argument("--r1024", default=None, help="the same path re-rendered at 1024 spp")
    p.add_argument("--m4", nargs="*", default=[], help="LEVEL=RESULT_DIR,RUN_DIR, e.g. 1024=runs/m4-dolly-1024,runs/dolly-1024")
    p.add_argument("--cache", default=None, help="per-reference outputs (resumable; under runs/)")
    p.add_argument("--rows", nargs="*", default=None, help="only these rows (B128 is always computed)")
    p.add_argument("--dry-run", action="store_true", help="the synthetic room (Chat sandbox, not gated)")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage11b", help="#26: stage 11's rows by sector, range, depth edge, texture and SNR, the "
                                         "texture-limited ceilings, N1024's rejections by class (from stage11's cache)")
    p.add_argument("--r128", default=None)
    p.add_argument("--r1024", default=None)
    p.add_argument("--m4", nargs="*", default=[])
    p.add_argument("--cache", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage11c", help="#26: N1024's depth-edge variants (off; 10 %% of zeta) and the cross-tabulations "
                                         "range x SNR and sector x SNR (from stage11's cache, one new N run)")
    p.add_argument("--r128", default=None)
    p.add_argument("--r1024", default=None)
    p.add_argument("--m4", nargs="*", default=[])
    p.add_argument("--cache", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("standard-selftest", help="#27: S1's gate and check_record, and stage 12's span logic (controls)")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage12", help="#28: the confirmation distance under S1 (from stage 11's and 11c's caches)")
    p.add_argument("--r128", default=None)
    p.add_argument("--r1024", default=None)
    p.add_argument("--m4", nargs="*", default=[])
    p.add_argument("--cache", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage12b", help="#28, P-D5: two confirmation rules at spans 0-1.0 m, under S2 (stage 11's caches)")
    p.add_argument("--r128", default=None)
    p.add_argument("--r1024", default=None)
    p.add_argument("--m4", nargs="*", default=[])
    p.add_argument("--cache", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("demo3", help="#29: demo D3, N1024 along the Classroom's straight path as S2's model grows")
    p.add_argument("--r128", default=None, help="R3's straight_long run (128 spp): the truth and the judge")
    p.add_argument("--r1024", default=None, help="the same path at 1024 spp: N1024's images and the colours")
    p.add_argument("--cache", default=None, help="stage 11's cache (N1024 and its mirrored control)")
    p.add_argument("--rule", default="same cell", choices=["same cell", "within 2 cm", "neighbours >= 1", "neighbours >= 3"],
                   help="S2's confirmation, or (coverage first, #32) the neighbour filter")
    p.add_argument("--join-short", action="store_true", help="join N on +/-0.2 m windows (stage 11's cached row) to N1024")
    p.add_argument("--floor", type=float, default=0.97, help="coverage first: the usable floor (cells right)")
    p.add_argument("--span", type=float, default=0.5, help="S2's confirmation distance (m)")
    p.add_argument("--psi-max", type=float, default=179.0, help="the ray set: anchors at psi 1..psi_max deg from the heading")
    p.add_argument("--ply-dir", required=True, help="the PLY files and the video (under runs/, not committed)")
    p.add_argument("--out", required=True, help="the record's folder")
    p.add_argument("--limit", type=int, default=None, help="the first n looks only (a preview; the checks need all)")
    p.add_argument("--best", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="a synthetic room (Chat sandbox, not gated)")
    p.add_argument("--dry-looks", type=int, default=161)
    p.add_argument("--dry-y0", type=float, default=-0.8)
    p = sub.add_parser("stage13a", help="#31: coverage first, sources x ray sets x truth-free filters (stage 11's caches)")
    p.add_argument("--r128", default=None)
    p.add_argument("--r1024", default=None)
    p.add_argument("--m4", nargs="*", default=[])
    p.add_argument("--cache", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stage14-fixtures", help="#34, G1: M4's fixtures along an arc, N's per-frame path (Chat sandbox)")
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", default=None, help="fixture names (a preview; G1 is the full set)")
    p = sub.add_parser("stage14", help="#34: G0 on the straight path, G2 and C0 on the turn (N's per-frame path)")
    p.add_argument("--r128", default=None, help="R3's straight_long (128 spp)")
    p.add_argument("--r1024", default=None, help="R3's straight_long (1024 spp)")
    p.add_argument("--turn128", default=None, help="R3's turn90 (128 spp): the truth and B128")
    p.add_argument("--turn1024", default=None, help="R3's turn90 at 1024 spp")
    p.add_argument("--cache", default=None, help="stage 11's cache (the straight path's rows)")
    p.add_argument("--turn-cache", required=True, help="the turn's per-reference rows (under runs/)")
    p.add_argument("--skip-g0", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
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
    return {"stage11": stage11, "stage11b": stage11b, "stage11c": stage11c, "stage12": stage12, "stage12b": stage12b, "demo3": demo3, "stage13a": stage13a, "stage14-fixtures": stage14_fixtures, "stage14": stage14, "standard-selftest": standard_selftest, "stage11-selftest": stage11_selftest, "stage11-fixtures": stage11_fixtures, "demo2-ply": demo2_ply, "demo2": demo2, "assess": assess, "stage8": stage8, "stage7a": stage7a, "note-figs": note_figs, "moebius": moebius, "compare": compare, "demo": demo, "stage5b2-render": stage5b2_render, "stage5b2": stage5b2, "stage5a-table": stage5a_table, "selftest": selftest, "figures": figures, "stage4-render": stage4_render, "stage5a-render": stage5a_render,
            "stage5a": stage5a}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
