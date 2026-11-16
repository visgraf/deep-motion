"""M6a T2 (#22): an off-centre epipole with no rotation (the crab dolly), and M6's reduction to M4.

T2.0 (`repro`): on M4's own calibration-room run (`runs/m3-calib-1024`, Code's 1024-spp records), M6's matcher with
the recorded poses must give M4's frozen matcher's decision on every anchor, and its zeta within 1e-9 (relative) on
every accepted anchor, at the given reference frames (both5). Control (must fail): the same with every frame's
translation turned 1 deg about r's +Y (the epipoles misplaced by 1 deg): > 1 % of decisions must differ.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import analysis, flow_depth, matcher as m3, multibaseline as mb, m6_matcher as m6

BOTH5 = [o for o in range(-5, 6) if o]


def _lum(o):
    return m3.luminance(o.rgb)


def _turn_y(T, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return R @ T


def repro(run_dir, refs, out=None, git=None, log=print) -> dict:
    run = analysis.load_run(run_dir)
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    per = []
    for r in refs:
        ks = [r + o for o in BOTH5]
        ob = {k: flow_depth.load_observation(run, es[k]) for k in [r] + ks}
        h = ob[r].R @ [0, 0, -1.0]
        ts = [float((ob[k].c - ob[r].c) @ h) for k in ks]
        imgs = [_lum(ob[k]) for k in ks]
        t0 = time.perf_counter()
        a = mb.match_reference(_lum(ob[r]), imgs, ts, ob[r].model)
        t4 = time.perf_counter() - t0
        poses = m6.relative_poses(ob[r].c, ob[r].R, [(ob[k].c, ob[k].R) for k in ks])
        t0 = time.perf_counter()
        b = m6.match_reference(_lum(ob[r]), imgs, poses, ob[r].model)
        t6 = time.perf_counter() - t0
        c = m6.match_reference(_lum(ob[r]), imgs, [(A, _turn_y(T, 1.0)) for A, T in poses], ob[r].model)
        acc = (a["reason"] == 0) & (b["reason"] == 0)
        rel = np.abs(a["zeta"] - b["zeta"])[acc] / a["zeta"][acc]
        fin = np.isfinite(a["zeta"]) & np.isfinite(b["zeta"])
        d = {"r": r, "anchors": int(len(a["reason"])), "accepted_m4": float(np.mean(a["reason"] == 0)),
             "accepted_m6": float(np.mean(b["reason"] == 0)), "decisions_differ": int(np.sum(a["reason"] != b["reason"])),
             "max_rel_dzeta_accepted": float(rel.max()) if rel.size else None,
             "max_abs_dzeta_any_finite": float(np.max(np.abs(a["zeta"] - b["zeta"])[fin])),
             "coarse_ncc_max_diff": float(np.nanmax(np.abs(a["ncc_coarse"] - b["ncc_coarse"]))),
             "control_decisions_differ_fraction": float(np.mean(a["reason"] != c["reason"])),
             "control_accepted": float(np.mean(c["reason"] == 0)),
             "seconds_m4": round(t4, 1), "seconds_m6": round(t6, 1)}
        d["ok"] = d["decisions_differ"] == 0 and (d["max_rel_dzeta_accepted"] or 0.0) <= 1e-9
        d["control_fails"] = d["control_decisions_differ_fraction"] > 0.01
        per.append(d)
        log(f"[m6-t2] T2.0 r {r}: " + json.dumps(d))
    res = {"schema": "foveal-motion-m6-t2-repro-v1", "run": run["_dir"], "run_git": run["git"], "refs": list(refs),
           "window": "both5", "computed_by": git, "per_ref": per,
           "check": {"id": "T2.0", "ok": all(p["ok"] for p in per), "control_fails": all(p["control_fails"] for p in per)}}
    log(f"[m6-t2] T2.0 {'PASS' if res['check']['ok'] else 'FAIL'} (control {'fails' if res['check']['control_fails'] else 'DOES NOT FAIL'})")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "t2_repro.json").write_text(json.dumps(res, indent=1, default=float))
    return res
