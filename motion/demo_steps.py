"""D1's steps after rendering: estimate (the frozen M4), fuse, check, top view, report (motion/demo.py's CLI).

Numbers these steps write are MEASURED on the machine that runs them (env_demo-estimate.json records it); the
video and the top view are DERIVED display renders.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import analysis, tour
from .demo import PREFIX, WINDOW, _dolly, _write, load_tour, phase_index, phases_of

VOXEL_M = 0.02                 # #17: proposed
FUSION_TOL_REL = 0.05          # D1.3: #5's calibration-gate median (flow_depth_eval.GATES), on fused voxels
CONTROL_YAW_DEG = 5.0          # D1.3's control: leg B's poses yawed this much about the head (a heading error);
                               # #17 proposed a 0.10 m shift, but at the sandbox's 2-6 m ranges that is 2-5 %, under
                               # the tolerance (Chat sandbox, #18), so it could not be relied on to fail


# ----------------------------------------------------------------------------- estimate

def estimate(a) -> int:
    from . import m4_pipeline as mp
    d = _dolly()
    t = load_tour(a.out)
    out = Path(a.out)
    git = {"foveal-motion": d._git(d.REPO), "fov-render": d._git(d.FOV_RENDER)}
    d.m4_env(out / "m4", "demo-estimate")
    names = a.phases or phases_of(t)
    for name in names:
        if name not in t["phases"]:
            print(f"{PREFIX} refused: {name} is not rendered", file=sys.stderr)
            return 2
        run = analysis.load_run(out / name)
        ks = sorted(e["k"] for e in run["looks"]["ahead"])
        dd = out / "m4" / name
        if not (dd / WINDOW / "freeze.json").exists():
            mp.import_freeze(run, dd, WINDOW, Path(a.frozen), git=git)
        have = set(json.loads((dd / WINDOW / "config.json").read_text())["refs"])
        refs = [r for r in mp.references(ks, WINDOW) if (r - a.offset) % a.every == 0 and r not in have]
        print(f"{PREFIX} {name}: {len(refs)} references to estimate: {refs}", flush=True)
        if refs:
            mp.estimate(run, WINDOW, refs, dd, git=git, log=lambda m: print(m, flush=True))
    return 0


# ----------------------------------------------------------------------------- fuse

def _interval_scale(dd: Path) -> float:
    return float(json.loads((dd / WINDOW / "freeze.json").read_text())["frozen_settings"]["interval_scale"])


def anchors(t: dict) -> dict:
    """Every accepted anchor of every estimated reference, in world coordinates, with its truth and its time.

    Range along the anchor's ray: 1 / (zeta cos e) (zeta is axial inverse depth, #9); sigma from sigma_zeta times
    the frozen interval scale; P = c + range d, d the sample's world direction. hit: the anchor's ray has a truth
    surface (False where it leaves the scene, e.g. through the Classroom's windows: P_true and rel are NaN there,
    #19); every anchor is an estimate and stays in the cloud, only hits are compared with truth. t_avail: the video frame at which
    the window's last frame (r + 5) has been seen, when both5's estimate for r becomes available (M-D15, M-D17)."""
    from . import m4_pipeline as mp
    out = Path(t["_dir"])
    fo = tour.frame_of(t["timeline"])
    cols = {k: [] for k in ("P", "P_true", "w", "rgb", "rel", "t_avail", "t_ref", "leg", "phase", "r", "c", "flat", "range",
                            "hit")}
    for name in phases_of(t):
        dd = out / "m4" / name
        if not (dd / WINDOW / "config.json").exists():
            continue
        leg, p = phase_index(t, name)
        run = analysis.load_run(out / name)
        es = {e["k"]: e for e in run["looks"]["ahead"]}
        scale = _interval_scale(dd)
        for f in sorted((dd / WINDOW).glob("r*.npz")):
            z = dict(np.load(f))
            r = int(z["r"])
            rec, R, c, _ = analysis.load_look(run, es[r])
            tr = mp.truth(run, r, z)
            ok = (z["reason"] == 0) & (z["sample"] >= 0) & (z["zeta"] > 0)
            s = z["sample"][ok]
            dirs = rec.direction[s].astype(np.float64)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            ce = np.cos(z["e"][ok])
            rng = 1.0 / (z["zeta"][ok] * ce)
            sig = z["sigma_zeta"][ok] * scale / (z["zeta"][ok] ** 2 * ce)
            rt = tr["range"][ok]
            n = int(ok.sum())
            cols["P"].append(c + dirs * rng[:, None])
            cols["P_true"].append(c + dirs * rt[:, None])
            cols["w"].append(1.0 / np.maximum(sig, 1e-6) ** 2)
            cols["rgb"].append(np.asarray(rec.value)[s].astype(np.float64))
            cols["rel"].append(np.abs(rng - rt) / rt)
            last = r + max(mp.WINDOWS[WINDOW])
            cols["t_avail"].append(np.full(n, fo[(leg, p, last)]))
            cols["t_ref"].append(np.full(n, fo[(leg, p, r)]))
            cols["leg"].append(np.full(n, leg))
            cols["phase"].append(np.full(n, p))
            cols["r"].append(np.full(n, r))
            cols["c"].append(np.repeat(c[None], n, 0))
            cols["flat"].append(z["flat"][ok])
            cols["range"].append(rng)
            cols["hit"].append(tr["hit"][ok] & np.isfinite(rt))
    return {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in cols.items()}


def hits(A: dict) -> np.ndarray:
    """Anchors with a truth hit (as M4's evaluation); anchors.npz from before #19 has no `hit`: derive it."""
    if "hit" in A:
        return A["hit"].astype(bool)
    return np.isfinite(A["P_true"]).all(1) & np.isfinite(A["rel"])


def no_truth_counts(A: dict, names) -> dict:
    h = hits(A)
    return {"total": int((~h).sum()), "fraction": float((~h).mean()) if len(h) else 0.0,
            "per_leg": {nm: int((~h & (A["leg"] == l)).sum()) for l, nm in enumerate(names)}}


def d13(A: dict, names, yaw_deg: float = 0.0, moved_leg: int = 1) -> dict:
    """D1.3's measure: per leg, the median over fused voxels of |P_fused - P_true| / range, on the anchors with a
    truth hit only (#19: anchors whose rays leave the scene have no truth). yaw_deg rotates leg `moved_leg`'s
    estimates about each head (the control's heading error)."""
    h = hits(A)
    B = {k: v[h] for k, v in A.items() if hasattr(v, "__len__") and len(v) == len(h)}
    P = B["P"].copy()
    if yaw_deg:
        lb = B["leg"] == moved_leg
        ca, sa = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
        P[lb] = B["c"][lb] + (P[lb] - B["c"][lb]) @ Rz.T
    rt = np.linalg.norm(B["P_true"] - B["c"], axis=1)
    F = fuse(P, B["w"], B["rgb"], {"P_true": B["P_true"], "leg": B["leg"].astype(float), "rt": rt})
    rel = np.linalg.norm(F["P"] - F["P_true"], axis=1) / F["rt"]
    legv = np.round(F["leg"]).astype(int)              # voxels mixing legs go to the majority
    return {nm: float(np.median(rel[legv == l])) if np.any(legv == l) else None for l, nm in enumerate(names)}


def fuse(P, w, rgb, extra=None, voxel=VOXEL_M):
    """Voxel fusion: per occupied voxel the inverse-variance mean position, the mean colour, the weight, the count;
    extra: {name: (n, ...) array} averaged per voxel too (unweighted)."""
    if len(P) == 0:
        return {"P": np.zeros((0, 3)), "rgb": np.zeros((0, 3)), "w": np.zeros(0), "n": np.zeros(0, int), "inv": np.zeros(0, int)}
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    m = cnt.size
    W = np.bincount(inv, w, m)
    out = {"P": np.stack([np.bincount(inv, w * P[:, i], m) for i in range(3)], 1) / W[:, None],
           "rgb": np.stack([np.bincount(inv, rgb[:, i], m) for i in range(3)], 1) / cnt[:, None],
           "w": W, "n": cnt, "inv": inv}
    for k, v in (extra or {}).items():
        v = np.asarray(v, float)
        out[k] = (np.stack([np.bincount(inv, v[:, i], m) for i in range(v.shape[1])], 1) if v.ndim == 2
                  else np.bincount(inv, v, m)) / (cnt[:, None] if v.ndim == 2 else cnt)
    return out


def fuse_cmd(a) -> int:
    from .pointcloud import exposure_of, srgb_u8, write_ply
    t = load_tour(a.out)
    out = Path(a.out) / "fused"
    A = anchors(t)
    if len(A["P"]) == 0:
        print(f"{PREFIX} refused: no estimates under {Path(a.out) / 'm4'}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "anchors.npz", **A)
    names = [lg["name"] for lg in t["spec"]["legs"]]
    h = hits(A)
    F = fuse(A["P"], A["w"], A["rgb"])
    Fh = fuse(A["P"][h], A["w"][h], A["rgb"][h], {"P_true": A["P_true"][h]})
    ex = exposure_of(A["rgb"])
    err = np.linalg.norm(Fh["P"] - Fh["P_true"], axis=1)
    for legs, nm in ((None, "all"),) + tuple(((l,), t["spec"]["legs"][l]["name"]) for l in range(len(t["spec"]["legs"]))):
        sel = np.ones(len(A["P"]), bool) if legs is None else np.isin(A["leg"], legs)
        G = fuse(A["P"][sel], A["w"][sel], A["rgb"][sel])
        write_ply(out / f"fused_{nm}.ply", G["P"], srgb_u8(G["rgb"], ex),
                  comments=[f"D1 {t['spec']['name']} fused ({nm}), voxel {VOXEL_M} m, frozen M4 {WINDOW}"])
    summ = {"schema": "foveal-motion-d1-fused-v1", "anchors": int(len(A["P"])), "voxels": int(len(F["P"])),
            "voxel_m": VOXEL_M, "exposure": ex,
            "errors_on": "anchors with a truth hit (as M4's evaluation, #19); the cloud and the PLY keep every anchor",
            "anchors_no_truth": no_truth_counts(A, names), "voxels_with_truth": int(len(Fh["P"])),
            "median_voxel_error_m": float(np.median(err)), "p90_voxel_error_m": float(np.quantile(err, 0.9)),
            "median_anchor_rel": float(np.median(A["rel"][h])), "p90_anchor_rel": float(np.quantile(A["rel"][h], 0.9)),
            "references": int(len(np.unique(np.stack([A["leg"], A["phase"], A["r"]], 1), axis=0)))}
    _write(out / "fused.json", summ)
    print(f"{PREFIX} fused {summ['anchors']} anchors from {summ['references']} references into {summ['voxels']} "
          f"voxels; {summ['anchors_no_truth']['total']} anchors without truth left out of the errors; voxel error "
          f"median {summ['median_voxel_error_m'] * 100:.2f} cm, p90 {summ['p90_voxel_error_m'] * 100:.2f} cm")
    return 0


# ----------------------------------------------------------------------------- checks

def _straddle_run(t, out: Path, legA: int, across: bool) -> Path:
    """A run record whose looks are phase 0 of leg `legA` (the last 11, or for `across` the last 6) followed, for
    `across`, by phase 0 of the next leg from its second look: a window centred on the joint."""
    s = t["spec"]
    a_name, b_name = tour.phase_name(s, legA, 0), tour.phase_name(s, legA + 1, 0)
    ra = analysis.load_run(Path(t["_dir"]) / a_name)
    rb = analysis.load_run(Path(t["_dir"]) / b_name)
    la = ra["looks"]["ahead"]
    picks = ([(a_name, e) for e in la[-6:]] + [(b_name, e) for e in rb["looks"]["ahead"][1:6]]) if across else \
            [(a_name, e) for e in la[-11:]]
    nm = "straddle" if across else "straight"
    d = out / "checks" / nm
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(ra, looks={"ahead": [dict(e, k=i, out_dir=os.path.relpath(Path(t["_dir"]) / src / e["out_dir"], d))
                                    for i, (src, e) in enumerate(picks)]},
               experiment=f"D1.2 {nm}: a window {'across the pivot' if across else 'inside leg ' + a_name}")
    rec.pop("_dir", None)
    _write(d / "run.json", rec)
    return d


def checks(a) -> int:
    from . import flow_depth, m4_pipeline as mp, multibaseline as mb
    t = load_tour(a.out)
    out = Path(a.out)
    rows = []
    # D1.1: the frozen estimator
    frozen = json.loads(Path(a.frozen).read_text())
    try:
        mp.check_estimator(frozen["estimator_sha256"])
        ok1 = mp._nos(mb.SETTINGS) == frozen["settings"]
    except SystemExit:
        ok1 = False
    bad = dict(frozen["estimator_sha256"])
    bad[next(iter(bad))] = "0" * 64
    try:
        mp.check_estimator(bad)
        c1a = False
    except SystemExit:
        c1a = True
    c1b = mp._nos(dict(mb.SETTINGS, ncc_min=mb.SETTINGS["ncc_min"] - 0.01)) != frozen["settings"]
    imported = all(json.loads((out / "m4" / n / WINDOW / "freeze.json").read_text())["estimator_sha256"] ==
                   frozen["estimator_sha256"] for n in phases_of(t) if (out / "m4" / n / WINDOW / "freeze.json").exists())
    rows.append({"id": "D1.1", "what": "the frozen estimator: file hashes and settings equal docs/runs/issue-10/frozen.json, "
                                       "and every phase's result imported that freeze",
                 "ok": bool(ok1 and imported), "control_fails": bool(c1a and c1b),
                 "control": "a recorded hash changed must be refused; settings with ncc_min - 0.01 must differ"})
    # D1.2: the pivot
    poses = lambda run: {e["k"]: (lambda o: (o.c, o.R))(flow_depth.load_observation(run, e)) for e in run["looks"]["ahead"]}
    ra = analysis.load_run(_straddle_run(t, out, 0, True))
    rs = analysis.load_run(_straddle_run(t, out, 0, False))
    pairs = lambda: [(min(k, 5), max(k, 5)) for k in [5 + o for o in mp.WINDOWS[WINDOW]]]
    al_across = flow_depth.alignment(poses(ra), pairs())
    al_straight = flow_depth.alignment(poses(rs), pairs())
    dd = out / "checks" / "straddle_m4"
    if not (dd / WINDOW / "freeze.json").exists():
        mp.import_freeze(ra, dd, WINDOW, Path(a.frozen))
    try:
        mp.estimate(ra, WINDOW, [5], dd, log=lambda m: None)
        refused = False
        why = "estimated"
    except SystemExit as e:
        refused = "alignment gate failed" in str(e)
        why = str(e)
    rows.append({"id": "D1.2", "what": "a both5 window across the pivot is refused by #5's alignment gate inside the "
                                       "frozen estimate()",
                 "ok": bool(refused and not al_across["ok"]), "control_fails": bool(al_straight["ok"]),
                 "control": "the same construction inside leg A must pass the gate (the refusal comes from the pivot)",
                 "detail": {"refusal": why, "across_max_rotation_rad": al_across["detail"]["max_rotation_rad"],
                            "straight_max_rotation_rad": al_straight["detail"]["max_rotation_rad"]}})
    # D1.3: fusion against truth, per leg, on anchors with a truth hit (#19); control: the last leg's poses yawed
    A = anchors(t)
    names = [lg["name"] for lg in t["spec"]["legs"]]
    res = {0.0: d13(A, names), CONTROL_YAW_DEG: d13(A, names, CONTROL_YAW_DEG, len(names) - 1)}
    ok3 = all(v is not None and v <= FUSION_TOL_REL for v in res[0.0].values())
    c3 = res[CONTROL_YAW_DEG].get(names[-1]) if len(names) > 1 else None
    h = hits(A)
    rows.append({"id": "D1.3", "what": f"fused voxels ({VOXEL_M} m) lie on the truth surface: per leg, median "
                                       f"|P_fused - P_true| / range <= {FUSION_TOL_REL} (#5's calibration-gate median), "
                                       f"over the anchors with a truth hit (as M4's evaluation, #19)",
                 "ok": bool(ok3), "control_fails": bool(c3 is not None and c3 > FUSION_TOL_REL),
                 "control": f"leg {names[-1]}'s poses yawed {CONTROL_YAW_DEG} deg about the head: its median must exceed the tolerance",
                 "detail": {"median_rel": res[0.0], "control_median_rel": res[CONTROL_YAW_DEG], "anchors": int(len(A["P"])),
                            "left_out_no_truth": no_truth_counts(A, names),
                            "median_anchor_rel": float(np.median(A["rel"][h])) if h.any() else None}})
    for r in rows:
        print(f"{PREFIX} {r['id']} {'PASS' if r['ok'] else 'FAIL'}  control {'fails' if r['control_fails'] else 'DOES NOT FAIL'}"
              f"  {r['what']}" + (f"  {json.dumps(r.get('detail'), default=float)}" if r.get("detail") else ""))
    _write(out / "checks.json", {"schema": "foveal-motion-d1-checks-v1", "rows": rows,
                                 "git": _dolly()._git(_dolly().REPO)})
    return 0 if all(r["ok"] and r["control_fails"] for r in rows) else 1


# ----------------------------------------------------------------------------- top view

def topview(a) -> int:
    d = _dolly()
    from fov_render import launch
    t = load_tour(a.out)
    out = Path(a.out) / "topview"
    if a.bounds:
        b = a.bounds
    else:                                   # the tour's legs, with a margin (the Classroom: give --bounds, #19)
        pts = np.array([lg[k] for lg in t["spec"]["legs"] for k in ("start_w", "end_w")])
        lo, hi = pts.min(0) - a.margin, pts.max(0) + a.margin
        b = [lo[0], lo[1], hi[0], hi[1]]
    exe = launch.find(a.launcher)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(d.REPO), str(d.FOV_RENDER)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    args = ["--blend", t["blend"], "--out", str(out.resolve()), "--bounds", *map(str, b), "--cut-z", str(a.cut_z),
            "--m-per-px", str(a.m_per_px), "--spp", str(a.spp), "--device", a.device]
    script = d.REPO / "motion" / "bl_topview.py"
    cmd = ([exe, "-b", "--python-exit-code", "1", "-P", str(script), "--", *args] if launch.is_blender(exe)
           else [exe, "-m", "motion.bl_topview", *args])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "topview.log", "w") as log:
        rc = subprocess.run(cmd, env=env, cwd=str(d.REPO), stdout=log, stderr=subprocess.STDOUT).returncode
    print("\n".join(l for l in (out / "topview.log").read_text().splitlines() if l.startswith("[topview]")))
    return rc


# ----------------------------------------------------------------------------- report (check 4, leg B)

def report(a) -> int:
    from . import m4_pipeline as mp
    t = load_tour(a.out)
    res = {}
    for name in a.phases or phases_of(t):
        dd = Path(a.out) / "m4" / name
        if (dd / WINDOW / "config.json").exists():
            res[name] = mp.evaluate(dd, WINDOW, a.m3_scale, "classroom", log=lambda m: print(m, flush=True))
    _write(Path(a.out) / "report.json", {"schema": "foveal-motion-d1-report-v1", "what":
                                         "#5's Classroom metrics per phase, REPORTED, not gated (M-D17)", "phases": res})
    return 0


# ----------------------------------------------------------------------------- CLI

def add_parsers(sub):
    p = sub.add_parser("demo-estimate")
    p.add_argument("--out", required=True)
    p.add_argument("--phases", nargs="*", default=None)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--frozen", default="docs/runs/issue-10/frozen.json")
    p = sub.add_parser("demo-fuse")
    p.add_argument("--out", required=True)
    p = sub.add_parser("demo-checks")
    p.add_argument("--out", required=True)
    p.add_argument("--frozen", default="docs/runs/issue-10/frozen.json")
    p = sub.add_parser("demo-topview")
    p.add_argument("--out", required=True)
    p.add_argument("--bounds", type=float, nargs=4, default=None)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--cut-z", type=float, default=2.2)
    p.add_argument("--m-per-px", type=float, default=0.005)
    p.add_argument("--spp", type=int, default=64)
    p.add_argument("--device", default="OPTIX")
    p.add_argument("--launcher", default="auto")
    p = sub.add_parser("demo-report")
    p.add_argument("--out", required=True)
    p.add_argument("--phases", nargs="*", default=None)
    p.add_argument("--m3-scale", type=float, default=1.625204104083527, help="M3's frozen scale (#7)")
    from . import demo_video
    demo_video.add_parser(sub)


def dispatch(a):
    from . import demo_video
    return {"demo-estimate": estimate, "demo-fuse": fuse_cmd, "demo-checks": checks, "demo-topview": topview,
            "demo-report": report, "demo-video": demo_video.video}[a.cmd](a)


def selftest() -> list[tuple[str, bool]]:
    """#19: anchors without truth (rays leaving the scene) must not reach D1.3 or the error colours as NaN."""
    rng = np.random.default_rng(0)
    n = 4000
    c = np.zeros((n, 3))
    leg = (np.arange(n) >= n // 2).astype(int)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Pt = c + 3.0 * d
    P = Pt * (1 + 0.01 * rng.normal(size=(n, 1)))
    hit = np.ones(n, bool)
    hit[: n // 10] = False                                    # 10 % of leg A without truth
    Pt[~hit] = np.nan
    rel = np.where(hit, np.abs(np.linalg.norm(P - c, axis=1) - 3.0) / 3.0, np.nan)
    A = {"P": P, "P_true": Pt, "c": c, "w": np.ones(n), "rgb": np.ones((n, 3)), "leg": leg, "rel": rel, "hit": hit}
    A_old = {k: v for k, v in A.items() if k != "hit"}         # anchors.npz from before #19
    m0, m5 = d13(A, ["A", "B"]), d13(A, ["A", "B"], 5.0, 1)
    from .demo_video import NO_TRUTH, error_colours
    F = fuse(P, A["w"], A["rgb"], {"relh": np.where(hit, rel, 0.0), "h": hit.astype(float)})
    col = error_colours(F)
    nohit_vox = F["h"] == 0
    return [
        ("D1.3 finite with anchors lacking truth", all(v is not None and np.isfinite(v) for v in m0.values())),
        ("D1.3 near 1 % on the synthetic cloud", all(v < 0.02 for v in m0.values())),
        ("D1.3 control (5 deg) exceeds 0.05", m5["B"] > 0.05),
        ("hits derived for anchors.npz without `hit`", bool(np.array_equal(hits(A_old), hit))),
        ("no-truth counted per leg", no_truth_counts(A, ["A", "B"])["per_leg"] == {"A": n // 10, "B": 0}),
        ("error colours: no-truth voxels grey", bool(nohit_vox.any()) and bool(np.all(col[nohit_vox] == NO_TRUTH))),
        ("error colours: truth voxels on the ramp", bool(np.all(np.any(col[~nohit_vox] != NO_TRUTH, axis=1)))),
        ("the old median is NaN (the #19 bug, reproduced)", bool(np.isnan(np.median(np.where(hit, rel, np.nan))))),
    ]
