"""Stage 14 (#34), G1: M4's fixtures along an arc, and N's per-frame path run on them with M4's fixture gates.

C1's scenes (deep/nfixtures.py: M4's planes, textures, labels and foreign geometry; the 720-wide panorama, M4's 4 x 4
box filter and noise 0.005), with the reference where C1 has it (the origin, heading +Y, M4's R_EYE), so the anchors
and the truth are C1's. The frames leave the line: they lie on an arc of radius RADIUS (the Classroom turn's 1.0 m)
through the reference, tangent to the heading and turning left, the head locked to the tangent, every 10 cm of arc,
five each side (N's window). Off the line, N takes its per-frame path (deep/ntrace.py, stage 14).

Gates (declared on #34): C1's, median <= 0.1 px and p90 <= 0.25 px with >= 100 accepted on front, tilted, corridor's
parallel and front_far; the periodic wall also <= 1 % of accepted over 1 px. C1's radial wall is left out: its texture
is constant along a line's traces, not an arc's. Controls on the front wall, each must fail its gate: N given the poses
of an arc of radius 1.2 x RADIUS (the images stay those of RADIUS); every frame from the scene scaled 1.5x about the
reference's eye.
"""
from __future__ import annotations

import time

import numpy as np

from motion import multibaseline_fixtures as mbf

from . import nfixtures as nf
from . import ntrace as nt

RADIUS = 1.0
WRONG_RADIUS = 1.2 * RADIUS


def arc_pose(k, radius=RADIUS, step=nf.STEP):
    """Frame k's pose on the arc: arc length k step from the reference at the origin, heading +Y, turning left
    (towards -X); the head rotated about world Z with the tangent."""
    th = k * step / radius
    c = np.array([-radius + radius * np.cos(th), radius * np.sin(th), 0.0])
    cz, sz = np.cos(th), np.sin(th)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return c, Rz @ nf.R_EYE


class ArcScene(nf.Scene):
    def arc_frame(self, k, geometry=False):
        key = ("arc", k, geometry)
        if key not in self.cache:
            pl = self.far if geometry else self.planes
            c, R = arc_pose(k)
            self.cache[key] = nf.render(pl, c, seed=self.seed + 500 + k, R=R)
        return self.cache[key]


def run_n_arc(sc: ArcScene, *, radius_given=RADIUS, foreign_geometry=False, psi=nf.FORWARD_PSI, dev=None):
    imgs = [sc.arc_frame(k, geometry=foreign_geometry) for k in nf.WINDOW]
    poses = [arc_pose(k, radius=radius_given) for k in nf.WINDOW]
    t0 = time.perf_counter()
    r = nt.match_reference(sc.frame(0), np.zeros(3), nf.R_EYE, imgs, poses, nf.W, mask=nf.psi_mask(*psi), dev=dev)
    tr = sc.truth(r["px"])
    err = (r["zeta"] - 1.0 / tr["range"]) * r["px_per_zeta"]
    acc = r["reason"] == 0
    out = {"fixture": sc.name, "path": "arc", "radius_m": RADIUS, "radius_given_m": radius_given, "frames": nf.WINDOW,
           "step_m": nf.STEP, "foreign_geometry": foreign_geometry, "general": r["general"], "psi_deg": list(psi),
           "seconds": round(time.perf_counter() - t0, 1), "n_zeta": r["n_zeta"],
           "reasons": {nt.REASONS[k]: int((r["reason"] == k).sum()) for k in range(len(nt.REASONS))},
           "all": nf._stats(err, acc, np.ones_like(acc)), "by_label": {},
           "median_frames_accepted": float(np.median(r["n_frames"][acc])) if acc.any() else None}
    for L in sorted(set(tr["label"])):
        out["by_label"][L] = nf._stats(err, acc, tr["label"] == L)
    if sc.name == "corridor":
        p = np.degrees(r["psi"])
        out["parallel_by_psi"] = {"1-44": nf._stats(err, acc, (tr["label"] == "parallel") & (p <= 44)),
                                  "44-136": nf._stats(err, acc, (tr["label"] == "parallel") & (p > 44) & (p < 136)),
                                  "136-179": nf._stats(err, acc, (tr["label"] == "parallel") & (p >= 136))}
    if sc.name == "occlusion":
        out["edge"], out["interior"] = nf._stats(err, acc, tr["edge"]), nf._stats(err, acc, ~tr["edge"])
    return out, r, err


def run_all(log=print, names=None) -> dict:
    """G1 as declared on #34."""
    plan = [("front", nf.FORWARD_PSI), ("tilted", nf.FORWARD_PSI), ("corridor", (1.0, 179.0)),
            ("periodic", nf.FORWARD_PSI), ("occlusion", nf.FORWARD_PSI)]
    if names:
        plan = [p for p in plan if p[0] in names]
    gated = {"front": ("front",), "tilted": ("tilted",), "corridor": ("parallel", "front_far"), "periodic": ("periodic",)}
    scenes, cases, gates, ctrl = {}, [], [], []
    for name, psi in plan:
        scenes[name] = ArcScene(name)
        c, _, _ = run_n_arc(scenes[name], psi=psi)
        cases.append(c)
        for L in gated.get(name, ()):
            st = c["by_label"][L]
            gates.append({"fixture": name, "label": L, "ok": nf.gate_ok(st, 0.01 if name == "periodic" else None),
                          **{k: st[k] for k in ("accepted", "accepted_fraction", "median_px", "p90_px", "over_1px_of_accepted")}})
        log(f"[G1] {name} ({c['seconds']} s, {c['all']['anchors']} anchors, per-frame path {c['general']}): " + "; ".join(
            f"{L} acc {v['accepted_fraction']:.3f} med {nf._r(v['median_px'])} p90 {nf._r(v['p90_px'])} "
            f">1px {nf._r(v['over_1px_of_accepted'])}" for L, v in c["by_label"].items() if v["anchors"]))
    if "front" in scenes:
        for lab, kw in ((f"the poses of an arc of radius {WRONG_RADIUS:g} m (images of {RADIUS:g} m)", {"radius_given": WRONG_RADIUS}),
                        (f"every frame from the scene scaled {mbf.FOREIGN_GEOMETRY}x about the reference's eye",
                         {"foreign_geometry": True})):
            c, _, _ = run_n_arc(scenes["front"], **kw)
            ctrl.append({"control": lab, "fails": not nf.gate_ok(c["by_label"]["front"]), "stats": c["by_label"]["front"]})
            log(f"[G1] control {'fails' if ctrl[-1]['fails'] else 'DOES NOT FAIL'}: {lab}: {c['by_label']['front']}")
    return {"gates": gates, "controls": ctrl, "cases": cases, "ok": all(g["ok"] for g in gates),
            "controls_fail": all(x["fails"] for x in ctrl)}


def selftest():
    """The arc's poses: on the circle, tangent to the head. (name, ok, value, control, control fails, value)."""
    ks = np.arange(-5, 6)
    err_r, err_t = 0.0, 0.0
    for k in ks:
        c, R = arc_pose(int(k))
        err_r = max(err_r, abs(np.linalg.norm(c - np.array([-RADIUS, 0, 0])) - RADIUS))
        c2, _ = arc_pose(float(k) + 1e-6)
        tang = (c2 - c) / np.linalg.norm(c2 - c)
        err_t = max(err_t, float(np.linalg.norm(tang - (-R[:, 2]))))
    c5, _ = arc_pose(5)
    chord = float(np.linalg.norm(c5))
    c0, R0 = arc_pose(0)
    ok = err_r < 1e-12 and err_t < 1e-5 and np.allclose(c0, 0) and np.allclose(R0, nf.R_EYE)
    cw, _ = arc_pose(5, radius=WRONG_RADIUS)
    return [("A1: the arc's frames lie on the circle, the head on the tangent, the reference at C1's pose", bool(ok),
             {"radius_err": err_r, "tangent_err": err_t, "chord_5_m": chord},
             f"frame 5 on radius {WRONG_RADIUS:g} m is not frame 5 on {RADIUS:g} m",
             bool(np.linalg.norm(cw - c5) > 1e-3), float(np.linalg.norm(cw - c5)))]
