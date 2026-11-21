"""Stage 11 (#26), C1: M4's fixtures rebuilt for the panorama, and N run on them with M4's fixture gates.

The scenes are M4's (motion/flow_fixtures.py: the same planes, textures and labels; motion/multibaseline_fixtures.py:
the foreign geometry), ray-cast through the 720-wide panorama's rays with M4's 4 x 4 box filter and noise 0.005. The
reference sits at the origin, heading +Y (M4's R_EYE); N's frames are 10 cm apart on +Y, five each side (N's window,
every 10th look of a 1 cm run).

Anchors: the forward fixtures (front, tilted, periodic, radial, occlusion) at psi 1-44 deg, M4's band, where the planes
are as M4 saw them; the corridor at N's whole range, psi 1-179 deg (its lateral walls are the panorama's new part).

Error in equivalent px, as M4's: (zeta - zeta_true) x px per unit zeta at the longest step / ts_scale.
Gates (declared on #26): median <= 0.1 px and p90 <= 0.25 px with >= 100 accepted on front, tilted, corridor's parallel
and front_far; the periodic wall also <= 1 % of accepted over 1 px; the radial wall (texture constant along the
traces) accepts <= 1 %. Controls on the front wall, each must fail its gate: steps 20 % too long; every frame from the
scene scaled 1.5x about the reference's eye.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from motion import flow_fixtures as fx
from motion import multibaseline_fixtures as mbf

from . import ntrace as nt

R_EYE = fx.R_EYE
W = 720
STEP = 0.10
WINDOW = [k for k in range(-5, 6) if k]
NOISE = fx.NOISE
SS = fx.SS
GATE = dict(fx.GATE)
FORWARD_PSI = (1.0, 44.0)


def render(planes, c, W=W, ss=SS, noise=NOISE, seed=0, R=None):
    """The panorama (H, W) of luminance a look at c (rotation R, default R_EYE) sees of the planes, box-filtered over
    ss x ss sub-pixel rays; nan where any sub-ray misses."""
    R = R_EYE if R is None else np.asarray(R, float)
    H = W // 2
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    o = (np.arange(ss) + 0.5) / ss - 0.5
    oi, oj = np.meshgrid(o, o, indexing="ij")
    out = np.empty((H, W))
    for r0 in range(0, H, 45):                                  # in row blocks, to bound memory
        vv, uu = v[r0:r0 + 45, :, None] + oi.reshape(-1), u[r0:r0 + 45, :, None] + oj.reshape(-1)
        lon = ((uu + 0.5) / W - 0.5) * 2 * np.pi
        lat = (0.5 - (vv + 0.5) / H) * np.pi
        d = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1) @ R.T
        _, _, val = fx._hit(planes, np.asarray(c, float), d)
        full = np.isfinite(val).all(-1)
        out[r0:r0 + 45] = np.where(full, np.where(full[..., None], val, 0.0).mean(-1), np.nan)
    if noise:
        out = out + np.random.default_rng(seed).normal(scale=noise, size=out.shape)
    return out


def edge_mask(L, radius=3):
    """Pixels within `radius` of a change of plane label (columns wrap)."""
    E = np.zeros_like(L, bool)
    H = L.shape[0]
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            S = np.full_like(L, -1)
            S[max(-di, 0):H + min(-di, 0)] = L[max(di, 0):H + min(di, 0)]
            S = np.roll(S, -dj, 1)
            E |= (S != L) & (S >= 0) & (L >= 0)
    return E


class Scene:
    def __init__(self, name, seed=0):
        self.name, self.seed = name, seed
        self.planes = fx.scene(name, seed)
        self.far = mbf.scaled(self.planes, mbf.FOREIGN_GEOMETRY)
        self.cache = {}

    def frame(self, k, geometry=False):
        key = (k, geometry)
        if key not in self.cache:
            pl = self.far if geometry else self.planes
            self.cache[key] = render(pl, np.array([0.0, k * STEP, 0.0]), seed=self.seed + 500 + k)
        return self.cache[key]

    def truth(self, px):
        H = W // 2
        rows, cols = px[:, 0], px[:, 1]
        lon = (cols + 0.5) / W * 2 * np.pi - np.pi
        lat = (0.5 - (rows + 0.5) / H) * np.pi
        u = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1) @ R_EYE.T
        rng, idx, _ = fx._hit(self.planes, np.zeros(3), u)
        v, uu = np.mgrid[0:H, 0:W]
        lonf = (uu + 0.5) / W * 2 * np.pi - np.pi
        latf = (0.5 - (v + 0.5) / H) * np.pi
        uf = np.stack([np.sin(lonf) * np.cos(latf), np.sin(latf), -np.cos(lonf) * np.cos(latf)], -1) @ R_EYE.T
        _, Lf, _ = fx._hit(self.planes, np.zeros(3), uf)
        labs = [p["label"] for p in self.planes]
        return {"range": rng, "label": np.array([labs[i] if i >= 0 else "none" for i in idx]),
                "edge": edge_mask(Lf)[rows, cols]}


def _stats(err, acc, sel):
    n = int(sel.sum())
    a = acc & sel
    ae = np.abs(err[a])
    return {"anchors": n, "accepted": int(a.sum()), "accepted_fraction": float(a.sum() / n) if n else None,
            "median_px": float(np.median(ae)) if ae.size else None, "p90_px": float(np.quantile(ae, 0.9)) if ae.size else None,
            "over_1px_of_accepted": float(np.mean(ae > 1)) if ae.size else None}


def gate_ok(st, max_over_1px=None):
    ok = (st["accepted"] >= GATE["min_accepted"] and st["median_px"] is not None and st["median_px"] <= GATE["median_px"]
          and st["p90_px"] <= GATE["p90_px"])
    return bool(ok and (max_over_1px is None or st["over_1px_of_accepted"] <= max_over_1px))


def psi_mask(lo_deg, hi_deg):
    H = W // 2
    v, u = np.mgrid[0:H, 0:W]
    lon = (u + 0.5) / W * 2 * np.pi - np.pi
    lat = (0.5 - (v + 0.5) / H) * np.pi
    d = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1) @ R_EYE.T
    psi = np.degrees(np.arccos(np.clip(d @ (-R_EYE[:, 2]), -1, 1)))
    return (psi >= lo_deg) & (psi <= hi_deg)


def run_n(sc: Scene, *, ts_scale=1.0, foreign_geometry=False, psi=FORWARD_PSI, dev=None):
    imgs = [sc.frame(k, geometry=foreign_geometry) for k in WINDOW]
    poses = [(np.array([0.0, k * STEP * ts_scale, 0.0]), R_EYE) for k in WINDOW]
    t0 = time.perf_counter()
    r = nt.match_reference(sc.frame(0), np.zeros(3), R_EYE, imgs, poses, W, mask=psi_mask(*psi), dev=dev)
    tr = sc.truth(r["px"])
    err = (r["zeta"] - 1.0 / tr["range"]) * r["px_per_zeta"] / ts_scale
    acc = r["reason"] == 0
    out = {"fixture": sc.name, "frames": WINDOW, "step_m": STEP, "ts_scale": ts_scale, "foreign_geometry": foreign_geometry,
           "psi_deg": list(psi), "seconds": round(time.perf_counter() - t0, 1), "n_zeta": r["n_zeta"],
           "reasons": {nt.REASONS[k]: int((r["reason"] == k).sum()) for k in range(len(nt.REASONS))},
           "all": _stats(err, acc, np.ones_like(acc)), "by_label": {},
           "median_frames_accepted": float(np.median(r["n_frames"][acc])) if acc.any() else None}
    for L in sorted(set(tr["label"])):
        out["by_label"][L] = _stats(err, acc, tr["label"] == L)
    if sc.name == "corridor":
        lat_ok = np.degrees(r["psi"])
        out["parallel_by_psi"] = {"1-44": _stats(err, acc, (tr["label"] == "parallel") & (lat_ok <= 44)),
                                  "44-136": _stats(err, acc, (tr["label"] == "parallel") & (lat_ok > 44) & (lat_ok < 136)),
                                  "136-179": _stats(err, acc, (tr["label"] == "parallel") & (lat_ok >= 136))}
    if sc.name == "occlusion":
        out["edge"], out["interior"] = _stats(err, acc, tr["edge"]), _stats(err, acc, ~tr["edge"])
    return out, r, err


def run_all(log=print, names=None) -> dict:
    """C1 as declared on #26. Returns the gates, the controls and every case."""
    plan = [("front", FORWARD_PSI), ("tilted", FORWARD_PSI), ("corridor", (1.0, 179.0)), ("periodic", FORWARD_PSI),
            ("radial", FORWARD_PSI), ("occlusion", FORWARD_PSI)]
    if names:
        plan = [p for p in plan if p[0] in names]
    gated = {"front": ("front",), "tilted": ("tilted",), "corridor": ("parallel", "front_far"), "periodic": ("periodic",)}
    scenes, cases, gates, ctrl = {}, [], [], []
    for name, psi in plan:
        scenes[name] = Scene(name)
        c, _, _ = run_n(scenes[name], psi=psi)
        cases.append(c)
        for L in gated.get(name, ()):
            st = c["by_label"][L]
            gates.append({"fixture": name, "label": L, "ok": gate_ok(st, 0.01 if name == "periodic" else None),
                          **{k: st[k] for k in ("accepted", "accepted_fraction", "median_px", "p90_px", "over_1px_of_accepted")}})
        if name == "radial":
            st = c["all"]
            gates.append({"fixture": name, "label": "radial", "ok": st["accepted_fraction"] <= 0.01,
                          **{k: st[k] for k in ("accepted", "accepted_fraction", "median_px", "p90_px", "over_1px_of_accepted")}})
        log(f"[C1] {name} ({c['seconds']} s, {c['all']['anchors']} anchors): " + "; ".join(
            f"{L} acc {v['accepted_fraction']:.3f} med {_r(v['median_px'])} p90 {_r(v['p90_px'])} >1px {_r(v['over_1px_of_accepted'])}"
            for L, v in c["by_label"].items() if v["anchors"]))
    if "front" in scenes:
        for lab, kw in (("steps 20 % too long", {"ts_scale": 1.2}),
                        (f"every frame from the scene scaled {mbf.FOREIGN_GEOMETRY}x about the reference's eye",
                         {"foreign_geometry": True})):
            c, _, _ = run_n(scenes["front"], **kw)
            ctrl.append({"control": lab, "fails": not gate_ok(c["by_label"]["front"]), "stats": c["by_label"]["front"]})
            log(f"[C1] control {'fails' if ctrl[-1]['fails'] else 'DOES NOT FAIL'}: {lab}: {c['by_label']['front']}")
    return {"gates": gates, "controls": ctrl, "cases": cases, "ok": all(g["ok"] for g in gates),
            "controls_fail": all(x["fails"] for x in ctrl)}


def _r(x):
    return None if x is None else round(x, 3)


def main(out: Path, names=None):
    import platform
    import subprocess
    import torch
    t0 = time.perf_counter()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "deep"], capture_output=True, text=True).stdout.strip())
    res = run_all(names=names)                                  # git is read before the run (#26's C1 record)
    res.update({"schema": "deep-motion-stage11-c1-v1", "computed_at": {"commit": git, "deep_dirty": dirty},
                "machine": platform.node(), "torch": torch.__version__, "seconds": round(time.perf_counter() - t0, 1)})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "c1_fixtures.json").write_text(json.dumps(res, indent=1, default=float))
    return res
