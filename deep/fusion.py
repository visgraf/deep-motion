"""Stage 5b.2 (#7): a composed path (line, arc, line) and the oracle's fusion into world 3D, gated in range.

Plan: docs/plan.md, stage 5; P-D3. The design and checks were declared on #7 before this code.

Cells are 2 cm voxels of the true points (the oracle stage); a cell's representative is its first hit. Each cell's
3-D point is fitted by Gauss-Newton to its bearing and elevation (the Panorama's lon, lat) in every look that sees it
(M1.3's agreement, as 5a), each look once: windows would share looks and count them again. Covariance:
sigma^2 res^2 H^-1. Usable: the 95 % interval along the ray within +/-10 % of range from every look that sees it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import oracle as orc
from . import primitives as pr
from . import sensor as sn

CELL = 0.02
STEP = 0.025
PIECES = (("line", 0.30, 0.0), ("arc", 0.30, 1.0 / 1.6), ("line", 0.30, 0.0))
SEG_NAMES = ("line A", "arc", "line B")
PATHS = {"composed": PIECES, "straight": (("line", 0.90, 0.0),),   # straight: the same length, for comparison
         # #7 (Luiz agreed to a real turn): 90 deg left; R = 1 m, since at R = 1.6 m the arc runs into a board of the
         # calibration room (0.8 cm clearance from its surface points; R = 1 m: 18 cm)
         "turn90": (("line", 0.30, 0.0), ("arc", 1.0 * np.pi / 2, 1.0), ("line", 0.30, 0.0)),
         # a straight path as long as turn90 (2.17 m), to separate turning from length; in the calibration room it fits
         # only from further back (--start 0,-2,90)
         "straight_long": (("line", 0.60 + 1.0 * np.pi / 2, 0.0),)}
START = {"p0": (0.0, 0.0), "heading_deg": 90.0, "h": 1.6}          # the calibration room's EYE, heading +Y


def composed_path(pieces=PIECES, p0=(0.0, 0.0), phi0=np.pi / 2, h=1.6):
    return pr.Path([pr.Piece(k, L, kap) for k, L, kap in pieces], p0, phi0, h)


def path_poses(path=None, step=STEP, kind="composed", start=None):
    """[(s, c, R, piece)] every `step` of arc length over the whole path, joins included once."""
    st = {**START, **(start or {})}
    path = path or composed_path(PATHS[kind], st["p0"], np.radians(st["heading_deg"]), st["h"])
    total = sum(pc.length for pc in path.pieces)
    out = []
    for s in np.arange(0.0, total + 1e-9, step):
        acc = 0.0
        for i, (pc, (p0, phi0)) in enumerate(zip(path.pieces, path.starts)):
            if s <= acc + pc.length + 1e-9 or i == len(path.pieces) - 1:
                c, R, _, _ = pc.pose(p0, phi0, path.h, min(s - acc, pc.length))
                out.append((float(s), c, R, i))
                break
            acc += pc.length
    return out


def load_looks(run_dirs):
    """Looks of one or more runs, in order, without their records (loaded one at a time): [(path, c, R, s)]."""
    looks = []
    for d in run_dirs:
        run = json.loads((Path(d) / "run.json").read_text())
        looks += [(Path(d) / lk["out_dir"] / "record_C.npz", np.asarray(lk["c"], float), np.asarray(lk["R"], float),
                   lk["s"]) for lk in run["looks"]]
    return looks


def _record(path):
    return orc.record(path)


def run_meta(run_dirs):
    """The path's kind and start from the first run's record (runs before #7's turn carry none: composed, EYE)."""
    run = json.loads((Path(run_dirs[0]) / "run.json").read_text())
    return run.get("path", "composed"), {**START, **run.get("start", {})}, int(run["carrier"]["width"])


def cells(looks, cell=CELL, hit_stride=1):
    """Representative true points, one per voxel of `cell` m, the first hit wins (looks in order). With hit_stride > 1,
    a random 1/hit_stride of each look's hits is binned (seeded by the look's index): a regular stride would fall on
    fixed columns whenever it divides the width (#6)."""
    Ps = []
    seen = np.empty((0, 3), dtype=np.int64)
    for i, lk in enumerate(looks):
        rec = _record(lk[0])
        P = rec.position_w[np.isfinite(rec.distance_m)]
        if hit_stride > 1:
            P = P[np.sort(np.random.default_rng(1000 + i).choice(len(P), size=len(P) // hit_stride, replace=False))]
        P = P.astype(np.float64)
        key = np.floor(P / cell).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        first = np.sort(first)
        key, P = key[first], P[first]
        allk = np.concatenate([seen, key])
        _, idx = np.unique(allk, axis=0, return_index=True)
        new = np.sort(idx[idx >= len(seen)]) - len(seen)
        seen = np.concatenate([seen, key[new]])
        Ps.append(P[new])
    return np.concatenate(Ps)


def visibility(looks, P, fan):
    """(N, K): M1.3's agreement in every look, one record in memory at a time."""
    return np.stack([orc.visible(_record(lk[0]), lk[1], lk[2], P, fan) for lk in looks], 1)


def _derivs(X, c, R):
    """lon, lat of X (N, 3) from (c, R) and their gradients in X (N, 3) each."""
    v = (X - c) @ R
    hx = v[:, 0] ** 2 + v[:, 2] ** 2
    n2 = hx + v[:, 1] ** 2
    lon = np.arctan2(v[:, 0], -v[:, 2])
    lat = np.arcsin(np.clip(v[:, 1] / np.sqrt(n2), -1, 1))
    g_lon = np.stack([-v[:, 2], np.zeros_like(hx), v[:, 0]], -1) / hx[:, None]
    g_lat = (np.stack([np.zeros_like(hx), n2, np.zeros_like(hx)], -1) - v[:, 1:2] * v) / (n2 * np.sqrt(hx))[:, None]
    return lon, lat, g_lon @ R.T, g_lat @ R.T


def rays_start(obs, vis, poses, X_fallback):
    """Each cell's least-squares intersection of the rays it is observed along (its lon, lat in every look that sees
    it): the point nearest all of them. A weak pull (1e-6 of A's mean eigenvalue) toward X_fallback (the earlier
    start) settles the directions the rays leave free when they are almost parallel. #7: starting 5 % beyond the
    truth along one ray, a cell under the arc stalled on a look's nadir, where lon is undefined; from here it
    converges."""
    N = len(vis)
    A = np.zeros((N, 3, 3))
    b = np.zeros((N, 3))
    for k, (c, R) in enumerate(poses):
        lon, lat = obs[k]
        v = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1)
        d = v @ R.T                                     # world direction: X - c = R v
        M = np.eye(3) - d[:, :, None] * d[:, None, :]
        w = vis[:, k][:, None, None]
        A += w * M
        b += (w * M) @ c
    lam = 1e-6 * np.trace(A, axis1=1, axis2=2)[:, None, None] / 3 + 1e-12
    A += lam * np.eye(3)
    b += lam[:, :, 0] * X_fallback
    return np.linalg.solve(A, b[..., None])[..., 0]


def fuse(looks, P, fan, sigma_px=0.0, rng=None, poses=None, iters=60, vis=None, start="rays"):
    """Joint fit of each cell's 3-D point. poses: [(c, R)] the fit believes (default: the looks' own; a control may
    move them); the observations always come from the true poses.
    Returns X (N, 3), H (N, 3, 3) at the truth, vis (N, K)."""
    K = len(looks)
    if vis is None:
        vis = visibility(looks, P, fan)
    poses = poses or [(lk[1], lk[2]) for lk in looks]
    obs = []
    for k, lk in enumerate(looks):
        lon, lat, _, _ = _derivs(P, lk[1], lk[2])
        if sigma_px > 0:
            lon = lon + rng.normal(size=len(P)) * sigma_px * fan.res
            lat = lat + rng.normal(size=len(P)) * sigma_px * fan.res
        obs.append((lon, lat))
    H = np.zeros((len(P), 3, 3))
    for k in range(K):
        _, _, gl, gt = _derivs(P, *poses[k])
        w = vis[:, k][:, None, None]
        H += w * (gl[:, :, None] * gl[:, None, :] + gt[:, :, None] * gt[:, None, :])
    first = np.argmax(vis, 1)                           # "ray5": 5 % beyond the truth along the first look's ray
    c0 = np.stack([poses[k][0] for k in range(K)])[first]   # (the start until #7; F1b's control)
    X = c0 + 1.05 * (P - c0)
    if start == "rays":
        X = rays_start(obs, vis, poses, X)
    act = np.flatnonzero(vis.sum(1) >= 2)

    def cost(Xa, idx):
        c = np.zeros(len(idx))
        for k in range(K):
            lon, lat, _, _ = _derivs(Xa, *poses[k])
            rl = np.angle(np.exp(1j * (obs[k][0][idx] - lon)))
            c += np.where(vis[idx, k], rl ** 2 + (obs[k][1][idx] - lat) ** 2, 0.0)
        return c

    for _ in range(iters):
        # only the cells still moving; a step that raises the cost is halved (near a look's pole lon is singular
        # and a full Gauss-Newton step can throw a cell away)
        Xa, va = X[act], vis[act]
        Hx, g = np.zeros((len(act), 3, 3)), np.zeros((len(act), 3))
        for k in range(K):
            lon, lat, gl, gt = _derivs(Xa, *poses[k])
            rl = np.angle(np.exp(1j * (obs[k][0][act] - lon)))
            rt = obs[k][1][act] - lat
            w = va[:, k][:, None]
            Hx += w[:, :, None] * (gl[:, :, None] * gl[:, None, :] + gt[:, :, None] * gt[:, None, :])
            g += w * (gl * rl[:, None] + gt * rt[:, None])
        Hx += 1e-12 * np.eye(3)
        bad = ~(np.isfinite(Hx).all((1, 2)) & np.isfinite(g).all(1))
        Hx[bad], g[bad] = np.eye(3), 0.0                # a cell on a look's centre or pole: no step; reported by F1
        dX = np.linalg.solve(Hx, g[..., None])[..., 0]
        c0 = cost(Xa, act)
        lam = np.ones(len(act))
        todo = np.arange(len(act))
        for _ in range(40):
            cn = cost(Xa[todo] + lam[todo, None] * dX[todo], act[todo])
            todo = todo[cn > c0[todo] * (1 + 1e-12) + 1e-30]
            if not len(todo):
                break
            lam[todo] *= 0.5
        lam[todo] = 0.0
        step = lam[:, None] * dX
        X[act] += step
        act = act[np.linalg.norm(step, axis=-1) > 1e-14 * np.linalg.norm(Xa - poses[0][0], axis=-1)]
        if not len(act):
            break
    return X, H, vis


def ray_interval(P, H, vis, looks, sigma_px, fan):
    """95 % half-interval / range along the ray from every look that sees the cell; returns the worst (N,), and the
    along-ray sigma from each cell's first look (N,) with that look's unit ray (N, 3)."""
    with np.errstate(invalid="ignore"):
        C = np.linalg.pinv(H) * (sigma_px * fan.res) ** 2
    worst = np.zeros(len(P))
    for k, lk in enumerate(looks):
        d = P - lk[1]
        r = np.linalg.norm(d, axis=-1)
        u = d / r[:, None]
        s = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", u, C, u), 0))
        worst = np.where(vis[:, k], np.maximum(worst, orc.Z95 * s / r), worst)
    worst = np.where(vis.sum(1) >= 2, worst, np.inf)
    first = np.argmax(vis, 1)
    c0 = np.stack([lk[1] for lk in looks])[first]
    u0 = (P - c0) / np.linalg.norm(P - c0, axis=-1)[:, None]
    s0 = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", u0, C, u0), 0))
    return worst, s0, u0


def progress(P, vis, looks, fan, seg, sigma_px=0.5, points=40):
    """Usable fraction of the cells after the first m looks (about `points` values of m, the last look included),
    and each cell's first-usable segment (resolved to the same spacing)."""
    K = len(looks)
    every = max(1, int(np.ceil(K / points)))
    H = np.zeros((len(P), 3, 3))
    frac, at, first_seg = [], [], np.full(len(P), -1)
    for m in range(K):
        _, _, gl, gt = _derivs(P, looks[m][1], looks[m][2])
        w = vis[:, m][:, None, None]
        H += w * (gl[:, :, None] * gl[:, None, :] + gt[:, :, None] * gt[:, None, :])
        if m % every and m != K - 1:
            continue
        at.append(m)
        worst, _, _ = ray_interval(P, H, vis[:, :m + 1], looks[:m + 1], sigma_px, fan)
        u = worst <= orc.USABLE_REL
        first_seg = np.where(u & (first_seg < 0), seg[m], first_seg)
        frac.append(float(u.mean()))
    return frac, first_seg, at


def pole_check(n_rep=40, seed=0):
    """F1b's cells and both starts; returns its numbers."""
    fan = sn.SlitFan(width=3600)
    ps = path_poses(kind="turn90", step=0.01)
    looks = [(None, c, R, s) for s, c, R, _ in ps]
    C = np.stack([lk[1] for lk in looks])
    rng = np.random.default_rng(seed)
    P = np.concatenate([C[40:90, :2] + rng.normal(scale=0.01, size=(50, 2)) for _ in range(n_rep)])
    P = np.c_[P, np.zeros(len(P))]
    vis = np.ones((len(P), len(looks)), bool)
    for i in range(len(P)):
        vis[i, rng.choice(len(looks), size=8, replace=False)] = False
    rmin = np.min(np.linalg.norm(P[:, None] - C[None], axis=-1), 1)
    out = {}
    for st, key in (("rays", "F1b"), ("ray5", "F1b_control")):
        X, _, _ = fuse(looks, P, fan, vis=vis, start=st)
        e = np.linalg.norm(X - P, axis=-1) / rmin
        out[f"{key}_failures"], out[f"{key}_max_rel"] = int((e > 1e-9).sum()), float(e.max())
    out["F1b_cells"] = int(len(P))
    return out


def selftest(run_dirs, fan=None, cell=CELL, seed=7, kind=None, hit_stride=1, max_cells=None):
    looks = load_looks(run_dirs)
    k_meta, start, width = run_meta(run_dirs)
    kind = kind or k_meta
    fan = fan or sn.SlitFan(width=width)
    s = np.array([lk[3] for lk in looks])
    ends = np.cumsum([L for _, L, _ in PATHS[kind]])
    seg = np.searchsorted(ends - 1e-9, s)        # a join belongs to the piece it ends, as in path_poses
    P = cells(looks, cell, hit_stride)
    rng = np.random.default_rng(seed)
    n_all = len(P)
    if max_cells and len(P) > max_cells:
        P = P[np.sort(rng.choice(len(P), size=max_cells, replace=False))]
    res, num = [], {"path": kind, "start": start, "cells_binned": int(n_all), "cells": int(len(P)), "looks": len(looks),
                    "cell_m": cell, "hit_stride": hit_stride, "max_cells": max_cells}
    X0, H, vis = fuse(looks, P, fan)
    seen2 = vis.sum(1) >= 2
    rmin = np.min(np.where(vis, np.stack([np.linalg.norm(P - lk[1], axis=-1) for lk in looks], 1), np.inf), 1)
    e0 = np.linalg.norm(X0 - P, axis=-1) / rmin
    num["seen_by_two"] = float(seen2.mean())
    worst_ref, _, _ = ray_interval(P, H, vis, looks, 0.5, fan)
    well = seen2 & (worst_ref <= orc.USABLE_REL)
    num["F1_max_rel"] = float(np.max(e0[well]))
    num["F1_max_rel_all_seen_by_two"] = float(np.max(e0[seen2]))
    j = int(np.argmax(seg > 0))
    moved = [(lk[1] + (np.array([0.01, 0, 0]) if k >= j else 0.0), lk[2]) for k, lk in enumerate(looks)]
    Xc, _, _ = fuse(looks, P, fan, poses=moved, vis=vis)
    num["F1_control_median_rel"] = float(np.median((np.linalg.norm(Xc - P, axis=-1) / rmin)[well]))
    res.append(("F1 no noise: the fused points are the true points, every cell seen by two looks (<= 1e-9 of range)",
                num["F1_max_rel_all_seen_by_two"] <= 1e-9))
    res.append(("F1 control: the looks after the first join moved 1 cm (median >= 1e-3)", num["F1_control_median_rel"] >= 1e-3))
    # F1b the solver near a look's pole (#7, Code: a floor cell under the arc stalled on a look's nadir at 3600 wide).
    # Synthetic, independent of the runs: 2000 floor cells within ~1 cm of the turn90 path's looks 40-89 (1 cm steps),
    # each missed by 8 random looks; no noise. Control: the start used until #7 (5 % along the first ray) stalls.
    num.update(pole_check())
    res.append(("F1b no noise, floor cells under the turn's looks (a look's nadir): every cell recovered (<= 1e-9)",
                num["F1b_failures"] == 0 and num["F1b_max_rel"] <= 1e-9))
    res.append(("F1b control: the earlier start (5 % along the first ray) stalls on at least one cell",
                num["F1b_control_failures"] >= 1))
    # F2 honest sigma: 30 draws at 0.5 px on 2000 usable cells
    idx = rng.choice(np.flatnonzero(well), size=min(2000, int(well.sum())), replace=False)
    Ps, Hs, vs = P[idx], H[idx], vis[idx]
    _, s0, u0 = ray_interval(Ps, Hs, vs, looks, 0.5, fan)
    _, s0d, _ = ray_interval(Ps, 2 * Hs, vs, looks, 0.5, fan)
    z, zd = [], []
    for _ in range(30):
        X, _, _ = fuse(looks, Ps, fan, sigma_px=0.5, rng=rng, vis=vs)
        e = np.einsum("ni,ni->n", X - Ps, u0)
        z.append(e / s0)
        zd.append(e / s0d)
    num["F2_std_z"], num["F2_control_std_z"] = float(np.std(np.concatenate(z))), float(np.std(np.concatenate(zd)))
    res.append(("F2 the fused sigma is honest: std of along-ray error / Fisher sigma within 3 % of 1", abs(num["F2_std_z"] - 1) <= 0.03))
    res.append(("F2 control: each look counted twice (>= 20 % off)", abs(num["F2_control_std_z"] - 1) >= 0.20))
    # F3 coverage per rung, and the tables
    num["rungs"], cov_ok, ctl_max = {}, True, 0.0
    for sg in orc.SIGMAS_PX:
        X, _, _ = fuse(looks, P, fan, sigma_px=sg, rng=rng, vis=vis)
        worst, s0a, u0a = ray_interval(P, H, vis, looks, sg, fan)
        usable = worst <= orc.USABLE_REL
        e = np.abs(np.einsum("ni,ni->n", X - P, u0a))
        cov = float(np.mean(e[usable] <= orc.Z95 * s0a[usable]))
        ctl = float(np.mean(e[usable] <= orc.Z95 * 0.5 * s0a[usable]))
        err3 = np.linalg.norm(X - P, axis=-1)[usable]
        num["rungs"][f"{sg:g}"] = {"usable": float(usable.mean()), "coverage": cov, "coverage_sigma_halved": ctl,
                                   "median_3d_error_m": float(np.median(err3)), "p95_3d_error_m": float(np.percentile(err3, 95))}
        cov_ok &= abs(cov - 0.95) <= 0.01
        ctl_max = max(ctl_max, ctl)
    res.append(("F3 the usable cells' 95 % intervals cover the truth at 95 +/- 1 % (0.25, 0.5, 1 px)", bool(cov_ok)))
    res.append(("F3 control: sigma halved covers <= 85 % on every rung", ctl_max <= 0.85))
    sub = rng.choice(len(P), size=min(20000, len(P)), replace=False)
    frac, fseg, at = progress(P[sub], vis[sub], looks, fan, seg, 0.5)
    num["progress_cells"] = int(len(sub))
    num["progress_0.5px"] = {"s_m": s[at].tolist(), "usable": frac}
    names = SEG_NAMES if len(PATHS[kind]) == 3 else ("line",)
    num["first_usable_segment_0.5px"] = {names[i]: float(np.mean(fseg == i)) for i in range(len(names))}
    num["first_usable_segment_0.5px"]["never"] = float(np.mean(fseg < 0))
    return res, num, {"P": P, "vis": vis, "H": H, "sub": sub, "first_seg": fseg, "seg": seg, "s": s}


def usable_map(run_dirs, sigma_px=0.5, cell=CELL, hit_stride=1):
    """The Fisher gate only (no fit): cells, usable at sigma_px, and their bearing from the path's middle look
    relative to the start heading, for the figure and the cone summary."""
    looks = load_looks(run_dirs)
    _, start, width = run_meta(run_dirs)
    fan = sn.SlitFan(width=width)
    P = cells(looks, cell, hit_stride)
    vis = visibility(looks, P, fan)
    H = np.zeros((len(P), 3, 3))
    for lk, k in zip(looks, range(len(looks))):
        _, _, gl, gt = _derivs(P, lk[1], lk[2])
        H += vis[:, k][:, None, None] * (gl[:, :, None] * gl[:, None, :] + gt[:, :, None] * gt[:, None, :])
    worst, _, _ = ray_interval(P, H, vis, looks, sigma_px, fan)
    cm = looks[len(looks) // 2][1]
    d = P - cm
    h0 = np.radians(start["heading_deg"])
    b = np.degrees(np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - h0))))   # from the start heading, + left
    return {"P": P, "usable": worst <= orc.USABLE_REL, "worst": worst, "bearing": b,
            "path": np.stack([lk[1] for lk in looks])}


CONES = {"ahead |b| < 15°": (0, 15), "ahead 15–30°": (15, 30), "lateral 60–120°": (60, 120), "behind |b| > 165°": (165, 181)}


def cone_summary(m):
    ab = np.abs(m["bearing"])
    return {k: {"cells": int(((ab >= lo) & (ab < hi)).sum()), "usable": float(m["usable"][(ab >= lo) & (ab < hi)].mean())}
            for k, (lo, hi) in CONES.items()}
