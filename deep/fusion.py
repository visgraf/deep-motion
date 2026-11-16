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


def composed_path(pieces=PIECES, p0=(0.0, 0.0), phi0=np.pi / 2, h=1.6):
    return pr.Path([pr.Piece(k, L, kap) for k, L, kap in pieces], p0, phi0, h)


def path_poses(path=None, step=STEP):
    """[(s, c, R, piece)] every `step` of arc length over the whole path, joins included once."""
    path = path or composed_path()
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
    """Looks of one or more runs, in order: [(rec, c, R, s)]."""
    looks = []
    for d in run_dirs:
        looks += orc.load(d)[1]
    return looks


def cells(looks, cell=CELL):
    """Representative true points, one per voxel of `cell` m, the first hit wins (looks in order)."""
    P = np.concatenate([lk[0].position_w[np.isfinite(lk[0].distance_m)].astype(np.float64) for lk in looks])
    key = np.floor(P / cell).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return P[np.sort(first)]


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


def fuse(looks, P, fan, sigma_px=0.0, rng=None, poses=None, iters=60, vis=None):
    """Joint fit of each cell's 3-D point. poses: [(c, R)] the fit believes (default: the looks' own; a control may
    move them); the observations always come from the true poses.
    Returns X (N, 3), H (N, 3, 3) at the truth, vis (N, K)."""
    K = len(looks)
    if vis is None:
        vis = np.stack([orc.visible(lk[0], lk[1], lk[2], P, fan) for lk in looks], 1)
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
    first = np.argmax(vis, 1)
    c0 = np.stack([poses[k][0] for k in range(K)])[first]
    X = c0 + 1.05 * (P - c0)
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


def progress(P, vis, looks, fan, seg, sigma_px=0.5):
    """Usable fraction of the cells after the first m looks (m = 1..K), and each cell's first-usable segment."""
    K = len(looks)
    H = np.zeros((len(P), 3, 3))
    frac, first_seg = [], np.full(len(P), -1)
    for m in range(K):
        _, _, gl, gt = _derivs(P, looks[m][1], looks[m][2])
        w = vis[:, m][:, None, None]
        H += w * (gl[:, :, None] * gl[:, None, :] + gt[:, :, None] * gt[:, None, :])
        worst, _, _ = ray_interval(P, H, vis[:, :m + 1], looks[:m + 1], sigma_px, fan)
        u = worst <= orc.USABLE_REL
        first_seg = np.where(u & (first_seg < 0), seg[m], first_seg)
        frac.append(float(u.mean()))
    return frac, first_seg


def selftest(run_dirs, fan=None, cell=CELL, seed=7):
    looks = load_looks(run_dirs)
    fan = fan or sn.SlitFan(width=int(json.loads((Path(run_dirs[0]) / "run.json").read_text())["carrier"]["width"]))
    s = np.array([lk[3] for lk in looks])
    ends = np.cumsum([L for _, L, _ in PIECES])
    seg = np.searchsorted(ends - 1e-9, s)        # a join belongs to the piece it ends, as in path_poses
    P = cells(looks, cell)
    rng = np.random.default_rng(seed)
    res, num = [], {"cells": int(len(P)), "looks": len(looks), "cell_m": cell}
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
    frac, fseg = progress(P[sub], vis[sub], looks, fan, seg, 0.5)
    num["progress_cells"] = int(len(sub))
    num["progress_0.5px"] = {"s_m": s.tolist(), "usable": frac}
    num["first_usable_segment_0.5px"] = {SEG_NAMES[i]: float(np.mean(fseg == i)) for i in range(len(SEG_NAMES))}
    num["first_usable_segment_0.5px"]["never"] = float(np.mean(fseg < 0))
    return res, num, {"P": P, "vis": vis, "H": H, "sub": sub, "first_seg": fseg, "seg": seg, "s": s}
