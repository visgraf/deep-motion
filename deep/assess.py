"""Stage 9 (#22): the assessment. The perfect matcher (stage 5a's oracle fit) and the real one (B with its mark) at the
same references (every 10 looks with a full +/-50-look window), through stage 8's fusion and metric, on the same runs.
Fixed on #22 before the code: the real matcher on all textured anchors; the perfect one on every hit pixel of a 2-px
grid (|elevation| < 70 deg), noisy true bearings (sigma 0.5 px) where visible, usable = 95 % interval within +/-10 %.
"""
import numpy as np

from . import e2e
from . import match as mt
from . import oracle as orc

SIGMA_PX = 0.5
GRID = 2
ITERS = 12
Z95 = 1.959964


def references(K, every=10, half=e2e.HALF):
    return list(range(half, K - half, every))


# ------------------------------------------------------------------------------------------- the perfect matcher

def oracle_points(truth, j, looks, fan):
    """Reference j's perfect-matcher points: hit pixels on the 2-px grid with |elevation| < 70 deg."""
    P, idx = truth[j]
    H, W = idx.shape
    rows = np.arange(0, H, GRID)
    lat = (0.5 - (rows + 0.5) / H) * np.pi
    rows = rows[np.abs(lat) < e2e.LAT_MAX]
    q = idx[rows][:, ::GRID].ravel()
    q = q[q >= 0]
    Pt = P[q]
    return Pt[np.all(np.isfinite(Pt), 1)]


def visibility(looks, refs, pts, vis_fn):
    """vis[j] (n_j, 2 half + 1): whether reference j's points are seen in each look of its window (one pass over the
    looks; vis_fn(k, P) is the run's own test)."""
    h = e2e.HALF
    vis = {j: np.zeros((len(pts[j]), 2 * h + 1), bool) for j in refs}
    for k in range(len(looks)):
        js = [j for j in refs if abs(k - j) <= h and k != j]
        if not js:
            continue
        allp = np.concatenate([pts[j] for j in js])
        v = vis_fn(k, allp)
        a = 0
        for j in js:
            n = len(pts[j])
            vis[j][:, k - j + h] = v[a:a + n]
            a += n
    return vis


def oracle_fit(looks, j, Pt, vis, rng, mirror=False):
    """Stage 5a's fit at reference j: range along the true point's own ray from noisy bearings (sigma 0.5 px) in the
    window's looks where it is seen; Fisher sigma; usable. mirror: the fit given look 2j - k's pose (the control)."""
    h = e2e.HALF
    c_r = looks[j][1]
    rt = np.linalg.norm(Pt - c_r, axis=-1)
    u = (Pt - c_r) / rt[:, None]
    res = 2 * np.pi / looks[j][3].shape[1]
    ks = [k for k in range(j - h, j + h + 1) if k != j]
    obs = []
    for k in ks:
        lon, lat, _, _ = orc.lonlat_and_derivs(c_r, u, rt, looks[k][1], looks[k][2])
        obs.append((lon + rng.normal(size=len(Pt)) * SIGMA_PX * res, lat + rng.normal(size=len(Pt)) * SIGMA_PX * res))
    model = [looks[2 * j - k] if mirror else looks[k] for k in ks]
    w = [vis[:, k - j + h] for k in ks]
    I = np.zeros(len(Pt))
    for (lk, wk) in zip(model, w):
        _, _, dlon, dlat = orc.lonlat_and_derivs(c_r, u, rt, lk[1], lk[2])
        I += np.where(wk, dlon ** 2 + dlat ** 2, 0.0)
    r = rt * 1.05
    for _ in range(ITERS):
        Hs, gs = np.zeros(len(Pt)), np.zeros(len(Pt))
        for (lon_o, lat_o), lk, wk in zip(obs, model, w):
            lon, lat, dlon, dlat = orc.lonlat_and_derivs(c_r, u, r, lk[1], lk[2])
            rl = np.angle(np.exp(1j * (lon_o - lon)))
            Hs += np.where(wk, dlon ** 2 + dlat ** 2, 0.0)
            gs += np.where(wk, dlon * rl + dlat * (lat_o - lat), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(Hs > 0, np.clip(r + gs / Hs, 0.05, 1e3), r)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(I > 0, SIGMA_PX * res / np.sqrt(I), np.inf)
    usable = (np.asarray(w).sum(0) >= 1) & (Z95 * sr / rt <= orc.USABLE_REL)
    return c_r + u * r[:, None], usable


def b_plus_h_points(m, fan):
    """The real pipeline's points at one reference (match_reference's output m): B's reliable estimates, then H on
    the 2-px grid seeded from them (amended on #22)."""
    win = m["win"]
    apx, au, aidx, _ = mt.anchors(win, e2e.HALF, fan, max_n=10 ** 9, frac=1.0)
    g = (apx[:, 0] % GRID == 0) & (apx[:, 1] % GRID == 0)
    tpx, tu = apx[g], au[g]
    mk = m["mark"]
    r_h, _, _ = mt.seed_fill(win, e2e.HALF, fan, m["px"][mk], m["u"][mk], m["r"][mk], tpx, tu)
    f = np.isfinite(r_h)
    return np.concatenate([m["c"] + m["u"][mk] * m["r"][mk][:, None], m["c"] + tu[f] * r_h[f][:, None]])


# --------------------------------------------------------------------------------------------------- surface

def surface(per_ref, truth):
    """The surface's 2 cm cells (as stage 8's coverage): codes, representative points, the nearest reference's range."""
    pts = []
    for j in per_ref:
        P, idx = truth[j]
        H, W = idx.shape
        rows = np.arange(0, H, 2)
        lat = (0.5 - (rows + 0.5) / H) * np.pi
        rows = rows[np.abs(lat) < e2e.LAT_MAX]
        q = idx[rows][:, ::2].ravel()
        pts.append(P[q[q >= 0]])
    S = np.concatenate(pts)
    S = S[np.all(np.isfinite(S), 1)]
    code = e2e._encode(e2e._keys(S))
    uc, inv = np.unique(code, return_inverse=True)
    rep = np.zeros((len(uc), 3))
    np.add.at(rep, inv, S)
    rep /= np.bincount(inv)[:, None]
    C = np.stack([m["c"] for m in per_ref.values()])
    rng = np.min(np.linalg.norm(rep[:, None, :] - C[None], axis=-1), axis=1)
    return uc, rep, rng


def covered(rep, Xr):
    """Whether each surface cell lies within 2 cm of a point of Xr (stage 8's rule)."""
    out = np.zeros(len(rep), bool)
    if len(Xr) == 0:
        return out
    fcode = e2e._encode(e2e._keys(Xr))
    order = np.argsort(fcode)
    fcs = fcode[order]
    rk = e2e._keys(rep)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nc = e2e._encode(rk + np.array([dx, dy, dz]))
                pos = np.clip(np.searchsorted(fcs, nc), 0, len(fcs) - 1)
                out |= (fcs[pos] == nc) & (np.linalg.norm(Xr[order[pos]] - rep, axis=-1) <= e2e.CELL)
    return out


def in_cells(uc, X):
    """Whether each surface cell (codes uc, sorted) holds at least one of the points X."""
    if len(X) == 0:
        return np.zeros(len(uc), bool)
    c = np.unique(e2e._encode(e2e._keys(X)))
    return np.isin(uc, c)


# ---------------------------------------------------------------------------------------------------- stage

def fuse_and_judge(Xs, refs_of, per_ref, truth, fan):
    X, ref = np.concatenate(Xs), np.concatenate(refs_of)
    fin = np.all(np.isfinite(X), 1)
    X, ref = X[fin], ref[fin]
    Xf, nref, dom = e2e.fuse(X, ref)
    cell_of = e2e.fuse.cell_of
    right = e2e.cells_right(Xf, dom, per_ref, truth, fan) if len(Xf) else np.zeros(0, bool)
    return X, ref, Xf, nref, dom, right, cell_of


def by_refs(nref, right):
    out = {}
    for n in (1, 2, 3, 4):
        sel = (nref == n) if n < 4 else (nref >= 4)
        out[str(n) if n < 4 else "4+"] = {"cells": int(sel.sum()), "right": float(right[sel].mean()) if sel.any() else None}
    return out


def assess_path(looks, truth, fan, refs, vis_fn, seed=22):
    """Both matchers on one path. Returns the numbers and the check inputs."""
    rng = np.random.default_rng(seed)
    per_ref = {j: e2e.match_reference(looks, truth, j, fan, max_n=10 ** 9) for j in refs}
    per_ref_m = {j: e2e.match_reference(looks, truth, j, fan, max_n=10 ** 9, mirror=True) for j in refs}
    pts = {j: oracle_points(truth, j, looks, fan) for j in refs}
    vis = visibility(looks, refs, pts, vis_fn)
    ox, oxm, oref = [], [], []
    for j in refs:
        X, us = oracle_fit(looks, j, pts[j], vis[j], rng)
        Xm, usm = oracle_fit(looks, j, pts[j], vis[j], rng, mirror=True)
        ox.append((X[us], np.full(int(us.sum()), j), float(us.mean())))
        oxm.append((Xm[usm], np.full(int(usm.sum()), j)))
    real = fuse_and_judge([m["c"] + m["u"][m["mark"]] * m["r"][m["mark"]][:, None] for m in per_ref.values()],
                          [np.full(int(m["mark"].sum()), j) for j, m in per_ref.items()], per_ref, truth, fan)
    realm = fuse_and_judge([m["c"] + m["u"][m["mark"]] * m["r"][m["mark"]][:, None] for m in per_ref_m.values()],
                           [np.full(int(m["mark"].sum()), j) for j, m in per_ref_m.items()], per_ref, truth, fan)
    perf = fuse_and_judge([o[0] for o in ox], [o[1] for o in ox], per_ref, truth, fan)
    perfm = fuse_and_judge([o[0] for o in oxm], [o[1] for o in oxm], per_ref, truth, fan)
    # the real pipeline's second tier (amended on #22): H, seeded from each reference's reliable anchors, on the same
    # 2-px grid as the perfect matcher's points; reported, not gated
    hx, href = [], []
    for j, m in per_ref.items():
        hx.append(b_plus_h_points(m, fan))
        href.append(np.full(len(hx[-1]), j))
    realh = fuse_and_judge(hx, href, per_ref, truth, fan)
    X, ref, Xf, nref, dom, right, cell_of = real
    ctl_a3, ctl_a3_n = e2e.perturbed_confirmed(per_ref, X, ref, Xf, nref, dom, cell_of, truth, fan)
    uc, rep, srng = surface(per_ref, truth)
    cov_real = covered(rep, Xf[right])
    cov_perf = covered(rep, perf[2][perf[5]])
    cov_realh = covered(rep, realh[2][realh[5]])
    anyanchor = in_cells(uc, np.concatenate([m["ptrue"] for m in per_ref.values()]))
    anyrel = in_cells(uc, np.concatenate([m["ptrue"][m["mark"]] for m in per_ref.values()]))
    gap = cov_perf & ~cov_real
    edge_px = []
    for j, m in per_ref.items():
        e = mt.depth_edge(m["win"], e2e.HALF, truth[j][0], m["px"])
        edge_px.append(m["ptrue"][e])
    atedge = in_cells(uc, np.concatenate(edge_px))
    bins = {"< 1 m": (0, 1), "1-2 m": (1, 2), "2-3 m": (2, 3), ">= 3 m": (3, 99)}

    def frac(a, msk):
        return float(a[msk].mean()) if msk.any() else None
    out = {"references": refs,
           "real": {"per_reference_kept": {str(j): float(m["mark"].mean()) for j, m in per_ref.items()},
                    "anchors_per_reference": {str(j): int(len(m["mark"])) for j, m in per_ref.items()},
                    "cells": int(len(Xf)), "right": float(right.mean()), "by_references": by_refs(nref, right),
                    "confirmed_right": frac(right, nref >= 2), "coverage": float(cov_real.mean()),
                    "control_mirrored_right": float(realm[5].mean()) if len(realm[5]) else None,
                    "control_moved_20pc_right": ctl_a3, "control_moved_20pc_cells": ctl_a3_n},
           "perfect": {"usable_per_reference": {str(j): o[2] for j, o in zip(refs, ox)},
                       "cells": int(len(perf[2])), "right": float(perf[5].mean()), "by_references": by_refs(perf[3], perf[5]),
                       "coverage": float(cov_perf.mean()),
                       "control_mirrored_right": float(perfm[5].mean()) if len(perfm[5]) else None},
           "real_B_plus_H": {"cells": int(len(realh[2])), "right": float(realh[5].mean()),
                             "by_references": by_refs(realh[3], realh[5]), "coverage": float(cov_realh.mean()),
                             "covers_of_perfect": frac(cov_realh, cov_perf)},
           "surface_cells": int(len(uc)),
           "real_over_perfect_coverage": float(cov_real.mean() / cov_perf.mean()) if cov_perf.any() else None,
           "real_covers_of_perfect": frac(cov_real, cov_perf),
           "gap": {"cells": int(gap.sum()), "share_of_surface": float(gap.mean()),
                   "no anchor (not among the 15 % most textured)": frac(~anyanchor, gap),
                   "anchors, none reliable": frac(anyanchor & ~anyrel, gap),
                   "reliable, no right cell near": frac(anyrel, gap)},
           "by_range": {k: {"surface_cells": int(((srng >= a) & (srng < b)).sum()),
                            "perfect_coverage": frac(cov_perf, (srng >= a) & (srng < b)),
                            "real_coverage": frac(cov_real, (srng >= a) & (srng < b))} for k, (a, b) in bins.items()},
           "edges": {"surface_cells_at_edges": float(atedge.mean()),
                     "perfect_coverage_at/off": [frac(cov_perf, atedge), frac(cov_perf, ~atedge)],
                     "real_coverage_at/off": [frac(cov_real, atedge), frac(cov_real, ~atedge)]}}
    return out


def render_vis(run_dir, fan):
    _, L = orc.load(run_dir)

    def f(k, P):
        return orc.visible(orc.record(L[k][0]), np.asarray(L[k][1], float), np.asarray(L[k][2], float), P, fan)
    return f


def r0_vis(kind, fan):
    """R0's visibility: the point R0's z-buffer shows at the point's pixel is the point itself, within 2 px of range."""
    P, g, W = mt.r0_world()
    from . import fusion as fu
    poses = fu.path_poses(kind=kind, step=0.01)
    res = 2 * np.pi / W

    def f(k, Pq):
        s, c, R, _ = poses[k]
        _, idx = mt.r0_look(P, g, c, R, fan)
        rc, _ = fan.model().pixel_of((Pq - c) @ R)
        i = np.rint(rc[:, 0]).astype(np.int64)
        jj = np.rint(rc[:, 1]).astype(np.int64) % W
        ok = (i >= 0) & (i < idx.shape[0])
        q = np.full(len(Pq), -1)
        q[ok] = idx[i[ok], jj[ok]]
        rng = np.linalg.norm(Pq - c, axis=-1)
        err = np.where(q >= 0, np.linalg.norm(P[np.maximum(q, 0)] - Pq, axis=-1), np.inf)
        return (q >= 0) & (err <= orc.TOL_PX * res * rng)
    return f


def selftest(paths):
    """The checks declared on #22. paths: {name: (looks, truth, fan, refs, vis_fn)}."""
    res, num = [], {}
    for name, (looks, truth, fan, refs, vis_fn) in paths.items():
        o = assess_path(looks, truth, fan, refs, vis_fn)
        num[name] = o
        P, R = o["perfect"], o["real"]
        res.append((f"A1 {name}: the perfect matcher's fused cells >= 99 % right", P["right"] >= 0.99))
        res.append((f"A1 {name} control: its fit given the mirrored path's poses (< 99 %)",
                    P["control_mirrored_right"] is None or P["control_mirrored_right"] < 0.99))
        res.append((f"A2 {name}: the real matcher's fused cells >= 95 % right", R["right"] >= 0.95))
        res.append((f"A2 {name} control: the mirrored path (< 95 %)",
                    R["control_mirrored_right"] is None or R["control_mirrored_right"] < 0.95))
        res.append((f"A3 {name}: the real matcher's cells from >= 2 references >= 97 % right",
                    R["confirmed_right"] is not None and R["confirmed_right"] >= 0.97))
        res.append((f"A3 {name} control: their members moved 20 % along their rays (< 97 %)",
                    R["control_moved_20pc_right"] is None or R["control_moved_20pc_right"] < 0.97))
    return res, num
