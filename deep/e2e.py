"""Stage 8 (#19): end to end. B's reliable estimates at a few reference looks along a path, fused into 2 cm world cells
without truth, judged against the reference looks' truth; the composed path (turn90) against the straight one.

The method and the checks are fixed on #19 before the code: references m = 50, 79, 108, 138, 167 (a full +/-50-look
window each), B with its mark on 20,000 textured anchors per reference, cells keyed by the estimated point, the fused
point the per-axis median, a cell right by K10's surface metric from the reference that contributed most to it.
"""
import numpy as np

from . import match as mt
from . import oracle as orc
from . import fusion as fu
from . import sensor as sn

HALF = 50                                           # the window: +/-50 looks around each reference
N_REFS = 5
CELL = 0.02
LAT_MAX = np.radians(70)


def references(K, n=N_REFS, half=HALF):
    """n reference looks evenly over those with a full window: for 218 looks, 50, 79, 108, 138, 167."""
    return [int(round(x)) for x in np.linspace(half, K - 1 - half, n)]


# ------------------------------------------------------------------------------------------------------ loading

def render_path(run_dir, refs=None):
    """A render run's looks [(s, c, R, img, None)] (luminance scaled by the middle look's 99th percentile, as R1) and
    the truth at the references {j: (P_j, idx_j)} (oracle access; evaluation only)."""
    _, L = orc.load(run_dir)
    K = len(L)
    refs = refs or references(K)
    rec_mid = orc.record(L[K // 2][0])
    hit = np.isfinite(rec_mid.distance_m)
    scale = float(np.quantile(rec_mid.value[hit].astype(np.float64) @ mt.LUM, 0.99))
    H, W = (int(x) for x in rec_mid.raster_shape)
    looks, truth = [], {}
    for k, (path, c, R, s) in enumerate(L):
        rec = orc.record(path)
        g = rec.value.astype(np.float64) @ mt.LUM / scale
        ok = np.isfinite(rec.distance_m)
        g[~ok] = np.nan
        looks.append((float(s), np.asarray(c, float), np.asarray(R, float), g.reshape(H, W).astype(np.float32), None))
        if k in refs:
            truth[k] = (rec.position_w.astype(np.float64), np.where(ok, np.arange(H * W), -1).reshape(H, W))
    return looks, truth, sn.SlitFan(width=W), refs


def r0_path(kind, start=None, step=0.01, refs=None):
    """The same for R0 (the calibration room's Lambertian world re-projected), on a fusion.PATHS path: for the dry run."""
    P, g, W = mt.r0_world()
    fan = sn.SlitFan(width=W)
    poses = fu.path_poses(kind=kind, step=step, start=start)
    refs = refs or references(len(poses))
    looks, truth = [], {}
    for k, (s, c, R, _) in enumerate(poses):
        img, idx = mt.r0_look(P, g, c, R, fan)
        looks.append((float(s), c, R, img, None))
        if k in refs:
            truth[k] = (P, idx)
    return looks, truth, fan, refs


# ------------------------------------------------------------------------------------------------- per reference

def match_reference(looks, truth, j, fan, max_n=20000, mirror=False):
    """B with its mark at reference j over its +/-50-look window. Returns per anchor: the ray u, range r, the mark,
    surface-right (truth), psi from the reference's heading (deg), and the reference's (c, R)."""
    P, idx = truth[j]
    win = looks[j - HALF:j + HALF + 1]
    s, c, R, img, _ = win[HALF]
    win = list(win)
    win[HALF] = (s, c, R, img, idx)
    px, u, pidx, _ = mt.anchors(win, HALF, fan, max_n=max_n)
    r, conf, best = mt.sweep(win, HALF, px, u, fan, mirror=mirror)
    mk = mt.reliable(best, conf)
    nb = mt.neighbourhood_ranges(win, HALF, P, px)
    right = mt.surface_right(r, nb)
    fwd = -R[:, 2]
    psi = np.degrees(np.arccos(np.clip(u @ fwd, -1, 1)))
    return {"j": j, "c": c, "R": R, "u": u, "r": r, "mark": mk, "right": right, "psi": psi, "win": win}


# ------------------------------------------------------------------------------------------------------- fusion

def _keys(X, cell=CELL):
    return np.floor(X / cell).astype(np.int64)


def _encode(k, off=1 << 20):
    k = k + off
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def fuse(X, ref):
    """2 cm cells keyed by the estimated points X (n, 3) with their reference ids ref (n,): the fused point (per-axis
    median), the number of distinct references, the reference that contributed most."""
    code = _encode(_keys(X))
    order = np.argsort(code, kind="stable")
    cs = code[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], len(cs)]
    Xf = np.zeros((len(starts), 3))
    nref = np.zeros(len(starts), int)
    dom = np.zeros(len(starts), int)
    cell_of = np.zeros(len(X), int)
    for i, (a, b) in enumerate(zip(starts, ends)):
        mem = order[a:b]
        cell_of[mem] = i
        Xf[i] = np.median(X[mem], axis=0)
        vals, cnt = np.unique(ref[mem], return_counts=True)
        nref[i] = len(vals)
        dom[i] = vals[np.argmax(cnt)]
    fuse.cell_of = cell_of                              # the last call's membership (for E2's control)
    return Xf, nref, dom


def cells_right(Xf, dom, per_ref, truth, fan):
    """K10's surface metric for each fused point, seen from its dominant reference."""
    ok = np.zeros(len(Xf), bool)
    model = fan.model()
    for j, m in per_ref.items():
        sel = np.flatnonzero(dom == j)
        if len(sel) == 0:
            continue
        v = (Xf[sel] - m["c"]) @ m["R"]
        rc, _ = model.pixel_of(v)
        H, W = m["win"][HALF][3].shape
        pix = np.stack([np.clip(np.rint(rc[:, 0]), 0, H - 1), np.rint(rc[:, 1]) % W], -1).astype(np.int64)
        nb = mt.neighbourhood_ranges(m["win"], HALF, truth[j][0], pix)
        ok[sel] = mt.surface_right(np.linalg.norm(v, axis=-1), nb)
    return ok


def coverage(per_ref, truth, Xf_right, stride=2):
    """Share of the surface's 2 cm cells (the references' true points, |elevation| < 70 deg, every `stride` pixel)
    within 2 cm of a right fused point."""
    pts = []
    for j, m in per_ref.items():
        P, idx = truth[j]
        H, W = idx.shape
        rows = np.arange(0, H, stride)
        lat = (0.5 - (rows + 0.5) / H) * np.pi
        rows = rows[np.abs(lat) < LAT_MAX]
        q = idx[rows][:, ::stride].ravel()
        q = q[q >= 0]
        pts.append(P[q])
    S = np.concatenate(pts)
    S = S[np.all(np.isfinite(S), 1)]
    sk = _keys(S)
    code = _encode(sk)
    uc, first, inv = np.unique(code, return_index=True, return_inverse=True)
    rep = np.zeros((len(uc), 3))
    np.add.at(rep, inv, S)
    rep /= np.bincount(inv)[:, None]                    # each surface cell: the mean of its true points
    if len(Xf_right) == 0:
        return 0.0, int(len(uc))
    fk = _keys(Xf_right)
    fcode = _encode(fk)
    order = np.argsort(fcode)
    fcs = fcode[order]
    covered = np.zeros(len(uc), bool)
    rk = _keys(rep)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nc = _encode(rk + np.array([dx, dy, dz]))
                pos = np.clip(np.searchsorted(fcs, nc), 0, len(fcs) - 1)
                hit = fcs[pos] == nc
                d = np.linalg.norm(Xf_right[order[pos]] - rep, axis=-1)
                covered |= hit & (d <= CELL)
    return float(covered.mean()), int(len(uc))


# -------------------------------------------------------------------------------------------------------- stage

def run_path(looks, truth, fan, refs, max_n=20000, mirror=False):
    per_ref = {j: match_reference(looks, truth, j, fan, max_n=max_n, mirror=mirror) for j in refs}
    X, ref, pts = [], [], []
    for j, m in per_ref.items():
        mk = m["mark"]
        X.append(m["c"] + m["u"][mk] * m["r"][mk][:, None])
        ref.append(np.full(int(mk.sum()), j))
    X, ref = np.concatenate(X), np.concatenate(ref)
    fin = np.all(np.isfinite(X), 1)
    X, ref = X[fin], ref[fin]
    Xf, nref, dom = fuse(X, ref)
    cell_of = fuse.cell_of
    right = cells_right(Xf, dom, per_ref, truth, fan)
    return per_ref, X, ref, Xf, nref, dom, right, cell_of


def perturbed_confirmed(per_ref, X, ref, Xf, nref, dom, cell_of, truth, fan, seed=19, move=0.20):
    """E2's control (amended on #19 after the dry run): the confirmed cells (>= 2 references) keep their members; each
    member is moved 20 % along its ray (random sign) and the cell's median is taken again; the share still right."""
    rng = np.random.default_rng(seed)
    conf = np.flatnonzero(nref >= 2)
    if len(conf) == 0:
        return None, 0
    Xp = X.copy()
    for j, m in per_ref.items():
        q = np.flatnonzero(ref == j)
        Xp[q] = m["c"] + (X[q] - m["c"]) * (1 + move * rng.choice([-1.0, 1.0], size=len(q)))[:, None]
    Xf2 = np.stack([np.median(Xp[cell_of == i], axis=0) for i in conf])
    ok = cells_right(Xf2, dom[conf], per_ref, truth, fan)
    return float(ok.mean()), int(len(conf))


def summarise(per_ref, X, ref, Xf, nref, dom, right):
    out = {"references": {}, "cells": int(len(Xf)), "fused_right": float(right.mean()) if len(right) else None}
    for j, m in per_ref.items():
        mk = m["mark"]
        out["references"][str(j)] = {"s": float(m["win"][HALF][0]), "anchors": int(len(mk)), "kept": float(mk.mean()),
                                     "reliable_surface_right": float(m["right"][mk].mean()) if mk.any() else None,
                                     "all_surface_right": float(m["right"].mean())}
    out["cells_by_references"] = {str(n): {"cells": int((nref == n).sum()),
                                           "right": float(right[nref == n].mean()) if (nref == n).any() else None}
                                  for n in range(1, N_REFS + 1)}
    c2 = nref >= 2
    out["confirmed_cells"] = int(c2.sum())
    out["confirmed_right"] = float(right[c2].mean()) if c2.any() else None
    return out


SECTORS = {"0-15": (0, 15), "15-60": (15, 60), "60-120": (60, 120), "120-165": (120, 165), "165-180": (165, 181)}


def sector_yield(per_ref):
    """E3 (amended on #19 after the dry run): per sector, the share of all anchors that are both marked reliable and
    surface-right."""
    out = {}
    for k, (a, b) in SECTORS.items():
        good, n = 0, 0
        for m in per_ref.values():
            sel = (m["psi"] >= a) & (m["psi"] < b)
            good += int((m["mark"] & m["right"])[sel].sum())
            n += int(sel.sum())
        out[k] = {"anchors": n, "yield": good / n if n else None}
    return out


def sector_right(per_ref, reliable_only=True):
    out = {}
    for k, (a, b) in SECTORS.items():
        ok, n = 0, 0
        for m in per_ref.values():
            sel = (m["psi"] >= a) & (m["psi"] < b) & (m["mark"] if reliable_only else True)
            ok += int(m["right"][sel].sum())
            n += int(sel.sum())
        out[k] = {"n": n, "right": ok / n if n else None}
    return out


def selftest(paths, max_n=20000):
    """The checks declared on #19. paths: {"turn": (looks, truth, fan, refs), "straight": (...)}.
    E1: fused cells >= 95 % right (control: the mirrored path, < 95 %). E2: cells from >= 2 references >= 97 % right
    (control, amended: those cells' members moved 20 % along their rays, medians taken again, < 97 %). E3, amended:
    within 15 deg of the reference's heading, the turn's yield (anchors both marked reliable and surface-right) >= the
    straight path's + 10 points (control: 60-120 deg must miss the 10 points). Reported: coverage, per reference, cells by references, sectors."""
    res, num = [], {}
    sectors = {}
    for name, (looks, truth, fan, refs) in paths.items():
        per_ref, X, ref, Xf, nref, dom, right, cell_of = run_path(looks, truth, fan, refs, max_n=max_n)
        o = summarise(per_ref, X, ref, Xf, nref, dom, right)
        o["coverage"], o["surface_cells"] = coverage(per_ref, truth, Xf[right])
        o["control_E2_perturbed_right"], o["control_E2_confirmed_cells"] = perturbed_confirmed(
            per_ref, X, ref, Xf, nref, dom, cell_of, truth, fan)
        pm, Xm, rm, Xfm, nrm, dm, rtm, _ = run_path(looks, truth, fan, refs, max_n=max_n, mirror=True)
        o["control_E1_mirrored"] = {"cells": int(len(Xfm)), "right": float(rtm.mean()) if len(rtm) else None,
                                    "kept": float(np.mean(np.concatenate([m["mark"] for m in pm.values()])))}
        sectors[name] = sector_yield(per_ref)
        o["sectors_yield"] = sectors[name]
        o["sectors_reliable_right"] = sector_right(per_ref)
        num[name] = o
        res.append((f"E1 {name}: fused cells >= 95 % right", o["fused_right"] is not None and o["fused_right"] >= 0.95))
        res.append((f"E1 {name} control: the mirrored path (< 95 %)",
                     o["control_E1_mirrored"]["right"] is None or o["control_E1_mirrored"]["right"] < 0.95))
        res.append((f"E2 {name}: cells from >= 2 references >= 97 % right",
                    o["confirmed_right"] is not None and o["confirmed_right"] >= 0.97))
        res.append((f"E2 {name} control: their members moved 20 % along their rays (< 97 %)",
                    o["control_E2_perturbed_right"] is None or o["control_E2_perturbed_right"] < 0.97))
    if "turn" in sectors and "straight" in sectors:
        def diff(k):
            a, b = sectors["turn"][k]["yield"], sectors["straight"][k]["yield"]
            return None if a is None or b is None else a - b
        d0, d6 = diff("0-15"), diff("60-120")
        num["E3"] = {"heading_0_15_turn_minus_straight": d0, "lateral_60_120_turn_minus_straight": d6}
        res.append(("E3: within 15 deg of the heading, the turn's yield (reliable and right) >= the straight's + 10 points",
                    d0 is not None and d0 >= 0.10))
        res.append(("E3 control: the lateral sector 60-120 deg (< 10 points)", d6 is None or d6 < 0.10))
    return res, num
