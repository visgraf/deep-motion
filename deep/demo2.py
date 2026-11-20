"""Demo D2 (#23): the perfect and the real matcher side by side along #9's L-tour (DERIVED: a display).

Both matchers exactly as in the assessment (#22): references every 10 looks along the tour with a full +/-50-look
window; the perfect matcher (5a's oracle fit, 0.5 px) on a 2-px grid; the real pipeline B + H. A reference's points
are shown from look m + 50 on (its window complete), on both sides. Each side's points so far are fused into 2 cm cells
(stage 8) and judged against the truth (stage 8's metric).

    +--------------------------------+--------------------------------+
    | the current look, real cells   | progress: coverage, cells right |
    +--------------------------------+--------------------------------+
    | 3-D: the perfect matcher       | 3-D: the real matcher (B + H)  |
    +--------------------------------+--------------------------------+

Two videos: `match` (right cells by height on #9's blue ramp, wrong ones in the accent) and `error` (each cell's
relative error, light 0 -> dark >= 10 %). The videos stay on the workstation.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from . import assess as asx
from . import demo as dm
from . import e2e
from . import fusion as fu
from . import match as mt
from . import oracle as orc
from . import sensor as sn

W, H, FPS = dm.W, dm.H, dm.FPS
EVERY, LAG = 10, e2e.HALF
ERR_MAX = 0.10
MAX_DRAW = 60000                     # display only: at most this many cells drawn per side (a fixed subset)
PANO_BOX, PROG_BOX = (0, 0, 960, 480), (960, 0, 960, 480)
LEFT_BOX, RIGHT_BOX = (0, 480, 960, 600), (960, 480, 960, 600)
SIDES = ("perfect", "real")


def tour_references(K, every=EVERY, half=LAG):
    return list(range(half, K - half, every))


# ------------------------------------------------------------------------------------------------------ the tour

def render_tour(straight_dir, turn_dir, join_s=2.0):
    """#9's tour from two render runs: T (dm.tour), looks [(s, c, R, img, None)], truth at the references, the visibility
    function, the fan."""
    T = dm.tour([straight_dir], [turn_dir], join_s)
    _, _, width = fu.run_meta([turn_dir])
    fan = sn.SlitFan(width=width)
    refs = tour_references(len(T))
    rec_mid = orc.record(T[len(T) // 2][0])
    hit = np.isfinite(rec_mid.distance_m)
    scale = float(np.quantile(rec_mid.value[hit].astype(np.float64) @ mt.LUM, 0.99))
    Hh, Ww = (int(x) for x in rec_mid.raster_shape)
    looks, truth = [], {}
    for k, t in enumerate(T):
        rec = orc.record(t[0])
        g = rec.value.astype(np.float64) @ mt.LUM / scale
        ok = np.isfinite(rec.distance_m)
        g[~ok] = np.nan
        looks.append((t[3], np.asarray(t[1], float), np.asarray(t[2], float), g.reshape(Hh, Ww).astype(np.float32), None))
        if k in refs:
            truth[k] = (rec.position_w.astype(np.float64), np.where(ok, np.arange(Hh * Ww), -1).reshape(Hh, Ww))

    def vis(k, P):
        return orc.visible(orc.record(T[k][0]), looks[k][1], looks[k][2], P, fan)
    return T, looks, truth, vis, fan, refs


def r0_tour(join_s=2.0):
    """The dry run's tour in R0's calibration room: a straight 2 m from (0, -2) heading +Y, then turn90 from the EYE."""
    P, g, Wd = mt.r0_world()
    fan = sn.SlitFan(width=Wd)
    st = [p for p in fu.path_poses(kind="straight_long", step=0.01, start={"p0": (0.0, -join_s)}) if p[0] < join_s - 1e-6]
    tu = fu.path_poses(kind="turn90", step=0.01)
    ends = np.cumsum([L for _, L, _ in fu.PATHS["turn90"]])
    T = [(None, c, R, float(s), 0) for s, c, R, _ in st]
    T += [(None, c, R, join_s + float(s), 1 + int(np.searchsorted(ends - 1e-9, s))) for s, c, R, _ in tu]
    refs = tour_references(len(T))
    looks, truth, idxs = [], {}, []
    for k, t in enumerate(T):
        img, idx = mt.r0_look(P, g, t[1], t[2], fan)
        looks.append((t[3], t[1], t[2], img, None))
        idxs.append(idx.astype(np.int32))
        if k in refs:
            truth[k] = (P, idx)
    res = 2 * np.pi / Wd

    def vis(k, Pq):
        c, R = looks[k][1], looks[k][2]
        rc, _ = fan.model().pixel_of((Pq - c) @ R)
        i = np.rint(rc[:, 0]).astype(np.int64)
        j = np.rint(rc[:, 1]).astype(np.int64) % Wd
        ok = (i >= 0) & (i < idxs[k].shape[0])
        q = np.full(len(Pq), -1)
        q[ok] = idxs[k][i[ok], j[ok]]
        err = np.where(q >= 0, np.linalg.norm(P[np.maximum(q, 0)] - Pq, axis=-1), np.inf)
        return (q >= 0) & (err <= orc.TOL_PX * res * np.linalg.norm(Pq - c, axis=-1))
    return T, looks, truth, vis, fan, refs


def check_continuity_r0(T, step=0.01, shift=None):
    """V2 for the dry run (no run.json): arc length advances by `step` and the straight reaches the turn's start."""
    C = np.stack([t[1] for t in T]).copy()
    seg = np.array([t[4] for t in T])
    if shift is not None:
        C[seg > 0] += shift
    s = np.array([t[3] for t in T])
    kt = int(np.argmax(seg > 0))
    jump = float(np.linalg.norm(C[kt] - C[kt - 1]))
    ok = bool(np.all(np.abs(np.diff(s) - step) <= 1e-9)) and abs(jump - step) <= 1e-6
    return ok, {"join_step_m": jump, "looks": len(T)}


# --------------------------------------------------------------------------------------- both matchers, per ref

def per_reference(looks, truth, vis_fn, fan, refs, seed=23):
    """Each reference's points on both sides, as in #22: {j: (perfect points, real points)}, and the real matcher's
    per-reference structures (for the metric)."""
    rng = np.random.default_rng(seed)
    per_ref = {}
    pts = {}
    for j in refs:
        per_ref[j] = e2e.match_reference(looks, truth, j, fan, max_n=10 ** 9)
        pts[j] = asx.oracle_points(truth, j, looks, fan)
        print(f"[deep] demo2 reference {j}: real matched", flush=True)
    vis = asx.visibility(looks, refs, pts, vis_fn)
    out = {}
    for j in refs:
        Xp, us = asx.oracle_fit(looks, j, pts[j], vis[j], rng)
        Xr = asx.b_plus_h_points(per_ref[j], fan)
        out[j] = (Xp[us], Xr[np.all(np.isfinite(Xr), 1)])
        print(f"[deep] demo2 reference {j}: {int(us.sum())} perfect, {len(out[j][1])} real points", flush=True)
    return out, per_ref


def cells_error(Xf, dom, per_ref, truth, fan):
    """Stage 8's metric's quantity: each fused point's smallest relative range error over the 3x3 neighbourhood of its
    pixel, seen from its dominant reference (right iff <= 10 %)."""
    err = np.full(len(Xf), np.inf)
    model = fan.model()
    for j, m in per_ref.items():
        sel = np.flatnonzero(dom == j)
        if len(sel) == 0:
            continue
        v = (Xf[sel] - m["c"]) @ m["R"]
        rc, _ = model.pixel_of(v)
        Hh, Ww = m["win"][e2e.HALF][3].shape
        pix = np.stack([np.clip(np.rint(rc[:, 0]), 0, Hh - 1), np.rint(rc[:, 1]) % Ww], -1).astype(np.int64)
        nb = mt.neighbourhood_ranges(m["win"], e2e.HALF, truth[j][0], pix)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(np.linalg.norm(v, axis=-1)[:, None] / nb - 1)
        err[sel] = np.nanmin(np.where(np.isfinite(rel), rel, np.inf), axis=1)
    return err


def fuse_side(points, released):
    if not released:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int)
    X = np.concatenate([points[j] for j in released])
    ref = np.concatenate([np.full(len(points[j]), j) for j in released])
    return e2e.fuse(X, ref)


def released_at(t, refs, lag=LAG):
    return [j for j in refs if j + lag <= t]


def check_release(log, lag=LAG):
    """V4: every reference shown at look t satisfies m + lag <= t."""
    bad = [(t, j) for t, js in log for j in js if j + lag > t]
    return len(bad) == 0, len(bad)


# --------------------------------------------------------------------------------------------------- the frames

def pano(looks, m, Xr, colours, fan, box=PANO_BOX):
    x0, y0, w, h = box
    img = looks[m][3]
    Hf, Wf = img.shape
    rows = (np.arange(h) * Hf / h).astype(int)
    cols = (np.arange(w) * Wf / w).astype(int)
    g = np.nan_to_num(img[rows][:, cols], nan=0.0)
    a = np.repeat((np.clip(g, 0, 1) ** (1 / 2.2) * 255 * 0.55 + 0.5).astype(np.uint8)[..., None], 3, axis=2)
    c, R = looks[m][1], looks[m][2]
    if len(Xr):
        v = (Xr - c) @ R
        rc, _ = fan.model().pixel_of(v)
        rng = np.linalg.norm(v, axis=-1)
        o = np.argsort(-rng)                                       # far first: the nearest cell wins its pixel
        o = o[rng[o] < 30]
        py = (rc[o, 0] * h / Hf).astype(int).clip(0, h - 2)
        px = ((rc[o, 1] % Wf) * w / Wf).astype(int).clip(0, w - 2)
        a[py, px] = colours[o]
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    dm._text(d, (14, 8), "The current look (360° × 180°), with the real map's cells", 18, dm.LIGHT)
    return im


def prog(hist, total_s, box=PROG_BOX, seg_s=None):
    x0, y0, w, h = box
    im = Image.new("RGB", (w, h), dm.PAPER)
    d = ImageDraw.Draw(im)
    L, R_, T_, B_ = 70, w - 40, 70, h - 110
    dm._text(d, (16, 10), "As the tour goes: coverage of the surface (solid) and cells right (dashed)", 18)
    for v in (0, 25, 50, 75, 100):
        y = B_ - (B_ - T_) * v / 100
        d.line([(L, y), (R_, y)], fill=(225, 225, 220), width=1)
        dm._text(d, (L - 8, y), f"{v} %", 14, dm.MUTED, anchor="rm")
    for v in range(0, int(total_s) + 1):
        dm._text(d, (L + (R_ - L) * v / total_s, B_ + 10), f"{v}", 14, dm.MUTED, anchor="mt")
    dm._text(d, (R_, B_ + 34), "metres along the tour", 14, dm.MUTED, anchor="rt")
    if seg_s is not None:
        xj = L + (R_ - L) * seg_s / total_s
        d.line([(xj, T_), (xj, B_)], fill=dm.RULE, width=1)
        dm._text(d, (xj + 6, T_ + 2), "the turn", 14, dm.MUTED)
    cols = {"perfect": tuple(int(v) for v in dm._STOPS[6]), "real": dm.INK}
    for side in SIDES:
        pts_c = [(L + (R_ - L) * s / total_s, B_ - (B_ - T_) * hc[side][0]) for s, hc in hist]
        pts_r = [(L + (R_ - L) * s / total_s, B_ - (B_ - T_) * hc[side][1]) for s, hc in hist if hc[side][1] is not None]
        if len(pts_c) > 1:
            d.line(pts_c, fill=cols[side], width=3)
        for a_, b_ in zip(pts_r[::2], pts_r[1::2]):
            d.line([a_, b_], fill=cols[side], width=2)
    if hist:
        s, hc = hist[-1]
        yy = h - 64
        for i, side in enumerate(SIDES):
            c, r = hc[side]
            txt = f"{side}: coverage {100 * c:.1f} %" + (f", cells right {100 * r:.1f} %" if r is not None else "")
            d.rectangle([L, yy + 22 * i + 4, L + 14, yy + 22 * i + 16], fill=cols[side])
            dm._text(d, (L + 22, yy + 22 * i), txt, 16, dm.INK)
    return im


def cloud(sc, X, colours, az, size, title, m=None, azs=None, band=40):
    """dm.cloud_panel below a title band, without its path (drawn here in ink: orange means a wrong cell here)."""
    w, h = size
    st = SimpleNamespace(X=X, usable=np.ones(len(X), bool))
    inner = dm.cloud_panel(sc, st, colours, az=az, size=(w, h - band), title=False, m=None, azs=azs)
    d = ImageDraw.Draw(inner)
    if m is not None:
        k, xc, yc = dm._fit(sc, azs or (az,), w, h - band)[az]
        eye, B = dm._view(sc, az)
        q = (sc.C[: m + 1] - eye) @ B.T
        pts = [(w / 2 + k * (x / z - xc), (h - band) / 2 - k * (y / z - yc)) for x, y, z in q if z > 0.1]
        if len(pts) > 1:
            d.line(pts, fill=dm.INK, width=3)
    im = Image.new("RGB", (w, h), dm.PAPER)
    im.paste(inner, (0, band))
    dm._text(ImageDraw.Draw(im), (16, 10), title, 18)
    return im


def colours_for(X, err, kind, sc):
    if kind == "error":
        return dm.ramp(np.clip(err / ERR_MAX, 0, 1))
    z = (X[:, 2] - sc.zlo) / max(sc.zhi - sc.zlo, 1e-6)
    col = dm.ramp(np.clip(z, 0, 1))
    col[~(err <= ERR_MAX)] = np.array(dm.ACCENT, np.uint8)
    return col


def subset(n, k=MAX_DRAW, seed=5):
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def frame(sc, looks, m, maps, kind, hist, total_s, seg_s, fan, titles):
    im = Image.new("RGB", (W, H), dm.PAPER)
    Xr, er = maps["real"][0], maps["real"][1]
    im.paste(pano(looks, m, Xr, colours_for(Xr, er, kind, sc), fan), PANO_BOX[:2])
    im.paste(prog(hist, total_s, seg_s=seg_s), PROG_BOX[:2])
    for side, box in (("perfect", LEFT_BOX), ("real", RIGHT_BOX)):
        X, e = maps[side][0], maps[side][1]
        keep = subset(len(X))
        im.paste(cloud(sc, X[keep], colours_for(X[keep], e[keep], kind, sc), -115.0, box[2:], titles[side], m=m),
                 box[:2])
    d = ImageDraw.Draw(im)
    d.line([(0, 480), (W, 480)], fill=dm.RULE, width=1)
    d.line([(960, 0), (960, H)], fill=dm.RULE, width=1)
    t = sc.T[m]
    key = ("colour: right cells by height (dark = high), wrong cells in orange" if kind == "match" else
           f"colour: relative error along the ray, light 0 → dark ≥ {100 * ERR_MAX:g} %")
    dm._text(d, (976, 34), f"{dm.SEG_NAMES[t[4]]} · {t[3]:.2f} m · look {m + 1} of {len(sc.T)} · {key}", 14,
             dm.MUTED)
    return im


# ------------------------------------------------------------------------------------------------------ the run

def run(T, looks, truth, vis_fn, fan, refs, out, checks_v2, limit=None, best=False, git=None, videos=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    points, per_ref = per_reference(looks, truth, vis_fn, fan, refs)
    uc, rep, _ = asx.surface(per_ref, truth)
    perfect_pts = {j: points[j][0] for j in refs}
    real_pts = {j: points[j][1] for j in refs}
    allP = np.concatenate([perfect_pts[j] for j in refs])
    sc = dm.Scene(allP[subset(len(allP), 200000)], T, fan)
    total_s = sc.s[-1] + 0.02
    seg_s = next((t[3] for t in T if t[4] > 0), None)
    K = len(T) if not limit else min(limit, len(T))
    log, hist = [], []
    maps = {s: (np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)) for s in SIDES}
    cur = None
    encs = {k: dm._Enc(out / f"demo2_{k}.mp4", best) for k in ("match", "error")} if videos else {}
    last = {}
    titles = {}
    for m in range(K):
        rel = released_at(m, refs)
        log.append((m, list(rel)))
        if rel != cur:
            cur = list(rel)
            for side, pts in (("perfect", perfect_pts), ("real", real_pts)):
                Xf, nref, dom = fuse_side(pts, rel)
                err = cells_error(Xf, dom, per_ref, truth, fan) if len(Xf) else np.zeros(0)
                maps[side] = (Xf, err, nref)
        hc = {}
        for side in SIDES:
            Xf, err, _ = maps[side]
            right = err <= ERR_MAX
            hc[side] = (float(asx.covered(rep, Xf[right]).mean()) if len(Xf) else 0.0,
                        float(right.mean()) if len(Xf) else None)
        hist.append((T[m][3], hc))
        for side in SIDES:
            n = len(maps[side][0])
            titles[side] = (f"{'Perfect matcher (oracle, 0.5 px)' if side == 'perfect' else 'Real matcher (B + H)'}: "
                            f"{n:,} cells" + (f", {100 * hc[side][1]:.1f} % right" if hc[side][1] is not None else ""))
        for kind in encs:
            im = frame(sc, looks, m, maps, kind, hist, total_s, seg_s, fan, titles)
            encs[kind].put(im, int(dm.HOLD_START * FPS) if m == 0 else 1)
            last[kind] = im
        if m % 20 == 0:
            print(f"[deep] demo2 look {m + 1}/{K}: references shown {len(rel)}", flush=True)
    for kind, im in last.items():
        im.save(out / f"last_{kind}.png")
        encs[kind].put(im, int(dm.HOLD_END * FPS))
        if not limit:
            n = int(dm.ORBIT_S * FPS)
            azs = tuple(-115.0 + 360.0 * i / n for i in range(n))
            for i in range(n):
                card = Image.new("RGB", (W, H), dm.PAPER)
                for side, x0 in (("perfect", 0), ("real", 960)):
                    X, e = maps[side][0], maps[side][1]
                    keep = subset(len(X))
                    card.paste(cloud(sc, X[keep], colours_for(X[keep], e[keep], kind, sc), azs[i], (960, H),
                                     titles[side], m=len(T) - 1, azs=azs), (x0, 0))
                encs[kind].put(card)
        encs[kind].close()
    # V3: the last frame's maps against the batch fusion of every reference (control: without the last one)
    v3, v3c = {}, {}
    for side, pts in (("perfect", perfect_pts), ("real", real_pts)):
        Xb, nb_, db = fuse_side(pts, refs)
        Xc, _, _ = fuse_side(pts, refs[:-1])
        Xl = maps[side][0]
        same = len(Xl) == len(Xb) and (len(Xb) == 0 or float(np.max(np.abs(Xl - Xb))) <= 1e-12)
        v3[side] = bool(same and K == len(T))
        v3c[side] = len(Xl) == len(Xc) and (len(Xc) == 0 or float(np.max(np.abs(Xl - Xc))) <= 1e-12)
    ok4, n4 = check_release(log)
    log0 = [(t, released_at(t, refs, lag=0)) for t, _ in log]
    ok4c, n4c = check_release(log0)
    (ok2, n2), (ok2c, n2c) = checks_v2
    end = hist[-1][1]
    rec = {"command": "demo2", "looks": K, "tour_looks": len(T), "references": refs, "lag_looks": LAG,
           "end": {s: {"coverage": end[s][0], "cells_right": end[s][1], "cells": int(len(maps[s][0]))} for s in SIDES},
           "progress": [(s, {k: list(v) for k, v in hc.items()}) for s, hc in hist],
           "frames": {k: e.n for k, e in encs.items()}, "fps": FPS, "display_max_cells_per_side": MAX_DRAW,
           "computed_by": git,
           "checks": [{"name": "V2 the tour is the planned L-curve sampled evenly", "ok": ok2, **n2},
                      {"name": "V2 control: the turn's looks shifted 1 cm sideways", "ok": not ok2c, **n2c},
                      {"name": "V3 the last frame's maps equal the batch fusion of all references (both sides)",
                       "ok": all(v3.values())},
                      {"name": "V3 control: the batch without the last reference differs (both sides)",
                       "ok": not any(v3c.values())},
                      {"name": "V4 nothing shown before its window is complete (m + 50 <= t)", "ok": ok4, "violations": n4},
                      {"name": "V4 control: a schedule releasing at m fails the test", "ok": not ok4c,
                       "violations": n4c}]}
    (out / ("demo2.json" if T[0][0] is not None else "demo2_r0.json")).write_text(json.dumps(rec, indent=1, default=float))
    for c in rec["checks"]:
        print(f"[deep] demo2 {'PASS' if c['ok'] else 'FAIL'} {c['name']}")
    return 0 if all(c["ok"] for c in rec["checks"]) else 1
