"""Synthetic fixtures for M3's matcher (M3a #6, Stage 3), numpy only.

Analytic planar scenes ray-cast through the Foveated warp: each native pixel is the mean of SS x SS
sub-pixel rays (the renderer's box filter), so the images carry the sensor's real sampling. Truth:
the point where each pixel's **centre** ray meets the scene (the matcher anchors there), its range
from b, and its exact log-tangent displacement. Optional Gaussian noise (fixed seed) models render
noise.

The poses follow the dolly: eye axis along world +Y, world +Z up; look b is B ahead of look a.

Fixtures (declared before measuring; the gates are in GATES):
    front      a textured wall facing the eye, 3 m ahead
    tilted     the same texture on a plane turned 50 deg about the vertical, 3 m ahead on the axis
    corridor   floor, ceiling and side walls parallel to the heading, end wall 8 m ahead (the A1 case)
    radial     a wall 3 m ahead whose texture varies only in azimuth about the axis: edges along radial
               lines (the A2 case); must be rejected, not matched
    periodic   a wall 3 m ahead whose texture is periodic in log radius (period about 3 native px at
               20 deg): repeated texture, must be rejected or matched right
    occlusion  a board at 1.5 m in front of a wall at 5 m: depth edges
"""
from __future__ import annotations

import numpy as np

from fov_render import warp

from . import logpolar as lp

R_EYE = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # eye -Z -> world +Y, eye +Y -> world +Z
SS = 4


def _texture_noise(seed, k=48, f_lo=1.5, f_hi=120.0):
    rng = np.random.default_rng(seed)
    f = np.exp(rng.uniform(np.log(f_lo), np.log(f_hi), k))
    th = rng.uniform(0, 2 * np.pi, k)
    ph = rng.uniform(0, 2 * np.pi, k)
    amp = 1.0 / np.sqrt(f / f_lo) ** 0.5
    amp /= np.sqrt(np.sum(amp ** 2) / 2)

    def tex(s, t):
        x = (np.cos(th) * s[..., None] + np.sin(th) * t[..., None]) * f
        return 0.5 + 0.18 * np.sum(amp * np.sin(2 * np.pi * x + ph), -1)
    return tex


def _plane(p0, n, t1, tex, label, bounds=None):
    n, t1 = lp.unit(n), lp.unit(t1)
    return {"p0": np.asarray(p0, float), "n": n, "t1": t1, "t2": np.cross(n, t1), "tex": tex, "label": label,
            "bounds": bounds}


def scene(name: str, seed: int = 0) -> list:
    tn = _texture_noise(seed)
    Y = np.array([0, 1.0, 0])
    Z = np.array([0, 0, 1.0])
    X = np.array([1.0, 0, 0])
    if name == "front":
        return [_plane([0, 3, 0], -Y, X, tn, "front")]
    if name == "tilted":
        a = np.radians(50)
        n = -np.cos(a) * Y + np.sin(a) * X
        return [_plane([0, 3, 0], n, np.cross(Z, n), tn, "tilted")]
    if name == "corridor":
        tex = [_texture_noise(seed + i) for i in range(5)]
        return [_plane([0, 0, -1.0], Z, X, tex[0], "parallel"), _plane([0, 0, 1.4], -Z, X, tex[1], "parallel"),
                _plane([-1.5, 0, 0], X, Y, tex[2], "parallel"), _plane([1.5, 0, 0], -X, Y, tex[3], "parallel"),
                _plane([0, 8, 0], -Y, X, tex[4], "front_far")]
    if name == "radial":
        def tex(s, t):
            return 0.5 + 0.35 * np.sin(90 * np.arctan2(t, s))
        return [_plane([0, 3, 0], -Y, X, tex, "radial")]
    if name == "periodic":
        lam = 0.09                                  # in q: about 3 native px at 20 deg (34 px per unit q)

        def tex(s, t):
            return 0.5 + 0.35 * np.sin(2 * np.pi * np.log(np.hypot(s, t) + 1e-9) / lam)
        return [_plane([0, 3, 0], -Y, X, tex, "periodic")]
    if name == "occlusion":
        t2 = _texture_noise(seed + 7)
        return [_plane([0, 5, 0], -Y, X, tn, "far"),
                _plane([0.45, 1.5, 0.1], -Y, X, t2, "board", bounds=(-0.35, 0.35, -0.3, 0.3))]
    raise ValueError(name)


def _hit(planes, c, dw):
    """Nearest hit along world rays dw from c: (range, plane index, texture value)."""
    best = np.full(dw.shape[:-1], np.inf)
    idx = np.full(dw.shape[:-1], -1)
    val = np.full(dw.shape[:-1], np.nan)
    for i, p in enumerate(planes):
        den = dw @ p["n"]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((p["p0"] - c) @ p["n"]) / den
        X = c + t[..., None] * dw
        s_, t_ = (X - p["p0"]) @ p["t1"], (X - p["p0"]) @ p["t2"]
        ok = np.isfinite(t) & (t > 1e-6) & (t < best)
        if p["bounds"] is not None:
            s0, s1, t0, t1 = p["bounds"]
            ok &= (s_ >= s0) & (s_ <= s1) & (t_ >= t0) & (t_ <= t1)
        best = np.where(ok, t, best)
        idx = np.where(ok, i, idx)
        val = np.where(ok, p["tex"](s_, t_), val)
    return best, idx, val


def _subpixel_dirs(n, e2, emax, ss):
    """Eye-frame directions of ss x ss sub-pixel rays of every pixel: (n, n, ss*ss, 3)."""
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    o = (np.arange(ss) + 0.5) / ss - 0.5
    oi, oj = np.meshgrid(o, o, indexing="ij")
    px = (j[..., None] + 0.5 + oj.reshape(-1)) / n - 0.5
    py = 0.5 - (i[..., None] + 0.5 + oi.reshape(-1)) / n
    return warp.warp_direction(px, py, e2, emax) * np.array([1.0, 1.0, -1.0])


def render(model, planes, c, ss: int = SS, noise: float = 0.0, seed: int = 0, roll_deg: float = 0.0):
    """Luminance raster (nan off the disk) of the look at eye centre c. roll_deg turns the eye about its
    own axis (the wrong-azimuth control)."""
    Rr = R_EYE
    if roll_deg:
        t = np.radians(roll_deg)
        Rr = R_EYE @ np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    d = _subpixel_dirs(model.n, model.e2, model.emax, ss)
    _, _, v = _hit(planes, np.asarray(c, float), d @ Rr.T)
    img = np.nanmean(v, -1) if np.isfinite(v).any() else np.full(v.shape[:-1], np.nan)
    img = np.where(model.rays().valid & np.isfinite(v).all(-1), img, np.nan)
    if noise:
        img = img + np.random.default_rng(seed).normal(scale=noise, size=img.shape)
    return img


def truth(model, planes, ca, cb):
    """Per native pixel of b (flat raster order): range from b along the centre ray, plane label index,
    and the exact log-tangent displacement d = q_b - q_a of that point."""
    rays = model.rays()
    db = rays.direction.reshape(-1, 3)
    r, idx, _ = _hit(planes, np.asarray(cb, float), db @ R_EYE.T)
    P = cb + r[:, None] * (db @ R_EYE.T)
    with np.errstate(invalid="ignore"):
        da = lp.project_static(P, ca, R_EYE)
    qa, _ = lp.qphi(da)
    qb, _ = lp.qphi(db)
    vis_a, _, _ = _hit(planes, np.asarray(ca, float), lp.unit(P - ca))
    visible = np.isfinite(r) & (np.abs(vis_a - np.linalg.norm(P - ca, axis=-1)) < 1e-6)
    return {"range": r, "label": idx, "d": qb - qa, "q_a": qa, "visible_in_a": visible}


# ----------------------------------------------------------------------------- the Stage 3 harness

GATE = {"median_px": 0.10, "p90_px": 0.25, "min_accepted": 100}     # #6 Stage 3, fixed before measuring
NOISE = 0.005                    # declared: about 1 % of the mean luminance (0.5)
BASELINES = (0.0989565, 0.4947826, 0.989565)     # the dolly's strides 1, 5, 10 (calibration path: 0.1 m steps)
GATED = {"front": ("front",), "tilted": ("tilted",), "corridor": ("parallel", "front_far"), "occlusion": ("far", "board")}


def _stats(err, acc, sel):
    n = int(sel.sum())
    a = acc & sel
    ae = np.abs(err[a])
    return {"anchors": n, "accepted": int(a.sum()), "accepted_fraction": float(a.sum() / n) if n else None,
            "median_px": float(np.median(ae)) if ae.size else None, "p90_px": float(np.quantile(ae, 0.9)) if ae.size else None,
            "over_1px_of_accepted": float(np.mean(ae > 1)) if ae.size else None,
            "within_0p25px_of_anchors": float(np.sum(ae <= 0.25) / n) if n else None}


def _gate(st) -> bool:
    return (st["accepted"] >= GATE["min_accepted"] and st["median_px"] <= GATE["median_px"]
            and st["p90_px"] <= GATE["p90_px"])


def _edge_mask(labels_flat, n, radius=3):
    L = labels_flat.reshape(n, n)
    edge = np.zeros_like(L, bool)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            S = np.roll(np.roll(L, di, 0), dj, 1)
            edge |= (S != L) & (S >= 0) & (L >= 0)
    return edge.reshape(-1)


def case(model, name, B, *, noise=NOISE, settings=None, scramble=False, roll_deg=0.0, seed=0):
    from . import matcher
    planes = scene(name, seed)
    ca = np.zeros(3)
    cb = ca + np.array([0.0, B, 0.0])
    Ia = render(model, planes, ca, noise=noise, seed=seed + 11, roll_deg=roll_deg)
    Ib = render(model, scene(name, seed + 100) if scramble else planes, cb, noise=noise, seed=seed + 12)
    r = matcher.match_pair(Ia, Ib, model, B, settings)
    tr = truth(model, planes, ca, cb)
    f = r["flat"]
    err = (r["d"] - tr["d"][f]) * r["px_per_q_a"]
    acc = r["reason"] == 0
    labs = [p["label"] for p in planes]
    lab = np.array([labs[i] if i >= 0 else "none" for i in tr["label"][f]])
    edge = _edge_mask(tr["label"], model.n)[f]
    out = {"fixture": name, "B_m": B, "noise": noise, "scramble": scramble, "roll_deg": roll_deg,
           "stretch": (settings or {}).get("stretch", True),
           "reasons": {matcher.REASONS[k]: int((r["reason"] == k).sum()) for k in range(len(matcher.REASONS))},
           "all": _stats(err, acc, np.ones_like(acc)), "by_label": {}, "by_eccentricity": {},
           "median_kappa_accepted": float(np.median(r["kappa"][acc])) if acc.any() else None,
           "median_sigma_px_accepted": float(np.median(r["sigma_px"][acc])) if acc.any() else None}
    for L in sorted(set(lab)):
        out["by_label"][L] = _stats(err, acc, lab == L)
    if name == "occlusion":
        out["edge"] = _stats(err, acc, edge)
        out["interior"] = _stats(err, acc, ~edge)
    e = np.degrees(r["e_b"])
    for lo, hi in ((1, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 44)):
        out["by_eccentricity"][f"{lo}-{hi}"] = _stats(err, acc, (e >= lo) & (e < hi))
    return out


def run_fixtures(model, log=print) -> dict:
    """Every declared case, the gates of #6 Stage 3 and their controls."""
    cases, gates = [], []
    for name in ("front", "tilted", "corridor"):
        for B in BASELINES:
            c = case(model, name, B)
            cases.append(c)
            for L in GATED[name]:
                st = c["by_label"][L]
                gates.append({"fixture": name, "label": L, "B_m": B, "ok": _gate(st), **{k: st[k] for k in
                              ("accepted_fraction", "median_px", "p90_px", "over_1px_of_accepted")}})
            log(f"[fixtures] {name} B {B:.3f}: " + "; ".join(
                f"{L} acc {c['by_label'][L]['accepted_fraction']:.3f} med {c['by_label'][L]['median_px']:.3f} p90 {c['by_label'][L]['p90_px']:.3f}"
                for L in GATED[name]))
    for name in ("radial", "periodic", "occlusion"):
        c = case(model, name, BASELINES[1])
        cases.append(c)
        log(f"[fixtures] {name} B {BASELINES[1]:.3f}: accepted {c['all']['accepted_fraction']:.3f}, "
            f">1 px of accepted {c['all']['over_1px_of_accepted']}, reasons {c['reasons']}")
    occ = cases[-1]
    gates.append({"fixture": "occlusion", "label": "interior", "B_m": BASELINES[1], "ok": _gate(occ["interior"]),
                  **{k: occ["interior"][k] for k in ("accepted_fraction", "median_px", "p90_px", "over_1px_of_accepted")}})
    # controls of #6 Stage 3: each must fail the gate (or, for shift-only, do worse than shift + stretch)
    ctrl = []
    c = case(model, "front", BASELINES[1], scramble=True)
    ctrl.append({"control": "scrambled frame b (another texture)", "fails": not _gate(c["all"]), "stats": c["all"]})
    c = case(model, "front", BASELINES[1], roll_deg=1.0)
    ctrl.append({"control": "wrong azimuth: a's eye rolled 1 deg", "fails": not _gate(c["all"]), "stats": c["all"]})
    st_stretch = next(x for x in cases if x["fixture"] == "corridor" and x["B_m"] == BASELINES[1])["by_label"]["parallel"]
    c = case(model, "corridor", BASELINES[1], settings={"stretch": False})
    st_shift = c["by_label"]["parallel"]
    ctrl.append({"control": "shift-only matcher on the corridor's parallel surfaces (stride 5): fewer matches within 0.25 px",
                 "fails": st_shift["within_0p25px_of_anchors"] < st_stretch["within_0p25px_of_anchors"],
                 "stats": st_shift, "shift_plus_stretch": st_stretch})
    for x in ctrl:
        log(f"[fixtures] control {'fails' if x['fails'] else 'DOES NOT FAIL'}: {x['control']}: {x['stats']}")
    return {"gate": GATE, "noise": NOISE, "baselines": list(BASELINES), "gates": gates, "controls": ctrl, "cases": cases,
            "ok": all(g["ok"] for g in gates), "controls_fail": all(x["fails"] for x in ctrl)}
