"""Stage 5a (#5): the oracle evaluation on renders, short windows, line, arc and pivot (Chat sandbox).

Plan: docs/plan.md, stage 5; P-D1 (the Panorama carrier), P-D2 (5a Chat, 5b Code).

A perfect matcher (as M4's O-M): each hit of the reference look (the window's middle) has a true point P (Position);
it is projected into every other look, and kept where that look sees the same surface (M1.3's agreement, 2 px at the
range). There its bearing and elevation (the Panorama's lon, lat) are exact, or carry Gaussian noise of sigma pixels.
Range r along the anchor ray (from the reference eye toward P) is fitted by Gauss-Newton through the known poses,
with analytic derivatives: v_k(r) = R_k^T (c_ref + r u - c_k), lon = atan2(v_x, -v_z), lat = asin(v_y / |v|).
Fisher: sigma_r = sigma_ang / sqrt(sum_k (d lon/dr)^2 + (d lat/dr)^2), sigma_ang = sigma px x the pixel's angle.
Usable: a 95 % interval of at most +/-10 % of the range (as M4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import atlas as at
from . import primitives as pr
from . import sensor as sn

Z95 = 1.959963984540054
USABLE_REL = 0.10
TOL_PX = 2.0
SIGMAS_PX = (0.25, 0.5, 1.0)
SECTORS = {"heading ±15°": (0.0, 15.0), "lateral 60–120°": (60.0, 120.0), "behind ≥165°": (165.0, 180.1), "all": (0.0, 180.1)}
EYE_BAND_DEG = 10.0
# Sectors are reported three ways (#5 named the sectors, not the angle): by psi, the 3-D angle between the anchor's
# ray and the heading (the line's blind cone is psi -> 0); by bearing at eye height (|lat| < 10 deg, stage 3's atlas);
# and by bearing over all elevations (floor and ceiling anchors straight ahead are far from the heading in 3-D).


def run_poses(kind, n=21, ds=0.01, dang=np.radians(1.0), R=1.6, p0=(0.0, 0.0), phi0=np.pi / 2, h=1.6):
    """[(s, c, R)] for the stage-5a runs, from the calibration room's EYE heading +Y."""
    if kind == "line":
        pc, ss = pr.Piece("line", (n - 1) * ds), np.arange(n) * ds
    elif kind == "arc":
        pc, ss = pr.Piece("arc", (n - 1) * ds, 1.0 / R), np.arange(n) * ds
    elif kind == "pivot":
        pc, ss = pr.Piece("pivot", (n - 1) * dang), np.arange(n) * dang
    else:
        raise ValueError(kind)
    return [(float(s), *pc.pose(p0, phi0, h, s)[:2]) for s in ss]


def lonlat_and_derivs(c_ref, u, r, c_k, R_k):
    """lon, lat of c_ref + r u seen from (c_k, R_k), and their derivatives in r (analytic). u: (N, 3), r: (N,)."""
    v = (c_ref + r[:, None] * u - c_k) @ R_k
    dv = u @ R_k
    hx = v[:, 0] ** 2 + v[:, 2] ** 2
    n2 = hx + v[:, 1] ** 2
    lon = np.arctan2(v[:, 0], -v[:, 2])
    lat = np.arcsin(np.clip(v[:, 1] / np.sqrt(n2), -1, 1))
    dlon = (-v[:, 2] * dv[:, 0] + v[:, 0] * dv[:, 2]) / hx
    dlat = (dv[:, 1] * n2 - v[:, 1] * np.sum(v * dv, -1)) / (n2 * np.sqrt(hx))
    return lon, lat, dlon, dlat


def load(run_dir):
    from fov_render.record import SampleRecord
    run = json.loads((Path(run_dir) / "run.json").read_text())
    looks = []
    for lk in run["looks"]:
        rec = SampleRecord.load(Path(run_dir) / lk["out_dir"] / "record_C.npz", access="oracle")
        looks.append((rec, np.asarray(lk["c"], float), np.asarray(lk["R"], float), lk["s"]))
    return run, looks


def visible(rec_k, c_k, R_k, P, fan: sn.SlitFan):
    """M1.3's agreement in look k: P's nearest pixel holds the same surface within TOL_PX pixels at the range."""
    m = fan.model()
    d = (P - c_k) @ R_k
    rc, _ = m.pixel_of(d)
    W, H = fan.width, fan.width // 2
    ij = np.rint(rc).astype(np.int64)
    ij[:, 1] %= W
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < H)
    inv = np.full(H * W, -1, dtype=np.int64)
    inv[rec_k.raster_index] = np.arange(rec_k.n_samples)
    q = np.full(len(P), -1, dtype=np.int64)
    q[ok] = inv[ij[ok, 0] * W + ij[ok, 1]]
    vis = q >= 0
    rng = np.linalg.norm(P - c_k, axis=-1)
    err = np.full(len(P), np.inf)
    tol = np.zeros(len(P))
    err[vis] = np.linalg.norm(rec_k.position_w[q[vis]].astype(np.float64) - P[vis], axis=-1)
    tol[vis] = TOL_PX * rng[vis] * np.sqrt(rec_k.footprint_sr[q[vis]])
    return vis & np.isfinite(rec_k.distance_m[np.maximum(q, 0)]) & (err <= tol)


def evaluate(run_dir, fan=None, sigmas=SIGMAS_PX, seed=5, offset=None, iters=12):
    """The oracle on a run. offset: a world displacement added to every look's centre but the reference's (a control).
    Returns a dict of arrays per anchor and summaries."""
    fan = fan or sn.SlitFan()
    run, looks = load(run_dir)
    iref = len(looks) // 2
    rec_r, c_r, R_r, _ = looks[iref]
    hit = np.isfinite(rec_r.distance_m)
    P = rec_r.position_w[hit].astype(np.float64)
    pix = rec_r.raster_index[hit]
    rtrue = np.linalg.norm(P - c_r, axis=-1)
    u = (P - c_r) / rtrue[:, None]
    loc = u @ R_r
    b_ref = -np.arctan2(loc[:, 0], -loc[:, 2])
    lat_ref = np.arcsin(np.clip(loc[:, 1], -1, 1))
    others = [k for k in range(len(looks)) if k != iref]
    vis = np.stack([visible(looks[k][0], looks[k][1], looks[k][2], P, fan) for k in others], 1)
    obs = []
    for j, k in enumerate(others):
        c_k = looks[k][1] + (offset if offset is not None else 0.0)
        lon, lat, dlon, dlat = lonlat_and_derivs(c_r, u, rtrue, looks[k][1], looks[k][2])
        obs.append((lon, lat, c_k, looks[k][2]))
    I = np.zeros(len(P))
    for j, k in enumerate(others):
        _, _, dlon, dlat = lonlat_and_derivs(c_r, u, rtrue, looks[k][1], looks[k][2])
        I += np.where(vis[:, j], dlon ** 2 + dlat ** 2, 0.0)
    res_px = fan.res
    psi3 = np.arccos(np.clip(np.cos(lat_ref) * np.cos(b_ref), -1, 1))
    out = {"rtrue": rtrue, "b_ref": b_ref, "lat_ref": lat_ref, "psi3": psi3, "nvis": vis.sum(1), "I": I, "pix": pix, "fits": {}}
    rng = np.random.default_rng(seed)
    for sg in (0.0,) + tuple(sigmas):
        nz = [(o[0] + rng.normal(size=len(P)) * sg * res_px, o[1] + rng.normal(size=len(P)) * sg * res_px) for o in obs]
        r = rtrue * 1.05
        for _ in range(iters):
            Hs, gs = np.zeros(len(P)), np.zeros(len(P))
            for j, ((lon_o, lat_o), (_, _, c_k, R_k)) in enumerate(zip(nz, obs)):
                lon, lat, dlon, dlat = lonlat_and_derivs(c_r, u, r, c_k, R_k)
                rl = np.angle(np.exp(1j * (lon_o - lon)))
                rt = lat_o - lat
                w = vis[:, j]
                Hs += np.where(w, dlon ** 2 + dlat ** 2, 0.0)
                gs += np.where(w, dlon * rl + dlat * rt, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(Hs > 0, np.clip(r + gs / Hs, 0.05, 1e3), r)
        with np.errstate(divide="ignore"):
            sr = np.where(I > 0, max(sg, 1e-12) * res_px / np.sqrt(I), np.inf)
        usable = (out["nvis"] >= 1) & (Z95 * sr / rtrue <= USABLE_REL)
        cover = np.abs(r - rtrue) <= Z95 * sr
        out["fits"][sg] = {"r": r, "sigma_r": sr, "usable": usable, "cover": cover}
    return out


def sector_table(ev, sigmas=SIGMAS_PX):
    """Usable fraction per sector and rung, three ways (see SECTORS), and the 95 % coverage of the usable anchors."""
    absb = np.degrees(np.abs(ev["b_ref"]))
    eye = np.abs(np.degrees(ev["lat_ref"])) < EYE_BAND_DEG
    ways = {"psi_3d": (np.degrees(ev["psi3"]), np.ones_like(eye)), "bearing_eye_height": (absb, eye),
            "bearing_all_elevations": (absb, np.ones_like(eye))}
    tab = {}
    for sg in sigmas:
        f = ev["fits"][sg]
        row = {}
        for way, (ang, sel) in ways.items():
            row[way] = {}
            for name, (lo, hi) in SECTORS.items():
                m = sel & (ang >= lo) & (ang < hi)
                row[way][name] = float(f["usable"][m].mean()) if m.any() else None
        row["heading_fine_psi_3d"] = {}
        psi = np.degrees(ev["psi3"])
        for lo, hi in ((0.0, 2.0), (2.0, 5.0), (5.0, 15.0)):
            m = (psi >= lo) & (psi < hi)
            row["heading_fine_psi_3d"][f"{lo:g}–{hi:g}°"] = float(f["usable"][m].mean()) if m.any() else None
        u = f["usable"]
        row["coverage_of_usable"] = float(f["cover"][u].mean()) if u.any() else None
        tab[f"{sg:g}"] = row
    return tab


def selftest(runs: dict, fan=None):
    """O5.1-O5.3 on the runs {"line": dir, "arc": dir, "pivot": dir}."""
    fan = fan or sn.SlitFan()
    res, num = [], {}
    ev = {k: evaluate(v, fan) for k, v in runs.items()}
    # O5.1 exact recovery; control: looks' centres off by 1 cm
    worst = 0.0
    for k in ("line", "arc"):
        e = ev[k]
        m = e["nvis"] >= 1
        worst = max(worst, float(np.max(np.abs(e["fits"][0.0]["r"][m] / e["rtrue"][m] - 1))))
    ec = evaluate(runs["line"], fan, sigmas=(), offset=np.array([0.01, 0.0, 0.0]))
    mc = ec["nvis"] >= 1
    ctl = float(np.median(np.abs(ec["fits"][0.0]["r"][mc] / ec["rtrue"][mc] - 1)))
    num["O5.1_max_rel"], num["O5.1_control_median_rel"] = worst, ctl
    res.append(("O5.1 no noise: the fit recovers the true range on every anchor seen by another look (<= 1e-9)", worst <= 1e-9))
    res.append(("O5.1 control: the looks' centres off by 1 cm (median >= 1e-3)", ctl >= 1e-3))
    # O5.2 the oracle's Fisher equals the atlas's at eye height when every look sees the point
    worst2, ctl2 = 0.0, np.inf
    for k, R in (("line", None), ("arc", 1.6)):
        e = ev[k]
        n_other = e["nvis"].max()
        m = (np.abs(np.degrees(e["lat_ref"])) < 1.0) & (e["nvis"] == n_other) & (np.abs(np.sin(e["b_ref"])) > 0.1)
        s, poses = at.window_poses("line" if R is None else "arc", R, L=0.2, ds=0.01)
        orc = (fan.res / np.sqrt(e["I"][m])) / e["rtrue"][m]
        atl = at.sigma_rel(e["b_ref"][m], e["rtrue"][m], poses, s, sigma_b=fan.res)
        worst2 = max(worst2, float(np.median(np.abs(orc / atl - 1))))
        s2, poses2 = at.window_poses("line" if R is None else "arc", R, L=0.4, ds=0.02)
        atl2 = at.sigma_rel(e["b_ref"][m], e["rtrue"][m], poses2, s2, sigma_b=fan.res)
        ctl2 = min(ctl2, float(np.median(np.abs(orc / atl2 - 1))))
        num[f"O5.2_{k}_anchors"] = int(m.sum())
    num["O5.2_median_rel"], num["O5.2_control_median_rel"] = worst2, ctl2
    res.append(("O5.2 eye height, seen by every look: the oracle's Fisher bound equals stage 3's atlas (median within 1 %)", worst2 <= 0.01))
    res.append(("O5.2 control: the atlas with a window twice as long (median >= 10 % off)", ctl2 >= 0.10))
    # O5.3 the pivot carries no range; control: the line does
    Ip, Il = ev["pivot"]["I"], ev["line"]["I"]
    rel = float(np.max(Ip) / np.median(Il[Il > 0]))
    up = float(ev["pivot"]["fits"][0.25]["usable"].mean())
    ul = float(ev["line"]["fits"][0.25]["usable"].mean())
    num.update({"O5.3_pivot_I_over_line_median": rel, "O5.3_pivot_usable": up, "O5.3_line_usable": ul})
    res.append(("O5.3 the pivot carries no range: information <= 1e-20 of the line's, no anchor usable", rel <= 1e-20 and up == 0.0))
    res.append(("O5.3 control: the line's anchors are usable (> 0)", ul > 0.0))
    num["tables"] = {k: sector_table(e) for k, e in ev.items()}
    num["anchors"] = {k: int(len(e["rtrue"])) for k, e in ev.items()}
    num["seen_by_another"] = {k: float(np.mean(e["nvis"] >= 1)) for k, e in ev.items()}
    return res, num, ev
