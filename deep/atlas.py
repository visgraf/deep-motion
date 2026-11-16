"""Stage 3 (#3): the information atlas per primitive (numpy). Plan: docs/plan.md, stage 3.

The question: along a line or an arc, how precisely can the range of a point be recovered, by direction and distance,
and how finely must time be sampled for each direction?

Model (a design model, not a measurement). An anchor is a ray at bearing b0 (horizontal, head frame, from the heading,
+ to the left) at the window's centre s0; the point lies on it at range r, at eye height. Over a window of length L the
head is sampled every ds; at each sample the point's bearing is measured with Gaussian noise sigma_b (the sensor's
angular resolution), the anchor's own bearing exactly. Poses are known, so the head's rotation is predicted: only the
parallax carries r. The bound on r is Fisher's: sigma_r = 1 / sqrt(sum_k (d b_k / d r)^2 / sigma_b^2).

Time step. Between two samples, the part of the motion the poses cannot predict is the spread of parallax over the
range still uncertain, [r_lo, r_hi]: |sin psi| ds (1/r_lo - 1/r_hi), psi the angle from the direction of travel
(= b0, the head being locked to the tangent). Keeping it within one resolution element sigma_px gives
    ds_max = sigma_px / (|sin psi| (1/r_lo - 1/r_hi)),
the same for a line and an arc at the instant (the arc's rotation is predicted).
"""
from __future__ import annotations

import numpy as np

from . import primitives as pr

SIGMA_B = np.radians(0.1)      # proposed angular resolution of a designed sensor
WINDOW = 1.0                   # m, as M4's both5 (+/- 5 steps of ~0.1 m)
STEP = 0.01                    # m
H = 1.6


def window_poses(kind, R=None, L=WINDOW, ds=STEP):
    """Head poses (c, phi) every ds over a window of length L centred on s0 = 0, start at the origin heading +X."""
    s = np.arange(-L / 2, L / 2 + 1e-12, ds)
    pc = pr.Piece("line", L) if kind == "line" else pr.Piece("arc", L, 1.0 / R)
    # start the piece at -L/2 so that s = 0 is at the origin, heading 0
    if kind == "line":
        p0, phi0 = np.array([-L / 2, 0.0]), 0.0
    else:
        k = 1.0 / R
        phi0 = -k * L / 2
        p0 = -np.array([np.sin(0.0) - np.sin(phi0), -np.cos(0.0) + np.cos(phi0)]) / k
    out = []
    for si in s:
        c, Rh, p, phi = pc.pose(p0, phi0, H, si + L / 2)
        out.append((c, phi))
    return s, out


def bearings(P, poses):
    """Bearing of point(s) P in each head (from the heading, + left): shape (len(poses), N)."""
    out = []
    for c, phi in poses:
        d = P[..., :2] - c[:2]
        f, left = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
        out.append(np.arctan2(d @ left, d @ f))
    return np.array(out)


def dbearing_dr(b0, r, poses, analytic=True, h_rel=1e-6):
    """d b_k / d r for anchors at bearing b0 and range r from the head at s0 = 0 (origin, heading 0)."""
    u = np.stack([np.cos(b0), np.sin(b0)], -1)
    P = np.concatenate([r[..., None] * u, np.full(r.shape + (1,), H)], -1)
    if not analytic:
        h = h_rel * r
        Pp = np.concatenate([(r + h)[..., None] * u, P[..., 2:]], -1)
        Pm = np.concatenate([(r - h)[..., None] * u, P[..., 2:]], -1)
        bp, bm = bearings(Pp, poses), bearings(Pm, poses)
        return np.angle(np.exp(1j * (bp - bm))) / (2 * h)
    out = []
    for c, phi in poses:
        d = P[..., :2] - c[:2]
        f, left = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
        x, y = d @ f, d @ left
        dx, dy = u @ f, u @ left
        out.append((x * dy - y * dx) / (x * x + y * y))
    return np.array(out)


def sigma_rel(b0, r, poses, s, sigma_b=SIGMA_B, analytic=True):
    """sigma_r / r for anchors (b0, r): Fisher over the window's samples except the anchor's own (s = 0)."""
    J = dbearing_dr(b0, r, poses, analytic)
    w = (np.abs(s) > 1e-12)[:, None] if J.ndim == 2 else (np.abs(s) > 1e-12).reshape((-1,) + (1,) * (J.ndim - 1))
    I = np.sum(np.where(w, J * J, 0.0), 0) / sigma_b ** 2
    with np.errstate(divide="ignore"):
        return np.where(I > 0, 1.0 / np.sqrt(I), np.inf) / r


def ds_max(psi, r_lo, r_hi, sigma_px=SIGMA_B):
    """The largest time step that keeps the unpredicted parallax within one resolution element."""
    with np.errstate(divide="ignore"):
        return sigma_px / (np.abs(np.sin(psi)) * (1.0 / r_lo - 1.0 / r_hi))


def rate_spread(b0, r_lo, r_hi, poses, s):
    """Numerically: |d b/d s (r_lo) - d b/d s (r_hi)| at s0 = 0, rotation included (it cancels)."""
    i0 = int(np.argmin(np.abs(s)))
    def rate(r):
        u = np.array([np.cos(b0), np.sin(b0)])
        P = np.array([r * u[0], r * u[1], H])[None]
        b = bearings(P, poses)[:, 0]
        return np.angle(np.exp(1j * (b[i0 + 1] - b[i0 - 1]))) / (s[i0 + 1] - s[i0 - 1])
    return abs(rate(r_lo) - rate(r_hi)), rate(r_lo)


# ----------------------------------------------------------------------------- the atlas

CASES = (("line", None), ("arc R = 4 m", 4.0), ("arc R = 1.6 m", 1.6))
BEARINGS_DEG = np.arange(-179.0, 180.0, 1.0)
RANGES = np.geomspace(0.5, 20.0, 60)


def atlas(sigma_b=SIGMA_B, L=WINDOW, ds=STEP):
    """{case: sigma_r / r on (bearing, range)} and the profiles at r = 3 m."""
    out = {}
    B, Rg = np.meshgrid(np.radians(BEARINGS_DEG), RANGES, indexing="ij")
    for name, R in CASES:
        s, poses = window_poses("line" if R is None else "arc", R, L, ds)
        out[name] = sigma_rel(B, Rg, poses, s, sigma_b)
    return out


# ----------------------------------------------------------------------------- selftest (#3)

def selftest():
    from motion import derotate as dr
    res, num = [], {}
    rng = np.random.default_rng(3)

    # 1. analytic d b/d r equals the numerical one (line and arcs), the error measured against the derivative's typical
    #    size (relative error is undefined where the derivative crosses zero); control: the numerical one with the
    #    window's steps 2 % longer
    e1, c1 = 0.0, np.inf
    b0 = rng.uniform(-np.pi, np.pi, 300)
    r = np.exp(rng.uniform(np.log(0.6), np.log(15.0), 300))
    for name, R in CASES:
        s, poses = window_poses("line" if R is None else "arc", R)
        a, n_ = dbearing_dr(b0, r, poses, True), dbearing_dr(b0, r, poses, False)
        scale = np.median(np.abs(a))
        e1 = max(e1, float(np.max(np.abs(a - n_)) / scale))
        _, longer = window_poses("line" if R is None else "arc", R, L=WINDOW * 1.02, ds=STEP * 1.02)
        c1 = min(c1, float(np.median(np.abs(a - dbearing_dr(b0, r, longer, False))) / scale))
    num["deriv_err_over_median"], num["deriv_control_median_over_median"] = e1, c1
    res.append(("analytic d b/d r equals the numerical derivative, line and arcs (max error <= 1e-6 of its median size)",
                e1 <= 1e-6))
    res.append(("control: steps 2 % longer change it (median >= 1e-3 of its median size)", c1 >= 1e-3))

    # 2. the line with ten steps of 0.1 m reproduces derotate's Fisher bound (T0) with q-noise sigma_b / (sin e cos e);
    #    control: steps 20 % longer
    s = np.array([k * 0.1 for k in range(-5, 6)])
    poses = [(np.array([si, 0.0, H]), 0.0) for si in s]
    e = np.radians(rng.uniform(5.0, 80.0, 200)) * rng.choice([-1, 1], 200)
    rr = rng.uniform(1.0, 10.0, 200)
    ours = sigma_rel(e, rr, poses, s)
    ks = [k for k in range(-5, 6) if k]
    t = np.array([k * 0.1 for k in ks])
    rho = 1.0 / rr
    P = np.stack([rr * np.cos(e), rr * np.sin(e)], -1)
    ek = np.array([np.arctan2(np.abs(P[:, 1]), P[:, 0] - tk) for tk in t]).T          # eccentricity in frame k
    sq = SIGMA_B / (np.sin(ek) * np.cos(ek))
    cpsi = np.where(t[None] > 0, np.cos(e)[:, None], -np.cos(e)[:, None])
    t0 = dr.fisher_sigma_rho(np.abs(t)[None], rho[:, None], cpsi, sq) / rho
    ok = np.isfinite(t0) & np.isfinite(ours) & (np.abs(e) < np.radians(80))
    e2 = float(np.max(np.abs(ours[ok] / t0[ok] - 1)))
    t0c = dr.fisher_sigma_rho(1.2 * np.abs(t)[None], rho[:, None], cpsi, sq) / rho
    c2 = float(np.median(np.abs(ours[ok] / t0c[ok] - 1)))
    num["t0_line_rel_max"], num["t0_line_control_median"] = e2, c2
    res.append(("line, 10 steps of 0.1 m: the atlas equals T0's Fisher bound (derotate) (<= 1e-6 relative)", e2 <= 1e-6))
    res.append(("control: T0's bound with steps 20 % longer differs (median >= 5 %)", c2 >= 0.05))

    # 3. an arc of very large radius is the line; control: R = 20 m is not
    B, Rg = np.radians(rng.uniform(-170, 170, 300)), rng.uniform(0.6, 15.0, 300)
    s, pl = window_poses("line")
    _, pa = window_poses("arc", 1e7)
    _, pc = window_poses("arc", 20.0)
    sl, sa, sc = sigma_rel(B, Rg, pl, s), sigma_rel(B, Rg, pa, s), sigma_rel(B, Rg, pc, s)
    ok = np.isfinite(sl) & (np.abs(np.sin(B)) > 0.05)
    e3 = float(np.max(np.abs(sa[ok] / sl[ok] - 1)))
    c3 = float(np.max(np.abs(sc[ok] / sl[ok] - 1)))
    num["arc_to_line_rel_max"], num["arc20_vs_line_rel_max"] = e3, c3
    res.append(("an arc with R = 1e7 m is the line (<= 1e-5 relative)", e3 <= 1e-5))
    res.append(("control: R = 20 m differs (max >= 1e-3)", c3 >= 1e-3))

    # 4. blind sets. On a line the path line carries no range: every hypothesis along a ray on it stays on it and
    #    never moves. On an arc there is no blind direction: the turn axis is stationary in the head (stage 2), but any
    #    other range along that ray moves, so its range is still fixed. (Declared first as "the turn axis carries no
    #    range, I <= 1e-20"; it failed at I = 1.28, which is how this was found; see docs/plan.md, stage 3.)
    #    Controls: 1 cm off the path line carries range; the line's own atlas grid has blind cells.
    s, pl = window_poses("line")
    _, pa = window_poses("arc", 1.6)
    I = lambda b, rr_, ps: float(np.sum(dbearing_dr(np.array([b]), np.array([rr_]), ps)[np.abs(s) > 1e-12] ** 2))
    bl = max(I(0.0, 3.0, pl), I(np.pi, 3.0, pl))
    off_l = I(np.arctan2(0.01, 3.0), np.hypot(3.0, 0.01), pl)
    ba = I(np.pi / 2, 1.6, pa)
    A = atlas()
    arc_blind = int(sum(np.sum(~np.isfinite(A[n_])) for n_, R_ in CASES if R_ is not None))
    line_blind = int(np.sum(~np.isfinite(A["line"])))
    num.update(blind_line_I=bl, blind_line_control_I=off_l, axis_I=ba, arc_blind_cells=arc_blind, line_blind_cells=line_blind)
    res.append(("line: no information on the path line (I <= 1e-20)", bl <= 1e-20))
    res.append(("control: 1 cm off it carries information (I >= 1e-8)", off_l >= 1e-8))
    res.append(("arcs: no blind cell on the atlas grid, and the turn axis's range carries information (I >= 1e-3)",
                arc_blind == 0 and ba >= 1e-3))
    res.append(("control: the line's grid has blind cells (the heading and its opposite)", line_blind > 0))

    # 5. the time step: the closed form's parallax spread equals the trajectories' (rotation cancels), line and arcs;
    #    control: on the arc, the near point's rate alone (rotation not predicted) is not the spread
    e5, c5 = 0.0, np.inf
    for name, R in CASES:
        s, poses = window_poses("line" if R is None else "arc", R, ds=1e-4, L=0.01)
        for b in np.radians([20.0, 60.0, 90.0, 120.0, -45.0, -90.0]):
            spread, near = rate_spread(b, 0.5, 20.0, poses, s)
            cf = abs(np.sin(b)) * (1 / 0.5 - 1 / 20.0)
            e5 = max(e5, abs(spread / cf - 1))
            if R is not None:
                c5 = min(c5, abs(abs(near) / cf - 1))
    num["ds_closed_form_rel_max"], num["ds_control_rel_min"] = float(e5), float(c5)
    res.append(("time step: the parallax spread equals |sin psi| (1/r_lo - 1/r_hi), line and arcs (<= 1e-3 relative)", e5 <= 1e-3))
    res.append(("control: on the arcs, the near point's rate with the rotation left in is not it (>= 5 %)", c5 >= 0.05))
    return res, num
