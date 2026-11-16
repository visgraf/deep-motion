"""M6 geometry: rotation removal with the known poses, and translation about each frame's own epipole (numpy only).

M6a #22, T0 (plan #21, M-D18). Eye frame as `logpolar` (Blender camera: +X right, +Y up, -Z forward). A pose is
(c, R): the eye centre in the world and R mapping eye-frame vectors to the world (world = R @ local), as
`analysis.load_look` returns them.

Derotation. Frame k sees the world direction of reference ray d (r's eye frame) as R_rk^T d with
R_rk = R_r^T R_k. A rotation about the eye's centre adds no parallax, so this holds at every depth; it is folded
into the sampling (`pixel_of` of the derotated direction), never applied to an image.

Translation about the epipole. After derotation, frame k differs from r by the translation c_k - c_r, which in r's
eye frame is |t_k| h_k (h_k the unit epipole, toward c_k; |t_k| > 0). For P = c_r + R_r d / rho (rho the inverse
range along r's ray d) and psi the angle between d and h_k:

    tan psi' / tan psi = cos psi / (cos psi - |t_k| rho),      the azimuth about h_k unchanged,

so with the log-tangent coordinate about h_k, q = ln |tan psi|,

    d_k(rho) = q'_k - q_r = -ln(1 - |t_k| rho / cos psi),      valid while |t_k| rho / cos psi < 1.

The one unknown shared by the window is rho. With every h_k on the eye's axis (M4's aligned dolly) this is M4's
d_k(zeta) = -ln(1 - t_k zeta), zeta = rho / cos e, t_k signed along the heading (`multiframe.d_of_zeta`).

The information map (#21's hypothesis, a model, not a measurement): relative sigma_rho / rho of a window from the
poses alone, for a frontal plane at D, with per-frame matching noise proportional to the Foveated sample spacing
a (e + E2), in the reference and in frame k, added in quadrature.
"""
from __future__ import annotations

import numpy as np

from . import logpolar as lp
from . import multiframe as mf

STEP = 0.0989565               # M4's gated step (m), as the dolly's
E2 = np.radians(2.0)           # the Foveated sensor's E2
E_MAX = np.radians(45.0)       # its field
FWD = np.array([0.0, 0.0, -1.0])


# ----------------------------------------------------------------------------- poses

def yaw(a) -> np.ndarray:
    """Rotation about the eye's +Y (up) by a radians; a > 0 turns the forward axis -Z toward -X (left)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def straight_poses(ks, step=STEP, alpha=0.0):
    """Reference at the origin facing -Z; frame k at k steps along the travel direction yawed alpha to the left of the
    facing (alpha = 0: M4's aligned dolly; alpha != 0: the crab dolly of T2). No rotation."""
    h = yaw(alpha) @ FWD
    return {k: (k * step * h, np.eye(3)) for k in ks}


def arc_poses(ks, radius, step=STEP):
    """A level circular arc turning left, the head facing along the tangent (gaze `ahead`): frame k has turned
    k step / radius and sits on the circle of the given radius whose centre is `radius` to the reference's left."""
    out = {}
    for k in ks:
        a = k * step / radius
        out[k] = (np.array([-radius + radius * np.cos(a), 0.0, -radius * np.sin(a)]), yaw(a))
    return out


# ----------------------------------------------------------------------------- derotation and the epipole

def derotate(d_r, R_r, R_k) -> np.ndarray:
    """Reference eye-frame direction(s) d_r as directions in frame k's eye frame (rotation only)."""
    return np.asarray(d_r, float) @ (R_r.T @ R_k)          # rows: (R_k^T R_r d)^T


def epipole(c_r, R_r, c_k):
    """Frame k's epipole in r's eye frame (unit, toward c_k) and the baseline |t_k| > 0."""
    v = R_r.T @ (np.asarray(c_k, float) - np.asarray(c_r, float))
    t = float(np.linalg.norm(v))
    return v / t, t


def predict_dir(d_r, rho, c_r, R_r, c_k, R_k) -> np.ndarray:
    """Exact forward model: frame k's eye-frame unit direction of P = c_r + R_r d_r / rho (rows of d_r, rho broadcast)."""
    d_r = lp.unit(d_r)
    P = c_r + (d_r / np.asarray(rho, float)[..., None]) @ R_r.T
    return lp.unit((P - c_k) @ R_k)


def psi_q(d, h):
    """Angle from the epipole h and the log-tangent coordinate q = ln |tan psi| about h."""
    d = lp.unit(d)
    cp = np.clip(d @ h, -1.0, 1.0)
    sp = np.linalg.norm(np.cross(d, h), axis=-1)
    psi = np.arctan2(sp, cp)
    with np.errstate(divide="ignore"):
        return psi, np.log(sp / np.abs(cp))


def azimuth_about(d, h):
    """Azimuth of d about the axis h, measured from a fixed perpendicular (the same for every frame's derotated d)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(h[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = lp.unit(np.cross(h, a))
    v = np.cross(h, u)
    d = lp.unit(d)
    return np.arctan2(d @ v, d @ u)


def d_of_rho(t, rho, cos_psi):
    """The generalised law: displacement in q about frame k's epipole, nan where invalid."""
    x = np.asarray(t, float) * np.asarray(rho, float) / np.asarray(cos_psi, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x < 1, -np.log1p(-x), np.nan)


def ddrho(t, rho, cos_psi):
    """d d_k / d rho."""
    a = np.asarray(t, float) / np.asarray(cos_psi, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / (1 - a * np.asarray(rho, float))


def fisher_sigma_rho(t, rho, cos_psi, sig_q):
    """Fisher bound on rho from per-frame displacement noise sig_q (in q about each epipole), frames on the last axis."""
    J = ddrho(t, rho, cos_psi)
    I = np.nansum(np.where(np.isfinite(J), J * J / np.asarray(sig_q, float) ** 2, 0.0), axis=-1)
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(I)


def epipole_spread_deg(W, radius, step=STEP) -> float:
    """Half-width of the window's epipole band on an arc: W step / (2 radius), in degrees."""
    return float(np.degrees(W * step / (2.0 * radius)))


# ----------------------------------------------------------------------------- the information map (#21)

def _ang(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def info_sigma_rel(e, phi, poses, r=0, D=3.0, e2=E2, e_max=E_MAX, h=1e-6) -> float:
    """#21's model: relative sigma_rho / rho at reference eccentricity e, azimuth phi, over the window `poses`
    (frame k -> (c, R), the reference r among them), frontal plane at distance D. Noise per frame
    ~ (e + E2)^2 + (e_k + E2)^2 (sample spacing a (e + E2), a dropped: the map is relative)."""
    c_r, R_r = poses[r]
    d = np.array([np.sin(e) * np.cos(phi), np.sin(e) * np.sin(phi), -np.cos(e)])
    rho = np.cos(e) / D
    I = 0.0
    for k, (c_k, R_k) in poses.items():
        if k == r:
            continue
        d1 = predict_dir(d[None], np.array([rho]), c_r, R_r, c_k, R_k)[0]
        ek = np.arctan2(np.hypot(d1[0], d1[1]), -d1[2])
        if ek > e_max:
            continue
        d2 = predict_dir(d[None], np.array([rho + h]), c_r, R_r, c_k, R_k)[0]
        g = float(_ang(d1, d2)) / h
        I += g * g / ((e + e2) ** 2 + (ek + e2) ** 2)
    return np.inf if I == 0 else (1.0 / np.sqrt(I)) / rho


INFO_ECC_DEG = (0.25, 0.5, 1, 2, 5, 10, 20, 40)
INFO_CASES = (("straight, ahead (M4)", "straight", 0.0), ("crab, FOE 10 deg off", "straight", 10.0),
              ("arc R=4m", "arc", 4.0), ("arc R=2m", "arc", 2.0), ("arc R=1m", "arc", 1.0))
# #21's table, from #21's script (Chat sandbox, 2026-11-16) at 4 decimals; #21 prints it at 2. The module must reproduce it
INFO_TABLE_21 = {
    "straight, ahead (M4)": (7.1678, 3.9985, 2.4139, 1.6222, 1.1518, 1.0106, 1.0000, 1.5231),
    "crab, FOE 10 deg off": (0.2333, 0.2429, 0.2677, 0.3400, 0.5272, 0.7247, 0.9051, 1.3992),
    "arc R=4m": (1.6270, 1.6245, 1.5956, 1.4863, 1.2276, 1.0462, 1.0097, 1.5268),
    "arc R=2m": (1.3977, 1.3985, 1.3945, 1.3684, 1.2789, 1.1292, 1.0395, 1.7733),
    "arc R=1m": (1.3032, 1.3037, 1.3035, 1.2984, 1.2805, 1.2425, 1.2601, 2.0064),
}


def window_poses(kind, param, W=5):
    ks = range(-W, W + 1)
    if kind == "straight":
        return straight_poses(ks, alpha=np.radians(param))
    return arc_poses(ks, param)


def info_table(n_phi=24, W=5, e2=E2):
    """#21's table: median over azimuth of info_sigma_rel, normalised to the straight dolly at e = 20 deg."""
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    straight = window_poses("straight", 0.0, W)
    ref = np.mean([info_sigma_rel(np.radians(20), p, straight, e2=e2) for p in phis])
    out = {}
    for name, kind, param in INFO_CASES:
        P = window_poses(kind, param, W)
        out[name] = [float(np.median([info_sigma_rel(np.radians(e), p, P, e2=e2) for p in phis]) / ref)
                     for e in INFO_ECC_DEG]
    return out


# ----------------------------------------------------------------------------- T0 selftest

def _random_rot(rng, max_deg=40.0):
    ax = lp.unit(rng.normal(size=3))
    a = np.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _rays(rng, n, e_max=E_MAX):
    e = np.arccos(rng.uniform(np.cos(e_max), 1.0, n))
    p = rng.uniform(-np.pi, np.pi, n)
    return np.stack([np.sin(e) * np.cos(p), np.sin(e) * np.sin(p), -np.cos(e)], -1)


def selftest():
    """T0 (#22): every check with a control that must fail. Returns [(name, passed)], numbers."""
    rng = np.random.default_rng(0)
    res, num = [], {}

    # 1. the generalised law against the exact forward model, random poses (rotation and translation)
    errs, az_errs, ctrl = [], [], []
    for _ in range(300):
        c_r, R_r = rng.normal(size=3), _random_rot(rng, 180)
        R_k = R_r @ _random_rot(rng, 30)
        c_k = c_r + R_r @ (lp.unit(rng.normal(size=3)) * rng.uniform(0.02, 1.0))
        h, t = epipole(c_r, R_r, c_k)
        d = _rays(rng, 50)
        rho = 1.0 / rng.uniform(0.5, 20.0, 50)
        dk = predict_dir(d, rho, c_r, R_r, c_k, R_k)          # frame k's eye frame
        back = dk @ (R_r.T @ R_k).T                             # derotated into r's frame: R_rk d_k
        psi, q_r = psi_q(d, h)
        _, q_k = psi_q(back, h)
        law = d_of_rho(t, rho, np.cos(psi))
        ok = np.isfinite(law)
        errs.append(np.max(np.abs((q_k - q_r) - law)[ok]))
        az_errs.append(np.max(np.abs(np.angle(np.exp(1j * (azimuth_about(back, h) - azimuth_about(d, h)))))[ok]))
        # control: the wrong rotation sign (R_k^T R_r used as R_rk)
        back_w = dk @ (R_k.T @ R_r).T
        _, q_kw = psi_q(back_w, h)
        ctrl.append(np.median(np.abs((q_kw - q_r) - law)[ok]))
    num["law_max_abs_q"] = float(max(errs))
    num["azimuth_max_rad"] = float(max(az_errs))
    num["control_wrong_sign_median_q"] = float(np.median(ctrl))
    res.append(("the law d_k(rho) equals the exact forward model after derotation (300 random poses, <= 1e-9 in q)",
                max(errs) <= 1e-9))
    res.append(("the azimuth about each frame's epipole is unchanged (<= 1e-9 rad)", max(az_errs) <= 1e-9))
    res.append(("control: the wrong rotation sign breaks the law (median error >= 1e-2 in q)", float(np.median(ctrl)) >= 1e-2))

    # 2. reduction to M4: epipoles on the eye's axis (both sides), d_k(rho) = multiframe.d_of_zeta(t, zeta)
    d = _rays(rng, 2000)
    e = np.arctan2(np.hypot(d[:, 0], d[:, 1]), -d[:, 2])
    rho = 1.0 / rng.uniform(0.6, 15.0, d.shape[0])
    worst = 0.0
    for k, (c_k, _) in straight_poses(range(-10, 11)).items():
        if k == 0:
            continue
        h, t = epipole(np.zeros(3), np.eye(3), c_k)
        psi, _ = psi_q(d, h)
        g, m4 = d_of_rho(t, rho, np.cos(psi)), mf.d_of_zeta(k * STEP, rho / np.cos(e))
        ok = np.isfinite(g) & np.isfinite(m4)
        worst = max(worst, float(np.max(np.abs(g - m4)[ok])))
    num["reduction_to_m4_max"] = worst
    res.append(("with every epipole on the axis the law is M4's d_of_zeta (<= 1e-12)", worst <= 1e-12))

    # 3. Fisher: the straight case equals M4's sigma_zeta * cos e (fit_zeta's bound); control: epipoles misplaced 20 deg
    ks = [k for k in range(-5, 6) if k]
    tk = np.array([k * STEP for k in ks])
    e0, z0, s = np.radians(12.0), 4.0, 0.003
    zeta = 1 / z0
    _, sz, _ = mf.fit_zeta(tk[None], mf.d_of_zeta(tk, zeta)[None], np.full((1, tk.size), s), np.ones((1, tk.size), bool),
                           zeta0=np.array([zeta]), iters=0)
    d0 = np.array([np.sin(e0), 0.0, -np.cos(e0)])
    hs = [epipole(np.zeros(3), np.eye(3), c)[0] for k, (c, _) in straight_poses(ks).items()]
    cps = np.array([d0 @ h for h in hs])
    sr = fisher_sigma_rho(np.abs(tk), zeta * np.cos(e0), cps, np.full(tk.size, s))
    num["fisher_rel_diff"] = float(abs(sr / (sz[0] * np.cos(e0)) - 1))
    hs_w = [epipole(np.zeros(3), np.eye(3), c)[0] for k, (c, _) in straight_poses(ks, alpha=np.radians(20)).items()]
    sr_w = fisher_sigma_rho(np.abs(tk), zeta * np.cos(e0), np.array([d0 @ h for h in hs_w]), np.full(tk.size, s))
    num["fisher_control_rel_diff"] = float(abs(sr_w / (sz[0] * np.cos(e0)) - 1))
    res.append(("Fisher sigma_rho on the aligned dolly = M4's sigma_zeta cos e (<= 1e-9 relative)", num["fisher_rel_diff"] <= 1e-9))
    res.append(("control: epipoles 20 deg off the axis change the bound (>= 5 %)", num["fisher_control_rel_diff"] >= 0.05))

    # 4. derotation: round trip, and depth independence of a pure rotation about the eye's centre
    rt, dep, dep_c = [], [], []
    for _ in range(200):
        R_r, R_k = _random_rot(rng, 180), _random_rot(rng, 180)
        d = _rays(rng, 100)
        rt.append(np.max(np.abs(derotate(derotate(d, R_r, R_k), R_k, R_r) - d)))
        c = rng.normal(size=3)
        near, far = predict_dir(d, np.full(100, 1 / 0.5), c, R_r, c, R_k), predict_dir(d, np.full(100, 1 / 20.0), c, R_r, c, R_k)
        dep.append(np.max(np.abs(near - far)))
        dep.append(np.max(np.abs(near - derotate(d, R_r, R_k))))
        # control: the head turns about a point 5 cm behind the eye (+Z is backward), so the eye's centre moves
        o = np.array([0.0, 0.0, 0.05])
        c_k = c + R_r @ o - R_k @ o
        if np.linalg.norm(c_k - c) > 0.005:
            nk, fk = predict_dir(d, np.full(100, 1 / 0.5), c, R_r, c_k, R_k), predict_dir(d, np.full(100, 1 / 20.0), c, R_r, c_k, R_k)
            dep_c.append(np.degrees(np.median(_ang(nk, fk))))
    num["derotation_round_trip"] = float(max(rt))
    num["rotation_depth_dependence"] = float(max(dep))
    num["control_offset_centre_median_deg"] = float(np.median(dep_c))
    res.append(("derotation round trip (<= 1e-12)", max(rt) <= 1e-12))
    res.append(("a rotation about the eye's centre is depth-independent and equals derotate() (<= 1e-12)", max(dep) <= 1e-12))
    res.append(("control: turning about a point 5 cm off the eye's centre is depth-dependent (median >= 0.1 deg, 0.5 vs 20 m)",
                num["control_offset_centre_median_deg"] >= 0.1))

    # 5. arcs: the chord to r + k lies along the tangent at r + k/2, so its epipole sits k dpsi / 2 from the heading;
    #    the band's half-width W s / (2R); both5 inside the 2 deg fovea needs R >= 7.1 m
    err = 0.0
    for R in (1.0, 2.0, 4.0):
        P = arc_poses(range(-5, 6), R)
        for k in range(-5, 6):
            if k == 0:
                continue
            h, _ = epipole(P[0][0], P[0][1], P[k][0])
            hf = h if k > 0 else -h                                       # as a line, forward-facing
            want = yaw(k * STEP / (2 * R)) @ FWD
            err = max(err, float(np.max(np.abs(hf - want))))
    r_min = 5 * STEP / (2 * np.radians(2.0))
    num["arc_epipole_max"] = err
    num["fovea_radius_min_m"] = float(r_min)
    res.append(("arc epipoles at k dpsi / 2 from the heading, inside the turn ahead and outside behind (<= 1e-12)", err <= 1e-12))
    res.append(("both5's band within the 2 deg fovea needs R >= 7.09 m (W s / (2R) = 2 deg)",
                abs(r_min - 7.0874) < 1e-3 and abs(epipole_spread_deg(5, r_min) - 2.0) < 1e-12))
    # control: on the straight aligned dolly every epipole is on the axis (spread 0); the arc formula must not give 0
    straight_dev = max(float(np.degrees(_ang(epipole(np.zeros(3), np.eye(3), c)[0] * np.sign(k), FWD)))
                       for k, (c, _) in straight_poses([k for k in range(-5, 6) if k]).items())
    res.append(("control: a 2 m arc's band is not the straight dolly's (7.09 deg against 0)",
                straight_dev < 1e-9 and epipole_spread_deg(5, 2.0) > 7.0))

    # 6. #21's information map is this module's (2-decimal table); control: a sensor with the wrong E2
    tab = info_table()
    dmax = max(abs(a - b) for name in INFO_TABLE_21 for a, b in zip(tab[name], INFO_TABLE_21[name]))
    num["info_table"] = {k: [round(v, 4) for v in vals] for k, vals in tab.items()}
    num["info_table_max_diff"] = float(dmax)
    res.append(("#21's information table reproduced (<= 1e-4 at every entry, the 4-decimal rounding)", dmax <= 1e-4))
    tab_w = info_table(e2=np.radians(2.2))
    num["control_e2_2p2_max_diff"] = float(max(abs(a - b) for name in INFO_TABLE_21
                                               for a, b in zip(tab_w[name], INFO_TABLE_21[name])))
    res.append(("control: the same table with E2 = 2.2 deg differs from #21's (>= 0.05 somewhere)",
                num["control_e2_2p2_max_diff"] >= 0.05))
    return res, num
