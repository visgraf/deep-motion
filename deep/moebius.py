"""The primitives on the Riemann sphere (#11; Luiz, 2026-11-19, in Chat): elliptic, inverse-depth-weighted hyperbolic,
and their sum (numpy only; checks with controls, Chat sandbox).

Stereographic projection of the viewing sphere with the focus of expansion (FOE) at z = 0 and the focus of contraction
at z = infinity: for a direction at angle psi from the heading and azimuth omega about it,
z = tan(psi / 2) e^{i omega}; l = ln |z| = ln tan(psi / 2).

- A hyperbolic Moebius map with fixed points 0 and infinity, z -> k z (k > 0), shifts l by ln k for every point.
- An eye translating at unit speed along the heading moves a point at range r by dl/dt = 1 / r, along the same
  meridian (omega fixed): the hyperbolic generator weighted by inverse range. Finite motion: u = cot psi = -sinh l, and
  sinh l' = sinh l + t / D (the line's straight trace, u' = u - t / D).
- A rotation of the eye is a Moebius map (elliptic for a rotation about the heading axis; every rotation is in PSU(2)):
  it preserves the cross-ratio of any four directions, at any depths. A translation does not.
- The arc's instantaneous image motion is the turn's elliptic field (about the vertical, depth-free) plus the
  translation's 1/r-weighted hyperbolic field (about the tangent).
"""
from __future__ import annotations

import numpy as np

from . import primitives as pr


def stereo(d, f):
    """Stereographic coordinate z (complex) of unit directions d (N, 3) about the heading f, FOE at 0; omega measured
    from an axis e1 perpendicular to f."""
    f = f / np.linalg.norm(f)
    e1 = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(f, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(f, e1)
    c = np.clip(d @ f, -1, 1)
    psi = np.arccos(c)
    om = np.arctan2(d @ e2, d @ e1)
    return np.tan(psi / 2) * np.exp(1j * om)


def cross_ratio(z):
    a, b, c, d = z
    return (a - c) * (b - d) / ((a - d) * (b - c))


def selftest(seed=0):
    rng = np.random.default_rng(seed)
    res, num = [], {}
    f = np.array([0.0, 1.0, 0.0])
    # M1: translation speed in l is 1/r, along the meridian
    P = rng.normal(size=(2000, 3)) * 3
    P = P[np.linalg.norm(P, axis=1) > 0.3]
    r = np.linalg.norm(P, axis=1)
    h = 1e-6
    z0 = stereo(P / r[:, None], f)
    P1 = P - h * f
    z1 = stereo(P1 / np.linalg.norm(P1, axis=1)[:, None], f)
    dl = (np.log(np.abs(z1)) - np.log(np.abs(z0))) / h
    dom = np.abs(np.angle(z1 / z0)) / h
    num["M1_max_rel_err_dl_vs_1_over_r"] = float(np.max(np.abs(dl * r - 1)))
    num["M1_max_d_omega_dt"] = float(np.max(dom))
    num["M1_control_unweighted_max_rel_err"] = float(np.max(np.abs(dl - 1)))
    res.append(("M1 translation: dl/dt = 1/r for every direction and range (rel 1e-5), omega fixed (1e-6 rad/m)",
                num["M1_max_rel_err_dl_vs_1_over_r"] <= 1e-5 and num["M1_max_d_omega_dt"] <= 1e-6))
    res.append(("M1 control: the unweighted hyperbolic field, dl/dt = 1 (>= 10 % off)",
                num["M1_control_unweighted_max_rel_err"] >= 0.10))
    # M2: u = cot psi = -sinh l, and finite translation: sinh l' = sinh l + t / D
    t = 0.7
    D = np.linalg.norm(P - np.outer(P @ f, f), axis=1)
    psi = np.arccos(np.clip((P @ f) / r, -1, 1))
    l0 = np.log(np.abs(z0))
    Pt = P - t * f
    l1 = np.log(np.abs(stereo(Pt / np.linalg.norm(Pt, axis=1)[:, None], f)))
    num["M2_max_identity_err"] = float(np.max(np.abs(1 / np.tan(psi) + np.sinh(l0))))
    num["M2_max_finite_err"] = float(np.max(np.abs(np.sinh(l1) - np.sinh(l0) - t / D)))
    num["M2_control_uniform_shift_err"] = float(np.max(np.abs((l1 - l0) - np.median(l1 - l0))))
    res.append(("M2 u = cot psi = -sinh l, and a finite translation shifts sinh l by t/D (1e-9)",
                num["M2_max_identity_err"] <= 1e-9 and num["M2_max_finite_err"] <= 1e-9))
    res.append(("M2 control: a finite translation as one shift of l for all points (>= 0.1 off)",
                num["M2_control_uniform_shift_err"] >= 0.1))
    # M3: rotations preserve cross-ratios at any depth (Moebius); translations do not
    worst_rot, best_tr = 0.0, np.inf
    for _ in range(200):
        Q = rng.normal(size=(4, 3)) * rng.uniform(0.5, 5, size=(4, 1))
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        a = rng.uniform(0.1, 1.0)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        z = stereo(Q / np.linalg.norm(Q, axis=1)[:, None], f)
        Qr = Q @ Rm.T
        zr = stereo(Qr / np.linalg.norm(Qr, axis=1)[:, None], f)
        Qt = Q - 0.5 * f
        zt = stereo(Qt / np.linalg.norm(Qt, axis=1)[:, None], f)
        cr = cross_ratio(z)
        worst_rot = max(worst_rot, abs(cross_ratio(zr) - cr) / abs(cr))
        best_tr = min(best_tr, abs(cross_ratio(zt) - cr) / abs(cr))
    num["M3_rotation_max_rel_change"] = float(worst_rot)
    num["M3_control_translation_min_rel_change"] = float(best_tr)
    res.append(("M3 a rotation of the eye preserves the cross-ratio of any four directions (rel 1e-9): a Moebius map",
                worst_rot <= 1e-9))
    res.append(("M3 control: a translation does not (every trial changes it by >= 1e-4)", best_tr >= 1e-4))
    # M4: the arc's image motion = the turn's elliptic field + the translation's 1/r-weighted hyperbolic field
    R = 1.0
    pc = pr.Piece("arc", 1.0, 1.0 / R)
    c0, Rh0, _, _ = pc.pose((0.0, 0.0), np.pi / 2, 1.6, 0.0)
    c1, Rh1, _, _ = pc.pose((0.0, 0.0), np.pi / 2, 1.6, h)
    W = c0 + rng.normal(size=(500, 3)) * 2
    W = W[np.linalg.norm(W - c0, axis=1) > 0.3]
    v0 = (W - c0) @ Rh0
    v1 = (W - c1) @ Rh1
    flow = (v1 / np.linalg.norm(v1, axis=1)[:, None] - v0 / np.linalg.norm(v0, axis=1)[:, None]) / h
    d0 = v0 / np.linalg.norm(v0, axis=1)[:, None]
    rr = np.linalg.norm(v0, axis=1)
    fwd = np.array([0.0, 0.0, -1.0])                         # the heading in the head frame
    trans = (np.outer(d0 @ fwd, np.ones(3)) * d0 - fwd) / rr[:, None]    # d d / dt for unit translation: -(f - (d.f) d)/r
    upv = np.array([0.0, 1.0, 0.0])                          # the turn axis (vertical) in the head frame
    rot = -np.cross(upv / R, d0)                              # a left turn at rate 1/R (per metre) rotates the scene right
    num["M4_max_err"] = float(np.max(np.linalg.norm(flow - (trans + rot), axis=1)))
    num["M4_control_without_turn"] = float(np.median(np.linalg.norm(flow - trans, axis=1)))
    res.append(("M4 the arc's image motion = elliptic (turn, depth-free) + hyperbolic weighted by 1/r (1e-5)",
                num["M4_max_err"] <= 1e-5))
    res.append(("M4 control: without the turn's elliptic part (median >= 0.1 rad/m)", num["M4_control_without_turn"] >= 0.1))
    return res, num
