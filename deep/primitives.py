"""Stage 2 (#2): the path primitives and their motion-adapted coordinates (numpy only).

Plan: docs/plan.md, "The formulation". World metres, +Z up; paths are planar, at the head's height h. A head pose is
(c, R): the eye centre in the world and R = [right, up, back] (columns), world = R @ local, as motion.paths.head_axes.

Primitives (curvature kappa; a piece starts at position p0 = (x, y), heading phi0, measured from world +X toward +Y):

    line   kappa = 0           p(s) = p0 + s (cos phi0, sin phi0), heading phi0
    arc    0 < |kappa| < inf   heading phi0 + kappa s, centre p0 + (1/kappa) (-sin phi0, cos phi0) (left for kappa > 0)
    pivot  kappa = inf         p0 fixed, heading phi0 + a, a the turn angle (its parameter in place of s)

Motion-adapted coordinates of a ray (unit world direction r, at a pose), and the trace of a world point P:

    line   the path line's pencil. omega = angle of r's part across the path direction f, about f (from `right`
           toward `up`); u = cot psi = (r . f) / |r_perp|. P at distance D from the path line keeps its omega and has
           u(s) = (x_P - s) / D, x_P its position along the line: a straight trace of slope -1/D. Invariant D.
    arc    concentric mosaics. beta = horizontal angle of r from the tangent, positive toward the centre;
           v = tan(elevation). With the camera at angle theta on the circle (radius R about the centre), P at radius rho
           and angle phi_P about the centre and height dz above the eye: P is in slit beta when
           delta = phi_P - theta satisfies rho cos(delta - beta) = R cos beta (and sin(delta) / cos(beta) > 0, the ray
           going forward), i.e. delta = beta +/- arccos(R cos beta / rho); there v = dz cos(beta) / (rho sin(delta)).
           Traces of points with the same (rho, dz) are shifts in theta. Invariant rho (with dz).
    pivot  rotation about the eye: a point's direction turns with the head, at any depth. No invariant.

Blind sets: the path line (line) and the turn's vertical axis (arc) keep a fixed direction in the moving head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def head_R(phi) -> np.ndarray:
    """R = [right, up, back] for a level head with heading phi (rad, from world +X toward +Y)."""
    f = np.array([np.cos(phi), np.sin(phi), 0.0])
    right = np.cross(f, UP)
    return np.stack([right, UP, -f], axis=1)


@dataclass
class Piece:
    kind: str            # "line", "arc", "pivot"
    length: float        # metres (line, arc); turn angle in rad (pivot)
    kappa: float = 0.0   # arc only, 1/m, > 0 turns left

    def pose(self, p0, phi0, h, s):
        """Pose at parameter s (metres, or rad for a pivot) from the piece's start: (c, R, p, phi)."""
        p0 = np.asarray(p0, float)
        if self.kind == "line":
            p, phi = p0 + s * np.array([np.cos(phi0), np.sin(phi0)]), phi0
        elif self.kind == "arc":
            k = self.kappa
            phi = phi0 + k * s
            p = p0 + np.array([np.sin(phi) - np.sin(phi0), -np.cos(phi) + np.cos(phi0)]) / k
        elif self.kind == "pivot":
            p, phi = p0, phi0 + s
        else:
            raise ValueError(self.kind)
        return np.array([p[0], p[1], h]), head_R(phi), p, phi

    def end(self, p0, phi0, h):
        _, _, p, phi = self.pose(p0, phi0, h, self.length)
        return p, phi

    def centre(self, p0, phi0):
        """The arc's turn centre (2-D)."""
        return np.asarray(p0, float) + np.array([-np.sin(phi0), np.cos(phi0)]) / self.kappa


class Path:
    """A composition of pieces from a start (p0, phi0) at head height h."""

    def __init__(self, pieces, p0=(0.0, 0.0), phi0=0.0, h=1.6):
        self.pieces, self.h = list(pieces), h
        self.starts = []
        p, phi = np.asarray(p0, float), float(phi0)
        for pc in self.pieces:
            self.starts.append((p.copy(), phi))
            p, phi = pc.end(p, phi, h)

    def poses(self, step=0.1, pivot_step=np.radians(5.0)):
        """[(piece index, s, c, R)] sampled every `step` m (lines, arcs) or `pivot_step` rad (pivots), ends included."""
        out = []
        for i, (pc, (p0, phi0)) in enumerate(zip(self.pieces, self.starts)):
            d = pivot_step if pc.kind == "pivot" else step
            n = max(1, int(np.ceil(abs(pc.length) / d - 1e-12)))
            for s in np.linspace(0.0, pc.length, n + 1):
                c, R, _, _ = pc.pose(p0, phi0, self.h, s)
                out.append((i, float(s), c, R))
        return out


# ----------------------------------------------------------------------------- coordinates of rays

def line_coords(r, f):
    """(omega, u) of world unit ray(s) r for a line with unit direction f (horizontal): the pencil about the path line."""
    r = np.asarray(r, float)
    right = np.cross(f, UP)
    a = r @ f
    perp = r - a[..., None] * f
    n = np.linalg.norm(perp, axis=-1)
    return np.arctan2(perp @ UP, perp @ right), a / n


def arc_coords(r, phi):
    """(beta, v) of world unit ray(s) r for a head with heading phi on a left-turning arc: beta from the tangent
    toward the centre (left), v = tan(elevation)."""
    r = np.asarray(r, float)
    f = np.array([np.cos(phi), np.sin(phi), 0.0])
    left = np.array([-np.sin(phi), np.cos(phi), 0.0])
    hz = np.hypot(r @ f, r @ left)
    return np.arctan2(r @ left, r @ f), (r @ UP) / hz


def direction(P, c):
    """World unit direction from the eye centre c to point(s) P."""
    d = np.asarray(P, float) - np.asarray(c, float)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------- closed forms

def line_trace_u(x_P, D, s):
    """u of a point at along-track position x_P and distance D from the path line, seen from s along the line."""
    return (x_P - s) / D


def arc_slit_delta(beta, rho, R, branch=+1):
    """delta = phi_P - theta at which a point at radius rho is in slit beta (nan where it never is, or where the ray
    would go backward), branch +1 or -1 (the two tangents to the circle of radius R cos beta)."""
    beta = np.asarray(beta, float)
    with np.errstate(invalid="ignore"):
        g = np.arccos(R * np.cos(beta) / rho)
        d = beta + branch * g
        ok = np.sin(d) / np.cos(beta) > 0
    return np.where(ok, d, np.nan)


def arc_slit_v(beta, delta, rho, dz):
    """tan(elevation) of the point in slit beta at angle delta: dz over its horizontal distance rho sin(delta)/cos(beta)."""
    return dz * np.cos(beta) / (rho * np.sin(delta))


# ----------------------------------------------------------------------------- selftest (#2)

def selftest(seed=0):
    """Stage 2's checks, each with a control that must fail. Returns [(name, ok)], numbers."""
    from motion import multiframe as mf
    from motion import paths as mpaths
    rng = np.random.default_rng(seed)
    res, num = [], {}

    # 0. head axes: ours are motion.paths.head_axes; control: right and left swapped
    phis = rng.uniform(-np.pi, np.pi, 50)
    err = max(np.abs(head_R(a) - mpaths.head_axes([np.cos(a), np.sin(a), 0.0])).max() for a in phis)
    ctl = min(np.abs(head_R(a) * np.array([-1, 1, 1]) - mpaths.head_axes([np.cos(a), np.sin(a), 0.0])).max() for a in phis)
    num["head_axes_max"], num["head_axes_control_min"] = float(err), float(ctl)
    res.append(("head axes equal motion.paths.head_axes (<= 1e-15)", err <= 1e-15))
    res.append(("control: right flipped differs (>= 0.5)", ctl >= 0.5))

    # 1. line: the closed form against exact projection, along the line, for random points and a random line
    h = 1.6
    p0, phi0 = rng.normal(size=2), rng.uniform(-np.pi, np.pi)
    line = Piece("line", 2.0)
    f = np.array([np.cos(phi0), np.sin(phi0), 0.0])
    left = np.array([-np.sin(phi0), np.cos(phi0), 0.0])
    n = 400
    x_P = rng.uniform(3.0, 12.0, n)
    D, om = rng.uniform(0.3, 6.0, n), rng.uniform(-np.pi, np.pi, n)
    right = np.cross(f, UP)
    base = np.array([p0[0], p0[1], h])
    P = base + x_P[:, None] * f + D[:, None] * (np.cos(om)[:, None] * right + np.sin(om)[:, None] * UP)
    e_u, e_om, c_u = 0.0, 0.0, np.inf
    for s in np.linspace(0, 2.0, 21):
        c, Rh, _, _ = line.pose(p0, phi0, h, s)
        r = direction(P, c)
        om_s, u_s = line_coords(r, f)
        e_u = max(e_u, float(np.abs(u_s - line_trace_u(x_P, D, s)).max()))
        e_om = max(e_om, float(np.abs(np.angle(np.exp(1j * (om_s - om)))).max()))
        # control: a path line turned 1 deg (the trace computed with the wrong f)
        fw = np.array([np.cos(phi0 + np.radians(1)), np.sin(phi0 + np.radians(1)), 0.0])
        _, u_w = line_coords(r, fw)
        if s > 0:
            c_u = min(c_u, float(np.median(np.abs(u_w - line_trace_u(x_P, D, s)))))
    num["line_u_max"], num["line_omega_max"], num["line_control_median_min"] = e_u, e_om, c_u
    res.append(("line: u = (x_P - s)/D and omega constant, against exact projection (<= 1e-12)", e_u <= 1e-12 and e_om <= 1e-12))
    res.append(("control: a path line turned 1 deg breaks it (median >= 1e-3)", c_u >= 1e-3))

    # 2. arc: closed form against exact projection; slits beta, both branches
    R = rng.uniform(0.8, 4.0)
    kappa = 1.0 / R
    arc = Piece("arc", 3.0, kappa)
    p0a, phi0a = rng.normal(size=2), rng.uniform(-np.pi, np.pi)
    C2 = arc.centre(p0a, phi0a)
    rho = rng.uniform(R + 0.5, R + 8.0, n)
    phiP = rng.uniform(-np.pi, np.pi, n)
    dz = rng.uniform(-1.4, 1.2, n)
    Pa = np.stack([C2[0] + rho * np.cos(phiP), C2[1] + rho * np.sin(phiP), h + dz], -1)
    # theta of the start: angle of p0 about the centre
    th0 = np.arctan2(p0a[1] - C2[1], p0a[0] - C2[0])
    e_th, e_v, nchk, c_th = 0.0, 0.0, 0, np.inf
    for beta in np.radians([-80.0, -45.0, -10.0, 0.0, 20.0, 60.0, 85.0]):
        for br in (+1, -1):
            d = arc_slit_delta(beta, rho, R, br)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            theta = phiP[ok] - d[ok]                         # where the camera must be
            s = np.angle(np.exp(1j * (theta - th0))) / kappa  # its arc parameter (any turn of the circle)
            for j, (sj, Pj, dj) in enumerate(zip(s, Pa[ok], d[ok])):
                c, Rh, _, phi = arc.pose(p0a, phi0a, h, sj)
                bt, v = arc_coords(direction(Pj, c), phi)
                e_th = max(e_th, abs(float(np.angle(np.exp(1j * (bt - beta))))))
                e_v = max(e_v, abs(float(v - arc_slit_v(beta, dj, rho[ok][j], dz[ok][j]))))
                nchk += 1
                # control: the closed form with a radius 2 % wrong
                dw = arc_slit_delta(beta, rho[ok][j], R * 1.02, br)
                if np.isfinite(dw):
                    cw, _, _, phw = arc.pose(p0a, phi0a, h, np.angle(np.exp(1j * (phiP[ok][j] - dw - th0))) / kappa)
                    bw, _ = arc_coords(direction(Pj, cw), phw)
                    c_th = min(c_th, abs(float(np.angle(np.exp(1j * (bw - beta)))))) if abs(float(np.angle(np.exp(1j * (bw - beta))))) > 0 else c_th
    num["arc_R"], num["arc_checked"], num["arc_beta_max"], num["arc_v_max"] = float(R), nchk, e_th, e_v
    num["arc_control_min_beta_err"] = float(c_th)
    res.append(("arc: the point is in slit beta at theta = phi_P - delta(beta, rho), with v = dz cos(beta)/(rho sin(delta)), "
                "against exact projection (<= 1e-11)", e_th <= 1e-11 and e_v <= 1e-11 and nchk > 100))
    res.append(("control: a radius 2 % wrong misplaces every point (min slit error >= 1e-5 rad)", c_th >= 1e-5))

    # 3. stationarity, from exact projection: a point moved along the motion's orbit by Delta (along the line, or about
    #    the turn axis) is seen in the head at s + Delta exactly as the original at s; control: the moved point 5 %
    #    farther (D or rho), which is not on the same orbit
    def head_dir(piece, p0_, phi0_, Pt, s_):
        c, Rh, _, _ = piece.pose(p0_, phi0_, h, s_)
        return direction(Pt, c) @ Rh
    e3, c3 = 0.0, np.inf
    for Delta in (0.2, 0.7, 1.3):
        P2 = P + Delta * f                                                    # same omega and D, x_P + Delta
        P2w = base + (x_P + Delta)[:, None] * f + 1.05 * D[:, None] * (np.cos(om)[:, None] * right + np.sin(om)[:, None] * UP)
        a_ = Delta / R                                                         # the same arc length on the arc
        rot = np.array([[np.cos(a_), -np.sin(a_), 0], [np.sin(a_), np.cos(a_), 0], [0, 0, 1.0]])
        Q = (Pa - np.array([C2[0], C2[1], 0.0])) @ rot.T + np.array([C2[0], C2[1], 0.0])
        Qw = Q.copy()
        Qw[:, :2] = C2 + 1.05 * (Q[:, :2] - C2)
        for s0 in (0.0, 0.4):
            e3 = max(e3, float(np.abs(head_dir(line, p0, phi0, P2, s0 + Delta) - head_dir(line, p0, phi0, P, s0)).max()),
                     float(np.abs(head_dir(arc, p0a, phi0a, Q, s0 + Delta) - head_dir(arc, p0a, phi0a, Pa, s0)).max()))
            c3 = min(c3, float(np.median(np.abs(head_dir(line, p0, phi0, P2w, s0 + Delta) - head_dir(line, p0, phi0, P, s0)))),
                     float(np.median(np.abs(head_dir(arc, p0a, phi0a, Qw, s0 + Delta) - head_dir(arc, p0a, phi0a, Pa, s0)))))
    num["shift_max"], num["shift_control_median_min"] = e3, c3
    res.append(("same orbit, same trace shifted in time: line (along f) and arc (about the turn axis), exact projection "
                "(<= 1e-12)", e3 <= 1e-12))
    res.append(("control: the moved point 5 % farther from the path line or the turn axis (median >= 1e-4)", c3 >= 1e-4))

    # 4. blind sets: a point on the path line, or on the turn axis, keeps its direction in the head; 1 cm off does not
    def head_dirs(piece, p0_, phi0_, Pt, ss):
        return np.array([Rh.T @ direction(Pt, c) for c, Rh, _, _ in (piece.pose(p0_, phi0_, h, s_) for s_ in ss)])
    ss = np.linspace(0, 2.0, 21)
    on_line = base + 9.0 * f
    bl = float(np.abs(head_dirs(line, p0, phi0, on_line, ss) - head_dirs(line, p0, phi0, on_line, ss)[0]).max())
    off_line = on_line + 0.01 * left
    blc = float(np.abs(head_dirs(line, p0, phi0, off_line, ss) - head_dirs(line, p0, phi0, off_line, ss)[0]).max())
    axis_pt = np.array([C2[0], C2[1], h + 0.7])
    ba = float(np.abs(head_dirs(arc, p0a, phi0a, axis_pt, ss) - head_dirs(arc, p0a, phi0a, axis_pt, ss)[0]).max())
    off_axis = axis_pt + np.array([0.01, 0.0, 0.0])
    bac = float(np.abs(head_dirs(arc, p0a, phi0a, off_axis, ss) - head_dirs(arc, p0a, phi0a, off_axis, ss)[0]).max())
    num.update(blind_line=bl, blind_line_control=blc, blind_axis=ba, blind_axis_control=bac)
    res.append(("blind sets: path line and turn axis keep their head direction (<= 1e-12)", bl <= 1e-12 and ba <= 1e-12))
    res.append(("control: 1 cm off either does not (>= 1e-5)", blc >= 1e-5 and bac >= 1e-5))

    # 5. pivot: a point's head direction turns with the head at any depth; turning about a point 5 cm behind the eye does not
    piv = Piece("pivot", np.radians(40))
    Pn = rng.normal(size=(200, 3)) * 4 + np.array([0, 0, h])
    e_p, c_p = 0.0, np.inf
    for a in np.radians([5.0, 20.0, 40.0]):
        c, Rh, _, _ = piv.pose(p0, phi0, h, a)
        c0, R0, _, _ = piv.pose(p0, phi0, h, 0.0)
        e_p = max(e_p, float(np.abs(Rh.T @ R0 @ (R0.T @ direction(Pn, c0).T) - Rh.T @ direction(Pn, c).T).max()))
        o = np.array([0.0, 0.0, 0.05])                    # head-frame offset of the eye from the pivot (+Z back)
        cw = c0 - R0 @ o + Rh @ o
        dep = np.abs(Rh.T @ direction(Pn, cw).T - Rh.T @ R0 @ (R0.T @ direction(Pn, c0).T)).max()
        c_p = min(c_p, float(dep))
    num["pivot_max"], num["pivot_control_min"] = e_p, c_p
    res.append(("pivot: rotation about the eye moves every direction by the rotation alone (<= 1e-12)", e_p <= 1e-12))
    res.append(("control: a pivot 5 cm behind the eye adds depth-dependent motion (>= 1e-4)", c_p >= 1e-4))

    # 6. M4's dolly is the line seen from its own heading: q = ln tan e = -ln u, so d = -ln(u(s + t)/u(s)) = M4's d_k(zeta)
    #    (motion.multiframe.d_of_zeta), zeta = 1 / (axial depth); control: the head 1 deg off the path line
    zax = x_P                                                   # axial depth at s = 0 (gaze along f)
    ts = np.array([-0.5, -0.2, 0.1, 0.3, 0.5])
    e4, c4 = 0.0, np.inf
    for t in ts:
        c, Rh, _, _ = line.pose(p0, phi0, h, t) if t >= 0 else Piece("line", 1.0).pose(p0, phi0, h, t)
        _, u_t = line_coords(direction(P, c), f)
        _, u_0 = line_coords(direction(P, base), f)
        d_ours = -np.log(u_t / u_0)
        d_m4 = mf.d_of_zeta(t, 1.0 / zax)
        okm = np.isfinite(d_m4) & np.isfinite(d_ours)
        e4 = max(e4, float(np.abs(d_ours - d_m4)[okm].max()))
        Rw = head_R(phi0 + np.radians(1.0))
        e_head = lambda c_: np.arctan2(np.hypot(*(direction(P, c_) @ Rw)[:, :2].T), -(direction(P, c_) @ Rw)[:, 2])
        d_w = np.log(np.tan(e_head(c))) - np.log(np.tan(e_head(base)))
        c4 = min(c4, float(np.median(np.abs(d_w - d_m4)[okm])))
    num["m4_reduction_max"], num["m4_control_median_min"] = e4, c4
    res.append(("line with gaze along it: -ln(u(s+t)/u(s)) is M4's d_k(zeta) (<= 1e-12)", e4 <= 1e-12))
    res.append(("control: the head 1 deg off the path line breaks it (median >= 1e-4)", c4 >= 1e-4))

    # 7. composition: a line, arc, pivot, line path is continuous in position and heading; control: an arc 1 % short
    pieces = [Piece("line", 1.2), Piece("arc", 1.5, 1 / 1.6), Piece("pivot", np.radians(-30)), Piece("line", 1.0)]
    path = Path(pieces, p0=(0.3, -0.2), phi0=0.4, h=h)
    jumps = []
    for i in range(len(pieces) - 1):
        pe, fe = pieces[i].end(*path.starts[i], h)
        jumps.append(max(np.abs(pe - path.starts[i + 1][0]).max(), abs(fe - path.starts[i + 1][1])))
    bad = Path([Piece("line", 1.2), Piece("arc", 1.5 * 0.99, 1 / 1.6), Piece("pivot", np.radians(-30)), Piece("line", 1.0)],
               p0=(0.3, -0.2), phi0=0.4, h=h)
    pe, fe = pieces[1].end(*path.starts[1], h)
    ctl7 = max(np.abs(pe - bad.starts[2][0]).max(), abs(fe - bad.starts[2][1]))
    n_poses = len(path.poses(step=0.1))
    num["composition_max_jump"], num["composition_control_jump"], num["composition_poses"] = float(max(jumps)), float(ctl7), n_poses
    res.append(("composition: pieces join continuously in position and heading (<= 1e-12)", max(jumps) <= 1e-12))
    res.append(("control: an arc 1 % short leaves a gap (>= 1e-3)", ctl7 >= 1e-3))
    return res, num
