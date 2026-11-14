"""Experiment M3: depth from RGB motion under aligned forward translation (M3a #6, from #5).

This module is the **estimator** side. It opens records for observation only (RGB, poses, calibrated
rays, footprints, raster indices); oracle fields (Position, distance, instance, normal) are refused by
fov-render's SampleRecord and never imported here. Oracle correspondence sources live in
`motion.flow_depth_oracle` and are evaluation-side; this module never imports them.

    load_observation   one look, observation only
    alignment          the gate of #5: heading = eye axis in both looks, no rotation between them
    reconstruct        correspondences (any source) -> range, range interval, world points, colour
    selftest           Stage 1 known answers and controls (numpy only, no Blender)

A correspondence set for a pair (a, b) is a dict of arrays over b's samples:
    sample_b  index into b's record      e_b, phi_b  b's direction (eye frame, rad)
    d, d_lo, d_hi  log-tangent displacement q_b - q_a and its interval      source  str
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fov_render import record as R
from fov_render.sensors import FoveatedModel, PinholeModel

from . import logpolar as lp

ALIGN_TOL_RAD = 1e-6           # #5's alignment gate (proposed before measurement)
CONTROL_YAW_DEG = 0.1
CONTROL_LATERAL_M = 0.01
USABLE_REL = 0.10              # an interval of relative half-width <= 10 % supports a usable range (as M2)


# ----------------------------------------------------------------------------- observation

@dataclass
class Obs:
    k: int
    rec: R.SampleRecord        # access="observation": oracle fields raise AccessError
    R: np.ndarray              # eye-to-world rotation (Blender camera matrix_world, 3x3)
    c: np.ndarray              # eye centre, world
    model: object
    rgb: np.ndarray            # (h, w, 3) linear RGB raster, nan off the valid disk

    @property
    def d_loc(self) -> np.ndarray:
        """Each sample's calibrated eye-frame direction (from the record's world directions)."""
        return lp.unit(self.rec.direction.astype(np.float64)) @ self.R


def load_observation(run: dict, entry: dict) -> Obs:
    d = Path(run["_dir"]) / entry["out_dir"]
    meta = json.loads((d / "meta.json").read_text())
    m = np.asarray(meta["response"]["eyes"]["C"]["pose"]["matrix_world"], dtype=np.float64)
    rec = R.SampleRecord.load(str(d / "record_C.npz"), access="observation")
    sensor = meta["request"]["sensor"]
    if sensor.get("type", "foveated") not in ("foveated",) and "E2_deg" not in sensor:
        raise ValueError(f"M3a reads Foveated looks only, got {sensor}")
    model = FoveatedModel(rec.raster_shape[0], sensor["E2_deg"], sensor["e_max_deg"])
    return Obs(int(entry["k"]), rec, m[:3, :3], m[:3, 3], model, rec.as_raster("value").astype(np.float64))


# ----------------------------------------------------------------------------- alignment gate

def _rot_angle(Ra, Rb) -> float:
    Rr = Ra.T @ Rb
    return float(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1.0, 1.0)))


def pair_alignment(ca, Ra, cb, Rb) -> dict:
    T = cb - ca
    B = float(np.linalg.norm(T))
    h = T / B if B > 0 else np.full(3, np.nan)
    fa, fb = Ra @ [0, 0, -1.0], Rb @ [0, 0, -1.0]
    ang = lambda u, v: float(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v))
    return {"B_m": B, "axis_a_rad": ang(fa, h), "axis_b_rad": ang(fb, h), "rotation_rad": _rot_angle(Ra, Rb)}


def _ok(a: dict) -> bool:
    return a["B_m"] > 0 and max(a["axis_a_rad"], a["axis_b_rad"], a["rotation_rad"]) <= ALIGN_TOL_RAD


def _yaw(R_, deg):
    t = np.radians(deg)
    Rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])   # world +Z up
    return Rz @ R_


def alignment(poses: dict, pairs) -> dict:
    """poses: k -> (c, R). The gate of #5 for every pair, with its two controls (each must fail every pair)."""
    rows, yaw_fail, lat_fail = [], [], []
    for ka, kb in pairs:
        (ca, Ra), (cb, Rb) = poses[ka], poses[kb]
        a = pair_alignment(ca, Ra, cb, Rb)
        rows.append(dict(a, pair=[ka, kb], ok=_ok(a)))
        yaw_fail.append(not _ok(pair_alignment(ca, Ra, cb, _yaw(Rb, CONTROL_YAW_DEG))))
        side = Ra @ [1.0, 0, 0]
        lat_fail.append(not _ok(pair_alignment(ca, Ra, cb + CONTROL_LATERAL_M * side, Rb)))
    worst = lambda k: max(r[k] for r in rows) if rows else None
    return {"id": "M3.0", "what": "heading = eye axis in both looks, no rotation between them (#5's gate)",
            "ok": bool(rows) and all(r["ok"] for r in rows),
            "control_fails": bool(rows) and all(yaw_fail) and all(lat_fail),
            "control": f"{CONTROL_YAW_DEG} deg yaw of b, and b's centre {CONTROL_LATERAL_M} m sideways: every pair must fail",
            "detail": {"pairs": len(rows), "tol_rad": ALIGN_TOL_RAD, "max_axis_rad": max(worst("axis_a_rad"), worst("axis_b_rad")),
                       "max_rotation_rad": worst("rotation_rad"), "min_B_m": min(r["B_m"] for r in rows) if rows else None},
            "rows": rows}


# ----------------------------------------------------------------------------- depth from correspondences

def reconstruct(corr: dict, ob: Obs, B: float) -> dict:
    """Range along b's ray, its interval, world point and b's observed linear colour, per correspondence.
    Uses only b's pose, the baseline length and the correspondences; the alignment gate must have passed."""
    e_b, phi = corr["e_b"], corr["phi_b"]
    r = lp.range_b(corr["d"], B, e_b)
    near, far = lp.range_interval(corr["d_lo"], corr["d_hi"], B, e_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        half = 0.5 * (far - near) / r
    finite = np.isfinite(r) & (r > 0)
    P = lp.world_points(ob.c, ob.R, e_b, phi, np.where(finite, r, np.nan))
    return {"range": r, "near": near, "far": far, "rel_half": half, "finite": finite,
            "usable": finite & np.isfinite(half) & (half <= USABLE_REL), "P": P,
            "rgb_linear": ob.rec.value[corr["sample_b"]].astype(np.float32)}


# ----------------------------------------------------------------------------- Stage 1 selftest

def _rot(rng):
    A = rng.normal(size=(3, 3))
    Q, Rr = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(Rr))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


KNOWN = {"z_a": 5.0, "e_a_deg": 20.0, "B": 0.098956521739, "e_b_deg": 20.3709243868, "z_b": 4.90104347826,
         "r_a": 5.32088886238, "r_b": 5.22800970366, "d": 0.019989775242}
NATIVE_SHIFTS = {1: 0.268722, 5: 0.570785, 10: 0.654810, 20: 0.669951, 30: 0.619280, 40: 0.535353}


def selftest() -> tuple[list, dict]:
    """Stage 1: geometry and known answers. Returns ([(name, ok)], numbers)."""
    res, num = [], {}

    # 1. round trips through (q, phi), both Foveated profiles and a 90-degree pinhole
    for name, model in (("foveated small", FoveatedModel(126)), ("foveated full", FoveatedModel(253)),
                        ("pinhole 90deg", PinholeModel(253, 253, 126.0 / np.tan(np.radians(45))))):
        rays = model.rays()
        d = rays.direction.reshape(-1, 3)[rays.valid.reshape(-1)]
        e, _ = lp.e_phi(d)
        keep = e > 1e-9                                          # the exact centre has no azimuth, no q
        q, phi = lp.qphi(d[keep])
        rc, inside = model.pixel_of(lp.dir_of_qphi(q, phi))
        j, i = np.meshgrid(np.arange(model.width), np.arange(model.height))
        want = np.stack([i.reshape(-1), j.reshape(-1)], -1)[rays.valid.reshape(-1)][keep]
        err = float(np.max(np.abs(rc - want)))
        quad = {f"{sx}{sy}": int(((np.sign(d[keep, 0]) == sx) & (np.sign(d[keep, 1]) == sy)).sum())
                for sx in (-1, 1) for sy in (-1, 1)}
        num[f"roundtrip_{name}"] = {"max_px": err, "samples": int(keep.sum()), "quadrants": quad,
                                    "max_e_deg": float(np.degrees(e.max()))}
        res.append((f"{name}: pixel -> (q, phi) -> pixel round trip < 1e-6 px, all quadrants",
                    bool(inside.all() and err < 1e-6 and min(quad.values()) > 0)))
        if name == "foveated full":
            flip = np.stack([model.n - 1 - rc[:, 0], rc[:, 1]], -1)        # control: wrong row sign
            res.append(("control: wrong row sign breaks the round trip by >= 1 px",
                        bool(np.max(np.abs(flip - want)) >= 1.0)))

    # 2. the radial Jacobian against #5's closed form, Foveated full; control: wrong E2
    m = FoveatedModel(253)
    e = np.radians(np.array([1, 2, 5, 10, 20, 30, 40, 44.0]))
    phi = np.array([0.3, -2.0, 1.1, 2.9, -0.7, 0.0, 1.6, -3.0])
    K, N, E2 = np.log(1 + 45 / 2.0), 253, np.radians(2.0)
    dq_dw = 2 * K / N * (E2 + e) / (np.sin(e) * np.cos(e))
    got = 1.0 / lp.native_px_per_q(m, lp.q_of_e(e), phi)
    rel = float(np.max(np.abs(got / dq_dw - 1)))
    wrong = 1.0 / lp.native_px_per_q(FoveatedModel(253, 2.2), lp.q_of_e(e), phi)
    # (the control is the same check run on a wrong sensor; a wrong E2 changes K and E2 + e in opposite
    # directions, so its error crosses zero near one eccentricity and "everywhere" would be the wrong test)
    num["jacobian"] = {"max_rel": rel, "wrong_E2_max_rel": float(np.max(np.abs(wrong / dq_dw - 1)))}
    res.append(("dq/dw through pixel_of equals (2K/N)(E2+e)/(sin e cos e) within 1e-5", rel <= 1e-5))
    res.append(("control: a sensor with E2 = 2.2 deg fails that check (max rel >= 1e-3)",
                num["jacobian"]["wrong_E2_max_rel"] >= 1e-3))
    # the pinhole is exactly log-radius: native px per q = f tan e
    pm = PinholeModel(253, 253, 126.0)
    eq = np.radians(np.array([2, 10, 30, 44.0]))
    gp = lp.native_px_per_q(pm, lp.q_of_e(eq), np.array([0.2, 1.0, -2.0, 2.5]))
    res.append(("pinhole: native px per unit q = f tan e within 1e-6", float(np.max(np.abs(gp / (126 * np.tan(eq)) - 1))) <= 1e-6))

    # 3. #5's known answer and its native shifts
    ka = KNOWN
    za, ea, B = ka["z_a"], np.radians(ka["e_a_deg"]), ka["B"]
    L = za * np.tan(ea)
    eb = np.arctan(L / (za - B))
    d = lp.q_of_e(eb) - lp.q_of_e(ea)
    za2, zb2 = lp.depth_of_d(d, B)
    vals = {"d": float(d), "e_b_deg": float(np.degrees(eb)), "z_b": float(zb2), "z_a": float(za2),
            "r_a": float(za / np.cos(ea)), "r_b": float(zb2 / np.cos(eb))}
    worst = max(abs(vals[k] / ka[k] - 1) for k in ("d", "e_b_deg", "z_b", "z_a", "r_a", "r_b"))
    num["known_answer"] = dict(vals, max_rel_vs_issue=float(worst))
    res.append(("#5's known answer (z_a 5 m, e_a 20 deg, stride 1) reproduced within 1e-9", worst <= 1e-9))
    shifts = {}
    for ed, want in NATIVE_SHIFTS.items():
        e1 = np.radians(ed)
        e2_ = np.arctan(5 * np.tan(e1) / (5 - B))
        a, _ = m.pixel_of(lp.dir_of_e_phi(e1, 0.0)[None])
        b, _ = m.pixel_of(lp.dir_of_e_phi(e2_, 0.0)[None])
        shifts[ed] = float(np.linalg.norm(b - a))
    num["native_shifts_px"] = shifts
    res.append(("#5's native radial shifts on the 5 m plane reproduced within 1e-6 px",
                max(abs(shifts[k] - v) for k, v in NATIVE_SHIFTS.items()) <= 1e-6))

    # 4. static analytic points: random aligned poses, depths, eccentricities, azimuths, three baselines
    rng = np.random.default_rng(0)
    errs, phis, ctrl = [], [], []
    for B in (0.098956521739, 0.494782609, 0.989565217):
        Rw = _rot(rng)
        ca = rng.normal(size=3)
        cb = ca + B * (Rw @ [0, 0, -1.0])
        zs = np.array([0.6, 1, 2, 5, 10, 20.0])
        es = np.radians(np.array([1, 2, 5, 10, 20, 30, 40, 44.0]))
        ph = np.array([0.0, 1.0, 2.0, 3.0, -2.5, -1.0])
        Z, E, PH = (a.reshape(-1) for a in np.meshgrid(zs, es, ph, indexing="ij"))
        Lr = Z * np.tan(E)
        P = ca + np.stack([Lr * np.cos(PH), Lr * np.sin(PH), -Z], -1) @ Rw.T
        da, db = lp.project_static(P, ca, Rw), lp.project_static(P, cb, Rw)
        (qa, pa), (qb, pb) = lp.qphi(da), lp.qphi(db)
        e_b, _ = lp.e_phi(db)
        par = np.degrees(np.arccos(np.clip(np.sum(da * db, -1), -1, 1)))
        keep = (Z > B) & (par >= 0.01)
        r = lp.range_b(qb - qa, B, e_b)
        truth = np.linalg.norm(P - cb, axis=-1)
        Pw = lp.world_points(cb, Rw, e_b, pb, r)
        errs.append(np.maximum(np.abs(r / truth - 1), np.linalg.norm(Pw - P, axis=-1) / truth)[keep])
        phis.append(np.abs(np.angle(np.exp(1j * (pa - pb))))[keep])
        # control: the native foveated radius used as if it were the pinhole radius (ln w in place of ln tan e)
        wa = np.linalg.norm(m.pixel_of(da)[0] - (m.n - 1) / 2, axis=-1)
        wb = np.linalg.norm(m.pixel_of(db)[0] - (m.n - 1) / 2, axis=-1)
        wide = keep & (np.degrees(e_b) >= 20) & (np.degrees(e_b) <= 45)
        rc_ = lp.depth_of_d(np.log(wb / wa), B)[1] / np.cos(e_b)
        ctrl.append(np.abs(rc_ / truth - 1)[wide])
    err, dphi, ctl = np.concatenate(errs), np.concatenate(phis), np.concatenate(ctrl)
    num["analytic_points"] = {"n": int(err.size), "max_rel_range_or_point": float(err.max()),
                              "max_dphi_rad": float(dphi.max()), "control_native_radius_median_rel": float(np.median(ctl))}
    res.append(("analytic static points: range and world point within 1e-6 relative (parallax >= 0.01 deg)",
                float(err.max()) <= 1e-6))
    res.append(("analytic static points keep their azimuth (|dphi| <= 1e-9 rad)", float(dphi.max()) <= 1e-9))
    res.append(("control: native radius as pinhole radius at 20-45 deg gives median error >= 1 %",
                float(np.median(ctl)) >= 0.01))

    # 5. intervals: reaching zero displacement is unbounded, wholly non-positive is no range
    near, far = lp.range_interval(np.array([0.01, -0.001, -0.02]), np.array([0.03, 0.02, -0.01]), 0.1, np.zeros(3))
    res.append(("interval: d_lo > 0 finite; d_lo <= 0 unbounded far; d_hi <= 0 no range",
                bool(np.isfinite(far[0]) and near[0] < far[0] and np.isinf(far[1]) and np.isfinite(near[1])
                     and np.isnan(near[2]) and np.isnan(far[2]))))

    # 6. the analysis grid on the full raster: support inside the disk, A3's oversampling numbers
    g = lp.make_grid(m)
    sup = g.support
    i0 = np.floor(g.rc[..., 0]).astype(int)
    j0 = np.floor(g.rc[..., 1]).astype(int)
    v = m.rays().valid
    ok_sup = all(v[np.clip(i0 + a, 0, 252), np.clip(j0 + b, 0, 252)][sup].all() for a in (0, 1) for b in (0, 1))
    rows = {}
    for ed in (1, 5, 20, 45):
        iq = int(np.argmin(np.abs(g.e - np.radians(ed))))
        rows[ed] = {"native_px_per_q_step": float(np.median(g.px_per_q[iq] * g.dq)),
                    "native_px_per_phi_step": float(np.median(g.px_per_phi[iq] * g.dphi)),
                    "supported": float(sup[iq].mean())}
    num["grid"] = {"shape": list(g.shape), "supported_fraction": float(sup.mean()), "by_ecc_deg": rows}
    res.append(("grid: every supported node's bilinear footprint is on valid pixels", bool(ok_sup)))
    return res, num
