"""M3 geometry: forward translation along the eye's axis in log-tangent polar coordinates (numpy only).

Ray-based and sensor-agnostic (M3a, #6, amendment A4): every coordinate comes from eye-frame unit
directions, and a sensor enters only through its model's `rays()` / `pixel_of()`, never through the
Foveated warp formula, so the same code runs on a pinhole (`PinholeModel`, `PerspectiveModel`).

Eye frame (Blender camera): +X right, +Y up, -Z forward. For a unit direction d,

    e   = atan2(hypot(dx, dy), -dz)       eccentricity from the eye axis (rad)
    phi = atan2(dy, dx)                    azimuth (rad), the same in the shader frame
    q   = ln(tan e)                        the log-tangent radius

A static point at axial depth z_a, lateral offset L from the axis, seen again after a step B > 0
along the axis (no rotation): tan e_a = L / z_a, tan e_b = L / (z_a - B), phi_b = phi_a, and

    d   = q_b - q_a = ln(z_a / (z_a - B)) > 0
    z_a = B / (-expm1(-d)),  z_b = B / expm1(d),  r_b = z_b / cos e_b,  P = c_b + r_b R_b d_B

(#5's geometry contract). A displacement interval reaching d <= 0 gives an unbounded depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEG = np.pi / 180.0


# ----------------------------------------------------------------------------- coordinates

def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def e_phi(d_loc) -> tuple[np.ndarray, np.ndarray]:
    d = unit(d_loc)
    return np.arctan2(np.hypot(d[..., 0], d[..., 1]), -d[..., 2]), np.arctan2(d[..., 1], d[..., 0])


def q_of_e(e) -> np.ndarray:
    """-inf on the axis; nan behind the eye (e > 90 deg)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.tan(np.asarray(e, dtype=np.float64)))


def e_of_q(q) -> np.ndarray:
    return np.arctan(np.exp(np.asarray(q, dtype=np.float64)))


def qphi(d_loc) -> tuple[np.ndarray, np.ndarray]:
    e, phi = e_phi(d_loc)
    return q_of_e(e), phi


def dir_of_e_phi(e, phi) -> np.ndarray:
    e, phi = np.broadcast_arrays(np.asarray(e, np.float64), np.asarray(phi, np.float64))
    return np.stack([np.sin(e) * np.cos(phi), np.sin(e) * np.sin(phi), -np.cos(e)], axis=-1)


def dir_of_qphi(q, phi) -> np.ndarray:
    return dir_of_e_phi(e_of_q(q), phi)


def native_px_per_q(model, q, phi, h: float = 1e-5) -> np.ndarray:
    """Native raster pixels per unit q along the radial line (central difference through pixel_of).
    Its inverse is dq/dw, #5's Jacobian (2K/N)(E2 + e)/(sin e cos e) for the Foveated sensor."""
    rp, _ = model.pixel_of(dir_of_qphi(np.asarray(q) + h, phi))
    rm, _ = model.pixel_of(dir_of_qphi(np.asarray(q) - h, phi))
    return np.linalg.norm(rp - rm, axis=-1) / (2 * h)


# ----------------------------------------------------------------------------- depth

def depth_of_d(d, B: float) -> tuple[np.ndarray, np.ndarray]:
    """(z_a, z_b) from the log-tangent displacement d = q_b - q_a; nan where d <= 0 (no finite depth)."""
    d = np.asarray(d, dtype=np.float64)
    ok = d > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        za = np.where(ok, B / -np.expm1(-d), np.nan)
        zb = np.where(ok, B / np.expm1(d), np.nan)
    return za, zb


def range_b(d, B: float, e_b) -> np.ndarray:
    return depth_of_d(d, B)[1] / np.cos(e_b)


def range_interval(d_lo, d_hi, B: float, e_b) -> tuple[np.ndarray, np.ndarray]:
    """Range interval along b's ray for a displacement interval [d_lo, d_hi]: the larger d, the nearer.
    The far bound is inf when d_lo <= 0 (the interval reaches zero displacement); both are nan when
    d_hi <= 0."""
    near = range_b(d_hi, B, e_b)
    far = np.where(np.asarray(d_lo) > 0, range_b(np.maximum(d_lo, 1e-300), B, e_b), np.inf)
    far = np.where(np.isfinite(near), far, np.nan)
    return near, far


def world_points(c_b, R_b, e_b, phi, r) -> np.ndarray:
    """World points at range r along b's ray (e_b, phi); R_b maps eye-frame to world."""
    return np.asarray(c_b) + np.asarray(r)[..., None] * (dir_of_e_phi(e_b, phi) @ np.asarray(R_b).T)


def project_static(P_w, c, R) -> np.ndarray:
    """World points -> eye-frame unit directions from eye centre c with eye-to-world rotation R."""
    return unit(np.asarray(P_w) - np.asarray(c)) @ np.asarray(R)


# ----------------------------------------------------------------------------- the analysis grid

@dataclass
class Grid:
    """A (q, phi) grid over [e_lo, e_hi], mapped onto a sensor's native raster.

    rc (n_q, n_phi, 2)   continuous native (row, col) of each grid node (pixel centres at integers)
    support (n_q, n_phi) the node's whole bilinear footprint lies on valid native pixels
    px_per_q (n_q, n_phi) native pixels per unit q along the radius; px_per_phi likewise per radian
    The grid resamples what the sensor acquired; it adds no optical resolution."""
    q: np.ndarray
    phi: np.ndarray
    rc: np.ndarray
    support: np.ndarray
    px_per_q: np.ndarray
    px_per_phi: np.ndarray

    @property
    def shape(self):
        return self.rc.shape[:2]

    @property
    def dq(self) -> float:
        return float(self.q[1] - self.q[0])

    @property
    def dphi(self) -> float:
        return float(self.phi[1] - self.phi[0])

    @property
    def e(self) -> np.ndarray:
        return e_of_q(self.q)


def make_grid(model, n_q: int = 160, n_phi: int = 800, e_lo_deg: float = 1.0, e_hi_deg: float = 45.0) -> Grid:
    q = np.linspace(q_of_e(e_lo_deg * DEG), q_of_e(e_hi_deg * DEG), n_q)
    phi = -np.pi + (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)          # periodic, no seam node
    Q, PH = np.meshgrid(q, phi, indexing="ij")
    rc, inside = model.pixel_of(dir_of_qphi(Q, PH))
    valid = model.rays().valid
    h, w = valid.shape
    i0, j0 = np.floor(rc[..., 0]).astype(np.int64), np.floor(rc[..., 1]).astype(np.int64)
    sup = inside & (i0 >= 0) & (j0 >= 0) & (i0 + 1 < h) & (j0 + 1 < w)
    for di in (0, 1):
        for dj in (0, 1):
            ii, jj = np.clip(i0 + di, 0, h - 1), np.clip(j0 + dj, 0, w - 1)
            sup &= valid[ii, jj]
    hq, hp = 1e-5, 1e-5
    rq = model.pixel_of(dir_of_qphi(Q + hq, PH))[0] - model.pixel_of(dir_of_qphi(Q - hq, PH))[0]
    rp = model.pixel_of(dir_of_qphi(Q, PH + hp))[0] - model.pixel_of(dir_of_qphi(Q, PH - hp))[0]
    return Grid(q, phi, rc, sup, np.linalg.norm(rq, axis=-1) / (2 * hq), np.linalg.norm(rp, axis=-1) / (2 * hp))


def bilinear(img, rc) -> np.ndarray:
    """Sample img (h, w[, C]) at continuous (row, col); nodes outside the raster give nan."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    r, c = rc[..., 0], rc[..., 1]
    i0, j0 = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
    fr, fc = r - i0, c - j0
    out_of = (i0 < 0) | (j0 < 0) | (i0 + 1 >= h) | (j0 + 1 >= w)
    i0, j0 = np.clip(i0, 0, h - 2), np.clip(j0, 0, w - 2)
    if img.ndim == 3:
        fr, fc = fr[..., None], fc[..., None]
    v = (img[i0, j0] * (1 - fr) * (1 - fc) + img[i0 + 1, j0] * fr * (1 - fc)
         + img[i0, j0 + 1] * (1 - fr) * fc + img[i0 + 1, j0 + 1] * fr * fc)
    if img.ndim == 3:
        out_of = out_of[..., None]
    return np.where(out_of, np.nan, v)


def unwrap(img, grid: Grid) -> np.ndarray:
    """J(q, phi): the native image resampled on the grid; nan where the bilinear support leaves the disk."""
    J = bilinear(img, grid.rc)
    m = grid.support if J.ndim == 2 else grid.support[..., None]
    return np.where(m, J, np.nan)
