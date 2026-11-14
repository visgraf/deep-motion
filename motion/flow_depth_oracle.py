"""M3's oracle correspondence sources, the ladder O1-O3 (M3a #6, Stage 2). Evaluation side only:
these read oracle fields (b's Position, the M1.3 visibility test); `motion.flow_depth` never imports
this module, and every output records which source produced it.

    O1 oracle-exact   b's point P (its Position, the pixel's mean) seen from both eye centres: e_b, phi_b
                      and q_a all exact. Isolates geometry and PLY bugs; its cloud is the reference.
    O2 oracle-pixel   b anchored at its pixel centre (the record's direction), P's projection into a
                      rounded to a's pixel centre. Ties M3 to M2's pixel-limited range; its interval is
                      half a's pixel along the radial line (as M2's bracket along the epipolar curve).
                      phi_a and phi_b then differ by the rounding; M3's model ignores that.
    O3 oracle-noise   O1 with a's location moved along the radial line by Gaussian noise of sigma native
                      pixels (fixed seed); interval +/- 1.96 sigma. Coverage and error vs. the matcher's
                      precision. Its empirical 95 % coverage is a check of the interval propagation.

Correspondences are made only for b's hits that a sees at the projected pixel (M1.3's agreement):
jointly visible, decided by the oracle. That is legitimate for an oracle source and is never used to
accept RGB matches.
"""
from __future__ import annotations

import numpy as np

from . import analysis
from . import logpolar as lp

Z95 = 1.959963984540054


def _radial_half_pixel(model, q, phi):
    """Half a's native pixel along the radial line through (q, phi), in q units: the unit square's
    half-extent along the line's raster direction, divided by native px per q."""
    h = 1e-5
    rp, _ = model.pixel_of(lp.dir_of_qphi(q + h, phi))
    rm, _ = model.pixel_of(lp.dir_of_qphi(q - h, phi))
    g = (rp - rm) / (2 * h)
    gn = np.linalg.norm(g, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = g / gn[:, None]
        return 0.5 / np.max(np.abs(u), axis=-1) / gn


def correspondences(la, lb, kind: str, **kw) -> dict:
    """la, lb: analysis.load_look tuples (oracle access). sigma_px, seed: O3's noise. yaw_b_deg: the
    alignment control (b's eye yawed about world +Z by this much, as a wrong pose would be).
    A point on a's axis has q_a = -inf: its d is not finite and it counts as unresolved, not as a range."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _correspondences(la, lb, kind, **kw)


def _correspondences(la, lb, kind: str, *, sigma_px: float = 0.0, seed: int = 0, yaw_b_deg: float = 0.0) -> dict:
    rec_a, Ra, ca, model = la
    rec_b, Rb, cb, _ = lb
    if yaw_b_deg:
        t = np.radians(yaw_b_deg)
        Rb = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]]) @ Rb
    hit = np.flatnonzero(np.isfinite(rec_b.distance_m))
    P = rec_b.position_w[hit].astype(np.float64)
    agree, _, rc_a, _ = analysis._agree(rec_a, model, Ra, ca, P)
    s, P = hit[agree], P[agree]
    da_exact = lp.project_static(P, ca, Ra)
    qa, pa = lp.qphi(da_exact)
    if kind == "oracle-pixel":
        db = lp.unit(rec_b.direction[s].astype(np.float64)) @ Rb
        n = model.n
        ij = np.clip(np.rint(rc_a[agree]).astype(np.int64), 0, n - 1)
        da = lp.unit(model.rays().direction[ij[:, 0], ij[:, 1]])
        qa, pa = lp.qphi(da)
    else:
        db = lp.project_static(P, cb, Rb)
    e_b, pb = lp.e_phi(db)
    qb = lp.q_of_e(e_b)
    if kind == "oracle-noise":
        g = lp.native_px_per_q(model, qa, pa)
        qa = qa + np.random.default_rng(seed).normal(size=qa.shape) * sigma_px / g
        half = Z95 * sigma_px / g
    elif kind == "oracle-pixel":
        half = _radial_half_pixel(model, qa, pa)
    elif kind == "oracle-exact":
        half = np.zeros_like(qa)
    else:
        raise ValueError(f"unknown oracle source {kind}")
    d = qb - qa
    return {"source": kind, "sigma_px": sigma_px, "seed": seed, "sample_b": s, "e_b": e_b, "phi_b": pb,
            "phi_a": pa, "q_a": qa, "q_b": qb, "d": d, "d_lo": d - half, "d_hi": d + half,
            "truth_range": np.linalg.norm(P - cb, axis=-1), "truth_P": P,
            "parallax_deg": np.degrees(np.arccos(np.clip(np.sum(lp.project_static(P, ca, Ra)
                                                              * lp.project_static(P, cb, Rb), -1), -1, 1))),
            "n_hits_b": int(hit.size), "ecc_hits_b_deg": np.degrees(lp.e_phi(lp.unit(rec_b.direction[hit]) @ Rb)[0])}
