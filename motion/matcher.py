"""M3's RGB correspondence under aligned forward translation (M3a #6, Stage 3). Observation only:
it sees two luminance rasters, the sensor model and the baseline length, nothing else.

Under a step along the eye's axis a static point keeps its azimuth phi and moves outward in
q = ln tan e by d = q_b - q_a > 0. The search is one-dimensional, along fixed-phi lines.

1. **Coarse, on the log-polar grid.** Both images are resampled on a (q, phi) grid with uniform q step.
   There a fronto-parallel patch moves by the same q shift everywhere, so every node is searched at
   once by shifting the whole grid: zero-mean NCC over a window of +/- R_HALF native pixels radially
   and +/- A_HALF native pixels in azimuth (window sizes per grid row, in native pixels: A3). The
   curve over d gives the best shift, an ambiguity test (second-best peak outside +/- EXCL_PX) and
   the boundary flags. d = 0 is included as the explicit no-displacement hypothesis.
2. **Fine, on the native raster.** At each anchor (a native pixel of b at 1-44 deg), Gauss-Newton on
   I_b(q_b + u, phi + v) = alpha I_a(q_b + u - d - kappa u, phi + v) + beta, with u, v at native-pixel
   offsets of b, sampled through `pixel_of` (bilinear on the raster, never resampled twice).
   kappa is the radial stretch of a surface not facing the eye (A1): about B / z_b for one parallel
   to the heading. sigma_d from the Gauss-Newton covariance with the residual variance.
3. **Rejections**, each with its reason: invalid support, low radial texture (A2: the rms radial
   gradient per native pixel relative to the patch mean; edges along a radial line carry none),
   low NCC, ambiguous, beyond the depth envelope on the far side (unresolved, incl. d ~ 0), at the
   near boundary, Gauss-Newton moved > 1.5 px or kappa out of range, forward-backward inconsistent
   (the coarse search anchored in a, read at the matched q_a), and unsupported: fewer than
   support_min of the 8 neighbouring native pixels hold a candidate within support_tol_px. A correct
   match on a surface agrees with its neighbours; a false peak of the long 1-D search does not. This
   is a test, not a filter: no value is smoothed or replaced, and pixels on either side of a depth
   edge keep the support of their own surface. Last, depth_edge: a candidate within +/- edge_half_px
   (the patch's reach) whose d deviates from the anchor's own local model d + kappa (q_nb - q) by more
   than edge_tol_px. A patch straddling a depth edge is dragged to the more textured side (foreground
   fattening); its neighbours across the edge reveal it. The edge is left as a hole (#5).

Settings are fixed in SETTINGS before any measurement; M3a Stage 4 calibrates only the interval
scale on the declared calibration-room set, and the result is frozen.
"""
from __future__ import annotations

import numpy as np

from . import logpolar as lp

SETTINGS = {
    "coarse_dq": 0.01,          # log-tangent step of the coarse grid
    "n_phi": 800,               # azimuth samples of the coarse grid (1.0 native px at 45 deg)
    "e_grid_lo_deg": 0.1,       # the coarse grid reaches below the anchors: a's points are less eccentric
    "e_anchor_deg": (1.0, 44.0),
    "r_half_px": 3,             # patch: +/- 3 native px radially (7 samples)
    "a_half_px": 1,             # +/- 1 native px in azimuth (3 samples)
    "z_envelope_m": (0.25, 20.0),
    "excl_px": 1.5,             # second peak searched outside +/- 1.5 native px of the best
    "ncc_min": 0.8,
    "ambig_margin": 0.05,       # rejected when the second peak is within 0.05 of the best NCC
    "texture_min": 0.02,        # rms radial gradient per native px / patch mean
    "gn_iters": 8,
    "gn_max_move_px": 1.5,
    "kappa_range": (-0.5, 1.5),
    "fb_tol_px": 1.0,
    "support_min": 3,           # of the 8 neighbouring anchors, those with a candidate within support_tol_px
    "support_tol_px": 0.5,      # (added after the first fixture run, #6: accepted false peaks at stride 10)
    "half_tol_px": 0.5,         # half-window test: inner and outer radial halves fitted alone must agree ...
    "half_sigma_max_px": 0.25,  # ... and each be informative (added after calibration-room pair (0, 5), #6)
    "edge_half_px": 3,          # depth-edge test over +/- 3 native px (the patch's radial reach) ...
    "edge_tol_px": 0.5,         # ... no candidate may deviate from the anchor's d + kappa u by more (added
                                # after calibration-room pair (0, 5), #6: foreground fattening at card edges)
    "stretch": True,            # False: shift-only (the A1 control)
    "kappa_starts": (0.0, 0.25, 0.5),   # Gauss-Newton starts (added after the first fixture run, #6)
    "restart_below_ncc": 0.97,          # ... tried only where the first start's NCC is below this
    "interval_scale": 1.0,      # calibrated in Stage 4, then frozen
}
Z95 = 1.959963984540054
REASONS = ("accepted", "invalid_support", "low_texture", "low_ncc", "ambiguous", "unresolved_far",
           "near_boundary", "gn_failed", "fb_inconsistent", "unsupported", "depth_edge", "half_disagree")


# ----------------------------------------------------------------------------- sampling

def sample(img, model, q, phi):
    """Bilinear samples of a native raster (nan off the valid disk) at (q, phi); nan when any of the
    four neighbours is invalid or the direction is outside the sensor."""
    rc, inside = model.pixel_of(lp.dir_of_qphi(q, phi))
    v = lp.bilinear(img, rc)
    return np.where(inside, v, np.nan)


def luminance(rgb_raster) -> np.ndarray:
    return np.asarray(rgb_raster, np.float64) @ np.array([0.2126, 0.7152, 0.0722])


# ----------------------------------------------------------------------------- coarse

def _coarse_grid(model, s):
    dq = s["coarse_dq"]
    q_hi = 0.0                                     # e = 45 deg
    n = int(np.ceil((q_hi - lp.q_of_e(np.radians(s["e_grid_lo_deg"]))) / dq)) + 1
    q = q_hi - dq * np.arange(n)[::-1]
    phi = -np.pi + (np.arange(s["n_phi"]) + 0.5) * (2 * np.pi / s["n_phi"])
    Q, PH = np.meshgrid(q, phi, indexing="ij")
    rc, inside = model.pixel_of(lp.dir_of_qphi(Q, PH))
    gq = lp.native_px_per_q(model, q, np.zeros_like(q))            # radially symmetric sensor rows
    h = 1e-5
    gp = np.linalg.norm(model.pixel_of(lp.dir_of_qphi(q, h + 0 * q))[0]
                        - model.pixel_of(lp.dir_of_qphi(q, -h + 0 * q))[0], axis=-1) / (2 * h)
    return q, phi, rc, inside, gq, gp


def _unwrap(img, rc, inside):
    v = lp.bilinear(img, rc)
    return np.where(inside, v, np.nan)


def _box(X, ri, h, w, wmax):
    """Window sums: rows ri +/- h (per row), periodic columns +/- w (per row)."""
    C = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    S = C[ri + h + 1] - C[ri - h]
    P = S.shape[1]
    if wmax:
        S = np.concatenate([S[:, -wmax:], S, S[:, :wmax]], 1)
    C2 = np.concatenate([np.zeros((S.shape[0], 1)), np.cumsum(S, 1)], 1)
    j = np.arange(P)[None, :] + wmax
    return np.take_along_axis(C2, j + w[:, None] + 1, 1) - np.take_along_axis(C2, j - w[:, None], 1)


def coarse(J_anchor, J_target, rows, hq, hp, T, sign):
    """NCC curves (len(rows), n_phi, T+1) for target row = anchor row + sign * t, t = 0..T.
    Windows that touch an invalid sample give nan."""
    R, P = J_anchor.shape
    pad = T + int(hq.max()) + 2
    big = lambda J: np.concatenate([np.full((pad, P), np.nan), J, np.full((pad, P), np.nan)])
    A, Tg = big(J_anchor), big(J_target)
    lo, hi = rows.min() - hq.max(), rows.max() + hq.max()
    ext = np.arange(lo, hi + 1) + pad
    ri = rows + pad - ext[0]
    wmax = int(hp.max())
    Ma = np.isfinite(A[ext])
    Ae = np.where(Ma, A[ext], 0.0)
    n_full = ((2 * hq + 1) * (2 * hp + 1)).astype(np.float64)[:, None]
    Sa = _box(Ae, ri, hq, hp, wmax)
    Saa = _box(Ae * Ae, ri, hq, hp, wmax)
    na = _box(Ma.astype(np.float64), ri, hq, hp, wmax)
    va = Saa - Sa * Sa / n_full
    out = np.full((len(rows), P, T + 1), np.nan, dtype=np.float32)
    for t in range(T + 1):
        Te = Tg[ext + sign * t]
        Mt = np.isfinite(Te)
        Te = np.where(Mt, Te, 0.0)
        M = Ma & Mt
        n = _box(M.astype(np.float64), ri, hq, hp, wmax)
        St = _box(Te, ri, hq, hp, wmax)
        Stt = _box(Te * Te, ri, hq, hp, wmax)
        Sat = _box(Ae * Te, ri, hq, hp, wmax)
        vt = Stt - St * St / n_full
        cov = Sat - Sa * St / n_full
        with np.errstate(invalid="ignore", divide="ignore"):
            c = cov / np.sqrt(va * vt)
        ok = (n == n_full) & (na == n_full) & (va > 1e-12) & (vt > 1e-12)
        out[:, :, t] = np.where(ok, c, np.nan)
    return out


def _peaks(curves, excl_rows):
    """Best t, its NCC, parabolic sub-step offset, and the best NCC outside +/- excl_rows."""
    c = np.where(np.isfinite(curves), curves, -np.inf)
    t1 = np.argmax(c, -1)
    c1 = np.take_along_axis(c, t1[..., None], -1)[..., 0]
    T = c.shape[-1]
    tt = np.arange(T)
    far = np.abs(tt[None, None, :] - t1[..., None]) > excl_rows[:, None, None]
    c2 = np.max(np.where(far, c, -np.inf), -1)
    cm = np.take_along_axis(c, np.clip(t1 - 1, 0, T - 1)[..., None], -1)[..., 0]
    cp = np.take_along_axis(c, np.clip(t1 + 1, 0, T - 1)[..., None], -1)[..., 0]
    den = cm - 2 * c1 + cp
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where((t1 > 0) & (t1 < T - 1) & np.isfinite(den) & (den < 0), 0.5 * (cm - cp) / den, 0.0)
    return t1, c1, np.clip(delta, -0.5, 0.5), c2


# ----------------------------------------------------------------------------- fine

def _gn(img_a, img_b, model, qb, pb, gq, gp, d0, s):
    """Gauss-Newton from each kappa start in s["kappa_starts"] (shift-only: kappa = 0 only); per anchor the
    start with the highest final NCC wins (a stretched surface is a local minimum away from kappa = 0)."""
    starts = s["kappa_starts"] if s["stretch"] else (0.0,)
    best = _gn1(img_a, img_b, model, qb, pb, gq, gp, d0, starts[0], s)
    retry = np.flatnonzero(~(best["ncc_fine"] >= s["restart_below_ncc"]))     # only imperfect fits restart
    for k0 in starts[1:]:
        if retry.size == 0:
            break
        r = _gn1(img_a, img_b, model, qb[retry], pb[retry], gq[retry], gp[retry], d0[retry], k0, s)
        take = np.nan_to_num(r["ncc_fine"], nan=-2) > np.nan_to_num(best["ncc_fine"][retry], nan=-2)
        for k in best:
            if k != "texture":
                best[k][retry] = np.where(take, r[k], best[k][retry])
    return best


def _half(img_a, img_b, model, qb, pb, gq, gp, d0, kap, s, side):
    """Shift-only Gauss-Newton (kappa held) on one radial half of the patch: side -1 inner (k <= 0), +1 outer."""
    rh, ah = s["r_half_px"], s["a_half_px"]
    k = np.arange(-rh, 1, dtype=np.float64) if side < 0 else np.arange(0, rh + 1, dtype=np.float64)
    j = np.arange(-ah, ah + 1, dtype=np.float64)
    K, Jj = np.meshgrid(k, j, indexing="ij")
    u = K.reshape(1, -1) / gq[:, None]
    v = Jj.reshape(1, -1) / gp[:, None]
    Qb, Pb = qb[:, None] + u, pb[:, None] + v
    Ib = sample(img_b, model, Qb, Pb)
    hq = 0.25 / gq[:, None]
    d = d0.copy()
    Ia = sample(img_a, model, Qb - d[:, None] - kap[:, None] * u, Pb)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.nanstd(Ib, -1) / np.nanstd(Ia, -1)
        beta = np.nanmean(Ib, -1) - alpha * np.nanmean(Ia, -1)
    for it in range(s["gn_iters"] + 1):
        Qa = Qb - d[:, None] - kap[:, None] * u
        Ia = sample(img_a, model, Qa, Pb)
        dI = (sample(img_a, model, Qa + hq, Pb) - sample(img_a, model, Qa - hq, Pb)) / (2 * hq)
        r = Ib - alpha[:, None] * Ia - beta[:, None]
        J = np.stack([alpha[:, None] * dI, -Ia, -np.ones_like(Ia)], -1)
        good = np.isfinite(r).all(-1) & np.isfinite(J).all((-1, -2))
        Jg, rg = np.where(good[:, None, None], J, 0.0), np.where(good[:, None], r, 0.0)
        H = np.einsum("nki,nkj->nij", Jg, Jg) + 1e-12 * np.eye(3)
        if it == s["gn_iters"]:
            s2 = np.sum(rg ** 2, -1) / max(Ib.shape[1] - 3, 1)
            with np.errstate(invalid="ignore"):
                sig = np.sqrt(np.maximum(s2 * np.linalg.inv(H)[:, 0, 0], 0.0))
            return np.where(good, d, np.nan), np.where(good, sig, np.nan)
        step = -np.linalg.solve(H, np.einsum("nki,nk->ni", Jg, rg)[..., None])[..., 0]
        d = d + np.where(good, step[:, 0], 0.0)
        alpha = alpha + np.where(good, step[:, 1], 0.0)
        beta = beta + np.where(good, step[:, 2], 0.0)


def _gn1(img_a, img_b, model, qb, pb, gq, gp, d0, k0, s):
    rh, ah = s["r_half_px"], s["a_half_px"]
    k = np.arange(-rh, rh + 1, dtype=np.float64)
    j = np.arange(-ah, ah + 1, dtype=np.float64)
    K, Jj = np.meshgrid(k, j, indexing="ij")
    u = K.reshape(1, -1) / gq[:, None]                       # q offsets, b's native px
    v = Jj.reshape(1, -1) / gp[:, None]
    Qb, Pb = qb[:, None] + u, pb[:, None] + v
    Ib = sample(img_b, model, Qb, Pb)
    hq = 0.25 / gq[:, None]
    gb = (sample(img_b, model, Qb + 0.5 / gq[:, None], Pb) - sample(img_b, model, Qb - 0.5 / gq[:, None], Pb))
    with np.errstate(invalid="ignore", divide="ignore"):
        texture = np.sqrt(np.mean(gb ** 2, -1)) / np.abs(np.mean(Ib, -1))
    d, kap = d0.copy(), np.full_like(d0, k0)
    ma, sa = np.nanmean(Ib, -1), np.nanstd(Ib, -1)
    Ia0 = sample(img_a, model, Qb - d[:, None] - kap[:, None] * u, Pb)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = sa / np.nanstd(Ia0, -1)
        beta = ma - alpha * np.nanmean(Ia0, -1)
    npar = 4 if s["stretch"] else 3
    ok = np.isfinite(Ib).all(-1)
    for _ in range(s["gn_iters"]):
        Qa = Qb - d[:, None] - kap[:, None] * u
        Ia = sample(img_a, model, Qa, Pb)
        dI = (sample(img_a, model, Qa + hq, Pb) - sample(img_a, model, Qa - hq, Pb)) / (2 * hq)
        r = Ib - alpha[:, None] * Ia - beta[:, None]
        cols = [alpha[:, None] * dI, -Ia, -np.ones_like(Ia)]
        if s["stretch"]:
            cols.insert(1, alpha[:, None] * dI * u)
        J = np.stack(cols, -1)                                    # (N, K, npar)
        good = ok & np.isfinite(r).all(-1) & np.isfinite(J).all((-1, -2))
        Jg = np.where(good[:, None, None], J, 0.0)
        rg = np.where(good[:, None], r, 0.0)
        H = np.einsum("nki,nkj->nij", Jg, Jg) + 1e-12 * np.eye(npar)
        gvec = np.einsum("nki,nk->ni", Jg, rg)
        step = -np.linalg.solve(H, gvec[..., None])[..., 0]
        d = d + np.where(good, step[:, 0], 0.0)
        if s["stretch"]:
            kap = kap + np.where(good, step[:, 1], 0.0)
        alpha = alpha + np.where(good, step[:, -2], 0.0)
        beta = beta + np.where(good, step[:, -1], 0.0)
        ok = good
    Qa = Qb - d[:, None] - kap[:, None] * u
    Ia = sample(img_a, model, Qa, Pb)
    dI = (sample(img_a, model, Qa + hq, Pb) - sample(img_a, model, Qa - hq, Pb)) / (2 * hq)
    r = Ib - alpha[:, None] * Ia - beta[:, None]
    cols = [alpha[:, None] * dI, -Ia, -np.ones_like(Ia)]
    if s["stretch"]:
        cols.insert(1, alpha[:, None] * dI * u)
    J = np.stack(cols, -1)
    good = ok & np.isfinite(r).all(-1) & np.isfinite(J).all((-1, -2))
    Jg, rg = np.where(good[:, None, None], J, 0.0), np.where(good[:, None], r, 0.0)
    H = np.einsum("nki,nkj->nij", Jg, Jg) + 1e-12 * np.eye(npar)
    dof = max(Ib.shape[1] - npar, 1)
    s2 = np.sum(rg ** 2, -1) / dof
    with np.errstate(invalid="ignore"):
        cov00 = np.linalg.inv(H)[:, 0, 0]
        sigma_d = np.sqrt(np.maximum(s2 * cov00, 0.0))
        za, zb = Ib - Ib.mean(-1, keepdims=True), Ia - Ia.mean(-1, keepdims=True)
        ncc = np.sum(za * zb, -1) / np.sqrt(np.sum(za ** 2, -1) * np.sum(zb ** 2, -1))
    return {"d": d, "kappa": kap, "sigma_d": sigma_d, "ncc_fine": ncc, "valid": good, "texture": texture}


# ----------------------------------------------------------------------------- the pair

def match_pair(img_a, img_b, model, B: float, settings: dict | None = None) -> dict:
    """img_a, img_b: luminance rasters (nan off the disk) of the earlier and later look, aligned forward
    step B > 0. Returns per anchor (valid native pixels of b at e_anchor_deg): flat raster index,
    e_b, phi_b, d and its interval, reason code, diagnostics, and the coarse curves of the anchors."""
    s = dict(SETTINGS, **(settings or {}))
    q, phi, rc, inside, gq_row, gp_row = _coarse_grid(model, s)
    Ja, Jb = _unwrap(img_a, rc, inside), _unwrap(img_b, rc, inside)
    dq, dphi = s["coarse_dq"], 2 * np.pi / s["n_phi"]
    zmin, zmax = s["z_envelope_m"]
    d_min, d_max = np.log1p(B / zmax), np.log1p(B / zmin)
    T = int(np.ceil(d_max / dq))
    hq_all = np.maximum(1, np.rint(s["r_half_px"] / (gq_row * dq))).astype(int)
    hp_all = np.maximum(0, np.rint(s["a_half_px"] / (gp_row * dphi))).astype(int)
    e_lo, e_hi = np.radians(s["e_anchor_deg"][0]), np.radians(s["e_anchor_deg"][1])
    rows_b = np.flatnonzero((q >= lp.q_of_e(e_lo) - 2 * dq) & (q <= lp.q_of_e(e_hi) + 2 * dq))
    fwd = coarse(Jb, Ja, rows_b, hq_all[rows_b], hp_all[rows_b], T, -1)
    excl_b = np.ceil(s["excl_px"] / (gq_row[rows_b] * dq)).astype(int)
    t1, c1, dl, c2 = _peaks(fwd, excl_b)
    # anchors: valid native pixels of b in the anchor band
    rays = model.rays()
    dirs = rays.direction
    e_px, phi_px = lp.e_phi(dirs)
    sel = rays.valid & (e_px >= e_lo) & (e_px <= e_hi)
    flat = np.flatnonzero(sel.reshape(-1))
    eb, pb = e_px.reshape(-1)[flat], phi_px.reshape(-1)[flat]
    qb = lp.q_of_e(eb)
    ir = np.clip(np.rint((qb - q[rows_b[0]]) / dq).astype(int), 0, len(rows_b) - 1)
    ic = np.mod(np.rint((pb - phi[0]) / dphi).astype(int), s["n_phi"])
    tb, cb_, db_, c2b = t1[ir, ic], c1[ir, ic], dl[ir, ic], c2[ir, ic]
    gq_b = lp.native_px_per_q(model, qb, pb)
    gp_b = np.linalg.norm(model.pixel_of(lp.dir_of_e_phi(eb, pb + 1e-5))[0]
                          - model.pixel_of(lp.dir_of_e_phi(eb, pb - 1e-5))[0], axis=-1) / 2e-5
    d0 = np.clip((tb + db_) * dq, 0.0, None)
    fine = _gn(img_a, img_b, model, qb, pb, gq_b, gp_b, d0, s)
    d = fine["d"]
    qa = qb - d
    gq_a = lp.native_px_per_q(model, np.where(np.isfinite(qa), qa, qb), pb)
    # backward: the coarse search anchored in a, read at the matched q_a
    rows_a = np.flatnonzero((q >= lp.q_of_e(e_lo) - d_max - 2 * dq) & (q <= lp.q_of_e(e_hi) + 2 * dq))
    rows_a = rows_a[rows_a >= int(hq_all.max())]
    bwd = coarse(Ja, Jb, rows_a, hq_all[rows_a], hp_all[rows_a], T, +1)
    excl_a = np.ceil(s["excl_px"] / (gq_row[rows_a] * dq)).astype(int)
    ta, ca, dla, _ = _peaks(bwd, excl_a)
    ira = np.clip(np.rint((np.where(np.isfinite(qa), qa, qb) - q[rows_a[0]]) / dq).astype(int), 0, len(rows_a) - 1)
    d_back = (ta[ira, ic] + dla[ira, ic]) * dq
    # reasons, first that applies
    reason = np.zeros(len(flat), dtype=np.int8)

    def mark(cond, code):
        reason[(reason == 0) & cond] = code
    mark(~np.isfinite(cb_) | ~fine["valid"] | ~np.isfinite(d), REASONS.index("invalid_support"))
    mark(~(fine["texture"] >= s["texture_min"]), REASONS.index("low_texture"))
    mark(~(cb_ >= s["ncc_min"]), REASONS.index("low_ncc"))
    mark(c2b >= cb_ - s["ambig_margin"], REASONS.index("ambiguous"))
    mark((d < d_min) | (tb * dq < d_min), REASONS.index("unresolved_far"))
    mark(tb >= T, REASONS.index("near_boundary"))
    moved = np.abs(d - d0) * gq_a
    k_lo, k_hi = s["kappa_range"]
    mark(~(moved <= s["gn_max_move_px"]) | ~(fine["kappa"] >= k_lo) | ~(fine["kappa"] <= k_hi)
         | ~(fine["ncc_fine"] >= s["ncc_min"]) | ~np.isfinite(fine["sigma_d"]), REASONS.index("gn_failed"))
    mark(~(np.abs(d - d_back) * gq_a <= s["fb_tol_px"]), REASONS.index("fb_inconsistent"))
    # neighbour support among the candidates that passed every other test
    n = model.n
    D = np.full(n * n, np.nan)
    cand = reason == 0
    D[flat[cand]] = (d * gq_a)[cand]                 # in a's native px (q scale of each anchor)
    D = D.reshape(n, n)
    G = np.full(n * n, np.nan)
    G[flat] = gq_a
    G = G.reshape(n, n)
    sup = np.zeros((n, n), dtype=np.int16)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            Dn = np.full((n, n), np.nan)
            src = D[max(di, 0):n + min(di, 0), max(dj, 0):n + min(dj, 0)]
            Dn[max(-di, 0):n + min(-di, 0), max(-dj, 0):n + min(-dj, 0)] = src
            Gn = np.full((n, n), np.nan)
            Gn[max(-di, 0):n + min(-di, 0), max(-dj, 0):n + min(-dj, 0)] = G[max(di, 0):n + min(di, 0), max(dj, 0):n + min(dj, 0)]
            with np.errstate(invalid="ignore"):
                sup += (np.abs(Dn / Gn - D / G) * G <= s["support_tol_px"]).astype(np.int16)
    support = sup.reshape(-1)[flat]
    mark(cand & (support < s["support_min"]), REASONS.index("unsupported"))
    # half windows: inner and outer radial halves, each alone (kappa held), must be informative and agree
    ci = np.flatnonzero(reason == 0)
    half_dev = np.full(len(flat), np.nan)
    half_sig = np.full(len(flat), np.nan)
    if ci.size:
        args = (img_a, img_b, model, qb[ci], pb[ci], gq_b[ci], gp_b[ci], d[ci], fine["kappa"][ci], s)
        d1, s1 = _half(*args, side=-1)
        d2, s2_ = _half(*args, side=+1)
        half_dev[ci] = np.abs(d1 - d2) * gq_a[ci]
        half_sig[ci] = np.fmax(s1, s2_) * gq_a[ci]
    mark((reason == 0) & ~((half_dev <= s["half_tol_px"]) & (half_sig <= s["half_sigma_max_px"])),
         REASONS.index("half_disagree"))
    # depth edges: neighbours (candidates that passed every test above) against the anchor's local model
    cand2 = reason == 0
    Dq = np.full(n * n, np.nan)
    Dq[flat[cand2]] = d[cand2]
    Qb = np.full(n * n, np.nan)
    Qb[flat] = qb
    Dq, Qb = Dq.reshape(n, n), Qb.reshape(n, n)
    Kr = np.full(n * n, np.nan)
    Kr[flat] = fine["kappa"]
    Kr = Kr.reshape(n, n)
    dev = np.zeros((n, n))
    h = int(s["edge_half_px"])
    for di in range(-h, h + 1):
        for dj in range(-h, h + 1):
            if di == 0 and dj == 0:
                continue
            Dn = np.full((n, n), np.nan)
            Qn = np.full((n, n), np.nan)
            sl_src = (slice(max(di, 0), n + min(di, 0)), slice(max(dj, 0), n + min(dj, 0)))
            sl_dst = (slice(max(-di, 0), n + min(-di, 0)), slice(max(-dj, 0), n + min(-dj, 0)))
            Dn[sl_dst], Qn[sl_dst] = Dq[sl_src], Qb[sl_src]
            with np.errstate(invalid="ignore"):
                dv = np.abs(Dn - (Dq + Kr * (Qn - Qb))) * G
            dev = np.fmax(dev, np.where(np.isfinite(dv), dv, 0.0))
    edge_dev = dev.reshape(-1)[flat]
    mark(cand2 & (edge_dev > s["edge_tol_px"]), REASONS.index("depth_edge"))
    half = Z95 * s["interval_scale"] * fine["sigma_d"]
    return {"flat": flat, "e_b": eb, "phi_b": pb, "q_b": qb, "d": d, "d_lo": d - half, "d_hi": d + half,
            "sigma_d": fine["sigma_d"], "sigma_px": fine["sigma_d"] * gq_a, "kappa": fine["kappa"],
            "ncc_coarse": cb_, "ncc_second": c2b, "ncc_fine": fine["ncc_fine"], "texture": fine["texture"],
            "d_coarse": d0, "d_back": d_back, "px_per_q_a": gq_a, "reason": reason, "support": support, "edge_dev_px": edge_dev, "half_dev_px": half_dev,
            "half_sigma_px": half_sig,
            "envelope": {"d_min": float(d_min), "d_max": float(d_max), "T": T}, "settings": s}
