"""M4 geometry and the multi-frame oracle ladder (M4a #10, Stages 1-2), numpy only.

Under the aligned dolly an anchor of reference frame r, at axial inverse depth zeta = 1 / z_r, is seen by
frame k (signed step t_k = (c_k - c_r) . h along the heading h) at the same azimuth and at

    d_k(zeta) = q_k - q_r = -ln(1 - t_k zeta),        valid while t_k zeta < 1,

so one unknown is shared by every frame (Okutomi & Kanade's multiple-baseline stereo, in the log-polar
coordinate). With the anchor's own ray exact (q_r is the anchor's), estimating zeta from the d_k is a
one-parameter weighted least-squares problem:

    d d_k / d zeta = t_k / (1 - t_k zeta),     sigma_zeta^2 = 1 / sum_k (t_k / (1 - t_k zeta))^2 / sigma_k^2

with sigma_k the per-frame displacement noise in q. Range along r's ray: r_r = 1 / (zeta cos e_r).

The oracle ladder O-M (evaluation side): for each of r's hits, b's Position P projected into every window
frame k that sees it (M1.3's agreement test), with per-frame Gaussian noise of sigma native px along the
radial line, or rounded to k's pixel centre. Compare with M3's two-frame O-ladder.
"""
from __future__ import annotations

import numpy as np

from . import analysis
from . import logpolar as lp

Z95 = 1.959963984540054


def d_of_zeta(t, zeta):
    t, zeta = np.broadcast_arrays(np.asarray(t, float), np.asarray(zeta, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(t * zeta < 1, -np.log1p(-t * zeta), np.nan)


def ddz(t, zeta):
    with np.errstate(invalid="ignore", divide="ignore"):
        return t / (1 - t * zeta)


def zeta_init(t, d, sig, mask):
    """Each frame inverts alone: zeta_k = -expm1(-d_k) / t_k; their median weighted by |t_k| / sigma_k."""
    with np.errstate(invalid="ignore", divide="ignore"):
        zk = -np.expm1(-d) / t
    w = np.where(mask & np.isfinite(zk), np.abs(t) / np.maximum(sig, 1e-300), 0.0)
    zk = np.where(w > 0, zk, np.inf)
    o = np.argsort(zk, 1)
    zs, ws = np.take_along_axis(zk, o, 1), np.take_along_axis(w, o, 1)
    cw = np.cumsum(ws, 1)
    i = np.argmax(cw >= 0.5 * cw[:, -1:], 1)
    z0 = zs[np.arange(len(zs)), i]
    return np.where(np.isfinite(z0) & (cw[:, -1] > 0), np.clip(z0, 1e-6, None), 0.2)


def fit_zeta(t, d, sig, mask, zeta0=None, iters=20):
    """Per anchor (rows), WLS Gauss-Newton for zeta from the window's d_k (columns), sig the per-frame noise in
    q, mask the frames that contribute; started from zeta_init, steps kept where t_k zeta < 1 for every
    contributing frame. Returns zeta, sigma_zeta (Fisher), the number of frames."""
    zeta = zeta_init(t, d, sig, mask) if zeta0 is None else zeta0.copy()
    w = np.where(mask, 1.0 / np.maximum(sig, 1e-300) ** 2, 0.0)
    tpos = np.where(mask & (t > 0), t, 0.0).max(1)
    zmax = np.where(tpos > 0, 0.999 / np.maximum(tpos, 1e-300), np.inf)
    for _ in range(iters):
        J = ddz(t, zeta[:, None])
        r = np.where(mask, d - d_of_zeta(t, zeta[:, None]), 0.0)
        Jm = np.where(mask & np.isfinite(J), J, 0.0)
        H = np.sum(w * Jm * Jm, 1)
        g = np.sum(w * Jm * np.nan_to_num(r), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.where(H > 0, g / H, 0.0)
        new = zeta + step
        zeta = np.clip(np.where(new >= zmax, 0.5 * (zeta + zmax), new), 1e-6, None)
    J = ddz(t, zeta[:, None])
    H = np.sum(w * np.where(mask & np.isfinite(J), J, 0.0) ** 2, 1)
    with np.errstate(divide="ignore"):
        return zeta, np.where(H > 0, 1.0 / np.sqrt(H), np.inf), mask.sum(1)


def window(ks, r, W, sides):
    """Frames of the window: sides 'before' (r-W..r-1), 'after' (r+1..r+W) or 'both'."""
    lo = r - W if sides in ("before", "both") else r + 1
    hi = r + W if sides in ("after", "both") else r - 1
    return [k for k in ks if lo <= k <= hi and k != r]


def oracle(run, r, frames, kind, sigma_px=0.0, seed=0, loads=None):
    """O-M for reference frame r over the given frames. kind: 'exact', 'noise' (sigma px per frame, radial)
    or 'pixel' (rounded to each frame's pixel centre; half a pixel radially as its sigma for the fit).
    Returns per hit of r: e_r, range estimate, interval, truth range, frames used."""
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    loads = loads if loads is not None else {}
    get = lambda k: loads.setdefault(k, analysis.load_look(run, es[k]))
    rec_r, Rr, cr, model = get(r)
    h = Rr @ [0, 0, -1.0]
    hit = np.flatnonzero(np.isfinite(rec_r.distance_m))
    P = rec_r.position_w[hit].astype(np.float64)
    dr = lp.unit(rec_r.direction[hit].astype(np.float64)) @ Rr          # the anchor's (pixel-centre) ray
    e_r, phi_r = lp.e_phi(dr)
    q_r = lp.q_of_e(e_r)
    truth = np.linalg.norm(P - cr, axis=-1)
    X = cr + truth[:, None] * (dr @ Rr.T)                                # the truth point on the anchor's ray
    n, m = len(hit), len(frames)
    t = np.zeros((n, m))
    d = np.full((n, m), np.nan)
    sg = np.full((n, m), np.inf)
    mask = np.zeros((n, m), bool)
    rng = np.random.default_rng(seed + 7919 * r)
    for j, k in enumerate(frames):
        rec_k, Rk, ck, _ = get(k)
        t[:, j] = float((ck - cr) @ h)
        vis, _, rc_k, _ = analysis._agree(rec_k, model, Rk, ck, P)
        dk = lp.project_static(X, ck, Rk)
        qk, pk = lp.qphi(dk)
        g = lp.native_px_per_q(model, np.where(np.isfinite(qk), qk, 0.0), pk)
        if kind == "pixel":
            ij = np.clip(np.rint(rc_k).astype(np.int64), 0, model.n - 1)
            qk, _ = lp.qphi(lp.unit(model.rays().direction[ij[:, 0], ij[:, 1]]))
            s = 0.5 / g
        elif kind == "noise":
            qk = qk + rng.normal(size=n) * sigma_px / g
            s = sigma_px / g
        else:
            s = np.full(n, 1e-9)
        ok = vis & np.isfinite(qk) & np.isfinite(g) & (g > 0)
        d[:, j] = np.where(ok, qk - q_r, np.nan)
        sg[:, j] = np.where(ok, s, np.inf)
        mask[:, j] = ok
    zeta, sz, nf = fit_zeta(t, d, sg, mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        rng_est = 1.0 / (zeta * np.cos(e_r))
        lo_z, hi_z = zeta - Z95 * sz, zeta + Z95 * sz
        near = 1.0 / (hi_z * np.cos(e_r))
        far = np.where(lo_z > 0, 1.0 / (lo_z * np.cos(e_r)), np.inf)
        half = 0.5 * (far - near) / rng_est
    ok = nf >= 1
    return {"e_r_deg": np.degrees(e_r), "range": np.where(ok, rng_est, np.nan), "near": near, "far": far,
            "rel_half": np.where(ok, half, np.inf), "truth": truth, "frames_used": nf, "n_hits": n}


def summarize(res_list):
    cat = lambda k: np.concatenate([r[k] for r in res_list])
    e, rel_half = cat("e_r_deg"), cat("rel_half")
    rel = np.abs(cat("range") - cat("truth")) / cat("truth")
    cov = (cat("truth") >= cat("near")) & (cat("truth") <= cat("far"))
    usable = np.isfinite(rel_half) & (rel_half <= 0.10)
    out = {"hits": int(e.size), "field_usable": float(usable.mean()),
           "band_5_40_usable": float(usable[(e >= 5) & (e < 40)].mean()),
           "fovea_2deg_usable": float(usable[e < 2].mean()) if (e < 2).any() else None,
           "median_rel_usable": float(np.median(rel[usable])) if usable.any() else None,
           "coverage_finite": float(cov[np.isfinite(rel)].mean()) if np.isfinite(rel).any() else None}
    return out


# ----------------------------------------------------------------------------- Stage 1 selftest

def selftest():
    res, num = [], {}
    rng = np.random.default_rng(0)
    # 1. d_k(zeta) against the two-frame formula of M3: a pair at step B is t = -B seen from b... and +B from a
    z_b, B = 4.90104347826, 0.098956521739                # #5's known answer (z_a 5 m, e_a 20 deg)
    d_m3 = np.log((z_b + B) / z_b)                          # M3: q_b - q_a = ln(z_a / z_b) with z_a = z_b + B
    d_m4 = -d_of_zeta(-B, 1 / z_b)                          # M4: reference r = b, frame k = a at t = -B: q_a - q_b
    num["two_frame_consistency"] = float(abs(d_m4 - d_m3))
    res.append(("d_k(zeta) reproduces M3's two-frame displacement (#5's known answer) within 1e-12", abs(d_m4 - d_m3) <= 1e-12))
    # 2. exact recovery of zeta from analytic points across windows on both sides
    errs, ctrl_sign, ctrl_s = [], [], []
    for _ in range(200):
        z = rng.uniform(0.6, 15.0)
        tk = np.sort(rng.choice(np.arange(-10, 11), 6, replace=False)) * 0.0989565
        tk = tk[tk * (1 / z) < 0.95]
        if tk.size < 2:
            continue
        d = d_of_zeta(tk, 1 / z)
        zeta, sz, _ = fit_zeta(tk[None], d[None], np.full((1, tk.size), 1e-4), np.ones((1, tk.size), bool))
        errs.append(abs(zeta[0] * z - 1))
        zs, _, _ = fit_zeta(-tk[None], d[None], np.full((1, tk.size), 1e-4), np.ones((1, tk.size), bool))
        ctrl_sign.append(abs(zs[0] * z - 1))
        zw, _, _ = fit_zeta(tk[None] * 1.05, d[None], np.full((1, tk.size), 1e-4), np.ones((1, tk.size), bool))
        ctrl_s.append(abs(zw[0] * z - 1))
    num["zeta_recovery"] = {"n": len(errs), "max_rel": float(max(errs)), "control_sign_min_rel": float(min(ctrl_sign)),
                            "control_scale_median_rel": float(np.median(ctrl_s))}
    res.append(("analytic windows (both sides, 2-6 frames): zeta recovered within 1e-9", max(errs) <= 1e-9))
    res.append(("control: steps with the wrong sign give no correct zeta (every case off by >= 1 %)", min(ctrl_sign) >= 0.01))
    res.append(("control: steps 5 % too long give zeta off by >= 4 % (median)", float(np.median(ctrl_s)) >= 0.04))
    # 3. sigma_zeta: the Fisher bound against the empirical spread under Gaussian noise
    z, tk = 4.0, np.arange(-5, 6)[np.arange(-5, 6) != 0] * 0.0989565
    s = 0.003
    d0 = d_of_zeta(tk, 1 / z)
    N = 4000
    D = d0[None] + rng.normal(size=(N, tk.size)) * s
    zeta, sz, _ = fit_zeta(np.broadcast_to(tk, D.shape), D, np.full(D.shape, s), np.ones(D.shape, bool))
    ratio = float(np.std(zeta) / np.median(sz))
    num["sigma_zeta_check"] = {"empirical_over_fisher": ratio}
    res.append(("sigma_zeta (Fisher) matches the empirical spread within 5 %", abs(ratio - 1) <= 0.05))
    return res, num
