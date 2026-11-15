"""M4's multi-baseline log-polar matcher (M4a #10, Stage 3), in PyTorch, device-agnostic.

Observation only: luminance rasters of a reference frame r and of the window's frames k, their signed
steps t_k along the heading, and the sensor. Under the aligned dolly an anchor of r at axial inverse depth
zeta is seen by frame k at the same azimuth and at d_k(zeta) = q_k - q_r = -ln(1 - t_k zeta) (M4's
geometry, motion/multiframe.py). Every frame's matching cost is put on the common zeta axis and averaged
(Okutomi & Kanade's multiple-baseline stereo): the true minimum is shared, the false minima of repeated
texture fall at different zeta for different t_k.

1. **Coarse, on the log-polar grid.** For a given zeta, d_k does not depend on the eccentricity, so frame
   k's whole grid is shifted by d_k(zeta) at once (a fractional row shift). Zero-mean NCC over windows of
   +/- r_half_px radially and +/- a_half_px in azimuth (native px, per grid row, as M3), averaged over the
   frames whose window is valid. zeta = 0 (infinity) is included.
2. **Fine, on the native raster.** One joint Gauss-Newton fit per anchor: zeta, its radial slope
   zeta_q = d zeta / d q along the patch (a surface not facing the eye: it predicts every frame's
   stretch consistently), and a gain and offset per frame:
       I_r(q + u, phi + v) = alpha_k I_k(q + u + d_k(zeta + zeta_q u), phi + v) + beta_k.
   sigma_zeta from the joint covariance with the residual variance. Then **frame consistency**: frames
   whose own NCC under the fit is below frame_ncc_min are dropped and the fit is repeated once.
3. **Rejections**, each recorded: invalid support (anchor or too few valid frames), low radial texture
   (M3's A2: no number of frames helps an edge along a radial line), low mean NCC, ambiguous (a second
   peak of the averaged curve outside +/- excl_px), beyond the far envelope (unresolved, incl. zeta 0),
   at the near boundary, Gauss-Newton failed or moved too far or degenerate, too few consistent frames,
   unsupported by the neighbours, depth edge. M3's forward-backward and half-window tests are not
   carried over: the frames' mutual consistency takes their place (stated in #10).

The sensor mapping is a PyTorch transcription of the Foveated `pixel_of`; selftest checks it against
fov-render's to 1e-9 px. Settings in SETTINGS are fixed before measurement; Stage 5 calibrates only the
interval scale.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import logpolar as lp

SETTINGS = {
    "coarse_dq": 0.01, "n_phi": 800, "e_grid_lo_deg": 0.1, "e_anchor_deg": (1.0, 44.0),
    "r_half_px": 3, "a_half_px": 1, "z_envelope_m": (0.25, 20.0),
    "zeta_step_px": 0.5,        # coarse zeta step: 0.5 native px of the longest step's displacement
    "excl_px": 1.5, "ncc_min": 0.8, "ambig_margin": 0.05, "texture_min": 0.02,
    "gn_iters": 8, "gn_max_move_px": 1.5, "zeta_q_max": 2.0,
    "frame_ncc_min": 0.5, "min_frames": 3,
    "support_min": 3, "support_tol_px": 0.5, "edge_half_px": 3, "edge_tol_px": 0.5,
    "interval_scale": 1.0, "chunk": 6000,
}
Z95 = 1.959963984540054
FLAT_STD = 1e-9
REASONS = ("accepted", "invalid_support", "low_texture", "low_ncc", "ambiguous", "unresolved_far", "near_boundary",
           "gn_failed", "few_frames", "unsupported", "depth_edge")


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- sensor mapping (Foveated)

class FovMap:
    """(q, phi) -> continuous native (row, col) and inside, for fov-render's Foveated model (torch)."""

    def __init__(self, model, dev):
        self.n, self.e2, self.emax = int(model.n), float(model.e2), float(model.emax)
        self.k = math.log(1.0 + self.emax / self.e2)
        self.dev = dev

    def rc(self, q, phi):
        e_deg = torch.rad2deg(torch.atan(torch.exp(q)))
        r = torch.log1p(e_deg / self.e2) / self.k
        px, py = 0.5 * r * torch.cos(phi), 0.5 * r * torch.sin(phi)
        j = (px + 0.5) * self.n - 0.5
        i = (0.5 - py) * self.n - 0.5
        return i, j, r <= 1.0


def bilinear(img, i, j):
    """img (h, w) with nan off the disk; nan where any of the four neighbours is nan or outside."""
    h, w = img.shape
    i0, j0 = torch.floor(i), torch.floor(j)
    fi, fj = i - i0, j - j0
    i0, j0 = i0.long(), j0.long()
    out = (i0 < 0) | (j0 < 0) | (i0 + 1 >= h) | (j0 + 1 >= w)
    i0c, j0c = i0.clamp(0, h - 2), j0.clamp(0, w - 2)
    flat = img.reshape(-1)
    g = lambda a, b: flat[(a * w + b).reshape(-1)].reshape(a.shape)
    v = (g(i0c, j0c) * (1 - fi) * (1 - fj) + g(i0c + 1, j0c) * fi * (1 - fj)
         + g(i0c, j0c + 1) * (1 - fi) * fj + g(i0c + 1, j0c + 1) * fi * fj)
    return torch.where(out, torch.full_like(v, float("nan")), v)


def sample(img, fm: FovMap, q, phi):
    i, j, inside = fm.rc(q, phi)
    v = bilinear(img, i, j)
    return torch.where(inside, v, torch.full_like(v, float("nan")))


# ----------------------------------------------------------------------------- coarse

def _box(X, ri, h, w, wmax):
    """Window sums over rows ri +/- h (per row) and periodic columns +/- w (per row); X (E, P)."""
    C = torch.cat([torch.zeros_like(X[:1]), torch.cumsum(X, 0)])
    S = C[ri + h + 1] - C[ri - h]
    P = S.shape[1]
    if wmax:
        S = torch.cat([S[:, -wmax:], S, S[:, :wmax]], 1)
    C2 = torch.cat([torch.zeros_like(S[:, :1]), torch.cumsum(S, 1)], 1)
    j = torch.arange(P, device=X.device)[None, :] + wmax
    return torch.gather(C2, 1, j + w[:, None] + 1) - torch.gather(C2, 1, j - w[:, None])


class Grid:
    def __init__(self, model, fm: FovMap, s, dev, dtype=torch.float64):
        dq = s["coarse_dq"]
        n = int(np.ceil((0.0 - lp.q_of_e(np.radians(s["e_grid_lo_deg"]))) / dq)) + 1
        self.q = torch.tensor(0.0 - dq * np.arange(n)[::-1], dtype=dtype, device=dev)
        self.phi = torch.tensor(-np.pi + (np.arange(s["n_phi"]) + 0.5) * (2 * np.pi / s["n_phi"]), dtype=dtype, device=dev)
        Q, PH = torch.meshgrid(self.q, self.phi, indexing="ij")
        self.i, self.j, self.inside = fm.rc(Q, PH)
        qn = self.q.cpu().numpy()
        self.gq = torch.tensor(lp.native_px_per_q(model, qn, np.zeros_like(qn)), dtype=dtype, device=dev)
        hh = 1e-5
        gp = np.linalg.norm(model.pixel_of(lp.dir_of_qphi(qn, hh + 0 * qn))[0]
                            - model.pixel_of(lp.dir_of_qphi(qn, -hh + 0 * qn))[0], axis=-1) / (2 * hh)
        self.gp = torch.tensor(gp, dtype=dtype, device=dev)
        self.dq, self.dphi = dq, 2 * np.pi / s["n_phi"]

    def unwrap(self, img):
        v = bilinear(img, self.i, self.j)
        return torch.where(self.inside, v, torch.full_like(v, float("nan")))


def coarse_curves(Jr, Js, ts, zetas, grid: Grid, rows, s):
    """Mean NCC over frames (len(rows), n_phi, n_zeta) and the number of frames with a valid window."""
    dev = Jr.device
    hq = torch.clamp(torch.round(s["r_half_px"] / (grid.gq[rows] * grid.dq)), min=1).long()
    hp = torch.clamp(torch.round(s["a_half_px"] / (grid.gp[rows] * grid.dphi)), min=0).long()
    hmax, wmax = int(hq.max()), int(hp.max())
    R, P = Jr.shape
    lo, hi = int(rows.min()) - hmax, int(rows.max()) + hmax
    ext = torch.arange(lo, hi + 1, device=dev)
    ri = rows - lo
    nfull = ((2 * hq + 1) * (2 * hp + 1)).to(Jr.dtype)[:, None]
    def take_rows(J, rr):                                   # rows of J at integer rr, nan outside
        ok = (rr >= 0) & (rr < R)
        out = J[rr.clamp(0, R - 1)]
        return torch.where(ok[:, None], out, torch.full_like(out, float("nan")))
    A = take_rows(Jr, ext)
    Ma = torch.isfinite(A)
    Ae = torch.where(Ma, A, torch.zeros_like(A))
    Sa, Saa = _box(Ae, ri, hq, hp, wmax), _box(Ae * Ae, ri, hq, hp, wmax)
    na = _box(Ma.to(Jr.dtype), ri, hq, hp, wmax)
    va = Saa - Sa * Sa / nfull
    a_ok = (na == nfull) & (va > 1e-12)
    total = torch.zeros((len(rows), P, len(zetas)), dtype=torch.float32, device=dev)
    count = torch.zeros((len(rows), P, len(zetas)), dtype=torch.int16, device=dev)
    for Jk, t in zip(Js, ts):
        for z_i, zeta in enumerate(zetas):
            tz = t * zeta
            if tz >= 0.98:
                continue
            d = -math.log1p(-tz)
            sh = d / grid.dq                                  # fractional row shift: target row = row + sh
            f0 = math.floor(sh)
            fr = sh - f0
            T0, T1 = take_rows(Jk, ext + f0), take_rows(Jk, ext + f0 + 1)
            T = T0 * (1 - fr) + T1 * fr
            Mt = torch.isfinite(T)
            Te = torch.where(Mt, T, torch.zeros_like(T))
            n = _box((Ma & Mt).to(Jr.dtype), ri, hq, hp, wmax)
            St, Stt, Sat = _box(Te, ri, hq, hp, wmax), _box(Te * Te, ri, hq, hp, wmax), _box(Ae * Te, ri, hq, hp, wmax)
            vt = Stt - St * St / nfull
            c = (Sat - Sa * St / nfull) / torch.sqrt(torch.clamp(va * vt, min=1e-300))
            ok = a_ok & (n == nfull) & (vt > 1e-12)
            total[:, :, z_i] += torch.where(ok, c, torch.zeros_like(c)).float()
            count[:, :, z_i] += ok.to(torch.int16)
    return total, count


# ----------------------------------------------------------------------------- fine

def _patch(s, gq, gp, dtype):
    k = torch.arange(-s["r_half_px"], s["r_half_px"] + 1, dtype=dtype, device=gq.device)
    j = torch.arange(-s["a_half_px"], s["a_half_px"] + 1, dtype=dtype, device=gq.device)
    K, J = torch.meshgrid(k, j, indexing="ij")
    return K.reshape(1, -1) / gq[:, None], J.reshape(1, -1) / gp[:, None]


def _dk(t, zeta):
    return -torch.log1p(-t * zeta)


def _ddk(t, zeta):
    return t / (1 - t * zeta)


def joint_fit(Ir, Is, ts, fm, q, phi, gq, gp, zeta0, mask0, s):
    """Joint Gauss-Newton per anchor. Ir (h, w), Is list of K rasters, ts (K,), anchors (N,). mask0 (N, K)
    frames allowed. Returns zeta, zeta_q, sigma_zeta, per-frame NCC (N, K), frames used mask, ok."""
    dt = q.dtype
    N, K = len(q), len(Is)
    u, v = _patch(s, gq, gp, dt)                                     # (N, M)
    Q, PH = q[:, None] + u, phi[:, None] + v
    A = sample(Ir, fm, Q, PH)                                         # (N, M)
    a_ok = torch.isfinite(A).all(1)
    T = torch.tensor(ts, dtype=dt, device=q.device)
    zeta, zq = zeta0.clone(), torch.zeros_like(zeta0)
    h = 0.25 / gq[:, None]
    def warp(zeta, zq):
        Z = zeta[:, None, None] + zq[:, None, None] * u[:, None, :]  # (N, 1, M) -> per frame
        tz = T[None, :, None] * Z
        valid_t = tz < 0.98
        D = -torch.log1p(-torch.clamp(tz, max=0.98))
        Qk = Q[:, None, :] + D                                       # (N, K, M)
        Ik = torch.stack([sample(Is[k], fm, Qk[:, k], PH) for k in range(K)], 1)
        dIk = torch.stack([(sample(Is[k], fm, Qk[:, k] + h, PH) - sample(Is[k], fm, Qk[:, k] - h, PH)) / (2 * h)
                           for k in range(K)], 1)
        dD = T[None, :, None] / (1 - torch.clamp(tz, max=0.98))       # d d_k / d zeta at each sample
        return Ik, dIk, dD, valid_t.all(2)
    Ik, _, _, vt = warp(zeta, zq)
    fr_ok = mask0 & torch.isfinite(Ik).all(2) & vt
    sd = torch.std(Ik, 2)
    flat = torch.isfinite(sd) & (sd <= FLAT_STD)          # a truly flat patch (nan = off the disk, not flat)
    fr_ok &= ~flat
    alpha = torch.where(fr_ok, torch.std(A, 1)[:, None] / torch.clamp(sd, min=1e-300), torch.ones_like(sd))
    beta = torch.where(fr_ok, A.mean(1)[:, None] - alpha * Ik.mean(2), torch.zeros_like(sd))
    npar = 2 + 2 * K
    eye = torch.eye(npar, dtype=dt, device=q.device)
    def solve_pass(zeta, zq, alpha, beta, use, iters):
        for _ in range(iters + 1):
            Ik, dIk, dD, vt = warp(zeta, zq)
            good_f = use & torch.isfinite(Ik).all(2) & torch.isfinite(dIk).all(2) & vt
            r = A[:, None, :] - alpha[:, :, None] * Ik - beta[:, :, None]          # (N, K, M)
            gz = alpha[:, :, None] * dIk * dD                                      # d r / d zeta = -(...) ; sign below
            Jz = -gz
            Jzq = -gz * u[:, None, :]
            Ja = -Ik
            Jb = -torch.ones_like(Ik)
            w = good_f[:, :, None].to(dt)
            r = torch.nan_to_num(r) * w
            Jz, Jzq, Ja, Jb = (torch.nan_to_num(x) * w for x in (Jz, Jzq, Ja, Jb))
            H = torch.zeros((N, npar, npar), dtype=dt, device=q.device)
            g = torch.zeros((N, npar), dtype=dt, device=q.device)
            H[:, 0, 0] = (Jz * Jz).sum((1, 2)); H[:, 0, 1] = H[:, 1, 0] = (Jz * Jzq).sum((1, 2))
            H[:, 1, 1] = (Jzq * Jzq).sum((1, 2))
            g[:, 0], g[:, 1] = (Jz * r).sum((1, 2)), (Jzq * r).sum((1, 2))
            ia = torch.arange(K, device=q.device) + 2
            ib = ia + K
            H[:, 0, ia] = H[:, ia, 0] = (Jz * Ja).sum(2); H[:, 0, ib] = H[:, ib, 0] = (Jz * Jb).sum(2)
            H[:, 1, ia] = H[:, ia, 1] = (Jzq * Ja).sum(2); H[:, 1, ib] = H[:, ib, 1] = (Jzq * Jb).sum(2)
            H[:, ia, ia] = (Ja * Ja).sum(2); H[:, ib, ib] = (Jb * Jb).sum(2)
            H[:, ia, ib] = H[:, ib, ia] = (Ja * Jb).sum(2)
            g[:, ia], g[:, ib] = (Ja * r).sum(2), (Jb * r).sum(2)
            H = H + 1e-12 * eye
            if _ == iters:
                return zeta, zq, alpha, beta, good_f, H, (r * r).sum((1, 2)), Ik
            x = torch.linalg.solve(H, g[..., None])[..., 0]
            okn = torch.isfinite(x).all(1)
            x = torch.where(okn[:, None], x, torch.zeros_like(x))
            zeta = zeta - x[:, 0]
            zq = torch.clamp(zq - x[:, 1], -s["zeta_q_max"], s["zeta_q_max"])
            alpha, beta = alpha - x[:, ia], beta - x[:, ib]
            zeta = torch.clamp(zeta, min=0.0)
    zeta, zq, alpha, beta, good_f, H, ssr, Ik = solve_pass(zeta, zq, alpha, beta, fr_ok, s["gn_iters"])
    # frame consistency: each frame's own NCC under the fit
    za = A - A.mean(1, keepdim=True)
    zk = Ik - Ik.mean(2, keepdim=True)
    nccf = (zk * za[:, None, :]).sum(2) / torch.sqrt(torch.clamp((zk * zk).sum(2) * (za * za).sum(1)[:, None], min=1e-300))
    nccf = torch.where(good_f, nccf, torch.full_like(nccf, float("nan")))
    keep = good_f & (nccf >= s["frame_ncc_min"])
    redo = (keep != good_f).any(1)
    if bool(redo.any()):
        zeta2, zq2, a2, b2, gf2, H2, ssr2, Ik2 = solve_pass(zeta, zq, alpha, beta, keep, max(2, s["gn_iters"] // 2))
        sel = redo
        zeta, zq = torch.where(sel, zeta2, zeta), torch.where(sel, zq2, zq)
        good_f = torch.where(sel[:, None], gf2, good_f)
        H = torch.where(sel[:, None, None], H2, H)
        ssr = torch.where(sel, ssr2, ssr)
        Ik = torch.where(sel[:, None, None], Ik2, Ik)
        zk = Ik - Ik.mean(2, keepdim=True)
        nccf = (zk * za[:, None, :]).sum(2) / torch.sqrt(torch.clamp((zk * zk).sum(2) * (za * za).sum(1)[:, None], min=1e-300))
        nccf = torch.where(good_f, nccf, torch.full_like(nccf, float("nan")))
    nf = good_f.sum(1)
    M = A.shape[1]
    dof = torch.clamp(nf * M - (2 + 2 * nf), min=1).to(dt)
    s2 = ssr / dof
    try:
        cov00 = torch.linalg.inv(H)[:, 0, 0]
        inv_ok = torch.ones(N, dtype=torch.bool, device=q.device)
    except RuntimeError:
        cov00 = torch.full((N,), float("nan"), dtype=dt, device=q.device)
        inv_ok = torch.zeros(N, dtype=torch.bool, device=q.device)
        for n_ in range(N):
            try:
                cov00[n_] = torch.linalg.inv(H[n_])[0, 0]
                inv_ok[n_] = True
            except RuntimeError:
                pass
    sigma = torch.sqrt(torch.clamp(s2 * cov00, min=0.0))
    # radial texture of r's patch (A2)
    gr = sample(Ir, fm, Q + 0.5 / gq[:, None], PH) - sample(Ir, fm, Q - 0.5 / gq[:, None], PH)
    tex = torch.sqrt((gr ** 2).mean(1)) / torch.abs(A.mean(1))
    ok = a_ok & inv_ok & torch.isfinite(sigma) & torch.isfinite(zeta)
    return {"zeta": zeta, "zeta_q": zq, "sigma_zeta": sigma, "ncc_frames": nccf, "frames": good_f, "n_frames": nf,
            "ok": ok, "texture": tex, "degenerate": flat.any(1) & ~fr_ok.any(1)}


# ----------------------------------------------------------------------------- the reference frame

def match_reference(img_r, imgs, ts, model, settings=None, dev=None):
    """img_r and imgs: luminance rasters (numpy, nan off the disk) of the reference frame and the window's
    frames; ts their signed steps (m) along the heading (t_k = (c_k - c_r) . h). Returns per anchor (valid
    native pixels of r at e_anchor_deg) numpy arrays: flat index, e, phi, q, zeta, sigma_zeta, zeta_q,
    reason, diagnostics."""
    s = dict(SETTINGS, **(settings or {}))
    dev = dev or device()
    dt = torch.float64
    fm = FovMap(model, dev)
    grid = Grid(model, fm, s, dev, dt)
    to_t = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=dt, device=dev)
    Ir, Is = to_t(img_r), [to_t(x) for x in imgs]
    Jr, Js = grid.unwrap(Ir), [grid.unwrap(x) for x in Is]
    ts = [float(t) for t in ts]
    tmax = max(abs(t) for t in ts)
    zmin, zmax = s["z_envelope_m"]
    gmax = float(grid.gq.max())
    dz = s["zeta_step_px"] / (gmax * tmax)
    zetas = list(np.arange(0.0, 1.0 / zmin + dz, dz))
    e_lo, e_hi = np.radians(s["e_anchor_deg"][0]), np.radians(s["e_anchor_deg"][1])
    qn = grid.q.cpu().numpy()
    rows = torch.tensor(np.flatnonzero((qn >= lp.q_of_e(e_lo) - 2 * grid.dq) & (qn <= lp.q_of_e(e_hi) + 2 * grid.dq)),
                        device=dev)
    total, count = coarse_curves(Jr, Js, ts, zetas, grid, rows, s)
    with np.errstate(invalid="ignore"):
        C = torch.where(count >= s["min_frames"], total / torch.clamp(count.float(), min=1), torch.full_like(total, float("nan")))
    # anchors
    rays = model.rays()
    e_px, phi_px = lp.e_phi(rays.direction)
    sel = rays.valid & (e_px >= e_lo) & (e_px <= e_hi)
    flat = np.flatnonzero(sel.reshape(-1))
    eb, pb = e_px.reshape(-1)[flat], phi_px.reshape(-1)[flat]
    qb = lp.q_of_e(eb)
    ir = np.clip(np.rint((qb - qn[rows[0].item()]) / grid.dq).astype(int), 0, len(rows) - 1)
    ic = np.mod(np.rint((pb + np.pi) / grid.dphi - 0.5).astype(int), s["n_phi"])
    curves = C[torch.tensor(ir, device=dev), torch.tensor(ic, device=dev)]          # (N, Z)
    cnt_a = count[torch.tensor(ir, device=dev), torch.tensor(ic, device=dev)]
    cz = torch.nan_to_num(curves, nan=-2.0)
    j1 = torch.argmax(cz, 1)
    c1 = cz.gather(1, j1[:, None])[:, 0]
    gq_b = lp.native_px_per_q(model, qb, pb)
    gp_b = np.linalg.norm(model.pixel_of(lp.dir_of_e_phi(eb, pb + 1e-5))[0] - model.pixel_of(lp.dir_of_e_phi(eb, pb - 1e-5))[0],
                          axis=-1) / 2e-5
    geq = gq_b * tmax                                                                 # native px per unit zeta, longest step
    excl = torch.tensor(np.ceil(s["excl_px"] / (geq * dz)), device=dev)
    Z = len(zetas)
    jj = torch.arange(Z, device=dev)[None, :]
    c2 = torch.where((jj - j1[:, None]).abs() > excl[:, None], cz, torch.full_like(cz, -2.0)).max(1).values
    cm = cz.gather(1, (j1 - 1).clamp(0, Z - 1)[:, None])[:, 0]
    cp = cz.gather(1, (j1 + 1).clamp(0, Z - 1)[:, None])[:, 0]
    den = cm - 2 * c1 + cp
    delta = torch.where((j1 > 0) & (j1 < Z - 1) & (den < 0), 0.5 * (cm - cp) / torch.where(den < 0, den, -torch.ones_like(den)),
                        torch.zeros_like(den)).clamp(-0.5, 0.5)
    zeta0 = ((j1.double() + delta.double()) * dz).clamp(min=0.0)
    # fine, in chunks
    N = len(flat)
    out = {k: [] for k in ("zeta", "zeta_q", "sigma_zeta", "n_frames", "ok", "texture", "degenerate", "ncc_frames")}
    q_t, p_t, gq_t, gp_t = to_t(qb), to_t(pb), to_t(gq_b), to_t(gp_b)
    for a in range(0, N, s["chunk"]):
        b = min(N, a + s["chunk"])
        m0 = torch.ones((b - a, len(Is)), dtype=torch.bool, device=dev)
        f = joint_fit(Ir, Is, ts, fm, q_t[a:b], p_t[a:b], gq_t[a:b], gp_t[a:b], zeta0[a:b], m0, s)
        for k in out:
            out[k].append(f[k])
    fit = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    zeta, sig = fit["zeta"], fit["sigma_zeta"]
    z0 = zeta0.cpu().numpy()
    c1n, c2n, j1n = c1.cpu().numpy(), c2.cpu().numpy(), j1.cpu().numpy()
    zmin_zeta = 1.0 / zmax
    reason = np.zeros(N, dtype=np.int8)
    def mark(cond, name):
        reason[(reason == 0) & cond] = REASONS.index(name)
    mark(fit["degenerate"], "gn_failed")
    mark(~fit["ok"] | (c1n <= -2.0), "invalid_support")
    mark(~(fit["texture"] >= s["texture_min"]), "low_texture")
    mark(~(c1n >= s["ncc_min"]), "low_ncc")
    mark(c2n >= c1n - s["ambig_margin"], "ambiguous")
    mark((zeta < zmin_zeta) | (z0 < zmin_zeta), "unresolved_far")
    mark(j1n >= Z - 1, "near_boundary")
    moved = np.abs(zeta - z0) * geq
    mark(~(moved <= s["gn_max_move_px"]) | ~(np.abs(fit["zeta_q"]) < s["zeta_q_max"]), "gn_failed")
    mark(fit["n_frames"] < s["min_frames"], "few_frames")
    # neighbour support and depth edge (as M3), in native px of the longest step
    n = model.n
    def ras(v, cond):
        R_ = np.full(n * n, np.nan)
        R_[flat[cond]] = v[cond]
        return R_.reshape(n, n)
    def shifted(R_, di, dj):
        out_ = np.full_like(R_, np.nan)
        out_[max(-di, 0):n + min(-di, 0), max(-dj, 0):n + min(-dj, 0)] = R_[max(di, 0):n + min(di, 0), max(dj, 0):n + min(dj, 0)]
        return out_
    cand = reason == 0
    Dz = ras(zeta, cand)
    G = ras(geq, np.ones(N, bool))
    sup = np.zeros((n, n), np.int16)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                with np.errstate(invalid="ignore"):
                    sup += (np.abs(shifted(Dz, di, dj) - Dz) * G <= s["support_tol_px"]).astype(np.int16)
    support = sup.reshape(-1)[flat]
    mark(cand & (support < s["support_min"]), "unsupported")
    cand2 = reason == 0
    Dz2, Qr, Kq = ras(zeta, cand2), ras(qb, np.ones(N, bool)), ras(fit["zeta_q"], np.ones(N, bool))
    dev_ = np.zeros((n, n))
    hh = int(s["edge_half_px"])
    for di in range(-hh, hh + 1):
        for dj in range(-hh, hh + 1):
            if di or dj:
                with np.errstate(invalid="ignore"):
                    dv = np.abs(shifted(Dz2, di, dj) - (Dz2 + Kq * (shifted(Qr, di, dj) - Qr))) * G
                dev_ = np.fmax(dev_, np.where(np.isfinite(dv), dv, 0.0))
    edge_dev = dev_.reshape(-1)[flat]
    mark(cand2 & (edge_dev > s["edge_tol_px"]), "depth_edge")
    half = Z95 * s["interval_scale"] * sig
    return {"flat": flat, "e": eb, "phi": pb, "q": qb, "zeta": zeta, "zeta_lo": zeta - half, "zeta_hi": zeta + half,
            "sigma_zeta": sig, "sigma_px": sig * geq, "zeta_q": fit["zeta_q"], "n_frames": fit["n_frames"],
            "ncc_coarse": c1n, "ncc_second": c2n, "zeta_coarse": z0, "texture": fit["texture"], "support": support,
            "edge_dev_px": edge_dev, "px_per_zeta": geq, "reason": reason, "ts": np.asarray(ts),
            "frames_valid_coarse": cnt_a.max(1).values.cpu().numpy(), "zeta_step": dz, "settings": s,
            "ncc_frames": fit["ncc_frames"]}


# ----------------------------------------------------------------------------- Stage 3 selftest

def selftest(dev=None):
    """torch against numpy: the sensor mapping, sampling and window sums (each with a failing control)."""
    from fov_render.sensors import FoveatedModel
    from . import matcher as m3
    dev = dev or device()
    res, num = [], {}
    rng = np.random.default_rng(0)
    for N in (126, 253):
        model = FoveatedModel(N)
        fm = FovMap(model, dev)
        q = rng.uniform(lp.q_of_e(np.radians(0.2)), 0.0, 20000)
        ph = rng.uniform(-np.pi, np.pi, 20000)
        rc, inside = model.pixel_of(lp.dir_of_qphi(q, ph))
        i, j, ins = fm.rc(torch.tensor(q, dtype=torch.float64, device=dev), torch.tensor(ph, dtype=torch.float64, device=dev))
        err = float(np.max(np.abs(np.stack([i.cpu().numpy(), j.cpu().numpy()], -1) - rc)))
        num[f"pixel_of_{N}"] = err
        res.append((f"torch Foveated mapping equals fov-render's pixel_of (raster {N}) within 1e-9 px",
                    err <= 1e-9 and bool((ins.cpu().numpy() == inside).all())))
        if N == 253:
            fm.e2 = 2.2
            i2, j2, _ = fm.rc(torch.tensor(q, dtype=torch.float64, device=dev), torch.tensor(ph, dtype=torch.float64, device=dev))
            err2 = float(np.max(np.abs(np.stack([i2.cpu().numpy(), j2.cpu().numpy()], -1) - rc)))
            res.append(("control: the mapping with E2 = 2.2 deg misses by >= 1 px", err2 >= 1.0))
    img = rng.random((253, 253))
    img[rng.random((253, 253)) < 0.05] = np.nan
    rcs = rng.uniform(-2, 255, (30000, 2))
    a = lp.bilinear(img, rcs)
    b = bilinear(torch.tensor(img, device=dev), torch.tensor(rcs[:, 0], device=dev), torch.tensor(rcs[:, 1], device=dev)).cpu().numpy()
    same_nan = bool((np.isnan(a) == np.isnan(b)).all())
    err = float(np.nanmax(np.abs(a - b)))
    num["bilinear"] = err
    res.append(("torch bilinear equals logpolar.bilinear within 1e-12, same nan pattern", same_nan and err <= 1e-12))
    X = rng.random((120, 64))
    ri = np.arange(10, 110)
    h = rng.integers(1, 6, ri.size)
    w = rng.integers(0, 4, ri.size)
    A = m3._box(X, ri, h, w, int(w.max()))
    B = _box(torch.tensor(X, device=dev), torch.tensor(ri, device=dev), torch.tensor(h, device=dev),
             torch.tensor(w, device=dev), int(w.max())).cpu().numpy()
    err = float(np.max(np.abs(A - B)))
    num["box"] = err
    res.append(("torch window sums equal M3's numpy _box within 1e-9", err <= 1e-9))
    Bc = _box(torch.tensor(X, device=dev), torch.tensor(ri, device=dev), torch.tensor(h + 1, device=dev),
              torch.tensor(w, device=dev), int(w.max())).cpu().numpy()
    res.append(("control: windows one row wider differ by >= 0.1", float(np.max(np.abs(A - Bc))) >= 0.1))
    return res, num
