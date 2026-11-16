"""M6's multi-baseline matcher (M6a #22, T2): M4's matcher with the known poses, any rotation, any epipole.

Observation and poses only: luminance rasters of a reference frame r and of the window's frames k, and each
frame's pose relative to r: A_k = R_k^T R_r (r's eye frame -> k's) and T_k = R_r^T (c_k - c_r) (k's centre in r's
eye frame). The unknown is M4's: zeta = 1 / z_r, the inverse depth along r's axis. An anchor at reference
log-polar (q, phi) has the unit-depth ray m = (tan e cos phi, tan e sin phi, -1), tan e = exp(q); at zeta its
point is m / zeta, which frame k sees along

    v_k(zeta) = A_k (m - zeta T_k)            (derotation folded in: no image is resampled for it)

at q_k = ln(hypot(v_x, v_y) / -v_z), phi_k = atan2(v_y, v_x), valid while -v_z > 0.02 (M4's t zeta < 0.98).
With A_k = I and T_k = t_k (0, 0, -1) this is M4's q_k = q - ln(1 - t_k zeta), phi_k = phi (motion/derotate.py,
T0), and every step below reduces to M4's:

1. **Coarse, on r's log-polar grid.** For each zeta, every grid node of r is mapped into frame k's log-polar
   grid (the same grid on k's raster) and k's unwrapped image is sampled there, bilinear in (row, column);
   M4's row shift is the case where the column is the node's own. Zero-mean NCC over M4's windows, averaged
   over the frames whose window is valid. Columns within 1e-9 of a node are taken exactly.
2. **Fine, on the native raster.** M4's joint Gauss-Newton (zeta, its radial slope zeta_q along r's q, a gain
   and offset per frame), with frame k sampled at (q_k, phi_k) of the patch and d I / d zeta =
   I_q dq_k/dzeta + I_phi dphi_k/dzeta (analytic derivatives of the map; the second term is 0 where dphi_k/dzeta
   is exactly 0, as on M4's dolly). Frame consistency as M4.
3. **Rejections** as M4. Two are generalised:
   - *texture*: M4's radial texture along each frame's epipolar direction in r (the direction the point moves
     in r's coordinates after derotation), the largest over the window's frames (an edge carries no motion
     only if it lies along every frame's epipolar line);
   - *px per zeta* (geq, for the ambiguity exclusion, Gauss-Newton's move, support and depth edges): the
     largest over frames of the native px a frame's sample moves per unit zeta at zeta = 0.

SETTINGS are M4's frozen ones, unchanged. Imports M4's sensor map, sampling and window sums; M4's files are
not modified.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import logpolar as lp
from . import multibaseline as mb

SETTINGS = dict(mb.SETTINGS)
REASONS = mb.REASONS
Z95 = mb.Z95
DEPTH_MIN = 0.02            # -v_z > DEPTH_MIN: M4's t zeta < 0.98
COL_SNAP = 1e-9


def relative_poses(c_r, R_r, frames):
    """[(A_k, T_k)] for frames [(c_k, R_k)] relative to r (world = R @ local)."""
    return [(R_k.T @ R_r, R_r.T @ (np.asarray(c_k, float) - np.asarray(c_r, float))) for c_k, R_k in frames]


def aligned_poses(ts):
    """M4's dolly: no rotation, steps t_k along r's axis (-Z)."""
    return [(np.eye(3), np.array([0.0, 0.0, -float(t)])) for t in ts]


def _m(q, phi):
    te = torch.exp(q)
    return te * torch.cos(phi), te * torch.sin(phi), -torch.ones_like(q)


class FrameMap:
    """r's (q, phi) and zeta -> frame k's (q_k, phi_k), validity and derivatives in zeta (torch)."""

    def __init__(self, A, T, dev, dtype=torch.float64):
        self.A = torch.tensor(np.asarray(A, float), dtype=dtype, device=dev)
        self.T = torch.tensor(np.asarray(T, float), dtype=dtype, device=dev)
        self.b = self.A @ self.T                                       # d v_k / d zeta = -b

    def v(self, mx, my, mz, zeta):
        A, T = self.A, self.T
        ux, uy, uz = mx - zeta * T[0], my - zeta * T[1], mz - zeta * T[2]
        return (A[0, 0] * ux + A[0, 1] * uy + A[0, 2] * uz, A[1, 0] * ux + A[1, 1] * uy + A[1, 2] * uz,
                A[2, 0] * ux + A[2, 1] * uy + A[2, 2] * uz)

    def qphi(self, mx, my, mz, zeta):
        vx, vy, vz = self.v(mx, my, mz, zeta)
        ok = -vz > DEPTH_MIN
        vz = torch.where(ok, vz, torch.full_like(vz, -DEPTH_MIN))
        rho2 = vx * vx + vy * vy
        q = 0.5 * torch.log(rho2) - torch.log(-vz)
        phi = torch.atan2(vy, vx)
        bx, by, bz = -self.b[0], -self.b[1], -self.b[2]                # d v / d zeta
        dq = (vx * bx + vy * by) / rho2 - bz / vz
        dphi = (vx * by - vy * bx) / rho2
        return q, phi, ok, dq, dphi


def _grid_rc(grid, q, phi):
    """Fractional (row, col) on the log-polar grid (cols periodic)."""
    row = (q - grid.q[0]) / grid.dq
    col = (phi + math.pi) / grid.dphi - 0.5
    return row, col


def _sample_grid(J, row, col, n_phi):
    """Bilinear on the unwrapped grid J (R, P): rows nan outside, columns periodic; columns within COL_SNAP of a node
    are taken exactly (M4's row shift keeps the node's own column)."""
    R, P = J.shape
    r0 = torch.floor(row)
    fr = row - r0
    r0 = r0.long()
    cr = torch.round(col)
    snap = (col - cr).abs() <= COL_SNAP
    c0 = torch.where(snap, cr, torch.floor(col))
    fc = torch.where(snap, torch.zeros_like(col), col - c0)
    c0 = torch.remainder(c0.long(), P)
    c1 = torch.remainder(c0 + 1, P)
    flat = J.reshape(-1)
    nan = torch.full_like(row, float("nan"))

    def at(rr, cc):
        ok = (rr >= 0) & (rr < R)
        v = flat[(rr.clamp(0, R - 1) * P + cc).reshape(-1)].reshape(rr.shape)
        return torch.where(ok, v, nan)
    t0 = at(r0, c0) * (1 - fr) + at(r0 + 1, c0) * fr
    t1 = at(r0, c1) * (1 - fr) + at(r0 + 1, c1) * fr
    return torch.where(snap, t0, t0 * (1 - fc) + t1 * fc)


def coarse_curves(Jr, Js, maps, zetas, grid, rows, s):
    """M4's coarse_curves with frame k sampled through its FrameMap instead of a row shift."""
    dev = Jr.device
    hq = torch.clamp(torch.round(s["r_half_px"] / (grid.gq[rows] * grid.dq)), min=1).long()
    hp = torch.clamp(torch.round(s["a_half_px"] / (grid.gp[rows] * grid.dphi)), min=0).long()
    hmax, wmax = int(hq.max()), int(hp.max())
    R, P = Jr.shape
    lo, hi = int(rows.min()) - hmax, int(rows.max()) + hmax
    ext = torch.arange(lo, hi + 1, device=dev)
    ri = rows - lo
    nfull = ((2 * hq + 1) * (2 * hp + 1)).to(Jr.dtype)[:, None]
    def take_rows(J, rr):
        ok = (rr >= 0) & (rr < R)
        out = J[rr.clamp(0, R - 1)]
        return torch.where(ok[:, None], out, torch.full_like(out, float("nan")))
    A = take_rows(Jr, ext)
    Ma = torch.isfinite(A)
    Ae = torch.where(Ma, A, torch.zeros_like(A))
    Sa, Saa = mb._box(Ae, ri, hq, hp, wmax), mb._box(Ae * Ae, ri, hq, hp, wmax)
    na = mb._box(Ma.to(Jr.dtype), ri, hq, hp, wmax)
    va = Saa - Sa * Sa / nfull
    a_ok = (na == nfull) & (va > 1e-12)
    qn = grid.q[0] + ext.to(Jr.dtype) * grid.dq                         # node q for every ext row (also outside)
    Qn, PHn = torch.meshgrid(qn, grid.phi, indexing="ij")
    mx, my, mz = _m(Qn, PHn)
    row_n = ext.to(Jr.dtype)[:, None].expand_as(Qn)
    total = torch.zeros((len(rows), P, len(zetas)), dtype=torch.float32, device=dev)
    count = torch.zeros((len(rows), P, len(zetas)), dtype=torch.int16, device=dev)
    for Jk, fm_k in zip(Js, maps):
        for z_i, zeta in enumerate(zetas):
            qk, pk, okd, _, _ = fm_k.qphi(mx, my, mz, zeta)
            if not bool(okd.any()):
                continue
            row = row_n + (qk - Qn) / grid.dq
            col = (pk + math.pi) / grid.dphi - 0.5
            T = _sample_grid(Jk, row, col, P)
            T = torch.where(okd, T, torch.full_like(T, float("nan")))
            Mt = torch.isfinite(T)
            Te = torch.where(Mt, T, torch.zeros_like(T))
            n = mb._box((Ma & Mt).to(Jr.dtype), ri, hq, hp, wmax)
            St, Stt, Sat = mb._box(Te, ri, hq, hp, wmax), mb._box(Te * Te, ri, hq, hp, wmax), mb._box(Ae * Te, ri, hq, hp, wmax)
            vt = Stt - St * St / nfull
            c = (Sat - Sa * St / nfull) / torch.sqrt(torch.clamp(va * vt, min=1e-300))
            ok = a_ok & (n == nfull) & (vt > 1e-12)
            total[:, :, z_i] += torch.where(ok, c, torch.zeros_like(c)).float()
            count[:, :, z_i] += ok.to(torch.int16)
    return total, count


def joint_fit(Ir, Is, maps, fm, q, phi, gq, gp, zeta0, mask0, s):
    """M4's joint_fit with frame k sampled through its FrameMap (see the module docstring)."""
    dt = q.dtype
    N, K = len(q), len(Is)
    u, v = mb._patch(s, gq, gp, dt)
    Q, PH = q[:, None] + u, phi[:, None] + v
    A = mb.sample(Ir, fm, Q, PH)
    a_ok = torch.isfinite(A).all(1)
    mx, my, mz = _m(Q, PH)
    zeta, zq = zeta0.clone(), torch.zeros_like(zeta0)
    h = 0.25 / gq[:, None]
    hph = 0.25 / gp[:, None]

    def warp(zeta, zq):
        Z = zeta[:, None] + zq[:, None] * u                              # (N, M)
        Ik, dIk, vts = [], [], []
        for k in range(K):
            qk, pk, okd, dq, dphi = maps[k].qphi(mx, my, mz, Z)
            Ik.append(mb.sample(Is[k], fm, qk, pk))
            Iq = (mb.sample(Is[k], fm, qk + h, pk) - mb.sample(Is[k], fm, qk - h, pk)) / (2 * h)
            tphi = torch.where(dphi == 0, torch.zeros_like(dphi),
                               (mb.sample(Is[k], fm, qk, pk + hph) - mb.sample(Is[k], fm, qk, pk - hph)) / (2 * hph) * dphi)
            dIk.append(Iq * dq + tphi)
            vts.append(okd.all(1))
        return torch.stack(Ik, 1), torch.stack(dIk, 1), torch.stack(vts, 1)

    Ik, _, vt = warp(zeta, zq)
    fr_ok = mask0 & torch.isfinite(Ik).all(2) & vt
    sd = torch.std(Ik, 2)
    flat = torch.isfinite(sd) & (sd <= mb.FLAT_STD)
    fr_ok &= ~flat
    alpha = torch.where(fr_ok, torch.std(A, 1)[:, None] / torch.clamp(sd, min=1e-300), torch.ones_like(sd))
    beta = torch.where(fr_ok, A.mean(1)[:, None] - alpha * Ik.mean(2), torch.zeros_like(sd))
    npar = 2 + 2 * K
    eye = torch.eye(npar, dtype=dt, device=q.device)

    def solve_pass(zeta, zq, alpha, beta, use, iters):
        for it in range(iters + 1):
            Ik, dIk, vt = warp(zeta, zq)
            good_f = use & torch.isfinite(Ik).all(2) & torch.isfinite(dIk).all(2) & vt
            r = A[:, None, :] - alpha[:, :, None] * Ik - beta[:, :, None]
            gz = alpha[:, :, None] * dIk
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
            if it == iters:
                return zeta, zq, alpha, beta, good_f, H, (r * r).sum((1, 2)), Ik
            x = torch.linalg.solve(H, g[..., None])[..., 0]
            okn = torch.isfinite(x).all(1)
            x = torch.where(okn[:, None], x, torch.zeros_like(x))
            zeta = zeta - x[:, 0]
            zq = torch.clamp(zq - x[:, 1], -s["zeta_q_max"], s["zeta_q_max"])
            alpha, beta = alpha - x[:, ia], beta - x[:, ib]
            zeta = torch.clamp(zeta, min=0.0)

    zeta, zq, alpha, beta, good_f, H, ssr, Ik = solve_pass(zeta, zq, alpha, beta, fr_ok, s["gn_iters"])
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
    # texture along each frame's epipolar direction in r (derotated motion at the anchor), the largest over frames
    tex = torch.full((N,), -1.0, dtype=dt, device=q.device)
    m_a = _m(q, phi)
    for k in range(K):
        idm = FrameMap(np.eye(3), maps[k].T.cpu().numpy(), q.device, dt)
        _, _, _, dq_r, dphi_r = idm.qphi(*m_a, torch.zeros_like(q))
        ex, ey = dq_r * gq, dphi_r * gp
        nrm = torch.sqrt(ex * ex + ey * ey)
        cx, cy = ex / nrm, ey / nrm
        dq_s, dp_s = (0.5 * cx / gq)[:, None], (0.5 * cy / gp)[:, None]
        gr = mb.sample(Ir, fm, Q + dq_s, PH + dp_s) - mb.sample(Ir, fm, Q - dq_s, PH - dp_s)
        tk = torch.sqrt((gr ** 2).mean(1)) / torch.abs(A.mean(1))
        tex = torch.fmax(tex, torch.where(nrm > 0, tk, torch.full_like(tk, float("nan"))))
    tex = torch.where(tex < 0, torch.full_like(tex, float("nan")), tex)
    ok = a_ok & inv_ok & torch.isfinite(sigma) & torch.isfinite(zeta)
    return {"zeta": zeta, "zeta_q": zq, "sigma_zeta": sigma, "ncc_frames": nccf, "frames": good_f, "n_frames": nf,
            "ok": ok, "texture": tex, "degenerate": flat.any(1) & ~fr_ok.any(1)}


def px_per_zeta(model, maps, qb, pb, dev):
    """Per anchor and frame: native px of frame k's sample per unit zeta at zeta = 0; returns (N, K)."""
    out = []
    to_t = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float64, device=dev)
    mx, my, mz = _m(to_t(qb), to_t(pb))
    for fm_k in maps:
        qk, pk, _, dq, dphi = fm_k.qphi(mx, my, mz, torch.zeros_like(mx))
        qk, pk, dq, dphi = (x.cpu().numpy() for x in (qk, pk, dq, dphi))
        gq = lp.native_px_per_q(model, qk, pk)
        gp = np.linalg.norm(model.pixel_of(lp.dir_of_qphi(qk, pk + 1e-5))[0]
                            - model.pixel_of(lp.dir_of_qphi(qk, pk - 1e-5))[0], axis=-1) / 2e-5
        out.append(np.hypot(dq * gq, dphi * gp))
    return np.stack(out, 1)


def match_reference(img_r, imgs, poses, model, settings=None, dev=None):
    """As multibaseline.match_reference, with poses [(A_k, T_k)] (relative_poses) instead of signed steps."""
    s = dict(SETTINGS, **(settings or {}))
    dev = dev or mb.device()
    dt = torch.float64
    fm = mb.FovMap(model, dev)
    grid = mb.Grid(model, fm, s, dev, dt)
    to_t = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=dt, device=dev)
    Ir, Is = to_t(img_r), [to_t(x) for x in imgs]
    Jr, Js = grid.unwrap(Ir), [grid.unwrap(x) for x in Is]
    maps = [FrameMap(A, T, dev, dt) for A, T in poses]
    tmax = max(float(np.linalg.norm(T)) for _, T in poses)
    zmin, zmax = s["z_envelope_m"]
    gmax = float(grid.gq.max())
    dz = s["zeta_step_px"] / (gmax * tmax)
    zetas = list(np.arange(0.0, 1.0 / zmin + dz, dz))
    e_lo, e_hi = np.radians(s["e_anchor_deg"][0]), np.radians(s["e_anchor_deg"][1])
    qn = grid.q.cpu().numpy()
    rows = torch.tensor(np.flatnonzero((qn >= lp.q_of_e(e_lo) - 2 * grid.dq) & (qn <= lp.q_of_e(e_hi) + 2 * grid.dq)),
                        device=dev)
    total, count = coarse_curves(Jr, Js, maps, zetas, grid, rows, s)
    with np.errstate(invalid="ignore"):
        C = torch.where(count >= s["min_frames"], total / torch.clamp(count.float(), min=1), torch.full_like(total, float("nan")))
    rays = model.rays()
    e_px, phi_px = lp.e_phi(rays.direction)
    sel = rays.valid & (e_px >= e_lo) & (e_px <= e_hi)
    flat = np.flatnonzero(sel.reshape(-1))
    eb, pb = e_px.reshape(-1)[flat], phi_px.reshape(-1)[flat]
    qb = lp.q_of_e(eb)
    ir = np.clip(np.rint((qb - qn[rows[0].item()]) / grid.dq).astype(int), 0, len(rows) - 1)
    ic = np.mod(np.rint((pb + np.pi) / grid.dphi - 0.5).astype(int), s["n_phi"])
    curves = C[torch.tensor(ir, device=dev), torch.tensor(ic, device=dev)]
    cnt_a = count[torch.tensor(ir, device=dev), torch.tensor(ic, device=dev)]
    cz = torch.nan_to_num(curves, nan=-2.0)
    j1 = torch.argmax(cz, 1)
    c1 = cz.gather(1, j1[:, None])[:, 0]
    gq_b = lp.native_px_per_q(model, qb, pb)
    gp_b = np.linalg.norm(model.pixel_of(lp.dir_of_e_phi(eb, pb + 1e-5))[0] - model.pixel_of(lp.dir_of_e_phi(eb, pb - 1e-5))[0],
                          axis=-1) / 2e-5
    gk = px_per_zeta(model, maps, qb, pb, dev)
    geq = gk.max(1)
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
    N = len(flat)
    out = {k: [] for k in ("zeta", "zeta_q", "sigma_zeta", "n_frames", "ok", "texture", "degenerate", "ncc_frames")}
    q_t, p_t, gq_t, gp_t = to_t(qb), to_t(pb), to_t(gq_b), to_t(gp_b)
    for a in range(0, N, s["chunk"]):
        b = min(N, a + s["chunk"])
        m0 = torch.ones((b - a, len(Is)), dtype=torch.bool, device=dev)
        f = joint_fit(Ir, Is, maps, fm, q_t[a:b], p_t[a:b], gq_t[a:b], gp_t[a:b], zeta0[a:b], m0, s)
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
            "edge_dev_px": edge_dev, "px_per_zeta": geq, "px_per_zeta_frames": gk, "reason": reason,
            "frames_valid_coarse": cnt_a.max(1).values.cpu().numpy(), "zeta_step": dz, "settings": s,
            "ncc_frames": fit["ncc_frames"]}
