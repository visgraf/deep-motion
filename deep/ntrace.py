"""Stage 11 (#26): N, M4's matcher on exact traces of the panorama, for a straight path.

N is motion/multibaseline.py (M4, frozen at #10) carried over to the 720-wide panorama, step for step, with M4's frozen
settings imported, not copied (rule 7: motion/ is left as imported):

- Anchors: every panorama pixel with |elevation| < 70 deg and 1 deg <= psi <= 179 deg from the heading (M4: every
  native pixel at 1-44 deg).
- Patch: 7 samples along the trace by 3 across, one panorama pixel (0.5 deg) apart, laid out in angle on the sphere
  (M4: 7 native px radially by 3 in azimuth).
- Hypothesis: inverse range zeta along the anchor's ray, and a slope gamma of zeta per pixel along the trace (M4:
  axial inverse depth and its slope per unit q). Each patch sample is carried along its own ray at inverse range
  zeta + gamma a (a its offset along the trace, px) and projected through each frame's exact pose.
- Coarse: mean zero-mean NCC over the frames on a zeta grid of 0.5 px of the longest step's displacement, with M4's
  box-filter window sums. Where M4 unwraps its raster to (q, phi), N unwraps the panorama to the line's own canonical
  coordinates, the epipolar pencil about the path line (psi from the heading, plane angle omega; docs/plan.md, "The
  formulation"). A static point keeps omega and moves to psi' = atan2(sin psi, cos psi - t zeta); one grid row is one
  panorama pixel along the trace.
- Fine: M4's joint Gauss-Newton per anchor on zeta, gamma and a gain and offset per frame, on the native panorama
  through exact projection; then frame consistency (frames under NCC 0.5 dropped, refit).
- Rejections: M4's, in M4's order, with M4's thresholds in panorama px.

Two translations, said here:
- M4's slope bound zeta_q_max (per unit q, axial inverse depth) becomes |gamma| <= zeta_q_max res / sin psi, through
  dq = res / (sin psi cos psi) per px along the trace and d zeta_range = cos psi d zeta_axial.
- M4's coarse zeta step uses the largest px per unit zeta over its grid (the fovea's); N's is 1 / res (psi = 90 deg),
  the panorama's largest. Both are first order at zeta = 0, as M4's.

Straight paths only: every frame must share the reference's rotation and lie on its heading line (the analogue of
#5's alignment gate, checked in match_reference).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from motion import multibaseline as mb

_FROM_M4 = ("r_half_px", "a_half_px", "z_envelope_m", "zeta_step_px", "excl_px", "ncc_min", "ambig_margin",
            "texture_min", "gn_iters", "gn_max_move_px", "zeta_q_max", "frame_ncc_min", "min_frames", "support_min",
            "support_tol_px", "edge_half_px", "edge_tol_px", "chunk")
SETTINGS = {k: mb.SETTINGS[k] for k in _FROM_M4}
SETTINGS.update({"psi_anchor_deg": (1.0, 179.0), "lat_max_deg": 70.0, "align_tol": 1e-6})
REASONS = mb.REASONS
FLAT_STD = mb.FLAT_STD
Z95 = mb.Z95


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------- the panorama (torch)

def pix_of(v_local, W):
    """fov-render's PanoramaModel.pixel_of in torch: local directions (..., 3) -> (row, col), centres at integers."""
    H = W // 2
    n = torch.linalg.norm(v_local, dim=-1)
    lon = torch.atan2(v_local[..., 0], -v_local[..., 2])
    lat = torch.asin(torch.clamp(v_local[..., 1] / n, -1.0, 1.0))
    col = (lon / (2 * math.pi) + 0.5) * W - 0.5
    row = (0.5 - lat / math.pi) * H - 0.5
    return row, col


def bilinear(img, r, c):
    """img (H, W), columns wrap, rows do not; nan outside the rows or where a neighbour is nan."""
    H, W = img.shape
    r0f, c0f = torch.floor(r), torch.floor(c)
    fr, fc = r - r0f, c - c0f
    r0 = r0f.long()
    ok = (r0 >= 0) & (r0 + 1 <= H - 1)
    r0 = r0.clamp(0, H - 2)
    c0 = torch.remainder(c0f.long(), W)
    c1 = torch.remainder(c0 + 1, W)
    flat = img.reshape(-1)
    a, b = flat[r0 * W + c0], flat[r0 * W + c1]
    d, e = flat[(r0 + 1) * W + c0], flat[(r0 + 1) * W + c1]
    v = (1 - fr) * ((1 - fc) * a + fc * b) + fr * ((1 - fc) * d + fc * e)
    return torch.where(ok, v, torch.full_like(v, float("nan")))


def sample_world(img, X, c, R, W):
    """The panorama of a look at (c, R) sampled where it sees world points X (..., 3)."""
    v = (X - c) @ R
    r, col = pix_of(v, W)
    return bilinear(img, r, col)


# ----------------------------------------------------------------------------- the line's pencil

class Pencil:
    """Epipolar-plane pencil about the path line: psi from the heading h, plane angle omega about h (0 at up)."""

    def __init__(self, h, up=(0.0, 0.0, 1.0)):
        self.h = _unit(h)
        up = np.asarray(up, float)
        e1 = up - (up @ self.h) * self.h
        self.e1 = _unit(e1)
        self.e2 = np.cross(self.h, self.e1)

    def dirs(self, psi, om):
        if isinstance(psi, torch.Tensor):
            h, e1, e2 = (torch.tensor(x, dtype=psi.dtype, device=psi.device) for x in (self.h, self.e1, self.e2))
            s = torch.sin(psi)[..., None]
            return torch.cos(psi)[..., None] * h + s * (torch.cos(om)[..., None] * e1 + torch.sin(om)[..., None] * e2)
        psi, om = np.asarray(psi, float), np.asarray(om, float)
        s = np.sin(psi)[..., None]
        return np.cos(psi)[..., None] * self.h + s * (np.cos(om)[..., None] * self.e1 + np.sin(om)[..., None] * self.e2)

    def coords(self, u):
        u = _unit(u)
        psi = np.arccos(np.clip(u @ self.h, -1.0, 1.0))
        om = np.arctan2(u @ self.e2, u @ self.e1)
        return psi, om


def line_psi(psi, t, zeta):
    """Where a static point at inverse range zeta along a reference ray at psi is seen from t further along the line."""
    if isinstance(psi, torch.Tensor):
        return torch.atan2(torch.sin(psi), torch.cos(psi) - t * zeta)
    return np.arctan2(np.sin(psi), np.cos(psi) - t * zeta)


def alignment(c_r, R_r, poses, tol=SETTINGS["align_tol"]):
    """The line's gate: every frame shares the reference's rotation and lies on its heading line."""
    h = -R_r[:, 2]
    worst_rot, worst_off = 0.0, 0.0
    for c, R in poses:
        worst_rot = max(worst_rot, float(np.abs(np.asarray(R) - R_r).max()))
        d = np.asarray(c, float) - c_r
        worst_off = max(worst_off, float(np.linalg.norm(d - (d @ h) * h)))
    return {"ok": worst_rot <= tol and worst_off <= tol, "rotation": worst_rot, "off_line_m": worst_off}


# ----------------------------------------------------------------------------- coarse, on the pencil grid

class Grid:
    """The pencil grid: rows psi_i = (i + 1/2) res (one panorama pixel along the trace), columns omega_j at res."""

    def __init__(self, pencil: Pencil, W, dev, dtype=torch.float64):
        self.res = 2 * math.pi / W
        self.n_psi, self.n_om = int(round(math.pi / self.res)), W
        self.psi = torch.tensor((np.arange(self.n_psi) + 0.5) * self.res, dtype=dtype, device=dev)
        self.om = torch.tensor(-math.pi + (np.arange(self.n_om) + 0.5) * self.res, dtype=dtype, device=dev)
        PS, OM = torch.meshgrid(self.psi, self.om, indexing="ij")
        self.dirs = pencil.dirs(PS, OM)                                  # (n_psi, n_om, 3) world
        self.W = W

    def unwrap(self, img, c, R):
        """The look's panorama on the grid: the look sees direction d from its own centre (a translation keeps the
        pencil; a straight path has no rotation)."""
        R_t = torch.as_tensor(R, dtype=self.dirs.dtype, device=self.dirs.device)
        r, col = pix_of(self.dirs @ R_t, self.W)
        return bilinear(img, r, col)


def coarse_curves(Jr, Js, ts, zetas, grid: Grid, rows, s):
    """Mean NCC over frames (len(rows), n_om, n_zeta) and the count of frames with a full window; M4's coarse_curves
    with a per-row shift: row i of frame k under zeta is read at psi' = atan2(sin psi_i, cos psi_i - t_k zeta)."""
    dev, dt = Jr.device, Jr.dtype
    R_, P = Jr.shape
    hq = torch.full((len(rows),), int(s["r_half_px"]), dtype=torch.long, device=dev)
    sinp = torch.sin(grid.psi[rows])
    hp = torch.clamp(torch.round(s["a_half_px"] / sinp), min=0).long()
    hmax, wmax = int(hq.max()), int(hp.max())
    lo, hi = int(rows.min()) - hmax, int(rows.max()) + hmax
    ext = torch.arange(lo, hi + 1, device=dev)
    ri = rows - lo
    nfull = ((2 * hq + 1) * (2 * hp + 1)).to(dt)[:, None]

    def take_rows(J, rr):
        ok = (rr >= 0) & (rr < R_)
        out = J[rr.clamp(0, R_ - 1)]
        return torch.where(ok[:, None], out, torch.full_like(out, float("nan")))

    A = take_rows(Jr, ext)
    Ma = torch.isfinite(A)
    Ae = torch.where(Ma, A, torch.zeros_like(A))
    Sa, Saa = mb._box(Ae, ri, hq, hp, wmax), mb._box(Ae * Ae, ri, hq, hp, wmax)
    na = mb._box(Ma.to(dt), ri, hq, hp, wmax)
    va = Saa - Sa * Sa / nfull
    a_ok = (na == nfull) & (va > 1e-12)
    total = torch.zeros((len(rows), P, len(zetas)), dtype=torch.float32, device=dev)
    count = torch.zeros((len(rows), P, len(zetas)), dtype=torch.int16, device=dev)
    psi_ext = (ext.to(dt) + 0.5) * grid.res
    for Jk, t in zip(Js, ts):
        for z_i, zeta in enumerate(zetas):
            pk = line_psi(psi_ext, float(t), float(zeta))
            x = pk / grid.res - 0.5                                      # fractional source row of each ext row
            f0 = torch.floor(x)
            fr = (x - f0)[:, None]
            f0 = f0.long()
            T0, T1 = take_rows(Jk, f0), take_rows(Jk, f0 + 1)
            T = T0 * (1 - fr) + T1 * fr
            Mt = torch.isfinite(T)
            Te = torch.where(Mt, T, torch.zeros_like(T))
            n = mb._box((Ma & Mt).to(dt), ri, hq, hp, wmax)
            St, Stt, Sat = mb._box(Te, ri, hq, hp, wmax), mb._box(Te * Te, ri, hq, hp, wmax), mb._box(Ae * Te, ri, hq, hp, wmax)
            vt = Stt - St * St / nfull
            c = (Sat - Sa * St / nfull) / torch.sqrt(torch.clamp(va * vt, min=1e-300))
            ok = a_ok & (n == nfull) & (vt > 1e-12)
            total[:, :, z_i] += torch.where(ok, c, torch.zeros_like(c)).float()
            count[:, :, z_i] += ok.to(torch.int16)
    return total, count


# ----------------------------------------------------------------------------- fine, on the native panorama

def _patch(s, dt, dev):
    a = torch.arange(-s["r_half_px"], s["r_half_px"] + 1, dtype=dt, device=dev)
    b = torch.arange(-s["a_half_px"], s["a_half_px"] + 1, dtype=dt, device=dev)
    Aa, Bb = torch.meshgrid(a, b, indexing="ij")
    return Aa.reshape(1, -1), Bb.reshape(1, -1)                          # (1, M) along, across (px)


def patch_dirs(pencil, psi, om, a, b, res):
    """World directions of the patch samples: along the trace a px (psi), across b px (omega at res / sin psi)."""
    PS = psi[:, None] + a * res
    OM = om[:, None] + b * res / torch.sin(psi)[:, None]
    return pencil.dirs(PS, OM)


def joint_fit(Ir, Is, c_r, R_r, frames, pencil, psi, om, geq, zeta0, mask0, s, W, cost_pixel=False):
    """M4's joint_fit on the panorama. Ir (H, W), Is list of K images, frames list of K (c, R) tensors; anchors (N,)
    psi, om, geq (px per unit zeta, longest step), zeta0 (N,). Returns M4's fields."""
    dt, dev = psi.dtype, psi.device
    N, K = len(psi), len(Is)
    res = 2 * math.pi / W
    a, b = _patch(s, dt, dev)
    U = patch_dirs(pencil, psi, om, a, b, res)                           # (N, M, 3)
    A = sample_world(Ir, c_r + U, c_r, R_r, W)                           # (N, M)
    a_ok = torch.isfinite(A).all(1)
    gmax = s["zeta_q_max"] * res / torch.sin(psi)                        # the slope bound (module docstring)
    zeta, gam = zeta0.clone(), torch.zeros_like(zeta0)
    hpx = 0.25

    def warp(zeta, gam):
        Z = zeta[:, None] + gam[:, None] * a                             # (N, M) inverse range per sample
        valid = Z > 1e-9
        Zs = torch.clamp(Z, min=1e-9)
        X = c_r + U / Zs[..., None]                                      # (N, M, 3)
        dX = -U / (Zs ** 2)[..., None]                                   # d X / d Z
        Ik, dIk = [], []
        for (ck, Rk), Imk in zip(frames, Is):
            v = (X - ck) @ Rk
            r, col = pix_of(v, W)
            eps = 1e-6
            r2, c2 = pix_of(v + eps * (dX @ Rk), W)
            dr, dc = (r2 - r) / eps, torch.remainder(c2 - col + W / 2, W) - W / 2
            dc = dc / eps
            nrm = torch.sqrt(dr * dr + dc * dc)
            er, ec = dr / torch.clamp(nrm, min=1e-300), dc / torch.clamp(nrm, min=1e-300)
            Ik.append(bilinear(Imk, r, col))
            g = (bilinear(Imk, r + hpx * er, col + hpx * ec) - bilinear(Imk, r - hpx * er, col - hpx * ec)) / (2 * hpx)
            dIk.append(g * nrm)                                          # d I / d Z at each sample
        return torch.stack(Ik, 1), torch.stack(dIk, 1), valid.all(1)[:, None].expand(N, K)

    Ik, _, vt = warp(zeta, gam)
    fr_ok = mask0 & torch.isfinite(Ik).all(2) & vt
    sd = torch.std(Ik, 2)
    flat = torch.isfinite(sd) & (sd <= FLAT_STD)
    fr_ok &= ~flat
    alpha = torch.where(fr_ok, torch.std(A, 1)[:, None] / torch.clamp(sd, min=1e-300), torch.ones_like(sd))
    beta = torch.where(fr_ok, A.mean(1)[:, None] - alpha * Ik.mean(2), torch.zeros_like(sd))
    npar = 2 + 2 * K
    eye = torch.eye(npar, dtype=dt, device=dev)
    ia = torch.arange(K, device=dev) + 2
    ib = ia + K

    def solve_pass(zeta, gam, alpha, beta, use, iters):
        for it in range(iters + 1):
            Ik, dIk, vt = warp(zeta, gam)
            good_f = use & torch.isfinite(Ik).all(2) & torch.isfinite(dIk).all(2) & vt
            r = A[:, None, :] - alpha[:, :, None] * Ik - beta[:, :, None]
            gz = alpha[:, :, None] * dIk
            Jz, Jg, Ja, Jb = -gz, -gz * a[:, None, :], -Ik, -torch.ones_like(Ik)
            w = good_f[:, :, None].to(dt)
            r = torch.nan_to_num(r) * w
            Jz, Jg, Ja, Jb = (torch.nan_to_num(x) * w for x in (Jz, Jg, Ja, Jb))
            H = torch.zeros((N, npar, npar), dtype=dt, device=dev)
            g = torch.zeros((N, npar), dtype=dt, device=dev)
            H[:, 0, 0] = (Jz * Jz).sum((1, 2)); H[:, 0, 1] = H[:, 1, 0] = (Jz * Jg).sum((1, 2))
            H[:, 1, 1] = (Jg * Jg).sum((1, 2))
            g[:, 0], g[:, 1] = (Jz * r).sum((1, 2)), (Jg * r).sum((1, 2))
            H[:, 0, ia] = H[:, ia, 0] = (Jz * Ja).sum(2); H[:, 0, ib] = H[:, ib, 0] = (Jz * Jb).sum(2)
            H[:, 1, ia] = H[:, ia, 1] = (Jg * Ja).sum(2); H[:, 1, ib] = H[:, ib, 1] = (Jg * Jb).sum(2)
            H[:, ia, ia] = (Ja * Ja).sum(2); H[:, ib, ib] = (Jb * Jb).sum(2)
            H[:, ia, ib] = H[:, ib, ia] = (Ja * Jb).sum(2)
            g[:, ia], g[:, ib] = (Ja * r).sum(2), (Jb * r).sum(2)
            H = H + 1e-12 * eye
            if it == iters:
                return zeta, gam, alpha, beta, good_f, H, (r * r).sum((1, 2)), Ik
            x = torch.linalg.solve(H, g[..., None])[..., 0]
            okn = torch.isfinite(x).all(1)
            x = torch.where(okn[:, None], x, torch.zeros_like(x))
            zeta = torch.clamp(zeta - x[:, 0], min=0.0)
            gam = torch.maximum(torch.minimum(gam - x[:, 1], gmax), -gmax)
            alpha, beta = alpha - x[:, ia], beta - x[:, ib]

    zeta, gam, alpha, beta, good_f, H, ssr, Ik = solve_pass(zeta, gam, alpha, beta, fr_ok, s["gn_iters"])
    za = A - A.mean(1, keepdim=True)

    def frame_ncc(Ik, good_f):
        zk = Ik - Ik.mean(2, keepdim=True)
        n = (zk * za[:, None, :]).sum(2) / torch.sqrt(torch.clamp((zk * zk).sum(2) * (za * za).sum(1)[:, None], min=1e-300))
        return torch.where(good_f, n, torch.full_like(n, float("nan")))

    nccf = frame_ncc(Ik, good_f)
    keep = good_f & (nccf >= s["frame_ncc_min"])
    redo = (keep != good_f).any(1)
    if bool(redo.any()):
        z2, g2, a2, b2, gf2, H2, ssr2, Ik2 = solve_pass(zeta, gam, alpha, beta, keep, max(2, s["gn_iters"] // 2))
        sel = redo
        zeta, gam = torch.where(sel, z2, zeta), torch.where(sel, g2, gam)
        good_f = torch.where(sel[:, None], gf2, good_f)
        H = torch.where(sel[:, None, None], H2, H)
        ssr = torch.where(sel, ssr2, ssr)
        Ik = torch.where(sel[:, None, None], Ik2, Ik)
        nccf = frame_ncc(Ik, good_f)
    nf = good_f.sum(1)
    M = A.shape[1]
    dof = torch.clamp(nf * M - (2 + 2 * nf), min=1).to(dt)
    s2 = ssr / dof
    cov00 = torch.full((N,), float("nan"), dtype=dt, device=dev)
    inv_ok = torch.zeros(N, dtype=torch.bool, device=dev)
    try:
        cov00 = torch.linalg.inv(H)[:, 0, 0]
        inv_ok = torch.ones(N, dtype=torch.bool, device=dev)
    except RuntimeError:
        for n_ in range(N):
            try:
                cov00[n_] = torch.linalg.inv(H[n_])[0, 0]
                inv_ok[n_] = True
            except RuntimeError:
                pass
    sigma = torch.sqrt(torch.clamp(s2 * cov00, min=0.0))
    # texture along the trace of r's patch (M4's A2): samples half a pixel either side along the trace
    Up = patch_dirs(pencil, psi + 0.5 * res, om, a, b, res)
    Um = patch_dirs(pencil, psi - 0.5 * res, om, a, b, res)
    gr = sample_world(Ir, c_r + Up, c_r, R_r, W) - sample_world(Ir, c_r + Um, c_r, R_r, W)
    tex = torch.sqrt((gr ** 2).mean(1)) / torch.abs(A.mean(1))
    ok = a_ok & inv_ok & torch.isfinite(sigma) & torch.isfinite(zeta)
    return {"zeta": zeta, "gamma": gam, "gamma_max": gmax, "sigma_zeta": sigma, "ncc_frames": nccf, "frames": good_f,
            "n_frames": nf, "ok": ok, "texture": tex, "degenerate": flat.any(1) & ~fr_ok.any(1)}


# ----------------------------------------------------------------------------- the reference

def anchor_pixels(img_r, R_r, W, s, mask=None):
    """The panorama pixels N tries: finite value, |elevation| < lat_max, psi within psi_anchor_deg (and `mask`)."""
    H = W // 2
    rows, cols = np.mgrid[0:H, 0:W]
    lon = (cols + 0.5) / W * 2 * np.pi - np.pi
    lat = (0.5 - (rows + 0.5) / H) * np.pi
    vloc = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1)
    u = vloc @ R_r.T
    h = -R_r[:, 2]
    psi = np.arccos(np.clip(u @ h, -1, 1))
    lo, hi = np.radians(s["psi_anchor_deg"][0]), np.radians(s["psi_anchor_deg"][1])
    sel = np.isfinite(img_r) & (np.abs(lat) < np.radians(s["lat_max_deg"])) & (psi >= lo) & (psi <= hi)
    if mask is not None:
        sel &= mask
    return np.stack([rows[sel], cols[sel]], -1), u[sel]


def match_reference(img_r, c_r, R_r, imgs, poses, W, settings=None, dev=None, mask=None, ts_override=None):
    """N at one reference. img_r and imgs: luminance panoramas (numpy, nan where nothing is hit); c_r, R_r the
    reference's pose; poses [(c_k, R_k)] the frames' (the mirrored control passes them mirrored). mask: optional
    (H, W) bool restricting the anchors. Returns per anchor numpy arrays, M4's fields and N's."""
    s = dict(SETTINGS, **(settings or {}))
    dev = dev or device()
    dt = torch.float64
    c_r, R_r = np.asarray(c_r, float), np.asarray(R_r, float)
    al = alignment(c_r, R_r, poses, s["align_tol"])
    if not al["ok"]:
        raise ValueError(f"[N] refused: the frames are not on the reference's line ({al})")
    h = -R_r[:, 2]
    ts = [float((np.asarray(c, float) - c_r) @ h) for c, _ in poses] if ts_override is None else list(ts_override)
    tmax = max(abs(t) for t in ts)
    res = 2 * math.pi / W
    pencil = Pencil(h)
    grid = Grid(pencil, W, dev, dt)
    T = lambda x: torch.as_tensor(np.asarray(x, np.float64), dtype=dt, device=dev)
    Ir, Is = T(img_r), [T(x) for x in imgs]
    cr_t, Rr_t = T(c_r), T(R_r)
    frames = [(T(c), T(R)) for c, R in poses]
    Jr = grid.unwrap(Ir, c_r, R_r)
    Js = [grid.unwrap(I, c, R) for I, (c, R) in zip(Is, poses)]
    zmin, zmax = s["z_envelope_m"]
    dz = s["zeta_step_px"] * res / tmax                                  # 0.5 px at the largest px per zeta, 1 / res
    zetas = list(np.arange(0.0, 1.0 / zmin + dz, dz))
    px, u = anchor_pixels(img_r, R_r, W, s, mask)
    psi_a, om_a = pencil.coords(u)
    ir = np.clip(np.rint(psi_a / res - 0.5).astype(int), 0, grid.n_psi - 1)
    ic = np.mod(np.rint((om_a + np.pi) / res - 0.5).astype(int), grid.n_om)
    urow = np.unique(ir)
    rows = torch.tensor(np.arange(urow.min(), urow.max() + 1), device=dev)
    total, count = coarse_curves(Jr, Js, ts, zetas, grid, rows, s)
    C = torch.where(count >= s["min_frames"], total / torch.clamp(count.float(), min=1), torch.full_like(total, float("nan")))
    ir_t, ic_t = torch.tensor(ir - urow.min(), device=dev), torch.tensor(ic, device=dev)
    curves = C[ir_t, ic_t]
    cz = torch.nan_to_num(curves, nan=-2.0)
    j1 = torch.argmax(cz, 1)
    c1 = cz.gather(1, j1[:, None])[:, 0]
    geq = tmax * np.sin(psi_a) / res                                     # px per unit zeta at the longest step
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
    del total, count, C, curves
    N = len(px)
    keys = ("zeta", "gamma", "gamma_max", "sigma_zeta", "n_frames", "ok", "texture", "degenerate", "ncc_frames")
    out = {k: [] for k in keys}
    psi_t, om_t, geq_t = T(psi_a), T(om_a), T(geq)
    for a0 in range(0, N, s["chunk"]):
        b0 = min(N, a0 + s["chunk"])
        m0 = torch.ones((b0 - a0, len(Is)), dtype=torch.bool, device=dev)
        f = joint_fit(Ir, Is, cr_t, Rr_t, frames, pencil, psi_t[a0:b0], om_t[a0:b0], geq_t[a0:b0], zeta0[a0:b0], m0, s, W)
        for k in keys:
            out[k].append(f[k])
    fit = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    zeta, sig = fit["zeta"], fit["sigma_zeta"]
    z0 = zeta0.cpu().numpy()
    c1n, c2n, j1n = c1.cpu().numpy(), c2.cpu().numpy(), j1.cpu().numpy()
    reason = np.zeros(N, dtype=np.int8)

    def mark(cond, name):
        reason[(reason == 0) & cond] = REASONS.index(name)
    mark(fit["degenerate"], "gn_failed")
    mark(~fit["ok"] | (c1n <= -2.0), "invalid_support")
    mark(~(fit["texture"] >= s["texture_min"]), "low_texture")
    mark(~(c1n >= s["ncc_min"]), "low_ncc")
    mark(c2n >= c1n - s["ambig_margin"], "ambiguous")
    mark((zeta < 1.0 / zmax) | (z0 < 1.0 / zmax), "unresolved_far")
    mark(j1n >= Z - 1, "near_boundary")
    moved = np.abs(zeta - z0) * geq
    mark(~(moved <= s["gn_max_move_px"]) | ~(np.abs(fit["gamma"]) < fit["gamma_max"]), "gn_failed")
    mark(fit["n_frames"] < s["min_frames"], "few_frames")
    # neighbour support and depth edge (M4's), on the panorama raster (columns wrap), px of the longest step
    H = W // 2

    def ras(v, cond):
        R_ = np.full((H, W), np.nan)
        R_[px[cond, 0], px[cond, 1]] = v[cond]
        return R_

    def shifted(R_, di, dj):
        out_ = np.full_like(R_, np.nan)
        out_[max(-di, 0):H + min(-di, 0)] = R_[max(di, 0):H + min(di, 0)]
        return np.roll(out_, -dj, 1)

    allm = np.ones(N, bool)
    cand = reason == 0
    Dz, G = ras(zeta, cand), ras(geq, allm)
    sup = np.zeros((H, W), np.int16)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                with np.errstate(invalid="ignore"):
                    sup += (np.abs(shifted(Dz, di, dj) - Dz) * G <= s["support_tol_px"]).astype(np.int16)
    support = sup[px[:, 0], px[:, 1]]
    mark(cand & (support < s["support_min"]), "unsupported")
    cand2 = reason == 0
    Dz2, Pr, Kg = ras(zeta, cand2), ras(psi_a / res, allm), ras(fit["gamma"], allm)
    dev_ = np.zeros((H, W))
    hh = int(s["edge_half_px"])
    for di in range(-hh, hh + 1):
        for dj in range(-hh, hh + 1):
            if di or dj:
                with np.errstate(invalid="ignore"):
                    dv = np.abs(shifted(Dz2, di, dj) - (Dz2 + Kg * (shifted(Pr, di, dj) - Pr))) * G
                dev_ = np.fmax(dev_, np.where(np.isfinite(dv), dv, 0.0))
    edge_dev = dev_[px[:, 0], px[:, 1]]
    mark(cand2 & (edge_dev > s["edge_tol_px"]), "depth_edge")
    return {"px": px, "u": u, "psi": psi_a, "omega": om_a, "zeta": zeta, "gamma": fit["gamma"], "sigma_zeta": sig,
            "sigma_px": sig * geq, "n_frames": fit["n_frames"], "ncc_coarse": c1n, "ncc_second": c2n, "zeta_coarse": z0,
            "texture": fit["texture"], "support": support, "edge_dev_px": edge_dev, "px_per_zeta": geq, "reason": reason,
            "ts": np.asarray(ts), "zeta_step": dz, "n_zeta": Z, "settings": s, "alignment": al}


# ----------------------------------------------------------------------------- selftest (numpy and torch, no renders)

def selftest():
    """N's geometry against fov-render and the exact projection; each check ships a control that must fail."""
    from fov_render.sensors import PanoramaModel
    res_ = []
    W = 720
    rng = np.random.default_rng(26)
    d = _unit(rng.normal(size=(5000, 3)))
    ref, _ = PanoramaModel(W).pixel_of(d)
    r, c = pix_of(torch.tensor(d), W)
    got = np.stack([r.numpy(), c.numpy()], -1)
    e = np.abs(got - ref)
    e[:, 1] = np.minimum(e[:, 1], W - e[:, 1])
    bad = d * np.array([1, -1, 1])
    rb, cb = pix_of(torch.tensor(bad), W)
    eb = np.abs(np.stack([rb.numpy(), cb.numpy()], -1) - ref).max()
    res_.append(("N.1 torch pix_of equals fov-render's PanoramaModel.pixel_of (<= 1e-9 px)", float(e.max()) <= 1e-9,
                 float(e.max()), "control: elevation sign flipped (> 1 px)", eb > 1, float(eb)))
    R_EYE = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    pen = Pencil(-R_EYE[:, 2])
    psi, om = pen.coords(d)
    back = pen.dirs(psi, om)
    rt = float(np.abs(back - d).max())
    pen_bad = Pencil(-R_EYE[:, 2]); pen_bad.e2 = -pen_bad.e2
    rtb = float(np.abs(pen_bad.dirs(psi, om) - d).max())
    res_.append(("N.2 the pencil's coordinates round-trip (<= 1e-12)", rt <= 1e-12, rt,
                 "control: the plane angle's second axis flipped (> 0.1)", rtb > 0.1, rtb))
    # the line law against the exact projection
    h = pen.h
    zeta = rng.uniform(0.05, 2.0, 5000)
    t = rng.uniform(-0.5, 0.5, 5000)
    X = d / zeta[:, None]
    psi_k, om_k = pen.coords(X - t[:, None] * h)
    law = line_psi(psi, t, zeta)
    el = float(np.abs(law - psi_k).max())
    eo = float(np.abs(np.angle(np.exp(1j * (om_k - om)))).max())
    lawb = line_psi(psi, -t, zeta)
    elb = float(np.median(np.abs(lawb - psi_k)))
    res_.append(("N.3 the line law psi' = atan2(sin psi, cos psi - t zeta) and omega kept, against the exact projection"
                 " (<= 1e-9 rad)", max(el, eo) <= 1e-9, max(el, eo), "control: t's sign flipped (median > 1e-3 rad)",
                 elb > 1e-3, elb))
    # px per zeta at the longest step: first order, against a finite difference
    psi_s = rng.uniform(np.radians(1), np.radians(179), 2000)
    eps = 1e-7
    fd = np.abs(line_psi(psi_s, 0.5, eps) - psi_s) / eps / (2 * np.pi / W)
    geq = 0.5 * np.sin(psi_s) / (2 * np.pi / W)
    eg = float(np.max(np.abs(fd - geq) / geq))
    egb = float(np.median(np.abs(fd - geq * np.abs(np.cos(psi_s))) / geq))
    res_.append(("N.4 px per unit zeta at the longest step = t_max sin psi / res (relative <= 1e-5)", eg <= 1e-5, eg,
                 "control: times |cos psi| (median > 0.1)", egb > 0.1, egb))
    # the settings are M4's frozen ones
    same = all(SETTINGS[k] == mb.SETTINGS[k] for k in _FROM_M4)
    res_.append(("N.5 N's thresholds are M4's frozen SETTINGS (motion/multibaseline.py)", same, None,
                 "control: ncc_min - 0.01 differs", dict(SETTINGS, ncc_min=SETTINGS["ncc_min"] - 0.01)["ncc_min"] != mb.SETTINGS["ncc_min"], None))
    return res_
