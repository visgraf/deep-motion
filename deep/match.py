"""Stage 7a (#14): real matching along traces. R0 (a Lambertian world) and the trace sweep (B).

R0: the true points and colours of one clean look (256 spp, the room's EYE) re-projected to every pose of a window
with a z-buffer, nearest first per pixel, small holes filled from neighbours. Each point keeps its colour exactly;
there is no rendering noise and no shading change; surfaces the reference look does not see are missing.

B, the trace sweep: for each anchor of the reference look (the window's middle), inverse range along its ray is
swept; for each hypothesis the point is projected into every look and the image sampled there (bilinear, 3x3 box
pre-blur); the cost is the mean truncated squared difference from the anchor's own value. Coarse sweep on every
tenth look, then a fine sweep around the best minima on every look; parabolic refinement. Confidence: the second-best
minimum (outside +/-3 bins) over the best.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import oracle as orc
from . import sensor as sn

R0_REF = "runs/r0-ref/k0000/record_C.npz"
LUM = np.array([0.2126, 0.7152, 0.0722])


# ------------------------------------------------------------------------------------------------------------ R0

def r0_world(ref=R0_REF):
    rec = orc.record(ref)
    hit = np.isfinite(rec.distance_m)
    P = rec.position_w[hit].astype(np.float64)
    g = rec.value[hit].astype(np.float64) @ LUM
    g = g / np.quantile(g, 0.99)
    return P, g, int(rec.raster_shape[1])


def _fill(idx, max_gap=2):
    from .compare import _fill_1d
    return _fill_1d(_fill_1d(idx, max_gap, 1), max_gap, 0)


def r0_look(P, g, c, R, fan):
    """The panorama a look at (c, R) sees of the world (P, g): grey image (H, W) and the point index per pixel."""
    W = fan.width
    H = W // 2
    v = (P - c) @ R
    rng = np.linalg.norm(v, axis=-1)
    rc, _ = fan.model().pixel_of(v)
    i = np.rint(rc[:, 0]).astype(np.int64)                     # pixel_of puts pixel centres at integers
    j = np.rint(rc[:, 1]).astype(np.int64) % W
    ok = (i >= 0) & (i < H)
    order = np.argsort(-rng[ok])
    ii, jj = i[ok][order], j[ok][order]
    idx = np.full((H, W), -1, np.int64)
    idx[ii, jj] = np.flatnonzero(ok)[order]
    idx = _fill(idx)
    img = np.where(idx >= 0, g[np.maximum(idx, 0)], np.nan)
    return img.astype(np.float32), idx


def r0_run(kind, n=101, ds=0.01, R=1.6):
    """R0 looks for a 5a-style window from the room's EYE: [(s, c, R, img, idx)], the world (P, g) and the fan."""
    P, g, W = r0_world()
    fan = sn.SlitFan(width=W)
    poses = orc.run_poses(kind, n=n, ds=ds, R=R)
    looks = []
    for s, c, Rh in poses:
        img, idx = r0_look(P, g, c, Rh, fan)
        looks.append((s, c, Rh, img, idx))
    return looks, P, g, fan


# ------------------------------------------------------------------------------------------------- the sweep (B)

def _gauss(img, sigma):
    """Separable Gaussian blur (columns wrap, rows clamp), normalised over valid pixels; nan where support is thin."""
    if sigma <= 0:
        return _blur(img)
    rad = int(np.ceil(3 * sigma))
    x = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    a = np.nan_to_num(img, nan=0.0).astype(np.float64)
    m = np.isfinite(img).astype(np.float64)
    def conv(z):
        z = sum(w * np.roll(z, -o, 1) for w, o in zip(k, x))
        zp = np.pad(z, ((rad, rad), (0, 0)), mode="edge")
        return sum(w * zp[rad + o:rad + o + z.shape[0]] for w, o in zip(k, x))
    s, w = conv(a), conv(m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0.5, s / w, np.nan).astype(np.float32)


def _blur(img):
    a = np.nan_to_num(img, nan=0.0)
    m = np.isfinite(img).astype(np.float32)
    k = lambda x: (np.roll(x, 1, 1) + x + np.roll(x, -1, 1))
    s = k(a)
    s = np.vstack([s[:1], s[:-1]]) + s + np.vstack([s[1:], s[-1:]])
    w = k(m)
    w = np.vstack([w[:1], w[:-1]]) + w + np.vstack([w[1:], w[-1:]])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 4, s / w, np.nan).astype(np.float32)


def _sample_nearest(img, rc):
    H, W = img.shape
    r = np.rint(rc[:, 0]).astype(np.int64)
    c = np.rint(rc[:, 1]).astype(np.int64) % W
    ok = (r >= 0) & (r < H)
    return np.where(ok, img[np.clip(r, 0, H - 1), c], np.nan)


def _sample(img, rc):
    """Bilinear sample of img (H, W; columns wrap) at rc (N, 2); nan outside rows or on a nan neighbour."""
    H, W = img.shape
    r, c = rc[:, 0], rc[:, 1]                                  # pixel centres at integers
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    fr, fc = r - r0, c - c0
    ok = (r0 >= 0) & (r0 + 1 < H)
    r0c = np.clip(r0, 0, H - 2)
    c0w, c1w = c0 % W, (c0 + 1) % W
    v = ((1 - fr) * (1 - fc) * img[r0c, c0w] + (1 - fr) * fc * img[r0c, c1w]
         + fr * (1 - fc) * img[r0c + 1, c0w] + fr * fc * img[r0c + 1, c1w])
    return np.where(ok, v, np.nan)


def trace_direction(c_m, R_m, c_n, R_n, u, fan, r_probe=2.0):
    """Unit image direction (rows, cols) in which a point at r_probe along each anchor ray u moves from look m to n."""
    X = c_m + r_probe * u
    a, _ = fan.model().pixel_of((X - c_m) @ R_m)
    b, _ = fan.model().pixel_of((X - c_n) @ R_n)
    d = b - a
    W = fan.width
    d[:, 1] = (d[:, 1] + W / 2) % W - W / 2
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def anchors(looks, m, fan, max_n=20000, frac=0.15, seed=0, sigma=0.0):
    """Pixels of the reference look with a strong gradient along their trace direction: the top `frac` of those with a
    valid value and |elevation| < 70 deg, at most max_n (random). Returns pixel (row, col), ray u (world), the index."""
    s, c, R, img, idx = looks[m]
    H, W = img.shape
    rows, cols = np.mgrid[0:H, 0:W]
    lat = (0.5 - (rows + 0.5) / H) * np.pi
    ok = np.isfinite(img) & (idx >= 0) & (np.abs(lat) < np.radians(70))
    bl = _gauss(img, sigma)
    gr = (np.vstack([bl[1:], bl[-1:]]) - np.vstack([bl[:1], bl[:-1]])) / 2
    gc = (np.roll(bl, -1, 1) - np.roll(bl, 1, 1)) / 2
    pr, pc = rows[ok], cols[ok]
    rc = np.stack([pr + 0.5, pc + 0.5], -1)
    lon = (pc + 0.5) / W * 2 * np.pi - np.pi
    la = lat[pr, pc]
    vloc = np.stack([np.sin(lon) * np.cos(la), np.sin(la), -np.cos(lon) * np.cos(la)], -1)
    u = vloc @ R.T
    n = looks[min(m + 1, len(looks) - 1)] if m + 1 < len(looks) else looks[m - 1]
    dirn = trace_direction(c, R, n[1], n[2], u, fan)
    gdir = np.abs(gr[pr, pc] * dirn[:, 0] + gc[pr, pc] * dirn[:, 1])
    good = np.isfinite(gdir)
    thr = np.quantile(gdir[good], 1 - frac)
    sel = np.flatnonzero(good & (gdir >= thr))
    rng = np.random.default_rng(seed)
    if len(sel) > max_n:
        sel = np.sort(rng.choice(sel, size=max_n, replace=False))
    return np.stack([pr[sel], pc[sel]], -1), u[sel], idx[pr[sel], pc[sel]], gdir[sel]


def sweep(looks, m, px, u, fan, z_lo=1 / 25.0, z_hi=1 / 0.25, nz=256, tau=0.08, coarse_every=10, keep=3,
          fine=9, mirror=False, sigma=0.0, halves=False, nearest=False):
    """B on the anchors (px, u) of look m. mirror: pair image k with the pose of look 2m - k (the control).
    Returns r_est, confidence, cost curves' best values."""
    blurred = [lk[3] if sigma < 0 else _gauss(lk[3], sigma) for lk in looks]     # sigma < 0: the raw images
    K = len(looks)
    ref = blurred[m][px[:, 0], px[:, 1]]
    c_m = looks[m][1]
    pose = (lambda k: looks[2 * m - k] if 0 <= 2 * m - k < K else None) if mirror else (lambda k: looks[k])
    zs = np.linspace(z_lo, z_hi, nz)

    def cost(zv, ks, sl):
        """zv (n, Z) inverse ranges for the anchors in slice sl; cost (n, Z) over looks ks."""
        N, Z = zv.shape
        acc = np.zeros((N, Z))
        cnt = np.zeros((N, Z))
        rf = ref[sl]
        X = c_m[None, None, :] + u[sl][:, None, :] / zv[:, :, None]
        for k in ks:
            if k == m:
                continue
            pk = pose(k)
            if pk is None:
                continue
            rc, _ = fan.model().pixel_of(((X - pk[1]) @ pk[2]).reshape(-1, 3))
            v = (_sample_nearest if nearest else _sample)(blurred[k], rc).reshape(N, Z)
            d = np.minimum((v - rf[:, None]) ** 2, tau ** 2)
            f = np.isfinite(d)
            acc += np.where(f, d, 0.0)
            cnt += f
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt >= max(2, 0.3 * len(ks)), acc / cnt, np.inf)

    if halves:
        base = cost

        def cost(zv, ks, sl):
            """Occlusion-robust: the best of the whole window, the looks before the reference and the looks after."""
            b = [k for k in ks if k < m]
            a = [k for k in ks if k > m]
            parts = [base(zv, ks, sl)] + [base(zv, h, sl) for h in (b, a) if len(h) >= 3]
            return np.minimum.reduce(parts)

    N = len(px)
    coarse_ks = list(range(0, K, coarse_every)) + [K - 1]
    C = np.full((N, nz), np.inf)
    for a in range(0, N, 2000):
        sl = slice(a, min(a + 2000, N))
        C[sl] = cost(np.broadcast_to(zs, (sl.stop - a, nz)).copy(), coarse_ks, sl)
    # the best `keep` separated minima, refined on all looks
    order = np.argsort(C, axis=1)
    cands = np.full((N, keep), -1)
    for i in range(N):
        got = []
        for j in order[i]:
            if not np.isfinite(C[i, j]):
                break
            if all(abs(j - g) > 3 for g in got):
                got.append(j)
            if len(got) == keep:
                break
        cands[i, :len(got)] = got
    dz = zs[1] - zs[0]
    best_z = np.full(N, np.nan)
    best_c = np.full(N, np.inf)
    second_c = np.full(N, np.inf)
    off = np.linspace(-2 * dz, 2 * dz, fine)
    allks = list(range(K))
    for q in range(keep):
        ok = cands[:, q] >= 0
        zq = np.where(ok, zs[np.maximum(cands[:, q], 0)], zs[0])
        zv = np.clip(zq[:, None] + off[None, :], z_lo, z_hi)
        Cf = np.full(zv.shape, np.inf)
        for a in range(0, N, 1000):
            sl = slice(a, min(a + 1000, N))
            Cf[sl] = cost(zv[sl], allks, sl)
        Cf[~ok] = np.inf
        j = np.argmin(Cf, axis=1)
        cj = Cf[np.arange(N), j]
        # parabolic refinement on the fine grid
        jl, jr = np.clip(j - 1, 0, fine - 1), np.clip(j + 1, 0, fine - 1)
        cl, cr = Cf[np.arange(N), jl], Cf[np.arange(N), jr]
        den = cl - 2 * cj + cr
        with np.errstate(invalid="ignore", divide="ignore"):
            sh = np.where((j > 0) & (j < fine - 1) & np.isfinite(den) & (den > 0), 0.5 * (cl - cr) / den, 0.0)
        zr = zv[np.arange(N), j] + sh * (off[1] - off[0])
        better = cj < best_c
        second_c = np.where(better, best_c, np.minimum(second_c, cj))
        best_c = np.where(better, cj, best_c)
        best_z = np.where(better, zr, best_z)
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = np.where(np.isfinite(second_c), second_c / np.maximum(best_c, 1e-9), np.inf)
    return 1.0 / best_z, conf, best_c


def evaluate_b(kind, n=101, ds=0.01, R=1.6, max_n=20000, mirror=False, looks_world=None):
    looks, P, g, fan = looks_world or r0_run(kind, n=n, ds=ds, R=R)
    m = len(looks) // 2
    px, u, pidx, gd = anchors(looks, m, fan, max_n=max_n)
    r_true = np.linalg.norm(P[pidx] - looks[m][1], axis=-1)
    r_est, conf, best = sweep(looks, m, px, u, fan, mirror=mirror)
    rel = np.abs(r_est / r_true - 1)
    fwd = -looks[m][2][:, 2]
    psi = np.degrees(np.arccos(np.clip(u @ fwd, -1, 1)))
    return {"rel": rel, "conf": conf, "best": best, "psi": psi, "r_true": r_true, "px": px,
            "median": float(np.nanmedian(rel)), "within10": float(np.mean(rel <= 0.10)), "n": int(len(rel))}, \
        (looks, P, g, fan)


def depth_edge(looks, m, P, px, ratio=1.10, rad=2):
    """Anchors within rad pixels of a depth discontinuity in the reference look (max/min range > ratio): a
    diagnostic for reports, using the truth, never used by the matcher."""
    c, idx = looks[m][1], looks[m][4]
    rng = np.where(idx >= 0, np.linalg.norm(P[np.maximum(idx, 0)] - c, axis=-1), np.nan)
    H, W = rng.shape
    mx, mn = np.zeros(len(px)), np.full(len(px), np.inf)
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            v = rng[np.clip(px[:, 0] + dr, 0, H - 1), (px[:, 1] + dc) % W]
            mx, mn = np.fmax(mx, v), np.fmin(mn, v)
    return (mx / mn) > ratio


SECTORS = {"0-15": (0, 15), "15-60": (15, 60), "60-120": (60, 120), "120-165": (120, 165), "165-180": (165, 180.1)}


def selftest_b(kinds=("line", "arc"), max_n=20000):
    """K1 (line) and K2 (arc), as declared on #14, with the mirrored control; plus diagnostics by sector and by
    depth edge (truth-based, for the report only)."""
    res, num = [], {}
    for kname, kind in (("K1", "line"), ("K2", "arc")):
        if kind not in kinds:
            continue
        ev, world = evaluate_b(kind, max_n=max_n)
        ctl, _ = evaluate_b(kind, max_n=max_n, mirror=True, looks_world=world)
        looks, P, g, fan = world
        m = len(looks) // 2
        edge = depth_edge(looks, m, P, ev["px"])
        rel, psi = ev["rel"], ev["psi"]
        num[kname] = {"anchors": ev["n"], "median_rel": ev["median"], "within10": ev["within10"],
                      "control_median_rel": ctl["median"],
                      "by_sector_within10": {k: float(np.mean(rel[(psi >= a) & (psi < b)] <= 0.1))
                                             for k, (a, b) in SECTORS.items() if ((psi >= a) & (psi < b)).any()},
                      "depth_edge_fraction": float(edge.mean()),
                      "within10_at_depth_edges": float(np.mean(rel[edge] <= 0.1)),
                      "within10_off_depth_edges": float(np.mean(rel[~edge] <= 0.1))}
        res.append((f"{kname} B on noise-free R0 ({kind}): median |error| <= 2 % and >= 90 % within +/-10 %",
                    ev["median"] <= 0.02 and ev["within10"] >= 0.90))
        res.append((f"{kname} control: the path traversed backwards (median >= 20 %)", ctl["median"] >= 0.20))
    return res, num
