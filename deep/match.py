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
          fine=9, mirror=False, sigma=0.0, halves=False, nearest=False, zmin=None, zmax=None, patch=0, trunc=1.0):
    """B on the anchors (px, u) of look m. mirror: pair image k with the pose of look 2m - k (the control).
    zmin, zmax: optional per-anchor bounds on inverse range (method F's ordering). patch > 0: B2 (#17), the cost per
    look is 1 - ZNCC of the (2 patch + 1)^2 patch translated to the predicted point against the reference patch,
    truncated at trunc. Returns r_est, confidence, cost curves' best values."""
    blurred = [lk[3] if sigma < 0 else _gauss(lk[3], sigma) for lk in looks]     # sigma < 0: the raw images
    K = len(looks)
    ref = blurred[m][px[:, 0], px[:, 1]]
    c_m = looks[m][1]
    pose = (lambda k: looks[2 * m - k] if 0 <= 2 * m - k < K else None) if mirror else (lambda k: looks[k])
    zs = np.linspace(z_lo, z_hi, nz)
    if patch > 0:
        o = np.arange(-patch, patch + 1, dtype=float)
        OFF = np.stack(np.meshgrid(o, o, indexing="ij"), -1).reshape(-1, 2)          # (P, 2) row, col offsets
        a = _sample(blurred[m], (px[:, None, :] + OFF[None, :, :]).reshape(-1, 2)).reshape(len(px), -1)
        a0 = a - a.mean(1, keepdims=True)
        na = np.sqrt((a0 ** 2).sum(1))
        a_ok = (a0 ** 2).mean(1) >= 1e-8
        a0 = np.where(a_ok[:, None], a0, np.nan)

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
            if patch > 0:
                b = _sample(blurred[k], (rc[:, None, :] + OFF[None, :, :]).reshape(-1, 2)).reshape(N, Z, -1)
                b0 = b - b.mean(2, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    zncc = (a0[sl][:, None, :] * b0).sum(2) / (na[sl][:, None] * np.sqrt((b0 ** 2).sum(2)))
                zncc = np.where((b0 ** 2).mean(2) >= 1e-8, zncc, np.nan)
                d = np.minimum(1.0 - zncc, trunc)
            else:
                v = (_sample_nearest if nearest else _sample)(blurred[k], rc).reshape(N, Z)
                d = np.minimum((v - rf[:, None]) ** 2, tau ** 2)
            f = np.isfinite(d)
            acc += np.where(f, d, 0.0)
            cnt += f
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(cnt >= max(2, 0.3 * len(ks)), acc / cnt, np.inf)
        if zmin is not None:
            out[zv < zmin[sl][:, None]] = np.inf
        if zmax is not None:
            out[zv > zmax[sl][:, None]] = np.inf
        return out

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
    ch = 2000 if patch == 0 else 200
    for a in range(0, N, ch):
        sl = slice(a, min(a + ch, N))
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
        for a in range(0, N, ch // 2):
            sl = slice(a, min(a + ch // 2, N))
            Cf[sl] = cost(zv[sl], allks, sl)
        Cf[~ok] = np.inf
        j = np.argmin(Cf, axis=1)
        cj = Cf[np.arange(N), j]
        # parabolic refinement on the fine grid
        jl, jr = np.clip(j - 1, 0, fine - 1), np.clip(j + 1, 0, fine - 1)
        cl, cr = Cf[np.arange(N), jl], Cf[np.arange(N), jr]
        with np.errstate(invalid="ignore"):
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


TAU = 0.08
RELIABLE_COST = TAU ** 2 / 4          # declared on #14 before the run
RELIABLE_RATIO = 2.0


def reliable(best, conf):
    return (best <= RELIABLE_COST) & (conf >= RELIABLE_RATIO)


def selftest_k3(max_n=20000, seed=11):
    """K3 as declared on #14: on noise-free R0 (line, arc), the anchors B marks reliable are >= 95 % within +/-10 %,
    with >= 50 % kept; control: the same number marked at random stays below 95 %. Also the accuracy-coverage
    curve over the two thresholds, and the depth-edge anchors removed (truth-based, report only)."""
    res, num = [], {}
    rng = np.random.default_rng(seed)
    for kind in ("line", "arc"):
        ev, world = evaluate_b(kind, max_n=max_n)
        looks, P, g, fan = world
        m = len(looks) // 2
        edge = depth_edge(looks, m, P, ev["px"])
        rel, best, conf = ev["rel"], ev["best"], ev["conf"]
        rel_ok = rel <= 0.10
        mark = reliable(best, conf)
        kept = float(mark.mean())
        acc = float(rel_ok[mark].mean()) if mark.any() else 0.0
        rnd = np.zeros(len(rel), bool)
        rnd[rng.choice(len(rel), size=int(mark.sum()), replace=False)] = True
        acc_rnd = float(rel_ok[rnd].mean())
        curve = []
        for cc in (TAU ** 2 / 16, TAU ** 2 / 8, TAU ** 2 / 4, TAU ** 2 / 2, TAU ** 2):
            for rr in (1.0, 1.5, 2.0, 3.0, 5.0):
                mk = (best <= cc) & (conf >= rr)
                if mk.any():
                    curve.append({"cost_max": cc, "ratio_min": rr, "kept": float(mk.mean()),
                                  "within10": float(rel_ok[mk].mean())})
        num[kind] = {"anchors": int(len(rel)), "kept": kept, "within10_reliable": acc,
                     "within10_all": float(rel_ok.mean()), "control_within10_random": acc_rnd,
                     "median_rel_reliable": float(np.median(rel[mark])) if mark.any() else None,
                     "depth_edge_all": float(edge.mean()), "depth_edge_reliable": float(edge[mark].mean()) if mark.any() else None,
                     "within10_unreliable": float(rel_ok[~mark].mean()) if (~mark).any() else None,
                     "curve": curve}
        res.append((f"K3 {kind}: the anchors B marks reliable are >= 95 % within +/-10 %, >= 50 % kept",
                    acc >= 0.95 and kept >= 0.50))
        res.append((f"K3 {kind} control: as many anchors marked at random (< 95 %)", acc_rnd < 0.95))
    return res, num


# ------------------------------------------------------------------------------------------------------------ R1

def r1_run(run_dir):
    """R1 looks from a stage-5a render run, in the same form as r0_run: [(s, c, R, img, idx)], P, None, fan.
    img is luminance scaled by the reference look's 99th percentile; idx (reference look only) is the flat raster
    index into P, the reference look's true positions (oracle access, for evaluation only)."""
    _, L = orc.load(run_dir)
    m = len(L) // 2
    rec_m = orc.record(L[m][0])
    H, W = (int(x) for x in rec_m.raster_shape)
    hit = np.isfinite(rec_m.distance_m)
    scale = float(np.quantile((rec_m.value[hit].astype(np.float64) @ LUM), 0.99))
    P = rec_m.position_w.astype(np.float64)
    fan = sn.SlitFan(width=W)
    looks = []
    for k, (path, c, R, s) in enumerate(L):
        rec = rec_m if k == m else orc.record(path)
        g = rec.value.astype(np.float64) @ LUM / scale
        g[~np.isfinite(rec.distance_m)] = np.nan
        idx = np.where(hit, np.arange(H * W), -1).reshape(H, W) if k == m else None
        looks.append((float(s), np.asarray(c, float), np.asarray(R, float), g.reshape(H, W).astype(np.float32), idx))
    return looks, P, None, fan


# --------------------------------------------------------------------------------------- the differential (A)

def _grad(img):
    """Central differences (rows clamp, columns wrap): d/drow, d/dcol."""
    gr = (np.vstack([img[1:], img[-1:]]) - np.vstack([img[:1], img[:-1]])) / 2
    gc = (np.roll(img, -1, 1) - np.roll(img, 1, 1)) / 2
    return gr, gc


def _wrapd(d, W):
    d = d.copy()
    d[..., 1] = (d[..., 1] + W / 2) % W - W / 2
    return d


def _pose_fn(looks, m, mirror):
    K = len(looks)
    if mirror:
        return lambda k: looks[2 * m - k] if 0 <= 2 * m - k < K else None
    return lambda k: looks[k]


A2_SCHEDULE = (1, 2, 4, 8, 16, 32, 50)      # declared on #14 before the code


def lagrange(looks, m, px, u, fan, z0=0.5, schedule=A2_SCHEDULE, iters=3, tau=TAU, z_lo=1 / 25.0, z_hi=1 / 0.25,
             mirror=False, grow=True):
    """A2 on the anchors (px, u) of look m: inverse range by Gauss-Newton on brightness constancy against the anchor's
    own value, over a window of looks that grows with the schedule (grow=False: the whole last window from the first
    step, the K4b control). Returns r_est, the final mean truncated squared residual over the last window (B's cost
    scale), the last step's relative change of z, and whether z ended at a bound."""
    K, N, W = len(looks), len(px), fan.width
    imgs = [_blur(lk[3]) for lk in looks]
    grads = [_grad(im) for im in imgs]
    ref = imgs[m][px[:, 0], px[:, 1]]
    c_m = looks[m][1]
    pose = _pose_fn(looks, m, mirror)
    model = fan.model()
    z = np.broadcast_to(np.asarray(z0, float), (N,)).copy()
    last = np.zeros(N)
    hs = list(schedule) if grow else [schedule[-1]] * len(schedule)

    def project(k, zz):
        pk = pose(k)
        rc, _ = model.pixel_of((c_m + u / zz[:, None] - pk[1]) @ pk[2])
        return rc

    def window(h):
        return [k for k in range(m - h, m + h + 1) if 0 <= k < K and k != m and pose(k) is not None]

    for h in hs:
        ks = window(h)
        for _ in range(iters):
            gJe, gJJ = np.zeros(N), np.zeros(N)
            dmax = np.zeros(N)
            for k in ks:
                rc = project(k, z)
                eps = 1e-4
                drc = _wrapd(project(k, z * (1 + eps)) - rc, W) / (eps * z)[:, None]
                v = _sample(imgs[k], rc)
                J = _sample(grads[k][0], rc) * drc[:, 0] + _sample(grads[k][1], rc) * drc[:, 1]
                e = v - ref
                w = np.isfinite(e) & np.isfinite(J) & (np.abs(e) <= tau)
                gJe += np.where(w, J * e, 0.0)
                gJJ += np.where(w, J * J, 0.0)
                dmax = np.maximum(dmax, np.linalg.norm(drc, axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                dz = np.where(gJJ > 0, -gJe / gJJ, 0.0)
                cap = np.where(dmax > 0, 1.0 / dmax, np.inf)          # at most 1 px in the farthest look
            dz = np.clip(dz, -cap, cap)
            zn = np.clip(z + dz, z_lo, z_hi)
            last = np.abs(zn - z) / z
            z = zn
    acc, cnt = np.zeros(N), np.zeros(N)
    for k in window(hs[-1]):
        e = _sample(imgs[k], project(k, z)) - ref
        f = np.isfinite(e)
        acc += np.where(f, np.minimum(e * e, tau ** 2), 0.0)
        cnt += f
    with np.errstate(invalid="ignore", divide="ignore"):
        cost = np.where(cnt > 0, acc / cnt, np.inf)
    at_bound = (z <= z_lo * (1 + 1e-9)) | (z >= z_hi * (1 - 1e-9))
    return 1.0 / z, cost, last, at_bound


def reliable_a2(cost, last, at_bound):
    """A2's mark, declared on #14: cost <= tau^2/4 (B's threshold), last step < 0.1 %, not at a bound."""
    return (cost <= RELIABLE_COST) & (last < 1e-3) & ~at_bound


def eulerian(looks, m, fan, px, sigma=1.0, half=2, mirror=False):
    """A1: inverse range per pixel of look m from looks m-1, m, m+1: brightness constancy I_t + grad I . f = 0 with
    the flow f = f_rot + z f_v known from the poses (first order in the baseline), least squares over a
    (2 half + 1)^2 window. Returned at the pixels px: r_est (inf where z <= 0 or undetermined)."""
    H, W = looks[m][3].shape
    pose = _pose_fn(looks, m, mirror)
    I = {k: _gauss(looks[k][3], sigma) for k in (m - 1, m, m + 1)}
    It = (I[m + 1] - I[m - 1]) / 2
    gr, gc = _grad(I[m])
    rows, cols = np.mgrid[0:H, 0:W]
    lat = (0.5 - (rows + 0.5) / H) * np.pi
    lon = (cols + 0.5) / W * 2 * np.pi - np.pi
    vloc = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1).reshape(-1, 3)
    c_m, R_m = looks[m][1], looks[m][2]
    uw = vloc @ R_m.T
    model = fan.model()

    def flow(z):
        out = []
        for k in (m - 1, m + 1):
            pk = pose(k)
            X = (c_m + uw / z - pk[1]) if z > 0 else uw          # z = 0: the direction alone (the rotation part)
            rc, _ = model.pixel_of((X @ pk[2]) if z > 0 else (uw @ pk[2]))
            out.append(rc)
        return _wrapd(out[1] - out[0], W) / 2

    eps = 0.01
    f0 = flow(0.0)
    fv = (flow(eps) - f0) / eps
    f0 = f0.reshape(H, W, 2)
    fv = fv.reshape(H, W, 2)
    a = gr * fv[..., 0] + gc * fv[..., 1]
    b = It + gr * f0[..., 0] + gc * f0[..., 1]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)

    def box(x):
        s = sum(np.roll(x, o, 1) for o in range(-half, half + 1))
        sp = np.pad(s, ((half, half), (0, 0)), mode="edge")
        return sum(sp[half + o:half + o + H] for o in range(-half, half + 1))

    Saa, Sab = box(a * a), box(a * b)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(Saa > 0, -Sab / Saa, np.nan)
    zp = z[px[:, 0], px[:, 1]]
    return np.where(np.isfinite(zp) & (zp > 0), 1.0 / zp, np.inf)


def _relerr(r_est, r_true):
    rel = np.abs(r_est / r_true - 1)
    return np.where(np.isfinite(rel), rel, 1e3)


def _summ(rel, mask=None):
    if mask is not None:
        rel = rel[mask]
    if len(rel) == 0:
        return {"n": 0}
    return {"n": int(len(rel)), "median": float(np.median(rel)), "within10": float(np.mean(rel <= 0.10))}


def selftest_a(rung="r0", kinds=("line", "arc"), max_n=20000, seed=11, runs=None):
    """The checks declared on #14 for A and for R1. rung r0: K4 (A2), K4b (the hypothesis, with the no-schedule
    control), K5 (A1). rung r1: K6 (B and A2 with their marks), K7 (A1). Every control as declared."""
    import time
    res, num = [], {}
    rnd = np.random.default_rng(seed)
    runs = runs or {"line": "runs/r1-line", "arc": "runs/r1-arc"}
    for kind in kinds:
        world = r0_run(kind) if rung == "r0" else r1_run(runs[kind])
        looks, P, g, fan = world
        m = len(looks) // 2
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_true = np.linalg.norm(P[pidx] - looks[m][1], axis=-1)
        fwd = -looks[m][2][:, 2]
        psi = np.degrees(np.arccos(np.clip(u @ fwd, -1, 1)))
        edge = depth_edge(looks, m, P, px)
        n = len(px)
        out = {"anchors": n, "depth_edge_fraction": float(edge.mean())}
        t = time.time(); rb, confb, bestb = sweep(looks, m, px, u, fan); tb = time.time() - t
        t = time.time(); ra, costa, lasta, bnda = lagrange(looks, m, px, u, fan); ta = time.time() - t
        t = time.time(); r1e = eulerian(looks, m, fan, px); t1 = time.time() - t
        eb, ea, e1 = _relerr(rb, r_true), _relerr(ra, r_true), _relerr(r1e, r_true)
        mb, ma = reliable(bestb, confb), reliable_a2(costa, lasta, bnda)
        band = (psi >= 15) & (psi <= 165)
        out["anchors_per_second"] = {"B": n / tb, "A2": n / ta, "A1": n / t1}
        for name, e in (("B", eb), ("A2", ea), ("A1", e1)):
            out[name] = {"all": _summ(e), "band_15_165": _summ(e, band), "off_edges": _summ(e, ~edge),
                         "at_edges": _summ(e, edge),
                         "by_sector_within10": {k: float(np.mean(e[(psi >= a) & (psi < b)] <= 0.1))
                                                for k, (a, b) in SECTORS.items() if ((psi >= a) & (psi < b)).any()}}
        for name, e, mk in (("B", eb, mb), ("A2", ea, ma)):
            out[name]["reliable"] = {"kept": float(mk.mean()), **_summ(e, mk),
                                     "depth_edge_share": float(edge[mk].mean()) if mk.any() else None}
        okb, oka = eb <= 0.1, ea <= 0.1
        out["A2_vs_B"] = {"both_right": float(np.mean(okb & oka)), "B_only": float(np.mean(okb & ~oka)),
                          "A2_only": float(np.mean(~okb & oka)), "both_wrong": float(np.mean(~okb & ~oka))}
        e1m = _relerr(eulerian(looks, m, fan, px, mirror=True), r_true)
        ctl_a1 = _summ(e1m, band)
        out["A1"]["control_mirrored_band"] = ctl_a1
        a1 = out["A1"]["band_15_165"]
        if rung == "r0":
            ram, _, _, _ = lagrange(looks, m, px, u, fan, mirror=True)
            ctl_a2 = _summ(_relerr(ram, r_true))
            out["A2"]["control_mirrored"] = ctl_a2
            hyp = ~mb & ~edge
            rag, _, _, _ = lagrange(looks, m, px, u, fan, grow=False)
            eag = _relerr(rag, r_true)
            out["K4b"] = {"anchors": int(hyp.sum()), "B": _summ(eb, hyp), "A2": _summ(ea, hyp),
                          "A2_no_schedule": _summ(eag, hyp), "A2_no_schedule_all": _summ(eag)}
            a2 = out["A2"]["all"]
            res.append((f"K4 A2 on noise-free R0 ({kind}): median |error| <= 2 % and >= 90 % within +/-10 %",
                        a2["median"] <= 0.02 and a2["within10"] >= 0.90))
            res.append((f"K4 {kind} control: the mirrored path (median >= 20 %)", ctl_a2["median"] >= 0.20))
            k4b = out["K4b"]
            res.append((f"K4b {kind}: on anchors off depth edges that B marks unreliable, A2 >= 90 % within +/-10 %",
                        k4b["A2"]["within10"] >= 0.90))
            res.append((f"K4b {kind} control: A2 without the schedule (< 90 % on the same anchors)",
                        k4b["A2_no_schedule"]["within10"] < 0.90))
            res.append((f"K5 A1 on noise-free R0 ({kind}), 15-165 deg: median <= 10 % and >= 50 % within +/-10 %",
                        a1["median"] <= 0.10 and a1["within10"] >= 0.50))
            res.append((f"K5 {kind} control: mirrored (median >= 20 %)", ctl_a1["median"] >= 0.20))
        else:
            rbm, _, _ = sweep(looks, m, px, u, fan, mirror=True)
            ram, _, _, _ = lagrange(looks, m, px, u, fan, mirror=True)
            out["B"]["control_mirrored"] = _summ(_relerr(rbm, r_true))
            out["A2"]["control_mirrored"] = _summ(_relerr(ram, r_true))
            for name, e, mk in (("B", eb, mb), ("A2", ea, ma)):
                rs = np.zeros(n, bool)
                rs[rnd.choice(n, size=int(mk.sum()), replace=False)] = True
                out[name]["control_random_marking"] = _summ(e, rs)
                rl = out[name]["reliable"]
                res.append((f"K6 {name} on R1 ({kind}): reliable anchors >= 95 % within +/-10 %, >= 50 % kept",
                            rl["n"] > 0 and rl["within10"] >= 0.95 and rl["kept"] >= 0.50))
                res.append((f"K6 {name} {kind} control: the mirrored path (median >= 20 %)",
                            out[name]["control_mirrored"]["median"] >= 0.20))
                res.append((f"K6 {name} {kind} control: as many marked at random (< 95 %)",
                            out[name]["control_random_marking"]["within10"] < 0.95))
            res.append((f"K7 A1 on R1 ({kind}), 15-165 deg: median <= 10 % and >= 50 % within +/-10 %",
                        a1["median"] <= 0.10 and a1["within10"] >= 0.50))
            res.append((f"K7 {kind} control: mirrored (median >= 20 %)", ctl_a1["median"] >= 0.20))
        num[kind] = out
    return res, num


# ------------------------------------------------------------------------------------- depth edges: method E

def pixel_rays(px, R, H, W):
    """World rays of pixel centres px (N, 2) for a look with head rotation R."""
    lat = (0.5 - (px[:, 0] + 0.5) / H) * np.pi
    lon = (px[:, 1] + 0.5) / W * 2 * np.pi - np.pi
    v = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1)
    return v @ R.T


EDGE_RATIO, FLANK_PX = 1.10, 2.0                 # declared on #14 before the code


def two_depth(looks, m, px, u, fan, r_b, rule="coverage", seed=5):
    """Method E on B's estimates r_b: flanks at +/-2 px along the trace, ranged by B; a detected edge when their ranges
    differ by > 10 %; the anchor takes the near flank's range when the near surface covers >= half its pixel
    (alpha from raw values), the far one otherwise; B stands without an edge or without contrast (< tau).
    rule: 'coverage' (E), 'flipped' or 'coin' (the controls). Returns r, and the decision per anchor
    (0 kept, 1 near, 2 far)."""
    H, W = looks[m][3].shape
    s, c, R = looks[m][:3]
    k = m + 1 if m + 1 < len(looks) else m - 1
    d = trace_direction(c, R, looks[k][1], looks[k][2], u, fan)
    raw = looks[m][3]
    fl = []
    for sg in (+1, -1):
        q = np.rint(px + sg * FLANK_PX * d).astype(np.int64)
        q[:, 0] = np.clip(q[:, 0], 0, H - 1)
        q[:, 1] %= W
        fl.append(q)
    rf = []
    for q in fl:
        r, _, _ = sweep(looks, m, q, pixel_rays(q, R, H, W), fan)
        rf.append(r)
    rp, rm = rf
    lo, hi = np.fmin(rp, rm), np.fmax(rp, rm)
    with np.errstate(invalid="ignore", divide="ignore"):
        edge = np.isfinite(lo) & np.isfinite(hi) & (hi / lo > EDGE_RATIO)
    near_is_p = rp <= rm
    Ip, Im = raw[fl[0][:, 0], fl[0][:, 1]], raw[fl[1][:, 0], fl[1][:, 1]]
    In, If = np.where(near_is_p, Ip, Im), np.where(near_is_p, Im, Ip)
    Ia = raw[px[:, 0], px[:, 1]]
    con = np.abs(In - If) >= TAU
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = (Ia - If) / (In - If)
    go_near = alpha >= 0.5
    if rule == "flipped":
        go_near = ~go_near
    elif rule == "coin":
        go_near = np.random.default_rng(seed).random(len(px)) < 0.5
    act = edge & con & np.isfinite(alpha)
    dec = np.where(act, np.where(go_near, 1, 2), 0)
    r = np.where(dec == 1, lo, np.where(dec == 2, hi, r_b))
    return r, dec, edge


def selftest_k8(rungs=("r1", "r0"), kinds=("line", "arc"), max_n=20000, runs=None):
    """K8 as declared on #14 (R1): E at true depth edges >= 70 % within +/-10 %, off edges at most 1 point below B,
    overall >= 90 %; controls: the flipped rule and a coin, each must miss 70 % at edges. R0: reported only."""
    runs = runs or {"line": "runs/r1-line", "arc": "runs/r1-arc"}
    res, num = [], {}
    for rung in rungs:
        for kind in kinds:
            looks, P, _, fan = r0_run(kind) if rung == "r0" else r1_run(runs[kind])
            m = len(looks) // 2
            px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
            r_true = np.linalg.norm(P[pidx] - looks[m][1], axis=-1)
            tedge = depth_edge(looks, m, P, px)
            r_b, _, _ = sweep(looks, m, px, u, fan)
            eb = _relerr(r_b, r_true)
            out = {"anchors": int(len(px)), "true_edges": float(tedge.mean()),
                   "B": {"all": _summ(eb), "at_edges": _summ(eb, tedge), "off_edges": _summ(eb, ~tedge)}}
            for rule in ("coverage", "flipped", "coin"):
                r_e, dec, det = two_depth(looks, m, px, u, fan, r_b, rule=rule)
                ee = _relerr(r_e, r_true)
                o = {"all": _summ(ee), "at_edges": _summ(ee, tedge), "off_edges": _summ(ee, ~tedge)}
                if rule == "coverage":
                    o["detection"] = {"detected": float(det.mean()),
                                      "precision": float(tedge[det].mean()) if det.any() else None,
                                      "recall": float(det[tedge].mean()) if tedge.any() else None,
                                      "decided_near": float((dec == 1).mean()), "decided_far": float((dec == 2).mean())}
                    o["by_decision"] = {nm: _summ(ee, dec == v) for nm, v in (("kept", 0), ("near", 1), ("far", 2))}
                    o["B_by_decision"] = {nm: _summ(eb, dec == v) for nm, v in (("kept", 0), ("near", 1), ("far", 2))}
                out["E" if rule == "coverage" else f"E_{rule}"] = o
            num[f"{rung}_{kind}"] = out
            if rung == "r1":
                E, B = out["E"], out["B"]
                res.append((f"K8 E on R1 ({kind}): >= 70 % within +/-10 % at true depth edges, off edges >= B - 1 pt, "
                            f"overall >= 90 %",
                            E["at_edges"]["within10"] >= 0.70
                            and E["off_edges"]["within10"] >= B["off_edges"]["within10"] - 0.01
                            and E["all"]["within10"] >= 0.90))
                for rule in ("flipped", "coin"):
                    res.append((f"K8 {kind} control: the {rule} rule (< 70 % at edges)",
                                out[f"E_{rule}"]["at_edges"]["within10"] < 0.70))
    return res, num


# ------------------------------------------------------------------------- depth edges: method F (ordering)

ORDER_MARGIN, ORDER_RATIO = 1.10, RELIABLE_RATIO          # declared on #14 before the code


def ordered(looks, m, px, u, fan, r_b, c_b, mode="far"):
    """Method F on B's (r_b, c_b): a hypothesis on each half-window (before, after the reference) by B's sweep,
    restricted to >= 10 % farther than r_b (mode 'far', the ordering), >= 10 % nearer ('near', the reversed control)
    or unrestricted ('any', the no-ordering control); the better half replaces r_b when its cost is below c_b / 2.
    Returns r, whether the half was taken, and which half (-1 before, +1 after, 0 none)."""
    z_b = 1.0 / r_b
    zmin = zmax = None
    if mode == "far":
        zmax = z_b / ORDER_MARGIN
    elif mode == "near":
        zmin = z_b * ORDER_MARGIN
    best_r = np.full(len(px), np.nan)
    best_c = np.full(len(px), np.inf)
    side = np.zeros(len(px), int)
    for sg, sub, mm in ((-1, looks[:m + 1], m), (+1, looks[m:], 0)):
        r, _, c = sweep(sub, mm, px, u, fan, zmin=zmin, zmax=zmax)
        better = c < best_c
        best_r, best_c, side = np.where(better, r, best_r), np.where(better, c, best_c), np.where(better, sg, side)
    take = np.isfinite(best_c) & (best_c * ORDER_RATIO < c_b)
    return np.where(take, best_r, r_b), take, np.where(take, side, 0)


def selftest_k9(kinds=("line", "arc"), max_n=20000, runs=None):
    """K9 as declared on #14 (R1): F at true depth edges >= 70 % within +/-10 %, off edges at most 1 point below B,
    overall >= 90 %; controls: the ordering reversed and no ordering, each must miss 70 % at edges."""
    runs = runs or {"line": "runs/r1-line", "arc": "runs/r1-arc"}
    res, num = [], {}
    for kind in kinds:
        looks, P, _, fan = r1_run(runs[kind])
        m = len(looks) // 2
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_true = np.linalg.norm(P[pidx] - looks[m][1], axis=-1)
        tedge = depth_edge(looks, m, P, px)
        r_b, _, c_b = sweep(looks, m, px, u, fan)
        eb = _relerr(r_b, r_true)
        out = {"anchors": int(len(px)), "true_edges": float(tedge.mean()),
               "B": {"all": _summ(eb), "at_edges": _summ(eb, tedge), "off_edges": _summ(eb, ~tedge)}}
        for mode, name in (("far", "F"), ("near", "F_reversed"), ("any", "F_no_ordering")):
            r, take, side = ordered(looks, m, px, u, fan, r_b, c_b, mode=mode)
            e = _relerr(r, r_true)
            o = {"all": _summ(e), "at_edges": _summ(e, tedge), "off_edges": _summ(e, ~tedge),
                 "taken": float(take.mean()), "taken_at_edges": float(take[tedge].mean()),
                 "taken_off_edges": float(take[~tedge].mean()),
                 "taken_before": float((side == -1).mean()), "taken_after": float((side == 1).mean()),
                 "by_decision": {"kept": _summ(e, ~take), "taken": _summ(e, take)},
                 "B_on_taken": _summ(eb, take)}
            out[name] = o
        num[kind] = out
        F, B = out["F"], out["B"]
        res.append((f"K9 F on R1 ({kind}): >= 70 % within +/-10 % at true depth edges, off edges >= B - 1 pt, "
                    f"overall >= 90 %",
                    F["at_edges"]["within10"] >= 0.70 and F["off_edges"]["within10"] >= B["off_edges"]["within10"] - 0.01
                    and F["all"]["within10"] >= 0.90))
        for name in ("F_reversed", "F_no_ordering"):
            res.append((f"K9 {kind} control: {name} (< 70 % at edges)", out[name]["at_edges"]["within10"] < 0.70))
    return res, num


# ----------------------------------------------------------------- (b): the surface metric for contour anchors

def neighbourhood_ranges(looks, m, P, px, rad=1):
    """True ranges of the (2 rad + 1)^2 pixels around each anchor in the reference look: (N, (2 rad + 1)^2)."""
    c, idx = looks[m][1], looks[m][4]
    rng = np.where(idx >= 0, np.linalg.norm(P[np.maximum(idx, 0)] - c, axis=-1), np.nan)
    H, W = rng.shape
    return np.stack([rng[np.clip(px[:, 0] + dr, 0, H - 1), (px[:, 1] + dc) % W]
                     for dr in range(-rad, rad + 1) for dc in range(-rad, rad + 1)], -1)


def surface_right(r_est, nb, tol=0.10):
    """Declared on #14: right if within +/-tol of the true range at any pixel of the 3x3 neighbourhood."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(r_est[:, None] / nb - 1)
    return np.nanmin(np.where(np.isfinite(rel), rel, np.inf), axis=1) <= tol


def selftest_k10(kinds=("line", "arc"), max_n=20000, runs=None, seed=11):
    """K10 as declared on #14 (R1, B): surface-right >= 70 % at true depth edges, off edges at most 1 point below the
    old metric, overall >= 90 %; controls: the blend, off by 25 %, the mirrored path (each must miss 70 % at edges)."""
    runs = runs or {"line": "runs/r1-line", "arc": "runs/r1-arc"}
    res, num = [], {}
    rnd = np.random.default_rng(seed)
    for kind in kinds:
        looks, P, _, fan = r1_run(runs[kind])
        m = len(looks) // 2
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_true = np.linalg.norm(P[pidx] - looks[m][1], axis=-1)
        tedge = depth_edge(looks, m, P, px)
        nb = neighbourhood_ranges(looks, m, P, px)
        r_b, conf, best = sweep(looks, m, px, u, fan)
        r_a, cost, last, bnd = lagrange(looks, m, px, u, fan)
        r_bm, _, _ = sweep(looks, m, px, u, fan, mirror=True)
        zlo, zhi = 1 / np.nanmax(nb, 1), 1 / np.nanmin(nb, 1)
        ests = {"B": r_b, "A2": r_a, "control_blend": 2 / (zlo + zhi), "control_off25": 1.25 * r_true,
                "control_mirrored": r_bm}
        out = {"anchors": int(len(px)), "true_edges": float(tedge.mean())}
        for name, r in ests.items():
            old = _relerr(r, r_true) <= 0.10
            new = surface_right(r, nb)
            o = {"old": {"all": float(old.mean()), "at_edges": float(old[tedge].mean()),
                         "off_edges": float(old[~tedge].mean())},
                 "surface": {"all": float(new.mean()), "at_edges": float(new[tedge].mean()),
                             "off_edges": float(new[~tedge].mean())}}
            if name in ("B", "A2"):
                fail_e = tedge & ~old
                with np.errstate(invalid="ignore", divide="ignore"):
                    between = (r > np.nanmin(nb, 1) * 1.1) & (r < np.nanmax(nb, 1) / 1.1)
                o["old_failures_at_edges"] = {"n": int(fail_e.sum()), "converted": float(new[fail_e].mean()),
                                              "still_wrong_between": float((~new & between)[fail_e].mean()),
                                              "still_wrong_off": float((~new & ~between)[fail_e].mean())}
                mk = reliable(best, conf) if name == "B" else reliable_a2(cost, last, bnd)
                o["reliable"] = {"kept": float(mk.mean()), "old": float(old[mk].mean()), "surface": float(new[mk].mean())}
            out[name] = o
        num[kind] = out
        B = out["B"]
        res.append((f"K10 B on R1 ({kind}), surface metric: >= 70 % at true depth edges, off edges >= old - 1 pt, "
                    f"overall >= 90 %",
                    B["surface"]["at_edges"] >= 0.70 and B["surface"]["off_edges"] >= B["old"]["off_edges"] - 0.01
                    and B["surface"]["all"] >= 0.90))
        for name in ("control_blend", "control_off25", "control_mirrored"):
            res.append((f"K10 {kind} control: {name[8:]} (< 70 % at edges, surface metric)",
                        out[name]["surface"]["at_edges"] < 0.70))
    return res, num


# ------------------------------------------------------------------- (a): method G, two layers per contour anchor

G_FLANK_PX = 3.0                                        # declared on #14 before the code


def two_layer(looks, m, px, u, fan, r_b, rule="coverage", halves=True):
    """Method G on B's estimates r_b. The flanks at +/-3 px along the trace are the candidate layers: each ranged as
    the near layer on the whole window, and as the far layer on each half-window (halves=False: the whole window, the
    'no time' control), the far fit restricted to >= 10 % farther than the opposite flank's near range. A far layer
    exists when the best far fit is reliable by B's rule; the anchor then takes the near or far range by its pixel's
    coverage (rule 'flipped': the control). Returns r, decision (0 kept, 1 near, 2 far), far-layer found."""
    H, W = looks[m][3].shape
    s, c, R = looks[m][:3]
    k = m + 1 if m + 1 < len(looks) else m - 1
    d = trace_direction(c, R, looks[k][1], looks[k][2], u, fan)
    raw = looks[m][3]
    q, uq, rn = {}, {}, {}
    for sg in (+1, -1):
        qq = np.rint(px + sg * G_FLANK_PX * d).astype(np.int64)
        qq[:, 0] = np.clip(qq[:, 0], 0, H - 1)
        qq[:, 1] %= W
        q[sg], uq[sg] = qq, pixel_rays(qq, R, H, W)
        rn[sg], _, _ = sweep(looks, m, qq, uq[sg], fan)
    N = len(px)
    best_c = np.full(N, np.inf)
    best_r, best_conf = np.full(N, np.nan), np.zeros(N)
    far_side = np.zeros(N, int)
    windows = ((looks[:m + 1], m), (looks[m:], 0)) if halves else ((looks, m),)
    for sg in (+1, -1):
        zmax = (1.0 / rn[-sg]) / ORDER_MARGIN
        for sub, mm in windows:
            r, conf, cst = sweep(sub, mm, q[sg], uq[sg], fan, zmax=zmax)
            better = cst < best_c
            best_c = np.where(better, cst, best_c)
            best_r = np.where(better, r, best_r)
            best_conf = np.where(better, conf, best_conf)
            far_side = np.where(better, sg, far_side)
    found = np.isfinite(best_c) & reliable(best_c, best_conf)
    near_side = -far_side
    r_near = np.where(near_side == 1, rn[1], rn[-1])
    I_n = np.where(near_side == 1, raw[q[1][:, 0], q[1][:, 1]], raw[q[-1][:, 0], q[-1][:, 1]])
    I_f = np.where(far_side == 1, raw[q[1][:, 0], q[1][:, 1]], raw[q[-1][:, 0], q[-1][:, 1]])
    I_a = raw[px[:, 0], px[:, 1]]
    con = np.abs(I_n - I_f) >= TAU
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = (I_a - I_f) / (I_n - I_f)
    go_near = alpha >= 0.5
    if rule == "flipped":
        go_near = ~go_near
    act = found & con & np.isfinite(alpha) & np.isfinite(r_near)
    dec = np.where(act, np.where(go_near, 1, 2), 0)
    return np.where(dec == 1, r_near, np.where(dec == 2, best_r, r_b)), dec, found


def selftest_k11(kinds=("line", "arc"), max_n=20000, runs=None):
    """K11 as declared on #14 (R1, old metric): G at true depth edges >= 70 % within +/-10 %, off edges at most
    1 point below B, overall >= 90 %; controls: the flipped coverage rule and 'no time' (each must miss 70 %)."""
    runs = runs or {"line": "runs/r1-line", "arc": "runs/r1-arc"}
    res, num = [], {}
    for kind in kinds:
        looks, P, _, fan = r1_run(runs[kind])
        m = len(looks) // 2
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_true = np.linalg.norm(P[pidx] - looks[m][1], axis=-1)
        tedge = depth_edge(looks, m, P, px)
        nb = neighbourhood_ranges(looks, m, P, px)
        r_b, _, _ = sweep(looks, m, px, u, fan)

        def score(r):
            old, new = _relerr(r, r_true) <= 0.10, surface_right(r, nb)
            return {"old": {"all": float(old.mean()), "at_edges": float(old[tedge].mean()),
                            "off_edges": float(old[~tedge].mean())},
                    "surface": {"all": float(new.mean()), "at_edges": float(new[tedge].mean()),
                                "off_edges": float(new[~tedge].mean())}}, old
        out = {"anchors": int(len(px)), "true_edges": float(tedge.mean())}
        out["B"], old_b = score(r_b)
        for name, kw in (("G", {}), ("G_flipped", {"rule": "flipped"}), ("G_no_time", {"halves": False})):
            r, dec, found = two_layer(looks, m, px, u, fan, r_b, **kw)
            o, old = score(r)
            o["far_layer_found"] = {"all": float(found.mean()), "at_edges": float(found[tedge].mean()),
                                    "off_edges": float(found[~tedge].mean())}
            o["decided"] = {"near": float((dec == 1).mean()), "far": float((dec == 2).mean())}
            o["old_by_decision"] = {nm: (float(old[dec == v].mean()) if (dec == v).any() else None)
                                    for nm, v in (("kept", 0), ("near", 1), ("far", 2))}
            o["B_old_by_decision"] = {nm: (float(old_b[dec == v].mean()) if (dec == v).any() else None)
                                      for nm, v in (("kept", 0), ("near", 1), ("far", 2))}
            out[name] = o
        num[kind] = out
        G, B = out["G"]["old"], out["B"]["old"]
        res.append((f"K11 G on R1 ({kind}), old metric: >= 70 % at true depth edges, off edges >= B - 1 pt, "
                    f"overall >= 90 %",
                    G["at_edges"] >= 0.70 and G["off_edges"] >= B["off_edges"] - 0.01 and G["all"] >= 0.90))
        for name in ("G_flipped", "G_no_time"):
            res.append((f"K11 {kind} control: {name} (< 70 % at edges, old metric)",
                        out[name]["old"]["at_edges"] < 0.70))
    return res, num


# ------------------------------------------------------------- anchors without texture: method H (seed and verify)

H_CELL, H_CANDIDATES, H_GRID = 10, 8, 2               # declared on #14 before the code


def _trace_cost(blurred, looks, m, ref, u, zv, fan, tau=TAU, chunk=4000):
    """B's cost (truncated mean squared difference to the reference value over all looks but m) at inverse ranges
    zv (N, Z) for the pixels with values ref and rays u."""
    N, Z = zv.shape
    out = np.full((N, Z), np.inf)
    c_m = looks[m][1]
    model = fan.model()
    for a in range(0, N, chunk):
        sl = slice(a, min(a + chunk, N))
        n = sl.stop - a
        X = c_m[None, None, :] + u[sl][:, None, :] / zv[sl][:, :, None]
        acc, cnt = np.zeros((n, Z)), np.zeros((n, Z))
        for k in range(len(looks)):
            if k == m:
                continue
            rc, _ = model.pixel_of(((X - looks[k][1]) @ looks[k][2]).reshape(-1, 3))
            d = np.minimum((_sample(blurred[k], rc).reshape(n, Z) - ref[sl][:, None]) ** 2, tau ** 2)
            f = np.isfinite(d)
            acc += np.where(f, d, 0.0)
            cnt += f
        with np.errstate(invalid="ignore", divide="ignore"):
            out[sl] = np.where(cnt >= max(2, 0.3 * (len(looks) - 1)), acc / cnt, np.inf)
    return out


def seed_candidates(seed_px, seed_u, tgt_px, tgt_u, W, H, cell=H_CELL, k=H_CANDIDATES):
    """For each target, the indices of the k nearest seeds by angle among those in the 3x3 block of cells around it
    (-1 where fewer)."""
    nr, nc = int(np.ceil(H / cell)), int(np.ceil(W / cell))
    scell = (seed_px[:, 0] // cell) * nc + seed_px[:, 1] // cell
    order = np.argsort(scell, kind="stable")
    starts = np.searchsorted(scell[order], np.arange(nr * nc + 1))
    tcell_r, tcell_c = tgt_px[:, 0] // cell, tgt_px[:, 1] // cell
    tcell = tcell_r * nc + tcell_c
    out = np.full((len(tgt_px), k), -1)
    for cid in np.unique(tcell):
        r0, c0 = divmod(int(cid), nc)
        block = [order[starts[rr * nc + cc]:starts[rr * nc + cc + 1]]
                 for rr in range(max(r0 - 1, 0), min(r0 + 2, nr)) for cc in ((c0 - 1) % nc, c0, (c0 + 1) % nc)]
        S = np.concatenate(block) if block else np.zeros(0, int)
        if len(S) == 0:
            continue
        T = np.flatnonzero(tcell == cid)
        dots = tgt_u[T] @ seed_u[S].T
        kk = min(k, len(S))
        top = np.argsort(-dots, axis=1)[:, :kk]
        out[T, :kk] = S[top]
    return out


def seed_fill(looks, m, fan, seed_px, seed_u, seed_r, tgt_px, tgt_u, z_lo=1 / 25.0, z_hi=1 / 0.25, nz=256, fine=9):
    """Method H: each target takes the best-verified of its seeds' ranges, refined like B's last stage. Returns r
    (nan where no candidate), its cost, and the nearest seed's range (the unverified reference)."""
    H, W = looks[m][3].shape
    blurred = [_blur(lk[3]) for lk in looks]
    ref = blurred[m][tgt_px[:, 0], tgt_px[:, 1]]
    cand = seed_candidates(seed_px, seed_u, tgt_px, tgt_u, W, H)
    has = cand >= 0
    zc = np.where(has, 1.0 / seed_r[np.maximum(cand, 0)], np.nan)
    C = _trace_cost(blurred, looks, m, ref, tgt_u, np.where(has, zc, 1.0), fan)
    C[~has] = np.inf
    j = np.argmin(C, axis=1)
    z0 = zc[np.arange(len(j)), j]
    ok = has.any(axis=1) & np.isfinite(C[np.arange(len(j)), j])
    dz = (z_hi - z_lo) / (nz - 1)
    off = np.linspace(-2 * dz, 2 * dz, fine)
    zv = np.clip(np.where(ok, z0, 1.0)[:, None] + off[None, :], z_lo, z_hi)
    Cf = _trace_cost(blurred, looks, m, ref, tgt_u, zv, fan)
    i = np.clip(np.argmin(Cf, axis=1), 1, fine - 2)
    rows = np.arange(len(i))
    cl, cj, cr = Cf[rows, i - 1], Cf[rows, i], Cf[rows, i + 1]
    with np.errstate(invalid="ignore"):
        den = cl - 2 * cj + cr
    step = off[1] - off[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where((den > 0) & np.isfinite(den), 0.5 * (cl - cr) / den, 0.0)
    z = np.clip(zv[rows, i] + np.clip(sh, -1, 1) * step, z_lo, z_hi)
    r = np.where(ok, 1.0 / z, np.nan)
    nearest = np.where(has[:, 0], seed_r[np.maximum(cand[:, 0], 0)], np.nan)
    return r, np.where(ok, cj, np.inf), nearest


def selftest_k12(kinds=("line", "arc"), max_n=20000, runs=None, seed=11):
    """K12 as declared on #14 (R1, surface metric): H covers >= 95 % of the targets and is >= 85 % surface-right
    (K12a); in the least-textured third it beats B alone by >= 10 points (K12b); control: shuffled seeds (< 85 %)."""
    runs = runs or {"line": "runs/r1-line", "arc": "runs/r1-arc"}
    res, num = [], {}
    rnd = np.random.default_rng(seed)
    for kind in kinds:
        looks, P, _, fan = r1_run(runs[kind])
        m = len(looks) // 2
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_b, conf, best = sweep(looks, m, px, u, fan)
        mk = reliable(best, conf)
        spx, su, sr = px[mk], u[mk], r_b[mk]
        apx, au, aidx, ag = anchors(looks, m, fan, max_n=10 ** 9, frac=1.0)
        grid = (apx[:, 0] % H_GRID == 0) & (apx[:, 1] % H_GRID == 0)
        tpx, tu, tidx, tg = apx[grid], au[grid], aidx[grid], ag[grid]
        r_true = np.linalg.norm(P[tidx] - looks[m][1], axis=-1)
        nb = neighbourhood_ranges(looks, m, P, tpx)
        tedge = depth_edge(looks, m, P, tpx)
        W = looks[m][3].shape[1]
        is_seed = np.isin(tpx[:, 0] * W + tpx[:, 1], spx[:, 0] * W + spx[:, 1])
        seed_at = {int(a * W + b): float(r) for (a, b), r in zip(spx, sr)}
        import time
        t = time.time()
        r_h, c_h, r_near = seed_fill(looks, m, fan, spx, su, sr, tpx, tu)
        t_h = time.time() - t
        r_h = np.where(is_seed, [seed_at.get(int(a * W + b), np.nan) for a, b in tpx], r_h)
        r_sh, _, _ = seed_fill(looks, m, fan, spx, su, rnd.permutation(sr), tpx, tu)
        t = time.time()
        r_bt, _, _ = sweep(looks, m, tpx, tu, fan)
        t_b = time.time() - t
        lo3, hi3 = np.quantile(tg[np.isfinite(tg)], [1 / 3, 2 / 3])
        thirds = {"least": tg < lo3, "middle": (tg >= lo3) & (tg < hi3), "most": tg >= hi3}
        fwd = -looks[m][2][:, 2]
        psi = np.degrees(np.arccos(np.clip(tu @ fwd, -1, 1)))

        def score(r):
            cov = np.isfinite(r)
            old = cov & (_relerr(np.where(cov, r, 1.0), r_true) <= 0.10)
            new = cov & surface_right(np.where(cov, r, np.nan), nb)
            o = {"coverage": float(cov.mean()), "surface": float(new[cov].mean()) if cov.any() else None,
                 "old": float(old[cov].mean()) if cov.any() else None}
            for nm, msk in thirds.items():
                o[f"surface_{nm}_texture"] = float(new[cov & msk].mean())
            o["surface_at_edges"] = float(new[cov & tedge].mean())
            o["surface_off_edges"] = float(new[cov & ~tedge].mean())
            o["surface_by_sector"] = {k: float(new[cov & (psi >= a) & (psi < b)].mean())
                                      for k, (a, b) in SECTORS.items() if (cov & (psi >= a) & (psi < b)).any()}
            return o
        out = {"seeds": int(mk.sum()), "targets": int(len(tpx)), "targets_that_are_seeds": int(is_seed.sum()),
               "true_edges": float(tedge.mean()), "seconds": {"H": t_h, "B_alone": t_b},
               "H": score(r_h), "control_shuffled_seeds": score(r_sh), "B_alone": score(r_bt),
               "nearest_seed_unverified": score(r_near)}
        num[kind] = out
        Hs, Bs = out["H"], out["B_alone"]
        res.append((f"K12a H on R1 ({kind}): covers >= 95 % of the targets, >= 85 % surface-right",
                    Hs["coverage"] >= 0.95 and Hs["surface"] >= 0.85))
        res.append((f"K12b {kind}: least-textured third, H >= B alone + 10 points (surface-right)",
                    Hs["surface_least_texture"] >= Bs["surface_least_texture"] + 0.10))
        res.append((f"K12 {kind} control: shuffled seeds (< 85 % surface-right)",
                    out["control_shuffled_seeds"]["surface"] < 0.85))
    return res, num


# -------------------------------------------------------- method H2: grow in rounds, verify where it discriminates

H2_ROUNDS = 20                                          # declared on #14 before the code


def _decide(blurred, looks, m, fan, ref, tu, zc, mode, z_lo=1 / 25.0, z_hi=1 / 0.25, nz=256, fine=9):
    """One round's decision for targets with candidate inverse ranges zc (N, k; nan where none), nearest first.
    mode 'gated' (H2): the lowest-cost candidate, refined, where the candidates' cost spread is >= tau^2/4, else the
    nearest; 'always': the lowest-cost candidate, refined (H); 'never': the nearest."""
    has = np.isfinite(zc)
    nearest = zc[:, 0]
    if mode == "never":
        return 1.0 / nearest
    C = _trace_cost(blurred, looks, m, ref, tu, np.where(has, zc, 1.0), fan)
    C[~has] = np.inf
    j = np.argmin(C, axis=1)
    rows = np.arange(len(j))
    cmin = C[rows, j]
    cmax = np.max(np.where(np.isfinite(C), C, -np.inf), axis=1)
    use = np.isfinite(cmin) if mode == "always" else (np.isfinite(cmin) & (cmax - cmin >= RELIABLE_COST))
    dz = (z_hi - z_lo) / (nz - 1)
    off = np.linspace(-2 * dz, 2 * dz, fine)
    z = nearest.copy()
    if use.any():
        ii = np.flatnonzero(use)
        zv = np.clip(zc[ii, j[ii]][:, None] + off[None, :], z_lo, z_hi)
        Cf = _trace_cost(blurred, looks, m, ref[ii], tu[ii], zv, fan)
        i = np.clip(np.argmin(Cf, axis=1), 1, fine - 2)
        rr = np.arange(len(i))
        cl, cj, cr = Cf[rr, i - 1], Cf[rr, i], Cf[rr, i + 1]
        with np.errstate(invalid="ignore"):
            den = cl - 2 * cj + cr
        with np.errstate(invalid="ignore", divide="ignore"):
            sh = np.where((den > 0) & np.isfinite(den), 0.5 * (cl - cr) / den, 0.0)
        z[ii] = np.clip(zv[rr, i] + np.clip(sh, -1, 1) * (off[1] - off[0]), z_lo, z_hi)
    return 1.0 / z


def seed_grow(looks, m, fan, seed_px, seed_u, seed_r, tgt_px, tgt_u, mode="gated", rounds=H2_ROUNDS):
    """Method H2: H's candidate search in rounds, the targets filled in a round seeding the next. Returns r (nan where
    never reached) and the round that filled each target (0: never)."""
    H, W = looks[m][3].shape
    blurred = [_blur(lk[3]) for lk in looks]
    ref_all = blurred[m][tgt_px[:, 0], tgt_px[:, 1]]
    r = np.full(len(tgt_px), np.nan)
    rnd = np.zeros(len(tgt_px), int)
    spx, su, sr = seed_px, seed_u, seed_r
    for k in range(1, rounds + 1):
        todo = np.flatnonzero(~np.isfinite(r))
        if len(todo) == 0 or len(spx) == 0:
            break
        cand = seed_candidates(spx, su, tgt_px[todo], tgt_u[todo], W, H)
        reach = cand[:, 0] >= 0
        if not reach.any():
            break
        ii, cc = todo[reach], cand[reach]
        zc = np.where(cc >= 0, 1.0 / sr[np.maximum(cc, 0)], np.nan)
        r[ii] = _decide(blurred, looks, m, fan, ref_all[ii], tgt_u[ii], zc, mode)
        rnd[ii] = k
        spx, su, sr = tgt_px[ii], tgt_u[ii], r[ii]
    return r, rnd


def selftest_k13(kinds=("line", "arc"), max_n=20000, runs=None, seed=11):
    """K13 as declared on #14 (R1, surface metric): H2 covers >= 95 % and is >= 85 % surface-right (K13a); least-
    textured third >= B alone + 10 points (K13b); control: shuffled seeds grown the same way (< 85 %)."""
    runs = runs or {"line": "runs/r1-line", "arc": "runs/r1-arc"}
    res, num = [], {}
    rng = np.random.default_rng(seed)
    for kind in kinds:
        looks, P, _, fan = r1_run(runs[kind])
        m = len(looks) // 2
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_b, conf, best = sweep(looks, m, px, u, fan)
        mk = reliable(best, conf)
        spx, su, sr = px[mk], u[mk], r_b[mk]
        apx, au, aidx, ag = anchors(looks, m, fan, max_n=10 ** 9, frac=1.0)
        grid = (apx[:, 0] % H_GRID == 0) & (apx[:, 1] % H_GRID == 0)
        tpx, tu, tidx, tg = apx[grid], au[grid], aidx[grid], ag[grid]
        W = looks[m][3].shape[1]
        r_true = np.linalg.norm(P[tidx] - looks[m][1], axis=-1)
        nb = neighbourhood_ranges(looks, m, P, tpx)
        tedge = depth_edge(looks, m, P, tpx)
        key_t = tpx[:, 0] * W + tpx[:, 1]
        seed_at = dict(zip((spx[:, 0] * W + spx[:, 1]).tolist(), sr.tolist()))
        is_seed = np.isin(key_t, spx[:, 0] * W + spx[:, 1])
        seed_r_t = np.array([seed_at.get(int(k), np.nan) for k in key_t])
        r_bt, _, _ = sweep(looks, m, tpx, tu, fan)
        lo3, hi3 = np.quantile(tg[np.isfinite(tg)], [1 / 3, 2 / 3])
        thirds = {"least": tg < lo3, "middle": (tg >= lo3) & (tg < hi3), "most": tg >= hi3}
        fwd = -looks[m][2][:, 2]
        psi = np.degrees(np.arccos(np.clip(tu @ fwd, -1, 1)))

        def score(r, rnd=None):
            cov = np.isfinite(r)
            new = cov & surface_right(np.where(cov, r, np.nan), nb)
            old = cov & (_relerr(np.where(cov, r, 1.0), r_true) <= 0.10)
            o = {"coverage": float(cov.mean()), "surface": float(new[cov].mean()), "old": float(old[cov].mean())}
            for nm, msk in thirds.items():
                o[f"surface_{nm}_texture"] = float(new[cov & msk].mean())
            o["surface_at_edges"] = float(new[cov & tedge].mean())
            o["surface_off_edges"] = float(new[cov & ~tedge].mean())
            o["surface_by_sector"] = {k: float(new[cov & (psi >= a) & (psi < b)].mean())
                                      for k, (a, b) in SECTORS.items() if (cov & (psi >= a) & (psi < b)).any()}
            if rnd is not None:
                o["by_round"] = {int(k): {"filled": float((rnd == k).mean()), "surface": float(new[rnd == k].mean())}
                                 for k in np.unique(rnd[rnd > 0])}
            return o
        out = {"seeds": int(mk.sum()), "targets": int(len(tpx)), "targets_that_are_seeds": int(is_seed.sum())}
        import time
        for name, mode, seeds_r in (("H2", "gated", sr), ("H2_always_verify", "always", sr),
                                    ("H2_never_verify", "never", sr),
                                    ("control_shuffled_seeds", "gated", rng.permutation(sr))):
            t = time.time()
            r, rnd = seed_grow(looks, m, fan, spx, su, seeds_r, tpx, tu, mode=mode)
            r = np.where(is_seed, seed_r_t if name != "control_shuffled_seeds" else r, r)
            out[name] = score(r, rnd)
            out[name]["seconds"] = time.time() - t
        out["B_alone"] = score(r_bt)
        num[kind] = out
        H2, B = out["H2"], out["B_alone"]
        res.append((f"K13a H2 on R1 ({kind}): covers >= 95 % of the targets, >= 85 % surface-right",
                    H2["coverage"] >= 0.95 and H2["surface"] >= 0.85))
        res.append((f"K13b {kind}: least-textured third, H2 >= B alone + 10 points (surface-right)",
                    H2["surface_least_texture"] >= B["surface_least_texture"] + 0.10))
        res.append((f"K13 {kind} control: shuffled seeds (< 85 % surface-right)",
                    out["control_shuffled_seeds"]["surface"] < 0.85))
    return res, num


# ------------------------------------------------ the planar prior, method P (reconstruction, after the matcher)

P_MAX_PLANES, P_HYPOTHESES, P_INLIER, P_MIN_INLIERS, P_GROW = 8, 1000, 0.03, 500, 0.10   # declared on #14


def _plane_basis(n):
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def fit_planes(X, seed=0):
    """Sequential RANSAC: up to 8 planes n.x = d with >= 500 inliers within 3 cm, each refitted by least squares;
    each with its support box (inliers in plane coordinates, grown by 10 cm)."""
    rng = np.random.default_rng(seed)
    pts = X[np.all(np.isfinite(X), axis=1)]
    planes = []
    for _ in range(P_MAX_PLANES):
        if len(pts) < P_MIN_INLIERS:
            break
        best_n, best_d, best_cnt = None, None, -1
        for _ in range(P_HYPOTHESES):
            a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            d = n @ a
            cnt = int(np.sum(np.abs(pts @ n - d) <= P_INLIER))
            if cnt > best_cnt:
                best_n, best_d, best_cnt = n, d, cnt
        if best_cnt < P_MIN_INLIERS:
            break
        inl = np.abs(pts @ best_n - best_d) <= P_INLIER
        cen = pts[inl].mean(0)
        _, _, vt = np.linalg.svd(pts[inl] - cen, full_matrices=False)
        n = vt[-1]
        d = n @ cen
        inl = np.abs(pts @ n - d) <= P_INLIER
        if inl.sum() < P_MIN_INLIERS:
            break
        e1, e2 = _plane_basis(n)
        q = np.stack([pts[inl] @ e1, pts[inl] @ e2], -1)
        planes.append({"n": n, "d": float(d), "e1": e1, "e2": e2, "inliers": int(inl.sum()),
                       "lo": q.min(0) - P_GROW, "hi": q.max(0) + P_GROW})
        pts = pts[~inl]
    return planes


def plane_fill(c, U, planes):
    """For rays U from c: the nearest positive intersection inside a plane's support (nan where none)."""
    best = np.full(len(U), np.inf)
    for p in planes:
        nu = U @ p["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (p["d"] - p["n"] @ c) / nu
        Xh = c + U * t[:, None]
        q1, q2 = Xh @ p["e1"], Xh @ p["e2"]
        inside = (t > 0) & np.isfinite(t) & (q1 >= p["lo"][0]) & (q1 <= p["hi"][0]) \
            & (q2 >= p["lo"][1]) & (q2 <= p["hi"][1])
        best = np.where(inside & (t < best), t, best)
    return np.where(np.isfinite(best), best, np.nan)


def selftest_k14(kinds=("line", "arc"), max_n=20000, runs=None, seed=11):
    """K14 as declared on #14 (R1, surface metric): of H's empty targets P fills >= 90 % at >= 85 % surface-right
    (K14a); H + P cover >= 95 % at >= 85 % (K14b); control: planes from H's points with shuffled ranges (< 85 %)."""
    runs = runs or {"line": "runs/r1-line", "arc": "runs/r1-arc"}
    res, num = [], {}
    rng = np.random.default_rng(seed)
    for kind in kinds:
        looks, P, _, fan = r1_run(runs[kind])
        m = len(looks) // 2
        c = looks[m][1]
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_b, conf, best = sweep(looks, m, px, u, fan)
        mk = reliable(best, conf)
        spx, su, sr = px[mk], u[mk], r_b[mk]
        apx, au, aidx, ag = anchors(looks, m, fan, max_n=10 ** 9, frac=1.0)
        grid = (apx[:, 0] % H_GRID == 0) & (apx[:, 1] % H_GRID == 0)
        tpx, tu, tidx = apx[grid], au[grid], aidx[grid]
        W = looks[m][3].shape[1]
        r_h, _, _ = seed_fill(looks, m, fan, spx, su, sr, tpx, tu)
        key_t = tpx[:, 0] * W + tpx[:, 1]
        seed_at = dict(zip((spx[:, 0] * W + spx[:, 1]).tolist(), sr.tolist()))
        is_seed = np.isin(key_t, spx[:, 0] * W + spx[:, 1])
        r_h = np.where(is_seed, np.array([seed_at.get(int(k), np.nan) for k in key_t]), r_h)
        nb = neighbourhood_ranges(looks, m, P, tpx)
        r_true = np.linalg.norm(P[tidx] - c, axis=-1)
        zt = P[tidx][:, 2]
        bands = {"floor": zt < 0.05, "between": (zt >= 0.05) & (zt < 3.15), "ceiling": zt >= 3.15}
        empty = ~np.isfinite(r_h)
        fh = np.isfinite(r_h)
        X = c + tu[fh] * r_h[fh][:, None]
        out = {"targets": int(len(tpx)), "H_coverage": float(fh.mean())}

        def run_p(Xpts, name):
            planes = fit_planes(Xpts, seed=seed)
            r_p = np.full(len(tpx), np.nan)
            r_p[empty] = plane_fill(c, tu[empty], planes)
            filled = empty & np.isfinite(r_p)
            sr_ok = surface_right(np.where(filled, r_p, np.nan), nb)
            old_ok = _relerr(np.where(filled, r_p, 1.0), r_true) <= 0.10
            r_all = np.where(fh, r_h, r_p)
            cov = np.isfinite(r_all)
            all_ok = surface_right(np.where(cov, r_all, np.nan), nb)
            o = {"planes": [{"n": [round(float(v), 4) for v in p["n"]], "d": round(p["d"], 4),
                             "inliers": p["inliers"]} for p in planes],
                 "fills_of_empty": float(filled.sum() / max(empty.sum(), 1)),
                 "filled_surface": float(sr_ok[filled].mean()) if filled.any() else None,
                 "filled_old": float(old_ok[filled].mean()) if filled.any() else None,
                 "filled_by_band": {nm: {"filled": int((filled & b).sum()),
                                         "surface": float(sr_ok[filled & b].mean()) if (filled & b).any() else None}
                                    for nm, b in bands.items()},
                 "combined_coverage": float(cov.mean()), "combined_surface": float(all_ok[cov].mean())}
            out[name] = o
            return o
        Ph = run_p(X, "P")
        Xs = c + tu[fh] * rng.permutation(r_h[fh])[:, None]
        Pc = run_p(Xs, "control_shuffled")
        num[kind] = out
        res.append((f"K14a P on R1 ({kind}): fills >= 90 % of H's empty targets, >= 85 % surface-right",
                    Ph["fills_of_empty"] >= 0.90 and (Ph["filled_surface"] or 0) >= 0.85))
        res.append((f"K14b H + P on R1 ({kind}): coverage >= 95 %, >= 85 % surface-right",
                    Ph["combined_coverage"] >= 0.95 and Ph["combined_surface"] >= 0.85))
        res.append((f"K14 {kind} control: planes from shuffled ranges (< 85 % surface-right on what it fills)",
                    (Pc["filled_surface"] if Pc["filled_surface"] is not None else 0.0) < 0.85))
    return res, num


# ------------------------------------------------------------------------ R2: the matcher on another scene (#17)

def selftest_r2(runs, max_n=20000, seed=11):
    """The checks declared on #17 for the matcher on new renders (surface metric): K15, B's reliable anchors >= 95 %
    surface-right with >= 50 % kept (controls: random marking < 95 %, the mirrored path median >= 20 %); K16, H >= 85 %
    surface-right on what it fills (control: shuffled seeds < 85 %); K17, H beats B alone by >= 10 points in the
    least-textured third. Reported: coverage, B and A2 at and off true edges, and P's fill (not gated)."""
    res, num = [], {}
    rng = np.random.default_rng(seed)
    for kind, run_dir in runs.items():
        looks, P, _, fan = r1_run(run_dir)
        m = len(looks) // 2
        c = looks[m][1]
        W = looks[m][3].shape[1]
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_true = np.linalg.norm(P[pidx] - c, axis=-1)
        tedge = depth_edge(looks, m, P, px)
        nb = neighbourhood_ranges(looks, m, P, px)
        r_b, conf, best = sweep(looks, m, px, u, fan)
        r_bm, _, _ = sweep(looks, m, px, u, fan, mirror=True)
        r_a, _, _, _ = lagrange(looks, m, px, u, fan)
        mk = reliable(best, conf)
        rs = np.zeros(len(px), bool)
        rs[rng.choice(len(px), size=int(mk.sum()), replace=False)] = True
        srB, srA = surface_right(r_b, nb), surface_right(r_a, nb)
        out = {"looks": len(looks), "anchors": int(len(px)), "true_edges": float(tedge.mean())}
        for name, e_old, e_new in (("B", _relerr(r_b, r_true) <= 0.1, srB), ("A2", _relerr(r_a, r_true) <= 0.1, srA)):
            out[name] = {"old": {"all": float(e_old.mean()), "at_edges": float(e_old[tedge].mean()),
                                 "off_edges": float(e_old[~tedge].mean())},
                         "surface": {"all": float(e_new.mean()), "at_edges": float(e_new[tedge].mean()),
                                     "off_edges": float(e_new[~tedge].mean())}}
        out["B"]["reliable"] = {"kept": float(mk.mean()), "surface": float(srB[mk].mean()),
                                "old": float((_relerr(r_b, r_true) <= 0.1)[mk].mean())}
        out["B"]["control_random_marking_surface"] = float(srB[rs].mean())
        out["B"]["control_mirrored_median_old"] = float(np.median(_relerr(r_bm, r_true)))
        # H on the 2-px grid
        spx, su, sr = px[mk], u[mk], r_b[mk]
        apx, au, aidx, ag = anchors(looks, m, fan, max_n=10 ** 9, frac=1.0)
        grid = (apx[:, 0] % H_GRID == 0) & (apx[:, 1] % H_GRID == 0)
        tpx, tu, tidx, tg = apx[grid], au[grid], aidx[grid], ag[grid]
        tnb = neighbourhood_ranges(looks, m, P, tpx)
        key_t = tpx[:, 0] * W + tpx[:, 1]
        seed_at = dict(zip((spx[:, 0] * W + spx[:, 1]).tolist(), sr.tolist()))
        is_seed = np.isin(key_t, spx[:, 0] * W + spx[:, 1])
        seed_r_t = np.array([seed_at.get(int(k), np.nan) for k in key_t])
        r_h, _, _ = seed_fill(looks, m, fan, spx, su, sr, tpx, tu)
        r_h = np.where(is_seed, seed_r_t, r_h)
        r_sh, _, _ = seed_fill(looks, m, fan, spx, su, rng.permutation(sr), tpx, tu)
        r_bt, _, _ = sweep(looks, m, tpx, tu, fan)
        lo3 = np.quantile(tg[np.isfinite(tg)], 1 / 3)
        least = tg < lo3

        def sfrac(r, msk=None):
            cov = np.isfinite(r)
            ok = cov & surface_right(np.where(cov, r, np.nan), tnb)
            sel = cov if msk is None else cov & msk
            return float(ok[sel].mean()) if sel.any() else None
        out["H"] = {"targets": int(len(tpx)), "coverage": float(np.isfinite(r_h).mean()), "surface": sfrac(r_h),
                    "surface_least_texture": sfrac(r_h, least)}
        out["H"]["control_shuffled_seeds_surface"] = sfrac(r_sh)
        out["B_alone"] = {"surface": sfrac(r_bt), "surface_least_texture": sfrac(r_bt, least)}
        # P, reported only (Luiz's (ii), 2026-11-20: the prior is kept where it is supported, not gated)
        fh = np.isfinite(r_h)
        planes = fit_planes(c + tu[fh] * r_h[fh][:, None], seed=seed)
        r_p = np.full(len(tpx), np.nan)
        r_p[~fh] = plane_fill(c, tu[~fh], planes)
        out["P_reported"] = {"planes": len(planes), "fills_of_empty": float(np.isfinite(r_p[~fh]).mean()),
                             "surface_of_filled": sfrac(r_p)}
        num[kind] = out
        rel = out["B"]["reliable"]
        res.append((f"K15 B on {kind}: reliable anchors >= 95 % surface-right, >= 50 % kept",
                    rel["surface"] >= 0.95 and rel["kept"] >= 0.50))
        res.append((f"K15 {kind} control: as many marked at random (< 95 %)",
                    out["B"]["control_random_marking_surface"] < 0.95))
        res.append((f"K15 {kind} control: the mirrored path (median >= 20 %)",
                    out["B"]["control_mirrored_median_old"] >= 0.20))
        res.append((f"K16 H on {kind}: >= 85 % surface-right on what it fills", out["H"]["surface"] >= 0.85))
        res.append((f"K16 {kind} control: shuffled seeds (< 85 %)", out["H"]["control_shuffled_seeds_surface"] < 0.85))
        res.append((f"K17 {kind}: least-textured third, H >= B alone + 10 points",
                    out["H"]["surface_least_texture"] >= out["B_alone"]["surface_least_texture"] + 0.10))
    return res, num


# ---------------------------------------------------------------------------------- B2: the patch cost (#17)

B2_PATCH, B2_TRUNC = 2, 1.0                                   # declared on #17 before the code
B2_RELIABLE_COST = B2_TRUNC / 4


def sweep_b2(looks, m, px, u, fan, **kw):
    return sweep(looks, m, px, u, fan, patch=B2_PATCH, trunc=B2_TRUNC, **kw)


def reliable_b2(best, conf):
    """B2's mark, declared on #17: B's rule with the cost threshold restated, T/4 (mean ZNCC >= 0.75), ratio >= 2."""
    return (best <= B2_RELIABLE_COST) & (conf >= RELIABLE_RATIO)


def selftest_b2(runs, rung="r1", max_n=20000, seed=11):
    """K18 (rung r1) or K19 (rung r2) as declared on #17. K18: B2 surface-right overall >= B's - 1 point; K19: B2 off
    true depth edges >= 90 % surface-right; both: B2's reliable anchors >= 95 % surface-right with >= 50 % kept.
    Controls: the mirrored path (median old-metric error >= 20 %), as many marked at random (< 95 %)."""
    import time
    res, num = [], {}
    rng = np.random.default_rng(seed)
    for kind, run_dir in runs.items():
        looks, P, _, fan = r1_run(run_dir)
        m = len(looks) // 2
        c = looks[m][1]
        px, u, pidx, _ = anchors(looks, m, fan, max_n=max_n)
        r_true = np.linalg.norm(P[pidx] - c, axis=-1)
        tedge = depth_edge(looks, m, P, px)
        nb = neighbourhood_ranges(looks, m, P, px)
        t = time.time(); r_b, conf_b, best_b = sweep(looks, m, px, u, fan); t_b = time.time() - t
        t = time.time(); r2, conf2, best2 = sweep_b2(looks, m, px, u, fan); t_2 = time.time() - t
        r2m, _, _ = sweep_b2(looks, m, px, u, fan, mirror=True)
        mk = reliable_b2(best2, conf2)
        rs = np.zeros(len(px), bool)
        rs[rng.choice(len(px), size=int(mk.sum()), replace=False)] = True
        sb, s2 = surface_right(r_b, nb), surface_right(r2, nb)
        ob, o2 = _relerr(r_b, r_true) <= 0.1, _relerr(r2, r_true) <= 0.1

        def split(x):
            return {"all": float(x.mean()), "at_edges": float(x[tedge].mean()), "off_edges": float(x[~tedge].mean())}
        out = {"anchors": int(len(px)), "true_edges": float(tedge.mean()), "seconds": {"B": t_b, "B2": t_2},
               "B": {"surface": split(sb), "old": split(ob), "reliable_kept": float(reliable(best_b, conf_b).mean()),
                     "reliable_surface": float(sb[reliable(best_b, conf_b)].mean())},
               "B2": {"surface": split(s2), "old": split(o2), "reliable_kept": float(mk.mean()),
                      "reliable_surface": float(s2[mk].mean()) if mk.any() else None,
                      "control_mirrored_median_old": float(np.median(_relerr(r2m, r_true))),
                      "control_random_marking_surface": float(s2[rs].mean())},
               "B_vs_B2_surface": {"both": float(np.mean(sb & s2)), "B_only": float(np.mean(sb & ~s2)),
                                   "B2_only": float(np.mean(~sb & s2)), "neither": float(np.mean(~sb & ~s2))}}
        if rung == "r2":
            W = looks[m][3].shape[1]
            spx, su, sr = px[mk], u[mk], r2[mk]
            apx, au, aidx, ag = anchors(looks, m, fan, max_n=10 ** 9, frac=1.0)
            grid = (apx[:, 0] % H_GRID == 0) & (apx[:, 1] % H_GRID == 0)
            tpx, tu, tidx, tg = apx[grid], au[grid], aidx[grid], ag[grid]
            tnb = neighbourhood_ranges(looks, m, P, tpx)
            key_t = tpx[:, 0] * W + tpx[:, 1]
            seed_at = dict(zip((spx[:, 0] * W + spx[:, 1]).tolist(), sr.tolist()))
            is_seed = np.isin(key_t, spx[:, 0] * W + spx[:, 1])
            r_h, _, _ = seed_fill(looks, m, fan, spx, su, sr, tpx, tu)
            r_h = np.where(is_seed, np.array([seed_at.get(int(k), np.nan) for k in key_t]), r_h)
            r_bt, _, _ = sweep(looks, m, tpx, tu, fan)
            least = tg < np.quantile(tg[np.isfinite(tg)], 1 / 3)

            def sfrac(r, msk=None):
                cov = np.isfinite(r)
                ok = cov & surface_right(np.where(cov, r, np.nan), tnb)
                sel = cov if msk is None else cov & msk
                return float(ok[sel].mean()) if sel.any() else None
            out["H_from_B2_seeds"] = {"seeds": int(mk.sum()), "coverage": float(np.isfinite(r_h).mean()),
                                      "surface": sfrac(r_h), "surface_least_texture": sfrac(r_h, least),
                                      "B_alone_surface_least_texture": sfrac(r_bt, least)}
        num[kind] = out
        B2, B = out["B2"], out["B"]
        if rung == "r1":
            res.append((f"K18 B2 on {kind}: surface-right overall >= B's - 1 point",
                        B2["surface"]["all"] >= B["surface"]["all"] - 0.01))
            tag = "K18"
        else:
            res.append((f"K19 B2 on {kind}: off true depth edges >= 90 % surface-right",
                        B2["surface"]["off_edges"] >= 0.90))
            tag = "K19"
        res.append((f"{tag} B2 on {kind}: reliable anchors >= 95 % surface-right, >= 50 % kept",
                    B2["reliable_surface"] is not None and B2["reliable_surface"] >= 0.95 and B2["reliable_kept"] >= 0.50))
        res.append((f"{tag} {kind} control: the mirrored path (median >= 20 %)", B2["control_mirrored_median_old"] >= 0.20))
        res.append((f"{tag} {kind} control: as many marked at random (< 95 %)",
                    B2["control_random_marking_surface"] < 0.95))
    return res, num
