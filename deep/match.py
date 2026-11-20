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
