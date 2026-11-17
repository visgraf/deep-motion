"""The demo (#9): a tour combining 5b.2's two paths, the oracle's fusion growing look by look (DERIVED: a display).

The tour: the straight path's looks up to where it reaches the turn's start, then the turn's looks (#9). One pass over
the looks, a record in memory at a time: each cell's information H and gradient g accumulate (the noise drawn per
look exactly as `fusion.fuse` draws it), a cell shows once the Fisher gate passes (+/-10 % along the ray from every
look that has seen it, as `fusion.ray_interval`), and its point is the first-order estimate P + H^-1 g (check V1).

    +--------------------------------------+------------------+
    |  panorama: the current look,         |  top view        |
    |  usable cells it sees, by range      |                  |
    +-------------------+------------------+------------------+
    |  3-D cloud        |  progress: usable fraction / s      |
    +-------------------+-------------------------------------+

Two videos from the one pass: `range` (cells by range / height) and `error` (cells by the relative error along the
ray, clipped at 1 %). Colours: one blue ramp, light -> dark (the dataviz skill's sequential hue), as D1.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import fusion as fu
from . import oracle as orc
from . import sensor as sn

PREFIX = "[deep]"
W, H, FPS = 1920, 1080, 24
PANO = (0, 0, 1280, 640)
TOP = (1280, 0, 640, 640)
CLOUD = (0, 640, 800, 440)
PROG = (800, 640, 1120, 440)
BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ACCENT = (235, 104, 52)
INK, MUTED, RULE, PAPER, NIGHT = (34, 34, 34), (110, 110, 105), (200, 200, 195), (247, 247, 245), (24, 24, 26)
LIGHT = (235, 235, 230)
ERR_MAX = 0.01
CEILING_CLIP = 2.4          # the 3-D views leave out cells above this height (m): the ceiling would hide the room
HOLD_START, HOLD_END, ORBIT_S = 1.0, 1.5, 8.0
_STOPS = np.stack([np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) for h in BLUE])


def ramp(v) -> np.ndarray:
    """v in [0, 1] -> uint8 rgb along the blue ramp (0 light, 1 dark)."""
    v = np.asarray(v, float)
    if np.isnan(v).any():
        raise ValueError("ramp: NaN input")
    v = np.clip(v, 0, 1) * (len(_STOPS) - 1)
    i = np.minimum(np.floor(v).astype(int), len(_STOPS) - 2)
    f = (v - i)[..., None]
    return (_STOPS[i] * (1 - f) + _STOPS[i + 1] * f + 0.5).astype(np.uint8)


def range_colour(r):
    lr = np.log(np.clip(r, 0.25, 8.0))
    return ramp(1 - (lr - math.log(0.25)) / (math.log(8.0) - math.log(0.25)))       # near = dark


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _text(d, xy, s, size, fill=INK, anchor=None):
    d.text(xy, s, font=_font(size), fill=fill, anchor=anchor)


# ---------------------------------------------------------------------------------------------------------- the tour

def tour(straight_dirs, turn_dirs, join_s=2.0):
    """[(record path, c, R, s_tour, segment)] : the straight's looks with s < join_s, then the turn's.
    Segments: 0 straight, 1 line, 2 arc, 3 line (the turn's pieces)."""
    st = [lk for lk in fu.load_looks(straight_dirs) if lk[3] < join_s - 1e-6]
    tu = fu.load_looks(turn_dirs)
    ends = np.cumsum([L for _, L, _ in fu.PATHS["turn90"]])
    out = [(p, c, R, float(s), 0) for p, c, R, s in st]
    out += [(p, c, R, join_s + float(s), 1 + int(np.searchsorted(ends - 1e-9, s))) for p, c, R, s in tu]
    return out


SEG_NAMES = ("straight", "line", "arc", "line")


def check_continuity(T, step, straight_dirs, turn_dirs, join_s=2.0, shift=None):
    """V2: the tour is the planned L-curve sampled evenly. (a) arc length advances by `step` (+/- 1e-9); (b) every
    head equals the planned head of its path at its s (the paths and starts recorded in run.json): position and
    axes within 1e-9; (c) the straight at join_s reaches the turn's start (<= 1e-6 m, <= 1e-9 rad). shift: added to
    the turn's looks (the control). Restated on #9 before any Classroom run: the first wording asked consecutive
    heads to be one step apart in space too, which a step straddling a line-arc join is not (1.4e-5 m short at 10 cm,
    Chat sandbox)."""
    C = np.stack([t[1] for t in T]).copy()
    seg = np.array([t[4] for t in T])
    if shift is not None:
        C[seg > 0] += shift
    s = np.array([t[3] for t in T])
    ok_a = bool(np.all(np.abs(np.diff(s) - step) <= 1e-9))
    planned = []
    for dirs, off in ((straight_dirs, 0.0), (turn_dirs, join_s)):
        kind, start, _ = fu.run_meta(dirs)
        st_step = json.loads((Path(dirs[0]) / "run.json").read_text()).get("step_m", step)
        planned.append((kind, start, st_step, off))
    errs_p, errs_R = [], []
    for k, t in enumerate(T):
        kind, start, st_step, off = planned[0 if seg[k] == 0 else 1]
        sp = t[3] - off
        path = fu.composed_path(fu.PATHS[kind], start["p0"], np.radians(start["heading_deg"]), start["h"])
        acc = 0.0
        for i, (pc, (p0, phi0)) in enumerate(zip(path.pieces, path.starts)):
            if sp <= acc + pc.length + 1e-9 or i == len(path.pieces) - 1:
                c, R, _, _ = pc.pose(p0, phi0, path.h, min(sp - acc, pc.length))
                break
            acc += pc.length
        errs_p.append(float(np.linalg.norm(C[k] - c)))
        errs_R.append(float(np.abs(t[2] - R).max()))
    ok_b = max(errs_p) <= 1e-9 and max(errs_R) <= 1e-9
    kind_s, start_s, _, _ = planned[0]
    path_s = fu.composed_path(fu.PATHS[kind_s], start_s["p0"], np.radians(start_s["heading_deg"]), start_s["h"])
    cj, Rj, _, _ = path_s.pieces[0].pose(*path_s.starts[0], path_s.h, join_s)
    kt = int(np.argmax(seg > 0))
    gap = float(np.linalg.norm(C[kt] - cj))
    rot = float(np.abs(T[kt][2] - Rj).max())
    ok_c = gap <= 1e-6 and rot <= 1e-9
    return bool(ok_a and ok_b and ok_c), {"max_ds_err": float(np.max(np.abs(np.diff(s) - step))),
                                         "max_pos_err_m": max(errs_p), "max_axes_err": max(errs_R),
                                         "join_gap_m": gap, "join_axes_err": rot, "looks": len(T)}


# ------------------------------------------------------------------------------------------------ the oracle state

class State:
    """Per-cell H, g and the 'worst' gate, updated look by look."""

    def __init__(self, P, fan, sigma_px, seed):
        self.P, self.fan, self.sigma = P, fan, sigma_px
        self.N = len(P)
        self.H = np.zeros((self.N, 3, 3))
        self.g = np.zeros((self.N, 3))
        self.nseen = np.zeros(self.N, int)
        self.rng = np.random.default_rng(seed)
        self.seen_looks = []                                       # (c, vis column) of every look so far
        self.usable = np.zeros(self.N, bool)
        self.X = P.copy()

    def add(self, c, R, vis):
        sr = self.sigma * self.fan.res
        nl = self.rng.normal(size=self.N)                          # the order fuse() draws them in
        nt = self.rng.normal(size=self.N)
        _, _, gl, gt = fu._derivs(self.P, c, R)
        w = vis[:, None]
        self.H += w[:, :, None] * (gl[:, :, None] * gl[:, None, :] + gt[:, :, None] * gt[:, None, :])
        self.g += w * (gl * (nl * sr)[:, None] + gt * (nt * sr)[:, None])
        self.nseen += vis
        self.seen_looks.append((c, vis))

    def update(self):
        """The gate over every look so far (as fusion.ray_interval) and the first-order points."""
        m = self.nseen >= 2
        C = np.zeros_like(self.H)
        with np.errstate(invalid="ignore"):
            C[m] = np.linalg.pinv(self.H[m]) * (self.sigma * self.fan.res) ** 2
        worst = np.zeros(self.N)
        idx = np.flatnonzero(m)
        for c, vis in self.seen_looks:
            j = idx[vis[idx]]
            d = self.P[j] - c
            r = np.linalg.norm(d, axis=-1)
            u = d / r[:, None]
            s = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", u, C[j], u), 0))
            worst[j] = np.maximum(worst[j], orc.Z95 * s / r)
        self.usable = m & (worst <= orc.USABLE_REL)
        self.X = self.P.copy()
        self.X[m] = self.P[m] + np.linalg.solve(self.H[m] + 1e-30 * np.eye(3), self.g[m][..., None])[..., 0]
        return worst


def check_estimate(T, P, fan, sigma_px, seed, n=2000, Xfirst=None, usable=None):
    """V1: the first-order points equal fusion.fuse's fit under the same noise draw (median along-ray difference
    <= 0.05 sigma); control: the first-order points of another draw (>= 0.5 sigma)."""
    looks = [(t[0], t[1], t[2], t[3]) for t in T]
    rng = np.random.default_rng(seed + 1)
    pick = np.sort(rng.choice(np.flatnonzero(usable), size=min(n, int(usable.sum())), replace=False))
    vis = fu.visibility(looks, P, fan)                             # every cell, as the pass drew noise for every cell
    X, H, _ = fu.fuse(looks, P, fan, sigma_px=sigma_px, rng=np.random.default_rng(seed), vis=vis)
    other = State(P, fan, sigma_px, seed + 99)
    for k, lk in enumerate(looks):
        other.add(lk[1], lk[2], vis[:, k])
    other.update()
    first = np.argmax(vis[pick], 1)
    c0 = np.stack([lk[1] for lk in looks])[first]
    u = (P[pick] - c0) / np.linalg.norm(P[pick] - c0, axis=-1)[:, None]
    C = np.linalg.pinv(H[pick]) * (sigma_px * fan.res) ** 2
    s = np.sqrt(np.einsum("ni,nij,nj->n", u, C, u))
    d = np.abs(np.einsum("ni,ni->n", Xfirst[pick] - X[pick], u)) / s
    dc = np.abs(np.einsum("ni,ni->n", other.X[pick] - X[pick], u)) / s
    return {"V1_median_sigma": float(np.median(d)), "V1_p99_sigma": float(np.percentile(d, 99)),
            "V1_control_median_sigma": float(np.median(dc)), "V1_cells": int(len(pick))}


# ----------------------------------------------------------------------------------------------------- the panels

class Scene:
    def __init__(self, P, T, fan):
        self.P, self.T, self.fan = P, T, fan
        lo = np.percentile(P[:, :2], 0.5, axis=0) - 0.3
        hi = np.percentile(P[:, :2], 99.5, axis=0) + 0.3
        self.lo, self.hi = lo, hi
        self.C = np.stack([t[1] for t in T])
        self.s = np.array([t[3] for t in T])
        self.seg = np.array([t[4] for t in T])
        self.zlo, self.zhi = np.percentile(P[:, 2], [1, 99])
        self.centre = np.array([*(0.5 * (lo + hi)), 0.5 * (self.zlo + self.zhi)])
        self.extent = float(np.max(hi - lo))
        self.exposure = None


def pano_panel(sc, rec, m, st, colours):
    x0, y0, w, h = PANO
    Wf = sc.fan.width
    Hf = Wf // 2
    grid = np.zeros((Hf * Wf, 3), np.float32)
    hit = np.isfinite(rec.distance_m)
    grid[rec.raster_index[hit]] = rec.value[hit]
    if sc.exposure is None:
        lum = rec.value[hit] @ np.array([0.2126, 0.7152, 0.0722])
        sc.exposure = float(0.9 / max(np.quantile(lum, 0.99), 1e-6))
    rows = (np.arange(h) * Hf / h).astype(int)
    cols = (np.arange(w) * Wf / w).astype(int)
    img = grid.reshape(Hf, Wf, 3)[rows][:, cols]
    x = np.clip(img * sc.exposure, 0, 1)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    a = (srgb * 255 * 0.55 + 0.5).astype(np.uint8)                 # dimmed, so the cells read
    c, R = sc.T[m][1], sc.T[m][2]
    sel = st.usable & st.vis_now
    if sel.any():
        rc, _ = sc.fan.model().pixel_of((st.X[sel] - c) @ R)
        py = (rc[:, 0] * h / Hf).astype(int).clip(0, h - 2)
        px = ((rc[:, 1] % Wf) * w / Wf).astype(int).clip(0, w - 2)
        col = colours[sel]
        for dy in (0, 1):
            for dx in (0, 1):
                a[py + dy, px + dx] = col
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    _text(d, (16, 10), "The sensor: a 360° × 180° slit fan (Panorama carrier); usable cells this look sees", 20, LIGHT)
    for b, lab in ((90, "left"), (0, "heading"), (-90, "right")):       # bearing b, + left: x = (0.5 - b/360) w
        _text(d, (int((0.5 - b / 360.0) * w), h - 26), lab, 16, LIGHT, anchor="mt")
    _text(d, (6, h - 26), "behind", 16, LIGHT)
    _text(d, (w - 6, h - 26), "behind", 16, LIGHT, anchor="rt")
    return im


def _w2top(sc, P, w, h, pad=30):
    sx = (w - 2 * pad) / (sc.hi[0] - sc.lo[0])
    sy = (h - 2 * pad - 30) / (sc.hi[1] - sc.lo[1])
    k = min(sx, sy)
    ox = pad + 0.5 * ((w - 2 * pad) - k * (sc.hi[0] - sc.lo[0]))
    oy = 30 + pad + 0.5 * ((h - 2 * pad - 30) - k * (sc.hi[1] - sc.lo[1]))
    return ox + (P[:, 0] - sc.lo[0]) * k, oy + (sc.hi[1] - P[:, 1]) * k


def top_panel(sc, m, st, colours):
    x0, y0, w, h = TOP
    im = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(im)
    a = np.asarray(im).copy()
    sel = st.usable
    if sel.any():
        o = np.argsort(st.X[sel, 2])                              # high cells drawn last
        X = st.X[sel][o]
        px, py = _w2top(sc, X, w, h)
        px, py = px.astype(int).clip(0, w - 2), py.astype(int).clip(0, h - 2)
        col = colours[sel][o]
        a[py, px] = col
        a[py + 1, px] = col
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    tx, ty = _w2top(sc, sc.C, w, h)
    d.line(list(zip(tx, ty)), fill=RULE, width=2)
    d.line(list(zip(tx[: m + 1], ty[: m + 1])), fill=ACCENT, width=3)
    fwd = -sc.T[m][2][:, 2]
    hx, hy = tx[m], ty[m]
    ang = math.atan2(-fwd[1], fwd[0])
    pts = [(hx + 14 * math.cos(ang), hy + 14 * math.sin(ang)),
           (hx + 8 * math.cos(ang + 2.5), hy + 8 * math.sin(ang + 2.5)),
           (hx + 8 * math.cos(ang - 2.5), hy + 8 * math.sin(ang - 2.5))]
    d.polygon(pts, fill=ACCENT)
    _text(d, (16, 10), f"Top view: {int(st.usable.sum()):,} usable cells (2 cm)", 20)
    return im


def _view(sc, az_deg, el_deg=40.0, dist=1.6):
    az, el = math.radians(az_deg), math.radians(el_deg)
    back = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    eye = sc.centre + dist * sc.extent * back
    f = sc.centre - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, np.stack([r, u, f])


def cloud_panel(sc, st, colours, az, size=(CLOUD[2], CLOUD[3]), title=True, splat=2, m=None):
    w, h = size
    im = Image.new("RGB", (w, h), PAPER)
    a = np.asarray(im).copy()
    eye, B = _view(sc, az)
    sel = st.usable & (st.X[:, 2] <= CEILING_CLIP)
    if sel.any():
        q = (st.X[sel] - eye) @ B.T
        ok = q[:, 2] > 0.1
        f = 0.5 * w / math.tan(math.radians(55) / 2)
        px = (w / 2 + f * q[ok, 0] / q[ok, 2]).astype(int)
        py = (h / 2 - f * q[ok, 1] / q[ok, 2]).astype(int)
        col = colours[sel][ok]
        o = np.argsort(-q[ok, 2])                                 # far first
        px, py, col = px[o], py[o], col[o]
        inb = (px >= 0) & (px < w - splat) & (py >= 0) & (py < h - splat)
        for dy in range(splat):
            for dx in range(splat):
                a[py[inb] + dy, px[inb] + dx] = col[inb]
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    if m is not None:
        q = (sc.C[: m + 1] - eye) @ B.T
        f = 0.5 * w / math.tan(math.radians(55) / 2)
        pts = [(w / 2 + f * x / z, h / 2 - f * y / z) for x, y, z in q if z > 0.1]
        if len(pts) > 1:
            d.line(pts, fill=ACCENT, width=3)
    if title:
        _text(d, (16, 10), f"3-D: the usable cells (below {CEILING_CLIP:g} m)", 20)
    return im


def prog_panel(sc, m, hist, total_s):
    x0, y0, w, h = PROG
    im = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(im)
    L, R_, T_, B_ = 70, w - 30, 56, h - 150
    _text(d, (16, 10), "Usable surface as the tour goes (oracle, 0.5 px)", 20)
    for v in (0, 25, 50, 75, 100):
        y = B_ - (B_ - T_) * v / 100
        d.line([(L, y), (R_, y)], fill=(225, 225, 220), width=1)
        _text(d, (L - 8, y), f"{v} %", 15, MUTED, anchor="rm")
    starts = [0] + [k for k in range(1, len(sc.seg)) if sc.seg[k] != sc.seg[k - 1]]
    for i, k in enumerate(starts):
        a_ = sc.s[k] if i == 0 else 0.5 * (sc.s[k] + sc.s[k - 1])
        kn = starts[i + 1] if i + 1 < len(starts) else None
        b_ = total_s if kn is None else 0.5 * (sc.s[kn] + sc.s[kn - 1])
        xa, xb = L + (R_ - L) * a_ / total_s, L + (R_ - L) * b_ / total_s
        if i:
            d.line([(xa, T_), (xa, B_)], fill=RULE, width=1)
        _text(d, (0.5 * (xa + xb), B_ + 8), SEG_NAMES[sc.seg[k]], 15, MUTED, anchor="mt")
    _text(d, (R_, B_ + 50), "arc length travelled, m", 15, MUTED, anchor="rt")
    for v in range(0, int(total_s) + 1):
        _text(d, (L + (R_ - L) * v / total_s, B_ + 30), f"{v}", 15, MUTED, anchor="mt")
    if len(hist) > 1:
        pts = [(L + (R_ - L) * s / total_s, B_ - (B_ - T_) * u) for s, u in hist]
        d.line(pts, fill=tuple(int(v) for v in _STOPS[5]), width=3)
        d.ellipse([pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5], fill=ACCENT)
        right = pts[-1][0] > R_ - 90
        _text(d, (pts[-1][0] + (-10 if right else 10), pts[-1][1] - 8), f"{100 * hist[-1][1]:.1f} %", 17, INK,
              anchor="rb" if right else "lb")
    return im


# --------------------------------------------------------------------------------------------------------- the run

class _Enc:
    def __init__(self, path, best=False):
        ff = shutil.which("ffmpeg")
        if ff is None:
            raise SystemExit(f"{PREFIX} refused: no ffmpeg")
        self.p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                   f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset",
                                   "slow" if best else "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
                                  stdin=subprocess.PIPE)
        self.n = 0

    def put(self, im, times=1):
        b = im.tobytes()
        for _ in range(times):
            self.p.stdin.write(b)
            self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        if self.p.returncode:
            raise RuntimeError("ffmpeg failed")


def run(straight_dirs, turn_dirs, out, sigma_px=0.5, seed=21, cell=fu.CELL, hit_stride=1, max_cells=60000,
        limit=None, best=False, git=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    T = tour(straight_dirs, turn_dirs)
    _, _, width = fu.run_meta(turn_dirs)
    fan = sn.SlitFan(width=width)
    step = float(np.median(np.diff([t[3] for t in T])))
    ok2, n2 = check_continuity(T, step, straight_dirs, turn_dirs)
    ok2c, n2c = check_continuity(T, step, straight_dirs, turn_dirs, shift=np.array([0.01, 0.0, 0.0]))
    looks = [(t[0], t[1], t[2], t[3]) for t in T]
    P = fu.cells(looks, cell, hit_stride)
    n_binned = len(P)
    rng = np.random.default_rng(seed + 7)
    if max_cells and len(P) > max_cells:
        P = P[np.sort(rng.choice(len(P), size=max_cells, replace=False))]
    sc = Scene(P, T, fan)
    st = State(P, fan, sigma_px, seed)
    total_s = sc.s[-1] + 0.02
    note = "oracle depth: perfect correspondences + 0.5 px noise (what the path makes measurable)"
    encs = {k: _Enc(out / f"demo_{k}.mp4", best) for k in ("range", "error")}
    hist = []
    K = len(T) if not limit else min(limit, len(T))
    last = {}
    for m in range(K):
        t = T[m]
        st.rec = orc.record(t[0])
        st.vis_now = orc.visible(st.rec, t[1], t[2], P, fan)
        st.add(t[1], t[2], st.vis_now)
        st.update()
        hist.append((t[3], float(st.usable.mean())))
        rng_c = range_colour(np.linalg.norm(st.X - t[1], axis=-1))
        z = (st.X[:, 2] - sc.zlo) / max(sc.zhi - sc.zlo, 1e-6)
        hcol = ramp(np.clip(z, 0, 1))
        u = (P - t[1]) / np.linalg.norm(P - t[1], axis=-1)[:, None]
        e = np.abs(np.einsum("ni,ni->n", st.X - P, u)) / np.linalg.norm(P - t[1], axis=-1)
        ecol = ramp(np.clip(e / ERR_MAX, 0, 1))
        for kind, cols in (("range", (rng_c, hcol)), ("error", (ecol, ecol))):
            im = frame_two(sc, m, st, cols, hist, total_s, kind, note)
            encs[kind].put(im, int(HOLD_START * FPS) if m == 0 else 1)
            last[kind] = (im, cols)
        print(f"{PREFIX} demo look {m + 1}/{K}: {100 * st.usable.mean():.1f} % usable", flush=True)
    for kind, (im, cols) in last.items():
        im.save(out / f"last_{kind}.png")
        encs[kind].put(im, int(HOLD_END * FPS))
        if not limit:
            n = int(ORBIT_S * FPS)
            for i in range(n):
                encs[kind].put(end_card(sc, st, cols[1], i, n, kind))
        encs[kind].close()
    rec = {"command": "demo", "straight": straight_dirs, "turn": turn_dirs, "looks": K, "tour_looks": len(T),
           "step_m": step, "sigma_px": sigma_px, "seed": seed, "cell_m": cell, "hit_stride": hit_stride,
           "cells_binned": n_binned, "cells": int(len(P)), "usable_end": float(st.usable.mean()),
           "progress": hist, "frames": {k: e.n for k, e in encs.items()}, "fps": FPS, "computed_by": git,
           "checks": [{"name": "V2 the tour is the planned L-curve sampled evenly (heads within 1e-9; the join within 1e-6 m)",
                       "ok": ok2, **n2},
                      {"name": "V2 control: the turn's looks shifted 1 cm sideways", "ok": not ok2c, **n2c}]}
    if not limit:
        n1 = check_estimate(T, P, fan, sigma_px, seed, Xfirst=st.X, usable=st.usable)
        rec["checks"] += [
            {"name": "V1 the first-order points equal fusion.fuse's fit (median along-ray difference <= 0.05 sigma)",
             "ok": n1["V1_median_sigma"] <= 0.05, **n1},
            {"name": "V1 control: another noise draw (median >= 0.5 sigma)", "ok": n1["V1_control_median_sigma"] >= 0.5}]
    (out / "demo.json").write_text(json.dumps(rec, indent=1, default=float))
    for c in rec["checks"]:
        print(f"{PREFIX} demo {'PASS' if c['ok'] else 'FAIL'} {c['name']}")
    return 0 if all(c["ok"] for c in rec["checks"]) else 1


def frame_two(sc, m, st, cols, hist, total_s, kind, note):
    """The frame with the panorama coloured by cols[0] and the top view and cloud by cols[1]."""
    im = Image.new("RGB", (W, H), PAPER)
    im.paste(pano_panel(sc, st.rec, m, st, cols[0]), PANO[:2])
    im.paste(top_panel(sc, m, st, cols[1]), TOP[:2])
    im.paste(cloud_panel(sc, st, cols[1], az=-115.0, m=m), CLOUD[:2])
    im.paste(prog_panel(sc, m, hist, total_s), PROG[:2])
    d = ImageDraw.Draw(im)
    d.line([(0, 640), (W, 640)], fill=RULE, width=1)
    d.line([(1280, 0), (1280, 640)], fill=RULE, width=1)
    d.line([(800, 640), (800, H)], fill=RULE, width=1)
    t = sc.T[m]
    _text(d, (816, H - 72), f"{SEG_NAMES[t[4]]} · {t[3]:.2f} m along · look {m + 1} of {len(sc.T)}", 17, INK)
    if kind == "error":
        _text(d, (816, H - 48), f"colour: |error along the ray| / range, light 0 → dark ≥ {100 * ERR_MAX:g} %", 16,
              MUTED)
    else:
        _text(d, (816, H - 48), "colour: range in the panorama, height in the top view and 3-D; dark = near / high",
              16, MUTED)
    _text(d, (816, H - 26), note, 16, MUTED)
    return im


def end_card(sc, st, cols, i, n, kind):
    im = Image.new("RGB", (W, H), PAPER)
    az = -115.0 + 360.0 * i / n
    im.paste(cloud_panel(sc, st, cols, az=az, size=(W, H - 60), title=False, splat=2, m=len(sc.T) - 1), (0, 60))
    d = ImageDraw.Draw(im)
    what = "relative error along the ray" if kind == "error" else "height"
    _text(d, (24, 16), f"The tour's fused surface: {int(st.usable.sum()):,} usable cells, coloured by {what} "
                       f"(oracle depth, 0.5 px; cells above {CEILING_CLIP:g} m left out of this view)", 24)
    return im
