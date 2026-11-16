"""D1's video: three panels, one look per frame (DERIVED: a display render, not a measurement).

    +----------------------------+---------------+
    |                            |   top view    |
    |   foveated sensor          +---------------+
    |   (+ the reference whose   |   3D cloud    |
    |    depth just arrived)     |               |
    +----------------------------+---------------+

The sensor panel is the sensor's own log-polar raster (`visuals.raster_image`), nearest-neighbour enlarged. The
inset is the latest reference whose both5 estimate is ready (its window's last frame has been seen: 5 looks of its
phase, 0.5 m, behind the head), with its accepted anchors coloured by range. The top view and the 3D cloud show the
voxel fusion of every estimate ready so far. Colours: one blue ramp, light -> dark (the dataviz skill's sequential
hue, steps 250-700): range in the inset, height on the top view, and in the `error` video |dz| / z clipped at 5 %.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import analysis, tour
from .demo import PREFIX, load_tour
from .demo_steps import fuse, hits
from .pointcloud import exposure_of as exposure_rgb, srgb_u8
from .visuals import _font, raster_image

W, H, FPS = 1920, 1080, 24
SENSOR = (0, 0, 1080, 1080)
TOP = (1080, 0, 840, 540)
CLOUD = (1080, 540, 840, 540)
BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, MUTED, RULE, PAPER, NIGHT = (34, 34, 34), (110, 110, 105), (200, 200, 195), (247, 247, 245), (24, 24, 26)
NO_TRUTH = (165, 165, 160)     # error video: voxels with no truth-hit anchor (rays that leave the scene, #19)
ERR_MAX = 0.05
HOLD_START, HOLD_END, ORBIT_S = 1.0, 1.0, 8.0


def _hex(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


_STOPS = np.stack([_hex(h) for h in BLUE])


def ramp(v) -> np.ndarray:
    """v in [0, 1] -> uint8 rgb along the blue ramp (0 light, 1 dark)."""
    v = np.clip(np.nan_to_num(np.asarray(v, float), nan=0.0), 0, 1) * (len(_STOPS) - 1)
    i = np.minimum(np.floor(v).astype(int), len(_STOPS) - 2)
    f = (v - i)[..., None]
    return (_STOPS[i] * (1 - f) + _STOPS[i + 1] * f + 0.5).astype(np.uint8)


class Ctx:
    def __init__(self, out: Path, color: str):
        self.t = load_tour(out)
        self.out = Path(out)
        self.color = color
        self.spec = self.t["spec"]
        self.tl = self.t["timeline"]
        self.runs = {}
        a0 = self.run(tour.phase_name(self.spec, 0, 0))
        from .visuals import exposure_of
        self.ex = exposure_of(a0, a0["looks"]["ahead"][0])
        A = dict(np.load(self.out / "fused" / "anchors.npz"))
        A["hit"] = hits(A)
        o = np.argsort(A["t_avail"], kind="stable")
        self.A = {k: v[o] for k, v in A.items()}
        self.ex_rgb = exposure_rgb(self.A["rgb"]) if len(self.A["rgb"]) else 1.0
        self._fz = (-1, None)
        tv = json.loads((self.out / "topview" / "topview.json").read_text())
        self.tv = tv
        img = np.asarray(Image.open(self.out / "topview" / "topview.png").convert("RGB"), float)
        self.top_bg = (img * 0.45 + 255 * 0.55).astype(np.uint8)
        self.heads = [self.head(f) for f in range(len(self.tl))]
        legs = self.spec["legs"]
        self.corners = [np.asarray(legs[0]["start_w"], float)] + [np.asarray(lg["end_w"], float) for lg in legs]
        b = tv["bounds_w"]
        self.centre = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0])
        self.radius = 0.9 * max(b[2] - b[0], b[3] - b[1])

    def run(self, name):
        if name not in self.runs:
            self.runs[name] = analysis.load_run(self.out / name)
        return self.runs[name]

    def entry(self, f):
        e = self.tl[f]
        if e["kind"] == "leg":
            r = self.run(tour.phase_name(self.spec, e["leg"], e["phase"]))
            return r, r["looks"]["ahead"][e["k"]]
        r = self.run(f"pivot{e['leg']}")
        return r, r["looks"]["ahead"][e["j"]]

    def head(self, f):
        _, en = self.entry(f)
        R = np.asarray(en["head"]["head_R_wh"], float)
        return np.asarray(en["head"]["head_origin_w_m"], float), -R[:, 2]

    def fused(self, f):
        """The fusion of every estimate ready at frame f, cut at the top view's height (the ceiling would hide the
        room from above, in both views; the PLY files keep everything)."""
        n = int(np.searchsorted(self.A["t_avail"], f, side="right"))
        if self._fz[0] != n:
            A = self.A
            h = A["hit"][:n].astype(float)
            F = fuse(A["P"][:n], A["w"][:n], A["rgb"][:n],
                     {"relh": np.where(h > 0, A["rel"][:n], 0.0), "h": h}) if n else None
            if F is not None:
                keep = F["P"][:, 2] < self.tv["cut_z_m"]
                F = {k: (v[keep] if k != "inv" else v) for k, v in F.items()}
            self._fz = (n, F)
        return self._fz[1]

    def ready_ref(self, f):
        """The latest (t_avail, leg, phase, r) ready at frame f, on the current leg; None during a pivot."""
        e = self.tl[f]
        if e["kind"] != "leg":
            return None
        A = self.A
        m = (A["t_avail"] <= f) & (A["leg"] == e["leg"])
        if not m.any():
            return None
        i = np.flatnonzero(m)[-1]
        return int(A["leg"][i]), int(A["phase"][i]), int(A["r"][i])

    def colours(self, F):
        if self.color == "error":
            return error_colours(F)
        return srgb_u8(F["rgb"], self.ex_rgb)


def error_colours(F) -> np.ndarray:
    """|dz|/z over a voxel's truth-hit anchors on the blue ramp (0 -> 5 %); NO_TRUTH grey where it has none."""
    has = F["h"] > 0
    rel = np.where(has, F["relh"] / np.where(has, F["h"], 1.0), 0.0)
    c = ramp(rel / ERR_MAX)
    c[~has] = NO_TRUTH
    return c


# ----------------------------------------------------------------------------- panels

def _text(d, xy, s, size, fill=INK):
    d.text(xy, s, fill=fill, font=_font(size))


def sensor_panel(ctx: Ctx, f: int) -> Image.Image:
    x, y, w, h = SENSOR
    im = Image.new("RGB", (w, h), NIGHT)
    d = ImageDraw.Draw(im)
    run, en = ctx.entry(f)
    ras = raster_image(run, en, ctx.ex)
    side = 900
    im.paste(ras.resize((side, side), Image.NEAREST), ((w - side) // 2, 70))
    e = ctx.tl[f]
    _text(d, (28, 18), "Foveated sensor: the eye's log-polar raster (253 × 253, fovea at the centre)", 22, (235, 235, 230))
    rr = ctx.ready_ref(f)
    if rr is not None:
        leg, p, r = rr
        rn = tour.phase_name(ctx.spec, leg, p)
        rrun = ctx.run(rn)
        rim = raster_image(rrun, rrun["looks"]["ahead"][r], ctx.ex)
        n = rim.width
        A = ctx.A
        m = (A["leg"] == leg) & (A["phase"] == p) & (A["r"] == r)
        S = 300
        inset = np.asarray(rim.resize((S, S), Image.NEAREST)).copy()
        inset = (inset * 0.55).astype(np.uint8)
        rows, cols = np.divmod(A["flat"][m].astype(int), n)
        py = ((rows + 0.5) * S / n).astype(int).clip(0, S - 2)
        px = ((cols + 0.5) * S / n).astype(int).clip(0, S - 2)
        lr = np.log(np.clip(A["range"][m], 0.25, 8.0))
        c = ramp(1 - (lr - math.log(0.25)) / (math.log(8.0) - math.log(0.25)))     # near = dark
        for dy in (0, 1):
            for dx in (0, 1):
                inset[py + dy, px + dx] = c
        iy = h - S - 70
        im.paste(Image.fromarray(inset), (w - S - 24, iy))
        d.rectangle((w - S - 25, iy - 1, w - 23, iy + S), outline=(235, 235, 230))
        _text(d, (w - S - 24, iy - 26), f"depth ready: {rn} r{r}, 0.5 m back", 18, (235, 235, 230))
        _text(d, (w - S - 24, iy + S + 4), "anchors: near (dark) → far (light)", 16, (200, 200, 195))
    if e["kind"] == "leg":
        lg = ctx.spec["legs"][e["leg"]]
        status = f"Leg {lg['name']} · {e['s_m']:.2f} m along · gaze ahead · a look every {ctx.spec['step_m'] / ctx.spec['phases'] * 100:.1f} cm"
    else:
        status = "Turning in place: no translation, so no depth (the alignment gate refuses these windows)"
    _text(d, (28, h - 44), status, 22, (235, 235, 230))
    return im


def _w2top(ctx, P, s, ox, oy):
    x0, y0, x1, y1 = ctx.tv["bounds_w"]
    m = ctx.tv["m_per_px"]
    return np.stack([(P[..., 0] - x0) / m * s + ox, (y1 - P[..., 1]) / m * s + oy], -1)


def top_panel(ctx: Ctx, f: int) -> Image.Image:
    _, _, w, h = TOP
    im = Image.new("RGB", (w, h), PAPER)
    bg = ctx.top_bg
    head = 40
    s = min((w - 20) / bg.shape[1], (h - head - 30) / bg.shape[0])
    bw, bh = int(bg.shape[1] * s), int(bg.shape[0] * s)
    ox, oy = (w - bw) // 2, head + (h - head - 30 - bh) // 2
    arr = np.asarray(im).copy()
    arr[oy:oy + bh, ox:ox + bw] = np.asarray(Image.fromarray(bg).resize((bw, bh), Image.BILINEAR))
    F = ctx.fused(f)
    if F is not None:
        q = _w2top(ctx, F["P"], s, ox, oy).astype(int)
        z = F["P"][:, 2]
        c = ramp(z / ctx.tv["cut_z_m"])
        o = np.argsort(z)                           # higher points drawn last
        q, c = q[o], c[o]
        ok = (q[:, 0] >= ox) & (q[:, 0] < ox + bw - 1) & (q[:, 1] >= oy) & (q[:, 1] < oy + bh - 1)
        for dy in (0, 1):
            for dx in (0, 1):
                arr[q[ok, 1] + dy, q[ok, 0] + dx] = c[ok]
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im, "RGBA")
    cor = _w2top(ctx, np.stack(ctx.corners), s, ox, oy)
    for a, b in zip(cor[:-1], cor[1:]):          # the whole L, dashed
        L = np.linalg.norm(b - a)
        for t0 in np.arange(0, L, 12):
            t1 = min(t0 + 6, L)
            d.line([tuple(a + (b - a) * t0 / L), tuple(a + (b - a) * t1 / L)], fill=MUTED, width=2)
    trav = np.stack([ctx.heads[g][0] for g in range(f + 1)])
    tp = _w2top(ctx, trav, s, ox, oy)
    d.line([tuple(p) for p in tp], fill=INK, width=3)
    c, fw = ctx.heads[f]
    cone = []
    for ang in np.linspace(-45, 45, 13):
        a = math.radians(ang)
        v = np.array([fw[0] * math.cos(a) - fw[1] * math.sin(a), fw[0] * math.sin(a) + fw[1] * math.cos(a), 0])
        cone.append(c + 1.0 * v)
    cp = _w2top(ctx, np.stack([c] + cone), s, ox, oy)
    d.polygon([tuple(p) for p in cp], fill=(34, 34, 34, 40), outline=(34, 34, 34, 160))
    hp = cp[0]
    d.ellipse((hp[0] - 6, hp[1] - 6, hp[0] + 6, hp[1] + 6), fill=INK)
    rr = ctx.ready_ref(f)
    if rr is not None:
        leg, p, r = rr
        fr = tour.frame_of(ctx.tl)[(leg, p, r)]
        q = _w2top(ctx, ctx.heads[fr][0], s, ox, oy)
        d.ellipse((q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7), outline=tuple(int(v) for v in _hex(BLUE[7])), width=3)
    _text(d, (16, 10), "Top view: the path, the head and its 45° field, fused depth by height", 18)
    bar = ramp(np.linspace(0, 1, 120))
    for i, col in enumerate(bar):
        d.line([(w - 180 + i, h - 20), (w - 180 + i, h - 10)], fill=tuple(int(v) for v in col))
    _text(d, (w - 214, h - 24), "0 m", 14, MUTED)
    _text(d, (w - 54, h - 24), f"{ctx.tv['cut_z_m']:g} m", 14, MUTED)
    _text(d, (16, h - 26), "○ depth ready (0.5 m behind the head)   — travelled   - - planned", 14, MUTED)
    return im


def _view(ctx, az_deg, el_deg=38.0, zoom=1.0):
    az, el = math.radians(az_deg), math.radians(el_deg)
    eye = ctx.centre + zoom * ctx.radius * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    fwd = ctx.centre - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, np.stack([right, up, fwd])


def _project(P, eye, B, w, h, fov_deg=55.0):
    q = (P - eye) @ B.T
    foc = (w / 2) / math.tan(math.radians(fov_deg) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = w / 2 + foc * q[..., 0] / q[..., 2]
        y = h / 2 - foc * q[..., 1] / q[..., 2]
    return x, y, q[..., 2]


def cloud_panel(ctx: Ctx, f: int, size=(CLOUD[2], CLOUD[3]), az=None, title=True, splat=2, zoom=1.0) -> Image.Image:
    w, h = size
    bgc = PAPER if ctx.color == "error" else NIGHT
    ink = INK if ctx.color == "error" else (235, 235, 230)
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:] = bgc
    az = (-120.0 + 360.0 * f / (40 * FPS)) if az is None else az
    eye, B = _view(ctx, az, zoom=zoom)
    F = ctx.fused(f)
    if F is not None:
        x, y, z = _project(F["P"], eye, B, w, h)
        c = ctx.colours(F)
        ok = (z > 0.1) & (x >= 0) & (x < w - splat) & (y >= 0) & (y < h - splat)
        o = np.argsort(-z[ok])                       # far first, near overwrite (painter's z-buffer)
        xi, yi, ci = x[ok][o].astype(int), y[ok][o].astype(int), c[ok][o]
        for dy in range(splat):
            for dx in range(splat):
                arr[yi + dy, xi + dx] = ci
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    trav = np.stack([ctx.heads[g][0] for g in range(f + 1)])
    x, y, z = _project(trav, eye, B, w, h)
    if len(x) > 1:
        d.line(list(zip(x, y)), fill=ink, width=2)
    d.ellipse((x[-1] - 5, y[-1] - 5, x[-1] + 5, y[-1] + 5), fill=ink)
    if title:
        n = 0 if F is None else len(F["P"])
        band = 54 if ctx.color == "error" else 34
        d.rectangle((0, 0, w, band), fill=bgc)
        _text(d, (16, 8), f"3D: the fused cloud so far (below {ctx.tv['cut_z_m']:g} m), {n:,} voxels of 2 cm", 18, ink)
        if ctx.color == "error":
            _text(d, (16, 31), "colour: |Δz|/z, 0 (light) → 5 % (dark); grey: no truth (the ray leaves the scene)", 15, MUTED)
    return im


def frame(ctx: Ctx, f: int) -> Image.Image:
    im = Image.new("RGB", (W, H), PAPER)
    im.paste(sensor_panel(ctx, f), SENSOR[:2])
    im.paste(top_panel(ctx, f), TOP[:2])
    im.paste(cloud_panel(ctx, f), CLOUD[:2])
    d = ImageDraw.Draw(im)
    d.line([(TOP[0], CLOUD[1]), (W, CLOUD[1])], fill=RULE, width=2)
    return im


def end_card(ctx: Ctx, i: int, n: int) -> Image.Image:
    f = len(ctx.tl) - 1
    im = cloud_panel(ctx, f, size=(W, H), az=-120.0 + 360.0 * i / n, title=False, splat=3, zoom=1.5)
    d = ImageDraw.Draw(im)
    F = ctx.fused(f)
    ink = INK if ctx.color == "error" else (235, 235, 230)
    _text(d, (40, 30), f"{ctx.spec['name']}: the frozen M4 (both5) on the whole tour", 34, ink)
    _text(d, (40, 80), f"{0 if F is None else len(F['P']):,} voxels of 2 cm from "
                       f"{len(np.unique(np.stack([ctx.A['leg'], ctx.A['phase'], ctx.A['r']], 1), axis=0))} references"
                       + ("  ·  colour: |Δz|/z, 0 → 5 %; grey: no truth (the ray leaves the scene)" if ctx.color == "error" else ""), 24, ink)
    return im


def video(a) -> int:
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise SystemExit(f"{PREFIX} refused: no ffmpeg")
    ctx = Ctx(Path(a.out), a.color)
    fr = Path(a.out) / "video" / f"frames_{a.color}"
    if fr.exists():
        shutil.rmtree(fr)
    fr.mkdir(parents=True)
    n = 0
    seq = [0] * int(HOLD_START * FPS) + list(range(len(ctx.tl))) + [len(ctx.tl) - 1] * int(HOLD_END * FPS)
    last = None
    for f in seq[: a.limit or None]:
        if f != last:
            img = frame(ctx, f)
            last = f
        img.save(fr / f"f{n:05d}.png")
        n += 1
    if not a.limit:
        m = int(ORBIT_S * FPS)
        for i in range(m):
            end_card(ctx, i, m).save(fr / f"f{n:05d}.png")
            n += 1
    out = Path(a.out) / "video" / f"d1_{ctx.spec['name']}_{a.color}.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(fr / "f%05d.png"),
                    "-c:v", "libx264", "-preset", "slow" if a.best else "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(out)], check=True)
    print(f"{PREFIX} {n} frames, {n / FPS:.1f} s -> {out}")
    return 0


def add_parser(sub):
    p = sub.add_parser("demo-video")
    p.add_argument("--out", required=True)
    p.add_argument("--color", default="rgb", choices=("rgb", "error"))
    p.add_argument("--limit", type=int, default=0, help="only the first n frames (a quick look)")
    p.add_argument("--best", action="store_true", help="-preset slow (default veryfast)")
