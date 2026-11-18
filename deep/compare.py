"""#11: the representations side by side, straight forward motion (M4) against the composed paths' primitives.

Figures for intuition (DERIVED, Chat sandbox), from calibration-room records:
- the images are renders (M3's foveated dolly run at 1024 spp; the 5b.2 panorama at 16 spp);
- the time slices (epipolar-plane images) re-project the reference look's true points (Position) to each pose of a
  dense time grid, nearest first per column: exact geometry, no rendering; what the reference look does not see
  (disocclusions) is missing.

The coordinates (docs/plan.md, "The formulation"; deep/primitives.py; motion/logpolar.py):
- M4, forward translation along the eye's axis: q = ln tan e (e from the axis), phi the azimuth about it. A step B
  shifts q by ln(z / (z - B)), the same for every point at axial depth z: the invariant is zeta = 1/z, which changes
  as the eye moves.
- line: omega about the path line, u = cot psi; u(s) = (x_P - s) / D, a straight trace. The invariant D, the
  distance to the path line, does not change along the path.
- arc: the slit bearing beta and the head angle theta (the Panorama's column is the slit); invariant rho, the
  distance from the arc's centre.
- pivot: every trace is the same shift; no invariant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import oracle as orc
from . import primitives as pr
from . import sensor as sn

M3_REF = "/home/claude/foveal-motion/runs/m3-calib-1024/ahead/k0000/record_C.npz"
PANO_REF = "runs/s5b2-turn/k0000/record_C.npz"          # the calibration room's EYE (0, 0, 1.6), heading +Y


def tonemap(rgb, exposure):
    x = np.clip(np.nan_to_num(rgb) * exposure, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def exposure_of(values):
    lum = values @ np.array([0.2126, 0.7152, 0.0722])
    return float(0.9 / max(np.quantile(lum, 0.99), 1e-6))


def points(rec):
    hit = np.isfinite(rec.distance_m)
    return rec.position_w[hit].astype(np.float64), rec.value[hit].astype(np.float64)


def epi(coord, rng, ncols, lo, hi, max_gap=10):
    """coord, rng: (T, N) column coordinate and range per time and point (nan: out). An image (T, ncols) of point
    indices, nearest first per cell (-1 empty); an empty cell takes the nearest filled cell of its row within
    max_gap columns (the points are samples of a surface, not dots)."""
    T, N = coord.shape
    img = np.full((T, ncols), -1, np.int64)
    best = np.full((T, ncols), np.inf)
    ok = np.isfinite(coord)
    col = np.full(coord.shape, -1, np.int64)
    col[ok] = np.floor((coord[ok] - lo) / (hi - lo) * ncols).astype(np.int64)
    ok &= (col >= 0) & (col < ncols)
    ti, ni = np.nonzero(ok)
    ci, ri = col[ti, ni], rng[ti, ni]
    order = np.argsort(-ri)                                   # far first, nearest written last
    img[ti[order], ci[order]] = ni[order]
    idx = np.arange(ncols)
    for r in range(T):
        f = img[r] >= 0
        if not f.any():
            continue
        left = np.maximum.accumulate(np.where(f, idx, -10 ** 9))
        right = np.minimum.accumulate(np.where(f, idx, 10 ** 9)[::-1])[::-1]
        dl, dr = idx - left, right - idx
        src = np.where(dl <= dr, left, right)
        fill = ~f & (np.minimum(dl, dr) <= max_gap)
        img[r, fill] = img[r, src[fill]]
    return img


def colour(img, cols, bg=(0.93, 0.93, 0.91)):
    out = np.empty(img.shape + (3,))
    out[:] = bg
    m = img >= 0
    out[m] = cols[img[m]]
    return out


# ------------------------------------------------------------------------------------------- straight motion (M4)

def m4_side(t_max=1.0, nt=300, ncols=480, band_deg=1.0):
    """M4's coordinates on M3's reference look: the foveated image, J(q, phi), and the EPI of the right half-plane."""
    from motion import logpolar as lp
    from fov_render import sensors
    rec = orc.record(M3_REF)
    model = sensors.Foveated(E2_deg=2.0, e_max_deg=45.0).model("full")
    img = rec.as_raster("value", fill=np.nan)
    ex = exposure_of(rec.value)
    grid = lp.make_grid(model, n_q=240, n_phi=480, e_lo_deg=1.0, e_hi_deg=45.0)
    J = lp.unwrap(img, grid)
    P, val = points(rec)
    c0 = np.asarray(rec.head_origin_w, float)
    Rwh = np.asarray(rec.head_R_wh, float)
    f = -Rwh[:, 2]                                            # the eye axis (forward), the motion direction
    right = Rwh[:, 0]
    up = Rwh[:, 1]
    d = P - c0
    x = d @ f
    lat = np.stack([d @ right, d @ up], -1)
    L = np.linalg.norm(lat, axis=-1)
    phi = np.arctan2(lat[:, 1], lat[:, 0])
    band = (np.abs(phi) < np.radians(band_deg)) & (x > 0)
    t = np.linspace(0, t_max, nt)
    z = x[band][None, :] - t[:, None]
    e = np.arctan2(L[band][None, :], z)
    q = np.where((z > 0) & (e <= np.radians(45.0)), np.log(np.tan(e)), np.nan)
    rr = np.hypot(L[band][None, :], z)
    q_lo, q_hi = float(grid.q[0]), float(grid.q[-1])
    E = epi(q, rr, ncols, q_lo, q_hi)
    return {"img": tonemap(img, ex), "J": tonemap(J, ex), "grid": grid, "epi": colour(E, tonemap(val[band], ex)),
            "t": t, "q_lo": q_lo, "q_hi": q_hi, "x": x[band], "L": L[band], "c0": c0, "q": q}


# ------------------------------------------------------------------------------------------- the panorama primitives

def pano_ref():
    rec = orc.record(PANO_REF)
    fan = sn.SlitFan(width=int(rec.raster_shape[1]))
    return rec, fan


def pano_image(rec):
    img = rec.as_raster("value", fill=np.nan)
    return tonemap(img, exposure_of(rec.value))


def line_canonical(rec, fan, n_omega=360, n_psi=720):
    """The panorama resampled on (omega, psi) about the heading: rows omega (-180..180, from right toward up), columns
    psi (0 ahead .. 180 behind), u = cot psi. Nearest-pixel resampling."""
    img = pano_image(rec)
    om = np.radians(np.linspace(-180, 180, n_omega, endpoint=False) + 180 / n_omega)
    ps = np.radians((np.arange(n_psi) + 0.5) * 180 / n_psi)
    O, S = np.meshgrid(om, ps, indexing="ij")
    loc = np.stack([np.sin(S) * np.cos(O), np.sin(S) * np.sin(O), -np.cos(S)], -1)   # head frame: x right, y up, -z fwd
    rc, _ = fan.model().pixel_of(loc.reshape(-1, 3))
    H, W = img.shape[:2]
    i = np.clip(np.rint(rc[:, 0]).astype(int), 0, H - 1)
    j = np.rint(rc[:, 1]).astype(int) % W
    return img[i, j].reshape(n_omega, n_psi, 3)


def line_epi(rec, t_max=1.0, nt=300, ncols=480, band_deg=1.0, u_lim=4.0):
    """The right half-plane (omega ~ 0) along the heading: u = cot psi against t, nearest first per column."""
    P, val = points(rec)
    c0 = np.asarray(rec.head_origin_w, float)
    Rwh = np.asarray(rec.head_R_wh, float)
    f, right, up = -Rwh[:, 2], Rwh[:, 0], Rwh[:, 1]
    d = P - c0
    x = d @ f
    lat = np.stack([d @ right, d @ up], -1)
    D = np.linalg.norm(lat, axis=-1)
    om = np.arctan2(lat[:, 1], lat[:, 0])
    band = np.abs(om) < np.radians(band_deg)
    t = np.linspace(0, t_max, nt)
    u = (x[band][None, :] - t[:, None]) / D[band][None, :]
    rr = np.hypot(D[band][None, :], x[band][None, :] - t[:, None])
    E = epi(np.where(np.abs(u) <= u_lim, u, np.nan), rr, ncols, -u_lim, u_lim)
    ex = exposure_of(rec.value)
    return {"epi": colour(E, tonemap(val[band], ex)), "t": t, "u_lim": u_lim, "x": x[band], "D": D[band],
            "u": np.where(np.abs(u) <= u_lim, u, np.nan)}


def bearing_epi(rec, kind, length, nt=300, ncols=720, band_deg=0.75, R=1.0):
    """The eye-height plane (|elevation| < band) seen from a line, an arc (left, radius R) or a pivot started at the
    reference pose: bearing (+ left, -180..180) against the path parameter, nearest first per column."""
    P, val = points(rec)
    c0 = np.asarray(rec.head_origin_w, float)
    Rwh = np.asarray(rec.head_R_wh, float)
    fwd = -Rwh[:, 2]
    phi0 = float(np.arctan2(fwd[1], fwd[0]))
    el = np.arcsin(np.clip((P[:, 2] - c0[2]) / np.linalg.norm(P - c0, axis=-1), -1, 1))
    band = np.abs(el) < np.radians(band_deg)
    Pb = P[band]
    piece = pr.Piece(kind, length, 1.0 / R if kind == "arc" else 0.0)
    ss = np.linspace(0, length, nt)
    B = np.full((nt, len(Pb)), np.nan)
    RR = np.full((nt, len(Pb)), np.nan)
    for i, s in enumerate(ss):
        c, Rh, _, _ = piece.pose(c0[:2], phi0, c0[2], s)
        v = (Pb - c) @ Rh
        B[i] = -np.arctan2(v[:, 0], -v[:, 2])
        RR[i] = np.linalg.norm(Pb - c, axis=-1)
    E = epi(np.degrees(B), RR, ncols, -180.0, 180.0)
    ex = exposure_of(rec.value)
    Bd = np.degrees(B)
    jump = np.abs(np.diff(Bd, axis=0, prepend=Bd[:1])) > 90
    Bd[jump] = np.nan
    return {"epi": colour(E, tonemap(val[band], ex)), "s": ss, "Pb": Pb, "phi0": phi0, "c0": c0, "R": R,
            "kind": kind, "length": length, "B": Bd}


def arc_invariant(Pb, c0, phi0, R):
    """rho and phi_P of points about the arc's centre (left turn from c0, heading phi0)."""
    centre = c0[:2] + R * np.array([-np.sin(phi0), np.cos(phi0)])
    d = Pb[:, :2] - centre
    return np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0]), centre


# --------------------------------------------------------------------------------------------------------- figures

def _traces(ax, X, y, inv, n=36, lo=None, hi=None, vmax=5.0, seed=3):
    """Draw n traces (columns of X against y), chosen spread over their first visible coordinate, coloured by their
    invariant (blue ramp, dark = small); returns the mappable for a colour bar."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from .figures import BLUE
    cmap = LinearSegmentedColormap.from_list("b", list(BLUE[::-1]))
    first = np.array([X[np.flatnonzero(np.isfinite(X[:, k]))[0], k] if np.isfinite(X[:, k]).any() else np.nan
                      for k in range(X.shape[1])])
    ok = np.flatnonzero(np.isfinite(first) & (np.isfinite(X).sum(0) > 5))
    bins = np.linspace(np.nanmin(first[ok]), np.nanmax(first[ok]), n + 1)
    rng = np.random.default_rng(seed)
    pick = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = ok[(first[ok] >= a) & (first[ok] < b)]
        if len(m):
            pick.append(int(rng.choice(m)))
    segs, vals = [], []
    for k in pick:
        f = np.isfinite(X[:, k]).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, f, 0]))
        for a, b in zip(edges[::2], edges[1::2]):               # one segment per finite run (wraps, exits)
            if b - a > 1:
                segs.append(np.stack([X[a:b, k], y[a:b]], -1))
                vals.append(inv[k])
    norm = Normalize(0, vmax)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=1.6)
    lc.set_array(np.asarray(vals))
    ax.add_collection(lc)
    return lc

def _pick(D, x, targets, rng_lo=0.3):
    """Indices of points with D nearest each target and x well ahead."""
    out = []
    for tgt in targets:
        m = np.flatnonzero(x > rng_lo)
        out.append(int(m[np.argmin(np.abs(D[m] - tgt))]))
    return out


def fig_straight(out):
    """Figure 1: forward motion, M4's log-tangent coordinates against the line's (omega, u = cot psi)."""
    from .figures import _plt, ACCENT, INK2, BLUE
    plt = _plt()
    a = m4_side()
    rec, fan = pano_ref()
    can = line_canonical(rec, fan)
    b = line_epi(rec)
    pano = pano_image(rec)
    fig = plt.figure(figsize=(16, 10.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.15], hspace=0.62, wspace=0.24)
    # row 1: M4
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.nan_to_num(a["img"], nan=0.93), interpolation="nearest")
    h = a["img"].shape[0]
    ax.plot([h / 2, h - 2], [h / 2, h / 2], color=ACCENT, lw=1.6)
    ax.plot(h / 2, h / 2, "+", color=ACCENT, ms=12, mew=1.6)
    ax.set_title("M4: the foveated image (out to 45°)\n+: focus of expansion; orange: azimuth φ = 0",
                 loc="left")
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    Jt = np.transpose(a["J"], (1, 0, 2))
    e_deg = np.degrees(a["grid"].e)
    ax.imshow(np.nan_to_num(Jt, nan=0.93), aspect="auto", extent=(a["q_lo"], a["q_hi"], 180, -180),
              interpolation="nearest")
    ax.axhline(0, color=ACCENT, lw=1.4)
    ax.set_xlabel("q = ln tan e   (e from the axis: 1° … 45°)")
    ax.set_ylabel("azimuth φ, deg")
    ax.set_title("Log-tangent unwrapping about the focus of expansion:\na step B shifts a point by ln(z/(z−B)) along its row",
                 loc="left")
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(a["epi"], aspect="auto", extent=(a["q_lo"], a["q_hi"], a["t"][-1], 0), interpolation="nearest",
              alpha=0.35)
    lc = _traces(ax, a["q"], a["t"], a["x"])
    cb = fig.colorbar(lc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("z₀, axial depth at t = 0 (m)")
    ax.set_xlim(a["q_lo"], a["q_hi"])
    ax.set_xlabel("q = ln tan e")
    ax.set_ylabel("t, m travelled")
    ax.set_ylim(a["t"][-1], 0)
    ax.set_title("The φ = 0 row over time: traces curve, faster as the\npoint nears; equal z₀ = the same curve shifted in q",
                 loc="left")
    # row 2: ours, the line
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(np.nan_to_num(pano, nan=0.0), extent=(180, -180, -90, 90), aspect="auto", interpolation="nearest")
    ax.plot([0, -180], [0, 0], color=ACCENT, lw=1.6)
    ax.plot(0, 0, "+", color=ACCENT, ms=12, mew=1.6)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-90, 91, 45))
    ax.set_xlabel("bearing, deg (+ left)")
    ax.set_title("Ours: the slit-fan panorama (all directions)\n+: heading; orange: the plane ω = 0", loc="left")
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(can, aspect="auto", extent=(0, 180, 180, -180), interpolation="nearest")
    ax.axhline(0, color=ACCENT, lw=1.4)
    sec = ax.secondary_xaxis("top", functions=(lambda p: 1 / np.tan(np.radians(np.clip(p, 1, 179))),
                                               lambda u: np.degrees(np.arctan2(1, u))))
    sec.set_xticks([4, 1, 0, -1, -4])
    sec.set_xlabel("u = cot ψ", color=INK2)
    ax.set_xlabel("ψ from the heading, deg (0 ahead … 180 behind)")
    ax.set_ylabel("ω about the path line, deg")
    ax.set_title("The line's pencil of planes: each row is one plane\nthrough the path line; a point never leaves its row",
                 loc="left", pad=34)
    ax = fig.add_subplot(gs[1, 2])
    ul = b["u_lim"]
    ax.imshow(b["epi"], aspect="auto", extent=(-ul, ul, b["t"][-1], 0), interpolation="nearest", alpha=0.35)
    lc = _traces(ax, b["u"], b["t"], b["D"])
    cb = fig.colorbar(lc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("D, distance to the path line (m)")
    ax.set_xlim(ul, -ul)
    ax.set_ylim(b["t"][-1], 0)
    ax.set_xlabel("u = cot ψ  (ahead → behind)")
    ax.set_ylabel("t, m travelled")
    ax.set_title("The ω = 0 row over time: every trace is straight,\nslope −1/D; equal D = the same line shifted in t",
                 loc="left")
    fig.suptitle("Straight forward motion in two coordinate systems (calibration room; images rendered, time slices "
                 "re-projected from the first look's true points)", x=0.01, ha="left", fontsize=12)
    p = Path(out) / "compare_straight.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_primitives(out):
    """Figure 2: one eye-height plane from the same start, under a line, an arc and a pivot: bearing against time."""
    from .figures import _plt, ACCENT, INK2
    plt = _plt()
    rec, fan = pano_ref()
    runs = [("line", "line, 1.5 m (bearing here; straight in u = cot ψ: figure 1)", 1.5, "s, m"),
            ("arc", "arc, R = 1 m, 90° left: this is its own canonical image (β, θ)", np.pi / 2, "θ = s / R, rad"),
            ("pivot", "pivot (turn in place), 90° left", np.pi / 2, "turn, rad")]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.8), gridspec_kw={"wspace": 0.42})
    for ax, (kind, name, L, ylab) in zip(axs, runs):
        e = bearing_epi(rec, kind, L)
        ax.imshow(e["epi"][:, ::-1], aspect="auto", extent=(180, -180, e["s"][-1], 0), interpolation="nearest")
        Pb, c0, phi0 = e["Pb"], e["c0"], e["phi0"]
        ax.images[0].set_alpha(0.35)
        if kind == "line":
            f = np.array([np.cos(phi0), np.sin(phi0)])
            inv = np.abs((Pb[:, :2] - c0[:2]) @ np.array([-f[1], f[0]]))
            note, lab = "colour: D; equal D = the same trace shifted in s", "D (m)"
        elif kind == "arc":
            inv = arc_invariant(Pb, c0, phi0, 1.0)[0]
            note, lab = "colour: ρ; equal ρ = the same trace shifted in θ", "ρ (m)"
        else:
            inv = np.linalg.norm(Pb - c0, axis=-1)
            note, lab = "colour: range; every trace the same shape:\nno parallax, no depth", "range (m)"
        lc = _traces(ax, e["B"], e["s"], inv, n=40)
        cb = fig.colorbar(lc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(lab)
        ax.set_xlim(180, -180)
        ax.set_ylim(e["s"][-1], 0)
        ax.set_xticks(range(-180, 181, 90))
        ax.set_xlabel("bearing, deg (+ left)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{name}\n{note}", loc="left", fontsize=10)
    fig.suptitle("One plane at eye height, the same first look, three motions: each row is the eye-height row of the "
                 "panorama at that moment (re-projected)", x=0.01, ha="left", fontsize=12)
    p = Path(out) / "compare_primitives.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_invariants(out):
    """Figure 3: what each one estimates, over the first look's panorama: z (M4's axial depth, within 45°), D, rho."""
    from matplotlib.colors import LinearSegmentedColormap
    from .figures import _plt, BLUE, GRID
    plt = _plt()
    rec, fan = pano_ref()
    c0 = np.asarray(rec.head_origin_w, float)
    Rwh = np.asarray(rec.head_R_wh, float)
    fwd = -Rwh[:, 2]
    phi0 = float(np.arctan2(fwd[1], fwd[0]))
    P = rec.as_raster("position_w", fill=np.nan).astype(np.float64)
    d = P - c0
    z = d @ fwd
    lat = np.linalg.norm(d - z[..., None] * fwd, axis=-1)
    e = np.degrees(np.arctan2(lat, z))
    zmap = np.where((z > 0) & (e <= 45), z, np.nan)
    rho, _, centre = arc_invariant(P.reshape(-1, 3), c0, phi0, 1.0)
    cmap = LinearSegmentedColormap.from_list("b", list(BLUE[::-1]))
    cmap.set_bad(GRID)
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.9), gridspec_kw={"wspace": 0.12})
    for ax, (m, name) in zip(axs, ((zmap, "M4: axial depth z = 1/ζ, within 45° of the axis\n(changes with every step)"),
                                   (lat, "line: D, the distance to the path line\n(the same at every step)"),
                                   (rho.reshape(lat.shape), "arc (R = 1 m, left): ρ, the distance to the turn's centre\n(the same at every step)"))):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=5, extent=(180, -180, -90, 90), aspect="auto",
                       interpolation="nearest")
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xticks(range(-180, 181, 90))
        ax.set_yticks(range(-90, 91, 45))
        ax.set_xlabel("bearing, deg (+ left)")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01)
    cb.set_label("m (dark = near)")
    fig.suptitle("What each representation estimates, over the first look's panorama (true values; the pivot has none)",
                 x=0.01, ha="left", fontsize=12, y=1.04)
    p = Path(out) / "compare_invariants.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def run(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    return [fig_straight(out), fig_primitives(out), fig_invariants(out)]
