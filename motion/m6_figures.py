"""M6 T0 figures (M6a #22): #21's information model drawn on the Foveated raster, from the poses alone (numpy).

    t0_field_maps.png     relative depth precision over the reference's field, on the Foveated `full` raster as the
                          sensor sees it, for the straight dolly, the crab (FOE 10 deg off) and arcs R = 4, 2, 1 m;
                          the window's epipoles marked (frames ahead filled, behind open), the 2 deg fovea dashed
    t0_profiles.png       the same, as the median over azimuth against eccentricity, with the 10-90 % band
    t0_resolution.png     what derotation costs: the sample spacing at which frame r + k sees the reference's fovea,
                          relative to the fovea's own, per arc radius

A model, not a measurement (#21): frontal plane at 3 m, per-frame matching noise proportional to the Foveated spacing
a (e + E2) in the reference and in frame k. "Precision" here is 1 / (sigma_rho / rho), normalised to the straight
dolly at e = 20 deg: 1 means as good as M4 at 20 deg; > 1 better.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import derotate as dr
from . import logpolar as lp

# reference palette (dataviz skill, validated light: categorical slots 1-5; sequential blue 100-700)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CASES = (("straight, ahead (M4)", "straight", 0.0), ("crab, FOE 10° off", "straight", 10.0),
         ("arc R = 4 m", "arc", 4.0), ("arc R = 2 m", "arc", 2.0), ("arc R = 1 m", "arc", 1.0))
D_PLANE = 3.0


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11})
    return plt


def sigma_rel_map(d, poses, r=0, D=D_PLANE, e2=dr.E2, e_max=dr.E_MAX, h=1e-6):
    """Vectorised dr.info_sigma_rel over reference eye-frame directions d (N, 3)."""
    d = lp.unit(d)
    e = np.arctan2(np.hypot(d[:, 0], d[:, 1]), -d[:, 2])
    rho = np.cos(e) / D
    c_r, R_r = poses[r]
    I = np.zeros(len(d))
    for k, (c_k, R_k) in poses.items():
        if k == r:
            continue
        d1 = dr.predict_dir(d, rho, c_r, R_r, c_k, R_k)
        d2 = dr.predict_dir(d, rho + h, c_r, R_r, c_k, R_k)
        ek = np.arctan2(np.hypot(d1[:, 0], d1[:, 1]), -d1[:, 2])
        g = dr._ang(d1, d2) / h
        I += np.where(ek <= e_max, g * g / ((e + e2) ** 2 + (ek + e2) ** 2), 0.0)
    with np.errstate(divide="ignore"):
        return np.where(I > 0, 1.0 / np.sqrt(I), np.inf) / rho


def _norm():
    phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    return float(np.mean([dr.info_sigma_rel(np.radians(20), p, dr.window_poses("straight", 0.0)) for p in phis]))


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)


def field_maps(out: Path, n=253):
    from fov_render.sensors import FoveatedModel
    from matplotlib.colors import LogNorm
    plt = _plt()
    model = FoveatedModel(n)
    rays = model.rays()
    valid = rays.valid
    d = lp.unit(rays.direction[valid].astype(np.float64))
    ref = _norm()
    # consistency with the selftested scalar model at a few rays (must agree to 1e-9)
    idx = np.linspace(0, len(d) - 1, 7).astype(int)
    P0 = dr.window_poses("straight", 10.0)
    vec = sigma_rel_map(d[idx], P0)
    sca = [dr.info_sigma_rel(*_e_phi(x), P0) for x in d[idx]]
    assert np.allclose(vec, sca, rtol=1e-9, atol=0), (vec, sca)

    fig, axs = plt.subplots(1, 5, figsize=(17, 4.3), gridspec_kw={"wspace": 0.06})
    norm, cmap = LogNorm(0.1, 10), _cmap()
    cmap.set_bad(GRID)
    fov = _circle_px(model, 2.0)
    for ax, (name, kind, par) in zip(axs, CASES):
        P = dr.window_poses(kind, par)
        prec = ref / sigma_rel_map(d, P)
        img = np.full(valid.shape, np.nan)
        img[valid] = np.where(np.isfinite(prec) & (prec > 0), prec, np.nan)
        im = ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
        ax.plot(fov[:, 1], fov[:, 0], ls="--", lw=1.0, color=SURFACE)
        for k, (c_k, _) in P.items():
            if k == 0:
                continue
            hk, _ = dr.epipole(P[0][0], P[0][1], c_k)
            hf = hk if k > 0 else -hk                          # the epipole's line, in the forward field
            rc, ok = model.pixel_of(hf[None])
            if ok[0]:
                ax.plot(rc[0, 1], rc[0, 0], "o", ms=5.5, mfc=SERIES[2] if k > 0 else "none",
                        mec=SERIES[2] if k > 0 else SERIES[1], mew=1.6)
        ax.set_title(name, color=INK)
        ax.set_axis_off()
        fov_med = np.nanmedian(prec[np.degrees(np.arctan2(np.hypot(d[:, 0], d[:, 1]), -d[:, 2])) < 2])
        ax.text(0.5, -0.04, f"fovea median {fov_med:.2f}", transform=ax.transAxes, ha="center", va="top", color=INK2)
    cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01)
    cb.set_label("depth precision (1 = M4 at 20°; log)", color=INK2)
    fig.suptitle("#21's model on the Foveated raster: where a both5 window gives depth (frontal plane at 3 m). "
                 "Dots: epipoles of frames ahead (filled) and behind (open); dashed: the 2° fovea; grey: no frame sees it",
                 color=INK, fontsize=11, y=1.02)
    fig.savefig(out / "t0_field_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _e_phi(x):
    return float(np.arctan2(np.hypot(x[0], x[1]), -x[2])), float(np.arctan2(x[1], x[0]))


def _circle_px(model, e_deg, n=256):
    a = np.linspace(0, 2 * np.pi, n)
    e = np.radians(e_deg)
    dd = np.stack([np.sin(e) * np.cos(a), np.sin(e) * np.sin(a), -np.cos(e) * np.ones_like(a)], -1)
    rc, _ = model.pixel_of(dd)
    return rc


def profiles(out: Path):
    plt = _plt()
    ref = _norm()
    es = np.geomspace(0.2, 44, 40)
    phis = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axvspan(0.2, 2.0, color=GRID, alpha=0.5, lw=0)
    ax.text(0.9, 0.13, "fovea", ha="center", color=INK2)
    styles = ("-", "--", "-", "-.", ":")
    for (name, kind, par), col, ls in zip(CASES, SERIES, styles):
        P = dr.window_poses(kind, par)
        dd = np.array([[np.sin(np.radians(e)) * np.cos(p), np.sin(np.radians(e)) * np.sin(p), -np.cos(np.radians(e))]
                       for e in es for p in phis])
        prec = (ref / sigma_rel_map(dd, P)).reshape(len(es), len(phis))
        med, lo, hi = np.median(prec, 1), np.percentile(prec, 10, 1), np.percentile(prec, 90, 1)
        ax.fill_between(es, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(es, med, ls, color=col, lw=2, label=name)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.2, 44), ax.set_ylim(0.1, 6)
    ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20, 40]), ax.set_xticklabels(["0.25", "0.5", "1", "2", "5", "10", "20", "40"])
    ax.set_yticks([0.1, 0.3, 1, 3]), ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.axhline(1, color=INK2, lw=0.8)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("eccentricity in the reference (deg, log)")
    ax.set_ylabel("depth precision (1 = M4 at 20°; log)")
    ax.set_title("#21's model: median over azimuth, 10–90 % band (frontal plane at 3 m, both5)", color=INK, loc="left")
    fig.savefig(out / "t0_profiles.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def resolution(out: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    radii = (8.0, 4.0, 2.0, 1.4, 1.0)
    ks = np.arange(1, 6)
    for R, col in zip(radii, SERIES):
        P = dr.arc_poses(range(0, 6), R)
        ratios = []
        for k in ks:
            d1 = dr.predict_dir(np.array([[0, 0, -1.0]]), np.array([1 / D_PLANE]), P[0][0], P[0][1], P[k][0], P[k][1])[0]
            ek = np.arctan2(np.hypot(d1[0], d1[1]), -d1[2])
            ratios.append((ek + dr.E2) / dr.E2)
        ax.plot(ks, ratios, "-o", color=col, lw=2, ms=7, mec=SURFACE, mew=1.5)
        ax.annotate(f"R = {R:g} m ({np.degrees(dr.STEP / R):.1f}°/step)", (ks[-1], ratios[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK)
    ax.set_xticks(ks), ax.set_xlim(0.8, 5.2), ax.set_ylim(0, 17)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("frame r + k")
    ax.set_ylabel("sample spacing where r + k sees r's fovea\n(× the fovea's own)")
    ax.set_title("What derotation can't remove: the far frames see the reference's fovea in their periphery",
                 color=INK, loc="left")
    fig.savefig(out / "t0_resolution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def all_figures(out) -> list:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    field_maps(out), profiles(out), resolution(out)
    return sorted(str(p) for p in out.glob("t0_*.png"))
