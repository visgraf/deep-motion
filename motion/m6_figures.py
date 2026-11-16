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


# ----------------------------------------------------------------------------- T1 (#22): pure rotation

def _tone(rgb):
    x = np.clip(np.nan_to_num(rgb, nan=0.0), 0, None)
    x = x / max(np.percentile(x, 99.5), 1e-9)
    return np.clip(x, 0, 1) ** (1 / 2.2)


def t1_figures(run_dir, res, out) -> list:
    """t1_residual.png: the photometric residual of derotation against the reference's eccentricity, per yaw,
    with the noise floor; t1_fovea.png: the reference, frames derotated into it, and the residual images."""
    from . import analysis, flow_depth, m6_rotation as m6r, matcher
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per = [p for p in res["per_look"] if not p["noise_repeat"]]
    floor = next(p for p in res["per_look"] if p["noise_repeat"])
    mids = lambda bs: [np.sqrt(max(b["bin_deg"][0], 0.5) * b["bin_deg"][1]) for b in bs]

    # 1. residual against e_r
    ramp = BLUE_RAMP[2:]                                       # ordinal: start no lighter than step 300
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for p, col in zip(per, np.array(ramp)[np.linspace(0, len(ramp) - 1, len(per)).round().astype(int)]):
        bs = p["residual_by_e_r"]
        ax.plot(mids(bs), [b["median"] for b in bs], "-o", color=col, lw=2, ms=6, mec=SURFACE, mew=1.2,
                label=f"yaw {p['yaw_deg']:g}°")
    bs = floor["residual_by_e_r"]
    ax.plot(mids(bs), [b["median"] for b in bs], "--", color=INK2, lw=1.5, label="noise floor (same pose, new seed)")
    ax.axvspan(0.4, 2.0, color=GRID, alpha=0.5, lw=0)
    ax.text(0.9, 0.205, "fovea", ha="center", color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1.4, 3.2, 7.1, 14, 24, 37]), ax.set_xticklabels(["0–1", "1–2", "2–5", "5–10", "10–20", "20–30", "30–45"])
    ax.set_xlim(0.4, 50), ax.set_ylim(0, 0.23)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("eccentricity in the reference (deg)")
    ax.set_ylabel("median |L_k − L_r| / L_r after derotation")
    ax.legend(frameon=False, labelcolor=INK, loc="upper right", fontsize=9)
    ax.set_title(f"T1, pure rotation (calibration room, {res['spp']} spp): the reference's fovea, seen by a turned "
                 "frame, loses detail", color=INK, loc="left", fontsize=10)
    fig.savefig(out / "t1_residual.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2. images on the reference's raster
    run = analysis.load_run(run_dir)
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    yaws = run["t1"]["yaws_deg"]
    ob_r = flow_depth.load_observation(run, es[0])
    model = ob_r.model
    rays = model.rays()
    valid = rays.valid
    d = lp_unit(rays.direction[valid].astype(np.float64))
    show = [k for k in es if yaws[k] in (10.0, 30.0) and k != run["t1"]["noise_repeat"]]
    Lr = matcher.luminance(ob_r.rgb)
    fig, axs = plt.subplots(2, 1 + len(show), figsize=(4.0 * (1 + len(show)), 8.2))
    axs[0, 0].imshow(_tone(ob_r.rgb)), axs[0, 0].set_title("reference (yaw 0)", color=INK)
    axs[1, 0].imshow(np.abs(matcher.luminance(flow_depth.load_observation(run, es[run["t1"]["noise_repeat"]]).rgb) - Lr)
                     / np.where(Lr > 0, Lr, np.nan), cmap=_cmap(), vmin=0, vmax=0.5)
    axs[1, 0].set_title("|ΔL|/L: same pose, new seed (noise)", color=INK)
    for j, k in enumerate(show, 1):
        ob_k = flow_depth.load_observation(run, es[k])
        dk = m6r.dr.derotate(d, ob_r.R, ob_k.R)
        rc, inside = model.pixel_of(dk)
        img = np.full(ob_r.rgb.shape, np.nan)
        vals = np.full((len(d), 3), np.nan)
        for c in range(3):
            vals[inside, c] = m6r._bilinear(ob_k.rgb[..., c], rc[inside])
        img[valid] = vals
        axs[0, j].imshow(_tone(img)), axs[0, j].set_title(f"yaw {yaws[k]:g}°, derotated into the reference", color=INK)
        L = matcher.luminance(img)
        im = axs[1, j].imshow(np.abs(L - Lr) / np.where(Lr > 0, Lr, np.nan), cmap=_cmap(), vmin=0, vmax=0.5)
        axs[1, j].set_title(f"|ΔL|/L, yaw {yaws[k]:g}°", color=INK)
    for a in axs.ravel():
        a.set_axis_off()
    fig.colorbar(im, ax=axs[1, :], fraction=0.02, pad=0.01).set_label("|ΔL| / L", color=INK2)
    fig.suptitle("T1 on the Foveated raster: grey = outside the turned frame's field; darker = larger residual",
                 color=INK, y=0.995)
    fig.savefig(out / "t1_fovea.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return sorted(str(p) for p in out.glob("t1_*.png"))


def lp_unit(v):
    return lp.unit(v)


# ----------------------------------------------------------------------------- T2 (#22): the crab dolly

def t2_figures(results: dict, out, ref=12) -> list:
    """results: {alpha_deg: {"run": run_dir, "est": dir with rgb/r####.npz, "oracle": {...}, "rgb": {...}}}.
    t2_summary.png: usable fractions against alpha (O-M at 0.25 px and RGB, field and fovea);
    t2_maps.png: for reference `ref`, M6's usable anchors on the Foveated raster, coloured by the 95 % interval's
    relative half-width, with the window's epipoles (as t0_field_maps)."""
    from matplotlib.colors import LogNorm
    from . import analysis, flow_depth, m6_matcher as m6, multibaseline as mb
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    al = sorted(results)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    series = (("O-M 0.25 px, field", lambda R: R["oracle"]["0.25"]["field_usable"], SERIES[0], "--", "o"),
              ("O-M 0.25 px, fovea < 2°", lambda R: R["oracle"]["0.25"]["fovea_usable"], SERIES[1], "--", "s"),
              ("M6 RGB, field", lambda R: R["rgb"]["field_usable_of_all_hits"], SERIES[0], "-", "o"),
              ("M6 RGB, fovea 1–2°", lambda R: R["rgb"]["fovea_usable_1_2deg"], SERIES[1], "-", "s"))
    for name, f, col, ls, mk in series:
        ax.plot(al, [f(results[a]) for a in al], ls, marker=mk, color=col, lw=2, ms=7, mec=SURFACE, mew=1.2, label=name)
    ax.set_xticks(al), ax.set_xticklabels([f"{a:g}°" for a in al])
    ax.set_xlabel("crab angle α (the epipole's distance from the fovea)")
    ax.set_ylabel("usable fraction (95 % interval ≤ 10 %)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="center right")
    ax.set_title("T2, crab dolly (calibration room, 256 spp, both5, r = 8, 12, 16): moving the epipole off the fovea",
                 color=INK, loc="left", fontsize=10)
    fig.savefig(out / "t2_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    from fov_render.sensors import FoveatedModel
    model = FoveatedModel(253)
    fig, axs = plt.subplots(1, len(al), figsize=(4.6 * len(al), 4.6), gridspec_kw={"wspace": 0.05})
    norm, cmap = LogNorm(0.003, 0.1), _cmap().reversed()
    cmap.set_bad(GRID)
    fov = _circle_px(model, 2.0)
    for ax, a in zip(np.atleast_1d(axs), al):
        R = results[a]
        z = dict(np.load(Path(R["est"]) / f"r{ref:04d}.npz"))
        half = mb.Z95 * 1.4875 * z["sigma_zeta"] / z["zeta"]            # relative half-width in zeta ~ in range
        use = (z["reason"] == 0) & np.isfinite(half) & (half <= 0.10)
        img = np.full(model.n * model.n, np.nan)
        img[z["flat"][use]] = half[use]
        ax.imshow(img.reshape(model.n, model.n), cmap=cmap, norm=norm, interpolation="nearest")
        ax.plot(fov[:, 1], fov[:, 0], ls="--", lw=1.0, color=INK2)
        run = analysis.load_run(R["run"])
        es = {e["k"]: e for e in run["looks"]["ahead"]}
        ob = {k: flow_depth.load_observation(run, es[k]) for k in [ref + o for o in range(-5, 6)]}
        for o in [o for o in range(-5, 6) if o]:
            _, T = m6.relative_poses(ob[ref].c, ob[ref].R, [(ob[ref + o].c, ob[ref + o].R)])[0]
            hf = T / np.linalg.norm(T) * (1 if o > 0 else -1)
            rc, ok = model.pixel_of(hf[None])
            if ok[0]:
                ax.plot(rc[0, 1], rc[0, 0], "o", ms=5.5, mfc=SERIES[2] if o > 0 else "none", mec=SERIES[2] if o > 0 else SERIES[1], mew=1.6)
        ax.set_title(f"α = {a:g}°: usable {use.mean():.2f} of anchors", color=INK)
        ax.set_axis_off()
    fig.suptitle(f"M6 on the crab dolly, reference {ref}: usable anchors coloured by the interval's half-width "
                 "(darker = tighter; grey = rejected or not usable; anchors start at 1°). Dots: epipoles; dashed: the 2° fovea",
                 color=INK, y=1.02, fontsize=10)
    fig.savefig(out / "t2_maps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return sorted(str(p) for p in out.glob("t2_*.png"))
