"""Figures for deep-motion's stages (matplotlib; palette: the dataviz reference palette, as foveal-motion's M6 figures).

    stage2_primitives.png   three panels: a line's traces in its pencil coordinates (u against s: straight, slope
                            -1/D); an arc's traces in concentric-mosaic coordinates (theta - phi_P against beta: one
                            curve per rho, the same for every point at that rho); a composed path from above, with the turn centre
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import primitives as pr

BLUE = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
ACCENT = "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 10.5})
    return plt


def _clean(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def stage2(out) -> list:
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"wspace": 0.28})

    # 1. line: u = (x_P - s) / D, the epipolar-plane image of the pencil
    ax = axs[0]
    s = np.linspace(0, 2.0, 50)
    for D, col in zip((0.5, 1.0, 2.0, 4.0), BLUE[1::1]):
        for x_P in (2.5, 3.5):
            ax.plot(pr.line_trace_u(x_P, D, s), s, color=col, lw=2, label=f"D = {D:g} m" if x_P == 2.5 else None)
    ax.set_xlabel("u = cot ψ  (ray coordinate within the epipolar plane)")
    ax.set_ylabel("s, distance travelled (m)")
    ax.set_title("Line: every trace is straight, slope −1/D", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _clean(ax)

    # 2. arc: theta - phi_P = -delta(beta, rho): the same curve for every point at radius rho
    ax = axs[1]
    R = 1.6
    betas = np.radians(np.linspace(-85, 85, 341))
    for rho, col in zip((R + 0.5, R + 1.5, R + 4.0, R + 10.0), BLUE[2:]):
        d = pr.arc_slit_delta(betas, rho, R, +1)
        ax.plot(np.degrees(betas), -np.degrees(d), color=col, lw=2, label=f"ρ = {rho:g} m")
    ax.set_xlabel("β, slit angle from the tangent (deg, + toward the centre)")
    ax.set_ylabel("θ − φ_P: where on the circle slit β sees the point (deg)")
    ax.set_title(f"Arc (R = {R:g} m): a trace depends on ρ alone", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _clean(ax)

    # 3. a composed path from above
    ax = axs[2]
    pieces = [pr.Piece("line", 1.2), pr.Piece("arc", 1.5, 1 / 1.6), pr.Piece("pivot", np.radians(-30)), pr.Piece("line", 1.0)]
    path = pr.Path(pieces, p0=(0.0, 0.0), phi0=np.radians(90), h=1.6)
    kinds = {"line": INK2, "arc": BLUE[3], "pivot": ACCENT}
    for i, s_, c, R_ in path.poses(step=0.1, pivot_step=np.radians(6)):
        k = pieces[i].kind
        fwd = -R_[:, 2]
        ax.plot([c[0], c[0] + 0.12 * fwd[0]], [c[1], c[1] + 0.12 * fwd[1]], color=kinds[k], lw=1.2)
        ax.plot(c[0], c[1], "o", ms=3.5, color=kinds[k])
    cen = pieces[1].centre(*path.starts[1])
    ax.plot(cen[0], cen[1], "x", ms=9, mew=2, color=BLUE[3])
    ax.annotate("turn axis (blind)", cen, xytext=(8, -14), textcoords="offset points", color=INK2, fontsize=9)
    for k, col in kinds.items():
        ax.plot([], [], "o-", color=col, label=k)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    ax.set_title("A composed path: line, arc, pivot, line (ticks: heading)", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
    _clean(ax)
    fig.savefig(out / "stage2_primitives.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return sorted(str(p) for p in out.glob("stage2_*.png"))


def stage3(out) -> list:
    """stage3_atlas.png: sigma_r/r around the head (polar: bearing, log range) for a line and two arcs;
    stage3_profiles.png: sigma_r/r against bearing at 3 m, and the largest time step against the angle from travel."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from . import atlas as at
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    A = at.atlas()
    th = np.radians(at.BEARINGS_DEG)
    rr = np.log10(at.RANGES)
    cmap = LinearSegmentedColormap.from_list("seq", ("#0d366b", "#256abf", "#6da7ec", "#cde2fb"))
    cmap.set_bad("#b9b7b0")
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.6), subplot_kw={"projection": "polar"})
    norm = LogNorm(1e-3, 1e-1)
    for ax, (name, R) in zip(axs, at.CASES):
        S = np.ma.masked_invalid(A[name])
        T, Rm = np.meshgrid(th, rr, indexing="ij")
        im = ax.pcolormesh(T, Rm - rr[0], S, cmap=cmap, norm=norm, shading="nearest")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_rticks(np.log10([1, 3, 10]) - rr[0])
        ax.set_yticklabels(["1 m", "3 m", "10 m"], color=INK2, fontsize=8.5)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["heading", "left", "behind", "right"], color=INK2, fontsize=9)
        ax.grid(color=GRID, lw=0.5)
        ax.set_title(name + (" (turning left)" if R else ""), color=INK, pad=14)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.04)
    cb.set_label("σ_r / r (log; darker = more precise)", color=INK2)
    fig.suptitle("Stage 3 atlas: range precision around the head, 1 m window, 1 cm steps, 0.1° resolution "
                 "(grey: blind; radius: range, log)", color=INK, y=1.0, fontsize=11)
    fig.savefig(out / "stage3_atlas.png", dpi=125, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"wspace": 0.25})
    ax = axs[0]
    j3 = int(np.argmin(np.abs(at.RANGES - 3.0)))
    for (name, R), col, ls in zip(at.CASES, (INK2, BLUE[2], BLUE[4]), ("-", "--", "-")):
        ax.plot(at.BEARINGS_DEG, A[name][:, j3], ls, color=col, lw=2, label=name)
    ax.set_yscale("log"), ax.set_ylim(1e-3, 0.3)
    ax.set_xticks([-180, -90, 0, 90, 180]), ax.set_xticklabels(["behind", "right", "heading", "left", "behind"])
    ax.set_ylabel("σ_r / r at 3 m (log)")
    ax.set_title("At 3 m: the arcs fill the line's blind heading", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _clean(ax)
    ax = axs[1]
    psi = np.radians(np.linspace(2, 90, 200))
    for (lo, hi, lab), col in zip(((0.5, 20.0, "prior band 0.5–20 m"), (2.7, 3.3, "converged ±10 % at 3 m")), (BLUE[4], ACCENT)):
        ax.plot(np.degrees(psi), 100 * at.ds_max(psi, lo, hi), color=col, lw=2, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("angle from the direction of travel ψ (deg)")
    ax.set_ylabel("largest step for ≤ 1 resolution element (cm, log)")
    ax.set_title("Time step: millimetres while uncertain, centimetres once converged", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _clean(ax)
    fig.savefig(out / "stage3_profiles.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return sorted(str(p) for p in out.glob("stage3_*.png"))


def stage4(run_dir, out) -> list:
    """stage4_slit_fan.png: the carrier's panorama of the calibration room from the arc's first look, with slits marked
    (columns at bearings from the heading) and each slit's time step from the schedule (prior band; converged +/-10 % at
    3 m)."""
    from fov_render import exr_lite
    from . import sensor
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fan = sensor.SlitFan()
    x = exr_lite.read_uncompressed_exr(str(Path(run_dir) / "k0000" / "raw_C.exr"))
    rgb = np.stack([x[f"ViewLayer.Combined.{c}"] for c in "RGB"], -1)
    rgb = np.clip(rgb / max(np.percentile(rgb, 99.5), 1e-9), 0, 1) ** (1 / 2.2)
    W, H = fan.width, fan.width // 2
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.imshow(rgb, extent=(180, -180, -90, 90))         # bearing + left: left edge of the picture = +180
    for b in (-150, -120, -90, -60, -30, -12, 12, 30, 60, 90, 120, 150):
        psi = np.radians(abs(b))
        prior, conv = 100 * fan.step(psi), 100 * fan.step(psi, (2.7, 3.3))
        ax.axvline(b, color=ACCENT, lw=1.2, alpha=0.9)
        ax.text(b, 84, f"{b:+d}°", ha="center", va="top", color="white", fontsize=9, fontweight="bold")
        ax.text(b, -62, f"{prior:.2g} cm\n{conv:.2g} cm", ha="center", va="center", color="white", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="#0d366b", ec="none", alpha=0.85))
    ax.axvline(0, color="white", lw=1.0, ls="--")
    ax.set_xlim(180, -180)
    ax.set_xticks([180, 90, 0, -90, -180]), ax.set_xticklabels(["behind", "left 90°", "heading", "right 90°", "behind"])
    ax.set_yticks([-90, -45, 0, 45, 90]), ax.set_yticklabels(["−90°", "−45°", "0°", "45°", "90°"])
    ax.set_ylabel("elevation")
    ax.set_title("Stage 4: the slit fan on fov-render's Panorama (calibration room, start of an arc, R = 1.6 m, 0.5°/px). "
                 "Each column is a slit; boxes: its largest step while 0.5–20 m uncertain / once ±10 % at 3 m",
                 color=INK, loc="left", fontsize=10)
    fig.savefig(out / "stage4_slit_fan.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return sorted(str(p) for p in out.glob("stage4_*.png"))


def stage5a(runs: dict, out, stride=1) -> list:
    """stage5a_maps.png: the oracle's 95 % half-interval (relative, 0.25 px) over the reference panorama of each run,
    the usable region (<= 10 %) outlined; stage5a_profile.png: its median against the 3-D angle from the heading."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from . import oracle as orc
    from . import sensor as sn
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("blue", list(BLUE[::-1]))
    cmap.set_bad(GRID)
    evs = {k: orc.evaluate(d, sigmas=(0.25,), stride=stride) for k, d in runs.items()}
    fig, axs = plt.subplots(len(evs), 1, figsize=(10.5, 2.9 * len(evs)), gridspec_kw={"hspace": 0.45})
    # the floor follows the runs (#6: at 0.1 deg/px most of the sphere sat below the fixed 0.005): the 2nd percentile
    floors = []
    for e in evs.values():
        with np.errstate(divide="ignore"):
            c = orc.Z95 * 0.25 * e["res"] / np.sqrt(e["I"]) / e["rtrue"]
        floors.append(np.percentile(c[np.isfinite(c)], 2))
    lo = min(floors)
    norm = LogNorm(0.005 if lo >= 0.005 else 10 ** np.floor(np.log10(lo)), 2.0)
    for ax, (k, e) in zip(np.atleast_1d(axs), evs.items()):
        W, H = e["width"], e["width"] // 2
        img = np.full(H * W, np.nan)
        with np.errstate(divide="ignore"):
            ci = orc.Z95 * 0.25 * e["res"] / np.sqrt(e["I"]) / e["rtrue"]
        val = np.where(np.isfinite(ci), ci, np.nan)
        if stride > 1:
            # Strided anchors are every stride-th hit in raster order (at width 3600 and stride 25, every 25th column):
            # each pixel shows its nearest anchor in raster order, i.e. along its row (#6). Assumes every pixel is a hit,
            # as in the calibration room; a miss would be painted with its neighbour.
            o = np.argsort(e["pix"])
            p, v = e["pix"][o], val[o]
            q = np.arange(H * W)
            j = np.clip(np.searchsorted(p, q), 1, len(p) - 1)
            j -= (q - p[j - 1]) < (p[j] - q)
            img = v[j]
        else:
            img[e["pix"]] = val
        img = img.reshape(H, W)
        im = ax.imshow(img, cmap=cmap, norm=norm, extent=(180, -180, -90, 90), aspect="auto", interpolation="nearest")
        ax.contour(np.linspace(180, -180, W), np.linspace(90, -90, H), np.nan_to_num(img, nan=9.0), levels=[orc.USABLE_REL],
                   colors=[ACCENT], linewidths=1.2)
        ax.set_title(f"{k}: 95 % half-interval of range, 0.25 px (orange: ±10 %, usable inside the darker regions)", loc="left")
        thin = f"; anchors every {stride}th pixel, each pixel its nearest along the row" if stride > 1 else ""
        ax.set_xlabel(f"bearing from the heading, deg (+ left){thin}")
        ax.set_ylabel("elevation, deg")
        ax.set_xticks(range(-180, 181, 45))
        ax.set_yticks(range(-90, 91, 45))
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("σ-based 95 % half-interval / range")
    f1 = out / "stage5a_maps.png"
    fig.savefig(f1, dpi=110, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    edges = np.array([0, 2, 5, 10, 15, 25, 35, 45, 60, 75, 90, 105, 120, 135, 145, 155, 165, 170, 175, 178, 180.0])
    mid = 0.5 * (edges[1:] + edges[:-1])
    styles = {"line": (BLUE[3], "o"), "arc": (ACCENT, "s")}
    for k, e in evs.items():
        with np.errstate(divide="ignore"):
            ci = orc.Z95 * 0.25 * e["res"] / np.sqrt(e["I"]) / e["rtrue"]
        psi = np.degrees(e["psi3"])
        med = [np.median(ci[(psi >= a) & (psi < b)]) if np.any((psi >= a) & (psi < b)) else np.nan for a, b in zip(edges[:-1], edges[1:])]
        col, mk = styles["arc" if k.startswith("arc") else "line"]
        ls = "-" if "50" in k else "--"
        ax.plot(mid, med, ls, color=col, marker=mk, ms=4, lw=2, label=k)
    ax.axhline(orc.USABLE_REL, color=INK2, lw=1, ls=":")
    ax.text(178, orc.USABLE_REL * 1.12, "usable: ±10 %", ha="right", color=INK2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xlabel("3-D angle from the heading, deg")
    ax.set_ylabel("median 95 % half-interval / range")
    res_deg = np.degrees(next(iter(evs.values()))["res"])
    ax.set_title(f"Oracle on renders, 0.25 px ({0.25 * res_deg:.3g}°): median by angle from the heading", loc="left")
    _clean(ax)
    ax.legend(frameon=False, fontsize=9)
    f2 = out / "stage5a_profile.png"
    fig.savefig(f2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return [f1, f2]


def stage5b2(runs: dict, out, sigma_px=0.5, hit_stride=1) -> list:
    """stage5b2_topdown.png: the cells from above, usable at sigma_px (dark) or not (orange), per path, with the path;
    stage5b2_progress.png: usable fraction against arc length (from the checks' records); stage5b2_cones.json."""
    import json as _json
    from . import fusion as fu
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    maps = {k: fu.usable_map(v, sigma_px, hit_stride=hit_stride) for k, v in runs.items()}
    fig, axs = plt.subplots(1, len(maps), figsize=(6.2 * len(maps), 6.4), gridspec_kw={"wspace": 0.15})
    for ax, (k, m) in zip(np.atleast_1d(axs), maps.items()):
        P, u = m["P"], m["usable"]
        o = np.argsort(P[:, 2])                      # floor first, so walls draw over it
        ax.scatter(P[o][u[o], 0], P[o][u[o], 1], s=0.15, c=BLUE[3], lw=0, rasterized=True)
        ax.scatter(P[o][~u[o], 0], P[o][~u[o], 1], s=0.6, c=ACCENT, lw=0, rasterized=True)
        ax.plot(m["path"][:, 0], m["path"][:, 1], color=INK, lw=2.2)
        ax.set_aspect("equal")
        ax.set_title(f"{k} path: {100 * u.mean():.1f} % usable at {sigma_px:g} px\n(blue usable, orange not; black: the path)", loc="left")
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        _clean(ax)
    f1 = out / "stage5b2_topdown.png"
    fig.savefig(f1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    # the checks' records beside the figures' folder: stage5b2.json for the composed path, stage5b2_<path>.json else
    # (#7, Code: the turn's and the long straight's records were not found under the first two names)
    recs = {k: out.parent / ("stage5b2.json" if k == "composed" else f"stage5b2_{k}.json") for k in runs}
    if all(p.exists() for p in recs.values()):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for (k, p), col, ls in zip(recs.items(), (ACCENT, BLUE[3], BLUE[1]), ("-", "--", ":")):
            pr_ = _json.loads(p.read_text())["numbers"]["progress_0.5px"]
            ax.plot(pr_["s_m"], 100 * np.asarray(pr_["usable"]), ls, color=col, lw=2, label=f"{k} path")
        for x in (0.3, 0.6):
            ax.axvline(x, color=GRID, lw=1)
        ax.text(0.15, 3, "line", ha="center", color=INK2, fontsize=9)
        ax.text(0.45, 3, "arc (composed)", ha="center", color=INK2, fontsize=9)
        ax.text(0.75, 3, "line", ha="center", color=INK2, fontsize=9)
        ax.set_xlabel("arc length travelled, m")
        ax.set_ylabel("cells usable at 0.5 px, %")
        ax.set_title("Fused surface: usable fraction as the path is travelled (records of the checks)", loc="left")
        ax.set_ylim(0, 100)
        _clean(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        f2 = out / "stage5b2_progress.png"
        fig.savefig(f2, dpi=130, bbox_inches="tight")
        plt.close(fig)
    summ = {k: fu.cone_summary(m) for k, m in maps.items()}
    (out / "stage5b2_cones.json").write_text(_json.dumps({"sigma_px": sigma_px, "runs": runs, "cones": summ}, indent=1,
                                                         ensure_ascii=False))
    return [p for p in (f1, out / "stage5b2_progress.png", out / "stage5b2_cones.json") if p.exists()]
