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
