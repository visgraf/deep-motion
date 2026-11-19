"""Schematic figures for the Moebius Traces tech note (lvelho/foveal-depth-tr), from formulas only (no renders).

- traces_line: four points beside a straight path; their traces in M4's q = ln tan e and in the line's u = cot psi.
- traces_primitives: the same four points, seen under a line, an arc (R = 1 m, 90 deg left) and a turn in place:
  top views and bearing against progress (the turn-in-place diagonal drawn on the arc).
- sphere_fields: image motion on the viewing sphere: the turn (elliptic), the unit hyperbolic Moebius field, a
  translation (the same field weighted by 1/r), and an arc (their sum).
These are the static versions of the two widgets shown in Chat on 2026-11-18 (#11).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .figures import _plt, _clean, BLUE, ACCENT, INK, INK2, GRID

PTS = [("A", 2.0, 0.5, ACCENT, "-", "o"), ("B", 3.0, 0.5, ACCENT, "--", "s"), ("C", 2.0, 0.25, BLUE[4], "-", "^"),
       ("E", 2.0, 1.0, BLUE[1], "-", "D")]          # name, distance ahead x, distance aside D, colour, line, marker


def traces_line(out):
    plt = _plt()
    T = 1.5
    t = np.linspace(0, T, 400)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), gridspec_kw={"width_ratios": [0.6, 1, 1], "wspace": 0.32})
    ax = axs[0]
    ax.axvline(0, color=INK2, lw=1)
    ax.annotate("", xy=(0, 1.55), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    ax.text(-0.05, 0.75, "path", rotation=90, ha="right", va="center", color=INK2, fontsize=9)
    for n, x, D, c, ls, mk in PTS:
        ax.plot(D, x, mk, color=c, ms=8)
        ax.plot([0, D], [0, x], ls, color=c, lw=0.8, alpha=0.6)
        ax.text(D + 0.05, x, n, va="center", fontsize=10)
    ax.plot(0, 0, "k^", ms=9)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.2, 3.3)
    ax.set_aspect("equal")
    ax.set_xlabel("aside, m")
    ax.set_ylabel("ahead, m")
    ax.set_title("(a) top view: the eye at t = 0", loc="left", fontsize=10)
    _clean(ax)
    for ax, kind in ((axs[1], "q"), (axs[2], "u")):
        for n, x, D, c, ls, mk in PTS:
            z = x - t
            if kind == "q":
                v = np.where((z > 0) & (D / np.maximum(z, 1e-9) <= 1), np.log(D / np.maximum(z, 1e-9)), np.nan)
            else:
                v = z / D
            ax.plot(v, t, ls, color=c, lw=2, label=n)
            j = np.flatnonzero(np.isfinite(v))
            if len(j):
                ax.plot(v[j[0]], t[j[0]], mk, color=c, ms=6)
        ax.set_ylim(T, 0)
        ax.set_ylabel("t, m travelled")
        _clean(ax)
    axs[1].set_xlim(-2.4, 0)
    axs[1].set_xlabel("q = ln tan e   (ahead → 45°)")
    axs[1].set_title("(b) M4's log-tangent: curved traces;\nequal axial depth (A, C, E) = one curve shifted in q",
                     loc="left", fontsize=10)
    axs[2].set_xlim(8, 0)
    axs[2].set_xlabel("u = cot ψ   (ahead → aside)")
    axs[2].set_title("(c) the line's u = cot ψ: straight traces, slope −1/D;\nequal D (A, B) = one line shifted in t",
                     loc="left", fontsize=10)
    axs[2].legend(frameon=False, fontsize=9, loc="lower left")
    return _save(fig, out, "traces_line")


def _pose(kind, s, R=1.0):
    if kind == "line":
        return np.array([0.0, s]), np.pi / 2
    if kind == "arc":
        return np.array([-R + R * np.cos(s / R), R * np.sin(s / R)]), np.pi / 2 + s / R
    return np.array([0.0, 0.0]), np.pi / 2 + s


def _bearing(p, kind, s):
    c, h = _pose(kind, s)
    d = p - c
    b = np.arctan2(d[1], d[0]) - h
    return np.degrees(np.arctan2(np.sin(b), np.cos(b)))


def traces_primitives(out):
    plt = _plt()
    a70 = np.radians(70)
    pts = [("A", np.array([1.0, 1.5]), ACCENT, "-", "o"), ("B", np.array([1.0, 3.0]), ACCENT, "--", "s"),
           ("C", np.array([-1 + 2.5 * np.cos(a70), 2.5 * np.sin(a70)]), BLUE[4], "-", "^"),
           ("E", np.array([0.4, 0.8]), BLUE[1], "-", "D")]
    kinds = [("line", 1.5, "line, 1.5 m", "s, m"), ("arc", np.pi / 2, "arc, R = 1 m, 90° left", "θ = s / R, rad"),
             ("pivot", np.pi / 2, "turn in place, 90° left", "turn, rad")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6), gridspec_kw={"height_ratios": [0.85, 1], "hspace": 0.38,
                                                                  "wspace": 0.28})
    for col, (kind, L, name, ylab) in enumerate(kinds):
        ax = axs[0, col]
        ss = np.linspace(0, L, 200)
        if kind != "pivot":
            cc = np.array([_pose(kind, s)[0] for s in ss])
            ax.plot(cc[:, 0], cc[:, 1], color=INK, lw=2)
        if kind == "arc":
            ax.plot(-1, 0, "+", color=INK2, ms=10)
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(-1 + 2.5 * np.cos(th), 2.5 * np.sin(th), ":", color=ACCENT, lw=0.9)
            ax.text(-1, -0.25, "centre", ha="center", va="top", fontsize=8.5, color=INK2)
        if kind == "line":
            ax.axvline(1.0, ls=":", color=ACCENT, lw=0.9)
        ax.plot(0, 0, "k^", ms=8)
        for n, p, c, ls, mk in pts:
            ax.plot(*p, mk, color=c, ms=7)
            ax.text(p[0] + 0.08, p[1], n, va="center", fontsize=9.5)
        ax.set_xlim(-2.4, 1.6)
        ax.set_ylim(-0.6, 3.4)
        ax.set_aspect("equal")
        ax.set_title(f"({'abc'[col]}) {name}", loc="left", fontsize=10)
        _clean(ax)
        ax = axs[1, col]
        ss = np.linspace(0, L, 600)
        if kind == "arc":
            ref = np.degrees(np.angle(np.exp(1j * (np.radians(0) - ss))))
            ax.plot(-np.degrees(ss), ss, color=GRID, lw=6, alpha=0.9, solid_capstyle="butt",
                    label="turn in place (reference)")
        for n, p, c, ls, mk in pts:
            b = np.array([_bearing(p, kind, s) for s in ss])
            b[np.abs(np.diff(b, prepend=b[0])) > 90] = np.nan
            ax.plot(b, ss, ls, color=c, lw=2, label=n)
        ax.set_xlim(180, -180)
        ax.set_ylim(L, 0)
        ax.set_xticks([180, 90, 0, -90, -180])
        ax.set_xticklabels(["behind", "left", "heading", "right", "behind"])
        ax.set_ylabel(ylab)
        _clean(ax)
    axs[1, 0].set_title("equal D (A, B): one trace, B later in s", loc="left", fontsize=10)
    axs[1, 1].set_title("equal ρ (A, C): one trace, C later in θ;\nnear points (E) bend away from the turn",
                        loc="left", fontsize=10)
    axs[1, 2].set_title("every point: the same diagonal, no depth", loc="left", fontsize=10)
    axs[1, 1].legend(frameon=False, fontsize=8.5, loc="lower right")
    return _save(fig, out, "traces_primitives")


def _sphere_axes(ax, title):
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=INK2, lw=0.8)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10)


def sphere_fields(out, seed=2):
    """Image-motion fields on the visible hemisphere of the viewing sphere, seen from outside along -view."""
    plt = _plt()
    rng = np.random.default_rng(seed)
    # view: from front-right-above; heading f = +y (into the page's upper right), vertical z
    az, el = np.radians(55), np.radians(22)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])     # towards the viewer
    e1 = np.cross([0, 0, 1.0], view)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(view, e1)
    proj = lambda P: np.stack([P @ e1, P @ e2], -1)
    # sample directions on a Fibonacci sphere, keep the visible side
    n = 520
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)
    d = d[d @ view > 0.08]
    f = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    rot = np.cross(up, d) * -1.0                                     # a left turn: the scene sweeps right
    hyp = (d @ f)[:, None] * d - f                                    # unit hyperbolic generator, away from the FOE
    r = np.exp(rng.uniform(np.log(0.8), np.log(5.0), len(d)))          # ranges, 0.8-5 m
    tra = hyp / r[:, None]
    arc = tra + rot
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.4), gridspec_kw={"wspace": 0.05})
    panels = [(rot, "(a) turn in place: elliptic\n(a Möbius map; no depth)", BLUE[4]),
              (hyp, "(b) hyperbolic Möbius generator\n(meridians from the FOE; a shift in ℓ)", INK2),
              (tra, "(c) translation: (b) weighted by 1/r\n(same directions, depth in the speed)", ACCENT),
              (arc, "(d) arc: (a) + (c)", BLUE[5])]
    P0 = proj(d)
    for ax, (V, title, c) in zip(axs, panels):
        _sphere_axes(ax, title)
        V2 = proj(d + 0.22 * V) - P0
        ax.quiver(P0[:, 0], P0[:, 1], V2[:, 0], V2[:, 1], angles="xy", scale_units="xy", scale=1, color=c,
                  width=0.004, headwidth=3.5, headlength=4)
        for vec, lab in ((f, "FOE"), (-f, "")):
            if vec @ view > 0:
                q = proj(vec[None])[0]
                ax.plot(*q, "o", color=INK, ms=5)
                ax.text(q[0] + 0.05, q[1] + 0.05, lab, fontsize=9)
        q = proj(up[None])[0]
        ax.plot(*q, "+", color=INK, ms=8)
    axs[0].text(proj(up[None])[0][0] + 0.06, proj(up[None])[0][1] + 0.04, "vertical", fontsize=8.5, color=INK2)
    return _save(fig, out, "sphere_fields")


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths.append(p)
    _plt().close(fig)
    return paths


def run(out):
    return traces_line(out) + traces_primitives(out) + sphere_fields(out)
