"""M4 process figures (M4a #10, Stage 6): what the multi-baseline estimator saw, decided and reconstructed, for one
reference frame r of a frozen calibration-room result (`m4_pipeline`), two windows side by side.

    overview_r####.png  r's native look with rings; per window: the decision per anchor, range (usable), the
                        relative interval half-width, frames used; usable pixels M4 vs M3's pair (r - 5, r); the
                        cloud from above; EVALUATION: relative range error
    curves_r####.png    for a few anchors, each window frame's NCC along zeta (thin, coloured by frame) and their
                        mean (thick), the estimate and its 95 % interval, one row per window: what the multi-baseline
                        sum does that one frame cannot
    slit_r####.png      one azimuth over the run: J_k(q, phi_0) stacked by frame, with the traces q_k = q_r - ln(1 - t_k
                        zeta) the window's estimates predict for every frame

Drawn from observation fields and the estimates. Truth (Position) only with truth=True, dashed and labelled
EVALUATION; the interval scale is always the window's frozen one (the figures refuse an unfrozen result).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import analysis, flow_depth, logpolar as lp, matcher as m3, m4_pipeline as mp, multibaseline as mb, pointcloud
from .flow_figures import REASON_COLORS, RINGS_DEG, _plt

Z95 = 1.959963984540054


class RefView:
    def __init__(self, out, window, r, truth=True):
        self.d = Path(out) / window
        self.window, self.r = window, int(r)
        fr = mp._verify(self.d)
        self.scale = float(fr["frozen_settings"]["interval_scale"])
        self.cfg = json.loads((self.d / "config_frozen.json").read_text())
        self.s = self.cfg["settings"]
        self.run = analysis.load_run(self.cfg["run"])
        self.es = {e["k"]: e for e in self.run["looks"]["ahead"]}
        self.z = dict(np.load(self.d / f"r{self.r:04d}.npz"))
        z = self.z
        self.ob = flow_depth.load_observation(self.run, self.es[self.r])
        self.m, self.n = self.ob.model, self.ob.model.n
        self.acc = z["reason"] == 0
        ce = np.cos(z["e"])
        half = Z95 * self.scale * z["sigma_zeta"]
        with np.errstate(divide="ignore", invalid="ignore"):
            self.range = np.where(self.acc, 1.0 / (z["zeta"] * ce), np.nan)
            near = 1.0 / ((z["zeta"] + half) * ce)
            far = np.where(z["zeta"] - half > 0, 1.0 / ((z["zeta"] - half) * ce), np.inf)
            self.rel_half = 0.5 * (far - near) / self.range
        self.usable = self.acc & np.isfinite(self.rel_half) & (self.rel_half <= flow_depth.USABLE_REL)
        self.truth = mp.truth(self.run, self.r, z, other=self.r - mp.M3_PAIR) if truth else None
        self.exposure = pointcloud.exposure_of(self.ob.rec.value)
        g = (self.cfg.get("computed_by") or {}).get("foveal-motion", {})
        self.by = str(g.get("commit", "?"))[:7] + ("" if g.get("clean", True) else " + local changes")

    def raster(self, v, fill=np.nan):
        R = np.full(self.n * self.n, fill, dtype=float)
        R[self.z["flat"]] = v
        return R.reshape(self.n, self.n)

    def label(self):
        ks = self.z["frames"]
        return f"{self.window} (frames {int(ks.min())}..{int(ks.max())}), frozen scale {self.scale:.2f}, estimates by {self.by}"


def _rings(ax, m, n):
    for e in RINGS_DEG:
        ph = np.linspace(-np.pi, np.pi, 361)
        rc, _ = m.pixel_of(lp.dir_of_e_phi(np.radians(min(e, 44.99)), ph))
        ax.plot(rc[:, 1], rc[:, 0], color="#00e5ff", lw=0.6, alpha=0.8)
        ax.text(rc[90, 1] + 2, rc[90, 0], f"{e}°", color="#00e5ff", fontsize=7)
    c = (n - 1) / 2
    ax.plot([c], [c], "+", color="#ff2d55", ms=10, mew=1.5)


def _m3_usable(v: RefView, m3_scale):
    """M3's frozen matcher on the pair (r - 5, r), the same anchors, M3's frozen scale (as m4_pipeline.evaluate)."""
    oa = flow_depth.load_observation(v.run, v.es[v.r - mp.M3_PAIR])
    B = float(np.linalg.norm(v.ob.c - oa.c))
    c3 = m3.match_pair(m3.luminance(oa.rgb), m3.luminance(v.ob.rgb), v.m, B)
    if not np.array_equal(c3["flat"], v.z["flat"]):
        raise SystemExit("[m4-figures] M3 and M4 anchors differ")
    acc3 = c3["reason"] == 0
    h3 = Z95 * m3_scale * c3["sigma_d"]
    d3 = np.where(acc3, c3["d"], np.nan)
    rec3 = flow_depth.reconstruct({"sample_b": v.z["sample"], "e_b": v.z["e"], "phi_b": v.z["phi"], "d": d3,
                                   "d_lo": d3 - h3, "d_hi": d3 + h3}, v.ob, B)
    return rec3["usable"] & acc3


def overview(views, out: Path, m3_scale):
    plt = _plt()
    from matplotlib.colors import ListedColormap
    v0 = views[0]
    use3 = _m3_usable(v0, m3_scale)
    nR = len(mb.REASONS)
    fig, ax = plt.subplots(len(views), 6, figsize=(33, 5.6 * len(views)), squeeze=False)
    for row, v in enumerate(views):
        z = v.z
        a = ax[row]
        if row == 0:
            a[0].imshow(pointcloud.srgb_u8(np.nan_to_num(v.ob.rgb), v.exposure))
            _rings(a[0], v.m, v.n)
            a[0].set_title(f"reference r = {v.r}, native raster, FOE +")
        else:
            cmp_ = np.where(v.usable & use3, 1, np.where(v.usable, 2, np.where(use3, 0, np.nan)))
            a[0].imshow(v0.raster(cmp_, fill=np.nan), cmap=ListedColormap(["#d62728", "#9e9e9e", "#2ca02c"]),
                        vmin=-0.5, vmax=2.5, interpolation="nearest")
            a[0].set_title(f"usable: green M4 {v.window} only {np.mean(v.usable & ~use3):.3f}, grey both "
                           f"{np.mean(v.usable & use3):.3f},\nred M3 pair ({v.r - 5}, {v.r}) only {np.mean(use3 & ~v.usable):.3f}",
                           fontsize=10)
        im = a[1].imshow(v.raster(z["reason"]), cmap=ListedColormap(REASON_COLORS[:nR]), vmin=-0.5, vmax=nR - 0.5,
                         interpolation="nearest")
        cb = fig.colorbar(im, ax=a[1], ticks=range(nR), fraction=0.046)
        cb.ax.set_yticklabels(mb.REASONS, fontsize=7)
        a[1].set_title(f"{v.window}: decision per anchor, accepted {v.acc.mean():.3f}")
        im = a[2].imshow(v.raster(np.where(v.usable, v.range, np.nan)), cmap="turbo", vmin=0, vmax=8)
        fig.colorbar(im, ax=a[2], fraction=0.046)
        a[2].set_title(f"range (m), usable (interval <= 10 %): {v.usable.mean():.3f} of anchors")
        im = a[3].imshow(v.raster(np.where(v.acc, v.rel_half, np.nan)), cmap="magma_r", vmin=0, vmax=0.2)
        fig.colorbar(im, ax=a[3], fraction=0.046)
        a[3].set_title("relative half-width of the 95 % interval, accepted")
        im = a[4].imshow(v.raster(np.where(v.acc, z["n_frames"], np.nan)), cmap="viridis", vmin=0, vmax=len(z["frames"]))
        fig.colorbar(im, ax=a[4], fraction=0.046)
        a[4].set_title("frames kept by frame consistency, accepted")
        if v.truth is not None:
            t = v.truth
            rel = np.abs(v.range - t["range"]) / t["range"]
            im = a[5].imshow(v.raster(np.where(v.usable & t["hit"], rel, np.nan)), cmap="magma_r", vmin=0, vmax=0.1)
            fig.colorbar(im, ax=a[5], fraction=0.046)
            sel = v.usable & t["hit"] & ~t["edge"]
            a[5].set_title(f"EVALUATION: relative range error, usable\ninterior median {np.nanmedian(rel[sel]):.4f}, "
                           f"p90 {np.nanquantile(rel[sel], 0.9):.4f}", fontsize=10)
        else:
            a[5].axis("off")
        for x in a:
            x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"M4 {Path(v0.cfg['run']).name}, reference r = {v0.r}; " + "; ".join(v.label() for v in views)
                 + "")
    fig.tight_layout()
    fig.savefig(out, dpi=70)
    plt.close(fig)


def cloud(views, out: Path):
    plt = _plt()
    fig, ax = plt.subplots(1, len(views), figsize=(9 * len(views), 9), squeeze=False)
    for a, v in zip(ax[0], views):
        u = v.usable
        P = lp.world_points(v.ob.c, v.ob.R, v.z["e"][u], v.z["phi"][u], v.range[u])
        rgb = pointcloud.srgb_u8(v.ob.rec.value[v.z["sample"][u]].astype(np.float32), v.exposure)
        a.scatter(P[:, 0], P[:, 1], c=rgb / 255, s=0.8)
        cs = np.array([flow_depth.load_observation(v.run, v.es[int(k)]).c for k in (v.z["frames"].min(), v.z["frames"].max())])
        a.plot(cs[:, 0], cs[:, 1], "r-", lw=2)
        a.plot([v.ob.c[0]], [v.ob.c[1]], "ro")
        a.set_aspect("equal")
        a.set_title(f"{v.window}: {int(u.sum())} usable points of r = {v.r}, top view (x, y); red: the window's eye path, "
                    "dot: r", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)


def _frame_curves(v: RefView, i, zetas):
    """NCC of r's patch at anchor i against each window frame along zeta (the slope zeta_q held at the fit, 0 when
    rejected), on the native rasters: the per-frame terms of the coarse cost, at full resolution."""
    s, z = v.s, v.z
    q, ph = z["q"][i], z["phi"][i]
    g = lp.native_px_per_q(v.m, np.array([q]), np.array([ph]))[0]
    gp = np.linalg.norm(v.m.pixel_of(lp.dir_of_e_phi(z["e"][i], ph + 1e-5))[0]
                        - v.m.pixel_of(lp.dir_of_e_phi(z["e"][i], ph - 1e-5))[0]) / 2e-5
    K, J = np.meshgrid(np.arange(-s["r_half_px"], s["r_half_px"] + 1) / g,
                       np.arange(-s["a_half_px"], s["a_half_px"] + 1) / gp, indexing="ij")
    u, w = K.reshape(-1), J.reshape(-1)
    zq = z["zeta_q"][i] if v.acc[i] else 0.0
    Ir = m3.sample(m3.luminance(v.ob.rgb), v.m, q + u, ph + w)
    zr = Ir - np.nanmean(Ir)
    C = np.full((len(z["frames"]), len(zetas)), np.nan)
    for j, (k, t) in enumerate(zip(z["frames"], z["ts"])):
        img = m3.luminance(flow_depth.load_observation(v.run, v.es[int(k)]).rgb)
        Z = zetas[:, None] + zq * u[None, :]
        with np.errstate(invalid="ignore"):
            dk = np.where(t * Z < 0.98, -np.log1p(-t * Z), np.nan)
        Ik = m3.sample(img, v.m, q + u[None, :] + dk, np.broadcast_to(ph + w, dk.shape))
        zk = Ik - np.nanmean(Ik, -1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            C[j] = np.sum(zk * zr, -1) / np.sqrt(np.sum(zk ** 2, -1) * np.sum(zr ** 2))
    return C


def _pick(views):
    """Anchors to show, the same pixels in every window: accepted (in the last window) interior anchors near 5 and
    20 deg, the median of their coarse NCC; and anchors the first window rejected as ambiguous and the last accepted."""
    a, b = views[0], views[-1]
    e = np.degrees(b.z["e"])
    edge = b.truth["edge"] if b.truth is not None else np.zeros_like(b.acc)
    picks = []
    for tgt in (5, 20):
        c = np.flatnonzero(a.acc & b.acc & ~edge & (np.abs(e - tgt) < 1.5))
        if c.size:
            picks.append((int(c[np.argsort(b.z["ncc_coarse"][c])[c.size // 2]]), "accepted in both"))
    amb = mb.REASONS.index("ambiguous")
    for tgt in (3, 15):
        c = np.flatnonzero((a.z["reason"] == amb) & b.acc & ~edge & (np.abs(e - tgt) < 2.5))
        if c.size:
            picks.append((int(c[np.argsort(b.z["ncc_coarse"][c])[c.size // 2]]), f"{a.window} ambiguous, {b.window} accepted"))
    return picks


def curves(views, out: Path):
    plt = _plt()
    picks = _pick(views)
    if not picks:
        return None
    fig, ax = plt.subplots(len(views), len(picks), figsize=(6.2 * len(picks), 4.6 * len(views)), squeeze=False)
    cmap = plt.get_cmap("coolwarm")
    for row, v in enumerate(views):
        z = v.z
        for col, (i, why) in enumerate(picks):
            a = ax[row, col]
            zt = v.truth["zeta"][i] if v.truth is not None else np.nan
            zmax = max(3 * (z["zeta"][i] if v.acc[i] else z["zeta_coarse"][i]), 3 * (zt if np.isfinite(zt) else 0), 0.6)
            zetas = np.linspace(0, zmax, 300)
            C = _frame_curves(v, i, zetas)
            for j, t in enumerate(z["ts"]):
                a.plot(zetas, C[j], color=cmap(0.5 + 0.5 * t / np.abs(z["ts"]).max()), lw=0.7, alpha=0.8)
            a.plot(zetas, np.nanmean(C, 0), color="k", lw=2.2, label="mean over frames")
            if v.acc[i]:
                hw = Z95 * v.scale * z["sigma_zeta"][i]
                a.axvspan(z["zeta"][i] - hw, z["zeta"][i] + hw, color="#2ca02c", alpha=0.35, label="95 % interval")
                a.axvline(z["zeta"][i], color="#2ca02c", lw=1.4, label=f"estimate {1 / z['zeta'][i]:.2f} m axial")
            if np.isfinite(zt):
                a.axvline(zt, color="#d62728", ls="--", lw=1, label="EVALUATION: truth")
            a.set_ylim(-1, 1.02)
            a.set_xlim(0, zmax)
            a.set_xlabel("zeta = 1 / axial depth of r (1/m)")
            a.set_ylabel("NCC")
            a.set_title(f"{v.window}: anchor {np.degrees(z['e'][i]):.1f}°, {np.degrees(z['phi'][i]):.0f}° az: "
                        f"{mb.REASONS[z['reason'][i]]}\n({why}; thin: frames, blue behind r, red ahead)", fontsize=9)
            a.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"M4 matching along zeta, reference r = {views[0].r}: each frame's NCC and the mean the estimator "
                 f"maximises; " + "; ".join(v.label() for v in views))
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)
    return picks


def slit_scan(v: RefView, out: Path, n_tracks: int = 10, span=10):
    plt = _plt()
    z = v.z
    u = v.usable
    bins = np.linspace(-np.pi, np.pi, 73)
    hist, _ = np.histogram(z["phi"][u], bins)
    b = int(np.argmax(hist))
    phi0 = 0.5 * (bins[b] + bins[b + 1])
    q = np.linspace(lp.q_of_e(np.radians(1.0)), lp.q_of_e(np.radians(45.0)), 600)
    ks = [k for k in range(v.r - span, v.r + span + 1) if k in v.es]
    obs = {k: flow_depth.load_observation(v.run, v.es[k]) for k in ks}
    h = v.ob.R @ [0, 0, -1.0]
    s_k = {k: float((obs[k].c - v.ob.c) @ h) for k in ks}
    band = np.radians(np.linspace(-0.5, 0.5, 5))
    S = np.stack([np.nanmean(m3.sample(m3.luminance(obs[k].rgb), v.m, q[None, :], phi0 + band[:, None]), 0) for k in ks])
    fig, a = plt.subplots(1, 1, figsize=(16, 9))
    a.imshow(S, aspect="auto", cmap="gray", extent=[q[0], q[-1], ks[-1] + 0.5, ks[0] - 0.5], vmin=0,
             vmax=np.nanpercentile(S, 99))
    yt = [1, 2, 5, 10, 20, 30, 45]
    a.set_xticks([lp.q_of_e(np.radians(t)) for t in yt])
    a.set_xticklabels([f"{t}°" for t in yt])
    a.set_xlabel("eccentricity (log-tangent axis q)")
    a.set_ylabel("frame k (the head moves forward downward)")
    w0, w1 = int(z["frames"].min()), int(z["frames"].max())
    a.axhspan(w0 - 0.5, w1 + 0.5, color="#ffd60a", alpha=0.08)
    a.axhline(v.r, color="w", lw=0.6, ls=":")
    near = np.flatnonzero(u & (np.abs(np.angle(np.exp(1j * (z["phi"] - phi0)))) < np.radians(0.6)))
    if near.size:
        near = near[np.argsort(z["e"][near])]
        pick = near[np.linspace(0, near.size - 1, min(n_tracks, near.size)).astype(int)]
        cm = plt.get_cmap("tab10")
        kk = np.array(ks)
        tt = np.array([s_k[k] for k in ks])
        for c_, i in enumerate(pick):
            for zeta, ls, lw in ((z["zeta"][i], "-", 1.2),) + (((v.truth["zeta"][i], "--", 0.8),)
                                                                 if v.truth is not None and v.truth["hit"][i] else ()):
                ok = tt * zeta < 0.95
                qk = z["q"][i] - np.log1p(-tt[ok] * zeta)
                m = (qk >= q[0]) & (qk <= q[-1])
                a.plot(qk[m], kk[ok][m], color=cm(c_ % 10), lw=lw, ls=ls)
            a.plot([z["q"][i]], [v.r], "o", color=cm(c_ % 10), ms=4)
    a.set_title(f"slit scan at azimuth {np.degrees(phi0):.0f}°, J_k(q, phi_0) stacked by frame; yellow band: the "
                f"{v.window} window of r = {v.r}. Solid: q_k = q_r - ln(1 - t_k zeta), M4's estimates"
                + ("; dashed: EVALUATION truth" if v.truth is not None else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def figures(out, windows, r, dest, m3_scale, truth=True) -> list:
    views = [RefView(out, w, r, truth) for w in windows]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    p = dest / f"overview_r{r:04d}.png"; overview(views, p, m3_scale); paths.append(p)
    p = dest / f"cloud_r{r:04d}.png"; cloud(views, p); paths.append(p)
    p = dest / f"curves_r{r:04d}.png"
    if curves(views, p) is not None:
        paths.append(p)
    for v in views:
        p = dest / f"slit_{v.window}_r{r:04d}.png"; slit_scan(v, p); paths.append(p)
    return paths
