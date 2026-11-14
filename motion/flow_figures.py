"""M3 process figures (M3a #6, Stage 6): what the estimator saw, decided and reconstructed, for one pair.

    overview.png    native looks a and b with the focus of expansion and eccentricity rings; the decision
                    per anchor; the displacement; range and its uncertainty on b's raster; the cloud
    matching.png    the unwrapped (q, phi) images, J_b - J_a before and after moving J_a by the estimated d;
                    the matching curves of a few anchors with the estimate and its interval
    slit_scan.png   one azimuth over the whole sequence: J_k(q, phi_0) stacked by frame; the outward streaks
                    are points moving along q; the traces the two-frame depths predict for every frame

Everything is drawn from observation fields and the estimate. Truth overlays (Position) are drawn only
with truth=True, dashed and labelled "evaluation". The interval scale is the frozen one when the result
is frozen, otherwise the one passed in and labelled provisional.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import analysis, flow_depth, logpolar as lp, matcher, pointcloud

RINGS_DEG = (1, 2, 5, 10, 20, 30, 45)
REASON_COLORS = ["#2ca02c", "#7f7f7f", "#bcbd22", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                 "#17becf", "#1f77b4", "#000000", "#f7b6d2"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class PairView:
    def __init__(self, result, pair, scale=None, truth=True):
        self.result = Path(result)
        self.cfg = json.loads((self.result / "config.json").read_text())
        fr = self.result / "freeze.json"
        if fr.is_file():
            self.scale = json.loads(fr.read_text())["frozen_settings"]["interval_scale"]
            self.scale_label = f"frozen scale {self.scale:.2f}"
        elif scale is not None:
            self.scale = float(scale)
            self.scale_label = f"PROVISIONAL scale {self.scale:.2f} (not frozen)"
        else:
            raise SystemExit("[figures] the result is not frozen: pass --scale (it will be labelled provisional)")
        self.run = analysis.load_run(self.cfg["run"])
        self.es = {e["k"]: e for e in self.run["looks"][self.cfg["mode"]]}
        self.ka, self.kb = pair
        self.z = dict(np.load(self.result / "pairs" / f"s{self.kb - self.ka}" / f"a{self.ka:04d}-b{self.kb:04d}.npz"))
        self.oa = flow_depth.load_observation(self.run, self.es[self.ka])
        self.ob = flow_depth.load_observation(self.run, self.es[self.kb])
        self.m = self.ob.model
        self.n = self.m.n
        z = self.z
        self.acc = z["reason"] == 0
        d = np.where(self.acc, z["d"], np.nan)
        h = 1.959963984540054 * self.scale * z["sigma_d"]
        self.rec = flow_depth.reconstruct({"sample_b": z["sample_b"], "e_b": z["e_b"], "phi_b": z["phi_b"], "d": d,
                                           "d_lo": d - h, "d_hi": d + h}, self.ob, float(z["B"]))
        self.exposure = pointcloud.exposure_of(self.oa.rec.value)
        self.truth = None
        if truth:
            from . import flow_depth_eval as fe
            self.la, self.lb = analysis.load_look(self.run, self.es[self.ka]), analysis.load_look(self.run, self.es[self.kb])
            self.truth = fe.pair_truth(self.la, self.lb, z)

    def raster(self, v, fill=np.nan):
        R = np.full(self.n * self.n, fill, dtype=float)
        R[self.z["flat"]] = v
        return R.reshape(self.n, self.n)

    def rgb_img(self, o):
        return pointcloud.srgb_u8(np.nan_to_num(o.rgb), self.exposure)

    def rings(self, ax):
        for e in RINGS_DEG:
            ph = np.linspace(-np.pi, np.pi, 361)
            rc, inside = self.m.pixel_of(lp.dir_of_e_phi(np.radians(min(e, 44.99)), ph))
            ax.plot(rc[:, 1], rc[:, 0], color="#00e5ff", lw=0.6, alpha=0.8)
            ax.text(rc[90, 1] + 2, rc[90, 0], f"{e}°", color="#00e5ff", fontsize=7)
        c = (self.n - 1) / 2
        ax.plot([c], [c], "+", color="#ff2d55", ms=10, mew=1.5)

    def title(self):
        g = (self.cfg.get("computed_by") or {}).get("foveal-motion", {})
        by = str(g.get("commit", "?"))[:7] + ("" if g.get("clean") else " + local changes")
        return (f"M3 {Path(self.cfg['run']).name}, pair ({self.ka}, {self.kb}), B = {float(self.z['B']):.3f} m, "
                f"{self.scale_label}; estimate computed by {by}")


def overview(v: PairView, out: Path, reference=None):
    plt = _plt()
    from matplotlib.colors import ListedColormap
    z, rec = v.z, v.rec
    usable = rec["usable"] & v.acc
    fig, ax = plt.subplots(2, 4, figsize=(22, 11))
    ax[0, 0].imshow(v.rgb_img(v.oa)); ax[0, 0].set_title(f"look a = {v.ka} (native foveated raster), FOE +, rings")
    v.rings(ax[0, 0])
    ax[0, 1].imshow(v.rgb_img(v.ob)); ax[0, 1].set_title(f"look b = {v.kb}")
    v.rings(ax[0, 1])
    nR = len(matcher.REASONS)
    im = ax[0, 2].imshow(v.raster(z["reason"]), cmap=ListedColormap(REASON_COLORS[:nR]), vmin=-0.5, vmax=nR - 0.5,
                         interpolation="nearest")
    cb = fig.colorbar(im, ax=ax[0, 2], ticks=range(nR), fraction=0.046)
    cb.ax.set_yticklabels(matcher.REASONS, fontsize=7)
    ax[0, 2].set_title(f"decision per anchor: accepted {v.acc.mean():.2f}")
    im = ax[0, 3].imshow(v.raster(np.where(v.acc, z["d"] * z["px_per_q_a"], np.nan)), cmap="viridis")
    fig.colorbar(im, ax=ax[0, 3], fraction=0.046)
    ax[0, 3].set_title("displacement d, accepted (native px of a)")
    im = ax[1, 0].imshow(v.raster(np.where(usable, rec["range"], np.nan)), cmap="turbo")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)
    ax[1, 0].set_title(f"range (m), usable (interval <= 10 %): {usable.mean():.2f} of anchors")
    im = ax[1, 1].imshow(v.raster(np.where(v.acc, rec["rel_half"], np.nan)), cmap="magma_r", vmin=0, vmax=0.3)
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    ax[1, 1].set_title("relative half-width of the 95 % interval, accepted")
    if v.truth is not None:
        t = v.truth
        rel = np.abs(rec["range"] - t["range"]) / t["range"]
        im = ax[1, 2].imshow(v.raster(np.where(usable & t["hit"], rel, np.nan)), cmap="magma_r", vmin=0, vmax=0.2)
        fig.colorbar(im, ax=ax[1, 2], fraction=0.046)
        sel = usable & t["hit"] & ~t["edge"]
        ax[1, 2].set_title(f"EVALUATION: relative range error, usable\ninterior median {np.nanmedian(rel[sel]):.3f}, "
                           f"p90 {np.nanquantile(rel[sel], 0.9):.3f} (edges within 2 px apart)")
    else:
        ax[1, 2].axis("off")
    P = rec["P"][usable]
    rgb = pointcloud.srgb_u8(rec["rgb_linear"][usable], v.exposure)
    if reference is not None:
        O, orgb, _ = pointcloud.read_ply(Path(reference) / "reconstruction.ply")
        i = np.random.default_rng(0).choice(len(O), min(len(O), 60000), replace=False)
        ax[1, 3].scatter(O[i, 0], O[i, 1], c="#cccccc", s=0.15)
    ax[1, 3].scatter(P[:, 0], P[:, 1], c=rgb / 255, s=0.6)
    cs = [v.oa.c, v.ob.c]
    ax[1, 3].plot([c[0] for c in cs], [c[1] for c in cs], "r-", lw=2)
    ax[1, 3].set_aspect("equal")
    ax[1, 3].set_title("usable points, top view (x, y)\n" + ("grey: oracle reference; " if reference else "")
                       + "red: eye path a -> b", fontsize=10)
    for a in ax.flat[:7]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(v.title())
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def _pick(v: PairView):
    """A few anchors to show: accepted interior ones near 5, 15, 30 deg, and one ambiguous near 15 deg."""
    z = v.z
    e = np.degrees(z["e_b"])
    edge = v.truth["edge"] if v.truth is not None else np.zeros_like(v.acc)
    picks = []
    for tgt, want in ((5, 0), (15, 0), (30, 0), (15, matcher.REASONS.index("ambiguous"))):
        cand = np.flatnonzero((z["reason"] == want) & ~edge & (np.abs(e - tgt) < 1.5) & (z["texture"] > 0.05))
        if cand.size:
            picks.append(int(cand[np.argsort(z["ncc_coarse"][cand])[cand.size // 2]]))
    return picks


def _curve(v: PairView, i, n_d=400):
    """NCC of b's patch against a's along the search line, kappa fixed at the anchor's fit (0 when rejected)."""
    s = v.cfg["settings"]
    z = v.z
    B = float(z["B"])
    d_max = np.log1p(B / s["z_envelope_m"][0])
    ds = np.linspace(0, d_max, n_d)
    qb, pb = z["q_b"][i], z["phi_b"][i]
    g = lp.native_px_per_q(v.m, np.array([qb]), np.array([pb]))[0]
    gp = np.linalg.norm(v.m.pixel_of(lp.dir_of_e_phi(z["e_b"][i], pb + 1e-5))[0]
                        - v.m.pixel_of(lp.dir_of_e_phi(z["e_b"][i], pb - 1e-5))[0]) / 2e-5
    k = np.arange(-s["r_half_px"], s["r_half_px"] + 1) / g
    j = np.arange(-s["a_half_px"], s["a_half_px"] + 1) / gp
    K, J = np.meshgrid(k, j, indexing="ij")
    u, w = K.reshape(-1), J.reshape(-1)
    kap = z["kappa"][i] if z["reason"][i] == 0 else 0.0
    Ia_img, Ib_img = matcher.luminance(v.oa.rgb), matcher.luminance(v.ob.rgb)
    Ib = matcher.sample(Ib_img, v.m, qb + u, pb + w)
    Ia = matcher.sample(Ia_img, v.m, qb + u[None, :] - ds[:, None] - kap * u[None, :], pb + w[None, :])
    zb = Ib - np.nanmean(Ib)
    za = Ia - np.nanmean(Ia, -1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nansum(za * zb, -1) / np.sqrt(np.nansum(za ** 2, -1) * np.nansum(zb ** 2))
    return ds * g, c, g


def matching(v: PairView, out: Path):
    plt = _plt()
    z = v.z
    grid = lp.make_grid(v.m, n_q=240, n_phi=720)
    Jb = lp.unwrap(matcher.luminance(v.ob.rgb), grid)
    Ja = lp.unwrap(matcher.luminance(v.oa.rgb), grid)
    # d at each node from the accepted anchor at the node's native pixel
    D = v.raster(np.where(v.acc, z["d"], np.nan)).reshape(-1)
    ij = np.clip(np.rint(grid.rc).astype(int), 0, v.n - 1)
    dn = D[ij[..., 0] * v.n + ij[..., 1]].reshape(grid.shape)
    Q = np.broadcast_to(grid.q[:, None], grid.shape)
    PH = np.broadcast_to(grid.phi[None, :], grid.shape)
    Ja_moved = matcher.sample(matcher.luminance(v.oa.rgb), v.m, Q - np.nan_to_num(dn), PH)
    Ja_moved = np.where(np.isfinite(dn), Ja_moved, np.nan)
    ext = [-180, 180, np.degrees(grid.e[0]), np.degrees(grid.e[-1])]
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9])
    yt = [1, 2, 5, 10, 20, 30, 45]
    def show(axx, img, title, **kw):
        axx.imshow(img, origin="lower", aspect="auto", extent=[-180, 180, 0, len(grid.q)], **kw)
        axx.set_yticks([np.interp(lp.q_of_e(np.radians(t)), grid.q, np.arange(len(grid.q))) for t in yt])
        axx.set_yticklabels([f"{t}°" for t in yt])
        axx.set_xlabel("azimuth phi (deg)")
        axx.set_ylabel("eccentricity (log-tangent axis q)")
        axx.set_title(title)
    vmax = np.nanpercentile(Jb, 99)
    show(fig.add_subplot(gs[0, 0:2]), Jb, f"J_b(q, phi): look {v.kb} unwrapped", cmap="gray", vmin=0, vmax=vmax)
    show(fig.add_subplot(gs[0, 2:4]), Ja, f"J_a(q, phi): look {v.ka}; a point keeps phi and moves up by d", cmap="gray", vmin=0, vmax=vmax)
    r0 = np.abs(Jb - Ja)
    show(fig.add_subplot(gs[1, 0:2]), r0, "|J_b - J_a| before: everything moved outward", cmap="inferno", vmin=0, vmax=0.25 * vmax)
    r1 = np.abs(Jb - Ja_moved)
    show(fig.add_subplot(gs[1, 2:4]), r1, "|J_b - J_a moved by the estimated d| (accepted only): aligned", cmap="inferno",
         vmin=0, vmax=0.25 * vmax)
    picks = _pick(v)
    for c_, i in enumerate(picks[:4]):
        axx = fig.add_subplot(gs[2, c_])
        x, c, g = _curve(v, i)
        axx.plot(x, c, color="k", lw=1)
        d_px = z["d"][i] * g
        lab = matcher.REASONS[z["reason"][i]]
        if z["reason"][i] == 0:
            hw = 1.959963984540054 * v.scale * z["sigma_d"][i] * g
            axx.axvspan(d_px - hw, d_px + hw, color="#2ca02c", alpha=0.3, label="95 % interval")
            axx.axvline(d_px, color="#2ca02c", lw=1.5, label=f"estimate {d_px:.2f} px")
        if v.truth is not None and np.isfinite(v.truth["d"][i]):
            axx.axvline(v.truth["d"][i] * g, color="#d62728", ls="--", lw=1, label="EVALUATION: truth")
        axx.set_xlim(0, min(x[-1], max(4 * max(d_px, 1), 15)))
        axx.set_ylim(-1, 1.02)
        axx.set_xlabel("d (native px of b at the anchor)")
        axx.set_ylabel("NCC")
        axx.set_title(f"anchor at {np.degrees(z['e_b'][i]):.1f}°, {np.degrees(z['phi_b'][i]):.0f}° az: {lab}", fontsize=10)
        axx.legend(fontsize=7, loc="lower right")
    fig.suptitle(v.title())
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)


def slit_scan(v: PairView, out: Path, n_tracks: int = 10):
    plt = _plt()
    z = v.z
    acc = v.acc & v.rec["usable"]
    bins = np.linspace(-np.pi, np.pi, 73)
    hist, _ = np.histogram(z["phi_b"][acc], bins)
    b = int(np.argmax(hist))
    phi0 = 0.5 * (bins[b] + bins[b + 1])
    q = np.linspace(lp.q_of_e(np.radians(1.0)), lp.q_of_e(np.radians(45.0)), 600)
    ks = sorted(v.es)
    obs = {k: flow_depth.load_observation(v.run, v.es[k]) for k in ks}
    h = (v.ob.R @ [0, 0, -1.0])
    s_k = {k: float((obs[k].c - v.ob.c) @ h) for k in ks}           # camera position along the heading, from b
    band = np.radians(np.linspace(-0.5, 0.5, 5))
    S = np.stack([np.nanmean(matcher.sample(matcher.luminance(obs[k].rgb), v.m, q[None, :], phi0 + band[:, None]), 0)
                  for k in ks])
    fig, ax = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [2.2, 1]})
    e_ax = np.degrees(lp.e_of_q(q))
    ax[0].imshow(S, aspect="auto", cmap="gray", extent=[q[0], q[-1], ks[-1] + 0.5, ks[0] - 0.5],
                 vmin=0, vmax=np.nanpercentile(S, 99))
    yt = [1, 2, 5, 10, 20, 30, 45]
    ax[0].set_xticks([lp.q_of_e(np.radians(t)) for t in yt])
    ax[0].set_xticklabels([f"{t}°" for t in yt])
    ax[0].set_xlabel("eccentricity (log-tangent axis q)")
    ax[0].set_ylabel("frame k (the head moves forward downward)")
    near = np.flatnonzero(acc & (np.abs(np.angle(np.exp(1j * (z["phi_b"] - phi0)))) < np.radians(0.6)))
    if near.size:
        near = near[np.argsort(z["e_b"][near])]
        pick = near[np.linspace(0, near.size - 1, min(n_tracks, near.size)).astype(int)]
        cmap = plt.get_cmap("tab10")
        for c_, i in enumerate(pick):
            zb = v.rec["range"][i] * np.cos(z["e_b"][i])
            L = zb * np.tan(z["e_b"][i])
            kk = np.array([k for k in ks if zb - s_k[k] > 0.05])
            qk = np.log(L / (zb - np.array([s_k[k] for k in kk])))
            m = (qk >= q[0]) & (qk <= q[-1])
            ax[0].plot(qk[m], kk[m], color=cmap(c_ % 10), lw=1.2)
            ax[1].plot([s_k[k] for k in kk], (zb - np.array([s_k[k] for k in kk])) / L, color=cmap(c_ % 10), lw=1.2,
                       label=f"{np.degrees(z['e_b'][i]):.1f}°, z_b {zb:.2f} m")
            if v.truth is not None and v.truth["hit"][i]:
                zt = v.truth["range"][i] * np.cos(z["e_b"][i])
                Lt = zt * np.tan(z["e_b"][i])
                kt = np.array([k for k in ks if zt - s_k[k] > 0.05])
                qt = np.log(Lt / (zt - np.array([s_k[k] for k in kt])))
                mt = (qt >= q[0]) & (qt <= q[-1])
                ax[0].plot(qt[mt], kt[mt], color=cmap(c_ % 10), lw=0.8, ls="--")
            ax[0].plot([z["q_b"][i]], [v.kb], "o", color=cmap(c_ % 10), ms=4)
        ax[0].axhline(v.ka, color="w", lw=0.6, ls=":")
        ax[0].axhline(v.kb, color="w", lw=0.6, ls=":")
    ax[0].set_title(f"slit scan at azimuth {np.degrees(phi0):.0f}°: J_k(q, phi_0) stacked by frame. Solid: the traces the "
                    f"two-frame depths of pair ({v.ka}, {v.kb}) predict for every frame"
                    + ("; dashed: EVALUATION truth" if v.truth is not None else ""), fontsize=10)
    ax[1].set_xlabel("camera position s along the heading, from b (m)")
    ax[1].set_ylabel("cot e = (z_b - s) / L")
    ax[1].set_title("the multi-frame relation: cot e is linear in s, one line per point")
    ax[1].legend(fontsize=7)
    ax[1].axvline(0, color="k", lw=0.5)
    ax[1].axvline(s_k[v.ka], color="k", lw=0.5, ls=":")
    fig.suptitle(v.title())
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def figures(result, pair, out, scale=None, reference=None, truth=True) -> list:
    v = PairView(result, pair, scale, truth)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in (("overview", lambda p: overview(v, p, reference)), ("matching", lambda p: matching(v, p)),
                     ("slit_scan", lambda p: slit_scan(v, p))):
        p = out / f"{name}_a{v.ka:04d}-b{v.kb:04d}.png"
        fn(p)
        paths.append(p)
    return paths
