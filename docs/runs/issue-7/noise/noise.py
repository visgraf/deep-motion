"""Noise of the Classroom observations: per-pixel sigma against local contrast, and what the matcher does
with clean looks (issue #7, Code; Chat's "The next question: is it the noise?", Luiz's OK).

Inputs (observation fields only: luminance and pose, no oracle field, no truth):
    runs/noise/k20-spp64-seed0   look 20, 64 spp, seed 0 (the seed #1's dolly-1 used for every look)
    runs/noise/k20-spp64-seed1   look 20, 64 spp, seed 1
    runs/noise/spp1024-seed2     looks 20 and 25, 1024 spp, seed 2 (shares no seed with the 64-spp looks)
    runs/dolly-1                 looks 20 and 25 as the quick look matched them (64 spp, seed 0)
    runs/m3-classroom-look       the frozen matcher on dolly-1, pair (20, 25)
    runs/noise/m3-look-1024      the frozen matcher on the 1024-spp looks, pair (20, 25)

    PYTHONPATH=. .venv/bin/python docs/runs/issue-7/noise/noise.py --out docs/runs/issue-7/noise

Definitions (luminance = matcher.luminance, linear, on the native 253 x 253 foveated raster):
  sigma_64     per pixel from the two 64-spp seeds: (L_s0 - L_s1)^2 / 2 estimates sigma_64^2. Binned values
               are RMS over the bin; the local value is the mean over the WIN x WIN window.
  sigma_1024   per bin: sqrt(mean((L_s0 - L_1024)^2) - sigma_64^2), i.e. what the 1024-spp reference
               adds on top of the 64-spp noise.
  contrast c   the standard deviation of L_1024 in the WIN x WIN window (the matcher's patch is 7 x 3
               native px). It includes the reference's own noise, so c has a floor near sigma_1024.
  SNR          c / sigma_64 (local). For two independent noisy copies of a patch with signal std c and
               noise std sigma, the expected NCC is c^2 / (c^2 + sigma^2): ncc_min 0.8 needs c / sigma >= 2.
               That is a model (estimated), not a measurement; the anchor table tests it.

Checks, each with a control that must fail:
  pose      every new look's eye pose equals dolly-1's at the same k, exactly; control: look 20 against
            dolly-1's look 25.
  same      the seed-0 render is dolly-1's noise realisation: corr(L_s0 - L_1024, L_dolly1 - L_1024) >=
            0.99 (Code's threshold, proposed); control: the seed-1 render must fall below it.
  anchors   the two matcher results hold the same anchors (sample_b equal); control: pair (20, 25)
            against pair (0, 5) of the quick look must differ.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from motion import analysis, flow_depth, matcher  # noqa: E402
from motion.dolly import FOV_RENDER, _git  # noqa: E402

WIN = 5
C_EDGES = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, np.inf]
SNR_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]
SAME_MIN = 0.99
# reference palette, categorical slots 1-3 in order; secondary text ink for reference lines
BLUE, ORANGE, AQUA, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def obs(run, k):
    r = analysis.load_run(REPO / run)
    e = {x["k"]: x for x in r["looks"]["ahead"]}[k]
    return flow_depth.load_observation(r, e)


def window(a, fn, min_valid):
    p = WIN // 2
    v = np.lib.stride_tricks.sliding_window_view(np.pad(a, p, constant_values=np.nan), (WIN, WIN))
    v = v.reshape(*a.shape, WIN * WIN)
    n = np.isfinite(v).sum(-1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = fn(v, axis=-1)
    return np.where(n >= min_valid, out, np.nan)


def fmt(x, nd=4):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    o = {"s0": obs("runs/noise/k20-spp64-seed0", 20), "s1": obs("runs/noise/k20-spp64-seed1", 20),
         "r20": obs("runs/noise/spp1024-seed2", 20), "r25": obs("runs/noise/spp1024-seed2", 25),
         "d20": obs("runs/dolly-1", 20), "d25": obs("runs/dolly-1", 25)}
    L = {k: matcher.luminance(v.rgb) for k, v in o.items()}
    res = {"schema": "foveal-motion-issue7-noise-v1", "window_px": WIN,
           "computed_by": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}, "checks": {}}
    lines = []
    P = lines.append
    cb = res["computed_by"]["foveal-motion"]
    P(f"# PYTHONPATH=. .venv/bin/python docs/runs/issue-7/noise/noise.py --out {a.out}  "
      f"(at {cb['commit'][:7]}, clean {cb['clean']})")
    P("")

    # ---------------------------------------------------------------- checks
    pose = {k: bool(np.array_equal(o[k].R, o[d].R) and np.array_equal(o[k].c, o[d].c))
            for k, d in (("s0", "d20"), ("s1", "d20"), ("r20", "d20"), ("r25", "d25"))}
    pose_ctrl = bool(np.array_equal(o["r20"].c, o["d25"].c))
    res["checks"]["pose"] = {"ok": all(pose.values()), "control_fails": not pose_ctrl, "detail": pose}
    valid = np.isfinite(L["s0"]) & np.isfinite(L["s1"]) & np.isfinite(L["r20"]) & np.isfinite(L["d20"])
    rd, r0, r1 = (L["d20"] - L["r20"])[valid], (L["s0"] - L["r20"])[valid], (L["s1"] - L["r20"])[valid]
    corr0, corr1 = float(np.corrcoef(r0, rd)[0, 1]), float(np.corrcoef(r1, rd)[0, 1])
    rgb_d = o["s0"].rgb - o["d20"].rgb
    res["checks"]["same"] = {"ok": corr0 >= SAME_MIN, "control_fails": corr1 < SAME_MIN,
                             "detail": {"corr_seed0": corr0, "corr_seed1_control": corr1,
                                        "bitwise_equal": bool(np.array_equal(o["s0"].rgb, o["d20"].rgb, equal_nan=True)),
                                        "max_abs_rgb_diff": float(np.nanmax(np.abs(rgb_d))),
                                        "pixels_differing": int(np.sum(np.any(rgb_d != 0, -1) & valid))}}
    za = dict(np.load(REPO / "runs/m3-classroom-look/pairs/s5/a0020-b0025.npz"))
    zb = dict(np.load(REPO / "runs/noise/m3-look-1024/pairs/s5/a0020-b0025.npz"))
    zc = dict(np.load(REPO / "runs/m3-classroom-look/pairs/s5/a0000-b0005.npz"))
    same_anchor = bool(np.array_equal(za["sample_b"], zb["sample_b"]) and np.array_equal(za["flat"], zb["flat"]))
    ctrl_anchor = bool(np.array_equal(za["e_b"], zc["e_b"]) and np.array_equal(za["d"], zc["d"], equal_nan=True))
    res["checks"]["anchors"] = {"ok": same_anchor, "control_fails": not ctrl_anchor,
                                "detail": {"n_anchors": int(za["flat"].size)}}
    P("# checks (each with a control that must fail)")
    for k, c in res["checks"].items():
        P(f"{k:8s} {'PASS' if c['ok'] else 'FAIL'}  control {'fails' if c['control_fails'] else 'DOES NOT FAIL'}  "
          f"{json.dumps(c['detail'])}")

    # ---------------------------------------------------------------- sigma against contrast, look 20
    d01 = L["s0"] - L["s1"]
    v64 = d01 ** 2 / 2                                   # per pixel, estimates sigma_64^2
    e0r = (L["s0"] - L["r20"]) ** 2                      # sigma_64^2 + sigma_1024^2
    c = window(np.where(np.isfinite(L["r20"]), L["r20"], np.nan), np.nanstd, int(0.8 * WIN * WIN))
    mu = window(L["r20"], np.nanmean, int(0.8 * WIN * WIN))
    s64_loc = np.sqrt(window(np.where(valid, v64, np.nan), np.nanmean, int(0.8 * WIN * WIN)))
    m = valid & np.isfinite(c) & np.isfinite(s64_loc)
    zero = m & (s64_loc == 0)                            # the two seeds agree exactly over the window
    with np.errstate(all="ignore"):
        snr = np.where(s64_loc > 0, c / s64_loc, np.nan)
    ms = m & ~zero
    g64 = float(np.sqrt(v64[m].mean()))
    g1024 = float(np.sqrt(max(e0r[m].mean() - v64[m].mean(), 0.0)))
    res["global_k20"] = {"pixels": int(m.sum()), "sigma_64_rms": g64, "sigma_1024_rms": g1024,
                         "ratio_64_over_1024": g64 / g1024 if g1024 > 0 else None, "expected_ratio_sqrt16": 4.0,
                         "sigma_64_robust": float(1.4826 * np.median(np.abs(d01[m])) / np.sqrt(2)),
                         "luminance_median": float(np.median(L["r20"][m])),
                         "contrast_median": float(np.median(c[m])),
                         "zero_noise_pixels": int(zero.sum()),
                         "zero_noise_luminance": sorted({round(float(x), 6) for x in L["r20"][zero]})[:10],
                         "zero_noise_contrast_max": float(c[zero].max()) if zero.any() else None,
                         "snr_pixels": int(ms.sum()), "snr64_median": float(np.median(snr[ms])),
                         "frac_snr64_ge2": float(np.mean(snr[ms] >= 2)), "frac_snr1024_ge2": float(np.mean(4 * snr[ms] >= 2))}
    G = res["global_k20"]
    P("")
    P(f"# look 20, {G['pixels']} pixels with a full window ({WIN} x {WIN}); luminance linear (matcher.luminance)")
    P(f"sigma_64 RMS {G['sigma_64_rms']:.4f} (robust {G['sigma_64_robust']:.4f}); sigma_1024 RMS {G['sigma_1024_rms']:.4f}; "
      f"ratio {G['ratio_64_over_1024']:.2f} (1/sqrt(spp) predicts 4)")
    P(f"median luminance {G['luminance_median']:.4f}, median local contrast {G['contrast_median']:.4f}, "
      f"median c/sigma_64 {G['snr64_median']:.2f} (over {G['snr_pixels']} pixels)")
    P(f"zero-noise pixels (the seeds agree exactly over the window): {G['zero_noise_pixels']}, luminance "
      f"{G['zero_noise_luminance']}, contrast at most {G['zero_noise_contrast_max']}; left out of c/sigma")
    P(f"pixels with c/sigma >= 2 (NCC 0.8 by the model): {G['frac_snr64_ge2']:.3f} at 64 spp, "
      f"{G['frac_snr1024_ge2']:.3f} at 1024 spp (sigma_1024 = sigma_64 / 4)")
    P("")
    P("sigma_1024 per bin is the difference of two nearly equal heavy-tailed means: noisy; the overall ratio above is the stable one")
    P("contrast c bin  | pixels | median L | sigma_64 RMS | sigma_1024 RMS | sigma_64 / median L | zero-noise px | median c/sigma_64 | c/sigma_64 >= 2 | c/sigma_1024 >= 2")
    rows = []
    for lo, hi in zip(C_EDGES[:-1], C_EDGES[1:]):
        b = m & (c >= lo) & (c < hi)
        if not b.any():
            continue
        s64 = float(np.sqrt(v64[b].mean()))
        s1024 = float(np.sqrt(max(e0r[b].mean() - v64[b].mean(), 0.0)))
        row = {"c_lo": lo, "c_hi": None if not np.isfinite(hi) else hi, "pixels": int(b.sum()),
               "median_L": float(np.median(L["r20"][b])), "sigma_64": s64, "sigma_1024": s1024,
               "rel_sigma_64": s64 / float(np.median(L["r20"][b])), "zero_noise": int((b & zero).sum()),
               "median_snr64": float(np.median(snr[b & ms])) if (b & ms).any() else None,
               "frac_snr64_ge2": float(np.mean(snr[b & ms] >= 2)), "frac_snr1024_ge2": float(np.mean(4 * snr[b & ms] >= 2))}
        rows.append(row)
        P(f"{lo:.4f}-{'inf' if not np.isfinite(hi) else f'{hi:.4f}'} | {row['pixels']:6d} | {row['median_L']:.4f} | "
          f"{s64:.4f} | {s1024:.4f} | {row['rel_sigma_64']:.2f} | {row['zero_noise']} | {fmt(row['median_snr64'], 2)} | "
          f"{row['frac_snr64_ge2']:.3f} | {row['frac_snr1024_ge2']:.3f}")
    res["by_contrast_k20"] = rows

    # ---------------------------------------------------------------- the same seed in every dolly-1 look
    v2 = valid & np.isfinite(L["d25"]) & np.isfinite(L["r25"])
    q25 = (L["d25"] - L["r25"])[v2]
    cc = float(np.corrcoef((L["d20"] - L["r20"])[v2], q25)[0, 1])
    cc_ctrl = float(np.corrcoef((L["s1"] - L["r20"])[v2], q25)[0, 1])
    res["cross_frame_same_pixel"] = {"corr_dolly1_k20_k25_seed0_both": cc, "corr_seed1_k20_vs_seed0_k25": cc_ctrl}
    P("")
    P("# noise at the same pixel in looks 20 and 25 (residuals against the 1024-spp looks)")
    P(f"dolly-1 (seed 0 in both): corr {cc:.3f}; seed 1 at k = 20 against dolly-1's k = 25: corr {cc_ctrl:.3f}")

    # ---------------------------------------------------------------- the matcher's anchors, pair (20, 25)
    flat = za["flat"]
    c25 = window(L["r25"], np.nanstd, int(0.8 * WIN * WIN)).ravel()[flat]
    s25 = np.sqrt(window(np.where(v2, (L["d25"] - L["r25"]) ** 2 / (1 + 1 / 16), np.nan), np.nanmean,
                         int(0.8 * WIN * WIN))).ravel()[flat]
    with np.errstate(all="ignore"):
        snr_a = np.where(s25 > 0, c25 / s25, np.nan)
    res["anchors_without_snr"] = int(np.sum(~np.isfinite(snr_a)))
    ra, rb = za["reason"], zb["reason"]
    names = matcher.REASONS
    P("")
    P(f"# the {flat.size} anchors of pair (20, 25), placed on look 25; c and sigma_64 from look 25 "
      f"(sigma_64^2 = (L_dolly1 - L_1024)^2 / (1 + 1/16), window-averaged)")
    cnt = {n: (int(np.sum(ra == i)), int(np.sum(rb == i))) for i, n in enumerate(names)}
    res["reasons_64_vs_1024"] = cnt
    P("reason | 64 spp (dolly-1) | 1024 spp")
    for n, (x, y) in sorted(cnt.items(), key=lambda t: -t[1][0]):
        P(f"{n} | {x} ({100 * x / flat.size:.1f} %) | {y} ({100 * y / flat.size:.1f} %)")
    P("")
    P(f"anchors without an SNR (off the full window, or zero noise): {res['anchors_without_snr']}")
    P("SNR c/sigma_64 bin | anchors | accepted 64 | accepted 1024 | low_ncc 64 | low_ncc 1024 | ambiguous 64 | ambiguous 1024")
    srows = []
    i_low, i_amb = names.index("low_ncc"), names.index("ambiguous")
    for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
        b = np.isfinite(snr_a) & (snr_a >= lo) & (snr_a < hi)
        if not b.any():
            continue
        r = {"snr_lo": lo, "snr_hi": None if not np.isfinite(hi) else hi, "anchors": int(b.sum()),
             "acc64": float(np.mean(ra[b] == 0)), "acc1024": float(np.mean(rb[b] == 0)),
             "low64": float(np.mean(ra[b] == i_low)), "low1024": float(np.mean(rb[b] == i_low)),
             "amb64": float(np.mean(ra[b] == i_amb)), "amb1024": float(np.mean(rb[b] == i_amb))}
        srows.append(r)
        P(f"{lo:g}-{'inf' if not np.isfinite(hi) else f'{hi:g}'} | {r['anchors']} | {r['acc64']:.3f} | {r['acc1024']:.3f} | "
          f"{r['low64']:.3f} | {r['low1024']:.3f} | {r['amb64']:.3f} | {r['amb1024']:.3f}")
    res["anchors_by_snr64"] = srows
    trans = {}
    for i, n in enumerate(names):
        sel = ra == i
        if sel.sum() == 0:
            continue
        trans[n] = {names[j]: int(np.sum(rb[sel] == j)) for j in range(len(names)) if np.sum(rb[sel] == j)}
    res["transitions_64_to_1024"] = trans
    P("")
    P("where the 64-spp anchors go at 1024 spp (rows: reason at 64 spp)")
    for n, t in trans.items():
        P(f"{n}: " + ", ".join(f"{k} {v}" for k, v in sorted(t.items(), key=lambda kv: -kv[1])))

    (out / "noise.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "noise.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figure(out / "noise.png", L, c, s64_loc, snr, m, v64, e0r, srows)
    print(f"wrote {out / 'noise.json'}, {out / 'noise.txt'}, {out / 'noise.png'}")
    ok = all(c_["ok"] and c_["control_fails"] for c_ in res["checks"].values())
    return 0 if ok else 1


def figure(path, L, c, s64_loc, snr, m, v64, e0r, srows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#0b0b0b", "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": False, "font.size": 10})
    fig = plt.figure(figsize=(18, 10), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    hi = np.nanpercentile(L["r20"], 99)
    disp = lambda x: np.clip(np.nan_to_num(x) / hi, 0, 1) ** (1 / 2.2)
    for i, (img, t) in enumerate(((disp(L["s0"]), "look 20, 64 spp (seed 0 = dolly-1's)"),
                                  (disp(L["r20"]), "look 20, 1024 spp (seed 2)"))):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t + "\nluminance, display gamma 2.2", fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(np.log10(np.where(s64_loc > 0, s64_loc, np.nan)), cmap="Blues", vmin=-3, vmax=-0.5)
    ax.set_title(f"local sigma_64 (log10), {WIN} x {WIN} window\nfrom the two 64-spp seeds", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(np.log2(np.clip(snr, 1 / 16, 64)), cmap="Blues", vmin=-2, vmax=5)
    ax.contour(np.nan_to_num(snr, nan=0), levels=[2], colors=[ORANGE], linewidths=0.8)
    ax.set_title("c / sigma_64 (log2); orange: c / sigma = 2\n(NCC 0.8 by the model, estimated)", fontsize=10)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_ticks(range(-2, 6))
    cb.set_ticklabels([f"{2.0 ** k:g}" for k in range(-2, 6)])

    ax = fig.add_subplot(gs[1, 0:2])
    cc, ss = c[m], np.sqrt(v64[m])
    ax.hexbin(np.log10(np.clip(cc, 1e-4, None)), np.log10(np.clip(ss, 1e-5, None)), gridsize=70, cmap="Greys",
              bins="log", mincnt=1, linewidths=0)
    edges = np.logspace(-3.2, 0.3, 22)
    xc, y64 = [], []
    g_ratio = np.sqrt(v64[m].mean()) / np.sqrt(max(e0r[m].mean() - v64[m].mean(), 1e-12))
    for lo, hi_ in zip(edges[:-1], edges[1:]):
        b = (cc >= lo) & (cc < hi_)
        if b.sum() < 50:
            continue
        xc.append(np.sqrt(lo * hi_))
        y64.append(np.sqrt(v64[m][b].mean()))
    lx = np.log10(xc)
    ax.plot(lx, np.log10(y64), "-o", color=BLUE, lw=2, ms=4, label="sigma_64, RMS per contrast bin (measured)")
    ax.plot(lx, np.log10(np.asarray(y64) / 4), "--", color=ORANGE, lw=2,
            label=f"sigma_64 / 4: 1024 spp by 1/sqrt(spp) (overall ratio measured: {g_ratio:.2f})")
    xx = np.linspace(-3.4, 0.4, 10)
    ax.plot(xx, xx - np.log10(2), "--", color=INK2, lw=1.2, label="sigma = c / 2: below it c / sigma >= 2")
    ax.set_xlim(-3.4, 0.4)
    ax.set_ylim(-3.6, 0.3)
    ax.set_xlabel("local contrast c (std of L_1024 in the window), log10")
    ax.set_ylabel("per-pixel noise sigma, log10 (grey: |L_s0 - L_s1| / sqrt 2)")
    ax.set_title("look 20: noise against local contrast", fontsize=11)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[1, 2:4])
    xs = np.arange(len(srows))
    lab = [f"{r['snr_lo']:g}-{'' if r['snr_hi'] is None else f'{r['snr_hi']:g}'}" for r in srows]
    ax.plot(xs, [r["acc64"] for r in srows], "-o", color=BLUE, lw=2, ms=5, label="accepted, 64 spp (dolly-1)")
    ax.plot(xs, [r["acc1024"] for r in srows], "-o", color=ORANGE, lw=2, ms=5, label="accepted, 1024 spp")
    ax.plot(xs, [r["low64"] for r in srows], "--o", color=BLUE, lw=1.2, ms=4, label="low_ncc, 64 spp")
    ax.plot(xs, [r["low1024"] for r in srows], "--o", color=ORANGE, lw=1.2, ms=4, label="low_ncc, 1024 spp")
    for x, r in zip(xs, srows):
        ax.annotate(f"{r['anchors']}", (x, 1.02), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("c / sigma_64 at the anchor (look 25); numbers: anchors per bin")
    ax.set_ylabel("fraction of anchors in the bin")
    ax.set_title("pair (20, 25): the frozen matcher's decision by the anchor's SNR at 64 spp", fontsize=11)
    ax.grid(True, axis="y", color="#e6e5e0", lw=0.6)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    fig.suptitle("Issue #7: render noise in the Classroom observations (Code, workstation, OPTIX renders)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
