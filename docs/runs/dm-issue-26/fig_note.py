"""The figures of the tech note on #26 ("Why the real matcher fell short on the straight path"), from stage11.json (Code,
workstation), stage11c.json (Code, workstation) and their rows:
  s11_precision.png  coverage against cells right, full field and M4's cone, every row of stage 11 and 11c;
  s11_sectors.png    coverage and cells right by stage 9's sectors (11c);
  s11_noise.png      coverage by gradient SNR, and N1024's rejections at NCC < 0.8 by range x SNR and sector x SNR (11c).

Drawn in a Claude Science compute cell (Chat sandbox); this file is that cell's code. Its first lines load the
figure-style helpers of the Claude Science plugin; where they are not installed it prints "figure helpers not loaded" and
draws with matplotlib's defaults.
Usage: python docs/runs/dm-issue-26/fig_note.py  (reads and writes in this folder)
"""
import glob as _glob, os, pathlib
_plugins = os.environ.get("CS_PLUGINS") or (os.environ["CS_WORKDIR"].rpartition("/claude-science-work/")[0] if os.environ.get("CS_WORKDIR") else os.environ.get("HOME", "")) + "/.claude/plugins/synced/*"
try:
    _helper_files = [f for g in ("", "~g*") for f in _glob.glob(_plugins + "/claude-science" + g + "/skills/sci-core-figure-style/kernel.py") if os.path.isfile(f) and os.access(f, os.R_OK) and (_plugins == os.environ.get("CS_PLUGINS") or os.name == "posix" and all(not s.st_mode & (0o22 if s.st_gid else 2) and s.st_uid in (0, os.getuid()) for s in map(os.stat, [f, *list(pathlib.Path(f).parents)[:8]])))]
    exec(pathlib.Path(max(_helper_files, key=os.path.getctime) if len({f.rpartition("/claude-science")[0] for f in _helper_files}) == 1 else "/helpers-not-found").read_text(encoding="utf-8"))
except OSError:
    print("figure helpers not loaded")
    apply_figure_style = lambda **k: None
    set_frame = panel_letter = goodness_arrow = lambda *a, **k: None
    panel_crops = lambda *a, **k: {}
    render_check = lambda *a, **k: dict.fromkeys(("overlaps", "outside", "cut", "small", "on_data"), ())

import json
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path("../deep-motion/docs/runs/dm-issue-26")
OUT = HERE if "__file__" in globals() else pathlib.Path(".")
apply_figure_style(sizes=(9, 8, 7))
s11 = json.load(open(HERE / "stage11.json"))["rows"]
s11c = json.load(open(HERE / "stage11c.json"))
B_ALL_OLD, B_ALL = "B cost on N's pixels and frames, 1024", "B on all pixels, N's frames, 1024"
N10, NOFF = "N1024, depth-edge test at 10 %", "N1024, depth-edge test off"

# one colour per matcher, the same in every figure (perfect and B as in the Moebius note's figure 12)
COL = {"perfect": "#0072B2", "ceiling": "#56B4E9", "B": "#D55E00", "B all": "#E69F00", "N": "#009E73",
       "N 10": "#004D38", "N off": "#7FCDB5", "M4": "#CC79A7"}


def row(name, band):
    if name in s11c["full"] and band in s11c["full"][name]:
        o = s11c["full"][name][band]
    else:
        o = s11[B_ALL_OLD if name == B_ALL else name][band]
    return 100 * o["coverage"], 100 * o["right"]


# ------------------------------------------------------------------------------------------- figure: precision
# Methods: s11_precision.png
# 1. Each point is one matcher run on the 12 references of R3's straight path in the Classroom, fused into 2 cm cells.
# 2. Coverage is the share of the surface's cells within 2 cm of a right cell; a cell is right by stage 9's surface
#    metric (within +/-10 % of the true range in a 3 x 3 neighbourhood).
# 3. Panel a scores the full field; panel b only anchors and surface seen within 1-44 deg of the heading, M4's field.
# 4. Open markers are 128-spp renders, filled markers 1024 spp; N's variants change only its depth-edge tolerance.
PTS = [("perfect", "perfect", "perfect", True), ("B128", "B, 128 spp", "B", False), ("B1024", "B", "B", True),
       (B_ALL, "B on all pixels", "B all", True), ("N128", "N, 128 spp", "N", False), ("N1024", "N (M4's 0.5 px)", "N", True),
       (N10, "N, edge test 10 %", "N 10", True), (NOFF, "N, edge test off", "N off", True)]
fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.0), gridspec_kw={"wspace": 0.34})
# label positions (coverage %, cells right %): text placed in free space, a thin leader to its point
LAB = {"a": {"perfect": (70, 98.6), "B128": (1.5, 91.4), "B1024": (17, 92.6), B_ALL: (40, 85.6),
             "N128": (1.5, 97.6), "N1024": (25, 98.6), N10: (36, 96.6), NOFF: (38, 92.0)},
       "b": {"perfect": (69, 99.96), "B128": (1.5, 94.4), "B1024": (14, 95.0), B_ALL: (35, 92.6),
             "N128": (21.5, 96.1), "N1024": (46, 97.15), N10: (50, 98.3), NOFF: (51, 95.8),
             "M4@1024": (33, 100.6), "M4@128": (6, 99.6)}}
for ax, band, key, title, ylo in ((axs[0], "full", "a", "Full field: far from perfect", 84),
                                  (axs[1], "shared 1-44", "b", "M4's field (1–44°): N matches M4", 91)):
    pts = list(PTS) + ([("M4@1024", "M4", "M4", True), ("M4@128", "M4, 128 spp", "M4", False)] if key == "b" else [])
    for name, lab, c, filled in pts:
        x, y = row(name, band)
        ax.scatter(x, y, s=30, marker="o", facecolor=COL[c] if filled else "white", edgecolor=COL[c], linewidth=1.2, zorder=3)
        tx, ty = LAB[key][name]
        near = abs(tx - x) < 4 and abs(ty - y) < 0.45
        ax.annotate(lab, (x, y), xytext=(tx, ty), color=COL[c], ha="left", va="center", fontsize=7,
                    arrowprops=None if near else dict(arrowstyle="-", color=COL[c], lw=0.6, shrinkA=1, shrinkB=3.5))
    ax.set_xlim(0, 90)
    ax.set_ylim(ylo, 101.2)
    ax.set_xlabel("coverage of the surface (%)")
    ax.set_title(title)
axs[0].set_ylabel("fused cells right (%)")
panel_letter(axs[0], "a")
panel_letter(axs[1], "b")
fig.savefig(OUT / "s11_precision.png")
figs = {"s11_precision.png": fig}

# ------------------------------------------------------------------------------------------- figure: sectors
# Methods: s11_sectors.png
# 1. Rows of stage 11c, each scored within stage 9's sectors of the angle from the heading (psi), 1024-spp renders.
# 2. A point's sector is read at its pixel in its reference; a surface cell's at the pixel of the true point it comes from.
# 3. "Perfect, SNR >= 3" keeps the perfect matcher's points where the gradient's signal-to-noise ratio along the trace is at
#    least 3 at 1024 spp (noise estimated from the 128- and 1024-spp renders' difference).
# 4. M4 sees only 1-44 deg: its 15-60 bar holds only its 15-44 part, and it has no value beyond.
SEC = ["0-15", "15-60", "60-120", "120-165", "165-180"]
SROWS = [("perfect", "perfect", "perfect"), ("perfect where gradient SNR >= 3", "perfect, SNR ≥ 3", "ceiling"),
         ("M4@1024", "M4", "M4"), ("N1024", "N (0.5 px)", "N"), (N10, "N, 10 %", "N 10"), ("B1024", "B", "B")]
bc = s11c["by_class"]["sector"]
fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.2, 4.4), gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.55})
w = 0.13
xs = np.arange(len(SEC))
TICKS = ["0–15", "15–60", "60–120", "120–165", "165–180"]
for i, (name, lab, c) in enumerate(SROWS):
    vals = [bc[s]["rows"].get(name) for s in SEC]
    v = [np.nan if (o is None or o["coverage"] is None or o["cells"] == 0) else 100 * o["coverage"] for o in vals]
    pos = xs + (i - (len(SROWS) - 1) / 2) * w
    ax.bar(pos, np.nan_to_num(v), width=w * 0.92, color=COL[c], label=lab, zorder=2)
    for p, val in zip(pos, v):
        if not np.isfinite(val):
            ax.text(p, 1.0, "–", color=COL[c], ha="center", va="bottom", fontsize=7)
ax.set_xticks(xs, TICKS)
ax.set_ylim(0, 100)
ax.set_ylabel("coverage (%)")
ax.set_title("To the side, N stays well below the noise ceiling", pad=18)
ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.9, fontsize=7)
for name, lab, c in SROWS[2:]:
    vals = [bc[s]["rows"].get(name) for s in SEC]
    v = [np.nan if (o is None or o["right"] is None) else 100 * o["right"] for o in vals]
    bx.plot(xs, v, "-o", color=COL[c], ms=4, lw=1.2, label=lab)
bx.set_xticks(xs, TICKS)
bx.set_xlim(-0.4, len(SEC) - 0.6)
bx.set_ylim(90, 100.6)
bx.set_ylabel("cells right (%)")
bx.set_xlabel("angle from the heading (°)")
bx.set_title("Where each matcher errs")
bx.legend(ncol=4, loc="lower left", handlelength=1.4, columnspacing=1.0, fontsize=7)
panel_letter(ax, "a", dx=-0.08)
panel_letter(bx, "b", dx=-0.08)
fig.savefig(OUT / "s11_sectors.png")
figs["s11_sectors.png"] = fig

# ------------------------------------------------------------------------------------------- figure: noise
# Methods: s11_noise.png
# 1. Panel a: rows of stage 11c scored within classes of the gradient's signal-to-noise ratio along the trace at 1024 spp
#    (noise estimated from the 128- and 1024-spp renders' difference; Cycles without adaptive sampling or denoiser).
# 2. Panels b and c: the share of N1024's anchors rejected because their best mean NCC over the frames is under 0.8, by
#    true range and by angle from the heading, within each SNR class; each cell counts the anchors tried in it.
SNR = ["< 1", "1-3", "3-10", ">= 10"]
SNRL = ["< 1", "1–3", "3–10", "≥ 10"]
NROWS = [("perfect", "perfect", "perfect"), ("N1024", "N (0.5 px)", "N"), (N10, "N, 10 %", "N 10"), (B_ALL, "B on all pixels", "B all")]
bs = s11c["by_class"]["snr"]
fig = plt.figure(figsize=(6.2, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.62, wspace=0.55)
ax = fig.add_subplot(gs[0, :])
w = 0.19
xs = np.arange(len(SNR))
for i, (name, lab, c) in enumerate(NROWS):
    v = [100 * bs[s]["rows"][name]["coverage"] for s in SNR]
    ax.bar(xs + (i - 1.5) * w, v, width=w * 0.92, color=COL[c], label=lab, zorder=2)
ax.set_xticks(xs, [f"{l}  ({bs[s]['surface_cells'] / 1000:.0f}k cells)" for s, l in zip(SNR, SNRL)])
ax.set_xlabel("gradient signal-to-noise ratio along the trace, 1024 spp")
ax.set_ylabel("coverage (%)")
ax.set_ylim(0, 100)
ax.set_title("Below SNR 1 no image matcher works; the perfect one ignores the image")
ax.legend(ncol=4, loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.0, columnspacing=1.0, fontsize=7)
cr = s11c["n1024_reasons"]
for k, (cname, rows_, rowl, title, letter) in enumerate((
        ("range x snr", ["< 1 m", "1-2 m", "2-3 m", ">= 3 m"], ["< 1 m", "1–2 m", "2–3 m", "≥ 3 m"],
         "Nearer fails more, at SNR ≥ 1", "b"),
        ("sector x snr", ["0-15", "15-60", "60-120", "120-165", "165-180"],
         ["0–15°", "15–60°", "60–120°", "120–165°", "165–180°"],
         "Sideways fails more, at SNR ≥ 1", "c"))):
    hx = fig.add_subplot(gs[1, k])
    M = np.array([[100 * (cr[cname][f"{r} | {s}"].get("low_ncc", [0, 0.0])[1] or 0.0) for s in SNR] for r in rows_])
    im = hx.imshow(M, cmap="cividis_r", vmin=0, vmax=100, aspect="auto")
    for i_ in range(M.shape[0]):
        for j_ in range(M.shape[1]):
            hx.text(j_, i_, f"{M[i_, j_]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i_, j_] > 55 else "black")
    hx.set_xticks(range(len(SNR)), SNRL)
    hx.set_yticks(range(len(rows_)), rowl)
    hx.set_xlabel("gradient SNR")
    hx.set_title(title)
    set_frame(hx, "none")
    hx.tick_params(length=0)
    panel_letter(hx, letter, dx=-0.32)
cb = fig.colorbar(im, ax=fig.axes[-2:], fraction=0.035, pad=0.03)
cb.set_label("N's anchors rejected at NCC < 0.8 (%)")
panel_letter(ax, "a", dx=-0.08)
fig.savefig(OUT / "s11_noise.png")
figs["s11_noise.png"] = fig

# ------------------------------------------------------------------------------------------- checks (the figure-style skill's 8.1 and 8.2)
from PIL import Image
scratch = pathlib.Path(os.environ.get("CS_WORKDIR", ".")).absolute() / ".cache"
scratch.mkdir(parents=True, exist_ok=True)
for name, f in figs.items():
    found = render_check(f, final_width_in=6.2, min_pt=7)
    findings = {k: v for k, v in found.items() if v}
    im = Image.open(OUT / name)
    dpi = round(im.info.get("dpi", (0,))[0], 1) or 400
    print(f"{name}: {round(im.size[0] / dpi, 2)} in wide (target 6.2 in);",
          f"found {findings}" if findings else "not checked" if () in found.values() else "nothing found")
    stem = name[:-4]
    for stale in scratch.glob(f"{stem}_*.png"):
        stale.unlink(missing_ok=True)
    wpx = 620
    im.resize((wpx, round(im.height * wpx / im.width))).save(scratch / f"{stem}_whole.png")
    for letter, box in panel_crops(f, dpi=dpi).items():
        im.crop(box).save(scratch / f"{stem}_crop_{letter}.png")
    plt.close(f)
