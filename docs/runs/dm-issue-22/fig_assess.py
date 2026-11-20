"""The assessment's figure for the note "Moebius Traces" (#22): (a) coverage of the surface against fused cells right,
for the perfect matcher, B and B + H, in the Classroom (Code, workstation) and the calibration room (R0, Chat sandbox);
(b) the gap (cells the perfect matcher covers and B doesn't) by what B had there.

Drawn in a Claude Science compute cell (Chat sandbox) from assess.json and assess_r0.json; this file is that cell's code.
Its first lines load the figure-style helpers of the Claude Science plugin; where they are not installed it prints
"figure helpers not loaded" and draws with matplotlib's defaults.
Usage: python docs/runs/dm-issue-22/fig_assess.py  (writes assess_tradeoff.png here)
"""
import glob as _glob, os, pathlib
_plugins = os.environ.get("CS_PLUGINS") or (os.environ["CS_WORKDIR"].rpartition("/claude-science-work/")[0] if os.environ.get("CS_WORKDIR") else os.environ.get("HOME", "")) + "/.claude/plugins/synced/*"
try:
    _helper_files = [f for g in ("", "~g*") for f in _glob.glob(_plugins + "/claude-science" + g + "/skills/sci-core-figure-style/kernel.py") if os.path.isfile(f) and os.access(f, os.R_OK) and (_plugins == os.environ.get("CS_PLUGINS") or os.name == "posix" and all(not s.st_mode & (0o22 if s.st_gid else 2) and s.st_uid in (0, os.getuid()) for s in map(os.stat, [f, *list(pathlib.Path(f).parents)[:8]])))]
    exec(pathlib.Path(max(_helper_files, key=os.path.getmtime) if len({f.rpartition("/claude-science")[0] for f in _helper_files}) == 1 else "/helpers-not-found").read_text(encoding="utf-8"))
except OSError:
    print("figure helpers not loaded")
    apply_figure_style = lambda **k: None
    panel_letter = lambda *a, **k: None

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent
apply_figure_style(sizes=(9, 8, 7))
cls = json.load(open(HERE / "assess.json"))["numbers"]
r0 = json.load(open(HERE / "assess_r0.json"))["numbers"]
COL = {"perfect": "#0072B2", "B": "#D55E00", "B+H": "#009E73"}
pts = []
for scene, d in (("Classroom", cls), ("calibration room", r0)):
    for path in ("turn", "straight"):
        o = d[path]
        pts += [("perfect", scene, path, o["perfect"]["coverage"], o["perfect"]["right"]),
                ("B", scene, path, o["real"]["coverage"], o["real"]["right"]),
                ("B+H", scene, path, o["real_B_plus_H"]["coverage"], o["real_B_plus_H"]["right"])]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.6})
MK = {("Classroom", "turn"): ("o", True), ("Classroom", "straight"): ("s", True),
      ("calibration room", "turn"): ("o", False), ("calibration room", "straight"): ("s", False)}
for m, scene, path, cov, right in pts:
    mk, filled = MK[(scene, path)]
    ax.scatter(100 * cov, 100 * right, s=34, marker=mk, facecolor=COL[m] if filled else "white",
               edgecolor=COL[m], linewidth=1.2, zorder=3)
ax.set_xlim(0, 100)
ax.set_ylim(65, 102)
ax.set_xlabel("coverage of the surface (%)")
ax.set_ylabel("fused cells right (%)")
ax.set_title("Accurate but sparse")
ax.text(84, 97.0, "perfect", color=COL["perfect"], ha="center", va="top", fontsize=8)
ax.text(17, 97.5, "B", color=COL["B"], ha="left", va="center", fontsize=8)
ax.text(45, 79.0, "B + H", color=COL["B+H"], ha="left", va="center", fontsize=8)
h = [Line2D([], [], marker="o", ls="", mfc="0.3", mec="0.3", label="Classroom, turn"),
     Line2D([], [], marker="s", ls="", mfc="0.3", mec="0.3", label="Classroom, straight"),
     Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", label="calib. room, turn"),
     Line2D([], [], marker="s", ls="", mfc="white", mec="0.3", label="calib. room, straight")]
ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=7, handletextpad=0.2,
          borderaxespad=0.3, labelspacing=0.25)
panel_letter(ax, "a")
rows = [("Classroom, turn", cls["turn"]), ("Classroom, straight", cls["straight"]),
        ("calib. room, turn", r0["turn"]), ("calib. room, straight", r0["straight"])]
keys = [("no anchor (not among the 15 % most textured)", "no anchor", "0.80"),
        ("anchors, none reliable", "none reliable", "0.55"),
        ("reliable, no right cell near", "wrong", "0.25")]
y = np.arange(len(rows))[::-1]
left = np.zeros(len(rows))
for k, lab, c in keys:
    v = np.array([100 * r[1]["gap"][k] for r in rows])
    bx.barh(y, v, left=left, color=c, height=0.62, label=lab)
    left += v
bx.set_yticks(y)
bx.set_yticklabels([r[0] for r in rows])
bx.set_xlim(0, 100)
bx.set_xlabel("share of the gap (%)")
bx.set_title("The gap is texture first")
bx.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, fontsize=7, handlelength=0.9,
          columnspacing=0.8, handletextpad=0.3, borderaxespad=0)
panel_letter(bx, "b", dx=-0.62)
fig.savefig(HERE / "assess_tradeoff.png")
plt.close(fig)
