"""#9, Code: the 3-D panel's and the orbit's framing in the Classroom, before the full run (a display check only).

    .venv/bin/python docs/runs/issue-9/classroom/framing/framing.py cells runs/demo-framing/P.npz
    .venv/bin/python docs/runs/issue-9/classroom/framing/framing.py draw  runs/demo-framing/P.npz OUTDIR

cells: the demo's cells exactly as deep/demo.py's run makes them (the tour of runs/s5b2k-slong + runs/s5b2k-turn,
cell 0.02, hit_stride 23, max_cells 60000, seed 21 + 7).
draw: the end-of-tour cloud as deep/demo.py draws it, with every cell shown at its true point (at the tour's end
about 99 % are usable, so the framing is the final one): the frame's 3-D panel (az -115) and the orbit's end card at
four azimuths. Uses deep/demo.py's own Scene, cloud_panel and end_card.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "fov-render"))

from deep import demo as dm  # noqa: E402
from deep import fusion as fu  # noqa: E402
from deep import sensor as sn  # noqa: E402

STRAIGHT, TURN = ["runs/s5b2k-slong"], ["runs/s5b2k-turn"]


def cells(out):
    T = dm.tour(STRAIGHT, TURN)
    looks = [(t[0], t[1], t[2], t[3]) for t in T]
    P = fu.cells(looks, 0.02, 23)
    rng = np.random.default_rng(21 + 7)
    n = len(P)
    P = P[np.sort(rng.choice(len(P), size=60000, replace=False))]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, P=P, n_binned=n)
    print(f"[framing] {n} binned, {len(P)} kept -> {out}")


class _All:
    def __init__(self, P):
        self.X, self.usable = P, np.ones(len(P), bool)


def draw(npz, outdir):
    P = np.load(npz)["P"]
    T = dm.tour(STRAIGHT, TURN)
    sc = dm.Scene(P, T, sn.SlitFan(width=3600))
    st = _All(P)
    z = (P[:, 2] - sc.zlo) / max(sc.zhi - sc.zlo, 1e-6)
    hcol = dm.ramp(np.clip(z, 0, 1))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    dm.cloud_panel(sc, st, hcol, az=-115.0, m=len(T) - 1).save(out / "panel_az-115.png")
    n = 4
    for i in range(n):
        dm.end_card(sc, st, hcol, int(i * dm.ORBIT_S * dm.FPS / n), int(dm.ORBIT_S * dm.FPS), "range").save(
            out / f"orbit_{i}.png")
    print(f"[framing] scene lo {sc.lo.round(2).tolist()} hi {sc.hi.round(2).tolist()} z {sc.zlo:.2f}..{sc.zhi:.2f} "
          f"centre {sc.centre.round(2).tolist()} extent {sc.extent:.2f}")


if __name__ == "__main__":
    {"cells": cells, "draw": draw}[sys.argv[1]](*sys.argv[2:])
