"""Stage 5b.2 (#7), Code: what the Classroom's window band does to F1-F3 (diagnosis only; deep/ is not changed).

    .venv/bin/python docs/runs/issue-7/code/classroom/window/window_band.py runs/s5b2k-turn  OUT.json
    .venv/bin/python docs/runs/issue-7/code/classroom/window/window_band.py runs/s5b2k-slong OUT.json

The rays through the windows hit nothing (distance_m = inf); fusion.cells and oracle.visible keep only finite
distances, so the band itself makes no cells. What it can touch are the cells beside it (frames, sills, the wall
around the openings). This script makes the cells exactly as deep/fusion.py's selftest does (cell 0.02, hit_stride 23,
max_cells 200000, seed 7), and in one pass over the records takes, per look:
  - vis: oracle.visible, as fusion.visibility does;
  - edge: the cell is visible and its pixel lies within EDGE_PX of a missed ray (the band's edge);
  - the share of missed rays.
A cell is "at the band's edge" when some seeing look sees it there. Then, for those cells and for the rest: seen by
two looks, F1 (no noise, fusion.fuse), and per rung F3's usable share and coverage (one fresh noise draw, seed 11:
the same procedure as the check, not its draws).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "fov-render"))

from deep import fusion as fu  # noqa: E402
from deep import oracle as orc  # noqa: E402
from deep import sensor as sn  # noqa: E402

EDGE_PX = 2


def dilate(mask, r):
    """Square dilation by r pixels; columns wrap (longitude), rows do not."""
    out = mask.copy()
    H = mask.shape[0]
    for dy in range(-r, r + 1):
        rows = np.roll(mask, dy, axis=0)
        if dy > 0:
            rows[:dy] = False
        elif dy < 0:
            rows[dy:] = False
        for dx in range(-r, r + 1):
            out |= np.roll(rows, dx, axis=1)
    return out


def main(run, out):
    t0 = time.perf_counter()
    looks = fu.load_looks([run])
    fan = sn.SlitFan(width=3600)
    W, H = fan.width, fan.width // 2
    m = fan.model()
    P = fu.cells(looks, 0.02, 23)
    rng = np.random.default_rng(7)
    n_all = len(P)
    if len(P) > 200000:
        P = P[np.sort(rng.choice(len(P), size=200000, replace=False))]
    K = len(looks)
    vis = np.zeros((len(P), K), bool)
    edge = np.zeros((len(P), K), bool)
    miss_share = []
    for k, lk in enumerate(looks):
        rec = orc.record(lk[0])
        vis[:, k] = orc.visible(rec, lk[1], lk[2], P, fan)
        miss = np.zeros(H * W, bool)
        miss[rec.raster_index[~np.isfinite(rec.distance_m)]] = True
        miss_share.append(float(miss.mean()))
        near = dilate(miss.reshape(H, W), EDGE_PX).reshape(-1)
        rc, _ = m.pixel_of((P - lk[1]) @ lk[2])
        ij = np.rint(rc).astype(np.int64)
        ij[:, 1] %= W
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < H)
        e = np.zeros(len(P), bool)
        e[ok] = near[ij[ok, 0] * W + ij[ok, 1]]
        edge[:, k] = vis[:, k] & e
    t_pass = time.perf_counter() - t0
    band = edge.any(1)
    seen2 = vis.sum(1) >= 2
    cs = np.stack([lk[1] for lk in looks])
    rmin = np.min(np.where(vis, np.linalg.norm(P[:, None, :] - cs[None], axis=-1), np.inf), 1)
    X0, Hm, _ = fu.fuse(looks, P, fan, vis=vis)
    e0 = np.linalg.norm(X0 - P, axis=-1) / rmin
    groups = {"at the band's edge": band, "elsewhere": ~band, "all": np.ones(len(P), bool)}
    res = {"run": str(run), "edge_px": EDGE_PX, "cells_binned": int(n_all), "cells": int(len(P)),
           "missed_rays_share_per_look": {"min": min(miss_share), "median": float(np.median(miss_share)),
                                          "max": max(miss_share)},
           "groups": {}}
    for g, sel in groups.items():
        s2 = sel & seen2
        res["groups"][g] = {"cells": int(sel.sum()), "seen_by_two": float(seen2[sel].mean()) if sel.any() else None,
                            "seen_by_one_or_none": int((sel & ~seen2).sum()),
                            "F1_max_rel": float(e0[s2].max()) if s2.any() else None, "rungs": {}}
    # where the cells seen by fewer than two looks are: at the band's edge or not, and their height
    lone = ~seen2
    res["seen_by_fewer_than_two"] = {"cells": int(lone.sum()), "at_the_band_edge": int((lone & band).sum()),
                                     "z_m_quantiles": np.quantile(P[lone, 2], [0, 0.5, 1]).tolist() if lone.any() else None,
                                     "x_m_quantiles": np.quantile(P[lone, 0], [0, 0.5, 1]).tolist() if lone.any() else None}
    rng2 = np.random.default_rng(11)
    for sg in orc.SIGMAS_PX:
        X, _, _ = fu.fuse(looks, P, fan, sigma_px=sg, rng=rng2, vis=vis)
        worst, s0a, u0a = fu.ray_interval(P, Hm, vis, looks, sg, fan)
        usable = worst <= orc.USABLE_REL
        e = np.abs(np.einsum("ni,ni->n", X - P, u0a))
        inside = e <= orc.Z95 * s0a
        for g, sel in groups.items():
            u = sel & usable
            res["groups"][g]["rungs"][f"{sg:g}"] = {
                "usable": float(usable[sel].mean()) if sel.any() else None,
                "usable_cells": int(u.sum()),
                "coverage": float(inside[u].mean()) if u.any() else None,
                "coverage_sigma_halved": float((e[u] <= orc.Z95 * 0.5 * s0a[u]).mean()) if u.any() else None}
    res["seconds"] = {"cells_and_pass": round(t_pass, 1), "total": round(time.perf_counter() - t0, 1)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
