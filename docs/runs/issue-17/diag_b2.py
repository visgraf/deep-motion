"""Diagnostic for K19's failure (truth-based, no method changed): why B2's patch cost loses to B off true depth edges
in the Classroom. Two measurable explanations, against the anchors' outcome under B and B2 (surface metric):
  scale     - over the window, the true point's range varies (max/min): a fixed 5x5 patch stops matching when the
              image of the surface around it grows or shrinks (a single pixel does not care);
  occlusion - the fraction of looks in which all 25 true points of the reference patch are seen (oracle.visible):
              a nearer object can cover part of the patch without covering the anchor's pixel.
Usage: python docs/runs/issue-17/diag_b2.py [--max-n N] line=runs/r2-line arc=runs/r2-arc > docs/runs/issue-17/diag_b2.txt"""
import json, sys
import numpy as np
from deep import match as mt, oracle as orc

args = sys.argv[1:]
max_n = 20000
if args and args[0] == "--max-n":
    max_n, args = int(args[1]), args[2:]
out = {}
for kv in args:
    kind, run_dir = kv.split("=", 1)
    looks, P, _, fan = mt.r1_run(run_dir)
    _, L = orc.load(run_dir)
    m = len(looks) // 2
    H, W = looks[m][3].shape
    px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=max_n)
    tedge = mt.depth_edge(looks, m, P, px)
    nb = mt.neighbourhood_ranges(looks, m, P, px)
    r_b, _, _ = mt.sweep(looks, m, px, u, fan)
    r_2, _, _ = mt.sweep_b2(looks, m, px, u, fan)
    sb, s2 = mt.surface_right(r_b, nb), mt.surface_right(r_2, nb)
    off = np.flatnonzero(~tedge)
    Pt = P[pidx[off]]
    rk = np.stack([np.linalg.norm(Pt - np.asarray(ck, float), axis=-1) for (_, ck, _, _) in L], -1)
    scale = rk.max(1) / rk.min(1)
    idx = looks[m][4]
    o = np.arange(-mt.B2_PATCH, mt.B2_PATCH + 1)
    pts = []
    for dr in o:
        for dc in o:
            q = idx[np.clip(px[off, 0] + dr, 0, H - 1), (px[off, 1] + dc) % W]
            pts.append(P[np.maximum(q, 0)])
    pts = np.stack(pts, 1)                                     # (n, 25, 3)
    allvis = np.zeros(len(off))
    for k, (path, ck, Rk, s) in enumerate(L):
        if k == m:
            continue
        v = orc.visible(orc.record(path), np.asarray(ck, float), np.asarray(Rk, float), pts.reshape(-1, 3), fan)
        allvis += v.reshape(len(off), -1).all(1)
    allvis /= (len(L) - 1)
    groups = {"both right": sb[off] & s2[off], "B only (B2 wrong)": sb[off] & ~s2[off],
              "B2 only (B wrong)": ~sb[off] & s2[off], "neither": ~sb[off] & ~s2[off]}
    res = {"off-edge anchors": int(len(off))}
    for nm, g in groups.items():
        res[nm] = {"n": int(g.sum()), "share": float(g.mean()),
                   "range max/min over the window, median": float(np.median(scale[g])) if g.any() else None,
                   "range max/min > 1.3": float(np.mean(scale[g] > 1.3)) if g.any() else None,
                   "looks with the whole patch seen, median fraction": float(np.median(allvis[g])) if g.any() else None,
                   "whole patch seen in < 80 % of looks": float(np.mean(allvis[g] < 0.8)) if g.any() else None}
    big = scale > 1.3
    res["B2 surface-right off edges, by range max/min"] = {"<= 1.3": float(s2[off][~big].mean()),
                                                           "> 1.3": float(s2[off][big].mean()) if big.any() else None}
    res["B surface-right off edges, by range max/min"] = {"<= 1.3": float(sb[off][~big].mean()),
                                                          "> 1.3": float(sb[off][big].mean()) if big.any() else None}
    out[kind] = res
print(json.dumps(out, indent=1))
