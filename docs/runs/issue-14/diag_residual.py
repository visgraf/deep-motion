"""Diagnostic after K10 (truth-based, no matcher changed): where B's edge anchors that stay wrong under the surface
metric sit, against the true ranges of wider neighbourhoods, on R1. Chat sandbox."""
import json, sys
import numpy as np
from deep import match as mt

kind = sys.argv[1] if len(sys.argv) > 1 else "line"
looks, P, _, fan = mt.r1_run(f"runs/r1-{kind}")
m = len(looks) // 2
px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=20000)
tedge = mt.depth_edge(looks, m, P, px)
r_b, conf, best = mt.sweep(looks, m, px, u, fan)
nb3 = mt.neighbourhood_ranges(looks, m, P, px, 1)
wrong = tedge & ~mt.surface_right(r_b, nb3)
out = {"edge_anchors": int(tedge.sum()), "still_wrong": int(wrong.sum())}
for rad in (2, 3, 5):
    nb = mt.neighbourhood_ranges(looks, m, P, px, rad)
    out[f"right within {2*rad+1}x{2*rad+1}"] = float(mt.surface_right(r_b, nb)[wrong].mean())
nb5 = mt.neighbourhood_ranges(looks, m, P, px, 2)
lo, hi = np.nanmin(nb5, 1), np.nanmax(nb5, 1)
w = wrong
out["in front of every 5x5 surface (< min/1.1)"] = float(np.mean(r_b[w] < lo[w] / 1.1))
out["behind every 5x5 surface (> max*1.1)"] = float(np.mean(r_b[w] > hi[w] * 1.1))
out["between 5x5 surfaces"] = float(np.mean((r_b[w] >= lo[w] / 1.1) & (r_b[w] <= hi[w] * 1.1)))
out["marked reliable"] = float(mt.reliable(best, conf)[w].mean())
out["depth ratio max/min in 5x5, median"] = float(np.median(hi[w] / lo[w]))
print(kind, json.dumps(out, indent=1))
