"""Diagnostic for K8's failure (truth-based, no method changed): how well B ranges E's flank pixels, by the flank's
gradient along the trace, on R1. Chat sandbox."""
import json, sys
import numpy as np
from deep import match as mt

kind = sys.argv[1] if len(sys.argv) > 1 else "line"
looks, P, _, fan = mt.r1_run(f"runs/r1-{kind}")
m = len(looks) // 2
H, W = looks[m][3].shape
s, c, R = looks[m][:3]
px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=5000)
edge = mt.depth_edge(looks, m, P, px)
d = mt.trace_direction(c, R, looks[m + 1][1], looks[m + 1][2], u, fan)
bl = mt._blur(looks[m][3])
gr, gc = mt._grad(bl)
out = {}
for sg in (+1, -1):
    q = np.rint(px + sg * mt.FLANK_PX * d).astype(np.int64)
    q[:, 0] = np.clip(q[:, 0], 0, H - 1); q[:, 1] %= W
    r, _, _ = mt.sweep(looks, m, q, mt.pixel_rays(q, R, H, W), fan)
    rt = np.linalg.norm(P[q[:, 0] * W + q[:, 1]] - c, axis=-1)
    e = mt._relerr(r, rt)
    gd = np.abs(gr[q[:, 0], q[:, 1]] * d[:, 0] + gc[q[:, 0], q[:, 1]] * d[:, 1])
    ga = np.abs(gr[px[:, 0], px[:, 1]] * d[:, 0] + gc[px[:, 0], px[:, 1]] * d[:, 1])
    rel = gd / ga
    out["+" if sg > 0 else "-"] = {
        "flank within10": float(np.mean(e <= .1)), "flank within10 at edge anchors": float(np.mean(e[edge] <= .1)),
        "flank within10 off edge anchors": float(np.mean(e[~edge] <= .1)),
        "flank gradient / anchor gradient, median": float(np.median(rel)),
        "flank within10 by gradient ratio": {f"{a}-{b}": float(np.mean(e[(rel >= a) & (rel < b)] <= .1))
                                              for a, b in ((0, .25), (.25, .5), (.5, 1), (1, 99))}}
print(kind, json.dumps(out, indent=1))
