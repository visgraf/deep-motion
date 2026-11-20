"""Diagnostic for K11's failure (truth-based, no method changed): G's false far layers off depth edges against the
far flank's texture along the trace, and where the far fit's range sits against its bound. R1, Chat sandbox."""
import json, sys
import numpy as np
from deep import match as mt

kind = sys.argv[1] if len(sys.argv) > 1 else "line"
looks, P, _, fan = mt.r1_run(f"runs/r1-{kind}")
m = len(looks) // 2
H, W = looks[m][3].shape
s, c, R = looks[m][:3]
px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=3000)
tedge = mt.depth_edge(looks, m, P, px)
d = mt.trace_direction(c, R, looks[m + 1][1], looks[m + 1][2], u, fan)
bl = mt._blur(looks[m][3]); gr, gc = mt._grad(bl)
ga = np.abs(gr[px[:, 0], px[:, 1]] * d[:, 0] + gc[px[:, 0], px[:, 1]] * d[:, 1])
out = {}
q, uq, rn = {}, {}, {}
for sg in (+1, -1):
    qq = np.rint(px + sg * mt.G_FLANK_PX * d).astype(np.int64); qq[:, 0] = np.clip(qq[:, 0], 0, H - 1); qq[:, 1] %= W
    q[sg], uq[sg] = qq, mt.pixel_rays(qq, R, H, W)
    rn[sg], _, _ = mt.sweep(looks, m, qq, uq[sg], fan)
sg = +1                                      # the + flank as the far layer, the after half-window
zmax = (1.0 / rn[-1]) / mt.ORDER_MARGIN
r, conf, cst = mt.sweep(looks[m:], 0, q[1], uq[1], fan, zmax=zmax)
found = np.isfinite(cst) & mt.reliable(cst, conf)
gf = np.abs(gr[q[1][:, 0], q[1][:, 1]] * d[:, 0] + gc[q[1][:, 0], q[1][:, 1]] * d[:, 1]) / ga
off = ~tedge
out["found off edges (+ flank, after half)"] = float(found[off].mean())
out["found off edges by flank/anchor gradient"] = {f"{a}-{b}": float(found[off & (gf >= a) & (gf < b)].mean())
                                                   for a, b in ((0, .1), (.1, .25), (.25, .5), (.5, 99))}
at_bound = np.abs(1 / r / (1 / zmax) - 1) < 0.02
out["found off edges with the far range at its bound (within 2 %)"] = float(at_bound[off & found].mean())
print(kind, json.dumps(out, indent=1))
