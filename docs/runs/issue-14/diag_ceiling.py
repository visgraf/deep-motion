"""Diagnostic before the planar prior (truth-based): which true surfaces H's estimates lie on, by height band, and
how many lie on the ceiling itself. R1, Chat sandbox."""
import json, sys
import numpy as np
from deep import match as mt

kind = sys.argv[1] if len(sys.argv) > 1 else "line"
looks, P, _, fan = mt.r1_run(f"runs/r1-{kind}")
m = len(looks) // 2
px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=20000)
r_b, conf, best = mt.sweep(looks, m, px, u, fan)
mk = mt.reliable(best, conf)
apx, au, aidx, ag = mt.anchors(looks, m, fan, max_n=10 ** 9, frac=1.0)
g = (apx[:, 0] % 2 == 0) & (apx[:, 1] % 2 == 0)
tpx, tu, tidx = apx[g], au[g], aidx[g]
r, c_h, _ = mt.seed_fill(looks, m, fan, px[mk], u[mk], r_b[mk], tpx, tu)
Pt = P[tidx]
f = np.isfinite(r)
zt = Pt[:, 2]
bands = {"floor (z < 0.05)": zt < 0.05, "0.05-3.15": (zt >= 0.05) & (zt < 3.15), "ceiling (z >= 3.15)": zt >= 3.15}
out = {"targets": int(len(r)), "filled": float(f.mean()),
       "true z range": [float(np.nanmin(zt)), float(np.nanmax(zt))]}
for nm, b in bands.items():
    out[nm] = {"share of targets": float(b.mean()), "filled": float(f[b].mean()), "filled count": int((f & b).sum())}
X = looks[m][1] + tu * r[:, None]
out["estimated points with z > 3.0 (count)"] = int(np.sum(f & (X[:, 2] > 3.0)))
print(kind, json.dumps(out, indent=1))
