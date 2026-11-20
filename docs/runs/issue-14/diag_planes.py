"""Diagnostic for K14's failure (truth-based, no method changed): how close H's ceiling points lie to the true ceiling
(z = 3.2 m), and the floor's for comparison. R1, Chat sandbox."""
import json, sys
import numpy as np
from deep import match as mt

kind = sys.argv[1] if len(sys.argv) > 1 else "line"
looks, P, _, fan = mt.r1_run(f"runs/r1-{kind}")
m = len(looks) // 2
c = looks[m][1]
px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=20000)
r_b, conf, best = mt.sweep(looks, m, px, u, fan)
mk = mt.reliable(best, conf)
apx, au, aidx, ag = mt.anchors(looks, m, fan, max_n=10 ** 9, frac=1.0)
g = (apx[:, 0] % 2 == 0) & (apx[:, 1] % 2 == 0)
tpx, tu, tidx = apx[g], au[g], aidx[g]
r, _, _ = mt.seed_fill(looks, m, fan, px[mk], u[mk], r_b[mk], tpx, tu)
f = np.isfinite(r)
X = c + tu[f] * r[f][:, None]
zt = P[tidx][f][:, 2]
out = {}
for nm, msk, z0 in (("ceiling (true z >= 3.15)", zt >= 3.15, 3.2), ("floor (true z < 0.05)", zt < 0.05, 0.0)):
    dz = np.abs(X[msk, 2] - z0)
    out[nm] = {"points": int(msk.sum()), "within 3 cm of the plane": int(np.sum(dz <= 0.03)),
               "within 10 cm": int(np.sum(dz <= 0.10)), "median |dz| (m)": float(np.median(dz)),
               "true range, median (m)": float(np.median(np.linalg.norm(P[tidx][f][msk] - c, axis=-1)))}
print(kind, json.dumps(out, indent=1))

# the reliable seeds alone, against the same planes
Xs = c + u[mk] * r_b[mk][:, None]
zs = P[pidx][mk][:, 2]
out2 = {}
for nm, msk, z0 in (("seeds on the ceiling", zs >= 3.15, 3.2), ("seeds on the floor", zs < 0.05, 0.0)):
    dz = np.abs(Xs[msk, 2] - z0)
    out2[nm] = {"seeds": int(msk.sum()), "within 3 cm": int(np.sum(dz <= 0.03)), "median |dz| (m)": float(np.median(dz))}
out2["seeds, all"] = int(mk.sum())
print(kind, "reliable seeds", json.dumps(out2, indent=1))
