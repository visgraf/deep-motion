"""Diagnostic for K13's failure (truth-based, no method changed): where H2's later rounds land (elevation, range
slope along the image) and how a local-plane transfer would do there. R1, Chat sandbox."""
import json, sys
import numpy as np
from deep import match as mt

kind = sys.argv[1] if len(sys.argv) > 1 else "line"
looks, P, _, fan = mt.r1_run(f"runs/r1-{kind}")
m = len(looks) // 2
H, W = looks[m][3].shape
px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=20000)
r_b, conf, best = mt.sweep(looks, m, px, u, fan)
mk = mt.reliable(best, conf)
apx, au, aidx, ag = mt.anchors(looks, m, fan, max_n=10 ** 9, frac=1.0)
g = (apx[:, 0] % 2 == 0) & (apx[:, 1] % 2 == 0)
tpx, tu, tidx = apx[g], au[g], aidx[g]
r, rnd = mt.seed_grow(looks, m, fan, px[mk], u[mk], r_b[mk], tpx, tu, mode="never")
lat = np.degrees((0.5 - (tpx[:, 0] + 0.5) / H) * np.pi)
c = looks[m][1]
rt = np.linalg.norm(P[tidx] - c, axis=-1)
Pt = P[tidx]
out = {}
for name, msk in (("round 1", rnd == 1), ("rounds 2+", rnd >= 2)):
    out[name] = {"share": float(msk.mean()),
                 "elevation: below -30 deg": float(np.mean(lat[msk] < -30)),
                 "elevation: -30..30 deg": float(np.mean(np.abs(lat[msk]) <= 30)),
                 "elevation: above 30 deg": float(np.mean(lat[msk] > 30)),
                 "true point height (z), median": float(np.median(Pt[msk, 2])),
                 "true range, median": float(np.median(rt[msk]))}
print(kind, json.dumps(out, indent=1))
