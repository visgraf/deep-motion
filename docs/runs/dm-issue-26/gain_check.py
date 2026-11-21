"""#26, stage 11b (Code, workstation): where stage 11b's gain g (128 over 1024 spp, about 0.973) comes from.

g is a least-squares fit through the origin. Two readings are possible: a real brightness difference between the two
renders, or a property of each run's normalisation (e2e.render_path divides luminance by the middle look's 99th
percentile). This compares, per reference, the fitted g with the ratio of means and the median ratio of the
normalised images. It also compares the two runs' raw (unnormalised) luminance statistics at the middle look.

    PYTHONPATH=. .venv/bin/python docs/runs/dm-issue-26/gain_check.py runs/r3-straight runs/r3-straight-1024
"""
import sys

import numpy as np

from deep import assess as asx, e2e, match as mt, oracle as orc

r128, r1024 = sys.argv[1], sys.argv[2]
_, L = orc.load(r128)
refs = asx.references(len(L))
l8, _, _, _ = e2e.render_path(r128, refs=refs)
l1, _, _, _ = e2e.render_path(r1024, refs=refs)
print("ref  LS-gain  ratio-of-means  median-ratio(I1024>0.05)")
for j in refs:
    I1, I8 = np.asarray(l1[j][3], float), np.asarray(l8[j][3], float)
    ok = np.isfinite(I1) & np.isfinite(I8)
    g = (I8[ok] * I1[ok]).sum() / (I1[ok] ** 2).sum()
    br = ok & (I1 > 0.05)
    print(f"{j:4d}  {g:.4f}   {I8[ok].sum() / I1[ok].sum():.4f}          {np.median(I8[br] / I1[br]):.4f}")
st = {}
for r in (r128, r1024):
    _, Lr = orc.load(r)
    rec = orc.record(Lr[len(Lr) // 2][0])
    lum = rec.value[np.isfinite(rec.distance_m)].astype(np.float64) @ mt.LUM
    st[r] = (np.quantile(lum, 0.99), lum.mean(), np.median(lum))
    print(f"{r}: raw middle look p99 {st[r][0]:.5f}, mean {st[r][1]:.5f}, median {st[r][2]:.5f}")
a, b = st[r128], st[r1024]
print(f"scale ratio p99(1024)/p99(128) {b[0] / a[0]:.4f}; raw mean ratio 128/1024 {a[1] / b[1]:.4f}; "
      f"raw median ratio {a[2] / b[2]:.4f}")
