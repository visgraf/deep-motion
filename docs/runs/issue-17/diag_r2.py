"""Diagnostic for #17 (truth-based, no method changed): why B's anchors off true depth edges are wrong, by cause.
For each anchor, along its true trace (the anchor's ray at its true range, as B scores it):
  visible  - the fraction of the window's looks in which the true point is seen (oracle.visible);
  c_vis    - B's truncated cost over the looks where it is seen (is the true trace photometrically consistent?);
  c_true   - B's cost at the true range over all looks, against B's best cost.
Causes, in order: occluded (visible < 0.8); photometric (c_vis > tau^2/4); ambiguous (B's best < c_true: another
range fits better); other. Same split for the anchors B gets right, for comparison.
Usage: python docs/runs/issue-17/diag_r2.py line=runs/r2-line arc=runs/r2-arc > docs/runs/issue-17/diag_r2.txt"""
import json, sys
import numpy as np
from deep import match as mt, oracle as orc

out = {}
for kv in sys.argv[1:]:
    kind, run_dir = kv.split("=", 1)
    looks, P, _, fan = mt.r1_run(run_dir)
    _, L = orc.load(run_dir)
    m = len(looks) // 2
    c = looks[m][1]
    px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=20000)
    tedge = mt.depth_edge(looks, m, P, px)
    nb = mt.neighbourhood_ranges(looks, m, P, px)
    r_b, conf, best = mt.sweep(looks, m, px, u, fan)
    r_true = np.linalg.norm(P[pidx] - c, axis=-1)
    right = mt.surface_right(r_b, nb)
    Xt = c + u * r_true[:, None]
    blurred = [mt._blur(lk[3]) for lk in looks]
    ref = blurred[m][px[:, 0], px[:, 1]]
    model = fan.model()
    nvis = np.zeros(len(px)); acc = np.zeros(len(px)); cnt = np.zeros(len(px))
    for k, (path, ck, Rk, s) in enumerate(L):
        if k == m:
            continue
        vis = orc.visible(orc.record(path), np.asarray(ck, float), np.asarray(Rk, float), P[pidx], fan)
        rc, _ = model.pixel_of((Xt - looks[k][1]) @ looks[k][2])
        d = np.minimum((mt._sample(blurred[k], rc) - ref) ** 2, mt.TAU ** 2)
        f = vis & np.isfinite(d)
        nvis += vis; acc += np.where(f, d, 0.0); cnt += f
    visible = nvis / (len(L) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_vis = np.where(cnt > 0, acc / cnt, np.inf)
    c_true = mt._trace_cost(blurred, looks, m, ref, u, (1.0 / r_true)[:, None], fan)[:, 0]
    occl = visible < 0.8
    phot = ~occl & (c_vis > mt.RELIABLE_COST)
    amb = ~occl & ~phot & (best < c_true)
    other = ~occl & ~phot & ~amb
    o = {"anchors": int(len(px)), "off_edges": int((~tedge).sum()), "B surface-right off edges": float(right[~tedge].mean())}
    for nm, msk in (("wrong, off edges", ~tedge & ~right), ("right, off edges", ~tedge & right)):
        o[nm] = {"n": int(msk.sum()), "occluded (visible < 0.8)": float(occl[msk].mean()),
                 "photometric (cost over visible looks > tau^2/4)": float(phot[msk].mean()),
                 "ambiguous (another range fits better)": float(amb[msk].mean()), "other": float(other[msk].mean()),
                 "visible fraction, median": float(np.median(visible[msk]))}
    out[kind] = o
print(json.dumps(out, indent=1))
