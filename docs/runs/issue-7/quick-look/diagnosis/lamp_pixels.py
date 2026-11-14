"""Issue #7 quick look (Code): the flat, bright pixels of the ahead looks, and which object they are.

    PYTHONPATH=. .venv/bin/python -W ignore docs/runs/issue-7/quick-look/diagnosis/lamp_pixels.py runs/dolly-1 0 5 20 25 40 45

Luminance as the matcher computes it (matcher.luminance of the observation's value raster). The object comes
from the oracle record's instance id (diagnosis only; the estimator reads observations). lamp_name.py names it.
"""
import sys

import numpy as np

from motion import analysis, flow_depth, logpolar as lp, matcher

run = analysis.load_run(sys.argv[1])
es = {e["k"]: e for e in run["looks"]["ahead"]}
for k in [int(x) for x in sys.argv[2:]]:
    o = flow_depth.load_observation(run, es[k])
    Y = matcher.luminance(o.rgb).reshape(-1)
    rec, R_wb, c, model = analysis.load_look(run, es[k])
    Yr = Y[rec.raster_index]
    sel = Yr > 1.9999
    ids, cnt = np.unique(rec.instance_id[sel], return_counts=True)
    e, _ = lp.e_phi(rec.direction[sel] @ R_wb)
    vals = np.unique(np.round(Yr[sel], 9))
    print(f"k={k:2d}: max luminance {np.nanmax(Y):.6f}; {int(sel.sum())} px above 1.9999, distinct values {vals.tolist()}, "
          f"instance ids {dict(zip(ids.tolist(), cnt.tolist()))}"
          + (f", eccentricity {np.degrees(e).min():.1f}..{np.degrees(e).max():.1f} deg" if sel.any() else ""))
