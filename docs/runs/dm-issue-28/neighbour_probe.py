"""Chat sandbox probe (not a stage, not gated): S1's confirmation by the same 2 cm cell against confirmation by another
reference's estimate within 2 cm of the cell's fused point (stage 8's coverage distance). Dry room of #26 (2 references)."""
import sys, time, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from deep import straight as st, confirm as cf, assess as asx, e2e
paths, m4in = st.synthetic_room(cache=Path("runs/stage11-dry/room.npz"))
looks, truth, fan, refs, vis = paths["128"]
R, per_ref = st.collect(paths, None, "runs/stage11-dry", rows=["perfect", "N1024"], extra=(st.N_EDGE,), log=lambda m: None)
edges = R.pop(st.N_EDGE)
R[cf.N10] = {j: st.edge_variant(o, truth, j, looks, "rel") for j, o in edges.items()}
surf = st.surface_band(per_ref, truth, looks, None)
out = {}
for name in ("perfect", "N1024", cf.N10):
    row = R[name]
    Xs = [o["X"] for o in row.values()]; rs = [np.full(len(o["X"]), j) for j, o in row.items()]
    X, ref, Xf, nref, dom, right, cell_of = asx.fuse_and_judge(Xs, rs, per_ref, truth, fan)
    near = np.zeros((len(Xf), len(refs)), bool)
    for k, j in enumerate(refs):
        near[:, k] = asx.covered(Xf, X[ref == j])          # a point of reference j within 2 cm of the cell
    same = nref >= 2
    nb = near.sum(1) >= 2
    cov = lambda m: float(asx.covered(surf[1], Xf[right & m]).mean())
    out[name] = {"cells": int(len(Xf)), "same cell: confirmed": int(same.sum()), "right": float(right[same].mean()),
                 "coverage": cov(same), "within 2 cm: confirmed": int(nb.sum()), "right_nb": float(right[nb].mean()),
                 "coverage_nb": cov(nb), "coverage_all": cov(np.ones(len(Xf), bool))}
    print(name, out[name], flush=True)
import subprocess
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
clean = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout == ""
Path(__file__).with_suffix(".json").write_text(json.dumps({"note": "Chat sandbox probe, not a stage, not gated: the dry room "
    "of #26 (2 references, 0.1 m apart)", "computed_by": {"commit": git, "clean": clean}, "rows": out}, indent=1))
