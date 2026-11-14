"""Issue #7 quick look (Code): why the frozen RGB matcher raises LinAlgError on pair (20, 25).

    PYTHONPATH=. .venv/bin/python -W ignore docs/runs/issue-7/quick-look/diagnosis/singular_probe.py runs/dolly-1 OUT 20:25

The estimator runs unchanged (flow_depth.estimate, stride 5, mode ahead). Only numpy.linalg.solve is wrapped:
when a batch raises, it finds the singular rows one by one and records their patches from the caller's locals
(_gn1: Ib, Ia, qb, pb, alpha), then re-raises, so the run fails exactly as it does without the wrapper.
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import numpy.linalg as nla

from motion import analysis, flow_depth, logpolar as lp

orig, seen = nla.solve, {}


def solve(H, b):
    try:
        return orig(H, b)
    except nla.LinAlgError:
        f = inspect.currentframe().f_back
        L = f.f_locals
        bad = []
        for i in range(H.shape[0]):
            try:
                orig(H[i], b[i])
            except nla.LinAlgError:
                bad.append(i)
        bad = np.array(bad)
        Ib, Ia = L["Ib"][bad], L["Ia"][bad]
        seen.update(caller=f.f_code.co_name, batch=int(H.shape[0]), singular_rows=int(bad.size), patch_px=int(Ib.shape[1]),
                    e_b_deg=np.degrees(lp.e_of_q(L["qb"][bad])).round(2).tolist(),
                    phi_b_deg=np.degrees(L["pb"][bad]).round(1).tolist(),
                    Ib_mean=np.nanmean(Ib, -1).tolist(), Ib_std=np.nanstd(Ib, -1).tolist(),
                    Ia_mean=np.nanmean(Ia, -1).tolist(), Ia_std=np.nanstd(Ia, -1).tolist(),
                    alpha=L["alpha"][bad].tolist(), H_diag=[np.diag(H[i]).tolist() for i in bad],
                    H_rank=[int(np.linalg.matrix_rank(H[i])) for i in bad])
        raise


nla.solve = np.linalg.solve = solve
run = analysis.load_run(sys.argv[1])
try:
    flow_depth.estimate(run, "ahead", [5], Path(sys.argv[2]), pairs=flow_depth.parse_pairs(sys.argv[3]), git=None,
                        log=lambda m: print(m, flush=True))
    print("no LinAlgError")
except nla.LinAlgError as e:
    print(f"LinAlgError reproduced: {e}")
print(json.dumps(seen, indent=1))
