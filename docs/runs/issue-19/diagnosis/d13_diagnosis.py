"""Diagnosis for #19 (NOT the check): D1.3's statistic as motion/demo_steps.checks computes it, on the same anchors
(runs/d1-classroom/fused/anchors.npz, written by demo-fuse from demo_steps.anchors), (a) as is and (b) without the
accepted anchors whose sample has no truth hit (P_true not finite). Uses demo_steps.fuse and its constants unchanged.
Usage: python d13_diagnosis.py RUN_OUT"""
import math, sys, json
import numpy as np
from motion import demo_steps as ds
A = dict(np.load(f"{sys.argv[1]}/fused/anchors.npz"))
hit = np.isfinite(A["P_true"]).all(1)
out = {}
for label, keep in (("as_is", np.ones(len(hit), bool)), ("hits_only", hit)):
    B = {k: v[keep] for k, v in A.items()}
    rt = np.linalg.norm(B["P_true"] - B["c"], axis=1)
    res = {}
    for yaw in (0.0, ds.CONTROL_YAW_DEG):
        P = B["P"].copy()
        if yaw:
            lb = B["leg"] == 1
            ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
            P[lb] = B["c"][lb] + (P[lb] - B["c"][lb]) @ Rz.T
        F = ds.fuse(P, B["w"], B["rgb"], {"P_true": B["P_true"], "leg": B["leg"].astype(float), "rt": rt})
        rel = np.linalg.norm(F["P"] - F["P_true"], axis=1) / F["rt"]
        legv = np.round(F["leg"]).astype(int)
        res[str(yaw)] = {n: float(np.median(rel[legv == l])) for l, n in enumerate("AB")}
        res[str(yaw) + "_nan_voxels"] = {n: int(np.isnan(rel[legv == l]).sum()) for l, n in enumerate("AB")}
        if yaw == 0.0:
            res["voxels"] = int(len(rel))
            err = np.linalg.norm(F["P"] - F["P_true"], axis=1)
            res["voxel_error_m_median_p90"] = [float(np.median(err)), float(np.quantile(err, 0.9))]
    res["anchors"] = int(keep.sum())
    out[label] = res
    print(label, json.dumps(res))
print("tolerance", ds.FUSION_TOL_REL)
