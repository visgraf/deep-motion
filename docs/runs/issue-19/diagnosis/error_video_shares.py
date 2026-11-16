"""#19 reading (not a check): what the error video's colours show over the whole cloud, from runs/d1-classroom's
anchors.npz, with demo_steps.fuse and demo_video's error_colours inputs (relh, h) and its ERR_MAX and the top view's cut."""
import json, sys
import numpy as np
from motion.demo_steps import fuse, hits
from motion.demo_video import ERR_MAX
A = dict(np.load("runs/d1-classroom/fused/anchors.npz"))
h = hits(A)
F = fuse(A["P"], A["w"], A["rgb"], {"relh": np.where(h, A["rel"], 0.0), "h": h.astype(float)})
cut = json.load(open("runs/d1-classroom/topview/topview.json")).get("cut_z_m", 2.2)
out = {}
for nm, keep in (("all", np.ones(len(F["P"]), bool)), (f"below_{cut}", F["P"][:, 2] < cut)):
    has = (F["h"] > 0) & keep
    rel = F["relh"][has] / F["h"][has]
    out[nm] = {"voxels": int(keep.sum()), "grey_no_truth": int((keep & ~has).sum()),
               "grey_fraction": float((keep & ~has).sum() / keep.sum()),
               "truth_voxels": int(has.sum()), "rel_median": float(np.median(rel)), "rel_p90": float(np.quantile(rel, 0.9)),
               "at_dark_end_ge_ERR_MAX": float((rel >= ERR_MAX).mean()), "below_1pct": float((rel < 0.01).mean())}
print(json.dumps({"ERR_MAX": ERR_MAX, "cut_z_m": cut, **out}, indent=1))
