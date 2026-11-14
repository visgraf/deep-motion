"""#7 fix: the guarded matcher against the frozen one (1a1ffc9). Chat sandbox.

    python docs/runs/issue-7/fix/identity_check.py <frozen-worktree> > docs/runs/issue-7/fix/identity_check.txt

1. Calibration-room pair (0, 5): every array of the new estimate equals the stored smoke estimate
   (runs/m3-smoke, matcher as frozen) bit for bit.
2. Fixtures front and corridor at the stride-5 baseline: new and frozen matcher, identical outputs.
3. The #7 failure reproduced: the front fixture with a disc of look a set to a constant 2.00018 (the lamp
   glass, around e 27 deg, az 30.9 deg as in #7); the frozen matcher must raise LinAlgError, the guarded
   one must complete, with the flat anchors rejected as gn_failed.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "fov-render"))
from fov_render.sensors import FoveatedModel  # noqa: E402
from motion import analysis, flow_depth, flow_fixtures as fx, matcher  # noqa: E402

frozen_dir = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("frozen_matcher", frozen_dir / "motion" / "matcher.py",
                                              submodule_search_locations=[])
pkg = type(sys)("frozen_motion")
pkg.__path__ = [str(frozen_dir / "motion")]
sys.modules["frozen_motion"] = pkg
spec = importlib.util.spec_from_file_location("frozen_motion.matcher", frozen_dir / "motion" / "matcher.py")
fm = importlib.util.module_from_spec(spec)
sys.modules["frozen_motion.logpolar"] = __import__("motion.logpolar", fromlist=["x"])
spec.loader.exec_module(fm)

KEYS = ("d", "sigma_d", "kappa", "ncc_coarse", "ncc_second", "ncc_fine", "texture", "d_back", "reason", "support",
        "edge_dev_px", "half_dev_px", "half_sigma_px", "flat")


def same(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=np.asarray(a[k]).dtype.kind == "f")
               for k in KEYS if k in a and k in b)


run = analysis.load_run(REPO / "runs" / "m3-calib-source")
es = {e["k"]: e for e in run["looks"]["ahead"]}
oa, ob = flow_depth.load_observation(run, es[0]), flow_depth.load_observation(run, es[5])
new = flow_depth.rgb_correspondences(oa, ob, float(np.linalg.norm(ob.c - oa.c)))
old = dict(np.load(REPO / "runs" / "m3-smoke" / "pairs" / "s5" / "a0000-b0005.npz"))
print("1. calibration pair (0, 5), new vs stored frozen estimate, identical:", same(new, old))

m = FoveatedModel(253)
B = fx.BASELINES[1]
for name in ("front", "corridor"):
    pl = fx.scene(name)
    Ia = fx.render(m, pl, np.zeros(3), noise=fx.NOISE, seed=11)
    Ib = fx.render(m, pl, np.array([0, B, 0.0]), noise=fx.NOISE, seed=12)
    rn, ro = matcher.match_pair(Ia, Ib, m, B), fm.match_pair(Ia, Ib, m, B)
    print(f"2. fixture {name}, new vs frozen matcher, identical:", same(rn, ro))

pl = fx.scene("front")
Ia = fx.render(m, pl, np.zeros(3), noise=fx.NOISE, seed=11)
Ib = fx.render(m, pl, np.array([0, B, 0.0]), noise=fx.NOISE, seed=12)
rc = m.pixel_of(fx.lp.dir_of_e_phi(np.radians(27.0), np.radians(30.9))[None])[0][0]
j, i = np.meshgrid(np.arange(253), np.arange(253))
disc = np.hypot(i - rc[0], j - rc[1]) <= 8
Ia_flat = np.where(disc & np.isfinite(Ia), 2.000180216, Ia)
Ib_flat = np.where(np.hypot(i - rc[0], j - rc[1]) <= 9, Ib, Ib)        # b keeps its texture, as in #7
try:
    fm.match_pair(Ia_flat, Ib_flat, m, B)
    print("3. frozen matcher on the flat-disc case: did NOT raise (the reproduction failed)")
except np.linalg.LinAlgError as e:
    print("3. frozen matcher on the flat-disc case raises LinAlgError:", e)
r = matcher.match_pair(Ia_flat, Ib_flat, m, B)
print("   guarded matcher: completes; gn_failed", int((r["reason"] == matcher.REASONS.index("gn_failed")).sum()),
      "accepted", int((r["reason"] == 0).sum()), "of", len(r["reason"]))
