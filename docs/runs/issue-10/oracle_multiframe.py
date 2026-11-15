"""M4a Stage 2 (#10): the multi-frame oracle ladder O-M on the calibration room (Chat sandbox).

    PYTHONPATH=. python docs/runs/issue-10/oracle_multiframe.py docs/runs/issue-10/oracle_multiframe.json > docs/runs/issue-10/oracle_multiframe.txt

Uses the geometry (Position, poses) of runs/m3-calib-source (64 spp, CPU, #6); the oracle reads no RGB.
"""
import sys, json, time, numpy as np
sys.path.insert(0,'external/fov-render')
from motion import analysis, multiframe as mf
run = analysis.load_run('runs/m3-calib-source'); ks = sorted(e['k'] for e in run['looks']['ahead'])
loads = {}
R = [10, 15]
cfgs = {"M3-like: one frame 0.5 m back": lambda r: [r - 5],
        "5 short steps back (0.5 m)": lambda r: mf.window(ks, r, 5, "before"),
        "10 frames, 5 each side (+/-0.5 m)": lambda r: mf.window(ks, r, 5, "both"),
        "10 short steps back (1 m)": lambda r: mf.window(ks, r, 10, "before")}
kinds = [("exact", 0.0), ("pixel", 0.0), ("noise", 0.05), ("noise", 0.1), ("noise", 0.25), ("noise", 0.5)]
out = {}
t0 = time.time()
for name, fr in cfgs.items():
    out[name] = {}
    for kind, sg in kinds:
        lab = kind if kind != "noise" else f"{sg} px"
        s = mf.summarize([mf.oracle(run, r, fr(r), kind, sg, seed=0, loads=loads) for r in R])
        out[name][lab] = s
print(f"{time.time()-t0:.0f} s", file=sys.stderr)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
labs = [k if k != "noise" else f"{s} px" for k, s in kinds]
print("field usable (interval <= 10 %), calibration room, reference frames r = 10, 15 (Chat sandbox, 64-spp CPU source's geometry)")
print(f"{'':36s}" + "".join(f"{l:>9s}" for l in labs))
for name in cfgs: print(f"{name:36s}" + "".join(f"{out[name][l]['field_usable']:9.3f}" for l in labs))
print("\n5-40 deg band usable"); 
for name in cfgs: print(f"{name:36s}" + "".join(f"{out[name][l]['band_5_40_usable']:9.3f}" for l in labs))
print("\nfovea (< 2 deg) usable")
for name in cfgs: print(f"{name:36s}" + "".join(f"{(out[name][l]['fovea_2deg_usable'] or 0):9.3f}" for l in labs))
print("\ncoverage of the 95 % intervals (finite estimates)")
for name in cfgs: print(f"{name:36s}" + "".join(f"{(out[name][l]['coverage_finite'] or 0):9.3f}" for l in labs))
