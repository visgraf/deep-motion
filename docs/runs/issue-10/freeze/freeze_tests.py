import sys, json, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, 'external/fov-render')
from motion import m4_pipeline as mp, analysis
log = lambda m: print(m, flush=True)
# A: the new evaluate on the Stage 5 results reproduces Stage 5 and its controls fail
for w in ("before5", "both5"):
    old = json.loads(Path(f"docs/runs/issue-10/stage5/{w}/evaluation.json").read_text())
    new = mp.evaluate(Path("runs/m4-calib"), w, 1.625204104083527, "calibration", log=log)
    same = all(old[t][k] == new[t][k] for t in ("M4", "M3") for k in old[t] if k != "gate") and \
        all(old[t]["gate"][k] == new[t]["gate"][k] for t in ("M4", "M3") for k in old[t]["gate"])
    log(f"[A] {w}: Stage 5 numbers reproduced {same}; checks " + json.dumps([(c["id"], c["ok"], c["control_fails"]) for c in new["checks"]])
        + " control detail " + json.dumps([c["control_detail"] for c in new["checks"]], default=float))
# B: the import path
fz = Path("/tmp/s7/frozen.json")
rec = mp.write_frozen({"both5": "docs/runs/issue-10/stage5/both5", "before5": "docs/runs/issue-10/stage5/before5"}, fz)
log(f"[B] frozen test record: {json.dumps({w: v['interval_scale'] for w, v in rec['windows'].items()})}")
run = analysis.load_run("runs/m3-calib-1024")
for lab, edit in (("estimator file hash changed", lambda r: r["estimator_sha256"].update({"motion/multibaseline.py": "0" * 64})),
                  ("settings changed", lambda r: r["settings"].update({"ncc_min": 0.7})),
                  ("source freeze changed", lambda r: r["windows"]["both5"].update({"source_freeze_sha256": "0" * 64}))):
    r2 = json.loads(fz.read_text()); edit(r2); bad = Path("/tmp/s7/frozen_bad.json"); bad.write_text(json.dumps(r2))
    out = Path("/tmp/s7/bad"); shutil.rmtree(out, ignore_errors=True)
    try:
        mp.import_freeze(run, out, "both5", bad); log(f"[B] control DOES NOT FAIL: {lab}")
    except SystemExit as e:
        log(f"[B] control fails ({lab}): {e}")
out = Path("/tmp/s7/imp"); shutil.rmtree(out, ignore_errors=True)
res = mp.run_import("runs/m3-calib-1024", out, ["both5"], 1.625204104083527, 12, [], "calibration", fz, log=log)
a, b = np.load(out / "both5/r0012.npz"), np.load("runs/m4-calib/both5/r0012.npz")
log(f"[B] r 12 estimate identical to Stage 5's: {all(np.array_equal(a[k], b[k], equal_nan=True) for k in b.files)}")
log("[B] " + json.dumps({k: res['both5'][k] for k in ('held_out_refs', 'interval_scale', 'imported_from')}))
log("[B] M4 " + json.dumps(res["both5"]["M4"], default=float))
# exclusion: excluding 12 leaves nothing to evaluate -> refuse
try:
    mp.evaluate(out, "both5", 1.625204104083527, "calibration", exclude=[12], log=log); log("[B] exclusion DOES NOT remove r 12")
except SystemExit as e:
    log(f"[B] exclusion removes r 12: {e}")
