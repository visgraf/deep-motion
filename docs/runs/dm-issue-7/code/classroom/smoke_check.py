"""Stage 5b.2 (#7): does the Classroom work with deep/sensor.py render as it stands? Reads a smoke run's records (oracle access).

    .venv/bin/python docs/runs/issue-7/code/classroom/smoke_check.py runs/s5b2-smoke-classroom
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "external/fov-render")  # run from the repo root
from fov_render.record import SampleRecord
d = Path(sys.argv[1])
run = json.loads((d / "run.json").read_text())
print("device", run.get("device"), "blend", run["blend"], "width", run["carrier"]["width"])
for lk in run["looks"]:
    o = d / lk["out_dir"]
    rec = SampleRecord.load(o / "record_C.npz", access="oracle")
    meta = json.loads((o / "meta.json").read_text())
    dist = np.asarray(rec.distance_m, float)
    hit = np.isfinite(dist) & (dist > 0)
    iid = np.asarray(rec.instance_id)
    print(lk["k"], "c", np.round(lk["c"], 3), "files", sorted(p.name for p in o.iterdir()),
          "n", dist.size, "hit %.4f" % hit.mean(), "dist min/med/max %.3f %.3f %.3f" % (dist[hit].min(), np.median(dist[hit]), dist[hit].max()),
          "instances", len(np.unique(iid[hit])), "cycles s", meta["response"]["eyes"]["C"]["render_seconds"], "wall", lk["wall_seconds"])
    # the rays that hit nothing: where they point, and what the record holds for them (deep/fusion.py's cells and
    # deep/oracle.py's visible keep only finite distance_m)
    miss = ~np.isfinite(dist)
    r = np.asarray(rec.raster_index).reshape(-1)
    W = run["carrier"]["width"]; H = W // 2
    rows, cols = r[miss] // W, r[miss] % W
    lat = 90 - (rows + 0.5) / H * 180
    bear = -((cols + 0.5) / W * 360 - 180)
    print("   misses: %.4f of rays, distance_m %s, instance_id %s, bearing %.1f..%.1f deg (+ left), elevation %.1f..%.1f deg"
          % (miss.mean(), np.unique(dist[miss])[:3], np.unique(iid[miss])[:3], bear.min(), bear.max(), lat.min(), lat.max()))
