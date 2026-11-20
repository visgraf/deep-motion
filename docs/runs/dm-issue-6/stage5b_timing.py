"""Stage 5b.1 (#6): seconds per look, from the runs' own records (Code, RTX 4090 workstation).

    .venv/bin/python docs/runs/issue-6/stage5b_timing.py runs/s5b-line runs/s5b-arc runs/s5b-pivot \
        --out docs/runs/issue-6/stage5b_timing.json

Per look: wall_seconds (deep/sensor.py's clock around Session.look: the render, the EXR and the record written) and
render_seconds (fov-render's response, Cycles alone, from each look's meta.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(x):
    x = np.asarray(x, float)
    return {"n": int(x.size), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
            "sum": float(x.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    for d in a.runs:
        d = Path(d)
        run = json.loads((d / "run.json").read_text())
        wall, cyc, bytes_ = [], [], 0
        for lk in run["looks"]:
            meta = json.loads((d / lk["out_dir"] / "meta.json").read_text())
            wall.append(lk["wall_seconds"])
            cyc.append(meta["response"]["eyes"]["C"]["render_seconds"])
            bytes_ += sum(f.stat().st_size for f in (d / lk["out_dir"]).iterdir())
        out[d.name] = {"device": run.get("device"), "width": run["carrier"]["width"], "spp": run["spp"],
                       "looks": len(run["looks"]), "run_seconds": run["seconds"],
                       "wall_seconds_per_look": summary(wall), "render_seconds_per_look": summary(cyc),
                       "disk_GB": round(bytes_ / 1e9, 2)}
        o = out[d.name]
        print(f"{d.name}: {o['looks']} looks, {o['device']}, wall {o['wall_seconds_per_look']['median']:.2f} s/look "
              f"(Cycles {o['render_seconds_per_look']['median']:.2f} s), run {o['run_seconds']:.0f} s, {o['disk_GB']} GB")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
