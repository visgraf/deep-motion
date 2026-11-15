"""Render chosen looks of a path at a chosen spp and seed, as a run folder `flow-depth` reads (issue #7, Code).

Chat's noise question (#7, "The next question: is it the noise?", Luiz's OK): look k = 20 of
paths/classroom_dolly.json, `ahead`, profile full, twice at 64 spp with different seeds and once at
1024 spp; k = 25 at 1024 spp too, so the frozen matcher can rerun on (20, 25) with clean looks.

`motion.dolly run` renders every head with the default seed, so this script does the same thing for a
subset: the same spec, preflight, heads, gaze, sensor, Session and outputs as `run` (motion/dolly.py at
3927982), plus `seeds={"C": seed}`. It writes the same run.json schema with only the chosen looks and
one mode; nothing in motion/ changes.

    .venv/bin/python docs/runs/issue-7/noise/render_looks.py --run runs/noise/k20-spp64-seed0 --looks 20 --spp 64 --seed 0
    .venv/bin/python docs/runs/issue-7/noise/render_looks.py --run runs/noise/k20-spp64-seed1 --looks 20 --spp 64 --seed 1
    .venv/bin/python docs/runs/issue-7/noise/render_looks.py --run runs/noise/spp1024-seed2 --looks 20 25 --spp 1024 --seed 2

The preflight is #1's (runs/dolly-1/preflight.json, copied into the new folder): same path, same
heads. Refuses a folder that already holds a run.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from motion import dolly, paths  # noqa: E402
from motion.dolly import FOV_RENDER, PREFIX, RenderConfig, Rig, Session, sensors  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(REPO / "paths" / "classroom_dolly.json"))
    ap.add_argument("--preflight-from", default=str(REPO / "runs" / "dolly-1"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--looks", type=int, nargs="+", required=True)
    ap.add_argument("--spp", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", default="ahead")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--device", default="OPTIX")
    ap.add_argument("--launcher", default="auto")
    a = ap.parse_args(argv)

    spec = dolly._load_spec(a.path)
    blend = dolly._blend(spec, None)
    rd = Path(a.run)
    if (rd / "run.json").exists():
        print(f"{PREFIX} refused: {rd} already holds a run", file=sys.stderr)
        return 2
    pre_src = Path(a.preflight_from) / "preflight.json"
    pre_ok = json.loads(pre_src.read_text())
    if not (pre_ok["ok"] and pre_ok["control_fails"]):
        print(f"{PREFIX} refused: the preflight in {pre_src} failed", file=sys.stderr)
        return 2
    spec = paths.resolved(spec, pre_ok)
    heads = paths.heads(spec)
    bad = [k for k in a.looks if not 0 <= k < len(heads)]
    if bad:
        print(f"{PREFIX} refused: looks {bad} not on the path (0..{len(heads) - 1})", file=sys.stderr)
        return 2
    rd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pre_src, rd / "preflight.json")
    sensor = sensors.Foveated(E2_deg=2.0, e_max_deg=45.0)
    cfg = RenderConfig(profile=a.profile, device=a.device)
    record = {"schema": "foveal-motion-run-v1", "experiment": "M1 dolly (subset: issue #7 noise)", "path": spec,
              "target_source": "target_ray, found by the preflight" if "target_ray" in spec else "target_w",
              "path_file": str(Path(a.path)), "blend": str(blend), "blend_sha256": dolly._sha256(blend),
              "profile": a.profile, "spp": a.spp, "seed": {"C": a.seed}, "device_requested": a.device,
              "modes": [a.mode], "sensor": {"type": "foveated", "E2_deg": 2.0, "e_max_deg": 45.0}, "eye": "C",
              "git": {"foveal-motion": dolly._git(REPO), "fov-render": dolly._git(FOV_RENDER)},
              "command": sys.argv, "preflight": {k: pre_ok[k] for k in ("ok", "control_fails", "summary")},
              "preflight_from": str(pre_src), "subset": {"looks": a.looks, "of": len(heads)},
              "looks": {a.mode: []}}
    t0 = time.perf_counter()
    with Session(blend, cfg, Rig(), access="oracle", launcher=a.launcher, workdir=rd / "session",
                 prepare=sensor) as s:
        record["session"] = {"start_seconds": round(s.start_seconds, 2), "launcher": s.launcher,
                             "device": s.ready.get("device"), "session_head": s.ready.get("head")}
        g = paths.gaze(spec, a.mode)
        for k in a.looks:
            h = heads[k]
            out = rd / a.mode / f"k{k:04d}"
            t1 = time.perf_counter()
            look = s.look(sensors.Foveated(E2_deg=2.0, e_max_deg=45.0), ("C",), g, head=h, spp=a.spp,
                          seeds={"C": a.seed}, out_dir=out, outputs=("exr", "records"))
            record["looks"][a.mode].append({"k": k, "out_dir": str(out.relative_to(rd)), "head": h.as_dict(),
                                            "head_source": look.response.get("head_source"),
                                            "seed": look.response["eyes"]["C"]["seed"],
                                            "render_seconds": look.response["eyes"]["C"]["render_seconds"],
                                            "wall_seconds": round(time.perf_counter() - t1, 3)})
            e = record["looks"][a.mode][-1]
            print(f"{PREFIX} {a.mode} k={k:3d} head {np.round(h.origin_w, 3).tolist()} {a.spp} spp seed "
                  f"{e['seed']}: render {e['render_seconds']:.3f} s, wall {e['wall_seconds']:.2f} s", flush=True)
        record["device"] = s.ready.get("device")
    record["seconds"] = round(time.perf_counter() - t0, 2)
    (rd / "run.json").write_text(json.dumps(record, indent=1))
    print(f"{PREFIX} {len(a.looks)} looks in {record['seconds']} s on {record['device']} -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
