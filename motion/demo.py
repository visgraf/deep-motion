"""D1, the demonstration (#17, M-D17): the frozen M4 on a tour of straight legs joined by turns in place.

    python -m motion.dolly demo-render   --tour paths/tour_calib_L.json --out runs/d1-calib --device CPU --spp 64
    python -m motion.dolly demo-estimate --out runs/d1-calib [--phases A0 B0] [--every 1]
    python -m motion.dolly demo-fuse     --out runs/d1-calib
    python -m motion.dolly demo-checks   --out runs/d1-calib
    python -m motion.dolly demo-topview  --out runs/d1-calib --device CPU      the orthographic plan (Blender)
    python -m motion.dolly demo-video    --out runs/d1-calib [--color rgb|error]

Layout of <out>: tour.json (the spec, the compiled timeline, git), <phase>/ (a plain run per phase path, made by
`preflight` + `run --modes ahead`), pivot<l>/ (the turn in place after leg l, a run record without preflight:
display only), m4/<phase>/both5/ (the frozen estimator's results, imported from docs/runs/issue-10/frozen.json),
fused/ (the voxel fusion), topview/, video/.

What the demo does NOT do: change the estimator (its files are hashed by the freeze), put a pivot look in a window
(the alignment gate refuses it, check D1.2), or gate leg B (#5's metrics there are reported, M-D17).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import paths, tour

PREFIX = "[demo]"
WINDOW = "both5"                 # M-D17


def _dolly():
    from . import dolly
    return dolly


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))


def load_tour(out) -> dict:
    t = json.loads((Path(out) / "tour.json").read_text())
    t["_dir"] = str(Path(out).resolve())
    return t


def phases_of(t: dict) -> list[str]:
    s = t["spec"]
    return [tour.phase_name(s, l, p) for l in range(len(s["legs"])) for p in range(s["phases"])]


def phase_index(t: dict, name: str) -> tuple[int, int]:
    s = t["spec"]
    for l in range(len(s["legs"])):
        for p in range(s["phases"]):
            if tour.phase_name(s, l, p) == name:
                return l, p
    raise KeyError(name)


# ----------------------------------------------------------------------------- render

def render(a) -> int:
    d = _dolly()
    spec = tour.load(a.tour)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    blend = d._blend(spec, a.blend)
    tl = tour.timeline(spec)
    rec = {"schema": "foveal-motion-d1-tour-v1", "spec": spec, "tour_file": str(Path(a.tour)), "blend": str(blend),
           "spp": a.spp, "device": a.device, "profile": a.profile, "timeline": tl,
           "git": {"foveal-motion": d._git(d.REPO), "fov-render": d._git(d.FOV_RENDER)}, "command": sys.argv,
           "phases": {}, "pivots": {}}
    if (out / "tour.json").exists():
        old = json.loads((out / "tour.json").read_text())
        if old["spec"] != spec:
            print(f"{PREFIX} refused: {out} holds another tour", file=sys.stderr)
            return 2
        rec["phases"], rec["pivots"] = old.get("phases", {}), old.get("pivots", {})
    _write(out / "tour.json", rec)
    ns = argparse.Namespace
    for leg in range(len(spec["legs"])):
        for p in range(spec["phases"]):
            name = tour.phase_name(spec, leg, p)
            rd = out / name
            if (rd / "run.json").exists():
                print(f"{PREFIX} {name}: already rendered")
                continue
            ps = tour.phase_spec(spec, leg, p)
            sp = out / "specs" / f"{name}.json"
            _write(sp, ps)
            common = dict(path=str(sp), run=str(rd), blend=str(blend), launcher=a.launcher)
            r = d.preflight(ns(device=a.device, width=a.pf_width, spp=a.pf_spp, **common))
            if r:
                print(f"{PREFIX} refused: {name}'s preflight failed; see {rd / 'preflight.json'}", file=sys.stderr)
                return r
            r = d.run(ns(device=a.device, profile=a.profile, spp=a.spp, modes=["ahead"], e2=2.0, emax=45.0,
                         no_preflight=False, seed_base=ps["seed_base"], **common))
            if r:
                return r
            rec["phases"][name] = {"run": name, "spec": f"specs/{name}.json", "seed_base": ps["seed_base"]}
            _write(out / "tour.json", rec)
        if leg + 1 < len(spec["legs"]):
            name = f"pivot{leg}"
            if not (out / name / "run.json").exists():
                r = render_pivot(spec, leg, out / name, blend, a)
                if r:
                    return r
            rec["pivots"][str(leg)] = {"run": name}
            _write(out / "tour.json", rec)
    print(f"{PREFIX} tour rendered: {len(tl)} video frames -> {out}")
    return 0


def render_pivot(spec, leg, rd: Path, blend, a) -> int:
    """The turn in place after leg `leg`: display looks, gaze ahead, in a run record like `run`'s (no preflight:
    the head stays at the leg's last position, which every phase's preflight has checked)."""
    d = _dolly()
    from fov_render import RenderConfig, Rig, Session, sensors
    heads = tour.pivot_heads(spec, leg)
    rd.mkdir(parents=True, exist_ok=True)
    g = paths.gaze({"target_w": [0, 0, 0]}, "ahead")
    record = {"schema": "foveal-motion-run-v1", "experiment": "D1 pivot (display only)", "path": None,
              "pivot": {"leg": leg, "n": len(heads), "pivot_deg": spec["pivot_deg"]}, "blend": str(blend),
              "blend_sha256": d._sha256(Path(blend)), "profile": a.profile, "spp": a.spp,
              "device_requested": a.device, "modes": ["ahead"],
              "seeds": f"per look: C seed = {spec['seed_base'] + tour.PIVOT_SEED + 1000 * leg} + j",
              "sensor": {"type": "foveated", "E2_deg": 2.0, "e_max_deg": 45.0}, "eye": "C",
              "git": {"foveal-motion": d._git(d.REPO), "fov-render": d._git(d.FOV_RENDER)},
              "command": sys.argv, "preflight": None, "looks": {"ahead": []}}
    t0 = time.perf_counter()
    cfg = RenderConfig(profile=a.profile, device=a.device)
    with Session(Path(blend), cfg, Rig(), access="oracle", launcher=a.launcher, workdir=rd / "session",
                 prepare=sensors.Foveated(E2_deg=2.0, e_max_deg=45.0)) as s:
        for j, h in enumerate(heads):
            o = rd / "ahead" / f"k{j:04d}"
            t1 = time.perf_counter()
            look = s.look(sensors.Foveated(E2_deg=2.0, e_max_deg=45.0), ("C",), g, head=h, spp=a.spp,
                          seeds={"C": spec["seed_base"] + tour.PIVOT_SEED + 1000 * leg + j}, out_dir=o,
                          outputs=("exr", "records"))
            record["looks"]["ahead"].append({"k": j, "out_dir": str(o.relative_to(rd)), "head": h.as_dict(),
                                             "head_source": look.response.get("head_source"),
                                             "render_seconds": look.response["eyes"]["C"]["render_seconds"],
                                             "wall_seconds": round(time.perf_counter() - t1, 3)})
            print(f"{PREFIX} pivot{leg} j={j:3d} {record['looks']['ahead'][-1]['wall_seconds']:.2f} s", flush=True)
        record["device"] = s.ready.get("device")
    record["seconds"] = round(time.perf_counter() - t0, 2)
    _write(rd / "run.json", record)
    return 0


# ----------------------------------------------------------------------------- CLI

def main(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m motion.dolly demo-*", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo-render")
    p.add_argument("--tour", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--blend", default=None)
    p.add_argument("--device", default="OPTIX")
    p.add_argument("--launcher", default="auto")
    p.add_argument("--profile", default="full", choices=("small", "full"))
    p.add_argument("--spp", type=int, default=1024)
    p.add_argument("--pf-width", type=int, default=720)
    p.add_argument("--pf-spp", type=int, default=4)
    from . import demo_steps
    demo_steps.add_parsers(sub)
    a = ap.parse_args(argv)
    if a.cmd == "demo-render":
        return render(a)
    return demo_steps.dispatch(a)
