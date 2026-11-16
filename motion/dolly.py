"""Experiment M1, the dolly: the head moves straight along a path, one monocular foveated look per step.

The cyclopean eye ("C", at the head origin) of fov-render's rig, the Foveated sensor, the head given
per look (fov-render #20, PLAN D19). One Blender session renders both gaze modes.

    python -m motion.dolly selftest                                      numpy only, seconds
    python -m motion.dolly preflight --path paths/classroom_dolly.json --run runs/dolly-1 --device OPTIX
    python -m motion.dolly run       --path paths/classroom_dolly.json --run runs/dolly-1 --device OPTIX
    python -m motion.dolly checks    --run runs/dolly-1
    python -m motion.dolly visualize --run runs/dolly-1 --vis visuals/dolly-1
    python -m motion.dolly depth     --run runs/dolly-1 [--strides 1 5 10]   M2: range from the head's motion
    python -m motion.dolly sandbox   --run runs/calib-dolly            the calibration room on CPU, end to end
    python -m motion.dolly flow-depth-selftest --out runs/m3-selftest    M3 geometry known answers (numpy only)
    python -m motion.dolly demo-*    ...                                 D1, the demonstration (#17): motion/demo.py
    python -m motion.dolly flow-depth --run R --mode ahead --strides 1 5 10 --correspondence oracle-exact --out O
                                                                       M3: range from log-polar flow (#6)

`preflight` writes <run>/preflight.json, `run` refuses a run folder that already holds a run.json,
records the commit of this repository and of fov-render and whether both trees were clean. Exit
status: 0 done and every check passed with its control failing, 1 otherwise, 2 refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FOV_RENDER = REPO / "external" / "fov-render"
if (FOV_RENDER / "fov_render").is_dir() and str(FOV_RENDER) not in sys.path:
    sys.path.insert(0, str(FOV_RENDER))

from fov_render import RenderConfig, Rig, Session, launch, sensors  # noqa: E402

from . import analysis, paths  # noqa: E402

PREFIX = "[dolly]"


def _git(cwd: Path) -> dict:
    def g(*a):
        r = subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = g("status", "--porcelain", "--ignore-submodules=dirty")
    return {"commit": g("rev-parse", "HEAD"), "clean": status == "" if status is not None else None}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _blend(spec: dict, given) -> Path:
    b = Path(given) if given else REPO / spec["scene"]
    if not b.is_file():
        raise SystemExit(f"{PREFIX} refused: no scene at {b} (scripts/setup_scene.sh builds the Classroom)")
    return b.resolve()


def _load_spec(p) -> dict:
    try:
        return paths.load(p)
    except paths.PathError as e:
        for r in e.reasons:
            print(f"{PREFIX} refused: {p}: {r}", file=sys.stderr)
        raise SystemExit(2)


# ----------------------------------------------------------------------------- preflight

def preflight(a) -> int:
    spec = _load_spec(a.path)
    blend = _blend(spec, a.blend)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    out = run / "preflight.json"
    exe = launch.find(a.launcher)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(FOV_RENDER)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    args = ["--blend", str(blend), "--path", str(Path(a.path).resolve()), "--out", str(out.resolve()),
            "--device", a.device, "--width", str(a.width), "--spp", str(a.spp)]
    script = REPO / "motion" / "bl_preflight.py"
    cmd = ([exe, "-b", "--python-exit-code", "1", "-P", str(script), "--", *args] if launch.is_blender(exe)
           else [exe, "-m", "motion.bl_preflight", *args])
    print(f"{PREFIX} preflight: {len(paths.heads(spec))} heads + 1 control, {a.width} px, {a.spp} spp, {a.device}",
          flush=True)
    with open(run / "preflight.log", "w") as log:
        rc = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT).returncode
    tail = [ln for ln in (run / "preflight.log").read_text().splitlines() if ln.startswith("[preflight]")]
    print("\n".join(tail[-3:]))
    if not out.is_file():
        print(f"{PREFIX} preflight wrote no {out}; see {run / 'preflight.log'}", file=sys.stderr)
        return 1
    return rc


# ----------------------------------------------------------------------------- run

def run(a) -> int:
    spec = _load_spec(a.path)
    blend = _blend(spec, a.blend)
    rd = Path(a.run)
    if (rd / "run.json").exists():
        print(f"{PREFIX} refused: {rd} already holds a run", file=sys.stderr)
        return 2
    modes = list(dict.fromkeys(a.modes))
    bad = [m for m in modes if m not in paths.MODES]
    if bad:
        print(f"{PREFIX} refused: unknown modes {bad}; expected {paths.MODES}", file=sys.stderr)
        return 2
    rd.mkdir(parents=True, exist_ok=True)
    pre = rd / "preflight.json"
    pre_ok = json.loads(pre.read_text()) if pre.is_file() else None
    if pre_ok is None and not a.no_preflight:
        print(f"{PREFIX} refused: no {pre}; run `preflight` first (or --no-preflight for a quick look)", file=sys.stderr)
        return 2
    if pre_ok is not None and not (pre_ok["ok"] and pre_ok["control_fails"]) and not a.no_preflight:
        print(f"{PREFIX} refused: the preflight failed (heads {pre_ok['summary']['failing_heads']}, control fails "
              f"{pre_ok['control_fails']}); see {pre}", file=sys.stderr)
        return 2
    try:
        spec = paths.resolved(spec, pre_ok)
    except paths.PathError as e:
        print(f"{PREFIX} refused: {e}", file=sys.stderr)
        return 2
    heads = paths.heads(spec)
    cfg = RenderConfig(profile=a.profile, device=a.device)
    record = {"schema": "foveal-motion-run-v1", "experiment": "M1 dolly", "path": spec,
              "target_source": "target_ray, found by the preflight" if "target_ray" in spec else "target_w",
              "path_file": str(Path(a.path)), "blend": str(blend), "blend_sha256": _sha256(blend),
              "profile": a.profile, "spp": a.spp, "device_requested": a.device, "modes": modes,
              "seeds": ("the session's default for every look (noise pattern repeated across looks, #7)"
                        if a.seed_base is None else f"per look: C seed = {a.seed_base} + k (independent noise, #7)"),
              "sensor": {"type": "foveated", "E2_deg": a.e2, "e_max_deg": a.emax}, "eye": "C",
              "git": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)},
              "command": sys.argv, "preflight": None if pre_ok is None else
              {k: pre_ok[k] for k in ("ok", "control_fails", "summary")}, "looks": {m: [] for m in modes}}
    t0 = time.perf_counter()
    with Session(blend, cfg, Rig(), access="oracle", launcher=a.launcher, workdir=rd / "session",
                 prepare=sensors.Foveated(E2_deg=a.e2, e_max_deg=a.emax)) as s:
        record["session"] = {"start_seconds": round(s.start_seconds, 2), "launcher": s.launcher,
                             "device": s.ready.get("device"), "session_head": s.ready.get("head")}
        for mode in modes:
            g = paths.gaze(spec, mode)
            for k, h in enumerate(heads):
                out = rd / mode / f"k{k:04d}"
                t1 = time.perf_counter()
                seeds = None if a.seed_base is None else {"C": int(a.seed_base) + k}
                look = s.look(sensors.Foveated(E2_deg=a.e2, e_max_deg=a.emax), ("C",), g, head=h, spp=a.spp,
                              seeds=seeds, out_dir=out, outputs=("exr", "records"))
                record["looks"][mode].append({"k": k, "out_dir": str(out.relative_to(rd)), "head": h.as_dict(),
                                              "head_source": look.response.get("head_source"),
                                              "render_seconds": look.response["eyes"]["C"]["render_seconds"],
                                              "wall_seconds": round(time.perf_counter() - t1, 3)})
                print(f"{PREFIX} {mode} k={k:3d} head {np.round(h.origin_w, 3).tolist()} "
                      f"{record['looks'][mode][-1]['wall_seconds']:.2f} s", flush=True)
        record["device"] = s.ready.get("device")
    record["seconds"] = round(time.perf_counter() - t0, 2)
    (rd / "run.json").write_text(json.dumps(record, indent=1))
    print(f"{PREFIX} {sum(len(v) for v in record['looks'].values())} looks in {record['seconds']} s -> {rd}")
    return 0


# ----------------------------------------------------------------------------- checks

def _alignment_check(r: dict) -> dict:
    """M3.0 (#5, #6): in `ahead` mode every pair of consecutive looks has heading = eye axis and no
    rotation, from the recorded eye poses (observation fields only)."""
    from . import flow_depth
    es = sorted(r["looks"]["ahead"], key=lambda e: e["k"])
    poses = {}
    for e in es:
        o = flow_depth.load_observation(r, e)
        poses[o.k] = (o.c, o.R)
    ks = sorted(poses)
    res = flow_depth.alignment(poses, list(zip(ks[:-1], ks[1:])))
    # the estimator's loader opens records for observation only: an oracle field must be refused
    from fov_render.record import AccessError
    try:
        _ = o.rec.position_w
        refused = False
    except AccessError:
        refused = True
    res["detail"]["observation_loader_refuses_oracle"] = refused
    res["ok"] = res["ok"] and refused
    res["detail"] = {"ahead": res["detail"]}
    return res


def checks(a) -> int:
    r = analysis.load_run(a.run)
    res = [analysis.sensor_check(r), analysis.check_heads(r)]
    if "target" in r["looks"]:
        res.append(analysis.fovea_on_target(r))
    else:   # an ahead-only run has no fixation to check (#6 Stage 0); M3.0 below holds its invariant
        res.append({"id": "M1.2", "what": "in target mode the fovea's centre lands on target_w", "applicable": False,
                    "ok": None, "control_fails": None, "detail": {},
                    "note": "not applicable: the run has no target mode; M3.0 checks the ahead looks' alignment"})
    rep, flow = analysis.reprojection(r)
    res.append(rep)
    if "ahead" in r["looks"]:
        res.append(_alignment_check(r))
    summary = analysis.flow_summary(flow)
    rd = Path(a.run)
    np.savez_compressed(rd / "flow.npz", **{f"{m}__{k}": v for m, F in flow.items() for k, v in F.items()})
    out = {"schema": "foveal-motion-checks-v1", "run": str(rd), "git": r["git"], "device": r.get("device"),
           "computed_by": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)},
           "step_m": paths.step_m(r["path"]), "checks": res, "flow_by_eccentricity": summary}
    (rd / "checks.json").write_text(json.dumps(out, indent=1))
    allok = True
    for c in res:
        if c.get("applicable") is False:
            print(f"{PREFIX} {c['id']:5s} n/a   {c['note']}")
            continue
        good = c["ok"] and c["control_fails"]
        allok &= good
        print(f"{PREFIX} {c['id']:5s} {'PASS' if c['ok'] else 'FAIL'}  control "
              f"{'fails' if c['control_fails'] else 'DOES NOT FAIL'}  {c['what']}")
        for m, d in c["detail"].items():
            print(f"        {m}: {json.dumps(d)}")
    print(f"{PREFIX} flow per step ({out['step_m']:.4f} m), median deg | px, by eccentricity:")
    for m, rows in summary.items():
        cells = [f"{x['ecc_deg'][0]:g}-{x['ecc_deg'][1]:g}: "
                 + ("-" if x["median_flow_deg"] is None else f"{x['median_flow_deg']:.3f}|{x['median_flow_px']:.2f}")
                 for x in rows]
        print(f"        {m:6s} " + "  ".join(cells))
    print(f"{PREFIX} {'ALL PASS' if allok else 'NOT ALL PASS'} -> {rd / 'checks.json'}")
    return 0 if allok else 1


# ----------------------------------------------------------------------------- depth (M2)

def depth_cmd(a) -> int:
    from . import depth
    r = analysis.load_run(a.run)
    rd = Path(a.run)
    summary, examples, check = depth.run_depth(r, tuple(a.strides))
    np.savez_compressed(rd / "depth_examples.npz", **examples)
    step = paths.step_m(r["path"])
    out = {"schema": "foveal-motion-depth-v1", "run": str(rd), "git": r["git"], "device": r.get("device"),
           "computed_by": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)},
           "note_git": "git: the commits that rendered the run; computed_by: the code that computed this file",
           "step_m": step, "usable_rel": depth.USABLE_REL, "strides": list(a.strides), "check": check,
           "target_w": r["path"]["target_w"], "summary": summary}
    (rd / "depth.json").write_text(json.dumps(out, indent=1))
    good = check["ok"] and check["control_fails"]
    print(f"{PREFIX} {check['id']} {'PASS' if check['ok'] else 'FAIL'}  control "
          f"{'fails' if check['control_fails'] else 'DOES NOT FAIL'}  {check['what']}: {json.dumps(check['detail'])}")
    print(f"{PREFIX} range from motion, pixel-limited: median relative error | usable (<= {depth.USABLE_REL:.0%}) "
          f"by eccentricity; fovea: error, bracket (what the pixel supports), same-pixel share; "
          f"model: median error / (range sqrt(omega_a/12) / (|T| sin psi)), 0.87 for rounding alone")
    for mode, d in summary.items():
        for key, s in d.items():
            def g(v):
                return "-" if v is None else f"{v:.3f}"
            cells = [f"{x['bin'][0]:g}-{x['bin'][1]:g}: {g(x['median_rel'])}|{g(x['usable_fraction'])}"
                     for x in s["by_eccentricity"]]
            fv = s["fovea_2deg"]
            print(f"        {mode:6s} {s['stride'] * step:4.2f} m  " + "  ".join(cells)
                  + f"   fovea<2deg {g(fv['median_rel'])}|{g(fv['usable_fraction'])}"
                  + f" bracket {g(fv['median_bracket'])}|{g(fv['usable_bracket_fraction'])}"
                  + f" same px {g(fv['same_pixel_fraction'])}"
                  + f"   field usable {s['usable_fraction']:.3f} bracket {s['usable_bracket_fraction']:.3f}"
                  + f"   no range {s['no_range_fraction']:.2f}   measured/model {g(s['median_measured_over_pred'])}")
    print(f"{PREFIX} {'PASS' if good else 'NOT PASS'} -> {rd / 'depth.json'}")
    return 0 if good else 1


# ----------------------------------------------------------------------------- M3: flow depth

def flow_depth_selftest(a) -> int:
    from . import flow_depth
    res, num = flow_depth.selftest()
    for name, ok in res:
        print(f"{PREFIX} M3 selftest {'PASS' if ok else 'FAIL'} {name}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "selftest.json").write_text(json.dumps(
            {"schema": "foveal-motion-m3-selftest-v1", "computed_by": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)},
             "results": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num}, indent=1, default=float))
    return 0 if all(ok for _, ok in res) else 1


def flow_depth_cmd(a) -> int:
    from . import flow_depth_eval
    r = analysis.load_run(a.run)
    if a.mode not in r["looks"]:
        print(f"{PREFIX} refused: the run has no {a.mode} looks", file=sys.stderr)
        return 2
    git = {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}
    if a.correspondence == "rgb":
        from . import flow_depth
        cfg = flow_depth.estimate(r, a.mode, a.strides, Path(a.out), pairs=flow_depth.parse_pairs(a.pairs), git=git,
                                  log=lambda m: print(m, flush=True))
        print(f"{PREFIX} rgb estimate: {len(cfg['pairs'])} pairs in {cfg['seconds']} s -> {a.out}")
        return 0
    if a.pairs:
        print(f"{PREFIX} refused: --pairs is for the RGB matcher", file=sys.stderr)
        return 2
    sigmas = a.sigma_px if a.correspondence == "oracle-noise" else [0.0]
    rc = 0
    for sg in sigmas:
        out = Path(a.out) / (f"sigma-{sg:.3f}" if a.correspondence == "oracle-noise" else "")
        res = flow_depth_eval.run_oracle(r, a.correspondence, a.strides, out, sigma_px=sg, seed=a.seed, mode=a.mode, git=git)
        print(f"{PREFIX} {a.correspondence}" + (f" sigma {sg:g} px" if a.correspondence == "oracle-noise" else "")
              + f": {res['seconds']} s -> {out}")
        for c in res["checks"]:
            good = c["ok"] and c["control_fails"]
            rc = rc or (0 if good else 1)
            print(f"        {c['id']:7s} {'PASS' if c['ok'] else 'FAIL'}  control {'fails' if c['control_fails'] else 'DOES NOT FAIL'}"
                  f"  {c['what']}: {json.dumps(c['detail'])}")
        g = lambda v: "-" if v is None else f"{v:.3f}"
        for key, s in res["summary"].items():
            f, b = s["field"], s["band_5_40"]
            print(f"        stride {s['stride']:2d} ({s['B_m']:.3f} m): field usable {g(f['usable_of_all'])} within10% {g(f['within_10pct_of_all'])}"
                  f" | 5-40deg usable {g(b['usable_of_all'])} median {g(b['median_rel'])} p90 {g(b['p90_rel'])} cov {g(b['coverage'])}"
                  f" | fovea usable {g(s['fovea_2deg']['usable_of_all'])}")
    return rc


def flow_depth_fixtures(a) -> int:
    from . import flow_fixtures, matcher
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = flow_fixtures.run_fixtures(sensors.Foveated().model("full"), log=lambda m: print(m, flush=True))
    res.update({"schema": "foveal-motion-m3-fixtures-v1", "settings": matcher.SETTINGS,
                "computed_by": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)},
                "seconds": round(time.perf_counter() - t0, 1)})
    (out / "fixtures.json").write_text(json.dumps(res, indent=1, default=float))
    for g in res["gates"]:
        print(f"{PREFIX} fixture gate {'PASS' if g['ok'] else 'FAIL'} {g['fixture']}/{g['label']} B {g['B_m']:.3f}: "
              f"median {g['median_px']} p90 {g['p90_px']} accepted {g['accepted_fraction']}")
    print(f"{PREFIX} {'ALL PASS' if res['ok'] else 'NOT ALL PASS'}, controls {'all fail' if res['controls_fail'] else 'NOT ALL FAIL'}"
          f" ({res['seconds']} s) -> {out / 'fixtures.json'}")
    return 0 if res["ok"] and res["controls_fail"] else 1


def flow_depth_calibrate(a) -> int:
    from . import flow_depth_eval
    from . import flow_depth
    if a.from_result:
        cal = flow_depth_eval.import_freeze(Path(a.result), Path(a.from_result),
                                            exclude=flow_depth.parse_pairs(a.exclude) or ())
        print(f"{PREFIX} interval scale {cal['interval_scale']:.3f} imported from {a.from_result}; every pair held out except --exclude")
        print(f"{PREFIX} frozen -> {Path(a.result) / 'freeze.json'}")
        return 0
    cal = flow_depth_eval.calibrate(Path(a.result), a.parity, also=flow_depth.parse_pairs(a.also) or ())
    print(f"{PREFIX} interval scale {cal['interval_scale']:.3f} from {cal['n']} accepted anchors of {len(cal['pairs'])} pairs "
          f"({cal['set']}); |z| quantiles {json.dumps(cal['abs_z_quantiles'])}; by stride {json.dumps(cal['by_stride'])}")
    print(f"{PREFIX} frozen -> {Path(a.result) / 'freeze.json'}")
    return 0


def flow_depth_evaluate(a) -> int:
    from . import flow_depth_eval
    res = flow_depth_eval.evaluate(Path(a.result), log=lambda m: print(m, flush=True), gate=a.gate)
    res["computed_by"] = {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}
    (Path(a.result) / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    rc = 0
    for c in res["checks"]:
        rc = rc or (0 if c["ok"] and c["control_fails"] else 1)
        print(f"{PREFIX} {c['id']:7s} {'PASS' if c['ok'] else 'FAIL'}  control {'fails' if c['control_fails'] else 'DOES NOT FAIL'}"
              f"  {c['what']}: {json.dumps(c['detail'])}")
    print(f"{PREFIX} displacement error, accepted held-out interior: {json.dumps(res['displacement_error_px_interior'])}")
    print(f"{PREFIX} usable edge points, range error: {json.dumps(res['edge_usable_range_error'])}")
    g = lambda v: "-" if v is None else f"{v:.3f}"
    for key, s in res["summary"].items():
        f, b = s["field"], s["band_5_40"]
        print(f"        stride {s['stride']:2d}: field usable {g(f['usable_of_all'])} within10% {g(f['within_10pct_of_all'])}"
              f" | 5-40deg usable {g(b['usable_of_all'])} median {g(b['median_rel'])} | fovea usable {g(s['fovea_2deg']['usable_of_all'])}"
              f" | ladder " + ", ".join(f"{n} {g(v.get(key))}" for n, v in res["ladder_same_pairs_field_usable"].items()))
    return rc


def flow_depth_preview(a) -> int:
    from . import visuals
    p = visuals.flow_depth_preview(Path(a.result), Path(a.out), Path(a.reference) if a.reference else None)
    print(f"{PREFIX} wrote {p}")
    return 0


def flow_depth_figures(a) -> int:
    from . import flow_depth, flow_figures
    for pr in flow_depth.parse_pairs(a.pair):
        for p in flow_figures.figures(Path(a.result), pr, Path(a.out), scale=a.scale, reference=a.reference,
                                      truth=not a.no_truth):
            print(f"{PREFIX} wrote {p}")
    return 0


def m4_selftest(a) -> int:
    from . import multibaseline, multiframe
    res, num = multiframe.selftest()
    r2, _ = multibaseline.selftest()
    res = res + r2
    for name, ok in res:
        print(f"{PREFIX} M4 selftest {'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(ok for _, ok in res) else 1


def m6_selftest(a) -> int:
    """M6 T0 (#22): rotation removal and translation about each frame's epipole, numpy only. With --out, a record."""
    from . import derotate
    res, num = derotate.selftest()
    for name, ok in res:
        print(f"{PREFIX} M6 T0 {'PASS' if ok else 'FAIL'} {name}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rec = {"command": "m6-selftest", "checks": [{"name": n, "ok": bool(ok)} for n, ok in res], "numbers": num,
               "numpy": np.__version__, "python": sys.version.split()[0],
               "computed_by": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}}
        (out / "t0_selftest.json").write_text(json.dumps(rec, indent=1, default=float))
    return 0 if all(ok for _, ok in res) else 1


def _m4_log(m):
    print(m, flush=True)


def _m4_summary(res):
    for w, r in res.items():
        c = {x["id"]: (x["ok"], x["control_fails"]) for x in r["checks"]}
        print(f"{PREFIX} M4 {w} ({r['gate_name']} gate, held-out {r['held_out_refs']}): "
              + json.dumps({k: r[k] for k in ("M4", "M3")}, default=float))
        print(f"{PREFIX} M4 {w}: gate {'PASS' if c['M4.GATE'][0] else 'FAIL'} (control {'fails' if c['M4.GATE'][1] else 'DOES NOT FAIL'}), "
              f"interval coverage {'PASS' if c['M4.COVER'][0] else 'FAIL'} (control {'fails' if c['M4.COVER'][1] else 'DOES NOT FAIL'})")


def m4_env(out: Path, cmd: str) -> dict:
    """What ran the M4 matcher (device, torch and CUDA build, numpy, Python), beside a result: env_<cmd>.json."""
    import platform
    import torch
    from . import multibaseline as mb
    dev = mb.device()
    env = {"command": cmd, "argv": sys.argv[1:], "device": str(dev),
           "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor() or platform.machine(),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
           "computed_by": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"env_{cmd}.json").write_text(json.dumps(env, indent=1))
    print(f"{PREFIX} M4 matcher on {env['device']} ({env['device_name']}), torch {env['torch']}")
    return env


def m4_calibration(a) -> int:
    from . import m4_pipeline as mp
    git = {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}
    m4_env(Path(a.out), "m4-calibration")
    _m4_summary(mp.run_calibration(a.run, Path(a.out), a.windows, a.m3_scale, git=git, log=_m4_log))
    return 0


def m4_classroom(a) -> int:
    from . import m4_pipeline as mp
    git = {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}
    m4_env(Path(a.out), "m4-classroom")
    _m4_summary(mp.run_import(a.run, Path(a.out), a.windows, a.m3_scale, a.every, a.exclude, a.gate, Path(a.frozen),
                              git=git, log=_m4_log))
    return 0


def m4_freeze(a) -> int:
    from . import m4_pipeline as mp
    rec = mp.write_frozen({w: Path(a.source) / w for w in a.windows}, Path(a.out),
                          git={"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)})
    print(json.dumps({w: v["interval_scale"] for w, v in rec["windows"].items()}), "->", a.out)
    return 0


def m4_oracle(a) -> int:
    from . import analysis, multiframe as mf
    run = analysis.load_run(a.run)
    ks = sorted(e["k"] for e in run["looks"]["ahead"])
    refs = [r for r in a.refs if r not in set(a.exclude)]
    cfgs = {"M3 pair (r - 5, r)": lambda r: [r - 5], "before5": lambda r: mf.window(ks, r, 5, "before"),
            "both5": lambda r: mf.window(ks, r, 5, "both")}
    kinds = [("exact", 0.0), ("pixel", 0.0), ("noise", 0.05), ("noise", 0.1), ("noise", 0.25), ("noise", 0.5)]
    loads, out = {}, {"run": run["_dir"], "refs": refs,
                      "computed_by": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}, "ladder": {}}
    t0 = time.perf_counter()
    for name, fr in cfgs.items():
        ok = [r for r in refs if all(k in ks for k in fr(r)) and len(fr(r)) == (1 if name.startswith("M3") else
                                                                                  (5 if name == "before5" else 10))]
        out["ladder"][name] = {"refs": ok}
        for kind, sg in kinds:
            lab = kind if kind != "noise" else f"{sg} px"
            out["ladder"][name][lab] = mf.summarize([mf.oracle(run, r, fr(r), kind, sg, seed=0, loads=loads) for r in ok])
            print(f"{PREFIX} O-M {name} {lab}: field usable {out['ladder'][name][lab]['field_usable']:.3f}", flush=True)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


def m4_figures(a) -> int:
    from . import m4_figures as mf
    for p in mf.figures(Path(a.result), a.windows, a.ref, Path(a.out), a.m3_scale, truth=not a.no_truth):
        print(p)
    return 0


def m4_fixtures(a) -> int:
    from . import multibaseline, multibaseline_fixtures
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    git = {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)}      # when the code is loaded, not at the end
    t0 = time.perf_counter()
    res = multibaseline_fixtures.run_all(sensors.Foveated().model("full"), log=lambda m: print(m, flush=True))
    import torch
    res.update({"schema": "foveal-motion-m4-fixtures-v1", "settings": multibaseline.SETTINGS,
                "torch": torch.__version__, "device": str(multibaseline.device()),
                "computed_by": git, "seconds": round(time.perf_counter() - t0, 1)})
    (out / "fixtures.json").write_text(json.dumps(res, indent=1, default=float))
    for g in res["gates"]:
        print(f"{PREFIX} M4 fixture gate {'PASS' if g['ok'] else 'FAIL'} {g['fixture']}/{g['label']} {g['window']}: "
              f"accepted {g['accepted_fraction']} median {g['median_px']} p90 {g['p90_px']}")
    print(f"{PREFIX} {'ALL PASS' if res['ok'] else 'NOT ALL PASS'}, controls {'all fail' if res['controls_fail'] else 'NOT ALL FAIL'}"
          f" ({res['seconds']} s) -> {out / 'fixtures.json'}")
    return 0 if res["ok"] and res["controls_fail"] else 1


# ----------------------------------------------------------------------------- visualize

def visualize(a) -> int:
    from . import visuals
    return visuals.main(Path(a.run), Path(a.vis), fps=a.fps)


# ----------------------------------------------------------------------------- selftest and sandbox

def selftest(_a) -> int:
    res = list(paths.selftest())
    from . import tour
    res += [(f"tour: {n}", ok) for n, ok in tour.selftest()]          # D1 (#17)
    from . import demo_steps
    res += [(f"demo: {n}", ok) for n, ok in demo_steps.selftest()]    # D1 (#19)
    # the inverse warp round-trips the forward warp: the projection M1.3 relies on
    m = sensors.Foveated().model("small")
    rays = m.rays()
    d = rays.direction.reshape(-1, 3)[rays.valid.reshape(-1)]
    rc, inside = m.pixel_of(d)
    j, i = np.meshgrid(np.arange(m.n), np.arange(m.n))
    want = np.stack([i.reshape(-1), j.reshape(-1)], -1)[rays.valid.reshape(-1)]
    res.append(("pixel_of inverts the warp (max 1e-6 px)", bool(inside.all() and np.max(np.abs(rc - want)) < 1e-6)))
    c = int(np.argmin(analysis.eccentricity_deg(d)))          # a central ray, turned by one foveal pixel
    off = d[c:c + 1] + [[np.radians(m.s0_deg), 0.0, 0.0]]
    res.append(("control: a central ray turned by one foveal pixel moves >= 0.5 px",
                bool(np.max(np.abs(m.pixel_of(off)[0] - want[c:c + 1])) >= 0.5)))
    for name, ok in res:
        print(f"{PREFIX} selftest {'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(ok for _, ok in res) else 1


def sandbox(a) -> int:
    """The calibration room (fov-render's fixture) on CPU, small profile: preflight, run, checks, visualize."""
    from fov_render.checks.step1 import calib_room
    rd = Path(a.run)
    blend = calib_room({"work": str(REPO / "runs" / "calib-scene"), "device": "CPU"})
    common = dict(path=str(REPO / "paths" / a.path), run=str(rd), blend=blend, launcher=a.launcher)
    ns = argparse.Namespace
    steps = [("preflight", preflight, ns(device="CPU", width=360, spp=2, **common)),
             ("run", run, ns(device="CPU", profile="small", spp=a.spp, modes=a.modes, e2=2.0, emax=45.0,
                             no_preflight=False, seed_base=None, **common)),
             ("checks", checks, ns(run=str(rd))),
             ("depth", depth_cmd, ns(run=str(rd), strides=[1, 5, 10])),
             ("visualize", visualize, ns(run=str(rd), vis=str(rd / "visuals"), fps=6))]
    rc = 0
    for name, fn, args in steps:
        print(f"{PREFIX} sandbox: {name}", flush=True)
        r = fn(args)
        rc = rc or r
        if r and name in ("preflight", "run"):
            return r
    return rc


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0].startswith("demo-"):       # D1 (#17): motion/demo.py
        from . import demo
        return demo.main(args)
    ap = argparse.ArgumentParser(prog="python -m motion.dolly", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "run"):
        p = sub.add_parser(name)
        p.add_argument("--path", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--blend", default=None, help="override the spec's scene")
        p.add_argument("--device", default="OPTIX")
        p.add_argument("--launcher", default="auto")
    sub.choices["preflight"].add_argument("--width", type=int, default=720)
    sub.choices["preflight"].add_argument("--spp", type=int, default=4)
    r = sub.choices["run"]
    r.add_argument("--profile", default="full", choices=("small", "full"))
    r.add_argument("--spp", type=int, default=64)
    r.add_argument("--modes", nargs="+", default=list(paths.MODES))
    r.add_argument("--e2", type=float, default=2.0)
    r.add_argument("--emax", type=float, default=45.0)
    r.add_argument("--no-preflight", action="store_true")
    r.add_argument("--seed-base", type=int, default=None,
                   help="per-look render seeds base + k, so noise is independent between looks (#7); default: the session's")
    p = sub.add_parser("checks")
    p.add_argument("--run", required=True)
    p = sub.add_parser("visualize")
    p.add_argument("--run", required=True)
    p.add_argument("--vis", required=True)
    p.add_argument("--fps", type=int, default=6)
    p = sub.add_parser("depth")
    p.add_argument("--run", required=True)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 5, 10])
    sub.add_parser("selftest")
    p = sub.add_parser("flow-depth-selftest")
    p.add_argument("--out", default=None)
    p = sub.add_parser("flow-depth-calibrate")
    p.add_argument("--result", required=True)
    p.add_argument("--parity", default="even", choices=("even", "odd"))
    p.add_argument("--also", default=None, help="further pairs for the calibration set, e.g. 1:6")
    p.add_argument("--from-result", default=None, help="import the frozen scale of this result (M3b); opens no truth")
    p.add_argument("--exclude", default=None, help="with --from-result: pairs left out of the evaluation, e.g. 0:5,20:25")
    p = sub.add_parser("flow-depth-evaluate")
    p.add_argument("--result", required=True)
    p.add_argument("--gate", default="calibration", choices=("calibration", "classroom"))
    p = sub.add_parser("flow-depth-preview")
    p.add_argument("--result", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default=None, help="an oracle result (e.g. O1) to show beside")
    p = sub.add_parser("flow-depth-figures")
    p.add_argument("--result", required=True)
    p.add_argument("--pair", required=True, help="e.g. 0:5 (or several: 0:5,20:25)")
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=None, help="interval scale when the result is not frozen (labelled provisional)")
    p.add_argument("--reference", default=None, help="an oracle result (O1) drawn grey under the cloud")
    p.add_argument("--no-truth", action="store_true", help="omit the evaluation overlays")
    sub.add_parser("m4-selftest")
    p = sub.add_parser("m6-selftest", help="M6 T0: derotation and the generalised law (numpy)")
    p.add_argument("--out", default=None, help="write t0_selftest.json here")
    p = sub.add_parser("m4-fixtures")
    p.add_argument("--out", required=True)
    p = sub.add_parser("m4-calibration", help="M4 Stage 5 protocol: calibrate, freeze, estimate held-out, evaluate")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", nargs="+", default=["both5", "before5"])
    p.add_argument("--m3-scale", type=float, default=1.625204104083527, help="M3's frozen scale (#7)")
    p = sub.add_parser("m4-classroom", help="the frozen M4 estimator and scale on a run, every n-th reference frame")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", nargs="+", default=["both5", "before5"])
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--exclude", type=int, nargs="*", default=[])
    p.add_argument("--gate", default="classroom", choices=["classroom", "calibration"])
    p.add_argument("--frozen", default="docs/runs/issue-10/frozen.json")
    p.add_argument("--m3-scale", type=float, default=1.625204104083527, help="M3's frozen scale (#7)")
    p = sub.add_parser("m4-freeze", help="write the frozen M4 record from calibration-room freezes (Chat)")
    p.add_argument("--source", default="docs/runs/issue-10/stage5")
    p.add_argument("--windows", nargs="+", default=["both5", "before5"])
    p.add_argument("--out", default="docs/runs/issue-10/frozen.json")
    p = sub.add_parser("m4-oracle", help="the multi-frame oracle ladder O-M at given reference frames")
    p.add_argument("--run", required=True)
    p.add_argument("--refs", type=int, nargs="+", required=True)
    p.add_argument("--exclude", type=int, nargs="*", default=[])
    p.add_argument("--out", required=True)
    p = sub.add_parser("m4-figures", help="M4 process figures for one reference of a frozen m4_pipeline result")
    p.add_argument("--result", required=True, help="the m4_pipeline output directory (holding one folder per window)")
    p.add_argument("--windows", nargs="+", default=["before5", "both5"])
    p.add_argument("--ref", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--m3-scale", type=float, default=1.625204104083527, help="M3's frozen scale (#7)")
    p.add_argument("--no-truth", action="store_true")
    p = sub.add_parser("flow-depth-fixtures")
    p.add_argument("--out", required=True)
    p = sub.add_parser("flow-depth")
    p.add_argument("--run", required=True)
    p.add_argument("--mode", default="ahead")
    p.add_argument("--strides", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--correspondence", required=True, choices=("oracle-exact", "oracle-pixel", "oracle-noise", "rgb"))
    p.add_argument("--sigma-px", type=float, nargs="+", default=[0.05, 0.1, 0.25])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("sandbox")
    p.add_argument("--run", required=True)
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--launcher", default="auto")
    p.add_argument("--path", default="calib_dolly.json", help="a spec in paths/")
    p.add_argument("--modes", nargs="+", default=list(paths.MODES))
    a = ap.parse_args(argv)
    return {"preflight": preflight, "run": run, "checks": checks, "visualize": visualize, "selftest": selftest,
            "sandbox": sandbox, "depth": depth_cmd, "flow-depth-selftest": flow_depth_selftest,
            "flow-depth": flow_depth_cmd, "flow-depth-fixtures": flow_depth_fixtures,
            "flow-depth-calibrate": flow_depth_calibrate, "flow-depth-evaluate": flow_depth_evaluate,
            "flow-depth-preview": flow_depth_preview, "flow-depth-figures": flow_depth_figures,
            "m4-selftest": m4_selftest, "m6-selftest": m6_selftest, "m4-fixtures": m4_fixtures, "m4-figures": m4_figures,
            "m4-calibration": m4_calibration, "m4-classroom": m4_classroom, "m4-freeze": m4_freeze,
            "m4-oracle": m4_oracle}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
