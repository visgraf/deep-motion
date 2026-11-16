"""M6a T1 (#22): pure rotation in the calibration room. Derotation alone, and what it costs in resolution.

The head stays at one point and turns in place (yaw to the left), gaze `ahead`. Look 0 is the reference (yaw 0);
looks 1.. turn by the given yaws; a last look repeats yaw 0 with another seed, so that two renders of the same pose
give the noise floor. Rendering: `render`, a run record like D1's pivot (no preflight: the head is the calibration
room's EYE, which fov-render's scene gate checks, and it does not move).

Checks (`analyze`, each with a control that must fail; proposed in #22 before any T1 render):

    T1.0  poses     every look's eye centre is the reference's (<= 1e-6 m) and its rotation from the reference
                    is the requested yaw about world +Z (<= 1e-6 rad, float32 matrices); control: the next look's
                    yaw requested instead
    T1.1  rays      frame k's model ray at each rendered pixel, derotated into the reference with the recorded poses
                    (R_r^T R_k) and placed through pixel_of: the reference's model ray there is the world ray k's
                    record holds for that pixel (<= 1e-4 deg: float32 directions); control: a 0.5 deg yaw error

Two corrections after the first analysis of the T1 render (cc0929e), both because a control did not fail:
T1.1 compared k's world rays with themselves through R_r alone (R_k never entered: a tautology, its control could
not fail); T1.2's control took the points at e_r < 5 deg, but the tolerance is set by k's pixel where they land
(at 30 deg of yaw they land at 25-35 deg, where 2 px is ~1.6 deg), so it now takes those landing at e_k < 5 deg,
as its stated reason requires. No threshold moved.
    T1.2  surface   every reference hit, derotated into frame k, lands on a pixel whose Position is the same
                    point within 2 pixels at the range (M1.3's tolerance), for >= 0.80 of those inside k's field
                    (M1.3's REPROJECT_MIN), every frame; control: a 0.5 deg yaw error must lose at least half of
                    the agreeing points that land at e_k < 5 deg in frame k (2 px there is <= 0.35 deg)

Reported, not gated (M-D18: measure before compensating): the photometric residual |L_k(derotated) - L_r| / L_r
per eccentricity at which frame k samples the point, against the noise floor of the repeated render.
Observation only for T1.1 and the residual; T1.2 uses Position (oracle) and is labelled so.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from fov_render import warp
from fov_render.sensors import _SHADER_TO_LOCAL

from . import analysis, derotate as dr, flow_depth, logpolar as lp, matcher, paths

PREFIX = "[m6-t1]"
REPROJECT_MIN = analysis.REPROJECT_MIN
TOL_PX = analysis.TOL_PIXELS
EYE_W = (0.0, 0.0, 1.6)            # the calibration room's EYE (fov-render calib_room.json)
FACING_W = (0.0, 1.0, 0.0)         # toward the Siemens-star rings centred on (0, 2, 1.6)
YAWS_DEG = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
ECC_BINS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0)


# ----------------------------------------------------------------------------- render

def heads(yaws_deg):
    from fov_render import HeadPose
    out = []
    for a in (0.0,) + tuple(yaws_deg) + (0.0,):
        y = np.radians(90.0 + a)                          # facing +Y is yaw 90 deg about +Z; + a turns left
        out.append(HeadPose(paths.head_axes([np.cos(y), np.sin(y), 0.0]), np.asarray(EYE_W, float)))
    return out


def render(out, yaws_deg=YAWS_DEG, spp=256, seed_base=7000, profile="full", device="CPU", launcher=None, git=None) -> int:
    from fov_render import RenderConfig, Rig, Session, sensors
    from fov_render.checks.step1 import calib_room
    from . import dolly
    rd = Path(out)
    if (rd / "run.json").exists():
        print(f"{PREFIX} refused: {rd} already holds a run", file=sys.stderr)
        return 2
    blend = calib_room({"work": str(dolly.REPO / "runs" / "calib-scene"), "device": "CPU"})
    hs = heads(yaws_deg)
    yaws = (0.0,) + tuple(yaws_deg) + (0.0,)
    g = paths.gaze({"target_w": [0, 0, 0]}, "ahead")
    rd.mkdir(parents=True, exist_ok=True)
    record = {"schema": "foveal-motion-run-v1", "experiment": "M6a T1: pure rotation (#22)", "path": None,
              "t1": {"eye_w": list(EYE_W), "facing_w": list(FACING_W), "yaws_deg": list(yaws),
                     "reference": 0, "noise_repeat": len(yaws) - 1},
              "blend": str(blend), "blend_sha256": dolly._sha256(Path(blend)), "profile": profile, "spp": spp,
              "device_requested": device, "modes": ["ahead"], "seeds": f"per look: C seed = {seed_base} + k",
              "sensor": {"type": "foveated", "E2_deg": 2.0, "e_max_deg": 45.0}, "eye": "C",
              "git": git or {"foveal-motion": dolly._git(dolly.REPO), "fov-render": dolly._git(dolly.FOV_RENDER)},
              "command": sys.argv, "preflight": None, "looks": {"ahead": []}}
    t0 = time.perf_counter()
    cfg = RenderConfig(profile=profile, device=device)
    with Session(Path(blend), cfg, Rig(), access="oracle", launcher=launcher, workdir=rd / "session",
                 prepare=sensors.Foveated(E2_deg=2.0, e_max_deg=45.0)) as s:
        for k, (h, a) in enumerate(zip(hs, yaws)):
            o = rd / "ahead" / f"k{k:04d}"
            t1 = time.perf_counter()
            look = s.look(sensors.Foveated(E2_deg=2.0, e_max_deg=45.0), ("C",), g, head=h, spp=spp,
                          seeds={"C": seed_base + k}, out_dir=o, outputs=("exr", "records"))
            record["looks"]["ahead"].append({"k": k, "yaw_deg": a, "out_dir": str(o.relative_to(rd)), "head": h.as_dict(),
                                             "head_source": look.response.get("head_source"),
                                             "render_seconds": look.response["eyes"]["C"]["render_seconds"],
                                             "wall_seconds": round(time.perf_counter() - t1, 3)})
            print(f"{PREFIX} k={k} yaw {a:5.1f} deg {record['looks']['ahead'][-1]['wall_seconds']:.1f} s", flush=True)
        record["device"] = s.ready.get("device")
    record["seconds"] = round(time.perf_counter() - t0, 2)
    (rd / "run.json").write_text(json.dumps(record, indent=1))
    print(f"{PREFIX} {len(hs)} looks in {record['seconds']} s -> {rd}")
    return 0


# ----------------------------------------------------------------------------- geometry helpers

def dir_of_rc(model, rc) -> np.ndarray:
    """Eye-frame unit direction at fractional raster position(s) rc = (row, col): pixel_of's inverse."""
    rc = np.asarray(rc, float)
    n = model.n
    px = (rc[..., 1] + 0.5) / n - 0.5
    py = 0.5 - (rc[..., 0] + 0.5) / n
    return lp.unit(warp.warp_direction(px, py, model.e2, model.emax) * _SHADER_TO_LOCAL)


def yaw_about_z(R_a, R_b) -> float:
    """Signed rotation from a to b about world +Z (rad), for level heads."""
    fa, fb = R_a @ [0, 0, -1.0], R_b @ [0, 0, -1.0]
    return float(np.arctan2(fa[0] * fb[1] - fa[1] * fb[0], fa[0] * fb[0] + fa[1] * fb[1]))


def _yaw_err(R, deg):
    """R turned further about world +Z by deg (a pose error)."""
    a = np.radians(deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    return Rz @ R


def _bilinear(img, rc):
    return lp.bilinear(img, rc)


# ----------------------------------------------------------------------------- the checks

def ray_check(rec_k, model, R_r, R_k):
    """T1.1: frame k's model rays at its rendered pixels, derotated into r (R_r^T R_k) and placed through pixel_of;
    r's model ray there, in the world, against the world ray k's record holds for that pixel (deg)."""
    w = lp.unit(rec_k.direction.astype(np.float64))                  # the renderer's world rays at k's pixels
    d_k = lp.unit(model.rays().direction.reshape(-1, 3)[rec_k.raster_index].astype(np.float64))
    d_r = dr.derotate(d_k, R_k, R_r)                                  # k's eye frame -> r's
    rc, inside = model.pixel_of(d_r)
    back = dir_of_rc(model, rc[inside]) @ R_r.T                       # r's model ray there, world
    return analysis.angle_deg(back, w[inside]), inside


def surface_check(rec_r, rec_k, model, R_r, c_r, R_k):
    """T1.2 (oracle: Position): r's hits derotated into k; agreement at the nearest pixel within TOL_PX px at range."""
    hit = np.isfinite(rec_r.distance_m)
    P = rec_r.position_w[hit].astype(np.float64)
    d_loc = lp.unit((P - c_r) @ R_r)
    e_r = analysis.eccentricity_deg(d_loc)
    dk = dr.derotate(d_loc, R_r, R_k)
    rc, inside = model.pixel_of(dk)
    n = model.n
    ij = np.rint(rc).astype(np.int64)
    inr = inside & np.all((ij >= 0) & (ij < n), axis=-1)
    inv = np.full(n * n, -1, dtype=np.int64)
    inv[rec_k.raster_index] = np.arange(rec_k.n_samples)
    s = np.full(len(P), -1, dtype=np.int64)
    s[inr] = inv[ij[inr, 0] * n + ij[inr, 1]]
    considered = s >= 0
    q = s[considered]
    rng = np.linalg.norm(P[considered] - c_r, axis=-1)
    err = np.linalg.norm(rec_k.position_w[q] - P[considered], axis=-1)
    tol = TOL_PX * rng * np.sqrt(rec_k.footprint_sr[q])
    agree = np.zeros(len(P), bool)
    agree[considered] = np.isfinite(rec_k.distance_m[q]) & (err <= tol)
    return agree, considered, e_r, analysis.eccentricity_deg(dk)


def photometric(ob_r, ob_k, model, R_r, R_k):
    """Observation only: r's pixel-centre rays derotated into k, L_k bilinear there against L_r at r's pixel.
    Returns per valid r pixel: e_r, e_k (deg), relative residual |L_k - L_r| / L_r (nan outside k's field)."""
    rays = model.rays()
    valid = rays.valid
    d = lp.unit(rays.direction[valid].astype(np.float64))
    Lr = matcher.luminance(ob_r.rgb)[valid]
    Lk_img = matcher.luminance(ob_k.rgb)
    dk = dr.derotate(d, R_r, R_k)
    rc, inside = model.pixel_of(dk)
    Lk = np.full(len(d), np.nan)
    Lk[inside] = _bilinear(Lk_img, rc[inside])
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(Lk - Lr) / Lr
    return analysis.eccentricity_deg(d), analysis.eccentricity_deg(dk), rel, rc, inside


def _binned(x, v, bins=ECC_BINS):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(v)
        out.append({"bin_deg": [lo, hi], "n": int(m.sum()),
                    "median": float(np.median(v[m])) if m.any() else None,
                    "p90": float(np.percentile(v[m], 90)) if m.any() else None})
    return out


def analyze(run_dir, out=None, log=print) -> dict:
    run = analysis.load_run(run_dir)
    es = sorted(run["looks"]["ahead"], key=lambda e: e["k"])
    yaws = run["t1"]["yaws_deg"]
    r, nrep = run["t1"]["reference"], run["t1"]["noise_repeat"]
    rec_r, R_r, c_r, model = analysis.load_look(run, es[r])
    ob_r = flow_depth.load_observation(run, es[r])
    checks, per = [], []
    pose_err_c, pose_err_a, ctrl_pose = [], [], []
    ray_max, ray_ctrl_min = 0.0, np.inf
    surf_ok, surf_ctrl_ok = True, True
    for e in es:
        k = e["k"]
        if k == r:
            continue
        rec_k, R_k, c_k, _ = analysis.load_look(run, e)
        ob_k = flow_depth.load_observation(run, e)
        # T1.0
        pose_err_c.append(float(np.linalg.norm(c_k - c_r)))
        got = yaw_about_z(R_r, R_k)
        pose_err_a.append(abs(got - np.radians(yaws[k])))
        nxt = yaws[k + 1] if k + 1 < len(yaws) else yaws[1]
        ctrl_pose.append(abs(got - np.radians(nxt)))
        # T1.1
        ang, _ = ray_check(rec_k, model, R_r, R_k)
        angc, _ = ray_check(rec_k, model, R_r, _yaw_err(R_k, 0.5))
        ray_max = max(ray_max, float(np.max(ang)))
        ray_ctrl_min = min(ray_ctrl_min, float(np.median(angc)))
        # T1.2 (oracle)
        agree, cons, e_r, e_k = surface_check(rec_r, rec_k, model, R_r, c_r, R_k)
        agree_c, cons_c, _, _ = surface_check(rec_r, rec_k, model, R_r, c_r, _yaw_err(R_k, 0.5))
        frac = float(agree[cons].mean()) if cons.any() else float("nan")
        fov = e_k < 5.0                    # where they land in k: 2 px there is <= 0.35 deg
        a5, a5c = int((agree & fov).sum()), int((agree_c & fov).sum())
        lost = 1.0 - a5c / a5 if a5 else float("nan")
        surf_ok &= frac >= REPROJECT_MIN
        surf_ctrl_ok &= lost >= 0.5
        # photometric (observation)
        er, ek, rel, _, _ = photometric(ob_r, ob_k, model, R_r, R_k)
        per.append({"k": k, "yaw_deg": yaws[k], "noise_repeat": k == nrep,
                    "T1.2_agree_fraction": frac, "T1.2_considered": int(cons.sum()),
                    "T1.2_control_lost_e_k_below_5deg": lost,
                    "residual_by_e_k": _binned(ek, rel), "residual_by_e_r": _binned(er, rel),
                    "residual_median_all": float(np.nanmedian(rel))})
        log(f"{PREFIX} k={k} yaw {yaws[k]:5.1f}: T1.2 agree {frac:.4f} (control loses {lost:.3f} at e_k < 5 deg), "
            f"residual median {np.nanmedian(rel):.4f}")
    checks.append({"id": "T1.0", "ok": max(pose_err_c) <= 1e-6 and max(pose_err_a) <= 1e-6,
                   "control_fails": min(ctrl_pose) > 1e-6,
                   "numbers": {"max_centre_m": max(pose_err_c), "max_yaw_rad": max(pose_err_a),
                               "control_min_yaw_rad": min(ctrl_pose)}})
    checks.append({"id": "T1.1", "ok": ray_max <= 1e-4, "control_fails": ray_ctrl_min > 1e-4,
                   "numbers": {"max_deg": ray_max, "control_min_median_deg": ray_ctrl_min}})
    checks.append({"id": "T1.2", "oracle": True, "ok": bool(surf_ok), "control_fails": bool(surf_ctrl_ok),
                   "numbers": {"min_agree": min(p["T1.2_agree_fraction"] for p in per),
                               "min_control_lost_e_k_below_5deg": min(p["T1.2_control_lost_e_k_below_5deg"] for p in per)}})
    from . import dolly
    res = {"schema": "foveal-motion-m6-t1-v1", "run": run["_dir"], "run_git": run["git"], "spp": run["spp"],
           "computed_by": {"foveal-motion": dolly._git(dolly.REPO), "fov-render": dolly._git(dolly.FOV_RENDER)},
           "checks": checks, "per_look": per}
    for c in checks:
        log(f"{PREFIX} {c['id']} {'PASS' if c['ok'] else 'FAIL'} (control {'fails' if c['control_fails'] else 'DOES NOT FAIL'}) "
            + json.dumps(c["numbers"], default=float))
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "t1.json").write_text(json.dumps(res, indent=1, default=float))
    return res
