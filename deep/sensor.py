"""Stage 4 (#4): the sensor, a heading-locked slit fan, carried by fov-render's Panorama.

Plan: docs/plan.md, "The sensor"; stage 3's atlas (deep/atlas.py) sets the time schedule.

The carrier. fov-render's Panorama is equirectangular and head-locked: pixel (row, col) has local direction
d = (sin lon cos lat, sin lat, -cos lon cos lat), lon = (col + 0.5)/W 2 pi - pi, lat = pi/2 - (row + 0.5)/H pi. With the
head locked to the path's tangent, a column is a vertical slit at bearing b = -lon from the heading (+ left) and a row an
elevation lat. That is the arc's concentric-mosaic (beta, v) = (b, tan lat) exactly, and the line's pencil (omega, u) is
computed from the same ray.

The schedule. A slit at angle psi from the direction of travel needs steps short enough to keep the unpredicted parallax
within one resolution element; r_lo, r_hi = the range still uncertain. Stage 3's ds_max(psi, r_lo, r_hi) is the
first-order answer; SlitFan.step solves the straight step exactly. Rendering dense looks (every ds) and keeping each slit's samples every floor(ds_max / ds) looks emulates a
sensor that samples each slit at its own rate.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import atlas as at
from . import primitives as pr

DESIGN_RES_DEG = 0.1          # stage 3's design resolution (width 3600): Code, on the GPU
SANDBOX_WIDTH = 720           # 0.5 deg per pixel: Chat sandbox


@dataclass
class SlitFan:
    width: int = SANDBOX_WIDTH                # the carrier's width (360 deg)
    r_band: tuple = (0.5, 20.0)               # the range still uncertain (m)

    @property
    def res(self) -> float:
        return 2 * np.pi / self.width

    def model(self):
        from fov_render.sensors import PanoramaModel
        return PanoramaModel(self.width)

    def bearing_lat(self):
        """(bearing from the heading, + left; elevation) of every pixel, (H, W)."""
        lon, lat = self.model().lonlat()
        return -lon, lat

    def step(self, psi, r_band=None):
        """The largest time step for a slit at angle psi from the direction of travel: the exact bearing changes of the
        band's two ends over a straight step differ by one pixel. Stage 3's ds_max is its first-order limit; over a
        finite step a near point gets closer and its parallax grows, which the first-order step underestimates by up to
        5 % (S4.3's first run; docs/plan.md, stage 4)."""
        lo, hi = r_band or self.r_band
        psi = np.atleast_1d(np.asarray(psi, float))
        out = np.empty_like(psi)
        for i, p in enumerate(psi):
            sp, cp = abs(np.sin(p)), np.cos(p)
            def spread(dlt):
                ch = lambda r: np.arctan2(r * sp, r * cp - dlt) - np.arctan2(r * sp, r * cp)
                return abs(ch(lo) - ch(hi))
            a, b = 0.0, 2.0 * float(at.ds_max(p, lo, hi, sigma_px=self.res))
            while spread(b) < self.res:
                b *= 2
            for _ in range(80):
                m = 0.5 * (a + b)
                a, b = (m, b) if spread(m) < self.res else (a, m)
            out[i] = a
        return out if out.size > 1 else float(out[0])

    def keep_every(self, psi, ds_render, r_band=None):
        """How many rendered looks a slit may skip: floor(step / ds_render), at least 1."""
        return np.maximum(1, np.floor(self.step(psi, r_band) / ds_render + 1e-9)).astype(int)


# ----------------------------------------------------------------------------- render (sandbox: calibration room)

def arc_run_poses(R=1.6, s_list=(0.0, 0.01, 0.10), p0=(0.0, 0.0), phi0=np.pi / 2, h=1.6):
    """Head poses along a left-turning arc from the calibration room's EYE, heading +Y."""
    pc = pr.Piece("arc", max(s_list) + 1e-9, 1.0 / R)
    return [(s, *pc.pose(p0, phi0, h, s)[:2]) for s in s_list]


def render(out, fan: SlitFan, poses, spp=16, seed_base=9000, device="CPU", launcher="auto", git=None, blend=None,
           meta=None, drop_exr=False) -> int:
    from fov_render import HeadPose, RenderConfig, Rig, Session, sensors
    from fov_render.checks.step1 import calib_room
    from motion import paths
    rd = Path(out)
    if (rd / "run.json").exists():
        print(f"[deep] refused: {rd} already holds a run", file=sys.stderr)
        return 2
    blend = blend or calib_room({"work": str(Path(__file__).resolve().parents[1] / "runs" / "calib-scene"), "device": "CPU"})
    rd.mkdir(parents=True, exist_ok=True)
    g = paths.gaze({"target_w": [0, 0, 0]}, "ahead")
    rec = {"schema": "deep-motion-run-v1", "experiment": "stage 4: the slit fan on the Panorama carrier (#4)",
           "carrier": {"sensor": "panorama", "width": fan.width}, "spp": spp, "device_requested": device,
           "seeds": f"seed_base {seed_base} + k", "blend": str(blend), "git": git, "command": sys.argv, "looks": []}
    rec.update(meta or {})
    rec["exr_dropped"] = bool(drop_exr)
    t0 = time.perf_counter()
    with Session(Path(blend), RenderConfig(profile="full", device=device), Rig(), access="oracle", launcher=launcher,
                 workdir=rd / "session", prepare=sensors.Panorama(width=fan.width)) as sess:
        for k, (s, c, R) in enumerate(poses):
            o = rd / f"k{k:04d}"
            t1 = time.perf_counter()
            sess.look(sensors.Panorama(width=fan.width), ("C",), g, head=HeadPose(R, c), spp=spp,
                      seeds={"C": seed_base + k}, out_dir=o, outputs=("exr", "records"))
            if drop_exr:
                # fov-render always writes the EXR (the record is built from it); deep's oracle reads only the record,
                # so the EXR may go once the record exists (#6: 207 MB per look at 3600 wide). Disk, not time.
                if not (o / "record_C.npz").exists():
                    raise RuntimeError(f"no record in {o}; the EXR is kept")
                (o / "raw_C.exr").unlink()
            rec["looks"].append({"k": k, "s": float(s), "c": c.tolist(), "R": R.tolist(), "out_dir": o.name,
                                 "wall_seconds": round(time.perf_counter() - t1, 2)})
            print(f"[deep] stage 4 look {k} s = {s:.3f} m: {rec['looks'][-1]['wall_seconds']} s", flush=True)
        rec["device"] = sess.ready.get("device")
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    (rd / "run.json").write_text(json.dumps(rec, indent=1))
    return 0


# ----------------------------------------------------------------------------- checks (#4)

def _ang_deg(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


def check_mapping(fan: SlitFan, phi=np.radians(37.0)):
    """S4.1: the carrier's rays give the designed slit coordinates (numpy). Returns (max error, control error)."""
    m = fan.model()
    rays = m.rays().direction.reshape(-1, 3)
    b, lat = (x.reshape(-1) for x in fan.bearing_lat())
    R = pr.head_R(phi)
    w = rays @ R.T
    beta, v = pr.arc_coords(w, phi)
    ok = np.abs(lat) < np.radians(89.0)
    e = max(float(np.max(np.abs(np.angle(np.exp(1j * (beta - b)))[ok]))), float(np.max(np.abs(v - np.tan(lat))[ok] / np.maximum(1, np.abs(np.tan(lat[ok]))))))
    c = float(np.median(np.abs(np.angle(np.exp(1j * (beta + b)))[ok & (np.abs(b) > 0.1)])))
    return e, c


def check_rendered(run_dir, fan: SlitFan):
    """S4.2: every rendered ray (the record's world direction) is the designed ray R_head d_local (deg); control: the
    designed head yawed 0.5 deg. Also the rendered rays' arc coordinates against (b, tan lat)."""
    from fov_render.record import SampleRecord
    run = json.loads((Path(run_dir) / "run.json").read_text())
    m = fan.model()
    loc = m.rays().direction.reshape(-1, 3)
    b, lat = (x.reshape(-1) for x in fan.bearing_lat())
    worst, ctl, worst_b = 0.0, np.inf, 0.0
    for lk in run["looks"]:
        rec = SampleRecord.load(Path(run_dir) / lk["out_dir"] / "record_C.npz")
        R = np.asarray(lk["R"])
        d = rec.direction.astype(np.float64)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        des = loc[rec.raster_index] @ R.T
        worst = max(worst, float(np.max(_ang_deg(d, des))))
        a = np.radians(0.5)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        ctl = min(ctl, float(np.median(_ang_deg(d, des @ Rz.T)[np.abs(lat[rec.raster_index]) < np.radians(60)])))
        phi = float(np.arctan2(-R[1, 2], -R[0, 2]))
        beta, _ = pr.arc_coords(d, phi)
        ok = np.abs(lat[rec.raster_index]) < np.radians(89.0)
        worst_b = max(worst_b, float(np.degrees(np.max(np.abs(np.angle(np.exp(1j * (beta - b[rec.raster_index])))[ok])))))
    return worst, ctl, worst_b


def check_schedule(fan: SlitFan, R=1.6, n=400, seed=4):
    """S4.3: decimating each slit at its scheduled step keeps the unpredicted parallax within one pixel, for points across
    the band, on a line and an arc (exact trajectories); control: twice the step. Returns (max / res, control min / res)."""
    rng = np.random.default_rng(seed)
    worst, ctl = 0.0, np.inf
    for kind, Rr in (("line", None), ("arc", R)):
        for psi_deg in (10.0, 30.0, 60.0, 90.0, 120.0, -45.0, -90.0):
            psi = np.radians(psi_deg)
            step = float(fan.step(psi))
            for mult, store in ((1.0, "w"), (2.0, "c")):
                L = 2 * mult * step
                s, poses = at.window_poses(kind, Rr, L=L, ds=mult * step)
                i0 = int(np.argmin(np.abs(s)))
                lo, hi = fan.r_band
                u = np.array([np.cos(psi), np.sin(psi)])
                P = lambda r: np.array([[r * u[0], r * u[1], at.H]])
                d = lambda r: np.angle(np.exp(1j * (at.bearings(P(r), poses)[i0 + 1, 0] - at.bearings(P(r), poses)[i0, 0])))
                unpred = abs(d(lo) - d(hi)) / fan.res
                if store == "w":
                    worst = max(worst, unpred)
                else:
                    ctl = min(ctl, unpred)
    return worst, ctl


def selftest(run_dir=None, fan=None):
    fan = fan or SlitFan()
    res, num = [], {}
    e, c = check_mapping(fan)
    num["S4.1_max"], num["S4.1_control_median_rad"] = e, c
    res.append(("S4.1 the carrier's pixel rays have the designed slit coordinates (bearing = -lon, v = tan lat) (<= 1e-12)", e <= 1e-12))
    res.append(("S4.1 control: the bearing's sign flipped (median >= 0.1 rad)", c >= 0.1))
    w, wc = check_schedule(fan)
    num["S4.3_max_px"], num["S4.3_control_min_px"] = w, wc
    res.append(("S4.3 each slit at its scheduled step: unpredicted parallax within one pixel (<= 1.05 px)", w <= 1.05))
    res.append(("S4.3 control: twice the step exceeds it (>= 1.8 px)", wc >= 1.8))
    if run_dir:
        worst, ctl, wb = check_rendered(run_dir, fan)
        num["S4.2_max_deg"], num["S4.2_control_median_deg"], num["S4.2_bearing_max_deg"] = worst, ctl, wb
        res.append(("S4.2 every rendered ray is the designed ray; its bearing is its column's (<= 1e-4 deg)", worst <= 1e-4 and wb <= 1e-4))
        res.append(("S4.2 control: the designed head yawed 0.5 deg (median >= 0.25 deg)", ctl >= 0.25))
    return res, num
