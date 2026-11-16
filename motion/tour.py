"""D1: a tour, straight legs joined by turns in place, compiled into plain path specs (numpy only).

A tour spec is a JSON file (`paths/tour_*.json`, schema foveal-motion-tour-v1):

    scene            the .blend, relative to the repository root
    legs             [{"name", "start_w", "end_w"}, ...]: straight legs, each starting where the one before ends;
                     the head faces along its leg, gaze `ahead`
    step_m           the frozen estimator's step (M4 was gated at 0.099-0.100 m); each leg is cut into
                     round(length / step_m) equal steps
    phases           P interleaved paths per leg, offset by step / P: phase p's looks sit at
                     start + (i + p / P) step, every one a plain foveal-motion-path-v1 spec with the gated step,
                     so the frozen windows (which count frames) see exactly M4's geometry
    pivot_deg        at most this yaw between two pivot looks; the pivot is a turn in place at the joint,
                     display only (no window may contain it: the alignment gate of #5 refuses one that does)
    seed_base        C seed of look k of leg l, phase p: seed_base + SEED_STRIDE (P l + p) + k; pivot looks
                     seed_base + PIVOT_SEED + j
    preflight        the v1 limits, applied to every phase path (control_head_w included)
    source           where every number comes from

The video's timeline interleaves the phases along each leg (phase 0 k, phase 1 k, ..., phase 0 k+1, ...), then
the pivot, then the next leg: one look per video frame, step / P apart.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from fov_render import HeadPose

from . import paths

SCHEMA = "foveal-motion-tour-v1"
SEED_STRIDE = 200
PIVOT_SEED = 50_000
REQUIRED = ("name", "scene", "legs", "step_m", "phases", "pivot_deg", "seed_base", "preflight", "source")
TARGET_AHEAD_M = 50.0        # the phase paths' target_ray points this far along the leg (the gaze is `ahead`;
                             # v1 needs a target, and the preflight checks the ray's hit is visible from every head)


class TourError(ValueError):
    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


def _v(x):
    return np.asarray(x, dtype=np.float64)


def validate(spec: dict) -> list[str]:
    out = []
    if spec.get("schema") != SCHEMA:
        out.append(f"schema must be {SCHEMA!r}")
    out += [f"missing {k}" for k in REQUIRED if k not in spec]
    if out:
        return out
    legs = spec["legs"]
    if not (isinstance(legs, list) and len(legs) >= 1):
        return out + ["legs: a non-empty list"]
    for i, lg in enumerate(legs):
        for k in ("start_w", "end_w"):
            if not paths._vec3(lg.get(k)):
                out.append(f"legs[{i}].{k}: three finite numbers")
        if out:
            continue
        d = _v(lg["end_w"]) - _v(lg["start_w"])
        if abs(d[2]) > 1e-9:
            out.append(f"legs[{i}]: level (start and end at the same height)")
        if np.linalg.norm(d[:2]) < spec["step_m"]:
            out.append(f"legs[{i}]: shorter than one step")
        if i and not np.allclose(lg["start_w"], legs[i - 1]["end_w"], 0, 1e-9):
            out.append(f"legs[{i}].start_w: must equal legs[{i - 1}].end_w (turns are in place)")
    if not (isinstance(spec["phases"], int) and spec["phases"] >= 1):
        out.append("phases: an integer >= 1")
    if not (isinstance(spec["step_m"], (int, float)) and spec["step_m"] > 0):
        out.append("step_m: positive")
    if not (isinstance(spec["pivot_deg"], (int, float)) and 0 < spec["pivot_deg"] <= 45):
        out.append("pivot_deg: in (0, 45]")
    pf = spec["preflight"]
    out += [f"preflight.{k} missing" for k in paths.PREFLIGHT if k not in pf]
    return out


def load(path) -> dict:
    spec = json.loads(Path(path).read_text())
    bad = validate(spec)
    if bad:
        raise TourError(bad)
    return spec


def _leg_geom(spec, i):
    lg = spec["legs"][i]
    a, b = _v(lg["start_w"]), _v(lg["end_w"])
    L = float(np.linalg.norm(b - a))
    n = max(1, int(round(L / spec["step_m"])))
    return a, b, (b - a) / L, L / n, n


def phase_name(spec, leg, p) -> str:
    return f"{spec['legs'][leg]['name']}{p}"


def phase_spec(spec: dict, leg: int, p: int) -> dict:
    """Phase p of leg `leg` as a foveal-motion-path-v1 spec (start offset p/P of a step, the leg's step)."""
    a, b, f, step, n = _leg_geom(spec, leg)
    P = spec["phases"]
    n_looks = n + 1 if p == 0 else n
    s0 = a + (p / P) * step * f
    s1 = s0 + (n_looks - 1) * step * f
    pf = dict(spec["preflight"])
    return {"schema": paths.SCHEMA, "name": f"{spec['name']}_{phase_name(spec, leg, p)}", "scene": spec["scene"],
            "start_w": s0.tolist(), "end_w": s1.tolist(), "n_looks": n_looks, "facing_w": [float(f[0]), float(f[1]), 0.0],
            "target_ray": {"from_k": 0, "toward_w": (s0 + TARGET_AHEAD_M * f).tolist()},
            "preflight": pf,
            "source": {"tour": spec["name"], "leg": spec["legs"][leg]["name"], "phase": p, "phases": P,
                       "step_m": step, "what": f"phase {p} of {P}: looks at start + (i + {p}/{P}) step along the leg, "
                                               f"i = 0..{n_looks - 1}; written by motion/tour.py"},
            "seed_base": int(spec["seed_base"] + SEED_STRIDE * (P * leg + p))}


def _yaw(f):
    return math.atan2(f[1], f[0])


def pivot_heads(spec: dict, leg: int) -> list[HeadPose]:
    """The turn in place between leg `leg` and leg `leg + 1`, without its two ends (those are the legs' looks)."""
    fa = _leg_geom(spec, leg)[2]
    fb = _leg_geom(spec, leg + 1)[2]
    ya, yb = _yaw(fa), _yaw(fb)
    d = (yb - ya + math.pi) % (2 * math.pi) - math.pi
    n = max(1, int(math.ceil(abs(math.degrees(d)) / spec["pivot_deg"])))
    pos = _v(spec["legs"][leg]["end_w"])
    out = []
    for j in range(1, n):
        y = ya + d * j / n
        out.append(HeadPose(paths.head_axes([math.cos(y), math.sin(y), 0.0]), pos))
    return out


def timeline(spec: dict) -> list[dict]:
    """One entry per video frame: {"kind": "leg", "leg", "phase", "k", "s_m"} or {"kind": "pivot", "leg", "j"}."""
    P = spec["phases"]
    out = []
    for leg in range(len(spec["legs"])):
        _, _, _, step, n = _leg_geom(spec, leg)
        for i in range(n + 1):
            for p in range(P):
                if p and i == n:
                    break
                out.append({"kind": "leg", "leg": leg, "phase": p, "k": i, "s_m": (i + p / P) * step})
        if leg + 1 < len(spec["legs"]):
            for j in range(len(pivot_heads(spec, leg))):
                out.append({"kind": "pivot", "leg": leg, "j": j})
    return out


def frame_of(tl: list[dict]) -> dict:
    """(leg, phase, k) -> video frame index."""
    return {(e["leg"], e["phase"], e["k"]): f for f, e in enumerate(tl) if e["kind"] == "leg"}


def selftest() -> list[tuple[str, bool]]:
    spec = {"schema": SCHEMA, "name": "t", "scene": "x.blend", "step_m": 0.1, "phases": 3, "pivot_deg": 4.0,
            "seed_base": 7000, "source": "selftest",
            "legs": [{"name": "A", "start_w": [0, 0, 1.5], "end_w": [0, 2.0, 1.5]},
                     {"name": "B", "start_w": [0, 2.0, 1.5], "end_w": [-1.5, 2.0, 1.5]}],
            "preflight": {"min_clearance_m": 0.25, "max_hole_fraction": 0.01, "max_backface_fraction_other": 0.001,
                          "backface_allowed": [], "control_head_w": [0, 9, 1.5]}}
    ph = [phase_spec(spec, 0, p) for p in range(3)]
    pos = np.concatenate([paths.positions(s) for s in ph])
    tl = timeline(spec)
    legA = [e for e in tl if e["kind"] == "leg" and e["leg"] == 0]
    s = np.array([e["s_m"] for e in legA])
    pv = pivot_heads(spec, 0)
    fw = [h.dir_to_world([0.0, 0.0, -1.0]) for h in pv]
    res = [
        ("accepted tour", validate(spec) == []),
        ("phase specs are valid v1 paths", all(paths.validate(p) == [] for p in ph)),
        ("every phase keeps the leg's step", all(abs(paths.step_m(p) - 0.1) < 1e-12 for p in ph)),
        ("phases interleave at step / 3", bool(np.allclose(np.diff(np.sort(pos[:, 1])), 0.1 / 3))),
        ("phase looks: n + 1, n, n", [p["n_looks"] for p in ph] == [21, 20, 20]),
        ("timeline along leg A is monotone, step / 3", bool(np.allclose(np.diff(s), 0.1 / 3))),
        ("timeline holds every look once", len(legA) == sum(p["n_looks"] for p in ph)),
        ("pivot: 90 deg at <= 4 deg -> 22 inner looks", len(pv) == 22),
        ("pivot stays at the joint", all(np.allclose(h.origin_w, [0, 2.0, 1.5]) for h in pv)),
        ("pivot turns left, monotone", bool(np.all(np.diff([math.atan2(f[1], f[0]) for f in fw]) > 0))),
        ("seeds differ across phases", len({p["seed_base"] for p in ph}) == 3),
        ("refused: legs not joined", bool(validate(dict(spec, legs=[spec["legs"][0],
                                                                     dict(spec["legs"][1], start_w=[0, 2.1, 1.5])])))),
        ("refused: a climbing leg", bool(validate(dict(spec, legs=[dict(spec["legs"][0], end_w=[0, 2.0, 1.6])])))),
        ("refused: pivot_deg 0", bool(validate(dict(spec, pivot_deg=0)))),
    ]
    return res
