"""The process standard S1 (docs/standard.md; P-D4, Luiz, 2026-11-21, #27) as code: the task profiles as data, the
error gate, and check_record, which every harness from #28 on calls on its record.

- A profile fixes `values`; a record lists the values as run, and every value that differs from the profile's must be
  named in the record's `departures`, with its reason. A profile's `min_controls` must all be in the record's
  `controls`.
- The gate: the model's cells right >= 99.5 %, overall and in each included sector of psi (the angle from the heading,
  taken from the cell's dominant reference to its fused point: observable, no truth). Below 1,000 cells a part is not
  evaluated, never passed; the whole passes only if every part passes, and fails if any part fails.

Self-test with controls: `python -m deep standard-selftest`.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = "S1"
GATE_RIGHT_PERMILLE = 995              # 99.5 % (Luiz, 2026-11-21); counted in integers, no float edge
GATE_RIGHT = GATE_RIGHT_PERMILLE / 1000
GATE_MIN_CELLS = 1000
CONFIRM_SPAN_M = 0.5                   # provisional until #28 measures it
SECTORS = {"0-15": (0, 15), "15-60": (15, 60), "60-120": (60, 120), "120-165": (120, 165), "165-180": (165, 181)}
# stage 9's sectors (e2e.SECTORS): [lo, hi) in degrees; the self-test checks they have not drifted

_REAL = {                              # the values the two real-matching profiles share (docs/standard.md, parts 1-6)
    "scene": "classroom", "spp": 1024, "noise_measured": True, "look_spacing_m": 0.01,
    "sensor": "panorama", "width": 720, "psi_deg": [1.0, 179.0], "lat_max_deg": 70.0,
    "frames": 11, "frame_spacing_m": 0.10, "cell_m": 0.02, "metric": "stage 9",
    "confirm_span_m": CONFIRM_SPAN_M, "gate_right": GATE_RIGHT, "gate_min_cells": GATE_MIN_CELLS,
}
PROFILES = {
    "dry": {"task": "a stage's dry run", "who": "Chat sandbox", "gated": False,
            "values": {"scene": "synthetic room of M4's fixture planes, ray-cast", "gated": False},
            "min_controls": []},
    "fixture": {"task": "a matcher's unit test", "who": "Chat sandbox", "gated": True,
                "values": {"scene": "M4's fixtures (deep/nfixtures.py)", "step_m": 0.10, "frames_each_side": 5,
                           "gates": "M4's fixture gates"},
                "min_controls": ["steps 20 % long", "scene scaled 1.5x"]},
    "oracle": {"task": "geometry, perfect matcher", "who": "Chat; Code at design resolution", "gated": True,
               "values": {"matcher": "perfect", "sigma_px": 0.5, "grid_px": 2, "keep_rel": 0.10, "cell_m": 0.02},
               "min_controls": ["mirrored poses"]},
    "ablation": {"task": "one change, on one primitive", "who": "Code", "gated": True,
                 "values": dict(_REAL), "min_controls": ["mirrored poses"]},
    "build": {"task": "incremental reconstruction", "who": "Code", "gated": True,
              "values": dict(_REAL, path="composed"), "min_controls": ["mirrored poses"]},
}


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def stamp(profile, values, departures=None, controls=()):
    """The fields a record carries (docs/standard.md, 'How it is used', 2), checked."""
    rec = {"standard": VERSION, "profile": profile, "values": dict(values), "departures": dict(departures or {}),
           "controls": list(controls)}
    rec["check_record"] = check_record(rec)
    return rec


def check_record(rec):
    """The problems of a record against S1 (an empty list when it conforms)."""
    out = []
    if rec.get("standard") != VERSION:
        out.append(f"standard is {rec.get('standard')!r}, not {VERSION!r}")
    prof = PROFILES.get(rec.get("profile"))
    if prof is None:
        return out + [f"unknown profile {rec.get('profile')!r} (one of {sorted(PROFILES)})"]
    vals, dep = rec.get("values") or {}, rec.get("departures") or {}
    for k, v in prof["values"].items():
        if k not in vals:
            out.append(f"value {k!r} not recorded")
        elif not _same(vals[k], v) and not (k in dep and str(dep[k]).strip()):
            out.append(f"undeclared departure: {k} = {vals[k]!r}, the profile's is {v!r}")
    for k in dep:
        if k not in prof["values"]:
            out.append(f"departure {k!r} names no value of profile {rec['profile']!r}")
    have = set(rec.get("controls") or [])
    out += [f"control {c!r} missing" for c in prof["min_controls"] if c not in have]
    return out


def sector_of(psi_deg):
    """Stage 9's sector name for each psi (degrees); '' outside them."""
    psi = np.asarray(psi_deg, float)
    out = np.full(psi.shape, "", dtype=object)
    for name, (lo, hi) in SECTORS.items():
        out[(psi >= lo) & (psi < hi)] = name
    return out


def _part(n_right, n):
    if n < GATE_MIN_CELLS:
        res = "not evaluated"
    else:
        res = "pass" if n_right * 1000 >= GATE_RIGHT_PERMILLE * n else "fail"
    return {"cells": int(n), "right": (n_right / n) if n else None, "result": res}


def gate(right, psi_deg, sectors=None):
    """S1's error gate on a model: `right` (bool, one per model cell), `psi_deg` (each cell's angle from the heading,
    from its dominant reference), `sectors` the included sector names (default: all five)."""
    right = np.asarray(right, bool)
    sec = sector_of(psi_deg)
    names = list(SECTORS) if sectors is None else list(sectors)
    parts = {"overall": _part(int(right.sum()), len(right))}
    for s in names:
        m = sec == s
        parts[s] = _part(int(right[m].sum()), int(m.sum()))
    res = [p["result"] for p in parts.values()]
    whole = "fail" if "fail" in res else ("pass" if all(r == "pass" for r in res) else "not evaluated")
    return {"result": whole, "parts": parts, "gate_right": GATE_RIGHT, "min_cells": GATE_MIN_CELLS}


def selftest():
    """(name, ok, value, control, control fails, control value) for each check."""
    from . import e2e
    rng = np.random.default_rng(27)
    out = []
    psi = rng.uniform(0, 180, 12000)
    psi[:2000] = rng.uniform(0, 15, 2000)                 # every sector well above 1,000 cells
    psi[2000:4000] = rng.uniform(165, 180, 2000)
    good = np.ones(len(psi), bool)
    bad = good.copy()
    bad[rng.choice(len(psi), int(0.02 * len(psi)), replace=False)] = False      # 98 % right
    g, gb = gate(good, psi), gate(bad, psi)
    out.append(("G1: the gate passes 12,000 cells all right", g["result"] == "pass", g["result"],
                "98 % right", gb["result"] == "fail", gb["result"]))
    edge = good.copy()
    edge[rng.choice(len(psi), 60, replace=False)] = False                       # exactly 99.5 % (11,940 / 12,000)
    less = edge.copy()
    less[np.flatnonzero(less)[0]] = False                                       # one cell fewer right
    ge, gl = gate(edge, psi, sectors=[]), gate(less, psi, sectors=[])
    out.append(("G2: exactly 99.5 % right passes", ge["result"] == "pass", ge["parts"]["overall"],
                "one cell fewer right", gl["result"] == "fail", gl["parts"]["overall"]))
    few, enough = gate(np.ones(999, bool), rng.uniform(0, 180, 999), []), gate(np.ones(1000, bool), rng.uniform(0, 180, 1000), [])
    out.append(("G3: 999 cells all right are not evaluated, not passed", few["result"] == "not evaluated", few["result"],
                "1,000 cells all right are not 'not evaluated'", enough["result"] != "not evaluated", enough["result"]))
    vals = dict(PROFILES["ablation"]["values"])
    ok_rec = stamp("ablation", vals, controls=["mirrored poses"])
    bad_rec = stamp("ablation", dict(vals, spp=128), controls=["mirrored poses"])
    out.append(("R1: a conforming ablation record has no problems", not ok_rec["check_record"], ok_rec["check_record"],
                "spp 128, undeclared", bool(bad_rec["check_record"]), bad_rec["check_record"]))
    dep_rec = stamp("ablation", dict(vals, spp=128), departures={"spp": "the noise ablation"}, controls=["mirrored poses"])
    blank = stamp("ablation", dict(vals, spp=128), departures={"spp": " "}, controls=["mirrored poses"])
    out.append(("R2: a departure declared with its reason is no problem", not dep_rec["check_record"],
                dep_rec["check_record"], "declared without a reason", bool(blank["check_record"]), blank["check_record"]))
    noctl = stamp("ablation", vals)
    out.append(("R3: the profile's controls are required", not ok_rec["check_record"], ok_rec["controls"],
                "without the mirrored poses", bool(noctl["check_record"]), noctl["check_record"]))
    shifted = dict(e2e.SECTORS, **{"0-15": (0, 16)})
    out.append(("S1: S1's sectors are stage 9's", {k: tuple(v) for k, v in SECTORS.items()}
                == {k: tuple(v) for k, v in e2e.SECTORS.items()}, list(SECTORS),
                "a copy with 0-16", {k: tuple(v) for k, v in SECTORS.items()} != {k: tuple(v) for k, v in shifted.items()},
                "differs"))
    return out
