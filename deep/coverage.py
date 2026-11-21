"""Stage 13a (#31): coverage first. How far coverage can go within a usable reconstruction.

Luiz, 2026-11-21, after demo D3 (#29): S2's model covered 3.8 % of the surface, every cell N reported 17.1 % at
97.77 % right. He asks for the opposite test, coverage first, "to have a feeling of how much we can push coverage within
a usable reconstruction, perhaps even beyond these 17 %".

Everything comes from stage 11's caches on R3's straight path (12 references): no new matching. Each map is an anchor
source in a ray set, fused by stage 9's fusion and judged by its metric (2 cm cells; right within 10 % in a 3 x 3
neighbourhood; coverage of the whole surface within 2 cm of a right cell), then thinned by a filter that uses no truth.

- **Sources** (N on #26's references, stage 11's and 11c's caches):
  - N1024, M4's 0.5 px depth-edge test (demo D3's "every cell");
  - N at 10 % of zeta, and N with the depth-edge test off (stage 11c's variants, from its every-anchor run);
  - N1024 joined with N on +/-0.2 m windows (stage 11's row; shorter windows for near lateral points, #26);
  - N at 10 % joined with N on +/-0.2 m windows;
  - N with the depth-edge test off and the anchors whose first rejection was neighbour support readmitted. Only each
    anchor's first failed test is cached, so this readmits those that failed support first, whatever later tests
    they would have failed: an approximation, labelled so.
- **Ray sets:** psi 1-120 deg (demo D3's) and 1-179 deg (N's own).
- **Filters** (no truth):
  - none;
  - not isolated: at least 1, or at least 3, other fused cells among the 26 cells around it;
  - supported within 2 cm by a second reference (stage 12b's rule, span 0).

Each map reports cells, cells right, wrong cells, gross errors (relative range error over 25 %), coverage, its share of
the perfect matcher's, and S2's gate as a report. The frontier: the largest coverage at cells right >= 98, 97 and 95 %.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from . import assess as asx
from . import confirm as cf
from . import demo2 as d2
from . import e2e
from . import ntrace as nt
from . import standard as sd
from . import straight as st

RAYS = {"1-120": (1.0, 120.0), "1-179": (1.0, 179.0)}
FILTERS = ("none", "neighbours >= 1", "neighbours >= 3", "within 2 cm of a second reference")
FLOORS = (0.98, 0.97, 0.95)
GROSS = 0.25
SHORT = "N1024 +/-0.2 m"
S_N = "N1024 (0.5 px)"
S_10 = "N at 10 %"
S_OFF = "N, depth-edge test off"
S_NS = "N1024 + N on +/-0.2 m windows"
S_10S = "N at 10 % + N on +/-0.2 m windows"
S_LOOSE = "N, edge test off, unsupported readmitted (approx.)"
SOURCES = (S_N, S_10, S_OFF, S_NS, S_10S, S_LOOSE)


def readmitted(o, truth, j, looks, reasons=("unsupported",)):
    """N's every-anchor run at reference j, accepting the anchors that passed every test before the depth edge or whose
    first failed test is in `reasons`; the depth-edge test off."""
    pre = o["reason_pre_edge"]
    acc = (pre == 0)
    for r in reasons:
        acc |= pre == nt.REASONS.index(r)
    acc &= o["zeta_all"] > 0
    c = looks[j][1]
    rng = 1.0 / o["zeta_all"][acc]
    return {"j": j, "X": c + o["u_all"][acc] * rng[:, None], "psi": np.degrees(o["psi_all"][acc]),
            "tried": int(len(pre)), "accepted": int(acc.sum()), "right_acc": None}


def union(a, b):
    return {j: {"j": j, "X": np.concatenate([a[j]["X"], b[j]["X"]]), "psi": np.concatenate([a[j]["psi"], b[j]["psi"]]),
                "tried": a[j]["tried"] + b[j]["tried"], "accepted": a[j]["accepted"] + b[j]["accepted"],
                "right_acc": None} for j in a}


def neighbours(Xf):
    """For each fused cell, the number of other fused cells among the 26 around it (truth-free)."""
    if len(Xf) == 0:
        return np.zeros(0, int)
    k = e2e._keys(Xf)
    code = e2e._encode(k)
    cs = np.sort(code)
    cnt = np.zeros(len(Xf), int)
    for d in itertools.product((-1, 0, 1), repeat=3):
        if d == (0, 0, 0):
            continue
        nc = e2e._encode(k + np.array(d))
        pos = np.clip(np.searchsorted(cs, nc), 0, len(cs) - 1)
        cnt += cs[pos] == nc
    return cnt


def maps(row, per_ref, truth, fan, surf, ray, looks, refs, pcov, sectors):
    """One fusion of a source in a ray set, and its maps under FILTERS."""
    Xs, refs_of = [], []
    for j, o in row.items():
        sel = (o["psi"] >= ray[0]) & (o["psi"] <= ray[1])
        Xs.append(o["X"][sel])
        refs_of.append(np.full(int(sel.sum()), j))
    X, ref, Xf, nref, dom, right, cell_of = asx.fuse_and_judge(Xs, refs_of, per_ref, truth, fan)
    err = d2.cells_error(Xf, dom, per_ref, truth, fan)
    psi = cf.cell_psi(Xf, dom, looks)
    nb = neighbours(Xf)
    sup = cf.support(X, ref, Xf, cell_of, sorted(refs), "within 2 cm").sum(1) >= 2
    masks = {"none": np.ones(len(Xf), bool), "neighbours >= 1": nb >= 1, "neighbours >= 3": nb >= 3,
             "within 2 cm of a second reference": sup}
    out = {}
    for name, m in masks.items():
        cov = asx.covered(surf[1], Xf[right & m]) if m.any() else np.zeros(len(surf[1]), bool)
        n, nr = int(m.sum()), int((right & m).sum())
        if name == "none":
            cov_none = cov
        out[name] = {"cells": n, "right": nr / n if n else None, "wrong": n - nr,
                     "gross": int((m & ~(err <= GROSS)).sum()), "coverage": float(cov.mean()),
                     "covers_of_perfect": float((cov & pcov).sum() / pcov.sum()) if pcov is not None and pcov.any() else None,
                     "gate": sd.gate(right[m], psi[m], sectors)}
    return out, masks, cov_none


def run(paths, cache=None, rec_demo=None, log=print, dev=None):
    """Stage 13a. paths and cache as stage 11's; rec_demo: demo D3's record (#29, or its dry run) for K1."""
    t0 = time.perf_counter()
    looks128, truth, fan, refs, vis_fn = paths["128"]
    R, per_ref = st.collect(paths, None, cache, rows=["perfect", "N1024", cf.MIRROR, SHORT], log=log, dev=dev,
                            extra=(st.N_EDGE,))
    edges = R.pop(st.N_EDGE)
    src = {S_N: R["N1024"],
           S_10: {j: st.edge_variant(o, truth, j, looks128, "rel") for j, o in edges.items()},
           S_OFF: {j: st.edge_variant(o, truth, j, looks128, "off") for j, o in edges.items()},
           S_LOOSE: {j: readmitted(o, truth, j, looks128) for j, o in edges.items()}}
    if SHORT in R:
        src[S_NS] = union(R["N1024"], R[SHORT])
        src[S_10S] = union(src[S_10], R[SHORT])
    attr = st.attributes(paths, refs, dev)
    ceil = st.ceiling_rows(R, attr, looks128, fan)[cf.CEIL]
    surf = st.surface_band(per_ref, truth, looks128, None)
    log(f"[stage13a] sources ready ({time.perf_counter() - t0:.0f} s)")
    out = {"references": refs, "rays": RAYS, "filters": list(FILTERS), "floors": list(FLOORS), "gross_rel": GROSS,
           "sources": {}, "ceilings": {}, "controls": {}}
    nested_ok, converse = True, None
    for rname, ray in RAYS.items():
        sectors = [k for k, (lo, hi) in sd.SECTORS.items() if lo < ray[1]]
        pm, _, pcov = maps(R["perfect"], per_ref, truth, fan, surf, ray, looks128, refs, None, sectors)
        out["ceilings"][f"perfect, rays {rname}"] = pm["none"]
        cm, _, _ = maps(ceil, per_ref, truth, fan, surf, ray, looks128, refs, pcov, sectors)
        out["ceilings"][f"perfect where gradient SNR >= 3, rays {rname}"] = cm["none"]
        for sname, row in src.items():
            res, masks, _ = maps(row, per_ref, truth, fan, surf, ray, looks128, refs, pcov, sectors)
            out["sources"].setdefault(sname, {})[rname] = res
            nested_ok &= bool(np.all(masks["none"] | ~masks["neighbours >= 1"]) and
                              np.all(masks["neighbours >= 1"] | ~masks["neighbours >= 3"]) and
                              np.all(masks["none"] | ~masks["within 2 cm of a second reference"]))
            if converse is None:
                converse = bool(np.all(masks["neighbours >= 3"] | ~masks["none"]))
            log(f"[stage13a] {sname}, rays {rname} ({time.perf_counter() - t0:.0f} s)")
        if cf.MIRROR in R:
            mres, _, _ = maps(R[cf.MIRROR], per_ref, truth, fan, surf, ray, looks128, refs, pcov, sectors)
            out["controls"][f"N1024 mirrored, rays {rname}"] = mres
    out["frontier"] = frontier(out)
    out["checks"] = checks(out, rec_demo, nested_ok, converse)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def frontier(out):
    res = {}
    rows = [(s, r, f, m) for s, rr in out["sources"].items() for r, ff in rr.items() for f, m in ff.items()]
    for floor in FLOORS:
        ok = [x for x in rows if x[3]["right"] is not None and x[3]["cells"] >= sd.GATE_MIN_CELLS and x[3]["right"] >= floor]
        if ok:
            s, r, f, m = max(ok, key=lambda x: x[3]["coverage"])
            res[f"{floor:g}"] = {"source": s, "rays": r, "filter": f, "cells": m["cells"], "right": m["right"],
                                 "coverage": m["coverage"], "gross": m["gross"]}
        else:
            res[f"{floor:g}"] = None
    return res


def checks(out, rec_demo, nested_ok, converse):
    res = []
    a = out["sources"][S_N]["1-120"]["none"]
    if rec_demo is not None:
        e = rec_demo["end"]["all"]
        same = a["cells"] == e["cells"] and abs(a["right"] - e["right"]) <= 1e-12 and abs(a["coverage"] - e["coverage"]) <= 1e-12
        b = out["sources"][S_N]["1-179"]["none"]
        res.append({"check": "K1: N1024 in rays 1-120 with no filter equals demo D3's every cell (cells, right, coverage)",
                    "ok": bool(same), "value": [a["cells"], a["right"], a["coverage"]],
                    "demo": [e["cells"], e["right"], e["coverage"]], "control": "the same in rays 1-179",
                    "control_fails": b["cells"] != e["cells"], "control_value": b["cells"]})
    else:
        res.append({"check": "K1", "ok": None, "note": "no demo D3 record given"})
    res.append({"check": "K2: every filter only removes cells (neighbours >= 3 within >= 1 within none; 2 cm within none)",
                "ok": bool(nested_ok), "control": "the converse (none within neighbours >= 3), first source and ray set",
                "control_fails": converse is False, "control_value": converse})
    bad = {k: {f: (m["cells"], m["right"]) for f, m in v.items() if m["cells"] >= sd.GATE_MIN_CELLS and (m["right"] or 0) >= min(FLOORS)}
           for k, v in out["controls"].items()}
    res.append({"check": "the standard control: N1024 mirrored must not give a usable map (>= 1,000 cells at >= 95 % right)",
                "ok": None, "control": "N1024 mirrored, every ray set and filter",
                "control_fails": all(not v for v in bad.values()), "control_value": bad})
    return res


def table(out):
    f = lambda x, p=2: "—" if x is None else f"{100 * x:.{p}f} %"
    lines = ["| source | rays | filter | cells | right | wrong | gross (> 25 %) | coverage | of perfect | S2 gate |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(s, r, fl, m) for s, rr in out["sources"].items() for r, ff in rr.items() for fl, m in ff.items()]
    for s, r, fl, m in sorted(rows, key=lambda x: -x[3]["coverage"]):
        lines.append(f"| {s} | {r} | {fl} | {m['cells']:,} | {f(m['right'])} | {m['wrong']:,} | {m['gross']:,} | "
                     f"{f(m['coverage'], 1)} | {f(m['covers_of_perfect'], 1)} | {m['gate']['result']} |")
    lines += ["", "| ceiling | cells | right | coverage |", "|---|---|---|---|"]
    for k, m in out["ceilings"].items():
        lines.append(f"| {k} | {m['cells']:,} | {f(m['right'])} | {f(m['coverage'], 1)} |")
    lines += ["", "| floor (cells right) | the largest coverage | source, rays, filter | cells | right | gross |",
              "|---|---|---|---|---|---|"]
    for fl, b in out["frontier"].items():
        if b is None:
            lines.append(f"| ≥ {100 * float(fl):g} % | none | | | | |")
        else:
            lines.append(f"| ≥ {100 * float(fl):g} % | {f(b['coverage'], 1)} | {b['source']}, {b['rays']}, {b['filter']} | "
                         f"{b['cells']:,} | {f(b['right'])} | {b['gross']:,} |")
    return lines


def selftest():
    """The neighbour count on known cells. (name, ok, value, control, control fails, value)."""
    c = 0.011
    plane = np.array([[c + 0.02 * i, c + 0.02 * k, c] for i in range(3) for k in range(3)])     # a 3 x 3 patch of cells
    lone = np.array([[c + 1.0, c, c]])
    Xf = np.concatenate([plane, lone])
    nb = neighbours(Xf)
    want = [3, 5, 3, 5, 8, 5, 3, 5, 3, 0]
    near = np.concatenate([plane, np.array([[c + 0.06, c, c]])])                                 # one cell beside the patch
    nb2 = neighbours(near)
    return [("N1: neighbours on a 3 x 3 patch and an isolated cell", nb.tolist() == want, nb.tolist(),
             "the same cell 2 cm from the patch's edge is not isolated", nb2[-1] != 0, int(nb2[-1]))]
