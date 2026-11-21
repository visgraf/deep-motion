"""Stage 12 (#28): the confirmation distance, the first run under S1 (docs/standard.md; P-D4).

S1's model is the 2 cm cells confirmed by references at least CONFIRM_SPAN_M apart along the path; the 0.5 m is
provisional until this stage measures it. On R3's straight_long at #26's 12 references (m = 50, 60, ..., 160), from
stage 11's and 11c's per-reference caches, each row is fused once by stage 9's fusion and metric, unchanged
(assess.fuse_and_judge, assess.covered), and its cells are split by the span of the references that support them:
- the span of a cell: the largest distance along the path between two of its references (their arc-length positions,
  summed chord by chord over the references in order; on a line, the distance between them);
- confirmed at span s: at least two references and a span >= s (1 mm of slack); span 0 is stage 9's "cells from two or
  more references";
- the model at s: the confirmed cells; the candidates: the others. Coverage is the model's right cells' (stage 8's rule).
Each model is gated by S1 (deep/standard.py gate): the sector of a cell is its psi from its dominant reference to its
fused point (observable).

Rows: the perfect matcher; the perfect matcher where the gradient SNR >= 3 (the noise-limited ceiling, coverage);
N1024 (M4's 0.5 px depth-edge test) and N1024 at 10 % of zeta, both from stage 11c's run keeping every anchor's state
(the 0.5 px case rebuilt there anchor for anchor); M4@1024 in its field. Controls: N1024 given the mirrored path's
poses (stage 11's C3 control); the perfect matcher with every point moved 20 % farther along its ray.

Checks, declared on #28 before data:
- K1 (reproduction): at span 0, every row's cells, cells right and cells-from-two-references right equal #26's records
  (stage11c.json for N1024 and N at 10 %, stage11.json for the perfect matcher and M4) to 1e-12. Control: N1024's
  cells right over all its cells, in place of the model's, must differ from #26's confirmed value.
- K2 (the gate can pass and fail on this data): the perfect matcher's model at 0.5 m passes S1's gate on the full
  field. Control: the perfect matcher moved 20 % must fail it.
- The standard control: N1024 mirrored must not pass the gate at any span.
"""
from __future__ import annotations

import time

import numpy as np

from . import assess as asx
from . import ntrace as nt
from . import standard as sd
from . import straight as st

SPANS_M = (0.0, 0.2, 0.5)               # accepted on #27 (Luiz, 2026-11-21)
SLACK_M = 1e-3
MOVE = 1.20                             # the control: every point 20 % farther along its ray
N10 = "N1024, depth-edge test at 10 %"
MIRROR = "N1024 mirrored (C3 control)"
MOVED = "perfect, points moved 20 % (control)"
CEIL = "perfect where gradient SNR >= 3"
BAND_ROWS = {"full": ("perfect", CEIL, "N1024", N10, MIRROR, MOVED),
             "shared 1-44": ("perfect", CEIL, "N1024", N10, "M4@1024")}
BAND_SECTORS = {"full": list(sd.SECTORS), "shared 1-44": ["0-15", "15-60"]}


def _surface_snr(snr, idx):
    """The gradient SNR of a reference's pixels that see a surface, within N's latitude (|lat| < lat_max_deg)."""
    H = idx.shape[0]
    lat = np.degrees((0.5 - (np.arange(H) + 0.5) / H) * np.pi)
    m = (idx >= 0) & (np.abs(lat) < nt.SETTINGS["lat_max_deg"])[:, None] & np.isfinite(snr)
    return snr[m]


def positions(looks, refs):
    """Arc-length position of each reference along the path (m), chord by chord over the references in order."""
    refs = sorted(refs)
    pos, s = {refs[0]: 0.0}, 0.0
    for a, b in zip(refs[:-1], refs[1:]):
        s += float(np.linalg.norm(np.asarray(looks[b][1], float) - np.asarray(looks[a][1], float)))
        pos[b] = s
    return pos


def spans(ref, cell_of, ncell, pos):
    """Each cell's span (m): the largest distance along the path between two of its references."""
    p = np.array([pos[int(j)] for j in ref]) if len(ref) else np.zeros(0)
    lo, hi = np.full(ncell, np.inf), np.full(ncell, -np.inf)
    np.minimum.at(lo, cell_of, p)
    np.maximum.at(hi, cell_of, p)
    return hi - lo


def confirmed(nref, span, s):
    return (nref >= 2) & (span >= s - SLACK_M)


def cell_psi(Xf, dom, looks):
    psi = np.full(len(Xf), np.nan)
    for j in np.unique(dom):
        m = dom == j
        psi[m] = st._psi_of(np.asarray(looks[j][1], float), np.asarray(looks[j][2], float), Xf[m])
    return psi


def models(row, per_ref, truth, fan, surf, band, looks, pos, sectors, perfect_cov=None):
    """One fusion of a row in a band (stage 11's inputs, as st.score builds them) and its models at SPANS_M."""
    Xs, refs_of = [], []
    for j, o in row.items():
        sel = np.ones(len(o["psi"]), bool) if band is None else (o["psi"] >= band[0]) & (o["psi"] <= band[1])
        Xs.append(o["X"][sel])
        refs_of.append(np.full(int(sel.sum()), j))
    if sum(len(x) for x in Xs) == 0:
        X = ref = np.zeros(0)
        Xf, nref, dom, right, cell_of = np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool), np.zeros(0, int)
    else:
        X, ref, Xf, nref, dom, right, cell_of = asx.fuse_and_judge(Xs, refs_of, per_ref, truth, fan)
    sp = spans(ref, cell_of, len(Xf), pos)
    psi = cell_psi(Xf, dom, looks)
    cov_all = asx.covered(surf[1], Xf[right]) if len(Xf) else np.zeros(len(surf[1]), bool)
    two = nref >= 2
    out = {"all": {"cells": int(len(Xf)), "right": float(right.mean()) if len(Xf) else None,
                   "confirmed_right": float(right[two].mean()) if two.any() else None,
                   "coverage": float(cov_all.mean()), "surface_cells": int(len(surf[1])), "points": int(len(X))},
           "spans": {}}
    for s in SPANS_M:
        c = confirmed(nref, sp, s)
        cov = asx.covered(surf[1], Xf[right & c]) if c.any() else np.zeros(len(surf[1]), bool)
        e = {"model": {"cells": int(c.sum()), "right": float(right[c].mean()) if c.any() else None,
                       "coverage": float(cov.mean())},
             "candidates": {"cells": int((~c).sum()), "right": float(right[~c].mean()) if (~c).any() else None},
             "gate": sd.gate(right[c], psi[c], sectors)}
        if perfect_cov is not None and perfect_cov.any():
            e["model"]["covers_of_perfect"] = float((cov & perfect_cov).sum() / perfect_cov.sum())
        out["spans"][f"{s:g}"] = e
    return out, cov_all


def moved(row, looks):
    return {j: dict(o, X=np.asarray(looks[j][1], float) + MOVE * (o["X"] - np.asarray(looks[j][1], float)))
            for j, o in row.items()}


def run(paths, m4=None, cache=None, rec11=None, rec11c=None, log=print, dev=None):
    """Stage 12. paths, m4 and cache as stage 11's (deep/__main__.py _stage11_inputs); rec11, rec11c: #26's records
    (stage11.json, stage11c.json, or their dry runs) for K1. Returns the record's body."""
    t0 = time.perf_counter()
    looks128, truth, fan, refs, vis_fn = paths["128"]
    R, per_ref = st.collect(paths, m4, cache, rows=["perfect", "N1024", MIRROR], log=log, dev=dev, extra=(st.N_EDGE,))
    edges = R.pop(st.N_EDGE)
    R[N10] = {j: st.edge_variant(o, truth, j, looks128, "rel") for j, o in edges.items()}
    R[MOVED] = moved(R["perfect"], looks128)
    attr = st.attributes(paths, refs, dev)
    R.update({CEIL: st.ceiling_rows(R, attr, looks128, fan)[CEIL]})
    snr = np.concatenate([_surface_snr(attr[j]["snr"], truth[j][1]) for j in refs])
    log(f"[stage12] rows ready ({time.perf_counter() - t0:.0f} s)")
    pos = positions(looks128, refs)
    out = {"references": refs, "positions_m": {str(j): round(p, 6) for j, p in pos.items()}, "spans_m": list(SPANS_M),
           "noise": {"gradient_snr_median": float(np.median(snr)), "share_under_1": float((snr < 1).mean()),
                     "share_under_3": float((snr < 3).mean()),
                     "pixels": "the references' pixels that see a surface, |latitude| < 70 deg (stage 11b's raster; "
                               "#26 quoted every 7th pixel of all, so its 1.65 differs slightly)"},
           "bands": {}}
    for bname, rows in BAND_ROWS.items():
        band = st.BANDS[bname]
        surf = st.surface_band(per_ref, truth, looks128, band)
        res, pcov = {}, None
        for name in rows:
            if name not in R:
                log(f"[stage12] {bname}: no row {name!r}")
                continue
            res[name], cov = models(R[name], per_ref, truth, fan, surf, band, looks128, pos, BAND_SECTORS[bname], pcov)
            if name == "perfect":
                pcov = cov
            log(f"[stage12] {bname} {name} ({time.perf_counter() - t0:.0f} s)")
        out["bands"][bname] = res
    out["checks"] = checks(out, rec11, rec11c)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def _eq(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-12


def checks(out, rec11, rec11c):
    res = []
    full, shared = out["bands"]["full"], out["bands"].get("shared 1-44", {})
    # K1: reproduction at span 0
    if rec11 is not None and rec11c is not None:
        refs_ = [("N1024", full["N1024"], rec11c["full"]["N1024"]["full"]),
                 (N10, full[N10], rec11c["full"][N10]["full"]),
                 ("perfect", full["perfect"], rec11["rows"]["perfect"]["full"])]
        if "M4@1024" in shared and "M4@1024" in rec11["rows"]:
            refs_.append(("M4@1024 (1-44)", shared["M4@1024"], rec11["rows"]["M4@1024"]["shared 1-44"]))
        lines = []
        ok = True
        for name, row, r in refs_:
            a, m0 = row["all"], row["spans"]["0"]["model"]
            same = (a["cells"] == r["cells"] and _eq(a["right"], r["right"])
                    and _eq(m0["right"], r["confirmed_right"]) and _eq(a["confirmed_right"], r["confirmed_right"]))
            ok &= bool(same)
            lines.append({"row": name, "equal": bool(same), "cells": [a["cells"], r["cells"]],
                          "model_right_span0": [m0["right"], r["confirmed_right"]], "right": [a["right"], r["right"]]})
        ctl = full["N1024"]["all"]["right"]
        res.append({"check": "K1: at span 0 every row equals #26's records (cells, right, cells from >= 2 references right)",
                    "ok": ok, "rows": lines,
                    "control": "N1024's cells right over all its cells in place of the model's",
                    "control_fails": not _eq(ctl, rec11c["full"]["N1024"]["full"]["confirmed_right"]),
                    "control_value": ctl})
    else:
        res.append({"check": "K1", "ok": None, "note": "no #26 records given"})
    # K2: the gate passes on the perfect matcher at 0.5 m and fails on it moved 20 %
    gp = full["perfect"]["spans"]["0.5"]["gate"]
    gm = full[MOVED]["spans"]["0.5"]["gate"] if MOVED in full else None
    res.append({"check": "K2: the perfect matcher's model at 0.5 m passes S1's gate (full field)",
                "ok": gp["result"] == "pass" if gp["result"] != "not evaluated" else None, "value": gp["result"],
                "control": MOVED + " must fail it",
                "control_fails": None if gm is None or gp["result"] == "not evaluated" else
                {"fail": True, "pass": False, "not evaluated": None}[gm["result"]],
                "control_value": gm["result"] if gm is not None else None})
    # the standard control: N mirrored never passes
    if MIRROR in full:
        r = {s: full[MIRROR]["spans"][s]["gate"]["result"] for s in full[MIRROR]["spans"]}
        res.append({"check": "the standard control: N1024 mirrored must not pass S1's gate at any span", "ok": None,
                    "control": MIRROR, "control_fails": all(v != "pass" for v in r.values()), "control_value": r})
    return res


def table(out):
    """The measurement, one line per row and span: the model, the candidates, S1's gate."""
    lines = ["| band | row | span | model cells | right | coverage | of perfect | candidates | right | S1 gate |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, p=2: "—" if x is None else f"{100 * x:.{p}f} %"
    for bname, rows in out["bands"].items():
        for name, r in rows.items():
            a = r["all"]
            lines.append(f"| {bname} | {name} | all cells | {a['cells']:,} | {f(a['right'])} | {f(a['coverage'], 1)} | | | | |")
            for s, e in r["spans"].items():
                m, c, g = e["model"], e["candidates"], e["gate"]
                fails = [k for k, p in g["parts"].items() if p["result"] == "fail"]
                nev = [k for k, p in g["parts"].items() if p["result"] == "not evaluated"]
                gs = g["result"] + "".join(f"; {lab}: {', '.join(v)}" for lab, v in (("fails", fails), ("too few", nev)) if v)
                lines.append(f"| {bname} | {name} | ≥ {s} m | {m['cells']:,} | {f(m['right'])} | {f(m['coverage'], 1)} | "
                             f"{f(m.get('covers_of_perfect'), 1)} | {c['cells']:,} | {f(c['right'])} | {gs} |")
    return lines


def selftest():
    """The span logic on cells with known references (no data). (name, ok, value, control, control fails, value)."""
    looks = {j: (0.0, np.array([0.0, 0.01 * j, 0.0]), np.eye(3)) for j in (50, 60, 100, 110, 120)}
    pos = positions(looks, list(looks))
    ref = np.array([50, 60, 50, 100, 50, 60, 110, 120])
    cell = np.array([0, 0, 1, 1, 2, 3, 3, 3])
    nref = np.array([2, 2, 1, 3])
    sp = spans(ref, cell, 4, pos)
    got = {s: confirmed(nref, sp, s).tolist() for s in SPANS_M}
    want = {0.0: [True, True, False, True], 0.2: [False, True, False, True], 0.5: [False, True, False, True]}
    pos_looks = {j: p * 100 for j, p in pos.items()}                     # positions in looks, not metres
    sp_bad = spans(ref, cell, 4, pos_looks)
    bad = {s: confirmed(nref, sp_bad, s).tolist() for s in SPANS_M}
    out = [("Z1: spans and confirmation on known cells", got == want and np.allclose(sp, [0.1, 0.5, 0.0, 0.6]),
            {"spans": sp.round(6).tolist(), "confirmed": got}, "positions in looks instead of metres", bad != want, bad)]
    near = {0.0: [True], 0.2: [False], 0.5: [False]}
    one = {s: confirmed(np.array([2]), np.array([0.2 - 0.5e-3]), s).tolist() for s in SPANS_M}
    off = {s: confirmed(np.array([2]), np.array([0.2 - 2e-3]), s).tolist() for s in SPANS_M}
    out.append(("Z2: a span 0.5 mm short of 0.2 m counts (1 mm slack)", one == {0.0: [True], 0.2: [True], 0.5: [False]},
                one, "2 mm short", off == near, off))
    # Z3: the within-2 cm rule on known points: reference 50's estimate at a cell, reference 100's 1.5 cm away (in
    # another 2 cm cell), and 3 cm away for the control
    def sup_of(d):
        X = np.array([[0.011, 0.011, 0.011], [0.011 + d, 0.011, 0.011]])
        ref = np.array([50, 100])
        Xf, cell_of = X.copy(), np.array([0, 1])          # each point is its own cell (2 cm cells: 0.011 + d is not in the first)
        same = support(X, ref, Xf, cell_of, [50, 100], "same cell")
        nb = support(X, ref, Xf, cell_of, [50, 100], "within 2 cm")
        return same[0].tolist(), nb[0].tolist()
    s15, n15 = sup_of(0.015)
    s30, n30 = sup_of(0.030)
    out.append(("Z3: an estimate 1.5 cm away in another cell supports under 'within 2 cm' only",
                s15 == [True, False] and n15 == [True, True], {"same cell": s15, "within 2 cm": n15},
                "3 cm away", n30 == [True, False], {"within 2 cm": n30}))
    return out


# ---------------------------------------------------------------- stage 12b (#28): two confirmation rules (P-D5)

SPANS_12B = (0.0, 0.2, 0.5, 0.8, 1.0)   # P-D5 (Luiz, 2026-11-21)
RULES = ("same cell", "within 2 cm")
# - same cell (S1's, S2's): a reference supports a cell when one of its estimates lies in the cell;
# - within 2 cm: ... or when one of its estimates lies within 2 cm of the cell's fused point (stage 8's coverage
#   distance, assess.covered). It contains the first by construction.


def support(X, ref, Xf, cell_of, refs, rule):
    """(cells x references) bool: which references support each cell under `rule`."""
    col = {j: k for k, j in enumerate(refs)}
    sup = np.zeros((len(Xf), len(refs)), bool)
    if len(X):
        sup[cell_of, np.array([col[int(j)] for j in ref])] = True
    if rule == "within 2 cm":
        for j, k in col.items():
            m = ref == j
            if m.any():
                sup[:, k] |= asx.covered(Xf, X[m])
    elif rule != "same cell":
        raise ValueError(rule)
    return sup


def models_b(row, per_ref, truth, fan, surf, band, looks, pos, sectors, refs, perfect_cov=None, spans=SPANS_12B):
    """Stage 12b: one fusion of a row in a band (as models()), its models under both RULES at `spans` (SPANS_12B)."""
    Xs, refs_of = [], []
    for j, o in row.items():
        sel = np.ones(len(o["psi"]), bool) if band is None else (o["psi"] >= band[0]) & (o["psi"] <= band[1])
        Xs.append(o["X"][sel])
        refs_of.append(np.full(int(sel.sum()), j))
    if sum(len(x) for x in Xs) == 0:
        X, ref = np.zeros((0, 3)), np.zeros(0, int)
        Xf, dom, right, cell_of = np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, bool), np.zeros(0, int)
    else:
        X, ref, Xf, nref, dom, right, cell_of = asx.fuse_and_judge(Xs, refs_of, per_ref, truth, fan)
    psi = cell_psi(Xf, dom, looks)
    pv = np.array([pos[j] for j in refs])
    cov_all = asx.covered(surf[1], Xf[right]) if len(Xf) else np.zeros(len(surf[1]), bool)
    out = {"all": {"cells": int(len(Xf)), "right": float(right.mean()) if len(Xf) else None,
                   "coverage": float(cov_all.mean())}, "rules": {}}
    conf = {}
    for rule in RULES:
        sup = support(X, ref, Xf, cell_of, refs, rule)
        nsup = sup.sum(1)
        hi = np.where(sup, pv[None, :], -np.inf).max(1) if len(Xf) else np.zeros(0)
        lo = np.where(sup, pv[None, :], np.inf).min(1) if len(Xf) else np.zeros(0)
        span = np.where(nsup > 0, hi - lo, 0.0)
        out["rules"][rule] = {"spans": {}}
        for s in spans:
            c = (nsup >= 2) & (span >= s - SLACK_M)
            conf[(rule, s)] = c
            cov = asx.covered(surf[1], Xf[right & c]) if c.any() else np.zeros(len(surf[1]), bool)
            nr = int((right & c).sum())
            e = {"model": {"cells": int(c.sum()), "right": nr / int(c.sum()) if c.any() else None,
                           "right_cells": nr, "wrong_cells": int(c.sum()) - nr, "coverage": float(cov.mean())},
                 "candidates": {"cells": int((~c).sum()), "right": float(right[~c].mean()) if (~c).any() else None},
                 "gate": sd.gate(right[c], psi[c], sectors)}
            if perfect_cov is not None and perfect_cov.any():
                e["model"]["covers_of_perfect"] = float((cov & perfect_cov).sum() / perfect_cov.sum())
            out["rules"][rule]["spans"][f"{s:g}"] = e
    out["contains"] = {f"{s:g}": bool(np.all(conf[("within 2 cm", s)] | ~conf[("same cell", s)])) for s in spans}
    out["contained"] = {f"{s:g}": bool(np.all(conf[("same cell", s)] | ~conf[("within 2 cm", s)])) for s in spans}
    return out, cov_all


def run_b(paths, m4=None, cache=None, rec12=None, log=print, dev=None):
    """Stage 12b. As run(), with both RULES and SPANS_12B; rec12: #28's stage12.json (or its dry run) for K1b."""
    t0 = time.perf_counter()
    looks128, truth, fan, refs, vis_fn = paths["128"]
    R, per_ref = st.collect(paths, m4, cache, rows=["perfect", "N1024", MIRROR], log=log, dev=dev, extra=(st.N_EDGE,))
    edges = R.pop(st.N_EDGE)
    R[N10] = {j: st.edge_variant(o, truth, j, looks128, "rel") for j, o in edges.items()}
    R[MOVED] = moved(R["perfect"], looks128)
    attr = st.attributes(paths, refs, dev)
    R.update({CEIL: st.ceiling_rows(R, attr, looks128, fan)[CEIL]})
    log(f"[stage12b] rows ready ({time.perf_counter() - t0:.0f} s)")
    pos = positions(looks128, refs)
    out = {"references": refs, "positions_m": {str(j): round(p, 6) for j, p in pos.items()}, "spans_m": list(SPANS_12B),
           "rules": list(RULES), "bands": {}}
    for bname, rows in BAND_ROWS.items():
        band = st.BANDS[bname]
        surf = st.surface_band(per_ref, truth, looks128, band)
        res, pcov = {}, None
        for name in rows:
            if name not in R:
                log(f"[stage12b] {bname}: no row {name!r}")
                continue
            res[name], cov = models_b(R[name], per_ref, truth, fan, surf, band, looks128, pos, BAND_SECTORS[bname],
                                      sorted(refs), pcov)
            if name == "perfect":
                pcov = cov
            log(f"[stage12b] {bname} {name} ({time.perf_counter() - t0:.0f} s)")
        out["bands"][bname] = res
    out["checks"] = checks_b(out, rec12)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def checks_b(out, rec12):
    res = []
    full = out["bands"]["full"]
    if rec12 is not None:
        lines, ok = [], True
        for bname, rows in out["bands"].items():
            for name, r in rows.items():
                old = rec12["bands"].get(bname, {}).get(name)
                if old is None:
                    continue
                for s in SPANS_M:
                    a, b = r["rules"]["same cell"]["spans"][f"{s:g}"]["model"], old["spans"][f"{s:g}"]["model"]
                    same = a["cells"] == b["cells"] and (a["right"] == b["right"] or _eq(a["right"], b["right"]))
                    ok &= bool(same)
                    if not same:
                        lines.append({"band": bname, "row": name, "span": s, "new": [a["cells"], a["right"]],
                                      "stage12": [b["cells"], b["right"]]})
        p0 = full["perfect"]["rules"]
        res.append({"check": "K1b: the same-cell rule at 0, 0.2 and 0.5 m equals stage 12's models (every row and band)",
                    "ok": ok, "differences": lines,
                    "control": "the within-2 cm rule at span 0 (perfect, full field) in place of the same-cell rule",
                    "control_fails": p0["within 2 cm"]["spans"]["0"]["model"]["cells"]
                    != rec12["bands"]["full"]["perfect"]["spans"]["0"]["model"]["cells"],
                    "control_value": [p0["within 2 cm"]["spans"]["0"]["model"]["cells"],
                                      rec12["bands"]["full"]["perfect"]["spans"]["0"]["model"]["cells"]]})
    else:
        res.append({"check": "K1b", "ok": None, "note": "no stage 12 record given"})
    allc = all(v for rows in out["bands"].values() for r in rows.values() for v in r["contains"].values())
    conv = full["perfect"]["contained"]["0"]
    res.append({"check": "K2b: at every span, row and band, the within-2 cm model contains the same-cell model",
                "ok": allc, "control": "the converse (perfect, full field, span 0)", "control_fails": not conv,
                "control_value": conv})
    gp = full["perfect"]["rules"]["within 2 cm"]["spans"]["0.5"]["gate"]
    gm = full[MOVED]["rules"]["within 2 cm"]["spans"]["0.5"]["gate"] if MOVED in full else None
    res.append({"check": "K3b: the perfect matcher's within-2 cm model at 0.5 m passes S2's gate (full field)",
                "ok": None if gp["result"] == "not evaluated" else gp["result"] == "pass",
                "value": {"result": gp["result"], "unverified": gp["unverified"]},
                "control": MOVED + " must fail it",
                "control_fails": None if gm is None or gp["result"] == "not evaluated" else
                {"fail": True, "pass": False, "not evaluated": None}[gm["result"]],
                "control_value": gm["result"] if gm is not None else None})
    if MIRROR in full:
        r = {f"{rule} {s}": full[MIRROR]["rules"][rule]["spans"][s]["gate"]["result"]
             for rule in RULES for s in full[MIRROR]["rules"][rule]["spans"]}
        res.append({"check": "the standard control: N1024 mirrored must not pass S2's gate under either rule at any span",
                    "ok": None, "control": MIRROR, "control_fails": all(v != "pass" for v in r.values()),
                    "control_value": r})
    return res


def table_b(out):
    lines = ["| band | row | rule | span | model cells | right | wrong cells | coverage | of perfect | S2 gate |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, p=2: "—" if x is None else f"{100 * x:.{p}f} %"
    for bname, rows in out["bands"].items():
        for name, r in rows.items():
            a = r["all"]
            lines.append(f"| {bname} | {name} | all cells | | {a['cells']:,} | {f(a['right'])} | | {f(a['coverage'], 1)} | | |")
            for rule, rr in r["rules"].items():
                for s, e in rr["spans"].items():
                    m, g = e["model"], e["gate"]
                    fails = [k for k, p in g["parts"].items() if p["result"] == "fail"]
                    gs = g["result"] + (f"; fails: {', '.join(fails)}" if fails else "") + (
                        f"; unverified: {', '.join(g['unverified'])}" if g["unverified"] else "")
                    lines.append(f"| {bname} | {name} | {rule} | ≥ {s} m | {m['cells']:,} | {f(m['right'])} | "
                                 f"{m['wrong_cells']:,} | {f(m['coverage'], 1)} | {f(m.get('covers_of_perfect'), 1)} | {gs} |")
    return lines
