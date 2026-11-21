"""Demo D3 (#29): N1024 along the Classroom's straight path, as S2's model grows (DERIVED: a display; it adds no gate).

Luiz, 2026-11-21 (P-D5): a test demo on the Classroom's straight path, as a video and PLY clouds, with N1024 alone.
- **The run:** R3's straight_long (218 looks). N1024 (deep/ntrace.py, M4's 0.5 px depth-edge test) at stage 11's 12
  references, taken from stage 11's cache, as in stages 12 and 12b. Each is judged by stage 9's metric against the
  128-spp run's truth, with B128's windows as the judge.
- **The release:** a reference's estimates are shown from look m + 50 on, when its window is complete (D2's rule).
- **The model:** after each release, all released references are fused into 2 cm cells (stage 8) and split by S2's
  confirmation (deep/confirm.py, at the chosen rule and span) into the model and the candidates.

Frame (1920 x 1080):
    +-------------------------------------------+-------------------------------------------+
    | the current look, with the model's cells   | as the path goes: the model and every cell |
    +-------------------------------------------+-------------------------------------------+
    | 3-D: S2's model (right cells by height,    | 3-D: every cell (the model as on the left, |
    |      wrong cells in orange)                |      the candidates in grey)               |
    +-------------------------------------------+-------------------------------------------+
It ends with an orbit of the end maps. The video stays on the workstation (runs/).

PLY (#25's format, deep/ply.py), coloured from the scene: <name>_model.ply (the end model) and <name>_all.ply (every
cell, with a `confirmed` flag). Per vertex: x y z, red green blue, right, err, nref, nsup, span_m, confirmed, psi.

Checks (declared on #29 before data), each with a control that must fail:
- V1a: the end model equals the batch model on the same inputs (deep/confirm.py models_b, full field), in cells,
  points and cells right. Control: the model before the last release differs.
- V1b (real run): the end model equals stage 12b's N1024 model at the rule and span (cells and cells right).
  Control: the same against stage 12b's next shorter span.
- V2: nothing is shown before its window is complete (m + 50 <= t). Control: a schedule releasing at m.
- P1: each PLY read back holds exactly its map (points to float32, right and confirmed flags). Control: the map
  without the last reference.
- C: the colours' error (D2's cells_error <= 10 %) equals stage 9's cells right on every end cell. Control: 5 %.
- The standard control (S2, build): N1024 mirrored's end model must not pass S2's gate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import assess as asx
from . import confirm as cf
from . import demo as dm
from . import demo2 as d2
from . import e2e
from . import standard as sd
from . import straight as st

W, H, FPS = dm.W, dm.H, dm.FPS
LAG = e2e.HALF
GREY = np.array((196, 196, 190), np.uint8)
TOP_BOX, PROG_BOX = (0, 0, 960, 480), (960, 0, 960, 480)
LEFT_BOX, RIGHT_BOX = (0, 480, 960, 600), (960, 480, 960, 600)


def model_of(pts, released, per_ref, truth, fan, pos, rule, span):
    """S2's model of the released references' points ({j: X}): fused cells, their judgement and confirmation."""
    if not released:
        z = np.zeros(0)
        return {"Xf": np.zeros((0, 3)), "right": z.astype(bool), "err": z, "nref": z.astype(int), "nsup": z.astype(int),
                "span": z, "conf": z.astype(bool), "dom": z.astype(int), "points": 0}
    X = np.concatenate([pts[j] for j in released])
    ref = np.concatenate([np.full(len(pts[j]), j) for j in released])
    fin = np.all(np.isfinite(X), 1)
    X, ref = X[fin], ref[fin]
    Xf, nref, dom = e2e.fuse(X, ref)
    cell_of = e2e.fuse.cell_of
    right = e2e.cells_right(Xf, dom, per_ref, truth, fan)
    err = d2.cells_error(Xf, dom, per_ref, truth, fan)
    refs = sorted(released)
    sup = cf.support(X, ref, Xf, cell_of, refs, rule)
    nsup = sup.sum(1)
    pv = np.array([pos[j] for j in refs])
    hi = np.where(sup, pv[None, :], -np.inf).max(1)
    lo = np.where(sup, pv[None, :], np.inf).min(1)
    sp = np.where(nsup > 0, hi - lo, 0.0)
    conf = (nsup >= 2) & (sp >= span - cf.SLACK_M)
    return {"Xf": Xf, "right": right, "err": err, "nref": nref, "nsup": nsup, "span": sp, "conf": conf, "dom": dom,
            "points": int(len(X))}


def colours(mp, sc, with_candidates):
    """D2's 'match' colours (right by height, wrong in orange) for the model; the candidates grey, or left out."""
    Xf, conf = mp["Xf"], mp["conf"]
    sel = np.ones(len(Xf), bool) if with_candidates else conf
    col = d2.colours_for(Xf[sel], mp["err"][sel], "match", sc)
    if with_candidates:
        col[~conf] = GREY
    return Xf[sel], col, conf[sel]


def prog(hist, total_s, box=PROG_BOX):
    x0, y0, w, h = box
    im = Image.new("RGB", (w, h), dm.PAPER)
    d = ImageDraw.Draw(im)
    L, R_, T_, B_ = 70, w - 40, 70, h - 110
    dm._text(d, (16, 10), "As the path goes: coverage of the surface (solid) and cells right (dashed)", 18)
    for v in (0, 25, 50, 75, 100):
        y = B_ - (B_ - T_) * v / 100
        d.line([(L, y), (R_, y)], fill=(225, 225, 220), width=1)
        dm._text(d, (L - 8, y), f"{v} %", 14, dm.MUTED, anchor="rm")
    for v in np.arange(0, total_s + 1e-9, 0.5):
        dm._text(d, (L + (R_ - L) * v / total_s, B_ + 10), f"{v:g}", 14, dm.MUTED, anchor="mt")
    dm._text(d, (R_, B_ + 34), "metres along the path", 14, dm.MUTED, anchor="rt")
    cols = {"model": dm.INK, "all": tuple(int(v) for v in GREY * 0.8)}
    names = {"model": "S2's model", "all": "every cell"}
    for key in ("all", "model"):
        pts_c = [(L + (R_ - L) * s / total_s, B_ - (B_ - T_) * hc[key][0]) for s, hc in hist]
        pts_r = [(L + (R_ - L) * s / total_s, B_ - (B_ - T_) * hc[key][1]) for s, hc in hist if hc[key][1] is not None]
        if len(pts_c) > 1:
            d.line(pts_c, fill=cols[key], width=3)
        for a_, b_ in zip(pts_r[::2], pts_r[1::2]):
            d.line([a_, b_], fill=cols[key], width=2)
    if hist:
        s, hc = hist[-1]
        yy = h - 64
        for i, key in enumerate(("model", "all")):
            c, r, n = hc[key]
            txt = f"{names[key]}: {n:,} cells, coverage {100 * c:.1f} %" + (f", {100 * r:.2f} % right" if r is not None else "")
            d.rectangle([L, yy + 22 * i + 4, L + 14, yy + 22 * i + 16], fill=cols[key])
            dm._text(d, (L + 22, yy + 22 * i), txt, 16, dm.INK)
    return im


def frame(sc, looks, m, mp, hist, total_s, fan, rule, span, label):
    im = Image.new("RGB", (W, H), dm.PAPER)
    Xm, cm, _ = colours(mp, sc, False)
    im.paste(d2.pano(looks, m, Xm, cm, fan), TOP_BOX[:2])
    im.paste(prog(hist, total_s), PROG_BOX[:2])
    for box, cand, title in ((LEFT_BOX, False, f"S2's model ({rule}, references ≥ {span:g} m apart)"),
                             (RIGHT_BOX, True, "Every cell N reported: the model, and the candidates in grey")):
        X, c, _ = colours(mp, sc, cand)
        keep = d2.subset(len(X))
        im.paste(d2.cloud(sc, X[keep], c[keep], -115.0, box[2:], title, m=m), box[:2])
    d = ImageDraw.Draw(im)
    d.line([(0, 480), (W, 480)], fill=dm.RULE, width=1)
    d.line([(960, 0), (960, H)], fill=dm.RULE, width=1)
    dm._text(d, (976, 34), f"{label} · {sc.T[m][3]:.2f} m · look {m + 1} of {len(sc.T)} · right cells by height, "
                           f"wrong cells in orange", 14, dm.MUTED)
    return im


def scene_rgb(Xf, dom, per_ref, rgb_of, fan):
    """Each cell's colour from its dominant reference's image at the cell's pixel (#25's export_ply)."""
    rgb = np.zeros((len(Xf), 3), np.uint8)
    model = fan.model()
    for j in np.unique(dom):
        sel = np.flatnonzero(dom == j)
        img = rgb_of(int(j))
        Hh, Ww = img.shape[:2]
        rc, _ = model.pixel_of((Xf[sel] - per_ref[j]["c"]) @ per_ref[j]["R"])
        r = np.clip(np.rint(rc[:, 0]), 0, Hh - 1).astype(int)
        c = (np.rint(rc[:, 1]).astype(int)) % Ww
        v = np.nan_to_num(img[r, c], nan=0.0)
        rgb[sel] = (np.clip(v, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    return rgb


def write_plys(mp, rgb, ply_dir, name, rule, span, n_refs):
    from . import ply
    ply_dir = Path(ply_dir)
    ply_dir.mkdir(parents=True, exist_ok=True)
    Xf = mp["Xf"]
    psi = np.full(len(Xf), np.nan)
    ex = {"right": mp["right"].astype(np.uint8),
          "err": np.where(np.isfinite(mp["err"]), mp["err"], -1).astype(np.float32),
          "nref": np.minimum(mp["nref"], 255).astype(np.uint8), "nsup": np.minimum(mp["nsup"], 255).astype(np.uint8),
          "span_m": mp["span"].astype(np.float32), "confirmed": mp["conf"].astype(np.uint8)}
    if "psi" in mp:
        psi = mp["psi"]
    ex["psi"] = psi.astype(np.float32)
    out = {}
    for which, sel in (("model", mp["conf"]), ("all", np.ones(len(Xf), bool))):
        path = ply_dir / f"{name}_{which}.ply"
        size = ply.write_ply(path, Xf[sel], rgb[sel], extras={k: v[sel] for k, v in ex.items()},
                             comments=(f"deep-motion demo D3 (#29): N1024, the Classroom's straight path, {n_refs} references",
                                       f"S2's model: {rule}, references >= {span:g} m apart; world metres, +Z up",
                                       "right = stage 9's metric (within 10 % in a 3x3 neighbourhood); confirmed = in S2's model"))
        back = ply.read_ply(path)
        out[which] = {"path": str(path), "bytes": int(size), "points": int(sel.sum()), "back": back, "sel": sel}
    return out


def run(paths, cache, out_dir, ply_dir, T, rgb_of, rule="same cell", span=sd.CONFIRM_SPAN_M, rec12b=None, name="classroom_straight",
        label="Classroom, straight path", limit=None, best=False, video=True, log=print):
    """Demo D3. paths as stage 11's ({'128': (looks, truth, fan, refs, vis), '1024': ...}); T [(path or None, c, R, s, 0)]
    for the scene; rgb_of(j) a reference's image as (H, W, 3) linear RGB. Returns the record (body)."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    looks128, truth, fan, refs, _ = paths["128"]
    looks_show = paths["1024"][0] if paths.get("1024") else looks128
    R, per_ref = st.collect(paths, None, cache, rows=["N1024", cf.MIRROR], log=log)
    pts = {j: R["N1024"][j]["X"] for j in refs}
    pts_m = {j: R[cf.MIRROR][j]["X"] for j in refs}
    pos = cf.positions(looks128, refs)
    surf = st.surface_band(per_ref, truth, looks128, None)
    P = np.concatenate([truth[j][0][np.all(np.isfinite(truth[j][0]), 1)] for j in refs])
    sc = dm.Scene(P[d2.subset(len(P), 200000)], T, fan)
    total_s = float(sc.s[-1]) + 0.02
    K = len(T) if not limit else min(limit, len(T))
    enc = None
    if video:
        vdir = Path(ply_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        enc = dm._Enc(vdir / f"{name}_demo3.mp4", best)
    log_rel, hist, cur, mp, hc, models_at = [], [], None, None, None, []
    last = None
    for m in range(K):
        rel = d2.released_at(m, refs)
        log_rel.append((m, list(rel)))
        if rel != cur:
            cur = list(rel)
            mp = model_of(pts, rel, per_ref, truth, fan, pos, rule, span)
            mp["psi"] = cf.cell_psi(mp["Xf"], mp["dom"], looks128)
            r, c = mp["right"], mp["conf"]
            hc = {"model": (float(asx.covered(surf[1], mp["Xf"][r & c]).mean()) if c.any() else 0.0,
                            float(r[c].mean()) if c.any() else None, int(c.sum())),
                  "all": (float(asx.covered(surf[1], mp["Xf"][r]).mean()) if len(r) else 0.0,
                          float(r.mean()) if len(r) else None, int(len(r)))}
            models_at.append({"look": m, "released": len(rel), "model_cells": hc["model"][2], "model_right": hc["model"][1],
                              "model_coverage": hc["model"][0], "cells": hc["all"][2], "right": hc["all"][1],
                              "coverage": hc["all"][0]})
            log(f"[demo3] look {m + 1}: {len(rel)} references released, model {hc['model'][2]:,} cells, "
                f"all {hc['all'][2]:,} ({time.perf_counter() - t0:.0f} s)")
        hist.append((float(T[m][3]), hc))
        if enc is not None:
            last = frame(sc, looks_show, m, mp, hist, total_s, fan, rule, span, label)
            enc.put(last, int(dm.HOLD_START * FPS) if m == 0 else 1)
    if enc is not None and last is not None:
        last.save(out_dir / f"last_frame{'' if T[0][0] is not None else '_dry'}.png")
        enc.put(last, int(dm.HOLD_END * FPS))
        if not limit:
            n = int(dm.ORBIT_S * FPS)
            azs = tuple(-115.0 + 360.0 * i / n for i in range(n))
            for i in range(n):
                card = Image.new("RGB", (W, H), dm.PAPER)
                for x0, cand, title in ((0, False, "S2's model"), (960, True, "Every cell (candidates in grey)")):
                    X, c, _ = colours(mp, sc, cand)
                    keep = d2.subset(len(X))
                    card.paste(d2.cloud(sc, X[keep], c[keep], azs[i], (960, H), title, m=len(T) - 1, azs=azs), (x0, 0))
                enc.put(card)
        enc.close()
    full = K == len(T)
    # ---------------------------------------------------------------- the checks
    checks = []
    batch, _ = cf.models_b(R["N1024"], per_ref, truth, fan, surf, None, looks128, pos, list(sd.SECTORS), sorted(refs))
    key = f"{span:g}"
    prev_all = model_of(pts, refs[:-1], per_ref, truth, fan, pos, rule, span)
    b = batch["rules"][rule]["spans"].get(key)
    if b is None:
        checks.append({"name": f"V1a: no batch span {key} m in stage 12b's SPANS_12B", "ok": None})
    else:
        bm = b["model"]
        endm = {"cells": int(mp["conf"].sum()), "right_cells": int((mp["right"] & mp["conf"]).sum())}
        prev = prev_all
        checks.append({"name": "V1a: the end model equals the batch model (cells, cells right)", "ok": bool(full and
                       endm["cells"] == bm["cells"] and endm["right_cells"] == bm["right_cells"]),
                       "end": endm, "batch": {"cells": bm["cells"], "right_cells": bm["right_cells"]},
                       "control": "the model before the last release", "control_fails": int(prev["conf"].sum()) != bm["cells"],
                       "control_value": int(prev["conf"].sum())})
    if rec12b is not None:
        spans = rec12b["bands"]["full"]["N1024"]["rules"][rule]["spans"]
        r12 = spans.get(key)
        shorter = [s for s in spans if float(s) < span - 1e-9]
        if r12 is None:
            checks.append({"name": f"V1b: stage 12b has no span {key} m", "ok": None})
        else:
            endc, endr = int(mp["conf"].sum()), float(mp["right"][mp["conf"]].mean()) if mp["conf"].any() else None
            ctl = spans[max(shorter, key=float)]["model"] if shorter else None
            checks.append({"name": f"V1b: the end model equals stage 12b's N1024 model ({rule}, ≥ {key} m)",
                           "ok": bool(full and endc == r12["model"]["cells"] and endr is not None
                                      and abs(endr - r12["model"]["right"]) <= 1e-12),
                           "end": [endc, endr], "stage12b": [r12["model"]["cells"], r12["model"]["right"]],
                           "control": "stage 12b's next shorter span",
                           "control_fails": None if ctl is None else ctl["cells"] != endc,
                           "control_value": None if ctl is None else ctl["cells"]})
    ok2, n2 = d2.check_release(log_rel)
    ok2c, n2c = d2.check_release([(t, d2.released_at(t, refs, lag=0)) for t, _ in log_rel])
    checks.append({"name": "V2: nothing shown before its window is complete (m + 50 <= t)", "ok": bool(ok2),
                   "violations": n2, "control": "a schedule releasing at m", "control_fails": not ok2c,
                   "control_value": n2c})
    e10 = mp["err"] <= 0.10
    e05 = mp["err"] <= 0.05
    checks.append({"name": "C: the colours' error (<= 10 %) equals stage 9's cells right on every end cell",
                   "ok": bool(np.array_equal(e10, mp["right"])), "control": "5 % in place of 10 %",
                   "control_fails": not np.array_equal(e05, mp["right"]), "control_value": int((e05 != mp["right"]).sum())})
    # PLY
    rgb = scene_rgb(mp["Xf"], mp["dom"], per_ref, rgb_of, fan)
    files = write_plys(mp, rgb, ply_dir, name, rule, span, len(refs))
    for which, f in files.items():
        back, sel = f.pop("back"), f.pop("sel")
        Xb = np.stack([back["x"], back["y"], back["z"]], 1)
        same = (len(back) == int(sel.sum()) and np.array_equal(Xb, mp["Xf"][sel].astype(np.float32))
                and np.array_equal(back["right"].astype(bool), mp["right"][sel])
                and np.array_equal(back["confirmed"].astype(bool), mp["conf"][sel]))
        psel = prev_all["conf"] if which == "model" else np.ones(len(prev_all["Xf"]), bool)
        ctl = len(back) == int(psel.sum()) and np.array_equal(Xb, prev_all["Xf"][psel].astype(np.float32))
        checks.append({"name": f"P1 {which}: the file read back holds exactly its map", "ok": bool(same),
                       "control": "the map without the last reference", "control_fails": not ctl})
        f["mb"] = round(f["bytes"] / 1e6, 2)
    # the standard control: the mirrored poses' end model
    mm = model_of(pts_m, refs, per_ref, truth, fan, pos, rule, span)
    mm_psi = cf.cell_psi(mm["Xf"], mm["dom"], looks128)
    gm = sd.gate(mm["right"][mm["conf"]], mm_psi[mm["conf"]])
    checks.append({"name": "the standard control: N1024 mirrored's end model must not pass S2's gate", "ok": None,
                   "control": cf.MIRROR, "control_fails": gm["result"] != "pass", "control_value": gm["result"]})
    gate = sd.gate(mp["right"][mp["conf"]], mp["psi"][mp["conf"]])
    rec = {"command": "demo3", "references": refs, "lag_looks": LAG, "looks": K, "path_looks": len(T), "rule": rule,
           "span_m": span, "end": {"model": {"cells": int(mp["conf"].sum()),
                                             "right": float(mp["right"][mp["conf"]].mean()) if mp["conf"].any() else None,
                                             "coverage": hist[-1][1]["model"][0]},
                                   "all": {"cells": int(len(mp["Xf"])), "right": float(mp["right"].mean()) if len(mp["Xf"]) else None,
                                           "coverage": hist[-1][1]["all"][0]},
                                   "candidates": int((~mp["conf"]).sum()), "gate": gate},
           "at_releases": models_at, "files": files, "frames": enc.n if enc is not None else 0, "fps": FPS,
           "display_max_cells_per_side": d2.MAX_DRAW, "checks": checks, "seconds": round(time.perf_counter() - t0, 1)}
    return rec
