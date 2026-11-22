"""Demo D5 (#35): demo D2 redone on the Classroom's L-tour with N's coverage map, as the validation of P-D8 (P-D9).

Luiz, 2026-11-21 (P-D9; accepted on #34): redo demo D2 (#23) on its tour, the straight path's first 2 m then the turn
(418 looks, references every 10 looks), with the perfect matcher beside the coverage map of P-D8. The match video draws
every cell in the scene's RGB colour; the error video stays as D2's; the PLY files are written.

- **The perfect matcher** is D2's, with D2's draws (seed 23, in D2's order), so its end map is D2's (check V0).
- **The coverage map** is P-D8's: N on each reference's +/-0.5 m window joined with N on its +/-0.2 m window, rays
  1-179 deg, the fused cells with at least 3 neighbours among the 26 around them. N takes its line path where a window
  is on one line and its per-frame path elsewhere (stage 14, #34), on the 1024-spp renders.
- **The judge** is stage 9's metric against the 128-spp truth, B's windows at each reference (stage 11's).
- **The release** is D2's: a reference's points are shown from look m + 50 on.

Checks (declared on #35 before data), each with a control that must fail:
- V0: the perfect side's end map equals D2's (cells and cells right, demo2.json). Control: without the last reference.
- V2: the tour is the planned L-curve sampled evenly (D2's). Control: the turn's looks shifted 1 cm sideways.
- V3: both end maps equal their batch fusion (the map: with the neighbour filter). Control: without the last reference.
- V4: nothing is shown before its window is complete. Control: a schedule releasing at m.
- C: the error video's colours (err <= 10 %) equal stage 9's cells right on the map. Control: 5 %.
- P1: each PLY file read back holds exactly its map. Control: the map without the last reference.
- The standard control: N with the mirrored path's poses must not give a usable map (>= 1,000 cells at >= 97 %).
The validation (declared before data, the experiment's answer, not a code check): on the cells whose dominant
reference's window leaves the line (the turn's arc, where N takes its per-frame path), is the map at least 97 % right?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import assess as asx
from . import coverage as cv
from . import demo as dm
from . import demo2 as d2
from . import demo3 as d3
from . import e2e
from . import standard as sd
from . import straight as st

W, H, FPS = dm.W, dm.H, dm.FPS
LAG = e2e.HALF
FLOOR = 0.97
MIN_NB = 3


def perfect_points(looks, truth, vis_fn, fan, refs, seed=23):
    """D2's perfect matcher with D2's draws (demo2.per_reference's order: one oracle_fit per reference)."""
    rng = np.random.default_rng(seed)
    pts = {j: asx.oracle_points(truth, j, looks, fan) for j in refs}
    vis = asx.visibility(looks, refs, pts, vis_fn)
    out = {}
    for j in refs:
        X, us = asx.oracle_fit(looks, j, pts[j], vis[j], rng)
        out[j] = X[us]
    return out


def real_points(looks1024, truth, fan, refs, cache, dev=None, log=print):
    """N's points per reference: the +/-0.5 m window joined with the +/-0.2 m window (rays 1-179: N's own), and N on
    the mirrored path's poses (the standard control). Cached per reference (runs/)."""
    cdir = Path(cache) if cache else None
    cp = lambda name, j: (cdir / name / f"r{j:04d}.npz") if cdir else None
    out, mir, paths_used = {}, {}, {}
    t0 = time.perf_counter()
    for j in refs:
        a = st._cache(cp("N1024", j), lambda j=j: st.n_reference(looks1024, truth, j, fan, dev=dev))
        b = st._cache(cp("N1024_short", j), lambda j=j: st.n_reference(looks1024, truth, j, fan, offsets=st.WIN_SHORT, dev=dev))
        m = st._cache(cp("N1024_mirrored", j), lambda j=j: st.n_reference(looks1024, truth, j, fan, mirror=True, dev=dev))
        out[j] = np.concatenate([a["X"], b["X"]])
        mir[j] = m["X"]
        paths_used[j] = {"long": "per-frame" if a.get("general") else "line", "short": "per-frame" if b.get("general") else "line"}
        log(f"[demo5] reference {j}: N {a['accepted']} + {b['accepted']} points ({paths_used[j]['long']}, "
            f"{paths_used[j]['short']}) ({time.perf_counter() - t0:.0f} s)")
    return out, mir, paths_used


def fuse_map(pts, released, per_ref, truth, fan, nb=None):
    """Fused cells of the released references, judged; with nb, the map keeps cells with >= nb neighbours."""
    Xf, nref, dom = d2.fuse_side(pts, released)
    if len(Xf) == 0:
        z = np.zeros(0)
        return {"Xf": Xf, "nref": nref, "dom": dom, "right": z.astype(bool), "err": z, "keep": z.astype(bool)}
    right = e2e.cells_right(Xf, dom, per_ref, truth, fan)
    err = d2.cells_error(Xf, dom, per_ref, truth, fan)
    keep = cv.neighbours(Xf) >= nb if nb is not None else np.ones(len(Xf), bool)
    return {"Xf": Xf, "nref": nref, "dom": dom, "right": right, "err": err, "keep": keep}


def sel(mp):
    k = mp["keep"]
    return {key: (v[k] if isinstance(v, np.ndarray) and len(v) == len(k) else v) for key, v in mp.items()}


class RGB:
    """Each reference's image as linear RGB on the matcher's scale, loaded once."""

    def __init__(self, rgb_of):
        self.f, self.c = rgb_of, {}

    def __call__(self, j):
        if j not in self.c:
            self.c[j] = self.f(j)
        return self.c[j]


def colours(mp, kind, sc, per_ref, rgb, fan):
    if kind == "match":
        return d3.scene_rgb(mp["Xf"], mp["dom"], per_ref, rgb, fan)
    return d2.colours_for(mp["Xf"], mp["err"], "error", sc)


def frame(sc, looks, m, shown, cols, hist, total_s, seg_s, fan, titles, kind):
    im = Image.new("RGB", (W, H), dm.PAPER)
    im.paste(d2.pano(looks, m, shown["real"]["Xf"], cols["real"], fan,
                     title="The current look (360° × 180°), with the coverage map's cells"), d2.PANO_BOX[:2])
    im.paste(d2.prog(hist, total_s, seg_s=seg_s), d2.PROG_BOX[:2])
    for side, box in (("perfect", d2.LEFT_BOX), ("real", d2.RIGHT_BOX)):
        X, c = shown[side]["Xf"], cols[side]
        keep = d2.subset(len(X))
        im.paste(d2.cloud(sc, X[keep], c[keep], -115.0, box[2:], titles[side], m=m), box[:2])
    d = ImageDraw.Draw(im)
    d.line([(0, 480), (W, 480)], fill=dm.RULE, width=1)
    d.line([(960, 0), (960, H)], fill=dm.RULE, width=1)
    t = sc.T[m]
    key = ("colour: the scene's RGB at each cell" if kind == "match" else
           f"colour: relative error along the ray, light 0 → dark ≥ {100 * d2.ERR_MAX:g} %")
    dm._text(d, (976, 34), f"{dm.SEG_NAMES[t[4]]} · {t[3]:.2f} m · look {m + 1} of {len(sc.T)} · {key}", 14, dm.MUTED)
    return im


def run(T, looks128, looks1024, truth, vis_fn, fan, refs, rgb_of, out, ply_dir, checks_v2, cache=None, rec_d2=None,
        limit=None, best=False, videos=True, log=print, name="classroom_tour"):
    t0 = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_ref = {j: st.b_reference(looks128, truth, j, fan)["m"] for j in refs}                # the judge (stage 11's)
    log(f"[demo5] the judge's windows ({time.perf_counter() - t0:.0f} s)")
    perfect = perfect_points(looks128, truth, vis_fn, fan, refs)
    log(f"[demo5] the perfect matcher ({time.perf_counter() - t0:.0f} s)")
    real, mirrored, paths_used = real_points(looks1024, truth, fan, refs, cache, log=log)
    uc, rep, _ = asx.surface(per_ref, truth)
    allP = np.concatenate([perfect[j] for j in refs])
    sc = dm.Scene(allP[d2.subset(len(allP), 200000)], T, fan)
    total_s = sc.s[-1] + 0.02
    seg_s = next((t[3] for t in T if t[4] > 0), None)
    rgb = RGB(rgb_of)
    K = len(T) if not limit else min(limit, len(T))
    encs = {k: dm._Enc(Path(ply_dir) / f"{name}_demo5_{k}.mp4", best) for k in ("match", "error")} if videos else {}
    if videos:
        Path(ply_dir).mkdir(parents=True, exist_ok=True)
    log_rel, hist, cur, last = [], [], None, {}
    maps = {"perfect": fuse_map(perfect, [], per_ref, truth, fan), "real": fuse_map(real, [], per_ref, truth, fan, MIN_NB)}
    shown = {s: sel(m_) for s, m_ in maps.items()}
    cols = {k: {s: np.zeros((0, 3), np.uint8) for s in d2.SIDES} for k in encs}
    hc = {s: (0.0, None) for s in d2.SIDES}
    titles = {}
    for m in range(K):
        rel = d2.released_at(m, refs)
        log_rel.append((m, list(rel)))
        if rel != cur:
            cur = list(rel)
            maps = {"perfect": fuse_map(perfect, rel, per_ref, truth, fan), "real": fuse_map(real, rel, per_ref, truth, fan, MIN_NB)}
            shown = {s: sel(m_) for s, m_ in maps.items()}
            for s in d2.SIDES:
                r_ = shown[s]["right"]
                hc[s] = (float(asx.covered(rep, shown[s]["Xf"][r_]).mean()) if len(r_) else 0.0, float(r_.mean()) if len(r_) else None)
            cols = {k: {s: colours(shown[s], k, sc, per_ref, rgb, fan) if len(shown[s]["Xf"]) else np.zeros((0, 3), np.uint8)
                        for s in d2.SIDES} for k in encs}
            log(f"[demo5] look {m + 1}: {len(rel)} references, perfect {len(shown['perfect']['Xf']):,} cells, "
                f"map {len(shown['real']['Xf']):,} ({time.perf_counter() - t0:.0f} s)")
        hist.append((T[m][3], dict(hc)))
        for s in d2.SIDES:
            n = len(shown[s]["Xf"])
            titles[s] = ((f"Perfect matcher (oracle, 0.5 px): " if s == "perfect" else
                          "N's coverage map (±0.5 and ±0.2 m windows, ≥ 3 neighbours): ")
                         + f"{n:,} cells" + (f", {100 * hc[s][1]:.1f} % right" if hc[s][1] is not None else ""))
        for k in encs:
            im = frame(sc, looks128, m, shown, cols[k], hist, total_s, seg_s, fan, titles, k)
            encs[k].put(im, int(dm.HOLD_START * FPS) if m == 0 else 1)
            last[k] = im
    for k, im in last.items():
        im.save(out / f"last_{k}{'' if T[0][0] is not None else '_dry'}.png")
        encs[k].put(im, int(dm.HOLD_END * FPS))
        if not limit:
            n = int(dm.ORBIT_S * FPS)
            azs = tuple(-115.0 + 360.0 * i / n for i in range(n))
            for i in range(n):
                card = Image.new("RGB", (W, H), dm.PAPER)
                for s, x0 in (("perfect", 0), ("real", 960)):
                    X, c = shown[s]["Xf"], cols[k][s]
                    keep = d2.subset(len(X))
                    card.paste(d2.cloud(sc, X[keep], c[keep], azs[i], (960, H), titles[s], m=len(T) - 1, azs=azs), (x0, 0))
                encs[k].put(card)
        encs[k].close()
    full = K == len(T)
    checks = []
    # V0: the perfect side is D2's
    if rec_d2 is not None:
        e = rec_d2["end"]["perfect"]
        pf = shown["perfect"]
        prev = fuse_map(perfect, refs[:-1], per_ref, truth, fan)
        checks.append({"name": "V0: the perfect side's end map equals demo D2's (cells, cells right)",
                       "ok": bool(full and len(pf["Xf"]) == e["cells"] and abs(float(pf["right"].mean()) - e["cells_right"]) <= 1e-12),
                       "end": [len(pf["Xf"]), float(pf["right"].mean()) if len(pf["Xf"]) else None],
                       "demo2": [e["cells"], e["cells_right"]], "control": "the map without the last reference",
                       "control_fails": len(prev["Xf"]) != e["cells"], "control_value": len(prev["Xf"])})
    # V2: the tour
    (ok2, n2), (ok2c, n2c) = checks_v2
    checks.append({"name": "V2: the tour is the planned L-curve sampled evenly", "ok": bool(ok2), **n2,
                   "control": "the turn's looks shifted 1 cm sideways", "control_fails": not ok2c})
    # V3: batch fusion
    v3, v3c = {}, {}
    for s, pts, nb in (("perfect", perfect, None), ("real", real, MIN_NB)):
        b = sel(fuse_map(pts, refs, per_ref, truth, fan, nb))
        c_ = sel(fuse_map(pts, refs[:-1], per_ref, truth, fan, nb))
        X = shown[s]["Xf"]
        v3[s] = bool(full and len(X) == len(b["Xf"]) and (len(X) == 0 or float(np.max(np.abs(X - b["Xf"]))) <= 1e-12))
        v3c[s] = bool(len(X) == len(c_["Xf"]) and (len(X) == 0 or float(np.max(np.abs(X - c_["Xf"]))) <= 1e-12))
    checks.append({"name": "V3: both end maps equal their batch fusion (the map with its neighbour filter)",
                   "ok": all(v3.values()), "control": "the batch without the last reference", "control_fails": not any(v3c.values())})
    ok4, n4 = d2.check_release(log_rel)
    ok4c, n4c = d2.check_release([(t, d2.released_at(t, refs, lag=0)) for t, _ in log_rel])
    checks.append({"name": "V4: nothing shown before its window is complete (m + 50 <= t)", "ok": bool(ok4), "violations": n4,
                   "control": "a schedule releasing at m", "control_fails": not ok4c, "control_value": n4c})
    rm = shown["real"]
    e10, e05 = rm["err"] <= 0.10, rm["err"] <= 0.05
    checks.append({"name": "C: the error video's colours (<= 10 %) equal stage 9's cells right on the map",
                   "ok": bool(np.array_equal(e10, rm["right"])), "control": "5 % in place of 10 %",
                   "control_fails": not np.array_equal(e05, rm["right"])})
    # PLY: the perfect map, the coverage map, and every cell N reported (with the 'kept' flag)
    from . import ply
    Path(ply_dir).mkdir(parents=True, exist_ok=True)
    files = {}
    allm = maps["real"]
    prev_r = fuse_map(real, refs[:-1], per_ref, truth, fan, MIN_NB)
    prev_p = fuse_map(perfect, refs[:-1], per_ref, truth, fan)
    for which, mp, keep, prev in (("perfect", maps["perfect"], np.ones(len(maps["perfect"]["Xf"]), bool), prev_p),
                                  ("map", allm, allm["keep"], prev_r), ("all", allm, np.ones(len(allm["Xf"]), bool), prev_r)):
        X = mp["Xf"][keep]
        colr = d3.scene_rgb(X, mp["dom"][keep], per_ref, rgb, fan)
        ex = {"right": mp["right"][keep].astype(np.uint8), "err": np.where(np.isfinite(mp["err"][keep]), mp["err"][keep], -1).astype(np.float32),
              "nref": np.minimum(mp["nref"][keep], 255).astype(np.uint8)}
        if which != "perfect":
            ex["kept"] = mp["keep"][keep].astype(np.uint8)
        path = Path(ply_dir) / f"{name}_{which}.ply"
        size = ply.write_ply(path, X, colr, extras=ex, comments=(
            f"deep-motion demo D5 (#35): the Classroom L-tour, {len(refs)} references; {which}",
            "world metres, +Z up; right = stage 9's metric" + ("; kept = in the coverage map (>= 3 neighbours)" if which != "perfect" else "")))
        back = ply.read_ply(path)
        Xb = np.stack([back["x"], back["y"], back["z"]], 1)
        same = len(back) == len(X) and np.array_equal(Xb, X.astype(np.float32)) and np.array_equal(back["right"].astype(bool), mp["right"][keep])
        pk = prev["keep"] if which == "map" else np.ones(len(prev["Xf"]), bool)
        ctl = len(back) == int(pk.sum()) and np.array_equal(Xb, prev["Xf"][pk].astype(np.float32))
        checks.append({"name": f"P1 {which}: the file read back holds exactly its map", "ok": bool(same),
                       "control": "the map without the last reference", "control_fails": not ctl})
        files[which] = {"path": str(path), "points": int(len(X)), "mb": round(size / 1e6, 2)}
    # the standard control
    mm = sel(fuse_map(mirrored, refs, per_ref, truth, fan, MIN_NB))
    nm, rmm = len(mm["Xf"]), float(mm["right"].mean()) if len(mm["Xf"]) else None
    checks.append({"name": "the standard control: N mirrored must not give a usable map (>= 1,000 cells at >= 97 % right)",
                   "ok": None, "control": "N with the mirrored path's poses",
                   "control_fails": not (nm >= sd.GATE_MIN_CELLS and rmm is not None and rmm >= FLOOR), "control_value": [nm, rmm]})
    # the validation: the map by the segment of each cell's dominant reference
    pf_refs = sorted(j for j, u in paths_used.items() if u["long"] == "per-frame")     # windows that leave the line
    turn = np.isin(rm["dom"], pf_refs)
    seg = {}
    for lab, msk in (("references whose window is on one line", ~turn),
                     ("references whose window leaves the line (the turn's arc)", turn)):
        n_ = int(msk.sum())
        seg[lab] = {"cells": n_, "right": float(rm["right"][msk].mean()) if n_ else None,
                    "gross": int((~(rm["err"][msk] <= 0.25)).sum())}
    vt = seg["references whose window leaves the line (the turn's arc)"]
    validation = {"question": "on the cells whose dominant reference's window leaves the line (the turn's arc, N's "
                              "per-frame path), is the map >= 97 % right?",
                  "per_frame_references": pf_refs,
                  "answer": None if vt["right"] is None else ("yes" if vt["right"] >= FLOOR else "no"), "segments": seg,
                  "by_sector": sd.gate(rm["right"], _cell_psi(rm, looks128))}
    end = {s: {"cells": len(shown[s]["Xf"]), "right": float(shown[s]["right"].mean()) if len(shown[s]["Xf"]) else None,
               "coverage": hist[-1][1][s][0] if hist else None,
               "gross": int((~(shown[s]["err"] <= 0.25)).sum())} for s in d2.SIDES}
    end["every cell N reported"] = {"cells": len(allm["Xf"]), "right": float(allm["right"].mean()) if len(allm["Xf"]) else None,
                                    "coverage": float(asx.covered(rep, allm["Xf"][allm["right"]]).mean()) if len(allm["Xf"]) else 0.0}
    return {"command": "demo5", "demo": "D5", "looks": K, "tour_looks": len(T), "references": refs, "lag_looks": LAG,
            "map": {"source": cv.S_NS, "rays_deg": [1, 179], "filter": f"neighbours >= {MIN_NB}", "floor": FLOOR},
            "n_paths": paths_used, "end": end, "validation": validation,
            "progress": [(s_, {k: list(v) for k, v in h_.items()}) for s_, h_ in hist],
            "files": files, "frames": {k: e.n for k, e in encs.items()}, "fps": FPS, "checks": checks,
            "seconds": round(time.perf_counter() - t0, 1)}


def _cell_psi(mp, looks):
    from . import confirm as cf
    return cf.cell_psi(mp["Xf"], mp["dom"], looks)
