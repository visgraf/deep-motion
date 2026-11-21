"""Stage 11 (#26): why the real matcher falls short, on the straight path alone. The ablation, its scoring and checks.

Every row is scored through stage 9's fusion and metric, unchanged (e2e.fuse, e2e.cells_right, assess.covered):
2 cm world cells keyed by the estimated point, a cell right by K10's surface metric from its dominant reference, and
coverage the share of the surface's 2 cm cells within 2 cm of a right cell. One truth for every row: the 128-spp
run's (stage 9's), the panorama references' Position.

Rows (references m = 50, 60, ..., 160 of R3's straight_long; N's frames every 10th look, five each side):
- perfect: stage 9's oracle fit, with its seeded draws in stage 9's order;
- B128, B1024: B with its mark on its textured anchors over the +/-50-look window (stage 9's);
- N128, N1024: N (deep/ntrace.py);
- M4@1024, M4@128: the frozen M4 both5 on the Foveated classroom_dolly, frames 28-39, each judged from the panorama
  reference nearest it (<= 3 cm; within 2 cm on R3, #26) with the same truth and code.
Variants at 1024 spp (reported): N on B's anchors; B's cost on N's pixels and frames; B on N's 11 looks; N with
+/-0.2 m windows; and N given the mirrored path's poses (C3's control).

Fields: the full field (stage 9's); the shared field psi 1-44 deg from the heading (M4's band), and its bands 1-18 and
18-44 deg. A band keeps the anchors whose ray is in it and the surface cells of the true points seen in it.

Checks (#26, as accepted on 2026-11-21): C0 (reproduction), C2 (the sensor and the coordinates), C3 (the cost); C1 is
deep/nfixtures.py's.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import assess as asx
from . import e2e
from . import match as mt
from . import ntrace as nt

STEP_LOOKS = 10
WIN = [i for i in range(-5, 6) if i]
WIN_SHORT = [-2, -1, 1, 2]
BANDS = {"full": None, "shared 1-44": (1.0, 44.0), "1-18": (1.0, 18.0), "18-44": (18.0, 44.0)}
C2_POINTS, C2_FACTOR = 0.03, 1.25                    # accepted on #26, 2026-11-21
C3_FACTOR, C3_POINTS, C3_CONTROL = 2.0, 0.02, 0.50
M4_MATCH_M = 0.03


# ------------------------------------------------------------------------------------------------------ helpers

def _psi_of(c, R, X):
    h = -R[:, 2]
    d = X - c
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(d @ h, -1, 1)))


def _pix_psi(R, px, H, W):
    lon = (px[:, 1] + 0.5) / W * 2 * np.pi - np.pi
    lat = (0.5 - (px[:, 0] + 0.5) / H) * np.pi
    u = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1) @ R.T
    return np.degrees(np.arccos(np.clip(u @ (-R[:, 2]), -1, 1)))


def _nb_right(c, P, idx, px, r_est):
    """K10's surface metric for per-anchor estimates (mt.neighbourhood_ranges + mt.surface_right)."""
    looks = [(0.0, c, None, None, idx)]
    return mt.surface_right(r_est, mt.neighbourhood_ranges(looks, 0, P, px))


def _window(looks, truth, j, offsets):
    """The looks of j's window (every STEP_LOOKS-th), the reference in the middle carrying the truth's idx."""
    ks = [j + STEP_LOOKS * i for i in offsets if i < 0] + [j] + [j + STEP_LOOKS * i for i in offsets if i > 0]
    win = [looks[k] for k in ks]
    m = sum(1 for i in offsets if i < 0)
    s, c, R, img, _ = win[m]
    win[m] = (s, c, R, img, truth[j][1])
    return win, m


# -------------------------------------------------------------------------------------------------- the matchers

def b_reference(looks, truth, j, fan, mirror=False):
    """B with its mark over the full +/-50-look window: stage 9's (e2e.match_reference, uncapped)."""
    m = e2e.match_reference(looks, truth, j, fan, max_n=10 ** 9, mirror=mirror)
    mk = m["mark"]
    return {"j": j, "X": m["c"] + m["u"][mk] * m["r"][mk][:, None], "psi": m["psi"][mk], "tried": int(len(mk)),
            "accepted": int(mk.sum()), "right_acc": int(m["right"][mk].sum()), "px_tried": m["px"], "m": m}


def b_short(looks, truth, j, fan, all_pixels=False):
    """B's sweep and mark on N's 11 frames (coarse sweep on every one of them). all_pixels: on every pixel N tries
    (B's cost on N's pixels and frames); otherwise on B's own anchors (B on 11 looks)."""
    win, m = _window(looks, truth, j, WIN)
    W = fan.width
    if all_pixels:
        px, u, pidx, _ = mt.anchors(win, m, fan, max_n=10 ** 9, frac=1.0)
        psi = _pix_psi(win[m][2], px, W // 2, W)
        keep = (psi >= nt.SETTINGS["psi_anchor_deg"][0]) & (psi <= nt.SETTINGS["psi_anchor_deg"][1])
        px, u, pidx = px[keep], u[keep], pidx[keep]
    else:
        px, u, pidx, _ = mt.anchors(win, m, fan, max_n=10 ** 9)
    r, conf, best = mt.sweep(win, m, px, u, fan, coarse_every=1)
    mk = mt.reliable(best, conf)
    c, R = win[m][1], win[m][2]
    right = _nb_right(c, truth[j][0], truth[j][1], px, r)
    return {"j": j, "X": c + u[mk] * r[mk][:, None], "psi": _pix_psi(R, px[mk], W // 2, W), "tried": int(len(px)),
            "accepted": int(mk.sum()), "right_acc": int(right[mk].sum())}


def n_reference(looks, truth, j, fan, offsets=WIN, mirror=False, dev=None, keep_all=False, keep_edge=False, general=None):
    """N at reference j over the frames j + 10 i, i in offsets; mirror: frame k given look 2j - k's pose. keep_all:
    also every anchor's pixel and rejection reason (stage 11b's reasons by class). keep_edge: also, for every anchor,
    its ray, zeta, reason before the depth-edge test and both deviations (stage 11c's variants)."""
    s, c, R, img, _ = looks[j]
    ks = [j + STEP_LOOKS * i for i in offsets]
    poses = [(looks[2 * j - k][1], looks[2 * j - k][2]) if mirror else (looks[k][1], looks[k][2]) for k in ks]
    P, idx = truth[j]
    r = nt.match_reference(img, c, R, [looks[k][3] for k in ks], poses, fan.width, mask=idx >= 0, dev=dev, general=general)
    acc = (r["reason"] == 0) & (r["zeta"] > 0)
    rng = np.where(r["zeta"] > 0, 1.0 / np.maximum(r["zeta"], 1e-12), np.inf)
    right = _nb_right(c, P, idx, r["px"], rng)
    o = {"j": j, "X": c + r["u"][acc] * rng[acc][:, None], "psi": np.degrees(r["psi"][acc]), "tried": int(len(acc)),
         "accepted": int(acc.sum()), "right_acc": int(right[acc].sum()), "px_acc": r["px"][acc],
         "reasons": {nt.REASONS[k]: int((r["reason"] == k).sum()) for k in range(len(nt.REASONS))}}
    if keep_all or keep_edge:
        o.update({"px_all": r["px"].astype(np.int16), "reason_all": r["reason"].astype(np.int8)})
    if keep_edge:
        o.update({"u_all": r["u"], "zeta_all": r["zeta"], "psi_all": r["psi"],
                  "reason_pre_edge": r["reason_pre_edge"].astype(np.int8),
                  "edge_dev_px": r["edge_dev_px"].astype(np.float32), "edge_dev_rel": r["edge_dev_rel"].astype(np.float32)})
    return o


def perfect_points(looks, truth, fan, refs, vis_fn, seed=22):
    """Stage 9's perfect matcher with stage 9's draws: assess.assess_path's calls, in its order (the mirrored fit after
    each reference's own consumes the same generator)."""
    rng = np.random.default_rng(seed)
    pts = {j: asx.oracle_points(truth, j, looks, fan) for j in refs}
    vis = asx.visibility(looks, refs, pts, vis_fn)
    out = {}
    for j in refs:
        X, us = asx.oracle_fit(looks, j, pts[j], vis[j], rng)
        asx.oracle_fit(looks, j, pts[j], vis[j], rng, mirror=True)
        out[j] = {"j": j, "X": X[us], "psi": _psi_of(looks[j][1], looks[j][2], X[us]), "tried": int(len(us)),
                  "accepted": int(us.sum()), "right_acc": None}
    return out


def m4_points(result_dir, run_dir, refs=None):
    """The frozen M4's accepted anchors from an m4_pipeline result (`<result_dir>/both5/r*.npz`) on its run, as world
    points: range 1 / (zeta cos e) along the sample's world direction (motion/demo_steps.anchors, D1's rule)."""
    from motion import analysis
    run = analysis.load_run(Path(run_dir))
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    out = []
    for f in sorted((Path(result_dir) / "both5").glob("r*.npz")):
        z = dict(np.load(f))
        r = int(z["r"])
        if refs is not None and r not in refs:
            continue
        rec, R, c, _ = analysis.load_look(run, es[r])
        ok = (z["reason"] == 0) & (z["sample"] >= 0) & (z["zeta"] > 0)
        d = rec.direction[z["sample"][ok]].astype(np.float64)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rng = 1.0 / (z["zeta"][ok] * np.cos(z["e"][ok]))
        out.append({"r": r, "c": np.asarray(c, float), "R": np.asarray(R, float), "X": c + d * rng[:, None],
                    "psi": np.degrees(z["e"][ok]), "tried": int(len(z["reason"])), "accepted": int(ok.sum()),
                    "right_acc": None, "file": f.name})
    return out


def assign_m4(m4, looks, refs):
    """Each M4 reference to the panorama reference nearest it; refuses beyond M4_MATCH_M."""
    out = {}
    for o in m4:
        d = {j: float(np.linalg.norm(o["c"] - looks[j][1])) for j in refs}
        j = min(d, key=d.get)
        if d[j] > M4_MATCH_M:
            continue
        if j in out:
            raise ValueError(f"[stage11] two M4 references ({out[j]['r']}, {o['r']}) map to panorama reference {j}")
        out[j] = dict(o, j=j, offset_m=d[j])
    return out


# ------------------------------------------------------------------------------------------------------ scoring

def surface_band(per_ref, truth, looks, band):
    """assess.surface's cells (every 2nd pixel, |elevation| < 70 deg), from the true points whose reference pixel is in
    the band; band None is assess.surface exactly."""
    pts = []
    for j in per_ref:
        P, idx = truth[j]
        H, W = idx.shape
        rows = np.arange(0, H, 2)
        lat = (0.5 - (rows + 0.5) / H) * np.pi
        rows = rows[np.abs(lat) < e2e.LAT_MAX]
        sub = idx[rows][:, ::2]
        q = sub.ravel()
        keep = q >= 0
        if band is not None:
            rr, cc = np.meshgrid(rows, np.arange(0, W, 2), indexing="ij")
            psi = _pix_psi(looks[j][2], np.stack([rr.ravel(), cc.ravel()], -1), H, W)
            keep &= (psi >= band[0]) & (psi <= band[1])
        pts.append(P[q[keep]])
    S = np.concatenate(pts)
    S = S[np.all(np.isfinite(S), 1)]
    code = e2e._encode(e2e._keys(S))
    uc, inv = np.unique(code, return_inverse=True)
    rep = np.zeros((len(uc), 3))
    np.add.at(rep, inv, S)
    rep /= np.bincount(inv)[:, None]
    return uc, rep


def score(row, per_ref, truth, fan, surf, band, perfect_cov=None):
    """A row ({j: {X, psi, tried, accepted, right_acc}}) in a band: fused cells, right, coverage, of the perfect's."""
    Xs, refs_of = [], []
    tried = acc = racc = 0
    have_r = True
    for j, o in row.items():
        sel = np.ones(len(o["psi"]), bool) if band is None else (o["psi"] >= band[0]) & (o["psi"] <= band[1])
        Xs.append(o["X"][sel])
        refs_of.append(np.full(int(sel.sum()), j))
        tried += o["tried"]
        acc += o["accepted"]
        if o["right_acc"] is None:
            have_r = False
        else:
            racc += o["right_acc"]
    if sum(len(x) for x in Xs) == 0:                         # e2e.fuse needs at least one point
        Xf, nref, right, X = np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, bool), np.zeros((0, 3))
    else:
        X, ref, Xf, nref, dom, right, cell_of = asx.fuse_and_judge(Xs, refs_of, per_ref, truth, fan)
    cov = asx.covered(surf[1], Xf[right]) if len(Xf) else np.zeros(len(surf[1]), bool)
    out = {"cells": int(len(Xf)), "right": float(right.mean()) if len(Xf) else None,
           "confirmed_right": float(right[nref >= 2].mean()) if (nref >= 2).any() else None,
           "coverage": float(cov.mean()), "surface_cells": int(len(surf[1])), "points": int(len(X))}
    if band is None:
        out.update({"tried": tried, "accepted_share": acc / tried if tried else None,
                    "accepted_surface_right": (racc / acc if acc else None) if have_r else None})
    if perfect_cov is not None and perfect_cov.any():
        out["covers_of_perfect"] = float((cov & perfect_cov).sum() / perfect_cov.sum())
    return out, cov


# ---------------------------------------------------------------------------------------------------- the stage

def _cache(path, fn):
    """Per-reference outputs are cached as .npz so a long run survives an interruption (runs/, not committed)."""
    if path is not None and Path(path).exists():
        z = dict(np.load(path, allow_pickle=True))
        return z["obj"].item()
    o = fn()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in o.items() if k != "m"}
        np.savez_compressed(path, obj=np.array(slim, dtype=object))
    return o


B_ALL = "B on all pixels, N's frames, 1024"
N_ALL = "N1024, every anchor's reason"
N_EDGE = "N1024, every anchor kept (stage 11c)"
ROW_ALIAS = {B_ALL: "B cost on N's pixels and frames, 1024"}   # its cache's name before #26's fix a


def collect(paths, m4=None, cache=None, rows=None, log=print, dev=None, extra=()):
    """Every row's per-reference output ({name: {j: {X, psi, tried, accepted, right_acc, ...}}}) and B128's
    per-reference windows (the cells' judge). Rows are cached per reference under `cache`; B128 and the perfect
    matcher are always recomputed. extra: rows only stage 11b asks for (N_ALL)."""
    looks128, truth, fan, refs, vis_fn = paths["128"]
    looks1024 = paths["1024"][0] if paths.get("1024") else None
    cache = Path(cache) if cache else None
    safe = lambda name: "".join(ch if ch.isalnum() or ch in "-_.@" else "_" for ch in name)
    cp = lambda name, j: (cache / safe(ROW_ALIAS.get(name, name)) / f"r{j:04d}.npz") if cache else None
    want = set(rows) if rows else None
    on = lambda name: (want is None and name not in (N_ALL, N_EDGE)) or (want is not None and name in want) or name in extra
    R = {}

    def per(name, fn):
        if not on(name):
            return
        tt = time.perf_counter()
        R[name] = {j: _cache(cp(name, j), lambda j=j: fn(j)) for j in refs}
        log(f"[stage11] {name}: {sum(o['accepted'] for o in R[name].values())} accepted of "
            f"{sum(o['tried'] for o in R[name].values())} ({time.perf_counter() - tt:.0f} s)")

    # B128 first: its per-reference windows are the cells' judge (e2e.cells_right needs c, R and the window)
    b128 = {j: b_reference(looks128, truth, j, fan) for j in refs}
    per_ref = {j: o["m"] for j, o in b128.items()}
    R["B128"] = {j: {k: v for k, v in o.items() if k != "m"} for j, o in b128.items()}
    log(f"[stage11] B128: {sum(o['accepted'] for o in R['B128'].values())} accepted")
    if on("perfect") or "perfect" in extra:
        tt = time.perf_counter()
        R["perfect"] = perfect_points(looks128, truth, fan, refs, vis_fn)
        log(f"[stage11] perfect ({time.perf_counter() - tt:.0f} s)")
    per("N128", lambda j: n_reference(looks128, truth, j, fan, dev=dev))
    if looks1024 is not None:
        per("B1024", lambda j: {k: v for k, v in b_reference(looks1024, truth, j, fan).items() if k != "m"})
        per("N1024", lambda j: n_reference(looks1024, truth, j, fan, dev=dev))
        per("N1024 mirrored (C3 control)", lambda j: n_reference(looks1024, truth, j, fan, mirror=True, dev=dev))
        per("N1024 +/-0.2 m", lambda j: n_reference(looks1024, truth, j, fan, offsets=WIN_SHORT, dev=dev))
        per("B1024 on 11 looks", lambda j: b_short(looks1024, truth, j, fan))
        per(B_ALL, lambda j: b_short(looks1024, truth, j, fan, all_pixels=True))
        per(N_ALL, lambda j: n_reference(looks1024, truth, j, fan, dev=dev, keep_all=True))
        per(N_EDGE, lambda j: n_reference(looks1024, truth, j, fan, dev=dev, keep_edge=True))
        if "N1024" in R and "B1024" in R and on("N1024 on B's anchors"):
            sub = {}
            for j in refs:
                n, b = R["N1024"][j], R["B1024"][j]
                H, W = truth[j][1].shape
                code_b = b["px_tried"][:, 0].astype(np.int64) * W + b["px_tried"][:, 1]
                sel = np.isin(n["px_acc"][:, 0].astype(np.int64) * W + n["px_acc"][:, 1], code_b)
                sub[j] = {"j": j, "X": n["X"][sel], "psi": n["psi"][sel], "tried": int(len(b["px_tried"])),
                          "accepted": int(sel.sum()), "right_acc": None}
            R["N1024 on B's anchors"] = sub
    for spp, lst in (m4 or {}).items():
        a = assign_m4(lst, looks128, refs)
        R[f"M4@{spp}"] = a
        log(f"[stage11] M4@{spp}: references {sorted((o['r'], j, round(o['offset_m'], 4)) for j, o in a.items())}")
    return R, per_ref


def run(paths, m4=None, cache=None, rows=None, assess_record=None, log=print, dev=None):
    """paths: {"128": (looks, truth, fan, refs, vis_fn), "1024": (looks, truth, fan, refs, vis_fn) or None}; the truth
    and visibility used are the 128 run's. m4: {"1024": [m4 refs], "128": [...]} (m4_points output). rows: names to
    compute (default all). assess_record: stage 9's assess.json (C0). Returns the record."""
    t0 = time.perf_counter()
    looks128, truth, fan, refs, vis_fn = paths["128"]
    R, per_ref = collect(paths, m4, cache, rows, log, dev)
    # scoring
    out = {"references": refs, "rows": {}, "bands": {k: v for k, v in BANDS.items()}}
    surfs = {b: surface_band(per_ref, truth, looks128, rng) for b, rng in BANDS.items()}
    pcov = {}
    if "perfect" in R:
        for b, rng in BANDS.items():
            s, pcov[b] = score(R["perfect"], per_ref, truth, fan, surfs[b], rng)
            out["rows"].setdefault("perfect", {})[b] = s
    for name, row in R.items():
        if name == "perfect":
            continue
        if name.startswith("M4"):
            row = {j: o for j, o in row.items()}
            bands = [b for b in BANDS if b != "full"]
        else:
            bands = list(BANDS)
        for b in bands:
            s, _ = score(row, per_ref, truth, fan, surfs[b], BANDS[b], pcov.get(b))
            out["rows"].setdefault(name, {})[b] = s
        if name.startswith("N") and name in R and "reasons" in next(iter(R[name].values())):
            tot = {}
            for o in R[name].values():
                for k, v in o["reasons"].items():
                    tot[k] = tot.get(k, 0) + v
            out["rows"][name]["reasons"] = tot
    out["checks"] = checks(out["rows"], assess_record)
    out["gap"] = gap(out["rows"])
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def checks(rows, assess_record=None):
    """C0, C2 and C3 as accepted on #26 (C1 is deep/nfixtures.py's)."""
    res = []
    g = lambda name, band, k: rows.get(name, {}).get(band, {}).get(k)
    if assess_record is not None and "B128" in rows and "perfect" in rows:
        st = assess_record["numbers"]["straight"]
        want = {"B right": st["real"]["right"], "B coverage": st["real"]["coverage"],
                "perfect right": st["perfect"]["right"], "perfect coverage": st["perfect"]["coverage"]}
        got = {"B right": g("B128", "full", "right"), "B coverage": g("B128", "full", "coverage"),
               "perfect right": g("perfect", "full", "right"), "perfect coverage": g("perfect", "full", "coverage")}
        ok = all(got[k] is not None and abs(got[k] - want[k]) <= 1e-12 for k in want)
        ctl = None
        if "B1024" in rows:
            ctl = {"B right": g("B1024", "full", "right"), "B coverage": g("B1024", "full", "coverage")}
            ctl_fails = any(abs(ctl[k] - want[k]) > 1e-9 for k in ctl)
        else:
            ctl_fails = None
        res.append({"check": "C0 B128 and the perfect matcher reproduce stage 9's straight-path numbers exactly",
                    "ok": ok, "want": want, "got": got,
                    "control": "B on the 1024-spp renders must not reproduce them", "control_fails": ctl_fails,
                    "control_values": ctl})

    def c2(name):
        n_r, n_c = g(name, "18-44", "right"), g(name, "18-44", "coverage")
        m_r, m_c = g("M4@1024", "18-44", "right"), g("M4@1024", "18-44", "coverage")
        if None in (n_r, n_c, m_r, m_c):
            return None, {}
        return (n_r >= m_r - C2_POINTS) and (n_c >= m_c / C2_FACTOR), {"right": n_r, "coverage": n_c,
                                                                      "M4 right": m_r, "M4 coverage": m_c}
    ok, v = c2("N1024")
    ctl, vc = c2("B1024")
    if ok is not None:
        res.append({"check": "C2 18-44 deg, 1024 spp: N's cells right >= M4's - 3 points and N's coverage >= M4's / 1.25",
                    "ok": ok, "values": v, "control": "B1024 in place of N must fail",
                    "control_fails": (not ctl) if ctl is not None else None, "control_values": vc})
    n_r, n_c = g("N1024", "full", "right"), g("N1024", "full", "coverage")
    b_r, b_c = g("B1024", "full", "right"), g("B1024", "full", "coverage")
    if None not in (n_r, n_c, b_r, b_c):
        mr = g("N1024 mirrored (C3 control)", "full", "right")
        res.append({"check": "C3 full field, 1024 spp: N's coverage >= 2 x B's and N's cells right >= B's - 2 points",
                    "ok": (n_c >= C3_FACTOR * b_c) and (n_r >= b_r - C3_POINTS),
                    "values": {"N right": n_r, "N coverage": n_c, "B right": b_r, "B coverage": b_c},
                    "control": "N given the mirrored path's poses must fall below 50 % right",
                    "control_fails": (mr is None) or (mr < C3_CONTROL), "control_values": {"right": mr}})
    return res


def gap(rows):
    """The gap from B128 to the perfect matcher on the full field, attributed in the declared order (coverage), and
    the sensor in the shared field (N1024 against M4@1024). Reported, not gated."""
    cov = lambda name, b="full": rows.get(name, {}).get(b, {}).get("coverage")
    chain = ["B128", "B1024", "N1024 on B's anchors", "N1024", "perfect"]
    steps = []
    vals = [cov(n) for n in chain]
    total = (vals[-1] - vals[0]) if None not in (vals[0], vals[-1]) else None
    labels = ["noise (B128 -> B1024)", "the cost on the same pixels (B1024 -> N1024 on B's anchors)",
              "the pixels tried (-> N1024)", "what remains (-> perfect)"]
    for i, lab in enumerate(labels):
        a, b = vals[i], vals[i + 1]
        if None in (a, b):
            steps.append({"step": lab, "gain": None})
        else:
            steps.append({"step": lab, "from": a, "to": b, "gain": b - a, "share_of_gap": (b - a) / total if total else None})
    sensor = {}
    for b in ("shared 1-44", "1-18", "18-44"):
        sensor[b] = {"N1024": cov("N1024", b), "M4@1024": cov("M4@1024", b), "perfect": cov("perfect", b)}
    return {"chain": chain, "steps": steps, "total": total, "sensor_shared_field": sensor}


def table(out):
    """The decomposition table as markdown lines."""
    lines = ["| row | field | cells | right | coverage | of perfect's | accepted share | accepted surface-right |",
             "|---|---|---|---|---|---|---|---|"]
    f = lambda x, p=1: "—" if x is None else (f"{100 * x:.{p}f} %" if isinstance(x, float) else str(x))
    for name, r in out["rows"].items():
        for b, s in r.items():
            if b == "reasons":
                continue
            lines.append(f"| {name} | {b} | {s['cells']} | {f(s['right'], 2)} | {f(s['coverage'])} | "
                         f"{f(s.get('covers_of_perfect'))} | {f(s.get('accepted_share'))} | {f(s.get('accepted_surface_right'))} |")
    return lines


# ------------------------------------------------------------------------- stage 11b: the breakdown (#26, 2026-11-21)

RANGE_BINS = {"< 1 m": (0.0, 1.0), "1-2 m": (1.0, 2.0), "2-3 m": (2.0, 3.0), ">= 3 m": (3.0, 99.0)}     # stage 9's
TEXTURE_BINS = {"< 0.02": (0.0, 0.02), "0.02-0.05": (0.02, 0.05), "0.05-0.1": (0.05, 0.1), "0.1-0.2": (0.1, 0.2),
                ">= 0.2": (0.2, np.inf)}
SNR_BINS = {"< 1": (0.0, 1.0), "1-3": (1.0, 3.0), "3-10": (3.0, 10.0), ">= 10": (10.0, np.inf)}
CEILINGS = {"perfect where signal texture >= 0.02": ("texture", 0.02), "perfect where gradient SNR >= 3": ("snr", 3.0)}
SPP_RATIO = 8.0                                        # 1024 / 128; Cycles' noise scales as 1/sqrt(spp) (no adaptive
                                                       # sampling, no denoiser: fov-render bl/config.py)


def attributes(paths, refs, dev=None):
    """Per reference j, rasters (H, W): psi (deg from the heading), true range, depth edge (stage 9's: max/min true
    range over +/-2 px > 1.10), and texture along the trace by M4's measure on the 7 x 3 angular patch of N:
    - raw: rms of the half-pixel differences along the trace over the patch mean, on the 1024-spp reference;
    - signal: the same with the noise's share removed. D = I128 / g - I1024 (g the least-squares gain between the two
      runs' scalings) has variance (SPP_RATIO + 1) sigma_1024^2, so the noise's share of the mean squared difference is
      that of D over (SPP_RATIO + 1); the same operator on both images, so interpolation cancels;
    - snr: the signal's rms difference over the noise's.
    The 1024 run is the texture's; the truth is the 128 run's."""
    import torch
    looks128, truth, fan, _, _ = paths["128"]
    looks1024 = paths["1024"][0]
    dev = dev or nt.device()
    W = fan.width
    H = W // 2
    res = 2 * np.pi / W
    s = nt.SETTINGS
    a, b = nt._patch(s, torch.float64, dev)
    out = {}
    rows, cols = np.mgrid[0:H, 0:W]
    lon = (cols + 0.5) / W * 2 * np.pi - np.pi
    lat = (0.5 - (rows + 0.5) / H) * np.pi
    vloc = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1).reshape(-1, 3)
    for j in refs:
        c, R = looks128[j][1], looks128[j][2]
        P, idx = truth[j]
        u = vloc @ R.T
        pen = nt.Pencil(-R[:, 2])
        psi, om = pen.coords(u)
        rng = np.where(idx.ravel() >= 0, np.linalg.norm(P[np.maximum(idx.ravel(), 0)] - c, axis=-1), np.nan).reshape(H, W)
        mx, mn = np.zeros((H, W)), np.full((H, W), np.inf)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                sh = rng[np.clip(np.arange(H) + dr, 0, H - 1)][:, (np.arange(W) + dc) % W]
                mx, mn = np.fmax(mx, sh), np.fmin(mn, sh)
        with np.errstate(invalid="ignore", divide="ignore"):
            edge = (mx / mn) > mt.EDGE_RATIO
        I1, I8 = np.asarray(looks1024[j][3], float), np.asarray(looks128[j][3], float)
        ok = np.isfinite(I1) & np.isfinite(I8)
        g = float((I8[ok] * I1[ok]).sum() / (I1[ok] ** 2).sum())
        D = I8 / g - I1
        T = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)
        I1t, Dt = T(I1), T(D)
        ct, Rt = T(c), T(R)
        tex_raw, tex_sig, snr = (np.full(H * W, np.nan) for _ in range(3))
        ps, os_ = T(psi), T(om)
        for a0 in range(0, H * W, 40000):
            b0 = min(H * W, a0 + 40000)
            p_, o_ = ps[a0:b0], os_[a0:b0]
            U0 = nt.patch_dirs(pen, p_, o_, a, b, res)
            Up = nt.patch_dirs(pen, p_ + 0.5 * res, o_, a, b, res)
            Um = nt.patch_dirs(pen, p_ - 0.5 * res, o_, a, b, res)
            A = nt.sample_world(I1t, ct + U0, ct, Rt, W)
            gI = nt.sample_world(I1t, ct + Up, ct, Rt, W) - nt.sample_world(I1t, ct + Um, ct, Rt, W)
            gD = nt.sample_world(Dt, ct + Up, ct, Rt, W) - nt.sample_world(Dt, ct + Um, ct, Rt, W)
            m2, n2 = (gI ** 2).mean(1), (gD ** 2).mean(1) / (SPP_RATIO + 1.0)
            mean = torch.abs(A.mean(1))
            tex_raw[a0:b0] = (torch.sqrt(m2) / mean).cpu().numpy()
            sig = torch.sqrt(torch.clamp(m2 - n2, min=0.0))
            tex_sig[a0:b0] = (sig / mean).cpu().numpy()
            snr[a0:b0] = (sig / torch.sqrt(torch.clamp(n2, min=1e-30))).cpu().numpy()
        out[j] = {"psi": np.degrees(psi).reshape(H, W), "range": rng, "edge": edge, "texture_raw": tex_raw.reshape(H, W),
                  "texture": tex_sig.reshape(H, W), "snr": snr.reshape(H, W), "gain_128_over_1024": g}
    return out


def point_pixels(X, c, R, W):
    """The panorama pixel of reference (c, R) that sees each world point (row, col), nearest centre."""
    H = W // 2
    v = (X - c) @ R
    n = np.linalg.norm(v, axis=-1)
    lon = np.arctan2(v[:, 0], -v[:, 2])
    lat = np.arcsin(np.clip(v[:, 1] / np.maximum(n, 1e-12), -1, 1))
    col = np.mod(np.rint((lon / (2 * np.pi) + 0.5) * W - 0.5), W).astype(np.int64)
    row = np.clip(np.rint((0.5 - lat / np.pi) * H - 0.5), 0, H - 1).astype(np.int64)
    return np.stack([row, col], -1)


def classify(attr, j, px, kind):
    """The value of the class `kind` at reference j's pixels px."""
    A = attr[j]
    key = {"sector": "psi", "range": "range", "texture": "texture", "snr": "snr", "edge": "edge"}[kind]
    return A[key][px[:, 0], px[:, 1]]


CLASSES = {"sector": mt.SECTORS, "range": RANGE_BINS, "edge": {"at depth edges": True, "interior": False},
           "texture": TEXTURE_BINS, "snr": SNR_BINS}


def _in(v, rng):
    if isinstance(rng, bool):
        return v.astype(bool) == rng
    with np.errstate(invalid="ignore"):
        return (v >= rng[0]) & (v < rng[1])


def surface_where(per_ref, truth, attr, kind, rng):
    """assess.surface's cells from the true points (every 2nd pixel, |elevation| < 70 deg) whose reference pixel is in
    the class."""
    pts = []
    for j in per_ref:
        P, idx = truth[j]
        H, W = idx.shape
        rows = np.arange(0, H, 2)
        lat = (0.5 - (rows + 0.5) / H) * np.pi
        rows = rows[np.abs(lat) < e2e.LAT_MAX]
        rr, cc = np.meshgrid(rows, np.arange(0, W, 2), indexing="ij")
        px = np.stack([rr.ravel(), cc.ravel()], -1)
        q = idx[px[:, 0], px[:, 1]]
        keep = (q >= 0) & _in(classify(attr, j, px, kind), rng)
        pts.append(P[q[keep]])
    S = np.concatenate(pts) if pts else np.zeros((0, 3))
    S = S[np.all(np.isfinite(S), 1)]
    if len(S) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3))
    code = e2e._encode(e2e._keys(S))
    uc, inv = np.unique(code, return_inverse=True)
    rep = np.zeros((len(uc), 3))
    np.add.at(rep, inv, S)
    rep /= np.bincount(inv)[:, None]
    return uc, rep


def score_where(row, per_ref, truth, fan, surf, attr, kind, rng, looks, perfect_cov=None):
    """score() for the points whose reference pixel is in the class."""
    W = fan.width
    Xs, refs_of = [], []
    for j, o in row.items():
        if len(o["X"]) == 0:
            continue
        px = point_pixels(o["X"], looks[j][1], looks[j][2], W)
        sel = _in(classify(attr, j, px, kind), rng)
        Xs.append(o["X"][sel])
        refs_of.append(np.full(int(sel.sum()), j))
    if sum(len(x) for x in Xs) == 0:
        Xf, right = np.zeros((0, 3)), np.zeros(0, bool)
    else:
        _, _, Xf, nref, dom, right, _ = asx.fuse_and_judge(Xs, refs_of, per_ref, truth, fan)
    cov = asx.covered(surf[1], Xf[right]) if len(Xf) and len(surf[1]) else np.zeros(len(surf[1]), bool)
    out = {"cells": int(len(Xf)), "right": float(right.mean()) if len(Xf) else None,
           "coverage": float(cov.mean()) if len(cov) else None, "surface_cells": int(len(surf[1]))}
    if perfect_cov is not None and perfect_cov.any():
        out["covers_of_perfect"] = float((cov & perfect_cov).sum() / perfect_cov.sum())
        out["gap_cells"] = int((perfect_cov & ~cov).sum())
    return out, cov


def ceiling_rows(R, attr, looks, fan):
    """The perfect matcher kept only where an image could carry a match (CEILINGS)."""
    out = {}
    W = fan.width
    for name, (kind, thr) in CEILINGS.items():
        row = {}
        for j, o in R["perfect"].items():
            px = point_pixels(o["X"], looks[j][1], looks[j][2], W)
            with np.errstate(invalid="ignore"):
                sel = classify(attr, j, px, kind) >= thr
            row[j] = {"j": j, "X": o["X"][sel], "psi": o["psi"][sel], "tried": o["tried"], "accepted": int(sel.sum()),
                      "right_acc": None}
        out[name] = row
    return out


def breakdown(paths, m4=None, cache=None, assess_record=None, log=print, dev=None):
    """Stage 11b: every row of stage 11 by class (CLASSES), the two ceilings, and N1024's rejection reasons by class.
    From the cache; adds only N_ALL (N1024 re-run keeping every anchor's reason)."""
    t0 = time.perf_counter()
    looks128, truth, fan, refs, vis_fn = paths["128"]
    R, per_ref = collect(paths, m4, cache, None, log, dev, extra=(N_ALL, "perfect"))
    consistency = {}
    if assess_record is not None:
        st = assess_record["numbers"]["straight"]
        surf = surface_band(per_ref, truth, looks128, None)
        for name, key in (("B128", "real"), ("perfect", "perfect")):
            s_, _ = score(R[name], per_ref, truth, fan, surf, None)
            consistency[name] = {"right": s_["right"], "coverage": s_["coverage"],
                                 "equal_to_stage9": abs(s_["right"] - st[key]["right"]) <= 1e-12
                                 and abs(s_["coverage"] - st[key]["coverage"]) <= 1e-12}
        log(f"[stage11b] consistency with stage 9 (C0's numbers again): {consistency}")
    tt = time.perf_counter()
    attr = attributes(paths, refs, dev)
    log(f"[stage11b] attributes ({time.perf_counter() - tt:.0f} s); gains 128/1024: "
        f"{[round(attr[j]['gain_128_over_1024'], 4) for j in refs]}")
    R.update(ceiling_rows(R, attr, looks128, fan))
    reasons = R.pop(N_ALL)
    out = {"references": refs, "classes": {k: {b: (list(v) if not isinstance(v, bool) else v) for b, v in c.items()}
                                           for k, c in CLASSES.items()},
           "ceilings": {k: list(v) for k, v in CEILINGS.items()}, "consistency": consistency, "by_class": {},
           "n1024_reasons_by_class": {}, "texture_distribution": {}}
    for kind, bins in CLASSES.items():
        out["by_class"][kind] = {}
        for bname, rng in bins.items():
            surf = surface_where(per_ref, truth, attr, kind, rng)
            entry = {"surface_cells": int(len(surf[1])), "rows": {}}
            s_p, pcov = score_where(R["perfect"], per_ref, truth, fan, surf, attr, kind, rng, looks128)
            entry["rows"]["perfect"] = s_p
            for name, row in R.items():
                if name in ("perfect", "N1024 mirrored (C3 control)"):
                    continue
                s_, _ = score_where(row, per_ref, truth, fan, surf, attr, kind, rng, looks128, pcov)
                entry["rows"][name] = s_
            out["by_class"][kind][bname] = entry
        log(f"[stage11b] {kind} ({time.perf_counter() - t0:.0f} s)")
        # N1024's rejection reasons by class
        tab = {}
        for bname, rng in bins.items():
            cnt = np.zeros(len(nt.REASONS), np.int64)
            for j, o in reasons.items():
                sel = _in(classify(attr, j, o["px_all"].astype(np.int64), kind), rng)
                cnt += np.bincount(o["reason_all"][sel].astype(np.int64), minlength=len(nt.REASONS))
            tot = int(cnt.sum())
            tab[bname] = {"anchors": tot, **{nt.REASONS[k]: (int(cnt[k]), float(cnt[k] / tot) if tot else None)
                                             for k in range(len(nt.REASONS)) if cnt[k]}}
        out["n1024_reasons_by_class"][kind] = tab
    # how the surface's pixels are textured, raw against signal (the noise's share of M4's measure)
    for kind in ("texture_raw", "texture", "snr"):
        v = np.concatenate([attr[j][kind][np.abs(attr[j]["psi"]) >= 0][::7] for j in refs])
        v = v[np.isfinite(v)]
        out["texture_distribution"][kind] = {f"p{q}": float(np.quantile(v, q / 100)) for q in (10, 25, 50, 75, 90)}
    out["gains_128_over_1024"] = {str(j): attr[j]["gain_128_over_1024"] for j in refs}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def breakdown_tables(out, rows=("perfect", "perfect where signal texture >= 0.02", "perfect where gradient SNR >= 3",
                                "B1024", "N1024", B_ALL, "N1024 +/-0.2 m", "M4@1024")):
    """The breakdown as markdown: coverage (cells right) per class and row, and N1024's main rejections."""
    f = lambda x, p=1: "—" if x is None else f"{100 * x:.{p}f}"
    L = []
    for kind, bins in out["by_class"].items():
        L += [f"### by {kind}", "", "| row | " + " | ".join(f"{b} ({e['surface_cells']} cells)" for b, e in bins.items()) + " |",
              "|---|" + "---|" * len(bins)]
        for r in rows:
            cells = []
            for b, e in bins.items():
                s = e["rows"].get(r)
                cells.append("—" if s is None else f"{f(s['coverage'])} % ({f(s['right'], 1)} %)")
            L.append(f"| {r} | " + " | ".join(cells) + " |")
        L += ["", f"N1024's rejections by {kind} (share of the anchors tried):", "",
              "| reason | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
        tab = out["n1024_reasons_by_class"][kind]
        names = [n for n in nt.REASONS if any(n in tab[b] for b in bins)]
        for n in names:
            L.append(f"| {n} | " + " | ".join(f(tab[b][n][1]) + " %" if n in tab[b] else "—" for b in bins) + " |")
        L.append(f"| anchors | " + " | ".join(str(tab[b]["anchors"]) for b in bins) + " |")
        L.append("")
    return L


# ------------------------------------------------------------------------- stage 11c: the diagnosis (#26, 2026-11-21)

EDGE_REL = 0.10                     # stage 9's EDGE_RATIO - 1: the share by which stage 9 calls a range change an edge
EDGE_VARIANTS = {"N1024, depth-edge test off": "off", "N1024, depth-edge test at 10 %": "rel"}
CROSS = {"range x snr": ("range", "snr"), "sector x snr": ("sector", "snr")}
DIAG_ROWS = ("perfect", "perfect where gradient SNR >= 3", "B1024", B_ALL, "N1024", "N1024, depth-edge test off",
             "N1024, depth-edge test at 10 %", "M4@1024")


def edge_variant(o, truth, j, looks, mode):
    """N1024 at reference j with its last test, the depth edge, as `mode`: 'px' (M4's 0.5 px of the longest step: N1024
    itself), 'off', or 'rel' (a neighbour's deviation from the anchor's planar prediction over 10 % of its zeta)."""
    pre = o["reason_pre_edge"] == 0
    if mode == "px":
        acc = pre & ~(o["edge_dev_px"] > nt.SETTINGS["edge_tol_px"])
    elif mode == "rel":
        acc = pre & ~(o["edge_dev_rel"] > EDGE_REL)
    else:
        acc = pre.copy()
    acc &= o["zeta_all"] > 0
    c = looks[j][1]
    rng = 1.0 / o["zeta_all"][acc]
    px = o["px_all"][acc].astype(np.int64)
    right = _nb_right(c, truth[j][0], truth[j][1], px, rng)
    return {"j": j, "X": c + o["u_all"][acc] * rng[:, None], "psi": np.degrees(o["psi_all"][acc]),
            "tried": int(len(pre)), "accepted": int(acc.sum()), "right_acc": int(right.sum()), "px_acc": px}


def pred_of(attr, spec):
    """A predicate on (j, px) for a list of (kind, bin range) pairs: every pair must hold."""
    def f(j, px):
        m = np.ones(len(px), bool)
        for kind, rng in spec:
            m &= _in(classify(attr, j, px, kind), rng)
        return m
    return f


def surface_pred(per_ref, truth, pred):
    """surface_where for any predicate on the true points' reference pixels."""
    pts = []
    for j in per_ref:
        P, idx = truth[j]
        H, W = idx.shape
        rows = np.arange(0, H, 2)
        lat = (0.5 - (rows + 0.5) / H) * np.pi
        rows = rows[np.abs(lat) < e2e.LAT_MAX]
        rr, cc = np.meshgrid(rows, np.arange(0, W, 2), indexing="ij")
        px = np.stack([rr.ravel(), cc.ravel()], -1)
        q = idx[px[:, 0], px[:, 1]]
        keep = (q >= 0) & pred(j, px)
        pts.append(P[q[keep]])
    S = np.concatenate(pts) if pts else np.zeros((0, 3))
    S = S[np.all(np.isfinite(S), 1)]
    if len(S) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3))
    code = e2e._encode(e2e._keys(S))
    uc, inv = np.unique(code, return_inverse=True)
    rep = np.zeros((len(uc), 3))
    np.add.at(rep, inv, S)
    rep /= np.bincount(inv)[:, None]
    return uc, rep


def score_pred(row, per_ref, truth, fan, surf, pred, looks, perfect_cov=None):
    """score_where for any predicate on the points' reference pixels."""
    W = fan.width
    Xs, refs_of = [], []
    for j, o in row.items():
        if len(o["X"]) == 0:
            continue
        sel = pred(j, point_pixels(o["X"], looks[j][1], looks[j][2], W))
        Xs.append(o["X"][sel])
        refs_of.append(np.full(int(sel.sum()), j))
    if sum(len(x) for x in Xs) == 0:
        Xf, right = np.zeros((0, 3)), np.zeros(0, bool)
    else:
        _, _, Xf, nref, dom, right, _ = asx.fuse_and_judge(Xs, refs_of, per_ref, truth, fan)
    cov = asx.covered(surf[1], Xf[right]) if len(Xf) and len(surf[1]) else np.zeros(len(surf[1]), bool)
    out = {"cells": int(len(Xf)), "right": float(right.mean()) if len(Xf) else None,
           "coverage": float(cov.mean()) if len(cov) else None, "surface_cells": int(len(surf[1]))}
    if perfect_cov is not None and perfect_cov.any():
        out["covers_of_perfect"] = float((cov & perfect_cov).sum() / perfect_cov.sum())
    return out, cov


def _reasons(rows_all, attr, pred, key="reason_all"):
    cnt = np.zeros(len(nt.REASONS), np.int64)
    for j, o in rows_all.items():
        sel = pred(j, o["px_all"].astype(np.int64))
        cnt += np.bincount(o[key][sel].astype(np.int64), minlength=len(nt.REASONS))
    tot = int(cnt.sum())
    return {"anchors": tot, **{nt.REASONS[k]: (int(cnt[k]), float(cnt[k] / tot) if tot else None)
                               for k in range(len(nt.REASONS)) if cnt[k]}}


def diagnose(paths, m4=None, cache=None, assess_record=None, log=print, dev=None):
    """Stage 11c: the two depth-edge variants of N1024 (from one run keeping every anchor, the 0.5 px case rebuilding
    N1024 exactly), stage 11's full-field scores and stage 11b's classes for them, and the cross-tabulations range x
    SNR and sector x SNR of coverage, cells right and N1024's rejections. Reported; no check changes."""
    t0 = time.perf_counter()
    looks128, truth, fan, refs, vis_fn = paths["128"]
    R, per_ref = collect(paths, m4, cache, None, log, dev, extra=(N_EDGE, "perfect"))
    consistency = {}
    if assess_record is not None:
        st = assess_record["numbers"]["straight"]
        surf = surface_band(per_ref, truth, looks128, None)
        for name, key in (("B128", "real"), ("perfect", "perfect")):
            s_, _ = score(R[name], per_ref, truth, fan, surf, None)
            consistency[name] = {"right": s_["right"], "coverage": s_["coverage"],
                                 "equal_to_stage9": abs(s_["right"] - st[key]["right"]) <= 1e-12
                                 and abs(s_["coverage"] - st[key]["coverage"]) <= 1e-12}
    edges = R.pop(N_EDGE)
    rebuilt = {j: edge_variant(o, truth, j, looks128, "px") for j, o in edges.items()}
    same = all(rebuilt[j]["accepted"] == R["N1024"][j]["accepted"] and len(rebuilt[j]["X"]) == len(R["N1024"][j]["X"])
               and np.allclose(rebuilt[j]["X"], R["N1024"][j]["X"], atol=1e-9) for j in refs)
    consistency["N1024 rebuilt at 0.5 px equals the cached N1024"] = {
        "equal": bool(same), "accepted": [int(sum(o["accepted"] for o in rebuilt.values())),
                                          int(sum(o["accepted"] for o in R["N1024"].values()))]}
    log(f"[stage11c] consistency: {consistency}")
    for name, mode in EDGE_VARIANTS.items():
        R[name] = {j: edge_variant(o, truth, j, looks128, mode) for j, o in edges.items()}
        log(f"[stage11c] {name}: {sum(o['accepted'] for o in R[name].values())} accepted")
    tt = time.perf_counter()
    attr = attributes(paths, refs, dev)
    log(f"[stage11c] attributes ({time.perf_counter() - tt:.0f} s)")
    R.update(ceiling_rows(R, attr, looks128, fan))
    rows = [r for r in DIAG_ROWS if r in R]
    out = {"references": refs, "edge_rel": EDGE_REL, "variants": EDGE_VARIANTS, "consistency": consistency,
           "full": {}, "by_class": {}, "cross": {}, "n1024_reasons": {}, "variant_rejections": {}}
    # stage 11's full-field scores and bands for the variants (and N1024 beside them)
    surfs = {b: surface_band(per_ref, truth, looks128, rng) for b, rng in BANDS.items()}
    pcov = {b: score(R["perfect"], per_ref, truth, fan, surfs[b], rng)[1] for b, rng in BANDS.items()}
    for name in ["N1024"] + list(EDGE_VARIANTS):
        out["full"][name] = {b: score(R[name], per_ref, truth, fan, surfs[b], rng, pcov[b])[0] for b, rng in BANDS.items()}
    # how many anchors each tolerance rejects at a depth edge
    for name, mode in [("N1024 (0.5 px)", "px")] + list((n, m) for n, m in EDGE_VARIANTS.items()):
        tot = rej = 0
        for o in edges.values():
            pre = o["reason_pre_edge"] == 0
            tot += int(pre.sum())
            if mode == "px":
                rej += int((pre & (o["edge_dev_px"] > nt.SETTINGS["edge_tol_px"])).sum())
            elif mode == "rel":
                rej += int((pre & (o["edge_dev_rel"] > EDGE_REL)).sum())
        out["variant_rejections"][name] = {"candidates_before_the_test": tot, "rejected_by_it": rej}
    log(f"[stage11c] full field ({time.perf_counter() - t0:.0f} s)")
    # one-dimensional classes for the rows (11b's, with the variants)
    for kind in ("sector", "range", "edge", "snr"):
        out["by_class"][kind] = {}
        for bname, rng in CLASSES[kind].items():
            pred = pred_of(attr, [(kind, rng)])
            surf = surface_pred(per_ref, truth, pred)
            e = {"surface_cells": int(len(surf[1])), "rows": {}}
            sp, pc = score_pred(R["perfect"], per_ref, truth, fan, surf, pred, looks128)
            e["rows"]["perfect"] = sp
            for name in rows[1:]:
                e["rows"][name] = score_pred(R[name], per_ref, truth, fan, surf, pred, looks128, pc)[0]
            out["by_class"][kind][bname] = e
        log(f"[stage11c] {kind} ({time.perf_counter() - t0:.0f} s)")
    # cross-tabulations
    for cname, (k1, k2) in CROSS.items():
        out["cross"][cname] = {}
        out["n1024_reasons"][cname] = {}
        for b1, r1 in CLASSES[k1].items():
            for b2, r2 in CLASSES[k2].items():
                pred = pred_of(attr, [(k1, r1), (k2, r2)])
                surf = surface_pred(per_ref, truth, pred)
                key = f"{b1} | {b2}"
                e = {"surface_cells": int(len(surf[1])), "rows": {}}
                sp, pc = score_pred(R["perfect"], per_ref, truth, fan, surf, pred, looks128)
                e["rows"]["perfect"] = sp
                for name in rows[1:]:
                    e["rows"][name] = score_pred(R[name], per_ref, truth, fan, surf, pred, looks128, pc)[0]
                out["cross"][cname][key] = e
                out["n1024_reasons"][cname][key] = _reasons(edges, attr, pred)
        log(f"[stage11c] {cname} ({time.perf_counter() - t0:.0f} s)")
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def diagnose_tables(out):
    """Stage 11c as markdown."""
    f = lambda x, p=1: "—" if x is None else f"{100 * x:.{p}f}"
    L = ["### The depth-edge variants on the full field and the bands (coverage, cells right)", "",
         "| row | " + " | ".join(BANDS) + " | rejected by the test |", "|---|" + "---|" * (len(BANDS) + 1)]
    vr = out["variant_rejections"]
    vmap = {"N1024": "N1024 (0.5 px)"}
    for name, b in out["full"].items():
        r = vr.get(vmap.get(name, name), {})
        L.append(f"| {name} | " + " | ".join(f"{f(s['coverage'])} % ({f(s['right'], 2)} %)" for s in b.values())
                 + f" | {r.get('rejected_by_it', '—')} of {r.get('candidates_before_the_test', '—')} |")
    L.append("")
    for kind, bins in out["by_class"].items():
        L += [f"### by {kind} (coverage, cells right)", "",
              "| row | " + " | ".join(f"{b} ({e['surface_cells']})" for b, e in bins.items()) + " |", "|---|" + "---|" * len(bins)]
        names = list(next(iter(bins.values()))["rows"])
        for n in names:
            L.append(f"| {n} | " + " | ".join("—" if e["rows"][n] is None else
                                              f"{f(e['rows'][n]['coverage'])} % ({f(e['rows'][n]['right'], 1)} %)" for e in bins.values()) + " |")
        L.append("")
    for cname, cells in out["cross"].items():
        k1, k2 = CROSS[cname]
        b1s, b2s = list(CLASSES[k1]), list(CLASSES[k2])
        for n in ("perfect", "N1024", "N1024, depth-edge test off", "N1024, depth-edge test at 10 %", B_ALL):
            L += [f"### {cname}: {n} (coverage, cells right; surface cells)", "", f"| {k1} \\ {k2} | " + " | ".join(b2s) + " |",
                  "|---|" + "---|" * len(b2s)]
            for b1 in b1s:
                row = []
                for b2 in b2s:
                    e = cells[f"{b1} | {b2}"]
                    s_ = e["rows"].get(n)
                    row.append("—" if not s_ or s_["coverage"] is None else
                               f"{f(s_['coverage'])} % ({f(s_['right'], 1)} %; {e['surface_cells']})")
                L.append(f"| {b1} | " + " | ".join(row) + " |")
            L.append("")
        L += [f"### {cname}: N1024's rejections at NCC < 0.8 / at a depth edge / accepted (share of anchors tried)", "",
              f"| {k1} \\ {k2} | " + " | ".join(b2s) + " |", "|---|" + "---|" * len(b2s)]
        for b1 in b1s:
            row = []
            for b2 in b2s:
                t = out["n1024_reasons"][cname][f"{b1} | {b2}"]
                g = lambda k: f(t[k][1], 0) if k in t else "0"
                row.append(f"{g('low_ncc')} / {g('depth_edge')} / {g('accepted')} ({t['anchors']})")
            L.append(f"| {b1} | " + " | ".join(row) + " |")
        L.append("")
    return L


# ------------------------------------------------------------------------------- the dry run: a synthetic room

DRY_NOISE = {"1024": 0.005, "128": 0.005 * np.sqrt(8.0)}      # M4's fixture noise, and 1/sqrt(spp) for 128 spp


def synthetic_room(K=111, step=0.01, y0=-0.55, W=720, ss=2, seed=0, fov_frames=True, log=print, cache=None):
    """Cached under `cache` (runs/, not committed) when given; see _synthetic_room."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        paths, m4in = _synthetic_room(K, step, y0, W, ss, seed, fov_frames, log, imgs=z["obj"].item())
        return paths, m4in
    paths, m4in = _synthetic_room(K, step, y0, W, ss, seed, fov_frames, log)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        imgs = {"pano": {n: [l[3] for l in paths[n][0]] for n in ("128", "1024")},
                "fov": m4in["frames"] if m4in else None}
        np.savez_compressed(cache, obj=np.array(imgs, dtype=object))
    return paths, m4in


def _synthetic_room(K, step, y0, W, ss, seed, fov_frames, log, imgs=None):
    """A closed room of M4's textured fixture planes, with an occluding board, ray-cast exactly for both sensors along
    one straight line (heading +Y, M4's R_EYE): panorama looks every 1 cm (box filter ss x ss) at two noise levels
    (DRY_NOISE, standing in for 1024 and 128 spp) and Foveated frames every 10 cm at both. For the dry run only (Chat
    sandbox, not gated). Returns (paths, m4 inputs)."""
    from motion import flow_fixtures as fx
    from . import nfixtures as nf
    from . import sensor as sn
    tex = [fx._texture_noise(seed + 30 + i) for i in range(7)]
    X, Y, Z = np.eye(3)
    planes = [fx._plane([0, 0, -1.6], Z, X, tex[0], "floor"), fx._plane([0, 0, 1.4], -Z, X, tex[1], "ceiling"),
              fx._plane([-2.0, 0, 0], X, Y, tex[2], "left"), fx._plane([2.5, 0, 0], -X, Y, tex[3], "right"),
              fx._plane([0, 4.0, 0], -Y, X, tex[4], "front"), fx._plane([0, -3.5, 0], Y, X, tex[5], "back"),
              fx._plane([-0.6, 1.8, -0.2], -Y, X, tex[6], "board", bounds=(-0.4, 0.4, -0.35, 0.35))]
    R = nf.R_EYE
    fan = sn.SlitFan(width=W)
    H = W // 2
    looks = {"128": [], "1024": []}
    t0 = time.perf_counter()
    for k in range(K):
        c = np.array([0.0, y0 + step * k, 0.0])
        if imgs is not None:
            for name in DRY_NOISE:
                looks[name].append((step * k, c, R, imgs["pano"][name][k], None))
            continue
        clean = nf.render(planes, c, W=W, ss=ss, noise=0.0)
        for name, sg in DRY_NOISE.items():
            img = clean + np.random.default_rng(seed + (1000 if name == "128" else 5000) + k).normal(scale=sg, size=clean.shape)
            looks[name].append((step * k, c, R, img.astype(np.float32), None))
    log(f"[stage11 dry run] {K} panorama looks at two noise levels ({time.perf_counter() - t0:.0f} s)")
    refs = asx.references(K)
    v, u = np.mgrid[0:H, 0:W]
    lon = (u + 0.5) / W * 2 * np.pi - np.pi
    lat = (0.5 - (v + 0.5) / H) * np.pi
    d = (np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1) @ R.T).reshape(-1, 3)
    truth = {}
    L = looks["128"]
    for j in refs:
        rng, idx, _ = fx._hit(planes, L[j][1], d)
        P = L[j][1] + d * rng[:, None]
        ok = np.isfinite(rng)
        truth[j] = (np.where(ok[:, None], P, np.nan), np.where(ok, np.arange(H * W), -1).reshape(H, W))
    res = 2 * np.pi / W

    def vis_fn(k, P):
        c = L[k][1]
        dd = P - c
        r = np.linalg.norm(dd, axis=-1)
        rh, _, _ = fx._hit(planes, c, dd / r[:, None])
        return np.isfinite(rh) & (np.abs(rh - r) <= 2.0 * res * r)
    paths = {name: (looks[name], truth, fan, refs, vis_fn) for name in ("128", "1024")}
    m4in = None
    if fov_frames:
        from fov_render import sensors
        model = sensors.Foveated().model("full")
        m4in = {"model": model, "frames": {"128": {}, "1024": {}}}
        if imgs is not None and imgs.get("fov") is not None:
            m4in["frames"] = imgs["fov"]
            return paths, m4in
        for k in range(0, K, STEP_LOOKS):
            clean = fx.render(model, planes, L[k][1], noise=0.0)
            for name, sg in DRY_NOISE.items():
                m4in["frames"][name][k] = clean + np.random.default_rng(seed + (2000 if name == "128" else 6000) + k).normal(
                    scale=sg, size=clean.shape)
        log(f"[stage11 dry run] {len(m4in['frames']['128'])} Foveated frames at two noise levels")
    return paths, m4in


def m4_dry(m4in, looks, refs, level, log=print):
    """The frozen M4 matcher (motion/multibaseline.py, both5) on the dry run's Foveated frames, as m4_points' output."""
    from motion import multibaseline as mb
    from motion import logpolar as lp
    model = m4in["model"]
    frames = m4in["frames"][level]
    rays = model.rays()
    R = looks[0][2]
    out = []
    for j in refs:
        ks = [j + STEP_LOOKS * i for i in WIN]
        t0 = time.perf_counter()
        ts = [float((looks[k][1] - looks[j][1]) @ (-R[:, 2])) for k in ks]
        r = mb.match_reference(frames[j], [frames[k] for k in ks], ts, model)
        ok = (r["reason"] == 0) & (r["zeta"] > 0)
        dl = rays.direction.reshape(-1, 3)[r["flat"][ok]]
        dw = lp.unit(dl) @ R.T
        rng = 1.0 / (r["zeta"][ok] * np.cos(r["e"][ok]))
        c = looks[j][1]
        out.append({"r": j, "c": c, "R": R, "X": c + dw * rng[:, None], "psi": np.degrees(r["e"][ok]),
                    "tried": int(len(ok)), "accepted": int(ok.sum()), "right_acc": None})
        log(f"[stage11 dry run] M4 at {j}: accepted {ok.mean():.3f} ({time.perf_counter() - t0:.0f} s)")
    return out
