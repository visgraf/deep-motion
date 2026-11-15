"""M4a Stage 5 (#10): estimate (truth-free), calibrate on a declared set, freeze, evaluate held-out.

Reference frames r of a run's `ahead` looks; a window is a list of frame offsets (WINDOWS). The split follows
M3's on the same run, so both are judged on the same unseen pairs: a reference frame r belongs to the
calibration set when M3's pair (r - 5, r) did (r - 5 even, plus r = 6, 8, 10, whose truth Chat's smoke tests
opened); the rest are held out. M3's calibration-room held-out stride-5 pairs are (7, 12), (9, 14), (11, 16),
(13, 18), (15, 20), so M4's held-out reference frames are 12, 14, 16, 18, 20.

Truth (evaluation side only): r's Position per anchor, the range along r's centre ray, zeta_true =
1 / (range cos e); anchors within 2 px of an object boundary in r are "edge" (as M3).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import analysis, flow_depth, flow_depth_eval as fe, matcher as m3, multibaseline as mb, pointcloud
from . import logpolar as lp

WINDOWS = {"before5": list(range(-5, 0)), "both5": [k for k in range(-5, 6) if k], "before10": list(range(-10, 0))}
EXTRA_CAL = (6, 8, 10)
M3_PAIR = 5                         # M3's stride for the head-to-head


def is_calibration(r: int) -> bool:
    return (r - M3_PAIR) % 2 == 0 or r in EXTRA_CAL


def references(ks, window):
    off = WINDOWS[window]
    return [r for r in ks if all((r + o) in ks for o in off)]


def _lum(o):
    return m3.luminance(o.rgb)


def estimate(run: dict, window: str, refs, out: Path, settings=None, git=None, log=print) -> dict:
    out = Path(out) / window
    out.mkdir(parents=True, exist_ok=True)
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    s = dict(mb.SETTINGS, **(settings or {}))
    cfg_p = out / "config.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {
        "schema": "foveal-motion-m4-estimate-v1", "run": run["_dir"], "run_git": run["git"], "window": window,
        "offsets": WINDOWS[window], "settings": s, "computed_by": git, "refs": [], "inputs_sha256": {}}
    if cfg["settings"] != json.loads(json.dumps(s)):
        raise SystemExit("[m4] refused: settings differ from this result's earlier estimates")
    obs = {}
    get = lambda k: obs.setdefault(k, flow_depth.load_observation(run, es[k]))
    for r in refs:
        p = out / f"r{r:04d}.npz"
        if p.exists():
            raise SystemExit(f"[m4] refused: {p} exists")
        t0 = time.perf_counter()
        ks = [r + o for o in WINDOWS[window]]
        ob = get(r)
        al = flow_depth.alignment({k: (get(k).c, get(k).R) for k in [r] + ks}, [(k, r) for k in ks])
        if not al["ok"]:
            raise SystemExit(f"[m4] refused: alignment gate failed at r = {r}: {al['detail']}")
        h = ob.R @ [0, 0, -1.0]
        ts = [float((get(k).c - ob.c) @ h) for k in ks]
        res = mb.match_reference(_lum(ob), [_lum(get(k)) for k in ks], ts, ob.model, s)
        n = ob.model.n
        inv = np.full(n * n, -1, dtype=np.int64)
        inv[ob.rec.raster_index] = np.arange(ob.rec.n_samples)
        sample = inv[res["flat"]]
        keys = ("flat", "e", "phi", "q", "zeta", "sigma_zeta", "zeta_q", "n_frames", "ncc_coarse", "ncc_second",
                "zeta_coarse", "texture", "support", "edge_dev_px", "px_per_zeta", "reason")
        np.savez_compressed(p, r=r, frames=np.array(ks), ts=np.array(ts), sample=sample, **{k: res[k] for k in keys})
        for k in [r] + ks:
            for f in ("record_C.npz", "meta.json"):
                cfg["inputs_sha256"][f"{k}/{f}"] = flow_depth._sha(Path(run["_dir"]) / es[k]["out_dir"] / f)
        cfg["refs"] = sorted(set(cfg["refs"]) | {r})
        cfg_p.write_text(json.dumps(cfg, indent=1, default=float))
        log(f"[m4] {window} r {r}: accepted {np.mean(res['reason'] == 0):.3f} ({time.perf_counter() - t0:.0f} s)")
    return cfg


def truth(run, r, z, other=None):
    """Range along r's centre rays, zeta_true, hit, edge; with other = k: whether frame k sees the point."""
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    rec, R, c, model = analysis.load_look(run, es[r])
    s = z["sample"]
    hit = np.isfinite(rec.distance_m[s])
    P = rec.position_w[s].astype(np.float64)
    rng = np.where(hit, np.linalg.norm(P - c, axis=-1), np.nan)
    zt = 1.0 / (rng * np.cos(z["e"]))
    n = model.n
    I = np.full(n * n, -1)
    I[rec.raster_index] = rec.instance_id
    I = I.reshape(n, n)
    E = np.zeros((n, n), bool)
    for di in range(-fe.EDGE_PX, fe.EDGE_PX + 1):
        for dj in range(-fe.EDGE_PX, fe.EDGE_PX + 1):
            S = np.full((n, n), -1)
            S[max(-di, 0):n + min(-di, 0), max(-dj, 0):n + min(-dj, 0)] = I[max(di, 0):n + min(di, 0), max(dj, 0):n + min(dj, 0)]
            E |= (S != I) & (S >= 0) & (I >= 0)
    out = {"range": rng, "zeta": zt, "hit": hit, "edge": E.reshape(-1)[z["flat"]],
           "ecc_hits_deg": np.degrees(lp.e_phi(lp.unit(rec.direction[np.isfinite(rec.distance_m)]) @ R)[0])}
    if other is not None:
        reca, Ra, ca, _ = analysis.load_look(run, es[other])
        vis = np.zeros(len(s), bool)
        if hit.any():
            vis[hit] = analysis._agree(reca, model, Ra, ca, P[hit])[0]
        out["visible_in_other"] = vis
    return out


def calibrate(out: Path, window: str) -> dict:
    d = Path(out) / window
    if (d / "freeze.json").exists():
        raise SystemExit(f"[m4] refused: {d} is frozen")
    cfg = json.loads((d / "config.json").read_text())
    run = analysis.load_run(cfg["run"])
    zs, used = [], []
    for r in cfg["refs"]:
        if not is_calibration(r):
            continue
        z = dict(np.load(d / f"r{r:04d}.npz"))
        t = truth(run, r, z)
        acc = (z["reason"] == 0) & t["hit"] & ~t["edge"]
        zs.append(np.abs(z["zeta"][acc] - t["zeta"][acc]) / z["sigma_zeta"][acc])
        used.append(r)
    allz = np.concatenate(zs)
    scale = float(np.quantile(allz, 0.95) / mb.Z95)
    cal = {"schema": "foveal-motion-m4-calibration-v1", "window": window, "refs": used, "n": int(allz.size),
           "interval_scale": scale, "abs_z_quantiles": {p: float(np.quantile(allz, float(p))) for p in ("0.5", "0.68", "0.9", "0.95", "0.99")},
           "method": "95th percentile of |zeta - zeta_true| / sigma_zeta over accepted interior anchors with a hit, / 1.96"}
    (d / "calibration.json").write_text(json.dumps(cal, indent=1))
    files = {p.name: flow_depth._sha(p) for p in [d / "config.json", d / "calibration.json", *sorted(d.glob("r*.npz"))]}
    fr = {"schema": "foveal-motion-m4-freeze-v1", "frozen_settings": dict(cfg["settings"], interval_scale=scale), "sha256": files,
          "calibration_refs": used, "held_out_refs": [r for r in cfg["refs"] if not is_calibration(r)]}
    (d / "freeze.json").write_text(json.dumps(fr, indent=1, default=float))
    return cal


def _verify(d: Path) -> dict:
    fr = json.loads((d / "freeze.json").read_text())
    bad = [k for k, h in fr["sha256"].items() if flow_depth._sha(d / k) != h]
    if bad:
        raise SystemExit(f"[m4] refused: frozen files changed: {bad}")
    return fr


def held_out_estimates_are_new(d: Path, fr: dict) -> list:
    """Held-out npz files may be added after the freeze (truth-free); they are hashed at evaluation."""
    return sorted(p.name for p in d.glob("r*.npz") if p.name not in fr["sha256"])


def evaluate(out: Path, window: str, m3_scale: float, log=print) -> dict:
    """Held-out: M4's ranges with the frozen scale; M3's frozen matcher on the pair (r - 5, r) with M3's frozen
    scale, on the same anchors; #5's calibration gate on M3's subset (textured, visible in r - 5, interior,
    5-40 deg) for both; coverage of the 95 % intervals."""
    d = Path(out) / window
    fr = _verify(d)
    cfg = json.loads((d / "config.json").read_text())
    if json.loads(json.dumps({k: v for k, v in fr["frozen_settings"].items() if k != "interval_scale"})) != cfg["settings"]:
        raise SystemExit("[m4] refused: settings differ from the frozen ones")
    scale = fr["frozen_settings"]["interval_scale"]
    run = analysis.load_run(cfg["run"])
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    G = fe.GATES["calibration"]
    rows, late = [], held_out_estimates_are_new(d, fr)
    for r in sorted(set(cfg["refs"]) - set(fr["calibration_refs"])):
        z = dict(np.load(d / f"r{r:04d}.npz"))
        ob = flow_depth.load_observation(run, es[r])
        t = truth(run, r, z, other=r - M3_PAIR)
        acc = z["reason"] == 0
        half = mb.Z95 * scale * z["sigma_zeta"]
        ce = np.cos(z["e"])
        rng = 1.0 / (z["zeta"] * ce)
        near = 1.0 / ((z["zeta"] + half) * ce)
        with np.errstate(divide="ignore", invalid="ignore"):
            far = np.where(z["zeta"] - half > 0, 1.0 / ((z["zeta"] - half) * ce), np.inf)
            relh = 0.5 * (far - near) / rng
        usable = acc & np.isfinite(relh) & (relh <= flow_depth.USABLE_REL)
        rel = np.abs(rng - t["range"]) / t["range"]
        cov = (t["range"] >= near) & (t["range"] <= far)
        e = np.degrees(z["e"])
        subset = t["hit"] & t["visible_in_other"] & ~t["edge"] & (z["texture"] >= cfg["settings"]["texture_min"]) & (e >= 5) & (e < 40)
        # M3 on the pair (r - 5, r), same anchors (M3 anchors are b's pixels at 1-44 deg, the same raster indices)
        oa = flow_depth.load_observation(run, es[r - M3_PAIR])
        B = float(np.linalg.norm(ob.c - oa.c))
        c3 = m3.match_pair(_lum(oa), _lum(ob), ob.model, B)
        if not np.array_equal(c3["flat"], z["flat"]):
            raise SystemExit("[m4] M3 and M4 anchors differ")
        acc3 = c3["reason"] == 0
        h3 = mb.Z95 * m3_scale * c3["sigma_d"]
        d3 = np.where(acc3, c3["d"], np.nan)
        rec3 = flow_depth.reconstruct({"sample_b": z["sample"], "e_b": z["e"], "phi_b": z["phi"], "d": d3, "d_lo": d3 - h3,
                                       "d_hi": d3 + h3}, ob, B)
        use3 = rec3["usable"] & acc3
        rel3 = np.abs(rec3["range"] - t["range"]) / t["range"]
        cov3 = (t["range"] >= rec3["near"]) & (t["range"] <= rec3["far"])
        rows.append({"r": r, "n_hits": int(t["ecc_hits_deg"].size), "usable": usable, "use3": use3, "rel": rel, "rel3": rel3,
                     "cov": cov, "cov3": cov3, "acc": acc, "acc3": acc3, "hit": t["hit"], "edge": t["edge"], "subset": subset,
                     "e": e, "fovea": e < 2})
        log(f"[m4] held-out r {r}: M4 usable {usable.mean():.3f}, M3 usable {use3.mean():.3f}")
    cat = lambda k: np.concatenate([x[k] for x in rows])
    hits = sum(x["n_hits"] for x in rows)
    res = {"window": window, "interval_scale": scale, "m3_scale": m3_scale, "held_out_refs": [x["r"] for x in rows],
           "late_estimates_hashed_now": {n: flow_depth._sha(d / n) for n in late}}
    for tag, U, REL, COV, ACC in (("M4", "usable", "rel", "cov", "acc"), ("M3", "use3", "rel3", "cov3", "acc3")):
        use, rel, cov, acc = cat(U), cat(REL), cat(COV), cat(ACC)
        hit, edge, sub = cat("hit"), cat("edge"), cat("subset")
        inter = acc & hit & ~edge & np.isfinite(rel)
        gs = sub & use & hit
        g = {"coverage_of_subset": float(gs.sum() / max(int(sub.sum()), 1)),
             "median_rel": float(np.median(rel[gs])) if gs.any() else None,
             "p90_rel": float(np.quantile(rel[gs], 0.9)) if gs.any() else None,
             "over_50pct": float(np.mean(rel[gs] > 0.5)) if gs.any() else None}
        g["gate_pass"] = bool(g["median_rel"] is not None and g["median_rel"] <= G["median"] and g["p90_rel"] <= G["p90"]
                              and g["coverage_of_subset"] >= G["coverage"])
        res[tag] = {"field_usable_of_all_hits": float(use.sum() / hits), "accepted_fraction": float(acc.mean()),
                    "interval_coverage_interior": float(cov[inter].mean()) if inter.any() else None,
                    "usable_interior_median_rel": float(np.median(rel[use & hit & ~edge])) if (use & hit & ~edge).any() else None,
                    "usable_interior_p90_rel": float(np.quantile(rel[use & hit & ~edge], 0.9)) if (use & hit & ~edge).any() else None,
                    "fovea_usable": float(use[cat("fovea")].mean()), "gate": g}
    (d / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    return res
