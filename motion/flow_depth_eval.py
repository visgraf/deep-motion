"""M3 evaluation side (M3a #6): oracle-ladder runs, truth metrics, gates with their controls (numpy only).

    run_oracle    O1 / O2 / O3 through M3's own depth stage (motion.flow_depth.reconstruct) and PLY writer
    summarize     per stride and eccentricity bin: error, interval, coverage, usable fractions

Denominators: `of_all` counts every hit of look b in the bin (as M2's usable fractions);
`matched` the jointly visible ones the source gave. A range is **usable** when its interval's relative
half-width is <= 10 % (M3's analogue of M2's bracket); **within 10 %** is the error-based count (M2's
"usable"). Oracle runs need no freeze: they are labelled oracle and never feed RGB acceptance.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import analysis, flow_depth, pointcloud
from . import flow_depth_oracle as fo
from . import logpolar as lp

ECC_BINS = analysis.ECC_BINS
PRIMARY_STRIDE = 5
O1_DPHI_MAX_RAD = 1e-6          # proposed (#6 Stage 2)
O1_DPHI_MIN_ECC_DEG = 1.0       # the analysis grid's inner edge; azimuth is ill-conditioned at the axis
O1_REL_MAX = 1e-6
O1_MIN_PARALLAX_DEG = 0.01
O1_CONTROL_BASELINE = 1.01      # a 1 % wrong baseline must give median error >= 1e-3
O3_COVERAGE_TOL = 0.01          # |empirical - 0.95| <= 0.01
O3_CONTROL_SCALE = 0.5          # intervals half as wide must cover <= 0.80
PLY_RAY_MAX_DEG = 1e-3          # float32 storage of metre-scale points


def _q(x, p):
    return float(np.quantile(x, p, method="nearest")) if x.size else None


def pair_result(la, lb, ob, kind, **kw) -> dict:
    corr = fo.correspondences(la, lb, kind, **kw)
    B = float(np.linalg.norm(lb[2] - la[2]))
    rec = flow_depth.reconstruct(corr, ob, B)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(rec["finite"], np.abs(rec["range"] - corr["truth_range"]) / corr["truth_range"], np.inf)
        covered = (corr["truth_range"] >= rec["near"]) & (corr["truth_range"] <= rec["far"])
    return {"corr": corr, "rec": rec, "B": B, "rel": rel, "covered": covered,
            "ecc_b_deg": np.degrees(corr["e_b"])}


def summarize(rows: list) -> dict:
    """rows: pair_result dicts of one stride."""
    cat = lambda f: np.concatenate([f(r) for r in rows]) if rows else np.zeros(0)
    ecc_all = cat(lambda r: r["corr"]["ecc_hits_b_deg"])
    ecc, rel = cat(lambda r: r["ecc_b_deg"]), cat(lambda r: r["rel"])
    half = cat(lambda r: r["rec"]["rel_half"])
    fin, usable, cov = cat(lambda r: r["rec"]["finite"]), cat(lambda r: r["rec"]["usable"]), cat(lambda r: r["covered"])
    unres = cat(lambda r: np.isinf(r["rec"]["far"]))

    def block(sel, sel_all):
        n_all, n = int(sel_all.sum()), int(sel.sum())
        f = sel & fin
        return {"of_all": n_all, "matched": n, "finite": int(f.sum()),
                "median_rel": _q(rel[sel], 0.5), "p90_rel": _q(rel[sel], 0.9),
                "median_rel_half": _q(half[f], 0.5),
                "within_10pct_of_all": float((rel[sel] <= 0.10).sum() / n_all) if n_all else None,
                "usable_of_all": float(usable[sel].sum() / n_all) if n_all else None,
                "usable_of_matched": float(usable[sel].sum() / n) if n else None,
                "outlier_50pct_of_finite": float((rel[f] > 0.5).mean()) if f.any() else None,
                "unresolved_of_matched": float(unres[sel].mean()) if n else None,
                "coverage": float(cov[sel].mean()) if n and np.nanmax(half[f], initial=0) > 0 else None}
    out = {"pairs": len(rows), "B_m": rows[0]["B"] if rows else None, "field": block(np.ones_like(ecc, bool), np.ones_like(ecc_all, bool)),
           "band_5_40": block((ecc >= 5) & (ecc < 40), (ecc_all >= 5) & (ecc_all < 40)),
           "fovea_2deg": block(ecc < 2, ecc_all < 2), "by_eccentricity": []}
    for lo, hi in zip(ECC_BINS[:-1], ECC_BINS[1:]):
        b = block((ecc >= lo) & (ecc < hi), (ecc_all >= lo) & (ecc_all < hi))
        b["bin"] = [lo, hi]
        out["by_eccentricity"].append(b)
    return out


def _save_pair(out: Path, stride, ka, kb, r, exposure):
    c, rec = r["corr"], r["rec"]
    pdir = out / "pairs" / f"s{stride}"
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / f"a{ka:04d}-b{kb:04d}.npz", sample_b=c["sample_b"], e_b=c["e_b"], phi_b=c["phi_b"],
                        phi_a=c["phi_a"], d=c["d"], d_lo=c["d_lo"], d_hi=c["d_hi"], range=rec["range"],
                        near=rec["near"], far=rec["far"], usable=rec["usable"], source=c["source"])
    keep = rec["finite"] & rec["usable"]
    if keep.any():
        pointcloud.write_ply(out / "clouds" / f"s{stride}" / f"a{ka:04d}-b{kb:04d}.ply", rec["P"][keep],
                             pointcloud.srgb_u8(rec["rgb_linear"][keep], exposure),
                             [f"foveal-motion M3 {c['source']} stride {stride} pair {ka}-{kb}"])
    return keep


def run_oracle(run: dict, kind: str, strides, out: Path, *, sigma_px: float = 0.0, seed: int = 0,
               mode: str = "ahead", git: dict | None = None) -> dict:
    t0 = time.perf_counter()
    if (out / "evaluation.json").exists():
        raise SystemExit(f"[flow-depth] refused: {out} already holds a result")
    out.mkdir(parents=True, exist_ok=True)
    es = sorted(run["looks"][mode], key=lambda e: e["k"])
    loads = {e["k"]: analysis.load_look(run, e) for e in es}
    obs = {e["k"]: flow_depth.load_observation(run, e) for e in es}
    ks = sorted(loads)
    exposure = pointcloud.exposure_of(obs[ks[0]].rec.value)
    pairs_all = [(ka, ka + s) for s in strides for ka in ks if ka + s in loads]
    align = flow_depth.alignment({k: (o.c, o.R) for k, o in obs.items()}, pairs_all)
    if not align["ok"]:
        raise SystemExit(f"[flow-depth] refused: the alignment gate failed: {json.dumps(align['detail'])}")
    summary, checks, prim = {}, [align], []
    o1_dphi, o1_rel, o1_ctrl, o1_dphi_ctrl, cov, cov_ctrl = [], [], [], [], [], []
    for s in strides:
        rows = []
        for ka in ks:
            kb = ka + s
            if kb not in loads:
                continue
            r = pair_result(loads[ka], loads[kb], obs[kb], kind, sigma_px=sigma_px, seed=seed + 1000 * s + ka)
            keep = _save_pair(out, s, ka, kb, r, exposure)
            if s == PRIMARY_STRIDE:
                ray = lp.dir_of_e_phi(r["corr"]["e_b"][keep], r["corr"]["phi_b"][keep]) @ obs[kb].R.T
                prim.append((r["rec"]["P"][keep], pointcloud.srgb_u8(r["rec"]["rgb_linear"][keep], exposure),
                             r["rec"]["rgb_linear"][keep], np.full(int(keep.sum()), ka, np.int32),
                             r["corr"]["sample_b"][keep], r["rec"]["rel_half"][keep],
                             np.broadcast_to(obs[kb].c, ray.shape).copy(), ray))
            if kind == "oracle-exact":
                c = r["corr"]
                wide = np.degrees(c["e_b"]) >= O1_DPHI_MIN_ECC_DEG
                o1_dphi.append(np.abs(np.angle(np.exp(1j * (c["phi_a"] - c["phi_b"]))))[wide])
                good = c["parallax_deg"] >= O1_MIN_PARALLAX_DEG
                o1_rel.append(r["rel"][good])
                ctrl = lp.range_b(c["d"], r["B"] * O1_CONTROL_BASELINE, c["e_b"])
                o1_ctrl.append((np.abs(ctrl - c["truth_range"]) / c["truth_range"])[good])
                if ka % 5 == 0:
                    cy = fo.correspondences(loads[ka], loads[kb], kind, yaw_b_deg=flow_depth.CONTROL_YAW_DEG)
                    w2 = np.degrees(cy["e_b"]) >= O1_DPHI_MIN_ECC_DEG
                    o1_dphi_ctrl.append(np.abs(np.angle(np.exp(1j * (cy["phi_a"] - cy["phi_b"]))))[w2])
            if kind == "oracle-noise":
                cov.append(r["covered"])
                c = r["corr"]
                mid, hw = 0.5 * (c["d_lo"] + c["d_hi"]), 0.5 * (c["d_hi"] - c["d_lo"]) * O3_CONTROL_SCALE
                near, far = lp.range_interval(mid - hw, mid + hw, r["B"], c["e_b"])
                cov_ctrl.append((c["truth_range"] >= near) & (c["truth_range"] <= far))
            rows.append(r)
        summary[f"stride_{s}"] = dict(summarize(rows), stride=s)
        del rows
    if kind == "oracle-exact":
        dphi, rel, ctl, dctl = (np.concatenate(x) for x in (o1_dphi, o1_rel, o1_ctrl, o1_dphi_ctrl))
        checks.append({"id": "M3.O1a", "what": f"O1 keeps azimuth: |phi_a - phi_b| <= {O1_DPHI_MAX_RAD} rad at e_b >= {O1_DPHI_MIN_ECC_DEG} deg",
                       "ok": float(dphi.max()) <= O1_DPHI_MAX_RAD, "control_fails": float(dctl.max()) > O1_DPHI_MAX_RAD,
                       "control": f"b's eye yawed {flow_depth.CONTROL_YAW_DEG} deg (every fifth pair)",
                       "detail": {"n": int(dphi.size), "max_rad": float(dphi.max()), "median_rad": float(np.median(dphi)),
                                  "control_max_rad": float(dctl.max())}})
        checks.append({"id": "M3.O1b", "what": f"O1 depth: relative range error <= {O1_REL_MAX} at parallax >= {O1_MIN_PARALLAX_DEG} deg",
                       "ok": float(rel.max()) <= O1_REL_MAX, "control_fails": float(np.median(ctl)) >= 1e-3,
                       "control": f"baseline x {O1_CONTROL_BASELINE}: median relative error >= 1e-3",
                       "detail": {"n": int(rel.size), "max_rel": float(rel.max()), "control_median_rel": float(np.median(ctl))}})
    if kind == "oracle-noise":
        c, cc = np.concatenate(cov), np.concatenate(cov_ctrl)
        checks.append({"id": "M3.O3", "what": f"O3 nominal 95 % intervals cover the truth within +/- {O3_COVERAGE_TOL}",
                       "ok": abs(float(c.mean()) - 0.95) <= O3_COVERAGE_TOL, "control_fails": float(cc.mean()) <= 0.80,
                       "control": f"intervals x {O3_CONTROL_SCALE}: coverage <= 0.80",
                       "detail": {"n": int(c.size), "coverage": float(c.mean()), "control_coverage": float(cc.mean())}})
    if prim:
        P, rgb, lin, ka_, sb, hw, cb_, ray = (np.concatenate([p[i] for p in prim]) for i in range(8))
        pointcloud.write_ply(out / "reconstruction.ply", P, rgb, [f"foveal-motion M3 {kind} stride {PRIMARY_STRIDE}"])
        np.savez_compressed(out / "reconstruction.npz", P=P, rgb_linear=lin, pair_a=ka_, sample_b=sb, rel_half=hw,
                            source=kind, stride=PRIMARY_STRIDE)
        xyz, rgb_back, _ = pointcloud.read_ply(out / "reconstruction.ply")
        ang = lambda X: np.degrees(np.arctan2(np.linalg.norm(np.cross(X - cb_, ray), axis=-1), np.sum((X - cb_) * ray, -1)))
        a_ok, a_ctrl = ang(xyz.astype(np.float64)), ang(xyz[:, [1, 0, 2]].astype(np.float64))
        same = bool(np.isfinite(xyz).all() and np.array_equal(xyz, P.astype(np.float32)) and np.array_equal(rgb_back, rgb))
        checks.append({"id": "M3.PLY", "what": f"reconstruction.ply reads back (finite, same xyz and colours) and every point lies on its anchor ray (<= {PLY_RAY_MAX_DEG} deg)",
                       "ok": same and float(a_ok.max()) <= PLY_RAY_MAX_DEG,
                       "control_fails": float(np.median(a_ctrl)) > PLY_RAY_MAX_DEG,
                       "control": "the read-back cloud with x and y swapped: median angle to the anchor rays > the limit",
                       "detail": {"points": int(len(xyz)), "readback_identical": same, "max_ray_angle_deg": float(a_ok.max()),
                                  "control_median_deg": float(np.median(a_ctrl))}})
    res = {"schema": "foveal-motion-m3-eval-v1", "source": kind, "sigma_px": sigma_px, "seed": seed, "mode": mode,
           "strides": list(strides), "run": run["_dir"], "run_git": run["git"], "computed_by": git,
           "exposure": exposure, "checks": checks, "summary": summary, "seconds": round(time.perf_counter() - t0, 1)}
    (out / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ----------------------------------------------------------------------------- Stages 4-5: RGB calibration and evaluation

EDGE_PX = 2                      # #5: Position averages a pixel's samples, so truth is unreliable at object
                                 # boundaries; anchors within 2 px of one in b are "edge", evaluated apart
COV_MIN = 0.90                   # held-out empirical coverage of nominal 95 % intervals (#6 Stage 4)
COV_CONTROL_SCALE = 0.5
GATES = {   # fixed in #5 before measurement; stride 5, textured, jointly visible, interior 5-40 deg hits
    "calibration": {"median": 0.05, "p90": 0.10, "coverage": 0.50, "outlier50": None, "band_deg": (5.0, 40.0)},
    "classroom": {"median": 0.10, "p90": 0.25, "coverage": 0.25, "outlier50": 0.05, "band_deg": (5.0, 40.0)}}
CAL_GATE = GATES["calibration"]
BASELINE_CONTROL = 1.2


def _pairs(result: Path):
    return sorted((result / "pairs").glob("s*/a*-b*.npz"))


def _sha(p) -> str:
    return flow_depth._sha(p)


def pair_truth(la, lb, z) -> dict:
    """Truth for one pair's anchors (evaluation side): range along b's centre ray (|P - c_b|, P b's
    Position), the true log-tangent displacement of that point, and whether a sees P."""
    rec_a, Ra, ca, model = la
    rec_b, Rb, cb, _ = lb
    s = z["sample_b"]
    P = rec_b.position_w[s].astype(np.float64)
    hit = np.isfinite(rec_b.distance_m[s])
    r = np.where(hit, np.linalg.norm(P - cb, axis=-1), np.nan)
    ray = lp.dir_of_e_phi(z["e_b"], z["phi_b"]) @ Rb.T
    X = cb + np.nan_to_num(r)[:, None] * ray
    with np.errstate(invalid="ignore", divide="ignore"):
        qa, _ = lp.qphi(lp.project_static(X, ca, Ra))
    vis = np.zeros(len(s), bool)
    if hit.any():
        vis[hit] = analysis._agree(rec_a, model, Ra, ca, P[hit])[0]
    ecc_all = np.degrees(lp.e_phi(lp.unit(rec_b.direction[np.isfinite(rec_b.distance_m)]) @ Rb)[0])
    n = model.n
    I = np.full(n * n, -1)
    I[rec_b.raster_index] = rec_b.instance_id
    I = I.reshape(n, n)
    E = np.zeros((n, n), bool)
    for di in range(-EDGE_PX, EDGE_PX + 1):
        for dj in range(-EDGE_PX, EDGE_PX + 1):
            S = np.full((n, n), -1)
            S[max(-di, 0):n + min(-di, 0), max(-dj, 0):n + min(-dj, 0)] = I[max(di, 0):n + min(di, 0), max(dj, 0):n + min(dj, 0)]
            E |= (S != I) & (S >= 0) & (I >= 0)
    edge = E.reshape(-1)[z["flat"]]
    return {"range": r, "hit": hit, "d": z["q_b"] - qa, "visible_in_a": vis, "ecc_hits_b_deg": ecc_all, "edge": edge}


def calibrate(result: Path, parity: str = "even", also=()) -> dict:
    """Stage 4: the interval scale from the declared calibration set (pairs whose ka has this parity),
    then the freeze. Truth of the other pairs is not opened."""
    result = Path(result)
    if (result / "freeze.json").exists():
        raise SystemExit(f"[flow-depth] refused: {result} is already frozen")
    cfg = json.loads((result / "config.json").read_text())
    run = analysis.load_run(cfg["run"])
    es = {e["k"]: e for e in run["looks"][cfg["mode"]]}
    want = 0 if parity == "even" else 1
    zs, by = [], {}
    used = []
    for p in _pairs(result):
        z = np.load(p)
        ka, kb = int(z["ka"]), int(z["kb"])
        if ka % 2 != want and (ka, kb) not in set(map(tuple, also)):
            continue
        used.append([ka, kb])
        la, lb = analysis.load_look(run, es[ka]), analysis.load_look(run, es[kb])
        t = pair_truth(la, lb, z)
        acc = (z["reason"] == 0) & t["hit"] & np.isfinite(t["d"]) & ~t["edge"]
        zz = np.abs(z["d"][acc] - t["d"][acc]) / z["sigma_d"][acc]
        zs.append(zz)
        by.setdefault(kb - ka, []).append(zz)
    allz = np.concatenate(zs)
    scale = float(np.quantile(allz, 0.95) / fo.Z95)
    q = lambda x, p: float(np.quantile(x, p))
    cal = {"schema": "foveal-motion-m3-calibration-v1", "set": f"pairs with {parity} ka, and {[list(p) for p in also]}", "pairs": used,
           "method": f"interval_scale = 95th percentile of |d - d_true| / sigma_d over accepted interior anchors (hit, more than {EDGE_PX} px from an object boundary in b), / 1.96",
           "n": int(allz.size), "interval_scale": scale,
           "abs_z_quantiles": {p: q(allz, float(p)) for p in ("0.5", "0.68", "0.9", "0.95", "0.99")},
           "by_stride": {str(s): {"n": int(np.concatenate(v).size),
                                  "scale_if_alone": float(np.quantile(np.concatenate(v), 0.95) / fo.Z95)} for s, v in by.items()}}
    (result / "calibration.json").write_text(json.dumps(cal, indent=1))
    frozen = dict(cfg["settings"], interval_scale=scale)
    files = {str(p.relative_to(result)): _sha(p) for p in [result / "config.json", result / "calibration.json", *_pairs(result)]}
    held = [[int(np.load(p)["ka"]), int(np.load(p)["kb"])] for p in _pairs(result)
            if [int(np.load(p)["ka"]), int(np.load(p)["kb"])] not in used]
    fr = {"schema": "foveal-motion-m3-freeze-v1", "frozen_settings": frozen, "sha256": files, "held_out_pairs": held,
          "note": "estimation outputs and the calibrated scale; evaluation verifies these before opening held-out truth"}
    (result / "freeze.json").write_text(json.dumps(fr, indent=1, default=float))
    return cal


def import_freeze(result: Path, source: Path, exclude=()) -> dict:
    """M3b: freeze a new estimate (e.g. the Classroom) with the interval scale frozen on the calibration room,
    without opening any of its truth. The matcher settings must be identical; every pair is held out."""
    result, source = Path(result), Path(source)
    if (result / "freeze.json").exists():
        raise SystemExit(f"[flow-depth] refused: {result} is already frozen")
    src = _verify(source)
    cfg = json.loads((result / "config.json").read_text())
    a = {k: v for k, v in cfg["settings"].items() if k != "interval_scale"}
    b = {k: v for k, v in src["frozen_settings"].items() if k != "interval_scale"}
    if json.loads(json.dumps(a)) != json.loads(json.dumps(b)):
        raise SystemExit("[flow-depth] refused: the estimate's matcher settings differ from the frozen ones")
    cal = {"schema": "foveal-motion-m3-calibration-v1", "set": f"imported from {source}", "pairs": [],
           "interval_scale": src["frozen_settings"]["interval_scale"], "source_freeze_sha256": _sha(source / "freeze.json"),
           "source_calibration": json.loads((source / "calibration.json").read_text())}
    (result / "calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    files = {str(p.relative_to(result)): _sha(p) for p in [result / "config.json", result / "calibration.json", *_pairs(result)]}
    ex = {tuple(p) for p in exclude}
    held = [[int(np.load(p)["ka"]), int(np.load(p)["kb"])] for p in _pairs(result)]
    held = [p for p in held if tuple(p) not in ex]
    fr = {"schema": "foveal-motion-m3-freeze-v1", "frozen_settings": dict(src["frozen_settings"]), "sha256": files,
          "held_out_pairs": held, "excluded_pairs": sorted(list(p) for p in ex),
          "note": f"scale imported from {source}; no truth of this result was opened; excluded pairs are not evaluated"}
    (result / "freeze.json").write_text(json.dumps(fr, indent=1, default=float))
    return cal


def _verify(result: Path) -> dict:
    fr = json.loads((result / "freeze.json").read_text())
    bad = [k for k, h in fr["sha256"].items() if not (result / k).is_file() or _sha(result / k) != h]
    if bad:
        raise SystemExit(f"[flow-depth] refused: frozen files changed or missing: {bad[:5]}")
    return fr


def _rgb_rows(z, t, ob, scale, B=None):
    """Estimator reconstruction with the frozen scale (no truth), then truth metrics."""
    acc = z["reason"] == 0
    d = np.where(acc, z["d"], np.nan)
    half = fo.Z95 * scale * z["sigma_d"]
    corr = {"sample_b": z["sample_b"], "e_b": z["e_b"], "phi_b": z["phi_b"], "d": d, "d_lo": d - half, "d_hi": d + half}
    Bv = float(z["B"]) if B is None else B
    rec = flow_depth.reconstruct(corr, ob, Bv)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(rec["finite"] & t["hit"], np.abs(rec["range"] - t["range"]) / t["range"], np.inf)
        covered = (t["range"] >= rec["near"]) & (t["range"] <= rec["far"])
    corr["ecc_hits_b_deg"] = t["ecc_hits_b_deg"]
    return {"corr": corr, "rec": rec, "B": Bv, "rel": rel, "covered": covered, "ecc_b_deg": np.degrees(z["e_b"]),
            "acc": acc, "truth": t, "z": z}


def evaluate(result: Path, log=print, gate: str = "calibration") -> dict:
    """Stage 5: verify the freeze, reconstruct every pair with the frozen scale (PLY), then open truth for
    the held-out pairs only and run the gates with their controls."""
    t0 = time.perf_counter()
    result = Path(result)
    if (result / "evaluation.json").exists():
        raise SystemExit(f"[flow-depth] refused: {result} already holds an evaluation")
    fr = _verify(result)
    cfg = json.loads((result / "config.json").read_text())
    scale = fr["frozen_settings"]["interval_scale"]
    G = GATES[gate]
    frozen = {k: v for k, v in fr["frozen_settings"].items() if k != "interval_scale"}
    if json.loads(json.dumps(frozen)) != json.loads(json.dumps({k: v for k, v in cfg["settings"].items() if k != "interval_scale"})):
        raise SystemExit("[flow-depth] refused: estimate settings differ from the frozen settings")
    run = analysis.load_run(cfg["run"])
    es = {e["k"]: e for e in run["looks"][cfg["mode"]]}
    held = {tuple(p) for p in fr["held_out_pairs"]}
    first = min(es)
    exposure = pointcloud.exposure_of(flow_depth.load_observation(run, es[first]).rec.value)
    # 1. reconstruction, no truth: every pair's PLY and the stride-5 cloud
    prim = []
    for p in _pairs(result):
        z = np.load(p)
        ka, kb = int(z["ka"]), int(z["kb"])
        ob = flow_depth.load_observation(run, es[kb])
        acc = z["reason"] == 0
        d = np.where(acc, z["d"], np.nan)
        half = fo.Z95 * scale * z["sigma_d"]
        rec = flow_depth.reconstruct({"sample_b": z["sample_b"], "e_b": z["e_b"], "phi_b": z["phi_b"], "d": d,
                                      "d_lo": d - half, "d_hi": d + half}, ob, float(z["B"]))
        keep = rec["finite"] & rec["usable"]
        if keep.any():
            rgb = pointcloud.srgb_u8(rec["rgb_linear"][keep], exposure)
            pointcloud.write_ply(result / "clouds" / f"s{kb - ka}" / f"a{ka:04d}-b{kb:04d}.ply", rec["P"][keep], rgb,
                                 [f"foveal-motion M3 rgb stride {kb - ka} pair {ka}-{kb}"])
            if kb - ka == PRIMARY_STRIDE:
                ray = lp.dir_of_e_phi(z["e_b"][keep], z["phi_b"][keep]) @ ob.R.T
                prim.append((rec["P"][keep], rgb, rec["rgb_linear"][keep], np.full(int(keep.sum()), ka, np.int32),
                             z["sample_b"][keep], rec["rel_half"][keep], np.broadcast_to(ob.c, ray.shape).copy(), ray))
    checks = []
    if prim:
        P, rgb, lin, ka_, sb, hw, cb_, ray = (np.concatenate([x[i] for x in prim]) for i in range(8))
        pointcloud.write_ply(result / "reconstruction.ply", P, rgb, [f"foveal-motion M3 rgb stride {PRIMARY_STRIDE}"])
        np.savez_compressed(result / "reconstruction.npz", P=P, rgb_linear=lin, pair_a=ka_, sample_b=sb, rel_half=hw,
                            source="rgb", stride=PRIMARY_STRIDE)
        xyz, rgb_back, _ = pointcloud.read_ply(result / "reconstruction.ply")
        ang = lambda X: np.degrees(np.arctan2(np.linalg.norm(np.cross(X - cb_, ray), axis=-1), np.sum((X - cb_) * ray, -1)))
        a_ok, a_ctrl = ang(xyz.astype(np.float64)), ang(xyz[:, [1, 0, 2]].astype(np.float64))
        same = bool(np.isfinite(xyz).all() and np.array_equal(xyz, P.astype(np.float32)) and np.array_equal(rgb_back, rgb))
        checks.append({"id": "M3.PLY", "what": f"reconstruction.ply reads back and every point lies on its anchor ray (<= {PLY_RAY_MAX_DEG} deg)",
                       "ok": same and float(a_ok.max()) <= PLY_RAY_MAX_DEG, "control_fails": float(np.median(a_ctrl)) > PLY_RAY_MAX_DEG,
                       "control": "x and y swapped", "detail": {"points": int(len(xyz)), "max_ray_angle_deg": float(a_ok.max()),
                                                                "control_median_deg": float(np.median(a_ctrl))}})
    log(f"[flow-depth] reconstruction written ({time.perf_counter() - t0:.1f} s); opening held-out truth")
    # 2. held-out truth
    rows_by, cov, cov_c, disp, edge_rel = {}, [], [], [], []
    band, ctrlB = [], []
    ladder = {k: {} for k in ("O2 pixel", "O3 0.05 px", "O3 0.10 px")}
    for p in _pairs(result):
        z = np.load(p)
        ka, kb = int(z["ka"]), int(z["kb"])
        if (ka, kb) not in held:
            continue
        la, lb = analysis.load_look(run, es[ka]), analysis.load_look(run, es[kb])
        ob = flow_depth.load_observation(run, es[kb])
        t = pair_truth(la, lb, z)
        r = _rgb_rows(z, t, ob, scale)
        s = kb - ka
        rows_by.setdefault(s, []).append(r)
        acc = r["acc"] & t["hit"] & r["rec"]["finite"] & ~t["edge"]
        cov.append(r["covered"][acc])
        ue = r["rec"]["usable"] & r["acc"] & t["hit"] & t["edge"]
        edge_rel.append(r["rel"][ue])
        rc = _rgb_rows(z, t, ob, scale * COV_CONTROL_SCALE)
        cov_c.append(rc["covered"][acc])
        ok_d = acc & np.isfinite(t["d"])                       # interior
        disp.append(((z["d"] - t["d"]) * z["px_per_q_a"])[ok_d])
        e = np.degrees(z["e_b"])
        subset = t["hit"] & t["visible_in_a"] & ~t["edge"] & (z["texture"] >= cfg["settings"]["texture_min"]) \
            & (e >= G["band_deg"][0]) & (e < G["band_deg"][1])
        band.append((subset, r["rec"]["usable"] & r["acc"], r["rel"], s))
        rB = _rgb_rows(z, t, ob, scale, B=float(z["B"]) * BASELINE_CONTROL)
        ctrlB.append((subset, rB["rec"]["usable"] & rB["acc"], rB["rel"]))
        for name, kind, sg in (("O2 pixel", "oracle-pixel", 0.0), ("O3 0.05 px", "oracle-noise", 0.05), ("O3 0.10 px", "oracle-noise", 0.10)):
            ladder[name].setdefault(s, []).append(pair_result(la, lb, ob, kind, sigma_px=sg, seed=1000 * s + ka))
        log(f"[flow-depth] held-out ({ka}, {kb}) evaluated")
    summary = {f"stride_{s}": dict(summarize(v), stride=s) for s, v in sorted(rows_by.items())}
    lad = {n: {f"stride_{s}": summarize(v)["field"]["usable_of_all"] for s, v in sorted(d.items())} for n, d in ladder.items()}
    c, cc = np.concatenate(cov), np.concatenate(cov_c)
    er = np.concatenate(edge_rel) if edge_rel else np.zeros(0)
    checks.append({"id": "M3.COV", "what": f"held-out coverage of nominal 95 % intervals >= {COV_MIN} (interior anchors)",
                   "ok": float(c.mean()) >= COV_MIN, "control_fails": float(cc.mean()) < COV_MIN,
                   "control": f"intervals x {COV_CONTROL_SCALE}", "detail": {"n": int(c.size), "coverage": float(c.mean()),
                                                                              "control_coverage": float(cc.mean()), "scale": scale}})
    gate_rows = {}
    for s in sorted(rows_by):
        sel = [b for b in band if b[3] == s]
        sub = np.concatenate([b[0] for b in sel])
        use = np.concatenate([b[1] for b in sel])
        rel = np.concatenate([b[2] for b in sel])
        a = sub & use
        gate_rows[s] = {"subset": int(sub.sum()), "accepted_usable": int(a.sum()),
                        "coverage": float(a.sum() / max(int(sub.sum()), 1)),
                        "median_rel": float(np.median(rel[a])) if a.any() else None,
                        "p90_rel": float(np.quantile(rel[a], 0.9)) if a.any() else None,
                        "over_50pct": float(np.mean(rel[a] > 0.5)) if a.any() else None}
    g5 = gate_rows.get(PRIMARY_STRIDE)
    okg = lambda g: g is not None and g["median_rel"] is not None and g["median_rel"] <= G["median"] \
        and g["p90_rel"] <= G["p90"] and g["coverage"] >= G["coverage"] \
        and (G["outlier50"] is None or g["over_50pct"] <= G["outlier50"])
    sel = [x for x, b in zip(ctrlB, band) if b[3] == PRIMARY_STRIDE]
    subC, useC, relC = (np.concatenate([x[i] for x in sel]) for i in range(3))
    aC = subC & useC
    gC = {"coverage": float(aC.sum() / max(int(subC.sum()), 1)), "median_rel": float(np.median(relC[aC])) if aC.any() else None,
          "p90_rel": float(np.quantile(relC[aC], 0.9)) if aC.any() else None,
          "over_50pct": float(np.mean(relC[aC] > 0.5)) if aC.any() else None}
    checks.append({"id": "M3.CAL" if gate == "calibration" else "M3.ROOM", "what": f"held-out {gate} gate at stride 5: textured, jointly visible, interior 5-40 deg hits: "
                   f"median <= {G['median']}, p90 <= {G['p90']}, accepted >= {G['coverage']}"
                   + (f", > 50 % errors <= {G['outlier50']} of accepted" if G["outlier50"] is not None else ""),
                   "ok": okg(g5), "control_fails": not okg(gC), "control": f"baseline x {BASELINE_CONTROL}",
                   "detail": {"by_stride": gate_rows, "control": gC}})
    dd = np.abs(np.concatenate(disp))
    res = {"schema": "foveal-motion-m3-eval-v1", "source": "rgb", "freeze_verified": True, "interval_scale": scale, "gate": gate,
           "held_out_pairs": sorted(list(p) for p in held), "checks": checks, "summary": summary,
           "displacement_error_px_interior": {"n": int(dd.size), "median": float(np.median(dd)), "p90": float(np.quantile(dd, 0.9)),
                                              "over_1px": float(np.mean(dd > 1))},
           "edge_usable_range_error": {"n": int(er.size), "median": float(np.median(er)) if er.size else None,
                                       "p90": float(np.quantile(er, 0.9)) if er.size else None,
                                       "over_50pct": float(np.mean(er > 0.5)) if er.size else None},
           "ladder_same_pairs_field_usable": lad, "run": run["_dir"], "computed_by": None,
           "seconds": round(time.perf_counter() - t0, 1)}
    (result / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    return res
