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
