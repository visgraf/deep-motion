"""M6a T2 (#22): an off-centre epipole with no rotation (the crab dolly), and M6's reduction to M4.

T2.0 (`repro`): on M4's own calibration-room run (`runs/m3-calib-1024`, Code's 1024-spp records), M6's matcher with
the recorded poses must give M4's frozen matcher's decision on every anchor, and its zeta within 1e-9 (relative) on
every accepted anchor, at the given reference frames (both5). Control (must fail): the same with every frame's
translation turned 1 deg about r's +Y (the epipoles misplaced by 1 deg): > 1 % of decisions must differ.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import analysis, flow_depth, logpolar as lp, matcher as m3, multibaseline as mb, m6_matcher as m6

BOTH5 = [o for o in range(-5, 6) if o]


def _lum(o):
    return m3.luminance(o.rgb)


def _turn_y(T, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return R @ T


def repro(run_dir, refs, out=None, git=None, log=print) -> dict:
    run = analysis.load_run(run_dir)
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    per = []
    for r in refs:
        ks = [r + o for o in BOTH5]
        ob = {k: flow_depth.load_observation(run, es[k]) for k in [r] + ks}
        h = ob[r].R @ [0, 0, -1.0]
        ts = [float((ob[k].c - ob[r].c) @ h) for k in ks]
        imgs = [_lum(ob[k]) for k in ks]
        t0 = time.perf_counter()
        a = mb.match_reference(_lum(ob[r]), imgs, ts, ob[r].model)
        t4 = time.perf_counter() - t0
        poses = m6.relative_poses(ob[r].c, ob[r].R, [(ob[k].c, ob[k].R) for k in ks])
        t0 = time.perf_counter()
        b = m6.match_reference(_lum(ob[r]), imgs, poses, ob[r].model)
        t6 = time.perf_counter() - t0
        c = m6.match_reference(_lum(ob[r]), imgs, [(A, _turn_y(T, 1.0)) for A, T in poses], ob[r].model)
        acc = (a["reason"] == 0) & (b["reason"] == 0)
        rel = np.abs(a["zeta"] - b["zeta"])[acc] / a["zeta"][acc]
        fin = np.isfinite(a["zeta"]) & np.isfinite(b["zeta"])
        d = {"r": r, "anchors": int(len(a["reason"])), "accepted_m4": float(np.mean(a["reason"] == 0)),
             "accepted_m6": float(np.mean(b["reason"] == 0)), "decisions_differ": int(np.sum(a["reason"] != b["reason"])),
             "max_rel_dzeta_accepted": float(rel.max()) if rel.size else None,
             "max_abs_dzeta_any_finite": float(np.max(np.abs(a["zeta"] - b["zeta"])[fin])),
             "coarse_ncc_max_diff": float(np.nanmax(np.abs(a["ncc_coarse"] - b["ncc_coarse"]))),
             "control_decisions_differ_fraction": float(np.mean(a["reason"] != c["reason"])),
             "control_accepted": float(np.mean(c["reason"] == 0)),
             "seconds_m4": round(t4, 1), "seconds_m6": round(t6, 1)}
        d["ok"] = d["decisions_differ"] == 0 and (d["max_rel_dzeta_accepted"] or 0.0) <= 1e-9
        d["control_fails"] = d["control_decisions_differ_fraction"] > 0.01
        per.append(d)
        log(f"[m6-t2] T2.0 r {r}: " + json.dumps(d))
    res = {"schema": "foveal-motion-m6-t2-repro-v1", "run": run["_dir"], "run_git": run["git"], "refs": list(refs),
           "window": "both5", "computed_by": git, "per_ref": per,
           "check": {"id": "T2.0", "ok": all(p["ok"] for p in per), "control_fails": all(p["control_fails"] for p in per)}}
    log(f"[m6-t2] T2.0 {'PASS' if res['check']['ok'] else 'FAIL'} (control {'fails' if res['check']['control_fails'] else 'DOES NOT FAIL'})")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "t2_repro.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ----------------------------------------------------------------------------- T2.1-T2.2: the crab runs

"""T2.1 (`oracle`): M4's oracle ladder O-M, generalised (evaluation side, truth). For each hit of reference r, the
truth point on the anchor's ray is projected into every window frame that sees it (M1.3's agreement test); its
correspondence there is exact, or moved by Gaussian noise of sigma native px along the frame's epipolar direction
(the direction its sample moves with zeta); zeta is then fitted to all frames by weighted Gauss-Newton through the
same map as the matcher (v_k = A_k (m - zeta T_k)). Reported: usable (95 % interval half-width <= 10 %), whole field
and fovea (< 2 deg), against #21's model (T0) for the same geometry.

T2.2 (`rgb`): M6's matcher (M4's frozen settings) on held-out references, evaluated as M4b: ranges with both5's frozen
interval scale (1.4875, docs/runs/issue-10/frozen.json); #5's calibration gate on M3's subset (hit, visible in
r - 5, interior, textured, 5-40 deg), reported per alpha. Controls, each of which must fail: the gate with every
step 20 % too long (M4's); and on alpha = 10 the crab's images matched with M4's aligned poses (the epipole put back
on the axis), which must fail the gate or lose at least half the coverage. The second replaces #21's foreign-geometry
sequence for T2 (stated in #22 before the RGB runs): it tests what T2 adds, that the epipole's place matters.
"""

FROZEN_BOTH5_SCALE = 1.4875
GATE_CONTROL_STEP = 1.2


def _frame_poses(ob, r, ks):
    return m6.relative_poses(ob[r].c, ob[r].R, [(ob[k].c, ob[k].R) for k in ks])


def _pix(model, A, T, m, zeta):
    v = (m - zeta[:, None] * T[None]) @ A.T                  # unit depth along r's axis; pixel_of needs unit vectors
    rc, inside = model.pixel_of(lp.unit(v))
    return rc, inside & (-v[:, 2] > m6.DEPTH_MIN)


def oracle(run_dir, refs, sigmas=(0.0, 0.1, 0.25, 0.5), seed=0, iters=12, noise_dir="epipolar", turn_deg=0.0, log=print) -> dict:
    from . import multiframe as mf
    run = analysis.load_run(run_dir)
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    rows = {s: [] for s in sigmas}
    for r in refs:
        ks = [r + o for o in BOTH5]
        looks = {k: analysis.load_look(run, es[k]) for k in [r] + ks}
        rec_r, R_r, c_r, model = looks[r]
        hit = np.flatnonzero(np.isfinite(rec_r.distance_m))
        P = rec_r.position_w[hit].astype(np.float64)
        d = lp.unit(rec_r.direction[hit].astype(np.float64)) @ R_r
        e = np.arctan2(np.hypot(d[:, 0], d[:, 1]), -d[:, 2])
        rng_t = np.linalg.norm(P - c_r, axis=-1)
        zeta_t = 1.0 / (rng_t * np.cos(e))
        m = d / np.cos(e)[:, None]
        poses = m6.relative_poses(c_r, R_r, [(looks[k][2], looks[k][1]) for k in ks])
        poses = [(A, _turn_y(T, turn_deg)) for A, T in poses]
        vis = np.stack([analysis._agree(looks[k][0], model, looks[k][1], looks[k][2], P)[0] for k in ks], 1)
        h = 1e-6
        obs, U = [], []
        for (A, T) in poses:
            rc0, in0 = _pix(model, A, T, m, zeta_t)
            rc1, _ = _pix(model, A, T, m, zeta_t + h)
            g = (rc1 - rc0) / h
            gn = np.linalg.norm(g, axis=-1)
            obs.append((rc0, in0, g / np.where(gn > 0, gn, 1)[:, None]))
        for s in sigmas:
            rs = np.random.default_rng(seed + 7919 * r)
            perp = lambda u: np.stack([-u[:, 1], u[:, 0]], -1)
            O = [(rc0 + (rs.normal(size=len(m)) * s)[:, None] * (u if noise_dir == "epipolar" else perp(u)), ok)
                 for rc0, ok, u in obs]
            sig = max(s, 1e-3)
            zeta = zeta_t * (1 + 0.05 * rs.normal(size=len(m)))         # started near the truth: O-M fits, it does not search
            zeta = np.clip(zeta, 1e-6, None)
            for _ in range(iters):
                Hs, gs = np.zeros(len(m)), np.zeros(len(m))
                for j, ((A, T), (o, ok), (_, _, u)) in enumerate(zip(poses, O, obs)):
                    rc, inside = _pix(model, A, T, m, zeta)
                    rc1, _ = _pix(model, A, T, m, zeta + h)
                    J = np.sum((rc1 - rc) / h * u, -1)
                    res = np.sum((o - rc) * u, -1)
                    w = vis[:, j] & ok & inside & np.isfinite(J) & np.isfinite(res)
                    Hs += np.where(w, J * J, 0.0)
                    gs += np.where(w, J * res, 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    step = np.where(Hs > 0, gs / Hs, 0.0)
                zeta = np.clip(zeta + step, 1e-6, None)
            with np.errstate(divide="ignore", invalid="ignore"):
                sz = np.where(Hs > 0, sig / np.sqrt(Hs), np.inf)
                lo, hi = zeta - mb.Z95 * sz, zeta + mb.Z95 * sz
                near, far = 1 / (hi * np.cos(e)), np.where(lo > 0, 1 / (lo * np.cos(e)), np.inf)
                rng = 1 / (zeta * np.cos(e))
                half = 0.5 * (far - near) / rng
            nvis = vis.sum(1)
            usable = (nvis >= 1) & np.isfinite(half) & (half <= 0.10)
            cover = (rng_t >= near) & (rng_t <= far)
            rows[s].append({"usable": usable, "e": np.degrees(e), "cover": cover[usable], "rel": np.abs(rng - rng_t)[usable] / rng_t[usable],
                            "rel_sigma": np.where(nvis >= 1, sz / zeta, np.inf)})
        log(f"[m6-t2] O-M r {r}: " + ", ".join(f"{s:g} px {np.mean(rows[s][-1]['usable']):.3f}" for s in sigmas))
    out = {}
    for s in sigmas:
        U = np.concatenate([x["usable"] for x in rows[s]])
        E = np.concatenate([x["e"] for x in rows[s]])
        C = np.concatenate([x["cover"] for x in rows[s]])
        RS = np.concatenate([x["rel_sigma"] for x in rows[s]])
        out[f"{s:g}"] = {"field_usable": float(U.mean()), "fovea_usable": float(U[E < 2].mean()),
                         "median_rel_sigma_fovea": float(np.median(RS[E < 2])), "median_rel_sigma_5_40": float(np.median(RS[(E >= 5) & (E < 40)])),
                         "fovea_1deg_usable": float(U[E < 1].mean()), "band_5_40_usable": float(U[(E >= 5) & (E < 40)].mean()),
                         "interval_coverage": float(C.mean()) if C.size else None, "hits": int(U.size)}
    return out


def rgb(run_dir, refs, out, git=None, log=print, poses_override=None) -> dict:
    """M6's matcher on the references (estimates saved as npz), evaluated with truth (see the T2.2 docstring)."""
    from . import flow_depth_eval as fe, m4_pipeline as mp
    run = analysis.load_run(run_dir)
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    G = fe.GATES["calibration"]
    rows, hits = [], 0
    for r in refs:
        ks = [r + o for o in BOTH5]
        ob = {k: flow_depth.load_observation(run, es[k]) for k in [r] + ks}
        poses = _frame_poses(ob, r, ks) if poses_override is None else poses_override(ob, r, ks)
        p = out / f"r{r:04d}.npz"
        t0 = time.perf_counter()
        if p.exists():
            z = dict(np.load(p))
        else:
            res = m6.match_reference(_lum(ob[r]), [_lum(ob[k]) for k in ks], poses, ob[r].model)
            n = ob[r].model.n
            inv = np.full(n * n, -1, dtype=np.int64)
            inv[ob[r].rec.raster_index] = np.arange(ob[r].rec.n_samples)
            keys = ("flat", "e", "phi", "q", "zeta", "sigma_zeta", "zeta_q", "n_frames", "ncc_coarse", "ncc_second",
                    "zeta_coarse", "texture", "support", "edge_dev_px", "px_per_zeta", "reason")
            z = {k: res[k] for k in keys}
            z["sample"] = inv[res["flat"]]
            np.savez_compressed(p, r=r, frames=np.array(ks), **z)
        secs = time.perf_counter() - t0
        t = mp.truth(run, r, z, other=r - 5)
        acc = z["reason"] == 0
        half = mb.Z95 * FROZEN_BOTH5_SCALE * z["sigma_zeta"]
        ce = np.cos(z["e"])

        def ranges(zeta, half):
            with np.errstate(divide="ignore", invalid="ignore"):
                rng = 1.0 / (zeta * ce)
                near = 1.0 / ((zeta + half) * ce)
                far = np.where(zeta - half > 0, 1.0 / ((zeta - half) * ce), np.inf)
                return rng, near, far, 0.5 * (far - near) / rng
        rng, near, far, relh = ranges(z["zeta"], half)
        usable = acc & np.isfinite(relh) & (relh <= flow_depth.USABLE_REL)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(rng - t["range"]) / t["range"]
        cov = (t["range"] >= near) & (t["range"] <= far)
        rngS, _, _, relhS = ranges(z["zeta"] / GATE_CONTROL_STEP, half / GATE_CONTROL_STEP)
        useS = acc & np.isfinite(relhS) & (relhS <= flow_depth.USABLE_REL)
        with np.errstate(invalid="ignore", divide="ignore"):
            relS = np.abs(rngS - t["range"]) / t["range"]
        e = np.degrees(z["e"])
        subset = t["hit"] & t["visible_in_other"] & ~t["edge"] & (z["texture"] >= m6.SETTINGS["texture_min"]) \
            & (e >= G["band_deg"][0]) & (e < G["band_deg"][1])
        hits += int(t["ecc_hits_deg"].size)
        rows.append({"usable": usable, "rel": rel, "cov": cov, "useS": useS, "relS": relS, "acc": acc, "hit": t["hit"],
                     "edge": t["edge"], "subset": subset, "fovea": e < 2, "reason": z["reason"]})
        log(f"[m6-t2] rgb r {r}: accepted {acc.mean():.3f}, usable {usable.mean():.3f}, fovea (1-2 deg) usable "
            f"{usable[e < 2].mean():.3f} ({secs:.0f} s)")
    cat = lambda k: np.concatenate([x[k] for x in rows])
    use, rel, cov, acc, hit, edge, sub = (cat(k) for k in ("usable", "rel", "cov", "acc", "hit", "edge", "subset"))
    inter = acc & hit & ~edge & np.isfinite(rel)
    g = mp._gate(sub, use & hit, rel, G)
    gS = mp._gate(sub, cat("useS") & hit, cat("relS"), G)
    reasons = cat("reason")
    res = {"schema": "foveal-motion-m6-t2-rgb-v1", "run": run["_dir"], "run_git": run["git"], "refs": list(refs),
           "window": "both5", "interval_scale": FROZEN_BOTH5_SCALE, "computed_by": git,
           "field_usable_of_all_hits": float(use.sum() / hits), "accepted_fraction": float(acc.mean()),
           "fovea_usable_1_2deg": float(use[cat("fovea")].mean()),
           "interval_coverage_interior": float(cov[inter].mean()) if inter.any() else None,
           "rejections": {name: float(np.mean(reasons == i)) for i, name in enumerate(m6.REASONS)},
           "gate": g, "gate_thresholds": G, "control_steps_x1.2": gS}
    log(f"[m6-t2] rgb: gate {'PASS' if g['gate_pass'] else 'FAIL'} {json.dumps(g)}; steps x1.2 control "
        f"{'fails' if not gS['gate_pass'] else 'DOES NOT FAIL'}")
    (out / "t2_rgb.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def aligned_override(ob, r, ks):
    """The control: M4's assumption (no rotation, steps along r's axis by the true baseline's axial component)."""
    h = ob[r].R @ [0, 0, -1.0]
    return m6.aligned_poses([float((ob[k].c - ob[r].c) @ h) for k in ks])


# M4's O-M, both5, refs 10 and 15 of the calibration room's geometry (docs/runs/issue-10/oracle_multiframe.txt)
M4_OM_BOTH5 = {"field_usable": {"0": 1.000, "0.1": 0.991, "0.25": 0.856, "0.5": 0.279},
               "band_5_40_usable": {"0": 1.000, "0.1": 0.999, "0.25": 0.928, "0.5": 0.293},
               "fovea_usable": {"0": 1.000, "0.1": 0.839, "0.25": 0.021, "0.5": 0.005}}


def oracle_repro(run_dir="runs/m3-calib-1024", log=print) -> dict:
    """T2.1a: the generalised O-M reproduces M4's O-M on M4's geometry within 0.01 at every rung and region. Control
    (must fail): every frame's translation turned 20 deg about r's +Y (a consistent other geometry: the epipoles move
    off the fovea). A first control, noise across the epipolar line, could not fail: O-M's usable fraction is the
    Fisher interval's, set by the geometry and the stated sigma, not by the drawn noise (found on its first run)."""
    def worst(o):
        return max(abs(o[sg][reg] - want) for reg, d in M4_OM_BOTH5.items() for sg, want in d.items())
    o = oracle(run_dir, [10, 15], log=log)
    oc = oracle(run_dir, [10, 15], turn_deg=20.0, log=log)
    res = {"id": "T2.1a", "max_diff": worst(o), "control_max_diff": worst(oc), "oracle": o, "control_oracle": oc}
    res["ok"], res["control_fails"] = res["max_diff"] <= 0.01, res["control_max_diff"] > 0.01
    log(f"[m6-t2] T2.1a {'PASS' if res['ok'] else 'FAIL'} max diff {res['max_diff']:.4f} (control "
        f"{'fails' if res['control_fails'] else 'DOES NOT FAIL'}: {res['control_max_diff']:.3f})")
    return res


def direction_breakdown(run_dir, est_dir, refs, bins=(0.5, 0.8, 0.95)) -> dict:
    """Diagnosis (observation and poses only): M6's acceptance by |cos| of the angle between an anchor's motion (its
    derotated displacement toward r + 5, in native px) and r's radial direction, the direction M4's windows are long in."""
    run = analysis.load_run(run_dir)
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    acc, n = np.zeros(len(bins) + 1), np.zeros(len(bins) + 1)
    for r in refs:
        z = dict(np.load(Path(est_dir) / f"r{r:04d}.npz"))
        ob_r, ob_k = flow_depth.load_observation(run, es[r]), flow_depth.load_observation(run, es[r + 5])
        T = ob_r.R.T @ (ob_k.c - ob_r.c)
        e, ph = z["e"], z["phi"]
        v = np.stack([np.tan(e) * np.cos(ph), np.tan(e) * np.sin(ph), -np.ones_like(e)], -1)
        b = -T
        rho2 = v[:, 0] ** 2 + v[:, 1] ** 2
        dq = (v[:, 0] * b[0] + v[:, 1] * b[1]) / rho2 - b[2] / v[:, 2]
        dphi = (v[:, 0] * b[1] - v[:, 1] * b[0]) / rho2
        model = ob_r.model
        gq = lp.native_px_per_q(model, z["q"], ph)
        gp = np.linalg.norm(model.pixel_of(lp.dir_of_e_phi(e, ph + 1e-5))[0]
                            - model.pixel_of(lp.dir_of_e_phi(e, ph - 1e-5))[0], axis=-1) / 2e-5
        c = np.abs(dq * gq) / np.hypot(dq * gq, dphi * gp)
        idx = np.digitize(c, bins)
        for i in range(len(bins) + 1):
            acc[i] += np.sum(z["reason"][idx == i] == 0)
            n[i] += np.sum(idx == i)
    edges = (0.0,) + tuple(bins) + (1.0,)
    return {"bins_abs_cos": [[edges[i], edges[i + 1]] for i in range(len(bins) + 1)],
            "accepted": (acc / np.maximum(n, 1)).tolist(), "share_of_anchors": (n / n.sum()).tolist()}
