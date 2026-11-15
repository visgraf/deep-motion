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


GATE_CONTROL_STEP = 1.2            # the gate's control: every step 20 % too long (zeta / 1.2, ranges x 1.2), as M3's baseline x 1.2
COV_CONTROL_SCALE = fe.COV_CONTROL_SCALE    # the interval check's control: intervals x 0.5 must cover < COV_MIN
COV_MIN = fe.COV_MIN
ESTIMATOR_FILES = ("motion/multibaseline.py", "motion/m4_pipeline.py", "motion/matcher.py", "motion/logpolar.py",
                   "motion/flow_depth.py")
REPO = Path(__file__).resolve().parents[1]


def _nos(st):
    """Settings without the interval scale (which only sizes the intervals; no decision reads it), JSON-normalised."""
    return json.loads(json.dumps({k: v for k, v in st.items() if k != "interval_scale"}))


def estimator_sha256() -> dict:
    return {f: flow_depth._sha(REPO / f) for f in ESTIMATOR_FILES}


def check_estimator(recorded: dict):
    now = estimator_sha256()
    bad = [f for f in recorded if now.get(f) != recorded[f]]
    if bad:
        raise SystemExit(f"[m4] refused: the frozen estimator's files changed: {bad}")


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
    if _nos(cfg["settings"]) != _nos(s):
        raise SystemExit("[m4] refused: settings differ from this result's earlier estimates")
    if (out / "freeze.json").exists():
        fr = _verify(out)
        if "estimator_sha256" in fr:
            check_estimator(fr["estimator_sha256"])
            if _nos(fr["frozen_settings"]) != _nos(s):
                raise SystemExit("[m4] refused: settings differ from the imported frozen ones")
    obs = {}
    get = lambda k: obs.setdefault(k, flow_depth.load_observation(run, es[k]))
    for r in refs:
        p = out / f"r{r:04d}.npz"
        if p.exists():
            raise SystemExit(f"[m4] refused: {p} exists")
        t0 = time.perf_counter()
        ks = [r + o for o in WINDOWS[window]]
        ob = get(r)
        al = flow_depth.alignment({k: (get(k).c, get(k).R) for k in [r] + ks}, [(min(k, r), max(k, r)) for k in ks])  # #5's gate is earlier -> later
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
    (d / "config_frozen.json").write_text((d / "config.json").read_text())   # config.json may gain held-out refs later
    files = {p.name: flow_depth._sha(p) for p in [d / "config_frozen.json", d / "calibration.json", *sorted(d.glob("r*.npz"))]}
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


def _gate(sub, use, rel, G):
    a = sub & use
    g = {"subset": int(sub.sum()), "accepted_usable": int(a.sum()), "coverage_of_subset": float(a.sum() / max(int(sub.sum()), 1)),
         "median_rel": float(np.median(rel[a])) if a.any() else None,
         "p90_rel": float(np.quantile(rel[a], 0.9)) if a.any() else None,
         "over_50pct": float(np.mean(rel[a] > 0.5)) if a.any() else None}
    g["gate_pass"] = bool(g["median_rel"] is not None and g["median_rel"] <= G["median"] and g["p90_rel"] <= G["p90"]
                          and g["coverage_of_subset"] >= G["coverage"]
                          and (G["outlier50"] is None or g["over_50pct"] <= G["outlier50"]))
    return g


def evaluate(out: Path, window: str, m3_scale: float, gate: str = "calibration", exclude=(), log=print) -> dict:
    """Held-out: M4's ranges with the frozen scale; M3's frozen matcher on the pair (r - 5, r) with M3's frozen
    scale, on the same anchors; #5's gate (calibration or classroom) on M3's subset (hit, visible in r - 5,
    interior, textured, in the gate's band) for both; coverage of the 95 % intervals. Reference frames in exclude
    (truth opened earlier) are left out. Controls, each of which must fail: the gate with every step 20 % too long,
    the interval check with intervals x 0.5."""
    d = Path(out) / window
    fr = _verify(d)
    cfg = json.loads((d / "config.json").read_text())
    frozen_cfg = json.loads((d / "config_frozen.json").read_text())
    if not (_nos(fr["frozen_settings"]) == _nos(cfg["settings"]) == _nos(frozen_cfg["settings"])):
        raise SystemExit("[m4] refused: settings differ from the frozen ones")
    if "estimator_sha256" in fr:
        check_estimator(fr["estimator_sha256"])
    scale = fr["frozen_settings"]["interval_scale"]
    run = analysis.load_run(cfg["run"])
    es = {e["k"]: e for e in run["looks"]["ahead"]}
    G = fe.GATES[gate]
    exclude = sorted(int(x) for x in exclude)
    rows, late = [], held_out_estimates_are_new(d, fr)
    for r in sorted(set(cfg["refs"]) - set(fr["calibration_refs"]) - set(exclude)):
        z = dict(np.load(d / f"r{r:04d}.npz"))
        ob = flow_depth.load_observation(run, es[r])
        t = truth(run, r, z, other=r - M3_PAIR)
        acc = z["reason"] == 0
        half = mb.Z95 * scale * z["sigma_zeta"]
        ce = np.cos(z["e"])

        def ranges(zeta, half):
            with np.errstate(divide="ignore", invalid="ignore"):
                rng = 1.0 / (zeta * ce)
                near = 1.0 / ((zeta + half) * ce)
                far = np.where(zeta - half > 0, 1.0 / ((zeta - half) * ce), np.inf)
                return rng, near, far, 0.5 * (far - near) / rng
        rng, near, far, relh = ranges(z["zeta"], half)
        usable = acc & np.isfinite(relh) & (relh <= flow_depth.USABLE_REL)
        rel = np.abs(rng - t["range"]) / t["range"]
        cov = (t["range"] >= near) & (t["range"] <= far)
        _, nearC, farC, _ = ranges(z["zeta"], half * COV_CONTROL_SCALE)
        covC = (t["range"] >= nearC) & (t["range"] <= farC)
        rngS, _, _, relhS = ranges(z["zeta"] / GATE_CONTROL_STEP, half / GATE_CONTROL_STEP)
        useS = acc & np.isfinite(relhS) & (relhS <= flow_depth.USABLE_REL)
        relS = np.abs(rngS - t["range"]) / t["range"]
        e = np.degrees(z["e"])
        subset = t["hit"] & t["visible_in_other"] & ~t["edge"] & (z["texture"] >= cfg["settings"]["texture_min"]) \
            & (e >= G["band_deg"][0]) & (e < G["band_deg"][1])
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
                     "cov": cov, "cov3": cov3, "covC": covC, "useS": useS, "relS": relS, "acc": acc, "acc3": acc3,
                     "hit": t["hit"], "edge": t["edge"], "subset": subset, "e": e, "fovea": e < 2})
        log(f"[m4] held-out r {r}: M4 usable {usable.mean():.3f}, M3 usable {use3.mean():.3f}")
    if not rows:
        raise SystemExit("[m4] refused: no held-out reference frames to evaluate")
    cat = lambda k: np.concatenate([x[k] for x in rows])
    hits = sum(x["n_hits"] for x in rows)
    hit, edge, sub = cat("hit"), cat("edge"), cat("subset")
    res = {"schema": "foveal-motion-m4-eval-v2", "window": window, "gate_name": gate, "gate_thresholds": G,
           "interval_scale": scale, "m3_scale": m3_scale, "held_out_refs": [x["r"] for x in rows], "excluded_refs": exclude,
           "imported_from": fr.get("imported_from"), "late_estimates_hashed_now": {n: flow_depth._sha(d / n) for n in late}}
    for tag, U, REL, COV, ACC in (("M4", "usable", "rel", "cov", "acc"), ("M3", "use3", "rel3", "cov3", "acc3")):
        use, rel, cov, acc = cat(U), cat(REL), cat(COV), cat(ACC)
        inter = acc & hit & ~edge & np.isfinite(rel)
        res[tag] = {"field_usable_of_all_hits": float(use.sum() / hits), "accepted_fraction": float(acc.mean()),
                    "interval_coverage_interior": float(cov[inter].mean()) if inter.any() else None,
                    "usable_interior_median_rel": float(np.median(rel[use & hit & ~edge])) if (use & hit & ~edge).any() else None,
                    "usable_interior_p90_rel": float(np.quantile(rel[use & hit & ~edge], 0.9)) if (use & hit & ~edge).any() else None,
                    "fovea_usable": float(use[cat("fovea")].mean()), "gate": _gate(sub, use & hit, rel, G)}
    inter = cat("acc") & hit & ~edge & np.isfinite(cat("rel"))
    gS = _gate(sub, cat("useS") & hit, cat("relS"), G)
    covC = float(cat("covC")[inter].mean()) if inter.any() else None
    res["checks"] = [
        {"id": "M4.GATE", "what": f"#5's {gate} gate on M3's subset ({G['band_deg'][0]:g}-{G['band_deg'][1]:g} deg): median <= "
         f"{G['median']}, p90 <= {G['p90']}, coverage >= {G['coverage']}"
         + (f", > 50 % errors <= {G['outlier50']}" if G["outlier50"] is not None else ""),
         "ok": res["M4"]["gate"]["gate_pass"], "control": f"every step x {GATE_CONTROL_STEP}", "control_fails": not gS["gate_pass"],
         "control_detail": gS},
        {"id": "M4.COVER", "what": f"interior coverage of the 95 % intervals >= {COV_MIN}",
         "ok": bool(res["M4"]["interval_coverage_interior"] is not None and res["M4"]["interval_coverage_interior"] >= COV_MIN),
         "control": f"intervals x {COV_CONTROL_SCALE}", "control_fails": bool(covC is not None and covC < COV_MIN),
         "control_detail": {"coverage": covC}}]
    (d / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ----------------------------------------------------------------------------- the frozen estimator and its import (M4b)

FROZEN = REPO / "docs" / "runs" / "issue-10" / "frozen.json"


def _rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def write_frozen(sources: dict, path: Path = FROZEN, git=None) -> dict:
    """The frozen M4 estimator: the hashes of its files, its settings, and per window the interval scale with the
    calibration-room freeze it comes from (sources: window -> directory holding freeze.json, config_frozen.json,
    calibration.json). Each source must agree with the others and with the code's settings."""
    win = {}
    for w, src in sources.items():
        src = Path(src)
        fr = json.loads((src / "freeze.json").read_text())
        for f in ("config_frozen.json", "calibration.json"):
            if flow_depth._sha(src / f) != fr["sha256"][f]:
                raise SystemExit(f"[m4] refused: {src / f} differs from its freeze")
        if _nos(fr["frozen_settings"]) != _nos(mb.SETTINGS):
            raise SystemExit(f"[m4] refused: {w}'s frozen settings differ from the code's")
        win[w] = {"interval_scale": fr["frozen_settings"]["interval_scale"], "offsets": WINDOWS[w],
                  "source": _rel(src), "source_freeze_sha256": flow_depth._sha(src / "freeze.json"),
                  "calibration_refs": fr["calibration_refs"]}
    rec = {"schema": "foveal-motion-m4-frozen-v1", "estimator_sha256": estimator_sha256(), "settings": _nos(mb.SETTINGS),
           "windows": win, "computed_by": git}
    Path(path).write_text(json.dumps(rec, indent=1, default=float))
    return rec


def import_freeze(run: dict, out: Path, window: str, frozen: Path = FROZEN, git=None) -> dict:
    """Start a result whose estimator and interval scale come from the frozen record: config.json (settings, no
    references yet), config_frozen.json and freeze.json (no calibration references; every reference is held out).
    Refuses if the estimator's files, the settings or the source freeze differ from the record."""
    rec = json.loads(Path(frozen).read_text())
    check_estimator(rec["estimator_sha256"])
    if _nos(mb.SETTINGS) != rec["settings"]:
        raise SystemExit("[m4] refused: the code's settings differ from the frozen record")
    w = rec["windows"][window]
    src = REPO / w["source"]           # absolute paths stay absolute
    if flow_depth._sha(src / "freeze.json") != w["source_freeze_sha256"]:
        raise SystemExit(f"[m4] refused: {src / 'freeze.json'} differs from the frozen record")
    d = Path(out) / window
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"[m4] refused: {d} is not empty")
    d.mkdir(parents=True, exist_ok=True)
    s = dict(mb.SETTINGS)
    cfg = {"schema": "foveal-motion-m4-estimate-v1", "run": run["_dir"], "run_git": run["git"], "window": window,
           "offsets": WINDOWS[window], "settings": s, "computed_by": git, "refs": [], "inputs_sha256": {}}
    (d / "config.json").write_text(json.dumps(cfg, indent=1, default=float))
    (d / "config_frozen.json").write_text(json.dumps(cfg, indent=1, default=float))
    fr = {"schema": "foveal-motion-m4-freeze-v1", "frozen_settings": dict(s, interval_scale=w["interval_scale"]),
          "sha256": {"config_frozen.json": flow_depth._sha(d / "config_frozen.json")}, "calibration_refs": [],
          "held_out_refs": None, "estimator_sha256": rec["estimator_sha256"],
          "imported_from": {"frozen": _rel(frozen), "frozen_sha256": flow_depth._sha(Path(frozen)),
                            "source": w["source"], "source_freeze_sha256": w["source_freeze_sha256"]}}
    (d / "freeze.json").write_text(json.dumps(fr, indent=1, default=float))
    return fr


def every_refs(ks, window, every, exclude=()):
    """Reference frames on a stride (multiples of `every`) whose window fits, without the excluded ones."""
    return [r for r in references(ks, window) if r % every == 0 and r not in set(exclude)]


def run_calibration(run_dir, out, windows, m3_scale, git=None, log=print) -> dict:
    """The calibration-room protocol of Stage 5 for each window: estimate the calibration references, calibrate and
    freeze (unless frozen), estimate the held-out references, evaluate with #5's calibration gate."""
    run = analysis.load_run(run_dir)
    ks = sorted(e["k"] for e in run["looks"]["ahead"])
    res = {}
    for w in windows:
        cp = Path(out) / w / "config.json"
        have = set(json.loads(cp.read_text())["refs"]) if cp.exists() else set()
        refs = references(ks, w)
        cal = [r for r in refs if is_calibration(r) and r not in have]
        held = [r for r in refs if not is_calibration(r) and r not in have]
        if cal:
            estimate(run, w, cal, out, git=git, log=log)
        fz = Path(out) / w / "freeze.json"
        c = json.loads((Path(out) / w / "calibration.json").read_text()) if fz.exists() else calibrate(out, w)
        log(f"[m4] {w}: interval scale {c['interval_scale']:.3f} from {c['n']} anchors")
        if held:
            estimate(run, w, held, out, git=git, log=log)
        res[w] = evaluate(out, w, m3_scale, "calibration", log=log)
    return res


def run_import(run_dir, out, windows, m3_scale, every, exclude, gate="classroom", frozen=FROZEN, git=None, log=print) -> dict:
    """A run judged with the frozen estimator and scale (the Classroom): import the freeze, estimate every
    `every`-th reference frame truth-free (excluded ones are not estimated), evaluate with the given gate."""
    run = analysis.load_run(run_dir)
    ks = sorted(e["k"] for e in run["looks"]["ahead"])
    res = {}
    for w in windows:
        d = Path(out) / w
        if not (d / "freeze.json").exists():
            import_freeze(run, out, w, frozen, git=git)
        have = set(json.loads((d / "config.json").read_text())["refs"])
        refs = [r for r in every_refs(ks, w, every, exclude) if r not in have]
        log(f"[m4] {w}: {len(refs)} reference frames to estimate: {refs}")
        if refs:
            estimate(run, w, refs, out, git=git, log=log)
        res[w] = evaluate(out, w, m3_scale, gate, exclude=exclude, log=log)
    return res
