"""M4a Stage 4 (#10): M3's synthetic fixtures as sequences, for the multi-baseline matcher (numpy + torch).

The reference frame r is at the origin; window frames at signed steps t_k = k x 0.0989565 m (the dolly's
step) along the eye axis (world +Y). Frames are rendered once per scene and reused. Truth: the range along
each of r's centre rays, zeta_true = 1 / (range cos e). Error in **equivalent native px**:
|zeta - zeta_true| x (native px per q at the anchor) x max |t_k|, the displacement error on the window's
longest step, comparable with M3's px of a.

Gates (median <= 0.1 px, p90 <= 0.25 px, >= 100 accepted) are #6's, fixed before M3; the corridor's bar
(more accepted than M3's 0.173 at a 1 m baseline) is #10's. Set after the first M4 trials and stated so in
#10: the periodic wall must pass the accuracy gate with <= 1 % of accepted over 1 px; the radial wall must
accept <= 1 % (an edge along a radial line carries no motion, whatever the number of frames).
"""
from __future__ import annotations

import time

import numpy as np

from . import flow_fixtures as fx, matcher as m3, multibaseline as mb

STEP = 0.0989565
NOISE = fx.NOISE
WINDOWS = {"before5": list(range(-5, 0)), "both5": [k for k in range(-5, 6) if k], "before10": list(range(-10, 0))}
GATE = dict(fx.GATE)
M3_CORRIDOR_1M = 0.173            # M3's accepted fraction on the parallel surfaces at B = 0.99 m (#6, afe3c58)


class Scene:
    def __init__(self, model, name, seed=0, foreign=False):
        self.m, self.name, self.seed = model, name, seed
        self.planes = fx.scene(name, seed)
        self.alt = fx.scene(name, seed + 100) if foreign else None
        self.cache = {}

    def frame(self, k, foreign=False):
        key = (k, foreign)
        if key not in self.cache:
            pl = self.alt if foreign else self.planes
            self.cache[key] = fx.render(self.m, pl, np.array([0.0, k * STEP, 0.0]), noise=NOISE, seed=self.seed + 500 + k)
        return self.cache[key]

    def truth(self):
        tr = fx.truth(self.m, self.planes, np.array([0.0, -1.0, 0.0]), np.zeros(3))
        labs = [p["label"] for p in self.planes]
        tr["labels"] = np.array([labs[i] if i >= 0 else "none" for i in tr["label"]])
        tr["edge"] = fx._edge_mask(tr["label"], self.m.n)
        return tr


def _stats(err, acc, sel):
    n = int(sel.sum())
    a = acc & sel
    ae = np.abs(err[a])
    return {"anchors": n, "accepted": int(a.sum()), "accepted_fraction": float(a.sum() / n) if n else None,
            "median_px": float(np.median(ae)) if ae.size else None, "p90_px": float(np.quantile(ae, 0.9)) if ae.size else None,
            "over_1px_of_accepted": float(np.mean(ae > 1)) if ae.size else None}


def gate_ok(st, max_over_1px=None):
    ok = st["accepted"] >= GATE["min_accepted"] and st["median_px"] <= GATE["median_px"] and st["p90_px"] <= GATE["p90_px"]
    return bool(ok and (max_over_1px is None or st["over_1px_of_accepted"] <= max_over_1px))


def run_m4(sc: Scene, window, *, ts_scale=1.0, shuffle=False, foreign_frames=(), all_foreign=False, settings=None):
    ks = WINDOWS[window]
    imgs = [sc.frame(k, foreign=all_foreign or (k in foreign_frames)) for k in ks]
    ts = np.array(ks, float) * STEP * ts_scale
    if shuffle:
        ts = np.random.default_rng(1).permutation(ts)
    t0 = time.perf_counter()
    r = mb.match_reference(sc.frame(0), imgs, list(ts), sc.m, settings)
    tr = sc.truth()
    f = r["flat"]
    z_true = 1.0 / (tr["range"][f] * np.cos(r["e"]))
    err = (r["zeta"] - z_true) * r["px_per_zeta"] / ts_scale          # equivalent px on the true steps
    acc = r["reason"] == 0
    out = {"fixture": sc.name, "window": window, "frames": ks, "ts_scale": ts_scale, "shuffle": shuffle,
           "foreign_frames": list(foreign_frames), "all_foreign": all_foreign, "seconds": round(time.perf_counter() - t0, 1),
           "reasons": {mb.REASONS[k]: int((r["reason"] == k).sum()) for k in range(len(mb.REASONS))},
           "all": _stats(err, acc, np.ones_like(acc)), "by_label": {},
           "median_frames_accepted": float(np.median(r["n_frames"][acc])) if acc.any() else None,
           "median_sigma_px_accepted": float(np.median(r["sigma_px"][acc])) if acc.any() else None}
    labs = tr["labels"][f]
    for L in sorted(set(labs)):
        out["by_label"][L] = _stats(err, acc, labs == L)
    if sc.name == "occlusion":
        e_ = tr["edge"][f]
        out["edge"], out["interior"] = _stats(err, acc, e_), _stats(err, acc, ~e_)
    return out, r, err


def run_m3_pair(sc: Scene, k_back):
    """M3's matcher on the pair (frame k_back, r) on the same anchors: a = frame k_back, b = r."""
    B = -k_back * STEP
    r = m3.match_pair(sc.frame(k_back), sc.frame(0), sc.m, B)
    tr = sc.truth()
    f = r["flat"]
    z_b = tr["range"][f] * np.cos(r["e_b"])
    d_true = np.log((z_b + B) / z_b)                                  # d = q_b - q_a = ln(z_a / z_b), z_a = z_b + B
    err = (r["d"] - d_true) * r["px_per_q_a"]
    acc = r["reason"] == 0
    labs = tr["labels"][f]
    out = {"fixture": sc.name, "pair": [k_back, 0], "B_m": B, "all": _stats(err, acc, np.ones_like(acc)), "by_label": {}}
    for L in sorted(set(labs)):
        out["by_label"][L] = _stats(err, acc, labs == L)
    return out


def run_all(model, log=print) -> dict:
    cases, gates, ctrl, m3cmp = [], [], [], []
    scenes = {n: Scene(model, n) for n in ("front", "tilted", "corridor", "periodic", "radial", "occlusion")}
    plan = [("front", "before5"), ("front", "both5"), ("tilted", "both5"), ("corridor", "before5"), ("corridor", "before10"),
            ("periodic", "before5"), ("periodic", "both5"), ("radial", "both5"), ("occlusion", "both5")]
    gated = {"front": ("front",), "tilted": ("tilted",), "corridor": ("parallel", "front_far"), "periodic": ("periodic",)}
    for name, win in plan:
        c, _, _ = run_m4(scenes[name], win)
        cases.append(c)
        if name in gated:
            for L in gated[name]:
                st = c["by_label"][L]
                ok = gate_ok(st, 0.01 if name == "periodic" else None)
                if name == "corridor" and win == "before10" and L == "parallel":
                    ok = ok and st["accepted_fraction"] > M3_CORRIDOR_1M
                gates.append({"fixture": name, "window": win, "label": L, "ok": ok, **{k: st[k] for k in
                              ("accepted_fraction", "median_px", "p90_px", "over_1px_of_accepted")}})
        if name == "radial":
            st = c["all"]
            gates.append({"fixture": name, "window": win, "label": "radial", "ok": st["accepted_fraction"] <= 0.01,
                          "accepted_fraction": st["accepted_fraction"], "median_px": st["median_px"], "p90_px": st["p90_px"],
                          "over_1px_of_accepted": st["over_1px_of_accepted"]})
        if name == "occlusion":
            st = c["interior"]
            gates.append({"fixture": name, "window": win, "label": "interior", "ok": gate_ok(st), **{k: st[k] for k in
                          ("accepted_fraction", "median_px", "p90_px", "over_1px_of_accepted")}})
        log(f"[m4-fixtures] {name} {win} ({c['seconds']} s): " + "; ".join(
            f"{L} acc {v['accepted_fraction']:.3f} med {v['median_px'] if v['median_px'] is None else round(v['median_px'], 3)}"
            f" p90 {v['p90_px'] if v['p90_px'] is None else round(v['p90_px'], 3)}" for L, v in c["by_label"].items()))
    # M3 on the equivalent pair, same anchors
    for name, kb in (("front", -5), ("tilted", -5), ("corridor", -5), ("corridor", -10), ("periodic", -5), ("occlusion", -5)):
        o = run_m3_pair(scenes[name], kb)
        m3cmp.append(o)
        log(f"[m4-fixtures] M3 pair ({kb}, 0) on {name}: " + "; ".join(
            f"{L} acc {v['accepted_fraction']:.3f}" for L, v in o["by_label"].items()))
    # controls: each must fail the gate on the front wall, both5
    for lab, kw in (("frame steps shuffled", {"shuffle": True}), ("steps 20 % too long", {"ts_scale": 1.2}),
                    ("every window frame from another scene", {"all_foreign": True})):
        c, _, _ = run_m4(scenes["front"], "both5", **kw)
        ctrl.append({"control": lab, "fails": not gate_ok(c["by_label"]["front"]), "stats": c["by_label"]["front"]})
        log(f"[m4-fixtures] control {'fails' if ctrl[-1]['fails'] else 'DOES NOT FAIL'}: {lab}: {c['by_label']['front']}")
    # robustness (reported, not gated): one foreign frame in the window
    c, r, _ = run_m4(scenes["front"], "both5", foreign_frames=(3,))
    robust = {"one foreign frame (k = 3) in both5": c["by_label"]["front"], "median_frames_accepted": c["median_frames_accepted"]}
    log(f"[m4-fixtures] robustness, one foreign frame: {robust}")
    return {"gates": gates, "controls": ctrl, "m3_same_anchors": m3cmp, "robustness": robust, "cases": cases,
            "ok": all(g["ok"] for g in gates), "controls_fail": all(x["fails"] for x in ctrl)}
