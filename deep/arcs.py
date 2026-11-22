"""Stage 14 (#34): N on curved paths in the Classroom. Checks G0 and G2 as accepted by Luiz on 2026-11-21 (G1 is
deep/arcfixtures.py's).

- **G0 (no regression on the line).** N's per-frame path forced on R3's straight path, at stage 11's 12 references,
  against N's line path from stage 11's cache (N1024 and its every-anchor reasons):
  - at least 99.9 % of anchors take the same decision;
  - the accepted zeta of the anchors both accept agree to 1e-9, relative;
  - the fused cells right and coverage (stage 9's fusion and metric) agree to 0.05 points.
  Control: the per-frame path given the mirrored path's poses must fall below 50 % right.
- **G2 (N on the turn), C3's form.** On R3's turn at 1024 spp, at its 12 references (every window touches the arc, so
  N takes its per-frame path at each), N covers at least twice B1024's coverage at cells right at most 2 points below
  B1024's. Control: N given the mirrored path's poses must fall below 50 % right.
- **C0 on the turn (consistency).** B128 and the perfect matcher reproduce stage 9's turn numbers exactly (#22). Control:
  B on the 1024-spp renders must not.
- **Reported:** N on the turn by sector of psi, beside N on the straight path (stage 11, #26), and the perfect matcher.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import assess as asx
from . import confirm as cf
from . import standard as sd
from . import straight as st

G0_AGREE, G0_ZETA_REL, G0_POINTS = 0.999, 1e-9, 0.0005
GEN = "N1024, per-frame path forced"
GEN_M = "N1024 mirrored, per-frame path forced"


def _rows_by_px(o, px_key="px_all", val_key="reason_all"):
    return {tuple(p): v for p, v in zip(np.asarray(o[px_key]).tolist(), np.asarray(o[val_key]).tolist())}


def g0(paths, cache, log=print, dev=None):
    t0 = time.perf_counter()
    looks128, truth, fan, refs, _ = paths["128"]
    looks1024 = paths["1024"][0]
    R, per_ref = st.collect(paths, None, cache, rows=["N1024"], log=log, dev=dev, extra=(st.N_ALL,))
    cdir = Path(cache) if cache else None
    safe = lambda name: "".join(ch if ch.isalnum() or ch in "-_.@" else "_" for ch in name)
    cp = lambda name, j: (cdir / safe(name) / f"r{j:04d}.npz") if cdir else None
    gen = {j: st._cache(cp(GEN, j), lambda j=j: st.n_reference(looks1024, truth, j, fan, keep_all=True, general=True, dev=dev))
           for j in refs}
    log(f"[stage14 G0] {GEN}: {sum(o['accepted'] for o in gen.values())} accepted ({time.perf_counter() - t0:.0f} s)")
    genm = {j: st._cache(cp(GEN_M, j), lambda j=j: st.n_reference(looks1024, truth, j, fan, mirror=True, general=True, dev=dev))
            for j in refs}
    log(f"[stage14 G0] {GEN_M} ({time.perf_counter() - t0:.0f} s)")
    n_anchor = n_same = 0
    zrel = 0.0
    n_common = 0
    for j in refs:
        a, b = _rows_by_px(R[st.N_ALL][j]), _rows_by_px(gen[j])
        keys = a.keys() & b.keys()
        n_anchor += len(a)
        n_same += sum((a[k] == 0) == (b[k] == 0) for k in keys)
        c = looks128[j][1]
        za = {tuple(p): 1.0 / np.linalg.norm(x - c) for p, x in zip(np.asarray(R["N1024"][j]["px_acc"]).tolist(), R["N1024"][j]["X"])}
        zb = {tuple(p): 1.0 / np.linalg.norm(x - c) for p, x in zip(np.asarray(gen[j]["px_acc"]).tolist(), gen[j]["X"])}
        common = za.keys() & zb.keys()
        n_common += len(common)
        if common:
            zrel = max(zrel, max(abs(za[k] - zb[k]) / za[k] for k in common))
    surf = st.surface_band(per_ref, truth, looks128, None)
    s_line, _ = st.score(R["N1024"], per_ref, truth, fan, surf, None)
    s_gen, _ = st.score(gen, per_ref, truth, fan, surf, None)
    s_mir, _ = st.score(genm, per_ref, truth, fan, surf, None)
    agree = n_same / n_anchor if n_anchor else None
    ok = (agree is not None and agree >= G0_AGREE and zrel <= G0_ZETA_REL
          and abs(s_gen["right"] - s_line["right"]) <= G0_POINTS and abs(s_gen["coverage"] - s_line["coverage"]) <= G0_POINTS)
    return {"check": "G0: N's per-frame path forced on the straight path gives N's line path back (>= 99.9 % same decisions, "
                     "zeta to 1e-9, cells right and coverage to 0.05 points)",
            "ok": bool(ok), "anchors": n_anchor, "same_decision": agree, "accepted_both": n_common, "max_rel_zeta": zrel,
            "line": {k: s_line[k] for k in ("cells", "right", "coverage")},
            "per_frame": {k: s_gen[k] for k in ("cells", "right", "coverage")},
            "all_per_frame": all(o["general"] for o in gen.values()),
            "control": "the per-frame path given the mirrored path's poses must fall below 50 % right",
            "control_fails": (s_mir["right"] is None) or (s_mir["right"] < st.C3_CONTROL),
            "control_value": {k: s_mir[k] for k in ("cells", "right", "coverage")}, "seconds": round(time.perf_counter() - t0, 1)}


def sectors_of(row, per_ref, truth, fan, looks):
    """All of a row's fused cells, by sector of psi (S2's gate as a report)."""
    Xs = [o["X"] for o in row.values()]
    rs = [np.full(len(o["X"]), j) for j, o in row.items()]
    X, ref, Xf, nref, dom, right, cell_of = asx.fuse_and_judge(Xs, rs, per_ref, truth, fan)
    return sd.gate(right, cf.cell_psi(Xf, dom, looks))


def g2(paths, cache, assess_record=None, rec11=None, log=print, dev=None):
    t0 = time.perf_counter()
    looks128, truth, fan, refs, _ = paths["128"]
    R, per_ref = st.collect(paths, None, cache, rows=["perfect", "N1024", "B1024", cf.MIRROR], log=log, dev=dev)
    surf = st.surface_band(per_ref, truth, looks128, None)
    rows = {}
    pcov = None
    for name in ("perfect", "B128", "B1024", "N1024", cf.MIRROR):
        if name in R:
            rows[name], cov = st.score(R[name], per_ref, truth, fan, surf, None, pcov)
            if name == "perfect":
                pcov = cov
    all_gen = all(o.get("general", False) for o in R["N1024"].values())
    g = lambda name, k: rows.get(name, {}).get(k)
    out = {"references": refs, "rows": rows, "checks": [], "n_all_per_frame": all_gen}
    if assess_record is not None:
        t = assess_record["numbers"]["turn"]
        want = {"B right": t["real"]["right"], "B coverage": t["real"]["coverage"],
                "perfect right": t["perfect"]["right"], "perfect coverage": t["perfect"]["coverage"]}
        got = {"B right": g("B128", "right"), "B coverage": g("B128", "coverage"),
               "perfect right": g("perfect", "right"), "perfect coverage": g("perfect", "coverage")}
        ok = all(got[k] is not None and abs(got[k] - want[k]) <= 1e-12 for k in want)
        ctl = {"B right": g("B1024", "right"), "B coverage": g("B1024", "coverage")}
        out["checks"].append({"check": "C0 (turn): B128 and the perfect matcher reproduce stage 9's turn numbers exactly",
                              "ok": bool(ok), "want": want, "got": got,
                              "control": "B on the 1024-spp renders must not reproduce them",
                              "control_fails": any(ctl[k] is None or abs(ctl[k] - want[k]) > 1e-9 for k in ctl),
                              "control_value": ctl})
    n_r, n_c, b_r, b_c, m_r = g("N1024", "right"), g("N1024", "coverage"), g("B1024", "right"), g("B1024", "coverage"), g(cf.MIRROR, "right")
    out["checks"].append({"check": "G2: on the turn at 1024 spp, N covers >= 2 x B1024's coverage at cells right >= B1024's - 2 points",
                          "ok": None if None in (n_r, n_c, b_r, b_c) else bool(n_c >= st.C3_FACTOR * b_c and n_r >= b_r - st.C3_POINTS),
                          "values": {"N right": n_r, "N coverage": n_c, "B1024 right": b_r, "B1024 coverage": b_c},
                          "n_all_per_frame": all_gen,
                          "control": "N given the mirrored path's poses must fall below 50 % right",
                          "control_fails": (m_r is None) or (m_r < st.C3_CONTROL), "control_value": {"right": m_r}})
    out["n_by_sector"] = sectors_of(R["N1024"], per_ref, truth, fan, looks128)
    if rec11 is not None:
        out["n_straight_stage11"] = {k: rec11["rows"]["N1024"]["full"][k] for k in ("cells", "right", "coverage")}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


# --------------------------------------------------------------------------------------- the dry run's arc room

def synthetic_arc_room(K=111, step=0.01, radius=1.0, W=720, ss=2, seed=0, log=print, cache=None):
    """#26's synthetic room of M4's fixture planes (deep/straight.py synthetic_room's planes), ray-cast exactly along an
    arc of `radius` (start (0, -0.55), heading +Y, turning left, the head on the tangent), at the two noise levels of
    DRY_NOISE. For G2's dry run only (Chat sandbox, not gated). Returns paths as stage 11's."""
    from motion import flow_fixtures as fx
    from . import arcfixtures as af
    from . import nfixtures as nf
    from . import sensor as sn
    tex = [fx._texture_noise(seed + 30 + i) for i in range(7)]
    X, Y, Z = np.eye(3)
    planes = [fx._plane([0, 0, -1.6], Z, X, tex[0], "floor"), fx._plane([0, 0, 1.4], -Z, X, tex[1], "ceiling"),
              fx._plane([-2.0, 0, 0], X, Y, tex[2], "left"), fx._plane([2.5, 0, 0], -X, Y, tex[3], "right"),
              fx._plane([0, 4.0, 0], -Y, X, tex[4], "front"), fx._plane([0, -3.5, 0], Y, X, tex[5], "back"),
              fx._plane([-0.6, 1.8, -0.2], -Y, X, tex[6], "board", bounds=(-0.4, 0.4, -0.35, 0.35))]
    fan = sn.SlitFan(width=W)
    H = W // 2
    start = np.array([0.0, -0.55, 0.0])
    poses = []
    for k in range(K):
        c, R = af.arc_pose(k * step / nf.STEP, radius=radius)                    # arc length k step
        poses.append((step * k, c + start, R))
    imgs = None
    if cache is not None and Path(cache).exists():
        imgs = np.load(cache, allow_pickle=True)["obj"].item()
    looks = {"128": [], "1024": []}
    t0 = time.perf_counter()
    store = {"128": [], "1024": []}
    for k, (s_, c, R) in enumerate(poses):
        if imgs is not None:
            for name in st.DRY_NOISE:
                looks[name].append((s_, c, R, imgs[name][k], None))
            continue
        clean = nf.render(planes, c, W=W, ss=ss, noise=0.0, R=R)
        for name, sg in st.DRY_NOISE.items():
            img = (clean + np.random.default_rng(seed + (1000 if name == "128" else 5000) + k).normal(scale=sg, size=clean.shape)).astype(np.float32)
            looks[name].append((s_, c, R, img, None))
            store[name].append(img)
    if imgs is None:
        log(f"[stage14 dry run] {K} panorama looks along the arc at two noise levels ({time.perf_counter() - t0:.0f} s)")
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, obj=np.array(store, dtype=object))
    refs = asx.references(K)
    v, u = np.mgrid[0:H, 0:W]
    lon = (u + 0.5) / W * 2 * np.pi - np.pi
    lat = (0.5 - (v + 0.5) / H) * np.pi
    dl = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1).reshape(-1, 3)
    truth = {}
    L = looks["128"]
    for j in refs:
        d = dl @ L[j][2].T
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
    return {name: (looks[name], truth, fan, refs, vis_fn) for name in ("128", "1024")}
