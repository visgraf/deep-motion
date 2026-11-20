"""Stage 5b.2 (#7), Code: why turn90 fails F1 at 3600 wide (diagnosis only; deep/ is not changed).

    .venv/bin/python docs/runs/issue-7/code/f1/f1_diag.py prepare runs/s5b2c-turn runs/s5b2-f1diag/turn.npz
    .venv/bin/python docs/runs/issue-7/code/f1/f1_diag.py analyse runs/s5b2c-turn runs/s5b2-f1diag/turn.npz
    .venv/bin/python docs/runs/issue-7/code/f1/f1_diag.py probe runs/s5b2c-turn runs/s5b2-f1diag/turn.npz
    .venv/bin/python docs/runs/issue-7/code/f1/f1_diag.py nadir runs/s5b2c-turn runs/s5b2-f1diag/turn.npz

prepare: the cells exactly as deep/fusion.py's selftest makes them (cell 0.02, hit_stride 23, max_cells 200000,
seed 7), then fuse(looks, P, fan) at no noise, as F1 does; saves P, X0, vis.
analyse: the cells that miss F1 (error > 1e-9 of the nearest seeing look's range), where they are, which looks see
them, and a traced copy of fuse's loop on them alone (the same steps, the same line search) to tell a stall from slow
convergence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "external" / "fov-render"))

from deep import fusion as fu  # noqa: E402
from deep import oracle as orc  # noqa: E402
from deep import sensor as sn  # noqa: E402


def prepare(run, out):
    looks = fu.load_looks([run])
    fan = sn.SlitFan(width=3600)
    P = fu.cells(looks, 0.02, 23)
    rng = np.random.default_rng(7)
    n_all = len(P)
    if len(P) > 200000:
        P = P[np.sort(rng.choice(len(P), size=200000, replace=False))]
    X0, H, vis = fu.fuse(looks, P, fan)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, P=P, X0=X0, H=H, vis=vis, n_all=n_all)
    print(f"[f1] {n_all} cells binned, {len(P)} kept; saved {out}")


def traced(looks, P, vis, iters=60, halvings=40):
    """fuse's loop at no noise for these cells, recording per iteration the cost, the step taken and whether the line
    search gave up (lam = 0, so the step is 0 and the cell leaves `act` as if converged)."""
    K = len(looks)
    poses = [(lk[1], lk[2]) for lk in looks]
    obs = [fu._derivs(P, *poses[k])[:2] for k in range(K)]
    first = np.argmax(vis, 1)
    c0 = np.stack([poses[k][0] for k in range(K)])[first]
    X = c0 + 1.05 * (P - c0)
    trace = []

    def cost(Xa):
        c = np.zeros(len(Xa))
        for k in range(K):
            lon, lat, _, _ = fu._derivs(Xa, *poses[k])
            rl = np.angle(np.exp(1j * (obs[k][0] - lon)))
            c += np.where(vis[:, k], rl ** 2 + (obs[k][1] - lat) ** 2, 0.0)
        return c

    for it in range(iters):
        Hx, g = np.zeros((len(P), 3, 3)), np.zeros((len(P), 3))
        for k in range(K):
            lon, lat, gl, gt = fu._derivs(X, *poses[k])
            rl = np.angle(np.exp(1j * (obs[k][0] - lon)))
            rt = obs[k][1] - lat
            w = vis[:, k][:, None]
            Hx += w[:, :, None] * (gl[:, :, None] * gl[:, None, :] + gt[:, :, None] * gt[:, None, :])
            g += w * (gl * rl[:, None] + gt * rt[:, None])
        Hx += 1e-12 * np.eye(3)
        dX = np.linalg.solve(Hx, g[..., None])[..., 0]
        cc = cost(X)
        lam = np.ones(len(P))
        todo = np.arange(len(P))
        for _ in range(halvings):
            cn = cost(X + lam[:, None] * dX)[todo]
            todo = todo[cn > cc[todo] * (1 + 1e-12) + 1e-30]
            if not len(todo):
                break
            lam[todo] *= 0.5
        gave_up = np.zeros(len(P), bool)
        gave_up[todo] = True
        lam[todo] = 0.0
        X = X + lam[:, None] * dX
        trace.append({"cost": cc.tolist(), "lam": lam.tolist(), "gave_up": gave_up.tolist(),
                      "err_m": np.linalg.norm(X - P, axis=-1).tolist(),
                      "full_step_m": np.linalg.norm(dX, axis=-1).tolist()})
    return X, trace


def analyse(run, npz):
    looks = fu.load_looks([run])
    fan = sn.SlitFan(width=3600)
    d = np.load(npz)
    P, X0, H, vis = d["P"], d["X0"], d["H"], d["vis"]
    K = len(looks)
    cs = np.stack([lk[1] for lk in looks])
    s = np.array([lk[3] for lk in looks])
    ends = np.cumsum([L for _, L, _ in fu.PATHS["turn90"]])
    seg = np.searchsorted(ends - 1e-9, s)
    seen2 = vis.sum(1) >= 2
    rng_k = np.linalg.norm(P[:, None, :] - cs[None], axis=-1)
    rmin = np.min(np.where(vis, rng_k, np.inf), 1)
    e0 = np.linalg.norm(X0 - P, axis=-1) / rmin
    worst_ref, _, _ = fu.ray_interval(P, H, vis, looks, 0.5, fan)
    well = seen2 & (worst_ref <= orc.USABLE_REL)
    bad = np.flatnonzero(seen2 & (e0 > 1e-9))
    out = {"cells": int(len(P)), "seen_by_two": int(seen2.sum()), "well": int(well.sum()),
           "F1_max_rel_well": float(e0[well].max()), "F1_max_rel_seen2": float(e0[seen2].max()),
           "bad_seen2": int(len(bad)), "bad_well": int((well & (e0 > 1e-9)).sum()),
           "e0_quantiles_seen2": {q: float(np.quantile(e0[seen2], q)) for q in (0.5, 0.99, 0.9999, 1.0)}}
    print(json.dumps(out, indent=1))
    order = bad[np.argsort(-e0[bad])]
    rows = []
    for i in order[:40]:
        ks = np.flatnonzero(vis[i])
        lats = []
        for k in ks:
            lon, lat, _, _ = fu._derivs(P[i:i + 1], looks[k][1], looks[k][2])
            lats.append((float(np.degrees(lon[0])), float(np.degrees(lat[0]))))
        lats = np.array(lats)
        rows.append({"i": int(i), "P": np.round(P[i], 4).tolist(), "X0": np.round(X0[i], 4).tolist(),
                     "err_rel": float(e0[i]), "well": bool(well[i]), "worst_ref": float(worst_ref[i]),
                     "n_seeing": int(len(ks)), "segments_seeing": np.bincount(seg[ks], minlength=3).tolist(),
                     "k_range": [int(ks.min()), int(ks.max())], "rmin_m": float(rmin[i]),
                     "abs_lat_max_deg": float(np.abs(lats[:, 1]).max()),
                     "lon_deg_range": [float(lats[:, 0].min()), float(lats[:, 0].max())],
                     "cond_H": float(np.linalg.cond(H[i]))})
    for r in rows[:15]:
        print(json.dumps(r))
    tb = order[:min(200, len(order))]
    Xt, trace = traced(looks, P[tb], vis[tb], iters=60)
    et = np.linalg.norm(Xt - P[tb], axis=-1) / rmin[tb]
    gave = np.array([t["gave_up"] for t in trace])          # (iters, n)
    first_give = np.where(gave.any(0), np.argmax(gave, 0), -1)
    Xl, _ = traced(looks, P[tb], vis[tb], iters=600)
    el = np.linalg.norm(Xl - P[tb], axis=-1) / rmin[tb]
    summ = {"traced_cells": int(len(tb)), "traced_reproduces_fuse": float(np.max(np.abs(et - e0[tb]))),
            "gave_up_at_some_iteration": int((first_give >= 0).sum()),
            "first_give_up_iteration_quantiles": np.quantile(first_give[first_give >= 0], [0, 0.5, 1]).tolist()
            if (first_give >= 0).any() else None,
            "err_rel_after_600_iterations_max": float(el.max()),
            "converged_with_600_iterations": int((el <= 1e-9).sum())}
    print(json.dumps(summ, indent=1))
    worst = order[0]
    t0 = [{"it": it, "cost": t["cost"][0], "lam": t["lam"][0], "gave_up": t["gave_up"][0], "err_m": t["err_m"][0],
           "full_step_m": t["full_step_m"][0]} for it, t in enumerate(trace)]
    out.update({"worst": rows[:40], "traced": summ, "trace_of_worst": t0})
    o = Path(npz).with_suffix(".json")
    o.write_text(json.dumps(out, indent=1))
    print(f"[f1] wrote {o}")


def _start_traced(looks, P, vis, X_start, iters=60):
    """traced(), from a given start instead of fuse's (5 % beyond the truth along the first seeing look's ray)."""
    K = len(looks)
    poses = [(lk[1], lk[2]) for lk in looks]
    obs = [fu._derivs(P, *poses[k])[:2] for k in range(K)]
    X = X_start.copy()

    def cost(Xa):
        c = np.zeros(len(Xa))
        for k in range(K):
            lon, lat, _, _ = fu._derivs(Xa, *poses[k])
            rl = np.angle(np.exp(1j * (obs[k][0] - lon)))
            c += np.where(vis[:, k], rl ** 2 + (obs[k][1] - lat) ** 2, 0.0)
        return c

    for _ in range(iters):
        Hx, g = np.zeros((len(P), 3, 3)), np.zeros((len(P), 3))
        for k in range(K):
            lon, lat, gl, gt = fu._derivs(X, *poses[k])
            rl = np.angle(np.exp(1j * (obs[k][0] - lon)))
            rt = obs[k][1] - lat
            w = vis[:, k][:, None]
            Hx += w[:, :, None] * (gl[:, :, None] * gl[:, None, :] + gt[:, :, None] * gt[:, None, :])
            g += w * (gl * rl[:, None] + gt * rt[:, None])
        Hx += 1e-12 * np.eye(3)
        dX = np.linalg.solve(Hx, g[..., None])[..., 0]
        cc = cost(X)
        lam = np.ones(len(P))
        todo = np.arange(len(P))
        for _ in range(40):
            cn = cost(X + lam[:, None] * dX)[todo]
            todo = todo[cn > cc[todo] * (1 + 1e-12) + 1e-30]
            if not len(todo):
                break
            lam[todo] *= 0.5
        lam[todo] = 0.0
        X = X + lam[:, None] * dX
    return X, cost(X)


def probe(run, npz):
    """The failing cell alone: where the stalled point is, which looks carry its cost, and whether the same loop
    converges from other starts (the truth + 1 mm; the least-squares intersection of the seeing looks' rays)."""
    looks = fu.load_looks([run])
    d = np.load(npz)
    P, X0, vis = d["P"], d["X0"], d["vis"]
    cs = np.stack([lk[1] for lk in looks])
    rmin = np.min(np.where(vis, np.linalg.norm(P[:, None, :] - cs[None], axis=-1), np.inf), 1)
    e0 = np.linalg.norm(X0 - P, axis=-1) / rmin
    i = int(np.argmax(np.where(vis.sum(1) >= 2, e0, -1)))
    p, x, v = P[i:i + 1], X0[i:i + 1], vis[i:i + 1]
    ks = np.flatnonzero(v[0])
    per = []
    for k in ks:
        lo_t, la_t, _, _ = fu._derivs(p, looks[k][1], looks[k][2])
        lo_x, la_x, _, _ = fu._derivs(x, looks[k][1], looks[k][2])
        rl = float(np.angle(np.exp(1j * (lo_t[0] - lo_x[0]))))
        per.append({"k": int(k), "s_m": float(looks[k][3]), "lat_true_deg": float(np.degrees(la_t[0])),
                    "lat_fused_deg": float(np.degrees(la_x[0])), "lon_residual_rad": rl,
                    "lat_residual_rad": float(la_t[0] - la_x[0])})
    big = [q for q in per if abs(q["lon_residual_rad"]) > 0.1]
    # rays of the seeing looks, the least-squares point closest to all of them (exact at no noise)
    A, b = np.zeros((3, 3)), np.zeros(3)
    for k in ks:
        u = (p[0] - cs[k]) / np.linalg.norm(p[0] - cs[k])
        M = np.eye(3) - np.outer(u, u)
        A += M
        b += M @ cs[k]
    x_ls = np.linalg.solve(A, b)
    rel = lambda X: float(np.linalg.norm(X[0] - p[0]) / rmin[i])
    res = {"cell": int(i), "P": p[0].tolist(), "X0_fused": x[0].tolist(), "err_rel": float(e0[i]),
           "looks_seeing": int(len(ks)),
           "looks_with_lon_residual_over_0.1_rad": [{kk: q[kk] for kk in ("k", "s_m", "lat_true_deg", "lat_fused_deg",
                                                                          "lon_residual_rad")} for q in big],
           "sum_sq_lon_residual_of_those": float(sum(q["lon_residual_rad"] ** 2 for q in big)),
           "nadir_closest_look": min(per, key=lambda q: q["lat_true_deg"])}
    for name, X_start in (("truth + 1 mm", p + np.array([[1e-3, 0, 0]])), ("rays' least-squares point", x_ls[None])):
        Xs, c = _start_traced(looks, p, v, X_start)
        res[f"from {name}"] = {"start_err_rel": rel(X_start), "end_err_rel": rel(Xs), "end_cost": float(c[0])}
    print(json.dumps(res, indent=1))
    o = Path(npz).with_name(Path(npz).stem + "_probe.json")
    o.write_text(json.dumps(res, indent=1))
    print(f"[f1] wrote {o}")


def nadir(run, npz):
    """How many cells pass near a seeing look's pole, and how many of those miss F1."""
    looks = fu.load_looks([run])
    d = np.load(npz)
    P, X0, vis = d["P"], d["X0"], d["vis"]
    cs = np.stack([lk[1] for lk in looks])
    rmin = np.min(np.where(vis, np.linalg.norm(P[:, None, :] - cs[None], axis=-1), np.inf), 1)
    e0 = np.linalg.norm(X0 - P, axis=-1) / rmin
    seen2 = vis.sum(1) >= 2
    pole = np.full(len(P), np.inf)              # deg from the nearest seeing look's pole
    for k, lk in enumerate(looks):
        _, lat, _, _ = fu._derivs(P, lk[1], lk[2])
        pole = np.where(vis[:, k], np.minimum(pole, 90 - np.degrees(np.abs(lat))), pole)
    out = {}
    for t in (0.5, 1.0, 2.0, 5.0):
        m = seen2 & (pole < t)
        out[f"within {t} deg of a seeing look's pole"] = {"cells": int(m.sum()), "miss_F1": int((m & (e0 > 1e-9)).sum())}
    out["failing cell's distance from a pole, deg"] = float(pole[np.argmax(np.where(seen2, e0, -1))])
    print(json.dumps(out, indent=1))
    Path(npz).with_name(Path(npz).stem + "_nadir.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "analyse": analyse, "probe": probe, "nadir": nadir}[sys.argv[1]](*sys.argv[2:])
