"""#34, G0 (Code, RTX 4090 workstation): are N's line path and its forced per-frame path reproducible run to run?

Read only: computes N1024 at the references where G0's zeta differ by more than 1e-9 (60, 130, 160) and where one
decision differs (110), twice on each path, with deep/straight.py's n_reference as stage 14 calls it, and compares
each with the other run and with the cached rows (stage 11's N1024 and every-anchor reasons, at dade15d; stage 14's
forced per-frame row, at 9da6a65). Nothing is written to any cache.

    .venv/bin/python docs/runs/dm-issue-34/g0_repro.py > docs/runs/dm-issue-34/g0_repro.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from deep import assess as asx, e2e, oracle as orc, straight as st  # noqa: E402

CACHE = REPO / "runs/stage11-cache"
REFS = (60, 110, 130, 160)


def load(d, j):
    return np.load(CACHE / d / f"r{j:04d}.npz", allow_pickle=True)["obj"].item()


def compare(a, b, c):
    """a, b: rows with px_all/reason_all (or None) and px_acc/X; c the reference's centre."""
    za = {tuple(p): 1 / np.linalg.norm(x - c) for p, x in zip(np.asarray(a["px_acc"]).tolist(), a["X"])}
    zb = {tuple(p): 1 / np.linalg.norm(x - c) for p, x in zip(np.asarray(b["px_acc"]).tolist(), b["X"])}
    k = za.keys() & zb.keys()
    rel = np.array([abs(za[q] - zb[q]) / za[q] for q in k]) if k else np.zeros(0)
    out = {"accepted": (len(za), len(zb)), "common": len(k), "max_rel": float(rel.max()) if rel.size else 0.0,
           "n_gt_1e-9": int((rel > 1e-9).sum()), "n_nonzero": int((rel > 0).sum())}
    if "reason_all" in a and "reason_all" in b:
        ra = {tuple(p): v for p, v in zip(np.asarray(a["px_all"]).tolist(), np.asarray(a["reason_all"]).tolist())}
        rb = {tuple(p): v for p, v in zip(np.asarray(b["px_all"]).tolist(), np.asarray(b["reason_all"]).tolist())}
        out["decisions_differ"] = sorted((q, ra[q], rb[q]) for q in ra.keys() & rb.keys() if (ra[q] == 0) != (rb[q] == 0))
    return out


def main():
    r128, r1024 = str(REPO / "runs/r3-straight"), str(REPO / "runs/r3-straight-1024")
    _, L = orc.load(r128)
    refs = asx.references(len(L))
    looks128, truth, fan, _ = e2e.render_path(r128, refs=refs)
    looks1024, _, _, _ = e2e.render_path(r1024, refs=refs)
    print(f"references {refs}; checked {REFS}")
    for j in REFS:
        c = looks128[j][1]
        line_cache = dict(load("N1024", j), **{k: v for k, v in load("N1024__every_anchor_s_reason", j).items()
                                                 if k in ("px_all", "reason_all")})
        gen_cache = load("N1024__per-frame_path_forced", j)
        line = [st.n_reference(looks1024, truth, j, fan, keep_all=True) for _ in range(2)]
        gen = [st.n_reference(looks1024, truth, j, fan, keep_all=True, general=True) for _ in range(2)]
        print(f"\nref {j} (line runs general={[bool(o.get('general')) for o in line]}, per-frame runs general="
              f"{[bool(o.get('general')) for o in gen]})")
        for name, a, b in (("line now #1 vs line now #2", line[0], line[1]),
                           ("line now #1 vs stage 11's cache", line[0], line_cache),
                           ("per-frame now #1 vs per-frame now #2", gen[0], gen[1]),
                           ("per-frame now #1 vs stage 14's cache", gen[0], gen_cache),
                           ("line now #1 vs per-frame now #1 (G0's comparison)", line[0], gen[0]),
                           ("stage 11's cache vs stage 14's cache (G0 as run)", line_cache, gen_cache)):
            print(f"  {name}: {compare(a, b, c)}")


if __name__ == "__main__":
    main()
