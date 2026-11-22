"""#34, G0 (Code, read only): where the accepted zeta of N's line path (stage 11's cache) and its forced per-frame
path (stage 14's cache) differ, per reference, and which anchors change decision.

    .venv/bin/python docs/runs/dm-issue-34/g0_zeta.py > docs/runs/dm-issue-34/g0_zeta.txt
"""
import json, sys, numpy as np
from pathlib import Path
C = Path("runs/stage11-cache")
looks = json.loads(Path("runs/r3-straight/run.json").read_text())["looks"]
ld = lambda d, j: np.load(C / d / f"r{j:04d}.npz", allow_pickle=True)["obj"].item()
refs = list(range(50, 161, 10))
allrel = []
for j in refs:
    a, g = ld("N1024", j), ld("N1024__per-frame_path_forced", j)
    na, ng = ld("N1024__every_anchor_s_reason", j), g
    c = np.array(looks[j]["c"])
    za = {tuple(p): 1/np.linalg.norm(x - c) for p, x in zip(np.asarray(a["px_acc"]).tolist(), a["X"])}
    zb = {tuple(p): 1/np.linalg.norm(x - c) for p, x in zip(np.asarray(g["px_acc"]).tolist(), g["X"])}
    k = sorted(za.keys() & zb.keys())
    rel = np.array([abs(za[q]-zb[q])/za[q] for q in k])
    allrel.append(rel)
    ra = {tuple(p): v for p, v in zip(np.asarray(na["px_all"]).tolist(), np.asarray(na["reason_all"]).tolist())}
    rb = {tuple(p): v for p, v in zip(np.asarray(ng["px_all"]).tolist(), np.asarray(ng["reason_all"]).tolist())}
    diff = [(q, ra[q], rb[q]) for q in ra.keys() & rb.keys() if (ra[q] == 0) != (rb[q] == 0)]
    print(f"ref {j}: common {len(k):6d}, rel max {rel.max():.2e}, p99 {np.quantile(rel,.99):.2e}, median {np.median(rel):.2e}, "
          f"> 1e-9: {(rel>1e-9).sum()}, > 1e-12: {(rel>1e-12).sum()}, exactly 0: {(rel==0).sum()}; decisions differ: {diff}; "
          f"general {g.get('general')}")
r = np.concatenate(allrel)
print(f"all: {r.size} common, max {r.max():.3e}, > 1e-9: {(r>1e-9).sum()} ({(r>1e-9).mean():.4%}), > 1e-10: {(r>1e-10).sum()}, "
      f"> 1e-12: {(r>1e-12).sum()}, exactly 0: {(r==0).sum()}, median {np.median(r):.2e}, p99.9 {np.quantile(r,.999):.2e}")
