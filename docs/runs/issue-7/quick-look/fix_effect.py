"""What the flat-patch guard (7bb5e70) changes on a Classroom pair that also ran without it (issue #7, Code).

The pair (0, 5) completes with both matchers, so its estimate at e53f5d6 (before the guard) and at 3927982
(7bb5e70 merged) can be compared array by array:

    git worktree add --detach WT e53f5d6
    (cd WT && REPO/.venv/bin/python -m motion.dolly flow-depth --run REPO/runs/dolly-1 --strides 5 \
        --correspondence rgb --pairs 0:5 --out OLD)
    PYTHONPATH=. .venv/bin/python docs/runs/issue-7/quick-look/fix_effect.py OLD/pairs/s5/a0000-b0005.npz \
        runs/m3-classroom-look/pairs/s5/a0000-b0005.npz
"""
import sys

import numpy as np

from motion import matcher as m

o, n = np.load(sys.argv[1]), np.load(sys.argv[2])
same, diff = [], []
for k in sorted(set(o.files) | set(n.files)):
    if k not in o.files or k not in n.files:
        diff.append((k, "in one file only"))
        continue
    a, b = o[k], n[k]
    eq = np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if eq:
        same.append(k)
    else:
        diff.append((k, int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))) if a.dtype.kind == "f" else int(np.sum(a != b)))))
print("identical arrays:", same)
print("differing (array, elements):", diff)
ch = o["reason"] != n["reason"]
pairs, cnt = np.unique(np.stack([o["reason"][ch], n["reason"][ch]], 1), axis=0, return_counts=True)
print(f"anchors whose reason changed: {int(ch.sum())} of {ch.size}:",
      {f"{m.REASONS[a]} -> {m.REASONS[b]}": int(c) for (a, b), c in zip(pairs, cnt)})
print("accepted before / after:", int((o["reason"] == 0).sum()), int((n["reason"] == 0).sum()))
