import numpy as np, json, sys
from deep import match as mt
kind = sys.argv[1]
looks, P, _, fan = mt.r1_run(f"runs/r1-{kind}")
m = len(looks)//2
px, u, pidx, _ = mt.anchors(looks, m, fan, max_n=20000)
c = looks[m][1]
rt = np.linalg.norm(P[pidx]-c, axis=-1)
edge = mt.depth_edge(looks, m, P, px)
rb, conf, best = mt.sweep(looks, m, px, u, fan)
idx = looks[m][4]; H, W = idx.shape
rng = np.linalg.norm(P - c, axis=-1).reshape(H, W)
near = np.full(len(px), np.inf); far = np.zeros(len(px))
for dr in range(-2,3):
    for dc in range(-2,3):
        v = rng[np.clip(px[:,0]+dr,0,H-1), (px[:,1]+dc)%W]
        near = np.fmin(near, v); far = np.fmax(far, v)
e = edge & (np.abs(rb/rt-1) > .1)
okn = np.abs(rb/near-1) <= .1; okf = np.abs(rb/far-1) <= .1
between = (rb > near*1.1) & (rb < far/1.1)
centre_is_near = np.abs(rt/near-1) <= .1
out = {"edge_anchors": int(edge.sum()), "edge_wrong": int(e.sum()),
 "wrong: est at the near surface": float(np.mean(okn[e])), "wrong: est at the far surface": float(np.mean(okf[e])),
 "wrong: between": float(np.mean(between[e])), "wrong: outside": float(np.mean((~okn & ~okf & ~between)[e])),
 "wrong: centre ray on the far surface": float(np.mean(~centre_is_near[e])),
 "right edge anchors: centre on near": float(np.mean(centre_is_near[edge & ~e])),
 "wrong & est near & centre far": float(np.mean((okn & ~centre_is_near)[e]))}
print(kind, json.dumps({k: round(v,3) if isinstance(v,float) else v for k,v in out.items()}, indent=0))
