"""#7, Code: clearance of candidate 5b.2 paths in the Classroom (Blender side; no render).

    .venv/bin/python docs/runs/issue-7/code/classroom/clearance.py poses OUT.json      # host: the candidates' poses
    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P docs/runs/issue-7/code/classroom/clearance.py -- OUT.json RESULT.json      # Blender: nearest surface per pose

The host step writes every look's head centre (deep/fusion.py's path_poses, 1 cm steps) for each candidate. The Blender
step finds, for each, the nearest point on any evaluated mesh (BVH over the depsgraph's instances, world space), and
reports the minimum and where it occurs, with the object's name; it also records every mesh's world AABB. Then

    .venv/bin/python docs/runs/issue-7/code/classroom/clearance.py plot OUT.json RESULT.json PLAN.png
"""
import json
import sys


def host(out):
    import numpy as np
    sys.path.insert(0, ".")
    from deep import fusion
    cands = {
        "turn90 @ 0.897,0.459,90": ("turn90", (0.897, 0.459), 90.0),
        "turn90 @ 0.897,0.259,90": ("turn90", (0.897, 0.259), 90.0),
        "turn90 @ 0.897,0.659,90": ("turn90", (0.897, 0.659), 90.0),
        "straight_long @ 0.897,-1.412,90": ("straight_long", (0.897, -1.412), 90.0),
        "straight_long @ 0.897,0.459,90": ("straight_long", (0.897, 0.459), 90.0),
        # the calibration room's pairing: the straight starts 2.0 m behind the turn (#7's commands: 0,-2 and 0,0)
        "straight_long @ 0.897,-1.541,90": ("straight_long", (0.897, -1.541), 90.0),
    }
    rec = {}
    for name, (kind, p0, hd) in cands.items():
        ps = fusion.path_poses(kind=kind, step=0.01, start={"p0": p0, "heading_deg": hd, "h": 1.6})
        rec[name] = {"path": kind, "start": [*p0, hd], "h": 1.6, "s": [p[0] for p in ps],
                     "c": [np.asarray(p[1]).tolist() for p in ps], "piece": [int(p[3]) for p in ps]}
    with open(out, "w") as f:
        json.dump(rec, f)


def blender(inp, out):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    trees, boxes = [], []
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH" or ob.original.hide_render:
            continue
        me = ob.to_mesh()
        if me is None or not len(me.polygons):
            ob.to_mesh_clear()
            continue
        M = inst.matrix_world.copy()
        verts = [M @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
        trees.append((ob.name, BVHTree.FromPolygons(verts, polys)))
        lo = [min(v[i] for v in verts) for i in range(3)]
        hi = [max(v[i] for v in verts) for i in range(3)]
        boxes.append([ob.name] + lo + hi)
        ob.to_mesh_clear()
    cands = json.load(open(inp))
    res = {"meshes": len(trees), "aabb_world": boxes}
    for name, c in cands.items():
        best = []
        for s, p, pc in zip(c["s"], c["c"], c["piece"]):
            q = Vector(p)
            d, who, at = 1e9, None, None
            for nm, t in trees:
                hit = t.find_nearest(q)
                if hit[0] is not None and hit[3] < d:
                    d, who, at = hit[3], nm, list(hit[0])
            best.append((d, s, pc, who, at))
        m = min(best)
        per_piece = {}
        for d, s, pc, who, at in best:
            if pc not in per_piece or d < per_piece[pc][0]:
                per_piece[pc] = (d, s, who)
        res[name] = {"start": c["start"], "end_c": c["c"][-1], "looks": len(best), "min_clearance_m": m[0], "at_s_m": m[1],
                     "nearest_object": m[3], "nearest_point": m[4],
                     "min_per_piece": {str(k): {"m": v[0], "s": v[1], "object": v[2]} for k, v in sorted(per_piece.items())}}
        print(f"[clearance] {name}: min {m[0]:.3f} m at s = {m[1]:.2f} m ({m[3]}); per piece "
              + ", ".join(f"{k}: {v[0]:.3f} ({v[2]})" for k, v in sorted(per_piece.items())), flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


def plot(cands_json, result_json, png):
    """Plan view: every mesh's world AABB with its top below 1.85 m (furniture; walls and ceiling excluded by size),
    and the candidates' head paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cands, res = json.load(open(cands_json)), json.load(open(result_json))
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    for nm, x0, y0, z0, x1, y1, z1 in res["aabb_world"]:
        if z1 < 1.85 and (x1 - x0) < 4 and (y1 - y0) < 4:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#d9d6cf", ec="#9a958a", lw=0.3))
        elif z0 > 1.85 and (x1 - x0) < 4 and (y1 - y0) < 4:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="#c9c4ba", lw=0.3, ls=":"))
    # the straights share the aisle's line; drawn thinner in turn so each one's span and start dot show
    cols = {"turn90 @ 0.897,0.459,90": ("#e8590c", 2.4), "straight_long @ 0.897,0.459,90": ("#7aa6e0", 4.0),
            "straight_long @ 0.897,-1.412,90": ("#1c5fb8", 2.6), "straight_long @ 0.897,-1.541,90": ("#2bb3a3", 1.0)}
    for name, (col, lw) in cols.items():
        c = cands[name]["c"]
        ax.plot([p[0] for p in c], [p[1] for p in c], color=col, lw=lw,
                label=f"{name} (min clearance {res[name]['min_clearance_m']:.2f} m)")
        ax.plot(c[0][0], c[0][1], "o", color=col, ms=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m (the board at y = 3.26)")
    ax.set_title("Classroom, plan: furniture (AABB, top < 1.85 m; dotted: above 1.85 m) and the candidate paths", loc="left", fontsize=9)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.savefig(png, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if argv[0] == "poses":
        host(argv[1])
    elif argv[0] == "plot":
        plot(*argv[1:4])
    else:
        blender(argv[0], argv[1])
