"""Blender side (D1b Step 0, #19): vertex-level detail for a few objects (world coordinates, evaluated).

    blender -b --python-exit-code 1 -P bl_detail.py -- --blend B --out J --names N1 N2 ... [--zband z0 z1]

For each named object (scene objects and instanced meshes): its world vertices within the z band (all when
none given) summarised as the convex hull of their (x, y) in plan, plus the distinct x and y values (rounded to
1 mm) with counts. MEASURED from the .blend.
"""
import argparse
import json
import sys
from pathlib import Path


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--zband", type=float, nargs=2, default=None)
    a = ap.parse_args(argv)
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    dg = bpy.context.evaluated_depsgraph_get()
    res = {}
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH" or ob.name not in a.names:
            continue
        M = inst.matrix_world.copy()
        me = ob.to_mesh()
        V = [M @ v.co for v in me.vertices]
        ob.to_mesh_clear()
        if a.zband:
            V = [v for v in V if a.zband[0] <= v.z <= a.zband[1]]
        xy = [(round(v.x, 4), round(v.y, 4)) for v in V]
        key = ob.name + (f" <{inst.parent.name}>" if inst.is_instance and inst.parent else "")
        xs, ys = {}, {}
        for v in V:
            xs[round(v.x, 3)] = xs.get(round(v.x, 3), 0) + 1
            ys[round(v.y, 3)] = ys.get(round(v.y, 3), 0) + 1
        res[key] = {"n_vertices_in_band": len(V), "plan_hull": hull(xy),
                    "z_range": [min(v.z for v in V), max(v.z for v in V)] if V else None,
                    "x_values": sorted(xs.items()), "y_values": sorted(ys.items())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"schema": "foveal-motion-detail-v1", "blend": str(Path(a.blend).resolve()),
                                       "zband": a.zband, "objects": res}, indent=1))
    for k, r in res.items():
        print(f"[detail] {k}: {r['n_vertices_in_band']} vertices in band, plan hull {r['plan_hull']}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sys.exit(main(argv))
