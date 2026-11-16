"""Blender side (D1b Step 0, #19): world AABBs of every evaluated mesh *instance* in a scene, including the
linked collection instances (desks, chairs, ...) that motion/bl_objects.py does not list (it walks
scene.objects of type MESH only).

    blender -b --python-exit-code 1 -P bl_instances.py -- --blend B --out J

Writes J: one row per instance: the instanced object's name, the instancer's name (parent EMPTY, or itself),
world AABB min/max (from the evaluated object's bound_box through the instance's world matrix), and whether it
renders. MEASURED from the .blend.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    dg = bpy.context.evaluated_depsgraph_get()
    rows = []
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH":
            continue
        M = inst.matrix_world.copy()
        pts = [M @ Vector(c) for c in ob.bound_box]
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        parent = inst.parent.name if inst.is_instance and inst.parent else None
        rows.append({"name": ob.name, "instancer": parent, "is_instance": bool(inst.is_instance),
                     "min_w": lo, "max_w": hi, "centre_w": [(l + h) / 2 for l, h in zip(lo, hi)],
                     "visible_render": bool(inst.show_self) and not ob.original.hide_render})
    rows.sort(key=lambda r: (r["instancer"] or "", r["name"], r["min_w"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"schema": "foveal-motion-instances-v1", "blend": str(Path(a.blend).resolve()),
                                       "instances": rows}, indent=1))
    print(f"[instances] {len(rows)} mesh instances -> {a.out}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sys.exit(main(argv))
