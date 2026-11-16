"""Blender side: world bounding boxes of a scene's meshes whose name matches (D1b Step 0, #19).

Meshes are walked through the evaluated depsgraph's object instances, so the meshes inside linked collection
instances (the Classroom's desks, chairs, teacherDesk) are listed too, one row per instance, with the instancer's
name (#19's Step 0 found the first version, which listed scene objects only, missed them).

    blender -b --python-exit-code 1 -P motion/bl_objects.py -- --blend B --out J [--match door desk table board]
    python -m motion.bl_objects --blend B --out J [--match ...]                         (the bpy wheel)

Writes J: one row per mesh instance (name, instancer or null, collection, world AABB min/max, centre); a row matches
when any of --match occurs, case-insensitively, in the mesh object's name or its instancer's (all when none given). MEASURED from the .blend; it is how Step 0 finds the front door
and the free strip between the board and the first desk row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog="bl_objects")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", nargs="*", default=[])
    a = ap.parse_args(argv)
    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    dg = bpy.context.evaluated_depsgraph_get()
    rows = []
    pats = [m.lower() for m in a.match]
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH":
            continue
        parent = inst.parent.original.name if inst.is_instance and inst.parent is not None else None
        names = [ob.original.name.lower()] + ([parent.lower()] if parent else [])
        if pats and not any(p in n for p in pats for n in names):
            continue
        M = inst.matrix_world.copy()
        pts = [M @ Vector(c) for c in ob.bound_box]
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        orig = ob.original
        rows.append({"name": orig.name, "instancer": parent,
                     "collection": orig.users_collection[0].name if orig.users_collection else None,
                     "min_w": lo, "max_w": hi, "centre_w": [(l + h) / 2 for l, h in zip(lo, hi)],
                     "visible_render": not orig.hide_render})
    rows.sort(key=lambda r: (r["instancer"] or "", r["name"], r["min_w"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"schema": "foveal-motion-objects-v1", "blend": str(Path(a.blend).resolve()),
                                       "match": a.match, "objects": rows}, indent=1))
    for r in rows:
        print(f"[objects] {(r['instancer'] + '/' if r['instancer'] else '') + r['name']:48s} min {[round(v, 3) for v in r['min_w']]} max {[round(v, 3) for v in r['max_w']]}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(argv))
