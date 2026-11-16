"""Blender side: world bounding boxes of a scene's mesh objects whose name matches (D1b Step 0, #19).

    blender -b --python-exit-code 1 -P motion/bl_objects.py -- --blend B --out J [--match door desk table board]
    python -m motion.bl_objects --blend B --out J [--match ...]                         (the bpy wheel)

Writes J: one row per mesh object (name, collection, world AABB min/max, centre), names matched case-insensitively
by any of --match (all objects when none is given). MEASURED from the .blend; it is how Step 0 finds the front door
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
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        if pats and not any(p in ob.name.lower() for p in pats):
            continue
        ev = ob.evaluated_get(dg)
        pts = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        rows.append({"name": ob.name, "collection": ob.users_collection[0].name if ob.users_collection else None,
                     "min_w": lo, "max_w": hi, "centre_w": [(l + h) / 2 for l, h in zip(lo, hi)],
                     "visible_render": not ob.hide_render})
    rows.sort(key=lambda r: r["name"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"schema": "foveal-motion-objects-v1", "blend": str(Path(a.blend).resolve()),
                                       "match": a.match, "objects": rows}, indent=1))
    for r in rows:
        print(f"[objects] {r['name']:40s} min {[round(v, 3) for v in r['min_w']]} max {[round(v, 3) for v in r['max_w']]}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(argv))
