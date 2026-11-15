"""Issue #7 quick look (Code): name an instance id with fov-render's own catalogue (Blender side, read-only).

    blender -b --python-exit-code 1 -P docs/runs/issue-7/quick-look/diagnosis/lamp_name.py -- REPO BLEND 259
"""
import sys

a = sys.argv[sys.argv.index("--") + 1:]
sys.path[:0] = [a[0], a[0] + "/external/fov-render"]
import bpy  # noqa: E402
from fov_render.bl import env  # noqa: E402
from fov_render.bl.scene import open_scene  # noqa: E402

env.prepare()
h = open_scene(a[1])
for e in h.catalog["instances"]:
    if int(e["id"]) == int(a[2]):
        print("[lamp_name]", e)
        o = bpy.data.objects.get(e["name"].split(" [")[0])
        print("[lamp_name] materials", [m.name for m in o.data.materials if m] if o is not None and o.type == "MESH" else None)
