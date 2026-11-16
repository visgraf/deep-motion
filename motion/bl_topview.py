"""Blender side: D1's top view, an orthographic render of the scene from above (DERIVED: a display render).

The camera looks straight down from cut_z (default 2.2 m): what is above it (the ceiling, ceiling lamps) is
behind the camera and not drawn, while the scene's lights still light the room. The view covers `bounds`
(x0 y0 x1 y1, world metres) with square pixels; topview.json records the world <-> pixel map:

    px = (x - x0) / m_per_px,   py = (y1 - y) / m_per_px      (image rows grow toward -Y)

    blender -b --python-exit-code 1 -P motion/bl_topview.py -- --blend B --out DIR --bounds x0 y0 x1 y1 [--device OPTIX]
    python -m motion.bl_topview --blend B --out DIR --bounds x0 y0 x1 y1 --device CPU          (the bpy wheel)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "external" / "fov-render"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog="bl_topview")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bounds", type=float, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--cut-z", type=float, default=2.2)
    ap.add_argument("--m-per-px", type=float, default=0.005)
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--device", default="OPTIX")
    a = ap.parse_args(argv)

    import bpy
    from fov_render.bl import config
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    scene = bpy.context.scene
    backend = config.setup_device(scene, config.RenderConfig(profile="small", device=a.device))
    x0, y0, x1, y1 = a.bounds
    w = int(round((x1 - x0) / a.m_per_px))
    h = int(round((y1 - y0) / a.m_per_px))
    cam_data = bpy.data.cameras.new("d1_topview")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(x1 - x0, y1 - y0)
    cam_data.sensor_fit = "AUTO"
    cam_data.clip_start = 0.001
    cam_data.clip_end = a.cut_z + 10.0
    cam = bpy.data.objects.new("d1_topview", cam_data)
    scene.collection.objects.link(cam)
    cam.location = ((x0 + x1) / 2, (y0 + y1) / 2, a.cut_z)
    cam.rotation_euler = (0.0, 0.0, 0.0)          # looks down -Z, image up = +Y
    scene.camera = cam
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.film_transparent = False
    r.image_settings.file_format = "PNG"
    scene.cycles.samples = a.spp
    scene.cycles.use_denoising = False
    r.filepath = str((out / "topview.png").resolve())
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    rec = {"schema": "foveal-motion-d1-topview-v1", "blend": str(Path(a.blend).resolve()), "bounds_w": [x0, y0, x1, y1],
           "cut_z_m": a.cut_z, "m_per_px": (x1 - x0) / w, "size_px": [w, h], "spp": a.spp, "device": backend,
           "seconds": round(time.perf_counter() - t0, 2),
           "map": "px = (x - x0) / m_per_px, py = (y1 - y) / m_per_px",
           "view_transform": scene.view_settings.view_transform}
    (out / "topview.json").write_text(json.dumps(rec, indent=1))
    print(f"[topview] {w} x {h} px, {a.spp} spp, {backend}, {rec['seconds']} s -> {out / 'topview.png'}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(argv))
