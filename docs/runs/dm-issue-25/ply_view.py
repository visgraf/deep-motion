"""#25, run by Code: render a PLY point cloud as stored (unlit vertex colours, Cycles, headless Blender 5.2.1).
The pictures beside the record, one camera for both clouds; --zmax 2.6 hides the ceiling (display only):
  blender -b --factory-startup --python-exit-code 1 -P docs/runs/dm-issue-25/ply_view.py -- \
    runs/demo2-classroom/classroom_SIDE.ply $PWD/docs/runs/dm-issue-25/classroom_SIDE.png \
    --eye 3.5 -6.5 6.5 --target -0.6 0.2 0.6 --lens 20 --zmax 2.6 --res 1280 800
(SIDE = perfect, real; the output path must be absolute.) Each point is a sphere of radius --radius (1 cm: the 2 cm
cells touch). Not part of deep/; a viewer for the files, nothing more.
Usage: blender -b --factory-startup --python-exit-code 1 -P ply_view.py -- PLY OUT.png --eye x y z --target x y z
       [--lens 18 | --ortho SCALE] [--zmax Z] [--radius 0.01] [--res W H] [--spp 16]"""
import argparse
import sys

import bpy
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument("ply")
ap.add_argument("out")
ap.add_argument("--eye", type=float, nargs=3, required=True)
ap.add_argument("--target", type=float, nargs=3, required=True)
ap.add_argument("--lens", type=float, default=18.0)
ap.add_argument("--ortho", type=float, default=None, help="orthographic scale (m); default perspective")
ap.add_argument("--zmax", type=float, default=None, help="hide points above this height (a cut-away, display only)")
ap.add_argument("--radius", type=float, default=0.01)
ap.add_argument("--res", type=int, nargs=2, default=(1600, 1000))
ap.add_argument("--spp", type=int, default=16)
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=a.ply)
ob = bpy.context.selected_objects[0]
me = ob.data
print("[view] attributes:", [(at.name, at.domain, at.data_type) for at in me.attributes], "verts", len(me.vertices))
col_name = next(at.name for at in me.attributes if at.data_type in ("BYTE_COLOR", "FLOAT_COLOR"))

# geometry nodes: optional cut-away above zmax, then mesh -> points, material
ng = bpy.data.node_groups.new("pts", "GeometryNodeTree")
ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
N, Lk = ng.nodes, ng.links
gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
m2p = N.new("GeometryNodeMeshToPoints")
m2p.inputs["Radius"].default_value = a.radius
setm = N.new("GeometryNodeSetMaterial")
src = gi.outputs["Geometry"]
if a.zmax is not None:
    pos, sep, cmp_ = N.new("GeometryNodeInputPosition"), N.new("ShaderNodeSeparateXYZ"), N.new("FunctionNodeCompare")
    cmp_.data_type, cmp_.operation = "FLOAT", "GREATER_THAN"
    cmp_.inputs[1].default_value = a.zmax
    dele = N.new("GeometryNodeDeleteGeometry")
    Lk.new(pos.outputs[0], sep.inputs[0]); Lk.new(sep.outputs["Z"], cmp_.inputs[0])
    Lk.new(src, dele.inputs["Geometry"]); Lk.new(cmp_.outputs[0], dele.inputs["Selection"])
    src = dele.outputs["Geometry"]
Lk.new(src, m2p.inputs["Mesh"]); Lk.new(m2p.outputs["Points"], setm.inputs["Geometry"])
Lk.new(setm.outputs["Geometry"], go.inputs["Geometry"])

mat = bpy.data.materials.new("vcol")
mat.use_nodes = True
mn, ml = mat.node_tree.nodes, mat.node_tree.links
mn.clear()
attr, emi, mo = mn.new("ShaderNodeAttribute"), mn.new("ShaderNodeEmission"), mn.new("ShaderNodeOutputMaterial")
attr.attribute_name = col_name
ml.new(attr.outputs["Color"], emi.inputs["Color"]); ml.new(emi.outputs[0], mo.inputs["Surface"])
setm.inputs["Material"].default_value = mat
mod = ob.modifiers.new("pts", "NODES")
mod.node_group = ng

sc = bpy.context.scene
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
cam.location = Vector(a.eye)
cam.rotation_euler = (Vector(a.target) - Vector(a.eye)).to_track_quat("-Z", "Y").to_euler()
if a.ortho:
    cam.data.type, cam.data.ortho_scale = "ORTHO", a.ortho
else:
    cam.data.lens = a.lens
cam.data.clip_start, cam.data.clip_end = 0.01, 200
sc.camera = cam
w = bpy.data.worlds.new("w"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.02, 0.025, 1)
sc.world = w
sc.render.engine = "CYCLES"
sc.cycles.samples = a.spp
sc.cycles.use_denoising = False
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
except Exception as e:  # noqa: BLE001
    print("[view] GPU not set:", e)
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x, sc.render.resolution_y = a.res
sc.render.filepath = a.out
bpy.ops.render.render(write_still=True)
print("[view] wrote", a.out)
