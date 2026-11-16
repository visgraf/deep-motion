import bpy, sys
bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
dg = bpy.context.evaluated_depsgraph_get()
ob = bpy.context.scene.objects["wall.008"].evaluated_get(dg)
M = ob.matrix_world
me = ob.to_mesh()
pts = sorted({(round((M @ v.co).x, 3), round((M @ v.co).y, 3)) for v in me.vertices if 1.3 <= (M @ v.co).z <= 1.9})
for p in pts:
    print("[wall008]", p)
# faces whose normal points into the room (-Y) at head height: their x span and y
for f in me.polygons:
    n = (M.to_3x3() @ f.normal).normalized()
    c = M @ f.center
    if n.y < -0.9 and 0.5 < c.z < 2.5:
        xs = [(M @ me.vertices[i].co).x for i in f.vertices]; ys = [(M @ me.vertices[i].co).y for i in f.vertices]
        print("[wall008-face -Y]", round(min(xs), 3), round(max(xs), 3), "y", round(min(ys), 3), round(max(ys), 3))
