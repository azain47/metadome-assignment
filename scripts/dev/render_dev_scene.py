"""Dev-only: build a contract-conformant test scene in a running Blender (raw socket).

Used by the render-tool tests while the scene tools (`load_car`, `build_set`, ...) are
not available. Follows the conventions in the shared contract: TS_* collections,
`ts_role` on every object, TS_CAR_ROOT (front = +Y) at the centre of the tyre contacts,
scene props `ts_road_centerline` / `ts_road_width_m`, camera TS_CAM, sun TS_SUN.

    python scripts/dev/render_dev_scene.py --host 172.24.112.1 --port 9882 \
        --car assets/cars/ferrari.glb [--azimuth 35 --elevation 4 --focal 50]
"""

import argparse
import json
import socket
import subprocess

SCENE_CODE = r'''
import bpy, math
from mathutils import Vector, Matrix, Euler

GLB = __GLB__
AZ, EL, FOCAL, DIST_M = __AZ__, __EL__, __FOCAL__, __DIST__
scene = bpy.context.scene

prefs = bpy.context.preferences
prefs.view.render_display_type = 'NONE'
cp = prefs.addons['cycles'].preferences
cp.get_devices()
for t in ('OPTIX', 'CUDA'):
    try:
        cp.compute_device_type = t
        if any(d.type == t for d in cp.devices):
            break
    except TypeError:
        pass
for d in cp.devices:
    d.use = d.type == cp.compute_device_type
scene.cycles.device = 'GPU'
scene.unit_settings.system = 'METRIC'

def coll(name):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in scene.collection.children:
        scene.collection.children.link(c)
    return c

for name in ('TS_CAR', 'TS_SET', 'TS_RIG'):
    c = coll(name)
    for o in list(c.all_objects):
        bpy.data.objects.remove(o, do_unlink=True)
bpy.data.orphans_purge(do_recursive=True)
car_c, set_c, rig_c = coll('TS_CAR'), coll('TS_SET'), coll('TS_RIG')

# ---- car
import re
lc = next(l for l in bpy.context.view_layer.layer_collection.children if l.collection == car_c)
bpy.context.view_layer.active_layer_collection = lc
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=GLB)
new = [o for o in bpy.data.objects if o not in before]
root = bpy.data.objects.new('TS_CAR_ROOT', None)
car_c.objects.link(root)
root['ts_role'] = 'car_other'
toks = lambda o: set(re.split('[^a-z]+', o.name.lower()))
for o in new:
    n = o.name.lower()
    if toks(o) & {'wheel', 'tire', 'tyre', 'rim'} and 'steering' not in n:
        o['ts_role'] = 'car_wheel'
    elif 'glass' in n or 'window' in n:
        o['ts_role'] = 'car_glass'
    elif 'light' in n or 'lamp' in n or 'led' in n:
        o['ts_role'] = 'car_lamp'
    elif 'body' in n or 'paint' in n:
        o['ts_role'] = 'car_body'
    else:
        o['ts_role'] = 'car_other'
wheels = [o for o in new if o.get('ts_role') == 'car_wheel'
          and not (o.parent and o.parent.get('ts_role') == 'car_wheel')]
bpy.context.view_layer.update()
def world_bbox(objs):
    pts = []
    for o in objs:
        for d in [o] + list(o.children_recursive):
            if d.type == 'MESH':
                pts += [d.matrix_world @ Vector(c) for c in d.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return lo, hi
contacts = []
for w in wheels:
    lo, hi = world_bbox([w])
    contacts.append(Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)))
c0 = sum(contacts, Vector()) / len(contacts)
for o in new:
    if o.parent is None:
        o.parent = root
        o.location -= c0
blo, bhi = world_bbox([o for o in new if o.parent == root])
whi = world_bbox(wheels)
root['ts_car_key'] = 'cars/dev.glb'
root['ts_dims_m'] = [bhi.y - blo.y, bhi.x - blo.x, bhi.z - blo.z]
root['ts_wheel_radius_m'] = (whi[1].z - whi[0].z) / 2
# Wheel pivots (scene-tool convention): empties on the axle centre, parented to the root with
# identity rotation, wheel objects re-parented under them; spin = local X, forward = negative.
bpy.context.view_layer.update()
pivots = []
for w in wheels:
    lo, hi = world_bbox([w])
    centre_w = (lo + hi) / 2
    piv = bpy.data.objects.new('TS_WHEEL_' + w.name, None)
    car_c.objects.link(piv)
    piv.parent = root
    piv.location = root.matrix_world.inverted() @ centre_w
    local = root.matrix_world.inverted() @ centre_w
    piv['ts_role'] = 'car_wheel'; piv['ts_wheel_pivot'] = True
    piv['ts_wheel_radius_m'] = (hi.z - lo.z) / 2
    piv['ts_wheel_pos'] = ('F' if local.y > 0 else 'R') + ('L' if local.x < 0 else 'R')
    bpy.context.view_layer.update()
    mw = w.matrix_world.copy()
    w.parent = piv
    w.matrix_world = mw
    pivots.append(piv)
root['ts_wheels'] = [p.name for p in pivots]
root['ts_wheel_axis_local'] = 'X'
root['ts_wheel_forward_sign'] = -1

# Glass as clear transmission (what load_car does), so transparent-glass film applies.
for o in new:
    if o.get('ts_role') == 'car_glass':
        for slot in o.material_slots:
            b = next((n for n in slot.material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if slot.material else None
            if b is not None:
                b.inputs['Transmission Weight'].default_value = 1.0
                b.inputs['Roughness'].default_value = 0.02
                b.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)

# ---- road: straight 60 m behind, gentle left curve ahead, 400 m long
pts = []
x, y, h = 0.0, -60.0, 0.0
for i in range(int(460 / 0.5) + 1):
    pts.append((x, y))
    s = i * 0.5
    k = 0.0 if s < 80 else 1 / 300.0
    h += k * 0.5
    x -= math.sin(h) * 0.5
    y += math.cos(h) * 0.5
width = 7.0
verts, faces = [], []
for i, (px, py) in enumerate(pts):
    j = min(i + 1, len(pts) - 1); k2 = max(i - 1, 0)
    tx, ty = pts[j][0] - pts[k2][0], pts[j][1] - pts[k2][1]
    L = math.hypot(tx, ty); nx, ny = ty / L, -tx / L
    verts += [(px - nx * width / 2, py - ny * width / 2, 0.01), (px + nx * width / 2, py + ny * width / 2, 0.01)]
    if i:
        a = 2 * (i - 1); faces.append((a, a + 1, a + 3, a + 2))
me = bpy.data.meshes.new('TS_ROAD'); me.from_pydata(verts, [], faces)
road = bpy.data.objects.new('TS_ROAD', me); set_c.objects.link(road); road['ts_role'] = 'road'
mat = bpy.data.materials.new('TS_ROAD_MAT'); mat.diffuse_color = (0.05, 0.05, 0.05, 1)
mat.use_nodes = True if hasattr(mat, 'use_nodes') else None
bsdf = mat.node_tree.nodes.get('Principled BSDF')
if bsdf: bsdf.inputs['Base Color'].default_value = (0.04, 0.04, 0.045, 1); bsdf.inputs['Roughness'].default_value = 0.8
me.materials.append(mat)
scene['ts_road_centerline'] = [v for p in pts for v in p]
scene['ts_road_width_m'] = width
scene['ts_set_terrain'] = 'alpine'

bpy.ops.mesh.primitive_plane_add(size=4000, location=(0, 0, 0))
g = bpy.context.active_object; g.name = 'TS_GROUND'; g['ts_role'] = 'ground'
for c in list(g.users_collection): c.objects.unlink(g)
set_c.objects.link(g)
def card(name, role, verts, faces):
    m = bpy.data.meshes.new(name); m.from_pydata(verts, [], faces)
    o = bpy.data.objects.new(name, m); set_c.objects.link(o); o['ts_role'] = role
    return o
card('TS_RIDGE_0', 'standin_ridge', [(-900, 700, 0), (-300, 900, 450), (100, 800, 250), (500, 1000, 600), (1100, 750, 0)], [(0, 1, 2, 3, 4)])
card('TS_RIDGE_1', 'standin_ridge', [(-1500, 500, 0), (-1100, 600, 350), (-700, 550, 0)], [(0, 1, 2)])
for i, (bx, by) in enumerate(((25, 40), (-30, 70))):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(bx, by, 6))
    b = bpy.context.active_object; b.scale = (10, 10, 12); b.name = f'TS_BLD_{i}'; b['ts_role'] = 'standin_building'
    for c in list(b.users_collection): c.objects.unlink(b)
    set_c.objects.link(b)

# ---- place car on road at s = 60 m, right lane
idx = int(60 / 0.5)
(px, py), (qx, qy) = pts[idx], pts[idx + 1]
tx, ty = qx - px, qy - py; L = math.hypot(tx, ty); tx, ty = tx / L, ty / L
yaw = math.atan2(-tx, ty)
root.location = (px + ty * width / 4, py - tx * width / 4, 0)
root.rotation_euler = (0, 0, yaw)

# ---- camera
cam_data = bpy.data.cameras.new('TS_CAM'); cam_data.lens = FOCAL; cam_data.sensor_width = 36
cam = bpy.data.objects.new('TS_CAM', cam_data); rig_c.objects.link(cam); scene.camera = cam
bpy.context.view_layer.update()
target = root.matrix_world @ Vector((0, 0, 0.7))
fwd = (root.matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()
left = (root.matrix_world.to_3x3() @ Vector((-1, 0, 0))).normalized()
a, e = math.radians(AZ), math.radians(EL)
d = DIST_M or 9.0
dirv = (fwd * math.cos(a) + left * math.sin(a)) * math.cos(e) + Vector((0, 0, math.sin(e)))
cam.location = target + dirv * d
cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 1536, 1024, 100

sun_d = bpy.data.lights.new('TS_SUN', 'SUN'); sun_d.energy = 3.0
sun = bpy.data.objects.new('TS_SUN', sun_d); rig_c.objects.link(sun)
sun.rotation_euler = Euler((math.radians(55), 0, math.radians(135)))
if scene.world is None:
    scene.world = bpy.data.worlds.new('World')
w = scene.world
bg = w.node_tree.nodes.get('Background') if w.node_tree else None
if bg: bg.inputs['Color'].default_value = (0.45, 0.6, 0.9, 1); bg.inputs['Strength'].default_value = 1.0
for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
        area.tag_redraw()
result = {'status': 'ok', 'wheels': root['ts_wheels'], 'dims': list(root['ts_dims_m']),
          'wheel_r': root['ts_wheel_radius_m'], 'n_objects': len(new)}
'''


def send(host: str, port: int, code: str, timeout: float = 180.0) -> dict:
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall((json.dumps({"type": "execute", "code": code, "strict_json": True}) + "\0").encode())
        buf = b""
        while not buf.endswith(b"\0"):
            chunk = s.recv(1 << 16)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.rstrip(b"\0"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--car", default="assets/cars/ferrari.glb")
    ap.add_argument("--azimuth", type=float, default=35.0)
    ap.add_argument("--elevation", type=float, default=4.0)
    ap.add_argument("--focal", type=float, default=50.0)
    ap.add_argument("--distance", type=float, default=0.0)
    a = ap.parse_args()
    glb = subprocess.check_output(["wslpath", "-w", a.car], text=True).strip()
    code = (SCENE_CODE.replace("__GLB__", repr(glb)).replace("__AZ__", repr(a.azimuth))
            .replace("__EL__", repr(a.elevation)).replace("__FOCAL__", repr(a.focal))
            .replace("__DIST__", repr(a.distance)))
    print(json.dumps(send(a.host, a.port, code), indent=1)[:3000])


if __name__ == "__main__":
    main()
