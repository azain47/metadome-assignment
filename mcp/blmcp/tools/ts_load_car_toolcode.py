"""Tool-code: import a car glTF into TS_CAR under TS_CAR_ROOT (start/poll job).

Pipeline (one phase per timer tick so Blender redraws in between):
download (thread) -> clear old car -> import glTF or append the cached library .blend ->
drop helper meshes (all materials fully transparent, e.g. Sketchfab backdrop boxes) ->
normalise (scale to length_m or a unit guess, front = +Y, origin at the tyre contact centre) ->
classify roles, wheel pivots, glass -> write the library cache -> done.
"""

__all__ = ("Params", "main", "FORWARD_AXES")

from typing import NamedTuple

# Yaw (degrees about +Z) that turns the model's front axis into +Y. "auto" guesses it.
_AXIS_YAW = {"+Y": 0.0, "-Y": 180.0, "+X": 90.0, "-X": -90.0}
FORWARD_AXES = ("auto",) + tuple(_AXIS_YAW)

# Bump when the normalisation changes, so stale library caches are not reused.
_CACHE_VERSION = "v5"


class Params(NamedTuple):
    action: str  # "start" | "status"
    car_key: str
    url: str
    etag: str
    forward_axis: str
    length_m: float  # real overall length; 0 = guess the unit from the bounding box


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end

# Name tokens (lower-case words of object, parent and material names). Tokens shorter than
# five letters must start a word ("rim" matches "rim_fl" but not "trim").
_TOKENS = {
    "wheel": ("wheel", "tire", "tyre", "rim"),
    "wheel_never": ("brake", "caliper", "pad", "steering", "fender", "arch"),
    "wheel_always": ("disc", "disk", "rotor"),
    "glass": ("glass", "window", "windshield", "windscreen"),
    "lamp": ("light", "lamp", "led", "signal", "indicator", "projector", "beam"),
    "lamp_never": ("interior", "lightweight"),
    "paint": ("paint", "body", "carpaint", "lacquer", "color", "colour"),
    "paint_never": ("interior", "under", "underside", "mechanical", "chassis"),
    "front": ("head", "front", "grille", "grill", "projector", "bonnet", "hood", "windshield", "windscreen"),
    "rear": ("tail", "rear", "brakelight", "trunk", "boot", "exhaust", "spoiler", "license", "licence"),
    "front_never": ("headrest",),
}

# Forward rolling (car moving along root +Y) is a NEGATIVE rotation about each pivot's local X.
_WHEEL_AXIS_LOCAL = "X"
_WHEEL_FORWARD_SIGN = -1


def _words(*names) -> list:
    import re
    out = []
    for name in names:
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
        out.extend(w for w in re.split(r"[^a-z]+", name.lower()) if w)
    return out


def _has(words, group: str) -> bool:
    for tok in _TOKENS[group]:
        for w in words:
            if w.startswith(tok) or (len(tok) >= 5 and tok in w):
                return True
    return False


def _material_kind(mat_name: str) -> str:
    w = _words(mat_name)
    if _has(w, "lamp") and not _has(w, "lamp_never"):
        return "lamp"
    if _has(w, "glass"):
        return "glass"
    if _has(w, "wheel") and not _has(w, "wheel_never"):
        return "wheel"
    if _has(w, "paint") and not _has(w, "paint_never"):
        return "paint"
    return "other"


def _ancestor_names(ob) -> list:
    names = []
    p = ob.parent
    while p is not None:
        names.append(p.name)
        p = p.parent
    return names


def _world_bbox(obs, max_points: int = 2_000_000):
    """(min, max) numpy vectors of the mesh objects' world-space vertices.

    Vertex-based on purpose: transformed `bound_box` corners overstate rotated parts
    (steered wheels, tilted rims)."""
    import numpy as np
    pts = _mesh_points([o for o in obs if o.type == "MESH"], max_points)
    if len(pts) == 0:
        return np.zeros(3), np.zeros(3)
    return pts.min(axis=0), pts.max(axis=0)


def _mesh_points(obs, max_points: int = 30000):
    """World-space vertex positions of mesh objects (numpy, subsampled)."""
    import numpy as np
    chunks = []
    for o in obs:
        me = o.data
        n = len(me.vertices)
        if n == 0:
            continue
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        m = np.array(o.matrix_world)
        chunks.append(co @ m[:3, :3].T + m[:3, 3])
    if not chunks:
        return np.zeros((0, 3))
    pts = np.concatenate(chunks)
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    return pts


def _guess_forward(meshes) -> tuple:
    """(axis, votes) for the model's front, from front/rear name tokens and position."""
    lo, hi = _world_bbox(meshes)
    ext = hi - lo
    ax = 0 if ext[0] > ext[1] else 1
    centre, half = (lo[ax] + hi[ax]) / 2.0, max(ext[ax] / 2.0, 1e-6)
    score = 0.0
    for o in meshes:
        w = _words(o.name, *(s.material.name for s in o.material_slots if s.material), *_ancestor_names(o)[:2])
        vote = int(_has(w, "front") and not _has(w, "front_never")) - int(_has(w, "rear"))
        if vote:
            a, b = _world_bbox([o])
            c = ((a[ax] + b[ax]) / 2.0 - centre) / half
            score += vote * max(-1.0, min(1.0, c))
    letter = "XY"[ax]
    if abs(score) < 0.05:
        # glTF convention: the asset's front is +Z, which the importer maps to Blender -Y.
        return ("-Y" if ax == 1 else "+X"), round(score, 3)
    return ("+" if score > 0 else "-") + letter, round(score, 3)


def _scale_factor(length: float) -> float:
    if 3.0 <= length <= 6.5:
        return 1.0
    for f in (0.01, 0.001, 0.1, 10.0, 100.0, 0.0254, 1.0 / 0.0254):
        if 3.0 <= length * f <= 6.5:
            return f
    return 4.6 / max(length, 1e-6)


def _find_wheels(meshes, glass_meshes) -> dict:
    """Quadrant -> list of wheel mesh objects (car already normalised: front +Y, at origin)."""
    lo, hi = _world_bbox(meshes)
    L, H = hi[1] - lo[1], hi[2] - lo[2]
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0

    def quadrant(o):
        a, b = _world_bbox([o])
        return ("F" if (a[1] + b[1]) / 2.0 > cy else "R") + ("L" if (a[0] + b[0]) / 2.0 < cx else "R"), a, b

    def by_names():
        found = {}
        for o in meshes:
            own = _words(o.name)
            anc = _words(*_ancestor_names(o))
            mats = _words(*(s.material.name for s in o.material_slots if s.material))
            if _has(own + anc, "wheel_never") and not _has(own, "wheel_always"):
                continue
            if not (_has(own + anc + mats, "wheel")):
                continue
            q, a, b = quadrant(o)
            if (b[1] - a[1]) > 0.35 * L or (b[2] - a[2]) > 0.8 * H or (a[2] - lo[2]) > 0.2 * H:
                continue
            found.setdefault(q, []).append(o)
        return found

    def by_shape():
        found = {}
        for o in meshes:
            if o in glass_meshes:
                continue
            q, a, b = quadrant(o)
            d = b - a
            if not (0.4 <= d[2] <= 1.2 and abs(d[1] - d[2]) < 0.3 * d[2] and d[0] < 0.7 * d[2]):
                continue
            if (a[2] - lo[2]) > 0.08 * H:
                continue
            found.setdefault(q, []).append(o)
        return found

    found = by_names()
    if len(found) < 4:
        found = by_shape()
    return found if len(found) == 4 else {}


def _wheel_frame(obs):
    """(centre Vector, radius, axle unit Vector pointing to +X side) from the meshes' vertices."""
    import numpy as np
    from mathutils import Vector  # pylint: disable=import-error
    pts = _mesh_points(obs)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2.0
    radius = (hi[2] - lo[2]) / 2.0
    cov = np.cov((pts - centre).T)
    vals, vecs = np.linalg.eigh(cov)
    axle = vecs[:, int(np.argmin(vals))]
    if axle[0] < 0:
        axle = -axle
    return Vector(centre.tolist()), float(radius), Vector(axle.tolist()).normalized()


def _set_glass(mat) -> bool:
    if not mat.use_nodes or mat.node_tree is None:
        return False
    changed = False
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            node.inputs["Transmission Weight"].default_value = 1.0
            rough = node.inputs["Roughness"]
            if not rough.is_linked:
                rough.default_value = min(rough.default_value, 0.05)
            node.inputs["IOR"].default_value = 1.5
            changed = True
    return changed


def _paint_fallback(meshes) -> list:
    """Material with the largest surface on the remaining car meshes."""
    import numpy as np
    area = {}
    for o in meshes:
        me = o.data
        n = len(me.polygons)
        if n == 0 or not o.material_slots:
            continue
        a = np.empty(n)
        idx = np.empty(n, dtype=np.int32)
        me.polygons.foreach_get("area", a)
        me.polygons.foreach_get("material_index", idx)
        sums = np.bincount(idx, weights=a, minlength=len(o.material_slots))
        for i, slot in enumerate(o.material_slots):
            if slot.material is not None and i < len(sums):
                area[slot.material.name] = area.get(slot.material.name, 0.0) + float(sums[i])
    return [max(area, key=area.get)] if area else []


def _tri_count(meshes) -> int:
    import numpy as np
    total = 0
    for me in {o.data for o in meshes}:
        n = len(me.polygons)
        if n:
            lt = np.empty(n, dtype=np.int32)
            me.polygons.foreach_get("loop_total", lt)
            total += int((lt - 2).sum())
    return total


def _drop_helper_meshes(objs) -> tuple:
    """Delete leaf meshes whose every material is fully transparent (Principled Alpha 0, not
    textured): backdrop boxes and shadow planes some exporters ship with the model. They never
    render but would inflate the bounding box that drives scale, framing and masks.
    Returns (remaining objects, removed names)."""
    import bpy  # pylint: disable=import-error

    def invisible(mat) -> bool:
        nt = getattr(mat, "node_tree", None) if mat is not None else None
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None) if nt else None
        if bsdf is None:
            return False
        alpha = bsdf.inputs["Alpha"]
        return not alpha.is_linked and alpha.default_value <= 0.001

    helpers = {o.name for o in objs if o.type == "MESH" and not o.children and o.material_slots
               and all(invisible(s.material) for s in o.material_slots)}
    remaining = [o for o in objs if o.name not in helpers]
    for name in sorted(helpers):
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    return remaining, sorted(helpers)


def _coaxial_parts(meshes, claimed: set, centre, radius: float, axis) -> list:
    """Meshes that turn with a wheel but are separate objects from its tyre: rim and spokes,
    brake disc, hub and centre nut. They are centred on the axle and fit inside the tyre; a
    caliper sits off the axle and stays with the body. Found meshes are added to `claimed`."""
    from mathutils import Vector  # pylint: disable=import-error
    found = []
    for o in meshes:
        if o in claimed:
            continue
        pts = [o.matrix_world @ Vector(corner) for corner in o.bound_box]
        lo = Vector([min(p[i] for p in pts) for i in range(3)])
        hi = Vector([max(p[i] for p in pts) for i in range(3)])
        d = (lo + hi) / 2.0 - centre
        along = d.dot(axis)
        off_axis = (d - axis * along).length
        if off_axis < 0.1 * radius and abs(along) < 1.2 * radius and max(hi - lo) < 2.1 * radius:
            found.append(o)
            claimed.add(o)
    return found


def _normalise_and_classify(job, new_obs) -> None:
    """Scale, orient, ground and classify freshly imported objects; build root and pivots."""
    import json
    import math
    import bpy  # pylint: disable=import-error
    from mathutils import Matrix, Vector  # pylint: disable=import-error

    new_obs, removed = _drop_helper_meshes(new_obs)
    meshes = [o for o in new_obs if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("The glTF contains no meshes")
    tops = [o for o in new_obs if o.parent is None]

    def transform_tops(M):
        bpy.context.view_layer.update()
        for o in tops:
            o.matrix_world = M @ o.matrix_world
        bpy.context.view_layer.update()

    # Scale and orientation.
    lo, hi = _world_bbox(meshes)
    raw_length = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    scale = job["length_m"] / raw_length if job["length_m"] > 0 else _scale_factor(raw_length)
    if job["forward_axis"] in _AXIS_YAW:
        axis, votes = job["forward_axis"], None
    else:
        axis, votes = _guess_forward(meshes)
    yaw = math.radians(_AXIS_YAW[axis])
    transform_tops(Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(scale, 4))

    # Materials by kind.
    mats = {s.material for o in meshes for s in o.material_slots if s.material is not None}
    kinds = {m.name: _material_kind(m.name) for m in mats}
    glass_meshes = {o for o in meshes
                    if o.material_slots and all(s.material and kinds[s.material.name] == "glass" for s in o.material_slots)}

    # Wheels and the contact centre.
    wheel_groups = _find_wheels(meshes, glass_meshes)
    wheel_frames = {q: _wheel_frame(obs) for q, obs in wheel_groups.items()}
    if wheel_frames:
        cs = [c for c, _r, _a in wheel_frames.values()]
        contact = Vector((sum(c.x for c in cs) / 4.0, sum(c.y for c in cs) / 4.0,
                          min(c.z - r for c, r, _a in wheel_frames.values())))
    else:
        lo, hi = _world_bbox(meshes)
        contact = Vector(((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]))
    transform_tops(Matrix.Translation(-contact))
    wheel_frames = {q: (c - contact, r, a) for q, (c, r, a) in wheel_frames.items()}

    coll = _ts_collection(TS_CAR)
    root = bpy.data.objects.new("TS_CAR_ROOT", None)
    root.empty_display_type = "ARROWS"
    root.empty_display_size = 1.0
    root["ts_role"] = "car_other"
    coll.objects.link(root)
    for o in tops:
        mw = o.matrix_world.copy()
        o.parent = root
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = mw

    # Wheel pivots: local X = axle (towards the car's right), local Z ~ up, at the hub centre.
    wheel_meshes = set()
    claimed = {o for obs in wheel_groups.values() for o in obs} | glass_meshes
    pivots = []
    for q in ("FL", "FR", "RL", "RR"):
        if q not in wheel_frames:
            continue
        c, r, a = wheel_frames[q]
        z = Vector((0.0, 0.0, 1.0))
        z = (z - a * z.dot(a)).normalized()
        y = z.cross(a)
        P = Matrix((
            (a.x, y.x, z.x, c.x),
            (a.y, y.y, z.y, c.y),
            (a.z, y.z, z.z, c.z),
            (0.0, 0.0, 0.0, 1.0),
        ))
        piv = bpy.data.objects.new("TS_WHEEL_" + q, None)
        piv.empty_display_type = "CIRCLE"
        piv.empty_display_size = r
        coll.objects.link(piv)
        piv.parent = root
        piv.matrix_basis = P
        piv["ts_role"] = "car_wheel"
        piv["ts_wheel_pivot"] = True
        piv["ts_wheel_pos"] = q
        piv["ts_wheel_radius_m"] = round(r, 4)
        pinv = P.inverted()
        for o in list(wheel_groups[q]) + _coaxial_parts(meshes, claimed, c, r, a):
            mw = o.matrix_world.copy()
            o.parent = piv
            o.matrix_parent_inverse = Matrix.Identity(4)
            o.matrix_basis = pinv @ mw
            wheel_meshes.add(o)
        pivots.append(piv)
    bpy.context.view_layer.update()

    # Roles and glass.
    paint = sorted(n for n, k in kinds.items() if k == "paint")
    glass_mats = sorted(n for n, k in kinds.items() if k == "glass")
    for name in glass_mats:
        _set_glass(bpy.data.materials[name])
    rest = [o for o in meshes if o not in wheel_meshes and o not in glass_meshes]
    if not paint:
        paint = _paint_fallback([o for o in rest if not any(
            s.material and kinds[s.material.name] == "lamp" for s in o.material_slots)])
        for n in paint:
            kinds[n] = "paint"
    for o in new_obs:
        if o in wheel_meshes:
            role = "car_wheel"
        elif o in glass_meshes:
            role = "car_glass"
        elif o.type != "MESH":
            role = "car_other"
        else:
            mk = {kinds[s.material.name] for s in o.material_slots if s.material}
            w = _words(o.name)
            if "lamp" in mk or (_has(w, "lamp") and not _has(w, "lamp_never")):
                role = "car_lamp"
            elif "paint" in mk:
                role = "car_body"
            else:
                role = "car_other"
        o["ts_role"] = role

    lo, hi = _world_bbox(meshes)
    radius = sum(f[1] for f in wheel_frames.values()) / len(wheel_frames) if wheel_frames else 0.34
    root["ts_car_key"] = job["car_key"]
    root["ts_dims_m"] = [round(float(hi[1] - lo[1]), 4), round(float(hi[0] - lo[0]), 4), round(float(hi[2] - lo[2]), 4)]
    root["ts_bbox_local"] = [round(float(v), 4) for v in (*lo, *hi)]
    root["ts_wheel_radius_m"] = round(float(radius), 4)
    root["ts_wheels"] = [p.name for p in pivots]
    root["ts_wheel_axis_local"] = _WHEEL_AXIS_LOCAL
    root["ts_wheel_forward_sign"] = _WHEEL_FORWARD_SIGN
    root["ts_paint_materials"] = paint
    root["ts_glass_materials"] = glass_mats
    root["ts_load_meta"] = json.dumps({
        "forward_axis_used": axis,
        "forward_votes": votes,
        "scale_applied": round(scale, 6),
        "tri_count": _tri_count(meshes),
        "materials": [{"name": n, "kind": kinds[n]} for n in sorted(kinds)],
        "wheels_found": bool(wheel_frames),
        "helpers_removed": removed,
    })
    job["root"] = root.name


def _root_result(root, job, from_cache: bool) -> dict:
    import json
    import time
    meta = json.loads(str(root.get("ts_load_meta", "{}")))
    return _ts_result(
        "ok",
        car_key=str(root["ts_car_key"]),
        root=root.name,
        dims_m=[float(v) for v in root["ts_dims_m"]],
        wheels=[str(w) for w in root["ts_wheels"]],
        wheel_radius_m=float(root["ts_wheel_radius_m"]),
        paint_materials=[str(m) for m in root["ts_paint_materials"]],
        forward_axis_used=meta.get("forward_axis_used"),
        forward_votes=meta.get("forward_votes"),
        scale_applied=meta.get("scale_applied"),
        tri_count=meta.get("tri_count"),
        materials=meta.get("materials", []),
        helpers_removed=meta.get("helpers_removed", []),
        wheels_found=meta.get("wheels_found", False),
        import_s=round(job["import_s"], 2),
        download_s=round(job.get("download_s", 0.0), 2),
        phases_s={k: round(v, 2) for k, v in job.get("phases_s", {}).items()},
        total_s=round(time.monotonic() - job["t0"], 2),
        from_cache=from_cache,
    )


def _view_car(root) -> None:
    """Without a shot camera, point the 3D viewports at the car (no operators)."""
    import bpy  # pylint: disable=import-error
    from mathutils import Euler, Vector  # pylint: disable=import-error
    if bpy.context.scene.camera is not None:
        return
    length = float(root["ts_dims_m"][0])
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                r3d = area.spaces.active.region_3d
                r3d.view_perspective = "PERSP"
                r3d.view_location = root.matrix_world @ Vector((0.0, 0.0, 0.6))
                r3d.view_rotation = Euler((1.25, 0.0, 2.4), "XYZ").to_quaternion()
                r3d.view_distance = 2.2 * length


def _step(job) -> float | None:
    """Time one phase of the load job (see `_step_phase`)."""
    import time
    t = time.monotonic()
    phase = job["phase"]
    try:
        return _step_phase(job)
    finally:
        times = job.setdefault("phases_s", {})
        times[phase] = times.get(phase, 0.0) + time.monotonic() - t


def _step_phase(job) -> float | None:
    """Advance the load job by one phase; return the next timer interval or None when done."""
    import os
    import time
    import traceback
    import bpy  # pylint: disable=import-error
    try:
        phase = job["phase"]
        if phase == "download":
            th = job.get("thread")
            if th is not None and th.is_alive():
                return 0.2
            if job.get("download_error"):
                raise RuntimeError(job["download_error"])
            job["download_s"] = time.monotonic() - job["t0"]
            job["phase"] = "clear"
            return 0.01
        if phase == "clear":
            old = _ts_car_root()
            job["placement"] = [list(r) for r in old.matrix_world] if old is not None else None
            _ts_clear_collection(TS_CAR)
            bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
            _ts_collection(TS_CAR)
            job["phase"] = "import"
            return 0.01
        if phase == "import":
            t = time.monotonic()
            coll = _ts_collection(TS_CAR)
            if job["from_cache"]:
                with bpy.data.libraries.load(job["lib_path"], link=False) as (src, dst):
                    dst.objects = list(src.objects)
                for o in dst.objects:
                    if o is not None:
                        coll.objects.link(o)
                job["import_s"] = time.monotonic() - t
                job["phase"] = "finish"
                return 0.01
            before = {o.as_pointer() for o in bpy.data.objects}
            win = bpy.context.window_manager.windows[0]
            layer_coll = next(lc for lc in bpy.context.view_layer.layer_collection.children if lc.collection == coll)
            with bpy.context.temp_override(window=win, scene=bpy.context.scene):
                bpy.context.view_layer.active_layer_collection = layer_coll
                bpy.ops.import_scene.gltf(filepath=job["glb_path"], import_scene_as_collection=False)
            new_obs = [o for o in bpy.data.objects if o.as_pointer() not in before]
            for o in new_obs:
                for c in list(o.users_collection):
                    if c != coll:
                        c.objects.unlink(o)
                if coll not in o.users_collection:
                    coll.objects.link(o)
            job["new"] = [o.name for o in new_obs]
            job["import_s"] = time.monotonic() - t
            job["phase"] = "normalise"
            return 0.01
        if phase == "normalise":
            _normalise_and_classify(job, [bpy.data.objects[n] for n in job["new"]])
            job["phase"] = "write_cache"
            return 0.01
        if phase == "write_cache":
            root = bpy.data.objects[job["root"]]
            obs = {root, *root.children_recursive}
            tmp = job["lib_path"] + ".part.blend"
            bpy.data.libraries.write(tmp, obs, path_remap="NONE", fake_user=False, compress=False)
            os.replace(tmp, job["lib_path"])
            job["phase"] = "finish"
            return 0.01
        if phase == "finish":
            from mathutils import Matrix  # pylint: disable=import-error
            root = _ts_car_root()
            if root is None:
                raise RuntimeError("Car root not found after import")
            # Wheel pivot names may have changed on append.
            root["ts_wheels"] = sorted(c.name for c in root.children if c.get("ts_wheel_pivot"))
            if job.get("placement"):
                root.matrix_world = Matrix(job["placement"])
            _view_car(root)
            _ts_redraw()
            job["result"] = _root_result(root, job, job["from_cache"])
            job["phase"] = "done"
            return None
        return None
    except Exception:  # pylint: disable=broad-exception-caught
        job["result"] = _ts_result("error", message=traceback.format_exc()[-3000:], phase=job.get("phase"))
        job["phase"] = "done"
        return None


def _safe(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def main(params: Params):
    import functools
    import os
    import threading
    import time
    import bpy  # pylint: disable=import-error

    state = _ts_state()
    job = state.get("load_car")

    if params.action == "status":
        if job is None:
            return _ts_result("error", message="No car load in progress")
        if job["phase"] != "done":
            return _ts_result("running", phase=job["phase"])
        state.pop("load_car", None)
        return job["result"]

    if job is not None and job["phase"] != "done":
        return _ts_result("running", phase=job["phase"])
    if _ts_busy():
        return _ts_busy_result()
    if params.forward_axis not in FORWARD_AXES:
        return _ts_result("error", message="forward_axis must be one of {}".format(FORWARD_AXES))

    base = _safe(params.car_key) + "_" + _safe(params.etag)
    lib_path = os.path.join(_ts_cache_dir(), "car_{}_{}_{}_{}.blend".format(
        base, _CACHE_VERSION, _safe(params.forward_axis), _safe("{:.3f}".format(params.length_m))))
    job = {
        "phase": "download",
        "t0": time.monotonic(),
        "car_key": params.car_key,
        "forward_axis": params.forward_axis,
        "length_m": float(params.length_m),
        "lib_path": lib_path,
        "glb_path": os.path.join(_ts_cache_dir(), base + ".glb"),
        "from_cache": os.path.exists(lib_path),
        "import_s": 0.0,
    }
    if not job["from_cache"]:
        def download():
            try:
                _ts_http_get_cached(params.url, base + ".glb")
            except Exception as ex:  # pylint: disable=broad-exception-caught
                job["download_error"] = "Download failed: {}".format(ex)
        job["thread"] = threading.Thread(target=download, daemon=True)
        job["thread"].start()
    state["load_car"] = job
    bpy.app.timers.register(functools.partial(_step, job), first_interval=0.05)
    return _ts_result("started", from_cache=job["from_cache"])
