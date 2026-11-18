"""Smoke test of the Trueshot scene tools through the MCP protocol (streamable HTTP).

Starts nothing: run Blender (scripts/dev/blender_instance.sh start scene 9881) and the MCP
server (`blender-mcp --transport http --port 8781`) first, then:

    mcp/.venv/bin/python scripts/dev/scene_smoke.py                 # full run
    mcp/.venv/bin/python scripts/dev/scene_smoke.py --steps car,set  # some steps
    mcp/.venv/bin/python scripts/dev/scene_smoke.py call frame_shot azimuth_deg=90

Quick OpenGL renders (dev only, via the official `execute_blender_code` tool) are copied to
/tmp/scene_checks/ when the Windows temp folder is reachable from WSL.
"""

import argparse
import asyncio
import glob
import json
import os
import shutil
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

CHECK_DIR = "/tmp/scene_checks"
CARS = ("cars/ferrari.glb", "cars/CarConcept.glb")
TERRAIN_SHOTS = (
    ("alpine", "hairpin_left"),
    ("high_desert", "straight"),
    ("coastal_town", "curve_right"),
    ("city", "straight"),
    ("forest", "curve_left"),
    ("plains", "hairpin_right"),
)

_SNAPSHOT_CODE = r'''
import bpy, os, tempfile
scene = bpy.context.scene
sh = scene.display.shading
vs = scene.view_settings
world = scene.world or bpy.data.worlds.new("TS_snapshot_world")
scene.world = world
saved = (scene.render.resolution_percentage, sh.light, sh.color_type, scene.render.film_transparent,
         vs.view_transform, tuple(world.color), scene.render.engine)
scene.render.resolution_percentage = 50
sh.light, sh.color_type = ("FLAT" if FLAT else "STUDIO"), "OBJECT"
scene.render.film_transparent = False
vs.view_transform = "Standard"
world.color = (0.92, 0.92, 0.92)
scene.render.engine = "BLENDER_WORKBENCH"
win = bpy.context.window_manager.windows[0]
area = next(a for a in win.screen.areas if a.type == "VIEW_3D")
with bpy.context.temp_override(window=win, area=area):
    bpy.ops.render.opengl(view_context=False)
out = os.path.join(tempfile.gettempdir(), "trueshot_out", "scene_checks")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, NAME + ".png")
bpy.data.images["Render Result"].save_render(path)
(scene.render.resolution_percentage, sh.light, sh.color_type, scene.render.film_transparent,
 vs.view_transform, world.color, scene.render.engine) = saved
result = {"path": path}
'''


def _windows_temp() -> str | None:
    for path in glob.glob("/mnt/c/Users/*/AppData/Local/Temp/trueshot_out/scene_checks"):
        return path
    return None


class Client:
    def __init__(self, session: ClientSession) -> None:
        self.session = session
        self.failures: list[str] = []

    async def call(self, tool: str, **args):
        t = time.monotonic()
        res = await self.session.call_tool(tool, args)
        dt = time.monotonic() - t
        text = "".join(getattr(c, "text", "") for c in res.content)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"status": "error", "message": text}
        short = json.dumps(data)
        print(f"[{dt:6.2f}s] {tool}({', '.join(f'{k}={v!r}' for k, v in args.items())}) -> "
              f"{short[:600]}{'…' if len(short) > 600 else ''}", flush=True)
        data["_elapsed_s"] = round(dt, 2)
        if data.get("status") != "ok" and tool != "execute_blender_code":
            self.failures.append(f"{tool}: {data.get('message', data)}"[:300])
        return data

    def check(self, ok: bool, what: str) -> None:
        print(f"    {'PASS' if ok else 'FAIL'}: {what}", flush=True)
        if not ok:
            self.failures.append(what)

    async def snapshot(self, name: str, flat: bool = True) -> None:
        code = "NAME = {!r}\nFLAT = {!r}\n".format(name, flat) + _SNAPSHOT_CODE
        await self.call("execute_blender_code", code=code)
        src_dir = _windows_temp()
        if src_dir:
            os.makedirs(CHECK_DIR, exist_ok=True)
            shutil.copy(os.path.join(src_dir, name + ".png"), os.path.join(CHECK_DIR, name + ".png"))


async def step_setup(c: Client) -> None:
    r = await c.call("setup_blender", use_gpu=True)
    c.check(r.get("device_type") == "OPTIX", "setup_blender device_type OPTIX")


async def step_car(c: Client) -> None:
    for key in CARS:
        first = await c.call("load_car", car_key=key)
        second = await c.call("load_car", car_key=key)
        for tag, r in (("first", first), ("second", second)):
            print(f"    {tag}: " + ", ".join(f"{k}={r.get(k)}" for k in (
                "from_cache", "download_s", "import_s", "phases_s", "total_s", "_elapsed_s", "wheel_radius_m", "dims_m")))
        c.check(second.get("from_cache") is True, f"{key}: second load from_cache")
        c.check(second.get("_elapsed_s", 1e9) < first.get("_elapsed_s", 0), f"{key}: cached load faster")
        dims = first.get("dims_m") or [0, 0, 0]
        c.check(3.0 <= dims[0] <= 6.5 and dims[0] > dims[1], f"{key}: plausible dims {dims}")
        paint = await c.call("set_materials", assignments={"auto": {"preset": "car_paint", "base": "#1E5AA8"}})
        c.check(bool(paint.get("changed")), f"{key}: set_materials auto changed {paint.get('changed')}")
        c.check(len(first.get("wheels", [])) == 4, f"{key}: four wheels")
        lamps = await c.call("execute_blender_code", code=_FRONT_CHECK_CODE)
        res = lamps.get("result", lamps)
        c.check(res.get("front_lamp_y", 0) > 0 and res.get("rear_lamp_y", 0) <= 0,
                f"{key}: head lamps at +Y ({res})")


_FRONT_CHECK_CODE = r'''
import bpy
from mathutils import Vector
root = next(o for o in bpy.data.collections["TS_CAR"].all_objects if "ts_car_key" in o)
inv = root.matrix_world.inverted()
def mean_y(words):
    ys = []
    for o in root.children_recursive:
        if o.type != "MESH":
            continue
        names = (o.name + " " + " ".join(s.material.name for s in o.material_slots if s.material)).lower()
        if any(w in names for w in words):
            c = sum(((inv @ o.matrix_world) @ Vector(v) for v in o.bound_box), Vector()) / 8
            ys.append(c.y)
    return sum(ys) / len(ys) if ys else 0.0
result = {"front_lamp_y": round(mean_y(("headl", "projector")), 3), "rear_lamp_y": round(mean_y(("taill", "brakelight")), 3)}
'''


async def step_set(c: Client) -> None:
    for terrain, shape in TERRAIN_SHOTS:
        r = await c.call("build_set", road_shape=shape, terrain=terrain, seed=3)
        c.check(r.get("status") == "ok" and r.get("_elapsed_s", 99) < 6, f"build_set {terrain}/{shape}")
        await c.call("place_car_on_road", distance_m=0.0, lane="right")
        f = await c.call("frame_shot", azimuth_deg=35.0, elevation_deg=6.0)
        c.check(bool(f.get("all_in_frame")), f"{terrain}: car in frame")
        await c.snapshot(f"set_{terrain}_{shape}_hero")
        # Wide view from behind-left along the direction of travel: road to the horizon.
        await c.call("frame_shot", azimuth_deg=165.0, elevation_deg=5.0, car_width_frac=0.22, focal_mm=35.0)
        await c.snapshot(f"set_{terrain}_{shape}_wide")


async def step_frame(c: Client) -> None:
    await c.call("build_set", road_shape="curve_left", terrain="plains", seed=1)
    await c.call("place_car_on_road", distance_m=10.0, lane="right")
    for az in (0.0, 35.0, 90.0, 180.0, 270.0):
        f = await c.call("frame_shot", azimuth_deg=az, elevation_deg=8.0, car_width_frac=0.55)
        frac = f.get("car_bbox_frac") or [0, 0, 0, 0]
        c.check(bool(f.get("all_in_frame")) and abs((frac[2] - frac[0]) - 0.55) < 0.03,
                f"frame az={az}: in frame, width {frac[2] - frac[0]:.3f}")
        await c.snapshot(f"frame_az{int(az):03d}")
    f = await c.call("frame_shot", azimuth_deg=30.0, elevation_deg=4.0, distance_m=12.0, frame_x=0.4, frame_y=0.7)
    c.check(f.get("status") == "ok", "frame with fixed distance")
    await c.snapshot("car_studio_az030", flat=False)


async def step_light(c: Client, hdri_key: str) -> None:
    r = await c.call("set_lighting", sky="procedural", sun_azimuth_deg=120.0, sun_elevation_deg=30.0)
    c.check(r.get("status") == "ok", "procedural sky")
    r = await c.call("set_lighting", sky="hdri", hdri_key=hdri_key, hdri_rotation_deg=45.0, strength=1.0)
    c.check(r.get("status") == "ok", f"hdri sky {hdri_key}")


async def step_reset(c: Client) -> None:
    info = await c.call("scene_info")
    c.check(info.get("car") is not None and info.get("set") is not None, "scene_info shows car and set")
    await c.call("reset_scene", keep_car=True)
    info = await c.call("scene_info")
    c.check(info.get("car") is not None and info.get("set") is None and info.get("camera") is None,
            "reset keeps the car, removes set and rig")


async def run(url: str, steps: list[str], hdri_key: str, call_args: list[str]) -> int:
    async with streamable_http_client(url) as (read, write, _sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            c = Client(session)
            if call_args:
                kwargs = {}
                for item in call_args[1:]:
                    k, v = item.split("=", 1)
                    try:
                        kwargs[k] = json.loads(v)
                    except json.JSONDecodeError:
                        kwargs[k] = v
                await c.call(call_args[0], **kwargs)
                return 0
            table = {
                "setup": step_setup,
                "car": step_car,
                "set": step_set,
                "frame": step_frame,
                "light": lambda cl: step_light(cl, hdri_key),
                "reset": step_reset,
            }
            for name in steps:
                print(f"== {name}", flush=True)
                await table[name](c)
            print("\nFAILURES:\n  " + "\n  ".join(c.failures) if c.failures else "\nALL PASS")
            return 1 if c.failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8781/")
    ap.add_argument("--steps", default="setup,car,set,frame,light,reset")
    ap.add_argument("--hdri-key", default="hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr")
    ap.add_argument("call", nargs="*", help="`call <tool> key=value ...` for one ad-hoc call")
    a = ap.parse_args()
    call_args = a.call[1:] if a.call and a.call[0] == "call" else []
    return asyncio.run(run(a.url, a.steps.split(","), a.hdri_key, call_args))


if __name__ == "__main__":
    sys.exit(main())
