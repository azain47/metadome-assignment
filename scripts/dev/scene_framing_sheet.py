"""Grey-guide contact sheet of the stand-in sets at the planner's usual framings.

For 6 terrains x 2 road shapes x azimuth 35/150 (elevation 5, frame_shot defaults otherwise)
it builds the set through the MCP tools, quick-renders a flat grey guide (Workbench, object
colours = role tones, sky = 0.92) and measures the sky fraction. Writes
/tmp/scene_checks/framing/*.png and /tmp/scene_checks/framing_sheet.png.

Needs Blender + the MCP server running (see scene_smoke.py), then:
    uv run --no-project --with "mcp==1.30.0" --with pillow --with numpy \
        python scripts/dev/scene_framing_sheet.py [--car cars/ferrari.glb] [--seed 3]
"""

import argparse
import asyncio
import glob
import json
import os
import shutil

import numpy as np
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image, ImageDraw

OUT = "/tmp/scene_checks"
SHOTS = (  # terrain, two road shapes
    ("alpine", ("hairpin_left", "curve_right")),
    ("high_desert", ("straight", "curve_left")),
    ("coastal_town", ("curve_right", "straight")),
    ("city", ("straight", "curve_left")),
    ("forest", ("curve_left", "hairpin_right")),
    ("plains", ("straight", "hairpin_left")),
)
AZIMUTHS = (35.0, 150.0)
SKY_SRGB = (238, 252)  # sky 0.92 linear -> ~245 in Standard view; the white car is 255

_GUIDE_CODE = r'''
import bpy, os, tempfile
scene = bpy.context.scene
sh, vs = scene.display.shading, scene.view_settings
world = scene.world or bpy.data.worlds.new("TS_guide_world")
scene.world = world
saved = (scene.render.resolution_percentage, sh.light, sh.color_type, scene.render.film_transparent,
         vs.view_transform, tuple(world.color), scene.render.engine)
scene.render.resolution_percentage = 50
sh.light, sh.color_type = "FLAT", "OBJECT"
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
bpy.data.images["Render Result"].save_render(os.path.join(out, NAME + ".png"))
(scene.render.resolution_percentage, sh.light, sh.color_type, scene.render.film_transparent,
 vs.view_transform, world.color, scene.render.engine) = saved
result = {"ok": True}
'''


async def call(session, tool, **args):
    res = await session.call_tool(tool, args)
    text = "".join(getattr(c, "text", "") for c in res.content)
    data = json.loads(text)
    if tool != "execute_blender_code" and data.get("status") != "ok":
        raise RuntimeError(f"{tool}{args}: {data}")
    return data


def sky_fraction(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    lo, hi = SKY_SRGB
    sky = np.all((a >= lo) & (a <= hi), axis=2)
    return float(sky.mean())


async def run(url, car, seed):
    os.makedirs(os.path.join(OUT, "framing"), exist_ok=True)
    win_dir = None
    rows = []
    async with streamable_http_client(url) as (read, write, _sid):
        async with ClientSession(read, write) as s:
            await s.initialize()
            await call(s, "setup_blender", use_gpu=True)
            await call(s, "load_car", car_key=car)
            for terrain, shapes in SHOTS:
                row = []
                for shape in shapes:
                    built = await call(s, "build_set", road_shape=shape, terrain=terrain, seed=seed)
                    await call(s, "place_car_on_road")
                    for az in AZIMUTHS:
                        f = await call(s, "frame_shot", azimuth_deg=az, elevation_deg=5.0)
                        name = f"{terrain}_{shape}_az{int(az):03d}"
                        await call(s, "execute_blender_code", code=f"NAME = {name!r}\n" + _GUIDE_CODE)
                        win_dir = win_dir or next(iter(glob.glob(
                            "/mnt/c/Users/*/AppData/Local/Temp/trueshot_out/scene_checks")), None)
                        dst = os.path.join(OUT, "framing", name + ".png")
                        shutil.copy(os.path.join(win_dir, name + ".png"), dst)
                        frac = sky_fraction(dst)
                        row.append((name, dst, frac, f.get("horizon_y_frac"), built.get("build_s")))
                        print(f"{name:40s} sky {frac:5.1%}  horizon_y {f.get('horizon_y_frac')}  "
                              f"build {built.get('build_s')} s", flush=True)
                rows.append(row)

    tw, th = 384, 256
    sheet = Image.new("RGB", (tw * 4, th * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, (name, path, frac, _hy, _b) in enumerate(row):
            sheet.paste(Image.open(path).convert("RGB").resize((tw, th)), (c * tw, r * th))
            label = f"{name}  sky {frac:.0%}"
            draw.rectangle((c * tw, r * th, c * tw + 7 * len(label), r * th + 14), fill=(0, 0, 0))
            draw.text((c * tw + 3, r * th + 2), label, fill=(255, 255, 0))
    sheet.save(os.path.join(OUT, "framing_sheet.png"))
    fracs = [f for row in rows for _n, _p, f, _h, _b in row]
    print(f"sheet: {OUT}/framing_sheet.png  sky min {min(fracs):.1%} max {max(fracs):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8781/")
    ap.add_argument("--car", default="cars/ferrari.glb")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    asyncio.run(run(a.url, a.car, a.seed))


if __name__ == "__main__":
    main()
