"""Dev smoke test for the render tools, through the MCP protocol (streamable HTTP).

Run the MCP server locally first (see the task notes), with a scene in Blender, then:

    cd mcp && .venv/bin/python ../scripts/dev/render_smoke.py --url http://127.0.0.1:8782/ \
        --blender 172.24.112.1:9882 --out /tmp/render_smoke [--spokes 5 10] [--samples 64] \
        [--scene-tools --car-key cars/ferrari.glb --terrain alpine --azimuth 35]

What it does:
- render_preview (timed), render_guides, render_car for each spoke count, viewport_snapshot;
- during render_car: pings Blender over the raw socket every 0.25 s (worst latency = longest
  main-thread block), samples `nvidia-smi` VRAM, and checks that render_preview and a raw
  scene-changing tool-code guard (`_ts_busy`) return busy;
- downloads every uploaded object into --out (S3 at localhost:8333, dev credentials).
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import threading
import time
import uuid

import boto3
from botocore.config import Config
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def raw_call(host: str, port: int, code: str, timeout: float = 30.0) -> dict:
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall((json.dumps({"type": "execute", "code": code, "strict_json": True}) + "\0").encode())
        buf = b""
        while not buf.endswith(b"\0"):
            chunk = s.recv(1 << 16)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.rstrip(b"\0"))


class Probe:
    """Background sampling of Blender socket latency and GPU memory."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.lat: list[float] = []
        self.vram: list[int] = []
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._ping, daemon=True),
                         threading.Thread(target=self._gpu, daemon=True)]

    def _ping(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                raw_call(self.host, self.port, "result = {'ok': True}")
                self.lat.append(time.perf_counter() - t)
            except OSError:
                self.lat.append(float("inf"))
            time.sleep(0.25)

    def _gpu(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.vram.append(int(out.split()[0]))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            time.sleep(0.5)

    def __enter__(self) -> "Probe":
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)

    def summary(self) -> dict:
        return {"pings": len(self.lat), "max_ping_s": round(max(self.lat or [0]), 3),
                "vram_peak_mib": max(self.vram or [0]), "vram_min_mib": min(self.vram or [0])}


BUSY_GUARD = """
import bpy
def _ts_busy():
    return bool(bpy.app.is_job_running("RENDER"))
result = {"status": "busy"} if _ts_busy() else {"status": "ok", "note": "would change the scene"}
"""


def SCENE_STEPS(a: argparse.Namespace) -> list:
    """Scene built through the scene tools (--scene-tools), contract defaults otherwise."""
    return [
        ("setup_blender", {}),
        ("reset_scene", {"keep_car": False}),
        ("load_car", {"car_key": a.car_key}),
        ("build_set", {"road_shape": a.road_shape, "terrain": a.terrain}),
        ("place_car_on_road", {"distance_m": 0.0, "lane": "right"}),
        ("frame_shot", {"azimuth_deg": a.azimuth, "elevation_deg": a.elevation, "focal_mm": a.focal}),
        ("set_lighting", {}),
    ]


async def tool(session: ClientSession, name: str, **args) -> dict:
    t = time.perf_counter()
    res = await session.call_tool(name, args)
    dt = time.perf_counter() - t
    data = res.structuredContent or {}
    data = data.get("result", data)
    if not data and res.content:
        data = json.loads(res.content[0].text)
    print(f"--- {name}({args}) {dt:.2f} s -> status={data.get('status')}")
    data["_client_s"] = round(dt, 2)
    return data


async def run(a: argparse.Namespace) -> None:
    host, port = a.blender.split(":")
    prefix = f"jobs/dev-{uuid.uuid4().hex[:8]}/a1/"
    report: dict = {"job_prefix": prefix}
    async with streamablehttp_client(a.url, timeout=60, sse_read_timeout=3600) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            names = sorted(t.name for t in (await session.list_tools()).tools)
            print("tools:", names)
            if a.scene_tools:
                for name, args in SCENE_STEPS(a):
                    res = await tool(session, name, **args)
                    if res.get("status") != "ok":
                        raise SystemExit(f"{name} failed: {json.dumps(res)[:2000]}")
                    report.setdefault("scene", {})[name] = {k: v for k, v in res.items() if k != "_client_s"}
            report["preview"] = await tool(session, "render_preview", job_prefix=prefix, index=0)
            report["guides"] = await tool(session, "render_guides", job_prefix=prefix)
            for n in a.spokes:
                sub = f"{prefix}s{n}/"
                with Probe(host, int(port)) as probe:
                    task = asyncio.create_task(tool(session, "render_car", job_prefix=sub, samples=a.samples,
                                                    spoke_count=n, speed_kmh=a.speed))
                    await asyncio.sleep(4.0)
                    busy_preview = await tool(session, "render_preview", job_prefix=sub, index=9)
                    busy_raw = raw_call(host, int(port), BUSY_GUARD)
                    car = await task
                car["probe"] = probe.summary()
                car["busy_preview"] = busy_preview.get("status")
                car["busy_raw"] = busy_raw.get("result", {}).get("status")
                report[f"car_s{n}"] = car
            report["snapshot"] = await tool(session, "viewport_snapshot", job_prefix=prefix, index=0)

    s3 = boto3.client("s3", endpoint_url="http://localhost:8333", aws_access_key_id="trueshot",
                      aws_secret_access_key="trueshot-dev-secret", region_name="us-east-1",
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
    out = os.path.join(a.out, prefix.strip("/").replace("/", "_"))
    os.makedirs(out, exist_ok=True)
    listing = s3.list_objects_v2(Bucket="trueshot", Prefix=prefix).get("Contents", [])
    for obj in listing:
        dest = os.path.join(out, obj["Key"][len(prefix):].replace("/", "_"))
        s3.download_file("trueshot", obj["Key"], dest)
    report["downloaded"] = {o["Key"]: o["Size"] for o in listing}
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1)[:12000])
    print("outputs in", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8782/")
    ap.add_argument("--blender", required=True, help="host:port of the Blender add-on (for latency pings)")
    ap.add_argument("--out", default="/tmp/render_smoke")
    ap.add_argument("--spokes", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--speed", type=float, default=40.0)
    ap.add_argument("--scene-tools", action="store_true", help="build the scene with the scene tools first")
    ap.add_argument("--car-key", default="cars/ferrari.glb")
    ap.add_argument("--road-shape", default="curve_left")
    ap.add_argument("--terrain", default="alpine")
    ap.add_argument("--azimuth", type=float, default=35.0)
    ap.add_argument("--elevation", type=float, default=4.0)
    ap.add_argument("--focal", type=float, default=50.0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
