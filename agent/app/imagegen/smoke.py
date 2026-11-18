"""Smoke-test one plate generation call from the command line.

    uv run python -m app.imagegen.smoke --backend fal --model fal-ai/z-image/turbo/controlnet \
        --guide path.png --prompt "..." --n 1 --out /tmp/x/

`--guide` is the grey guide; `--lines` / `--depth` default to it (pass the real line / depth
guides for ControlNet endpoints). `--ref` may repeat. `--dry-run` prints the request shape
(no uploads, no network, no key needed). Without `--dry-run` it refuses to run when the
backend's key (OPENAI_API_KEY / FAL_KEY) is missing. `--env-file` loads KEY=VALUE lines
(existing environment variables win).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from . import MissingCredentialsError, PlateRequest, get_backend


def _load_env_file(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _summarise(obj):
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, tuple):
        return tuple(_summarise(x) for x in obj)
    if isinstance(obj, list):
        return [_summarise(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _summarise(v) for k, v in obj.items()}
    return obj


async def _dry_run(backend, req: PlateRequest) -> dict:
    if backend.name == "openai":
        return _summarise(backend.build_request(req))
    if backend.name == "fal":
        from .fal_backend import endpoint_for

        async def upload(data: bytes, content_type: str, file_name: str) -> str:
            return f"<upload {file_name} {content_type} {len(data)} bytes>"

        builder, _, family = endpoint_for(backend.model)
        return {"endpoint": backend.model, "family": family, "arguments": await builder(req, upload, backend.options)}
    return {"backend": backend.name, "note": "no remote request"}


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.imagegen.smoke", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", required=True, help="openai | fal | fake")
    ap.add_argument("--model", default=None, help="model / fal endpoint id (default: backend default)")
    ap.add_argument("--guide", required=True, type=Path, help="grey guide PNG")
    ap.add_argument("--lines", type=Path, help="line guide PNG (default: --guide)")
    ap.add_argument("--depth", type=Path, help="depth guide PNG (default: --guide)")
    ap.add_argument("--ref", type=Path, action="append", default=[], help="reference photo (repeatable)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--control-guide", choices=("depth", "lines"), default="depth", help="fal ControlNet input")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.env_file:
        _load_env_file(a.env_file)
    backend = get_backend(a.backend, a.model)
    if backend.name == "fal":
        backend.options["control_guide"] = a.control_guide
    grey = a.guide.read_bytes()
    req = PlateRequest(
        guide_grey=grey,
        guide_lines=a.lines.read_bytes() if a.lines else grey,
        guide_depth=a.depth.read_bytes() if a.depth else grey,
        references=[p.read_bytes() for p in a.ref],
        prompt=a.prompt,
        n=a.n,
        width=a.width,
        height=a.height,
        seed=a.seed,
    )
    est = backend.estimate_cost(req)
    print(f"backend={backend.name} model={backend.model} n={a.n} size={a.width}x{a.height} estimated_cost=${est:.4f}")
    if a.dry_run:
        print(json.dumps(await _dry_run(backend, req), indent=2, default=str))
        return 0
    if backend.name != "fake":
        try:
            backend.client  # the backend's own key lookup
        except MissingCredentialsError as e:
            print(f"refusing to run: {e}", file=sys.stderr)
            return 2

    t0 = time.monotonic()
    res = await backend.generate(req)
    dt = time.monotonic() - t0
    a.out.mkdir(parents=True, exist_ok=True)
    for k, png in enumerate(res.images):
        (a.out / f"plate_{k}.png").write_bytes(png)
    report = {"backend": res.backend, "model": res.model, "cost_usd": res.cost_usd, "seconds": round(dt, 1),
              "images": len(res.images), "meta": res.meta}
    (a.out / "result.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
