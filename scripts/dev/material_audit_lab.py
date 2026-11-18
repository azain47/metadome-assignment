"""Material-audit lab: material sheets of the loaded car + reference photos -> set_materials assignments,
scored against the hand-made materials in a product-facts file.

    cd agent && uv run python ../scripts/dev/material_audit_lab.py --mcp http://127.0.0.1:8791/
    uv run python ../scripts/dev/material_audit_lab.py --ref <photo url> --car "<car name>" --views front34,rear34,side

1. `render_material_sheet` in Blender on the car that is loaded (raw import, before set_materials;
   `--load-car` loads one first), timed;
2. downloads the sheets and materials.json, fetches the reference photos (long side <= 768 px; default:
   the facts file's `reference_images`), runs `app.material_audit.audit_materials`;
3. prints agreement with the facts file's `materials` (same preset family; colour within dE76 15
   for coloured paint).

Writes outputs/lab/audit/: material_sheet_<k>.png, materials.json, refs/, cards/ (what the model saw),
result.json (assignments, verdicts, usage, agreement); spend is appended to outputs/lab/spend.jsonl.
`--car` defaults to the facts file's `name` (its `livery` is never sent: it is the answer).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
for _k, _v in dotenv_values(ROOT / ".env").items():
    if _v:
        os.environ.setdefault(_k, _v)
os.environ.setdefault("S3_ENDPOINT", "http://127.0.0.1:8333")
sys.path[:0] = [str(ROOT / "agent")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import material_audit as ma  # noqa: E402
from app.blender import BlenderClient  # noqa: E402
from app.config import settings  # noqa: E402
from app.references import Vlm  # noqa: E402
from app.storage import S3Store  # noqa: E402

LAB = ROOT / "outputs" / "lab"
DE_PAINT = 15.0

# Preset families for agreement: a near-miss inside a family (glass_dark vs glass_tinted) is not
# a material error the way paint vs carbon is.
FAMILY = {
    "car_paint": "paint", "metallic_paint": "paint", "matte_paint": "paint",
    "carbon_fibre": "dark_trim", "satin_trim": "dark_trim", "rubber_tyre": "rubber",
    "wheel_rim": "wheel", "metal_polished": "metal", "metal_brushed": "metal", "brake_disc": "metal",
    "glass_clear": "glass", "glass_tinted": "glass", "glass_dark": "glass",
    "lamp_lens": "lamp", "lamp_emissive": "lamp", "keep": "keep",
}


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")


def hex_lab(h: str) -> np.ndarray:
    rgb = np.array([[[int(h[i:i + 2], 16) for i in (1, 3, 5)]]], np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]  # L 0..100, a/b signed


def delta_e(a: str, b: str) -> float:
    return float(np.linalg.norm(hex_lab(a) - hex_lab(b)))


def agreement(truth: dict[str, dict[str, Any]], result: ma.AuditResult) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per facts material: predicted vs true preset/colour and a verdict."""
    got = {v.name: v for v in result.materials}
    rows = []
    for name, spec in truth.items():
        t_preset, t_base = spec.get("preset", ""), spec.get("base")
        v = got.get(name)
        row = {"name": name, "true": t_preset, "true_base": t_base, "pred": v.preset if v else None,
               "pred_base": v.base if v else None, "part": v.part if v else "", "conf": v.confidence if v else None,
               "area_pct": v.area_pct if v else None, "dE": None}
        if v is None:
            row["verdict"] = "not audited"
        elif t_preset == "keep":
            row["verdict"] = "n/a (truth keeps the material)"
        else:
            same = FAMILY.get(v.preset) == FAMILY.get(t_preset)
            ok = same
            if same and FAMILY[t_preset] == "paint" and t_base:
                row["dE"] = round(delta_e(v.base, t_base), 1) if v.base else None
                ok = row["dE"] is not None and row["dE"] <= DE_PAINT
            row["verdict"] = ("exact" if ok and v.preset == t_preset else "family") if ok else (
                "colour" if same else "WRONG")
        rows.append(row)
    scored = [r for r in rows if r["verdict"] in ("exact", "family", "colour", "WRONG")]
    area = sum(r["area_pct"] or 0.0 for r in scored) or 1.0
    summary = {
        "scored": len(scored),
        "agree": sum(r["verdict"] in ("exact", "family") for r in scored),
        "exact": sum(r["verdict"] == "exact" for r in scored),
        "agree_area_pct": round(100.0 * sum(r["area_pct"] or 0.0 for r in scored
                                             if r["verdict"] in ("exact", "family")) / area, 1),
        "not_audited": [r["name"] for r in rows if r["verdict"] == "not audited"],
    }
    return rows, summary


async def run(a: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = LAB / "audit"
    for sub in ("refs", "cards"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    facts = json.loads(Path(a.facts).read_text(encoding="utf-8")) if a.facts else {}
    refs_urls = a.ref or facts.get("reference_images") or []
    if not refs_urls:
        raise SystemExit("no reference photo: pass --ref URL")
    car = a.car or facts.get("name", "")

    bl = BlenderClient(a.mcp or settings.blender_mcp_url)
    store = S3Store(endpoint=settings.s3_endpoint, public_endpoint=settings.s3_public_endpoint,
                    access_key=settings.s3_access_key, secret_key=settings.s3_secret_key, bucket=settings.s3_bucket)
    if a.load_car:
        await bl.call("load_car", car_key=a.load_car, length_m=a.length_m)

    prefix = f"lab/audit/{int(time.time())}"
    t0 = time.monotonic()
    sheet_res = await bl.call("render_material_sheet", job_prefix=prefix, views=a.views,
                              max_materials=a.max_materials, cell_px=a.cell_px)
    sheet_wall = time.monotonic() - t0
    print(f"render_material_sheet: {sheet_wall:.1f} s wall (tool total {sheet_res.get('total_s')} s, "
          f"passes {sheet_res.get('pass_s')}), {sheet_res['materials']} materials on {sheet_res['sheets']} "
          f"sheet(s), {sheet_res['skipped']} skipped")
    for f in out.glob("material_sheet_*.png"):
        f.unlink()
    keys = sheet_res["keys"]
    index = json.loads(await store.get(keys["index"]))
    (out / "materials.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    sheets = []
    for k in range(sheet_res["sheets"]):
        data = await store.get(keys[f"sheet_{k}"])
        (out / f"material_sheet_{k}.png").write_bytes(data)
        sheets.append(data)

    refs = [await ma.fetch_reference(u) for u in refs_urls]
    for i, r in enumerate(refs):
        (out / "refs" / f"ref_{i}.jpg").write_bytes(r)
    for f in (out / "cards").glob("*.jpg"):
        f.unlink()
    for m, card in zip(index["materials"], ma.material_cards(sheets, index["materials"])):
        (out / "cards" / f"{m['index']:02d}.jpg").write_bytes(card)

    vlm = Vlm(max_side=ma.MAX_SIDE)
    print(f"model {vlm.model}: {len(index['materials'])} cards + {len(refs)} reference(s), car {car!r}")
    t1 = time.monotonic()
    result = await ma.audit_materials(sheets, index["materials"], refs, car, vlm=vlm, max_per_call=a.max_per_call)
    audit_s = time.monotonic() - t1

    truth = facts.get("materials") or {}
    rows, summary = agreement(truth, result) if truth else ([], {})
    save_json(out / "result.json", {**result.to_dict(), "car": car, "references": refs_urls, "model": vlm.model,
                                    "sheet": {"wall_s": round(sheet_wall, 2), **{k: sheet_res.get(k) for k in (
                                        "total_s", "pass_s", "materials", "sheets", "skipped", "keys")}},
                                    "audit_s": round(audit_s, 2), "agreement": rows, "agreement_summary": summary})
    LAB.mkdir(parents=True, exist_ok=True)
    with open(LAB / "spend.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.time(), "step": "material_audit_lab", "model": vlm.model,
                             **{k: v for k, v in result.usage.items() if k != "per_call"}}) + "\n")

    print(f"livery seen: {result.livery}")
    if result.problems:
        print("problems:", "; ".join(result.problems))
    if rows:
        print(f"\n{'material':<14}{'area%':>6}  {'truth':<24}{'audit':<24}{'dE':>5} {'conf':>5}  verdict  part")
        for r in sorted(rows, key=lambda r: -(r["area_pct"] or 0)):
            t = r["true"] + (f" {r['true_base']}" if r["true_base"] else "")
            p = (r["pred"] or "-") + (f" {r['pred_base']}" if r["pred_base"] else "")
            print(f"{r['name']:<14}{(r['area_pct'] or 0):>6.2f}  {t:<24}{p:<24}{(r['dE'] if r['dE'] is not None else ''):>5}"
                  f" {(r['conf'] if r['conf'] is not None else ''):>5}  {r['verdict']:<8} {r['part'][:50]}")
        print(f"\nagreement: {summary['agree']}/{summary['scored']} same family "
              f"({summary['exact']} exact preset), {summary['agree_area_pct']}% of the scored visible area; "
              f"not audited: {summary['not_audited']}")
    u = result.usage
    print(f"usage: {u.get('calls')} call(s), {u.get('input_tokens')} in / {u.get('output_tokens')} out tokens, "
          f"${u.get('cost_usd', 0):.4f}; audit {audit_s:.1f} s -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcp", default="", help="Blender MCP URL (default BLENDER_MCP_URL)")
    p.add_argument("--ref", action="append", default=[], help="reference photo URL (repeatable)")
    p.add_argument("--car", default="", help="short car description for the prompt (default: facts name)")
    p.add_argument("--facts", default=str(ROOT / "assets" / "product_facts.json"))
    p.add_argument("--views", default="front34,rear34,side")
    p.add_argument("--max-materials", type=int, default=48, help="4 sheets; small lamp parts rank low")
    p.add_argument("--cell-px", type=int, default=320)
    p.add_argument("--max-per-call", type=int, default=ma.MAX_PER_CALL)
    p.add_argument("--load-car", default="", help="S3 car key to load first (raw import)")
    p.add_argument("--length-m", type=float, default=0.0)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
