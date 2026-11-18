"""Material audit (app.material_audit): schema, colour normalisation, downscaling, assignments.
Offline: the vision model is a fake that records what it was sent."""

import ast
import typing
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app import material_audit as ma
from app.imaging.io import decode_rgb, encode_jpeg, encode_png

SET_MATERIALS_TOOLCODE = Path(__file__).resolve().parents[2] / "mcp/blmcp/tools/ts_set_materials_toolcode.py"
CELL_W, CELL_H, BAND = 320, 213, 36
ORANGE = (255, 92, 0)


def toolcode_presets() -> set[str]:
    tree = ast.parse(SET_MATERIALS_TOOLCODE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PRESETS" for t in node.targets):
            return {k.value for k in node.value.keys}
    raise AssertionError("PRESETS not found")


VIEWS = ("front34", "rear34", "side", "closeup")
CLOSEUP = (30, 60, 200)


def make_sheet(n_rows: int) -> tuple[bytes, list[dict]]:
    """Synthetic render_material_sheet output: per row three grey view cells with an orange blob
    (row r: blob at x=40+20r) and a blue close-up cell."""
    w = len(VIEWS) * CELL_W + (len(VIEWS) - 1) * 3
    row_h = BAND + CELL_H
    sheet = np.full((n_rows * row_h + (n_rows - 1) * 6, w, 3), 64, np.uint8)
    materials = []
    for r in range(n_rows):
        y = r * (row_h + 6)
        cells = []
        for v, view in enumerate(VIEWS):
            x = v * (CELL_W + 3)
            if view == "closeup":
                sheet[y + BAND:y + row_h, x:x + CELL_W] = CLOSEUP
            else:
                sheet[y + BAND:y + row_h, x:x + CELL_W] = 160
                bx, by = x + 40 + 20 * r, y + BAND + 100
                sheet[by:by + 8, bx:bx + 12] = ORANGE
            cells.append({"view": view, "x": x, "y": y + BAND, "w": CELL_W, "h": CELL_H, "pixels": 96})
        materials.append({"index": r, "name": f"mat.{r:03d}", "area_pct": 5.0 - r, "sheet": 0,
                          "base_hex": "#4C00B3", "metallic": 0.0, "roughness": 0.97, "alpha": 1.0,
                          "transmission": 0.0, "emission_strength": 0.0, "texture": False, "see_through": False,
                          "cells": cells})
    return encode_png(sheet), materials


class FakeVlm:
    def __init__(self, answers: list[list[dict]]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, list[tuple[str, bytes]]]] = []

    async def parse(self, prompt, images, schema, *, label=""):
        self.calls.append((prompt, images))
        return schema.model_validate({"livery": "white with blue stripes", "materials": self.answers.pop(0)})


def answer(row: int, preset: str, base=None, emission=None, confidence=0.8) -> dict:
    return {"row": row, "part": f"part {row}", "seen_in_reference": "seen", "preset": preset, "base_hex": base,
            "emission_hex": emission, "confidence": confidence}


def test_presets_match_set_materials_tool():
    assert set(ma.PRESETS) == toolcode_presets()
    assert set(typing.get_args(ma.Preset)) == set(ma.PRESETS)
    assert ma.COLOUR_PRESETS <= set(ma.PRESETS)


def test_unknown_preset_rejected_by_schema():
    with pytest.raises(ValidationError):
        ma.AuditAnswer.model_validate({"livery": "", "materials": [answer(0, "chrome_paint")]})


@pytest.mark.parametrize(("raw", "want"), [("#abc", "#AABBCC"), ("aabbcc", "#AABBCC"), (" #19c3F0 ", "#19C3F0"),
                                           ("#F2F2F0", "#F2F2F0")])
def test_hex_normalised(raw, want):
    assert ma.normalise_hex(raw) == want


@pytest.mark.parametrize("raw", ["white", "#12345", "#GGGGGG", "", "rgb(1,2,3)"])
def test_bad_hex_rejected(raw):
    with pytest.raises(ValueError):
        ma.normalise_hex(raw)


async def test_audit_assignments_shape_and_validation():
    sheet, materials = make_sheet(6)
    fake = FakeVlm([[
        answer(0, "car_paint", "#f2f2f0"),
        answer(1, "rubber_tyre", "#111111"),  # colour ignored: not a colour preset
        answer(2, "keep"),
        answer(3, "lamp_emissive", emission="ff1a0a"),
        answer(4, "car_paint", "bright blue"),  # invalid colour: preset kept, colour dropped
        answer(4, "glass_dark"),  # duplicate row: first answer wins
        answer(99, "satin_trim"),  # unknown row
        # row 5 unanswered
    ]])
    result = await ma.audit_materials([sheet], materials, [encode_jpeg(np.zeros((100, 150, 3), np.uint8))],
                                      "test car", vlm=fake)
    assert result.assignments == {
        "mat.000": {"preset": "car_paint", "base": "#F2F2F0"},
        "mat.001": {"preset": "rubber_tyre"},
        "mat.003": {"preset": "lamp_emissive", "emission": "#FF1A0A"},
        "mat.004": {"preset": "car_paint"},
    }
    assert [v.name for v in result.materials] == ["mat.000", "mat.001", "mat.002", "mat.003", "mat.004"]
    assert result.livery == "white with blue stripes"
    assert any("mat.004" in p for p in result.problems)
    assert any("unknown row 99" in p for p in result.problems)
    assert any("mat.005" in p and "no answer" in p for p in result.problems)
    assert all(set(spec) <= {"preset", "base", "emission"} for spec in result.assignments.values())


async def test_inputs_downscaled_and_batched():
    sheet, materials = make_sheet(5)
    big_ref = encode_jpeg(np.full((2400, 3600, 3), 200, np.uint8))
    fake = FakeVlm([[answer(i, "satin_trim") for i in (0, 1, 2)], [answer(i, "satin_trim") for i in (3, 4)]])
    result = await ma.audit_materials([sheet], materials, [big_ref, big_ref], vlm=fake, max_per_call=3)
    assert len(fake.calls) == 2  # 5 materials, at most 3 per call -> 2 balanced calls
    for _prompt, images in fake.calls:
        refs = [b for c, b in images if c.startswith("Reference photo")]
        cards = [(c, b) for c, b in images if c.startswith("Card row")]
        assert len(refs) == 2
        for b in refs + [b for _c, b in cards]:
            h, w = decode_rgb(b).shape[:2]
            assert max(h, w) <= ma.MAX_SIDE
    captions = [c for _p, imgs in fake.calls for c, _b in imgs if c.startswith("Card row")]
    assert captions[0].startswith("Card row #00 (material 'mat.000'")
    assert "#4C00B3" in captions[0]
    assert len(result.assignments) == 5


def test_material_card_layout():
    sheet, materials = make_sheet(2)
    card = ma.material_card(decode_rgb(sheet), materials[1])
    assert card.shape == (2 * CELL_H + ma.CARD_GAP, 2 * CELL_W + ma.CARD_GAP, 3)
    first = card[:CELL_H, :CELL_W].astype(int)
    # row 1's blob (x 60..72, y 100..108 in the cell), not row 0's, and no label band
    assert (np.abs(first[104, 66] - ORANGE) < 12).all()
    assert (np.abs(first[104, 44] - 160) < 12).all()
    closeup = card[CELL_H + ma.CARD_GAP:, CELL_W + ma.CARD_GAP:].astype(int)
    assert (np.abs(closeup.mean(axis=(0, 1)) - CLOSEUP) < 4).all()


def test_view_words_cover_sheet_views():
    tool = SET_MATERIALS_TOOLCODE.with_name("ts_render_material_sheet_toolcode.py")
    tree = ast.parse(tool.read_text(encoding="utf-8"))
    views = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", "") == "VIEWS" for t in node.targets))
    assert {k.value for k in views.keys} | {"closeup"} == set(ma.VIEW_WORDS)


async def test_fetch_reference_downscales():
    src = encode_jpeg(np.full((1500, 2000, 3), 128, np.uint8))

    async def fetch(url: str) -> bytes:
        assert url == "https://example.org/car.jpg"
        return src

    out = await ma.fetch_reference("https://example.org/car.jpg", fetch_bytes=fetch)
    assert decode_rgb(out).shape[:2] == (576, 768)
    small = await ma.fetch_reference("https://example.org/car.jpg", max_side=4000, fetch_bytes=fetch)
    assert decode_rgb(small).shape[:2] == (1500, 2000)  # never upscaled
