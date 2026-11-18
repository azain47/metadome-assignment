"""Liveries: group recolouring over fixed materials, and researched colours -> livery specs."""

import pytest

from app.livery import material_assignments
from app.livery_research import ResearchedColour, colour_to_livery

FACTS = {
    "materials": {"glass": {"preset": "glass_car"}, "body_a": {"preset": "keep"}},
    "material_groups": {"body": ["body_a", "body_b"], "calipers": ["cal"]},
    "livery_default": "orange",
    "liveries": {
        "orange": {"groups": {"body": {"preset": "metallic_paint", "base": "#FF6A13"},
                              "calipers": {"preset": "car_paint", "base": "#0A0A0A"}}},
        "grey": {"groups": {"body": {"preset": "metallic_paint", "base": "#6B6E72"}}},
    },
}


def test_default_livery_recolours_every_group_member_and_keeps_fixed_parts():
    out = material_assignments(FACTS)
    assert out["body_a"]["base"] == out["body_b"]["base"] == "#FF6A13"
    assert out["cal"]["base"] == "#0A0A0A"
    assert out["glass"] == {"preset": "glass_car"}


def test_named_livery_and_unknown_livery():
    assert material_assignments(FACTS, "grey")["body_b"]["base"] == "#6B6E72"
    with pytest.raises(ValueError):
        material_assignments(FACTS, "purple")


def test_researched_colour_maps_finish_and_bare_carbon_accents():
    name, spec = colour_to_livery(ResearchedColour(
        name="Volcano Yellow", finish="metallic", hex="f5c400", accent_hex="#121316", caliper_hex="nope",
        source_url="https://example.com/car", evidence="press photo"))
    assert name == "volcano_yellow"
    assert spec["groups"]["body"]["preset"] == "metallic_paint" and spec["groups"]["body"]["base"] == "#F5C400"
    assert spec["groups"]["accents"] == {"preset": "carbon_fibre"}
    assert spec["groups"]["calipers"]["base"] == "#0A0A0A"  # invalid hex falls back
