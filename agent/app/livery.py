"""Car materials from product facts: fixed per-part materials plus the active livery.

Product facts hold `materials` (material name -> set_materials spec for parts every livery
shares: glass, tyres, carbon, lamps), `material_groups` (group -> material names, e.g. body,
accents, calipers) and `liveries` (name -> {description, groups: group -> spec}). A livery
recolours whole groups, so a new colour scheme is data, never code.
"""

from __future__ import annotations

from typing import Any


def livery_name(facts: dict[str, Any], livery: str | None = None) -> str | None:
    return livery or facts.get("livery_default") or next(iter(facts.get("liveries", {})), None)


# Presets that emit light: switched off (strength 0) unless the shot wants the lamps on.
LAMP_EMITTERS = frozenset({"lamp_led", "lamp_emissive"})


def material_assignments(facts: dict[str, Any], livery: str | None = None, *,
                         lamps_on: bool = True) -> dict[str, dict]:
    """`set_materials` assignments for the car in `livery` (default: facts' livery_default).
    `lamps_on=False` keeps the lamp materials but stops them emitting (daylight shots)."""
    out = _livery_assignments(facts, livery)
    if not lamps_on:
        out = {k: {**v, "emission_strength": 0.0} if v.get("preset") in LAMP_EMITTERS else v for k, v in out.items()}
    return out


def _livery_assignments(facts: dict[str, Any], livery: str | None) -> dict[str, dict]:
    out = dict(facts.get("materials", {}))
    name = livery_name(facts, livery)
    if name is None:
        return out
    liveries = facts.get("liveries", {})
    if name not in liveries:
        raise ValueError(f"unknown livery {name!r}; liveries: {sorted(liveries)}")
    groups = facts.get("material_groups", {})
    for group, spec in liveries[name].get("groups", {}).items():
        for material in groups.get(group, []):
            out[material] = dict(spec)
    return out
