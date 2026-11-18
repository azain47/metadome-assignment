"""Long-term memory, in ADK user/app state (persisted in Postgres by the session service).

- `app:product_facts`  facts about the real car (paint, wheel spoke count, lamp signature...),
                       loaded from assets/product_facts.json; shared by every user and session.
- `user:approved_shots` {location_key: approved settings}: written when a shot is approved,
                       reused by the planner ("golden hour like last time").
- `user:looks`          {time_of_day label: light settings}: reusable looks across locations.

Recall is an exact lookup by location/look name: the planner sees these maps in its prompt.
"""

import json
import os
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Mapping, MutableMapping

from app.config import settings

from .shot import ShotState

FACTS = "app:product_facts"
APPROVED = "user:approved_shots"
LOOKS = "user:looks"


@lru_cache(maxsize=1)
def _facts_file() -> dict[str, Any]:
    path = os.path.join(settings.assets_dir, "product_facts.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ensure_product_facts(state: MutableMapping[str, Any]) -> None:
    """Copy the product facts file into app state when it is new or changed."""
    facts = _facts_file()
    if facts and state.get(FACTS) != facts:
        state[FACTS] = facts


def product_facts(state: Mapping[str, Any]) -> dict[str, Any]:
    return dict(state.get(FACTS) or {})


def remember_approval(state: MutableMapping[str, Any], shot: ShotState, final_key: str) -> None:
    plan = shot.plan
    approved = dict(state.get(APPROVED) or {})
    approved[plan.location_key] = {
        "job_id": shot.job_id,
        "final_key": final_key,
        "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": plan.model_dump(mode="json"),
    }
    state[APPROVED] = approved
    looks = dict(state.get(LOOKS) or {})
    looks[plan.light.time_of_day.lower()] = {
        "light": plan.light.model_dump(mode="json"), "from_job": shot.job_id, "location": plan.location_key,
    }
    state[LOOKS] = looks


def summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """What the planner gets to see: compact, JSON-serialisable."""
    approved = {
        loc: {"job_id": v.get("job_id"), "approved_at": v.get("approved_at"),
              **{k: v.get("plan", {}).get(k) for k in ("concept", "set", "placement", "camera", "motion", "light")}}
        for loc, v in (state.get(APPROVED) or {}).items()
    }
    return {"approved_shots": approved, "looks": state.get(LOOKS) or {}}


def reference_catalog() -> dict[str, list[str]]:
    """{location folder: [refs/<folder>/<file>, ...]} from the mounted assets."""
    root = os.path.join(settings.assets_dir, "refs")
    catalog: dict[str, list[str]] = {}
    if not os.path.isdir(root):
        return catalog
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
            if files:
                catalog[folder] = [f"refs/{folder}/{f}" for f in files]
    return catalog
