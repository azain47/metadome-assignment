"""Offline end-to-end run of the real graph: real Blender (through the MCP server), real S3 and
Postgres, the `fake` image backend, and a scripted stand-in for the LLM nodes.

It proves the wiring (plan -> setup_scene -> guides -> plates -> render_car -> composite ->
checks -> approval pause and resume -> finalize and export) without spending money. Not part
of the pytest suite: it needs the dev stack and a Blender instance.

    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres s3 mcp-blender
    scripts/dev/blender_instance.sh start e2e 9876
    cd agent && PYTHONPATH=. uv run python tests/e2e_offline.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, AsyncGenerator

os.environ.setdefault("IMAGE_BACKEND", "fake")
os.environ.setdefault("DATABASE_URL", "postgresql://trueshot:trueshot-dev-pw@localhost:5433/trueshot")
os.environ.setdefault("S3_SECRET_KEY", "trueshot-dev-secret")
os.environ.setdefault("BLENDER_MCP_URL", "http://127.0.0.1:8765/")
os.environ.setdefault("OUTPUTS_DIR", "/tmp/trueshot_outputs")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents"))

from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import DatabaseSessionService  # noqa: E402
from google.genai import types  # noqa: E402

PLAN = {
    "plan": {
        "location_key": "alps", "location_label": "Swiss Alps hairpin",
        "concept": "low rolling shot out of the first hairpin, peaks behind",
        "set": {"road": [{"straight_m": 40}, {"turn_deg": 180, "radius_m": 14}, {"straight_m": 80}],
                "elements": [{"kind": "hillside", "side": "near", "grade": 0.3, "height_deg": [3, 5]},
                             {"kind": "mountains", "count": 8, "distance_m": [700, 1600], "height_deg": [1, 2.4]}],
                "seed": 3},
        "placement": {"distance_m": 10, "lane": "near"},
        "camera": {"azimuth_deg": 32, "elevation_deg": 2.5, "focal_mm": 40, "car_width_frac": 0.62,
                   "frame_x": 0.47, "frame_y": 0.64, "target_height_m": 0.55, "fstop": 5.6},
        "motion": {"speed_kmh": 60, "shutter_s": 0.01},
        "light": {"time_of_day": "late morning, bright sun, clear blue sky",
                  "hdri_query": "alpine mountain meadow road midday clear", "sun_rel_deg": 135.0},
        "scene_prompt": ("A mountain pass road high in the Swiss Alps: an empty, deserted road of fresh dark "
                         "asphalt with white edge lines, alpine meadow and pine forest, snowy peaks behind."),
        "reference_keys": [],
    },
    "start_stage": "SETUP", "memory_used": [],
}

# Which LLM node is calling, recognised by the first words of its static instruction.
_ROLES = {"You are the intake step": "intake", "You are the photographer and set designer": "plan",
          "You fix a failed product shot": "repair"}


def _text(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


class Scripted(BaseLlm):
    """Replies by node with fixed structured outputs."""

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        system = str(llm_request.config.system_instruction or "")
        role = next((r for prefix, r in _ROLES.items() if prefix in system), "answer")
        if role == "intake":
            yield _text(json.dumps({"kind": "new_shots", "locations": ["Swiss Alps hairpin at golden hour"],
                                    "shot_ref": "", "request": "A golden-hour shot on an Alpine hairpin"}))
        elif role == "plan":
            yield _text(json.dumps(PLAN))
        elif role == "repair":
            yield _text(json.dumps({"stage": "PLATES", "reason": "scripted repair", "plan": PLAN["plan"]}))
        else:
            yield _text("ok")


# Swap the model factory before the graph modules build their agents.
import trueshot.llm  # noqa: E402

trueshot.llm.make_model = lambda: Scripted(model="scripted")

from app.config import settings  # noqa: E402
from app.services import services  # noqa: E402
from app.storage import seed_assets  # noqa: E402
from trueshot.agent import app  # noqa: E402


def _summarise(event: Any) -> str | None:
    d = event.model_dump(mode="json", by_alias=True, exclude_none=True)
    step = (d.get("customMetadata") or {}).get("ts_step")
    if step and step["status"] != "running":
        return (f"  step  {step['node']:12s} {step['name']:18s} {step['status']:5s} "
                f"{step.get('duration_ms', 0):6d} ms  {step.get('summary', '')[:90]}")
    for part in (d.get("content") or {}).get("parts", []):
        if "functionCall" in part:
            fc = part["functionCall"]
            return f"  call  {d.get('author', ''):12s} {fc['name']}({json.dumps(fc.get('args', {}))[:90]})"
        if "functionResponse" in part:
            resp = part["functionResponse"].get("response", {})
            status = (resp.get("structuredContent") or resp).get("status", "")
            return f"  resp  {d.get('author', ''):12s} {part['functionResponse']['name']} -> {status}"
        if "text" in part and d.get("author") != "user":
            return f"  msg   {d.get('author', ''):12s} {part['text'][:150]}"
    return None


async def _drain(runner: Runner, session_id: str, message: types.Content) -> dict[str, Any] | None:
    pending = None
    async for event in runner.run_async(user_id="demo", session_id=session_id, new_message=message):
        if line := _summarise(event):
            print(line, flush=True)
        for part in (event.content.parts if event.content else []) or []:
            if part.function_call and part.function_call.name == "adk_request_input":
                pending = {"id": part.function_call.id, "args": part.function_call.args}
    return pending


def _answer(interrupt_id: str, decision: str, note: str = "") -> types.Content:
    return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=interrupt_id, name="adk_request_input", response={"decision": decision, "note": note}))])


async def _drain_to_final(runner: Runner, session_id: str, message: types.Content) -> dict[str, Any] | None:
    """Drain, approving every guide-review pause, up to the next final-still approval."""
    pending = await _drain(runner, session_id, message)
    while pending and str(pending["id"]).startswith("guide_"):
        print(f"--- guide review {pending['id']}: approve", flush=True)
        pending = await _drain(runner, session_id, _answer(pending["id"], "approve"))
    return pending


async def main() -> None:
    svc = services()
    await svc.db.start()
    await seed_assets(svc.store, settings.assets_dir)
    sessions = DatabaseSessionService(db_url=settings.session_service_uri)
    runner = Runner(app=app, session_service=sessions)
    session = await sessions.create_session(app_name="trueshot", user_id="demo")
    t0 = time.monotonic()
    pending = await _drain_to_final(runner, session.id, types.Content(role="user", parts=[types.Part(
        text="Make a golden hour shot on an Alpine hairpin")]))
    print(f"--- paused for approval after {time.monotonic() - t0:.1f}s: {json.dumps(pending, default=str)[:300]}")
    assert pending, "expected an approval request"

    # Simulate a restart while the run waits for approval: fresh session service and runner.
    sessions = DatabaseSessionService(db_url=settings.session_service_uri)
    runner = Runner(app=app, session_service=sessions)
    print("--- restarted runner; answering 'revise'")
    pending = await _drain_to_final(runner, session.id, _answer(pending["id"], "revise", "warmer light, more road visible"))
    print(f"--- paused again: {json.dumps(pending, default=str)[:200]}")
    assert pending, "expected a second approval request after the revision"
    await _drain(runner, session.id, _answer(pending["id"], "approve"))
    final = await sessions.get_session(app_name="trueshot", user_id="demo", session_id=session.id)
    st = final.state
    print("--- run:", st.get("run"))
    print("--- spend:", st.get("spend"))
    print("--- gallery kinds:", [g["kind"] for g in st.get("gallery", [])])
    print("--- approved memory:", list((st.get("user:approved_shots") or {}).keys()))
    print("--- shots:", st.get("shots"))
    shot = st.get("shot") or {}
    print("--- scene:", json.dumps(shot.get("scene") or {}, default=str)[:900])
    print("--- scene_ops:", [op["tool"] for op in shot.get("scene_ops", [])])
    print("--- keys: plate", shot.get("plate_key"), "car", (shot.get("car") or {}).get("keys"),
          "final", (shot.get("best") or {}).get("key"))
    print(f"--- total {time.monotonic() - t0:.1f}s; events {len(final.events)}; "
          f"max event bytes {max(len(e.model_dump_json()) for e in final.events)}")
    await svc.db.stop()


if __name__ == "__main__":
    asyncio.run(main())
