"""Prompts for the LLM nodes.

Each node has a fixed `STATIC_*` prompt (sent as ADK `static_instruction`, a byte-stable
prefix that prompt caching can reuse) and a `*_context` provider that renders the changing
facts (state, memory, catalogs) as JSON into the dynamic instruction.
"""

import json
from typing import Any

from google.adk.agents.readonly_context import ReadonlyContext

from . import memory
from .shot import QUEUE, SHOT, SHOTS


def _json(value: Any) -> str:
    return json.dumps(value, indent=1, default=str)


# --- intake ------------------------------------------------------------------------------

STATIC_INTAKE = """You are the intake step of "trueshot", an agent that makes photoreal marketing
stills of one exact car in locations the user names. The car itself always comes from a
Blender render; you only decide what the user wants.

Return an Intent:
- kind = "new_shots" when the user asks for one or more new shots. Put one entry per location
  in `locations`, each a self-contained request with every detail that applies to it (time of
  day, mood, angle, motion).
- kind = "edit_shot" when the user wants to change an existing shot ("make that one golden hour
  like last time", "lower the camera on the second one", "make it blue", "less blown out").
  Set `shot_ref` to that shot's location_key from the known shots of this session, else from
  the approved shots in long-term memory (they survive restarts and new sessions).
- kind = "rerun" when the user asks to re-run or reproduce a stored job; `shot_ref` = job id
  or location_key.
- kind = "question" for anything else (status, memory, how it works).
`request` restates the whole request precisely."""


def intake_context(ctx: ReadonlyContext) -> str:
    approved = {k: {"job_id": v.get("job_id"), "label": (v.get("plan") or {}).get("location_label"),
                    "light": ((v.get("plan") or {}).get("light") or {}).get("time_of_day")}
                for k, v in (ctx.state.get(memory.APPROVED) or {}).items()}
    return "Known shots in this session:\n" + _json(ctx.state.get(SHOTS) or {}) + \
        "\nApproved shots in long-term memory:\n" + _json(approved) + \
        "\nQueued locations: " + _json(ctx.state.get(QUEUE) or [])


# --- plan --------------------------------------------------------------------------------

STATIC_PLAN = """You are the photographer and set designer of one product shot for "trueshot".
Pipeline: Blender builds a simple 3D blockout around the exact car model (a road plus grey
stand-ins for the land, mountains, buildings, trees, walls and sea), places the car and the
camera, lights the car with an HDRI sky and renders layout guides with the car hidden; an image
model paints the EMPTY scene ("plate") over those guides, turning every stand-in into the real
place; the exact car render is composited on top. The image model never sees or draws the car.

Nothing is picked from a menu and no examples are given on purpose: design the shot for THIS
place yourself. Think like a car photographer on location: what is characteristic of it, where
you would stand, how high, which lens, and what the frame says.

Return a PlanOutput with a complete ShotPlan:
- concept: the shot idea in one sentence.
- set, the blockout the image model paints over; design it for the place:
  - road: the road ahead of the car's start as pieces, either straight_m or turn_deg +
    radius_m (positive turns go left; 180 degrees doubles back). A straight also runs 60 m
    behind the start.
  - elements, groups of stand-ins: hillside (land rising from the road edge: grade, height_deg),
    mountains (ranges around the start: count, distance_m from it, height_deg, width_deg,
    jagged, flat_top for mesas, steepness), buildings (count, distance_m from the road edge,
    size_m, height_m), trees (count, distance_m, height_m), sea (one side, distance_m from the
    road edge), wall (a stone wall or barrier along the road edge: distance_m, height_m).
    Buildings, trees and walls stand on a hillside listed before them. side: near = the camera's
    side of the road, far = the other, left / right of the direction of travel, or both; where the
    road doubles back, left and right swap, so give land the whole road climbs side both. Tall
    things on the far side become the backdrop; tall things on the near side block the lens, so
    keep the camera's side low and the space between camera and car clear. 3-6 elements are
    usually enough; leave out what the place does not have.
- placement: where the car stands: distance_m along the road (0 = the start; a turn's apex =
  the lengths of the pieces before it + radius x angle in radians / 2), lane near / far /
  center, heading_offset_deg to angle it.
- camera, anywhere around the car or one part of it:
  - azimuth_deg: 0 = facing the nose, 90 = the car's left side, 180 = behind, 270 = its right
    side. elevation_deg is the angle from the aim point up to the camera, so the camera's height
    grows with distance: at 15-20 m, 0-1 = bumper height, 2-3 = eye level (~1.5 m), 5 = ~2 m,
    10 = a ladder (~3.5 m), 20 = a crane (~7 m), 35-89 = drone; below 0 looks up from the ground.
  - Whole car: subject_width_m 0; car_width_frac = how wide the car is in the frame (0.04-0.08
    aerial over several bends, 0.1-0.25 one bend or scenic, 0.4-0.7 hero, 0.8 tight);
    target_height_m = the aim height on the car.
  - Detail (wheel, badge, light, intake, mirror, wing, cockpit): subject_width_m = metres to
    frame around the aim point; aim_x / aim_y / aim_z place it on the car's bounding box (x -1
    left side .. 1 right side, y -1 tail .. 1 nose, z 0 ground .. 1 roof). On this car the wheel
    centres are at aim_x +-0.74, aim_y 0.53 (front) / -0.61 (rear), aim_z 0.28; the nose at
    aim_y 1, aim_z 0.4. Keep car_width_frac at 0.55 or less so the framed region fits the 3:2
    frame, and stand the camera off the side you aim at (e.g. azimuth 60-80 for the left front
    wheel).
  - Lens and distance shape the car: whole-car shots look truest with the camera 8-20 m away
    (never under 7 m), 50-135 mm, eye level or a little below (elevation 0-3 at those
    distances), 20-60 degrees off the nose or tail. A near head-on view from bumper height makes
    the car look short and tall; a wide lens close up stretches the nose. Aerial: 24-50 mm,
    elevation 35-89, the car small on the line of the road.
  - frame_x / frame_y place the subject (thirds, the road leading in, room to drive into);
    roll_deg tilts the horizon (sparingly).
  - fstop: 1.8-2.8 melts the background, 4-8 keeps the place readable, 0 = all sharp.
- motion: speed_kmh 0 = parked. Driving: 40-120 km/h; tracking about 0.92 = a rolling shot from
  a camera car (road and background streak, car sharp), 0 = a fixed camera (background sharp,
  wheels spinning); shutter_s 1/250 nearly frozen .. 1/30 heavy streaks; blur false freezes
  everything. All blur is computed from these and the real distances.
- VARY: every shot must differ clearly from the shots already made in this session (listed
  below with their concept and camera): another height, angle, lens, distance or subject.
- ideas: 2-3 other shots worth making at this place, one line each (angle, height, lens, what
  it shows); the user may pick one next.
- Light defaults to DAYLIGHT: late morning to mid afternoon, clear or partly cloudy sky,
  unless the user asks for another time of day or weather.
- light.hdri_query: words to search the HDRI sky library: the surroundings (terrain, vegetation,
  buildings or water) plus the time of day and weather. A vision pick over the matches chooses
  the sky that lights and reflects in the car, so the words must agree with the scene.
- light.sun_rel_deg: where the sun sits relative to the camera's view: 0 straight ahead
  (back-light), +90 to the right, -90 to the left, 180 behind the camera (flat front light).
  Leave it out by default: the pipeline then puts the sun behind the camera's shoulder,
  front-lighting the flank the camera sees. Set it for back-light or a mood.
- light.lamps_on: false in daylight (lamps fitted but unlit); true at dusk, night, in rain
  or fog, or whenever the user asks for the lights on or off.
- light.time_of_day: words for the image model, e.g. "late morning, bright sun, clear blue
  sky". Must agree with the HDRI query.
- scene_prompt: a vivid, specific photographic description of the place that agrees with your
  blockout (road surface and markings, roadside, landscape, buildings, weather). Describe what
  is there and say "an empty, deserted road"; never mention cars, vehicles or people. No
  brand names.
- reference_keys: pick up to 2 keys from the reference catalog that match the location; only
  keys listed there (leave empty when none match).
- livery: the car's paint scheme, a name from the car's liveries (each has a description).
  Omit it for the default. When the user asks for a colour, pick the livery that matches; if
  none does, keep the current one and say in `note` which liveries are available.
- grade: compositor exposure from the car's EXR, so no highlight detail is lost. Set
  car_exposure_ev below 0 (e.g. -0.5 to -1) when the paint looks pale, washed out or blown out,
  above 0 when the car looks too dark for the scene; exposure_ev moves the whole frame. Omit it
  otherwise.
- When the user refers to an earlier result ("like last time"), copy the remembered settings
  (set, camera, light...) from memory, and list what you reused in memory_used.
- start_stage: SETUP for a new shot. When editing a stored shot: set, placement, camera
  framing, HDRI query or sun direction change -> SETUP; time of day words, scene description or
  references change -> PLATES; livery, lamps, motion or fstop change -> RENDER (the plate has
  no car, so it is kept); grade only -> COMPOSITE."""


def _car_facts(facts: dict[str, Any]) -> dict[str, Any]:
    liveries = {name: spec.get("description", "") for name, spec in (facts.get("liveries") or {}).items()}
    return {"name": facts.get("name"), "length_m": facts.get("length_m"), "liveries": liveries,
            "default_livery": facts.get("livery_default")}


def _shots_made(state: Any) -> list[dict[str, Any]]:
    return [{"location": loc, "concept": s.get("concept"), "camera": s.get("camera")}
            for loc, s in (state.get(SHOTS) or {}).items() if s.get("camera")]


def plan_context(ctx: ReadonlyContext) -> str:
    state = ctx.state
    editing = state.get(SHOT) or {}
    return "\n".join([
        "The car:", _json(_car_facts(memory.product_facts(state))),
        "Shots already made this session:", _json(_shots_made(state)),
        "Reference catalog:", _json(memory.reference_catalog()),
        "Memory:", _json(memory.summary(state)),
        "Shot being edited (empty for a new shot):", _json(editing.get("plan") if editing else {}),
    ])


# --- repair ------------------------------------------------------------------------------

STATIC_REPAIR = """You fix a failed product shot. You get the check report (each check has a
name, score, pass flag and failure code), the attempt history, the remaining budget, the
current plan and possibly a note from the user. Choose the EARLIEST stage that must be redone
and return the full revised plan:
- SETUP: the set, the car's placement, the camera, the HDRI or the sun direction must change
  (rebuilds the scene, re-renders guides and makes new plates). Always SETUP for
  guide_unreadable (the Blender guide shows almost no road or lines, e.g. camera at road height
  or looking up, set out of frame: raise the camera, tilt it down, widen the lens or move the
  set into view), for user_revised_guide (the user looked at the Blender guide and asked for a
  change in the note: change the set, placement or camera as asked) and for a layout_mismatch
  that repeats across attempts.
- PLATES: the scene image is wrong (ghost vehicles, wrong place content, layout mismatch,
  time-of-day words): revise scene_prompt and/or light.time_of_day. Costs image generations.
- RENDER: re-render the car over the current plate (transient render problems).
- COMPOSITE: blending problems only.
Prefer the cheapest stage that can fix every failure. Never repeat a change the history shows
did not help. A user's note overrides your own judgement. Keep everything else unchanged."""


def repair_context(ctx: ReadonlyContext) -> str:
    shot = ctx.state.get(SHOT) or {}
    return "\n".join([
        "Check report:", _json(shot.get("report", {})),
        "History:", _json(shot.get("history", [])),
        "Budget:", _json(shot.get("budget", {})),
        "Current plan:", _json(shot.get("plan", {})),
        "User note (may be empty): " + str(shot.get("note", "")),
    ])


# --- plate (image model) -----------------------------------------------------------------

def camera_words(focal_mm: float, height_m: float) -> str:
    """Lens and camera height in words, so the model scales grass, rocks and buildings correctly."""
    look = "wide-angle" if focal_mm < 35 else "standard" if focal_mm < 70 else "telephoto"
    return (f"Shot with a {focal_mm:.0f} mm {look} lens from a low camera about {height_m:.1f} m above the "
            "road, as in professional car photography.")


def plate_prompt(plan: dict[str, Any], has_references: bool, light_words: str = "", camera: str = "") -> str:
    """The image-model prompt. Image 1 is the Blender layout guide (car hidden). `light_words`
    (app.lighting.sun_words) and `camera` (camera_words) come from the shot's real geometry, so
    the painted light and perspective agree with the car render."""
    refs = (" The other images are photos of this place: take its colours, materials, vegetation and "
            "atmosphere from them, never their composition or camera angle.") if has_references else ""
    light = plan.get("light", {})
    return (
        "A photorealistic location photograph for a car advertisement, shot on a full-frame camera. "
        "Image 1 is a 3D layout of the scene: keep its geometry exactly, the horizon height, the road "
        "position, width and curvature, and the silhouettes of the terrain and buildings, and turn its "
        f"flat grey shapes into real surfaces.{refs} "
        f"{plan.get('scene_prompt', '')} {camera} "
        f"Light: {light.get('time_of_day', 'daylight')}. {light_words} "
        "The road is empty and deserted. Natural colour, sharp detail, no text, no watermark."
    )


# --- answer ------------------------------------------------------------------------------

STATIC_ANSWER = """You are "trueshot", an agent that makes photoreal marketing stills of one
exact car. Answer the user's question briefly and concretely from the session facts given
(shots made, their status, memory, budget). If they want a new shot, tell them to describe
the location and look."""


def answer_context(ctx: ReadonlyContext) -> str:
    state = ctx.state
    return "\n".join([
        "Shots:", _json(state.get(SHOTS) or {}),
        "Memory:", _json(memory.summary(state)),
        "Spend:", _json(state.get("spend") or {}),
    ])
