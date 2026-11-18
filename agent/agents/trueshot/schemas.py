"""Typed data passed between graph nodes.

A ShotPlan is the shot as data that maps onto Blender tool calls without translation: `set` ->
build_set, `placement` -> place_car_on_road, `camera` -> frame_shot (plus the lens f-stop),
`motion` -> the exposure, `light.hdri_query` -> find_hdris. Nothing is picked from a fixed menu:
the planner designs the road, the stand-in set, where the car stands and the camera for each
place. LLM nodes emit these as structured output (`output_schema`).
"""

from typing import Literal

from pydantic import BaseModel, Field

# Pipeline stages a (re)try can start from, earliest first. OUT = stop and hand the best
# candidate to approval. REPLAY = rebuild a stored scene from its recorded tool calls.
Stage = Literal["SETUP", "PLATES", "RENDER", "COMPOSITE"]
STAGE_ORDER: tuple[str, ...] = ("SETUP", "PLATES", "RENDER", "COMPOSITE")

Side = Literal["near", "far", "left", "right", "both"]


class RoadSegment(BaseModel):
    """One piece of the road ahead of the car's spot: a straight (straight_m) or a turn
    (turn_deg + radius_m)."""
    straight_m: float | None = Field(None, ge=1, le=1000, description="length of a straight piece")
    turn_deg: float | None = Field(None, ge=-270, le=270, description="a turn: positive = left, negative = right; "
                                   "180 doubles back")
    radius_m: float | None = Field(None, ge=6, le=2000, description="turn radius in metres (6 = tightest)")


class SetElement(BaseModel):
    """A group of grey stand-ins. The image model paints them into the real place, so they set
    where the land rises, where buildings, trees, walls and sea are, and how tall they look."""
    kind: Literal["hillside", "mountains", "buildings", "trees", "sea", "wall"]
    side: Side = Field("both", description="side of the road: near = the camera's side, far = the other; "
                       "left/right seen driving away from the start")
    count: int | None = Field(None, ge=1, le=6000, description="mountains, buildings, trees: how many")
    distance_m: list[float] | None = Field(None, description="[min, max] from the road edge (buildings, trees, "
                                           "sea, wall: one value) or from the car (mountains)")
    height_m: list[float] | None = Field(None, description="[min, max] height (buildings, trees; wall: one value)")
    height_deg: list[float] | None = Field(None, description="[min, max] how tall they look from the car, in "
                                           "degrees above the horizon (mountains, hillside; a cap for buildings "
                                           "and trees). 1-3 = low distant range, 4-8 = big mountains, 10+ = walls of rock")
    size_m: list[float] | None = Field(None, description="[min, max] building footprint")
    width_deg: list[float] | None = Field(None, description="[min, max] how wide each mountain looks, degrees")
    grade: float | None = Field(None, ge=0.05, le=1.5, description="hillside: rise per metre away from the road")
    jagged: float | None = Field(None, ge=0, le=1, description="mountains: 0 smooth, 1 jagged crests")
    flat_top: float | None = Field(None, ge=0, le=0.6, description="mountains: 0 peaks, 0.3-0.6 mesas / plateaus")
    steepness: float | None = Field(None, ge=0.5, le=6, description="mountains: flank width per unit of height "
                                    "(1 = steep flanks, 3-4 = broad, gentle ranges)")


class SetPlan(BaseModel):
    """The 3D blockout around the car: the road and the stand-ins, designed for this place."""
    road: list[RoadSegment] = Field(default_factory=list, description="the road ahead of the car's start, in "
                                    "order (a straight runs 60 m behind the start; after the last piece the "
                                    "road runs straight on). Empty = one long straight")
    road_width_m: float = Field(7.0, ge=3, le=30)
    elements: list[SetElement] = Field(default_factory=list, description="stand-in groups, in order")
    seed: int = Field(0, ge=0, le=99, description="varies the random layout inside each element")


class Placement(BaseModel):
    """Where the car stands."""
    distance_m: float = Field(0.0, ge=-50, le=380, description="along the road from its start: 0 = start of "
                              "the designed road; a turn's apex = the pieces before it + radius x angle(rad) / 2")
    lane: Literal["near", "far", "center"] = Field("near", description="near = the lane on the camera's side")
    heading_offset_deg: float = Field(0.0, ge=-90, le=90, description="turn the car away from the road direction "
                                      "(clockwise seen from above), e.g. parked at an angle")


class CameraPlan(BaseModel):
    """The camera, placed freely around the car (or one part of it)."""
    azimuth_deg: float = Field(ge=0, lt=360, description="where the camera stands: 0 = facing the nose, 90 = the "
                               "car's left side, 180 = behind the tail, 270 = the car's right side")
    elevation_deg: float = Field(ge=-10, le=89, description="angle from the aim point up to the camera; camera "
                                 "height = aim height + distance x tan(angle). At the usual 15-20 m: 0-1 = bumper "
                                 "height (~0.8 m), 2-3 = eye level (~1.5 m), 5 = ~2 m, 10 = ladder (~3.5 m), 20 = "
                                 "crane (~7 m), 35+ = drone; below 0 looks up from the ground")
    focal_mm: float = Field(ge=12, le=400, description="lens; with car_width_frac it sets the distance (wide = "
                            "close, long = far)")
    car_width_frac: float = Field(ge=0.03, le=3.0, description="how much of the frame width the subject spans: car "
                                  "0.1-0.25 for aerial or scenic, 0.4-0.7 hero, 0.8+ tight; above 1 crops into it")
    frame_x: float = Field(0.5, ge=0, le=1, description="subject centre across the frame (0 left, 1 right)")
    frame_y: float = Field(0.6, ge=0, le=1, description="subject centre down the frame (0 top, 1 bottom)")
    roll_deg: float = Field(0.0, ge=-30, le=30, description="Dutch tilt")
    target_height_m: float = Field(0.6, ge=0, le=3, description="whole-car shots: the height on the car the "
                                   "camera aims at")
    aim_x: float = Field(0.0, ge=-1.2, le=1.2, description="detail shots: aim point across the car, -1 left "
                         "side .. 1 right side")
    aim_y: float = Field(0.0, ge=-1.2, le=1.2, description="detail shots: aim point along the car, -1 tail .. 1 nose")
    aim_z: float = Field(0.4, ge=0, le=1.2, description="detail shots: aim point height, 0 ground .. 1 roof")
    subject_width_m: float = Field(0.0, ge=0, le=4, description="0 = the whole car; > 0 = a detail shot that "
                                   "frames this many metres around the aim point (0.3 badge, 0.8 wheel, 2 front end)")
    fstop: float = Field(5.6, ge=0, le=22, description="aperture: 1.8-2.8 soft background, 4-8 sharper, 0 = "
                         "everything sharp")


class MotionPlan(BaseModel):
    """The car's motion during the exposure (one shutter for the car, its wheels and the plate)."""
    speed_kmh: float = Field(0.0, ge=0, le=300, description="0 = parked (sharp wheels); 40-120 = driving")
    shutter_s: float = Field(0.01, gt=0, le=0.25, description="1/250 nearly frozen, 1/100 typical rolling shot, "
                             "1/30 heavy streaks")
    tracking: float = Field(0.92, ge=0, le=1, description="share of the car's travel the camera follows: ~0.92 = "
                            "rolling shot from a camera car (road and background streak, car sharp); 0 = camera "
                            "standing still (background sharp, wheels spin)")
    blur: bool = Field(True, description="false = no motion blur anywhere")


class LightPlan(BaseModel):
    time_of_day: str = Field(description="words for the image model, e.g. 'golden hour, warm low sunlight, clear sky'")
    hdri_query: str = Field(description="words for the HDRI sky search: surroundings (terrain, vegetation, "
                            "buildings or water) plus time of day and weather; it lights and reflects in the car")
    sun_rel_deg: float | None = Field(None, ge=-180, le=180, description="sun relative to the camera's view: "
                                      "0 ahead, +90 right, -90 left, 180 behind; omit to front-light the side "
                                      "of the car the camera sees (sun behind the camera's shoulder)")
    lamps_on: bool = Field(False, description="car lamps lit: false in daylight; true at dusk, night, in rain "
                           "or fog, or whenever the user asks for the lights on")


class GradePlan(BaseModel):
    """Compositor grade, from the car's scene-linear EXR: highlight detail is never lost."""
    car_exposure_ev: float = Field(0.0, ge=-3, le=2, description="the car's exposure in stops: lower it when the "
                                   "paint looks pale or blown out, raise it when the car is too dark for the scene")
    exposure_ev: float = Field(0.0, ge=-2, le=1, description="whole-frame exposure in stops (car and background)")


class ShotPlan(BaseModel):
    location_key: str = Field(description="short slug for the place, used as its memory key")
    location_label: str = Field(description="human label for the place")
    concept: str = Field(description="one sentence: the shot idea (angle, what it shows, why it sells this place)")
    set: SetPlan
    placement: Placement = Field(default_factory=Placement)
    camera: CameraPlan
    motion: MotionPlan = Field(default_factory=MotionPlan)
    light: LightPlan
    livery: str | None = Field(None, description="the car's livery: a name from the liveries list; omit for the "
                               "default")
    grade: GradePlan | None = Field(None, description="compositor grade; set only to fix exposure or when asked")
    scene_prompt: str = Field(description="what the place looks like, for the image model; 'empty, deserted road'")
    reference_keys: list[str] = Field(default_factory=list, description="refs/<location>/... images, look only")


class PlanOutput(BaseModel):
    """What the `plan` node returns."""
    plan: ShotPlan
    start_stage: Stage = Field("SETUP", description="SETUP for a new shot; a later stage when editing "
                               "a stored shot only needs part of the pipeline redone")
    memory_used: list[str] = Field(default_factory=list, description="which remembered settings were reused")
    ideas: list[str] = Field(default_factory=list, description="2-3 other shots worth making at this place, one "
                             "line each (angle, height, lens, what it shows); shown to the user as suggestions")
    note: str = Field("", description="one sentence for the user: what changes and why, or what could not be "
                      "done (e.g. a colour the car has no livery for, with the available ones)")


class Intent(BaseModel):
    """What the `intake` node returns."""
    kind: Literal["new_shots", "edit_shot", "rerun", "question"]
    locations: list[str] = Field(default_factory=list, description="one entry per requested location/shot")
    shot_ref: str = Field("", description="location_key or job_id of an existing shot (edit_shot, rerun)")
    request: str = Field(description="the user's request restated with every specific detail")


class RepairDecision(BaseModel):
    """What the `repair` node returns."""
    stage: Stage = Field(description="earliest stage that must be redone")
    reason: str
    plan: ShotPlan = Field(description="the full revised plan (unchanged fields copied over)")


class Approval(BaseModel):
    """The user's answer to the approval request."""
    decision: Literal["approve", "revise", "reject"]
    note: str = ""
