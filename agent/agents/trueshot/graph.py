"""The workflow graph: the only place nodes are wired together.

Design: a verified pipeline with a repair router. Plain-code nodes run the expensive, visible
stages (the pipeline of scripts/dev/lab.py); the LLM decides at intake, plan and repair. Every
pass and retry goes through `dispatch`, which enforces the budget and jumps to the earliest
stage that must be redone.

  intake_prep -> intake -> route_intent --question--> answer -> answered
                                       --plan--> plan -> accept_plan -> dispatch
                                       --rerun--> dispatch
  dispatch --SETUP-----> setup_scene -> guides --ok--> review_guide --approved--> plate_jobs -> plate_worker (x N) -> pick_plate
                                                 --bad--> decide   (guide rejected: no image spent)
                                       review_guide --revise--> repair   --rejected--> finalize   (human guide approval)
           --PLATES----> plate_jobs
           --RENDER----> render_car
           --COMPOSITE-> composite
           --REPLAY----> replay_scene --GUIDES|PLATES|RENDER|COMPOSITE--> ...
           --OUT-------> approve
  pick_plate --ok--> render_car -> composite -> check -> decide
             --none--> decide
  decide --pass|out--> approve   --fix--> dispatch   --repair--> repair -> accept_repair -> dispatch
  approve --approved|rejected|none--> finalize   --revise--> repair
  finalize --next--> plan        (multi-location requests; the next queued location)

Add a node: write it in nodes/<module>.py and add its edges here.
"""

from google.adk import Workflow

from .nodes.approve import approve, finalize
from .nodes.dispatch import dispatch
from .nodes.finish import check, composite, decide
from .nodes.guides import guides
from .nodes.intake import answer, answered, intake, intake_prep, route_intent
from .nodes.plan import accept_plan, plan
from .nodes.plates import pick_plate, plate_jobs, plate_worker
from .nodes.render import render_car
from .nodes.review_guide import review_guide
from .nodes.repair import accept_repair, repair
from .nodes.setup import replay_scene, setup_scene

EDGES = [
    ("START", intake_prep, intake, route_intent),
    (route_intent, {"question": answer, "plan": plan, "rerun": dispatch}),
    (answer, answered),
    (plan, accept_plan, dispatch),
    (dispatch, {"SETUP": setup_scene, "PLATES": plate_jobs, "RENDER": render_car, "COMPOSITE": composite,
                "REPLAY": replay_scene, "OUT": approve}),
    (replay_scene, {"GUIDES": guides, "PLATES": plate_jobs, "RENDER": render_car, "COMPOSITE": composite}),
    (setup_scene, guides),
    (guides, {"ok": review_guide, "bad": decide}),
    (review_guide, {"approved": plate_jobs, "revise": repair, "rejected": finalize}),
    (plate_jobs, plate_worker, pick_plate),
    (pick_plate, {"ok": render_car, "none": decide}),
    (render_car, composite, check, decide),
    (decide, {"pass": approve, "out": approve, "fix": dispatch, "repair": repair}),
    (repair, accept_repair, dispatch),
    (approve, {"approved": finalize, "rejected": finalize, "none": finalize, "revise": repair}),
    (finalize, {"next": plan}),
]


def build_workflow() -> Workflow:
    return Workflow(name="trueshot", edges=EDGES)
