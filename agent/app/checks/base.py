"""Types shared by the checks (kept apart from the registry to avoid import cycles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from app.compositing import CarRender, Composite

Stage = Literal["guide", "plate", "final"]


@dataclass
class CheckContext:
    plate: np.ndarray  # HxWx3 uint8
    guides: dict[str, np.ndarray]  # "grey", "lines", optional "road_mask", "car_mask"
    layout: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    composite: Composite | None = None
    car: CarRender | None = None


@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float  # 0..1
    blocking: bool
    code: str | None  # failure code when not passed
    detail: str


class Check(Protocol):
    name: str
    stage: Stage
    blocking: bool

    async def run(self, ctx: CheckContext) -> CheckResult: ...
