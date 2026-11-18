"""Tool-code: build the stand-in set (road, ground, grey stand-ins) in TS_SET.

The set is data. The road is a segment list after a fixed straight behind the origin; the
stand-ins are a list of elements (`ELEMENT_SPEC`: hillside, mountains, buildings, trees, sea,
wall). The presets are named entries of the same data: `ROAD_SHAPES` (name -> segment list
builder) and `TERRAINS` (name -> element list), so presets and free elements share one code path.
"""

__all__ = ("Params", "main", "ROAD_SHAPES", "TERRAINS", "ELEMENT_SPEC", "SIDES")

import json
from typing import NamedTuple

ROAD_BEHIND_M = 60.0
ROAD_AHEAD_M = 400.0
ROAD_Z = 0.01
GROUND_HALF_M = 12000.0
# The ground is a grid: GROUND_CELL_M cells over the road's extent + GROUND_FINE_MARGIN_M, then
# cells doubling in size outwards (at most GROUND_MAX_CELLS fine cells per axis).
GROUND_CELL_M = 50.0
GROUND_FINE_MARGIN_M = 300.0
GROUND_MAX_CELLS = 60
# Road markings (role "road_marking"): solid edge lines inset from the road edge and a dashed
# centre line. They give the image model exact lane geometry running parallel to the car.
MARK_WIDTH_M = 0.15
MARK_EDGE_INSET_M = 0.3
MARK_DASH_M, MARK_GAP_M = 3.0, 9.0
MARK_LIFT_M = 0.004
# Free road segments: {"straight_m": L} or {"turn_deg": A, "radius_m": R} (A > 0 = left turn).
ROAD_STRAIGHT_M = (1.0, 1000.0)
ROAD_RADIUS_M = (6.0, 2000.0)
ROAD_TURN_MAX_DEG = 270.0


class Params(NamedTuple):
    road_json: str
    elements_json: str
    road_shape: str
    terrain: str
    camera_side: str
    road_width_m: float
    seed: int


# ---------------------------------------------------------------------------
# Road shapes: segments ("S", length) or ("A", radius, angle_deg; + = left turn). The road
# starts ROAD_BEHIND_M behind the origin heading +Y; the remainder is filled with a straight.

def _road_straight(rng):
    return []


def _road_curve(sign):
    def build(rng):
        return [("S", rng.uniform(25.0, 60.0)), ("A", rng.uniform(80.0, 200.0), sign * rng.uniform(40.0, 90.0))]
    return build


def _road_hairpin(sign):
    def build(rng):
        return [("S", rng.uniform(30.0, 55.0)), ("A", rng.uniform(12.0, 18.0), sign * 180.0),
                ("S", rng.uniform(70.0, 110.0)), ("A", rng.uniform(40.0, 80.0), -sign * 90.0)]
    return build


ROAD_SHAPES = {
    "straight": _road_straight,
    "curve_left": _road_curve(+1),
    "curve_right": _road_curve(-1),
    "hairpin_left": _road_hairpin(+1),
    "hairpin_right": _road_hairpin(-1),
}


def _parse_road(items) -> list:
    """Free road items -> segment tuples; raises ValueError naming the bad item."""
    if not isinstance(items, list):
        raise ValueError("road must be a list of segments")
    segs = []
    for k, it in enumerate(items):
        where = "road[{}] {!r}".format(k, it)
        if not isinstance(it, dict):
            raise ValueError(where + ": expected {\"straight_m\": L} or {\"turn_deg\": A, \"radius_m\": R}")
        keys = set(it)
        if keys == {"straight_m"}:
            length = _number(it["straight_m"], where)
            if not ROAD_STRAIGHT_M[0] <= length <= ROAD_STRAIGHT_M[1]:
                raise ValueError(where + ": straight_m must be in [{}, {}]".format(*ROAD_STRAIGHT_M))
            segs.append(("S", length))
        elif keys == {"turn_deg", "radius_m"}:
            angle, radius = _number(it["turn_deg"], where), _number(it["radius_m"], where)
            if not ROAD_RADIUS_M[0] <= radius <= ROAD_RADIUS_M[1]:
                raise ValueError(where + ": radius_m must be in [{}, {}]".format(*ROAD_RADIUS_M))
            if angle == 0.0 or abs(angle) > ROAD_TURN_MAX_DEG:
                raise ValueError(where + ": turn_deg must be non-zero with |turn_deg| <= {}".format(ROAD_TURN_MAX_DEG))
            segs.append(("A", radius, angle))
        else:
            raise ValueError(where + ": expected exactly {\"straight_m\"} or {\"turn_deg\", \"radius_m\"}")
    return segs


def _road_centerline(segments, step: float):
    """Sample the segment list every `step` metres; returns an (N, 2) numpy array."""
    import math
    import numpy as np
    total = ROAD_BEHIND_M + ROAD_AHEAD_M
    segs = [("S", ROAD_BEHIND_M)] + list(segments)
    used = sum(s[1] if s[0] == "S" else s[1] * math.radians(abs(s[2])) for s in segs)
    segs.append(("S", max(0.0, total - used)))
    x, y, h = 0.0, -ROAD_BEHIND_M, math.pi / 2.0  # heading as math angle; +Y = 90 deg
    pts = [(x, y)]
    carry = 0.0
    for seg in segs:
        if seg[0] == "S":
            length, curv = seg[1], 0.0
        else:
            length, curv = seg[1] * math.radians(abs(seg[2])), math.copysign(1.0 / seg[1], seg[2])
        s = step - carry
        x0, y0, h0 = x, y, h
        while s <= length + 1e-9:
            if curv == 0.0:
                px, py = x0 + s * math.cos(h0), y0 + s * math.sin(h0)
            else:
                r = 1.0 / curv
                px = x0 + r * (math.sin(h0 + curv * s) - math.sin(h0))
                py = y0 - r * (math.cos(h0 + curv * s) - math.cos(h0))
            pts.append((px, py))
            s += step
        carry = length - (s - step)
        if curv == 0.0:
            x, y = x0 + length * math.cos(h0), y0 + length * math.sin(h0)
        else:
            r = 1.0 / curv
            x = x0 + r * (math.sin(h0 + curv * length) - math.sin(h0))
            y = y0 - r * (math.cos(h0 + curv * length) - math.cos(h0))
            h = h0 + curv * length
    return np.asarray(pts)


# ---------------------------------------------------------------------------
# Elements. Every field is optional (defaults below). "range" fields take [lo, hi] or one
# number; per-item ranges (distance, size, height of each building, ...) are drawn per item,
# the others once per element. Bounds are validated; unknown kinds/fields are errors.
#
# Sides are relative to the road at the nearest road point (left/right of the direction of
# travel; on the straight at the origin, left = x < 0). "near" is `camera_side`, "far" the other.

SIDES = ("near", "far", "left", "right", "both")
LEFT, RIGHT = 1, -1  # sign along the road's left normal

ELEMENT_SPEC = {
    # Height field rising away from the road edge; height_deg caps it (angle above the horizon
    # seen from the origin at camera height).
    "hillside": {"side": ("side", None, "near"), "grade": ("range", (0.05, 1.5), 0.3),
                 "height_deg": ("range", (0.1, 45.0), 4.0)},
    # Ridges around the origin: distance from the origin, crest height / length as angles seen
    # from the origin, jagged crest, flat_top = mesa cut, steepness = half-width / height.
    "mountains": {"side": ("side", None, "both"), "count": ("int", (1, 60), 8),
                  "distance_m": ("range", (100.0, 20000.0), (1500.0, 5000.0)),
                  "height_deg": ("range", (0.1, 30.0), (1.0, 2.5)), "width_deg": ("range", (2.0, 120.0), (20.0, 50.0)),
                  "jagged": ("range", (0.0, 1.0), 0.4), "flat_top": ("range", (0.0, 0.6), 0.0),
                  "steepness": ("range", (0.5, 6.0), 2.0), "road_clearance_m": ("range", (0.0, 1000.0), 60.0)},
    # Boxes facing the road; distance from the road edge; height_deg caps roofs. Buildings,
    # trees and walls stand on the hillsides of elements listed before them.
    "buildings": {"side": ("side", None, "both"), "count": ("int", (1, 3000), 100),
                  "distance_m": ("range", (0.0, 3000.0), (10.0, 200.0)), "size_m": ("range", (2.0, 200.0), (7.0, 16.0)),
                  "height_m": ("range", (1.0, 500.0), (5.0, 18.0)), "height_deg": ("range", (0.1, 45.0), 12.0)},
    "trees": {"side": ("side", None, "both"), "count": ("int", (1, 20000), 1000),
              "distance_m": ("range", (0.0, 3000.0), (3.0, 220.0)), "height_m": ("range", (2.0, 80.0), (12.0, 28.0)),
              "height_deg": ("range", (0.1, 45.0), 9.0)},
    # Sea beyond a straight shoreline on one side (the whole road stays on land); distance_m =
    # shore margin from the road edge. At most one sea per set.
    "sea": {"side": ("side1", None, "far"), "distance_m": ("range", (0.0, 2000.0), 30.0)},
    # Continuous wall/barrier along the road (role standin_wall).
    "wall": {"side": ("side", None, "far"), "distance_m": ("range", (0.0, 20.0), 0.5),
             "height_m": ("range", (0.3, 6.0), 1.0), "thickness_m": ("range", (0.1, 3.0), 0.5)},
}

# Terrain presets: element lists in the same vocabulary (near = the camera's side).
TERRAINS = {
    # Slopes rise on the camera side; the far side stays low, so the frame shows mid ridges and
    # narrow far peaks breaking the sky line.
    "alpine": [
        {"kind": "hillside", "side": "near", "grade": [0.25, 0.4], "height_deg": 4.5},
        {"kind": "hillside", "side": "far", "grade": [0.25, 0.4], "height_deg": 1.2},
        {"kind": "mountains", "count": 8, "distance_m": [700, 1600], "height_deg": [1.0, 2.4],
         "width_deg": [20, 45], "steepness": 1.4, "jagged": 0.5},
        {"kind": "mountains", "count": 16, "distance_m": [2500, 7000], "height_deg": [2.8, 5.8],
         "width_deg": [8, 20], "steepness": 1.1, "jagged": 0.7, "road_clearance_m": 200},
    ],
    "high_desert": [
        {"kind": "mountains", "count": 7, "distance_m": [1500, 3500], "height_deg": [0.8, 2.0],
         "width_deg": [30, 70], "steepness": 4.0, "jagged": 0.15, "flat_top": 0.35},
        {"kind": "mountains", "count": 9, "distance_m": [4000, 10000], "height_deg": [1.2, 2.8],
         "width_deg": [25, 60], "steepness": 3.0, "jagged": 0.2, "flat_top": 0.2},
    ],
    # Town on the camera side, sea and a few low houses on the far side.
    "coastal_town": [
        {"kind": "sea", "side": "far", "distance_m": [15, 45]},
        {"kind": "buildings", "side": "near", "count": 140, "distance_m": [10, 260], "size_m": [7, 16],
         "height_m": [5, 18], "height_deg": 12},
        {"kind": "buildings", "side": "far", "count": 40, "distance_m": [20, 200], "size_m": [7, 16],
         "height_m": [5, 18], "height_deg": 4},
        {"kind": "mountains", "count": 9, "distance_m": [900, 2600], "height_deg": [1.5, 4.0],
         "width_deg": [25, 55], "steepness": 2.4, "jagged": 0.3},
    ],
    "city": [
        {"kind": "buildings", "side": "near", "count": 300, "distance_m": [6, 160], "size_m": [12, 30],
         "height_m": [8, 45], "height_deg": 12},
        {"kind": "buildings", "side": "far", "count": 300, "distance_m": [14, 160], "size_m": [12, 30],
         "height_m": [8, 45], "height_deg": 4.5},
        {"kind": "buildings", "side": "near", "count": 25, "distance_m": [250, 900], "size_m": [20, 40],
         "height_m": [40, 200], "height_deg": 6},
        {"kind": "buildings", "side": "far", "count": 25, "distance_m": [250, 900], "size_m": [20, 40],
         "height_m": [40, 200], "height_deg": 4.5},
    ],
    # Dense trees on the camera side; a meadow before the tree line on the far side.
    "forest": [
        {"kind": "trees", "side": "near", "count": 3000, "distance_m": [3, 220], "height_m": [12, 28],
         "height_deg": 9},
        {"kind": "trees", "side": "far", "count": 1500, "distance_m": [120, 420], "height_m": [12, 28],
         "height_deg": 3.5},
        {"kind": "mountains", "count": 8, "distance_m": [1500, 5000], "height_deg": [0.8, 2.2],
         "width_deg": [25, 55], "steepness": 3.0, "jagged": 0.2, "road_clearance_m": 200},
    ],
    "plains": [
        {"kind": "mountains", "count": 8, "distance_m": [4000, 10000], "height_deg": [0.2, 0.6],
         "width_deg": [30, 70], "steepness": 5.0, "jagged": 0.1, "road_clearance_m": 300},
    ],
}


def _number(v, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("{}: {!r} is not a number".format(where, v))
    return float(v)


def _parse_element(item, where: str) -> dict:
    """Validate one element dict -> {kind, side, field: value or (lo, hi), ...} with defaults."""
    if not isinstance(item, dict):
        raise ValueError(where + ": expected an object with \"kind\"")
    kind = item.get("kind")
    if kind not in ELEMENT_SPEC:
        raise ValueError("{}: kind {!r} must be one of {}".format(where, kind, tuple(ELEMENT_SPEC)))
    spec = ELEMENT_SPEC[kind]
    where = "{} ({})".format(where, kind)
    unknown = sorted(set(item) - set(spec) - {"kind"})
    if unknown:
        raise ValueError("{}: unknown field(s) {}; allowed: {}".format(where, unknown, tuple(spec)))
    out = {"kind": kind}
    for name, (typ, bounds, default) in spec.items():
        v = item.get(name, default)
        field = "{}: {}".format(where, name)
        if typ in ("side", "side1"):
            allowed = SIDES if typ == "side" else tuple(s for s in SIDES if s != "both")
            if v not in allowed:
                raise ValueError("{} {!r} must be one of {}".format(field, v, allowed))
            out[name] = v
        elif typ == "int":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
                raise ValueError("{} {!r} must be an integer".format(field, v))
            if not bounds[0] <= v <= bounds[1]:
                raise ValueError("{} {!r} must be in [{}, {}]".format(field, v, *bounds))
            out[name] = int(v)
        else:
            if isinstance(v, (list, tuple)):
                if len(v) != 2:
                    raise ValueError("{} {!r} must be a number or [lo, hi]".format(field, v))
                lo, hi = sorted((_number(v[0], field), _number(v[1], field)))
            else:
                lo = hi = _number(v, field)
            if lo < bounds[0] or hi > bounds[1]:
                raise ValueError("{} {!r} must be within [{}, {}]".format(field, v, *bounds))
            out[name] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# Framing model. Stand-ins are sized by the angle they subtend seen from the car's usual
# spot (the road origin) at camera height: a crest at distance r is at most
# CAM_H + tan(theta) * r high. Typical shots (focal 35-85 mm, elevation 2-8 deg, car in the
# lower part of the frame) show only ~6-13 deg above the horizon, so crests of 1-5 deg leave
# roughly 20-45 % sky. The presets keep the far side (the background seen across the road)
# lower and further away than the near (camera) side so the lens never faces a wall.

CAM_H = 1.5
EXTEND_M = 500.0  # stand-ins also line virtual road extensions beyond both road ends


class _Ctx:
    """Everything an element builder needs: rng, road, sides, clearance tests and output parts."""

    def __init__(self, rng, cl, width, near_sign):
        import numpy as np
        from mathutils.kdtree import KDTree  # pylint: disable=import-error
        self.rng, self.cl, self.width, self.near = rng, cl, width, near_sign
        tan = np.gradient(cl, axis=0)
        self.tan = tan / np.linalg.norm(tan, axis=1, keepdims=True)
        self.left = np.stack([-self.tan[:, 1], self.tan[:, 0]], axis=1)
        self.kd = KDTree(len(cl))
        for i, (x, y) in enumerate(cl):
            self.kd.insert((x, y, 0.0), i)
        self.kd.balance()
        # Placement line: the road plus straight extensions, so views looking past either road
        # end (e.g. back along the road from a front three-quarter camera) are not empty.
        steps = np.arange(1, int(EXTEND_M / 10.0) + 1)[:, None] * 10.0
        before = cl[0] - self.tan[0] * steps[::-1]
        after = cl[-1] + self.tan[-1] * steps
        self.line = np.concatenate([before, cl[::20], after])
        ltan = np.gradient(self.line, axis=0)
        ltan /= np.linalg.norm(ltan, axis=1, keepdims=True)
        self.line_tan = ltan
        self.line_left = np.stack([-ltan[:, 1], ltan[:, 0]], axis=1)
        self.sea = None  # (nx, ny, c): sea where nx * x + ny * y > c
        self.land = lambda x, y: True
        self.parts = {}  # role -> [verts list, faces list]
        self.slopes = []  # hillside height fields: (x0, y0, spacing, z array (ny, nx))

    def ground(self, x, y):
        """Height of the hillsides built so far at (x, y) (0 = flat ground), bilinear."""
        top = 0.0
        for x0, y0, sp, z in self.slopes:
            fx, fy = (x - x0) / sp, (y - y0) / sp
            i, j = int(fx), int(fy)
            if fx < 0.0 or fy < 0.0 or i >= z.shape[1] - 1 or j >= z.shape[0] - 1:
                continue
            tx, ty = fx - i, fy - j
            h = (z[j, i] * (1 - tx) + z[j, i + 1] * tx) * (1 - ty) + (z[j + 1, i] * (1 - tx) + z[j + 1, i + 1] * tx) * ty
            top = max(top, float(h) - 0.3)
        return top

    def road_dist(self, x, y):
        _co, i, d = self.kd.find((x, y, 0.0))
        return d, i

    def signs(self, side):
        """Side name -> tuple of LEFT / RIGHT signs."""
        return {"near": (self.near,), "far": (-self.near,), "left": (LEFT,), "right": (RIGHT,),
                "both": (LEFT, RIGHT)}[side]

    def side_of(self, x, y, i=None):
        """LEFT or RIGHT of the road at its nearest point (beyond the road ends this equals the
        side of the straight extension)."""
        if i is None:
            i = self.road_dist(x, y)[1]
        p, n = self.cl[i], self.left[i]
        return LEFT if (x - p[0]) * n[0] + (y - p[1]) * n[1] >= 0.0 else RIGHT

    def cap(self, x, y, theta_deg):
        """Height whose top subtends `theta_deg` above the horizon, seen from the origin."""
        import math
        return CAM_H + math.tan(math.radians(theta_deg)) * math.hypot(x, y)

    def clear(self, x, y, margin):
        return self.road_dist(x, y)[0] > self.width / 2.0 + margin and self.land(x, y)

    def add(self, role, verts, faces):
        vs, fs = self.parts.setdefault(role, ([], []))
        base = len(vs)
        vs.extend(verts)
        fs.extend(tuple(base + i for i in f) for f in faces)


def _fractal(rng, n_pow, rough):
    """1-D midpoint displacement in [0, 1], 2**n_pow + 1 samples."""
    n = 2 ** n_pow
    v = [0.0] * (n + 1)
    v[0], v[n] = rng.random(), rng.random()
    step, amp = n, 1.0
    while step > 1:
        half = step // 2
        for i in range(half, n, step):
            v[i] = (v[i - half] + v[i + half]) / 2.0 + rng.uniform(-amp, amp) * 0.5
        step, amp = half, amp * rough
    lo, hi = min(v), max(v)
    return [(x - lo) / ((hi - lo) or 1.0) for x in v]


def _ridge(ctx, cx, cy, heading, length, height, hw_factor, jag, mesa=0.0):
    """A ridge: jagged crest along `heading` (radians, math angle), slopes to below ground."""
    import math
    rng = ctx.rng
    prof = _fractal(rng, 5, 0.62 if jag > 0.3 else 0.45)
    n = len(prof)
    ux, uy = math.cos(heading), math.sin(heading)
    nx, ny = -uy, ux
    cross = (-1.0, -0.55, -0.2, 0.0, 0.2, 0.55, 1.0)
    zf = (0.0, 0.42, 0.86, 1.0, 0.86, 0.42, 0.0)
    verts = []
    for i, p in enumerate(prof):
        t = i / (n - 1)
        taper = math.sin(math.pi * t) ** 0.6
        h = height * taper * (1.0 - jag + jag * p)
        if mesa > 0.0:
            h = min(h, height * (1.0 - mesa))
        hw = hw_factor * max(h, 0.3 * height)
        ax, ay = cx + ux * (t - 0.5) * length, cy + uy * (t - 0.5) * length
        for c, z in zip(cross, zf):
            jitter = 1.0 + rng.uniform(-0.12, 0.12) * (c != 0.0)
            verts.append((ax + nx * c * hw * jitter, ay + ny * c * hw * jitter, z * h - 3.0))
    m = len(cross)
    faces = [(i * m + j, i * m + j + 1, (i + 1) * m + j + 1, (i + 1) * m + j)
             for i in range(n - 1) for j in range(m - 1)]
    ctx.add("standin_ridge", verts, faces)


def _ridge_ring(ctx, signs, count, dist, theta, width_deg, hw_factor, jag, mesa, margin):
    """`count` overlapping ridges around the origin on the sides `signs`. Crest height subtends
    `theta` degrees and length `width_deg` degrees seen from the origin; kept clear of the road
    and the sea. A ridge drawn on the wrong side is mirrored through the origin. Returns the
    number placed."""
    import math
    rng = ctx.rng
    base = rng.uniform(0.0, 2.0 * math.pi)
    placed = 0
    for k in range(count):
        a = base + 2.0 * math.pi * (k + rng.uniform(-0.35, 0.35)) / count
        d = rng.uniform(*dist)
        th = math.tan(math.radians(rng.uniform(*theta)))
        wd = math.radians(rng.uniform(*width_deg))
        if len(signs) == 1 and ctx.side_of(d * math.cos(a), d * math.sin(a)) != signs[0]:
            a += math.pi
            if ctx.side_of(d * math.cos(a), d * math.sin(a)) != signs[0]:
                continue
        heading = a + math.pi / 2.0 + rng.uniform(-0.3, 0.3)
        ux, uy = math.cos(heading), math.sin(heading)
        for _try in range(6):
            h, ln = th * d, wd * d
            hw = hw_factor * h
            cx, cy = d * math.cos(a), d * math.sin(a)
            rel = ctx.cl - (cx, cy)
            along = rel[:, 0] * ux + rel[:, 1] * uy
            across = -rel[:, 0] * uy + rel[:, 1] * ux
            hits_road = bool(((abs(along) < ln / 2.0 + margin) & (abs(across) < 1.15 * hw + margin)).any())
            on_land = all(ctx.land(cx + ux * t * ln - uy * c * hw, cy + uy * t * ln + ux * c * hw)
                          for t in (-0.5, 0.0, 0.5) for c in (-1.15, 0.0, 1.15))
            if not hits_road and on_land:
                _ridge(ctx, cx, cy, heading, ln, h, hw_factor, jag, mesa)
                placed += 1
                break
            d *= 1.25
    return placed


def _box(ctx, role, cx, cy, heading, sx, sy, h):
    import math
    c, s = math.cos(heading), math.sin(heading)
    corners = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]
    pts = [(cx + x * c - y * s, cy + x * s + y * c) for x, y in corners]
    ground = [ctx.ground(x, y) for x, y in pts]
    base, top = min(ground) - 0.5, max(ground) + h
    verts = [(x, y, base) for x, y in pts] + [(x, y, top) for x, y in pts]
    faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    ctx.add(role, verts, faces)


def _place(ctx, signs, dist, extra=0.0, near=0.4):
    """A random spot beside the (extended) road on one of the sides `signs`, `dist` = (min,
    max) distance from the road edge: (x, y, heading). `near` = share placed in the first 60 m."""
    import math
    rng = ctx.rng
    lo, hi = dist
    d = rng.uniform(lo, min(hi, lo + 60.0)) if rng.random() < near else rng.uniform(lo, hi)
    i = rng.randrange(len(ctx.line))
    sign = rng.choice(signs)
    x, y = ctx.line[i] + ctx.line_left[i] * sign * (ctx.width / 2.0 + d + extra)
    return float(x), float(y), math.atan2(ctx.line_tan[i][1], ctx.line_tan[i][0])


def _buildings(ctx, signs, count, dist, size, height, theta):
    """Boxes facing the road; roofs capped at angular height `theta` (degrees) seen from the
    origin. Returns the number placed."""
    import math
    rng = ctx.rng
    min_h = min(height[0], max(4.0, 0.4 * height[0]))
    placed = []
    for _attempt in range(count * 12):
        if len(placed) >= count:
            break
        sx, sy = rng.uniform(*size), rng.uniform(*size)
        px, py, heading = _place(ctx, signs, dist, sy / 2.0)
        r = 0.5 * math.hypot(sx, sy)
        if any((px - qx) ** 2 + (py - qy) ** 2 < (r + qr) ** 2 * 0.55 for qx, qy, qr in placed):
            continue
        c, s = math.cos(heading), math.sin(heading)
        corners = [(px + x * c - y * s, py + x * s + y * c) for x, y in
                   ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
        if not all(ctx.clear(x, y, 3.0) for x, y in corners):
            continue
        # Roofs vary below the cap so skylines stay broken rather than one flat line.
        h = max(min_h, min(rng.uniform(*height), ctx.cap(px, py, theta) * rng.uniform(0.55, 1.0)))
        _box(ctx, "standin_building", px, py, heading, sx, sy, h)
        placed.append((px, py, r))
    return len(placed)


def _trees(ctx, signs, count, dist, height, theta):
    """Conifer cones down to the ground. Returns the number placed."""
    import math
    rng = ctx.rng
    n_seg = 6
    min_h = min(height[0], 4.0)
    placed = 0
    for _attempt in range(count * 4):
        if placed >= count:
            break
        px, py, _heading = _place(ctx, signs, dist, near=0.6)
        px, py = px + rng.uniform(-4, 4), py + rng.uniform(-4, 4)
        h = max(min_h, min(rng.uniform(*height), ctx.cap(px, py, theta)))
        r = h * rng.uniform(0.18, 0.28)
        if not ctx.clear(px, py, r + 1.5):
            continue
        ring = [(px + r * math.cos(2 * math.pi * k / n_seg), py + r * math.sin(2 * math.pi * k / n_seg))
                for k in range(n_seg)]
        ground = [ctx.ground(x, y) for x, y in ring]
        verts = [(x, y, min(ground) - 0.3) for x, y in ring] + [(px, py, ctx.ground(px, py) + h)]
        ctx.add("standin_tree", verts, [(k, (k + 1) % n_seg, n_seg) for k in range(n_seg)])
        placed += 1
    return placed


def _hillsides(ctx, signs, grade, theta, reach=420.0, spacing=6.0):
    """Height field rising away from the road (role standin_ridge) on the sides `signs`, capped
    by the angular height `theta` (degrees) seen from the origin; flat parts are dropped.
    Returns 1 when any slope was built."""
    import math
    import numpy as np
    rng, cl = ctx.rng, ctx.cl
    lo = cl.min(axis=0) - reach
    hi = cl.max(axis=0) + reach
    nx = int((hi[0] - lo[0]) / spacing) + 1
    ny = int((hi[1] - lo[1]) / spacing) + 1
    xs = lo[0] + np.arange(nx) * spacing
    ys = lo[1] + np.arange(ny) * spacing
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(4)]
    d0 = ctx.width / 2.0 + 4.0
    top = math.tan(math.radians(theta))
    z = np.zeros((ny, nx))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            d, k = ctx.road_dist(x, y)
            if d <= d0 or ctx.side_of(x, y, k) not in signs or not ctx.land(x, y):
                continue
            noise = 1.0 + 0.2 * (math.sin(x / 37.0 + ph[0]) * math.sin(y / 53.0 + ph[1])
                                 + 0.5 * math.sin(x / 13.0 + ph[2]) * math.cos(y / 17.0 + ph[3]))
            edge = min(x - lo[0], hi[0] - x, y - lo[1], hi[1] - y)
            fade = min(1.0, edge / 120.0)
            h = min(grade * (d - d0), top * math.hypot(x, y))
            z[j, i] = max(0.0, h) * noise * fade * fade
    verts = [(float(x), float(y), float(z[j, i]) - 0.3) for j, y in enumerate(ys) for i, x in enumerate(xs)]
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if max(z[j, i], z[j, i + 1], z[j + 1, i], z[j + 1, i + 1]) < 0.5:
                continue
            a = j * nx + i
            faces.append((a, a + 1, a + nx + 1, a + nx))
    used = sorted({v for f in faces for v in f})
    remap = {v: k for k, v in enumerate(used)}
    ctx.add("standin_ridge", [verts[v] for v in used], [tuple(remap[v] for v in f) for f in faces])
    ctx.slopes.append((float(xs[0]), float(ys[0]), spacing, z))
    return 1 if faces else 0


def _coast(ctx, sign, margin):
    """Put the sea on side `sign`, diagonal to the road at the origin (seeded: behind or ahead)
    so a three-quarter view sees sea horizon on one part of the frame and land on the other; the
    whole road stays on land, with the shore as close as the road allows."""
    import math
    import numpy as np
    i0 = min(len(ctx.cl) - 1, int(round(ROAD_BEHIND_M / ROAD_STEP_M)))
    out = ctx.left[i0] * sign
    base = math.degrees(math.atan2(out[1], out[0]))
    prefer = ctx.rng.choice((-45.0, 45.0))
    best = None
    for deg in range(-100, 101, 10):  # 0 = straight out from the road at the origin
        a = math.radians(base + deg + ctx.rng.uniform(-4.0, 4.0))
        n = np.array([math.cos(a), math.sin(a)])
        c = float((ctx.cl @ n).max()) + ctx.width / 2.0 + margin
        score = c * (1.0 + 0.006 * abs(deg - prefer))
        if best is None or score < best[0]:
            best = (score, float(n[0]), float(n[1]), c)
    _score, nx, ny, c = best
    ctx.sea = (nx, ny, c)
    ctx.land = lambda x, y: nx * x + ny * y < c - 5.0
    return 1


def _wall(ctx, signs, offset, height, thick, every=4):
    """A continuous wall strip `offset` metres beyond the road edge (role standin_wall),
    broken where it would cut into another part of the road or the sea. Returns the number of
    ~2 m pieces built."""
    cl, left = ctx.cl, ctx.left
    inner = ctx.width / 2.0 + offset
    idx = list(range(0, len(cl), every))
    if idx[-1] != len(cl) - 1:
        idx.append(len(cl) - 1)
    pieces = 0
    for sign in signs:
        run = []

        def flush(run):
            if len(run) < 2:
                return 0
            verts, faces = [], []
            for a, b in run:
                g = (ctx.ground(*a), ctx.ground(*b))
                lo, hi = min(g) - 0.3, max(g) + height
                verts += [(a[0], a[1], lo), (a[0], a[1], hi), (b[0], b[1], hi), (b[0], b[1], lo)]
            for k in range(len(run) - 1):
                p, q = 4 * k, 4 * k + 4
                faces += [(p, q, q + 1, p + 1), (p + 1, q + 1, q + 2, p + 2), (p + 2, q + 2, q + 3, p + 3)]
            last = 4 * (len(run) - 1)
            faces += [(0, 1, 2, 3), (last + 3, last + 2, last + 1, last)]
            ctx.add("standin_wall", verts, faces)
            return len(run) - 1

        for i in idx:
            n = left[i] * sign
            a = cl[i] + n * inner
            b = a + n * thick
            ok = ctx.road_dist(float(a[0]), float(a[1]))[0] >= inner - 0.25 and ctx.land(a[0], a[1]) \
                and ctx.land(b[0], b[1])
            if ok:
                run.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
            else:
                pieces += flush(run)
                run = []
        pieces += flush(run)
    return pieces


def _apply(ctx, el) -> dict:
    """Build one parsed element; returns its summary {kind, side, resolved, placed}."""
    rng = ctx.rng
    kind, side = el["kind"], el["side"]
    signs = ctx.signs(side)
    once = lambda name: rng.uniform(*el[name])  # noqa: E731
    if kind == "hillside":
        placed = _hillsides(ctx, signs, once("grade"), once("height_deg"))
    elif kind == "mountains":
        placed = _ridge_ring(ctx, signs, el["count"], el["distance_m"], el["height_deg"], el["width_deg"],
                             once("steepness"), once("jagged"), once("flat_top"), once("road_clearance_m"))
    elif kind == "buildings":
        placed = _buildings(ctx, signs, el["count"], el["distance_m"], el["size_m"], el["height_m"],
                            once("height_deg"))
    elif kind == "trees":
        placed = _trees(ctx, signs, el["count"], el["distance_m"], el["height_m"], once("height_deg"))
    elif kind == "sea":
        placed = _coast(ctx, signs[0], once("distance_m"))
    else:
        placed = _wall(ctx, signs, once("distance_m"), once("height_m"), once("thickness_m"))
    return {"kind": kind, "side": side, "resolved": ["left" if s == LEFT else "right" for s in signs],
            "placed": placed}


def _clip_half_plane(poly, nx, ny, c, keep_below):
    """Clip a convex polygon [(x, y), ...] to nx*x + ny*y <= c (or >= c)."""
    sgn = 1.0 if keep_below else -1.0
    f = [sgn * (nx * x + ny * y - c) for x, y in poly]
    out = []
    for k, (p, fp) in enumerate(zip(poly, f)):
        q, fq = poly[(k + 1) % len(poly)], f[(k + 1) % len(poly)]
        if fp <= 0.0:
            out.append(p)
        if (fp < 0.0) != (fq < 0.0) and fp != fq:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _grid_axis(lo: float, hi: float) -> list:
    """Cell edges along one axis: GROUND_CELL_M cells over [lo, hi], then doubling outwards to
    +/- GROUND_HALF_M."""
    import math
    n = max(1, min(GROUND_MAX_CELLS, int(math.ceil((hi - lo) / GROUND_CELL_M))))
    step = (hi - lo) / n
    edges = [lo + k * step for k in range(n + 1)]
    below, above, s = [], [], step
    while (below[-1] if below else lo) > -GROUND_HALF_M or (above[-1] if above else hi) < GROUND_HALF_M:
        s *= 2.0
        below.append(max(-GROUND_HALF_M, (below[-1] if below else lo) - s))
        above.append(min(GROUND_HALF_M, (above[-1] if above else hi) + s))
    return sorted(set(below)) + edges + sorted(set(above))


def _ground_cells(cl, sea_plane):
    """((verts, faces) of the land, (verts, faces) of the sea) as grid cells, fine around the
    road. A plane of a few huge polygons loses depth precision in the viewport/Workbench render
    and swallows the road 1 cm above it in elevated views."""
    xs = _grid_axis(float(cl[:, 0].min()) - GROUND_FINE_MARGIN_M, float(cl[:, 0].max()) + GROUND_FINE_MARGIN_M)
    ys = _grid_axis(float(cl[:, 1].min()) - GROUND_FINE_MARGIN_M, float(cl[:, 1].max()) + GROUND_FINE_MARGIN_M)
    parts = (([], []), ([], []))
    for y0, y1 in zip(ys, ys[1:]):
        for x0, x1 in zip(xs, xs[1:]):
            cell = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            polys = [cell, []] if sea_plane is None else [_clip_half_plane(cell, *sea_plane, keep_below=True),
                                                         _clip_half_plane(cell, *sea_plane, keep_below=False)]
            for (verts, faces), poly in zip(parts, polys):
                if len(poly) >= 3:
                    faces.append(tuple(range(len(verts), len(verts) + len(poly))))
                    verts.extend((x, y, 0.0) for x, y in poly)
    return parts


def _road_markings(cl, left, width):
    """(verts, faces) of the edge lines and the dashed centre line along centreline `cl`."""
    verts, faces = [], []

    def strip(offset, i0, i1):
        base = len(verts)
        for i in range(i0, i1 + 1):
            p, n = cl[i], left[i]
            for s in (1.0, -1.0):
                o = offset + s * MARK_WIDTH_M / 2.0
                verts.append((float(p[0] + n[0] * o), float(p[1] + n[1] * o), ROAD_Z + MARK_LIFT_M))
        faces.extend((base + 2 * k, base + 2 * k + 1, base + 2 * k + 3, base + 2 * k + 2) for k in range(i1 - i0))

    last = len(cl) - 1
    edge = width / 2.0 - MARK_EDGE_INSET_M
    for offset in (edge, -edge):
        strip(offset, 0, last)
    dash = max(1, int(round(MARK_DASH_M / ROAD_STEP_M)))
    period = dash + max(1, int(round(MARK_GAP_M / ROAD_STEP_M)))
    for i0 in range(0, last, period):
        strip(0.0, i0, min(i0 + dash, last))
    return verts, faces


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end


def main(params: Params):
    import random
    import time
    import bpy  # pylint: disable=import-error

    if _ts_busy():
        return _ts_busy_result()
    if params.road_shape and params.road_shape not in ROAD_SHAPES:
        return _ts_result("error", message="road_shape must be \"\" or one of {}".format(tuple(ROAD_SHAPES)))
    if params.terrain and params.terrain not in TERRAINS:
        return _ts_result("error", message="terrain must be \"\" or one of {}".format(tuple(TERRAINS)))
    if params.camera_side not in ("left", "right"):
        return _ts_result("error", message="camera_side must be \"left\" or \"right\"")
    try:
        road = _parse_road(json.loads(params.road_json))
        free = json.loads(params.elements_json)
        if not isinstance(free, list):
            raise ValueError("elements must be a list of objects")
        elements = [(_parse_element(e, "terrain {} element {}".format(params.terrain, k)), "terrain")
                    for k, e in enumerate(TERRAINS.get(params.terrain, []))]
        elements += [(_parse_element(e, "elements[{}]".format(k)), "elements") for k, e in enumerate(free)]
    except ValueError as ex:
        return _ts_result("error", message=str(ex))
    if sum(el["kind"] == "sea" for el, _src in elements) > 1:
        return _ts_result("error", message="at most one sea per set (the terrain preset may already add one)")
    width = max(3.0, min(30.0, float(params.road_width_m)))
    t0 = time.monotonic()

    _ts_clear_collection(TS_SET)
    shape = params.road_shape or "straight"
    if road:
        shape = "custom"
    else:
        road = ROAD_SHAPES[shape](random.Random("road:{}:{}".format(params.seed, shape)))
    cl = _road_centerline(road, ROAD_STEP_M)
    key = "set:{}:{}:{}".format(params.seed, params.terrain, shape)
    if params.road_json != "[]" or params.elements_json != "[]" or params.camera_side != "left":
        key += ":{}:{}:{}".format(params.road_json, params.elements_json, params.camera_side)
    ctx = _Ctx(random.Random(key), cl, width, LEFT if params.camera_side == "left" else RIGHT)
    # The sea goes first: every other element keeps to the land.
    elements.sort(key=lambda e: e[0]["kind"] != "sea")
    summaries = []
    for el, src in elements:
        summary = _apply(ctx, el)
        summary["source"] = src
        summaries.append(summary)

    # Road strip.
    half = width / 2.0
    verts = [(float(p[0] + s * n[0] * half), float(p[1] + s * n[1] * half), ROAD_Z)
             for p, n in zip(cl, ctx.left) for s in (1.0, -1.0)]
    faces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(cl) - 1)]
    objects = [_ts_mesh_object("TS_road", "road", verts, faces)]
    objects.append(_ts_mesh_object("TS_road_markings", "road_marking", *_road_markings(cl, ctx.left, width)))

    # Ground (and sea): planes at z = 0, split at the shoreline for coastal sets.
    land, sea = _ground_cells(cl, ctx.sea)
    objects.append(_ts_mesh_object("TS_ground", "ground", *land))
    if sea[1]:
        objects.append(_ts_mesh_object("TS_sea", "sea", *sea))

    names = {"standin_ridge": "TS_ridges", "standin_building": "TS_buildings", "standin_tree": "TS_trees",
             "standin_wall": "TS_walls"}
    for role, (vs, fs) in ctx.parts.items():
        if fs:
            objects.append(_ts_mesh_object(names.get(role, "TS_" + role), role, vs, fs))

    scene = bpy.context.scene
    scene["ts_road_centerline"] = [round(float(v), 3) for v in cl.reshape(-1)]
    scene["ts_road_width_m"] = width
    scene["ts_set_terrain"] = params.terrain
    scene["ts_road_shape"] = shape
    scene["ts_road_s0_m"] = -ROAD_BEHIND_M
    scene["ts_road_z"] = ROAD_Z
    scene["ts_set_seed"] = int(params.seed)
    if ctx.sea is not None:
        scene["ts_sea_side"] = 1 if ctx.sea[0] >= 0.0 else -1
    elif "ts_sea_side" in scene:
        del scene["ts_sea_side"]

    _ts_redraw()
    return _ts_result(
        "ok",
        terrain=params.terrain,
        road_shape=shape,
        camera_side=params.camera_side,
        road_width_m=width,
        road_length_m=round((len(cl) - 1) * ROAD_STEP_M, 1),
        road_start_m=-ROAD_BEHIND_M,
        elements=summaries,
        objects=[{"name": o.name, "role": o["ts_role"], "faces": len(o.data.polygons)} for o in objects],
        build_s=round(time.monotonic() - t0, 2),
    )
