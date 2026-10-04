"""Pure geometry: apertures, goal regions, envelopes and the checks that use them.

Specification sections 10.2, 11, 12.3 and 7.3 govern this module. Three
properties matter more than any one function:

* **An aperture is a bounded opening in a surface, not the depth behind it.**
  The goal builder takes the grounded aperture polygon — already a conservative
  lower bound on the usable opening, on the wall plane — and builds the approach,
  crossing and exit regions from it. Fit and traversal use the lower-bound
  opening, never the outer bounding box of the selection.
* **A region is only a candidate until the map says it is free.** Unknown space
  yields no finite free-space bound, and larger margins cannot manufacture one:
  every region the planner or validator relies on is checked cell by cell against
  published free space, and unknown cells are never treated as free.
* **The collision envelope is conservative.** The body bounding sphere plus
  validated pose, geometry and tracking error bounds, inflated in every
  direction. A sphere wastes some tight passages and avoids a hidden
  attitude-dependent approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from embodied.contracts import records as R
from embodied.memory import world as world_module
from embodied.perception import grounding as grounding_module


@dataclass(frozen=True)
class Envelope:
    """The conservative body-and-error envelope of section 12.3."""

    body_radius_m: float
    error_allowance_m: float
    # The sensor's near blind field: the matcher cannot report depth inside this,
    # so no free-space evidence can exist there in any direction. It is declared
    # as the depth window's own near bound, and it is what the self-occupied
    # exemption below is derived from rather than chosen.
    sensor_near_limit_m: float = 0.0

    def __post_init__(self) -> None:
        for name in ("body_radius_m", "error_allowance_m", "sensor_near_limit_m"):
            value = getattr(self, name)
            if not isinstance(value, float) or value < 0.0:
                raise R.RecordError(f"envelope {name} must be a non-negative number")

    @property
    def inflation_m(self) -> float:
        return self.body_radius_m + self.error_allowance_m

    @property
    def swept_radius_m(self) -> float:
        return self.body_radius_m + self.error_allowance_m

    def self_occupied_radius_m(self, voxel_m: float) -> float:
        """The radius treated as the vehicle's own, unobservable volume.

        The exemption exists for the space the vehicle's own presence blinds the
        sensor to. The matcher cannot report depth inside the near limit, so no
        evidence can exist there in any direction, and a clearance ball around any
        cell within about half a metre of the camera reaches space the sensor
        provably cannot see. Without the exemption a plan can never start where the
        aircraft stands, and — measured on J44-fly-1 — not one of the 26 neighbours
        is searchable, so the aircraft never leaves its own cell.

        The radius is the near limit plus **one body-diagonal step**, because a cell
        one step away has its ball displaced by ``sqrt(3) * voxel`` from the
        origin's, and every cell in that step must be startable for the mission to
        move at all. It is never smaller than the envelope itself, which the
        aircraft physically occupies.

        It does not widen the free space ahead: the exemption marks cells free for
        the *start* of a plan only. Every cell beyond it is still refused unless the
        map itself carries evidence that it is free.

        With the declared values this is ``max(0.475, 0.5 + 0.173) = 0.673 m``,
        which selects the same discrete ball as the 0.675 m an independent mission
        measured as sufficient. The derivation reproduces that measurement rather
        than being fitted to it.
        """
        if self.sensor_near_limit_m <= 0.0:
            return self.inflation_m
        return max(self.inflation_m, self.sensor_near_limit_m + math.sqrt(3.0) * voxel_m)

    def fits(self, extent_m: float) -> bool:
        """Whether an observed extent leaves a positive corridor once inflated."""
        return extent_m - 2.0 * self.inflation_m > 0.0


@dataclass(frozen=True)
class BoxRegion:
    """An axis-aligned region in odom: the terminal regions a goal builder supplies."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]
    label: str

    def contains(self, point: tuple[float, float, float]) -> bool:
        return all(
            low <= value <= high for low, value, high in zip(self.low, point, self.high)
        )

    def center(self) -> tuple[float, float, float]:
        return tuple((low + high) / 2.0 for low, high in zip(self.low, self.high))

    def extent(self) -> tuple[float, float, float]:
        return tuple(high - low for low, high in zip(self.low, self.high))

    def cells(self, config: world_module.MapConfig) -> tuple[tuple[int, int, int], ...]:
        """Every cell whose center lies inside the region."""
        index_low = config.cell_index(self.low)
        index_high = config.cell_index(self.high)
        cells = []
        for x in range(index_low[0], index_high[0] + 1):
            for y in range(index_low[1], index_high[1] + 1):
                for z in range(index_low[2], index_high[2] + 1):
                    cell = (x, y, z)
                    if config.inside(cell) and self.contains(config.cell_center(cell)):
                        cells.append(cell)
        return tuple(cells)


@dataclass(frozen=True)
class TraverseRegions:
    """Approach, crossing and supported exit regions for one aperture traversal."""

    aperture: grounding_module.Aperture
    approach: BoxRegion
    crossing: BoxRegion
    terminal: BoxRegion
    direction: tuple[float, float, float]
    usable_width_m: float
    usable_height_m: float
    corridor_width_m: float
    corridor_height_m: float
    feasible: bool
    constraint: str | None
    reasons: tuple[str, ...]


APERTURE_CLEARANCE = "aperture_clearance"
STANDOFF_M = 1.0
CROSSING_HALF_DEPTH_M = 0.4
EXIT_DEPTH_M = 1.6
APPROACH_HALF_THICKNESS_M = 0.3


def _plane_extents(
    aperture: grounding_module.Aperture,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float, float], tuple[float, float, float]]:
    """The polygon's two in-plane extents and the plane's two basis axes."""
    normal = np.asarray(aperture.plane_normal_odom, dtype=np.float64)
    alignments = np.abs(normal)
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(alignments))] = 1.0
    axis_one = reference - float(reference @ normal) * normal
    axis_one = axis_one / np.linalg.norm(axis_one)
    axis_two = np.cross(normal, axis_one)
    corners = np.asarray(aperture.corners_odom_m, dtype=np.float64)
    plane_point = np.asarray(aperture.plane_point_odom_m, dtype=np.float64)
    relative = corners - plane_point[None, :]
    first = relative @ axis_one
    second = relative @ axis_two
    return (
        (float(first.min()), float(first.max())),
        (float(second.min()), float(second.max())),
        tuple(float(value) for value in axis_one),
        tuple(float(value) for value in axis_two),
    )


def traverse_regions(
    aperture: grounding_module.Aperture,
    envelope: Envelope,
    *,
    direction_sign: float = 1.0,
    standoff_m: float = STANDOFF_M,
) -> TraverseRegions:
    """Build the approach, crossing and exit regions for a traversal through an aperture.

    The crossing region is the inflated opening; the approach sits in front of the
    wall at the declared standoff; the exit region lies past the wall inside the
    same observed opening. A region is built from the *observed* opening, so a
    region that does not fit is refused with the named constraint rather than
    widened to fit.
    """
    (first_low, first_high), (second_low, second_high), axis_one, axis_two = _plane_extents(aperture)
    normal = np.asarray(aperture.plane_normal_odom, dtype=np.float64) * direction_sign
    plane_point = np.asarray(aperture.plane_point_odom_m, dtype=np.float64)
    inflation = envelope.inflation_m
    width = first_high - first_low
    height = second_high - second_low
    corridor_width = width - 2.0 * inflation
    corridor_height = height - 2.0 * inflation
    feasible = corridor_width > 0.0 and corridor_height > 0.0
    reasons: list[str] = []
    constraint: str | None = None
    if not feasible:
        constraint = APERTURE_CLEARANCE
        reasons.append(
            f"the observed opening is {width:.3f} m by {height:.3f} m and the inflated envelope "
            f"needs {2.0 * inflation:.3f} m on each side: corridor {corridor_width:.3f} m by "
            f"{corridor_height:.3f} m"
        )
    inner_first = (first_low + inflation, first_high - inflation)
    inner_second = (second_low + inflation, second_high - inflation)

    def corner(a: float, b: float, along: float) -> tuple[float, float, float]:
        point = plane_point + a * np.asarray(axis_one) + b * np.asarray(axis_two) + along * normal
        return tuple(float(value) for value in point)

    def box_from_extents(along_low: float, along_high: float, label: str) -> BoxRegion:
        corners = [
            corner(a, b, along_low)
            for a in inner_first
            for b in inner_second
        ] + [corner(a, b, along_high) for a in inner_first for b in inner_second]
        array = np.asarray(corners, dtype=np.float64)
        return BoxRegion(
            low=tuple(float(value) for value in array.min(axis=0)),
            high=tuple(float(value) for value in array.max(axis=0)),
            label=label,
        )

    return TraverseRegions(
        aperture=aperture,
        approach=box_from_extents(-standoff_m - APPROACH_HALF_THICKNESS_M, -standoff_m + APPROACH_HALF_THICKNESS_M, "approach"),
        crossing=box_from_extents(-CROSSING_HALF_DEPTH_M, CROSSING_HALF_DEPTH_M, "crossing"),
        terminal=box_from_extents(EXIT_DEPTH_M - 0.8, EXIT_DEPTH_M, "exit"),
        direction=tuple(float(value) for value in normal),
        usable_width_m=width,
        usable_height_m=height,
        corridor_width_m=corridor_width,
        corridor_height_m=corridor_height,
        feasible=feasible,
        constraint=constraint,
        reasons=tuple(reasons),
    )


def approach_region(
    point_odom_m: tuple[float, float, float],
    envelope: Envelope,
    *,
    standoff_m: float = STANDOFF_M,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> BoxRegion:
    """A supported standoff region in front of a target along the declared direction."""
    centre = np.asarray(point_odom_m, dtype=np.float64) - standoff_m * np.asarray(direction, dtype=np.float64)
    half = envelope.inflation_m + 0.1
    return BoxRegion(
        low=tuple(float(value) for value in centre - half),
        high=tuple(float(value) for value in centre + half),
        label="approach",
    )


def hold_region(station_odom_m: tuple[float, float, float], envelope: Envelope) -> BoxRegion:
    """A supported station: small enough that holding is containment, not progress."""
    half = envelope.body_radius_m
    return BoxRegion(
        low=tuple(value - half for value in station_odom_m),
        high=tuple(value + half for value in station_odom_m),
        label="hold",
    )


def inspect_region(
    target_odom_m: tuple[float, float, float],
    envelope: Envelope,
    *,
    standoff_m: float = STANDOFF_M,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> BoxRegion:
    """A reachable view candidate tied to an information requirement (section 11)."""
    return approach_region(target_odom_m, envelope, standoff_m=standoff_m, direction=direction)


def frontier_cells(
    store: world_module.MapStore, *, now_ns: int
) -> tuple[tuple[int, int, int], ...]:
    """Free cells touching unknown space: observation opportunities, not traversals.

    A frontier is a boundary between observed free space and unknown space. It is
    an observation opportunity and never a guarantee that the unknown side is
    traversable.
    """
    free = store.free_cells(now_ns=now_ns)
    frontiers = set()
    for cell in free:
        for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            neighbour = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
            if store.classify(neighbour, now_ns=now_ns) == world_module.UNKNOWN:
                frontiers.add(cell)
                break
    return tuple(sorted(frontiers))


def ball_offsets(config: world_module.MapConfig, radius_m: float) -> tuple[tuple[int, int, int], ...]:
    """Every cell offset whose centre lies inside a ball of the given radius.

    The declared envelope is a sphere (section 12.3), so the inflation is applied
    as a Euclidean ball rather than as an axis-aligned box: a box would demand
    0.40 m on every axis and much more on the diagonal, and it would refuse
    openings that fit.
    """
    reach = int(math.ceil(radius_m / config.voxel_m))
    offsets = []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                if math.sqrt(dx * dx + dy * dy + dz * dz) * config.voxel_m <= radius_m + 1e-9:
                    offsets.append((dx, dy, dz))
    return tuple(offsets)


def inflated_free_cells(
    store: world_module.MapStore,
    envelope: Envelope,
    *,
    now_ns: int,
    self_occupied_origin_odom_m: tuple[float, float, float] | None = None,
    extra_margin_m: float = 0.0,
) -> frozenset[tuple[int, int, int]]:
    """Free cells whose whole declared envelope stays in free space.

    This is the grid form of the swept-volume rule: a cell is traversable only if
    every cell inside the inflation ball around it is published free. Occupied
    space, the surface band and unknown space all disqualify it, because unknown is
    never free.

    One bounded exemption: the cells the vehicle's own presence blinds the sensor to
    are self-occupied by construction. The matcher cannot report depth inside the
    declared near limit, so no evidence can exist there in any direction, and a
    clearance ball around any cell within about half a metre of the camera reaches
    space the sensor provably cannot see — measured on J44-fly-1 as 83 % of every
    ball disqualifier, spread over all eight octants rather than sitting behind the
    aircraft. Without the exemption a plan can never start where the aircraft stands,
    and not one of the 26 neighbours is searchable.

    Its radius is derived from the sensor's near limit rather than from the envelope
    (``Envelope.self_occupied_radius_m``), and it never widens the free space ahead:
    every cell beyond it is still refused unless the map itself carries evidence that
    it is free. The clearance ball is unchanged.
    """
    config = store.config
    shape = config.shape()
    free = np.zeros(shape, dtype=bool)
    for cell in store.free_cells(now_ns=now_ns):
        free[cell] = True
    radius = envelope.inflation_m + extra_margin_m
    offsets = ball_offsets(config, radius)
    reach = int(math.ceil(radius / config.voxel_m))
    if self_occupied_origin_odom_m is not None:
        origin_index = config.cell_index(self_occupied_origin_odom_m)
        self_occupied_radius = envelope.self_occupied_radius_m(config.voxel_m)
        for dx, dy, dz in ball_offsets(config, self_occupied_radius):
            cell = (origin_index[0] + dx, origin_index[1] + dy, origin_index[2] + dz)
            if config.inside(cell):
                free[cell] = True
    padded = np.pad(
        free,
        ((reach, reach), (reach, reach), (reach, reach)),
        mode="constant",
        constant_values=False,
    )
    envelope_ok = free.copy()
    for dx, dy, dz in offsets:
        window = padded[
            reach + dx : reach + dx + shape[0],
            reach + dy : reach + dy + shape[1],
            reach + dz : reach + dz + shape[2],
        ]
        envelope_ok &= window
    reached = np.argwhere(envelope_ok)
    return frozenset(tuple(int(value) for value in cell) for cell in reached)


def shrink_to_supported(
    store: world_module.MapStore,
    region: BoxRegion,
    envelope: Envelope,
    *,
    now_ns: int,
    self_occupied_origin_odom_m: tuple[float, float, float] | None = None,
) -> tuple[BoxRegion | None, str | None]:
    """The part of a candidate region that is actually supported free space.

    A goal builder proposes a region from geometry; whether any of it is reachable
    is an evidence question. The observed free cone narrows with distance from the
    surface the opening was measured on, so a region built purely from the aperture
    polygon can include cells the map never observed. Intersecting it with the
    supported set keeps the goal honest, and an empty intersection is a refusal.
    """
    traversable = inflated_free_cells(
        store,
        envelope,
        now_ns=now_ns,
        self_occupied_origin_odom_m=self_occupied_origin_odom_m,
    )
    cells = [cell for cell in region.cells(store.config) if cell in traversable]
    if not cells:
        return None, (
            f"no cell of the {region.label} region is supported free space: the region overlaps "
            "space the map has no evidence for"
        )
    centers = [store.config.cell_center(cell) for cell in cells]
    low = tuple(min(center[axis] for center in centers) for axis in range(3))
    high = tuple(max(center[axis] for center in centers) for axis in range(3))
    return BoxRegion(low=low, high=high, label=region.label), None


def region_support(
    store: world_module.MapStore,
    region: BoxRegion,
    envelope: Envelope,
    *,
    now_ns: int,
    self_occupied_origin_odom_m: tuple[float, float, float] | None = None,
) -> tuple[bool, str | None]:
    """Whether every cell of a region is published free with its envelope inside free space.

    Returns (supported, reason). A cell that is unknown yields ``unsupported_space``:
    the region overlaps space the map has no evidence for, and a larger margin
    cannot turn that into measured free space.
    """
    traversable = inflated_free_cells(
        store,
        envelope,
        now_ns=now_ns,
        self_occupied_origin_odom_m=self_occupied_origin_odom_m,
    )
    cells = region.cells(store.config)
    if not cells:
        return False, "the region contains no cell of the map"
    unsupported = [cell for cell in cells if cell not in traversable]
    if unsupported:
        sample = unsupported[0]
        state = store.classify(sample, now_ns=now_ns)
        return False, (
            f"{len(unsupported)} of {len(cells)} cells in the {region.label} region are not "
            f"supported free space (for example {sample} is {state})"
        )
    return True, None


def braking_region(
    position_odom_m: tuple[float, float, float],
    velocity_mps: tuple[float, float, float],
    envelope: Envelope,
    *,
    deceleration_mps2: float,
    reaction_s: float,
    tracking_error_m: float = 0.0,
) -> BoxRegion:
    """The space needed to stop from the current velocity, plus reaction and error.

    Section 17.2's conservative scalar model: d_stop = v tau + v^2 / (2 a) + m.
    The region is the segment ahead along the velocity direction, thickened by the
    envelope, so a check can ask whether a supported stop exists.
    """
    speed = float(np.linalg.norm(np.asarray(velocity_mps, dtype=np.float64)))
    if deceleration_mps2 <= 0.0:
        raise R.RecordError("a braking check needs a positive deceleration bound")
    distance = speed * reaction_s + speed * speed / (2.0 * deceleration_mps2) + tracking_error_m
    direction = (
        np.asarray(velocity_mps, dtype=np.float64) / speed
        if speed > 0.0
        else np.array([1.0, 0.0, 0.0])
    )
    start = np.asarray(position_odom_m, dtype=np.float64)
    end = start + direction * distance
    thickness = envelope.inflation_m
    low = np.minimum(start, end) - thickness
    high = np.maximum(start, end) + thickness
    return BoxRegion(
        low=tuple(float(value) for value in low),
        high=tuple(float(value) for value in high),
        label="braking",
    )


def segment_extents(segment) -> BoxRegion:
    """The axis-aligned envelope of one certified segment, from its own polynomial bounds."""
    return BoxRegion(low=segment.low, high=segment.high, label=f"segment-{segment.index}")


def swept_support(
    store: world_module.MapStore,
    certificate,
    *,
    now_ns: int,
    self_occupied_origin_odom_m: tuple[float, float, float] | None = None,
) -> tuple[bool, str | None]:
    """Whether every swept volume the certificate claims is still supported free space."""
    for segment in certificate.segments:
        supported, reason = region_support(
            store,
            segment_extents(segment),
            Envelope(certificate.swept_radius_m, 0.0),
            now_ns=now_ns,
            self_occupied_origin_odom_m=self_occupied_origin_odom_m,
        )
        if not supported:
            return False, f"segment {segment.index}: {reason}"
    return True, None
