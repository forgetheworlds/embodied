"""The one short-horizon planner: A* route, convex corridors, certified quintics.

Specification section 13.1 makes this the only place motion is proposed. It
searches the inflated local free-space grid with A*, builds overlapping convex
corridors along the resulting path, and fits piecewise quintic minimum-jerk
trajectories inside them. Nothing here is a second avoidance policy: an obstacle
in the route changes *these* constraints, and the replan runs through this same
function.

Three properties are load-bearing:

* **Terminal regions, not a mandatory point.** A goal builder supplies acceptable
  regions; the search stops at any cell inside one.
* **Certification or silence.** A segment is published only if its complete
  continuous extent passes an independent bound check — extreme values of
  position, velocity, acceleration and jerk taken from the polynomial's own
  derivative roots, and a clearance check over the cells of its convex corridor.
  Sampling a few points is not certification, so it is not used.
* **Failure is named.** No route gives ``no_known_supported_route``; a solver or
  expansion bound gives ``computation_limit``. Neither is silently smoothed into
  a shorter, unverified path.

The minimum-jerk solve is the fixed-duration equality-constrained system of
section 13.1: piecewise quintics minimising integrated squared jerk, C4 at the
knots, clamped at both ends. When limits are violated the durations lengthen
inside a declared bound and the system is re-solved; if it still fails, the
planner refuses rather than publishing an uncertified curve.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from embodied.contracts import records as R
from embodied.memory import world as world_module
from embodied.navigation import geometry as geometry_module

NO_KNOWN_SUPPORTED_ROUTE = "no_known_supported_route"
COMPUTATION_LIMIT = "computation_limit"
UNSUPPORTED_SPACE = "unsupported_space"
START_STATE_MISMATCH = "start_state_mismatch"

POLYNOMIAL_ORDER = 6


@dataclass(frozen=True)
class PlanLimits:
    """The declared motion limits a certificate must respect."""

    v_max_mps: float = 0.5
    a_max_mps2: float = 1.0
    jerk_max_mps3: float = 4.0
    deceleration_mps2: float = 0.5
    reaction_s: float = 0.2

    def __post_init__(self) -> None:
        for name in ("v_max_mps", "a_max_mps2", "jerk_max_mps3", "deceleration_mps2"):
            value = getattr(self, name)
            if not isinstance(value, float) or value <= 0.0:
                raise R.RecordError(f"{name} must be a positive number")


@dataclass(frozen=True)
class PlanConfig:
    """Declared planning parameters: limits, envelope, tolerances and bounds."""

    limits: PlanLimits
    inflation_m: float
    start_state_tolerance_m: float
    setpoint_prefix_horizon_s: float
    setpoint_period_s: float
    max_expansions: int = 200_000
    max_duration_s: float = 30.0
    max_lengthening_rounds: int = 8
    duration_slack: float = 1.25
    settle_s: float = 1.0


@dataclass(frozen=True)
class Segment:
    """One certified quintic segment, with the bounds its own polynomials reach."""

    index: int
    t_start_s: float
    duration_s: float
    coefficients: tuple[tuple[float, ...], ...]
    low: tuple[float, float, float]
    high: tuple[float, float, float]
    max_speed_mps: float
    max_acceleration_mps2: float
    max_jerk_mps3: float
    distance_m: float

    @property
    def t_end_s(self) -> float:
        return self.t_start_s + self.duration_s

    def value(self, t_s: float) -> tuple[float, ...]:
        """Position at a time inside the segment."""
        return tuple(
            float(np.polyval(list(reversed(axis)), t_s - self.t_start_s)) for axis in self.coefficients
        )

    def derivative(self, t_s: float, order: int) -> tuple[float, ...]:
        values = []
        for axis in self.coefficients:
            coefficients = list(reversed(axis))
            for _ in range(order):
                coefficients = np.polyder(coefficients)
            values.append(float(np.polyval(coefficients, t_s - self.t_start_s)))
        return tuple(values)


@dataclass(frozen=True)
class TrajectoryCertificate:
    """The planner's record: identity, segments, envelope, dependencies and horizon.

    Defined here rather than in the shared contract because P03 is its first real
    writer (records.py:1196 names the planner and supervisor as the owners). Its
    promotion into ``contracts/records.py`` ``IMPLEMENTED_RECORDS`` is a serialized
    integrator change, requested at handoff.
    """

    certificate_id: str
    goal_id: str
    goal_revision: int
    mission_revision: int
    nav_epoch: str
    state_sequence: int
    snapshot_id: str
    map_revision: str
    anchor_id: str
    anchor_revision: str
    target_refs: tuple[str, ...]
    segments: tuple[Segment, ...]
    start_position_odom_m: tuple[float, float, float]
    start_velocity_odom_mps: tuple[float, float, float]
    start_tolerance_m: float
    t_start_s: float
    t_end_s: float
    swept_radius_m: float
    dependent_cells: tuple[tuple[int, int, int], ...]
    horizon_s: float
    limiting_reasons: tuple[str, ...]
    certified: bool
    constraints: tuple[str, ...]
    backup: Segment | None

    def sample(self, t_s: float) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Position, velocity and acceleration at an absolute time inside the plan."""
        for segment in self.segments:
            if segment.t_start_s <= t_s <= segment.t_end_s + 1e-12:
                return (
                    segment.value(t_s),
                    segment.derivative(t_s, 1),
                    segment.derivative(t_s, 2),
                )
        last = self.segments[-1]
        return (last.value(last.t_end_s), last.derivative(last.t_end_s, 1), last.derivative(last.t_end_s, 2))


def _quintic_coefficients(
    waypoints: np.ndarray, durations: np.ndarray, start_velocity: float, start_acceleration: float
) -> np.ndarray:
    """Solve the fixed-duration minimum-jerk system for one axis.

    Unknowns: six coefficients per segment. Constraints: the segment endpoints
    interpolate the waypoints, position/velocity/acceleration/jerk/snap are
    continuous at every interior knot (the natural conditions of the minimum-jerk
    problem), and the ends are clamped to the declared start state and to rest.
    """
    segments = durations.size
    unknowns = POLYNOMIAL_ORDER * segments
    matrix = np.zeros((unknowns, unknowns), dtype=np.float64)
    rhs = np.zeros(unknowns, dtype=np.float64)
    row = 0

    def coefficient_index(segment: int, power: int) -> int:
        return segment * POLYNOMIAL_ORDER + power

    def derivative_row(segment: int, order: int, at_end: bool) -> np.ndarray:
        coefficients = np.zeros(unknowns, dtype=np.float64)
        offset = durations[segment] if at_end else 0.0
        for power in range(order, POLYNOMIAL_ORDER):
            factor = 1.0
            for step in range(order):
                factor *= power - step
            coefficients[coefficient_index(segment, power)] = factor * offset ** (power - order)
        return coefficients

    for segment in range(segments):
        matrix[row, coefficient_index(segment, 0)] = 1.0
        rhs[row] = waypoints[segment]
        row += 1
        row_vector = derivative_row(segment, 0, True)
        matrix[row] = row_vector
        rhs[row] = waypoints[segment + 1]
        row += 1
    for segment in range(segments - 1):
        for order in (1, 2, 3, 4):
            matrix[row] = derivative_row(segment, order, True) - derivative_row(segment + 1, order, False)
            rhs[row] = 0.0
            row += 1
    matrix[row] = derivative_row(0, 1, False)
    rhs[row] = start_velocity
    row += 1
    matrix[row] = derivative_row(0, 2, False)
    rhs[row] = start_acceleration
    row += 1
    matrix[row] = derivative_row(segments - 1, 1, True)
    rhs[row] = 0.0
    row += 1
    matrix[row] = derivative_row(segments - 1, 2, True)
    rhs[row] = 0.0
    row += 1
    if row != unknowns:
        raise R.RecordError(f"minimum-jerk system is {row} by {unknowns}: it is not square")
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise R.RecordError(f"minimum-jerk system is singular: {error}") from None
    return solution.reshape(segments, POLYNOMIAL_ORDER)


def _norm_max(axis_coefficients: np.ndarray, duration_s: float, order: int) -> float:
    """The exact maximum norm of the order-th derivative on [0, T].

    The squared norm is stationary where the derivative of the norm vanishes, which
    is where the order-th derivative is orthogonal to the next one. That inner
    product is a low-degree polynomial in time, so its roots give the candidate
    extremes exactly; the endpoints are added. Nothing here samples the curve.
    """
    derivatives = []
    following = []
    for axis in axis_coefficients:
        polynomial = np.poly1d(list(reversed(axis)))
        derivatives.append(polynomial.deriv(order))
        following.append(polynomial.deriv(order + 1))
    inner = np.poly1d([0.0])
    for first, second in zip(derivatives, following):
        inner = inner + first * second
    candidates = [0.0, float(duration_s)]
    if inner.order >= 1:
        for root in np.atleast_1d(inner.roots):
            if abs(root.imag) < 1e-9 and 0.0 < root.real < duration_s:
                candidates.append(float(root.real))
    return max(
        float(
            sum(float(derivative(time)) ** 2 for derivative in derivatives) ** 0.5
        )
        for time in candidates
    )


def _initial_durations(distances: np.ndarray, limits: PlanLimits) -> np.ndarray:
    """Opening durations: a fraction of the speed limit per segment, with a floor.

    The minimum-jerk system is C4 across knots, so a segment does not have to
    accelerate from and brake to rest within its own duration; the conservative
    clamped-quintic estimate would make every short segment take over a second and
    the whole route exceed its bound. The durations are a starting point only: each
    solve is then certified exactly, and a violating segment lengthens and re-solves.
    """
    return np.maximum(distances / (0.8 * limits.v_max_mps), 0.2)


def _astar(
    traversable: frozenset[tuple[int, int, int]],
    start_cell: tuple[int, int, int],
    goal_cells: frozenset[tuple[int, int, int]],
    config: world_module.MapConfig,
    *,
    max_expansions: int,
) -> list[tuple[int, int, int]] | None:
    """A* over the inflated free-space grid, with no corner cutting."""
    import heapq

    if start_cell not in traversable:
        return None
    goal_low = np.array([min(cell[axis] for cell in goal_cells) - 0.5 for axis in range(3)])
    goal_high = np.array([max(cell[axis] for cell in goal_cells) + 0.5 for axis in range(3)])

    def heuristic(cell: tuple[int, int, int]) -> float:
        position = np.asarray(cell, dtype=np.float64)
        return float(np.linalg.norm(np.maximum(np.maximum(goal_low - position, position - goal_high), 0.0)))

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) != (0, 0, 0):
                    offsets.append((dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz)))
    frontier = [(heuristic(start_cell), 0.0, start_cell)]
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    cost = {start_cell: 0.0}
    expansions = 0
    while frontier:
        _estimate, travelled, current = heapq.heappop(frontier)
        if current in goal_cells:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        expansions += 1
        if expansions > max_expansions:
            return None
        for dx, dy, dz, step in offsets:
            neighbour = (current[0] + dx, current[1] + dy, current[2] + dz)
            if neighbour not in traversable:
                continue
            if dx and dy and (current[0] + dx, current[1], current[2]) not in traversable:
                continue
            if dx and dz and (current[0] + dx, current[1], current[2] + dz) not in traversable:
                continue
            if dy and dz and (current[0], current[1] + dy, current[2] + dz) not in traversable:
                continue
            candidate = travelled + step
            if candidate < cost.get(neighbour, float("inf")):
                cost[neighbour] = candidate
                came_from[neighbour] = current
                heapq.heappush(frontier, (candidate + heuristic(neighbour), candidate, neighbour))
    return None


MAX_MERGED_CELLS = 4


def _straight_run_is_supported(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    traversable: frozenset[tuple[int, int, int]],
    grid: world_module.MapConfig,
    envelope: geometry_module.Envelope,
) -> bool:
    """Whether a straight run between two cell centres stays in supported free space."""
    start = np.asarray(grid.cell_center(first), dtype=np.float64)
    end = np.asarray(grid.cell_center(second), dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    steps = max(int(math.ceil(distance / (grid.voxel_m / 4.0))), 1)
    radius = envelope.inflation_m + grid.voxel_m * 0.5
    offsets = geometry_module.ball_offsets(grid, radius)
    for step in range(steps + 1):
        point = start + (end - start) * step / steps
        index = grid.cell_index(tuple(float(value) for value in point))
        for offset in offsets:
            cell = (index[0] + offset[0], index[1] + offset[1], index[2] + offset[2])
            if grid.inside(cell) and cell not in traversable:
                return False
    return True


def _shortcut(
    path: list[tuple[int, int, int]],
    traversable: frozenset[tuple[int, int, int]],
    grid: world_module.MapConfig,
    envelope: geometry_module.Envelope,
) -> list[tuple[int, int, int]]:
    """Replace cell-by-cell staircases with supported straight runs.

    A grid search returns a staircase, and a spline fitted to a staircase cuts its
    corners — into space the search never certified. Pulling the path straight
    wherever a straight run is itself supported removes the staircases, so the
    fitted curve has little to cut and every waypoint it passes is supported.
    """
    if len(path) <= 2:
        return list(path)
    waypoints = [path[0]]
    index = 0
    while index < len(path) - 1:
        target = len(path) - 1
        while target > index + 1 and not _straight_run_is_supported(
            path[index], path[target], traversable, grid, envelope
        ):
            target -= 1
        waypoints.append(path[target])
        index = target
    return waypoints


def _merge_collinear(
    path: list[tuple[int, int, int]], max_cells: int = MAX_MERGED_CELLS
) -> list[tuple[int, int, int]]:
    """Keep every path cell as a knot.

    Merging collinear cells would cut the number of unknowns, but the minimum-jerk
    spline is a global C4 curve: with widely spaced knots it bows away from the
    cell path between them, and in a corridor a few cells wide that bow leaves the
    supported free space. Keeping each cell as a knot pins the curve near the path
    the search certified, and the system stays small (six unknowns per cell per
    axis) for the routes this stage plans.
    """
    merged = [path[0]]
    direction = None
    run = 0
    for previous, current in zip(path, path[1:]):
        step = (current[0] - previous[0], current[1] - previous[1], current[2] - previous[2])
        if direction is not None and (step != direction or run >= max_cells):
            merged.append(previous)
            run = 0
        direction = step
        run += 1
    merged.append(path[-1])
    return merged


def plan(
    store: world_module.MapStore,
    envelope: geometry_module.Envelope,
    config: PlanConfig,
    goal_region: geometry_module.BoxRegion,
    *,
    navigation_state: R.NavigationState,
    nav_epoch: str,
    goal_id: str,
    goal_revision: int,
    mission_revision: int,
    target_refs: tuple[str, ...],
    anchor_id: str,
    anchor_revision: str,
    snapshot_id: str,
    now_ns: int,
    start_position_odom_m: tuple[float, float, float] | None = None,
    start_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> "TrajectoryCertificate | PlanRefusal":
    """Plan one certified trajectory into a goal region, or refuse with a named reason."""
    start = start_position_odom_m or navigation_state.pose.position_m
    grid = store.config
    # The set is inflated by the envelope plus a quarter voxel: a certificate walks
    # the curve at a spacing whose displacement bound is that quarter voxel, so a
    # curve that stays inside this set has its whole swept tube inside free space.
    traversable = geometry_module.inflated_free_cells(
        store,
        envelope,
        now_ns=now_ns,
        self_occupied_origin_odom_m=start,
        extra_margin_m=grid.voxel_m / 4.0,
    )
    searchable = geometry_module.inflated_free_cells(
        store,
        envelope,
        now_ns=now_ns,
        self_occupied_origin_odom_m=start,
        extra_margin_m=grid.voxel_m * 0.5,
    )
    start_cell = grid.cell_index(start)
    goal_cells = frozenset(cell for cell in goal_region.cells(grid) if cell in searchable)
    if start_cell not in searchable:
        return PlanRefusal(
            NO_KNOWN_SUPPORTED_ROUTE,
            f"the start cell {start_cell} is not supported free space",
            support="uncertain",
        )
    if not goal_cells:
        return PlanRefusal(
            NO_KNOWN_SUPPORTED_ROUTE,
            f"no cell of the {goal_region.label} region is supported free space",
            support="uncertain",
        )
    path = _astar(searchable, start_cell, goal_cells, grid, max_expansions=config.max_expansions)
    if path is not None:
        path = _merge_collinear(_shortcut(path, searchable, grid, envelope))
    if path is None:
        return PlanRefusal(
            NO_KNOWN_SUPPORTED_ROUTE,
            "no route exists in the known free map; an unobserved connection may still exist, so "
            "this is uncertain rather than physically impossible",
            support="uncertain",
        )
    waypoint_cells = _merge_collinear(path)
    waypoints = np.asarray([grid.cell_center(cell) for cell in waypoint_cells], dtype=np.float64)
    waypoints[0] = np.asarray(start, dtype=np.float64)
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    durations = _initial_durations(distances, config.limits)
    if float(durations.sum()) > config.max_duration_s:
        return PlanRefusal(
            COMPUTATION_LIMIT,
            f"the route needs {durations.sum():.1f} s, beyond the declared {config.max_duration_s:.1f} s bound",
            support="uncertain",
        )
    coefficients = None
    segments: tuple[Segment, ...] | None = None
    attempts = 0
    while True:
        attempts += 1
        per_axis = []
        try:
            for axis in range(3):
                per_axis.append(
                    _quintic_coefficients(
                        waypoints[:, axis],
                        durations,
                        float(start_velocity_mps[axis]),
                        0.0,
                    )
                )
        except R.RecordError as error:
            return PlanRefusal(COMPUTATION_LIMIT, f"the quintic solve failed: {error}", support="uncertain")
        coefficients = np.stack(per_axis, axis=1)  # (segments, axis, power)
        candidate, violation = _certify_segments(
            coefficients, durations, distances, traversable, grid, config, envelope, start
        )
        if violation is None:
            segments = candidate
            break
        lengthened = durations * config.duration_slack
        if attempts > config.max_lengthening_rounds or (
            float(lengthened.sum()) > config.max_duration_s
        ):
            # The C4 minimum-jerk curve is a global spline: it can bow away from the
            # certified cell path between knots. Fall back to clamped per-segment
            # quintics through the same waypoints, which keep position, velocity and
            # acceleration continuity and cannot leave their own chord's box, and
            # certify those instead. A curve that still cannot be certified is
            # refused: an uncertified curve is never published.
            fallback = _clamped_segments(
                waypoints, durations, distances, traversable, grid, config, envelope, start, attempts
            )
            if isinstance(fallback, PlanRefusal):
                return PlanRefusal(
                    fallback.reason,
                    f"{violation}; {fallback.detail}",
                    support="uncertain",
                )
            segments = fallback
            coefficients = None
            break
        durations = lengthened
    if segments is None:
        return PlanRefusal(COMPUTATION_LIMIT, "the solver produced no segments", support="uncertain")
    t_start = now_ns / 1e9
    dependent = sorted({cell for segment in segments for cell in _cells_of(segment, grid)})
    backup = _backup_segment(start, start_velocity_mps, config, traversable, grid)
    if backup is None:
        return PlanRefusal(
            NO_KNOWN_SUPPORTED_ROUTE,
            "no checked stopping continuation exists from the start state",
            support="uncertain",
        )
    return TrajectoryCertificate(
        certificate_id=f"cert-{goal_id}-{store.revision}",
        goal_id=goal_id,
        goal_revision=goal_revision,
        mission_revision=mission_revision,
        nav_epoch=nav_epoch,
        state_sequence=navigation_state.state_sequence,
        snapshot_id=snapshot_id,
        map_revision=store.revision,
        anchor_id=anchor_id,
        anchor_revision=anchor_revision,
        target_refs=target_refs,
        segments=tuple(segments),
        start_position_odom_m=tuple(float(value) for value in waypoints[0]),
        start_velocity_odom_mps=tuple(float(value) for value in start_velocity_mps),
        start_tolerance_m=config.start_state_tolerance_m,
        t_start_s=t_start,
        t_end_s=t_start + float(durations.sum()),
        swept_radius_m=envelope.swept_radius_m,
        dependent_cells=tuple(dependent),
        horizon_s=float(durations.sum()),
        limiting_reasons=(
            "certified under the inflated free-space map and the declared motion limits",
            "curve_form: "
            + ("minimum_jerk_C4" if coefficients is not None else "clamped_quintic_fallback"),
        ),
        certified=True,
        constraints=("inflated_free_space", "complete_segment_certification"),
        backup=backup,
    )


@dataclass(frozen=True)
class PlanRefusal:
    """A planner refusal, mirroring the grounding refusal's shape.

    ``support`` is always ``uncertain``: failing to find a route in the known map,
    or hitting a computation bound, is missing evidence, not proof that no route
    exists (section 12.1).
    """

    reason: str
    detail: str
    support: str = "uncertain"


def violation_segment(violation: str) -> int:
    """The segment index named in a certification violation message."""
    for token in violation.split():
        if token.startswith("segment-"):
            return int(token.split("-")[1])
    return 0


def _clamped_segments(
    waypoints: np.ndarray,
    durations: np.ndarray,
    distances: np.ndarray,
    traversable: frozenset[tuple[int, int, int]],
    grid: world_module.MapConfig,
    config: PlanConfig,
    envelope: geometry_module.Envelope,
    start: tuple[float, float, float],
    attempts: int,
) -> "tuple[Segment, ...] | PlanRefusal":
    """Per-segment quintics that start and end at rest, with no overshoot.

    Each segment is the monotone quintic ``10u^3 - 15u^4 + 6u^5`` between its two
    waypoints, so its position stays inside the box its chord spans: the clearance
    check then needs only the chord, which the shortcut pass already certified.
    Position, velocity and acceleration are continuous because every knot is a
    stop. This is not the jerk-minimising curve; it is the certified one.
    """
    velocity_bound = 1.875 * distances / config.limits.v_max_mps
    acceleration_bound = np.sqrt(5.7735 * distances / config.limits.a_max_mps2)
    jerk_bound = (60.0 * distances / config.limits.jerk_max_mps3) ** (1.0 / 3.0)
    segment_durations = np.maximum(
        np.maximum(velocity_bound, np.maximum(acceleration_bound, jerk_bound)), 0.1
    )
    if float(segment_durations.sum()) > config.max_duration_s:
        return PlanRefusal(
            COMPUTATION_LIMIT,
            "the clamped fallback needs "
            f"{segment_durations.sum():.1f} s, beyond the declared {config.max_duration_s:.1f} s bound",
            support="uncertain",
        )
    coefficients = np.zeros((distances.size, 3, POLYNOMIAL_ORDER), dtype=np.float64)
    for index in range(distances.size):
        duration = float(segment_durations[index])
        delta = waypoints[index + 1] - waypoints[index]
        for axis in range(3):
            coefficients[index, axis] = (
                float(waypoints[index][axis]),
                0.0,
                0.0,
                10.0 * delta[axis] / duration**3,
                -15.0 * delta[axis] / duration**4,
                6.0 * delta[axis] / duration**5,
            )
    segments, violation = _certify_segments(
        coefficients, segment_durations, distances, traversable, grid, config, envelope, start
    )
    if violation is not None:
        # A curve whose sweep leaves supported free space is not a computation limit:
        # the map does not support that motion, and publishing a shorter, unverified
        # path instead would be exactly the substitution the specification forbids.
        return PlanRefusal(
            UNSUPPORTED_SPACE,
            f"neither curve form is certifiable after {attempts} lengthening rounds: {violation}",
            support="uncertain",
        )
    return segments


def _sample_interval(grid: world_module.MapConfig, config: PlanConfig) -> float:
    """A sampling interval whose displacement bound is a quarter voxel."""
    return min(SAMPLE_INTERVAL_MAX_S, grid.voxel_m / (4.0 * config.limits.v_max_mps))


def _cells_of(segment: Segment, config: world_module.MapConfig) -> tuple[tuple[int, int, int], ...]:
    region = geometry_module.BoxRegion(low=segment.low, high=segment.high, label="segment")
    return region.cells(config)


SAMPLES_PER_SEGMENT_MIN = 8
SAMPLE_INTERVAL_MAX_S = 0.05
# The route is searched inside a slightly larger inflation than the certificate
# requires, so the fitted curve - which can bow a few centimetres away from the
# certified cell path between knots - still cannot leave supported free space.
# Authored margin, not a measured one: two voxels is the deviation the fitting
# observed on this stage's fixture.
SEARCH_MARGIN_M = 0.10


def _certify_segments(
    coefficients: np.ndarray,
    durations: np.ndarray,
    distances: np.ndarray,
    traversable: frozenset[tuple[int, int, int]],
    grid: world_module.MapConfig,
    config: PlanConfig,
    envelope: geometry_module.Envelope,
    start: tuple[float, float, float],
) -> tuple[tuple[Segment, ...] | None, str | None]:
    """Certify every segment: exact derivative bounds plus a bounded sweep check.

    The limit check reads the extremes of the segment's own derivatives, so it can
    never miss a violation between samples. The clearance check walks the segment
    at a spacing whose displacement bound is a quarter voxel (``v_max * dt <=
    voxel/4``) and requires each sampled cell's envelope — inflated with that same
    quarter voxel of slack — to be free. The curve between two samples stays within
    ``v_max * dt / 2`` of them, so the whole swept tube is covered rather than
    merely sampled.
    """
    segments = []
    time = 0.0
    for index in range(durations.size):
        bounds = []
        extremes = []
        for axis in range(3):
            polynomial = np.poly1d(list(reversed(coefficients[index, axis])))
            positions = [float(polynomial(0.0)), float(polynomial(durations[index]))]
            derivative = polynomial.deriv()
            if derivative.order >= 1:
                for root in np.atleast_1d(derivative.roots):
                    if abs(root.imag) < 1e-9 and 0.0 < root.real < durations[index]:
                        positions.append(float(polynomial(root.real)))
            bounds.append((min(positions), max(positions)))
        for order in (1, 2, 3):
            values = []
            for axis in range(3):
                derivative = np.poly1d(list(reversed(coefficients[index, axis]))).deriv(order)
                axis_values = [float(derivative(0.0)), float(derivative(durations[index]))]
                if derivative.order >= 1:
                    for root in np.atleast_1d(derivative.roots):
                        if abs(root.imag) < 1e-9 and 0.0 < root.real < durations[index]:
                            axis_values.append(float(derivative(root.real)))
                values.extend(axis_values)
            extremes.append(values)
        low = tuple(bound[0] for bound in bounds)
        high = tuple(bound[1] for bound in bounds)
        # The limits are on the *norm* of the motion, not on each axis: a certificate
        # whose every axis stayed under v_max could still fly at sqrt(3) v_max. The
        # norm's extremes are found from the roots of its own derivative (v . a for
        # speed, a . j for acceleration, j . p'''' for jerk) rather than by sampling.
        max_speed = _norm_max(coefficients[index], durations[index], 1)
        max_acceleration = _norm_max(coefficients[index], durations[index], 2)
        max_jerk = _norm_max(coefficients[index], durations[index], 3)
        segment = Segment(
            index=index,
            t_start_s=time,
            duration_s=float(durations[index]),
            coefficients=tuple(tuple(float(value) for value in coefficients[index, axis]) for axis in range(3)),
            low=low,
            high=high,
            max_speed_mps=max_speed,
            max_acceleration_mps2=max_acceleration,
            max_jerk_mps3=max_jerk,
            distance_m=float(distances[index]),
        )
        if max_speed > config.limits.v_max_mps + 1e-9:
            return None, f"segment-{index} reaches {max_speed:.3f} m/s over the declared v_max"
        if max_acceleration > config.limits.a_max_mps2 + 1e-9:
            return None, f"segment-{index} reaches {max_acceleration:.3f} m/s^2 over the declared a_max"
        if max_jerk > config.limits.jerk_max_mps3 + 1e-9:
            return None, f"segment-{index} reaches {max_jerk:.3f} m/s^3 over the declared jerk limit"
        slack = grid.voxel_m / 4.0
        samples = max(
            int(math.ceil(segment.duration_s / _sample_interval(grid, config))),
            SAMPLES_PER_SEGMENT_MIN,
        ) + 1
        unsupported = None
        self_occupied_radius = envelope.inflation_m + slack
        for step in range(samples):
            time = segment.t_start_s + segment.duration_s * step / samples
            centre = segment.value(time)
            # The part of the curve still inside the envelope the aircraft already
            # occupies at its start is not a claim about new space: the aircraft is
            # there. Everything beyond that ball must be supported free space.
            if float(np.linalg.norm(np.asarray(centre) - np.asarray(start))) <= self_occupied_radius:
                continue
            here = grid.cell_index(tuple(float(value) for value in centre))
            for offset in geometry_module.ball_offsets(grid, self_occupied_radius):
                cell = (here[0] + offset[0], here[1] + offset[1], here[2] + offset[2])
                if grid.inside(cell) and cell not in traversable:
                    unsupported = cell
                    break
            if unsupported is not None:
                break
        if unsupported is not None:
            return None, (
                f"segment-{index} sweeps cell {unsupported}, whose envelope is not supported "
                "free space"
            )
        segments.append(segment)
        time += float(durations[index])
    return tuple(segments), None


def _backup_segment(
    start: tuple[float, float, float],
    velocity_mps: tuple[float, float, float],
    config: PlanConfig,
    traversable: frozenset[tuple[int, int, int]],
    grid: world_module.MapConfig,
) -> Segment | None:
    """The checked stopping continuation from the plan's start state.

    With the aircraft already at rest this is a stationary segment: a supported
    stop that needs no distance. With velocity it is the braking distance of the
    declared deceleration bound, and it is certified by the same corridor check,
    so a backup is only claimed when the space to stop is supported.
    """
    speed = float(np.linalg.norm(np.asarray(velocity_mps, dtype=np.float64)))
    if speed <= 1e-9:
        segment = Segment(
            index=-1,
            t_start_s=0.0,
            duration_s=config.settle_s,
            coefficients=tuple((float(value), 0.0, 0.0, 0.0, 0.0, 0.0) for value in start),
            low=start,
            high=start,
            max_speed_mps=0.0,
            max_acceleration_mps2=0.0,
            max_jerk_mps3=0.0,
            distance_m=0.0,
        )
        missing = [cell for cell in _cells_of(segment, grid) if cell not in traversable]
        return None if missing else segment
    distance = speed * config.limits.reaction_s + speed * speed / (2.0 * config.limits.deceleration_mps2)
    direction = np.asarray(velocity_mps, dtype=np.float64) / speed
    end = np.asarray(start, dtype=np.float64) + direction * distance
    duration = max(distance / max(speed, 1e-6), 0.1)
    low = tuple(float(value) for value in np.minimum(np.asarray(start), end))
    high = tuple(float(value) for value in np.maximum(np.asarray(start), end))
    segment = Segment(
        index=-1,
        t_start_s=0.0,
        duration_s=duration,
        coefficients=tuple((float(value), 0.0, 0.0, 0.0, 0.0, 0.0) for value in start),
        low=low,
        high=high,
        max_speed_mps=speed,
        max_acceleration_mps2=config.limits.deceleration_mps2,
        max_jerk_mps3=0.0,
        distance_m=distance,
    )
    missing = [cell for cell in _cells_of(segment, grid) if cell not in traversable]
    return None if missing else segment


def replan_is_current(
    certificate: TrajectoryCertificate,
    current_position_odom_m: tuple[float, float, float],
) -> _Refusal | None:
    """Whether a planned start state still matches where the aircraft actually is.

    A late plan whose start moved beyond its declared tolerance is rejected rather
    than adopted with a jump (section 13.2).
    """
    displacement = float(
        np.linalg.norm(
            np.asarray(current_position_odom_m, dtype=np.float64)
            - np.asarray(certificate.start_position_odom_m, dtype=np.float64)
        )
    )
    if displacement > certificate.start_tolerance_m:
        return PlanRefusal(
            START_STATE_MISMATCH,
            f"the intended start moved {displacement:.3f} m from the planned start, beyond the "
            f"declared {certificate.start_tolerance_m:.3f} m tolerance",
            support="uncertain",
        )
    return None
