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


class BudgetExhausted:
    """The route search hit its own expansion budget.

    Distinct from ``None``, which means the frontier emptied without reaching a
    goal cell — a fact about the known map. Section 13.1 requires the two to be
    reported differently: a computation limit is missing computation, not
    evidence that no route exists. Until this existed they were the same
    ``None``, and ``plan`` reported both as ``no_known_supported_route``.
    """


BUDGET_EXHAUSTED = BudgetExhausted()

# The margin, in voxels, that a cell must hold beyond the declared envelope to
# count as free space for this planner. One value, one home (R2). It answers
# both questions this module asks of a cell — the radius of the occupancy set
# a route walks and every knot is chosen from (a knot is a place the vehicle
# may stop and hold), and the slack inside the clearance ball the certificate
# sweeps against the map's published free cells.
#
# The quarter voxel is the certificate's own. It walks the curve at a spacing
# whose displacement bound is that quarter voxel, and the sweep demands every
# cell inside a ball of the declared envelope plus that slack to be published
# free. The sweep's membership is the published free set, not the occupancy
# set: the ball already carries the envelope once, and membership in the
# occupancy set would inflate a second time — 0.95 m of raw free space against
# the declared 0.475 m clearance on this map. That second width had no
# derivation, and it measured refusing all eight offered frontiers at the
# certification axis on the doorway map while the same occupancy set admitted
# their routes (2026-10-03, night/motion).
#
# R23 (2026-10-02): the goal search used a HALF voxel, which had no stated
# justification and was stricter than the clearance the certificate then
# verifies — 0.500 m against 0.475 m on the declared 0.45 m envelope with a
# 0.1 m voxel. That 2.5 cm cost the map every goal: at the half voxel a scene
# held no evidence-based searchable cell at all, where the certificate's own
# margin left 113, and because the vantage search returns the FIRST candidate
# that passes, the first false positive became the goal, admission refused that
# exact region, and the frontier was then marked blocked and never offered
# again. The certificate keeps the final say: cells this set contains that
# ``_certify_segments`` then refuses remain refused.
#
# The runtime's navigability test imports this rather than restating it: a test
# that answers "is there a point this planner could admit" must ask with this
# planner's own margin. Before the unification it used a quarter voxel against
# this file's half, clearing 38,016 cells against 29,240 on one map — and that
# difference was never the clearance, only the disagreement.
CERTIFICATE_MARGIN_VOXELS = 0.25

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
            float(np.polyval(list(reversed(axis)), t_s - self.t_start_s))
            for axis in self.coefficients
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

    def sample(
        self, t_s: float
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Position, velocity and acceleration at an absolute time inside the plan.

        The segments carry plan-relative times, but the certificate's own window
        is absolute — ``t_start_s`` is the epoch second the plan was certified at
        — and every caller samples in that domain: the executor builds the
        published prefix from ``now_ns``, the validator walks ``t_start_s`` to
        ``t_end_s``, and the runtime samples at ``time.monotonic()`` against a
        plan taken from ``monotonic_ns``. The conversion therefore happens here,
        once. Before it did, an absolute ``t_s`` matched no segment and the
        sample clamped to the curve's end: the first published setpoint would
        have been the goal point itself rather than the certified prefix.
        """
        relative = min(max(t_s - self.t_start_s, 0.0), self.horizon_s)
        for segment in self.segments:
            if segment.t_start_s <= relative <= segment.t_end_s + 1e-12:
                return (
                    segment.value(relative),
                    segment.derivative(relative, 1),
                    segment.derivative(relative, 2),
                )
        last = self.segments[-1]
        return (
            last.value(last.t_end_s),
            last.derivative(last.t_end_s, 1),
            last.derivative(last.t_end_s, 2),
        )


def _quintic_coefficients(
    waypoints: np.ndarray,
    durations: np.ndarray,
    start_velocity: float,
    start_acceleration: float,
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
            coefficients[coefficient_index(segment, power)] = factor * offset ** (
                power - order
            )
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
            matrix[row] = derivative_row(segment, order, True) - derivative_row(
                segment + 1, order, False
            )
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
        raise R.RecordError(
            f"minimum-jerk system is {row} by {unknowns}: it is not square"
        )
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
        float(sum(float(derivative(time)) ** 2 for derivative in derivatives) ** 0.5)
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


# The adjacency a path may use, written down once. `_neighbours`, `reachable_from`
# and `_astar` all step by this, so "a cell is reachable" and "A* reaches it"
# cannot answer differently. A vantage the mission accepts and admission then
# refuses is this project's most expensive recurring defect, and it is exactly
# that disagreement.
_NEIGHBOUR_STEPS: tuple[tuple[int, int, int, float], ...] = tuple(
    (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _neighbours(
    cell: tuple[int, int, int], traversable: frozenset[tuple[int, int, int]]
):
    """Yield ``(neighbour, step_cost)`` for the cells a path may step to.

    No corner cutting: a diagonal step also requires the straight steps sharing
    its edges to be traversable, so a path never slips through the gap between
    two occupied cells.
    """
    x, y, z = cell
    for dx, dy, dz, step in _NEIGHBOUR_STEPS:
        neighbour = (x + dx, y + dy, z + dz)
        if neighbour not in traversable:
            continue
        if dx and dy and (x + dx, y, z) not in traversable:
            continue
        if dx and dz and (x + dx, y, z + dz) not in traversable:
            continue
        if dy and dz and (x, y + dy, z + dz) not in traversable:
            continue
        yield neighbour, step


def reachable_from(
    traversable: frozenset[tuple[int, int, int]], start_cell: tuple[int, int, int]
) -> frozenset[tuple[int, int, int]]:
    """The cells a path from ``start_cell`` can reach, by the rule A* walks.

    This is admission's actual question. ``plan`` runs A* from the aircraft's own
    cell and refuses with ``no_known_supported_route`` when no path connects it
    to the goal region — so a region that holds free cells no route reaches is a
    goal admission will refuse. Measured on ``J48-fly-1``: the vantage walk
    accepted a target on membership alone, admission refused it, and the frontier
    was marked blocked and never offered again.

    The empty set is the honest answer when the start is not traversable: the
    aircraft cannot take one step, and saying so is what lets the caller decline
    the frontier rather than publish a hover as progress.
    """
    if start_cell not in traversable:
        return frozenset()
    seen = {start_cell}
    stack = [start_cell]
    while stack:
        current = stack.pop()
        for neighbour, _step in _neighbours(current, traversable):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return frozenset(seen)


def _astar(
    traversable: frozenset[tuple[int, int, int]],
    start_cell: tuple[int, int, int],
    goal_cells: frozenset[tuple[int, int, int]],
    config: world_module.MapConfig,
    *,
    max_expansions: int,
) -> "list[tuple[int, int, int]] | BudgetExhausted | None":
    """A* over the inflated free-space grid, with no corner cutting.

    Three outcomes, and they are three different facts: the path; ``None`` when
    the frontier emptied without reaching a goal cell, which is what the known
    map holds; and ``BUDGET_EXHAUSTED`` when the search was cut off by its own
    budget, which is not a statement about the map at all.
    """
    import heapq

    if start_cell not in traversable:
        return None
    goal_low = np.array(
        [min(cell[axis] for cell in goal_cells) - 0.5 for axis in range(3)]
    )
    goal_high = np.array(
        [max(cell[axis] for cell in goal_cells) + 0.5 for axis in range(3)]
    )

    def heuristic(cell: tuple[int, int, int]) -> float:
        position = np.asarray(cell, dtype=np.float64)
        return float(
            np.linalg.norm(
                np.maximum(np.maximum(goal_low - position, position - goal_high), 0.0)
            )
        )

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
            return BUDGET_EXHAUSTED
        for neighbour, step in _neighbours(current, traversable):
            candidate = travelled + step
            if candidate < cost.get(neighbour, float("inf")):
                cost[neighbour] = candidate
                came_from[neighbour] = current
                heapq.heappush(
                    frontier, (candidate + heuristic(neighbour), candidate, neighbour)
                )
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
    # The membership set is the map's published free cells, so this ball IS a
    # clearance test — and its radius stays inflation + half a voxel, above the
    # certificate's own ball (inflation + quarter voxel): a run the shortcut
    # keeps is certificate-clean at every sampled point. The radius itself is
    # declared and not in question; R23 licenses unifying the goal search with
    # the clearance the certificate verifies, and this run check now answers the
    # certificate's question, neither more nor less.
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
        step = (
            current[0] - previous[0],
            current[1] - previous[1],
            current[2] - previous[2],
        )
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
    # Two sets, each answering one question, both carved from the same evidence
    # (R23: never two answers to one question). ``free_space`` is the occupancy
    # set — cells whose own declared envelope lies in published free space — and
    # it is what the route search walks and every knot is chosen from, because a
    # knot is a place the curve may stop and the vehicle may hold. The sweep test
    # is a different question: the curve's clearance ball already carries the
    # declared envelope plus its quarter-voxel slack, so membership there is
    # tested against ``published_free`` directly. Testing it against
    # ``free_space`` instead would inflate a second time — a 0.95 m corridor of
    # raw free space against the declared 0.475 m clearance on this map — and
    # that second width has no derivation anywhere; it was measured refusing all
    # eight offered frontiers at the certification axis on the doorway map while
    # the same search set admitted their routes (2026-10-03, night/motion).
    free_space = geometry_module.inflated_free_cells(
        store,
        envelope,
        now_ns=now_ns,
        self_occupied_origin_odom_m=start,
        extra_margin_m=grid.voxel_m * CERTIFICATE_MARGIN_VOXELS,
    )
    published_free = store.free_cells(now_ns=now_ns)
    start_cell = grid.cell_index(start)
    goal_cells = frozenset(
        cell for cell in goal_region.cells(grid) if cell in free_space
    )
    if start_cell not in free_space:
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
    path = _astar(
        free_space, start_cell, goal_cells, grid, max_expansions=config.max_expansions
    )
    if path is BUDGET_EXHAUSTED:
        return PlanRefusal(
            COMPUTATION_LIMIT,
            f"the route search reached its declared {config.max_expansions}-expansion budget "
            "before it could decide: a computation limit, not a statement that no route exists",
            support="uncertain",
        )
    if path is not None:
        path = _merge_collinear(_shortcut(path, published_free, grid, envelope))
    if path is None:
        return PlanRefusal(
            NO_KNOWN_SUPPORTED_ROUTE,
            "no route exists in the known free map; an unobserved connection may still exist, so "
            "this is uncertain rather than physically impossible",
            support="uncertain",
        )
    waypoint_cells = _merge_collinear(path)
    waypoints = np.asarray(
        [grid.cell_center(cell) for cell in waypoint_cells], dtype=np.float64
    )
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
            return PlanRefusal(
                COMPUTATION_LIMIT,
                f"the quintic solve failed: {error}",
                support="uncertain",
            )
        coefficients = np.stack(per_axis, axis=1)  # (segments, axis, power)
        candidate, violation = _certify_segments(
            coefficients,
            durations,
            distances,
            published_free,
            grid,
            config,
            envelope,
            start,
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
                waypoints,
                durations,
                distances,
                published_free,
                grid,
                config,
                envelope,
                start,
                attempts,
            )
            if isinstance(fallback, PlanRefusal):
                return PlanRefusal(
                    fallback.reason,
                    f"{violation}; {fallback.detail}",
                    support="uncertain",
                )
            segments = fallback
            coefficients = None
            # The certificate's horizon must be the fallback's own time axis: the
            # loop's ``durations`` belong to the minimum-jerk form the fallback
            # replaced, and building t_end_s from them publishes a window longer
            # than the curve it certifies.
            durations = np.asarray([segment.duration_s for segment in fallback])
            break
        durations = lengthened
    if segments is None:
        return PlanRefusal(
            COMPUTATION_LIMIT, "the solver produced no segments", support="uncertain"
        )
    t_start = now_ns / 1e9
    dependent = sorted(
        {cell for segment in segments for cell in _cells_of(segment, grid)}
    )
    backup = _backup_segment(start, start_velocity_mps, config, free_space, grid)
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
            + (
                "minimum_jerk_C4"
                if coefficients is not None
                else "clamped_quintic_fallback"
            ),
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
        coefficients,
        segment_durations,
        distances,
        traversable,
        grid,
        config,
        envelope,
        start,
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


def _cells_of(
    segment: Segment, config: world_module.MapConfig
) -> tuple[tuple[int, int, int], ...]:
    region = geometry_module.BoxRegion(
        low=segment.low, high=segment.high, label="segment"
    )
    return region.cells(config)


SAMPLES_PER_SEGMENT_MIN = 8
SAMPLE_INTERVAL_MAX_S = 0.05


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
    voxel/4``) and requires every cell inside a ball of the declared envelope plus
    that same quarter voxel of slack to be published free — ``traversable`` here is
    the map's published free cells, not the occupancy set: the ball already carries
    the envelope once, and membership in the occupancy set would inflate a second
    time. The curve between two samples stays within ``v_max * dt / 2`` of them,
    so the whole swept tube is covered rather than merely sampled.
    """
    segments = []
    # The only writer of this clock is the knot bookkeeping below. The sweep
    # loop samples at its own local time: reusing this name there once silently
    # shifted every following segment's start by (samples-1)/samples of the
    # previous duration, gapping the certificate's time axis and clamping every
    # mid-plan sample to the curve's end (2026-10-03, night/motion).
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
                derivative = np.poly1d(list(reversed(coefficients[index, axis]))).deriv(
                    order
                )
                axis_values = [
                    float(derivative(0.0)),
                    float(derivative(durations[index])),
                ]
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
            coefficients=tuple(
                tuple(float(value) for value in coefficients[index, axis])
                for axis in range(3)
            ),
            low=low,
            high=high,
            max_speed_mps=max_speed,
            max_acceleration_mps2=max_acceleration,
            max_jerk_mps3=max_jerk,
            distance_m=float(distances[index]),
        )
        if max_speed > config.limits.v_max_mps + 1e-9:
            return (
                None,
                f"segment-{index} reaches {max_speed:.3f} m/s over the declared v_max",
            )
        if max_acceleration > config.limits.a_max_mps2 + 1e-9:
            return (
                None,
                f"segment-{index} reaches {max_acceleration:.3f} m/s^2 over the declared a_max",
            )
        if max_jerk > config.limits.jerk_max_mps3 + 1e-9:
            return (
                None,
                f"segment-{index} reaches {max_jerk:.3f} m/s^3 over the declared jerk limit",
            )
        slack = grid.voxel_m / 4.0
        samples = (
            max(
                int(math.ceil(segment.duration_s / _sample_interval(grid, config))),
                SAMPLES_PER_SEGMENT_MIN,
            )
            + 1
        )
        unsupported = None
        self_occupied_radius = envelope.inflation_m + slack
        for step in range(samples):
            sample_time_s = segment.t_start_s + segment.duration_s * step / samples
            centre = segment.value(sample_time_s)
            # The part of the curve still inside the envelope the aircraft already
            # occupies at its start is not a claim about new space: the aircraft is
            # there. Everything beyond that ball must be supported free space.
            if (
                float(np.linalg.norm(np.asarray(centre) - np.asarray(start)))
                <= self_occupied_radius
            ):
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
                f"segment-{index} sweeps cell {unsupported}, which is not published "
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
            coefficients=tuple(
                (float(value), 0.0, 0.0, 0.0, 0.0, 0.0) for value in start
            ),
            low=start,
            high=start,
            max_speed_mps=0.0,
            max_acceleration_mps2=0.0,
            max_jerk_mps3=0.0,
            distance_m=0.0,
        )
        missing = [cell for cell in _cells_of(segment, grid) if cell not in traversable]
        return None if missing else segment
    distance = speed * config.limits.reaction_s + speed * speed / (
        2.0 * config.limits.deceleration_mps2
    )
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
