"""The independent checker: it re-derives, and it can refuse the planner's output.

Specification section 14.1 and 14.3 fix this module's role and its limits. The
checker reads fresh state and sensor ages directly rather than the planner's
claim that its plan is safe, validates the active trajectory *and its stopping
continuation* against current nearby occupancy, the whole swept tube, the braking
region and the currently valid field of view, and uses a separate validation path
so a trajectory-generation defect can be caught.

So this module never calls the planner and never trusts the certificate's own
verdict: it re-evaluates the certificate's polynomials from their coefficients,
re-derives their extremes, re-checks every dependent cell against the *current*
map, rechecks the frame epoch and the state's declared error bound, and then
either permits or refuses with a named reason. It can refuse a certificate the
planner certified — that is its purpose.

Process separation does not supply independent evidence: the checker and planner
share physical sensors, and a common pose bias defeats both (section 14.3). That
limit travels with every result as :data:`SHARED_LIMITATION`.

Outcomes: permit, or switch to a supported backup; if no backup and no stop are
supported, no supported control remains and normal goals are invalidated
(``loss_of_supported_control``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embodied.contracts import records as R
from embodied.memory import world as world_module
from embodied.navigation import geometry as geometry_module
from embodied.navigation import planner as planner_module

PERMIT = "permit"
REFUSE = "refuse"

UNSUPPORTED_SPACE = "unsupported_space"
STALE_STATE = "stale_state"
DEPENDENCIES_EXPIRED = "dependencies_expired"
UNCERTIFIED_TRAJECTORY = "uncertified_trajectory"
LIMIT_VIOLATION = "limit_violation"
LOSS_OF_SUPPORTED_CONTROL = "loss_of_supported_control"
FRAME_EPOCH_MISMATCH = "frame_epoch_mismatch"
POSE_ERROR_EXCEEDS_ALLOWANCE = "pose_error_exceeds_allowance"

SHARED_LIMITATION = (
    "the checker and the planner share physical sensors: a common pose bias, glass or an "
    "unobserved wire can defeat both, and process separation is not independent evidence"
)

# Reason vocabulary of the execution horizon (section 17.1).
LIMIT_TRAJECTORY_SUPPORT = "remaining_trajectory_support"
LIMIT_STATE_VALIDITY = "state_validity"
LIMIT_GOAL_LEASE = "objective_lease"
LIMIT_RESOURCE = "resource_allowance"
LIMIT_DECISION_BOUNDARY = "time_to_semantic_decision_boundary"
LIMIT_UNKNOWN_INTRUSION = "unknown_boundary_intrusion"


@dataclass(frozen=True)
class Validation:
    """One verdict, with the evidence and the horizon that produced it."""

    outcome: str
    reasons: tuple[str, ...]
    limiting_reason: str | None
    horizon_s: float | None
    supported_progress_distance_m: float | None
    backup_outcome: str | None
    capabilities: tuple[str, ...]
    limitations: tuple[str, ...] = (SHARED_LIMITATION,)

    def permitted(self) -> bool:
        return self.outcome == PERMIT


def _independent_extremes(segment: planner_module.Segment) -> tuple[float, float, float]:
    """Re-derive a segment's speed, acceleration and jerk extremes from its coefficients."""
    speed = acceleration = jerk = 0.0
    for axis in segment.coefficients:
        polynomial = np.poly1d(list(reversed(axis)))
        for order, tracker in ((1, "speed"), (2, "acceleration"), (3, "jerk")):
            derivative = polynomial.deriv(order)
            values = [float(derivative(0.0)), float(derivative(segment.duration_s))]
            if derivative.order >= 1:
                for root in np.atleast_1d(derivative.roots):
                    if abs(root.imag) < 1e-9 and 0.0 < root.real < segment.duration_s:
                        values.append(float(derivative(root.real)))
            greatest = max(abs(value) for value in values)
            if tracker == "speed":
                speed = max(speed, greatest)
            elif tracker == "acceleration":
                acceleration = max(acceleration, greatest)
            else:
                jerk = max(jerk, greatest)
    return speed, acceleration, jerk


def validate(
    certificate: planner_module.TrajectoryCertificate,
    store: world_module.MapStore,
    state: R.NavigationState,
    envelope: geometry_module.Envelope,
    config: planner_module.PlanConfig,
    *,
    now_ns: int,
    pose_validity_s: float,
    lease_remaining_s: float | None,
    resource_allowance_s: float | None,
    decision_boundary_s: float | None,
) -> Validation:
    """Check one certified trajectory against fresh state, the current map and its backup."""
    reasons: list[str] = []
    capabilities: list[str] = ["brake", "hold"]

    if not state.pose.valid:
        return Validation(
            outcome=REFUSE,
            reasons=(f"{STALE_STATE}: the state's pose is marked invalid",),
            limiting_reason=LIMIT_STATE_VALIDITY,
            horizon_s=0.0,
            supported_progress_distance_m=0.0,
            backup_outcome=None,
            capabilities=(),
        )
    age_s = (now_ns - state.pose.stamp.monotonic_ns) / 1e9
    state_validity = pose_validity_s - age_s
    if age_s < 0.0:
        # A stamp from the future is not fresh state: the pair is inconsistent, and
        # treating it as valid would let a clock or ordering fault pass as freshness.
        return Validation(
            outcome=REFUSE,
            reasons=(
                f"{STALE_STATE}: the state's pose is stamped {abs(age_s):.3f} s after the present "
                "instant, so its freshness cannot be established",
            ),
            limiting_reason=LIMIT_STATE_VALIDITY,
            horizon_s=0.0,
            supported_progress_distance_m=0.0,
            backup_outcome=None,
            capabilities=(),
        )
    if state_validity <= 0.0:
        return Validation(
            outcome=REFUSE,
            reasons=(
                f"{STALE_STATE}: the state's pose is {age_s:.3f} s old, beyond the declared "
                f"{pose_validity_s:.1f} s validity",
            ),
            limiting_reason=LIMIT_STATE_VALIDITY,
            horizon_s=0.0,
            supported_progress_distance_m=0.0,
            backup_outcome=None,
            capabilities=(),
        )
    if state.pose.covariance is not None:
        bound = float(np.sqrt(max(max(state.pose.covariance[:3]), 0.0)))
        if 3.0 * bound > envelope.error_allowance_m:
            return Validation(
                outcome=REFUSE,
                reasons=(
                    f"{POSE_ERROR_EXCEEDS_ALLOWANCE}: the state declares a {bound:.3f} m pose "
                    f"sigma, so its three-sigma bound {3.0 * bound:.3f} m exceeds the declared "
                    f"{envelope.error_allowance_m:.3f} m error allowance",
                ),
                limiting_reason=LIMIT_STATE_VALIDITY,
                horizon_s=0.0,
                supported_progress_distance_m=0.0,
                backup_outcome=None,
                capabilities=(),
            )

    if certificate.nav_epoch != state.nav_epoch:
        return Validation(
            outcome=REFUSE,
            reasons=(
                f"{FRAME_EPOCH_MISMATCH}: the certificate belongs to nav_epoch "
                f"{certificate.nav_epoch!r} and the current state is {state.nav_epoch!r}; a reset "
                "invalidates prior control references",
            ),
            limiting_reason=LIMIT_STATE_VALIDITY,
            horizon_s=0.0,
            supported_progress_distance_m=0.0,
            backup_outcome=None,
            capabilities=(),
        )

    if not certificate.certified:
        return Validation(
            outcome=REFUSE,
            reasons=(
                f"{UNCERTIFIED_TRAJECTORY}: the certificate carries no complete-segment "
                "certification, and an uncertified curve is not published",
            ),
            limiting_reason=LIMIT_TRAJECTORY_SUPPORT,
            horizon_s=0.0,
            supported_progress_distance_m=0.0,
            backup_outcome=None,
            capabilities=(),
        )

    if store.revision != certificate.map_revision:
        return Validation(
            outcome=REFUSE,
            reasons=(
                f"{DEPENDENCIES_EXPIRED}: the certificate depends on map revision "
                f"{certificate.map_revision!r} and the map has advanced to {store.revision!r}; "
                "the trajectory must be revalidated against the new revision, not assumed valid",
            ),
            limiting_reason=LIMIT_TRAJECTORY_SUPPORT,
            horizon_s=0.0,
            supported_progress_distance_m=0.0,
            backup_outcome=None,
            capabilities=tuple(capabilities),
        )

    for segment in certificate.segments:
        speed, acceleration, jerk = _independent_extremes(segment)
        if speed > config.limits.v_max_mps + 1e-6:
            reasons.append(f"{LIMIT_VIOLATION}: segment-{segment.index} reaches {speed:.3f} m/s")
        if acceleration > config.limits.a_max_mps2 + 1e-6:
            reasons.append(
                f"{LIMIT_VIOLATION}: segment-{segment.index} reaches {acceleration:.3f} m/s^2"
            )
        if jerk > config.limits.jerk_max_mps3 + 1e-6:
            reasons.append(f"{LIMIT_VIOLATION}: segment-{segment.index} reaches {jerk:.3f} m/s^3")

    swept_ok, swept_reason = geometry_module.swept_support(
        store,
        certificate,
        now_ns=now_ns,
        self_occupied_origin_odom_m=state.pose.position_m,
    )
    if not swept_ok:
        reasons.append(f"{UNSUPPORTED_SPACE}: {swept_reason}")
    dependent_unsupported = [
        cell
        for cell in certificate.dependent_cells
        if store.classify(cell, now_ns=now_ns) != world_module.FREE
    ]
    if dependent_unsupported:
        sample = dependent_unsupported[0]
        reasons.append(
            f"{UNSUPPORTED_SPACE}: {len(dependent_unsupported)} dependent cells are no longer "
            f"published free space (for example {sample} is "
            f"{store.classify(sample, now_ns=now_ns)}); unknown is never free"
        )

    velocity = state.velocity_mps or (0.0, 0.0, 0.0)
    braking = geometry_module.braking_region(
        state.pose.position_m,
        velocity,
        envelope,
        deceleration_mps2=config.limits.deceleration_mps2,
        reaction_s=config.limits.reaction_s,
        tracking_error_m=envelope.error_allowance_m,
    )
    braking_supported, braking_reason = geometry_module.region_support(
        store,
        braking,
        envelope,
        now_ns=now_ns,
        self_occupied_origin_odom_m=state.pose.position_m,
    )
    if not braking_supported:
        reasons.append(f"{UNSUPPORTED_SPACE}: the braking continuation is not supported: {braking_reason}")
    backup_supported = certificate.backup is not None
    if not backup_supported:
        reasons.append(f"{LOSS_OF_SUPPORTED_CONTROL}: the certificate carries no checked backup")

    remaining_support = max(certificate.t_end_s - now_ns / 1e9, 0.0)
    candidates = [
        (remaining_support, LIMIT_TRAJECTORY_SUPPORT),
        (state_validity, LIMIT_STATE_VALIDITY),
    ]
    if lease_remaining_s is not None:
        candidates.append((lease_remaining_s, LIMIT_GOAL_LEASE))
    if resource_allowance_s is not None:
        candidates.append((resource_allowance_s, LIMIT_RESOURCE))
    if decision_boundary_s is not None:
        candidates.append((decision_boundary_s, LIMIT_DECISION_BOUNDARY))
    horizon, limiting = min(candidates, key=lambda entry: entry[0])
    intrusion = store.unknown_intrusion()
    if intrusion and store.intrusion_supported():
        shortened = _time_until_intrusion(certificate, state, intrusion, store)
        if shortened is not None and shortened < horizon:
            horizon, limiting = shortened, LIMIT_UNKNOWN_INTRUSION
    progress = _progress_within(certificate, state, horizon)
    backup_outcome = "brake" if braking_supported else None
    if not braking_supported and not dependent_unsupported and swept_ok:
        backup_outcome = "hold" if store.classify(store.config.cell_index(state.pose.position_m), now_ns=now_ns) == world_module.FREE else None
    if reasons:
        return Validation(
            outcome=REFUSE,
            reasons=tuple(reasons),
            limiting_reason=limiting,
            horizon_s=horizon,
            supported_progress_distance_m=0.0,
            backup_outcome=backup_outcome,
            capabilities=tuple(capabilities) if backup_outcome else (),
        )
    return Validation(
        outcome=PERMIT,
        reasons=("certificate re-derived, dependencies current and the backup supported",),
        limiting_reason=limiting,
        horizon_s=horizon,
        supported_progress_distance_m=progress,
        backup_outcome=backup_outcome,
        capabilities=tuple(capabilities),
    )


def _time_until_intrusion(
    certificate: planner_module.TrajectoryCertificate,
    state: R.NavigationState,
    intrusion: frozenset[tuple[int, int, int]],
    store: world_module.MapStore,
) -> float | None:
    """Seconds of travel before the route enters the intrusion envelope.

    Section 7.1: intrusion from reachable unknown boundaries shortens the future
    free corridor. It bounds how far the aircraft may commit, so it is reported as
    a horizon term with its own limiting reason rather than as a silent reduction
    of the free space the map actually published.
    """
    step_s = max(store.config.voxel_m / max(2.0 * certificate.swept_radius_m, 1e-6), 0.02)
    time = now = certificate.t_start_s
    while time <= certificate.t_end_s:
        position = tuple(float(value) for value in certificate.sample(time)[0])
        if store.config.cell_index(position) in intrusion:
            return time - now
        time += step_s
    return None


def _progress_within(
    certificate: planner_module.TrajectoryCertificate,
    state: R.NavigationState,
    horizon_s: float,
) -> float:
    """Distance along the intended relationship that can be completed inside the horizon.

    Holding preserves containment but produces no mission progress, so progress is
    reported separately from the horizon itself (section 17.1).
    """
    try:
        if horizon_s <= 0.0:
            return 0.0
        start = np.asarray(state.pose.position_m, dtype=np.float64)
        end_time = min(certificate.t_start_s + horizon_s, certificate.t_end_s)
        travelled = 0.0
        previous = start
        steps = max(int((end_time - certificate.t_start_s) / 0.05), 1)
        for index in range(steps + 1):
            time = certificate.t_start_s + (end_time - certificate.t_start_s) * index / steps
            position = np.asarray(certificate.sample(time)[0], dtype=np.float64)
            travelled += float(np.linalg.norm(position - previous))
            previous = position
        return travelled
    except Exception:  # pragma: no cover - defensive: a malformed certificate is refused above
        return 0.0
