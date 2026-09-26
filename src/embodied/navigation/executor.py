"""Goal admission, the accepted objective and the single setpoint producer.

Specification sections 12.2 and 13.3, and the slice's own words, put three
responsibilities here: a staged feasibility assessment attached to admission (not
a second planning system), the accepted local goal, and the one place a
:class:`embodied.contracts.records.MotionSetpoint` is built — from a validated
prefix of a certified trajectory, through the P00 adapter's frame conventions.

The stage order matters and is enforced:

1. **Target identity.** The goal must resolve to a grounded target. A goal that
   cites only image selections, with no grounded geometry, is rejected here,
   *before* any planner call: no path is emitted from an image coordinate alone.
2. **Frames, sensor and state validity.** nav_epoch, pose validity, the target's
   anchor revision.
3. **Dimensional fit.** The observed opening must leave a positive corridor once
   the declared envelope is inflated; otherwise the goal is infeasible under the
   named constraint.
4. **Scope and resources.** The goal's own lease bounds must be positive and the
   objective must fit inside its authorized scope.
5. **The planner's witness.** Only then is the shared planner asked for a route,
   a trajectory and a stopping continuation, and its concrete result is the
   witness that the motion is feasible under its assumptions.

This stage's executor stops at record construction: no live MAVLink publication
occurs, and the setpoints it builds are records, not commands. P04 submits through
this canonical executor rather than a duplicate admission state machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embodied.contracts import records as R
from embodied.memory import world as world_module
from embodied.navigation import geometry as geometry_module
from embodied.navigation import planner as planner_module
from embodied.navigation import validator as validator_module
from embodied.perception import grounding as grounding_module

FEASIBLE = "feasible"
INFEASIBLE = "infeasible"
UNCERTAIN = "uncertain"

NO_GROUNDED_TARGET = "no_grounded_target"
TARGET_NOT_CURRENT = "target_not_current"
INSTANT_BOUNDS_MISSING = "lease_bounds_missing"


@dataclass(frozen=True)
class GoalAssessment:
    """The admission service's verdict, its reasons and the planner's witness.

    Defined here as its first real writer (records.py:1194 names the admission
    service); promotion into the shared contract is a serialized integrator change.
    """

    verdict: str
    reasons: tuple[str, ...]
    conditions: tuple[tuple[str, str], ...]
    planner_witness: str | None
    information_alternative: str | None
    dependencies: tuple[tuple[str, str], ...]
    estimated_cost: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.verdict not in (FEASIBLE, INFEASIBLE, UNCERTAIN):
            raise R.RecordError(f"{self.verdict!r} is not a feasibility verdict")
        if self.verdict == FEASIBLE and self.planner_witness is None:
            raise R.RecordError("a feasible assessment cites the planner's witness")


@dataclass(frozen=True)
class AcceptedGoal:
    """The locally admitted objective, with what it resolved to."""

    goal_id: str
    goal_revision: int
    proposal_id: str
    objective: R.SpatialGoal
    resolved_targets: tuple[str, ...]
    nav_epoch: str
    disposition: R.ExecutionDisposition
    lease_s: float

    def __post_init__(self) -> None:
        for name in ("goal_id", "proposal_id", "nav_epoch"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise R.RecordError(f"an accepted goal needs a non-empty {name}")
        if not self.resolved_targets:
            raise R.RecordError("an accepted goal resolves at least one grounded target")
        if not isinstance(self.objective, R.SpatialGoal):
            raise R.RecordError("an accepted goal carries the proposal it admitted")


@dataclass(frozen=True)
class AdmissionResult:
    """Everything admission produced: the status, the assessment and what it accepted."""

    status: R.GoalStatus
    assessment: GoalAssessment
    accepted: AcceptedGoal | None
    certificate: planner_module.TrajectoryCertificate | None


def _named_lease(goal: R.SpatialGoal) -> dict[str, float]:
    return {name: value for name, value in goal.lease_bounds}


def admit(
    goal: R.SpatialGoal,
    targets: tuple[R.GroundedTarget, ...],
    store: world_module.MapStore,
    state: R.NavigationState,
    envelope: geometry_module.Envelope,
    config: planner_module.PlanConfig,
    *,
    mission_revision: int,
    snapshot_id: str,
    now_ns: int,
    anchor_revision: str | None = None,
    pose_validity_s: grounding_module.POSE_VALIDITY_S,
    direction_sign: float = 1.0,
) -> AdmissionResult:
    """Assess and admit one goal, calling the planner only when the earlier stages pass."""
    dependencies = (
        ("map_revision", store.revision),
        ("anchor_revision", anchor_revision or store.revision),
        ("snapshot_id", snapshot_id),
        ("mission_revision", str(mission_revision)),
    )
    lease = _named_lease(goal)
    conditions = (
        ("intent", goal.intent),
        ("lease_s", f"{lease.get('lease_s', 0.0):.3f}"),
        ("nav_epoch", state.nav_epoch),
    )

    def rejected(reason: str, detail: str, verdict: str = INFEASIBLE, alternative: str | None = None) -> AdmissionResult:
        return AdmissionResult(
            status=R.GoalStatus(
                proposal_id=goal.proposal_id,
                request_id=goal.request_id or goal.proposal_id,
                disposition=R.GoalDisposition.REJECTED,
                reason=f"{reason}: {detail}",
                admission_ref=None,
                current_disposition=R.ExecutionDisposition.BLOCKED,
            ),
            assessment=GoalAssessment(
                verdict=verdict,
                reasons=(f"{reason}: {detail}",),
                conditions=conditions,
                planner_witness=None,
                information_alternative=alternative,
                dependencies=dependencies,
                estimated_cost=(),
            ),
            accepted=None,
            certificate=None,
        )

    if not lease or any(value <= 0.0 for value in lease.values()):
        return rejected(
            INSTANT_BOUNDS_MISSING,
            "an objective without positive lease bounds is not an authorization",
            verdict=UNCERTAIN,
        )
    if not targets:
        return rejected(
            NO_GROUNDED_TARGET,
            "the goal cites only image selections and no grounded geometry; a path is never "
            "emitted from an image coordinate alone",
            verdict=INFEASIBLE,
            alternative="ground the selection and re-propose the goal with a resolved target",
        )
    for target in targets:
        refusal = grounding_module.target_currency(
            target, state, anchor_revision=anchor_revision
        )
        if refusal is not None:
            return rejected(
                TARGET_NOT_CURRENT,
                refusal.detail,
                verdict=UNCERTAIN,
                alternative="re-observe the target and re-ground it against a current anchor",
            )
    if not state.pose.valid:
        return rejected(TARGET_NOT_CURRENT, "the current state's pose is invalid", verdict=UNCERTAIN)

    target = targets[0]
    aperture = grounding_module.parse_aperture(target.geometry)
    point = grounding_module.parse_point(target.geometry)
    terminal: geometry_module.BoxRegion | None = None
    if goal.intent == "traverse":
        if aperture is None:
            return rejected(
                NO_GROUNDED_TARGET,
                "a traverse needs an observed aperture; this target carries "
                f"{len(target.geometry)} numbers, which is not an opening rectangle",
                verdict=UNCERTAIN,
                alternative="select the doorway opening itself and re-ground it",
            )
        regions = geometry_module.traverse_regions(aperture, envelope, direction_sign=direction_sign)
        if not regions.feasible:
            return rejected(
                regions.constraint or geometry_module.APERTURE_CLEARANCE,
                "; ".join(regions.reasons),
                verdict=INFEASIBLE,
                alternative="approach for a closer view instead of traversing",
            )
        terminal, reason = geometry_module.shrink_to_supported(
            store,
            regions.terminal,
            envelope,
            now_ns=now_ns,
            self_occupied_origin_odom_m=state.pose.position_m,
        )
        if terminal is None:
            return rejected(
                validator_module.UNSUPPORTED_SPACE,
                f"the exit region is not supported: {reason}",
                verdict=UNCERTAIN,
                alternative="approach for another view before committing to a crossing",
            )
    elif point is not None or aperture is not None:
        anchor = np.asarray(aperture.center_odom_m() if aperture is not None else point, dtype=np.float64)
        direction = np.asarray(aperture.plane_normal_odom if aperture is not None else (1.0, 0.0, 0.0))
        terminal = geometry_module.approach_region(
            tuple(float(value) for value in anchor),
            envelope,
            direction=tuple(float(value) for value in direction),
        )
        terminal, reason = geometry_module.shrink_to_supported(
            store,
            terminal,
            envelope,
            now_ns=now_ns,
            self_occupied_origin_odom_m=state.pose.position_m,
        )
        if terminal is None:
            return rejected(
                validator_module.UNSUPPORTED_SPACE,
                f"the {goal.intent} region is not supported: {reason}",
                verdict=UNCERTAIN,
                alternative="observe a reachable viewpoint for this target first",
            )
    else:
        return rejected(
            NO_GROUNDED_TARGET,
            f"the target geometry is not a point or an aperture ({len(target.geometry)} numbers)",
            verdict=UNCERTAIN,
        )

    certificate = planner_module.plan(
        store,
        envelope,
        config,
        terminal,
        navigation_state=state,
        nav_epoch=state.nav_epoch,
        goal_id=goal.proposal_id,
        goal_revision=goal.base_goal_revision + 1,
        mission_revision=mission_revision,
        target_refs=(target.target_id,),
        anchor_id=target.anchor_id,
        anchor_revision=target.anchor_revision,
        snapshot_id=snapshot_id,
        now_ns=now_ns,
    )
    if isinstance(certificate, planner_module.PlanRefusal):
        verdict = UNCERTAIN if certificate.reason == planner_module.NO_KNOWN_SUPPORTED_ROUTE else UNCERTAIN
        return rejected(
            certificate.reason,
            certificate.detail,
            verdict=verdict,
            alternative="observe a candidate route before committing to this goal",
        )
    accepted = AcceptedGoal(
        goal_id=f"goal-{goal.proposal_id}",
        goal_revision=goal.base_goal_revision + 1,
        proposal_id=goal.proposal_id,
        objective=goal,
        resolved_targets=(target.target_id,),
        nav_epoch=state.nav_epoch,
        disposition=R.ExecutionDisposition.NOT_STARTED,
        lease_s=lease.get("lease_s", 0.0),
    )
    return AdmissionResult(
        status=R.GoalStatus(
            proposal_id=goal.proposal_id,
            request_id=goal.request_id or goal.proposal_id,
            disposition=R.GoalDisposition.ACCEPTED,
            reason=(
                "admitted on a staged assessment: target identity, frames, dimensional fit, "
                "scope and resources passed, and the planner supplied a certified trajectory"
            ),
            admission_ref=accepted.goal_id,
            current_disposition=R.ExecutionDisposition.NOT_STARTED,
        ),
        assessment=GoalAssessment(
            verdict=FEASIBLE,
            reasons=(
                "the shared planner returned a certified trajectory with a checked stopping "
                "continuation inside the observed opening",
            ),
            conditions=conditions,
            planner_witness=certificate.certificate_id,
            information_alternative=None,
            dependencies=dependencies,
            estimated_cost=(
                ("path_length_m", sum(segment.distance_m for segment in certificate.segments)),
                ("duration_s", certificate.horizon_s),
            ),
        ),
        accepted=accepted,
        certificate=certificate,
    )


def publish_prefix(
    certificate: planner_module.TrajectoryCertificate,
    state: R.NavigationState,
    config: planner_module.PlanConfig,
    *,
    goal_revision: int,
    mission_revision: int,
    issue_stamp: R.ClockStamp,
    command_sequence_start: int = 1,
    now_ns: int | None = None,
) -> tuple[R.MotionSetpoint, ...]:
    """Build the setpoint prefix: only a short certified prefix, later segments replaceable.

    The published fields are position and velocity in the odom frame, converted to
    the P00 adapter's NED ordering with its own ``enu_to_ned`` (keep x, negate y
    and z) so the wire convention is the adapter's rather than a second invention.
    No live publication happens here: these are records.
    """
    from embodied.platform.webots_ardupilot import enu_to_ned

    start_s = certificate.t_start_s if now_ns is None else max(now_ns / 1e9, certificate.t_start_s)
    horizon = min(start_s + config.setpoint_prefix_horizon_s, certificate.t_end_s)
    count = max(int(round((horizon - start_s) / config.setpoint_period_s)), 1) + 1
    setpoints = []
    for index in range(count):
        time_s = min(start_s + index * config.setpoint_period_s, certificate.t_end_s)
        position, velocity, _acceleration = certificate.sample(time_s)
        target = R.MotionTarget(
            position_ned=enu_to_ned(position),
            velocity_ned=enu_to_ned(velocity),
            acceleration_ned=None,
            yaw_rad=None,
            yaw_rate_rad_s=None,
        )
        setpoints.append(
            R.MotionSetpoint(
                command_sequence=command_sequence_start + index,
                mission_revision=mission_revision,
                goal_revision=goal_revision,
                nav_epoch=certificate.nav_epoch,
                frame=R.Frame.ODOM,
                type_mask=R.TYPE_MASK_POSITION_VELOCITY,
                target=target,
                issue_stamp=issue_stamp,
                deadline_s=config.setpoint_period_s,
                certificate_ref=certificate.certificate_id,
                sample_ref=f"{certificate.certificate_id}#{index}",
                source=R.SetpointSource.NORMAL,
            )
        )
    return tuple(setpoints)


def execution_status(
    accepted: AcceptedGoal,
    certificate: planner_module.TrajectoryCertificate,
    setpoints: tuple[R.MotionSetpoint, ...],
    *,
    horizon_s: float,
    limiting_reason: str | None,
    disposition: R.ExecutionDisposition,
    evidence: tuple[str, ...],
    reasons: tuple[str, ...],
    capabilities: tuple[str, ...],
) -> R.ExecutionStatus:
    """Report what the supervisor believes is happening, and on what evidence."""
    return R.ExecutionStatus(
        goal_ref=accepted.goal_id,
        certificate_ref=certificate.certificate_id,
        command_ref=setpoints[-1].sample_ref if setpoints else None,
        disposition=disposition,
        evidence=evidence,
        reasons=reasons,
        horizon_s=horizon_s,
        capabilities=capabilities,
    )


def completion_status(
    accepted: AcceptedGoal,
    certificate: planner_module.TrajectoryCertificate,
    setpoints: tuple[R.MotionSetpoint, ...],
    *,
    position_odom_m: tuple[float, float, float],
    speed_mps: float,
    terminal_region: geometry_module.BoxRegion | None,
    settle_position_tolerance_m: float,
    settle_speed_tolerance_mps: float,
    evidence: tuple[str, ...],
) -> R.ExecutionStatus:
    """A completion is a claim with evidence over a settle interval, never a guess.

    Passing through one point is not proof of a stable viewpoint, so arrival needs
    the declared position and velocity relationship to hold, and it cites the
    evidence that it did (records.py's own rule for a completed status).
    """
    if not evidence:
        raise R.RecordError("a completion cites the evidence it was observed on")
    if terminal_region is not None and terminal_region.contains(position_odom_m):
        if speed_mps <= settle_speed_tolerance_mps:
            disposition = R.ExecutionDisposition.COMPLETED
            reasons = (
                f"settled inside the {terminal_region.label} region at {speed_mps:.3f} m/s, "
                f"within the declared {settle_speed_tolerance_mps:.3f} m/s settle tolerance",
            )
        else:
            disposition = R.ExecutionDisposition.RUNNING
            reasons = (f"inside the terminal region but still moving at {speed_mps:.3f} m/s",)
    else:
        distance = min(
            abs(position_odom_m[axis] - terminal_region.center()[axis]) for axis in range(3)
        ) if terminal_region is not None else float("inf")
        disposition = R.ExecutionDisposition.RUNNING
        reasons = (
            f"outside the terminal region (nearest-axis margin {distance:.3f} m, settle tolerance "
            f"{settle_position_tolerance_m:.3f} m)",
        )
    return execution_status(
        accepted,
        certificate,
        setpoints,
        horizon_s=certificate.horizon_s,
        limiting_reason=validator_module.LIMIT_TRAJECTORY_SUPPORT,
        disposition=disposition,
        evidence=evidence,
        reasons=reasons,
        capabilities=("brake", "hold"),
    )
