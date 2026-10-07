"""EstimationPort producers for the stacked Perception surface.

Adapters may wrap ``platform.localization`` feeds; stubs speak the full
``NavigationState`` contract with explicit unhealthy/unavailable values.
"""

from __future__ import annotations

from dataclasses import replace

from embodied.contracts.perception_ports import (
    AgeDomain,
    ApDisagreement,
    EvidenceClass,
    EstimationPort,
    NavEpoch,
    NavPose,
    NavStatus,
    NavigationState,
    uncompared_disagreement,
)
from embodied.contracts.records import ClockStamp, PoseEstimate
from embodied.platform.localization import EstimatorState


class StaticEstimationPort:
    """Holds a single publishable state (or None when unwired)."""

    def __init__(self, state: NavigationState | None = None) -> None:
        self._state = state

    def set(self, state: NavigationState | None) -> None:
        self._state = state

    def latest(self) -> NavigationState | None:
        return self._state

    def current_epoch(self) -> NavEpoch | None:
        state = self._state
        return None if state is None else state.nav_epoch

    def healthy(self) -> bool:
        state = self._state
        return (
            state is not None
            and state.valid
            and state.status in (NavStatus.HEALTHY, NavStatus.DEGRADED)
        )


def unavailable_navigation_state(
    *,
    nav_epoch: NavEpoch,
    stamp: ClockStamp,
    source_ids: tuple[str, ...] = ("stub_estimation",),
    evidence_class: EvidenceClass = EvidenceClass.UNAVAILABLE,
) -> NavigationState:
    """Full-type refusal: port up, no usable pose."""
    if evidence_class is EvidenceClass.ORACLE:
        evidence_class = EvidenceClass.UNAVAILABLE
    pose = NavPose(
        parent_frame="odom",
        child_frame="body",
        stamp=stamp,
        position_m=(0.0, 0.0, 0.0),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance=None,
        nav_epoch=nav_epoch,
        source_ids=source_ids,
        valid=False,
    )
    return NavigationState(
        nav_epoch=nav_epoch,
        state_sequence=0,
        controller_alignment_id=None,
        stamp=stamp,
        sim_time_s=None,
        age_s=0.0,
        age_domain=AgeDomain.MONOTONIC,
        monotonic_observed_at_s=None,
        pose=pose,
        velocity_mps=None,
        covariance=None,
        status=NavStatus.UNAVAILABLE,
        valid=False,
        sigma_pos_m=None,
        visual_source_ids=(),
        imu_source_ids=(),
        visual_age_s=None,
        imu_age_s=None,
        feed_stall=False,
        ap_disagreement=uncompared_disagreement(),
        evidence_class=evidence_class,
        source_ids=source_ids,
    )


def navigation_state_from_estimator(
    state: EstimatorState,
    *,
    nav_epoch: NavEpoch,
    stamp: ClockStamp,
    sim_time_s: float | None,
    age_s: float,
    age_domain: AgeDomain,
    status: NavStatus,
    valid: bool,
    evidence_class: EvidenceClass,
    controller_alignment_id: str | None,
    state_sequence: int,
    visual_source_ids: tuple[str, ...],
    imu_source_ids: tuple[str, ...],
    visual_age_s: float | None,
    imu_age_s: float | None,
    feed_stall: bool,
    ap_disagreement: ApDisagreement | None = None,
    source_ids: tuple[str, ...] = ("ov_stream",),
    monotonic_observed_at_s: float | None = None,
) -> NavigationState:
    """Map an OV ``EstimatorState`` into the stacked ``NavigationState``."""
    if evidence_class is EvidenceClass.ORACLE:
        raise ValueError("oracle evidence_class cannot enter EstimationPort autonomy publishes")
    pose_valid = bool(valid and state.initialized and not feed_stall)
    if status in (NavStatus.STALE, NavStatus.LOST, NavStatus.UNAVAILABLE):
        pose_valid = False
        valid = False
    pose = NavPose(
        parent_frame="odom",
        child_frame="body",
        stamp=stamp,
        position_m=tuple(float(v) for v in state.position_m),  # type: ignore[arg-type]
        quaternion_wxyz=tuple(float(v) for v in state.quat_wxyz),  # type: ignore[arg-type]
        covariance=None,
        nav_epoch=nav_epoch,
        source_ids=source_ids,
        valid=pose_valid,
    )
    return NavigationState(
        nav_epoch=nav_epoch,
        state_sequence=state_sequence,
        controller_alignment_id=controller_alignment_id,
        stamp=stamp,
        sim_time_s=sim_time_s,
        age_s=age_s,
        age_domain=age_domain,
        monotonic_observed_at_s=monotonic_observed_at_s,
        pose=pose,
        velocity_mps=tuple(float(v) for v in state.velocity_mps),  # type: ignore[arg-type]
        covariance=None,
        status=status,
        valid=valid and pose_valid,
        sigma_pos_m=tuple(float(v) for v in state.sigma_pos_m),  # type: ignore[arg-type]
        visual_source_ids=visual_source_ids,
        imu_source_ids=imu_source_ids,
        visual_age_s=visual_age_s,
        imu_age_s=imu_age_s,
        feed_stall=feed_stall,
        ap_disagreement=ap_disagreement if ap_disagreement is not None else uncompared_disagreement(),
        evidence_class=evidence_class,
        source_ids=source_ids,
    )


def pose_estimate_from_nav(state: NavigationState) -> PoseEstimate:
    """Legacy ``PoseEstimate`` for MapStore.integrate from stacked nav."""
    return PoseEstimate(
        parent_frame=state.pose.parent_frame,
        child_frame=state.pose.child_frame,
        stamp=state.pose.stamp,
        position_m=state.pose.position_m,
        quaternion_wxyz=state.pose.quaternion_wxyz,
        covariance=state.pose.covariance,
        nav_epoch=state.pose.nav_epoch,
        source_ids=state.pose.source_ids,
        valid=state.pose.valid and state.evidence_class is EvidenceClass.SENSOR_DERIVED,
    )


def with_status(state: NavigationState, status: NavStatus, *, feed_stall: bool | None = None) -> NavigationState:
    """Return a copy with status/validity invariants applied."""
    stall = state.feed_stall if feed_stall is None else feed_stall
    valid = state.valid
    pose_valid = state.pose.valid
    if status in (NavStatus.STALE, NavStatus.LOST, NavStatus.UNAVAILABLE) or stall:
        valid = False
        pose_valid = False
    if stall and status not in (NavStatus.STALE, NavStatus.LOST):
        status = NavStatus.STALE
    pose = replace(state.pose, valid=pose_valid)
    return replace(state, status=status, valid=valid, feed_stall=stall, pose=pose)


# Protocol structural satisfaction for type checkers / runtime_checkable.
_: type[EstimationPort] = StaticEstimationPort
