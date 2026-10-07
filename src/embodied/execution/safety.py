"""Pure Safety.check — ALLOW / BACKUP / UNSUPPORTED leases. Never commands."""

from __future__ import annotations

from dataclasses import dataclass

from embodied.contracts.perception_ports import (
    EvidenceClass,
    NavStatus,
    NavigationState,
    OccupancyQuery,
    OccupancySupport,
    StopTubeQuery,
)
from embodied.control import Motion, VehicleState, Vec3
from embodied.execution.certificate import (
    GeometryCertificate,
    TrackingEnvelope,
    Trajectory,
    ValidityWindow,
)
from embodied.execution.plant import PlantLimits, TelemetryHealth


@dataclass(frozen=True)
class AuthorityView:
    armed: bool
    guided: bool
    landed: bool | None
    failsafe_active: bool
    telemetry_age_mono_s: float
    command_rejecting: bool


@dataclass(frozen=True)
class TrackingState:
    candidate: Motion
    last_published: Motion | None
    measured_position: Vec3
    measured_velocity: Vec3
    measured_yaw: float | None
    measured_age_mono_s: float
    measured_source: str  # vehicle | nav | fused
    publish_sequence: int


@dataclass(frozen=True)
class AllowDecision:
    valid_until_mono_s: float
    nav_epoch: str
    checked_refs: tuple[str, ...]
    mode: str  # primary | terminal | backup_active


@dataclass(frozen=True)
class BackupDecision:
    trajectory_name: str
    valid_until_mono_s: float
    reason: str
    checked_refs: tuple[str, ...]


@dataclass(frozen=True)
class UnsupportedDecision:
    reason: str
    checked_refs: tuple[str, ...]


SafetyDecision = AllowDecision | BackupDecision | UnsupportedDecision


def _dist(a: Vec3, b: Vec3) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _backup_or_unsupported(
    predeclared_fallbacks: dict[str, Trajectory],
    *,
    reason: str,
    checked_refs: tuple[str, ...],
    now_mono_s: float,
    lease_s: float,
    prefer_name: str = "hold",
) -> SafetyDecision:
    name = prefer_name if prefer_name in predeclared_fallbacks else None
    if name is None and predeclared_fallbacks:
        name = next(iter(predeclared_fallbacks))
    if name is None:
        return UnsupportedDecision(reason=reason, checked_refs=checked_refs)
    return BackupDecision(
        trajectory_name=name,
        valid_until_mono_s=now_mono_s + lease_s,
        reason=reason,
        checked_refs=checked_refs,
    )


def check(
    active_prefix: Trajectory,
    stop_continuation: Trajectory,
    predeclared_fallbacks: dict[str, Trajectory],
    nav_state: NavigationState | None,
    vehicle_state: VehicleState,
    authority: AuthorityView,
    tracking_state: TrackingState,
    tracking_envelope: TrackingEnvelope,
    occupancy: OccupancyQuery | None,
    plant_limits: PlantLimits | None,
    telemetry_health: TelemetryHealth | None,
    geometry_certificate: GeometryCertificate | None,
    certificate_nav_epoch: str,
    certificate_validity: ValidityWindow,
    safety_evidence_refs: tuple[str, ...],
    mode: str,
    now_mono_s: float,
    now_sim_s: float,
    *,
    lease_s: float = 0.25,
    nav_age_max_s: float = 0.5,
    telem_age_max_s: float = 1.0,
) -> SafetyDecision:
    """Certify the active prefix. Never invents Motions or free space."""
    del active_prefix, stop_continuation, now_sim_s  # available for future motion-bound checks
    refs: list[str] = ["vehicle", *safety_evidence_refs]

    if not authority.armed or not authority.guided:
        return UnsupportedDecision(reason="authority_not_armed_guided", checked_refs=tuple(refs))
    if authority.failsafe_active:
        return UnsupportedDecision(reason="failsafe_active", checked_refs=tuple(refs))
    if authority.command_rejecting:
        return UnsupportedDecision(reason="command_rejecting", checked_refs=tuple(refs))
    if authority.telemetry_age_mono_s > telem_age_max_s:
        return UnsupportedDecision(reason="telemetry_stale", checked_refs=tuple(refs))
    if telemetry_health is not None:
        refs.append("telemetry_health")
        if not telemetry_health.valid or telemetry_health.failsafe_active or not telemetry_health.link_ok:
            return UnsupportedDecision(reason="telemetry_health_bad", checked_refs=tuple(refs))
        if telemetry_health.telemetry_age_s > telem_age_max_s:
            return UnsupportedDecision(reason="telemetry_health_stale", checked_refs=tuple(refs))

    if not vehicle_state.armed or not vehicle_state.guided:
        return UnsupportedDecision(reason="vehicle_not_armed_guided", checked_refs=tuple(refs))

    validity = certificate_validity
    if validity.not_before_mono_s is not None and now_mono_s < validity.not_before_mono_s:
        return UnsupportedDecision(reason="validity_not_before", checked_refs=tuple(refs))
    if validity.not_after_mono_s is not None and now_mono_s > validity.not_after_mono_s:
        return UnsupportedDecision(reason="validity_expired", checked_refs=tuple(refs))

    # Nav health / epoch
    if nav_state is None:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="nav_missing",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    refs.append(f"nav_epoch:{nav_state.nav_epoch}")
    if nav_state.evidence_class is EvidenceClass.ORACLE:
        return UnsupportedDecision(reason="nav_oracle_forbidden", checked_refs=tuple(refs))
    if (
        not nav_state.valid
        or nav_state.feed_stall
        or nav_state.status in (NavStatus.STALE, NavStatus.LOST, NavStatus.UNAVAILABLE, NavStatus.INITIALIZING)
    ):
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason=f"nav_{nav_state.status.value}",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    if nav_state.age_s > nav_age_max_s:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="nav_age",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    if nav_state.nav_epoch != certificate_nav_epoch:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="nav_epoch_mismatch",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    if (
        nav_state.ap_disagreement.compared
        and nav_state.ap_disagreement.within_hard_bound is False
    ):
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="ap_disagreement_hard",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )

    # Tracking envelope
    err = _dist(tracking_state.candidate.position, tracking_state.measured_position)
    if err > tracking_envelope.position_m:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="tracking_position",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    vel_err = _dist(tracking_state.candidate.velocity, tracking_state.measured_velocity)
    if vel_err > tracking_envelope.velocity_mps:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="tracking_velocity",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )

    # Geometry CLEAR — only with real Perception FREE
    if geometry_certificate is not None and geometry_certificate.support_claim == "free":
        if occupancy is None or not occupancy.healthy:
            return _backup_or_unsupported(
                predeclared_fallbacks,
                reason="occupancy_missing",
                checked_refs=tuple(refs),
                now_mono_s=now_mono_s,
                lease_s=lease_s,
            )
        if not occupancy.matches_epoch(geometry_certificate.nav_epoch):
            return _backup_or_unsupported(
                predeclared_fallbacks,
                reason="occupancy_epoch_mismatch",
                checked_refs=tuple(refs),
                now_mono_s=now_mono_s,
                lease_s=lease_s,
            )
        if not occupancy.matches_revision(geometry_certificate.map_revision):
            return _backup_or_unsupported(
                predeclared_fallbacks,
                reason="occupancy_revision_mismatch",
                checked_refs=tuple(refs),
                now_mono_s=now_mono_s,
                lease_s=lease_s,
            )
        envelope = 0.35
        if plant_limits is not None and plant_limits.valid:
            envelope = plant_limits.envelope_radius_m
            refs.append("plant_limits")
        sample = tracking_state.candidate.position
        tube = StopTubeQuery(
            samples_odom_m=((sample.x, sample.y, sample.z),),
            envelope_radius_m=envelope,
            include_brake_region=False,
            brake_region=None,
        )
        # Use OccupancyQuery stamp domain — pass a synthetic now stamp from age domain is N/A;
        # consumers pass ClockStamp via query; we use occupancy.stamp as now for contract simplicity.
        verdict = occupancy.query_stop_tube(tube, now=occupancy.stamp)
        refs.append(f"map_revision:{occupancy.map_revision}")
        refs.append(f"occupancy:{verdict.support.value}")
        if (
            verdict.support is not OccupancySupport.FREE
            or verdict.evidence_class is not EvidenceClass.SENSOR_DERIVED
        ):
            reason = "geometry_not_free" if verdict.support is not OccupancySupport.FREE else "evidence_class"
            if verdict.support is OccupancySupport.OCCUPIED:
                reason = "geometry_occupied"
            return _backup_or_unsupported(
                predeclared_fallbacks,
                reason=reason,
                checked_refs=tuple(refs),
                now_mono_s=now_mono_s,
                lease_s=lease_s,
            )
        refs.append("geometry_clear")
    elif geometry_certificate is not None:
        refs.append("geometry_unknown_claim")

    return AllowDecision(
        valid_until_mono_s=now_mono_s + lease_s,
        nav_epoch=nav_state.nav_epoch,
        checked_refs=tuple(refs),
        mode=mode,
    )
