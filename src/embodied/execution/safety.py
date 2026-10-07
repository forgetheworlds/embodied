"""Pure Safety.check — ALLOW / BACKUP / UNSUPPORTED leases.

Owns the Safety surface only. Never invents Motions, never commands Vehicle.
Geometry CLEAR requires OccupancyVerdict FREE + sensor_derived + epoch match
with active nav (Perception consumer rules).

Rebuild, not glue: this module does **not** wrap or call legacy
``embodied.navigation.validator`` (or planner/geometry). It consumes frozen
Perception ports + Execution certificate/plant types only.
"""

from __future__ import annotations

from dataclasses import dataclass

from embodied.contracts.perception_ports import (
    EvidenceClass,
    NavStatus,
    NavigationState,
    OccupancyQuery,
    OccupancySupport,
    OccupancyVerdict,
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

_BAD_NAV_STATUS = (
    NavStatus.STALE,
    NavStatus.LOST,
    NavStatus.UNAVAILABLE,
    NavStatus.INITIALIZING,
)


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


def _geometry_clear(
    *,
    occupancy: OccupancyQuery,
    plant_limits: PlantLimits | None,
    geometry_certificate: GeometryCertificate,
    nav_state: NavigationState,
    candidate: Motion,
    refs: list[str],
    predeclared_fallbacks: dict[str, Trajectory],
    now_mono_s: float,
    lease_s: float,
) -> SafetyDecision | None:
    """Return a fail decision, or None when geometry CLEAR is evidenced.

    CLEAR only if verdict FREE + sensor_derived + epoch match with active nav.
    Does not invent free space. Stop-capable credit needs PlantLimits.valid.
    """
    if nav_state.evidence_class is not EvidenceClass.SENSOR_DERIVED:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="nav_evidence_class",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    if geometry_certificate.nav_epoch != nav_state.nav_epoch:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="geometry_nav_epoch_mismatch",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    if not occupancy.healthy:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="occupancy_unhealthy",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    # Active nav epoch must match snapshot (Perception consumer rule).
    if not occupancy.matches_epoch(nav_state.nav_epoch):
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
    plant_ok = plant_limits is not None and plant_limits.valid
    if plant_ok:
        assert plant_limits is not None
        envelope = plant_limits.envelope_radius_m
        refs.append("plant_limits")

    sample = candidate.position
    tube = StopTubeQuery(
        samples_odom_m=((sample.x, sample.y, sample.z),),
        envelope_radius_m=envelope,
        include_brake_region=False,
        brake_region=None,
    )
    # Re-query every check; use snapshot stamp as query `now` (freshness is on the handle).
    verdict: OccupancyVerdict = occupancy.query_stop_tube(tube, now=occupancy.stamp)
    refs.append(f"map_revision:{occupancy.map_revision}")
    refs.append(f"occupancy:{verdict.support.value}")

    if verdict.nav_epoch != nav_state.nav_epoch:
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason="verdict_epoch_mismatch",
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
    if (
        verdict.support is not OccupancySupport.FREE
        or verdict.evidence_class is not EvidenceClass.SENSOR_DERIVED
    ):
        if verdict.support is OccupancySupport.OCCUPIED:
            reason = "geometry_occupied"
        elif verdict.evidence_class is not EvidenceClass.SENSOR_DERIVED:
            reason = "evidence_class"
        else:
            reason = "geometry_not_free"
        return _backup_or_unsupported(
            predeclared_fallbacks,
            reason=reason,
            checked_refs=tuple(refs),
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )

    refs.append("geometry_clear")
    # Rule 7: stop-capable CLEAR needs PlantLimits.valid — never claim without it.
    if plant_ok:
        refs.append("stop_capable_clear")
    return None


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
    del active_prefix, stop_continuation, now_sim_s  # traj correlation reserved for callers
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
        or nav_state.status in _BAD_NAV_STATUS
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

    # Geometry CLEAR — only with real Perception FREE (never invent).
    if geometry_certificate is not None and geometry_certificate.support_claim == "free":
        if occupancy is None:
            return _backup_or_unsupported(
                predeclared_fallbacks,
                reason="occupancy_missing",
                checked_refs=tuple(refs),
                now_mono_s=now_mono_s,
                lease_s=lease_s,
            )
        fail = _geometry_clear(
            occupancy=occupancy,
            plant_limits=plant_limits,
            geometry_certificate=geometry_certificate,
            nav_state=nav_state,
            candidate=tracking_state.candidate,
            refs=refs,
            predeclared_fallbacks=predeclared_fallbacks,
            now_mono_s=now_mono_s,
            lease_s=lease_s,
        )
        if fail is not None:
            return fail
    elif geometry_certificate is not None:
        refs.append("geometry_unknown_claim")

    return AllowDecision(
        valid_until_mono_s=now_mono_s + lease_s,
        nav_epoch=nav_state.nav_epoch,
        checked_refs=tuple(refs),
        mode=mode,
    )
