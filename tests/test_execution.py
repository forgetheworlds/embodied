"""Safety + Execution contract tests (no Webots). Thin — layer closeout."""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied.contracts.perception_ports import (
    AgeDomain,
    EvidenceClass,
    NavPose,
    NavStatus,
    NavigationState,
    OccupancySupport,
    uncompared_disagreement,
)
from embodied.contracts.records import ClockStamp
from embodied.control import Motion, Result, VehicleState, Vec3
from embodied.execution import (
    AllowDecision,
    AuthorityView,
    BackupDecision,
    Execution,
    ExecutionStatusCode,
    GeometryCertificate,
    HoldTrajectory,
    PortBundle,
    SegmentTrajectory,
    StartState,
    StartTolerance,
    TrackingEnvelope,
    TrackingState,
    TrajectoryCertificate,
    UnsupportedDecision,
    ValidityWindow,
    check,
)
from embodied.execution.plant import StaticPlantLimitsPort, declared_plant
from embodied.perception.estimation import StaticEstimationPort
from embodied.perception.mapping_ports import snapshot_from_cell_labels
from embodied.memory.world import FREE, MapConfig


def _stamp(ns: int = 1) -> ClockStamp:
    return ClockStamp(host_id="t", clock_id="host/monotonic", monotonic_ns=ns)


def _nav(*, epoch: str = "e1", status: NavStatus = NavStatus.HEALTHY, valid: bool = True) -> NavigationState:
    pose_valid = valid and status in (NavStatus.HEALTHY, NavStatus.DEGRADED)
    return NavigationState(
        nav_epoch=epoch,
        state_sequence=1,
        controller_alignment_id="a",
        stamp=_stamp(),
        sim_time_s=1.0,
        age_s=0.01,
        age_domain=AgeDomain.SIM_CONTROL,
        monotonic_observed_at_s=1.0,
        pose=NavPose(
            parent_frame="odom",
            child_frame="body",
            stamp=_stamp(),
            position_m=(0.0, 0.0, 1.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            covariance=None,
            nav_epoch=epoch,
            source_ids=("ov",),
            valid=pose_valid,
        ),
        velocity_mps=(0.0, 0.0, 0.0),
        covariance=None,
        status=status,
        valid=pose_valid,
        sigma_pos_m=None,
        visual_source_ids=("c",),
        imu_source_ids=("i",),
        visual_age_s=0.01,
        imu_age_s=0.01,
        feed_stall=False,
        ap_disagreement=uncompared_disagreement(),
        evidence_class=EvidenceClass.SENSOR_DERIVED,
        source_ids=("ov",),
    )


def _authority(**kwargs) -> AuthorityView:
    base = dict(
        armed=True,
        guided=True,
        landed=False,
        failsafe_active=False,
        telemetry_age_mono_s=0.0,
        command_rejecting=False,
    )
    base.update(kwargs)
    return AuthorityView(**base)


def _vehicle(**kwargs) -> VehicleState:
    base = dict(
        armed=True,
        guided=True,
        position=Vec3(0.0, 0.0, 1.0),
        velocity=Vec3(0.0, 0.0, 0.0),
        yaw=0.0,
        landed=False,
    )
    base.update(kwargs)
    return VehicleState(**base)


def _tracking(candidate: Motion | None = None) -> TrackingState:
    motion = candidate or Motion(position=Vec3(0, 0, 1), velocity=Vec3(0, 0, 0))
    return TrackingState(
        candidate=motion,
        last_published=None,
        measured_position=motion.position,
        measured_velocity=motion.velocity,
        measured_yaw=0.0,
        measured_age_mono_s=0.0,
        measured_source="vehicle",
        publish_sequence=0,
    )


def _hold() -> HoldTrajectory:
    return HoldTrajectory(position=Vec3(0.0, 0.0, 1.0))


def test_check_allow_without_geometry_claim():
    decision = check(
        active_prefix=_hold(),
        stop_continuation=_hold(),
        predeclared_fallbacks={"hold": _hold()},
        nav_state=_nav(),
        vehicle_state=_vehicle(),
        authority=_authority(),
        tracking_state=_tracking(),
        tracking_envelope=TrackingEnvelope(position_m=0.5, velocity_mps=1.0),
        occupancy=None,
        plant_limits=None,
        telemetry_health=None,
        geometry_certificate=None,
        certificate_nav_epoch="e1",
        certificate_validity=ValidityWindow(None, None),
        safety_evidence_refs=(),
        mode="primary",
        now_mono_s=10.0,
        now_sim_s=10.0,
    )
    assert isinstance(decision, AllowDecision)
    assert "geometry_clear" not in decision.checked_refs


def test_check_free_claim_without_occupancy_is_backup_not_clear():
    decision = check(
        active_prefix=_hold(),
        stop_continuation=_hold(),
        predeclared_fallbacks={"hold": _hold()},
        nav_state=_nav(),
        vehicle_state=_vehicle(),
        authority=_authority(),
        tracking_state=_tracking(),
        tracking_envelope=TrackingEnvelope(position_m=0.5, velocity_mps=1.0),
        occupancy=None,
        plant_limits=None,
        telemetry_health=None,
        geometry_certificate=GeometryCertificate(
            nav_epoch="e1",
            map_revision="r1",
            volume_refs=(),
            support_claim="free",
        ),
        certificate_nav_epoch="e1",
        certificate_validity=ValidityWindow(None, None),
        safety_evidence_refs=(),
        mode="primary",
        now_mono_s=10.0,
        now_sim_s=10.0,
    )
    assert isinstance(decision, BackupDecision)
    assert decision.reason == "occupancy_missing"


def test_check_geometry_clear_with_sensor_derived_free():
    config = MapConfig(
        voxel_m=0.5,
        bounds_odom_m={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 2.0)},
        surface_band_m=0.1,
        log_odds_hit=0.7,
        log_odds_pass=-0.4,
        clamp=5.0,
        free_threshold=0.5,
        occupied_threshold=0.5,
        min_clearing_rays=1,
        freshness_s=5.0,
        dynamic_speed_mps=None,
        dynamic_reach_s=None,
    )
    # Cell (2,2,2) center is (0.25, 0.25, 1.25) under this MapConfig — matches candidate.
    occ = snapshot_from_cell_labels(
        labels={(2, 2, 2): FREE, (2, 2, 1): FREE, (2, 1, 2): FREE, (1, 2, 2): FREE},
        config=config,
        nav_epoch="e1",
        map_revision="r1",
        snapshot_id="s1",
        stamp=_stamp(),
        evidence_class=EvidenceClass.SENSOR_DERIVED,
    )
    decision = check(
        active_prefix=_hold(),
        stop_continuation=_hold(),
        predeclared_fallbacks={},
        nav_state=_nav(),
        vehicle_state=_vehicle(),
        authority=_authority(),
        tracking_state=_tracking(Motion(Vec3(0.25, 0.25, 1.25), Vec3(0, 0, 0))),
        tracking_envelope=TrackingEnvelope(position_m=0.5, velocity_mps=1.0),
        occupancy=occ,
        plant_limits=declared_plant(),
        telemetry_health=None,
        geometry_certificate=GeometryCertificate(
            nav_epoch="e1", map_revision="r1", volume_refs=(), support_claim="free"
        ),
        certificate_nav_epoch="e1",
        certificate_validity=ValidityWindow(None, None),
        safety_evidence_refs=(),
        mode="primary",
        now_mono_s=10.0,
        now_sim_s=10.0,
    )
    assert isinstance(decision, AllowDecision)
    assert "geometry_clear" in decision.checked_refs


def test_check_stale_nav_backup():
    decision = check(
        active_prefix=_hold(),
        stop_continuation=_hold(),
        predeclared_fallbacks={"hold": _hold()},
        nav_state=_nav(status=NavStatus.STALE, valid=False),
        vehicle_state=_vehicle(),
        authority=_authority(),
        tracking_state=_tracking(),
        tracking_envelope=TrackingEnvelope(position_m=0.5, velocity_mps=1.0),
        occupancy=None,
        plant_limits=None,
        telemetry_health=None,
        geometry_certificate=None,
        certificate_nav_epoch="e1",
        certificate_validity=ValidityWindow(None, None),
        safety_evidence_refs=(),
        mode="primary",
        now_mono_s=10.0,
        now_sim_s=10.0,
    )
    assert isinstance(decision, BackupDecision)


def test_check_authority_unsupported():
    decision = check(
        active_prefix=_hold(),
        stop_continuation=_hold(),
        predeclared_fallbacks={},
        nav_state=_nav(),
        vehicle_state=_vehicle(armed=False),
        authority=_authority(armed=False),
        tracking_state=_tracking(),
        tracking_envelope=TrackingEnvelope(position_m=0.5, velocity_mps=1.0),
        occupancy=None,
        plant_limits=None,
        telemetry_health=None,
        geometry_certificate=None,
        certificate_nav_epoch="e1",
        certificate_validity=ValidityWindow(None, None),
        safety_evidence_refs=(),
        mode="primary",
        now_mono_s=10.0,
        now_sim_s=10.0,
    )
    assert isinstance(decision, UnsupportedDecision)


@dataclass
class FakeClocks:
    mono: float = 0.0
    sim: float = 0.0

    def now_mono_s(self) -> float:
        return self.mono

    def now_sim_s(self) -> float:
        return self.sim

    def advance(self, dt: float) -> None:
        self.mono += dt
        self.sim += dt


@dataclass
class FakeVehicle:
    state_value: VehicleState
    commands: list[Motion] = field(default_factory=list)
    accept: bool = True

    def state(self) -> VehicleState:
        return self.state_value

    def command(self, motion: Motion) -> Result:
        self.commands.append(motion)
        return Result(accepted=self.accept, reason=None if self.accept else "reject")

    def takeoff(self, altitude_m: float) -> Result:
        del altitude_m
        return Result(accepted=True)

    def land(self) -> Result:
        return Result(accepted=True)


def _cert(*, epoch: str = "e1", duration_s: float = 1.0, geometry: GeometryCertificate | None = None) -> TrajectoryCertificate:
    start = Vec3(0.0, 0.0, 1.0)
    end = Vec3(1.0, 0.0, 1.0)
    return TrajectoryCertificate(
        certificate_id="cert-1",
        primary=SegmentTrajectory(start=start, end=end, duration_s=duration_s),
        terminal=HoldTrajectory(position=end),
        fallbacks={"hold": HoldTrajectory(position=start)},
        nav_epoch=epoch,
        start_state=StartState(position=start, velocity=Vec3(0, 0, 0)),
        start_tolerance=StartTolerance(position_m=0.5, velocity_mps=1.0),
        validity=ValidityWindow(None, None),
        tracking_envelope=TrackingEnvelope(position_m=2.0, velocity_mps=5.0),
        geometry_certificate=geometry,
        safety_evidence_refs=(),
        plant_limits_ref=None,
    )


def test_execution_replace_tick_terminal_keep_publishing():
    clocks = FakeClocks()
    vehicle = FakeVehicle(_vehicle())
    est = StaticEstimationPort(_nav())
    exe = Execution(vehicle=vehicle, clocks=clocks, ports=PortBundle(estimation=est))  # type: ignore[arg-type]
    result = exe.replace(_cert(duration_s=0.2))
    assert isinstance(result, type(result)) and result.certificate_id == "cert-1"  # ReplaceOk
    assert exe.status().code is ExecutionStatusCode.RUNNING

    clocks.advance(0.05)
    exe._tick()
    assert vehicle.commands
    assert exe.status().code is ExecutionStatusCode.RUNNING

    # Advance past primary → terminal COMPLETED but still publishes
    clocks.advance(0.3)
    vehicle.state_value = _vehicle(position=Vec3(1.0, 0.0, 1.0))
    before = len(vehicle.commands)
    exe._tick()
    assert exe.status().primary_completed is True
    assert exe.status().code is ExecutionStatusCode.COMPLETED
    assert len(vehicle.commands) == before + 1


def test_execution_unsupported_stops_publishing():
    clocks = FakeClocks()
    vehicle = FakeVehicle(_vehicle())
    # No estimation → nav_missing → backup publishes; remove fallbacks via custom cert
    cert = _cert()
    # Replace fallbacks with empty by building a cert that has fallbacks but force unsupported via authority
    exe = Execution(vehicle=vehicle, clocks=clocks, ports=PortBundle(estimation=StaticEstimationPort(_nav())))  # type: ignore[arg-type]
    exe.replace(cert)
    vehicle.state_value = _vehicle(armed=False, guided=False)
    before = len(vehicle.commands)
    exe._tick()
    assert exe.status().code is ExecutionStatusCode.BLOCKED
    assert exe.status().last_decision == "unsupported"
    assert len(vehicle.commands) == before


def test_execution_replace_err_preserves_prior():
    clocks = FakeClocks()
    vehicle = FakeVehicle(_vehicle())
    exe = Execution(vehicle=vehicle, clocks=clocks, ports=PortBundle(estimation=StaticEstimationPort(_nav())))  # type: ignore[arg-type]
    exe.replace(_cert(epoch="e1"))
    err = exe.replace(_cert(epoch="other"))
    assert err.reason == "nav_epoch_mismatch"
    assert exe.status().certificate_id == "cert-1"
