"""Safety.check contract tests (no Webots). Thin — CLEAR honesty only."""

from __future__ import annotations

from embodied.contracts.perception_ports import (
    AgeDomain,
    EvidenceClass,
    NavPose,
    NavStatus,
    NavigationState,
    uncompared_disagreement,
)
from embodied.contracts.records import ClockStamp
from embodied.control import Motion, VehicleState, Vec3
from embodied.execution import (
    AllowDecision,
    AuthorityView,
    BackupDecision,
    GeometryCertificate,
    HoldTrajectory,
    TrackingEnvelope,
    TrackingState,
    UnsupportedDecision,
    ValidityWindow,
    check,
)
from embodied.execution.plant import declared_plant
from embodied.memory.world import FREE, MapConfig
from embodied.perception.mapping_ports import snapshot_from_cell_labels


def _stamp(ns: int = 1) -> ClockStamp:
    return ClockStamp(host_id="t", clock_id="host/monotonic", monotonic_ns=ns)


def _nav(
    *,
    epoch: str = "e1",
    status: NavStatus = NavStatus.HEALTHY,
    valid: bool = True,
    evidence: EvidenceClass = EvidenceClass.SENSOR_DERIVED,
) -> NavigationState:
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
        evidence_class=evidence,
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


def _free_occ(*, epoch: str = "e1", revision: str = "r1", evidence: EvidenceClass = EvidenceClass.SENSOR_DERIVED):
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
    return snapshot_from_cell_labels(
        labels={(2, 2, 2): FREE, (2, 2, 1): FREE, (2, 1, 2): FREE, (1, 2, 2): FREE},
        config=config,
        nav_epoch=epoch,
        map_revision=revision,
        snapshot_id="s1",
        stamp=_stamp(),
        evidence_class=evidence,
    )


def _check(**kwargs):
    base = dict(
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
    base.update(kwargs)
    return check(**base)


def test_allow_without_geometry_claim_omits_clear():
    decision = _check()
    assert isinstance(decision, AllowDecision)
    assert "geometry_clear" not in decision.checked_refs
    assert "stop_capable_clear" not in decision.checked_refs


def test_free_claim_without_occupancy_is_backup():
    decision = _check(
        geometry_certificate=GeometryCertificate(
            nav_epoch="e1", map_revision="r1", volume_refs=(), support_claim="free"
        ),
    )
    assert isinstance(decision, BackupDecision)
    assert decision.reason == "occupancy_missing"
    assert "geometry_clear" not in decision.checked_refs


def test_geometry_clear_requires_sensor_derived_free_and_epoch():
    decision = _check(
        predeclared_fallbacks={},
        tracking_state=_tracking(Motion(Vec3(0.25, 0.25, 1.25), Vec3(0, 0, 0))),
        occupancy=_free_occ(),
        plant_limits=declared_plant(),
        geometry_certificate=GeometryCertificate(
            nav_epoch="e1", map_revision="r1", volume_refs=(), support_claim="free"
        ),
    )
    assert isinstance(decision, AllowDecision)
    assert "geometry_clear" in decision.checked_refs
    assert "stop_capable_clear" in decision.checked_refs


def test_pose_assisted_nav_blocks_geometry_clear():
    decision = _check(
        nav_state=_nav(evidence=EvidenceClass.POSE_ASSISTED),
        tracking_state=_tracking(Motion(Vec3(0.25, 0.25, 1.25), Vec3(0, 0, 0))),
        occupancy=_free_occ(),
        plant_limits=declared_plant(),
        geometry_certificate=GeometryCertificate(
            nav_epoch="e1", map_revision="r1", volume_refs=(), support_claim="free"
        ),
    )
    assert isinstance(decision, BackupDecision)
    assert decision.reason == "nav_evidence_class"
    assert "geometry_clear" not in decision.checked_refs


def test_occupancy_epoch_mismatch_blocks_clear():
    decision = _check(
        tracking_state=_tracking(Motion(Vec3(0.25, 0.25, 1.25), Vec3(0, 0, 0))),
        occupancy=_free_occ(epoch="other"),
        plant_limits=declared_plant(),
        geometry_certificate=GeometryCertificate(
            nav_epoch="e1", map_revision="r1", volume_refs=(), support_claim="free"
        ),
    )
    assert isinstance(decision, BackupDecision)
    assert decision.reason == "occupancy_epoch_mismatch"
    assert "geometry_clear" not in decision.checked_refs


def test_pose_assisted_map_never_clears():
    decision = _check(
        tracking_state=_tracking(Motion(Vec3(0.25, 0.25, 1.25), Vec3(0, 0, 0))),
        occupancy=_free_occ(evidence=EvidenceClass.POSE_ASSISTED),
        plant_limits=declared_plant(),
        geometry_certificate=GeometryCertificate(
            nav_epoch="e1", map_revision="r1", volume_refs=(), support_claim="free"
        ),
    )
    assert isinstance(decision, BackupDecision)
    assert decision.reason == "evidence_class"
    assert "geometry_clear" not in decision.checked_refs


def test_invalid_plant_clears_geometry_but_not_stop_capable():
    decision = _check(
        predeclared_fallbacks={},
        tracking_state=_tracking(Motion(Vec3(0.25, 0.25, 1.25), Vec3(0, 0, 0))),
        occupancy=_free_occ(),
        plant_limits=declared_plant(valid=False),
        geometry_certificate=GeometryCertificate(
            nav_epoch="e1", map_revision="r1", volume_refs=(), support_claim="free"
        ),
    )
    assert isinstance(decision, AllowDecision)
    assert "geometry_clear" in decision.checked_refs
    assert "stop_capable_clear" not in decision.checked_refs


def test_stale_nav_backup():
    decision = _check(nav_state=_nav(status=NavStatus.STALE, valid=False))
    assert isinstance(decision, BackupDecision)


def test_authority_unsupported():
    decision = _check(
        predeclared_fallbacks={},
        nav_state=_nav(),
        vehicle_state=_vehicle(armed=False),
        authority=_authority(armed=False),
    )
    assert isinstance(decision, UnsupportedDecision)


def test_safety_module_has_no_vehicle_command():
    import embodied.execution.safety as safety_mod
    import inspect

    src = inspect.getsource(safety_mod)
    assert "Vehicle.command" not in src
    assert ".command(" not in src


def test_safety_does_not_import_legacy_navigation():
    """prefer-rebuild: Safety must not wrap navigation.validator / planner."""
    import ast
    from pathlib import Path

    import embodied.execution.safety as safety_mod

    tree = ast.parse(Path(safety_mod.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name == "embodied.navigation" or name.startswith("embodied.navigation.") for name in imported)
    assert safety_mod.__name__ == "embodied.execution.safety"
