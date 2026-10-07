"""Perception / Map stacked ports — no-sim contract pytest (layer closeout).

Covers docs/perception-ports-frozen.md §9.1.
"""

from __future__ import annotations

import pytest

from embodied.contracts.perception_ports import (
    Aabb,
    AgeDomain,
    ApDisagreement,
    EvidenceClass,
    EstimationPort,
    MappingPort,
    NavPose,
    NavStatus,
    NavigationState,
    OccupancyQuery,
    OccupancySupport,
    OccupancyVerdict,
    Sphere,
    StopTubeQuery,
    uncompared_disagreement,
)
from embodied.contracts.records import ClockStamp, RecordError
from embodied.memory.world import FREE, OCCUPIED, UNKNOWN, MapConfig
from embodied.perception.estimation import (
    StaticEstimationPort,
    unavailable_navigation_state,
    with_status,
)
from embodied.perception.mapping_ports import (
    NullMappingPort,
    StubMappingPort,
    StubOccupancyQuery,
    snapshot_from_cell_labels,
)


def _stamp(ns: int = 1_000_000_000) -> ClockStamp:
    return ClockStamp(host_id="test", clock_id="host/monotonic", monotonic_ns=ns)


def _pose(*, epoch: str = "epoch-1", valid: bool = True) -> NavPose:
    return NavPose(
        parent_frame="odom",
        child_frame="body",
        stamp=_stamp(),
        position_m=(0.0, 0.0, 1.0),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance=None,
        nav_epoch=epoch,
        source_ids=("ov_stream",),
        valid=valid,
    )


def _nav(
    *,
    status: NavStatus = NavStatus.HEALTHY,
    valid: bool = True,
    feed_stall: bool = False,
    epoch: str = "epoch-1",
    evidence_class: EvidenceClass = EvidenceClass.SENSOR_DERIVED,
    disagreement: ApDisagreement | None = None,
) -> NavigationState:
    pose_valid = valid
    if status in (NavStatus.STALE, NavStatus.LOST, NavStatus.UNAVAILABLE) or feed_stall:
        valid = False
        pose_valid = False
        if feed_stall and status not in (NavStatus.STALE, NavStatus.LOST):
            status = NavStatus.STALE
    return NavigationState(
        nav_epoch=epoch,
        state_sequence=1,
        controller_alignment_id="align-1",
        stamp=_stamp(),
        sim_time_s=1.25,
        age_s=0.05,
        age_domain=AgeDomain.SIM_CONTROL,
        monotonic_observed_at_s=10.0,
        pose=_pose(epoch=epoch, valid=pose_valid),
        velocity_mps=(0.1, 0.0, 0.0),
        covariance=None,
        status=status,
        valid=valid,
        sigma_pos_m=(0.02, 0.02, 0.03),
        visual_source_ids=("cam_l", "cam_r"),
        imu_source_ids=("imu0",),
        visual_age_s=0.02,
        imu_age_s=0.01,
        feed_stall=feed_stall,
        ap_disagreement=disagreement if disagreement is not None else uncompared_disagreement(),
        evidence_class=evidence_class,
        source_ids=("ov_stream",),
    )


def _tiny_config() -> MapConfig:
    return MapConfig(
        voxel_m=0.5,
        bounds_odom_m={"x": (0.0, 2.0), "y": (0.0, 2.0), "z": (0.0, 2.0)},
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


# ---------------------------------------------------------------------------
# §9.1.1 method surface
# ---------------------------------------------------------------------------


def test_estimation_and_mapping_port_surface_exists():
    est: EstimationPort = StaticEstimationPort(None)
    mapping: MappingPort = StubMappingPort()
    query: OccupancyQuery = StubOccupancyQuery()
    assert est.latest() is None
    assert est.current_epoch() is None
    assert est.healthy() is False
    assert mapping.occupancy() is not None
    assert mapping.current_revision() == "stub-revision"
    now = _stamp()
    for method in (
        query.query_volume(Aabb(min_m=(0, 0, 0), max_m=(1, 1, 1), frame="odom"), now=now),
        query.query_stop_tube(
            StopTubeQuery(
                samples_odom_m=((0.5, 0.5, 0.5),),
                envelope_radius_m=0.2,
                include_brake_region=False,
                brake_region=None,
            ),
            now=now,
        ),
        query.query_cells(((0, 0, 0),), now=now),
        query.classify_cell((0, 0, 0), now=now),
        query.query_fov_clearance((0, 0, 0), (1, 0, 0), 2.0, now=now),
        query.query_unknown_intrusion(
            Sphere(center_m=(0.5, 0.5, 0.5), radius_m=0.5, frame="odom"),
            dynamic_speed_mps=None,
            dynamic_reach_s=None,
            now=now,
        ),
    ):
        assert method.support is OccupancySupport.UNSUPPORTED
        assert method.support is not OccupancySupport.FREE
    assert query.dynamic_envelopes(horizon_s=1.0, now=now) == ()
    assert query.matches_epoch("stub-epoch")
    assert query.matches_revision("stub-revision")


# ---------------------------------------------------------------------------
# §9.1.2 NavigationState fields + status invariants
# ---------------------------------------------------------------------------


def test_navigation_state_all_fields_constructible():
    state = _nav()
    assert state.valid is True
    assert state.status is NavStatus.HEALTHY
    assert state.pose.parent_frame == "odom"
    assert state.evidence_class is EvidenceClass.SENSOR_DERIVED


@pytest.mark.parametrize("status", [NavStatus.STALE, NavStatus.LOST, NavStatus.UNAVAILABLE])
def test_unhealthy_status_requires_valid_false(status: NavStatus):
    state = _nav(status=status, valid=False)
    assert state.valid is False
    assert state.pose.valid is False
    with pytest.raises(RecordError):
        NavigationState(
            nav_epoch="epoch-1",
            state_sequence=1,
            controller_alignment_id=None,
            stamp=_stamp(),
            sim_time_s=None,
            age_s=0.0,
            age_domain=AgeDomain.MONOTONIC,
            monotonic_observed_at_s=None,
            pose=_pose(valid=True),
            velocity_mps=None,
            covariance=None,
            status=status,
            valid=True,
            sigma_pos_m=None,
            visual_source_ids=(),
            imu_source_ids=(),
            visual_age_s=None,
            imu_age_s=None,
            feed_stall=False,
            ap_disagreement=uncompared_disagreement(),
            evidence_class=EvidenceClass.SENSOR_DERIVED,
            source_ids=("x",),
        )


def test_feed_stall_not_healthy():
    state = _nav(feed_stall=True)
    assert state.status is NavStatus.STALE
    assert state.valid is False
    port = StaticEstimationPort(state)
    assert port.healthy() is False
    assert with_status(_nav(), NavStatus.HEALTHY, feed_stall=True).status is NavStatus.STALE


def test_oracle_evidence_class_forbidden():
    with pytest.raises(RecordError):
        _nav(evidence_class=EvidenceClass.ORACLE)


# ---------------------------------------------------------------------------
# §9.1.3 ApDisagreement
# ---------------------------------------------------------------------------


def test_ap_disagreement_uncompared_and_filled():
    empty = uncompared_disagreement()
    assert empty.compared is False
    filled = ApDisagreement(
        compared=True,
        position_err_m=0.15,
        velocity_err_mps=0.05,
        yaw_err_rad=0.02,
        ap_stamp=_stamp(2),
        estimator_stamp=_stamp(2),
        within_soft_bound=True,
        within_hard_bound=True,
    )
    state = _nav(disagreement=filled)
    assert state.ap_disagreement.compared is True
    # Port has no averaging helper — only the record fields.
    assert not hasattr(StaticEstimationPort, "average_with_ap")


# ---------------------------------------------------------------------------
# §9.1.4 Epoch bump
# ---------------------------------------------------------------------------


def test_epoch_match_and_pose_epoch_equality():
    query = snapshot_from_cell_labels(
        labels={(0, 0, 0): FREE},
        config=_tiny_config(),
        nav_epoch="epoch-a",
        map_revision="rev-1",
        snapshot_id="snap-1",
        stamp=_stamp(),
        evidence_class=EvidenceClass.SENSOR_DERIVED,
    )
    assert query.matches_epoch("epoch-a")
    assert not query.matches_epoch("epoch-b")
    assert query.matches_revision("rev-1")
    assert not query.matches_revision("rev-2")
    with pytest.raises(RecordError):
        NavigationState(
            nav_epoch="epoch-a",
            state_sequence=0,
            controller_alignment_id=None,
            stamp=_stamp(),
            sim_time_s=None,
            age_s=0.0,
            age_domain=AgeDomain.SIM_CONTROL,
            monotonic_observed_at_s=None,
            pose=_pose(epoch="epoch-b"),
            velocity_mps=None,
            covariance=None,
            status=NavStatus.HEALTHY,
            valid=True,
            sigma_pos_m=None,
            visual_source_ids=(),
            imu_source_ids=(),
            visual_age_s=None,
            imu_age_s=None,
            feed_stall=False,
            ap_disagreement=uncompared_disagreement(),
            evidence_class=EvidenceClass.SENSOR_DERIVED,
            source_ids=("x",),
        )


# ---------------------------------------------------------------------------
# §9.1.5 Stub occupancy ≠ free; empty map ≠ free world
# ---------------------------------------------------------------------------


def test_stub_and_null_mapping_never_free():
    stub = StubMappingPort()
    query = stub.occupancy()
    assert query is not None
    now = _stamp()
    verdict = query.query_volume(Aabb(min_m=(0, 0, 0), max_m=(1, 1, 1), frame="odom"), now=now)
    assert verdict.support is OccupancySupport.UNSUPPORTED
    assert NullMappingPort().occupancy() is None


# ---------------------------------------------------------------------------
# §9.1.6 FREE requires sensor_derived
# ---------------------------------------------------------------------------


def test_free_verdict_requires_sensor_derived_evidence_class():
    labels = {(0, 0, 0): FREE, (1, 0, 0): FREE}
    assisted = snapshot_from_cell_labels(
        labels=labels,
        config=_tiny_config(),
        nav_epoch="epoch-1",
        map_revision="rev-1",
        snapshot_id="snap-1",
        stamp=_stamp(),
        evidence_class=EvidenceClass.POSE_ASSISTED,
    )
    now = _stamp()
    verdict = assisted.classify_cell((0, 0, 0), now=now)
    assert verdict.support is OccupancySupport.UNSUPPORTED
    assert verdict.reason == "evidence_class"
    with pytest.raises(RecordError):
        OccupancyVerdict(
            support=OccupancySupport.FREE,
            reason="free",
            nav_epoch="e",
            map_revision="r",
            snapshot_id="s",
            query_stamp=now,
            sim_time_s=None,
            age_s=0.0,
            age_domain=AgeDomain.SIM_CONTROL,
            evidence_class=EvidenceClass.POSE_ASSISTED,
            limiting_refs=(),
            free_fraction=1.0,
            capabilities=("classify_cell",),
            limitations=(),
        )
    sensor = snapshot_from_cell_labels(
        labels=labels,
        config=_tiny_config(),
        nav_epoch="epoch-1",
        map_revision="rev-1",
        snapshot_id="snap-1",
        stamp=_stamp(),
        evidence_class=EvidenceClass.SENSOR_DERIVED,
    )
    free = sensor.classify_cell((0, 0, 0), now=now)
    assert free.support is OccupancySupport.FREE
    assert free.evidence_class is EvidenceClass.SENSOR_DERIVED


# ---------------------------------------------------------------------------
# §9.1.7 query_cells precedence + intrusion without bound
# ---------------------------------------------------------------------------


def test_query_cells_precedence_occupied_over_unknown_over_free():
    query = snapshot_from_cell_labels(
        labels={
            (0, 0, 0): FREE,
            (1, 0, 0): UNKNOWN,
            (0, 1, 0): OCCUPIED,
        },
        config=_tiny_config(),
        nav_epoch="epoch-1",
        map_revision="rev-1",
        snapshot_id="snap-1",
        stamp=_stamp(),
        evidence_class=EvidenceClass.SENSOR_DERIVED,
    )
    now = _stamp()
    mixed = query.query_cells(((0, 0, 0), (1, 0, 0), (0, 1, 0)), now=now)
    assert mixed.support is OccupancySupport.OCCUPIED
    unknownish = query.query_cells(((0, 0, 0), (1, 0, 0)), now=now)
    assert unknownish.support is OccupancySupport.UNKNOWN
    free = query.query_cells(((0, 0, 0),), now=now)
    assert free.support is OccupancySupport.FREE
    empty = query.query_cells((), now=now)
    assert empty.support is OccupancySupport.UNSUPPORTED
    assert empty.reason == "other"

    intrusion = query.query_unknown_intrusion(
        Aabb(min_m=(0, 0, 0), max_m=(2, 2, 2), frame="odom"),
        dynamic_speed_mps=None,
        dynamic_reach_s=None,
        now=now,
    )
    assert intrusion.support is OccupancySupport.UNSUPPORTED
    assert intrusion.reason == "no_dynamic_bound"


def test_frame_mismatch_unsupported():
    query = snapshot_from_cell_labels(
        labels={(0, 0, 0): FREE},
        config=_tiny_config(),
        nav_epoch="epoch-1",
        map_revision="rev-1",
        snapshot_id="snap-1",
        stamp=_stamp(),
        evidence_class=EvidenceClass.SENSOR_DERIVED,
    )
    verdict = query.query_volume(
        Aabb(min_m=(0, 0, 0), max_m=(1, 1, 1), frame="ned"),
        now=_stamp(),
    )
    assert verdict.support is OccupancySupport.UNSUPPORTED
    assert verdict.reason == "frame_mismatch"


# ---------------------------------------------------------------------------
# §9.1.8 latest None vs unavailable
# ---------------------------------------------------------------------------


def test_latest_none_distinct_from_unavailable_state():
    unwired = StaticEstimationPort(None)
    assert unwired.latest() is None
    assert unwired.current_epoch() is None
    assert unwired.healthy() is False

    up = StaticEstimationPort(unavailable_navigation_state(nav_epoch="epoch-1", stamp=_stamp()))
    state = up.latest()
    assert state is not None
    assert state.status is NavStatus.UNAVAILABLE
    assert state.valid is False
    assert up.healthy() is False
    assert up.current_epoch() == "epoch-1"


def test_degraded_still_healthy_enough_for_port():
    port = StaticEstimationPort(_nav(status=NavStatus.DEGRADED))
    assert port.healthy() is True


def test_volume_and_stop_tube_over_labeled_snapshot():
    query = snapshot_from_cell_labels(
        labels={(0, 0, 0): FREE, (1, 0, 0): OCCUPIED},
        config=_tiny_config(),
        nav_epoch="epoch-1",
        map_revision="rev-1",
        snapshot_id="snap-1",
        stamp=_stamp(),
        evidence_class=EvidenceClass.SENSOR_DERIVED,
    )
    now = _stamp()
    free_vol = query.query_volume(
        Aabb(min_m=(0.0, 0.0, 0.0), max_m=(0.6, 0.6, 0.6), frame="odom"),
        now=now,
    )
    assert free_vol.support is OccupancySupport.FREE
    tube = query.query_stop_tube(
        StopTubeQuery(
            samples_odom_m=((0.25, 0.25, 0.25), (0.75, 0.25, 0.25)),
            envelope_radius_m=0.3,
            include_brake_region=True,
            brake_region=Sphere(center_m=(0.75, 0.25, 0.25), radius_m=0.3, frame="odom"),
        ),
        now=now,
    )
    assert tube.support is OccupancySupport.OCCUPIED
