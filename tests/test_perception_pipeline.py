"""No-sim: MapStoreMappingPort integrate → sensor_derived FREE (and refuse pose_assisted)."""

from __future__ import annotations

import numpy as np
import pytest

from embodied.contracts.perception_ports import (
    Aabb,
    AgeDomain,
    EvidenceClass,
    NavPose,
    NavStatus,
    NavigationState,
    OccupancySupport,
    uncompared_disagreement,
)
from embodied.contracts.records import ClockStamp
from embodied.memory.world import FREE
from embodied.perception.camera import (
    REASON_VALID,
    PoseProvenance,
    DepthProduct,
    build_calibration,
)
from embodied.perception.estimation import pose_estimate_from_nav
from embodied.perception.mapping_ports import MapStoreMappingPort
from embodied.perception.pipeline import PerceptionPipeline, default_map_config
from embodied.perception.stereo_imu_nav import StereoImuNav
from embodied.memory.world import MapStore


def _stamp(ns: int = 1_000_000_000) -> ClockStamp:
    return ClockStamp(host_id="test", clock_id="host/monotonic", monotonic_ns=ns)


def _nav(*, evidence: EvidenceClass, valid: bool = True) -> NavigationState:
    pose = NavPose(
        parent_frame="odom",
        child_frame="body",
        stamp=_stamp(),
        position_m=(0.0, 0.0, 0.0),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance=None,
        nav_epoch="epoch-pipe",
        source_ids=("stereo_imu_nav",),
        valid=valid,
    )
    return NavigationState(
        nav_epoch="epoch-pipe",
        state_sequence=1,
        controller_alignment_id=None,
        stamp=_stamp(),
        sim_time_s=1.0,
        age_s=0.01,
        age_domain=AgeDomain.SIM_CONTROL,
        monotonic_observed_at_s=1.0,
        pose=pose,
        velocity_mps=(0.0, 0.0, 0.0),
        covariance=None,
        status=NavStatus.HEALTHY if valid else NavStatus.STALE,
        valid=valid,
        sigma_pos_m=None,
        visual_source_ids=("stereo_pair",),
        imu_source_ids=("imu_accel", "imu_gyro"),
        visual_age_s=0.01,
        imu_age_s=0.01,
        feed_stall=False,
        ap_disagreement=uncompared_disagreement(),
        evidence_class=evidence,
        source_ids=("stereo_imu_nav",),
    )


def _planar_depth(*, z_m: float = 2.0, height: int = 48, width: int = 64) -> DepthProduct:
    """Synthetic valid depth plane ahead of the camera (no Webots)."""
    cal = build_calibration()
    depth = np.full((height, width), z_m, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    # Leave a thin border invalid so border filter semantics stay honest.
    valid[:2, :] = False
    valid[-2:, :] = False
    valid[:, :2] = False
    valid[:, -2:] = False
    reasons = np.zeros((height, width), dtype=np.uint8)
    reasons[valid] = REASON_VALID
    disparity = np.full((height, width), np.nan, dtype=np.float32)
    focal = float(cal.left_intrinsics.focal_length_px[0])
    disparity[valid] = (focal * cal.baseline_m) / z_m
    uncertainty = np.full((height, width), np.nan, dtype=np.float32)
    uncertainty[valid] = (z_m**2) * 1.0 / (focal * cal.baseline_m)
    return DepthProduct(
        calibration_id=cal.calibration_id,
        calibration_version=cal.version,
        pair_id="synthetic-1",
        capture_stamp=_stamp(),
        receipt_stamp=_stamp(),
        sim_time_s=1.0,
        pose_provenance=PoseProvenance(
            label="SENSOR_DERIVED",
            detail="unit-test synthetic depth with sensor_derived capture pose",
        ),
        frame="test",
        disparity_px=disparity,
        depth_m=depth,
        valid=valid,
        reasons=reasons,
        uncertainty_m=uncertainty,
        texture_threshold_applied=False,
    )


def test_integrate_sensor_derived_emits_free():
    cal = build_calibration()
    store = MapStore(config=default_map_config(), submap_id="t", nav_epoch="epoch-pipe")
    mapping = MapStoreMappingPort(
        store,
        nav_epoch="epoch-pipe",
        evidence_class=EvidenceClass.SENSOR_DERIVED,
        stamp=_stamp(),
    )
    nav = _nav(evidence=EvidenceClass.SENSOR_DERIVED)
    depth = _planar_depth()
    mapping.integrate(
        depth,
        pose_estimate_from_nav(nav),
        cal,
        stamp_ns=1_000_000_000,
        observation_id="obs-1",
        now_ns=1_000_000_000,
        nav=nav,
    )
    query = mapping.occupancy()
    assert query is not None
    assert query.evidence_class is EvidenceClass.SENSOR_DERIVED
    assert query.healthy is True
    assert any(label == FREE for _, label in query.cells)
    verdict = query.query_volume(
        Aabb(min_m=(0.5, -0.5, -0.5), max_m=(1.8, 0.5, 0.5), frame="odom"),
        now=_stamp(),
    )
    assert verdict.support is OccupancySupport.FREE
    assert verdict.evidence_class is EvidenceClass.SENSOR_DERIVED


def test_integrate_pose_assisted_nav_refuses_free_path():
    cal = build_calibration()
    store = MapStore(config=default_map_config(), submap_id="t", nav_epoch="epoch-pipe")
    mapping = MapStoreMappingPort(
        store,
        nav_epoch="epoch-pipe",
        evidence_class=EvidenceClass.SENSOR_DERIVED,
        stamp=_stamp(),
    )
    nav = _nav(evidence=EvidenceClass.POSE_ASSISTED)
    before = store.revision
    mapping.integrate(
        _planar_depth(),
        pose_estimate_from_nav(nav),
        cal,
        stamp_ns=1_000_000_000,
        observation_id="obs-assisted",
        now_ns=1_000_000_000,
        nav=nav,
    )
    assert store.revision == before
    assert any("sensor_derived" in r for r in store.rejections)


def test_stereo_imu_nav_marks_sensor_derived_and_stalls():
    est = StereoImuNav(nav_epoch="e1", stall_after_s=0.05)
    for i in range(20):
        est.ingest_imu(
            accelerometer=(0.0, 0.0, 9.81),
            gyro=(0.0, 0.0, 0.0),
            capture_host_ns=1_000_000_000 + i * 5_000_000,
            sim_time_s=0.01 * i,
        )
    est.ingest_pair(capture_host_ns=1_000_100_000, sim_time_s=0.1)
    nav = est.latest(stamp=_stamp(1_000_100_000), now_host_ns=1_000_100_000)
    assert nav is not None
    assert nav.evidence_class is EvidenceClass.SENSOR_DERIVED
    assert nav.valid is True
    stalled = est.latest(stamp=_stamp(1_200_000_000), now_host_ns=1_200_000_000)
    assert stalled is not None
    assert stalled.feed_stall is True
    assert stalled.valid is False


def test_visual_age_alone_does_not_stall_imu_nav():
    """SGBM / pair gaps must not mark IMU-derived nav STALE."""
    est = StereoImuNav(nav_epoch="e1", stall_after_s=1.0)
    for i in range(10):
        est.ingest_imu(
            accelerometer=(0.0, 0.0, 9.81),
            gyro=(0.0, 0.0, 0.0),
            capture_host_ns=1_000_000_000 + i * 5_000_000,
            sim_time_s=0.005 * i,
        )
    est.ingest_pair(capture_host_ns=1_000_050_000, sim_time_s=0.05)
    # 0.5s after last IMU, 0.5s after last pair — under IMU stall, over old visual×2 rule.
    nav = est.latest(stamp=_stamp(1_000_550_000), now_host_ns=1_000_550_000)
    assert nav is not None
    assert nav.valid is True
    assert nav.feed_stall is False
    assert nav.status is NavStatus.HEALTHY


def test_latest_at_capture_uses_sensor_clock_not_wall_sgbm():
    est = StereoImuNav(nav_epoch="e1", stall_after_s=1.0, capture_imu_slop_s=0.25)
    est.ingest_imu(
        accelerometer=(0.0, 0.0, 9.81),
        gyro=(0.0, 0.0, 0.0),
        capture_host_ns=1_000_000_000,
        sim_time_s=1.0,
    )
    est.ingest_pair(capture_host_ns=1_000_050_000, sim_time_s=1.05)
    # Wall "now" is 2s later (SGBM burst), but capture is near IMU on sensor clock.
    later = 1_000_000_000 + 2_000_000_000
    cap = est.latest_at_capture(stamp=_stamp(later), capture_host_ns=1_000_050_000)
    assert cap is not None and cap.valid is True
    live = est.latest(stamp=_stamp(later), now_host_ns=later)
    assert live is not None and live.feed_stall is True


def test_pipeline_on_pair_path_uses_ports():
    """Smoke: pipeline wires EstimationPort + MappingPort (depth may be empty on noise)."""
    pipe = PerceptionPipeline(nav_epoch="epoch-pipe")
    stamp = _stamp()
    for i in range(30):
        pipe.on_imu(
            accelerometer=(0.0, 0.0, 9.81),
            gyro=(0.0, 0.0, 0.0),
            capture_host_ns=1_000_000_000 + i * 2_000_000,
            sim_time_s=0.002 * i,
            stamp=stamp,
        )
    # Uniform images → matcher may yield no valid depth; ensure no crash and nav publishes.
    h, w = 48, 64
    left = np.full((h, w, 3), 40, dtype=np.uint8)
    right = np.full((h, w, 3), 40, dtype=np.uint8)
    pipe.on_pair(
        left_rgb=left,
        right_rgb=right,
        pair_id=1,
        capture_host_ns=1_000_060_000,
        sim_time_s=0.06,
        stamp=_stamp(1_000_060_000),
    )
    nav = pipe.estimation.latest()
    assert nav is not None
    assert nav.evidence_class is EvidenceClass.SENSOR_DERIVED
