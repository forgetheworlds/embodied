"""Thin live Perception pipeline: stereo → depth → MapStore → OccupancyQuery.

Small rebuild on frozen ports. Reuses only ``compute_validated_depth`` and
``MapStore`` / ``MapStoreMappingPort``. No mission_runtime / navigation wrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied.contracts.perception_ports import (
    Aabb,
    AgeDomain,
    EvidenceClass,
    OccupancySupport,
    OccupancyVerdict,
)
from embodied.contracts.records import Calibration, ClockStamp
from embodied.control.vehicle import ned_to_enu
from embodied.memory.world import FREE, MapConfig, MapStore
from embodied.perception.camera import PoseProvenance, build_calibration, compute_validated_depth
from embodied.perception.estimation import StaticEstimationPort, pose_estimate_from_nav
from embodied.perception.mapping_ports import MapStoreMappingPort
from embodied.perception.stereo_imu_nav import StereoImuNav


DEFAULT_DEPTH_SETTINGS: dict[str, Any] = {
    "num_disparities": 128,
    "block_size": 5,
    "uniqueness_ratio": 10,
    "speckle_window_size": 200,
    "speckle_range": 2,
    "texture_threshold": 10,
    "lr_tolerance_px": 1.0,
    "disparity_quantization_sigma_px": 1.0,
    "border_px": 8,
    "depth_range_m": [0.5, 6.0],
    "min_disparity": 0,
    "p1": 8 * 5 * 5,
    "p2": 32 * 5 * 5,
}


def default_map_config() -> MapConfig:
    return MapConfig(
        voxel_m=0.25,
        bounds_odom_m={"x": (-1.0, 6.0), "y": (-2.0, 2.0), "z": (-1.0, 3.0)},
        surface_band_m=0.15,
        log_odds_hit=0.85,
        log_odds_pass=-0.45,
        clamp=5.0,
        free_threshold=0.55,
        occupied_threshold=0.6,
        min_clearing_rays=1,
        freshness_s=60.0,
        dynamic_speed_mps=None,
        dynamic_reach_s=None,
    )


@dataclass
class PerceptionPipeline:
    """Sensor-derived EstimationPort + MappingPort writer."""

    nav_epoch: str = "perception-live"
    depth_settings: dict[str, Any] | None = None
    map_config: MapConfig | None = None
    calibration: Calibration | None = None

    def __post_init__(self) -> None:
        self._estimator = StereoImuNav(nav_epoch=self.nav_epoch)
        self._estimation = StaticEstimationPort(None)
        self._calibration = self.calibration or build_calibration()
        cfg = self.map_config or default_map_config()
        self._store = MapStore(config=cfg, submap_id="perception-live", nav_epoch=self.nav_epoch)
        stamp = ClockStamp(host_id="perception", clock_id="host/monotonic", monotonic_ns=0)
        self._mapping = MapStoreMappingPort(
            self._store,
            nav_epoch=self.nav_epoch,
            evidence_class=EvidenceClass.SENSOR_DERIVED,
            stamp=stamp,
            limitations=("shared_sensor_stereo_imu", "stereo_imu_nav"),
        )
        self._depth_settings = dict(self.depth_settings or DEFAULT_DEPTH_SETTINGS)
        self._integrates = 0
        self._valid_depth_frames = 0
        self._last_depth_valid_fraction: float | None = None
        self._rejections: list[str] = []

    @property
    def estimation(self) -> StaticEstimationPort:
        return self._estimation

    @property
    def mapping(self) -> MapStoreMappingPort:
        return self._mapping

    @property
    def estimator(self) -> StereoImuNav:
        return self._estimator

    @property
    def store(self) -> MapStore:
        return self._store

    def align_odom_position(self, position_m: tuple[float, float, float]) -> bool:
        """One-shot odom↔Vehicle frame glue before map integrate (joint prove)."""
        return self._estimator.align_odom_position(position_m)

    def refresh_nav(self, stamp: ClockStamp) -> None:
        """Re-publish EstimationPort from latest IMU/pair ages at ``stamp``."""
        nav = self._estimator.latest(stamp=stamp, now_host_ns=stamp.monotonic_ns)
        self._estimation.set(nav)
        if nav is not None:
            self._mapping._sim_time_s = nav.sim_time_s  # noqa: SLF001
            self._mapping._stamp = stamp  # noqa: SLF001
            self._mapping._age_s = nav.age_s  # noqa: SLF001
            self._mapping._age_domain = (
                AgeDomain.SIM_CONTROL if nav.sim_time_s is not None else AgeDomain.MONOTONIC
            )

    def mark_pair_for_nav(
        self,
        *,
        capture_host_ns: int,
        sim_time_s: float | None,
        stamp: ClockStamp,
    ) -> None:
        """Advance visual age without map integrate (frozen-revision segment)."""
        self._estimator.ingest_pair(capture_host_ns=capture_host_ns, sim_time_s=sim_time_s)
        self.refresh_nav(stamp)

    def stats(self) -> dict[str, Any]:
        return {
            "imu_count": self._estimator.imu_count,
            "pair_count": self._estimator.pair_count,
            "odom_aligned": self._estimator.odom_aligned,
            "integrates": self._integrates,
            "valid_depth_frames": self._valid_depth_frames,
            "last_depth_valid_fraction": self._last_depth_valid_fraction,
            "known_cells": len(self._store.known_cells()),
            "free_cells": sum(
                1
                for cell in self._store.known_cells()
                if self._store.classify(cell, now_ns=self._store._newest_stamp_ns() or 0)  # noqa: SLF001
                == FREE
            ),
            "map_revision": self._store.revision,
            "rejections": list(self._rejections[-8:]),
            "store_rejections": list(self._store.rejections[-8:]),
        }

    def on_imu(
        self,
        *,
        accelerometer: tuple[float, float, float],
        gyro: tuple[float, float, float],
        capture_host_ns: int,
        sim_time_s: float | None,
        stamp: ClockStamp,
    ) -> None:
        # Controller ships IMU in ArduPilot NED; cameras/calib are body ENU/FLU.
        self._estimator.ingest_imu(
            accelerometer=ned_to_enu(accelerometer),
            gyro=ned_to_enu(gyro),
            capture_host_ns=capture_host_ns,
            sim_time_s=sim_time_s,
        )
        nav = self._estimator.latest(stamp=stamp, now_host_ns=stamp.monotonic_ns)
        self._estimation.set(nav)
        if nav is not None:
            self._mapping._sim_time_s = nav.sim_time_s  # noqa: SLF001 — freshness on handle
            self._mapping._stamp = stamp  # noqa: SLF001
            self._mapping._age_s = nav.age_s  # noqa: SLF001
            self._mapping._age_domain = (
                AgeDomain.SIM_CONTROL if nav.sim_time_s is not None else AgeDomain.MONOTONIC
            )

    def on_pair(
        self,
        *,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        pair_id: int | str,
        capture_host_ns: int,
        sim_time_s: float | None,
        stamp: ClockStamp,
    ) -> OccupancyVerdict | None:
        self._estimator.ingest_pair(capture_host_ns=capture_host_ns, sim_time_s=sim_time_s)
        nav = self._estimator.latest(stamp=stamp, now_host_ns=stamp.monotonic_ns)
        self._estimation.set(nav)
        if nav is None or not nav.valid:
            self._rejections.append("pair skipped: nav not valid sensor_derived yet")
            return None

        provenance = PoseProvenance(
            label="SENSOR_DERIVED",
            detail="stereo_imu_nav capture pose nearest the pair (accel+gyro; no POSE/truth)",
        )
        depth = compute_validated_depth(
            left_rgb,
            right_rgb,
            self._calibration,
            self._depth_settings,
            pair_id=str(pair_id),
            capture_stamp=stamp,
            receipt_stamp=stamp,
            sim_time_s=sim_time_s,
            pose_provenance=provenance,
        )
        valid_frac = float(np.mean(depth.valid)) if depth.valid.size else 0.0
        self._last_depth_valid_fraction = valid_frac
        if valid_frac <= 0.0:
            self._rejections.append("pair skipped: no valid depth pixels")
            return None
        self._valid_depth_frames += 1

        pose = pose_estimate_from_nav(nav)
        before = self._store.revision
        revision = self._mapping.integrate(
            depth,
            pose,
            self._calibration,
            stamp_ns=capture_host_ns,
            observation_id=f"pair-{pair_id}",
            now_ns=capture_host_ns,
            nav=nav,
        )
        if revision != before:
            self._integrates += 1
        return self.query_sensor_derived_free(stamp)

    def query_sensor_derived_free(self, stamp: ClockStamp) -> OccupancyVerdict | None:
        """Prefer querying known FREE cells; else forward AABB; else known-cell AABB."""
        query = self._mapping.occupancy()
        if query is None:
            return None
        free_cells = tuple(cell for cell, label in query.cells if label == FREE)
        if free_cells:
            return query.query_cells(free_cells[:128], now=stamp)
        nav = self._estimation.latest()
        if nav is not None:
            x, y, z = nav.pose.position_m
            forward = Aabb(
                min_m=(x + 0.4, y - 0.8, z - 0.8),
                max_m=(x + 4.0, y + 0.8, z + 0.8),
                frame="odom",
            )
            verdict = query.query_volume(forward, now=stamp)
            if verdict.support is not OccupancySupport.UNSUPPORTED or verdict.reason != "never_observed":
                return verdict
        if not query.cells:
            return query.query_volume(
                Aabb(min_m=(0.0, -0.5, -0.5), max_m=(2.0, 0.5, 0.5), frame="odom"),
                now=stamp,
            )
        # Expand to cover whatever the map actually wrote (attitude may differ).
        centers = [query._cell_center(cell) for cell, _ in query.cells]  # noqa: SLF001
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        zs = [c[2] for c in centers]
        pad = 0.15
        return query.query_volume(
            Aabb(
                min_m=(min(xs) - pad, min(ys) - pad, min(zs) - pad),
                max_m=(max(xs) + pad, max(ys) + pad, max(zs) + pad),
                frame="odom",
            ),
            now=stamp,
        )

    def query_forward_volume(self, stamp: ClockStamp) -> OccupancyVerdict | None:
        return self.query_sensor_derived_free(stamp)


def decode_pair_rgb(
    left_bytes: bytes,
    right_bytes: bytes,
    *,
    width: int,
    height: int,
    encoding: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode gateway pair bytes to HxWx3 RGB uint8."""
    if encoding == "rgb8":
        channels = 3
        left = np.frombuffer(left_bytes, dtype=np.uint8).reshape(height, width, channels)
        right = np.frombuffer(right_bytes, dtype=np.uint8).reshape(height, width, channels)
        return left.copy(), right.copy()
    if encoding == "bgra8":
        left_b = np.frombuffer(left_bytes, dtype=np.uint8).reshape(height, width, 4)
        right_b = np.frombuffer(right_bytes, dtype=np.uint8).reshape(height, width, 4)
        left = left_b[:, :, [2, 1, 0]].copy()
        right = right_b[:, :, [2, 1, 0]].copy()
        return left, right
    if encoding == "gray8":
        left_g = np.frombuffer(left_bytes, dtype=np.uint8).reshape(height, width)
        right_g = np.frombuffer(right_bytes, dtype=np.uint8).reshape(height, width)
        left = np.stack([left_g, left_g, left_g], axis=-1)
        right = np.stack([right_g, right_g, right_g], axis=-1)
        return left, right
    raise ValueError(f"unsupported pair encoding {encoding!r}")


def is_sensor_derived_free(verdict: OccupancyVerdict | None) -> bool:
    return (
        verdict is not None
        and verdict.support is OccupancySupport.FREE
        and verdict.evidence_class is EvidenceClass.SENSOR_DERIVED
    )
