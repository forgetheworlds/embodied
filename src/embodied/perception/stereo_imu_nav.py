"""Minimal sensor-derived NavigationState from stereo + IMU only.

Rebuild (not a mission_runtime/OV wrapper). Uses accelerometer + gyro only —
never Webots POSE, never InertialUnit absolute RPY, never Vehicle EKF.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from embodied.contracts.perception_ports import (
    AgeDomain,
    EvidenceClass,
    EstimationPort,
    NavEpoch,
    NavPose,
    NavStatus,
    NavigationState,
    uncompared_disagreement,
)
from embodied.contracts.records import ClockStamp


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_from_gyro(gyro_rad_s: np.ndarray, dt_s: float) -> np.ndarray:
    angle = float(np.linalg.norm(gyro_rad_s) * dt_s)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = gyro_rad_s / float(np.linalg.norm(gyro_rad_s))
    half = 0.5 * angle
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _rotate_body_to_odom(quat_wxyz: np.ndarray, vec_body: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    # R from body→odom for Hamilton wxyz
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float64)
    return R @ vec_body


def _accel_to_tilt_quat(accel_mps2: np.ndarray) -> np.ndarray | None:
    """Gravity-aligned attitude from accel alone (yaw free → identity yaw)."""
    g = float(np.linalg.norm(accel_mps2))
    if g < 1.0:
        return None
    # Body accel ≈ -g in world when static; want body +z ≈ world +z (ENU up).
    a = accel_mps2 / g
    # Desired up in body is opposite measured specific force when hovering.
    up = -a
    up = up / float(np.linalg.norm(up))
    # Rotate body +z onto up with shortest rotation; yaw left free (identity).
    z_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(z_body, up)
    c = float(np.dot(z_body, up))
    if c < -0.999:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    if float(np.linalg.norm(v)) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    s = math.sqrt((1.0 + c) * 2.0)
    inv = 1.0 / s
    return _quat_normalize(np.array([0.25 * s, v[0] * inv, v[1] * inv, v[2] * inv], dtype=np.float64))


@dataclass
class StereoImuNav:
    """Pull-model EstimationPort producer: sensor_derived from stereo+IMU."""

    nav_epoch: NavEpoch = "stereo-imu-epoch"
    source_ids: tuple[str, ...] = ("stereo_imu_nav",)
    imu_source_ids: tuple[str, ...] = ("imu_accel", "imu_gyro")
    visual_source_ids: tuple[str, ...] = ("stereo_pair",)
    # Complementary filter weight toward accel tilt when nearly static.
    accel_tilt_alpha: float = 0.02
    static_gyro_norm_max: float = 0.15
    static_accel_err_max: float = 1.5
    # Host SGBM can burn >1s while a PAIR queue drains without new IMU ingest;
    # stall is IMU-only (pose comes from accel+gyro). 2.5s covers a short burst.
    stall_after_s: float = 2.5
    # IMU must be this close to a stereo capture on the sensor clock for map bind.
    capture_imu_slop_s: float = 0.25

    def __post_init__(self) -> None:
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._position_m = np.zeros(3, dtype=np.float64)
        self._velocity_mps = np.zeros(3, dtype=np.float64)
        self._last_imu_host_ns: int | None = None
        self._last_imu_sim_s: float | None = None
        self._last_pair_host_ns: int | None = None
        self._last_pair_sim_s: float | None = None
        self._initialized = False
        self._aligned = False
        self._state_sequence = 0
        self._pair_count = 0
        self._imu_count = 0

    @property
    def pair_count(self) -> int:
        return self._pair_count

    @property
    def imu_count(self) -> int:
        return self._imu_count

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def odom_aligned(self) -> bool:
        return self._aligned

    def align_odom_position(self, position_m: tuple[float, float, float]) -> bool:
        """One-shot odom origin for this epoch (joint prove frame glue).

        Does not change evidence_class. Call once before map integrate so
        occupancy and Vehicle Motions share a frame for Safety stop-tube CLEAR.
        """
        if self._aligned or not self._initialized:
            return False
        self._position_m = np.asarray(position_m, dtype=np.float64).copy()
        self._velocity_mps[:] = 0.0
        self._aligned = True
        return True

    def ingest_imu(
        self,
        *,
        accelerometer: tuple[float, float, float],
        gyro: tuple[float, float, float],
        capture_host_ns: int,
        sim_time_s: float | None,
    ) -> None:
        """Integrate one accel+gyro sample. Ignores InertialUnit / POSE truth."""
        accel = np.asarray(accelerometer, dtype=np.float64)
        gyro_v = np.asarray(gyro, dtype=np.float64)
        self._imu_count += 1
        if self._last_imu_host_ns is None:
            tilt = _accel_to_tilt_quat(accel)
            if tilt is not None:
                self._quat = tilt
                self._initialized = True
            self._last_imu_host_ns = capture_host_ns
            self._last_imu_sim_s = sim_time_s
            return
        dt = (capture_host_ns - self._last_imu_host_ns) * 1e-9
        self._last_imu_host_ns = capture_host_ns
        self._last_imu_sim_s = sim_time_s
        if dt <= 0.0 or dt > 0.05:
            return
        self._quat = _quat_normalize(_quat_multiply(self._quat, _quat_from_gyro(gyro_v, dt)))
        # Accel tilt correction when nearly static (no absolute yaw from magnetometer).
        gyro_n = float(np.linalg.norm(gyro_v))
        accel_n = float(np.linalg.norm(accel))
        if (
            gyro_n < self.static_gyro_norm_max
            and abs(accel_n - 9.81) < self.static_accel_err_max
        ):
            tilt = _accel_to_tilt_quat(accel)
            if tilt is not None:
                # Blend only roll/pitch; keep current yaw by slerp-lite on full quat with small alpha.
                a = self.accel_tilt_alpha
                self._quat = _quat_normalize((1.0 - a) * self._quat + a * tilt)
        # ZUPT: hold position/velocity when static — honest for hover proof; no truth used.
        if gyro_n < self.static_gyro_norm_max and abs(accel_n - 9.81) < self.static_accel_err_max:
            self._velocity_mps[:] = 0.0
        else:
            # Specific force → approx linear accel in odom (remove gravity).
            acc_odom = _rotate_body_to_odom(self._quat, accel) - np.array([0.0, 0.0, 9.81])
            self._velocity_mps = self._velocity_mps + acc_odom * dt
            self._position_m = self._position_m + self._velocity_mps * dt

    def ingest_pair(self, *, capture_host_ns: int, sim_time_s: float | None) -> None:
        """Mark a stereo observation (vision age). Pose motion from IMU; stereo feeds depth."""
        self._pair_count += 1
        self._last_pair_host_ns = capture_host_ns
        self._last_pair_sim_s = sim_time_s
        if self._initialized:
            self._state_sequence += 1

    def _publish(
        self,
        *,
        stamp: ClockStamp,
        now_ns: int,
        imu_age_s: float | None,
        visual_age_s: float | None,
        feed_stall: bool,
        status: NavStatus,
        valid: bool,
    ) -> NavigationState:
        pose = NavPose(
            parent_frame="odom",
            child_frame="body",
            stamp=stamp,
            position_m=(
                float(self._position_m[0]),
                float(self._position_m[1]),
                float(self._position_m[2]),
            ),
            quaternion_wxyz=(
                float(self._quat[0]),
                float(self._quat[1]),
                float(self._quat[2]),
                float(self._quat[3]),
            ),
            covariance=None,
            nav_epoch=self.nav_epoch,
            source_ids=self.source_ids,
            valid=valid,
        )
        age_s = float(imu_age_s if imu_age_s is not None else 0.0)
        return NavigationState(
            nav_epoch=self.nav_epoch,
            state_sequence=self._state_sequence,
            controller_alignment_id=None,
            stamp=stamp,
            sim_time_s=self._last_pair_sim_s if self._last_pair_sim_s is not None else self._last_imu_sim_s,
            age_s=age_s,
            age_domain=AgeDomain.SIM_CONTROL if self._last_imu_sim_s is not None else AgeDomain.MONOTONIC,
            monotonic_observed_at_s=now_ns * 1e-9,
            pose=pose,
            velocity_mps=(
                float(self._velocity_mps[0]),
                float(self._velocity_mps[1]),
                float(self._velocity_mps[2]),
            ),
            covariance=None,
            status=status,
            valid=valid,
            sigma_pos_m=None,
            visual_source_ids=self.visual_source_ids,
            imu_source_ids=self.imu_source_ids,
            visual_age_s=visual_age_s,
            imu_age_s=imu_age_s,
            feed_stall=feed_stall,
            ap_disagreement=uncompared_disagreement(),
            evidence_class=EvidenceClass.SENSOR_DERIVED,
            source_ids=self.source_ids,
        )

    def latest(
        self,
        *,
        stamp: ClockStamp,
        now_host_ns: int | None = None,
    ) -> NavigationState | None:
        """Live Safety pull: ages vs wall/host now. IMU-only feed_stall."""
        if not self._initialized:
            return None
        now_ns = now_host_ns if now_host_ns is not None else stamp.monotonic_ns
        imu_age = None
        if self._last_imu_host_ns is not None:
            imu_age = max(0.0, (now_ns - self._last_imu_host_ns) * 1e-9)
        visual_age = None
        if self._last_pair_host_ns is not None:
            visual_age = max(0.0, (now_ns - self._last_pair_host_ns) * 1e-9)
        # Pose is IMU-derived: stereo latency alone must not mark nav STALE (map
        # bind uses latest_at_capture). Missing IMU past stall_after_s does.
        feed_stall = bool(imu_age is not None and imu_age > self.stall_after_s)
        status = NavStatus.HEALTHY
        valid = True
        if feed_stall:
            status = NavStatus.STALE
            valid = False
        elif self._pair_count < 1:
            status = NavStatus.INITIALIZING
            valid = False
        return self._publish(
            stamp=stamp,
            now_ns=now_ns,
            imu_age_s=imu_age,
            visual_age_s=visual_age,
            feed_stall=feed_stall,
            status=status,
            valid=valid,
        )

    def latest_at_capture(
        self,
        *,
        stamp: ClockStamp,
        capture_host_ns: int,
    ) -> NavigationState | None:
        """Capture-time nav for depth→map: IMU freshness vs pair stamp, not wall SGBM."""
        if not self._initialized or self._last_imu_host_ns is None:
            return None
        imu_age = abs(capture_host_ns - self._last_imu_host_ns) * 1e-9
        visual_age = 0.0
        if self._last_pair_host_ns is not None:
            visual_age = abs(capture_host_ns - self._last_pair_host_ns) * 1e-9
        feed_stall = imu_age > self.capture_imu_slop_s
        status = NavStatus.HEALTHY
        valid = True
        if feed_stall:
            status = NavStatus.STALE
            valid = False
        elif self._pair_count < 1:
            status = NavStatus.INITIALIZING
            valid = False
        return self._publish(
            stamp=stamp,
            now_ns=capture_host_ns,
            imu_age_s=imu_age,
            visual_age_s=visual_age,
            feed_stall=feed_stall,
            status=status,
            valid=valid,
        )


class LiveStereoImuEstimationPort:
    """EstimationPort that recomputes age/stall on every latest() pull."""

    def __init__(self, estimator: StereoImuNav, *, host_id: str = "perception") -> None:
        self._estimator = estimator
        self._host_id = host_id

    def latest(self) -> NavigationState | None:
        now_ns = time.monotonic_ns()
        stamp = ClockStamp(
            host_id=self._host_id,
            clock_id="host/monotonic",
            monotonic_ns=now_ns,
        )
        return self._estimator.latest(stamp=stamp, now_host_ns=now_ns)

    def current_epoch(self) -> NavEpoch | None:
        state = self.latest()
        return None if state is None else state.nav_epoch

    def healthy(self) -> bool:
        state = self.latest()
        return (
            state is not None
            and state.valid
            and state.status in (NavStatus.HEALTHY, NavStatus.DEGRADED)
        )


_: type[EstimationPort] = LiveStereoImuEstimationPort
