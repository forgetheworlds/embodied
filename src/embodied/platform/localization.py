"""P01-L sensor-derived localization: the estimator adapter and its seams.

The adapter owns three things and nothing else: the socket client for the pinned
estimator process (OpenVINS v2.7 behind ``ov_stream``, plan section 3.4), the
fixed odom-to-local-NED alignment (specification section 6.3), and the
health/freshness state machine that decides whether an estimate may be
published at all (plan sections 4.5 and 6). Publishing rides the same
ExternalNav discipline the P00 gate proved: ``VISION_POSITION_ESTIMATE`` and
``VISION_SPEED_ESTIMATE`` at a 10 ms cadence on the adapter's own MAVLink
connection — never commanded by this process, never fed simulator truth.

The estimator process receives exactly what the declared sensors carry — stereo
pairs converted to grayscale and inertial samples — stamped in simulator time,
the one clock both declared devices share (specification section 4.1). The
protocol below has no truth field by construction.

Health is enforced here, not downstream. ``VISION_POSITION_ESTIMATE`` carries
quality hardcoded 0 at the pinned commit (GCS_Common.cpp:4150), so
``VISO_QUAL_MIN`` cannot gate anything: the only honest gate is not sending.
When the machine stops, it stops completely — no zero pose, no frozen last pose.
"""

from __future__ import annotations

import math
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "AlignmentError",
    "EstimatorState",
    "ExternalNavPublisher",
    "HealthBounds",
    "HealthEvent",
    "HealthMachine",
    "OdomAlignment",
    "OvStreamClient",
    "ProtocolError",
    "decode_state",
    "encode_imu",
    "encode_reset",
    "encode_stereo",
    "grayscale_rgb8",
    "publish_period_s",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "rotmat_to_rpy",
]


class ProtocolError(Exception):
    """A frame is malformed, a connection is lost, or the client is not connected."""


# ---------------------------------------------------------------------------
# The ov_stream protocol (plan section 3.4)
# ---------------------------------------------------------------------------
#
# One length-prefixed frame per message, little-endian throughout:
#   [magic u16][kind u8][flags u16][payload length u32][payload]
# host→estimator kinds IMU/STEREO/RESET; estimator→host kind STATE. The STATE
# payload is time_ns, initialized, quat wxyz, position, velocity, gyro bias,
# accel bias, position sigma, n_tracks, last visual update, reset counter.

FRAME_MAGIC = 0x4F56  # "OV"
FRAME_HEADER = struct.Struct("<HBBI")
FRAME_HEADER_SIZE = FRAME_HEADER.size

KIND_IMU = 1
KIND_STEREO = 2
KIND_RESET = 3
KIND_STATE = 4

_IMU_PAYLOAD = struct.Struct("<q6d")
_STEREO_PAYLOAD_HEADER = struct.Struct("<QII")
_STATE_PAYLOAD = struct.Struct("<QB19dIQB")


def _frame(kind: int, payload: bytes) -> bytes:
    return FRAME_HEADER.pack(FRAME_MAGIC, kind, 0, len(payload)) + payload


def encode_imu(time_ns: int, gyro: Sequence[float], accel: Sequence[float]) -> bytes:
    """One inertial sample in the estimator's body convention (rad/s, m/s²)."""
    if len(gyro) != 3 or len(accel) != 3:
        raise ProtocolError("an imu frame carries three gyro and three accel values")
    return _frame(KIND_IMU, _IMU_PAYLOAD.pack(time_ns, *gyro, *accel))


def encode_stereo(time_ns: int, left: bytes, right: bytes, width: int, height: int) -> bytes:
    """One stereo pair as two width×height 8-bit grayscale planes."""
    expected = width * height
    if len(left) != expected or len(right) != expected:
        raise ProtocolError(
            f"stereo planes must be {expected} bytes each, got {len(left)} and {len(right)}"
        )
    return _frame(KIND_STEREO, _STEREO_PAYLOAD_HEADER.pack(time_ns, width, height) + left + right)


def encode_reset() -> bytes:
    """Ask the estimator for a fresh initialization; the old state is abandoned."""
    return _frame(KIND_RESET, b"")


def decode_state(frame: bytes) -> EstimatorState:
    """Parse one STATE frame. A truncated or foreign frame is refused, never guessed."""
    if len(frame) < FRAME_HEADER_SIZE:
        raise ProtocolError(f"frame header truncated: {len(frame)} bytes")
    magic, kind, _flags, length = FRAME_HEADER.unpack_from(frame)
    if magic != FRAME_MAGIC:
        raise ProtocolError(f"frame magic 0x{magic:04x} is not the ov_stream magic")
    if kind != KIND_STATE:
        raise ProtocolError(f"frame kind {kind} is not a STATE frame")
    payload = frame[FRAME_HEADER_SIZE:]
    if len(payload) != length:
        raise ProtocolError(f"STATE payload is {len(payload)} bytes, its header says {length}")
    if len(payload) != _STATE_PAYLOAD.size:
        raise ProtocolError(
            f"STATE payload is {len(payload)} bytes, the pinned layout is {_STATE_PAYLOAD.size}"
        )
    (
        time_ns,
        initialized,
        quat_w, quat_x, quat_y, quat_z,
        px, py, pz,
        vx, vy, vz,
        gb_x, gb_y, gb_z,
        ab_x, ab_y, ab_z,
        sx, sy, sz,
        n_tracks,
        t_last_visual_ns,
        reset_counter,
    ) = _STATE_PAYLOAD.unpack(payload)
    return EstimatorState(
        time_ns=time_ns,
        initialized=bool(initialized),
        quat_wxyz=(quat_w, quat_x, quat_y, quat_z),
        position_m=(px, py, pz),
        velocity_mps=(vx, vy, vz),
        gyro_bias=(gb_x, gb_y, gb_z),
        accel_bias=(ab_x, ab_y, ab_z),
        sigma_pos_m=(sx, sy, sz),
        n_tracks=n_tracks,
        t_last_visual_ns=t_last_visual_ns,
        reset_counter=reset_counter,
    )


@dataclass(frozen=True)
class EstimatorState:
    """One published filter state, exactly the ov_stream STATE fields."""

    time_ns: int
    initialized: bool
    quat_wxyz: tuple[float, float, float, float]
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    gyro_bias: tuple[float, float, float]
    accel_bias: tuple[float, float, float]
    sigma_pos_m: tuple[float, float, float]
    n_tracks: int
    t_last_visual_ns: int
    reset_counter: int


# ---------------------------------------------------------------------------
# Feed formatting
# ---------------------------------------------------------------------------


def grayscale_rgb8(rgb: bytes, width: int, height: int) -> bytes:
    """ITU-R BT.601 luma of one rgb8 frame, as the estimator's grayscale plane."""
    image = np.frombuffer(rgb, dtype=np.uint8)
    if image.size != width * height * 3:
        raise ProtocolError(f"an rgb8 frame must carry {width * height * 3} bytes")
    channels = image.reshape(height, width, 3).astype(np.float32)
    luma = 0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]
    return luma.round().astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# Rotation helpers and the odom → local-NED alignment (specification 6.3)
# ---------------------------------------------------------------------------


class AlignmentError(Exception):
    """The alignment was asked to consume a state it cannot convert."""


def quat_to_rotmat(q_wxyz: Sequence[float]) -> np.ndarray:
    """Rotation matrix of one normalized (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q_wxyz)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise AlignmentError("a zero quaternion has no rotation")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """The (w, x, y, z) quaternion of one rotation matrix."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = (1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) ** 0.5 * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = (1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) ** 0.5 * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = (1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) ** 0.5 * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z])
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def rotmat_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw of one rotation matrix in the NED 3-2-1 convention."""
    roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
    pitch = float(np.arcsin(max(-1.0, min(1.0, -matrix[2, 0]))))
    yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return (roll, pitch, yaw)


# The fixed axis maps of one nav_epoch, each derived from the pinned chain
# (plan sections 0.3 item 2 and 0.5, the latter written from measurement):
#
# - the world is x-north, y-west, z-up (dev-a-single/world.wbt:12) and the
#   bridge's gate-proven world→autopilot conversion keeps x and negates y and z
#   (webots_ardupilot.py enu_to_ned, pinned to SIM_Webots_Python.cpp);
# - the estimator's body is FLU (ov_stream rotates the autopilot-shaped
#   inertial samples by (x, −y, −z) so a level vehicle presents gravity along
#   +z, plan section 3.4 convention 1), while ArduPilot's body is FRD, so an
#   attitude must carry the FLU→FRD map on the right — a 180° roll, measured
#   missing on the first textured invocation (receipt
#   p01l-run5-textured-20260926T104945Zb: the −180.0 roll delta A1 reported);
# - the odom frame's *yaw* is not derivable from geometry at all. The pinned
#   static initializer builds its world frame with gram_schmidt
#   (StaticInitializer.cpp:120-123, ov_init/src/utils/helper.h:138-157), whose
#   branch is decided by |e₁·z| < |e₂·z|; at a level start both terms are
#   accelerometer noise, so the initializer picks x_axis = z×e₁ or z×e₂ run by
#   run, i.e. a yaw that is arbitrary and unobservable (no magnetometer and no
#   GPS in this arm; the options header exposes no yaw parameter). The second
#   textured invocation measured exactly that: a +90.0 yaw delta where the
#   geometric prediction said zero.
#
# Therefore the epoch rotation is *derived once* from two things that are known
# at the declared stationary start — the estimator's own first initialized
# attitude and the world's declared start attitude — and then frozen for the
# epoch. It is a fixed alignment, never re-estimated in flight, so estimator
# drift still shows up in the published state instead of being absorbed.
WORLD_TO_NED_AXES = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
FLU_TO_FRD_AXES = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def rotmat_from_rpy(rpy: Sequence[float]) -> np.ndarray:
    """The NED 3-2-1 rotation matrix of one roll, pitch, yaw triple."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ]
    )


def _delivered_body_to_odom(quat_wxyz: Sequence[float]) -> np.ndarray:
    """The body->odom rotation of one STATE quaternion as the wire delivers it.

    ov_stream conjugates the pinned state quaternion's JPL vector part and packs
    it (w, x, y, z) intending body->odom (estimator/ov_stream.cpp "Convention
    3"). The pin's quat_2_Rot, however, is the JPL form -- the negative skew
    term, quat_ops.h:142-152 of the pinned tarball -- which is the transpose of
    this module's Hamilton quat_to_rotmat for the same four numbers, so the
    delivered numbers read as Hamilton are the ODOM->BODY rotation. Measured,
    run p01l-fix3b-20260927T050042Z: the published attitude carried the
    opposite sign to the dataflash SIM truth on every axis through the whole
    flight, and transposing the published rotation recovers truth to a
    transport lag; positions stayed correct only because that run's
    gram_schmidt yaw sealed near zero (epoch yaw 179.54 deg), which the epoch
    derivation absorbs -- at a +/-90 degree seal the pre-fix epoch rotated
    true displacements by the double yaw. The alignment therefore transposes
    the delivered read itself: what it composes is the body->odom rotation.
    """
    return quat_to_rotmat(quat_wxyz).T

class OdomAlignment:
    """The fixed odom→local-NED transform of one nav_epoch.

    The odom origin is the estimator's initialization point, taken from the
    configured world's own vehicle translation rather than the calibration
    referee's declared pose (plan section 0.3 item 3). The odom *attitude* is
    sealed once, at the first initialized state while the vehicle is static at
    the declared start: the epoch rotation that takes the estimator's reported
    attitude to the autopilot's NED/FRD attitude is computed from the world's
    declared start attitude and the estimator's own initial attitude, then
    frozen. Before sealing, no state may be converted — an unsealed alignment
    raises rather than guessing a yaw.
    """

    def __init__(
        self,
        odom_origin_world_m: Sequence[float],
        declared_start_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        origin = np.asarray(odom_origin_world_m, dtype=np.float64)
        if origin.shape != (3,):
            raise AlignmentError("the odom origin must be one ENU position")
        if len(declared_start_rpy) != 3:
            raise AlignmentError("the declared start attitude must be one rpy triple")
        self._origin_ned = WORLD_TO_NED_AXES @ origin
        self._declared_start_rpy = tuple(float(value) for value in declared_start_rpy)
        self._epoch_rotation: np.ndarray | None = None

    @property
    def sealed(self) -> bool:
        return self._epoch_rotation is not None

    @property
    def declared_start_rpy(self) -> tuple[float, float, float]:
        return self._declared_start_rpy

    @property
    def epoch_rotation(self) -> np.ndarray:
        self._require_sealed()
        return self._epoch_rotation

    def epoch_yaw_deg(self) -> float:
        """The sealed rotation's yaw on the vertical: recorded, never gated."""
        return math.degrees(rotmat_to_rpy(self.epoch_rotation)[2])

    def seal(self, initial_quat_odom_wxyz: Sequence[float]) -> None:
        """Derive and freeze the epoch rotation from the first initialized state.

        Position and velocity are published as ``A · p_odom`` and ``A · v_odom``
        with ``A = R_declared_start · FLU→FRD · R_body→odom(start)⁻¹``: one fixed
        odom→NED rotation per epoch, the estimator's own initialization frame
        related to the autopilot's frame by the two things known at the declared
        stationary start. Idempotent: the first seal wins and the rotation never
        moves afterwards.
        """
        if self._epoch_rotation is not None:
            return
        initial = _delivered_body_to_odom(initial_quat_odom_wxyz)
        declared = rotmat_from_rpy(self._declared_start_rpy)
        self._epoch_rotation = declared @ FLU_TO_FRD_AXES @ initial.T
        # Why the attitude is not composed through the epoch rotation: the
        # epoch rotation carries the initializer's noise-decided gram_schmidt
        # yaw (ov_init/src/utils/helper.h:138-157), and composing the attitude
        # through it conjugates the rotation since start by that yaw, which
        # permutes pitch into roll whenever the yaw is near +/-90 degrees.
        # Measured: run p01l-bringup-20260926T202549Z published a physical
        # pitch sequence of (0.117, 0.414, 1.52) rad nose down as roll
        # (0.112, 0.439, 1.557) with pitch near zero, value for value, and the
        # 204753Z bring-up (sealed epoch yaw 89.62 degrees) shows the same
        # permutation from the first degree of rotation. aligned_attitude_rpy
        # therefore publishes
        #   declared @ FLU→FRD @ (R_body→odom(start)⁻¹ @ R_body→odom(t)) @ FLU→FRD
        # -- the rotation since the sealed start taken in the BODY frame at the
        # seal, conjugated into FRD and composed onto the declared start -- and
        # keeps the epoch rotation on position and velocity only. The order is
        # load-bearing: the product the other way round is the same rotation in
        # the odom WORLD frame, and composing that conjugates by the
        # gram_schmidt yaw, which is the b477ee7 permutation. Validated against
        # the recorded publications of 202549Z itself and, for the composition
        # settled here, against run p01l-fix3b-20260927T050042Z end to end:
        # transposing the pre-fix published rotation recovers the dataflash SIM
        # truth to a transport lag through the whole flight (test file,
        # TestConventions::test_the_published_attitude_is_the_true_rotation_not_its_inverse).
        self._declared_rotation = declared
        self._initial_body_to_odom = initial

    def _require_sealed(self) -> None:
        if self._epoch_rotation is None:
            raise AlignmentError(
                "the odom alignment is not sealed: no initialized estimator state has "
                "defined this epoch's rotation, and an unsealed alignment must not guess "
                "the odom frame's unobservable yaw"
            )

    def aligned_position_ned(self, position_odom_m: Sequence[float]) -> tuple[float, float, float]:
        rotation = self.epoch_rotation
        point = self._origin_ned + rotation @ np.asarray(position_odom_m, dtype=np.float64)
        return (float(point[0]), float(point[1]), float(point[2]))

    def aligned_velocity_ned(self, velocity_odom_mps: Sequence[float]) -> tuple[float, float, float]:
        ned = self.epoch_rotation @ np.asarray(velocity_odom_mps, dtype=np.float64)
        return (float(ned[0]), float(ned[1]), float(ned[2]))

    def _aligned_attitude_rotation(self, quat_odom_wxyz: Sequence[float]) -> np.ndarray:
        # See seal(): the rotation since the sealed start -- taken in the body
        # frame at the seal, R(start)⁻¹ @ R(t) -- conjugated into FRD and
        # composed onto the declared start attitude. The epoch rotation stays
        # on position and velocity only.
        self._require_sealed()
        current = _delivered_body_to_odom(quat_odom_wxyz)
        since_start = self._initial_body_to_odom.T @ current
        return self._declared_rotation @ FLU_TO_FRD_AXES @ since_start @ FLU_TO_FRD_AXES

    def aligned_quat_ned_wxyz(
        self, quat_odom_wxyz: Sequence[float]
    ) -> tuple[float, float, float, float]:
        return rotmat_to_quat(self._aligned_attitude_rotation(quat_odom_wxyz))

    def aligned_attitude_rpy(self, quat_odom_wxyz: Sequence[float]) -> tuple[float, float, float]:
        return rotmat_to_rpy(self._aligned_attitude_rotation(quat_odom_wxyz))

    def aligned_state(self, state: EstimatorState) -> dict[str, object]:
        """One aligned state: NED position, NED attitude RPY and velocity."""
        return {
            "time_ns": state.time_ns,
            "position_ned_m": self.aligned_position_ned(state.position_m),
            "attitude_rpy": self.aligned_attitude_rpy(state.quat_wxyz),
            "velocity_ned_mps": self.aligned_velocity_ned(state.velocity_mps),
            "sigma_pos_m": state.sigma_pos_m,
            "reset_counter": state.reset_counter,
        }


# ---------------------------------------------------------------------------
# Health and freshness (plan sections 4.5 and 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthBounds:
    """The predeclared H and F bounds, frozen in the configuration before any run."""

    publish_period_s: float
    state_lost_after_s: float
    published_state_age_max_s: float
    max_publish_gap_s: float
    visual_update_warn_s: float
    visual_update_fail_s: float
    valid_fraction_min: float
    sigma_min_m: float
    sigma_max_m: float


@dataclass(frozen=True)
class HealthEvent:
    """One transition the machine recorded.

    ``at_ns`` is sim time when the triggering state carried one (init, recover,
    sigma) and wall time for wall-clock events (silence timeouts, publish
    failures); the detail names the kind.
    """

    at_ns: int
    event: str
    detail: str


class HealthMachine:
    """The publish gate: when an estimate is initialized, fresh and in-envelope.

    States: ``waiting`` until the estimator first reports initialized; ``healthy``
    while every bound holds; ``stopped`` the moment one does not. Stopping is
    total — a stopped machine publishes nothing, and the firmware's own failsafe
    path takes over (unhealthy at 300 ms without a message, dead-reckon
    declaration at 1000 ms, FS_EKF_ACTION Land). A fresh initialized state after
    a stop is a recovery and increments the adapter's reset counter, which rides
    every later publication (H2).
    """

    def __init__(self, bounds: HealthBounds) -> None:
        self.bounds = bounds
        self.state = "waiting"
        self.reset_counter = 0
        self.events: list[HealthEvent] = []
        self.publish_gaps_s: list[float] = []
        self.published_state_ages_s: list[float] = []
        self.outages_s: list[float] = []
        self._last_state_wall_ns: int | None = None
        self._last_publish_wall_ns: int | None = None
        self._window_open_wall_ns: int | None = None
        self._window_closed_wall_ns: int | None = None
        self._stopped_at_wall_ns: int | None = None
        # The FLIGHT's end, distinct from the window's end. The scored window is declared
        # arm to disarm and governs E1/H1/H2; the freshness metrics stop earlier, at the
        # LAND command, because the owner's ruling is that the bound protects the control
        # loop WHILE FLYING and that takeoff and landing are low-speed phases where pose
        # age does not matter.
        self._flight_end_wall_ns: int | None = None
        self.state_at_close: str | None = None

    def on_state(self, state: EstimatorState | None, now_wall_ns: int) -> bool:
        """Feed one consumed STATE (or None) on the wall clock; report publishability.

        This is the machine's single driver: the publisher's loop calls it once
        per tick with the newest state it was offered, so silence decays into a
        stop here and nowhere else.
        """
        if self._window_open_wall_ns is None:
            self._window_open_wall_ns = now_wall_ns
        if state is None:
            self._timeout_if_silent(now_wall_ns)
            return False
        self._last_state_wall_ns = now_wall_ns
        if not state.initialized:
            self.stop(now_wall_ns, "the estimator reported an uninitialized state")
            return False
        if self.state == "stopped":
            self._recover(state, now_wall_ns)
        elif self.state == "waiting":
            self.state = "healthy"
            self.events.append(HealthEvent(state.time_ns, "initialized", ""))
        sigma = state.sigma_pos_m
        if min(sigma) < self.bounds.sigma_min_m or max(sigma) > self.bounds.sigma_max_m:
            self.stop(
                now_wall_ns,
                f"sigma {tuple(sigma)} is outside the declared envelope "
                f"[{self.bounds.sigma_min_m}, {self.bounds.sigma_max_m}] m",
            )
            return False
        return True

    def _timeout_if_silent(self, now_wall_ns: int) -> None:
        if self.state == "stopped" or self._last_state_wall_ns is None:
            return
        silent_s = (now_wall_ns - self._last_state_wall_ns) / 1e9
        if silent_s > self.bounds.state_lost_after_s:
            self.stop(now_wall_ns, f"no estimator state for {silent_s:.3f} s")

    def _recover(self, state: EstimatorState, now_wall_ns: int) -> None:
        if self._stopped_at_wall_ns is not None:
            self.outages_s.append((now_wall_ns - self._stopped_at_wall_ns) / 1e9)
            self._stopped_at_wall_ns = None
        self.reset_counter += 1
        self.state = "healthy"
        self.events.append(
            HealthEvent(state.time_ns, "recovered", f"reset_counter={self.reset_counter}")
        )

    def stop(self, now_wall_ns: int, detail: str) -> None:
        """Stop publishing. Idempotent: the first stop carries the reason."""
        if self.state == "stopped":
            return
        self.state = "stopped"
        self._stopped_at_wall_ns = now_wall_ns
        self.events.append(HealthEvent(now_wall_ns, "stopped", detail))

    def open_window(self, now_wall_ns: int) -> None:
        """Start the scored window (arm to disarm), discarding bring-up accounting."""
        self._window_open_wall_ns = now_wall_ns
        self._window_closed_wall_ns = None
        self.state_at_close = None
        self._flight_end_wall_ns = None
        self._last_publish_wall_ns = None
        self.publish_gaps_s = []
        self.published_state_ages_s = []
        self.outages_s = []

    def close_window(self, now_wall_ns: int) -> None:
        """End the scored window: the flight is over and the harness is shutting down.

        The window is declared arm to disarm, and the vehicle has already disarmed by
        the time the flight sequence ends; everything after this point is the harness
        stopping Webots and SITL. Measured, run p01l-zupt5-20260927T042005Z: the adapter
        republished one frozen state 332 times over 10.13 s of that shutdown -- 32 % of
        the scored window -- and every one of them was scored against a truth sample that
        was equally frozen, so the scored window's p95 was pinned to the pose of a
        vehicle that had already crashed. Freezing here keeps the window the flight.
        """
        if self._window_open_wall_ns is None or now_wall_ns < self._window_open_wall_ns:
            return
        self._window_closed_wall_ns = now_wall_ns
        # H2/H4 read "final machine state healthy", and the flight's own end is where
        # that must be read: after this point the adapter's input is gone, so the
        # declared 300 ms silence stop is the correct behaviour and not a fault.
        # Measured, run p01l-fix3-20260927T045304Z: the machine stopped on silence in the
        # shutdown and H2/H4 read `stopped` at process exit.
        self.state_at_close = self.state


    def mark_flight_end(self, now_wall_ns: int) -> None:
        """End the FLIGHT. Earlier than the window's end, and only the freshness metrics.

        The LAND command is where flying stops, so F1's gaps, F2's ages and F3's gaps
        stop being recorded there. E1/H1/H2 keep the declared arm-to-disarm window: the
        descent and landing are still judged for accuracy, and only pose AGE stops being
        judged once the aircraft is coming down. Measured, run
        p01l-fix7a-20260927T172100Z: the 27 publications after LAND carried ONE frozen
        state (t=53.940) for 274 ms while the vehicle sat parked and disarmed -- the
        adapter running out its own declared 300 ms silence watchdog -- and those 27 rows
        were the entirety of F2's 0.056 s max. In flight the same run's F2 max was
        0.021 s.
        """
        if self._window_open_wall_ns is None or now_wall_ns < self._window_open_wall_ns:
            return
        self._flight_end_wall_ns = now_wall_ns
    def _window_end_wall_ns(self, now_wall_ns: int) -> int:
        """The window's end: now, or the close, whichever came first."""
        if self._window_closed_wall_ns is None:
            return now_wall_ns
        return min(now_wall_ns, self._window_closed_wall_ns)

    def on_published(self, state: EstimatorState, now_wall_ns: int, newest_imu_ns: int) -> None:
        """Record one publication for the freshness accounting (F2, F3).

        Only inside the scored window. Publications after ``close_window`` are the
        harness's teardown, not the flight: the drain loop has exited, so no new
        offer can arrive, and the publisher keeps re-sending the last offered state
        against a frozen newest-IMU stamp — exactly the frozen-tail pathology the
        window's declared end (arm to the flight's end) exists to exclude, and the
        same one E1's accounting was fixed for in b48b44d. Measured, run
        p01l-fix5-20260927T163256Z: the flight's own 1878 publications aged at most
        ~0.019 s, then 28 teardown publications carried one frozen state (t=54.030)
        at a constant 0.214 s for 316 ms until the thread stopped, and that teardown
        tail was 100 % of the run's F2 failure. H1's ``valid_fraction`` already ends
        at ``_window_end_wall_ns``; F2/F3 now do too.
        """
        end = self._window_end_wall_ns(now_wall_ns)
        if self._flight_end_wall_ns is not None:
            end = min(end, self._flight_end_wall_ns)
        if now_wall_ns > end:
            return
        if self._last_publish_wall_ns is not None:
            self.publish_gaps_s.append((now_wall_ns - self._last_publish_wall_ns) / 1e9)
        self._last_publish_wall_ns = now_wall_ns
        self.published_state_ages_s.append(max(0.0, (newest_imu_ns - state.time_ns) / 1e9))

    def visual_update_verdict(self, state: EstimatorState) -> str:
        """The visual-update age verdict for one state: ok, warn or fail (F4)."""
        age_s = (state.time_ns - state.t_last_visual_ns) / 1e9
        if age_s >= self.bounds.visual_update_fail_s:
            return "fail"
        if age_s >= self.bounds.visual_update_warn_s:
            return "warn"
        return "ok"

    def valid_fraction(self, now_wall_ns: int) -> float:
        """The healthy fraction of the scored window so far (H1), outages charged whole."""
        if self._window_open_wall_ns is None:
            return 1.0
        end_wall_ns = self._window_end_wall_ns(now_wall_ns)
        window_s = (end_wall_ns - self._window_open_wall_ns) / 1e9
        if window_s <= 0.0:
            return 1.0
        stopped_s = sum(self.outages_s)
        if self._stopped_at_wall_ns is not None and self._stopped_at_wall_ns < end_wall_ns:
            stopped_s += (end_wall_ns - self._stopped_at_wall_ns) / 1e9
        return max(0.0, 1.0 - stopped_s / window_s)


# ---------------------------------------------------------------------------
# The estimator client
# ---------------------------------------------------------------------------


class OvStreamClient:
    """The socket client for the pinned estimator process.

    Sends feed frames and pulls at most the newest STATE per poll; a dead or
    silent socket raises, and the caller stops the machine — a crashed or hung
    estimator is indistinguishable from a dead sensor, which is exactly the
    unresolved protocol. Silence is not death: a poll with nothing to read returns
    None, and only a closed or failed connection ends the stream.
    """

    def __init__(self, host: str, port: int, timeout_s: float = 1.0) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._buffer = b""

    def connect(self) -> None:
        self._socket = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
        # The socket stays blocking, and that is deliberate. The feed is a bulk,
        # strictly ordered stream: a stereo pair is 614 KB, so a write that meets a
        # full send buffer must wait for the estimator to consume it rather than be
        # reported as a dead link or silently dropped -- dropping the newest frame
        # would leave the estimator with a stream whose frames no longer line up with
        # the inertial samples that were meant to precede them. Reading, by contrast,
        # must never wait, so polling asks select first and never blocks on the
        # estimator's silence.
        self._socket.settimeout(None)

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def send(self, frame: bytes) -> None:
        if self._socket is None:
            raise ProtocolError("the estimator client is not connected")
        try:
            self._socket.sendall(frame)
        except OSError as error:
            self._lost(f"send failed: {error}")

    def poll_state(self) -> EstimatorState | None:
        """Return the newest complete STATE frame, or None when none has arrived."""
        if self._socket is None:
            raise ProtocolError("the estimator client is not connected")
        try:
            readable, _writable, _errored = select.select([self._socket], [], [], 0)
        except (OSError, ValueError) as error:
            self._lost(f"the estimator socket could not be polled: {error}")
        if not readable:
            return None
        try:
            data = self._socket.recv(1 << 16)
        except InterruptedError:
            return None
        except OSError as error:
            self._lost(f"read failed: {error}")
        if not data:
            self._lost("the estimator closed the connection")
        self._buffer += data
        state: EstimatorState | None = None
        while True:
            frame, rest = self._next_frame()
            self._buffer = rest
            if frame is None:
                break
            state = decode_state(frame)
        return state

    def _next_frame(self) -> tuple[bytes | None, bytes]:
        buffer = self._buffer
        if len(buffer) < FRAME_HEADER_SIZE:
            return None, buffer
        magic, _kind, _flags, length = FRAME_HEADER.unpack_from(buffer)
        if magic != FRAME_MAGIC:
            raise ProtocolError(f"frame magic 0x{magic:04x} is not the ov_stream magic")
        total = FRAME_HEADER_SIZE + length
        if len(buffer) < total:
            return None, buffer
        return buffer[:total], buffer[total:]

    def _lost(self, detail: str) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
        self._buffer = b""
        raise ProtocolError(f"estimator connection lost: {detail}")


# ---------------------------------------------------------------------------
# The ExternalNav publisher
# ---------------------------------------------------------------------------


def publish_period_s() -> float:
    """The seam cadence: 10 ms (100 Hz), with the EKF delay corrected.

    25 ms through FIXER5's first four flights. The declared F2 bound
    (published-state age max 20 ms) is smaller than one 25 ms publish slot on
    both this tick and ov_stream's PUBLISH_PERIOD_S puts the floor at one slot
    per side. This is the retry after the first attempt (db3153f, 2026-09-27)
    flew clean twice and then lost two runs mid-route: FIXER5 then measured the
    EKF's ext-nav fusion delay and found it declared 50 ms against a pose that
    arrives p50 2.5-5.7 ms after its own validity stamp, and set VISO_DELAY_MS
    to the pin's default 10 ms (commit 03e1213). The rate and the delay are one
    coupled choice: at 100 Hz with the old 50 ms delay several updates contended
    for the same back-dated IMU window.

    EKF3 still fuses external navigation at most every 20 ms
    (AP_NavEKF3.h:516, extNavIntervalMin_ms) — faster sends are dropped silently
    at the EKF's writeExtNavData gate, while the vision-position health window
    (AP_VisualOdom_Backend.cpp:32-35) is a 300 ms timeout and is unaffected. The
    declared localization.publish.period_ms carries the same 10 ms.
    """
    return 0.010


class ExternalNavPublisher:
    """Publishes the aligned estimator state to SITL at the seam's 10 ms cadence.

    Two message types on the adapter's own MAVLink TCP connection — the gate's
    discipline (ALLOWED_OUTBOUND_TYPES, VISION_POSE_PERIOD_S = 0.025 at the
    pinned bridge), with the same source identity the gate's feed used. The usec
    stamp is sim time exactly as the gate published it; the declared pipeline
    delay rides ``VISO_DELAY_MS``, never a re-stamped field. Covariance entries
    0/6/11 carry the estimator's σ² per axis; the firmware computes posErr from
    them and floors it at ``VISO_POS_M_NSE``.

    The endpoint is resolved before ``start()`` and must be a port the autopilot
    actually serves: the pinned SITL accepts one TCP client per serial port, so a
    port another client owns yields a connection that is accepted and never read.
    """

    def __init__(
        self,
        endpoint: str,
        alignment: OdomAlignment,
        machine: HealthMachine,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_publish: Callable[[EstimatorState, dict[str, object]], None] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._alignment = alignment
        self._machine = machine
        self._clock = clock
        self._on_publish = on_publish
        self._latest: EstimatorState | None = None
        self._latest_offer_s: float | None = None
        # The state clock's freeze watch: the wall time the current state clock was
        # first seen, so a clock that stops advancing is detected by the same declared
        # bound that governs a silent feed (see ``state_for_publish``).
        self._clock_first_seen_s: float | None = None
        self._clock_seen_ns: int | None = None
        self._newest_imu_ns = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection = None
        self.published = 0
        self.publish_failures: list[str] = []

    def retarget(self, endpoint: str) -> None:
        """Point the publisher at the autopilot link it will open in ``start()``.

        Called once, after SITL is running and has declared its serial ports, and
        before any publication. The adapter's endpoint is a property of the live
        process, not of the configuration: the configured port belongs to the
        check's own session, and the pinned SITL serves one client per port.
        """
        if self._connection is not None:
            raise RuntimeError("the publisher is already connected; retarget before start()")
        self._endpoint = endpoint

    def offer(self, state: EstimatorState, newest_imu_ns: int) -> None:
        """Hand one candidate state to the publisher's loop."""
        self._latest = state
        self._latest_offer_s = self._clock()
        self._newest_imu_ns = max(self._newest_imu_ns, newest_imu_ns)

    def state_for_publish(self, now_s: float) -> EstimatorState | None:
        """The state the machine may act on, or None once the feed has gone silent.

        ``state_lost_after_ms`` is a declared bound -- "300 ms of silence stops
        transmission: the adapter never publishes a stale estimate" -- and the machine's
        watchdog (``HealthMachine._timeout_if_silent``) is the only thing that enforces
        it. That watchdog runs on ``on_state(None, ...)``, so a publisher that always
        hands over its last state leaves the bound unreachable and keeps a frozen pose on
        the wire for as long as the process lives. Measured, run
        p01l-zupt5-20260927T042005Z: 332 identical publications over 10.13 s after the
        simulator had stopped, 32 % of the scored window. Staleness is measured from the
        newest OFFER, so a live feed at any rate is unaffected.

        The same declared bound also governs the state's OWN clock, because a feed can
        keep offering while the estimate behind it is frozen. Measured on the four
        mid-climb tumbles of 2026-09-27 (p01l-fix5c/-5d/-7c/-flightscope): ov_stream
        kept publishing at its 100 Hz tick while its state clock stood still -- a burst
        of ~614 KB stereo frames on the one shared, ordered, blocking connection kept
        the inertial samples behind it unconsumed, so the published state carried a
        healthy sigma and a pose frozen at the last accepted image while the vehicle
        climbed. The adapter republished that confident frozen pose for whole seconds
        until the backlog cleared and the pose stepped 0.4-0.7 m mid-air; the yaw
        excursion and the roll-yaw flip followed within 1.5 s. A clock that has not
        advanced for ``state_lost_after_s`` is the same stale estimate the bound
        already forbids, so it hands the machine None the same way, and the declared
        silence path takes over. A live feed advances its clock with every inertial
        sample (2 ms at the declared 500 Hz), so only a stalled estimator can trip it.
        """
        if self._latest is None or self._latest_offer_s is None:
            return None
        if now_s - self._latest_offer_s > self._machine.bounds.state_lost_after_s:
            return None
        if self._clock_seen_ns != self._latest.time_ns:
            self._clock_seen_ns = self._latest.time_ns
            self._clock_first_seen_s = now_s
        elif (
            self._clock_first_seen_s is not None
            and now_s - self._clock_first_seen_s
            > self._machine.bounds.state_lost_after_s
        ):
            return None
        return self._latest

    def start(self, heartbeat_timeout_s: float = 15.0) -> None:
        """Open the adapter's own autopilot link and start publishing.

        The heartbeat is CHECKED, not assumed. The pinned SITL serves exactly one
        TCP client per serial port (UARTDriver.cpp: a single accept(), then
        ``_connected``), so a connection to a port another client already owns is
        accepted by the kernel and never read: sends succeed locally and nothing
        reaches the autopilot. The third textured invocation measured exactly that
        -- 2404 publications, ``VisOdom: not healthy`` -- so an unanswered
        heartbeat is a raised error, not a silent write into a void.
        """
        try:
            from pymavlink import mavutil
        except ImportError as error:  # pragma: no cover - pinned dependency
            raise RuntimeError(f"pymavlink is required to publish external nav: {error}") from error
        self._connection = mavutil.mavlink_connection(
            self._endpoint, source_system=250, source_component=190
        )
        if self._connection.wait_heartbeat(timeout=heartbeat_timeout_s) is None:
            self._connection.close()
            self._connection = None
            raise RuntimeError(
                f"no MAVLink heartbeat on {self._endpoint} within "
                f"{heartbeat_timeout_s:.0f}s: the autopilot does not serve this link, so "
                "publishing into it would be a silent void (the pinned SITL accepts one "
                "TCP client per serial port)"
            )
        self._thread = threading.Thread(target=self._loop, name="externalnav-publish", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    def _loop(self) -> None:
        next_send = self._clock()
        while not self._stop.is_set():
            now = self._clock()
            if now < next_send:
                if self._stop.wait(next_send - now):
                    return
                continue
            next_send = now + publish_period_s()
            state = self.state_for_publish(now)
            if not self._machine.on_state(state, int(now * 1e9)):
                continue
            # The epoch's one-time rotation: the first state the machine accepts as
            # healthy is the estimator's attitude at the declared stationary start
            # (initialization happens before any publication), and it is the only
            # observation that can define the odom frame's unobservable yaw. Sealed
            # here, once, then never moved.
            if not self._alignment.sealed:
                self._alignment.seal(state.quat_wxyz)
            try:
                self._send_state(state)
            except (OSError, ProtocolError, RuntimeError) as error:
                self.publish_failures.append(f"{now:.3f}: {error}")
                self._machine.stop(int(now * 1e9), f"publish failed: {error}")
                return
            self._machine.on_published(state, int(now * 1e9), self._newest_imu_ns)
            if self._on_publish is not None:
                self._on_publish(state, self._alignment.aligned_state(state))
            self.published += 1

    def _send_state(self, state: EstimatorState) -> None:
        aligned = self._alignment.aligned_state(state)
        position = aligned["position_ned_m"]
        rpy = aligned["attitude_rpy"]
        velocity = aligned["velocity_ned_mps"]
        usec = int(round(state.time_ns / 1000.0))
        sigma = state.sigma_pos_m
        covariance = [0.0] * 21
        covariance[0] = sigma[0] ** 2
        covariance[6] = sigma[1] ** 2
        covariance[11] = sigma[2] ** 2
        # The reset counter rides the wire as one unsigned byte (MAVLink's
        # reset_counter field); a run whose health machine cycles stop/recover
        # more than 255 times overflows it and raises struct.error inside
        # pymavlink's packer, killing this thread -- measured in run
        # p01l-fix2l-20260927T033814Z, where the ZUPT defect produced 292
        # recoveries and the publisher died with "'B' format requires 0 <=
        # number <= 255". The counter's meaning on the wire is a change
        # detector, so wrapping it keeps the semantics while making the
        # transmission impossible to overflow; the machine's own count stays
        # exact and is what the receipts report.
        self._connection.mav.vision_position_estimate_send(
            usec,
            position[0],
            position[1],
            position[2],
            rpy[0],
            rpy[1],
            rpy[2],
            covariance,
            self._machine.reset_counter % 256,
        )
        self._connection.mav.vision_speed_estimate_send(
            usec, velocity[0], velocity[1], velocity[2]
        )
