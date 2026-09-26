"""P01-L sensor-derived localization: the estimator adapter and its seams.

The adapter owns three things and nothing else: the socket client for the pinned
estimator process (OpenVINS v2.7 behind ``ov_stream``, plan section 3.4), the
fixed odom-to-local-NED alignment (specification section 6.3), and the
health/freshness state machine that decides whether an estimate may be
published at all (plan sections 4.5 and 6). Publishing rides the same
ExternalNav discipline the P00 gate proved: ``VISION_POSITION_ESTIMATE`` and
``VISION_SPEED_ESTIMATE`` at a 25 ms cadence on the adapter's own MAVLink
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
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "AlignmentError",
    "BringUpLink",
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

        ``published = A · q_odom(t) · FLU→FRD`` with
        ``A = R_declared_start · FLU→FRD · q_odom(start)⁻¹``: one fixed
        odom→NED rotation per epoch, the estimator's own initialization frame
        related to the autopilot's frame by the two things known at the declared
        stationary start. Idempotent: the first seal wins and the rotation never
        moves afterwards.
        """
        if self._epoch_rotation is not None:
            return
        initial = quat_to_rotmat(initial_quat_odom_wxyz)
        declared = rotmat_from_rpy(self._declared_start_rpy)
        self._epoch_rotation = declared @ FLU_TO_FRD_AXES @ initial.T

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

    def aligned_quat_ned_wxyz(
        self, quat_odom_wxyz: Sequence[float]
    ) -> tuple[float, float, float, float]:
        rotation = self.epoch_rotation @ quat_to_rotmat(quat_odom_wxyz) @ FLU_TO_FRD_AXES
        return rotmat_to_quat(rotation)

    def aligned_attitude_rpy(self, quat_odom_wxyz: Sequence[float]) -> tuple[float, float, float]:
        rotation = self.epoch_rotation @ quat_to_rotmat(quat_odom_wxyz) @ FLU_TO_FRD_AXES
        return rotmat_to_rpy(rotation)

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
        self._stopped_at_wall_ns: int | None = None

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
        self._last_publish_wall_ns = None
        self.publish_gaps_s = []
        self.published_state_ages_s = []
        self.outages_s = []

    def on_published(self, state: EstimatorState, now_wall_ns: int, newest_imu_ns: int) -> None:
        """Record one publication for the freshness accounting (F2, F3)."""
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
        window_s = (now_wall_ns - self._window_open_wall_ns) / 1e9
        if window_s <= 0.0:
            return 1.0
        stopped_s = sum(self.outages_s)
        if self._stopped_at_wall_ns is not None:
            stopped_s += (now_wall_ns - self._stopped_at_wall_ns) / 1e9
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
    """The proven seam cadence: 25 ms, inside the filter's 20 ms minimum."""
    return 0.025


class ExternalNavPublisher:
    """Publishes the aligned estimator state to SITL at the proven cadence.

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
        self._newest_imu_ns = max(self._newest_imu_ns, newest_imu_ns)

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
            state = self._latest
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
        self._connection.mav.vision_position_estimate_send(
            usec,
            position[0],
            position[1],
            position[2],
            rpy[0],
            rpy[1],
            rpy[2],
            covariance,
            self._machine.reset_counter,
        )
        self._connection.mav.vision_speed_estimate_send(
            usec, velocity[0], velocity[1], velocity[2]
        )
# ---------------------------------------------------------------------------
# The ordered bring-up's own link to the autopilot (plan sections 0.6 item 6
# and 0.8 item 7, Deliverable B)
# ---------------------------------------------------------------------------
#
# The bridge's ``PymavlinkSession`` refuses every message type outside its own
# ``ALLOWED_OUTBOUND_TYPES`` (webots_ardupilot.py:1951-1958), and two of the
# bring-up's declarations are not in that set: the EKF origin datum
# (SET_GPS_GLOBAL_ORIGIN) and the window's parameter writes (PARAM_SET). Adding
# them to the bridge's list would widen what the SCORED arm can put on the wire,
# and that file is outside this slice's bounded ownership, so the bring-up sends
# them on its own connection -- the pattern this module already established for
# the estimator's seam, where ``ExternalNavPublisher`` opens its own autopilot
# link rather than borrowing the session's.
#
# Exactly three outbound types are allowed here, and each one is a declaration
# the plan names. The link carries no pose and no motion of its own beyond the
# one bounded takeoff the window declares.
BRING_UP_ALLOWED_OUTBOUND_TYPES = frozenset(
    {
        # The EKF ORIGIN datum. This is a frame definition, not a pose feed:
        # it names the geographic anchor of the local frame the vehicle's own
        # pose is measured in, and the vehicle's attitude and position are its
        # own. EKF3 never sets that origin from ExternalNav data
        # (AP_NavEKF3_Measurements.cpp:680-715) -- only from GPS, a beacon, or
        # this GCS declaration (``GCS_MAVLINK::set_ekf_origin``,
        # GCS_Common.cpp:3961, reached by SET_GPS_GLOBAL_ORIGIN at :3982-4014)
        # -- so a GPS-off flight must declare it or nothing has an origin.
        "SET_GPS_GLOBAL_ORIGIN",
        # The declared exception window's parameter writes: the ARMING_CHECK
        # mask and the window's source selection, each restored with the
        # vehicle's own readback before the claimed arm.
        "PARAM_SET",
        # The ALT_HOLD excitation's takeoff, with the one flag that mode needs
        # (``webots_ardupilot.py:2248-2264`` sends param3 = 0, which makes
        # ``must_navigate`` true at ArduCopter/GCS_Mavlink_Copter.cpp:594 and
        # refuses ALT_HOLD's user takeoff at mode.h:512-514).
        "COMMAND_LONG",
    }
)

# MAV_CMD_NAV_TAKEOFF and its documented param3 flag: "the horizontal position
# is not required to take off" (common.xml:1005). ALT_HOLD's takeoff is a
# manual-throttle-mode climb, not a navigation, so the flag is what makes it
# legal for that mode; without it the command is refused outright.
MAV_CMD_NAV_TAKEOFF = 22
NAV_TAKEOFF_FLAGS_HORIZONTAL_POSITION_NOT_REQUIRED = 1


class BringUpLink:
    """The declared ordered bring-up's own MAVLink connection to the autopilot.

    Three sends, each recorded as it goes out so the receipt states exactly what
    the window declared rather than what it intended to declare: the origin
    datum, one parameter write at a time, and the bounded excitation's takeoff.
    Every reply the autopilot makes on this link is kept beside them -- a
    COMMAND_ACK's result is the vehicle's own answer to the takeoff, and a
    PARAM_VALUE or PARAM_ERROR is its answer to a write.

    The endpoint must be a port the running autopilot actually serves: the
    pinned SITL accepts one TCP client per serial port, so a port another client
    owns yields a connection that is accepted and never read. An unanswered
    heartbeat is therefore a raised error, exactly as the publisher's is.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        source_system: int = 250,
        source_component: int = 191,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._endpoint = endpoint
        self._source_system = source_system
        self._source_component = source_component
        self._clock = clock
        self._connection: Any = None
        self._mavutil: Any = None
        self.target_system = 0
        self.target_component = 0
        self.sent: list[dict[str, Any]] = []
        self.replies: list[dict[str, Any]] = []

    def connect(self, timeout_s: float = 15.0) -> None:
        try:
            from pymavlink import mavutil
        except ImportError as error:  # pragma: no cover - pinned dependency
            raise RuntimeError(f"pymavlink is required for the bring-up link: {error}") from error
        self._mavutil = mavutil
        self._connection = mavutil.mavlink_connection(
            self._endpoint, source_system=self._source_system, source_component=self._source_component
        )
        if self._connection.wait_heartbeat(timeout=timeout_s) is None:
            self._connection.close()
            self._connection = None
            raise RuntimeError(
                f"no MAVLink heartbeat on {self._endpoint} within {timeout_s:.0f}s: the "
                "autopilot does not serve this link, so the bring-up's declarations "
                "would be a silent void (the pinned SITL accepts one TCP client per "
                "serial port)"
            )
        self.target_system = self._connection.target_system
        self.target_component = self._connection.target_component

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    def _send(self, message: Any, declaration: dict[str, Any]) -> dict[str, Any]:
        message_type = message.get_type()
        if message_type not in BRING_UP_ALLOWED_OUTBOUND_TYPES:
            raise RuntimeError(
                f"refusing to send {message_type} on the bring-up link: the declared "
                "bring-up carries the origin datum, the window's parameter writes and "
                "the bounded excitation's takeoff, and nothing else"
            )
        if self._connection is None:
            raise RuntimeError("the bring-up link is not connected")
        self._connection.mav.send(message)
        record = {"at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **declaration}
        self.sent.append(record)
        return record

    def declare_origin_datum(
        self, latitude_deg: float, longitude_deg: float, altitude_m: float
    ) -> dict[str, Any]:
        """Declare the local frame's origin once (SET_GPS_GLOBAL_ORIGIN, msgid 48).

        The message carries the frame's geographic anchor and no vehicle state:
        ArduPilot's handler converts it to a ``Location`` and calls
        ``set_ekf_origin``, which refuses a re-declaration
        (``get_origin`` already set -> MAV_RESULT_FAILED), so this is a datum set
        once per invocation. The altitude is absolute (MSL) and goes on the wire
        in millimetres, which is the message's declared unit.
        """
        message = self._connection.mav.set_gps_global_origin_encode(
            self.target_system,
            int(round(latitude_deg * 1e7)),
            int(round(longitude_deg * 1e7)),
            int(round(altitude_m * 1e3)),
            int(round(self._clock() * 1e6)),
        )
        return self._send(
            message,
            {
                "message": "SET_GPS_GLOBAL_ORIGIN",
                "message_id": 48,
                "latitude_deg": latitude_deg,
                "longitude_deg": longitude_deg,
                "altitude_msl_m": altitude_m,
                "meaning": (
                    "the local frame's origin datum: a frame definition set once, "
                    "carrying no vehicle pose"
                ),
            },
        )

    def set_parameter(self, name: str, value: float) -> dict[str, Any]:
        """Write one parameter (PARAM_SET). The write is verified by readback."""
        message = self._connection.mav.param_set_encode(
            self.target_system,
            self.target_component,
            name.encode("ascii"),
            float(value),
            self._mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        return self._send(
            message, {"message": "PARAM_SET", "name": name, "value": float(value)}
        )

    def takeoff_without_horizontal_position(self, altitude_m: float) -> dict[str, Any]:
        """The excitation's takeoff: MAV_CMD_NAV_TAKEOFF with the flag ALT_HOLD needs.

        ``param3 = NAV_TAKEOFF_FLAGS_HORIZONTAL_POSITION_NOT_REQUIRED`` is the
        documented declaration that the climb is not a navigation
        (common.xml:1005): it makes ``must_navigate`` false
        (GCS_Mavlink_Copter.cpp:594), which is what lets a mode that requires no
        position accept a user takeoff (mode.h:512-514). ``param7`` is the
        bounded excitation's own altitude.
        """
        message = self._connection.mav.command_long_encode(
            self.target_system,
            self.target_component,
            MAV_CMD_NAV_TAKEOFF,
            0,
            0.0,
            0.0,
            float(NAV_TAKEOFF_FLAGS_HORIZONTAL_POSITION_NOT_REQUIRED),
            0.0,
            0.0,
            0.0,
            float(altitude_m),
        )
        return self._send(
            message,
            {
                "message": "COMMAND_LONG",
                "command": MAV_CMD_NAV_TAKEOFF,
                "command_name": "MAV_CMD_NAV_TAKEOFF",
                "param3": NAV_TAKEOFF_FLAGS_HORIZONTAL_POSITION_NOT_REQUIRED,
                "param3_meaning": "NAV_TAKEOFF_FLAGS_HORIZONTAL_POSITION_NOT_REQUIRED",
                "param7_altitude_m": float(altitude_m),
            },
        )

    def drain(self) -> list[dict[str, Any]]:
        """Every reply that has arrived since the last call, stamped as it is read."""
        if self._connection is None:
            return []
        arrived: list[dict[str, Any]] = []
        while len(arrived) < 200:
            message = self._connection.recv_match(blocking=False)
            if message is None:
                break
            document = message.to_dict()
            document.setdefault("mavpackettype", message.get_type())
            document["received_at_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
            arrived.append(document)
        self.replies.extend(arrived)
        return arrived

    def command_ack(self, command: int) -> dict[str, Any] | None:
        """The autopilot's own answer to one command, if it has arrived."""
        for reply in reversed(self.replies):
            if reply.get("mavpackettype") == "COMMAND_ACK" and reply.get("command") == command:
                return reply
        return None
