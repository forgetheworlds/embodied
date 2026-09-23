"""Webots/ArduPilot platform adapter and the compatibility probe.

This is a deep module: a small surface in front of four awkward mechanisms, which
are process launch, an MAVLink session, the sensor stream between the Webots
controller and this process, and fault injection. A caller that wants to know
whether the simulator path works calls :func:`run_compatibility_probe`; everything
else here exists to make that function honest.

Three seams are replaceable, so the probe's own logic is tested without Webots,
SITL or a real autopilot:

* :class:`ProcessRunner` spawns and terminates the two child processes.
* :class:`MavlinkSession` speaks to the autopilot. It is the only path by which
  this program commands the aircraft, and it accepts only the message types in
  :data:`ALLOWED_OUTBOUND_TYPES`. There is no motor, throttle or pulse command
  anywhere in this project: motion reaches the vehicle as a supported MAVLink
  guided setpoint or not at all.
* :class:`SensorGateway` carries the controller's sensor records to this process
  and injection commands back to the controller.

Three formats live here rather than in two places. The Webots controller imports
:func:`pack_fdm` and :func:`unpack_controls` for its UDP exchange with SITL, whose
layout is fixed by the pinned ArduPilot model. It imports the ``EMB1`` framing,
:func:`bgra_to_rgb8` and :class:`OutboundStream` for the sensor stream back to this
process. Each format has its own test, and the socket layouts are checked against
the pinned C structures rather than against themselves.

Only the standard library is imported at module import time; numpy, pymavlink and
YAML are imported inside the functions that need them, so the Webots controller
can import the framing under whatever interpreter Webots handed it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import struct
import subprocess
import time
from typing import Any, Callable, Iterable, Sequence

from embodied.cli import (
    CommandError,
    repository_root,
    CommandOutcome,
    CommandStatus,
    ConfigError,
    GateStatus,
    load_config,
    register_command,
)
from embodied.contracts.records import (
    ClockStamp,
    Frame,
    FrameQuality,
    MotionSetpoint,
    MotionTarget,
    Observation,
    RECORDS_REVISION,
    SensorIds,
    SensorMode,
    RecordError,
    SetpointSource,
    TYPE_MASK_ACCELERATION,
    TYPE_MASK_FORCE,
    TYPE_MASK_POSITION,
    TYPE_MASK_VELOCITY,
    TYPE_MASK_YAW,
    TYPE_MASK_YAW_RATE,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PlatformUnavailable(Exception):
    """A prerequisite is missing. The command is blocked, not failed."""


class ProbeFailure(Exception):
    """A probe step could not produce its evidence. The run is invalid, not judged."""


class FramingError(Exception):
    """A sensor message is malformed, truncated or of an unknown version."""


# ---------------------------------------------------------------------------
# The SITL wire: sixteen servo values in, sixteen doubles of flight state out
# ---------------------------------------------------------------------------

# Both layouts are fixed by ArduPilot's Webots-Python model at the pinned commit
# (libraries/SITL/SIM_Webots_Python.cpp): a 16-float servo packet the simulator
# receives, and a 16-double flight-state packet it sends back. The struct formats
# carry no byte-order prefix because the firmware packs host order and both ends
# run on this machine; changing that would break the exchange silently. The sizes
# are 64 and 128 bytes, and the test pins them against the C structures.
CONTROL_FORMAT = "f" * 16
CONTROL_SIZE = struct.calcsize(CONTROL_FORMAT)
FDM_FORMAT = "d" * (1 + 3 + 3 + 3 + 3 + 3)
FDM_SIZE = struct.calcsize(FDM_FORMAT)


@dataclass(frozen=True)
class FlightState:
    """One flight-state packet the simulator sends to SITL, in ArduPilot's NED frame."""

    timestamp_s: float
    gyro_rpy: tuple[float, float, float]
    accel_xyz: tuple[float, float, float]
    attitude_rpy: tuple[float, float, float]
    velocity_xyz: tuple[float, float, float]
    position_xyz: tuple[float, float, float]


def enu_to_ned(values: Sequence[float]) -> tuple[float, float, float]:
    """Convert a Webots ENU triple into ArduPilot's NED frame.

    Webots measures east-north-up; ArduPilot's local frame is north-east-down. The
    pinned bridge converts by keeping x and negating y and z, and this adapter
    follows it exactly rather than inventing a second convention. The conversion is
    its own inverse, so the same function converts back.
    """
    if len(values) != 3:
        raise FramingError("a frame conversion takes three components")
    return (float(values[0]), -float(values[1]), -float(values[2]))


def unpack_controls(packet: bytes) -> tuple[float, ...]:
    """Read SITL's control packet into one fraction per channel.

    SITL sends ``(pulse_width - 1000) / 1000`` for each of its sixteen channels, so
    a value is a fraction of full throttle and a negative one marks a channel the
    autopilot is not using. Reading is the only direction this project needs: the
    motor values are the simulator model's business, and nothing here builds one.
    """
    if len(packet) != CONTROL_SIZE:
        raise FramingError(f"control packet is {len(packet)} bytes, expected {CONTROL_SIZE}")
    return struct.unpack(CONTROL_FORMAT, packet)


def pack_fdm(state: FlightState) -> bytes:
    """Pack flight state for SITL, in the field order the pinned model reads."""
    return struct.pack(
        FDM_FORMAT,
        state.timestamp_s,
        *state.gyro_rpy,
        *state.accel_xyz,
        *state.attitude_rpy,
        *state.velocity_xyz,
        *state.position_xyz,
    )


def propeller_velocity(fraction: float, *, max_velocity: float) -> float:
    """The angular velocity one motor should take for a throttle fraction.

    A propeller's thrust is quadratic in its angular velocity, while ArduPilot's
    throttle model with ``MOT_THST_EXPO 0`` is linear in the fraction it sends. The
    square root of the magnitude is what makes the two agree, which is why the pinned
    parameter file sets that exponential to zero. The Webots controller calls this
    rather than repeating the conversion on its own side.
    """
    linearized = math.copysign(math.sqrt(abs(fraction)), fraction)
    return linearized * max_velocity


# ---------------------------------------------------------------------------
# Sensor framing: one definition, imported by the Webots controller
# ---------------------------------------------------------------------------

MAGIC = b"EMB1"
FORMAT_VERSION = 1
# magic, format version, kind, flags, simulation time, sequence, payload length
HEADER_FORMAT = ">4sHBHdII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class Kind(IntEnum):
    STATUS = 1
    PAIR = 2
    IMU = 3
    FAULT = 4
    FAULT_ACK = 5


@dataclass(frozen=True)
class Message:
    """One framed message from the sensor stream or the fault channel."""

    kind: Kind
    flags: int
    sim_time_s: float
    sequence: int
    payload: bytes


def pack_message(
    kind: Kind, *, sim_time_s: float, sequence: int, payload: bytes = b"", flags: int = 0
) -> bytes:
    """Frame one message. The layout is fixed here and versioned by FORMAT_VERSION."""
    if not isinstance(kind, Kind):
        raise FramingError(f"unknown message kind {kind!r}")
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        int(kind),
        int(flags),
        float(sim_time_s),
        int(sequence),
        len(payload),
    )
    return header + payload


def read_message(buffer: bytes) -> tuple[Message, int]:
    """Read one message from ``buffer``, returning it and how many bytes it used.

    A truncated header or payload raises rather than yielding a short frame: a
    partial frame accepted quietly is a partial sensor record presented as a whole
    one.
    """
    if len(buffer) < HEADER_SIZE:
        raise FramingError(f"need {HEADER_SIZE} bytes of header, have {len(buffer)}")
    magic, version, kind, flags, sim_time_s, sequence, payload_len = struct.unpack(
        HEADER_FORMAT, buffer[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise FramingError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version != FORMAT_VERSION:
        raise FramingError(f"format version {version} is not {FORMAT_VERSION}")
    try:
        message_kind = Kind(kind)
    except ValueError:
        raise FramingError(f"unknown message kind {kind}") from None
    end = HEADER_SIZE + payload_len
    if len(buffer) < end:
        raise FramingError(
            f"payload needs {payload_len} bytes, only {len(buffer) - HEADER_SIZE} arrived"
        )
    message = Message(
        kind=message_kind,
        flags=flags,
        sim_time_s=sim_time_s,
        sequence=sequence,
        payload=buffer[HEADER_SIZE:end],
    )
    return message, end


class FrameReader:
    """Incremental reader for the sensor stream.

    Bytes arrive in whatever chunks the socket delivers, so the reader buffers them
    and hands back whole messages only. A message that has not fully arrived is not
    returned, which is what stops half a stereo pair from looking complete.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        self._buffer.extend(data)

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def next_message(self) -> Message | None:
        """The next complete message, or None while more bytes are needed.

        The header is read first so an incomplete message is simply not available yet;
        a header that is complete but wrong is a real framing fault and is raised.
        """
        if len(self._buffer) < HEADER_SIZE:
            return None
        magic, version, kind, _, _, _, payload_length = struct.unpack(
            HEADER_FORMAT, bytes(self._buffer[:HEADER_SIZE])
        )
        if magic != MAGIC:
            raise FramingError(f"bad magic {magic!r}, expected {MAGIC!r}")
        if version != FORMAT_VERSION:
            raise FramingError(f"format version {version} is not {FORMAT_VERSION}")
        try:
            Kind(kind)
        except ValueError:
            raise FramingError(f"unknown message kind {kind}") from None
        if len(self._buffer) < HEADER_SIZE + payload_length:
            return None
        message, consumed = read_message(bytes(self._buffer))
        del self._buffer[:consumed]
        return message


# The controller's outgoing queue bound. A 640x480 rgb8 pair is about 1.8 MB and the
# cameras produce ten a second, so a reader that pauses for a few hundred milliseconds
# leaves a queue of several megabytes behind. This bound covers that pause; beyond it
# the newest frame is dropped and counted rather than the stream being corrupted.
MAX_QUEUED_STREAM_BYTES = 8 << 20


class OutboundStream:
    """Whole frames queued for a non-blocking socket, with a bound and a drop count.

    The controller writes into this and then flushes; the reader may be slower than
    the cameras, and a socket that is not ready raises rather than waiting, because
    waiting would stall the simulation itself. Two rules keep the reader's view
    honest. A frame is queued whole, and a frame that has already begun to go out is
    never discarded: half a frame leaves the reader with bytes it cannot
    resynchronise, which is worse than a frame it never sees. When the queue is over
    its bound the newest frame is dropped and counted, so a slow reader receives
    whole frames at a lower rate and the run can report what was lost.
    """

    def __init__(self, *, max_queued_bytes: int = MAX_QUEUED_STREAM_BYTES) -> None:
        self.max_queued_bytes = int(max_queued_bytes)
        self._queued = bytearray()
        self.dropped_frames = 0

    @property
    def queued_bytes(self) -> int:
        return len(self._queued)

    def queue(self, frame: bytes) -> bool:
        """Queue one whole frame. False means it was dropped instead of queued."""
        if len(frame) > self.max_queued_bytes:
            raise FramingError(
                f"a {len(frame)}-byte frame does not fit in a {self.max_queued_bytes}-byte "
                "stream queue, so it could never be sent"
            )
        if len(self._queued) + len(frame) > self.max_queued_bytes:
            self.dropped_frames += 1
            return False
        self._queued += frame
        return True

    def flush(self, sock: socket.socket) -> None:
        """Send what the socket will take now, keeping the rest for the next call.

        A refused send is not an error: the socket buffer is full and the reader has
        not caught up. Anything else is a real fault and is raised for the caller to
        handle, because a stream that has failed silently is a stream that lies.
        """
        while self._queued:
            try:
                sent = sock.send(self._queued)
            except (BlockingIOError, InterruptedError):
                return
            if sent <= 0:
                return
            del self._queued[:sent]

    def reset(self) -> None:
        """Forget queued bytes when the reader is gone. The drop count stays."""
        self._queued.clear()


PAIR_PAYLOAD_FORMAT = ">QIIIHHB"
PAIR_PAYLOAD_SIZE = struct.calcsize(PAIR_PAYLOAD_FORMAT)
PIXEL_CHANNELS = {"rgb8": 3, "bgra8": 4, "gray8": 1}


@dataclass(frozen=True)
class PairPayload:
    """One stereo pair as it travelled over the wire."""

    capture_host_ns: int
    pair_id: int
    left_frame_id: int
    right_frame_id: int
    width: int
    height: int
    encoding: str
    left_bytes: bytes
    right_bytes: bytes


def encode_pair_payload(
    *,
    capture_host_ns: int,
    pair_id: int,
    left_frame_id: int,
    right_frame_id: int,
    width: int,
    height: int,
    encoding: str,
    left_bytes: bytes,
    right_bytes: bytes,
) -> bytes:
    """Pack a stereo pair: both eyes, their counters, and the capture host stamp.

    The host stamp is what lets the reader compare capture and receipt inside one
    clock domain. Simulation time travels in the frame header instead, separately,
    because simulated physics may run faster or slower than the wall clock.
    """
    encoding_bytes = encoding.encode("utf-8")
    if len(encoding_bytes) > 255:
        raise FramingError("encoding tag is too long")
    header = struct.pack(
        PAIR_PAYLOAD_FORMAT,
        capture_host_ns,
        pair_id,
        left_frame_id,
        right_frame_id,
        width,
        height,
        len(encoding_bytes),
    )
    return header + encoding_bytes + left_bytes + right_bytes


def decode_pair_payload(payload: bytes) -> PairPayload:
    if len(payload) < PAIR_PAYLOAD_SIZE:
        raise FramingError("pair payload is shorter than its header")
    (
        capture_host_ns,
        pair_id,
        left_frame_id,
        right_frame_id,
        width,
        height,
        encoding_len,
    ) = struct.unpack(PAIR_PAYLOAD_FORMAT, payload[:PAIR_PAYLOAD_SIZE])
    encoding_end = PAIR_PAYLOAD_SIZE + encoding_len
    if len(payload) < encoding_end:
        raise FramingError("pair payload ends inside its encoding tag")
    encoding = payload[PAIR_PAYLOAD_SIZE:encoding_end].decode("utf-8")
    channels = PIXEL_CHANNELS.get(encoding)
    if channels is None:
        raise FramingError(f"unknown pixel encoding {encoding!r}")
    if width <= 0 or height <= 0:
        raise FramingError("a pair payload must state its frame size")
    pixels = payload[encoding_end:]
    expected = width * height * channels
    if len(pixels) != 2 * expected:
        raise FramingError(
            f"pair payload holds {len(pixels)} pixel bytes, expected {2 * expected} for "
            f"two {width}x{height} {encoding} frames"
        )
    return PairPayload(
        capture_host_ns=capture_host_ns,
        pair_id=pair_id,
        left_frame_id=left_frame_id,
        right_frame_id=right_frame_id,
        width=width,
        height=height,
        encoding=encoding,
        left_bytes=pixels[:expected],
        right_bytes=pixels[expected:],
    )


IMU_PAYLOAD_FORMAT = ">Q9d"
IMU_PAYLOAD_SIZE = struct.calcsize(IMU_PAYLOAD_FORMAT)


@dataclass(frozen=True)
class ImuPayload:
    """One inertial sample: three devices, their device names and their units."""

    capture_host_ns: int
    accelerometer: tuple[float, float, float]
    gyro: tuple[float, float, float]
    inertial_unit_rpy: tuple[float, float, float]
    device_names: tuple[str, str, str]
    units: str


def encode_imu_payload(
    *,
    capture_host_ns: int,
    accelerometer: Sequence[float],
    gyro: Sequence[float],
    inertial_unit_rpy: Sequence[float],
    device_names: Sequence[str],
    units: str,
) -> bytes:
    """Pack one inertial sample with the names and units of the devices it came from."""
    if len(device_names) != 3:
        raise FramingError("an inertial sample names three devices")
    names = [name.encode("utf-8") for name in device_names]
    units_bytes = units.encode("utf-8")
    if any(len(name) > 255 for name in names) or len(units_bytes) > 255:
        raise FramingError("device names and units tags must fit in one byte of length")
    body = struct.pack(
        IMU_PAYLOAD_FORMAT, capture_host_ns, *accelerometer, *gyro, *inertial_unit_rpy
    )
    for name in names:
        body += bytes([len(name)]) + name
    return body + bytes([len(units_bytes)]) + units_bytes


def decode_imu_payload(payload: bytes) -> ImuPayload:
    """Read one inertial sample. A payload that ends early is refused, never padded.

    The fixed part holds the stamp and nine values; each device then contributes a
    one-byte name length followed by its name, and the units tag has the same shape.
    Every length byte is checked before it is read and every field before it is
    sliced, so a truncated payload raises instead of running off the end of the
    buffer, where a short frame could look like a complete one.
    """
    if len(payload) < IMU_PAYLOAD_SIZE + 4:
        raise FramingError("inertial payload is shorter than its header")
    values = struct.unpack(IMU_PAYLOAD_FORMAT, payload[:IMU_PAYLOAD_SIZE])
    offset = IMU_PAYLOAD_SIZE
    names = []
    for _ in range(3):
        if len(payload) <= offset:
            raise FramingError("inertial payload ends before a device name")
        length = payload[offset]
        offset += 1
        if len(payload) < offset + length:
            raise FramingError("inertial payload ends inside a device name")
        names.append(payload[offset : offset + length].decode("utf-8"))
        offset += length
    if len(payload) <= offset:
        raise FramingError("inertial payload ends before its units tag")
    units_len = payload[offset]
    offset += 1
    if len(payload) < offset + units_len:
        raise FramingError("inertial payload ends inside its units tag")
    units = payload[offset : offset + units_len].decode("utf-8")
    return ImuPayload(
        capture_host_ns=values[0],
        accelerometer=(values[1], values[2], values[3]),
        gyro=(values[4], values[5], values[6]),
        inertial_unit_rpy=(values[7], values[8], values[9]),
        device_names=(names[0], names[1], names[2]),
        units=units,
    )


def encode_status_payload(status: dict[str, Any]) -> bytes:
    """Pack the controller's status, which is JSON by design.

    The status message is how the controller reports facts a reader must not guess
    at: the interpreter it runs under, the device names it found, the camera
    sampling periods, the declared rig baseline, the motor names it drives and the
    scene's own declaration of a known-colour object.
    """
    return json.dumps(status, sort_keys=True).encode("utf-8")


def decode_status_payload(payload: bytes) -> dict[str, Any]:
    return _decode_json_payload(payload, "status")


def encode_fault_payload(injection: "Injection") -> bytes:
    """Pack one injection request. The controller acknowledges it separately."""
    return json.dumps(
        {
            "apply": injection.apply,
            "hold_s": injection.hold_s,
            "injection_id": injection.injection_id,
            "kind": injection.kind,
            "magnitude_m": injection.magnitude_m,
        },
        sort_keys=True,
    ).encode("utf-8")


def decode_fault_payload(payload: bytes) -> dict[str, Any]:
    return _decode_json_payload(payload, "fault")


def _decode_json_payload(payload: bytes, what: str) -> dict[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FramingError(f"{what} payload is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise FramingError(f"{what} payload must be a JSON object")
    return document


def bgra_to_rgb8(buffer: bytes, width: int, height: int) -> bytes:
    """Convert Webots' 4-byte-per-pixel camera buffer to packed RGB.

    Webots documents the buffer as BGRA. The pinned example slices off the first
    three bytes and calls the result RGB, which is why channel order is a real
    question here rather than a formality; this conversion takes each channel by
    name and drops alpha, keeping colour instead of averaging it away. The work is
    done with strided slice assignment so it stays in C rather than looping over
    three hundred thousand pixels in Python.
    """
    expected = width * height * 4
    if len(buffer) != expected:
        raise FramingError(f"camera buffer holds {len(buffer)} bytes, expected {expected}")
    out = bytearray(width * height * 3)
    out[0::3] = buffer[2::4]
    out[1::3] = buffer[1::4]
    out[2::3] = buffer[0::4]
    return bytes(out)


def ppm_bytes(rgb: bytes, width: int, height: int) -> bytes:
    """A plain binary PPM, so a human can look at a stored frame without a library."""
    if len(rgb) != width * height * 3:
        raise FramingError("a PPM payload must be packed RGB of the stated size")
    return b"P6\n%d %d\n255\n" % (width, height) + rgb
# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoints:
    """Where the autopilot, the simulator's flight-state socket and the controller listen."""

    sitl: str
    fdm_port: int
    controller_port: int


@dataclass(frozen=True)
class StereoSettings:
    """The declared stereo rig and the criterion for calling a frame colour."""

    left: str
    right: str
    width: int
    height: int
    baseline_m: float
    sampling_period_ms: int
    encoding: str
    # A frame whose three channels agree on at least this fraction of pixels
    # carries no colour information, whatever shape its buffer has. This is the
    # declared criterion behind the boolean in FrameQuality.
    identical_channel_fraction_limit: float


@dataclass(frozen=True)
class ImuSettings:
    accelerometer: str
    gyro: str
    inertial_unit: str
    gps: str
    sampling_period_ms: int


@dataclass(frozen=True)
class EstimatorFault:
    kind: str
    magnitude_m: float
    hold_s: float


@dataclass(frozen=True)
class StepTimeouts:
    startup: float
    ready: float
    flight: float


@dataclass(frozen=True)
class PlatformSettings:
    """The configuration, checked and resolved. Building this performs no I/O."""

    root: Path
    stage: str
    name: str
    ardupilot_root: Path
    ardupilot_commit: str
    sitl_binary: Path
    webots_home: Path
    webots_version: str
    webots_mode: str
    sim_model: str
    vehicle: str
    sitl_home: str
    endpoints: Endpoints
    world: Path
    params: tuple[Path, ...]
    estimator_params: tuple[Path, ...]
    stereo: StereoSettings
    imu: ImuSettings
    calibration_path: Path
    hover_altitude_m: float
    waypoints_local_ned: tuple[tuple[float, float, float], ...]
    hold_per_waypoint_s: float
    stream_loss_window_s: float
    settle_s: float
    at_rest_window_s: float
    estimator_fault: EstimatorFault
    timebase_samples: int
    timebase_spread_limit_ms: float
    step_timeout_s: StepTimeouts
    budget_wall_clock_s: float
    output: str
    host_id: str
    clock_id: str

    @property
    def webots_binary(self) -> Path:
        """The Webots executable inside the configured application bundle."""
        return self.webots_home / "Contents" / "MacOS" / "webots"

    @property
    def webots_version_file(self) -> Path:
        """The installed Webots version, read from the bundle without starting it."""
        return self.webots_home / "Contents" / "Resources" / "version.txt"

    @property
    def sim_port_in(self) -> int:
        """The UDP port SITL binds to receive flight state (SIM_IN_PORT)."""
        return self.endpoints.fdm_port + 1

    @property
    def sim_port_out(self) -> int:
        """The UDP port the controller binds to receive motor commands (SIM_OUT_PORT)."""
        return self.endpoints.fdm_port

    @property
    def mavlink_endpoint(self) -> str:
        """The endpoint string the MAVLink session connects to."""
        return self.endpoints.sitl

    @property
    def stereo_stale_after_s(self) -> float:
        """How long the stereo stream may be silent before it counts as stale.

        Derived from the declared sampling period rather than invented: five
        periods, and never less than a second, so a load spike is not reported as a
        dead camera.
        """
        return max(5.0 * self.stereo.sampling_period_ms / 1000.0, 1.0)

    @property
    def imu_stale_after_s(self) -> float:
        """How long the inertial stream may be silent before it counts as stale.

        The inertial devices have their own configured period, so they get their own
        bound: judging a 10 ms stream against a camera's 100 ms period would call a
        healthy stream stale.
        """
        return max(10.0 * self.imu.sampling_period_ms / 1000.0, 1.0)

    def capture_stamp(self, monotonic_ns: int) -> ClockStamp:
        """A stamp in this host's monotonic domain."""
        return ClockStamp(host_id=self.host_id, clock_id=self.clock_id, monotonic_ns=monotonic_ns)

    @staticmethod
    def from_config(document: dict[str, Any], *, root: Path | None = None) -> "PlatformSettings":
        """Build settings from an already-validated configuration document."""
        base = Path(root).resolve() if root is not None else Path.cwd()

        def resolve(relative: str) -> Path:
            candidate = Path(relative).expanduser()
            return candidate if candidate.is_absolute() else (base / candidate)

        platform = document["platform"]
        stereo = document["sensors"]["stereo"]
        imu = document["sensors"]["imu"]
        probe = document["probe"]
        fault = probe["estimator_fault"]
        timeouts = probe["step_timeout_s"]
        settings = PlatformSettings(
            root=base,
            stage=document["project"]["stage"],
            name=document["project"]["name"],
            ardupilot_root=resolve(platform["ardupilot_root"]),
            ardupilot_commit=platform["ardupilot_commit"],
            sitl_binary=resolve(platform["sitl_binary"]),
            webots_home=resolve(platform["webots_home"]),
            webots_version=platform["webots_version"],
            webots_mode=platform["webots_mode"],
            sim_model=platform["sim_model"],
            vehicle=platform["vehicle"],
            sitl_home=platform["sitl_home"],
            endpoints=Endpoints(
                sitl=platform["endpoints"]["sitl"],
                fdm_port=int(platform["endpoints"]["fdm_port"]),
                controller_port=int(platform["endpoints"]["controller_port"]),
            ),
            world=resolve(document["scenario"]["world"]),
            params=tuple(resolve(name) for name in document["scenario"]["params"]),
            estimator_params=tuple(
                resolve(name) for name in document["scenario"]["estimator_params"]
            ),
            stereo=StereoSettings(
                left=stereo["left"],
                right=stereo["right"],
                width=int(stereo["width"]),
                height=int(stereo["height"]),
                baseline_m=float(stereo["baseline_m"]),
                sampling_period_ms=int(stereo["sampling_period_ms"]),
                encoding=stereo["encoding"],
                identical_channel_fraction_limit=float(
                    stereo["identical_channel_fraction_limit"]
                ),
            ),
            imu=ImuSettings(
                accelerometer=imu["accelerometer"],
                gyro=imu["gyro"],
                inertial_unit=imu["inertial_unit"],
                gps=imu["gps"],
                sampling_period_ms=int(imu["sampling_period_ms"]),
            ),
            calibration_path=resolve(document["sensors"]["calibration"]),
            hover_altitude_m=float(probe["hover_altitude_m"]),
            waypoints_local_ned=tuple(
                (float(step[0]), float(step[1]), float(step[2]))
                for step in probe["waypoints_local_ned"]
            ),
            hold_per_waypoint_s=float(probe["hold_per_waypoint_s"]),
            stream_loss_window_s=float(probe["stream_loss_window_s"]),
            settle_s=float(probe["settle_s"]),
            at_rest_window_s=float(probe["at_rest_window_s"]),
            estimator_fault=EstimatorFault(
                kind=fault["kind"],
                magnitude_m=float(fault["magnitude_m"]),
                hold_s=float(fault["hold_s"]),
            ),
            timebase_samples=int(probe["timebase_samples"]),
            timebase_spread_limit_ms=float(probe["timebase_spread_limit_ms"]),
            step_timeout_s=StepTimeouts(
                startup=float(timeouts["startup"]),
                ready=float(timeouts["ready"]),
                flight=float(timeouts["flight"]),
            ),
            budget_wall_clock_s=float(probe["budget_wall_clock_s"]),
            output=document["output"],
            host_id=socket.gethostname(),
            clock_id="monotonic",
        )
        settings.check_values()
        return settings

    def check_values(self) -> None:
        """Reject settings that cannot describe a real run, naming the value."""
        if self.stereo.encoding != "rgb8":
            raise ConfigError(
                f"sensors.stereo.encoding must be rgb8 for the compat gate, "
                f"got {self.stereo.encoding}"
            )
        if not 0.0 < self.stereo.identical_channel_fraction_limit <= 1.0:
            raise ConfigError("sensors.stereo.identical_channel_fraction_limit must be in (0, 1]")
        if self.stereo.width <= 0 or self.stereo.height <= 0:
            raise ConfigError("sensors.stereo width and height must be positive")
        if self.stereo.baseline_m <= 0.0:
            raise ConfigError("sensors.stereo.baseline_m must be positive")
        if self.stereo.left == self.stereo.right:
            raise ConfigError("sensors.stereo needs two different device names")
        if not self.params:
            raise ConfigError("scenario.params must name at least the pinned parameter file")
        if not self.estimator_params:
            raise ConfigError(
                "scenario.estimator_params must name the EKF-active set; without it the "
                "estimator item has no configuration in which it means anything"
            )
        if self.hover_altitude_m <= 0.0:
            raise ConfigError("probe.hover_altitude_m must be positive")
        if self.settle_s <= 0.0:
            raise ConfigError("probe.settle_s must be positive: the scene needs time to settle")
        if self.at_rest_window_s <= 0.0:
            raise ConfigError("probe.at_rest_window_s must be positive")
        if not self.waypoints_local_ned:
            raise ConfigError("probe.waypoints_local_ned must name at least one waypoint")
        if self.timebase_samples < 2:
            raise ConfigError("probe.timebase_samples must be at least 2 to fit a line")
        if self.estimator_fault.magnitude_m <= 0.0 or self.estimator_fault.hold_s <= 0.0:
            raise ConfigError("probe.estimator_fault magnitude and hold must be positive")
        if self.budget_wall_clock_s <= 0.0:
            raise ConfigError("probe.budget_wall_clock_s must be positive")
        if self.endpoints.fdm_port <= 0 or self.endpoints.controller_port <= 0:
            raise ConfigError("platform.endpoints ports must be positive integers")

    def simulator_argv(self) -> tuple[str, ...]:
        """The exact Webots command line, headless at the configured simulation mode."""
        return (
            str(self.webots_binary),
            "--batch",
            f"--mode={self.webots_mode}",
            "--stdout",
            "--stderr",
            str(self.world),
        )

    def sitl_argv(self, extra_params: Sequence[Path] = ()) -> tuple[str, ...]:
        """The exact SITL command line for the pinned vehicle and ports.

        Parameter files are layered in order, so a second run that adds an
        EKF-active set keeps every pinned value it does not name. ``--wipe`` makes each
        run start from those files: SITL keeps its parameters in an ``eeprom.bin`` in
        the autopilot checkout and rewrites it as it runs, so without the wipe run B
        would begin from whatever run A had saved and neither receipt would describe
        the configuration it claims to.
        """
        argv = [
            str(self.sitl_binary),
            "--model",
            self.sim_model,
            "--sim-address",
            "127.0.0.1",
            "--sim-port-in",
            str(self.sim_port_in),
            "--sim-port-out",
            str(self.sim_port_out),
            "--wipe",
            "--home",
            self.sitl_home,
        ]
        for name in (*self.params, *extra_params):
            argv += ["--defaults", str(name)]
        return tuple(argv)

    def controller_environment(self) -> dict[str, str]:
        """The environment the Webots controller needs to import this package.

        Webots runs the controller under whatever interpreter it was configured
        with, which is not necessarily the one that installed this package. The
        source directory goes on PYTHONPATH so the shared framing and record
        helpers are imported from one place instead of being copied into the scene.
        """
        source_root = str(Path(__file__).resolve().parents[2])
        existing = os.environ.get("PYTHONPATH", "")
        merged = os.pathsep.join(part for part in (source_root, existing) if part)
        return {"PYTHONPATH": merged, "EMBODIED_SRC": source_root}


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prerequisite:
    """One thing the run cannot start without, and what was actually found."""

    name: str
    satisfied: bool
    detail: str


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def check_prerequisites(settings: PlatformSettings, output_dir: Path) -> tuple[Prerequisite, ...]:
    """Check every prerequisite before any process is started.

    Every item is reported, not only the first failure, because a blocked run is
    most useful when it says everything that is missing in one pass. Nothing is
    substituted for a missing prerequisite: there is no fake simulator behind the
    real command.
    """
    checks: list[Prerequisite] = []

    present = settings.webots_binary.is_file() and os.access(settings.webots_binary, os.X_OK)
    checks.append(
        Prerequisite(
            "webots_application",
            present,
            f"{settings.webots_binary}"
            + ("" if present else " is missing or not executable"),
        )
    )

    installed_version = None
    if settings.webots_version_file.is_file():
        installed_version = settings.webots_version_file.read_text(encoding="utf-8").strip()
    version_matches = installed_version == settings.webots_version
    checks.append(
        Prerequisite(
            "webots_version",
            version_matches,
            f"installed {installed_version!r}, configured {settings.webots_version!r}",
        )
    )

    checks.append(
        Prerequisite("scenario_world", settings.world.is_file(), f"{settings.world}")
    )
    for name in (*settings.params, *settings.estimator_params):
        checks.append(Prerequisite(f"parameter_file[{name.name}]", name.is_file(), f"{name}"))
    checks.append(
        Prerequisite(
            "calibration_declaration",
            settings.calibration_path.is_file(),
            f"{settings.calibration_path}",
        )
    )

    head = _git_head(settings.ardupilot_root)
    commit_matches = head == settings.ardupilot_commit
    checks.append(
        Prerequisite(
            "autopilot_commit",
            commit_matches,
            f"{settings.ardupilot_root} is at {head}, configured {settings.ardupilot_commit}",
        )
    )
    sitl_present = settings.sitl_binary.is_file() and os.access(settings.sitl_binary, os.X_OK)
    checks.append(
        Prerequisite(
            "sitl_binary",
            sitl_present,
            f"{settings.sitl_binary}" + ("" if sitl_present else " is not a built binary"),
        )
    )

    for role, port in (
        ("fdm_port", settings.sim_port_out),
        ("fdm_port_in", settings.sim_port_in),
        ("controller_port", settings.endpoints.controller_port),
    ):
        free = _port_is_free(port)
        checks.append(
            Prerequisite(
                f"port_{role}",
                free,
                f"udp/tcp {port} is " + ("free" if free else "already in use"),
            )
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write-check"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        writable = True
        detail = f"{output_dir} is writable"
    except OSError as error:
        writable = False
        detail = f"{output_dir} is not writable: {error}"
    checks.append(Prerequisite("output_directory", writable, detail))

    return tuple(checks)


def require_prerequisites(settings: PlatformSettings, output_dir: Path) -> tuple[Prerequisite, ...]:
    """Return the prerequisite table, raising when any item is unsatisfied."""
    checks = check_prerequisites(settings, output_dir)
    missing = [check for check in checks if not check.satisfied]
    if missing:
        raise PlatformUnavailable(
            "missing prerequisites: " + "; ".join(f"{check.name}: {check.detail}" for check in missing)
        )
    return checks


def load_calibration_declaration(path: Path) -> Any:
    """Read the declared rig geometry as a shared Calibration record."""
    from embodied.contracts.records import Calibration, from_dict

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PlatformUnavailable(f"cannot read the calibration declaration {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CommandError(f"{path} is not valid JSON: {error}") from error
    try:
        return from_dict(Calibration, document)
    except Exception as error:  # noqa: BLE001 - the record names its own problem
        raise CommandError(f"{path} is not a valid Calibration record: {error}") from error


# ---------------------------------------------------------------------------
# Value types the probe hands around
# ---------------------------------------------------------------------------

# Probe-originated messages carry no simulation time of their own: the command is
# issued against the wall clock, so the header says so rather than inventing one.
UNKNOWN_SIM_TIME_S = -1.0

# A servo channel that has never reported a value. Kept distinct from a real pulse
# width so "no output" and "output of zero" cannot be confused.
UNKNOWN_SERVO_RAW = -1


@dataclass(frozen=True)
class Injection:
    """One fault request. Whether it was applied is answered separately."""

    injection_id: str
    kind: str
    magnitude_m: float
    hold_s: float
    apply: bool

    @staticmethod
    def publish(index: int, kind: str, magnitude_m: float, hold_s: float) -> "Injection":
        return Injection(
            injection_id=f"inj-{index:03d}-{kind}",
            kind=kind,
            magnitude_m=magnitude_m,
            hold_s=hold_s,
            apply=True,
        )

    @staticmethod
    def clear(index: int, kind: str) -> "Injection":
        """Remove whatever was injected, before any conclusion is drawn from the run."""
        return Injection(
            injection_id=f"inj-{index:03d}-clear",
            kind=kind,
            magnitude_m=0.0,
            hold_s=0.0,
            apply=False,
        )


@dataclass(frozen=True)
class InjectionReceipt:
    """What was requested, and what the controller confirmed."""

    requested: Injection
    applied: bool
    acknowledged_state: dict[str, Any] | None
    reason: str | None


@dataclass(frozen=True)
class SensorRecord:
    """One decoded controller message: a pair, an inertial sample, a status or an ack."""

    kind: Kind
    sim_time_s: float
    sequence: int
    flags: int
    received_stamp: ClockStamp
    pair: PairPayload | None = None
    imu: ImuPayload | None = None
    status: dict[str, Any] | None = None
    fault_ack: dict[str, Any] | None = None


COPTER_MODES = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    9: "LAND",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    19: "AVOID_ADSB",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL",
    22: "FLOWHOLD",
    23: "FOLLOW",
    24: "ZIGZAG",
}


@dataclass(frozen=True)
class TelemetrySample:
    """The latest known value of each telemetry field, as observed in this stream.

    A field is None until a message carrying it has been seen. Carrying the last
    observed value forward is not a default: it is the most recent measurement from
    an arriving stream, and the sample never invents a value it has not seen.
    """

    received_stamp: ClockStamp
    messages_seen: int
    heartbeats: int
    mode_name: str | None
    custom_mode: int | None
    armed: bool | None
    system_status: int | None
    attitude_rpy: tuple[float, float, float] | None
    local_position_ned: tuple[float, float, float] | None
    velocity_ned: tuple[float, float, float] | None
    servo_outputs: tuple[int, ...] | None
    ekf_flags: int | None
    ekf_velocity_variance: float | None
    ekf_pos_horiz_variance: float | None
    statustexts: tuple[str, ...]
    boot_time_ms: int | None
    home_position: tuple[float, float, float] | None
    autopilot_version: dict[str, Any] | None

    @property
    def in_guided_mode(self) -> bool:
        return self.mode_name == "GUIDED"

    def document(self) -> dict[str, Any]:
        """A JSON-ready view for the evidence files."""
        return {
            "mode_name": self.mode_name,
            "custom_mode": self.custom_mode,
            "armed": self.armed,
            "system_status": self.system_status,
            "local_position_ned": list(self.local_position_ned)
            if self.local_position_ned is not None
            else None,
            "velocity_ned": list(self.velocity_ned) if self.velocity_ned is not None else None,
            "servo_outputs": list(self.servo_outputs) if self.servo_outputs is not None else None,
            "ekf_flags": self.ekf_flags,
            "ekf_velocity_variance": self.ekf_velocity_variance,
            "ekf_pos_horiz_variance": self.ekf_pos_horiz_variance,
            "statustexts": list(self.statustexts),
            "boot_time_ms": self.boot_time_ms,
            "received_at_monotonic_ns": self.received_stamp.monotonic_ns,
        }


def _triple(message: dict[str, Any], *fields: str) -> tuple[float, float, float] | None:
    """Three related numbers from one message, or None when the message omits any.

    A vector with a missing component is not a measurement: carrying it forward as
    ``(x, y, None)`` would put a hole in the middle of a record every later reader
    treats as a number.
    """
    values = tuple(message.get(field) for field in fields)
    if any(value is None for value in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def decode_telemetry(
    messages: Iterable[dict[str, Any]],
    *,
    stamp: ClockStamp,
    previous: TelemetrySample | None = None,
) -> TelemetrySample:
    """Fold a batch of raw MAVLink messages into one sample.

    A pure function over decoded messages, so the probe's reading of telemetry is
    tested against scripted records rather than a live autopilot.
    """
    latest: dict[str, Any] = {
        "heartbeats": 0,
        "mode_name": None,
        "custom_mode": None,
        "armed": None,
        "system_status": None,
        "attitude_rpy": None,
        "local_position_ned": None,
        "velocity_ned": None,
        "servo_outputs": None,
        "ekf_flags": None,
        "ekf_velocity_variance": None,
        "ekf_pos_horiz_variance": None,
        "statustexts": [],
        "boot_time_ms": None,
        "home_position": None,
        "autopilot_version": None,
        "messages_seen": 0,
    }
    if previous is not None:
        latest.update(
            {
                "heartbeats": previous.heartbeats,
                "mode_name": previous.mode_name,
                "custom_mode": previous.custom_mode,
                "armed": previous.armed,
                "system_status": previous.system_status,
                "attitude_rpy": previous.attitude_rpy,
                "local_position_ned": previous.local_position_ned,
                "velocity_ned": previous.velocity_ned,
                "servo_outputs": previous.servo_outputs,
                "ekf_flags": previous.ekf_flags,
                "ekf_velocity_variance": previous.ekf_velocity_variance,
                "ekf_pos_horiz_variance": previous.ekf_pos_horiz_variance,
                "statustexts": list(previous.statustexts),
                "boot_time_ms": previous.boot_time_ms,
                "home_position": previous.home_position,
                "autopilot_version": previous.autopilot_version,
            }
        )

    for message in messages:
        latest["messages_seen"] += 1
        kind = message.get("mavpackettype")
        # The autopilot's own boot clock travels on almost every message. Reading it
        # wherever it appears is what lets the timebase join sample a rate rather than
        # wait for AUTOPILOT_VERSION, which answers once when it is asked.
        if message.get("time_boot_ms") is not None:
            latest["boot_time_ms"] = message["time_boot_ms"]
        if kind == "HEARTBEAT":
            latest["heartbeats"] += 1
            latest["custom_mode"] = message.get("custom_mode")
            latest["mode_name"] = COPTER_MODES.get(message.get("custom_mode"))
            base_mode = message.get("base_mode")
            latest["armed"] = None if base_mode is None else bool(base_mode & 128)
            latest["system_status"] = message.get("system_status")
        elif kind == "ATTITUDE":
            latest["attitude_rpy"] = _triple(message, "roll", "pitch", "yaw")
        elif kind == "LOCAL_POSITION_NED":
            latest["local_position_ned"] = _triple(message, "x", "y", "z")
            latest["velocity_ned"] = _triple(message, "vx", "vy", "vz")
        elif kind == "SERVO_OUTPUT_RAW":
            latest["servo_outputs"] = tuple(
                UNKNOWN_SERVO_RAW if message.get(f"servo{index}_raw") is None
                else int(message[f"servo{index}_raw"])
                for index in range(1, 9)
            )
        elif kind == "EKF_STATUS_REPORT":
            latest["ekf_flags"] = message.get("flags")
            latest["ekf_velocity_variance"] = message.get("velocity_variance")
            latest["ekf_pos_horiz_variance"] = message.get("pos_horiz_variance")
        elif kind == "STATUSTEXT":
            text = message.get("text")
            if text:
                latest["statustexts"] = (latest["statustexts"] + [str(text)])[-50:]
        elif kind == "HOME_POSITION":
            latest["home_position"] = (
                message.get("latitude"),
                message.get("longitude"),
                message.get("altitude"),
            )
        elif kind == "AUTOPILOT_VERSION":
            latest["autopilot_version"] = {
                key: message.get(key)
                for key in (
                    "capabilities",
                    "flight_sw_version",
                    "middleware_sw_version",
                    "os_sw_version",
                    "board_version",
                    "flight_custom_version",
                    "vendor_id",
                    "product_id",
                    "uid",
                )
            }

    return TelemetrySample(
        received_stamp=stamp,
        messages_seen=latest["messages_seen"],
        heartbeats=latest["heartbeats"],
        mode_name=latest["mode_name"],
        custom_mode=latest["custom_mode"],
        armed=latest["armed"],
        system_status=latest["system_status"],
        attitude_rpy=latest["attitude_rpy"],
        local_position_ned=latest["local_position_ned"],
        velocity_ned=latest["velocity_ned"],
        servo_outputs=latest["servo_outputs"],
        ekf_flags=latest["ekf_flags"],
        ekf_velocity_variance=latest["ekf_velocity_variance"],
        ekf_pos_horiz_variance=latest["ekf_pos_horiz_variance"],
        statustexts=tuple(latest["statustexts"]),
        boot_time_ms=latest["boot_time_ms"],
        home_position=latest["home_position"],
        autopilot_version=latest["autopilot_version"],
    )


@dataclass(frozen=True)
class TimebaseJoin:
    """The measured relation between two clocks, and whether it is tight enough to use."""

    samples: int
    offset_s: float | None
    drift_ppm: float | None
    spread_ms: float | None
    joined: bool
    reason: str | None


def fit_timebase(samples: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares fit of host time against device boot time.

    Returns (offset_s, drift_ppm, residual spread in ms) for
    ``host = offset + scale * boot``. Two clocks are compared by measurement here,
    never by assuming that two numbers are comparable because both are seconds.
    """
    if len(samples) < 2:
        raise ProbeFailure("joining two clocks needs at least two paired samples")
    count = len(samples)
    mean_boot = sum(boot for _, boot in samples) / count
    mean_host = sum(host for host, _ in samples) / count
    variance = sum((boot - mean_boot) ** 2 for _, boot in samples)
    if variance == 0.0:
        raise ProbeFailure("the device clock did not advance, so the join is undetermined")
    scale = sum((boot - mean_boot) * (host - mean_host) for host, boot in samples) / variance
    offset = mean_host - scale * mean_boot
    residuals_ms = [1000.0 * (host - (offset + scale * boot)) for host, boot in samples]
    return offset, (scale - 1.0) * 1e6, max(residuals_ms) - min(residuals_ms)


def join_timebase(samples: Sequence[tuple[float, float]], spread_limit_ms: float) -> TimebaseJoin:
    """Join two clocks, or report that they cannot be joined inside the limit."""
    try:
        offset, drift_ppm, spread_ms = fit_timebase(samples)
    except ProbeFailure as error:
        return TimebaseJoin(
            samples=len(samples),
            offset_s=None,
            drift_ppm=None,
            spread_ms=None,
            joined=False,
            reason=str(error),
        )
    joined = spread_ms <= spread_limit_ms
    reason = None
    if not joined:
        reason = (
            f"residual spread {spread_ms:.1f} ms exceeds the configured "
            f"{spread_limit_ms:.1f} ms"
        )
    return TimebaseJoin(
        samples=len(samples),
        offset_s=offset,
        drift_ppm=drift_ppm,
        spread_ms=spread_ms,
        joined=joined,
        reason=reason,
    )


# The autopilot's own view of itself, from MAV_STATE: it reports BOOT while it
# initialises, STANDBY when it is landed and ready for commands, ACTIVE while flying,
# and CRITICAL after a failsafe. Only the middle two mean "this vehicle can be given
# a target", which is what readiness has to establish before any item measures it.
MAV_STATE_STANDBY = 4
MAV_STATE_ACTIVE = 5
READY_SYSTEM_STATUSES = (MAV_STATE_STANDBY, MAV_STATE_ACTIVE)


# The MAVLink identifiers this adapter uses, named so the code reads as intent
# rather than as numbers. MAV_CMD_DO_SET_MODE is written inline where it is used
# because it appears once.
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_SET_MESSAGE_INTERVAL = 511
# MAV_CMD_REQUEST_MESSAGE asks for one message now. AUTOPILOT_VERSION is a
# request-and-answer message rather than a streamed one, so it is asked for once.
MAV_CMD_REQUEST_MESSAGE = 512
MAV_FRAME_LOCAL_NED = 1
MAV_FRAME_BODY_NED = 8

# Message identifiers this adapter asks the autopilot for.
MSG_ID_ATTITUDE = 30
MSG_ID_LOCAL_POSITION_NED = 32
MSG_ID_SERVO_OUTPUT_RAW = 36
MSG_ID_AUTOPILOT_VERSION = 148
MSG_ID_EKF_STATUS_REPORT = 193

# The only message types this program may send. Motion travels as a guided
# setpoint; everything else is a procedure such as a mode request or an interval
# request. There is no motor, throttle, RC or attitude command in this set.
ALLOWED_OUTBOUND_TYPES = frozenset(
    {
        "SET_POSITION_TARGET_LOCAL_NED",
        "COMMAND_LONG",
        "SET_MESSAGE_INTERVAL",
        "HEARTBEAT",
    }
)


# ---------------------------------------------------------------------------
# Real seams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChildProcess:
    """One process this adapter started and is therefore responsible for stopping."""

    name: str
    argv: tuple[str, ...]
    pid: int
    log_path: Path


class SubprocessRunner:
    """Spawns the simulator and the autopilot, and never leaves them behind.

    Output goes to a log file per child, so a failed step can quote what the child
    actually said instead of guessing.
    """

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen] = {}

    def spawn(
        self,
        name: str,
        argv: Sequence[str],
        *,
        log_path: Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> ChildProcess:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        merged = dict(os.environ)
        if env:
            merged.update(env)
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                list(argv),
                cwd=str(cwd) if cwd is not None else None,
                env=merged,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        self._processes[name] = process
        return ChildProcess(name=name, argv=tuple(argv), pid=process.pid, log_path=log_path)

    def poll(self, child: ChildProcess) -> int | None:
        process = self._processes.get(child.name)
        return None if process is None else process.poll()

    def terminate(self, child: ChildProcess, timeout_s: float = 10.0) -> int | None:
        process = self._processes.get(child.name)
        if process is None:
            return None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout_s)
        return process.returncode

    def tail(self, child: ChildProcess, lines: int = 200) -> list[str]:
        """The last lines of a child's output, bounded so a runaway log cannot fill memory."""
        try:
            with child.log_path.open("rb") as handle:
                data = handle.read(1 << 22)
        except OSError:
            return []
        text = data.decode("utf-8", errors="replace").splitlines()
        return text[-lines:]


class PymavlinkSession:
    """The real MAVLink session, wrapping pymavlink.

    ``_send`` is the single choke point for outbound traffic: a message type that
    is not in :data:`ALLOWED_OUTBOUND_TYPES` is refused here, so no motor, RC or
    attitude command can leave this program by accident.
    """

    def __init__(self, *, source_system: int = 250, source_component: int = 190) -> None:
        self._source_system = source_system
        self._source_component = source_component
        self._connection = None
        self._mavutil = None
        self.target_system = 0
        self.target_component = 0

    def connect(self, endpoint: str, timeout_s: float) -> dict[str, Any]:
        try:
            from pymavlink import mavutil
        except ImportError as error:  # pragma: no cover - pinned dependency
            raise ProbeFailure(f"pymavlink is required for the autopilot session: {error}") from error
        self._mavutil = mavutil
        self._connection = mavutil.mavlink_connection(
            endpoint, source_system=self._source_system, source_component=self._source_component
        )
        heartbeat = self._connection.wait_heartbeat(timeout=timeout_s)
        if heartbeat is None:
            raise ProbeFailure(f"no MAVLink heartbeat on {endpoint} within {timeout_s:.0f}s")
        self.target_system = self._connection.target_system
        self.target_component = self._connection.target_component
        return {
            "endpoint": endpoint,
            "target_system": self.target_system,
            "target_component": self.target_component,
            "heartbeat": heartbeat.to_dict(),
        }

    def _send(self, message: Any) -> None:
        message_type = message.get_type()
        if message_type not in ALLOWED_OUTBOUND_TYPES:
            raise ProbeFailure(
                f"refusing to send {message_type}: this project commands motion only "
                "through supported MAVLink guided setpoints"
            )
        self._connection.mav.send(message)

    def request_message_interval(self, message_id: int, hz: float) -> None:
        self._send(
            self._connection.mav.command_long_encode(
                self.target_system,
                self.target_component,
                MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                int(1e6 / hz) if hz > 0 else -1,
                0,
                0,
                0,
                0,
                0,
            )
        )

    def request_message(self, message_id: int) -> None:
        """Ask for one message now, for messages the autopilot only answers on request."""
        self._send(
            self._connection.mav.command_long_encode(
                self.target_system,
                self.target_component,
                MAV_CMD_REQUEST_MESSAGE,
                0,
                message_id,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        )

    def set_mode(self, mode_name: str) -> None:
        """Request one copter mode by name, by command rather than by assumed numbering."""
        mapping = {name: mode for mode, name in COPTER_MODES.items()}
        if mode_name not in mapping:
            raise ProbeFailure(f"{mode_name} is not a mode this project requests")
        self._send(
            self._connection.mav.command_long_encode(
                self.target_system,
                self.target_component,
                176,  # MAV_CMD_DO_SET_MODE
                0,
                1,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                mapping[mode_name],
                0,
                0,
                0,
                0,
                0,
            )
        )

    def arm(self) -> None:
        self._send(
            self._connection.mav.command_long_encode(
                self.target_system,
                self.target_component,
                MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        )

    def takeoff(self, altitude_m: float) -> None:
        self._send(
            self._connection.mav.command_long_encode(
                self.target_system,
                self.target_component,
                MAV_CMD_NAV_TAKEOFF,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                altitude_m,
            )
        )

    def send_setpoint(self, setpoint: MotionSetpoint) -> None:
        """Publish one guided local-NED target, using exactly the fields the mask selects."""
        target = setpoint.target
        position = target.position_ned or (0.0, 0.0, 0.0)
        velocity = target.velocity_ned or (0.0, 0.0, 0.0)
        acceleration = target.acceleration_ned or (0.0, 0.0, 0.0)
        frame = {
            Frame.ODOM: MAV_FRAME_LOCAL_NED,
            Frame.BODY: MAV_FRAME_BODY_NED,
        }.get(setpoint.frame)
        if frame is None:
            raise ProbeFailure(f"{setpoint.frame} is not a frame this adapter publishes in")
        self._send(
            self._connection.mav.set_position_target_local_ned_encode(
                0,
                self.target_system,
                self.target_component,
                frame,
                setpoint.type_mask,
                position[0],
                position[1],
                position[2],
                velocity[0],
                velocity[1],
                velocity[2],
                acceleration[0],
                acceleration[1],
                acceleration[2],
                target.yaw_rad or 0.0,
                target.yaw_rate_rad_s or 0.0,
            )
        )

    def drain(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        while len(messages) < 2000:
            message = self._connection.recv_match(blocking=False)
            if message is None:
                break
            document = message.to_dict()
            document.setdefault("mavpackettype", message.get_type())
            messages.append(document)
        return messages

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class TcpSensorGateway:
    """The sensor stream client: one loopback connection carries records both ways."""

    def __init__(self, stamp: Callable[[], ClockStamp]) -> None:
        # The caller supplies the stamp rather than this class inventing a host and
        # clock identity of its own: a receipt time is only comparable with the
        # capture time it is subtracted from when both name the same clock domain.
        self._stamp = stamp
        self._socket: socket.socket | None = None
        self._reader = FrameReader()
        self._sequence = 0
        self._last_sim_time_s = UNKNOWN_SIM_TIME_S

    def open(self, host: str, port: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                self._socket = socket.create_connection((host, port), timeout=timeout_s)
                break
            except OSError as error:
                last_error = error
                time.sleep(0.1)
        else:
            raise ProbeFailure(f"no controller connection on {host}:{port}: {last_error}")
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.setblocking(False)

    def read_record(self, timeout_s: float) -> SensorRecord | None:
        """The next decoded record, or None when the stream stays silent for the timeout."""
        deadline = time.monotonic() + timeout_s
        while True:
            record = self._decode_one()
            if record is not None:
                return record
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            readable, _, _ = select.select([self._socket], [], [], min(remaining, 0.25))
            if not readable:
                continue
            try:
                chunk = self._socket.recv(1 << 16)
            except BlockingIOError:
                continue
            except OSError as error:
                raise ProbeFailure(f"the controller stream failed: {error}") from error
            if not chunk:
                raise ProbeFailure("the controller closed the sensor stream")
            self._reader.feed(chunk)

    def _decode_one(self) -> SensorRecord | None:
        message = self._reader.next_message()
        if message is None:
            return None
        received = self._stamp()
        record = SensorRecord(
            kind=message.kind,
            sim_time_s=message.sim_time_s,
            sequence=message.sequence,
            flags=message.flags,
            received_stamp=received,
        )
        if message.sim_time_s >= 0.0:
            self._last_sim_time_s = message.sim_time_s
        if message.kind is Kind.PAIR:
            record = replace(record, pair=decode_pair_payload(message.payload))
        elif message.kind is Kind.IMU:
            record = replace(record, imu=decode_imu_payload(message.payload))
        elif message.kind is Kind.STATUS:
            record = replace(record, status=decode_status_payload(message.payload))
        elif message.kind is Kind.FAULT_ACK:
            record = replace(record, fault_ack=decode_fault_payload(message.payload))
        return record

    def send_fault(self, injection: Injection) -> None:
        if self._socket is None:
            raise ProbeFailure("the controller stream is not open")
        self._sequence += 1
        framed = pack_message(
            Kind.FAULT,
            sim_time_s=self._last_sim_time_s,
            sequence=self._sequence,
            payload=encode_fault_payload(injection),
        )
        try:
            self._socket.sendall(framed)
        except OSError as error:
            raise ProbeFailure(f"cannot send the injection to the controller: {error}") from error

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


# ---------------------------------------------------------------------------
# Frame measurement
# ---------------------------------------------------------------------------


def measure_frame_quality(
    rgb: bytes, width: int, height: int, *, identical_channel_fraction_limit: float
) -> FrameQuality:
    """Measure whether a frame actually carries colour, and how much of it is saturated.

    The measurement is per pixel: a frame whose three channels agree everywhere is
    one channel copied three times, which is exactly what an averaging bridge
    produces. The declared limit decides how much agreement still counts as colour.
    """
    try:
        import numpy
    except ImportError as error:  # pragma: no cover - numpy is a pinned dependency
        raise ProbeFailure(f"numpy is required to measure a frame: {error}") from error
    expected = width * height * 3
    if len(rgb) != expected:
        raise FramingError(f"frame holds {len(rgb)} bytes, expected {expected}")
    image = numpy.frombuffer(rgb, dtype=numpy.uint8).reshape(height, width, 3)
    red = image[:, :, 0].astype(numpy.float64)
    green = image[:, :, 1].astype(numpy.float64)
    blue = image[:, :, 2].astype(numpy.float64)
    identical = float(numpy.mean((red == green) & (green == blue)))
    saturated = float(numpy.mean(image.max(axis=2) >= 255))
    return FrameQuality(
        is_colour=identical < identical_channel_fraction_limit,
        channel_identical_fraction=identical,
        channel_means=(float(red.mean()), float(green.mean()), float(blue.mean())),
        saturated_pixel_fraction=saturated,
    )


def dominant_channel(means: Sequence[float]) -> str:
    """Which of red, green or blue is largest in a measured mean."""
    names = ("red", "green", "blue")
    return names[int(max(range(3), key=lambda index: means[index]))]


def find_colour_witness(
    rgb: bytes, width: int, height: int, expected_rgb: Sequence[float]
) -> dict[str, Any]:
    """Measure the scene's known-colour object and report which channel leads.

    The expected dominant channel is known before the run. If the capture pipeline
    swapped or dropped channels, the measured dominant channel is a different one,
    which is how a channel-order mistake becomes visible instead of plausible.
    """
    try:
        import numpy
    except ImportError as error:  # pragma: no cover - numpy is a pinned dependency
        raise ProbeFailure(f"numpy is required to find the colour witness: {error}") from error
    if len(expected_rgb) != 3:
        raise CommandError("the scene's witness colour must be three channel values")
    expected_dominant = dominant_channel(expected_rgb)
    image = numpy.frombuffer(rgb, dtype=numpy.uint8).reshape(height, width, 3).astype(numpy.int16)
    order = {"red": 0, "green": 1, "blue": 2}
    leading = order[expected_dominant]
    others = [index for index in range(3) if index != leading]
    margin = 40  # a saturated marker is far from neutral; this ignores grey scene pixels
    mask = (image[:, :, leading] - numpy.maximum(image[:, :, others[0]], image[:, :, others[1]])) > margin
    pixels = int(mask.sum())
    document: dict[str, Any] = {
        "expected_dominant_channel": expected_dominant,
        "expected_rgb": [float(value) for value in expected_rgb],
        "matched_pixels": pixels,
        "image_pixels": width * height,
    }
    if pixels == 0:
        document.update(
            {
                "mean_rgb": None,
                "centroid_px": None,
                "observed_dominant_channel": None,
                "matches": False,
            }
        )
        return document
    rows, columns = numpy.nonzero(mask)
    selected = image[mask].astype(numpy.float64)
    means = (float(selected[:, 0].mean()), float(selected[:, 1].mean()), float(selected[:, 2].mean()))
    document.update(
        {
            "mean_rgb": list(means),
            "centroid_px": [float(columns.mean()), float(rows.mean())],
            "observed_dominant_channel": dominant_channel(means),
            "matches": dominant_channel(means) == expected_dominant,
        }
    )
    return document


# ---------------------------------------------------------------------------
# Evidence files
# ---------------------------------------------------------------------------


class EvidenceWriter:
    """Writes a run's raw material before anything is evaluated.

    Artifact paths are relative to the probe's output directory so the receipt can
    name and hash them from one place.
    """

    def __init__(self, output_dir: Path, label: str) -> None:
        self.output_dir = output_dir
        self.label = label
        self.directory = output_dir / label
        self.directory.mkdir(parents=True, exist_ok=True)
        self.artifacts: list[str] = []

    def path(self, name: str) -> Path:
        target = self.directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        relative = str(target.relative_to(self.output_dir))
        if relative not in self.artifacts:
            self.artifacts.append(relative)
        return target

    def write_json(self, name: str, document: Any) -> str:
        target = self.path(name)
        target.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
        return str(target.relative_to(self.output_dir))

    def write_bytes(self, name: str, data: bytes) -> str:
        target = self.path(name)
        target.write_bytes(data)
        return str(target.relative_to(self.output_dir))

    def append_jsonl(self, name: str, document: Any) -> None:
        with self.path(name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(document, default=str) + "\n")

    def tail_lines(self, path: Path, lines: int = 200) -> list[str]:
        try:
            with path.open("rb") as handle:
                data = handle.read(1 << 22)
        except OSError:
            return []
        return data.decode("utf-8", errors="replace").splitlines()[-lines:]


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalNedTarget:
    """One requested motion target in the local NED frame."""

    position_ned: tuple[float, float, float] | None
    velocity_ned: tuple[float, float, float] | None
    yaw_rad: float | None
    deadline_s: float
    certificate_ref: str | None


@dataclass(frozen=True)
class SetpointPublication:
    """One published setpoint, with the honest statement of what is known about it.

    Publication is measurable locally. Adoption by the autopilot is not: it shows up
    later as telemetry, so this record never claims it.
    """

    setpoint: MotionSetpoint
    published_stamp: ClockStamp
    adoption: str = "publication recorded; adoption is judged from telemetry, not here"


@dataclass(frozen=True)
class StartupEvidence:
    webots: ChildProcess
    sitl: ChildProcess
    ports: dict[str, int]
    argv: dict[str, tuple[str, ...]]
    webots_version: str


@dataclass(frozen=True)
class ReadinessEvidence:
    controller_status: dict[str, Any]
    telemetry: TelemetrySample
    waited_s: float
    pair_seen: bool
    imu_seen: bool


@dataclass(frozen=True)
class FlightStateEvidence:
    commanded_mode: str
    mode_reached: bool
    armed: bool
    takeoff_commanded_m: float
    altitude_m: float | None
    statustexts: tuple[str, ...]
    refused: bool


@dataclass(frozen=True)
class ShutdownEvidence:
    exits: dict[str, int | None]
    log_tails: dict[str, list[str]]


@dataclass(frozen=True)
class ControlEvent:
    """One observed change in who is flying the aircraft, read from telemetry.

    The autopilot leaves Guided flight on its own when something fails; this adapter
    records that and stops publishing rather than continuing to send targets into a
    mode that is not following them. The change is an observation, not a decision.
    """

    at: ClockStamp
    from_mode: str | None
    to_mode: str | None
    armed_before: bool | None
    armed_after: bool | None
    system_status: int | None
    statustexts: tuple[str, ...]
    guidance_held: bool

    def document(self) -> dict[str, Any]:
        return {
            "at_monotonic_ns": self.at.monotonic_ns,
            "from_mode": self.from_mode,
            "to_mode": self.to_mode,
            "armed_before": self.armed_before,
            "armed_after": self.armed_after,
            "system_status": self.system_status,
            "statustexts": list(self.statustexts),
            "guidance_held": self.guidance_held,
        }


def mask_for_target(target: MotionTarget) -> int:
    """The type mask that matches a target: a field the mask ignores must be absent.

    A set bit means "ignore this field", so each field the target omits gets its bit
    set. Force is always ignored: this project never commands a force target.
    """
    mask = TYPE_MASK_FORCE
    if target.position_ned is None:
        mask |= TYPE_MASK_POSITION
    if target.velocity_ned is None:
        mask |= TYPE_MASK_VELOCITY
    if target.acceleration_ned is None:
        mask |= TYPE_MASK_ACCELERATION
    if target.yaw_rad is None:
        mask |= TYPE_MASK_YAW
    if target.yaw_rate_rad_s is None:
        mask |= TYPE_MASK_YAW_RATE
    return mask


# How long an arming and mode request is given before the autopilot's answer is read.
# The autopilot needs a moment, and it reports its pre-arm checks continuously, so the
# grace is measured from the attempt rather than from the first refusal text. After it,
# a vehicle that has not reached armed Guided flight has answered the request: either
# its mode is not Guided or its pre-arm checks refuse, and polling it for the whole
# flight timeout produces no more evidence than that first answer did.
CONTROL_GRANT_GRACE_S = 3.0

# How long to let an arming request be answered before reading the result. The
# autopilot needs a moment, and the sensor stream is read throughout the wait.
ARM_SETTLE_S = 1.0

def autopilot_is_ready(sample: TelemetrySample) -> bool:
    """Whether the autopilot has finished booting and is streaming its own clock.

    A heartbeat arrives while the vehicle is still initialising, so status and a
    streamed message carrying the boot clock are what show it is up. The boot clock is
    the cheapest proof that its telemetry loop is running, which is what every later
    item reads.
    """
    return (
        sample.heartbeats > 0
        and sample.system_status in READY_SYSTEM_STATUSES
        and sample.boot_time_ms is not None
    )


def autopilot_state_summary(sample: TelemetrySample) -> str:
    """Why the autopilot is not ready yet, in the words the receipt can carry."""
    return (
        f"system_status={sample.system_status} "
        f"(ready states {list(READY_SYSTEM_STATUSES)}), "
        f"mode={sample.mode_name!r}, boot_time_ms={sample.boot_time_ms}, "
        f"heartbeats={sample.heartbeats}"
    )


class WebotsArduPilot:
    """One Webots/ArduPilot candidate vehicle, as a single object with a small surface.

    It hides argv construction, readiness detection, MAVLink decoding into shared
    records, the sensor framing, the ENU-to-NED conversion, injection and cleanup.
    A caller states what it wants to know; the class owns everything a caller should
    not have to know.
    """

    # The MAVLink log is evidence, not an archive: past this many messages it stops
    # growing so a long run cannot fill the disk with telemetry.
    MAX_MAVLINK_LOG_LINES = 20000

    def __init__(
        self,
        settings: PlatformSettings,
        *,
        runner: SubprocessRunner,
        session: PymavlinkSession,
        gateway: TcpSensorGateway,
        evidence: EvidenceWriter,
        label: str,
        extra_params: Sequence[Path] = (),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.label = label
        self.evidence = evidence
        self._runner = runner
        self._session = session
        self._gateway = gateway
        self._extra_params = tuple(extra_params)
        self.controller_status: dict[str, Any] = {}
        self._monotonic_ns = monotonic_ns
        self._monotonic = monotonic
        self._sleep = sleep
        self._children: dict[str, ChildProcess] = {}
        self._telemetry: TelemetrySample | None = None
        self._publications: list[SetpointPublication] = []
        self._pending_records: list[SensorRecord] = []
        self._statustexts: list[str] = []
        self._sequence = 0
        self._mavlink_lines = 0
        # Mode changes and refused publications are evidence: what the aircraft did
        # belongs in the run's record even when it is not what was asked for.
        self.control_events: list[ControlEvent] = []
        self.refusals: list[dict[str, Any]] = []
        self.navigation_epoch = "nav-1"

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> StartupEvidence:
        """Start Webots and SITL, then open the sensor and MAVLink channels."""
        simulator = self._runner.spawn(
            "webots",
            self.settings.simulator_argv(),
            log_path=self.evidence.path("webots.log"),
            env=self.settings.controller_environment(),
        )
        sitl = self._runner.spawn(
            "sitl",
            self.settings.sitl_argv(self._extra_params),
            log_path=self.evidence.path("sitl.log"),
            cwd=self.settings.ardupilot_root,
        )
        self._children = {"webots": simulator, "sitl": sitl}
        self._gateway.open("127.0.0.1", self.settings.endpoints.controller_port, 30.0)
        connection = self._session.connect(self.settings.mavlink_endpoint, 30.0)
        # AUTOPILOT_VERSION answers once, on request, and it is the only statement of
        # which firmware is being exercised.
        self._session.request_message(MSG_ID_AUTOPILOT_VERSION)
        self.evidence.write_json(
            "connection.json",
            {"autopilot": connection, "sitl_argv": list(sitl.argv), "webots_argv": list(simulator.argv)},
        )
        return StartupEvidence(
            webots=simulator,
            sitl=sitl,
            ports={
                "fdm_port": self.settings.sim_port_out,
                "fdm_port_in": self.settings.sim_port_in,
                "controller_port": self.settings.endpoints.controller_port,
                "mavlink": self.settings.mavlink_endpoint,
            },
            argv={"webots": simulator.argv, "sitl": sitl.argv},
            webots_version=self.settings.webots_version,
        )

    def wait_ready(self, timeout_s: float) -> ReadinessEvidence:
        """Wait until the simulator is streaming and the autopilot can be commanded.

        Readiness is four separate facts: the controller declared its status, the
        sensor stream produced a pair and an inertial sample, and the autopilot has
        finished booting. ArduPilot answers a heartbeat seconds before it can be given
        a target — it is still in MAV_STATE_BOOT, its telemetry loop is not streaming
        and it cannot arm — and every later item reads that telemetry, so waiting for
        the vehicle rather than for a pulse is what keeps those items meaningful.
        """
        started = self._monotonic()
        deadline = started + timeout_s
        pair_seen = False
        imu_seen = False
        status_seen = False
        # Sensor records that arrive before readiness are parked, not re-read: the
        # caller receives them in order once the stream is ready, so nothing is
        # measured twice and nothing is dropped.
        parked: list[SensorRecord] = []
        while self._monotonic() < deadline:
            self._die_if_child_exited()
            record = self.sensor_record(0.25)
            if record is not None:
                if record.kind is Kind.STATUS:
                    status_seen = True
                elif record.kind is Kind.PAIR:
                    pair_seen = True
                    parked.append(record)
                elif record.kind is Kind.IMU:
                    imu_seen = True
                    parked.append(record)
            sample = self.telemetry()
            if (
                status_seen
                and pair_seen
                and imu_seen
                and autopilot_is_ready(sample)
            ):
                self._pending_records = parked + self._pending_records
                return ReadinessEvidence(
                    controller_status=self.controller_status,
                    telemetry=sample,
                    waited_s=self._monotonic() - started,
                    pair_seen=pair_seen,
                    imu_seen=imu_seen,
                )
            self._sleep(0.05)
        raise ProbeFailure(
            f"the simulator did not become ready within {timeout_s:.0f}s: "
            f"status={'yes' if status_seen else 'no'}, "
            f"pair={'yes' if pair_seen else 'no'}, imu={'yes' if imu_seen else 'no'}, "
            f"autopilot={autopilot_state_summary(self.telemetry())}"
        )

    def _die_if_child_exited(self) -> None:
        """Stop with the child's own output if either process has already gone."""
        for name, child in self._children.items():
            code = self._runner.poll(child)
            if code is not None:
                tail = "\n".join(self._runner.tail(child, lines=40))
                raise ProbeFailure(f"{name} exited with status {code}:\n{tail}")

    def stop(self) -> ShutdownEvidence:
        """Close everything this adapter opened, whatever happened before.

        Called from a finally block by the probe, so a failed step still leaves the
        two children terminated and their exit codes and last output recorded.
        """
        self._gateway.close()
        try:
            self._session.close()
        except Exception:  # noqa: BLE001 - cleanup must not raise over the real failure
            pass
        exits: dict[str, int | None] = {}
        tails: dict[str, list[str]] = {}
        for name, child in self._children.items():
            exits[name] = self._runner.terminate(child)
            tails[name] = self._runner.tail(child, lines=200)
            self.evidence.write_json(f"{name}-tail.json", {"lines": tails[name], "exit": exits[name]})
        self.evidence.write_json("shutdown.json", {"exits": exits})
        self._children = {}
        return ShutdownEvidence(exits=exits, log_tails=tails)

    # -- reading -----------------------------------------------------------

    def telemetry(self) -> TelemetrySample:
        """Drain the MAVLink stream, record it, and fold it into the latest known state."""
        messages = self._session.drain()
        for document in messages:
            self._record_mavlink(document)
        sample = decode_telemetry(messages, stamp=self._stamp(), previous=self._telemetry)
        previous = self._telemetry
        self._telemetry = sample
        if sample.statustexts:
            self._statustexts = list(sample.statustexts)
        if previous is not None and (
            sample.mode_name != previous.mode_name or sample.armed != previous.armed
        ):
            self.control_events.append(
                ControlEvent(
                    at=sample.received_stamp,
                    from_mode=previous.mode_name,
                    to_mode=sample.mode_name,
                    armed_before=previous.armed,
                    armed_after=sample.armed,
                    system_status=sample.system_status,
                    statustexts=tuple(sample.statustexts[-5:]),
                    guidance_held=self.guidance_held,
                )
            )
        return sample

    @property
    def guidance_held(self) -> bool:
        """Whether the last telemetry showed the autopilot flying under Guided control.

        False covers every other state, including "no telemetry yet", which is why
        publishing is refused rather than attempted while it is False.
        """
        sample = self._telemetry
        return bool(sample is not None and sample.in_guided_mode and sample.armed)

    @property
    def latest_telemetry(self) -> TelemetrySample | None:
        """The most recent sample, without draining the stream again."""
        return self._telemetry

    def sensor_record(self, timeout_s: float) -> SensorRecord | None:
        """The next sensor record, or None when the stream stays silent for the timeout."""
        if self._pending_records:
            record = self._pending_records.pop(0)
        else:
            record = self._gateway.read_record(timeout_s)
        if record is None:
            return None
        if record.kind is Kind.STATUS and record.status is not None:
            self.controller_status = record.status
        return record


    # -- command -----------------------------------------------------------


    def arm_and_guided(
        self, timeout_s: float, *, drain: Callable[[], None] | None = None
    ) -> FlightStateEvidence:
        """Request Guided mode, arm, and take off to the configured altitude.

        A refusal is evidence: the mode and the arming state are read back from
        telemetry, and any STATUSTEXT the autopilot gave is retained with them. A
        vehicle that refuses to arm will keep refusing for the whole window, so the
        first refusal after the attempt is what ends the wait; polling a check that has
        already failed for two minutes produces no further evidence.

        ``drain`` is the caller's loop over the sensor stream. This wait is measured in
        seconds, and the cameras and inertial devices keep producing throughout it, so
        without it the controller's queue fills and frames are dropped before anything
        reads them.
        """
        for message_id, hz in (
            (MSG_ID_ATTITUDE, 20.0),
            (MSG_ID_LOCAL_POSITION_NED, 10.0),
            (MSG_ID_SERVO_OUTPUT_RAW, 5.0),
            (MSG_ID_EKF_STATUS_REPORT, 2.0),
        ):
            self._session.request_message_interval(message_id, hz)
        attempt_at = self._monotonic()
        self._session.set_mode("GUIDED")
        self._session.arm()
        # Give the autopilot a moment to answer the arming request, reading the sensor
        # stream while it does: this wait is a second of frames the controller would
        # otherwise queue and then drop.
        settle_until = attempt_at + ARM_SETTLE_S
        while self._monotonic() < settle_until:
            if drain is not None:
                drain()
            self._sleep(0.1)
        sample = self.telemetry()
        if sample.armed:
            self._session.takeoff(self.settings.hover_altitude_m)
        deadline = self._monotonic() + timeout_s
        altitude = None
        while self._monotonic() < deadline:
            sample = self.telemetry()
            position = sample.local_position_ned
            if position is not None:
                altitude = -position[2]
                if (
                    sample.in_guided_mode
                    and sample.armed
                    and altitude >= 0.5 * self.settings.hover_altitude_m
                ):
                    break
            if (
                self._monotonic() - attempt_at >= CONTROL_GRANT_GRACE_S
                and not (sample.in_guided_mode and sample.armed)
            ):
                break
            if drain is not None:
                drain()
            self._sleep(0.2)
        evidence = FlightStateEvidence(
            commanded_mode="GUIDED",
            mode_reached=sample.in_guided_mode,
            armed=bool(sample.armed),
            takeoff_commanded_m=self.settings.hover_altitude_m,
            altitude_m=altitude,
            statustexts=tuple(self._statustexts),
            refused=not (sample.in_guided_mode and sample.armed),
        )
        self.evidence.write_json(
            "flight-state.json",
            {
                "commanded_mode": evidence.commanded_mode,
                "mode_reached": evidence.mode_reached,
                "armed": evidence.armed,
                "takeoff_commanded_m": evidence.takeoff_commanded_m,
                "altitude_m": evidence.altitude_m,
                "statustexts": list(evidence.statustexts),
                "refused": evidence.refused,
            },
        )
        return evidence

    def send_local_ned(self, target: LocalNedTarget) -> SetpointPublication | None:
        """Publish one guided local-NED setpoint, and record that it was published.

        Returns None when the last telemetry did not show the autopilot in Guided
        flight. A setpoint sent in any other mode is a target nothing is following,
        so it is not sent: the refusal is recorded with the mode and status that
        caused it, and the caller reports what the aircraft was actually doing
        instead of what it was told to do.
        """
        if not self.guidance_held:
            sample = self._telemetry
            refusal = {
                "at_monotonic_ns": int(self._monotonic_ns()),
                "requested_target_ned": list(target.position_ned)
                if target.position_ned is not None
                else None,
                "observed_mode": None if sample is None else sample.mode_name,
                "observed_armed": None if sample is None else sample.armed,
                "reason": (
                    "no telemetry has arrived, so the aircraft's mode is unknown and this "
                    "adapter does not command blind"
                    if sample is None
                    else "the autopilot is not in armed Guided flight"
                ),
            }
            self.refusals.append(refusal)
            self.evidence.append_jsonl("refused-publications.jsonl", refusal)
            return None
        motion = MotionTarget(
            position_ned=target.position_ned,
            velocity_ned=target.velocity_ned,
            acceleration_ned=None,
            yaw_rad=target.yaw_rad,
            yaw_rate_rad_s=None,
        )
        self._sequence += 1
        setpoint = MotionSetpoint(
            command_sequence=self._sequence,
            mission_revision=0,
            goal_revision=0,
            nav_epoch=self.navigation_epoch,
            frame=Frame.ODOM,
            type_mask=mask_for_target(motion),
            target=motion,
            issue_stamp=self._stamp(),
            deadline_s=target.deadline_s,
            certificate_ref=target.certificate_ref,
            sample_ref=None,
            source=SetpointSource.NORMAL,
        )
        self._session.send_setpoint(setpoint)
        publication = SetpointPublication(setpoint=setpoint, published_stamp=self._stamp())
        self._publications.append(publication)
        return publication

    def inject(self, fault: Injection) -> InjectionReceipt:
        """Send one injection and wait for the controller to confirm it was applied.

        A request is not reported as applied because it was sent; it is reported as
        applied when the controller's acknowledgement arrives, and not before.
        """
        self._gateway.send_fault(fault)
        deadline = self._monotonic() + 10.0
        while self._monotonic() < deadline:
            record = self.sensor_record(0.25)
            if record is None or record.fault_ack is None:
                # Wait a little rather than spinning: the acknowledgement arrives on
                # the same stream as the sensor records, so it needs its own step.
                self._sleep(0.05)
                continue
            ack = record.fault_ack
            if ack.get("injection_id") != fault.injection_id:
                self._pending_records.append(record)
                self._sleep(0.05)
                continue
            self.evidence.append_jsonl(
                "injections.jsonl",
                {"injection": fault.injection_id, "kind": fault.kind, "apply": fault.apply, "ack": ack},
            )
            return InjectionReceipt(
                requested=fault,
                applied=bool(ack.get("applied")),
                acknowledged_state=ack.get("state"),
                reason=None if ack.get("applied") == fault.apply else str(ack.get("reason")),
            )
        self.evidence.append_jsonl(
            "injections.jsonl",
            {"injection": fault.injection_id, "kind": fault.kind, "apply": fault.apply, "ack": None},
        )
        return InjectionReceipt(
            requested=fault,
            applied=False,
            acknowledged_state=None,
            reason="the controller did not acknowledge the injection within 10 s",
        )

    # -- internals ---------------------------------------------------------

    def _stamp(self) -> ClockStamp:
        return self.settings.capture_stamp(int(self._monotonic_ns()))

    def _record_mavlink(self, document: dict[str, Any]) -> None:
        if self._mavlink_lines >= self.MAX_MAVLINK_LOG_LINES:
            return
        self._mavlink_lines += 1
        self.evidence.append_jsonl("mavlink.jsonl", document)

    @property
    def publications(self) -> tuple[SetpointPublication, ...]:
        return tuple(self._publications)

    @property
    def mavlink_message_count(self) -> int:
        return self._mavlink_lines


# ---------------------------------------------------------------------------
# Observations from sensor records
# ---------------------------------------------------------------------------


def build_observation(
    record: SensorRecord,
    *,
    settings: PlatformSettings,
    calibration: Any,
    label: str,
    sequence: int,
    controller_status: dict[str, Any],
    writer: EvidenceWriter,
    store_payload: bool,
) -> "PairDescription":
    """Describe one stereo pair, and store its pixels when the run is keeping them.

    Every pair's metadata is recorded: identifiers, both frame counters, the shared
    capture instant, the host receipt instant, payload hashes and the measured frame
    quality. The pixels themselves are written for a bounded number of pairs, so a
    long run cannot fill the disk with frames nobody will look at; which pairs were
    stored is stated in the run's stereo evidence rather than left implicit.
    """
    if record.pair is None:
        raise FramingError("an observation needs a decoded stereo pair")
    pair = record.pair
    if pair.encoding != settings.stereo.encoding:
        raise FramingError(
            f"the controller sent {pair.encoding}, the configuration expects "
            f"{settings.stereo.encoding}"
        )
    if (pair.width, pair.height) != (settings.stereo.width, settings.stereo.height):
        raise FramingError(
            f"the controller sent {pair.width}x{pair.height}, the configuration expects "
            f"{settings.stereo.width}x{settings.stereo.height}"
        )
    left_quality = measure_frame_quality(
        pair.left_bytes,
        pair.width,
        pair.height,
        identical_channel_fraction_limit=settings.stereo.identical_channel_fraction_limit,
    )
    right_quality = measure_frame_quality(
        pair.right_bytes,
        pair.width,
        pair.height,
        identical_channel_fraction_limit=settings.stereo.identical_channel_fraction_limit,
    )
    declaration = (controller_status.get("scene") or {}).get("witness_colour_rgb")
    witness = None
    if declaration:
        witness = find_colour_witness(
            pair.left_bytes, pair.width, pair.height, declaration
        )
    metadata = {
        "sequence": sequence,
        "pair_id": pair.pair_id,
        "left_frame_id": pair.left_frame_id,
        "right_frame_id": pair.right_frame_id,
        "frame_id_gap": pair.right_frame_id - pair.left_frame_id,
        "sim_time_s": record.sim_time_s if record.sim_time_s >= 0.0 else None,
        "capture_monotonic_ns": pair.capture_host_ns,
        "receipt_monotonic_ns": record.received_stamp.monotonic_ns,
        "encoding": pair.encoding,
        "width": pair.width,
        "height": pair.height,
        "left_sha256": hashlib_sha256(pair.left_bytes),
        "right_sha256": hashlib_sha256(pair.right_bytes),
        "left_quality": describe_quality(left_quality),
        "right_quality": describe_quality(right_quality),
        "witness": witness,
        "stored": store_payload,
    }
    if not store_payload:
        writer.append_jsonl("pairs.jsonl", metadata)
        return PairDescription(metadata=metadata, observation=None)
    base = f"pairs/{sequence:05d}"
    left_path = writer.write_bytes(
        f"{base}-left.ppm", ppm_bytes(pair.left_bytes, pair.width, pair.height)
    )
    right_path = writer.write_bytes(
        f"{base}-right.ppm", ppm_bytes(pair.right_bytes, pair.width, pair.height)
    )
    metadata["left_ppm"] = left_path
    metadata["right_ppm"] = right_path
    writer.append_jsonl("pairs.jsonl", metadata)
    host_id = controller_status.get("host_id")
    clock_id = controller_status.get("clock_id")
    if not host_id or not clock_id:
        raise FramingError(
            "the controller did not declare its host and clock, so its capture stamp "
            "cannot be placed in a clock domain and cannot be compared with the receipt"
        )
    observation = Observation(
        episode_id=label,
        record_id=f"{label}-obs-{sequence:05d}",
        sensor_ids=SensorIds(
            left=settings.stereo.left, right=settings.stereo.right, imu=settings.imu.inertial_unit
        ),
        sequence=sequence,
        capture_stamp=ClockStamp(
            host_id=str(host_id), clock_id=str(clock_id), monotonic_ns=int(pair.capture_host_ns)
        ),
        receipt_stamp=ClockStamp(
            host_id=record.received_stamp.host_id,
            clock_id=record.received_stamp.clock_id,
            monotonic_ns=record.received_stamp.monotonic_ns,
        ),
        sim_time_s=record.sim_time_s if record.sim_time_s >= 0.0 else None,
        pair_id=f"{label}-pair-{pair.pair_id:06d}",
        left_payload=left_path,
        right_payload=right_path,
        encoding=pair.encoding,
        width=pair.width,
        height=pair.height,
        calibration_id=calibration.calibration_id,
        capture_pose_ref=None,
        quality=left_quality,
        depth_source=None,
    )
    return PairDescription(metadata=metadata, observation=observation)


@dataclass(frozen=True)
class PairDescription:
    """One pair's metadata, and its stored record when the pair was kept."""

    metadata: dict[str, Any]
    observation: Observation | None


def describe_quality(quality: FrameQuality) -> dict[str, Any]:
    """A JSON-ready view of one frame's measured quality."""
    return {
        "is_colour": quality.is_colour,
        "channel_identical_fraction": quality.channel_identical_fraction,
        "channel_means": list(quality.channel_means),
        "saturated_pixel_fraction": quality.saturated_pixel_fraction,
    }


def hashlib_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Probe results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeCheck:
    """One checklist item: what was measured, and whether it met the criterion."""

    name: str
    status: str
    evidence: dict[str, Any]
    reason: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Everything the probe concluded, with the artifacts that support it."""

    checks: tuple[ProbeCheck, ...]
    limitations: tuple[str, ...]
    manifest: dict[str, Any]
    artifacts: tuple[str, ...]

    @property
    def failed(self) -> tuple[ProbeCheck, ...]:
        return tuple(check for check in self.checks if check.status == "fail")

    @property
    def gate_status(self) -> GateStatus:
        """The gate passes only when every checklist item passed in some run."""
        return GateStatus.FAIL if self.failed else GateStatus.PASS

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f"{check.name}: {check.reason}" for check in self.failed if check.reason)


class _FlightLog:
    """What a flight produced, collected before anything about it is judged."""

    def __init__(self) -> None:
        self.publications: list[SetpointPublication] = []
        self.telemetry: list[TelemetrySample] = []
        self.pair_metadata: list[dict[str, Any]] = []
        self.observations: list[Observation] = []
        self.imu: list[ImuPayload] = []
        self.at_rest_sim_times: list[float] = []
        self.waypoints: list[dict[str, Any]] = []
        self.sim_time_pairs: list[tuple[float, float]] = []
        self.servo_table: list[tuple[int, ...]] = []


def window_reached(
    sim_times: Sequence[float], start_sim_time: float | None, window_s: float
) -> bool:
    """Whether a measurement window measured in simulation time has run its course.

    Simulation time arrives as a float, so a window that ends exactly on a sampling
    boundary can miss it by a rounding step. Without the tolerance the window would
    never close and the probe would sit there until its wall-clock ceiling.
    """
    if start_sim_time is None or not sim_times:
        return False
    return sim_times[-1] - start_sim_time >= window_s - TIME_COMPARISON_TOLERANCE_S


class CompatibilityProbe:
    """The live compatibility checklist, driven through replaceable seams.

    The probe owns the checklist order and the judgement of each item; the adapter
    owns everything a caller should not have to know about the simulator. Tests
    substitute factories for the three seams and drive the same sequence with
    scripted processes, telemetry and sensor streams.
    """

    def __init__(
        self,
        settings: PlatformSettings,
        *,
        output_dir: Path,
        runner_factory: Callable[[], Any] | None = None,
        session_factory: Callable[[], Any] | None = None,
        gateway_factory: Callable[[], Any] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.output_dir = Path(output_dir)
        self._runner_factory = runner_factory or SubprocessRunner
        self._session_factory = session_factory or PymavlinkSession
        # The gateway stamps each arriving record, so it is built with this run's clock
        # identity rather than inventing one of its own.
        self._gateway_factory = gateway_factory or (
            lambda: TcpSensorGateway(stamp=self._stamp)
        )
        self._monotonic_ns = monotonic_ns
        self._monotonic = monotonic
        self._sleep = sleep
        self._budget_deadline = 0.0
        self._calibration: Any = None
        self._artifacts: list[str] = []
        self._prerequisites: tuple[Prerequisite, ...] = ()

    def _stamp(self) -> ClockStamp:
        """One reading of this host's monotonic clock, in this run's clock domain."""
        return self.settings.capture_stamp(int(self._monotonic_ns()))

    # -- orchestration -----------------------------------------------------

    def run(self) -> ProbeResult:
        """Run the checklist twice and return everything that was measured.

        Run A uses the pinned parameters, where the autopilot's estimator is the
        simulator's own state. Run B adds the EKF-active parameter set so the
        estimator item has a configuration in which it means something.
        """
        self._prerequisites = require_prerequisites(self.settings, self.output_dir)
        self._record_artifacts(
            self._write_json(
                "prerequisites.json",
                [
                    {"name": check.name, "satisfied": check.satisfied, "detail": check.detail}
                    for check in self._prerequisites
                ],
            )
        )
        self._calibration = load_calibration_declaration(self.settings.calibration_path)

        checks_a, manifest_a, notes_a = self._run_once("run-a", (), estimator_run=False)
        checks_b, manifest_b, notes_b = self._run_once(
            "run-b", self.settings.estimator_params, estimator_run=True
        )

        checks = checks_a + checks_b
        self._record_artifacts(
            self._write_json(
                "checks.json",
                [
                    {
                        "name": check.name,
                        "status": check.status,
                        "reason": check.reason,
                        "evidence": check.evidence,
                    }
                    for check in checks
                ],
            )
        )
        limitations = (
            "sensor_mode is simulator-interface: the autopilot's attitude and position come "
            "from Webots devices through the flight-state packet, so this is compatibility "
            "evidence and not a sensor-derived result",
            "run A uses the pinned parameter set, where AHRS_EKF_TYPE 10 makes the simulator's "
            "own state the estimator, so its estimator item is not applicable and run B is the "
            "configuration that exercises estimator health",
            "this run proves transport, clocks and frames only: no localization, obstacle "
            "avoidance, timing suitability, autonomy or mission performance is claimed",
            "per-camera exposure offsets are not modelled; both eyes share the capture instant "
            "of the step in which they were read",
            "no depth device is used: stereo depth and its validity belong to P01-C",
        )
        manifest = {
            "stage": self.settings.stage,
            "name": self.settings.name,
            "records_revision": RECORDS_REVISION,
            "sensor_mode": SensorMode.SIMULATOR_INTERFACE.value,
            "prerequisites": [
                {"name": check.name, "detail": check.detail} for check in self._prerequisites
            ],
            "run_a": manifest_a,
            "run_b": manifest_b,
            "search_notes": notes_a + notes_b,
        }
        return ProbeResult(
            checks=checks,
            limitations=limitations,
            manifest=manifest,
            artifacts=tuple(self._artifacts),
        )

    def _run_once(
        self, label: str, extra_params: Sequence[Path], *, estimator_run: bool
    ) -> tuple[tuple[ProbeCheck, ...], dict[str, Any], tuple[str, ...]]:
        self._budget_deadline = self._monotonic() + self.settings.budget_wall_clock_s
        writer = EvidenceWriter(self.output_dir, label)
        adapter = self._new_adapter(writer, label, extra_params)
        checks: list[ProbeCheck] = []
        notes: list[str] = []
        try:
            startup_check = self._item_startup(adapter, writer, label)
            checks.append(startup_check)
            if startup_check.status == "fail":
                manifest = self._run_manifest(label, adapter, writer, extra_params)
                return tuple(checks), manifest, tuple(notes)

            log = _FlightLog()
            at_rest_imu, pre_settle = self._collect_imu(adapter, writer, log, label=label)
            checks.append(self._item_frames_and_timebases(adapter, writer, log, label=label))
            motion_check = self._item_guided_motion(adapter, writer, log)
            checks.append(motion_check)
            checks.append(self._item_actuator_mapping(adapter, writer, log))
            checks.append(self._item_stereo_pairs(adapter, writer, log, label=label))
            checks.append(self._item_imu_stream(adapter, writer, at_rest_imu, pre_settle, log))
            checks.append(self._item_stream_loss(adapter, writer, log))
            if estimator_run:
                checks.append(self._item_estimator_health(adapter, writer))
            if not estimator_run:
                notes.append(
                    "run A: the estimator item is not applicable because AHRS_EKF_TYPE 10 "
                    "makes the simulator's own state the estimator; see run B"
                )
                self._record_artifacts(
                    self._write_json(
                        f"{label}/estimator-not-applicable.json",
                        {
                            "applicable": False,
                            "reason": (
                                "AHRS_EKF_TYPE 10 (simulator AHRS) in the pinned parameter set"
                            ),
                            "parameter_evidence": self._parameter_evidence("AHRS_EKF_TYPE"),
                        },
                    )
                )
        finally:
            shutdown = adapter.stop()
        # Everything the evidence writer produced belongs in the run's artifact list,
        # including the files the pair and injection helpers wrote on the way through.
        self._record_artifacts(*writer.artifacts)
        manifest = self._run_manifest(label, adapter, writer, extra_params)
        manifest["shutdown"] = {"exits": shutdown.exits, "log_tails": shutdown.log_tails}
        return tuple(checks), manifest, tuple(notes)

    def _new_adapter(
        self, writer: EvidenceWriter, label: str, extra_params: Sequence[Path]
    ) -> WebotsArduPilot:
        return WebotsArduPilot(
            self.settings,
            runner=self._runner_factory(),
            session=self._session_factory(),
            gateway=self._gateway_factory(),
            evidence=writer,
            label=label,
            extra_params=extra_params,
            monotonic_ns=self._monotonic_ns,
            monotonic=self._monotonic,
            sleep=self._sleep,
        )

    def _run_manifest(
        self, label: str, adapter: WebotsArduPilot, writer: EvidenceWriter, extra_params: Sequence[Path]
    ) -> dict[str, Any]:
        """The configuration identity of one run: what ran, not what it concluded."""
        return {
            "label": label,
            "webots": {
                "application": str(self.settings.webots_home),
                "version": self.settings.webots_version,
                "mode": self.settings.webots_mode,
                "argv": list(self.settings.simulator_argv()),
            },
            "autopilot": {
                "root": str(self.settings.ardupilot_root),
                "configured_commit": self.settings.ardupilot_commit,
                "head": _git_head(self.settings.ardupilot_root),
                "vehicle": self.settings.vehicle,
                "sim_model": self.settings.sim_model,
                "home": self.settings.sitl_home,
                "argv": [str(part) for part in self.settings.sitl_argv(extra_params)],
                "parameter_files": [str(name) for name in (*self.settings.params, *extra_params)],
                "firmware": firmware_identity(adapter.latest_telemetry),
                "mavlink_messages_logged": adapter.mavlink_message_count,
            },
            "ports": {
                "fdm_port": self.settings.sim_port_out,
                "fdm_port_in": self.settings.sim_port_in,
                "controller_port": self.settings.endpoints.controller_port,
                "mavlink": self.settings.mavlink_endpoint,
            },
            "assets": {
                "world": str(self.settings.world),
                "world_sha256": _file_sha256(self.settings.world),
                "protos": scenario_proto_assets(self.settings.world),
                "parameter_sha256": {
                    str(name): _file_sha256(name)
                    for name in (*self.settings.params, *extra_params)
                },
                "calibration": str(self.settings.calibration_path),
                "calibration_sha256": _file_sha256(self.settings.calibration_path),
            },
            "sensors": {
                "stereo": {
                    "left": self.settings.stereo.left,
                    "right": self.settings.stereo.right,
                    "width": self.settings.stereo.width,
                    "height": self.settings.stereo.height,
                    "baseline_m": self.settings.stereo.baseline_m,
                    "sampling_period_ms": self.settings.stereo.sampling_period_ms,
                    "encoding": self.settings.stereo.encoding,
                    "identical_channel_fraction_limit": (
                        self.settings.stereo.identical_channel_fraction_limit
                    ),
                },
                "imu": {
                    "accelerometer": self.settings.imu.accelerometer,
                    "gyro": self.settings.imu.gyro,
                    "inertial_unit": self.settings.imu.inertial_unit,
                    "gps": self.settings.imu.gps,
                    "sampling_period_ms": self.settings.imu.sampling_period_ms,
                },
            },
            "controller": adapter.controller_status,
        }

    # -- checklist item 1 --------------------------------------------------

    def _item_startup(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, label: str
    ) -> ProbeCheck:
        startup = adapter.start()
        readiness = adapter.wait_ready(self.settings.step_timeout_s.startup)
        sample = readiness.telemetry
        firmware = sample.autopilot_version or {}
        document = {
            "label": label,
            "webots_version": startup.webots_version,
            "ports": startup.ports,
            "argv": {name: list(argv) for name, argv in startup.argv.items()},
            "pids": {"webots": startup.webots.pid, "sitl": startup.sitl.pid},
            "firmware": firmware,
            "controller_status": readiness.controller_status,
            "waited_s": readiness.waited_s,
            "log_head": {
                "webots": writer.tail_lines(writer.directory / "webots.log"),
                "sitl": writer.tail_lines(writer.directory / "sitl.log"),
            },
        }
        path = writer.write_json("startup.json", document)
        self._record_artifacts(path)
        reasons: list[str] = []
        if not readiness.controller_status:
            reasons.append("the Webots controller never reported its status")
        if sample.heartbeats == 0:
            reasons.append("no MAVLink heartbeat arrived")
        if not firmware:
            reasons.append("AUTOPILOT_VERSION did not arrive, so the firmware identity is unknown")
        declared = readiness.controller_status.get("devices") or {}
        expected = {
            "left": self.settings.stereo.left,
            "right": self.settings.stereo.right,
            "accelerometer": self.settings.imu.accelerometer,
            "gyro": self.settings.imu.gyro,
            "inertial_unit": self.settings.imu.inertial_unit,
        }
        for role, expected_name in expected.items():
            found = declared.get(role)
            if found != expected_name:
                reasons.append(
                    f"the scene's {role} device is {found!r}, the configuration expects "
                    f"{expected_name!r}"
                )
        return ProbeCheck(
            name=f"1_startup_and_transport[{label}]",
            status="fail" if reasons else "pass",
            evidence=document,
            reason="; ".join(reasons) if reasons else None,
        )

    # -- checklist item 2 --------------------------------------------------

    def _sample_autopilot_clock(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, log: _FlightLog, *, label: str
    ) -> list[tuple[float, float]]:
        """Pair host receipt times with the autopilot's own clock, once per message batch.

        The sensor stream is read on every pass as well. The cameras and the inertial
        devices produce about ten times as fast as this loop samples the autopilot, so a
        loop that ignored the stream would fill the controller's queue and its frames
        would be dropped before any of them reached the stereo evidence.
        """
        pairs: list[tuple[float, float]] = []
        deadline = self._monotonic() + self.settings.step_timeout_s.ready
        while len(pairs) < self.settings.timebase_samples and self._monotonic() < deadline:
            self._check_budget()
            sample = adapter.telemetry()
            if sample.boot_time_ms is not None:
                pairs.append(
                    (sample.received_stamp.monotonic_ns / 1e9, sample.boot_time_ms / 1000.0)
                )
            self._read_records(adapter, writer, log, label=label)
            self._sleep(0.02)
        return pairs

    def _item_frames_and_timebases(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, log: _FlightLog, *, label: str
    ) -> ProbeCheck:
        pairs = self._sample_autopilot_clock(adapter, writer, log, label=label)
        join = join_timebase(pairs, self.settings.timebase_spread_limit_ms)
        # A second, separate comparison: the host clock against the simulator's own
        # clock. Simulated physics may run faster, slower or stop, so the two are
        # related by measurement and never assumed equal.
        simulator_join = join_timebase(
            log.sim_time_pairs, spread_limit_ms=self.settings.timebase_spread_limit_ms
        )
        self._record_artifacts(
            self._write_json(
                f"{label}/timebase.json",
                {
                    "samples": join.samples,
                    "offset_s": join.offset_s,
                    "drift_ppm": join.drift_ppm,
                    "spread_ms": join.spread_ms,
                    "joined": join.joined,
                    "reason": join.reason,
                    "device_clock": "telemetry time_boot_ms",
                    "host_clock": f"{self.settings.host_id}/{self.settings.clock_id}",
                },
            )
        )
        world_zero = self._world_zero_point()
        document = {
            "frames": {
                "body": {
                    "axes": "forward-left-up",
                    "source": "declaration (specification section 4.2)",
                },
                "camera_optical": {
                    "axes": "right-down-forward",
                    "source": "declaration (specification section 4.2)",
                },
                "odom": {
                    "axes": "gravity-aligned local NED, horizontal axes fixed at start",
                    "source": "declaration; the autopilot's local frame is the conversion target",
                },
                "world": {"zero_point": world_zero, "source": "the scene's own declaration"},
                "autopilot_local_origin": {
                    "home": self.settings.sitl_home,
                    "source": "the configured --home and the origin SITL reports",
                },
            },
            "host_to_autopilot": {
                "samples": join.samples,
                "offset_s": join.offset_s,
                "drift_ppm": join.drift_ppm,
                "spread_ms": join.spread_ms,
                "joined": join.joined,
                "reason": join.reason,
                "device_clock": "telemetry time_boot_ms",
                "host_clock": f"{self.settings.host_id}/{self.settings.clock_id}",
            },
            "host_to_simulator": {
                "samples": simulator_join.samples,
                "offset_s": simulator_join.offset_s,
                "drift_ppm": simulator_join.drift_ppm,
                "spread_ms": simulator_join.spread_ms,
                "joined": simulator_join.joined,
                "reason": simulator_join.reason,
                "note": (
                    "recorded, not required: this spread reflects real scheduling jitter in "
                    "the simulator process, and a loose fit here does not hide a record whose "
                    "capture and receipt stamps are in one clock domain"
                ),
            },
            "cross_domain_rule": (
                "stamps are compared only inside one host/clock domain; simulation time is "
                "carried separately as sim_time_s"
            ),
        }
        path = writer.write_json("frames-and-timebases.json", document)
        self._record_artifacts(path)
        reasons: list[str] = []
        if not join.joined:
            reasons.append(f"unjoined clocks: {join.reason}")
        if simulator_join.joined is False and simulator_join.reason is not None and (
            "at least two" in simulator_join.reason or "did not advance" in simulator_join.reason
        ):
            reasons.append(f"the host and simulator clocks cannot be related: {simulator_join.reason}")
        if join.samples < self.settings.timebase_samples:
            reasons.append(
                f"only {join.samples} of {self.settings.timebase_samples} timebase samples "
                "arrived, so the join rests on fewer measurements than configured"
            )
        return ProbeCheck(
            name="2_frames_and_timebases",
            status="fail" if reasons else "pass",
            evidence=document,
            reason="; ".join(reasons) if reasons else None,
        )

    def _world_zero_point(self) -> Any:
        """The scene's declared zero point, read from its header rather than assumed."""
        try:
            text = self.settings.world.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            if "EMBODIED_WORLD_ZERO" in line:
                return line.strip()
        return None

    # -- helpers -----------------------------------------------------------

    def _write_json(self, name: str, document: Any) -> str:
        target = self.output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
        return str(target.relative_to(self.output_dir))

    def _record_artifacts(self, *paths: str | None) -> None:
        for path in paths:
            if path and path not in self._artifacts:
                self._artifacts.append(path)

    @property
    def artifacts(self) -> tuple[str, ...]:
        """Everything this probe has written so far, relative to the output directory."""
        return tuple(self._artifacts)

    def _check_budget(self) -> None:
        if self._monotonic() > self._budget_deadline:
            raise ProbeFailure(
                f"the run exceeded its configured budget of "
                f"{self.settings.budget_wall_clock_s:.0f}s and was stopped"
            )

    def _parameter_evidence(self, parameter: str) -> list[dict[str, Any]]:
        """The lines that set a parameter, with the file each came from.

        Reading the parameter files makes a claim such as "the estimator is the
        simulator's own state" checkable at the point where it matters.
        """
        found: list[dict[str, Any]] = []
        for path in (*self.settings.params, *self.settings.estimator_params):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or not stripped:
                    continue
                fields = stripped.split()
                if fields and fields[0] == parameter:
                    found.append({"file": path.name, "line": stripped})
        return found




    def _collect_imu(
        self,
        adapter: WebotsArduPilot,
        writer: EvidenceWriter,
        log: _FlightLog,
        *,
        label: str,
    ) -> tuple[list[ImuPayload], int]:
        """Measure the inertial devices at rest, once the scene has settled.

        The window is measured in simulation time, because that is the time the scene
        settles in: a freshly loaded airframe is pushed out of the floor and its
        accelerometer converges over about a second of simulated time, so a window
        measured on the wall clock would average the transient into the measurement.
        Samples before the settle time are counted and left out, and the raw material
        keeps them, tagged with the window they arrived in.
        """
        settled: list[ImuPayload] = []
        pre_settle = 0
        window_start_sim = None
        deadline = self._monotonic() + AT_REST_WALL_CLOCK_LIMIT_S
        while self._monotonic() < deadline:
            self._check_budget()
            for record in self._read_records(adapter, writer, log, label=label):
                if record.imu is None:
                    continue
                settle_time = self.settings.settle_s
                if record.sim_time_s < settle_time:
                    pre_settle += 1
                    self._write_imu_sample(writer, "settling", record)
                    continue
                if window_start_sim is None:
                    window_start_sim = record.sim_time_s
                settled.append(record.imu)
                self._write_imu_sample(writer, "at_rest", record)
                log.at_rest_sim_times.append(record.sim_time_s)
            if window_reached(
                log.at_rest_sim_times, window_start_sim, self.settings.at_rest_window_s
            ):
                return settled, pre_settle
            self._sleep(0.02)
        return settled, pre_settle

    def _read_records(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, log: _FlightLog, *, label: str
    ) -> list[SensorRecord]:
        """Take whatever the sensor stream has waiting, and file each record.

        The cameras and the inertial devices produce about 110 records a second while
        the probe's own cadence is a few samples a second, so this drains until the
        stream is momentarily empty or a short window closes. Reading a fixed small
        number instead would leave a backlog that grows for the whole run, and the
        controller drops frames when that backlog reaches its bound.
        """
        drained: list[SensorRecord] = []
        drain_until = self._monotonic() + SENSOR_DRAIN_WINDOW_S
        for _ in range(MAX_DRAINED_RECORDS):
            if self._monotonic() >= drain_until:
                return drained
            record = adapter.sensor_record(0.02)
            if record is None:
                return drained
            drained.append(record)
            if record.sim_time_s >= 0.0:
                log.sim_time_pairs.append(
                    (record.received_stamp.monotonic_ns / 1e9, record.sim_time_s)
                )
            if record.pair is not None:
                self._file_pair(
                    record, writer, log, label=label, controller_status=adapter.controller_status
                )
            elif record.imu is not None:
                log.imu.append(record.imu)
                self._write_imu_sample(writer, "flight", record)
        return drained

    def _file_pair(
        self,
        record: SensorRecord,
        writer: EvidenceWriter,
        log: _FlightLog,
        *,
        label: str,
        controller_status: dict[str, Any],
    ) -> None:
        """Turn one arriving pair into an observation, or record why it could not be."""
        sequence = len(log.pair_metadata) + 1
        store = len(log.observations) < PAIRS_STORED_PER_RUN
        try:
            description = build_observation(
                record,
                settings=self.settings,
                calibration=self._calibration,
                label=label,
                sequence=sequence,
                controller_status=controller_status,
                writer=writer,
                store_payload=store,
            )
        except RecordError as error:
            log.pair_metadata.append({"sequence": sequence, "record_error": str(error)})
            return
        log.pair_metadata.append(description.metadata)
        if description.observation is not None:
            log.observations.append(description.observation)

    def _write_imu_sample(self, writer: EvidenceWriter, phase: str, record: SensorRecord) -> None:
        """One inertial sample, tagged with the window it arrived in."""
        sample = record.imu
        assert sample is not None
        writer.append_jsonl(
            "imu.jsonl",
            {
                "phase": phase,
                "sim_time_s": record.sim_time_s,
                "capture_monotonic_ns": sample.capture_host_ns,
                "receipt_monotonic_ns": record.received_stamp.monotonic_ns,
                "accelerometer": list(sample.accelerometer),
                "gyro": list(sample.gyro),
                "inertial_unit_rpy": list(sample.inertial_unit_rpy),
                "device_names": list(sample.device_names),
                "units": sample.units,
            },
        )
        self._record_artifacts(
            str((writer.directory / "imu.jsonl").relative_to(self.output_dir))
        )

    def _sample_once(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, log: _FlightLog, *, label: str
    ) -> TelemetrySample:
        sample = adapter.telemetry()
        log.telemetry.append(sample)
        if sample.servo_outputs is not None:
            log.servo_table.append(sample.servo_outputs)
        self._read_records(adapter, writer, log, label=label)
        return sample

    # -- checklist item 3 --------------------------------------------------

    def _item_guided_motion(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, log: _FlightLog
    ) -> ProbeCheck:
        label = adapter.label
        flight = adapter.arm_and_guided(
            self.settings.step_timeout_s.flight,
            drain=lambda: self._read_records(adapter, writer, log, label=label),
        )
        reasons: list[str] = []
        if flight.refused:
            detail = "; ".join(flight.statustexts[-3:]) or "no STATUSTEXT was given"
            reasons.append(
                f"the autopilot did not enter Guided flight (mode reached="
                f"{flight.mode_reached}, armed={flight.armed}): {detail}"
            )
            # Nothing is published after a refusal. Continuing to send targets would be
            # exactly the mistake of assuming control the autopilot has not granted.
            return ProbeCheck(
                name=f"3_guided_local_ned_motion[{label}]",
                status="fail",
                evidence={
                    "mode_reached": flight.mode_reached,
                    "armed": flight.armed,
                    "takeoff_commanded_m": flight.takeoff_commanded_m,
                    "altitude_m": flight.altitude_m,
                    "statustexts": list(flight.statustexts),
                    "waypoints": [],
                    "publications": 0,
                    "adoption": "nothing was published: no motion command was attempted",
                },
                reason="; ".join(reasons),
            )
        for index, waypoint in enumerate(self.settings.waypoints_local_ned):
            self._check_budget()
            target = (
                waypoint[0],
                waypoint[1],
                waypoint[2] - self.settings.hover_altitude_m,
            )
            before = self._sample_once(adapter, writer, log, label=label)
            publication = None
            publications_here = 0
            refusals_here: list[dict[str, Any]] = []
            hold_until = self._monotonic() + self.settings.hold_per_waypoint_s
            while self._monotonic() < hold_until:
                self._check_budget()
                # Keep the stream alive while holding. A guided target has a life of
                # its own inside the autopilot, so one target per waypoint would hold
                # nothing; the deadline below is how long this sample stays valid.
                sent = adapter.send_local_ned(
                    LocalNedTarget(
                        position_ned=target,
                        velocity_ned=(0.0, 0.0, 0.0),
                        yaw_rad=None,
                        deadline_s=self.settings.hold_per_waypoint_s,
                        certificate_ref=None,
                    )
                )
                if sent is None:
                    # The autopilot stopped flying Guided, so continuing to publish
                    # would be assuming control it has not granted. The hold ends here
                    # and what the aircraft did instead is measured below.
                    refusals_here.append(adapter.refusals[-1])
                    reasons.append(
                        f"Guided flight was lost while holding {list(target)}: "
                        f"{adapter.refusals[-1]['reason']}, observed mode "
                        f"{adapter.refusals[-1]['observed_mode']!r}"
                    )
                    break
                publication = sent
                publications_here += 1
                log.publications.append(sent)
                self._sample_once(adapter, writer, log, label=label)
                self._sleep(0.2)
            after = self._sample_once(adapter, writer, log, label=label)
            start = before.local_position_ned
            end = after.local_position_ned
            displacement = None
            residual_m = None
            if start is not None and end is not None:
                displacement = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
                residual_m = math.dist(end, target)
            published = None if publication is None else publication.setpoint
            record = {
                "waypoint": list(waypoint),
                "target_ned": list(target),
                "setpoint_sequence": None if published is None else published.command_sequence,
                "setpoint_type_mask": None if published is None else published.type_mask,
                "setpoint_frame": None if published is None else published.frame.value,
                "publications": publications_here,
                "published_at_monotonic_ns": (
                    None if publication is None else publication.published_stamp.monotonic_ns
                ),
                "deadline_s": None if published is None else published.deadline_s,
                "position_before_ned": list(start) if start is not None else None,
                "position_after_ned": list(end) if end is not None else None,
                "displacement_ned": list(displacement) if displacement is not None else None,
                "tracking_residual_m": residual_m,
                "refusals": refusals_here,
                "statustexts": list(after.statustexts[-5:]),
            }
            log.waypoints.append(record)
            writer.append_jsonl("motion.jsonl", record)
            self._record_artifacts(str((writer.directory / "motion.jsonl").relative_to(self.output_dir)))
        if not flight.mode_reached and not reasons:
            reasons.append("Guided mode was never reached")
        displacements = [entry["displacement_ned"] for entry in log.waypoints]
        measured = [value for entry in displacements if entry is not None for value in entry]
        if measured and max(abs(value) for value in measured) < MOTION_MIN_DISPLACEMENT_M:
            reasons.append(
                "the vehicle did not move: the largest measured displacement was "
                f"{max(abs(value) for value in measured):.3f} m"
            )
        for entry in log.waypoints:
            displacement = entry["displacement_ned"]
            if displacement is None:
                reasons.append(
                    f"waypoint {entry['waypoint']} produced no local position, so its motion "
                    "could not be measured"
                )
                continue
            for axis, (commanded, measured_axis) in enumerate(
                zip(entry["target_ned"], displacement)
            ):
                moved_opposite = (
                    commanded * measured_axis < 0.0
                    and abs(measured_axis) > AXIS_AGREEMENT_MARGIN_M
                )
                if abs(commanded) >= AXIS_AGREEMENT_COMMAND_M and moved_opposite:
                    reasons.append(
                        f"axis {axis}: commanded {commanded:+.2f} m but the vehicle moved "
                        f"{measured_axis:+.2f} m, opposite to the command"
                    )
        check = ProbeCheck(
            name=f"3_guided_local_ned_motion[{label}]",
            status="fail" if reasons else "pass",
            evidence={
                "mode_reached": flight.mode_reached,
                "armed": flight.armed,
                "takeoff_commanded_m": flight.takeoff_commanded_m,
                "altitude_m": flight.altitude_m,
                "statustexts": list(flight.statustexts),
                "waypoints": log.waypoints,
                "publications": len(log.publications),
                "refused_publications": len(adapter.refusals),
                "control_events": [event.document() for event in adapter.control_events],
                "adoption": "publication is recorded; adoption is judged from the telemetry above",
            },
            reason="; ".join(reasons) if reasons else None,
        )
        return check

    # -- checklist item 4 --------------------------------------------------

    def _item_actuator_mapping(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, log: _FlightLog
    ) -> ProbeCheck:
        motors = list(adapter.controller_status.get("motors") or [])
        reasons: list[str] = []
        active = 0
        best: tuple[int, ...] | None = None
        for table in log.servo_table:
            count = sum(1 for value in table if value > PWM_IDLE_RAW)
            if best is None or count > active:
                active, best = count, table
        if best is None:
            # No sample at all is a different finding from a sample showing nothing.
            reasons.append("SERVO_OUTPUT_RAW never arrived, so actuation was never observed")
        elif active == 0:
            reasons.append(
                "unmapped actuation: every servo output stayed at the idle pulse while the "
                "vehicle was commanded to move"
            )
        elif motors and active < len(motors):
            reasons.append(
                f"only {active} of the {len(motors)} configured motors reported an output"
            )
        if not motors:
            reasons.append("the controller did not declare its motor mapping")
        document = {
            "motor_names": motors,
            "channel_outputs": list(best) if best is not None else None,
            "active_channels": active,
            "idle_pulse_us": PWM_IDLE_RAW,
            "requirement": "each configured motor channel must leave the idle pulse in flight",
        }
        path = writer.write_json("actuators.json", document)
        self._record_artifacts(path)
        return ProbeCheck(
            name=f"4_actuator_mapping[{adapter.label}]",
            status="fail" if reasons else "pass",
            evidence=document,
            reason="; ".join(reasons) if reasons else None,
        )

    # -- checklist item 5 --------------------------------------------------

    def _item_stereo_pairs(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, log: _FlightLog, *, label: str
    ) -> ProbeCheck:
        reasons: list[str] = []
        metadata = log.pair_metadata
        stored = [entry for entry in metadata if entry.get("stored")]
        if not metadata:
            reasons.append("no stereo pair arrived from the controller")
        for entry in metadata:
            if entry.get("record_error"):
                reasons.append(f"pair {entry['sequence']}: {entry['record_error']}")
                continue
            if entry["frame_id_gap"] != 0:
                reasons.append(
                    f"pair {entry['pair_id']}: the left counter is {entry['left_frame_id']} and "
                    f"the right is {entry['right_frame_id']}, so the eyes were not read in one step"
                )
        colourless = [
            entry
            for entry in metadata
            if entry.get("left_quality")
            and not entry["left_quality"]["is_colour"]
        ]
        if colourless:
            first = colourless[0]
            reasons.append(
                "the frames carry no colour information: "
                f"{len(colourless)} pair(s) have identical channels on "
                f"{first['left_quality']['channel_identical_fraction']:.3f} of pixels "
                f"(declared limit {self.settings.stereo.identical_channel_fraction_limit})"
            )
        witness = None
        declaration = (adapter.controller_status.get("scene") or {}).get("witness_colour_rgb")
        if metadata and not declaration:
            reasons.append(
                "the scene did not declare a known-colour object, so channel order cannot be "
                "checked against a witness"
            )
        elif metadata:
            witness = metadata[0].get("witness")
            if witness is None:
                reasons.append("the colour witness was not measured in the first stored pair")
            elif witness["matched_pixels"] == 0:
                reasons.append(
                    "the scene's known-colour object was not found in the frame, so the colour "
                    "pipeline is unproven"
                )
            elif not witness["matches"]:
                reasons.append(
                    "the scene's object appears with dominant channel "
                    f"{witness['observed_dominant_channel']!r} while the scene declares "
                    f"{witness['expected_dominant_channel']!r}: the channel order is wrong"
                )
        capture_times = [
            entry["capture_monotonic_ns"] / 1e9 for entry in metadata if "capture_monotonic_ns" in entry
        ]
        intervals = [
            later - earlier for earlier, later in zip(capture_times, capture_times[1:])
        ]
        if len(metadata) >= 2 and intervals and max(intervals) > self.settings.stereo_stale_after_s:
            reasons.append(
                f"the stereo stream stalled for {max(intervals):.2f}s, beyond the declared "
                f"{self.settings.stereo_stale_after_s:.2f}s"
            )
        if len(metadata) < 2:
            reasons.append(f"only {len(metadata)} stereo pair(s) arrived during the run")
        if not log.observations:
            reasons.append(
                "no stereo pair was stored, so no Observation record exists for this run"
            )
        document = {
            "pairs_captured": len(metadata),
            "pairs_stored": len(stored),
            "storage_bound": PAIRS_STORED_PER_RUN,
            "storage_note": (
                "every pair is described in pairs.jsonl with its frame counters, stamps, "
                "payload hashes and measured quality; pixels are written for the first "
                f"{PAIRS_STORED_PER_RUN} pairs so a run cannot fill the disk with frames"
            ),
            "records": [observation.record_id for observation in log.observations],
            "encoding": self.settings.stereo.encoding,
            "width": self.settings.stereo.width,
            "height": self.settings.stereo.height,
            "max_pair_interval_s": max(intervals) if intervals else None,
            "stale_after_s": self.settings.stereo_stale_after_s,
            "witness": witness,
            "controller_stream": (adapter.controller_status or {}).get("stream"),
            "stamp_semantics": (
                "capture is the controller's own monotonic reading at the step in which "
                "both eyes were read, receipt is this process's monotonic reading when the "
                "bytes arrived, and sim_time_s is that step's simulation time; per-camera "
                "exposure offsets are not modelled"
            ),
            "depth_source": None,
            "depth_note": "P00 uses no depth device; stereo depth belongs to P01-C",
        }
        path = writer.write_json(
            "stereo.json",
            {
                **document,
                "observations": [
                    write_record(observation) for observation in log.observations
                ],
            },
        )
        self._record_artifacts(path)
        return ProbeCheck(
            name=f"5_stereo_colour_pairs[{label}]",
            status="fail" if reasons else "pass",
            evidence=document,
            reason="; ".join(reasons) if reasons else None,
        )

    # -- checklist item 6 --------------------------------------------------

    def _item_imu_stream(
        self,
        adapter: WebotsArduPilot,
        writer: EvidenceWriter,
        at_rest: Sequence[ImuPayload],
        pre_settle: int,
        log: _FlightLog,
    ) -> ProbeCheck:
        reasons: list[str] = []
        samples = list(at_rest) + list(log.imu)
        names = {
            "accelerometer": self.settings.imu.accelerometer,
            "gyro": self.settings.imu.gyro,
            "inertial_unit": self.settings.imu.inertial_unit,
        }
        if not samples:
            reasons.append("no inertial sample arrived")
        device_names = samples[0].device_names if samples else ()
        for index, role in enumerate(("accelerometer", "gyro", "inertial_unit")):
            if samples and device_names[index] != names[role]:
                reasons.append(
                    f"the {role} device is named {device_names[index]!r}, the configuration "
                    f"expects {names[role]!r}"
                )
        units = samples[0].units if samples else None
        if not at_rest:
            reasons.append(
                f"no inertial sample arrived after the {self.settings.settle_s:.1f}s the scene "
                "needs to settle, so the at-rest measurement has nothing to report"
            )
        # A value that is not a number is not a measurement: averaging one into the
        # window would turn the whole window into a number that means nothing.
        non_finite = [
            index
            for index, sample in enumerate(at_rest)
            if not all(math.isfinite(value) for value in sample.accelerometer + sample.gyro)
        ]
        if non_finite:
            reasons.append(
                f"{len(non_finite)} of {len(at_rest)} inertial samples inside the settled "
                "window report a value that is not a number"
            )
        # Gaps are measured inside each window in which the probe was reading the
        # stream. The pause between two windows is the reader being busy, not the
        # stream stalling, and the evidence says which windows were measured.
        window_gaps = {
            "at_rest": imu_gaps(at_rest),
            "flight": imu_gaps(log.imu),
        }
        largest_gap = max(
            (gap for gaps in window_gaps.values() for gap in gaps), default=None
        )
        if largest_gap is not None and largest_gap > self.settings.imu_stale_after_s:
            reasons.append(
                f"the inertial stream stalled for {largest_gap:.2f}s inside a sampling "
                f"window, beyond the declared {self.settings.imu_stale_after_s:.2f}s"
            )
        measured = [sample for index, sample in enumerate(at_rest) if index not in non_finite]
        at_rest_mean = axis_mean_and_spread([sample.accelerometer for sample in measured])
        gravity_expected = GRAVITY_NED_Z
        gravity_ok = None
        if at_rest_mean["mean"] is not None:
            gravity_ok = abs(at_rest_mean["mean"][2] - gravity_expected) <= GRAVITY_TOLERANCE
            if not gravity_ok:
                reasons.append(
                    f"at rest the accelerometer reads {at_rest_mean['mean'][2]:+.2f} m/s^2 on the "
                    f"down axis while an at-rest reading in this frame is "
                    f"{gravity_expected:+.2f}: the sign or the frame is wrong"
                )
        cross_check = self._imu_cross_check(adapter, writer, samples, log)
        if cross_check["joined"] is False:
            reasons.append(cross_check["reason"])
        window_start = log.at_rest_sim_times[0] if log.at_rest_sim_times else None
        document = {
            "samples": len(samples),
            "at_rest_samples": len(at_rest),
            "at_rest_window_sim_time_s": (
                None
                if window_start is None
                else round(log.at_rest_sim_times[-1] - window_start, 3)
            ),
            "at_rest_window_start_sim_time_s": window_start,
            "settle_s": self.settings.settle_s,
            "pre_settle_samples_discarded": pre_settle,
            "non_finite_samples": len(non_finite),
            "device_names": list(device_names),
            "units": units,
            "largest_gap_s": largest_gap,
            "gap_windows": {name: len(gaps) for name, gaps in window_gaps.items()},
            "stale_after_s": self.settings.imu_stale_after_s,
            "at_rest_accelerometer": at_rest_mean,
            "gravity_expectation_m_s2": gravity_expected,
            "gravity_tolerance_m_s2": GRAVITY_TOLERANCE,
            "gravity_sign_as_expected": gravity_ok,
            "cross_check": cross_check,
        }
        path = writer.write_json("imu.json", document)
        self._record_artifacts(path)
        return ProbeCheck(
            name=f"6_imu_stream[{adapter.label}]",
            status="fail" if reasons else "pass",
            evidence=document,
            reason="; ".join(reasons) if reasons else None,
        )

    def _imu_cross_check(
        self,
        adapter: WebotsArduPilot,
        writer: EvidenceWriter,
        samples: Sequence[ImuPayload],
        log: _FlightLog,
    ) -> dict[str, Any]:
        """Compare the scene's inertial attitude with the autopilot's, as a relation not an equality.

        The two are different quantities computed in different places, so the result
        recorded here is the measured difference at matched local times, one way.
        """
        if not samples or not log.telemetry:
            return {
                "joined": False,
                "reason": "the inertial cross-check has no telemetry at a declared clock to join to",
                "differences_rad": None,
            }
        sample = samples[-1]
        nearest = min(
            log.telemetry,
            key=lambda item: abs(item.received_stamp.monotonic_ns - sample.capture_host_ns),
        )
        if nearest.attitude_rpy is None:
            return {
                "joined": False,
                "reason": "the autopilot never reported ATTITUDE, so the cross-check has no reference",
                "differences_rad": None,
            }
        differences = [
            abs(measured - reported)
            for measured, reported in zip(sample.inertial_unit_rpy, nearest.attitude_rpy)
        ]
        document = {
            "joined": True,
            "relation": "measured difference, one way: scene device minus autopilot telemetry",
            "differences_rad": differences,
            "scene_rpy_rad": list(sample.inertial_unit_rpy),
            "autopilot_rpy_rad": list(nearest.attitude_rpy),
            "matched_within_s": abs(
                nearest.received_stamp.monotonic_ns - sample.capture_host_ns
            )
            / 1e9,
        }
        writer.append_jsonl("imu-cross-check.jsonl", document)
        return document

    # -- checklist item 7 --------------------------------------------------

    def _item_stream_loss(
        self, adapter: WebotsArduPilot, writer: EvidenceWriter, log: _FlightLog
    ) -> ProbeCheck:
        reasons: list[str] = []
        if not log.publications:
            return ProbeCheck(
                name=f"7_setpoint_stream_loss[{adapter.label}]",
                status="fail",
                evidence={
                    "last_publication": None,
                    "window_s": self.settings.stream_loss_window_s,
                    "samples_in_window": [],
                    "samples_after_resume": [],
                    "control_regained": False,
                    "observed_not_designed": (
                        "no setpoint was ever published, so there was no stream to lose"
                    ),
                },
                reason=(
                    "the setpoint stream was never established, so losing it cannot be probed"
                ),
            )
        last_publication = log.publications[-1]
        target = log.waypoints[-1]["target_ned"] if log.waypoints else None
        window_samples: list[TelemetrySample] = []
        window_until = self._monotonic() + self.settings.stream_loss_window_s
        while self._monotonic() < window_until:
            self._check_budget()
            window_samples.append(self._sample_once(adapter, writer, log, label=adapter.label).document())
            self._sleep(0.2)
        resumed: list[dict[str, Any]] = []
        regained = False
        resume_refused: dict[str, Any] | None = None
        if target is not None:
            resent = adapter.send_local_ned(
                LocalNedTarget(
                    position_ned=tuple(target),
                    velocity_ned=(0.0, 0.0, 0.0),
                    yaw_rad=None,
                    deadline_s=self.settings.hold_per_waypoint_s,
                    certificate_ref=None,
                )
            )
            if resent is None:
                # Publishing again is only safe if the autopilot is again flying the
                # targets. When it is not, the refusal itself is the observation.
                resume_refused = adapter.refusals[-1]
            resume_until = self._monotonic() + 2.0
            while self._monotonic() < resume_until:
                self._check_budget()
                sample = self._sample_once(adapter, writer, log, label=adapter.label)
                resumed.append(sample.document())
                self._sleep(0.2)
            regained = bool(
                resumed and resumed[-1]["mode_name"] == "GUIDED" and resumed[-1]["armed"]
            )
        if len(window_samples) < 2:
            reasons.append(
                "fewer than two telemetry samples arrived during the stream-loss window, so no "
                "behaviour can be reported from evidence"
            )
        if not resumed:
            reasons.append("no telemetry arrived after the setpoint stream resumed")
        document = {
            "last_publication": (
                {
                    "sequence": last_publication.setpoint.command_sequence,
                    "published_at_monotonic_ns": last_publication.published_stamp.monotonic_ns,
                    "type_mask": last_publication.setpoint.type_mask,
                }
                if last_publication
                else None
            ),
            "window_s": self.settings.stream_loss_window_s,
            "samples_in_window": window_samples,
            "samples_after_resume": resumed,
            "control_regained": regained,
            "resume_refused": resume_refused,
            "control_events": [event.document() for event in adapter.control_events],
            "observed_not_designed": (
                "what the autopilot did during the gap is reported from its own telemetry; this "
                "run does not claim that any behaviour was safe or intended"
            ),
        }
        path = writer.write_json("stream-loss.json", document)
        self._record_artifacts(path)
        return ProbeCheck(
            name=f"7_setpoint_stream_loss[{adapter.label}]",
            status="fail" if reasons else "pass",
            evidence=document,
            reason="; ".join(reasons) if reasons else None,
        )

    # -- checklist item 8 --------------------------------------------------

    def _item_estimator_health(self, adapter: WebotsArduPilot, writer: EvidenceWriter) -> ProbeCheck:
        reasons: list[str] = []
        before = adapter.telemetry()
        injection = Injection.publish(
            0,
            kind=self.settings.estimator_fault.kind,
            magnitude_m=self.settings.estimator_fault.magnitude_m,
            hold_s=self.settings.estimator_fault.hold_s,
        )
        receipt = adapter.inject(injection)
        if not receipt.applied:
            reasons.append(
                f"the injection was requested but not applied: {receipt.reason or 'no reason given'}"
            )
        during: list[dict[str, Any]] = []
        hold_until = self._monotonic() + self.settings.estimator_fault.hold_s
        while self._monotonic() < hold_until:
            self._check_budget()
            during.append(adapter.telemetry().document())
            self._sleep(0.2)
        removal = adapter.inject(Injection.clear(1, kind=self.settings.estimator_fault.kind))
        if not removal.applied:
            reasons.append("the injection was not removed, so the run cannot report a recovery")
        after: list[dict[str, Any]] = []
        recover_until = self._monotonic() + 2.0
        while self._monotonic() < recover_until:
            self._check_budget()
            after.append(adapter.telemetry().document())
            self._sleep(0.2)
        variances = [
            entry["ekf_pos_horiz_variance"]
            for entry in (before.document(), *during)
            if entry.get("ekf_pos_horiz_variance") is not None
        ]
        flags = {
            entry.get("ekf_flags")
            for entry in (before.document(), *during, *after)
            if entry.get("ekf_flags") is not None
        }
        baseline = variances[0] if variances else None
        peak = max(variances) if variances else None
        ratio = None
        if baseline is not None and peak is not None and baseline > 0.0:
            ratio = peak / baseline
        observable = bool(ratio is not None and ratio >= EKF_VARIANCE_GROWTH) or len(flags) > 1
        if receipt.applied and not observable:
            reasons.append(
                "the injected degradation produced no observable estimator signal: "
                f"horizontal position variance ratio {ratio} and unchanged EKF flags {sorted(flags)}"
            )
        if not variances:
            reasons.append("EKF_STATUS_REPORT never reported a position variance")
        document = {
            "parameter_evidence": self._parameter_evidence("AHRS_EKF_TYPE"),
            "fault": {
                "kind": self.settings.estimator_fault.kind,
                "magnitude_m": self.settings.estimator_fault.magnitude_m,
                "hold_s": self.settings.estimator_fault.hold_s,
                "requested": injection.injection_id,
                "applied": receipt.applied,
                "acknowledged_state": receipt.acknowledged_state,
                "removed": removal.applied,
            },
            "before": before.document(),
            "during": during,
            "after": after,
            "variance_baseline": baseline,
            "variance_peak": peak,
            "variance_ratio": ratio,
            "ekf_flags_seen": sorted(flag for flag in flags if flag is not None),
            "observable_signal": observable,
            "growth_required": EKF_VARIANCE_GROWTH,
        }
        path = writer.write_json("estimator.json", document)
        self._record_artifacts(path)
        return ProbeCheck(
            name=f"8_estimator_health_loss[{adapter.label}]",
            status="fail" if reasons else "pass",
            evidence=document,
            reason="; ".join(reasons) if reasons else None,
        )


# ---------------------------------------------------------------------------
# Probe constants
# ---------------------------------------------------------------------------


# A motion check needs thresholds. They are declared here rather than hidden in a
# comparison, and they are deliberately loose: this gate asks whether a commanded
# axis produces motion in that direction, not whether the controller tracks it well.
# Stored stereo payloads are bounded per run: a hundred pairs at 640x480 is a
# hundred megabytes of frames nobody will open. Every pair is still described and
# hashed, so the bound is visible in the evidence rather than hidden in the code.
PAIRS_STORED_PER_RUN = 12

MOTION_MIN_DISPLACEMENT_M = 0.25
AXIS_AGREEMENT_COMMAND_M = 0.5
AXIS_AGREEMENT_MARGIN_M = 0.25

# A servo output sits around this pulse when the channel is idle. A channel that
# never leaves it is not being driven.
PWM_IDLE_RAW = 1000

# At rest in a north-east-down frame the accelerometer measures gravity on the
# down axis. The tolerance is generous on purpose: this check is about sign and
# frame, not about calibration.
GRAVITY_NED_Z = -9.81
GRAVITY_TOLERANCE = 1.0

# An injected position step counts as observable when the reported horizontal
# position variance at least this much larger than its pre-injection value, or
# when the EKF flags change at all.
EKF_VARIANCE_GROWTH = 1.2

# One pass over the sensor stream reads until it is momentarily empty or this window
# closes, up to a record count that also bounds the pass when the clock is a fake.
# The cameras and the inertial devices produce about 110 records a second, and the
# probe's own cadence is a few samples a second, so a fixed small count per pass
# would leave a backlog that grows for the whole run.
SENSOR_DRAIN_WINDOW_S = 0.05
MAX_DRAINED_RECORDS = 64

# A wall-clock ceiling on the at-rest measurement. The window itself is measured in
# simulation time; this only stops the probe from waiting forever if the simulation
# stops advancing, which is a finding rather than a reason to hang.
AT_REST_WALL_CLOCK_LIMIT_S = 60.0

# Simulation time is compared with this tolerance when a window's end is decided. It
# is far below a sampling period and far above the rounding noise on a second.
TIME_COMPARISON_TOLERANCE_S = 1e-6


def imu_gaps(samples: Sequence[ImuPayload]) -> list[float]:
    """Seconds between consecutive inertial samples, on the capture clock."""
    return [
        (later.capture_host_ns - earlier.capture_host_ns) / 1e9
        for earlier, later in zip(samples, samples[1:])
    ]


def axis_mean_and_spread(samples: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Mean and spread per axis, or None when there is nothing to average."""
    if not samples:
        return {"mean": None, "spread": None, "samples": 0}
    means = tuple(sum(sample[axis] for sample in samples) / len(samples) for axis in range(3))
    spreads = tuple(
        max(sample[axis] for sample in samples) - min(sample[axis] for sample in samples)
        for axis in range(3)
    )
    return {"mean": list(means), "spread": list(spreads), "samples": len(samples)}


def write_record(record: Any) -> Any:
    """The JSON document of one shared record, or the record itself when it has no codec."""
    from embodied.contracts.records import to_dict

    try:
        return to_dict(record)
    except RecordError:  # pragma: no cover - records written here always have a codec
        return None


def _file_sha256(path: Path) -> str | None:
    import hashlib

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def firmware_identity(sample: TelemetrySample | None) -> dict[str, Any] | None:
    """What the autopilot reported about itself, with its version blobs written as hex.

    AUTOPILOT_VERSION carries the custom firmware version as the eight bytes of the
    build's git hash and the board's unique id as sixteen. They arrive as byte lists,
    so they are written as hex: a reader comparing this run with the firmware it was
    made from should not have to guess the byte order of a list of small integers.
    """
    if sample is None or sample.autopilot_version is None:
        return None
    reported = dict(sample.autopilot_version)
    for field in (
        "flight_custom_version",
        "middleware_custom_version",
        "os_custom_version",
        "uid",
    ):
        value = reported.get(field)
        if isinstance(value, (bytes, bytearray, list)) and value:
            reported[field] = bytes(value).hex()
    return reported


def scenario_proto_assets(world: Path) -> list[dict[str, Any]]:
    """Every proto and mesh the world reaches, with its path and hash.

    The scene is copied from a pinned upstream example, so the manifest records what
    was actually loaded rather than what the configuration intended. The world names
    its protos with EXTERNPROTO and each proto names its meshes in a url field; both
    are read out of the files, because an asset that is never hashed is an asset
    nobody can compare with the revision it came from.
    """
    try:
        world_text = world.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    assets: list[dict[str, Any]] = []
    for line in world_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("EXTERNPROTO"):
            continue
        reference = stripped.split('"')
        if len(reference) < 2 or _is_remote_reference(reference[1]):
            continue
        proto = (world.parent / reference[1]).resolve()
        assets.append({"role": "proto", "path": str(proto), "sha256": _file_sha256(proto)})
        try:
            proto_text = proto.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for reference_field in re.findall(r"url\s*(\[[^\]]*\]|\"[^\"]*\")", proto_text):
            for name in re.findall(r'"([^"]+)"', reference_field):
                if _is_remote_reference(name):
                    continue
                mesh = (proto.parent / name).resolve()
                assets.append(
                    {"role": "mesh", "path": str(mesh), "sha256": _file_sha256(mesh)}
                )
    return assets


def _is_remote_reference(reference: str) -> bool:
    """Whether an asset reference would be fetched over the network rather than read."""
    return reference.startswith(("http://", "https://", "webots://"))


def build_compatibility_probe(settings: PlatformSettings, output_dir: Path) -> CompatibilityProbe:
    """Build the probe with the real seams.

    This is the one place that decides which process runner, MAVLink session and
    sensor gateway a real run uses. Tests substitute this function to drive the same
    checklist against scripted components.
    """
    return CompatibilityProbe(settings, output_dir=Path(output_dir))


def run_compatibility_probe(settings: PlatformSettings, output_dir: Path) -> ProbeResult:
    """Run the live compatibility checklist against the real simulator and autopilot."""
    return build_compatibility_probe(settings, output_dir).run()


# ---------------------------------------------------------------------------
# The compat command
# ---------------------------------------------------------------------------


def _add_compat_arguments(parser: Any) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/first_indoor.yaml"),
        help="the scenario and platform configuration this run declares",
    )


def _compat_command(args: Any, output_dir: Path) -> CommandOutcome:
    """Probe the Webots/ArduPilot candidate and record what actually happened.

    A malformed configuration is a command error (exit 1); a missing prerequisite
    is blocked (exit 2) and is reported before any process starts; a step that
    cannot produce its evidence is invalid (exit 1); a completed probe whose checks
    fail the gate is still a completed result (exit 0) with ``gate_status: fail``.
    """
    document = load_config(Path(args.config))
    settings = PlatformSettings.from_config(document, root=repository_root())
    probe = build_compatibility_probe(settings, output_dir)
    try:
        result = probe.run()
    except PlatformUnavailable as error:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(str(error),),
            limitations=(
                "no process was started and nothing was substituted: a blocked run reports the "
                "prerequisites it needs rather than faking the missing parts",
            ),
            manifest={
                "stage": settings.stage,
                "name": settings.name,
                "prerequisites": [
                    {"name": check.name, "satisfied": check.satisfied, "detail": check.detail}
                    for check in check_prerequisites(settings, output_dir)
                ],
                "configuration": _configuration_document(settings),
            },
            artifacts=probe.artifacts,
            sensor_mode=SensorMode.SIMULATOR_INTERFACE,
        )
    except ProbeFailure as error:
        return CommandOutcome(
            status=CommandStatus.INVALID,
            gate_status=GateStatus.FAIL,
            reasons=(str(error),),
            limitations=(
                "a probe step ended without its evidence; the partial material is retained "
                "beside this receipt and no run was retried silently",
            ),
            manifest={
                "stage": settings.stage,
                "name": settings.name,
                "configuration": _configuration_document(settings),
            },
            artifacts=probe.artifacts,
            sensor_mode=SensorMode.SIMULATOR_INTERFACE,
        )
    return CommandOutcome(
        status=CommandStatus.COMPLETE,
        gate_status=result.gate_status,
        reasons=result.reasons,
        limitations=result.limitations,
        manifest=result.manifest,
        artifacts=result.artifacts,
        sensor_mode=SensorMode.SIMULATOR_INTERFACE,
    )


def _configuration_document(settings: PlatformSettings) -> dict[str, Any]:
    """The configuration identity, recorded even when the run never started."""
    return {
        "stage": settings.stage,
        "name": settings.name,
        "webots": {
            "application": str(settings.webots_home),
            "configured_version": settings.webots_version,
            "mode": settings.webots_mode,
        },
        "autopilot": {
            "root": str(settings.ardupilot_root),
            "configured_commit": settings.ardupilot_commit,
            "head": _git_head(settings.ardupilot_root),
            "vehicle": settings.vehicle,
            "sim_model": settings.sim_model,
            "sitl_binary": str(settings.sitl_binary),
            "home": settings.sitl_home,
            "parameter_files": [str(name) for name in settings.params],
            "estimator_parameter_files": [str(name) for name in settings.estimator_params],
        },
        "scenario": {"world": str(settings.world)},
        "calibration": {"declaration": str(settings.calibration_path)},
        "endpoints": {
            "sitl": settings.endpoints.sitl,
            "fdm_port": settings.endpoints.fdm_port,
            "controller_port": settings.endpoints.controller_port,
        },
        "host": {"host_id": settings.host_id, "clock_id": settings.clock_id},
    }


register_command(
    "compat",
    _compat_command,
    help_text=(
        "Start one pinned Webots/ArduPilot candidate, probe sensor, command and failure "
        "behaviour, and retain a receipt with the raw evidence."
    ),
    stage_id="P00",
    run_prefix="p00-compat",
    add_arguments=_add_compat_arguments,
)
