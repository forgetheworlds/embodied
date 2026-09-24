"""Shared runtime records: the SYSTEM-SPECIFICATION.md section 22 schemas.

One module, one definition per record, imported by every later stage so that no
stage invents a second version of the same idea. Two rules shape everything here.

**Missing information is missing, never a default.** Every field is required, and
a field that may be unavailable is a union with ``None`` that still has no
default, so a caller must write ``None`` deliberately and it stays ``null`` in
JSON. No field falls back to ``0``, ``0.0``, ``""`` or ``False``. A fabricated
default is indistinguishable from a measurement at every later reader.

**Records are frozen.** A reader may keep a record while it computes and know it
did not change underneath; corrected geometry is published as a new record with a
new revision.

The module imports with only the standard library, so a worker that needs the
schemas does not need numpy, YAML or MAVLink present.

Transform convention, written once: ``T_A_B`` maps a point expressed in frame B
into frame A. Frames come from the closed vocabulary in :class:`Frame`.

Records implemented here are the ones P00 freezes. Records deliberately left to a
later stage, with the owner named, are listed in :data:`DEFERRED_RECORDS`, so a
later stage does not re-invent a name for something already named.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
import types as _types
from typing import Any, Sequence, Union, get_args, get_origin, get_type_hints


# The identity of this contract revision. P01 and P02 consume it, and a change to
# any record below is an integration-owner decision rather than a worker's local
# edit. The authoritative identity of what actually ran is the code revision in the
# run's receipt; this name says which version of the contract that code carried.
RECORDS_REVISION = "p00-records-1"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class RecordError(ValueError):
    """A record was built or loaded with malformed or self-contradictory content."""


class ClockDomainError(RecordError):
    """Stamps from different hosts or clock domains were subtracted."""


def _plain(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise RecordError(f"{name} must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise RecordError(f"{name} must be positive, got {number}")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise RecordError(f"{name} must not be negative, got {value}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise RecordError(f"{name} must be positive, got {value}")
    return value


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name)


def _sequence(value: Any, name: str) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecordError(f"{name} must be a sequence")
    return value


def _numbers(value: Any, name: str, length: int | None = None) -> tuple[float, ...]:
    items = tuple(
        _finite(item, f"{name}[{index}]") for index, item in enumerate(_sequence(value, name))
    )
    if length is not None and len(items) != length:
        raise RecordError(f"{name} must hold {length} numbers, got {len(items)}")
    return items


def _optional_numbers(value: Any, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _numbers(value, name)


def _texts(value: Any, name: str) -> tuple[str, ...]:
    return tuple(
        _plain(item, f"{name}[{index}]") for index, item in enumerate(_sequence(value, name))
    )


def _optional_texts(value: Any, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _texts(value, name)


def _named_numbers(value: Any, name: str) -> tuple[tuple[str, float], ...]:
    """A sequence of (name, value) pairs, ordered and free of duplicate names."""
    pairs = []
    for index, item in enumerate(_sequence(value, name)):
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            raise RecordError(f"{name}[{index}] must be a (name, value) pair")
        pairs.append(
            (_plain(item[0], f"{name}[{index}] name"), _finite(item[1], f"{name}[{index}] value"))
        )
    names = [entry[0] for entry in pairs]
    if len(set(names)) != len(names):
        raise RecordError(f"{name} must not repeat a name")
    return tuple(pairs)


def _quaternion(value: Any, name: str) -> tuple[float, float, float, float]:
    numbers = _numbers(value, name, 4)
    norm = sum(component * component for component in numbers) ** 0.5
    if abs(norm - 1.0) > 1e-3:
        raise RecordError(f"{name} must be a unit quaternion, got norm {norm:.6f}")
    return (numbers[0], numbers[1], numbers[2], numbers[3])


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if not isinstance(value, kind):
        raise RecordError(f"{name} must be a {kind.__name__} value, got {value!r}")
    return value


def _record(value: Any, kind: type, name: str) -> Any:
    if not isinstance(value, kind):
        raise RecordError(f"{name} must be a {kind.__name__}")
    return value


def _optional_record(value: Any, kind: type, name: str) -> Any:
    if value is None:
        return None
    return _record(value, kind, name)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class Frame(str, Enum):
    """The four frame classes of specification section 4.2."""

    BODY = "body"
    CAMERA_OPTICAL = "camera_optical"
    # Continuous, gravity-aligned local frame whose horizontal axes are fixed at
    # start and need not point north. Control targets live here.
    ODOM = "odom"
    # Corrected frame for persistent places and routes. It is discontinuous when
    # a correction is published, so a control target is never expressed only in it.
    MAP = "map"


class SelectionGeometry(str, Enum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"
    MASK = "mask"


class ClaimKind(str, Enum):
    """Whether a reported fact was observed or inferred."""

    OBSERVATION = "observation"
    INFERENCE = "inference"


class GoalDisposition(str, Enum):
    """The supervisor's verdict on a proposed objective."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ExecutionDisposition(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


class SetpointSource(str, Enum):
    """Which motion the final publisher selected for one sample."""

    NORMAL = "normal"
    BACKUP = "backup"


class SensorMode(str, Enum):
    """How a run obtained its state. Run modes are never pooled or relabelled."""

    SENSOR_DERIVED = "sensor-derived"
    POSE_ASSISTED = "pose-assisted"
    # A transport and compatibility check: it scores no mission and claims no autonomy.
    SIMULATOR_INTERFACE = "simulator-interface"


# MAVLink position-target type-mask bits, taken from the message the autopilot reads
# rather than invented here (MAVLink POSITION_TARGET_TYPEMASK, the dialect the pinned
# firmware speaks). A set bit means "ignore this field"; the bit numbers are the
# dialect's, so the mask a record carries is the mask that goes on the wire.
#
# The autopilot reads these bits in *groups*: setting one axis of a group ignores the
# whole group, so a mask with part of a group set asks for something the sender did
# not write. The constants are therefore the groups themselves — x|y|z, vx|vy|vz,
# ax|ay|az — and a mask is only ever built out of whole groups.
TYPE_MASK_POSITION_IGNORE = 1 | 2 | 4  # x | y | z
TYPE_MASK_VELOCITY_IGNORE = 8 | 16 | 32  # vx | vy | vz
TYPE_MASK_ACCELERATION_IGNORE = 64 | 128 | 256  # ax | ay | az
# Not an ignore bit: FORCE_SET tells the autopilot this message is a force target.
# This system never commands a force target, so the bit stays clear in every mask.
TYPE_MASK_FORCE_SET = 512
TYPE_MASK_YAW_IGNORE = 1024
TYPE_MASK_YAW_RATE_IGNORE = 2048
TYPE_MASK_ALL = (
    TYPE_MASK_POSITION_IGNORE
    | TYPE_MASK_VELOCITY_IGNORE
    | TYPE_MASK_ACCELERATION_IGNORE
    | TYPE_MASK_FORCE_SET
    | TYPE_MASK_YAW_IGNORE
    | TYPE_MASK_YAW_RATE_IGNORE
)
# Position and velocity enabled: acceleration and both heading fields are ignored, and
# no force target is claimed. This is the mask the compatibility probe publishes.
TYPE_MASK_POSITION_VELOCITY = (
    TYPE_MASK_ACCELERATION_IGNORE | TYPE_MASK_YAW_IGNORE | TYPE_MASK_YAW_RATE_IGNORE
)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockStamp:
    """A reading in one named clock domain.

    Simulator time does not live here: simulated physics may run faster, slower or
    stop, so it travels beside the stamp as ``sim_time_s`` on the records that
    have it.
    """

    host_id: str
    clock_id: str
    monotonic_ns: int

    def __post_init__(self) -> None:
        _plain(self.host_id, "host_id")
        _plain(self.clock_id, "clock_id")
        _non_negative_int(self.monotonic_ns, "monotonic_ns")


def same_clock(first: ClockStamp, second: ClockStamp) -> bool:
    """Whether two stamps can be compared at all."""
    return (first.host_id, first.clock_id) == (second.host_id, second.clock_id)


def elapsed_ns(earlier: ClockStamp, later: ClockStamp) -> int:
    """Nanoseconds from ``earlier`` to ``later``, refusing cross-domain subtraction.

    This is the only subtraction of stamps in the system. Subtracting the raw
    integers is exactly the mistake specification section 4.1 describes: two
    numbers from different hosts or clock domains are not comparable merely
    because both are numbers.
    """
    if not same_clock(earlier, later):
        raise ClockDomainError(
            "cannot compare stamps from different clock domains: "
            f"{earlier.host_id}/{earlier.clock_id} and {later.host_id}/{later.clock_id}"
        )
    return later.monotonic_ns - earlier.monotonic_ns


# ---------------------------------------------------------------------------
# Support value types used by the records below
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels."""

    focal_length_px: tuple[float, float]
    principal_point_px: tuple[float, float]

    def __post_init__(self) -> None:
        _numbers(self.focal_length_px, "focal_length_px", 2)
        _numbers(self.principal_point_px, "principal_point_px", 2)
        if min(self.focal_length_px) <= 0.0:
            raise RecordError("focal_length_px must be positive on both axes")


@dataclass(frozen=True)
class Distortion:
    """Lens distortion model and its coefficients, as the calibration declared them."""

    model: str
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        _plain(self.model, "model")
        _numbers(self.coefficients, "coefficients")


@dataclass(frozen=True)
class Transform:
    """T_A_B: the pose of frame B expressed in frame A.

    Named so a reader cannot mistake a transform for its inverse: this is the
    transform that takes a point expressed in ``child_frame`` and expresses it in
    ``parent_frame``.
    """

    parent_frame: str
    child_frame: str
    translation_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _plain(self.parent_frame, "parent_frame")
        _plain(self.child_frame, "child_frame")
        if self.parent_frame == self.child_frame:
            raise RecordError("a transform needs two different frames")
        _numbers(self.translation_m, "translation_m", 3)
        _quaternion(self.quaternion_wxyz, "quaternion_wxyz")


@dataclass(frozen=True)
class SensorIds:
    """The device names a sensor record came from, recorded rather than assumed."""

    left: str
    right: str
    imu: str

    def __post_init__(self) -> None:
        _plain(self.left, "left")
        _plain(self.right, "right")
        _plain(self.imu, "imu")
        if self.left == self.right:
            raise RecordError("a stereo pair needs two distinct device names")


@dataclass(frozen=True)
class FrameQuality:
    """Measured properties of one captured frame.

    ``channel_identical_fraction`` is the measurement that exposes a grayscale
    image delivered through a colour-shaped buffer: when a bridge averages the
    colour channels, every pixel has R = G = B and the fraction is 1.0.
    """

    is_colour: bool
    channel_identical_fraction: float
    channel_means: tuple[float, float, float]
    saturated_pixel_fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.is_colour, bool):
            raise RecordError("is_colour must be decided against a declared criterion")
        fraction = _finite(self.channel_identical_fraction, "channel_identical_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise RecordError("channel_identical_fraction is a fraction of the image")
        _numbers(self.channel_means, "channel_means", 3)
        saturated = _finite(self.saturated_pixel_fraction, "saturated_pixel_fraction")
        if not 0.0 <= saturated <= 1.0:
            raise RecordError("saturated_pixel_fraction is a fraction of the image")


@dataclass(frozen=True)
class MotionTarget:
    """The motion fields one published setpoint activates.

    An absent field is None, and the setpoint's type mask must agree: a field the
    mask tells the autopilot to ignore cannot also carry a value.
    """

    position_ned: tuple[float, float, float] | None
    velocity_ned: tuple[float, float, float] | None
    acceleration_ned: tuple[float, float, float] | None
    yaw_rad: float | None
    yaw_rate_rad_s: float | None

    def __post_init__(self) -> None:
        _optional_numbers(self.position_ned, "position_ned")
        _optional_numbers(self.velocity_ned, "velocity_ned")
        _optional_numbers(self.acceleration_ned, "acceleration_ned")
        _optional_number(self.yaw_rad, "yaw_rad")
        _optional_number(self.yaw_rate_rad_s, "yaw_rate_rad_s")
        if (
            self.position_ned is None
            and self.velocity_ned is None
            and self.acceleration_ned is None
        ):
            raise RecordError("a motion target needs at least one supported motion field")
        if self.yaw_rad is not None and self.yaw_rate_rad_s is not None:
            raise RecordError("request either a heading or a heading rate, not both")


# ---------------------------------------------------------------------------
# Records P00 freezes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """One immutable calibration version.

    A changed calibration is a new version, never an edited record, so geometry
    derived under the old version stays explainable.
    """

    calibration_id: str
    version: str
    left_intrinsics: CameraIntrinsics
    right_intrinsics: CameraIntrinsics
    left_distortion: Distortion
    right_distortion: Distortion
    T_camera_left_camera_right: Transform
    T_body_camera_left: Transform
    T_body_imu: Transform
    baseline_m: float
    pixel_convention: str
    depth_convention: str
    time_offset_s: float | None
    time_offset_error_s: float | None
    validated_limits: str | None
    source: str

    def __post_init__(self) -> None:
        _plain(self.calibration_id, "calibration_id")
        _plain(self.version, "version")
        _record(self.left_intrinsics, CameraIntrinsics, "left_intrinsics")
        _record(self.right_intrinsics, CameraIntrinsics, "right_intrinsics")
        _record(self.left_distortion, Distortion, "left_distortion")
        _record(self.right_distortion, Distortion, "right_distortion")
        for name, expected in (
            ("T_camera_left_camera_right", ("camera_left", "camera_right")),
            ("T_body_camera_left", ("body", "camera_left")),
            ("T_body_imu", ("body", "imu")),
        ):
            transform = _record(getattr(self, name), Transform, name)
            if (transform.parent_frame, transform.child_frame) != expected:
                raise RecordError(
                    f"{name} must map {expected[1]} into {expected[0]}, got "
                    f"{transform.child_frame} into {transform.parent_frame}"
                )
        _positive(self.baseline_m, "baseline_m")
        _plain(self.pixel_convention, "pixel_convention")
        _plain(self.depth_convention, "depth_convention")
        _optional_number(self.time_offset_s, "time_offset_s")
        _optional_number(self.time_offset_error_s, "time_offset_error_s")
        if self.time_offset_error_s is not None and self.time_offset_error_s < 0.0:
            raise RecordError("time_offset_error_s must not be negative")
        if (self.time_offset_s is None) != (self.time_offset_error_s is None):
            raise RecordError("a time offset and its error come as a pair, or both stay unknown")
        if self.validated_limits is not None:
            _plain(self.validated_limits, "validated_limits")
        _plain(self.source, "source")


@dataclass(frozen=True)
class Observation:
    """One synchronized stereo pair with its calibration and capture-time identity.

    Capture and receipt are separate stamps because bytes become available after
    the exposure they describe, and a consumer that needs a fresh frame cares
    about both.
    """

    episode_id: str
    record_id: str
    sensor_ids: SensorIds
    sequence: int
    capture_stamp: ClockStamp
    receipt_stamp: ClockStamp
    sim_time_s: float | None
    pair_id: str | None
    left_payload: str | None
    right_payload: str | None
    encoding: str
    width: int
    height: int
    calibration_id: str
    capture_pose_ref: str | None
    quality: FrameQuality | None
    depth_source: str | None

    def __post_init__(self) -> None:
        _plain(self.episode_id, "episode_id")
        _plain(self.record_id, "record_id")
        _record(self.sensor_ids, SensorIds, "sensor_ids")
        _non_negative_int(self.sequence, "sequence")
        _record(self.capture_stamp, ClockStamp, "capture_stamp")
        _record(self.receipt_stamp, ClockStamp, "receipt_stamp")
        if elapsed_ns(self.capture_stamp, self.receipt_stamp) < 0:
            raise RecordError("an observation cannot be received before it was captured")
        _optional_number(self.sim_time_s, "sim_time_s")
        pair_fields = (self.pair_id, self.left_payload, self.right_payload)
        if any(field is not None for field in pair_fields) and not all(
            field is not None for field in pair_fields
        ):
            raise RecordError(
                "an observation carries the pair id and both payloads, or none of them"
            )
        if self.left_payload is not None:
            _plain(self.left_payload, "left_payload")
            _plain(self.right_payload, "right_payload")
            if self.left_payload == self.right_payload:
                raise RecordError("the two halves of a pair are two separate payloads")
            _plain(self.pair_id, "pair_id")
        _plain(self.encoding, "encoding")
        _positive_int(self.width, "width")
        _positive_int(self.height, "height")
        _plain(self.calibration_id, "calibration_id")
        if self.capture_pose_ref is not None:
            _plain(self.capture_pose_ref, "capture_pose_ref")
        _optional_record(self.quality, FrameQuality, "quality")
        if self.depth_source is not None:
            _plain(self.depth_source, "depth_source")


@dataclass(frozen=True)
class PoseEstimate:
    """One pose estimate with the frames, sources and navigation epoch it belongs to."""

    parent_frame: str
    child_frame: str
    stamp: ClockStamp
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    covariance: tuple[float, ...] | None
    nav_epoch: str
    source_ids: tuple[str, ...]
    valid: bool

    def __post_init__(self) -> None:
        _plain(self.parent_frame, "parent_frame")
        _plain(self.child_frame, "child_frame")
        if self.parent_frame == self.child_frame:
            raise RecordError("a pose needs two different frames")
        _record(self.stamp, ClockStamp, "stamp")
        _numbers(self.position_m, "position_m", 3)
        _quaternion(self.quaternion_wxyz, "quaternion_wxyz")
        _optional_numbers(self.covariance, "covariance")
        _plain(self.nav_epoch, "nav_epoch")
        _texts(self.source_ids, "source_ids")
        if not isinstance(self.valid, bool):
            raise RecordError("valid must be a bool")


@dataclass(frozen=True)
class NavigationState:
    """The estimator's current state, its health and the alignment it was produced under."""

    state_sequence: int
    pose: PoseEstimate
    velocity_mps: tuple[float, float, float] | None
    covariance: tuple[float, ...] | None
    nav_epoch: str
    visual_source_ids: tuple[str, ...]
    imu_source_ids: tuple[str, ...]
    status: str
    controller_alignment_id: str | None

    def __post_init__(self) -> None:
        _non_negative_int(self.state_sequence, "state_sequence")
        _record(self.pose, PoseEstimate, "pose")
        _optional_numbers(self.velocity_mps, "velocity_mps")
        _optional_numbers(self.covariance, "covariance")
        _plain(self.nav_epoch, "nav_epoch")
        _texts(self.visual_source_ids, "visual_source_ids")
        _texts(self.imu_source_ids, "imu_source_ids")
        _plain(self.status, "status")
        if self.controller_alignment_id is not None:
            _plain(self.controller_alignment_id, "controller_alignment_id")


@dataclass(frozen=True)
class VisualSelection:
    """A region chosen in one image, in that image's own coordinate convention.

    A selection is not geometry. It becomes a target only through grounding, which
    needs calibration and depth.
    """

    selection_id: str
    observation_id: str
    coordinate_convention: str
    geometry_kind: SelectionGeometry
    geometry: tuple[float, ...] | str
    crop_transform: Transform | None
    description: str | None
    confidence: float | None

    def __post_init__(self) -> None:
        _plain(self.selection_id, "selection_id")
        _plain(self.observation_id, "observation_id")
        _plain(self.coordinate_convention, "coordinate_convention")
        kind = _enum(self.geometry_kind, SelectionGeometry, "geometry_kind")
        if kind is SelectionGeometry.MASK:
            _plain(self.geometry, "geometry")
        else:
            numbers = _numbers(self.geometry, "geometry")
            if kind is SelectionGeometry.POINT and len(numbers) != 2:
                raise RecordError("a point selection is one (x, y) pair")
            if kind is SelectionGeometry.BOX and len(numbers) != 4:
                raise RecordError("a box selection is (x_min, y_min, x_max, y_max)")
            if kind is SelectionGeometry.POLYGON and len(numbers) < 6:
                raise RecordError("a polygon selection needs at least three (x, y) pairs")
        _optional_record(self.crop_transform, Transform, "crop_transform")
        if self.description is not None:
            _plain(self.description, "description")
        if self.confidence is not None:
            value = _finite(self.confidence, "confidence")
            if not 0.0 <= value <= 1.0:
                raise RecordError("confidence is a fraction in 0..1, calibrated or not")


@dataclass(frozen=True)
class GroundedTarget:
    """A selection resolved to geometry in an anchor frame, with its alternatives.

    Geometry is stored with the anchor identity and revision, so a later map
    correction can move the target without inventing a new global coordinate.
    """

    target_id: str
    track_id: str | None
    place_id: str | None
    selection_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    geometry: tuple[float, ...]
    frame: Frame
    anchor_id: str
    anchor_revision: str
    uncertainty: tuple[float, ...] | None
    identity_alternatives: tuple[str, ...] | None
    last_observed_stamp: ClockStamp | None
    valid: bool

    def __post_init__(self) -> None:
        _plain(self.target_id, "target_id")
        if self.track_id is not None:
            _plain(self.track_id, "track_id")
        if self.place_id is not None:
            _plain(self.place_id, "place_id")
        _texts(self.selection_ids, "selection_ids")
        _texts(self.observation_ids, "observation_ids")
        if not self.selection_ids or not self.observation_ids:
            raise RecordError("a grounded target cites the selection and observation it came from")
        _numbers(self.geometry, "geometry")
        _enum(self.frame, Frame, "frame")
        _plain(self.anchor_id, "anchor_id")
        _plain(self.anchor_revision, "anchor_revision")
        _optional_numbers(self.uncertainty, "uncertainty")
        _optional_texts(self.identity_alternatives, "identity_alternatives")
        _optional_record(self.last_observed_stamp, ClockStamp, "last_observed_stamp")
        if not isinstance(self.valid, bool):
            raise RecordError("valid must be a bool")


@dataclass(frozen=True)
class WorldSnapshot:
    """One immutable view of the map, published from a single commit."""

    snapshot_id: str
    map_revision: str
    anchor_transforms: tuple[Transform, ...]
    occupancy_ref: str | None
    moving_envelopes: tuple[str, ...] | None
    targets: tuple[str, ...]
    places: tuple[str, ...]
    data_ages: tuple[tuple[str, float], ...]
    navigation_state_ref: str | None

    def __post_init__(self) -> None:
        _plain(self.snapshot_id, "snapshot_id")
        _plain(self.map_revision, "map_revision")
        for index, transform in enumerate(_sequence(self.anchor_transforms, "anchor_transforms")):
            _record(transform, Transform, f"anchor_transforms[{index}]")
        if self.occupancy_ref is not None:
            _plain(self.occupancy_ref, "occupancy_ref")
        _optional_texts(self.moving_envelopes, "moving_envelopes")
        _texts(self.targets, "targets")
        _texts(self.places, "places")
        _named_numbers(self.data_ages, "data_ages")
        if self.navigation_state_ref is not None:
            _plain(self.navigation_state_ref, "navigation_state_ref")


@dataclass(frozen=True)
class MissionContract:
    """The exact instruction, what it was interpreted to require, and its bounds."""

    mission_id: str
    instruction: str
    interpreted_requirements: tuple[str, ...]
    revision: int
    evidence_obligations: tuple[str, ...]
    return_obligation: str
    allowed_scope: str
    budget: tuple[tuple[str, float], ...]
    unresolved_questions: tuple[str, ...]

    def __post_init__(self) -> None:
        _plain(self.mission_id, "mission_id")
        _plain(self.instruction, "instruction")
        _texts(self.interpreted_requirements, "interpreted_requirements")
        _non_negative_int(self.revision, "revision")
        _texts(self.evidence_obligations, "evidence_obligations")
        _plain(self.return_obligation, "return_obligation")
        _plain(self.allowed_scope, "allowed_scope")
        if not _named_numbers(self.budget, "budget"):
            raise RecordError("a mission contract without a budget bound is not a contract")
        _texts(self.unresolved_questions, "unresolved_questions")


@dataclass(frozen=True)
class DecisionRequest:
    """One cloud reasoning request: what it was given, and when an answer stops helping."""

    request_id: str
    sequence: int
    mission_revision: int
    base_goal_revision: int
    observation_ids: tuple[str, ...]
    snapshot_id: str | None
    response_deadline_s: float
    model_identity: str | None

    def __post_init__(self) -> None:
        _plain(self.request_id, "request_id")
        _non_negative_int(self.sequence, "sequence")
        _non_negative_int(self.mission_revision, "mission_revision")
        _non_negative_int(self.base_goal_revision, "base_goal_revision")
        _texts(self.observation_ids, "observation_ids")
        if self.snapshot_id is not None:
            _plain(self.snapshot_id, "snapshot_id")
        _positive(self.response_deadline_s, "response_deadline_s")
        if self.model_identity is not None:
            _plain(self.model_identity, "model_identity")


@dataclass(frozen=True)
class SpatialGoal:
    """A proposed objective with its bounds. Admission, not this record, authorizes motion."""

    proposal_id: str
    request_id: str | None
    fingerprint: str
    mission_revision: int
    base_goal_revision: int
    selection_ids: tuple[str, ...]
    target_refs: tuple[str, ...]
    intent: str
    constraints: tuple[str, ...]
    completion_condition: str
    lease_bounds: tuple[tuple[str, float], ...]
    local_discretion_bounds: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _plain(self.proposal_id, "proposal_id")
        if self.request_id is not None:
            _plain(self.request_id, "request_id")
        _plain(self.fingerprint, "fingerprint")
        _non_negative_int(self.mission_revision, "mission_revision")
        _non_negative_int(self.base_goal_revision, "base_goal_revision")
        _texts(self.selection_ids, "selection_ids")
        _texts(self.target_refs, "target_refs")
        if not (self.selection_ids or self.target_refs):
            raise RecordError("a spatial goal cites the evidence it was formed from")
        _plain(self.intent, "intent")
        _texts(self.constraints, "constraints")
        _plain(self.completion_condition, "completion_condition")
        lease = _named_numbers(self.lease_bounds, "lease_bounds")
        if not lease:
            raise RecordError("an unbounded objective is not a lease")
        for name, value in lease:
            if value <= 0.0:
                raise RecordError(f"lease_bounds[{name}] must be positive")
        _named_numbers(self.local_discretion_bounds, "local_discretion_bounds")


@dataclass(frozen=True)
class GoalStatus:
    """The admission decision and the objective's current execution disposition."""

    proposal_id: str
    request_id: str
    disposition: GoalDisposition
    reason: str | None
    admission_ref: str | None
    current_disposition: ExecutionDisposition

    def __post_init__(self) -> None:
        _plain(self.proposal_id, "proposal_id")
        _plain(self.request_id, "request_id")
        _enum(self.disposition, GoalDisposition, "disposition")
        if self.reason is not None:
            _plain(self.reason, "reason")
        if self.admission_ref is not None:
            _plain(self.admission_ref, "admission_ref")
        _enum(self.current_disposition, ExecutionDisposition, "current_disposition")


@dataclass(frozen=True)
class MotionSetpoint:
    """One published motion target: the only record that describes a command.

    There is no motor, throttle or pulse field here by design. Motion reaches the
    aircraft only as a supported MAVLink guided setpoint, so a record that cannot
    express a motor command cannot be mistaken for authority over one.
    """

    command_sequence: int
    mission_revision: int
    goal_revision: int
    nav_epoch: str
    frame: Frame
    type_mask: int
    target: MotionTarget
    issue_stamp: ClockStamp
    deadline_s: float
    certificate_ref: str | None
    sample_ref: str | None
    source: SetpointSource

    def __post_init__(self) -> None:
        _non_negative_int(self.command_sequence, "command_sequence")
        _non_negative_int(self.mission_revision, "mission_revision")
        _non_negative_int(self.goal_revision, "goal_revision")
        _plain(self.nav_epoch, "nav_epoch")
        frame = _enum(self.frame, Frame, "frame")
        if frame is Frame.MAP:
            raise RecordError(
                "a control target is never expressed only in the corrected map frame; "
                "resolve it into odom or body before publishing"
            )
        if isinstance(self.type_mask, bool) or not isinstance(self.type_mask, int):
            raise RecordError("type_mask must be an integer bitfield")
        if self.type_mask <= 0:
            raise RecordError("type_mask must select the fields this setpoint uses")
        if self.type_mask & ~TYPE_MASK_ALL:
            raise RecordError("type_mask has bits outside the position-target mask")
        _record(self.target, MotionTarget, "target")
        _record(self.issue_stamp, ClockStamp, "issue_stamp")
        _positive(self.deadline_s, "deadline_s")
        if self.certificate_ref is not None:
            _plain(self.certificate_ref, "certificate_ref")
        if self.sample_ref is not None:
            _plain(self.sample_ref, "sample_ref")
        _enum(self.source, SetpointSource, "source")
        self._check_mask_matches_target()

    def _check_mask_matches_target(self) -> None:
        """Every field group the mask tells the autopilot to ignore must be absent.

        Two rules, both from the receiving end. The autopilot reads a field group as a
        unit — one position axis set ignores all three — so a mask must name whole
        groups or it will be understood as something other than what it says. And every
        field a record's target carries has to be one the mask activates.
        """
        for group, name in (
            (TYPE_MASK_POSITION_IGNORE, "position"),
            (TYPE_MASK_VELOCITY_IGNORE, "velocity"),
            (TYPE_MASK_ACCELERATION_IGNORE, "acceleration"),
        ):
            if (self.type_mask & group) not in (0, group):
                raise RecordError(
                    f"type_mask sets part of the {name} group; the autopilot ignores a "
                    "field group whole, so a mask names complete groups"
                )
        for bit, value, name in (
            (TYPE_MASK_POSITION_IGNORE, self.target.position_ned, "position"),
            (TYPE_MASK_VELOCITY_IGNORE, self.target.velocity_ned, "velocity"),
            (TYPE_MASK_ACCELERATION_IGNORE, self.target.acceleration_ned, "acceleration"),
            (TYPE_MASK_YAW_IGNORE, self.target.yaw_rad, "yaw"),
            (TYPE_MASK_YAW_RATE_IGNORE, self.target.yaw_rate_rad_s, "yaw_rate"),
        ):
            if bool(self.type_mask & bit) == (value is not None):
                verb = "ignored" if self.type_mask & bit else "used"
                raise RecordError(
                    f"type_mask says {name} is {verb} but the target "
                    f"{'carries' if value is not None else 'omits'} it"
                )
        if self.type_mask & TYPE_MASK_FORCE_SET:
            raise RecordError(
                "this system never commands a force target, so its mask bit stays clear"
            )


@dataclass(frozen=True)
class ExecutionStatus:
    """What the supervisor believes is happening for one goal, and on what evidence."""

    goal_ref: str
    certificate_ref: str | None
    command_ref: str | None
    disposition: ExecutionDisposition
    evidence: tuple[str, ...]
    reasons: tuple[str, ...]
    horizon_s: float | None
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _plain(self.goal_ref, "goal_ref")
        if self.certificate_ref is not None:
            _plain(self.certificate_ref, "certificate_ref")
        if self.command_ref is not None:
            _plain(self.command_ref, "command_ref")
        disposition = _enum(self.disposition, ExecutionDisposition, "disposition")
        _texts(self.evidence, "evidence")
        _texts(self.reasons, "reasons")
        _optional_number(self.horizon_s, "horizon_s")
        _texts(self.capabilities, "capabilities")
        if disposition is ExecutionDisposition.COMPLETED and not self.evidence:
            raise RecordError("a completion is a claim with evidence; cite the evidence")


@dataclass(frozen=True)
class ReportClaim:
    """One fact a report asserts, separated into observed and inferred."""

    predicate: str
    target: str
    observed: int | float | bool | str
    support_refs: tuple[str, ...]
    kind: ClaimKind
    stamp: ClockStamp
    uncertainty: float | None
    unmet_requirements: tuple[str, ...] | None

    def __post_init__(self) -> None:
        _plain(self.predicate, "predicate")
        _plain(self.target, "target")
        if isinstance(self.observed, bool) or not isinstance(self.observed, (int, float, str)):
            raise RecordError("observed must be a count, a value or a state name")
        if isinstance(self.observed, float):
            _finite(self.observed, "observed")
        if isinstance(self.observed, str):
            _plain(self.observed, "observed")
        _texts(self.support_refs, "support_refs")
        kind = _enum(self.kind, ClaimKind, "kind")
        _record(self.stamp, ClockStamp, "stamp")
        if self.uncertainty is not None:
            uncertainty = _finite(self.uncertainty, "uncertainty")
            if uncertainty < 0.0:
                raise RecordError("uncertainty must not be negative")
        _optional_texts(self.unmet_requirements, "unmet_requirements")
        if kind is ClaimKind.OBSERVATION and not self.support_refs:
            raise RecordError("an observed fact cites the evidence it was observed in")


@dataclass(frozen=True)
class FinalReport:
    """A mission's claims, why it stopped, and which snapshots support them."""

    mission_revision: int
    claims: tuple[ReportClaim, ...]
    termination_reason: str
    unmet_requirements: tuple[str, ...]
    physical_return_status: str | None
    evidence_snapshot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_negative_int(self.mission_revision, "mission_revision")
        for index, claim in enumerate(_sequence(self.claims, "claims")):
            _record(claim, ReportClaim, f"claims[{index}]")
        _plain(self.termination_reason, "termination_reason")
        _texts(self.unmet_requirements, "unmet_requirements")
        if self.physical_return_status is not None:
            _plain(self.physical_return_status, "physical_return_status")
        _texts(self.evidence_snapshot_ids, "evidence_snapshot_ids")


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def to_dict(record: Any) -> dict[str, Any]:
    """Encode a record as plain JSON types, with every level sorted by key.

    Sorted keys make the encoding byte-stable, so two equal records hash the same
    and a receipt can be compared with a stored one.
    """
    if not dataclasses.is_dataclass(record) or isinstance(record, type):
        raise RecordError("to_dict takes one record instance")
    encoded = _encode(record)
    if not isinstance(encoded, dict):
        raise RecordError("a record must encode to a JSON object")
    return encoded


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encode(getattr(value, field.name))
            for field in sorted(dataclasses.fields(value), key=lambda item: item.name)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise RecordError(f"cannot write {type(value).__name__} into a document")


def from_dict(record: Any, document: Any) -> Any:
    """Decode a record from plain JSON types, rejecting unknown or missing keys."""
    record_type = _resolve_record_type(record)
    return _decode_record(record_type, document)


def _resolve_record_type(record: Any) -> type:
    if isinstance(record, str):
        try:
            return RECORD_TYPES[record]
        except KeyError:
            raise RecordError(f"{record!r} is not a known record name") from None
    if isinstance(record, type) and dataclasses.is_dataclass(record):
        return record
    raise RecordError("from_dict takes a record class or a record name")


def _decode_record(record_type: type, document: Any) -> Any:
    if not isinstance(document, dict):
        raise RecordError(f"{record_type.__name__} must be decoded from an object")
    hints = _resolved_hints(record_type)
    known = {field.name for field in dataclasses.fields(record_type)}
    unknown = sorted(set(document) - known)
    if unknown:
        raise RecordError(f"{record_type.__name__} does not have {', '.join(unknown)}")
    missing = sorted(known - set(document))
    if missing:
        raise RecordError(f"{record_type.__name__} is missing {', '.join(missing)}")
    values = {
        name: _decode_value(hints[name], document[name], f"{record_type.__name__}.{name}")
        for name in sorted(known)
    }
    return record_type(**values)


def _resolved_hints(record_type: type) -> dict[str, Any]:
    return get_type_hints(record_type)


def _decode_value(annotation: Any, document: Any, name: str) -> Any:
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, _types.UnionType):
        return _decode_union(get_args(annotation), document, name)
    if origin in (tuple, list):
        return _decode_tuple(get_args(annotation), document, name)
    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return _decode_enum(annotation, document, name)
        if dataclasses.is_dataclass(annotation):
            return _decode_record(annotation, document)
        if annotation is bool:
            if not isinstance(document, bool):
                raise RecordError(f"{name} must be a bool")
            return document
        if annotation is int:
            if isinstance(document, bool) or not isinstance(document, int):
                raise RecordError(f"{name} must be an integer")
            return document
        if annotation is float:
            return _finite(document, name)
        if annotation is str:
            return _plain(document, name)
    raise RecordError(f"{name} uses a type this module cannot decode")


def _decode_union(alternatives: tuple[Any, ...], document: Any, name: str) -> Any:
    failures = []
    for alternative in alternatives:
        if alternative is type(None):
            if document is None:
                return None
            continue
        try:
            return _decode_value(alternative, document, name)
        except RecordError as error:
            failures.append(str(error))
            continue
    raise RecordError(f"{name} does not match any allowed type ({'; '.join(failures)})")


def _decode_tuple(arguments: tuple[Any, ...], document: Any, name: str) -> tuple:
    if not isinstance(document, (list, tuple)):
        raise RecordError(f"{name} must be a list")
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return tuple(
            _decode_value(arguments[0], item, f"{name}[{index}]")
            for index, item in enumerate(document)
        )
    if len(document) != len(arguments):
        raise RecordError(f"{name} must hold {len(arguments)} values")
    return tuple(
        _decode_value(annotation, item, f"{name}[{index}]")
        for index, (annotation, item) in enumerate(zip(arguments, document))
    )


def _decode_enum(kind: type[Enum], document: Any, name: str) -> Any:
    if not isinstance(document, str):
        raise RecordError(f"{name} must be a {kind.__name__} name")
    try:
        return kind(document)
    except ValueError:
        allowed = ", ".join(member.value for member in kind)
        raise RecordError(f"{name} must be one of {allowed}, got {document!r}") from None


# ---------------------------------------------------------------------------
# What this revision implements, and what a later stage still owes
# ---------------------------------------------------------------------------

# The records this revision implements. P00 owns them and freezes them before P01
# and P02 consume the contract.
IMPLEMENTED_RECORDS = (
    "ClockStamp",
    "Calibration",
    "Observation",
    "PoseEstimate",
    "NavigationState",
    "VisualSelection",
    "GroundedTarget",
    "WorldSnapshot",
    "MissionContract",
    "DecisionRequest",
    "SpatialGoal",
    "GoalStatus",
    "MotionSetpoint",
    "ExecutionStatus",
    "ReportClaim",
    "FinalReport",
)

# Records section 22 names but this revision deliberately leaves to the stage that
# owns their behaviour, with the reason: defining them now would be scaffolding
# that nothing consumes, and a stub invites a consumer to code against fields
# nobody has validated. A stage that needs one of these adds it with its first
# real writer and records the contract change with the integrator.
DEFERRED_RECORDS: dict[str, str] = {
    "DepthEstimate": "P01 (stereo depth worker)",
    "GoalAssessment": "P03 (admission service)",
    "AcceptedGoal": "P03 (execution supervisor)",
    "TrajectoryCertificate": "P03 (planner and supervisor)",
    "EpisodeResult": "P02/P06 (recorder and scorer)",
}

# Every record name in section 22, implemented or deferred, so the contract can be
# checked as a whole instead of by whichever names someone happens to need today.
SPECIFICATION_RECORDS = IMPLEMENTED_RECORDS + tuple(sorted(DEFERRED_RECORDS))

# The support value types these records are built from.
SUPPORT_TYPES = (
    "CameraIntrinsics",
    "Distortion",
    "Transform",
    "SensorIds",
    "FrameQuality",
    "MotionTarget",
)

RECORD_TYPES: dict[str, type] = {
    name: globals()[name] for name in (IMPLEMENTED_RECORDS + SUPPORT_TYPES)
}
