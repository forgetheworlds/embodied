"""Stacked Perception / Map public ports (frozen Checkpoint 1).

Safety / Planner / Execution import from here. Legacy
``contracts.records.NavigationState`` remains for the fat mission stack and is
not this surface.

See ``docs/perception-ports-frozen.md`` in the project store.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from embodied.contracts.records import ClockStamp, RecordError


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------


class AgeDomain(str, Enum):
    SIM_CONTROL = "sim_control"
    MONOTONIC = "monotonic"


class NavStatus(str, Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    LOST = "lost"
    UNAVAILABLE = "unavailable"


class EvidenceClass(str, Enum):
    SENSOR_DERIVED = "sensor_derived"
    POSE_ASSISTED = "pose_assisted"
    ORACLE = "oracle"
    UNAVAILABLE = "unavailable"


class OccupancySupport(str, Enum):
    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]
NavEpoch = str
MapRevision = str
SnapshotId = str
CellIndex = tuple[int, int, int]


def _plain(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{name} must be a non-empty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"{name} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise RecordError(f"{name} must be finite")
    return number


def _vec3(value: object, name: str) -> Vec3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise RecordError(f"{name} must be a 3-tuple")
    return (_finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]"), _finite(value[2], f"{name}[2]"))


def _quat(value: object, name: str) -> QuatWXYZ:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise RecordError(f"{name} must be a 4-tuple quaternion")
    return (
        _finite(value[0], f"{name}[0]"),
        _finite(value[1], f"{name}[1]"),
        _finite(value[2], f"{name}[2]"),
        _finite(value[3], f"{name}[3]"),
    )


def _texts(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise RecordError(f"{name} must be a tuple of strings")
    for index, item in enumerate(value):
        _plain(item, f"{name}[{index}]")
    return value


def _optional_vec3(value: object, name: str) -> Vec3 | None:
    if value is None:
        return None
    return _vec3(value, name)


def _optional_numbers(value: object, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise RecordError(f"{name} must be a tuple of floats or None")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApDisagreement:
    compared: bool
    position_err_m: float | None
    velocity_err_mps: float | None
    yaw_err_rad: float | None
    ap_stamp: ClockStamp | None
    estimator_stamp: ClockStamp | None
    within_soft_bound: bool | None
    within_hard_bound: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.compared, bool):
            raise RecordError("compared must be a bool")
        if self.position_err_m is not None:
            _finite(self.position_err_m, "position_err_m")
        if self.velocity_err_mps is not None:
            _finite(self.velocity_err_mps, "velocity_err_mps")
        if self.yaw_err_rad is not None:
            _finite(self.yaw_err_rad, "yaw_err_rad")
        if self.ap_stamp is not None and not isinstance(self.ap_stamp, ClockStamp):
            raise RecordError("ap_stamp must be a ClockStamp or None")
        if self.estimator_stamp is not None and not isinstance(self.estimator_stamp, ClockStamp):
            raise RecordError("estimator_stamp must be a ClockStamp or None")
        for name in ("within_soft_bound", "within_hard_bound"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise RecordError(f"{name} must be a bool or None")
        if not self.compared:
            if self.within_soft_bound is not None or self.within_hard_bound is not None:
                raise RecordError("bounds must be None when compared is False")


def uncompared_disagreement() -> ApDisagreement:
    return ApDisagreement(
        compared=False,
        position_err_m=None,
        velocity_err_mps=None,
        yaw_err_rad=None,
        ap_stamp=None,
        estimator_stamp=None,
        within_soft_bound=None,
        within_hard_bound=None,
    )


@dataclass(frozen=True)
class NavPose:
    parent_frame: str
    child_frame: str
    stamp: ClockStamp
    position_m: Vec3
    quaternion_wxyz: QuatWXYZ
    covariance: tuple[float, ...] | None
    nav_epoch: NavEpoch
    source_ids: tuple[str, ...]
    valid: bool

    def __post_init__(self) -> None:
        _plain(self.parent_frame, "parent_frame")
        _plain(self.child_frame, "child_frame")
        if self.parent_frame == self.child_frame:
            raise RecordError("a pose needs two different frames")
        if not isinstance(self.stamp, ClockStamp):
            raise RecordError("stamp must be a ClockStamp")
        object.__setattr__(self, "position_m", _vec3(self.position_m, "position_m"))
        object.__setattr__(self, "quaternion_wxyz", _quat(self.quaternion_wxyz, "quaternion_wxyz"))
        object.__setattr__(self, "covariance", _optional_numbers(self.covariance, "covariance"))
        _plain(self.nav_epoch, "nav_epoch")
        object.__setattr__(self, "source_ids", _texts(self.source_ids, "source_ids"))
        if not isinstance(self.valid, bool):
            raise RecordError("valid must be a bool")


@dataclass(frozen=True)
class NavigationState:
    """Stacked estimator publish for Safety / Planner / Execution."""

    nav_epoch: NavEpoch
    state_sequence: int
    controller_alignment_id: str | None
    stamp: ClockStamp
    sim_time_s: float | None
    age_s: float
    age_domain: AgeDomain
    monotonic_observed_at_s: float | None
    pose: NavPose
    velocity_mps: Vec3 | None
    covariance: tuple[float, ...] | None
    status: NavStatus
    valid: bool
    sigma_pos_m: Vec3 | None
    visual_source_ids: tuple[str, ...]
    imu_source_ids: tuple[str, ...]
    visual_age_s: float | None
    imu_age_s: float | None
    feed_stall: bool
    ap_disagreement: ApDisagreement
    evidence_class: EvidenceClass
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _plain(self.nav_epoch, "nav_epoch")
        if isinstance(self.state_sequence, bool) or not isinstance(self.state_sequence, int) or self.state_sequence < 0:
            raise RecordError("state_sequence must be a non-negative int")
        if self.controller_alignment_id is not None:
            _plain(self.controller_alignment_id, "controller_alignment_id")
        if not isinstance(self.stamp, ClockStamp):
            raise RecordError("stamp must be a ClockStamp")
        if self.sim_time_s is not None:
            _finite(self.sim_time_s, "sim_time_s")
        _finite(self.age_s, "age_s")
        if not isinstance(self.age_domain, AgeDomain):
            raise RecordError("age_domain must be an AgeDomain")
        if self.monotonic_observed_at_s is not None:
            _finite(self.monotonic_observed_at_s, "monotonic_observed_at_s")
        if not isinstance(self.pose, NavPose):
            raise RecordError("pose must be a NavPose")
        if self.nav_epoch != self.pose.nav_epoch:
            raise RecordError("nav_epoch must equal pose.nav_epoch")
        object.__setattr__(self, "velocity_mps", _optional_vec3(self.velocity_mps, "velocity_mps"))
        object.__setattr__(self, "covariance", _optional_numbers(self.covariance, "covariance"))
        if not isinstance(self.status, NavStatus):
            raise RecordError("status must be a NavStatus")
        if not isinstance(self.valid, bool):
            raise RecordError("valid must be a bool")
        object.__setattr__(self, "sigma_pos_m", _optional_vec3(self.sigma_pos_m, "sigma_pos_m"))
        object.__setattr__(self, "visual_source_ids", _texts(self.visual_source_ids, "visual_source_ids"))
        object.__setattr__(self, "imu_source_ids", _texts(self.imu_source_ids, "imu_source_ids"))
        if self.visual_age_s is not None:
            _finite(self.visual_age_s, "visual_age_s")
        if self.imu_age_s is not None:
            _finite(self.imu_age_s, "imu_age_s")
        if not isinstance(self.feed_stall, bool):
            raise RecordError("feed_stall must be a bool")
        if not isinstance(self.ap_disagreement, ApDisagreement):
            raise RecordError("ap_disagreement must be an ApDisagreement")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise RecordError("evidence_class must be an EvidenceClass")
        object.__setattr__(self, "source_ids", _texts(self.source_ids, "source_ids"))

        if self.status in (NavStatus.STALE, NavStatus.LOST, NavStatus.UNAVAILABLE):
            if self.valid or self.pose.valid:
                raise RecordError(f"status={self.status.value} requires valid=False and pose.valid=False")
        if self.feed_stall and self.status not in (NavStatus.STALE, NavStatus.LOST):
            raise RecordError("feed_stall=True requires status stale or lost")
        if self.evidence_class is EvidenceClass.ORACLE:
            raise RecordError("oracle evidence_class is forbidden on autonomy NavigationState")


# ---------------------------------------------------------------------------
# Occupancy geometry + verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Aabb:
    min_m: Vec3
    max_m: Vec3
    frame: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_m", _vec3(self.min_m, "min_m"))
        object.__setattr__(self, "max_m", _vec3(self.max_m, "max_m"))
        _plain(self.frame, "frame")
        for axis in range(3):
            if self.min_m[axis] > self.max_m[axis]:
                raise RecordError("Aabb min_m must be <= max_m on every axis")


@dataclass(frozen=True)
class Sphere:
    center_m: Vec3
    radius_m: float
    frame: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_m", _vec3(self.center_m, "center_m"))
        object.__setattr__(self, "radius_m", _finite(self.radius_m, "radius_m"))
        if self.radius_m <= 0.0:
            raise RecordError("Sphere radius_m must be positive")
        _plain(self.frame, "frame")


@dataclass(frozen=True)
class StopTubeQuery:
    samples_odom_m: tuple[Vec3, ...]
    envelope_radius_m: float
    include_brake_region: bool
    brake_region: Aabb | Sphere | None

    def __post_init__(self) -> None:
        if not isinstance(self.samples_odom_m, tuple) or not self.samples_odom_m:
            raise RecordError("samples_odom_m must be a non-empty tuple")
        object.__setattr__(
            self,
            "samples_odom_m",
            tuple(_vec3(sample, f"samples_odom_m[{index}]") for index, sample in enumerate(self.samples_odom_m)),
        )
        object.__setattr__(self, "envelope_radius_m", _finite(self.envelope_radius_m, "envelope_radius_m"))
        if self.envelope_radius_m <= 0.0:
            raise RecordError("envelope_radius_m must be positive")
        if not isinstance(self.include_brake_region, bool):
            raise RecordError("include_brake_region must be a bool")
        if self.include_brake_region:
            if self.brake_region is None:
                raise RecordError("brake_region is required when include_brake_region is True")
            if not isinstance(self.brake_region, (Aabb, Sphere)):
                raise RecordError("brake_region must be an Aabb or Sphere")
        elif self.brake_region is not None:
            raise RecordError("brake_region must be None when include_brake_region is False")


@dataclass(frozen=True)
class DynamicEnvelope:
    track_id: str
    center_m: Vec3
    radius_m: float
    horizon_s: float
    frame: str

    def __post_init__(self) -> None:
        _plain(self.track_id, "track_id")
        object.__setattr__(self, "center_m", _vec3(self.center_m, "center_m"))
        object.__setattr__(self, "radius_m", _finite(self.radius_m, "radius_m"))
        object.__setattr__(self, "horizon_s", _finite(self.horizon_s, "horizon_s"))
        _plain(self.frame, "frame")


@dataclass(frozen=True)
class OccupancyVerdict:
    support: OccupancySupport
    reason: str
    nav_epoch: NavEpoch
    map_revision: MapRevision
    snapshot_id: SnapshotId
    query_stamp: ClockStamp
    sim_time_s: float | None
    age_s: float
    age_domain: AgeDomain
    evidence_class: EvidenceClass
    limiting_refs: tuple[str, ...]
    free_fraction: float | None
    capabilities: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.support, OccupancySupport):
            raise RecordError("support must be an OccupancySupport")
        _plain(self.reason, "reason")
        _plain(self.nav_epoch, "nav_epoch")
        _plain(self.map_revision, "map_revision")
        _plain(self.snapshot_id, "snapshot_id")
        if not isinstance(self.query_stamp, ClockStamp):
            raise RecordError("query_stamp must be a ClockStamp")
        if self.sim_time_s is not None:
            _finite(self.sim_time_s, "sim_time_s")
        _finite(self.age_s, "age_s")
        if not isinstance(self.age_domain, AgeDomain):
            raise RecordError("age_domain must be an AgeDomain")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise RecordError("evidence_class must be an EvidenceClass")
        object.__setattr__(self, "limiting_refs", _texts(self.limiting_refs, "limiting_refs"))
        if self.free_fraction is not None:
            fraction = _finite(self.free_fraction, "free_fraction")
            if not 0.0 <= fraction <= 1.0:
                raise RecordError("free_fraction must be in 0..1 when present")
        object.__setattr__(self, "capabilities", _texts(self.capabilities, "capabilities"))
        object.__setattr__(self, "limitations", _texts(self.limitations, "limitations"))
        if self.support is OccupancySupport.FREE and self.evidence_class is not EvidenceClass.SENSOR_DERIVED:
            raise RecordError("support=free requires evidence_class=sensor_derived")


# ---------------------------------------------------------------------------
# Ports (protocols)
# ---------------------------------------------------------------------------


@runtime_checkable
class OccupancyQuery(Protocol):
    nav_epoch: NavEpoch
    map_revision: MapRevision
    snapshot_id: SnapshotId
    submap_id: str
    stamp: ClockStamp
    sim_time_s: float | None
    age_s: float
    age_domain: AgeDomain
    healthy: bool
    evidence_class: EvidenceClass
    data_ages_s: tuple[tuple[str, float], ...]

    def query_volume(self, volume: Aabb | Sphere, *, now: ClockStamp) -> OccupancyVerdict: ...

    def query_stop_tube(self, tube: StopTubeQuery, *, now: ClockStamp) -> OccupancyVerdict: ...

    def query_cells(self, cells: tuple[CellIndex, ...], *, now: ClockStamp) -> OccupancyVerdict: ...

    def classify_cell(self, cell: CellIndex, *, now: ClockStamp) -> OccupancyVerdict: ...

    def query_fov_clearance(
        self,
        origin_m: Vec3,
        direction_unit: Vec3,
        max_range_m: float,
        *,
        now: ClockStamp,
    ) -> OccupancyVerdict: ...

    def query_unknown_intrusion(
        self,
        region: Aabb | Sphere,
        *,
        dynamic_speed_mps: float | None,
        dynamic_reach_s: float | None,
        now: ClockStamp,
    ) -> OccupancyVerdict: ...

    def dynamic_envelopes(self, *, horizon_s: float, now: ClockStamp) -> tuple[DynamicEnvelope, ...]: ...

    def matches_epoch(self, nav_epoch: NavEpoch) -> bool: ...

    def matches_revision(self, map_revision: MapRevision) -> bool: ...


@runtime_checkable
class EstimationPort(Protocol):
    def latest(self) -> NavigationState | None: ...

    def current_epoch(self) -> NavEpoch | None: ...

    def healthy(self) -> bool: ...


@runtime_checkable
class MappingPort(Protocol):
    def occupancy(self) -> OccupancyQuery | None: ...

    def current_revision(self) -> MapRevision | None: ...
