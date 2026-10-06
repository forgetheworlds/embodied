"""Deterministic Vehicle API v1 contract tests (no Webots / SITL).

Public surface only: takeoff, command(Motion), land, state.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from embodied.contracts.records import (
    ClockStamp,
    Frame,
    TYPE_MASK_ACCELERATION_IGNORE,
    TYPE_MASK_POSITION_IGNORE,
    TYPE_MASK_YAW_IGNORE,
    TYPE_MASK_YAW_RATE_IGNORE,
)
from embodied.control import Motion, Result, Vehicle, VehicleState, Vec3
from embodied.control.vehicle import (
    _arm_refusal_recoverable,
    _estimate_ready,
    enu_to_ned,
    mask_for_target,
    ned_to_enu,
    wrap_angle_rad,
)


class _FakeEvidence:
    def append_jsonl(self, *_args, **_kwargs) -> None:
        return None

    def write_json(self, *_args, **_kwargs) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.setpoints: list = []
        self.modes: list[str] = []
        self.takeoffs: list[float] = []

    def send_setpoint(self, setpoint) -> None:
        self.setpoints.append(setpoint)

    def set_mode(self, mode: str) -> None:
        self.modes.append(mode)

    def arm(self) -> None:
        return None

    def takeoff(self, altitude_m: float) -> None:
        self.takeoffs.append(altitude_m)


class _FakeAdapter:
    def __init__(self, *, armed: bool = True, guided: bool = True) -> None:
        self.settings = SimpleNamespace(
            hover_altitude_m=1.5,
            host_id="test-host",
            clock_id="monotonic",
            pre_arm_wait_s=1.0,
            step_timeout_s=SimpleNamespace(flight=5.0),
            realtime_ratio_envelope=(0.5, 1.5),
        )
        self.evidence = _FakeEvidence()
        self.refusals: list[dict] = []
        self.navigation_epoch = "test-epoch"
        self._sequence = 0
        self._publications: list = []
        self._session = _FakeSession()
        self._now = 0.0
        self._boot_ms = 0
        self._statustexts: list[str] = []
        mode = "GUIDED" if guided else "LOITER"
        self._telemetry = SimpleNamespace(
            in_guided_mode=guided,
            armed=armed,
            mode_name=mode,
            local_position_ned=(0.0, 0.0, -1.5),
            velocity_ned=(0.1, -0.2, 0.0),
            attitude_rpy=(0.0, 0.0, 0.25),
            servo_outputs=(1100, 1100, 1100, 1100),
            home_position=(-353632610, 1491652300, 584000),
            boot_time_ms=self._boot_ms,
            statustexts=(),
        )

    @property
    def guidance_held(self) -> bool:
        sample = self._telemetry
        return bool(sample.in_guided_mode and sample.armed)

    @property
    def latest_telemetry(self):
        return self._telemetry

    def telemetry(self):
        return self._telemetry

    def _monotonic(self) -> float:
        return self._now

    def _monotonic_ns(self) -> int:
        return int(self._now * 1e9)

    def _sleep(self, seconds: float) -> None:
        self._now += seconds
        self._boot_ms = int(self._now * 1000)
        self._telemetry.boot_time_ms = self._boot_ms

    def _stamp(self):
        return ClockStamp(
            host_id=self.settings.host_id,
            clock_id=self.settings.clock_id,
            monotonic_ns=self._monotonic_ns(),
        )


def test_enu_ned_round_trip() -> None:
    assert enu_to_ned((1.0, 2.0, 3.0)) == (1.0, -2.0, -3.0)
    assert ned_to_enu((1.0, -2.0, -3.0)) == (1.0, 2.0, 3.0)


def test_command_is_one_shot_odom_to_ned() -> None:
    adapter = _FakeAdapter()
    vehicle = Vehicle(adapter)
    motion = Motion(
        position=Vec3(2.5, 0.4, 1.5),
        velocity=Vec3(0.0, 0.0, 0.0),
        yaw=0.0,
    )
    result = vehicle.command(motion)
    assert result == Result(accepted=True)
    assert len(adapter._session.setpoints) == 1
    last = adapter._session.setpoints[-1]
    assert last.frame is Frame.ODOM
    assert last.target.position_ned == enu_to_ned((2.5, 0.4, 1.5))
    assert last.target.velocity_ned == (0.0, 0.0, 0.0)
    assert last.target.yaw_rad == 0.0
    assert last.target.yaw_rate_rad_s is None
    assert not (last.type_mask & TYPE_MASK_POSITION_IGNORE)
    assert last.type_mask & TYPE_MASK_YAW_RATE_IGNORE
    assert last.type_mask & TYPE_MASK_ACCELERATION_IGNORE


def test_command_yaw_rate_turn_keeps_position() -> None:
    adapter = _FakeAdapter()
    vehicle = Vehicle(adapter)
    motion = Motion(
        position=Vec3(7.5, 0.5, 1.5),
        velocity=Vec3(0.0, 0.0, 0.0),
        yaw_rate=0.6,
    )
    assert vehicle.command(motion).accepted is True
    last = adapter._session.setpoints[-1]
    assert last.target.yaw_rad is None
    assert last.target.yaw_rate_rad_s == 0.6
    assert last.target.position_ned == enu_to_ned((7.5, 0.5, 1.5))
    assert last.type_mask & TYPE_MASK_YAW_IGNORE
    assert not (last.type_mask & TYPE_MASK_YAW_RATE_IGNORE)


def test_command_refuses_when_not_guided() -> None:
    adapter = _FakeAdapter(armed=False, guided=True)
    vehicle = Vehicle(adapter)
    result = vehicle.command(
        Motion(position=Vec3(1.0, 0.0, 1.5), velocity=Vec3(0.0, 0.0, 0.0))
    )
    assert result.accepted is False
    assert result.reason == "not in armed Guided flight"
    assert adapter._session.setpoints == []
    assert adapter.refusals


def test_command_refuses_yaw_and_yaw_rate_together() -> None:
    adapter = _FakeAdapter()
    vehicle = Vehicle(adapter)
    result = vehicle.command(
        Motion(
            position=Vec3(0.0, 0.0, 1.5),
            velocity=Vec3(0.0, 0.0, 0.0),
            yaw=0.0,
            yaw_rate=0.5,
        )
    )
    assert result.accepted is False
    assert "heading" in (result.reason or "")
    assert adapter._session.setpoints == []


def test_command_refuses_non_finite() -> None:
    adapter = _FakeAdapter()
    vehicle = Vehicle(adapter)
    result = vehicle.command(
        Motion(position=Vec3(math.nan, 0.0, 1.5), velocity=Vec3(0.0, 0.0, 0.0))
    )
    assert result.accepted is False
    assert adapter._session.setpoints == []


def test_takeoff_rejects_bad_altitude() -> None:
    vehicle = Vehicle(_FakeAdapter())
    assert vehicle.takeoff(0.0).accepted is False
    assert vehicle.takeoff(-1.0).accepted is False


def test_takeoff_accepts_when_already_guided() -> None:
    adapter = _FakeAdapter(armed=True, guided=True)
    result = Vehicle(adapter).takeoff(1.5)
    assert result.accepted is True
    assert adapter._session.takeoffs == [1.5]


def test_arm_refusal_recoverable_markers() -> None:
    assert _arm_refusal_recoverable("PreArm: Need Position Estimate")
    assert _arm_refusal_recoverable("Arm: Gyro 1 rate 440Hz < loop rate*1.8 450Hz")
    assert _arm_refusal_recoverable("Arm: VisOdom: not healthy")
    assert not _arm_refusal_recoverable("PreArm: 3D Accel calibration needed")


def test_estimate_ready_requires_pose_home_and_aiding() -> None:
    adapter = _FakeAdapter()
    sample = adapter.telemetry()
    assert _estimate_ready(adapter, sample) is True
    adapter._statustexts = ["EKF3 IMU0 stopped aiding"]
    assert _estimate_ready(adapter, sample) is False
    adapter._statustexts = ["EKF3 IMU0 is using external nav data"]
    sample.statustexts = ("PreArm: Need Position Estimate",)
    assert _estimate_ready(adapter, sample) is False


def test_land_sets_land_mode() -> None:
    adapter = _FakeAdapter()
    vehicle = Vehicle(adapter)
    assert vehicle.land().accepted is True
    assert adapter._session.modes == ["LAND"]


def test_state_reports_odom_pose() -> None:
    adapter = _FakeAdapter()
    state = Vehicle(adapter).state()
    assert isinstance(state, VehicleState)
    assert state.armed is True
    assert state.guided is True
    assert state.position == Vec3(*ned_to_enu((0.0, 0.0, -1.5)))
    assert state.velocity == Vec3(*ned_to_enu((0.1, -0.2, 0.0)))
    assert state.yaw == 0.25


def test_wrap_angle_rad() -> None:
    assert abs(wrap_angle_rad(math.pi + 0.1) + (math.pi - 0.1)) < 1e-9


def test_mask_for_hold_motion() -> None:
    from embodied.contracts.records import MotionTarget

    target = MotionTarget(
        position_ned=(1.0, 0.0, -1.5),
        velocity_ned=(0.0, 0.0, 0.0),
        acceleration_ned=None,
        yaw_rad=0.0,
        yaw_rate_rad_s=None,
    )
    mask = mask_for_target(target)
    assert mask & TYPE_MASK_ACCELERATION_IGNORE
    assert mask & TYPE_MASK_YAW_RATE_IGNORE
    assert not (mask & TYPE_MASK_POSITION_IGNORE)
    assert not (mask & TYPE_MASK_YAW_IGNORE)
