"""Deterministic control-layer contract tests (no Webots / SITL).

Checks the Vehicle API behaviour we rely on: ENU↔NED, setpoint masks,
yaw-rate spin with position held (not angle slam / open-loop XY), and
refuse-publish when not armed Guided.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from embodied.contracts.records import (
    ClockStamp,
    Frame,
    MotionTarget,
    TYPE_MASK_ACCELERATION_IGNORE,
    TYPE_MASK_POSITION_IGNORE,
    TYPE_MASK_YAW_IGNORE,
    TYPE_MASK_YAW_RATE_IGNORE,
)
from embodied.control.vehicle import (
    REFRESH_S,
    SPIN_RATE_RAD_S,
    LocalNedTarget,
    Vehicle,
    enu_to_ned,
    mask_for_target,
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
    """Minimal stand-in for WebotsArduPilot used only by Vehicle unit tests."""

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
        mode = "GUIDED" if guided else "LOITER"
        self._telemetry = SimpleNamespace(
            in_guided_mode=guided,
            armed=armed,
            mode_name=mode,
            local_position_ned=(0.0, 0.0, -1.5),
            attitude_rpy=(0.0, 0.0, 0.0),
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


def test_enu_to_ned_axes() -> None:
    assert enu_to_ned((1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)
    assert enu_to_ned((0.0, 1.0, 0.0)) == (0.0, -1.0, 0.0)
    assert enu_to_ned((0.0, 0.0, 1.0)) == (0.0, 0.0, -1.0)


def test_goto_mask_keeps_position_velocity_and_yaw() -> None:
    motion = MotionTarget(
        position_ned=(1.0, 2.0, -1.5),
        velocity_ned=(0.0, 0.0, 0.0),
        acceleration_ned=None,
        yaw_rad=0.0,
        yaw_rate_rad_s=None,
    )
    mask = mask_for_target(motion)
    assert mask & TYPE_MASK_ACCELERATION_IGNORE
    assert mask & TYPE_MASK_YAW_RATE_IGNORE
    assert not (mask & TYPE_MASK_POSITION_IGNORE)
    assert not (mask & TYPE_MASK_YAW_IGNORE)


def test_spin_uses_yaw_rate_not_angle() -> None:
    adapter = _FakeAdapter()
    vehicle = Vehicle(adapter)
    result = vehicle.spin(math.pi / 2)
    assert result["ok"] is True
    assert adapter._session.setpoints
    last = adapter._session.setpoints[-1]
    assert last.frame is Frame.ODOM
    assert last.target.yaw_rad is None
    assert last.target.yaw_rate_rad_s == SPIN_RATE_RAD_S
    # Position is held during spin so XY control stays closed-loop.
    assert last.target.position_ned == (0.0, 0.0, -1.5)
    assert not (last.type_mask & TYPE_MASK_POSITION_IGNORE)
    assert last.type_mask & TYPE_MASK_YAW_IGNORE
    assert not (last.type_mask & TYPE_MASK_YAW_RATE_IGNORE)


def test_reverse_spin_uses_negative_yaw_rate() -> None:
    """Reface after +π must unwind via rate, not absolute yaw=0 (tip-strike)."""
    adapter = _FakeAdapter()
    vehicle = Vehicle(adapter)
    result = vehicle.spin(-math.pi)
    assert result["ok"] is True
    last = adapter._session.setpoints[-1]
    assert last.target.yaw_rad is None
    assert last.target.yaw_rate_rad_s == -SPIN_RATE_RAD_S
    assert last.target.position_ned == (0.0, 0.0, -1.5)


def test_publish_refused_when_not_guided() -> None:
    adapter = _FakeAdapter(armed=False, guided=True)
    vehicle = Vehicle(adapter)
    published = vehicle.publish(
        LocalNedTarget(
            position_ned=(1.0, 0.0, -1.5),
            velocity_ned=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            deadline_s=1.0,
            certificate_ref=None,
        )
    )
    assert published is None
    assert adapter.refusals
    assert adapter._session.setpoints == []


def test_refresh_period_is_named_const() -> None:
    assert REFRESH_S == 0.05


def test_wrap_angle_rad() -> None:
    assert abs(wrap_angle_rad(math.pi + 0.1) + (math.pi - 0.1)) < 1e-9
