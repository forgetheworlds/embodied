"""Motion-only checks for the bottom layer (:class:`embodied.platform.vehicle.Vehicle`).

These tests use a scripted MAVLink session that moves when it receives guided
setpoints. They confirm the vehicle layer commands motion correctly — not the
full compat probe or mission stack.

Run in simulation (Webots + SITL required) after unit tests pass::

    .venv/bin/python -m embodied compat \\
        --config configs/first_indoor.yaml \\
        --output work/runs/motion-proof-1

A passing compat run with green ``3_guided_local_ned_motion`` is the sim proof
that the aircraft holds waypoints within a few centimetres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied.contracts.records import (
    ClockStamp,
    MotionTarget,
    TYPE_MASK_ACCELERATION_IGNORE,
    TYPE_MASK_POSITION_IGNORE,
    TYPE_MASK_YAW_IGNORE,
    TYPE_MASK_YAW_RATE_IGNORE,
)
from embodied.platform import webots_ardupilot as W
from embodied.platform.vehicle import REFRESH_S, LocalNedTarget, Vehicle, mask_for_target


class FakeClock:
    def __init__(self) -> None:
        self._t = 100.0

    def monotonic(self) -> float:
        return self._t

    def monotonic_ns(self) -> int:
        return int(self._t * 1e9)

    def sleep(self, seconds: float) -> None:
        self._t += seconds


@dataclass
class MovingSession:
    """Autopilot stub: steps toward each position setpoint."""

    clock: FakeClock
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    mode: str = "GUIDED"
    armed: bool = True
    sent: list = field(default_factory=list)

    def send_setpoint(self, setpoint) -> None:
        self.sent.append(setpoint)
        target = setpoint.target
        if target.position_ned is not None and not (
            setpoint.type_mask & TYPE_MASK_POSITION_IGNORE
        ):
            step = 0.4
            self.position = tuple(
                current + step * (goal - current)
                for current, goal in zip(self.position, target.position_ned)
            )
        if target.yaw_rate_rad_s is not None and not (
            setpoint.type_mask & TYPE_MASK_YAW_RATE_IGNORE
        ):
            self.yaw += target.yaw_rate_rad_s * Vehicle.REFRESH_S

    def set_mode(self, mode_name: str) -> None:
        self.mode = mode_name

    def drain(self) -> list:
        return [
            {
                "mavpackettype": "HEARTBEAT",
                "custom_mode": 4 if self.mode == "GUIDED" else 5,
                "base_mode": 209,
                "system_status": W.MAV_STATE_ACTIVE,
            },
            {
                "mavpackettype": "ATTITUDE",
                "time_boot_ms": int(self.clock.monotonic() * 1000),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": self.yaw,
            },
            {
                "mavpackettype": "LOCAL_POSITION_NED",
                "time_boot_ms": int(self.clock.monotonic() * 1000),
                "x": self.position[0],
                "y": self.position[1],
                "z": self.position[2],
            },
        ]


class FakeEvidence:
    def write_json(self, *_args, **_kwargs) -> None:
        return None

    def append_jsonl(self, *_args, **_kwargs) -> None:
        return None


class FakeAdapter:
    def __init__(self, session: MovingSession, clock: FakeClock) -> None:
        self._session = session
        self._sequence = 0
        self._telemetry = None
        self._statustexts: list[str] = []
        self._monotonic = clock.monotonic
        self._monotonic_ns = clock.monotonic_ns
        self._sleep = clock.sleep
        self.settings = type(
            "Settings",
            (),
            {
                "hover_altitude_m": 1.5,
                "pre_arm_wait_s": 1.0,
                "host_id": "motion-test",
                "clock_id": "monotonic",
            },
        )()
        self.navigation_epoch = "nav-test"
        self.refusals = []
        self._publications = []
        self.evidence = FakeEvidence()

    def _stamp(self) -> ClockStamp:
        return ClockStamp(
            host_id=self.settings.host_id,
            clock_id=self.settings.clock_id,
            monotonic_ns=self._monotonic_ns(),
        )

    def telemetry(self):
        messages = self._session.drain()
        self._telemetry = W.decode_telemetry(
            messages,
            stamp=self._stamp(),
            previous=self._telemetry,
        )
        return self._telemetry

    @property
    def latest_telemetry(self):
        return self._telemetry

    @property
    def guidance_held(self) -> bool:
        sample = self._telemetry
        return bool(sample is not None and sample.in_guided_mode and sample.armed)


def test_publish_refuses_when_not_in_guided():
    clock = FakeClock()
    session = MovingSession(clock, mode="LOITER", armed=True)
    adapter = FakeAdapter(session, clock)
    vehicle = Vehicle(adapter)
    adapter.telemetry()
    result = vehicle.publish(
        LocalNedTarget(
            position_ned=(2.0, 0.0, -1.5),
            velocity_ned=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            deadline_s=1.0,
            certificate_ref=None,
        )
    )
    assert result is None
    assert session.sent == []
    assert adapter.refusals


def test_goto_moves_toward_target_and_holds():
    clock = FakeClock()
    session = MovingSession(clock)
    adapter = FakeAdapter(session, clock)
    vehicle = Vehicle(adapter)
    adapter.telemetry()
    outcome = vehicle.goto(2.0, 0.0, down_m=-1.5, hold_s=0.3, yaw_rad=0.0)
    assert outcome["ok"] is True
    assert outcome["publications"] >= 2
    assert outcome["residual_m"] is not None
    assert outcome["residual_m"] < 0.5
    assert session.position[0] > 0.5


def test_spin_commands_yaw_rate_not_position():
    clock = FakeClock()
    session = MovingSession(clock)
    adapter = FakeAdapter(session, clock)
    vehicle = Vehicle(adapter)
    adapter.telemetry()
    start_yaw = session.yaw
    outcome = vehicle.spin(0.5, rate_rad_s=0.6)
    assert outcome["ok"] is True
    assert session.sent
    last = session.sent[-1]
    assert last.target.yaw_rate_rad_s is not None
    assert last.type_mask & TYPE_MASK_YAW_RATE_IGNORE == 0
    assert last.type_mask & TYPE_MASK_POSITION_IGNORE
    assert session.yaw != start_yaw


def test_setpoint_mask_matches_fields():
    target = MotionTarget(
        position_ned=(1.0, 0.0, -1.5),
        velocity_ned=(0.0, 0.0, 0.0),
        acceleration_ned=None,
        yaw_rad=0.0,
        yaw_rate_rad_s=None,
    )
    mask = mask_for_target(target)
    assert mask & TYPE_MASK_ACCELERATION_IGNORE
    assert not (mask & TYPE_MASK_POSITION_IGNORE)
    assert not (mask & TYPE_MASK_YAW_IGNORE)
