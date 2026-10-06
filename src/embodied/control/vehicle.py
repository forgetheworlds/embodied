"""Control layer: Vehicle API v1 (GUIDED motion on a live MAVLink link).

Public surface::

    Vehicle.takeoff(altitude_m) -> Result
    Vehicle.command(Motion) -> Result
    Vehicle.land() -> Result
    Vehicle.state() -> VehicleState

``Motion`` is odom-only (ENU: x north, y west, z up). Vehicle converts
odom→NED→MAVLink. Each ``command()`` is one shot — the caller owns refresh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Sequence

from embodied.contracts.records import (
    ClockStamp,
    Frame,
    MotionSetpoint,
    MotionTarget,
    SetpointSource,
    TYPE_MASK_ACCELERATION_IGNORE,
    TYPE_MASK_POSITION_IGNORE,
    TYPE_MASK_VELOCITY_IGNORE,
    TYPE_MASK_YAW_IGNORE,
    TYPE_MASK_YAW_RATE_IGNORE,
)

if TYPE_CHECKING:
    from embodied.platform.webots_ardupilot import WebotsArduPilot


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))


@dataclass(frozen=True)
class Motion:
    """One odom motion sample. Hold = zero velocity; turns add ``yaw_rate``."""

    position: Vec3
    velocity: Vec3
    yaw: float | None = None
    yaw_rate: float | None = None


@dataclass(frozen=True)
class Result:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class VehicleState:
    armed: bool
    guided: bool
    position: Vec3 | None
    velocity: Vec3 | None
    yaw: float | None
    landed: bool | None = None


# ---------------------------------------------------------------------------
# Frames / masks (Vehicle-owned odom↔NED)
# ---------------------------------------------------------------------------


class FrameError(ValueError):
    pass


def enu_to_ned(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise FrameError("a frame conversion takes three components")
    return (float(values[0]), -float(values[1]), -float(values[2]))


def ned_to_enu(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise FrameError("a frame conversion takes three components")
    return (float(values[0]), -float(values[1]), -float(values[2]))


def wrap_angle_rad(angle: float) -> float:
    return math.remainder(angle, math.tau)


def _finite3(values: Sequence[float], label: str) -> None:
    if len(values) != 3:
        raise FrameError(f"{label} needs three components")
    for value in values:
        if not math.isfinite(float(value)):
            raise FrameError(f"{label} has a non-finite component")


def mask_for_target(target: MotionTarget) -> int:
    mask = 0
    if target.position_ned is None:
        mask |= TYPE_MASK_POSITION_IGNORE
    if target.velocity_ned is None:
        mask |= TYPE_MASK_VELOCITY_IGNORE
    if target.acceleration_ned is None:
        mask |= TYPE_MASK_ACCELERATION_IGNORE
    if target.yaw_rad is None:
        mask |= TYPE_MASK_YAW_IGNORE
    if target.yaw_rate_rad_s is None:
        mask |= TYPE_MASK_YAW_RATE_IGNORE
    return mask


# ---------------------------------------------------------------------------
# Adapter glue kept for platform probes (not Vehicle API v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalNedTarget:
    position_ned: tuple[float, float, float] | None
    velocity_ned: tuple[float, float, float] | None
    yaw_rad: float | None
    deadline_s: float
    certificate_ref: str | None
    yaw_rate_rad_s: float | None = None


@dataclass(frozen=True)
class SetpointPublication:
    setpoint: MotionSetpoint
    published_stamp: ClockStamp
    adoption: str = "publication recorded; adoption is judged from telemetry, not here"


@dataclass(frozen=True)
class AutopilotControlEvidence:
    commanded_mode: str
    mode_reached: bool
    armed: bool
    takeoff_commanded_m: float
    altitude_m: float | None
    statustexts: tuple[str, ...]
    refused: bool
    control_attempts: int
    refusals: tuple[str, ...]


ARM_SETTLE_S = 1.0
CONTROL_RETRY_S = 5.0
CONTROL_GRANT_GRACE_S = 3.0
# Validity of one command() sample on the wire. Caller must refresh sooner.
COMMAND_DEADLINE_S = 0.25
TAKEOFF_ATTEMPTS = 5
TAKEOFF_RETRY_SIM_S = 1.5


def _sim_time_s(adapter: WebotsArduPilot) -> float | None:
    sample = adapter.latest_telemetry
    if sample is None or sample.boot_time_ms is None:
        return None
    return sample.boot_time_ms / 1000.0


def _wall_backstop_s(adapter: WebotsArduPilot, sim_duration_s: float) -> float:
    min_ratio = min(adapter.settings.realtime_ratio_envelope)
    return sim_duration_s / min_ratio


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------


class Vehicle:
    """Public GUIDED motion API. Caller owns the command refresh loop."""

    def __init__(self, adapter: WebotsArduPilot) -> None:
        self._adapter = adapter
        self._last_takeoff: AutopilotControlEvidence | None = None

    # -- public ------------------------------------------------------------

    def takeoff(self, altitude_m: float) -> Result:
        """Arm, enter GUIDED, wait for EKF origin, climb to ``altitude_m``."""
        if not math.isfinite(altitude_m) or altitude_m <= 0.0:
            return Result(
                accepted=False,
                reason="takeoff altitude must be a positive finite metres",
            )
        evidence = self._bring_up(altitude_m)
        self._last_takeoff = evidence
        if evidence.refused:
            reason = evidence.refusals[0] if evidence.refusals else "takeoff refused"
            return Result(accepted=False, reason=reason)
        return Result(accepted=True)

    def command(self, motion: Motion) -> Result:
        """Publish one odom motion sample. Does not republish or auto-hold."""
        try:
            self._validate_motion(motion)
        except FrameError as error:
            return Result(accepted=False, reason=str(error))
        if motion.yaw is not None and motion.yaw_rate is not None:
            return Result(
                accepted=False,
                reason="request either a heading or a heading rate, not both",
            )
        publication = self.publish(
            LocalNedTarget(
                position_ned=enu_to_ned(motion.position.as_tuple()),
                velocity_ned=enu_to_ned(motion.velocity.as_tuple()),
                yaw_rad=motion.yaw,
                yaw_rate_rad_s=motion.yaw_rate,
                deadline_s=COMMAND_DEADLINE_S,
                certificate_ref=None,
            )
        )
        if publication is None:
            return Result(accepted=False, reason="not in armed Guided flight")
        return Result(accepted=True)

    def land(self) -> Result:
        self._adapter._session.set_mode("LAND")
        return Result(accepted=True)

    def state(self) -> VehicleState:
        sample = self._adapter.telemetry()
        position = None
        velocity = None
        yaw = None
        landed: bool | None = None
        if sample is not None:
            if sample.local_position_ned is not None:
                position = Vec3(*ned_to_enu(sample.local_position_ned))
            if sample.velocity_ned is not None:
                velocity = Vec3(*ned_to_enu(sample.velocity_ned))
            if sample.attitude_rpy is not None:
                yaw = float(sample.attitude_rpy[2])
            if position is not None and not sample.armed:
                landed = position.z < 0.35
            elif sample.mode_name == "LAND" and position is not None:
                landed = position.z < 0.35
        return VehicleState(
            armed=bool(sample.armed) if sample is not None else False,
            guided=bool(sample.in_guided_mode) if sample is not None else False,
            position=position,
            velocity=velocity,
            yaw=yaw,
            landed=landed,
        )

    # -- adapter glue (platform probes; not public API v1) -----------------

    def request_control(
        self,
        timeout_s: float,
        *,
        drain: Callable[[], None] | None = None,
    ) -> tuple[AutopilotControlEvidence, ClockStamp]:
        adapter = self._adapter
        deadline = adapter._monotonic() + timeout_s
        refusals: dict[str, str] = {}
        attempts = 0
        attempt_at = adapter._monotonic()
        sample = adapter.telemetry()
        while True:
            attempts += 1
            attempt_at = adapter._monotonic()
            adapter._session.set_mode("GUIDED")
            adapter._session.arm()
            settle_until = adapter._monotonic() + ARM_SETTLE_S
            while adapter._monotonic() < settle_until:
                self._pump(drain)
                adapter._sleep(0.1)
            sample = adapter.telemetry()
            for text in sample.statustexts:
                if text.startswith("PreArm:") or text.startswith("Arm:"):
                    refusals[text] = text
            if sample.armed or adapter._monotonic() >= deadline:
                break
            retry_until = adapter._monotonic() + CONTROL_RETRY_S
            while adapter._monotonic() < retry_until and adapter._monotonic() < deadline:
                self._pump(drain)
                adapter._sleep(0.5)
        evidence = AutopilotControlEvidence(
            commanded_mode="GUIDED",
            mode_reached=sample.in_guided_mode,
            armed=bool(sample.armed),
            takeoff_commanded_m=adapter.settings.hover_altitude_m,
            altitude_m=None,
            statustexts=tuple(adapter._statustexts),
            refused=not (sample.in_guided_mode and sample.armed),
            control_attempts=attempts,
            refusals=tuple(sorted(refusals)),
        )
        stamp = ClockStamp(
            host_id=adapter.settings.host_id,
            clock_id=adapter.settings.clock_id,
            monotonic_ns=int(attempt_at * 1e9),
        )
        return evidence, stamp

    def publish(self, target: LocalNedTarget) -> SetpointPublication | None:
        adapter = self._adapter
        adapter.telemetry()
        if not adapter.guidance_held:
            sample = adapter._telemetry
            refusal = {
                "at_monotonic_ns": int(adapter._monotonic_ns()),
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
            adapter.refusals.append(refusal)
            adapter.evidence.append_jsonl("refused-publications.jsonl", refusal)
            return None
        motion = MotionTarget(
            position_ned=target.position_ned,
            velocity_ned=target.velocity_ned,
            acceleration_ned=None,
            yaw_rad=target.yaw_rad,
            yaw_rate_rad_s=target.yaw_rate_rad_s,
        )
        adapter._sequence += 1
        setpoint = MotionSetpoint(
            command_sequence=adapter._sequence,
            mission_revision=0,
            goal_revision=0,
            nav_epoch=adapter.navigation_epoch,
            frame=Frame.ODOM,
            type_mask=mask_for_target(motion),
            target=motion,
            issue_stamp=adapter._stamp(),
            deadline_s=target.deadline_s,
            certificate_ref=target.certificate_ref,
            sample_ref=None,
            source=SetpointSource.NORMAL,
        )
        adapter._session.send_setpoint(setpoint)
        publication = SetpointPublication(
            setpoint=setpoint, published_stamp=adapter._stamp()
        )
        adapter._publications.append(publication)
        return publication

    # -- internals ---------------------------------------------------------

    def _validate_motion(self, motion: Motion) -> None:
        if not isinstance(motion, Motion):
            raise FrameError("command expects a Motion")
        _finite3(motion.position.as_tuple(), "position")
        _finite3(motion.velocity.as_tuple(), "velocity")
        if motion.yaw is not None and not math.isfinite(motion.yaw):
            raise FrameError("yaw is not finite")
        if motion.yaw_rate is not None and not math.isfinite(motion.yaw_rate):
            raise FrameError("yaw_rate is not finite")

    def _pump(self, drain: Callable[[], None] | None = None) -> None:
        if drain is not None:
            drain()
            return
        adapter = self._adapter
        sensor_record = getattr(adapter, "sensor_record", None)
        if sensor_record is None:
            return
        record = sensor_record(0.02)
        while record is not None:
            record = sensor_record(0.0)

    def _bring_up(self, altitude_m: float) -> AutopilotControlEvidence:
        adapter = self._adapter
        evidence, attempt_at = self.request_control(adapter.settings.pre_arm_wait_s)
        altitude = None
        sample = adapter.latest_telemetry
        if evidence.armed:
            origin_start_sim: float | None = None
            origin_wall_until = adapter._monotonic() + _wall_backstop_s(
                adapter, adapter.settings.pre_arm_wait_s
            )
            while (
                sample.in_guided_mode
                and sample.armed
                and adapter._monotonic() < origin_wall_until
            ):
                sample = adapter.telemetry()
                if sample.home_position is not None:
                    break
                sim = _sim_time_s(adapter)
                if sim is not None:
                    if origin_start_sim is None:
                        origin_start_sim = sim
                    elif sim - origin_start_sim >= adapter.settings.pre_arm_wait_s:
                        break
                self._pump()
                adapter._sleep(0.1)
            climb_start_sim: float | None = None
            next_takeoff_sim: float | None = None
            takeoff_attempts = 0
            climb_timeout = adapter.settings.step_timeout_s.flight
            climb_wall_until = adapter._monotonic() + _wall_backstop_s(
                adapter, climb_timeout
            )
            while (
                sample.in_guided_mode
                and sample.armed
                and adapter._monotonic() < climb_wall_until
            ):
                sim = _sim_time_s(adapter)
                if (
                    takeoff_attempts < TAKEOFF_ATTEMPTS
                    and sample.home_position is not None
                    and (
                        next_takeoff_sim is None
                        or (sim is not None and sim >= next_takeoff_sim)
                    )
                ):
                    adapter._session.takeoff(altitude_m)
                    takeoff_attempts += 1
                    if sim is not None:
                        if climb_start_sim is None:
                            climb_start_sim = sim
                        next_takeoff_sim = sim + TAKEOFF_RETRY_SIM_S
                sample = adapter.telemetry()
                position = sample.local_position_ned
                if position is not None:
                    altitude = -position[2]
                    if altitude >= 0.5 * altitude_m:
                        break
                if (
                    climb_start_sim is not None
                    and sim is not None
                    and sim - climb_start_sim >= climb_timeout
                ):
                    break
                self._pump()
                adapter._sleep(0.2)
            evidence = replace(
                evidence,
                takeoff_commanded_m=altitude_m,
                mode_reached=sample.in_guided_mode,
                armed=bool(sample.armed),
                altitude_m=altitude,
                refused=not (
                    sample.in_guided_mode
                    and sample.armed
                    and altitude is not None
                    and altitude >= 0.5 * altitude_m
                ),
            )
        adapter.evidence.write_json(
            "flight-state.json",
            {
                "commanded_mode": evidence.commanded_mode,
                "mode_reached": evidence.mode_reached,
                "armed": evidence.armed,
                "takeoff_commanded_m": evidence.takeoff_commanded_m,
                "altitude_m": evidence.altitude_m,
                "control_attempts": evidence.control_attempts,
                "refusals": list(evidence.refusals),
                "pre_arm_wait_s": adapter.settings.pre_arm_wait_s,
                "control_requested_at_monotonic_ns": attempt_at.monotonic_ns,
                "statustexts": list(evidence.statustexts),
                "refused": evidence.refused,
            },
        )
        return evidence
