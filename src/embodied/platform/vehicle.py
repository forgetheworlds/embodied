"""Bottom layer: ArduPilot GUIDED motion on a live MAVLink link.

One module owns frame conversion, setpoint records, publishing, arming, takeoff,
goto, hold, spin, and land. The platform adapter starts processes and sensor I/O;
everything that moves the aircraft goes through :class:`Vehicle`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Sequence

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
# Frames
# ---------------------------------------------------------------------------


class FrameError(ValueError):
    pass


def enu_to_ned(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise FrameError("a frame conversion takes three components")
    return (float(values[0]), -float(values[1]), -float(values[2]))


def wrap_angle_rad(angle: float) -> float:
    return math.remainder(angle, math.tau)


# ---------------------------------------------------------------------------
# Wire types
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
# Vehicle
# ---------------------------------------------------------------------------

ARM_SETTLE_S = 1.0
CONTROL_RETRY_S = 5.0
CONTROL_GRANT_GRACE_S = 3.0
REFRESH_S = 0.05
SPIN_RATE_RAD_S = 0.6
# After arm, Copter holds the motor interlock down for ARMING_DELAY_SEC (2.0 s)
# then MOT_IDLE_SEC ground-idle (compat_arming.parm: 4.0). NAV_TAKEOFF inside
# that window returns MAV_RESULT_FAILED (motion-proof-2: cmd 22 result 4 at
# arm+5.6 s while servos still at 1100 us). Cover both intervals plus margin.
POST_ARM_SPOOL_S = 7.0
TAKEOFF_ATTEMPTS = 3
TAKEOFF_RETRY_S = 2.0


class Vehicle:
    """Move the aircraft: the only layer that commands ArduPilot guided motion."""

    def __init__(self, adapter: WebotsArduPilot) -> None:
        self._adapter = adapter

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
                if drain is not None:
                    drain()
                adapter._sleep(0.1)
            sample = adapter.telemetry()
            for text in sample.statustexts:
                if text.startswith("PreArm:") or text.startswith("Arm:"):
                    refusals[text] = text
            if sample.armed or adapter._monotonic() >= deadline:
                break
            retry_until = adapter._monotonic() + CONTROL_RETRY_S
            while adapter._monotonic() < retry_until and adapter._monotonic() < deadline:
                if drain is not None:
                    drain()
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

    def takeoff(
        self,
        timeout_s: float,
        *,
        drain: Callable[[], None] | None = None,
    ) -> AutopilotControlEvidence:
        adapter = self._adapter
        evidence, attempt_at = self.request_control(
            adapter.settings.pre_arm_wait_s, drain=drain
        )
        altitude = None
        sample = adapter.latest_telemetry
        if evidence.armed:
            spool_until = adapter._monotonic() + POST_ARM_SPOOL_S
            while adapter._monotonic() < spool_until:
                if drain is not None:
                    drain()
                adapter._sleep(0.1)
            takeoff_at = adapter._monotonic()
            deadline = takeoff_at + timeout_s
            next_command_at = takeoff_at
            takeoff_attempts = 0
            while adapter._monotonic() < deadline:
                if (
                    takeoff_attempts < TAKEOFF_ATTEMPTS
                    and adapter._monotonic() >= next_command_at
                ):
                    adapter._session.takeoff(adapter.settings.hover_altitude_m)
                    takeoff_attempts += 1
                    next_command_at = adapter._monotonic() + TAKEOFF_RETRY_S
                sample = adapter.telemetry()
                position = sample.local_position_ned
                if position is not None:
                    altitude = -position[2]
                    if (
                        sample.in_guided_mode
                        and sample.armed
                        and altitude >= 0.5 * adapter.settings.hover_altitude_m
                    ):
                        break
                if adapter._monotonic() - takeoff_at >= CONTROL_GRANT_GRACE_S and not (
                    sample.in_guided_mode and sample.armed
                ):
                    break
                if drain is not None:
                    drain()
                adapter._sleep(0.2)
            evidence = replace(
                evidence,
                mode_reached=sample.in_guided_mode,
                armed=bool(sample.armed),
                altitude_m=altitude,
                refused=not (
                    sample.in_guided_mode
                    and sample.armed
                    and altitude is not None
                    and altitude >= 0.5 * adapter.settings.hover_altitude_m
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

    def _refresh(
        self,
        target: LocalNedTarget,
        duration_s: float,
        *,
        drain: Callable[[], None] | None = None,
        refresh_s: float = REFRESH_S,
    ) -> dict[str, Any]:
        adapter = self._adapter
        until = adapter._monotonic() + duration_s
        publications = 0
        while adapter._monotonic() < until:
            if self.publish(target) is None:
                return {"ok": False, "publications": publications, "guided_lost": True}
            publications += 1
            if drain is not None:
                drain()
            adapter._sleep(refresh_s)
        sample = adapter.latest_telemetry
        after = None if sample is None else sample.local_position_ned
        residual = None
        if target.position_ned is not None and after is not None:
            residual = math.dist(after, target.position_ned)
        return {
            "ok": True,
            "publications": publications,
            "guided_lost": False,
            "residual_m": residual,
            "position_after_ned": after,
        }

    def goto(
        self,
        north_m: float,
        east_m: float,
        *,
        down_m: float | None = None,
        hold_s: float = 8.0,
        yaw_rad: float = 0.0,
        drain: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        hover = self._adapter.settings.hover_altitude_m
        if down_m is None:
            down_m = -hover
        target = LocalNedTarget(
            position_ned=(north_m, east_m, down_m),
            velocity_ned=(0.0, 0.0, 0.0),
            yaw_rad=yaw_rad,
            deadline_s=hold_s,
            certificate_ref=None,
        )
        result = self._refresh(target, hold_s, drain=drain)
        result["target_ned"] = target.position_ned
        return result

    def hold(
        self,
        duration_s: float,
        *,
        yaw_rad: float = 0.0,
        drain: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        sample = self._adapter.latest_telemetry
        position = None if sample is None else sample.local_position_ned
        if position is None:
            position = (0.0, 0.0, -self._adapter.settings.hover_altitude_m)
        target = LocalNedTarget(
            position_ned=position,
            velocity_ned=(0.0, 0.0, 0.0),
            yaw_rad=yaw_rad,
            deadline_s=duration_s,
            certificate_ref=None,
        )
        return self._refresh(target, duration_s, drain=drain)

    def spin(
        self,
        angle_rad: float,
        *,
        rate_rad_s: float = SPIN_RATE_RAD_S,
        drain: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if rate_rad_s == 0.0:
            raise ValueError("spin rate must be non-zero")
        sample = self._adapter.latest_telemetry
        start_yaw = None if sample is None or sample.attitude_rpy is None else sample.attitude_rpy[2]
        duration = abs(angle_rad) / abs(rate_rad_s)
        signed_rate = rate_rad_s if angle_rad >= 0.0 else -rate_rad_s
        target = LocalNedTarget(
            position_ned=None,
            velocity_ned=(0.0, 0.0, 0.0),
            yaw_rad=None,
            yaw_rate_rad_s=signed_rate,
            deadline_s=duration,
            certificate_ref=None,
        )
        result = self._refresh(target, duration, drain=drain)
        sample = self._adapter.latest_telemetry
        end_yaw = None if sample is None or sample.attitude_rpy is None else sample.attitude_rpy[2]
        delta = None
        if start_yaw is not None and end_yaw is not None:
            delta = wrap_angle_rad(end_yaw - start_yaw)
        result["requested_rad"] = angle_rad
        result["delta_rad"] = delta
        return result

    def land(self) -> None:
        self._adapter._session.set_mode("LAND")
