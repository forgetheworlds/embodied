"""Live control-layer proof: caller-owned Vehicle API v1 command loop.

Route (public methods only):

1. takeoff(altitude)
2. stream hold Motions at each outbound odom waypoint
3. stream hold at the far end
4. stream yaw_rate turn (+π) then reverse (−π)
5. stream return waypoints
6. land()

Run::

    ./configs/layers/control
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Callable

from embodied.cli import (
    CommandOutcome,
    CommandStatus,
    GateStatus,
    load_config,
    register_command,
    repository_root,
)
from embodied.contracts.records import SensorMode
from embodied.control import Motion, Vehicle, Vec3, wrap_angle_rad
from embodied.control.vehicle import ned_to_enu
from embodied.platform.webots_ardupilot import (
    EvidenceWriter,
    PlatformSettings,
    PlatformUnavailable,
    ProbeFailure,
    PymavlinkSession,
    SubprocessRunner,
    TcpSensorGateway,
    WebotsArduPilot,
    check_prerequisites,
)

DEFAULT_SPIN_RAD = math.pi
DEFAULT_SPIN_TOLERANCE_RAD = 0.40
DEFAULT_RESIDUAL_MAX_M = 0.15
REFRESH_S = 0.05
SPIN_RATE_RAD_S = 0.6
ZERO = Vec3(0.0, 0.0, 0.0)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def _drain(platform: WebotsArduPilot) -> None:
    record = platform.sensor_record(0.02)
    while record is not None:
        record = platform.sensor_record(0.0)


def _motion_section(document: dict[str, Any]) -> dict[str, Any]:
    motion = document.get("motion")
    probe = document["probe"]
    if motion is not None:
        return motion
    return {
        "hover_altitude_m": probe["hover_altitude_m"],
        "hold_per_waypoint_s": probe["hold_per_waypoint_s"],
        "residual_max_m": DEFAULT_RESIDUAL_MAX_M,
        "waypoints_local_ned": probe["waypoints_local_ned"],
    }


def _sim_time_s(platform: WebotsArduPilot) -> float | None:
    sample = platform.latest_telemetry
    if sample is None or sample.boot_time_ms is None:
        return None
    return sample.boot_time_ms / 1000.0


def _wall_backstop_s(platform: WebotsArduPilot, sim_duration_s: float) -> float:
    min_ratio = min(platform.settings.realtime_ratio_envelope)
    return sim_duration_s / min_ratio


def _ned_waypoint_to_odom(
    north: float, east: float, z_offset: float, hover_m: float
) -> Vec3:
    """Config stores absolute local-NED holds; Motion is odom ENU."""
    down = -(hover_m - z_offset)
    return Vec3(*ned_to_enu((north, east, down)))


def _stream(
    vehicle: Vehicle,
    platform: WebotsArduPilot,
    *,
    build_motion: Callable[[], Motion],
    duration_s: float,
    drain: Callable[[], None],
) -> dict[str, Any]:
    """Caller-owned refresh: one command() per tick for ``duration_s`` sim time."""
    start_sim: float | None = None
    wall_until = platform._monotonic() + _wall_backstop_s(platform, duration_s)
    publications = 0
    while platform._monotonic() < wall_until:
        sim = _sim_time_s(platform)
        if sim is not None:
            if start_sim is None:
                start_sim = sim
            elif sim - start_sim >= duration_s:
                break
        result = vehicle.command(build_motion())
        if not result.accepted:
            return {
                "ok": False,
                "publications": publications,
                "guided_lost": True,
                "reason": result.reason,
            }
        publications += 1
        drain()
        platform._sleep(REFRESH_S)
    return {"ok": True, "publications": publications, "guided_lost": False}


def _score_hold(
    steps: list[dict[str, Any]],
    reasons: list[str],
    *,
    label: str,
    target_odom: Vec3,
    vehicle: Vehicle,
    stream: dict[str, Any],
    residual_max_m: float,
) -> None:
    state = vehicle.state()
    residual = None
    if state.position is not None:
        residual = math.dist(state.position.as_tuple(), target_odom.as_tuple())
    ok = (
        stream.get("ok") is True
        and residual is not None
        and residual <= residual_max_m
    )
    steps.append(
        {
            "task": label,
            "target_odom": list(target_odom.as_tuple()),
            "residual_m": residual,
            "publications": stream.get("publications"),
            "ok": ok,
        }
    )
    if stream.get("guided_lost"):
        reasons.append(f"{label}: guided flight lost")
    elif residual is None:
        reasons.append(f"{label}: no position to score")
    elif residual > residual_max_m:
        reasons.append(
            f"{label}: residual {residual:.3f} m > {residual_max_m:.3f} m"
        )


def fly_control_route(
    platform: WebotsArduPilot,
    *,
    waypoints: tuple[tuple[float, float, float], ...],
    hold_s: float,
    hover_m: float,
    residual_max_m: float,
    spin_rad: float = DEFAULT_SPIN_RAD,
    spin_tolerance_rad: float = DEFAULT_SPIN_TOLERANCE_RAD,
    return_waypoints: bool = True,
) -> dict[str, Any]:
    """Fly the layer route through Vehicle.takeoff / command / land / state."""
    vehicle = platform.vehicle
    drain: Callable[[], None] = lambda: _drain(platform)
    reasons: list[str] = []
    steps: list[dict[str, Any]] = []

    takeoff = vehicle.takeoff(hover_m)
    state = vehicle.state()
    steps.append(
        {
            "task": "takeoff",
            "ok": takeoff.accepted,
            "altitude_m": None if state.position is None else state.position.z,
            "reason": takeoff.reason,
        }
    )
    if not takeoff.accepted:
        reasons.append(takeoff.reason or "takeoff refused")
        return {
            "status": "fail",
            "reasons": reasons,
            "steps": steps,
            "residual_max_m": residual_max_m,
        }

    def hold_motion(target: Vec3, *, yaw: float | None = 0.0) -> Motion:
        return Motion(position=target, velocity=ZERO, yaw=yaw)

    for index, waypoint in enumerate(waypoints):
        target = _ned_waypoint_to_odom(*waypoint, hover_m=hover_m)
        stream = _stream(
            vehicle,
            platform,
            build_motion=lambda t=target: hold_motion(t, yaw=0.0),
            duration_s=hold_s,
            drain=drain,
        )
        _score_hold(
            steps,
            reasons,
            label=f"goto[{index}]",
            target_odom=target,
            vehicle=vehicle,
            stream=stream,
            residual_max_m=residual_max_m,
        )
        if stream.get("guided_lost"):
            break

    if not any("guided flight lost" in reason for reason in reasons):
        state = vehicle.state()
        if state.position is None:
            reasons.append("hold: no position")
            steps.append({"task": "hold", "ok": False})
        else:
            target = state.position
            stream = _stream(
                vehicle,
                platform,
                build_motion=lambda t=target: hold_motion(t, yaw=0.0),
                duration_s=hold_s,
                drain=drain,
            )
            _score_hold(
                steps,
                reasons,
                label="hold",
                target_odom=target,
                vehicle=vehicle,
                stream=stream,
                residual_max_m=residual_max_m,
            )

    def yaw_turn(angle_rad: float, label: str) -> None:
        if any("guided flight lost" in reason for reason in reasons):
            return
        state = vehicle.state()
        if state.position is None or state.yaw is None:
            reasons.append(f"{label}: missing pose")
            steps.append({"task": label, "ok": False})
            return
        start_yaw = state.yaw
        hold_pos = state.position
        rate = SPIN_RATE_RAD_S if angle_rad >= 0.0 else -SPIN_RATE_RAD_S
        duration = abs(angle_rad) / abs(SPIN_RATE_RAD_S)
        stream = _stream(
            vehicle,
            platform,
            build_motion=lambda p=hold_pos, r=rate: Motion(
                position=p, velocity=ZERO, yaw_rate=r
            ),
            duration_s=duration,
            drain=drain,
        )
        after = vehicle.state()
        delta = None
        if after.yaw is not None:
            delta = wrap_angle_rad(after.yaw - start_yaw)
        ok = (
            stream.get("ok") is True
            and delta is not None
            and abs(abs(delta) - abs(angle_rad)) <= spin_tolerance_rad
        )
        steps.append(
            {
                "task": label,
                "requested_rad": angle_rad,
                "delta_rad": delta,
                "publications": stream.get("publications"),
                "ok": ok,
            }
        )
        if stream.get("guided_lost"):
            reasons.append(f"{label}: guided flight lost")
        elif delta is None:
            reasons.append(f"{label}: no yaw delta to score")
        elif not ok:
            reasons.append(
                f"{label}: |delta| {abs(delta):.3f} rad vs requested {abs(angle_rad):.3f} "
                f"(tol {spin_tolerance_rad:.3f})"
            )

    yaw_turn(spin_rad, "spin")
    if return_waypoints:
        yaw_turn(-spin_rad, "reface")

    # After ±π: settle in place with yaw=0 (small heading fix, no translate),
    # then align east with the first inbound hold before moving north—diagonal
    # 7.5/-0.5 → 5.5/-1.2 after the spin pair tip-struck even with yaw≈0.
    if return_waypoints and not any(
        "guided flight lost" in reason for reason in reasons
    ):
        state = vehicle.state()
        if state.position is not None:
            settle_pos = state.position
            stream = _stream(
                vehicle,
                platform,
                build_motion=lambda p=settle_pos: hold_motion(p, yaw=0.0),
                duration_s=hold_s,
                drain=drain,
            )
            _score_hold(
                steps,
                reasons,
                label="settle",
                target_odom=settle_pos,
                vehicle=vehicle,
                stream=stream,
                residual_max_m=residual_max_m,
            )

    if return_waypoints and not any(
        "guided flight lost" in reason for reason in reasons
    ):
        inbound = tuple(reversed(waypoints[:-1])) if len(waypoints) > 1 else ()
        if inbound and waypoints:
            far_north, _far_east, far_z = waypoints[-1]
            first_north, first_east, first_z = inbound[0]
            if abs(first_east - _far_east) > 1e-6:
                align = _ned_waypoint_to_odom(
                    far_north, first_east, far_z, hover_m=hover_m
                )
                stream = _stream(
                    vehicle,
                    platform,
                    build_motion=lambda t=align: hold_motion(t, yaw=0.0),
                    duration_s=hold_s,
                    drain=drain,
                )
                _score_hold(
                    steps,
                    reasons,
                    label="align",
                    target_odom=align,
                    vehicle=vehicle,
                    stream=stream,
                    residual_max_m=residual_max_m,
                )
        for index, waypoint in enumerate(inbound):
            if any("guided flight lost" in reason for reason in reasons):
                break
            target = _ned_waypoint_to_odom(*waypoint, hover_m=hover_m)
            stream = _stream(
                vehicle,
                platform,
                build_motion=lambda t=target: hold_motion(t, yaw=0.0),
                duration_s=hold_s,
                drain=drain,
            )
            _score_hold(
                steps,
                reasons,
                label=f"return[{index}]",
                target_odom=target,
                vehicle=vehicle,
                stream=stream,
                residual_max_m=residual_max_m,
            )
            if stream.get("guided_lost"):
                break

    land = vehicle.land()
    steps.append({"task": "land", "ok": land.accepted})
    if not land.accepted:
        reasons.append(land.reason or "land refused")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        drain()
        time.sleep(0.05)

    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "steps": steps,
        "residual_max_m": residual_max_m,
    }


fly_motion_route = fly_control_route


def _command(args: argparse.Namespace, output_dir: Path) -> CommandOutcome:
    document = load_config(args.config)
    settings = PlatformSettings.from_config(
        document, root=repository_root(), arm=SensorMode.SIMULATOR_INTERFACE.value
    )
    missing = [
        item for item in check_prerequisites(settings, output_dir) if not item.satisfied
    ]
    if missing:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=tuple(f"{item.name}: {item.detail}" for item in missing),
        )

    motion = _motion_section(document)
    waypoints = tuple(
        (float(item[0]), float(item[1]), float(item[2]))
        for item in motion["waypoints_local_ned"]
    )
    hold_s = float(motion["hold_per_waypoint_s"])
    hover_m = float(motion["hover_altitude_m"])
    residual_max_m = float(motion["residual_max_m"])
    spin_rad = float(motion.get("spin_rad", DEFAULT_SPIN_RAD))
    spin_tolerance_rad = float(
        motion.get("spin_tolerance_rad", DEFAULT_SPIN_TOLERANCE_RAD)
    )
    return_waypoints = bool(motion.get("return_waypoints", True))

    writer = EvidenceWriter(output_dir, "run-a")
    platform = WebotsArduPilot(
        settings,
        runner=SubprocessRunner(),
        session=PymavlinkSession(),
        gateway=TcpSensorGateway(
            stamp=lambda: settings.capture_stamp(time.monotonic_ns())
        ),
        evidence=writer,
        label="motion-proof",
        extra_params=settings.compat_estimator_params,
    )
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.ready)
        result = fly_control_route(
            platform,
            waypoints=waypoints,
            hold_s=hold_s,
            hover_m=hover_m,
            residual_max_m=residual_max_m,
            spin_rad=spin_rad,
            spin_tolerance_rad=spin_tolerance_rad,
            return_waypoints=return_waypoints,
        )
        writer.write_json("motion-proof.json", result)
        gate = GateStatus.PASS if result["status"] == "pass" else GateStatus.FAIL
        return CommandOutcome(
            status=CommandStatus.COMPLETE,
            gate_status=gate,
            reasons=tuple(result["reasons"]),
            limitations=(
                "Control proof: Vehicle API v1 takeoff/command/land through "
                "doorways; no estimator or obstacle-avoidance checks",
            ),
            manifest={
                "command": "motion-proof",
                "world": str(settings.world),
                "waypoints": [list(item) for item in waypoints],
                "spin_rad": spin_rad,
                "return_waypoints": return_waypoints,
                "steps": result["steps"],
            },
            artifacts=("run-a/motion-proof.json",),
            sensor_mode=settings.sensor_mode,
        )
    except PlatformUnavailable as error:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(str(error),),
        )
    except ProbeFailure as error:
        return CommandOutcome(
            status=CommandStatus.INVALID,
            gate_status=GateStatus.FAIL,
            reasons=(str(error),),
        )
    finally:
        platform.stop()


register_command(
    "motion-proof",
    _command,
    help_text=(
        "Launch the simulator and fly the Vehicle API v1 control route "
        "(takeoff, command-loop holds/turns, land)."
    ),
    stage_id="P00-motion",
    run_prefix="p00-motion-proof",
    add_arguments=_add_arguments,
)
