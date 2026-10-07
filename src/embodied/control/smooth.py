"""Throwaway continuous smooth flight through the doorway polyline.

Not the layer gate. Complex command probe: sliding position + path velocity
feedforward + limited yaw_rate toward the path heading (not discrete
left/right/hold steps). Throwaway — delete when done exploring.

Run::

    ./configs/layers/control-smooth
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import yaml

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

ROUTE_PATH = Path(__file__).with_name("route.yaml")
REFRESH_S = 0.05
# Gentler than the AngErr=84 tip (was 0.35 m/s + snapped path yaw).
CRUISE_SPEED_M_S = 0.20
MAX_YAW_RATE_RAD_S = 0.25
YAW_RATE_GAIN = 1.0
END_HOLD_S = 2.0
ZERO = Vec3(0.0, 0.0, 0.0)


def _path_yaw_rad(tangent: Vec3) -> float:
    """NED yaw from an ENU unit tangent (x north, y west)."""
    return math.atan2(-tangent.y, tangent.x)


def _limited_yaw_rate(current_yaw: float | None, desired_yaw: float) -> float:
    if current_yaw is None:
        return 0.0
    error = wrap_angle_rad(desired_yaw - current_yaw)
    rate = YAW_RATE_GAIN * error
    return max(-MAX_YAW_RATE_RAD_S, min(MAX_YAW_RATE_RAD_S, rate))


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def _load_route() -> dict[str, Any]:
    return yaml.safe_load(ROUTE_PATH.read_text())


def _odom_polyline(waypoints_ned: list, hover_m: float) -> list[Vec3]:
    points: list[Vec3] = []
    for north, east, _down in waypoints_ned:
        x, y, _z = ned_to_enu((float(north), float(east), 0.0))
        points.append(Vec3(x, y, hover_m))
    return points


def _polyline_length(points: list[Vec3]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.dist(a.as_tuple(), b.as_tuple())
    return total


def _sample_polyline(
    points: list[Vec3], distance_m: float
) -> tuple[Vec3, Vec3]:
    """Return (position, unit tangent) at arc length ``distance_m`` along the path."""
    if len(points) == 1:
        return points[0], Vec3(1.0, 0.0, 0.0)
    remaining = max(0.0, distance_m)
    for a, b in zip(points, points[1:]):
        ax, ay, az = a.as_tuple()
        bx, by, bz = b.as_tuple()
        seg = math.dist((ax, ay, az), (bx, by, bz))
        if seg < 1e-9:
            continue
        if remaining <= seg:
            t = remaining / seg
            pos = Vec3(
                ax + t * (bx - ax),
                ay + t * (by - ay),
                az + t * (bz - az),
            )
            tangent = Vec3((bx - ax) / seg, (by - ay) / seg, (bz - az) / seg)
            return pos, tangent
        remaining -= seg
    # Past the end: sit on the last point, last segment tangent.
    a, b = points[-2], points[-1]
    ax, ay, az = a.as_tuple()
    bx, by, bz = b.as_tuple()
    seg = math.dist((ax, ay, az), (bx, by, bz)) or 1.0
    return points[-1], Vec3((bx - ax) / seg, (by - ay) / seg, (bz - az) / seg)


def _drain(platform: WebotsArduPilot) -> None:
    record = platform.sensor_record(0.02)
    while record is not None:
        record = platform.sensor_record(0.0)


def fly_smooth_route(
    platform: WebotsArduPilot,
    *,
    waypoints_ned: list,
    hover_m: float,
    speed_m_s: float = CRUISE_SPEED_M_S,
) -> dict[str, Any]:
    """Cruise with position + velocity FF + capped yaw_rate toward path heading."""
    vehicle: Vehicle = platform.vehicle
    steps: list[dict[str, Any]] = []
    reasons: list[str] = []

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
        return {"status": "fail", "reasons": reasons, "steps": steps}

    points = _odom_polyline(waypoints_ned, hover_m)
    path_m = _polyline_length(points)
    distance = 0.0
    publications = 0
    guided_lost = False
    max_abs_yaw_rate = 0.0

    while distance < path_m and not guided_lost:
        pos, tangent = _sample_polyline(points, distance)
        state = vehicle.state()
        desired_yaw = _path_yaw_rad(tangent)
        yaw_rate = _limited_yaw_rate(state.yaw, desired_yaw)
        max_abs_yaw_rate = max(max_abs_yaw_rate, abs(yaw_rate))
        velocity = Vec3(
            tangent.x * speed_m_s,
            tangent.y * speed_m_s,
            tangent.z * speed_m_s,
        )
        # Complex Motion: no yaw snap — rate-limited turn while translating.
        result = vehicle.command(
            Motion(position=pos, velocity=velocity, yaw_rate=yaw_rate)
        )
        publications += 1
        if not result.accepted:
            guided_lost = True
            reasons.append("smooth: guided flight lost")
            break
        distance += speed_m_s * REFRESH_S
        _drain(platform)
        platform._sleep(REFRESH_S)

    # Brief hold at the tip so land isn't commanded mid-cruise.
    if not guided_lost:
        end, _ = _sample_polyline(points, path_m)
        hold_until = platform._monotonic() + END_HOLD_S
        while platform._monotonic() < hold_until:
            result = vehicle.command(Motion(position=end, velocity=ZERO, yaw=0.0))
            publications += 1
            if not result.accepted:
                guided_lost = True
                reasons.append("smooth-hold: guided flight lost")
                break
            _drain(platform)
            platform._sleep(REFRESH_S)

    steps.append(
        {
            "task": "smooth_cruise",
            "path_m": path_m,
            "speed_m_s": speed_m_s,
            "max_yaw_rate_rad_s": MAX_YAW_RATE_RAD_S,
            "peak_commanded_yaw_rate": max_abs_yaw_rate,
            "publications": publications,
            "ok": not guided_lost and distance >= path_m * 0.95,
        }
    )
    if not guided_lost and distance < path_m * 0.95:
        reasons.append(
            f"smooth: only reached {distance:.2f} m of {path_m:.2f} m path"
        )

    land = vehicle.land()
    steps.append({"task": "land", "ok": land.accepted})
    if not land.accepted:
        reasons.append(land.reason or "land refused")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        _drain(platform)
        time.sleep(0.05)

    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "steps": steps,
    }


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

    route = _load_route()
    waypoints = list(route["waypoints_local_ned"])
    hover_m = float(route["hover_altitude_m"])

    writer = EvidenceWriter(output_dir, "run-a")
    platform = WebotsArduPilot(
        settings,
        runner=SubprocessRunner(),
        session=PymavlinkSession(),
        gateway=TcpSensorGateway(
            stamp=lambda: settings.capture_stamp(time.monotonic_ns())
        ),
        evidence=writer,
        label="control-smooth",
        extra_params=settings.compat_estimator_params,
    )
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.ready)
        result = fly_smooth_route(
            platform, waypoints_ned=waypoints, hover_m=hover_m
        )
        writer.write_json("smooth-flight.json", result)
        gate = GateStatus.PASS if result["status"] == "pass" else GateStatus.FAIL
        return CommandOutcome(
            status=CommandStatus.COMPLETE,
            gate_status=gate,
            reasons=tuple(result["reasons"]),
            limitations=(
                "Throwaway continuous smooth cruise; not the control-layer gate.",
            ),
            manifest={
                "command": "control-smooth",
                "world": str(settings.world),
                "route": str(ROUTE_PATH),
                "steps": result["steps"],
            },
            artifacts=("run-a/smooth-flight.json",),
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
    "control-smooth",
    _command,
    help_text="Throwaway continuous smooth cruise through the control route.",
    stage_id="P00-control-smooth",
    run_prefix="p00-control-smooth",
    add_arguments=_add_arguments,
)
