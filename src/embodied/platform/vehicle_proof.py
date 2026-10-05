"""Sim motion proof: start Webots + SITL and fly Vehicle tasks on an open field.

Default scene: ``scenarios/motion/open_field/world.wbt`` via
``configs/motion_open.yaml``.

Tasks (preset waypoints from the config's ``motion`` section):

1. takeoff to hover
2. goto each waypoint in order and hold
3. land

Run::

    python -m embodied motion-proof --config configs/motion_open.yaml \\
        --output work/runs/motion-proof-1
"""

from __future__ import annotations

import argparse
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
from embodied.platform.vehicle import Vehicle
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def _drain(platform: WebotsArduPilot) -> None:
    record = platform.sensor_record(0.02)
    while record is not None:
        record = platform.sensor_record(0.0)


def _motion_section(document: dict[str, Any]) -> dict[str, Any]:
    """Return motion-proof task settings, preferring ``motion`` over ``probe``."""
    motion = document.get("motion")
    probe = document["probe"]
    if motion is not None:
        return motion
    return {
        "hover_altitude_m": probe["hover_altitude_m"],
        "hold_per_waypoint_s": probe["hold_per_waypoint_s"],
        "residual_max_m": 0.15,
        "waypoints_local_ned": probe["waypoints_local_ned"],
    }


def fly_motion_route(
    platform: WebotsArduPilot,
    *,
    waypoints: tuple[tuple[float, float, float], ...],
    hold_s: float,
    hover_m: float,
    residual_max_m: float,
) -> dict[str, Any]:
    """Fly takeoff → each waypoint → land through :class:`Vehicle`."""
    vehicle = platform.vehicle
    drain: Callable[[], None] = lambda: _drain(platform)
    reasons: list[str] = []
    steps: list[dict[str, Any]] = []

    takeoff = vehicle.takeoff(
        platform.settings.step_timeout_s.flight, drain=drain
    )
    steps.append(
        {
            "task": "takeoff",
            "ok": not takeoff.refused,
            "altitude_m": takeoff.altitude_m,
        }
    )
    if takeoff.refused:
        reasons.append("takeoff refused")
        return {
            "status": "fail",
            "reasons": reasons,
            "steps": steps,
            "residual_max_m": residual_max_m,
        }

    for index, waypoint in enumerate(waypoints):
        north, east, z_offset = waypoint
        down = -(hover_m - z_offset)
        result = vehicle.goto(
            north,
            east,
            down_m=down,
            hold_s=hold_s,
            yaw_rad=0.0,
            drain=drain,
        )
        residual = result.get("residual_m")
        ok = (
            result.get("ok") is True
            and residual is not None
            and residual <= residual_max_m
        )
        steps.append(
            {
                "task": f"goto[{index}]",
                "target_ned": list(result.get("target_ned") or ()),
                "residual_m": residual,
                "publications": result.get("publications"),
                "ok": ok,
            }
        )
        if result.get("guided_lost"):
            reasons.append(f"goto[{index}]: guided flight lost")
        elif residual is None:
            reasons.append(f"goto[{index}]: no position to score")
        elif residual > residual_max_m:
            reasons.append(
                f"goto[{index}]: residual {residual:.3f} m > {residual_max_m:.3f} m"
            )

    vehicle.land()
    steps.append({"task": "land", "ok": True})
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


def _command(args: argparse.Namespace, output_dir: Path) -> CommandOutcome:
    document = load_config(args.config)
    settings = PlatformSettings.from_config(
        document, root=repository_root(), arm=SensorMode.SIMULATOR_INTERFACE.value
    )
    missing = [item for item in check_prerequisites(settings, output_dir) if not item.satisfied]
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
    )
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.ready)
        result = fly_motion_route(
            platform,
            waypoints=waypoints,
            hold_s=hold_s,
            hover_m=hover_m,
            residual_max_m=residual_max_m,
        )
        writer.write_json("motion-proof.json", result)
        gate = GateStatus.PASS if result["status"] == "pass" else GateStatus.FAIL
        return CommandOutcome(
            status=CommandStatus.COMPLETE,
            gate_status=gate,
            reasons=tuple(result["reasons"]),
            limitations=(
                "Vehicle motion proof: takeoff, all motion waypoints, land; "
                "no estimator or obstacle-avoidance checks",
            ),
            manifest={
                "command": "motion-proof",
                "world": str(settings.world),
                "waypoints": [list(item) for item in waypoints],
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
        "Launch the open-field simulator and fly takeoff + motion waypoints through Vehicle."
    ),
    stage_id="P00-motion",
    run_prefix="p00-motion-proof",
    add_arguments=_add_arguments,
)
