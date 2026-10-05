"""Live control-layer proof: Webots + SITL exercising every Vehicle primitive.

Default complex scene: ``configs/motion_doorway.yaml`` (first_indoor apertures).

Route:

1. takeoff
2. goto each outbound waypoint and hold
3. explicit hold at the far end
4. spin (relative yaw rate)
5. reface (reverse yaw-rate spin back toward north)
6. goto return waypoints back toward the pad
7. land

Run the layer gate::

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
DEFAULT_SPIN_TOLERANCE_RAD = 0.35
DEFAULT_RESIDUAL_MAX_M = 0.15


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
        "residual_max_m": DEFAULT_RESIDUAL_MAX_M,
        "waypoints_local_ned": probe["waypoints_local_ned"],
    }


def _score_goto(
    steps: list[dict[str, Any]],
    reasons: list[str],
    *,
    label: str,
    result: dict[str, Any],
    residual_max_m: float,
) -> None:
    residual = result.get("residual_m")
    ok = (
        result.get("ok") is True
        and residual is not None
        and residual <= residual_max_m
    )
    steps.append(
        {
            "task": label,
            "target_ned": list(result.get("target_ned") or ()),
            "residual_m": residual,
            "publications": result.get("publications"),
            "ok": ok,
        }
    )
    if result.get("guided_lost"):
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
    """Fly takeoff → outbound → hold → spin → return → land through Vehicle."""
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
        _score_goto(
            steps,
            reasons,
            label=f"goto[{index}]",
            result=result,
            residual_max_m=residual_max_m,
        )
        if result.get("guided_lost"):
            break

    if not any("guided flight lost" in reason for reason in reasons):
        hold = vehicle.hold(hold_s, yaw_rad=0.0, drain=drain)
        hold_residual = hold.get("residual_m")
        hold_ok = hold.get("ok") is True and not hold.get("guided_lost")
        steps.append(
            {
                "task": "hold",
                "residual_m": hold_residual,
                "publications": hold.get("publications"),
                "ok": hold_ok,
            }
        )
        if hold.get("guided_lost"):
            reasons.append("hold: guided flight lost")
        elif not hold_ok:
            reasons.append("hold: failed")

    if not any("guided flight lost" in reason for reason in reasons):
        spun = vehicle.spin(spin_rad, drain=drain)
        delta = spun.get("delta_rad")
        spin_ok = (
            spun.get("ok") is True
            and not spun.get("guided_lost")
            and delta is not None
            and abs(abs(delta) - abs(spin_rad)) <= spin_tolerance_rad
        )
        steps.append(
            {
                "task": "spin",
                "requested_rad": spun.get("requested_rad"),
                "delta_rad": delta,
                "publications": spun.get("publications"),
                "ok": spin_ok,
            }
        )
        if spun.get("guided_lost"):
            reasons.append("spin: guided flight lost")
        elif delta is None:
            reasons.append("spin: no yaw delta to score")
        elif not spin_ok:
            reasons.append(
                f"spin: |delta| {abs(delta):.3f} rad vs requested {abs(spin_rad):.3f} "
                f"(tol {spin_tolerance_rad:.3f})"
            )

    # Unwind with a rate-controlled reverse spin before inbound legs.
    # Absolute yaw=0 after ~π tip-strikes (AngErr≈120); return while still
    # yawed also tip-struck on return[0] through the 1 m doorway.
    if return_waypoints and not any(
        "guided flight lost" in reason for reason in reasons
    ):
        unwind = vehicle.spin(-spin_rad, drain=drain)
        unwind_delta = unwind.get("delta_rad")
        reface_ok = (
            unwind.get("ok") is True
            and not unwind.get("guided_lost")
            and unwind_delta is not None
            and abs(abs(unwind_delta) - abs(spin_rad)) <= spin_tolerance_rad
        )
        steps.append(
            {
                "task": "reface",
                "requested_rad": unwind.get("requested_rad"),
                "delta_rad": unwind_delta,
                "publications": unwind.get("publications"),
                "ok": reface_ok,
            }
        )
        if unwind.get("guided_lost"):
            reasons.append("reface: guided flight lost")
        elif unwind_delta is None:
            reasons.append("reface: no yaw delta to score")
        elif not reface_ok:
            reasons.append(
                f"reface: |delta| {abs(unwind_delta):.3f} rad vs requested "
                f"{abs(spin_rad):.3f} (tol {spin_tolerance_rad:.3f})"
            )

    if return_waypoints and not any(
        "guided flight lost" in reason for reason in reasons
    ):
        inbound = tuple(reversed(waypoints[:-1])) if len(waypoints) > 1 else ()
        for index, waypoint in enumerate(inbound):
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
            _score_goto(
                steps,
                reasons,
                label=f"return[{index}]",
                result=result,
                residual_max_m=residual_max_m,
            )
            if result.get("guided_lost"):
                break

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


# Keep the old name as an alias for callers/tests that still import it.
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
    # Same EKF-active layering as compat: without it SITL stays on simulator AHRS.
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
                "Control proof: takeoff, outbound gotos, hold, spin, return, land; "
                "no estimator or obstacle-avoidance checks",
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
        "Launch the simulator and fly the full Vehicle control route "
        "(takeoff, gotos, hold, spin, return, land)."
    ),
    stage_id="P00-motion",
    run_prefix="p00-motion-proof",
    add_arguments=_add_arguments,
)
