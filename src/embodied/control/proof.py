"""Live control-layer proof: gapless caller-owned Vehicle API v1 loop.

One publication loop while armed GUIDED::

    while flying:
        choose current phase
        produce one Motion
        Vehicle.command(Motion)
        ~50 ms

Phase changes (outbound / hold / spin / reface / settle / align / return)
replace the current Motion without stopping publication. Scoring happens on
transition ticks; the next Motion is commanded in the same loop cadence.

Run::

    ./configs/layers/control
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
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


@dataclass
class Phase:
    """One route segment. Kind selects how Motion is built each tick."""

    name: str
    kind: str  # hold | yaw_rate
    duration_s: float
    target: Vec3 | None = None
    yaw: float | None = 0.0
    yaw_rate: float | None = None
    requested_rad: float | None = None
    # Filled on phase entry (first tick); never stop publishing to capture these.
    frozen_position: Vec3 | None = None
    start_yaw: float | None = None
    start_sim: float | None = None
    publications: int = 0
    entered: bool = False


@dataclass
class FlightLog:
    steps: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _hold_motion(target: Vec3, *, yaw: float | None) -> Motion:
    return Motion(position=target, velocity=ZERO, yaw=yaw)


def _phase_motion(phase: Phase) -> Motion:
    if phase.kind == "yaw_rate":
        assert phase.frozen_position is not None
        assert phase.yaw_rate is not None
        return Motion(
            position=phase.frozen_position,
            velocity=ZERO,
            yaw_rate=phase.yaw_rate,
        )
    assert phase.target is not None or phase.frozen_position is not None
    position = phase.target if phase.target is not None else phase.frozen_position
    assert position is not None
    return _hold_motion(position, yaw=phase.yaw)


def _score_hold_phase(
    log: FlightLog,
    phase: Phase,
    vehicle: Vehicle,
    *,
    residual_max_m: float,
    guided_lost: bool,
) -> None:
    state = vehicle.state()
    target = phase.target if phase.target is not None else phase.frozen_position
    residual = None
    if state.position is not None and target is not None:
        residual = math.dist(state.position.as_tuple(), target.as_tuple())
    ok = (
        not guided_lost
        and residual is not None
        and residual <= residual_max_m
    )
    log.steps.append(
        {
            "task": phase.name,
            "target_odom": None if target is None else list(target.as_tuple()),
            "residual_m": residual,
            "publications": phase.publications,
            "ok": ok,
        }
    )
    if guided_lost:
        log.reasons.append(f"{phase.name}: guided flight lost")
    elif residual is None:
        log.reasons.append(f"{phase.name}: no position to score")
    elif residual > residual_max_m:
        log.reasons.append(
            f"{phase.name}: residual {residual:.3f} m > {residual_max_m:.3f} m"
        )


def _score_yaw_phase(
    log: FlightLog,
    phase: Phase,
    vehicle: Vehicle,
    *,
    spin_tolerance_rad: float,
    guided_lost: bool,
) -> None:
    state = vehicle.state()
    delta = None
    if phase.start_yaw is not None and state.yaw is not None:
        delta = wrap_angle_rad(state.yaw - phase.start_yaw)
    requested = phase.requested_rad or 0.0
    ok = (
        not guided_lost
        and delta is not None
        and abs(abs(delta) - abs(requested)) <= spin_tolerance_rad
    )
    log.steps.append(
        {
            "task": phase.name,
            "requested_rad": requested,
            "delta_rad": delta,
            "publications": phase.publications,
            "ok": ok,
        }
    )
    if guided_lost:
        log.reasons.append(f"{phase.name}: guided flight lost")
    elif delta is None:
        log.reasons.append(f"{phase.name}: no yaw delta to score")
    elif not ok:
        log.reasons.append(
            f"{phase.name}: |delta| {abs(delta):.3f} rad vs requested "
            f"{abs(requested):.3f} (tol {spin_tolerance_rad:.3f})"
        )


def _build_phases(
    *,
    waypoints: tuple[tuple[float, float, float], ...],
    hold_s: float,
    hover_m: float,
    spin_rad: float,
    return_waypoints: bool,
) -> list[Phase]:
    phases: list[Phase] = []
    for index, waypoint in enumerate(waypoints):
        phases.append(
            Phase(
                name=f"goto[{index}]",
                kind="hold",
                duration_s=hold_s,
                target=_ned_waypoint_to_odom(*waypoint, hover_m=hover_m),
                yaw=0.0,
            )
        )
    phases.append(
        Phase(
            name="hold",
            kind="hold",
            duration_s=hold_s,
            target=None,  # freeze XY on entry
            yaw=0.0,
        )
    )
    phases.append(
        Phase(
            name="spin",
            kind="yaw_rate",
            duration_s=abs(spin_rad) / SPIN_RATE_RAD_S,
            yaw=None,
            yaw_rate=SPIN_RATE_RAD_S if spin_rad >= 0.0 else -SPIN_RATE_RAD_S,
            requested_rad=spin_rad,
        )
    )
    if return_waypoints:
        phases.append(
            Phase(
                name="reface",
                kind="yaw_rate",
                duration_s=abs(spin_rad) / SPIN_RATE_RAD_S,
                yaw=None,
                yaw_rate=-SPIN_RATE_RAD_S if spin_rad >= 0.0 else SPIN_RATE_RAD_S,
                requested_rad=-spin_rad,
            )
        )
        # Explicit hold Motion at entry pose with yaw=0 — still published every tick.
        phases.append(
            Phase(
                name="settle",
                kind="hold",
                duration_s=hold_s,
                target=None,
                yaw=0.0,
            )
        )
        inbound = tuple(reversed(waypoints[:-1])) if len(waypoints) > 1 else ()
        if inbound and waypoints:
            far_north, far_east, far_z = waypoints[-1]
            _first_north, first_east, _first_z = inbound[0]
            if abs(first_east - far_east) > 1e-6:
                phases.append(
                    Phase(
                        name="align",
                        kind="hold",
                        duration_s=hold_s,
                        target=_ned_waypoint_to_odom(
                            far_north, first_east, far_z, hover_m=hover_m
                        ),
                        yaw=0.0,
                    )
                )
        for index, waypoint in enumerate(inbound):
            phases.append(
                Phase(
                    name=f"return[{index}]",
                    kind="hold",
                    duration_s=hold_s,
                    target=_ned_waypoint_to_odom(*waypoint, hover_m=hover_m),
                    yaw=0.0,
                )
            )
    return phases


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
    """Fly the layer route with one gapless Vehicle.command() loop."""
    vehicle = platform.vehicle
    drain: Callable[[], None] = lambda: _drain(platform)
    log = FlightLog()

    takeoff = vehicle.takeoff(hover_m)
    state = vehicle.state()
    log.steps.append(
        {
            "task": "takeoff",
            "ok": takeoff.accepted,
            "altitude_m": None if state.position is None else state.position.z,
            "reason": takeoff.reason,
        }
    )
    if not takeoff.accepted:
        log.reasons.append(takeoff.reason or "takeoff refused")
        return {
            "status": "fail",
            "reasons": log.reasons,
            "steps": log.steps,
            "residual_max_m": residual_max_m,
        }

    phases = _build_phases(
        waypoints=waypoints,
        hold_s=hold_s,
        hover_m=hover_m,
        spin_rad=spin_rad,
        return_waypoints=return_waypoints,
    )
    index = 0
    wall_budget = sum(phase.duration_s for phase in phases) + hold_s
    wall_until = platform._monotonic() + _wall_backstop_s(platform, wall_budget)
    guided_lost = False

    while index < len(phases) and platform._monotonic() < wall_until:
        phase = phases[index]
        state = vehicle.state()

        if not phase.entered:
            # Capture freezes on the same tick we still publish — no silent gap.
            if phase.target is None:
                if state.position is None:
                    log.reasons.append(f"{phase.name}: no position at entry")
                    break
                phase.frozen_position = state.position
                phase.target = state.position
            if phase.kind == "yaw_rate":
                phase.frozen_position = (
                    state.position if state.position is not None else phase.target
                )
                phase.start_yaw = state.yaw
            phase.entered = True
            phase.start_sim = _sim_time_s(platform)

        motion = _phase_motion(phase)
        result = vehicle.command(motion)
        phase.publications += 1
        if not result.accepted:
            guided_lost = True
            if phase.kind == "yaw_rate":
                _score_yaw_phase(
                    log,
                    phase,
                    vehicle,
                    spin_tolerance_rad=spin_tolerance_rad,
                    guided_lost=True,
                )
            else:
                _score_hold_phase(
                    log,
                    phase,
                    vehicle,
                    residual_max_m=residual_max_m,
                    guided_lost=True,
                )
            break

        sim = _sim_time_s(platform)
        phase_done = False
        if phase.start_sim is not None and sim is not None:
            phase_done = (sim - phase.start_sim) >= phase.duration_s
        elif phase.start_sim is None and sim is not None:
            phase.start_sim = sim
        elif phase.publications * REFRESH_S >= phase.duration_s / min(
            platform.settings.realtime_ratio_envelope
        ):
            # Wall-clock fallback if sim time is unavailable.
            phase_done = True

        drain()
        platform._sleep(REFRESH_S)

        if not phase_done:
            continue

        # Score then immediately advance — next loop iteration publishes the
        # next phase Motion without an inter-phase silence.
        if phase.kind == "yaw_rate":
            _score_yaw_phase(
                log,
                phase,
                vehicle,
                spin_tolerance_rad=spin_tolerance_rad,
                guided_lost=False,
            )
        else:
            _score_hold_phase(
                log,
                phase,
                vehicle,
                residual_max_m=residual_max_m,
                guided_lost=False,
            )
        index += 1

    land = vehicle.land()
    log.steps.append({"task": "land", "ok": land.accepted})
    if not land.accepted:
        log.reasons.append(land.reason or "land refused")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        drain()
        time.sleep(0.05)

    return {
        "status": "pass" if not log.reasons else "fail",
        "reasons": log.reasons,
        "steps": log.steps,
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
                "Control proof: gapless Vehicle.command() loop through doorways; "
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
        "Launch the simulator and fly the Vehicle API v1 control route "
        "with a gapless caller-owned command() loop."
    ),
    stage_id="P00-motion",
    run_prefix="p00-motion-proof",
    add_arguments=_add_arguments,
)
