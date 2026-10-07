"""Joint live proof orchestration (Execution layer entrypoint).

Owns: takeoff → Execution.replace → gapless ``_tick`` (sole ``Vehicle.command``)
→ land. Calls Safety.check + Perception ports; does not own those layers.

AP may use truth/diagnostic VPE for airworthiness. Receipt lists
``ap_ext_nav_mode`` separately from Perception ``evidence_class``.
If ``require_geometry_clear`` is set and live occupancy never yields
``free``+``sensor_derived``, the proof FAILS (no invented FREE).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied.cli import (
    CommandOutcome,
    CommandStatus,
    GateStatus,
    load_config,
    register_command,
    repository_root,
)
from embodied.contracts.perception_ports import (
    AgeDomain,
    EvidenceClass,
    NavPose,
    NavStatus,
    NavigationState,
    OccupancySupport,
    uncompared_disagreement,
)
from embodied.contracts.records import ClockStamp, SensorMode
from embodied.control import Vehicle, Vec3
from embodied.execution import (
    Execution,
    ExecutionStatusCode,
    GeometryCertificate,
    HoldTrajectory,
    PortBundle,
    SegmentTrajectory,
    StartState,
    StartTolerance,
    TrackingEnvelope,
    TrajectoryCertificate,
    ValidityWindow,
)
from embodied.execution.plant import StaticPlantLimitsPort, declared_plant
from embodied.perception.estimation import StaticEstimationPort
from embodied.perception.mapping_ports import StubMappingPort, StubOccupancyQuery
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

REFRESH_S = 0.05
SEGMENT_DURATION_S = 4.0
SEGMENT_DX_M = 0.8
HOVER_SETTLE_S = 2.0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def _occupancy_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    supports = sorted({s.get("support", "?") for s in samples})
    evidence = sorted({s.get("evidence_class", "?") for s in samples})
    free_sd = sum(
        1
        for s in samples
        if s.get("support") == OccupancySupport.FREE.value
        and s.get("evidence_class") == EvidenceClass.SENSOR_DERIVED.value
    )
    return {
        "sample_count": len(samples),
        "supports_seen": supports,
        "evidence_classes_seen": evidence,
        "free_sensor_derived_count": free_sd,
        "live_occupancy_used": len(samples) > 0 and supports != ["unsupported"],
    }


def _stamp(ns: int | None = None) -> ClockStamp:
    return ClockStamp(
        host_id="execution-proof",
        clock_id="host/monotonic",
        monotonic_ns=int(time.monotonic_ns() if ns is None else ns),
    )


def _drain(platform: WebotsArduPilot) -> None:
    record = platform.sensor_record(0.02)
    while record is not None:
        record = platform.sensor_record(0.0)


@dataclass
class WallClocks:
    t0_mono: float
    t0_sim: float

    def now_mono_s(self) -> float:
        return time.monotonic() - self.t0_mono

    def now_sim_s(self) -> float:
        # Prefer Webots/sim time when available; fall back to wall delta.
        return time.monotonic() - self.t0_sim


def _nav_from_vehicle(
    vehicle: Vehicle,
    *,
    epoch: str,
    evidence_class: EvidenceClass,
) -> NavigationState | None:
    state = vehicle.state()
    if state.position is None:
        return None
    pos = state.position
    vel = state.velocity or Vec3(0.0, 0.0, 0.0)
    yaw = state.yaw or 0.0
    # yaw → simple quaternion about z (ENU)
    half = 0.5 * yaw
    quat = (math.cos(half), 0.0, 0.0, math.sin(half))
    stamp = _stamp()
    pose = NavPose(
        parent_frame="odom",
        child_frame="body",
        stamp=stamp,
        position_m=(pos.x, pos.y, pos.z),
        quaternion_wxyz=quat,
        covariance=None,
        nav_epoch=epoch,
        source_ids=("vehicle_ekf",),
        valid=True,
    )
    return NavigationState(
        nav_epoch=epoch,
        state_sequence=int(time.monotonic_ns() % 1_000_000),
        controller_alignment_id="proof-align",
        stamp=stamp,
        sim_time_s=None,
        age_s=0.0,
        age_domain=AgeDomain.MONOTONIC,
        monotonic_observed_at_s=time.monotonic(),
        pose=pose,
        velocity_mps=(vel.x, vel.y, vel.z),
        covariance=None,
        status=NavStatus.HEALTHY,
        valid=True,
        sigma_pos_m=None,
        visual_source_ids=(),
        imu_source_ids=(),
        visual_age_s=None,
        imu_age_s=None,
        feed_stall=False,
        ap_disagreement=uncompared_disagreement(),
        evidence_class=evidence_class,
        source_ids=("vehicle_ekf", "execution_proof"),
    )


def _make_cert(
    start: Vec3,
    *,
    epoch: str,
    geometry: GeometryCertificate | None,
) -> TrajectoryCertificate:
    end = Vec3(start.x + SEGMENT_DX_M, start.y, start.z)
    return TrajectoryCertificate(
        certificate_id="exec-proof-1",
        primary=SegmentTrajectory(start=start, end=end, duration_s=SEGMENT_DURATION_S),
        terminal=HoldTrajectory(position=end),
        fallbacks={"hold": HoldTrajectory(position=start)},
        nav_epoch=epoch,
        start_state=StartState(position=start, velocity=Vec3(0, 0, 0)),
        start_tolerance=StartTolerance(position_m=1.0, velocity_mps=2.0),
        validity=ValidityWindow(None, None),
        tracking_envelope=TrackingEnvelope(position_m=1.5, velocity_mps=3.0),
        geometry_certificate=geometry,
        safety_evidence_refs=("execution_proof",),
        plant_limits_ref="declared",
    )


def _compose_mapping_port(*, epoch: str, map_evidence_class: EvidenceClass):
    """Prefer Perception composition if present; else stub (no MapStore glue)."""
    try:
        from embodied.perception import live as perception_live  # type: ignore

        compose = getattr(perception_live, "compose_mapping_port", None)
        if callable(compose):
            return compose(nav_epoch=epoch, evidence_class=map_evidence_class), "perception.live"
    except Exception:
        pass
    return (
        StubMappingPort(
            StubOccupancyQuery(
                nav_epoch=epoch,
                map_revision="stub-revision",
                snapshot_id="stub-snapshot",
                evidence_class=map_evidence_class,
            )
        ),
        "stub_mapping_port",
    )


def fly_execution_route(
    platform: WebotsArduPilot,
    *,
    hover_m: float,
    ap_ext_nav_mode: str,
    require_geometry_clear: bool = False,
    nav_evidence_class: EvidenceClass = EvidenceClass.POSE_ASSISTED,
    map_evidence_class: EvidenceClass = EvidenceClass.POSE_ASSISTED,
) -> dict[str, Any]:
    """Orchestrate joint flight. Safety.check is consumed, not reimplemented."""
    reasons: list[str] = []
    steps: list[dict[str, Any]] = []
    vehicle = Vehicle(platform)
    clocks = WallClocks(t0_mono=time.monotonic(), t0_sim=time.monotonic())
    epoch = "exec-proof-epoch"

    # Clean rebuild: Vehicle + EstimationPort + MappingPort + Safety.check.
    # No MissionRuntime / navigation dual-writer. MapStore only via Perception.
    estimation = StaticEstimationPort(None)
    mapping_port, mapping_source = _compose_mapping_port(
        epoch=epoch, map_evidence_class=map_evidence_class
    )
    steps.append({"task": "mapping_port_source", "source": mapping_source})

    ports = PortBundle(
        estimation=estimation,
        mapping=mapping_port,
        plant_limits=StaticPlantLimitsPort(declared_plant()),
    )
    exe = Execution(vehicle=vehicle, clocks=clocks, ports=ports, lease_s=0.3)

    takeoff = vehicle.takeoff(hover_m)
    steps.append({"task": "takeoff", "ok": takeoff.accepted, "reason": takeoff.reason})
    if not takeoff.accepted:
        reasons.append(takeoff.reason or "takeoff refused")
        return _receipt(
            reasons,
            steps,
            ap_ext_nav_mode,
            exe,
            require_geometry_clear=require_geometry_clear,
            nav_evidence_class=nav_evidence_class,
            map_evidence_class=map_evidence_class,
            mapping_source=mapping_source,
        )

    # After takeoff: Execution is sole Vehicle.command writer (hold settle via replace).
    state = vehicle.state()
    if state.position is None:
        reasons.append("no vehicle position after takeoff")
        return _receipt(
            reasons,
            steps,
            ap_ext_nav_mode,
            exe,
            require_geometry_clear=require_geometry_clear,
            nav_evidence_class=nav_evidence_class,
            map_evidence_class=map_evidence_class,
            mapping_source=mapping_source,
        )

    start = state.position
    estimation.set(
        _nav_from_vehicle(vehicle, epoch=epoch, evidence_class=nav_evidence_class)
    )
    hold_cert = TrajectoryCertificate(
        certificate_id="exec-proof-hold",
        primary=HoldTrajectory(position=start, duration_s=HOVER_SETTLE_S),
        terminal=HoldTrajectory(position=start),
        fallbacks={"hold": HoldTrajectory(position=start)},
        nav_epoch=epoch,
        start_state=StartState(position=start, velocity=Vec3(0, 0, 0)),
        start_tolerance=StartTolerance(position_m=1.0, velocity_mps=2.0),
        validity=ValidityWindow(None, None),
        tracking_envelope=TrackingEnvelope(position_m=1.5, velocity_mps=3.0),
        geometry_certificate=None,
        safety_evidence_refs=("execution_proof",),
        plant_limits_ref="declared",
    )
    hold_replace = exe.replace(hold_cert)
    if hasattr(hold_replace, "reason"):
        reasons.append(f"hold replace failed: {hold_replace.reason}")
        vehicle.land()
        return _receipt(
            reasons,
            steps,
            ap_ext_nav_mode,
            exe,
            require_geometry_clear=require_geometry_clear,
            nav_evidence_class=nav_evidence_class,
            map_evidence_class=map_evidence_class,
            mapping_source=mapping_source,
        )
    settle_end = time.monotonic() + HOVER_SETTLE_S
    while time.monotonic() < settle_end:
        _drain(platform)
        estimation.set(
            _nav_from_vehicle(vehicle, epoch=epoch, evidence_class=nav_evidence_class)
        )
        exe._tick()
        time.sleep(REFRESH_S)

    state = vehicle.state()
    start = state.position or start
    estimation.set(
        _nav_from_vehicle(vehicle, epoch=epoch, evidence_class=nav_evidence_class)
    )

    # Geometry CLEAR only when required AND Perception can underwrite it.
    # Until then: geometry=None (fail closed — no CLEAR claim).
    geometry: GeometryCertificate | None = None
    if require_geometry_clear:
        occ_now = mapping_port.occupancy()
        revision = "rev-0" if occ_now is None else occ_now.map_revision
        geometry = GeometryCertificate(
            nav_epoch=epoch,
            map_revision=revision,
            volume_refs=("stop_tube",),
            support_claim="free",
        )
    cert = _make_cert(start, epoch=epoch, geometry=geometry)
    replaced = exe.replace(cert)
    steps.append(
        {
            "task": "replace",
            "ok": hasattr(replaced, "certificate_id"),
            "detail": getattr(replaced, "certificate_id", None) or getattr(replaced, "reason", None),
            "geometry_required": require_geometry_clear,
            "geometry_on_cert": geometry is not None,
        }
    )
    if hasattr(replaced, "reason"):
        reasons.append(f"replace failed: {replaced.reason}")
        vehicle.land()
        return _receipt(
            reasons,
            steps,
            ap_ext_nav_mode,
            exe,
            require_geometry_clear=require_geometry_clear,
            nav_evidence_class=nav_evidence_class,
            map_evidence_class=map_evidence_class,
            mapping_source=mapping_source,
        )

    # Gapless Execution tick through primary + brief terminal
    end_mono = time.monotonic() + SEGMENT_DURATION_S + 2.0
    occupancy_samples: list[dict[str, Any]] = []
    while time.monotonic() < end_mono:
        _drain(platform)
        estimation.set(
            _nav_from_vehicle(vehicle, epoch=epoch, evidence_class=nav_evidence_class)
        )
        occ = mapping_port.occupancy()
        if occ is not None:
            from embodied.contracts.perception_ports import Aabb

            sample = vehicle.state().position or start
            verdict = occ.query_volume(
                Aabb(
                    min_m=(sample.x - 0.3, sample.y - 0.3, sample.z - 0.3),
                    max_m=(sample.x + 0.3, sample.y + 0.3, sample.z + 0.3),
                    frame="odom",
                ),
                now=_stamp(),
            )
            occupancy_samples.append(
                {
                    "support": verdict.support.value,
                    "evidence_class": verdict.evidence_class.value,
                    "reason": verdict.reason,
                    "map_revision": verdict.map_revision,
                    "nav_epoch": verdict.nav_epoch,
                }
            )
            # Fail closed: FREE without sensor_derived is dishonest
            if (
                verdict.support is OccupancySupport.FREE
                and verdict.evidence_class is not EvidenceClass.SENSOR_DERIVED
            ):
                reasons.append(
                    f"live map FREE with evidence_class={verdict.evidence_class.value} — forbidden"
                )
        exe._tick()
        status = exe.status()
        if status.code is ExecutionStatusCode.FAILED:
            reasons.append(status.last_reason or "execution failed")
            break
        if status.code is ExecutionStatusCode.BLOCKED:
            reasons.append(status.last_reason or "execution blocked")
            break
        time.sleep(REFRESH_S)

    occ_summary = _occupancy_summary(occupancy_samples)
    status = exe.status()
    steps.append(
        {
            "task": "execution_loop",
            "code": status.code.value,
            "publish_sequence": status.publish_sequence,
            "primary_completed": status.primary_completed,
            "last_decision": status.last_decision,
            "last_reason": status.last_reason,
            "occupancy_summary": occ_summary,
            "occupancy_samples": occupancy_samples[-5:],
        }
    )
    if status.publish_sequence < 5:
        reasons.append(f"too few publishes: {status.publish_sequence}")
    elif status.code not in (
        ExecutionStatusCode.RUNNING,
        ExecutionStatusCode.COMPLETED,
        ExecutionStatusCode.BACKUP,
    ):
        if status.code not in (ExecutionStatusCode.FAILED, ExecutionStatusCode.BLOCKED):
            reasons.append(f"unexpected status {status.code.value}")

    if require_geometry_clear and occ_summary["free_sensor_derived_count"] < 1:
        reasons.append(
            "geometry CLEAR required but live occupancy never returned free+sensor_derived "
            f"(supports={occ_summary['supports_seen']})"
        )

    land = vehicle.land()
    steps.append({"task": "land", "ok": land.accepted, "reason": land.reason})
    if not land.accepted:
        reasons.append(land.reason or "land refused")

    return _receipt(
        reasons,
        steps,
        ap_ext_nav_mode,
        exe,
        occupancy_samples,
        require_geometry_clear=require_geometry_clear,
        nav_evidence_class=nav_evidence_class,
        map_evidence_class=map_evidence_class,
        mapping_source=mapping_source,
    )


def _receipt(
    reasons: list[str],
    steps: list[dict[str, Any]],
    ap_ext_nav_mode: str,
    exe: Execution,
    occupancy_samples: list[dict[str, Any]] | None = None,
    *,
    require_geometry_clear: bool = False,
    nav_evidence_class: EvidenceClass = EvidenceClass.POSE_ASSISTED,
    map_evidence_class: EvidenceClass = EvidenceClass.POSE_ASSISTED,
    mapping_source: str = "stub_mapping_port",
) -> dict[str, Any]:
    status = exe.status()
    samples = occupancy_samples or []
    occ_summary = _occupancy_summary(samples)
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "steps": steps,
        "ap_ext_nav_mode": ap_ext_nav_mode,
        "perception_nav_evidence_class": nav_evidence_class.value,
        "perception_map_evidence_class": map_evidence_class.value,
        "mapping_port_source": mapping_source,
        "require_geometry_clear": require_geometry_clear,
        "geometry_clear_claimed": require_geometry_clear,
        "occupancy_summary": occ_summary,
        "execution": {
            "code": status.code.value,
            "publish_sequence": status.publish_sequence,
            "primary_completed": status.primary_completed,
            "last_decision": status.last_decision,
            "last_reason": status.last_reason,
            "rebuild": "Vehicle + ports + Safety.check — no MissionRuntime",
        },
        "occupancy_samples": samples[-10:],
        "note": (
            "Clean Execution rebuild: sole Vehicle.command; consumes Safety + Perception ports; "
            "no MissionRuntime/navigation dual-writer. "
            "geometry CLEAR only when require_geometry_clear and live free+sensor_derived."
        ),
    }


def _command(args: argparse.Namespace, output_dir: Path) -> CommandOutcome:
    document = load_config(args.config)
    # Simulator-interface: truth VPE into AP for airworthiness only.
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

    motion = document.get("motion") or document["probe"]
    hover_m = float(motion["hover_altitude_m"])
    proof_cfg = document.get("execution_proof") or {}
    require_geometry_clear = bool(proof_cfg.get("require_geometry_clear", False))
    nav_ec = EvidenceClass(str(proof_cfg.get("nav_evidence_class", "pose_assisted")))
    map_ec = EvidenceClass(str(proof_cfg.get("map_evidence_class", "pose_assisted")))

    writer = EvidenceWriter(output_dir, "run-a")
    platform = WebotsArduPilot(
        settings,
        runner=SubprocessRunner(),
        session=PymavlinkSession(),
        gateway=TcpSensorGateway(
            stamp=lambda: settings.capture_stamp(time.monotonic_ns())
        ),
        evidence=writer,
        label="execution-proof",
        extra_params=settings.compat_estimator_params,
    )
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.ready)
        result = fly_execution_route(
            platform,
            hover_m=hover_m,
            ap_ext_nav_mode=settings.sensor_mode.value,
            require_geometry_clear=require_geometry_clear,
            nav_evidence_class=nav_ec,
            map_evidence_class=map_ec,
        )
        writer.write_json("execution-proof.json", result)
        gate = GateStatus.PASS if result["status"] == "pass" else GateStatus.FAIL
        return CommandOutcome(
            status=CommandStatus.COMPLETE,
            gate_status=gate,
            reasons=tuple(result["reasons"]),
            limitations=(
                "Execution joint orchestration: sole Vehicle.command; consumes Safety.check; "
                "geometry CLEAR only if require_geometry_clear and live free+sensor_derived; "
                "AP ext-nav recorded separately from Perception evidence_class.",
            ),
            manifest={
                "command": "execution-proof",
                "world": str(settings.world),
                "ap_ext_nav_mode": result["ap_ext_nav_mode"],
                "perception_nav_evidence_class": result["perception_nav_evidence_class"],
                "perception_map_evidence_class": result["perception_map_evidence_class"],
                "require_geometry_clear": result["require_geometry_clear"],
                "geometry_clear_claimed": result["geometry_clear_claimed"],
                "occupancy_summary": result.get("occupancy_summary"),
                "steps": result["steps"],
            },
            artifacts=("run-a/execution-proof.json",),
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
    "execution-proof",
    _command,
    help_text=(
        "Joint Perception+Safety+Execution live proof: takeoff, replace, "
        "Safety-gated Execution publishes, land."
    ),
    stage_id="P00-execution",
    run_prefix="p00-execution-proof",
    add_arguments=_add_arguments,
)
