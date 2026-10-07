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
import queue
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
    EvidenceClass,
    OccupancySupport,
    StopTubeQuery,
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
from embodied.memory.world import FREE
from embodied.perception.pipeline import (
    PerceptionPipeline,
    decode_pair_rgb,
    is_sensor_derived_free,
)
from embodied.platform.webots_ardupilot import (
    EvidenceWriter,
    Kind,
    PlatformSettings,
    PlatformUnavailable,
    ProbeFailure,
    PymavlinkSession,
    SensorRecord,
    SubprocessRunner,
    TcpSensorGateway,
    WebotsArduPilot,
    check_prerequisites,
)

REFRESH_S = 0.05
SEGMENT_DURATION_S = 3.0
SEGMENT_DX_M = 0.4
HOVER_SETTLE_S = 2.0
COLLECT_S = 12.0
ALIGN_WAIT_S = 2.0
STOP_TUBE_ENVELOPE_M = 0.35


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


def _install_pair_sink(platform: WebotsArduPilot) -> queue.Queue[SensorRecord]:
    """Pixels only arrive through record_sink; sensor_record strips pair payloads."""
    pair_q: queue.Queue[SensorRecord] = queue.Queue(maxsize=8)

    def _sink(record: SensorRecord) -> None:
        if record.kind is not Kind.PAIR or record.pair is None:
            return
        try:
            pair_q.put_nowait(record)
        except queue.Full:
            try:
                pair_q.get_nowait()
            except queue.Empty:
                pass
            try:
                pair_q.put_nowait(record)
            except queue.Full:
                pass

    platform.record_sink = _sink
    return pair_q


def _feed_imu_burst(
    platform: WebotsArduPilot, pipeline: PerceptionPipeline
) -> int:
    """Drain pending IMU (and ignore non-IMU) so nav age stays honest mid-pair."""
    imus = 0
    while True:
        record = platform.sensor_record(0.0)
        if record is None:
            break
        if record.kind is not Kind.IMU or record.imu is None:
            continue
        stamp = _stamp()
        sim = (
            record.sim_time_s
            if record.sim_time_s is not None and record.sim_time_s >= 0.0
            else None
        )
        imus += 1
        pipeline.on_imu(
            accelerometer=record.imu.accelerometer,
            gyro=record.imu.gyro,
            capture_host_ns=record.imu.capture_host_ns,
            sim_time_s=sim,
            stamp=stamp,
        )
    return imus


def _drain_into_pipeline(
    platform: WebotsArduPilot,
    pipeline: PerceptionPipeline,
    pair_q: queue.Queue[SensorRecord],
    *,
    integrate_map: bool = True,
) -> dict[str, Any]:
    """Consume IMU + stereo. Interleave IMU around pairs so SGBM cannot starve nav.

    When ``integrate_map`` is False (geometry segment with pinned revision), pairs
    only refresh visual age — no MapStore integrate.
    """
    last_verdict = None
    pairs = 0
    imus = _feed_imu_burst(platform, pipeline)
    # Cap pairs per tick so one drain cannot burn >stall_after wall time.
    max_pairs = 2 if integrate_map else 4
    while pairs < max_pairs:
        try:
            record = pair_q.get_nowait()
        except queue.Empty:
            break
        if record.pair is None:
            continue
        pair = record.pair
        stamp = _stamp()
        sim = (
            record.sim_time_s
            if record.sim_time_s is not None and record.sim_time_s >= 0.0
            else None
        )
        pairs += 1
        if integrate_map:
            left, right = decode_pair_rgb(
                pair.left_bytes,
                pair.right_bytes,
                width=pair.width,
                height=pair.height,
                encoding=pair.encoding,
            )
            verdict = pipeline.on_pair(
                left_rgb=left,
                right_rgb=right,
                pair_id=pair.pair_id,
                capture_host_ns=pair.capture_host_ns,
                sim_time_s=sim,
                stamp=stamp,
            )
            if verdict is not None:
                last_verdict = verdict
        else:
            pipeline.mark_pair_for_nav(
                capture_host_ns=pair.capture_host_ns,
                sim_time_s=sim,
                stamp=stamp,
            )
        # SGBM can take hundreds of ms — refresh IMU before age trips stall.
        imus += _feed_imu_burst(platform, pipeline)
    imus += _feed_imu_burst(platform, pipeline)
    pipeline.refresh_nav(_stamp())
    return {"pairs": pairs, "imus": imus, "last_verdict": last_verdict}


def _stop_tube_free_at(
    pipeline: PerceptionPipeline, position: Vec3, *, envelope_m: float
) -> dict[str, Any] | None:
    """Preflight Safety's stop-tube query at a candidate (no invented FREE)."""
    occ = pipeline.mapping.occupancy()
    if occ is None:
        return None
    verdict = occ.query_stop_tube(
        StopTubeQuery(
            samples_odom_m=((position.x, position.y, position.z),),
            envelope_radius_m=envelope_m,
            include_brake_region=False,
            brake_region=None,
        ),
        now=occ.stamp,
    )
    return {
        "support": verdict.support.value,
        "evidence_class": verdict.evidence_class.value,
        "reason": verdict.reason,
        "map_revision": verdict.map_revision,
        "free_fraction": verdict.free_fraction,
        "clear": is_sensor_derived_free(verdict),
    }


def _clear_anchor_near(
    pipeline: PerceptionPipeline,
    vehicle: Vec3,
    *,
    envelope_m: float,
    max_dist_m: float,
) -> tuple[Vec3 | None, dict[str, Any] | None]:
    """Pick a known FREE cell center near the vehicle where stop_tube is CLEAR.

    Body hover is often never_observed (sparse map). Safety CLEAR queries the
    candidate Motions sample — so the short geometry segment must sit on a
    real free+sensor_derived stop tube, within tracking distance of the vehicle.
    """
    occ = pipeline.mapping.occupancy()
    if occ is None:
        return None, None
    ranked: list[tuple[float, Vec3]] = []
    for cell, label in occ.cells:
        if label != FREE:
            continue
        cx, cy, cz = occ._cell_center(cell)  # noqa: SLF001 — same centers Safety volumes use
        dx = cx - vehicle.x
        dy = cy - vehicle.y
        dz = cz - vehicle.z
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist <= max_dist_m:
            ranked.append((dist, Vec3(cx, cy, cz)))
    ranked.sort(key=lambda item: item[0])
    for dist, center in ranked[:64]:
        preflight = _stop_tube_free_at(pipeline, center, envelope_m=envelope_m)
        if preflight is not None and preflight.get("clear"):
            preflight = {**preflight, "anchor_dist_m": dist}
            return center, preflight
    # Best-effort report: nearest free cell even if stop_tube not clear.
    if ranked:
        center = ranked[0][1]
        return None, _stop_tube_free_at(pipeline, center, envelope_m=envelope_m)
    return None, None


@dataclass
class WallClocks:
    t0_mono: float
    t0_sim: float

    def now_mono_s(self) -> float:
        return time.monotonic() - self.t0_mono

    def now_sim_s(self) -> float:
        return time.monotonic() - self.t0_sim


def _make_cert(
    start: Vec3,
    *,
    epoch: str,
    geometry: GeometryCertificate | None,
    clear_anchor: Vec3 | None = None,
) -> TrajectoryCertificate:
    # When geometry CLEAR is attached, hold on a verified free stop-tube anchor
    # (not the never_observed body voxel). Candidate stays CLEAR for the short prove.
    if clear_anchor is not None:
        primary = HoldTrajectory(position=clear_anchor, duration_s=SEGMENT_DURATION_S)
        terminal = HoldTrajectory(position=clear_anchor)
        hold_fallback = clear_anchor
        start_state_pos = clear_anchor
    else:
        end = Vec3(start.x + SEGMENT_DX_M, start.y, start.z)
        primary = SegmentTrajectory(start=start, end=end, duration_s=SEGMENT_DURATION_S)
        terminal = HoldTrajectory(position=end)
        hold_fallback = start
        start_state_pos = start
    return TrajectoryCertificate(
        certificate_id="exec-proof-1",
        primary=primary,
        terminal=terminal,
        fallbacks={"hold": HoldTrajectory(position=hold_fallback)},
        nav_epoch=epoch,
        start_state=StartState(position=start_state_pos, velocity=Vec3(0, 0, 0)),
        start_tolerance=StartTolerance(position_m=1.5, velocity_mps=2.0),
        validity=ValidityWindow(None, None),
        tracking_envelope=TrackingEnvelope(position_m=1.5, velocity_mps=3.0),
        geometry_certificate=geometry,
        safety_evidence_refs=("execution_proof",),
        plant_limits_ref="declared",
    )


def _sample_occupancy(pipeline: PerceptionPipeline) -> dict[str, Any] | None:
    verdict = pipeline.query_sensor_derived_free(_stamp())
    if verdict is None:
        return None
    return {
        "support": verdict.support.value,
        "evidence_class": verdict.evidence_class.value,
        "reason": verdict.reason,
        "map_revision": verdict.map_revision,
        "nav_epoch": verdict.nav_epoch,
        "free_fraction": verdict.free_fraction,
    }


def fly_execution_route(
    platform: WebotsArduPilot,
    *,
    hover_m: float,
    ap_ext_nav_mode: str,
    pair_q: queue.Queue[SensorRecord],
    require_geometry_clear: bool = False,
) -> dict[str, Any]:
    """Orchestrate joint flight. Safety.check is consumed, not reimplemented."""
    reasons: list[str] = []
    steps: list[dict[str, Any]] = []
    occupancy_samples: list[dict[str, Any]] = []
    vehicle = Vehicle(platform)
    clocks = WallClocks(t0_mono=time.monotonic(), t0_sim=time.monotonic())
    epoch = "exec-proof-epoch"

    # Perception producer (not empty pose_assisted stub). No MissionRuntime.
    pipeline = PerceptionPipeline(nav_epoch=epoch)
    mapping_source = "perception.pipeline"
    steps.append({"task": "mapping_port_source", "source": mapping_source})

    ports = PortBundle(
        estimation=pipeline.estimation,
        mapping=pipeline.mapping,
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
            pipeline,
            require_geometry_clear=require_geometry_clear,
            mapping_source=mapping_source,
        )

    state = vehicle.state()
    if state.position is None:
        reasons.append("no vehicle position after takeoff")
        return _receipt(
            reasons,
            steps,
            ap_ext_nav_mode,
            exe,
            pipeline,
            require_geometry_clear=require_geometry_clear,
            mapping_source=mapping_source,
        )

    start = state.position
    hold_cert = TrajectoryCertificate(
        certificate_id="exec-proof-hold",
        primary=HoldTrajectory(
            position=start, duration_s=ALIGN_WAIT_S + HOVER_SETTLE_S + COLLECT_S
        ),
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
            pipeline,
            require_geometry_clear=require_geometry_clear,
            mapping_source=mapping_source,
        )

    # Align stereo_imu odom to Vehicle once before map integrate (frame glue).
    aligned = False
    align_deadline = time.monotonic() + ALIGN_WAIT_S
    while time.monotonic() < align_deadline and not aligned:
        _drain_into_pipeline(platform, pipeline, pair_q, integrate_map=False)
        exe._tick()
        pos = vehicle.state().position or start
        aligned = pipeline.align_odom_position((pos.x, pos.y, pos.z))
        time.sleep(REFRESH_S)
    steps.append(
        {
            "task": "odom_align",
            "ok": aligned,
            "pipeline": pipeline.stats(),
        }
    )

    # Settle + collect: integrate map until free+sensor_derived (or timeout).
    free_hits = 0
    settle_end = time.monotonic() + HOVER_SETTLE_S
    collect_end = settle_end + COLLECT_S
    while time.monotonic() < collect_end:
        drained = _drain_into_pipeline(platform, pipeline, pair_q, integrate_map=True)
        exe._tick()
        verdict = drained["last_verdict"] or pipeline.query_sensor_derived_free(_stamp())
        if is_sensor_derived_free(verdict):
            free_hits += 1
            sample = _sample_occupancy(pipeline)
            if sample is not None:
                occupancy_samples.append(sample)
        elif time.monotonic() >= settle_end:
            sample = _sample_occupancy(pipeline)
            if sample is not None:
                occupancy_samples.append(sample)
        if free_hits > 0 and time.monotonic() >= settle_end:
            break
        time.sleep(REFRESH_S)

    steps.append(
        {
            "task": "perception_collect",
            "free_hits": free_hits,
            "pipeline": pipeline.stats(),
            "occupancy_samples_n": len(occupancy_samples),
        }
    )

    # Geometry CLEAR only when required AND stop-tube at candidate is FREE.
    # Do not attach a free-claim cert without live free+sensor_derived evidence.
    geometry: GeometryCertificate | None = None
    geometry_attached = False
    stop_tube_preflight: dict[str, Any] | None = None
    clear_anchor: Vec3 | None = None
    state = vehicle.state()
    start = state.position or start
    if require_geometry_clear:
        if free_hits < 1:
            reasons.append(
                "geometry CLEAR required but Perception never returned free+sensor_derived "
                f"(stats={pipeline.stats()})"
            )
            vehicle.land()
            return _receipt(
                reasons,
                steps,
                ap_ext_nav_mode,
                exe,
                pipeline,
                occupancy_samples,
                require_geometry_clear=require_geometry_clear,
                mapping_source=mapping_source,
                free_hits=free_hits,
                geometry_attached=False,
            )
        # Body voxel is often never_observed; Safety CLEAR needs a free stop-tube
        # on the Motions candidate — pick a nearby known FREE cell center.
        clear_anchor, stop_tube_preflight = _clear_anchor_near(
            pipeline,
            start,
            envelope_m=STOP_TUBE_ENVELOPE_M,
            max_dist_m=1.4,
        )
        if clear_anchor is None or stop_tube_preflight is None or not stop_tube_preflight.get(
            "clear"
        ):
            reasons.append(
                "geometry CLEAR required but no nearby FREE stop_tube anchor "
                f"(vehicle_stop_tube={_stop_tube_free_at(pipeline, start, envelope_m=STOP_TUBE_ENVELOPE_M)}, "
                f"anchor_preflight={stop_tube_preflight})"
            )
            vehicle.land()
            return _receipt(
                reasons,
                steps,
                ap_ext_nav_mode,
                exe,
                pipeline,
                occupancy_samples,
                require_geometry_clear=require_geometry_clear,
                mapping_source=mapping_source,
                free_hits=free_hits,
                geometry_attached=False,
            )
        occ_now = pipeline.mapping.occupancy()
        revision = "rev-0" if occ_now is None else occ_now.map_revision
        geometry = GeometryCertificate(
            nav_epoch=epoch,
            map_revision=revision,
            volume_refs=("stop_tube",),
            support_claim="free",
        )
        geometry_attached = True

    steps.append(
        {
            "task": "geometry_preflight",
            "stop_tube": stop_tube_preflight,
            "clear_anchor": None
            if clear_anchor is None
            else (clear_anchor.x, clear_anchor.y, clear_anchor.z),
            "geometry_on_cert": geometry_attached,
            "map_revision": None if geometry is None else geometry.map_revision,
        }
    )

    cert = _make_cert(
        start, epoch=epoch, geometry=geometry, clear_anchor=clear_anchor
    )
    replaced = exe.replace(cert)
    steps.append(
        {
            "task": "replace",
            "ok": hasattr(replaced, "certificate_id"),
            "detail": getattr(replaced, "certificate_id", None)
            or getattr(replaced, "reason", None),
            "geometry_required": require_geometry_clear,
            "geometry_on_cert": geometry_attached,
            "free_hits_before_replace": free_hits,
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
            pipeline,
            occupancy_samples,
            require_geometry_clear=require_geometry_clear,
            mapping_source=mapping_source,
            free_hits=free_hits,
            geometry_attached=geometry_attached,
        )

    # Gapless tick: freeze map revision while geometry is attached (IMU+pair age only).
    end_mono = time.monotonic() + SEGMENT_DURATION_S + 1.5
    geometry_allow_seen = False
    decision_counts: dict[str, int] = {}
    while time.monotonic() < end_mono:
        _drain_into_pipeline(
            platform,
            pipeline,
            pair_q,
            integrate_map=not geometry_attached,
        )
        sample = _sample_occupancy(pipeline)
        if sample is not None:
            occupancy_samples.append(sample)
            if (
                sample["support"] == OccupancySupport.FREE.value
                and sample["evidence_class"] != EvidenceClass.SENSOR_DERIVED.value
            ):
                reasons.append(
                    f"live map FREE with evidence_class={sample['evidence_class']} — forbidden"
                )
            if (
                sample["support"] == OccupancySupport.FREE.value
                and sample["evidence_class"] == EvidenceClass.SENSOR_DERIVED.value
            ):
                free_hits += 1
        exe._tick()
        status = exe.status()
        key = status.last_decision or "none"
        if status.last_reason:
            key = f"{key}:{status.last_reason}"
        decision_counts[key] = decision_counts.get(key, 0) + 1
        if geometry_attached and status.last_decision == "allow":
            geometry_allow_seen = True
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
            "geometry_allow_seen": geometry_allow_seen,
            "decision_counts": decision_counts,
            "map_frozen": geometry_attached,
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
    if require_geometry_clear and not geometry_allow_seen:
        reasons.append(
            "geometry CLEAR required but Safety never ALLOW under free-claim cert "
            f"(last_decision={status.last_decision}, last_reason={status.last_reason}, "
            f"decision_counts={decision_counts})"
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
        pipeline,
        occupancy_samples,
        require_geometry_clear=require_geometry_clear,
        mapping_source=mapping_source,
        free_hits=free_hits,
        geometry_attached=geometry_attached,
        geometry_allow_seen=geometry_allow_seen,
    )


def _receipt(
    reasons: list[str],
    steps: list[dict[str, Any]],
    ap_ext_nav_mode: str,
    exe: Execution,
    pipeline: PerceptionPipeline,
    occupancy_samples: list[dict[str, Any]] | None = None,
    *,
    require_geometry_clear: bool = False,
    mapping_source: str = "perception.pipeline",
    free_hits: int = 0,
    geometry_attached: bool = False,
    geometry_allow_seen: bool = False,
) -> dict[str, Any]:
    status = exe.status()
    samples = occupancy_samples or []
    occ_summary = _occupancy_summary(samples)
    nav = pipeline.estimation.latest()
    query = pipeline.mapping.occupancy()
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "steps": steps,
        # AP airworthiness vs Perception evidence — keep separate.
        "ap_ext_nav_mode": ap_ext_nav_mode,
        "perception_nav_evidence_class": None
        if nav is None
        else nav.evidence_class.value,
        "perception_map_evidence_class": None
        if query is None
        else query.evidence_class.value,
        "mapping_port_source": mapping_source,
        "require_geometry_clear": require_geometry_clear,
        "geometry_certificate_attached": geometry_attached,
        # Only claim CLEAR when Safety actually allowed under a free-claim cert.
        "geometry_clear_claimed": bool(geometry_attached and geometry_allow_seen),
        "live_sensor_derived_free": free_hits > 0,
        "free_hits": free_hits,
        "occupancy_summary": occ_summary,
        "pipeline": pipeline.stats(),
        "execution": {
            "code": status.code.value,
            "publish_sequence": status.publish_sequence,
            "primary_completed": status.primary_completed,
            "last_decision": status.last_decision,
            "last_reason": status.last_reason,
            "rebuild": "Vehicle + PerceptionPipeline ports + Safety.check — no MissionRuntime",
        },
        "occupancy_samples": samples[-10:],
        "note": (
            "Clean Execution rebuild: sole Vehicle.command; consumes Safety + PerceptionPipeline; "
            "no MissionRuntime/navigation dual-writer. "
            "AP ext-nav (airworthiness) is orthogonal to Perception evidence_class. "
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
    # Must install before start(): reader strips pair pixels after the sink runs.
    pair_q = _install_pair_sink(platform)
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.ready)
        result = fly_execution_route(
            platform,
            hover_m=hover_m,
            ap_ext_nav_mode=settings.sensor_mode.value,
            pair_q=pair_q,
            require_geometry_clear=require_geometry_clear,
        )
        writer.write_json("execution-proof.json", result)
        gate = GateStatus.PASS if result["status"] == "pass" else GateStatus.FAIL
        return CommandOutcome(
            status=CommandStatus.COMPLETE,
            gate_status=gate,
            reasons=tuple(result["reasons"]),
            limitations=(
                "Execution joint orchestration: sole Vehicle.command; consumes Safety.check; "
                "PerceptionPipeline producer (sensor_derived); "
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
                "geometry_certificate_attached": result["geometry_certificate_attached"],
                "geometry_clear_claimed": result["geometry_clear_claimed"],
                "live_sensor_derived_free": result["live_sensor_derived_free"],
                "free_hits": result["free_hits"],
                "occupancy_summary": result.get("occupancy_summary"),
                "mapping_port_source": result["mapping_port_source"],
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
