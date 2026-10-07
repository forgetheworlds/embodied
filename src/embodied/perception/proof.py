"""Live Perception layer proof: sensor_derived FREE from stereo→depth→map.

Bring up Webots+SITL for sensor streams, fly a short hover (Vehicle API), feed
stereo+IMU into the thin Perception pipeline, assert OccupancyQuery FREE with
``evidence_class=sensor_derived``. Never uses POSE truth or Vehicle EKF as map
capture pose.

Run::

    ./configs/layers/perception
"""

from __future__ import annotations

import argparse
import queue
import time
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
from embodied.contracts.perception_ports import EvidenceClass, OccupancySupport
from embodied.contracts.records import ClockStamp, SensorMode
from embodied.control import Motion, Vehicle, Vec3
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
from embodied.perception.pipeline import (
    PerceptionPipeline,
    decode_pair_rgb,
    is_sensor_derived_free,
)

REFRESH_S = 0.05
HOVER_SETTLE_S = 3.0
COLLECT_S = 12.0
ZERO = Vec3(0.0, 0.0, 0.0)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def _stamp() -> ClockStamp:
    return ClockStamp(
        host_id="perception-proof",
        clock_id="host/monotonic",
        monotonic_ns=time.monotonic_ns(),
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


def _drain_into_pipeline(
    platform: WebotsArduPilot,
    pipeline: PerceptionPipeline,
    pair_q: queue.Queue[SensorRecord],
) -> dict[str, Any]:
    """Consume IMU from sensor_record and stereo pixels from the pair sink."""
    last_verdict = None
    pairs = 0
    imus = 0
    while True:
        record = platform.sensor_record(0.0)
        if record is None:
            break
        stamp = _stamp()
        sim = record.sim_time_s if record.sim_time_s is not None and record.sim_time_s >= 0.0 else None
        if record.kind is Kind.IMU and record.imu is not None:
            imus += 1
            pipeline.on_imu(
                accelerometer=record.imu.accelerometer,
                gyro=record.imu.gyro,
                capture_host_ns=record.imu.capture_host_ns,
                sim_time_s=sim,
                stamp=stamp,
            )
        # POSE / STATUS / PAIR-without-pixels ignored for Perception evidence.
    while True:
        try:
            record = pair_q.get_nowait()
        except queue.Empty:
            break
        if record.pair is None:
            continue
        pair = record.pair
        stamp = _stamp()
        sim = record.sim_time_s if record.sim_time_s is not None and record.sim_time_s >= 0.0 else None
        left, right = decode_pair_rgb(
            pair.left_bytes,
            pair.right_bytes,
            width=pair.width,
            height=pair.height,
            encoding=pair.encoding,
        )
        pairs += 1
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
    return {"pairs": pairs, "imus": imus, "last_verdict": last_verdict}


def fly_perception_route(
    platform: WebotsArduPilot,
    *,
    hover_m: float,
    ap_ext_nav_mode: str,
    pair_q: queue.Queue[SensorRecord],
) -> dict[str, Any]:
    reasons: list[str] = []
    steps: list[dict[str, Any]] = []
    occupancy_samples: list[dict[str, Any]] = []
    pipeline = PerceptionPipeline(nav_epoch="perception-proof-epoch")
    vehicle = Vehicle(platform)

    takeoff = vehicle.takeoff(hover_m)
    steps.append({"task": "takeoff", "ok": takeoff.accepted, "reason": takeoff.reason})
    if not takeoff.accepted:
        reasons.append(takeoff.reason or "takeoff refused")
        return _receipt(reasons, steps, ap_ext_nav_mode, pipeline, occupancy_samples)

    deadline = time.monotonic() + HOVER_SETTLE_S
    while time.monotonic() < deadline:
        _drain_into_pipeline(platform, pipeline, pair_q)
        hold = vehicle.state().position or Vec3(0.0, 0.0, hover_m)
        vehicle.command(Motion(position=hold, velocity=ZERO))
        time.sleep(REFRESH_S)

    collect_until = time.monotonic() + COLLECT_S
    free_hits = 0
    while time.monotonic() < collect_until:
        drained = _drain_into_pipeline(platform, pipeline, pair_q)
        hold = vehicle.state().position or Vec3(0.0, 0.0, hover_m)
        vehicle.command(Motion(position=hold, velocity=ZERO))
        verdict = drained["last_verdict"] or pipeline.query_forward_volume(_stamp())
        if verdict is not None:
            sample = {
                "support": verdict.support.value,
                "evidence_class": verdict.evidence_class.value,
                "reason": verdict.reason,
                "map_revision": verdict.map_revision,
                "free_fraction": verdict.free_fraction,
            }
            occupancy_samples.append(sample)
            if is_sensor_derived_free(verdict):
                free_hits += 1
        time.sleep(REFRESH_S)

    steps.append(
        {
            "task": "sensor_collect",
            "ok": free_hits > 0,
            "free_hits": free_hits,
            "pipeline": pipeline.stats(),
            "occupancy_samples_n": len(occupancy_samples),
        }
    )
    if free_hits <= 0:
        reasons.append(
            "no live OccupancyVerdict support=free with evidence_class=sensor_derived "
            f"(stats={pipeline.stats()})"
        )

    # Stall honesty: stop feeding; re-publish nav with wall now so age/stall updates.
    stall_start = time.monotonic()
    while time.monotonic() - stall_start < 1.2:
        record = platform.sensor_record(0.0)
        while record is not None:
            record = platform.sensor_record(0.0)
        hold = vehicle.state().position or Vec3(0.0, 0.0, hover_m)
        vehicle.command(Motion(position=hold, velocity=ZERO))
        time.sleep(REFRESH_S)
    pipeline.estimation.set(pipeline.estimator.latest(stamp=_stamp()))
    nav_after = pipeline.estimation.latest()
    stall_ok = nav_after is None or (not pipeline.estimation.healthy())
    steps.append(
        {
            "task": "feed_stall_honesty",
            "ok": stall_ok,
            "healthy": pipeline.estimation.healthy(),
            "status": None if nav_after is None else nav_after.status.value,
        }
    )
    if not stall_ok:
        reasons.append("estimation stayed healthy after feed stall window")

    land = vehicle.land()
    steps.append({"task": "land", "ok": land.accepted, "reason": land.reason})
    if not land.accepted:
        reasons.append(land.reason or "land refused")

    return _receipt(reasons, steps, ap_ext_nav_mode, pipeline, occupancy_samples, free_hits)


def _receipt(
    reasons: list[str],
    steps: list[dict[str, Any]],
    ap_ext_nav_mode: str,
    pipeline: PerceptionPipeline,
    occupancy_samples: list[dict[str, Any]],
    free_hits: int = 0,
) -> dict[str, Any]:
    nav = pipeline.estimation.latest()
    query = pipeline.mapping.occupancy()
    free_samples = [
        s
        for s in occupancy_samples
        if s.get("support") == OccupancySupport.FREE.value
        and s.get("evidence_class") == EvidenceClass.SENSOR_DERIVED.value
    ]
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "steps": steps,
        "ap_ext_nav_mode": ap_ext_nav_mode,
        "perception_nav_evidence_class": None
        if nav is None
        else nav.evidence_class.value,
        "perception_map_evidence_class": None
        if query is None
        else query.evidence_class.value,
        "geometry_clear_claimed": False,
        "live_sensor_derived_free": free_hits > 0,
        "free_hits": free_hits,
        "free_samples": free_samples[:10],
        "occupancy_samples": occupancy_samples[-20:],
        "pipeline": pipeline.stats(),
        "note": (
            "AP ext-nav mode is orthogonal to Perception evidence_class. "
            "Capture pose from stereo_imu_nav (accel+gyro); depth from stereo SGBM; "
            "MapStoreMappingPort.integrate(nav=sensor_derived)."
        ),
    }


def _command(args: argparse.Namespace, output_dir: Path) -> CommandOutcome:
    document = load_config(args.config)
    # Truth VPE into AP for airworthiness only — not Perception capture pose.
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

    writer = EvidenceWriter(output_dir, "run-a")
    platform = WebotsArduPilot(
        settings,
        runner=SubprocessRunner(),
        session=PymavlinkSession(),
        gateway=TcpSensorGateway(
            stamp=lambda: settings.capture_stamp(time.monotonic_ns())
        ),
        evidence=writer,
        label="perception-proof",
        extra_params=settings.compat_estimator_params,
    )
    # Must install before start(): reader strips pair pixels after the sink runs.
    pair_q = _install_pair_sink(platform)
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.ready)
        result = fly_perception_route(
            platform,
            hover_m=hover_m,
            ap_ext_nav_mode=settings.sensor_mode.value,
            pair_q=pair_q,
        )
        writer.write_json("perception-proof.json", result)
        gate = GateStatus.PASS if result["status"] == "pass" else GateStatus.FAIL
        return CommandOutcome(
            status=CommandStatus.COMPLETE,
            gate_status=gate,
            reasons=tuple(result["reasons"]),
            limitations=(
                "Perception live: stereo_imu_nav + SGBM depth + MapStore → "
                "sensor_derived FREE; AP ext-nav orthogonal.",
            ),
            manifest={
                "command": "perception-proof",
                "world": str(settings.world),
                "ap_ext_nav_mode": result["ap_ext_nav_mode"],
                "perception_nav_evidence_class": result["perception_nav_evidence_class"],
                "perception_map_evidence_class": result["perception_map_evidence_class"],
                "live_sensor_derived_free": result["live_sensor_derived_free"],
                "free_hits": result["free_hits"],
                "steps": result["steps"],
            },
            artifacts=("run-a/perception-proof.json",),
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
    "perception-proof",
    _command,
    help_text=(
        "Live Perception proof: stereo+IMU → depth → MapStore → "
        "sensor_derived FREE OccupancyQuery."
    ),
    stage_id="P00-perception",
    run_prefix="p00-perception-proof",
    add_arguments=_add_arguments,
)
