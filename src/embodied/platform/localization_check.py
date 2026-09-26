"""The ``localize-check`` command: P01-L's acceptance run and its receipt.

The command runs one way in two shapes: as a module entry point
(``python -m embodied.platform.localization_check``, the recipe that works while
the dispatch registration is serialized) and as a registered command
(``python -m embodied localize-check`` once the integrator adds the one dispatch
line; this module registers itself at import, and never edits the parser).

Modes are the honesty surface (plan section 8). ``--mode sensor-derived`` is the
only mode that can pass P01-L: GPS off, no bridge truth republish, the pinned
estimator fed only the declared stereo and inertial streams.
``--mode pose-assisted`` is a labelled diagnostic that can never pass — its
receipt carries ``gate_status: not_applicable`` and is never pooled with a
sensor-derived arm. Any other mode value is refused at this boundary.

A run is one of three things, and the receipt says which:

* a live sensor-derived run, scored against the bounds frozen in the
  configuration before any measurement (plan section 6);
* a blocked run: a prerequisite of the claimed arm is missing — among them the
  serialized integrator actions the plan names (merge, ``p01l_sensor.parm``,
  ``estimator/ov_stream``) — nothing is started, and the receipt records
  ``localization=unresolved`` with every concrete blocker. This is a complete,
  honest outcome (plan section 11): no bound is relaxed, no truth is fed;
* a pose-assisted diagnostic, labelled and not applicable to the gate.

Evaluator truth measures error; it never enters the runtime. On a branch
without the merged gate's truth channel the E1 statistics are recorded as
``not_measured`` with the reason, and the gate cannot pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from embodied.cli import (
    CONFIG_SCHEMA,
    CommandOutcome,
    CommandStatus,
    ConfigError,
    GateStatus,
    _validate,
    register_command,
    repository_root,
)
from embodied.contracts.records import SensorMode
from embodied.platform import localization as loc
from embodied.platform.sensors import SensorSample, capture_latency_ns, sim_time_ns
from embodied.platform.webots_ardupilot import (
    EvidenceWriter,
    Kind,
    LocalNedTarget,
    PlatformSettings,
    PymavlinkSession,
    ProbeFailure,
    SubprocessRunner,
    TcpSensorGateway,
    WebotsArduPilot,
    check_prerequisites,
    read_configured_parameters,
)

STAGE_ID = "P01-L"
COMMAND_NAME = "localize-check"
RUN_PREFIX = "p01-localization"

DISPATCH_REGISTRATION_NOTE = (
    "the localize-check dispatch line in embodied.cli COMMAND_MODULES is the integrator's "
    "serialized registration; this module registers itself at import and runs standalone, "
    "so the live check works before that line is added"
)

# The parameter layer the claimed arm needs, and where each requirement lives.
# The EKF source selection is the merged gate's (compat_ekf.parm on main); GPS
# off and the visual-odometer layer are the p01l_sensor.parm file the plan
# materializes as a new file (plan section 4.3).
SEAM_REQUIREMENTS: tuple[tuple[str, float, str], ...] = (
    ("EK3_SRC1_POSXY", 6.0, "merged compat_ekf.parm"),
    ("EK3_SRC1_VELXY", 6.0, "merged compat_ekf.parm"),
    ("EK3_SRC1_POSZ", 6.0, "merged compat_ekf.parm"),
    ("EK3_SRC1_YAW", 6.0, "merged compat_ekf.parm"),
    ("VISO_TYPE", 1.0, "merged compat_ekf.parm"),
    ("COMPASS_USE", 0.0, "merged compat_ekf.parm"),
    ("GPS_TYPE", 0.0, "p01l_sensor.parm"),
    ("VISO_DELAY_MS", 50.0, "p01l_sensor.parm"),
    ("VISO_QUAL_MIN", 0.0, "p01l_sensor.parm"),
    ("FS_EKF_ACTION", 1.0, "p01l_sensor.parm"),
)
P01L_PARAMS_FILENAME = "p01l_sensor.parm"

# The additive localization section: the pin's identity, the publish cadence,
# the declared pipeline delay, the parameter layer, and the predeclared bounds.
# The bounds are the plan's values (plan section 6), frozen here before any
# measurement; they are never adjusted after seeing a run.
LOCALIZATION_SECTION: dict[str, Any] = {
    "estimator": {
        "name": str,
        "tag": str,
        "commit": str,
        "tarball_path": str,
        "tarball_sha256": str,
        "build_log": str,
        "build_success_marker": str,
        "library": str,
        "executable": str,
        "socket_port": int,
    },
    "publish": {"period_ms": int},
    "declared": {"viso_delay_ms": int},
    "params_file": str,
    "bounds": {
        "state_lost_after_ms": int,
        "published_state_age_max_ms": int,
        "max_publish_gap_ms": int,
        "visual_update_warn_ms": int,
        "visual_update_fail_ms": int,
        "valid_fraction_min": float,
        "sigma_min_m": float,
        "sigma_max_m": float,
        "disagreement_p95_m": float,
        "disagreement_max_m": float,
    },
}

# The schema this stage validates: the shared schema plus the additive
# localization section. The configuration's ``calibration`` section is P01-C's
# offline-derivation contract; this stage consumes only the derived record at
# sensors.calibration (and its declared start pose, below) and leaves that
# section to its owner, so it is excluded from the view validated here. The
# shared-file schema extension rides the same serialized cli.py edit as the
# dispatch registration.
LOCALIZATION_CONFIG_SCHEMA: dict[str, Any] = {
    **CONFIG_SCHEMA,
    "localization": LOCALIZATION_SECTION,
}


def _load_localization_config(path: Path) -> dict[str, Any]:
    """Load and validate the configuration for this stage.

    The loading discipline is the shared loader's; the one difference is the
    schema, which carries this stage's additive section.
    """
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - PyYAML is a pinned dependency
        raise ConfigError(f"PyYAML is required to read {path}: {error}") from error
    try:
        document = yaml.safe_load(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{path} is not readable YAML: {error}") from error
    if not isinstance(document, dict):
        raise ConfigError(f"{path} must hold a mapping at the top level")
    consumable = {key: value for key, value in document.items() if key != "calibration"}
    _validate(consumable, LOCALIZATION_CONFIG_SCHEMA, path.name)
    return document


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/first_indoor.yaml"),
        help="the scenario, platform and localization configuration this run declares",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[SensorMode.SENSOR_DERIVED.value, SensorMode.POSE_ASSISTED.value],
        required=True,
        help="sensor-derived is the claimed arm; pose-assisted is a labelled diagnostic",
    )


# ---------------------------------------------------------------------------
# Preflight: everything the claimed arm needs, reported in one pass
# ---------------------------------------------------------------------------


def _parse_mavlink_port(endpoint: str) -> int | None:
    tail = endpoint.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pin_blockers(localization: dict[str, Any], root: Path) -> list[str]:
    """Version evidence for the pin: sha256 on disk, a successful build, the library."""
    estimator = localization["estimator"]
    blockers: list[str] = []
    tarball = root / estimator["tarball_path"]
    if not tarball.is_file():
        blockers.append(f"the pinned tarball {estimator['tarball_path']} is not on disk")
    else:
        digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
        if digest != estimator["tarball_sha256"]:
            blockers.append(
                f"the pinned tarball's sha256 {digest} does not match the configured pin "
                f"{estimator['tarball_sha256']}"
            )
    build_log = root / estimator["build_log"]
    if not build_log.is_file():
        blockers.append(
            f"the pinned estimator has no build log at {estimator['build_log']}; a pin needs "
            "a successful build on this host as version evidence (plan section 3)"
        )
    elif estimator["build_success_marker"] not in build_log.read_text(
        encoding="utf-8", errors="replace"
    ):
        blockers.append(
            f"the estimator build log {estimator['build_log']} does not record "
            f"{estimator['build_success_marker']!r}: no successful build of the pinned "
            "estimator tree is evidenced"
        )
    library = root / estimator["library"]
    if not library.is_file():
        blockers.append(f"the built estimator library {estimator['library']} is not on disk")
    return blockers


def _executable_blockers(localization: dict[str, Any], root: Path) -> list[str]:
    executable = root / localization["estimator"]["executable"]
    if executable.is_file():
        return []
    return [
        f"the estimator process executable {localization['estimator']['executable']} does not "
        "exist; estimator/ov_stream.cpp and its build script are integrator-materialized "
        "(plan section 1, action 5), and without them there is no estimator process to run"
    ]


def _seam_blockers(document: dict[str, Any], root: Path) -> list[str]:
    """The ExternalNav seam's parameter selection, read from the files that would run.

    Every requirement is reported, including the ones a missing file hides:
    a blocked run is most useful when it names everything that is absent in one
    pass, and the missing file and the missing selection have different owners.
    """
    paths = [root / name for name in document["scenario"]["estimator_params"]]
    blockers: list[str] = []
    for path in paths:
        if not path.is_file():
            blockers.append(
                f"the estimator parameter layer is missing {path}; {P01L_PARAMS_FILENAME} "
                "is integrator action 2 (plan section 1)"
            )
    configured = read_configured_parameters([path for path in paths if path.is_file()])
    for name, expected, source in SEAM_REQUIREMENTS:
        actual = configured.get(name)
        if actual is None:
            blockers.append(
                f"{name} is set by no estimator parameter file; the claimed arm needs "
                f"{name} {expected:g} from {source}"
            )
        elif actual != expected:
            blockers.append(
                f"{name} is {actual:g} in the applied parameter files; the claimed arm needs "
                f"{name} {expected:g} from {source}"
            )
    if not any(path.name == P01L_PARAMS_FILENAME for path in paths):
        blockers.append(
            f"{P01L_PARAMS_FILENAME} is not listed in scenario.estimator_params; GPS cannot "
            "be disabled for the claimed arm without it"
        )
    return blockers


def _preflight(document: dict[str, Any], output_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    """Every prerequisite, reported in one pass. Nothing is started by this function."""
    root = repository_root()
    settings = PlatformSettings.from_config(document, root=root)
    rows: list[dict[str, Any]] = []
    satisfied = True
    for check in check_prerequisites(settings, output_dir):
        rows.append({"name": check.name, "satisfied": check.satisfied, "detail": check.detail})
        satisfied = satisfied and check.satisfied
    mavlink_port = _parse_mavlink_port(settings.mavlink_endpoint)
    port_free = mavlink_port is not None and _port_is_free(mavlink_port)
    rows.append(
        {
            "name": "port_mavlink",
            "satisfied": port_free,
            "detail": f"tcp {settings.mavlink_endpoint} is "
            + ("free" if port_free else "already in use; kill orphaned SITL processes first"),
        }
    )
    satisfied = satisfied and port_free
    localization = document["localization"]
    for blocker in (
        *_pin_blockers(localization, root),
        *_executable_blockers(localization, root),
        *_seam_blockers(document, root),
    ):
        rows.append({"name": "estimator_seam", "satisfied": False, "detail": blocker})
        satisfied = False
    return rows, satisfied


# ---------------------------------------------------------------------------
# Command outcomes
# ---------------------------------------------------------------------------


def _blocked_unresolved(
    reasons: Sequence[str],
    limitations: Sequence[str],
    manifest: dict[str, Any],
    artifacts: Sequence[str],
    mode: SensorMode,
) -> CommandOutcome:
    """The unresolved protocol: blocked, every concrete blocker named, nothing faked."""
    return CommandOutcome(
        status=CommandStatus.BLOCKED,
        gate_status=GateStatus.NOT_APPLICABLE,
        reasons=(*reasons, "localization=unresolved"),
        limitations=limitations,
        manifest={**manifest, "localization": "unresolved"},
        artifacts=tuple(artifacts),
        sensor_mode=mode,
    )


def _pose_assisted_outcome(reasons: Sequence[str]) -> CommandOutcome:
    """The labelled diagnostic outcome: never a gate result, never pooled."""
    return CommandOutcome(
        status=CommandStatus.BLOCKED if reasons else CommandStatus.COMPLETE,
        gate_status=GateStatus.NOT_APPLICABLE,
        reasons=(*reasons, "diagnostic mode cannot pass P01-L"),
        limitations=(
            "sensor_mode pose-assisted-diagnostic: this run would use the gate's "
            "truth-quality pose feed and is labelled a diagnostic in every artifact; it can "
            "never pass P01-L and is never pooled with a sensor-derived arm",
            DISPATCH_REGISTRATION_NOTE,
        ),
        manifest={
            "stage_id": STAGE_ID,
            "sensor_mode_label": "pose-assisted-diagnostic",
            "localization": "not_applicable",
        },
        sensor_mode=SensorMode.POSE_ASSISTED,
    )


def _localize_check_command(args: argparse.Namespace, output_dir: Path) -> CommandOutcome:
    mode = SensorMode(args.mode)
    document = _load_localization_config(Path(args.config))
    if mode is SensorMode.POSE_ASSISTED:
        return _pose_assisted_outcome(
            (
                "the gate's truth-quality pose feed exists only on the merged branch "
                "(integrator actions 1 and 3); this branch has no truth channel to "
                "diagnose with",
            ),
        )

    rows, satisfied = _preflight(document, output_dir)
    preflight = {
        "mode": mode.value,
        "sensor_mode_label": "sensor-derived",
        "checks": rows,
        "satisfied": satisfied,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    if not satisfied:
        blockers = tuple(f"{row['name']}: {row['detail']}" for row in rows if not row["satisfied"])
        return _blocked_unresolved(
            blockers,
            (
                "no process was started and nothing was substituted: the claimed arm runs "
                "only when every prerequisite exists",
                "P01-L is blocked for scored sensor-only mode; pose-assisted diagnostics "
                "carry on labelled",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {
                "stage_id": STAGE_ID,
                "sensor_mode_label": "sensor-derived",
                "preflight": rows,
                "estimator_pin": document["localization"]["estimator"],
                "bounds": document["localization"]["bounds"],
            },
            ("preflight.json",),
            mode,
        )
    return _run_sensor_derived_live(document, output_dir)


# ---------------------------------------------------------------------------
# The live sensor-derived run
# ---------------------------------------------------------------------------


class _FeedStats:
    """What the estimator feed consumed, for the receipt's accounting."""

    def __init__(self) -> None:
        self.pairs = 0
        self.imu_samples = 0
        self.pair_latencies_ns: list[int] = []
        self.imu_latencies_ns: list[int] = []
        self.newest_imu_ns = 0


def _start_estimator(estimator: dict[str, Any], root: Path, writer: EvidenceWriter):
    executable = root / estimator["executable"]
    port = int(estimator["socket_port"])
    log_path = writer.path("estimator.log")
    process = subprocess.Popen(
        [str(executable), str(port)],
        cwd=str(root),
        stdout=log_path.open("ab"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if _port_listening(port):
            return process
        if process.poll() is not None:
            return None
        time.sleep(0.1)
    process.terminate()
    return None


def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _stop_estimator(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()


def _write_health_events(writer: EvidenceWriter, machine: loc.HealthMachine) -> None:
    with writer.path("health-events.jsonl").open("a", encoding="utf-8") as handle:
        for event in machine.events:
            handle.write(
                json.dumps(
                    {
                        "at_ns": event.at_ns,
                        "event": event.event,
                        "detail": event.detail,
                        "reset_counter": machine.reset_counter,
                    }
                )
                + "\n"
            )


def _write_log(writer: EvidenceWriter, lines: Sequence[str]) -> None:
    writer.path("log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_sensor_derived_live(document: dict[str, Any], output_dir: Path) -> CommandOutcome:
    root = repository_root()
    settings = PlatformSettings.from_config(document, root=root)
    localization = document["localization"]
    estimator = localization["estimator"]
    bounds_config = localization["bounds"]
    bounds = loc.HealthBounds(
        publish_period_s=localization["publish"]["period_ms"] / 1000.0,
        state_lost_after_s=bounds_config["state_lost_after_ms"] / 1000.0,
        published_state_age_max_s=bounds_config["published_state_age_max_ms"] / 1000.0,
        max_publish_gap_s=bounds_config["max_publish_gap_ms"] / 1000.0,
        visual_update_warn_s=bounds_config["visual_update_warn_ms"] / 1000.0,
        visual_update_fail_s=bounds_config["visual_update_fail_ms"] / 1000.0,
        valid_fraction_min=bounds_config["valid_fraction_min"],
        sigma_min_m=bounds_config["sigma_min_m"],
        sigma_max_m=bounds_config["sigma_max_m"],
    )
    writer = EvidenceWriter(output_dir, "run-a")
    alignment_origin = document["calibration"]["referee"]["body_position_world_m"]
    alignment = loc.OdomAlignment(alignment_origin)
    machine = loc.HealthMachine(bounds)

    estimator_process = _start_estimator(estimator, root, writer)
    if estimator_process is None:
        _stop = f"the estimator process {estimator['executable']} did not open port {estimator['socket_port']}; see run-a/estimator.log"
        _write_log(writer, [f"UNRESOLVED: {_stop}"])
        return _blocked_unresolved(
            (_stop,),
            (
                "no bound was relaxed and no truth was fed; the predeclared stop rule "
                "(plan section 11) records the blocker and stops",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {"stage_id": STAGE_ID, "sensor_mode_label": "sensor-derived", "estimator_pin": estimator},
            (*writer.artifacts, "preflight.json"),
            SensorMode.SENSOR_DERIVED,
        )

    client = loc.OvStreamClient("127.0.0.1", int(estimator["socket_port"]))
    feed_log_path = writer.path("estimator-feed.jsonl")
    stats = _FeedStats()
    latest_aligned: dict[str, object] | None = None

    def on_publish(state: loc.EstimatorState, aligned: dict[str, object]) -> None:
        nonlocal latest_aligned
        latest_aligned = aligned
        with feed_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "published_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        **aligned,
                        "n_tracks": state.n_tracks,
                        "t_last_visual_ns": state.t_last_visual_ns,
                        "visual_age_verdict": machine.visual_update_verdict(state),
                    },
                    default=str,
                )
                + "\n"
            )

    publisher = loc.ExternalNavPublisher(
        settings.mavlink_endpoint, alignment, machine, on_publish=on_publish
    )
    session = PymavlinkSession()
    platform = WebotsArduPilot(
        settings,
        runner=SubprocessRunner(),
        session=session,
        gateway=TcpSensorGateway(stamp=lambda: settings.capture_stamp(time.monotonic_ns())),
        evidence=writer,
        label="run-a",
        extra_params=settings.estimator_params,
    )
    log_lines: list[str] = [
        f"P01-L sensor-derived live run, started {datetime.now(timezone.utc).isoformat()}",
        f"estimator pin: {estimator['name']} {estimator['tag']} ({estimator['commit']})",
        f"odom origin (declared start, ENU): {alignment_origin}",
        f"parameter layer: {[str(path) for path in settings.estimator_params]}",
    ]

    def feed_record(record: Any) -> None:
        if record.kind is Kind.PAIR and record.pair is not None:
            pair = record.pair
            sample = SensorSample(
                value=pair,
                capture_stamp=settings.capture_stamp(pair.capture_host_ns),
                receipt_stamp=record.received_stamp,
                sim_time_s=record.sim_time_s,
            )
            stats.pair_latencies_ns.append(capture_latency_ns(sample))
            left = loc.grayscale_rgb8(
                pair.left_bytes, settings.stereo.width, settings.stereo.height
            )
            right = loc.grayscale_rgb8(
                pair.right_bytes, settings.stereo.width, settings.stereo.height
            )
            client.send(
                loc.encode_stereo(
                    sim_time_ns(record.sim_time_s),
                    left,
                    right,
                    settings.stereo.width,
                    settings.stereo.height,
                )
            )
            stats.pairs += 1
        elif record.kind is Kind.IMU and record.imu is not None:
            imu = record.imu
            sample = SensorSample(
                value=imu,
                capture_stamp=settings.capture_stamp(imu.capture_host_ns),
                receipt_stamp=record.received_stamp,
                sim_time_s=record.sim_time_s,
            )
            stats.imu_latencies_ns.append(capture_latency_ns(sample))
            stamp_ns = sim_time_ns(record.sim_time_s)
            stats.newest_imu_ns = max(stats.newest_imu_ns, stamp_ns)
            client.send(loc.encode_imu(stamp_ns, imu.gyro, imu.accelerometer))
            stats.imu_samples += 1

    def drain() -> None:
        """Consume the sensor stream and the estimator's answers without judging.

        The health machine is driven only by the publisher's tick; this loop
        feeds, and stops the machine only when the feed itself fails.
        """
        while True:
            try:
                record = platform.sensor_record(0.0)
            except ProbeFailure as error:
                machine.stop(time.monotonic_ns(), f"the sensor stream failed: {error}")
                return
            if record is None:
                try:
                    state = client.poll_state()
                except loc.ProtocolError as error:
                    machine.stop(time.monotonic_ns(), str(error))
                    return
                if state is not None:
                    publisher.offer(state, stats.newest_imu_ns)
                return
            try:
                feed_record(record)
            except loc.ProtocolError as error:
                machine.stop(time.monotonic_ns(), str(error))
                return

    disagreements: list[dict[str, Any]] = []

    def sample_disagreement(phase: str) -> None:
        """H3: the estimator's published position against EKF3's, in the common frame."""
        sample = platform.telemetry()
        if sample.local_position_ned is None or latest_aligned is None:
            return
        estimate = latest_aligned["position_ned_m"]
        deltas = [abs(estimate[i] - sample.local_position_ned[i]) for i in range(3)]
        disagreements.append(
            {
                "phase": phase,
                "at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "ekf3_local_position_ned": list(sample.local_position_ned),
                "published_position_ned": [float(value) for value in estimate],
                "abs_delta_m": deltas,
                "norm_m": sum(delta * delta for delta in deltas) ** 0.5,
            }
        )

    live_blockers: list[str] = []
    shutdown: Any = None
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.startup)
        platform.request_telemetry_streams()
        applied = platform.read_parameters(
            (
                "GPS_TYPE",
                "VISO_TYPE",
                "VISO_DELAY_MS",
                "VISO_QUAL_MIN",
                "FS_EKF_ACTION",
                "EK3_SRC1_POSXY",
                "EK3_SRC1_VELXY",
                "EK3_SRC1_POSZ",
                "EK3_SRC1_YAW",
                "COMPASS_USE",
            ),
            timeout_s=30.0,
            drain=drain,
        )
        writer.write_json("params-applied.json", applied)
        log_lines.append(f"autopilot parameter readback: {applied}")
        publisher.start()

        if not _wait_initialized(machine, drain, settings.pre_arm_wait_s):
            live_blockers.append(
                "the estimator did not initialize inside the pre-arm window (H5); "
                "initialization is from the declared stationary launch interval, and a "
                "degenerate initialization is unavailable-navigation, not a delayed arm"
            )
        else:
            log_lines.append("estimator initialized before arm (H5)")
            control = platform.arm_and_guided(settings.step_timeout_s.flight, drain=drain)
            if control.refused:
                live_blockers.append(
                    f"the autopilot refused Guided flight with GPS off: mode_reached="
                    f"{control.mode_reached}, armed={control.armed}, refusals="
                    f"{list(control.refusals)}; that is a blocker under plan section 11, "
                    "not a reason to re-enable GPS"
                )
            else:
                machine.open_window(time.monotonic_ns())
                for index, waypoint in enumerate(settings.waypoints_local_ned, start=1):
                    platform.send_local_ned(LocalNedTarget(*waypoint))
                    log_lines.append(f"waypoint {index} commanded at local-NED {waypoint}")
                    hold_end = time.monotonic() + settings.hold_per_waypoint_s
                    while time.monotonic() < hold_end:
                        drain()
                        time.sleep(0.05)
                    sample_disagreement(f"hold-{index}")
                log_lines.append("route complete; commanding LAND")
                session.set_mode("LAND")
                drain_deadline = time.monotonic() + 5.0
                while time.monotonic() < drain_deadline:
                    drain()
                    time.sleep(0.05)
    except Exception as error:  # noqa: BLE001 - the run's outer guard records, never swallows
        live_blockers.append(f"the platform failed during the run: {error}")
    finally:
        shutdown = platform.stop()
        publisher.stop()
        client.close()
        _stop_estimator(estimator_process)

    _write_health_events(writer, machine)
    valid_fraction = machine.valid_fraction(time.monotonic_ns())
    log_lines.extend(
        [
            f"pairs fed: {stats.pairs}, imu samples fed: {stats.imu_samples}",
            f"published: {publisher.published}, final health: {machine.state}, "
            f"valid fraction: {valid_fraction:.4f}, adapter resets: {machine.reset_counter}",
            f"shutdown: {shutdown.exits}",
        ]
    )

    if live_blockers:
        for blocker in live_blockers:
            log_lines.append(f"UNRESOLVED: {blocker}")
        _write_log(writer, log_lines)
        return _blocked_unresolved(
            tuple(live_blockers),
            (
                "no bound was relaxed and no truth was fed; the predeclared stop rule "
                "(plan section 11) records the blocker and stops",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {
                "stage_id": STAGE_ID,
                "sensor_mode_label": "sensor-derived",
                "estimator_pin": estimator,
                "published": publisher.published,
                "shutdown": {"exits": shutdown.exits},
            },
            (*writer.artifacts, "preflight.json"),
            SensorMode.SENSOR_DERIVED,
        )

    truth_comparison = {
        "measured": False,
        "reason": (
            "this branch carries no evaluator-truth channel for the estimator frame; the "
            "merged gate's pose records are the truth channel, and until the merge E1 is "
            "not_measured, so the gate cannot pass here"
        ),
        "p95_horizontal_error_m": None,
        "p95_vertical_error_m": None,
        "max_horizontal_error_m": None,
        "max_vertical_error_m": None,
    }
    writer.write_json("truth-comparison.json", truth_comparison)
    disagreement_summary = _summarise_disagreements(
        disagreements,
        bounds_config["disagreement_p95_m"],
        bounds_config["disagreement_max_m"],
    )
    writer.write_json("disagreement.json", disagreement_summary)
    checks = _score(
        machine,
        bounds,
        bounds_config,
        valid_fraction,
        truth_comparison,
        disagreement_summary,
    )
    writer.write_json("checks.json", checks)
    _write_log(writer, log_lines)
    passed = all(check["status"] == "pass" for check in checks)
    manifest = {
        "stage_id": STAGE_ID,
        "sensor_mode_label": "sensor-derived",
        "localization": "resolved" if passed else "unresolved",
        "estimator_pin": estimator,
        "bounds": bounds_config,
        "published": publisher.published,
        "reset_counter": machine.reset_counter,
        "valid_fraction": valid_fraction,
        "params_applied": applied,
        "shutdown": {"exits": shutdown.exits},
    }
    return CommandOutcome(
        status=CommandStatus.COMPLETE,
        gate_status=GateStatus.PASS if passed else GateStatus.FAIL,
        reasons=tuple(
            f"{check['name']}: {check['status']} — {check['detail']}"
            for check in checks
            if check["status"] != "pass"
        )
        or ("all predeclared bounds met on the frozen route",),
        limitations=(
            "E1 is not measurable on this branch: no evaluator-truth channel for the "
            "estimator frame exists outside the merged gate's pose records",
            "reset-signalling agreement (H2) records the adapter's reset counter; the "
            "firmware's posReset count rides the SITL log and is compared at integration",
            "per-camera exposure offsets are not modelled; both eyes share the capture "
            "instant of the step in which they were read",
            DISPATCH_REGISTRATION_NOTE,
        ),
        manifest=manifest,
        artifacts=(*writer.artifacts, "preflight.json"),
        sensor_mode=SensorMode.SENSOR_DERIVED,
    )


def _wait_initialized(
    machine: loc.HealthMachine, drain: Callable[[], None], wait_s: float
) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        drain()
        if machine.state == "healthy":
            return True
        time.sleep(0.05)
    return False


def _summarise_disagreements(
    disagreements: list[dict[str, Any]], p95_bound_m: float, max_bound_m: float
) -> dict[str, Any]:
    norms = sorted(row["norm_m"] for row in disagreements)
    if not norms:
        return {
            "measured": False,
            "reason": "no common-frame samples: the autopilot never reported a local position "
            "while the adapter published none either",
            "bound_p95_m": p95_bound_m,
            "bound_max_m": max_bound_m,
            "samples": disagreements,
        }
    index = min(len(norms) - 1, int(round(0.95 * (len(norms) - 1))))
    return {
        "measured": True,
        "p95_m": norms[index],
        "max_m": norms[-1],
        "bound_p95_m": p95_bound_m,
        "bound_max_m": max_bound_m,
        "samples": disagreements,
    }


def _score(
    machine: loc.HealthMachine,
    bounds: loc.HealthBounds,
    bounds_config: dict[str, Any],
    valid_fraction: float,
    truth_comparison: dict[str, Any],
    disagreement_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """The predeclared bounds, evaluated exactly as frozen. E1 unmeasured fails the gate."""
    checks: list[dict[str, Any]] = []
    if truth_comparison["measured"]:
        checks.append({"name": "E1", "status": "pass", "detail": "measured against evaluator truth"})
    else:
        checks.append({"name": "E1", "status": "fail", "detail": truth_comparison["reason"]})
    gaps = machine.publish_gaps_s
    checks.append(
        {
            "name": "F1/F3",
            "status": "pass"
            if gaps and max(gaps) <= bounds.max_publish_gap_s
            else "fail",
            "detail": (
                f"{len(gaps)} publishes; max gap "
                f"{max(gaps):.3f} s against the {bounds.max_publish_gap_s:.3f} s bound"
                if gaps
                else "no publishes in the scored window"
            ),
        }
    )
    ages = machine.published_state_ages_s
    checks.append(
        {
            "name": "F2",
            "status": "pass"
            if ages and max(ages) <= bounds.published_state_age_max_s
            else "fail",
            "detail": (
                f"published-state age max {max(ages):.3f} s against the "
                f"{bounds.published_state_age_max_s:.3f} s bound"
                if ages
                else "no publishes in the scored window"
            ),
        }
    )
    checks.append(
        {
            "name": "H1",
            "status": "pass" if valid_fraction >= bounds.valid_fraction_min else "fail",
            "detail": f"valid fraction {valid_fraction:.4f} against "
            f"{bounds.valid_fraction_min:.2f}; outages "
            f"{[round(outage, 3) for outage in machine.outages_s]} s",
        }
    )
    if disagreement_summary["measured"]:
        h3_ok = (
            disagreement_summary["p95_m"] <= bounds_config["disagreement_p95_m"]
            and disagreement_summary["max_m"] <= bounds_config["disagreement_max_m"]
        )
        checks.append(
            {
                "name": "H3",
                "status": "pass" if h3_ok else "fail",
                "detail": f"p95 {disagreement_summary['p95_m']:.3f} m, max "
                f"{disagreement_summary['max_m']:.3f} m against "
                f"({bounds_config['disagreement_p95_m']:.2f}, "
                f"{bounds_config['disagreement_max_m']:.2f}) m",
            }
        )
    else:
        checks.append({"name": "H3", "status": "fail", "detail": disagreement_summary["reason"]})
    checks.append(
        {
            "name": "H2/H4",
            "status": "pass" if machine.state == "healthy" else "fail",
            "detail": f"final machine state {machine.state}; events "
            f"{[event.event for event in machine.events]}; adapter resets "
            f"{machine.reset_counter}",
        }
    )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    """The module entry point: dispatch through the shared CLI as ``localize-check``."""
    from embodied.cli import main as cli_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    return cli_main([COMMAND_NAME, *arguments])


register_command(
    COMMAND_NAME,
    _localize_check_command,
    help_text="run the P01-L localization check against the pinned estimator",
    stage_id=STAGE_ID,
    run_prefix=RUN_PREFIX,
    add_arguments=_add_arguments,
)


if __name__ == "__main__":
    sys.exit(main())
