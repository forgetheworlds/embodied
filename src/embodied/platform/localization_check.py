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
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from embodied.cli import (
    COMMAND_REGISTRY,
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
    "this module self-registers on import and is also listed in embodied.cli "
    "COMMAND_MODULES, so both `python -m embodied localize-check` and the module entry "
    "point work: the self-registration is idempotent because running the module as an "
    "entry point imports it twice, once as __main__ and once under its package name"
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

# How many stereo pairs the reader's sink may hold for the feed. A pair is ~614 KB of
# pixels at the declared 640x480, and the sink is called from the reader's thread while
# the drain loop feeds the estimator, so a small bound keeps memory predictable while
# a healthy stream never fills it: at the declared 10 Hz this is 1.6 s of frames, and a
# backlog that deep is a stopped estimator, which the health machine is already about
# to catch.
PAIR_QUEUE_FRAMES = 16

# How long one parameter readback waits for the autopilot's own answer. A local SITL
# answers in well under a second; this is generous enough that an answer would have to
# be absent rather than slow, which is the distinction the gate depends on.
PARAMETER_READ_TIMEOUT_S = 5.0

# The additive localization section: the pin's identity, the publish cadence,
# the declared pipeline delay, the parameter layer, and the predeclared bounds.
# The bounds are the plan's values (plan section 6), frozen here before any
# measurement; they are never adjusted after seeing a run.
LOCALIZATION_SECTION: dict[str, Any] = {
    "mode": str,
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
        # E1, against evaluator truth: plan-fixed values, the upper edge of the
        # advisory band, fitted to the smallest declared opening (plan section 6).
        "error_p95_horizontal_m": float,
        "error_p95_vertical_m": float,
        "error_max_horizontal_m": float,
        "error_max_vertical_m": float,
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
        f"the estimator process {localization['estimator']['executable']} does not exist; "
        "estimator/ov_stream.cpp is built by estimator/build-openvins.sh against the pinned "
        "tarball, and without that binary there is no estimator process to run"
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
                f"the estimator parameter layer {path} is missing; the claimed arm applies "
                "exactly the layers scenario.estimator_params names, so a missing one is a "
                "selection that did not happen"
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

def _mode_blockers(document: dict[str, Any], mode: SensorMode) -> list[str]:
    """The arm the configuration declares and the arm asked for must be the same arm.

    The bridge's truth republish follows ``localization.mode``, so a configuration that
    declares one arm while the invocation asks for another would run a mode it did not
    declare.
    """
    declared = document["localization"].get("mode")
    if declared == mode.value:
        return []
    return [
        f"the configuration declares localization.mode {declared!r} but this invocation "
        f"asks for {mode.value!r}; the bridge's truth republish follows the configuration, "
        "so the two must agree before anything starts"
    ]


def _truth_republish_blockers(settings: PlatformSettings) -> list[str]:
    """Whether the bridge that will be built republishes the simulator's own pose (4.6).

    Read from the settings object the bridge's own ``from_config`` produced, so this is
    the switch's actual value and not prose about it. With it on, the bridge's vision
    feed would send simulator pose to the autopilot's external-navigation source while
    the estimator's adapter publishes to that same source: a scored arm would fly on
    truth it must not receive.
    """
    if not settings.truth_republish:
        return []
    return [
        "the bridge's truth republish is ON (PlatformSettings.truth_republish is True): its "
        "vision feed would send simulator pose to the autopilot's external-navigation "
        "source, which the estimator's adapter also publishes to, so a scored arm would "
        "receive truth it must not (plan section 4.6)"
    ]


def _readback_blockers(applied: dict[str, float]) -> list[str]:
    """The vehicle's own parameter readback against the claimed arm (plan section 4.6).

    The applied file is what we asked for; this is what the autopilot reported about
    itself, which is the only statement of the configuration actually running. GPS_TYPE
    is the load-bearing entry — the synthesized GPS is simulator-truth-derived, so a run
    that still has it on is not the claimed arm — and the rest of the seam is checked in
    the same pass because they fail the same way.
    """
    unmet = [
        f"{name} read back as {applied.get(name)!r} from the vehicle itself"
        for name, expected, _source in SEAM_REQUIREMENTS
        if applied.get(name) != expected
    ]
    if not unmet:
        return []
    return [
        "the vehicle's own parameter readback does not match the claimed arm: "
        + "; ".join(unmet)
    ]

def _preflight(
    document: dict[str, Any], output_dir: Path, mode: SensorMode
) -> tuple[list[dict[str, Any]], bool]:
    """Every prerequisite, reported in one pass. Nothing is started by this function.

    Row order is the order a reader needs: the arm's own declaration first, then the
    gate that decides whether truth can reach the estimate, then the files and
    selections the claimed arm applies.
    """
    root = repository_root()
    settings = PlatformSettings.from_config(document, root=root)
    rows: list[dict[str, Any]] = []
    satisfied = True
    mode_blockers = _mode_blockers(document, mode)
    rows.append(
        {
            "name": "localization_mode",
            "satisfied": not mode_blockers,
            "detail": mode_blockers[0]
            if mode_blockers
            else f"the configuration declares localization.mode {mode.value}, the arm asked for",
        }
    )
    satisfied = satisfied and not mode_blockers
    truth_blockers = _truth_republish_blockers(settings)
    rows.append(
        {
            "name": "bridge_truth_republish",
            "satisfied": not truth_blockers,
            "detail": truth_blockers[0]
            if truth_blockers
            else "the bridge's truth republish is off: the autopilot's external-navigation "
            "source has exactly one publisher, the estimator's adapter",
        }
    )
    satisfied = satisfied and not truth_blockers
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
                "the pose-assisted diagnostic arm is not built by this stage: driving the "
                "adapter from the gate's truth pose needs a configuration declaring "
                "localization.mode pose-assisted, so that the bridge republishes the truth "
                f"pose; the configuration declares {document['localization'].get('mode')!r}",
                *_mode_blockers(document, mode),
            ),
        )

    rows, satisfied = _preflight(document, output_dir, mode)
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
    """What the estimator feed consumed, and what truth was read beside it.

    ``truth_samples`` is the evaluator-truth channel: the controller's own pose samples
    arrive on the same sensor stream as the pairs and the inertial samples, and the feed
    writes them here for scoring while sending them nowhere. The estimator receives
    stereo and inertial frames and nothing else, which is the whole point of keeping the
    two lists side by side in one object.
    """

    def __init__(self) -> None:
        self.pairs = 0
        self.imu_samples = 0
        self.pair_latencies_ns: list[int] = []
        self.imu_latencies_ns: list[int] = []
        self.newest_imu_ns = 0
        self.truth_samples: list[tuple[int, tuple[float, float, float]]] = []
        # Pairs the reader's sink offered with their pixels, and how many of those the
        # feed's bounded queue could not take. The two together are the difference
        # between "the stream carried no pairs" and "the feed was too slow".
        self.pair_records_filed = 0
        self.pair_records_dropped = 0


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


def _write_environment(
    writer: EvidenceWriter, settings: PlatformSettings, estimator: dict[str, Any]
) -> None:
    """Which tree and which interpreter this run actually measured (plan section 9).

    A stage can be run from a worktree while the installed package resolves to a
    different one; that is not a detail a reader should have to infer from a stray
    failure, so the resolved module paths are part of the run's record.
    """
    import embodied
    from embodied.platform import webots_ardupilot as bridge

    writer.write_json(
        "environment.json",
        {
            "interpreter": sys.executable,
            "python_version": sys.version,
            "embodied_module": str(Path(embodied.__file__).resolve()),
            "bridge_module": str(Path(bridge.__file__).resolve()),
            "repository_root": str(repository_root()),
            "estimator_executable": estimator["executable"],
            "truth_republish": settings.truth_republish,
            "sensor_mode": settings.sensor_mode.value,
        },
    )


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
    _write_environment(writer, settings, estimator)
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
                "no bound was relaxed and no truth was fed to the estimator; the predeclared "
                "stop rule (plan section 11) records the blocker and stops",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {"stage_id": STAGE_ID, "sensor_mode_label": "sensor-derived", "estimator_pin": estimator},
            (*writer.artifacts, "preflight.json"),
            SensorMode.SENSOR_DERIVED,
        )

    client = loc.OvStreamClient("127.0.0.1", int(estimator["socket_port"]))
    try:
        client.connect()
    except (OSError, loc.ProtocolError) as error:
        _stop = (
            f"the adapter could not connect to the estimator on 127.0.0.1:"
            f"{estimator['socket_port']}: {error}"
        )
        _stop_estimator(estimator_process)
        _write_log(writer, [f"UNRESOLVED: {_stop}"])
        return _blocked_unresolved(
            (_stop,),
            (
                "no bound was relaxed and no truth was fed to the estimator; the predeclared "
                "stop rule (plan section 11) records the blocker and stops",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {"stage_id": STAGE_ID, "sensor_mode_label": "sensor-derived", "estimator_pin": estimator},
            (*writer.artifacts, "preflight.json"),
            SensorMode.SENSOR_DERIVED,
        )
    feed_log_path = writer.path("estimator-feed.jsonl")
    stats = _FeedStats()
    latest_aligned: dict[str, object] | None = None
    # The scored window's publications, kept for E1. Only samples inside arm-to-disarm
    # count: bring-up publishes are reported by the freshness accounting and charged to
    # nothing, exactly as the machine's own window does.
    published_states: list[tuple[int, tuple[float, float, float]]] = []
    scored_window_open = False
    # Pixels reach a reader's sink before the handoff queue strips them: the queue
    # carries metadata only, so a stereo pair arrives there as a kind with no planes.
    # The sink therefore keeps whole pair records for the feed, in a bounded queue
    # that this one thread drains, so every write to the estimator socket comes from
    # the drain loop and frames cannot interleave.
    pending_pairs: queue.Queue = queue.Queue(maxsize=PAIR_QUEUE_FRAMES)

    def file_record(record: Any) -> None:
        if record.kind is not Kind.PAIR or record.pair is None:
            return
        stats.pair_records_filed += 1
        try:
            pending_pairs.put_nowait(record)
        except queue.Full:
            stats.pair_records_dropped += 1

    def on_publish(state: loc.EstimatorState, aligned: dict[str, object]) -> None:
        nonlocal latest_aligned
        latest_aligned = aligned
        if scored_window_open:
            published_states.append((state.time_ns, tuple(aligned["position_ned_m"])))
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
    # Pixels only ever arrive through the sink: the reader strips them before the
    # handoff queue, which carries metadata so a queue of 614 KB frames cannot grow
    # unbounded. Without this line a stereo pair reaches the feed as a kind with no
    # planes, and the estimator is fed nothing.
    platform.record_sink = file_record
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
        elif record.kind is Kind.POSE and record.pose is not None:
            # Evaluator truth, read for scoring and sent nowhere. The estimator's feed
            # handles PAIR and IMU only, so this branch cannot reach it; the sample is
            # already in ArduPilot's NED frame, which is the frame the published state is
            # converted into, so E1 is a subtraction in one common frame.
            stats.truth_samples.append(
                (sim_time_ns(record.sim_time_s), tuple(record.pose.position_xyz))
            )

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
                # The metadata stream is momentarily empty, so the inertial samples
                # either side of a queued image have been fed: the sink's pairs can go
                # now, which keeps every image behind the IMU that must precede it.
                try:
                    while True:
                        feed_record(pending_pairs.get_nowait())
                except queue.Empty:
                    pass
                except loc.ProtocolError as error:
                    machine.stop(time.monotonic_ns(), str(error))
                    return
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

    def gate_the_scored_arm(applied: dict[str, float]) -> list[str]:
        """Section 4.6, as observation: what the bridge sent, and what the vehicle reports.

        Both are read from the things that acted — the bridge's own count of simulator
        poses it has sent, and the autopilot's own parameter readback — so neither is a
        promise about behaviour. A failure here stops the arm before the scored window
        opens, which is the only place it can be stopped without spending a flight.
        """
        blockers: list[str] = []
        truth_published = platform.truth_feed_published
        log_lines.append(
            f"bridge truth poses sent: {truth_published} "
            f"(republish declared {settings.truth_republish})"
        )
        if truth_published:
            blockers.append(
                f"the bridge sent {truth_published} simulator poses before this arm was "
                "scored: the truth republish is not off, so the estimator's input and the "
                "autopilot's external-navigation source would both carry truth (plan "
                "section 4.6)"
            )
        blockers.extend(_readback_blockers(applied))
        return blockers

    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.startup)
        platform.request_telemetry_streams()
        # Read one parameter at a time. The shared reader waits for a whole batch, so a
        # single name the autopilot never answers would consume the deadline and leave
        # the later names unrequested -- a readback that cannot say whether it asked.
        # Asked individually, every name gets its own chance and its own answer or its
        # own absence, which is exactly what the gate records.
        applied: dict[str, float] = {}
        for name, _expected, _source in SEAM_REQUIREMENTS:
            applied.update(
                platform.read_parameters((name,), timeout_s=PARAMETER_READ_TIMEOUT_S, drain=drain)
            )
        writer.write_json("params-applied.json", applied)
        log_lines.append(f"autopilot parameter readback: {applied}")
        live_blockers.extend(gate_the_scored_arm(applied))
        publisher.start()

        if not _wait_initialized(machine, drain, settings.pre_arm_wait_s):
            live_blockers.append(
                "the estimator did not initialize inside the pre-arm window (H5); "
                "initialization is from the declared stationary launch interval, and a "
                "degenerate initialization is unavailable-navigation, not a delayed arm"
            )
        elif live_blockers:
            # A section-4.6 gate failed above. The arm stops here, before the scored window
            # opens, so the flight is not spent: nothing is substituted to get past it.
            log_lines.append("not arming: a precondition of the scored arm failed")
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
                scored_window_open = True
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
    truth_published = platform.truth_feed_published
    log_lines.extend(
        [
            f"pairs fed: {stats.pairs}, imu samples fed: {stats.imu_samples}, "
            f"truth pose samples read: {len(stats.truth_samples)}",
            f"pair records the reader filed with pixels: {stats.pair_records_filed}, "
            f"dropped by a full feed queue: {stats.pair_records_dropped}",
            f"published: {publisher.published}, final health: {machine.state}, "
            f"valid fraction: {valid_fraction:.4f}, adapter resets: {machine.reset_counter}",
            f"bridge truth poses sent over the whole run: {truth_published}",
            f"scored-window publications compared against truth: {len(published_states)}",
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
                "no bound was relaxed and no truth was fed to the estimator; the predeclared "
                "stop rule (plan section 11) records the blocker and stops",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {
                "stage_id": STAGE_ID,
                "sensor_mode_label": "sensor-derived",
                "estimator_pin": estimator,
                "truth_republish": settings.truth_republish,
                "bridge_truth_published": truth_published,
                "pairs_filed": stats.pair_records_filed,
                "pairs_fed": stats.pairs,
                "imu_samples_fed": stats.imu_samples,
                "published": publisher.published,
                "shutdown": {"exits": shutdown.exits},
            },
            (*writer.artifacts, "preflight.json"),
            SensorMode.SENSOR_DERIVED,
        )

    truth_comparison = _truth_error_statistics(
        published_states, stats.truth_samples, bounds_config
    )
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
        truth_published,
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
        "truth_republish": settings.truth_republish,
        "bridge_truth_published": truth_published,
        "truth_samples_read": len(stats.truth_samples),
        "pairs_filed": stats.pair_records_filed,
        "pairs_fed": stats.pairs,
        "imu_samples_fed": stats.imu_samples,
        "published": publisher.published,
        "scored_publications": len(published_states),
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
            "E1 compares the scored window's publications against the controller's pose "
            "stream, which is read in this process for scoring only and sent to nothing: "
            "the estimator's feed carries stereo pairs and inertial samples and no other "
            "record. A publication with no truth sample inside the join tolerance is "
            "counted as unjoined rather than interpolated",
            "a passing E1 is bounded by this route and this scene: two waypoints with 8 s "
            "holds indoors, never a general navigation claim",
            "reset-signalling agreement (H2) records the adapter's reset counter; the "
            "firmware's posReset count rides the SITL log and is compared at integration",
            "per-camera exposure offsets are not modelled; both eyes share the capture "
            "instant of the step in which they were read",
            "the floor-plane metric-depth gap P01-C recorded at grazing incidence is not "
            "repaired by this run and its vertical channel does not inherit that honesty",
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

def _percentile(values: Sequence[float], fraction: float) -> float:
    """The nearest-rank percentile of an unsorted sample, order-statistic honest at any n."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _truth_error_statistics(
    published: Sequence[tuple[int, Sequence[float]]],
    truth: Sequence[tuple[int, Sequence[float]]],
    bounds_config: dict[str, Any],
) -> dict[str, Any]:
    """E1: the published state against evaluator truth, joined on simulator time.

    Both sides are already in local NED — the adapter converts the estimator's odom
    frame through the fixed alignment, and the controller converts the Webots devices
    exactly as it does for the flight-state packet — so the comparison is a subtraction
    in one common frame. Truth is read for scoring and reaches nothing else: the
    estimator's feed carries stereo pairs and inertial samples.

    A publication is joined to the nearest truth sample and used only if that sample is
    inside the tolerance; anything further apart is counted as unjoined rather than
    interpolated, because an interpolated truth value is not a measurement.
    """
    bounds = {
        "bound_p95_horizontal_m": bounds_config["error_p95_horizontal_m"],
        "bound_p95_vertical_m": bounds_config["error_p95_vertical_m"],
        "bound_max_horizontal_m": bounds_config["error_max_horizontal_m"],
        "bound_max_vertical_m": bounds_config["error_max_vertical_m"],
    }
    join_tolerance_ns = 100_000_000
    if not published or not truth:
        return {
            "measured": False,
            "reason": (
                f"E1 has nothing to compare: {len(published)} scored publications and "
                f"{len(truth)} truth samples were read"
            ),
            **bounds,
        }
    truth_times = [time_ns for time_ns, _position in truth]
    horizontal: list[float] = []
    vertical: list[float] = []
    unjoined = 0
    for time_ns, estimate in published:
        index = bisect_left(truth_times, time_ns)
        candidates = [truth[i] for i in (index - 1, index) if 0 <= i < len(truth)]
        nearest_time_ns, nearest_position = min(
            candidates, key=lambda sample: abs(sample[0] - time_ns)
        )
        if abs(nearest_time_ns - time_ns) > join_tolerance_ns:
            unjoined += 1
            continue
        horizontal.append(
            math.hypot(estimate[0] - nearest_position[0], estimate[1] - nearest_position[1])
        )
        vertical.append(abs(estimate[2] - nearest_position[2]))
    if not horizontal:
        return {
            "measured": False,
            "reason": (
                f"E1 has no joined samples: {len(published)} scored publications, every one "
                f"further than {join_tolerance_ns / 1e6:.0f} ms from a truth sample"
            ),
            "scored_publications": len(published),
            "truth_samples": len(truth),
            "unjoined": unjoined,
            **bounds,
        }
    return {
        "measured": True,
        "scored_publications": len(published),
        "truth_samples": len(truth),
        "joined_samples": len(horizontal),
        "unjoined": unjoined,
        "join_tolerance_ms": join_tolerance_ns / 1e6,
        "p95_horizontal_error_m": _percentile(horizontal, 0.95),
        "p95_vertical_error_m": _percentile(vertical, 0.95),
        "max_horizontal_error_m": max(horizontal),
        "max_vertical_error_m": max(vertical),
        **bounds,
    }


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
    truth_published: int,
) -> list[dict[str, Any]]:
    """The predeclared bounds, evaluated exactly as frozen. E1 unmeasured fails the gate."""
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "bridge_truth_republish",
            "status": "pass" if truth_published == 0 else "fail",
            "detail": (
                "the bridge sent no simulator pose for the whole run: the autopilot's "
                "external-navigation source had exactly one publisher, the adapter"
                if truth_published == 0
                else f"the bridge sent {truth_published} simulator poses during the run"
            ),
        }
    )
    if truth_comparison["measured"]:
        measured = (
            truth_comparison["p95_horizontal_error_m"],
            truth_comparison["p95_vertical_error_m"],
            truth_comparison["max_horizontal_error_m"],
            truth_comparison["max_vertical_error_m"],
        )
        limit = (
            bounds_config["error_p95_horizontal_m"],
            bounds_config["error_p95_vertical_m"],
            bounds_config["error_max_horizontal_m"],
            bounds_config["error_max_vertical_m"],
        )
        met = all(value <= bound for value, bound in zip(measured, limit))
        checks.append(
            {
                "name": "E1",
                "status": "pass" if met else "fail",
                "detail": (
                    f"over {truth_comparison['joined_samples']} joined samples "
                    f"({truth_comparison['unjoined']} unjoined): p95 horizontal "
                    f"{measured[0]:.3f} m, p95 vertical {measured[1]:.3f} m, max horizontal "
                    f"{measured[2]:.3f} m, max vertical {measured[3]:.3f} m against "
                    f"({limit[0]:.2f}, {limit[1]:.2f}, {limit[2]:.2f}, {limit[3]:.2f}) m"
                ),
            }
        )
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


def _register_once() -> None:
    """Register the command unless this stage already registered it.

    Running this module as an entry point executes it twice: once as ``__main__``,
    and again under its package name when ``build_parser`` imports the dispatch
    list. Plain registration raises on the second import and takes the command down
    before it parses its own arguments, so the registration is idempotent for this
    stage's own spec — and still refuses to stand down for anyone else's.
    """
    existing = COMMAND_REGISTRY.get(COMMAND_NAME)
    if existing is not None:
        if existing.stage_id == STAGE_ID:
            return
        raise CommandError(
            f"command {COMMAND_NAME!r} is already registered by stage {existing.stage_id}"
        )
    register_command(
        COMMAND_NAME,
        _localize_check_command,
        help_text="run the P01-L localization check against the pinned estimator",
        stage_id=STAGE_ID,
        run_prefix=RUN_PREFIX,
        add_arguments=_add_arguments,
    )


_register_once()


if __name__ == "__main__":
    sys.exit(main())
