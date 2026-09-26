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
import ast
from bisect import bisect_left
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import re
import socket
import struct
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
    _Optional,
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
    ("GPS1_TYPE", 0.0, "p01l_sensor.parm"),
    ("GPS2_TYPE", 0.0, "p01l_sensor.parm"),
    ("VISO_DELAY_MS", 50.0, "p01l_sensor.parm"),
    ("VISO_QUAL_MIN", 0.0, "p01l_sensor.parm"),
    ("FS_EKF_ACTION", 1.0, "p01l_sensor.parm"),
)
P01L_PARAMS_FILENAME = "p01l_sensor.parm"
VEHICLE_DEFAULT_REQUIREMENTS: tuple[tuple[str, float], ...] = (
    ("EK3_SRC2_POSXY", 0.0),
    ("EK3_SRC2_VELXY", 0.0),
    ("EK3_SRC2_YAW", 0.0),
    ("EK3_SRC3_POSXY", 0.0),
    ("EK3_SRC3_VELXY", 0.0),
    ("EK3_SRC3_YAW", 0.0),
)
# Everything the running vehicle must answer for the claimed arm (plan section 4.7):
# the applied layer's names (G1) and the source-set defaults observed, not assumed (G3).
VEHICLE_REQUIREMENTS: tuple[tuple[str, float, str], ...] = SEAM_REQUIREMENTS + tuple(
    (name, expected, "the firmware's own defaults, observed from the vehicle")
    for name, expected in VEHICLE_DEFAULT_REQUIREMENTS
)

# G2: a name the vehicle refuses is a failure, never an absence. An unknown parameter
# is answered with PARAM_ERROR (msgid 345) carrying MAV_PARAM_ERROR_DOES_NOT_EXIST
# (GCS_Param.cpp:414-421, sent at :502-510). This host's pymavlink predates the
# message -- the recorded answer arrived as UNKNOWN_345 -- so the frame is parsed
# here with a local struct and no new dependency. The wire order is pinned against
# the vehicle's own recorded frame (run-2026-09-26T07-40-00Z/run-a/mavlink.jsonl):
# param_index -1, target_system 250, target_component 190, param_id "GPS_TYPE",
# error 1.
PARAM_ERROR_MSG_ID = 345
MAV_PARAM_ERROR_DOES_NOT_EXIST = 1
_PARAM_ERROR_PAYLOAD = struct.Struct("<hBB16sB")

# G4: the vehicle's runtime answer. SYS_STATUS (msgid 1) carries the GPS-present bit
# (MAV_SYS_STATUS_SENSOR_GPS = 32, common.xml:121) only when a GPS driver is running
# (GCS.cpp:499-506); GPS_RAW_INT (msgid 24) carries fix_type; the driver's own probe
# and detect notices ride STATUSTEXT (GPS_Backend.cpp:136). The streams are requested
# through the session's existing SET_MESSAGE_INTERVAL path and every inbound message
# is recorded by platform.telemetry(), so the whole-run verdict is re-derivable from
# the run's own artifacts.
MSG_ID_SYS_STATUS = 1
MSG_ID_GPS_RAW_INT = 24
GPS_AIDING_SAMPLE_HZ = 2.0
GPS_SENSOR_PRESENT_BIT = 32

# The drain loop folds the recorded telemetry at this cadence: the readback and arming
# sample the stream through their own paths, but the rest of the run would otherwise
# be recorded only incidentally, and a gate about what the vehicle reports during the
# run needs the run's whole window sampled (plan section 4.7).
TELEMETRY_SAMPLE_PERIOD_S = 0.2

# T7 (plan sections 3.6 and 12.6): the scene-admission check. The pinned initializer
# needs at least feat_thresh = 15 trackable features per window
# (InertialInitializer.cpp:115-119) and its tracker hunts with cv::FAST at the pinned
# default threshold 20 with non-max suppression (VioManagerOptions.h:424,
# Grider_GRID.h:125, TrackKLT.cpp:494). A scene that gives that detector nothing makes
# initialization structurally impossible, so the preflight measures the scene's own
# recorded frames instead of letting a run spend its pre-arm window on an initializer
# that cannot fire. Captures are the platform's accepted-run artifacts: P00's
# compatibility gate and this stage's own runs.
SCENE_ADMISSION_CAPTURE_GLOBS = (
    "work/runs/p00-compat/accept-*/run-*/pairs",
    "work/runs/p01-localization/*-*/run-*/pairs",
)
SCENE_ADMISSION_MAX_FRAMES = 24
INITIALIZER_FEATURE_FLOOR = 15
FAST_THRESHOLD = 20

# H5's measured cause, and the two markers that separate "the initializer never
# fired" from "the filter initialised but the pinned readiness accessor never
# became true". VioManager::initialized() is `is_initialized_vio && timelastupdate
# != -1` (VioManager.h:99), and timelastupdate is assigned only at the tail of
# do_feature_propagate_update (VioManager.cpp:651); the zero-velocity updater
# returns early from track_image_and_update before that assignment
# (VioManager.cpp:294), so a stationary start with try_zupt enabled reaches it
# only on a camera frame where ZUPT itself had no bracketing inertial data.
# Recorded from the run's own log instead of inferring the cause from the bound.
INITIALIZER_SUCCESS_MARKER = "[init]: successful initialization"
ZUPT_ACCEPTED_MARKER = "[ZUPT]: accepted"
ZUPT_STARVED_MARKER = "[ZUPT]: There are no IMU data"
# Revision 5 (plan section 12 item 14): the per-frame decision the criterion rests on.
# UpdaterZeroVelocity::try_update prints one disparity line and, when it reaches a
# verdict, one accept/reject line per camera frame (UpdaterZeroVelocity.cpp:231-249).
# The frames it declines are the only ones that reach do_feature_propagate_update,
# where propagate_and_clone makes the clones the readiness accessor waits for
# (VioManager.cpp:299-305, :341, :348) -- so counting them per frame turns plan
# section 0.6 item 5's sufficiency criterion into a measurement rather than an
# argument from the log tail.
ZUPT_DISPARITY_PATTERN = re.compile(
    r"\[ZUPT\]: (passed|failed) disparity \(([-\d.]+) [<>] ([-\d.]+), (\d+) features?\)"
)
ZUPT_VERDICT_PATTERN = re.compile(
    r"\[ZUPT\]: (accepted|rejected) \|v_IinG\| = ([-\d.]+) \(chi2 ([-\d.]+) [<>] ([-\d.]+)\)"
)
ZUPT_VISUAL_PATH_DECISIONS = ("declined_motion", "declined_no_imu")
# The artifact carries a bounded window of decisions, not the whole log: a pre-arm
# window at the declared 10 Hz holds hundreds of frames and the summary counts above
# carry the total.
ZUPT_FRAME_LIMIT = 64

# Revision 4: the run's own bounded static-start capture (plan section 0.3 item 4).
# Same conditions as P00's accept-5 capture -- the vehicle at the declared start
# on the ground -- recorded so the arm gate can measure the configured world
# itself when no hash-matched capture exists yet, and so every later preflight
# has a hash-anchored capture to measure instead of an unprovenanced one.
SCENE_CAPTURE_MAX_FRAMES = 24
SCENE_CAPTURE_MIN_SPACING_S = 0.3
SCENE_CAPTURE_TIMEOUT_S = 30.0

# A1: the pre-arm attitude gate (plan section 0.3 item 5). A frame-map defect is
# a 90- or 180-degree error, not a 5-degree one; the gate converts nothing and
# calibrates nothing -- it refuses the arm naming the per-axis error.
ATTITUDE_GATE_TOLERANCE_DEG = 5.0

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
    # The sensor-derived development route's world, optional because older
    # configurations -- and the compatibility gate's own -- do not carry it. The
    # shared cli.py schema carries the same key (eb6b180); this section
    # overrides that schema for the check's own loader, so the key must live in
    # both. The check runs in this world when present and falls back to
    # scenario.world otherwise (plan section 0.3 item 1).
    "world": _Optional(str),
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


def _pin_evidence(localization: dict[str, Any], root: Path) -> tuple[dict[str, Any], list[str]]:
    """The pin's version evidence, measured on disk rather than quoted (plan section 3).

    A pin that is only asserted is not a pin. This re-derives the pinned tarball's
    sha256 from the bytes on disk, checks that the build log carries the success
    marker the configuration names, and records what the built library and the
    estimator process actually are. The record travels into the preflight beside
    the row it justifies, so a reader of the receipt sees the measurement instead
    of inferring it from the absence of a failure — the standard section 4.6's
    truth-republish gate was re-pointed at, applied to the pin leg.
    """
    estimator = localization["estimator"]
    build_log = root / estimator["build_log"]
    build_text = (
        build_log.read_text(encoding="utf-8", errors="replace") if build_log.is_file() else None
    )
    measured_tarball = _sha256(root / estimator["tarball_path"])
    record = {
        "name": estimator["name"],
        "tag": estimator["tag"],
        "commit": estimator["commit"],
        "tarball_path": estimator["tarball_path"],
        "tarball_sha256_configured": estimator["tarball_sha256"],
        "tarball_sha256_measured": measured_tarball,
        "tarball_matches_configured_pin": measured_tarball == estimator["tarball_sha256"],
        "build_log": estimator["build_log"],
        "build_success_marker": estimator["build_success_marker"],
        "build_marker_present": bool(
            build_text is not None and estimator["build_success_marker"] in build_text
        ),
        "library": {"path": estimator["library"], "sha256": _sha256(root / estimator["library"])},
        "executable": {
            "path": estimator["executable"],
            "sha256": _sha256(root / estimator["executable"]),
        },
    }
    blockers: list[str] = []
    if measured_tarball is None:
        blockers.append(f"the pinned tarball {estimator['tarball_path']} is not on disk")
    elif not record["tarball_matches_configured_pin"]:
        blockers.append(
            f"the pinned tarball's sha256 {measured_tarball} does not match the configured pin "
            f"{estimator['tarball_sha256']}"
        )
    if build_text is None:
        blockers.append(
            f"the pinned estimator has no build log at {estimator['build_log']}; a pin needs "
            "a successful build on this host as version evidence (plan section 3)"
        )
    elif not record["build_marker_present"]:
        blockers.append(
            f"the estimator build log {estimator['build_log']} does not record "
            f"{estimator['build_success_marker']!r}: no successful build of the pinned "
            "estimator tree is evidenced"
        )
    if record["library"]["sha256"] is None:
        blockers.append(f"the built estimator library {estimator['library']} is not on disk")
    return record, blockers


def _pin_summary(record: dict[str, Any]) -> str:
    """One line stating what was measured, for the preflight row's detail."""
    return (
        f"tarball {record['tarball_path']} sha256 {record['tarball_sha256_measured']} matches "
        f"the configured pin; the build log records {record['build_success_marker']!r}; "
        f"library {record['library']['path']} sha256 {record['library']['sha256']}; "
        f"process {record['executable']['path']} sha256 {record['executable']['sha256']}"
    )


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


def _readback_blockers(applied: dict[str, float], refusals: dict[str, int]) -> list[str]:
    """The vehicle's own parameter readback against the claimed arm (plan section 4.7).

    The applied file is what we asked for; this is what the autopilot reported about
    itself, which is the only statement of the configuration actually running. Every
    required name must be answered with the claimed value, and the outcomes are kept
    distinct (G2): answered, refused — the vehicle itself replied that the name does
    not exist, which means the layer that named it never took effect — and silent,
    which is a readback that cannot confirm anything and so confirms nothing.
    """
    blockers: list[str] = []
    for name, expected, _source in VEHICLE_REQUIREMENTS:
        if name in applied:
            if applied[name] != expected:
                blockers.append(
                    f"{name} read back as {applied[name]:g} from the vehicle itself; "
                    f"the claimed arm needs {name} {expected:g}"
                )
        elif name in refusals:
            error = refusals[name]
            error_name = (
                "MAV_PARAM_ERROR_DOES_NOT_EXIST"
                if error == MAV_PARAM_ERROR_DOES_NOT_EXIST
                else str(error)
            )
            blockers.append(
                f"the vehicle refused {name} with PARAM_ERROR {error_name}: a defaults "
                "line under a name the vehicle does not have is silently dropped "
                "(AP_Param.cpp:2421-2431), so the layer that named it did not take effect"
            )
        else:
            blockers.append(
                f"{name} was never answered by the vehicle; a silent readback confirms "
                "nothing and the claimed arm does not run on it"
            )
    return blockers


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_bytes(document: dict[str, Any]) -> bytes | None:
    """The raw frame behind a recorded message dict, whatever shape the recorder kept."""
    data = document.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str) and data.startswith("bytearray(") and data.endswith(")"):
        try:
            parsed = ast.literal_eval(data[len("bytearray(") : -1])
        except (ValueError, SyntaxError):
            return None
        return parsed if isinstance(parsed, bytes) else None
    return None


def _decode_param_error(document: dict[str, Any]) -> dict[str, Any] | None:
    """One PARAM_ERROR, decoded from the vehicle's own frame (plan section 4.7 G2).

    A pymavlink new enough to know the message records it decoded; this host's records
    it as UNKNOWN_345 with the raw frame, which is parsed against the wire layout
    pinned with the constants above.
    """
    kind = str(document.get("mavpackettype", ""))
    if kind == "PARAM_ERROR":
        param_id = str(document.get("param_id", "")).rstrip("\x00")
        index = document.get("param_index")
        error = document.get("error")
        if param_id and index is not None and error is not None:
            return {"param_id": param_id, "param_index": int(index), "error": int(error)}
        return None
    if kind != f"UNKNOWN_{PARAM_ERROR_MSG_ID}":
        return None
    frame = _frame_bytes(document)
    if frame is None or len(frame) < 10 + _PARAM_ERROR_PAYLOAD.size:
        return None
    if frame[7] | (frame[8] << 8) | (frame[9] << 16) != PARAM_ERROR_MSG_ID:
        return None
    param_index, target_system, target_component, raw_id, error = _PARAM_ERROR_PAYLOAD.unpack(
        frame[10 : 10 + _PARAM_ERROR_PAYLOAD.size]
    )
    param_id = raw_id.split(b"\x00")[0].decode("utf-8", errors="replace")
    return {
        "param_id": param_id,
        "param_index": param_index,
        "error": error,
        "target_system": target_system,
        "target_component": target_component,
    }


def _param_error_refusals(mavlink_log: Path) -> dict[str, int]:
    """Every parameter name the vehicle itself refused, from the run's own record."""
    refusals: dict[str, int] = {}
    if not mavlink_log.is_file():
        return refusals
    with mavlink_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(document, dict):
                continue
            decoded = _decode_param_error(document)
            if decoded and decoded["param_index"] == -1 and decoded["param_id"]:
                refusals.setdefault(decoded["param_id"], decoded["error"])
    return refusals


_GPS_STATUSTEXT = re.compile(r"GPS\s*\d*\s*:", re.IGNORECASE)


def _gps_aiding_verdict(mavlink_log: Path) -> dict[str, Any]:
    """What the vehicle reported about GPS over the covered window (plan section 4.7 G4).

    Three signals, each required clean: no SYS_STATUS sample may carry the GPS-present
    bit, no GPS_RAW_INT sample may carry a fix, and no STATUSTEXT may be the driver's
    own probe or detect notice. Zero samples is not a pass: an unsampled claim is an
    asserted-only prerequisite, which this gate exists to remove.
    """
    sys_status = 0
    gps_present = 0
    raw_int = 0
    fix_types: list[int] = []
    statustexts: list[str] = []
    if mavlink_log.is_file():
        with mavlink_log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    document = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(document, dict):
                    continue
                kind = document.get("mavpackettype")
                if kind == "SYS_STATUS":
                    sys_status += 1
                    present = int(document.get("onboard_control_sensors_present") or 0)
                    if present & GPS_SENSOR_PRESENT_BIT:
                        gps_present += 1
                elif kind == "GPS_RAW_INT":
                    raw_int += 1
                    fix_type = document.get("fix_type")
                    if fix_type is not None:
                        fix_types.append(int(fix_type))
                elif kind == "STATUSTEXT":
                    text = str(document.get("text", ""))
                    if _GPS_STATUSTEXT.search(text):
                        statustexts.append(text)
    blockers: list[str] = []
    if sys_status == 0:
        blockers.append(
            "the vehicle's SYS_STATUS was never sampled, so GPS-off is unconfirmed"
        )
    if gps_present:
        blockers.append(
            f"SYS_STATUS carried the GPS-present bit (MAV_SYS_STATUS_SENSOR_GPS) in "
            f"{gps_present} of {sys_status} samples"
        )
    if any(fix_type != 0 for fix_type in fix_types):
        blockers.append(
            f"GPS_RAW_INT reported a fix: fix_type values {sorted(set(fix_types))} over "
            f"{raw_int} samples"
        )
    for text in statustexts:
        blockers.append(f"the GPS driver announced itself in STATUSTEXT: {text!r}")
    return {
        "sys_status_samples": sys_status,
        "sys_status_gps_present": gps_present,
        "gps_raw_int_samples": raw_int,
        "fix_types": sorted(set(fix_types)),
        "statustexts": statustexts,
        "blockers": blockers,
    }


def _params_applied_record(applied: dict[str, float], refusals: dict[str, int]) -> dict[str, Any]:
    """Every required name with its outcome kept distinct (plan section 4.7 G2)."""
    record: dict[str, Any] = {}
    for name, expected, source in VEHICLE_REQUIREMENTS:
        if name in applied:
            record[name] = {
                "outcome": "answered",
                "value": applied[name],
                "expected": expected,
                "source": source,
            }
        elif name in refusals:
            record[name] = {
                "outcome": "refused",
                "param_error": refusals[name],
                "expected": expected,
                "source": source,
            }
        else:
            record[name] = {"outcome": "silent", "expected": expected, "source": source}
    return record


def _find_world_sha256(node: Any) -> str | None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "world_sha256" and isinstance(value, str):
                return value
            found = _find_world_sha256(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_world_sha256(item)
            if found:
                return found
    return None


def _recorded_world_sha256(run_dir: Path) -> str | None:
    """The world hash a capture's own artifacts record, when one is recorded."""
    for path in sorted(run_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        found = _find_world_sha256(document)
        if found:
            return found
    return None


def _platform_settings(document: dict[str, Any], root: Path) -> PlatformSettings:
    """Settings with the sensor-derived development route applied (plan 0.3 item 1).

    ``localization.world``, when present, is the world this check runs in; an
    absent key leaves ``scenario.world`` -- and with it the compatibility gate's
    measured vehicle -- untouched. The shared schema carries the key as optional
    (main eb6b180), so older configurations load unchanged.
    """
    settings = PlatformSettings.from_config(document, root=root)
    world_name = (document.get("localization") or {}).get("world")
    if not world_name:
        return settings
    world = Path(world_name).expanduser()
    if not world.is_absolute():
        world = settings.root / world
    return replace(settings, world=world)


_VEHICLE_TRANSLATION_RE = re.compile(
    r"^\s*translation\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
)


def _declared_start_origin(world: Path) -> tuple[float, float, float]:
    """The configured world's own vehicle translation, as the odom origin (0.3 item 3).

    The odom origin is a property of the world the route flies. The calibration
    referee's declared start cannot serve: it disagrees with the compat world's
    own spawn by 7 cm in z (plan section 13's recorded gap), and it describes the
    compat scene's depth checks, not this route. ``mission.yaml`` pins the
    vehicle node's translation as the spawn ("identical to the Iris translation
    in world.wbt, plan section 6 pin 8"), so the world file is the anchor.
    """
    lines = world.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("Iris {"):
            for candidate in lines[index + 1 : index + 40]:
                match = _VEHICLE_TRANSLATION_RE.match(candidate)
                if match:
                    return (
                        float(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                    )
            break
    raise ConfigError(
        f"{world} declares no Iris vehicle translation; the odom origin cannot be derived "
        "from the scene (plan section 0.3 item 3)"
    )

_VEHICLE_ROTATION_RE = re.compile(
    r"^\s*rotation\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
    r"\s+(-?\d+(?:\.\d+)?)"
)


def _declared_start_attitude(world: Path) -> tuple[float, float, float]:
    """The vehicle's declared start attitude, as NED roll/pitch/yaw (0.3 item 5).

    The declared stationary start is what the epoch rotation is derived from,
    together with the estimator's own first initialized attitude, because the
    odom frame's yaw is unobservable (see ``OdomAlignment.seal``). A pure yaw is
    accepted and converted from Webots' rotation about +z; a tilted start is
    refused rather than approximated, since a wrong declaration would rotate the
    whole published frame. No rotation field means the identity start, which is
    what both declared worlds carry; ``mission.yaml``'s ``spawn_pose.yaw_rad``
    is the independent cross-check.
    """
    lines = world.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("Iris {"):
            for candidate in lines[index + 1 : index + 20]:
                stripped = candidate.strip()
                if stripped.startswith("controllerArgs") or stripped.startswith("children"):
                    break
                match = _VEHICLE_ROTATION_RE.match(candidate)
                if match:
                    x, y, z, angle = (float(value) for value in match.groups())
                    norm = math.sqrt(x * x + y * y + z * z)
                    if norm == 0.0:
                        return (0.0, 0.0, 0.0)
                    axis = (x / norm, y / norm, z / norm)
                    if abs(abs(axis[2]) - 1.0) > 1e-6:
                        raise ConfigError(
                            f"{world} declares a tilted start rotation {match.groups()}; "
                            "the declared start must be level with a pure yaw (plan "
                            "section 0.3 item 5)"
                        )
                    # A right-handed rotation about +z turns north toward west, which
                    # is a negative NED yaw.
                    yaw = -angle if axis[2] > 0.0 else angle
                    return (0.0, 0.0, yaw)
            return (0.0, 0.0, 0.0)
    raise ConfigError(
        f"{world} declares no Iris vehicle node; the start attitude cannot be derived "
        "(plan section 0.3 item 5)"
    )


_SITL_SERIAL_PORT_RE = re.compile(r"^SERIAL(\d+) on TCP port (\d+)$", re.MULTILINE)


def _autopilot_feed_endpoint(sitl_log: Path, session_endpoint: str) -> str:
    """The autopilot link the adapter publishes on (plan section 0.5).

    The pinned SITL serves exactly one TCP client per serial port
    (``UARTDriver.cpp``: a single ``accept()``, then ``_connected``), and this
    check's own session already owns the configured port. A second client on that
    port is accepted by the kernel and never read -- the third textured
    invocation published 2404 poses into exactly such a connection and the
    autopilot reported ``VisOdom: not healthy``. The free port is taken from the
    running SITL's own declaration of what it listens on, which its log records;
    no port is guessed and no convention is assumed.
    """
    session_port = session_endpoint.rsplit(":", 1)[-1]
    if sitl_log.is_file():
        ports = [
            (int(number), int(port))
            for number, port in _SITL_SERIAL_PORT_RE.findall(
                sitl_log.read_text(encoding="utf-8", errors="replace")
            )
            if str(port) != session_port
        ]
        if ports:
            number, port = sorted(ports)[0]
            return f"tcp:127.0.0.1:{port}"
    raise ConfigError(
        f"{sitl_log} declares no free SITL serial port besides {session_endpoint}: the "
        "adapter cannot be given a link the autopilot actually serves, and publishing "
        "into an unserved one is a silent void (plan section 0.5)"
    )

def _fast_keypoint_counts(frame_paths: Sequence[Path]) -> list[int] | None:
    """Measure frames with exactly the detector call the pinned tracker makes.

    Returns None when OpenCV is not importable -- unmeasurable, not zero.
    """
    try:
        import cv2
    except ImportError:
        return None
    detector = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=True)
    counts: list[int] = []
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            counts.append(len(detector.detect(image, None)))
    return counts


def _stereo_capture_measurement(pairs_dir: Path, limit: int) -> dict[str, Any]:
    """Per-eye FAST counts for a recorded capture, or why it cannot gate a stereo arm.

    The gate answers one question — can the pinned tracker's front end see this
    scene — and the arm it gates feeds the estimator two image planes. A capture
    holding only the left eye cannot answer it, and neither can one whose two eyes
    are byte-identical: one view duplicated is not a stereo pair. Both are reported
    as unusable rather than measured. The defect this replaces is a gate that passed
    at 16 left-eye keypoints while nothing in the run recorded what the second view
    contained, so a stereo arm could open on a single eye's worth of evidence.
    """
    left_paths = sorted(path for path in pairs_dir.glob("*-left.ppm") if path.is_file())
    measurement: dict[str, Any] = {
        "pairs_dir": str(pairs_dir),
        "pairs": 0,
        "left_counts": [],
        "right_counts": [],
        "missing_right": [],
        "identical_pairs": [],
        "cv2_unavailable": False,
    }
    pairs: list[tuple[Path, Path]] = []
    for left_path in left_paths:
        right_path = left_path.with_name(left_path.name.replace("-left.ppm", "-right.ppm"))
        if not right_path.is_file():
            measurement["missing_right"].append(left_path.name)
            continue
        if left_path.read_bytes() == right_path.read_bytes():
            measurement["identical_pairs"].append(left_path.name)
            continue
        pairs.append((left_path, right_path))
    measurement["pairs"] = len(pairs)
    if not pairs:
        return measurement
    measured = pairs[:limit]
    left_counts = _fast_keypoint_counts([left for left, _right in measured])
    right_counts = _fast_keypoint_counts([right for _left, right in measured])
    if left_counts is None or right_counts is None:
        measurement["cv2_unavailable"] = True
        return measurement
    measurement["left_counts"] = left_counts
    measurement["right_counts"] = right_counts
    return measurement


def _scene_admission_check(
    settings: PlatformSettings, root: Path
) -> tuple[bool, str, str]:
    """T7: the scene must admit the pinned initializer, measured on its own frames.

    The initializer needs at least 15 trackable features per window and the pinned
    tracker hunts with FAST at the pinned threshold with non-max suppression. The
    newest recorded capture of this scenario is measured with exactly that call.

    Revision 4, two changes and one unchanged property. A capture that records no
    world hash is now skipped rather than measured: it cannot prove which world
    its frames show, and with a development route configured it would measure one
    scene against another world's gate. And when no hash-matched capture of the
    configured world exists, the check defers to the arm gate instead of refusing
    to start: the run records its own static-start frames with the world's hash,
    and the arm is refused unless those measured frames clear the floor -- so no
    flight can be spent on a scene the initializer cannot fire in, which is the
    property that has held since run 4's receipt.

    Revision 5 adds a third: the capture must hold **both eyes**. The arm this gates
    feeds the estimator two planes, and every capture recorded before this revision
    holds only `-left.ppm` -- so the gate passed at 16 left-eye keypoints while
    nothing in the run showed what the second view contained. A capture missing its
    right eye, or whose right eye is byte-identical to its left, is now skipped
    rather than measured, exactly as an unhashed capture is, and the run's own
    arm-gate capture records and measures both.
    """
    capture_dirs: list[Path] = []
    for pattern in SCENE_ADMISSION_CAPTURE_GLOBS:
        capture_dirs.extend(path for path in root.glob(pattern) if path.is_dir())
    current_sha256 = _sha256(settings.world)
    unhashed = 0
    unusable = 0
    for pairs_dir in sorted(capture_dirs, key=lambda path: path.stat().st_mtime, reverse=True):
        if not any(pairs_dir.glob("*-left.ppm")):
            continue
        recorded = _recorded_world_sha256(pairs_dir.parent)
        if recorded is None:
            unhashed += 1
            continue
        if recorded != current_sha256:
            continue
        measurement = _stereo_capture_measurement(pairs_dir, SCENE_ADMISSION_MAX_FRAMES)
        if measurement["cv2_unavailable"]:
            return (
                False,
                "cv2_unavailable",
                "the scene-admission check needs OpenCV (cv2) to measure the scene's "
                "recorded frames the way the pinned tracker does; it is not importable",
            )
        measured_both_eyes = (
            measurement["pairs"]
            and measurement["left_counts"]
            and measurement["right_counts"]
        )
        if not measured_both_eyes:
            unusable += 1
            continue
        lowest_left = min(measurement["left_counts"])
        lowest_right = min(measurement["right_counts"])
        lowest = min(lowest_left, lowest_right)
        passed = lowest >= INITIALIZER_FEATURE_FLOOR
        detail = (
            f"{measurement['pairs']} recorded stereo pairs of {settings.world} measured "
            f"with FAST({FAST_THRESHOLD}, non-max suppression) in both eyes: keypoint "
            f"counts left {measurement['left_counts']}, right "
            f"{measurement['right_counts']}, against the pinned initializer's floor of "
            f"{INITIALIZER_FEATURE_FLOOR} features per window; the capture's recorded "
            "world sha256 matches the configured world"
        )
        return passed, ("measured_pass" if passed else "measured_fail"), detail
    detail = (
        f"no hash-matched recorded stereo capture of {settings.world} (sha256 "
        f"{current_sha256}) exists under the accepted-run artifacts ({unhashed} "
        "capture(s) skipped for recording no world hash, "
        f"{unusable} for recording no complete, distinct, readable stereo pair); the "
        "arm gate will measure this run's own static-start frames in both eyes against "
        f"the pinned initializer's floor of {INITIALIZER_FEATURE_FLOOR} features and "
        "refuse the arm if they do not clear it (plan section 0.3 item 4)"
    )
    return True, "deferred_to_arm_gate", detail


def _zupt_frame_decisions(lines: Sequence[str]) -> list[dict[str, Any]]:
    """The zero-velocity updater's per-frame decision, in the order it made them.

    One record per camera frame the updater was asked about, carrying the numbers it
    decided from: the mean disparity against ``zupt_max_disparity`` with the feature
    count, and, where the frame reached a verdict, the velocity and chi2 against their
    limits. The decisions are the three ways out of ``UpdaterZeroVelocity::try_update``
    the log can show: ``accepted`` (the frame is consumed, UpdaterZeroVelocity.cpp:248),
    ``declined_motion`` (chi2 or velocity over its limit, the frame reaches the visual
    path, :244) and ``declined_no_imu`` (no bracketing inertial data, same consequence,
    :104). A frame whose disparity line was written but whose verdict never followed --
    the log tail, or a process that stopped mid-frame -- keeps ``no_verdict`` rather
    than being silently classified.
    """
    frames: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for line in lines:
        if ZUPT_STARVED_MARKER in line:
            frames.append({"decision": "declined_no_imu"})
            pending = None
            continue
        disparity = ZUPT_DISPARITY_PATTERN.search(line)
        if disparity is not None:
            pending = {
                "disparity_passed": disparity.group(1) == "passed",
                "disparity_px": float(disparity.group(2)),
                "max_disparity_px": float(disparity.group(3)),
                "feature_count": int(disparity.group(4)),
            }
            continue
        verdict = ZUPT_VERDICT_PATTERN.search(line)
        if verdict is not None:
            frames.append(
                {
                    **(pending or {}),
                    "decision": (
                        "accepted" if verdict.group(1) == "accepted" else "declined_motion"
                    ),
                    "velocity_m_s": float(verdict.group(2)),
                    "chi2": float(verdict.group(3)),
                    "chi2_limit": float(verdict.group(4)),
                }
            )
            pending = None
    if pending is not None:
        frames.append({**pending, "decision": "no_verdict"})
    return frames


def _initializer_diagnostics(estimator_log: Path) -> dict[str, Any]:
    """What the estimator itself said about initialization, from its own log.

    The pinned initializer prints nothing while its feature database is empty — the
    silent return at InertialInitializer.cpp:86-88 — so an empty initializer_output
    behind consumed frames means the scene's pixels gave the front end nothing to
    track, which the preflight's scene_admission check measures directly.

    Revision 5 adds the per-frame decisions themselves: the summary counts say how
    many frames went each way, and ``zupt_frames`` carries the numbers each decision
    was made from, so the sufficiency criterion of plan section 0.6 item 5 (the
    accessor needs five frames that ZUPT declined before its first completed visual
    update can write ``timelastupdate``) is measurable per frame from the artifact.
    """
    lines: list[str] = []
    if estimator_log.is_file():
        lines = estimator_log.read_text(encoding="utf-8", errors="replace").splitlines()
    frames = _zupt_frame_decisions(lines)
    return {
        "estimator_log_lines": len(lines),
        "initializer_output": [line for line in lines if "[init" in line][-50:],
        "progress_reports": [line for line in lines if "initialized=" in line][-12:],
        "initializer_succeeded": any(
            INITIALIZER_SUCCESS_MARKER in line for line in lines
        ),
        "zupt_accepted_updates": sum(
            1 for line in lines if ZUPT_ACCEPTED_MARKER in line
        ),
        "zupt_frames_without_imu": sum(
            1 for line in lines if ZUPT_STARVED_MARKER in line
        ),
        "zupt_rejected_updates": sum(
            1 for frame in frames if frame["decision"] == "declined_motion"
        ),
        "zupt_frames_reaching_visual_path": sum(
            1 for frame in frames if frame["decision"] in ZUPT_VISUAL_PATH_DECISIONS
        ),
        "zupt_frames": frames[-ZUPT_FRAME_LIMIT:],
        "note": (
            "the pinned initializer prints nothing while its feature database is empty; "
            "an empty initializer_output behind consumed frames means the scene gave "
            "the front end nothing to track. initializer_succeeded says whether the "
            "initializer's own success line is present; initialized() also needs "
            "timelastupdate, set only by a completed visual update, which the "
            "zero-velocity updater pre-empts (VioManager.cpp:294). "
            "zupt_frames_reaching_visual_path counts the frames ZUPT declined -- the "
            "only ones that reach do_feature_propagate_update and can make a clone"
        ),
    }


def _initialization_blocker(diagnostics: dict[str, Any]) -> str:
    """H5's stop reason, stated from the estimator's own log (plan section 11).

    Two different failures read the same from the bound alone, and they point a
    later session in opposite directions: an initializer that never fired is a
    scene/feature question, while an initializer that fired behind a readiness
    accessor that stayed false is a filter-configuration question. The first
    textured re-run's receipt said "did not initialize" while its own
    initializer_output recorded the success line, and the next plan revision was
    written against the wrong cause; this function is what stops that reading.
    """
    if diagnostics["initializer_succeeded"]:
        return (
            "the pinned estimator's readiness accessor stayed false through the whole "
            "pre-arm window (H5), although the initializer itself fired: its own log "
            f"records {INITIALIZER_SUCCESS_MARKER!r}, while VioManager::initialized() "
            "(VioManager.h:99) is `is_initialized_vio && timelastupdate != -1` and "
            "timelastupdate is assigned only at the tail of do_feature_propagate_update "
            "(VioManager.cpp:651). The zero-velocity updater returned early from "
            "track_image_and_update before that assignment (VioManager.cpp:294): "
            f"{diagnostics['zupt_accepted_updates']} accepted zero-velocity update(s), "
            f"{diagnostics['zupt_rejected_updates']} frame(s) declined on motion and "
            f"{diagnostics['zupt_frames_without_imu']} frame(s) where it had no "
            "bracketing inertial data, so "
            f"{diagnostics['zupt_frames_reaching_visual_path']} frame(s) reached the "
            "visual path, against the five the accessor needs before its first "
            "completed update can write timelastupdate (VioManager.cpp:348-352). No "
            "state was published, so no bound was measured; changing the "
            "zero-velocity configuration or the H5 criterion is the owner's "
            "disposition, not a worker's"
        )
    return (
        "the estimator did not initialize inside the pre-arm window (H5): the pinned "
        "initializer's own log records no successful initialization, and initialization "
        "is from the declared stationary launch interval -- a degenerate initialization "
        "is unavailable-navigation, not a delayed arm"
    )

# ---------------------------------------------------------------------------
# Preflight: everything the claimed arm needs, reported in one pass
# ---------------------------------------------------------------------------
def _preflight(
    document: dict[str, Any], output_dir: Path, mode: SensorMode
) -> tuple[list[dict[str, Any]], bool]:
    """Every prerequisite, reported in one pass. Nothing is started by this function.

    Row order is the order a reader needs: the arm's own declaration first, then the
    gate that decides whether truth can reach the estimate, then whether the scene can
    feed the pinned estimator at all, then the files and selections the claimed arm
    applies.
    """
    root = repository_root()
    settings = _platform_settings(document, root)
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
    scene_ok, scene_state, scene_detail = _scene_admission_check(settings, root)
    rows.append(
        {
            "name": "scene_admission",
            "satisfied": scene_ok,
            "state": scene_state,
            "detail": scene_detail,
        }
    )
    satisfied = satisfied and scene_ok
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
    pin_record, pin_blockers = _pin_evidence(localization, root)
    rows.append(
        {
            "name": "estimator_pin",
            "satisfied": not pin_blockers,
            "detail": pin_blockers[0] if pin_blockers else _pin_summary(pin_record),
            "evidence": pin_record,
        }
    )
    satisfied = satisfied and not pin_blockers
    for blocker in (
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
        # A1's evidence channel: the same pose records' attitudes, read for the
        # pre-arm attitude gate and sent nowhere (plan section 0.3 item 5).
        self.truth_attitudes: list[tuple[int, tuple[float, float, float]]] = []
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
    settings = _platform_settings(document, root)
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
    alignment_origin = _declared_start_origin(settings.world)
    declared_start_rpy = _declared_start_attitude(settings.world)
    alignment = loc.OdomAlignment(alignment_origin, declared_start_rpy)
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

    scene_capture = {"count": 0, "last_s": 0.0}

    def file_record(record: Any) -> None:
        if record.kind is not Kind.PAIR or record.pair is None:
            return
        stats.pair_records_filed += 1
        try:
            pending_pairs.put_nowait(record)
        except queue.Full:
            stats.pair_records_dropped += 1
        # The run's own scene capture (plan section 0.3 item 4): bounded pairs at
        # the declared static start, with the world's own pixels — the same
        # conditions as P00's accept-5 capture, written so the arm gate and every
        # later preflight measure the configured world instead of trusting a
        # capture that cannot say which world it shows. Both eyes are written, and
        # from the same pair record the feed converts and sends, so the gate
        # measures exactly the two planes the estimator is given: the captures
        # recorded before this revision held only the left frame, and the gate
        # passed on one eye while nothing recorded what the second one contained.
        now = time.monotonic()
        if (
            scene_capture["count"] < SCENE_CAPTURE_MAX_FRAMES
            and now - scene_capture["last_s"] >= SCENE_CAPTURE_MIN_SPACING_S
        ):
            scene_capture["last_s"] = now
            index = scene_capture["count"] + 1
            header = f"P6\n{settings.stereo.width} {settings.stereo.height}\n255\n".encode()
            writer.write_bytes(
                f"pairs/{index:05d}-left.ppm", header + bytes(record.pair.left_bytes)
            )
            writer.write_bytes(
                f"pairs/{index:05d}-right.ppm", header + bytes(record.pair.right_bytes)
            )
            scene_capture["count"] += 1

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
        f"odom origin (the world's own vehicle translation, ENU): {alignment_origin}, "
        f"declared start attitude rpy: {declared_start_rpy}",
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
            stats.truth_attitudes.append(
                (sim_time_ns(record.sim_time_s), tuple(record.pose.attitude_rpy))
            )

    last_telemetry_sample = 0.0
    def drain() -> None:
        """Consume the sensor stream and the estimator's answers without judging.

        The health machine is driven only by the publisher's tick; this loop
        feeds, and stops the machine only when the feed itself fails.
        """
        nonlocal last_telemetry_sample
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
                if time.monotonic() - last_telemetry_sample >= TELEMETRY_SAMPLE_PERIOD_S:
                    last_telemetry_sample = time.monotonic()
                    # G4's evidence is the run's own record: folding the telemetry here
                    # records every inbound MAVLink message through the whole window,
                    # not only the windows the readback and arming happen to sample.
                    try:
                        platform.telemetry()
                    except ProbeFailure as error:
                        machine.stop(
                            time.monotonic_ns(), f"the telemetry stream failed: {error}"
                        )
                        return
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

    def gate_the_scored_arm(applied: dict[str, float], refusals: dict[str, int]) -> list[str]:
        """Sections 4.6 and 4.7, as observation: what was sent, and what the vehicle reports.

        Every item is read from the things that acted — the bridge's own count of
        simulator poses it has sent, the autopilot's own parameter answers and
        refusals, and the vehicle's own GPS status streams — so none of it is a
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
        blockers.extend(_readback_blockers(applied, refusals))
        gps = _gps_aiding_verdict(writer.path("mavlink.jsonl"))
        log_lines.append(
            f"GPS-off at the gate: {gps['sys_status_samples']} SYS_STATUS samples, "
            f"{gps['gps_raw_int_samples']} GPS_RAW_INT samples, "
            f"{len(gps['statustexts'])} GPS STATUSTEXTs recorded so far"
        )
        blockers.extend(gps["blockers"])
        return blockers

    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.startup)
        # The adapter's own link: a port the running SITL declares it serves and
        # that this check's session does not already own (plan section 0.5).
        feed_endpoint = _autopilot_feed_endpoint(
            writer.path("sitl.log"), settings.mavlink_endpoint
        )
        publisher.retarget(feed_endpoint)
        log_lines.append(f"adapter publish endpoint: {feed_endpoint}")
        platform.request_telemetry_streams()
        # G4 needs the vehicle's own GPS status during the run, so the two status
        # streams are requested through the session's existing interval path; every
        # inbound message is recorded by platform.telemetry() either way.
        for message_id in (MSG_ID_SYS_STATUS, MSG_ID_GPS_RAW_INT):
            session.request_message_interval(message_id, GPS_AIDING_SAMPLE_HZ)
        # Read one parameter at a time. The shared reader waits for a whole batch, so a
        # single name the autopilot never answers would consume the deadline and leave
        # the later names unrequested -- a readback that cannot say whether it asked.
        # Asked individually, every name gets its own chance and its own answer or its
        # own absence, which is exactly what the gate records.
        applied: dict[str, float] = {}
        for name, _expected, _source in VEHICLE_REQUIREMENTS:
            applied.update(
                platform.read_parameters((name,), timeout_s=PARAMETER_READ_TIMEOUT_S, drain=drain)
            )
        refusals = _param_error_refusals(writer.path("mavlink.jsonl"))
        writer.write_json("params-applied.json", _params_applied_record(applied, refusals))
        log_lines.append(
            f"autopilot parameter readback: {applied}; refused by the vehicle: {sorted(refusals)}"
        )
        live_blockers.extend(gate_the_scored_arm(applied, refusals))
        try:
            publisher.start()
        except RuntimeError as error:
            # An unserved link: the autopilot never answered on this port, so
            # publishing would be a silent void (plan section 0.5). Named, not
            # ignored -- the third textured invocation's measured failure mode.
            live_blockers.append(str(error))
            log_lines.append(f"UNRESOLVED: {error}")

        scene_blocker = _scene_capture_gate(writer, settings, drain, scene_capture, log_lines)
        if scene_blocker:
            live_blockers.append(scene_blocker)
        initialized_in_window = _wait_initialized(machine, drain, settings.pre_arm_wait_s)
        if not initialized_in_window:
            live_blockers.append(
                _initialization_blocker(_initializer_diagnostics(writer.path("estimator.log")))
            )
        else:
            log_lines.append("estimator initialized before arm (H5)")
            a1_blocker = _attitude_gate(writer, latest_aligned, alignment, stats, log_lines)
            if a1_blocker:
                live_blockers.append(a1_blocker)
        if live_blockers:
            # A pre-arm gate failed (sections 4.6, 4.7, or revision 4's scene and
            # attitude gates). The arm stops here, before the scored window opens,
            # so the flight is not spent: nothing is substituted to get past it.
            log_lines.append("not arming: a precondition of the scored arm failed")
        else:
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
    gps_aiding = _gps_aiding_verdict(writer.path("mavlink.jsonl"))
    writer.write_json("gps-aiding.json", gps_aiding)
    writer.write_json(
        "initializer-diagnostics.json",
        _initializer_diagnostics(writer.path("estimator.log")),
    )
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
                "gps_aiding_blockers": gps_aiding["blockers"],
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
        gps_aiding,
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

def _scene_capture_gate(
    writer: EvidenceWriter,
    settings: PlatformSettings,
    drain: Callable[[], None],
    scene_capture: dict[str, Any],
    log_lines: list[str],
) -> str | None:
    """Revision 4's arm-side scene admission (plan section 0.3 item 4), both eyes.

    The preflight measured a hash-matched capture when one existed; this gate
    measures the run's own recorded static-start frames, with the world's own
    hash written beside them, and refuses the arm when they do not clear the
    initializer's floor. No flight can be spent on a scene the initializer
    cannot fire in.

    Revision 5 measures the pair, not the left frame: both eyes must clear the
    floor, and a capture with no complete, distinct pair cannot open the arm at
    all. The estimator this arm feeds receives two planes, so one eye's keypoint
    count is not evidence about it.
    """
    deadline = time.monotonic() + SCENE_CAPTURE_TIMEOUT_S
    while scene_capture["count"] < SCENE_CAPTURE_MAX_FRAMES and time.monotonic() < deadline:
        drain()
        time.sleep(0.05)
    pairs_dir = writer.directory / "pairs"
    measurement = _stereo_capture_measurement(pairs_dir, SCENE_ADMISSION_MAX_FRAMES)
    record: dict[str, Any] = {
        "world": str(settings.world),
        "world_sha256": _sha256(settings.world),
        "pairs_recorded": measurement["pairs"],
        "left_keypoint_counts": measurement["left_counts"],
        "right_keypoint_counts": measurement["right_counts"],
        "missing_right": measurement["missing_right"],
        "identical_pairs": measurement["identical_pairs"],
        "feature_floor": INITIALIZER_FEATURE_FLOOR,
        "fast_threshold": FAST_THRESHOLD,
    }
    if measurement["cv2_unavailable"]:
        record["state"] = "cv2_unavailable"
        writer.write_json("scene-capture.json", record)
        return (
            "the scene-admission gate could not measure this run's own frames: OpenCV "
            "(cv2) is not importable"
        )
    if not measurement["pairs"]:
        record["state"] = "no_stereo_pair"
        writer.write_json("scene-capture.json", record)
        return (
            "the scene-admission gate refuses the arm: this run recorded no complete "
            "stereo pair (both eyes present and not the same image twice), so nothing "
            "in its own evidence says what the second view feeds the estimator "
            f"(missing right frames {len(measurement['missing_right'])}, identical "
            f"pairs {len(measurement['identical_pairs'])})"
        )
    if not measurement["left_counts"] or not measurement["right_counts"]:
        record["state"] = "unreadable"
        writer.write_json("scene-capture.json", record)
        return (
            "the scene-admission gate refuses the arm: this run's recorded pairs could "
            "not be read back as images, so neither eye could be measured"
        )
    lowest_left = min(measurement["left_counts"])
    lowest_right = min(measurement["right_counts"])
    passed = min(lowest_left, lowest_right) >= INITIALIZER_FEATURE_FLOOR
    record["state"] = "measured_pass" if passed else "measured_fail"
    writer.write_json("scene-capture.json", record)
    log_lines.append(
        f"scene admission at the arm gate: {measurement['pairs']} stereo pairs measured, "
        f"keypoints left {measurement['left_counts']}, right "
        f"{measurement['right_counts']}, floor {INITIALIZER_FEATURE_FLOOR}"
    )
    if passed:
        return None
    worst_eye = "left" if lowest_left <= lowest_right else "right"
    return (
        f"the configured scene's own recorded frames carry "
        f"{min(lowest_left, lowest_right)} FAST keypoints in the {worst_eye} eye "
        f"(left {lowest_left}, right {lowest_right}) against the pinned initializer's "
        f"floor of {INITIALIZER_FEATURE_FLOOR} (plan section 0.3 item 4): the stereo "
        "stream this front end is given cannot initialize, so the arm is refused"
    )

def _wrap_angle(radius: float) -> float:
    return (radius + math.pi) % (2.0 * math.pi) - math.pi
def _attitude_gate(
    writer: EvidenceWriter,
    aligned: dict[str, object] | None,
    alignment: loc.OdomAlignment,
    stats: _FeedStats,
    log_lines: list[str],
) -> str | None:
    """A1, second form: the sealed epoch frame, checked before the arm (0.3 item 5).

    The first textured invocation measured a −180° roll (the missing FLU→FRD body
    map) and the second a +90° yaw (the odom frame's yaw, unobservable and chosen
    by the initializer's noise-decided gram_schmidt branch). Both were refused
    before any arm, and the adapter now derives its epoch rotation from the
    estimator's own first initialized attitude and the world's declared start
    attitude. A1 therefore checks the two things that remain independently
    checkable: that the published attitude equals the *declared* start attitude
    (the composition the adapter performed), and that the declared start attitude
    equals the simulator's own truth attitude (the declaration itself). It also
    gates the vertical chain — the sealed rotation must carry odom-up to NED-down
    — and records the derived yaw, which is not gated because nothing in this arm
    observes it. That the seal absorbs the estimator's own initialization tilt is
    a recorded limitation, not a hidden one: a real tilt error shows up in E1 and
    H3 once the vehicle moves.
    """
    declared = tuple(alignment.declared_start_rpy)
    record: dict[str, Any] = {
        "tolerance_deg": ATTITUDE_GATE_TOLERANCE_DEG,
        "declared_start_rpy_rad": list(declared),
    }
    if aligned is None or not stats.truth_attitudes:
        reason = (
            "no published state to compare"
            if aligned is None
            else "no truth attitude arrived on the pose records"
        )
        record.update({"state": "unmeasured", "reason": reason})
        writer.write_json("attitude-gate.json", record)
        return f"A1 could not run: {reason}"
    published = aligned["attitude_rpy"]
    truth_time_ns, truth_rpy = stats.truth_attitudes[-1]
    composition_deg = [
        math.degrees(_wrap_angle(p - d)) for p, d in zip(published, declared)
    ]
    declaration_deg = [math.degrees(_wrap_angle(d - t)) for d, t in zip(declared, truth_rpy)]
    odom_up_down_component = float(alignment.epoch_rotation[2][2])
    level_error_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, -odom_up_down_component)))
    )
    passed = (
        max(abs(delta) for delta in composition_deg) <= ATTITUDE_GATE_TOLERANCE_DEG
        and max(abs(delta) for delta in declaration_deg) <= ATTITUDE_GATE_TOLERANCE_DEG
        and level_error_deg <= ATTITUDE_GATE_TOLERANCE_DEG
    )
    record.update(
        {
            "state": "measured_pass" if passed else "measured_fail",
            "published_rpy_rad": list(published),
            "truth_rpy_rad": list(truth_rpy),
            "truth_time_ns": truth_time_ns,
            "composition_deltas_deg": composition_deg,
            "declaration_deltas_deg": declaration_deg,
            "epoch_yaw_deg": alignment.epoch_yaw_deg(),
            "level_error_deg": level_error_deg,
            "note": (
                "epoch_yaw_deg is recorded, not gated: nothing in this arm observes the "
                "odom frame's yaw, which the pinned initializer fixes by a "
                "noise-decided branch; the seal derives it once from the declared start"
            ),
        }
    )
    writer.write_json("attitude-gate.json", record)
    log_lines.append(
        f"A1 attitude gate: composition deltas (deg) {composition_deg}, declaration "
        f"deltas (deg) {declaration_deg}, epoch yaw {record['epoch_yaw_deg']:.2f} deg, "
        f"level error {level_error_deg:.3f} deg"
    )
    if passed:
        return None
    return (
        f"A1: the sealed epoch frame does not hold up: composition deltas "
        f"{composition_deg} deg, declaration deltas {declaration_deg} deg, level error "
        f"{level_error_deg:.3f} deg against the declared "
        f"{ATTITUDE_GATE_TOLERANCE_DEG} deg tolerance (plan section 0.3 item 5), so the "
        "arm is refused"
    )


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
    gps_aiding: dict[str, Any],
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
    gps_ok = not gps_aiding["blockers"]
    checks.append(
        {
            "name": "G4",
            "status": "pass" if gps_ok else "fail",
            "detail": (
                f"GPS-off confirmed at runtime: {gps_aiding['sys_status_samples']} "
                "SYS_STATUS samples without the GPS-present bit, "
                f"{gps_aiding['gps_raw_int_samples']} GPS_RAW_INT samples without a "
                f"fix, no GPS driver STATUSTEXT"
                if gps_ok
                else "; ".join(gps_aiding["blockers"])
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
