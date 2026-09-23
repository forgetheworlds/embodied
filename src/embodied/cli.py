"""Command dispatch, configuration loading and the shared receipt.

Three things live here because every command must do them identically: the
mapping from a run's outcome to a process exit code, the JSON receipt that records
what happened, and the loader for the one configuration file a command reads.
Command handlers live in the module that owns the work and register themselves
here, so a later stage adds behaviour without editing a shared parser.

Exit codes, from CLI-PLAN:

* ``0`` the command produced a valid complete result. This says nothing about
  whether the drone succeeded: a completed losing episode is valid data.
* ``1`` a command error, with diagnostics retained.
* ``2`` blocked by a missing prerequisite or budget.
* ``3`` pending a mandatory adjudication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from embodied.contracts.records import SensorMode

RECEIPT_VERSION = "receipt-1"
RECEIPT_FILENAME = "receipt.json"
MANIFEST_FILENAME = "manifest.json"


class CommandStatus(str, Enum):
    """What happened. ``complete`` is about the command, not about mission success."""

    COMPLETE = "complete"
    BLOCKED = "blocked"
    INVALID = "invalid"
    PENDING = "pending"


class GateStatus(str, Enum):
    """Whether the stage's own acceptance condition was met."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


# The one place a status becomes an exit code.
EXIT_CODES: dict[CommandStatus, int] = {
    CommandStatus.COMPLETE: 0,
    CommandStatus.INVALID: 1,
    CommandStatus.BLOCKED: 2,
    CommandStatus.PENDING: 3,
}


class CommandError(Exception):
    """The command could not do its work. The message is retained as the reason."""


@dataclass(frozen=True)
class CommandOutcome:
    """What a command reports back to the dispatcher.

    ``artifacts`` are paths relative to the run directory. The dispatcher hashes
    them so a reader can tell whether the file in front of them is the file the
    receipt describes. ``manifest`` is the configuration identity: what was
    configured, not what was claimed.
    """

    status: CommandStatus
    gate_status: GateStatus
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    episode_id: str | None = None
    trial_group_id: str | None = None
    sensor_mode: SensorMode | None = None

    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


CommandHandler = Callable[[argparse.Namespace, Path], CommandOutcome]
ArgumentDeclaration = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class CommandSpec:
    """A registered command: how to parse it, what to call, which stage it serves."""

    name: str
    help_text: str
    stage_id: str
    # The stem of the default output directory, which CLI-PLAN names per command
    # ("work/runs/p00-compat-...") rather than per stage.
    run_prefix: str
    handler: CommandHandler
    add_arguments: ArgumentDeclaration | None


COMMAND_REGISTRY: dict[str, CommandSpec] = {}

# Modules that register commands. They are imported when the parser is built, not
# at import time, so a command module can import this one for the registry without
# creating a cycle.
COMMAND_MODULES = ("embodied.platform.webots_ardupilot",)


def register_command(
    name: str,
    handler: CommandHandler,
    *,
    help_text: str,
    stage_id: str,
    run_prefix: str,
    add_arguments: ArgumentDeclaration | None = None,
) -> None:
    """Register one command. Later stages call this; they never edit the parser."""
    if name in COMMAND_REGISTRY:
        raise CommandError(f"command {name!r} is already registered")
    COMMAND_REGISTRY[name] = CommandSpec(
        name=name,
        help_text=help_text,
        stage_id=stage_id,
        run_prefix=run_prefix,
        handler=handler,
        add_arguments=add_arguments,
    )


class _Parser(argparse.ArgumentParser):
    """An argument parser that reports usage problems as command errors.

    argparse exits with status 2 for bad usage, and this project uses 2 for
    "blocked by a missing prerequisite". A mistyped flag must not be able to
    borrow that meaning.
    """

    def error(self, message: str) -> None:
        raise CommandError(message)


def build_parser() -> argparse.ArgumentParser:
    for module_name in COMMAND_MODULES:
        importlib.import_module(module_name)
    parser = _Parser(prog="embodied", description="Continuous multimodal drone pilot tools.")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for spec in sorted(COMMAND_REGISTRY.values(), key=lambda item: item.name):
        subparser = subparsers.add_parser(
            spec.name, help=spec.help_text, description=spec.help_text
        )
        subparser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="directory for the receipt, manifest and artifacts "
            "(default: a fresh directory under work/runs)",
        )
        if spec.add_arguments is not None:
            spec.add_arguments(subparser)
        subparser.set_defaults(_spec=spec)
    return parser


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(CommandError):
    """The configuration file is missing, unreadable or does not match the schema."""


class _Choice:
    """A leaf that must be one of a small set of names."""

    def __init__(self, *values: str) -> None:
        self.values = values


# The configuration schema. A section is a dict of expected keys, a leaf is a
# Python type, a sequence of one type is a one-element list. Unknown keys at the
# top level are rejected: a silently ignored section is a substituted experiment.
CONFIG_SCHEMA: dict[str, Any] = {
    "project": {"stage": str, "name": str},
    "platform": {
        "ardupilot_root": str,
        "ardupilot_commit": str,
        "sitl_binary": str,
        "webots_home": str,
        "webots_version": str,
        "webots_mode": _Choice("realtime", "fast"),
        "sim_model": str,
        "vehicle": str,
        "sitl_home": str,
        "endpoints": {"sitl": str, "fdm_port": int, "controller_port": int},
    },
    "scenario": {
        "world": str,
        "params": [str],
        "estimator_params": [str],
    },
    "sensors": {
        "stereo": {
            "left": str,
            "right": str,
            "width": int,
            "height": int,
            "baseline_m": float,
            "sampling_period_ms": int,
            "encoding": str,
            # The declared criterion for calling a frame colour: a frame whose three
            # channels agree on at least this fraction of pixels carries no colour.
            "identical_channel_fraction_limit": float,
        },
        "imu": {
            "accelerometer": str,
            "gyro": str,
            "inertial_unit": str,
            "gps": str,
            "sampling_period_ms": int,
        },
        "calibration": str,
    },
    "probe": {
        "hover_altitude_m": float,
        "waypoints_local_ned": [[float]],
        "hold_per_waypoint_s": float,
        "stream_loss_window_s": float,
        # Simulated time the scene needs to settle, simulated time the at-rest
        # measurement covers once it has, and how long the probe waits for the
        # autopilot's own pre-arm checks to clear before it asks the vehicle to arm.
        "settle_s": float,
        "at_rest_window_s": float,
        "pre_arm_wait_s": float,
        "estimator_fault": {"kind": str, "magnitude_m": float, "hold_s": float},
        "timebase_samples": int,
        # The window the simulator's rate is recorded in, and the envelope it has to stay
        # inside: both are declared engineering parameters of this stage, carried in the
        # configuration rather than hidden in the check.
        "realtime_window_s": float,
        "realtime_ratio_envelope": [float],
        "timebase_poll_s": float,
        "step_timeout_s": {"startup": float, "ready": float, "flight": float},
        "budget_wall_clock_s": float,
    },
    "output": str,
}


def load_config(path: Path) -> dict[str, Any]:
    """Read and validate the experiment configuration.

    Only ``yaml.safe_load`` is used, and the result must match
    :data:`CONFIG_SCHEMA`. Every problem names the key it came from, because a
    configuration mistake that is reported as "invalid config" costs the reader a
    debugging session.
    """
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - PyYAML is a pinned dependency
        raise ConfigError(f"PyYAML is required to read {path}: {error}") from error

    try:
        text = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error
    try:
        document = yaml.safe_load(text.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{path} is not readable YAML: {error}") from error
    if not isinstance(document, dict):
        raise ConfigError(f"{path} must hold a mapping at the top level")
    _validate(document, CONFIG_SCHEMA, path.name)
    return document


def config_hash(path: Path) -> str | None:
    """SHA-256 of the configuration bytes exactly as they were read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _validate(value: Any, schema: Any, where: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{where} must be a mapping")
        unknown = sorted(set(value) - set(schema))
        if unknown:
            raise ConfigError(f"{where} has unknown keys: {', '.join(unknown)}")
        for key, sub_schema in schema.items():
            if key not in value:
                raise ConfigError(f"{where}.{key} is required")
            _validate(value[key], sub_schema, f"{where}.{key}")
        return
    if isinstance(schema, list):
        if value is None or isinstance(value, (str, bytes)) or not isinstance(value, list):
            raise ConfigError(f"{where} must be a list")
        if value and isinstance(schema[0], list):
            for index, item in enumerate(value):
                _validate(item, schema[0], f"{where}[{index}]")
            return
        element = schema[0]
        for index, item in enumerate(value):
            _validate(item, element, f"{where}[{index}]")
        return
    if isinstance(schema, _Choice):
        if value not in schema.values:
            raise ConfigError(f"{where} must be one of {', '.join(schema.values)}")
        return
    if isinstance(schema, type):
        if schema is float:
            # An integer in a float field is the same number; accept it rather
            # than make the author write 2.0 for a count of two seconds.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{where} must be a number")
            return
        if schema is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{where} must be an integer")
            return
        if schema is str:
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{where} must be a non-empty string")
            return
    raise ConfigError(f"{where} uses a schema entry this loader cannot check")


# ---------------------------------------------------------------------------
# Receipt and manifest
# ---------------------------------------------------------------------------


def repository_root() -> Path:
    """The checkout this package is installed from."""
    return Path(__file__).resolve().parents[2]


def default_output_directory(run_prefix: str) -> Path:
    """A fresh directory under the ignored ``work/runs`` tree, named for the run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repository_root() / "work" / "runs" / f"{run_prefix}-{stamp}-{secrets.token_hex(2)}"


def code_revision(root: Path) -> str:
    """``git rev-parse HEAD``, with a dirty flag when the tree has local edits."""
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-revision"
    return f"{head}-dirty" if dirty else head


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _artifact_entries(output: Path, relatives: Sequence[str]) -> list[dict[str, Any]]:
    entries = []
    for relative in relatives:
        path = output / relative
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return entries


def write_artifacts(
    output: Path,
    outcome: CommandOutcome,
    *,
    spec: CommandSpec,
    argv: Sequence[str],
    config_hash_value: str | None,
    started_monotonic_s: float,
    started_at_utc: str,
) -> tuple[Path, Path]:
    """Write ``manifest.json`` and ``receipt.json`` for one command."""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(outcome.manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    artifacts = tuple(dict.fromkeys((*outcome.artifacts, MANIFEST_FILENAME)))
    finished_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "stage_id": spec.stage_id,
        "code_revision": code_revision(repository_root()),
        "config_hash": config_hash_value,
        "status": outcome.status.value,
        "gate_status": outcome.gate_status.value,
        "episode_id": outcome.episode_id,
        "trial_group_id": outcome.trial_group_id,
        "sensor_mode": outcome.sensor_mode.value if outcome.sensor_mode else None,
        "artifacts": _artifact_entries(output, artifacts),
        "reasons": list(outcome.reasons),
        "limitations": list(outcome.limitations),
        "command": [str(argument) for argument in argv],
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "started_at_monotonic_s": round(started_monotonic_s, 6),
        "finished_at_monotonic_s": round(
            started_monotonic_s + (time.monotonic() - started_monotonic_s), 6
        ),
    }
    receipt_path = output / RECEIPT_FILENAME
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt_path, manifest_path


def _summarise(outcome: CommandOutcome, receipt_path: Path) -> str:
    lines = [f"status: {outcome.status.value}  gate: {outcome.gate_status.value}"]
    lines += [f"  reason: {reason}" for reason in outcome.reasons]
    lines += [f"  limitation: {limitation}" for limitation in outcome.limitations]
    lines.append(f"  receipt: {receipt_path}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run one command, write its receipt and return an exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    started_monotonic_s = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        parser = build_parser()
        args = parser.parse_args(arguments)
        spec: CommandSpec | None = getattr(args, "_spec", None)
        if spec is None:
            raise CommandError("no command given; run 'python -m embodied --help'")
    except CommandError as error:
        print(f"embodied: {error}", file=sys.stderr)
        return EXIT_CODES[CommandStatus.INVALID]

    output = (
        Path(args.output)
        if args.output is not None
        else default_output_directory(spec.run_prefix)
    )
    if (output / RECEIPT_FILENAME).exists():
        print(
            f"embodied: {output / RECEIPT_FILENAME} already exists; "
            "choose another --output rather than overwrite a run's record",
            file=sys.stderr,
        )
        return EXIT_CODES[CommandStatus.INVALID]

    config_argument = getattr(args, "config", None)
    config_hash_value = config_hash(Path(config_argument)) if config_argument is not None else None

    try:
        outcome = spec.handler(args, output)
    except CommandError as error:
        outcome = CommandOutcome(
            status=CommandStatus.INVALID,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(str(error),),
        )
    except KeyboardInterrupt:
        outcome = CommandOutcome(
            status=CommandStatus.INVALID,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(f"interrupted after {time.monotonic() - started_monotonic_s:.1f}s",),
        )

    try:
        receipt_path, _ = write_artifacts(
            output,
            outcome,
            spec=spec,
            argv=["python", "-m", "embodied", *arguments],
            config_hash_value=config_hash_value,
            started_monotonic_s=started_monotonic_s,
            started_at_utc=started_at_utc,
        )
    except OSError as error:
        print(f"embodied: cannot write the receipt in {output}: {error}", file=sys.stderr)
        return EXIT_CODES[CommandStatus.INVALID]

    print(_summarise(outcome, receipt_path))
    return outcome.exit_code()
