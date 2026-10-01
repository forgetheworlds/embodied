"""The live recording transport: a registered suite becomes a scorable episode.

``bench record`` refused for as long as no transport was attached to a
registered suite. This module is that transport, and it is deliberately the
only place where the three surfaces of an episode meet:

* the **agent stream** (:class:`embodied.bench.recorder.Recorder`) is written
  by the mission runtime as it flies — mission, observations, selections,
  goals, setpoints, executions and the final report, all in one clock domain;
* the **bench-side store** (:class:`embodied.bench.referee.Referee`) is a
  sibling directory outside the episode, written once and only by this
  module: the scenario's hidden world state from the suite's truth seed, and
  the per-run physical outcome this transport measured from the simulator's
  own pose stream;
* the **episode directory** itself, which the runtime side is handed and which
  can never contain a bench-private member (the projection in
  :mod:`embodied.bench.recorder` is the only vocabulary).

Truth isolation here is structural, not a convention: the mission runtime
never reads a POSE record, and it is handed only ``sensor_tap`` — a callback
that hands the record *out*. The truth poses go to :class:`TruthCollector`,
which lives in this bench-side module; the runtime holds no reference to it
and the episode directory never holds its output.

Physical predicates (declared engineering parameters, R2; the grader compares
claim against these, and it is the aircraft's own reported state plus the
simulator's pose stream that decide them — never the report):

* ``inspected`` — the aircraft's truth position sustained within
  ``INSPECT_RADIUS_M`` of the target's declared true position for at least
  ``INSPECT_HOLD_S``. The radius is the planner's own stand-off (1.0 m) plus
  the declared tracking envelope; passing through one point is not an
  inspection, hence the hold.
* ``return_verified`` — the last truth sample after landing within
  ``RETURN_RADIUS_M`` horizontally of the first truth sample (the spawn).
* ``violations`` — the aircraft's own reports of a crash-disarm, and any
  observed loss of Guided flight while armed.
* ``takeover`` — an operator intervention, which this arm never records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import subprocess
from typing import Any

import yaml

from embodied.cli import (
    CommandOutcome,
    CommandStatus,
    GateStatus,
    code_revision,
    repository_root,
)
from embodied.contracts.records import ClockStamp, SensorMode
from embodied.bench import recorder as recorder_module
from embodied.bench import referee as referee_module
from embodied.bench.recorder import Recorder
from embodied.platform.localization_check import (
    _load_localization_config,
    _platform_settings,
)

SUITE_ID = "first-indoor"
SUITE_CONFIG_RELATIVE = Path("configs/suites/first-indoor.yaml")
PLATFORM_CONFIG_RELATIVE = Path("configs/first_indoor.yaml")
TRUTH_SEED_RELATIVE = Path("scenarios/first_indoor/truth.yaml")
# The one arm this transport admits. B1/B2 are P06's comparison arms, gated on
# a recorded budget cap; refusing them here is the arm's own admission rule.
ADMITTED_ARMS = ("B0",)

# Declared physical predicates (R2). Justifications are in the module docstring.
INSPECT_RADIUS_M = 1.5
INSPECT_HOLD_S = 1.0
RETURN_RADIUS_M = 1.0

# The host gate (BASELINE-GUARD rows 16-18): a freeze-blocked run is not
# evidence, so a loaded host blocks the run before the simulator is started.
HOST_LOAD_1M_MAX = 10.0
HOST_SWAP_FREE_MIN_MB = 500.0
# Every port a live run needs. The platform's own prerequisite check covers the
# configured endpoints; these are checked here too because a port already held
# by another process fails the SITL's own bind with a message that arrives as a
# crash rather than as the prerequisite it is (measured on this transport's
# first attempt: 5760 was held and SITL exited 1 before any flight).
LIVE_PORTS = (9002, 9003, 5760, 9010, 9021)


class SuiteConfigError(Exception):
    """The suite's configuration exists but cannot be used as declared."""


# ---------------------------------------------------------------------------
# Suite registration (conditional on the suite's own config existing)
# ---------------------------------------------------------------------------


def suite_config_path(root: Path | None = None, *, relative: Path | None = None) -> Path:
    return (root or repository_root()) / (relative or SUITE_CONFIG_RELATIVE)


def load_suite_document(path: Path | None = None) -> dict[str, Any] | None:
    """The suite's declaration, or None when the suite has not landed yet."""
    config_path = Path(path) if path is not None else suite_config_path()
    if not config_path.is_file():
        return None
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SuiteConfigError(f"{config_path} does not hold a suite declaration object")
    for key in ("name", "registered_by", "localization_mode", "provider_budget"):
        if key not in document:
            raise SuiteConfigError(f"{config_path} is missing the required key {key!r}")
    return document


def register_suite(document: dict[str, Any] | None) -> str | None:
    """Register the suite from its own declaration; idempotent within a process.

    Registration is conditional on the declaration existing: a build in which
    the suite has not landed keeps refusing ``bench record --suite
    first-indoor`` for the honest reason that nothing is registered, rather
    than recording something the suite never declared.
    """
    if document is None:
        return None
    name = str(document["name"])
    existing = recorder_module.SUITE_REGISTRY.get(name)
    if existing is not None:
        if existing.registered_by == str(document["registered_by"]):
            return name
        raise SuiteConfigError(
            f"suite {name!r} is already registered by {existing.registered_by!r}"
        )
    budget = document["provider_budget"]
    if not isinstance(budget, dict) or not budget:
        raise SuiteConfigError(f"{name!r} declares no provider budget")
    recorder_module.register_suite(
        recorder_module.SuiteSpec(
            name=name,
            registered_by=str(document["registered_by"]),
            provider_budget={str(key): float(value) for key, value in budget.items()},
            localization_mode=str(document["localization_mode"]),
        )
    )
    return name


# ---------------------------------------------------------------------------
# The host gate
# ---------------------------------------------------------------------------

_LOAD_RE = re.compile(r"load averages?:\s*([\d.]+)")
_SWAP_RE = re.compile(r"total = ([\d.]+)M\s+used = ([\d.]+)M\s+free = ([\d.]+)M")


def host_state() -> dict[str, Any]:
    """The 1-minute load and the swap figures a flight is gated on."""
    state: dict[str, Any] = {"load_1m": None, "swap_total_mb": None, "swap_free_mb": None}
    try:
        text = subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout
        match = _LOAD_RE.search(text)
        if match:
            state["load_1m"] = float(match.group(1))
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        text = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
        ).stdout
        match = _SWAP_RE.search(text)
        if match:
            state["swap_total_mb"] = float(match.group(1))
            state["swap_free_mb"] = float(match.group(3))
    except (OSError, subprocess.CalledProcessError):
        pass
    return state


def port_blockers(ports: tuple[int, ...] = LIVE_PORTS) -> list[str]:
    """Every declared port that is already held, named with the check that failed."""
    import socket

    blockers: list[str] = []
    for port in ports:
        for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            probe = socket.socket(socket.AF_INET, kind)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                blockers.append(f"port {port} is already in use")
                break
            finally:
                probe.close()
    return blockers


def host_blockers(state: dict[str, Any]) -> list[str]:
    """The reasons the host cannot fly, measured rather than assumed.

    An unmeasured figure is not a pass: if ``uptime`` or ``sysctl`` answered
    nothing, the blocker names the missing measurement and the run does not
    start.
    """
    blockers: list[str] = []
    load = state.get("load_1m")
    if load is None:
        blockers.append("the 1-minute load could not be measured (uptime gave no figure)")
    elif load >= HOST_LOAD_1M_MAX:
        blockers.append(
            f"1-minute load {load:.2f} is not in single digits (BASELINE-GUARD row 17)"
        )
    free = state.get("swap_free_mb")
    if free is None:
        blockers.append("the swap free figure could not be measured (sysctl gave no figure)")
    elif free < HOST_SWAP_FREE_MIN_MB:
        blockers.append(
            f"swap free {free:.0f} MB is under the declared {HOST_SWAP_FREE_MIN_MB:.0f} MB "
            "margin (BASELINE-GUARD row 16)"
        )
    return blockers


# ---------------------------------------------------------------------------
# The bench-side truth collector and the physical outcome
# ---------------------------------------------------------------------------


@dataclass
class TruthSample:
    sim_time_s: float
    position_ned: tuple[float, float, float]


@dataclass
class TruthCollector:
    """The simulator's pose stream, read only by the bench side.

    Fed by ``sensor_tap`` on the feed thread; appended under the GIL. Nothing
    the agent side holds references this object.
    """

    samples: list[TruthSample] = field(default_factory=list)

    def __call__(self, record: Any) -> None:
        pose = getattr(record, "pose", None)
        if pose is None:
            return
        self.samples.append(
            TruthSample(
                sim_time_s=float(record.sim_time_s),
                position_ned=tuple(float(value) for value in pose.position_xyz),
            )
        )

    # -- the physical outcome -------------------------------------------------

    def inspected_within(
        self, target_ned: tuple[float, float, float], *, radius_m: float, hold_s: float
    ) -> tuple[bool, str]:
        """Whether the aircraft held inside the radius of the target, and why."""
        if not self.samples:
            return False, "no truth samples were received, so proximity cannot be decided"
        hold_start: float | None = None
        for sample in self.samples:
            dx = sample.position_ned[0] - target_ned[0]
            dy = sample.position_ned[1] - target_ned[1]
            if (dx * dx + dy * dy) ** 0.5 <= radius_m:
                if hold_start is None:
                    hold_start = sample.sim_time_s
                elif sample.sim_time_s - hold_start >= hold_s:
                    return True, (
                        f"held within {radius_m:.2f} m of the target from "
                        f"{hold_start:.2f} s to {sample.sim_time_s:.2f} s"
                    )
            else:
                hold_start = None
        return False, (
            f"no {hold_s:.1f} s interval inside {radius_m:.2f} m of the target was observed"
        )

    def returned_near(self, *, radius_m: float) -> tuple[bool, str]:
        if len(self.samples) < 2:
            return False, "fewer than two truth samples, so a return cannot be decided"
        start = self.samples[0].position_ned
        end = self.samples[-1].position_ned
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= radius_m:
            return True, f"ended {distance:.2f} m from the start position"
        return False, f"ended {distance:.2f} m from the start position (bound {radius_m:.2f} m)"


def truth_world_state(seed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The seed's declared world state: (targets, world_counts).

    The seed may declare it under a ``world_state`` key (the suite's own
    layout) or at the top level; both name the same facts, and the facts are
    what the referee records, never the wrapper.
    """
    state = seed.get("world_state") if isinstance(seed.get("world_state"), dict) else seed
    targets = state.get("targets")
    counts = state.get("world_counts")
    return targets or {}, counts or {}


def load_truth_seed(path: Path) -> dict[str, Any]:
    """The bench-side hidden facts for the suite's scenario."""
    if not path.is_file():
        raise SuiteConfigError(f"the suite's truth seed {path} is missing")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SuiteConfigError(f"{path} must hold a truth seed object")
    targets, counts = truth_world_state(document)
    if not targets:
        raise SuiteConfigError(f"{path} must declare a targets object")
    for name, entry in targets.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("present"), bool):
            raise SuiteConfigError(f"{path} target {name!r} must declare a boolean present")
    if not counts:
        raise SuiteConfigError(f"{path} must declare world_counts")
    return document


def target_position_ned(seed: dict[str, Any], name: str) -> tuple[float, float, float]:
    """The target's declared true position, in the NED frame the poses use.

    The world-state event carries ``present`` and nothing else (the bench-side
    envelope accepts exactly that key), so the geometry lives in the seed's
    ``identity`` section: either the NED form directly, or the world's own ENU
    translation, converted with the adapter's one convention rather than a
    second invention. The simulator's pose stream and this value share the
    world-origin NED frame, which is the frame the vehicle's published state is
    also converted into.
    """
    from embodied.platform.webots_ardupilot import enu_to_ned

    identity = seed.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get(name), dict):
        raise SuiteConfigError(
            f"the seed's identity section declares no geometry for target {name!r}; the "
            "inspected predicate cannot be measured, and a guessed false would be as wrong "
            "as a guessed true"
        )
    entry = identity[name]
    values = entry.get("position_ned_from_world_origin_m")
    if values is not None:
        if len(values) != 3:
            raise SuiteConfigError(f"identity[{name}].position_ned_from_world_origin_m must hold three numbers")
        return tuple(float(value) for value in values)
    values = entry.get("position_enu_m")
    if values is not None:
        if len(values) != 3:
            raise SuiteConfigError(f"identity[{name}].position_enu_m must hold three numbers")
        return enu_to_ned(tuple(float(value) for value in values))
    raise SuiteConfigError(
        f"identity[{name}] declares no position (position_ned_from_world_origin_m or "
        "position_enu_m)"
    )


# ---------------------------------------------------------------------------
# The recording itself
# ---------------------------------------------------------------------------


def world_state_payload(seed: dict[str, Any]) -> dict[str, Any]:
    """The hidden world state the referee records, in the envelope's own shape.

    ``targets[name]`` holds exactly ``present`` (the bench-side envelope accepts
    no other key), and the counts ride beside it.
    """
    targets, counts = truth_world_state(seed)
    return {
        "targets": {
            name: {"present": bool(entry["present"])} for name, entry in targets.items()
        },
        "world_counts": {str(key): int(value) for key, value in counts.items()},
    }


def measure_physical_outcome(
    collector: "TruthCollector",
    *,
    target_ned: tuple[float, float, float],
    target_id: str,
    end_state: dict[str, Any],
    crash_statustexts: list[str],
    guidance_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """The bench side's own per-run physical outcome.

    Measured from the simulator's pose stream and the aircraft's own reported
    state — never from the mission's report. ``end_state`` is recorded beside
    the flags so a reader can see what the aircraft was doing at the end.
    """
    inspected, inspected_detail = collector.inspected_within(
        target_ned, radius_m=INSPECT_RADIUS_M, hold_s=INSPECT_HOLD_S
    )
    returned, returned_detail = collector.returned_near(radius_m=RETURN_RADIUS_M)
    violations: list[str] = []
    if crash_statustexts:
        violations.append("crash_disarm: " + "; ".join(crash_statustexts[:3]))
    guidance_losses = [
        event for event in guidance_events if not event.get("guidance_held")
    ]
    if guidance_losses:
        violations.append(f"guidance_lost:{len(guidance_losses)}")
    return {
        "payload": {
            "inspected": {target_id: inspected},
            "return_verified": returned,
            "violations": violations,
            "takeover": False,
        },
        "inspected_detail": inspected_detail,
        "return_detail": returned_detail,
        "violations": violations,
        "end_state": end_state,
    }


def live_mission_driver(**kwargs: Any):
    """The one driver a real recording uses: the live mission runtime."""
    from embodied.platform import mission_runtime

    runtime = mission_runtime.MissionRuntime(**kwargs)
    result = runtime.run()
    report = mission_runtime.write_final_report(runtime)
    return result, report


def _write_mission_summary(output: Path, payload: dict[str, Any]) -> None:
    (output / "mission.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def record(
    *,
    suite_name: str,
    suite_document: dict[str, Any],
    arm: str,
    sensor_mode: SensorMode,
    output: Path,
    root: Path | None = None,
    platform_config: Path | None = None,
    truth_seed: Path | None = None,
    suite_config: Path | None = None,
    mission_driver: Any = None,
) -> CommandOutcome:
    """Record one live episode for a registered suite; never substitute one."""
    repository = root or repository_root()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    limitations = (
        "a recorded episode proves the transport, the mission and the grading "
        "chain ran; it does not prove flight beyond the declared simulator",
    )
    manifest_common = {
        "suite": suite_name,
        "arm": arm,
        "sensor_mode": sensor_mode.value,
        "host_state": host_state(),
    }
    if arm not in ADMITTED_ARMS:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(
                f"arm {arm!r} is not admitted by this transport: "
                f"admitted arms are {', '.join(ADMITTED_ARMS)}; the cloud arms are P06's "
                "comparison and need a recorded budget cap",
            ),
            limitations=limitations,
            manifest=manifest_common,
            sensor_mode=sensor_mode,
        )
    blockers = host_blockers(manifest_common["host_state"]) + port_blockers()
    manifest_common["ports_blocked"] = [
        reason for reason in blockers if reason.startswith("port ")
    ]
    if blockers:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=tuple(blockers),
            limitations=(
                "a freeze-blocked run is not evidence, so this run did not start",
                *limitations,
            ),
            manifest=manifest_common,
            sensor_mode=sensor_mode,
        )
    platform_path = Path(platform_config) if platform_config else repository / PLATFORM_CONFIG_RELATIVE
    seed_path = Path(truth_seed) if truth_seed else repository / TRUTH_SEED_RELATIVE
    try:
        document = _load_localization_config(platform_path)
        seed = load_truth_seed(seed_path)
    except Exception as error:  # config errors are the run's blocker, not a crash
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(f"{type(error).__name__}: {error}",),
            limitations=limitations,
            manifest={**manifest_common, "platform_config": str(platform_path)},
            sensor_mode=sensor_mode,
        )
    world_value = suite_document.get("world")
    if not world_value:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(f"the suite {suite_name!r} declares no world",),
            limitations=limitations,
            manifest=manifest_common,
            sensor_mode=sensor_mode,
        )
    # The suite's scene is what this run flies; the platform/estimator pins
    # stay the configuration's own (configs/first_indoor.yaml), which is not
    # edited here — only the world key of the loaded document is pointed at
    # the suite's scene.
    document.setdefault("localization", {})["world"] = str(world_value)
    settings = _platform_settings(document, repository)
    truth_targets, _world_counts = truth_world_state(seed)
    present_targets = sorted(
        name for name, entry in truth_targets.items() if entry.get("present") is True
    )
    if len(present_targets) != 1:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(
                f"the truth seed declares {len(present_targets)} present targets "
                f"({', '.join(present_targets) or 'none'}); the first-indoor mission searches "
                "for exactly one",
            ),
            limitations=limitations,
            manifest=manifest_common,
            sensor_mode=sensor_mode,
        )
    target_id = present_targets[0]
    try:
        target_ned = target_position_ned(seed, target_id)
    except SuiteConfigError as error:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(str(error),),
            limitations=limitations,
            manifest=manifest_common,
            sensor_mode=sensor_mode,
        )
    instruction = str(
        suite_document.get("mission_instruction")
        or f"Find the {target_id.replace('_', ' ')}, inspect it, and return to the start."
    )
    # ``mission_driver`` exists so the transport itself can be exercised
    # without a simulator: the CLI every recording goes through passes
    # nothing, and the default is the live runtime above.
    driver = mission_driver or live_mission_driver
    episode_dir = output / "episode"
    recorder = Recorder(episode_dir)
    collector = TruthCollector()
    episode_id = f"{suite_name}-{arm}-{output.name}"
    manifest = recorder_module.RunManifest(
        episode_id=episode_id,
        episode_kind="physical-run",
        suite=suite_name,
        trial_group_id=suite_document.get("trial_group_id"),
        arm=arm,
        sensor_mode=sensor_mode,
        code_revision=code_revision(repository),
        config_hash=recorder_module.sha256(suite_config or suite_config_path(root=repository)),
        model_identity=None,  # B0 makes no model call; the field's absence is the fact
    )
    try:
        result, report = driver(
            settings=settings,
            config_document=document,
            episode_dir=episode_dir,
            evidence_dir=output / "platform",
            recorder=recorder,
            instruction=instruction,
            target_id=target_id,
            episode_id=episode_id,
            sensor_tap=collector,
        )
    except Exception as error:  # a crashed runtime is still a recorded attempt
        _write_mission_summary(
            output,
            {
                "status": "crashed",
                "error": f"{type(error).__name__}: {error}",
                "log": [],
            },
        )
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(f"the mission runtime raised {type(error).__name__}: {error}",),
            limitations=limitations,
            manifest={**manifest_common, "episode": str(episode_dir)},
            artifacts=("mission.json",),
            sensor_mode=sensor_mode,
        )
    # The bench side's own physical measurements, from the truth stream and
    # the aircraft's reported state — never from the report.
    outcome = measure_physical_outcome(
        collector,
        target_ned=target_ned,
        target_id=target_id,
        end_state=getattr(result, "end_state", {}),
        crash_statustexts=list(getattr(result, "crash_statustexts", []) or []),
        guidance_events=list(getattr(result, "guidance_events", []) or []),
    )
    inspected = outcome["payload"]["inspected"][target_id]
    returned = outcome["payload"]["return_verified"]
    violations = outcome["violations"]
    referee = referee_module.Referee(episode_dir)
    referee.record(
        "world_state",
        world_state_payload(seed),
        ClockStamp(host_id=settings.host_id, clock_id=settings.clock_id, monotonic_ns=1),
    )
    referee.record(
        "physical_outcome",
        outcome["payload"],
        ClockStamp(host_id=settings.host_id, clock_id=settings.clock_id, monotonic_ns=2),
    )
    referee.close()
    recorder.close(manifest)
    summary = {
        "status": "recorded",
        "flew": result.flew,
        "termination_reason": result.termination_reason,
        "blockers": result.blockers,
        "phases": [
            {"status": phase.status, "reason": phase.reason} for phase in result.phases
        ],
        "claims": [
            {
                "predicate": claim.predicate,
                "target": claim.target,
                "observed": claim.observed,
                "support_refs": list(claim.support_refs),
            }
            for claim in report.claims
        ],
        "outcome": {
            "inspected": inspected,
            "inspected_detail": outcome["inspected_detail"],
            "return_verified": returned,
            "return_detail": outcome["return_detail"],
            "violations": violations,
            "takeover": False,
            "end_state": outcome["end_state"],
        },
        "publications": result.publications,
        "publish_refusals": result.publish_refusals,
        "truth_samples": len(collector.samples),
        "stream": getattr(result, "stream", {}),
        "end_state": result.end_state,
        "log": result.log,
    }
    _write_mission_summary(output, summary)
    artefacts = ["mission.json"]
    for candidate in sorted((output / "platform").glob("*.json")):
        artefacts.append(str(candidate.relative_to(output)))
    reasons = (
        f"episode recorded: {episode_dir}",
        f"mission termination: {result.termination_reason}"
        + (f" (flew: {result.flew})" if not result.flew else ""),
        f"bench-side outcome: inspected={inspected} return_verified={returned} "
        f"violations={violations or 'none'}",
    )
    if result.blockers:
        reasons = (*reasons, f"blockers: {'; '.join(result.blockers[:3])}")
    return CommandOutcome(
        status=CommandStatus.COMPLETE,
        gate_status=GateStatus.PASS if result.flew else GateStatus.FAIL,
        reasons=reasons,
        limitations=(
            *limitations,
            "support is adjudicated separately (bench score); an unadjudicated claim is "
            "pending, never a pass",
        ),
        manifest={
            **manifest_common,
            "episode": str(episode_dir),
            "episode_id": episode_id,
            "mission_instruction": instruction,
            "target_id": target_id,
            "truth_seed": str(seed_path),
            "platform_config": str(platform_path),
            "world": str(world_value),
            "score_command": f"python -m embodied bench score --episode {episode_dir}",
        },
        artifacts=tuple(artefacts),
        episode_id=episode_id,
        sensor_mode=sensor_mode,
    )
