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
* ``return_verified`` — the last truth sample within ``RETURN_RADIUS_M``
  horizontally of the first truth sample (the spawn), **and** an end state a
  landing can leave behind (disarmed, upright within
  ``END_STATE_MAX_LANDING_TILT_DEG``, with the attitude actually measured), and
  a flown approach: the horizontal distance from the spawn not
  increasing over the final ``RETURN_APPROACH_WINDOW_S`` of the truth stream.
  All three are required (ROLL-DEPARTURE.md: the radius alone admitted J58's
  crashed resting place — a monotone-away flight whose terminus happened to sit
  0.79 m from the spawn, then a dead fall to a rest 0.70 m from it — as the
  run's first ``return_verified: TRUE``), so a crashed aircraft's resting pose
  is a geometry fact, never a return.
* ``violations`` — the aircraft's own reports of a crash-disarm, any observed
  loss of Guided flight while armed, and an end state that no landing produces.
  The last of those exists because this record once affirmed that nothing went
  wrong on a run that ended inverted on the ground for about a minute: a
  landing and a crash recorded identically, and a mission then read a crashed
  aircraft's resting pose as a healthy one's measurement error and built two
  claims on it. It is measured from the vehicle's own reported attitude against
  ``END_STATE_MAX_LANDING_TILT_DEG``, and it reports the end state rather than
  diagnosing it: an estimate that says the vehicle came to rest inverted is
  itself a run that did not land as the record previously implied.
* ``takeover`` — an operator intervention, which this arm never records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
import subprocess
from typing import Any, Sequence

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
SUITE_DIR_RELATIVE = Path("configs/suites")
PLATFORM_CONFIG_RELATIVE = Path("configs/first_indoor.yaml")
TRUTH_SEED_RELATIVE = Path("scenarios/first_indoor/truth.yaml")
# The arms this transport can run. B0 is the conventional arm and makes no
# cloud call at all. The cloud arms are the ones MissionPilot itself admits
# (``pilot/mission_executive.CLOUD_ARMS`` — the one place the arm rule lives,
# so it is read from there rather than repeated here); they spend money on
# every run, so they are admitted only against a budget cap recorded for them.
# B3 is deferred with written rationale (R6) and is not an arm this transport
# can run.
CONVENTIONAL_ARM = "B0"

# Declared physical predicates (R2). Justifications are in the module docstring.
INSPECT_RADIUS_M = 1.5
INSPECT_HOLD_S = 1.0
RETURN_RADIUS_M = 1.0

# The flown-approach gate on ``return_verified`` (ROLL-DEPARTURE.md, grading
# lane): the horizontal distance from the spawn must not be increasing over the
# final window of the truth stream, so a trajectory still moving away cannot end
# "returned" however its resting place sits.
#
# * The window is two seconds of truth samples. Measured on J58's pose stream
#   (`platform/run-a/sensor-capture/records.jsonl`): pose rows arrive every
#   0.020 s, so the window holds ~100 rows — far more than the two samples the
#   radius needs, and short enough to sit inside a terminal approach.
# * The tolerance is 0.10 m. Measured on J58's real in-flight hold
#   (sim 61.5-63.5 s): the distance from spawn ranged 0.652-0.712 m across the
#   window, a 0.06 m envelope of honest hover noise; 0.10 m clears that
#   measured envelope with margin while staying a tenth of RETURN_RADIUS_M, so
#   sustained away-motion (the J58 shape) is refused, not excused.
RETURN_APPROACH_WINDOW_S = 2.0
RETURN_APPROACH_TOLERANCE_M = 0.10

# The physics-wedge detector (ROLL-DEPARTURE.md, transport lane): J58 flew into
# a Webots wedge — the world clock, cameras and feeds kept running, but the
# body's truth pose froze for 2.14 s (measured drift <= 5.5e-06 m per 0.020 s
# row on the same capture) while armed, airborne and commanding thrust — and
# the crash checker, reading the wedged state, disarmed at 2.15 m into a dead
# fall. A frozen aircraft that the record then reads as a flight is exactly the
# dishonesty this lane exists to refuse, so the signature is a named
# ``sim_fault`` and grading is refused. Pause-on-freeze is R19: out of scope
# here by ruling.
#
# Declared bounds (R2), each derived from the measured wedge and the measured
# honest-flight populations on J58's own capture:
#
# * ``SIM_WEDGE_IMMOBILITY_M = 1e-4`` per truth row. Measured wedge window
#   (sim 65.36-67.50 s): per-row displacement median 5.0e-07, max 5.5e-06.
#   Measured flying hold (sim 61.5-63.5 s): median 3.6e-03, max 6.8e-03.
#   1e-4 sits inside that gap with >18x margin to the wedge's worst row and
#   >14x below the honest hold's median.
# * ``SIM_WEDGE_MIN_FREEZE_S = 0.5`` on the host receipt clock. The measured
#   wedge lasted 2.14 s of sim time; the run's realtime_ratio_envelope
#   [0.5, 1.5] bounds the same interval to >= 1.05 s of host time, so the
#   bound keeps >2x margin to the measured fault under the declared envelope
#   while staying far below any control-loop timescale.
# * ``SIM_WEDGE_AIRBORNE_M = 0.5``: the aircraft counts as airborne when its
#   truth z is at least this far above the ground in the pose stream's NED
#   frame. Measured ground rest reads z ~= -0.035 m (J58 spawn and crash rest);
#   the declared hover is 1.5 m. This gate is what leaves a parked or landed
#   aircraft's legitimate stillness outside the detector — a wedged run that
#   disarms inverted on the ground is graded by the end-state rule, correctly.
# * ``SIM_WEDGE_THRUST_FLOOR_US = 1100`` on SERVO_OUTPUT_RAW: motors cut is
#   exactly 1000 us (measured, J58 67.238 s onward; J53 55.4 s onward) and the
#   armed-idle spin sits just above it, so anything commanded reads above the
#   floor. The airborne gate excludes the armed-idle-on-ground case.
SIM_WEDGE_IMMOBILITY_M = 1e-4
SIM_WEDGE_MIN_FREEZE_S = 0.5
SIM_WEDGE_AIRBORNE_M = 0.5
SIM_WEDGE_THRUST_FLOOR_US = 1100

# The end-state predicate: what attitude a landing can leave behind. A multirotor
# that has landed rests on its base, so its own up-axis sits near vertical; a
# vehicle that came to rest on its side or on its back did not land.
#
# The boundary is placed in the measured gap between the two populations rather
# than at a geometric landmark, because a real run sits on the landmark. Reading
# each run's own last telemetry sample across the 32 runs on disk: every run that
# **landed** measured **0.971 deg or less**, and the six that did **not** measured
# **89.980 to 179.660 deg** — each of those six *steady*, constant across its last
# six samples, so they are resting poses rather than a tumble caught mid-flight.
#
# That gap is 89.0 deg wide, so any value inside it separates the two populations
# and the verdict does not depend on which one is chosen. Forty-five sits closest
# to the middle and so leaves the largest margin to the nearest real case: 44.0
# deg one way, 45.0 the other. Ninety was the more attractive landmark — the
# point where the up-axis reaches the horizon — and it is where `live-19`
# measured, at 89.980 deg, two hundredths of a degree inside it, which would have
# made a real run's verdict turn on rounding.
#
# Applied only to an end state that has finished flying (disarmed), because a run
# that ends still armed is mid-flight and its attitude is a manoeuvre, not a
# resting pose.
END_STATE_MAX_LANDING_TILT_DEG = 45.0

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


def suite_config_path(
    root: Path | None = None, *, relative: Path | None = None
) -> Path:
    return (root or repository_root()) / (relative or SUITE_CONFIG_RELATIVE)


_SUITE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def suite_config_path_for(name: str, root: Path | None = None) -> Path:
    """The declaration file a suite name resolves to, or a refusal.

    A suite is selected by the name its own declaration carries, so the name is
    a file component and never a path: anything that could step outside the
    suites directory is refused here rather than resolved. A missing file is not
    an error — ``load_suite_document`` answers None for it, which leaves the
    unregistered-suite refusal in the caller's hands.
    """
    if not _SUITE_NAME_RE.fullmatch(name):
        raise SuiteConfigError(
            f"suite name {name!r} is not a plain name; a suite is selected by its own "
            "declared name, never by a path"
        )
    return (root or repository_root()) / SUITE_DIR_RELATIVE / f"{name}.yaml"


def declared_path(
    document: dict[str, Any], key: str, *, root: Path | None = None
) -> Path:
    """A repository-relative path a suite declaration names, or a refusal.

    The declaration carries the pointers the transport reads — its scene and its
    truth seed. Resolving them from the document is what keeps a suite graded
    against its own truth rather than against the default suite's.
    """
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SuiteConfigError(f"the suite declaration names no {key!r}")
    return (root or repository_root()) / value


def load_suite_document(path: Path | None = None) -> dict[str, Any] | None:
    """The suite's declaration, or None when the suite has not landed yet."""
    config_path = Path(path) if path is not None else suite_config_path()
    if not config_path.is_file():
        return None
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SuiteConfigError(
            f"{config_path} does not hold a suite declaration object"
        )
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
    state: dict[str, Any] = {
        "load_1m": None,
        "swap_total_mb": None,
        "swap_free_mb": None,
    }
    try:
        text = subprocess.run(
            ["uptime"], capture_output=True, text=True, check=True
        ).stdout
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
        blockers.append(
            "the 1-minute load could not be measured (uptime gave no figure)"
        )
    elif load >= HOST_LOAD_1M_MAX:
        blockers.append(
            f"1-minute load {load:.2f} is not in single digits (BASELINE-GUARD row 17)"
        )
    free = state.get("swap_free_mb")
    if free is None:
        blockers.append(
            "the swap free figure could not be measured (sysctl gave no figure)"
        )
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
    # The host receipt time of the record this sample came from (0 when the
    # record carried no stamp). This is the join against the MAVLink stream's
    # own receipt stamps, so the wedge detector never has to assume the
    # simulator clock and the autopilot clock agree (ROLL-DEPARTURE.md
    # transport lane).
    receipt_monotonic_ns: int = 0


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
        received = getattr(record, "received_stamp", None)
        self.samples.append(
            TruthSample(
                sim_time_s=float(record.sim_time_s),
                position_ned=tuple(float(value) for value in pose.position_xyz),
                receipt_monotonic_ns=int(getattr(received, "monotonic_ns", 0) or 0),
            )
        )

    # -- the physical outcome -------------------------------------------------

    def inspected_within(
        self, target_ned: tuple[float, float, float], *, radius_m: float, hold_s: float
    ) -> tuple[bool, str]:
        """Whether the aircraft held inside the radius of the target, and why."""
        if not self.samples:
            return (
                False,
                "no truth samples were received, so proximity cannot be decided",
            )
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

    def returned_near(
        self, *, radius_m: float, end_state: dict[str, Any]
    ) -> tuple[bool, str]:
        """Whether the aircraft returned to its start, and why not when it did not.

        Three conjuncts, all measured (ROLL-DEPARTURE.md, grading lane):

        * the last truth sample lies within ``radius_m`` of the first,
          horizontally — the geometry the old criterion was;
        * the end state is one a landing can leave: disarmed, with a measured
          attitude upright within ``END_STATE_MAX_LANDING_TILT_DEG``. An armed
          end is mid-flight; an unmeasured attitude cannot tell a landing from
          a crash; an inverted rest is the crashed end J58's record affirmed;
        * the horizontal distance from the start was not increasing over the
          final ``RETURN_APPROACH_WINDOW_S`` of truth samples (tolerance
          ``RETURN_APPROACH_TOLERANCE_M``), so the run flew an approach to the
          start instead of ending near it by coincidence or by falling.

        The old criterion was the first conjunct alone, and it verified the
        return of an aircraft that fell out of the sky inverted 0.70 m from its
        spawn because its third explore terminus happened to sit 0.79 m away.
        """
        if len(self.samples) < 2:
            return False, "fewer than two truth samples, so a return cannot be decided"
        start = self.samples[0].position_ned
        end = self.samples[-1].position_ned
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > radius_m:
            return (
                False,
                f"ended {distance:.2f} m from the start position (bound {radius_m:.2f} m)",
            )
        if not end_state:
            return False, (
                f"ended {distance:.2f} m from the start position, but the end state was "
                "never measured, so a landing cannot be told from a crash"
            )
        if end_state.get("armed") is not False:
            return False, (
                f"ended {distance:.2f} m from the start position, but the aircraft was "
                "still armed at the end, so no landing was observed"
            )
        landing_violation = end_state_violation(end_state)
        if landing_violation is not None:
            return False, (
                f"ended {distance:.2f} m from the start position, but the end state is "
                f"no landing: {landing_violation}"
            )
        if end_state_tilt_deg(end_state.get("attitude_rpy")) is None:
            # ``end_state_violation`` passes an absent attitude as undecided, not
            # upright; a verified return needs the landing decided, not undecided.
            return False, (
                f"ended {distance:.2f} m from the start position, but the end attitude "
                "was not measured, so a landing cannot be told from a crash"
            )
        horizon = self.samples[-1].sim_time_s - RETURN_APPROACH_WINDOW_S
        window = [sample for sample in self.samples if sample.sim_time_s >= horizon]
        if len(window) < 2:
            window = self.samples[-2:]
        distances = [
            (
                (sample.position_ned[0] - start[0]) ** 2
                + (sample.position_ned[1] - start[1]) ** 2
            )
            ** 0.5
            for sample in window
        ]
        final = distances[-1]
        # Away-motion signature: an earlier sample of the window sits well
        # inside the final distance, so the aircraft was still travelling away
        # from the start when the window closed. An approach (distances
        # shrinking onto the final value) and honest hover noise around the
        # rest position never produce that shape (ROLL-DEPARTURE.md grading
        # lane).
        growing = [
            value
            for value in distances[:-1]
            if value < final - RETURN_APPROACH_TOLERANCE_M
        ]
        if growing:
            return False, (
                f"ended {distance:.2f} m from the start position (bound {radius_m:.2f} m), "
                f"but the horizontal distance was still up to {final - min(growing):.2f} m "
                f"smaller within the final {RETURN_APPROACH_WINDOW_S:.1f} s of truth "
                f"samples (tolerance {RETURN_APPROACH_TOLERANCE_M:.2f} m) — the aircraft "
                "was moving away from the start, not approaching it"
            )
        return True, (
            f"ended {distance:.2f} m from the start position after a flown approach to "
            "an upright, disarmed rest"
        )


def armed_thrust_rows(mavlink_log: Path | str) -> list[tuple[int, bool]]:
    """When the run was commanding flight, on the host receipt clock.

    One row per HEARTBEAT or SERVO_OUTPUT_RAW message that changed the picture,
    as ``(receipt_monotonic_ns, flying)`` where *flying* is the aircraft's own
    reported state: armed (HEARTBEAT ``base_mode``, the convention
    ``decode_telemetry`` uses) **and** commanding thrust (any motor output above
    ``SIM_WEDGE_THRUST_FLOOR_US``). Last-known-value semantics: a row speaks
    until a later row contradicts it. Rows without a receipt stamp are skipped
    rather than guessed, and a missing log answers an empty list — the detector
    stays silent when there is no evidence a wedge would contradict.
    """
    from embodied.platform.webots_ardupilot import RECEIVED_AT_KEY

    rows: list[tuple[int, bool]] = []
    armed: bool | None = None
    thrusting: bool | None = None
    path = Path(mavlink_log)
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(document, dict):
                continue
            kind = document.get("mavpackettype")
            receipt_ns = document.get(RECEIVED_AT_KEY)
            if receipt_ns is None:
                continue
            changed = False
            if kind == "HEARTBEAT":
                base_mode = document.get("base_mode")
                if base_mode is not None:
                    value = bool(base_mode & 128)
                    changed = value != armed
                    armed = value
            elif kind == "SERVO_OUTPUT_RAW":
                outputs = [
                    float(document[key])
                    for key in document
                    if key.startswith("servo")
                    and key.endswith("_raw")
                    and isinstance(document[key], (int, float))
                ]
                if outputs:
                    value = max(outputs) > SIM_WEDGE_THRUST_FLOOR_US
                    changed = value != thrusting
                    thrusting = value
            if changed and armed is not None and thrusting is not None:
                rows.append((int(receipt_ns), bool(armed and thrusting)))
    return rows


def sim_wedge_fault(
    samples: "Sequence[TruthSample]",
    flying_rows: Sequence[tuple[int, bool]],
    *,
    immobility_m: float = SIM_WEDGE_IMMOBILITY_M,
    min_freeze_s: float = SIM_WEDGE_MIN_FREEZE_S,
    airborne_m: float = SIM_WEDGE_AIRBORNE_M,
) -> str | None:
    """The named physics-wedge fault, or ``None`` when the flight moved.

    ROLL-DEPARTURE.md transport lane: a Webots wedge froze the body's truth
    pose for 2.14 s — drift at most 5.5e-06 m per row — while the world clock
    ran on, the aircraft stayed armed and its motors stayed on the rails; the
    crash checker then read the wedged state and disarmed into a dead fall. The
    signature is therefore: truth pose immobile beyond ``min_freeze_s`` (on the
    host receipt clock the samples and the MAVLink stream share), while armed,
    airborne and commanding thrust. Any one condition absent means the
    stillness has an honest explanation — parked, landed, disarmed, or
    motors cut — and there is no fault to name.

    Returns the fault text for the longest qualifying window, so the record
    names what froze, where, and for how long.
    """
    if len(samples) < 2 or not flying_rows:
        return None
    rows = sorted(flying_rows)
    row_index = 0
    flying = False
    best: tuple[float, TruthSample, TruthSample] | None = None
    run_start: int | None = None
    for index, sample in enumerate(samples):
        while (
            row_index < len(rows) and rows[row_index][0] <= sample.receipt_monotonic_ns
        ):
            flying = rows[row_index][1]
            row_index += 1
        still = (
            flying
            and sample.position_ned[2] <= -airborne_m
            and index > 0
            and math.dist(samples[index - 1].position_ned, sample.position_ned)
            <= immobility_m
        )
        if still:
            if run_start is None:
                run_start = index - 1
        elif run_start is not None:
            first, last = samples[run_start], samples[index - 1]
            span_s = (last.receipt_monotonic_ns - first.receipt_monotonic_ns) / 1e9
            if span_s >= min_freeze_s and (best is None or span_s > best[0]):
                best = (span_s, first, last)
            run_start = None
    if run_start is not None:
        span_s = (
            samples[-1].receipt_monotonic_ns - samples[run_start].receipt_monotonic_ns
        ) / 1e9
        if span_s >= min_freeze_s and (best is None or span_s > best[0]):
            best = (span_s, samples[run_start], samples[-1])
    if best is None:
        return None
    span_s, first, last = best
    position = ", ".join(f"{value:.3f}" for value in last.position_ned)
    return (
        f"sim_fault: the simulator's truth pose held still for {span_s:.2f} s while the "
        f"aircraft was armed, airborne and commanding thrust (immobility bound "
        f"{immobility_m:g} m per row, fault bound {min_freeze_s:.2f} s; window ends at "
        f"truth pose ({position})). A physics wedge manufactured the state the crash "
        "checker read, so this episode cannot be graded as a flight (ROLL-DEPARTURE.md)"
    )


def truth_world_state(seed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The seed's declared world state: (targets, world_counts).

    The seed may declare it under a ``world_state`` key (the suite's own
    layout) or at the top level; both name the same facts, and the facts are
    what the referee records, never the wrapper.
    """
    state = (
        seed.get("world_state") if isinstance(seed.get("world_state"), dict) else seed
    )
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
            raise SuiteConfigError(
                f"{path} target {name!r} must declare a boolean present"
            )
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
            raise SuiteConfigError(
                f"identity[{name}].position_ned_from_world_origin_m must hold three numbers"
            )
        return tuple(float(value) for value in values)
    values = entry.get("position_enu_m")
    if values is not None:
        if len(values) != 3:
            raise SuiteConfigError(
                f"identity[{name}].position_enu_m must hold three numbers"
            )
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


# The mode a mission flies under, and the mode its declared termination lands in.
GUIDED_MODE = "GUIDED"
LANDING_MODE = "LAND"


# The phrase ArduPilot uses when a failsafe takes the flight mode away from whoever
# was controlling the vehicle ("EKF Failsafe: changed to Land Mode"). It is the
# aircraft's own statement that the change was not asked for, which is the one signal
# that separates a failsafe landing from the landing a mission is required to make.
AUTOPILOT_MODE_CHANGE_MARKER = "changed to"


def autopilot_mode_change_statustexts(mavlink_log: Path | str) -> list[str]:
    """The autopilot's own reports that it changed the flight mode by itself.
    Read from the run's recorded STATUSTEXT stream exactly as the crash-disarm scan
    reads it, and kept as a separate signal because a failsafe can take the aircraft
    to LAND without crashing it — and a landing is what the guidance accounting is
    most tempted to excuse. ``live-14`` is the worked case: the mode changed to LAND
    at message #11255 and the autopilot said why five messages later.
    """
    texts: list[str] = []
    path = Path(mavlink_log)
    if not path.is_file():
        return texts
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(document, dict):
                continue
            if document.get("mavpackettype") != "STATUSTEXT":
                continue
            text = str(document.get("text", "")).strip()
            if AUTOPILOT_MODE_CHANGE_MARKER in text.lower() and text not in texts:
                texts.append(text)
    return texts


def _guidance_departures(
    events: list[dict[str, Any]],
    end_state: dict[str, Any],
    autopilot_mode_changes: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the platform's control events into control losses and ordinary changes.

    The platform records one event for every change of mode or arming state and
    reports whether the aircraft was under armed Guided control *after* the change
    (``webots_ardupilot.drain``). That is the right raw material and the wrong
    count. A mission necessarily changes mode while it is not yet flying — the
    declared bring-up, which is disarmed — changes mode once to take control, and
    changes mode again to land. None of those is a loss of control, and counting
    them reported a violation on every run that had not committed one: a
    ``guidance_lost:7`` stood in the receipt of ``live-motion-8`` while the
    aircraft in fact never left armed Guided control except to land.

    A **departure** is a change that leaves armed Guided control: the aircraft was
    flying under it immediately before and is not after. Transitions before any
    telemetry arrived (``from_mode`` is None — a report artifact, since the
    aircraft was flying nothing yet) and transitions while not armed fall outside
    that definition by themselves, so neither needs a special case.

    Of the departures, all but the terminal one are losses, because the aircraft
    came back: an autopilot that takes control away mid-flight and later returns it
    still took it away. The terminal departure is the end of the flight and is
    excused only when all three of these hold: the run ended in the mission's
    declared landing state (in ``LAND``, disarmed); the departure is into that
    landing or is the shutdown while still in Guided; and **the autopilot did not
    report taking the mode itself**. That last condition is what stops the rule
    hiding the case it exists for — ``live-14`` flew under a failsafe that changed
    the mode to LAND and then crashed, and a landing alone cannot be told from the
    landing a mission is required to make. A terminal departure to ``LOITER`` or
    ``RTL`` stays a loss without needing the autopilot's explanation.
    """
    ordered = sorted(events, key=lambda event: int(event.get("at_monotonic_ns") or 0))
    departures = [
        event
        for event in ordered
        if event.get("armed_before") is True
        and event.get("from_mode") == GUIDED_MODE
        and not (event.get("armed_after") and event.get("to_mode") == GUIDED_MODE)
    ]
    declared_landing = (
        str(end_state.get("mode") or "") == LANDING_MODE
        and end_state.get("armed") is False
        and not autopilot_mode_changes
    )
    losses: list[dict[str, Any]] = []
    for index, event in enumerate(departures):
        terminal = index == len(departures) - 1
        ends_the_mission = str(event.get("to_mode") or "") in (
            LANDING_MODE,
            GUIDED_MODE,
        )
        if terminal and declared_landing and ends_the_mission:
            continue
        losses.append(event)
    return losses, departures


def _guidance_record(
    losses: list[dict[str, Any]], departures: list[dict[str, Any]], seen: int
) -> dict[str, Any]:
    """What the run's control events were, in a form a reader can audit.

    The count alone cannot say whether it was earned, and the platform's events are
    not persisted anywhere else, so the departures and the losses travel with the
    outcome — a count with no evidence behind it is what went unnoticed here.
    """

    def short(event: dict[str, Any]) -> dict[str, Any]:
        return {
            key: event.get(key)
            for key in (
                "at_monotonic_ns",
                "from_mode",
                "to_mode",
                "armed_before",
                "armed_after",
            )
        }

    return {
        "events": seen,
        "departures_from_armed_guided": len(departures),
        "losses": len(losses),
        "lost": [short(event) for event in losses],
    }


def end_state_tilt_deg(attitude_rpy: Sequence[float] | None) -> float | None:
    """How far the vehicle's own up-axis is from vertical, in degrees.

    With the aviation rotation order the body's up-axis in the world has a
    vertical component of ``cos(roll)·cos(pitch)``, so this needs no yaw and no
    matrix: the arc-cosine of that product is the tilt — 0 deg upright, 180 deg
    resting on its back. ``None`` when the attitude is absent or unusable, which
    is a different statement from "upright" and must not be read as one.
    """
    if attitude_rpy is None or len(attitude_rpy) != 3:
        return None
    try:
        roll, pitch = float(attitude_rpy[0]), float(attitude_rpy[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(roll) and math.isfinite(pitch)):
        return None
    vertical = math.cos(math.radians(roll)) * math.cos(math.radians(pitch))
    return math.degrees(math.acos(max(-1.0, min(1.0, vertical))))


def end_state_violation(end_state: dict[str, Any]) -> str | None:
    """An end state no landing produces, or ``None`` when it could be one.

    The whole rule in one place, so the live path and the re-derivation of an
    older run cannot come to different answers about the same data.
    """
    if end_state.get("armed") is not False:
        # Armed, or arming unknown. An end state that is still armed is
        # mid-flight and its attitude is a manoeuvre, not a resting pose.
        return None
    tilt = end_state_tilt_deg(end_state.get("attitude_rpy"))
    if tilt is None or tilt <= END_STATE_MAX_LANDING_TILT_DEG:
        # Absent attitude is not evidence of a good landing — the field postdates
        # the runs it would judge, which is what ``rederive_end_state_violation``
        # exists for. It is only that this record cannot decide it.
        return None
    roll, pitch = (
        float(end_state["attitude_rpy"][0]),
        float(end_state["attitude_rpy"][1]),
    )
    return (
        f"end_state_inverted: the vehicle came to rest {tilt:.1f} deg from vertical "
        f"(roll {roll:.1f}, pitch {pitch:.1f}) while disarmed, past the "
        f"{END_STATE_MAX_LANDING_TILT_DEG:.0f} deg a landing can leave it"
    )


def end_state_attitude_from_telemetry(
    run_dir: Path,
) -> tuple[float, float, float] | None:
    """The attitude a run ended in, read from that run's own recorded telemetry.

    ``attitude_rpy`` reaches an end state only for runs recorded after the field
    existed. An older run still recorded the same quantity, because the platform
    writes every ``ATTITUDE`` message it receives to its own log, and the last
    one is the attitude that run ended in. Read-only: nothing is written back.
    """
    for log_path in sorted(Path(run_dir).glob("platform/*/mavlink.jsonl")):
        last: dict[str, Any] | None = None
        try:
            with log_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if '"ATTITUDE"' not in line:
                        continue
                    try:
                        document = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if document.get("mavpackettype") == "ATTITUDE":
                        last = document
        except OSError:
            continue
        if last is None:
            continue
        try:
            return (
                math.degrees(float(last["roll"])),
                math.degrees(float(last["pitch"])),
                math.degrees(float(last["yaw"])),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _attitude_for_reading(end_state: dict[str, Any], reading: str) -> list[float]:
    """The attitude triple the named tilt reading was computed from."""
    values = [float(value) for value in end_state["attitude_rpy"]]
    if reading == "radians-stored-raw":
        return [math.degrees(value) for value in values]
    return values


def _end_state_tilt_readings(
    attitude_rpy: Sequence[float] | None,
) -> list[tuple[str, float]]:
    """Every tilt the recorded attitude could honestly mean, with its reading.

    The bench rule reads degrees, and the live record now writes degrees — but
    runs recorded before that conversion stored the telemetry sample's radians
    raw (ROLL-DEPARTURE.md: J58's [3.141, −0.0008, 2.613] rad read as a 3.1
    degree tilt under the degrees rule). A radians triple is bounded by pi in
    every component, so when the recorded values all sit inside that bound the
    radians reading is possible and is reported beside the degrees one. The
    radians reading of an honest landing can never cross the 45-degree bound
    (pi radians read as degrees is 3.14), so the extra reading can only ever
    add a violation a units mismatch hid — it cannot invent one.
    """
    primary = end_state_tilt_deg(attitude_rpy)
    readings: list[tuple[str, float]] = []
    if primary is not None:
        readings.append(("degrees", primary))
    if attitude_rpy and len(attitude_rpy) == 3:
        try:
            values = [float(value) for value in attitude_rpy]
        except (TypeError, ValueError):
            return readings
        if all(
            math.isfinite(value) and abs(value) <= math.pi + 0.01 for value in values
        ):
            converted = end_state_tilt_deg([math.degrees(value) for value in values])
            if converted is not None:
                readings.append(("radians-stored-raw", converted))
    return readings


def rederive_end_state_violation(run_dir: Path) -> dict[str, Any]:
    """What the end-state rule says about a run already on disk.

    A run recorded before ``attitude_rpy`` existed cannot be re-scored by
    re-running the transport, so this reads what that run itself recorded: its
    end state from ``mission.json`` and, when that carries no attitude, the last
    ``ATTITUDE`` message from its own platform log. When the recorded attitude
    is a raw radians triple (the pre-conversion live records), the radians
    reading is judged beside the degrees one (ROLL-DEPARTURE.md grading lane).
    It returns the finding; the caller decides where it goes, and nothing here
    writes to the run.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "mission.json"
    document: dict[str, Any] = {}
    if summary_path.exists():
        document = json.loads(summary_path.read_text(encoding="utf-8"))
    outcome = document.get("outcome") or {}
    recorded_end_state = dict(outcome.get("end_state") or {})
    recorded = list(outcome.get("violations") or [])
    end_state = dict(recorded_end_state)
    source = "end_state"
    if end_state_tilt_deg(end_state.get("attitude_rpy")) is None:
        telemetry = end_state_attitude_from_telemetry(run_dir)
        if telemetry is None:
            source = "absent"
        else:
            end_state["attitude_rpy"] = list(telemetry)
            source = "platform/*/mavlink.jsonl (the last ATTITUDE message)"
    readings = _end_state_tilt_readings(end_state.get("attitude_rpy"))
    tilt = next((value for name, value in readings if name == "degrees"), None)
    radians_tilt = next(
        (value for name, value in readings if name == "radians-stored-raw"), None
    )
    # The violation is judged through the one rule, on the end state each
    # reading produces, so live and rederived paths cannot drift apart. The
    # degrees reading is the default and speaks plainly; only a finding that
    # depends on the radians reading carries the reading's name.
    violation = None
    for name, value in readings:
        if value > END_STATE_MAX_LANDING_TILT_DEG:
            finding = end_state_violation(
                dict(end_state, attitude_rpy=_attitude_for_reading(end_state, name))
            )
            if finding is not None:
                suffix = (
                    ""
                    if name == "degrees"
                    else f" [the {name} reading of the recorded attitude]"
                )
                violation = finding + suffix
                break
    return {
        "run": run_dir.name,
        "recorded_violations": recorded,
        "end_state_recorded": recorded_end_state,
        "attitude_source": source,
        "attitude_rpy_deg": (
            [round(float(value), 3) for value in end_state["attitude_rpy"]]
            if end_state.get("attitude_rpy")
            else None
        ),
        "tilt_deg": None if tilt is None else round(tilt, 3),
        "tilt_deg_radians_reading": (
            None if radians_tilt is None else round(radians_tilt, 3)
        ),
        "violation": violation,
        "violations_rederived": [*recorded, violation] if violation else list(recorded),
    }


def receipt_verdict(
    *, flew: bool, crashed: bool, sim_fault: str | None
) -> tuple[CommandStatus, GateStatus]:
    """What the recording receipt reads, without lying about the flight.

    Two different dishonesties, two different answers:

    * A **crashed** flight (truth records crash_disarm — J56/J57/J58) keeps the
      command status ``complete`` — the command did record its episode — and
      fails its gate, with the crash named in the reasons and, above this
      transport, in the mission's own ``crashed`` termination. It must not read
      ``blocked``: the P06 comparison classifies a blocked receipt as an
      instrument failure and pools the aircraft's failure with the host's
      (specification 20.5 — a policy crash is an outcome and stays in the
      table).
    * A **sim_fault** (the physics wedge, ROLL-DEPARTURE.md transport lane)
      reads ``blocked``: a frozen-physics episode is an instrument failure, the
      episode is recorded as evidence, and the grader refuses it — it is not a
      flight and must never enter a comparison as one.
    """
    if sim_fault:
        return CommandStatus.BLOCKED, GateStatus.FAIL
    if crashed:
        return CommandStatus.COMPLETE, GateStatus.FAIL
    return CommandStatus.COMPLETE, GateStatus.PASS if flew else GateStatus.FAIL


def measure_physical_outcome(
    collector: "TruthCollector",
    *,
    target_ned: tuple[float, float, float] | None,
    target_id: str,
    end_state: dict[str, Any],
    crash_statustexts: list[str],
    guidance_events: list[dict[str, Any]],
    autopilot_mode_changes: Sequence[str] = (),
    sim_fault: str | None = None,
) -> dict[str, Any]:
    """The bench side's own per-run physical outcome.

    Measured from the simulator's pose stream and the aircraft's own reported
    state — never from the mission's report. ``end_state`` is recorded beside
    the flags so a reader can see what the aircraft was doing at the end, and
    it also gates ``return_verified`` (a return needs a landed end state).
    ``sim_fault`` carries the transport's physics-wedge finding (ROLL-DEPARTURE.md)
    when the truth stream froze under thrust; it is the simulator's fault and
    refuses grading, never the aircraft's violation.
    """
    if target_ned is None:
        # The world declares no such object, so there is no position to be near
        # and no proximity to measure. The false is a fact about the world, not
        # a failed look, and the detail says which it is.
        inspected, inspected_detail = (
            False,
            (
                f"the world declares no {target_id}, so there is no position an inspection "
                "could have been measured against"
            ),
        )
    else:
        inspected, inspected_detail = collector.inspected_within(
            target_ned, radius_m=INSPECT_RADIUS_M, hold_s=INSPECT_HOLD_S
        )
    returned, returned_detail = collector.returned_near(
        radius_m=RETURN_RADIUS_M, end_state=end_state
    )
    violations: list[str] = []
    if crash_statustexts:
        violations.append("crash_disarm: " + "; ".join(crash_statustexts[:3]))
    guidance_losses, guidance_departures = _guidance_departures(
        guidance_events, end_state, autopilot_mode_changes
    )
    if guidance_losses:
        violations.append(f"guidance_lost:{len(guidance_losses)}")
    end_state_finding = end_state_violation(end_state)
    if end_state_finding is not None:
        violations.append(end_state_finding)
    return {
        "payload": {
            "inspected": {target_id: inspected},
            "return_verified": returned,
            "violations": violations,
            "takeover": False,
            # The simulator's own fault, never the aircraft's: a wedge is not a
            # violation of the aircraft, but the record must name it and the
            # grader must refuse it (ROLL-DEPARTURE.md transport lane). None
            # when the transport measured no wedge.
            "sim_fault": sim_fault,
        },
        "inspected_detail": inspected_detail,
        "return_detail": returned_detail,
        "violations": violations,
        "sim_fault": sim_fault,
        "end_state": end_state,
        "guidance": {
            **_guidance_record(
                guidance_losses, guidance_departures, len(guidance_events)
            ),
            "autopilot_mode_changes": list(autopilot_mode_changes),
        },
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
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def recorded_budget_cap(repository: Path) -> tuple[tuple[int, float] | None, str]:
    """The cloud arms' recorded (``max_calls``, ``spend_ceiling_usd``), and why not.

    The cap a paid arm's spend is governed by is the runtime-model
    configuration's own ``probe`` section, and ``pilot/probe.py`` already owns
    both the load and the validation of that file, so this reuses those rather
    than reading the section a second way.

    Absent, unreadable or unusable is not a recorded cap. An unbudgeted cloud
    run is exactly what this gate exists to prevent, so anything other than a
    usable pair refuses and says which.
    """
    from embodied.pilot import probe as probe_module
    from embodied.platform import mission_runtime

    path = repository / mission_runtime.RUNTIME_MODEL_CONFIG_RELATIVE
    try:
        document = probe_module.load_runtime_config(path)
        limits = probe_module.probe_limits(document)
    except probe_module.ProbeConfigError as error:
        return None, str(error)
    if limits is None:
        return None, (
            f"{path} records no probe.max_calls and probe.spend_ceiling_usd, "
            "which is what a cloud arm's spend is capped by"
        )
    return limits, ""


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
    from embodied.pilot import mission_executive as executive_module

    if arm != CONVENTIONAL_ARM and arm not in executive_module.CLOUD_ARMS:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(
                f"arm {arm!r} is not one this transport can run: {CONVENTIONAL_ARM} is the "
                f"conventional arm and {', '.join(executive_module.CLOUD_ARMS)} are the cloud "
                "arms, and B3 is deferred with written rationale (R6)",
            ),
            limitations=limitations,
            manifest=manifest_common,
            sensor_mode=sensor_mode,
        )
    if arm in executive_module.CLOUD_ARMS:
        limits, problem = recorded_budget_cap(repository)
        if limits is None:
            return CommandOutcome(
                status=CommandStatus.BLOCKED,
                gate_status=GateStatus.NOT_APPLICABLE,
                reasons=(
                    f"arm {arm!r} spends money and is admitted only against a recorded "
                    f"budget cap, and none is usable: {problem}",
                ),
                limitations=limitations,
                manifest=manifest_common,
                sensor_mode=sensor_mode,
            )
        # Recorded on the run itself, so a receipt names the cap that governed
        # the spend rather than leaving a reader to find that day's config.
        manifest_common["provider_budget"] = {
            "max_calls": limits[0],
            "spend_ceiling_usd": limits[1],
        }
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
    platform_path = (
        Path(platform_config)
        if platform_config
        else repository / PLATFORM_CONFIG_RELATIVE
    )
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
    declared_targets = sorted(truth_targets)
    if len(declared_targets) != 1:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(
                f"the truth seed declares {len(declared_targets)} targets "
                f"({', '.join(declared_targets) or 'none'}); the mission searches for "
                "exactly one",
            ),
            limitations=limitations,
            manifest=manifest_common,
            sensor_mode=sensor_mode,
        )
    target_id = declared_targets[0]
    # Declaring which object the mission searches for and that object being
    # there are two different facts. The first is the query; the second is the
    # world's own state. A scenario whose subject is legitimately absent asks
    # for a report of absence rather than an inspection, so it is a different
    # task type and not an exemption from finding a present one. Only a present
    # target has a position, and only a present target's inspection can be
    # measured: a target declared present without geometry is still refused.
    target_present = truth_targets[target_id].get("present") is True
    if target_present:
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
    else:
        target_ned = None
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
        config_hash=recorder_module.sha256(
            suite_config or suite_config_path(root=repository)
        ),
        model_identity=None,  # B0 makes no model call; the field's absence is the fact
    )
    try:
        result, report = driver(
            settings=settings,
            arm=arm,
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
    mavlink_log = output / "platform" / "run-a" / "mavlink.jsonl"
    # The transport lane's own detector (ROLL-DEPARTURE.md): truth pose frozen
    # while armed, airborne and commanding thrust is a simulator fault, not a
    # flight. R2 derivation beside SIM_WEDGE_* above; pause-on-freeze is R19.
    sim_fault = sim_wedge_fault(collector.samples, armed_thrust_rows(mavlink_log))
    outcome = measure_physical_outcome(
        collector,
        target_ned=target_ned,
        target_id=target_id,
        end_state=getattr(result, "end_state", {}),
        crash_statustexts=list(getattr(result, "crash_statustexts", []) or []),
        guidance_events=list(getattr(result, "guidance_events", []) or []),
        # Read from the run's own record rather than trusted to a summary: a
        # failsafe that lands the aircraft looks exactly like the landing the
        # mission owes, and only the aircraft's own words separate them.
        autopilot_mode_changes=autopilot_mode_change_statustexts(mavlink_log),
        sim_fault=sim_fault,
    )
    inspected = outcome["payload"]["inspected"][target_id]
    returned = outcome["payload"]["return_verified"]
    violations = outcome["violations"]
    # The receipt honesty gate (ROLL-DEPARTURE.md: J56/J57/J58 read
    # mission_completed / a complete receipt over a crash-disarmed aircraft,
    # and a wedged run must not grade as a flight at all). The episode and its
    # evidence are still written — the refusal is about the outcome, not about
    # erasing the record.
    crashed = bool(list(getattr(result, "crash_statustexts", []) or []))
    status, gate_status = receipt_verdict(
        flew=result.flew, crashed=crashed, sim_fault=sim_fault
    )
    referee = referee_module.Referee(episode_dir)
    referee.record(
        "world_state",
        world_state_payload(seed),
        ClockStamp(
            host_id=settings.host_id, clock_id=settings.clock_id, monotonic_ns=1
        ),
    )
    referee.record(
        "physical_outcome",
        outcome["payload"],
        ClockStamp(
            host_id=settings.host_id, clock_id=settings.clock_id, monotonic_ns=2
        ),
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
            "sim_fault": sim_fault,
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
    if sim_fault:
        # Grading refused on the simulator's fault (ROLL-DEPARTURE.md transport
        # lane): a frozen-physics episode is not a flight, whatever it reads as.
        reasons = (
            *reasons,
            f"grading refused: {sim_fault}",
        )
    if crashed:
        # ROLL-DEPARTURE.md: three runs (J56/J57/J58) recorded a crash-disarm and
        # still read complete; the receipt stops reading complete here.
        reasons = (
            *reasons,
            "the aircraft crash-disarmed in flight: the episode is recorded as "
            "evidence, and the flight it documents did not complete",
        )
    if result.blockers:
        reasons = (*reasons, f"blockers: {'; '.join(result.blockers[:3])}")
    guidance = outcome.get("guidance") or {}
    if guidance.get("events"):
        # The count is no longer the whole story, so the story travels with it: a
        # reader can see how many control events the run had and how many of them
        # actually left armed Guided control, rather than being asked to trust a
        # number that a declared landing used to inflate.
        autopilot_changes = guidance.get("autopilot_mode_changes") or []
        reasons = (
            *reasons,
            f"guidance: {guidance['events']} control event(s), "
            f"{guidance['departures_from_armed_guided']} departure(s) from armed Guided, "
            f"{guidance['losses']} loss(es)"
            + (
                f"; the autopilot changed the mode itself: {'; '.join(autopilot_changes[:2])}"
                if autopilot_changes
                else ""
            ),
        )
    return CommandOutcome(
        status=status,
        gate_status=gate_status,
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
            "target_present": target_present,
            "truth_seed": str(seed_path),
            "platform_config": str(platform_path),
            "world": str(world_value),
            "score_command": f"python -m embodied bench score --episode {episode_dir}",
        },
        artifacts=tuple(artefacts),
        episode_id=episode_id,
        sensor_mode=sensor_mode,
    )
