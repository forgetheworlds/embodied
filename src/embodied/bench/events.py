"""The event envelope shared by the two streams in an episode directory.

An episode stores its history as append-only JSON-lines files, one line per
event. Every event carries a sequence number, a kind, a clock stamp and a
payload. The payload of every agent-stream kind except ``intervention`` is a
record from ``embodied.contracts.records`` encoded with that module's own JSON
rules, so an event can never drift from the contract it carries;
``intervention`` is bench-local because no runtime record describes one.

Both kind vocabularies are closed and they do not overlap, so a kind from one
stream is rejected by any reader configured for the other. This module owns
the schema — envelope, vocabularies, payload validation, clock-order rule and
the line bytes — and it names no file inside an episode directory. Which file
exists, who may open it and what a reader can reach is decided by the module
that owns each surface:

* ``embodied.bench.recorder`` owns the agent stream and the projection a
  non-scorer reader is confined to.
* ``embodied.bench.referee`` owns the bench-side stream and can only append.
* ``embodied.bench.grader`` is the only reader of the bench-side stream.

Record contract consumed unchanged: RECORDS_REVISION. This envelope's schema
revision: EVENTS_REVISION.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from embodied.contracts.records import ClockStamp, RecordError, from_dict, to_dict

EVENTS_REVISION = "p02-events-1"


class EventError(ValueError):
    """An event is malformed, of an unknown kind, or carries a wrong payload."""


class EpisodeError(Exception):
    """An episode directory is missing, incomplete or internally inconsistent."""


# The closed vocabularies. A kind belongs to exactly one stream, which is what
# makes the two files separate message surfaces rather than one log split for
# tidiness.
AGENT_EVENT_KINDS = frozenset(
    {
        "mission",
        "request",
        "selection",
        "goal",
        "goal_status",
        "setpoint",
        "execution",
        "observation",
        "intervention",
        "report",
    }
)
TRUTH_EVENT_KINDS = frozenset({"world_state", "physical_outcome"})

# Which P00 record carries the payload of each agent kind. Consumed unchanged:
# the event stores that record's encoding and hands the bytes back to
# from_dict for validation, so the contract stays the single definition.
PAYLOAD_RECORDS: dict[str, str] = {
    "mission": "MissionContract",
    "request": "DecisionRequest",
    "selection": "VisualSelection",
    "goal": "SpatialGoal",
    "goal_status": "GoalStatus",
    "setpoint": "MotionSetpoint",
    "execution": "ExecutionStatus",
    "observation": "Observation",
    "report": "FinalReport",
}

_INTERVENTION_FIELDS = ("intervention_id", "actor", "category", "reason")

_EVENT_FIELDS = ("seq", "kind", "stamp", "sim_time_s", "payload")


@dataclass(frozen=True)
class Event:
    """One ordered entry of a stream.

    Construction validates everything: the envelope fields here, and the
    payload against its kind's schema. A malformed event cannot exist in
    memory, so a writer cannot append one and a reader cannot decode one.
    """

    seq: int
    kind: str
    stamp: ClockStamp
    sim_time_s: float | None
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0:
            raise EventError(f"seq must be a non-negative integer, got {self.seq!r}")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise EventError(f"kind must be a non-empty string, got {self.kind!r}")
        if not isinstance(self.stamp, ClockStamp):
            raise EventError("stamp must be a ClockStamp")
        if self.sim_time_s is not None:
            if isinstance(self.sim_time_s, bool) or not isinstance(
                self.sim_time_s, (int, float)
            ):
                raise EventError(f"sim_time_s must be a number or None, got {self.sim_time_s!r}")
            if not math.isfinite(float(self.sim_time_s)):
                raise EventError("sim_time_s must be finite")
        validate_payload(self.kind, self.payload)


def _check_exact_keys(payload: dict[str, Any], expected: Sequence[str], where: str) -> None:
    missing = sorted(set(expected) - set(payload))
    unknown = sorted(set(payload) - set(expected))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown {', '.join(unknown)}")
        raise EventError(f"{where} payload must hold exactly {', '.join(expected)} ({'; '.join(detail)})")


def _check_flag(value: Any, where: str) -> None:
    if not isinstance(value, bool):
        raise EventError(f"{where} must be a bool")


def _check_named_flags(value: Any, where: str) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise EventError(f"{where} must be an object of name -> bool")
    result = {}
    for name, flag in value.items():
        if not isinstance(name, str) or not name.strip():
            raise EventError(f"{where} keys must be non-empty strings")
        _check_flag(flag, f"{where}[{name}]")
        result[name] = flag
    return result


def _check_intervention(payload: dict[str, Any]) -> None:
    _check_exact_keys(payload, _INTERVENTION_FIELDS, "intervention")
    for field in _INTERVENTION_FIELDS:
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise EventError(f"intervention.{field} must be a non-empty string")


def _check_truth_payload(kind: str, payload: dict[str, Any]) -> None:
    if kind == "world_state":
        _check_exact_keys(payload, ("targets", "world_counts"), "world_state")
        targets = payload["targets"]
        if not isinstance(targets, dict):
            raise EventError("world_state.targets must be an object")
        for name, entry in targets.items():
            if not isinstance(name, str) or not name.strip():
                raise EventError("world_state.targets keys must be non-empty strings")
            if not isinstance(entry, dict):
                raise EventError(f"world_state.targets[{name}] must be an object")
            _check_exact_keys(entry, ("present",), f"world_state.targets[{name}]")
            _check_flag(entry["present"], f"world_state.targets[{name}].present")
        counts = payload["world_counts"]
        if not isinstance(counts, dict):
            raise EventError("world_state.world_counts must be an object")
        for name, count in counts.items():
            if not isinstance(name, str) or not name.strip():
                raise EventError("world_state.world_counts keys must be non-empty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise EventError(f"world_state.world_counts[{name}] must be a non-negative integer")
        return
    if kind == "physical_outcome":
        _check_exact_keys(
            payload, ("inspected", "return_verified", "violations", "takeover"), "physical_outcome"
        )
        _check_named_flags(payload["inspected"], "physical_outcome.inspected")
        _check_flag(payload["return_verified"], "physical_outcome.return_verified")
        violations = payload["violations"]
        if not isinstance(violations, list) or any(
            not isinstance(entry, str) or not entry.strip() for entry in violations
        ):
            raise EventError("physical_outcome.violations must be a list of non-empty strings")
        _check_flag(payload["takeover"], "physical_outcome.takeover")
        return
    raise EventError(f"unknown truth-stream kind {kind!r}")


def validate_payload(kind: str, payload: Any) -> dict[str, Any]:
    """Check one payload against its kind's schema and return it as an object."""
    if not isinstance(payload, dict):
        raise EventError(f"a {kind} payload must be a JSON object")
    if kind in TRUTH_EVENT_KINDS:
        _check_truth_payload(kind, payload)
    elif kind in AGENT_EVENT_KINDS:
        if kind == "intervention":
            _check_intervention(payload)
        else:
            record_name = PAYLOAD_RECORDS[kind]
            try:
                from_dict(record_name, payload)
            except RecordError as error:
                raise EventError(f"payload of a {kind} event is not a valid {record_name}: {error}")
    else:
        raise EventError(f"unknown event kind {kind!r}")
    return payload


def build_event(
    seq: int,
    kind: str,
    payload: Any,
    stamp: ClockStamp,
    sim_time_s: float | None = None,
) -> Event:
    """Encode ``payload`` (a runtime record or a plain object) into an event."""
    if not isinstance(payload, dict):
        try:
            payload = to_dict(payload)
        except RecordError:
            raise EventError(f"cannot encode a {type(payload).__name__} as a {kind} payload")
    return Event(seq=seq, kind=kind, stamp=stamp, sim_time_s=sim_time_s, payload=payload)


def encode_event(event: Event) -> dict[str, Any]:
    """The JSON document for one event, every level plain JSON types."""
    return {
        "seq": event.seq,
        "kind": event.kind,
        "stamp": to_dict(event.stamp),
        "sim_time_s": event.sim_time_s,
        "payload": event.payload,
    }


def decode_event(
    document: Any, allowed_kinds: frozenset[str], stream: str
) -> Event:
    """Decode one line, refusing any kind outside the reading stream's vocabulary."""
    if not isinstance(document, dict):
        raise EventError(f"a {stream} event line must be a JSON object")
    missing = [field for field in _EVENT_FIELDS if field not in document]
    unknown = sorted(set(document) - set(_EVENT_FIELDS))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown {', '.join(unknown)}")
        raise EventError(f"a {stream} event must hold exactly {', '.join(_EVENT_FIELDS)} ({'; '.join(detail)})")
    kind = document["kind"]
    if not isinstance(kind, str) or not kind.strip():
        raise EventError("kind must be a non-empty string")
    if kind not in allowed_kinds:
        raise EventError(f"a {stream} event cannot be of kind {kind!r}")
    try:
        stamp = from_dict(ClockStamp, document["stamp"])
    except RecordError as error:
        raise EventError(f"stamp of a {kind} event is not a ClockStamp: {error}")
    return Event(
        seq=document["seq"],
        kind=kind,
        stamp=stamp,
        sim_time_s=document["sim_time_s"],
        payload=document["payload"],
    )


def check_stamp_order(previous: ClockStamp | None, new: ClockStamp) -> None:
    """The append rule: one clock domain, non-decreasing stamps."""
    if previous is None:
        return
    if (previous.host_id, previous.clock_id) != (new.host_id, new.clock_id):
        raise EventError(
            "one stream records one clock domain: cannot append "
            f"{new.host_id}/{new.clock_id} after {previous.host_id}/{previous.clock_id}"
        )
    if new.monotonic_ns < previous.monotonic_ns:
        raise EventError(
            f"events are appended in stamp order: {new.monotonic_ns} ns follows "
            f"{previous.monotonic_ns} ns"
        )


def verify_stream_order(events: Sequence[Event]) -> None:
    """The read rule, so a reordered or mixed-domain file cannot pose as one.

    Sequence numbers must be contiguous from zero, every event must share one
    clock domain, and stamps must be non-decreasing. Together with the append
    rule this pins intervention order: replay can only present the order in
    which the events were recorded, or refuse the file.
    """
    domain: tuple[str, str] | None = None
    previous_ns: int | None = None
    for position, event in enumerate(events):
        if event.seq != position:
            raise EpisodeError(
                f"event sequence {event.seq} sits at position {position}; "
                "the stream is not in recorded order"
            )
        current = (event.stamp.host_id, event.stamp.clock_id)
        if domain is None:
            domain = current
        elif current != domain:
            raise EpisodeError(
                f"events in one stream share one clock domain, found {domain} "
                f"and {current}"
            )
        if previous_ns is not None and event.stamp.monotonic_ns < previous_ns:
            raise EpisodeError(
                f"event {event.seq} is stamped earlier than the event before it; "
                "order and timestamps disagree"
            )
        previous_ns = event.stamp.monotonic_ns


def append_event_line(path: Path, event: Event) -> None:
    """Append one event as a single sorted JSON line."""
    line = json.dumps(encode_event(event), sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def read_event_lines(
    path: Path, allowed_kinds: frozenset[str], stream: str
) -> tuple[Event, ...]:
    """Read every line of one stream, in file order, or raise."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EpisodeError(f"{path} is missing") from None
    except OSError as error:
        raise EpisodeError(f"cannot read {path}: {error}") from error
    events = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise EventError(f"{path.name}:{number} is blank")
        try:
            document = json.loads(line)
        except json.JSONDecodeError as error:
            raise EventError(f"{path.name}:{number} is not JSON: {error}") from error
        events.append(decode_event(document, allowed_kinds, stream))
    return tuple(events)
