"""Recording, replay and the agent-facing projection of an episode directory.

Three responsibilities live here because they share one storage schema:

* :class:`Recorder` appends agent-stream events and the final report, and
  closes the episode with a :class:`RunManifest` whose artifact hashes bind the
  recorded files to the manifest.
* :class:`AgentSurface` is the projection a non-scorer reader is confined to:
  the manifest, the agent stream, the report and the referenced payloads. Any
  other member — a bench-private file, an annotation, an output of a later
  stage, a path that climbs out of the directory — is refused by name, and
  :meth:`AgentSurface.read_member` is the only generic member accessor.
* :func:`replay` reconstructs the recorded timeline for review: same order,
  same stamps, same revisions, no model call and no new physical run.

Ordering is enforced in both directions: the writer appends within one clock
domain with non-decreasing stamps, and the reader re-verifies contiguous
sequence numbers, one clock domain and non-decreasing stamps before anything
is rendered. A file that was reordered after the fact is refused rather than
silently re-sorted.

The record payloads are ``embodied.contracts.records`` records consumed
unchanged (RECORDS_REVISION). Schema revisions owned here: MANIFEST_REVISION
for the run manifest; the event envelope is ``embodied.bench.events``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any

from embodied.contracts.records import (
    ClockStamp,
    FinalReport,
    RECORDS_REVISION,
    RecordError,
    SensorMode,
    from_dict,
    to_dict,
)
from embodied.bench.events import (
    AGENT_EVENT_KINDS,
    EVENTS_REVISION,
    Event,
    EpisodeError,
    append_event_line,
    build_event,
    check_stamp_order,
    read_event_lines,
    verify_stream_order,
)

MANIFEST_REVISION = "p02-manifest-1"
AGENT_EVENTS_FILENAME = "agent-events.jsonl"
MANIFEST_FILENAME = "manifest.json"
FINAL_REPORT_FILENAME = "final-report.json"
PAYLOAD_DIRECTORY = "payloads"

# What a non-scorer reader may reach in an episode directory, and nothing
# else. The manifest must declare exactly this projection, so a manifest can
# never quietly widen what the runtime side is handed.
PROJECTION: tuple[str, ...] = (
    "manifest.json",
    "agent-events.jsonl",
    "final-report.json",
    "payloads/",
)

_EPISODE_KINDS = ("synthetic-fixture", "physical-run")


class SurfaceViolation(EpisodeError):
    """A reader asked for an episode member outside the agent projection."""


# ---------------------------------------------------------------------------
# Suites: a live recording refuses anything that is not registered here
# ---------------------------------------------------------------------------


class SuiteError(Exception):
    """A suite cannot be resolved, so a live episode must not be recorded."""


@dataclass(frozen=True)
class SuiteSpec:
    """What a registered suite must declare before a live run may record.

    The provider budget and the localization mode are required fields rather
    than optional ones: a suite missing either cannot be constructed, which is
    how ``bench record`` refuses missing budget or localization instead of
    substituting a fake episode. Suites are registered by the stage that owns
    them (P05 registers first-indoor); P02 registers none.
    """

    name: str
    registered_by: str
    provider_budget: dict[str, float]
    localization_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SuiteError("a suite needs a non-empty name")
        if not isinstance(self.registered_by, str) or not self.registered_by.strip():
            raise SuiteError(f"suite {self.name!r} needs the stage that registered it")
        if not isinstance(self.provider_budget, dict) or not self.provider_budget:
            raise SuiteError(f"suite {self.name!r} must declare a provider budget")
        for key, value in self.provider_budget.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise SuiteError(f"suite {self.name!r} budget {key!r} must be a positive number")
        if not isinstance(self.localization_mode, str) or not self.localization_mode.strip():
            raise SuiteError(f"suite {self.name!r} must declare its localization mode")


SUITE_REGISTRY: dict[str, SuiteSpec] = {}


def register_suite(spec: SuiteSpec) -> None:
    """Register one suite. Two registrations of a name are a conflict."""
    if spec.name in SUITE_REGISTRY:
        raise SuiteError(f"suite {spec.name!r} is already registered")
    SUITE_REGISTRY[spec.name] = spec


def resolve_suite(name: str) -> SuiteSpec:
    """Return a registered suite, or refuse with the reason a run is blocked."""
    try:
        return SUITE_REGISTRY[name]
    except KeyError:
        registered = ", ".join(sorted(SUITE_REGISTRY)) or "none"
        raise SuiteError(
            f"suite {name!r} is not registered; registered suites: {registered} "
            "(P05 registers first-indoor — refusing rather than substituting a fake)"
        ) from None


# ---------------------------------------------------------------------------
# JSON and hashing helpers used by the agent-facing side
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise EpisodeError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EpisodeError(f"{path} is missing") from None
    except OSError as error:
        raise EpisodeError(f"cannot read {path}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise EpisodeError(f"{path} is not JSON: {error}") from error


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    """Configuration identity of one recorded episode.

    ``None`` means the value is unavailable, never a default: a synthetic
    fixture flew nothing, so it names no suite, no arm, no sensor mode and no
    model. ``artifacts`` hashes only agent-projection members — the manifest
    binds what the runtime side can read, and it lists no bench-private file,
    so it cannot point a reader at one either.
    """

    episode_id: str
    episode_kind: str
    suite: str | None = None
    trial_group_id: str | None = None
    arm: str | None = None
    sensor_mode: SensorMode | None = None
    code_revision: str | None = None
    config_hash: str | None = None
    model_identity: str | None = None
    manifest_revision: str = MANIFEST_REVISION
    records_revision: str = RECORDS_REVISION
    events_revision: str = EVENTS_REVISION
    permitted_projection: tuple[str, ...] = PROJECTION
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise EpisodeError("a manifest needs a non-empty episode_id")
        if self.episode_kind not in _EPISODE_KINDS:
            raise EpisodeError(f"episode_kind must be one of {', '.join(_EPISODE_KINDS)}")
        for name in ("suite", "trial_group_id", "arm", "code_revision", "config_hash", "model_identity"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise EpisodeError(f"{name} must be a non-empty string or None")
        if self.sensor_mode is not None and not isinstance(self.sensor_mode, SensorMode):
            raise EpisodeError("sensor_mode must be a SensorMode or None")
        if tuple(self.permitted_projection) != PROJECTION:
            raise EpisodeError(
                "a manifest may only declare the agent projection this recorder "
                f"implements: {list(PROJECTION)}"
            )
        for member in self.artifacts:
            if member in (AGENT_EVENTS_FILENAME, FINAL_REPORT_FILENAME):
                continue
            prefix = PAYLOAD_DIRECTORY + "/"
            if member.startswith(prefix) and "/" not in member[len(prefix):]:
                continue
            raise EpisodeError(
                f"manifest artifacts must be agent-projection members, got {member!r}"
            )


_MANIFEST_FIELDS = (
    "manifest_revision",
    "records_revision",
    "events_revision",
    "episode_id",
    "episode_kind",
    "suite",
    "trial_group_id",
    "arm",
    "sensor_mode",
    "code_revision",
    "config_hash",
    "model_identity",
    "permitted_projection",
    "artifacts",
)


def manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    return {
        "manifest_revision": manifest.manifest_revision,
        "records_revision": manifest.records_revision,
        "events_revision": manifest.events_revision,
        "episode_id": manifest.episode_id,
        "episode_kind": manifest.episode_kind,
        "suite": manifest.suite,
        "trial_group_id": manifest.trial_group_id,
        "arm": manifest.arm,
        "sensor_mode": manifest.sensor_mode.value if manifest.sensor_mode else None,
        "code_revision": manifest.code_revision,
        "config_hash": manifest.config_hash,
        "model_identity": manifest.model_identity,
        "permitted_projection": list(manifest.permitted_projection),
        "artifacts": dict(manifest.artifacts),
    }


def manifest_from_dict(document: Any) -> RunManifest:
    if not isinstance(document, dict):
        raise EpisodeError("manifest.json must hold an object")
    missing = [name for name in _MANIFEST_FIELDS if name not in document]
    unknown = sorted(set(document) - set(_MANIFEST_FIELDS))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown {', '.join(unknown)}")
        raise EpisodeError(f"manifest.json must hold exactly {', '.join(_MANIFEST_FIELDS)} ({'; '.join(detail)})")
    sensor_mode = document["sensor_mode"]
    if sensor_mode is not None:
        try:
            sensor_mode = SensorMode(sensor_mode)
        except ValueError:
            raise EpisodeError(f"manifest sensor_mode {sensor_mode!r} is not a SensorMode") from None
    projection = document["permitted_projection"]
    if not isinstance(projection, list) or any(not isinstance(entry, str) for entry in projection):
        raise EpisodeError("manifest permitted_projection must be a list of names")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in artifacts.items()
    ):
        raise EpisodeError("manifest artifacts must map member names to hash strings")
    return RunManifest(
        episode_id=document["episode_id"],
        episode_kind=document["episode_kind"],
        suite=document["suite"],
        trial_group_id=document["trial_group_id"],
        arm=document["arm"],
        sensor_mode=sensor_mode,
        code_revision=document["code_revision"],
        config_hash=document["config_hash"],
        model_identity=document["model_identity"],
        manifest_revision=document["manifest_revision"],
        records_revision=document["records_revision"],
        events_revision=document["events_revision"],
        permitted_projection=tuple(projection),
        artifacts=dict(artifacts),
    )


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Append-only writer for one episode's agent stream.

    The recorder has no method and no attribute for a bench-private stream:
    its whole vocabulary is :data:`AGENT_EVENT_KINDS`, the report file and the
    payload directory, all inside the projection.
    """

    def __init__(self, episode_dir: Path) -> None:
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        if (self.episode_dir / MANIFEST_FILENAME).exists():
            raise EpisodeError(
                f"{self.episode_dir} already holds a manifest; a closed episode is not reopened"
            )
        self.events_path = self.episode_dir / AGENT_EVENTS_FILENAME
        if self.events_path.exists():
            existing = read_event_lines(self.events_path, AGENT_EVENT_KINDS, "agent")
            verify_stream_order(existing)
            self._next_seq = len(existing)
            self._last_stamp: ClockStamp | None = existing[-1].stamp
        else:
            self._next_seq = 0
            self._last_stamp = None
        self._closed = False

    def _guard_open(self) -> None:
        if self._closed:
            raise EpisodeError("the episode is closed; open a new recorder to record more")

    def write_payload(self, name: str, data: bytes) -> str:
        """Write one referenced-evidence file; return its episode-relative name."""
        self._guard_open()
        if not name or Path(name).name != name:
            raise EpisodeError(f"a payload name must be a plain filename, got {name!r}")
        directory = self.episode_dir / PAYLOAD_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
        return f"{PAYLOAD_DIRECTORY}/{name}"

    def record(
        self,
        kind: str,
        payload: Any,
        stamp: ClockStamp,
        sim_time_s: float | None = None,
    ) -> Event:
        """Append one agent-stream event."""
        self._guard_open()
        if kind not in AGENT_EVENT_KINDS:
            raise EpisodeError(f"{kind!r} is not an agent-stream event kind")
        if kind == "report":
            raise EpisodeError("record a report with write_report() so the event and file cannot disagree")
        check_stamp_order(self._last_stamp, stamp)
        event = build_event(self._next_seq, kind, payload, stamp, sim_time_s)
        append_event_line(self.events_path, event)
        self._last_stamp = stamp
        self._next_seq += 1
        return event

    def write_report(
        self, report: FinalReport, stamp: ClockStamp, sim_time_s: float | None = None
    ) -> Event:
        """Write the final report once: as the last stream event and as the file."""
        self._guard_open()
        if not isinstance(report, FinalReport):
            raise EpisodeError("write_report takes a FinalReport record")
        if (self.episode_dir / FINAL_REPORT_FILENAME).exists():
            raise EpisodeError("the episode already has a final report")
        check_stamp_order(self._last_stamp, stamp)
        event = build_event(self._next_seq, "report", report, stamp, sim_time_s)
        write_json(self.episode_dir / FINAL_REPORT_FILENAME, to_dict(report))
        append_event_line(self.events_path, event)
        self._last_stamp = stamp
        self._next_seq += 1
        return event

    def close(self, manifest: RunManifest) -> RunManifest:
        """Hash the agent-side files into the manifest and close the episode.

        The manifest is written last: an episode without one is incomplete and
        every reader refuses it.
        """
        self._guard_open()
        artifacts: dict[str, str] = {}
        events_file = self.episode_dir / AGENT_EVENTS_FILENAME
        if events_file.is_file():
            artifacts[AGENT_EVENTS_FILENAME] = sha256(events_file)
        report_file = self.episode_dir / FINAL_REPORT_FILENAME
        if report_file.is_file():
            artifacts[FINAL_REPORT_FILENAME] = sha256(report_file)
        payload_dir = self.episode_dir / PAYLOAD_DIRECTORY
        if payload_dir.is_dir():
            for member in sorted(payload_dir.iterdir()):
                if member.is_file():
                    artifacts[f"{PAYLOAD_DIRECTORY}/{member.name}"] = sha256(member)
        closed = replace(manifest, artifacts=artifacts)
        write_json(self.episode_dir / MANIFEST_FILENAME, manifest_to_dict(closed))
        self._closed = True
        return closed


# ---------------------------------------------------------------------------
# Agent-facing projection
# ---------------------------------------------------------------------------


class AgentSurface:
    """The projection a non-scorer reader is confined to.

    ``read_member`` accepts only names inside :data:`PROJECTION`, matched
    literally against the manifest's own artifact list, so neither a
    bench-private filename nor a path traversal can be reached through it.
    """

    def __init__(self, episode_dir: Path, manifest: RunManifest) -> None:
        self.episode_dir = Path(episode_dir)
        self.manifest = manifest
        self._events: tuple[Event, ...] | None = None

    @classmethod
    def open(cls, episode_dir: Path) -> "AgentSurface":
        path = Path(episode_dir)
        if not path.is_dir():
            raise EpisodeError(f"{path} is not an episode directory")
        manifest_path = path / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise EpisodeError(f"{path} has no manifest.json — the episode is incomplete")
        return cls(path, manifest_from_dict(read_json(manifest_path)))

    def agent_events(self) -> tuple[Event, ...]:
        """Every agent-stream event, order-verified, in recorded order."""
        if self._events is None:
            events = read_event_lines(
                self.episode_dir / AGENT_EVENTS_FILENAME, AGENT_EVENT_KINDS, "agent"
            )
            verify_stream_order(events)
            self._events = events
        return self._events

    def final_report(self) -> FinalReport | None:
        """The final report, cross-checked against its stream event.

        The file and the event are written together by
        :meth:`Recorder.write_report`; here they must agree, so neither can be
        edited after the episode closed without detection.
        """
        report_path = self.episode_dir / FINAL_REPORT_FILENAME
        events = [event for event in self.agent_events() if event.kind == "report"]
        if report_path.is_file():
            try:
                report = from_dict(FinalReport, read_json(report_path))
            except RecordError as error:
                raise EpisodeError(f"{FINAL_REPORT_FILENAME} is not a FinalReport: {error}")
            if len(events) != 1:
                raise EpisodeError(
                    "final-report.json and the stream disagree about the report: "
                    f"{len(events)} report events"
                )
            if to_dict(report) != events[0].payload:
                raise EpisodeError("final-report.json does not match the report event")
            return report
        if events:
            raise EpisodeError("the stream has a report event but the episode has no final-report.json")
        return None

    def payload_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                member
                for member in self.manifest.artifacts
                if member.startswith(PAYLOAD_DIRECTORY + "/")
            )
        )

    def read_payload(self, name: str) -> bytes:
        if name not in self.payload_names():
            raise SurfaceViolation(f"{name!r} is not a payload recorded in the manifest")
        return (self.episode_dir / name).read_bytes()

    def read_member(self, member: str) -> bytes:
        """The only generic member accessor, and it enforces the projection."""
        if member in (MANIFEST_FILENAME, AGENT_EVENTS_FILENAME, FINAL_REPORT_FILENAME):
            path = self.episode_dir / member
            if not path.is_file():
                raise EpisodeError(f"{member} is missing from the episode")
            return path.read_bytes()
        if member in self.payload_names():
            return (self.episode_dir / member).read_bytes()
        raise SurfaceViolation(f"{member!r} is outside the agent projection {list(PROJECTION)}")

    def verify_artifacts(self) -> None:
        """The manifest's hashes must match the agent-side files on disk."""
        on_disk: set[str] = set()
        for member in (AGENT_EVENTS_FILENAME, FINAL_REPORT_FILENAME):
            if (self.episode_dir / member).is_file():
                on_disk.add(member)
        payload_dir = self.episode_dir / PAYLOAD_DIRECTORY
        if payload_dir.is_dir():
            on_disk.update(
                f"{PAYLOAD_DIRECTORY}/{member.name}"
                for member in payload_dir.iterdir()
                if member.is_file()
            )
        recorded = set(self.manifest.artifacts)
        if recorded != on_disk:
            raise EpisodeError(
                "manifest artifacts do not match the episode's agent-side files: "
                f"recorded {sorted(recorded)}, on disk {sorted(on_disk)}"
            )
        for member, expected in sorted(self.manifest.artifacts.items()):
            actual = sha256(self.episode_dir / member)
            if actual != expected:
                raise EpisodeError(
                    f"artifact {member} does not match the hash recorded in the manifest "
                    f"(recorded {expected[:12]}…, on disk {actual[:12]}…)"
                )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _summary(event: Event) -> dict[str, Any]:
    """The fields a reviewer checks for one kind of event."""
    payload = event.payload
    kind = event.kind
    if kind == "mission":
        return {"mission_id": payload["mission_id"], "revision": payload["revision"]}
    if kind == "observation":
        return {
            "record_id": payload["record_id"],
            "sequence": payload["sequence"],
            "pair_id": payload["pair_id"],
        }
    if kind == "request":
        return {"request_id": payload["request_id"], "sequence": payload["sequence"]}
    if kind == "selection":
        return {
            "selection_id": payload["selection_id"],
            "observation_id": payload["observation_id"],
        }
    if kind == "goal":
        return {
            "proposal_id": payload["proposal_id"],
            "request_id": payload["request_id"],
            "base_goal_revision": payload["base_goal_revision"],
            "intent": payload["intent"],
        }
    if kind == "goal_status":
        return {
            "proposal_id": payload["proposal_id"],
            "disposition": payload["disposition"],
            "current_disposition": payload["current_disposition"],
        }
    if kind == "setpoint":
        return {
            "command_sequence": payload["command_sequence"],
            "goal_revision": payload["goal_revision"],
            "nav_epoch": payload["nav_epoch"],
            "frame": payload["frame"],
            "position_ned": payload["target"]["position_ned"],
        }
    if kind == "execution":
        return {"goal_ref": payload["goal_ref"], "disposition": payload["disposition"]}
    if kind == "intervention":
        return {
            "intervention_id": payload["intervention_id"],
            "actor": payload["actor"],
            "category": payload["category"],
        }
    if kind == "report":
        return {
            "claims": len(payload["claims"]),
            "termination_reason": payload["termination_reason"],
        }
    raise EpisodeError(f"no replay summary is defined for kind {kind!r}")


def replay(episode_dir: Path) -> dict[str, Any]:
    """Reconstruct the recorded timeline: order, stamps, revisions, results.

    Purely offline: no model call, no new physical run, and no member outside
    the agent projection is opened.
    """
    surface = AgentSurface.open(episode_dir)
    surface.verify_artifacts()
    events = surface.agent_events()  # order-verified inside
    report = surface.final_report()
    first, last = events[0].stamp, events[-1].stamp
    counts: dict[str, int] = {}
    timeline = []
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
        timeline.append(
            {
                "seq": event.seq,
                "kind": event.kind,
                "host_id": event.stamp.host_id,
                "clock_id": event.stamp.clock_id,
                "monotonic_ns": event.stamp.monotonic_ns,
                "sim_time_s": event.sim_time_s,
                "summary": _summary(event),
            }
        )
    return {
        "episode_id": surface.manifest.episode_id,
        "episode_kind": surface.manifest.episode_kind,
        "sensor_mode": surface.manifest.sensor_mode.value if surface.manifest.sensor_mode else None,
        "records_revision": surface.manifest.records_revision,
        "events_revision": surface.manifest.events_revision,
        "clock_domain": {"host_id": first.host_id, "clock_id": first.clock_id},
        "event_count": len(events),
        "span_ns": last.monotonic_ns - first.monotonic_ns,
        "counts": dict(sorted(counts.items())),
        "report": None
        if report is None
        else {"claims": len(report.claims), "termination_reason": report.termination_reason},
        "timeline": timeline,
    }


def episode_members(episode_dir: Path) -> Sequence[str]:
    """Every regular file directly in the episode directory, sorted by name.

    Tests use this to prove that a read-only command left an episode alone.
    """
    path = Path(episode_dir)
    return tuple(sorted(entry.name for entry in path.iterdir() if entry.is_file()))
