"""The provider adapter: request construction, transport and reply parsing.

The cloud pilot owns request lifecycle and evidence selection; this module owns
only the mechanics of talking to a multimodal provider. Everything timestamped
is handed in by the caller, so offline replays and the live probe share one
code path and no wall-clock read hides inside a latency number.

The transport seam is where degradation is injected. Every arm that makes a
cloud call at all goes through the same transport object — the deterministic
:class:`ScriptedTransport` in tests and replays, the live transport in the
probe — driven by the same broker, so injected delay, reordering, duplication
and silence are a property of the shared path rather than of any one
executive. B0 makes no cloud call, so the cloud-side injection is a no-op for
it by construction; the observation-side injection (scripted arrival stamps
evaluated against the shared freshness bound) applies to every arm through the
same broker inlet.

Model identity comes from ``configs/runtime-model.yaml``. Images travel as
base64 data URIs only: the project's measured finding (APPROVAL-RECORD) is
that this model family fails to fetch image URLs server-side.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.request
from typing import Any

from embodied.contracts.records import (
    ClockStamp,
    DecisionRequest,
    FinalReport,
    RecordError,
    SpatialGoal,
    elapsed_ns,
    from_dict,
)


class TransportError(Exception):
    """The transport could not complete a call. Recorded, never raised into the broker."""


# ---------------------------------------------------------------------------
# Model configuration (configs/runtime-model.yaml, section ``model``)
# ---------------------------------------------------------------------------


CALL_INITIAL = "initial"
CALL_CONTINUOUS = "continuous"
CALL_CLASSES = (CALL_INITIAL, CALL_CONTINUOUS)


@dataclass(frozen=True)
class CallProfile:
    """The declared shape of one class of cloud call.

    The owner's ruling of 2026-10-01 is the reason this exists: the initial
    mission interpretation needs deliberation, but the tactical updates that
    arrive while the aircraft is moving must fit the decision budget or the
    continuous-cloud arm cannot exist at all. One class therefore reasons and
    sees the captured frame; the other does not reason and delivers a smaller
    frame. The base ``model.generation`` still applies underneath both.
    """

    generation: tuple[tuple[str, Any], ...] = ()
    image_scale: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    """The configured runtime model, recorded rather than assumed.

    ``reply_format`` selects the reply-discipline variant of the prompt
    ("default", or "strict": exactly one tool call, no prose, short
    arguments). ``generation`` carries optional OpenAI-completions generation
    parameters (for example ``max_tokens`` or ``reasoning_effort``) that the
    configuration or a measurement run sets explicitly; an absent mapping adds
    no fields to the request. ``call_profiles`` carries those same parameters
    per call class, so the effort and the frame scale are a property of the
    call rather than of the process.
    """

    provider: str
    id: str
    base_url: str
    api: str
    image_transport: str
    reply_format: str = "default"
    generation: tuple[tuple[str, Any], ...] = ()
    call_profiles: tuple[tuple[str, CallProfile], ...] = ()

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "ModelConfig":
        for key in ("provider", "id", "base_url", "api", "image_transport"):
            if not isinstance(section.get(key), str) or not section[key].strip():
                raise ValueError(f"model.{key} must be a non-empty string")
        reply_format = section.get("reply_format", "default")
        if reply_format not in ("default", "strict"):
            raise ValueError("model.reply_format must be 'default' or 'strict'")
        generation_section = section.get("generation") or {}
        if not isinstance(generation_section, dict):
            raise ValueError("model.generation must be a mapping when present")
        generation = tuple(sorted((str(k), v) for k, v in generation_section.items()))
        return cls(
            provider=section["provider"],
            id=section["id"],
            base_url=section["base_url"],
            api=section["api"],
            image_transport=section["image_transport"],
            reply_format=reply_format,
            generation=generation,
            call_profiles=parse_call_profiles(section.get("call_profiles")),
        )

    def profile(self, call_class: str) -> CallProfile:
        """The declared profile for one call class.

        An undeclared class returns an empty profile, which is the correct
        answer for a configuration that draws no distinction and for every
        caller that predates the distinction: base generation, full frame.
        """
        for name, profile in self.call_profiles:
            if name == call_class:
                return profile
        return CallProfile()

    def generation_for(self, call_class: str) -> tuple[tuple[str, Any], ...]:
        """Base generation parameters with the class profile applied over them."""
        merged = dict(self.generation)
        merged.update(dict(self.profile(call_class).generation))
        return tuple(sorted(merged.items()))

    def image_scale_for(self, call_class: str) -> float:
        """The frame scale this class delivers its evidence at."""
        return self.profile(call_class).image_scale

    @property
    def identity(self) -> str:
        return self.id


def parse_call_profiles(raw: Any) -> tuple[tuple[str, CallProfile], ...]:
    """Parse ``model.call_profiles``: class -> {image_scale, <generation>}.

    ``image_scale`` is the one reserved key; every other key in a profile is a
    generation parameter, so the configuration reads as one description of a
    call rather than two parallel tables that can drift apart.
    """
    if raw in (None, {}):
        return ()
    if not isinstance(raw, dict):
        raise ValueError("model.call_profiles must be a mapping when present")
    profiles: list[tuple[str, CallProfile]] = []
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"model.call_profiles.{name} must be a mapping")
        scale = body.get("image_scale", 1.0)
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not 0.0 < float(scale) <= 1.0:
            raise ValueError(f"model.call_profiles.{name}.image_scale must be a number in (0, 1]")
        generation: list[tuple[str, Any]] = []
        for key, value in body.items():
            if key == "image_scale":
                continue
            # YAML 1.1 reads an unquoted `off`/`on`/`no`/`yes` as a boolean, so
            # `reasoning_effort: off` would reach the request body as `false`
            # rather than the string the API accepts, and nothing downstream
            # would complain. Measured the hard way on 2026-10-01.
            if key == "reasoning_effort" and not isinstance(value, str):
                raise ValueError(
                    f"model.call_profiles.{name}.reasoning_effort must be a quoted string "
                    f"(got {value!r}; an unquoted 'off' reads as the boolean false)"
                )
            generation.append((str(key), value))
        profiles.append(
            (str(name), CallProfile(generation=tuple(sorted(generation)), image_scale=float(scale)))
        )
    return tuple(sorted(profiles))

# ---------------------------------------------------------------------------
# Request envelope
# ---------------------------------------------------------------------------


def scale_frame_payload(payload: bytes, scale: float) -> bytes:
    """Downscale one captured frame and return PNG bytes; 1.0 passes through.

    This is the production path for the frame-scale lever, not a measurement
    knob. The continuous call class delivers its frame at the measured scale
    and the initial plan at captured resolution; both go through here, so what
    was measured is what a mission sends. Measured 2026-10-01 (J4): quarter
    scale took the continuous p50 from 5.35 s to 2.67 s with 10/10 decisions
    still naming real scene content. It does reduce the evidence the cloud
    sees, which is why the initial plan does not use it.
    """
    if scale == 1.0:
        return payload
    import io

    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        resized = image.resize((width, height))
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
    return buffer.getvalue()


def encode_image_data_uri(payload: bytes, *, scale: float = 1.0) -> str:
    """One base64 data URI, PNG, optionally downscaled first.

    The pinned commandcode route rejects ``image/ppm`` data URIs with HTTP 400
    regardless of size (measured 2026-10-01, the J3 diagnostic matrix: a 5.6 KB
    fixture PPM and a 2.46 MB captured stereo pair both refused; the same
    frames as PNG answered). Captures are P6 PPM, so they are re-encoded with
    Pillow — which also shrinks the upload about 4x. A payload that is already
    PNG passes through unchanged.
    """
    import base64

    payload = scale_frame_payload(payload, scale)
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        encoded = payload
    else:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.open(io.BytesIO(payload)).save(buffer, format="PNG")
        encoded = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


@dataclass(frozen=True)
class RequestPacket:
    """The evidence packet for one cloud request (specification section 9.1).

    ``image_parts`` pairs an observation id with the encoded data URI of the
    frame actually delivered; the mapping back to sensor pixels is retained by
    the caller that built the packet.

    ``call_class`` names which declared profile this call belongs to
    (``"initial"`` for the one-shot mission interpretation, ``"continuous"``
    for everything that arrives during motion). It defaults to continuous, so
    a packet built by a caller that predates the split is unchanged.
    """

    request: DecisionRequest
    mission_instruction: str
    image_parts: tuple[tuple[str, str], ...] = ()
    target_refs: tuple[str, ...] = ()
    navigation_status: str = "unknown"
    uncertainty_summary: str = "none recorded"
    explicit_question: str | None = None
    call_class: str = CALL_CONTINUOUS


def build_request_document(
    config: ModelConfig, packet: RequestPacket, tool_schemas: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """One OpenAI-completions chat document carrying text and base64 images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": _prompt_text(packet, config.reply_format)}]
    for observation_id, data_uri in packet.image_parts:
        content.append({"type": "text", "text": f"image for observation {observation_id}:"})
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
    document = {
        "model": config.id,
        "messages": [{"role": "user", "content": content}],
        "tools": list(tool_schemas),
        # The correlation identity travels with the request, not only in local state.
        "metadata": {
            "request_id": packet.request.request_id,
            "mission_revision": packet.request.mission_revision,
            "base_goal_revision": packet.request.base_goal_revision,
        },
    }
    # Generation parameters are explicit and recorded (configs/runtime-model.yaml
    # model.generation, with the packet's call class applied over it); never
    # inferred. The class is what makes "reason for the plan, but not for the
    # tactical update" a property of the call rather than of the process.
    document.update(dict(config.generation_for(packet.call_class)))
    return document


_STRICT_REPLY_LINES = (
    "Reply with EXACTLY ONE tool call and no prose. Keep every argument under 30 words.",
)


def _prompt_text(packet: RequestPacket, reply_format: str = "default") -> str:
    lines = [
        "You are the cloud pilot of an indoor drone. Reply with tool calls and, "
        "when a spatial objective should change, a spatial_goal proposal. "
        "You never command motion directly; the local supervisor owns admission.",
    ]
    if reply_format == "strict":
        lines.extend(_STRICT_REPLY_LINES)
    if packet.call_class == CALL_INITIAL:
        # The initial class is the one planning call, made on the ground before
        # takeoff. It answers with a mission_recipe document, and the question
        # that carries its schema is built by the caller that owns the recipe
        # vocabulary (pilot.flight_plan), so the vocabulary lives in one place.
        lines.append(
            "this call is the initial mission plan, made on the ground before takeoff: "
            "answer it with the mission_recipe JSON object the explicit question specifies, "
            "not with tool calls"
        )
    lines.extend(
        [
            f"mission instruction: {packet.mission_instruction}",
            f"mission revision: {packet.request.mission_revision}",
            f"active goal revision: {packet.request.base_goal_revision}",
            f"navigation status: {packet.navigation_status}",
            f"target references: {', '.join(packet.target_refs) or 'none'}",
            f"uncertainty: {packet.uncertainty_summary}",
        ]
    )
    if packet.explicit_question:
        lines.append(f"explicit question: {packet.explicit_question}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class LiveTransport:
    """stdlib transport; no new dependency.

    ``send`` performs the one blocking HTTP round trip and stores the result;
    ``poll`` yields it to the caller at the caller's clock. A browser
    User-Agent header is required by the provider edge (measured finding,
    APPROVAL-RECORD). The API key is read from the environment at call time
    and is never logged, returned or stored.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        api_key_env: str,
        timeout_s: float = 30.0,
        opener: Callable[[urllib.request.Request], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._opener = opener
        self._completed: list[tuple[ClockStamp, dict[str, Any] | TransportError]] = []

    def send(self, document: dict[str, Any], sent_at: ClockStamp) -> None:
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(document).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (embodied-pilot-probe)",
                "Authorization": f"Bearer {os.environ.get(self._api_key_env, '')}",
            },
            method="POST",
        )
        try:
            if self._opener is not None:
                response = self._opener(request)
            else:
                with urllib.request.urlopen(request, timeout=self._timeout_s) as handle:
                    response = json.loads(handle.read().decode("utf-8"))
            self._completed.append((sent_at, response))
        except urllib.error.HTTPError as error:
            # The body is the diagnosis: a 400 from this edge carries the
            # reason ("invalid_request_error: unsupported image media type")
            # that "transport failed: HTTP Error 400" hides. Bounded read.
            try:
                body = error.read(2048).decode("utf-8", "replace")
            except Exception:  # pragma: no cover - best-effort body recovery
                body = ""
            self._completed.append(
                (sent_at, TransportError(f"HTTP {error.code}: {body[:1000] or error.reason}"))
            )
        except (urllib.error.URLError, OSError, ValueError) as error:
            self._completed.append((sent_at, TransportError(f"transport failed: {error}")))

    def poll(self, now: ClockStamp) -> tuple["Arrival", ...]:
        arrivals = tuple(
            Arrival(send_stamp=send, arrived_at=now, document=payload)
            if isinstance(payload, dict)
            else Arrival(send_stamp=send, arrived_at=now, error=payload)
            for send, payload in self._completed
        )
        self._completed.clear()
        return arrivals


@dataclass(frozen=True)
class Arrival:
    """One reply that has arrived, or one transport failure, with its stamps."""

    send_stamp: ClockStamp
    arrived_at: ClockStamp
    document: dict[str, Any] | None = None
    error: TransportError | None = None
    usage: dict[str, Any] | None = None

    @property
    def round_trip_ns(self) -> int:
        return elapsed_ns(self.send_stamp, self.arrived_at)


@dataclass(frozen=True)
class ScriptedReply:
    """One scripted outcome, applied to the send with the matching call index.

    This is the shared degradation injector: delay, reordering (per-index
    delays), duplication, silence (a missing entry) and malformed output are
    scripted here, on the transport seam every cloud-calling arm shares.
    """

    delay_s: float = 0.0
    document: dict[str, Any] | None = None
    error: TransportError | None = None
    malformed: bool = False
    duplicate: bool = False
    usage: dict[str, Any] | None = None


class ScriptedTransport:
    """The deterministic fake. No network, no thread; a reply arrives when the
    harness's clock says it does. ``sent_documents`` and ``sent_stamps`` are
    the audit trail tests assert against."""

    def __init__(self) -> None:
        self.script: list[ScriptedReply] = []
        self.sent_documents: list[dict[str, Any]] = []
        self.sent_stamps: list[ClockStamp] = []
        self._arrivals: list[Arrival] = []

    def schedule(self, *replies: ScriptedReply) -> None:
        self.script.extend(replies)

    def send(self, document: dict[str, Any], sent_at: ClockStamp) -> None:
        index = len(self.sent_documents)
        self.sent_documents.append(document)
        self.sent_stamps.append(sent_at)
        if index >= len(self.script):
            return
        entry = self.script[index]
        arrival_ns = sent_at.monotonic_ns + int(entry.delay_s * 1_000_000_000)
        arrival_stamp = ClockStamp(sent_at.host_id, sent_at.clock_id, arrival_ns)
        payload = entry.document
        if entry.malformed and payload is None:
            payload = {"error": {"message": "scripted malformed reply"}}
        if payload is not None and entry.usage is not None:
            # Usage travels where the provider puts it: in the response body.
            payload = {**payload, "usage": entry.usage}
        if payload is not None or entry.error is not None:
            arrival = Arrival(
                send_stamp=sent_at,
                arrived_at=arrival_stamp,
                document=payload,
                error=entry.error,
                usage=entry.usage,
            )
            self._arrivals.append(arrival)
            if entry.duplicate:
                self._arrivals.append(arrival)

    def poll(self, now: ClockStamp) -> tuple[Arrival, ...]:
        due = [arrival for arrival in self._arrivals if arrival.arrived_at.monotonic_ns <= now.monotonic_ns]
        for arrival in due:
            self._arrivals.remove(arrival)
        return tuple(due)


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedReply:
    """What one provider reply carried, each part validated on its own terms.

    ``malformed_reason`` records unparseable output as an outcome; the broker
    never receives an exception from parsing. ``mission_recipe`` is B1's
    proposed recipe as a plain document; the executive re-validates its
    vocabulary and bounds before it becomes a MissionRecipe.
    """

    request_id: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    proposals: tuple[SpatialGoal, ...] = ()
    report: FinalReport | None = None
    mission_recipe: dict[str, Any] | None = None
    content: str | None = None
    usage: dict[str, Any] | None = None
    malformed_reason: str | None = None


def _unfence(content: str) -> str:
    """Strip a markdown code fence from a reply's JSON block.

    Measured 2026-10-01 on the pinned route: asked for one JSON object, the
    model returned it inside a ```json fence, so ``json.loads`` saw no JSON at
    all and a perfectly well-formed plan read as absent. The fence is a reply
    formatting artifact, not content: removing it changes what is parsed,
    never what is accepted — the parsed document still has to satisfy the
    record's own validation.
    """
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:] if lines else lines
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_reply(document: dict[str, Any], request: DecisionRequest) -> ParsedReply:
    try:
        choice = document["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ParsedReply(
            request_id=request.request_id,
            malformed_reason="reply is not an OpenAI-completions choice document",
        )
    tool_calls = tuple(choice.get("tool_calls") or ())
    content = choice.get("content")
    proposals: list[SpatialGoal] = []
    report: FinalReport | None = None
    recipe: dict[str, Any] | None = None
    malformed: list[str] = []
    try:
        structured = json.loads(_unfence(content)) if isinstance(content, str) else content
    except json.JSONDecodeError:
        # Free text is a valid answer: only a structured block that fails its
        # record's own validation is malformed.
        structured = None
    if isinstance(structured, dict):
        for index, candidate in enumerate(structured.get("spatial_goals") or []):
            try:
                proposals.append(from_dict(SpatialGoal, candidate))
            except RecordError as error:
                malformed.append(f"spatial_goals[{index}] refused by the record: {error}")
        if structured.get("final_report") is not None:
            try:
                report = from_dict(FinalReport, structured["final_report"])
            except RecordError as error:
                malformed.append(f"final_report refused by the record: {error}")
        if structured.get("mission_recipe") is not None:
            candidate = structured["mission_recipe"]
            if isinstance(candidate, dict) and isinstance(candidate.get("steps"), list):
                recipe = candidate
            else:
                malformed.append("mission_recipe must be an object with a steps list")
        unknown = set(structured) - {"spatial_goals", "final_report", "mission_recipe"}
        if unknown:
            malformed.append(f"unknown structured keys: {', '.join(sorted(unknown))}")
    return ParsedReply(
        request_id=request.request_id,
        tool_calls=tool_calls,
        proposals=tuple(proposals),
        report=report,
        mission_recipe=recipe,
        content=content if isinstance(content, str) else None,
        usage=document.get("usage") if isinstance(document.get("usage"), dict) else None,
        malformed_reason="; ".join(malformed) if malformed else None,
    )


# ---------------------------------------------------------------------------
# The provider the broker drives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SendRecord:
    """Latency accounting for one request (specification section 16.1)."""

    request_id: str
    send_stamp: ClockStamp
    request_bytes: int
    call_index: int


@dataclass(frozen=True)
class ProviderReply:
    """One parsed arrival correlated to its request by ``request_id``."""

    parsed: ParsedReply
    arrival: Arrival
    request: DecisionRequest
    response_bytes: int
    # Client-measured round trips combine network and server work; without a
    # provider-side breakdown the aggregate is reported as unseparated.
    latency_unseparated: bool = True


class Provider:
    """Constructs, sends and parses. Owns no lifecycle policy: one-outstanding,
    expiry and discard rules live in the broker."""

    def __init__(self, config: ModelConfig, transport, tool_schemas: Sequence[dict[str, Any]]) -> None:
        self.config = config
        self.transport = transport
        self.tool_schemas = tuple(tool_schemas)
        self.send_records: list[SendRecord] = []
        self.replies: list[ProviderReply] = []
        self.failures: list[tuple[str, str]] = []
        self._sent_requests: list[DecisionRequest] = []

    def submit(
        self, request: DecisionRequest, packet: RequestPacket, now: ClockStamp
    ) -> SendRecord:
        document = build_request_document(self.config, packet, self.tool_schemas)
        self.transport.send(document, now)
        record = SendRecord(
            request_id=request.request_id,
            send_stamp=now,
            request_bytes=len(json.dumps(document).encode("utf-8")),
            call_index=len(self.send_records),
        )
        self.send_records.append(record)
        self._sent_requests.append(request)
        return record

    def poll(self, now: ClockStamp) -> tuple[ProviderReply, ...]:
        replies = []
        for arrival in self.transport.poll(now):
            record = self._record_for(arrival.send_stamp)
            if record is None:
                continue
            request = self._sent_requests[record.call_index]
            if arrival.error is not None:
                self.failures.append((record.request_id, str(arrival.error)))
                continue
            parsed = parse_reply(arrival.document or {}, request)
            replies.append(
                ProviderReply(
                    parsed=parsed,
                    arrival=arrival,
                    request=request,
                    response_bytes=len(json.dumps(arrival.document or {}).encode("utf-8")),
                )
            )
        self.replies.extend(replies)
        return tuple(replies)

    def _record_for(self, send_stamp: ClockStamp) -> SendRecord | None:
        for record in self.send_records:
            if record.send_stamp == send_stamp:
                return record
        return None
