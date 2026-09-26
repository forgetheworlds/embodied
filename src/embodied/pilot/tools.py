"""The five tools the cloud may call: observe, ground, set_goal, status, cancel.

Five is the specification's interface choice (section 11), not a minimum. A
tool is a typed operation with a stated authority boundary:

* ``observe`` never moves the aircraft. It returns existing evidence, a
  pending request id, or an unmet information requirement with proposed
  observation objectives — movement is authorized only through ``set_goal``.
* ``ground`` returns hypotheses and geometry status, never a route or a
  safety claim. With no verified detector its honest answer is the
  ``detector_unavailable`` refusal, not a substitute.
* ``set_goal`` builds a proposal and returns a disposition; admission is the
  seam's, never the model's.
* ``status`` is read-only. ``cancel`` cancels only the goal it names.

Text seen in an image is scene evidence, never an instruction; nothing here
treats model-quoted signage as authority.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from embodied.contracts.records import ClockStamp, SpatialGoal, elapsed_ns

from embodied.pilot.broker import PilotBroker


# ---------------------------------------------------------------------------
# The tool schemas handed to the provider
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "observe",
            "description": "Request information. Never moves the aircraft: returns existing "
            "evidence, a pending request id, or an unmet requirement with proposed "
            "observation objectives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "max_age_s": {"type": "number"},
                    "detail": {"type": "string"},
                    "requires_movement": {
                        "type": "boolean",
                        "description": "true only when the answer cannot come from the "
                        "current pose; observe will not move for you",
                    },
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ground",
            "description": "Resolve a selection or description to target hypotheses with "
            "geometry status. No route or safety claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection_id": {"type": "string"},
                    "description": {"type": "string"},
                    "target_ref": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_goal",
            "description": "Propose a spatial objective. The local supervisor admits it; "
            "the reply carries the admission disposition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "proposal_id": {"type": "string"},
                    "base_goal_revision": {"type": "integer"},
                    "mission_revision": {"type": "integer"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "completion_condition": {"type": "string"},
                    "lease_bounds": {
                        "type": "object",
                        "description": "name -> positive seconds; an unbounded objective is refused",
                        "additionalProperties": {"type": "number"},
                    },
                    "local_discretion_bounds": {
                        "type": "object", "additionalProperties": {"type": "number"}
                    },
                    "selection_ids": {"type": "array", "items": {"type": "string"}},
                    "target_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["intent", "proposal_id", "completion_condition", "lease_bounds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "status",
            "description": "Read-only view of a goal's state, evidence and reasons.",
            "parameters": {
                "type": "object",
                "properties": {"goal_id": {"type": "string"}},
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel",
            "description": "Cancel exactly the named goal at its expected revision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
                    "expected_revision": {"type": "integer"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["goal_id", "expected_revision", "idempotency_key"],
            },
        },
    },
)


# ---------------------------------------------------------------------------
# Grounding seam (P03 owns the real grounding; the honest default is refusal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingResult:
    """Hypotheses with geometry status, or a named refusal. Never a route."""

    hypotheses: tuple[dict[str, Any], ...]
    status: str
    reason: str | None = None


class UnavailableGrounding:
    """The product default: the pinned dependency set has no verified detector,
    so grounding without one is refused by name (never a truth substitute)."""

    def ground(self, selection_id=None, description=None, target_ref=None) -> GroundingResult:
        return GroundingResult(
            hypotheses=(),
            status="unavailable",
            reason="detector_unavailable",
        )


# ---------------------------------------------------------------------------
# The tool surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObserveOutcome:
    state: str  # existing | pending | unmet
    observation_ids: tuple[str, ...] = ()
    request_id: str | None = None
    unmet_requirement: str | None = None
    proposed_objectives: tuple[str, ...] = ()


class ToolSurface:
    """Binds the five tool names to broker/seam methods. The model requests;
    this surface decides what each request is allowed to touch."""

    def __init__(
        self,
        broker: PilotBroker,
        grounding=None,
        now: Callable[[], ClockStamp] | None = None,
    ) -> None:
        self.broker = broker
        self.grounding = grounding if grounding is not None else UnavailableGrounding()
        self._now = (
            now if now is not None else lambda: ClockStamp(broker.host_id, broker.clock_id, 0)
        )

    # -- observe: check state first; never a hidden movement -----------------

    def observe(
        self, subject: str, max_age_s: float, detail: str = "any", requires_movement: bool = False
    ) -> ObserveOutcome:
        if requires_movement:
            return ObserveOutcome(
                state="unmet",
                unmet_requirement=(
                    f"answering {subject!r} requires movement; observe does not move the aircraft"
                ),
                proposed_objectives=(
                    f"approach a viewpoint observing {subject}",
                    f"inspect {subject} from the reached viewpoint",
                ),
            )
        now = self._now()
        freshness = min(max_age_s, self.broker.parameters.observation_freshness_s)
        existing = tuple(
            observation_id
            for observation_id, observation in self.broker.observations.items()
            if observation_id == subject
            and elapsed_ns(observation.capture_stamp, now) / 1_000_000_000 <= freshness
        )
        if existing and detail == "any":
            return ObserveOutcome(state="existing", observation_ids=existing)
        if existing:
            # The subject is known but not at the requested detail: the stored
            # evidence does not answer the question, so a capture is scheduled.
            return ObserveOutcome(
                state="pending",
                request_id=f"observe-{subject}-{now.monotonic_ns}",
                unmet_requirement=(
                    f"evidence for {subject!r} is present but not at detail {detail!r}; "
                    "a fresh capture from the current pose was scheduled"
                ),
            )
        if subject in self.broker.observations:
            return ObserveOutcome(
                state="pending",
                request_id=f"observe-{subject}-{now.monotonic_ns}",
                unmet_requirement=(
                    f"evidence for {subject!r} is older than {freshness}s; a fresh capture "
                    "from the current pose was scheduled"
                ),
            )
        return ObserveOutcome(
            state="unmet",
            unmet_requirement=(
                f"no evidence for {subject!r}; it is not visible from the current pose"
            ),
            proposed_objectives=(f"explore toward {subject}",),
        )

    # -- ground, set_goal, status, cancel ------------------------------------

    def ground(self, selection_id=None, description=None, target_ref=None) -> GroundingResult:
        return self.grounding.ground(
            selection_id=selection_id, description=description, target_ref=target_ref
        )

    def set_goal(
        self,
        intent: str,
        proposal_id: str,
        completion_condition: str,
        lease_bounds: dict[str, float],
        mission_revision: int = 0,
        base_goal_revision: int | None = None,
        constraints: tuple[str, ...] = (),
        local_discretion_bounds: dict[str, float] | None = None,
        selection_ids: tuple[str, ...] = (),
        target_refs: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        expected = (
            self.broker.active_revision if base_goal_revision is None else base_goal_revision
        )
        proposal = SpatialGoal(
            proposal_id=proposal_id,
            request_id=None,
            fingerprint=fingerprint_of(
                {
                    "intent": intent,
                    "proposal_id": proposal_id,
                    "completion": completion_condition,
                    "lease": sorted(lease_bounds.items()),
                    "selections": sorted(selection_ids),
                    "targets": sorted(target_refs),
                }
            ),
            mission_revision=mission_revision,
            base_goal_revision=expected,
            selection_ids=tuple(selection_ids),
            target_refs=tuple(target_refs),
            intent=intent,
            constraints=tuple(constraints),
            completion_condition=completion_condition,
            lease_bounds=tuple(sorted(lease_bounds.items())),
            local_discretion_bounds=tuple(sorted((local_discretion_bounds or {}).items())),
        )
        outcome = self.broker.set_goal(proposal, self._now())
        status = outcome.status
        return {
            "disposition": status.disposition.value,
            "admission_ref": status.admission_ref,
            "reason": status.reason,
            "current_disposition": status.current_disposition.value,
        }

    def status(self, goal_id: str) -> dict[str, Any]:
        execution = self.broker.status(goal_id)
        return {
            "goal_ref": execution.goal_ref,
            "disposition": execution.disposition.value,
            "evidence": list(execution.evidence),
            "reasons": list(execution.reasons),
            "capabilities": list(execution.capabilities),
        }

    def cancel(self, goal_id: str, expected_revision: int, idempotency_key: str) -> dict[str, Any]:
        outcome = self.broker.cancel(goal_id, expected_revision, idempotency_key, self._now())
        status = outcome.status
        return {
            "disposition": status.disposition.value,
            "reason": status.reason,
            "current_disposition": status.current_disposition.value,
        }


def fingerprint_of(content: Any) -> str:
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dispatch of model-requested tool calls
# ---------------------------------------------------------------------------

_TOOLS = {"observe", "ground", "set_goal", "status", "cancel"}


def dispatch_tool_call(surface: ToolSurface, call: dict[str, Any]) -> dict[str, Any]:
    """One model tool call, executed against its stated authority. A malformed
    or unknown call is a recorded error outcome, never an exception."""

    def error(message: str) -> dict[str, Any]:
        return {"error": message}

    name = call.get("function", {}).get("name")
    if name not in _TOOLS:
        return error(f"unknown tool {name!r}; the interface is exactly: {sorted(_TOOLS)}")
    raw = call.get("function", {}).get("arguments", "{}")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except json.JSONDecodeError as exc:
        return error(f"arguments are not JSON: {exc}")
    try:
        if name == "observe":
            outcome = surface.observe(
                subject=arguments["subject"],
                max_age_s=float(
                    arguments.get("max_age_s", surface.broker.parameters.observation_freshness_s)
                ),
                detail=str(arguments.get("detail", "any")),
                requires_movement=bool(arguments.get("requires_movement", False)),
            )
            return {
                "state": outcome.state,
                "observation_ids": list(outcome.observation_ids),
                "request_id": outcome.request_id,
                "unmet_requirement": outcome.unmet_requirement,
                "proposed_objectives": list(outcome.proposed_objectives),
            }
        if name == "ground":
            result = surface.ground(
                selection_id=arguments.get("selection_id"),
                description=arguments.get("description"),
                target_ref=arguments.get("target_ref"),
            )
            return {
                "hypotheses": list(result.hypotheses),
                "status": result.status,
                "reason": result.reason,
            }
        if name == "set_goal":
            return surface.set_goal(
                intent=arguments["intent"],
                proposal_id=arguments["proposal_id"],
                completion_condition=arguments["completion_condition"],
                lease_bounds=dict(arguments.get("lease_bounds") or {}),
                mission_revision=int(arguments.get("mission_revision", 0)),
                base_goal_revision=arguments.get("base_goal_revision"),
                constraints=tuple(arguments.get("constraints") or ()),
                local_discretion_bounds=dict(arguments.get("local_discretion_bounds") or {}),
                selection_ids=tuple(arguments.get("selection_ids") or ()),
                target_refs=tuple(arguments.get("target_refs") or ()),
            )
        if name == "status":
            return surface.status(arguments["goal_id"])
        return surface.cancel(
            goal_id=arguments["goal_id"],
            expected_revision=int(arguments["expected_revision"]),
            idempotency_key=arguments["idempotency_key"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return error(f"malformed arguments for {name}: {exc}")
