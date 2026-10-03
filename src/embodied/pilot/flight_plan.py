"""The pre-flight plan: one reasoned call on the ground, then never again.

The owner's ruling of 2026-10-01: "the initial high level goal is sent before
the drone lifts off with any available information it can gather, and then
after that reasoning is off." Measured on the pinned route the initial class
answers in p50 20.2 s with 3 of 12 calls past a 30 s read timeout — a cost
that is intolerable while the aircraft holds a decision horizon and free
while it sits on the ground, where a retry costs only time.

This module turns that sequencing into objects:

* :func:`plan_from_reply` validates the model's proposal into the same
  bounded :class:`MissionRecipe` the conventional arm runs, with the same
  declared effort — so a measured difference between arms is the decision
  source and not a longer plan.
* :class:`PreflightPlanner` issues exactly one initial-class call, may retry
  a transport failure up to the declared retry budget, and returns a
  :class:`FlightPlan` whether or not the answer was usable — a refusal is a
  recorded outcome, never an exception into the mission.

The plan is a recipe, not a route: intents, deferred target selectors and
guards, executed by the shared runner under the same admission seam. The
cloud never commands motion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from embodied.contracts.records import ClockStamp, DecisionRequest
from embodied.pilot.mission_packet import MissionPacketBuilder
from embodied.pilot.provider import CALL_INITIAL, Provider
from embodied.pilot.recipe_runner import (
    GUARDS,
    INTENTS,
    Guard,
    MissionRecipe,
    RecipeStep,
    TargetSelector,
)

# ---------------------------------------------------------------------------
# Declared effort: the plan is bounded by what the conventional arm declared
# ---------------------------------------------------------------------------

# B0's own policy (pilot/mission.py) declares 4 explore steps + 1 inspect +
# 1 return = 6 steps, and at most 2 attempts per step. The cloud's plan gets
# exactly the same envelope. These are declared engineering parameters (R2):
# a longer cloud plan would confound the P06 comparison it exists for, and
# the numbers are the conventional arm's own, not new ones.
PLAN_STEPS_CAP = 6
PLAN_STEP_ATTEMPTS_CAP = 2
# A resource ceiling must be positive and no larger than the conventional
# arm's whole-mission ceiling (pilot/mission.py's three recipes sum to 6.0).
PLAN_RESOURCE_CEILING_CAP = 6.0

_TARGET_KINDS = ("candidate", "frontier", "place")


# The plan arrives as a typed tool call. Measured 2026-10-01 on the pinned
# route: four prose attempts each invented a different schema (first its own
# inner keys, then a top-level plan object with phases and abort conditions),
# while the same model, asked through this schema, returned the runner's own
# vocabulary in 6.99 s and 369 output tokens instead of 1,887. The schema is
# the interface, so the vocabulary is the runner's, imported rather than
# retyped.
RECIPE_TOOL_NAME = "propose_mission_recipe"

RECIPE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": RECIPE_TOOL_NAME,
        "description": (
            "Propose the bounded mission plan for the instruction as an ordered list of "
            "steps plus its bounds. Call this exactly once with the whole plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": list(INTENTS)},
                            "target_kind": {
                                "type": "string",
                                "enum": ["candidate", "frontier", "place"],
                            },
                            "target_ref": {"type": "string"},
                            "guard_kind": {"type": "string", "enum": list(GUARDS)},
                            "max_attempts": {"type": "integer", "minimum": 1, "maximum": 2},
                        },
                        "required": ["action", "max_attempts"],
                    },
                },
                "bounds": {
                    "type": "object",
                    "properties": {
                        "max_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": PLAN_STEPS_CAP,
                        },
                        "resource_ceiling": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": PLAN_RESOURCE_CEILING_CAP,
                        },
                    },
                    "required": ["max_steps", "resource_ceiling"],
                },
            },
            "required": ["steps", "bounds"],
        },
    },
}


def recipe_document_from(parsed) -> dict[str, Any] | None:
    """The recipe a reply proposed, from its content or from the typed tool call.

    Both shapes are validated identically. The typed call is the one that
    measured working: every prose answer invented a schema of its own.
    """
    if parsed.mission_recipe is not None:
        return parsed.mission_recipe
    for call in getattr(parsed, "tool_calls", ()) or ():
        function = (call or {}).get("function") or {}
        if function.get("name") != RECIPE_TOOL_NAME:
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            return None
        return arguments if isinstance(arguments, dict) else None
    return None



class PlanRefused(Exception):
    """The reply's plan cannot be used. Recorded with the reason."""


@dataclass(frozen=True)
class FlightPlan:
    """What the one pre-flight call produced, usable or not.

    ``usable`` False with a ``refusal_reason`` is a complete, honest outcome:
    the mission may still fly on the local fallback recipe, and the receipt
    records that the cloud's plan was refused rather than quietly replaced.
    """

    usable: bool
    recipe: MissionRecipe | None
    refusal_reason: str | None
    request_id: str
    model_identity: str | None
    trace: dict[str, Any] | None
    attempts: int
    round_trip_s: float | None

    def document(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "refusal_reason": self.refusal_reason,
            "request_id": self.request_id,
            "model_identity": self.model_identity,
            "trace": self.trace,
            "attempts": self.attempts,
            "round_trip_s": self.round_trip_s,
            "recipe_source": self.recipe.source if self.recipe else None,
            "recipe_steps": [step.action for step in self.recipe.steps] if self.recipe else [],
        }


def plan_question(instruction: str) -> str:
    """The plan request, carrying the schema the runner will actually enforce.

    Written here rather than in the provider's prompt because this module owns
    the recipe vocabulary: the actions, target kinds and guards it names are
    imported from the runner, so the question cannot drift from what
    :func:`plan_from_reply` accepts.

    Three measured failures shaped this text (2026-10-01, one route, one frame):
    with the generic pilot prompt the model proposed no recipe at all; asked
    for a schema written with ``<placeholder>`` terms it answered with its own
    inner keys (``objective``, ``start_pose``, ``frame``); and it wrapped the
    object in a markdown fence. So the shape is shown as a concrete worked
    example in the runner's own vocabulary, and it is the entire reply.
    """
    return (
        "You are on the ground before takeoff and nothing has moved yet. Call "
        f"{RECIPE_TOOL_NAME} exactly once with the whole plan for the instruction below, and "
        "reply with nothing else. A step says where to act relative to evidence the local "
        "system will find later — a frontier, the next uninspected candidate, or the start "
        "place — so do not name a route, a doorway order, a position or where any object is. "
        f"Instruction: {instruction}"
    )


def plan_from_reply(parsed, *, source: str = "cloud-initial") -> MissionRecipe:
    """Validate a reply's ``mission_recipe`` into a bounded MissionRecipe.

    Every step is re-checked against the vocabulary the runner enforces, and
    the whole plan against the conventional arm's declared effort, so the
    model can propose but cannot widen its own envelope.
    """
    document = recipe_document_from(parsed)
    if document is None:
        raise PlanRefused("the reply proposed no mission_recipe document")
    raw_steps = document.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanRefused("mission_recipe.steps must be a non-empty list")
    if len(raw_steps) > PLAN_STEPS_CAP:
        raise PlanRefused(
            f"the plan carries {len(raw_steps)} steps against the declared effort of "
            f"{PLAN_STEPS_CAP}; a longer plan is not a better plan here"
        )
    steps: list[RecipeStep] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict) or "action" not in raw:
            raise PlanRefused(f"steps[{index}] is not an action object")
        target = None
        kind = raw.get("target_kind")
        if kind:
            if kind not in _TARGET_KINDS:
                raise PlanRefused(
                    f"steps[{index}].target_kind {kind!r} is not one of {', '.join(_TARGET_KINDS)}"
                )
            target = TargetSelector(kind, str(raw.get("target_ref", "")))
        attempts = int(raw.get("max_attempts", 1))
        if not 1 <= attempts <= PLAN_STEP_ATTEMPTS_CAP:
            raise PlanRefused(
                f"steps[{index}].max_attempts {attempts} is outside the declared "
                f"1..{PLAN_STEP_ATTEMPTS_CAP}"
            )
        steps.append(
            RecipeStep(
                action=raw["action"],
                target=target,
                guard=Guard(raw.get("guard_kind", "always"), raw.get("guard_ref")),
                max_attempts=attempts,
                completion=raw.get("completion", "reached the declared local relationship"),
                step_cost=float(raw.get("step_cost", 1.0)),
            )
        )
    bounds = document.get("bounds") or {}
    max_steps = int(bounds.get("max_steps", len(steps)))
    # The declared bound is PLAN_STEPS_CAP, the conventional arm's own total
    # effort. An earlier version of this rule required max_steps <= len(steps),
    # which is stricter than anything declared: the measured 2026-10-01 probe
    # showed the model proposing a sound 3-step plan with max_steps 6 — headroom
    # for the runner to repeat a step, which is a bounded loop the spec allows.
    # The declared cap is what binds, and it still does.
    if not 1 <= max_steps <= PLAN_STEPS_CAP:
        raise PlanRefused(
            f"bounds.max_steps {max_steps} is outside the declared 1..{PLAN_STEPS_CAP}"
        )
    ceiling = float(bounds.get("resource_ceiling", float(len(steps))) or float(len(steps)))
    if not 0.0 < ceiling <= PLAN_RESOURCE_CEILING_CAP:
        raise PlanRefused(
            f"bounds.resource_ceiling {ceiling} is outside the declared "
            f"(0, {PLAN_RESOURCE_CEILING_CAP}]"
        )
    # MissionRecipe's own __post_init__ re-validates the vocabulary; building
    # it here is the check, and its ValueError becomes a PlanRefused so a
    # refusal is always an outcome, never an exception into the mission.
    try:
        return MissionRecipe(
            steps=tuple(steps),
            max_steps=max_steps,
            resource_ceiling=ceiling,
            source=source,
        )
    except ValueError as error:
        raise PlanRefused(str(error)) from error


class PreflightPlanner:
    """Issues the one reasoned call, on the ground, and holds its answer."""

    def __init__(
        self,
        provider: Provider,
        builder: MissionPacketBuilder,
        *,
        retry_budget: int = 0,
    ) -> None:
        self.provider = provider
        self.builder = builder
        self.retry_budget = retry_budget

    def plan(
        self,
        *,
        observation,
        payloads: dict[str, bytes],
        now: ClockStamp,
        deadline_s: float,
        sequence: int = 0,
        mission_revision: int = 0,
        base_goal_revision: int = 0,
        explicit_question: str | None = None,
    ) -> FlightPlan:
        """One initial-class call, retried only on a transport failure.

        ``deadline_s`` is the caller's declared pre-flight window — the
        suite's own startup budget, not a new number and not the in-flight
        ``response_deadline_s``, which governs tactical calls and is not
        moved by this module. The planner polls at the deadline, so the
        caller's clock decides and no wall read hides in here.
        """
        question = explicit_question or plan_question(self.builder.instruction)
        attempt = 0
        # Every transport failure this call produces is new; the count is what
        # says which ones are ours.
        failures_before = len(self.provider.failures)
        last_failure: str | None = None
        while attempt <= self.retry_budget:
            request_id = f"preflight-{sequence}-{attempt}"
            request = DecisionRequest(
                request_id=request_id,
                sequence=sequence,
                mission_revision=mission_revision,
                base_goal_revision=base_goal_revision,
                observation_ids=(observation.record_id,),
                snapshot_id=None,
                # The record's own expiry field carries this call's window.
                response_deadline_s=deadline_s,
                model_identity=self.provider.config.identity,
            )
            packet, trace = self.builder.build(
                request=request,
                observation=observation,
                payloads=payloads,
                call_class=CALL_INITIAL,
                navigation_status="on ground, pre-flight",
                explicit_question=question,
            )
            self.provider.submit(request, packet, now)
            arrivals = self.provider.poll(
                ClockStamp(
                    now.host_id, now.clock_id, now.monotonic_ns + int(deadline_s * 1_000_000_000)
                )
            )
            attempt += 1
            if arrivals:
                reply = arrivals[-1]
                if reply.parsed.malformed_reason:
                    return self._refused(
                        request_id, trace, attempt, reply.parsed.malformed_reason, reply
                    )
                try:
                    recipe = plan_from_reply(reply.parsed)
                except PlanRefused as error:
                    return self._refused(request_id, trace, attempt, str(error), reply)
                return FlightPlan(
                    usable=True,
                    recipe=recipe,
                    refusal_reason=None,
                    request_id=request_id,
                    model_identity=self.provider.config.identity,
                    trace=trace.document(),
                    attempts=attempt,
                    round_trip_s=reply.arrival.round_trip_ns / 1_000_000_000,
                )
            last_failure = "no reply arrived inside the pre-flight window"
        # The transport's own diagnosis, if there was one. ``Provider.poll``
        # appends a transport failure to ``provider.failures`` and returns it to
        # nobody, so an HTTP status and its response body — the actual reason a
        # live pre-flight call failed — surfaced here as "no reply arrived
        # inside the pre-flight window": true, and useless. The message is
        # carried verbatim; the transport bounds its own read at 1000 chars and
        # it never contains the API key.
        #
        # Read by count, not by request id: the planner sends every retry on the
        # caller's own stamp and the provider attributes an arrival to a record
        # by that stamp, so two attempts share one id — measured, not assumed
        # (``provider.failures`` read ``[('preflight-0-0', ...), ('preflight-0-0',
        # ...)]`` for a two-attempt call). A count is immune to that.
        new_failures = self.provider.failures[failures_before:]
        if new_failures:
            last_failure = f"{last_failure}: {new_failures[-1][1]}"
        return FlightPlan(
            usable=False,
            recipe=None,
            refusal_reason=last_failure,
            request_id=f"preflight-{sequence}",
            model_identity=self.provider.config.identity,
            trace=None,
            attempts=attempt,
            round_trip_s=None,
        )

    def _refused(self, request_id, trace, attempts, reason, reply) -> FlightPlan:
        return FlightPlan(
            usable=False,
            recipe=None,
            refusal_reason=reason,
            request_id=request_id,
            model_identity=self.provider.config.identity,
            trace=trace.document(),
            attempts=attempts,
            round_trip_s=reply.arrival.round_trip_ns / 1_000_000_000,
        )
