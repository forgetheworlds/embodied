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

from dataclasses import dataclass
from typing import Any

from embodied.contracts.records import ClockStamp, DecisionRequest
from embodied.pilot.mission_packet import MissionPacketBuilder
from embodied.pilot.provider import (
    CALL_INITIAL,
    Provider,
)
from embodied.pilot.recipe_runner import (
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


def plan_from_reply(parsed, *, source: str = "cloud-initial") -> MissionRecipe:
    """Validate a reply's ``mission_recipe`` into a bounded MissionRecipe.

    Every step is re-checked against the vocabulary the runner enforces, and
    the whole plan against the conventional arm's declared effort, so the
    model can propose but cannot widen its own envelope.
    """
    document = parsed.mission_recipe
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
                    f"steps[{index}].target_kind {kind!r} is not one of "
                    f"{', '.join(_TARGET_KINDS)}"
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
    if max_steps > len(steps):
        raise PlanRefused(
            f"bounds.max_steps {max_steps} exceeds the plan's own {len(steps)} steps; "
            "the model cannot grant itself extra iterations"
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
        attempt = 0
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
                explicit_question=explicit_question,
            )
            self.provider.submit(request, packet, now)
            arrivals = self.provider.poll(
                ClockStamp(now.host_id, now.clock_id, now.monotonic_ns + int(deadline_s * 1_000_000_000))
            )
            attempt += 1
            if arrivals:
                reply = arrivals[-1]
                if reply.parsed.malformed_reason:
                    return self._refused(request_id, trace, attempt, reply.parsed.malformed_reason, reply)
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
