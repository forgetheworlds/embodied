"""The three comparison executives. Only the decision source differs.

* B0 conventional — deterministic frontier/place exploration, candidate
  verification and return from permitted current evidence. No provider object
  is ever constructed; the arm's isolation value is a competent automation
  baseline.
* B1 one-shot cloud — exactly one initial multimodal call produces a validated
  MissionRecipe; the common runner then executes it with local recovery,
  deferred target resolution and declared retries. No further cloud reasoning
  calls exist to make: the budget is one, enforced here.
* B2 continuous cloud — the same initial inputs and runner, plus fresh
  requests driven by the shared trigger rules while moving, one outstanding
  request at a time, with admitted tactical revisions.

All three run through the same RecipeRunner, the same admission seam and the
same shared-path degradation injection, so a measured difference between arms
is the executive treatment and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from embodied.contracts.records import (
    ClockStamp,
    MissionContract,
)

from embodied.pilot.broker import PilotBroker
from embodied.pilot.decisions import DecisionEngine, SceneStatus
from embodied.pilot.provider import Provider
from embodied.pilot.recipe_runner import (
    Guard,
    MissionRecipe,
    RecipeStep,
    TargetSelector,
)


# ---------------------------------------------------------------------------
# B0 conventional
# ---------------------------------------------------------------------------


@dataclass
class B0Conventional:
    """A deterministic executive. It never touches a provider — the field is
    absent by construction, which tests assert rather than trust."""

    def build_recipe(self, world, mission: MissionContract) -> MissionRecipe:
        return MissionRecipe(
            steps=(
                RecipeStep(
                    action="explore",
                    target=TargetSelector("frontier", "next_unvisited"),
                    max_attempts=2,
                ),
                RecipeStep(
                    action="inspect",
                    target=TargetSelector("candidate", "next_uninspected"),
                    guard=Guard("candidate_present"),
                    max_attempts=2,
                ),
                RecipeStep(
                    action="return",
                    target=TargetSelector("place", "start"),
                    max_attempts=1,
                ),
            ),
            max_steps=8,
            resource_ceiling=12.0,
            source="B0-local",
        )


# ---------------------------------------------------------------------------
# B1 one-shot cloud
# ---------------------------------------------------------------------------


class OneShotBudgetSpent(RuntimeError):
    """B1's entire cloud budget is the initial call. A second attempt is a
    recorded refusal, never a silent extra call."""


@dataclass
class B1OneShot:
    provider: Provider

    def __post_init__(self) -> None:
        self.calls_made = 0

    def initial_request(self, broker: PilotBroker, request, packet, now: ClockStamp):
        if self.calls_made >= 1:
            raise OneShotBudgetSpent("B1 makes exactly one cloud reasoning call")
        self.calls_made += 1
        broker.submit(request, packet, now)

    def recipe_from_reply(self, parsed) -> MissionRecipe:
        """Validate the model's recipe proposal into a bounded MissionRecipe.

        The reply carries a plain document; every step is re-checked against
        the allowed vocabulary here, so a model cannot smuggle in an
        unbounded loop, an unknown intent or an executable guard."""
        document = parsed.mission_recipe
        if document is None:
            raise ValueError("the one-shot reply proposed no mission_recipe document")
        steps = []
        for raw in document.get("steps") or ():
            step = RecipeStep(
                action=raw["action"],
                target=TargetSelector(raw["target_kind"], raw.get("target_ref", ""))
                if raw.get("target_kind")
                else None,
                guard=Guard(raw.get("guard_kind", "always"), raw.get("guard_ref")),
                max_attempts=int(raw.get("max_attempts", 1)),
                completion=raw.get("completion", "reached the declared local relationship"),
                step_cost=float(raw.get("step_cost", 1.0)),
            )
            steps.append(step)
        bounds = document.get("bounds") or {}
        return MissionRecipe(
            steps=tuple(steps),
            max_steps=int(bounds.get("max_steps", len(steps))),
            resource_ceiling=float(bounds.get("resource_ceiling", 0.0) or 1.0),
            source="B1-one-shot",
        )


# ---------------------------------------------------------------------------
# B2 continuous cloud
# ---------------------------------------------------------------------------


@dataclass
class B2Continuous:
    """The B1 machinery plus the shared trigger rules while moving.

    ``tick`` is called between runner steps. It asks the decision engine
    whether evidence merits a request, submits at most one new request (the
    broker enforces one outstanding), polls arrivals, and expires deadlines —
    the loop never waits on inference."""

    provider: Provider
    engine: DecisionEngine

    def tick(self, broker: PilotBroker, scene: SceneStatus, packet_builder, now: ClockStamp) -> None:
        decision = self.engine.consider(
            now,
            scene,
            broker.observations,
            broker.mission.revision if broker.mission else 0,
            broker.base_goal_revision,
            broker.outstanding_request,
        )
        if decision.send and decision.request is not None:
            broker.submit(
                decision.request,
                packet_builder(decision),
                now,
                supersede_outstanding=decision.supersede_outstanding,
            )
        broker.poll(now)
        broker.tick(now)
