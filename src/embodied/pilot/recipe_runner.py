"""The common MissionRecipe runner and the completion-claim assembler.

A MissionRecipe is a bounded composition of reusable actions and conditions
(section 20.2). The semantics live here once, and every arm — B0, B1 and B2 —
runs through this one runner, which is what makes the arms comparable:

* Actions use the intent vocabulary's first-indoor subset (approach, inspect,
  explore, hold, plus return). Guards evaluate only permitted broker/seam/world
  state; they cannot query hidden truth and no step executes generated code.
* A step's target is a selector resolved at execution time through the
  grounding seam and the permitted local view. This is what lets B1 inspect
  candidates discovered after the recipe was issued instead of stopping after
  its initial waypoint.
* Bounds are real: per-step attempt bounds, a recipe step budget and a
  resource ceiling. Exhaustion is a terminal reason, never success (§18.3).
* Completion is a claim with evidence: the assembler never copies a requested
  count into an observed count, and unsupported claims come back partial with
  their unmet requirements named.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from embodied.contracts.records import (
    ClaimKind,
    ClockStamp,
    ExecutionDisposition,
    ExecutionStatus,
    FinalReport,
    Observation,
    ReportClaim,
    SpatialGoal,
    to_dict,
)

from embodied.pilot.broker import PilotBroker
from embodied.pilot.tools import fingerprint_of

INTENTS = ("approach", "inspect", "explore", "hold", "return")
GUARDS = (
    "always",
    "candidate_present",
    "goal_completed",
    "goal_blocked",
    "information_requirement",
    "target_lost",
    "budget_remaining",
)


# ---------------------------------------------------------------------------
# Recipe as data, not code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSelector:
    """Deferred target binding: a name for what the step will act on, resolved
    when the step runs, from the evidence available then."""

    kind: str  # candidate | frontier | place
    ref: str  # e.g. "next_uninspected", "next_unvisited", "start"


@dataclass(frozen=True)
class Guard:
    kind: str
    ref: str | None = None


@dataclass(frozen=True)
class RecipeStep:
    action: str
    target: TargetSelector | None
    guard: Guard = Guard("always")
    max_attempts: int = 1
    completion: str = "reached the declared local relationship"
    step_cost: float = 1.0


@dataclass(frozen=True)
class MissionRecipe:
    steps: tuple[RecipeStep, ...]
    max_steps: int
    resource_ceiling: float
    source: str  # B0-local | B1-one-shot | B2-continuous | ...

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("an empty recipe is not a mission")
        if self.max_steps <= 0 or self.resource_ceiling <= 0:
            raise ValueError("a recipe without bounds is not a recipe")
        for step in self.steps:
            if step.action not in INTENTS:
                raise ValueError(f"intent {step.action!r} is outside the first-indoor subset")
            if step.guard.kind not in GUARDS:
                raise ValueError(f"guard {step.guard.kind!r} is outside the allowed vocabulary")


# ---------------------------------------------------------------------------
# The permitted world the runner acts in (test doubles implement this)
# ---------------------------------------------------------------------------


class RunnerWorld(Protocol):
    """The harness-side world, exposing permitted current-evidence views only."""

    def discovered_candidates(self) -> tuple[str, ...]: ...

    def known_frontiers(self) -> tuple[str, ...]: ...

    def start_place(self) -> str: ...

    def is_blocked(self, target_ref: str) -> bool: ...

    def budget_remaining_s(self) -> float: ...

    def step(self, action: str, target_ref: str | None) -> tuple[Observation, ...]: ...

    def completion(self, action: str, target_ref: str | None) -> tuple[str, tuple[str, ...]]:
        """(achieved | blocked | needs_review, evidence ids)"""
        ...


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    action: str
    target_ref: str | None
    attempts: int
    outcome: str  # achieved | needs_review | blocked | guard_false | unresolved
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunnerResult:
    status: str  # completed | bounds_exhausted | resources_exhausted | budget_exhausted | blocked
    reason: str
    steps: tuple[StepResult, ...] = ()
    admitted_goal_ids: tuple[str, ...] = ()


class RecipeRunner:
    """Owns the recipe semantics for every arm. One instance per episode."""

    def __init__(
        self,
        broker: PilotBroker,
        world: RunnerWorld,
        now: Callable[[], ClockStamp],
        sink,
        on_observation=None,
    ) -> None:
        self.broker = broker
        self.world = world
        self.now = now
        self.sink = sink
        self.on_observation = on_observation
        self._inspected: set[str] = set()
        self._visited: set[str] = set()

    def run(self, recipe: MissionRecipe) -> RunnerResult:
        steps: list[StepResult] = []
        admitted: list[str] = []
        used_steps = 0
        used_resources = 0.0
        for step in recipe.steps:
            if used_steps >= recipe.max_steps:
                return self._terminal(
                    "bounds_exhausted",
                    f"step budget {recipe.max_steps} reached before the recipe finished",
                    steps,
                    admitted,
                )
            if used_resources >= recipe.resource_ceiling:
                return self._terminal(
                    "resources_exhausted",
                    f"resource ceiling {recipe.resource_ceiling} reached before the recipe finished",
                    steps,
                    admitted,
                )
            if self.world.budget_remaining_s() <= 0:
                return self._terminal("budget_exhausted", "mission budget spent", steps, admitted)
            if not self._guard_holds(step):
                steps.append(StepResult(step.action, None, 0, "guard_false"))
                continue
            result, goal_id = self._run_step(step)
            used_steps += 1
            used_resources += step.step_cost
            if goal_id:
                admitted.append(goal_id)
            steps.append(result)
            if result.outcome == "blocked":
                return self._terminal(
                    "blocked",
                    f"step {step.action} stayed blocked through {step.max_attempts} attempts",
                    steps,
                    admitted,
                )
        return RunnerResult(
            status="completed",
            reason="every step reached its declared completion or was skipped by its guard",
            steps=tuple(steps),
            admitted_goal_ids=tuple(admitted),
        )

    def _terminal(self, status: str, reason: str, steps, admitted) -> RunnerResult:
        return RunnerResult(
            status=status, reason=reason, steps=tuple(steps), admitted_goal_ids=tuple(admitted)
        )

    def _guard_holds(self, step: RecipeStep) -> bool:
        guard = step.guard
        if guard.kind == "always":
            return True
        if guard.kind == "candidate_present":
            return bool(self.world.discovered_candidates())
        if guard.kind == "budget_remaining":
            return self.world.budget_remaining_s() > 0
        if guard.kind == "target_lost":
            return guard.ref is not None and guard.ref not in self.world.discovered_candidates()
        if guard.kind == "goal_blocked":
            execution = (
                self.broker.status(self.broker.active_goal_id)
                if self.broker.active_goal_id
                else None
            )
            return execution is None or execution.disposition is ExecutionDisposition.BLOCKED
        if guard.kind == "goal_completed":
            execution = (
                self.broker.status(self.broker.active_goal_id)
                if self.broker.active_goal_id
                else None
            )
            return execution is not None and execution.disposition is ExecutionDisposition.COMPLETED
        if guard.kind == "information_requirement":
            return guard.ref is not None
        return False

    def _resolve_target(self, step: RecipeStep) -> str | None:
        selector = step.target
        if selector is None:
            return None
        if selector.kind == "candidate":
            for candidate in self.world.discovered_candidates():
                if candidate not in self._inspected:
                    return candidate
            return None
        if selector.kind == "frontier":
            for frontier in self.world.known_frontiers():
                if frontier not in self._visited:
                    return frontier
            return None
        if selector.kind == "place":
            return self.world.start_place() if selector.ref == "start" else selector.ref
        return None

    def _run_step(self, step: RecipeStep) -> tuple[StepResult, str | None]:
        for attempt in range(1, step.max_attempts + 1):
            target_ref = self._resolve_target(step)
            if target_ref is None:
                continue
            if (
                step.target is not None
                and step.target.kind != "place"
                and self.world.is_blocked(target_ref)
            ):
                self._emit_execution(
                    target_ref,
                    ExecutionDisposition.BLOCKED,
                    (f"route to {target_ref} blocked; retry {attempt}/{step.max_attempts}",),
                )
                continue
            goal_id = self._admit_step_goal(step, target_ref)
            observations = self.world.step(step.action, target_ref)
            self._absorb(observations)
            outcome, evidence = self.world.completion(step.action, target_ref)
            if outcome == "achieved":
                disposition = ExecutionDisposition.COMPLETED
                reasons: tuple[str, ...] = (step.completion,)
            elif outcome == "needs_review":
                disposition = ExecutionDisposition.COMPLETED
                reasons = (
                    "required observations captured; identity needs_review, not verified",
                )
            else:
                disposition = ExecutionDisposition.BLOCKED
                reasons = (f"{step.action} {target_ref} blocked at attempt {attempt}",)
            self._emit_execution(target_ref, disposition, reasons, evidence)
            if step.action == "inspect" and target_ref:
                self._inspected.add(target_ref)
            if step.action == "explore" and target_ref:
                self._visited.add(target_ref)
            if outcome in ("achieved", "needs_review"):
                return (
                    StepResult(step.action, target_ref, attempt, outcome, evidence),
                    goal_id,
                )
        return (
            StepResult(step.action, self._resolve_target(step), step.max_attempts, "blocked"),
            None,
        )

    def _admit_step_goal(self, step: RecipeStep, target_ref: str | None) -> str | None:
        proposal = SpatialGoal(
            proposal_id=f"{step.action}-{target_ref}-{self.now().monotonic_ns}",
            request_id=None,
            fingerprint=fingerprint_of(
                {"action": step.action, "target": target_ref, "t": self.now().monotonic_ns}
            ),
            mission_revision=self.broker.mission.revision if self.broker.mission else 0,
            base_goal_revision=self.broker.active_revision,
            selection_ids=(),
            target_refs=(target_ref,) if target_ref else (),
            intent=step.action,
            constraints=(),
            completion_condition=step.completion,
            lease_bounds=(("step_lease_s", max(1.0, step.step_cost * 2)),),
            local_discretion_bounds=(),
        )
        outcome = self.broker.set_goal(proposal, self.now())
        if outcome.status is not None and outcome.status.admission_ref:
            return outcome.status.admission_ref
        return None

    def _absorb(self, observations: tuple[Observation, ...]) -> None:
        for observation in observations:
            self.sink("observation", to_dict(observation), observation.receipt_stamp)
            age = self.broker.on_observation(observation, self.now())
            if self.on_observation is not None:
                self.on_observation(observation, age)

    def _emit_execution(
        self,
        target_ref: str | None,
        disposition: ExecutionDisposition,
        reasons: tuple[str, ...],
        evidence: tuple[str, ...] = (),
    ) -> None:
        if disposition is ExecutionDisposition.COMPLETED and not evidence:
            raise ValueError("a completion is a claim with evidence; cite the evidence")
        self.sink(
            "execution",
            to_dict(ExecutionStatus(
                goal_ref=target_ref or self.broker.active_goal_id or "none",
                certificate_ref=None,
                command_ref=None,
                disposition=disposition,
                evidence=evidence,
                reasons=reasons,
                horizon_s=None,
                capabilities=(),
            )),
            self.now(),
        )


# ---------------------------------------------------------------------------
# Per-claim completion (§18.3): requested facts never become observed facts
# ---------------------------------------------------------------------------


def assemble_claims(
    requested_counts: dict[str, int],
    observed_counts: dict[str, int],
    support_refs: dict[str, tuple[str, ...]],
    now: ClockStamp,
) -> tuple[ReportClaim, ...]:
    """Build one claim per target. The observed count comes only from the
    observed record; a requested count shows up, if at all, as an unmet
    requirement — never as the observed value."""
    claims = []
    for target in sorted(set(requested_counts) | set(observed_counts)):
        observed = observed_counts.get(target, 0)
        refs = support_refs.get(target, ())
        unmet: list[str] = []
        requested = requested_counts.get(target)
        if requested is not None and observed < requested:
            unmet.append(f"requested_count={requested}_observed={observed}")
        if not refs:
            unmet.append("no supporting observation")
        claims.append(
            ReportClaim(
                predicate="count_present",
                target=target,
                observed=observed,
                support_refs=refs,
                kind=ClaimKind.OBSERVATION if refs else ClaimKind.INFERENCE,
                stamp=now,
                uncertainty=None,
                unmet_requirements=tuple(unmet) if unmet else None,
            )
        )
    return tuple(claims)


def build_final_report(
    requested_counts: dict[str, int],
    observed_counts: dict[str, int],
    support_refs: dict[str, tuple[str, ...]],
    termination_reason: str,
    mission_revision: int,
    now: ClockStamp,
) -> FinalReport:
    claims = assemble_claims(requested_counts, observed_counts, support_refs, now)
    unmet = sorted(
        {requirement for claim in claims for requirement in (claim.unmet_requirements or ())}
    )
    return FinalReport(
        mission_revision=mission_revision,
        claims=claims,
        termination_reason=termination_reason,
        unmet_requirements=tuple(unmet),
        physical_return_status=None,
        evidence_snapshot_ids=(),
    )
