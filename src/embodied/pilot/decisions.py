"""Evidence selection: what merits a cloud request, and when.

The broker owns request *lifecycle*; this module owns the judgement of *what*
and *when*. Every function is pure over explicit timestamps — nothing here
reads a wall clock.

Freshness is a first-class input, not an afterthought: an observation is
cited as current evidence only while its age (now − capture) is within the
declared ``observation_freshness_s`` bound. An observation that arrives or is
considered after that bound is still recorded — its measured age travels in
every decision outcome and in the recorded reasons built from them — but it
never grounds a fresh spatial goal, and it is never silently treated as
current.

The numeric parameters (deadband, silence interval, latency budget, guard,
freshness bound, deadline) are declared engineering values carried in
``configs/runtime-model.yaml`` until development-data workload tests select
them (specification sections 9.1, 16.3). They are assumptions, not
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied.contracts.records import ClockStamp, DecisionRequest, Observation, elapsed_ns


# ---------------------------------------------------------------------------
# Declared parameters (configs/runtime-model.yaml, section ``pilot``)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotParameters:
    """The declared decision parameters. Each is an assumption pending
    development-data selection; the config file names the justification."""

    event_deadband: float
    max_silence_s: float
    latency_budget_s: float
    guard_margin_s: float
    response_deadline_s: float
    observation_freshness_s: float
    retry_budget: int

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "PilotParameters":
        required = (
            "event_deadband",
            "max_silence_s",
            "latency_budget_s",
            "guard_margin_s",
            "response_deadline_s",
            "observation_freshness_s",
            "retry_budget",
        )
        values = {}
        for key in required:
            value = section.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"pilot.{key} must be a non-negative number")
            if key == "response_deadline_s" and value <= 0:
                raise ValueError("pilot.response_deadline_s must be positive")
            values[key] = float(value) if key != "retry_budget" else int(value)
        return cls(**values)


# ---------------------------------------------------------------------------
# Evidence state the decisions read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceAge:
    """One observation's age at a decision, measured rather than assumed."""

    observation_id: str
    age_s: float
    fresh: bool

    def __str__(self) -> str:
        return f"{self.observation_id}:age_s={self.age_s:.3f}"


@dataclass(frozen=True)
class SceneStatus:
    """The navigation/execution input one decision considers (section 9.1 packet)."""

    signature: float | None
    new_targets: tuple[str, ...] = ()
    lost_targets: tuple[str, ...] = ()
    identity_ambiguity: bool = False
    map_conflict: bool = False
    horizon_s: float | None = None
    decision_boundary_s: float | None = None
    explicit_question: str | None = None


@dataclass(frozen=True)
class Decision:
    """Either a built request with its packet, or a hold with its reason.

    ``evidence_ages`` records the age every considered observation had when
    this decision was made, fresh or stale: the decision carries the ages it
    was made from.
    """

    send: bool
    reason: str
    request: DecisionRequest | None = None
    evidence_ages: tuple[EvidenceAge, ...] = ()
    cited_observation_ids: tuple[str, ...] = ()
    stale_observation_ids: tuple[str, ...] = ()
    supersede_outstanding: bool = False

    @property
    def stale_reason(self) -> str:
        stale = [str(age) for age in self.evidence_ages if not age.fresh]
        return "; ".join(stale)



# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------

# Boundary-approaching sends are urgent (§17.3): the local system must not
# cross a decision boundary merely because a reply is expected soon.
_URGENT = ("map_conflict", "lost_target", "explicit_question", "decision_boundary_approaching")


@dataclass
class DecisionEngine:
    """The trigger rules. One instance per episode; it remembers what it sent
    so deduplication and the silence interval are real, not per-call fictions."""

    parameters: PilotParameters
    model_identity: str | None
    parameters: PilotParameters
    model_identity: str | None
    _last_signature: float | None = None
    _last_send_s: float | None = None
    _last_subject: str | None = None
    _last_horizon: float | None = None
    _last_trigger: str | None = None

    def consider(
        self,
        now: ClockStamp,
        scene: SceneStatus,
        observations: dict[str, Observation],
        mission_revision: int,
        base_goal_revision: int,
        outstanding_request: DecisionRequest | None,
    ) -> Decision:
        ages = _ages(observations, now, self.parameters.observation_freshness_s)
        fresh_ids = tuple(age.observation_id for age in ages if age.fresh)
        trigger, supersede = self._trigger(now, scene, outstanding_request)
        if trigger is None:
            return Decision(
                send=False,
                reason="no_trigger",
                evidence_ages=ages,
                cited_observation_ids=fresh_ids,
                stale_observation_ids=tuple(age.observation_id for age in ages if not age.fresh),
            )
        if outstanding_request is not None and not supersede:
            return Decision(
                send=False,
                reason=f"request_outstanding:{outstanding_request.request_id}",
                evidence_ages=ages,
                cited_observation_ids=fresh_ids,
            )
        if not fresh_ids and trigger not in _URGENT and trigger != "max_silence":
            return Decision(
                send=False,
                reason="no_fresh_evidence",
                evidence_ages=ages,
            )
        now_s = now.monotonic_ns / 1_000_000_000
        if (
            self._last_send_s is not None
            and trigger == self._last_trigger
            and trigger not in _URGENT
        ):
            # Deduplicated by subject and evidence requirement (§9.2): the same
            # question about the same subject inside the freshness window does
            # not become a second request.
            if self._dedup_subject(scene) == self._last_subject and (
                now_s - self._last_send_s
            ) < self.parameters.observation_freshness_s:
                return Decision(
                    send=False,
                    reason="deduplicated",
                    evidence_ages=ages,
                    cited_observation_ids=fresh_ids,
                )
        sequence = 0 if outstanding_request is None else outstanding_request.sequence + 1
        request = DecisionRequest(
            request_id=f"req-{sequence}-{int(now_s)}",
            sequence=sequence,
            mission_revision=mission_revision,
            base_goal_revision=base_goal_revision,
            observation_ids=fresh_ids,
            snapshot_id=None,
            response_deadline_s=self.parameters.response_deadline_s,
            model_identity=self.model_identity,
        )
        self._last_signature = scene.signature
        self._last_send_s = now_s
        self._last_subject = self._dedup_subject(scene)
        self._last_trigger = trigger
        return Decision(
            send=True,
            reason=trigger,
            request=request,
            evidence_ages=ages,
            cited_observation_ids=fresh_ids,
            stale_observation_ids=tuple(age.observation_id for age in ages if not age.fresh),
            supersede_outstanding=supersede and outstanding_request is not None,
        )

    def _trigger(
        self, now: ClockStamp, scene: SceneStatus, outstanding: DecisionRequest | None
    ) -> tuple[str | None, bool]:
        if scene.explicit_question:
            return "explicit_question", True
        if scene.map_conflict:
            return "map_conflict", True
        if scene.lost_targets:
            return "lost_target", True
        if scene.new_targets:
            return "new_target", False
        if scene.identity_ambiguity:
            return "identity_ambiguity", False
        if (
            scene.decision_boundary_s is not None
            and scene.decision_boundary_s
            <= self.parameters.latency_budget_s + self.parameters.guard_margin_s
        ):
            # Urgent: the boundary must not be crossed merely because a reply
            # is expected, so this supersedes an outstanding request (§17.3).
            return "decision_boundary_approaching", True
        if scene.horizon_s is not None and (
            self._last_horizon is not None and scene.horizon_s < self._last_horizon * 0.5
        ):
            self._last_horizon = scene.horizon_s
            return "reduced_horizon", False
        self._last_horizon = scene.horizon_s
        if (
            self._last_signature is not None
            and scene.signature is not None
            and abs(scene.signature - self._last_signature) > self.parameters.event_deadband
        ):
            return "signature_change", False
        now_s = now.monotonic_ns / 1_000_000_000
        if self._last_send_s is not None and (now_s - self._last_send_s) >= self.parameters.max_silence_s:
            return "max_silence", False
        if self._last_send_s is None and scene.signature is not None:
            # Nothing has been sent yet: the pilot's first broad look goes out.
            return "initial_scene", False
        return None, False

    @staticmethod
    def _dedup_subject(scene: SceneStatus) -> str:
        return ",".join((*scene.new_targets, *scene.lost_targets)) or "scene"



def _ages(
    observations: dict[str, Observation], now: ClockStamp, freshness_s: float
) -> tuple[EvidenceAge, ...]:
    ages = []
    for observation_id, observation in observations.items():
        age_s = elapsed_ns(observation.capture_stamp, now) / 1_000_000_000
        ages.append(EvidenceAge(observation_id=observation_id, age_s=age_s, fresh=age_s <= freshness_s))
    return tuple(ages)
