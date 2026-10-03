"""The asynchronous cloud-pilot broker: one event-driven request lifecycle.

The broker is a state machine over explicit timestamps — ``on_observation``,
``on_reply``, ``tick(now)`` — never a thread farm and never a wall clock. Its
invariants, each traceable to the specification:

* One effective cloud reasoning request is outstanding (§15.3). New frames do
  not enter a generating response; an urgent event marks the outstanding
  request obsolete and its late output is discarded locally even if the
  remote computation finishes.
* Idempotency (§15.2): a proposal fingerprint table. An identical repeat
  returns the original admission and current disposition with no second seam
  call; a different payload under the same proposal id is a conflict.
* Revision checks (§15.2): a reply whose expected base-goal revision no
  longer matches the active goal is stale and rejected — an older reply
  cannot overwrite a newer accepted goal. Supersession atomically invalidates
  the superseded goal's queued samples through the seam.
* Freshness (§15.2, §9.2): a fresh spatial goal is never grounded on an
  observation older than the declared freshness bound. The measured age is
  recorded in the rejection reason, not silently dropped.
* Leases (§15.1): goals retire on lease end or supersession; a blocked goal
  stays pending within its original bounds and never extends them.
* Expiry (§17.3): a request past its local deadline is marked expired and its
  remaining delay is never set to zero; the active goal continues within its
  lease and the continuation is recorded with its cause.

No setpoint exists anywhere in this module. The broker emits proposals and
records; motion authority lives behind the admission seam, and no cloud goal
bypasses local admission (§22).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from embodied.contracts.records import (
    ClockStamp,
    DecisionRequest,
    ExecutionDisposition,
    GoalDisposition,
    GoalStatus,
    MissionContract,
    Observation,
    ExecutionStatus,
    SpatialGoal,
    elapsed_ns,
    to_dict,
)

from embodied.pilot.decisions import EvidenceAge, PilotParameters
from embodied.pilot.provider import ProviderReply


# ---------------------------------------------------------------------------
# Event recording (P02's vocabulary only; P04 adds no kind)
# ---------------------------------------------------------------------------


class EventSink(Protocol):
    def __call__(
        self, kind: str, payload: dict[str, Any], stamp: ClockStamp, sim_time_s: float | None
    ) -> None: ...


class EventLog:
    """The default in-memory sink. The episode builder flushes it into the
    real Recorder so a fixture's provenance is the broker's own output."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, Any], ClockStamp, float | None]] = []

    def __call__(
        self,
        kind: str,
        payload: dict[str, Any],
        stamp: ClockStamp,
        sim_time_s: float | None = None,
    ) -> None:
        self.entries.append((kind, payload, stamp, sim_time_s))


# ---------------------------------------------------------------------------
# The P03 admission seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionContext:
    """What the seam may know when it judges one proposal."""

    now: ClockStamp
    mission_revision: int
    active_goal_id: str | None
    active_goal_revision: int


class AdmissionSeam(Protocol):
    """Final goal admission. P03's executor is the real implementation; tests
    inject a fixture double with the frozen refusal tokens. The broker never
    duplicates admission's judgement — it only gates what reaches the seam."""

    def admit(self, proposal: SpatialGoal, context: AdmissionContext) -> GoalStatus: ...

    def cancel(
        self, goal_id: str, expected_revision: int, idempotency_key: str
    ) -> GoalStatus: ...

    def status(self, goal_id: str) -> ExecutionStatus: ...

    def invalidate_queued(self, goal_id: str) -> int: ...


# ---------------------------------------------------------------------------
# Broker state
# ---------------------------------------------------------------------------


class RequestState:
    OUTSTANDING = "outstanding"
    OBSOLETE = "obsolete"
    EXPIRED = "expired"
    ANSWERED = "answered"


@dataclass
class RequestRecord:
    request: DecisionRequest
    send_stamp: ClockStamp
    state: str = RequestState.OUTSTANDING
    delay_not_zeroed_s: float | None = None


@dataclass
class GoalRecord:
    goal_id: str
    revision: int
    proposal: SpatialGoal
    request_id: str
    admitted: ClockStamp
    lease_s: float
    execution: ExecutionDisposition = ExecutionDisposition.NOT_STARTED


@dataclass(frozen=True)
class BrokerOutcome:
    """What one reply or tick produced, for the runner and the tests."""

    kind: str
    reason: str
    request_id: str | None = None
    proposal_id: str | None = None
    status: GoalStatus | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    report: Any = None
    recipe: dict[str, Any] | None = None
    parsed: Any = None
    malformed_reason: str | None = None


def _status(
    proposal: SpatialGoal | None,
    request_id: str,
    disposition: GoalDisposition,
    reason: str | None,
    admission_ref: str | None,
    current: ExecutionDisposition,
) -> GoalStatus:
    return GoalStatus(
        proposal_id=proposal.proposal_id if proposal is not None else request_id,
        request_id=request_id,
        disposition=disposition,
        reason=reason,
        admission_ref=admission_ref,
        current_disposition=current,
    )


# ---------------------------------------------------------------------------
# The broker
# ---------------------------------------------------------------------------


class PilotBroker:
    def __init__(
        self,
        seam: AdmissionSeam,
        parameters: PilotParameters,
        provider=None,
        *,
        host_id: str = "pilot-0",
        clock_id: str = "monotonic",
        sink: EventSink | None = None,
    ) -> None:
        self.seam = seam
        self.parameters = parameters
        self.provider = provider
        self.host_id = host_id
        self.clock_id = clock_id
        self.sink = sink if sink is not None else EventLog()
        self.mission: MissionContract | None = None
        self.observations: dict[str, Observation] = {}
        self.selections: dict[str, Any] = {}
        self.requests: dict[str, RequestRecord] = {}
        self.goals: dict[str, GoalRecord] = {}
        self.active_goal_id: str | None = None
        self.active_revision: int = 0
        self.outcomes: list[BrokerOutcome] = []
        self._idempotency: dict[str, tuple[str, GoalStatus]] = {}
        self._cancel_keys: dict[str, GoalStatus] = {}
        self._goal_counter = 0
        self._surfaced_failures = 0

    # -- mission and evidence ---------------------------------------------

    def set_mission(self, contract: MissionContract, now: ClockStamp) -> None:
        self.mission = contract
        self._emit("mission", contract, now)

    def on_observation(self, observation: Observation, now: ClockStamp) -> EvidenceAge:
        """Record evidence and its measured age. The caller records the
        ``observation`` event; the broker records what the evidence *meant*.
        A stale observation is kept and its age is reported — it is never
        silently promoted to current."""
        self.observations[observation.record_id] = observation
        age_s = elapsed_ns(observation.capture_stamp, now) / 1_000_000_000
        return EvidenceAge(
            observation_id=observation.record_id,
            age_s=age_s,
            fresh=age_s <= self.parameters.observation_freshness_s,
        )

    def register_selection(self, selection) -> None:
        self.selections[selection.selection_id] = selection

    # -- view the decisions engine reads -----------------------------------

    @property
    def outstanding_request(self) -> DecisionRequest | None:
        for record in self.requests.values():
            if record.state == RequestState.OUTSTANDING:
                return record.request
        return None

    @property
    def base_goal_revision(self) -> int:
        return self.active_revision

    # -- sending -----------------------------------------------------------

    def submit(
        self,
        request: DecisionRequest,
        packet,
        now: ClockStamp,
        *,
        supersede_outstanding: bool = False,
    ) -> None:
        if self.provider is None:
            raise RuntimeError("no provider is configured; a local-only arm submits no requests")
        outstanding = self.outstanding_record()
        if outstanding is not None and outstanding.request.request_id != request.request_id:
            if supersede_outstanding:
                outstanding.state = RequestState.OBSOLETE
            # Otherwise the new request waits: new frames never enter a
            # generating response, and an unbounded set of overlapping
            # replacement calls is never created.
            else:
                return
        self.requests[request.request_id] = RequestRecord(
            request=request,
            send_stamp=now,
        )
        self._emit("request", request, now)
        self.provider.submit(request, packet, now)

    def outstanding_record(self) -> RequestRecord | None:
        for record in self.requests.values():
            if record.state == RequestState.OUTSTANDING:
                return record
        return None

    def poll(self, now: ClockStamp) -> tuple[BrokerOutcome, ...]:
        if self.provider is None:
            return ()
        outcomes = []
        for reply in self.provider.poll(now):
            outcomes.extend(self.on_reply(reply, now))
        # A transport failure is appended to ``provider.failures`` and returned
        # to nobody, so an in-flight cloud call that never arrived was invisible
        # in the run's own record — the request simply expired and the record
        # could not say why. Surfaced here as an outcome, because a record whose
        # job is to say what happened must not hide the reason it did not. The
        # state machine is deliberately untouched: the outstanding request still
        # expires on its own deadline, which is the broker's declared policy and
        # not this line's to change.
        pending = self.provider.failures[self._surfaced_failures :]
        self._surfaced_failures = len(self.provider.failures)
        outcomes.extend(
            BrokerOutcome(kind="transport_failure", reason=message, request_id=request_id)
            for request_id, message in pending
        )
        return tuple(outcomes)

    # -- replies -----------------------------------------------------------

    def on_reply(self, reply: ProviderReply, now: ClockStamp) -> tuple[BrokerOutcome, ...]:
        parsed = reply.parsed
        record = self.requests.get(parsed.request_id)
        if record is None:
            return (
                BrokerOutcome(
                    kind="discard", reason="unknown_request", request_id=parsed.request_id
                ),
            )
        outcomes: list[BrokerOutcome] = []
        state = record.state
        delay_s = elapsed_ns(record.send_stamp, now) / 1_000_000_000
        if state != RequestState.OUTSTANDING:
            disposition = (
                GoalDisposition.EXPIRED if state == RequestState.EXPIRED
                else GoalDisposition.SUPERSEDED
            )
            for proposal in parsed.proposals:
                status = _status(
                    proposal,
                    record.request.request_id,
                    disposition,
                    f"reordered_reply: request {record.request.request_id} is {state}; "
                    "late proposal discarded",
                    None,
                    ExecutionDisposition.NOT_STARTED,
                )
                outcomes.append(self._record_disposition(proposal, status, now))
            outcomes.append(
                BrokerOutcome(
                    kind="discard",
                    reason=f"reordered_reply: request {record.request.request_id} is {state}",
                    request_id=record.request.request_id,
                )
            )
        elif delay_s > record.request.response_deadline_s:
            # Delayed: discarded, deadline cited, and the measured delay is
            # recorded rather than zeroed (§17.3).
            record.state = RequestState.EXPIRED
            record.delay_not_zeroed_s = delay_s
            for proposal in parsed.proposals:
                status = _status(
                    proposal,
                    record.request.request_id,
                    GoalDisposition.EXPIRED,
                    f"delayed_reply: {delay_s:.3f}s exceeds response_deadline_s="
                    f"{record.request.response_deadline_s}",
                    None,
                    ExecutionDisposition.NOT_STARTED,
                )
                outcomes.append(self._record_disposition(proposal, status, now))
            outcomes.append(
                BrokerOutcome(
                    kind="discard",
                    reason=f"delayed_reply:{delay_s:.3f}s",
                    request_id=record.request.request_id,
                )
            )
        else:
            record.state = RequestState.ANSWERED
            if parsed.malformed_reason is not None:
                outcomes.append(
                    self._record_malformed(record, parsed.malformed_reason, now)
                )
            for proposal in parsed.proposals:
                outcomes.append(
                    self.process_proposal(proposal, now, via_request=record, collect=outcomes)
                )
            if parsed.report is not None:
                outcomes.append(
                    BrokerOutcome(
                        kind="report_proposed",
                        reason="model proposed a final report; assembly and scoring stay separate",
                        report=parsed.report,
                        request_id=record.request.request_id,
                    )
                )
        if parsed.tool_calls:
            outcomes.append(
                BrokerOutcome(
                    kind="tool_calls",
                    reason="reply requested tools",
                    request_id=record.request.request_id,
                    tool_calls=parsed.tool_calls,
                )
            )
        outcomes.append(
            BrokerOutcome(
                kind="reply",
                reason=f"reply processed for {record.request.request_id}",
                request_id=record.request.request_id,
                recipe=parsed.mission_recipe,
                parsed=parsed,
            )
        )
        self.outcomes.extend(outcomes)
        return tuple(outcomes)

    def _record_malformed(
        self, record: RequestRecord, malformed_reason: str, now: ClockStamp
    ) -> BrokerOutcome:
        goal_ref = self.active_goal_id or "none"
        self._emit(
            "execution",
            ExecutionStatus(
                goal_ref=goal_ref,
                certificate_ref=None,
                command_ref=None,
                disposition=self._active_execution() or ExecutionDisposition.RUNNING,
                evidence=(f"request:{record.request.request_id}",),
                reasons=(f"malformed_reply:{malformed_reason}",),
                horizon_s=None,
                capabilities=(),
            ),
            now,
        )
        return BrokerOutcome(
            kind="malformed_reply",
            reason=malformed_reason,
            request_id=record.request.request_id,
            malformed_reason=malformed_reason,
        )
    def _record_disposition(
        self, proposal: SpatialGoal, status: GoalStatus, now: ClockStamp
    ) -> BrokerOutcome:
        """Record a proposal with a disposition decided outside admission."""
        self._emit("goal", proposal, now)
        self._emit("goal_status", status, now)
        return BrokerOutcome(
            kind=status.disposition.value,
            reason=status.reason or status.disposition.value,
            proposal_id=proposal.proposal_id,
            request_id=status.request_id,
            status=status,
        )

    # -- proposals: dedup, revision, freshness, admission -------------------

    def process_proposal(
        self,
        proposal: SpatialGoal,
        now: ClockStamp,
        *,
        via_request: RequestRecord | None = None,
        local: bool = False,
        collect: list[BrokerOutcome] | None = None,
    ) -> BrokerOutcome:
        request_id = via_request.request.request_id if via_request is not None else (
            proposal.request_id or f"local-{proposal.proposal_id}"
        )
        seen = self._idempotency.get(proposal.proposal_id)
        if seen is not None:
            fingerprint, original = seen
            if fingerprint == proposal.fingerprint:
                current = self.seam.status(original.admission_ref or proposal.proposal_id)
                status = _status(
                    proposal,
                    request_id,
                    original.disposition,
                    f"duplicate: idempotent repeat of {proposal.proposal_id}",
                    original.admission_ref,
                    current.disposition if current else original.current_disposition,
                )
                self._emit("goal_status", status, now)
                return BrokerOutcome(
                    kind="duplicate",
                    reason="idempotent repeat; no second admission",
                    proposal_id=proposal.proposal_id,
                    request_id=request_id,
                    status=status,
                )
            status = _status(
                proposal,
                request_id,
                GoalDisposition.REJECTED,
                f"proposal_conflict: proposal_id {proposal.proposal_id} "
                "already holds a different fingerprint",
                None,
                ExecutionDisposition.NOT_STARTED,
            )
            self._emit("goal_status", status, now)
            return BrokerOutcome(
                kind="conflict",
                reason=status.reason or "proposal_conflict",
                proposal_id=proposal.proposal_id,
                request_id=request_id,
                status=status,
            )

        if proposal.base_goal_revision != self.active_revision:
            status = _status(
                proposal,
                request_id,
                GoalDisposition.REJECTED,
                f"stale_revision: proposal expects base_goal_revision="
                f"{proposal.base_goal_revision}, active is {self.active_revision}",
                None,
                ExecutionDisposition.NOT_STARTED,
            )
            self._emit("goal", proposal, now)
            self._emit("goal_status", status, now)
            return BrokerOutcome(
                kind="stale_revision", reason=status.reason or "stale_revision",
                proposal_id=proposal.proposal_id, request_id=request_id, status=status,
            )

        stale = self.stale_citations(proposal, now)
        if stale:
            ages = "; ".join(
                f"{age.observation_id} age_s={age.age_s:.3f}>{self.parameters.observation_freshness_s}"
                for age in stale
            )
            status = _status(
                proposal,
                request_id,
                GoalDisposition.REJECTED,
                f"stale_observation: fresh goal not grounded on stale evidence ({ages})",
                None,
                ExecutionDisposition.NOT_STARTED,
            )
            self._emit("goal", proposal, now)
            self._emit("goal_status", status, now)
            return BrokerOutcome(
                kind="stale_observation", reason=status.reason or "stale_observation",
                proposal_id=proposal.proposal_id, request_id=request_id, status=status,
            )

        context = AdmissionContext(
            now=now,
            mission_revision=proposal.mission_revision,
            active_goal_id=self.active_goal_id,
            active_goal_revision=self.active_revision,
        )
        self._emit("goal", proposal, now)
        admitted = self.seam.admit(proposal, context)
        if admitted.disposition is GoalDisposition.ACCEPTED and admitted.admission_ref:
            previous = self.active_goal_id
            self._goal_counter += 1
            lease_s = min(value for _, value in proposal.lease_bounds)
            goal = GoalRecord(
                goal_id=admitted.admission_ref,
                revision=self.active_revision + 1,
                proposal=proposal,
                request_id=request_id,
                admitted=now,
                lease_s=lease_s,
                execution=admitted.current_disposition,
            )
            self.goals[goal.goal_id] = goal
            self.active_goal_id = goal.goal_id
            self.active_revision = goal.revision
            if previous is not None and previous != goal.goal_id:
                old = self.goals[previous]
                invalidated = self.seam.invalidate_queued(previous)
                old.execution = ExecutionDisposition.CANCELLED
                superseded = _status(
                    old.proposal,
                    old.request_id,
                    GoalDisposition.SUPERSEDED,
                    f"superseded by {goal.goal_id}; {invalidated} queued samples invalidated",
                    goal.goal_id,
                    ExecutionDisposition.CANCELLED,
                )
                self._emit("goal_status", superseded, now)
                collected = BrokerOutcome(
                    kind="superseded",
                    reason=superseded.reason or "superseded",
                    proposal_id=old.proposal.proposal_id,
                    request_id=old.request_id,
                    status=superseded,
                )
                if collect is not None:
                    collect.append(collected)
        self._idempotency[proposal.proposal_id] = (proposal.fingerprint, admitted)
        self._emit("goal_status", admitted, now)
        return BrokerOutcome(
            kind="admitted" if admitted.disposition is GoalDisposition.ACCEPTED else "refused",
            reason=admitted.reason or admitted.disposition.value,
            proposal_id=proposal.proposal_id,
            request_id=request_id,
            status=admitted,
        )

    def set_goal(self, proposal: SpatialGoal, now: ClockStamp) -> BrokerOutcome:
        """The local/tool path into the same admission seam (§10.1, §22)."""
        return self.process_proposal(proposal, now, local=True)

    def stale_citations(self, proposal: SpatialGoal, now: ClockStamp) -> tuple[EvidenceAge, ...]:
        """Every grounding citation older than the freshness bound, measured."""
        stale = []
        for selection_id in proposal.selection_ids:
            selection = self.selections.get(selection_id)
            observation_id = getattr(selection, "observation_id", None)
            observation = self.observations.get(observation_id) if observation_id else None
            if observation is None:
                continue
            age_s = elapsed_ns(observation.capture_stamp, now) / 1_000_000_000
            if age_s > self.parameters.observation_freshness_s:
                stale.append(
                    EvidenceAge(observation_id=observation_id, age_s=age_s, fresh=False)
                )
        for ref in proposal.target_refs:
            observation = self.observations.get(ref)
            if observation is None:
                continue
            age_s = elapsed_ns(observation.capture_stamp, now) / 1_000_000_000
            if age_s > self.parameters.observation_freshness_s:
                stale.append(
                    EvidenceAge(observation_id=ref, age_s=age_s, fresh=False)
                )
        return tuple(stale)

    # -- cancellation -------------------------------------------------------

    def cancel(
        self, goal_id: str, expected_revision: int, idempotency_key: str, now: ClockStamp
    ) -> BrokerOutcome:
        repeated = self._cancel_keys.get(idempotency_key)
        if repeated is not None:
            # The repeat is recorded with its original disposition: idempotency
            # is visible in the timeline, not inferred from a missing event.
            self._emit("goal_status", repeated, now)
            return BrokerOutcome(
                kind="cancel",
                reason="idempotent repeat",
                proposal_id=idempotency_key,
                status=repeated,
            )
        goal = self.goals.get(goal_id)
        if goal is None:
            status = GoalStatus(
                proposal_id=idempotency_key,
                request_id=f"cancel-{idempotency_key}",
                disposition=GoalDisposition.REJECTED,
                reason=f"unknown_goal: {goal_id}",
                admission_ref=None,
                current_disposition=ExecutionDisposition.NOT_STARTED,
            )
        elif expected_revision != goal.revision:
            status = GoalStatus(
                proposal_id=idempotency_key,
                request_id=f"cancel-{idempotency_key}",
                disposition=GoalDisposition.REJECTED,
                reason=f"revision_mismatch: goal {goal_id} is revision {goal.revision}, "
                f"cancel expected {expected_revision}",
                admission_ref=None,
                current_disposition=goal.execution,
            )
        else:
            status = self.seam.cancel(goal_id, expected_revision, idempotency_key)
            if status.disposition is GoalDisposition.CANCELLED:
                goal.execution = ExecutionDisposition.CANCELLED
                if self.active_goal_id == goal_id:
                    self.active_goal_id = None
        self._cancel_keys[idempotency_key] = status
        self._emit("goal_status", status, now)
        return BrokerOutcome(
            kind="cancel", reason=status.reason or status.disposition.value,
            proposal_id=idempotency_key, status=status,
        )

    # -- status -------------------------------------------------------------

    def status(self, goal_id: str) -> ExecutionStatus:
        goal = self.goals.get(goal_id)
        if goal is None:
            return self.seam.status(goal_id)
        local = self.seam.status(goal_id)
        if local is not None:
            return local
        return ExecutionStatus(
            goal_ref=goal.goal_id,
            certificate_ref=None,
            command_ref=None,
            disposition=goal.execution,
            evidence=(f"proposal:{goal.proposal.proposal_id}",),
            reasons=(),
            horizon_s=None,
            capabilities=(),
        )

    def _active_execution(self) -> ExecutionDisposition | None:
        goal = self.goals.get(self.active_goal_id) if self.active_goal_id else None
        return goal.execution if goal else None

    # -- the clock-driven half ----------------------------------------------

    def tick(self, now: ClockStamp) -> tuple[BrokerOutcome, ...]:
        outcomes: list[BrokerOutcome] = []
        record = self.outstanding_record()
        if record is not None:
            delay_s = elapsed_ns(record.send_stamp, now) / 1_000_000_000
            if delay_s > record.request.response_deadline_s:
                record.state = RequestState.EXPIRED
                record.delay_not_zeroed_s = delay_s
                reason = (
                    f"cloud request {record.request.request_id} expired after "
                    f"{delay_s:.3f}s; continuing within lease"
                )
                goal = self.goals.get(self.active_goal_id) if self.active_goal_id else None
                if goal is not None:
                    self._emit(
                        "execution",
                        ExecutionStatus(
                            goal_ref=goal.goal_id,
                            certificate_ref=None,
                            command_ref=None,
                            disposition=goal.execution,
                            evidence=(f"request:{record.request.request_id}",),
                            reasons=(reason,),
                            horizon_s=None,
                            capabilities=(),
                        ),
                        now,
                    )
                outcomes.append(
                    BrokerOutcome(
                        kind="request_expired",
                        reason=reason,
                        request_id=record.request.request_id,
                    )
                )
        goal = self.goals.get(self.active_goal_id) if self.active_goal_id else None
        if goal is not None:
            age_s = elapsed_ns(goal.admitted, now) / 1_000_000_000
            if age_s > goal.lease_s:
                goal.execution = ExecutionDisposition.STOPPED
                self._emit(
                    "execution",
                    ExecutionStatus(
                        goal_ref=goal.goal_id,
                        certificate_ref=None,
                        command_ref=None,
                        disposition=ExecutionDisposition.STOPPED,
                        evidence=(f"proposal:{goal.proposal.proposal_id}",),
                        reasons=(
                            f"lease_expired: {age_s:.3f}s exceeds lease_s={goal.lease_s}; "
                            "bounds are never self-extended",
                        ),
                        horizon_s=None,
                        capabilities=(),
                    ),
                    now,
                )
                self.active_goal_id = None
                outcomes.append(
                    BrokerOutcome(
                        kind="goal_retired",
                        reason="lease_expired",
                        proposal_id=goal.proposal.proposal_id,
                    )
                )
        self.outcomes.extend(outcomes)
        return tuple(outcomes)

    # -- recording ----------------------------------------------------------

    def _emit(self, kind: str, record, now: ClockStamp, sim_time_s: float | None = None) -> None:
        self.sink(kind, to_dict(record), now, sim_time_s)
