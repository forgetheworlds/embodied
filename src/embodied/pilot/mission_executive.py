"""The production construction of a cloud-calling arm, in one object.

Before this module the B1/B2 executives were constructed only by tests: no
production path built a provider, a broker or a packet, so the cloud arms
could not drive a mission at all. This is the wiring the runtime was missing,
kept out of the runtime's own file by the integration fence.

One object, :class:`MissionPilot`, owns the whole cloud side of one arm:

* construction from the runtime model configuration, with the tool surface,
  the broker and the transport assembled exactly once;
* the pre-flight phase — one reasoned call through the
  :class:`~embodied.pilot.flight_plan.PreflightPlanner`, which must happen
  before liftoff because its class reasons for p50 20.2 s;
* the airborne transition, which closes the packet builder's latch and makes
  a reasoned call unbuildable for the rest of the mission;
* the in-flight tick, which is continuous-class only: B2 submits through the
  shared trigger rules, B1 makes no further call at all, and both poll
  arrivals and drive the broker's lifecycle without ever waiting on
  inference.

B0 is refused here on purpose: it constructs no provider by design, and the
runtime's own conventional policy owns that arm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from embodied.contracts.records import ClockStamp, MissionContract, Observation
from embodied.pilot.broker import PilotBroker
from embodied.pilot.decisions import DecisionEngine, PilotParameters, SceneStatus
from embodied.pilot.flight_plan import RECIPE_TOOL, FlightPlan, PreflightPlanner
from embodied.pilot.mission_packet import MissionPacketBuilder
from embodied.pilot.provider import (
    CALL_CONTINUOUS,
    ModelConfig,
    Provider,
    RequestPacket,
)
from embodied.pilot.tools import TOOL_SCHEMAS

CLOUD_ARMS = ("B1", "B2")


class ArmRefused(Exception):
    """The arm cannot be constructed. Recorded with the reason."""


def load_runtime_config(path: str | Path = "configs/runtime-model.yaml") -> dict[str, Any]:
    """The runtime model configuration, loaded once for the whole mission."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return document


class MissionPilot:
    """The cloud side of one arm: plan on the ground, act continuously in flight."""

    def __init__(
        self,
        *,
        arm: str,
        config: ModelConfig,
        parameters: PilotParameters,
        contract: MissionContract,
        seam,
        transport,
        sink=None,
        host_id: str = "pilot-0",
    ) -> None:
        if arm not in CLOUD_ARMS:
            raise ArmRefused(
                f"arm {arm!r} constructs no cloud pilot: B0 is deterministic and owns "
                "its own policy, and an unknown arm is never guessed"
            )
        self.arm = arm
        self.config = config
        self.parameters = parameters
        self.contract = contract
        self.provider = Provider(config, transport, TOOL_SCHEMAS)
        self.broker = PilotBroker(seam, parameters, self.provider, host_id=host_id, sink=sink)
        self.packet_builder = MissionPacketBuilder(config, contract.instruction)
        # The pre-flight call gets its own provider so the recipe tool is
        # declared for the plan and for nothing else: the in-flight surface
        # stays exactly the five declared tools (specification section 11).
        # Both providers share the transport, and the plan happens before any
        # in-flight call exists, so they never interleave.
        self.planner = PreflightPlanner(
            Provider(config, transport, (RECIPE_TOOL,)),
            self.packet_builder,
            retry_budget=int(parameters.retry_budget),
        )
        self.engine = DecisionEngine(parameters, config.identity)
        self.in_flight_plan: FlightPlan | None = None
        self._sequence = 0
        self._airborne = False

    # -- construction -------------------------------------------------------

    @classmethod
    def for_arm(
        cls,
        *,
        arm: str,
        config_document: dict[str, Any],
        contract: MissionContract,
        seam,
        transport,
        sink=None,
        host_id: str = "pilot-0",
    ) -> "MissionPilot":
        return cls(
            arm=arm,
            config=ModelConfig.from_config(config_document["model"]),
            parameters=PilotParameters.from_config(config_document["pilot"]),
            contract=contract,
            seam=seam,
            transport=transport,
            sink=sink,
            host_id=host_id,
        )

    # -- pre-flight ---------------------------------------------------------

    def set_mission(self, now: ClockStamp) -> None:
        self.broker.set_mission(self.contract, now)

    def plan_preflight(
        self,
        observation: Observation,
        payloads: dict[str, bytes],
        now: ClockStamp,
        *,
        deadline_s: float,
    ) -> FlightPlan:
        """The one reasoned call. Must be called before :meth:`mark_airborne`.

        ``deadline_s`` is the suite's own declared startup window (for
        first-indoor, ``step_timeout_s.startup``), read by the caller from the
        suite it is running. The in-flight ``response_deadline_s`` is not used
        here and is not moved.
        """
        if self._airborne:
            raise ArmRefused(
                "plan_preflight was called after mark_airborne: the reasoned call "
                "belongs on the ground, before liftoff, per the owner's ruling"
            )
        plan = self.planner.plan(
            observation=observation,
            payloads=payloads,
            now=now,
            deadline_s=deadline_s,
            sequence=self._sequence,
            mission_revision=self.contract.revision,
            base_goal_revision=self.broker.base_goal_revision,
            explicit_question=(
                "You are on the ground before takeoff. From this frame, propose the "
                "mission_recipe for: " + self.contract.instruction
            ),
        )
        self._sequence += 1
        self.in_flight_plan = plan
        # The evidence the plan was made from is the broker's evidence too.
        self.broker.on_observation(observation, now)
        return plan

    # -- the airborne transition --------------------------------------------

    def mark_airborne(self, now: ClockStamp, reason: str = "guided takeoff") -> None:
        """Close the latch. Every later cloud call is continuous-class only."""
        self._airborne = True
        self.packet_builder.mark_airborne(reason)

    @property
    def airborne(self) -> bool:
        return self._airborne

    # -- in flight ----------------------------------------------------------

    def tick(
        self,
        scene: SceneStatus,
        observation: Observation,
        payloads: dict[str, bytes],
        now: ClockStamp,
    ) -> tuple:
        """One in-flight step: record evidence, maybe submit, poll, tick.

        Continuous-class by construction: the packet builder this uses has its
        latch closed, so the reasoned class cannot be built here even by a
        future edit that asks for it. B1 makes no further call at all.
        """
        self.broker.on_observation(observation, now)
        if self.arm == "B2":
            decision = self.engine.consider(
                now,
                scene,
                self.broker.observations,
                self.broker.mission.revision if self.broker.mission else 0,
                self.broker.base_goal_revision,
                self.broker.outstanding_request,
            )
            if decision.send and decision.request is not None:
                packet, trace = self.packet_builder.build(
                    request=decision.request,
                    observation=observation,
                    payloads=payloads,
                    call_class=CALL_CONTINUOUS,
                    navigation_status=(
                        f"in flight, horizon_s={scene.horizon_s}"
                        if scene.horizon_s is not None
                        else "in flight"
                    ),
                    target_refs=decision.cited_observation_ids,
                    explicit_question=scene.explicit_question,
                )
                # The trace is the only record of what this call actually
                # carried — the class it was built as and the image scale
                # applied — and it used to be discarded here, so a run could show
                # twenty in-flight requests and not one fact about any of them.
                # It goes to the broker, which names it in the reply outcome the
                # runtime writes into the run's own log. The episode's event
                # vocabulary is closed and exact-key, so a new event kind is
                # refused by the transport — measured, not assumed — and the
                # class cannot ride on the `request` event either.
                self.broker.submit(
                    decision.request,
                    packet,
                    now,
                    supersede_outstanding=decision.supersede_outstanding,
                    trace=trace,
                )
        outcomes = self.broker.poll(now)
        outcomes += self.broker.tick(now)
        return outcomes
