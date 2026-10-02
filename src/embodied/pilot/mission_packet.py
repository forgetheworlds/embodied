"""The mission packet builder: the one place a cloud request's evidence is chosen.

The owner's ruling of 2026-10-01 is the reason this module exists: the initial
high-level goal is sent **before the drone lifts off**, with any information it
can gather while stationary, and after that reasoning is off. Measured on the
pinned route, the initial class reasons for p50 20.2 s and 3 of 12 calls
exceed a 30 s read timeout — intolerable while the aircraft is moving, free on
the ground. Every call made while moving is therefore the continuous class
(p50 2.91 s, 12/12 usable).

Two responsibilities, deliberately in one object:

* it encodes the frames a call delivers at the scale its declared call class
  asks for, so what was measured is what a mission sends; and
* it owns the **airborne latch**. Once the aircraft is marked airborne, an
  initial-class packet cannot be built at all. That is the prohibition the
  ruling needs as structure rather than as convention: no in-flight caller
  can issue a reasoned call, because every caller shares this builder and the
  builder refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied.contracts.records import DecisionRequest
from embodied.pilot.provider import (
    CALL_CLASSES,
    CALL_CONTINUOUS,
    CALL_INITIAL,
    ModelConfig,
    RequestPacket,
    encode_image_data_uri,
)


class PacketRefused(Exception):
    """A packet cannot be built. Recorded, never silently substituted."""


class ReasonedInFlightRefused(PacketRefused):
    """A reasoned call was attempted after the aircraft was marked airborne.

    This is the structural form of the owner's ruling: it is raised by the
    builder, not checked by a caller, so removing it means editing this class
    rather than forgetting a condition somewhere in the mission.
    """


@dataclass(frozen=True)
class PacketTrace:
    """What one packet actually carried, recorded rather than asserted.

    Every field is a fact of the request that left the process: which
    observation the frames came from, which frames, which declared call class
    and scale, and which model identity answered. A receipt can cite this
    instead of trusting a log line.
    """

    request_id: str
    observation_id: str
    frames: tuple[str, ...]
    call_class: str
    image_scale: float
    model_identity: str

    def document(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "observation_id": self.observation_id,
            "frames": list(self.frames),
            "call_class": self.call_class,
            "image_scale": self.image_scale,
            "model_identity": self.model_identity,
        }


class MissionPacketBuilder:
    """Builds the evidence packet for one cloud call from one observation.

    Stateless except for the airborne latch, which only ever closes. The
    frames delivered are the observation's own payloads, encoded PNG (the
    pinned route rejects ``image/ppm`` with HTTP 400 regardless of size),
    scaled by the declared profile of the call's class.
    """

    def __init__(self, config: ModelConfig, instruction: str) -> None:
        self.config = config
        self.instruction = instruction
        self._airborne = False
        self._airborne_reason: str | None = None

    # -- the airborne latch -----------------------------------------------

    def mark_airborne(self, reason: str = "liftoff") -> None:
        """Close the latch. There is no re-opening it inside one mission.

        The runtime calls this once, at liftoff. After it, the initial class
        is unbuildable for the rest of the mission's life.
        """
        self._airborne = True
        self._airborne_reason = self._airborne_reason or reason

    @property
    def airborne(self) -> bool:
        return self._airborne

    # -- packet construction ----------------------------------------------

    def build(
        self,
        *,
        request: DecisionRequest,
        observation,
        payloads: dict[str, bytes],
        call_class: str = CALL_CONTINUOUS,
        target_refs: tuple[str, ...] = (),
        navigation_status: str = "on ground, pre-flight",
        uncertainty_summary: str = "none recorded",
        explicit_question: str | None = None,
    ) -> tuple[RequestPacket, PacketTrace]:
        if call_class not in CALL_CLASSES:
            raise PacketRefused(
                f"call_class {call_class!r} is not one of {', '.join(CALL_CLASSES)}; "
                "a packet belongs to a declared class or it does not exist"
            )
        if call_class == CALL_INITIAL and self._airborne:
            raise ReasonedInFlightRefused(
                "the aircraft is airborne "
                f"(marked at {self._airborne_reason!r}), so the reasoned initial class "
                "cannot be sent: the owner's ruling is one reasoned call on the ground "
                "and continuous calls only in flight"
            )
        scale = self.config.image_scale_for(call_class)
        frames = tuple(sorted(payloads))
        image_parts = tuple(
            (observation.record_id, encode_image_data_uri(payloads[name], scale=scale))
            for name in frames
        )
        packet = RequestPacket(
            request=request,
            mission_instruction=self.instruction,
            image_parts=image_parts,
            target_refs=target_refs,
            navigation_status=navigation_status,
            uncertainty_summary=uncertainty_summary,
            explicit_question=explicit_question,
            call_class=call_class,
        )
        trace = PacketTrace(
            request_id=request.request_id,
            observation_id=observation.record_id,
            frames=frames,
            call_class=call_class,
            image_scale=scale,
            model_identity=self.config.identity,
        )
        return packet, trace
