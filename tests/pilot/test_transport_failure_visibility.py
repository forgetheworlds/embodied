"""A cloud call that never arrived must say so, and say why.

Measured cause of this file: a live B1 pre-flight call recorded

    pre-flight plan: usable=False ... refusal='no reply arrived inside the
    pre-flight window'

while the transport was holding the real diagnosis — an HTTP status and its
response body — in ``provider.failures``. ``Provider.poll`` appends a transport
failure there and returns it to **nobody**, so two seams that report failures
both reported a generic sentence instead of the reason.

That is the defect this project treats as first-class: the record did not merely
fail to explain the failure, it replaced the explanation with something true and
useless. A record whose job is to say what happened must not hide why it did not.

Both seams are held here: the pre-flight planner's refusal reason, and the
broker's in-flight outcome stream (the runtime turns those into the run's
``cloud_calls``).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embodied.pilot.broker import PilotBroker
from embodied.pilot.provider import (
    ModelConfig,
    Provider,
    RequestPacket,
    ScriptedReply,
    ScriptedTransport,
    TransportError,
)
from embodied.pilot.tools import TOOL_SCHEMAS

from test_live_observation_updates import (
    CLOCK,
    HOST,
    SeamFixture,
    S,
    make_observation,
    make_request,
    mission_contract,
    parameters,
)

FAILURE_TEXT = "HTTP 500: upstream exploded"


def _provider(transport: ScriptedTransport) -> Provider:
    return Provider(
        ModelConfig(
            provider="commandcode",
            id="deepseek/deepseek-v4.1-flash",
            base_url="https://api.commandcode.ai/provider/v1",
            api="openai-completions",
            image_transport="base64",
        ),
        transport,
        TOOL_SCHEMAS,
    )


def _broker(transport: ScriptedTransport) -> PilotBroker:
    return PilotBroker(
        SeamFixture(), parameters(), _provider(transport), host_id=HOST, clock_id=CLOCK
    )


def _submit_one(broker: PilotBroker) -> None:
    observation = make_observation("obs-1", 1, 2_000_000_000, 2_050_000_000)
    broker.on_observation(observation, S(1.05))
    request = make_request("req-1", 0, 1, ("obs-1",))
    broker.submit(request, RequestPacket(request=request, mission_instruction="m"), S(1.1))


def test_an_in_flight_transport_failure_reaches_the_record():
    """The reason travels, not just the fact that a call was made."""
    transport = ScriptedTransport()
    transport.schedule(ScriptedReply(error=TransportError(FAILURE_TEXT)))
    broker = _broker(transport)
    broker.set_mission(mission_contract(), S(1.0))
    _submit_one(broker)

    outcomes = broker.poll(S(1.2))
    failures = [outcome for outcome in outcomes if outcome.kind == "transport_failure"]
    assert failures, f"no failure surfaced; kinds were {[o.kind for o in outcomes]}"
    assert FAILURE_TEXT in failures[0].reason
    assert failures[0].request_id == "req-1"


def test_a_surfaced_failure_is_not_repeated_on_every_poll():
    """Surfaced once: the record gains a line, not a per-poll drumbeat."""
    transport = ScriptedTransport()
    transport.schedule(ScriptedReply(error=TransportError(FAILURE_TEXT)))
    broker = _broker(transport)
    broker.set_mission(mission_contract(), S(1.0))
    _submit_one(broker)

    first = [o for o in broker.poll(S(1.2)) if o.kind == "transport_failure"]
    second = [o for o in broker.poll(S(1.4)) if o.kind == "transport_failure"]
    assert len(first) == 1
    assert second == []


def test_a_failure_is_surfaced_without_changing_the_request_lifecycle():
    """The state machine is the broker's declared policy, not this line's.

    Surfacing must not silently resolve, cancel or supersede the outstanding
    request: it still expires on its own deadline, which is what the broker
    already declared.
    """
    transport = ScriptedTransport()
    transport.schedule(ScriptedReply(error=TransportError(FAILURE_TEXT)))
    broker = _broker(transport)
    broker.set_mission(mission_contract(), S(1.0))
    _submit_one(broker)

    outstanding_before = broker.outstanding_record()
    before_id = None if outstanding_before is None else outstanding_before.request.request_id
    broker.poll(S(1.2))
    outstanding_after = broker.outstanding_record()
    after_id = None if outstanding_after is None else outstanding_after.request.request_id
    assert after_id == before_id
