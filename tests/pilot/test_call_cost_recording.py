"""A cloud call's record must say what it cost and how long it took.

Measured cause of this file: a live pre-flight call recorded
``round_trip_s`` 90.0 — exactly the declared poll window — for a call class
the probe measured at 23.85 s (J33-b1-2), because the arrival was stamped at
the caller's poll clock instead of the moment the blocking send returned, and
the provider-reported token usage was dropped at the transport.

Three seams are held here: the live transport stamps its own completion; the
broker outcome and the flight plan carry usage and the measured duration; and
the scripted transport's own stamped arrivals (the harness the other tests
and the recorded episodes run on) are unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embodied.contracts.records import ClockStamp
from embodied.pilot.broker import PilotBroker
from embodied.pilot.provider import (
    LiveTransport,
    ModelConfig,
    RequestPacket,
    Provider,
    ScriptedReply,
    ScriptedTransport,
)
from embodied.pilot.tools import TOOL_SCHEMAS

from test_flight_plan import (
    GOOD_RECIPE,
    reply_with_recipe,
    make_config,
    make_observation,
    make_png,
)
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


USAGE = {"prompt_tokens": 511, "completion_tokens": 96, "total_tokens": 607}


def _stamp(ns: int) -> ClockStamp:
    return ClockStamp(HOST, CLOCK, ns)


def test_live_transport_stamps_completion_not_the_poll_clock():
    """round_trip_ns is the send-to-reply duration, never the poll gap."""

    def slow_opener(_request: urllib.request.Request):
        time.sleep(0.05)  # the only real wait in the suite, 50 ms, once
        return {"choices": [{"message": {"content": "ok"}}]}

    transport = LiveTransport(make_config(), api_key_env="UNUSED", opener=slow_opener)
    sent = _stamp(time.monotonic_ns())
    transport.send({}, sent)
    time.sleep(0.25)  # poll long after the call already completed
    arrivals = transport.poll(_stamp(time.monotonic_ns()))

    assert len(arrivals) == 1
    arrival = arrivals[0]
    assert arrival.error is None
    round_trip_s = arrival.round_trip_ns / 1_000_000_000
    # At least the call itself; strictly less than the send-to-poll gap,
    # which is what the old code reported (the 90.0 s defect, small scale).
    assert round_trip_s >= 0.05
    assert round_trip_s < 0.25


def test_live_transport_error_arrivals_carry_the_completion_stamp_too():
    """A failed call is measured like a successful one."""

    import io
    import urllib.error

    def refusing_opener(request: urllib.request.Request):
        raise urllib.error.HTTPError(
            request.full_url, 503, "upstream", {}, io.BytesIO(b"")
        )

    transport = LiveTransport(
        make_config(), api_key_env="UNUSED", opener=refusing_opener
    )
    sent = _stamp(time.monotonic_ns())
    transport.send({}, sent)
    time.sleep(0.2)
    (arrival,) = transport.poll(_stamp(time.monotonic_ns()))
    assert arrival.error is not None
    assert "HTTP 503" in str(arrival.error)
    # Measured duration of the failed attempt, not the poll gap: strictly
    # below the 0.2 s the harness waited before polling.
    assert arrival.round_trip_ns / 1_000_000_000 < 0.2


def _provider(transport) -> Provider:
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


def _broker(transport) -> PilotBroker:
    return PilotBroker(
        SeamFixture(), parameters(), _provider(transport), host_id=HOST, clock_id=CLOCK
    )


def test_a_reply_outcome_carries_usage_and_the_measured_round_trip():
    """The broker's reply outcome states what the call consumed and took."""
    transport = ScriptedTransport()
    transport.schedule(
        ScriptedReply(
            delay_s=0.5,
            document={"choices": [{"message": {"content": "{}"}}]},
            usage=USAGE,
        )
    )
    broker = _broker(transport)
    broker.set_mission(mission_contract(), S(1.0))
    broker.on_observation(
        make_observation("obs-1", 1, 2_000_000_000, 2_050_000_000), S(1.05)
    )
    request = make_request("req-1", 0, 1, ("obs-1",))
    broker.submit(
        request,
        RequestPacket(request=request, mission_instruction="m"),
        S(1.1),
    )

    outcomes = broker.poll(S(1.8))
    replies = [o for o in outcomes if o.kind == "reply"]
    assert replies, [o.kind for o in outcomes]
    assert replies[0].usage == USAGE
    assert replies[0].round_trip_s == 0.5


def test_a_flight_plan_carries_usage_into_its_document():
    """plan.document() is the run-facing record; usage must survive it."""
    from embodied.pilot.flight_plan import PreflightPlanner
    from embodied.pilot.mission_packet import MissionPacketBuilder

    config = make_config()
    transport = ScriptedTransport()
    provider = Provider(config, transport, ())
    builder = MissionPacketBuilder(
        config, "Find the red block, inspect it, and return to the start."
    )
    planner = PreflightPlanner(provider, builder, retry_budget=0)
    transport.schedule(
        ScriptedReply(delay_s=0.4, document=reply_with_recipe(GOOD_RECIPE), usage=USAGE)
    )
    plan = planner.plan(
        observation=make_observation("obs-1", 0, 1_000_000, 2_000_000),
        payloads={"left.ppm": make_png()},
        now=_stamp(1_000_000_000),
        deadline_s=90.0,
    )
    assert plan.usable is True
    assert plan.usage == USAGE
    assert plan.round_trip_s == 0.4
    document = plan.document()
    assert document["usage"] == USAGE
    assert document["round_trip_s"] == 0.4
    # No price is invented: the document carries tokens, never currency.
    assert not any("cost" in key or "usd" in key for key in document)


def test_scripted_transport_arrivals_are_unchanged():
    """The deterministic harness still delivers on its own stamped clock."""
    transport = ScriptedTransport()
    transport.schedule(
        ScriptedReply(
            delay_s=1.0,
            document=reply_with_recipe(GOOD_RECIPE),
            usage=USAGE,
        )
    )
    sent = _stamp(1_000_000_000)
    transport.send({}, sent)
    assert transport.poll(_stamp(1_400_000_000)) == ()  # not due yet
    (arrival,) = transport.poll(_stamp(2_000_000_000))
    assert arrival.document is not None
    assert arrival.usage == USAGE
    assert arrival.round_trip_ns == 1_000_000_000
