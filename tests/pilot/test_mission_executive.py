"""The MissionPilot: the production wiring of a cloud-calling arm.

What these tests pin is the shape the owner ruled on and the runtime will be
told to call: one reasoned call before liftoff, continuous calls only after,
B1 making no further call at all, and B0 refused because it owns no cloud.
"""

from __future__ import annotations

import base64
import io
import json

import pytest

from embodied.contracts.records import (
    ClockStamp,
    GoalStatus,
    MissionContract,
    Observation,
    SensorIds,
    SpatialGoal,
)
from embodied.pilot.decisions import SceneStatus
from embodied.pilot.mission import mission_contract
from embodied.pilot.mission_executive import (
    ArmRefused,
    MissionPilot,
)
from embodied.pilot.provider import (
    ScriptedReply,
    ScriptedTransport,
)

HOST, CLOCK = "exec-test", "monotonic"


def make_png(width: int = 8, height: int = 6) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (180, 40, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_config_document() -> dict:
    return {
        "model": {
            "provider": "commandcode",
            "id": "deepseek/deepseek-v4.1-flash",
            "base_url": "https://api.commandcode.ai/provider/v1",
            "api": "openai-completions",
            "image_transport": "base64",
            "call_profiles": {
                "initial": {"image_scale": 1.0, "reasoning_effort": "high"},
                "continuous": {"image_scale": 0.25, "reasoning_effort": "off"},
            },
        },
        "pilot": {
            "event_deadband": 0.08,
            "max_silence_s": 6.0,
            "latency_budget_s": 2.5,
            "guard_margin_s": 1.0,
            "response_deadline_s": 4.0,
            "observation_freshness_s": 6.0,
            "retry_budget": 1,
        },
    }


def make_contract() -> MissionContract:
    return mission_contract(
        mission_id="first-indoor-B2-test",
        instruction="Find the red block, inspect it, and return to the start.",
        budget=(("wall_clock_s", 600.0),),
    )


class SeamDouble:
    """The admission seam as the runtime provides it: everything is admitted."""

    def __init__(self) -> None:
        self.admitted: list[str] = []

    def admit(self, proposal: SpatialGoal, context) -> GoalStatus:
        self.admitted.append(proposal.intent)
        return GoalStatus(
            proposal_id=proposal.proposal_id,
            admission="accepted",
            reason="test seam admits",
            goal_id="goal-1",
            goal_revision=1,
        )

    def cancel(self, *args, **kwargs):
        return None

    def status(self, goal_id: str):
        return None

    def invalidate_queued(self, goal_id: str) -> int:
        return 0


def make_observation(record_id: str, sequence: int) -> Observation:
    return Observation(
        episode_id="ep-1",
        record_id=record_id,
        sensor_ids=SensorIds(left="l", right="r", imu="i"),
        sequence=sequence,
        capture_stamp=ClockStamp(HOST, CLOCK, 1_000 + sequence),
        receipt_stamp=ClockStamp(HOST, CLOCK, 2_000 + sequence),
        sim_time_s=None,
        pair_id=f"pair-{record_id}",
        left_payload="left.ppm",
        right_payload="right.ppm",
        encoding="ppm-p3",
        width=8,
        height=6,
        calibration_id="cal-1",
        capture_pose_ref=None,
        quality=None,
        depth_source=None,
    )


GOOD_RECIPE_REPLY = {
    "choices": [
        {
            "message": {
                "content": json.dumps(
                    {
                        "mission_recipe": {
                            "steps": [
                                {"action": "explore", "target_kind": "frontier",
                                 "target_ref": "next_unvisited", "max_attempts": 2},
                                {"action": "inspect", "target_kind": "candidate",
                                 "target_ref": "next_uninspected", "max_attempts": 2},
                                {"action": "return", "target_kind": "place",
                                 "target_ref": "start", "max_attempts": 1},
                            ],
                            "bounds": {"max_steps": 3, "resource_ceiling": 3.0},
                        }
                    }
                )
            }
        }
    ]
}


def make_pilot(arm: str, *, replies: list[ScriptedReply] | None = None):
    transport = ScriptedTransport()
    if replies:
        transport.schedule(*replies)
    pilot = MissionPilot.for_arm(
        arm=arm,
        config_document=make_config_document(),
        contract=make_contract(),
        seam=SeamDouble(),
        transport=transport,
    )
    return pilot, transport


def stamp(offset_s: float) -> ClockStamp:
    return ClockStamp(HOST, CLOCK, int(offset_s * 1_000_000_000))


def test_b0_is_refused_because_it_owns_no_cloud():
    with pytest.raises(ArmRefused) as refused:
        make_pilot("B0")
    assert "B0" in str(refused.value)


def test_b1_plans_once_on_the_ground_and_never_calls_again():
    pilot, transport = make_pilot("B1", replies=[ScriptedReply(document=GOOD_RECIPE_REPLY)])
    pilot.set_mission(stamp(0.0))
    plan = pilot.plan_preflight(
        make_observation("obs-0", 0), {"left.ppm": make_png()}, stamp(0.0), deadline_s=90.0
    )
    assert plan.usable is True
    assert len(transport.sent_documents) == 1
    pilot.mark_airborne(stamp(1.0))
    # A whole flight's worth of ticks: B1 makes no further cloud call.
    for index in range(1, 8):
        pilot.tick(
            SceneStatus(signature=0.9, new_targets=("c-1",)),
            make_observation(f"obs-{index}", index),
            {"left.ppm": make_png()},
            stamp(float(index)),
        )
    assert len(transport.sent_documents) == 1


def test_b2_in_flight_calls_are_continuous_class_at_the_declared_scale():
    pilot, transport = make_pilot("B2", replies=[ScriptedReply(document=GOOD_RECIPE_REPLY)])
    pilot.set_mission(stamp(0.0))
    pilot.plan_preflight(
        make_observation("obs-0", 0), {"left.ppm": make_png()}, stamp(0.0), deadline_s=90.0
    )
    pilot.mark_airborne(stamp(1.0))
    # An explicit question is a trigger the shared engine always sends on.
    observation = make_observation("obs-1", 1)
    pilot.tick(
        SceneStatus(signature=None, explicit_question="which candidate is the target?"),
        observation,
        {"left.ppm": make_png()},
        stamp(2.0),
    )
    assert len(transport.sent_documents) == 2
    inflight = transport.sent_documents[1]
    assert inflight["reasoning_effort"] == "off"
    data_uri = next(
        part["image_url"]["url"] for part in inflight["messages"][0]["content"]
        if part.get("type") == "image_url"
    )
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))) as image:
        assert image.size == (2, 2)  # quarter of 8x6

    # And the run's own record carries what the call *was*, not only that one was
    # made. The packet trace was discarded at this seam, so a live B2 episode
    # held twenty `request` events and not one fact about any of them: no class,
    # no scale. The behaviour above was already tested; the recording was not.
    packets = [entry for entry in pilot.broker.sink.entries if entry[0] == "packet"]
    assert len(packets) == 1, [entry[0] for entry in pilot.broker.sink.entries]
    trace = packets[0][1]
    assert trace["call_class"] == "continuous"
    assert trace["image_scale"] == 0.25
    assert trace["observation_id"] == observation.record_id
    assert trace["model_identity"]


def test_no_plan_can_be_made_after_liftoff():
    pilot, _ = make_pilot("B2", replies=[ScriptedReply(document=GOOD_RECIPE_REPLY)])
    pilot.set_mission(stamp(0.0))
    pilot.mark_airborne(stamp(1.0))
    with pytest.raises(ArmRefused) as refused:
        pilot.plan_preflight(
            make_observation("obs-0", 0), {"left.ppm": make_png()}, stamp(2.0), deadline_s=90.0
        )
    assert "on the ground" in str(refused.value)


def test_a_refused_plan_is_a_complete_outcome_the_mission_can_fly_on():
    empty = {"choices": [{"message": {"content": "{}"}}]}
    pilot, transport = make_pilot("B1", replies=[ScriptedReply(document=empty)])
    pilot.set_mission(stamp(0.0))
    plan = pilot.plan_preflight(
        make_observation("obs-0", 0), {"left.ppm": make_png()}, stamp(0.0), deadline_s=90.0
    )
    assert plan.usable is False
    assert plan.refusal_reason
    # The fallback is the local arm's own recipe, not a fabricated cloud plan.
    from embodied.pilot.mission import build_b0_recipes

    assert build_b0_recipes()[0].source == "B0-local"
