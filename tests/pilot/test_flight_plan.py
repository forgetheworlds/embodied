"""The pre-flight plan: one reasoned call on the ground, bounded, or refused.

These tests drive the real Provider and the real packet builder through the
deterministic ScriptedTransport, so what is asserted is the document that
would have left the process and the recipe the mission would have run.
"""

from __future__ import annotations

import base64
import io
import json

import pytest

from embodied.contracts.records import ClockStamp, Observation, SensorIds
from embodied.pilot.flight_plan import (
    PreflightPlanner,
    PlanRefused,
    plan_from_reply,
)
from embodied.pilot.mission_packet import MissionPacketBuilder, ReasonedInFlightRefused
from embodied.pilot.provider import (
    CallProfile,
    ModelConfig,
    Provider,
    ScriptedReply,
    ScriptedTransport,
    TransportError,
)

HOST, CLOCK = "plan-test", "monotonic"


def make_png(width: int = 8, height: int = 6) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (180, 40, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_config() -> ModelConfig:
    return ModelConfig(
        provider="commandcode",
        id="deepseek/deepseek-v4.1-flash",
        base_url="https://api.commandcode.ai/provider/v1",
        api="openai-completions",
        image_transport="base64",
        call_profiles=(
            ("initial", CallProfile(generation=(("reasoning_effort", "high"),), image_scale=1.0)),
            ("continuous", CallProfile(generation=(("reasoning_effort", "off"),), image_scale=0.25)),
        ),
    )


def make_observation() -> Observation:
    return Observation(
        episode_id="ep-1",
        record_id="obs-1",
        sensor_ids=SensorIds(left="l", right="r", imu="i"),
        sequence=0,
        capture_stamp=ClockStamp(HOST, CLOCK, 1_000),
        receipt_stamp=ClockStamp(HOST, CLOCK, 2_000),
        sim_time_s=None,
        pair_id="pair-1",
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


def reply_with_recipe(recipe: dict | None) -> dict:
    content = {} if recipe is None else {"mission_recipe": recipe}
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


GOOD_RECIPE = {
    "steps": [
        {"action": "explore", "target_kind": "frontier", "target_ref": "next_unvisited", "max_attempts": 2},
        {"action": "inspect", "target_kind": "candidate", "target_ref": "next_uninspected",
         "guard_kind": "candidate_present", "max_attempts": 2},
        {"action": "return", "target_kind": "place", "target_ref": "start", "max_attempts": 1},
    ],
    "bounds": {"max_steps": 3, "resource_ceiling": 3.0},
}


def make_planner(*, retry_budget: int = 0):
    config = make_config()
    transport = ScriptedTransport()
    provider = Provider(config, transport, ())
    builder = MissionPacketBuilder(config, "Find the red block, inspect it, and return to the start.")
    planner = PreflightPlanner(provider, builder, retry_budget=retry_budget)
    return planner, transport, builder


def run_plan(planner, *, payloads=None):
    now = ClockStamp(HOST, CLOCK, 1_000_000_000)
    return planner.plan(
        observation=make_observation(),
        payloads=payloads if payloads is not None else {"left.ppm": make_png()},
        now=now,
        deadline_s=90.0,
    )


def test_a_usable_plan_is_bounded_validated_and_traceable():
    planner, transport, _ = make_planner()
    transport.schedule(ScriptedReply(document=reply_with_recipe(GOOD_RECIPE)))
    plan = run_plan(planner)
    assert plan.usable is True
    assert plan.recipe.source == "cloud-initial"
    assert [step.action for step in plan.recipe.steps] == ["explore", "inspect", "return"]
    assert plan.attempts == 1
    assert plan.model_identity == "deepseek/deepseek-v4.1-flash"
    assert plan.trace["call_class"] == "initial"
    assert plan.trace["image_scale"] == 1.0
    # The document that would have left the process: the initial profile's
    # effort reached the request, and the frame went at captured resolution.
    document = transport.sent_documents[0]
    assert document["reasoning_effort"] == "high"
    data_uri = next(
        part["image_url"]["url"] for part in document["messages"][0]["content"]
        if part.get("type") == "image_url"
    )
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))) as image:
        assert image.size == (8, 6)


def test_a_reply_without_a_recipe_is_a_refusal_not_an_error():
    planner, _, _ = make_planner()
    planner.provider.transport.schedule(ScriptedReply(document=reply_with_recipe(None)))
    plan = run_plan(planner)
    assert plan.usable is False
    assert "no mission_recipe" in plan.refusal_reason


def test_a_plan_wider_than_the_declared_effort_is_refused():
    wide = {"steps": [{"action": "explore", "max_attempts": 1} for _ in range(7)]}
    with pytest.raises(PlanRefused) as refused:
        plan_from_reply(type("Parsed", (), {"mission_recipe": wide})())
    assert "declared effort" in str(refused.value)


def test_bounds_beyond_the_declared_step_cap_are_refused():
    inflated = {
        "steps": [{"action": "explore", "max_attempts": 1}],
        "bounds": {"max_steps": 9},
    }
    with pytest.raises(PlanRefused) as refused:
        plan_from_reply(type("Parsed", (), {"mission_recipe": inflated})())
    assert "declared 1..6" in str(refused.value)


def test_a_bounded_loop_inside_the_declared_cap_is_accepted():
    """The measured live plan: three steps with max_steps 6 for the runner.

    An earlier rule of mine refused this, but nothing declared it: the declared
    bound is the conventional arm's own total effort, and a bounded loop is
    allowed by the specification.
    """
    plan = {
        "steps": [{"action": "explore", "max_attempts": 2} for _ in range(3)],
        "bounds": {"max_steps": 6, "resource_ceiling": 6.0},
    }
    recipe = plan_from_reply(type("Parsed", (), {"mission_recipe": plan})())
    assert recipe.max_steps == 6
    assert len(recipe.steps) == 3


def test_a_recipe_delivered_as_a_tool_call_is_validated():
    """The shape that measured working on the pinned route."""
    planner, transport, _ = make_planner()
    tool_reply = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "propose_mission_recipe",
                                "arguments": json.dumps(GOOD_RECIPE),
                            }
                        }
                    ]
                }
            }
        ]
    }
    transport.schedule(ScriptedReply(document=tool_reply))
    plan = run_plan(planner)
    assert plan.usable is True
    assert [step.action for step in plan.recipe.steps] == ["explore", "inspect", "return"]


def test_the_recipe_tool_schema_carries_the_runner_vocabulary():
    """The schema is the interface, so it cannot drift from the validator."""
    from embodied.pilot.flight_plan import PLAN_STEPS_CAP, RECIPE_TOOL
    from embodied.pilot.recipe_runner import GUARDS, INTENTS

    props = RECIPE_TOOL["function"]["parameters"]["properties"]
    step_props = props["steps"]["items"]["properties"]
    assert step_props["action"]["enum"] == list(INTENTS)
    assert step_props["guard_kind"]["enum"] == list(GUARDS)
    assert step_props["target_kind"]["enum"] == ["candidate", "frontier", "place"]
    assert props["bounds"]["properties"]["max_steps"]["maximum"] == PLAN_STEPS_CAP


def test_a_plan_outside_the_runner_vocabulary_is_refused():
    alien = {"steps": [{"action": "fly_through_wall", "max_attempts": 1}]}
    with pytest.raises(PlanRefused) as refused:
        plan_from_reply(type("Parsed", (), {"mission_recipe": alien})())
    assert "fly_through_wall" in str(refused.value)


def test_a_transport_failure_is_retried_once_on_the_ground():
    planner, transport, _ = make_planner(retry_budget=1)
    transport.schedule(
        ScriptedReply(error=TransportError("HTTP 502: upstream")),
        ScriptedReply(document=reply_with_recipe(GOOD_RECIPE)),
    )
    plan = run_plan(planner)
    assert plan.usable is True
    assert plan.attempts == 2
    assert len(transport.sent_documents) == 2


def test_exhausted_retries_carry_the_transport_own_reason():
    """A recorded refusal names the failure, not only that there was one.

    Measured cause of this assertion existing: a live pre-flight call recorded
    "no reply arrived inside the pre-flight window" while the transport held the
    real diagnosis — an HTTP status and its body — in ``provider.failures``,
    returned to nobody. The refusal was true and useless, and the previous
    version of this test asserted only that the string was non-empty, so it
    passed while the reason was being thrown away.
    """
    planner, transport, _ = make_planner(retry_budget=1)
    transport.schedule(
        ScriptedReply(error=TransportError("HTTP 400: invalid tool schema")),
        ScriptedReply(error=TransportError("HTTP 400: invalid tool schema")),
    )
    plan = run_plan(planner)
    assert plan.usable is False
    assert plan.attempts == 2
    assert "no reply arrived" in plan.refusal_reason
    assert "HTTP 400: invalid tool schema" in plan.refusal_reason


def test_no_plan_can_be_made_once_the_aircraft_is_airborne():
    """The prohibition, at the level that issues the reasoned call."""
    planner, _, builder = make_planner()
    planner.provider.transport.schedule(ScriptedReply(document=reply_with_recipe(GOOD_RECIPE)))
    builder.mark_airborne("guided takeoff")
    with pytest.raises(ReasonedInFlightRefused):
        run_plan(planner)



def test_the_plan_question_defers_the_vocabulary_to_the_tool_schema():
    """The schema carries the vocabulary now, so the question only names the tool."""
    from embodied.pilot.flight_plan import RECIPE_TOOL_NAME, plan_question

    question = plan_question("Find the red block, inspect it, and return to the start.")
    assert RECIPE_TOOL_NAME in question
    assert "reply with nothing else" in question


def test_the_initial_prompt_names_the_plan_class_it_is_asking_for():
    """Measured cause of the first live refusal: the prompt never asked for a plan."""
    planner, transport, _ = make_planner()
    transport.schedule(ScriptedReply(document=reply_with_recipe(GOOD_RECIPE)))
    run_plan(planner)
    prompt = transport.sent_documents[0]["messages"][0]["content"][0]["text"]
    assert "initial mission plan" in prompt
    assert "mission_recipe" in prompt

def test_a_fenced_json_reply_is_parsed_not_discarded():
    """Measured: the third live reply was valid JSON inside a ```json fence."""
    planner, transport, _ = make_planner()
    fenced = {
        "choices": [
            {
                "message": {
                    "content": "```json\n"
                    + json.dumps({"mission_recipe": GOOD_RECIPE})
                    + "\n```"
                }
            }
        ]
    }
    transport.schedule(ScriptedReply(document=fenced))
    plan = run_plan(planner)
    assert plan.usable is True
    assert [step.action for step in plan.recipe.steps] == ["explore", "inspect", "return"]


def test_the_question_asks_for_the_typed_tool_by_name():
    """Prose schemas kept being reinvented; the question names the tool."""
    from embodied.pilot.flight_plan import RECIPE_TOOL_NAME, plan_question

    question = plan_question("Find the red block.")
    assert RECIPE_TOOL_NAME in question
    assert "do not name a route" in question