"""The mission packet builder: frames, class, scale, and the airborne latch.

The owner's ruling of 2026-10-01 is the contract under test: one reasoned
call while the aircraft is on the ground, continuous calls only once it moves.
The latch is the structural form of that ruling — these tests would fail if
any in-flight path could issue a reasoned call, because the builder is the
only way a packet exists.
"""

from __future__ import annotations

import base64
import io

import pytest

from embodied.contracts.records import ClockStamp, DecisionRequest, Observation, SensorIds
from embodied.pilot.mission_packet import (
    MissionPacketBuilder,
    PacketRefused,
    ReasonedInFlightRefused,
)
from embodied.pilot.provider import (
    CALL_CONTINUOUS,
    CALL_INITIAL,
    ModelConfig,
    encode_image_data_uri,
)

HOST, CLOCK = "packet-test", "monotonic"


def make_png(width: int = 8, height: int = 6, colour=(180, 40, 40)) -> bytes:
    """A real PNG of a solid colour, so scaling exercises Pillow, not a stub."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


SCALE_PNG_8X6 = make_png()


def make_config() -> ModelConfig:
    return ModelConfig(
        provider="commandcode",
        id="deepseek/deepseek-v4.1-flash",
        base_url="https://api.commandcode.ai/provider/v1",
        api="openai-completions",
        image_transport="base64",
        call_profiles=(
            ("initial", type("P", (), {"generation": (), "image_scale": 1.0})()),
            ("continuous", type("P", (), {"generation": (), "image_scale": 0.25})()),
        ),
    )


def make_observation(record_id: str = "obs-1") -> Observation:
    return Observation(
        episode_id="ep-1",
        record_id=record_id,
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


def make_request(request_id: str = "req-1") -> DecisionRequest:
    return DecisionRequest(
        request_id=request_id,
        sequence=0,
        mission_revision=0,
        base_goal_revision=0,
        observation_ids=("obs-1",),
        snapshot_id=None,
        response_deadline_s=90.0,
        model_identity="deepseek/deepseek-v4.1-flash",
    )


def png_size(data_uri: str) -> tuple[int, int]:
    from PIL import Image

    payload = base64.b64decode(data_uri.split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as image:
        return image.size


def test_a_continuous_packet_carries_both_frames_at_the_declared_scale():
    builder = MissionPacketBuilder(make_config(), "Find the red block, inspect it, and return to the start.")
    packet, trace = builder.build(
        request=make_request(),
        observation=make_observation(),
        payloads={"left.ppm": SCALE_PNG_8X6, "right.ppm": SCALE_PNG_8X6},
        call_class=CALL_CONTINUOUS,
    )
    assert packet.call_class == CALL_CONTINUOUS
    assert len(packet.image_parts) == 2
    # 0.25 of 8x6 is 2x2: the declared profile's scale reached the frames.
    assert png_size(packet.image_parts[0][1]) == (2, 2)
    assert trace.image_scale == 0.25
    assert trace.frames == ("left.ppm", "right.ppm")
    assert trace.observation_id == "obs-1"
    assert trace.model_identity == "deepseek/deepseek-v4.1-flash"
    assert packet.mission_instruction.startswith("Find the red block")


def test_an_initial_packet_on_the_ground_keeps_the_captured_resolution():
    builder = MissionPacketBuilder(make_config(), "Find the red block.")
    packet, trace = builder.build(
        request=make_request(),
        observation=make_observation(),
        payloads={"left.ppm": SCALE_PNG_8X6},
        call_class=CALL_INITIAL,
    )
    assert png_size(packet.image_parts[0][1]) == (8, 6)
    assert trace.image_scale == 1.0
    assert trace.call_class == CALL_INITIAL


def test_the_airborne_latch_makes_a_reasoned_call_unbuildable_in_flight():
    """The prohibition is structural: after liftoff the initial class refuses."""
    builder = MissionPacketBuilder(make_config(), "Find the red block.")
    builder.mark_airborne("guided takeoff")
    with pytest.raises(ReasonedInFlightRefused) as refused:
        builder.build(
            request=make_request(),
            observation=make_observation(),
            payloads={"left.ppm": SCALE_PNG_8X6},
            call_class=CALL_INITIAL,
        )
    assert "airborne" in str(refused.value)
    # The continuous class is unaffected: motion is exactly what it is for.
    packet, _ = builder.build(
        request=make_request("req-2"),
        observation=make_observation(),
        payloads={"left.ppm": SCALE_PNG_8X6},
        call_class=CALL_CONTINUOUS,
    )
    assert packet.call_class == CALL_CONTINUOUS


def test_the_latch_only_closes():
    builder = MissionPacketBuilder(make_config(), "Find the red block.")
    builder.mark_airborne("first liftoff")
    builder.mark_airborne("a later call cannot re-open it")
    with pytest.raises(ReasonedInFlightRefused) as refused:
        builder.build(
            request=make_request(),
            observation=make_observation(),
            payloads={"left.ppm": SCALE_PNG_8X6},
            call_class=CALL_INITIAL,
        )
    assert "first liftoff" in str(refused.value)


def test_an_undeclared_call_class_is_refused_not_guessed():
    builder = MissionPacketBuilder(make_config(), "Find the red block.")
    with pytest.raises(PacketRefused) as refused:
        builder.build(
            request=make_request(),
            observation=make_observation(),
            payloads={"left.ppm": SCALE_PNG_8X6},
            call_class="urgent",
        )
    assert "urgent" in str(refused.value)


def test_the_builder_agrees_with_the_provider_on_the_encoded_form():
    """The frames the builder sends are the frames the measured path produced."""
    encoded = encode_image_data_uri(SCALE_PNG_8X6, scale=0.25)
    builder = MissionPacketBuilder(make_config(), "Find the red block.")
    packet, _ = builder.build(
        request=make_request(),
        observation=make_observation(),
        payloads={"left.ppm": SCALE_PNG_8X6},
        call_class=CALL_CONTINUOUS,
    )
    assert packet.image_parts[0][1] == encoded
