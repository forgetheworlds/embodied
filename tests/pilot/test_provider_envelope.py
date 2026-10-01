"""The provider's request envelope and transport failure reporting.

These are the three facts J4 depended on, each of which was a real defect or a
real lever before it was a test:

* the pinned commandcode route refuses ``image/ppm`` data URIs outright, so the
  single encoding path emits PNG and re-encodes a captured P6/P3 PPM on the way;
* ``LiveTransport`` must surface an HTTP error's status and body, because the
  body is what named the accepted ``reasoning_effort`` vocabulary in one call
  after "transport failed" had hidden it;
* generation parameters and the reply format are explicit configuration, so a
  receipt can always name the request shape that produced its numbers.
"""

from __future__ import annotations

import base64
import io
import urllib.error
import urllib.request

from embodied.pilot.provider import (
    LiveTransport,
    ModelConfig,
    RequestPacket,
    build_request_document,
    encode_image_data_uri,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _ppm() -> bytes:
    """A 1x1 P3 (ASCII) PPM — the fixture format the captures are stored in."""
    return b"P3\n1 1\n255\n255 0 0\n"


def _packet() -> RequestPacket:
    from embodied.contracts.records import DecisionRequest

    request = DecisionRequest(
        request_id="req-0",
        sequence=0,
        mission_revision=0,
        base_goal_revision=0,
        observation_ids=(),
        snapshot_id=None,
        response_deadline_s=4.0,
        model_identity="test/model",
    )
    return RequestPacket(request=request, mission_instruction="probe")


def test_a_ppm_capture_is_delivered_as_png():
    uri = encode_image_data_uri(_ppm())
    assert uri.startswith("data:image/png;base64,")
    payload = base64.b64decode(uri.split(",", 1)[1])
    assert payload.startswith(PNG_MAGIC)


def test_an_already_png_payload_passes_through_unchanged():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (0, 0, 255)).save(buffer, format="PNG")
    original = buffer.getvalue()

    uri = encode_image_data_uri(original)

    assert base64.b64decode(uri.split(",", 1)[1]) == original


def test_the_transport_surfaces_an_http_error_status_and_body():
    config = ModelConfig(
        provider="test",
        id="test/model",
        base_url="https://example.invalid/v1",
        api="openai-completions",
        image_transport="base64",
    )
    body = b'{"error":{"message":"Invalid option: expected one of \\"off\\"|\\"low\\""}}'

    def refusing_opener(request: urllib.request.Request):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    from embodied.contracts.records import ClockStamp

    transport = LiveTransport(config, api_key_env="UNUSED", opener=refusing_opener)
    stamp = ClockStamp("host-0", "monotonic", 1_000)
    transport.send({}, stamp)
    arrivals = transport.poll(stamp)

    assert len(arrivals) == 1
    error = arrivals[0].error
    assert error is not None
    # The status and the provider's own message, not a bare "transport failed".
    assert "HTTP 400" in str(error)
    assert "Invalid option" in str(error)


def test_generation_and_reply_format_come_from_config_and_reach_the_request():
    config = ModelConfig.from_config(
        {
            "provider": "test",
            "id": "test/model",
            "base_url": "https://example.invalid/v1",
            "api": "openai-completions",
            "image_transport": "base64",
            "reply_format": "strict",
            "generation": {"reasoning_effort": "off", "max_tokens": 512},
        }
    )
    document = build_request_document(config, _packet(), ())

    assert document["reasoning_effort"] == "off"
    assert document["max_tokens"] == 512
    prompt = document["messages"][0]["content"][0]["text"]
    assert "EXACTLY ONE tool call" in prompt


def test_the_default_request_carries_no_generation_parameters():
    config = ModelConfig.from_config(
        {
            "provider": "test",
            "id": "test/model",
            "base_url": "https://example.invalid/v1",
            "api": "openai-completions",
            "image_transport": "base64",
        }
    )
    document = build_request_document(config, _packet(), ())

    assert "reasoning_effort" not in document
    assert "max_tokens" not in document
    assert "EXACTLY ONE tool call" not in document["messages"][0]["content"][0]["text"]
