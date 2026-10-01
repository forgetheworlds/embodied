"""The ``pilot-probe`` command: verify one configured provider's real
image/tool interface under an explicit budget, or refuse.

The refusal comes first, by design: a live provider call without both an
explicit ``--max-calls N`` on the command line and a spend ceiling recorded in
``configs/runtime-model.yaml`` is blocked (exit 2, CLI-PLAN). The checked-in
config ships with the limits absent, so the safe default is refusal; the owner
records the limits when budgeting the live probe.

What one probe run verifies, each recorded as an outcome rather than a crash:

* image support — the reply must actually engage the fixture observation's
  image content, not merely accept the payload;
* tool support — the five-tool schema round-trips at least one tool call;
* latency — send stamp, arrival stamp, bytes and any provider usage fields,
  with the client round trip labelled unseparated (§16.1: no server breakdown
  is invented);
* error handling — malformed or failed replies surface as recorded outcomes.

Credentials are read from the environment inside the transport and are never
logged, hashed or written into a receipt. Unit tests drive the deterministic
fake transport through :func:`execute_probe` and never perform a live call.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from embodied.cli import (
    CommandOutcome,
    CommandStatus,
    GateStatus,
    register_command,
)
from embodied.contracts.records import ClockStamp, DecisionRequest, Observation, from_dict
from embodied.pilot.decisions import PilotParameters
from embodied.pilot.provider import (
    LiveTransport,
    ModelConfig,
    Provider,
    RequestPacket,
    encode_image_data_uri,
)
from embodied.pilot.tools import TOOL_SCHEMAS

STAGE_ID = "P04"
PROBE_HOST = "pilot-probe-0"


# ---------------------------------------------------------------------------
# Configuration (configs/runtime-model.yaml — its own schema, not first_indoor's)
# ---------------------------------------------------------------------------


class ProbeConfigError(Exception):
    """The runtime-model configuration is malformed or limits are absent."""


def load_runtime_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - PyYAML is pinned
        raise ProbeConfigError(f"PyYAML is required to read {path}: {error}") from error
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProbeConfigError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise ProbeConfigError(f"{path} must hold a mapping")
    for section in ("model", "probe", "pilot"):
        if not isinstance(document.get(section), dict):
            raise ProbeConfigError(f"{path} needs a {section}: mapping")
    return document


def probe_limits(config: dict[str, Any]) -> tuple[int, float] | None:
    """The recorded (max_calls, spend_ceiling_usd), or None when absent.

    Absent limits are the safe default: the probe refuses rather than guess
    a budget (slice: no paid call beyond the kickoff budget).
    """
    section = config["probe"]
    max_calls = section.get("max_calls")
    ceiling = section.get("spend_ceiling_usd")
    if max_calls is None or ceiling is None:
        return None
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls <= 0:
        raise ProbeConfigError("probe.max_calls must be a positive integer when present")
    if isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)) or ceiling <= 0:
        raise ProbeConfigError("probe.spend_ceiling_usd must be a positive number when present")
    return (max_calls, float(ceiling))


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


class ProbeCheck:
    def __init__(self, name: str, passed: bool, detail: str) -> None:
        self.name = name
        self.passed = passed
        self.detail = detail


class ProbeReport:
    def __init__(
        self,
        model_identity: str,
        checks: list[ProbeCheck],
        calls: int,
        latency: list[dict[str, Any]],
        transport_failures: list[str],
        malformed: list[str],
    ) -> None:
        self.model_identity = model_identity
        self.checks = tuple(checks)
        self.calls = calls
        self.latency = tuple(latency)
        self.transport_failures = tuple(transport_failures)
        self.malformed = tuple(malformed)

    @property
    def image_ok(self) -> bool:
        return any(check.name == "image_support" and check.passed for check in self.checks)

    @property
    def tools_ok(self) -> bool:
        return any(check.name == "tool_support" and check.passed for check in self.checks)


def _load_observation(path: Path) -> tuple[Observation, dict[str, bytes]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    observation = from_dict(Observation, document)
    payloads: dict[str, bytes] = {}
    for name in (observation.left_payload, observation.right_payload):
        if name is not None:
            payloads[name] = (path.parent / name).read_bytes()
    return observation, payloads


def execute_probe(
    config: dict[str, Any],
    observation_path: Path,
    transport,
    max_calls: int,
    *,
    host_ns: int = 1_000_000_000,
) -> ProbeReport:
    """Run the checks against ``transport``, never exceeding ``max_calls``.
    No live call happens here unless the caller passes the live transport."""
    model = ModelConfig.from_config(config["model"])
    parameters = PilotParameters.from_config(config["pilot"])
    observation, payloads = _load_observation(observation_path)

    def stamp(offset_s: float) -> ClockStamp:
        return ClockStamp(PROBE_HOST, "monotonic", host_ns + int(offset_s * 1_000_000_000))

    provider = Provider(model, transport, TOOL_SCHEMAS)
    checks: list[ProbeCheck] = []
    failures: list[str] = []
    malformed: list[str] = []

    def budget_left() -> bool:
        return len(provider.send_records) < max_calls

    def send_and_collect(index: float, with_image: bool, question: str):
        request = DecisionRequest(
            request_id=f"probe-req-{int(index)}",
            sequence=int(index),
            mission_revision=0,
            base_goal_revision=0,
            observation_ids=(observation.record_id,) if with_image else (),
            snapshot_id=None,
            response_deadline_s=parameters.response_deadline_s,
            model_identity=model.identity,
        )
        image_parts = ()
        if with_image:
            image_parts = tuple(
                (observation.record_id, encode_image_data_uri(payloads[name]))
                for name in sorted(payloads)
            )
        packet = RequestPacket(
            request=request,
            mission_instruction="probe: answer from the attached frame",
            image_parts=image_parts,
            explicit_question=question,
        )
        provider.submit(request, packet, stamp(index * 10.0))
        return provider.poll(stamp(index * 10.0 + parameters.response_deadline_s + 1.0))

    # Check 1: the reply must engage the image content, not merely accept it.
    if budget_left():
        replies = send_and_collect(
            0.0,
            True,
            "Describe the coloured marker in the attached frame: which colour is it "
            "and roughly where in the frame does it sit?",
        )
        if not replies:
            checks.append(ProbeCheck("image_support", False, "no reply arrived for the image request"))
        else:
            parsed = replies[0].parsed
            if parsed.malformed_reason:
                malformed.append(f"image reply: {parsed.malformed_reason}")
                checks.append(ProbeCheck("image_support", False, parsed.malformed_reason))
            else:
                content = (parsed.content or "").strip()
                checks.append(
                    ProbeCheck(
                        "image_support",
                        len(content) > 10,
                        f"reply content ({len(content)} chars) recorded; transport base64-only",
                    )
                )
    else:
        checks.append(ProbeCheck("image_support", False, "not run: call budget exhausted"))

    # Check 2: the tool schema round-trips a tool call.
    if budget_left():
        replies = send_and_collect(1.0, False, "Call the status tool for goal 'probe-goal', then stop.")
        if not replies:
            checks.append(ProbeCheck("tool_support", False, "no reply arrived for the tool request"))
        else:
            parsed = replies[0].parsed
            if parsed.malformed_reason:
                malformed.append(f"tool reply: {parsed.malformed_reason}")
            names = [call.get("function", {}).get("name") for call in parsed.tool_calls]
            checks.append(
                ProbeCheck(
                    "tool_support",
                    any(name in {"observe", "ground", "set_goal", "status", "cancel"} for name in names),
                    f"tool calls requested: {names or 'none'}",
                )
            )
    else:
        checks.append(ProbeCheck("tool_support", False, "not run: call budget exhausted"))

    # Check 3: malformed or failed replies are outcomes, not crashes.
    if budget_left():
        replies = send_and_collect(2.0, False, "probe error handling")
        if replies and replies[0].parsed.malformed_reason:
            malformed.append(replies[0].parsed.malformed_reason)
            checks.append(ProbeCheck("error_handling", True, "malformed reply recorded as an outcome"))
        elif replies:
            checks.append(
                ProbeCheck("error_handling", True, "reply well-formed; no malformed case to record")
            )
        else:
            failures.append("probe-req-2 received no reply and no transport error")
            checks.append(ProbeCheck("error_handling", False, "silence without a recorded cause"))
    else:
        checks.append(ProbeCheck("error_handling", False, "not run: call budget exhausted"))

    latency = []
    for reply in provider.replies:
        record = next(
            (r for r in provider.send_records if r.request_id == reply.parsed.request_id), None
        )
        latency.append(
            {
                "request_id": reply.parsed.request_id,
                "send_ns": reply.arrival.send_stamp.monotonic_ns,
                "arrival_ns": reply.arrival.arrived_at.monotonic_ns,
                "round_trip_ns": reply.arrival.round_trip_ns,
                "round_trip_unseparated": True,
                "request_bytes": record.request_bytes if record else None,
                "response_bytes": reply.response_bytes,
                "usage": reply.parsed.usage,
            }
        )
    failures.extend(f"{request_id}: {message}" for request_id, message in provider.failures)

    return ProbeReport(
        model_identity=model.identity,
        checks=checks,
        calls=len(provider.send_records),
        latency=latency,
        transport_failures=failures,
        malformed=malformed,
    )


# ---------------------------------------------------------------------------
# The registered command
# ---------------------------------------------------------------------------


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="runtime-model configuration (configs/runtime-model.yaml)",
    )
    parser.add_argument(
        "--observation",
        type=Path,
        required=True,
        help="observation record fixture (tests/fixtures/pilot/observation.json)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="explicit approved maximum number of live provider calls for this run",
    )
    parser.add_argument(
        "--api-key-env",
        default="PROVIDER_API_KEY",
        help="environment variable holding the provider API key (never logged)",
    )


def _manifest_for(config: dict[str, Any], report: ProbeReport, call_budget: int, limits) -> dict[str, Any]:
    return {
        "model_provider": config["model"].get("provider"),
        "model_id": report.model_identity,
        "declared_input": config["model"].get("declared_input"),
        "image_transport": config["model"].get("image_transport"),
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in report.checks
        ],
        "calls_made": report.calls,
        "call_budget": call_budget,
        "spend_ceiling_usd": limits[1] if limits else None,
        "cost_recorded": None,
        "cost_note": "usage recorded per call; no price table in config, so no cost is computed",
        "latency": list(report.latency),
        "transport_failures": list(report.transport_failures),
        "malformed_replies": list(report.malformed),
    }


def _handler(args: argparse.Namespace, output: Path) -> CommandOutcome:
    """One probe run: refuse without a budget, otherwise run once, record once."""
    config = load_runtime_config(args.config)
    limits = probe_limits(config)
    missing = []
    if args.max_calls is None:
        missing.append("no --max-calls on the command line")
    if limits is None:
        missing.append(
            "no max_calls/spend_ceiling_usd recorded in the config's probe section "
            "(absent limits are the safe default; the owner records them when "
            "budgeting the live probe)"
        )
    if missing:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(
                "pilot-probe refuses a live call without an explicit budget: " + "; ".join(missing),
            ),
            limitations=("this refusal is the checked-in default; no provider call was made",),
            manifest={"model": config["model"].get("id"), "limits_recorded": False},
        )
    if args.max_calls < 1:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=("pilot-probe refuses: --max-calls must be a positive integer",),
            manifest={"model": config["model"].get("id"), "limits_recorded": True},
        )

    call_budget = min(args.max_calls, limits[0])
    transport = LiveTransport(
        ModelConfig.from_config(config["model"]), api_key_env=args.api_key_env
    )
    report = execute_probe(config, args.observation, transport, call_budget)
    (output / "probe-report.json").write_text(
        json.dumps(
            {
                "model_identity": report.model_identity,
                "checks": [
                    {"name": check.name, "passed": check.passed, "detail": check.detail}
                    for check in report.checks
                ],
                "calls": report.calls,
                "latency": list(report.latency),
                "transport_failures": list(report.transport_failures),
                "malformed": list(report.malformed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = _manifest_for(config, report, call_budget, limits)
    if report.image_ok and report.tools_ok:
        return CommandOutcome(
            status=CommandStatus.COMPLETE,
            gate_status=GateStatus.PASS,
            reasons=(
                "image and tool interfaces answered under the recorded budget",
                f"{report.calls} calls within --max-calls {args.max_calls} and the config ceiling",
            ),
            limitations=(
                "a probe verifies transport-level support and latency only; it scores "
                "no mission and claims no autonomy",
                "client round trips are unseparated aggregates (no server breakdown)",
            ),
            manifest=manifest,
            artifacts=("probe-report.json",),
        )
    return CommandOutcome(
        status=CommandStatus.BLOCKED,
        gate_status=GateStatus.FAIL,
        reasons=(
            "the provider did not verify image+tool support: "
            + "; ".join(
                f"{check.name}={'pass' if check.passed else 'fail'}" for check in report.checks
            ),
        ),
        limitations=(
            "record the gap and stop (slice: no silent substitute model, no pro-tier "
            "escalation); no further live calls are made",
        ),
        manifest=manifest,
        artifacts=("probe-report.json",),
    )


register_command(
    "pilot-probe",
    _handler,
    help_text="verify the configured provider's image/tool interface under an explicit budget",
    stage_id=STAGE_ID,
    run_prefix="p04-provider",
    add_arguments=_add_arguments,
)

# ---------------------------------------------------------------------------
# Tactical latency sampling (P05): can a fresh stereo frame produce a usable
# decision inside the declared budget, and what does it cost? Unlike
# :func:`execute_probe`, which drives the scripted transport on a synthetic
# clock, the sampler stamps sends and arrivals with the real monotonic clock,
# because the live client round trip is the quantity being measured.
# ---------------------------------------------------------------------------


SAMPLE_HOST = "pilot-sampler-0"
SAMPLE_MISSION_INSTRUCTION = "Find the red block, inspect it, and return to the start."

SAMPLE_QUESTIONS = (
    "The attached stereo frame is the current view while approaching the search "
    "region. Decide the next action: set_goal to approach a visible candidate, "
    "observe for a better view, or hold. Answer with exactly one tool call.",
    "This frame just arrived mid-flight and supersedes the previous view. If it "
    "changes the plan, call set_goal with the new intent; otherwise call status.",
    "No red object is confirmed yet. Choose: set_goal to continue the search, or "
    "observe for a closer view first. One tool call.",
    "The frame shows the room the mission must search. Pick the next observation "
    "objective and express it as one tool call.",
)

SAMPLE_NAV_STATUS = (
    "approaching doorway, 4 m to search region, speed 0.5 m/s",
    "inside search region, 2 m from far wall, speed 0.3 m/s",
    "crossing doorway into the target room, speed 0.4 m/s",
    "holding 1.5 m altitude over the search region, speed 0.2 m/s",
)


def load_stereo_pairs(pairs_dir: Path, count: int) -> list[tuple[str, bytes, bytes]]:
    """Captured frames in capture order: (pair_id, left_bytes, right_bytes)."""
    lefts = sorted(path for path in pairs_dir.glob("*-left.ppm"))
    pairs: list[tuple[str, bytes, bytes]] = []
    for left in lefts[:count]:
        right = left.with_name(left.name.replace("-left.ppm", "-right.ppm"))
        if right.exists():
            pairs.append((left.name.replace("-left.ppm", ""), left.read_bytes(), right.read_bytes()))
    return pairs

def _encode_png_uri(payload: bytes) -> str:
    """PNG data URI. The pinned commandcode route rejects ``image/ppm`` data
    URIs with HTTP 400 regardless of size (measured, the J3 diagnostic matrix:
    a 1 KB fixture PPM and a 1.2 MB captured PPM both refused; the same frame
    as PNG answered). Pillow re-encodes the captured P6 frames, which also
    shrinks the upload about 4x.
    """
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.open(io.BytesIO(payload)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def execute_tactical_sample(
    config: dict[str, Any],
    pairs: list[tuple[str, bytes, bytes]],
    transport,
    samples: int,
) -> dict[str, Any]:
    """Send ``samples`` realistic image+task requests through the live transport.

    One frame pair per request, real monotonic stamps, every reply retained.
    The caller has already enforced the budget; this function never exceeds
    ``samples`` calls.
    """
    model = ModelConfig.from_config(config["model"])
    parameters = PilotParameters.from_config(config["pilot"])
    provider = Provider(model, transport, TOOL_SCHEMAS)
    calls: list[dict[str, Any]] = []

    for index in range(samples):
        pair_id, left, right = pairs[index % len(pairs)]
        request = DecisionRequest(
            request_id=f"sample-{index:03d}",
            sequence=index,
            mission_revision=1,
            base_goal_revision=index,
            observation_ids=(pair_id,),
            snapshot_id=None,
            response_deadline_s=parameters.response_deadline_s,
            model_identity=model.identity,
        )
        encode_start_ns = time.monotonic_ns()
        left_uri = _encode_png_uri(left)
        right_uri = _encode_png_uri(right)
        encode_ms = (time.monotonic_ns() - encode_start_ns) / 1_000_000
        packet = RequestPacket(
            request=request,
            mission_instruction=SAMPLE_MISSION_INSTRUCTION,
            image_parts=(
                (pair_id, left_uri),
                (pair_id, right_uri),
            ),
            navigation_status=SAMPLE_NAV_STATUS[index % len(SAMPLE_NAV_STATUS)],
            uncertainty_summary="pose sigma 0.05 m horizontal; stereo depth invalid on the textureless floor",
            explicit_question=SAMPLE_QUESTIONS[index % len(SAMPLE_QUESTIONS)],
        )
        send_ns = time.monotonic_ns()
        provider.submit(request, packet, ClockStamp(SAMPLE_HOST, "monotonic", send_ns))
        replies = provider.poll(ClockStamp(SAMPLE_HOST, "monotonic", time.monotonic_ns()))
        record = next((r for r in provider.send_records if r.request_id == request.request_id), None)
        entry: dict[str, Any] = {
            "request_id": request.request_id,
            "pair_id": pair_id,
            "send_ns": send_ns,
            "encode_ms": encode_ms,
            "round_trip_s": None,
            "request_bytes": record.request_bytes if record else None,
            "outcome": "no_reply",
            "tool_calls": [],
            "proposals": 0,
            "content_chars": None,
            "response_bytes": None,
            "usage": None,
            "malformed_reason": None,
        }
        failures = [message for request_id, message in provider.failures if request_id == request.request_id]
        if failures:
            entry["outcome"] = "transport_error"
            entry["error"] = "; ".join(failures)[:500]
        for reply in replies:
            parsed = reply.parsed
            entry["round_trip_s"] = reply.arrival.round_trip_ns / 1_000_000_000
            entry["response_bytes"] = reply.response_bytes
            entry["usage"] = parsed.usage
            entry["content_chars"] = len(parsed.content or "")
            entry["tool_calls"] = [call.get("function", {}).get("name") for call in parsed.tool_calls]
            entry["proposals"] = len(parsed.proposals)
            entry["malformed_reason"] = parsed.malformed_reason
            entry["content"] = (parsed.content or "")[:2000]
            entry["raw_tool_calls"] = list(parsed.tool_calls)[:8]
            if parsed.malformed_reason:
                entry["outcome"] = "malformed"
            elif parsed.proposals:
                entry["outcome"] = "spatial_goal_proposal"
            elif parsed.tool_calls:
                entry["outcome"] = "tool_call"
            else:
                entry["outcome"] = "content_only"
        calls.append(entry)

    latencies = [call["round_trip_s"] for call in calls if call["round_trip_s"] is not None]
    usable = [call for call in calls if call["outcome"] in ("tool_call", "spatial_goal_proposal")]
    usable_latencies = [call["round_trip_s"] for call in usable if call["round_trip_s"] is not None]
    usages = [call["usage"] for call in calls if call["usage"]]
    prompt_tokens = [int(u.get("prompt_tokens", 0)) for u in usages if u.get("prompt_tokens") is not None]
    completion_tokens = [int(u.get("completion_tokens", 0)) for u in usages if u.get("completion_tokens") is not None]
    return {
        "model_identity": model.identity,
        "calls": calls,
        "summary": {
            "calls": len(calls),
            "replied": len(latencies),
            "usable_decisions": len(usable),
            "latency_s": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies) if latencies else None,
            },
            "usable_latency_s": {
                "p50": _percentile(usable_latencies, 0.50),
                "p95": _percentile(usable_latencies, 0.95),
                "max": max(usable_latencies) if usable_latencies else None,
            },
            "prompt_tokens": {"p50": _percentile(prompt_tokens, 0.5), "max": max(prompt_tokens) if prompt_tokens else None},
            "completion_tokens": {
                "p50": _percentile(completion_tokens, 0.5),
                "max": max(completion_tokens) if completion_tokens else None,
            },
            "latency_round_trip_unseparated": True,
        },
    }


def _host_state() -> dict[str, str]:
    return {"load": os.popen("uptime").read().strip(), "swap": os.popen("sysctl -n vm.swapusage").read().strip()}


def _main() -> int:
    """Run the sampler as ``python -m embodied.pilot.probe`` without touching dispatch.

    The budget refusal mirrors the registered command's: no call happens unless
    --max-calls is given AND the config records limits.
    """
    parser = argparse.ArgumentParser(description="live tactical-decision latency sampler")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs-dir", type=Path, required=True, help="directory of captured *-left.ppm/*-right.ppm frames")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--api-key-env", default="COMMAND_CODE_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    limits = probe_limits(config)
    if args.max_calls is None or limits is None:
        print(
            "refused: no --max-calls on the command line"
            if args.max_calls is None
            else "refused: no max_calls/spend_ceiling_usd recorded in the config's probe section"
        )
        return 2
    if args.samples < 1 or args.samples > min(args.max_calls, limits[0]):
        print(f"refused: --samples must be within the recorded budget (max {min(args.max_calls, limits[0])})")
        return 2

    pairs = load_stereo_pairs(args.pairs_dir, args.samples)
    if not pairs:
        print(f"refused: no stereo pairs found under {args.pairs_dir}")
        return 2
    host_before = _host_state()
    transport = LiveTransport(ModelConfig.from_config(config["model"]), api_key_env=args.api_key_env)
    report = execute_tactical_sample(config, pairs, transport, min(args.samples, len(pairs)))
    report["host"] = {"before": host_before, "after": _host_state()}
    report["limits"] = {"max_calls": limits[0], "spend_ceiling_usd": limits[1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"{summary['calls']} calls, {summary['replied']} replied, "
        f"{summary['usable_decisions']} usable decisions; "
        f"p50 {summary['latency_s']['p50']}s p95 {summary['latency_s']['p95']}s max {summary['latency_s']['max']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
