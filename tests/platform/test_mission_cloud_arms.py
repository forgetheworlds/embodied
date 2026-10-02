"""The cloud arms' wiring into the mission runtime.

What this file proves, each against the runtime the mission actually flies:

* **B0 is untouched.** The conventional arm builds no pilot and its broker has
  no provider, which is the isolation rather than a placeholder — asserted, not
  assumed, because the whole P06 comparison depends on B0 making no call.
* **A cloud arm is constructed from the declared route**, out of the runtime
  model configuration rather than the platform configuration, which carries no
  ``model`` section at all.
* **The reasoned call's deadline is a declared value**, the suite's own startup
  window, so the integration introduces no new number.
* **The latch closes at liftoff and never reopens**, through the runtime's own
  pilot object: after ``mark_airborne`` the reasoned call raises, and the packet
  builder refuses an initial-class packet.
* **The in-flight step carries what the runtime has**, and nothing it has not:
  the map revision as the scene signature, each newly grounded target once, and
  no horizon, because this runtime computes none.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
REAL_PLATFORM_CONFIG = REPOSITORY / "configs" / "first_indoor.yaml"
RUNTIME_MODEL_CONFIG = REPOSITORY / "configs" / "runtime-model.yaml"
SUITE_MISSION = REPOSITORY / "scenarios" / "first_indoor" / "mission.yaml"


def _runtime(tmp_path, *, arm: str = "B0", episode_id: str = "arms-0"):
    from embodied.bench.recorder import Recorder
    from embodied.platform.localization_check import _load_localization_config, _platform_settings
    from embodied.platform.mission_runtime import MissionRuntime

    document = _load_localization_config(REAL_PLATFORM_CONFIG)
    settings = _platform_settings(document, REPOSITORY)
    episode_dir = tmp_path / "episode"
    recorder = Recorder(episode_dir)
    runtime = MissionRuntime(
        settings=settings,
        config_document=document,
        episode_dir=episode_dir,
        evidence_dir=tmp_path / "platform",
        recorder=recorder,
        instruction="Find the red block, inspect it, and return to the start.",
        target_id="red_block",
        episode_id=episode_id,
        arm=arm,
    )
    return runtime


# ---------------------------------------------------------------------------
# B0 is untouched
# ---------------------------------------------------------------------------


def test_b0_builds_no_pilot_and_its_broker_has_no_provider(tmp_path):
    """The conventional arm makes no cloud call, and that is a structure here."""
    runtime = _runtime(tmp_path, arm="B0")
    assert runtime._pilot is None
    assert runtime.broker.provider is None, (
        "B0's broker must carry no provider: the field's absence is the isolation, "
        "and a P06 comparison depends on B0 making no call at all"
    )
    assert runtime.arm == "B0"


def test_a_b0_run_records_no_plan_and_no_cloud_calls(tmp_path):
    runtime = _runtime(tmp_path, arm="B0")
    assert runtime.result.plan == {}
    assert runtime.result.cloud_calls == []
    assert runtime._preflight_observation is None


# ---------------------------------------------------------------------------
# A cloud arm is constructed from the declared route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["B1", "B2"])
def test_a_cloud_arm_builds_a_pilot_and_runs_on_the_pilots_broker(tmp_path, arm):
    runtime = _runtime(tmp_path, arm=arm)
    assert runtime._pilot is not None
    assert runtime._pilot.arm == arm
    assert runtime.broker is runtime._pilot.broker, (
        "a cloud mission must run on the pilot's broker, which owns the provider "
        "and the cloud evidence; a second broker would split the mission's state"
    )
    assert runtime.broker.provider is not None
    assert runtime._airborne is False, "the aircraft starts on the ground"


def test_the_cloud_route_is_read_from_the_runtime_model_configuration(tmp_path):
    """The pilot's route is the declared one, read from its own file."""
    declared = yaml.safe_load(RUNTIME_MODEL_CONFIG.read_text(encoding="utf-8"))
    runtime = _runtime(tmp_path, arm="B1")
    assert runtime._pilot.config.identity == declared["model"]["id"]
    # The two call profiles the owner's ruling fixes: initial reasons at full
    # scale, continuous reasons off at a quarter scale.
    assert runtime._pilot.config.image_scale_for("initial") == 1.0
    assert runtime._pilot.config.image_scale_for("continuous") == 0.25


def test_the_preflight_deadline_is_the_suites_own_declared_startup_window(tmp_path):
    """No new number: the deadline is the declared startup window, both sides read."""
    suite = yaml.safe_load(SUITE_MISSION.read_text(encoding="utf-8"))
    runtime = _runtime(tmp_path, arm="B1")
    declared_suite_window = float(suite["budgets"]["step_timeout_s"]["startup"])
    assert runtime.settings.step_timeout_s.startup == declared_suite_window, (
        "the runtime's startup window and the suite's must be the same declared "
        "value, or the integration has introduced a number nobody declared"
    )


def test_an_unknown_arm_is_refused_rather_than_guessed(tmp_path):
    from embodied.pilot.mission_executive import ArmRefused

    with pytest.raises(ArmRefused):
        _runtime(tmp_path, arm="B3")


# ---------------------------------------------------------------------------
# The latch
# ---------------------------------------------------------------------------


def test_after_liftoff_the_reasoned_call_is_impossible_through_the_pilot(tmp_path):
    """The runtime's own pilot refuses the reasoned call once it is airborne."""
    from embodied.pilot.mission_executive import ArmRefused

    runtime = _runtime(tmp_path, arm="B1")
    pilot = runtime._pilot
    assert pilot.airborne is False
    pilot.mark_airborne(runtime._clock(), reason="guided takeoff")
    assert pilot.airborne is True
    with pytest.raises(ArmRefused):
        pilot.plan_preflight(None, {}, runtime._clock(), deadline_s=90.0)


def test_after_liftoff_an_initial_class_packet_cannot_be_built(tmp_path):
    """The builder's latch is the structural half of the owner's ruling."""
    from embodied.pilot.mission_packet import ReasonedInFlightRefused
    from embodied.pilot.provider import CALL_INITIAL

    runtime = _runtime(tmp_path, arm="B1")
    runtime._pilot.mark_airborne(runtime._clock(), reason="guided takeoff")
    builder = runtime._pilot.packet_builder
    with pytest.raises(ReasonedInFlightRefused):
        builder.build(
            request=SimpleNamespace(request_id="req-0"),
            observation=SimpleNamespace(record_id="obs-0"),
            payloads={},
            call_class=CALL_INITIAL,
        )


def test_the_runtime_marks_liftoff_only_after_a_successful_arming(tmp_path):
    """Liftoff is marked where the aircraft is armed, and nowhere earlier.

    Read from the source rather than driven, because driving it needs a
    simulator: the runtime's own marking must sit after the arming call and
    before the mission phases, or the reasoned call could be made in flight.
    """
    import inspect

    from embodied.platform import mission_runtime as mr

    source = inspect.getsource(mr.MissionRuntime.run)
    arm_at = source.index("arm_and_guided(")
    mark_at = source.index("mark_airborne(")
    plan_at = source.index("_plan_before_liftoff(")
    fly_at = source.index("_fly_the_mission(")
    assert plan_at < arm_at, "the reasoned call must be made before arming"
    assert arm_at < mark_at, "liftoff is marked only once the arming succeeded"
    assert mark_at < fly_at, "the latch must close before any in-flight call exists"


# ---------------------------------------------------------------------------
# The in-flight step carries what the runtime has, and nothing it has not
# ---------------------------------------------------------------------------
class _RecordingPilot:
    """A stand-in that records what the runtime handed it. No network."""

    def __init__(self, outcomes=()):
        self.calls = []
        self._outcomes = tuple(outcomes)

    def tick(self, scene, observation, payloads, now):
        self.calls.append((scene, observation, payloads, now))
        return self._outcomes


def test_the_inflight_step_reports_the_scene_signature_and_each_new_target_once(tmp_path):
    runtime = _runtime(tmp_path, arm="B2")
    stub = _RecordingPilot()
    runtime._pilot = stub
    runtime._airborne = True
    runtime._candidate_targets.extend(["cand-a", "cand-b"])

    runtime._tick_cloud("obs-1", {"left": b"l", "right": b"r"})
    assert len(stub.calls) == 1
    scene, _, payloads, _ = stub.calls[0]
    assert scene.signature == float(len(runtime.store.free_cells(now_ns=runtime._now_ns()))), (
        "the scene signature is the map's free-cell count, a declared proxy for a "
        "scene-change score this runtime does not compute"
    )
    assert set(scene.new_targets) == {"cand-a", "cand-b"}
    assert scene.horizon_s is None, (
        "this runtime computes no horizon; passing a made-up one would let the "
        "reduced-horizon trigger fire on a number nothing measured"
    )
    assert payloads == {"left": b"l", "right": b"r"}

    # The same targets are not new a second time.
    runtime._tick_cloud("obs-2", {"left": b"l", "right": b"r"})
    assert stub.calls[1][0].new_targets == ()


def test_an_inflight_outcome_is_recorded_on_the_result(tmp_path):
    outcome = SimpleNamespace(
        kind="tool_call", reason="ground", request_id="req-1", proposal_id=None
    )
    runtime = _runtime(tmp_path, arm="B2")
    runtime._pilot = _RecordingPilot(outcomes=(outcome,))
    runtime._airborne = True

    runtime._tick_cloud("obs-1", {})

    assert runtime.result.cloud_calls == [
        {
            "kind": "tool_call",
            "reason": "ground",
            "request_id": "req-1",
            "proposal_id": None,
        }
    ]
    assert any("cloud B2: tool_call" in line for line in runtime.result.log)
