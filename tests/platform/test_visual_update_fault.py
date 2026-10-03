"""The declared visual-update bound is evaluated, not merely carried.

``visual_update_fail_ms`` has been declared since P01-L, measured in every
flight, and built into the runtime's own ``HealthMachine`` — and until this
change the runtime never asked that machine a question. The consequence is in
retained runs: the aircraft climbed into a 2.5 m ceiling and the crash detector
disarmed it while the visual-update age was already past the bound. A stopped
visual feed lets the estimator's altitude drift, and GUIDED holds the
*estimated* position, so the aircraft moves to keep a falling number where it
was. Past the bound the estimate is no longer a pose, and the mission's declared
protective behaviour is to come down.
"""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
REAL_PLATFORM_CONFIG = REPOSITORY / "configs" / "first_indoor.yaml"


def _runtime(tmp_path, episode_id: str):
    from embodied.bench.recorder import Recorder
    from embodied.platform.localization_check import (
        _load_localization_config,
        _platform_settings,
    )
    from embodied.platform.mission_runtime import MissionRuntime

    document = _load_localization_config(REAL_PLATFORM_CONFIG)
    settings = _platform_settings(document, REPOSITORY)
    episode_dir = tmp_path / "episode"
    return MissionRuntime(
        settings=settings,
        config_document=document,
        episode_dir=episode_dir,
        evidence_dir=tmp_path / "platform",
        recorder=Recorder(episode_dir),
        instruction="Find the red block, inspect it, and return to the start.",
        target_id="red_block",
        episode_id=episode_id,
        arm="B0",
    )


def _state(*, time_ns: int, t_last_visual_ns: int, initialized: bool = True):
    from embodied.platform import localization as loc

    return loc.EstimatorState(
        time_ns=time_ns,
        initialized=initialized,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_m=(0.0, 0.0, 0.0),
        velocity_mps=(0.0, 0.0, 0.0),
        gyro_bias=(0.0, 0.0, 0.0),
        accel_bias=(0.0, 0.0, 0.0),
        sigma_pos_m=(0.1, 0.1, 0.1),
        n_tracks=4,
        t_last_visual_ns=t_last_visual_ns,
        reset_counter=0,
    )


def test_a_healthy_visual_feed_is_not_a_fault(tmp_path):
    runtime = _runtime(tmp_path, "visual-ok")
    now = 100_000_000_000
    runtime._latest_state = _state(time_ns=now, t_last_visual_ns=now - 100_000_000)
    assert runtime._visual_fault() is None


def test_no_fault_before_the_first_visual_update(tmp_path):
    """An age measured against a clock that has never ticked is not a fault."""
    runtime = _runtime(tmp_path, "visual-none")
    runtime._latest_state = _state(time_ns=10_000_000_000, t_last_visual_ns=0)
    assert runtime._visual_fault() is None


def test_no_fault_before_the_estimator_initialises(tmp_path):
    runtime = _runtime(tmp_path, "visual-uninit")
    runtime._latest_state = _state(
        time_ns=10_000_000_000, t_last_visual_ns=1_000_000_000, initialized=False
    )
    assert runtime._visual_fault() is None


def test_a_stopped_visual_feed_past_the_declared_bound_is_a_fault(tmp_path):
    runtime = _runtime(tmp_path, "visual-fail")
    fail_s = runtime._machine.bounds.visual_update_fail_s
    now = 100_000_000_000
    runtime._latest_state = _state(
        time_ns=now, t_last_visual_ns=now - int(fail_s * 1e9) - 1
    )
    reason = runtime._visual_fault()
    assert reason is not None
    assert "visual updates stopped" in reason
    assert f"{fail_s:.2f}" in reason


def test_the_bound_is_the_declared_one(tmp_path):
    """The fault is the declared 500 ms, not a number invented here."""
    runtime = _runtime(tmp_path, "visual-declared")
    assert runtime._machine.bounds.visual_update_fail_s == 0.5


def test_the_pump_stops_and_records_when_the_visual_feed_has_stopped(tmp_path):
    """The pump is the loop that runs when the mission has nothing to do."""
    runtime = _runtime(tmp_path, "visual-pump")
    now = 100_000_000_000
    runtime._latest_state = _state(time_ns=now, t_last_visual_ns=now - 5_000_000_000)
    cycles = runtime.pump_perception(lambda: None, 1.0)
    assert cycles == 0, "the bound was already passed, so no cycle should run"
    assert runtime._visual_fault_reason is not None
    assert any("visual update fault" in line for line in runtime.result.log)
