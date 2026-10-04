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
        # A healthy tracker count, not an arbitrary one. These states are meant to
        # be healthy, and the declared tracking floor now gives that meaning: a
        # gate-passing P01-L flight tracks 44-59, and the runs that collapse read
        # 0 or 1 (measurements cited in configs/first_indoor.yaml beside
        # tracking_lost_min_tracks). 4 was arbitrary and is below the floor, so it
        # would have made every "healthy" case here a tracking fault.
        n_tracks=40,
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


# ---------------------------------------------------------------------------
# The tracker's own floor. The age bounds above cannot see an estimate that is
# wrong while the feed is alive, which is the measured shape of the runaway: on
# J43-move-3 1300 stereo frames arrived and the feed's clock advanced throughout,
# the tracker's count fell to 1, and the filter integrated 772 m of travel inside
# a ~6 m room before the crash detector ended the run inverted.
# ---------------------------------------------------------------------------


def _state_with_tracks(
    *,
    n_tracks: int,
    time_ns: int = 100_000_000_000,
    visual_age_ns: int = 100_000_000,
):
    from embodied.platform import localization as loc

    return loc.EstimatorState(
        time_ns=time_ns,
        initialized=True,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_m=(0.0, 0.0, 0.0),
        velocity_mps=(0.0, 0.0, 0.0),
        gyro_bias=(0.0, 0.0, 0.0),
        accel_bias=(0.0, 0.0, 0.0),
        sigma_pos_m=(0.1, 0.1, 0.1),
        n_tracks=n_tracks,
        t_last_visual_ns=time_ns - visual_age_ns,
        reset_counter=0,
    )


def test_a_healthy_tracker_is_not_a_fault(tmp_path):
    runtime = _runtime(tmp_path, "tracks-ok")
    runtime._latest_state = _state_with_tracks(n_tracks=40)
    assert runtime._visual_fault() is None


def test_a_collapsed_tracker_field_is_a_diagnostic_not_a_bound(tmp_path):
    """The wire field cannot gate a run, and this is the measurement.

    ``n_tracks`` is filled from ``get_active_tracks``, which hands out the pin's
    ``active_tracks_uvd`` — written only by ``retriangulate_active_tracks``,
    keeping just the tracks that triangulate with positive depth and project
    inside the image. It reads 0 whenever triangulation yields nothing, which is
    exactly the stationary case a mission begins in. Measured on ``J48-fly-1``:
    it read 0 on every summary while the feature database held 88 to 93 features
    and the images were healthy (``left_sd`` 21.09), and a floor of 5 on it ended
    five runs. The pin's own encoder records the count as a diagnostic.
    """
    runtime = _runtime(tmp_path, "tracks-dead")
    runtime._latest_state = _state_with_tracks(n_tracks=0)
    assert runtime._visual_fault() is None, "a diagnostic must not end a run"
    assert any("tracker diagnostic" in line for line in runtime.result.log)


def test_the_declared_tracking_floor_is_no_longer_consulted(tmp_path):
    """The declared value stays for schema stability and is no longer a gate.

    An unused declared bound is the trap this project has closed twice, so it is
    named here rather than left for the next reader to find.
    """
    runtime = _runtime(tmp_path, "tracks-declared")
    assert runtime._machine.bounds.tracking_lost_min_tracks == 5
    runtime._latest_state = _state_with_tracks(n_tracks=1)
    assert runtime._visual_fault() is None


def test_the_transmission_path_is_guarded_by_the_visual_age_not_the_tracker(tmp_path):
    """A feed that stops carrying usable updates is the fault that was declared."""
    runtime = _runtime(tmp_path, "tracks-publish")
    now_ns = 100_000_000_000

    # A live feed with a collapsed tracker is not a fault: the field is a
    # diagnostic, so no run is ended on it.
    runtime._on_state(_state_with_tracks(n_tracks=1, time_ns=now_ns))
    assert runtime._visual_fault() is None

    # The declared guard still fires: a newest estimate whose last visual update
    # is past the bound. The tracker count is irrelevant to it.
    past_ns = int(runtime._machine.bounds.visual_update_fail_s * 1e9) + 500_000_000
    runtime._on_state(
        _state_with_tracks(n_tracks=40, time_ns=now_ns, visual_age_ns=past_ns)
    )
    fault = runtime._visual_fault()
    assert fault is not None, "a feed past its declared visual-update bound must be a fault"
    assert "visual updates stopped" in fault, fault
