"""The observation sweep's hold is self-neutralizing (Z-CLIMB.md; J55 verdict).

During the §12.2 sweep the VIO estimate wanders under the declared 0.6 rad/s
rotation. A hold captured once at the observe call converts that wander into a
full-authority chase — J53 (AngErr 105) and J55 (AngErr 83) crashed in it;
J54 survived by luck of the wander. The fix: for the sweep's declared 10 s
window the hold re-pins to the CURRENT aligned position at every publication
(own x, y, z), so no fixed target exists to chase — the commanded displacement
against the live estimate is identically zero, structurally, not merely
smaller. The yaw ramp is untouched, and the non-sweep renewal hold keeps its
fixed declared hover altitude (J52's own fix; it must never re-pin).

The tests drive the runtime's real publication machinery on injected states
and an injected simulator clock (T18): only the platform boundary
(``send_local_ned``) is a recording stub, and the estimate enters through the
runtime's own pose gate via ``_on_state``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from embodied.platform import mission_runtime as MR

REPOSITORY = Path(__file__).resolve().parents[2]
REAL_PLATFORM_CONFIG = REPOSITORY / "configs" / "first_indoor.yaml"
HOST, CLOCK = "sweep-hold-test-0", "monotonic"


def _runtime(tmp_path, episode_id: str) -> "MR.MissionRuntime":
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


def _state(position_m, time_ns: int):
    """A healthy initialized estimator state, in the aligned frame by identity."""
    from embodied.platform import localization as loc

    return loc.EstimatorState(
        time_ns=time_ns,
        initialized=True,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_m=position_m,
        velocity_mps=(0.0, 0.0, 0.0),
        gyro_bias=(0.0, 0.0, 0.0),
        accel_bias=(0.0, 0.0, 0.0),
        sigma_pos_m=(0.1, 0.1, 0.1),
        # A healthy tracker count and a fresh visual stamp, so every verdict
        # except the one under test passes (the pose gate shares the publisher's
        # bounds; these states are meant to be servable poses).
        n_tracks=40,
        t_last_visual_ns=time_ns - 50_000_000,
        reset_counter=0,
    )


class _RecordingPlatform:
    """The one boundary ``publish_active`` speaks through, recording each target."""

    def __init__(self) -> None:
        self.targets: list[Any] = []
        self._sequence = 0

    def send_local_ned(self, target):
        from embodied.contracts.records import (
            ClockStamp,
            Frame,
            MotionSetpoint,
            MotionTarget,
        )
        from embodied.platform.webots_ardupilot import (
            SetpointPublication,
            SetpointSource,
            mask_for_target,
        )

        self.targets.append(target)
        motion = MotionTarget(
            position_ned=target.position_ned,
            velocity_ned=target.velocity_ned,
            acceleration_ned=None,
            yaw_rad=target.yaw_rad,
            yaw_rate_rad_s=None,
        )
        self._sequence += 1
        setpoint = MotionSetpoint(
            command_sequence=self._sequence,
            mission_revision=0,
            goal_revision=0,
            nav_epoch="sweep-hold-test",
            frame=Frame.ODOM,
            type_mask=mask_for_target(motion),
            target=motion,
            issue_stamp=ClockStamp(
                host_id=HOST, clock_id=CLOCK, monotonic_ns=self._sequence * 1_000_000
            ),
            deadline_s=target.deadline_s,
            certificate_ref=target.certificate_ref,
            sample_ref=None,
            source=SetpointSource.NORMAL,
        )
        return SetpointPublication(
            setpoint=setpoint,
            published_stamp=ClockStamp(
                host_id=HOST,
                clock_id=CLOCK,
                monotonic_ns=self._sequence * 1_000_000 + 1_000,
            ),
        )


def _sweep_goal(hold_position_odom, *, hold_yaw_rad: float) -> MR._ActiveGoal:
    """The goal ``observe_in_place`` installs, with the declared rate and window."""
    return MR._ActiveGoal(
        goal_id="observe-1",
        hold_position_odom=hold_position_odom,
        hold_yaw_rad=hold_yaw_rad,
        hold_yaw_rate_rad_s=MR.OBSERVATION_YAW_RATE_RAD_S,
        hold_since_sim_s=0.0,
    )


# J53's measured wander shape: decimetre drift, then 0.5-0.9 m jumps at each
# vision resumption, then a whole-metre-class jump. Under the old fixed hold
# the max commanded error over this trace is the largest jump; under the
# re-pinned hold it must be exactly zero.
WANDER = (
    (0.00, 0.00, -1.58),
    (0.52, -0.31, -1.10),
    (-0.44, 0.83, -2.31),
    (0.97, -1.24, -0.83),
    (3.10, 2.20, -4.50),
    (-1.05, 0.44, -1.02),
)
TICK_SIM_S = 0.1


# ---------------------------------------------------------------------------
# The sweep's hold tracks the live estimate; the chase cannot exist.
# ---------------------------------------------------------------------------


def test_the_sweeps_published_holds_track_the_live_estimate(tmp_path):
    runtime = _runtime(tmp_path, "sweep-hold-track")
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    platform = _RecordingPlatform()
    runtime._platform = platform
    runtime._active_goal = _sweep_goal(WANDER[0], hold_yaw_rad=0.4)

    for index, estimate in enumerate(WANDER):
        sim_s = (index + 1) * TICK_SIM_S
        runtime._on_state(_state(estimate, time_ns=int(sim_s * 1e9)))
        runtime._stats.sim_clock.observe(sim_s)
        assert runtime.publish_active() is None

    # The wander is fed as raw estimator positions; the hold lives in the
    # aligned odom frame the wire speaks, through the same frozen alignment.
    aligned_wander = [
        tuple(runtime.alignment.aligned_position_ned(estimate)) for estimate in WANDER
    ]
    # Every publication commanded exactly the estimate it was published with:
    # the max commanded step over the whole wander trace is zero, so whatever
    # the estimate did between ticks, Guided had no displacement to chase.
    commanded_error = max(
        math.dist(target.position_ned, aligned)
        for target, aligned in zip(platform.targets, aligned_wander)
    )
    assert commanded_error == pytest.approx(0.0, abs=1e-12)
    assert [tuple(target.position_ned) for target in platform.targets] == aligned_wander
    # The pin is the goal's own durable state, so a refused publication would
    # still record what the aircraft was actually holding.
    assert runtime._active_goal.hold_position_odom == pytest.approx(aligned_wander[-1])
    # A hold commands no velocity; its certificate reference is the goal itself.
    assert all(
        tuple(target.velocity_ned) == (0.0, 0.0, 0.0) for target in platform.targets
    )
    assert all(target.certificate_ref == "observe-1" for target in platform.targets)


def test_the_sweeps_yaw_ramp_is_unchanged_including_its_clamp(tmp_path):
    runtime = _runtime(tmp_path, "sweep-hold-yaw")
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    platform = _RecordingPlatform()
    runtime._platform = platform
    runtime._active_goal = _sweep_goal(WANDER[0], hold_yaw_rad=0.4)

    sim_times = (0.1, 0.2, 0.3, 10.0, 10.3)
    for index, sim_s in enumerate(sim_times):
        runtime._on_state(_state(WANDER[index % len(WANDER)], time_ns=int(sim_s * 1e9)))
        runtime._stats.sim_clock.observe(sim_s)
        assert runtime.publish_active() is None

    expected = tuple(
        0.4 + MR.OBSERVATION_YAW_RATE_RAD_S * min(t, 10.0) for t in sim_times
    )
    assert [target.yaw_rad for target in platform.targets] == pytest.approx(expected)
    # Past the declared window the swept angle stops: the clamp is the declared
    # window, measured on the simulator clock, unchanged by the re-pinning.
    assert platform.targets[-1].yaw_rad == pytest.approx(0.4 + 6.0)


def test_a_sweep_publication_with_no_fresh_estimate_keeps_the_last_pin(tmp_path):
    """A tick whose pose the publisher's gate refuses invents nothing.

    The honest behavior is the last pinned position — the same one the wire
    already holds — never a guessed coordinate. The visual-fault machinery
    ends the sweep on this condition; the publication itself must not fail.
    """
    runtime = _runtime(tmp_path, "sweep-hold-nopose")
    platform = _RecordingPlatform()
    runtime._platform = platform
    runtime._active_goal = _sweep_goal((0.1, 0.2, -1.5), hold_yaw_rad=0.0)
    runtime._position_odom = lambda: None
    runtime._stats.sim_clock.observe(TICK_SIM_S)
    assert runtime.publish_active() is None
    assert tuple(platform.targets[0].position_ned) == (0.1, 0.2, -1.5)
    assert runtime._active_goal.hold_position_odom == (0.1, 0.2, -1.5)


# ---------------------------------------------------------------------------
# The non-sweep renewal hold keeps its fixed declared hover altitude.
# ---------------------------------------------------------------------------


class _StubAlignment:
    """A sealed alignment whose origin is the test's own (vantage-policy shape)."""

    def __init__(self, origin):
        self.sealed = True
        self._origin = origin

    def aligned_position_ned(self, point):
        return self._origin


def test_the_renewal_hold_keeps_its_fixed_hover_altitude(tmp_path):
    """J52's fix is NOT reverted: a rate-0 hold never re-pins.

    The renewal hold's target is the declared hover altitude captured at the
    refusal; even with the estimate wandering beneath it, every publication
    sends exactly that fixed target. Re-pinning here would re-open J52's
    park-and-disarm.
    """
    runtime = _runtime(tmp_path, "sweep-hold-renewal")
    platform = _RecordingPlatform()
    runtime._platform = platform
    runtime.alignment = _StubAlignment((-1.0, 0.0, -0.09))
    hold = runtime._renewal_hold_position((0.17, 0.31, -0.15))
    assert hold == (0.17, 0.31, -0.09 - 1.5)

    goal = MR._ActiveGoal(
        goal_id="cert-explore-frontier:8:7:2", hold_position_odom=hold
    )
    runtime._active_goal = goal
    wandering = iter([(5.0, 5.0, -0.2), (6.0, 4.0, -0.5), (7.0, 3.0, -0.9)])
    runtime._position_odom = lambda: next(wandering, (7.0, 3.0, -0.9))
    for index, sim_s in enumerate((0.1, 0.2, 0.3)):
        runtime._stats.sim_clock.observe(sim_s)
        assert runtime.publish_active() is None

    assert [tuple(target.position_ned) for target in platform.targets] == [hold] * 3
    assert goal.hold_position_odom == hold
    # A zero rate is the mission's declared hold yaw — no ramp on a renewal hold.
    assert all(
        target.yaw_rad == pytest.approx(MR.MISSION_YAW_HOLD_RAD)
        for target in platform.targets
    )
