"""The observation sweep's hold is a velocity hold (XY-ROTATION-CRASH.md).

The captures settled the old story: the sweep's commanded yaw-ANGLE bursts —
not a lying wire — excited the yaw snap-overshoot cascade that rolled both
aircraft inverted (J56 27.0°/29.5° and J57 16.5°/15.6° steps; the wire was
honest to ≤ 0.068 m when the rotation blinded the tracker), and a position
pin — fixed or re-pinned — left Guided a displacement to chase from a
possibly-blind estimate. For the sweep's declared 10 s window the target is
therefore zero velocity plus the declared 0.6 rad/s yaw RATE, with position
and yaw angle absent: a rate stream is step-free even at burst cadence, and
with no position field there is nothing to chase. The non-sweep renewal hold
keeps its fixed declared hover altitude (J52's own fix; it must never take
the velocity branch).

The tests drive the runtime's real publication machinery on injected states
and an injected simulator clock (T18): only the platform boundary
(``send_local_ned``) is a recording stub, and the estimate enters through the
runtime's own pose gate via ``_on_state``.
"""

from __future__ import annotations

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
            yaw_rate_rad_s=target.yaw_rate_rad_s,
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
# vision resumption, then a whole-metre-class jump. Under the velocity hold
# every publication is identical whatever the estimate does — the wander can
# reach the wire as nothing, because the wire carries no position to correct.
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
# The sweep's hold is a velocity hold; nothing on the wire tracks the estimate.
# ---------------------------------------------------------------------------


def test_the_sweeps_published_holds_are_velocity_holds(tmp_path):
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

    # The wander is fed as raw estimator positions, jumps and all. Every
    # publication is the SAME velocity hold anyway: zero velocity, no
    # position, no yaw angle, the declared rate — Guided is never handed a
    # displacement to chase, whatever the estimate did between ticks.
    shapes = {
        (
            target.position_ned,
            tuple(target.velocity_ned),
            target.yaw_rad,
            target.yaw_rate_rad_s,
            target.certificate_ref,
        )
        for target in platform.targets
    }
    assert shapes == {
        (None, (0.0, 0.0, 0.0), None, MR.OBSERVATION_YAW_RATE_RAD_S, "observe-1")
    }
    # The goal's own hold is never mutated: with no position field on the
    # wire, a refused publication cannot invent a pin either.
    assert runtime._active_goal.hold_position_odom == WANDER[0]


def test_the_sweeps_yaw_command_is_a_rate_stream_not_an_angle_ramp(tmp_path):
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

    # The captures' burst cadence (0.3–0.8 s between sweep ticks) cannot step
    # the yaw any more: the command carries the declared RATE, not an angle,
    # so every tick — inside the window and past it — is identical, and the
    # old angle ramp (with its 10 s clamp) is gone from the wire.
    assert [target.yaw_rad for target in platform.targets] == [None] * len(sim_times)
    assert all(
        target.yaw_rate_rad_s == pytest.approx(MR.OBSERVATION_YAW_RATE_RAD_S)
        for target in platform.targets
    )
    assert platform.targets[0].yaw_rate_rad_s == platform.targets[-1].yaw_rate_rad_s


def test_a_sweep_publication_with_no_fresh_estimate_still_coasts(tmp_path):
    """A tick whose pose the publisher's gate refuses invents nothing.

    The velocity hold does not read the estimate at all: with no position
    field to pin and no angle to ramp, a refused pose changes nothing on the
    wire — the aircraft coasts on the velocity loop through the blind window
    (both captures flipped 0.48 s after their last publication, in holds that
    DID depend on the estimate). The visual-fault machinery still ends the
    sweep; the publication itself must not fail.
    """
    runtime = _runtime(tmp_path, "sweep-hold-nopose")
    platform = _RecordingPlatform()
    runtime._platform = platform
    runtime._active_goal = _sweep_goal((0.1, 0.2, -1.5), hold_yaw_rad=0.0)
    runtime._position_odom = lambda: None
    runtime._stats.sim_clock.observe(TICK_SIM_S)
    assert runtime.publish_active() is None
    assert platform.targets[0].position_ned is None
    assert tuple(platform.targets[0].velocity_ned) == (0.0, 0.0, 0.0)
    assert platform.targets[0].yaw_rad is None
    assert platform.targets[0].yaw_rate_rad_s == MR.OBSERVATION_YAW_RATE_RAD_S
    # The commanded record is a faithful image of the wire: absent fields
    # recorded as null, not as a pinned coordinate the wire never carried.
    recorded = runtime._commanded[-1]
    assert recorded["position_m"] is None
    assert recorded["velocity_mps"] == [0.0, 0.0, 0.0]
    assert recorded["yaw_rad"] is None
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
    # And the wire shape is exactly what J52 flew: fixed position, zero
    # velocity, an angle rather than a rate — no velocity-hold fields.
    assert all(
        tuple(target.velocity_ned) == (0.0, 0.0, 0.0) and target.yaw_rate_rad_s is None
        for target in platform.targets
    )
