"""The mission reads the estimator only through the publisher's declared verdicts.

ESTIMATOR-DIVERGENCE.md: the feed thread hands every raw STATE to ``_on_state``
before ``publisher.offer`` sees it, and the mission's pose accessors consumed
that raw stream with none of the gates the publish path applies — so J51 flew
on, and recorded, ``here`` = (-0.22, -6.76, -1.42), a 6.9 m pose inside a 6 m
room, while the autopilot wire stayed clean (VPE max |y| <= 0.09 m in every
diverged run). These tests hold the closure: a state the publisher's verdicts
refuse never reaches ``here``, a healthy state passes unchanged, the J51 shapes
(the one-step flip and the live-feed tracked runaway) are refused while the
declared protective landing still sees the raw evidence, and the mission's gate
is the publisher's gate state for state — not stricter, not looser.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
REAL_PLATFORM_CONFIG = REPOSITORY / "configs" / "first_indoor.yaml"
HOST, CLOCK = "posegate-test-0", "monotonic"


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


def _state(
    *,
    time_ns: int,
    t_last_visual_ns: int,
    n_tracks: int = 40,
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initialized: bool = True,
):
    from embodied.platform import localization as loc

    return loc.EstimatorState(
        time_ns=time_ns,
        initialized=initialized,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_m=position_m,
        velocity_mps=(0.0, 0.0, 0.0),
        gyro_bias=(0.0, 0.0, 0.0),
        accel_bias=(0.0, 0.0, 0.0),
        sigma_pos_m=(0.1, 0.1, 0.1),
        # A healthy tracker count, for the same reason the visual-fault tests
        # use one: these states are meant to be healthy under every verdict
        # except the one the test is about.
        n_tracks=n_tracks,
        t_last_visual_ns=t_last_visual_ns,
        reset_counter=0,
    )


def _stale_visual_ns(runtime, now_ns: int) -> int:
    """A visual stamp one tick past the declared fail bound, as the wire sees it."""
    return now_ns - int(runtime._machine.bounds.visual_update_fail_s * 1e9) - 1


# ---------------------------------------------------------------------------
# (b) A healthy state passes unchanged.
# ---------------------------------------------------------------------------


def test_a_healthy_state_passes_the_accessors_unchanged(tmp_path):
    runtime = _runtime(tmp_path, "posegate-healthy")
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    now = 100_000_000_000
    odom_position = (1.0, 2.0, -1.5)
    runtime._on_state(
        _state(
            time_ns=now,
            t_last_visual_ns=now - 100_000_000,
            position_m=odom_position,
        )
    )
    expected = runtime.alignment.aligned_position_ned(odom_position)
    assert runtime._position_odom() == pytest.approx(expected)
    nav = runtime._navigation_state()
    assert nav is not None
    assert nav.pose.valid is True
    assert nav.pose.position_m == pytest.approx(expected)
    assert runtime._speed() == pytest.approx(0.0)
    assert runtime._publishable_state(runtime._latest_state) is runtime._latest_state


# ---------------------------------------------------------------------------
# (c) The J51 shapes: diverged raw numbers behind a not-publishable verdict.
# ---------------------------------------------------------------------------


def test_the_j51_flip_state_never_reaches_here(tmp_path):
    runtime = _runtime(tmp_path, "posegate-flip")
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    # J51-move-2's one-step flip, inside the collapsed window: the attitude
    # turned ~176 degrees and p jumped (0.041, 0.023, 2.448) -> (20.19, -8.52,
    # -1.26) in one publish step — after the wire had already stopped on the
    # visual-age gate. The mission kept consuming the flip as `here`; it must
    # now be refused by the same verdict that stopped the wire.
    now = 100_000_000_000
    state = _state(
        time_ns=now,
        t_last_visual_ns=_stale_visual_ns(runtime, now),
        position_m=(20.19, -8.52, -1.26),
    )
    runtime._on_state(state)
    assert runtime._position_odom() is None
    assert runtime._navigation_state() is None
    assert runtime._speed() == 0.0
    # The evidence stays: the protective landing still measures the raw newest
    # state, and the diagnosis still has its stream.
    assert runtime._latest_state is state
    reason = runtime._visual_fault()
    assert reason is not None
    assert "visual updates stopped" in reason
    assert f"{runtime._machine.bounds.visual_update_fail_s:.2f}" in reason


def test_a_live_feed_with_a_collapsed_tracker_never_reaches_here(tmp_path):
    runtime = _runtime(tmp_path, "posegate-tracks")
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    # J43-move-3's shape, which the visual-age bound cannot see: 1300 stereo
    # frames arrived and the feed's clock advanced throughout, the tracker's
    # count fell to 1, and the filter integrated 772 m of travel inside a ~6 m
    # room. The publisher refuses to carry that pose; the mission now refuses
    # to serve it, by the publisher's own verdict.
    now = 100_000_000_000
    runtime._on_state(
        _state(
            time_ns=now,
            t_last_visual_ns=now - 10_000_000,
            n_tracks=1,
            position_m=(772.0, 0.0, -1.5),
        )
    )
    assert runtime._position_odom() is None
    assert runtime._navigation_state() is None


def test_the_capture_pose_refuses_a_refused_ring_state(tmp_path):
    runtime = _runtime(tmp_path, "posegate-capture")
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    now = 100_000_000_000
    flip = _state(
        time_ns=now,
        t_last_visual_ns=_stale_visual_ns(runtime, now),
        position_m=(20.19, -8.52, -1.26),
    )
    runtime._on_state(flip)

    def record_at(sim_time_s: float):
        return SimpleNamespace(
            sim_time_s=sim_time_s,
            pair=SimpleNamespace(capture_host_ns=123_456_789),
        )

    # The pose nearest the capture instant is the flip, and the verdicts refuse
    # it: no pose reaches the depth product, exactly the existing not-current
    # path, never the raw diverged numbers.
    assert runtime._capture_pose(record_at(now / 1e9)) is None
    # A publishable state beside it is still served, unchanged.
    healthy = _state(
        time_ns=now - 1,
        t_last_visual_ns=now - 11_000_000,
        position_m=(0.5, 0.5, -1.5),
    )
    runtime._on_state(healthy)
    pose = runtime._capture_pose(record_at((now - 1) / 1e9))
    assert pose is not None
    assert pose.position_m == pytest.approx(
        runtime.alignment.aligned_position_ned((0.5, 0.5, -1.5))
    )


# ---------------------------------------------------------------------------
# The gate is the publisher's verdict chain, state for state.
# ---------------------------------------------------------------------------


def test_the_gate_is_the_publishers_verdict_chain_state_for_state(tmp_path):
    """For every state, the mission refuses exactly what ``state_for_publish``

    refuses once the offer is fresh: the runtime shares the publisher's
    ``HealthMachine``, and the gate mirrors its two verdict conditions — the
    same never-ticked-clock exemption on the visual age, the same
    initialized-only condition on the tracking verdict. Nothing stricter (a
    mission-side bound the wire lacks), nothing looser (a pose the wire would
    refuse reaching the mission).
    """
    from embodied.platform import localization as loc

    runtime = _runtime(tmp_path, "posegate-parity")
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    publisher = loc.ExternalNavPublisher(
        "tcp://127.0.0.1:0",  # never connected; state_for_publish is pure
        runtime.alignment,
        runtime._machine,
        clock=lambda: 0.0,
    )
    now = 100_000_000_000
    cases = [
        # Healthy: fresh visual update, tracker above the floor.
        _state(time_ns=now, t_last_visual_ns=now - 10_000_000),
        # The declared visual-update failure: J51's flip window.
        _state(
            time_ns=now + 1,
            t_last_visual_ns=_stale_visual_ns(runtime, now + 1),
            position_m=(20.19, -8.52, -1.26),
        ),
        # The declared tracking failure: the live-feed runaway's count.
        _state(
            time_ns=now + 2,
            t_last_visual_ns=now + 2 - 10_000_000,
            n_tracks=1,
            position_m=(772.0, 0.0, -1.5),
        ),
        # Before initialization neither verdict applies, on the wire or here.
        _state(time_ns=now + 3, t_last_visual_ns=0, initialized=False, n_tracks=0),
        # Before the first visual update the age is not measurable.
        _state(time_ns=now + 4, t_last_visual_ns=0, n_tracks=0),
    ]
    for state in cases:
        runtime._latest_state = state
        publisher.offer(state, state.time_ns)
        wire = publisher.state_for_publish(0.0)
        mission = runtime._publishable_state(state)
        assert (mission is None) == (wire is None), f"state at {state.time_ns}"
        if mission is not None:
            assert mission is state  # served states pass through, never rewritten


# ---------------------------------------------------------------------------
# (a) Admission refuses on a gated pose.
# ---------------------------------------------------------------------------


def _grounded_start_runtime(tmp_path):
    """A runtime whose `start` target resolves, from one integrated observation.

    The scaffolding is the integration suite's offline surface: the alignment
    seals, one depth product integrates, the observation is citable — so a
    later admission refusal can only be the navigation state, never an
    ungrounded target.
    """
    from embodied.contracts.records import ClockStamp, PoseEstimate, SpatialGoal
    from embodied.perception.camera import DepthProduct, PoseProvenance

    runtime = _runtime(tmp_path, "posegate-admission")
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    height, width = 480, 640
    valid = np.zeros((height, width), dtype=bool)
    valid[200:280, 280:360] = True
    product = DepthProduct(
        calibration_id=runtime.calibration.calibration_id,
        calibration_version=runtime.calibration.version,
        pair_id="pair-1",
        capture_stamp=ClockStamp(host_id=HOST, clock_id=CLOCK, monotonic_ns=1),
        receipt_stamp=ClockStamp(host_id=HOST, clock_id=CLOCK, monotonic_ns=2),
        sim_time_s=0.5,
        pose_provenance=PoseProvenance(label="SENSOR_DERIVED", detail="test"),
        frame="camera_optical",
        disparity_px=np.full((height, width), 50.0, dtype=np.float32),
        depth_m=np.full((height, width), 2.0, dtype=np.float32),
        valid=valid,
        reasons=np.zeros((height, width), dtype=np.uint8),
        uncertainty_m=np.full((height, width), 0.02, dtype=np.float32),
    )
    pose = PoseEstimate(
        parent_frame="odom",
        child_frame="body",
        stamp=ClockStamp(host_id=HOST, clock_id=CLOCK, monotonic_ns=3),
        position_m=(0.0, 0.0, -1.5),  # NED: 1.5 m above the spawn
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance=None,
        nav_epoch=runtime.nav_epoch,
        source_ids=("ov_stream",),
        valid=True,
    )
    runtime._capture_clock_ns = 500_000_000
    runtime.store.integrate(
        product,
        pose,
        runtime.calibration,
        stamp_ns=500_000_000,
        observation_id="obs-1",
        now_ns=500_000_000,
    )
    runtime._last_observation_id = "obs-1"
    goal = SpatialGoal(
        proposal_id="p-start",
        request_id=None,
        fingerprint="fp",
        mission_revision=0,
        base_goal_revision=0,
        selection_ids=(),
        target_refs=("start",),
        intent="return",
        constraints=(),
        completion_condition="settled at the start position",
        lease_bounds=(("step_lease_s", 30.0),),
        local_discretion_bounds=(),
    )
    assert runtime.resolve_targets(goal), "the start target must ground"
    return runtime, goal


def test_admission_refuses_when_the_pose_gate_refuses(tmp_path):
    from embodied.contracts.records import GoalDisposition
    from embodied.platform.mission_runtime import _LiveAdmission

    runtime, goal = _grounded_start_runtime(tmp_path)
    now = 100_000_000_000
    runtime._on_state(
        _state(
            time_ns=now,
            t_last_visual_ns=_stale_visual_ns(runtime, now),
            position_m=(20.19, -8.52, -1.26),
        )
    )
    status = _LiveAdmission(runtime).admit(goal, context=None)
    assert status.disposition is GoalDisposition.REJECTED
    assert status.reason.startswith("no_navigation_state")


def test_admission_with_a_healthy_pose_is_not_refused_for_lack_of_one(tmp_path):
    """The over-blocking direction: a publishable state must still admit.

    Whatever the executor then decides about the route, the refusal may not be
    the pose gate's — a gate that refused everything would ground the mission
    as surely as the leak did.
    """
    from embodied.platform.mission_runtime import _LiveAdmission

    runtime, goal = _grounded_start_runtime(tmp_path)
    now = 100_000_000_000
    runtime._on_state(
        _state(time_ns=now, t_last_visual_ns=now - 100_000_000, n_tracks=40)
    )
    status = _LiveAdmission(runtime).admit(goal, context=None)
    assert not status.reason.startswith("no_navigation_state")
