"""The first offered frontier must admit: a certified setpoint, honestly swept.

The night this pins (2026-10-03, night/motion): the mission chain —
navigable_frontiers → resolve_targets → executor.admit → planner.plan — refused
every one of the eight offered frontiers at the certification axis on this real
carved doorway map ("segment-1 sweeps cell (75, 38, 14) ... the clamped fallback
needs 39.9 s, beyond the declared 30.0 s bound"). The mechanism was not the map
and not a declared bound: the sweep test checked membership in the occupancy set
(cells whose own envelope is free) instead of the published free cells, which
applies the declared envelope a second time — a ~0.95 m raw-free corridor against
the declared 0.475 m clearance — and the doubled corridor cost the route its
shortcut, leaving a 26-knot staircase whose honest clamped-fallback durations
overran the 30 s route bound. Both tests here fail on that code: the first
because admission refuses, the second because it re-verifies the certificate's
sweep against the published free set with the declared ball itself.

Fixture, store and runtime follow tests/platform/test_frontier_excursion.py.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from embodied.contracts import records as R
from embodied.navigation import executor as EX
from embodied.navigation import geometry as GE
from embodied.navigation import planner as PL
from embodied.perception import grounding as grounding_module
from embodied.platform.mission_runtime import ENVELOPE, PLAN_CONFIG, MissionRuntime

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "navigation" / "doorway"


def _fixture_module():
    spec = importlib.util.spec_from_file_location(
        "p11_doorway_generate", FIXTURE / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FIXTURE = _fixture_module()
_SCENE = json.loads((FIXTURE / "truth" / "scene.json").read_text(encoding="utf-8"))


def _store():
    """The doorway fixture's own map: real carving, real free space."""
    from embodied.memory import world as world_module

    store = world_module.MapStore(
        world_module.MapConfig.from_scene(_SCENE),
        submap_id="submap-first-setpoint",
        nav_epoch=_SCENE["nav_epoch"],
    )
    surfaces = _FIXTURE.variant_surfaces("nominal")
    for index in range(len(_FIXTURE.VIEWS)):
        store.integrate(
            _FIXTURE.declared_depth_product(index, surfaces),
            _FIXTURE.capture_pose_record(index),
            _FIXTURE.calibration(),
            stamp_ns=_FIXTURE.VIEWS[index]["capture_ns"],
            observation_id=_FIXTURE.VIEWS[index]["observation_id"],
        )
    return store


STORE = _store()
HERE = tuple(float(v) for v in _FIXTURE.VIEWS[0]["body_position_odom_m"])
NOW_NS = 2_000_000_000


class _Runtime(MissionRuntime):
    """The runtime's map half, without the platform configuration."""

    def __init__(self, store, here, now_ns):
        self.store = store
        self._here = here
        self._now = now_ns
        self.result = type("Res", (), {"log": []})()
        self._grounded = {}
        self._candidate_observation_ids = []
        self._last_observation_id = "obs-2"
        self._observation_counter = 2
        # The mission frame: the aligned origin sits on the fixture's floor
        # (its deepest occupied cell centre), so the declared hover altitude
        # (configs/first_indoor.yaml probe.hover_altitude_m) is expressible
        # and the flyable band is stated -- the admission chain under test now
        # runs exactly as the live mission's does, band gate included.
        self.alignment = type(
            "A",
            (),
            {
                "sealed": True,
                "aligned_position_ned": staticmethod(
                    lambda point: (0.0, 0.0, 2.95)
                ),
            },
        )()
        self.settings = type("S", (), {"hover_altitude_m": 1.5})()

    def _position_odom(self):
        return self._here

    def _now_ns(self):
        return self._now


def _navigation_state():
    stamp = R.ClockStamp(host_id="test", clock_id="monotonic", monotonic_ns=NOW_NS)
    pose = R.PoseEstimate(
        parent_frame="odom",
        child_frame="body",
        stamp=stamp,
        position_m=HERE,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance=None,
        nav_epoch=STORE.config.nav_epoch
        if hasattr(STORE.config, "nav_epoch")
        else "probe-epoch",
        source_ids=("obs-2",),
        valid=True,
    )
    return R.NavigationState(
        state_sequence=1,
        pose=pose,
        velocity_mps=(0.0, 0.0, 0.0),
        covariance=None,
        nav_epoch=pose.nav_epoch,
        visual_source_ids=("obs-2",),
        imu_source_ids=(),
        status="ok",
        controller_alignment_id=None,
    )


def _admit_first_offered():
    runtime = _Runtime(STORE, HERE, NOW_NS)
    offered = runtime.navigable_frontiers()
    assert offered, "the carved doorway map must offer a frontier"
    ref = offered[0]
    proposal = R.SpatialGoal(
        proposal_id=f"explore-{ref}-first-setpoint",
        request_id=None,
        fingerprint="test",
        mission_revision=1,
        base_goal_revision=0,
        selection_ids=(),
        target_refs=(ref,),
        intent="explore",
        constraints=(),
        completion_condition="reached",
        lease_bounds=(("step_lease_s", 60.0),),
        local_discretion_bounds=(),
    )
    targets = runtime.resolve_targets(proposal)
    assert targets, f"the offered frontier {ref} must resolve to a grounded target"
    result = EX.admit(
        proposal,
        targets,
        STORE,
        _navigation_state(),
        ENVELOPE,
        PLAN_CONFIG,
        mission_revision=1,
        snapshot_id=STORE.snapshot_id,
        now_ns=NOW_NS,
        pose_validity_s=grounding_module.POSE_VALIDITY_S,
    )
    return ref, targets[0], result


def test_the_first_offered_frontier_admits_with_a_certified_setpoint():
    """Admission's verdict on the head of the offered list, end to end.

    Before the 2026-10-03 fix this exact call returned
    ``computation_limit: segment-1 sweeps cell (75, 38, 14) ...`` for every
    offered frontier, and the runner burned its attempts without translating.
    """
    ref, _, result = _admit_first_offered()
    assert result.accepted is not None, (
        f"the first offered frontier {ref} must admit; got: {result.status.reason}"
    )
    certificate = result.certificate
    assert certificate is not None and certificate.certified is True
    assert certificate.horizon_s <= PLAN_CONFIG.max_duration_s, (
        f"{certificate.horizon_s:.1f} s leaves the declared "
        f"{PLAN_CONFIG.max_duration_s:.1f} s route bound"
    )
    first = certificate.sample(certificate.t_start_s)[0]
    last = certificate.sample(certificate.t_end_s)[0]
    displacement = float(np.linalg.norm(np.asarray(last) - np.asarray(first)))
    assert displacement > ENVELOPE.inflation_m, (
        f"an admitted explore goal must translate; the certificate moves "
        f"{displacement:.3f} m"
    )
    setpoints = EX.publish_prefix(
        certificate,
        _navigation_state(),
        PLAN_CONFIG,
        goal_revision=certificate.goal_revision,
        mission_revision=1,
        issue_stamp=R.ClockStamp(
            host_id="test", clock_id="monotonic", monotonic_ns=NOW_NS
        ),
    )
    assert setpoints, "an admitted goal must yield a setpoint prefix to publish"
    assert setpoints[0].certificate_ref == certificate.certificate_id
    # The first setpoint is the certified start, not the curve's end: the
    # certificate's window is absolute and the publisher samples it at now_ns,
    # so a time-base confusion here would command the goal point itself.
    from embodied.platform.webots_ardupilot import enu_to_ned

    start_ned = np.asarray(enu_to_ned(certificate.start_position_odom_m))
    first_ned = np.asarray(setpoints[0].target.position_ned)
    stride = PLAN_CONFIG.limits.v_max_mps * PLAN_CONFIG.setpoint_period_s
    assert float(np.linalg.norm(first_ned - start_ned)) <= 2.0 * stride + 1e-9, (
        f"the first setpoint sits {np.linalg.norm(first_ned - start_ned):.3f} m "
        f"from the certified start (declared stride {stride:.3f} m)"
    )


def test_the_certificate_time_axis_is_contiguous_and_honest():
    """The segments tile [0, horizon]; the published window is the curve's own.

    Two defects this pins, both measured on this map (2026-10-03, night/motion):
    the certification sweep once wrote its sample time over the knot clock, so
    every segment's start drifted by ``(samples-1)/samples`` of its predecessor's
    duration and mid-plan samples clamped to the curve's end — a live publisher
    would have commanded the goal point partway through the plan; and the
    fallback's certificate was stamped with the durations of the minimum-jerk
    form it replaced, publishing a window longer than the curve it certifies.
    """
    _, _, result = _admit_first_offered()
    assert result.accepted is not None
    certificate = result.certificate
    clock = 0.0
    for segment in certificate.segments:
        assert abs(segment.t_start_s - clock) <= 1e-9, (
            f"segment {segment.index} starts at {segment.t_start_s:.3f} s but the "
            f"previous segment ends at {clock:.3f} s"
        )
        clock = segment.t_end_s
    assert abs(clock - certificate.horizon_s) <= 1e-9, (
        f"the segments end at {clock:.3f} s but the certificate's horizon is "
        f"{certificate.horizon_s:.3f} s"
    )
    assert (
        abs(certificate.t_end_s - certificate.t_start_s - certificate.horizon_s) <= 1e-9
    )
    # A mid-plan sample is the curve at that time, not the clamped endpoint.
    mid = np.asarray(
        certificate.sample(certificate.t_start_s + certificate.horizon_s / 2.0)[0]
    )
    end = np.asarray(certificate.sample(certificate.t_end_s)[0])
    assert float(np.linalg.norm(mid - end)) > 0.1, (
        "the mid-plan sample equals the curve's end: the time axis is not tiled"
    )


def test_the_certificate_sweeps_published_free_space_under_the_declared_ball():
    """The sweep claim, re-derived from the map rather than trusted from the planner.

    Every cell inside the declared ball — the envelope plus the certificate's own
    quarter voxel — around each sampled point of the published curve must be
    published free. This is the membership the certification is supposed to test;
    pinning it here means a future relaxation of that membership cannot pass this
    file while the claim is false, and the boundary cells the doubled corridor
    once refused (published free, but not envelope-OK) stay honestly swept.
    """
    _, _, result = _admit_first_offered()
    assert result.accepted is not None
    certificate = result.certificate
    published_free = STORE.free_cells(now_ns=NOW_NS)
    grid = STORE.config
    radius = ENVELOPE.inflation_m + grid.voxel_m * PL.CERTIFICATE_MARGIN_VOXELS
    offsets = GE.ball_offsets(grid, radius)
    interval = PL._sample_interval(grid, PLAN_CONFIG)
    t = certificate.t_start_s
    checked = 0
    while t <= certificate.t_end_s + 1e-12:
        position, _, _ = certificate.sample(t)
        index = grid.cell_index(tuple(float(value) for value in position))
        for offset in offsets:
            cell = (index[0] + offset[0], index[1] + offset[1], index[2] + offset[2])
            assert grid.inside(cell) and cell in published_free, (
                f"the certified curve sweeps {cell} at t={t:.2f}s, which is not "
                "published free space"
            )
        checked += 1
        t += interval
    assert checked > 0
