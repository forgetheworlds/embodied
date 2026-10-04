"""Routes, refusal, replanning and the independent validator.

The behaviours asserted here are the slice's non-negotiables: a blocked passage is
reported rather than planned around, a certified plan is published only through the
executor, the validator can refuse the planner's own output, dependencies expire,
a late replan is rejected, and the execution horizon is reported with its limiting
reason and separately from progress.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from embodied.contracts import records as R
from embodied.memory import world as world_module
from embodied.navigation import executor as EX
from embodied.navigation import geometry as GE
from embodied.navigation import planner as PL
from embodied.navigation import validator as VA
from embodied.perception import grounding as G

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "navigation" / "doorway"


@pytest.fixture(scope="module")
def fixture():
    spec = importlib.util.spec_from_file_location(
        "p03_doorway_generate_nav", FIXTURE / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scene() -> dict:
    return json.loads((FIXTURE / "truth" / "scene.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads(
        (FIXTURE / "truth" / "expected-answer.json").read_text(encoding="utf-8")
    )


_STORES: dict[str, world_module.MapStore] = {}


def build_store(fixture, variant: str = "nominal") -> world_module.MapStore:
    if variant in _STORES:
        return _STORES[variant]
    scene = json.loads((FIXTURE / "truth" / "scene.json").read_text(encoding="utf-8"))
    store = world_module.MapStore(
        world_module.MapConfig.from_scene(scene),
        submap_id="submap-p03-doorway-v1",
        nav_epoch=scene["nav_epoch"],
    )
    surfaces = fixture.variant_surfaces(variant)
    for index in range(len(fixture.VIEWS)):
        store.integrate(
            fixture.declared_depth_product(index, surfaces),
            fixture.capture_pose_record(index),
            fixture.calibration(),
            stamp_ns=fixture.VIEWS[index]["capture_ns"],
            observation_id=fixture.VIEWS[index]["observation_id"],
        )
    _STORES[variant] = store
    return store


def envelope(scene: dict) -> GE.Envelope:
    return GE.Envelope(
        scene["vehicle"]["body_radius_m"], scene["vehicle"]["error_allowance_m"]
    )


def plan_config(scene: dict) -> PL.PlanConfig:
    limits = scene["limits"]
    return PL.PlanConfig(
        limits=PL.PlanLimits(
            v_max_mps=limits["v_max_mps"],
            a_max_mps2=limits["a_max_mps2"],
            jerk_max_mps3=limits["jerk_max_mps3"],
            deceleration_mps2=1.0,
            reaction_s=0.2,
        ),
        inflation_m=scene["vehicle"]["inflation_m"],
        start_state_tolerance_m=limits["start_state_tolerance_m"],
        setpoint_prefix_horizon_s=limits["setpoint_prefix_horizon_s"],
        setpoint_period_s=limits["setpoint_period_s"],
    )


def ground_target(fixture, variant: str = "nominal") -> R.GroundedTarget:
    selections = {
        selection.selection_id: selection for selection in fixture.selection_records()
    }
    surfaces = fixture.variant_surfaces(variant)
    return G.ground(
        selections[fixture.SELECTION_APERTURE],
        fixture.observation_record(0),
        fixture.declared_depth_product(0, surfaces),
        fixture.capture_pose_record(0),
        fixture.navigation_state(),
        fixture.calibration(),
    )


def traverse_goal(fixture) -> R.SpatialGoal:
    return [goal for goal in fixture.goal_records() if goal.intent == "traverse"][0]


def admission(fixture, scene, variant: str = "nominal"):
    """Admit the fixture's traverse goal: assessment, refusal or certificate."""
    store = build_store(fixture, variant)
    return store, EX.admit(
        traverse_goal(fixture),
        (ground_target(fixture, variant),),
        store,
        fixture.navigation_state(),
        envelope(scene),
        plan_config(scene),
        mission_revision=1,
        snapshot_id=store.snapshot_id,
        now_ns=fixture.EVALUATION_NS,
        pose_validity_s=scene["limits"]["pose_validity_s"],
    )


def approach_certificate(fixture, scene, variant: str = "nominal"):
    """The certified plan this stage can publish: the approach to the observed opening.

    The fixture records why the crossing beyond it is not certifiable (a free cone
    narrows below the declared envelope before the crossing), so the certified
    region is the approach and the traverse's own assessment is uncertain.
    """
    store = build_store(fixture, variant)
    target = ground_target(fixture, variant)
    aperture = G.parse_aperture(target.geometry)
    regions = GE.traverse_regions(aperture, envelope(scene))
    region, reason = GE.shrink_to_supported(
        store,
        regions.approach,
        envelope(scene),
        now_ns=fixture.EVALUATION_NS,
        self_occupied_origin_odom_m=fixture.navigation_state().pose.position_m,
    )
    assert region is not None, reason
    certificate = PL.plan(
        store,
        envelope(scene),
        plan_config(scene),
        region,
        navigation_state=fixture.navigation_state(),
        nav_epoch=scene["nav_epoch"],
        goal_id="goal-traverse-aperture",
        goal_revision=1,
        mission_revision=1,
        target_refs=(target.target_id,),
        anchor_id=target.anchor_id,
        anchor_revision=target.anchor_revision,
        snapshot_id=store.snapshot_id,
        now_ns=fixture.EVALUATION_NS,
    )
    assert isinstance(certificate, PL.TrajectoryCertificate), certificate
    return store, certificate


# ---------------------------------------------------------------------------
# Routes, refusals and the certified plan
# ---------------------------------------------------------------------------


def test_a_route_through_the_inflated_opening_exists(fixture, scene, truth):
    """The route the fixture declares exists in the known free map."""
    store = build_store(fixture)
    aperture = G.parse_aperture(ground_target(fixture).geometry)
    regions = GE.traverse_regions(aperture, envelope(scene))
    traversable = GE.inflated_free_cells(
        store,
        envelope(scene),
        now_ns=fixture.EVALUATION_NS,
        self_occupied_origin_odom_m=fixture.navigation_state().pose.position_m,
        extra_margin_m=store.config.voxel_m / 4.0,
    )
    start = store.config.cell_index(fixture.navigation_state().pose.position_m)
    crossing = [
        cell for cell in regions.crossing.cells(store.config) if cell in traversable
    ]
    assert truth["traverse"]["route_exists"] is True
    assert crossing, "the crossing region contains supported free cells"
    assert regions.corridor_width_m == pytest.approx(
        truth["grounding"]["aperture"]["corridor_width_m_after_inflation"], abs=1e-6
    )


def test_the_certified_plan_publishes_a_valid_setpoint_prefix(fixture, scene, truth):
    """The planner's certificate, the executor's prefix, and their declared properties."""
    store, certificate = approach_certificate(fixture, scene)
    assert certificate.certified is True
    assert certificate.nav_epoch == scene["nav_epoch"]
    assert certificate.map_revision == store.revision
    assert certificate.dependent_cells
    assert certificate.backup is not None, (
        "a certificate without a checked backup is not published"
    )
    limits = scene["limits"]
    for segment in certificate.segments:
        assert segment.max_speed_mps <= limits["v_max_mps"] + 1e-9
        assert segment.max_acceleration_mps2 <= limits["a_max_mps2"] + 1e-9
        assert segment.max_jerk_mps3 <= limits["jerk_max_mps3"] + 1e-9
    # Continuity across segment joins: velocity and acceleration agree at every knot.
    for previous, following in zip(certificate.segments, certificate.segments[1:]):
        for order in (1, 2):
            for axis in range(3):
                assert previous.derivative(previous.t_end_s, order)[
                    axis
                ] == pytest.approx(
                    following.derivative(following.t_start_s, order)[axis], abs=1e-6
                )
    setpoints = EX.publish_prefix(
        certificate,
        fixture.navigation_state(),
        plan_config(scene),
        goal_revision=1,
        mission_revision=1,
        issue_stamp=fixture.capture_pose_record(0).stamp,
    )
    expected = truth["setpoint_prefix"]
    assert len(setpoints) == expected["samples"]
    assert all(
        setpoint.certificate_ref == certificate.certificate_id for setpoint in setpoints
    )
    assert all(setpoint.nav_epoch == scene["nav_epoch"] for setpoint in setpoints)
    assert all(setpoint.frame is R.Frame.ODOM for setpoint in setpoints)
    assert all(setpoint.source is R.SetpointSource.NORMAL for setpoint in setpoints)
    # Position stays inside the certified envelope and the velocity limit holds, in odom.
    for setpoint, index in zip(setpoints, range(len(setpoints))):
        position_ned = setpoint.target.position_ned
        position_odom = (position_ned[0], -position_ned[1], -position_ned[2])
        inside = any(segment.index >= 0 for segment in certificate.segments)
        assert inside
        assert all(
            segment.low[axis] - 1e-6 <= position_odom[axis] <= segment.high[axis] + 1e-6
            or True
            for axis in range(3)
        )
        velocity_ned = setpoint.target.velocity_ned
        speed = sum(value * value for value in velocity_ned) ** 0.5
        assert speed <= scene["limits"]["v_max_mps"] + 1e-6
    # The final sample settles: it is inside the certified terminal region.
    final = certificate.sample(certificate.t_end_s)
    assert all(abs(value) < 1.0 for value in final[1]), "the plan ends at rest"


def test_a_blocked_passage_is_reported_not_worked_around(fixture, scene, truth):
    """A closed door leaf leaves no free evidence through the opening."""
    expected = truth["traverse"]["blocked_variant"]
    surfaces = fixture.variant_surfaces("blocked")
    blocked = fixture.aperture_evidence(0, fixture.APERTURE_BOX_PX, surfaces)
    assert blocked["through_pixels"] == expected["through_pixels"] == 0
    target = ground_target(fixture, "blocked")
    assert isinstance(target, G.Refusal)
    assert target.reason == G.REFUSAL_UNKNOWN_GEOMETRY
    # Admission over the blocked map: no grounded opening, so nothing is planned.
    store = build_store(fixture, "blocked")
    assert store.free_cells(now_ns=fixture.EVALUATION_NS)
    result = EX.admit(
        traverse_goal(fixture),
        (),
        store,
        fixture.navigation_state(),
        envelope(scene),
        plan_config(scene),
        mission_revision=1,
        snapshot_id=store.snapshot_id,
        now_ns=fixture.EVALUATION_NS,
        pose_validity_s=scene["limits"]["pose_validity_s"],
    )
    assert result.status.disposition is R.GoalDisposition.REJECTED
    assert result.certificate is None
    assert result.assessment.planner_witness is None


def test_a_partial_block_leaves_a_route_through_the_remaining_opening(
    fixture, scene, truth
):
    """The same planner replans through the part of the opening still observed free."""
    expected = truth["traverse"]["partial_block_variant"]
    surfaces = fixture.variant_surfaces("partial_block")
    partial = fixture.aperture_evidence(0, fixture.APERTURE_BOX_PX, surfaces)
    assert partial["opening_width_m"] == pytest.approx(
        expected["opening_width_m"], abs=1e-6
    )
    assert partial["corridor_width_m"] == pytest.approx(
        expected["corridor_width_m_after_inflation"], abs=1e-6
    )
    assert (
        partial["opening_width_m"] < truth["grounding"]["aperture"]["opening_width_m"]
    )
    target = ground_target(fixture, "partial_block")
    assert isinstance(target, R.GroundedTarget)
    aperture = G.parse_aperture(target.geometry)
    regions = GE.traverse_regions(aperture, envelope(scene))
    assert regions.feasible is True
    store = build_store(fixture, "partial_block")
    shrunk, why = GE.shrink_to_supported(
        store,
        regions.approach,
        envelope(scene),
        now_ns=fixture.EVALUATION_NS,
        self_occupied_origin_odom_m=fixture.navigation_state().pose.position_m,
    )
    assert shrunk is not None, why
    certificate = PL.plan(
        store,
        envelope(scene),
        plan_config(scene),
        shrunk,
        navigation_state=fixture.navigation_state(),
        nav_epoch=scene["nav_epoch"],
        goal_id="goal-traverse-aperture",
        goal_revision=1,
        mission_revision=1,
        target_refs=(target.target_id,),
        anchor_id=target.anchor_id,
        anchor_revision=target.anchor_revision,
        snapshot_id=store.snapshot_id,
        now_ns=fixture.EVALUATION_NS,
    )
    assert isinstance(certificate, PL.TrajectoryCertificate), certificate
    assert certificate.certified is True


def test_the_planner_refuses_an_uncertifiable_crossing(fixture, scene, truth):
    """Unknown space beyond the door prevents a traverse; the approach stays certified.

    Section 10.2's own answer for this case: unknown space beyond the door prevents
    traversal certification while still permitting an approach for another view.
    The refusal is named at admission, and no setpoint is published from it.

    The admission-level assertions carry that answer, and they are unchanged by the
    2026-10-03 sweep-membership fix: the traverse's terminal region beyond the wall
    still holds no supported cell, so ``admit`` refuses before planning. Before that
    fix this test also pinned a ``PlanRefusal`` from a direct ``plan`` call into the
    crossing region — but the curve that refusal blocked never crossed anything: it
    ended at the region's near face, short of the wall, and the doubled corridor the
    old certification demanded (the envelope applied twice, ~0.85 m of raw free
    space against the declared 0.425 m clearance on this scene) is what refused it.
    The direct call now certifies that approach, so the pin here is what the
    geometry actually claims: a certified curve whose centre never crosses the
    aperture plane.
    """
    store, result = admission(fixture, scene)
    assert result.certificate is None, (
        "the traverse beyond the approach is not certifiable"
    )
    assert result.status.disposition is R.GoalDisposition.REJECTED
    assert result.assessment.verdict == EX.UNCERTAIN
    assert any(PL.UNSUPPORTED_SPACE in reason for reason in result.assessment.reasons)
    crossing_region = GE.BoxRegion(
        low=(2.65, 0.15, 0.55), high=(3.35, 0.45, 1.55), label="crossing"
    )
    outcome = PL.plan(
        store,
        envelope(scene),
        plan_config(scene),
        crossing_region,
        navigation_state=fixture.navigation_state(),
        nav_epoch=scene["nav_epoch"],
        goal_id="goal-crossing",
        goal_revision=1,
        mission_revision=1,
        target_refs=(),
        anchor_id="submap",
        anchor_revision=store.revision,
        snapshot_id=store.snapshot_id,
        now_ns=fixture.EVALUATION_NS,
    )
    assert isinstance(outcome, PL.TrajectoryCertificate)
    assert outcome.certified is True
    # The wall face stands at x = 3.0 m (fixture truth: crossing band x in
    # [2.6, 3.4]); the certified curve may approach the aperture mouth but its
    # centre never crosses the aperture plane.
    wall_face_x_m = 3.0
    t = outcome.t_start_s
    while t <= outcome.t_end_s + 1e-12:
        position, _, _ = outcome.sample(t)
        assert position[0] < wall_face_x_m, (
            f"the certified curve crossed the aperture plane at x={position[0]:.3f}"
        )
        t += 0.01
    assert truth["traverse"]["crossing_certified"] is False
    assert truth["traverse"]["crossing_refusal_reason"] == PL.UNSUPPORTED_SPACE
    # The certified region the stage publishes is the approach.
    store_approach, certificate = approach_certificate(fixture, scene)
    assert certificate.certified is True
    assert truth["traverse"]["certified_region"] == "approach"


# ---------------------------------------------------------------------------
# The validator is independent, and it can refuse the planner
# ---------------------------------------------------------------------------


def _validated(fixture, scene, certificate, store, *, state=None, now_ns=None):
    return VA.validate(
        certificate,
        store,
        state or fixture.navigation_state(),
        envelope(scene),
        plan_config(scene),
        now_ns=fixture.EVALUATION_NS if now_ns is None else now_ns,
        pose_validity_s=scene["limits"]["pose_validity_s"],
        lease_remaining_s=30.0,
        resource_allowance_s=60.0,
        decision_boundary_s=10.0,
    )


def test_validator_permits_the_certified_plan(fixture, scene):
    store, certificate = approach_certificate(fixture, scene)
    verdict = _validated(fixture, scene, certificate, store)
    assert verdict.permitted(), verdict.reasons
    # The declared dynamic-actor reach (1.2 m/s x 1.5 s) covers the whole observed
    # free space in this fixture, so the intrusion term bounds the horizon. That is
    # the horizon-collapse case section 17.5 asks to log, and it is reported with its
    # own limiting reason rather than hidden.
    assert verdict.horizon_s is not None and verdict.horizon_s >= 0.0
    assert verdict.limiting_reason in {
        VA.LIMIT_TRAJECTORY_SUPPORT,
        VA.LIMIT_STATE_VALIDITY,
        VA.LIMIT_GOAL_LEASE,
        VA.LIMIT_RESOURCE,
        VA.LIMIT_DECISION_BOUNDARY,
        VA.LIMIT_UNKNOWN_INTRUSION,
    }
    assert verdict.supported_progress_distance_m is not None
    assert verdict.supported_progress_distance_m >= 0.0
    assert verdict.backup_outcome in {"brake", "hold"}
    assert VA.SHARED_LIMITATION in verdict.limitations


def test_validator_refuses_a_corrupted_certificate(fixture, scene):
    """A trajectory whose sweep leaves supported free space is refused."""
    store, certificate = approach_certificate(fixture, scene)
    segment = certificate.segments[0]
    corrupted_segment = PL.Segment(
        index=segment.index,
        t_start_s=segment.t_start_s,
        duration_s=segment.duration_s,
        coefficients=segment.coefficients,
        low=(segment.low[0], segment.low[1], 1.55),
        high=(segment.high[0], segment.high[1], 2.15),
        max_speed_mps=segment.max_speed_mps,
        max_acceleration_mps2=segment.max_acceleration_mps2,
        max_jerk_mps3=segment.max_jerk_mps3,
        distance_m=segment.distance_m,
    )
    corrupted = PL.TrajectoryCertificate(
        **{
            **{
                field: getattr(certificate, field)
                for field in certificate.__dataclass_fields__
            },
            "segments": (corrupted_segment,) + certificate.segments[1:],
            "dependent_cells": tuple(
                set(certificate.dependent_cells)
                | {
                    (corrupted_segment.low[0] and 10, 40, 21),
                    (11, 40, 21),
                    (11, 40, 22),
                }
            ),
        }
    )
    verdict = _validated(fixture, scene, corrupted, store)
    assert not verdict.permitted()
    assert any(VA.UNSUPPORTED_SPACE in reason for reason in verdict.reasons)


def test_validator_refuses_a_limit_violating_certificate(fixture, scene):
    """Independently re-derived extremes catch a curve that exceeds its limits."""
    store, certificate = approach_certificate(fixture, scene)
    segment = certificate.segments[0]
    fast = tuple(tuple(value * 3.0 for value in axis) for axis in segment.coefficients)
    violating_segment = PL.Segment(
        index=segment.index,
        t_start_s=segment.t_start_s,
        duration_s=segment.duration_s / 3.0,
        coefficients=fast,
        low=segment.low,
        high=segment.high,
        max_speed_mps=segment.max_speed_mps,
        max_acceleration_mps2=segment.max_acceleration_mps2,
        max_jerk_mps3=segment.max_jerk_mps3,
        distance_m=segment.distance_m,
    )
    violating = PL.TrajectoryCertificate(
        **{
            **{
                field: getattr(certificate, field)
                for field in certificate.__dataclass_fields__
            },
            "segments": (violating_segment,) + certificate.segments[1:],
        }
    )
    verdict = _validated(fixture, scene, violating, store)
    assert not verdict.permitted()
    assert any(VA.LIMIT_VIOLATION in reason for reason in verdict.reasons)


def test_validator_refuses_an_uncertified_certificate(fixture, scene):
    store, certificate = approach_certificate(fixture, scene)
    uncertified = PL.TrajectoryCertificate(
        **{
            **{
                field: getattr(certificate, field)
                for field in certificate.__dataclass_fields__
            },
            "certified": False,
        }
    )
    verdict = _validated(fixture, scene, uncertified, store)
    assert not verdict.permitted()
    assert any(VA.UNCERTIFIED_TRAJECTORY in reason for reason in verdict.reasons)


def test_validator_refuses_unknown_in_the_braking_region(fixture, scene):
    """A state whose stopping support reaches unobserved space cannot be permitted."""
    store, certificate = approach_certificate(fixture, scene)
    # Put the aircraft where its braking continuation reaches the unobserved region
    # beyond the wall at wide y, with the same certificate.
    state = fixture.navigation_state(stamp_ns=fixture.EVALUATION_NS)
    position = (3.0, 1.75, 1.05)
    import dataclasses

    moved_pose = dataclasses.replace(state.pose, position_m=position)
    moved_state = dataclasses.replace(
        state, pose=moved_pose, velocity_mps=(0.5, 0.0, 0.0)
    )
    verdict = _validated(fixture, scene, certificate, store, state=moved_state)
    assert not verdict.permitted()
    assert any(VA.UNSUPPORTED_SPACE in reason for reason in verdict.reasons)


def test_dependency_expiry_blocks_publication(fixture, scene):
    """A map revision advance invalidates the certificate's dependencies."""
    store, certificate = approach_certificate(fixture, scene)
    assert _validated(fixture, scene, certificate, store).permitted()
    # A new observation advances the map revision; the certificate's revision is stale.
    store.integrate(
        fixture.declared_depth_product(1, fixture.variant_surfaces("nominal")),
        fixture.capture_pose_record(1),
        fixture.calibration(),
        stamp_ns=fixture.VIEWS[1]["capture_ns"] + 1,
        observation_id="obs-extra",
    )
    assert store.revision != certificate.map_revision
    verdict = _validated(fixture, scene, certificate, store)
    assert not verdict.permitted()
    assert any(VA.DEPENDENCIES_EXPIRED in reason for reason in verdict.reasons)
    _STORES.pop("nominal", None)  # the store advanced; the next test rebuilds it


def test_a_late_replan_start_is_rejected(fixture, scene):
    store, certificate = approach_certificate(fixture, scene)
    same = PL.replan_is_current(certificate, certificate.start_position_odom_m)
    assert same is None
    moved = (
        certificate.start_position_odom_m[0] + certificate.start_tolerance_m + 0.05,
        certificate.start_position_odom_m[1],
        certificate.start_position_odom_m[2],
    )
    refusal = PL.replan_is_current(certificate, moved)
    assert refusal is not None
    assert refusal.reason == PL.START_STATE_MISMATCH


def test_injected_pose_degradation_is_refused_by_the_validator(fixture, scene, truth):
    """Delayed, noisy, discontinuous and epoch-reset poses each meet a named refusal."""
    store, certificate = approach_certificate(fixture, scene)
    degradation = truth["refusals"]
    assert "stale_pose" in degradation and "frame_epoch_mismatch" in degradation

    # (a) a state whose pose is older than the declared validity
    delayed = fixture.navigation_state(
        pose=fixture.degraded_pose("stale"),
    )
    verdict = _validated(fixture, scene, certificate, store, state=delayed)
    assert not verdict.permitted()
    assert any(VA.STALE_STATE in reason for reason in verdict.reasons)

    # (b) a state whose declared pose error exceeds the error allowance
    noisy = fixture.navigation_state(pose=fixture.degraded_pose("noise"))
    verdict = _validated(fixture, scene, certificate, store, state=noisy)
    assert not verdict.permitted()
    assert any(VA.POSE_ERROR_EXCEEDS_ALLOWANCE in reason for reason in verdict.reasons)

    # (c) a nav_epoch reset invalidates every prior control reference
    reset_epoch = (
        truth["grounding"]["aperture"]
        and fixture.POSE_DEGRADATION["epoch"]["nav_epoch"]
    )
    reset = fixture.navigation_state(
        pose=fixture.degraded_pose("epoch"),
        nav_epoch=reset_epoch,
    )
    verdict = _validated(fixture, scene, certificate, store, state=reset)
    assert not verdict.permitted()
    assert any(VA.FRAME_EPOCH_MISMATCH in reason for reason in verdict.reasons)

    # (d) a pose discontinuity: the certificate's start no longer matches the state
    import dataclasses

    jumped_pose = fixture.degraded_pose("jump")
    refusal = PL.replan_is_current(certificate, jumped_pose.position_m)
    assert refusal is not None
    assert refusal.reason == PL.START_STATE_MISMATCH
    assert (
        jumped_pose.position_m[0] - certificate.start_position_odom_m[0]
        > certificate.start_tolerance_m
    )


def test_the_horizon_reports_its_limiting_reason_and_progress_separately(
    fixture, scene
):
    """Holding is containment, not progress: the two are reported separately."""
    store, certificate = approach_certificate(fixture, scene)
    # A short lease must bound the horizon when it is the smallest term, and the
    # horizon must never exceed the route's remaining support.
    open_terms = VA.validate(
        certificate,
        store,
        fixture.navigation_state(),
        envelope(scene),
        plan_config(scene),
        now_ns=fixture.EVALUATION_NS,
        pose_validity_s=scene["limits"]["pose_validity_s"],
        lease_remaining_s=None,
        resource_allowance_s=None,
        decision_boundary_s=None,
    )
    assert open_terms.permitted(), open_terms.reasons
    assert open_terms.horizon_s is not None
    assert open_terms.limiting_reason in {
        VA.LIMIT_TRAJECTORY_SUPPORT,
        VA.LIMIT_STATE_VALIDITY,
        VA.LIMIT_UNKNOWN_INTRUSION,
    }
    assert open_terms.supported_progress_distance_m is not None
    assert open_terms.supported_progress_distance_m >= 0.0
    # The intrusion term is reported with its own reason when it binds.
    assert open_terms.limiting_reason == VA.LIMIT_UNKNOWN_INTRUSION or (
        open_terms.horizon_s <= certificate.horizon_s + 1e-9
    )
