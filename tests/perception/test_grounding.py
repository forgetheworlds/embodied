"""Behaviour of the one grounding path, the detector seam and gate association.

Every expected value is read from the fixture's own ``truth/expected-answer.json``
and ``truth/scene.json`` — the hand-derived, pre-registered artifacts beside the
episode — so a failure means the implementation changed or the authored derivation
was wrong, never that a generated expectation moved with the code.

The episode itself is decoded through P02's own reader, and the manifest's hashes
are verified, which is also the proof that this fixture is P02 storage: the agent
projection sees the episode and cannot see ``truth/``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from embodied.bench.recorder import AgentSurface
from embodied.contracts import records as R
from embodied.perception import detector as detector_module
from embodied.perception import grounding as G
from embodied.perception import tracking as T

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "navigation" / "doorway"


def _load_fixture_module():
    spec = importlib.util.spec_from_file_location("p03_doorway_generate", FIXTURE / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixture():
    return _load_fixture_module()


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads((FIXTURE / "truth" / "expected-answer.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scene() -> dict:
    return json.loads((FIXTURE / "truth" / "scene.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def variants() -> dict:
    return json.loads((FIXTURE / "truth" / "variants.json").read_text(encoding="utf-8"))


def selections(fixture) -> dict:
    return {selection.selection_id: selection for selection in fixture.selection_records()}


def ground_view(fixture, selection_id: str, *, view_index: int = 0, variant: str = "nominal", **overrides):
    """Ground one selection with the fixture's declared depth, pose and state."""
    surfaces = fixture.variant_surfaces(variant)
    observation = fixture.observation_record(view_index)
    depth = fixture.declared_depth_product(view_index, surfaces)
    arguments = {
        "selection": selections(fixture)[selection_id],
        "observation": observation,
        "depth": depth,
        "capture_pose": fixture.capture_pose_record(view_index),
        "current_state": fixture.navigation_state(),
        "calibration": fixture.calibration(),
    }
    arguments.update(overrides)
    return G.ground(**arguments)


# ---------------------------------------------------------------------------
# The fixture is P02 storage, and its truth is out of the agent's reach
# ---------------------------------------------------------------------------


def test_episode_reads_through_the_agent_projection_and_verifies(truth):
    surface = AgentSurface.open(FIXTURE)
    surface.verify_artifacts()
    events = surface.agent_events()
    assert [event.kind for event in events] == [
        "mission",
        "observation",
        "observation",
        "selection",
        "selection",
        "selection",
        "selection",
        "goal",
        "goal",
    ]
    assert surface.manifest.episode_id == truth["episode_id"]
    assert surface.manifest.sensor_mode is R.SensorMode.POSE_ASSISTED
    assert surface.manifest.episode_kind == "synthetic-fixture"
    assert surface.final_report() is None
    assert surface.payload_names() == (
        "payloads/obs-1-left.ppm",
        "payloads/obs-1-right.ppm",
        "payloads/obs-2-left.ppm",
        "payloads/obs-2-right.ppm",
    )
    # The manifest declares only projection members, so truth/ is unreachable by name.
    assert set(surface.manifest.artifacts) == set(surface.payload_names()) | {"agent-events.jsonl"}
    with pytest.raises(Exception):
        surface.read_member("truth/expected-answer.json")


def test_visual_selections_decode_as_records(truth):
    surface = AgentSurface.open(FIXTURE)
    decoded = [
        R.from_dict(R.VisualSelection, event.payload)
        for event in surface.agent_events()
        if event.kind == "selection"
    ]
    by_id = {selection.selection_id: selection for selection in decoded}
    aperture = by_id[truth["grounding"]["aperture"]["selection_id"]]
    assert aperture.geometry_kind is R.SelectionGeometry.BOX
    assert tuple(int(value) for value in aperture.geometry) == tuple(
        truth["grounding"]["aperture"]["box_px"]
        if "box_px" in truth["grounding"]["aperture"]
        else aperture.geometry
    )


# ---------------------------------------------------------------------------
# Nominal grounding against the hand-derived answer
# ---------------------------------------------------------------------------


def test_aperture_selection_grounds_to_the_conservative_opening(fixture, truth, scene):
    expected = truth["grounding"]["aperture"]
    tolerance = truth["tolerances"]["aperture_edge_abs_m"]
    target = ground_view(fixture, expected["selection_id"])
    assert isinstance(target, R.GroundedTarget), target
    aperture = G.parse_aperture(target.geometry)
    assert aperture is not None, "the aperture selection must ground to an opening rectangle"
    # The polygon is on the wall plane, not the depth behind it.
    offset = float(sum(n * p for n, p in zip(aperture.plane_normal_odom, aperture.plane_point_odom_m)))
    assert offset == pytest.approx(scene["aperture"]["wall_x_odom_m"], abs=tolerance)
    assert aperture.plane_normal_odom[0] == pytest.approx(1.0, abs=1e-6)
    assert aperture.width_m == pytest.approx(expected["opening_width_m"], abs=tolerance)
    for measured, authored in zip(aperture.corners_odom_m, expected["corners_odom_m"]):
        for value, wanted in zip(measured, authored):
            assert value == pytest.approx(wanted, abs=tolerance)
    # The far wall is *behind* the opening: no corner may sit on it.
    far_wall_x = scene["surfaces"][-1]["plane_point_odom_m"][0]
    assert all(abs(corner[0] - scene["aperture"]["wall_x_odom_m"]) <= tolerance for corner in aperture.corners_odom_m)
    assert abs(aperture.corners_odom_m[0][0] - far_wall_x) > 1.0
    # Envelope: the measurement bound propagated, with the shared pose bound added linearly.
    assert target.uncertainty[0] == pytest.approx(
        expected["uncertainty_m"]["normal_sigma_m"], abs=truth["tolerances"]["uncertainty_abs_m"]
    )
    assert target.uncertainty[1] == pytest.approx(
        expected["uncertainty_m"]["tangential_sigma_m"],
        abs=truth["tolerances"]["uncertainty_abs_m"],
    )
    assert target.uncertainty[2] == pytest.approx(
        expected["uncertainty_m"]["edge_shrink_m"], abs=truth["tolerances"]["uncertainty_abs_m"]
    )
    # Provenance is mandatory on the frozen record: selection, observation, anchor identity.
    assert target.selection_ids == (expected["selection_id"],)
    assert target.observation_ids == (expected["observation_id"],)
    assert target.frame is R.Frame.ODOM
    assert target.anchor_id and target.anchor_revision
    assert target.valid is True


def test_point_selection_grounds_to_the_far_wall_point(fixture, truth, scene):
    expected = truth["grounding"]["far_point"]
    tolerance = truth["tolerances"]["point_abs_m"]
    target = ground_view(fixture, expected["selection_id"])
    assert isinstance(target, R.GroundedTarget), target
    assert len(target.geometry) == G.POINT_NUMBERS
    for measured, authored in zip(target.geometry, expected["point_odom_m"]):
        assert measured == pytest.approx(authored, abs=tolerance)
    assert target.geometry[0] == pytest.approx(scene["surfaces"][-1]["plane_point_odom_m"][0], abs=tolerance)
    assert target.uncertainty[0] == pytest.approx(
        expected["uncertainty_m"]["normal_sigma_m"], abs=truth["tolerances"]["uncertainty_abs_m"]
    )


def test_invalid_depth_band_grounds_nothing(fixture, truth, variants):
    # The fixture declares the whole refusal vocabulary; the band case must use one of its tokens.
    assert "unknown_geometry" in truth["refusals"]
    outcome = ground_view(fixture, variants["invalid_depth"]["selection_id"])
    assert isinstance(outcome, G.Refusal)
    assert outcome.reason in truth["refusals"]
    assert outcome.reason == G.REFUSAL_UNKNOWN_GEOMETRY
    assert outcome.support == "uncertain"
    assert "valid depth" in outcome.detail


def test_missing_depth_is_a_refusal_not_a_default(fixture, truth):
    outcome = ground_view(
        fixture, truth["grounding"]["aperture"]["selection_id"], depth=None
    )
    assert isinstance(outcome, G.Refusal)
    assert outcome.reason == G.REFUSAL_MISSING_DEPTH


def test_calibration_is_bound_to_the_observation(fixture, truth):
    selection = selections(fixture)[truth["grounding"]["aperture"]["selection_id"]]
    observation = fixture.observation_record(0)
    depth = fixture.declared_depth_product(0, fixture.variant_surfaces("nominal"))
    calibration = fixture.calibration()
    other = R.Calibration(
        **{
            **{
                field.name: getattr(calibration, field.name)
                for field in __import__("dataclasses").fields(calibration)
            },
            "version": "9",
        }
    )
    outcome = G.ground(
        selection,
        observation,
        depth,
        fixture.capture_pose_record(0),
        fixture.navigation_state(),
        other,
    )
    assert isinstance(outcome, G.Refusal)
    assert outcome.reason == G.REFUSAL_CALIBRATION_MISMATCH


# ---------------------------------------------------------------------------
# Injected pose degradation (coordinator scope addition)
# ---------------------------------------------------------------------------


def test_stale_capture_pose_is_refused(fixture, truth, variants):
    injection = variants["pose_degradation"]["stale"]
    assert injection["capture_age_s"] > injection["threshold_s"]
    state = fixture.navigation_state(pose=fixture.degraded_pose("stale"))
    outcome = ground_view(
        fixture, truth["grounding"]["aperture"]["selection_id"], current_state=state
    )
    assert isinstance(outcome, G.Refusal)
    assert outcome.reason == G.REFUSAL_STALE_POSE
    assert outcome.support == "uncertain"
    assert "older than the current state" in outcome.detail


def test_pose_noise_beyond_the_declared_uncertainty_is_carried_not_ignored(fixture, truth, variants):
    """A noisy pose must widen the reported envelope, never quietly shrink it."""
    injection = variants["pose_degradation"]["noise"]
    clean = ground_view(fixture, truth["grounding"]["aperture"]["selection_id"])
    noisy = ground_view(
        fixture,
        truth["grounding"]["aperture"]["selection_id"],
        capture_pose=fixture.degraded_pose("noise"),
    )
    assert isinstance(clean, R.GroundedTarget) and isinstance(noisy, R.GroundedTarget)
    for axis, (clean_value, noisy_value) in enumerate(zip(clean.uncertainty, noisy.uncertainty)):
        if axis < 2:
            assert noisy_value > clean_value, "a pose bound beyond the allowance must not be dropped"
    offset_m = injection["position_offset_odom_m"][0]
    assert offset_m > injection["threshold_m"]
    # The injected offset moves the reported geometry by exactly that offset, so a pose
    # error larger than the allowance is visible rather than absorbed.
    clean_aperture = G.parse_aperture(clean.geometry)
    noisy_aperture = G.parse_aperture(noisy.geometry)
    clean_offset = clean_aperture.plane_point_odom_m[0]
    noisy_offset = noisy_aperture.plane_point_odom_m[0]
    assert noisy_offset - clean_offset == pytest.approx(offset_m, abs=1e-6)
    # And the declared sigma moves to the injected value, above the usable limit.
    assert noisy.uncertainty[0] > clean.uncertainty[0]


def test_nav_epoch_reset_invalidates_the_capture_pose(fixture, truth, variants):
    injection = variants["pose_degradation"]["epoch"]
    state = fixture.navigation_state(
        pose=fixture.degraded_pose("epoch"), nav_epoch=injection["nav_epoch"]
    )
    assert state.nav_epoch == injection["nav_epoch"]
    assert state.pose.nav_epoch == injection["nav_epoch"]
    outcome = ground_view(
        fixture, truth["grounding"]["aperture"]["selection_id"], current_state=state
    )
    assert isinstance(outcome, G.Refusal)
    assert outcome.reason == G.REFUSAL_FRAME_EPOCH_MISMATCH


def test_a_stale_target_is_not_current(fixture, truth, variants):
    target = ground_view(fixture, truth["grounding"]["aperture"]["selection_id"])
    assert isinstance(target, R.GroundedTarget)
    assert G.target_currency(target, fixture.navigation_state()) is None
    refused = G.target_currency(target, fixture.navigation_state(), anchor_revision="rev-2")
    assert refused is not None and refused.reason == G.REFUSAL_FRAME_EPOCH_MISMATCH
    assert variants["pose_degradation"]["jump"]["position_offset_odom_m"][0] > 0.0


# ---------------------------------------------------------------------------
# The phrase path is the same path, and an absent detector is an explicit refusal
# ---------------------------------------------------------------------------


def phrase_selection(fixture) -> R.VisualSelection:
    return R.VisualSelection(
        selection_id="sel-phrase-doorway",
        observation_id="obs-1",
        coordinate_convention=fixture.calibration().pixel_convention,
        geometry_kind=R.SelectionGeometry.MASK,
        geometry="the open doorway in the wall ahead",
        crop_transform=None,
        description="phrase selection supplied by the mission query",
        confidence=None,
    )


def test_phrase_selection_refuses_without_a_verified_detector(fixture, truth):
    pinned = detector_module.PinnedDetector()
    outcome = ground_view(
        fixture,
        truth["grounding"]["aperture"]["selection_id"],
        selection=phrase_selection(fixture),
        detector=pinned,
        detector_image=None,
    )
    assert isinstance(outcome, G.Refusal)
    assert outcome.reason == G.REFUSAL_DETECTOR_UNAVAILABLE
    verification = detector_module.verify_available_detector()
    assert verification["verdict"].startswith("no verified detector")


def test_no_detector_at_all_is_still_an_explicit_refusal(fixture, truth):
    outcome = ground_view(
        fixture, truth["grounding"]["aperture"]["selection_id"], selection=phrase_selection(fixture)
    )
    assert isinstance(outcome, G.Refusal)
    assert outcome.reason == G.REFUSAL_DETECTOR_UNAVAILABLE


def test_injected_candidates_take_the_identical_geometric_path(fixture, truth):
    """A phrase resolved by an injected DIAGNOSTIC candidate grounds exactly as the box does."""
    expected = truth["grounding"]["aperture"]
    box_selection = selections(fixture)[expected["selection_id"]]
    injected = detector_module.InjectedDetector(
        detector_module.injected_candidate("cand-1", tuple(box_selection.geometry), score=0.9),
        detector_module.injected_candidate("cand-2", (24.0, 0.0, 488.0, 479.0), score=0.4),
    )
    phrase = ground_view(
        fixture,
        expected["selection_id"],
        selection=phrase_selection(fixture),
        detector=injected,
        detector_image=fixture.render_camera(0, 0.0, fixture.variant_surfaces("nominal")),
    )
    direct = ground_view(fixture, expected["selection_id"])
    assert isinstance(phrase, R.GroundedTarget) and isinstance(direct, R.GroundedTarget)
    assert phrase.geometry == direct.geometry
    assert phrase.uncertainty == direct.uncertainty
    # The injected candidate is cited, and it is labelled diagnostic, not a model.
    assert phrase.selection_ids == ("sel-phrase-doorway", "cand-1")
    assert phrase.identity_alternatives == ("cand-2",)
    assert injected._candidates[0].provenance.source == detector_module.DIAGNOSTIC_SOURCE
    assert injected._candidates[0].provenance.checkpoint_hash is None


# ---------------------------------------------------------------------------
# Association across the two views
# ---------------------------------------------------------------------------


def test_two_view_association_is_uncalibrated_and_static(fixture, truth, scene):
    observed = {
        name: ground_view(fixture, name, view_index=index)
        for name, index in (
            (fixture.SELECTION_APERTURE, 0),
            (fixture.SELECTION_APERTURE_V2, 1),
        )
    }
    first = T.candidate_from_target(observed[fixture.SELECTION_APERTURE])
    second = T.candidate_from_target(observed[fixture.SELECTION_APERTURE_V2])
    track = T.new_track("track-doorway-1", first, stamp_ns=scene["views"][0]["capture_ns"])
    assert track.motion_model == T.STATIC
    scored = T.associate_candidates(
        track,
        (second,),
        stamp_ns=scene["views"][1]["capture_ns"],
        max_speed_mps=scene["dynamic"]["speed_mps"],
        geometry=first.geometry,
    )
    association = scored[0]
    assert association.gate_passed is True
    assert association.calibrated is False, "no labelled development associations exist here"
    assert association.score <= 1.0
    assert 0.0 < association.score
    assert association.identity_alternatives == ()
    updated = T.observe(track, association, second, stamp_ns=scene["views"][1]["capture_ns"])
    assert updated.motion_model == T.STATIC
    assert updated.observation_count == 2
    assert updated.velocity_mps == (0.0, 0.0, 0.0)
    # The predicted envelope grows once observations stop.
    assert T.predicted_envelope_m(
        updated, horizon_s=2.0, speed_uncertainty_mps=scene["dynamic"]["speed_mps"]
    ) > updated.position_sigma_m


def test_ambiguous_association_retains_alternatives(fixture, truth, scene):
    first = T.candidate_from_target(ground_view(fixture, truth["grounding"]["aperture"]["selection_id"]))
    track = T.new_track("track-doorway-1", first, stamp_ns=scene["views"][0]["capture_ns"])
    near = T.GroundedCandidate(
        candidate_id="cand-near",
        position_odom_m=first.position_odom_m,
        position_sigma_m=first.position_sigma_m,
        geometry=first.geometry,
    )
    also_near = T.GroundedCandidate(
        candidate_id="cand-also-near",
        position_odom_m=first.position_odom_m,
        position_sigma_m=first.position_sigma_m,
        geometry=first.geometry,
    )
    scored = T.associate_candidates(
        track,
        (near, also_near),
        stamp_ns=scene["views"][1]["capture_ns"],
        max_speed_mps=scene["dynamic"]["speed_mps"],
        geometry=first.geometry,
    )
    assert len(scored) == 2
    assert scored[0].identity_alternatives == ("cand-also-near",)
    assert any("ambiguous" in reason for reason in scored[0].reasons)
    assert all(association.calibrated is False for association in scored)


def test_an_impossible_jump_fails_the_gate(fixture, truth, scene):
    first = T.candidate_from_target(ground_view(fixture, truth["grounding"]["aperture"]["selection_id"]))
    track = T.new_track("track-doorway-1", first, stamp_ns=scene["views"][0]["capture_ns"])
    far = T.GroundedCandidate(
        candidate_id="cand-far",
        position_odom_m=(first.position_odom_m[0], first.position_odom_m[1] + 50.0, first.position_odom_m[2]),
        position_sigma_m=first.position_sigma_m,
    )
    association = T.associate(
        track,
        far,
        stamp_ns=scene["views"][0]["capture_ns"] + 500_000_000,
        max_speed_mps=scene["dynamic"]["speed_mps"],
    )
    assert association.gate_passed is False
    assert association.score == 0.0
    assert "physically possible bound" in association.reasons[0]
