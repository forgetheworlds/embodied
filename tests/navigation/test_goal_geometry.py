"""Goal geometry: apertures, goal regions, envelopes and the unknown-is-never-free rule.

Expected values come from the fixture's ``truth/`` artifacts. The scene's
declared parameters are also asserted equal to the constants the implementation
declares, so a drift between the hand derivation and the code is loud rather than
silent.
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
    spec = importlib.util.spec_from_file_location("p03_doorway_generate_goals", FIXTURE / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scene() -> dict:
    return json.loads((FIXTURE / "truth" / "scene.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads((FIXTURE / "truth" / "expected-answer.json").read_text(encoding="utf-8"))


_STORES: dict[str, world_module.MapStore] = {}


@pytest.fixture(scope="module")
def store(fixture) -> world_module.MapStore:
    return build_store(fixture)


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
    return GE.Envelope(scene["vehicle"]["body_radius_m"], scene["vehicle"]["error_allowance_m"])


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


def ground_aperture(fixture, variant: str = "nominal") -> R.GroundedTarget:
    selections = {selection.selection_id: selection for selection in fixture.selection_records()}
    surfaces = fixture.variant_surfaces(variant)
    return G.ground(
        selections[fixture.SELECTION_APERTURE],
        fixture.observation_record(0),
        fixture.declared_depth_product(0, surfaces),
        fixture.capture_pose_record(0),
        fixture.navigation_state(),
        fixture.calibration(),
    )


def test_declared_parameters_match_the_implementation(fixture, scene, truth):
    """The fixture's declared parameters and the implementation's constants agree."""
    assert scene["depth"]["sigma_pixel_px"] == G.SIGMA_PIXEL_PX
    assert scene["map"]["surface_band_m"] == G.SEED_BAND_M
    assert scene["limits"]["pose_validity_s"] == G.POSE_VALIDITY_S
    assert scene["map"]["voxel_m"] == fixture.VOXEL_M
    assert scene["vehicle"]["inflation_m"] == pytest.approx(
        scene["vehicle"]["body_radius_m"] + scene["vehicle"]["error_allowance_m"]
    )
    assert truth["provenance"] == "DIAGNOSTIC"
    assert truth["grounding"]["aperture"]["corridor_width_m_after_inflation"] == pytest.approx(
        truth["grounding"]["aperture"]["opening_width_m"] - 2.0 * scene["vehicle"]["inflation_m"],
        abs=1e-6,
    )


def test_aperture_regions_come_from_the_observed_opening(fixture, scene, truth):
    target = ground_aperture(fixture)
    assert isinstance(target, R.GroundedTarget)
    aperture = G.parse_aperture(target.geometry)
    regions = GE.traverse_regions(aperture, envelope(scene))
    expected = truth["grounding"]["aperture"]
    assert regions.feasible is True
    assert regions.corridor_width_m == pytest.approx(expected["corridor_width_m_after_inflation"], abs=1e-6)
    # The crossing region sits on the opening's plane, not at the depth behind it.
    wall_x = scene["aperture"]["wall_x_odom_m"]
    assert regions.crossing.low[0] < wall_x < regions.crossing.high[0]
    assert regions.crossing.high[0] - regions.crossing.low[0] == pytest.approx(
        2 * GE.CROSSING_HALF_DEPTH_M, abs=1e-9
    )
    # Approach, crossing and exit are ordered along the crossing direction, and the
    # approach region stands off from the wall rather than touching it.
    assert regions.approach.high[0] < wall_x
    assert regions.terminal.low[0] > wall_x
    assert regions.crossing.low[1] >= aperture.corners_odom_m[0][1] + scene["vehicle"]["inflation_m"] - 1e-9
    assert regions.crossing.high[1] <= aperture.corners_odom_m[1][1] - scene["vehicle"]["inflation_m"] + 1e-9
    # The opening is the wall plane: no corner of any region may sit on the far wall.
    far_wall_x = scene["surfaces"][-1]["plane_point_odom_m"][0]
    assert abs(regions.terminal.high[0] - far_wall_x) > 1.0


def test_narrow_opening_is_infeasible_under_a_named_constraint(fixture, scene, truth):
    target = ground_aperture(fixture, "narrow")
    aperture = G.parse_aperture(target.geometry)
    regions = GE.traverse_regions(aperture, envelope(scene))
    expected = truth["traverse"]["narrow_variant"]
    assert regions.feasible is False
    assert regions.constraint == GE.APERTURE_CLEARANCE
    assert regions.corridor_width_m == pytest.approx(
        expected["corridor_width_m_after_inflation"], abs=1e-6
    )
    assert "opening" in regions.reasons[0]


def test_unknown_space_is_never_free(fixture, scene, truth, store):
    now_ns = fixture.EVALUATION_NS
    for probe in truth["occupancy"]["probes"]:
        classes = [store.state_of(tuple(cell["index"])) for cell in probe["cells"]]
        if probe["expect"] == "free":
            assert set(classes) == {world_module.FREE}, probe["probe_id"]
        elif probe["expect"] == "unknown":
            assert set(classes) == {world_module.UNKNOWN}, probe["probe_id"]
            reasons = {store.unknown_reason(tuple(cell["index"]), now_ns=now_ns) for cell in probe["cells"]}
            assert reasons <= {world_module.NEVER_OBSERVED, world_module.STALE, world_module.CONFLICTING, world_module.SURFACE_BAND}
        else:
            assert world_module.OCCUPIED in classes, probe["probe_id"]
            assert world_module.FREE not in classes, probe["probe_id"]
    # No published free cell may sit inside the surface band of an occupied cell.
    free = store.free_cells(now_ns=now_ns)
    occupied = store.occupied_cells(now_ns=now_ns)
    neighbours = {
        (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
        for cell in occupied
        for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    }
    assert not (free & neighbours), "a cell beside a surface must not be published free"


def test_a_goal_region_overlapping_unknown_is_unsupported(fixture, scene, truth, store):
    store = build_store(fixture)
    declared = truth["traverse"].get("unknown_region") or {
        "x_m": [4.35, 4.75],
        "y_m": [1.75, 1.95],
        "z_m": [0.95, 1.15],
    }
    region = GE.BoxRegion(
        low=(declared["x_m"][0], declared["y_m"][0], declared["z_m"][0]),
        high=(declared["x_m"][1], declared["y_m"][1], declared["z_m"][1]),
        label="unknown-region",
    )
    supported, reason = GE.region_support(
        store, region, envelope(scene), now_ns=fixture.EVALUATION_NS,
        self_occupied_origin_odom_m=fixture.navigation_state().pose.position_m,
    )
    assert supported is False
    assert reason and "not supported free space" in reason
    shrunk, why = GE.shrink_to_supported(
        store, region, envelope(scene), now_ns=fixture.EVALUATION_NS,
        self_occupied_origin_odom_m=fixture.navigation_state().pose.position_m,
    )
    assert shrunk is None and why and "no evidence" in why


def test_admission_rejects_an_image_only_goal_before_planning(fixture, truth, store):
    """A goal citing only a selection, with no grounded target, never reaches the planner."""
    store = build_store(fixture)
    goal = [g for g in fixture.goal_records() if g.intent == "approach"][0]
    result = EX.admit(
        goal,
        (),
        store,
        fixture.navigation_state(),
        envelope(json.loads((FIXTURE / "truth" / "scene.json").read_text(encoding="utf-8"))),
        plan_config(json.loads((FIXTURE / "truth" / "scene.json").read_text(encoding="utf-8"))),
        mission_revision=1,
        snapshot_id=store.snapshot_id,
        now_ns=fixture.EVALUATION_NS,
        pose_validity_s=json.loads(
            (FIXTURE / "truth" / "scene.json").read_text(encoding="utf-8")
        )["limits"]["pose_validity_s"],
    )
    assert result.status.disposition is R.GoalDisposition.REJECTED
    assert result.assessment.planner_witness is None, "no planner call may happen for this goal"
    assert result.assessment.verdict == EX.INFEASIBLE
    assert EX.NO_GROUNDED_TARGET in result.status.reason
    assert result.certificate is None
    assert result.assessment.information_alternative is not None
    # The goal cites a real selection that grounds to nothing: the refusal cites it too.
    band = G.ground(
        {selection.selection_id: selection for selection in fixture.selection_records()}[
            fixture.SELECTION_OPEN_BAND
        ],
        fixture.observation_record(0),
        fixture.declared_depth_product(0, fixture.variant_surfaces("nominal")),
        fixture.capture_pose_record(0),
        fixture.navigation_state(),
        fixture.calibration(),
    )
    assert isinstance(band, G.Refusal)
    assert band.reason == G.REFUSAL_UNKNOWN_GEOMETRY
    approach_goal = goal
    assert approach_goal.selection_ids == (fixture.SELECTION_OPEN_BAND,)


def test_swept_volume_and_inflation_refuse_near_the_jamb(fixture, scene, store):
    """A region drawn beside the jamb is not supported free space, envelope included."""
    store = build_store(fixture)
    aperture = G.parse_aperture(ground_aperture(fixture).geometry)
    jamb_y = aperture.corners_odom_m[0][1] - 0.1  # inside the wall, beside the opening
    region = GE.BoxRegion(
        low=(scene["aperture"]["wall_x_odom_m"] - 0.05, jamb_y - 0.05, 0.95),
        high=(scene["aperture"]["wall_x_odom_m"] + 0.05, jamb_y + 0.05, 1.15),
        label="jamb",
    )
    supported, reason = GE.region_support(
        store, region, envelope(scene), now_ns=fixture.EVALUATION_NS,
        self_occupied_origin_odom_m=fixture.navigation_state().pose.position_m,
    )
    assert supported is False and reason
    assert world_module.OCCUPIED in {
        store.state_of(tuple(cell)) for cell in region.cells(store.config)
    } or "not supported free space" in reason
