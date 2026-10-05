"""Every goal the mission admits stays inside the flyable band.

J54-zguard-1 measured the crash head-on: the admitted explore goal regions
carried NED z bands reaching +0.2 .. +0.8 — at and below the floor datum (the
run's own ``free_bbox`` reached z +0.25, and its log lines read
``goal region ... z[-0.96, 0.14]``) — the aircraft chased one, dove below the
floor plane and flipped (``crash_disarm: AngErr=170``). The mechanism, traced:
the frontier clusters inherit the free cells' z extent, and the carve admits
floor-adjacent and below-floor cells; the vantage walk accepted whatever
altitude the searchable cells supported; and resolution's centre fallback
published the region's centre — the deep z included — when the walk found
nothing. Nothing in the chain ever asked whether the goal was flyable.

The fix derives the flyable band (``MissionRuntime._flyable_band_z``): the
declared hover band — the aligned origin minus the declared
``probe.hover_altitude_m``, widened by the executor's own terminal-region
half-extent — tightened by the map's own occupied surfaces where their
evidence reaches beyond it. Every candidate vantage is gated on it (counted
``out_of_band`` when refused), a frontier with no in-band vantage is refused
at the walk with its named gate counts, and resolution returns no target
instead of the region's centre.

Fixture, store and runtime idiom follow ``test_frontier_excursion.py``. The
stub's mission frame puts the aligned origin on the fixture's floor (its
deepest occupied cell centre), so the declared hover altitude is expressible
exactly as the live mission's frame expresses it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from embodied.navigation import geometry as GE
from embodied.platform.mission_runtime import (
    ENVELOPE,
    FRONTIER_VIEW_DIRECTION,
    MissionRuntime,
)

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

# The declared cruise altitude (configs/first_indoor.yaml probe.hover_altitude_m).
DECLARED_HOVER_ALTITUDE_M = 1.5
# The mission frame's origin: the spawn ground, here the fixture floor's own
# occupied evidence (its deepest occupied cell centre).
ORIGIN_ON_FIXTURE_FLOOR = (0.0, 0.0, 2.95)
NOW_NS = 2_000_000_000
HERE = tuple(float(v) for v in _FIXTURE.VIEWS[0]["body_position_odom_m"])


def _store(surfaces=None):
    """The doorway fixture's own map: real carving, real free space.

    ``surfaces`` replaces the declared scene for a synthetic carve (the
    J54-shaped below-floor evidence below).
    """
    from embodied.memory import world as world_module

    store = world_module.MapStore(
        world_module.MapConfig.from_scene(_SCENE),
        submap_id="submap-frontier-flyband",
        nav_epoch=_SCENE["nav_epoch"],
    )
    if surfaces is None:
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


def _occupied_z_faces(store):
    """The floor and ceiling the map's own occupied evidence states."""
    occupied = store.occupied_cells(now_ns=NOW_NS)
    centres_z = [store.config.cell_center(cell)[2] for cell in occupied]
    half_face = store.config.voxel_m / 2.0
    return max(centres_z) - half_face, min(centres_z) + half_face


class _StubAlignment:
    def __init__(self, origin):
        self.sealed = True
        self._origin = origin

    def aligned_position_ned(self, point):
        return self._origin


class _Runtime(MissionRuntime):
    """The runtime's map half plus the mission frame the band derives from."""

    def __init__(
        self,
        store,
        here,
        now_ns,
        *,
        origin=ORIGIN_ON_FIXTURE_FLOOR,
        hover=DECLARED_HOVER_ALTITUDE_M,
        sealed=True,
    ):
        self.store = store
        self._here = here
        self._now = now_ns
        self.result = type("R", (), {"log": []})()
        self.alignment = _StubAlignment(origin)
        self.alignment.sealed = sealed
        self.settings = SimpleNamespace(hover_altitude_m=hover)
        # What resolve_targets reads beyond the map and the frame.
        self._grounded = {}
        self._candidate_observation_ids = []
        self._last_observation_id = "obs-1"

    def _position_odom(self):
        return self._here

    def _now_ns(self):
        return self._now


def _goal_region(point):
    return GE.approach_region(point, ENVELOPE, direction=FRONTIER_VIEW_DIRECTION)


def _band_of(runtime):
    band = runtime._flyable_band_z()
    assert band is not None
    return band


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


def test_the_band_is_the_declared_hover_band_tightened_by_the_map():
    """Closed-form: hover band from the declared altitude, faces from the map."""
    runtime = _Runtime(STORE, HERE, NOW_NS)
    hover_z = ORIGIN_ON_FIXTURE_FLOOR[2] - DECLARED_HOVER_ALTITUDE_M
    half_region = GE.approach_region((0.0, 0.0, 0.0), ENVELOPE).extent()[2] / 2.0
    floor_face, ceiling_face = _occupied_z_faces(STORE)
    expected_low = max(hover_z - half_region, ceiling_face + half_region)
    expected_high = min(hover_z + half_region, floor_face - half_region)
    assert _band_of(runtime) == (expected_low, expected_high)
    # The declared altitude is inside the band the mission derives from it.
    assert expected_low <= hover_z <= expected_high


def test_a_partial_map_leaves_the_declared_band_intact():
    """An eye-level wall patch is not a ceiling: no evidence beyond the band."""
    patch = [
        _FIXTURE.DeclaredSurface(
            "eye_level_patch",
            (4.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            {"y": (-0.2, 0.2), "z": (1.0, 1.4)},
        )
    ]
    store = _store(patch)
    occupied_z = [
        store.config.cell_center(c)[2] for c in store.occupied_cells(now_ns=NOW_NS)
    ]
    assert occupied_z, "the synthetic patch must be observed"
    runtime = _Runtime(store, HERE, NOW_NS)
    hover_z = ORIGIN_ON_FIXTURE_FLOOR[2] - DECLARED_HOVER_ALTITUDE_M
    half_region = GE.approach_region((0.0, 0.0, 0.0), ENVELOPE).extent()[2] / 2.0
    # The patch's evidence stays inside the hover band, so neither side of the
    # band tightens: the map's span is not mistaken for the room's.
    assert min(occupied_z) - store.config.voxel_m / 2.0 > hover_z - half_region
    assert max(occupied_z) + store.config.voxel_m / 2.0 < hover_z + half_region
    assert _band_of(runtime) == (hover_z - half_region, hover_z + half_region)


def test_an_unstated_band_is_none_and_the_runtime_half_states_nothing():
    """No mission frame (or the offline half-runtime): no band, so no gate."""
    bare = _Runtime(STORE, HERE, NOW_NS)
    del bare.alignment
    del bare.settings
    assert bare._flyable_band_z() is None
    unsealed = _Runtime(STORE, HERE, NOW_NS, sealed=False)
    assert unsealed._flyable_band_z() is None


# ---------------------------------------------------------------------------
# The gate on the walk
# ---------------------------------------------------------------------------


def test_a_floor_to_ceiling_cluster_yields_only_in_band_vantages():
    """The cluster spans the room; the offered vantage and region do not."""
    runtime = _Runtime(STORE, HERE, NOW_NS)
    band = _band_of(runtime)
    floor_face, ceiling_face = _occupied_z_faces(STORE)
    tall = GE.BoxRegion(
        low=(2.6, -0.6, ceiling_face),
        high=(4.0, 0.6, floor_face + 0.05),
        label="frontier",
    )
    searchable = runtime._searchable_cells(HERE)
    point = runtime._frontier_goal_point(
        tall, here=HERE, searchable=searchable, band=band
    )
    assert point is not None, "the tall cluster has an in-band vantage"
    assert band[0] <= point[2] <= band[1]
    region = _goal_region(point)
    assert ceiling_face <= region.low[2] and region.high[2] <= floor_face, (
        "the published region's z band must stay between the map's own faces"
    )


def test_the_J54_shape_is_clamped_to_the_band_not_dived_into():
    """The regression pin: a deep cluster like J54's admitted regions.

    J54's admitted explore regions carried z bands at and below the floor
    datum (``z[-0.96, 0.14]``, ``z[-0.30, 0.80]`` in the mission frame). This
    region is that shape in the fixture frame: its z band reaches past the
    floor face the map states. The ungated walk (what flew in J54) accepts a
    vantage below the floor face; the gated walk refuses every such candidate
    (``out_of_band``) and answers only inside the band.
    """
    runtime = _Runtime(STORE, HERE, NOW_NS)
    band = _band_of(runtime)
    floor_face, _ = _occupied_z_faces(STORE)
    deep = GE.BoxRegion(
        low=(3.4, 0.2, floor_face - 0.05),
        high=(4.6, 1.4, floor_face + 0.55),
        label="frontier",
    )
    assert deep.center()[2] > band[1], "the J54 shape's centre is out of band"
    searchable = runtime._searchable_cells(HERE)
    ungated = runtime._frontier_goal_point(
        deep, here=HERE, searchable=searchable, band=None
    )
    assert ungated is not None, "the ungated walk (J54's code) accepts a vantage"
    assert ungated[2] > band[1], "the J54 code's vantage sits below the floor face"
    runtime._gate_counts = {}
    gated = runtime._frontier_goal_point(
        deep, here=HERE, searchable=searchable, band=band
    )
    assert gated is not None, "an in-band vantage for this frontier exists"
    assert band[0] <= gated[2] <= band[1]
    assert runtime._gate_counts.get("out_of_band", 0) >= 1
    region = _goal_region(gated)
    floor_face2, ceiling_face = _occupied_z_faces(STORE)
    assert ceiling_face <= region.low[2] and region.high[2] <= floor_face2


def test_a_frontier_with_no_in_band_vantage_is_refused_with_its_count():
    """Every vantage the walk can express is out of band: refused, named.

    The cluster hangs above the ceiling face the map states and behind the
    aircraft. Out-of-band candidates are refused by the band gate before
    anything else is asked; the candidates that survive into the band are all
    within the aircraft's own standoff, so the walk has no vantage it may
    return. The refusal carries its named gate counts.
    """
    runtime = _Runtime(STORE, HERE, NOW_NS)
    band = _band_of(runtime)
    _, ceiling_face = _occupied_z_faces(STORE)
    above_ceiling = GE.BoxRegion(
        low=(-3.6, -0.6, -2.0),
        high=(-2.4, 0.6, -1.2),
        label="frontier",
    )
    assert above_ceiling.center()[2] < band[0]
    point = runtime._frontier_goal_point(
        above_ceiling,
        here=HERE,
        searchable=runtime._searchable_cells(HERE),
        band=band,
    )
    assert point is None
    assert runtime._gate_counts.get("out_of_band", 0) >= 1
    assert (
        runtime._gate_counts.get("too_near", 0) >= 1
        or runtime._gate_counts.get("too_close", 0) >= 1
    )


class _RefusedRuntime(_Runtime):
    """The runtime whose only frontier is the deep one resolution resolves."""

    def __init__(self, store, here, now_ns, region):
        super().__init__(store, here, now_ns)
        self._region = region

    def frontier_regions(self):
        return {"frontier:deep": self._region}


def test_resolution_holds_no_target_for_a_frontier_the_band_refuses():
    """The centre fallback is gone: a refused frontier resolves to nothing."""
    above_ceiling = GE.BoxRegion(
        low=(-3.6, -0.6, -2.0),
        high=(-2.4, 0.6, -1.2),
        label="frontier",
    )
    runtime = _RefusedRuntime(STORE, HERE, NOW_NS, above_ceiling)
    from embodied.contracts import records as R

    targets = runtime.resolve_targets(
        R.SpatialGoal(
            proposal_id="p-deep",
            request_id=None,
            fingerprint="fp",
            mission_revision=1,
            base_goal_revision=0,
            selection_ids=(),
            target_refs=("frontier:deep",),
            intent="explore",
            constraints=(),
            completion_condition="reached",
            lease_bounds=(("step_lease_s", 60.0),),
            local_discretion_bounds=(),
        )
    )
    assert targets == ()
    joined = "\n".join(runtime.result.log)
    assert "has no admissible vantage" in joined
    assert "out_of_band" in joined


# ---------------------------------------------------------------------------
# The offering path and the other z assemblies
# ---------------------------------------------------------------------------


def test_navigable_frontiers_offer_only_in_band_vantages():
    """Whatever the listing offers, the walk has already gated onto the band."""
    runtime = _Runtime(STORE, HERE, NOW_NS)
    band = _band_of(runtime)
    searchable = runtime._searchable_cells(HERE)
    regions = runtime.frontier_regions()
    offered = runtime.navigable_frontiers()
    assert offered, "the fixture map holds frontiers, or this proves nothing"
    for ref in offered:
        point = runtime._frontier_goal_point(
            regions[ref], here=HERE, searchable=searchable, band=band
        )
        assert point is not None, ref
        assert band[0] <= point[2] <= band[1], ref


def test_the_return_target_and_the_renewal_hold_compose_with_the_band():
    """The two assemblies that already target the declared altitude."""
    from embodied.contracts import records as R

    runtime = _Runtime(STORE, HERE, NOW_NS)
    band = _band_of(runtime)
    targets = runtime.resolve_targets(
        R.SpatialGoal(
            proposal_id="p-start",
            request_id=None,
            fingerprint="fp",
            mission_revision=1,
            base_goal_revision=0,
            selection_ids=(),
            target_refs=("start",),
            intent="return",
            constraints=(),
            completion_condition="settled at the start position",
            lease_bounds=(("step_lease_s", 60.0),),
            local_discretion_bounds=(),
        )
    )
    assert len(targets) == 1
    hover_z = ORIGIN_ON_FIXTURE_FLOOR[2] - DECLARED_HOVER_ALTITUDE_M
    assert targets[0].geometry[2] == hover_z
    assert band[0] <= targets[0].geometry[2] <= band[1]
    # The renewal hold climbs to the same declared altitude and never deeper.
    hold = runtime._renewal_hold_position((0.4, -0.2, hover_z + 1.0))
    assert hold[2] == hover_z
    assert band[0] <= hold[2] <= band[1]
