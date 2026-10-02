"""An explore goal must be somewhere to go.

A frontier is a boundary between observed free space and unknown space: an
observation opportunity, not a destination (specification 11). The executor
builds each goal's terminal region as ``approach_region(target)``, which sits
``STANDOFF_M`` *behind* the target — so a goal whose region already contains the
aircraft is satisfied the instant it is admitted. The certified prefix is then a
hover, the step completes on arrival without moving, and the map never grows.

That is not a hypothesis about live-17: its episode records 19 setpoints, every
one commanding the aircraft's own x and y with zero velocity, while every
certificate named an "explore" frontier. A frontier the aircraft is already
standing on is therefore not a place to fly to, and ``navigable_frontiers``
must not offer one.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from embodied.navigation import geometry as GE
from embodied.platform.mission_runtime import ENVELOPE, FRONTIER_VIEW_DIRECTION, MissionRuntime

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
        submap_id="submap-p11-frontier",
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
    """The runtime's map half, without the platform configuration.

    Constructed the way the queue tests construct theirs: the methods under
    test need a store and a position, not a simulator.
    """

    def __init__(self, store, here, now_ns):
        self.store = store
        self._here = here
        self._now = now_ns
        self.result = type("R", (), {"log": []})()

    def _position_odom(self):
        return self._here

    def _now_ns(self):
        return self._now


def _goal_region(point):
    return GE.approach_region(
        point, ENVELOPE, direction=FRONTIER_VIEW_DIRECTION
    )


def test_a_frontier_that_wraps_the_aircraft_is_not_a_place_to_fly_to():
    """The defect: a cluster whose bounding box is the aircraft's own position.

    Every candidate vantage for such a frontier lies within one standoff of the
    aircraft, and the last of them — ``here + STANDOFF_M`` — has an approach
    region that contains the aircraft. Returning that point is what turned
    "explore" into a hover.
    """
    runtime = _Runtime(STORE, HERE, NOW_NS)
    wrapping = GE.BoxRegion(
        low=(HERE[0] - 0.3, HERE[1] - 0.3, HERE[2] - 0.3),
        high=(HERE[0] + 0.3, HERE[1] + 0.3, HERE[2] + 0.3),
        label="frontier",
    )
    assert (
        runtime._frontier_goal_point(
            wrapping, here=HERE, searchable=runtime._searchable_cells(HERE)
        )
        is None
    )


def test_no_navigable_frontier_is_one_the_aircraft_already_stands_on():
    """The invariant, over whatever frontiers the map actually holds."""
    runtime = _Runtime(STORE, HERE, NOW_NS)
    searchable = runtime._searchable_cells(HERE)
    regions = runtime.frontier_regions()
    offered = runtime.navigable_frontiers()
    assert offered, "the fixture map holds frontiers, or this proves nothing"
    for ref in offered:
        point = runtime._frontier_goal_point(
            regions[ref], here=HERE, searchable=searchable
        )
        assert point is not None, ref
        assert not _goal_region(point).contains(HERE), (
            f"{ref} is admitted-and-completed in the same instant"
        )
        assert float(np.linalg.norm(np.asarray(point) - np.asarray(HERE))) >= (
            GE.STANDOFF_M
        ), ref


def test_navigable_frontiers_are_offered_farthest_vantage_first():
    """The known map is what has been seen; the evidence is beyond its boundary."""
    runtime = _Runtime(STORE, HERE, NOW_NS)
    searchable = runtime._searchable_cells(HERE)
    regions = runtime.frontier_regions()
    distances = [
        float(
            np.linalg.norm(
                np.asarray(
                    runtime._frontier_goal_point(
                        regions[ref], here=HERE, searchable=searchable
                    )
                )
                - np.asarray(HERE)
            )
        )
        for ref in runtime.navigable_frontiers()
    ]
    assert distances == sorted(distances, reverse=True)


def test_the_offered_order_is_not_the_voxel_order():
    """Voxel order offers the cluster beside the aircraft first, which is the
    cluster that produced the hover. The offered order must differ from it
    whenever more than one frontier is navigable."""
    runtime = _Runtime(STORE, HERE, NOW_NS)
    offered = runtime.navigable_frontiers()
    if len(offered) < 2:
        return
    assert offered != tuple(runtime.frontier_regions())


def test_the_candidate_the_old_rule_accepted_lands_the_goal_on_the_aircraft():
    """Why the gate exists, stated as the geometry that produced it.

    The last candidate of the walk is ``here + STANDOFF_M``. Its approach
    region has supported cells, so the old rule — "some cell of the goal region
    is in the searchable set" — returned it. That region contains the aircraft,
    so the goal was complete on admission.
    """
    target = (HERE[0] + GE.STANDOFF_M, HERE[1], HERE[2])
    assert _goal_region(target).contains(HERE)
