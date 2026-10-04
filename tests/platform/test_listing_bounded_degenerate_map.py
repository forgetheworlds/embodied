"""The frontier listing must answer in bounded work on any map state.

J51-move-2 (2026-10-04): the aircraft sat off the map, ``reachable_from``
refused its cell, and every traversable cell was in-grid -- so the reachable
set was provably empty. The listing walked all 261 frontier regions anyway,
materialized each candidate vantage's approach-cell tuple to test membership
in an empty set, and stalled the mission loop past its 90 s beat bound. The
answer never needed the walk: an empty reachable set admits nothing.

The pins here are work counts, not host timing (T18): on the degenerate state
the listing touches no region's cells at all, and on the healthy carved map
the membership tests that replaced the materializing one answer exactly what
``BoxRegion.cells`` answered.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Iterator

from embodied.navigation import geometry as GE
from embodied.platform.mission_runtime import (
    MissionRuntime,
    _index_bounds,
    _region_holds_searchable,
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


def _store():
    """The doorway fixture's own map: real carving, real free space."""
    from embodied.memory import world as world_module

    store = world_module.MapStore(
        world_module.MapConfig.from_scene(_SCENE),
        submap_id="submap-listing-wall",
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
# J51 measured 261 regions against its empty reachable set; 512 holds the
# shape of that state with room to spare while staying hermetic.
DEGENERATE_REGION_COUNT = 512


class _Runtime(MissionRuntime):
    """The runtime's map half, without the platform configuration.

    Constructed the way the queue tests construct theirs. ``regions``
    replaces the cluster walk so a test can hold many regions without
    carving a map for each; ``None`` keeps the production clustering.
    """

    def __init__(self, store, here, now_ns, regions=None):
        self.store = store
        self._here = here
        self._now = now_ns
        self.result = type("R", (), {"log": []})()
        self._regions = regions

    def _position_odom(self):
        return self._here

    def _now_ns(self):
        return self._now

    def frontier_regions(self):
        if self._regions is not None:
            return dict(self._regions)
        return super().frontier_regions()


def _off_map_position(config):
    """A position outside the map on every axis: no cell of it is traversable."""
    return tuple(config.bounds_odom_m[axis][1] + 10.0 for axis in ("x", "y", "z"))


def _many_regions(config, count):
    """Frontier-like terminal regions spread wider than the map itself.

    Each is a box a few voxels per side, laid out on a lattice so the listing
    has as many regions to decline as J51's 261-cluster map held.
    """
    voxel = config.voxel_m
    regions = {}
    lattice = 8
    spacing = 24
    for index in range(count):
        gx = (index % lattice) * spacing
        gy = ((index // lattice) % lattice) * spacing
        gz = (index // (lattice * lattice)) * spacing
        half = 2.0 * voxel
        low = (gx * voxel - half, gy * voxel - half, gz * voxel - half)
        high = (gx * voxel + half, gy * voxel + half, gz * voxel + half)
        regions[f"frontier:{gx}:{gy}:{gz}"] = GE.BoxRegion(
            low=low, high=high, label="frontier"
        )
    return regions


class _RegionCellCounter:
    """Counts region-cell work without changing what any call returns.

    ``materialized`` counts ``BoxRegion.cells`` calls -- each one builds the
    region's whole cell tuple before a caller can test anything. ``iterated``
    counts ``iter_cells`` calls. Everything else is pass-through.
    """

    def __init__(self):
        self.materialized = 0
        self.iterated = 0

    def __enter__(self):
        real_cells = GE.BoxRegion.cells
        real_iter = GE.BoxRegion.iter_cells
        counter = self

        def cells(region_self, config):
            counter.materialized += 1
            return real_cells(region_self, config)

        def iter_cells(region_self, config) -> Iterator[tuple[int, int, int]]:
            counter.iterated += 1
            return real_iter(region_self, config)

        self._saved = (real_cells, real_iter)
        GE.BoxRegion.cells = cells
        GE.BoxRegion.iter_cells = iter_cells
        return self

    def __exit__(self, *exc):
        GE.BoxRegion.cells, GE.BoxRegion.iter_cells = self._saved
        return False


def test_the_degenerate_listing_answers_without_walking_a_region():
    """J51's state -- off-map aircraft, empty reachable set, many regions.

    The reachable set is empty by construction, so every region's answer is
    ``None`` before its first vantage is computed. The listing must say so
    without touching a single region's cells, which is the work count the
    90 s stall burned through.
    """
    config = STORE.config
    regions = _many_regions(config, DEGENERATE_REGION_COUNT)
    off_map = _off_map_position(config)
    runtime = _Runtime(STORE, off_map, NOW_NS, regions)
    # The precondition J51 measured: reachable_from refuses the start cell.
    assert runtime._searchable_cells(off_map) == set()

    with _RegionCellCounter() as work:
        offered = runtime.navigable_frontiers()

    assert offered == ()
    assert work.materialized == 0, "the empty listing built a region cell tuple"
    assert work.iterated == 0, "the empty listing walked a region"
    assert runtime._gate_counts == {"no_searchable_cells": 1}
    assert len(runtime.result.log) == 1
    assert "no navigable frontier" in runtime.result.log[0]
    assert "'searchable_cells': 0" in runtime.result.log[0]


def test_the_degenerate_vantage_resolution_walks_no_region():
    """Resolving one frontier's vantage on the degenerate map stays bounded too.

    ``_frontier_vantage`` is the per-frontier fallback admission resolves
    with; on an empty reachable set every vantage is refused before any
    approach region's cells are needed.
    """
    config = STORE.config
    regions = _many_regions(config, DEGENERATE_REGION_COUNT)
    off_map = _off_map_position(config)
    runtime = _Runtime(STORE, off_map, NOW_NS, regions)
    ref, region = next(iter(regions.items()))

    with _RegionCellCounter() as work:
        point = runtime._frontier_vantage(region, ref=ref)

    assert point == tuple(float(v) for v in region.center())
    assert work.materialized == 0
    assert work.iterated == 0
    assert runtime._gate_counts, "the vantage walk still counted its refusals"
    assert "has no admissible vantage" in runtime.result.log[0]


def test_the_healthy_listing_materializes_no_region_tuple():
    """On the carved fixture map the listing works without building tuples.

    The membership question is answered lazily (region cells one at a time,
    or the reachable set against ``holds_cell``), so no call ever pays for a
    whole region tuple before short-circuiting.
    """
    runtime = _Runtime(STORE, HERE, NOW_NS)
    with _RegionCellCounter() as work:
        offered = runtime.navigable_frontiers()

    assert offered, "the fixture map holds frontiers, or this proves nothing"
    assert work.materialized == 0
    # Not every region is offered even on a healthy map -- a frontier the
    # aircraft already stands in has no excursion -- but the ones offered are
    # the real listing's answer, pinned by the excursion and order tests.
    assert set(offered) <= set(runtime.frontier_regions())


def test_the_membership_helpers_answer_what_cells_answered():
    """``holds_cell`` and ``iter_cells`` agree with the materialized tuple.

    Exhaustive over every region the fixture map clusters, plus probe cells
    on and around each region's index-box faces -- the disagreement the
    floor rule could in principle hide.
    """
    config = STORE.config
    runtime = _Runtime(STORE, HERE, NOW_NS)
    regions = runtime.frontier_regions()
    assert len(regions) >= 2

    for region in regions.values():
        materialized = region.cells(config)
        materialized_set = set(materialized)
        assert tuple(region.iter_cells(config)) == materialized
        index_low = config.cell_index(region.low)
        index_high = config.cell_index(region.high)
        probes = set(materialized)
        for x in (index_low[0] - 1, index_low[0], index_high[0], index_high[0] + 1):
            for y in (index_low[1] - 1, index_low[1], index_high[1], index_high[1] + 1):
                for z in (
                    index_low[2] - 1,
                    index_low[2],
                    index_high[2],
                    index_high[2] + 1,
                ):
                    cell = (x, y, z)
                    if config.inside(cell):
                        probes.add(cell)
        for cell in probes:
            assert region.holds_cell(cell, config) == (cell in materialized_set)


def test_the_index_box_pretest_skips_only_disjoint_regions():
    """The cheap pre-test refuses exactly the regions the exact test refuses.

    A region that cannot overlap the reachable set's index box is refused
    without walking a cell; a region built around a searchable cell is
    accepted, by either walk, exactly as the materialized form answers.
    """
    config = STORE.config
    runtime = _Runtime(STORE, HERE, NOW_NS)
    searchable = runtime._searchable_cells(HERE)
    assert searchable
    bounds = _index_bounds(searchable)
    assert bounds is not None and _index_bounds(set()) is None

    disjoint = GE.BoxRegion(
        low=tuple(config.bounds_odom_m[axis][1] + 50.0 for axis in ("x", "y", "z")),
        high=tuple(config.bounds_odom_m[axis][1] + 52.0 for axis in ("x", "y", "z")),
        label="frontier",
    )
    sample = sorted(searchable)[len(searchable) // 2]
    centre = config.cell_center(sample)
    wrapping = GE.BoxRegion(
        low=tuple(c - config.voxel_m for c in centre),
        high=tuple(c + config.voxel_m for c in centre),
        label="frontier",
    )

    for region in (disjoint, wrapping):
        expected = any(cell in searchable for cell in region.cells(config))
        with _RegionCellCounter() as work:
            answer = _region_holds_searchable(region, searchable, bounds, config)
        assert answer == expected
        if region is disjoint:
            assert not answer
            assert work.iterated == 0, "a disjoint region was walked anyway"
        else:
            assert answer
        assert work.materialized == 0
