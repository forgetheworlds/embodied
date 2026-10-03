"""The declared flight values match the constants currently in force.

The values that decide whether a flight moves were module-level constants in the
code with no declared home, so a reader could not see them without reading every
module in the causal path. They are now declared in ``configs/first_indoor.yaml``
and named in ``work/runs/p05/J41-schema-REPORT.md``.

**The code does not yet READ them.** The wiring is a separate change, held
behind sibling missions that own ``navigation/``, ``platform/`` and ``memory/``
tonight. This file is the bridge between the declaration and the constant: it
fails the moment one is edited without the other moving with it, which is the
drift R2 exists to prevent.

When the handoff lands, the constants lose their authority and these comparisons
become vacuous. That is the point at which this file should be replaced by one
that asserts the shipped configuration supplies the values the code uses.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from embodied.cli import ConfigError, load_config
from embodied.navigation import geometry as geometry_module
from embodied.navigation import planner as planner_module
from embodied.platform import mission_runtime as runtime

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "first_indoor.yaml"

DECLARED_SECTIONS = ("vehicle", "planner", "mission", "map")

def _same(declared_value: object, constant: object) -> bool:
    """Whether a declared value and the constant behind it are the same number.

    Floats are compared with a tolerance rather than bitwise, and the reason is
    this file's first real finding: ``ERROR_ALLOWANCE_M`` is written as the
    product ``3.0 * ESTIMATOR_HEALTHY_SIGMA_MAX_M`` and evaluates to
    0.15000000000000002, one float ulp from the 0.15 a config author would write.
    That ulp is not the drift R2 cares about, and demanding bit equality would
    force the configuration to carry a float artefact to satisfy a test.
    """
    if isinstance(declared_value, float) or isinstance(constant, float):
        return math.isclose(
            float(declared_value), float(constant), rel_tol=1e-12, abs_tol=1e-12
        )
    return declared_value == constant

@pytest.fixture(scope="module")
def declared() -> dict:
    return load_config(CONFIG)


def test_every_declared_value_matches_the_constant_in_force(declared: dict) -> None:
    """One table, so a drift shows up as a named key rather than a mystery.

    Each row is (section, key, the constant it is declared against). The walk is
    deliberately explicit rather than reflective: a reader should be able to see
    which source value each declared key stands for. A key added to the
    configuration without a row here fails ``test_no_declared_key_is_unpinned``.
    """
    rows: list[tuple[str, str, object]] = [
        ("vehicle", "body_radius_m", runtime.ENVELOPE.body_radius_m),
        ("vehicle", "error_allowance_m", runtime.ERROR_ALLOWANCE_M),
        ("planner", "certificate_margin_voxels", planner_module.CERTIFICATE_MARGIN_VOXELS),
        ("planner", "search_margin_m", planner_module.SEARCH_MARGIN_M),
        ("planner", "standoff_m", geometry_module.STANDOFF_M),
        ("planner", "approach_half_thickness_m", geometry_module.APPROACH_HALF_THICKNESS_M),
        ("planner", "crossing_half_depth_m", geometry_module.CROSSING_HALF_DEPTH_M),
        ("planner", "exit_depth_m", geometry_module.EXIT_DEPTH_M),
        ("mission", "budget_sim_s", runtime.MISSION_BUDGET_SIM_S),
        ("mission", "cold_start_perception_sim_s", runtime.COLD_START_PERCEPTION_SIM_S),
        ("mission", "min_perception_interval_s", runtime.MIN_PERCEPTION_INTERVAL_S),
        ("mission", "frontier_cluster_cells", runtime.FRONTIER_CLUSTER_CELLS),
        ("mission", "frontier_vantage_step_m", runtime.FRONTIER_VANTAGE_STEP_M),
        ("mission", "settle_speed_mps", runtime.SETTLE_SPEED_MPS),
        ("mission", "settle_hold_s", runtime.SETTLE_HOLD_S),
        ("map", "voxel_m", runtime.MAP_PARAMETERS["voxel_m"]),
        ("map", "surface_band_m", runtime.MAP_PARAMETERS["surface_band_m"]),
        ("map", "log_odds_hit", runtime.MAP_PARAMETERS["log_odds_hit"]),
        ("map", "log_odds_pass", runtime.MAP_PARAMETERS["log_odds_pass"]),
        ("map", "clamp", runtime.MAP_PARAMETERS["clamp"]),
        ("map", "free_threshold", runtime.MAP_PARAMETERS["free_threshold"]),
        ("map", "occupied_threshold", runtime.MAP_PARAMETERS["occupied_threshold"]),
        ("map", "min_clearing_rays", runtime.MAP_PARAMETERS["min_clearing_rays"]),
        ("map", "freshness_s", runtime.MAP_PARAMETERS["freshness_s"]),
    ]
    drifted = [
        f"{section}.{key}: declared {declared[section][key]!r}, constant {constant!r}"
        for section, key, constant in rows
        if not _same(declared[section][key], constant)
    ]
    assert not drifted, "the declaration and the code disagree:\n  " + "\n  ".join(drifted)


def test_the_map_bounds_are_declared_and_match_the_runtime(declared: dict) -> None:
    """Per axis, because the z pair is the bound a climbing vehicle runs into.

    A tuple in the runtime and a list in the configuration are the same numbers;
    the comparison normalises rather than making the author pick a YAML shape for
    a Python type.
    """
    declared_bounds = declared["map"]["bounds_odom_m"]
    runtime_bounds = runtime.MAP_PARAMETERS["bounds_odom_m"]
    assert set(declared_bounds) == set(runtime_bounds) == {"x", "y", "z"}
    for axis in "xyz":
        assert tuple(declared_bounds[axis]) == tuple(runtime_bounds[axis]), axis


def test_no_declared_key_is_unpinned(declared: dict) -> None:
    """Every key in the four sections is pinned by the table above.

    This stops the table going stale in the other direction: a key added to the
    configuration with no constant behind it would otherwise be declared and
    never checked.
    """
    pinned = {
        "vehicle": {"body_radius_m", "error_allowance_m"},
        "planner": {
            "certificate_margin_voxels",
            "search_margin_m",
            "standoff_m",
            "approach_half_thickness_m",
            "crossing_half_depth_m",
            "exit_depth_m",
        },
        "mission": {
            "budget_sim_s",
            "cold_start_perception_sim_s",
            "min_perception_interval_s",
            "frontier_cluster_cells",
            "frontier_vantage_step_m",
            "settle_speed_mps",
            "settle_hold_s",
        },
        "map": {
            "voxel_m",
            "bounds_odom_m",
            "surface_band_m",
            "log_odds_hit",
            "log_odds_pass",
            "clamp",
            "free_threshold",
            "occupied_threshold",
            "min_clearing_rays",
            "freshness_s",
        },
    }
    for section, expected in pinned.items():
        assert set(declared[section]) == expected, section


def test_the_error_allowance_keeps_its_derivation(declared: dict) -> None:
    """The allowance is 3 x the estimator's sigma floor, not a round number.

    Written as the product it is, so the value cannot drift away from the
    measurement behind it (R2). Measured: the estimator's own feed never reports
    a sigma below 0.0500 m across 4 087 in-flight samples, so 3 x 0.05 is the
    rig's floor rather than a conservative choice.
    """
    assert runtime.ERROR_ALLOWANCE_M == pytest.approx(
        3.0 * runtime.ESTIMATOR_HEALTHY_SIGMA_MAX_M
    )
    assert declared["vehicle"]["error_allowance_m"] == pytest.approx(
        runtime.ERROR_ALLOWANCE_M
    )


def test_a_configuration_predating_these_sections_still_loads(
    declared: dict, tmp_path: Path
) -> None:
    """The sections are optional, so the compatibility gate's file still loads.

    ``_Optional`` exists for exactly this: a stage can declare a section before
    the shared schema knows about it, and a configuration that predates it must
    keep loading. Without that, adding a section here would break every command
    that reads another configuration, which is the failure recorded as
    LEARNED-FAILURES defect 23.
    """
    without = {
        key: value for key, value in declared.items() if key not in DECLARED_SECTIONS
    }
    path = tmp_path / "predates.yaml"
    path.write_text(yaml.safe_dump(without), encoding="utf-8")
    reloaded = load_config(path)
    for section in DECLARED_SECTIONS:
        assert section not in reloaded


def test_an_unknown_key_is_still_rejected(declared: dict, tmp_path: Path) -> None:
    """Adding sections does not relax the schema's rejection of unknown keys.

    That rejection is what makes a typo a failure instead of a silently ignored
    section, and a silently ignored section is a substituted experiment.
    """
    assert declared  # the shipped file loads, so the rejection is not blanket
    text = CONFIG.read_text(encoding="utf-8")

    top = tmp_path / "top.yaml"
    top.write_text(text + "\nnot_a_section:\n  x: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys: not_a_section"):
        load_config(top)

    inner = tmp_path / "inner.yaml"
    inner.write_text(
        text.replace("  voxel_m: 0.1", "  voxel_m: 0.1\n  not_a_key: 1"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"map has unknown keys: not_a_key"):
        load_config(inner)
