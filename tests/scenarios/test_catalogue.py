"""Catalogue consistency: what scenarios/missions/catalogue.yaml says about itself.

These tests parse the catalogue only — the worlds themselves are checked in
test_world_static.py, and the headless load check lives in
scenarios/missions/tools/validate_scenarios.py. Nothing here starts Webots.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = REPO_ROOT / "scenarios" / "missions" / "catalogue.yaml"

GROUPS = ("dev", "holdout")
# Six axes live under ``variations``; the seventh (target object class and placement,
# plan section 3 axis 7) is the ``mission`` block, asserted below.
VARIATION_AXES = (
    "rooms",
    "doorways",
    "corridor_turns",
    "obstacles",
    "textures",
    "lighting",
)
# Plan section 3: the families are disjoint between the groups, so a held-out result
# cannot be an artefact of one layout family.
DEV_WALL_TEXTURES = {"Plaster", "Roughcast", "PaintedWood"}
HOLDOUT_WALL_TEXTURES = {"RedBricks", "Marble"}
DEV_FLOOR_TEXTURES = {"Parquetry", "DarkParquetry", "ChequeredParquetry"}
HOLDOUT_FLOOR_TEXTURES = {"CementTiles", "PorcelainChevronTiles"}


def load_catalogue():
    with CATALOGUE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def entries_in(catalogue, group):
    return [entry for entry in catalogue["scenarios"] if entry["group"] == group]


def test_catalogue_declares_both_groups_with_four_each():
    catalogue = load_catalogue()
    assert sorted(catalogue["groups"]) == sorted(GROUPS)
    for group in GROUPS:
        assert len(entries_in(catalogue, group)) == catalogue["groups"][group]


def test_scenario_ids_are_unique_and_name_their_directories():
    catalogue = load_catalogue()
    ids = [entry["id"] for entry in catalogue["scenarios"]]
    assert len(ids) == len(set(ids))
    for entry in catalogue["scenarios"]:
        world = Path(entry["world"])
        assert entry["group"] in world.parts, entry["id"]
        assert entry["id"] in world.parts, entry["id"]


def test_world_paths_follow_the_group_directory_layout():
    for entry in load_catalogue()["scenarios"]:
        expected = Path("scenarios/missions") / entry["group"] / entry["id"] / "world.wbt"
        assert Path(entry["world"]) == expected


def test_every_entry_carries_the_required_sections():
    for entry in load_catalogue()["scenarios"]:
        assert entry["id"]
        assert entry["group"] in GROUPS
        mission = entry["mission"]
        for key in ("type", "instruction_family", "target_class", "target_present"):
            assert mission[key] is not None, entry["id"]
        variations = entry["variations"]
        for axis in VARIATION_AXES:
            assert axis in variations, f"{entry['id']} does not declare axis {axis}"
        assert isinstance(variations["obstacles"]["count"], int)
        assert variations["obstacles"]["count"] >= 2
        assert len(variations["spawn_pose"]) == 3


def test_declared_device_set_is_identical_complete_and_distinct():
    catalogue = load_catalogue()
    reference = None
    for entry in catalogue["scenarios"]:
        devices = entry["devices"]
        assert devices == list(dict.fromkeys(devices)), entry["id"]  # no duplicates
        cameras = [d for d in devices if d.startswith("camera ")]
        assert len(cameras) == 2 and len(set(cameras)) == 2, entry["id"]
        for required in ("accelerometer", "gyro", "inertial unit", "gps"):
            assert required in devices, f"{entry['id']} misses {required}"
        motors = [d for d in devices if d.startswith("m") and d.endswith("_motor")]
        assert len(motors) == 4, entry["id"]
        if reference is None:
            reference = devices
        assert devices == reference, entry["id"]


def test_dev_and_holdout_are_disjoint_on_the_declared_families():
    catalogue = load_catalogue()
    dev = entries_in(catalogue, "dev")
    holdout = entries_in(catalogue, "holdout")
    for axis, dev_pool, holdout_pool in (
        ("wall", DEV_WALL_TEXTURES, HOLDOUT_WALL_TEXTURES),
        ("floor", DEV_FLOOR_TEXTURES, HOLDOUT_FLOOR_TEXTURES),
    ):
        dev_values = {e["variations"]["textures"][axis] for e in dev}
        holdout_values = {e["variations"]["textures"][axis] for e in holdout}
        assert dev_values <= dev_pool and holdout_values <= holdout_pool
        assert not dev_values & holdout_values
    assert all(e["variations"]["lighting"]["directional_intensity"] >= 1.0 for e in dev)
    assert all(e["variations"]["lighting"]["directional_intensity"] <= 0.6 for e in holdout)
    dev_classes = {e["mission"]["target_class"] for e in dev}
    assert dev_classes <= {"box", "sphere", "none"}
    for entry in holdout:
        if entry["mission"]["target_present"]:
            assert entry["mission"]["target_class"] == "cylinder", entry["id"]


def test_held_out_group_is_at_least_as_large_as_the_development_group():
    catalogue = load_catalogue()
    assert len(entries_in(catalogue, "holdout")) >= len(entries_in(catalogue, "dev"))
