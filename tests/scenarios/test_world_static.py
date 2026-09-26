"""Static gates over the listed worlds: existence, text parsing and determinism pins.

Everything here is read from the file on disk — no simulator is started. The headless
load and the device-exposure check live in
scenarios/missions/tools/validate_scenarios.py; the catalogue's own internal
consistency is tested in test_catalogue.py.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = REPO_ROOT / "scenarios" / "missions" / "catalogue.yaml"

# Plan section 6: these are the pin values, asserted independently of the catalogue so
# a catalogue edited down to weaker pins cannot pass its own gate.
EXPECTED_PINS = {
    "header": "#VRML_SIM R2025a utf8",
    "basic_time_step_ms": 2,
    "optimal_thread_count": 1,
    "random_seed_min": 0,
    "remote_externproto_allowed": False,
}
REMOTE_SCHEMES = ("http://", "https://", "webots://")


def catalogue():
    with CATALOGUE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def world_text(entry):
    path = REPO_ROOT / entry["world"]
    assert path.is_file(), f"missing world file for {entry['id']}: {path}"
    return path.read_text(encoding="utf-8")


def body_without_comments(text):
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_catalogue_pins_are_the_required_values():
    pins = catalogue()["pins"]
    for key, expected in EXPECTED_PINS.items():
        assert pins[key] == expected, key


def test_every_listed_world_exists_and_parses_as_text():
    for entry in catalogue()["scenarios"]:
        text = world_text(entry)
        body = body_without_comments(text)
        # A structural parse: delimiters must balance outside comments.
        assert body.count("{") == body.count("}"), entry["id"]
        assert body.count("[") == body.count("]"), entry["id"]


def test_every_world_carries_the_header_and_worldinfo_pins():
    pins = catalogue()["pins"]
    for entry in catalogue()["scenarios"]:
        text = world_text(entry)
        assert text.splitlines()[0] == pins["header"], entry["id"]
        match = re.search(r"WorldInfo\s*\{[^}]*\}", text)
        assert match, f"{entry['id']} has no WorldInfo block"
        block = match.group(0)
        step = re.search(r"basicTimeStep\s+(-?\d+)", block)
        threads = re.search(r"optimalThreadCount\s+(-?\d+)", block)
        seed = re.search(r"randomSeed\s+(-?\d+)", block)
        assert step and int(step.group(1)) == pins["basic_time_step_ms"], entry["id"]
        assert threads and int(threads.group(1)) == pins["optimal_thread_count"], entry["id"]
        assert seed and int(seed.group(1)) >= pins["random_seed_min"], entry["id"]


def test_no_world_declares_a_remote_externproto():
    for entry in catalogue()["scenarios"]:
        text = world_text(entry)
        externprotos = re.findall(r'EXTERNPROTO\s+"([^"]+)"', text)
        assert externprotos, f"{entry['id']} declares no EXTERNPROTO at all"
        for reference in externprotos:
            assert not reference.startswith(REMOTE_SCHEMES), (
                f"{entry['id']} references a remote proto: {reference}"
            )
            assert reference.startswith("."), (
                f"{entry['id']} EXTERNPROTO is not a relative path: {reference}"
            )


def test_every_world_declares_its_device_set():
    for entry in catalogue()["scenarios"]:
        text = world_text(entry)
        for device in entry["devices"]:
            assert device in text, f"{entry['id']} does not declare device {device!r}"
