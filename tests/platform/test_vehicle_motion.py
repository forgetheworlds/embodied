"""Sim proof for the vehicle motion layer — three tasks, real Webots + SITL.

1. takeoff to hover
2. goto waypoint A and hold (residual ≤ 0.15 m)
3. goto waypoint B and hold, then land

Skipped when Webots / SITL are not installed on the host. Run explicitly::

    python -m embodied motion-proof --config configs/first_indoor.yaml \\
        --output work/runs/motion-proof-1
"""

from __future__ import annotations

from pathlib import Path

import pytest

from embodied.cli import CommandStatus, GateStatus, load_config, repository_root
from embodied.contracts.records import SensorMode
from embodied.platform import vehicle_proof
from embodied.platform.webots_ardupilot import PlatformSettings, check_prerequisites

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "first_indoor.yaml"


def _sim_ready() -> bool:
    document = load_config(CONFIG)
    settings = PlatformSettings.from_config(
        document, root=repository_root(), arm=SensorMode.SIMULATOR_INTERFACE.value
    )
    return all(item.satisfied for item in check_prerequisites(settings, REPO / "work"))


@pytest.mark.skipif(not _sim_ready(), reason="Webots + ArduPilot SITL not installed")
def test_vehicle_flies_three_preset_waypoint_tasks(tmp_path):
    """Launch the scene and fly takeoff → waypoint A → waypoint B through Vehicle."""
    args = type("Args", (), {"config": CONFIG})()
    output = tmp_path / "motion-proof"
    outcome = vehicle_proof._command(args, output)
    assert outcome.status == CommandStatus.COMPLETE, outcome.reasons
    assert outcome.gate_status == GateStatus.PASS, outcome.reasons
    steps = outcome.manifest["steps"]
    assert [step["task"] for step in steps] == [
        "takeoff",
        "goto[0]",
        "goto[1]",
        "land",
    ]
    assert all(step["ok"] for step in steps)
