"""Sim proof for the vehicle motion layer — open field, real Webots + SITL.

1. takeoff to hover
2. goto each motion waypoint and hold (residual ≤ configured limit)
3. land

Skipped when Webots / SITL are not installed on the host. Run explicitly::

    python -m embodied motion-proof --config configs/motion_open.yaml \\
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
CONFIG = REPO / "configs" / "motion_open.yaml"


def _sim_ready() -> bool:
    document = load_config(CONFIG)
    settings = PlatformSettings.from_config(
        document, root=repository_root(), arm=SensorMode.SIMULATOR_INTERFACE.value
    )
    return all(item.satisfied for item in check_prerequisites(settings, REPO / "work"))


@pytest.mark.skipif(not _sim_ready(), reason="Webots + ArduPilot SITL not installed")
def test_vehicle_flies_open_field_motion_route(tmp_path):
    """Launch the open field and fly the full motion waypoint route through Vehicle."""
    args = type("Args", (), {"config": CONFIG})()
    output = tmp_path / "motion-proof"
    outcome = vehicle_proof._command(args, output)
    assert outcome.status == CommandStatus.COMPLETE, outcome.reasons
    assert outcome.gate_status == GateStatus.PASS, outcome.reasons
    steps = outcome.manifest["steps"]
    motion = load_config(CONFIG)["motion"]
    waypoint_count = len(motion["waypoints_local_ned"])
    assert [step["task"] for step in steps] == [
        "takeoff",
        *[f"goto[{index}]" for index in range(waypoint_count)],
        "land",
    ]
    assert all(step["ok"] for step in steps)
    assert "open_field" in str(outcome.manifest["world"])
