"""The self-occupied exemption is derived from the sensor's near limit.

The exemption exists for the space the vehicle's own presence blinds the sensor
to. The stereo matcher cannot report depth inside the declared near bound, so no
free-space evidence can exist there in any direction — and a clearance ball
around any cell within about half a metre of the camera reaches into that space.

Measured on ``J44-fly-1``, with the exemption at the envelope radius: **zero of
the 26 neighbours were searchable**, every one of the mission's 36 setpoints was
a station hold, and **83 %** of the ball disqualifiers sat within that distance
of the camera. The aircraft could not take one step, from any number of
vantages, because the ball it had to certify was the size of the blind field it
had to certify it from.

So the radius is derived, not chosen: the near limit plus one body-diagonal
step. This file pins the derivation to the declared values rather than to a
transcription of them.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from embodied.navigation import geometry as geo
from embodied.platform import mission_runtime as runtime

REPOSITORY = Path(__file__).resolve().parents[2]
PLATFORM_CONFIG = REPOSITORY / "configs" / "first_indoor.yaml"


def _declared() -> dict:
    document = yaml.safe_load(PLATFORM_CONFIG.read_text(encoding="utf-8"))
    return document


def test_the_near_limit_is_the_depth_windows_own_near_bound():
    """The value is the sensor's, not a number invented for the exemption.

    If the depth window moves, this fails, which is the point: the exemption
    would otherwise keep compensating for a limit the config no longer declares.
    """
    bounds = _declared()["calibration"]["bounds"]
    near_bound = float(bounds["depth_range_m"][0])
    assert near_bound == runtime.SENSOR_NEAR_LIMIT_M, (
        "the declared depth window starts at "
        f"{near_bound} m and the exemption is derived from "
        f"{runtime.SENSOR_NEAR_LIMIT_M} m"
    )


def test_the_exemption_is_derived_and_not_fitted():
    """The radius is the near limit plus one body-diagonal step, or the envelope.

    One step matters because a cell one step away has its ball displaced by
    ``sqrt(3) * voxel`` from the origin's, and every cell in that step must be
    startable for the mission to move at all.
    """
    voxel = float(runtime.MAP_PARAMETERS["voxel_m"])
    envelope = runtime.ENVELOPE
    expected = max(
        envelope.inflation_m, envelope.sensor_near_limit_m + math.sqrt(3.0) * voxel
    )
    assert envelope.self_occupied_radius_m(voxel) == expected
    # With the declared values this is 0.5 + 0.173 = 0.673 m, which selects the
    # same discrete ball as the 0.675 m an independent measurement found
    # sufficient. The derivation reproduces that measurement rather than fitting
    # it.
    assert expected > envelope.inflation_m


def test_a_hull_with_no_declared_near_limit_keeps_the_old_behaviour():
    """A caller that declares nothing keeps the envelope radius, unchanged."""
    bare = geo.Envelope(body_radius_m=0.3, error_allowance_m=0.15)
    assert bare.sensor_near_limit_m == 0.0
    assert bare.self_occupied_radius_m(0.1) == bare.inflation_m
