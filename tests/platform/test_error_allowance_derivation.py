"""The error allowance is derived from a measurement, and the derivation is pinned.

R37 (2026-10-03) established why the allowance cannot simply be lowered: the
validator refuses any state whose ``3 * sigma`` exceeds it, so the allowance *is*
the largest 3-sigma pose the mission will act on — and the rig's own feed never
reports a sigma below ``ESTIMATOR_HEALTHY_SIGMA_MAX_M``. These tests fail if the
value drifts away from that measurement, or if the composition it takes part in
moves without the arithmetic being restated here.
"""

from __future__ import annotations

from embodied.navigation import planner as planner_module
from embodied.platform import mission_runtime as runtime


def test_the_allowance_is_three_sigma_of_the_measured_healthy_bound() -> None:
    """It is the product it claims to be, not a typed number."""
    assert runtime.ERROR_ALLOWANCE_M == 3.0 * runtime.ESTIMATOR_HEALTHY_SIGMA_MAX_M


def test_the_envelope_carries_the_derived_allowance() -> None:
    assert runtime.ENVELOPE.error_allowance_m == runtime.ERROR_ALLOWANCE_M


def test_the_healthy_bound_is_not_below_the_declared_sigma_floor() -> None:
    """`configs/first_indoor.yaml` declares sigma_min 0.02 as the point below
    which the filter is claiming more than this rig knows. A healthy bound under
    it would be a bound the estimator cannot reach."""
    assert runtime.ESTIMATOR_HEALTHY_SIGMA_MAX_M >= 0.02


def test_the_composed_ball_is_the_one_the_reports_quote() -> None:
    """body 0.30 + allowance 0.15 + planner margin 0.025 = 0.475 m.

    This is the composed value every clearance discussion turns on, so it is
    pinned here rather than restated in prose: a cell can carry a certificate
    only if it holds a ball of this radius free.
    """
    voxel = float(runtime.MAP_PARAMETERS["voxel_m"])
    ball = runtime.ENVELOPE.inflation_m + voxel * planner_module.CERTIFICATE_MARGIN_VOXELS
    assert abs(ball - 0.475) < 1e-12


def test_the_allowance_is_the_rig_floor_not_a_conservative_choice() -> None:
    """A 3-sigma pose at the measured floor is admissible; the allowance admits it.

    If the allowance were ever lowered below 3 * the measured floor, a pose this
    rig actually produces would be refused by the validator's own rule — the
    measurement in the merged report, expressed as an assertion.
    """
    assert runtime.ERROR_ALLOWANCE_M >= 3.0 * runtime.ESTIMATOR_HEALTHY_SIGMA_MAX_M
