"""The survivorship arithmetic is checkable without a stereo pipeline.

The measurement is the P01-C pipeline, covered by its own tests, measured by this
tool's own run. What matters here is the one piece of arithmetic a reader has to
trust: whether restoring the samples the declared window discarded changes the
within-tolerance verdict, against the declared per-sample criterion.
"""

from __future__ import annotations

import numpy as np
import pytest

from embodied.perception.depth_window_survivorship import restored_verdict, within_fraction


def test_the_criterion_is_the_declared_max_of_absolute_and_relative():
    # at 5.883 m the declared relative term is 0.588 m and dominates the 0.15 m floor
    declared = np.array([5.883, 5.883, 5.883])
    errors = np.array([-0.20, 0.50, 0.60])
    assert within_fraction(errors, declared, 0.15, 0.10) == pytest.approx(2 / 3)


def test_the_absolute_floor_governs_close_in():
    declared = np.array([0.8, 0.8])
    errors = np.array([0.14, 0.16])
    # 0.10 * 0.8 = 0.08, so the 0.15 m absolute floor is what decides
    assert within_fraction(errors, declared, 0.15, 0.10) == pytest.approx(0.5)


def test_restoring_the_discarded_samples_can_only_be_reported_not_hidden():
    kept_e = np.array([0.05, -0.05])
    kept_d = np.array([5.9, 5.9])
    drop_e = np.array([0.30, 0.40])   # discarded for leaving the window, not for being wrong
    drop_d = np.array([5.9, 5.9])
    verdict = restored_verdict(kept_e, kept_d, drop_e, drop_d, 0.15, 0.10)
    assert verdict["n_kept"] == 2
    assert verdict["n_dropped"] == 2
    assert verdict["kept_within_fraction"] == pytest.approx(1.0)
    # 0.588 m still admits 0.40 m, so the verdict is unchanged -- and the tool says so
    assert verdict["restored_within_fraction"] == pytest.approx(1.0)
    assert verdict["dropped_fraction_of_surface"] == pytest.approx(0.5)


def test_a_discard_that_would_have_failed_is_reported_as_such():
    kept_e = np.array([0.05])
    kept_d = np.array([5.9])
    drop_e = np.array([0.70])         # beyond the 0.588 m criterion
    drop_d = np.array([5.9])
    verdict = restored_verdict(kept_e, kept_d, drop_e, drop_d, 0.15, 0.10)
    assert verdict["kept_within_fraction"] == pytest.approx(1.0)
    assert verdict["restored_within_fraction"] == pytest.approx(0.5)


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError):
        within_fraction(np.array([0.1, 0.2]), np.array([5.0]), 0.15, 0.10)
