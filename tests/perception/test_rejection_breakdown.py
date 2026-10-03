"""The invalidity buckets merge causes that call for different action.

Measured over 119 real retained frames, the pooled breakdown is valid 23.0 %,
no_return 43.2 %, depth_range 15.3 %, lr_mismatch 12.8 %, border 5.8 % — and the
per-source spread is enormous (valid 9.1 % on the P01-L scene against 55.9 % on
a mission run). The measurement behind these tests is
``work/runs/p05/J27-depthvalid-REPORT.md``.

Nothing here weakens a validity test, and no declared value moves. What the
tests pin is that the *reading* of a depth field distinguishes causes the reason
codes merge:

* ``depth_range`` on the near side is the floor under a forward-looking camera;
  on the far side it is a surface beyond the declared window. One source was
  entirely too-far, another mostly too-near.
* ``no_return`` is either SGBM's invalid marker or a zero-disparity sample, and
  only one of those is what a reader assumes.
* and the declared ``texture_threshold`` is either in force or it is not, which
  until now no artifact said.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from embodied.perception import camera

REPO = Path(__file__).resolve().parents[2]


def _declared_settings() -> dict:
    """The matcher and window exactly as the runtime builds them."""
    config = yaml.safe_load((REPO / "configs" / "first_indoor.yaml").read_text())
    calibration = config["calibration"]
    return {
        **calibration["matcher"],
        "border_px": calibration["bounds"]["border_px"],
        "depth_range_m": calibration["bounds"]["depth_range_m"],
    }


def _pair_at_disparity(disparity_px: int, *, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """A rectified pair whose true correspondence is ``disparity_px`` pixels apart.

    The right image is the left image translated along x, so every genuine
    correspondence lies on the same row and the depth it implies is
    ``f * B / disparity_px`` — which is what lets a test place a pair on either
    side of the declared window without touching the window.
    """
    import cv2  # lazy: the package does not require it at import time

    rng = np.random.default_rng(seed)
    height, width, pad = 480, 640, 320  # pad >= 2 * the largest shift used
    base = (rng.normal(0.0, 1.0, (height + pad, width + pad)) * 40.0 + 128.0).clip(0, 255)
    base = cv2.GaussianBlur(base.astype(np.float32), (5, 5), 1.2)
    left = base[pad // 2 : pad // 2 + height, pad // 2 : pad // 2 + width]
    right = base[pad // 2 : pad // 2 + height, pad // 2 + disparity_px : pad // 2 + disparity_px + width]
    # The depth boundary expects an RGB pair; the content is grey and only the
    # shape is checked there.
    left_rgb = np.repeat(left.astype(np.uint8)[:, :, None], 3, axis=2)
    right_rgb = np.repeat(right.astype(np.uint8)[:, :, None], 3, axis=2)
    return left_rgb, right_rgb


def _depth_of(disparity_px: int) -> float:
    calibration = camera.build_calibration()
    focal = calibration.left_intrinsics.focal_length_px[0]
    return focal * calibration.baseline_m / disparity_px


def test_the_window_can_be_reached_from_both_sides_by_a_synthetic_pair():
    """The instrument is valid: a pair's disparity places it in or out of range."""
    z_min, z_max = (float(v) for v in _declared_settings()["depth_range_m"])
    assert _depth_of(4) > z_max, "a 4 px correspondence is beyond the declared far bound"
    assert _depth_of(120) < z_min, "a 120 px correspondence is nearer than the declared near bound"


def test_the_breakdown_partitions_every_sample_exactly_once():
    """Exhaustive and disjoint, or the split would be a plausible story."""
    settings = _declared_settings()
    left, right = _pair_at_disparity(10)
    product = camera.compute_validated_depth(
        left, right, camera.build_calibration(), settings
    )
    parts = camera.rejection_breakdown(product, settings)
    total = (
        parts["valid"]
        + parts["border"]
        + parts["depth_range_near"]
        + parts["depth_range_far"]
        + parts["no_return_zero_disparity"]
        + parts["no_return_unmatched"]
        + parts["lr_mismatch"]
    )
    assert total == parts["samples"] == product.reasons.size


def test_the_two_sides_of_the_range_window_are_reported_apart():
    """A near rejection and a far rejection are different scenes, not one number."""
    settings = _declared_settings()
    calibration = camera.build_calibration()

    far_left, far_right = _pair_at_disparity(4)
    far = camera.rejection_breakdown(
        camera.compute_validated_depth(far_left, far_right, calibration, settings), settings
    )
    assert far["depth_range_far"] > 0, "a 4 px correspondence is beyond the far bound"
    assert far["depth_range_near"] == 0

    near_left, near_right = _pair_at_disparity(120)
    near = camera.rejection_breakdown(
        camera.compute_validated_depth(near_left, near_right, calibration, settings), settings
    )
    assert near["depth_range_near"] > 0, "a 120 px correspondence is nearer than the near bound"
    assert near["depth_range_far"] == 0


def test_no_return_separates_a_zero_disparity_from_an_unmatched_sample():
    """The two are disjoint and together account for every no_return sample."""
    settings = _declared_settings()
    left, right = _pair_at_disparity(10)
    product = camera.compute_validated_depth(
        left, right, camera.build_calibration(), settings
    )
    parts = camera.rejection_breakdown(product, settings)
    no_return = int((product.reasons == camera.REASON_NO_RETURN).sum())
    assert parts["no_return_zero_disparity"] + parts["no_return_unmatched"] == no_return
    # and each bucket holds exactly the samples it names
    assert parts["no_return_zero_disparity"] == int(
        ((product.reasons == camera.REASON_NO_RETURN) & (product.disparity_px == 0.0)).sum()
    )
    assert parts["no_return_unmatched"] == int(
        ((product.reasons == camera.REASON_NO_RETURN) & (product.disparity_px < 0.0)).sum()
    )


def test_the_breakdown_changes_nothing_about_the_product():
    """A reading is a reading: no reason moves, no validity changes."""
    settings = _declared_settings()
    left, right = _pair_at_disparity(10)
    product = camera.compute_validated_depth(
        left, right, camera.build_calibration(), settings
    )
    before = (product.reasons.copy(), product.valid.copy(), product.depth_m.copy())
    camera.rejection_breakdown(product, settings)
    assert np.array_equal(product.reasons, before[0])
    assert np.array_equal(product.valid, before[1])
    # depth is NaN wherever there was no match, and NaN != NaN, so compare with
    # NaNs treated as equal — that is the claim: the field did not move.
    assert np.array_equal(product.depth_m, before[2], equal_nan=True)
    assert int(product.valid.sum()) == int((product.reasons == camera.REASON_VALID).sum())


def test_a_depth_product_says_whether_the_declared_texture_filter_was_in_force():
    """A declared filter that is silently absent is a defect, not a detail.

    ``configs/first_indoor.yaml`` declares ``texture_threshold: 10`` and the CLI
    schema validates it, but OpenCV's ``StereoSGBM_create`` no longer accepts the
    parameter, so on the pinned backend it is not applied. That is recorded on
    the product rather than assumed away — and the flag must agree with what the
    backend actually did, whichever way that goes.
    """
    import cv2

    settings = _declared_settings()
    left, right = _pair_at_disparity(10)
    product = camera.compute_validated_depth(
        left, right, camera.build_calibration(), settings
    )
    _matcher, applied = camera._make_sgbm(cv2, settings)
    assert product.texture_threshold_applied is applied
    # On a backend that dropped the parameter this must be False, and the test
    # says so rather than asserting a value the backend no longer honours.
    assert product.texture_threshold_applied is False
