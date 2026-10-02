"""The FAST threshold is the parameter that decides how many features B1 sees.

B1 (rectification geometry) runs ORB and reports the row residuals of its
retained matches. The config declares the detector, ``nfeatures``, the scale
factor, the level count and the ratio threshold — and nothing else — so
``cv2.ORB_create`` leaves the FAST corner threshold at OpenCV's own default of
20. That default, not the scene and not the rig, turned out to be what decided
whether the mission scene retained any usable matches at all: the mission pairs
yield 36 keypoints at 20 and 951 at 5, while the compat pairs yield 25 either
way, and at 20 the mission's retained matches include geometrically impossible
ones (negative disparity, row residuals up to 27 px).

These tests pin the two things that follow, and neither changes a declared
value:

* the threshold in force is **named and reported in the evidence**, so no B1
  number can again depend on a parameter nobody can see; and
* on a low-contrast finely-textured pair the threshold genuinely decides whether
  the pair is used — which is why it is the operative constraint on this scene
  rather than an incidental setting.

The measurement behind that claim, on the real frames, is
``work/runs/p05/J18-rectify-REPORT.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from embodied.perception import camera

FEATURES = {
    "detector": "ORB",
    "nfeatures": 2000,
    "scale_factor": 1.2,
    "nlevels": 8,
    "descriptor_ratio_threshold": 0.8,
}


def _low_contrast_rectified_pair(*, disparity_px: int = 10, amplitude: float = 4.0, seed: int = 7):
    """A low-contrast, finely-textured pair that is rectified by construction.

    The right image is the left image translated along x by ``disparity_px``, so
    every genuine correspondence lies on the same row. Contrast is low and the
    texture is fine, which is the condition under which the FAST threshold — not
    the geometry — decides whether features are found.
    """
    import cv2  # lazy: the package does not require it at import time

    rng = np.random.default_rng(seed)
    h, w, pad = 480, 640, 40
    base = (rng.normal(0.0, 1.0, (h + pad, w + pad)) * amplitude + 128.0).clip(0, 255)
    base = cv2.GaussianBlur(base.astype(np.float32), (3, 3), 0.8)
    left = base[20 : 20 + h, 20 : 20 + w]
    right = base[20 : 20 + h, 20 + disparity_px : 20 + disparity_px + w]
    return left.astype(np.uint8), right.astype(np.uint8)


def test_the_threshold_in_force_is_named_and_reported():
    """An undeclared threshold is OpenCV's default, and the evidence says so."""
    left, right = _low_contrast_rectified_pair()
    geometry = camera.match_keypoints(left, right, FEATURES)
    assert "settings" in geometry
    assert geometry["settings"]["fast_threshold"] == camera.OPENCV_DEFAULT_FAST_THRESHOLD
    # the rest of the declared mechanism is reported beside it, unchanged
    assert geometry["settings"]["nfeatures"] == FEATURES["nfeatures"]
    assert geometry["settings"]["nlevels"] == FEATURES["nlevels"]
    assert (
        geometry["settings"]["descriptor_ratio_threshold"]
        == FEATURES["descriptor_ratio_threshold"]
    )


def test_a_declared_threshold_is_honoured_and_reported():
    """If the config ever declares one, the code uses it and the evidence says so."""
    left, right = _low_contrast_rectified_pair()
    declared = dict(FEATURES, fast_threshold=5)
    geometry = camera.match_keypoints(left, right, declared)
    assert geometry["settings"]["fast_threshold"] == 5


def test_every_return_carries_the_settings():
    """Including the paths that retain nothing, so a null result is readable."""
    flat = np.full((480, 640), 128, dtype=np.uint8)
    geometry = camera.match_keypoints(flat, flat, FEATURES)
    assert geometry["n_retained"] == 0
    assert geometry["settings"]["fast_threshold"] == camera.OPENCV_DEFAULT_FAST_THRESHOLD


def test_the_threshold_decides_whether_a_low_contrast_pair_is_used():
    """The threshold, not the geometry, is what the scene is sensitive to.

    Both calls see the same pixels. Only the FAST threshold differs. If the
    default retained as much as a lowered one, the threshold would be incidental
    on this kind of scene and the mission scene's failure would need another
    cause. It does not: the default retains nothing here and the lowered value
    retains a full set.
    """
    left, right = _low_contrast_rectified_pair()
    in_force = camera.match_keypoints(left, right, FEATURES)
    lowered = camera.match_keypoints(left, right, dict(FEATURES, fast_threshold=5))

    assert in_force["n_retained"] < 10, "the documented default is starved on this pair"
    assert lowered["n_retained"] >= 20, "a full set exists in the same pixels"

    # And when features ARE found, this rig's geometry lines the correspondences
    # up exactly -- the rows agree because the pair is rectified, not because the
    # detector was lenient. This is the control that separates a rectification
    # failure from a matching failure.
    assert lowered["positive_disparity_fraction"] == pytest.approx(1.0)
    assert lowered["row_residual_p95_px"] <= 1.0
