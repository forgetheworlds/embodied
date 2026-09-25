"""P01-C calibration behaviours: record round-trip, rectification geometry, refusal.

T1-T3 of the P01-C plan (section 8), one test per named behaviour. Nothing here
starts a simulator or touches a network: the stereo pair in T2 is procedurally
rendered from the declared geometry, and every record is built in memory.
"""

import dataclasses
import json

import numpy as np
import pytest

from embodied.contracts import records as R
from embodied.perception import camera as C


# --- T1: the record round-trips; missing information stays missing -----------


def test_t1_calibration_record_round_trip():
    calibration = C.build_calibration(
        time_offset_s=0.0,
        time_offset_error_s=0.0607,
        validated_limits="measured bounds from the pre-registered validation",
    )
    document = C.record_document(calibration)
    parsed = C.load_calibration_document(json.loads(json.dumps(document)))
    assert parsed == calibration
    # every section-22 field is present in the artifact body, each with a
    # provenance class from the declared vocabulary
    record_fields = {field.name for field in dataclasses.fields(R.Calibration)}
    assert record_fields == set(document["record"])
    assert set(document["provenance"]) == record_fields
    assert {entry["class"] for entry in document["provenance"].values()} <= {
        "declared",
        "declared-derived",
        "measured",
        "validated-limit",
    }


def test_t1_absent_value_stays_absent():
    document = C.record_document(C.build_calibration())
    assert document["record"]["time_offset_s"] is None
    assert document["record"]["time_offset_error_s"] is None
    # a document missing a key is refused on load, never silently defaulted
    broken = json.loads(json.dumps(document))
    del broken["record"]["baseline_m"]
    with pytest.raises(R.RecordError):
        C.load_calibration_document(broken)


# --- T2: rectification geometry on a planted synthetic scene -----------------


def _render_fronto_parallel_pair(disparity_px: float, seed: int = 7):
    """A textured fronto-parallel plane at disparity d: the right image is the
    left image shifted left by d, so d = u_left - u_right > 0."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    shift = int(round(disparity_px))
    right = np.empty_like(left)
    right[:, :-shift] = left[:, shift:]
    right[:, -shift:] = left[:, -1:]  # border padding; retained matches live inside
    return left, right


FEATURES = {
    "nfeatures": 2000,
    "scale_factor": 1.2,
    "nlevels": 8,
    "descriptor_ratio_threshold": 0.8,
}
MATCHER = {
    "num_disparities": 128,
    "block_size": 5,
    "uniqueness_ratio": 10,
    "speckle_window_size": 200,
    "speckle_range": 2,
    "texture_threshold": 10,
    "lr_tolerance_px": 1.0,
    "depth_range_m": [0.5, 4.0],
    "border_px": 8,
    "disparity_quantization_sigma_px": 1.0,
}


def test_t2_rectification_geometry_and_depth():
    planted_disparity = 23.0
    baseline_m = 0.10
    planted_depth = C.FOCAL_LENGTH_PX * baseline_m / planted_disparity
    left, right = _render_fronto_parallel_pair(planted_disparity)
    calibration = C.build_calibration()

    geometry = C.match_keypoints(C.cv2_gray(left), C.cv2_gray(right), FEATURES)
    assert geometry["n_retained"] >= 20
    assert geometry["row_residual_median_px"] <= 1.0
    assert geometry["row_residual_p95_px"] <= 2.0
    assert geometry["positive_disparity_fraction"] >= 0.99
    assert abs(geometry["disparity_median_px"] - planted_disparity) <= 0.5

    product = C.compute_validated_depth(
        left, right, calibration, MATCHER, pair_id="synthetic"
    )
    # no pose exists for a synthetic product: the absence is recorded, not defaulted
    assert product.pose_provenance.label == "NONE"
    assert product.capture_stamp is None and product.receipt_stamp is None
    assert product.valid.mean() > 0.5
    measured = product.depth_m[product.valid]
    assert abs(float(np.median(measured)) - planted_depth) <= 0.02 * planted_depth
    # the uncertainty envelope is the declared quantization bound, sample for sample
    sigma = product.uncertainty_m[product.valid]
    assert np.allclose(sigma, measured**2 * 1.0 / (C.FOCAL_LENGTH_PX * baseline_m))


# --- T3: invalid input refusal at the record and function boundaries ---------
def test_t3_wrong_transform_frame_names_refused():
    calibration = C.build_calibration()
    wrong_frames = R.Transform(
        parent_frame="body",
        child_frame="camera_right",  # T_body_camera_left must name camera_left
        translation_m=(0.05, 0.05, 0.05),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    # the record boundary: a rebuilt Calibration re-validates its frame names
    with pytest.raises(R.RecordError):
        dataclasses.replace(calibration, T_body_camera_left=wrong_frames)
    # the load boundary: a document whose transform names the wrong pair is refused
    tampered = C.record_document(calibration)
    tampered["record"]["T_body_camera_left"]["parent_frame"] = "odom"
    with pytest.raises(R.RecordError):
        C.load_calibration_document(tampered)


def test_t3_non_positive_baseline_refused():
    tampered = C.record_document(C.build_calibration())
    tampered["record"]["baseline_m"] = 0.0
    with pytest.raises(R.RecordError):
        C.load_calibration_document(tampered)


def test_t3_offset_without_error_refused():
    with pytest.raises(R.RecordError):
        C.build_calibration(time_offset_s=0.0, time_offset_error_s=None)


def test_t3_mismatched_pair_dimensions_refused():
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(R.RecordError):
        C.compute_validated_depth(left, right, C.build_calibration(), MATCHER)
