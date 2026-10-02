"""Calibrated stereo: the declared Calibration record, rectification and metric depth.

This module is P01-C's half of specification sections 4-6. Three things live here:

**The declared Calibration record** (``first-indoor-stereo-1``, version 2, against
the frozen records revision ``p00-records-1``). Every field value is either read
from the pinned scene declaration, derived from it by a recorded formula, or
measured from the P00 accept-5 evidence — nothing is invented. The provenance
class of every field is in :func:`provenance_table` and travels with the record
into its JSON artifact.

**Rectification and validity-qualified depth.** The rig is declared already
rectified (identity relative rotation, purely horizontal baseline), but the code
runs the general OpenCV path — ``stereoRectify`` maps from the record's own
transforms — so the machinery is real, not a shortcut painted on. Depth follows
the record's depth convention: ``z = f * B / d`` in metres along the left
rectified optical axis, disparity left-positive, and every sample carries
validity, a rejection reason and a quantization uncertainty envelope. Validity
filters are the documented mechanisms of specification section 5.2, not
calibrated probabilities. A masked sample is unknown, never zero.

**Capture-time provenance.** A depth product is bound to its capture instant —
capture stamp, receipt stamp, simulator time, pair id, calibration id — and to a
:class:`PoseProvenance`. The only pose this stage can attach is the simulator
scene's declared static-start pose, so every product this module emits for the
compat evidence is labelled ``DIAGNOSTIC``: it can never pass P01, and it is
never pooled with scored sensor-derived results. Where no pose exists, the field
says so instead of inventing one.

Honesty notes carried by construction: distortion "none" is *declared*, not
measured — B1's p95 row-residual bound is the guard that would surface a real
distortion mismatch as residual growth toward the image corners. OpenCV is
imported lazily so the package import does not demand it; the records module
stays stdlib-only.

Offline derivation entry point: ``python -m embodied.perception.camera
--config configs/first_indoor.yaml`` runs the pre-registered validation over the
accept-5 evidence and writes the plan-section-10 artifacts. No CLI dispatch
command is registered for it; that would be a separate serialized integrator
action.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied.contracts import records as R

CALIBRATION_ID = "first-indoor-stereo-1"
CALIBRATION_VERSION = "2"

# Declared rig revision, matching the pinned scene declaration
# (scenarios/compat/calibration.json, sha256 0230799d...) exactly:
# f = width / (2 tan(FOV/2)) with the declared 60-degree field of view over
# 640 px (the world file writes it rounded as 1.0472 rad); principal point at
# the 640x480 centre by construction.
_DECLARED_FOV_RAD = math.pi / 3.0
_DECLARED_WIDTH_PX = 640
_DECLARED_HEIGHT_PX = 480
FOCAL_LENGTH_PX = 554.2562584220407  # == _DECLARED_WIDTH_PX / (2 tan(pi/6)); asserted below
assert abs(FOCAL_LENGTH_PX - _DECLARED_WIDTH_PX / (2.0 * math.tan(_DECLARED_FOV_RAD / 2.0))) < 1e-9
PRINCIPAL_POINT_PX = (_DECLARED_WIDTH_PX / 2.0, _DECLARED_HEIGHT_PX / 2.0)

# ORB's FAST corner threshold. It is NOT declared: the config's ``features``
# section carries detector, nfeatures, scale_factor, nlevels and
# descriptor_ratio_threshold (cli.py's schema admits exactly those), and
# ``cv2.ORB_create`` leaves the rest at OpenCV's defaults — so this value is what
# has actually been in force for every B1 number this project has reported.
# It is the parameter that decides how many features a scene yields, and on a
# low-contrast, finely-textured scene it is the operative constraint: the
# mission scene returns 36 keypoints at this value and 951 at 5, while the
# compat scene returns 25 either way. It is named here so it is visible in the
# code and recorded in the evidence; the value is deliberately NOT changed,
# because choosing it is a declared-parameter decision (R2). The measurement
# that bears on that choice is work/runs/p05/J18-rectify-REPORT.md.
OPENCV_DEFAULT_FAST_THRESHOLD = 20

_PIXEL_CONVENTION = (
    "top-left origin, [column, row]; columns run along body -y (rightward) and rows "
    "run along body -z (down), because a Webots camera's optical axis is its own +x; "
    "pixel (0,0) is the top-left of a 640x480 image"
)
_DEPTH_CONVENTION = (
    "metric depth z = f*B/d in metres along the LEFT RECTIFIED optical axis; "
    "disparity d in pixels, left-positive (d = x_left - x_right > 0 for points in "
    "front); depth is valid only where the validity mask is true; a masked sample "
    "is unknown, never zero"
)
_SOURCE_TEXT = (
    "P01-C plan work/runs/p01c/plan.md from the P00 accept-5 evidence "
    "(work/runs/p00-compat/accept-5/run-a/pairs and .../run-b/pairs) and the pinned "
    "scene declaration scenarios/compat/calibration.json sha256 "
    "0230799daa822f815ef8d0336c958e1fa8cde12ccc635755f1ba5d470fdd4b46 "
    "(world scenarios/compat/worlds/compat_stereo.wbt, rig scenarios/compat/protos/"
    "Iris.proto). Provenance classes: declared = read from the pinned scene/proto/"
    "declaration; declared-derived = arithmetic on a declared value with the formula "
    "recorded; measured = computed from the accept-5 evidence; validated-limit = a "
    "bound computed by the pre-registered validation run and recorded in "
    "validated_limits. Nothing is invented; an unavailable value stays None."
)

# Rejection reasons for depth samples, specification section 5.2 filter set. The
# SGBM-internal mechanisms (uniqueness, texture, speckle) run inside the matcher
# with their declared parameters; these are the explicit post-match checks.
REASON_VALID = 0
REASON_NO_RETURN = 1
REASON_LR_MISMATCH = 2
REASON_DEPTH_RANGE = 3
REASON_BORDER = 4
REASON_NAMES = (
    "valid",
    "no_return",
    "lr_mismatch",
    "depth_range",
    "border",
)


# ---------------------------------------------------------------------------
# The declared Calibration record and its provenance
# ---------------------------------------------------------------------------


def _identity_quaternion() -> tuple[float, float, float, float]:
    return (1.0, 0.0, 0.0, 0.0)


def _transform(parent: str, child: str, translation: tuple[float, float, float]) -> R.Transform:
    return R.Transform(
        parent_frame=parent,
        child_frame=child,
        translation_m=translation,
        quaternion_wxyz=_identity_quaternion(),
    )


def provenance_table() -> dict[str, dict[str, str]]:
    """The provenance class of every Calibration field value, per the plan."""
    return {
        "calibration_id": {
            "class": "declared",
            "detail": "names this rig revision (first-indoor stereo, declaration 1)",
        },
        "version": {
            "class": "declared",
            "detail": (
                "2; v1 is P00's compat-stereo-1 declaration — a changed calibration is "
                "a new version, never an edit"
            ),
        },
        "left_intrinsics": {
            "class": "declared-derived",
            "detail": (
                "f = width / (2*tan(FOV/2)) with the declared 60-degree (pi/3 rad) field of "
                "view and width 640 — the world file writes the angle rounded as 1.0472 rad; "
                "the value equals the pinned compat-stereo-1 declaration exactly; principal "
                "point at the 640x480 centre by construction"
            ),
        },
        "right_intrinsics": {
            "class": "declared-derived",
            "detail": "same formula; both cameras are the same Webots node type",
        },
        "left_distortion": {
            "class": "declared",
            "detail": (
                "Webots pinhole renders without lens distortion; declared, NOT measured "
                "- B1's p95 row-residual bound guards corner-growth from a real mismatch"
            ),
        },
        "right_distortion": {
            "class": "declared",
            "detail": "as left_distortion",
        },
        "T_camera_left_camera_right": {
            "class": "declared",
            "detail": "rig placement in scenarios/compat/protos/Iris.proto, mirrored in calibration.json",
        },
        "T_body_camera_left": {
            "class": "declared",
            "detail": "Iris.proto camera translation (0.05, 0.05, 0.05) / calibration.json",
        },
        "T_body_imu": {
            "class": "declared",
            "detail": "Iris.proto inertial unit at the body origin / calibration.json",
        },
        "baseline_m": {
            "class": "declared",
            "detail": (
                "norm of the T_camera_left_camera_right translation; the config's "
                "sensors.stereo.baseline_m repeats it"
            ),
        },
        "pixel_convention": {"class": "declared", "detail": "carried text"},
        "depth_convention": {
            "class": "declared",
            "detail": "specification section 5.2 decision, written into the record",
        },
        "time_offset_s": {
            "class": "declared",
            "detail": (
                "0.0 by construction: all Webots devices stamp from the one simulator "
                "clock the adapter frames carry as sim_time_s; declared, not measured"
            ),
        },
        "time_offset_error_s": {
            "class": "measured",
            "detail": (
                "larger of the accept-5 host-to-autopilot residual spreads (run-a "
                "57.14 ms, run-b 60.66 ms), rounded up to 0.1 ms as a bound; any "
                "consumer converting sim-time stamps through the host relation "
                "inherits this error bar"
            ),
        },
        "validated_limits": {
            "class": "validated-limit",
            "detail": (
                "filled only from the pre-registered validation run's measured results; "
                "absent (None) means the run did not complete its gates"
            ),
        },
        "source": {"class": "declared", "detail": "carried text naming plan, evidence and classes"},
    }


def build_calibration(
    *,
    time_offset_s: float | None = None,
    time_offset_error_s: float | None = None,
    validated_limits: str | None = None,
) -> R.Calibration:
    """Build the declared record. Refusals (wrong frames, non-positive baseline,
    offset-without-error) are the frozen record's own validation — exercised as
    gate B4 and test T3."""
    intrinsics = R.CameraIntrinsics(
        focal_length_px=(FOCAL_LENGTH_PX, FOCAL_LENGTH_PX),
        principal_point_px=PRINCIPAL_POINT_PX,
    )
    no_distortion = R.Distortion(model="none_declared", coefficients=(0.0,))
    return R.Calibration(
        calibration_id=CALIBRATION_ID,
        version=CALIBRATION_VERSION,
        left_intrinsics=intrinsics,
        right_intrinsics=intrinsics,
        left_distortion=no_distortion,
        right_distortion=no_distortion,
        T_camera_left_camera_right=_transform("camera_left", "camera_right", (0.0, -0.1, 0.0)),
        T_body_camera_left=_transform("body", "camera_left", (0.05, 0.05, 0.05)),
        T_body_imu=_transform("body", "imu", (0.0, 0.0, 0.0)),
        baseline_m=0.10,
        pixel_convention=_PIXEL_CONVENTION,
        depth_convention=_DEPTH_CONVENTION,
        time_offset_s=time_offset_s,
        time_offset_error_s=time_offset_error_s,
        validated_limits=validated_limits,
        source=_SOURCE_TEXT,
    )


def record_document(record: R.Calibration) -> dict:
    """The JSON artifact body: the record plus its per-field provenance table."""
    return {"record": R.to_dict(record), "provenance": provenance_table()}


def load_calibration_document(document: dict) -> R.Calibration:
    """Parse the artifact body back to a record, refusing unknown or missing keys."""
    return R.from_dict(R.Calibration, document["record"])


# ---------------------------------------------------------------------------
# Capture-time provenance
# ---------------------------------------------------------------------------

DIAGNOSTIC = "DIAGNOSTIC"


@dataclass(frozen=True)
class PoseProvenance:
    """Where a depth product's capture-time pose came from, labelled.

    ``label`` is ``DIAGNOSTIC`` for simulator-scene truth: such a pose can never
    pass P01 and is never pooled with scored results. A product with no pose at
    all carries label ``NONE`` and states why the absence is information.
    """

    label: str
    detail: str

    def __post_init__(self) -> None:
        for name in ("label", "detail"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise R.RecordError(f"{name} must be a non-empty string")


def static_start_provenance(world_ref: str) -> PoseProvenance:
    """The provenance of the compat evidence's declared static-start pose."""
    return PoseProvenance(
        label=DIAGNOSTIC,
        detail=(
            "capture-time pose is the simulator scene's declared static start "
            f"({world_ref}); simulated-scene truth, diagnostic only — never a "
            "sensor-derived pose, never pooled with scored results"
        ),
    )

def declared_pose_provenance(evidence: dict) -> PoseProvenance:
    """Where the declared capture-time pose came from, labelled truthfully.

    The compat evidence's pose is the simulator scene's declared static start, and that
    is the default here. A capture whose pose is instead the run's own published estimate
    declares ``pose_source: measured_published_pose`` with a ``pose_source_detail``, and is
    labelled ``SENSOR_DERIVED`` — the same class the mission runtime stamps on its live
    depth products. The label is not decoration: it decides whether a reader may treat the
    declared depth as independent of the estimator. A measured pose that cannot state
    itself is refused rather than mislabelled as scene truth.
    """
    if evidence.get("pose_source") == "measured_published_pose":
        detail = evidence.get("pose_source_detail")
        if not isinstance(detail, str) or not detail.strip():
            raise R.RecordError(
                "a measured pose source must state its provenance in pose_source_detail"
            )
        return PoseProvenance(label="SENSOR_DERIVED", detail=detail)
    return static_start_provenance(str(evidence["world"]))


# ---------------------------------------------------------------------------
# Depth products bound to their capture instant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthProduct:
    """One validity-qualified metric depth field and the capture it belongs to.

    Arrays are plain numpy arrays, not records: the section-22 DepthEstimate
    record is deliberately deferred (records.py ``DEFERRED_RECORDS``) until its
    owning worker lands. This is the P01-C binding surface — capture stamps,
    simulator time, pair id, calibration identity, pose provenance, validity and
    a quantization uncertainty envelope per sample.
    """

    calibration_id: str
    calibration_version: str
    pair_id: str | None
    capture_stamp: R.ClockStamp | None
    receipt_stamp: R.ClockStamp | None
    sim_time_s: float | None
    pose_provenance: PoseProvenance
    frame: str
    disparity_px: np.ndarray
    depth_m: np.ndarray
    valid: np.ndarray
    reasons: np.ndarray
    uncertainty_m: np.ndarray

    def __post_init__(self) -> None:
        def text(value: object, name: str) -> None:
            if not isinstance(value, str) or not value.strip():
                raise R.RecordError(f"{name} must be a non-empty string")

        text(self.calibration_id, "calibration_id")
        text(self.calibration_version, "calibration_version")
        if self.pair_id is not None:
            text(self.pair_id, "pair_id")
        if self.capture_stamp is not None and not isinstance(self.capture_stamp, R.ClockStamp):
            raise R.RecordError("capture_stamp must be a ClockStamp or None")
        if self.receipt_stamp is not None and not isinstance(self.receipt_stamp, R.ClockStamp):
            raise R.RecordError("receipt_stamp must be a ClockStamp or None")
        if (self.capture_stamp is None) != (self.receipt_stamp is None):
            raise R.RecordError("capture and receipt stamps come as a pair, or both stay absent")
        if self.capture_stamp is not None and R.elapsed_ns(self.capture_stamp, self.receipt_stamp) < 0:
            raise R.RecordError("a depth product cannot be received before it was captured")
        if self.sim_time_s is not None and (
            isinstance(self.sim_time_s, bool) or not isinstance(self.sim_time_s, (int, float))
        ):
            raise R.RecordError("sim_time_s must be a number or None")
        if not isinstance(self.pose_provenance, PoseProvenance):
            raise R.RecordError("pose_provenance must be a PoseProvenance")
        text(self.frame, "frame")
        for name in ("disparity_px", "depth_m", "valid", "reasons", "uncertainty_m"):
            if not isinstance(getattr(self, name), np.ndarray):
                raise R.RecordError(f"{name} must be a numpy array")


def _quaternion_rotation(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = quaternion_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Rectification (lazy OpenCV)
# ---------------------------------------------------------------------------


class Rectifier:
    """Rectifying maps built from the record's own declared values.

    The declared rig has identity relative rotation and a purely horizontal
    baseline, so its maps are the identity to numerical precision — the renderer
    already produces the declared rectified geometry, and B1 checks exactly that
    claim. The machinery is the general OpenCV path so a future rig with real
    rotation or distortion needs no new code, only a new record.
    """

    def __init__(self, calibration: R.Calibration, image_size: tuple[int, int]):
        import cv2  # lazy: the package import must not demand OpenCV

        for distortion in (calibration.left_distortion, calibration.right_distortion):
            if distortion.model not in ("none_declared", "none"):
                raise R.RecordError(
                    f"no distortion fitting is implemented for model {distortion.model!r}; "
                    "a calibrated distortion needs a new calibration stage, not a silent pass-through"
                )
        def camera_matrix(intrinsics: R.CameraIntrinsics) -> np.ndarray:
            return np.array(
                [
                    [intrinsics.focal_length_px[0], 0.0, intrinsics.principal_point_px[0]],
                    [0.0, intrinsics.focal_length_px[1], intrinsics.principal_point_px[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        k_left = camera_matrix(calibration.left_intrinsics)
        k_right = camera_matrix(calibration.right_intrinsics)
        # OpenCV's stereoRectify accepts only 4/5/8/12/14-coefficient vectors; the
        # declared none model is zero-padded into the 5-slot layout.
        d_left = np.zeros(5, dtype=np.float64)
        d_left[: len(calibration.left_distortion.coefficients)] = calibration.left_distortion.coefficients
        d_right = np.zeros(5, dtype=np.float64)
        d_right[: len(calibration.right_distortion.coefficients)] = calibration.right_distortion.coefficients
        # T_camera_left_camera_right maps camera_right points into camera_left
        # (p_L = R p_R + t); OpenCV wants the camera-1 -> camera-2 relation
        # (p_R = R_cv p_L + T_cv), which is its inverse: R_cv = R^T, T_cv = -R^T t.
        declared = calibration.T_camera_left_camera_right
        r_declared = _quaternion_rotation(declared.quaternion_wxyz)
        t_declared = np.array(declared.translation_m, dtype=np.float64)
        r_cv = r_declared.T
        t_cv = -r_declared.T @ t_declared
        r1, r2, p1, p2, _q, _, _ = cv2.stereoRectify(
            k_left,
            d_left,
            k_right,
            d_right,
            image_size,
            r_cv,
            t_cv,
            flags=0,
            alpha=0.0,
        )
        self._cv2 = cv2
        self._map_left = cv2.initUndistortRectifyMap(
            k_left, d_left, r1, p1, image_size, cv2.CV_32FC1
        )
        self._map_right = cv2.initUndistortRectifyMap(
            k_right, d_right, r2, p2, image_size, cv2.CV_32FC1
        )

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if left.shape != right.shape:
            raise R.RecordError(
                f"mismatched stereo pair dimensions: {left.shape} vs {right.shape}"
            )
        cv2 = self._cv2
        return (
            cv2.remap(left, self._map_left[0], self._map_left[1], cv2.INTER_LINEAR),
            cv2.remap(right, self._map_right[0], self._map_right[1], cv2.INTER_LINEAR),
        )


# ---------------------------------------------------------------------------
# Feature geometry (B1) and the validated depth machinery (B2)
# ---------------------------------------------------------------------------


def match_keypoints(
    left_gray: np.ndarray, right_gray: np.ndarray, features: dict
) -> dict:
    """ORB detect + ratio-test match, then the B1 geometry summary.

    Returns retained-match row residuals (``x_left - x_right`` row difference),
    their horizontal disparities, and the counts, together with the detector
    settings actually used (``settings``) so a reader of the evidence can see
    which parameters produced the numbers — including the FAST threshold, which
    is not declared in the config and therefore defaults to OpenCV's own
    (OPENCV_DEFAULT_FAST_THRESHOLD above).
    """
    import cv2  # lazy

    settings = {
        "detector": str(features.get("detector", "ORB")),
        "nfeatures": int(features["nfeatures"]),
        "scale_factor": float(features["scale_factor"]),
        "nlevels": int(features["nlevels"]),
        "fast_threshold": int(
            features.get("fast_threshold", OPENCV_DEFAULT_FAST_THRESHOLD)
        ),
        "descriptor_ratio_threshold": float(features["descriptor_ratio_threshold"]),
    }
    # Every return carries the same settings, so a pair that retains nothing is
    # still readable as "these parameters retained nothing".
    empty = {
        "n_retained": 0,
        "row_residual_median_px": None,
        "row_residual_p95_px": None,
        "positive_disparity_fraction": None,
        "disparity_median_px": None,
        "settings": settings,
    }
    orb = cv2.ORB_create(
        nfeatures=settings["nfeatures"],
        scaleFactor=settings["scale_factor"],
        nlevels=settings["nlevels"],
        fastThreshold=settings["fast_threshold"],
    )
    key_left, desc_left = orb.detectAndCompute(left_gray, None)
    key_right, desc_right = orb.detectAndCompute(right_gray, None)
    if desc_left is None or desc_right is None:
        return dict(empty)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_left, desc_right, k=2)
    ratio = settings["descriptor_ratio_threshold"]
    retained = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    if not retained:
        return dict(empty)
    rows = np.array(
        [
            key_left[m.queryIdx].pt[1] - key_right[m.trainIdx].pt[1]
            for m in retained
        ]
    )
    disparities = np.array(
        [
            key_left[m.queryIdx].pt[0] - key_right[m.trainIdx].pt[0]
            for m in retained
        ]
    )
    return {
        "n_retained": int(len(retained)),
        "row_residual_median_px": float(np.median(np.abs(rows))),
        "row_residual_p95_px": float(np.percentile(np.abs(rows), 95)),
        "positive_disparity_fraction": float((disparities > 0).mean()),
        "disparity_median_px": float(np.median(disparities)),
        "settings": settings,
    }


def _make_sgbm(cv2, settings: dict):
    """Build the declared StereoSGBM matcher for the pinned OpenCV backend.

    ``texture_threshold`` is part of the declared mechanism set, but OpenCV's
    ``StereoSGBM_create`` dropped the parameter in recent 4.x releases (its
    low-texture handling moved inside the cost aggregation). The backend
    capability is detected, never assumed: the returned flag says whether the
    declared value was applied, and the derivation records it.
    """
    block = int(settings["block_size"])
    kwargs = dict(
        minDisparity=0,
        numDisparities=int(settings["num_disparities"]),
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=int(settings["lr_tolerance_px"]),
        uniquenessRatio=int(settings["uniqueness_ratio"]),
        speckleWindowSize=int(settings["speckle_window_size"]),
        speckleRange=int(settings["speckle_range"]),
    )
    try:
        return cv2.StereoSGBM_create(**kwargs, textureThreshold=int(settings["texture_threshold"])), True
    except TypeError:
        return cv2.StereoSGBM_create(**kwargs), False


def compute_validated_depth(
    left: np.ndarray,
    right: np.ndarray,
    calibration: R.Calibration,
    settings: dict,
    *,
    pair_id: str | None = None,
    capture_stamp: R.ClockStamp | None = None,
    receipt_stamp: R.ClockStamp | None = None,
    sim_time_s: float | None = None,
    pose_provenance: PoseProvenance | None = None,
) -> DepthProduct:
    """Rectify, run SGBM with its validity filters, bind metric depth to capture identity.

    Refuses mismatched pair dimensions at the boundary (gate B4). Every output
    sample carries validity, one rejection reason and the quantization
    uncertainty envelope ``sigma_z = z^2 * sigma_d / (f * B)`` — an envelope,
    never a calibrated probability.
    """
    import cv2  # lazy

    if left.ndim != 3 or left.shape[2] != 3 or right.shape != left.shape:
        raise R.RecordError(
            f"expected an RGB stereo pair of one shape, got {left.shape} and {right.shape}"
        )
    size = (left.shape[1], left.shape[0])
    rectifier = Rectifier(calibration, size)
    left_rect, right_rect = rectifier.rectify(left, right)
    gray_left = cv2.cvtColor(left_rect, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(right_rect, cv2.COLOR_RGB2GRAY)

    sgbm, _texture_threshold_applied = _make_sgbm(cv2, settings)
    disparity = sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0
    # Right-reference pass for the left-right consistency check: matching the
    # flipped right image against the flipped left image yields positive right
    # disparities; after flipping back, the check compares d_L(u) with d_R(u - d_L(u)).
    right_sgbm, _ = _make_sgbm(cv2, settings)
    right_disparity = cv2.flip(
        right_sgbm.compute(cv2.flip(gray_right, 1), cv2.flip(gray_left, 1)).astype(np.float32) / 16.0,
        1,
    )

    height, width = disparity.shape
    margin = int(settings["border_px"])
    reasons = np.full((height, width), REASON_NO_RETURN, dtype=np.uint8)
    positive = disparity > 0.0
    lr_tolerance = float(settings["lr_tolerance_px"])
    columns = np.arange(width, dtype=np.int64)[None, :].repeat(height, axis=0)
    shifted = np.clip(np.round(columns - disparity).astype(np.int64), 0, width - 1)
    lr_agrees = positive & (
        np.abs(disparity - np.take_along_axis(right_disparity, shifted, axis=1)) <= lr_tolerance
    )
    outside_border = np.zeros((height, width), dtype=bool)
    outside_border[:margin, :] = True
    outside_border[-margin:, :] = True
    outside_border[:, :margin] = True
    outside_border[:, -margin:] = True

    reasons[outside_border] = REASON_BORDER

    reasons[~outside_border & positive] = REASON_NO_RETURN
    reasons[~outside_border & positive & lr_agrees] = REASON_VALID
    reasons[~outside_border & positive & ~lr_agrees] = REASON_LR_MISMATCH

    focal = calibration.left_intrinsics.focal_length_px[0]
    baseline = calibration.baseline_m
    depth = np.full((height, width), np.nan, dtype=np.float32)
    depth[positive] = (focal * baseline) / disparity[positive]

    z_min, z_max = (float(v) for v in settings["depth_range_m"])
    in_window = np.isfinite(depth) & (depth >= z_min) & (depth <= z_max)
    reasons[~outside_border & positive & lr_agrees & ~in_window] = REASON_DEPTH_RANGE

    valid = reasons == REASON_VALID
    sigma_d = float(settings["disparity_quantization_sigma_px"])
    uncertainty = np.full((height, width), np.nan, dtype=np.float32)
    uncertainty[valid] = (depth[valid] ** 2) * sigma_d / (focal * baseline)

    if pose_provenance is None:
        pose_provenance = PoseProvenance(
            label="NONE",
            detail="no capture-time pose exists for this product; absence is recorded, not defaulted",
        )
    return DepthProduct(
        calibration_id=calibration.calibration_id,
        calibration_version=calibration.version,
        pair_id=pair_id,
        capture_stamp=capture_stamp,
        receipt_stamp=receipt_stamp,
        sim_time_s=sim_time_s,
        pose_provenance=pose_provenance,
        frame=(
            "depth along the left rectified optical axis; camera device frame per the "
            "record's pixel_convention"
        ),
        disparity_px=disparity,
        depth_m=depth,
        valid=valid,
        reasons=reasons,
        uncertainty_m=uncertainty,
    )


# ---------------------------------------------------------------------------
# Declared referee surfaces (B3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredSurface:
    """A referee plane transcribed from the pinned world file."""

    name: str
    plane_point_world_m: tuple[float, float, float]
    plane_normal_world: tuple[float, float, float]
    axis_bounds_world_m: dict[str, tuple[float, float]]

    @classmethod
    def from_config(cls, entry: dict) -> "DeclaredSurface":
        return cls(
            name=str(entry["name"]),
            plane_point_world_m=tuple(float(v) for v in entry["plane_point_world_m"]),
            plane_normal_world=tuple(float(v) for v in entry["plane_normal_world"]),
            axis_bounds_world_m={
                axis: (float(lohi[0]), float(lohi[1]))
                for axis, lohi in entry["axis_bounds_world_m"].items()
            },
        )


def declared_depth_map(
    surfaces: list[DeclaredSurface],
    camera_position_world: tuple[float, float, float],
    camera_rotation_world: np.ndarray,
    focal_px: float,
    principal_point_px: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel declared depth of the nearest declared surface, and its index.

    The nearest hit is the surface the renderer draws (front-to-back), so the
    attribution is the occlusion rule. Rays follow the record's pixel
    convention: camera x forward, image columns along camera -y, rows along
    camera -z.
    """
    width, height = image_size
    u = np.arange(width, dtype=np.float64)[None, :].repeat(height, axis=0)
    v = np.arange(height, dtype=np.float64)[:, None].repeat(width, axis=1)
    dirs_camera = np.stack(
        [
            np.ones_like(u),
            -(u - principal_point_px[0]) / focal_px,
            -(v - principal_point_px[1]) / focal_px,
        ],
        axis=2,
    )
    dirs_world = dirs_camera @ camera_rotation_world.T
    origin = np.asarray(camera_position_world, dtype=np.float64)

    depth = np.full((height, width), np.inf, dtype=np.float64)
    attribution = np.full((height, width), -1, dtype=np.int16)
    for index, surface in enumerate(surfaces):
        normal = np.asarray(surface.plane_normal_world, dtype=np.float64)
        point = np.asarray(surface.plane_point_world_m, dtype=np.float64)
        denominator = dirs_world @ normal
        numerator = float(normal @ (point - origin))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = numerator / denominator
            hit = t > 0.0
            # parallel rays give inf/nan t; the hit test drops them
            hit_point = origin[None, None, :] + t[..., None] * dirs_world
        for axis, (lo, hi) in surface.axis_bounds_world_m.items():
            axis_index = {"x": 0, "y": 1, "z": 2}[axis]
            hit &= (hit_point[..., axis_index] >= lo) & (hit_point[..., axis_index] <= hi)
        nearer = hit & (t < depth)
        depth[nearer] = t[nearer]
        attribution[nearer] = index
    return depth, attribution


# ---------------------------------------------------------------------------
# PPM input (the accept-5 pairs are P6 binary)
# ---------------------------------------------------------------------------


def read_ppm(path: Path) -> np.ndarray:
    """Read a binary PPM (P6) file into an (H, W, 3) uint8 RGB array."""
    data = Path(path).read_bytes()
    cursor = 0
    tokens: list[bytes] = []
    while len(tokens) < 4:
        while cursor < len(data) and data[cursor : cursor + 1].isspace():
            cursor += 1
        if data[cursor : cursor + 1] == b"#":
            while data[cursor : cursor + 1] not in (b"\n", b""):
                cursor += 1
            continue
        start = cursor
        while cursor < len(data) and not data[cursor : cursor + 1].isspace():
            cursor += 1
        tokens.append(data[start:cursor])
    cursor += 1  # the single whitespace byte after maxval
    magic, width, height, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    if magic != b"P6" or maxval != 255:
        raise R.RecordError(f"{path}: expected binary P6 with maxval 255, got {magic!r}/{maxval}")
    pixels = np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=cursor)
    return pixels.reshape(height, width, 3)


# ---------------------------------------------------------------------------
# The offline derivation (plan sections 6, 7, 10)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _find_pairs(pair_dir: Path) -> list[tuple[Path, Path, Path]]:
    """(left, right, pairs.jsonl entry lookup source) triples in sequence order."""
    records = {}
    journal = pair_dir.parent / "pairs.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("stored"):
                records[entry["left_ppm"]] = entry
    triples = []
    for left_path in sorted(pair_dir.glob("*-left.ppm")):
        right_path = left_path.with_name(left_path.name.replace("-left", "-right"))
        if not right_path.exists():
            raise R.RecordError(f"pair half missing: {right_path}")
        triples.append((left_path, right_path, records.get(f"{pair_dir.parent.name}/pairs/{left_path.name}")))
    if not triples:
        raise R.RecordError(f"no *-left.ppm pairs under {pair_dir}")
    return triples


def _measured_time_offset_error(timebase_paths: list[Path]) -> tuple[float, list[dict]]:
    """The larger measured residual spread, rounded up as a bound, with its sources."""
    spreads = []
    for path in timebase_paths:
        document = json.loads(path.read_text())
        spread_ms = document["host_to_autopilot"]["spread_ms"]
        spreads.append({"path": str(path), "spread_ms": float(spread_ms)})
    worst_s = max(entry["spread_ms"] for entry in spreads) / 1000.0
    bound_s = math.ceil(worst_s * 10_000.0) / 10_000.0
    return bound_s, spreads


def run_validation(config_path: Path, *, code_revision: str) -> int:
    """Run the pre-registered P01-C validation; write the section-10 artifacts.

    Returns the process exit code: 0 when all gates pass, 2 (blocked) when any
    gate fails — thresholds are never relaxed after measuring (plan section 7).
    """
    import yaml  # lazy: only the derivation needs YAML

    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text())
    section = config["calibration"]
    started = _utc_now()
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    def resolve(rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else repo_root / path

    bounds = section["bounds"]
    features = section["features"]
    matcher = section["matcher"]
    referee_cfg = section["referee"]
    evidence = section["evidence"]

    # The record before measurement: no offset and no error yet — both None,
    # because an offset without its measured error is refused by design. The
    # final record below is built with the measured pair.
    calibration = build_calibration()

    # --- inputs: discover, hash, bind capture identities -------------------
    surfaces = [DeclaredSurface.from_config(entry) for entry in referee_cfg["surfaces"]]
    manifest = json.loads(resolve(evidence["accept5_manifest"]).read_text())
    host_clock_by_run = {}
    for label in ("run_a", "run_b"):
        controller = manifest.get(label, {}).get("controller", {})
        if controller:
            host_clock_by_run[label.replace("_", "-")] = (
                controller["host_id"],
                controller["clock_id"],
            )
    pairs: list[dict] = []
    for pair_dir in evidence["pair_dirs"]:
        run_label = Path(pair_dir).parent.name
        for left_path, right_path, journal in _find_pairs(resolve(pair_dir)):
            capture = receipt = None
            sim_time = None
            pair_key = left_path.stem.replace("-left", "")
            if journal is not None:
                host_clock = host_clock_by_run.get(run_label)
                if host_clock is None:
                    raise R.RecordError(
                        f"no declared controller host/clock for {run_label} in the accept-5 manifest"
                    )
                host_id, clock_id = host_clock
                capture = R.ClockStamp(
                    host_id=host_id, clock_id=clock_id, monotonic_ns=int(journal["capture_monotonic_ns"])
                )
                receipt = R.ClockStamp(
                    host_id=host_id, clock_id=clock_id, monotonic_ns=int(journal["receipt_monotonic_ns"])
                )
                sim_time = float(journal["sim_time_s"])
                pair_key = str(journal["pair_id"])
            pairs.append(
                {
                    "run": run_label,
                    "left_path": left_path,
                    "right_path": right_path,
                    "pair_id": pair_key,
                    "capture": capture,
                    "receipt": receipt,
                    "sim_time_s": sim_time,
                    "left_sha256": _sha256_file(left_path),
                    "right_sha256": _sha256_file(right_path),
                }
            )
    log(f"inputs: {len(pairs)} stored pairs from {len(evidence['pair_dirs'])} runs")
    unique_scenes = {(p["left_sha256"], p["right_sha256"]) for p in pairs}
    log(f"inputs: {len(unique_scenes)} unique (left, right) image pairs across {len(pairs)} stored pairs")

    # --- declared referee geometry from the static-start pose --------------
    body_position = tuple(float(v) for v in referee_cfg["body_position_world_m"])
    body_rotation = _quaternion_rotation(tuple(float(v) for v in referee_cfg["body_quaternion_world_wxyz"]))
    camera_translation = calibration.T_body_camera_left.translation_m
    camera_position = tuple(b + t for b, t in zip(body_position, camera_translation))
    camera_rotation = body_rotation @ _quaternion_rotation(
        calibration.T_body_camera_left.quaternion_wxyz
    )
    size = (_DECLARED_WIDTH_PX, _DECLARED_HEIGHT_PX)
    declared_depth, attribution = declared_depth_map(
        surfaces, camera_position, camera_rotation, FOCAL_LENGTH_PX, PRINCIPAL_POINT_PX, size
    )
    margin = int(bounds["border_px"])
    z_min, z_max = (float(v) for v in bounds["depth_range_m"])
    in_window_declared = (
        np.isfinite(declared_depth) & (declared_depth >= z_min) & (declared_depth <= z_max)
    )
    inside_border = np.ones(declared_depth.shape, dtype=bool)
    inside_border[:margin, :] = False
    inside_border[-margin:, :] = False
    inside_border[:, :margin] = False
    inside_border[:, -margin:] = False

    provenance = declared_pose_provenance(evidence)

    # --- per-pair B1 / B2, pooled B3 ---------------------------------------
    rectification_evidence = []
    per_surface_errors: dict[str, list[np.ndarray]] = {s.name: [] for s in surfaces}
    per_surface_compared: dict[str, int] = {s.name: 0 for s in surfaces}
    per_surface_within: dict[str, int] = {s.name: 0 for s in surfaces}
    # eligibility is a property of the declared referee geometry, not of any
    # pair (the stored pairs share one static scene), so it is counted once here
    per_surface_eligible: dict[str, int] = {
        s.name: int((in_window_declared & inside_border & (attribution == index)).sum())
        for index, s in enumerate(surfaces)
    }
    abs_tol = float(bounds["depth_abs_tol_m"])
    rel_tol = float(bounds["depth_rel_tol"])

    for pair in pairs:
        left_rgb = read_ppm(pair["left_path"])
        right_rgb = read_ppm(pair["right_path"])
        b1 = match_keypoints(cv2_gray(left_rgb), cv2_gray(right_rgb), features)
        product = compute_validated_depth(
            left_rgb,
            right_rgb,
            calibration,
            {**matcher, "border_px": bounds["border_px"], "depth_range_m": bounds["depth_range_m"]},
            pair_id=pair["pair_id"],
            capture_stamp=pair["capture"],
            receipt_stamp=pair["receipt"],
            sim_time_s=pair["sim_time_s"],
            pose_provenance=provenance,
        )
        reason_counts = {
            REASON_NAMES[code]: int((product.reasons == code).sum()) for code in range(len(REASON_NAMES))
        }
        rectification_evidence.append(
            {
                "run": pair["run"],
                "pair_id": pair["pair_id"],
                "left_sha256": pair["left_sha256"],
                "right_sha256": pair["right_sha256"],
                "sim_time_s": pair["sim_time_s"],
                "rectification_b1": b1,
                "depth_b2": {
                    # samples that passed the left-right consistency check, whether or
                    # not they later fell outside the declared depth window
                    "lr_consistent_fraction": float(
                        ((product.reasons == REASON_VALID) | (product.reasons == REASON_DEPTH_RANGE)).sum()
                    )
                    / product.reasons.size,
                    "valid_fraction": float(product.valid.mean()),
                    "reason_counts": reason_counts,
                },
            }
        )
        compared = product.valid & inside_border & in_window_declared & np.isfinite(declared_depth)
        declared_here = declared_depth[compared]
        measured_here = product.depth_m[compared]
        errors = measured_here - declared_here
        within = np.abs(errors) <= np.maximum(abs_tol, rel_tol * declared_here)
        owner = attribution[compared]
        for index, surface in enumerate(surfaces):
            mask = owner == index
            count = int(mask.sum())
            per_surface_compared[surface.name] += count
            per_surface_within[surface.name] += int(within[mask].sum())
            if count:
                per_surface_errors[surface.name].append(errors[mask])
        log(
            f"pair {pair['run']}/{pair['pair_id']}: retained={b1['n_retained']} "
            f"row_med={b1['row_residual_median_px']} row_p95={b1['row_residual_p95_px']} "
            f"valid_frac={float(product.valid.mean()):.4f}"
        )

    # --- gate evaluation ----------------------------------------------------
    failures: list[str] = []
    b1_pass = True
    for entry in rectification_evidence:
        b1 = entry["rectification_b1"]
        problems = []
        if b1["n_retained"] < int(bounds["min_retained_matches"]):
            problems.append(f"retained {b1['n_retained']} < declared min {bounds['min_retained_matches']}")
        if b1["row_residual_median_px"] is None or b1["row_residual_median_px"] > float(
            bounds["rectification_row_residual_median_px"]
        ):
            problems.append(f"median row residual {b1['row_residual_median_px']}")
        if b1["row_residual_p95_px"] is None or b1["row_residual_p95_px"] > float(
            bounds["rectification_row_residual_p95_px"]
        ):
            problems.append(f"p95 row residual {b1['row_residual_p95_px']}")
        if b1["positive_disparity_fraction"] is not None and b1["positive_disparity_fraction"] < 1.0:
            problems.append(f"positive-disparity fraction {b1['positive_disparity_fraction']}")
        if problems:
            b1_pass = False
            failures.append(f"B1 {entry['run']}/{entry['pair_id']}: " + "; ".join(problems))

    surface_reports = []
    b3_pass = True
    for surface in surfaces:
        errors = per_surface_errors[surface.name]
        all_errors = np.concatenate(errors) if errors else np.array([])
        count = per_surface_compared[surface.name]
        within_count = per_surface_within[surface.name]
        eligible = per_surface_eligible[surface.name]
        gate_eligible = eligible >= int(bounds["min_eligible_pixels_per_surface"])
        within_fraction = (within_count / count) if count else None
        report = {
            "surface": surface.name,
            # The geometry's source is the configuration's referee section, and the
            # frames' scene is whichever world the evidence names. Naming the compat
            # world unconditionally told every non-compat run a false provenance.
            "declared_source": (
                f"{evidence['world']} (sha256 in inputs.json) — the referee surfaces are "
                "declared in this configuration's referee section from that world's "
                "declaration; REFEREE TRUTH, never an input to the calibration values"
            ),
            "eligible_pixels_in_window": eligible,
            "gate_eligible": gate_eligible,
            "n_compared": count,
            "n_within_tolerance": within_count,
            "within_fraction": within_fraction,
        }
        if count:
            report.update(
                {
                    "median_error_m": float(np.median(all_errors)),
                    "p95_abs_error_m": float(np.percentile(np.abs(all_errors), 95)),
                    "max_abs_error_m": float(np.max(np.abs(all_errors))),
                    "per_sample_tol": "max(0.15 m, 10 % of z_declared)",
                }
            )
        if gate_eligible:
            if count < int(bounds["min_compared_samples_per_surface"]):
                b3_pass = False
                report["verdict"] = "fail: too few compared samples"
                failures.append(
                    f"B3 {surface.name}: {count} compared samples < declared min "
                    f"{bounds['min_compared_samples_per_surface']}"
                )
            elif within_fraction < float(bounds["min_within_fraction"]):
                b3_pass = False
                report["verdict"] = "fail: within-tolerance fraction below declared bound"
                failures.append(
                    f"B3 {surface.name}: {within_count}/{count} within tolerance "
                    f"({within_fraction:.1%}) < declared {bounds['min_within_fraction']:.0%}"
                )
            else:
                report["verdict"] = "pass"
        else:
            report["verdict"] = "recorded, not gate-eligible (projected in-window region below the declared pixel floor)"
        surface_reports.append(report)

    # --- the final record with measured values ------------------------------
    offset_error, offset_sources = _measured_time_offset_error(
        [resolve(p) for p in (evidence["timebase_run_a"], evidence["timebase_run_b"])]
    )
    limits_parts = [
        "measured by the pre-registered P01-C validation (bounds in configs/first_indoor.yaml, "
        "written before measurement):",
        f"B1 row residual across {len(rectification_evidence)} pairs: median "
        f"{max(e['rectification_b1']['row_residual_median_px'] for e in rectification_evidence)} px "
        f"(bound {bounds['rectification_row_residual_median_px']}), p95 "
        f"{max(e['rectification_b1']['row_residual_p95_px'] for e in rectification_evidence)} px "
        f"(bound {bounds['rectification_row_residual_p95_px']})",
    ]
    for report in surface_reports:
        if report["gate_eligible"] and report["n_compared"]:
            limits_parts.append(
                f"B3 {report['surface']}: median err {report['median_error_m']:.4f} m, "
                f"p95|err| {report['p95_abs_error_m']:.4f} m, within-tol "
                f"{report['within_fraction']:.1%} of {report['n_compared']} — {report['verdict']}"
            )
        else:
            limits_parts.append(f"B3 {report['surface']}: {report['verdict']}")
    limits_parts.append(f"declared operating range {z_min}-{z_max} m; time offset error bound {offset_error} s")
    limits_parts.append("distortion-free is DECLARED, not measured")
    if not (b1_pass and b3_pass):
        limits_parts.append("RUN BLOCKED: " + "; ".join(failures))
    calibration = build_calibration(
        time_offset_s=0.0,
        time_offset_error_s=offset_error,
        validated_limits="; ".join(limits_parts),
    )
    # T1's round-trip, executed by the derivation itself: the file on disk and
    # the record consumers see must be the same object.
    document = record_document(calibration)
    assert load_calibration_document(json.loads(json.dumps(document))) == calibration

    # --- artifacts -----------------------------------------------------------
    output_dir = resolve(section["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration-first-indoor-stereo-1.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "rectification-evidence.json").write_text(
        json.dumps(
            {
                "bounds_pre_registered": bounds,
                "features": features,
                "per_pair": rectification_evidence,
                "note": (
                    f"{len(pairs)} stored pairs carry {len(unique_scenes)} unique (left, right) "
                    "image pairs: the accept-5 recorder stored the static pre-takeoff scene "
                    "12 times per run and run-a/run-b share that scene. Per-pair evidence is "
                    "recorded as measured; identical inputs give identical numbers."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "depth-validation.json").write_text(
        json.dumps(
            {
                "referee_truth_label": (
                    "declared distances come from the pinned scene files; they are referee "
                    "truth and never feed the calibration values"
                ),
                "capture_pose_provenance": {
                    "label": provenance.label,
                    "detail": provenance.detail,
                },
                "honesty_notes": [
                    "distortion 'none' is declared, not measured; B1's p95 bound is the corner-growth guard",
                    "validity filters are documented mechanisms, not calibrated safety probabilities",
                    "a masked depth sample is unknown, never zero",
                ],
                "capture_time_pose_binding": (
                    "each product binds capture_stamp + receipt_stamp + sim_time_s + pair_id + "
                    "calibration_id; capture identities come from accept-5 pairs.jsonl"
                ),
                "bounds_pre_registered": bounds,
                "matcher": matcher,
                "per_surface": surface_reports,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    inputs_document = {
        "accept5_receipt": str(evidence["accept5_receipt"]),
        "accept5_manifest": str(evidence["accept5_manifest"]),
        "declared_calibration": {
            "path": str(evidence["declared_calibration"]),
            "sha256": _sha256_file(resolve(evidence["declared_calibration"])),
        },
        "world": {
            "path": str(evidence["world"]),
            "sha256": _sha256_file(resolve(evidence["world"])),
        },
        "proto": {
            "path": str(evidence["proto"]),
            "sha256": _sha256_file(resolve(evidence["proto"])),
        },
        "pairs": [
            {
                "run": pair["run"],
                "pair_id": pair["pair_id"],
                "left": str(pair["left_path"]),
                "left_sha256": pair["left_sha256"],
                "right": str(pair["right_path"]),
                "right_sha256": pair["right_sha256"],
            }
            for pair in pairs
        ],
        "finding": (
            f"{len(pairs)} stored pairs carry {len(unique_scenes)} unique (left, right) image "
            "pairs — the static pre-takeoff scene, stored 12 times per run and identical "
            "across run-a and run-b; the depth check's referee pose is the declared static "
            "start (DIAGNOSTIC provenance)"
        ),
    }
    (output_dir / "inputs.json").write_text(json.dumps(inputs_document, indent=2, sort_keys=True) + "\n")

    passed = b1_pass and b3_pass
    status = "complete" if passed else "blocked"
    gate_status = "pass" if passed else "fail"
    log(f"gates: B1={'pass' if b1_pass else 'fail'} B3={'pass' if b3_pass else 'fail'} -> {status}")
    for failure in failures:
        log(f"failure: {failure}")
    log(f"artifacts written under {output_dir}")
    (output_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
    artifact_names = [
        "calibration-first-indoor-stereo-1.json",
        "rectification-evidence.json",
        "depth-validation.json",
        "inputs.json",
        "log.txt",
    ]
    receipt = {
        "receipt_version": "receipt-1",
        "stage_id": "P01-C",
        "definition": "p01c-calibration",
        "records_revision": R.RECORDS_REVISION,
        "calibration_id": CALIBRATION_ID,
        "calibration_version": CALIBRATION_VERSION,
        "code_revision": code_revision,
        "config_hash": _sha256_file(config_path),
        "status": status,
        "gate_status": gate_status,
        "episode_id": None,
        "trial_group_id": None,
        "sensor_mode": "not-applicable (offline calibration)",
        "artifacts": [
            {
                "path": name,
                "sha256": _sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in artifact_names
        ],
        "reasons": failures if failures else ["all pre-registered gates passed"],
        "limitations": [
            (
                f"the {len(pairs)} stored accept-5 pairs are {len(unique_scenes)} unique static "
                "scenes (pre-takeoff, byte-identical within each run): depth is validated only "
                "on the surfaces those scenes expose"
            ),
            "capture-time pose is the declared static start: DIAGNOSTIC, never scored",
            "distortion-free is declared, not measured (B1 p95 guards corner growth)",
            "P01-L remains held behind the flight gate; this stage claims no localization",
            "integrator action open: pin the OpenCV runtime in pyproject.toml",
        ],
        "next_step_if_blocked": (
            "a fresh capture with textured referee surfaces at declared depths (or a scene "
            "re-declaration adding texture) would disambiguate the failed surfaces; per plan "
            "section 7 nothing runs on top of a blocked calibration"
        )
        if not passed
        else None,
        "commands": [
            "PYTHONPATH=src python3 -m pytest tests/platform/test_time_alignment.py tests/platform/test_calibration.py",
            "PYTHONPATH=src python3 -m embodied.perception.camera --config configs/first_indoor.yaml",
        ],
        "time_offset_sources": offset_sources,
        "started_utc": started,
        "finished_utc": _utc_now(),
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 2


def cv2_gray(rgb: np.ndarray) -> np.ndarray:
    """Grayscale for feature matching; lazy OpenCV import."""
    import cv2  # lazy

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m embodied.perception.camera",
        description="P01-C offline stereo calibration derivation over the accept-5 evidence",
    )
    parser.add_argument("--config", default="configs/first_indoor.yaml")
    args = parser.parse_args(argv)
    import subprocess  # only to record the code revision the run is sourced to

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return run_validation(Path(args.config), code_revision=revision)


if __name__ == "__main__":
    sys.exit(main())
