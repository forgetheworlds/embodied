"""The one grounding path: a visual selection plus valid depth plus capture pose become geometry.

Specification section 10 states the problem and its only accepted solution. A
selection is not geometry; an image coordinate has no metric meaning on its own;
using the aircraft's current pose to transform an old image is invalid. So every
selection kind — point, box, polygon, mask or phrase — funnels into this single
geometric core:

1. Resolve the selection's pixels in the observation's own coordinate convention.
2. Keep only pixels whose depth sample is **valid**. Invalid depth grounds
   nothing and clears nothing (section 7.1).
3. Build each viewing ray from the record's declared pixel convention and
   intrinsics, and place the point at the product's declared depth convention.
4. Transform with the **capture-time** pose, never the current one.
5. Fit the surface and construct the conservative geometry, with the measurement
   envelope propagated through the linear transform and the shared pose bound
   added linearly (section 7.3).
6. Emit a :class:`embodied.contracts.records.GroundedTarget` citing the
   selection and observation it came from, with an anchor identity so a later map
   correction can rebase it.

**There is no second route.** A phrase selection takes the same steps after the
pinned detector seam supplies a candidate region; when no verified detector
exists the result is an explicit ``detector_unavailable`` refusal, never a
substituted truth value. There is no image-relative shortcut for traversal,
clearance or standoff (section 10.3).

**Refusals are uncertain, never infeasible** (section 12.1). The refusal tokens
are part of the seam: ``stale_pose``, ``missing_depth``, ``unknown_geometry``,
``frame_epoch_mismatch``, ``calibration_mismatch``, ``detector_unavailable``.

Declared engineering parameters of this stage (authored, not measured — the same
values the doorway fixture declares in its ``truth/scene.json``):
``POSE_VALIDITY_S``, ``THROUGH_MARGIN_M``, ``EDGE_SHRINK_M``,
``FIT_TOLERANCE_M``, ``MIN_INLIER_FRACTION``, ``SIGMA_PIXEL_PX``.
The fixture's MANUAL-VALUES.md names each one, and
``tests/navigation/test_goal_geometry.py`` asserts that the fixture's declared
values equal these constants, so a drift between the two is loud rather than
silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embodied.contracts import records as R
from embodied.perception import camera as camera_module
from embodied.perception import detector as detector_module

# Refusal tokens. Exact strings: a caller switches on these, so they do not drift.
REFUSAL_STALE_POSE = "stale_pose"
REFUSAL_MISSING_DEPTH = "missing_depth"
REFUSAL_UNKNOWN_GEOMETRY = "unknown_geometry"
REFUSAL_FRAME_EPOCH_MISMATCH = "frame_epoch_mismatch"
REFUSAL_CALIBRATION_MISMATCH = "calibration_mismatch"
REFUSAL_DETECTOR_UNAVAILABLE = "detector_unavailable"

REFUSAL_REASONS = (
    REFUSAL_STALE_POSE,
    REFUSAL_MISSING_DEPTH,
    REFUSAL_UNKNOWN_GEOMETRY,
    REFUSAL_FRAME_EPOCH_MISMATCH,
    REFUSAL_CALIBRATION_MISMATCH,
    REFUSAL_DETECTOR_UNAVAILABLE,
)

# Declared engineering parameters (AUTHORED; mirrored by the fixture's truth/).
POSE_VALIDITY_S = 5.0
THROUGH_MARGIN_M = 0.5
EDGE_SHRINK_M = 0.02
FIT_TOLERANCE_M = 0.05
SEED_BAND_M = 0.10
SEED_MIN_FRACTION = 0.10
MIN_INLIER_FRACTION = 0.25
SIGMA_PIXEL_PX = 1.0
PLANE_ITERATIONS = 3

# GroundedTarget.geometry holds one of two documented layouts.
POINT_NUMBERS = 3
APERTURE_NUMBERS = 18


@dataclass(frozen=True)
class Anchor:
    """The submap anchor a target's geometry is stored in."""

    submap_id: str
    revision: str
    frame: R.Frame = R.Frame.ODOM

    def __post_init__(self) -> None:
        for name in ("submap_id", "revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise R.RecordError(f"anchor {name} must be a non-empty string")
        if not isinstance(self.frame, R.Frame):
            raise R.RecordError("an anchor frame is a Frame value")
        if self.frame is R.Frame.MAP:
            raise R.RecordError(
                "an anchor is a submap identity, not the corrected map frame; a target is "
                "resolved into odom before it is used as a control reference"
            )


@dataclass(frozen=True)
class Refusal:
    """A refusal to ground, with the reason a consumer switches on.

    ``support`` is always ``uncertain``: a refusal says the evidence is missing,
    stale, mismatched or inconsistent. It never claims the goal is physically
    infeasible (section 12.1).
    """

    reason: str
    detail: str
    selection_id: str | None = None
    observation_id: str | None = None
    support: str = "uncertain"

    def __post_init__(self) -> None:
        if self.reason not in REFUSAL_REASONS:
            raise R.RecordError(f"{self.reason!r} is not a grounding refusal reason")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise R.RecordError("a refusal states why it refused")
        if self.support != "uncertain":
            raise R.RecordError("a grounding refusal is uncertain, never infeasible")


@dataclass(frozen=True)
class Aperture:
    """An observed opening: a plane, a conservative rectangle on it, and its envelope."""

    plane_point_odom_m: tuple[float, float, float]
    plane_normal_odom: tuple[float, float, float]
    corners_odom_m: tuple[tuple[float, float, float], ...]
    width_m: float
    height_m: float
    uncertainty_m: tuple[float, float, float]

    def center_odom_m(self) -> tuple[float, float, float]:
        corners = np.asarray(self.corners_odom_m, dtype=np.float64)
        return tuple(float(value) for value in corners.mean(axis=0))


def default_anchor(observation: R.Observation) -> Anchor:
    """The observation's submap anchor at the revision this stage publishes."""
    return Anchor(submap_id=f"submap-{observation.episode_id}", revision="rev-1")


def _refusal(
    reason: str,
    detail: str,
    selection: R.VisualSelection | None = None,
    observation: R.Observation | None = None,
) -> Refusal:
    return Refusal(
        reason=reason,
        detail=detail,
        selection_id=None if selection is None else selection.selection_id,
        observation_id=None if observation is None else observation.record_id,
    )


def _rotation(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(transform: R.Transform, points: np.ndarray) -> np.ndarray:
    """Apply T_A_B to an (N, 3) array of points expressed in B."""
    rotation = _rotation(transform.quaternion_wxyz)
    translation = np.asarray(transform.translation_m, dtype=np.float64)
    return points @ rotation.T + translation


def pixel_ray_directions(pixels: np.ndarray, calibration: R.Calibration) -> np.ndarray:
    """Per-pixel viewing directions in the camera's own frame, with the forward axis = 1.

    This is the record's pixel convention in one place: columns run along body -y
    and rows along body -z, the optical axis is body +x, and the direction's x
    component is 1 so the declared optical-axis depth is the ray parameter.
    """
    focal = calibration.left_intrinsics.focal_length_px[0]
    principal = calibration.left_intrinsics.principal_point_px
    u = np.asarray(pixels, dtype=np.float64)[:, 0]
    v = np.asarray(pixels, dtype=np.float64)[:, 1]
    return np.stack(
        [np.ones_like(u), -(u - principal[0]) / focal, -(v - principal[1]) / focal], axis=1
    )


def rotation_of(pose: R.PoseEstimate) -> np.ndarray:
    """The rotation a pose applies when it maps its child frame into its parent."""
    return _rotation(pose.quaternion_wxyz)


def camera_points_to_odom(
    pixel_points: np.ndarray, depth_m: np.ndarray, calibration: R.Calibration, pose: R.PoseEstimate
) -> np.ndarray:
    """Declared pixels plus their optical-axis depth, in odom, using the capture-time pose.

    The record's pixel convention is authoritative: columns run along body -y,
    rows along body -z, and the camera optical axis is body +x. With the ray
    parameterisation used here the direction's x component is 1, so the declared
    optical-axis depth is also the ray parameter.
    """
    focal = calibration.left_intrinsics.focal_length_px[0]
    principal = calibration.left_intrinsics.principal_point_px
    u = pixel_points[:, 0]
    v = pixel_points[:, 1]
    directions = np.stack(
        [np.ones_like(u), -(u - principal[0]) / focal, -(v - principal[1]) / focal], axis=1
    )
    camera_points = directions * depth_m[:, None]
    body_points = _transform_points(calibration.T_body_camera_left, camera_points)
    return _transform_points(
        R.Transform(
            parent_frame="odom",
            child_frame="body",
            translation_m=pose.position_m,
            quaternion_wxyz=pose.quaternion_wxyz,
        ),
        body_points,
    )


def _pixels_in_polygon(polygon: tuple[float, ...], width: int, height: int) -> np.ndarray:
    """Even-odd point-in-polygon over the integer pixel grid, clipped to the image."""
    coordinates = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if coordinates.shape[0] < 3:
        raise R.RecordError("a polygon selection needs at least three vertices")
    u_min = max(0, int(np.floor(coordinates[:, 0].min())))
    u_max = min(width - 1, int(np.ceil(coordinates[:, 0].max())))
    v_min = max(0, int(np.floor(coordinates[:, 1].min())))
    v_max = min(height - 1, int(np.ceil(coordinates[:, 1].max())))
    if u_max < u_min or v_max < v_min:
        return np.empty((0, 2), dtype=np.int64)
    u = np.arange(u_min, u_max + 1)
    v = np.arange(v_min, v_max + 1)
    grid_u, grid_v = np.meshgrid(u, v)
    inside = np.zeros(grid_u.shape, dtype=bool)
    count = coordinates.shape[0]
    for index in range(count):
        u_a, v_a = coordinates[index]
        u_b, v_b = coordinates[(index + 1) % count]
        if (v_a > grid_v) != (v_b > grid_v):
            crossing_u = (u_b - u_a) * (grid_v - v_a) / (v_b - v_a) + u_a
            inside ^= grid_u < crossing_u
    return np.stack([grid_u[inside], grid_v[inside]], axis=1).astype(np.int64)


def selection_pixels(selection: R.VisualSelection, width: int, height: int) -> np.ndarray:
    """The (u, v) integer pixels a selection covers, in the image's own convention."""
    geometry = selection.geometry
    if selection.geometry_kind is R.SelectionGeometry.POINT:
        u, v = int(round(geometry[0])), int(round(geometry[1]))
        if not (0 <= u < width and 0 <= v < height):
            return np.empty((0, 2), dtype=np.int64)
        return np.array([[u, v]], dtype=np.int64)
    if selection.geometry_kind is R.SelectionGeometry.BOX:
        u0, v0, u1, v1 = (int(np.floor(geometry[0])), int(np.floor(geometry[1])),
                          int(np.ceil(geometry[2])), int(np.ceil(geometry[3])))
        u0, u1 = max(0, u0), min(width - 1, u1)
        v0, v1 = max(0, v0), min(height - 1, v1)
        if u1 < u0 or v1 < v0:
            return np.empty((0, 2), dtype=np.int64)
        grid_u, grid_v = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        return np.stack([grid_u.ravel(), grid_v.ravel()], axis=1).astype(np.int64)
    if selection.geometry_kind is R.SelectionGeometry.POLYGON:
        return _pixels_in_polygon(geometry, width, height)
    raise R.RecordError(
        "a mask selection is resolved by the detector seam before grounding; call "
        "ground() with the phrase selection and a detector"
    )


def _pose_bound_m(pose: R.PoseEstimate) -> float:
    """The largest declared 1-sigma pose bound, or 0.0 when the pose declares none."""
    if pose.covariance is None:
        return 0.0
    variances = np.asarray(pose.covariance, dtype=np.float64)
    if variances.size < 3:
        return float(np.sqrt(max(variances.max(), 0.0)))
    return float(np.sqrt(max(variances[:3].max(), 0.0)))


def _propagate_envelope(
    rotation: np.ndarray, sigma_forward_m: float, sigma_lateral_m: float, pose_bound_m: float
) -> tuple[float, float, float]:
    """Rotate the measurement envelope into odom and add the shared pose bound linearly.

    Section 7.3: propagate through the local linear approximation, and add
    separately validated bounds rather than multiplying marginal confidences.
    The depth envelope and the shared pose bound may move a target together, so
    the returned per-axis envelope is their linear sum and is a bound, not a
    probability.
    """
    local = np.array([sigma_forward_m, sigma_lateral_m, sigma_lateral_m], dtype=np.float64)
    rotated = np.abs(rotation) @ local
    return tuple(float(value + pose_bound_m) for value in rotated)


def _fit_plane(
    points: np.ndarray, depths: np.ndarray, camera_center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Robustly fit one surface to the selected points, rejecting other surfaces.

    Section 10.2: reject points that cannot agree with one surface instead of
    averaging foreground geometry together with the depth behind an opening. The
    seed is the *nearest well-supported surface*: depths are binned at the
    declared seed width, and the closest bin holding at least the declared share
    of the samples wins. An aperture lives in the surface the selection was drawn
    on, and the depth seen through it is further away by construction, so the
    nearest supported surface is the frame rather than the room beyond it. A
    selection dominated by a nearer occluder therefore fits that occluder and
    finds no opening through it, which returns a refusal rather than invented
    geometry. The plane is refit on points that agree with it; the final inlier
    set is what the caller uses.
    """
    bins = np.round(depths / SEED_BAND_M).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    minimum = max(3, int(np.ceil(SEED_MIN_FRACTION * depths.shape[0])))
    supporters = np.nonzero(counts >= minimum)[0]
    if supporters.size == 0:
        return None
    seed = bins == unique[supporters[0]]
    inliers = seed
    normal = None
    for _ in range(PLANE_ITERATIONS):
        if int(inliers.sum()) < 3:
            return None
        selected = points[inliers]
        centroid = selected.mean(axis=0)
        centred = selected - centroid
        _values, vectors = np.linalg.eigh(centred.T @ centred)
        normal = vectors[:, 0]
        if normal @ (centroid - camera_center) < 0.0:
            normal = -normal
        offset = float(normal @ centroid)
        distances = np.abs(points @ normal - offset)
        inliers = distances <= FIT_TOLERANCE_M
    if normal is None:
        return None
    selected = points[inliers]
    centroid = selected.mean(axis=0)
    point = normal * float(normal @ centroid)
    fraction = float(inliers.sum()) / float(points.shape[0])
    return normal, point, float(normal @ point), fraction


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two in-plane axes, chosen deterministically from the world axis least aligned with n."""
    alignments = np.abs(normal)
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(alignments))] = 1.0
    axis_one = reference - float(reference @ normal) * normal
    axis_one = axis_one / np.linalg.norm(axis_one)
    axis_two = np.cross(normal, axis_one)
    return axis_one, axis_two


@dataclass(frozen=True)
class FittedAperture:
    """The robust plane fit and the conservative opening rectangle on it."""

    plane_point_odom_m: tuple[float, float, float]
    plane_normal_odom: tuple[float, float, float]
    corners_odom_m: tuple[tuple[float, float, float], ...]
    width_m: float
    height_m: float
    wall_sigma_m: float
    wall_depth_m: float
    inlier_fraction: float


def _aperture_from_points(
    points: np.ndarray,
    depths: np.ndarray,
    uncertainties: np.ndarray,
    directions_odom: np.ndarray,
    camera_center: np.ndarray,
) -> FittedAperture | None:
    """The conservative opening rectangle on the fitted plane, from through-pixels only.

    Section 10.2: fit the surrounding wall plane robustly, project the observed
    boundaries into that plane, and build a conservative free opening polygon
    shrunk by the edge uncertainty. The opening plane is not the depth behind it:
    rays that pass the opening end on the far wall, and only their intersection
    with the wall plane bounds the opening.
    """
    fit = _fit_plane(points, depths, camera_center)
    if fit is None:
        return None
    normal, plane_point, offset, fraction = fit
    if fraction < MIN_INLIER_FRACTION:
        return None
    axis_one, axis_two = _plane_basis(normal)
    plane_component = directions_odom @ normal
    plane_offset_from_centre = offset - float(camera_center @ normal)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = plane_offset_from_centre / plane_component
    wall = np.isfinite(t_plane) & (np.abs(depths - t_plane) <= FIT_TOLERANCE_M)
    passed = np.isfinite(t_plane) & (depths > t_plane + THROUGH_MARGIN_M)
    if not bool(passed.any()) or not bool(wall.any()):
        return None
    hit = camera_center[None, :] + t_plane[passed, None] * directions_odom[passed]
    relative = hit - plane_point[None, :]
    coordinates = np.stack([relative @ axis_one, relative @ axis_two], axis=1)
    low = coordinates.min(axis=0) + EDGE_SHRINK_M
    high = coordinates.max(axis=0) - EDGE_SHRINK_M
    if not (high[0] > low[0] and high[1] > low[1]):
        return None
    corners = tuple(
        tuple(float(value) for value in plane_point + corner[0] * axis_one + corner[1] * axis_two)
        for corner in ((low[0], low[1]), (high[0], low[1]), (high[0], high[1]), (low[0], high[1]))
    )
    return FittedAperture(
        plane_point_odom_m=tuple(float(value) for value in plane_point),
        plane_normal_odom=tuple(float(value) for value in normal),
        corners_odom_m=corners,
        width_m=float(high[0] - low[0]),
        height_m=float(high[1] - low[1]),
        wall_sigma_m=float(np.median(uncertainties[wall])),
        wall_depth_m=float(np.median(depths[wall])),
        inlier_fraction=fraction,
    )


def parse_point(geometry: tuple[float, ...]) -> tuple[float, float, float] | None:
    """A point target's geometry, or None when the payload is not a point."""
    if len(geometry) != POINT_NUMBERS:
        return None
    return (float(geometry[0]), float(geometry[1]), float(geometry[2]))


def parse_aperture(geometry: tuple[float, ...]) -> Aperture | None:
    """An aperture target's geometry, or None when the payload is not an aperture."""
    if len(geometry) != APERTURE_NUMBERS:
        return None
    values = np.asarray(geometry, dtype=np.float64)
    point = tuple(float(value) for value in values[0:3])
    normal = tuple(float(value) for value in values[3:6])
    corners = tuple(
        tuple(float(value) for value in values[6 + 3 * index : 9 + 3 * index])
        for index in range(4)
    )
    axis_one, axis_two = _plane_basis(np.asarray(normal))
    relative = np.asarray(corners[0]) - np.asarray(point)
    low = np.array([relative @ axis_one, relative @ axis_two])
    relative = np.asarray(corners[2]) - np.asarray(point)
    high = np.array([relative @ axis_one, relative @ axis_two])
    return Aperture(
        plane_point_odom_m=point,
        plane_normal_odom=normal,
        corners_odom_m=corners,
        width_m=float(high[0] - low[0]),
        height_m=float(high[1] - low[1]),
        uncertainty_m=(0.0, 0.0, 0.0),
    )


def _calibration_refusal(
    observation: R.Observation,
    depth: camera_module.DepthProduct | None,
    calibration: R.Calibration,
    selection: R.VisualSelection,
) -> Refusal | None:
    if observation.calibration_id != calibration.calibration_id:
        return _refusal(
            REFUSAL_CALIBRATION_MISMATCH,
            f"the observation names calibration {observation.calibration_id!r} but grounding was "
            f"given {calibration.calibration_id!r}",
            selection,
            observation,
        )
    if depth is not None and (
        depth.calibration_id != observation.calibration_id
        or depth.calibration_version != calibration.version
    ):
        return _refusal(
            REFUSAL_CALIBRATION_MISMATCH,
            "the depth product names calibration "
            f"{depth.calibration_id!r} v{depth.calibration_version!r} but the observation names "
            f"{observation.calibration_id!r} v{calibration.version!r}",
            selection,
            observation,
        )
    return None


def _pose_refusal(
    observation: R.Observation,
    capture_pose: R.PoseEstimate,
    current_state: R.NavigationState,
    selection: R.VisualSelection,
) -> Refusal | None:
    if not capture_pose.valid:
        return _refusal(
            REFUSAL_STALE_POSE,
            "the capture-time pose is marked invalid, so the selection cannot be placed",
            selection,
            observation,
        )
    if not current_state.pose.valid:
        return _refusal(
            REFUSAL_STALE_POSE,
            "the current state's pose is marked invalid, so the capture pose cannot be checked",
            selection,
            observation,
        )
    if (capture_pose.parent_frame, capture_pose.child_frame) != ("odom", "body"):
        return _refusal(
            REFUSAL_FRAME_EPOCH_MISMATCH,
            "the capture-time pose maps "
            f"{capture_pose.child_frame!r} into {capture_pose.parent_frame!r}; grounding places "
            "geometry in odom and will not convert an undeclared frame pair",
            selection,
            observation,
        )
    age_ns = R.elapsed_ns(capture_pose.stamp, current_state.pose.stamp)
    if age_ns < 0:
        return _refusal(
            REFUSAL_STALE_POSE,
            "the capture-time pose is stamped after the current state; the pair is inconsistent "
            "and a current pose on a future image is not a usable transform",
            selection,
            observation,
        )
    if age_ns > POSE_VALIDITY_S * 1e9:
        return _refusal(
            REFUSAL_STALE_POSE,
            f"the capture-time pose is {age_ns / 1e9:.3f} s older than the current state, beyond "
            f"the declared {POSE_VALIDITY_S:.1f} s validity; transforming an old image with "
            "today's pose is invalid",
            selection,
            observation,
        )
    if capture_pose.nav_epoch != current_state.nav_epoch:
        return _refusal(
            REFUSAL_FRAME_EPOCH_MISMATCH,
            f"the capture-time pose belongs to nav_epoch {capture_pose.nav_epoch!r} but the "
            f"current state is {current_state.nav_epoch!r}; a reset invalidates prior control "
            "references",
            selection,
            observation,
        )
    return None


def _phrase_candidate(
    selection: R.VisualSelection,
    detector,
    detector_image,
    observation: R.Observation,
) -> tuple[R.VisualSelection, tuple[str, ...], tuple[str, ...] | None] | Refusal:
    """Resolve a phrase selection through the pinned detector seam, or refuse explicitly."""
    if detector is None:
        return _refusal(
            REFUSAL_DETECTOR_UNAVAILABLE,
            "a phrase selection needs the pinned goal-conditioned detector seam, and none was "
            "supplied; grounding will not substitute an authored region for a detected one",
            selection,
            observation,
        )
    result = detector.candidates(detector_image, str(selection.geometry))
    if isinstance(result, detector_module.DetectorUnavailable):
        return _refusal(
            REFUSAL_DETECTOR_UNAVAILABLE,
            f"{result.detail} ({result.reason})",
            selection,
            observation,
        )
    if not result:
        return _refusal(
            REFUSAL_DETECTOR_UNAVAILABLE,
            "the detector seam proposed no candidate for this query",
            selection,
            observation,
        )
    best = max(result, key=lambda candidate: candidate.score if candidate.score is not None else 0.0)
    if best.region is None:
        return _refusal(
            REFUSAL_UNKNOWN_GEOMETRY,
            f"candidate {best.candidate_id!r} carries only a mask reference; this stage consumes "
            "box regions and records the gap rather than inventing a region",
            selection,
            observation,
        )
    alternatives = tuple(
        candidate.candidate_id for candidate in result if candidate.candidate_id != best.candidate_id
    )
    resolved = R.VisualSelection(
        selection_id=selection.selection_id,
        observation_id=selection.observation_id,
        coordinate_convention=selection.coordinate_convention,
        geometry_kind=R.SelectionGeometry.BOX,
        geometry=tuple(float(value) for value in best.region),
        crop_transform=selection.crop_transform,
        description=f"{selection.description or str(selection.geometry)} (candidate {best.candidate_id})",
        confidence=best.score,
    )
    return resolved, (selection.selection_id, best.candidate_id), (alternatives or None)


def ground(
    selection: R.VisualSelection,
    observation: R.Observation,
    depth,
    capture_pose: R.PoseEstimate,
    current_state: R.NavigationState,
    calibration: R.Calibration,
    *,
    anchor: Anchor | None = None,
    detector=None,
    detector_image=None,
) -> R.GroundedTarget | Refusal:
    """Resolve one selection against its own observation into anchor-frame geometry.

    Check order is part of the seam and is documented so a caller can predict the
    token it gets: calibration binding, then capture-pose validity and age, then
    nav_epoch, then depth presence, then selection resolution, then geometry. A
    refusal is uncertain and never infeasible.
    """
    resolved_anchor = anchor if anchor is not None else default_anchor(observation)
    refusal = _calibration_refusal(observation, depth, calibration, selection)
    if refusal is not None:
        return refusal
    refusal = _pose_refusal(observation, capture_pose, current_state, selection)
    if refusal is not None:
        return refusal
    if depth is None:
        return _refusal(
            REFUSAL_MISSING_DEPTH,
            "no depth product was supplied for this observation; a selection without depth is "
            "not geometry",
            selection,
            observation,
        )
    if depth.capture_stamp is not None and depth.pair_id is not None:
        if depth.capture_stamp.monotonic_ns != observation.capture_stamp.monotonic_ns:
            return _refusal(
                REFUSAL_MISSING_DEPTH,
                "the depth product was captured at "
                f"{depth.capture_stamp.monotonic_ns} ns but the observation at "
                f"{observation.capture_stamp.monotonic_ns} ns: the samples are not this frame's",
                selection,
                observation,
            )
    working = selection
    cited = (selection.selection_id,)
    alternatives: tuple[str, ...] | None = None
    if selection.geometry_kind is R.SelectionGeometry.MASK:
        resolution = _phrase_candidate(selection, detector, detector_image, observation)
        if isinstance(resolution, Refusal):
            return resolution
        working, cited, alternatives = resolution

    height, width = int(depth.valid.shape[0]), int(depth.valid.shape[1])
    pixels = selection_pixels(working, width, height)
    if pixels.size == 0:
        return _refusal(
            REFUSAL_UNKNOWN_GEOMETRY,
            "the selection covers no pixel of this image",
            selection,
            observation,
        )
    rows = pixels[:, 1]
    columns = pixels[:, 0]
    valid = depth.valid[rows, columns]
    if not bool(valid.any()):
        return _refusal(
            REFUSAL_UNKNOWN_GEOMETRY,
            f"none of the {pixels.shape[0]} selected samples carries valid depth; invalid depth "
            "grounds nothing and clears nothing",
            selection,
            observation,
        )
    selected_pixels = pixels[valid]
    selected_depths = np.asarray(depth.depth_m[rows, columns][valid], dtype=np.float64)
    points = camera_points_to_odom(selected_pixels, selected_depths, calibration, capture_pose)
    focal = calibration.left_intrinsics.focal_length_px[0]
    principal = calibration.left_intrinsics.principal_point_px
    directions_odom = np.stack(
        [
            np.ones_like(selected_pixels[:, 0], dtype=np.float64),
            -(selected_pixels[:, 0] - principal[0]) / focal,
            -(selected_pixels[:, 1] - principal[1]) / focal,
        ],
        axis=1,
    )
    rotation = _rotation(capture_pose.quaternion_wxyz) @ _rotation(
        calibration.T_body_camera_left.quaternion_wxyz
    )
    camera_center = np.asarray(points[0], dtype=np.float64) - selected_depths[0] * (
        rotation @ directions_odom[0]
    )
    pose_bound = _pose_bound_m(capture_pose)

    if working.geometry_kind is R.SelectionGeometry.POINT or selected_pixels.shape[0] == 1:
        measured_sigma = float(depth.uncertainty_m[selected_pixels[0, 1], selected_pixels[0, 0]])
        sigma_forward = measured_sigma if np.isfinite(measured_sigma) else 0.0
        sigma_lateral = float(selected_depths[0]) * SIGMA_PIXEL_PX / focal
        geometry = tuple(float(value) for value in points[0])
        uncertainty = _propagate_envelope(rotation, sigma_forward, sigma_lateral, pose_bound)
    else:
        measured = np.asarray(
            depth.uncertainty_m[rows, columns][valid], dtype=np.float64
        )
        measured = np.where(np.isfinite(measured), measured, 0.0)
        fitted = _aperture_from_points(
            points, selected_depths, measured, directions_odom, camera_center
        )
        if fitted is None:
            return _refusal(
                REFUSAL_UNKNOWN_GEOMETRY,
                "the selected samples do not agree with one surface plus an observed free opening: "
                "either no consistent plane was supported or no ray was seen past it",
                selection,
                observation,
            )
        geometry = (
            *fitted.plane_point_odom_m,
            *fitted.plane_normal_odom,
            *(value for corner in fitted.corners_odom_m for value in corner),
        )
        sigma_lateral = fitted.wall_depth_m * SIGMA_PIXEL_PX / focal
        propagated = _propagate_envelope(rotation, fitted.wall_sigma_m, sigma_lateral, pose_bound)
        uncertainty = (propagated[0], propagated[1], EDGE_SHRINK_M)

    return R.GroundedTarget(
        target_id=f"gt-{selection.selection_id}-{observation.record_id}",
        track_id=None,
        place_id=None,
        selection_ids=cited,
        observation_ids=(observation.record_id,),
        geometry=tuple(geometry),
        frame=resolved_anchor.frame,
        anchor_id=resolved_anchor.submap_id,
        anchor_revision=resolved_anchor.revision,
        uncertainty=tuple(uncertainty),
        identity_alternatives=alternatives,
        last_observed_stamp=observation.capture_stamp,
        valid=True,
    )


def target_currency(
    target: R.GroundedTarget,
    current_state: R.NavigationState,
    *,
    anchor_revision: str | None = None,
) -> Refusal | None:
    """Whether a target may still be acted on: current frame epoch and anchor revision.

    A consumer resolves a target through one consistent snapshot and revalidates
    before use (CONTRACTS.md, section 7.2). The target carries no nav_epoch field
    of its own, so the epoch travels with the anchor revision the caller supplies
    and with the capture pose that produced it; this helper checks the target's
    own validity and anchor revision, and the caller checks the epoch it holds.
    """
    if not target.valid:
        return Refusal(
            reason=REFUSAL_UNKNOWN_GEOMETRY,
            detail=f"target {target.target_id!r} is marked invalid",
            selection_id=target.selection_ids[0] if target.selection_ids else None,
        )
    if anchor_revision is not None and anchor_revision != target.anchor_revision:
        return Refusal(
            reason=REFUSAL_FRAME_EPOCH_MISMATCH,
            detail=(
                f"target {target.target_id!r} is stored at anchor revision "
                f"{target.anchor_revision!r}, but the current snapshot is {anchor_revision!r}"
            ),
            selection_id=target.selection_ids[0] if target.selection_ids else None,
        )
    if not current_state.pose.valid:
        return Refusal(
            reason=REFUSAL_STALE_POSE,
            detail="the current state's pose is invalid, so no target can be revalidated",
            selection_id=target.selection_ids[0] if target.selection_ids else None,
        )
    return None
