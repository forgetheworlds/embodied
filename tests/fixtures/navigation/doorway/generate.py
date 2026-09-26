"""Deterministic generator for the P03 doorway fixture (hand-authored, DIAGNOSTIC).

This directory is one episode of P02's storage: ``manifest.json``,
``agent-events.jsonl`` and ``payloads/`` are the agent projection a run may read,
and ``truth/`` holds the hand-derived expected answer beside them, unreachable
through :class:`embodied.bench.recorder.AgentSurface` because the manifest's
projection lock enumerates projection members only (recorder.py:509-528).

Everything here is authored or derived in closed form from the authored scene;
nothing is measured on a vehicle and nothing is scored. The capture pose is
simulator-scene truth, so the whole fixture is provenance ``DIAGNOSTIC`` and can
never pass a sensor-derived gate (APPROVAL-RECORD.md:1053-1068 records why the
floor-plane depth claim is unresolved; this fixture therefore declares **no floor
surface** and its expected answer never depends on floor-plane depth).

The scene, the closed-form derivations and the pre-registered tolerances live in
``truth/MANUAL-VALUES.md`` and are encoded as literals below; :func:`check_numbers`
recomputes each closed-form value and refuses if a literal disagrees, so a wrong
literal cannot pass as a generated expectation and a generated value cannot drift
away from the hand derivation it claims.

Run ``python generate.py`` to rebuild the fixture in place, or
``python generate.py --check`` to rebuild into a temporary directory and verify
every byte in place. ``python generate.py --agreement`` runs the non-gate
agreement diagnostic (SGBM on the rendered pairs against the declared depth).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from embodied.bench import recorder
from embodied.bench.recorder import Recorder, RunManifest
from embodied.contracts import records as R
from embodied.perception import camera as cam
from embodied.perception.camera import DeclaredSurface

FIXTURE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Authored declarations (every value is a design choice, not a measurement)
# ---------------------------------------------------------------------------

EPISODE_ID = "p03-doorway-v1"
HOST_ID = "p03-fixture-host"
CLOCK_ID = "p03-fixture-monotonic"
NAV_EPOCH = "p03-epoch-1"
CALIBRATION_ID = cam.CALIBRATION_ID
CALIBRATION_VERSION = cam.CALIBRATION_VERSION

WALL_X_ODOM_M = 3.0
FAR_WALL_X_ODOM_M = 6.0
WALL_Y_ODOM_M = (-2.0, 2.0)
# The wall continues below the level anyone flies at. With only forward-looking
# views, a wall that stopped at z=0 would leave the space under the approach
# corridor unobserved, and no motion under the declared 0.40 m envelope could be
# certified anywhere near the start. The wall below the opening is still a
# declared surface, so those rays return the wall instead of nothing. No floor is
# declared: floor-plane metric depth stays unresolved.
WALL_Z_ODOM_M = (-1.0, 2.4)
APERTURE_Y_ODOM_M = (-0.3, 0.9)
APERTURE_Z_ODOM_M = (0.0, 2.0)
FAR_WALL_Y_ODOM_M = (-2.0, 2.0)
# The far surface continues below the band anyone looks through: without it, rays
# that pass the opening below the far wall's bottom edge return nothing, so the
# free space under the approach corridor would be unobserved. That is an artifact
# of a wall that stops in mid-air, not a property of the room, and it would make
# every trajectory near the doorway unsupported. No floor is declared: floor-plane
# metric depth stays unresolved (APPROVAL-RECORD.md:1053-1068).
# Tall and deep enough to catch every ray that passes the opening: a ray through
# the opening's top edge reaches z ~ 3.0 at the far wall, and one through its
# bottom edge reaches z ~ -1.0. A shorter far surface would leave the space above
# and below the corridor unobserved, which no envelope can be certified against.
FAR_WALL_Z_ODOM_M = (-1.0, 3.5)

IMAGE_SIZE = (640, 480)
BORDER_MARGIN_PX = 8
SIGMA_DISPARITY_PX = 1.0
SIGMA_PIXEL_PX = 1.0
APERTURE_EDGE_SHRINK_M = 0.02
THROUGH_MARGIN_M = 0.5

BODY_RADIUS_M = 0.30
ERROR_ALLOWANCE_M = 0.10
INFLATION_M = BODY_RADIUS_M + ERROR_ALLOWANCE_M

VOXEL_M = 0.10
SUBMAP_BOUNDS_ODOM_M = {"x": (-5.0, 7.0), "y": (-4.0, 4.0), "z": (-0.5, 3.0)}
SURFACE_BAND_M = 0.10
LOG_ODDS_HIT = 0.7
LOG_ODDS_PASS = -0.4
LOG_ODDS_CLAMP = 4.0
FREE_THRESHOLD = 1.4
OCCUPIED_THRESHOLD = 1.4
MIN_CLEARING_RAYS = 3
FRESHNESS_S = 5.0
DYNAMIC_CLASS = "pedestrian"
DYNAMIC_SPEED_MPS = 1.2
DYNAMIC_REACH_S = 1.5

# The instant at which P03 evaluates the fixture, and the declared pose bound.
EVALUATION_NS = 1_600_000_000
NOMINAL_POSE_SIGMA_M = 0.02
POSE_SIGMA_LIMIT_M = ERROR_ALLOWANCE_M / 3.0

V_MAX_MPS = 0.5
A_MAX_MPS2 = 1.0
JERK_MAX_MPS3 = 4.0
POSE_VALIDITY_S = 5.0
START_STATE_TOLERANCE_M = 0.20
SETPOINT_PREFIX_HORIZON_S = 2.0
SETPOINT_PERIOD_S = 0.1
AMBIGUITY_MARGIN = 0.15

# Views: body pose in odom (identity rotation, camera optical axis +x).
VIEWS = (
    {
        "view_index": 0,
        "observation_id": "obs-1",
        "role": "mission view: the selection the goal cites is made here",
        "record_id": "obs-1",
        "pair_id": "pair-1",
        "sequence": 0,
        "body_position_odom_m": (0.0, 0.0, 1.0),
        "capture_ns": 1_000_000_000,
        "receipt_ns": 1_005_000_000,
        "sim_time_s": 1.0,
        "left_file": "obs-1-left.ppm",
        "right_file": "obs-1-right.ppm",
    },
    {
        "view_index": 1,
        "observation_id": "obs-2",
        "record_id": "obs-2",
        "pair_id": "pair-2",
        "sequence": 1,
        # Two metres apart and well behind the start. Free space is only what some
        # view observed: a forward-looking camera's knowledge is a cone, and at the
        # cone's apex - the aircraft itself - the cone is narrower than the declared
        # 0.40 m envelope, so a view taken from where the aircraft stands cannot
        # certify motion there. Views 4 m and 2 m back see the start's surroundings
        # with room to spare, and they triangulate the aperture from two angles.
        "body_position_odom_m": (-4.0, 0.0, 1.0),
        "capture_ns": 1_500_000_000,
        "receipt_ns": 1_505_000_000,
        "sim_time_s": 1.5,
        "left_file": "obs-2-left.ppm",
        "right_file": "obs-2-right.ppm",
    },
)

# Authored selections. Boxes are the closed-form projection of an odom rectangle
# on the wall plane, rounded outward to whole pixels (see MANUAL-VALUES.md).
SELECTION_APERTURE = "sel-aperture"
SELECTION_FAR_POINT = "sel-far-point"
SELECTION_OPEN_BAND = "sel-open-band"
SELECTION_APERTURE_V2 = "sel-aperture-v2"

APERTURE_SELECTION_RECTANGLE_ODOM_M = {
    "plane_x_odom_m": WALL_X_ODOM_M,
    "y_odom_m": (-0.6, 1.2),
    "z_odom_m": (-0.05, 2.3),
}
APERTURE_BOX_PX = (103, 5, 443, 447)
APERTURE_BOX_V2_PX = (228, 140, 372, 328)
FAR_POINT_PX = (297, 245)
# A selection inside the masked border. With the far surface catching every ray the
# image can see, an invalid sample now means a masked one: this box is the smallest
# honest way to exercise "no valid depth in the selection grounds nothing".
OPEN_BAND_BOX_PX = (0, 0, 5, 5)

# Hand-derived expectations (each derivation is in truth/MANUAL-VALUES.md).
EXPECTED_APERTURE_PLANE_X_M = 3.0
EXPECTED_APERTURE_CORNERS_ODOM_M = (
    (WALL_X_ODOM_M, -0.275959, 0.058735),
    (WALL_X_ODOM_M, 0.876269, 0.058735),
    (WALL_X_ODOM_M, 0.876269, 1.977396),
    (WALL_X_ODOM_M, -0.275959, 1.977396),
)
EXPECTED_APERTURE_WIDTH_M = 1.152228
EXPECTED_APERTURE_CORRIDOR_M = 0.352228
EXPECTED_FAR_POINT_ODOM_M = (6.0, 0.296907451, 0.996324467)
EXPECTED_FAR_POINT_DEPTH_M = 5.95
EXPECTED_WALL_DEPTH_M = 2.95
EXPECTED_SIGMA_Z_WALL_M = 0.157012
EXPECTED_SIGMA_Z_FAR_M = 0.638739
EXPECTED_SIGMA_LATERAL_WALL_M = 0.005322
EXPECTED_SIGMA_LATERAL_FAR_M = 0.010735
# A grounded target's reported envelope adds the capture pose's declared bound
# linearly (section 7.3: add separately validated bounds; never multiply marginal
# confidences). The depth envelope above is the measurement term only.
POSE_BOUND_ADDED_M = NOMINAL_POSE_SIGMA_M
EXPECTED_TARGET_SIGMA_WALL_M = 0.177012
EXPECTED_TARGET_SIGMA_FAR_M = 0.658739
EXPECTED_TARGET_SIGMA_LATERAL_WALL_M = 0.025322
EXPECTED_TARGET_SIGMA_LATERAL_FAR_M = 0.030735
EXPECTED_NARROW_WIDTH_M = 0.651918
EXPECTED_NARROW_CORRIDOR_M = -0.148082
EXPECTED_PARTIAL_PANEL_Y_ODOM_M = (-0.3, 0.7)
EXPECTED_PARTIAL_WIDTH_M = 0.955298
EXPECTED_PARTIAL_CORRIDOR_M = 0.155298

TOLERANCES = {
    "aperture_edge_abs_m": 0.01,
    "point_abs_m": 0.02,
    "uncertainty_abs_m": 1e-6,
    "setpoint_position_abs_m": 1e-9,
    "setpoint_speed_abs_mps": 1e-9,
    "setpoint_accel_abs_mps2": 1e-9,
    "settle_position_abs_m": 0.10,
    "settle_speed_abs_mps": 0.02,
    "probe_cell_rule": "cell index = floor((value - low_bound) / resolution) in float64",
    "uncertainty_note": (
        "an absolute tolerance, because the authored uncertainty literals are rounded to six "
        "decimal places and a relative bound on a small term would be tighter than that rounding"
    ),
    "note": (
        "pre-registered before any comparison ran; a failed tolerance is a finding, "
        "never a value to adjust"
    ),
}

PROBE_REGIONS = (
    {
        "probe_id": "free-before-wall",
        "expect": "free",
        "x_m": (2.35, 2.55),
        "y_m": (-1.15, -0.95),
        "z_m": (0.95, 1.15),
        "derivation": (
            "lines of sight from both views cross the whole region before the wall plane; every "
            "cell center projects to a valid pixel whose declared depth exceeds the cell's ray "
            "distance by more than a voxel"
        ),
    },
    {
        "probe_id": "occupied-wall-band",
        "expect": "at_least_one_occupied",
        "x_m": (2.90, 3.10),
        "y_m": (-1.15, -0.95),
        "z_m": (0.95, 1.15),
        "derivation": (
            "the wall plane x=3.0 lies inside this band; rays from both views end in the cell "
            "holding the hit point (+0.7 log-odds each, clamp +4.0, occupied at >= +1.4)"
        ),
    },
    {
        "probe_id": "free-through-aperture",
        "expect": "free",
        "x_m": (3.40, 4.60),
        "y_m": (0.25, 0.45),
        "z_m": (0.95, 1.15),
        "derivation": (
            "rays through the aperture end on the far wall x=6.0, so this region is strictly "
            "before the hit on valid rays: four passes at -0.4 = -1.6, and the ray count is far "
            "above the three required"
        ),
    },
    {
        "probe_id": "free-just-before-far-wall",
        "expect": "free",
        "x_m": (5.35, 5.55),
        "y_m": (0.25, 0.45),
        "z_m": (0.95, 1.15),
        "derivation": "as free-through-aperture, one half-voxel before the hit on the same rays",
    },
    {
        "probe_id": "occupied-far-band",
        "expect": "at_least_one_occupied",
        "x_m": (5.90, 6.10),
        "y_m": (0.25, 0.45),
        "z_m": (0.95, 1.15),
        "derivation": "the far wall plane x=6.0 lies inside this band, as occupied-wall-band",
    },
    {
        "probe_id": "unknown-beyond-wall",
        "expect": "unknown",
        "x_m": (4.35, 4.75),
        "y_m": (1.75, 1.95),
        "z_m": (0.95, 1.15),
        "derivation": (
            "the wall spans y in [-2,2], so a ray toward this region meets the wall and stops; the "
            "rays that do pass the aperture reach no higher than y = 1.5 at x = 4.65 from either "
            "view, so the region receives no evidence at all: unknown, never free"
        ),
    },
    {
        "probe_id": "unknown-outside-wall-span",
        "expect": "unknown",
        "x_m": (1.35, 1.75),
        "y_m": (-3.65, -3.45),
        "z_m": (0.95, 1.15),
        "derivation": (
            "rays toward this region leave the wall's y span and the far wall's y span, so the "
            "declared depth is a no-return and nothing is cleared there"
        ),
    },
)

# Pose-degradation injections (coordinator scope addition; see MANUAL-VALUES.md).
POSE_DEGRADATION = {
    "stale": {
        "capture_age_s": 6.0,
        "threshold_s": POSE_VALIDITY_S,
        "provenance": (
            "declared pose validity 5.0 s (P03 plan section 4); the injected 6.0 s is validity "
            "plus 1.0 s of clock skew"
        ),
        "expected": ["stale_pose", "stale_state"],
    },
    "noise": {
        "position_offset_odom_m": (0.12, 0.0, 0.0),
        "threshold_m": ERROR_ALLOWANCE_M,
        "provenance": (
            "declared error allowance 0.10 m inside the 0.40 m inflation (P03 plan section 4); "
            "the injected 0.12 m is 20 percent beyond it"
        ),
        "expected": ["pose_error_exceeds_allowance"],
    },
    "jump": {
        "position_offset_odom_m": (0.50, 0.0, 0.0),
        "threshold_m": START_STATE_TOLERANCE_M,
        "provenance": (
            "authored start-state tolerance 0.20 m; the injected 0.50 m is 2.5x it, the size of "
            "an estimator realignment that keeps the same nav_epoch"
        ),
        "expected": ["start_state_mismatch"],
    },
    "epoch": {
        "nav_epoch": "p03-epoch-2",
        "provenance": "CONTRACTS.md:7 - a nav_epoch reset invalidates every prior control reference",
        "expected": ["frame_epoch_mismatch"],
    },
}
# ---------------------------------------------------------------------------
# Scene, calibration and camera geometry
# ---------------------------------------------------------------------------


def calibration() -> R.Calibration:
    """The pinned rig. Its time-offset pair stays absent: unknown, never zero."""
    return cam.build_calibration()


def make_surfaces(
    aperture_y: tuple[float, float] = APERTURE_Y_ODOM_M,
    aperture_z: tuple[float, float] = APERTURE_Z_ODOM_M,
    panel: DeclaredSurface | None = None,
) -> list[DeclaredSurface]:
    """The wall as three patches around the aperture, plus the far wall."""
    return [
        DeclaredSurface(
            "wall_left",
            (WALL_X_ODOM_M, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            {"y": (WALL_Y_ODOM_M[0], aperture_y[0]), "z": WALL_Z_ODOM_M},
        ),
        DeclaredSurface(
            "wall_right",
            (WALL_X_ODOM_M, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            {"y": (aperture_y[1], WALL_Y_ODOM_M[1]), "z": WALL_Z_ODOM_M},
        ),
        DeclaredSurface(
            "wall_sill",
            (WALL_X_ODOM_M, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            {"y": aperture_y, "z": (WALL_Z_ODOM_M[0], aperture_z[0])},
        ),
        DeclaredSurface(
            "wall_header",
            (WALL_X_ODOM_M, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            {"y": aperture_y, "z": (aperture_z[1], WALL_Z_ODOM_M[1])},
        ),
        DeclaredSurface(
            "far_wall",
            (FAR_WALL_X_ODOM_M, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            {"y": FAR_WALL_Y_ODOM_M, "z": FAR_WALL_Z_ODOM_M},
        ),
    ] + ([panel] if panel is not None else [])


def panel_surface(name: str, y_m: tuple[float, float], z_m: tuple[float, float]) -> DeclaredSurface:
    """A closed door leaf filling part of the opening, coplanar with the wall."""
    return DeclaredSurface(name, (WALL_X_ODOM_M, 0.0, 0.0), (1.0, 0.0, 0.0), {"y": y_m, "z": z_m})


def panel_for(variant: str) -> DeclaredSurface | None:
    """The authored panel for a variant name, or None for an open doorway."""
    if variant == "nominal":
        return None
    if variant == "blocked":
        return panel_surface("closed_door_leaf", APERTURE_Y_ODOM_M, APERTURE_Z_ODOM_M)
    if variant == "partial_block":
        return panel_surface("half_door_leaf", (0.7, 0.9), APERTURE_Z_ODOM_M)
    raise ValueError(f"no authored panel for variant {variant!r}")


def variant_surfaces(variant: str) -> list[DeclaredSurface]:
    """Declared surfaces for one authored variant of the doorway scene."""
    if variant == "narrow":
        return make_surfaces(aperture_y=(-0.15, 0.55))
    return make_surfaces(panel=panel_for(variant))


def camera_center_odom(view_index: int) -> tuple[float, float, float]:
    """The camera optical centre: authored body position plus the rig's camera offset."""
    body = VIEWS[view_index]["body_position_odom_m"]
    offset = calibration().T_body_camera_left.translation_m
    return tuple(b + o for b, o in zip(body, offset))


_DEPTH_CACHE: dict[tuple, np.ndarray] = {}


def _surfaces_key(surfaces: list[DeclaredSurface]) -> tuple:
    return tuple(
        (
            surface.name,
            surface.plane_point_world_m,
            surface.plane_normal_world,
            tuple(sorted(surface.axis_bounds_world_m.items())),
        )
        for surface in surfaces
    )


def declared_depth(view_index: int, surfaces: list[DeclaredSurface]) -> np.ndarray:
    """`camera.declared_depth_map` for one view: t with the ray direction's x component 1.

    Memoised per (view, surface set). The returned array is shared: read it, never
    write to it.
    """
    key = (view_index, _surfaces_key(surfaces))
    cached = _DEPTH_CACHE.get(key)
    if cached is not None:
        return cached
    depth, _attribution = cam.declared_depth_map(
        surfaces,
        camera_center_odom(view_index),
        np.eye(3),
        cam.FOCAL_LENGTH_PX,
        cam.PRINCIPAL_POINT_PX,
        IMAGE_SIZE,
    )
    _DEPTH_CACHE[key] = depth
    return depth


def declared_depth_product(view_index: int, surfaces: list[DeclaredSurface]) -> cam.DepthProduct:
    """A declared DepthProduct: valid where a surface was hit outside the border margin."""
    depth = declared_depth(view_index, surfaces)
    height, width = depth.shape
    reasons = np.full((height, width), cam.REASON_NO_RETURN, dtype=np.uint8)
    reasons[np.isfinite(depth)] = cam.REASON_VALID
    reasons[:BORDER_MARGIN_PX, :] = cam.REASON_BORDER
    reasons[-BORDER_MARGIN_PX:, :] = cam.REASON_BORDER
    reasons[:, :BORDER_MARGIN_PX] = cam.REASON_BORDER
    reasons[:, -BORDER_MARGIN_PX:] = cam.REASON_BORDER
    valid = reasons == cam.REASON_VALID
    focal = cam.FOCAL_LENGTH_PX
    baseline = calibration().baseline_m
    uncertainty = np.full((height, width), np.nan, dtype=np.float64)
    uncertainty[valid] = (depth[valid] ** 2) * SIGMA_DISPARITY_PX / (focal * baseline)
    view = VIEWS[view_index]
    return cam.DepthProduct(
        calibration_id=CALIBRATION_ID,
        calibration_version=CALIBRATION_VERSION,
        pair_id=view["pair_id"],
        capture_stamp=R.ClockStamp(HOST_ID, CLOCK_ID, view["capture_ns"]),
        receipt_stamp=R.ClockStamp(HOST_ID, CLOCK_ID, view["receipt_ns"]),
        sim_time_s=view["sim_time_s"],
        pose_provenance=cam.static_start_provenance(
            f"tests/fixtures/navigation/doorway/truth/scene.json#{view['observation_id']}"
        ),
        frame="depth along the left rectified optical axis, fixture pixel convention",
        disparity_px=np.where(valid, focal * baseline / depth, np.nan),
        depth_m=depth,
        valid=valid,
        reasons=reasons,
        uncertainty_m=uncertainty,
    )


def project(view_index: int, point_odom_m: tuple[float, float, float]) -> tuple[float, float]:
    """Closed-form projection into the rectified left image (column, row)."""
    centre = np.asarray(camera_center_odom(view_index), dtype=np.float64)
    relative = np.asarray(point_odom_m, dtype=np.float64) - centre
    if relative[0] <= 0.0:
        raise ValueError("the authored point is behind the authored camera")
    focal = cam.FOCAL_LENGTH_PX
    u = cam.PRINCIPAL_POINT_PX[0] - focal * relative[1] / relative[0]
    v = cam.PRINCIPAL_POINT_PX[1] - focal * relative[2] / relative[0]
    return (float(u), float(v))


def unproject(view_index: int, u: float, v: float, depth_m: float) -> tuple[float, float, float]:
    """Closed-form inverse of :func:`project` at the declared optical-axis depth."""
    centre = np.asarray(camera_center_odom(view_index), dtype=np.float64)
    focal = cam.FOCAL_LENGTH_PX
    dy = -(u - cam.PRINCIPAL_POINT_PX[0]) / focal
    dz = -(v - cam.PRINCIPAL_POINT_PX[1]) / focal
    return tuple(float(value) for value in centre + depth_m * np.array([1.0, dy, dz]))


def wall_plane_crossing(view_index: int, u: float, v: float) -> tuple[float, float]:
    """Where a pixel's ray meets the wall plane, in odom (y, z)."""
    centre = camera_center_odom(view_index)
    along = WALL_X_ODOM_M - centre[0]
    focal = cam.FOCAL_LENGTH_PX
    dy = -(u - cam.PRINCIPAL_POINT_PX[0]) / focal
    dz = -(v - cam.PRINCIPAL_POINT_PX[1]) / focal
    return (centre[1] + along * dy, centre[2] + along * dz)


def aperture_evidence(
    view_index: int, box_px: tuple[int, int, int, int], surfaces: list[DeclaredSurface]
) -> dict:
    """Closed-form free-opening evidence inside a selection box.

    A pixel is *through* when its declared depth is finite and beyond the wall
    plane by more than :data:`THROUGH_MARGIN_M`: the ray passed the wall and
    ended on the far wall. Intersecting those rays with the wall plane bounds the
    observed opening; the opening polygon is that bound shrunk by the authored
    edge uncertainty, so it is a lower bound on the usable opening.
    """
    depth = declared_depth(view_index, surfaces)
    u0, v0, u1, v1 = box_px
    wall_depth = WALL_X_ODOM_M - camera_center_odom(view_index)[0]
    sub = depth[v0 : v1 + 1, u0 : u1 + 1]
    through = np.isfinite(sub) & (sub > wall_depth + THROUGH_MARGIN_M)
    wall = np.isfinite(sub) & (np.abs(sub - wall_depth) <= SURFACE_BAND_M)
    result = {
        "box_px": list(box_px),
        "wall_depth_m": float(wall_depth),
        "through_pixels": int(through.sum()),
        "wall_pixels": int(wall.sum()),
        "no_return_pixels": int(sub.size - int(through.sum()) - int(wall.sum())),
        "through_pixel_bbox_px": None,
        "opening_plane_y_m": None,
        "opening_plane_z_m": None,
        "opening_width_m": None,
        "corridor_width_m": None,
    }
    if through.sum() == 0:
        return result
    rows, columns = np.nonzero(through)
    columns = columns + u0
    rows = rows + v0
    crossings = np.array(
        [wall_plane_crossing(view_index, float(u), float(v)) for u, v in zip(columns, rows)]
    )
    y_lo, y_hi = float(crossings[:, 0].min()), float(crossings[:, 0].max())
    z_lo, z_hi = float(crossings[:, 1].min()), float(crossings[:, 1].max())
    shrink = APERTURE_EDGE_SHRINK_M
    result["through_pixel_bbox_px"] = [
        int(columns.min()),
        int(columns.max()),
        int(rows.min()),
        int(rows.max()),
    ]
    result["opening_plane_y_m"] = [y_lo + shrink, y_hi - shrink]
    result["opening_plane_z_m"] = [z_lo + shrink, z_hi - shrink]
    result["opening_width_m"] = float((y_hi - shrink) - (y_lo + shrink))
    result["corridor_width_m"] = float(result["opening_width_m"] - 2.0 * INFLATION_M)
    return result


def probe_region_cells(region: dict) -> tuple[tuple[int, int, int], ...]:
    """Every cell index whose center lies in one authored probe region."""
    spans = []
    for axis, key in ((0, "x_m"), (1, "y_m"), (2, "z_m")):
        low, high = region[key]
        lower_bound = SUBMAP_BOUNDS_ODOM_M[("x", "y", "z")[axis]][0]
        first = int(np.floor((low - lower_bound) / VOXEL_M))
        last = int(np.floor((high - lower_bound) / VOXEL_M))
        spans.append(range(first, last + 1))
    return tuple(
        (x, y, z) for x in spans[0] for y in spans[1] for z in spans[2]
    )


def cell_center(index: tuple[int, int, int]) -> tuple[float, float, float]:
    """Center of one cell index, by the same declared rule."""
    return tuple(
        SUBMAP_BOUNDS_ODOM_M[axis][0] + (index[position] + 0.5) * VOXEL_M
        for position, axis in enumerate(("x", "y", "z"))
    )

# Rendering (real projective renders of the same authored scene)

_TEXTURE_TABLE = np.random.Generator(np.random.PCG64(20260926)).integers(0, 1 << 16, size=(1 << 16, 3))


def _texture(hit_points: np.ndarray, surface_index: int) -> np.ndarray:
    """World-anchored texture: the value depends on the quantized hit point."""
    quantized = np.floor(hit_points * 20.0).astype(np.int64)
    mixed = (
        quantized[..., 0] * 73856093
        + quantized[..., 1] * 19349663
        + quantized[..., 2] * 83492791
        + surface_index * 2654435761
    )
    return _TEXTURE_TABLE[np.abs(mixed) % _TEXTURE_TABLE.shape[0]]


def render_camera(
    view_index: int, camera_offset_y_m: float, surfaces: list[DeclaredSurface]
) -> np.ndarray:
    """One projective render: per-pixel declared surface, textured at the hit point."""
    height, width = IMAGE_SIZE[1], IMAGE_SIZE[0]
    centre = np.asarray(camera_center_odom(view_index), dtype=np.float64).copy()
    centre[1] += camera_offset_y_m
    columns = np.arange(width, dtype=np.float64)[None, :].repeat(height, axis=0)
    rows = np.arange(height, dtype=np.float64)[:, None].repeat(width, axis=1)
    focal = cam.FOCAL_LENGTH_PX
    directions = np.stack(
        [
            np.ones_like(columns),
            -(columns - cam.PRINCIPAL_POINT_PX[0]) / focal,
            -(rows - cam.PRINCIPAL_POINT_PX[1]) / focal,
        ],
        axis=2,
    )
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for index, surface in enumerate(surfaces):
        normal = np.asarray(surface.plane_normal_world, dtype=np.float64)
        point = np.asarray(surface.plane_point_world_m, dtype=np.float64)
        numerator = float(normal @ (point - centre))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = numerator / (directions @ normal)
            hit = t > 0.0
            hit_points = centre[None, None, :] + t[..., None] * directions
        for axis, (low, high) in surface.axis_bounds_world_m.items():
            axis_index = {"x": 0, "y": 1, "z": 2}[axis]
            hit &= (hit_points[..., axis_index] >= low) & (hit_points[..., axis_index] <= high)
        image[hit] = _texture(hit_points, index)[hit]
    return image


def render_ppm(image: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 image as a binary P6 PPM."""
    height, width = image.shape[:2]
    return b"P6\n%d %d\n255\n" % (width, height) + image.astype(np.uint8).tobytes()


def render_pair(view_index: int, surfaces: list[DeclaredSurface]) -> tuple[bytes, bytes]:
    """Left and right renders for one view, with the pinned rig's baseline."""
    baseline = abs(calibration().T_camera_left_camera_right.translation_m[1])
    return (
        render_ppm(render_camera(view_index, 0.0, surfaces)),
        render_ppm(render_camera(view_index, -baseline, surfaces)),
    )


# Capture-time pose records, current state and the injected degradations


def capture_pose_record(view_index: int) -> R.PoseEstimate:
    """The capture-time pose bound to one observation (declared scene truth)."""
    view = VIEWS[view_index]
    return R.PoseEstimate(
        parent_frame="odom",
        child_frame="body",
        stamp=R.ClockStamp(HOST_ID, CLOCK_ID, view["capture_ns"]),
        position_m=view["body_position_odom_m"],
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance=(NOMINAL_POSE_SIGMA_M**2,) * 3,
        nav_epoch=NAV_EPOCH,
        source_ids=("declared:fixture-scene",),
        valid=True,
    )


def navigation_state(
    stamp_ns: int = EVALUATION_NS,
    *,
    pose: R.PoseEstimate | None = None,
    nav_epoch: str = NAV_EPOCH,
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    state_sequence: int = 1,
) -> R.NavigationState:
    """The estimator state at the evaluation instant, or carrying an injected pose."""
    if pose is None:
        pose = R.PoseEstimate(
            parent_frame="odom",
            child_frame="body",
            stamp=R.ClockStamp(HOST_ID, CLOCK_ID, stamp_ns),
            position_m=(0.0, 0.0, 1.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            covariance=(NOMINAL_POSE_SIGMA_M**2,) * 3,
            nav_epoch=nav_epoch,
            source_ids=("declared:fixture-scene",),
            valid=True,
        )
    return R.NavigationState(
        state_sequence=state_sequence,
        pose=pose,
        velocity_mps=velocity_mps,
        covariance=(NOMINAL_POSE_SIGMA_M**2,) * 3,
        nav_epoch=nav_epoch,
        visual_source_ids=("camera_left", "camera_right"),
        imu_source_ids=("imu",),
        status="nominal",
        controller_alignment_id="fixture-guided-1",
    )


def degraded_pose(kind: str) -> R.PoseEstimate:
    """One injected pose degradation, at the magnitude the fixture declares."""
    nominal = navigation_state().pose
    if kind == "stale":
        age_ns = int(POSE_DEGRADATION["stale"]["capture_age_s"] * 1e9)
        return R.PoseEstimate(
            parent_frame="odom",
            child_frame="body",
            stamp=R.ClockStamp(HOST_ID, CLOCK_ID, VIEWS[0]["capture_ns"] + age_ns),
            position_m=nominal.position_m,
            quaternion_wxyz=nominal.quaternion_wxyz,
            covariance=nominal.covariance,
            nav_epoch=nominal.nav_epoch,
            source_ids=nominal.source_ids,
            valid=True,
        )
    if kind == "noise":
        offset = POSE_DEGRADATION["noise"]["position_offset_odom_m"]
        sigma = ERROR_ALLOWANCE_M / 2.0
        return R.PoseEstimate(
            parent_frame="odom",
            child_frame="body",
            stamp=nominal.stamp,
            position_m=tuple(n + o for n, o in zip(nominal.position_m, offset)),
            quaternion_wxyz=nominal.quaternion_wxyz,
            covariance=(sigma**2,) * 3,
            nav_epoch=nominal.nav_epoch,
            source_ids=nominal.source_ids,
            valid=True,
        )
    if kind == "jump":
        offset = POSE_DEGRADATION["jump"]["position_offset_odom_m"]
        return R.PoseEstimate(
            parent_frame="odom",
            child_frame="body",
            stamp=nominal.stamp,
            position_m=tuple(n + o for n, o in zip(nominal.position_m, offset)),
            quaternion_wxyz=nominal.quaternion_wxyz,
            covariance=nominal.covariance,
            nav_epoch=nominal.nav_epoch,
            source_ids=nominal.source_ids,
            valid=True,
        )
    if kind == "epoch":
        return R.PoseEstimate(
            parent_frame="odom",
            child_frame="body",
            stamp=nominal.stamp,
            position_m=nominal.position_m,
            quaternion_wxyz=nominal.quaternion_wxyz,
            covariance=nominal.covariance,
            nav_epoch=POSE_DEGRADATION["epoch"]["nav_epoch"],
            source_ids=nominal.source_ids,

            valid=True,
        )
    raise ValueError(f"no authored degradation {kind!r}")

# The declared stereo settings the agreement diagnostic runs with. Copied from the
# pinned configuration's shape (configs/first_indoor.yaml) with the fixture's own
# depth window; the diagnostic is non-gate and its statistics are recorded, not scored.
SGBM_SETTINGS = {
    "min_disparity": 0,
    "num_disparities": 128,
    "block_size": 5,
    "uniqueness_ratio": 10,
    "speckle_window_size": 200,
    "speckle_range": 2,
    "texture_threshold": 10,
    "lr_tolerance_px": 1.0,
    "border_px": BORDER_MARGIN_PX,
    "depth_range_m": [0.5, 8.0],
    "disparity_quantization_sigma_px": SIGMA_DISPARITY_PX,
}


def scene_document() -> dict:
    """truth/scene.json: the authored scene, poses, stamps and declared parameters."""
    return {
        "fixture_revision": "p03-doorway-fixture-1",
        "provenance": "DIAGNOSTIC (declared scene and declared depth; never sensor-derived)",
        "frame": {
            "name": "odom",
            "handedness": "right",
            "gravity_aligned": True,
            "axes": "x through the doorway, y left, z up; identity quaternions at both views",
        },
        "units": "metres, seconds, nanoseconds",
        "nav_epoch": NAV_EPOCH,
        "evaluation_ns": EVALUATION_NS,
        "surfaces": [
            {
                "name": surface.name,
                "plane_point_odom_m": list(surface.plane_point_world_m),
                "plane_normal_odom": list(surface.plane_normal_world),
                "axis_bounds_odom_m": {k: list(v) for k, v in surface.axis_bounds_world_m.items()},
            }
            for surface in make_surfaces()
        ],
        "aperture": {
            "wall_x_odom_m": WALL_X_ODOM_M,
            "y_odom_m": list(APERTURE_Y_ODOM_M),
            "z_odom_m": list(APERTURE_Z_ODOM_M),
            "width_m": APERTURE_Y_ODOM_M[1] - APERTURE_Y_ODOM_M[0],
            "crossing_direction": "+x (the requested side and the current approach agree)",
        },
        "floor": {
            "declared": False,
            "reason": (
                "floor-plane metric depth is unresolved (APPROVAL-RECORD.md:1053-1068); no floor "
                "surface is declared and no expected value depends on one"
            ),
        },
        "calibration": {
            "calibration_id": CALIBRATION_ID,
            "version": CALIBRATION_VERSION,
            "focal_length_px": cam.FOCAL_LENGTH_PX,
            "principal_point_px": list(cam.PRINCIPAL_POINT_PX),
            "baseline_m": calibration().baseline_m,
            "T_body_camera_left_translation_m": list(
                calibration().T_body_camera_left.translation_m
            ),
            "T_camera_left_camera_right_translation_m": list(
                calibration().T_camera_left_camera_right.translation_m
            ),
            "pixel_convention": calibration().pixel_convention,
            "depth_convention": calibration().depth_convention,
        },
        "views": [
            {
                "observation_id": view["observation_id"],
                "record_id": view["record_id"],
                "pair_id": view["pair_id"],
                "body_position_odom_m": list(view["body_position_odom_m"]),
                "body_quaternion_odom_wxyz": [1.0, 0.0, 0.0, 0.0],
                "camera_center_odom_m": list(camera_center_odom(view["view_index"])),
                "capture_ns": view["capture_ns"],
                "sim_time_s": view["sim_time_s"],
                "left_payload": f"payloads/{view['left_file']}",
                "right_payload": f"payloads/{view['right_file']}",
            }
            for view in VIEWS
        ],
        "depth": {
            "source": "embodied.perception.camera.declared_depth_map over the declared surfaces",
            "provenance": "DIAGNOSTIC",
            "wall_depth_m": EXPECTED_WALL_DEPTH_M,
            "far_wall_depth_m": EXPECTED_FAR_POINT_DEPTH_M,
            "sigma_disparity_px": SIGMA_DISPARITY_PX,
            "sigma_pixel_px": SIGMA_PIXEL_PX,
            "border_margin_px": BORDER_MARGIN_PX,
            "reasons": {name: index for index, name in enumerate(cam.REASON_NAMES)},
            "note": "a masked sample is unknown; invalid depth clears nothing and grounds nothing",
        },
        "vehicle": {
            "body_radius_m": BODY_RADIUS_M,
            "error_allowance_m": ERROR_ALLOWANCE_M,
            "inflation_m": INFLATION_M,
        },
        "limits": {
            "v_max_mps": V_MAX_MPS,
            "a_max_mps2": A_MAX_MPS2,
            "jerk_max_mps3": JERK_MAX_MPS3,
            "pose_validity_s": POSE_VALIDITY_S,
            "start_state_tolerance_m": START_STATE_TOLERANCE_M,
            "pose_sigma_nominal_m": NOMINAL_POSE_SIGMA_M,
            "pose_sigma_limit_m": POSE_SIGMA_LIMIT_M,
            "setpoint_prefix_horizon_s": SETPOINT_PREFIX_HORIZON_S,
            "setpoint_period_s": SETPOINT_PERIOD_S,
            "ambiguity_margin": AMBIGUITY_MARGIN,
        },
        "map": {
            "voxel_m": VOXEL_M,
            "bounds_odom_m": {k: list(v) for k, v in SUBMAP_BOUNDS_ODOM_M.items()},
            "surface_band_m": SURFACE_BAND_M,
            "log_odds_hit": LOG_ODDS_HIT,
            "log_odds_pass": LOG_ODDS_PASS,
            "log_odds_clamp": LOG_ODDS_CLAMP,
            "free_threshold": FREE_THRESHOLD,
            "occupied_threshold": OCCUPIED_THRESHOLD,
            "min_clearing_rays": MIN_CLEARING_RAYS,
            "freshness_s": FRESHNESS_S,
        },
        "dynamic": {"class": DYNAMIC_CLASS, "speed_mps": DYNAMIC_SPEED_MPS, "reach_s": DYNAMIC_REACH_S},
        "selections": {
            "aperture": {
                "selection_id": SELECTION_APERTURE,
                "box_px": list(APERTURE_BOX_PX),
                "encloses_rectangle_odom_m": APERTURE_SELECTION_RECTANGLE_ODOM_M,
            },
            "far_point": {"selection_id": SELECTION_FAR_POINT, "point_px": list(FAR_POINT_PX)},
            "open_band": {"selection_id": SELECTION_OPEN_BAND, "box_px": list(OPEN_BAND_BOX_PX)},
            "aperture_v2": {"selection_id": SELECTION_APERTURE_V2, "box_px": list(APERTURE_BOX_V2_PX)},
        },
    }


def _cell_pixel(view_index: int, cell: tuple[int, int, int]) -> tuple[int, int] | None:
    """The nearest pixel whose ray passes nearest the cell center, or None off-image."""
    u, v = project(view_index, cell_center(cell))
    pixel = (int(round(u)), int(round(v)))
    if not (0 <= pixel[0] < IMAGE_SIZE[0] and 0 <= pixel[1] < IMAGE_SIZE[1]):
        return None
    return pixel


def _derived_probe_class(cell: tuple[int, int, int], surfaces: list[DeclaredSurface]) -> str:
    """Closed-form class of one cell from its representative rays, by the declared rules.

    free     - a valid ray passes the cell strictly before its hit
    occupied - a valid ray's hit point lies in or beside the cell along the look axis
    unknown  - no ray reaches the cell at all, or every ray ends before it
    """
    pass_rays = 0
    hit_rays = 0
    for view_index in range(len(VIEWS)):
        pixel = _cell_pixel(view_index, cell)
        if pixel is None:
            continue
        depth = declared_depth(view_index, surfaces)
        u, v = pixel
        t_hit = float(depth[v, u])
        centre = camera_center_odom(view_index)
        cell_position = cell_center(cell)
        s_cell = cell_position[0] - centre[0]
        if not np.isfinite(t_hit):
            continue
        if s_cell < t_hit - VOXEL_M:
            pass_rays += 1
            continue
        hit_point = unproject(view_index, u, v, t_hit)
        lower = SUBMAP_BOUNDS_ODOM_M["x"][0] + (cell[0] - 1) * VOXEL_M
        upper = SUBMAP_BOUNDS_ODOM_M["x"][0] + (cell[0] + 2) * VOXEL_M
        others = [
            SUBMAP_BOUNDS_ODOM_M[axis][0] + (cell[index] - 1) * VOXEL_M
            <= hit_point[index]
            <= SUBMAP_BOUNDS_ODOM_M[axis][0] + (cell[index] + 2) * VOXEL_M
            for index, axis in enumerate(("x", "y", "z"))
        ]
        if lower <= hit_point[0] <= upper and all(others[1:]):
            hit_rays += 1
    if hit_rays:
        return "occupied"
    if pass_rays:
        return "free"
    return "unknown"


def expected_answer() -> dict:
    """truth/expected-answer.json: the hand-derived expectations for the nominal episode."""
    nominal = variant_surfaces("nominal")
    aperture = aperture_evidence(0, APERTURE_BOX_PX, nominal)
    narrow = aperture_evidence(0, APERTURE_BOX_PX, variant_surfaces("narrow"))
    partial = aperture_evidence(0, APERTURE_BOX_PX, variant_surfaces("partial_block"))
    blocked = aperture_evidence(0, APERTURE_BOX_PX, variant_surfaces("blocked"))
    point_depth = EXPECTED_FAR_POINT_DEPTH_M
    u, v = FAR_POINT_PX
    return {
        "fixture_revision": "p03-doorway-fixture-1",
        "episode_id": EPISODE_ID,
        "nav_epoch": NAV_EPOCH,
        "provenance": "DIAGNOSTIC",
        "grounding": {
            "aperture": {
                "selection_id": SELECTION_APERTURE,
                "observation_id": VIEWS[0]["observation_id"],
                "selection_kind": "box",
                "geometry_kind": "oriented_aperture_rectangle",
                "geometry_layout": (
                    "plane_point_odom_m[3], plane_normal_odom[3], corner_odom_m[4x3] ordered "
                    "(y_lo,z_lo), (y_hi,z_lo), (y_hi,z_hi), (y_lo,z_hi)"
                ),
                "plane_point_odom_m": [WALL_X_ODOM_M, 0.0, 0.0],
                "plane_normal_odom": [1.0, 0.0, 0.0],
                "corners_odom_m": [list(corner) for corner in EXPECTED_APERTURE_CORNERS_ODOM_M],
                "opening_width_m": EXPECTED_APERTURE_WIDTH_M,
                "corridor_width_m_after_inflation": EXPECTED_APERTURE_CORRIDOR_M,
                "through_pixels": aperture["through_pixels"],
                "wall_pixels": aperture["wall_pixels"],
                "no_return_pixels": aperture["no_return_pixels"],
                "uncertainty_m": {
                    "normal_sigma_m": EXPECTED_TARGET_SIGMA_WALL_M,
                    "tangential_sigma_m": EXPECTED_TARGET_SIGMA_LATERAL_WALL_M,
                    "edge_shrink_m": APERTURE_EDGE_SHRINK_M,
                    "depth_envelope_term_m": EXPECTED_SIGMA_Z_WALL_M,
                    "pose_bound_term_m": POSE_BOUND_ADDED_M,
                    "composition": (
                        "the depth envelope propagated through the linear transform plus the "
                        "capture pose's declared 1-sigma bound, added linearly because "
                        "independence between them is not established (section 7.3)"
                    ),
                },
                "not_the_depth_behind": (
                    "the polygon lies on the wall plane x=3.0; the far wall at x=6.0 is the depth "
                    "behind the opening and never stands in for the opening itself"
                ),
                "opening_is_a_lower_bound": (
                    "the observed opening is bounded by where rays that pass it still return a "
                    "surface, so the observed z extent (0.059..1.695) is narrower than the authored "
                    "aperture (0..2.0): a lower bound on the usable opening, never an outer box"
                ),
            },
            "far_point": {
                "selection_id": SELECTION_FAR_POINT,
                "observation_id": VIEWS[0]["observation_id"],
                "selection_kind": "point",
                "point_px": [u, v],
                "declared_depth_m": point_depth,
                "point_odom_m": list(EXPECTED_FAR_POINT_ODOM_M),
                "uncertainty_m": {
                    "normal_sigma_m": EXPECTED_TARGET_SIGMA_FAR_M,
                    "tangential_sigma_m": EXPECTED_TARGET_SIGMA_LATERAL_FAR_M,
                    "depth_envelope_term_m": EXPECTED_SIGMA_Z_FAR_M,
                    "pose_bound_term_m": POSE_BOUND_ADDED_M,
                    "composition": "as the aperture: depth envelope plus the shared pose bound",
                },
            },
            "requirements": [
                "the target cites the selection ids and observation ids it came from",
                "the target carries an anchor id and anchor revision",
                "the target frame is odom",
                "calibration identity travels by observation -> Observation.calibration_id",
                "an observation/depth calibration mismatch is refused, never repaired",
            ],
        },
        "occupancy": {
            "probes": [
                {
                    "probe_id": region["probe_id"],
                    "expect": region["expect"],
                    "derivation": region["derivation"],
                    "cells": [
                        {"index": list(cell), "center_odom_m": list(cell_center(cell))}
                        for cell in probe_region_cells(region)
                    ],
                }
                for region in PROBE_REGIONS
            ],
            "never_free_rule": (
                "a cell is free only above the support threshold with at least the declared ray "
                "count and freshness; every other cell is unknown with a reason and unknown is "
                "never traversable"
            ),
        },
        "traverse": {
            "goal_proposal_id": "goal-traverse-aperture",
            "intent": "traverse",
            "route_exists": True,
            "certified_region": "approach",
            "crossing_certified": False,
            "crossing_refusal_reason": "unsupported_space",
            "crossing_refusal_derivation": (
                "the observed free space is a cone through the aperture, and its boundary "
                "narrows below the declared 0.40 m envelope before the crossing, so no "
                "traverse curve out of the approach can be certified; the specification's own "
                "answer for this case is that unknown space beyond the door prevents traversal "
                "certification while still permitting an approach for another view (10.2)"
            ),
            "crossing_x_m": [2.6, 3.4],
            "inflated_opening_y_m": [
                aperture["opening_plane_y_m"][0] + INFLATION_M,
                aperture["opening_plane_y_m"][1] - INFLATION_M,
            ],
            "terminal_region": "the supported exit region beyond the wall, entered through the opening",
            "blocked_variant": {
                "through_pixels": blocked["through_pixels"],
                "expected": ["no_known_supported_route", "blocked"],
            },
            "narrow_variant": {
                "opening_width_m": EXPECTED_NARROW_WIDTH_M,
                "corridor_width_m_after_inflation": EXPECTED_NARROW_CORRIDOR_M,
                "expected": ["aperture_clearance"],
            },
            "partial_block_variant": {
                "opening_width_m": EXPECTED_PARTIAL_WIDTH_M,
                "corridor_width_m_after_inflation": EXPECTED_PARTIAL_CORRIDOR_M,
                "expected": ["route_through_the_remaining_observed_free_part"],
            },
        },
        "setpoint_prefix": {
            "horizon_s": SETPOINT_PREFIX_HORIZON_S,
            "period_s": SETPOINT_PERIOD_S,
            "samples": int(SETPOINT_PREFIX_HORIZON_S / SETPOINT_PERIOD_S) + 1,
            "frame": "odom",
            "type_mask": "position and velocity enabled, acceleration and heading ignored",
            "properties": [
                "every sample's position lies inside the certified corridor envelope",
                "|velocity| <= v_max and |acceleration| <= a_max at every sample",
                "position, velocity and acceleration are continuous across segment joins",
                "the final sample settles inside the terminal region",
                "every sample cites the certificate and carries the goal's nav_epoch",
            ],
        },
        "refusals": {
            "stale_pose": "capture-time pose older than the declared validity, or an invalid pose",
            "missing_depth": "no depth product was supplied for the observation",
            "unknown_geometry": "no valid depth exists anywhere in the selection",
            "frame_epoch_mismatch": "the target's nav_epoch differs from the current one",
            "calibration_mismatch": "observation and depth product name different calibrations",
            "detector_unavailable": "a phrase selection with no verified detector at the seam",
            "no_known_supported_route": "no route exists in the known free map",
            "unsupported_space": "a swept or braking volume touches unknown or unsupported space",
            "aperture_clearance": "the observed opening is smaller than the inflated body envelope",
            "computation_limit": "the planner hit its declared computation bound",
            "start_state_mismatch": "a replan's start state moved beyond the declared tolerance",
            "loss_of_supported_control": "no supported backup or stop remains",
            "pose_error_exceeds_allowance": "the state's declared pose error exceeds the error allowance",
        },
        "tolerances": TOLERANCES,
    }


def variants_document() -> dict:
    """truth/variants.json: the authored variants the tests apply to the same scene."""
    return {
        "fixture_revision": "p03-doorway-fixture-1",
        "provenance": "DIAGNOSTIC",
        "nominal": {
            "surfaces": "wall with the authored aperture, far wall, no panel",
            "aperture_y_odom_m": list(APERTURE_Y_ODOM_M),
        },
        "blocked": {
            "panel": {
                "name": "closed_door_leaf",
                "plane_point_odom_m": [WALL_X_ODOM_M, 0.0, 0.0],
                "plane_normal_odom": [1.0, 0.0, 0.0],
                "axis_bounds_odom_m": {
                    "y": list(APERTURE_Y_ODOM_M),
                    "z": list(APERTURE_Z_ODOM_M),
                },
            },
            "expected": ["no_known_supported_route", "blocked"],
        },
        "partial_block": {
            "panel": {
                "name": "half_door_leaf",
                "plane_point_odom_m": [WALL_X_ODOM_M, 0.0, 0.0],
                "plane_normal_odom": [1.0, 0.0, 0.0],
                "axis_bounds_odom_m": {
                    "y": list(EXPECTED_PARTIAL_PANEL_Y_ODOM_M),
                    "z": list(APERTURE_Z_ODOM_M),
                },
            },
            "expected": ["replanned_route_through_the_remaining_observed_free_part"],
        },
        "narrow": {
            "aperture_y_odom_m": [-0.15, 0.55],
            "opening_width_m": EXPECTED_NARROW_WIDTH_M,
            "expected": ["aperture_clearance"],
        },
        "invalid_depth": {
            "selection_id": SELECTION_OPEN_BAND,
            "box_px": list(OPEN_BAND_BOX_PX),
            "expected": ["unknown_geometry"],
        },
        "unknown": {
            "region_odom_m": {"x": [4.35, 4.75], "y": [1.45, 1.65], "z": [0.95, 1.15]},
            "expected": ["unsupported_space"],
        },
        "pose_degradation": POSE_DEGRADATION,
    }


def check_numbers() -> None:
    """Recompute every closed-form value and refuse if an authored literal disagrees."""
    for view_index, y_range, z_range, box, height in (
        (0, APERTURE_SELECTION_RECTANGLE_ODOM_M["y_odom_m"],
         APERTURE_SELECTION_RECTANGLE_ODOM_M["z_odom_m"], APERTURE_BOX_PX, IMAGE_SIZE[1]),
        (1, APERTURE_SELECTION_RECTANGLE_ODOM_M["y_odom_m"],
         APERTURE_SELECTION_RECTANGLE_ODOM_M["z_odom_m"], APERTURE_BOX_V2_PX, IMAGE_SIZE[1]),
    ):
        top = project(view_index, (WALL_X_ODOM_M, y_range[1], z_range[1]))
        bottom = project(view_index, (WALL_X_ODOM_M, y_range[0], z_range[0]))
        expected_edges = (
            int(np.floor(top[0])),
            max(0, int(np.floor(min(top[1], bottom[1])))),
            int(np.ceil(max(top[0], bottom[0]))),
            min(height - 1, int(np.ceil(max(top[1], bottom[1])))),
        )
        if expected_edges != box:
            raise AssertionError(
                f"authored box {box} is not the outward-rounded projection {expected_edges}"
            )
    nominal = variant_surfaces("nominal")
    aperture = aperture_evidence(0, APERTURE_BOX_PX, nominal)
    for label, measured, expected in (
        ("opening y_lo", aperture["opening_plane_y_m"][0], EXPECTED_APERTURE_CORNERS_ODOM_M[0][1]),
        ("opening y_hi", aperture["opening_plane_y_m"][1], EXPECTED_APERTURE_CORNERS_ODOM_M[1][1]),
        ("opening z_lo", aperture["opening_plane_z_m"][0], EXPECTED_APERTURE_CORNERS_ODOM_M[0][2]),
        ("opening z_hi", aperture["opening_plane_z_m"][1], EXPECTED_APERTURE_CORNERS_ODOM_M[2][2]),
        ("opening width", aperture["opening_width_m"], EXPECTED_APERTURE_WIDTH_M),
        ("corridor width", aperture["corridor_width_m"], EXPECTED_APERTURE_CORRIDOR_M),
    ):
        if abs(measured - expected) > 1e-6:
            raise AssertionError(f"{label}: literal {expected} != closed form {measured}")
    narrow = aperture_evidence(0, APERTURE_BOX_PX, variant_surfaces("narrow"))
    partial = aperture_evidence(0, APERTURE_BOX_PX, variant_surfaces("partial_block"))
    blocked = aperture_evidence(0, APERTURE_BOX_PX, variant_surfaces("blocked"))
    for label, measured, expected in (
        ("narrow width", narrow["opening_width_m"], EXPECTED_NARROW_WIDTH_M),
        ("narrow corridor", narrow["corridor_width_m"], EXPECTED_NARROW_CORRIDOR_M),
        ("partial width", partial["opening_width_m"], EXPECTED_PARTIAL_WIDTH_M),
        ("partial corridor", partial["corridor_width_m"], EXPECTED_PARTIAL_CORRIDOR_M),
    ):
        if abs(measured - expected) > 1e-6:
            raise AssertionError(f"{label}: literal {expected} != closed form {measured}")
    if blocked["through_pixels"] != 0:
        raise AssertionError("a closed door leaf left through-pixels: the blocked variant is wrong")
    u, v = FAR_POINT_PX
    depth = declared_depth(0, nominal)
    if abs(float(depth[v, u]) - EXPECTED_FAR_POINT_DEPTH_M) > 1e-9:
        raise AssertionError("the far-point pixel does not carry the declared far-wall depth")
    point = unproject(0, u, v, EXPECTED_FAR_POINT_DEPTH_M)
    for axis, (measured, expected) in enumerate(zip(point, EXPECTED_FAR_POINT_ODOM_M)):
        if abs(measured - expected) > 1e-6:
            raise AssertionError(f"far point axis {axis}: literal {expected} != closed form {measured}")
    for label, measured, expected in (
        ("sigma_z at the wall", EXPECTED_WALL_DEPTH_M**2 * SIGMA_DISPARITY_PX
         / (cam.FOCAL_LENGTH_PX * calibration().baseline_m), EXPECTED_SIGMA_Z_WALL_M),
        ("sigma_z at the far wall", EXPECTED_FAR_POINT_DEPTH_M**2 * SIGMA_DISPARITY_PX
         / (cam.FOCAL_LENGTH_PX * calibration().baseline_m), EXPECTED_SIGMA_Z_FAR_M),
        ("sigma_lateral at the wall", EXPECTED_WALL_DEPTH_M * SIGMA_PIXEL_PX / cam.FOCAL_LENGTH_PX,
         EXPECTED_SIGMA_LATERAL_WALL_M),
        ("sigma_lateral at the far wall",
         EXPECTED_FAR_POINT_DEPTH_M * SIGMA_PIXEL_PX / cam.FOCAL_LENGTH_PX,
         EXPECTED_SIGMA_LATERAL_FAR_M),
        ("target sigma normal at the wall",
         EXPECTED_SIGMA_Z_WALL_M + POSE_BOUND_ADDED_M, EXPECTED_TARGET_SIGMA_WALL_M),
        ("target sigma normal at the far wall",
         EXPECTED_SIGMA_Z_FAR_M + POSE_BOUND_ADDED_M, EXPECTED_TARGET_SIGMA_FAR_M),
        ("target sigma lateral at the wall",
         EXPECTED_SIGMA_LATERAL_WALL_M + POSE_BOUND_ADDED_M, EXPECTED_TARGET_SIGMA_LATERAL_WALL_M),
        ("target sigma lateral at the far wall",
         EXPECTED_SIGMA_LATERAL_FAR_M + POSE_BOUND_ADDED_M, EXPECTED_TARGET_SIGMA_LATERAL_FAR_M),
    ):
        if abs(measured - expected) > 1e-6:
            raise AssertionError(f"{label}: literal {expected} != closed form {measured}")
    for region in PROBE_REGIONS:
        classes = {_derived_probe_class(cell, nominal) for cell in probe_region_cells(region)}
        expect = region["expect"]
        if expect == "free" and classes != {"free"}:
            raise AssertionError(f"{region['probe_id']}: derived {classes}, expected all free")
        if expect == "unknown" and classes != {"unknown"}:
            raise AssertionError(f"{region['probe_id']}: derived {classes}, expected all unknown")
        if expect == "at_least_one_occupied" and "occupied" not in classes:
            raise AssertionError(f"{region['probe_id']}: derived {classes}, expected an occupied cell")
        if expect == "free":
            for cell in probe_region_cells(region):
                centre = cell_center(cell)
                corners = [
                    project(
                        0,
                        tuple(
                            centre[axis] + sign * VOXEL_M / 2.0
                            for axis, sign in enumerate(signs)
                        ),
                    )
                    for signs in (
                        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
                    )
                ]
                u_span = max(corner[0] for corner in corners) - min(corner[0] for corner in corners)
                v_span = max(corner[1] for corner in corners) - min(corner[1] for corner in corners)
                if u_span * v_span < 9.0:
                    raise AssertionError(
                        f"{region['probe_id']} cell {cell} subtends {u_span * v_span:.1f} px: fewer "
                        "than the three distinct clearing rays the free rule requires"
                    )


def manual_values_markdown() -> str:
    """truth/MANUAL-VALUES.md: the closed-form derivations and pre-registered tolerances."""
    aperture = aperture_evidence(0, APERTURE_BOX_PX, variant_surfaces("nominal"))
    return f"""# Doorway fixture: hand-derived values and pre-registered tolerances

Provenance: **DIAGNOSTIC**. Every value is authored or derived in closed form from the
authored scene below. Nothing here was measured on a vehicle, and nothing here is
scored. The capture pose is simulator-scene truth, so no result from this fixture can
enter a sensor-derived gate.

This file was written from the derivation, before any comparison ran. Its tolerances
are pre-registered: a failed tolerance is a finding to report, never a number to
adjust (the P01-C F1 lesson, APPROVAL-RECORD.md:1053-1068). Floor-plane metric depth
is unresolved there, so **no floor surface is declared here** and no expected value
depends on floor depth.

## 1. Authored scene (odom, right-handed, gravity-aligned; x through the doorway,
y left, z up)

| Element | Value |
|---|---|
| wall plane | x = {WALL_X_ODOM_M} m, y in [{WALL_Y_ODOM_M[0]}, {WALL_Y_ODOM_M[1]}], z in [{WALL_Z_ODOM_M[0]}, {WALL_Z_ODOM_M[1]}] (it continues below the opening, so downward rays return the wall and the space under the corridor is observed) |
| aperture | y in [{APERTURE_Y_ODOM_M[0]}, {APERTURE_Y_ODOM_M[1]}] ({APERTURE_Y_ODOM_M[1] - APERTURE_Y_ODOM_M[0]} m wide), z in [{APERTURE_Z_ODOM_M[0]}, {APERTURE_Z_ODOM_M[1]}] |
| far wall | x = {FAR_WALL_X_ODOM_M} m, y in [{FAR_WALL_Y_ODOM_M[0]}, {FAR_WALL_Y_ODOM_M[1]}], z in [{FAR_WALL_Z_ODOM_M[0]}, {FAR_WALL_Z_ODOM_M[1]}] (it continues below the band anyone looks through, so downward rays still return a surface) |
| floor | none declared (see above) |
| view 1 body | ({VIEWS[0]['body_position_odom_m'][0]}, {VIEWS[0]['body_position_odom_m'][1]}, {VIEWS[0]['body_position_odom_m'][2]}) m, identity quaternion, t = {VIEWS[0]['sim_time_s']} s |
| view 2 body | ({VIEWS[1]['body_position_odom_m'][0]}, {VIEWS[1]['body_position_odom_m'][1]}, {VIEWS[1]['body_position_odom_m'][2]}) m, identity quaternion, t = {VIEWS[1]['sim_time_s']} s |
| nav_epoch | `{NAV_EPOCH}` |

The wall is declared as three rectangular patches around the aperture (left, right and
header), so `camera.declared_depth_map` (camera.py:678) reproduces the scene exactly.
Rays that pass the aperture continue to the far wall; a ray that leaves the far wall's
span returns nothing.

## 2. Camera and depth conventions (pinned rig, `first-indoor-stereo-1` v2)

* f = {cam.FOCAL_LENGTH_PX:.9f} px, principal point = ({cam.PRINCIPAL_POINT_PX[0]}, {cam.PRINCIPAL_POINT_PX[1]}) px, baseline B = {calibration().baseline_m} m.
* Pixel convention (columns along body -y, rows along body -z, optical axis body +x):
  u = c_x - f (P_y - C_y)/(P_x - C_x) and v = c_y - f (P_z - C_z)/(P_x - C_x),
  with C the camera optical centre = body position + (0.05, 0.05, 0.05) m.
* Depth convention: z = f B / d metres along the left rectified optical axis. In the
  fixture's ray parameterisation the direction's x component is 1, so the declared
  depth equals the x-distance from the camera.
* Wall depth (camera at x = 0.05 m) = 3.00 - 0.05 = **{EXPECTED_WALL_DEPTH_M} m**;
  far-wall depth = 6.00 - 0.05 = **{EXPECTED_FAR_POINT_DEPTH_M} m**.
* Uncertainty envelope sigma_z = z^2 sigma_d / (f B) with the authored sigma_d =
  {SIGMA_DISPARITY_PX} px: sigma_z({EXPECTED_WALL_DEPTH_M} m) = **{EXPECTED_SIGMA_Z_WALL_M:.6f} m**,
  sigma_z({EXPECTED_FAR_POINT_DEPTH_M} m) = **{EXPECTED_SIGMA_Z_FAR_M:.6f} m**. Transverse sigma = z sigma_px / f
  with sigma_px = {SIGMA_PIXEL_PX} px: **{EXPECTED_SIGMA_LATERAL_WALL_M:.6f} m** at the wall and
  **{EXPECTED_SIGMA_LATERAL_FAR_M:.6f} m** at the far wall.
* A grounded target reports the *propagated* envelope: the depth term above plus the
  capture pose's declared 1-sigma bound of {POSE_BOUND_ADDED_M} m, added linearly rather than in
  quadrature because independence between the depth envelope and the shared pose bound is
  not established (section 7.3). So the aperture's normal and tangential terms are
  **{EXPECTED_TARGET_SIGMA_WALL_M:.6f} m** and **{EXPECTED_TARGET_SIGMA_LATERAL_WALL_M:.6f} m**, and the far-wall point's are
  **{EXPECTED_TARGET_SIGMA_FAR_M:.6f} m** and **{EXPECTED_TARGET_SIGMA_LATERAL_FAR_M:.6f} m**.
* Validity: border margin {BORDER_MARGIN_PX} px and every no-return sample are invalid.
  An invalid sample grounds nothing and clears nothing.

## 3. Selections (closed form)

The aperture selection is the outward-rounded projection of the wall rectangle
y in {APERTURE_SELECTION_RECTANGLE_ODOM_M['y_odom_m']}, z in {APERTURE_SELECTION_RECTANGLE_ODOM_M['z_odom_m']}: **box {APERTURE_BOX_PX}** in view 1.
The far-point selection is the single pixel **{FAR_POINT_PX}**, whose declared depth is {EXPECTED_FAR_POINT_DEPTH_M} m, so

    p_odom = C + z (1, -(u-c_x)/f, -(v-c_y)/f) = ({EXPECTED_FAR_POINT_ODOM_M[0]}, {EXPECTED_FAR_POINT_ODOM_M[1]:.9f}, {EXPECTED_FAR_POINT_ODOM_M[2]:.9f}) m

The invalid-depth selection is the box {OPEN_BAND_BOX_PX}: inside the declared border margin,
so every sample is masked and it grounds to nothing at all.

## 4. Aperture polygon (the one grounding path)

A pixel is *through* when its declared depth is finite and more than {THROUGH_MARGIN_M} m beyond the wall.
Each through pixel's ray meets the wall plane at

    y = C_y + (3.0 - C_x) (-(u-c_x)/f),  z = C_z + (3.0 - C_x) (-(v-c_y)/f)

The through pixels of view 1 span u in [{aperture['through_pixel_bbox_px'][0]}, {aperture['through_pixel_bbox_px'][1]}],
v in [{aperture['through_pixel_bbox_px'][2]}, {aperture['through_pixel_bbox_px'][3]}], giving wall-plane
y in [{aperture['opening_plane_y_m'][0] - APERTURE_EDGE_SHRINK_M:.6f}, {aperture['opening_plane_y_m'][1] + APERTURE_EDGE_SHRINK_M:.6f}], z in [{aperture['opening_plane_z_m'][0] - APERTURE_EDGE_SHRINK_M:.6f}, {aperture['opening_plane_z_m'][1] + APERTURE_EDGE_SHRINK_M:.6f}].
Shrinking by the authored edge uncertainty {APERTURE_EDGE_SHRINK_M} m (1 px of ray spread at the wall is
{EXPECTED_WALL_DEPTH_M / cam.FOCAL_LENGTH_PX:.6f} m, so {APERTURE_EDGE_SHRINK_M} m covers {APERTURE_EDGE_SHRINK_M * cam.FOCAL_LENGTH_PX / EXPECTED_WALL_DEPTH_M:.2f} px) gives the
**conservative opening polygon y in [{aperture['opening_plane_y_m'][0]:.6f}, {aperture['opening_plane_y_m'][1]:.6f}],
z in [{aperture['opening_plane_z_m'][0]:.6f}, {aperture['opening_plane_z_m'][1]:.6f}]** on the plane x = 3.0, width
{EXPECTED_APERTURE_WIDTH_M} m.

The polygon plane is the wall. The far wall at x = 6.0 is the depth *behind* the opening:
a ray through the doorway ends there, and that establishes visible space through the
aperture without locating the door frame. Ray direction is `-x` on the far side, so the
traversed direction is `+x`.

The observed opening is bounded by where rays that pass it still return a surface: the
observed z extent ({aperture['opening_plane_z_m'][0]:.6f}..{aperture['opening_plane_z_m'][1]:.6f}) is narrower than the authored
aperture (0.0..2.0). The polygon is therefore a **lower bound on the usable opening**,
never an outer bounding box, exactly as the specification requires for fit and traversal.

## 5. Fit and route (declared development parameters)

Voxel {VOXEL_M} m; inflation = body sphere {BODY_RADIUS_M} m + error allowance {ERROR_ALLOWANCE_M} m =
**{INFLATION_M} m**. Inflated corridor width = {EXPECTED_APERTURE_WIDTH_M} - 2({INFLATION_M}) = **{EXPECTED_APERTURE_CORRIDOR_M} m > 0: it fits**.
The cell rule is `index = floor((value - lower_bound)/{VOXEL_M})` in float64 over the authored
bounds (x [{SUBMAP_BOUNDS_ODOM_M['x'][0]}, {SUBMAP_BOUNDS_ODOM_M['x'][1]}], y [{SUBMAP_BOUNDS_ODOM_M['y'][0]}, {SUBMAP_BOUNDS_ODOM_M['y'][1]}], z [{SUBMAP_BOUNDS_ODOM_M['z'][0]}, {SUBMAP_BOUNDS_ODOM_M['z'][1]}]).
Inflation of the authored opening leaves the passable band y in
[{aperture['opening_plane_y_m'][0] + INFLATION_M:.6f}, {aperture['opening_plane_y_m'][1] - INFLATION_M:.6f}] and the crossing plane x in [2.6, 3.4].

Narrow variant (aperture y in [-0.15, 0.55]): width {EXPECTED_NARROW_WIDTH_M} m, corridor
{EXPECTED_NARROW_CORRIDOR_M} m < 0, so the constraint named `aperture_clearance` rules the traversal out
with well-supported edges. Blocked variant (a door leaf filling the opening): zero through
pixels, so no free evidence exists through the aperture and no route can be found
(`no_known_supported_route`, execution blocked). Partial block (a leaf over y in
[0.7, 0.9]): width {EXPECTED_PARTIAL_WIDTH_M} m, corridor {EXPECTED_PARTIAL_CORRIDOR_M} m > 0, so the same planner
replans through the remaining observed free part.

## 6. Occupancy probes (closed form)

Log-odds hit +{LOG_ODDS_HIT} / pass {LOG_ODDS_PASS}, clamped to +/-{LOG_ODDS_CLAMP}; free at score >= +{FREE_THRESHOLD} with
at least {MIN_CLEARING_RAYS} distinct clearing rays and age <= {FRESHNESS_S} s; occupied at score >= +{OCCUPIED_THRESHOLD}.
Each probe region below lists closed-form classes derived from the authored surfaces by
the rules in `_derived_probe_class`, before any map code ran:

{chr(10).join(f"* `{region['probe_id']}` ({region['expect']}): x {region['x_m']}, y {region['y_m']}, z {region['z_m']} - {region['derivation']}" for region in PROBE_REGIONS)}

## 7. Pose-degradation injections (coordinator scope addition)

Each injection is a declared construction with a provenance-carrying threshold, applied
by the tests to the nominal fixture. The nominal state carries a declared pose sigma of
{NOMINAL_POSE_SIGMA_M} m, which is 3 sigma = {3 * NOMINAL_POSE_SIGMA_M} m <= the {ERROR_ALLOWANCE_M} m allowance; the limit a state must
stay inside is therefore sigma <= {POSE_SIGMA_LIMIT_M:.6f} m (three-sigma envelope inside the declared allowance).

{chr(10).join(f"* `{name}`: {definition['provenance']} -> expected {definition['expected']}" for name, definition in POSE_DEGRADATION.items())}

## 8. Pre-registered tolerances

{chr(10).join(f"* `{name}` = {value}" for name, value in TOLERANCES.items())}
"""


def agreement_diagnostic() -> dict:
    """Non-gate: SGBM on the rendered pairs versus the declared depth. Records, scores nothing."""
    surfaces = variant_surfaces("nominal")
    baseline = abs(calibration().T_camera_left_camera_right.translation_m[1])
    report = {"note": "non-gate diagnostic; nothing scored and no sensor claim is made"}
    for view_index, view in enumerate(VIEWS):
        left = render_camera(view_index, 0.0, surfaces)
        right = render_camera(view_index, -baseline, surfaces)
        product = cam.compute_validated_depth(
            left, right, calibration(), SGBM_SETTINGS, pair_id=view["pair_id"]
        )
        declared = declared_depth(view_index, surfaces)
        comparable = product.valid & np.isfinite(declared)
        error = np.abs(product.depth_m[comparable] - declared[comparable])
        relative = error / declared[comparable]
        within = (error <= 0.15) | (relative <= 0.10)
        report[view["observation_id"]] = {
            "valid_pixels": int(product.valid.sum()),
            "comparable_pixels": int(comparable.sum()),
            "median_abs_error_m": float(np.median(error)),
            "p95_abs_error_m": float(np.percentile(error, 95)),
            "within_declared_tolerance_fraction": float(within.mean()),
        }
    return report


# ---------------------------------------------------------------------------
# Building the fixture
# ---------------------------------------------------------------------------


def generated_files(root: Path) -> tuple[str, ...]:
    """Every file this generator writes, relative to the fixture directory."""
    members = [recorder.MANIFEST_FILENAME, recorder.AGENT_EVENTS_FILENAME]
    payload_dir = root / "payloads"
    if payload_dir.is_dir():
        members += [f"payloads/{entry.name}" for entry in sorted(payload_dir.iterdir()) if entry.is_file()]
    members += [
        "truth/scene.json",
        "truth/expected-answer.json",
        "truth/variants.json",
        "truth/MANUAL-VALUES.md",
    ]
    return tuple(members)


def build(root: Path) -> Path:
    """Write the whole fixture under ``root``, deterministically."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for stale in (root / recorder.MANIFEST_FILENAME, root / recorder.AGENT_EVENTS_FILENAME):
        if stale.exists():
            stale.unlink()
    payload_dir = root / "payloads"
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    truth_dir = root / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    recorder_ = Recorder(root)
    for seq, kind, stamp, sim_time_s, payload in episode_events():
        recorder_.record(kind, payload, stamp, sim_time_s)
    surfaces = variant_surfaces("nominal")
    for view in VIEWS:
        left, right = render_pair(view["view_index"], surfaces)
        recorder_.write_payload(view["left_file"], left)
        recorder_.write_payload(view["right_file"], right)
    recorder_.close(
        RunManifest(
            episode_id=EPISODE_ID,
            episode_kind="synthetic-fixture",
            suite=None,
            trial_group_id=None,
            arm=None,
            sensor_mode=R.SensorMode.POSE_ASSISTED,
            code_revision=None,
            config_hash=None,
            model_identity=None,
        )
    )
    for name, document in (
        ("scene.json", scene_document()),
        ("expected-answer.json", expected_answer()),
        ("variants.json", variants_document()),
    ):
        (truth_dir / name).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (truth_dir / "MANUAL-VALUES.md").write_text(manual_values_markdown(), encoding="utf-8")
    return root


def check() -> None:
    """Rebuild into a temporary directory and verify every byte in place."""
    check_numbers()
    with tempfile.TemporaryDirectory() as scratch:
        built = build(Path(scratch) / "doorway")
        problems = []
        for member in generated_files(built):
            on_disk = FIXTURE_DIR / member
            if not on_disk.is_file():
                problems.append(f"{member}: missing from the fixture")
                continue
            expected = (built / member).read_bytes()
            actual = on_disk.read_bytes()
            if expected != actual:
                problems.append(
                    f"{member}: on disk {hashlib.sha256(actual).hexdigest()[:12]} != "
                    f"regenerated {hashlib.sha256(expected).hexdigest()[:12]}"
                )
        if problems:
            raise SystemExit("fixture is not byte-stable:\n  " + "\n  ".join(problems))
    print(
        f"fixture byte-stable: {len(generated_files(FIXTURE_DIR))} files, closed-form values "
        "agree with their authored literals"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the fixture byte-for-byte")
    parser.add_argument(
        "--agreement", action="store_true", help="run the non-gate stereo agreement diagnostic"
    )
    parser.add_argument("--root", type=Path, default=FIXTURE_DIR, help="fixture directory")
    arguments = parser.parse_args()
    if arguments.check:
        check()
        return 0
    if arguments.agreement:
        print(json.dumps(agreement_diagnostic(), indent=2, sort_keys=True))
        return 0
    check_numbers()
    build(arguments.root)
    print(f"wrote the doorway fixture under {arguments.root}")
    return 0



# Episode events: the agent projection of one synthetic doorway episode, written
# through P02's recorder so the fixture is P02 storage and not a bespoke file set.


def observation_record(view_index: int) -> R.Observation:
    view = VIEWS[view_index]
    return R.Observation(
        episode_id=EPISODE_ID,
        record_id=view["record_id"],
        sensor_ids=R.SensorIds(left="camera_left", right="camera_right", imu="imu"),
        sequence=view["sequence"],
        capture_stamp=R.ClockStamp(HOST_ID, CLOCK_ID, view["capture_ns"]),
        receipt_stamp=R.ClockStamp(HOST_ID, CLOCK_ID, view["receipt_ns"]),
        sim_time_s=view["sim_time_s"],
        pair_id=view["pair_id"],
        left_payload=f"payloads/{view['left_file']}",
        right_payload=f"payloads/{view['right_file']}",
        encoding="ppm-p6",
        width=IMAGE_SIZE[0],
        height=IMAGE_SIZE[1],
        calibration_id=CALIBRATION_ID,
        capture_pose_ref="pose-record-1",
        quality=None,
        depth_source="declared:fixture-declared-depth",
    )


def selection_records() -> tuple[R.VisualSelection, ...]:
    convention = calibration().pixel_convention
    u0, v0, u1, v1 = APERTURE_BOX_PX
    b0, c0, b1, c1 = OPEN_BAND_BOX_PX
    w0, x0, w1, x1 = APERTURE_BOX_V2_PX
    return (
        R.VisualSelection(
            selection_id=SELECTION_APERTURE,
            observation_id=VIEWS[0]["observation_id"],
            coordinate_convention=convention,
            geometry_kind=R.SelectionGeometry.BOX,
            geometry=(float(u0), float(v0), float(u1), float(v1)),
            crop_transform=None,
            description="the doorway opening in the wall ahead",
            confidence=None,
        ),
        R.VisualSelection(
            selection_id=SELECTION_FAR_POINT,
            observation_id=VIEWS[0]["observation_id"],
            coordinate_convention=convention,
            geometry_kind=R.SelectionGeometry.POINT,
            geometry=(float(FAR_POINT_PX[0]), float(FAR_POINT_PX[1])),
            crop_transform=None,
            description="a point seen through the doorway",
            confidence=None,
        ),
        R.VisualSelection(
            selection_id=SELECTION_OPEN_BAND,
            observation_id=VIEWS[0]["observation_id"],
            coordinate_convention=convention,
            geometry_kind=R.SelectionGeometry.BOX,
            geometry=(float(b0), float(c0), float(b1), float(c1)),
            crop_transform=None,
            description="a selection inside the masked border: every sample is invalid",
            confidence=None,
        ),
        R.VisualSelection(
            selection_id=SELECTION_APERTURE_V2,
            observation_id=VIEWS[1]["observation_id"],
            coordinate_convention=convention,
            geometry_kind=R.SelectionGeometry.BOX,
            geometry=(float(w0), float(x0), float(w1), float(x1)),
            crop_transform=None,
            description="the same doorway from the second viewpoint",
            confidence=None,
        ),
    )


def mission_record() -> R.MissionContract:
    return R.MissionContract(
        mission_id="p03-doorway-mission",
        instruction="Fly through the open doorway to the far side of the wall and settle there.",
        interpreted_requirements=("traverse_observed_aperture", "settle_beyond_wall"),
        revision=1,
        evidence_obligations=("cite_the_selection_and_observation_for_each_grounded_target",),
        return_obligation="return_to_launch_documented_not_exercised",
        allowed_scope="single_room_pair_across_one_doorway",
        budget=(("mission_time_s", 60.0), ("route_length_m", 20.0)),
        unresolved_questions=("floor_plane_metric_depth_is_unresolved",),
    )


def goal_records() -> tuple[R.SpatialGoal, ...]:
    return (
        R.SpatialGoal(
            proposal_id="goal-traverse-aperture",
            request_id=None,
            fingerprint="p03-traverse-aperture-1",
            mission_revision=1,
            base_goal_revision=0,
            selection_ids=(SELECTION_APERTURE,),
            target_refs=(),
            intent="traverse",
            constraints=("uav_must_fit_observed_opening", "unknown_space_is_not_free"),
            completion_condition="cross_to_the_far_side_and_settle_in_the_exit_region",
            lease_bounds=(("lease_s", 30.0), ("path_length_m", 15.0)),
            local_discretion_bounds=(("speed_mps", V_MAX_MPS),),
        ),
        R.SpatialGoal(
            proposal_id="goal-approach-unknown-band",
            request_id=None,
            fingerprint="p03-approach-open-band-1",
            mission_revision=1,
            base_goal_revision=1,
            selection_ids=(SELECTION_OPEN_BAND,),
            target_refs=(),
            intent="approach",
            constraints=("standoff_from_an_unobserved_surface",),
            completion_condition="reach_a_supported_standoff_region",
            lease_bounds=(("lease_s", 20.0),),
            local_discretion_bounds=(("speed_mps", V_MAX_MPS),),
        ),
    )


def episode_events() -> tuple[tuple[int, str, R.ClockStamp, float | None, object], ...]:
    """The input events in recorded order: mission, observations, selections, goals."""
    events: list[tuple[int, str, R.ClockStamp, float | None, object]] = [
        (0, "mission", R.ClockStamp(HOST_ID, CLOCK_ID, 900_000_000), None, mission_record())
    ]
    for view_index, view in enumerate(VIEWS):
        events.append(
            (
                len(events),
                "observation",
                R.ClockStamp(HOST_ID, CLOCK_ID, view["capture_ns"]),
                view["sim_time_s"],
                observation_record(view_index),
            )
        )
    for selection in selection_records():
        events.append(
            (
                len(events),
                "selection",
                R.ClockStamp(HOST_ID, CLOCK_ID, 1_600_000_000 + len(events) * 1_000_000),
                None,
                selection,
            )
        )
    for goal in goal_records():
        events.append(
            (
                len(events),
                "goal",
                R.ClockStamp(HOST_ID, CLOCK_ID, 1_700_000_000 + len(events) * 1_000_000),
                None,
                goal,
            )
        )
    return tuple(events)

if __name__ == "__main__":
    sys.exit(main())
