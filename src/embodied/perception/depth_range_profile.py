"""Where the depth channel stops being accurate, as a function of range.

The B3 gate answers pass or fail for a surface. It does not answer *at what
range* accuracy stops, which is the question a ruling that widens the depth
operating range has to answer: ``sigma_z = z^2 * sigma_d / (f * B)`` grows
quadratically, so the samples a wider window newly admits can be systematically
worse than the ones it already had, and a ray whose far endpoint is badly placed
clears space past a real surface.

This tool runs the same pipeline, the declared referee surfaces and the declared
tolerances as the P01-C check, and bins every compared sample by its TRUE
(declared) depth. It is a measurement, not a gate: it moves no bound, writes no
calibration, and its artifact is a table.

Why a separate tool rather than an extra key in the check's receipt: the check is
a pre-registered artifact and its criterion is frozen (R19). Adding analysis to it
would change a sealed document; reading the same numbers out beside it does not.

Usage::

    python -m embodied.perception.depth_range_profile --config configs/first_indoor.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied.perception import camera as camera_module

# One band per metre of the declared window, so the bands are derived from the
# declaration rather than chosen to flatter any particular result.
BAND_WIDTH_M = 1.0


def band_edges(
    z_min: float, z_max: float, width: float = BAND_WIDTH_M
) -> list[tuple[float, float]]:
    """The bands, derived from the declared window rather than chosen by hand.

    A reader can then tell that no band was placed to flatter a result: move
    ``depth_range_m`` and the bands move with it.
    """
    edges: list[tuple[float, float]] = []
    edge = z_min
    while edge < z_max:
        edges.append((edge, min(edge + width, z_max)))
        edge += width
    return edges


def predicted_sigma_m(
    z_m: float, focal_px: float, baseline_m: float, sigma_d_px: float
) -> float:
    """The declared quantization envelope at one range: sigma_z = z^2 sigma_d / (f B).

    Reported beside the measured error so a reader can tell a quantization-limited
    result from a correspondence failure: an error at or under this envelope is
    what the declared disparity sigma predicts; an error several times above it is
    not quantization and no envelope widening will explain it.
    """
    return (z_m * z_m) * sigma_d_px / (focal_px * baseline_m)


def profile(config_path: Path) -> dict:
    """Measure depth error against declared depth, banded by range."""
    import yaml  # lazy: only a reader of the config needs it

    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text())
    section = config["calibration"]
    bounds = section["bounds"]
    matcher = section["matcher"]
    referee_cfg = section["referee"]
    evidence = section["evidence"]

    def resolve(rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else repo_root / path

    calibration = camera_module.build_calibration()
    surfaces = [
        camera_module.DeclaredSurface.from_config(entry) for entry in referee_cfg["surfaces"]
    ]

    # The capture-time pose is the scene's declared static start, exactly as the
    # check binds it. Simulated-scene truth, used offline by a measurement and
    # never fed to a runtime.
    body_position = tuple(float(v) for v in referee_cfg["body_position_world_m"])
    body_rotation = camera_module._quaternion_rotation(
        tuple(float(v) for v in referee_cfg["body_quaternion_world_wxyz"])
    )
    camera_translation = calibration.T_body_camera_left.translation_m
    camera_position = tuple(b + t for b, t in zip(body_position, camera_translation))
    camera_rotation = body_rotation @ camera_module._quaternion_rotation(
        calibration.T_body_camera_left.quaternion_wxyz
    )
    size = (camera_module._DECLARED_WIDTH_PX, camera_module._DECLARED_HEIGHT_PX)
    declared_depth, attribution = camera_module.declared_depth_map(
        surfaces,
        camera_position,
        camera_rotation,
        camera_module.FOCAL_LENGTH_PX,
        camera_module.PRINCIPAL_POINT_PX,
        size,
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

    abs_tol = float(bounds["depth_abs_tol_m"])
    rel_tol = float(bounds["depth_rel_tol"])
    provenance = camera_module.static_start_provenance(str(evidence["world"]))
    settings = {
        **matcher,
        "border_px": bounds["border_px"],
        "depth_range_m": bounds["depth_range_m"],
    }

    edges = band_edges(z_min, z_max)
    bands: dict[str, dict] = {
        f"{lo:.1f}-{hi:.1f}": {"n": 0, "within": 0, "errors": []} for lo, hi in edges
    }
    by_surface: dict[str, dict] = {
        surface.name: {"n": 0, "within": 0, "errors": []} for surface in surfaces
    }

    pairs_seen = 0
    for pair_dir in evidence["pair_dirs"]:
        for left_path, right_path, _journal in camera_module._find_pairs(resolve(pair_dir)):
            left = camera_module.read_ppm(left_path)
            right = camera_module.read_ppm(right_path)
            product = camera_module.compute_validated_depth(
                left, right, calibration, settings, pose_provenance=provenance
            )
            # The comparison the check itself makes, unchanged.
            compared = (
                product.valid & inside_border & in_window_declared & np.isfinite(declared_depth)
            )
            declared_here = declared_depth[compared]
            measured_here = product.depth_m[compared]
            errors = measured_here - declared_here
            within = np.abs(errors) <= np.maximum(abs_tol, rel_tol * declared_here)
            owner = attribution[compared]
            pairs_seen += 1

            for lo, hi in edges:
                key = f"{lo:.1f}-{hi:.1f}"
                if hi >= z_max:
                    mask = (declared_here >= lo) & (declared_here <= hi)
                else:
                    mask = (declared_here >= lo) & (declared_here < hi)
                bands[key]["n"] += int(mask.sum())
                bands[key]["within"] += int(within[mask].sum())
                bands[key]["errors"].append(errors[mask])

            for index, surface in enumerate(surfaces):
                mask = owner == index
                by_surface[surface.name]["n"] += int(mask.sum())
                by_surface[surface.name]["within"] += int(within[mask].sum())
                by_surface[surface.name]["errors"].append(errors[mask])

    def summarize(entry: dict) -> dict:
        finite = [e[np.isfinite(e)] for e in entry["errors"] if e.size]
        stacked = np.concatenate(finite) if finite else np.array([])
        n = int(entry["n"])
        magnitude = np.abs(stacked) if stacked.size else np.array([])
        return {
            "n_compared": n,
            "n_within_tolerance": int(entry["within"]),
            "within_fraction": (entry["within"] / n) if n else None,
            "median_abs_error_m": float(np.median(magnitude)) if magnitude.size else None,
            "p95_abs_error_m": float(np.percentile(magnitude, 95)) if magnitude.size else None,
            "max_abs_error_m": float(np.max(magnitude)) if magnitude.size else None,
            "median_signed_error_m": float(np.median(stacked)) if stacked.size else None,
        }

    return {
        "depth_range_m": list(bounds["depth_range_m"]),
        "tolerances": {"depth_abs_tol_m": abs_tol, "depth_rel_tol": rel_tol},
        "band_width_m": BAND_WIDTH_M,
        "pairs_measured": pairs_seen,
        "calibration": {
            "focal_px": float(calibration.left_intrinsics.focal_length_px[0]),
            "baseline_m": float(calibration.baseline_m),
            "disparity_quantization_sigma_px": float(
                matcher.get("disparity_quantization_sigma_px", 1.0)
            ),
        },
        "predicted_sigma_m_by_band": {
            key: predicted_sigma_m(
                (float(key.split("-")[0]) + float(key.split("-")[1])) / 2.0,
                float(calibration.left_intrinsics.focal_length_px[0]),
                float(calibration.baseline_m),
                float(matcher.get("disparity_quantization_sigma_px", 1.0)),
            )
            for key in bands
        },
        "by_range_band": {key: summarize(value) for key, value in bands.items()},
        "by_surface": {key: summarize(value) for key, value in by_surface.items()},
        "note": (
            "n_compared counts exactly the samples the check compares: a valid disparity, "
            "inside the border margin, inside the declared depth window, and with a finite "
            "declared depth. within_fraction is against max(depth_abs_tol_m, "
            "depth_rel_tol * declared depth), the declared per-sample criterion. "
            "predicted_sigma_m is the declared quantization envelope at the band's "
            "midpoint: an error far above it is not quantization."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m embodied.perception.depth_range_profile",
        description="depth accuracy against declared depth, banded by range",
    )
    parser.add_argument("--config", default="configs/first_indoor.yaml")
    parser.add_argument("--output", default=None, help="write the table here as JSON")
    args = parser.parse_args(argv)

    result = profile(Path(args.config))
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
