"""What the declared depth window discards, and whether it would have changed the verdict.

The B3 gate compares only the samples the declared window keeps: a measured depth
outside ``depth_range_m`` is marked ``REASON_DEPTH_RANGE`` and never reaches the
comparison. Near the window's far edge that filter is narrower than the accuracy
criterion — at 5.88 m the criterion accepts ``max(0.15, 0.10·z) = 0.588 m``, so it
would accept measurements out to 6.47 m, while the window rejects everything past
6.00 m. A surface measured near the edge therefore reports a within-fraction over a
survivor set, and a reader cannot tell from the gate alone whether the discarded
samples would have passed.

This tool answers that for one surface: it counts the reasons on the surface's
declared pixels, and it recomputes the within-fraction with the discarded samples
restored. It moves no bound and writes no calibration; its artifact is a table.

Usage::

    python -m embodied.perception.depth_window_survivorship \
        --config work/fardepth.config.yaml --surface wall_x5_front
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from embodied.perception import camera as C


def within_fraction(errors_m: np.ndarray, declared_m: np.ndarray,
                    abs_tol_m: float, rel_tol: float) -> float | None:
    """The fraction of samples inside ``|error| <= max(abs_tol, rel_tol · declared)``.

    Kept pure so the one piece of arithmetic a reader has to trust can be checked
    without a stereo pipeline.
    """
    if errors_m.size == 0:
        return None
    if errors_m.shape != declared_m.shape:
        raise ValueError("an error needs the declared depth it was measured against")
    return float(np.mean(np.abs(errors_m) <= np.maximum(abs_tol_m, rel_tol * declared_m)))


def restored_verdict(kept_errors_m: np.ndarray, kept_declared_m: np.ndarray,
                     dropped_errors_m: np.ndarray, dropped_declared_m: np.ndarray,
                     abs_tol_m: float, rel_tol: float) -> dict:
    """Within-tolerance fractions before and after restoring the window's discards."""
    all_errors = np.concatenate([kept_errors_m, dropped_errors_m])
    all_declared = np.concatenate([kept_declared_m, dropped_declared_m])
    return {
        "n_kept": int(kept_errors_m.size),
        "n_dropped": int(dropped_errors_m.size),
        "kept_within_fraction": within_fraction(kept_errors_m, kept_declared_m, abs_tol_m, rel_tol),
        "restored_within_fraction": within_fraction(all_errors, all_declared, abs_tol_m, rel_tol),
        "kept_median_abs_error_m": float(np.median(np.abs(kept_errors_m))) if kept_errors_m.size else None,
        "dropped_median_abs_error_m": float(np.median(np.abs(dropped_errors_m))) if dropped_errors_m.size else None,
        "dropped_max_abs_error_m": float(np.max(np.abs(dropped_errors_m))) if dropped_errors_m.size else None,
        "dropped_fraction_of_surface": (
            float(dropped_errors_m.size / all_errors.size) if all_errors.size else None
        ),
    }


def measure(config_path: Path, surface_name: str) -> dict:
    section = yaml.safe_load(Path(config_path).read_text())["calibration"]
    repo_root = Path(config_path).resolve().parents[1]
    bounds, matcher, referee, evidence = (
        section["bounds"], section["matcher"], section["referee"], section["evidence"]
    )
    calibration = C.build_calibration()
    surfaces = [C.DeclaredSurface.from_config(entry) for entry in referee["surfaces"]]
    named = [i for i, s in enumerate(surfaces) if s.name == surface_name]
    if not named:
        raise SystemExit(f"no declared surface named {surface_name!r}")
    index = named[0]

    body = tuple(float(v) for v in referee["body_position_world_m"])
    rotation = C._quaternion_rotation(tuple(float(v) for v in referee["body_quaternion_world_wxyz"]))
    camera_position = tuple(b + t for b, t in zip(body, calibration.T_body_camera_left.translation_m))
    camera_rotation = rotation @ C._quaternion_rotation(
        calibration.T_body_camera_left.quaternion_wxyz
    )
    declared, attribution = C.declared_depth_map(
        surfaces, camera_position, camera_rotation, C.FOCAL_LENGTH_PX, C.PRINCIPAL_POINT_PX, (640, 480)
    )
    margin = int(bounds["border_px"])
    z_min, z_max = (float(v) for v in bounds["depth_range_m"])
    border = np.ones(declared.shape, dtype=bool)
    border[:margin, :] = False
    border[-margin:, :] = False
    border[:, :margin] = False
    border[:, -margin:] = False
    in_window = np.isfinite(declared) & (declared >= z_min) & (declared <= z_max)
    target = border & (attribution == index) & in_window

    reasons: dict[str, int] = {}
    kept_errors, kept_declared, dropped_errors, dropped_declared = [], [], [], []
    pairs = sorted((repo_root / evidence["pair_dirs"][0]).glob("*-left.ppm"))
    for left in pairs:
        right = left.with_name(left.name.replace("-left", "-right"))
        product = C.compute_validated_depth(
            C.read_ppm(left), C.read_ppm(right), calibration,
            {**matcher, "border_px": bounds["border_px"], "depth_range_m": bounds["depth_range_m"]},
            pair_id=left.stem, capture_stamp=None, receipt_stamp=None, sim_time_s=None,
            pose_provenance=C.PoseProvenance(label="DIAGNOSTIC", detail="survivorship analysis"),
        )
        codes = product.reasons[target]
        for code in np.unique(codes):
            name = C.REASON_NAMES[int(code)]
            reasons[name] = reasons.get(name, 0) + int((codes == code).sum())
        depth = product.depth_m[target]
        declared_here = declared[target]
        finite = np.isfinite(depth)
        keep = product.valid[target] & finite
        drop = finite & (~product.valid[target]) & (codes == C.REASON_DEPTH_RANGE)
        kept_errors.append(depth[keep] - declared_here[keep])
        kept_declared.append(declared_here[keep])
        dropped_errors.append(depth[drop] - declared_here[drop])
        dropped_declared.append(declared_here[drop])

    kept_e = np.concatenate(kept_errors)
    kept_d = np.concatenate(kept_declared)
    drop_e = np.concatenate(dropped_errors)
    drop_d = np.concatenate(dropped_declared)
    total = sum(reasons.values())
    return {
        "surface": surface_name,
        "pairs": len(pairs),
        "pixels_on_surface": int(target.sum()),
        "reasons_fraction": {
            name: (count / total if total else None)
            for name, count in sorted(reasons.items(), key=lambda kv: -kv[1])
        },
        "declared_depth_m": {
            "min": float(declared[target].min()),
            "max": float(declared[target].max()),
            "mean": float(declared[target].mean()),
        },
        "tolerance_m": {
            "abs": float(bounds["depth_abs_tol_m"]),
            "rel": float(bounds["depth_rel_tol"]),
            "at_declared_mean": float(
                max(bounds["depth_abs_tol_m"],
                    bounds["depth_rel_tol"] * float(declared[target].mean()))
            ),
            "window_far_edge": z_max,
        },
        **restored_verdict(
            kept_e, kept_d, drop_e, drop_d,
            float(bounds["depth_abs_tol_m"]), float(bounds["depth_rel_tol"]),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m embodied.perception.depth_window_survivorship")
    parser.add_argument("--config", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    table = measure(Path(args.config), args.surface)
    text = json.dumps(table, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
