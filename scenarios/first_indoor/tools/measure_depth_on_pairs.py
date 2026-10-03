#!/usr/bin/env python3
"""Measure the depth channel on captured stereo pairs.

Reads the declared matcher, border and depth window out of
``configs/first_indoor.yaml`` the same way the runtime does, so the numbers
describe the shipped configuration rather than a copy of it, and reports:

* per-frame image statistics and the project's own gradient metric,
* the pooled per-reason rejection breakdown over every frame,
* per-pose rows so a pose that behaves differently is visible as a pose.

Usage, from the repository root:

    PYTHONPATH=src python3 scenarios/first_indoor/tools/measure_depth_on_pairs.py \\
        --label before --pairs work/runs/p05/scenetexture/before \\
        --out work/runs/p05/scenetexture/before-depth.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from embodied.perception import camera as C

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs" / "first_indoor.yaml"


def depth_settings(config: dict) -> dict:
    calibration = config["calibration"]
    return {
        **calibration["matcher"],
        "border_px": calibration["bounds"]["border_px"],
        "depth_range_m": calibration["bounds"]["depth_range_m"],
    }


def frame_stats(left: np.ndarray) -> dict:
    grey = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return {
        "mean": float(left.mean()),
        "sd": float(left.std()),
        "grad": float(np.abs(np.diff(grey, axis=1)).mean() + np.abs(np.diff(grey, axis=0)).mean()),
        "saturated_frac": float((left >= 250).mean()),
        "dark_frac": float((left <= 5).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = depth_settings(config)
    calibration = C.build_calibration()

    totals: dict[str, int] = defaultdict(int)
    rows = []
    for left_path in sorted(args.pairs.glob("*-left.ppm")):
        right_path = left_path.with_name(left_path.name.replace("-left.ppm", "-right.ppm"))
        if not right_path.is_file():
            continue
        left = C.read_ppm(left_path)
        right = C.read_ppm(right_path)
        product = C.compute_validated_depth(left, right, calibration, settings)
        counts = {reason: int((product.reasons == code).sum())
                  for code, reason in enumerate(C.REASON_NAMES)}
        for reason, count in counts.items():
            totals[reason] += count
        total_px = sum(counts.values()) or 1
        row = {
            "pose": left_path.name.replace("-left.ppm", ""),
            **frame_stats(left),
            **{f"frac_{r}": counts[r] / total_px for r in C.REASON_NAMES},
        }
        rows.append(row)

    total_px = sum(totals.values()) or 1
    pooled = {r: totals[r] / total_px for r in C.REASON_NAMES}

    print(f"=== {args.label}: {len(rows)} frame(s)")
    print(f"{'pose':<12}{'grad':>7}{'valid':>8}{'no_ret':>8}{'lr_mm':>8}{'range':>8}{'border':>8}")
    print("-" * 60)
    for row in rows:
        print(f"{row['pose']:<12}{row['grad']:>7.2f}{row['frac_valid']:>8.3f}"
              f"{row['frac_no_return']:>8.3f}{row['frac_lr_mismatch']:>8.3f}"
              f"{row['frac_depth_range']:>8.3f}{row['frac_border']:>8.3f}")
    print("-" * 60)
    print(f"{'POOLED':<12}{np.mean([r['grad'] for r in rows]):>7.2f}{pooled['valid']:>8.3f}"
          f"{pooled['no_return']:>8.3f}{pooled['lr_mismatch']:>8.3f}"
          f"{pooled['depth_range']:>8.3f}{pooled['border']:>8.3f}")

    payload = {
        "label": args.label,
        "pairs_dir": str(args.pairs),
        "frames": len(rows),
        "pooled": pooled,
        "mean_grad": float(np.mean([r["grad"] for r in rows])) if rows else None,
        "per_pose": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
