#!/usr/bin/env python3
"""Capture stereo pairs at declared poses -- no autopilot, no estimator, no SITL.

Why this exists
---------------
Every other capture path in this project runs a full bring-up: Webots plus SITL
plus the estimator feed, and the shared controller publishes pairs only once
SITL connects. That makes a scene change unmeasurable at fixed poses without
flying -- and a flight moves the camera, so a before/after comparison would
confound the scene with the route it happened to fly.

This controller poses the airframe kinematically. It sets the robot's own
translation and rotation fields and re-asserts them on **every** step, so the
pose at capture time is exactly the pose that was declared rather than whatever
the physics did with it. Both cameras are then read in a single step with no
step between the two reads, which is the same guarantee the flight adapter's
pair read gives: one capture instant per pair.

It never starts the autopilot, never arms, never writes `armed`, and produces
nothing but the pairs, their capture stamps and a JSON of what it did.

Arguments (from the world's ``controllerArgs``), all `--flag value`:

  --out-dir DIR        directory to write ``<pose-id>-left.ppm`` into
  --poses FILE         JSON {"poses": [{"id", "translation", "yaw_deg", ...}]}
  --camera-left NAME   default "camera left"
  --camera-right NAME  default "camera right"
  --settle-steps N     steps to hold the pose before reading (default 30)
  --period-ms MS       camera sampling period (default 40)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

_CONVERSION = None
_IMPORT_ERROR = None


def _import_conversion():
    """Import the project's own BGRA->RGB and PPM helpers.

    The conversion is imported rather than re-implemented so a captured frame
    is byte-identical in format to one a flight wrote. Uses the same documented
    source-tree handoff the shared controller uses.
    """
    global _IMPORT_ERROR
    try:
        from embodied.platform.webots_ardupilot import bgra_to_rgb8, ppm_bytes

        return bgra_to_rgb8, ppm_bytes
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        _IMPORT_ERROR = f"{type(error).__name__}: {error}"
    source_root = os.environ.get("EMBODIED_SRC")
    if not source_root:
        _IMPORT_ERROR = f"{_IMPORT_ERROR} (EMBODIED_SRC was not set, so there was no fallback)"
        return None
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        from embodied.platform.webots_ardupilot import bgra_to_rgb8, ppm_bytes

        return bgra_to_rgb8, ppm_bytes
    except Exception as error:  # noqa: BLE001
        _IMPORT_ERROR = f"{_IMPORT_ERROR}; fallback failed: {type(error).__name__}: {error}"
        return None


_CONVERSION = _import_conversion()

if _CONVERSION is None:
    print(
        "capture_poses_controller: cannot import the conversion helpers\n"
        f"  interpreter: {sys.executable}\n"
        f"  error: {_IMPORT_ERROR}",
        flush=True,
    )
    raise SystemExit(2)

bgra_to_rgb8, ppm_bytes = _CONVERSION

# Supervisor, not Robot: the pose is set by writing the node's own fields, and
# only a supervisor may write them. The generated capture world declares
# ``supervisor TRUE`` for exactly this reason.
from controller import Supervisor  # noqa: E402 - Webots puts this on the controller path


def parse_args(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and index + 1 < len(argv):
            out[token] = argv[index + 1]
            index += 2
        else:
            index += 1
    return out


def main() -> int:
    args = parse_args(sys.argv[1:])
    out_dir = Path(args.get("--out-dir", "capture-pairs")).expanduser().resolve()
    poses_file = Path(args.get("--poses", "poses.json")).expanduser().resolve()
    left_name = args.get("--camera-left", "camera left")
    right_name = args.get("--camera-right", "camera right")
    settle_steps = int(args.get("--settle-steps", "30"))
    period_ms = int(args.get("--period-ms", "40"))

    if not poses_file.is_file():
        print(f"capture_poses_controller: no poses file at {poses_file}", flush=True)
        return 2
    poses = json.loads(poses_file.read_text(encoding="utf-8"))["poses"]
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    left = robot.getDevice(left_name)
    right = robot.getDevice(right_name)
    for camera in (left, right):
        camera.enable(period_ms)

    self_node = robot.getSelf()
    translation_field = self_node.getField("translation")
    rotation_field = self_node.getField("rotation")

    width = int(left.getWidth())
    height = int(left.getHeight())
    print(
        f"capture_poses_controller: timestep {timestep} ms, camera {width}x{height}, "
        f"period {period_ms} ms, {len(poses)} pose(s), out {out_dir}",
        flush=True,
    )

    records = []
    for pose in poses:
        pose_id = str(pose["id"])
        position = [float(v) for v in pose["translation"]]
        yaw_rad = math.radians(float(pose.get("yaw_deg", 0.0)))
        axis = [float(v) for v in pose.get("axis", [0.0, 0.0, 1.0])]

        # Hold the declared pose for the whole settle window. Re-asserting it on
        # every step is what makes the captured pose the declared one: gravity
        # and contact act between steps, and a pose set once would be gone.
        for _ in range(settle_steps):
            translation_field.setSFVec3f(position)
            rotation_field.setSFRotation([axis[0], axis[1], axis[2], yaw_rad])
            if robot.step(timestep) == -1:
                print("capture_poses_controller: Webots quit mid-capture", flush=True)
                return 3

        # Both eyes in one step, no step between the reads: one capture instant.
        left_buffer = left.getImage()
        right_buffer = right.getImage()
        if not left_buffer or not right_buffer:
            print(f"capture_poses_controller: empty image for {pose_id}", flush=True)
            return 4

        left_path = out_dir / f"{pose_id}-left.ppm"
        right_path = out_dir / f"{pose_id}-right.ppm"
        left_path.write_bytes(ppm_bytes(bgra_to_rgb8(left_buffer, width, height), width, height))
        right_path.write_bytes(ppm_bytes(bgra_to_rgb8(right_buffer, width, height), width, height))

        # The pose the vehicle actually reported at capture time, which is the
        # claim worth recording rather than the one that was asked for.
        reported = list(self_node.getPosition())
        records.append(
            {
                "id": pose_id,
                "commanded_translation": position,
                "commanded_yaw_deg": math.degrees(yaw_rad),
                "reported_translation": reported,
                "left": str(left_path),
                "right": str(right_path),
            }
        )
        print(
            f"CAPTURED {pose_id} at "
            f"({reported[0]:.4f}, {reported[1]:.4f}, {reported[2]:.4f})",
            flush=True,
        )

    (out_dir / "capture-record.json").write_text(
        json.dumps({"poses": records}, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(f"CAPTURE_DONE {len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
