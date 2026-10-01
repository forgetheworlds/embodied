#!/usr/bin/env python3
"""Synthesize a small sensor-capture record to smoke-test the replay driver.

Generates a capture directory (records.jsonl + frames/ PPMs + luma hashes) for
a synthetic 45 s flight: 15 s static, 15 s lateral motion at 0.4 m/s (with 1 s
acceleration ramps), 15 s static again. The scene is a textured wall at 1.5 m
projected through the rig's own geometry (554.256 px focal, 0.1 m baseline,
optical x = -body y per the ov_stream extrinsics), and the IMU is the exact
body-frame FRD reading of that same motion, so the estimator's static
initializer has consistent inertial and visual evidence -- the same shape of
excitation the real bring-up uses (climb; here lateral).

No Webots, no SITL: this exists to validate the replay machinery end to end
(frames -> hashes -> socket -> STATE frames -> diff tools) before a single
flight is spent on the real capture.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied.platform import localization as loc  # noqa: E402

WIDTH, HEIGHT = 640, 480
FOCAL = 554.2562584220407
CX, CY = 320.0, 240.0
DEPTH = 1.5  # wall distance, m
BASELINE = 0.1  # stereo baseline, m
FPS_IMU = 500.0
FPS_PAIR = 10.0
DURATION = 45.0
RAMP = 1.0
# 0.15 m/s keeps the whole 2.25 m traverse inside the wall texture (a clamp
# would freeze the view and kill the disparity the initializer needs), and
# still moves the view ~5.5 px per frame at this depth -- well past the
# zero-velocity gate's 1 px temporal-disparity threshold. Motion is along
# body -y, which is optical +x: the view slides one way, the accelerometers
# read the other, and both agree with the same physical translation.
SPEED = -0.15  # body y (optical +x) lateral speed after the ramp, m/s
GRAVITY = 9.81
TEXTURE_WIDTH = 1700
CROP_BASE = 100  # where the left eye starts on the wall, px


def body_y_position(t: float) -> float:
    """The camera's body-y offset at time t, with 1 s acceleration ramps."""
    start, stop = 15.0, 30.0
    if t < start:
        return 0.0
    if t < start + RAMP:
        return 0.5 * (SPEED / RAMP) * (t - start) ** 2
    if t < stop - RAMP:
        return 0.5 * SPEED * RAMP + SPEED * (t - start - RAMP)
    if t < stop:
        tau = t - (stop - RAMP)
        return 0.5 * SPEED * RAMP + SPEED * (stop - RAMP - start - RAMP) + SPEED * tau - 0.5 * (SPEED / RAMP) * tau ** 2
    return SPEED * RAMP + SPEED * (stop - RAMP - start - RAMP) - 0.5 * SPEED * RAMP + SPEED * RAMP - 0.5 * (SPEED / RAMP) * RAMP**2 + 0.0


def body_y_accel(t: float) -> float:
    start, stop = 15.0, 30.0
    if start < t < start + RAMP:
        return SPEED / RAMP
    if stop - RAMP < t < stop:
        return -SPEED / RAMP
    return 0.0


def make_texture(seed: int = 7) -> np.ndarray:
    """A trackable textured wall: blurred noise keeps gradients everywhere."""
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(1200, TEXTURE_WIDTH), dtype=np.uint8)
    texture = noise
    for _ in range(3):
        texture = cv2.GaussianBlur(texture, (0, 0), 3.0)
        # re-inject mid-frequency detail so KLT has corners at every scale
        texture = np.clip(
            texture.astype(np.int16) + (noise.astype(np.int16) - 128) // 4, 0, 255
        ).astype(np.uint8)
    return texture



def frame(texture: np.ndarray, u0: float) -> bytes:
    """One view: the wall through the pinhole at lateral optical offset u0 px."""
    left = max(0, min(TEXTURE_WIDTH - WIDTH - 1, int(round(u0))))
    crop = texture[360 : 360 + HEIGHT, left : left + WIDTH]
    rgb = np.repeat(crop[:, :, np.newaxis], 3, axis=2).tobytes()
    return rgb


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "work/runs/p01-localization/synthetic-smoke/run-a")
    capture = out / "sensor-capture"
    frames_dir = capture / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = capture / "records.jsonl"

    with records.open("w", encoding="utf-8") as handle:
        rows = 0
        seq = 0

        def write(row: dict) -> None:
            nonlocal rows
            handle.write(json.dumps(row) + "\n")
            rows += 1

        # One merged timeline, in the feed's own order: every inertial sample
        # older than an image precedes that image (the ordering invariant I2),
        # which the real capture inherits from the feed seam itself.
        texture = make_texture()
        t_imu = 0.0
        t_pair = 0.0
        imu_period = 1.0 / FPS_IMU
        pair_period = 1.0 / FPS_PAIR
        while t_pair <= DURATION or t_imu <= DURATION:
            if t_pair <= DURATION and (t_imu > DURATION or t_pair <= t_imu):
                y = body_y_position(t_pair)
                # the rig moves along body -y = optical +x, so the view slides
                # across the wall by -y*f/z pixels and the left crop starts at
                # CROP_BASE plus that travel
                u0 = CROP_BASE - FOCAL * y / DEPTH
                left_rgb = frame(texture, u0)
                # the right eye sits +baseline along optical x: every point
                # lands f*b/z px further LEFT in it, which for crops means the
                # right window starts f*b/z px further RIGHT (u_R = u_L + b_px)
                right_rgb = frame(texture, u0 + FOCAL * BASELINE / DEPTH)
                left = loc.grayscale_rgb8(left_rgb, WIDTH, HEIGHT)
                right = loc.grayscale_rgb8(right_rgb, WIDTH, HEIGHT)
                seq += 1
                lname = f"sensor-capture/frames/{seq:06d}-left.pgm"
                rname = f"sensor-capture/frames/{seq:06d}-right.pgm"
                header5 = f"P5\n{WIDTH} {HEIGHT}\n255\n".encode()
                (out / lname).write_bytes(header5 + left)
                (out / rname).write_bytes(header5 + right)
                write(
                    {
                        "kind": "pair",
                        "seq": seq,
                        "sim_time_ns": int(round(t_pair * 1e9)),
                        "capture_host_ns": 0,
                        "width": WIDTH,
                        "height": HEIGHT,
                        "left": lname,
                        "right": rname,
                        "luma_sha256": hashlib.sha256(left + right).hexdigest(),
                    }
                )
                t_pair += pair_period
            else:
                # body FRD, gravity -z; the estimator's frame map takes (x, -y, -z)
                gyro = (0.0, 0.0, 0.0)
                accel = (0.0, body_y_accel(t_imu), -GRAVITY)
                rows += 1
                write(
                    {
                        "kind": "imu",
                        "seq": rows,
                        "sim_time_ns": int(round(t_imu * 1e9)),
                        "capture_host_ns": 0,
                        "gyro": list(gyro),
                        "accel": list(accel),
                    }
                )
                t_imu += imu_period
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
