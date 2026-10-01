"""The per-run sensor capture: one flight's whole sensor stream on disk.

Purpose (2026-09-30, the replay surface): a scored run's retained artifacts cannot
reconstruct the estimator's input -- the ``pairs/`` scene capture keeps only a bounded
head of the stereo stream and no per-sample inertial data or feed ordering is retained
anywhere -- so every estimator change costs a flight to validate. This module records,
inert by default, the run's own sensor events under the run directory as a GENERAL
record other consumers can read: the stereo frames as the sensor delivered them (with
their simulator-time stamps), the inertial samples, the truth poses read for scoring,
and the setpoints and mode/arm commands the check itself issues.

The estimator replay is the first consumer, not the only one: the frames are plain P6
PPMs with per-row stamps, the index is newline-delimited JSON, and no field requires
this package to decode. Two properties make the record replayable to the estimator
BYTE EXACTLY rather than approximately:

1. every ``imu``/``pair`` row is written with exactly the arguments the feed encoders
   received, at the single seam (``localization_check.feed_record``) through which
   every estimator frame passes, in feed order;
2. every ``pair`` row carries the SHA-256 of the two luma planes actually handed to
   ``encode_stereo``, so a replay can verify that its reconstruction (PPM round-trip
   plus the same pure BT.601 conversion) is bit-identical instead of arguing it.

Truth isolation is structural, not procedural: the protocol the estimator speaks has
no field that can carry a pose, so ``pose`` rows can never be encoded into a frame.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["CAPTURE_ENV", "SensorCapture"]

#: The environment variable that turns the recorder on. Absent (or any value other
#: than ``"1"``) the recorder does not exist: no files are opened, no rows written,
#: and the flown path's behaviour and timing are untouched.
CAPTURE_ENV = "EMBODIED_SENSOR_CAPTURE"

_RECORDS_NAME = "sensor-capture/records.jsonl"
_HEADER_NAME = "sensor-capture/header.json"
_FRAMES_DIR = "sensor-capture/frames"
_FORMAT = "embodied-sensor-capture-1"


class SensorCapture:
    """Writes the per-run sensor record through the run's own evidence writer.

    All methods are called from the estimator feed's single-writer thread (sensor
    rows) or from the choreography's thread (setpoint/command rows); the writer's
    own lock keeps each row and each frame file atomic, and the sequence numbers
    order the record exactly as the run produced it.
    """

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._seq = 0
        writer.write_json(
            _HEADER_NAME,
            {
                "format": _FORMAT,
                "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "records": _RECORDS_NAME,
                "frames_dir": _FRAMES_DIR,
                "kinds": {
                    "imu": "one inertial sample: sim_time_ns, capture_host_ns, gyro[3], accel[3] -- the exact arguments of localization.encode_imu, in feed order",
                    "pair": "one stereo pair: sim_time_ns, capture_host_ns, width, height, left/right P6 PPM paths (rgb8 as the sensor delivered), luma_sha256 of the two planes localization.encode_stereo received, in feed order",
                    "pose": "one evaluator truth sample, read for scoring and sent nowhere: sim_time_ns, position_xyz (NED), attitude_rpy; structurally unencodable into the estimator's protocol",
                    "setpoint": "one commanded target the check issued: sim_time_ns (feed clock), wall_ns, target fields as commanded",
                    "command": "one mode/arm/takeoff/land/override event the check issued: sim_time_ns (feed clock), wall_ns, command name, detail",
                },
                "time_base": "sim_time_ns is simulator time in nanoseconds, the one clock the sensors share; capture_host_ns/wall_ns are host monotonic stamps for latency accounting only",
                "replay": "re-feed imu/pair rows in seq order through localization.encode_imu/encode_stereo (PPM payload -> grayscale_rgb8 -> encode_stereo; verify luma_sha256 first); never feed pose/setpoint/command rows to the estimator",
            },
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str], writer: Any) -> "SensorCapture | None":
        """The recorder, or ``None`` when the environment did not ask for one."""
        if env.get(CAPTURE_ENV) != "1":
            return None
        return cls(writer)

    # -- sensor rows (feed order, the estimator's exact inputs) -----------------

    def imu(
        self,
        sim_time_ns: int,
        capture_host_ns: int,
        gyro: Sequence[float],
        accel: Sequence[float],
    ) -> None:
        self._append(
            {
                "kind": "imu",
                "sim_time_ns": sim_time_ns,
                "capture_host_ns": capture_host_ns,
                "gyro": list(gyro),
                "accel": list(accel),
            }
        )

    def pair(
        self,
        sim_time_ns: int,
        capture_host_ns: int,
        width: int,
        height: int,
        left_rgb: bytes,
        right_rgb: bytes,
        left_luma: bytes,
        right_luma: bytes,
    ) -> None:
        self._seq += 1
        header = f"P6\n{width} {height}\n255\n".encode()
        left_name = f"{_FRAMES_DIR}/{self._seq:06d}-left.ppm"
        right_name = f"{_FRAMES_DIR}/{self._seq:06d}-right.ppm"
        self._writer.write_bytes(left_name, header + left_rgb)
        self._writer.write_bytes(right_name, header + right_rgb)
        self._append(
            {
                "kind": "pair",
                "sim_time_ns": sim_time_ns,
                "capture_host_ns": capture_host_ns,
                "width": width,
                "height": height,
                "left": left_name,
                "right": right_name,
                "luma_sha256": hashlib.sha256(left_luma + right_luma).hexdigest(),
            },
            counted=False,
        )

    def pose(
        self,
        sim_time_ns: int,
        position_xyz: Sequence[float],
        attitude_rpy: Sequence[float],
    ) -> None:
        self._append(
            {
                "kind": "pose",
                "sim_time_ns": sim_time_ns,
                "position_xyz": list(position_xyz),
                "attitude_rpy": list(attitude_rpy),
            }
        )

    # -- choreography rows (what the run commanded, on the feed's clock) --------

    def setpoint(self, sim_time_ns: int | None, target: Mapping[str, Any]) -> None:
        self._append(
            {
                "kind": "setpoint",
                "sim_time_ns": sim_time_ns,
                "wall_ns": time.monotonic_ns(),
                "target": dict(target),
            }
        )

    def command(self, name: str, sim_time_ns: int | None, detail: Any = None) -> None:
        row: dict[str, Any] = {
            "kind": "command",
            "command": name,
            "sim_time_ns": sim_time_ns,
            "wall_ns": time.monotonic_ns(),
        }
        if detail is not None:
            row["detail"] = detail
        self._append(row)

    # -- internals ---------------------------------------------------------------

    def _append(self, row: dict[str, Any], counted: bool = True) -> None:
        if counted:
            self._seq += 1
        row["seq"] = self._seq
        self._writer.append_jsonl(_RECORDS_NAME, row)
