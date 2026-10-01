"""The per-run sensor capture: one flight's whole sensor stream on disk.

Purpose (2026-09-30, the replay surface): a scored run's retained artifacts cannot
reconstruct the estimator's input -- the ``pairs/`` scene capture keeps only a bounded
head of the stereo stream and no per-sample inertial data or feed ordering is retained
anywhere -- so every estimator change costs a flight to validate. This module records,
inert by default, the run's own sensor events under the run directory as a GENERAL
record other consumers can read: the stereo frames with their simulator-time stamps,
the inertial samples, the truth poses read for scoring, and the setpoints and
mode/arm commands the check itself issues.

The estimator replay is the first consumer, not the only one: the frames are plain
PGMs with per-row stamps, the index is newline-delimited JSON, and no field requires
this package to decode. Two properties make the record replayable to the estimator
BYTE EXACTLY rather than approximately:

1. every ``imu``/``pair`` row is written with exactly the arguments the feed encoders
   received, at the single seam (``localization_check.feed_record``) through which
   every estimator frame passes, in feed order;
2. every ``pair`` row carries the SHA-256 of the two luma planes actually handed to
   ``encode_stereo``, and the frames ARE those planes (P5 PGM, one channel), so a
   replay verifies bit-identity against the bytes themselves instead of arguing it.

The frames are the estimator's own grayscale input, not the sensor stream's rgb8.
That is a measured decision, not a preference: at the rgb8 rate the recorder's writer
thread could not keep up (flight p01l-replay-ref3-20261001T035216Z recorded 21
capture_drop rows at 1.84 MB per pair), while the luma planes are a third of the
bytes and are the exact quantity the estimator consumes. Consumers wanting colour
take the run's bounded ``pairs/`` scene capture, which the run already writes
independently of this recorder.

Truth isolation is structural, not procedural: the protocol the estimator speaks has
no field that can carry a pose, so ``pose`` rows can never be encoded into a frame.

Threading (measured, flights p01l-replay-ref/-ref2-20261001T0345/0347): the recorder
first wrote its PPMs and rows on the feed's single-writer thread, and those disk
writes -- 1.84 MB per pair -- delayed the inertial stream enough to reproduce the
known stall shape (published-state age p95 48 ms, max 112 ms, against the ~11/30 ms
clamped baseline; both flights lost the route mid-hold). The fence says the recorder
must not alter WHEN the adapter sends, so the feed thread now only enqueues
references into a bounded queue -- a no-block put that drops (and counts) rather
than waits -- and one dedicated writer thread does every byte of I/O, hashing
included. Drops become their own rows so a holed capture is visible, and the replay
driver refuses to replay one.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import queue
import threading
import time
from datetime import datetime, timezone
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
#: How many records may wait for the writer thread before the feed thread drops.
#: Measured: at the declared rates this writer occasionally stalls for up to ~0.5 s
#: on a memory-pressured host (dirty-page writeback), and a 256-deep queue (0.5 s of
#: inertial samples) dropped 23 rows over a 60 s soak. 1024 records -- two seconds
#: of the feed's worst burst many times over -- absorbs that; references only, so
#: the memory cost is the payloads' own size, and IMU rows dominate the count.
_QUEUE_DEPTH = 1024


class SensorCapture:
    """Records the per-run sensor stream through the run's own evidence writer.

    Every public method is safe to call from the estimator feed's single-writer
    thread: each builds a small tuple of already-existing objects, offers it to a
    bounded queue, and returns without ever touching a disk. A dedicated writer
    thread drains the queue in order and performs all file writes and hashing.
    """

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._seq = 0
        self._drops = 0
        self._queue: "queue.Queue[tuple[str, dict[str, Any]] | None]" = queue.Queue(
            maxsize=_QUEUE_DEPTH
        )
        self._stopped = threading.Event()
        self._drained = threading.Event()
        # Direct file handles, not EvidenceWriter's per-call path()/mkdir()/open():
        # measured on flights ref3/ref4, re-opening the index and re-statting its
        # directory for each of the 500 Hz rows is ~1500 syscalls/s under the
        # writer's RLock, which starved this thread until the queue overflowed
        # (21 and 25 capture_drop rows) and churned the GIL enough to lift the
        # feed's published-state ages (p95 32 ms against the ~11 ms baseline).
        # One persistent append handle and one pre-made frames directory reduce
        # the steady state to one write() per flushed batch.
        self._base_dir = writer.directory / "sensor-capture"
        self._frames_dir = self._base_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._records_handle = (self._base_dir / "records.jsonl").open("a", encoding="utf-8")
        (self._base_dir / "header.json").write_text(
            json.dumps(self._header_document(), indent=1, default=str) + "\n",
            encoding="utf-8",
        )
        self._thread = threading.Thread(
            target=self._write_loop, name="sensor-capture-writer", daemon=True
        )
        self._thread.start()
        atexit.register(self.close)

    def _header_document(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "records": _RECORDS_NAME,
            "frames_dir": _FRAMES_DIR,
            "queue_depth": _QUEUE_DEPTH,
            "kinds": {
                "imu": "one inertial sample: sim_time_ns, capture_host_ns, gyro[3], accel[3] -- the exact arguments of localization.encode_imu, in feed order",
                "pair": "one stereo pair: sim_time_ns, capture_host_ns, width, height, left/right P5 PGM paths carrying the two luma planes exactly as localization.encode_stereo received them, and luma_sha256 over those two planes, in feed order; rgb8 is not retained (see module docstring) -- colour consumers read the run's bounded pairs/ scene capture",
                "pose": "one evaluator truth sample, read for scoring and sent nowhere: sim_time_ns, position_xyz (NED), attitude_rpy; structurally unencodable into the estimator's protocol",
                "setpoint": "one commanded target the check issued: sim_time_ns (feed clock), wall_ns, target fields as commanded",
                "command": "one mode/arm/takeoff/land/override event the check issued: sim_time_ns (feed clock), wall_ns, command name, detail",
                "capture_drop": "the feed thread offered a record while the queue was full; that record is missing, and the capture must not be replayed as complete",
            },
            "time_base": "sim_time_ns is simulator time in nanoseconds, the one clock the sensors share; capture_host_ns/wall_ns are host monotonic stamps for latency accounting only",
            "replay": "re-feed imu/pair rows in seq order through localization.encode_imu/encode_stereo (P5 payload IS the luma plane; verify luma_sha256 first); refuse a capture containing capture_drop rows; never feed pose/setpoint/command rows to the estimator",
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str], writer: Any) -> "SensorCapture | None":
        """The recorder, or ``None`` when the environment did not ask for one."""
        if env.get(CAPTURE_ENV) != "1":
            return None
        return cls(writer)

    # -- feed-thread API: enqueue only, never block -------------------------------

    def imu(
        self,
        sim_time_ns: int,
        capture_host_ns: int,
        gyro: Sequence[float],
        accel: Sequence[float],
    ) -> None:
        self._offer(
            "row",
            {
                "kind": "imu",
                "sim_time_ns": sim_time_ns,
                "capture_host_ns": capture_host_ns,
                "gyro": list(gyro),
                "accel": list(accel),
            },
        )

    def pair(
        self,
        sim_time_ns: int,
        capture_host_ns: int,
        width: int,
        height: int,
        left_luma: bytes,
        right_luma: bytes,
    ) -> None:
        # References only: the caller's plane bytes already exist (they are what
        # encode_stereo consumed), so nothing is copied on the feed thread.
        self._offer(
            "pair",
            {
                "sim_time_ns": sim_time_ns,
                "capture_host_ns": capture_host_ns,
                "width": width,
                "height": height,
                "left_luma": left_luma,
                "right_luma": right_luma,
            },
        )

    def pose(
        self,
        sim_time_ns: int,
        position_xyz: Sequence[float],
        attitude_rpy: Sequence[float],
    ) -> None:
        self._offer(
            "row",
            {
                "kind": "pose",
                "sim_time_ns": sim_time_ns,
                "position_xyz": list(position_xyz),
                "attitude_rpy": list(attitude_rpy),
            },
        )

    def setpoint(self, sim_time_ns: int | None, target: Mapping[str, Any]) -> None:
        self._offer(
            "row",
            {
                "kind": "setpoint",
                "sim_time_ns": sim_time_ns,
                "wall_ns": time.monotonic_ns(),
                "target": dict(target),
            },
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
        self._offer("row", row)

    # -- lifecycle -----------------------------------------------------------------

    def close(self, timeout_s: float = 60.0) -> None:
        """Stop accepting records and wait for the writer to drain what it holds."""
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._queue.put(None)
        self._drained.wait(timeout_s)

    def _offer(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            self._drops += 1

    def _write_loop(self) -> None:
        seen_drops = 0
        while True:
            pending: list[str] = []
            item = self._queue.get()
            batch = [] if item is None else [item]
            stop = item is None
            # Drain everything already queued (the queue is the burst absorber),
            # then write it as one handle write: fewer syscalls and fewer GIL
            # handoffs than one open/write/close per row, which is what starved
            # this thread on flights ref3/ref4. The stop sentinel may arrive
            # inside this drain, so it is detected here and not only from the
            # blocking get: missing it would leave the flush unreached.
            while True:
                try:
                    entry = self._queue.get_nowait()
                except queue.Empty:
                    break
                if entry is None:
                    stop = True
                else:
                    batch.append(entry)
            for entry in batch:
                if entry is None:
                    continue
                kind, payload = entry
                if self._drops != seen_drops:
                    delta = self._drops - seen_drops
                    seen_drops = self._drops
                    pending.append(
                        self._line(
                            {
                                "kind": "capture_drop",
                                "dropped": delta,
                                "total_dropped": self._drops,
                                "wall_ns": time.monotonic_ns(),
                            }
                        )
                    )
                if kind == "pair":
                    pending.append(self._write_pair(payload))
                else:
                    pending.append(self._line(payload))
            if pending:
                self._records_handle.write("".join(pending))
            if stop:
                self._records_handle.flush()
                self._drained.set()
                return

    def _write_pair(self, payload: dict[str, Any]) -> str:
        self._seq += 1
        header = f"P5\n{payload['width']} {payload['height']}\n255\n".encode()
        left_name = f"{_FRAMES_DIR}/{self._seq:06d}-left.pgm"
        right_name = f"{_FRAMES_DIR}/{self._seq:06d}-right.pgm"
        (self._base_dir.parent / left_name).write_bytes(header + payload["left_luma"])
        (self._base_dir.parent / right_name).write_bytes(header + payload["right_luma"])
        return self._line(
            {
                "kind": "pair",
                "sim_time_ns": payload["sim_time_ns"],
                "capture_host_ns": payload["capture_host_ns"],
                "width": payload["width"],
                "height": payload["height"],
                "left": left_name,
                "right": right_name,
                "luma_sha256": hashlib.sha256(
                    payload["left_luma"] + payload["right_luma"]
                ).hexdigest(),
            },
            counted=False,
        )

    def _line(self, row: dict[str, Any], counted: bool = True) -> str:
        if counted:
            self._seq += 1
        row["seq"] = self._seq
        return json.dumps(row) + "\n"
