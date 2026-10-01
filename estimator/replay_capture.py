#!/usr/bin/env python3
"""Replay one run's sensor capture into ov_stream as fast as the CPU allows.

The replay surface's driver (2026-09-30). No Webots, no SITL, no autopilot, no
wall-clock pacing: it reconstructs the estimator's input frames from a run's
``sensor-capture/`` record, starts a fresh ``ov_stream`` in its input-driven
publish mode (``OV_REPLAY_PUBLISH_EVERY_IMU``), and pushes the whole stream
through the socket with backpressure supplied only by the estimator's own
consumption -- ``sendall`` blocks when the estimator is still chewing, which is
exactly the CPU speed limit this tool exists to measure.

What is byte-exact and what is reconstructed, stated here because the tool is
the claim: every IMU frame re-encodes the very values recorded from the
encoder's arguments, and every stereo frame re-derives its luma planes from the
run's own rgb8 PPMs through the same pure BT.601 conversion the feed used --
then VERIFIES the result against the SHA-256 recorded at capture time, so a
silent reconstruction error is a hard failure, not a diff that quietly grows.
The frame ORDER is the recorded feed order (the recorder sits at the feed's
single seam), so the estimator sees the same bytes in the same order as the
flight, with one honest exception: arrival TIMES are meaningless here by
design, and nothing downstream may read them.

Truth never enters the replay: pose/setpoint/command rows are read for the
summary and then skipped -- the ov_stream protocol has no field that can carry
a pose, so the skip is structural, not disciplinary.

Usage:
  python3 estimator/replay_capture.py <run-dir> [--out FILE] [--every-imu N]
      [--binary PATH] [--port N] [--idle-exit-s S] [--log FILE]

<run-dir> is a run directory containing ``run-a/sensor-capture`` (or the
sensor-capture directory itself).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied.platform import localization as loc  # noqa: E402


def _find_capture(root: Path) -> Path:
    for candidate in (root, root / "run-a" / "sensor-capture", root / "sensor-capture"):
        if (candidate / "records.jsonl").exists():
            return candidate
    raise SystemExit(f"no sensor-capture/records.jsonl under {root}")


def _read_ppm_p6(path: Path, width: int, height: int) -> bytes:
    """The rgb8 payload of one P6 PPM exactly as the capture wrote it."""
    data = path.read_bytes()
    header = f"P6\n{width} {height}\n255\n".encode()
    if not data.startswith(header) or len(data) != len(header) + width * height * 3:
        raise SystemExit(f"{path}: not the P6 {width}x{height} image the capture recorded")
    return data[len(header) :]


def iter_frames(records_path: Path) -> Iterator[tuple[bytes, dict[str, Any]]]:
    """Yield (wire frame, row) in recorded feed order, verifying luma hashes."""
    for line in records_path.read_text().splitlines():
        row = json.loads(line)
        kind = row["kind"]
        if kind == "imu":
            yield loc.encode_imu(row["sim_time_ns"], row["gyro"], row["accel"]), row
        elif kind == "pair":
            base = records_path.parent.parent  # rows carry run-dir-relative paths
            left_rgb = _read_ppm_p6(base / row["left"], row["width"], row["height"])
            right_rgb = _read_ppm_p6(base / row["right"], row["width"], row["height"])
            left = loc.grayscale_rgb8(left_rgb, row["width"], row["height"])
            right = loc.grayscale_rgb8(right_rgb, row["width"], row["height"])
            digest = hashlib.sha256(left + right).hexdigest()
            if digest != row["luma_sha256"]:
                raise SystemExit(
                    f"row {row['seq']}: reconstructed luma hash {digest} does not match "
                    f"the recorded {row['luma_sha256']} -- the capture cannot be replayed "
                    "byte-exactly and this run must not be diffed"
                )
            yield (
                loc.encode_stereo(
                    row["sim_time_ns"], left, right, row["width"], row["height"]
                ),
                row,
            )
        elif kind == "capture_drop":
            raise SystemExit(
                "the capture contains capture_drop rows: the feed outran the "
                "recorder and records are missing, so this capture is not a "
                "faithful input stream and must not be diffed"
            )
        else:
            yield b"", row  # pose/setpoint/command: read, never fed


def _wait_listening(port: int, process: subprocess.Popen, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.connect(("127.0.0.1", port))
                return
            except OSError:
                pass
        if process.poll() is not None:
            raise SystemExit(f"ov_stream exited early with {process.returncode}")
        time.sleep(0.1)
    raise SystemExit(f"ov_stream did not listen on {port} within {timeout_s:.0f}s")


def _next_frame(buffer: bytes) -> tuple[bytes | None, bytes]:
    if len(buffer) < loc.FRAME_HEADER_SIZE:
        return None, buffer
    magic, kind, _flags, length = loc.FRAME_HEADER.unpack_from(buffer)
    if magic != loc.FRAME_MAGIC:
        raise loc.ProtocolError(f"frame magic 0x{magic:04x} is not the ov_stream magic")
    if kind != loc.KIND_STATE:
        raise loc.ProtocolError(f"frame kind {kind} is not a STATE frame")
    total = loc.FRAME_HEADER_SIZE + length
    if len(buffer) < total:
        return None, buffer
    return buffer[:total], buffer[total:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="replay states jsonl")
    parser.add_argument("--every-imu", type=int, default=5, help="publish per N imu samples")
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "estimator" / "ov_stream")
    parser.add_argument("--port", type=int, default=9420)
    parser.add_argument("--idle-exit-s", type=float, default=3.0)
    parser.add_argument("--log", type=Path, default=None, help="ov_stream stdout log")
    args = parser.parse_args()

    capture_dir = _find_capture(args.run_dir)
    records_path = capture_dir / "records.jsonl"
    out_path = args.out or capture_dir.parent / "replay-states.jsonl"
    log_path = args.log or capture_dir.parent / "replay-estimator.log"

    env = dict(os.environ)
    env["OV_REPLAY_PUBLISH_EVERY_IMU"] = str(args.every_imu)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(args.binary), str(args.port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    try:
        _wait_listening(args.port, process)

        sock = socket.create_connection(("127.0.0.1", args.port))
        sock.settimeout(None)

        states: list[dict[str, Any]] = []
        first_state_wall: list[float] = []
        last_state_wall: list[float] = []
        send_done = threading.Event()

        def reader() -> None:
            buffer = b""
            try:
                while True:
                    readable, _, _ = select.select([sock], [], [], args.idle_exit_s)
                    if not readable:
                        if send_done.is_set():
                            break
                        continue
                    data = sock.recv(1 << 20)
                    if not data:
                        break
                    buffer += data
                    while True:
                        frame, buffer = _next_frame(buffer)
                        if frame is None:
                            break
                        state = loc.decode_state(frame)
                        now = time.monotonic()
                        if not first_state_wall:
                            first_state_wall.append(now)
                        last_state_wall[:] = [now]
                        states.append(
                            {
                                "index": len(states),
                                "wall_monotonic_s": now,
                                **{
                                    key: (list(value) if isinstance(value, tuple) else value)
                                    for key, value in asdict(state).items()
                                },
                            }
                        )
            except OSError:
                pass

        def sender() -> None:
            try:
                fed = {"imu": 0, "pair": 0, "skipped": 0}
                first_sim: list[int] = []
                last_sim = 0
                started = time.monotonic()
                for frame, row in iter_frames(records_path):
                    if row["kind"] in ("imu", "pair"):
                        if not first_sim:
                            first_sim.append(row["sim_time_ns"])
                        last_sim = row["sim_time_ns"]
                        fed[row["kind"]] += 1
                        sock.sendall(frame)
                    else:
                        fed["skipped"] += 1
                sender.started_wall = started
                sender.first_sim_ns = first_sim[0] if first_sim else 0
                sender.last_sim_ns = last_sim
                sender.fed = fed
                sender.finished_wall = time.monotonic()
            except BaseException as error:  # surfaced by the main thread below
                sender.error = error
            finally:
                send_done.set()

        reader_thread = threading.Thread(target=reader, name="replay-reader", daemon=True)
        reader_thread.start()
        sender_thread = threading.Thread(target=sender, name="replay-sender", daemon=True)
        sender_thread.start()
        sender_thread.join()
        reader_thread.join(timeout=args.idle_exit_s * 2 + 5.0)

        sender_error = getattr(sender, "error", None)
        if sender_error is not None:
            raise SystemExit(f"the replay feed failed: {sender_error}")
        sock.close()

        with out_path.open("w", encoding="utf-8") as handle:
            for state in states:
                handle.write(json.dumps(state) + "\n")

        fed = sender.fed  # type: ignore[attr-defined]
        sim_span_s = (sender.last_sim_ns - sender.first_sim_ns) / 1e9  # type: ignore[attr-defined]
        if states:
            wall_span_s = last_state_wall[0] - sender.started_wall
        else:
            print(
                f"NO STATES: the estimator never initialized on this capture (fed {fed}); "
                f"see {log_path}"
            )
            wall_span_s = sender.finished_wall - sender.started_wall
        summary = {
            "capture": str(records_path),
            "fed": fed,
            "states_received": len(states),
            "first_sim_ns": sender.first_sim_ns,
            "last_sim_ns": sender.last_sim_ns,
            "sim_span_s": round(sim_span_s, 3),
            "wall_span_s": round(wall_span_s, 3),
            "sim_per_wall": round(sim_span_s / wall_span_s, 2) if wall_span_s > 0 else None,
            "every_imu": args.every_imu,
            "port": args.port,
        }
        print(json.dumps(summary, indent=1))
        (capture_dir.parent / "replay-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        return 0 if states else 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
