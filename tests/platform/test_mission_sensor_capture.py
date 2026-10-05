"""The mission's feed seam records the replayable sensor capture (env-gated).

ESTIMATOR-FEJ-ZUPT.md's measured debt: no replayable captures exist anywhere —
the recorder hook lives only in ``localization_check`` — so the estimator A/B
lane (resumption chi-square, bias-RW sigmas) is blocked on real captures. The
mission's feed seam now feeds the SAME recorder (``SensorCapture``, one format,
the rows the replay surface already consumes) at ITS seam, inert unless
EMBODIED_SENSOR_CAPTURE=1: unset, no recorder exists, no files are opened and
the flown path is untouched.

The tests drive ``_capture_sensor_record`` — the exact function
``feed_record`` calls, once per record in feed order — on synthetic records of
the wire's own payload shapes, against a real ``EvidenceWriter`` in tmp_path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from embodied.platform import mission_runtime as MR
from embodied.platform import sensor_capture

HOST, CLOCK = "capture-test-0", "monotonic"


def _writer(tmp_path: Path):
    from embodied.platform.webots_ardupilot import EvidenceWriter

    return EvidenceWriter(tmp_path, "run-a")


def _record(kind, sim_time_s: float, *, pair=None, imu=None, pose=None):
    from embodied.contracts.records import ClockStamp
    from embodied.platform.webots_ardupilot import SensorRecord

    return SensorRecord(
        kind=kind,
        sim_time_s=sim_time_s,
        sequence=0,
        flags=0,
        received_stamp=ClockStamp(host_id=HOST, clock_id=CLOCK, monotonic_ns=0),
        pair=pair,
        imu=imu,
        pose=pose,
    )


def _pair(width: int = 4, height: int = 2):
    from embodied.platform.webots_ardupilot import PairPayload

    rgb = bytes(range(width * height * 3))
    return PairPayload(
        capture_host_ns=111,
        pair_id=1,
        left_frame_id=1,
        right_frame_id=2,
        width=width,
        height=height,
        encoding="rgb8",
        left_bytes=rgb,
        right_bytes=bytes(reversed(rgb)),
    )


def _imu():
    from embodied.platform.webots_ardupilot import ImuPayload

    return ImuPayload(
        capture_host_ns=222,
        accelerometer=(0.1, 0.2, 9.81),
        gyro=(0.01, 0.02, 0.03),
        inertial_unit_rpy=(0.0, 0.0, 0.0),
        device_names=("imu", "imu", "imu"),
        units="m/s^2",
    )


def _pose():
    from embodied.platform.webots_ardupilot import PosePayload

    return PosePayload(
        capture_host_ns=333,
        position_xyz=(1.0, 2.0, -1.5),
        attitude_rpy=(0.0, 0.0, 0.1),
    )


def _synthetic_feed():
    """Inertial samples, then the frame they precede, then one truth sample."""
    from embodied.platform.webots_ardupilot import Kind

    return [
        _record(Kind.IMU, 0.0, imu=_imu()),
        _record(Kind.IMU, 0.004, imu=_imu()),
        _record(Kind.PAIR, 0.004, pair=_pair()),
        _record(Kind.POSE, 0.004, pose=_pose()),
    ]


def _feed_through(capture, records) -> None:
    """The way feed_record drives the seam: convert planes for pairs, then record."""
    from embodied.platform import localization as loc
    from embodied.platform.webots_ardupilot import Kind

    for record in records:
        left_luma = right_luma = None
        if record.kind is Kind.PAIR and record.pair is not None:
            left_luma = loc.grayscale_rgb8(record.pair.left_bytes, 4, 2)
            right_luma = loc.grayscale_rgb8(record.pair.right_bytes, 4, 2)
        MR._capture_sensor_record(
            capture,
            record,
            width=4,
            height=2,
            left_luma=left_luma,
            right_luma=right_luma,
        )


def _rows(capture_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (capture_dir / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Env set: the capture appears, in the replay surface's own format.
# ---------------------------------------------------------------------------


def test_capture_on_records_the_expected_kind_sequence(tmp_path, monkeypatch):
    from embodied.platform import localization as loc

    monkeypatch.setenv(sensor_capture.CAPTURE_ENV, "1")
    writer = _writer(tmp_path)
    capture = sensor_capture.SensorCapture.from_env(os.environ, writer)
    assert capture is not None
    _feed_through(capture, _synthetic_feed())
    capture.close()

    capture_dir = writer.directory / "sensor-capture"
    header = json.loads((capture_dir / "header.json").read_text())
    assert header["format"] == "embodied-sensor-capture-1"

    rows = _rows(capture_dir)
    # The feed's own order — inertial, inertial, frame, truth — one seq per row.
    assert [row["kind"] for row in rows] == ["imu", "imu", "pair", "pose"]
    assert [row["seq"] for row in rows] == [1, 2, 3, 4]

    imu_row = rows[0]
    assert imu_row["sim_time_ns"] == 0
    assert imu_row["capture_host_ns"] == 222
    assert imu_row["gyro"] == [0.01, 0.02, 0.03]
    assert imu_row["accel"] == [0.1, 0.2, 9.81]

    pair_row = rows[2]
    assert pair_row["sim_time_ns"] == 4_000_000
    assert pair_row["capture_host_ns"] == 111
    assert pair_row["width"] == 4 and pair_row["height"] == 2
    # The frames ARE the luma planes the encoder received: P5 payloads whose
    # bytes hash to the row's recorded digest (the replay's bit-identity check).
    # Row frame paths are relative to the evidence writer's own directory.
    left = loc.grayscale_rgb8(_pair().left_bytes, 4, 2)
    right = loc.grayscale_rgb8(_pair().right_bytes, 4, 2)
    assert (
        writer.directory / pair_row["left"]
    ).read_bytes() == b"P5\n4 2\n255\n" + left
    assert (
        writer.directory / pair_row["right"]
    ).read_bytes() == b"P5\n4 2\n255\n" + right
    assert pair_row["luma_sha256"] == hashlib.sha256(left + right).hexdigest()

    pose_row = rows[3]
    # Truth rides along for offline scoring, structurally unencodable into the
    # estimator's protocol — never an estimator input.
    assert pose_row["position_xyz"] == [1.0, 2.0, -1.5]
    assert pose_row["attitude_rpy"] == [0.0, 0.0, 0.1]


# ---------------------------------------------------------------------------
# Env unset: no recorder, no files, no behavior change.
# ---------------------------------------------------------------------------


def test_capture_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(sensor_capture.CAPTURE_ENV, raising=False)
    writer = _writer(tmp_path)
    assert sensor_capture.SensorCapture.from_env(os.environ, writer) is None
    # The seam call the flown path makes every record: with no recorder it is
    # a no-op — nothing written, nothing raised.
    _feed_through(None, _synthetic_feed())
    assert not (writer.directory / "sensor-capture").exists()


def test_a_value_other_than_one_disables_the_capture(tmp_path, monkeypatch):
    monkeypatch.setenv(sensor_capture.CAPTURE_ENV, "true")
    writer = _writer(tmp_path)
    assert sensor_capture.SensorCapture.from_env(os.environ, writer) is None
    assert not (writer.directory / "sensor-capture").exists()
