"""T4: time-alignment behaviours (specification section 4.1, plan section 8).

Capture and receipt are separate stamps in one clock domain; simulator time
rides beside them and never folds into them; the calibration's camera/IMU
offset is available only as the (offset, error) pair; and no API in the adapter
surface turns an arrival time into a capture time.
"""

import pytest

from embodied.contracts import records as R
from embodied.perception import camera as C
from embodied.platform import sensors as S


def _stamp(ns: int, host: str = "host-1", clock: str = "monotonic") -> R.ClockStamp:
    return S.stamp(host, clock, ns)


def test_capture_precedes_receipt_in_one_domain():
    sample = S.SensorSample(
        value="frame",
        capture_stamp=_stamp(1_000_000_000),
        receipt_stamp=_stamp(1_000_050_000),
        sim_time_s=1.102,
    )
    assert S.capture_latency_ns(sample) == 50_000


def test_receipt_before_capture_refused():
    with pytest.raises(R.RecordError):
        S.SensorSample(
            value=1,
            capture_stamp=_stamp(200),
            receipt_stamp=_stamp(100),
            sim_time_s=None,
        )


def test_cross_domain_subtraction_refused():
    with pytest.raises(R.ClockDomainError):
        S.SensorSample(
            value=1,
            capture_stamp=_stamp(100, host="controller"),
            receipt_stamp=_stamp(150, host="reader"),
            sim_time_s=None,
        )
    with pytest.raises(R.ClockDomainError):
        R.elapsed_ns(_stamp(0, clock="monotonic"), _stamp(1, clock="simulator"))


def test_sim_time_rides_beside_the_stamps():
    stamp_ns = 1_000_000_000
    sample = S.SensorSample(
        value=2,
        capture_stamp=_stamp(stamp_ns),
        receipt_stamp=_stamp(stamp_ns + 5),
        sim_time_s=0.004,
    )
    assert sample.capture_stamp.monotonic_ns == stamp_ns  # sim time never folded in
    assert sample.sim_time_s == 0.004
    # a sensor with no simulator clock: missing stays missing, never zero
    without_sim = S.SensorSample(
        value=2, capture_stamp=_stamp(stamp_ns), receipt_stamp=_stamp(stamp_ns), sim_time_s=None
    )
    assert without_sim.sim_time_s is None


def test_calibration_offset_round_trips_with_its_error():
    calibration = C.build_calibration(time_offset_s=0.0, time_offset_error_s=0.0607)
    assert S.device_time_offset(calibration) == (0.0, 0.0607)
    parsed = R.from_dict(R.Calibration, R.to_dict(calibration))
    assert parsed == calibration
    assert S.device_time_offset(parsed) == (0.0, 0.0607)


def test_offset_without_error_refused():
    with pytest.raises(R.RecordError):
        C.build_calibration(time_offset_s=0.0, time_offset_error_s=None)
    with pytest.raises(R.RecordError):
        S.device_time_offset(C.build_calibration())


def test_arrival_is_never_a_capture_time():
    # shape-level guard: a sample cannot exist without an explicit capture stamp
    with pytest.raises(TypeError):
        S.SensorSample(value=1, receipt_stamp=_stamp(150), sim_time_s=None)  # type: ignore[call-arg]
    # two samples with the same receipt but different capture instants keep
    # their own latencies; nothing derives one capture from a receipt
    early = S.SensorSample(
        value=1, capture_stamp=_stamp(100), receipt_stamp=_stamp(150), sim_time_s=None
    )
    late = S.SensorSample(
        value=1, capture_stamp=_stamp(140), receipt_stamp=_stamp(150), sim_time_s=None
    )
    assert S.capture_latency_ns(early) != S.capture_latency_ns(late)
