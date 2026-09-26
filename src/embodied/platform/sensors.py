"""Capture-time identity for the stereo/IMU adapter surface.

Specification section 4.1 in module form. The one rule everything here serves:
**a callback arrival time is not automatically a capture time.** Every sample
therefore carries its two stamps as separate, explicitly named fields, and no
function in this module will derive one from the other. A consumer that wants a
capture instant must have been given one.

The stamps are :class:`embodied.contracts.records.ClockStamp` values, so all
subtraction goes through ``elapsed_ns`` and refuses cross-clock-domain operands.
Simulator time is carried beside the stamps, never folded into them: simulated
physics may run faster, slower or stop, so a sim-time number is not comparable
with a monotonic one merely because both are numbers.

The calibration's camera/IMU time offset is read only as the (offset, error)
pair the record stores. The frozen record refuses an offset without an error;
this module refuses to *use* an offset whose pair is missing, so a consumer can
never receive a silent 0.0 in place of a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied.contracts import records as R
from embodied.contracts.records import ClockDomainError, RecordError

__all__ = [
    "ClockDomainError",
    "SensorSample",
    "capture_latency_ns",
    "device_time_offset",
    "same_domain",
    "sim_time_ns",
    "stamp",
]


def stamp(host_id: str, clock_id: str, monotonic_ns: int) -> R.ClockStamp:
    """Build a ClockStamp; a thin named constructor over the frozen record."""
    return R.ClockStamp(host_id=host_id, clock_id=clock_id, monotonic_ns=monotonic_ns)


def same_domain(first: R.ClockStamp, second: R.ClockStamp) -> bool:
    """Whether two stamps may be subtracted at all."""
    return R.same_clock(first, second)


@dataclass(frozen=True)
class SensorSample:
    """One measurement with its capture identity kept separate from its arrival.

    ``capture_stamp`` names when the measurement is of; ``receipt_stamp`` names
    when its bytes became available to this host. Both are required arguments —
    there is no constructor that fills a missing capture from the receipt, which
    is the shape-level guard against the arrival-time mistake. ``sim_time_s``
    rides beside them (``None`` where the sensor has no simulator clock; missing
    stays missing, never zero).
    """

    value: Any
    capture_stamp: R.ClockStamp
    receipt_stamp: R.ClockStamp
    sim_time_s: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.capture_stamp, R.ClockStamp):
            raise RecordError("capture_stamp must be a ClockStamp")
        if not isinstance(self.receipt_stamp, R.ClockStamp):
            raise RecordError("receipt_stamp must be a ClockStamp")
        if not same_domain(self.capture_stamp, self.receipt_stamp):
            raise ClockDomainError(
                "capture and receipt stamps of one sample must share a clock domain: "
                f"{self.capture_stamp.host_id}/{self.capture_stamp.clock_id} and "
                f"{self.receipt_stamp.host_id}/{self.receipt_stamp.clock_id}"
            )
        if R.elapsed_ns(self.capture_stamp, self.receipt_stamp) < 0:
            raise RecordError("a sample cannot be received before it was captured")
        if self.sim_time_s is not None:
            if isinstance(self.sim_time_s, bool) or not isinstance(self.sim_time_s, (int, float)):
                raise RecordError("sim_time_s must be a number or None")
            if self.sim_time_s != self.sim_time_s or self.sim_time_s in (float("inf"), float("-inf")):
                raise RecordError("sim_time_s must be finite")


def capture_latency_ns(sample: SensorSample) -> int:
    """Nanoseconds from capture to receipt, inside the sample's one clock domain."""
    return R.elapsed_ns(sample.capture_stamp, sample.receipt_stamp)


def sim_time_ns(sim_time_s: float) -> int:
    """The estimator feed's timestamp: simulator time in nanoseconds.

    Sim time is the one clock the declared cameras and IMU share, so it is the
    one clock the estimator is given (plan section 4.1). A sample whose
    ``sim_time_s`` is missing has no estimator timestamp: passing ``None``
    here is a refused substitution, not a zero.
    """
    finite = sim_time_s is not None and sim_time_s == sim_time_s
    if not finite or sim_time_s in (float("inf"), float("-inf")):
        raise RecordError("a sample without a finite sim time has no estimator timestamp")
    return int(round(sim_time_s * 1_000_000_000))


def device_time_offset(calibration: R.Calibration) -> tuple[float, float]:
    """The calibration's (time_offset_s, time_offset_error_s) pair.

    Raises rather than returning a default: an offset whose measured error is
    absent is missing information, and a consumer converting stamps through the
    relation must inherit the error bar with the offset.
    """
    if calibration.time_offset_s is None or calibration.time_offset_error_s is None:
        raise RecordError(
            f"calibration {calibration.calibration_id} carries no measured time "
            "offset pair; refusing to substitute a default"
        )
    return (calibration.time_offset_s, calibration.time_offset_error_s)
