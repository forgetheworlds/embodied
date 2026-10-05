"""The run's end state carries the attitude it ended in.

A run that came to rest inverted used to read exactly like one that landed
upright: same mode, the same armed flag, a plausible position. That is how a
vertical-error measurement came to be taken on a crashed, inverted aircraft and
read as an estimator property — the telemetry sample had carried the attitude
all along and nothing wrote it down.
"""

from __future__ import annotations

import pytest

from embodied.platform import webots_ardupilot as W
from embodied.platform.mission_runtime import _end_state_record


def _sample(*, attitude_rpy, local_position_ned):
    return W.TelemetrySample(
        received_stamp=W.ClockStamp(host_id="h", clock_id="monotonic", monotonic_ns=1),
        messages_seen=1,
        heartbeats=1,
        mode_name="LAND",
        custom_mode=9,
        armed=False,
        system_status=3,
        attitude_rpy=attitude_rpy,
        local_position_ned=local_position_ned,
        velocity_ned=None,
        servo_outputs=None,
        ekf_flags=None,
        ekf_velocity_variance=None,
        ekf_pos_horiz_variance=None,
        statustexts=(),
        boot_time_ms=None,
        home_position=None,
        autopilot_version=None,
        parameters={},
    )


def test_an_inverted_end_state_is_visible_in_the_record() -> None:
    """A vehicle that ended on its back must not read like one that landed level.

    The telemetry sample carries radians (ATTITUDE's own fields); the record
    converts to degrees, the bench-side convention ``live_record.end_state_tilt_deg``
    reads — ROLL-DEPARTURE.md: J58's radians triple stored raw was measured as a
    3-degree tilt and the inversion never fired on the live record.
    """
    record = _end_state_record(
        _sample(attitude_rpy=(3.13, 0.02, 0.50), local_position_ned=(0.2, 0.1, -0.1))
    )
    assert record["mode"] == "LAND"
    assert record["armed"] is False
    assert record["local_position_ned"] == [0.2, 0.1, -0.1]
    # 3.13 rad is ~179.3 deg: on its back, and now legible as that.
    assert record["attitude_rpy"] == pytest.approx([179.336, 1.146, 28.648], abs=0.01)


def test_the_converted_end_state_satisfies_the_bench_side_tilt_rule() -> None:
    """The same record, through the rule that judges it: inverted is a violation."""
    from embodied.bench import live_record

    record = _end_state_record(
        _sample(attitude_rpy=(3.13, 0.02, 0.50), local_position_ned=(0.2, 0.1, -0.1))
    )
    assert live_record.end_state_violation(record) is not None
    level = _end_state_record(
        _sample(attitude_rpy=(0.01, -0.01, 0.5), local_position_ned=(0.2, 0.1, -0.1))
    )
    assert live_record.end_state_violation(level) is None


def test_an_attitude_never_seen_is_missing_rather_than_a_default() -> None:
    """The sample never invents a value it has not seen, and neither may the record."""
    record = _end_state_record(_sample(attitude_rpy=None, local_position_ned=None))
    assert record["attitude_rpy"] is None
    assert record["local_position_ned"] is None
