"""A crash-disarmed run never reads mission_completed, and the readback race never
hides an inverted rest.

ROLL-DEPARTURE.md (grading lane): J56, J57 and J58 each recorded a crash-disarm and
each receipt still read ``mission_completed``; and J58's live record missed its own
``end_state_inverted`` because the end-state sample raced the disarm readback (and
carried the attitude in radians against a degrees rule). The seams under test are
the two gates the runtime applies when the flight ends.
"""

import json

from embodied.bench import live_record
from embodied.platform import localization_check
from embodied.platform import mission_runtime

CRASH_TEXT = "Crash: Disarming: AngErr=49>30, Accel=0.0<3.0"


# ---------------------------------------------------------------------------
# The termination gate
# ---------------------------------------------------------------------------


def test_a_crash_disarm_replaces_the_complete_mission_termination():
    assert (
        mission_runtime.final_termination_reason("mission_completed", [CRASH_TEXT])
        == "crashed"
    )


def test_a_termination_that_already_says_what_happened_stands():
    """budget exhausted and localization loss name their own ends; a crash suffix
    would hide which of the two ended the mission."""
    assert (
        mission_runtime.final_termination_reason(
            "mission_budget_exhausted", [CRASH_TEXT]
        )
        == "mission_budget_exhausted"
    )
    assert (
        mission_runtime.final_termination_reason(
            "visual_localization_lost", [CRASH_TEXT]
        )
        == "visual_localization_lost"
    )


def test_an_unchashed_mission_keeps_its_own_termination():
    assert (
        mission_runtime.final_termination_reason("mission_completed", [])
        == "mission_completed"
    )


# ---------------------------------------------------------------------------
# The receipt verdict (the CommandOutcome gate in the transport)
# ---------------------------------------------------------------------------


def test_a_crashed_flight_fails_its_gate_and_stays_an_outcome():
    """complete-with-FAIL, not blocked: the P06 comparison must keep a policy
    crash among the outcomes instead of pooling it with instrument failures."""
    from embodied.cli import CommandStatus, GateStatus

    status, gate = live_record.receipt_verdict(flew=True, crashed=True, sim_fault=None)
    assert status is CommandStatus.COMPLETE
    assert gate is GateStatus.FAIL


def test_a_wedged_flight_is_blocked_as_an_instrument_fault():
    from embodied.cli import CommandStatus, GateStatus

    status, gate = live_record.receipt_verdict(
        flew=True, crashed=False, sim_fault="sim_fault: ..."
    )
    assert status is CommandStatus.BLOCKED
    assert gate is GateStatus.FAIL


def test_a_flight_that_flew_and_measured_no_fault_still_reads_complete():
    from embodied.cli import CommandStatus, GateStatus

    status, gate = live_record.receipt_verdict(flew=True, crashed=False, sim_fault=None)
    assert status is CommandStatus.COMPLETE
    assert gate is GateStatus.PASS


def test_a_flight_that_never_flew_still_fails_its_gate():
    from embodied.cli import CommandStatus, GateStatus

    status, gate = live_record.receipt_verdict(
        flew=False, crashed=False, sim_fault=None
    )
    assert status is CommandStatus.COMPLETE
    assert gate is GateStatus.FAIL


# ---------------------------------------------------------------------------
# The armed-readback race
# ---------------------------------------------------------------------------


def test_the_raced_armed_flag_is_corrected_toward_disarmed():
    """J58's shape: sampled armed in the race, heartbeats say disarmed."""
    end_state = {
        "armed": True,
        "mode": "GUIDED",
        "attitude_rpy": [179.9, -0.05, 150.2],
    }
    corrected, correction = mission_runtime.reconcile_armed_readback(end_state, False)
    assert corrected["armed"] is False
    assert correction is not None and "heartbeat" in correction
    # The correction is the difference between a record that hides the inverted
    # rest and one that names it.
    finding = live_record.end_state_violation(corrected)
    assert finding is not None and finding.startswith("end_state_inverted:")
    # And the sampled end state is not mutated in place.
    assert end_state["armed"] is True


def test_a_disarmed_sample_is_left_alone():
    end_state = {"armed": False, "mode": "LAND", "attitude_rpy": [0.5, 0.0, 0.0]}
    corrected, correction = mission_runtime.reconcile_armed_readback(end_state, False)
    assert corrected is end_state
    assert correction is None


def test_the_correction_never_arms_a_record_back_up():
    """A stream that says armed (e.g. a truncated log) cannot suppress a violation."""
    end_state = {"armed": False, "mode": "GUIDED", "attitude_rpy": [179.9, 0.0, 0.0]}
    corrected, correction = mission_runtime.reconcile_armed_readback(end_state, True)
    assert corrected["armed"] is False
    assert correction is None


def test_an_unmeasured_stream_corrects_nothing():
    end_state = {"armed": True, "mode": "GUIDED", "attitude_rpy": [179.9, 0.0, 0.0]}
    corrected, correction = mission_runtime.reconcile_armed_readback(end_state, None)
    assert corrected is end_state
    assert correction is None


def test_an_empty_end_state_is_left_alone():
    corrected, correction = mission_runtime.reconcile_armed_readback({}, False)
    assert corrected == {}
    assert correction is None


# ---------------------------------------------------------------------------
# The heartbeat reader the reconciliation reads
# ---------------------------------------------------------------------------


def _write_log(tmp_path, rows):
    log = tmp_path / "mavlink.jsonl"
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return log


def test_the_last_heartbeat_in_the_run_s_own_stream_wins(tmp_path):
    log = _write_log(
        tmp_path,
        [
            {"mavpackettype": "HEARTBEAT", "base_mode": 209},
            {"mavpackettype": "ATTITUDE", "roll": 0.1},
            {"mavpackettype": "HEARTBEAT", "base_mode": 81},
        ],
    )
    assert localization_check._last_heartbeat_armed(log) is False


def test_a_truly_armed_last_heartbeat_reads_armed(tmp_path):
    log = _write_log(
        tmp_path,
        [
            {"mavpackettype": "HEARTBEAT", "base_mode": 81},
            {"mavpackettype": "HEARTBEAT", "base_mode": 209},
        ],
    )
    assert localization_check._last_heartbeat_armed(log) is True


def test_no_heartbeat_answers_unmeasured(tmp_path):
    log = _write_log(tmp_path, [{"mavpackettype": "ATTITUDE", "roll": 0.1}])
    assert localization_check._last_heartbeat_armed(log) is None


def test_a_missing_log_answers_unmeasured(tmp_path):
    assert localization_check._last_heartbeat_armed(tmp_path / "absent.jsonl") is None


# ---------------------------------------------------------------------------
# The units: the end-state attitude is recorded in the bench's degrees
# ---------------------------------------------------------------------------


class _Sample:
    def __init__(self, attitude_rpy):
        self.armed = False
        self.mode_name = "GUIDED"
        self.local_position_ned = (0.0, 0.0, -0.035)
        self.attitude_rpy = attitude_rpy


def test_the_end_state_attitude_is_converted_from_the_telemetry_radians():
    """J58's recorded rest was [3.141, -0.0008, 2.613] rad and read as a 3-degree
    tilt under the bench's degrees rule, so the inversion never fired."""
    record = mission_runtime._end_state_record(_Sample((3.141, -0.0008, 2.613)))
    tilt = live_record.end_state_tilt_deg(record["attitude_rpy"])
    assert tilt is not None and tilt > 45.0
    finding = live_record.end_state_violation(record)
    assert finding is not None and finding.startswith("end_state_inverted:")


def test_an_upright_landing_in_radians_reads_upright():
    record = mission_runtime._end_state_record(_Sample((0.01, -0.01, 1.2)))
    tilt = live_record.end_state_tilt_deg(record["attitude_rpy"])
    assert tilt is not None and tilt < 2.0
    assert live_record.end_state_violation(record) is None
