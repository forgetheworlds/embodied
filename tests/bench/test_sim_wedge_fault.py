"""A frozen truth pose under thrust is a simulator fault, and it refuses grading.

ROLL-DEPARTURE.md (transport lane): J58's world kept running while the body's
truth pose froze for 2.14 s — measured drift at most 5.5e-06 m per 0.020 s row —
with the aircraft armed, airborne and commanding thrust; the crash checker read
the wedged state and disarmed at 2.15 m into a dead fall. The declared bounds and
their measured derivation live beside SIM_WEDGE_* in live_record.
"""

import json
import math

import pytest

from embodied.bench import events as events_module
from embodied.bench import grader
from embodied.bench import live_record
from embodied.contracts.records import ClockStamp

from pathlib import Path
import shutil

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "bench"
    / "hand-checkable-episode"
)

HOVER = (-0.219, -0.077, -2.260)
GROUND = (-0.300, 0.010, -0.035)


class _MutableRecord:
    """The pose-record shape the collector reads, mutable between rows."""

    def __init__(self):
        self.pose = None
        self.sim_time_s = 0.0
        self.received_stamp = ClockStamp(
            host_id="h", clock_id="monotonic", monotonic_ns=0
        )


def _frozen_collector(xyz=HOVER, rows=130, step_s=0.02, drift_m=1e-7):
    """The wedge shape: rows of bit-stable pose, 0.02 s apart on the receipt clock."""
    collector = live_record.TruthCollector()
    record = _MutableRecord()
    for index in range(rows):
        record.sim_time_s = index * step_s
        record.received_stamp = ClockStamp(
            host_id="h", clock_id="monotonic", monotonic_ns=int(index * step_s * 1e9)
        )
        # A wedge drifts far below any control loop's resolution; the drift
        # parameter is per row, like the measured 5.5e-06 m/row ceiling.
        record.pose = type(
            "Pose",
            (),
            {"position_xyz": tuple(value + drift_m * (index % 3 - 1) for value in xyz)},
        )()
        collector(record)
    return collector


def _flying_rows(*spans, start_ns=0):
    """(receipt_ns, flying) rows: one True at the start, or (start, value) spans."""
    rows = []
    for begin, value in spans:
        rows.append((start_ns + begin, value))
    return rows


def test_the_wedge_shape_is_named_and_refuses_grading():
    collector = _frozen_collector()  # 130 rows x 0.02 s = 2.60 s frozen
    fault = live_record.sim_wedge_fault(collector.samples, _flying_rows((0, True)))
    assert fault is not None
    assert fault.startswith("sim_fault:")
    assert "2.58 s" in fault
    assert "cannot be graded as a flight" in fault


def test_the_measured_wedge_drift_is_inside_the_bound_and_honest_motion_is_not():
    """The declared 1e-4 m/row separates the measured wedge from a measured hold."""
    wedged = live_record.sim_wedge_fault(
        _frozen_collector(drift_m=5.5e-6).samples, _flying_rows((0, True))
    )
    assert wedged is not None, "the measured wedge ceiling must trip the bound"
    honest = live_record.sim_wedge_fault(
        _frozen_collector(drift_m=3.6e-3).samples, _flying_rows((0, True))
    )
    assert honest is None, "the measured honest-hold median must not trip the bound"


def test_a_parked_or_landed_aircraft_is_no_fault():
    """Immobile on the ground is what parked and landed aircraft do."""
    collector = _frozen_collector(xyz=GROUND)
    assert (
        live_record.sim_wedge_fault(collector.samples, _flying_rows((0, True))) is None
    )


def test_motors_cut_is_no_fault():
    """An aircraft with its motors off may hang in the fall or rest wherever it is."""
    collector = _frozen_collector()
    assert (
        live_record.sim_wedge_fault(collector.samples, _flying_rows((0, False))) is None
    )


def test_a_freeze_below_the_bound_is_no_fault():
    collector = _frozen_collector(rows=15)  # 15 x 0.02 s = 0.28 s < 0.5 s
    assert (
        live_record.sim_wedge_fault(collector.samples, _flying_rows((0, True))) is None
    )


def test_no_flying_evidence_is_no_fault():
    """Without a stream to contradict, the detector stays silent rather than guesses."""
    collector = _frozen_collector()
    assert live_record.sim_wedge_fault(collector.samples, []) is None


# ---------------------------------------------------------------------------
# The flying rows come from the run's own MAVLink stream
# ---------------------------------------------------------------------------


def _write_mavlink(tmp_path, rows):
    log = tmp_path / "mavlink.jsonl"
    lines = []
    for document in rows:
        lines.append(json.dumps(document))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


def test_the_detector_reads_armed_and_thrust_from_the_recorded_stream(tmp_path):
    log = _write_mavlink(
        tmp_path,
        [
            {
                "mavpackettype": "HEARTBEAT",
                "base_mode": 209,
                "received_monotonic_ns": 100,
            },
            {
                "mavpackettype": "SERVO_OUTPUT_RAW",
                "servo1_raw": 1900,
                "servo2_raw": 1209,
                "servo3_raw": 1949,
                "servo4_raw": 1150,
                "received_monotonic_ns": 150,
            },
            {"mavpackettype": "ATTITUDE", "roll": 0.0, "received_monotonic_ns": 160},
            {
                "mavpackettype": "HEARTBEAT",
                "base_mode": 81,
                "received_monotonic_ns": 200,
            },
            {
                "mavpackettype": "SERVO_OUTPUT_RAW",
                "servo1_raw": 1000,
                "servo2_raw": 1000,
                "servo3_raw": 1000,
                "servo4_raw": 1000,
                "received_monotonic_ns": 250,
            },
        ],
    )
    rows = live_record.armed_thrust_rows(log)
    # Nothing is claimed before both signals have spoken; after that each change
    # is one row: armed+thrust at the servo message, disarmed at the heartbeat,
    # motors cut at the flat-1000 row.
    assert rows == [(150, True), (200, False), (250, False)]


def test_a_missing_log_answers_no_rows():
    assert live_record.armed_thrust_rows("/nonexistent/mavlink.jsonl") == []


def test_the_flying_rows_join_the_truth_stream_on_the_receipt_clock():
    """The wedge is found when the frozen rows are stamped inside a flying span."""
    collector = _frozen_collector()
    # Flying only after the frozen window ends: no fault.
    late = live_record.sim_wedge_fault(collector.samples, [(int(3.0 * 1e9), True)])
    assert late is None
    # Flying from the start: the same frozen rows are a fault.
    early = live_record.sim_wedge_fault(collector.samples, [(0, True)])
    assert early is not None


# ---------------------------------------------------------------------------
# The fault reaches the recorded outcome and refuses the grader
# ---------------------------------------------------------------------------


def test_the_fault_travels_in_the_recorded_outcome_payload():
    collector = _frozen_collector()
    fault = live_record.sim_wedge_fault(collector.samples, _flying_rows((0, True)))
    outcome = live_record.measure_physical_outcome(
        collector,
        target_ned=None,
        target_id="red_block",
        end_state={},
        crash_statustexts=[],
        guidance_events=[],
        sim_fault=fault,
    )
    assert outcome["payload"]["sim_fault"] == fault
    # The payload with the fault validates against the bench-side envelope...
    events_module.validate_payload("physical_outcome", outcome["payload"])
    # ...and so does a None one, the shape every new recording writes.
    none_payload = dict(outcome["payload"], sim_fault=None)
    events_module.validate_payload("physical_outcome", none_payload)


def test_the_envelope_still_accepts_episodes_recorded_before_the_detector():
    events_module.validate_payload(
        "physical_outcome",
        {
            "inspected": {"red_block": False},
            "return_verified": True,
            "violations": [],
            "takeover": False,
        },
    )


def test_the_envelope_refuses_a_fault_that_is_not_a_named_string():
    with pytest.raises(events_module.EventError):
        events_module.validate_payload(
            "physical_outcome",
            {
                "inspected": {"red_block": False},
                "return_verified": True,
                "violations": [],
                "takeover": False,
                "sim_fault": 17,
            },
        )


def _copied_episode(tmp_path):
    episode = tmp_path / "episode"
    shutil.copytree(FIXTURE, episode)
    shutil.copytree(_truth_store(FIXTURE), _truth_store(episode))
    return episode


def _truth_store(episode_dir):
    from embodied.bench.referee import truth_store_path

    return truth_store_path(episode_dir)


def test_scoring_refuses_an_episode_with_a_recorded_wedge(tmp_path):
    episode = _copied_episode(tmp_path)
    store = _truth_store(episode)
    stream = store / "truth-events.jsonl"
    lines = stream.read_text(encoding="utf-8").splitlines()
    outcome = json.loads(lines[1])
    assert outcome["kind"] == "physical_outcome"
    outcome["payload"]["sim_fault"] = (
        "sim_fault: the simulator's truth pose held still for 2.58 s while the "
        "aircraft was armed, airborne and commanding thrust"
    )
    lines[1] = json.dumps(outcome)
    stream.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(grader.GradeError, match="simulator fault"):
        grader.grade(episode)


def test_scoring_is_unchanged_for_an_episode_without_a_fault(tmp_path):
    episode = _copied_episode(tmp_path)
    score = grader.grade(episode)
    # The fixture's own committed verdict (an intervention keeps task completion
    # false); the point is that grading proceeds at all with no fault recorded.
    assert score.document()["mission"]["physical_return_verified"] is True


def test_the_measured_freeze_span_uses_the_host_clock_not_the_sim_clock():
    """The bound is declared on the receipt clock the samples and rows share."""
    # 2.14 s of wedge measured on J58, at the envelope's slowest ratio (0.5):
    # 2.14 s of sim time is then 4.28 s of host time — still far above 0.5 s.
    # At the fastest (1.5), the same interval is 1.43 s of host time — still
    # above the bound, which is the direction the margin derivation needs.
    assert 2.14 * 1.5 > live_record.SIM_WEDGE_MIN_FREEZE_S
    assert math.isclose(live_record.SIM_WEDGE_MIN_FREEZE_S, 0.5)
