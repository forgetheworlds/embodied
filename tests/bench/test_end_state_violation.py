"""An end state no landing produces must not read as a landing.

The case this exists for is recorded. ``j26-live-2`` came to rest inverted, on
the ground, for about a minute: its own telemetry ends at roll **179.1 deg**,
its estimator had run to 278 m inside a 6 m room, and its record said
``violations: none``. A landing and a crash were written identically, and the
next mission read that crashed aircraft's resting pose as a healthy one's
measurement error and built two claims on it before they were refuted.

The predicate is the vehicle's own reported attitude against
``END_STATE_MAX_LANDING_TILT_DEG``: a multirotor that has landed rests on its
base, so its up-axis sits near vertical, and the boundary is placed in the
measured gap between the two populations. Across the 32 runs on disk every run
that **landed** measured **0.971 deg or less**, and the six that did **not**
measured **89.980 to 179.660 deg** — each of those six steady across its last six
samples, so they are resting poses rather than a tumble caught mid-flight.

Nothing here reads the real run tree: every fixture is written into ``tmp_path``,
because a test that depends on what happens to be on disk fails for reasons that
have nothing to do with the code (LEARNED-FAILURES T18).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from embodied.bench import live_record

LANDED = {"armed": False, "mode": "LAND"}
INVERTED = {"armed": False, "mode": "LAND", "attitude_rpy": [179.1, 0.4, 96.7]}


class _Collector:
    """The two measurements ``measure_physical_outcome`` asks the truth stream for."""

    def inspected_within(self, target_ned, *, radius_m, hold_s):
        return False, "declared"

    def returned_near(self, *, radius_m):
        return True, "declared"


def _outcome(end_state, *, crash_statustexts=None):
    return live_record.measure_physical_outcome(
        _Collector(),
        target_ned=(0.0, 0.0, 1.0),
        target_id="red_block",
        end_state=end_state,
        crash_statustexts=list(crash_statustexts or []),
        guidance_events=[],
    )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_a_run_that_came_to_rest_inverted_is_a_violation():
    """Through the public seam: the inversion reaches the run's own record.

    The end state is ``j26-live-2``'s, with the attitude its telemetry carries
    and its record lacked.
    """
    outcome = _outcome(
        {
            "armed": False,
            "mode": "LAND",
            "local_position_ned": [-49.57256317138672, 30.603376388549805, -0.515224277973175],
            "attitude_rpy": [179.1, 0.4, 96.7],
        }
    )
    assert outcome["violations"] == [
        "end_state_inverted: the vehicle came to rest 179.0 deg from vertical "
        "(roll 179.1, pitch 0.4) while disarmed, past the 45 deg a landing can leave it"
    ]
    # The bench-side payload carries the same list, so the referee's store and the
    # receipt cannot disagree about it.
    assert outcome["payload"]["violations"] == outcome["violations"]


def test_the_boundary_lies_in_the_gap_between_landings_and_everything_else():
    """The margin is the point: a real run's verdict must not turn on a rounding.

    The two measured populations are 0.971 deg and 89.980 deg apart, so any
    boundary inside that gap separates them and 45 is the middle of it. The first
    case here is the nearest non-landing on disk — ``live-19``, a vehicle resting
    on its side, which a geometric boundary at 90 deg would have missed by two
    hundredths of a degree.
    """
    assert live_record.end_state_violation(
        {**LANDED, "attitude_rpy": [-84.58, 89.79, 12.0]}
    ).startswith("end_state_inverted:"), "the nearest non-landing must be caught"
    # Just inside the boundary is a pass, so the boundary is a boundary and not a
    # slope whose position we could drift.
    assert (
        live_record.end_state_violation({**LANDED, "attitude_rpy": [0.0, 44.9, 0.0]})
        is None
    )
    # And just outside it is not, which is what makes the boundary exact.
    assert live_record.end_state_violation(
        {**LANDED, "attitude_rpy": [0.0, 45.1, 0.0]}
    ).startswith("end_state_inverted:")


def test_a_landing_is_not_a_violation():
    """The false-positive check, at the attitudes real landings actually produce.

    These bracket what the runs on disk measured (0.0 to 1.0 deg) and add a
    generous margin either side. A criterion that flagged any of these would be
    worse than none: it would put a violation on every good run.
    """
    for attitude in (
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 12.0],
        [-0.2, -0.0, -3.0],
        [1.0, 1.0, 90.0],
        [-1.0, 0.5, 179.0],
    ):
        assert (
            live_record.end_state_violation({**LANDED, "attitude_rpy": attitude}) is None
        ), attitude


def test_a_run_that_ended_still_flying_is_not_judged():
    """The predicate asks whether the vehicle came to rest wrongly.

    An end state that is still armed is mid-flight — its attitude is a manoeuvre,
    and an aggressive one can pass 90 deg with nothing wrong. Unknown arming is
    not evidence of a landing either; it is undecidable, which is a different
    statement and must not be recorded as a pass.
    """
    assert (
        live_record.end_state_violation({"armed": True, "attitude_rpy": [179.0, 0.0, 0.0]})
        is None
    )
    assert live_record.end_state_violation({"attitude_rpy": [179.0, 0.0, 0.0]}) is None


def test_an_absent_attitude_is_undecidable_not_upright():
    """The field postdates the runs it would judge, so its absence proves nothing."""
    assert live_record.end_state_violation(dict(LANDED)) is None
    assert live_record.end_state_tilt_deg(None) is None
    assert live_record.end_state_tilt_deg([1.0, 2.0]) is None
    assert live_record.end_state_tilt_deg([float("nan"), 0.0, 0.0]) is None


def test_the_tilt_is_the_angle_the_up_axis_makes_with_vertical():
    """Known-good angles first — these are geometry, not this module's arithmetic."""
    assert live_record.end_state_tilt_deg([0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert live_record.end_state_tilt_deg([90.0, 0.0, 0.0]) == pytest.approx(90.0)
    assert live_record.end_state_tilt_deg([180.0, 0.0, 0.0]) == pytest.approx(180.0)
    assert live_record.end_state_tilt_deg([0.0, 90.0, 0.0]) == pytest.approx(90.0)
    assert live_record.end_state_tilt_deg([60.0, 0.0, 0.0]) == pytest.approx(60.0)
    # Yaw cannot move the up-axis, so it cannot move the tilt.
    assert live_record.end_state_tilt_deg([45.0, 30.0, 200.0]) == pytest.approx(
        live_record.end_state_tilt_deg([45.0, 30.0, 0.0])
    )
    # Then a cross-check by a different route: compose the three elementary
    # rotations and take the angle of the rotated up-axis to vertical. This is
    # not an independent authority for the numbers above — it is a check that the
    # closed form and the rotation order agree, which a sign or order error in
    # one of them would break.
    def tilt_by_composition(roll_deg: float, pitch_deg: float, yaw_deg: float) -> float:
        r, p, y = (math.radians(value) for value in (roll_deg, pitch_deg, yaw_deg))
        up_z = math.cos(p) * math.cos(r)
        return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))

    for roll, pitch, yaw in ((0.0, 0.0, 0.0), (179.1, 0.4, 96.7), (45.0, 30.0, 200.0)):
        assert live_record.end_state_tilt_deg([roll, pitch, yaw]) == pytest.approx(
            tilt_by_composition(roll, pitch, yaw), abs=1e-9
        ), (roll, pitch, yaw)


def test_a_crash_and_an_inversion_are_reported_independently():
    """Neither suppresses the other: a crash that ended inverted carries both."""
    outcome = _outcome(
        dict(INVERTED),
        crash_statustexts=["Crash: Disarming: AngErr=100>30, Accel=0.1<3.0"],
    )
    assert outcome["violations"][0] == (
        "crash_disarm: Crash: Disarming: AngErr=100>30, Accel=0.1<3.0"
    )
    assert outcome["violations"][1].startswith("end_state_inverted:")
    assert len(outcome["violations"]) == 2


# ---------------------------------------------------------------------------
# Making a run already on disk interpretable
# ---------------------------------------------------------------------------


def _write_run(
    root: Path,
    *,
    end_state: dict,
    telemetry: list[dict] | None,
    violations: list[str] | None = None,
    run: str = "run-x",
) -> Path:
    run_dir = root / run
    (run_dir / "platform" / "run-a").mkdir(parents=True)
    (run_dir / "mission.json").write_text(
        json.dumps(
            {
                "outcome": {
                    "end_state": end_state,
                    "violations": list(violations or []),
                }
            }
        ),
        encoding="utf-8",
    )
    if telemetry is not None:
        (run_dir / "platform" / "run-a" / "mavlink.jsonl").write_text(
            "\n".join(json.dumps(message) for message in telemetry) + "\n",
            encoding="utf-8",
        )
    return run_dir


def test_a_run_recorded_before_the_field_is_judged_on_its_own_telemetry(tmp_path):
    """The recorded case, end to end on data shaped like ``j26-live-2``'s.

    Its attitude is in its platform log because the end-state field postdates the
    run. Reaching it is the difference between making a past record interpretable
    and leaving it wrong.
    """
    run_dir = _write_run(
        tmp_path,
        end_state={
            "armed": False,
            "mode": "LAND",
            "local_position_ned": [-49.57, 30.60, -0.52],
        },
        telemetry=[
            {"mavpackettype": "HEARTBEAT"},
            {"mavpackettype": "ATTITUDE", "roll": math.radians(0.2), "pitch": 0.0, "yaw": 0.0},
            {
                "mavpackettype": "ATTITUDE",
                "roll": math.radians(179.1),
                "pitch": math.radians(0.4),
                "yaw": math.radians(96.7),
            },
            # A later message that is not an attitude must not displace the last
            # attitude: it is the last ATTITUDE that the run ended in.
            {"mavpackettype": "LOCAL_POSITION_NED", "x": -49.5, "y": 30.6, "z": -0.5},
        ],
    )
    finding = live_record.rederive_end_state_violation(run_dir)
    assert finding["attitude_source"] == (
        "platform/*/mavlink.jsonl (the last ATTITUDE message)"
    )
    assert finding["attitude_rpy_deg"] == [
        pytest.approx(179.1),
        pytest.approx(0.4),
        pytest.approx(96.7),
    ]
    assert finding["tilt_deg"] == pytest.approx(179.0, abs=0.05)
    assert finding["violation"] == (
        "end_state_inverted: the vehicle came to rest 179.0 deg from vertical "
        "(roll 179.1, pitch 0.4) while disarmed, past the 45 deg a landing can leave it"
    )
    assert finding["violations_rederived"] == [finding["violation"]]
    # The run's own recorded end state is reported as it was, not as edited.
    assert finding["end_state_recorded"] == {
        "armed": False,
        "mode": "LAND",
        "local_position_ned": [-49.57, 30.60, -0.52],
    }


def test_a_good_landing_is_left_alone_by_the_rederivation(tmp_path):
    """The false-positive check where it matters most: a silent, correct run."""
    run_dir = _write_run(
        tmp_path,
        end_state={"armed": False, "mode": "LAND"},
        telemetry=[
            {"mavpackettype": "ATTITUDE", "roll": math.radians(-0.2), "pitch": 0.0, "yaw": 0.0},
            {"mavpackettype": "ATTITUDE", "roll": 0.0, "pitch": math.radians(0.1), "yaw": 1.0},
        ],
    )
    finding = live_record.rederive_end_state_violation(run_dir)
    assert finding["tilt_deg"] == pytest.approx(0.1, abs=0.06)
    assert finding["violation"] is None
    assert finding["violations_rederived"] == []


def test_a_run_with_no_attitude_anywhere_is_reported_as_undecided(tmp_path):
    """No telemetry is not a landing, and the finding must not imply one."""
    run_dir = _write_run(
        tmp_path,
        end_state={"armed": False, "mode": "LAND"},
        telemetry=None,
        violations=["guidance_lost:1"],
    )
    finding = live_record.rederive_end_state_violation(run_dir)
    assert finding["attitude_source"] == "absent"
    assert finding["attitude_rpy_deg"] is None
    assert finding["tilt_deg"] is None
    assert finding["violation"] is None
    # Whatever the run did record is reported unchanged.
    assert finding["violations_rederived"] == ["guidance_lost:1"]


def test_the_rederivation_keeps_the_violations_the_run_already_had(tmp_path):
    """It adds a finding; it does not replace the record it is reading."""
    run_dir = _write_run(
        tmp_path,
        end_state={"armed": False, "mode": "LAND"},
        telemetry=[
            {
                "mavpackettype": "ATTITUDE",
                "roll": math.radians(179.7),
                "pitch": math.radians(-0.1),
                "yaw": 0.0,
            }
        ],
        violations=["crash_disarm: Crash: Disarming: AngErr=70>30, Accel=0.0<3.0"],
    )
    finding = live_record.rederive_end_state_violation(run_dir)
    assert finding["recorded_violations"] == [
        "crash_disarm: Crash: Disarming: AngErr=70>30, Accel=0.0<3.0"
    ]
    assert finding["violations_rederived"][0] == finding["recorded_violations"][0]
    assert finding["violations_rederived"][1].startswith("end_state_inverted:")
