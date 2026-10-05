"""What counts as a loss of Guided control, and what does not.

The platform emits one control event for every change of mode or arming state. This
file pins the bench side's reading of those events, because getting it wrong is
invisible: a ``guidance_lost:7`` stood in the receipt of every completed mission
while the aircraft never actually left armed Guided control except to land.

The sequences below are the ones the recorded runs produced, transcribed from
``work/runs/p05/live-motion-8/platform/run-a/mavlink.jsonl`` (its 67 heartbeats
replayed through the platform's own fold).
"""

from __future__ import annotations

from typing import Any

from embodied.bench import live_record


def event(
    t: int,
    from_mode: str | None,
    to_mode: str | None,
    armed_before: bool | None,
    armed_after: bool | None,
    *,
    guidance_held: bool | None = None,
) -> dict[str, Any]:
    """One control event, in the shape ``ControlEvent.document()`` emits."""
    return {
        "at_monotonic_ns": t,
        "from_mode": from_mode,
        "to_mode": to_mode,
        "armed_before": armed_before,
        "armed_after": armed_after,
        "system_status": 3,
        "statustexts": [],
        "guidance_held": (
            bool(to_mode == "GUIDED" and armed_after)
            if guidance_held is None
            else guidance_held
        ),
    }


def the_recorded_mission() -> list[dict[str, Any]]:
    """The seven-transition lifecycle every flown mission produced.

    Bring-up (disarmed, ALT_HOLD, the declared LAND, the disarm), the handover into
    Guided, and the mission's own terminal landing. Not one of these is a loss of
    control.
    """
    return [
        event(0, "STABILIZE", "ALT_HOLD", False, False),
        event(1, "ALT_HOLD", "ALT_HOLD", False, True),
        event(2, "ALT_HOLD", "LAND", True, True),
        event(3, "LAND", "LAND", True, False),
        event(4, "LAND", "GUIDED", False, True),
        event(5, "GUIDED", "LAND", True, True),
        event(6, "LAND", "LAND", True, False),
    ]


LANDED = {"armed": False, "mode": "LAND"}


def test_the_recorded_mission_lifecycle_is_not_a_control_loss():
    """The exact sequence that produced ``guidance_lost:7`` produces no loss.

    This is the test that fails on the old count: it asserted on
    ``not event["guidance_held"]``, which every bring-up and landing transition
    satisfies, so it reported seven losses for a flight that had none.
    """
    losses, departures = live_record._guidance_departures(
        the_recorded_mission(), LANDED
    )
    assert losses == []
    # One departure did happen: the mission's own landing, which is why it must be
    # excluded by the end state rather than by ignoring the events.
    assert [event["to_mode"] for event in departures] == ["LAND"]


def test_the_rule_this_replaced_would_have_counted_seven():
    """Pin the miscount this change corrects, so the correction cannot regress.

    The rule was ``not event["guidance_held"]``. On the eight events a mission
    actually produces — the lifecycle plus the pre-telemetry artifact — it reports
    seven, which is the number that stood in every mission receipt. Nothing here is
    a loss, so the two counts cannot both be right.
    """
    events = [event(0, None, "STABILIZE", None, False)] + the_recorded_mission()
    assert len([e for e in events if not e["guidance_held"]]) == 7
    losses, _ = live_record._guidance_departures(events, LANDED)
    assert losses == []


def test_a_transition_before_any_telemetry_is_not_a_loss():
    """``from_mode`` None is a report artifact: the aircraft was flying nothing yet.

    Whether the first drain lands before the first heartbeat depends on timing, which
    is why runs recorded six or seven losses for the same flight.
    """
    events = [event(0, None, "STABILIZE", None, False)] + the_recorded_mission()
    losses, departures = live_record._guidance_departures(events, LANDED)
    assert losses == []
    assert len(departures) == 1


def test_a_disarmed_bring_up_is_not_a_loss():
    """Nothing was under Guided control to lose, so nothing can have been lost."""
    events = [event(0, "STABILIZE", "LAND", False, False)]
    losses, departures = live_record._guidance_departures(events, LANDED)
    assert losses == []
    assert departures == []


def test_a_mid_flight_departure_is_a_loss():
    """The case the detector exists for: the autopilot takes control away.

    Taken from ``tests/platform/test_webots_ardupilot.py``'s failsafe scenario, where
    the aircraft goes to LOITER while armed and the mission's setpoints are refused.
    """
    events = [
        event(0, "LAND", "GUIDED", False, True),
        event(1, "GUIDED", "LOITER", True, True),
        event(2, "LOITER", "LAND", True, False),
    ]
    losses, departures = live_record._guidance_departures(events, LANDED)
    assert len(losses) == 1
    assert losses[0]["from_mode"] == "GUIDED"
    assert losses[0]["to_mode"] == "LOITER"


def test_a_departure_that_returns_is_still_a_loss():
    """Leaving Guided and coming back still took control away, so it still counts."""
    events = [
        event(0, "LAND", "GUIDED", False, True),
        event(1, "GUIDED", "LOITER", True, True),
        event(2, "LOITER", "GUIDED", True, True),
        event(3, "GUIDED", "LAND", True, True),
    ]
    losses, departures = live_record._guidance_departures(events, LANDED)
    assert [loss["to_mode"] for loss in losses] == ["LOITER"]
    assert len(departures) == 2


def test_a_terminal_departure_that_is_not_the_declared_landing_is_a_loss():
    """Ending in RTL is not the mission's declared termination, so it stays a loss."""
    events = [
        event(0, "LAND", "GUIDED", False, True),
        event(1, "GUIDED", "RTL", True, True),
        event(2, "RTL", "LAND", True, False),
    ]
    losses, _ = live_record._guidance_departures(events, LANDED)
    assert len(losses) == 1
    assert losses[0]["to_mode"] == "RTL"


def test_a_terminal_landing_is_a_loss_when_the_run_did_not_end_landed():
    """The excuse needs the declared end state, not just the word LAND.

    A run that stops armed, or somewhere else entirely, has not completed the
    termination that makes a landing ordinary.
    """
    events = [event(0, "GUIDED", "LAND", True, True)]
    losses, _ = live_record._guidance_departures(
        events, {"armed": True, "mode": "LAND"}
    )
    assert len(losses) == 1
    losses, _ = live_record._guidance_departures(
        events, {"armed": False, "mode": "LOITER"}
    )
    assert len(losses) == 1


class _Collector:
    """The two measurements ``measure_physical_outcome`` asks the truth stream for."""

    def __init__(self, inspected: bool = False, returned: bool = True) -> None:
        self._inspected = inspected
        self._returned = returned

    def inspected_within(self, target_ned, *, radius_m, hold_s):
        return self._inspected, "declared"

    def returned_near(self, *, radius_m, end_state):
        return self._returned, "declared"


def test_the_outcome_records_no_guidance_violation_for_the_recorded_lifecycle():
    """Through the public seam: the run's violations list stays free of guidance_lost."""
    outcome = live_record.measure_physical_outcome(
        _Collector(),
        target_ned=(0.0, 0.0, 1.0),
        target_id="red_block",
        end_state=LANDED,
        crash_statustexts=[],
        guidance_events=the_recorded_mission(),
    )
    assert outcome["violations"] == []
    assert outcome["payload"]["violations"] == []
    # The accounting travels with the outcome, so the count can be audited.
    assert outcome["guidance"]["events"] == 7
    assert outcome["guidance"]["departures_from_armed_guided"] == 1
    assert outcome["guidance"]["losses"] == 0


def test_the_outcome_still_reports_a_real_loss():
    """The seam keeps the violation, and names the departure that caused it."""
    events = [
        event(0, "LAND", "GUIDED", False, True),
        event(1, "GUIDED", "LOITER", True, True),
        event(2, "LOITER", "LAND", True, False),
    ]
    outcome = live_record.measure_physical_outcome(
        _Collector(),
        target_ned=(0.0, 0.0, 1.0),
        target_id="red_block",
        end_state=LANDED,
        crash_statustexts=[],
        guidance_events=events,
    )
    assert outcome["violations"] == ["guidance_lost:1"]
    assert outcome["payload"]["violations"] == ["guidance_lost:1"]
    assert outcome["guidance"]["lost"][0]["to_mode"] == "LOITER"


def test_a_failsafe_landing_is_not_excused():
    """The aircraft's own words override the end state.

    ``live-14`` is the recorded case: an EKF failsafe changed the mode to LAND and
    the aircraft then crashed. The run ends in the declared landing state, so the
    end state alone would excuse the departure — and would hide the one mode change
    on record that the autopilot made by itself.
    """
    events = the_recorded_mission()
    excused, _ = live_record._guidance_departures(events, LANDED, ())
    assert excused == []
    losses, departures = live_record._guidance_departures(
        events, LANDED, ("EKF Failsafe: changed to Land Mode",)
    )
    assert len(losses) == 1
    assert losses[0]["to_mode"] == "LAND"
    # The departure was there all along; only the excuse changed.
    assert len(departures) == 1


def test_the_autopilot_mode_change_scan_reads_the_runs_own_record(tmp_path):
    """What the scan finds, and what it must not: the mission's own changes."""
    log = tmp_path / "mavlink.jsonl"
    log.write_text(
        "\n".join(
            [
                '{"mavpackettype": "HEARTBEAT", "custom_mode": 4, "base_mode": 217}',
                '{"mavpackettype": "STATUSTEXT", "text": "EKF Failsafe"}',
                '{"mavpackettype": "STATUSTEXT", "text": "EKF Failsafe Cleared"}',
                '{"mavpackettype": "STATUSTEXT", "text": "Arming motors"}',
                '{"mavpackettype": "STATUSTEXT", "text": "EKF Failsafe: changed to Land Mode"}',
                '{"mavpackettype": "STATUSTEXT", "text": "EKF Failsafe: changed to Land Mode"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    found = live_record.autopilot_mode_change_statustexts(log)
    # Named once, and a failsafe that did not take the mode is not a mode change.
    assert found == ["EKF Failsafe: changed to Land Mode"]


def test_the_scan_is_empty_when_there_is_no_record(tmp_path):
    """A run with no log has no mode changes to report, and must not fail."""
    assert (
        live_record.autopilot_mode_change_statustexts(tmp_path / "absent.jsonl") == []
    )


def test_a_crash_disarm_is_still_reported_alongside_guidance():
    """The two violations are independent and neither suppresses the other."""
    outcome = live_record.measure_physical_outcome(
        _Collector(),
        target_ned=(0.0, 0.0, 1.0),
        target_id="red_block",
        end_state=LANDED,
        crash_statustexts=["Crash: Disarming: AngErr=100>30"],
        guidance_events=the_recorded_mission(),
    )
    assert outcome["violations"] == [
        "crash_disarm: Crash: Disarming: AngErr=100>30",
    ]
