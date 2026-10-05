"""A return is a flown approach to a landed end state, not a resting place near the spawn.

ROLL-DEPARTURE.md (grading lane): J58's record verified the return of an aircraft
that fell out of the sky inverted 0.70 m from its spawn, because its third explore
terminus happened to sit 0.79 m away and the criterion was the horizontal radius
alone. These tests pin the three conjuncts — the radius, the landed end state and
the flown approach — through the same public seam the referee records through.
"""

import math

from embodied.bench import live_record

SPAWN = (-1.0, 0.0, -0.035)
# J58's own measured numbers: the terminus 0.785 m from the spawn, the inverted
# rest 0.700 m from it (episode.truth and the capture's pose rows).
TERMINUS = (-0.219, -0.077, -2.260)
CRASH_REST = (-0.300, 0.010, -0.035)

LANDED = {"armed": False, "mode": "LAND", "attitude_rpy": [0.005, -0.003, 0.10]}
INVERTED = {
    # J58's inverted rest in the bench-side degrees convention (the recorded
    # radians triple [3.141, -0.0008, 2.613] converted): roll ~pi.
    "armed": False,
    "mode": "GUIDED",
    "attitude_rpy": [179.9, -0.05, 150.2],
}


class _FakeTruthRecord:
    """The pose-record shape the collector reads, with a host receipt stamp."""

    def __init__(self, sim_time_s, xyz, receipt_ns):
        self.sim_time_s = sim_time_s
        self.pose = type("Pose", (), {"position_xyz": xyz})()
        self.received_stamp = type("Stamp", (), {"monotonic_ns": receipt_ns})()


def _collector(points, *, receipt_step_ns=20_000_000):
    """points: (sim_time_s, (x, y, z)) triples, in stream order."""
    collector = live_record.TruthCollector()
    for index, (sim_time_s, xyz) in enumerate(points):
        collector(_FakeTruthRecord(sim_time_s, xyz, index * receipt_step_ns))
    return collector


def _j58_shaped_collector():
    """The measured shape: away to the terminus, hold, fall, inverted rest."""
    points = [(0.0, SPAWN)]
    for step in range(1, 21):  # 0-20 s: away from the spawn, climbing
        fraction = step / 20
        points.append(
            (
                fraction * 20.0,
                tuple(
                    start + (end - start) * fraction
                    for start, end in zip(SPAWN, TERMINUS)
                ),
            )
        )
    for step in range(40):  # 20-24 s: the terminus hold
        points.append((24.0 + step * 0.1, TERMINUS))
    for step in range(1, 9):  # the fall (1.05 s on J58; 0.2 s steps here)
        fraction = step / 8
        points.append(
            (
                25.0 + fraction,
                tuple(
                    start + (end - start) * fraction
                    for start, end in zip(TERMINUS, CRASH_REST)
                ),
            )
        )
    for step in range(150):  # ~3 s of static rest past the window
        points.append((26.5 + step * 0.02, CRASH_REST))
    return _collector(points)


# ---------------------------------------------------------------------------
# The three conjuncts
# ---------------------------------------------------------------------------


def test_the_j58_shape_is_refused_by_the_landed_end_state():
    """The measured case: crashed rest inside the radius, inverted, no return."""
    collector = _j58_shaped_collector()
    returned, detail = collector.returned_near(radius_m=1.0, end_state=INVERTED)
    assert returned is False
    assert "no landing" in detail
    assert "end_state_inverted" in detail


def test_a_landed_rest_inside_the_radius_after_a_flown_approach_verifies():
    """The honest case the stricter predicate must still verify."""
    points = [(0.0, (0.0, 0.0, -0.035))]
    approach = [
        (1.60, -1.20),
        (1.20, -1.10),
        (0.80, -0.90),
        (0.40, -0.50),
    ]
    for step, (distance, altitude) in enumerate(approach, start=1):
        points.append((step * 1.5, (distance, 0.0, altitude)))
    # The settle: hover noise around the rest position, then rest. The oscillation
    # (0.04 m, inside the measured honest-hold envelope) must not read as
    # away-motion.
    for step in range(60):
        distance = 0.05 + (0.04 if step % 2 else 0.0)
        points.append((10.0 + step * 0.1, (distance, 0.0, -0.035)))
    collector = _collector(points)
    returned, detail = collector.returned_near(radius_m=1.0, end_state=LANDED)
    assert returned is True, detail
    assert "flown approach" in detail


def test_a_monotone_away_tail_inside_the_radius_is_refused():
    """Crashed upright inside the radius while still travelling away: no return.

    The end state alone cannot catch this shape (upright, disarmed); the flown
    approach conjunct is what does.
    """
    points = [(0.0, (-0.90, 0.0, -1.00))]
    for step in range(1, 11):  # the window: 0.2 s rows, travelling away from the spawn
        points.append((step * 0.2, (-0.825 + 0.075 * step, 0.0, -0.05)))
    collector = _collector(points)
    landed = {"armed": False, "mode": "LAND", "attitude_rpy": [0.01, 0.0, 0.0]}
    returned, detail = collector.returned_near(radius_m=1.0, end_state=landed)
    assert returned is False
    assert "moving away from the start" in detail


def test_an_armed_end_is_not_a_return():
    """The radius is geometry; a return is also a landing, and armed is mid-flight."""
    points = [(0.0, SPAWN), (1.0, (-0.95, 0.0, -0.035))]
    collector = _collector(points)
    armed = dict(LANDED, armed=True)
    returned, detail = collector.returned_near(radius_m=1.0, end_state=armed)
    assert returned is False
    assert "still armed" in detail


def test_an_unmeasured_end_attitude_cannot_verify_a_return():
    """Undecided is not upright: a return needs the landing decided."""
    points = [(0.0, SPAWN), (1.0, (-0.95, 0.0, -0.035))]
    collector = _collector(points)
    undecided = {"armed": False, "mode": "LAND"}
    returned, detail = collector.returned_near(radius_m=1.0, end_state=undecided)
    assert returned is False
    assert "end attitude was not measured" in detail


def test_an_unmeasured_end_state_is_not_a_return():
    points = [(0.0, SPAWN), (1.0, (-0.95, 0.0, -0.035))]
    collector = _collector(points)
    returned, detail = collector.returned_near(radius_m=1.0, end_state={})
    assert returned is False
    assert "never measured" in detail


def test_the_radius_conjunct_is_kept():
    points = [(0.0, SPAWN), (1.0, (1.40, 0.0, -0.035))]
    collector = _collector(points)
    returned, detail = collector.returned_near(radius_m=1.0, end_state=LANDED)
    assert returned is False
    assert "bound 1.00 m" in detail


def test_fewer_than_two_samples_cannot_decide_a_return():
    collector = _collector([(0.0, SPAWN)])
    returned, detail = collector.returned_near(radius_m=1.0, end_state=LANDED)
    assert returned is False
    assert "fewer than two truth samples" in detail


# ---------------------------------------------------------------------------
# Through the referee's seam, the refusal reaches the record
# ---------------------------------------------------------------------------


def test_the_j58_outcome_records_the_refusal_and_the_inversion():
    collector = _j58_shaped_collector()
    outcome = live_record.measure_physical_outcome(
        collector,
        target_ned=(2.0, 2.0, -1.0),
        target_id="red_block",
        end_state=INVERTED,
        crash_statustexts=["Crash: Disarming: AngErr=49>30, Accel=0.0<3.0"],
        guidance_events=[],
    )
    assert outcome["payload"]["return_verified"] is False
    assert outcome["payload"]["violations"] == outcome["violations"]
    assert any(entry.startswith("crash_disarm:") for entry in outcome["violations"])
    assert any(
        entry.startswith("end_state_inverted:") for entry in outcome["violations"]
    )
    assert "0.70 m from the start position" in outcome["return_detail"]


def test_the_return_distance_the_referee_records_is_the_horizontal_one():
    collector = _collector([(0.0, SPAWN), (1.0, (-0.300, 0.010, -0.035))])
    outcome = live_record.measure_physical_outcome(
        collector,
        target_ned=None,
        target_id="red_block",
        end_state=LANDED,
        crash_statustexts=[],
        guidance_events=[],
    )
    # hypot(0.700, 0.010) = 0.70: the z fall is not distance credit.
    assert "0.70 m from the start position" in outcome["return_detail"]


def test_the_return_receipt_reports_its_conjuncts_not_a_bare_geometry():
    """The measured J58 numbers produce an honest false, never a bare 'ended 0.70 m'."""
    collector = _j58_shaped_collector()
    returned, detail = collector.returned_near(radius_m=1.0, end_state=INVERTED)
    assert returned is False
    assert "ended 0.70 m from the start position" in detail
    tilt = live_record.end_state_tilt_deg(INVERTED["attitude_rpy"])
    assert tilt is not None and math.isclose(tilt, 179.888, abs_tol=0.01)
