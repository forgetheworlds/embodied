"""The controller-side sim/wall clamp's schedule math, without any simulator.

The pacer is the scored path's enforcement of the owner's ruling of 2026-09-30
(APPROVAL-RECORD "F2's denominator"): simulated sensor time is produced at no more
than 1x wall, so a 10 ms publish tick means 10 ms of sensor time. What has to hold
is a property of the schedule, not of Webots: the simulation clock may never advance
faster than the wall clock over ANY window, which forbids Webots' realtime catch-up
sprint (a "sleep when sim leads wall" gate would never fire, because catch-up never
leads the wall). These tests drive the pacer with a scripted clock and sleep, so the
schedule itself is what is judged.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = REPO_ROOT / "scenarios" / "compat" / "controllers" / "compat_vehicle_controller"


def pacer_module():
    spec = importlib.util.spec_from_file_location("compat_sensors", CONTROLLER / "sensors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedClock:
    """A wall clock the test moves by hand; sleep advances it exactly as asked."""

    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make_pacer():
    module = pacer_module()
    clock = ScriptedClock()
    return module.SimWallPacer(2, clock=clock.clock, sleep=clock.sleep), clock


def test_a_fast_step_is_held_to_its_slot():
    # The measured defect this forbids: after a slow stretch Webots runs steps
    # back-to-back to resynchronise with the wall, delivering tens of milliseconds
    # of simulated sensor time inside one wall tick.
    pacer, clock = make_pacer()
    pacer.after_step()  # first step initialises the schedule: release due at 2 ms

    # A catch-up sprint: the step itself took no wall time at all.
    pacer.after_step()
    assert clock.now == pytest.approx(0.002), "held to the wall until 2 ms passed"
    pacer.after_step()
    assert clock.now == pytest.approx(0.004), "held to the wall until 4 ms passed"
    assert pacer.gates == 2


def test_a_late_step_pushes_the_schedule_back_and_no_early_step_recovers_it():
    # A step that overran its slot may push later slots back, but no later step may
    # run early to recover the lost time -- that is the whole difference between a
    # rate clamp and a phase alignment.
    pacer, clock = make_pacer()
    pacer.after_step()  # initialises the schedule: first release due at 2 ms
    pacer.after_step()  # held to 2 ms
    assert clock.now == pytest.approx(0.002)
    assert pacer.gates == 1

    # One step overruns to 7 ms of wall: no sleep, the schedule falls behind.
    clock.now = 0.007
    pacer.after_step()
    assert pacer.gates == 1, "an overrunning step is not gated: it may only fall behind"

    # The very next step tries to sprint at 7.5 ms: held to 9 ms, never to 8 ms --
    # an early release would be the catch-up sprint the clamp exists to forbid.
    clock.now = 0.0075
    pacer.after_step()
    assert clock.now == pytest.approx(0.009)
    assert pacer.gates == 2


def test_over_any_window_the_schedule_never_releases_faster_than_realtime():
    # The clamp's own invariant, checked the way the run measures it: consecutive
    # release times are never closer together than one timestep, whatever mixture of
    # fast, exact and overrunning steps came before.
    pacer, clock = make_pacer()
    pacer.after_step()  # initialises the schedule; releases are counted from here
    releases = [clock.now]
    for jump in (0.0, 0.5, 0.0, 5.0, 0.0, 0.0, 0.25, 0.0):
        clock.now += jump / 1000.0
        pacer.after_step()
        releases.append(clock.now)
    gaps = [later - earlier for earlier, later in zip(releases, releases[1:])]
    assert min(gaps) >= 0.002 - 1e-12


def test_the_document_states_what_was_enforced():
    pacer, clock = make_pacer()
    pacer.after_step()  # initialises the schedule
    clock.now = 0.0005
    pacer.after_step()  # held to 2 ms
    document = pacer.document()
    assert document["enabled"] is True
    assert document["timestep_ms"] == 2
    assert document["gates"] == 1
    assert document["slept_s"] == pytest.approx(0.002)
