"""The mission responds to being blocked by gathering evidence, not by ending.

Two measured defects (J52-discrim-1; work/runs/night/GROUNDING-INVALID-DEPTH.md,
RETURN-UNSUPPORTED.md) silenced the mission's own recovery machinery:

* the query's selection refused ``unknown_geometry`` 19 times — runs of 12 and 6
  consecutive candidate-bearing observations — and the mission never once
  changed view; the refusal was counted and nothing else read it;
* a leftover explore goal stayed installed as `_active_goal` through every
  return step, so the section-12.2 observation that would have re-evidenced the
  start never fired, and the renewal machinery even flew the stale goal during
  the return leases.

The vantage policy here is the fix for the first: at the declared consecutive
refusal count, the runtime spends its section-12.2 observation sweep toward the
selection's bearing — once per standing view, re-armed only by a new admitted
vantage — and records the sweep's honest terminal ("not measurable from here")
when it refuses again. The stale-goal demotion is the fix for the second: a goal
installed at ``step`` entry belongs to a finished step and is demoted, unless it
is a previous attempt of this same step's own target.

The stubs follow the established idiom (``test_frontier_excursion.py``): the
methods under test need a store-shaped runtime and a position, not a simulator.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from embodied.platform import mission_runtime as MR
from embodied.perception import camera as camera_module
from embodied.perception import grounding as G
from embodied.perception.detector import injected_candidate


def _selection_bearing_of(u: float) -> float:
    """The commanded bearing the runtime derives for a selection at pixel u."""
    intrinsics = camera_module.build_calibration().left_intrinsics
    return MR.MISSION_YAW_HOLD_RAD + math.atan2(
        u - float(intrinsics.principal_point_px[0]),
        float(intrinsics.focal_length_px[0]),
    )


class _StubRuntime:
    """The runtime surface the vantage policy and the step world read."""

    def __init__(self):
        self.calibration = camera_module.build_calibration()
        self.result = SimpleNamespace(log=[])
        self._query = "red block"
        self._refusals_log: list[str] = []
        self._perception_refusal_counts: dict[str, int] = {}
        self._candidate_targets: list[str] = []
        self._candidate_observation_ids: list[str] = []
        self._blocked_refs: dict[str, str] = {}
        self._observed_targets: set[str] = set()
        self._observation_counter = 0
        self._observation_ids: list[str] = []
        self._active_goal = None
        self._visual_fault_reason = None
        self._unmeasurable_streak = 0
        self._unmeasurable_bearing_rad = MR.MISSION_YAW_HOLD_RAD
        self._vantage_sweep_spent = False
        self.observe_calls: list[dict] = []
        self.settings = SimpleNamespace(realtime_ratio_envelope=(0.5, 1.5))
        self._stats = SimpleNamespace(
            sim_clock=SimpleNamespace(newest_s=100.0),
        )

    # -- perception-side surface -------------------------------------------------

    def _sink(self, *args, **kwargs):
        return None

    def _clock(self):
        return "clock"

    def _note_perception_refusal(self, line: str) -> None:
        self._refusals_log.append(line)
        self._perception_refusal_counts[line] = (
            self._perception_refusal_counts.get(line, 0) + 1
        )

    # -- motion-side surface ------------------------------------------------------

    def _position_odom(self):
        return None

    def _beat_watchdog(self):
        return None

    def perceive_if_due(self):
        return None

    def publish_active(self):
        return "no_active_goal"

    def observe_in_place(self, **kwargs):
        self.observe_calls.append(kwargs)
        return True, "observe-stub"

    # The real policy methods, bound from the class: the stub carries exactly the
    # attributes they read, so the code under test is the shipped code.
    _ground_candidates = MR.MissionRuntime._ground_candidates
    _selection_bearing = MR.MissionRuntime._selection_bearing
    _note_unmeasurable_selection = MR.MissionRuntime._note_unmeasurable_selection
    _reset_unmeasurable_streak = MR.MissionRuntime._reset_unmeasurable_streak
    _note_new_vantage = MR.MissionRuntime._note_new_vantage
    _consume_vantage_sweep = MR.MissionRuntime._consume_vantage_sweep
    _renewal_hold_position = MR.MissionRuntime._renewal_hold_position


class _StubObservation(SimpleNamespace):
    pass


def _observation(sequence: int = 3):
    return _StubObservation(
        sequence=sequence, record_id=f"ep-obs-{sequence:05d}", sim_time_s=None
    )


@contextmanager
def _grounding_refuses(reason: str = G.REFUSAL_UNKNOWN_GEOMETRY):
    """The grounding seam refuses every call with the given reason."""
    original = MR.G.ground
    MR.G.ground = lambda *args, **kwargs: G.Refusal(
        reason=reason,
        detail="none of the 1 selected samples carries valid depth",
    )
    try:
        yield
    finally:
        MR.G.ground = original


@contextmanager
def _grounding_returns_object():
    """The grounding seam grounds every call (a dummy target object)."""
    original = MR.G.ground
    MR.G.ground = lambda *args, **kwargs: object()
    try:
        yield
    finally:
        MR.G.ground = original


def _ground_candidate(runtime: _StubRuntime, sequence: int = 3) -> None:
    runtime._ground_candidates(
        (injected_candidate("injected-0", (60.0, 40.0, 140.0, 100.0)),),
        _observation(sequence),
        depth=None,
        pose=None,
        state=None,
    )


# ---------------------------------------------------------------------------
# Defect 1: persistent unknown_geometry changes the view
# ---------------------------------------------------------------------------


def test_one_refusal_does_not_spend_the_sweep():
    """A single refusing cycle is not a vantage verdict: no sweep is due."""
    runtime = _StubRuntime()
    with _grounding_refuses():
        _ground_candidate(runtime)
    assert runtime._unmeasurable_streak == 1
    runtime._consume_vantage_sweep()
    assert runtime.observe_calls == []
    assert not runtime._vantage_sweep_spent


def test_the_declared_count_triggers_one_sweep_at_the_selections_bearing():
    """Two consecutive refusals sweep ONCE, started at the selection's bearing.

    J52's pathology was the opposite: 19 selections, every one from the same
    standing view, none answered by anything but the next selection.
    """
    runtime = _StubRuntime()
    # The selection sits left of the image centre, as J52's sliver did
    # (centre u = 86 of 640), so its bearing is a negative offset from the
    # hold yaw in the commanded frame.
    with _grounding_refuses():
        _ground_candidate(runtime)
        _ground_candidate(runtime, sequence=4)
    assert runtime._unmeasurable_streak == MR.VANTAGE_REFUSAL_SWEEP_CYCLES
    runtime._consume_vantage_sweep()
    assert len(runtime.observe_calls) == 1
    call = runtime.observe_calls[0]
    assert call["target_ref"] is None
    assert call["bearing_rad"] == pytest.approx(_selection_bearing_of(100.0))
    assert call["bearing_rad"] < MR.MISSION_YAW_HOLD_RAD
    assert "unknown_geometry" in call["refused_reason"]
    # Spent: further refusals from this standing view sweep no more.
    with _grounding_refuses():
        _ground_candidate(runtime, sequence=5)
    runtime._consume_vantage_sweep()
    assert len(runtime.observe_calls) == 1
    assert runtime._vantage_sweep_spent


def test_a_grounding_resets_the_streak():
    """A measurable view answers the question the streak was asking."""
    runtime = _StubRuntime()
    with _grounding_refuses():
        _ground_candidate(runtime)
        _ground_candidate(runtime, sequence=4)
    assert runtime._unmeasurable_streak == 2
    with _grounding_returns_object():
        _ground_candidate(runtime, sequence=5)
    assert runtime._unmeasurable_streak == 0


def test_a_different_refusal_reason_resets_the_streak():
    """The declared count counts consecutive unknown_geometry, not refusals."""
    runtime = _StubRuntime()
    with _grounding_refuses():
        _ground_candidate(runtime)
    assert runtime._unmeasurable_streak == 1
    with _grounding_refuses(G.REFUSAL_STALE_POSE):
        _ground_candidate(runtime, sequence=4)
    assert runtime._unmeasurable_streak == 0


def test_a_new_admitted_vantage_rearms_the_sweep():
    """The sweep's only re-arm is a new vantage — a goal admitted and flown."""
    runtime = _StubRuntime()
    with _grounding_refuses():
        _ground_candidate(runtime)
        _ground_candidate(runtime, sequence=4)
    runtime._consume_vantage_sweep()
    assert runtime._vantage_sweep_spent
    runtime._note_new_vantage()
    assert not runtime._vantage_sweep_spent
    assert runtime._unmeasurable_streak == 0
    # The next standing view may spend its own sweep after the count rebuilds.
    with _grounding_refuses():
        _ground_candidate(runtime, sequence=5)
    runtime._consume_vantage_sweep()
    # One refusing cycle is still not the count: nothing new swept.
    assert len(runtime.observe_calls) == 1
    with _grounding_refuses():
        _ground_candidate(runtime, sequence=6)
    runtime._consume_vantage_sweep()
    assert len(runtime.observe_calls) == 2


def test_the_honest_terminal_is_recorded_when_the_sweep_refuses():
    """A refusal after a real look is a valid outcome, recorded as one."""
    runtime = _StubRuntime()
    with _grounding_refuses():
        _ground_candidate(runtime)
        _ground_candidate(runtime, sequence=4)
    runtime._consume_vantage_sweep()
    assert any("not measurable from here" in line for line in runtime.result.log)
    assert any("explore continues" in line for line in runtime.result.log)


def test_the_bearing_is_the_selections_offset_from_the_principal_point():
    """The image geometry is the only place a refused selection has a bearing."""
    runtime = _StubRuntime()
    selection = SimpleNamespace(geometry=(86.0, 470.0))
    assert runtime._selection_bearing(selection) == pytest.approx(
        _selection_bearing_of(86.0)
    )


# ---------------------------------------------------------------------------
# Defect 2(a): the stale-active-goal gate
# ---------------------------------------------------------------------------


def _spatial_goal(intent: str, target_ref: str) -> MR.R.SpatialGoal:
    return MR.R.SpatialGoal(
        proposal_id=f"{intent}-{target_ref}-1",
        request_id=None,
        fingerprint="fp",
        mission_revision=1,
        base_goal_revision=0,
        selection_ids=(),
        target_refs=(target_ref,),
        intent=intent,
        constraints=(),
        completion_condition="reached",
        lease_bounds=(("step_lease_s", 2.0),),
        local_discretion_bounds=(),
    )


def _active_goal(intent: str, target_ref: str, goal_id: str):
    return MR._ActiveGoal(
        goal_id=goal_id,
        proposal=_spatial_goal(intent, target_ref),
    )


def _step_world(runtime: _StubRuntime) -> MR._LiveRunnerWorld:
    return MR._LiveRunnerWorld(runtime)


def test_a_fresh_step_observes_despite_a_leftover_active_goal():
    """The J52 defect: a finished step's goal must not silence the next step.

    The return step's own admission was refused, the explore goal is still
    installed — the demotion clears it and the observation machinery runs for
    the refused target.
    """
    runtime = _StubRuntime()
    runtime._active_goal = _active_goal(
        "explore", "frontier:8:7:2", "cert-explore-frontier:8:7:2"
    )
    runtime._blocked_refs["start"] = "unsupported_space"
    world = _step_world(runtime)
    world.step("return", "start")
    assert runtime._active_goal is None
    assert "start" in runtime._observed_targets
    assert len(runtime.observe_calls) == 1
    assert runtime.observe_calls[0]["target_ref"] == "start"
    assert any("demoted installed goal" in line for line in runtime.result.log)
    assert any("cert-explore-frontier:8:7:2" in line for line in runtime.result.log)


def test_a_legitimately_executing_goal_is_untouched():
    """A goal for THIS step's own target is never demoted nor observed over.

    The stale blocked-map entry (a place target's earlier refusal) must not
    cancel the goal this step just admitted and is flying.
    """
    runtime = _StubRuntime()
    own = _active_goal("return", "start", "cert-return-start")
    runtime._active_goal = own
    runtime._blocked_refs["start"] = "unsupported_space"
    world = _step_world(runtime)
    world.step("return", "start")
    assert runtime._active_goal is own
    assert runtime.observe_calls == []
    assert "start" not in runtime._observed_targets
    assert not any("demoted installed goal" in line for line in runtime.result.log)


def test_a_previous_attempt_of_the_same_step_is_not_foreign():
    """Attempt 2 of a step may keep attempt 1's goal for the same target."""
    runtime = _StubRuntime()
    own = _active_goal("explore", "frontier:8:7:2", "cert-explore-frontier:8:7:2")
    runtime._active_goal = own
    world = _step_world(runtime)
    world.step("explore", "frontier:8:7:2")
    assert runtime._active_goal is own
    assert not any("demoted installed goal" in line for line in runtime.result.log)


def test_an_observation_objective_left_installed_is_foreign():
    """A goal with no proposal is not this step's: demoted, never published."""
    runtime = _StubRuntime()
    stray = MR._ActiveGoal(goal_id="observe-9", hold_position_odom=(0.0, 0.0, -1.5))
    runtime._active_goal = stray
    world = _step_world(runtime)
    world.step("return", "start")
    assert runtime._active_goal is None
    assert any("observe-9" in line for line in runtime.result.log)


# ---------------------------------------------------------------------------
# Defect 2(b): the refused-renewal hold keeps the aircraft flying
# ---------------------------------------------------------------------------


class _StubAlignment:
    def __init__(self, origin):
        self.sealed = True
        self._origin = origin

    def aligned_position_ned(self, point):
        return self._origin


def test_a_refused_renewal_holds_at_the_declared_hover_altitude():
    """J52's park-and-disarm: the hold pinned the aircraft at its descended z.

    The hold now climbs the vertical target to the declared cruise altitude
    (the aircraft's own x, y are unchanged — a hold does not translate).
    """
    runtime = _StubRuntime()
    runtime.alignment = _StubAlignment(origin=(-1.0, 0.0, -0.09))
    runtime.settings.hover_altitude_m = 1.5
    hold = runtime._renewal_hold_position((0.17, 0.31, -0.15))
    assert hold == (0.17, 0.31, -0.09 - 1.5)


def test_the_hold_never_commands_a_descent():
    """An aircraft already above the declared altitude holds where it is."""
    runtime = _StubRuntime()
    runtime.alignment = _StubAlignment(origin=(-1.0, 0.0, -0.09))
    runtime.settings.hover_altitude_m = 1.5
    hold = runtime._renewal_hold_position((0.17, 0.31, -2.50))
    assert hold == (0.17, 0.31, -2.50)


def test_an_unsealed_alignment_holds_at_the_current_position():
    """Without a frame the declared altitude is inexpressible: hold, don't guess."""
    runtime = _StubRuntime()
    runtime.alignment = _StubAlignment(origin=(-1.0, 0.0, -0.09))
    runtime.alignment.sealed = False
    runtime.settings.hover_altitude_m = 1.5
    assert runtime._renewal_hold_position((0.17, 0.31, -0.15)) == (0.17, 0.31, -0.15)


# ---------------------------------------------------------------------------
# The phase loop: a blocked phase does not end the flight while phases remain
# ---------------------------------------------------------------------------


def _full_runtime(tmp_path):
    """The offline runtime, built the way the integration suite builds it."""
    from embodied.bench.recorder import Recorder
    from embodied.platform.localization_check import (
        _load_localization_config,
        _platform_settings,
    )

    repository = MR.repository_root()
    document = _load_localization_config(repository / "configs" / "first_indoor.yaml")
    settings = _platform_settings(document, repository)
    episode_dir = tmp_path / "episode"
    recorder = Recorder(episode_dir)
    return MR.MissionRuntime(
        settings=settings,
        config_document=document,
        episode_dir=episode_dir,
        evidence_dir=tmp_path / "platform",
        recorder=recorder,
        instruction="Find the red block, inspect it, and return to the start.",
        target_id="red_block",
        episode_id="vantage-policy-1",
    )


def test_a_blocked_explore_does_not_end_the_flight_while_phases_remain(
    tmp_path, monkeypatch
):
    """The loop must route blocked to the next phase and land once, at the end.

    J52's explore phase blocked through every attempt; the mission went on to
    inspect and to four return attempts, and the one `_land` ran only after the
    phases were consumed. This pins that routing: a blocked explore terminal
    with phases remaining never reaches the protective landing, and the return
    step's observation machinery runs against the refused start target.
    """
    runtime = _full_runtime(tmp_path)
    land_calls: list[str] = []
    monkeypatch.setattr(
        runtime, "_land", lambda drain: land_calls.append("land"), raising=True
    )
    monkeypatch.setattr(
        runtime, "pump_perception", lambda *args, **kwargs: 1, raising=True
    )
    monkeypatch.setattr(
        runtime, "navigable_frontiers", lambda: ("frontier:8:7:2",), raising=True
    )
    runtime.result.termination_reason = "mission_completed"
    runtime._fly_the_mission(lambda: None)

    statuses = [phase.status for phase in runtime.result.phases]
    # Explore ran its initial attempt plus its three declared regather retries;
    # inspect skipped by its guard (nothing grounded); return ran its own
    # retries. Every phase ran — the blocked explore did not end the flight.
    assert statuses == ["blocked"] * 4 + ["completed"] + ["blocked"] * 4
    # The flight ended once, after every phase — not at the blocked explore.
    assert land_calls == ["land"]
    assert runtime.result.termination_reason == "mission_completed"
    # The return step's observation machinery ran for the refused start target:
    # the section-12.2 gate consumed it (offline there is no estimator pose, so
    # the objective itself reports nothing and the gate records the attempt).
    assert "start" in runtime._observed_targets
    # The explore target was observed once too: its admission was refused and
    # the observation machinery answered it instead of re-selecting silently.
    assert "frontier:8:7:2" in runtime._observed_targets
    # Each step ran grounded-but-running: it published nothing (no goal of its
    # own was admitted offline) and reported so, rather than ending the flight.
    assert any(
        "return start: publication stopped: no_active_goal" in line
        for line in runtime.result.log
    )
