"""The live mission runtime: one process that flies and records a first-indoor mission.

This module is the agent side of a recorded episode. It joins the merged
components instead of reimplementing any of them:

* the platform (``webots_ardupilot``): Webots + ArduPilot SITL, the sensor
  stream, telemetry, and the **single setpoint publisher**
  (``WebotsArduPilot.send_local_ned``) — every motion target this runtime puts
  on the wire goes through :meth:`MissionRuntime.publish_active`, which builds
  the target and calls that one method and nothing else;
* the estimator seam (``platform.localization``): the pinned OpenVINS feed,
  the health machine and the external-navigation publisher, driven as
  ``platform.localization_check`` drives them — including the declared ordered
  bring-up, imported from there rather than rebuilt;
* perception (``perception.camera``/``grounding``) and memory
  (``memory.world``): validity-qualified stereo depth integrated into the
  MapStore, so unknown space is never free and invalid depth clears nothing;
* navigation (``executor``/``planner``/``validator``): admission's staged
  assessment plans a certified trajectory, and the executor's own
  ``completion_status`` decides arrival; the planner is re-run (certificate
  renewal, specification 15.1) whenever the map advances, so no sample is
  published against a stale dependency;
* the pilot (``pilot.broker``/``recipe_runner``): the admission protocol and
  the shared recipe runner, with ``pilot.mission``'s B0 policy as executive.

Truth isolation is structural: this module dispatches sensor records for PAIR
(estimator feed and its own perception) and IMU (estimator feed), hands every
record to the ``sensor_tap`` the bench side owns, and keeps POSE records only
as the bring-up's own end-state accounting — the same
:class:`~embodied.platform.localization_check._FeedStats` lists the worked
example fills, read by the bring-up's declared P01-L gate and by nothing else
here. No truth value reaches the estimator's input, the map, a plan, an
admission or a published setpoint.

Frame and clock conventions. The mission's ``odom`` frame is the **aligned
local-NED frame**: the estimator's own initialization frame is not used
directly, because its yaw is unobservable at initialization (OpenVINS's
alignment picks it from accelerometer noise, the b477ee7 permutation). Every
pose, every map cell and every plan is expressed in the frame the frozen
:class:`~embodied.platform.localization.OdomAlignment` produces — the same
transform the external-navigation publisher seals and sends — so the map, the
published setpoints and the evaluator's own truth positions are all in one
frame, and a map bounding box means what it says about the scene. This is the
specification's own definition of odom (4.2): continuous, gravity-aligned,
horizontal axes fixed at startup, not necessarily pointing north.

The one time base for trajectory sampling is the host's monotonic clock, which
is also the clock the controller stamps its frames on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import faulthandler
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from embodied.cli import repository_root
from embodied.contracts import records as R
from embodied.memory import world as world_module
from embodied.navigation import executor as EX
from embodied.navigation import geometry as GE
from embodied.navigation import planner as PL
from embodied.perception import camera as camera_module
from embodied.perception import grounding as G
from embodied.perception.detector import DetectorUnavailable
from embodied.platform import localization as loc
# The worked example's own machinery, imported rather than rebuilt: the
# declared ordered bring-up, the simulator-second window, the estimator
# process and the pre-arm checks are the ones P01-L measured with.
from embodied.platform.localization_check import (
    BRING_UP_GCS_CONNECT_TIMEOUT_S,
    BRING_UP_GCS_SYSTEM_PARAMETER,
    FEED_STREAM_POLL_S,
    GPS_AIDING_SAMPLE_HZ,
    MSG_ID_GPS_RAW_INT,
    MSG_ID_SYS_STATUS,
    PAIR_QUEUE_FRAMES,
    PARAMETER_READ_TIMEOUT_S,
    VEHICLE_REQUIREMENTS,
    _autopilot_feed_endpoint,
    OrderedPairFeed,
    _crash_disarm_statustexts,
    _declared_start_attitude,
    _declared_start_origin,
    _gps_aiding_verdict,
    _param_error_refusals,
    _readback_blockers,
    _run_ordered_bring_up,
    _sim_window_wall_ceiling_s,
    _start_estimator,
    _wait_initialized,
    _FeedStats,
    _SimWindow,
)
from embodied.platform.sensors import SensorSample, capture_latency_ns, sim_time_ns
from embodied.platform.webots_ardupilot import (
    BringUpLink,
    EvidenceWriter,
    Kind,
    LocalNedTarget,
    LowPrioritySubprocessRunner,
    PlatformSettings,
    PymavlinkSession,
    TcpSensorGateway,
    WebotsArduPilot,
    baro_relative_altitude_m,
    measure_frame_quality,
    ppm_bytes,
)
from embodied.pilot import mission as mission_module
from embodied.pilot.broker import AdmissionContext, PilotBroker
from embodied.pilot.decisions import PilotParameters, SceneStatus
from embodied.pilot.recipe_runner import RecipeRunner

# ---------------------------------------------------------------------------
# Declared engineering parameters (R2). Each names what it was chosen against.
# ---------------------------------------------------------------------------

# Map values are the P03 slice's declared values (tests/fixtures/navigation/
# doorway/truth/scene.json); the bounds are widened to the live scene, whose
# spawn sits in a vestibule rather than at the fixture's origin.
# The bounds are the scene's own extents in the aligned NED frame (the world's
# floor is 12.4 x 5.4 m centred at (4, 0) with the vehicle spawning at
# (-1, 0, 0.09) and NED z pointing down), plus a voxel of margin.
# The z pair is the room's height and the margin past it, written down rather
# than left implicit because this is the bound a climbing vehicle runs into:
# NED z points down, the scene's ceiling sits at 2.55 m above its floor and the
# floor plane at -0.02 m (scenarios/first_indoor/world.wbt), so -2.8 admits the
# ceiling with 0.25 m to spare and +0.4 admits the small negative excursions a
# landed vehicle's own estimate reports. A bound tighter than the ceiling would
# clip the map exactly where a vehicle flying into it needs evidence.
MAP_PARAMETERS = dict(
    voxel_m=0.1,
    bounds_odom_m={"x": (-2.4, 10.6), "y": (-3.2, 3.2), "z": (-2.8, 0.4)},
    surface_band_m=0.1,
    log_odds_hit=0.7,
    log_odds_pass=-0.4,
    clamp=4.0,
    free_threshold=1.4,
    occupied_threshold=1.4,
    min_clearing_rays=3,
    freshness_s=5.0,
)
# The error allowance is the live estimator's measured bound, and it is the
# rig's floor rather than a conservative choice. The validator refuses any state
# whose 3*sigma exceeds it, so the allowance IS the largest 3-sigma pose the
# mission will act on — and no measured pose is better than it.
#
# Measured from the estimator's own feed, not assumed: across 4 087 in-flight
# samples in work/runs/p01-localization/p01l-streak-4-20260930T031611Z/run-a the
# smallest published sigma is 0.0500 m and the median 0.0512, and the best run
# on disk (run-2026-09-28T02-43-27-442Z) is no better at p50 0.0512. The
# declared E1 bound is p95 0.10 m, which is consistent with it.
#
# 3 * 0.05 = 0.15 m therefore admits every healthy pose this rig produces while
# still refusing the H4 outage bound (sigma up to 1.0 m). It is written as the
# product it is, so the value cannot drift away from its derivation (R2).
#
# This is the clearance inflation, a different quantity from R23's goal-search
# margin in planner.py, and the two compose: a cell must hold a ball of
# body + allowance + margin = 0.475 m to carry a certificate. R37 measured that
# composition against the vestibule doorway and found the allowance is NOT its
# binding term — see work/runs/p05/J37-allowance-REPORT.md.
ESTIMATOR_HEALTHY_SIGMA_MAX_M = 0.05
ERROR_ALLOWANCE_M = 3.0 * ESTIMATOR_HEALTHY_SIGMA_MAX_M
# The sensor's near blind field. The matcher cannot report depth inside it, so no
# free-space evidence can exist there in any direction, and a clearance ball around
# any cell within about half a metre of the camera reaches space the sensor provably
# cannot see. The self-occupied exemption is derived from this rather than from the
# envelope (GE.Envelope.self_occupied_radius_m), because the exemption exists for
# what the vehicle's own presence blinds, and the envelope is not that quantity.
#
# It mirrors the declared depth window's near bound —
# configs/first_indoor.yaml calibration.bounds.depth_range_m[0]. The mirror is pinned
# by tests/platform/test_near_field_clearance.py so the two cannot drift apart.
#
# Measured, J44-fly-1: with the exemption at the envelope radius, 0 of 26 neighbours
# were searchable and every one of the mission's 36 setpoints was a station hold.
# 83 % of the ball disqualifiers sat within this distance of the camera.
SENSOR_NEAR_LIMIT_M = 0.5
ENVELOPE = GE.Envelope(
    body_radius_m=0.3,
    error_allowance_m=ERROR_ALLOWANCE_M,
    sensor_near_limit_m=SENSOR_NEAR_LIMIT_M,
)
PLAN_CONFIG = PL.PlanConfig(
    limits=PL.PlanLimits(),
    inflation_m=0.4,
    start_state_tolerance_m=0.2,
    setpoint_prefix_horizon_s=2.0,
    setpoint_period_s=0.1,
)
# Frontier cells are clustered into one excursion target per 0.6 m block, so
# the neighbouring boundary cells of one doorway are one frontier, not eight.
FRONTIER_CLUSTER_CELLS = 6
# How a frontier becomes a flyable goal. A frontier is a free cell touching
# unknown space, so the approach region built around it can overlap space the map
# has no evidence for, and the planner refuses that correctly: a larger margin
# cannot turn unseen space into measured free space (specification 7.1). The
# vantage is therefore walked back toward the aircraft, which stands in free space
# by construction, until that region is supported. The direction matches the
# executor's own, so the region this search clears is the region the planner
# checks.
FRONTIER_VIEW_DIRECTION = (1.0, 0.0, 0.0)
#
# THE EXCURSION STANDOFF — which is NOT the approach standoff. ``GE.STANDOFF_M``
# (1.0 m) is how far the executor comes to rest before a doorway or the object it
# inspects, and lowering *that* would fly the aircraft to within 10 cm of the
# thing it is looking at. That is a different change and was not ruled on. This
# constant answers only the excursion question: how far a goal's region must be
# from the aircraft to be a view taken from somewhere else rather than a hover.
#
# It is derived, because the value it replaces (1.0 m, whose own config comment
# says "provenance unknown") could not be satisfied. The space the planner may
# search is set by the self-occupied exemption, and a cell is searchable only if
# its whole clearance ball is free. Measured on J47-nearfield-1: the live map's
# *evidence*-supported clear space is nil — the run reported one such cell — so
# every searchable cell is an exemption cell, and the deepest of them sits
#
#     self_occupied_radius_m - (inflation_m + voxel * CERTIFICATE_MARGIN_VOXELS)
#       = (0.5 + sqrt(3) * 0.1) - (0.45 + 0.1 * 0.25)
#       = 0.673 - 0.475
#       = 0.198 m
#
# from the aircraft. A goal region of half-extent 0.55 m must contain one of those
# cells while sitting at least `standoff` away, so the standoff can be at most
# 0.198 m. Half of that leaves the search a band to land in.
EXCURSION_STANDOFF_M = 0.10
#
# And the walk must be fine enough to land in that band, which is
# `0.198 - 0.10 = 0.098 m` wide. At the old 0.2 m step the walk stepped clean over
# it at every standoff — which is why lowering the standoff alone would not have
# moved the aircraft. A quarter of the shell is finer than the band, and a finer
# walk is a stricter search, never a weaker one.
FRONTIER_VANTAGE_STEP_M = 0.05
# The mission's own budget, in the simulator's seconds (the clock the bring-up
# and route windows spend, owner ruling 2026-09-28).
MISSION_BUDGET_SIM_S = 300.0
# Per-action leases: how long one recipe step may own the aircraft before the
# attempt is declared blocked. Simulator seconds.
ACTION_LEASE_SIM_S = {"explore": 60.0, "inspect": 45.0, "return": 60.0}
# Publication cadence for a certified prefix's samples (wall seconds; it paces
# the link, not the aircraft — the 50 ms the worked example re-sends at).
SETPOINT_PERIOD_S = 0.05
SETPOINT_DEADLINE_S = 0.5
# A yaw the mission holds on every published target. Declared rather than left
# to the firmware: an unspecified yaw resolves to LOOK_AT_NEXT_WP
# (LEARNED-FAILURES defect 14) and commands a heading nobody asked for.
MISSION_YAW_HOLD_RAD = 0.0
# Perception cadence: stereo depth, candidate proposal and map integration on
# the newest pair, at most this often (wall seconds). The estimator feed is on
# its own thread and never waits behind this.
MIN_PERCEPTION_INTERVAL_S = 0.3
# How often the perception pump checks its clock while it is gathering. This is
# a host-thread yield, not a rate: the rate perception actually runs at is
# MIN_PERCEPTION_INTERVAL_S, and this only decides how promptly the pump
# notices that a cycle is due.
PERCEPTION_PUMP_SLEEP_S = 0.02
# The most observation ids a step may cite. A bounded citation list keeps a
# long mission's report readable without letting a claim grow without limit.
MAX_CITED_OBSERVATIONS = 256
# How many times one blocked phase may gather again before the mission moves on.
# Retrying without a bound is what spent the whole 300 s mission budget inside
# `explore` on live-vision-1 and starved `inspect` and `return`, so the mission
# never returned and its own report said so. The mission may look again; it may
# not look so long that the later phases lose their budget.
MAX_PHASE_REGATHERS = 3
# One recorded commanded setpoint per this much SIMULATOR time, and the most
# that are kept. A mission publishes tens of times a second and records only a
# COUNT, so a reader could not see whether the aircraft was told to climb; one
# sample per second puts the commanded vertical target beside the achieved
# altitude without turning the run record into a setpoint log.
COMMANDED_SAMPLE_SIM_S = 1.0
MAX_COMMANDED_SAMPLES = 512
# The runtime model configuration, and the environment variable its pinned
# route authenticates with. Both are named here rather than inlined so a
# reader can see that the cloud arms read the same declared file, and the same
# declared key, that the probe does: the platform configuration this runtime is
# otherwise driven by carries no ``model`` section at all.
RUNTIME_MODEL_CONFIG_RELATIVE = "configs/runtime-model.yaml"
CLOUD_API_KEY_ENV = "COMMAND_CODE_API_KEY"
# The cold-start perception window: bounded, in simulator seconds, before the
# mission's phases begin. It exists to break the bootstrap on the mission's
# first flight — the map a frontier is resolved from can only be built from the
# aircraft's own frames, so with an empty store no goal can resolve a target and
# no step ever runs. Eight seconds is this stage's declared engineering
# parameter (R2): long enough for the first frames to arrive, be depth-validated
# and integrated, and short enough that an aircraft which sees nothing useful
# says so well inside the mission budget.
COLD_START_PERCEPTION_SIM_S = 8.0
# Certificate renewal cadence: the planner is re-run against the newest map at
# most this often (wall seconds). The cost is real (the P03 inflation pass runs
# over the whole grid: ~70 ms at this scene's declared bounds), so renewal is
# bounded rather than attempted on every perception cycle; the stored
# certificate names the map revision it was certified against, and the
# validator's dependency rule is what makes a stale one refused.
RENEWAL_PERIOD_S = 1.0
# An observation whose frames become episode payloads — the evidence a claim
# can cite — is one that grounded a candidate, bounded by this many payloads.
MAX_EVIDENCE_PAYLOADS = 48
# Arrival: inside the terminal region at or under this speed, held this long
# (specification 13.3; the settle values are the executor's declared ones).
SETTLE_SPEED_MPS = 0.15
SETTLE_HOLD_S = 1.0
# The post-LAND drain keeps the feed alive while the aircraft comes down.
LANDING_DRAIN_SIM_S = 12.0
POSE_RING = 64
# The executor's approach region stands off its target point by this much
# along -x (geometry.STANDOFF_M). The return-to-start goal compensates for it
# so the terminal region lands on the start position rather than one metre
# short of it; the frontier and inspect goals use the standoff as intended
# (stop before the unknown boundary; stand off the object).
RETURN_STANDOFF_COMPENSATION_M = GE.STANDOFF_M
# ---------------------------------------------------------------------------
# The mission-loop watchdog: supervision from outside the loop
# ---------------------------------------------------------------------------
#
# Tonight's failure mode (J50-move-1, 2026-10-04): the mission loop's main
# thread entered the pre-admission explore machinery after the cold start and
# never came back. Every declared bound — the mission window, the action
# leases, the regather caps, the budget checks in the recipe runner — is
# consulted BETWEEN computations, never inside them, so a computation that
# does not return holds the whole run: the last agent event stood at sim
# 50.2 s while the simulator, the controller, SITL and the estimator all kept
# running for 23 more minutes, and nothing ended the run until a human killed
# it. The runner blocked with no reason on disk at all: no receipt, no
# mission.json, no shutdown record.
#
# The watchdog is deliberately NOT a check the mission loop runs. It is a
# thread that watches the loop's own heartbeat — a beat wherever a healthy
# loop turns (the bring-up's drain, every perception cycle, every lease-loop
# and phase-loop iteration) — and when no beat arrives inside the declared
# window it (1) writes the stall, with every thread's stack, into the run's
# own artifacts, (2) names the termination, and (3) breaks the loop with an
# asynchronous exception so the run's normal shutdown and receipt still
# happen. It therefore answers the two failure modes of that night in one
# mechanism: silence now has a detector, and a hang now has a name.
#
# The bound is wall seconds because a stalled loop stops spending every other
# clock: the simulator's own clock kept advancing for the whole 23 minutes of
# tonight's hang, so no sim-clock budget could see it. Its derivation:
#   * the beats are dense in a healthy run — drain turns at >= 5 Hz, the
#     lease loop at 200 Hz, perception cycles measured ~0.5 s apart on
#     J50-move-1 itself (25 observations in ~13 s of wall);
#   * the largest measured LEGITIMATE single silent call is map integration
#     at 4.31 s (J24, pre-fix; 0.218 s since) and the cloud transport's
#     declared 30 s read timeout on the one blocking HTTP round trip a
#     B1/B2 ground call makes;
#   * 90 s clears the worst legitimate silence by 3x and cuts tonight's
#     1380 s hang at 1/15th of its cost.
MISSION_LOOP_STALL_S = 90.0
# How often the watchdog inspects the heartbeat. A poll, not a deadline: the
# bound above decides, this only decides how promptly.
MISSION_WATCHDOG_POLL_S = 1.0
# The exit code of the last resort, recorded here so a reader of a shell
# transcript can trace an abrupt exit back to this module's artifact.
MISSION_STALL_EXIT_CODE = 73


class MissionLoopStalled(BaseException):
    """Raised into the mission loop's own thread by the watchdog.

    Deliberately NOT an ``Exception``: the loop's legitimate handlers catch
    ``Exception`` (a failed depth product, a refused grounding, a feed error),
    and a stall must not be absorbed by a handler that then continues the
    loop it was supposed to stop.
    """


class _MissionLoopWatchdog:
    """Watches the mission loop's heartbeat and ends the run when it stops.

    The clock and sleep are injectable (a watchdog test must not measure the
    real machine — LEARNED-FAILURES T18). ``fire`` runs on the watchdog
    thread when the bound is passed once: it writes the run's own record of
    the stall and raises ``MissionLoopStalled`` into ``thread_id``. If the
    process is still alive one FULL bound later the beat never resumed, the
    asynchronous exception was not delivered (or did not unwind), and
    ``escalate`` ends the process from outside: the artifact is already on
    disk, so an abrupt exit still leaves a named cause — which a plain
    wall-clock kill never did. A delivery that lands in a loop already
    unwinding is absorbed by run()'s own handlers after it names the stall.
    """

    def __init__(
        self,
        *,
        thread_id: int,
        fire: Callable[[float], None],
        escalate: Callable[[float], None],
        stall_s: float = MISSION_LOOP_STALL_S,
        poll_s: float = MISSION_WATCHDOG_POLL_S,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._thread_id = thread_id
        self._fire = fire
        self._escalate = escalate
        self._stall_s = stall_s
        self._poll_s = poll_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._stop = threading.Event()
        self._last_beat_s = self._monotonic()
        self._fired = False
        self._fired_age_s: float | None = None
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        """One turn of the mission loop. Called by the loop itself, on its own thread."""
        self._last_beat_s = self._monotonic()

    def stall_age_s(self) -> float:
        return max(0.0, self._monotonic() - self._last_beat_s)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="mission-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout_s: float) -> None:
        """Wait for the watchdog thread to finish; a no-op if never started."""
        if self._thread is not None:
            self._thread.join(timeout_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sleep(self._poll_s)
            if self._stop.is_set():
                # The run ended (normally, or unwinding from a named stall)
                # while this poll was pending: its own shutdown owns the
                # process now, and escalation must never race a receipt.
                return
            age = self.stall_age_s()
            if age <= self._stall_s:
                continue
            if not self._fired:
                self._fired = True
                self._fired_age_s = age
                try:
                    self._fire(age)
                except Exception:  # the watchdog never dies before the run does
                    # Firing is what names the stall; if even that fails, the
                    # escalation below is the only honest ending left.
                    self._escalate(age)
                continue
            # A second FULL window with no beat: the exception never unwound
            # the loop (a C call that does not return delays delivery past any
            # bytecode boundary). Nothing inside the process can end this run
            # cleanly any more, so it is ended from outside, cause already on
            # disk. The full window, not the next poll, is what keeps a
            # run that IS unwinding safe to finish its own receipt.
            if age - self._fired_age_s < self._stall_s:
                continue
            try:
                self._escalate(age)
            except Exception:
                # Even the last resort must not leak an exception out of this
                # thread; the artifact is already on disk.
                os._exit(MISSION_STALL_EXIT_CODE)
            return


def _raise_in_thread(thread_id: int, exception: type[BaseException]) -> int:
    """Raise ``exception`` in ``thread_id`` at its next bytecode boundary.

    Returns the number of threads the request reached (1 = delivered). While
    the target thread runs a C call that does not return — a blocking socket
    read, an extension call — delivery is delayed until it does; the
    watchdog's escalation covers exactly that case.
    """
    setter = ctypes.pythonapi.PyThreadState_SetAsyncExc
    setter.argtypes = (ctypes.c_ulong, ctypes.py_object)
    setter.restype = ctypes.c_int
    return setter(ctypes.c_ulong(thread_id), ctypes.py_object(exception))

# Section 12.2's conditional observation objective, and the declared parameters
# that bound it (R2).
#
# The specification requires the assessment to offer "any supported observation
# alternative", and the supervisor to be able to "admit a conditional
# observation objective while leaving the requested traversal unadmitted", and to
# "return both facts clearly". Until this existed the implementation returned one
# fact — the refusal — and had nothing admissible left to do, so a mission whose
# first traversal could not be admitted had `publications: 0` for its whole life.
#
# Why turning is the action. The aircraft is in a room whose near field its own
# sensor cannot see: the space beside it is outside a forward-looking frustum,
# and the space within the depth window's declared minimum range is invisible from
# every heading. So the one thing it can do that gains evidence without moving
# into space the map has not evidenced is to hold its position and turn.
#
# The rate turns a full circle in 2*pi / rate seconds, the window bounds one
# observation in SIMULATOR seconds, and the per-target bound is what keeps the
# refuse-observe-re-propose loop finite: a target may be looked at once, and then
# re-proposed, but never looked at repeatedly.
OBSERVATION_YAW_RATE_RAD_S = 0.6
OBSERVATION_SWEEP_SIM_S = 10.0
OBSERVATION_MAX_PER_TARGET = 1

# The vantage policy's declared count (R2): the consecutive ``unknown_geometry``
# groundings of the query's selection that convert "selected but unmeasurable"
# into a change of view — the section-12.2 observation sweep toward the
# selection's bearing, then explore — instead of another selection from the same
# pose. Declared in configs/first_indoor.yaml (mission.vantage_refusal_sweep_cycles).
#
# 2, derived from J52-discrim-1 (measured, GROUNDING-INVALID-DEPTH.md): the
# mission re-selected the same unmeasurable decoy sliver 19 times — in runs of
# 12 and 6 consecutive candidate-bearing observations (obs 3-14, obs 21-26; 207
# further frames proposed no candidate) — and never once changed view, every
# refusal the same `none of the 1 selected samples carries valid depth`. One
# refusing cycle is not evidence of a vantage defect: a single frame can drop
# out of the stereo matcher on an otherwise measurable view, and grounding
# refuses honestly per pixel. The SECOND consecutive refusal of the same query
# from the same standing view is the smallest count that separates the measured
# pathology (invalid depth persisting over the selection) from a one-frame
# transient: at 2 the measured run's first series sweeps on its 2nd selection
# instead of re-selecting 19 times (12 wasted selections in that series alone).
# The sweep itself runs at most once per standing view — the flag re-arms only
# when a goal is admitted and flies, which is a new vantage — so the pair
# (count 2, one sweep per view) cannot loop, and a sweep that still refuses ends
# the episode honestly ("not measurable from here") rather than re-selecting.
VANTAGE_REFUSAL_SWEEP_CYCLES = 2


@dataclass
class PhaseOutcome:
    """What one recipe phase produced, in the runner's own vocabulary."""

    status: str
    reason: str
    steps: tuple[Any, ...] = ()


@dataclass
class MissionResult:
    """Everything the bench side needs from one flown mission."""

    flew: bool
    termination_reason: str
    blockers: list[str] = field(default_factory=list)
    found: mission_module.ClaimEvidence = mission_module.ClaimEvidence(False)
    inspected: mission_module.ClaimEvidence = mission_module.ClaimEvidence(False)
    returned: mission_module.ClaimEvidence = mission_module.ClaimEvidence(False)
    phases: list[PhaseOutcome] = field(default_factory=list)
    bring_up: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    end_state: dict[str, Any] = field(default_factory=dict)
    guidance_events: list[dict[str, Any]] = field(default_factory=list)
    crash_statustexts: list[str] = field(default_factory=list)
    publications: int = 0
    publish_refusals: int = 0
    stream: dict[str, Any] = field(default_factory=dict)
    # The one reasoned pre-flight call's own document: usable or refused, with
    # its recipe source, attempts and round trip. Empty for B0, which makes no
    # call at all, so a receipt can tell the arms apart by this field alone.
    plan: dict[str, Any] = field(default_factory=dict)
    # One entry per in-flight cloud exchange a cloud arm's broker produced, so a
    # receipt carries the exchanges the mission actually had rather than a count.
    cloud_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ActiveGoal:
    """The one goal whose certified prefix may be published.

    A goal with no certificate but a ``hold_position_odom`` is a station hold:
    the aircraft stays where it is and the publisher sends its own position with
    zero velocity. That is the shape section 12.2's conditional observation
    objective takes — it does not translate, so it has no route to certify.
    """

    # Everything but the id is optional, because an observation objective is a
    # goal with a station hold and NO trajectory: it never went through the
    # acceptance path, so it has no proposal, no accepted goal and no certificate.
    goal_id: str
    proposal: R.SpatialGoal | None = None
    accepted: EX.AcceptedGoal | None = None
    certificate: PL.TrajectoryCertificate | None = None
    terminal_region: GE.BoxRegion | None = None
    execution: R.ExecutionDisposition = R.ExecutionDisposition.NOT_STARTED
    hold_position_odom: tuple[float, float, float] | None = None
    setpoints: tuple[R.MotionSetpoint, ...] = ()
    # The yaw this goal commands. A non-zero rate sweeps it, which is what turns
    # a station hold into an observation: the aircraft turns where it stands, so
    # the camera sees parts of the room it has never seen, without translating
    # into space the map has not evidenced.
    hold_yaw_rad: float = MISSION_YAW_HOLD_RAD
    hold_yaw_rate_rad_s: float = 0.0
    # SIMULATOR seconds, not wall seconds. The sweep's duration is a declared sim
    # window, so the angle swept has to be measured against the same clock, or it
    # depends on how fast the host happened to run — defect 16's shape, measured on
    # J44-fly-1 as 3.08 rad swept against a declared 6.0.
    hold_since_sim_s: float = 0.0


class MissionRuntime:
    """Flies one first-indoor mission and records its agent stream."""

    def __init__(
        self,
        *,
        settings: PlatformSettings,
        arm: str = "B0",
        config_document: dict[str, Any],
        episode_dir: Path,
        evidence_dir: Path,
        recorder,
        instruction: str,
        target_id: str,
        episode_id: str,
        sensor_tap: Callable[[Any], None] | None = None,
    ) -> None:
        self.settings = settings
        self.document = config_document
        self.episode_dir = Path(episode_dir)
        self.evidence = EvidenceWriter(Path(evidence_dir), "run-a")
        self.recorder = recorder
        self.instruction = instruction
        self.target_id = target_id
        self.episode_id = episode_id
        self._sensor_tap = sensor_tap
        localization = config_document["localization"]
        self._localization = localization
        bounds = localization["bounds"]
        self._machine = loc.HealthMachine(
            loc.HealthBounds(
                publish_period_s=localization["publish"]["period_ms"] / 1000.0,
                state_lost_after_s=bounds["state_lost_after_ms"] / 1000.0,
                published_state_age_max_s=bounds["published_state_age_max_ms"] / 1000.0,
                max_publish_gap_s=bounds["max_publish_gap_ms"] / 1000.0,
                visual_update_warn_s=bounds["visual_update_warn_ms"] / 1000.0,
                visual_update_fail_s=bounds["visual_update_fail_ms"] / 1000.0,
                valid_fraction_min=bounds["valid_fraction_min"],
                sigma_min_m=bounds["sigma_min_m"],
                sigma_max_m=bounds["sigma_max_m"],
                # Declared in the config (R2). A config that omits it keeps the
                # module default, so nothing that declares nothing changes.
                tracking_lost_min_tracks=int(bounds.get("tracking_lost_min_tracks", 5)),
                # R24's z guard, same declared pattern: the residual bound, the
                # offset window and the reference staleness are the guard's three
                # parameters; the module defaults are the values
                # configs/first_indoor.yaml declares with their derivation.
                z_guard_max_residual_m=float(bounds.get("z_guard_max_residual_m", 0.6)),
                z_guard_offset_window_s=float(
                    bounds.get("z_guard_offset_window_ms", 10000) / 1000.0
                ),
                z_guard_baro_stale_after_s=float(
                    bounds.get("z_guard_baro_stale_after_ms", 500) / 1000.0
                ),
            )
        )
        self.alignment = loc.OdomAlignment(
            _declared_start_origin(settings.world),
            _declared_start_attitude(settings.world),
        )
        self.nav_epoch = f"mission-{episode_id}"
        map_config = world_module.MapConfig(
            **MAP_PARAMETERS, dynamic_speed_mps=None, dynamic_reach_s=None
        )
        self.store = world_module.MapStore(
            map_config, submap_id=f"submap-{episode_id}", nav_epoch=self.nav_epoch
        )
        self.calibration = camera_module.build_calibration()
        calibration_config = config_document["calibration"]
        self._depth_settings = {
            **calibration_config["matcher"],
            "border_px": calibration_config["bounds"]["border_px"],
            "depth_range_m": calibration_config["bounds"]["depth_range_m"],
        }
        self.proposer = mission_module.ConventionalColourProposer()
        interpreted = mission_module.interpret_instruction(instruction)
        self._query = interpreted.target_phrase or target_id
        # One clock domain for every event envelope in the agent stream.
        self._clock = lambda: settings.capture_stamp(time.monotonic_ns())
        self._sink = self._recorder_sink
        # The arm this runtime flies. B0 builds no provider at all; B1 and B2
        # build a pilot, which refuses any unknown arm rather than guessing one.
        self.arm = arm
        self.contract = mission_module.mission_contract(
            mission_id=episode_id,
            instruction=instruction,
            budget=(
                ("mission_sim_s", MISSION_BUDGET_SIM_S),
                ("explore_steps", float(mission_module.EXPLORE_STEPS)),
            ),
        )
        # One broker per mission. A cloud arm's broker is the one MissionPilot
        # builds, because that object owns the provider and the cloud evidence;
        # the conventional arm builds the provider-less one below, where the
        # field's absence is the isolation rather than a placeholder.
        self._pilot = self._build_mission_pilot(arm)
        if self._pilot is not None:
            self.broker = self._pilot.broker
        else:
            self.broker = PilotBroker(
                seam=_LiveAdmission(self),
                parameters=PilotParameters.from_config(
                    _pilot_section(repository_root())
                ),
                provider=None,  # B0 makes no cloud call; the field's absence is the isolation
                host_id=settings.host_id,
                clock_id=settings.clock_id,
                sink=self._sink,
            )
        # The one reasoned call belongs on the ground, so the runtime has to know
        # whether it is still there: set once at liftoff and never cleared. The
        # evidence that call is made from is the first frame captured while the
        # aircraft is still down, kept as PPM because that is what the packet
        # builder takes and it re-encodes to PNG itself.
        self._airborne = False
        self._preflight_observation: R.Observation | None = None
        self._preflight_payloads: dict[str, bytes] = {}
        # Candidate targets the cloud has already been shown, so an in-flight
        # step reports a target as new once rather than on every tick.
        self._told_targets: set[str] = set()
        self._perception_queue: queue.Queue = queue.Queue(maxsize=4)
        # Perception frames dropped because the perception queue was full. A
        # count that stays at zero says the map saw every frame; anything else
        # says the map was built from a subset, which a reader of the receipt
        # needs to know before trusting a frontier.
        self._perception_frames_dropped = 0
        # Perception cycles that actually took a frame, and every observation
        # the mission recorded. The counter above says how many frames the
        # producer could not hand over; these two say what the consumer did
        # with the ones it took, which is what tells a healthy run from a
        # starved one. A queue that drops frames is the design working
        # (specification 3.1: drop replaceable old snapshots rather than allow
        # backlog) whenever the frames arrive faster than the declared
        # cadence; the fault is a cycle count that does not scale with the
        # mission's duration.
        self._perception_cycles = 0
        self._observation_ids: list[str] = []
        self._perception_refusal_counts: dict[str, int] = {}
        self._frames_without_candidate = 0
        # Where a perception cycle's wall time actually goes. live-vision-3
        # spent 3.93 wall seconds per cycle against a declared cadence of 0.30,
        # so the cadence was not honoured and the reason was inside the cycle.
        # Splitting drain from depth makes that readable instead of a guess.
        self._perception_drain_s = 0.0
        self._perception_depth_s = 0.0
        # One commanded setpoint per declared sim interval, bounded. The run
        # recorded only the COUNT of publications, so no artifact could show
        # whether the aircraft was told to climb.
        self._commanded: list[dict[str, object]] = []
        self._last_commanded_sim_s: float | None = None
        self._state_ring: list[tuple[int, loc.EstimatorState]] = []
        self._latest_state: loc.EstimatorState | None = None
        self._latest_aligned: dict[str, object] | None = None
        self._capture_clock_ns: int | None = None
        self._controller_status: dict[str, Any] = {}
        self._observation_counter = 0
        self._last_observation_id: str | None = None
        self._payload_count = 0
        self._publication_count = 0
        # Set when the declared visual-update bound is passed. It ends normal
        # operation rather than being recorded and flown past.
        self._visual_fault_reason: str | None = None
        self._publish_refusal_count = 0
        self._grounded: dict[str, R.GroundedTarget] = {}
        self._candidate_targets: list[str] = []
        self._candidate_observation_ids: list[str] = []
        self._blocked_refs: dict[str, str] = {}
        self._refusals_log: list[str] = []
        self._active_goal: _ActiveGoal | None = None
        # Section 12.2's observation objective: how many have been taken, and
        # which targets have already been looked at once. The per-target set is
        # what bounds the refuse-observe-re-propose loop.
        self._observe_count = 0
        self._observed_targets: set[str] = set()
        # The vantage policy's own state (GROUNDING-INVALID-DEPTH.md): how many
        # consecutive cycles the query's selection has refused unknown_geometry,
        # the bearing that selection was last seen at, and whether the standing
        # view has already spent its one sweep. Re-armed by a new admitted
        # vantage (_note_new_vantage), never by the refusals themselves.
        self._unmeasurable_streak = 0
        self._unmeasurable_bearing_rad = MISSION_YAW_HOLD_RAD
        self._vantage_sweep_spent = False
        self._return_settled = False
        self._return_evidence: list[str] = []
        self._inspect_evidence: list[str] = []
        self._stats = _FeedStats()
        self._first_record_kind: str | None = None
        self._platform: WebotsArduPilot | None = None
        self._session: PymavlinkSession | None = None
        self._publisher: loc.ExternalNavPublisher | None = None
        # The z guard's barometric datum: the run's first primary-barometer
        # reading (pressure in hPa, temperature in centi-degrees C). Fixed once,
        # so every later conversion shares one datum and the guard's fixed
        # offset absorbs it whole.
        self._baro_reference: tuple[float, int] | None = None
        self.result = MissionResult(flew=False, termination_reason="not_started")
        # The mission-loop watchdog, once run() starts it. A field on the
        # runtime (and not a local of run()) because the beats come from
        # methods the runtime owns, on the mission loop's own thread.
        self._watchdog: _MissionLoopWatchdog | None = None
        # The watchdog's own clock and cadence. Production values are the
        # declared constants above; they sit on the instance so a test can
        # drive this exact machinery on an injected clock (T18) without
        # touching the declared bounds.
        self._watchdog_stall_s = MISSION_LOOP_STALL_S
        self._watchdog_poll_s = MISSION_WATCHDOG_POLL_S
        self._watchdog_monotonic = time.monotonic
        self._watchdog_sleep = time.sleep

    # -- records ----------------------------------------------------------

    def _recorder_sink(self, kind: str, payload: dict, stamp, sim_time_s=None) -> None:
        self.recorder.record(kind, payload, stamp, sim_time_s)

    # -- the cloud arms ----------------------------------------------------

    def _build_mission_pilot(self, arm: str):
        """The cloud arm's pilot, or ``None`` for the arm that makes no call.

        The runtime-model configuration is a different file from the platform
        configuration this runtime is otherwise driven by — that one carries no
        ``model`` section — so it is loaded here by name. Which arms may build a
        pilot is not decided here: ``MissionPilot`` refuses B0 and any unknown
        arm, and that refusal is the one place the rule lives.
        """
        if arm == "B0":
            return None
        from embodied.pilot.mission_executive import MissionPilot, load_runtime_config
        from embodied.pilot.provider import LiveTransport, ModelConfig

        document = load_runtime_config(repository_root() / RUNTIME_MODEL_CONFIG_RELATIVE)
        return MissionPilot.for_arm(
            arm=arm,
            config_document=document,
            contract=self.contract,
            seam=_LiveAdmission(self),
            transport=LiveTransport(
                ModelConfig.from_config(document["model"]),
                api_key_env=CLOUD_API_KEY_ENV,
            ),
            sink=self._sink,
            host_id=self.settings.host_id,
        )

    def _plan_before_liftoff(self, drain) -> None:
        """The one reasoned call, made while the aircraft is still on the ground.

        The plan is made from the aircraft's own frame, so a bounded perception
        window gathers one first — the same declared cold-start window the
        mission already uses after liftoff, so this introduces no new number. If
        no frame arrives, no call is made and the local recipes fly: an absent
        plan is a recorded outcome, never a silent substitution.
        """
        self.pump_perception(
            drain,
            COLD_START_PERCEPTION_SIM_S,
            until=lambda: self._preflight_observation is not None,
        )
        if self._preflight_observation is None:
            self.result.plan = {
                "usable": False,
                "refusal_reason": (
                    "no ground observation was captured, so no reasoned call was made"
                ),
                "model_identity": None,
            }
            self.result.log.append(
                "pre-flight plan: no ground observation was captured, so the reasoned "
                "call was not made and the local recipes fly"
            )
            return
        plan = self._pilot.plan_preflight(
            self._preflight_observation,
            self._preflight_payloads,
            self._clock(),
            deadline_s=self.settings.step_timeout_s.startup,
        )
        document = plan.document()
        document["model_identity"] = plan.model_identity
        self.result.plan = document
        self.result.log.append(
            "pre-flight plan: "
            f"usable={plan.usable} source={document.get('recipe_source')!r} "
            f"steps={document.get('recipe_steps')} attempts={plan.attempts} "
            f"round_trip_s={plan.round_trip_s}"
            + (f" refusal={plan.refusal_reason!r}" if plan.refusal_reason else "")
        )

    def _tick_cloud(self, observation: R.Observation, payloads: dict[str, bytes]) -> None:
        """One in-flight step for a cloud arm, after the latch has closed.

        Two fields are real runtime state and one is a declared proxy, and the
        difference is recorded rather than blurred:

        * ``new_targets`` is the targets this runtime has grounded and not yet
          put in front of the cloud — a genuine event, and the strongest reason
          the runtime has to ask anything.
        * ``signature`` is the number of cells the map currently holds free. This
          runtime computes no scene-change score, so the signature is a proxy:
          it moves while the map is still learning and stops moving once the map
          settles, which is the behaviour a broad re-look wants. It is NOT a
          measured scene delta, and no threshold was chosen against it.
        * ``horizon_s`` is ``None``, because this runtime computes no execution
          horizon. The reduced-horizon trigger therefore cannot fire here. That
          is a limitation of this integration, not a decision.
        """
        signature = float(len(self.store.free_cells(now_ns=self._now_ns())))
        new_targets = tuple(
            target for target in self._candidate_targets if target not in self._told_targets
        )
        scene = SceneStatus(signature=signature, new_targets=new_targets, horizon_s=None)
        outcomes = self._pilot.tick(scene, observation, payloads, self._clock())
        self._told_targets.update(new_targets)
        for outcome in outcomes:
            self.result.cloud_calls.append(
                {
                    "kind": outcome.kind,
                    "reason": outcome.reason,
                    "request_id": outcome.request_id,
                    "proposal_id": outcome.proposal_id,
                }
            )
            self.result.log.append(f"cloud {self.arm}: {outcome.kind} — {outcome.reason}")

    def _now_ns(self) -> int:
        return self._capture_clock_ns if self._capture_clock_ns is not None else time.monotonic_ns()

    def _state_stamp(self) -> R.ClockStamp:
        controller_host = str(self._controller_status.get("host_id") or self.settings.host_id)
        controller_clock = str(self._controller_status.get("clock_id") or self.settings.clock_id)
        return R.ClockStamp(
            host_id=controller_host, clock_id=controller_clock, monotonic_ns=self._now_ns()
        )

    def _publishable_state(self, state: loc.EstimatorState | None) -> loc.EstimatorState | None:
        """One raw estimator state, only while the publisher's verdicts would publish it.

        ESTIMATOR-DIVERGENCE.md measured where divergence reached this mission:
        the feed thread hands every raw STATE to ``_on_state`` before
        ``publisher.offer`` sees it, and the pose accessors consumed that state
        with none of the gates the publish path applies -- so J51 flew on, and
        recorded, ``here`` = (-0.22, -6.76, -1.42), a 6.9 m pose inside a 6 m
        room, while the autopilot wire stayed clean (VPE max |y| <= 0.09 m in
        every diverged run). The publisher's ``state_for_publish`` is the
        system's declaration of what is fit to act on, and this runtime shares
        its ``HealthMachine``; the mission reads the estimator through the same
        declaration, state for state, so a state the wire would refuse is never
        the mission's pose either:

        - the declared visual-update fail verdict (F4), under the same condition
          the publisher applies it -- an initialized state whose visual clock
          has ticked; an age measured against a clock that has never ticked is
          not a fault;
        - the publisher's tracking verdict under the declared floor, when
          initialized. This is the publisher's own bound (973b0a5) mirrored, not
          the bare-count run-ender re-raised: that run-ender fired on
          ``n_tracks`` summaries the pin reports as 0 beside healthy images and
          stays removed from ``_visual_fault`` (dd0df20). Here the count can only
          refuse to serve a pose at exactly the states the wire refuses to
          carry, so this gate cannot be stricter than the publisher that owns
          the floor -- if the verdict ever false-fires, the wire stops with it,
          which is the publisher's declared behaviour, not a new mission failure
          mode.

        Refused states still land in ``_latest_state`` and ``_state_ring``:
        ``_visual_fault`` needs the raw newest state to keep measuring the age
        that fires the declared protective landing, and the ring is evidence.
        What a refused state can never be again is a pose the mission acts on.
        """
        if state is None:
            return None
        if state.t_last_visual_ns and state.initialized:
            if self._machine.visual_update_verdict(state) == "fail":
                return None
        if state.initialized and self._machine.tracking_verdict(state) == "fail":
            return None
        return state

    def _navigation_state(self) -> R.NavigationState | None:
        state = self._publishable_state(self._latest_state)
        if state is None or not self.alignment.sealed:
            return None
        sigma = state.sigma_pos_m
        return R.NavigationState(
            state_sequence=len(self._state_ring),
            pose=R.PoseEstimate(
                parent_frame="odom",
                child_frame="body",
                stamp=self._state_stamp(),
                position_m=self.alignment.aligned_position_ned(state.position_m),
                quaternion_wxyz=self.alignment.aligned_quat_ned_wxyz(state.quat_wxyz),
                covariance=(sigma[0] ** 2, sigma[1] ** 2, sigma[2] ** 2),
                nav_epoch=self.nav_epoch,
                source_ids=("ov_stream",),
                valid=bool(state.initialized),
            ),
            velocity_mps=self.alignment.aligned_velocity_ned(state.velocity_mps),
            covariance=None,
            nav_epoch=self.nav_epoch,
            visual_source_ids=(),
            imu_source_ids=("ov_stream",),
            status="healthy" if state.initialized else "initializing",
            controller_alignment_id=None,
        )

    def _position_odom(self) -> tuple[float, float, float] | None:
        """The aircraft's position in the mission's odom frame (aligned local NED)."""
        state = self._publishable_state(self._latest_state)
        if state is None or not self.alignment.sealed:
            return None
        return self.alignment.aligned_position_ned(state.position_m)

    def _speed(self) -> float:
        state = self._publishable_state(self._latest_state)
        if state is None or not self.alignment.sealed:
            return 0.0
        return float(np.linalg.norm(self.alignment.aligned_velocity_ned(state.velocity_mps)))

    def _capture_pose(self, record: Any) -> R.PoseEstimate | None:
        """The estimator pose nearest one pair's capture instant, on its own clock."""
        ring = self._state_ring
        if not ring or record.pair is None:
            return None
        target_ns = sim_time_ns(record.sim_time_s if record.sim_time_s >= 0.0 else 0.0)
        best = min(ring, key=lambda entry: abs(entry[0] - target_ns))
        state = self._publishable_state(best[1])
        if state is None:
            return None
        controller_host = str(self._controller_status.get("host_id") or self.settings.host_id)
        controller_clock = str(self._controller_status.get("clock_id") or self.settings.clock_id)
        sigma = state.sigma_pos_m
        return R.PoseEstimate(
            parent_frame="odom",
            child_frame="body",
            stamp=R.ClockStamp(
                host_id=controller_host,
                clock_id=controller_clock,
                monotonic_ns=int(record.pair.capture_host_ns),
            ),
            position_m=self.alignment.aligned_position_ned(state.position_m),
            quaternion_wxyz=self.alignment.aligned_quat_ned_wxyz(state.quat_wxyz),
            covariance=(sigma[0] ** 2, sigma[1] ** 2, sigma[2] ** 2),
            nav_epoch=self.nav_epoch,
            source_ids=("ov_stream",),
            valid=bool(state.initialized),
        )

    # -- the whole mission -------------------------------------------------

    def _watchdog_start(self) -> None:
        """Start the mission-loop watchdog on the calling thread.

        Called by run() on the mission loop's own thread; kept as its own
        method so a test can start the run's real fire/escalate machinery on
        an injected clock instead of measuring the machine (T18).
        """
        thread_id = threading.get_ident()

        def fire(age_s: float) -> None:
            # The run's own record of the stall, beside its other evidence:
            # every thread's stack, so the hung computation is named by its
            # frames rather than guessed from a shell transcript. Written
            # directly, not through the recorder, which the watchdog thread
            # must not touch mid-write.
            dump_path = self.evidence.path("mission-loop-stall.txt")
            with open(dump_path, "w", encoding="utf-8") as handle:
                handle.write(
                    f"mission loop stalled: no beat for {age_s:.1f} s "
                    f"(bound {self._watchdog_stall_s:.1f} s); mission thread "
                    f"{thread_id}\n"
                )
                handle.write(
                    "result at stall: flew="
                    f"{self.result.flew} termination_reason="
                    f"{self.result.termination_reason!r}\n\n"
                )
                handle.flush()
                faulthandler.dump_traceback(file=handle, all_threads=True)
            self.result.termination_reason = "mission_loop_stalled"
            self.result.blockers.append(
                f"the mission loop stopped beating for {age_s:.1f} s "
                f"(bound {self._watchdog_stall_s:.1f} s); the stall and every "
                "thread's stack are in mission-loop-stall.txt"
            )
            self.result.log.append(
                f"mission loop stalled: no beat for {age_s:.1f} s; breaking the loop"
            )
            if _raise_in_thread(thread_id, MissionLoopStalled) != 1:
                self.result.log.append(
                    "the stall exception could not be delivered to the mission "
                    "thread; the watchdog will escalate"
                )

        def escalate(age_s: float) -> None:
            # One full bound after fire the loop still has not unwound: the
            # thread is wedged where the asynchronous exception cannot reach.
            # The artifact is on disk; keep its timeline and end the process,
            # because this run will never write its own receipt.
            marker = self.evidence.path("mission-loop-stall.txt")
            try:
                with marker.open(
                    "a" if marker.exists() else "w", encoding="utf-8"
                ) as handle:
                    handle.write(
                        f"escalated after {age_s:.1f} s without a beat; exiting "
                        f"{MISSION_STALL_EXIT_CODE}\n"
                    )
            except Exception:
                pass
            os._exit(MISSION_STALL_EXIT_CODE)

        self._watchdog = _MissionLoopWatchdog(
            thread_id=thread_id,
            fire=fire,
            escalate=escalate,
            stall_s=self._watchdog_stall_s,
            poll_s=self._watchdog_poll_s,
            monotonic=self._watchdog_monotonic,
            sleep=self._watchdog_sleep,
        )
        self._watchdog.start()

    def _watchdog_stop(self) -> None:
        """Stop the watchdog and wait for it. Every run() exit path calls this
        before the shutdown, so a healthy run never carries a live watchdog —
        and a stalled run's escalation can never race the receipt."""
        watchdog = self._watchdog
        if watchdog is None:
            return
        self._watchdog = None
        watchdog.stop()
        watchdog.join(timeout_s=3.0 * self._watchdog_poll_s)

    def _beat_watchdog(self) -> None:
        """One turn of the mission loop, on the loop's own thread.

        A no-op before run() starts the watchdog, so nothing outside a run
        pays for it.
        """
        watchdog = self._watchdog
        if watchdog is not None:
            watchdog.beat()

    def run(self) -> MissionResult:
        log = self.result.log
        estimator = self._localization["estimator"]
        estimator_process = _start_estimator(estimator, repository_root(), self.evidence)
        if estimator_process is None:
            self.result.termination_reason = "estimator_unavailable"
            self.result.blockers.append(
                f"the estimator process {estimator['executable']} did not open port "
                f"{estimator['socket_port']}"
            )
            return self.result
        client = loc.OvStreamClient("127.0.0.1", int(estimator["socket_port"]))
        try:
            client.connect()
        except (OSError, loc.ProtocolError) as error:
            self.result.termination_reason = "estimator_unavailable"
            self.result.blockers.append(
                f"the adapter could not connect to the estimator on 127.0.0.1:"
                f"{estimator['socket_port']}: {error}"
            )
            estimator_process.terminate()
            return self.result
        feed_stop = threading.Event()
        feed_failures: list[str] = []
        pending_pairs: queue.Queue = queue.Queue(maxsize=PAIR_QUEUE_FRAMES)
        publisher = loc.ExternalNavPublisher(
            self.settings.mavlink_endpoint,
            self.alignment,
            self._machine,
            clock=lambda: self._stats.sim_clock.newest_s or 0.0,
            on_publish=self._on_publish,
        )
        session = PymavlinkSession()
        platform = WebotsArduPilot(
            self.settings,
            runner=LowPrioritySubprocessRunner(),
            session=session,
            gateway=TcpSensorGateway(stamp=lambda: self.settings.capture_stamp(time.monotonic_ns())),
            evidence=self.evidence,
            label="run-a",
            extra_params=self.settings.estimator_params,
        )
        self._platform, self._session, self._publisher = platform, session, publisher

        def file_record(record: Any) -> None:
            if record.kind is not Kind.PAIR or record.pair is None:
                return
            self._stats.pair_records_filed += 1
            try:
                pending_pairs.put_nowait(record)
            except queue.Full:
                self._stats.pair_records_dropped += 1
            self._offer_to_perception(record)

        platform.record_sink = file_record

        def feed_record(record: Any) -> None:
            # The simulator's clock is read on the one path that consumes the
            # whole stream, exactly as the worked example does.
            if self._first_record_kind is None:
                self._first_record_kind = record.kind.name
                log.append(
                    f"first sensor record: {record.kind.name} at sim "
                    f"{record.sim_time_s:.3f} s"
                )
            self._stats.sim_clock.observe(record.sim_time_s)
            if self._sensor_tap is not None:
                # Bench-side observation only: this module never reads a POSE
                # record's contents.
                self._sensor_tap(record)
            if record.kind is Kind.PAIR and record.pair is not None:
                pair = record.pair
                sample = SensorSample(
                    value=pair,
                    capture_stamp=self.settings.capture_stamp(pair.capture_host_ns),
                    receipt_stamp=record.received_stamp,
                    sim_time_s=record.sim_time_s,
                )
                self._stats.pair_latencies_ns.append(capture_latency_ns(sample))
                left = loc.grayscale_rgb8(
                    pair.left_bytes, self.settings.stereo.width, self.settings.stereo.height
                )
                right = loc.grayscale_rgb8(
                    pair.right_bytes, self.settings.stereo.width, self.settings.stereo.height
                )
                client.send(
                    loc.encode_stereo(
                        sim_time_ns(record.sim_time_s),
                        left,
                        right,
                        self.settings.stereo.width,
                        self.settings.stereo.height,
                    )
                )
                self._stats.pairs += 1
            elif record.kind is Kind.IMU and record.imu is not None:
                imu = record.imu
                stamp_ns = sim_time_ns(record.sim_time_s)
                self._stats.newest_imu_ns = max(self._stats.newest_imu_ns, stamp_ns)
                client.send(loc.encode_imu(stamp_ns, imu.gyro, imu.accelerometer))
                self._stats.imu_samples += 1
            elif record.kind is Kind.POSE and record.pose is not None:
                # Bench-side accounting on the same feed object the worked
                # example keeps: the bring-up's declared end-state check reads
                # these, and they are the only truth fields this module touches.
                # Nothing here reaches the estimator (its feed handles PAIR and
                # IMU only) and nothing here feeds the map, a plan or a setpoint.
                self._stats.truth_samples.append(
                    (sim_time_ns(record.sim_time_s), tuple(record.pose.position_xyz))
                )
                self._stats.truth_attitudes.append(
                    (sim_time_ns(record.sim_time_s), tuple(record.pose.attitude_rpy))
                )
        # One stereo frame may wait here for the inertial samples that must precede it;
        # see OrderedPairFeed for what sending it early does to the estimator.
        ordered_pairs = OrderedPairFeed(
            pending_pairs,
            sim_time_ns_of=lambda held: sim_time_ns(held.sim_time_s),
            feed_one=feed_record,
            stats=self._stats,
        )

        def feed_cycle() -> None:
            while True:
                record = platform.sensor_record(FEED_STREAM_POLL_S)
                if record is None:
                    break
                feed_record(record)
            # A queued image may still be ahead of the inertial samples that must precede
            # it, because the sink files a pair the moment the reader sends it while the
            # inertial sample that follows it in the stream arrives on the other queue a
            # moment later. The guard holds such a frame until its own samples have gone:
            # an image with an empty inertial interval is dropped by the pin, and a run of
            # those freezes the filter -- which is the state that later diverges and drags
            # the EKF, the controller and the vehicle down with it.
            ordered_pairs.drain(self._stats.newest_imu_ns)
            state = client.poll_state()
            if state is not None:
                self._on_state(state)
                publisher.offer(state, self._stats.newest_imu_ns)

        def feed_loop() -> None:
            while not feed_stop.is_set():
                try:
                    feed_cycle()
                except Exception as error:  # the feed stops the machine on any failure
                    feed_failures.append(f"the estimator feed failed: {error}")
                    self._machine.stop(time.monotonic_ns(), str(error))
                    return

        last_telemetry = 0.0

        def drain() -> None:
            nonlocal last_telemetry
            # One turn of the mission loop: everything the ordered bring-up
            # waits inside of turns here.
            self._beat_watchdog()
            # Deliberately NOT a perception pump. `drain` runs inside the
            # ordered bring-up, whose RC-throttle override must be refreshed
            # every 0.5 s inside the firmware's own 3.0 s override timeout.
            # Running stereo depth here (SGBM, 128 disparities over 640x480)
            # starved that refresh: on live-10 the vehicle's own RC report read
            # 1000 us while the window sent 1899 us, so the excitation had no
            # thrust path at all, the open-loop takeoff ramp then ran unbounded
            # to 1.19 m, and the aircraft tipped over at roll -90 deg.
            # Perception is pumped where it belongs instead: a bounded
            # cold-start window before the phases, and inside each step's own
            # loop, which is where the map is built from flown observation.
            if time.monotonic() - last_telemetry < 0.2:
                return
            last_telemetry = time.monotonic()
            try:
                sample = platform.telemetry()
            except Exception as error:
                self._machine.stop(time.monotonic_ns(), f"the telemetry stream failed: {error}")
            else:
                self._offer_baro(sample)

        bring_up_link: BringUpLink | None = None
        feed_thread: threading.Thread | None = None
        try:
            # The supervisor starts with the run and stops in the finally
            # below, before the shutdown, on every exit path.
            self._watchdog_start()
            self.broker.set_mission(self.contract, self._clock())
            platform.start()
            readiness = platform.wait_ready(self.settings.step_timeout_s.startup)
            self._controller_status = readiness.controller_status
            log.append(
                f"platform ready after {readiness.waited_s:.1f} s: pair_seen="
                f"{readiness.pair_seen}, imu_seen={readiness.imu_seen}, "
                f"status={readiness.telemetry.system_status}"
            )
            feed_thread = threading.Thread(target=feed_loop, name="estimator-feed", daemon=True)
            feed_thread.start()
            feed_endpoint = _autopilot_feed_endpoint(
                self.evidence.path("sitl.log"), self.settings.mavlink_endpoint
            )
            publisher.retarget(feed_endpoint)
            log.append(f"adapter publish endpoint: {feed_endpoint}")
            bring_up_endpoint = _autopilot_feed_endpoint(
                self.evidence.path("sitl.log"),
                self.settings.mavlink_endpoint,
                already_taken=(feed_endpoint,),
            )
            platform.request_telemetry_streams()
            for message_id in (MSG_ID_SYS_STATUS, MSG_ID_GPS_RAW_INT):
                session.request_message_interval(message_id, GPS_AIDING_SAMPLE_HZ)
            applied: dict[str, float] = {}
            for name, _expected, _source in VEHICLE_REQUIREMENTS:
                applied.update(
                    platform.read_parameters(
                        (name,), timeout_s=PARAMETER_READ_TIMEOUT_S, drain=drain
                    )
                )
            refusals = _param_error_refusals(self.evidence.path("mavlink.jsonl"))
            blockers: list[str] = []
            truth_published = platform.truth_feed_published
            if truth_published:
                blockers.append(
                    f"the bridge sent {truth_published} simulator poses: the truth republish "
                    "is not off, so this sensor-derived arm is refused"
                )
            blockers.extend(_readback_blockers(applied, refusals))
            blockers.extend(_gps_aiding_verdict(self.evidence.path("mavlink.jsonl"))["blockers"])
            if blockers:
                self.result.blockers.extend(blockers)
                self.result.termination_reason = "preflight_refused"
                log.append("not starting the flight: a precondition of the scored arm failed")
                return self.result
            publisher.start()
            # The link speaks as the vehicle's OWN declared GCS system id, read
            # back here rather than assumed: the firmware drops an RC override
            # from any other id, and the bring-up's thrust path is exactly that
            # message (the worked example's own finding).
            gcs_id = platform.read_parameters(
                (BRING_UP_GCS_SYSTEM_PARAMETER,),
                timeout_s=PARAMETER_READ_TIMEOUT_S,
                drain=drain,
            ).get(BRING_UP_GCS_SYSTEM_PARAMETER)
            if gcs_id is None:
                self.result.blockers.append(
                    f"the vehicle did not answer {BRING_UP_GCS_SYSTEM_PARAMETER}: the bring-up's "
                    "RC override path is only accepted from the vehicle's own declared GCS"
                )
                self.result.termination_reason = "preflight_refused"
                return self.result
            bring_up_link = BringUpLink(bring_up_endpoint, source_system=int(gcs_id))
            bring_up_link.connect(timeout_s=BRING_UP_GCS_CONNECT_TIMEOUT_S)
            bring_up = _run_ordered_bring_up(
                self.settings,
                platform,
                session,
                bring_up_link,
                self.evidence,
                drain,
                log,
                self._stats,
                published_attitude=lambda: (
                    None
                    if self._latest_aligned is None
                    else self._latest_aligned["attitude_rpy"]
                ),
                capture=None,
            )
            self.result.bring_up = bring_up
            if bring_up["blockers"]:
                self.result.blockers.extend(bring_up["blockers"])
                self.result.termination_reason = "bring_up_refused"
                return self.result
            initialized = _wait_initialized(
                self._machine,
                drain,
                _SimWindow(
                    self._stats.sim_clock,
                    self.settings.pre_arm_wait_s,
                    label="the declared wait for the estimator to initialize",
                    wall_ceiling_s=_sim_window_wall_ceiling_s(
                        self.settings.pre_arm_wait_s,
                        self.settings.realtime_ratio_envelope[0],
                    ),
                ),
            )
            if feed_failures:
                self.result.blockers.extend(feed_failures)
                self.result.termination_reason = "estimator_feed_failed"
                return self.result
            if not initialized:
                self.result.blockers.append(
                    "the estimator did not reach a healthy initialized state inside the "
                    "declared pre-arm wait"
                )
                self.result.termination_reason = "estimator_not_initialized"
                return self.result
            # The owner's ruling: the single reasoned call happens on the ground,
            # using whatever the aircraft can gather while it is stationary. It
            # is made here — after the estimator is initialized and before the
            # guided takeoff — because past this point every cloud call is the
            # fast continuous class and the reasoned one is structurally
            # impossible.
            if self._pilot is not None:
                self._plan_before_liftoff(drain)
            control = platform.arm_and_guided(self.settings.step_timeout_s.flight, drain=drain)
            if control.refused:
                self.result.blockers.append(
                    f"the autopilot refused Guided flight: mode_reached={control.mode_reached}, "
                    f"armed={control.armed}, refusals={list(control.refusals)}"
                )
                self.result.termination_reason = "arming_refused"
                return self.result
            self.result.flew = True
            if self._pilot is not None:
                # Liftoff. The builder's latch closes here and never reopens, so
                # no in-flight path can issue a reasoned call, however it is
                # written later.
                self._pilot.mark_airborne(self._clock(), reason="guided takeoff")
                self._airborne = True
            self._machine.open_window(time.monotonic_ns())
            self._fly_the_mission(drain)
        except MissionLoopStalled:
            # The watchdog already wrote the stall artifact and named the
            # termination. Unwinding HERE, rather than out of run(), is what
            # lets the run's normal shutdown and receipt still happen — the
            # one thing J50-move-1 never got.
            log.append(
                "mission loop stalled: the loop unwound; the run's receipt follows"
            )
            self.result.termination_reason = "mission_loop_stalled"
        finally:
            try:
                self._watchdog_stop()
                self._shutdown(
                    platform, publisher, estimator_process, feed_stop, feed_thread, bring_up_link
                )
                self.result.publications = self._publication_count
                self.result.publish_refusals = self._publish_refusal_count
                self._surface_refusals()
                self.result.stream = {
                    "pairs": self._stats.pairs,
                    "pair_records_filed": self._stats.pair_records_filed,
                    "pair_records_dropped": self._stats.pair_records_dropped,
                    "pairs_held_for_imu": self._stats.pairs_held_for_imu,
                    "max_pair_hold_s": round(self._stats.max_pair_hold_s, 6),
                    "imu_samples": self._stats.imu_samples,
                    "truth_samples": len(self._stats.truth_samples),
                    "sim_clock_frames": self._stats.sim_clock.frames,
                    "sim_clock_newest_s": self._stats.sim_clock.newest_s,
                    "first_record_kind": self._first_record_kind,
                    "feed_failures": list(feed_failures),
                    "perception_frames_dropped": self._perception_frames_dropped,
                    # Cycles that took a frame, the observations they produced, the
                    # declared interval they were paced by, and how many distinct
                    # reasons perception refused. Frames dropped are the queue
                    # discarding an older snapshot for a newer one, which is the
                    # design; the fault this pair detects is a cycle count that does
                    # not scale with the run's own duration.
                    "perception_cycles": self._perception_cycles,
                    "perception_observations": self._observation_counter,
                    "perception_interval_s": MIN_PERCEPTION_INTERVAL_S,
                    "perception_refusal_reasons": len(self._perception_refusal_counts),
                    # Where a perception cycle's wall time went, and what was
                    # actually commanded. Both exist because a run that says only
                    # "85 cycles" and "0 publications" cannot be read: the first
                    # hides whether the cadence was honoured and why not, the second
                    # hides whether the aircraft was ever told to go anywhere.
                    "perception_drain_s": round(self._perception_drain_s, 3),
                    "perception_depth_s": round(self._perception_depth_s, 3),
                    "commanded_setpoints": list(self._commanded),
                    "publisher_published": getattr(publisher, "published", None),
                    "publisher_failures": list(getattr(publisher, "publish_failures", [])),
                    # The arm, the one reasoned call's own document, and every
                    # in-flight cloud exchange. The transport already carries this
                    # dict into the run's record verbatim, so the receipt can tell
                    # a conventional run from a cloud one and show what the cloud
                    # was asked and what came back without a second integration.
                    "arm": self.arm,
                    "plan": self.result.plan,
                    "cloud_calls": list(self.result.cloud_calls),
                }
            except MissionLoopStalled:
                # A delivery that lands in a run already shutting down: the
                # stall is named and its artifact is on disk, and the receipt
                # must survive even this.
                self.result.termination_reason = "mission_loop_stalled"
        return self.result

    # -- the mission phases -------------------------------------------------

    def _fly_the_mission(self, drain) -> None:
        mission_window = _SimWindow(
            self._stats.sim_clock,
            MISSION_BUDGET_SIM_S,
            label="the mission budget",
            wall_ceiling_s=_sim_window_wall_ceiling_s(
                MISSION_BUDGET_SIM_S, self.settings.realtime_ratio_envelope[0]
            ),
        )
        world = _LiveRunnerWorld(self)
        runner = RecipeRunner(self.broker, world, now=self._clock, sink=self._sink)
        termination = "mission_completed"
        # Look before leaping. A frontier is resolved out of the map, the map
        # is built from the aircraft's own frames, and an empty store resolves
        # nothing — so the first goal could never resolve a target, the step
        # never ran, perception never ran, and the mission was blocked against
        # its own empty map. Gather on perception's own cadence for a bounded
        # window before the phases begin, so the first goal resolves against a
        # real snapshot.
        #
        # The coupling that starved perception was never only a cold-start
        # problem, which is why the same gathering also runs between phases
        # below. live-motion-8 made 7 observations in 64 simulated seconds and
        # dropped 593 frames: five of the seven came from this window, and once
        # it closed no goal could be admitted, so the lease loop that pumped
        # perception never turned, so the map could not grow, so no goal could
        # be admitted. A mission with nothing to do has to keep looking.
        self.pump_perception(
            drain,
            COLD_START_PERCEPTION_SIM_S,
            # Only a grounded candidate ends the cold start early. A navigable
            # frontier does not: live-vision-1 and live-vision-2 both logged
            # "0 navigable" at the close of a cold start that had already
            # stopped after one observation, so a single frontier region is
            # satisfiable by the cell the aircraft is standing in and says
            # nothing about whether there is somewhere to go.
            until=lambda: bool(self._candidate_targets),
        )
        self.result.log.append(
            f"cold start: {len(self.frontier_regions())} frontier region(s), "
            f"{len(self.navigable_frontiers())} navigable, "
            f"{len(self._candidate_targets)} grounded candidate(s), "
            f"{self._observation_counter} observation(s) seen, "
            f"{self._perception_cycles} perception cycle(s); "
            f"map {self.map_summary()}"
        )
        # The inspect phase's own step guard (candidate_present) decides whether
        # there is anything to inspect, evaluated when that phase is reached. The
        # skip that used to live here read a snapshot taken BEFORE exploration
        # ran, so it could only ever skip the phase that exploration exists to
        # feed.
        # A cloud arm flies the plan it made on the ground; the conventional arm
        # flies its own three recipes. A refused plan is a complete outcome: the
        # local recipes fly and the refusal is already recorded, never replaced
        # silently. Either way the recipe goes through this one runner, so the
        # arms differ in where the plan came from and in nothing else.
        plan = self._pilot.in_flight_plan if self._pilot is not None else None
        if plan is not None and plan.usable and plan.recipe is not None:
            phases = ((f"cloud-plan({plan.recipe.source})", plan.recipe),)
            self.result.log.append(
                "cloud plan adopted: "
                f"{[step.action for step in plan.recipe.steps]} "
                f"(source {plan.recipe.source!r})"
            )
        else:
            if plan is not None:
                self.result.log.append(
                    "cloud plan refused: "
                    f"{plan.refusal_reason or 'no recipe was returned'}; the local "
                    "recipes fly and the refusal stands as recorded"
                )
            phases = tuple(
                zip(("explore", "inspect", "return"), mission_module.build_b0_recipes())
            )
        index = 0
        regathers: dict[str, int] = {}
        while index < len(phases):
            self._beat_watchdog()
            if self._visual_fault_reason is not None:
                # The estimate is no longer a pose. Landing is the declared
                # protective end, and the reason names the bound that was
                # passed rather than blaming the phase that happened to be
                # running.
                termination = "visual_localization_lost"
                break
            if mission_window.expired():
                termination = "mission_budget_exhausted"
                break
            phase, recipe = phases[index]
            outcome = runner.run(recipe)
            self.result.phases.append(
                PhaseOutcome(outcome.status, outcome.reason, outcome.steps)
            )
            self.result.log.append(f"phase {phase}: {outcome.status} — {outcome.reason}")
            if outcome.status == "budget_exhausted":
                termination = "mission_budget_exhausted"
                break
            used = regathers.get(phase, 0)
            if (
                outcome.status == "blocked"
                and not mission_window.expired()
                and used < MAX_PHASE_REGATHERS
            ):
                # Blocked means the mission had nothing to aim at, not that it
                # aimed and failed: the runner resolves a target out of the map,
                # and with nothing resolvable it spends its attempts without ever
                # reaching the lease loop. Gather on the same declared window and
                # run the phase again against a map that has grown.
                #
                # Bounded, because unbounded retrying is what spent the whole
                # mission budget inside `explore` on live-vision-1: `inspect` and
                # `return` never ran, and a mission that never returns reports
                # that it never returned. It also stops as soon as gathering
                # stops producing frames, so a scene that is giving the mission
                # nothing ends the retries early.
                regathers[phase] = used + 1
                gathered = self.pump_perception(drain, COLD_START_PERCEPTION_SIM_S)
                self.result.log.append(
                    f"phase {phase} blocked: gathered {gathered} further frame(s) "
                    f"before retry {used + 1}/{MAX_PHASE_REGATHERS} "
                    f"({self._observation_counter} observation(s) so far, "
                    f"{len(self.navigable_frontiers())} navigable)"
                )
                if gathered > 0:
                    continue
            index += 1
        # Landing is a protective end, not a mission action: whatever the
        # recipes reached, the aircraft comes down and the report says which
        # obligations were met.
        self._land(drain)
        self.result.found = mission_module.ClaimEvidence(
            bool(self._candidate_observation_ids),
            tuple(dict.fromkeys(self._candidate_observation_ids))[:4],
        )
        self.result.inspected = mission_module.ClaimEvidence(
            bool(self._inspect_evidence),
            tuple(dict.fromkeys(self._inspect_evidence))[:4],
        )
        # A physical return is checked from the mission's own state and
        # controller feedback (specification 18.3), so a settled return is
        # claimed as returned even when the step that made it had no observation
        # to cite. Requiring an observation here is what made a verified return
        # report "not_returned": the claim contradicted the referee, and an
        # under-claim that contradicts measured truth is still a wrong claim.
        # When the step did observe, those observations are cited as before.
        self.result.returned = mission_module.ClaimEvidence(
            bool(self._return_settled),
            tuple(dict.fromkeys(self._return_evidence))[:4],
        )
        self._persist_perception_refusals()
        self.result.termination_reason = termination

    def _land(self, drain) -> None:
        self._session.set_mode("LAND")
        self._machine.mark_flight_end(time.monotonic_ns())
        landing = _SimWindow(
            self._stats.sim_clock,
            LANDING_DRAIN_SIM_S,
            label="the post-mission landing drain",
            wall_ceiling_s=_sim_window_wall_ceiling_s(
                LANDING_DRAIN_SIM_S, self.settings.realtime_ratio_envelope[0]
            ),
        )
        while not landing.expired():
            drain()
            time.sleep(0.01)
        try:
            sample = self._platform.telemetry()
            self.result.end_state = _end_state_record(sample)
        except Exception:
            self.result.end_state = {}
        try:
            self.result.crash_statustexts = list(
                _crash_disarm_statustexts(self.evidence.path("mavlink.jsonl"))
            )
        except Exception:
            self.result.crash_statustexts = []
        self.result.guidance_events = [
            event.document() for event in getattr(self._platform, "control_events", [])
        ]

    def _surface_refusals(self) -> None:
        """Copy the mission's refusal reasons into the record that is written out.

        A refusal is a reason the aircraft did nothing: an admission the
        supervisor would not accept, a target that could not be grounded, a
        depth product that failed. They used to be accumulated in
        ``_refusals_log`` and then dropped, so a run could report
        "no_active_goal" and never state why the goal was refused. Live-19 is the
        worked case: a frontier resolved to a vantage 5.63 m away, publication
        then stopped, and the reason was on this list and invisible. Deduped,
        order preserved, and the count survives a long list.
        """
        # R24: if the z guard stopped the feed between setpoint ticks, the run's
        # record still says so. The line is handed over exactly once, so a trip
        # already surfaced in flight is not repeated here.
        guard_line = self._publisher.z_guard_receipt_line() if self._publisher else None
        if guard_line is not None:
            self.result.log.append(guard_line)
        unique_refusals = list(dict.fromkeys(self._refusals_log))
        for entry in unique_refusals[:40]:
            self.result.log.append(f"refused: {entry}")
        if len(unique_refusals) > 40:
            self.result.log.append(
                f"refused: ... and {len(unique_refusals) - 40} further distinct refusals"
            )

    def _shutdown(self, platform, publisher, estimator_process, feed_stop, feed_thread, link) -> None:
        for action in (
            lambda: link.close() if link is not None else None,
            publisher.stop,
            lambda: (feed_stop.set(), feed_thread.join(timeout=5.0) if feed_thread else None),
            platform.stop,
            estimator_process.terminate,
        ):
            try:
                action()
            except Exception:
                pass

    # -- estimator state -----------------------------------------------------

    def _offer_baro(self, sample: Any) -> None:
        """Convert the freshest primary-barometer reading for the z guard (R24).

        The reference is the vehicle's own barometer, and the only honest carrier
        of it on this firmware is SCALED_PRESSURE: GLOBAL_POSITION_INT is built
        from the EKF's own position (send_global_position_int reads
        ahrs.get_location; AP_NavEKF3_Outputs.cpp:316), and with EK3_SRC1_POSZ 6
        that position is the vision feed — comparing them would be vision against
        vision. The conversion is ArduPilot's own simple model with this run's
        first reading as the datum, so the guard's fixed-offset window absorbs
        the datum and no second atmospheric opinion enters.
        """
        if self._publisher is None:
            return
        pressure = getattr(sample, "press_abs_hpa", None)
        temperature = getattr(sample, "press_temp_cdegc", None)
        if pressure is None or temperature is None or pressure <= 0.0:
            return
        if self._baro_reference is None:
            self._baro_reference = (pressure, temperature)
            return
        ref_pressure, ref_temperature = self._baro_reference
        self._publisher.offer_baro(
            baro_relative_altitude_m(pressure, temperature, ref_pressure, ref_temperature)
        )

    def _on_state(self, state: loc.EstimatorState) -> None:
        self._latest_state = state
        self._state_ring.append((state.time_ns, state))
        if len(self._state_ring) > POSE_RING:
            del self._state_ring[0]

    def _on_publish(self, state: loc.EstimatorState, aligned: dict[str, object]) -> None:
        self._latest_aligned = aligned

    # -- perception ----------------------------------------------------------

    def _offer_to_perception(self, record: Any) -> None:
        """Hand one stereo pair to the perception queue, newest frame wins.

        A full queue drops its OLDEST frame rather than refusing the new one:
        an old image snapshot is worth less than the current one
        (specification 3.1, "drop replaceable old image snapshots rather than
        allow backlog"), and blocking here would stall the thread the frames
        arrive on. This method exists because the queue was once declared and
        drained but never fed, and the omission was invisible: perception
        simply never ran, so no observation, map, candidate or frontier was
        ever produced, and the mission deadlocked against its own empty map.
        """
        try:
            self._perception_queue.put_nowait(record)
            return
        except queue.Full:
            pass
        try:
            self._perception_queue.get_nowait()
        except queue.Empty:
            pass
        self._perception_frames_dropped += 1
        try:
            self._perception_queue.put_nowait(record)
        except queue.Full:
            self._perception_frames_dropped += 1

    def _persist_perception_refusals(self) -> None:
        """Write the perception refusals into the run's own record.

        The runtime collected these from the first flight and wrote them
        nowhere: five candidates were proposed, five groundings refused, and no
        artifact said why. Deduplicated and bounded, so a reader of the receipt
        learns the reason without the log growing with every frame.
        """
        if not self._perception_refusal_counts:
            self.result.log.append("perception refusals: none")
        else:
            total = sum(self._perception_refusal_counts.values())
            self.result.log.append(
                f"perception refusals: {total} over "
                f"{len(self._perception_refusal_counts)} distinct reason(s)"
            )
        # Always reported, refusals or none: it is the number that separates
        # "the mission looked and found nothing to aim at" from "the mission
        # never looked", and the two failures have nothing in common.
        self.result.log.append(
            f"perception candidates: {self._frames_without_candidate} frame(s) "
            "proposed no candidate of the query"
        )

    def _note_perception_refusal(self, line: str) -> None:
        """Record a perception refusal where the run's own record can see it.

        These were collected and never written anywhere: five candidates were
        proposed, five groundings refused, and the reason appeared in no
        artifact. A refusal a reader cannot read is a defect that hides a
        defect.
        """
        self._refusals_log.append(line)
        self._perception_refusal_counts[line] = (
            self._perception_refusal_counts.get(line, 0) + 1
        )

    def _append_observation_id(self, record_id: str) -> None:
        """Remember every observation the mission made, not only grounded ones.

        The step loop collected citations from grounded candidates alone, so a
        step that observed the room but grounded nothing cited nothing — which
        is how a settled return came out as "not_returned" against a referee
        that had verified the return.
        """
        self._observation_ids.append(record_id)
        if len(self._observation_ids) > MAX_CITED_OBSERVATIONS:
            del self._observation_ids[0]

    def _visual_fault(self) -> str | None:
        """The declared visual-update bound, evaluated (F4).

        ``visual_update_fail_ms`` has been declared, measured and logged since
        P01-L, and this runtime built a ``HealthMachine`` carrying it and then
        never asked that machine a single question. The consequence is
        measured: on three retained runs the aircraft climbed into the room's
        ceiling and crashed while its visual updates had already stopped, and
        nothing in the mission noticed.

        The mechanism those runs show is worth stating, because it is not
        obvious. ``EK3_SRC1_POSZ`` is ExternalNav, so the altitude channel is
        the estimator's own z. When the view degenerates the estimator's z
        drifts low: on the run whose dataflash is ``00000238.BIN`` the
        published altitude read 1.76 m while the simulator's own state said
        2.44 m, an error that grew to 1.12 m. GUIDED then holds the *estimated*
        position, so a drifting estimate is chased by real motion — the
        aircraft climbed about a metre to keep a falling number where it was,
        struck the 2.5 m ceiling, and the crash detector did the rest.

        So the bound is not a statistic about the log. Past it the estimate is
        no longer a pose, and the only honest options are the declared
        protective behaviours. This returns the reason past the bound, and the
        caller lands.

        Not evaluated before the first visual update (``t_last_visual_ns``
        zero) or before the estimator is initialized, because an age measured
        against a clock that has never ticked is not a fault.
        """
        state = self._latest_state
        if state is None or not state.t_last_visual_ns or not state.initialized:
            return None
        # The tracker's count is NOT a bound, and gating on it ended every run.
        #
        # `n_tracks` on the wire is filled from `get_active_tracks`, which hands
        # out the pin's `active_tracks_uvd`. That member is written only by
        # `VioManager::retriangulate_active_tracks` (VioManagerHelper.cpp:205
        # clears it, :378 inserts), and it keeps just the tracks that triangulate
        # with positive depth and project inside the image. So it reads 0
        # whenever triangulation yields nothing — which is exactly the
        # stationary case this mission begins in.
        #
        # The pin says the same thing in its own encoder (estimator/ov_stream.cpp):
        # the count is "recorded as a diagnostic, never as a bound: plan section
        # 7 declines to claim a threshold from it."
        #
        # Measured, J48-fly-1: the field read 0 on every summary while the
        # feature database held 88 to 93 features and the images were healthy
        # (left_sd 21.09), and a floor of 5 on it ended the run with "the
        # tracker holds 0 feature(s), below the declared floor 5". Five runs were
        # lost that way. It is a diagnostic here now, and the declared
        # visual-update age bound below is the guard that was actually declared
        # for this fault.
        if not getattr(self, "_tracker_field_noted", False):
            self._tracker_field_noted = True
            self.result.log.append(
                f"tracker diagnostic: {state.n_tracks} feature(s) on the wire field. "
                "That field counts re-triangulated tracks and reads 0 whenever "
                "triangulation yields nothing, so it is not a tracker-health bound "
                "and no run is ended on it (see _visual_fault)."
            )
        if self._machine.visual_update_verdict(state) != "fail":
            return None
        age_s = (state.time_ns - state.t_last_visual_ns) / 1e9
        return (
            f"visual updates stopped: the newest estimate's last visual update "
            f"is {age_s:.2f} s old, past the declared fail bound "
            f"{self._machine.bounds.visual_update_fail_s:.2f} s (F4)"
        )

    def pump_perception(self, drain, sim_seconds: float, until=None) -> int:
        """Pump perception on its declared cadence for a bounded sim window.

        Perception is the other half of the sensor path whose estimator feed
        already runs on its own thread, and it must not be coupled to whether a
        goal happens to be executing. live-motion-8 made 7 observations in 64
        simulated seconds while dropping 593 frames: no goal could be admitted,
        so the lease loop that pumped perception never turned, so the map could
        not grow, so no goal could be admitted. The map grows from frames and a
        mission that cannot see cannot decide, so gathering has to continue
        while the mission has nothing to do.

        The budget is spent as a COUNT of cycles — ``sim_seconds`` at one cycle
        per ``MIN_PERCEPTION_INTERVAL_S`` of the aircraft's own time — and not
        as a comparison against a running simulator clock. The distinction is
        measured, not stylistic: ``_SimWindow`` expires when the simulator's own
        clock has advanced the budget, and the simulator delivers its time in
        bursts, so on live-vision-2 the cold start's 8 s window expired after a
        SINGLE cycle, and a gathering that should have produced about 26
        observations produced one. The wall ceiling is kept, so a stalled
        simulator ends the wait instead of hanging the run.

        ``until`` is an optional predicate: gathering stops as soon as it is
        true, so a caller waiting for a real target need not spend the whole
        budget once it has one.

        Only for use once the ordered bring-up has finished: this runs stereo
        depth, and the ``drain`` inside it is the path the bring-up's
        RC-throttle override must never wait behind — live-10 starved that
        override exactly this way and the aircraft tipped over at roll -90 deg.
        Every call site below is after the bring-up, in the mission's phases.

        Returns the number of cycles that took a frame.
        """
        target_cycles = max(1, int(round(sim_seconds / MIN_PERCEPTION_INTERVAL_S)))
        deadline_s = time.monotonic() + _sim_window_wall_ceiling_s(
            sim_seconds, self.settings.realtime_ratio_envelope[0]
        )
        cycles = 0
        next_cycle = 0.0
        while cycles < target_cycles and time.monotonic() < deadline_s:
            fault = self._visual_fault()
            if fault is not None:
                self._visual_fault_reason = fault
                self.result.log.append(f"visual update fault: {fault}")
                break
            now = time.monotonic()
            if now >= next_cycle:
                next_cycle = now + MIN_PERCEPTION_INTERVAL_S
                t_drain_start = time.monotonic()
                drain()
                t_frame_start = time.monotonic()
                before = self._perception_cycles
                self.perceive_if_due()
                t_end = time.monotonic()
                self._perception_drain_s += t_frame_start - t_drain_start
                self._perception_depth_s += t_end - t_frame_start
                if self._perception_cycles > before:
                    cycles += 1
                    # The pump is exactly the mission-had-nothing-to-do state the
                    # vantage policy answers: perception ran, the selection (if
                    # any) refused, and nothing else was flying. The consume is
                    # bounded by the sweep's own once-per-view flag, so it cannot
                    # turn into a second sweep inside this window.
                    self._consume_vantage_sweep()
                    if until is not None and until():
                        break
            time.sleep(PERCEPTION_PUMP_SLEEP_S)
        return cycles

    def perceive_if_due(self) -> None:
        """Depth, candidates and map integration on the newest queued pair.

        One observation event per cycle, with the pair's payloads stored only
        when the frame grounded a candidate — the evidence a claim can cite.
        """
        self._beat_watchdog()
        record = None
        try:
            while True:
                record = self._perception_queue.get_nowait()
        except queue.Empty:
            pass
        if record is None or record.pair is None:
            return
        # A cycle that took a frame. This, not the drop counter, is what says
        # whether perception ran: dropped frames are the queue discarding an
        # older snapshot in favour of a newer one, which the design intends.
        self._perception_cycles += 1
        pair = record.pair
        # One clock for the map: its evidence stamps and its freshness clock are
        # both the controller's capture clock, which is the clock the poses are
        # transformed on. Stamping the map with simulator time while measuring
        # its age on the host clock would make every cell read stale the moment
        # it was written.
        self._capture_clock_ns = max(self._capture_clock_ns or 0, int(pair.capture_host_ns))
        # The alignment has to be sealed before a capture-time pose can be built:
        # an unsealed one has no odom rotation, and its own accessor refuses
        # rather than guessing the frame's unobservable yaw. So ask for the
        # navigation state FIRST and record nothing until the estimator has
        # defined the epoch. Perception now starts during the bring-up, on the
        # way up, so "not sealed yet" is the ordinary case rather than an error.
        state = self._navigation_state()
        if state is None:
            return
        pose = self._capture_pose(record)
        if pose is None:
            return
        try:
            left = np.frombuffer(pair.left_bytes, dtype=np.uint8).reshape(
                pair.height, pair.width, 3
            )
            right = np.frombuffer(pair.right_bytes, dtype=np.uint8).reshape(
                pair.height, pair.width, 3
            )
            outcome = self.proposer.candidates(left, self._query)
            depth = camera_module.compute_validated_depth(
                left,
                right,
                self.calibration,
                self._depth_settings,
                pair_id=f"{self.episode_id}-pair-{pair.pair_id:06d}",
                capture_stamp=pose.stamp,
                receipt_stamp=record.received_stamp,
                sim_time_s=record.sim_time_s if record.sim_time_s >= 0.0 else None,
                pose_provenance=camera_module.PoseProvenance(
                    label="SENSOR_DERIVED",
                    detail="the live estimator's pose nearest the pair's capture instant",
                ),
            )
        except Exception as error:  # a failed depth product clears nothing
            self._note_perception_refusal(f"depth_failed: {error}")
            return
        if isinstance(outcome, DetectorUnavailable):
            # A seam that refuses is a vision failure as much as a refused
            # grounding is, and this one was discarded silently: the mission
            # recorded nothing about why it proposed no candidate at all.
            self._note_perception_refusal(
                f"detector_unavailable {outcome.reason}: {outcome.detail}"
            )
        candidates = () if isinstance(outcome, DetectorUnavailable) else outcome
        if not candidates:
            # A frame that yields no candidate of the query is the ordinary
            # case for a colour proposer in a room the target is not visible
            # from, and it is the number a reader needs to tell "the mission
            # looked and saw nothing to aim at" from "the mission never looked".
            self._frames_without_candidate += 1
        observation = self._record_observation(record, store_payload=bool(candidates))
        self._append_observation_id(observation.record_id)
        # A cloud arm's call carries the observation's own frames, as PPM,
        # because the packet builder re-encodes them to PNG itself. Built only
        # for a cloud arm: the conventional arm sends nothing and would be
        # paying for a copy it never uses.
        frames: dict[str, bytes] = {}
        if self._pilot is not None:
            frames = {
                "left": ppm_bytes(pair.left_bytes, pair.width, pair.height),
                "right": ppm_bytes(pair.right_bytes, pair.width, pair.height),
            }
            if not self._airborne and self._preflight_observation is None:
                # The first frame the aircraft takes while it is still on the
                # ground is the evidence the one reasoned call is made from.
                self._preflight_observation = observation
                self._preflight_payloads = frames
        self.store.integrate(
            depth,
            pose,
            self.calibration,
            stamp_ns=int(pair.capture_host_ns),
            observation_id=observation.record_id,
            now_ns=self._now_ns(),
        )
        grounded = self._ground_candidates(candidates, observation, depth, pose, state)
        for target in grounded:
            if target.target_id in self._grounded:
                continue
            self._grounded[target.target_id] = target
            self._candidate_targets.append(target.target_id)
            self._candidate_observation_ids.append(observation.record_id)
            self.result.log.append(
                f"candidate grounded: {target.target_id} from {observation.record_id}"
            )
        if self._pilot is not None and self._airborne:
            # In flight the builder's latch is closed, so this can only ever be
            # the continuous class, and B1 makes no call at all.
            self._tick_cloud(observation, frames)

    def _ground_candidates(self, candidates, observation, depth, pose, state):
        grounded: list[R.GroundedTarget] = []
        for candidate in candidates:
            if candidate.region is None:
                continue
            u0, v0, u1, v1 = candidate.region
            selection = R.VisualSelection(
                selection_id=(
                    f"sel-{observation.sequence:05d}-{candidate.candidate_id}"
                ),
                observation_id=observation.record_id,
                coordinate_convention="pixel_uv_top_left_origin",
                geometry_kind=R.SelectionGeometry.POINT,
                geometry=(float((u0 + u1) / 2.0), float((v0 + v1) / 2.0)),
                crop_transform=None,
                description=f"candidate {candidate.candidate_id} of query {self._query!r}",
                confidence=None,
            )
            self._sink("selection", R.to_dict(selection), self._clock(), observation.sim_time_s)
            target = G.ground(
                selection, observation, depth, pose, state, self.calibration
            )
            if isinstance(target, G.Refusal):
                self._note_perception_refusal(
                    f"grounding_refused {target.reason}: {target.detail}"
                )
                # The refusal is honest and stays the refusal (no bound moves).
                # What was missing is the policy that reads it: a selection that
                # keeps refusing unknown_geometry is the vantage's verdict, and
                # the streak below is what converts it into a change of view
                # instead of a 20th selection of the same sliver.
                if target.reason == G.REFUSAL_UNKNOWN_GEOMETRY:
                    self._note_unmeasurable_selection(selection)
                else:
                    self._reset_unmeasurable_streak()
                continue
            grounded.append(target)
        if grounded:
            # A grounding is a measurable view: the streak was about a selection
            # that could not be measured, and that question is answered.
            self._reset_unmeasurable_streak()
        return grounded

    def _selection_bearing(self, selection: R.VisualSelection) -> float:
        """The selection's horizontal bearing in the commanded-yaw frame.

        The selection is image geometry and nothing else — the whole point of
        the refusal is that no depth exists to place it in the world. Its
        horizontal angle off the camera axis is ``atan2(u - u0, f)`` on the
        declared pinhole model (the calibration's own focal length and
        principal point), and the camera's heading is the mission's declared
        hold yaw on every published goal (``publish_active`` commands
        ``hold_yaw_rad`` for certificates and holds alike). So the bearing the
        vantage sweep must start from is the hold yaw plus that offset, in the
        same frame the yaw commands live in: a selection left of the image
        centre carries a negative offset, matching the frame's yaw sign.
        """
        u, _v = (float(value) for value in selection.geometry)
        intrinsics = self.calibration.left_intrinsics
        return MISSION_YAW_HOLD_RAD + math.atan2(
            u - float(intrinsics.principal_point_px[0]),
            float(intrinsics.focal_length_px[0]),
        )

    def _note_unmeasurable_selection(self, selection: R.VisualSelection) -> None:
        """Count one more consecutive unknown_geometry refusal of the query.

        GROUNDING-INVALID-DEPTH.md: J52 refused the same selection 19 times and
        the count went nowhere. The streak is what the vantage response reads
        (``_consume_vantage_sweep``); the bearing is remembered so the sweep it
        triggers starts at the selection rather than at the hold heading.
        """
        self._unmeasurable_streak += 1
        self._unmeasurable_bearing_rad = self._selection_bearing(selection)

    def _reset_unmeasurable_streak(self) -> None:
        """The view was measured (or refused for a different reason): start over."""
        self._unmeasurable_streak = 0

    def _note_new_vantage(self) -> None:
        """A new goal was admitted and will fly: the standing view is a new one.

        This is the sweep's only re-arm. Bounded by it, a refuse-sweep episode
        cannot loop: the refusals themselves never re-arm the sweep, so a
        selection that stays unmeasurable is answered once per view — with the
        sweep and then with explore, not with another selection.
        """
        self._unmeasurable_streak = 0
        self._vantage_sweep_spent = False

    def _consume_vantage_sweep(self) -> None:
        """Convert a persistent unknown_geometry refusal into a change of view.

        The response uses the machinery the runtime already owns: the
        section-12.2 observation objective, started at the selection's bearing,
        turning the aircraft where it stands. Nothing translates, so there is
        no route to certify and no unevidenced space is entered. The sweep's
        honest terminal is "not measurable from here": a refusal after a real
        look is a valid outcome and is recorded as one — explore, not another
        selection, is what follows.
        """
        if self._vantage_sweep_spent:
            return
        if self._unmeasurable_streak < VANTAGE_REFUSAL_SWEEP_CYCLES:
            return
        self._vantage_sweep_spent = True
        bearing = self._unmeasurable_bearing_rad
        before = len(self._candidate_targets)
        observed, goal_id = self.observe_in_place(
            target_ref=None,
            refused_reason=(
                f"{G.REFUSAL_UNKNOWN_GEOMETRY} on {self._unmeasurable_streak} "
                "consecutive selections of the query"
            ),
            bearing_rad=bearing,
        )
        grounded_during = len(self._candidate_targets) - before
        if grounded_during > 0:
            self.result.log.append(
                f"vantage sweep {goal_id}: the look grounded {grounded_during} "
                "candidate(s) of the query"
            )
        elif observed:
            self.result.log.append(
                f"vantage sweep {goal_id}: not measurable from here — the honest "
                "outcome of a full look at the selection's bearing is a refusal; "
                "explore continues, not another selection"
            )

    def _renewal_hold_position(
        self, current: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        """The station hold's position after a refused certificate renewal.

        The hold is the supported station keep of specification 14.2: it holds
        the aircraft FLYING while the map re-evidences. It must hold it at the
        declared hover altitude, not at whatever altitude the blocked excursion
        had descended to. J52-discrim-1 measured the difference: the explore
        vantage walk descended the aircraft to ~0.2 m, the renewal was refused
        (`unsupported_space`), and the hold then pinned it there — the
        autopilot's landing detector fired and its AUTO_DISARMING_DELAY (the
        pin's declared 10 s, restored after bring-up) disarmed the parked
        aircraft, so every later step ran grounded: 16 publications refused
        "not in armed Guided flight", mode GUIDED, armed False (RETURN-
        UNSUPPORTED.md; the run's refused-publications.jsonl).

        The lateral position is unchanged — a hold does not translate. The
        vertical target is the declared cruise altitude
        (`probe.hover_altitude_m`), the same altitude the return target is
        built at and the bring-up climbs to; the higher of the aircraft's
        current altitude and it is kept, so the hold climbs out of a
        landed-looking altitude but never commands a descent. In the aligned
        NED frame down is positive, so "higher" is the smaller z.
        """
        if current is None:
            return None
        if not self.alignment.sealed:
            # No frame yet: the declared altitude cannot be expressed. Hold at
            # the current position, as before — the unsealed case is bring-up,
            # where the aircraft is parked by design.
            return current
        origin = self.alignment.aligned_position_ned((0.0, 0.0, 0.0))
        hover_z = origin[2] - self.settings.hover_altitude_m
        return (current[0], current[1], min(current[2], hover_z))

    def _record_observation(self, record: Any, *, store_payload: bool) -> R.Observation:
        self._observation_counter += 1
        sequence = self._observation_counter
        pair = record.pair
        host_id = str(self._controller_status.get("host_id") or self.settings.host_id)
        clock_id = str(self._controller_status.get("clock_id") or self.settings.clock_id)
        left_payload = right_payload = None
        if store_payload and self._payload_count < MAX_EVIDENCE_PAYLOADS:
            left_name = f"obs-{sequence:05d}-left.ppm"
            right_name = f"obs-{sequence:05d}-right.ppm"
            self.recorder.write_payload(
                left_name, ppm_bytes(pair.left_bytes, pair.width, pair.height)
            )
            self.recorder.write_payload(
                right_name, ppm_bytes(pair.right_bytes, pair.width, pair.height)
            )
            left_payload, right_payload = f"payloads/{left_name}", f"payloads/{right_name}"
            self._payload_count += 1
        observation = R.Observation(
            episode_id=self.episode_id,
            record_id=f"{self.episode_id}-obs-{sequence:05d}",
            sensor_ids=R.SensorIds(
                left=self.settings.stereo.left,
                right=self.settings.stereo.right,
                imu=self.settings.imu.inertial_unit,
            ),
            sequence=sequence,
            capture_stamp=R.ClockStamp(
                host_id=host_id, clock_id=clock_id, monotonic_ns=int(pair.capture_host_ns)
            ),
            receipt_stamp=record.received_stamp,
            sim_time_s=record.sim_time_s if record.sim_time_s >= 0.0 else None,
            # The pair id travels with the payloads or not at all: the observation
            # record requires all three or none of them (records.py, "an
            # observation carries the pair id and both payloads, or none of
            # them"), and this runtime stores the frames only for an observation
            # that grounded a candidate — the evidence a claim can cite. Setting
            # the id unconditionally is a half-reference the recorder refuses,
            # and it was invisible while perception never ran.
            pair_id=(
                f"{self.episode_id}-pair-{pair.pair_id:06d}"
                if left_payload is not None
                else None
            ),
            left_payload=left_payload,
            right_payload=right_payload,
            encoding=pair.encoding,
            width=pair.width,
            height=pair.height,
            calibration_id=self.calibration.calibration_id,
            capture_pose_ref=None,
            quality=_quality_of(pair, self.settings),
            depth_source=None,
        )
        # The EVENT is stamped now; the pair's own capture and receipt stamps
        # live inside the record. Stamping the event with the receipt instant
        # would let an event recorded later (a setpoint published while the
        # frame waited in the perception queue) carry a later stamp than this
        # one, and the recorder refuses a stream whose order and stamps
        # disagree.
        self._sink("observation", R.to_dict(observation), self._clock(), record.sim_time_s)
        self._last_observation_id = observation.record_id
        return observation

    def _sim_now_s(self) -> float:
        """The simulator's own clock, in seconds — the one the declared windows use."""
        return self._stats.sim_clock.newest_s or 0.0

    def observe_in_place(
        self,
        *,
        target_ref: str | None,
        refused_reason: str,
        bearing_rad: float | None = None,
    ) -> tuple[bool, str]:
        """Section 12.2's conditional observation objective, admitted and flown.

        Called when a traversal could not be admitted, or when the query's
        selection has refused unknown_geometry for the declared consecutive
        count (the vantage policy). The aircraft holds its position and turns
        once, so its camera covers headings no earlier frame covered. Nothing
        translates, so there is no route to certify and no space the map has
        not evidenced is entered — which is why this is admissible when the
        traversal is not.

        ``bearing_rad`` starts the turn at a heading of interest (a refusing
        selection's bearing) instead of the mission's hold yaw. The sweep still
        runs for the whole declared window at the declared rate, so it covers
        every heading either way; starting at the bearing puts the object that
        could not be measured near-axis first, which is the geometry the
        diagnosis measured grounding from (large, near-axis, unclipped).

        It returns BOTH facts, as the specification requires: the traversal stays
        unadmitted and keeps its reason, and the observation is recorded with what
        it cost and what it saw. It is bounded by OBSERVATION_MAX_PER_TARGET, so
        the refuse-observe-re-propose loop cannot spin.
        """
        position = self._position_odom()
        if position is None:
            return False, "no position to observe from"
        self._observe_count += 1
        goal_id = f"observe-{self._observe_count}"
        self._active_goal = _ActiveGoal(
            goal_id=goal_id,
            hold_position_odom=position,
            hold_yaw_rad=MISSION_YAW_HOLD_RAD if bearing_rad is None else bearing_rad,
            hold_yaw_rate_rad_s=OBSERVATION_YAW_RATE_RAD_S,
            hold_since_sim_s=self._sim_now_s(),
        )
        window = _SimWindow(
            self._stats.sim_clock,
            OBSERVATION_SWEEP_SIM_S,
            label="the observation window",
            wall_ceiling_s=_sim_window_wall_ceiling_s(
                OBSERVATION_SWEEP_SIM_S, self.settings.realtime_ratio_envelope[0]
            ),
        )
        before = self._observation_counter
        published = 0
        next_perception = 0.0
        next_publish = 0.0
        stopped = "the window closed"
        while not window.expired():
            self._beat_watchdog()
            if self._visual_fault_reason is not None:
                stopped = f"the estimate stopped being a pose: {self._visual_fault_reason}"
                break
            now = time.monotonic()
            if now >= next_perception:
                next_perception = now + MIN_PERCEPTION_INTERVAL_S
                self.perceive_if_due()
            if now >= next_publish:
                next_publish = now + SETPOINT_PERIOD_S
                failure = self.publish_active()
                if failure is not None:
                    stopped = f"publication stopped: {failure}"
                    break
                published += 1
            time.sleep(PERCEPTION_PUMP_SLEEP_S)
        seen = self._observation_counter - before
        # The observation goal is not the mission's active goal. Clearing it keeps
        # `publish_active` honest: with no admitted traversal there is no goal to
        # publish, and the next step must admit one rather than inherit a hold.
        self._active_goal = None
        subject = (
            f"traversal {target_ref!r}" if target_ref is not None
            else "the query's unmeasurable selection"
        )
        self.result.log.append(
            f"observation objective {goal_id}: {subject} stays unadmitted "
            f"({refused_reason}); held position and turned {OBSERVATION_YAW_RATE_RAD_S} rad/s "
            f"for {OBSERVATION_SWEEP_SIM_S:.1f} sim s from "
            f"{(MISSION_YAW_HOLD_RAD if bearing_rad is None else bearing_rad):.3f} rad, "
            f"{published} setpoint(s), "
            f"{seen} further observation(s); ended: {stopped}"
        )
        return True, goal_id

    # -- publication: the single setpoint path --------------------------------

    def publish_active(self) -> str | None:
        """Publish one sample of the active goal's certified prefix.

        Returns None on success, or the reason publication stopped. Every
        motion target this runtime puts on the wire goes through this one
        method and the platform's one publisher.
        """
        # R24: the run's own record must say why the vision feed stopped. The
        # publisher owns the trip record; this line is what it owes the receipt,
        # surfaced at the first setpoint tick after the trip.
        guard_line = self._publisher.z_guard_receipt_line() if self._publisher else None
        if guard_line is not None:
            self.result.log.append(guard_line)
        active = self._active_goal
        if active is None:
            return "no_active_goal"
        certificate = active.certificate
        if certificate is not None:
            sample_t = min(time.monotonic(), certificate.t_end_s)
            position_ned, velocity_ned, _acceleration = certificate.sample(sample_t)
            certificate_ref = certificate.certificate_id
            self._log_certificate_shape(certificate)
        elif active.hold_position_odom is not None:
            position_ned = active.hold_position_odom
            velocity_ned = (0.0, 0.0, 0.0)
            certificate_ref = active.goal_id
        else:
            return "no_published_trajectory"
        # The commanded yaw. A zero rate is the mission's declared hold yaw. A
        # non-zero one is an observation: the aircraft turns where it stands, so
        # the camera sees parts of the room no earlier frame covered, and the
        # space it needs to certify stops being space nobody has looked at.
        yaw_rad = active.hold_yaw_rad
        if active.hold_yaw_rate_rad_s:
            # Measured on the SAME clock the sweep's declared window is spent on.
            # On the wall clock the swept angle depended on the host's own speed:
            # J44-fly-1 swept 3.08 rad against a declared 6.0. Clamped to the
            # declared window, so a clock that jumps cannot command an angle the
            # sweep never declared.
            elapsed_sim_s = min(
                self._sim_now_s() - active.hold_since_sim_s, OBSERVATION_SWEEP_SIM_S
            )
            yaw_rad = active.hold_yaw_rad + elapsed_sim_s * active.hold_yaw_rate_rad_s
        sent = self._platform.send_local_ned(
            LocalNedTarget(
                position_ned=position_ned,
                velocity_ned=velocity_ned,
                yaw_rad=yaw_rad,
                deadline_s=SETPOINT_DEADLINE_S,
                certificate_ref=certificate_ref,
            )
        )
        if sent is None:
            self._publish_refusal_count += 1
            return "guided_flight_lost"
        self._publication_count += 1
        self._record_commanded_setpoint(
            position_ned=position_ned,
            velocity_ned=velocity_ned,
            yaw_rad=yaw_rad,
            certificate_ref=certificate_ref,
            goal_ref=active.goal_id,
        )
        active.setpoints = (*active.setpoints[-3:], sent.setpoint)
        self._sink("setpoint", R.to_dict(sent.setpoint), sent.published_stamp, None)
        return None

    def _record_commanded_setpoint(
        self,
        *,
        position_ned,
        velocity_ned,
        yaw_rad: float,
        certificate_ref: str,
        goal_ref: str,
    ) -> None:
        """Record what was actually commanded, not merely that something was.

        The run kept a COUNT of publications, so no artifact could answer
        whether the aircraft was told to climb and never went anywhere: the
        commanded vertical target was written down nowhere. One sample per
        COMMANDED_SAMPLE_SIM_S of simulator time is enough to put the command
        beside the achieved altitude. Nothing here changes what is sent; it
        records it, in the frame it was sent in, with the goal and certificate
        it came from.
        """
        sim_now = self._stats.sim_clock.newest_s
        if sim_now is None:
            return
        if (
            self._last_commanded_sim_s is not None
            and sim_now - self._last_commanded_sim_s < COMMANDED_SAMPLE_SIM_S
        ):
            return
        self._last_commanded_sim_s = sim_now
        self._commanded.append(
            {
                "sim_s": round(float(sim_now), 3),
                "frame": "local_ned",
                "position_m": [round(float(value), 4) for value in position_ned],
                "velocity_mps": [round(float(value), 4) for value in velocity_ned],
                "yaw_rad": round(float(yaw_rad), 4),
                "goal_ref": goal_ref,
                "certificate_ref": certificate_ref,
            }
        )
        if len(self._commanded) > MAX_COMMANDED_SAMPLES:
            del self._commanded[0]

    def _log_certificate_shape(self, certificate) -> None:
        """Record once, per certificate, what it actually commands.

        ``publish_active`` samples the certified prefix at ``time.monotonic()``,
        so the question "is this trajectory a hover?" is answered by the
        certificate's own endpoints. live-17 published the aircraft's own
        position with zero velocity for a whole flight, and the shape of the
        curve is the evidence that says why.
        """
        if getattr(self, "_logged_certificate", None) == certificate.certificate_id:
            return
        self._logged_certificate = certificate.certificate_id
        first = certificate.sample(certificate.t_start_s)
        last = certificate.sample(certificate.t_end_s)
        now = time.monotonic()
        self.result.log.append(
            "certificate {cid}: t {t0:.3f}..{t1:.3f}, wall now {now:.3f} "
            "(first sample ({a0:.2f},{a1:.2f},{a2:.2f}) |t0, "
            "({b0:.2f},{b1:.2f},{b2:.2f}) |t1)".format(
                cid=certificate.certificate_id,
                t0=certificate.t_start_s,
                t1=certificate.t_end_s,
                now=now,
                a0=first[0][0], a1=first[0][1], a2=first[0][2],
                b0=last[0][0], b1=last[0][1], b2=last[0][2],
            )
        )

    # -- frontier helpers ------------------------------------------------------

    def _flyable_band_z(self) -> tuple[float, float] | None:
        """The NED z band every goal this mission admits must stay inside, or None.

        J54-zguard-1 measured the crash head-on: the admitted explore goal
        regions carried z bands reaching +0.2 .. +0.8 — at and below the floor
        datum — the aircraft chased one, dove below the floor plane and flipped
        (``crash_disarm: AngErr=170``). The mission's goals fly at the declared
        hover altitude, so the band is DERIVED (R2), not declared:

        * the band is the declared hover band: the aligned origin's z minus
          ``settings.hover_altitude_m`` — the same composition the return
          target and the renewal hold are built from — widened by the
          executor's own terminal-region half-extent, read off
          ``approach_region``'s geometry rather than restated. A region built
          around an in-band point therefore stays inside the band's faces too:
          the offered vantage, the approach region and the terminal region are
          constrained together.
        * the map's own occupancy then TIGHTENS the band where it states a
          surface beyond it. The floor and the ceiling are the map's occupied
          evidence classes: the deepest (largest z, NED down) and the highest
          occupied cell centres, each pulled half a declared voxel to its
          room-side face. A side tightens only when its evidence reaches
          beyond the hover band — the only case in which the map can correct
          the declared band — so a partial map (one observed wall patch at eye
          level, no floor or ceiling seen yet) leaves the declared band intact
          instead of mistaking its own span for the room's.
        * the grid's z bound is deliberately looser than the floor — it admits
          a landed vehicle's own estimate — so it is not the floor and is not
          used as one. The declared hover band alone already refuses J54's
          floor-piercing vantages: hover + half a region is 0.95 m above the
          spawn datum, and J54's admitted vantage sat 0.25 m BELOW it.

        ``None`` means the band cannot be stated: the alignment is unsealed (no
        mission frame, so no declared altitude is expressible — nothing
        resolves or is offered in that state anyway), or the runtime is the
        offline half some tests construct, which carries neither attribute. A
        stated band is always returned — the declared hover band when the map
        states no surface beyond it, tightened where it does — even when the
        derivation leaves it empty: an empty band then refuses every vantage
        honestly (every candidate counts ``out_of_band``) instead of silently
        dropping the gate.
        """
        alignment = getattr(self, "alignment", None)
        settings = getattr(self, "settings", None)
        if alignment is None or settings is None or not alignment.sealed:
            return None
        origin = alignment.aligned_position_ned((0.0, 0.0, 0.0))
        hover_z = origin[2] - settings.hover_altitude_m
        half_region = GE.approach_region((0.0, 0.0, 0.0), ENVELOPE).extent()[2] / 2.0
        low = hover_z - half_region
        high = hover_z + half_region
        config = self.store.config
        occupied = self.store.occupied_cells(now_ns=self._now_ns())
        if not occupied:
            return (low, high)
        centres_z = [config.cell_center(cell)[2] for cell in occupied]
        half_face = config.voxel_m / 2.0
        floor_face_z = max(centres_z) - half_face
        ceiling_face_z = min(centres_z) + half_face
        if floor_face_z > high:
            # The map has seen a surface below the band's bottom face: the
            # floor is stated, and the band's bottom keeps a region's height
            # above it.
            high = min(high, floor_face_z - half_region)
        if ceiling_face_z < low:
            # The map has seen a surface above the band's top face: the
            # ceiling is stated, and the band's top keeps a region's height
            # below it.
            low = max(low, ceiling_face_z + half_region)
        return (low, high)

    def frontier_regions(self) -> dict[str, GE.BoxRegion]:
        """The current frontier clusters, one terminal region each."""
        cells = GE.frontier_cells(self.store, now_ns=self._now_ns())
        stride = FRONTIER_CLUSTER_CELLS
        clusters: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        for cell in cells:
            clusters.setdefault(
                (cell[0] // stride, cell[1] // stride, cell[2] // stride), []
            ).append(cell)
        voxel = self.store.config.voxel_m
        regions: dict[str, GE.BoxRegion] = {}
        for key in sorted(clusters):
            centers = [self.store.config.cell_center(cell) for cell in clusters[key]]
            low = tuple(min(c[axis] for c in centers) - voxel for axis in range(3))
            high = tuple(max(c[axis] for c in centers) + voxel for axis in range(3))
            regions[f"frontier:{key[0]}:{key[1]}:{key[2]}"] = GE.BoxRegion(
                low=low, high=high, label="frontier"
            )
        return regions

    def _frontier_goal_point(
        self,
        region: GE.BoxRegion,
        *,
        here: tuple[float, float, float] | None = None,
        searchable: set[tuple[int, int, int]] | None = None,
        searchable_bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
        band: tuple[float, float] | None = None,
    ) -> tuple[float, float, float] | None:
        """The point an explore step would fly to for this frontier, or None.

        ``band`` is the flyable band (:meth:`_flyable_band_z`): when it is
        stated, a candidate whose z falls outside it is refused at the walk —
        counted ``out_of_band``, never returned, never silently clamped. This
        is the fix J54-zguard-1 asked for: the walk used to accept a vantage at
        any altitude the searchable cells supported, the admitted region's z
        band reached below the floor, and the aircraft dove into it. ``None``
        (the band cannot be stated: unsealed alignment, or the offline half
        some tests construct) skips the gate; nothing resolves or is offered in
        that state anyway.

        A frontier is a boundary between observed free space and unknown
        space: an observation opportunity, not a destination (specification
        11). A goal AT the boundary cannot be admitted, because the standoff
        region the executor builds around it overlaps space the map has no
        evidence for — which is how live-12's two explore goals were refused.
        So the vantage is searched back along the line to the aircraft, which
        stands in known free space by construction, and the first point the
        planner could admit is preferred.

        **A gate the aircraft is already standing in is not an excursion.**
        The executor's approach region is centred ``STANDOFF_M`` behind the
        target, so for a frontier that wraps the aircraft the only candidate
        left admissible is ``here + STANDOFF_M * direction`` — and the
        approach region of THAT point contains the aircraft itself. Such a
        goal is satisfied the instant it is admitted: the certified prefix is
        a hover, the step completes on arrival without moving, and the map
        never grows. That is exactly what live-17 did for a whole flight while
        every certificate said "explore" — 19 setpoints, every one commanding
        the aircraft's own x and y with zero velocity. Candidates whose goal
        region already contains the aircraft are therefore rejected, and when
        no candidate survives, this frontier has no vantage and is not a place
        to fly to. Returning ``None`` is the honest answer; the caller
        declines the frontier rather than publishing a hover as progress.
        How close a candidate may be is ``EXCURSION_STANDOFF_M``, derived from the
        searchable shell the exemption provides rather than borrowed from the
        approach standoff: with the map carrying no evidence-supported clear space,
        a goal region can only reach into the exemption bubble, and that bubble is
        0.198 m deep once a cell's clearance ball is fitted inside it.
        """
        centre = tuple(float(value) for value in region.center())
        here = here if here is not None else self._position_odom()
        if here is None:
            return None
        direction = np.asarray(FRONTIER_VIEW_DIRECTION, dtype=np.float64)
        ideal = np.asarray(centre, dtype=np.float64) - GE.STANDOFF_M * direction
        toward = np.asarray(here, dtype=np.float64)
        span = float(np.linalg.norm(toward - ideal))
        steps = max(1, int(span / FRONTIER_VANTAGE_STEP_M))
        if searchable is None:
            searchable = self._searchable_cells(here)
        if searchable_bounds is None:
            searchable_bounds = _index_bounds(searchable)
        # The planner admits a goal when at least one cell of its goal region is
        # in the set it may search, so this asks exactly that and no weaker
        # question: the same inflation, the same extra margin the planner adds,
        # and the aircraft's own position excluded as an obstacle. A weaker test
        # clears vantages the planner then refuses, which is how live-14's
        # explore goals were still rejected after the first attempt at this.
        for index in range(steps + 1):
            vantage = ideal + (index / steps) * (toward - ideal)
            target = vantage + GE.STANDOFF_M * direction
            target_point = tuple(float(value) for value in target)
            if band is not None and not band[0] <= target_point[2] <= band[1]:
                # The flyable band (J54-zguard-1): a goal here would command
                # the aircraft below the floor or above the ceiling. Refused
                # at the walk with its own tally -- the view direction is
                # horizontal, so the terminal region around this point spans
                # exactly the z the band forbids.
                self._count_gate("out_of_band")
                continue
            if float(np.linalg.norm(target - toward)) < GE.STANDOFF_M:
                # Closer to what it is looking at than the mission's own
                # declared standoff: there is no approach left to make.
                self._count_gate("too_close")
                continue
            approach = GE.approach_region(
                target_point,
                ENVELOPE,
                direction=FRONTIER_VIEW_DIRECTION,
            )
            if _region_gap_m(approach, here) < EXCURSION_STANDOFF_M:
                # A goal the aircraft already stands in -- or stands two
                # centimetres outside of -- is not an excursion. The step
                # completes on arrival without going anywhere, which is what
                # live-motion-7 did: its admitted region's near face was
                # 0.02 m from the aircraft, so "explore" advanced 2 cm. A view
                # from where you already are is not a second view.
                #
                # The measure is EXCURSION_STANDOFF_M, not the approach standoff:
                # this asks only how far the goal's region must be from the
                # aircraft, and the value is derived from the searchable shell the
                # exemption provides. See its definition above.
                self._count_gate("too_near")
                continue
            if _region_holds_searchable(
                approach, searchable, searchable_bounds, self.store.config
            ):
                self._count_gate("accepted")
                return target_point
            self._count_gate("not_searchable")
        return None

    def _count_gate(self, name: str) -> None:
        """Tally why a candidate vantage was or was not returned.

        Diagnostics only, and tolerant of a runtime built without __init__
        (the tests construct one that way): the counters exist to be logged,
        never to change a decision.
        """
        counts = getattr(self, "_gate_counts", None)
        if counts is None:
            counts = {}
            self._gate_counts = counts
        counts[name] = counts.get(name, 0) + 1

    def map_summary(self) -> dict[str, object]:
        """What the map actually holds, in the terms planning depends on.

        ``free_cells`` are cells the map publishes free; the searchable set is
        those free with the whole declared envelope around them free too. A map
        can hold many of the first and none of the second, and that difference
        is the whole question of whether the aircraft has anywhere to go.
        """
        now = self._now_ns()
        free = self.store.free_cells(now_ns=now)
        here = self._position_odom()
        searchable = self._searchable_cells(here) if here is not None else set()

        def bbox(cells):
            if not cells:
                return None
            centres = [self.store.config.cell_center(cell) for cell in cells]
            return tuple(
                (round(min(c[axis] for c in centres), 2), round(max(c[axis] for c in centres), 2))
                for axis in range(3)
            )

        return {
            "free_cells": len(free),
            "searchable_cells": len(searchable),
            # The decisive number: of the cells called searchable, how many the
            # map itself publishes free. The rest come from the self-occupied
            # exemption, which marks the aircraft's own envelope free by
            # construction — so a count of one can mean the map holds none.
            "searchable_from_evidence": sum(1 for cell in searchable if cell in free),
            "here": None if here is None else tuple(round(value, 2) for value in here),
            # Which rule withheld the observed cells, and what disqualifies the
            # ball around a free cell. Both are diagnostics: they count, and no
            # decision reads them.
            "evidence": self.store.evidence_breakdown(now_ns=now),
            "free_cell_fates": self._free_cell_fates(free, now=now),
            "free_bbox": bbox(free),
            "searchable_bbox": bbox(searchable),
            "frontiers": len(self.frontier_regions()),
        }

    def _free_cell_fates(
        self, free: frozenset, *, now: int, sample: int = 60
    ) -> dict[str, int]:
        """For a bounded sample of free cells, what disqualified the envelope ball.

        A map can publish many free cells and no cell whose whole declared
        envelope is free, and that difference is whether the aircraft has
        anywhere to go. Walking every free cell against every offset in the ball
        costs as much as the integration that built the map, so a sample is
        taken: this answers a proportion, which needs a sample, and it is logged
        from the same path as the counts it explains.
        """
        cells = sorted(free)
        if not cells:
            return {}
        stride = max(1, len(cells) // sample)
        picked = cells[::stride][:sample]
        radius = ENVELOPE.inflation_m + self.store.config.voxel_m / 4.0
        offsets = GE.ball_offsets(self.store.config, radius)
        fates: dict[str, int] = {}
        for cell in picked:
            for dx, dy, dz in offsets:
                neighbour = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                if not self.store.config.inside(neighbour):
                    reason = "outside_map"
                elif neighbour in free:
                    continue
                else:
                    state = self.store.classify(neighbour, now_ns=now)
                    reason = (
                        "occupied"
                        if state == world_module.OCCUPIED
                        else self.store.unknown_reason(neighbour, now_ns=now)
                    )
                fates[reason] = fates.get(reason, 0) + 1
        return fates

    def _searchable_cells(
        self, here: tuple[float, float, float]
    ) -> set[tuple[int, int, int]]:
        """The cells the planner may search: supported free space, minus the aircraft.

        Inflated by exactly the margin ``planner.plan`` demands of a goal or start
        cell, imported from that module rather than restated here (R2: a declared
        value has one home). This method exists to answer "is there a point the
        planner could admit", so asking it with a margin of its own would be
        answering a different question — which it did: a quarter voxel against the
        planner's half cleared roughly a fifth of the map that admission then
        refused, and ``_frontier_goal_point`` returned the first such false
        positive, so the frontier was marked blocked and never offered again.

        Since R23 that constant is the certificate's own quarter voxel, used for
        both routing and goal selection, so the number here is a quarter voxel
        again — but for a different reason than when this method had a quarter of
        its own. Then it agreed by coincidence and disagreed with the planner;
        now it agrees because it is the same constant, and the certificate's
        clearance is the value both questions inherit.
        """
        free = GE.inflated_free_cells(
            self.store,
            ENVELOPE,
            now_ns=self._now_ns(),
            self_occupied_origin_odom_m=here,
            extra_margin_m=self.store.config.voxel_m * PL.CERTIFICATE_MARGIN_VOXELS,
        )
        # Membership is not the question admission asks. `planner.plan` runs A*
        # from the aircraft's own cell and refuses with `no_known_supported_route`
        # when no path connects it to the goal region — so a region holding free
        # cells that no route reaches is a goal the planner will refuse.
        # Measured on J48-fly-1: this method's caller accepted a vantage on
        # membership alone, admission refused it, and the frontier was then marked
        # blocked and never offered again.
        #
        # So the set returned here is the component reachable from where the
        # aircraft stands, by the planner's own adjacency and its own margin. One
        # home for both questions, which is what R23 began and this finishes.
        if not free:
            return set()
        start_cell = self.store.config.cell_index(here)
        return set(PL.reachable_from(frozenset(free), start_cell))

    def navigable_frontiers(self) -> tuple[str, ...]:
        """The frontiers this mission may fly to, the most distant vantage first.

        A frontier whose vantage the aircraft is already standing in is not an
        excursion and is left out (see ``_frontier_goal_point``).

        When the reachable set is provably empty -- the aircraft's own cell is
        not traversable, so it could not take one step -- the answer is ``()``
        by construction and is returned without walking a single region.

        The order is the far vantage first. The known map is what the cameras
        have already seen, so the least-observed space lies beyond its
        boundary: the far frontier is the view that adds evidence, while the
        near one is close to a view the aircraft already has. Voxel order
        would offer the near ones first, and live-17 selected them for a whole
        flight. Ties break on the ref so the choice is deterministic.
        """
        here = self._position_odom()
        if here is None:
            return ()
        searchable = self._searchable_cells(here)
        searchable_bounds = _index_bounds(searchable)
        band = self._flyable_band_z()
        self._gate_counts = {}
        ranked: list[tuple[float, str]] = []
        if not searchable:
            # The reachable set is provably empty: reachable_from refused the
            # aircraft's own cell (J51-move-2 flew off the map and every
            # traversable cell was in-grid), so no region can hold a cell the
            # planner may search and no walk can change the answer. The empty
            # listing is returned in O(1) -- no region is walked at all.
            # Without this early answer the walk still ran, tested every
            # region's every vantage to exhaustion, and stalled the mission
            # loop past its 90 s beat bound.
            self._count_gate("no_searchable_cells")
        else:
            for ref, region in self.frontier_regions().items():
                point = self._frontier_goal_point(
                    region,
                    here=here,
                    searchable=searchable,
                    searchable_bounds=searchable_bounds,
                    band=band,
                )
                if point is None:
                    continue
                distance = float(np.linalg.norm(np.asarray(point) - np.asarray(here)))
                ranked.append((-distance, ref))
        ranked.sort()
        if not ranked:
            self.result.log.append(
                "no navigable frontier: gate counts {gates}; map {summary}".format(
                    gates=dict(self._gate_counts), summary=self.map_summary()
                )
            )
        return tuple(ref for _, ref in ranked)

    def _frontier_vantage(
        self, region: GE.BoxRegion, ref: str = "?"
    ) -> tuple[float, float, float] | None:
        """The vantage for one frontier, or ``None`` when the walk admits none.

        The centre used to be the fallback "so that resolution still holds a
        target for a frontier ``navigable_frontiers`` declined: admission then
        refuses it with its own named reason". J54-zguard-1 measured where that
        fallback and an ungated walk land: the admitted goal regions carried z
        bands at and below the floor datum, and the aircraft dove into one. A
        frontier whose vantage cannot be expressed inside the flyable band
        (:meth:`_flyable_band_z`) is now refused HERE — ``None``, with the
        walk's own named gate counts for that frontier — rather than handed to
        admission as a goal the aircraft must dive for. ``resolve_targets``
        skips a ``None`` and the step reports blocked, the same honest shape as
        any other unresolved ref.
        """
        self._gate_counts = {}
        band = self._flyable_band_z()
        point = self._frontier_goal_point(region, band=band)
        if point is not None:
            return point
        here = self._position_odom()
        searchable = self._searchable_cells(here) if here is not None else set()
        free = self.store.free_cells(now_ns=self._now_ns())
        centre = tuple(float(value) for value in region.center())
        self.result.log.append(
            "frontier {ref} has no admissible vantage: walk gates {gates}; "
            "flyable band {band}; "
            "searched {searchable} searchable cell(s) of {free} free; "
            "region centre ({cx:.2f}, {cy:.2f}, {cz:.2f}) is refused, not "
            "offered".format(
                ref=ref,
                gates=dict(self._gate_counts),
                band="unstated" if band is None else "z[{:.2f}, {:.2f}]".format(*band),
                searchable=len(searchable),
                free=len(free),
                cx=centre[0],
                cy=centre[1],
                cz=centre[2],
            )
        )
        return None

    def resolve_targets(self, proposal: R.SpatialGoal) -> tuple[R.GroundedTarget, ...]:
        """Resolve a proposal's target refs into grounded targets this runtime holds.

        A frontier or the start place is a map-evidenced target built at
        admission time against the current map revision; a candidate is the
        grounded target its observation produced. Refusing to resolve is the
        honest answer when nothing is held.
        """
        resolved: list[R.GroundedTarget] = []
        citation = (
            self._candidate_observation_ids[-1] if self._candidate_observation_ids
            else self._last_observation_id
        )
        for ref in proposal.target_refs:
            if ref in self._grounded:
                resolved.append(self._grounded[ref])
                continue
            # A map-derived target is evidence-backed or it is not resolved: a
            # frontier or the start place is only as good as the observation
            # that produced the map it is built from. With no recorded
            # observation there is nothing to cite, so the target is refused
            # and the step reports blocked rather than flying to a fabricated
            # citation.
            if citation is None:
                continue
            point: tuple[float, float, float] | None = None
            if ref == "start":
                # The start position in the mission frame is the alignment's own
                # origin, and the hover band sits above it (NED z points down).
                # An unsealed alignment has no frame yet, so nothing is resolved
                # from it (the seam refuses the goal rather than inventing one).
                if not self.alignment.sealed:
                    continue
                origin = self.alignment.aligned_position_ned((0.0, 0.0, 0.0))
                point = (
                    origin[0] + RETURN_STANDOFF_COMPENSATION_M,
                    origin[1],
                    origin[2] - self.settings.hover_altitude_m,
                )
            else:
                region = self.frontier_regions().get(ref)
                if region is not None:
                    # Not the region's own centre: a frontier cell sits beside
                    # unknown space, and a goal AT it is refused as unsupported.
                    point = self._frontier_vantage(region, ref)
                    if point is not None:
                        here = self._position_odom()
                        if here is not None:
                            excursion = float(
                                np.linalg.norm(np.asarray(point) - np.asarray(here))
                            )
                            self.result.log.append(
                                f"frontier {ref} resolved to a vantage "
                                f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}), "
                                f"{excursion:.2f} m from the aircraft"
                            )
            if point is None:
                continue
            resolved.append(
                R.GroundedTarget(
                    target_id=ref,
                    track_id=None,
                    place_id="start" if ref == "start" else None,
                    selection_ids=(f"sel-{ref}",),
                    observation_ids=(citation,),
                    geometry=tuple(float(value) for value in point),
                    frame=R.Frame.ODOM,
                    anchor_id=self.store.submap_id,
                    anchor_revision=self.store.revision,
                    uncertainty=None,
                    identity_alternatives=None,
                    last_observed_stamp=None,
                    valid=True,
                )
            )
        return tuple(resolved)


def _end_state_record(sample: Any) -> dict[str, Any]:
    """The state a run ended in, from the last telemetry sample it saw.

    Carries the attitude as well as the position. Without it a run that came to
    rest inverted reads exactly like one that landed upright — same mode, same
    armed flag, a plausible position — which is how a vertical-error measurement
    was once taken on a crashed, inverted aircraft and read as an estimator
    property.
    """
    return {
        "armed": sample.armed,
        "mode": sample.mode_name,
        "local_position_ned": list(sample.local_position_ned)
        if sample.local_position_ned is not None
        else None,
        "attitude_rpy": list(sample.attitude_rpy)
        if sample.attitude_rpy is not None
        else None,
    }


def _quality_of(pair, settings):
    return measure_frame_quality(
        pair.left_bytes,
        pair.width,
        pair.height,
        identical_channel_fraction_limit=settings.stereo.identical_channel_fraction_limit,
    )


class _LiveAdmission:
    """The admission seam: the executor's staged assessment, wired to the live map."""

    def __init__(self, runtime: MissionRuntime) -> None:
        self._runtime = runtime

    def admit(self, proposal: R.SpatialGoal, context: AdmissionContext) -> R.GoalStatus:
        runtime = self._runtime
        targets = runtime.resolve_targets(proposal)
        state = runtime._navigation_state()
        if not targets or state is None:
            for ref in proposal.target_refs:
                runtime._blocked_refs[ref] = "no_grounded_target"
            return R.GoalStatus(
                proposal_id=proposal.proposal_id,
                request_id=proposal.request_id or proposal.proposal_id,
                disposition=R.GoalDisposition.REJECTED,
                reason=(
                    "no_grounded_target" if not targets else "no_navigation_state"
                ) + ": the proposal cannot be admitted from the evidence this runtime holds",
                admission_ref=None,
                current_disposition=R.ExecutionDisposition.BLOCKED,
            )
        result = EX.admit(
            proposal,
            targets,
            runtime.store,
            state,
            ENVELOPE,
            PLAN_CONFIG,
            mission_revision=runtime.contract.revision,
            snapshot_id=runtime.store.snapshot_id,
            now_ns=runtime._now_ns(),
            # The one declared pose-validity parameter, taken from the module
            # that owns it rather than restated here (R2: a declared value has
            # one home). It bounds how old a pose may be before a selection
            # transformed with it is refused.
            pose_validity_s=G.POSE_VALIDITY_S,
        )
        if result.accepted is None:
            for ref in proposal.target_refs:
                runtime._blocked_refs[ref] = result.status.reason or "admission_refused"
            runtime._refusals_log.append(
                f"admission refused {proposal.intent} {list(proposal.target_refs)}: "
                f"{result.status.reason}"
            )
            return result.status
        if result.certificate is not None:
            region = _terminal_region_for(targets)
            start = runtime._position_odom()
            runtime.result.log.append(
                "admitted {intent} {refs}: goal region x[{lx:.2f},{hx:.2f}] "
                "y[{ly:.2f},{hy:.2f}] z[{lz:.2f},{hz:.2f}]; aircraft "
                "({px:.2f},{py:.2f},{pz:.2f}) inside={inside}; certificate "
                "t {t0:.3f}..{t1:.3f} ({dur:.3f} s)".format(
                    intent=proposal.intent,
                    refs=list(proposal.target_refs),
                    lx=region.low[0], hx=region.high[0],
                    ly=region.low[1], hy=region.high[1],
                    lz=region.low[2], hz=region.high[2],
                    px=start[0], py=start[1], pz=start[2],
                    inside=region.contains(start),
                    t0=result.certificate.t_start_s,
                    t1=result.certificate.t_end_s,
                    dur=result.certificate.t_end_s - result.certificate.t_start_s,
                )
            )
        runtime._active_goal = _ActiveGoal(
            goal_id=result.accepted.goal_id,
            proposal=proposal,
            accepted=result.accepted,
            certificate=result.certificate,
            terminal_region=_terminal_region_for(targets),
            execution=R.ExecutionDisposition.RUNNING,
        )
        # A new admitted traversal flies the aircraft somewhere new: that is a
        # new vantage, so the vantage policy may spend its sweep again from the
        # view this goal reaches (GROUNDING-INVALID-DEPTH.md). Refusals never
        # re-arm it — only a change of where the aircraft stands does.
        runtime._note_new_vantage()
        return result.status

    def cancel(self, goal_id: str, expected_revision: int, idempotency_key: str) -> R.GoalStatus:
        runtime = self._runtime
        if runtime._active_goal is not None and runtime._active_goal.goal_id == goal_id:
            runtime._active_goal.execution = R.ExecutionDisposition.CANCELLED
        return R.GoalStatus(
            proposal_id=idempotency_key,
            request_id=f"cancel-{idempotency_key}",
            disposition=R.GoalDisposition.CANCELLED,
            reason="cancelled by the executive",
            admission_ref=goal_id,
            current_disposition=R.ExecutionDisposition.CANCELLED,
        )

    def status(self, goal_id: str) -> R.ExecutionStatus:
        runtime = self._runtime
        active = runtime._active_goal
        if active is None or active.goal_id != goal_id:
            return R.ExecutionStatus(
                goal_ref=goal_id,
                certificate_ref=None,
                command_ref=None,
                disposition=R.ExecutionDisposition.NOT_STARTED,
                evidence=(f"goal:{goal_id}",),
                reasons=(),
                horizon_s=None,
                capabilities=(),
            )
        return R.ExecutionStatus(
            goal_ref=goal_id,
            certificate_ref=active.certificate.certificate_id if active.certificate else None,
            command_ref=(
                active.setpoints[-1].sample_ref if active.setpoints else None
            ),
            disposition=active.execution,
            evidence=(f"proposal:{active.proposal.proposal_id}",),
            reasons=(),
            horizon_s=active.certificate.horizon_s if active.certificate else None,
            capabilities=("brake", "hold"),
        )

    def invalidate_queued(self, goal_id: str) -> int:
        return 0


def _region_gap_m(region: GE.BoxRegion, point: tuple[float, float, float]) -> float:
    """The straight-line distance from a point to a box; 0.0 if it is inside.

    This is how far the aircraft would have to travel to enter the region, which
    is the only quantity an "excursion" can honestly be measured in: a region
    whose near face is two centimetres away is a goal that is reached without
    going anywhere.
    """
    squared = 0.0
    for axis in range(3):
        low, high = region.low[axis], region.high[axis]
        if point[axis] < low:
            squared += (low - point[axis]) ** 2
        elif point[axis] > high:
            squared += (point[axis] - high) ** 2
    return float(np.sqrt(squared))


def _index_bounds(
    cells: Iterable[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """The inclusive index box of ``cells``, or None when there are none.

    One pass, so a caller that tests many regions against one reachable set
    pays for the box once.
    """
    low: list[int] | None = None
    high: list[int] | None = None
    for cell in cells:
        if low is None:
            low = [cell[0], cell[1], cell[2]]
            high = [cell[0], cell[1], cell[2]]
            continue
        for axis in range(3):
            value = cell[axis]
            if value < low[axis]:
                low[axis] = value
            elif value > high[axis]:
                high[axis] = value
    if low is None or high is None:
        return None
    return (low[0], low[1], low[2]), (high[0], high[1], high[2])


def _region_holds_searchable(
    region: GE.BoxRegion,
    searchable: set[tuple[int, int, int]],
    bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None,
    config: world_module.MapConfig,
) -> bool:
    """Whether any cell of ``region`` is in ``searchable``, without materializing.

    This answers exactly the question ``any(cell in searchable for cell in
    region.cells(config))`` asks; only its cost changed. The region's index box
    against the reachable set's own index box settles the disjoint case without
    touching a cell: the index rule is floor, hence monotone, so a centre inside
    the region always indexes inside the region's index box -- the pre-test can
    only skip regions the exact test would refuse. Where the boxes overlap, the
    cheaper of the two walks runs: the region's cells one at a time (the walk
    short-circuits on the first hit instead of building the whole tuple), or the
    reachable set against ``BoxRegion.holds_cell``.
    """
    if bounds is None or not searchable:
        return False
    index_low = config.cell_index(region.low)
    index_high = config.cell_index(region.high)
    if any(
        index_low[axis] > bounds[1][axis] or bounds[0][axis] > index_high[axis]
        for axis in range(3)
    ):
        return False
    estimate = 1
    for axis in range(3):
        estimate *= index_high[axis] - index_low[axis] + 1
    if estimate <= len(searchable):
        return any(cell in searchable for cell in region.iter_cells(config))
    return any(region.holds_cell(cell, config) for cell in searchable)


def _terminal_region_for(targets: tuple[R.GroundedTarget, ...]) -> GE.BoxRegion | None:
    """The goal's terminal region, computed exactly as ``executor.admit`` does.

    The executor builds the approach region internally from the target's
    point geometry; this mirror makes the same call with the same inputs, so
    the completion monitor checks arrival against the region the planner was
    given, by construction rather than by coincidence.
    """
    point = G.parse_point(targets[0].geometry)
    if point is None:
        return None
    return GE.approach_region(point, ENVELOPE, direction=(1.0, 0.0, 0.0))


class _LiveRunnerWorld:
    """The harness-side world the shared RecipeRunner drives."""

    def __init__(self, runtime: MissionRuntime) -> None:
        self._runtime = runtime
        self._visited: set[str] = set()
        self._inspected: set[str] = set()
        self._step_observations: list[str] = []
        self._last_renewal_s = 0.0

    # -- the runner's views ---------------------------------------------------

    def discovered_candidates(self) -> tuple[str, ...]:
        return tuple(
            target_id
            for target_id in self._runtime._candidate_targets
            if target_id not in self._inspected
        )

    def known_frontiers(self) -> tuple[str, ...]:
        # Not every frontier is a place to fly to: one whose vantage the
        # aircraft already occupies yields a hover, and a hover explores
        # nothing. Of them, farthest vantage first.
        return self._runtime.navigable_frontiers()

    def start_place(self) -> str:
        return "start"

    def is_blocked(self, target_ref: str) -> bool:
        return target_ref in self._runtime._blocked_refs

    def budget_remaining_s(self) -> float:
        newest = self._runtime._stats.sim_clock.newest_s
        if newest is None:
            return MISSION_BUDGET_SIM_S
        return max(0.0, MISSION_BUDGET_SIM_S - newest)

    # -- execution -------------------------------------------------------------

    def step(self, action: str, target_ref: str | None) -> tuple[R.Observation, ...]:
        runtime = self._runtime
        self._step_observations = []
        # A goal still installed here belongs to a finished step: this step's own
        # goal, if it has one, is admitted immediately below by the runner and
        # not before. Leaving a foreign goal installed is what silenced the
        # return's recovery in J52-discrim-1 — its leftover explore goal kept
        # `_active_goal` non-None through every return step, so the observation
        # machinery below never fired, and the renewal machinery even flew the
        # explore goal during the return leases (RETURN-UNSUPPORTED.md).
        self._demote_foreign_active_goal(action, target_ref)
        # Section 12.2: a traversal that could not be admitted gets a conditional
        # observation objective instead. The runner admits before it calls this, so
        # a target present in the blocked map at this point means this step's
        # traversal was refused and there is no trajectory to fly. The
        # specification requires both facts to be returned clearly and the loop to
        # stay bounded, so the observation happens ONCE per target and the target
        # is then re-opened for a single re-proposal against the new evidence.
        # The `_active_goal is None` condition is what keeps a legitimately
        # executing goal untouched: an admitted goal for this very target is
        # never demoted, and while it flies there is no refused traversal to
        # observe for.
        if target_ref is not None and runtime._active_goal is None:
            refused_reason = runtime._blocked_refs.get(target_ref)
            if refused_reason is not None and target_ref not in runtime._observed_targets:
                runtime._observed_targets.add(target_ref)
                before_observations = runtime._observation_counter
                observed, detail = runtime.observe_in_place(
                    target_ref=target_ref, refused_reason=refused_reason
                )
                made = runtime._observation_counter - before_observations
                if made > 0:
                    self._step_observations.extend(runtime._observation_ids[-made:])
                if observed:
                    runtime._blocked_refs.pop(target_ref, None)
        lease = _SimWindow(
            runtime._stats.sim_clock,
            ACTION_LEASE_SIM_S.get(action, 45.0),
            label=f"the {action} lease",
            wall_ceiling_s=_sim_window_wall_ceiling_s(
                ACTION_LEASE_SIM_S.get(action, 45.0),
                runtime.settings.realtime_ratio_envelope[0],
            ),
        )
        next_publish = 0.0
        next_perception = 0.0
        settle_since: float | None = None
        while not lease.expired():
            runtime._beat_watchdog()
            if runtime._visual_fault_reason is not None:
                # A step also ends when the estimate stops being a pose, so a
                # fault during a flown step stops the step rather than being
                # noticed only after the phase returns.
                break
            now = time.monotonic()
            if now >= next_perception:
                next_perception = now + MIN_PERCEPTION_INTERVAL_S
                before_candidates = len(runtime._candidate_observation_ids)
                before_counter = runtime._observation_counter
                runtime.perceive_if_due()
                if len(runtime._candidate_observation_ids) > before_candidates:
                    self._step_observations.extend(
                        runtime._candidate_observation_ids[before_candidates:]
                    )
                # Every observation the step made is citable, not only the ones
                # that grounded a candidate. Counting from the monotone
                # observation counter rather than from the citation list keeps
                # this correct when the list has been trimmed at its bound. The
                # alternative was a return that settled while the aircraft was
                # observing and still claimed "not_returned", because a step
                # with no grounded candidate cited nothing at all.
                made = runtime._observation_counter - before_counter
                if made > 0:
                    self._step_observations.extend(runtime._observation_ids[-made:])
                self._renew_certificate()
                # The vantage policy's answer to a persistently unmeasurable
                # selection, consumed in the same state the section-12.2
                # objective answers a refused traversal: no goal of this step's
                # is flying (otherwise the sweep would replace a legitimately
                # executing goal). Bounded by the sweep's once-per-view flag.
                if runtime._active_goal is None:
                    runtime._consume_vantage_sweep()
            if now >= next_publish:
                next_publish = now + SETPOINT_PERIOD_S
                stopped = runtime.publish_active()
                if stopped is not None:
                    runtime.result.log.append(
                        f"{action} {target_ref}: publication stopped: {stopped}"
                    )
                    break
            if self._settled():
                if settle_since is None:
                    settle_since = now
                elif now - settle_since >= SETTLE_HOLD_S:
                    break
            else:
                settle_since = None
            time.sleep(0.005)
        self._finish(action, target_ref)
        return ()

    def completion(self, action: str, target_ref: str | None) -> tuple[str, tuple[str, ...]]:
        runtime = self._runtime
        active = runtime._active_goal
        observations = tuple(dict.fromkeys(self._step_observations))[:4]
        if active is None or active.certificate is None or active.terminal_region is None:
            return "blocked", observations
        position = runtime._position_odom()
        if position is None:
            return "blocked", observations
        nav_state = runtime._navigation_state()
        # A completion is a claim, and a claim cites. What the step observed is
        # the first choice of citation; when a step reaches its declared
        # condition without grounding a candidate — an explore step arriving at
        # a frontier, or the return to a start the aircraft already occupies —
        # the citation is the navigation state whose position and speed the
        # arrival was measured from. Handing the runner an empty tuple is what
        # raised "a completion is a claim with evidence" on live-12; the
        # mission's own claim evidence below stays observation-only, so the
        # found/inspected/returned claims never cite a state string.
        cited = observations or (
            f"state:{nav_state.state_sequence if nav_state is not None else 0}",
        )
        status = EX.completion_status(
            active.accepted,
            active.certificate,
            active.setpoints,
            position_odom_m=position,
            speed_mps=runtime._speed(),
            terminal_region=active.terminal_region,
            settle_position_tolerance_m=ENVELOPE.inflation_m,
            settle_speed_tolerance_mps=SETTLE_SPEED_MPS,
            evidence=cited,
        )
        if status.disposition is not R.ExecutionDisposition.COMPLETED:
            return "blocked", observations
        if action == "return":
            runtime._return_settled = True
            runtime._return_evidence.extend(observations)
        if action == "inspect":
            runtime._inspect_evidence.extend(observations)
        return "achieved", cited

    # -- internals ---------------------------------------------------------------

    def _settled(self) -> bool:
        runtime = self._runtime
        active = runtime._active_goal
        if active is None or active.terminal_region is None:
            return False
        position = runtime._position_odom()
        if position is None:
            return False
        return (
            active.terminal_region.contains(position)
            and runtime._speed() <= SETTLE_SPEED_MPS
        )

    def _finish(self, action: str, target_ref: str | None) -> None:
        if action == "explore" and target_ref:
            self._visited.add(target_ref)
        if action == "inspect" and target_ref:
            self._inspected.add(target_ref)

    def _demote_foreign_active_goal(self, action: str, target_ref: str | None) -> None:
        """Clear an installed goal that does not belong to the step about to run.

        A step's goal is admitted immediately before its lease loop, so a goal
        still installed when ``step`` begins was installed by an EARLIER step —
        one that has already finished, blocked or otherwise. J52-discrim-1
        measured what keeping it costs: the explore goal
        (cert-explore-frontier:8:7:2) stayed installed through every return
        step, kept `_active_goal` non-None, silenced the section-12.2
        observation that would have re-evidenced the start, and was even
        RENEWED and published by the return steps' own leases — the mission
        flew the wrong goal while believing it was flying the return.

        A previous attempt of THIS step is not foreign: its goal aims at this
        step's own target, may still be that target's best trajectory, and
        cancelling it would discard a legitimately executing goal. Everything
        else is demoted: the step has no trajectory of its own until one is
        admitted, so it publishes nothing, reports blocked, and the observation
        machinery in ``step`` runs against the refused target.
        """
        runtime = self._runtime
        active = runtime._active_goal
        if active is None:
            return
        proposal = active.proposal
        if (
            proposal is not None
            and proposal.intent == action
            and tuple(proposal.target_refs) == (target_ref,)
        ):
            return
        runtime._active_goal = None
        owned = (
            f"{proposal.intent} {list(proposal.target_refs)}"
            if proposal is not None
            else "no proposal (an observation objective)"
        )
        runtime.result.log.append(
            f"step {action} {target_ref}: demoted installed goal {active.goal_id} "
            f"({owned}), which belongs to a finished step; this step has no "
            "trajectory of its own until one is admitted"
        )

    def _renew_certificate(self) -> None:
        """Renew the active goal's certificate when the map has advanced.

        Planning renews certificates while the objective stays inside its
        lease (specification 15.1): the goal is unchanged, the trajectory is
        re-derived from the current state against the new map revision. A
        refusal switches publication to a hold at the current position — the
        supported station hold of specification 14.2 — rather than continuing
        to publish against a dependency the validator would refuse.
        """
        runtime = self._runtime
        active = runtime._active_goal
        if active is None or active.certificate is None:
            return
        if active.certificate.map_revision == runtime.store.revision:
            return
        now = time.monotonic()
        if now - getattr(self, "_last_renewal_s", 0.0) < RENEWAL_PERIOD_S:
            return
        self._last_renewal_s = now
        state = runtime._navigation_state()
        if state is None:
            return
        targets = runtime.resolve_targets(active.proposal)
        if not targets:
            return
        result = EX.admit(
            active.proposal,
            targets,
            runtime.store,
            state,
            ENVELOPE,
            PLAN_CONFIG,
            mission_revision=runtime.contract.revision,
            snapshot_id=runtime.store.snapshot_id,
            now_ns=runtime._now_ns(),
            pose_validity_s=G.POSE_VALIDITY_S,
        )
        if result.accepted is not None and result.certificate is not None:
            active.certificate = result.certificate
            active.accepted = result.accepted
            active.terminal_region = _terminal_region_for(targets)
            return
        active.hold_position_odom = runtime._renewal_hold_position(
            runtime._position_odom()
        )
        active.certificate = None
        runtime.result.log.append(
            f"certificate renewal refused for {active.proposal.intent}: "
            f"{result.status.reason}; holding at the declared hover altitude"
        )


def _pilot_section(root: Path) -> dict[str, Any]:
    """The declared pilot parameters, from their one config file."""
    import yaml

    return yaml.safe_load((root / "configs" / "runtime-model.yaml").read_text())["pilot"]


def write_final_report(runtime: MissionRuntime) -> R.FinalReport:
    """Assemble and record the mission's final report from its own evidence."""
    result = runtime.result
    report = mission_module.assemble_mission_claims(
        found=result.found,
        inspected=result.inspected,
        returned=result.returned,
        target_id=runtime.target_id,
        termination_reason=result.termination_reason,
        mission_revision=runtime.contract.revision,
        now=runtime._clock(),
    )
    runtime.recorder.write_report(report, runtime._clock())
    return report
