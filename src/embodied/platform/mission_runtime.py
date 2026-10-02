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
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

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
    measure_frame_quality,
    ppm_bytes,
)
from embodied.pilot import mission as mission_module
from embodied.pilot.broker import AdmissionContext, PilotBroker
from embodied.pilot.decisions import PilotParameters
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
# The error allowance is the live estimator's: the declared E1 bound is p95
# 0.10 m and the H4 envelope reports in-flight sigma up to 0.05 m, so 0.15 m
# admits a healthy 3-sigma pose under the validator's own rule while refusing
# the H4 outage bound. A pose outside it refuses publication — the coupling is
# the intended protective behaviour, not an accident.
ENVELOPE = GE.Envelope(body_radius_m=0.3, error_allowance_m=0.15)
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


@dataclass
class _ActiveGoal:
    """The one goal whose certified prefix may be published."""

    goal_id: str
    proposal: R.SpatialGoal
    accepted: EX.AcceptedGoal
    certificate: PL.TrajectoryCertificate | None
    terminal_region: GE.BoxRegion | None
    execution: R.ExecutionDisposition = R.ExecutionDisposition.NOT_STARTED
    hold_position_odom: tuple[float, float, float] | None = None
    setpoints: tuple[R.MotionSetpoint, ...] = ()


class MissionRuntime:
    """Flies one first-indoor mission and records its agent stream."""

    def __init__(
        self,
        *,
        settings: PlatformSettings,
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
        self.contract = mission_module.mission_contract(
            mission_id=episode_id,
            instruction=instruction,
            budget=(
                ("mission_sim_s", MISSION_BUDGET_SIM_S),
                ("explore_steps", float(mission_module.EXPLORE_STEPS)),
            ),
        )
        self._perception_queue: queue.Queue = queue.Queue(maxsize=4)
        # Perception frames dropped because the perception queue was full. A
        # count that stays at zero says the map saw every frame; anything else
        # says the map was built from a subset, which a reader of the receipt
        # needs to know before trusting a frontier.
        self._perception_frames_dropped = 0
        self._state_ring: list[tuple[int, loc.EstimatorState]] = []
        self._latest_state: loc.EstimatorState | None = None
        self._latest_aligned: dict[str, object] | None = None
        self._capture_clock_ns: int | None = None
        self._controller_status: dict[str, Any] = {}
        self._observation_counter = 0
        self._last_observation_id: str | None = None
        self._payload_count = 0
        self._publication_count = 0
        self._publish_refusal_count = 0
        self._grounded: dict[str, R.GroundedTarget] = {}
        self._candidate_targets: list[str] = []
        self._candidate_observation_ids: list[str] = []
        self._blocked_refs: dict[str, str] = {}
        self._refusals_log: list[str] = []
        self._active_goal: _ActiveGoal | None = None
        self._return_settled = False
        self._return_evidence: list[str] = []
        self._inspect_evidence: list[str] = []
        self._stats = _FeedStats()
        self._first_record_kind: str | None = None
        self._platform: WebotsArduPilot | None = None
        self._session: PymavlinkSession | None = None
        self._publisher: loc.ExternalNavPublisher | None = None
        self.result = MissionResult(flew=False, termination_reason="not_started")

    # -- records ----------------------------------------------------------

    def _recorder_sink(self, kind: str, payload: dict, stamp, sim_time_s=None) -> None:
        self.recorder.record(kind, payload, stamp, sim_time_s)

    def _now_ns(self) -> int:
        return self._capture_clock_ns if self._capture_clock_ns is not None else time.monotonic_ns()

    def _state_stamp(self) -> R.ClockStamp:
        controller_host = str(self._controller_status.get("host_id") or self.settings.host_id)
        controller_clock = str(self._controller_status.get("clock_id") or self.settings.clock_id)
        return R.ClockStamp(
            host_id=controller_host, clock_id=controller_clock, monotonic_ns=self._now_ns()
        )

    def _navigation_state(self) -> R.NavigationState | None:
        state = self._latest_state
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
        state = self._latest_state
        if state is None or not self.alignment.sealed:
            return None
        return self.alignment.aligned_position_ned(state.position_m)

    def _speed(self) -> float:
        state = self._latest_state
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
        state = best[1]
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

        def feed_cycle() -> None:
            while True:
                record = platform.sensor_record(FEED_STREAM_POLL_S)
                if record is None:
                    break
                feed_record(record)
            try:
                while True:
                    feed_record(pending_pairs.get_nowait())
            except queue.Empty:
                pass
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
                platform.telemetry()
            except Exception as error:
                self._machine.stop(time.monotonic_ns(), f"the telemetry stream failed: {error}")

        bring_up_link: BringUpLink | None = None
        feed_thread: threading.Thread | None = None
        try:
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
            control = platform.arm_and_guided(self.settings.step_timeout_s.flight, drain=drain)
            if control.refused:
                self.result.blockers.append(
                    f"the autopilot refused Guided flight: mode_reached={control.mode_reached}, "
                    f"armed={control.armed}, refusals={list(control.refusals)}"
                )
                self.result.termination_reason = "arming_refused"
                return self.result
            self.result.flew = True
            self._machine.open_window(time.monotonic_ns())
            self._fly_the_mission(drain)
        finally:
            self._shutdown(
                platform, publisher, estimator_process, feed_stop, feed_thread, bring_up_link
            )
            self.result.publications = self._publication_count
            self.result.publish_refusals = self._publish_refusal_count
            self.result.stream = {
                "pairs": self._stats.pairs,
                "pair_records_filed": self._stats.pair_records_filed,
                "imu_samples": self._stats.imu_samples,
                "truth_samples": len(self._stats.truth_samples),
                "sim_clock_frames": self._stats.sim_clock.frames,
                "sim_clock_newest_s": self._stats.sim_clock.newest_s,
                "first_record_kind": self._first_record_kind,
                "feed_failures": list(feed_failures),
                "perception_frames_dropped": self._perception_frames_dropped,
                "publisher_published": getattr(publisher, "published", None),
                "publisher_failures": list(getattr(publisher, "publish_failures", [])),
            }
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
        # its own empty map. Pump perception on its own cadence for a bounded
        # window before the phases begin, so the first goal resolves against a
        # real snapshot.
        cold_start = _SimWindow(
            self._stats.sim_clock,
            COLD_START_PERCEPTION_SIM_S,
            label="the cold-start perception window",
            wall_ceiling_s=_sim_window_wall_ceiling_s(
                COLD_START_PERCEPTION_SIM_S, self.settings.realtime_ratio_envelope[0]
            ),
        )
        while not cold_start.expired():
            if self._candidate_targets or self.frontier_regions():
                break
            drain()
            self.perceive_if_due()
            time.sleep(0.02)
        self.result.log.append(
            f"cold start: {len(self.frontier_regions())} frontier region(s), "
            f"{len(self._candidate_targets)} grounded candidate(s), "
            f"{self._observation_counter} observation(s) seen"
        )
        # The inspect phase's own step guard (candidate_present) decides whether
        # there is anything to inspect, evaluated when that phase is reached. The
        # skip that used to live here read a snapshot taken BEFORE exploration
        # ran, so it could only ever skip the phase that exploration exists to
        # feed.
        for phase, recipe in zip(("explore", "inspect", "return"), mission_module.build_b0_recipes()):
            if mission_window.expired():
                termination = "mission_budget_exhausted"
                break
            outcome = runner.run(recipe)
            self.result.phases.append(PhaseOutcome(outcome.status, outcome.reason, outcome.steps))
            self.result.log.append(f"phase {phase}: {outcome.status} — {outcome.reason}")
            if outcome.status == "budget_exhausted":
                termination = "mission_budget_exhausted"
                break
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
        self.result.returned = mission_module.ClaimEvidence(
            bool(self._return_settled and self._return_evidence),
            tuple(dict.fromkeys(self._return_evidence))[:4],
        )
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
            self.result.end_state = {
                "armed": sample.armed,
                "mode": sample.mode_name,
                "local_position_ned": list(sample.local_position_ned)
                if sample.local_position_ned is not None
                else None,
            }
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

    def perceive_if_due(self) -> None:
        """Depth, candidates and map integration on the newest queued pair.

        One observation event per cycle, with the pair's payloads stored only
        when the frame grounded a candidate — the evidence a claim can cite.
        """
        record = None
        try:
            while True:
                record = self._perception_queue.get_nowait()
        except queue.Empty:
            pass
        if record is None or record.pair is None:
            return
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
            self._refusals_log.append(f"depth_failed: {error}")
            return
        candidates = () if isinstance(outcome, DetectorUnavailable) else outcome
        observation = self._record_observation(record, store_payload=bool(candidates))
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
                self._refusals_log.append(f"{target.reason}: {target.detail}")
                continue
            grounded.append(target)
        return grounded

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

    # -- publication: the single setpoint path --------------------------------

    def publish_active(self) -> str | None:
        """Publish one sample of the active goal's certified prefix.

        Returns None on success, or the reason publication stopped. Every
        motion target this runtime puts on the wire goes through this one
        method and the platform's one publisher.
        """
        active = self._active_goal
        if active is None:
            return "no_active_goal"
        certificate = active.certificate
        if certificate is not None:
            sample_t = min(time.monotonic(), certificate.t_end_s)
            position_ned, velocity_ned, _acceleration = certificate.sample(sample_t)
            certificate_ref = certificate.certificate_id
        elif active.hold_position_odom is not None:
            position_ned = active.hold_position_odom
            velocity_ned = (0.0, 0.0, 0.0)
            certificate_ref = active.goal_id
        else:
            return "no_published_trajectory"
        sent = self._platform.send_local_ned(
            LocalNedTarget(
                position_ned=position_ned,
                velocity_ned=velocity_ned,
                yaw_rad=MISSION_YAW_HOLD_RAD,
                deadline_s=SETPOINT_DEADLINE_S,
                certificate_ref=certificate_ref,
            )
        )
        if sent is None:
            self._publish_refusal_count += 1
            return "guided_flight_lost"
        self._publication_count += 1
        active.setpoints = (*active.setpoints[-3:], sent.setpoint)
        self._sink("setpoint", R.to_dict(sent.setpoint), sent.published_stamp, None)
        return None

    # -- frontier helpers ------------------------------------------------------

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
                    point = region.center()
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
        runtime._active_goal = _ActiveGoal(
            goal_id=result.accepted.goal_id,
            proposal=proposal,
            accepted=result.accepted,
            certificate=result.certificate,
            terminal_region=_terminal_region_for(targets),
            execution=R.ExecutionDisposition.RUNNING,
        )
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
        return tuple(self._runtime.frontier_regions())

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
            now = time.monotonic()
            if now >= next_perception:
                next_perception = now + MIN_PERCEPTION_INTERVAL_S
                before = len(runtime._candidate_observation_ids)
                runtime.perceive_if_due()
                if len(runtime._candidate_observation_ids) > before:
                    self._step_observations.extend(
                        runtime._candidate_observation_ids[before:]
                    )
                self._renew_certificate()
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
        evidence = tuple(dict.fromkeys(self._step_observations))[:4]
        if active is None or active.certificate is None or active.terminal_region is None:
            return "blocked", evidence
        position = runtime._position_odom()
        if position is None:
            return "blocked", evidence
        nav_state = runtime._navigation_state()
        status = EX.completion_status(
            active.accepted,
            active.certificate,
            active.setpoints,
            position_odom_m=position,
            speed_mps=runtime._speed(),
            terminal_region=active.terminal_region,
            settle_position_tolerance_m=ENVELOPE.inflation_m,
            settle_speed_tolerance_mps=SETTLE_SPEED_MPS,
            evidence=evidence
            or (f"state:{nav_state.state_sequence if nav_state is not None else 0}",),
        )
        if status.disposition is not R.ExecutionDisposition.COMPLETED:
            return "blocked", evidence
        if action == "return":
            runtime._return_settled = True
            runtime._return_evidence.extend(evidence)
        if action == "inspect":
            runtime._inspect_evidence.extend(evidence)
        return "achieved", evidence

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
        active.hold_position_odom = runtime._position_odom()
        active.certificate = None
        runtime.result.log.append(
            f"certificate renewal refused for {active.proposal.intent}: "
            f"{result.status.reason}; holding at the current position"
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
