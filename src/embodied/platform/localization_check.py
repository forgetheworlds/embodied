"""The ``localize-check`` command: P01-L's acceptance run and its receipt.

The command runs one way in two shapes: as a module entry point
(``python -m embodied.platform.localization_check``, the recipe that works while
the dispatch registration is serialized) and as a registered command
(``python -m embodied localize-check`` once the integrator adds the one dispatch
line; this module registers itself at import, and never edits the parser).

Modes are the honesty surface (plan section 8). ``--mode sensor-derived`` is the
only mode that can pass P01-L: GPS off, no bridge truth republish, the pinned
estimator fed only the declared stereo and inertial streams.
``--mode pose-assisted`` is a labelled diagnostic that can never pass — its
receipt carries ``gate_status: not_applicable`` and is never pooled with a
sensor-derived arm. Any other mode value is refused at this boundary.

A run is one of three things, and the receipt says which:

* a live sensor-derived run, scored against the bounds frozen in the
  configuration before any measurement (plan section 6);
* a blocked run: a prerequisite of the claimed arm is missing — among them the
  serialized integrator actions the plan names (merge, ``p01l_sensor.parm``,
  ``estimator/ov_stream``) — nothing is started, and the receipt records
  ``localization=unresolved`` with every concrete blocker. This is a complete,
  honest outcome (plan section 11): no bound is relaxed, no truth is fed;
* a pose-assisted diagnostic, labelled and not applicable to the gate.

Evaluator truth measures error; it never enters the runtime. On a branch
without the merged gate's truth channel the E1 statistics are recorded as
``not_measured`` with the reason, and the gate cannot pass.
"""
from __future__ import annotations

import argparse
import ast
from bisect import bisect_left
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import re
import socket
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from embodied.cli import (
    COMMAND_REGISTRY,
    CONFIG_SCHEMA,
    CommandOutcome,
    CommandStatus,
    ConfigError,
    GateStatus,
    _Optional,
    _validate,
    register_command,
    repository_root,
)
from embodied.contracts.records import SensorMode
from embodied.platform import localization as loc
from embodied.platform.sensors import SensorSample, capture_latency_ns, sim_time_ns
from embodied.platform.webots_ardupilot import (
    BringUpLink,
    EvidenceWriter,
    Kind,
    LocalNedTarget,
    MAV_CMD_NAV_TAKEOFF,
    MSG_ID_RC_CHANNELS,
    PlatformSettings,
    PymavlinkSession,
    ProbeFailure,
    RC_THROTTLE_CHANNEL,
    RC_THROTTLE_RELEASE_PWM,
    SubprocessRunner,
    TcpSensorGateway,
    WebotsArduPilot,
    check_prerequisites,
    read_configured_parameters,
)

STAGE_ID = "P01-L"
COMMAND_NAME = "localize-check"
RUN_PREFIX = "p01-localization"

DISPATCH_REGISTRATION_NOTE = (
    "this module self-registers on import and is also listed in embodied.cli "
    "COMMAND_MODULES, so both `python -m embodied localize-check` and the module entry "
    "point work: the self-registration is idempotent because running the module as an "
    "entry point imports it twice, once as __main__ and once under its package name"
)

# The parameter layer the claimed arm needs, and where each requirement lives.
# The EKF source selection is the merged gate's (compat_ekf.parm on main); GPS
# off and the visual-odometer layer are the p01l_sensor.parm file the plan
# materializes as a new file (plan section 4.3).
SEAM_REQUIREMENTS: tuple[tuple[str, float, str], ...] = (
    ("EK3_SRC1_POSXY", 6.0, "merged compat_ekf.parm"),
    ("EK3_SRC1_VELXY", 6.0, "merged compat_ekf.parm"),
    ("EK3_SRC1_POSZ", 6.0, "merged compat_ekf.parm"),
    ("EK3_SRC1_YAW", 6.0, "merged compat_ekf.parm"),
    ("EK3_SRC1_VELZ", 6.0, "merged compat_ekf.parm"),
    ("VISO_TYPE", 1.0, "merged compat_ekf.parm"),
    ("COMPASS_USE", 0.0, "merged compat_ekf.parm"),
    ("GPS1_TYPE", 0.0, "p01l_sensor.parm"),
    ("GPS2_TYPE", 0.0, "p01l_sensor.parm"),
    # Measured, FIXER5: the pose arrives p50 2.5-5.7 ms (sim) after its own validity
    # stamp, so the EKF's declared fusion delay is the pin's default 10 ms
    # (AP_VisualOdom.cpp:83), not the 50 ms tuned when the pose was stale.
    ("VISO_DELAY_MS", 10.0, "p01l_sensor.parm"),
    ("VISO_QUAL_MIN", 0.0, "p01l_sensor.parm"),
    ("FS_EKF_ACTION", 1.0, "p01l_sensor.parm"),
)
P01L_PARAMS_FILENAME = "p01l_sensor.parm"
VEHICLE_DEFAULT_REQUIREMENTS: tuple[tuple[str, float], ...] = (
    ("EK3_SRC2_POSXY", 0.0),
    ("EK3_SRC2_VELXY", 0.0),
    ("EK3_SRC2_YAW", 0.0),
    ("EK3_SRC3_POSXY", 0.0),
    ("EK3_SRC3_VELXY", 0.0),
    ("EK3_SRC3_YAW", 0.0),
)
# Everything the running vehicle must answer for the claimed arm (plan section 4.7):
# the applied layer's names (G1) and the source-set defaults observed, not assumed (G3).
VEHICLE_REQUIREMENTS: tuple[tuple[str, float, str], ...] = SEAM_REQUIREMENTS + tuple(
    (name, expected, "the firmware's own defaults, observed from the vehicle")
    for name, expected in VEHICLE_DEFAULT_REQUIREMENTS
)

# G2: a name the vehicle refuses is a failure, never an absence. An unknown parameter
# is answered with PARAM_ERROR (msgid 345) carrying MAV_PARAM_ERROR_DOES_NOT_EXIST
# (GCS_Param.cpp:414-421, sent at :502-510). This host's pymavlink predates the
# message -- the recorded answer arrived as UNKNOWN_345 -- so the frame is parsed
# here with a local struct and no new dependency. The wire order is pinned against
# the vehicle's own recorded frame (run-2026-09-26T07-40-00Z/run-a/mavlink.jsonl):
# param_index -1, target_system 250, target_component 190, param_id "GPS_TYPE",
# error 1.
PARAM_ERROR_MSG_ID = 345
MAV_PARAM_ERROR_DOES_NOT_EXIST = 1
_PARAM_ERROR_PAYLOAD = struct.Struct("<hBB16sB")

# G4: the vehicle's runtime answer. SYS_STATUS (msgid 1) carries the GPS-present bit
# (MAV_SYS_STATUS_SENSOR_GPS = 32, common.xml:121) only when a GPS driver is running
# (GCS.cpp:499-506); GPS_RAW_INT (msgid 24) carries fix_type; the driver's own probe
# and detect notices ride STATUSTEXT (GPS_Backend.cpp:136). The streams are requested
# through the session's existing SET_MESSAGE_INTERVAL path and every inbound message
# is recorded by platform.telemetry(), so the whole-run verdict is re-derivable from
# the run's own artifacts.
MSG_ID_SYS_STATUS = 1
MSG_ID_GPS_RAW_INT = 24
GPS_AIDING_SAMPLE_HZ = 2.0
GPS_SENSOR_PRESENT_BIT = 32

# The drain loop folds the recorded telemetry at this cadence: the readback and arming
# sample the stream through their own paths, but the rest of the run would otherwise
# be recorded only incidentally, and a gate about what the vehicle reports during the
# run needs the run's whole window sampled (plan section 4.7).
TELEMETRY_SAMPLE_PERIOD_S = 0.2

# T7 (plan sections 3.6 and 12.6): the scene-admission check. The pinned initializer
# needs at least feat_thresh = 15 trackable features per window
# (InertialInitializer.cpp:115-119) and its tracker hunts with cv::FAST at the pinned
# default threshold 20 with non-max suppression (VioManagerOptions.h:424,
# Grider_GRID.h:125, TrackKLT.cpp:494). A scene that gives that detector nothing makes
# initialization structurally impossible, so the preflight measures the scene's own
# recorded frames instead of letting a run spend its pre-arm window on an initializer
# that cannot fire. Captures are the platform's accepted-run artifacts: P00's
# compatibility gate and this stage's own runs.
SCENE_ADMISSION_CAPTURE_GLOBS = (
    "work/runs/p00-compat/accept-*/run-*/pairs",
    "work/runs/p01-localization/*-*/run-*/pairs",
)
SCENE_ADMISSION_MAX_FRAMES = 24
INITIALIZER_FEATURE_FLOOR = 15
FAST_THRESHOLD = 20
# ---------------------------------------------------------------------------
# E1-DIAG (plan sections 0.6 item 7, 0.7, 0.8): the pose-assisted diagnostic arm
# ---------------------------------------------------------------------------

# The declared bounded excitation (E-EXC), frozen as module constants before any
# motion measurement existed (plan sections 0.6 item 5, 0.7 item 4, 0.8 item 3).
# Carrier GUIDED, no lateral setpoint at all, termination LAND always:
#
#   height    <= 0.60 m, the commanded takeoff altitude -- half the declared 1.5 m
#               hover altitude and well under the 2.0 m doorway lintel;
#   airtime   <= 5.0 s from the arm readback to the LAND command;
#   post-LAND drain 10.0 s, so the descent's frames are fed too -- still motion,
#               still inside the same flight event, still terminated by LAND.
#
# Why a 0.60 m climb excites feature propagation, read from the pinned source
# (plan section 0.6 item 5) rather than hoped: VioManager::initialized() needs
# timelastupdate written once (VioManager.cpp:651), which needs
# do_feature_propagate_update past the clone gate (VioManager.cpp:348), which
# needs >= 5 frames the zero-velocity updater declined
# (UpdaterZeroVelocity.cpp:246: disparity >= zupt_max_disparity 1.0 px together
# with a chi2 or velocity violation). A climb supplies both terms: thrust makes
# the accelerometer read other than gravity (the chi2 residual of the
# zero-velocity hypothesis) and, at the rig's 554 px focal length, ~2.7 mm of
# translation is ~1 px of mean disparity -- so >= 0.5 s of climb at the declared
# 10 Hz stereo rate is >= 5 declining frames, and the 5.0 s bound carries a
# factor of ten over a criterion read from the code.
#
# The envelope is carried as constants, not a localization.excitation
# configuration key, because the shared cli.py schema rejects unknown
# localization keys (cli.py:446-448) and the compatibility probe loads this
# stage's configuration through that shared loader (webots_ardupilot.py:5734),
# so the key would break `python -m embodied compat` -- the P00 gate's own
# command (plan section 0.8 item 3, Correction C). configs/first_indoor.yaml
# carries the same declaration as a comment beside the section it belongs to.
#
# THE UNIT OF EVERY WINDOW BELOW IS THE SIMULATOR'S SECOND, NOT THE HOST'S, and
# until 2026-09-28 the code spent them on the wall clock. That is a defect, not a
# preference: the airtime bound is "from the arm readback to the LAND command",
# a quantity of the aircraft's own time; the MOT_IDLE_SEC reasoning above weighs
# it against the firmware's own 4.0 s; a climb rate times the window is a
# distance the aircraft has to fly; and the configuration declares these windows
# in simulated time for the same reason ("a window measured in simulated time is
# comparable between the realtime and fast modes"). Measured on a loaded host
# (run-2026-09-28T02-43-27-442Z, receipt work/runs/p01-localization/
# receipt-p01l-2026-09-28T02-43-27-442Z.json): the simulator ran at 0.40-0.92
# simulated seconds per wall second, this 3.5 s climb window bought 2.20
# simulated seconds, and the LAND command was applied 0.80 simulated seconds
# BEFORE the thrust path's first motor output (the airframe's own log,
# work/ardupilot/logs/00000141.BIN: first motor output at 18.923 s of autopilot
# time) -- an excitation cut off by the host's pace. Every window here is spent
# with `_SimWindow` against the simulator's clock, with the wall ceiling
# `_sim_window_wall_ceiling_s` derives from the declared realtime envelope.
EXCITATION_MODE = "GUIDED"
EXCITATION_TAKEOFF_ALTITUDE_M = 0.60
EXCITATION_MAX_AIRTIME_S = 5.0
EXCITATION_CLIMB_DRAIN_S = 3.5
EXCITATION_POST_LAND_DRAIN_S = 10.0
EXCITATION_ALTITUDE_REACHED_MARGIN_M = 0.05

# The yaw the scored window's position targets carry, as a commanded angle.
# The frozen route is a position command (plan section 5, "the frozen route is
# unchanged as a command -- hover 1.5 m through waypoints local-NED [2,0,0] and
# [2,1,0] with 8 s holds"); it declares no yaw, and the vehicle spawns at rest
# yaw 0 in the vestibule facing the doorway (plan section 0.3 item 1,
# dev-a-single/world.wbt), so the declared route's own heading is the spawn
# heading. Leaving yaw out of the mask does NOT leave the vehicle at that
# heading: a yaw-ignored SET_POSITION_TARGET in GUIDED calls
# set_yaw_state_rad(use_yaw=false) -> AutoYaw::set_mode_to_default
# (mode_guided.cpp), which at this pin's default WP_YAW_BEHAVIOR 2
# (LOOK_AT_NEXT_WP_EXCEPT_RTL) takes the yaw target from the position
# controller's velocity-heading alignment (AC_PosControl.cpp:1658-1669,
# "if vehicle is moving significantly, align yaw to velocity vector") -- and
# that firmware-invented maneuver is what tumbles the vehicle at the end of the
# route. Measured, four runs (dataflash 00000085, 00000105, 00000106,
# 00000107): the second leg's eastward motion slews yaw 0 -> ~51 deg at up to
# ~67 deg/s (the first yaw maneuver of every flight); the yaw rate loop then
# overshoots, develops a growing ~3-4 Hz roll-yaw oscillation during the
# waypoint-2 hold (achieved rates +-30..60 deg/s against single-digit commands,
# with the EKF attitude estimate tracking dataflash truth to ~0.2 deg the
# whole time), saturates the motor mix, tumbles from 1.5 m and crash-disarms
# inside the scored window ("Crash: Disarming: AngErr=93>30"). Commanding the
# declared heading keeps the flown route equal to the declared one: position
# only, no yaw maneuver. This rig's yaw-rate loop remains marginally stable
# (ATC_RAT_YAW_P 0.18, compat_base.parm -- not this stage's owned path) and is
# recorded as an open airframe item.
ROUTE_YAW_HOLD_RAD = 0.0

# The diagnostic's labels, at full strength (plan sections 0.7 item 7, 0.8 item 5).
DIAGNOSTIC_SENSOR_MODE_LABEL = "pose-assisted-diagnostic"
DIAGNOSTIC_NON_CLAIM = (
    "this is a pose-assisted diagnostic: the autopilot's external-navigation source "
    "is the simulator's own pose (truth republish ON by construction), so this is NOT "
    "a sensor-derived result and may never be pooled with one, and no predeclared "
    "E/F/H bound is judged by it (gate_status not_applicable)"
)
DIAGNOSTIC_TRUTH_EXEMPTION = (
    "the 4.6 truth-republish gate is exempt for this arm by declaration, not skipped "
    "silently: that gate keeps truth out of the estimator's INPUT, and this arm's "
    "estimator input is unchanged (stereo pairs and inertial samples only; the "
    "ov_stream protocol has no truth field). Only the vehicle's navigation is "
    "truth-driven here, which is exactly what makes the arm flyable before the "
    "estimator works"
)

# ---------------------------------------------------------------------------
# The declared ordered bring-up (plan sections 0.6 item 6 and 0.8 item 7,
# Deliverable B): what makes the sensor-derived arm flyable at all
# ---------------------------------------------------------------------------
#
# E1-DIAG settled the deadlock (work/runs/p01-localization/E1-DIAG-report.md):
# VioManager::initialized() = is_initialized_vio && timelastupdate != -1
# (VioManager.h:99), timelastupdate is written only at the tail of
# do_feature_propagate_update (VioManager.cpp:651), which needs motion -- and
# the arm gate forbids motion until the vision source is healthy
# (Check::VISION, AP_Arming.cpp:2090 -> AP_VisualOdom::pre_arm_check) and home
# exists (Check::GPS, AP_Arming.cpp:748). The diagnostic measured the pinned
# estimator latching under measured motion, so the remedy is an ORDERED
# BRING-UP: a bounded window that applies a DECLARED exception and the origin
# DATUM, flies the excitation, and then RESTORES the full check set with the
# vehicle's own readback before the scored window opens -- never a silent
# disable, and never anything that changes what supplies the estimator's input
# or the scored arm's pose.
#
# The exception window is carried as constants rather than configuration keys:
# the shared cli.py schema rejects unknown localization keys
# (cli.py:446-448) and the compatibility probe loads this file through that
# loader (webots_ardupilot.py:5734), so a key here would break
# `python -m embodied compat` -- the P00 gate's own command (plan section 0.8
# item 3, Correction C). configs/first_indoor.yaml carries the same
# declaration as a comment beside the section it belongs to.

# The exception mask's bits and its parameter. AP_Arming::check_enabled is
# `(checks_to_skip & uint32_t(check)) == 0` (AP_Arming.cpp:329-332), so a SET
# bit SKIPS that check and 0 is "skip nothing", i.e. every check enabled.
ARMING_CHECK_BIT_GPS = 1 << 3  # Check::GPS, AP_Arming.h:31
ARMING_CHECK_BIT_VISION = 1 << 18  # Check::VISION, AP_Arming.h:46
# The window ORs in exactly the two checks the plan declares excepted.
BRING_UP_ARMING_SKIP_WINDOW = ARMING_CHECK_BIT_GPS | ARMING_CHECK_BIT_VISION
BRING_UP_ARMING_ALL_CHECKS_ENABLED = 0
#
# The parameter that carries it is ARMING_SKIPCHK, not the plan's ARMING_CHECK,
# and that is a measured correction rather than a preference: the pinned
# firmware renamed it (AP_Arming.cpp:199-205, AP_GROUPINFO("SKIPCHK", 13,
# AP_Arming, checks_to_skip, 0); the migration comment at :233 "ARMING_CHECK ->
# ARMING_SKIPCHK" and the conversion at :235-260). There is no name alias: the
# first invocation of this run wrote and then read "ARMING_CHECK", the write
# took nothing and the readback answered nothing, and its own artifact records
# exactly that (work/runs/p01-localization/p01l-bringup-20260926T161038Z,
# bring-up.json: readbacks.window.ARMING_CHECK = null). A write under a name the
# vehicle does not have is silently dropped or refused, which is the same class
# of defect as the GPS_TYPE/GPS1_TYPE rename the scored arm already records
# (plan section 4.7). The value semantics are unchanged: 0 = all checks enabled,
# and the window sets the two bits above.
BRING_UP_ARMING_PARAMETER = "ARMING_SKIPCHK"

# The window's mode. ALT_HOLD is the E-EXC mode row of plan section 0.6 item 5,
# and it is the only mode the sensor-derived window can take off in:
# `requires_position()` is false (ArduCopter/mode.h:506) so no position
# estimate is required of an aircraft that has none yet. (The diagnostic
# carrier flies GUIDED -- plan section 0.7 item 4 -- because its truth-driven
# arm has a position by construction; that is a different carrier and is
# recorded as one.)
BRING_UP_MODE = "ALT_HOLD"

# The one arming check no mask can except, and the window's answer to it.
# `alt_checks` fails with "Need Alt Estimate" unless the mode has manual
# throttle (AP_Arming_Copter.cpp:551-557); it is reached from
# run_pre_arm_checks (:82) in the normal path AND from mandatory_checks
# (:654-665) when the mask skips everything (:72-73), and forcing the arm does
# not avoid it either (AP_Arming.cpp:1910). ALT_HOLD has no manual throttle
# (mode.h:509), and the two modes that do -- STABILIZE, ACRO (mode.h:454) --
# support no user takeoff at all (mode.h:132). What the check requires is
# `copter.ekf_alt_ok()` = have_inertial_nav && VERT_POS && VERT_VEL
# (ArduCopter/system.cpp:294-308); `status.flags.vert_pos` is
# `!hgtTimeout && ...` (AP_NavEKF3_Control.cpp:795) and hgtTimeout clears only
# on a height fusion from the SELECTED source (AP_NavEKF3_PosVelFusion.cpp:
# 1363-1379 for ExternalNav, :1046-1060 for the timeout), with no fallback from
# EXTNAV to BARO (AP_NavEKF_Source.cpp getPosZSource). So an aircraft whose
# only height source is the external navigation it has not received yet cannot
# arm in any mode that can take off -- and the window exists precisely because
# that stream does not exist yet. The window therefore carries a HEIGHT SOURCE
# for its own duration: the airframe's own barometer, the only height reference
# that is not simulator truth. It is a height reference only (no horizontal
# position, no attitude, no velocity), and it is restored to the seam's
# ExternalNav before the scored window opens, verified by the vehicle's own
# readback like every other window parameter.
BRING_UP_HEIGHT_SOURCE_PARAMETER = "EK3_SRC1_POSZ"
BRING_UP_HEIGHT_SOURCE_WINDOW_VALUE = 1.0  # AP_NavEKF_Source.h SourceZ::BARO
BRING_UP_HEIGHT_SOURCE_RESTORE_VALUE = 6.0  # SourceZ::EXTNAV, the declared seam

# The window's parameter writes, each with the value it holds DURING the
# window, the value it is RESTORED to before the claimed arm, and the pinned
# source that makes the window value necessary or harmless. The scored arm's
# declared state is the restore column: no check skipped (ARMING_SKIPCHK 0) and
# EK3_SRC1_POSZ 6 (ExternalNav, the seam's own declared height source). A window
# parameter that is not restored exactly is a blocker, not a warning
# (`_bring_up_closure_blockers`).
#
# The window writes no VISO_TYPE, and that is a recorded correction to plan
# section 0.6 item 6 phase 1's "VISO_TYPE 0 for this phase only". At this pin
# the backend is created once, from the value the parameter holds at boot:
# `AP_VisualOdom::init()` switches on `_type.get()` and allocates the driver
# (AP_VisualOdom.cpp:132-152), the vehicle calls it exactly once
# (AP_Vehicle.cpp:479), and every later message path forwards only when
# `_driver != nullptr` (:227-229, :243-245). A runtime 0 -> 1 would therefore
# leave `_driver` null for the rest of the run: the seam could never be
# established and the claimed arm could never receive the estimator's pose.
# The seam's own parameter is consequently left alone, and the Check::VISION
# exception of the mask is what covers the interval before the adapter's first
# publication -- which is exactly what that exception is for.
# The window's third parameter, and the reason it is the window's: with the
# throttle override in force the airframe is spooled to THROTTLE_UNLIMITED as soon
# as the mode's own state machine sees it, and MOT_IDLE_SEC -- the airframe's
# declared post-arm idle delay, 4.0 s in compat_arming.parm -- then holds the
# motors in GROUND_IDLE for longer than this window's whole declared airtime. It
# is a declared window parameter with a declared restore value because the scored
# run's guided takeoff needs the delay it was added for.
BRING_UP_MOTOR_IDLE_PARAMETER = "MOT_IDLE_SEC"
BRING_UP_MOTOR_IDLE_WINDOW_VALUE = 0.0  # the firmware's own default
BRING_UP_MOTOR_IDLE_RESTORE_VALUE = 4.0  # compat_arming.parm, iteration 11

BRING_UP_WINDOW_PARAMETERS: tuple[tuple[str, float, float, str], ...] = (
    (
        BRING_UP_ARMING_PARAMETER,
        float(BRING_UP_ARMING_SKIP_WINDOW),
        float(BRING_UP_ARMING_ALL_CHECKS_ENABLED),
        "the two excepted checks, each by its pinned bit: Check::VISION "
        "(AP_Arming.h:46; the check itself at AP_Arming.cpp:2087-2100 -> "
        "AP_VisualOdom::pre_arm_check 'not healthy', AP_VisualOdom.cpp:277,292) "
        "and the home requirement inside Check::GPS (AP_Arming.h:31; the check "
        "at AP_Arming.cpp:747-750, 'AHRS: waiting for home'). The parameter name "
        "is the pinned one (ARMING_SKIPCHK, AP_Arming.cpp:199-205): the plan's "
        "ARMING_CHECK is the pre-4.7 name and does not resolve at this pin",
    ),
    (
        BRING_UP_HEIGHT_SOURCE_PARAMETER,
        BRING_UP_HEIGHT_SOURCE_WINDOW_VALUE,
        BRING_UP_HEIGHT_SOURCE_RESTORE_VALUE,
        "the mandatory altitude check above: baro is the airframe's own height "
        "reference, needed only while the external navigation does not exist yet, "
        "and restored to ExternalNav before the scored window opens",
    ),
    (
        BRING_UP_MOTOR_IDLE_PARAMETER,
        BRING_UP_MOTOR_IDLE_WINDOW_VALUE,
        BRING_UP_MOTOR_IDLE_RESTORE_VALUE,
        "the airframe's own post-arm idle delay, and the window's second measured "
        "impediment: MOT_IDLE_SEC holds the motor library in spool state GROUND_IDLE "
        "for that long after the desired spool state becomes THROTTLE_UNLIMITED "
        "(`_idle_time_delay_s`, AP_MotorsMulticopter.cpp:684,716-719), which is "
        "longer than this window's whole declared airtime. The window sets it to the "
        "firmware's own default (0) for its duration only, and restores the "
        "airframe's declared 4.0 before the scored window opens: the scored run's "
        "guided takeoff needs the delay (compat_arming.parm, iteration 11)",
    ),
)

# ---------------------------------------------------------------------------
# The window's ONE thrust path: the bounded local RC throttle override
# ---------------------------------------------------------------------------
#
# WHY IT EXISTS. With GPS off and the estimator not yet latched, the modes that
# can arm are the position-free ones, and every one of them lifts only on a PILOT
# throttle:
#
#   * a mode that requires a position estimate (GUIDED, AUTO, LOITER) cannot arm
#     at all: `mandatory_position_checks` demands `position_ok()`
#     (AP_Arming_Copter.cpp:444-470) and position arrives only from the
#     estimator's own external-nav publication, which does not exist before the
#     estimator latches -- the circle this whole bring-up exists to break;
#   * the position-free modes lift only through `get_pilot_desired_climb_rate_ms()`
#     (ArduCopter/Attitude.cpp:74-115), which reads the RC throttle channel.
#
# MEASURED, not inferred (SITL dataflash work/ardupilot/logs/00000070.BIN, the
# third invocation of the previous session):
#
#   RCIN C3 = 1000            the pilot throttle sits at its minimum;
#   CTUN ThI 0.000 -> 0.891   the takeoff routine's own ramp ran to full;
#   MOTB ThrOut 0.0, RCOU 1000 the motors never left idle;
#   SPOL 15.904 s SplDes 2, 15.908 s Spl 1, 16.900 s SplDes 1, LAND 17.148 s.
#
# So exactly two things stood between an accepted takeoff and thrust:
#
#   1. the pilot climb rate was negative, so the mode's own spool-state branch
#      (`if (target_climb_rate_ms < 0.0f && !using_interlock) GROUND_IDLE else
#      THROTTLE_UNLIMITED`, ArduCopter/mode.cpp:1047-1055) never asked for
#      THROTTLE_UNLIMITED before the takeoff command arrived -- and the takeoff
#      state itself never sets a spool state, so the desired state stayed
#      SHUT_DOWN (Spl 0) for the first 2.6 s of the window;
#   2. once it was finally asked for (SplDes 2 at 15.904 s, 2.62 s after the arm),
#      MOT_IDLE_SEC 4.0 held the spool in GROUND_IDLE past the LAND at 17.148 s.
#
# This is the same structural gap the previous session named: the project has
# never sent an RC, throttle or pulse command (ALLOWED_OUTBOUND_TYPES has none),
# and no mode that can arm here has any other thrust path. The window therefore
# declares one, and it is a bounded LOCAL bring-up action: it is not a cloud
# command, it carries no setpoint, and the rule that the cloud sends intent while
# the planner owns setpoints is untouched by it. The claimed arm sends none, and
# the session that owns the claimed arm's wire has no such capability at all
# (`PymavlinkSession` / ALLOWED_OUTBOUND_TYPES are unchanged; the only object
# that can send it is the bring-up link).
#
# WHAT IS DECLARED IS A CLIMB RATE, NOT A STICK POSITION. The declared quantity is
# the PILOT CLIMB RATE the override must produce (BRING_UP_THROTTLE_CLIMB_RATE_M_S),
# because that is the physical quantity the mode consumes and the one the window's
# own airtime bound is about. The PWM that produces it is DERIVED from the
# vehicle's own answers -- RC3_MIN/MAX/DZ, THR_DZ and PILOT_SPD_UP, read back from
# the running vehicle -- through the firmware's own arithmetic
# (`get_pilot_desired_climb_rate_ms`, Attitude.cpp:98-113: deadband top =
# get_control_mid() + THR_DZ, and the linear map to PILOT_SPD_UP above it). A
# hard-coded PWM would silently mean a different rate on any other calibration,
# which is the same class of defect as a parameter write under a name the vehicle
# does not have. The PWM actually sent is recorded in the receipt, so the
# declaration is a rate and the receipt is the measurement.
BRING_UP_THROTTLE_CLIMB_RATE_M_S = 0.5
# The rate the window will not exceed: the airframe's declared maximum pilot
# climb rate is WP_SPD_UP 1.0 m/s (compat_arming.parm), and a rate above it would
# be a faster climb than any other part of this project commands.
BRING_UP_THROTTLE_MAX_CLIMB_RATE_M_S = 1.0
# The vehicle's own answers the derivation is made of. Every one of them is read
# back from the running vehicle before the window flies: nothing about the RC
# throttle calibration is assumed.
BRING_UP_THROTTLE_CALIBRATION: tuple[str, ...] = (
    "RC3_MIN",
    "RC3_MAX",
    "RC3_DZ",
    "THR_DZ",
    "PILOT_SPD_UP",
)
# The firmware's own RC override timeout is 3.0 s (`RC_OVERRIDE_TIME`,
# RC_Channels_VarInfo.h:90): an override that stops being refreshed lapses. The
# window refreshes well inside it, and the release is explicit.
BRING_UP_OVERRIDE_REFRESH_S = 0.5
# How long the window lets the mode see the override before it commands the
# takeoff. The override must be in force for at least one mode iteration BEFORE
# the takeoff command: the takeoff state does not set a spool state, so the
# desired state has to have become THROTTLE_UNLIMITED first (mode.cpp:1047-1055).
BRING_UP_OVERRIDE_SETTLE_S = 0.3
# How long the window waits for the vehicle's own RC report to show the release.
BRING_UP_OVERRIDE_RELEASE_TIMEOUT_S = 5.0
# The vehicle's own report of its RC input is requested at this rate: the
# RC_CHANNELS message carries chan3_raw = the *effective* RC input, override
# included (GCS_Common.cpp:2172-2205).
BRING_UP_RC_REPORT_HZ = 5.0
# The vehicle's own declared GCS system id, read back rather than assumed: the
# firmware ignores an RC override from any other system id
# (`sysid_is_gcs`, GCS_Common.cpp:4216-4220 -> GCS.cpp:727-734, whose value is
# MAV_GCS_SYSID). The window's link speaks as this id, so the override is not a
# silent void -- the same discipline as the parameter readback, applied to a
# message the vehicle is entitled to ignore.
BRING_UP_GCS_SYSTEM_PARAMETER = "MAV_GCS_SYSID"
BRING_UP_GCS_CONNECT_TIMEOUT_S = 15.0
# The retry cadence of the window's own arm loop. The bridge's own numbers are
# module-private (webots_ardupilot.py ARM_SETTLE_S = 1.0, CONTROL_RETRY_S = 5.0)
# and this loop is the window's, not the claimed arm's, so they are declared
# here rather than borrowed.
BRING_UP_ARM_SETTLE_S = 1.0
BRING_UP_ARM_RETRY_S = 5.0
# How long the window waits for the vehicle's own answer to each declaration:
# the GPS_GLOBAL_ORIGIN echo that proves the origin was accepted, and the
# COMMAND_ACK that carries the takeoff's result.
BRING_UP_ORIGIN_ECHO_TIMEOUT_S = 5.0
BRING_UP_TAKEOFF_ACK_TIMEOUT_S = 5.0
# The seam's own parameters, read back after the window so the receipt shows
# the claimed arm's declared source set restored rather than assumed.
BRING_UP_SEAM_READBACK: tuple[str, ...] = (
    "VISO_TYPE",
    "EK3_SRC1_POSXY",
    "EK3_SRC1_VELXY",
    "EK3_SRC1_POSZ",
    "EK3_SRC1_YAW",
)

# The declared exception window's other two elements, recorded as declarations
# with their citations so the receipt can be read without the plan.
BRING_UP_EXCEPTED_CHECKS: tuple[dict[str, Any], ...] = (
    {
        "check": "Check::VISION",
        "bit": ARMING_CHECK_BIT_VISION,
        "citation": (
            "AP_Arming.h:46 (the bit); AP_Arming.cpp:2087-2100 visodom_checks, "
            "gated on check_enabled(Check::VISION), calling "
            "AP_VisualOdom::pre_arm_check, which reports 'not healthy' while the "
            "last external-navigation message is older than "
            "AP_VISUALODOM_TIMEOUT_MS (AP_VisualOdom_Backend.cpp:32-37); "
            "AP_VisualOdom.cpp:277,292"
        ),
        "why": (
            "the vision source is the estimator's own adapter, and it publishes "
            "only once VioManager::initialized() is true -- which needs motion, "
            "which needs this arm. The check is excepted for the window only: the "
            "seam is not required during the window (VISO_TYPE 0), and it is back "
            "in force, with the vehicle's readback as proof, before the scored arm"
        ),
    },
    {
        "check": "Check::GPS (the home requirement)",
        "bit": ARMING_CHECK_BIT_GPS,
        "citation": (
            "AP_Arming.h:31 (the bit); AP_Arming.cpp:747-750 inside "
            "AP_Arming::gps_checks, gated on check_enabled(Check::GPS): "
            "'if (!AP::ahrs().home_is_set())' -> 'AHRS: waiting for home'. The "
            "GPS-fix half of the same check is vacuous here: with GPS1_TYPE and "
            "GPS2_TYPE 0 no receiver driver exists, so num_instances is 0 "
            "(AP_GPS.cpp:1067-1073) and the fix loop never runs"
        ),
        "why": (
            "home is derived from the EKF origin (Copter::update_home_from_EKF, "
            "commands.cpp:4-20), and at this pin EKF3 sets its origin only from "
            "GPS, a beacon or a GCS declaration -- never from ExternalNav data "
            "(AP_NavEKF3_Measurements.cpp:680-715; GCS_Common.cpp:3961). The "
            "window declares the origin as a DATUM instead, and home follows it; "
            "the exception covers the seconds between the arm attempt and that "
            "derivation"
        ),
    },
)

# What the exception does NOT change, stated where the code that keeps it true
# lives, because this is the whole justification for allowing the window at all.
BRING_UP_JUSTIFICATION = (
    "The exception changes only WHEN the aircraft may move, never what supplies "
    "the scored pose. The estimator is the same pinned build, fed the same "
    "declared stereo and inertial stream; the exception is a bounded window "
    "(<= 5.0 s from the arm readback to LAND, no lateral setpoint, LAND always) "
    "that lets the airframe move so the estimator CAN latch, and every element "
    "of it is restored -- with the vehicle's own parameter readback -- before "
    "the scored window opens. The scored window's pose source is unchanged: the "
    "adapter is still the only publisher on the autopilot's external-navigation "
    "source, the bridge's truth republish is still off by construction, and the "
    "window publishes nothing to the autopilot whose origin is not the "
    "estimator's own state."
)
# H5's measured cause, and the two markers that separate "the initializer never
# fired" from "the filter initialised but the pinned readiness accessor never
# became true". VioManager::initialized() is `is_initialized_vio && timelastupdate
# != -1` (VioManager.h:99), and timelastupdate is assigned only at the tail of
# do_feature_propagate_update (VioManager.cpp:651); the zero-velocity updater
# returns early from track_image_and_update before that assignment
# (VioManager.cpp:294), so a stationary start with try_zupt enabled reaches it
# only on a camera frame where ZUPT itself had no bracketing inertial data.
# Recorded from the run's own log instead of inferring the cause from the bound.
INITIALIZER_SUCCESS_MARKER = "[init]: successful initialization"
ZUPT_ACCEPTED_MARKER = "[ZUPT]: accepted"
ZUPT_STARVED_MARKER = "[ZUPT]: There are no IMU data"
# Revision 5 (plan section 12 item 14): the per-frame decision the criterion rests on.
# UpdaterZeroVelocity::try_update prints one disparity line and, when it reaches a
# verdict, one accept/reject line per camera frame (UpdaterZeroVelocity.cpp:231-249).
# The frames it declines are the only ones that reach do_feature_propagate_update,
# where propagate_and_clone makes the clones the readiness accessor waits for
# (VioManager.cpp:299-305, :341, :348) -- so counting them per frame turns plan
# section 0.6 item 5's sufficiency criterion into a measurement rather than an
# argument from the log tail.
ZUPT_DISPARITY_PATTERN = re.compile(
    r"\[ZUPT\]: (passed|failed) disparity \(([-\d.]+) [<>] ([-\d.]+), (\d+) features?\)"
)
ZUPT_VERDICT_PATTERN = re.compile(
    r"\[ZUPT\]: (accepted|rejected) \|v_IinG\| = ([-\d.]+) \(chi2 ([-\d.]+) [<>] ([-\d.]+)\)"
)
ZUPT_VISUAL_PATH_DECISIONS = ("declined_motion", "declined_no_imu")
# The artifact carries a bounded window of decisions, not the whole log: a pre-arm
# window at the declared 10 Hz holds hundreds of frames and the summary counts above
# carry the total.
ZUPT_FRAME_LIMIT = 64

# Revision 4: the run's own bounded static-start capture (plan section 0.3 item 4).
# Same conditions as P00's accept-5 capture -- the vehicle at the declared start
# on the ground -- recorded so the arm gate can measure the configured world
# itself when no hash-matched capture exists yet, and so every later preflight
# has a hash-anchored capture to measure instead of an unprovenanced one.
SCENE_CAPTURE_MAX_FRAMES = 24
SCENE_CAPTURE_MIN_SPACING_S = 0.3
SCENE_CAPTURE_TIMEOUT_S = 30.0

# A1: the pre-arm attitude gate (plan section 0.3 item 5). A frame-map defect is
# a 90- or 180-degree error, not a 5-degree one; the gate converts nothing and
# calibrates nothing -- it refuses the arm naming the per-axis error.
ATTITUDE_GATE_TOLERANCE_DEG = 5.0

# How many stereo pairs the reader's sink may hold for the feed. A pair is ~614 KB of
# pixels at the declared 640x480, and the sink is called from the reader's thread while
# the drain loop feeds the estimator, so a small bound keeps memory predictable while
# a healthy stream never fills it: at the declared 10 Hz this is 1.6 s of frames, and a
# backlog that deep is a stopped estimator, which the health machine is already about
# to catch.
PAIR_QUEUE_FRAMES = 16

# How long one parameter readback waits for the autopilot's own answer. A local SITL
# answers in well under a second; this is generous enough that an answer would have to
# be absent rather than slow, which is the distinction the gate depends on.
PARAMETER_READ_TIMEOUT_S = 5.0

# The additive localization section: the pin's identity, the publish cadence,
# the declared pipeline delay, the parameter layer, and the predeclared bounds.
# The bounds are the plan's values (plan section 6), frozen here before any
# measurement; they are never adjusted after seeing a run.
LOCALIZATION_SECTION: dict[str, Any] = {
    "mode": str,
    # The sensor-derived development route's world, optional because older
    # configurations -- and the compatibility gate's own -- do not carry it. The
    # shared cli.py schema carries the same key (eb6b180); this section
    # overrides that schema for the check's own loader, so the key must live in
    # both. The check runs in this world when present and falls back to
    # scenario.world otherwise (plan section 0.3 item 1).
    "world": _Optional(str),
    "estimator": {
        "name": str,
        "tag": str,
        "commit": str,
        "tarball_path": str,
        "tarball_sha256": str,
        "build_log": str,
        "build_success_marker": str,
        "library": str,
        "executable": str,
        "socket_port": int,
    },
    "publish": {"period_ms": int},
    "declared": {"viso_delay_ms": int},
    "params_file": str,
    "bounds": {
        "state_lost_after_ms": int,
        "published_state_age_max_ms": int,
        "max_publish_gap_ms": int,
        "visual_update_warn_ms": int,
        "visual_update_fail_ms": int,
        "valid_fraction_min": float,
        "sigma_min_m": float,
        "sigma_max_m": float,
        "disagreement_p95_m": float,
        "disagreement_max_m": float,
        # E1, against evaluator truth: plan-fixed values, the upper edge of the
        # advisory band, fitted to the smallest declared opening (plan section 6).
        "error_p95_horizontal_m": float,
        "error_p95_vertical_m": float,
        "error_max_horizontal_m": float,
        "error_max_vertical_m": float,
    },
}

# The schema this stage validates: the shared schema plus the additive
# localization section. The configuration's ``calibration`` section is P01-C's
# offline-derivation contract; this stage consumes only the derived record at
# sensors.calibration (and its declared start pose, below) and leaves that
# section to its owner, so it is excluded from the view validated here. The
# shared-file schema extension rides the same serialized cli.py edit as the
# dispatch registration.
LOCALIZATION_CONFIG_SCHEMA: dict[str, Any] = {
    **CONFIG_SCHEMA,
    "localization": LOCALIZATION_SECTION,
}


def _load_localization_config(path: Path) -> dict[str, Any]:
    """Load and validate the configuration for this stage.

    The loading discipline is the shared loader's; the one difference is the
    schema, which carries this stage's additive section.
    """
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - PyYAML is a pinned dependency
        raise ConfigError(f"PyYAML is required to read {path}: {error}") from error
    try:
        document = yaml.safe_load(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{path} is not readable YAML: {error}") from error
    if not isinstance(document, dict):
        raise ConfigError(f"{path} must hold a mapping at the top level")
    consumable = {key: value for key, value in document.items() if key != "calibration"}
    _validate(consumable, LOCALIZATION_CONFIG_SCHEMA, path.name)
    return document


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/first_indoor.yaml"),
        help="the scenario, platform and localization configuration this run declares",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[SensorMode.SENSOR_DERIVED.value, SensorMode.POSE_ASSISTED.value],
        required=True,
        help="sensor-derived is the claimed arm; pose-assisted is a labelled diagnostic",
    )


# ---------------------------------------------------------------------------
# Preflight: everything the claimed arm needs, reported in one pass
# ---------------------------------------------------------------------------


def _parse_mavlink_port(endpoint: str) -> int | None:
    tail = endpoint.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pin_evidence(localization: dict[str, Any], root: Path) -> tuple[dict[str, Any], list[str]]:
    """The pin's version evidence, measured on disk rather than quoted (plan section 3).

    A pin that is only asserted is not a pin. This re-derives the pinned tarball's
    sha256 from the bytes on disk, checks that the build log carries the success
    marker the configuration names, and records what the built library and the
    estimator process actually are. The record travels into the preflight beside
    the row it justifies, so a reader of the receipt sees the measurement instead
    of inferring it from the absence of a failure — the standard section 4.6's
    truth-republish gate was re-pointed at, applied to the pin leg.
    """
    estimator = localization["estimator"]
    build_log = root / estimator["build_log"]
    build_text = (
        build_log.read_text(encoding="utf-8", errors="replace") if build_log.is_file() else None
    )
    measured_tarball = _sha256(root / estimator["tarball_path"])
    record = {
        "name": estimator["name"],
        "tag": estimator["tag"],
        "commit": estimator["commit"],
        "tarball_path": estimator["tarball_path"],
        "tarball_sha256_configured": estimator["tarball_sha256"],
        "tarball_sha256_measured": measured_tarball,
        "tarball_matches_configured_pin": measured_tarball == estimator["tarball_sha256"],
        "build_log": estimator["build_log"],
        "build_success_marker": estimator["build_success_marker"],
        "build_marker_present": bool(
            build_text is not None and estimator["build_success_marker"] in build_text
        ),
        "library": {"path": estimator["library"], "sha256": _sha256(root / estimator["library"])},
        "executable": {
            "path": estimator["executable"],
            "sha256": _sha256(root / estimator["executable"]),
        },
    }
    blockers: list[str] = []
    if measured_tarball is None:
        blockers.append(f"the pinned tarball {estimator['tarball_path']} is not on disk")
    elif not record["tarball_matches_configured_pin"]:
        blockers.append(
            f"the pinned tarball's sha256 {measured_tarball} does not match the configured pin "
            f"{estimator['tarball_sha256']}"
        )
    if build_text is None:
        blockers.append(
            f"the pinned estimator has no build log at {estimator['build_log']}; a pin needs "
            "a successful build on this host as version evidence (plan section 3)"
        )
    elif not record["build_marker_present"]:
        blockers.append(
            f"the estimator build log {estimator['build_log']} does not record "
            f"{estimator['build_success_marker']!r}: no successful build of the pinned "
            "estimator tree is evidenced"
        )
    if record["library"]["sha256"] is None:
        blockers.append(f"the built estimator library {estimator['library']} is not on disk")
    return record, blockers


def _pin_summary(record: dict[str, Any]) -> str:
    """One line stating what was measured, for the preflight row's detail."""
    return (
        f"tarball {record['tarball_path']} sha256 {record['tarball_sha256_measured']} matches "
        f"the configured pin; the build log records {record['build_success_marker']!r}; "
        f"library {record['library']['path']} sha256 {record['library']['sha256']}; "
        f"process {record['executable']['path']} sha256 {record['executable']['sha256']}"
    )


def _executable_blockers(localization: dict[str, Any], root: Path) -> list[str]:
    executable = root / localization["estimator"]["executable"]
    if executable.is_file():
        return []
    return [
        f"the estimator process {localization['estimator']['executable']} does not exist; "
        "estimator/ov_stream.cpp is built by estimator/build-openvins.sh against the pinned "
        "tarball, and without that binary there is no estimator process to run"
    ]


def _seam_blockers(document: dict[str, Any], root: Path) -> list[str]:
    """The ExternalNav seam's parameter selection, read from the files that would run.

    Every requirement is reported, including the ones a missing file hides:
    a blocked run is most useful when it names everything that is absent in one
    pass, and the missing file and the missing selection have different owners.
    """
    paths = [root / name for name in document["scenario"]["estimator_params"]]
    blockers: list[str] = []
    for path in paths:
        if not path.is_file():
            blockers.append(
                f"the estimator parameter layer {path} is missing; the claimed arm applies "
                "exactly the layers scenario.estimator_params names, so a missing one is a "
                "selection that did not happen"
            )
    configured = read_configured_parameters([path for path in paths if path.is_file()])
    for name, expected, source in SEAM_REQUIREMENTS:
        actual = configured.get(name)
        if actual is None:
            blockers.append(
                f"{name} is set by no estimator parameter file; the claimed arm needs "
                f"{name} {expected:g} from {source}"
            )
        elif actual != expected:
            blockers.append(
                f"{name} is {actual:g} in the applied parameter files; the claimed arm needs "
                f"{name} {expected:g} from {source}"
            )
    if not any(path.name == P01L_PARAMS_FILENAME for path in paths):
        blockers.append(
            f"{P01L_PARAMS_FILENAME} is not listed in scenario.estimator_params; GPS cannot "
            "be disabled for the claimed arm without it"
        )
    return blockers

def _mode_blockers(document: dict[str, Any], mode: SensorMode) -> list[str]:
    """The arm the configuration declares and the arm asked for must be the same arm.

    The bridge's truth republish follows ``localization.mode``, so a configuration that
    declares one arm while the invocation asks for another would run a mode it did not
    declare.
    """
    declared = document["localization"].get("mode")
    if declared == mode.value:
        return []
    return [
        f"the configuration declares localization.mode {declared!r} but this invocation "
        f"asks for {mode.value!r}; the bridge's truth republish follows the configuration, "
        "so the two must agree before anything starts"
    ]


def _truth_republish_blockers(settings: PlatformSettings) -> list[str]:
    """Whether the bridge that will be built republishes the simulator's own pose (4.6).

    Read from the settings object the bridge's own ``from_config`` produced, so this is
    the switch's actual value and not prose about it. With it on, the bridge's vision
    feed would send simulator pose to the autopilot's external-navigation source while
    the estimator's adapter publishes to that same source: a scored arm would fly on
    truth it must not receive.
    """
    if not settings.truth_republish:
        return []
    return [
        "the bridge's truth republish is ON (PlatformSettings.truth_republish is True): its "
        "vision feed would send simulator pose to the autopilot's external-navigation "
        "source, which the estimator's adapter also publishes to, so a scored arm would "
        "receive truth it must not (plan section 4.6)"
    ]


def _readback_blockers(applied: dict[str, float], refusals: dict[str, int]) -> list[str]:
    """The vehicle's own parameter readback against the claimed arm (plan section 4.7).

    The applied file is what we asked for; this is what the autopilot reported about
    itself, which is the only statement of the configuration actually running. Every
    required name must be answered with the claimed value, and the outcomes are kept
    distinct (G2): answered, refused — the vehicle itself replied that the name does
    not exist, which means the layer that named it never took effect — and silent,
    which is a readback that cannot confirm anything and so confirms nothing.
    """
    blockers: list[str] = []
    for name, expected, _source in VEHICLE_REQUIREMENTS:
        if name in applied:
            if applied[name] != expected:
                blockers.append(
                    f"{name} read back as {applied[name]:g} from the vehicle itself; "
                    f"the claimed arm needs {name} {expected:g}"
                )
        elif name in refusals:
            error = refusals[name]
            error_name = (
                "MAV_PARAM_ERROR_DOES_NOT_EXIST"
                if error == MAV_PARAM_ERROR_DOES_NOT_EXIST
                else str(error)
            )
            blockers.append(
                f"the vehicle refused {name} with PARAM_ERROR {error_name}: a defaults "
                "line under a name the vehicle does not have is silently dropped "
                "(AP_Param.cpp:2421-2431), so the layer that named it did not take effect"
            )
        else:
            blockers.append(
                f"{name} was never answered by the vehicle; a silent readback confirms "
                "nothing and the claimed arm does not run on it"
            )
    return blockers


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_bytes(document: dict[str, Any]) -> bytes | None:
    """The raw frame behind a recorded message dict, whatever shape the recorder kept."""
    data = document.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str) and data.startswith("bytearray(") and data.endswith(")"):
        try:
            parsed = ast.literal_eval(data[len("bytearray(") : -1])
        except (ValueError, SyntaxError):
            return None
        return parsed if isinstance(parsed, bytes) else None
    return None


def _decode_param_error(document: dict[str, Any]) -> dict[str, Any] | None:
    """One PARAM_ERROR, decoded from the vehicle's own frame (plan section 4.7 G2).

    A pymavlink new enough to know the message records it decoded; this host's records
    it as UNKNOWN_345 with the raw frame, which is parsed against the wire layout
    pinned with the constants above.
    """
    kind = str(document.get("mavpackettype", ""))
    if kind == "PARAM_ERROR":
        param_id = str(document.get("param_id", "")).rstrip("\x00")
        index = document.get("param_index")
        error = document.get("error")
        if param_id and index is not None and error is not None:
            return {"param_id": param_id, "param_index": int(index), "error": int(error)}
        return None
    if kind != f"UNKNOWN_{PARAM_ERROR_MSG_ID}":
        return None
    frame = _frame_bytes(document)
    if frame is None or len(frame) < 10 + _PARAM_ERROR_PAYLOAD.size:
        return None
    if frame[7] | (frame[8] << 8) | (frame[9] << 16) != PARAM_ERROR_MSG_ID:
        return None
    param_index, target_system, target_component, raw_id, error = _PARAM_ERROR_PAYLOAD.unpack(
        frame[10 : 10 + _PARAM_ERROR_PAYLOAD.size]
    )
    param_id = raw_id.split(b"\x00")[0].decode("utf-8", errors="replace")
    return {
        "param_id": param_id,
        "param_index": param_index,
        "error": error,
        "target_system": target_system,
        "target_component": target_component,
    }


def _param_error_refusals(mavlink_log: Path) -> dict[str, int]:
    """Every parameter name the vehicle itself refused, from the run's own record."""
    refusals: dict[str, int] = {}
    if not mavlink_log.is_file():
        return refusals
    with mavlink_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(document, dict):
                continue
            decoded = _decode_param_error(document)
            if decoded and decoded["param_index"] == -1 and decoded["param_id"]:
                refusals.setdefault(decoded["param_id"], decoded["error"])
    return refusals


_GPS_STATUSTEXT = re.compile(r"GPS\s*\d*\s*:", re.IGNORECASE)


def _gps_aiding_verdict(mavlink_log: Path) -> dict[str, Any]:
    """What the vehicle reported about GPS over the covered window (plan section 4.7 G4).

    Three signals, each required clean: no SYS_STATUS sample may carry the GPS-present
    bit, no GPS_RAW_INT sample may carry a fix, and no STATUSTEXT may be the driver's
    own probe or detect notice. Zero samples is not a pass: an unsampled claim is an
    asserted-only prerequisite, which this gate exists to remove.
    """
    sys_status = 0
    gps_present = 0
    raw_int = 0
    fix_types: list[int] = []
    statustexts: list[str] = []
    if mavlink_log.is_file():
        with mavlink_log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    document = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(document, dict):
                    continue
                kind = document.get("mavpackettype")
                if kind == "SYS_STATUS":
                    sys_status += 1
                    present = int(document.get("onboard_control_sensors_present") or 0)
                    if present & GPS_SENSOR_PRESENT_BIT:
                        gps_present += 1
                elif kind == "GPS_RAW_INT":
                    raw_int += 1
                    fix_type = document.get("fix_type")
                    if fix_type is not None:
                        fix_types.append(int(fix_type))
                elif kind == "STATUSTEXT":
                    text = str(document.get("text", ""))
                    if _GPS_STATUSTEXT.search(text):
                        statustexts.append(text)
    blockers: list[str] = []
    if sys_status == 0:
        blockers.append(
            "the vehicle's SYS_STATUS was never sampled, so GPS-off is unconfirmed"
        )
    if gps_present:
        blockers.append(
            f"SYS_STATUS carried the GPS-present bit (MAV_SYS_STATUS_SENSOR_GPS) in "
            f"{gps_present} of {sys_status} samples"
        )
    if any(fix_type != 0 for fix_type in fix_types):
        blockers.append(
            f"GPS_RAW_INT reported a fix: fix_type values {sorted(set(fix_types))} over "
            f"{raw_int} samples"
        )
    for text in statustexts:
        blockers.append(f"the GPS driver announced itself in STATUSTEXT: {text!r}")
    return {
        "sys_status_samples": sys_status,
        "sys_status_gps_present": gps_present,
        "gps_raw_int_samples": raw_int,
        "fix_types": sorted(set(fix_types)),
        "statustexts": statustexts,
        "blockers": blockers,
    }


def _params_applied_record(applied: dict[str, float], refusals: dict[str, int]) -> dict[str, Any]:
    """Every required name with its outcome kept distinct (plan section 4.7 G2)."""
    record: dict[str, Any] = {}
    for name, expected, source in VEHICLE_REQUIREMENTS:
        if name in applied:
            record[name] = {
                "outcome": "answered",
                "value": applied[name],
                "expected": expected,
                "source": source,
            }
        elif name in refusals:
            record[name] = {
                "outcome": "refused",
                "param_error": refusals[name],
                "expected": expected,
                "source": source,
            }
        else:
            record[name] = {"outcome": "silent", "expected": expected, "source": source}
    return record


def _find_world_sha256(node: Any) -> str | None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "world_sha256" and isinstance(value, str):
                return value
            found = _find_world_sha256(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_world_sha256(item)
            if found:
                return found
    return None


def _recorded_world_sha256(run_dir: Path) -> str | None:
    """The world hash a capture's own artifacts record, when one is recorded."""
    for path in sorted(run_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        found = _find_world_sha256(document)
        if found:
            return found
    return None


def _platform_settings(
    document: dict[str, Any], root: Path, arm: str | None = None
) -> PlatformSettings:
    """Settings with the sensor-derived development route applied (plan 0.3 item 1).

    ``localization.world``, when present, is the world this check runs in; an
    absent key leaves ``scenario.world`` -- and with it the compatibility gate's
    measured vehicle -- untouched. The shared schema carries the key as optional
    (main eb6b180), so older configurations load unchanged.

    ``arm`` is the per-run override of the localization mode, the bridge's own
    existing mechanism (``PlatformSettings.from_config``): the arm is a property
    of the RUN, not of the file. The sensor-derived arm passes no override and
    follows the configuration's declaration -- which is what keeps its truth
    republish off. Only the pose-assisted diagnostic passes one, so the bridge
    republishes the simulator's pose for that run alone while the configuration's
    declared arm stays untouched (plan sections 0.7 item 5, 0.8 item 3).
    """
    settings = PlatformSettings.from_config(document, root=root, arm=arm)
    world_name = (document.get("localization") or {}).get("world")
    if not world_name:
        return settings
    world = Path(world_name).expanduser()
    if not world.is_absolute():
        world = settings.root / world
    return replace(settings, world=world)


_VEHICLE_TRANSLATION_RE = re.compile(
    r"^\s*translation\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
)


def _declared_start_origin(world: Path) -> tuple[float, float, float]:
    """The configured world's own vehicle translation, as the odom origin (0.3 item 3).

    The odom origin is a property of the world the route flies. The calibration
    referee's declared start cannot serve: it disagrees with the compat world's
    own spawn by 7 cm in z (plan section 13's recorded gap), and it describes the
    compat scene's depth checks, not this route. ``mission.yaml`` pins the
    vehicle node's translation as the spawn ("identical to the Iris translation
    in world.wbt, plan section 6 pin 8"), so the world file is the anchor.
    """
    lines = world.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("Iris {"):
            for candidate in lines[index + 1 : index + 40]:
                match = _VEHICLE_TRANSLATION_RE.match(candidate)
                if match:
                    return (
                        float(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                    )
            break
    raise ConfigError(
        f"{world} declares no Iris vehicle translation; the odom origin cannot be derived "
        "from the scene (plan section 0.3 item 3)"
    )

_VEHICLE_ROTATION_RE = re.compile(
    r"^\s*rotation\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
    r"\s+(-?\d+(?:\.\d+)?)"
)


def _declared_start_attitude(world: Path) -> tuple[float, float, float]:
    """The vehicle's declared start attitude, as NED roll/pitch/yaw (0.3 item 5).

    The declared stationary start is what the epoch rotation is derived from,
    together with the estimator's own first initialized attitude, because the
    odom frame's yaw is unobservable (see ``OdomAlignment.seal``). A pure yaw is
    accepted and converted from Webots' rotation about +z; a tilted start is
    refused rather than approximated, since a wrong declaration would rotate the
    whole published frame. No rotation field means the identity start, which is
    what both declared worlds carry; ``mission.yaml``'s ``spawn_pose.yaw_rad``
    is the independent cross-check.
    """
    lines = world.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("Iris {"):
            for candidate in lines[index + 1 : index + 20]:
                stripped = candidate.strip()
                if stripped.startswith("controllerArgs") or stripped.startswith("children"):
                    break
                match = _VEHICLE_ROTATION_RE.match(candidate)
                if match:
                    x, y, z, angle = (float(value) for value in match.groups())
                    norm = math.sqrt(x * x + y * y + z * z)
                    if norm == 0.0:
                        return (0.0, 0.0, 0.0)
                    axis = (x / norm, y / norm, z / norm)
                    if abs(abs(axis[2]) - 1.0) > 1e-6:
                        raise ConfigError(
                            f"{world} declares a tilted start rotation {match.groups()}; "
                            "the declared start must be level with a pure yaw (plan "
                            "section 0.3 item 5)"
                        )
                    # A right-handed rotation about +z turns north toward west, which
                    # is a negative NED yaw.
                    yaw = -angle if axis[2] > 0.0 else angle
                    return (0.0, 0.0, yaw)
            return (0.0, 0.0, 0.0)
    raise ConfigError(
        f"{world} declares no Iris vehicle node; the start attitude cannot be derived "
        "(plan section 0.3 item 5)"
    )


_SITL_SERIAL_PORT_RE = re.compile(r"^SERIAL(\d+) on TCP port (\d+)$", re.MULTILINE)


def _autopilot_feed_endpoint(
    sitl_log: Path, session_endpoint: str, *, already_taken: Sequence[str] = ()
) -> str:
    """The autopilot link one seam will open (plan section 0.5).

    The pinned SITL serves exactly one TCP client per serial port
    (``UARTDriver.cpp``: a single ``accept()``, then ``_connected``), and this
    check's own session already owns the configured port. A second client on that
    port is accepted by the kernel and never read -- the third textured
    invocation published 2404 poses into exactly such a connection and the
    autopilot reported ``VisOdom: not healthy``. The free port is taken from the
    running SITL's own declaration of what it listens on, which its log records;
    no port is guessed and no convention is assumed.

    ``already_taken`` names endpoints this run has already given to another
    connection, so the adapter's publisher and the ordered bring-up's own link
    get distinct ports instead of contending for the same one.
    """
    session_port = session_endpoint.rsplit(":", 1)[-1]
    taken = {session_port}
    for endpoint in already_taken:
        taken.add(endpoint.rsplit(":", 1)[-1])
    if sitl_log.is_file():
        ports = [
            (int(number), int(port))
            for number, port in _SITL_SERIAL_PORT_RE.findall(
                sitl_log.read_text(encoding="utf-8", errors="replace")
            )
            if str(port) not in taken
        ]
        if ports:
            number, port = sorted(ports)[0]
            return f"tcp:127.0.0.1:{port}"
    raise ConfigError(
        f"{sitl_log} declares no free SITL serial port besides {session_endpoint}: the "
        "adapter cannot be given a link the autopilot actually serves, and publishing "
        "into an unserved one is a silent void (plan section 0.5)"
    )

def _fast_keypoint_counts(frame_paths: Sequence[Path]) -> list[int] | None:
    """Measure frames with exactly the detector call the pinned tracker makes.

    Returns None when OpenCV is not importable -- unmeasurable, not zero.
    """
    try:
        import cv2
    except ImportError:
        return None
    detector = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=True)
    counts: list[int] = []
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            counts.append(len(detector.detect(image, None)))
    return counts


def _stereo_capture_measurement(pairs_dir: Path, limit: int) -> dict[str, Any]:
    """Per-eye FAST counts for a recorded capture, or why it cannot gate a stereo arm.

    The gate answers one question — can the pinned tracker's front end see this
    scene — and the arm it gates feeds the estimator two image planes. A capture
    holding only the left eye cannot answer it, and neither can one whose two eyes
    are byte-identical: one view duplicated is not a stereo pair. Both are reported
    as unusable rather than measured. The defect this replaces is a gate that passed
    at 16 left-eye keypoints while nothing in the run recorded what the second view
    contained, so a stereo arm could open on a single eye's worth of evidence.
    """
    left_paths = sorted(path for path in pairs_dir.glob("*-left.ppm") if path.is_file())
    measurement: dict[str, Any] = {
        "pairs_dir": str(pairs_dir),
        "pairs": 0,
        "left_counts": [],
        "right_counts": [],
        "missing_right": [],
        "identical_pairs": [],
        "cv2_unavailable": False,
    }
    pairs: list[tuple[Path, Path]] = []
    for left_path in left_paths:
        right_path = left_path.with_name(left_path.name.replace("-left.ppm", "-right.ppm"))
        if not right_path.is_file():
            measurement["missing_right"].append(left_path.name)
            continue
        if left_path.read_bytes() == right_path.read_bytes():
            measurement["identical_pairs"].append(left_path.name)
            continue
        pairs.append((left_path, right_path))
    measurement["pairs"] = len(pairs)
    if not pairs:
        return measurement
    measured = pairs[:limit]
    left_counts = _fast_keypoint_counts([left for left, _right in measured])
    right_counts = _fast_keypoint_counts([right for _left, right in measured])
    if left_counts is None or right_counts is None:
        measurement["cv2_unavailable"] = True
        return measurement
    measurement["left_counts"] = left_counts
    measurement["right_counts"] = right_counts
    return measurement


def _scene_admission_check(
    settings: PlatformSettings, root: Path
) -> tuple[bool, str, str]:
    """T7: the scene must admit the pinned initializer, measured on its own frames.

    The initializer needs at least 15 trackable features per window and the pinned
    tracker hunts with FAST at the pinned threshold with non-max suppression. The
    newest recorded capture of this scenario is measured with exactly that call.

    Revision 4, two changes and one unchanged property. A capture that records no
    world hash is now skipped rather than measured: it cannot prove which world
    its frames show, and with a development route configured it would measure one
    scene against another world's gate. And when no hash-matched capture of the
    configured world exists, the check defers to the arm gate instead of refusing
    to start: the run records its own static-start frames with the world's hash,
    and the arm is refused unless those measured frames clear the floor -- so no
    flight can be spent on a scene the initializer cannot fire in, which is the
    property that has held since run 4's receipt.

    Revision 5 adds a third: the capture must hold **both eyes**. The arm this gates
    feeds the estimator two planes, and every capture recorded before this revision
    holds only `-left.ppm` -- so the gate passed at 16 left-eye keypoints while
    nothing in the run showed what the second view contained. A capture missing its
    right eye, or whose right eye is byte-identical to its left, is now skipped
    rather than measured, exactly as an unhashed capture is, and the run's own
    arm-gate capture records and measures both.
    """
    capture_dirs: list[Path] = []
    for pattern in SCENE_ADMISSION_CAPTURE_GLOBS:
        capture_dirs.extend(path for path in root.glob(pattern) if path.is_dir())
    current_sha256 = _sha256(settings.world)
    unhashed = 0
    unusable = 0
    for pairs_dir in sorted(capture_dirs, key=lambda path: path.stat().st_mtime, reverse=True):
        if not any(pairs_dir.glob("*-left.ppm")):
            continue
        recorded = _recorded_world_sha256(pairs_dir.parent)
        if recorded is None:
            unhashed += 1
            continue
        if recorded != current_sha256:
            continue
        measurement = _stereo_capture_measurement(pairs_dir, SCENE_ADMISSION_MAX_FRAMES)
        if measurement["cv2_unavailable"]:
            return (
                False,
                "cv2_unavailable",
                "the scene-admission check needs OpenCV (cv2) to measure the scene's "
                "recorded frames the way the pinned tracker does; it is not importable",
            )
        measured_both_eyes = (
            measurement["pairs"]
            and measurement["left_counts"]
            and measurement["right_counts"]
        )
        if not measured_both_eyes:
            unusable += 1
            continue
        lowest_left = min(measurement["left_counts"])
        lowest_right = min(measurement["right_counts"])
        lowest = min(lowest_left, lowest_right)
        passed = lowest >= INITIALIZER_FEATURE_FLOOR
        detail = (
            f"{measurement['pairs']} recorded stereo pairs of {settings.world} measured "
            f"with FAST({FAST_THRESHOLD}, non-max suppression) in both eyes: keypoint "
            f"counts left {measurement['left_counts']}, right "
            f"{measurement['right_counts']}, against the pinned initializer's floor of "
            f"{INITIALIZER_FEATURE_FLOOR} features per window; the capture's recorded "
            "world sha256 matches the configured world"
        )
        return passed, ("measured_pass" if passed else "measured_fail"), detail
    detail = (
        f"no hash-matched recorded stereo capture of {settings.world} (sha256 "
        f"{current_sha256}) exists under the accepted-run artifacts ({unhashed} "
        "capture(s) skipped for recording no world hash, "
        f"{unusable} for recording no complete, distinct, readable stereo pair); the "
        "arm gate will measure this run's own static-start frames in both eyes against "
        f"the pinned initializer's floor of {INITIALIZER_FEATURE_FLOOR} features and "
        "refuse the arm if they do not clear it (plan section 0.3 item 4)"
    )
    return True, "deferred_to_arm_gate", detail


def _zupt_frame_decisions(lines: Sequence[str]) -> list[dict[str, Any]]:
    """The zero-velocity updater's per-frame decision, in the order it made them.

    One record per camera frame the updater was asked about, carrying the numbers it
    decided from: the mean disparity against ``zupt_max_disparity`` with the feature
    count, and, where the frame reached a verdict, the velocity and chi2 against their
    limits. The decisions are the three ways out of ``UpdaterZeroVelocity::try_update``
    the log can show: ``accepted`` (the frame is consumed, UpdaterZeroVelocity.cpp:248),
    ``declined_motion`` (chi2 or velocity over its limit, the frame reaches the visual
    path, :244) and ``declined_no_imu`` (no bracketing inertial data, same consequence,
    :104). A frame whose disparity line was written but whose verdict never followed --
    the log tail, or a process that stopped mid-frame -- keeps ``no_verdict`` rather
    than being silently classified.
    """
    frames: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for line in lines:
        if ZUPT_STARVED_MARKER in line:
            frames.append({"decision": "declined_no_imu"})
            pending = None
            continue
        disparity = ZUPT_DISPARITY_PATTERN.search(line)
        if disparity is not None:
            pending = {
                "disparity_passed": disparity.group(1) == "passed",
                "disparity_px": float(disparity.group(2)),
                "max_disparity_px": float(disparity.group(3)),
                "feature_count": int(disparity.group(4)),
            }
            continue
        verdict = ZUPT_VERDICT_PATTERN.search(line)
        if verdict is not None:
            frames.append(
                {
                    **(pending or {}),
                    "decision": (
                        "accepted" if verdict.group(1) == "accepted" else "declined_motion"
                    ),
                    "velocity_m_s": float(verdict.group(2)),
                    "chi2": float(verdict.group(3)),
                    "chi2_limit": float(verdict.group(4)),
                }
            )
            pending = None
    if pending is not None:
        frames.append({**pending, "decision": "no_verdict"})
    return frames


def _initializer_diagnostics(estimator_log: Path) -> dict[str, Any]:
    """What the estimator itself said about initialization, from its own log.

    The pinned initializer prints nothing while its feature database is empty — the
    silent return at InertialInitializer.cpp:86-88 — so an empty initializer_output
    behind consumed frames means the scene's pixels gave the front end nothing to
    track, which the preflight's scene_admission check measures directly.

    Revision 5 adds the per-frame decisions themselves: the summary counts say how
    many frames went each way, and ``zupt_frames`` carries the numbers each decision
    was made from, so the sufficiency criterion of plan section 0.6 item 5 (the
    accessor needs five frames that ZUPT declined before its first completed visual
    update can write ``timelastupdate``) is measurable per frame from the artifact.

    Revision 6 adds the decisive marker: ``timelastupdate``'s assignment at
    VioManager.cpp:651 is immediately followed by a ``q_GtoI`` line at PRINT_INFO
    (:654), while ``ov_stream`` reports ``initialized=`` only every 25th frame
    (ov_stream.cpp:488-490) -- so the report count alone cannot distinguish "never
    latched" from "latched after the last report" (plan section 0.7 item 2,
    Correction B).
    """
    lines: list[str] = []
    if estimator_log.is_file():
        lines = estimator_log.read_text(encoding="utf-8", errors="replace").splitlines()
    frames = _zupt_frame_decisions(lines)
    return {
        "estimator_log_lines": len(lines),
        "initializer_output": [line for line in lines if "[init" in line][-50:],
        "progress_reports": [line for line in lines if "initialized=" in line][-12:],
        "q_GtoI_tail_lines": sum(1 for line in lines if "q_GtoI = " in line),
        "q_GtoI_last_lines": [line for line in lines if "q_GtoI = " in line][-12:],
        "initializer_succeeded": any(
            INITIALIZER_SUCCESS_MARKER in line for line in lines
        ),
        "zupt_accepted_updates": sum(
            1 for line in lines if ZUPT_ACCEPTED_MARKER in line
        ),
        "zupt_frames_without_imu": sum(
            1 for line in lines if ZUPT_STARVED_MARKER in line
        ),
        "zupt_rejected_updates": sum(
            1 for frame in frames if frame["decision"] == "declined_motion"
        ),
        "zupt_frames_reaching_visual_path": sum(
            1 for frame in frames if frame["decision"] in ZUPT_VISUAL_PATH_DECISIONS
        ),
        "zupt_frames": frames[-ZUPT_FRAME_LIMIT:],
        "note": (
            "the pinned initializer prints nothing while its feature database is empty; "
            "an empty initializer_output behind consumed frames means the scene gave "
            "the front end nothing to track. initializer_succeeded says whether the "
            "initializer's own success line is present; initialized() also needs "
            "timelastupdate, set only by a completed visual update, which the "
            "zero-velocity updater pre-empts (VioManager.cpp:294). "
            "zupt_frames_reaching_visual_path counts the frames ZUPT declined -- the "
            "only ones that reach do_feature_propagate_update and can make a clone"
        ),
    }


def _initialization_blocker(diagnostics: dict[str, Any]) -> str:
    """H5's stop reason, stated from the estimator's own log (plan section 11).

    Two different failures read the same from the bound alone, and they point a
    later session in opposite directions: an initializer that never fired is a
    scene/feature question, while an initializer that fired behind a readiness
    accessor that stayed false is a filter-configuration question. The first
    textured re-run's receipt said "did not initialize" while its own
    initializer_output recorded the success line, and the next plan revision was
    written against the wrong cause; this function is what stops that reading.
    """
    if diagnostics["initializer_succeeded"]:
        return (
            "the pinned estimator's readiness accessor stayed false through the whole "
            "pre-arm window (H5), although the initializer itself fired: its own log "
            f"records {INITIALIZER_SUCCESS_MARKER!r}, while VioManager::initialized() "
            "(VioManager.h:99) is `is_initialized_vio && timelastupdate != -1` and "
            "timelastupdate is assigned only at the tail of do_feature_propagate_update "
            "(VioManager.cpp:651). The zero-velocity updater returned early from "
            "track_image_and_update before that assignment (VioManager.cpp:294): "
            f"{diagnostics['zupt_accepted_updates']} accepted zero-velocity update(s), "
            f"{diagnostics['zupt_rejected_updates']} frame(s) declined on motion and "
            f"{diagnostics['zupt_frames_without_imu']} frame(s) where it had no "
            "bracketing inertial data, so "
            f"{diagnostics['zupt_frames_reaching_visual_path']} frame(s) reached the "
            "visual path, against the five the accessor needs before its first "
            "completed update can write timelastupdate (VioManager.cpp:348-352). No "
            "state was published, so no bound was measured; changing the "
            "zero-velocity configuration or the H5 criterion is the owner's "
            "disposition, not a worker's"
        )
    return (
        "the estimator did not initialize inside the pre-arm window (H5): the pinned "
        "initializer's own log records no successful initialization, and initialization "
        "is from the declared stationary launch interval -- a degenerate initialization "
        "is unavailable-navigation, not a delayed arm"
    )

# ---------------------------------------------------------------------------
# The declared ordered bring-up (plan sections 0.6 item 6, 0.8 item 7)
# ---------------------------------------------------------------------------


_SITL_HOME_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def _sitl_home_origin_datum(sitl_home: str) -> tuple[float, float, float]:
    """The configured vehicle's own spawn coordinate, as the origin datum.

    ``platform.sitl_home`` is the coordinate the autopilot itself is launched
    with (``--home``, webots_ardupilot.py sitl_argv), i.e. the world's own
    (0, 0, 0) in geographic terms. Declaring the EKF's origin there and nowhere
    else keeps the local frame the autopilot navigates in identical to the frame
    the scored arm's E1 comparison already uses. A configuration without a
    parsable home cannot declare an origin, and a GPS-off invocation with no
    origin cannot fly at all, so this raises rather than guessing.
    """
    match = _SITL_HOME_RE.match(sitl_home or "")
    if match is None:
        raise ConfigError(
            f"platform.sitl_home {sitl_home!r} is not 'lat,lon,alt[,yaw]'; the ordered "
            "bring-up declares the EKF origin from it and cannot invent one (plan "
            "section 0.6 item 6)"
        )
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def _bring_up_throttle_pwm(
    calibration: Mapping[str, float], climb_rate_ms: float
) -> tuple[int, float]:
    """The RC throttle PWM that yields ``climb_rate_ms``, from the vehicle's own numbers.

    The arithmetic is the pinned firmware's, not an approximation of it, because the
    quantity being derived is exactly what the firmware will compute from the value
    this function returns:

    * ``get_control_mid()`` for a RANGE channel is
      ``high_in * (mid_radio - radio_trim_low) / (radio_max - radio_trim_low)`` with
      ``radio_trim_low = radio_min + dead_zone`` (RC_Channel.cpp:329-340) and
      ``high_in`` 1000 (Copter::init_rc_in calls ``channel_throttle->set_range(1000)``,
      ArduCopter/radio.cpp:32);
    * ``pwm_to_range()`` (RC_Channel.cpp) maps a raw value the same way, so the
      inversion below is the same line rearranged;
    * above the deadband ``get_pilot_desired_climb_rate_ms`` is
      ``PILOT_SPD_UP * (throttle_control - deadband_top) / (1000 - deadband_top)``
      with ``deadband_top = mid_stick + THR_DZ`` (Attitude.cpp:98-113).

    The integer division is mirrored with the same truncation the firmware uses, so
    the rate reported beside the PWM is the rate the firmware computes rather than
    the ideal one. A calibration that cannot express the rate inside the channel's
    own range raises rather than returning a value that means something else.
    """
    missing = [name for name in BRING_UP_THROTTLE_CALIBRATION if name not in calibration]
    if missing:
        raise ConfigError(
            "the throttle override's derivation needs the vehicle's own answers for "
            f"{missing}: the RC throttle calibration decides what a channel value "
            "means, and this window does not guess it"
        )
    radio_min = int(calibration["RC3_MIN"])
    radio_max = int(calibration["RC3_MAX"])
    dead_zone = int(calibration["RC3_DZ"])
    throttle_deadzone = int(calibration["THR_DZ"])
    speed_up_ms = float(calibration["PILOT_SPD_UP"])
    range_low = radio_min + dead_zone
    span = radio_max - range_low
    if span <= 0:
        raise ConfigError(
            f"the throttle channel's own calibration is empty: RC3_MIN {radio_min} + "
            f"RC3_DZ {dead_zone} leaves no range below RC3_MAX {radio_max}"
        )
    mid_stick = int(1000 * ((radio_min + radio_max) // 2 - range_low) / span)
    deadband_top = mid_stick + throttle_deadzone
    if speed_up_ms <= 0.0 or deadband_top >= 1000:
        raise ConfigError(
            f"the throttle channel cannot express a climb: PILOT_SPD_UP {speed_up_ms} "
            f"m/s and deadband top {deadband_top} (mid stick {mid_stick} + THR_DZ "
            f"{throttle_deadzone}) leave no range above the deadband"
        )
    target_control_in = int(
        deadband_top + (1000 - deadband_top) * climb_rate_ms / speed_up_ms
    )
    pwm = int(range_low + span * target_control_in / 1000)
    if pwm > radio_max:
        raise ConfigError(
            f"the declared pilot climb rate {climb_rate_ms} m/s needs {pwm} us on "
            f"channel {RC_THROTTLE_CHANNEL}, above this vehicle's own RC3_MAX "
            f"{radio_max}: the calibration cannot express it"
        )
    measured_control_in = int(1000 * (pwm - range_low) / span)
    if measured_control_in <= deadband_top:
        raise ConfigError(
            f"the derived channel value {pwm} us lands at or below this vehicle's own "
            f"throttle deadband top {deadband_top}: the declared pilot climb rate "
            f"{climb_rate_ms} m/s would not be commanded at all"
        )
    measured_rate_ms = (
        speed_up_ms * (measured_control_in - deadband_top) / (1000.0 - deadband_top)
    )
    return pwm, measured_rate_ms


def _bring_up_window(settings: PlatformSettings) -> dict[str, Any]:
    """The declared exception window, every element with its citation.

    This is the receipt's own statement of what the bring-up is allowed to do,
    built from the frozen constants above and the configuration's own declared
    home. The scored window is judged by none of it: the envelope bounds here
    are the excitation's, they are the E-EXC numbers the diagnostic flew, and
    the restoration rows are the state the claimed arm is required to be in.
    """
    latitude, longitude, altitude = _sitl_home_origin_datum(settings.sitl_home)
    return {
        "platform_limitation": (
            "the bridge's own session cannot send this window's declarations and this "
            "run does not widen it: PARAM_SET and SET_GPS_GLOBAL_ORIGIN are not in "
            "PymavlinkSession's ALLOWED_OUTBOUND_TYPES (webots_ardupilot.py:1951-1958) "
            "and PymavlinkSession.takeoff hardcodes param3 = 0 "
            "(webots_ardupilot.py:2248-2264), which ALT_HOLD's user takeoff refuses. The "
            "bring-up therefore sends them on its own connection, the pattern the "
            "adapter's publisher already established (localization.py "
            "ExternalNavPublisher.start). Worth fixing later in the bridge, so a future "
            "window is not a second code path; not fixed here, because that file is "
            "outside this slice's bounded ownership"
        ),
        "kind": "declared_ordered_bring_up",
        "authority": (
            "plan sections 0.6 item 6 (the ordered bring-up) and 0.8 item 7 "
            "(Deliverable B), whose evidence is the E1-DIAG report: the pinned "
            "estimator latches under measured motion (188 STATE frames, "
            "initialized true) and the sensor-derived arm cannot move"
        ),
        "window": {
            "mode": BRING_UP_MODE,
            "takeoff_altitude_m": EXCITATION_TAKEOFF_ALTITUDE_M,
            "lateral_setpoint": None,
            "max_airtime_s": EXCITATION_MAX_AIRTIME_S,
            "post_land_drain_s": EXCITATION_POST_LAND_DRAIN_S,
            "ends_with": (
                "LAND (always), then the parameter restoration below, then the "
                "vehicle's own readback of it"
            ),
            "why_it_excites_propagation": (
                "VioManager::initialized() needs timelastupdate written once "
                "(VioManager.cpp:651), which needs do_feature_propagate_update past "
                "the clone gate (:348), which needs >= 5 camera frames the "
                "zero-velocity updater declined (UpdaterZeroVelocity.cpp:246: mean "
                "disparity >= 1.0 px together with a chi2 or velocity violation). A "
                "climb supplies both terms: thrust makes the accelerometer read other "
                "than gravity, and at this rig's 554 px focal length ~2.7 mm of "
                "translation is ~1 px of disparity"
            ),
        },
        "origin_datum": {
            "message": "SET_GPS_GLOBAL_ORIGIN",
            "message_id": 48,
            "frame": "MAV_FRAME_GLOBAL / Location::AltFrame::ABSOLUTE (GCS_Common.cpp:3982-4014)",
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "altitude_msl_m": altitude,
            "source": "platform.sitl_home, the coordinate the autopilot is launched with",
            "is_a_pose_feed": False,
            "is": (
                "a DATUM: the local frame's geographic anchor, set once. It carries "
                "no vehicle position, attitude or velocity, and nothing about it "
                "measures where the aircraft is"
            ),
            "why_it_must_be_declared": (
                "EKF3 sets its origin from GPS, from a beacon, or from this GCS "
                "declaration -- never from ExternalNav data "
                "(AP_NavEKF3_Measurements.cpp:680-715; GCS_MAVLINK::set_ekf_origin, "
                "GCS_Common.cpp:3961). With GPS off no origin would exist, so the "
                "aircraft could not derive home (Copter::update_home_from_EKF, "
                "commands.cpp:4-20) and the claimed arm's own Check::GPS could not "
                "pass. The diagnostic sharpened this: the truth-driven carrier got "
                "its origin from the synthesized GPS, and the scored layer has none"
            ),
        },
        "excepted_arming_checks": [dict(row) for row in BRING_UP_EXCEPTED_CHECKS],
        "arming_skip": {
            "name": BRING_UP_ARMING_PARAMETER,
            "window_value": float(BRING_UP_ARMING_SKIP_WINDOW),
            "restore_value": float(BRING_UP_ARMING_ALL_CHECKS_ENABLED),
            "bit_semantics": (
                "a SET bit SKIPS that check: check_enabled = (checks_to_skip & check) "
                "== 0 (AP_Arming.cpp:329-332); 0 is 'skip nothing', i.e. every check "
                "enabled, which is the PINNED parameter's own recommended default and "
                "its restore value here. The pinned name is ARMING_SKIPCHK "
                "(AP_Arming.cpp:199-205; renamed from ARMING_CHECK, migration at "
                ":233-260) and the old name does not resolve at this pin"
            ),
        },
        "parameter_window": [
            {
                "name": name,
                "window_value": window_value,
                "restore_value": restore_value,
                "why": why,
            }
            for name, window_value, restore_value, why in BRING_UP_WINDOW_PARAMETERS
        ],
        "not_exceptable": [
            {
                "check": "the mandatory altitude check ('Need Alt Estimate')",
                "citation": (
                    "AP_Arming_Copter.cpp:551-557; reached from run_pre_arm_checks :82 "
                    "and from mandatory_checks :654-665 when the mask skips every check "
                    "(:72-73); AP_Arming::arm() runs mandatory_checks even when arming "
                    "checks are disabled (AP_Arming.cpp:1910)"
                ),
                "resolution": (
                    "the window carries its own height reference "
                    f"({BRING_UP_HEIGHT_SOURCE_PARAMETER} = "
                    f"{BRING_UP_HEIGHT_SOURCE_WINDOW_VALUE:g}, baro) instead of "
                    "excepting the check, and restores it to the seam's "
                    f"{BRING_UP_HEIGHT_SOURCE_RESTORE_VALUE:g} (ExternalNav) before the "
                    "claimed arm"
                ),
            },
        ],
        "justification": BRING_UP_JUSTIFICATION,
        "thrust_path": {
            "kind": "bounded_local_bring_up_action",
            "statement": (
                "This is a bounded local bring-up action, and the one place in this "
                "project where a throttle leaves the program: the window sends ONE RC "
                "throttle channel override (RC_CHANNELS_OVERRIDE, channel "
                f"{RC_THROTTLE_CHANNEL}) with every other field left at MAVLink's own "
                "'ignore this field', for this window's declared duration only, and "
                "releases it before the scored window opens. It is local: the cloud "
                "still sends intent and the planner still owns setpoints, and nothing "
                "about this message is a setpoint, a pose or a mode"
            ),
            "channel": RC_THROTTLE_CHANNEL,
            "channel_name": "throttle",
            "declared_climb_rate_ms": BRING_UP_THROTTLE_CLIMB_RATE_M_S,
            "max_climb_rate_ms": BRING_UP_THROTTLE_MAX_CLIMB_RATE_M_S,
            "climb_target_m": EXCITATION_TAKEOFF_ALTITUDE_M,
            "max_airtime_s": EXCITATION_MAX_AIRTIME_S,
            "refresh_s": BRING_UP_OVERRIDE_REFRESH_S,
            "release": (
                "a zero on the same channel (RC_THROTTLE_RELEASE_PWM) clears the "
                "override outright, and the vehicle's own RC report has to show the "
                "channel back at its radio value before the scored window opens"
            ),
            "derived_from_the_vehicle": list(BRING_UP_THROTTLE_CALIBRATION),
            "why_a_rate_rather_than_a_stick": (
                "the quantity the mode consumes is a pilot CLIMB RATE "
                "(get_pilot_desired_climb_rate_ms, Attitude.cpp:74-115), and the "
                "channel value that produces it depends on this vehicle's own RC "
                "calibration and deadbands; the PWM is derived from the vehicle's "
                "readback of RC3_MIN/RC3_MAX/RC3_DZ/THR_DZ/PILOT_SPD_UP and the "
                "derivation is recorded with the value actually sent"
            ),
            "why_it_is_needed": (
                "with GPS off and the estimator unlatched, a mode that needs a "
                "position cannot arm (mandatory_position_checks, AP_Arming_Copter.cpp:"
                "444-470) and the position-free modes lift only on a pilot throttle; "
                "measured on work/ardupilot/logs/00000070.BIN the pilot throttle sat "
                "at its minimum (RCIN C3 1000), so the pilot climb rate was negative "
                "and the mode never asked for THROTTLE_UNLIMITED (SPOL Spl 0 for the "
                "window's first 2.6 s; mode.cpp:1047-1055), then MOT_IDLE_SEC 4.0 held "
                "the spool in GROUND_IDLE past the LAND"
            ),
            "confirmed_by_the_vehicle": (
                "the window asks the vehicle for RC_CHANNELS "
                f"(msgid {MSG_ID_RC_CHANNELS}) through COMMAND_LONG 511 and reads "
                "chan3_raw, which is rc().get_radio_in() per channel "
                "(GCS_Common.cpp:2172-2205) and therefore carries the override value "
                "while one is in force: the vehicle's own report is the evidence that "
                "the override took and that it was lifted, not this link's send record"
            ),
            "sent_by_the_scored_arm": False,
        },
        "scored_window_requires": [
            {
                "name": name,
                "value": restore_value,
                "meaning": "the vehicle's own readback must equal this before the arm",
            }
            for name, _window_value, restore_value, _why in BRING_UP_WINDOW_PARAMETERS
        ],
    }


def _bring_up_closure_blockers(
    applied: dict[str, float], rc_override: dict[str, Any] | None = None
) -> list[str]:
    """Whether the exception window is fully closed, from the vehicle's own answer.

    The window is bounded in two ways and this is the second one: not only is it
    time-bounded, it cannot still be in force when the scored window opens. Every
    parameter the window wrote is compared against the value the configuration
    declares for the scored arm, and a parameter still at its window value -- or
    one the vehicle did not answer at all -- refuses the arm. A silent readback
    confirms nothing, so it is a refusal rather than a pass, exactly as the
    claimed arm's own readback treats it.

    The RC throttle override is judged the same way and on the same rule, with the
    vehicle's own answer as the evidence: ``rc_override`` is the window's record of
    what the vehicle reported about its own RC input -- ``sent``, ``released`` and
    ``observed_after_release``. An override the window declared and never released
    is in force; an override whose release the vehicle did not confirm is not shown
    to be lifted; and a window that declared an override but has no vehicle answer
    at all is a refusal, because a silent vehicle cannot show that a command it
    never acknowledged has stopped.
    """
    blockers: list[str] = []
    for name, window_value, restore_value, _why in BRING_UP_WINDOW_PARAMETERS:
        if name not in applied:
            blockers.append(
                f"{name} was never answered by the vehicle after the bring-up window: a "
                "silent readback cannot show that the declared exception was lifted, so "
                "the scored window does not open"
            )
        elif applied[name] == window_value:
            blockers.append(
                f"{name} is still {window_value:g} -- its declared bring-up window value -- "
                f"at the claimed arm; the window must be restored to {restore_value:g} and "
                "read back before the scored window opens (plan sections 0.6 item 6, 0.8 "
                "item 7)"
            )
        elif applied[name] != restore_value:
            blockers.append(
                f"{name} read back as {applied[name]:g} after the bring-up window; the "
                f"scored arm's declared value is {restore_value:g}"
            )
    if rc_override is None:
        return blockers
    channel = rc_override.get("channel", RC_THROTTLE_CHANNEL)
    sent_pwm = rc_override.get("sent_pwm")
    if not rc_override.get("sent"):
        return blockers
    if not rc_override.get("released"):
        blockers.append(
            f"the bring-up window's RC throttle override (channel {channel}, "
            f"{sent_pwm} us) was never released: the window's thrust path is still in "
            "force at the claimed arm, and the scored window sends no RC override, so "
            "it does not open on a window that has one"
        )
        return blockers
    observed = rc_override.get("observed_after_release")
    if observed is None:
        blockers.append(
            "the bring-up window's RC throttle override was released on the wire, but "
            "the vehicle never answered with its own RC input (RC_CHANNELS chan"
            f"{channel}_raw) after the release: a silent vehicle cannot show that the "
            "override stopped, and the scored window does not open on an override the "
            "vehicle's own report still has to deny"
        )
    elif sent_pwm is not None and int(observed) == int(sent_pwm):
        blockers.append(
            f"the vehicle's own RC report still reads {observed} us on channel "
            f"{channel} -- the bring-up window's override value -- after the window "
            "released it, so the override is still in force and the scored window does "
            "not open"
        )
    return blockers
# ---------------------------------------------------------------------------
# Preflight: everything the claimed arm needs, reported in one pass
# ---------------------------------------------------------------------------
def _preflight(
    document: dict[str, Any], output_dir: Path, mode: SensorMode
) -> tuple[list[dict[str, Any]], bool]:
    """Every prerequisite, reported in one pass. Nothing is started by this function.

    Row order is the order a reader needs: the arm's own declaration first, then the
    gate that decides whether truth can reach the estimate, then whether the scene can
    feed the pinned estimator at all, then the files and selections the claimed arm
    applies.
    """
    root = repository_root()
    settings = _platform_settings(document, root)
    rows: list[dict[str, Any]] = []
    satisfied = True
    mode_blockers = _mode_blockers(document, mode)
    rows.append(
        {
            "name": "localization_mode",
            "satisfied": not mode_blockers,
            "detail": mode_blockers[0]
            if mode_blockers
            else f"the configuration declares localization.mode {mode.value}, the arm asked for",
        }
    )
    satisfied = satisfied and not mode_blockers
    truth_blockers = _truth_republish_blockers(settings)
    rows.append(
        {
            "name": "bridge_truth_republish",
            "satisfied": not truth_blockers,
            "detail": truth_blockers[0]
            if truth_blockers
            else "the bridge's truth republish is off: the autopilot's external-navigation "
            "source has exactly one publisher, the estimator's adapter",
        }
    )
    satisfied = satisfied and not truth_blockers
    scene_ok, scene_state, scene_detail = _scene_admission_check(settings, root)
    rows.append(
        {
            "name": "scene_admission",
            "satisfied": scene_ok,
            "state": scene_state,
            "detail": scene_detail,
        }
    )
    satisfied = satisfied and scene_ok
    for check in check_prerequisites(settings, output_dir):
        rows.append({"name": check.name, "satisfied": check.satisfied, "detail": check.detail})
        satisfied = satisfied and check.satisfied
    mavlink_port = _parse_mavlink_port(settings.mavlink_endpoint)
    port_free = mavlink_port is not None and _port_is_free(mavlink_port)
    rows.append(
        {
            "name": "port_mavlink",
            "satisfied": port_free,
            "detail": f"tcp {settings.mavlink_endpoint} is "
            + ("free" if port_free else "already in use; kill orphaned SITL processes first"),
        }
    )
    satisfied = satisfied and port_free
    localization = document["localization"]
    pin_record, pin_blockers = _pin_evidence(localization, root)
    rows.append(
        {
            "name": "estimator_pin",
            "satisfied": not pin_blockers,
            "detail": pin_blockers[0] if pin_blockers else _pin_summary(pin_record),
            "evidence": pin_record,
        }
    )
    satisfied = satisfied and not pin_blockers
    for blocker in (
        *_executable_blockers(localization, root),
        *_seam_blockers(document, root),
    ):
        rows.append({"name": "estimator_seam", "satisfied": False, "detail": blocker})
        satisfied = False
    return rows, satisfied


# ---------------------------------------------------------------------------
# Command outcomes
# ---------------------------------------------------------------------------


def _blocked_unresolved(
    reasons: Sequence[str],
    limitations: Sequence[str],
    manifest: dict[str, Any],
    artifacts: Sequence[str],
    mode: SensorMode,
) -> CommandOutcome:
    """The unresolved protocol: blocked, every concrete blocker named, nothing faked."""
    return CommandOutcome(
        status=CommandStatus.BLOCKED,
        gate_status=GateStatus.NOT_APPLICABLE,
        reasons=(*reasons, "localization=unresolved"),
        limitations=limitations,
        manifest={**manifest, "localization": "unresolved"},
        artifacts=tuple(artifacts),
        sensor_mode=mode,
    )


def _pose_assisted_outcome(
    reasons: Sequence[str],
    manifest: dict[str, Any] | None = None,
    artifacts: Sequence[str] = (),
) -> CommandOutcome:
    """The labelled diagnostic outcome: never a gate result, never pooled.

    Whatever the diagnostic measured, the outcome cannot be read as a scored one:
    ``gate_status`` is not applicable, ``localization`` is not applicable, and the
    limitations carry the non-claim and the truth-exemption in full. A completed
    diagnostic (``reasons`` empty) is still COMPLETE -- its measurement is a
    result, and the receipt's labels are what keep it out of every E/F/H verdict.
    """
    return CommandOutcome(
        status=CommandStatus.BLOCKED if reasons else CommandStatus.COMPLETE,
        gate_status=GateStatus.NOT_APPLICABLE,
        reasons=(*reasons, "diagnostic mode cannot pass P01-L"),
        limitations=(
            f"sensor_mode {DIAGNOSTIC_SENSOR_MODE_LABEL}: " + DIAGNOSTIC_NON_CLAIM,
            DIAGNOSTIC_TRUTH_EXEMPTION,
            DISPATCH_REGISTRATION_NOTE,
        ),
        manifest={
            "stage_id": STAGE_ID,
            "sensor_mode_label": DIAGNOSTIC_SENSOR_MODE_LABEL,
            "localization": "not_applicable",
            **(manifest or {}),
        },
        artifacts=tuple(artifacts),
        sensor_mode=SensorMode.POSE_ASSISTED,
    )


def _localize_check_command(args: argparse.Namespace, output_dir: Path) -> CommandOutcome:
    mode = SensorMode(args.mode)
    document = _load_localization_config(Path(args.config))
    if mode is SensorMode.POSE_ASSISTED:
        return _run_pose_assisted_diagnostic(document, output_dir)

    rows, satisfied = _preflight(document, output_dir, mode)
    preflight = {
        "mode": mode.value,
        "sensor_mode_label": "sensor-derived",
        "checks": rows,
        "satisfied": satisfied,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    if not satisfied:
        blockers = tuple(f"{row['name']}: {row['detail']}" for row in rows if not row["satisfied"])
        return _blocked_unresolved(
            blockers,
            (
                "no process was started and nothing was substituted: the claimed arm runs "
                "only when every prerequisite exists",
                "P01-L is blocked for scored sensor-only mode; pose-assisted diagnostics "
                "carry on labelled",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {
                "stage_id": STAGE_ID,
                "sensor_mode_label": "sensor-derived",
                "preflight": rows,
                "estimator_pin": document["localization"]["estimator"],
                "bounds": document["localization"]["bounds"],
            },
            ("preflight.json",),
            mode,
        )
    return _run_sensor_derived_live(document, output_dir)


# ---------------------------------------------------------------------------
# The live sensor-derived run
# ---------------------------------------------------------------------------

SIMULATOR_CLOCK_SOURCE = (
    "the simulator's own clock: the controller stamps every frame it sends with "
    "robot.getTime() (scenarios/compat/controllers/compat_vehicle_controller/"
    "sensors.py simulator_time_s), the frame carries it as sim_time_s, and the feed "
    "reads it on every record. A window declared in simulated seconds is spent "
    "against this clock, so the window means the same thing on a fast host and on a "
    "loaded one -- the unit probe's own windows already declare "
    "(\"a window measured in simulated time is comparable between the realtime and "
    "fast modes\", configs/first_indoor.yaml)")
#
# Simulation time arrives as a float, so a window that ends exactly on a frame
# boundary can miss it by a rounding step; the bridge's own sim-time window carries
# the same tolerance for the same reason (webots_ardupilot.py
# TIME_COMPARISON_TOLERANCE_S).
SIM_WINDOW_TOLERANCE_S = 1e-6
# How long a declared window's wait polls between readings, when the wait does not
# have a cadence of its own. The windows are decided by the simulator's clock, not by
# this pacing: it only bounds how stale a reading can be.
SIM_WINDOW_POLL_S = 0.05


def _sim_window_wall_ceiling_s(budget_s: float, envelope_floor: float) -> float:
    """The wall-clock ceiling on a budget declared in SIMULATOR seconds.

    ``probe.realtime_ratio_envelope`` declares the simulator's rate as simulated
    seconds per wall second, so its floor is the slowest pace the run admits: inside
    the envelope a budget of N simulated seconds cannot take longer than N/floor wall
    seconds. The ceiling is a liveness guard, not the window's deadline -- a
    simulator that stops advancing would otherwise hang the run (the bridge's own
    AT_REST_WALL_CLOCK_LIMIT_S states the same rule for its sim-time window). It is
    derived from the declared envelope rather than chosen, and the budget is checked
    before it, so a window whose simulated budget was genuinely spent is recorded as
    the budget being honoured no matter what the host did.
    """
    if envelope_floor <= 0.0:
        raise ConfigError("the declared realtime envelope's floor must be positive")
    return float(budget_s) / float(envelope_floor)


class _SimulatorClock:
    """The newest simulator time the feed has read, and how many frames carried one.

    A frame that carries no usable sim time (the sentinel the bridge uses before the
    controller has stamped anything) is not an observation and does not move the
    clock. The clock only moves forward: a frame read out of order is not a reason to
    make a window think its budget has been given back.
    """

    def __init__(self) -> None:
        self.newest_s: float | None = None
        self.frames = 0

    def observe(self, sim_time_s: float | None) -> None:
        if sim_time_s is None or sim_time_s < 0.0:
            return
        self.frames += 1
        if self.newest_s is None or sim_time_s > self.newest_s:
            self.newest_s = sim_time_s


class _SimWindow:
    """A declared budget in SIMULATOR seconds, and the wall clock's own ceiling on it.

    The ordered bring-up's windows are windows of the AIRCRAFT's time, and the
    code once spent them on the wall clock. Measured on a loaded host
    (run-2026-09-28T02-43-27-442Z): the simulator ran at 0.40-0.92 simulated seconds
    per wall second, the declared 3.5 s climb window bought 2.20 simulated seconds,
    and the excitation was cut off -- LAND at 2.34 simulated seconds after the arm,
    with the thrust path's first motor output 0.80 simulated seconds later, at
    18.923 s of autopilot time (the airframe's own log, 00000141.BIN). The budget is
    spent against the simulator's clock here; ``ended_by`` says which clock decided
    it, so a reader can tell a window that ran its declared course from one the host
    truncated.
    """

    def __init__(
        self,
        clock: _SimulatorClock,
        budget_s: float,
        *,
        label: str,
        wall_ceiling_s: float,
        wall_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.clock = clock
        self.budget_s = float(budget_s)
        self.label = label
        self.wall_ceiling_s = float(wall_ceiling_s)
        self._wall_clock = wall_clock
        # The simulator's own reading when the window opened. A window opens with
        # frames already flowing, so this is a reading of a running scene; if the
        # stream has not delivered one yet the first one to arrive becomes the start,
        # and the wall ceiling covers the case where none ever does.
        self.started_sim_s = clock.newest_s
        self.started_wall_s = wall_clock()
        self.ended_by: str | None = None
        # What the window cost, frozen when it ended. The readings have to be taken
        # THEN and not when the receipt is written: every window of one bring-up is
        # documented together at the end, so a live reading would report each window as
        # lasting until the last one closed -- measured on the first run with this code
        # (run-a/bring-up.json in work/runs/p01-localization/
        # p01l-clockfix-20260928T041129Z), where the 0.3 s override settle reported
        # 14.092 s because the clock was read after the descent.
        self.ended_sim_s: float | None = None
        self.ended_wall_s: float | None = None

    def _freeze(self) -> None:
        """Take this window's final readings, once."""
        if self.ended_wall_s is None:
            self.ended_wall_s = self._wall_clock()
            self.ended_sim_s = self.clock.newest_s

    def elapsed_simulator_s(self) -> float | None:
        """Simulated seconds this window cost, frozen once it has ended."""
        if self.ended_wall_s is not None:
            return self._span(self.ended_sim_s)
        return self._span(self.clock.newest_s)

    def _span(self, now_s: float | None) -> float | None:
        if self.started_sim_s is None:
            self.started_sim_s = self.clock.newest_s
        if self.started_sim_s is None or now_s is None:
            return None
        return max(0.0, now_s - self.started_sim_s)

    def elapsed_wall_s(self) -> float:
        """Host seconds this window cost, frozen once it has ended."""
        end = (
            self.ended_wall_s
            if self.ended_wall_s is not None
            else self._wall_clock()
        )
        return max(0.0, end - self.started_wall_s)

    def expired(self) -> bool:
        """Whether the window is over, recording which clock ended it.

        The declared budget is tested first, deliberately: a window that spent its
        simulated seconds has been given what it was declared, whatever the host
        took to produce them. The ceiling therefore only ever decides a window whose
        simulated budget was NOT spent -- which means the simulator ran slower than
        the declared envelope's floor over that window, the condition the probe's own
        realtime evidence already reports as timing-invalid.
        """
        if self.ended_by is not None:
            return True
        elapsed = self.elapsed_simulator_s()
        if elapsed is not None and elapsed >= self.budget_s - SIM_WINDOW_TOLERANCE_S:
            self._freeze()
            self.ended_by = "simulator"
            return True
        if self.elapsed_wall_s() >= self.wall_ceiling_s:
            self._freeze()
            self.ended_by = "wall_ceiling"
            return True
        return False

    def close(self, reason: str) -> None:
        """Record how a window ended when something other than the clocks ended it."""
        if self.ended_by is None:
            self._freeze()
            self.ended_by = reason
    def document(self) -> dict[str, Any]:
        """A JSON-ready view: the budget, what it cost in each clock, and what ended it."""
        elapsed = self.elapsed_simulator_s()
        return {
            "label": self.label,
            "budget_simulator_s": self.budget_s,
            "elapsed_simulator_s": None if elapsed is None else round(elapsed, 3),
            "elapsed_wall_s": round(self.elapsed_wall_s(), 3),
            "wall_ceiling_s": round(self.wall_ceiling_s, 3),
            "ended_by": self.ended_by,
            "unit": (
                "the budget is in SIMULATOR seconds and is spent against "
                f"{SIMULATOR_CLOCK_SOURCE}; the wall ceiling is budget / the declared "
                "realtime envelope's floor and only ends a window whose simulated "
                "budget was not spent"
            ),
        }


class _FeedStats:
    """What the estimator feed consumed, and what truth was read beside it.

    ``truth_samples`` is the evaluator-truth channel: the controller's own pose samples
    arrive on the same sensor stream as the pairs and the inertial samples, and the feed
    writes them here for scoring while sending them nowhere. The estimator receives
    stereo and inertial frames and nothing else, which is the whole point of keeping the
    two lists side by side in one object.
    """

    def __init__(self) -> None:
        self.pairs = 0
        self.imu_samples = 0
        self.pair_latencies_ns: list[int] = []
        self.imu_latencies_ns: list[int] = []
        self.newest_imu_ns = 0
        # The scene's own clock, read beside the frames it stamps. Every window this
        # stage declares in simulated seconds is spent against this one, so the same
        # declaration costs the same amount of the aircraft's time on any host.
        self.sim_clock = _SimulatorClock()
        self.truth_samples: list[tuple[int, tuple[float, float, float]]] = []
        # A1's evidence channel: the same pose records' attitudes, read for the
        # pre-arm attitude gate and sent nowhere (plan section 0.3 item 5).
        self.truth_attitudes: list[tuple[int, tuple[float, float, float]]] = []
        # Pairs the reader's sink offered with their pixels, and how many of those the
        # feed's bounded queue could not take. The two together are the difference
        # between "the stream carried no pairs" and "the feed was too slow".
        self.pair_records_filed = 0
        self.pair_records_dropped = 0


def _start_estimator(estimator: dict[str, Any], root: Path, writer: EvidenceWriter):
    executable = root / estimator["executable"]
    port = int(estimator["socket_port"])
    log_path = writer.path("estimator.log")
    process = subprocess.Popen(
        [str(executable), str(port)],
        cwd=str(root),
        stdout=log_path.open("ab"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if _port_listening(port):
            return process
        if process.poll() is not None:
            return None
        time.sleep(0.1)
    process.terminate()
    return None


def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _stop_estimator(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()


def _write_health_events(writer: EvidenceWriter, machine: loc.HealthMachine) -> None:
    with writer.path("health-events.jsonl").open("a", encoding="utf-8") as handle:
        for event in machine.events:
            handle.write(
                json.dumps(
                    {
                        "at_ns": event.at_ns,
                        "event": event.event,
                        "detail": event.detail,
                        "reset_counter": machine.reset_counter,
                    }
                )
                + "\n"
            )


def _write_log(writer: EvidenceWriter, lines: Sequence[str]) -> None:
    writer.path("log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_environment(
    writer: EvidenceWriter, settings: PlatformSettings, estimator: dict[str, Any]
) -> None:
    """Which tree and which interpreter this run actually measured (plan section 9).

    A stage can be run from a worktree while the installed package resolves to a
    different one; that is not a detail a reader should have to infer from a stray
    failure, so the resolved module paths are part of the run's record.
    """
    import embodied
    from embodied.platform import webots_ardupilot as bridge

    writer.write_json(
        "environment.json",
        {
            "interpreter": sys.executable,
            "python_version": sys.version,
            "embodied_module": str(Path(embodied.__file__).resolve()),
            "bridge_module": str(Path(bridge.__file__).resolve()),
            "repository_root": str(repository_root()),
            "estimator_executable": estimator["executable"],
            "truth_republish": settings.truth_republish,
            "sensor_mode": settings.sensor_mode.value,
        },
    )


def _run_sensor_derived_live(document: dict[str, Any], output_dir: Path) -> CommandOutcome:
    root = repository_root()
    settings = _platform_settings(document, root)
    localization = document["localization"]
    estimator = localization["estimator"]
    bounds_config = localization["bounds"]
    bounds = loc.HealthBounds(
        publish_period_s=localization["publish"]["period_ms"] / 1000.0,
        state_lost_after_s=bounds_config["state_lost_after_ms"] / 1000.0,
        published_state_age_max_s=bounds_config["published_state_age_max_ms"] / 1000.0,
        max_publish_gap_s=bounds_config["max_publish_gap_ms"] / 1000.0,
        visual_update_warn_s=bounds_config["visual_update_warn_ms"] / 1000.0,
        visual_update_fail_s=bounds_config["visual_update_fail_ms"] / 1000.0,
        valid_fraction_min=bounds_config["valid_fraction_min"],
        sigma_min_m=bounds_config["sigma_min_m"],
        sigma_max_m=bounds_config["sigma_max_m"],
    )
    writer = EvidenceWriter(output_dir, "run-a")
    _write_environment(writer, settings, estimator)
    alignment_origin = _declared_start_origin(settings.world)
    declared_start_rpy = _declared_start_attitude(settings.world)
    alignment = loc.OdomAlignment(alignment_origin, declared_start_rpy)
    machine = loc.HealthMachine(bounds)

    estimator_process = _start_estimator(estimator, root, writer)
    if estimator_process is None:
        _stop = f"the estimator process {estimator['executable']} did not open port {estimator['socket_port']}; see run-a/estimator.log"
        _write_log(writer, [f"UNRESOLVED: {_stop}"])
        return _blocked_unresolved(
            (_stop,),
            (
                "no bound was relaxed and no truth was fed to the estimator; the predeclared "
                "stop rule (plan section 11) records the blocker and stops",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {"stage_id": STAGE_ID, "sensor_mode_label": "sensor-derived", "estimator_pin": estimator},
            (*writer.artifacts, "preflight.json"),
            SensorMode.SENSOR_DERIVED,
        )

    client = loc.OvStreamClient("127.0.0.1", int(estimator["socket_port"]))
    try:
        client.connect()
    except (OSError, loc.ProtocolError) as error:
        _stop = (
            f"the adapter could not connect to the estimator on 127.0.0.1:"
            f"{estimator['socket_port']}: {error}"
        )
        _stop_estimator(estimator_process)
        _write_log(writer, [f"UNRESOLVED: {_stop}"])
        return _blocked_unresolved(
            (_stop,),
            (
                "no bound was relaxed and no truth was fed to the estimator; the predeclared "
                "stop rule (plan section 11) records the blocker and stops",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {"stage_id": STAGE_ID, "sensor_mode_label": "sensor-derived", "estimator_pin": estimator},
            (*writer.artifacts, "preflight.json"),
            SensorMode.SENSOR_DERIVED,
        )
    feed_log_path = writer.path("estimator-feed.jsonl")
    stats = _FeedStats()
    latest_aligned: dict[str, object] | None = None
    # The scored window's publications, kept for E1. Only samples inside arm-to-disarm
    # count: bring-up publishes are reported by the freshness accounting and charged to
    # nothing, exactly as the machine's own window does.
    published_states: list[tuple[int, tuple[float, float, float]]] = []
    # The scored route's own declared windows (each waypoint hold and the drain after the
    # LAND), recorded in the units they are declared in for the same reason the bring-up's
    # are: a hold is the aircraft holding a waypoint, not however much of one a loaded
    # host delivers.
    route_windows: list[_SimWindow] = []
    scored_window_open = False
    # Pixels reach a reader's sink before the handoff queue strips them: the queue
    # carries metadata only, so a stereo pair arrives there as a kind with no planes.
    # The sink therefore keeps whole pair records for the feed, in a bounded queue
    # that this one thread drains, so every write to the estimator socket comes from
    # the drain loop and frames cannot interleave.
    pending_pairs: queue.Queue = queue.Queue(maxsize=PAIR_QUEUE_FRAMES)

    scene_capture = {"count": 0, "last_s": 0.0}

    def file_record(record: Any) -> None:
        if record.kind is not Kind.PAIR or record.pair is None:
            return
        stats.pair_records_filed += 1
        try:
            pending_pairs.put_nowait(record)
        except queue.Full:
            stats.pair_records_dropped += 1
        # The run's own scene capture (plan section 0.3 item 4): bounded pairs at
        # the declared static start, with the world's own pixels — the same
        # conditions as P00's accept-5 capture, written so the arm gate and every
        # later preflight measure the configured world instead of trusting a
        # capture that cannot say which world it shows. Both eyes are written, and
        # from the same pair record the feed converts and sends, so the gate
        # measures exactly the two planes the estimator is given: the captures
        # recorded before this revision held only the left frame, and the gate
        # passed on one eye while nothing recorded what the second one contained.
        now = time.monotonic()
        if (
            scene_capture["count"] < SCENE_CAPTURE_MAX_FRAMES
            and now - scene_capture["last_s"] >= SCENE_CAPTURE_MIN_SPACING_S
        ):
            scene_capture["last_s"] = now
            index = scene_capture["count"] + 1
            header = f"P6\n{settings.stereo.width} {settings.stereo.height}\n255\n".encode()
            writer.write_bytes(
                f"pairs/{index:05d}-left.ppm", header + bytes(record.pair.left_bytes)
            )
            writer.write_bytes(
                f"pairs/{index:05d}-right.ppm", header + bytes(record.pair.right_bytes)
            )
            scene_capture["count"] += 1

    def on_publish(state: loc.EstimatorState, aligned: dict[str, object]) -> None:
        nonlocal latest_aligned
        latest_aligned = aligned
        if scored_window_open:
            published_states.append((state.time_ns, tuple(aligned["position_ned_m"])))
        with feed_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "published_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        **aligned,
                        "n_tracks": state.n_tracks,
                        "t_last_visual_ns": state.t_last_visual_ns,
                        "visual_age_verdict": machine.visual_update_verdict(state),
                    },
                    default=str,
                )
                + "\n"
            )

    publisher = loc.ExternalNavPublisher(
        settings.mavlink_endpoint, alignment, machine, on_publish=on_publish
    )
    session = PymavlinkSession()
    platform = WebotsArduPilot(
        settings,
        runner=SubprocessRunner(),
        session=session,
        gateway=TcpSensorGateway(stamp=lambda: settings.capture_stamp(time.monotonic_ns())),
        evidence=writer,
        label="run-a",
        extra_params=settings.estimator_params,
    )
    # Pixels only ever arrive through the sink: the reader strips them before the
    # handoff queue, which carries metadata so a queue of 614 KB frames cannot grow
    # unbounded. Without this line a stereo pair reaches the feed as a kind with no
    # planes, and the estimator is fed nothing.
    platform.record_sink = file_record
    log_lines: list[str] = [
        f"P01-L sensor-derived live run, started {datetime.now(timezone.utc).isoformat()}",
        f"estimator pin: {estimator['name']} {estimator['tag']} ({estimator['commit']})",
        f"odom origin (the world's own vehicle translation, ENU): {alignment_origin}, "
        f"declared start attitude rpy: {declared_start_rpy}",
        f"parameter layer: {[str(path) for path in settings.estimator_params]}",
    ]

    def feed_record(record: Any) -> None:
        # Every frame carries the simulator's own clock, and every wait this stage
        # declares in simulated seconds is measured against it. Reading it here, on the
        # one path that consumes the whole stream, is what makes the clock the scene's
        # rather than the host's.
        stats.sim_clock.observe(record.sim_time_s)
        if record.kind is Kind.PAIR and record.pair is not None:
            pair = record.pair
            sample = SensorSample(
                value=pair,
                capture_stamp=settings.capture_stamp(pair.capture_host_ns),
                receipt_stamp=record.received_stamp,
                sim_time_s=record.sim_time_s,
            )
            stats.pair_latencies_ns.append(capture_latency_ns(sample))
            left = loc.grayscale_rgb8(
                pair.left_bytes, settings.stereo.width, settings.stereo.height
            )
            right = loc.grayscale_rgb8(
                pair.right_bytes, settings.stereo.width, settings.stereo.height
            )
            client.send(
                loc.encode_stereo(
                    sim_time_ns(record.sim_time_s),
                    left,
                    right,
                    settings.stereo.width,
                    settings.stereo.height,
                )
            )
            stats.pairs += 1
        elif record.kind is Kind.IMU and record.imu is not None:
            imu = record.imu
            sample = SensorSample(
                value=imu,
                capture_stamp=settings.capture_stamp(imu.capture_host_ns),
                receipt_stamp=record.received_stamp,
                sim_time_s=record.sim_time_s,
            )
            stats.imu_latencies_ns.append(capture_latency_ns(sample))
            stamp_ns = sim_time_ns(record.sim_time_s)
            stats.newest_imu_ns = max(stats.newest_imu_ns, stamp_ns)
            client.send(loc.encode_imu(stamp_ns, imu.gyro, imu.accelerometer))
            stats.imu_samples += 1
        elif record.kind is Kind.POSE and record.pose is not None:
            # Evaluator truth, read for scoring and sent nowhere. The estimator's feed
            # handles PAIR and IMU only, so this branch cannot reach it; the sample is
            # already in ArduPilot's NED frame, which is the frame the published state is
            # converted into, so E1 is a subtraction in one common frame.
            stats.truth_samples.append(
                (sim_time_ns(record.sim_time_s), tuple(record.pose.position_xyz))
            )
            stats.truth_attitudes.append(
                (sim_time_ns(record.sim_time_s), tuple(record.pose.attitude_rpy))
            )

    last_telemetry_sample = 0.0
    def drain() -> None:
        """Consume the sensor stream and the estimator's answers without judging.

        The health machine is driven only by the publisher's tick; this loop
        feeds, and stops the machine only when the feed itself fails.
        """
        nonlocal last_telemetry_sample
        while True:
            try:
                record = platform.sensor_record(0.0)
            except ProbeFailure as error:
                machine.stop(time.monotonic_ns(), f"the sensor stream failed: {error}")
                return
            if record is None:
                # The metadata stream is momentarily empty, so the inertial samples
                # either side of a queued image have been fed: the sink's pairs can go
                # now, which keeps every image behind the IMU that must precede it.
                try:
                    while True:
                        feed_record(pending_pairs.get_nowait())
                except queue.Empty:
                    pass
                except loc.ProtocolError as error:
                    machine.stop(time.monotonic_ns(), str(error))
                    return
                try:
                    state = client.poll_state()
                except loc.ProtocolError as error:
                    machine.stop(time.monotonic_ns(), str(error))
                    return
                if state is not None:
                    publisher.offer(state, stats.newest_imu_ns)
                if time.monotonic() - last_telemetry_sample >= TELEMETRY_SAMPLE_PERIOD_S:
                    last_telemetry_sample = time.monotonic()
                    # G4's evidence is the run's own record: folding the telemetry here
                    # records every inbound MAVLink message through the whole window,
                    # not only the windows the readback and arming happen to sample.
                    try:
                        platform.telemetry()
                    except ProbeFailure as error:
                        machine.stop(
                            time.monotonic_ns(), f"the telemetry stream failed: {error}"
                        )
                        return
                return
            try:
                feed_record(record)
            except loc.ProtocolError as error:
                machine.stop(time.monotonic_ns(), str(error))
                return

    disagreements: list[dict[str, Any]] = []

    def sample_disagreement(phase: str) -> None:
        """H3: the estimator's published position against EKF3's, in the common frame."""
        sample = platform.telemetry()
        if sample.local_position_ned is None or latest_aligned is None:
            return
        estimate = latest_aligned["position_ned_m"]
        deltas = [abs(estimate[i] - sample.local_position_ned[i]) for i in range(3)]
        disagreements.append(
            {
                "phase": phase,
                "at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "ekf3_local_position_ned": list(sample.local_position_ned),
                "published_position_ned": [float(value) for value in estimate],
                "abs_delta_m": deltas,
                "norm_m": sum(delta * delta for delta in deltas) ** 0.5,
            }
        )

    live_blockers: list[str] = []
    shutdown: Any = None

    def gate_the_scored_arm(applied: dict[str, float], refusals: dict[str, int]) -> list[str]:
        """Sections 4.6 and 4.7, as observation: what was sent, and what the vehicle reports.

        Every item is read from the things that acted — the bridge's own count of
        simulator poses it has sent, the autopilot's own parameter answers and
        refusals, and the vehicle's own GPS status streams — so none of it is a
        promise about behaviour. A failure here stops the arm before the scored window
        opens, which is the only place it can be stopped without spending a flight.
        """
        blockers: list[str] = []
        truth_published = platform.truth_feed_published
        log_lines.append(
            f"bridge truth poses sent: {truth_published} "
            f"(republish declared {settings.truth_republish})"
        )
        if truth_published:
            blockers.append(
                f"the bridge sent {truth_published} simulator poses before this arm was "
                "scored: the truth republish is not off, so the estimator's input and the "
                "autopilot's external-navigation source would both carry truth (plan "
                "section 4.6)"
            )
        blockers.extend(_readback_blockers(applied, refusals))
        gps = _gps_aiding_verdict(writer.path("mavlink.jsonl"))
        log_lines.append(
            f"GPS-off at the gate: {gps['sys_status_samples']} SYS_STATUS samples, "
            f"{gps['gps_raw_int_samples']} GPS_RAW_INT samples, "
            f"{len(gps['statustexts'])} GPS STATUSTEXTs recorded so far"
        )
        blockers.extend(gps["blockers"])
        return blockers

    bring_up_link: BringUpLink | None = None
    bring_up: dict[str, Any] = {}
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.startup)
        # The adapter's own link: a port the running SITL declares it serves and
        # that this check's session does not already own (plan section 0.5).
        feed_endpoint = _autopilot_feed_endpoint(
            writer.path("sitl.log"), settings.mavlink_endpoint
        )
        publisher.retarget(feed_endpoint)
        log_lines.append(f"adapter publish endpoint: {feed_endpoint}")
        # The ordered bring-up's own link, on a port of its own (the pinned SITL
        # serves one client per serial port): the origin datum, the window's
        # parameter writes and the window's one bounded throttle override are not
        # message types the bridge's session sends, and widening that session's
        # allowlist would widen what the SCORED arm can put on the wire.
        bring_up_endpoint = _autopilot_feed_endpoint(
            writer.path("sitl.log"),
            settings.mavlink_endpoint,
            already_taken=(feed_endpoint,),
        )
        # The link speaks as the vehicle's OWN declared GCS system id, read back
        # here rather than assumed: the firmware ignores an RC channel override
        # from any system id other than its own GCS (`sysid_is_gcs`,
        # GCS_Common.cpp:4216-4220 -> GCS.cpp:727-734), and the window's thrust
        # path is exactly that message. Asking first means the window cannot be a
        # silent void: a vehicle that does not answer this gets no bring-up link
        # and the window is not attempted at all.
        bring_up_system_id = platform.read_parameters(
            (BRING_UP_GCS_SYSTEM_PARAMETER,),
            timeout_s=PARAMETER_READ_TIMEOUT_S,
            drain=drain,
        ).get(BRING_UP_GCS_SYSTEM_PARAMETER)
        if bring_up_system_id is None:
            system_reason = (
                f"the vehicle did not answer {BRING_UP_GCS_SYSTEM_PARAMETER}: the "
                "ordered bring-up's thrust path is an RC channel override, and "
                "handle_rc_channels_override drops one from any system id that is not "
                "the vehicle's own declared GCS (GCS_Common.cpp:4216-4220), so a link "
                "built on a guessed id would be a silent void"
            )
            live_blockers.append(system_reason)
            log_lines.append(f"UNRESOLVED: {system_reason}")
        else:
            bring_up_link = BringUpLink(
                bring_up_endpoint, source_system=int(bring_up_system_id)
            )
            try:
                bring_up_link.connect(timeout_s=BRING_UP_GCS_CONNECT_TIMEOUT_S)
                log_lines.append(
                    f"ordered bring-up endpoint: {bring_up_endpoint} as system id "
                    f"{int(bring_up_system_id)}"
                )
            except (RuntimeError, ProbeFailure) as error:
                live_blockers.append(
                    f"the ordered bring-up's link is unavailable: {error}"
                )
                log_lines.append(f"UNRESOLVED: {error}")
        platform.request_telemetry_streams()
        # G4 needs the vehicle's own GPS status during the run, so the two status
        # streams are requested through the session's existing interval path; every
        # inbound message is recorded by platform.telemetry() either way.
        for message_id in (MSG_ID_SYS_STATUS, MSG_ID_GPS_RAW_INT):
            session.request_message_interval(message_id, GPS_AIDING_SAMPLE_HZ)
        # Read one parameter at a time. The shared reader waits for a whole batch, so a
        # single name the autopilot never answers would consume the deadline and leave
        # the later names unrequested -- a readback that cannot say whether it asked.
        # Asked individually, every name gets its own chance and its own answer or its
        # own absence, which is exactly what the gate records.
        applied: dict[str, float] = {}
        for name, _expected, _source in VEHICLE_REQUIREMENTS:
            applied.update(
                platform.read_parameters((name,), timeout_s=PARAMETER_READ_TIMEOUT_S, drain=drain)
            )
        refusals = _param_error_refusals(writer.path("mavlink.jsonl"))
        writer.write_json("params-applied.json", _params_applied_record(applied, refusals))
        log_lines.append(
            f"autopilot parameter readback: {applied}; refused by the vehicle: {sorted(refusals)}"
        )
        live_blockers.extend(gate_the_scored_arm(applied, refusals))
        try:
            publisher.start()
        except RuntimeError as error:
            # An unserved link: the autopilot never answered on this port, so
            # publishing would be a silent void (plan section 0.5). Named, not
            # ignored -- the third textured invocation's measured failure mode.
            live_blockers.append(str(error))
            log_lines.append(f"UNRESOLVED: {error}")

        scene_blocker = _scene_capture_gate(writer, settings, drain, scene_capture, log_lines)
        if scene_blocker:
            live_blockers.append(scene_blocker)
        # The DECLARED ORDERED BRING-UP (plan sections 0.6 item 6, 0.8 item 7):
        # the origin datum, the bounded exception window with its own flight, and
        # the restoration with the vehicle's own readback. It runs only when
        # everything before it passed, and every blocker it finds stops the run
        # before the claimed arm -- the scored window never opens on a window
        # still in force.
        if live_blockers:
            log_lines.append(
                "ordered bring-up not attempted: a precondition of the scored arm failed"
            )
        else:
            bring_up = _run_ordered_bring_up(
                settings,
                platform,
                session,
                bring_up_link,
                writer,
                drain,
                log_lines,
                stats,
            )
            live_blockers.extend(bring_up["blockers"])
        # H5's wait, in the declared budget's own unit: the estimator's readiness is
        # simulated work (`_wait_initialized`).
        initialized_in_window = _wait_initialized(
            machine,
            drain,
            _SimWindow(
                stats.sim_clock,
                settings.pre_arm_wait_s,
                label="the declared wait for the estimator to initialize (pre_arm_wait_s)",
                wall_ceiling_s=_sim_window_wall_ceiling_s(
                    settings.pre_arm_wait_s, settings.realtime_ratio_envelope[0]
                ),
            ),
        )
        if not initialized_in_window:
            live_blockers.append(
                _initialization_blocker(_initializer_diagnostics(writer.path("estimator.log")))
            )
        else:
            log_lines.append("estimator initialized before arm (H5)")
            a1_blocker = _attitude_gate(writer, latest_aligned, alignment, stats, log_lines)
            if a1_blocker:
                live_blockers.append(a1_blocker)
        if live_blockers:
            # A pre-arm gate failed (sections 4.6, 4.7, or revision 4's scene and
            # attitude gates). The arm stops here, before the scored window opens,
            # so the flight is not spent: nothing is substituted to get past it.
            log_lines.append("not arming: a precondition of the scored arm failed")
        else:
            control = platform.arm_and_guided(settings.step_timeout_s.flight, drain=drain)
            if control.refused:
                live_blockers.append(
                    f"the autopilot refused Guided flight with GPS off: mode_reached="
                    f"{control.mode_reached}, armed={control.armed}, refusals="
                    f"{list(control.refusals)}; that is a blocker under plan section 11, "
                    "not a reason to re-enable GPS"
                )
            else:
                machine.open_window(time.monotonic_ns())
                scored_window_open = True
                lost_guided = False
                for index, waypoint in enumerate(settings.waypoints_local_ned, start=1):
                    if lost_guided:
                        break
                    # The configuration's own contract: "waypoints are local-NED
                    # offsets from the takeoff point; z is added to the hover
                    # altitude" (configs/first_indoor.yaml probe.waypoints_local_ned).
                    # The compatibility gate's own guided item subtracts the hover
                    # altitude the same way (webots_ardupilot.py:5103-5107). Sending
                    # z unchanged (measured, run p01l-bringup-20260926T204753Z:
                    # commanded local-NED (2.0, 0.0, 0.0) while the vehicle was at
                    # z=-1.69 mid-climb) targets the GROUND: a 1.5 m descent command
                    # racing the takeoff it interrupted.
                    target = (
                        float(waypoint[0]),
                        float(waypoint[1]),
                        float(waypoint[2]) - settings.hover_altitude_m,
                    )
                    log_lines.append(
                        f"waypoint {index} commanded at local-NED {target} "
                        f"(config offset {tuple(float(v) for v in waypoint)}, "
                        f"hover altitude {settings.hover_altitude_m})"
                    )
                    # The hold is a declared window of the AIRCRAFT's time (the
                    # configuration's own words for these windows: "a window measured in
                    # simulated time is comparable between the realtime and fast modes"),
                    # so an 8 s hold is 8 s of the vehicle holding the waypoint, not
                    # however much of it a loaded host happens to deliver. The 50 ms
                    # re-send cadence below stays on the wall clock: it paces the link,
                    # not the aircraft.
                    hold_window = _SimWindow(
                        stats.sim_clock,
                        settings.hold_per_waypoint_s,
                        label=f"the hold at waypoint {index}",
                        wall_ceiling_s=_sim_window_wall_ceiling_s(
                            settings.hold_per_waypoint_s,
                            settings.realtime_ratio_envelope[0],
                        ),
                    )
                    route_windows.append(hold_window)
                    # The estimator's published pose can only be as current as
                    # the newest inertial sample it has CONSUMED, and it consumes
                    # what this loop feeds it. Measured on run
                    # p01l-fix2-20260927T020445Z: draining once per 50 ms left
                    # 1416 of 1767 publications carrying an unchanged state
                    # timestamp (the pose stale by the burst gap, F2 max
                    # 0.232 s), because the estimator saw inertial data only in
                    # 50 ms bursts. Draining every 5 ms keeps its newest
                    # consumed sample within ~one Webots step of "now", while
                    # the guided target itself keeps its proven 50 ms re-send
                    # cadence (webots_ardupilot.py's own probe re-sends).
                    next_target_send = 0.0
                    while not hold_window.expired():
                        now = time.monotonic()
                        if now >= next_target_send:
                            # Keep the stream alive while holding: a guided target lapses
                            # inside the autopilot, so one publication per waypoint would
                            # hold nothing. The deadline is how long each sample stays
                            # valid, and the platform's own probe re-sends for exactly
                            # this reason (webots_ardupilot.py:5109-5120). The yaw is
                            # the declared route heading (ROUTE_YAW_HOLD_RAD above):
                            # yaw-ignored targets would hand the heading to the
                            # firmware's velocity-alignment behavior, which slews the
                            # vehicle into the end-of-route tumble measured above.
                            next_target_send = now + 0.05
                            sent = platform.send_local_ned(
                                LocalNedTarget(
                                    position_ned=target,
                                    velocity_ned=(0.0, 0.0, 0.0),
                                    yaw_rad=ROUTE_YAW_HOLD_RAD,
                                    deadline_s=settings.hold_per_waypoint_s,
                                    certificate_ref=None,
                                )
                            )
                            if sent is None:
                                live_blockers.append(
                                    f"Guided flight was lost while holding waypoint {index} at "
                                    f"local-NED {target}"
                                )
                                lost_guided = True
                                break
                        drain()
                        time.sleep(0.005)
                    sample_disagreement(f"hold-{index}")
                log_lines.append("route complete; commanding LAND")
                session.set_mode("LAND")
                # The FLIGHT ends here, at the LAND command. The freshness metrics --
                # F1's publish gaps and F2's published-state age -- stop being recorded,
                # because the bound protects the control loop WHILE FLYING and this is
                # the moment flying stops. Owner's ruling: takeoff and landing are
                # low-speed phases where pose age does not matter. The takeoff is
                # already outside the window (it opens after arm_and_guided returns).
                # The excluded span is real but is not flight: measured, run
                # p01l-fix7a-20260927T172100Z, the 27 publications after this point
                # carried ONE frozen state (t=53.940) for 274 ms while the vehicle sat
                # parked and disarmed -- the adapter running out its own declared 300 ms
                # silence watchdog -- and those 27 rows were the entirety of F2's
                # 0.056 s max. In flight the same run's F2 max was 0.021 s. E1/H1/H2 keep
                # the declared arm-to-disarm window untouched: only the freshness
                # accounting ends at the flight's end, so the descent and landing are
                # still judged for accuracy.
                machine.mark_flight_end(time.monotonic_ns())
                # The post-route drain, in the same seconds as the holds it follows: its
                # job is to keep feeding while the LAND runs, and the LAND is the
                # aircraft coming down in the aircraft's time.
                landing_drain = _SimWindow(
                    stats.sim_clock,
                    5.0,
                    label="the post-route drain after the LAND command",
                    wall_ceiling_s=_sim_window_wall_ceiling_s(
                        5.0, settings.realtime_ratio_envelope[0]
                    ),
                )
                while not landing_drain.expired():
                    drain()
                    time.sleep(0.005)
                route_windows.append(landing_drain)
                # The scored window is declared arm to disarm, and the vehicle has
                # disarmed by the time the sequence is over; everything after this is
                # the harness stopping Webots and SITL. Measured, run
                # p01l-zupt5-20260927T042005Z: the adapter republished one frozen state
                # 332 times over 10.13 s of that shutdown -- 32 % of the window, every
                # one of them scored against a truth sample that was equally frozen --
                # so E1's p95 came out exactly equal to its max, the signature that was
                # read as a constant bias.
                scored_window_open = False
                machine.close_window(time.monotonic_ns())
    except Exception as error:  # noqa: BLE001 - the run's outer guard records, never swallows
        live_blockers.append(f"the platform failed during the run: {error}")
    finally:
        shutdown = platform.stop()
        publisher.stop()
        client.close()
        _stop_estimator(estimator_process)
        if bring_up_link is not None:
            bring_up_link.close()

    _write_health_events(writer, machine)
    gps_aiding = _gps_aiding_verdict(writer.path("mavlink.jsonl"))
    writer.write_json("gps-aiding.json", gps_aiding)
    writer.write_json(
        "initializer-diagnostics.json",
        _initializer_diagnostics(writer.path("estimator.log")),
    )
    valid_fraction = machine.valid_fraction(time.monotonic_ns())
    truth_published = platform.truth_feed_published
    bring_up_receipt = _bring_up_manifest(bring_up)
    log_lines.extend(
        [
            f"pairs fed: {stats.pairs}, imu samples fed: {stats.imu_samples}, "
            f"truth pose samples read: {len(stats.truth_samples)}",
            f"pair records the reader filed with pixels: {stats.pair_records_filed}, "
            f"dropped by a full feed queue: {stats.pair_records_dropped}",
            f"published: {publisher.published}, health at the scored window's end: "
            f"{machine.state_at_close or machine.state} (at process exit {machine.state}), "
            f"valid fraction: {valid_fraction:.4f}, adapter resets: {machine.reset_counter}",
            f"bridge truth poses sent over the whole run: {truth_published}",
            f"scored-window publications compared against truth: {len(published_states)}",
            f"ordered bring-up: attempted={bring_up_receipt['attempted']}, "
            f"completed={bring_up_receipt.get('completed')}, "
            f"closure_blockers={bring_up_receipt.get('closure_blockers')}",
            f"shutdown: {shutdown.exits}",
        ]
    )

    if live_blockers:
        for blocker in live_blockers:
            log_lines.append(f"UNRESOLVED: {blocker}")
        _write_log(writer, log_lines)
        return _blocked_unresolved(
            tuple(live_blockers),
            (
                "no bound was relaxed and no truth was fed to the estimator; the predeclared "
                "stop rule (plan section 11) records the blocker and stops",
                "the declared ordered bring-up (plan sections 0.6 item 6, 0.8 item 7) is "
                "recorded in bring-up.json, including whether it ran and whether its "
                "exception window was restored",
                DISPATCH_REGISTRATION_NOTE,
            ),
            {
                "stage_id": STAGE_ID,
                "sensor_mode_label": "sensor-derived",
                "estimator_pin": estimator,
                "bring_up": bring_up_receipt,
                "truth_republish": settings.truth_republish,
                "bridge_truth_published": truth_published,
                "gps_aiding_blockers": gps_aiding["blockers"],
                "pairs_filed": stats.pair_records_filed,
                "pairs_fed": stats.pairs,
                "imu_samples_fed": stats.imu_samples,
                "published": publisher.published,
                "shutdown": {"exits": shutdown.exits},
            },
            (*writer.artifacts, "preflight.json"),
            SensorMode.SENSOR_DERIVED,
        )

    truth_comparison = _truth_error_statistics(
        published_states, stats.truth_samples, bounds_config
    )
    writer.write_json("truth-comparison.json", truth_comparison)
    disagreement_summary = _summarise_disagreements(
        disagreements,
        bounds_config["disagreement_p95_m"],
        bounds_config["disagreement_max_m"],
    )
    writer.write_json("disagreement.json", disagreement_summary)
    checks = _score(
        machine,
        bounds,
        bounds_config,
        valid_fraction,
        truth_comparison,
        disagreement_summary,
        truth_published,
        gps_aiding,
    )
    writer.write_json("checks.json", checks)
    _write_log(writer, log_lines)
    passed = all(check["status"] == "pass" for check in checks)
    manifest = {
        "stage_id": STAGE_ID,
        "sensor_mode_label": "sensor-derived",
        "localization": "resolved" if passed else "unresolved",
        "estimator_pin": estimator,
        "bounds": bounds_config,
        "bring_up": bring_up_receipt,
        "truth_republish": settings.truth_republish,
        "bridge_truth_published": truth_published,
        "truth_samples_read": len(stats.truth_samples),
        "pairs_filed": stats.pair_records_filed,
        "pairs_fed": stats.pairs,
        "imu_samples_fed": stats.imu_samples,
        "published": publisher.published,
        "scored_publications": len(published_states),
        "reset_counter": machine.reset_counter,
        "valid_fraction": valid_fraction,
        "params_applied": applied,
        # The scored route's declared windows in the units they are declared in, so the
        # receipt says whether an 8 s hold was 8 s of the aircraft holding the waypoint.
        "route_windows": [opened.document() for opened in route_windows],
        "shutdown": {"exits": shutdown.exits},
    }
    return CommandOutcome(
        status=CommandStatus.COMPLETE,
        gate_status=GateStatus.PASS if passed else GateStatus.FAIL,
        reasons=tuple(
            f"{check['name']}: {check['status']} — {check['detail']}"
            for check in checks
            if check["status"] != "pass"
        )
        or ("all predeclared bounds met on the frozen route",),
        limitations=(
            "E1 compares the scored window's publications against the controller's pose "
            "stream, which is read in this process for scoring only and sent to nothing: "
            "the estimator's feed carries stereo pairs and inertial samples and no other "
            "record. A publication with no truth sample inside the join tolerance is "
            "counted as unjoined rather than interpolated",
            "the claimed arm was reached through a DECLARED ordered bring-up: a bounded "
            "exception window (plan sections 0.6 item 6, 0.8 item 7) whose elements, "
            "citations, flight and restoration readbacks are in bring-up.json. The "
            "exception changed only WHEN the aircraft could move -- and the aircraft's own "
            "height reference during the window was its barometer, because an external-nav "
            "height source does not exist before the estimator latches. It was lifted, with "
            "the vehicle's own readback as proof, before the scored window opened. The "
            "estimator's input was stereo and inertial throughout, the bridge's truth "
            "republish stayed off, and no predeclared bound was relaxed",
            "a passing E1 is bounded by this route and this scene: two waypoints with 8 s "
            "holds indoors, never a general navigation claim",
            "reset-signalling agreement (H2) records the adapter's reset counter; the "
            "firmware's posReset count rides the SITL log and is compared at integration",
            "per-camera exposure offsets are not modelled; both eyes share the capture "
            "instant of the step in which they were read",
            "the floor-plane metric-depth gap P01-C recorded at grazing incidence is not "
            "repaired by this run and its vertical channel does not inherit that honesty",
            DISPATCH_REGISTRATION_NOTE,
        ),
        manifest=manifest,
        artifacts=(*writer.artifacts, "preflight.json"),
        sensor_mode=SensorMode.SENSOR_DERIVED,
    )


def _run_ordered_bring_up(
    settings: PlatformSettings,
    platform: WebotsArduPilot,
    session: PymavlinkSession,
    link: BringUpLink,
    writer: EvidenceWriter,
    drain: Callable[[], None],
    log_lines: list[str],
    stats: _FeedStats,
) -> dict[str, Any]:
    """The declared ordered bring-up: datum, exception window, excitation, restoration.

    The order is the point (plan section 0.6 item 6). The origin DATUM is declared
    first, because without it the aircraft has no home and the claimed arm would
    be refused however well the estimator did. Then the exception window's own
    parameter writes, each verified by the vehicle's readback before anything
    moves. Then the window's ONE thrust path -- a bounded RC throttle override,
    after the arm and before the takeoff command, so the position-free mode has a
    pilot climb rate to leave the ground on -- and the bounded excitation:
    ALT_HOLD, the flagged takeoff, the climb inside E-EXC's envelope, LAND always.
    Then the override's release, the restoration of every parameter, both verified
    by the same vehicle readbacks, and the closure check that refuses the scored
    arm while any window element is still in force.

    Nothing here publishes to the autopilot's external-navigation source and
    nothing here supplies a pose: the window's only outputs are a frame datum, a
    bounded set of parameter writes with their restores, one bounded RC throttle
    override with its release, and one bounded climb. The estimator's feed, and
    the adapter that publishes its state, are the ones the scored arm already had.

    Returns the window's record. ``record["blockers"]`` carries anything that
    stopped it; the caller must not open the scored window with a non-empty list.
    """
    declaration = _bring_up_window(settings)
    names = tuple(name for name, _window, _restore, _why in BRING_UP_WINDOW_PARAMETERS)
    record: dict[str, Any] = {
        "declaration": declaration,
        "sent": [],
        "replies": [],
        "readbacks": {},
        "readback_observations": {},
        "flight": {},
        "blockers": [],
        "completed": False,
    }
    blockers: list[str] = record["blockers"]

    # Every window below is declared in SIMULATOR seconds, and this is the clock that
    # spends them: the scene's own, read on every frame the feed consumes. The wall
    # clock keeps only two jobs here -- the ceiling `_sim_window_wall_ceiling_s`
    # derives from the declared realtime envelope, which ends a window whose simulated
    # budget was not spent (the simulator is then outside the envelope, which the
    # probe's own realtime evidence already reports), and the pacing of the loops.
    envelope_floor = settings.realtime_ratio_envelope[0]
    windows: list[_SimWindow] = []

    def window(budget_s: float, label: str) -> _SimWindow:
        opened = _SimWindow(
            stats.sim_clock,
            budget_s,
            label=label,
            wall_ceiling_s=_sim_window_wall_ceiling_s(budget_s, envelope_floor),
        )
        windows.append(opened)
        return opened

    record["window_clock"] = {
        "source": SIMULATOR_CLOCK_SOURCE,
        "frames_carrying_a_simulator_time_at_the_window_open": stats.sim_clock.frames,
        "declared_realtime_envelope": list(settings.realtime_ratio_envelope),
        "wall_ceiling_rule": (
            "budget_simulator_s / the envelope's floor: inside the declared envelope a "
            "budget of N simulated seconds cannot take longer than N/floor wall seconds"
        ),
    }
    # The window's ONE thrust path, recorded as what the VEHICLE answered rather
    # than as what this link sent: the values the derivation was made of, the PWM
    # that went out, what the vehicle's own RC report showed while the override was
    # in force, and what it showed after the release. The closure gate reads this
    # object and nothing else about the override.
    override: dict[str, Any] = {
        "declared_climb_rate_ms": BRING_UP_THROTTLE_CLIMB_RATE_M_S,
        "channel": RC_THROTTLE_CHANNEL,
        "channel_name": "throttle",
        "refresh_s": BRING_UP_OVERRIDE_REFRESH_S,
        "calibration": {},
        "calibration_observations": [],
        "gcs_system_id": None,
        "link_source_system": None,
        "sent_pwm": None,
        "measured_climb_rate_ms": None,
        "sent": False,
        "sent_at_utc": None,
        "refreshes": 0,
        # The override's refresh cadence is paced on the simulator's clock, because its
        # declared bound is the firmware's own RC_OVERRIDE_TIME of 3.0 s of the
        # vehicle's time.
        "last_refresh_simulator_s": None,
        "observed_during_window": None,
        "observed_after_release": None,
        "released": False,
        "released_at_utc": None,
        "confirmed_at_utc": None,
    }
    record["throttle_override"] = override
    log_lines.append(
        "ordered bring-up (plan sections 0.6 item 6, 0.8 item 7): "
        f"{BRING_UP_MODE} excitation to {EXCITATION_TAKEOFF_ALTITUDE_M} m, no lateral "
        f"setpoint, LAND within {EXCITATION_MAX_AIRTIME_S} s of the arm readback; "
        "exception window = "
        + ", ".join(
            f"{name} {window_value:g} -> {restore_value:g}"
            for name, window_value, restore_value, _why in BRING_UP_WINDOW_PARAMETERS
        )
    )
    log_lines.append(
        "ordered bring-up clock: every declared window below is spent in SIMULATOR "
        f"seconds against {SIMULATOR_CLOCK_SOURCE}; the wall ceiling on each is its "
        f"budget / the declared realtime envelope's floor "
        f"({settings.realtime_ratio_envelope[0]:g})"
    )

    # 1. The origin datum: a frame definition, set once, before anything moves.
    datum = declaration["origin_datum"]
    record["sent"].append(
        link.declare_origin_datum(
            datum["latitude_deg"], datum["longitude_deg"], datum["altitude_msl_m"]
        )
    )
    echo_deadline = time.monotonic() + BRING_UP_ORIGIN_ECHO_TIMEOUT_S
    echoed: dict[str, Any] | None = None
    while time.monotonic() < echo_deadline and echoed is None:
        drain()
        link.drain()
        echoed = next(
            (reply for reply in link.replies if reply.get("mavpackettype") == "GPS_GLOBAL_ORIGIN"),
            None,
        )
        time.sleep(0.05)
    record["origin_datum_echo"] = echoed
    if echoed is None:
        blockers.append(
            "the vehicle did not emit GPS_GLOBAL_ORIGIN after the origin datum was "
            f"declared, within {BRING_UP_ORIGIN_ECHO_TIMEOUT_S:.0f} s: "
            "set_ekf_origin sends MSG_ORIGIN on acceptance and returns before it when an "
            "origin already exists (GCS_Common.cpp:3961-3979), so a silent link is an "
            "undeclared origin -- and with GPS off, home is derived from that origin "
            "(Copter::update_home_from_EKF, commands.cpp:4-20). The window does not fly "
            "on an undeclared datum"
        )

    def readback(
        names: Sequence[str], expected: dict[str, float]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Ask the vehicle for each name and wait until it answers with ``expected``.

        ``WebotsArduPilot.read_parameters`` returns the newest value the run has
        seen, so a value seen before the request satisfies it: the second
        invocation's restoration readback was answered from the value the window
        write had left there, and reported a restored window as still in force.
        This asks, then polls until the expected value arrives or the deadline
        passes, recording every distinct answer so the receipt shows the
        transition rather than the last cached word.
        """
        reported: dict[str, float] = {}
        observations: list[dict[str, Any]] = []
        for name in names:
            session.request_parameter(name)
        deadline = time.monotonic() + PARAMETER_READ_TIMEOUT_S
        while True:
            drain()
            link.drain()
            reported.update(platform.telemetry().parameters)
            snapshot = {name: reported.get(name) for name in names}
            if not observations or observations[-1]["values"] != snapshot:
                observations.append(
                    {
                        "at_utc": datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        ),
                        "values": snapshot,
                    }
                )
            if all(reported.get(name) == expected[name] for name in names):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        return reported, observations

    def write_parameters(values: dict[str, float]) -> None:
        for name in names:
            record["sent"].append(link.set_parameter(name, values[name]))

    def read_vehicle_answers(
        wanted: Sequence[str],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Ask the vehicle for each name and wait for its own answers, then stop.

        The same discipline as the window's parameter readback, and for the same
        reason: the numbers the throttle derivation is made of are this vehicle's
        own calibration, so they are asked for now and read from the vehicle's own
        PARAM_VALUE answers. A name the vehicle does not answer stays absent, is
        recorded as absent, and is named by the caller -- an assumed RC calibration
        is the same class of defect as a parameter write under a name the vehicle
        does not have.
        """
        reported: dict[str, float] = {}
        observations: list[dict[str, Any]] = []
        for name in wanted:
            session.request_parameter(name)
        deadline = time.monotonic() + PARAMETER_READ_TIMEOUT_S
        while True:
            drain()
            link.drain()
            reported.update(platform.telemetry().parameters)
            snapshot = {name: reported.get(name) for name in wanted}
            if not observations or observations[-1]["values"] != snapshot:
                observations.append(
                    {
                        "at_utc": datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        ),
                        "values": snapshot,
                    }
                )
            if all(name in reported for name in wanted) or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        return reported, observations

    def refresh_override() -> None:
        """Keep the window's throttle override alive while the window is open.

        The firmware's own override timeout is 3.0 s (``RC_OVERRIDE_TIME``,
        RC_Channels_VarInfo.h:90), so an override that stops being refreshed is
        lapsed by the vehicle itself. The window re-sends it well inside that, and
        only while the window is open: ``released`` ends the sequence, and nothing
        in this program sends another.

        The cadence is measured on the simulator's clock, because the timeout it has
        to stay inside is the firmware's own 3.0 s of the vehicle's time. A wall-clock
        cadence would refresh more often than necessary on a slow host and less often
        than declared on a fast one; this is the same domain as the timeout it is
        declared against.
        """
        if not override["sent"] or override["released"] or override["sent_pwm"] is None:
            return
        now = stats.sim_clock.newest_s
        last = override["last_refresh_simulator_s"]
        if now is not None and last is not None and now - last < BRING_UP_OVERRIDE_REFRESH_S:
            return
        # With no simulator reading yet the cadence cannot be measured, and the vehicle
        # lapses an unrefreshed override on its own clock, which is stalled with the
        # simulator: refreshing is the safe side of an unmeasurable cadence.
        if now is not None:
            override["last_refresh_simulator_s"] = now
        record["sent"].append(link.send_rc_channels_override(int(override["sent_pwm"])))
        override["refreshes"] += 1

    def note_vehicle_rc_report() -> int | None:
        """File the vehicle's own latest report of its throttle RC input.

        ``RC_CHANNELS.chan3_raw`` is ``rc().get_radio_in()`` for the throttle
        channel (GCS_Common.cpp:2172-2205), which is the OVERRIDDEN value while an
        override is in force (RC_Channel.cpp:303-311), so this is the vehicle's own
        statement about the window's thrust path: what it reads while the window is
        open, and what it reads after the window releases it.
        """
        observed = link.latest_rc_throttle_raw()
        if observed is not None:
            if override["released"]:
                override["observed_after_release"] = observed
            else:
                override["observed_during_window"] = observed
        return observed

    window_values = {name: window_value for name, window_value, _r, _w in BRING_UP_WINDOW_PARAMETERS}
    restore_values = {name: restore_value for name, _w, restore_value, _r in BRING_UP_WINDOW_PARAMETERS}

    # 2. The window's own parameter writes, read back from the vehicle itself.
    if not blockers:
        record["window_opened_at_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        write_parameters(window_values)
        window_readback, observations = readback(names, window_values)
        record["readbacks"]["window"] = {name: window_readback.get(name) for name in names}
        record["readback_observations"] = {"window": observations}
        for name, window_value, _restore, _why in BRING_UP_WINDOW_PARAMETERS:
            if window_readback.get(name) != window_value:
                blockers.append(
                    f"the bring-up window's {name} did not take effect: the vehicle read "
                    f"back {window_readback.get(name)} against the declared window value "
                    f"{window_value:g}, so the aircraft would move under checks this run "
                    "has not declared"
                )

    # 2b. The window's ONE thrust path, decided from the vehicle's own numbers
    # BEFORE anything moves: the RC throttle calibration the PWM is derived from,
    # and the GCS system id the firmware requires an override to come from.
    if not blockers:
        answers, observations = read_vehicle_answers(
            (*BRING_UP_THROTTLE_CALIBRATION, BRING_UP_GCS_SYSTEM_PARAMETER)
        )
        override["calibration_observations"] = observations
        override["calibration"] = {
            name: answers.get(name) for name in BRING_UP_THROTTLE_CALIBRATION
        }
        override["gcs_system_id"] = answers.get(BRING_UP_GCS_SYSTEM_PARAMETER)
        override["link_source_system"] = link.source_system
        silent = [
            name
            for name in (*BRING_UP_THROTTLE_CALIBRATION, BRING_UP_GCS_SYSTEM_PARAMETER)
            if name not in answers
        ]
        if silent:
            blockers.append(
                f"the vehicle did not answer {silent}: the window's throttle override "
                "is derived from the vehicle's own RC calibration and is only accepted "
                "from its own declared GCS system id, so a window that has neither "
                "would send a value that means something else, or a value the vehicle "
                "ignores (sysid_is_gcs, GCS_Common.cpp:4216-4220)"
            )
        elif override["link_source_system"] != int(override["gcs_system_id"]):
            blockers.append(
                f"the bring-up link speaks as system id {override['link_source_system']} "
                f"and this vehicle's own {BRING_UP_GCS_SYSTEM_PARAMETER} is "
                f"{int(override['gcs_system_id'])}: handle_rc_channels_override drops an "
                "override from any system id that is not the vehicle's GCS "
                "(GCS_Common.cpp:4216-4220 -> sysid_is_gcs, GCS.cpp:727-734), so the "
                "window's thrust path would be a silent void"
            )
        else:
            try:
                pwm, measured_rate_ms = _bring_up_throttle_pwm(
                    {name: answers[name] for name in BRING_UP_THROTTLE_CALIBRATION},
                    BRING_UP_THROTTLE_CLIMB_RATE_M_S,
                )
            except ConfigError as error:
                blockers.append(
                    "the bring-up window's throttle override cannot be expressed on "
                    f"this vehicle: {error}"
                )
            else:
                override["sent_pwm"] = pwm
                override["measured_climb_rate_ms"] = round(measured_rate_ms, 4)
                if not (0.0 < measured_rate_ms <= BRING_UP_THROTTLE_MAX_CLIMB_RATE_M_S):
                    blockers.append(
                        f"the derived throttle value {pwm} us yields a pilot climb rate "
                        f"of {measured_rate_ms:.3f} m/s from the vehicle's own "
                        f"calibration, outside the declared band (0, "
                        f"{BRING_UP_THROTTLE_MAX_CLIMB_RATE_M_S:g}] m/s: the window "
                        "would either not leave the ground or climb faster than its "
                        "declared envelope allows"
                    )
        record["sent"].append(
            link.request_message_interval(MSG_ID_RC_CHANNELS, BRING_UP_RC_REPORT_HZ)
        )
        log_lines.append(
            "bring-up thrust path: "
            + (
                f"channel {RC_THROTTLE_CHANNEL} override {override['sent_pwm']} us for a "
                f"declared pilot climb rate of {override['measured_climb_rate_ms']} m/s "
                f"(derived from {override['calibration']}, "
                f"GCS system {override['gcs_system_id']}), refreshed every "
                f"{BRING_UP_OVERRIDE_REFRESH_S} s and released at LAND"
                if override["sent_pwm"] is not None
                else "no override (see the blockers below)"
            )
        )

    flight: dict[str, Any] = record["flight"]
    # The window's own motor evidence: the airframe's PWM outputs as the vehicle
    # reports them, and -- because that is the quantity the takeoff command below
    # waits on -- the floor those outputs sit at while no spool-up has been
    # requested. A takeoff the autopilot accepted but the motors never answered is
    # a different finding from a takeoff it refused, and only the outputs say which
    # happened.
    motors: dict[str, Any] = {
        "max_pwm": 0,
        "samples": 0,
        "floor_pwm": None,
        "floor_at_arm_readback": None,
    }
    if not blockers:
        # 3. The window's own arm: ALT_HOLD, retried, every refusal kept.
        record["arm_attempts"] = 0
        refusals: dict[str, str] = {}
        # A simulated autopilot needs simulated seconds for its pre-arm checks to clear
        # (the configuration's own words for this budget: "a simulated GPS needs a fix,
        # the EKF needs a home, and the IMU consistency check needs a quiet window, all
        # of which take simulated time"), so the budget is spent on the simulator's
        # clock. The retry cadence stays on the wall clock: it paces how often the
        # request is repeated, which is a property of the loop and not of the aircraft.
        arm_window = window(
            settings.pre_arm_wait_s,
            "the bring-up's wait for the vehicle's own pre-arm checks to clear",
        )
        sample = platform.telemetry()
        while True:
            record["arm_attempts"] += 1
            session.set_mode(BRING_UP_MODE)
            session.arm()
            settle_until = time.monotonic() + BRING_UP_ARM_SETTLE_S
            while time.monotonic() < settle_until:
                drain()
                time.sleep(0.05)
            sample = platform.telemetry()
            for text in sample.statustexts:
                if text.startswith("PreArm:") or text.startswith("Arm:"):
                    refusals[text] = text
            if sample.armed:
                arm_window.close("vehicle_armed")
                break
            if arm_window.expired():
                break
            retry_until = time.monotonic() + BRING_UP_ARM_RETRY_S
            while time.monotonic() < retry_until and not arm_window.expired():
                drain()
                time.sleep(0.25)
        record["refusals"] = sorted(refusals)
        if not sample.armed:
            blockers.append(
                f"the bring-up's own arm was refused in {BRING_UP_MODE}: mode="
                f"{sample.mode_name}, armed={sample.armed}, attempts="
                f"{record['arm_attempts']}, refusals={sorted(refusals)}. Without the arm "
                "there is no motion, the estimator still cannot latch, and the run stops "
                "here rather than spending a flight it cannot answer"
            )

        max_altitude_m = 0.0

        def sample_motion() -> None:
            """One telemetry read: the altitude readback and the motor outputs."""
            nonlocal sample, max_altitude_m
            sample = platform.telemetry()
            if sample.local_position_ned is not None:
                max_altitude_m = max(max_altitude_m, -sample.local_position_ned[2])
            if sample.servo_outputs is not None:
                motors["samples"] += 1
                motors["max_pwm"] = max(motors["max_pwm"], max(sample.servo_outputs[:4]))

        # The airtime budget the declaration bounds ("<= 5.0 s from the arm readback to
        # the LAND command") and the wall clock's reading of the same interval. The bound
        # is judged against the first: it is a quantity of the aircraft's own time, and on
        # a loaded host the wall reading of it says more about the host than the flight.
        airtime = window(
            EXCITATION_MAX_AIRTIME_S,
            "the excitation's declared airtime, arm readback to LAND",
        )
        arm_monotonic = time.monotonic()
        flight["armed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        flight["simulator_time_at_arm_s"] = stats.sim_clock.newest_s
        flight["mode"] = sample.mode_name
        flight["pairs_fed_at_arm"] = stats.pairs
        flight["imu_fed_at_arm"] = stats.imu_samples
        # The airframe's own PWM floor, as the vehicle reports it while the thrust
        # path is still off: a motor output sits at the airframe's own minimum until
        # a spool state has been asked for and accepted (`output_to_pwm`,
        # AP_MotorsMulticopter.cpp:439-450, SHUT_DOWN). "The motors left the floor"
        # below is therefore a comparison between two of the VEHICLE's own readings,
        # and this window supplies no PWM value of its own for it.
        if sample.servo_outputs is not None:
            motors["floor_pwm"] = max(sample.servo_outputs[:4])
            motors["floor_at_arm_readback"] = motors["floor_pwm"]
        # The window's ONE thrust path goes on here, and the order is the point:
        #
        #   * AFTER the arm, because `arm_checks` refuses to arm with a positive
        #     pilot climb rate -- "Throttle too high" (AP_Arming_Copter.cpp:621-635)
        #     -- and the arm loop above retries, so an override in force during a
        #     retry would refuse every attempt;
        #   * and the takeoff command it feeds is held until the vehicle's own report
        #     of its motor outputs shows the thrust path open (see the block below the
        #     settle). Commanding the takeoff earlier loses the excitation entirely,
        #     and that is the measured defect this ordering fixes.
        if override["sent_pwm"] is not None:
            record["sent"].append(
                link.send_rc_channels_override(int(override["sent_pwm"]))
            )
            override["sent"] = True
            override["sent_at_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
            override_settle = window(
                BRING_UP_OVERRIDE_SETTLE_S,
                "the override settle before the takeoff command",
            )
            while not override_settle.expired():
                drain()
                link.drain()
                note_vehicle_rc_report()
                time.sleep(0.05)
        # THE ORDERING, and why the takeoff command cannot be sent above.
        #
        # The takeoff command is what pins the mode into AltHoldModeState::Takeoff, and
        # that state asks for no spool state at all: `get_alt_hold_state_D_ms`
        # (mode.cpp:1030-1068) returns Takeoff on its second test -- `takeoff.running()
        # || takeoff.triggered_ms(...)` -- BEFORE the landed branch at :1041-1055 that
        # asks for THROTTLE_UNLIMITED, and `AltHold::run()`'s Takeoff case
        # (mode_althold.cpp:63-74) asks for none either. The desired spool state is a
        # latch, so the takeoff freezes whichever state was last accepted.
        #
        # Commanded one settle after the arm, that latch is SHUT_DOWN, and the run
        # therefore produces no thrust at all. The motors library FORCES SHUT_DOWN while
        # `!get_interlock()` (`set_desired_spool_state` and `output_logic`,
        # AP_MotorsMulticopter.cpp:619-638), and Copter holds the interlock down for its
        # own 2.0 s `ap.in_arming_delay` after arming (motors.cpp:59,75 -- the
        # MOTORS_INTERLOCK_ENABLED event). The 0.3 s settle is spent entirely inside that
        # delay, the THROTTLE_UNLIMITED the landed branch asks for is discarded, and the
        # takeoff then pins the latch: nothing in ALT_HOLD asks again for as long as
        # `takeoff.running()`.
        #
        # Measured in this window's own airframe log (work/ardupilot/logs/00000142.BIN):
        # CTUN.ThO (= motors->get_throttle(), Log.cpp:58) ramped to 1.000 as the
        # takeoff's own slew, while MOTB.ThrOut (= _throttle_out) stayed 0.000 and RCOU
        # stayed 1000 us for the whole 4.1 s; the run's first SPOL entry is
        # (Spl=0, SplDes=2) at the LAND mode change, 16 ms before the motors finally left
        # the floor. The counter-case is in the same file: the GUIDED flight of
        # 00000139.BIN asks for THROTTLE_UNLIMITED on every iteration, and SPOL records
        # it as accepted at exactly arm+2.001 s -- the interlock, not the request, is
        # what both flights waited on.
        #
        # So the takeoff waits for the vehicle's OWN report that the thrust path is open:
        # the motors above the floor they sat at before the override went on. The wait is
        # bounded by the excitation's own climb window, which therefore opens here rather
        # than at the command, and no declared budget changes -- the settle, the climb
        # window and the airtime bound are the declared ones, and the wait costs the climb
        # window part of its own 3.5 s instead of the airtime bound the LAND command must
        # stay inside.
        climb_window = window(EXCITATION_CLIMB_DRAIN_S, "the excitation's climb drain")

        def poll_while_waiting() -> Any:
            """One iteration of the wait: feed, keep the override alive, read the vehicle."""
            drain()
            link.drain()
            refresh_override()
            note_vehicle_rc_report()
            sample_motion()
            return sample.servo_outputs

        thrust_path: dict[str, Any] = {
            "floor_pwm": motors["floor_pwm"],
            "floor_at_arm_readback_pwm": motors["floor_at_arm_readback"],
            "opened_at_simulator_s": None,
            "opened_after_arm_simulator_s": None,
            "waited_within": "the excitation's climb drain (EXCITATION_CLIMB_DRAIN_S)",
            "why": (
                "the takeoff command is held until the vehicle's own report of its motor "
                "outputs leaves the floor it read at the arm readback: with the takeoff "
                "running, Mode::get_alt_hold_state_D_ms() (mode.cpp:1030-1068) returns "
                "Takeoff before the branch that asks for a spool state, and the takeoff "
                "state sets none, so a takeoff commanded while the airframe's motor "
                "interlock is still down (motors.cpp:59,75, the 2.0 s in_arming_delay) "
                "pins the desired spool state at SHUT_DOWN for the rest of the window "
                "(AP_MotorsMulticopter.cpp:619-638)"
            ),
        }
        record["thrust_path"] = thrust_path
        opened_at = _wait_for_thrust_path(
            poll_while_waiting, climb_window, motors, stats.sim_clock
        )
        thrust_path["opened_at_simulator_s"] = (
            None if opened_at is None else round(opened_at, 3)
        )
        thrust_path["floor_pwm"] = motors["floor_pwm"]
        arm_simulator_s = flight["simulator_time_at_arm_s"]
        thrust_path["opened_after_arm_simulator_s"] = (
            None
            if opened_at is None or arm_simulator_s is None
            else round(opened_at - arm_simulator_s, 3)
        )
        if opened_at is None and override["sent_pwm"] is not None:
            blockers.append(
                "the excitation had no thrust path: the vehicle's own report of its four "
                f"motor outputs never rose above the floor it read before the window "
                f"({motors['floor_pwm']} us at the arm readback, {motors['max_pwm']} us "
                f"maximum over {motors['samples']} samples) within the excitation's "
                f"declared {EXCITATION_CLIMB_DRAIN_S:g} s climb window, so the takeoff "
                "command that follows is the takeoff of an airframe whose motors are "
                "still shut down. The vehicle's own log records which spool state it "
                "was held in (SPOL Spl/SplDes) beside the mode changes"
            )
        record["sent"].append(
            link.takeoff_without_horizontal_position(EXCITATION_TAKEOFF_ALTITUDE_M)
        )
        flight["takeoff_commanded_at_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        flight["takeoff_commanded_after_arm_simulator_s"] = (
            None
            if stats.sim_clock.newest_s is None or arm_simulator_s is None
            else round(stats.sim_clock.newest_s - arm_simulator_s, 3)
        )
        # A liveness wait, and the wall clock is its right unit: what it asks is whether
        # the autopilot answers this command AT ALL, not whether the scene has had time
        # to do anything. The answer is milliseconds of the autopilot's own time away, so
        # a wall ceiling that admits a simulator at a fifth of realtime is still generous.
        ack_deadline = time.monotonic() + BRING_UP_TAKEOFF_ACK_TIMEOUT_S
        ack: dict[str, Any] | None = None
        while time.monotonic() < ack_deadline and ack is None:
            drain()
            link.drain()
            refresh_override()
            note_vehicle_rc_report()
            ack = link.command_ack(MAV_CMD_NAV_TAKEOFF)
            time.sleep(0.05)
        flight["takeoff_command_ack"] = ack
        if ack is None:
            blockers.append(
                "the autopilot did not answer the excitation's takeoff command within "
                f"{BRING_UP_TAKEOFF_ACK_TIMEOUT_S:.0f} s: no COMMAND_ACK for "
                "MAV_CMD_NAV_TAKEOFF arrived on the bring-up link, so whether the "
                "vehicle will climb is unmeasured"
            )
        elif ack.get("result") != 0:
            blockers.append(
                "the autopilot refused the excitation's takeoff: COMMAND_ACK result "
                f"{ack.get('result')} (0 is MAV_RESULT_ACCEPTED); the window's climb "
                "did not start"
            )
        # The rest of the excitation's climb window -- the part left after the thrust path
        # opened. Its declared job is to keep feeding while the climb happens (the
        # estimator latches on the frames the climb produces) so a wall deadline would
        # make the excitation's length depend on how loaded the host is; it was opened
        # above, before the takeoff command, because the climb cannot start until the
        # airframe's own thrust path is open. Measured both ways on this same host: 3.3
        # simulated seconds of window lifted the aircraft 0.27 m on run
        # p01l-rel10-4-20260927T201628Z at low load, and the 2.20 simulated seconds the
        # wall clock bought on run-2026-09-28T02-43-27-442Z lifted it 0.03 m, because
        # the LAND command arrived before the thrust path's first motor output.
        while not climb_window.expired():
            drain()
            link.drain()
            refresh_override()
            note_vehicle_rc_report()
            sample_motion()
            if max_altitude_m >= (
                EXCITATION_TAKEOFF_ALTITUDE_M - EXCITATION_ALTITUDE_REACHED_MARGIN_M
            ):
                climb_window.close("altitude_reached")
                break
            time.sleep(0.05)
        # The airtime is read and closed here, at the LAND command, because that is
        # what it measures: the window is never polled to expiry, so it would otherwise
        # report the time until the whole bring-up was documented.
        airtime.close("land_commanded")
        land_delay_s = airtime.elapsed_simulator_s()
        land_delay_wall_s = time.monotonic() - arm_monotonic
        flight["land_command_delay_s"] = (
            None if land_delay_s is None else round(land_delay_s, 3)
        )
        flight["land_command_delay_wall_s"] = round(land_delay_wall_s, 3)
        flight["land_command_delay_unit"] = (
            "land_command_delay_s is SIMULATOR seconds, the domain the airtime bound is "
            "declared in; land_command_delay_wall_s is the same interval on the host's "
            "clock and is recorded rather than judged"
        )
        if land_delay_s is None:
            blockers.append(
                "the excitation's airtime cannot be measured: no frame carrying the "
                "simulator's own clock was read between the arm readback and the LAND "
                "command, so the declared airtime bound has nothing to judge"
            )
        elif land_delay_s > EXCITATION_MAX_AIRTIME_S:
            blockers.append(
                f"the LAND command went out {land_delay_s:.3f} simulated s after the arm "
                f"readback ({land_delay_wall_s:.3f} wall s on this host), outside the "
                f"declared {EXCITATION_MAX_AIRTIME_S} s airtime bound"
            )
        session.set_mode("LAND")
        flight["land_commanded_at_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        flight["pairs_fed_at_land"] = stats.pairs
        flight["imu_fed_at_land"] = stats.imu_samples
        # The window's thrust path ends here, with the window. The release is an
        # explicit zero on the same channel, and what settles it is the vehicle's
        # own RC report: the descent drain below keeps reading chan3_raw, and the
        # closure gate refuses the scored window while that answer is still the
        # override's value.
        if override["sent"]:
            record["sent"].append(link.send_rc_channels_override(RC_THROTTLE_RELEASE_PWM))
            override["released"] = True
            override["released_at_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
        # The bounded post-LAND drain, in the same seconds as the window it ends: its
        # declared job is to feed the descent's frames to the estimator, and the descent
        # is a thing the aircraft does in the aircraft's time.
        descent_window = window(
            EXCITATION_POST_LAND_DRAIN_S, "the excitation's post-LAND descent drain"
        )
        while not descent_window.expired():
            drain()
            link.drain()
            note_vehicle_rc_report()
            sample_motion()
            time.sleep(0.05)
        if override["observed_after_release"] is not None:
            override["confirmed_at_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
        flight["max_altitude_readback_m"] = round(max_altitude_m, 3)
        flight["motor_output_max_pwm"] = motors["max_pwm"]
        flight["motor_output_samples"] = motors["samples"]
        flight["mode_after_window"] = sample.mode_name
        flight["armed_after_window"] = sample.armed
        if override["sent"] and override["observed_during_window"] != override["sent_pwm"]:
            blockers.append(
                "the window's throttle override never took effect: the vehicle's own RC "
                f"report read {override['observed_during_window']} us on channel "
                f"{RC_THROTTLE_CHANNEL} while the window sent "
                f"{override['sent_pwm']} us, so the excitation had no thrust path at "
                "all. The firmware ignores an RC override whose source system id is not "
                "the vehicle's own declared GCS (GCS_Common.cpp:4216-4220), and the "
                "vehicle's own RC report is what says whether it accepted this one"
            )
        if max_altitude_m < 0.10:
            blockers.append(
                f"the excitation produced no measured motion: max altitude readback "
                f"{max_altitude_m:.3f} m against the commanded "
                f"{EXCITATION_TAKEOFF_ALTITUDE_M} m climb, so the estimator had nothing "
                f"to latch on. The airframe's own outputs are recorded beside it: "
                f"{motors['max_pwm']} us maximum on the four motors over "
                f"{motors['samples']} samples -- an accepted takeoff whose motors never "
                "left idle produced no thrust, which is why there was no motion"
            )

    # 5. The window is closed: every declared value restored, then read back.
    if record["flight"]:
        write_parameters(restore_values)
        restore_readback, observations = readback(names, restore_values)
        record["readback_observations"]["restored"] = observations
        record["readbacks"]["restored"] = {name: restore_readback.get(name) for name in names}
        closure = _bring_up_closure_blockers(restore_readback, override)
        record["closure_blockers"] = closure
        blockers.extend(closure)
        # The seam's own source set, read back after the window: the claimed arm's
        # declared parameters, measured rather than assumed.
        record["window_closed_at_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        record["height_source_statement"] = (
            "baro supplied the aircraft's height during the bring-up window only; the "
            "claimed arm's vertical source is ExternalNav (EK3_SRC1_POSZ "
            f"{BRING_UP_HEIGHT_SOURCE_RESTORE_VALUE:g}, restored and verified by the "
            "vehicle's own readback before the claimed arm). Baro supplied no horizontal "
            "position, no attitude and no velocity, and no truth reached the estimator"
        )
        # The one line the receipt owes a reader about the thrust path, stated in
        # the same place as the height source it sits beside.
        record["throttle_statement"] = (
            "the bring-up sent a bounded local throttle: ONE RC channel override "
            f"(channel {override['channel']}, {override['sent_pwm']} us, a declared "
            f"pilot climb rate of {override['declared_climb_rate_ms']:g} m/s) while the "
            "declared window was in force, refreshed and then released, with the "
            "vehicle's own RC report as the evidence that it took and that it was "
            "lifted; the CLAIMED arm sends none -- it issues no RC, throttle or pulse "
            "command at all, and the session that owns its wire carries no such "
            "capability"
        )
        record["window_parameters_in_force"] = {
            "start": record.get("window_opened_at_utc"),
            "end": record.get("window_closed_at_utc"),
            "parameters": {
                name: {
                    "window": window_readback.get(name),
                    "restore": restore_readback.get(name),
                    "declared_window_value": window_value,
                    "declared_restore_value": restore_value,
                }
                for name, window_value, restore_value, _why in BRING_UP_WINDOW_PARAMETERS
            },
        }
        seam_expected = {
            name: expected
            for name, expected, _source in SEAM_REQUIREMENTS
            if name in BRING_UP_SEAM_READBACK
        }
        seam_readback, seam_observations = readback(BRING_UP_SEAM_READBACK, seam_expected)
        record["readback_observations"]["seam_after_window"] = seam_observations
        record["readbacks"]["seam_after_window"] = {
            name: seam_readback.get(name) for name in BRING_UP_SEAM_READBACK
        }
        for name, expected, source in SEAM_REQUIREMENTS:
            if name not in BRING_UP_SEAM_READBACK:
                continue
            if seam_readback.get(name) != expected:
                blockers.append(
                    f"{name} read back as {seam_readback.get(name)} after the bring-up "
                    f"window; the seam needs {name} {expected:g} from {source}"
                )
        record["seam"] = {
            "statement": (
                "the seam (VISO_TYPE 1, EK3_SRC1_* = 6) is the claimed arm's declared "
                "state; the height source returns to ExternalNav here, which changes the "
                "EKF's aiding and is recorded as a nav_epoch reset (SYSTEM-SPECIFICATION "
                "sections 4.3 and 6.3), not as a continuation"
            ),
            "readback_after_window": record["readbacks"]["seam_after_window"],
        }
    else:
        record["closure_blockers"] = []
        if not blockers:
            blockers.append(
                "the bring-up window never flew, so its exception was never applied and "
                "never restored; the scored arm does not run on an unmeasured window"
            )

    record["replies"] = list(link.replies)
    record["simulator_windows"] = [opened.document() for opened in windows]
    record["completed"] = not blockers
    writer.write_json("bring-up.json", record)
    log_lines.append(
        f"ordered bring-up: completed={record['completed']}, "
        f"refusals={record.get('refusals', [])}, "
        f"max altitude readback {record['flight'].get('max_altitude_readback_m')} m, "
        f"land delay {record['flight'].get('land_command_delay_s')} s of SIMULATOR time "
        f"({record['flight'].get('land_command_delay_wall_s')} s of wall time; declared "
        f"bound {EXCITATION_MAX_AIRTIME_S} s), "
        f"throttle override {record['throttle_override']['sent_pwm']} us, "
        f"released={record['throttle_override']['released']}, vehicle RC report "
        f"{record['throttle_override']['observed_during_window']} -> "
        f"{record['throttle_override']['observed_after_release']}, "
        f"window readback {record['readbacks'].get('window')}, "
        f"restored readback {record['readbacks'].get('restored')}"
    )
    log_lines.append(
        "ordered bring-up windows, in SIMULATOR seconds (the simulator's own clock; "
        "ended_by=simulator means the declared budget was spent, wall_ceiling means the "
        "host ran it outside the declared realtime envelope): "
        + "; ".join(
            f"{row['label']} budget {row['budget_simulator_s']:g} s, spent "
            f"{row['elapsed_simulator_s']} s in {row['elapsed_wall_s']} wall s, "
            f"ended_by={row['ended_by']}"
            for row in record["simulator_windows"]
        )
    )
    for blocker in blockers:
        log_lines.append(f"UNRESOLVED: {blocker}")
    return record


def _bring_up_manifest(record: dict[str, Any]) -> dict[str, Any]:
    """The bring-up's own line in the receipt, at its full declared strength."""
    if not record:
        return {"attempted": False}
    return {
        "attempted": True,
        "completed": record["completed"],
        "window_parameters": {
            row["name"]: {"window": row["window_value"], "restore": row["restore_value"]}
            for row in record["declaration"]["parameter_window"]
        },
        "origin_datum": {
            key: record["declaration"]["origin_datum"][key]
            for key in ("latitude_deg", "longitude_deg", "altitude_msl_m", "is_a_pose_feed")
        },
        "excitation": record["declaration"]["window"],
        # The units the declared windows were spent in, and what each one cost. A budget
        # in simulated seconds with ended_by=simulator is a window that ran its declared
        # course; ended_by=wall_ceiling is a host that ran it outside the declared
        # realtime envelope, which is the run's own timing-invalid condition rather than a
        # claim about the aircraft.
        "window_clock": record.get("window_clock", {}),
        "simulator_windows": record.get("simulator_windows", []),
        "arm_attempts": record.get("arm_attempts"),
        "refusals": record.get("refusals", []),
        "flight": record["flight"],
        "readbacks": record["readbacks"],
        "readback_observations": record.get("readback_observations", {}),
        "closure_blockers": record.get("closure_blockers", []),
        "window_opened_at_utc": record.get("window_opened_at_utc"),
        "window_closed_at_utc": record.get("window_closed_at_utc"),
        "height_source_statement": record.get("height_source_statement"),
        "throttle_statement": record.get("throttle_statement"),
        "throttle_override": record.get("throttle_override", {}),
        "window_parameters_in_force": record.get("window_parameters_in_force", {}),
        "blockers": list(record["blockers"]),
    }

# ---------------------------------------------------------------------------
# E1-DIAG: the live pose-assisted diagnostic (plan sections 0.6 item 7, 0.7, 0.8)
# ---------------------------------------------------------------------------


def _diagnostic_preflight(
    document: dict[str, Any], output_dir: Path
) -> tuple[list[dict[str, Any]], bool]:
    """Every prerequisite the diagnostic's flight needs, reported in one pass.

    Two of the sensor-derived preflight's rows are deliberately absent here, and
    the absence is the design (plan sections 0.7 item 7, 0.8 item 3):

    * ``localization_mode`` -- the arm is a property of the run, not of the file.
      The configuration keeps its declared ``sensor-derived`` arm untouched; this
      run overrides the arm through the bridge's own per-run ``arm=`` mechanism,
      which is what turns the truth republish on for this run alone.
    * the 4.6 truth-republish refusal -- replaced by a row that records the
      DECLARED exemption at full strength: the gate keeps truth out of the
      estimator's input, the estimator's input here is stereo and inertial only,
      and only the vehicle's navigation is truth-driven, which is what makes the
      arm flyable before the estimator works.

    Everything else the flight needs is still checked: the scene must be able to
      feed the pinned initializer (a featureless scene cannot answer the question),
      the simulator and SITL must exist, the ports must be free, and the pin must
      hold.
    """
    root = repository_root()
    settings = _platform_settings(document, root, arm=SensorMode.POSE_ASSISTED.value)
    rows: list[dict[str, Any]] = []
    satisfied = True
    declared_mode = (document.get("localization") or {}).get("mode")
    rows.append(
        {
            "name": "diagnostic_arm_override",
            "satisfied": True,
            "detail": (
                "the configuration declares localization.mode "
                f"{declared_mode!r} and that declaration is untouched; this run "
                "overrides the arm to pose-assisted through the bridge's per-run "
                "arm= mechanism (webots_ardupilot.py PlatformSettings.from_config), "
                "so the truth republish is on for this run alone and the "
                "sensor-derived arm's behaviour is unchanged (plan section 0.8)"
            ),
        }
    )
    rows.append(
        {
            "name": "bridge_truth_republish",
            "satisfied": True,
            "state": "declared_exemption",
            "detail": (
                "the bridge's truth republish is ON for this run by construction: "
                + DIAGNOSTIC_TRUTH_EXEMPTION
            ),
        }
    )
    scene_ok, scene_state, scene_detail = _scene_admission_check(settings, root)
    rows.append(
        {
            "name": "scene_admission",
            "satisfied": scene_ok,
            "state": scene_state,
            "detail": scene_detail,
        }
    )
    satisfied = satisfied and scene_ok
    for check in check_prerequisites(settings, output_dir):
        rows.append({"name": check.name, "satisfied": check.satisfied, "detail": check.detail})
        satisfied = satisfied and check.satisfied
    mavlink_port = _parse_mavlink_port(settings.mavlink_endpoint)
    port_free = mavlink_port is not None and _port_is_free(mavlink_port)
    rows.append(
        {
            "name": "port_mavlink",
            "satisfied": port_free,
            "detail": f"tcp {settings.mavlink_endpoint} is "
            + ("free" if port_free else "already in use; kill orphaned SITL processes first"),
        }
    )
    satisfied = satisfied and port_free
    localization = document["localization"]
    pin_record, pin_blockers = _pin_evidence(localization, root)
    rows.append(
        {
            "name": "estimator_pin",
            "satisfied": not pin_blockers,
            "detail": pin_blockers[0] if pin_blockers else _pin_summary(pin_record),
            "evidence": pin_record,
        }
    )
    satisfied = satisfied and not pin_blockers
    for blocker in (
        *_executable_blockers(localization, root),
        *_seam_blockers(document, root),
    ):
        rows.append({"name": "estimator_seam", "satisfied": False, "detail": blocker})
        satisfied = False
    return rows, satisfied


def _run_pose_assisted_diagnostic(document: dict[str, Any], output_dir: Path) -> CommandOutcome:
    """E1-DIAG: fly the bounded excitation on truth, observe the pinned estimator.

    The question is one fact (plan sections 0.6 item 7, 0.8 item 1): does
    VioManager::initialized() latch when the platform actually moves? The arm
    that can already fly supplies the motion -- the bridge republishes the
    simulator's own pose as the autopilot's external-navigation source (the
    P00 gate's proven arm), so the vehicle arms and flies a bounded vertical
    excitation while the pinned estimator runs alongside as a PURE OBSERVER on
    the same declared stereo and inertial stream:

    * no ``ExternalNavPublisher`` is constructed -- the estimator's states are
      recorded and published nowhere;
    * the estimator's input is the declared stereo pairs and inertial samples
      and nothing else (the ov_stream protocol has no truth field);
    * ``ov_stream`` sends a STATE frame only while ``sys->initialized()`` is
      true (ov_stream.cpp:513) and the frame's ``initialized`` field is that
      accessor's own value (ov_stream.cpp:301), so the first STATE frame to
      arrive IS the accessor flipping, independent of the 25-frame-quantized
      ``initialized=`` progress lines.

    Every artifact and the receipt carry the diagnostic labels and the
    non-claim; nothing from this directory enters any E/F/H verdict.
    """
    root = repository_root()
    settings = _platform_settings(document, root, arm=SensorMode.POSE_ASSISTED.value)
    localization = document["localization"]
    estimator = localization["estimator"]

    rows, satisfied = _diagnostic_preflight(document, output_dir)
    preflight = {
        "mode": SensorMode.POSE_ASSISTED.value,
        "sensor_mode_label": DIAGNOSTIC_SENSOR_MODE_LABEL,
        "checks": rows,
        "satisfied": satisfied,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    if not satisfied:
        blockers = tuple(f"{row['name']}: {row['detail']}" for row in rows if not row["satisfied"])
        return _pose_assisted_outcome(
            blockers,
            {"preflight": rows, "estimator_pin": estimator},
            ("preflight.json",),
        )

    writer = EvidenceWriter(output_dir, "run-a")
    _write_environment(writer, settings, estimator)
    writer.write_json(
        "diagnostic-identity.json",
        {
            "run_kind": "E1-DIAG",
            "sensor_mode_label": DIAGNOSTIC_SENSOR_MODE_LABEL,
            "question": (
                "does VioManager::initialized() latch when the platform actually "
                "moves? (plan sections 0.6 item 7, 0.7, 0.8)"
            ),
            "carrier": (
                "the truth-driven arm: GUIDED, the bridge republishing the "
                "simulator's own pose as the autopilot's external-navigation source, "
                "on the P00 gate's own parameter layers (GPS on, origin and home "
                "from the synthesized GPS) -- the arm that can already fly"
            ),
            "observer": (
                "the pinned estimator on the declared stereo+inertial stream; no "
                "ExternalNavPublisher exists in this run, so the estimator publishes "
                "nowhere and no truth field exists in its protocol"
            ),
            "excitation": {
                "mode": EXCITATION_MODE,
                "takeoff_altitude_m": EXCITATION_TAKEOFF_ALTITUDE_M,
                "max_airtime_s": EXCITATION_MAX_AIRTIME_S,
                "lateral_setpoint": None,
                "termination": "LAND",
            },
            "non_claim": DIAGNOSTIC_NON_CLAIM,
            "truth_exemption": DIAGNOSTIC_TRUTH_EXEMPTION,
            "reading_rule": (
                "latch beside measured motion = the deadlock is a bring-up "
                "artifact; motion declines with no latch = the estimator is refuted "
                "on this stream; fewer than five declining frames = inconclusive, "
                "one E-EXC-B re-fly (ceiling 1.5 m) is predeclared (plan section "
                "0.7 item 6). A latch that appears only beside starvation declines "
                "(no-IMU warnings) repeats the feed artifact and does not answer "
                "the question"
            ),
        },
    )

    estimator_process = _start_estimator(estimator, root, writer)
    if estimator_process is None:
        return _pose_assisted_outcome(
            (
                f"the estimator process {estimator['executable']} did not open port "
                f"{estimator['socket_port']}; see run-a/estimator.log",
            ),
            {"estimator_pin": estimator},
            (*writer.artifacts, "preflight.json"),
        )
    client = loc.OvStreamClient("127.0.0.1", int(estimator["socket_port"]))
    try:
        client.connect()
    except (OSError, loc.ProtocolError) as error:
        _stop_estimator(estimator_process)
        return _pose_assisted_outcome(
            (
                f"the observer could not connect to the estimator on 127.0.0.1:"
                f"{estimator['socket_port']}: {error}",
            ),
            {"estimator_pin": estimator},
            (*writer.artifacts, "preflight.json"),
        )

    stats = _FeedStats()
    states_log = writer.path("observer-states.jsonl")
    progress_log = writer.path("observer-progress.jsonl")
    # The phase names the flight's own clock: bring_up (feeding, waiting for the
    # arm), airborne (arm readback to the LAND command), descent (LAND command to
    # the end of the post-land drain). Every observer record carries one.
    phase: dict[str, str] = {"name": "bring_up"}
    observer: dict[str, Any] = {
        "states_received": 0,
        "first_state_at": None,
        "first_state_phase": None,
        "first_state_after_pairs": None,
        "max_altitude_m": 0.0,
        "feed_error": None,
    }
    pending_pairs: queue.Queue = queue.Queue(maxsize=PAIR_QUEUE_FRAMES)
    last_telemetry_sample = 0.0

    def file_record(record: Any) -> None:
        if record.kind is not Kind.PAIR or record.pair is None:
            return
        stats.pair_records_filed += 1
        try:
            pending_pairs.put_nowait(record)
        except queue.Full:
            stats.pair_records_dropped += 1

    def feed_record(record: Any) -> None:
        stats.sim_clock.observe(record.sim_time_s)
        if record.kind is Kind.PAIR and record.pair is not None:
            # Only reachable before the reader strips pixels; the sink's queue is
            # the normal path, but a pair that arrives with its planes is fed
            # rather than dropped.
            feed_pair(record)
        elif record.kind is Kind.IMU and record.imu is not None:
            imu = record.imu
            stamp_ns = sim_time_ns(record.sim_time_s)
            stats.newest_imu_ns = max(stats.newest_imu_ns, stamp_ns)
            client.send(loc.encode_imu(stamp_ns, imu.gyro, imu.accelerometer))
            stats.imu_samples += 1
        elif record.kind is Kind.POSE and record.pose is not None:
            # The vehicle's own pose on the sensor stream: read for the motion
            # record and sent nowhere. The estimator's feed handles PAIR and IMU
            # only, so this branch cannot reach it.
            stats.truth_samples.append(
                (sim_time_ns(record.sim_time_s), tuple(record.pose.position_xyz))
            )
            stats.truth_attitudes.append(
                (sim_time_ns(record.sim_time_s), tuple(record.pose.attitude_rpy))
            )

    def feed_pair(record: Any) -> None:
        pair = record.pair
        left = loc.grayscale_rgb8(pair.left_bytes, settings.stereo.width, settings.stereo.height)
        right = loc.grayscale_rgb8(
            pair.right_bytes, settings.stereo.width, settings.stereo.height
        )
        client.send(
            loc.encode_stereo(
                sim_time_ns(record.sim_time_s),
                left,
                right,
                settings.stereo.width,
                settings.stereo.height,
            )
        )
        stats.pairs += 1

    def observe_state(state: loc.EstimatorState) -> None:
        observer["states_received"] += 1
        if observer["first_state_at"] is None:
            observer["first_state_at"] = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
            observer["first_state_phase"] = phase["name"]
            observer["first_state_after_pairs"] = stats.pairs
        with states_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "at_utc": datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        ),
                        "phase": phase["name"],
                        "pairs_fed": stats.pairs,
                        "imu_fed": stats.imu_samples,
                        "state": {
                            "time_ns": state.time_ns,
                            "initialized": state.initialized,
                            "position_m": list(state.position_m),
                            "velocity_mps": list(state.velocity_mps),
                            "sigma_pos_m": list(state.sigma_pos_m),
                            "n_tracks": state.n_tracks,
                            "t_last_visual_ns": state.t_last_visual_ns,
                            "reset_counter": state.reset_counter,
                        },
                    },
                    default=str,
                )
                + "\n"
            )

    def drain() -> None:
        """Feed the estimator, record its answers, keep the telemetry folded.

        No machine runs here and nothing is judged: the diagnostic's product is
        the record, and the only stop condition is the feed itself failing.
        """
        nonlocal last_telemetry_sample
        while True:
            try:
                record = platform.sensor_record(0.0)
            except ProbeFailure as error:
                observer["feed_error"] = f"the sensor stream failed: {error}"
                return
            if record is None:
                try:
                    while True:
                        pending = pending_pairs.get_nowait()
                        if pending.kind is Kind.PAIR and pending.pair is not None:
                            feed_pair(pending)
                except queue.Empty:
                    pass
                except loc.ProtocolError as error:
                    observer["feed_error"] = str(error)
                    return
                try:
                    state = client.poll_state()
                except loc.ProtocolError as error:
                    observer["feed_error"] = str(error)
                    return
                if state is not None:
                    observe_state(state)
                if time.monotonic() - last_telemetry_sample >= TELEMETRY_SAMPLE_PERIOD_S:
                    last_telemetry_sample = time.monotonic()
                    try:
                        sample = platform.telemetry()
                    except ProbeFailure as error:
                        observer["feed_error"] = f"the telemetry stream failed: {error}"
                        return
                    altitude = (
                        None
                        if sample.local_position_ned is None
                        else -sample.local_position_ned[2]
                    )
                    if altitude is not None and altitude > observer["max_altitude_m"]:
                        observer["max_altitude_m"] = altitude
                    with progress_log.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "at_utc": datetime.now(timezone.utc).isoformat(
                                        timespec="milliseconds"
                                    ),
                                    "phase": phase["name"],
                                    "pairs_fed": stats.pairs,
                                    "imu_fed": stats.imu_samples,
                                    "states_received": observer["states_received"],
                                    "mode": sample.mode_name,
                                    "armed": sample.armed,
                                    "altitude_m": altitude,
                                },
                                default=str,
                            )
                            + "\n"
                        )
                return
            try:
                feed_record(record)
            except loc.ProtocolError as error:
                observer["feed_error"] = str(error)
                return

    session = PymavlinkSession()
    # The carrier's parameter layers are the compatibility gate's own
    # (``compat_estimator_params``), NOT the scored arm's: the diagnostic's arm is
    # "the arm that can already fly" (plan sections 0.6 item 7, 0.8 item 1), and
    # that arm was measured on the P00 gate's layer set -- GPS on, home and the
    # EKF origin from the synthesized GPS, external nav from the bridge's truth
    # republish. The scored arm's layer (``estimator_params``) turns the GPS off,
    # and at this pin an origin then exists only if a GCS declares it: EKF3 sets
    # its origin from GPS, from a beacon, or from set_ekf_origin -- never from
    # ExternalNav data (AP_NavEKF3_Measurements.cpp:680-715, GCS_Common.cpp:3957
    # "should only be used when there is no GPS") -- which is exactly the
    # first flight's measured refusal ('AHRS: waiting for home' behind 2129
    # published truth poses, 'EKF3 IMU0 is using external nav data'). The
    # estimator-observer is unaffected either way: its input is the stereo and
    # inertial stream, never the vehicle's parameters.
    carrier_params = settings.compat_estimator_params
    platform = WebotsArduPilot(
        settings,
        runner=SubprocessRunner(),
        session=session,
        gateway=TcpSensorGateway(stamp=lambda: settings.capture_stamp(time.monotonic_ns())),
        evidence=writer,
        label="run-a",
        extra_params=carrier_params,
    )
    platform.record_sink = file_record
    log_lines: list[str] = [
        f"E1-DIAG pose-assisted diagnostic, started {datetime.now(timezone.utc).isoformat()}",
        DIAGNOSTIC_NON_CLAIM,
        f"estimator pin (observer): {estimator['name']} {estimator['tag']} "
        f"({estimator['commit']})",
        f"excitation: {EXCITATION_MODE} takeoff to {EXCITATION_TAKEOFF_ALTITUDE_M} m, no "
        f"lateral setpoint, LAND within {EXCITATION_MAX_AIRTIME_S} s of the arm readback",
        f"carrier parameter layers (the P00 gate's own; GPS on, origin from GPS): "
        f"{[str(path) for path in carrier_params]}",
    ]
    blockers: list[str] = []
    motion_window: dict[str, Any] = {
        "sensor_mode_label": DIAGNOSTIC_SENSOR_MODE_LABEL,
        "commanded_sequence": [
            f"request_control (GUIDED + arm, the P00 gate's arm)",
            f"takeoff({EXCITATION_TAKEOFF_ALTITUDE_M})",
            "drain (climb)",
            "set_mode(LAND)",
            "drain (descent)",
        ],
        "excitation": {
            "mode": EXCITATION_MODE,
            "takeoff_altitude_m": EXCITATION_TAKEOFF_ALTITUDE_M,
            "max_airtime_s": EXCITATION_MAX_AIRTIME_S,
            "lateral_setpoint": None,
            "termination": "LAND",
        },
        "carrier_parameter_layers": [str(path) for path in carrier_params],
        "carrier_note": (
            "the P00 compatibility gate's own layers: GPS on (the synthesized GPS "
            "sets the EKF origin and home), external nav from the bridge's truth "
            "republish. The scored arm's GPS-off layer is NOT applied to this "
            "carrier: at this pin an origin then needs a GCS datum declaration, "
            "and the arm that can already fly is the P00 gate's (plan section 0.6 "
            "item 7). The estimator-observer's input is stereo+inertial only, "
            "untouched by the carrier's parameters"
        ),
    }
    flight: dict[str, Any] = {}
    control: Any = None
    shutdown: Any = None
    try:
        platform.start()
        platform.wait_ready(settings.step_timeout_s.startup)
        platform.request_telemetry_streams()
        for message_id in (MSG_ID_SYS_STATUS, MSG_ID_GPS_RAW_INT):
            session.request_message_interval(message_id, GPS_AIDING_SAMPLE_HZ)
        control, _attempt_at = platform.request_control(settings.pre_arm_wait_s, drain=drain)
        log_lines.append(
            f"control: mode_reached={control.mode_reached}, armed={control.armed}, "
            f"attempts={control.control_attempts}, refusals={list(control.refusals)}"
        )
        if control.refused:
            blockers.append(
                "the diagnostic's truth-driven arm was refused: "
                f"mode_reached={control.mode_reached}, armed={control.armed}, "
                f"refusals={list(control.refusals)}; without the flight there is no "
                "motion, so the question is not answered by this run"
            )
        else:
            # The same constants, and therefore the same units: the excitation's windows
            # belong to the aircraft's time here too (EXCITATION_* above). They are spent
            # against this run's own simulator clock, read beside the frames the observer
            # feeds, so one constant means one thing in both arms of this stage.
            def window(budget_s: float, label: str) -> _SimWindow:
                ceiling = _sim_window_wall_ceiling_s(
                    budget_s, settings.realtime_ratio_envelope[0]
                )
                opened = _SimWindow(
                    stats.sim_clock, budget_s, label=label, wall_ceiling_s=ceiling
                )
                motion_window.setdefault("simulator_windows", []).append(opened)
                return opened

            airtime = window(
                EXCITATION_MAX_AIRTIME_S,
                "the diagnostic excitation's declared airtime, arm readback to LAND",
            )
            arm_monotonic = time.monotonic()
            flight["armed_at_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
            flight["simulator_time_at_arm_s"] = stats.sim_clock.newest_s
            flight["pairs_fed_at_arm"] = stats.pairs
            flight["imu_fed_at_arm"] = stats.imu_samples
            flight["truth_pose_at_arm"] = (
                list(stats.truth_samples[-1][1]) if stats.truth_samples else None
            )
            phase["name"] = "airborne"
            session.takeoff(EXCITATION_TAKEOFF_ALTITUDE_M)
            flight["takeoff_commanded_at_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
            climb_window = window(EXCITATION_CLIMB_DRAIN_S, "the diagnostic's climb drain")
            while not climb_window.expired():
                drain()
                if observer["max_altitude_m"] >= (
                    EXCITATION_TAKEOFF_ALTITUDE_M - EXCITATION_ALTITUDE_REACHED_MARGIN_M
                ):
                    climb_window.close("altitude_reached")
                    break
                time.sleep(0.05)
            airtime.close("land_commanded")
            land_delay_s = airtime.elapsed_simulator_s()
            flight["land_command_delay_s"] = (
                None if land_delay_s is None else round(land_delay_s, 3)
            )
            flight["land_command_delay_wall_s"] = round(
                time.monotonic() - arm_monotonic, 3
            )
            if land_delay_s is None:
                blockers.append(
                    "the diagnostic excitation's airtime cannot be measured: no frame "
                    "carrying the simulator's own clock was read between the arm readback "
                    "and the LAND command"
                )
            elif land_delay_s > EXCITATION_MAX_AIRTIME_S:
                blockers.append(
                    f"the LAND command went out {land_delay_s:.3f} simulated s after the "
                    f"arm readback, outside the declared {EXCITATION_MAX_AIRTIME_S} s "
                    "airtime bound"
                )
            session.set_mode("LAND")
            phase["name"] = "descent"
            flight["land_commanded_at_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
            flight["pairs_fed_at_land"] = stats.pairs
            flight["imu_fed_at_land"] = stats.imu_samples
            flight["truth_pose_at_land"] = (
                list(stats.truth_samples[-1][1]) if stats.truth_samples else None
            )
            descent_window = window(
                EXCITATION_POST_LAND_DRAIN_S, "the diagnostic's post-LAND descent drain"
            )
            while not descent_window.expired():
                drain()
                time.sleep(0.05)
    except Exception as error:  # noqa: BLE001 - the run's outer guard records, never swallows
        blockers.append(f"the platform failed during the diagnostic: {error}")
    finally:
        shutdown = platform.stop()
        client.close()
        _stop_estimator(estimator_process)
        phase["name"] = "stopped"

    if observer["feed_error"] is not None:
        blockers.append(f"the estimator's feed failed mid-run: {observer['feed_error']}")
    if control is not None and not control.refused and observer["max_altitude_m"] < 0.10:
        # The arm succeeded but the airframe never measurably left the ground, so
        # the reading rule has no motion to read: named rather than silently
        # answered (a takeoff the autopilot refused in flight is exactly this).
        blockers.append(
            f"the excitation produced no measured motion: max altitude readback "
            f"{observer['max_altitude_m']:.3f} m against the commanded "
            f"{EXCITATION_TAKEOFF_ALTITUDE_M} m climb, so this run cannot answer "
            "the question"
        )
    diagnostics = _initializer_diagnostics(writer.path("estimator.log"))
    truth_published = platform.truth_feed_published
    motion_window.update(
        {
            "flight": flight,
            "max_altitude_readback_m": round(observer["max_altitude_m"], 3),
            "truth_positions_extremes_m": {
                "min": [min(p[i] for _t, p in stats.truth_samples) for i in range(3)],
                "max": [max(p[i] for _t, p in stats.truth_samples) for i in range(3)],
            }
            if stats.truth_samples
            else None,
            "pairs_fed_total": stats.pairs,
            "imu_fed_total": stats.imu_samples,
            "pair_records_filed": stats.pair_records_filed,
            "pair_records_dropped": stats.pair_records_dropped,
            "observer_states_received": observer["states_received"],
            "observer_first_state": {
                "at_utc": observer["first_state_at"],
                "phase": observer["first_state_phase"],
                "pairs_fed_before_it": observer["first_state_after_pairs"],
            },
            "estimator_log_markers": {
                "q_GtoI_tail_lines": diagnostics["q_GtoI_tail_lines"],
                "zupt_accepted_updates": diagnostics["zupt_accepted_updates"],
                "zupt_frames_without_imu": diagnostics["zupt_frames_without_imu"],
                "zupt_rejected_updates": diagnostics["zupt_rejected_updates"],
                "initializer_succeeded": diagnostics["initializer_succeeded"],
            },
            "bridge_truth_published": truth_published,
            "bridge_truth_published_meaning": (
                "positive by construction: the diagnostic arm is truth-driven, and "
                "the number is the bridge's own count of simulator poses it sent the "
                "autopilot. It never touches the estimator, whose protocol has no "
                "truth field (plan section 0.7 item 7)"
            ),
            "non_claim": DIAGNOSTIC_NON_CLAIM,
        }
    )
    writer.write_json("motion-window.json", motion_window)
    writer.write_json("initializer-diagnostics.json", diagnostics)
    writer.write_json("gps-aiding.json", _gps_aiding_verdict(writer.path("mavlink.jsonl")))
    log_lines.extend(
        [
            f"pairs fed: {stats.pairs}, imu samples fed: {stats.imu_samples}, "
            f"truth pose samples read (sent nowhere): {len(stats.truth_samples)}",
            f"observer states received (each one means initialized() was true): "
            f"{observer['states_received']}",
            f"first observer state: {observer['first_state_at']} in phase "
            f"{observer['first_state_phase']}, after {observer['first_state_after_pairs']} pairs",
            f"estimator log: q_GtoI tails {diagnostics['q_GtoI_tail_lines']}, ZUPT accepted "
            f"{diagnostics['zupt_accepted_updates']}, starved "
            f"{diagnostics['zupt_frames_without_imu']}, declined on motion (DEBUG lines, "
            f"visible only if printed) {diagnostics['zupt_rejected_updates']}",
            f"max altitude readback: {observer['max_altitude_m']:.3f} m; bridge truth poses "
            f"sent to the autopilot: {truth_published}",
            f"shutdown: {shutdown.exits}",
        ]
    )
    if blockers:
        for blocker in blockers:
            log_lines.append(f"UNRESOLVED: {blocker}")
    else:
        log_lines.append(
            "the diagnostic completed; the reading is in motion-window.json and "
            "initializer-diagnostics.json, never in a gate"
        )
    _write_log(writer, log_lines)
    return _pose_assisted_outcome(
        tuple(blockers),
        {
            "estimator_pin": estimator,
            "excitation": motion_window["excitation"],
            "truth_republish": settings.truth_republish,
            "bridge_truth_published": truth_published,
            "pairs_fed": stats.pairs,
            "imu_samples_fed": stats.imu_samples,
            "observer_states_received": observer["states_received"],
            "observer_first_state_at": observer["first_state_at"],
            "estimator_log_markers": motion_window["estimator_log_markers"],
            "flight": flight,
            "max_altitude_readback_m": round(observer["max_altitude_m"], 3),
            "shutdown": {"exits": shutdown.exits},
        },
        (*writer.artifacts, "preflight.json"),
    )



def _wait_initialized(
    machine: loc.HealthMachine,
    drain: Callable[[], None],
    wait: _SimWindow,
    *,
    poll_s: float = SIM_WINDOW_POLL_S,
) -> bool:
    """H5's wait: the estimator's readiness, spent in the simulator's seconds.

    The estimator latches on frames, and frames arrive in the scene's time, so the wait
    is a wait for simulated work -- the same unit the configuration declares this budget
    in ("a simulated GPS needs a fix, the EKF needs a home, and the IMU consistency check
    needs a quiet window, all of which take simulated time"). A wall deadline here would
    hand a loaded host fewer of the estimator's own seconds than the declaration granted,
    which is exactly how the excitation beside it was cut off. ``wait`` carries its own
    wall ceiling, so a stalled simulator ends the wait instead of hanging it.
    """
    while not wait.expired():
        drain()
        if machine.state == "healthy":
            wait.close("estimator_initialized")
            return True
        time.sleep(poll_s)
    return False


def _wait_for_thrust_path(
    poll: Callable[[], Sequence[int] | None],
    wait: _SimWindow,
    motors: dict[str, Any],
    clock: _SimulatorClock,
    *,
    poll_s: float = SIM_WINDOW_POLL_S,
) -> float | None:
    """Wait, inside the declared window, for the airframe's own thrust path to open.

    The gate is the VEHICLE's own report of its motor outputs, compared against the
    floor the same report read while no spool-up had been requested: the four outputs
    sit at the airframe's own minimum until a spool state has been asked for and
    accepted, so "the motors left the floor" is a comparison between two of the
    vehicle's own readings and this window supplies no PWM value of its own for it.

    Why the takeoff has to wait for it: the takeoff is what pins the mode's alt-hold
    state machine into its Takeoff state, and that state asks for no spool state at
    all (``get_alt_hold_state_D_ms``, mode.cpp:1030-1068, tests ``takeoff.running()``
    before the branch that asks), so the desired spool state it freezes is whichever
    one was last accepted -- SHUT_DOWN while the motors library is forcing it
    (AP_MotorsMulticopter.cpp:619-638), which is the whole of Copter's 2.0 s
    ``ap.in_arming_delay`` after the arm (motors.cpp:59,75).

    Returns the simulator time at which the vehicle's own report first rose above the
    floor, or ``None`` if ``wait`` ended first -- the caller owes the reader the
    difference. ``wait`` carries its own wall ceiling, so a stalled simulator ends the
    wait instead of hanging it.
    """
    floor = motors.get("floor_pwm")
    while not wait.expired():
        observed = poll()
        if observed is not None:
            top = max(observed[:4])
            if floor is None:
                # The arm readback carried no servo report, so the first one the window
                # can compare against is this one -- taken while the thrust path is
                # still off, which is what makes it a floor rather than a sample.
                floor = top
            elif top > floor:
                motors["floor_pwm"] = floor
                motors["left_floor_at_simulator_s"] = clock.newest_s
                return clock.newest_s
        time.sleep(poll_s)
    motors["floor_pwm"] = floor
    return None

def _scene_capture_gate(
    writer: EvidenceWriter,
    settings: PlatformSettings,
    drain: Callable[[], None],
    scene_capture: dict[str, Any],
    log_lines: list[str],
) -> str | None:
    """Revision 4's arm-side scene admission (plan section 0.3 item 4), both eyes.

    The preflight measured a hash-matched capture when one existed; this gate
    measures the run's own recorded static-start frames, with the world's own
    hash written beside them, and refuses the arm when they do not clear the
    initializer's floor. No flight can be spent on a scene the initializer
    cannot fire in.

    Revision 5 measures the pair, not the left frame: both eyes must clear the
    floor, and a capture with no complete, distinct pair cannot open the arm at
    all. The estimator this arm feeds receives two planes, so one eye's keypoint
    count is not evidence about it.
    """
    deadline = time.monotonic() + SCENE_CAPTURE_TIMEOUT_S
    while scene_capture["count"] < SCENE_CAPTURE_MAX_FRAMES and time.monotonic() < deadline:
        drain()
        time.sleep(0.05)
    pairs_dir = writer.directory / "pairs"
    measurement = _stereo_capture_measurement(pairs_dir, SCENE_ADMISSION_MAX_FRAMES)
    record: dict[str, Any] = {
        "world": str(settings.world),
        "world_sha256": _sha256(settings.world),
        "pairs_recorded": measurement["pairs"],
        "left_keypoint_counts": measurement["left_counts"],
        "right_keypoint_counts": measurement["right_counts"],
        "missing_right": measurement["missing_right"],
        "identical_pairs": measurement["identical_pairs"],
        "feature_floor": INITIALIZER_FEATURE_FLOOR,
        "fast_threshold": FAST_THRESHOLD,
    }
    if measurement["cv2_unavailable"]:
        record["state"] = "cv2_unavailable"
        writer.write_json("scene-capture.json", record)
        return (
            "the scene-admission gate could not measure this run's own frames: OpenCV "
            "(cv2) is not importable"
        )
    if not measurement["pairs"]:
        record["state"] = "no_stereo_pair"
        writer.write_json("scene-capture.json", record)
        return (
            "the scene-admission gate refuses the arm: this run recorded no complete "
            "stereo pair (both eyes present and not the same image twice), so nothing "
            "in its own evidence says what the second view feeds the estimator "
            f"(missing right frames {len(measurement['missing_right'])}, identical "
            f"pairs {len(measurement['identical_pairs'])})"
        )
    if not measurement["left_counts"] or not measurement["right_counts"]:
        record["state"] = "unreadable"
        writer.write_json("scene-capture.json", record)
        return (
            "the scene-admission gate refuses the arm: this run's recorded pairs could "
            "not be read back as images, so neither eye could be measured"
        )
    lowest_left = min(measurement["left_counts"])
    lowest_right = min(measurement["right_counts"])
    passed = min(lowest_left, lowest_right) >= INITIALIZER_FEATURE_FLOOR
    record["state"] = "measured_pass" if passed else "measured_fail"
    writer.write_json("scene-capture.json", record)
    log_lines.append(
        f"scene admission at the arm gate: {measurement['pairs']} stereo pairs measured, "
        f"keypoints left {measurement['left_counts']}, right "
        f"{measurement['right_counts']}, floor {INITIALIZER_FEATURE_FLOOR}"
    )
    if passed:
        return None
    worst_eye = "left" if lowest_left <= lowest_right else "right"
    return (
        f"the configured scene's own recorded frames carry "
        f"{min(lowest_left, lowest_right)} FAST keypoints in the {worst_eye} eye "
        f"(left {lowest_left}, right {lowest_right}) against the pinned initializer's "
        f"floor of {INITIALIZER_FEATURE_FLOOR} (plan section 0.3 item 4): the stereo "
        "stream this front end is given cannot initialize, so the arm is refused"
    )

def _wrap_angle(radius: float) -> float:
    return (radius + math.pi) % (2.0 * math.pi) - math.pi
def _attitude_gate(
    writer: EvidenceWriter,
    aligned: dict[str, object] | None,
    alignment: loc.OdomAlignment,
    stats: _FeedStats,
    log_lines: list[str],
) -> str | None:
    """A1, second form: the sealed epoch frame, checked before the arm (0.3 item 5).

    The first textured invocation measured a −180° roll (the missing FLU→FRD body
    map) and the second a +90° yaw (the odom frame's yaw, unobservable and chosen
    by the initializer's noise-decided gram_schmidt branch). Both were refused
    before any arm, and the adapter now derives its epoch rotation from the
    estimator's own first initialized attitude and the world's declared start
    attitude. A1 therefore checks the two things that remain independently
    checkable: that the published attitude equals the *declared* start attitude
    (the composition the adapter performed), and that the declared start attitude
    equals the simulator's own truth attitude (the declaration itself). It also
    gates the vertical chain — the sealed rotation must carry odom-up to NED-down
    — and records the derived yaw, which is not gated because nothing in this arm
    observes it. That the seal absorbs the estimator's own initialization tilt is
    a recorded limitation, not a hidden one: a real tilt error shows up in E1 and
    H3 once the vehicle moves.
    """
    declared = tuple(alignment.declared_start_rpy)
    record: dict[str, Any] = {
        "tolerance_deg": ATTITUDE_GATE_TOLERANCE_DEG,
        "declared_start_rpy_rad": list(declared),
    }
    if aligned is None or not stats.truth_attitudes:
        reason = (
            "no published state to compare"
            if aligned is None
            else "no truth attitude arrived on the pose records"
        )
        record.update({"state": "unmeasured", "reason": reason})
        writer.write_json("attitude-gate.json", record)
        return f"A1 could not run: {reason}"
    published = aligned["attitude_rpy"]
    truth_time_ns, truth_rpy = stats.truth_attitudes[-1]
    composition_deg = [
        math.degrees(_wrap_angle(p - d)) for p, d in zip(published, declared)
    ]
    declaration_deg = [math.degrees(_wrap_angle(d - t)) for d, t in zip(declared, truth_rpy)]
    odom_up_down_component = float(alignment.epoch_rotation[2][2])
    level_error_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, -odom_up_down_component)))
    )
    passed = (
        max(abs(delta) for delta in composition_deg) <= ATTITUDE_GATE_TOLERANCE_DEG
        and max(abs(delta) for delta in declaration_deg) <= ATTITUDE_GATE_TOLERANCE_DEG
        and level_error_deg <= ATTITUDE_GATE_TOLERANCE_DEG
    )
    record.update(
        {
            "state": "measured_pass" if passed else "measured_fail",
            "published_rpy_rad": list(published),
            "truth_rpy_rad": list(truth_rpy),
            "truth_time_ns": truth_time_ns,
            "composition_deltas_deg": composition_deg,
            "declaration_deltas_deg": declaration_deg,
            "epoch_yaw_deg": alignment.epoch_yaw_deg(),
            "level_error_deg": level_error_deg,
            "note": (
                "epoch_yaw_deg is recorded, not gated: nothing in this arm observes the "
                "odom frame's yaw, which the pinned initializer fixes by a "
                "noise-decided branch; the seal derives it once from the declared start"
            ),
        }
    )
    writer.write_json("attitude-gate.json", record)
    log_lines.append(
        f"A1 attitude gate: composition deltas (deg) {composition_deg}, declaration "
        f"deltas (deg) {declaration_deg}, epoch yaw {record['epoch_yaw_deg']:.2f} deg, "
        f"level error {level_error_deg:.3f} deg"
    )
    if passed:
        return None
    return (
        f"A1: the sealed epoch frame does not hold up: composition deltas "
        f"{composition_deg} deg, declaration deltas {declaration_deg} deg, level error "
        f"{level_error_deg:.3f} deg against the declared "
        f"{ATTITUDE_GATE_TOLERANCE_DEG} deg tolerance (plan section 0.3 item 5), so the "
        "arm is refused"
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    """The nearest-rank percentile of an unsorted sample, order-statistic honest at any n."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _truth_error_statistics(
    published: Sequence[tuple[int, Sequence[float]]],
    truth: Sequence[tuple[int, Sequence[float]]],
    bounds_config: dict[str, Any],
) -> dict[str, Any]:
    """E1: the published state against evaluator truth, joined on simulator time.

    Both sides are already in local NED — the adapter converts the estimator's odom
    frame through the fixed alignment, and the controller converts the Webots devices
    exactly as it does for the flight-state packet — so the comparison is a subtraction
    in one common frame. Truth is read for scoring and reaches nothing else: the
    estimator's feed carries stereo pairs and inertial samples.

    A publication is joined to the nearest truth sample and used only if that sample is
    inside the tolerance; anything further apart is counted as unjoined rather than
    interpolated, because an interpolated truth value is not a measurement.
    """
    bounds = {
        "bound_p95_horizontal_m": bounds_config["error_p95_horizontal_m"],
        "bound_p95_vertical_m": bounds_config["error_p95_vertical_m"],
        "bound_max_horizontal_m": bounds_config["error_max_horizontal_m"],
        "bound_max_vertical_m": bounds_config["error_max_vertical_m"],
    }
    join_tolerance_ns = 100_000_000
    if not published or not truth:
        return {
            "measured": False,
            "reason": (
                f"E1 has nothing to compare: {len(published)} scored publications and "
                f"{len(truth)} truth samples were read"
            ),
            **bounds,
        }
    truth_times = [time_ns for time_ns, _position in truth]
    horizontal: list[float] = []
    vertical: list[float] = []
    unjoined = 0
    for time_ns, estimate in published:
        index = bisect_left(truth_times, time_ns)
        candidates = [truth[i] for i in (index - 1, index) if 0 <= i < len(truth)]
        nearest_time_ns, nearest_position = min(
            candidates, key=lambda sample: abs(sample[0] - time_ns)
        )
        if abs(nearest_time_ns - time_ns) > join_tolerance_ns:
            unjoined += 1
            continue
        horizontal.append(
            math.hypot(estimate[0] - nearest_position[0], estimate[1] - nearest_position[1])
        )
        vertical.append(abs(estimate[2] - nearest_position[2]))
    if not horizontal:
        return {
            "measured": False,
            "reason": (
                f"E1 has no joined samples: {len(published)} scored publications, every one "
                f"further than {join_tolerance_ns / 1e6:.0f} ms from a truth sample"
            ),
            "scored_publications": len(published),
            "truth_samples": len(truth),
            "unjoined": unjoined,
            **bounds,
        }
    return {
        "measured": True,
        "scored_publications": len(published),
        "truth_samples": len(truth),
        "joined_samples": len(horizontal),
        "unjoined": unjoined,
        "join_tolerance_ms": join_tolerance_ns / 1e6,
        "p95_horizontal_error_m": _percentile(horizontal, 0.95),
        "p95_vertical_error_m": _percentile(vertical, 0.95),
        "max_horizontal_error_m": max(horizontal),
        "max_vertical_error_m": max(vertical),
        **bounds,
    }


def _summarise_disagreements(
    disagreements: list[dict[str, Any]], p95_bound_m: float, max_bound_m: float
) -> dict[str, Any]:
    norms = sorted(row["norm_m"] for row in disagreements)
    if not norms:
        return {
            "measured": False,
            "reason": "no common-frame samples: the autopilot never reported a local position "
            "while the adapter published none either",
            "bound_p95_m": p95_bound_m,
            "bound_max_m": max_bound_m,
            "samples": disagreements,
        }
    index = min(len(norms) - 1, int(round(0.95 * (len(norms) - 1))))
    return {
        "measured": True,
        "p95_m": norms[index],
        "max_m": norms[-1],
        "bound_p95_m": p95_bound_m,
        "bound_max_m": max_bound_m,
        "samples": disagreements,
    }


def _score(
    machine: loc.HealthMachine,
    bounds: loc.HealthBounds,
    bounds_config: dict[str, Any],
    valid_fraction: float,
    truth_comparison: dict[str, Any],
    disagreement_summary: dict[str, Any],
    truth_published: int,
    gps_aiding: dict[str, Any],
) -> list[dict[str, Any]]:
    """The predeclared bounds, evaluated exactly as frozen. E1 unmeasured fails the gate."""
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "bridge_truth_republish",
            "status": "pass" if truth_published == 0 else "fail",
            "detail": (
                "the bridge sent no simulator pose for the whole run: the autopilot's "
                "external-navigation source had exactly one publisher, the adapter"
                if truth_published == 0
                else f"the bridge sent {truth_published} simulator poses during the run"
            ),
        }
    )
    gps_ok = not gps_aiding["blockers"]
    checks.append(
        {
            "name": "G4",
            "status": "pass" if gps_ok else "fail",
            "detail": (
                f"GPS-off confirmed at runtime: {gps_aiding['sys_status_samples']} "
                "SYS_STATUS samples without the GPS-present bit, "
                f"{gps_aiding['gps_raw_int_samples']} GPS_RAW_INT samples without a "
                f"fix, no GPS driver STATUSTEXT"
                if gps_ok
                else "; ".join(gps_aiding["blockers"])
            ),
        }
    )
    if truth_comparison["measured"]:
        measured = (
            truth_comparison["p95_horizontal_error_m"],
            truth_comparison["p95_vertical_error_m"],
            truth_comparison["max_horizontal_error_m"],
            truth_comparison["max_vertical_error_m"],
        )
        limit = (
            bounds_config["error_p95_horizontal_m"],
            bounds_config["error_p95_vertical_m"],
            bounds_config["error_max_horizontal_m"],
            bounds_config["error_max_vertical_m"],
        )
        met = all(value <= bound for value, bound in zip(measured, limit))
        checks.append(
            {
                "name": "E1",
                "status": "pass" if met else "fail",
                "detail": (
                    f"over {truth_comparison['joined_samples']} joined samples "
                    f"({truth_comparison['unjoined']} unjoined): p95 horizontal "
                    f"{measured[0]:.3f} m, p95 vertical {measured[1]:.3f} m, max horizontal "
                    f"{measured[2]:.3f} m, max vertical {measured[3]:.3f} m against "
                    f"({limit[0]:.2f}, {limit[1]:.2f}, {limit[2]:.2f}, {limit[3]:.2f}) m"
                ),
            }
        )
    else:
        checks.append({"name": "E1", "status": "fail", "detail": truth_comparison["reason"]})
    gaps = machine.publish_gaps_s
    checks.append(
        {
            "name": "F1/F3",
            "status": "pass"
            if gaps and max(gaps) <= bounds.max_publish_gap_s
            else "fail",
            "detail": (
                f"{len(gaps)} publishes; max gap "
                f"{max(gaps):.3f} s against the {bounds.max_publish_gap_s:.3f} s bound"
                if gaps
                else "no publishes in the scored window"
            ),
        }
    )
    ages = machine.published_state_ages_s
    checks.append(
        {
            "name": "F2",
            "status": "pass"
            if ages and max(ages) <= bounds.published_state_age_max_s
            else "fail",
            "detail": (
                f"published-state age max {max(ages):.3f} s against the "
                f"{bounds.published_state_age_max_s:.3f} s bound"
                if ages
                else "no publishes in the scored window"
            ),
        }
    )
    checks.append(
        {
            "name": "H1",
            "status": "pass" if valid_fraction >= bounds.valid_fraction_min else "fail",
            "detail": f"valid fraction {valid_fraction:.4f} against "
            f"{bounds.valid_fraction_min:.2f}; outages "
            f"{[round(outage, 3) for outage in machine.outages_s]} s",
        }
    )
    if disagreement_summary["measured"]:
        h3_ok = (
            disagreement_summary["p95_m"] <= bounds_config["disagreement_p95_m"]
            and disagreement_summary["max_m"] <= bounds_config["disagreement_max_m"]
        )
        checks.append(
            {
                "name": "H3",
                "status": "pass" if h3_ok else "fail",
                "detail": f"p95 {disagreement_summary['p95_m']:.3f} m, max "
                f"{disagreement_summary['max_m']:.3f} m against "
                f"({bounds_config['disagreement_p95_m']:.2f}, "
                f"{bounds_config['disagreement_max_m']:.2f}) m",
            }
        )
    else:
        checks.append({"name": "H3", "status": "fail", "detail": disagreement_summary["reason"]})
    # H2/H4 is the machine's state at the END OF THE SCORED WINDOW, which is where the
    # window's declared end (arm to disarm) puts it. Reading the process's final state
    # instead makes the declared 300 ms silence stop -- the correct behaviour once the
    # adapter's input is gone -- look like a fault: measured, run
    # p01l-fix3-20260927T045304Z, whose shutdown left the machine stopped after a flight
    # whose own window closed healthy. A machine that is stopped at the close still fails.
    closed_state = machine.state_at_close
    final_state = machine.state if closed_state is None else closed_state
    checks.append(
        {
            "name": "H2/H4",
            "status": "pass" if final_state == "healthy" else "fail",
            "detail": f"machine state at the end of the scored window {final_state} "
            f"(at process exit {machine.state}); events "
            f"{[event.event for event in machine.events]}; adapter resets "
            f"{machine.reset_counter}",
        }
    )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    """The module entry point: dispatch through the shared CLI as ``localize-check``."""
    from embodied.cli import main as cli_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    return cli_main([COMMAND_NAME, *arguments])


def _register_once() -> None:
    """Register the command unless this stage already registered it.

    Running this module as an entry point executes it twice: once as ``__main__``,
    and again under its package name when ``build_parser`` imports the dispatch
    list. Plain registration raises on the second import and takes the command down
    before it parses its own arguments, so the registration is idempotent for this
    stage's own spec — and still refuses to stand down for anyone else's.
    """
    existing = COMMAND_REGISTRY.get(COMMAND_NAME)
    if existing is not None:
        if existing.stage_id == STAGE_ID:
            return
        raise CommandError(
            f"command {COMMAND_NAME!r} is already registered by stage {existing.stage_id}"
        )
    register_command(
        COMMAND_NAME,
        _localize_check_command,
        help_text="run the P01-L localization check against the pinned estimator",
        stage_id=STAGE_ID,
        run_prefix=RUN_PREFIX,
        add_arguments=_add_arguments,
    )


_register_once()


if __name__ == "__main__":
    sys.exit(main())
