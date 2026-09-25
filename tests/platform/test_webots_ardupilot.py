"""The platform adapter, its wire formats and the compat command, on substituted seams.

Nothing here starts Webots, SITL, or any network connection beyond loopback. The
three seams the adapter declares (process runner, MAVLink session, sensor gateway)
are replaced with scripted components, so the probe's own sequence and judgement are
under test rather than the simulator's behaviour.
"""

import ast
from pathlib import Path
import os
import json
import re
import socket
import signal
import struct
import subprocess
import sys
import threading
import time

import pytest
import yaml
from embodied import cli
from embodied.contracts import records as R
from embodied.platform import webots_ardupilot as W

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "platform"
SCENE = REPO_ROOT / "scenarios" / "compat"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeClock:
    """A clock that only moves when somebody waits, so tests do not sleep for real."""

    def __init__(self, start_s=1000.0, step_s=0.05):
        self.now = start_s
        self.step_s = step_s

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return int(self.now * 1e9)

    def sleep(self, seconds):
        self.now += max(float(seconds), self.step_s)


class FakeRunner:
    """Spawns nothing, and remembers what it was asked to spawn and terminate."""

    def __init__(self, exit_codes=None):
        self.spawned = []
        self.terminated = []
        self._exit_codes = exit_codes or {}

    def spawn(self, name, argv, *, log_path, cwd=None, env=None):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"{name} started: {' '.join(argv)}\n", encoding="utf-8")
        child = W.ChildProcess(name=name, argv=tuple(argv), pid=4200 + len(self.spawned), log_path=log_path)
        self.spawned.append((name, tuple(argv), dict(env or {})))
        return child

    def poll(self, child):
        return self._exit_codes.get(child.name)

    def terminate(self, child, timeout_s=10.0):
        self.terminated.append(child.name)
        return self._exit_codes.get(child.name, 0)

    def tail(self, child, lines=200):
        return [f"{child.name}: last line of its log"]


class FakeGateway:
    """A controller that streams status, stereo pairs and inertial samples on demand.

    It produces records at a rate in the tests' own time base and returns nothing once
    the reader has caught up, because that is what the real stream does and what the
    probe's drain relies on: a pass over the sensor stream ends when the stream is
    momentarily empty, so a fake that could always produce another record would never
    let a pass finish.
    """

    RECORDS_PER_SECOND = 110.0

    def __init__(self, clock, *, pair_bytes=None, colourless=False, acknowledge=True, motors=("m1_motor", "m2_motor", "m3_motor", "m4_motor")):
        self.clock = clock
        self.pair_bytes = pair_bytes
        self.colourless = colourless
        self.acknowledge = acknowledge
        self.motors = list(motors)
        self.faults = []
        self.applied = {}
        self.queue = []
        self.status_sent = False
        self.sequence = 0
        self.pair_counter = 0
        self.closed = False
        self._produced_until = None
        self._budget = 0.0

    def _status(self):
        return W.SensorRecord(
            kind=W.Kind.STATUS,
            sim_time_s=self.clock.monotonic() - 1000.0,
            sequence=0,
            flags=0,
            received_stamp=W.ClockStamp(host_id="test-host", clock_id="monotonic", monotonic_ns=self.clock.monotonic_ns()),
            status={
                "python": {"executable": sys.executable, "version": sys.version, "platform": sys.platform},
                "shared_module": {"file": "test", "records_revision": R.RECORDS_REVISION},
                "host_id": "test-host",
                "clock_id": "monotonic",
                "devices": {
                    "left": "camera left",
                    "right": "camera right",
                    "accelerometer": "accelerometer",
                    "gyro": "gyro",
                    "inertial_unit": "inertial unit",
                    "gps": "gps",
                },
                "motors": self.motors,
                "cameras": {"left": 100, "right": 100},
                "camera_size": [64, 48],
                "imu_period_ms": 10,
                "stream": {
                    "frames_produced": self.pair_counter,
                    "frames_dropped": 0,
                    "reader_connected": True,
                },
                "scene": {"witness_colour_rgb": [217, 13, 13], "witness_note": "test panel"},
            },
        )

    def _pair(self):
        self.pair_counter += 1
        left, right = self.pair_bytes or synthetic_pair(
            colourless=self.colourless, width=64, height=48
        )
        payload = W.PairPayload(
            capture_host_ns=self.clock.monotonic_ns(),
            pair_id=self.pair_counter,
            left_frame_id=self.pair_counter,
            right_frame_id=self.pair_counter,
            width=64,
            height=48,
            encoding="rgb8",
            left_bytes=left,
            right_bytes=right,
        )
        return W.SensorRecord(
            kind=W.Kind.PAIR,
            sim_time_s=self.clock.monotonic() - 1000.0,
            sequence=self.pair_counter,
            flags=0,
            received_stamp=W.ClockStamp(
                host_id="test-host", clock_id="monotonic", monotonic_ns=self.clock.monotonic_ns()
            ),
            pair=payload,
        )

    def _imu(self):
        return W.SensorRecord(
            kind=W.Kind.IMU,
            sim_time_s=self.clock.monotonic() - 1000.0,
            sequence=self.pair_counter,
            flags=0,
            received_stamp=W.ClockStamp(
                host_id="test-host", clock_id="monotonic", monotonic_ns=self.clock.monotonic_ns()
            ),
            imu=W.ImuPayload(
                capture_host_ns=self.clock.monotonic_ns(),
                accelerometer=(0.05, -0.03, -9.79),
                gyro=(0.001, -0.002, 0.0),
                inertial_unit_rpy=(0.01, -0.02, 0.3),
                device_names=("accelerometer", "gyro", "inertial unit"),
                units="m/s^2; rad/s; rad, ENU negated on y and z into NED",
            ),
        )

    def open(self, host, port, timeout_s):
        return None

    def read_record(self, timeout_s):
        """One record produced since the last read, or nothing when it has caught up."""
        if self.queue:
            return self.queue.pop(0)
        if not self.status_sent:
            self.status_sent = True
            return self._status()
        now = self.clock.monotonic()
        if self._produced_until is None:
            # The first read happens the moment the reader connects, with no time behind
            # it: nothing has been produced yet, which is what the real stream reports.
            self._produced_until = now
            return None
        self._budget += (now - self._produced_until) * self.RECORDS_PER_SECOND
        self._produced_until = now
        if self._budget < 1.0:
            return None
        self._budget -= 1.0
        self.sequence += 1
        return self._pair() if self.sequence % 2 else self._imu()

    def send_fault(self, injection):
        self.faults.append(injection)
        if not self.acknowledge:
            return
        if injection.apply:
            self.applied[injection.kind] = {"magnitude_m": injection.magnitude_m}
        else:
            self.applied = {}
        self.queue.append(
            W.SensorRecord(
                kind=W.Kind.FAULT_ACK,
                sim_time_s=self.clock.monotonic() - 1000.0,
                sequence=0,
                flags=0,
                received_stamp=W.ClockStamp(
                    host_id="test-host", clock_id="monotonic", monotonic_ns=self.clock.monotonic_ns()
                ),
                fault_ack={
                    "injection_id": injection.injection_id,
                    "applied": True,
                    "state": dict(self.applied),
                    "reason": None,
                },
            )
        )

    def close(self):
        self.closed = True


class ScriptedMavlinkSession:
    """A vehicle that answers like an autopilot: it moves when told to and reports it."""

    def __init__(
        self,
        clock,
        *,
        moves=True,
        direction=1.0,
        dead_servos=False,
        refuse_guided=False,
        refuse_arming=False,
        guided_timeout_s=0.6,
        failsafe_after_setpoints=None,
        boot_jitter_s=0.0,
        gateway_holder=None,
        parameter_values=None,
        pre_arm_failures=0,
    ):
        self.clock = clock
        self.moves = moves
        self.direction = direction
        self.dead_servos = dead_servos
        self.refuse_guided = refuse_guided
        self.refuse_arming = refuse_arming
        self.guided_timeout_s = guided_timeout_s
        # A count of publications after which the autopilot leaves Guided on its own,
        # the way a failsafe would: the mode change is the aircraft's, not the test's.
        self.failsafe_after_setpoints = failsafe_after_setpoints
        self.boot_jitter_s = boot_jitter_s
        self.gateway_holder = gateway_holder if gateway_holder is not None else {}
        # The parameters this scripted vehicle reports about itself, as an autopilot
        # answers a parameter request: the value that is running, not the value intended.
        self.parameter_values = dict(parameter_values or {})
        # How many control requests the pre-arm checks refuse before they clear, as a
        # simulated GPS, home position and IMU consistency check do.
        self.pre_arm_failures = pre_arm_failures
        self.pre_arm_text = pre_arm_failures > 0
        self.pending_parameters = []
        self.sent = []
        self.commands = []
        self.intervals = []
        self.requested_messages = []
        self.requested_parameters = []
        self.closed = False
        self.armed = False
        self.mode = "STABILIZE"
        self.position = (0.0, 0.0, 0.0)
        self.target = None
        self.last_setpoint_at = None
        self.ekf_variance = 0.5
        self.flags = 1
        self.drains = 0

    # -- outbound --

    def connect(self, endpoint, timeout_s):
        return {
            "endpoint": endpoint,
            "target_system": 1,
            "target_component": 1,
            "heartbeat": {"mavpackettype": "HEARTBEAT"},
        }

    def request_message_interval(self, message_id, hz):
        self.intervals.append((message_id, hz))

    def request_message(self, message_id):
        self.requested_messages.append(message_id)

    def request_parameter(self, name):
        self.requested_parameters.append(name)
        self.pending_parameters.append(
            {
                "mavpackettype": "PARAM_VALUE",
                "param_id": name,
                "param_value": self.parameter_values.get(name, 0.0),
            }
        )

    def send_setpoint(self, setpoint):
        self.sent.append(setpoint)
        self.target = setpoint.target.position_ned
        self.last_setpoint_at = self.clock.monotonic()
        if self.refuse_guided:
            self.mode = "LOITER"
        elif (
            self.failsafe_after_setpoints is not None
            and len(self.sent) >= self.failsafe_after_setpoints
        ):
            self.mode = "LOITER"
        else:
            self.mode = "GUIDED"

    def set_mode(self, mode_name):
        self.commands.append(("set_mode", mode_name))
        if not self.refuse_guided:
            self.mode = mode_name
        else:
            self.mode = "LOITER"

    def arm(self):
        self.commands.append(("arm",))
        if self.refuse_arming:
            return
        if self.pre_arm_failures > 0:
            # The vehicle refuses while its pre-arm checks fail, however often it is asked,
            # and stops reporting them once they clear: what the probe has to observe
            # rather than guess at.
            self.pre_arm_failures -= 1
            self.pre_arm_text = self.pre_arm_failures > 0
            return
        self.armed = True

    def takeoff(self, altitude_m):
        self.commands.append(("takeoff", altitude_m))
        if self.refuse_guided:
            self.mode = "LOITER"
        else:
            self.mode = "GUIDED"
        self.target = (0.0, 0.0, -altitude_m)

    # -- inbound --

    def drain(self):
        self._advance()
        self.drains += 1
        parameters, self.pending_parameters = self.pending_parameters, []
        return parameters + self._telemetry_batch()

    def _telemetry_batch(self):
        """One batch of streamed messages, as the autopilot would send them."""
        # An alternating offset stands in for a device clock that is not steady.
        jitter = self.boot_jitter_s * (1 if self.drains % 2 else -1)
        boot_ms = int((self.clock.monotonic() - 100.0 + jitter) * 1000)
        armed_flag = 128 if self.armed else 0
        guided = self.mode == "GUIDED"
        if guided and self.last_setpoint_at is not None:
            if self.clock.monotonic() - self.last_setpoint_at > self.guided_timeout_s:
                self.mode = "LOITER"
                guided = False
        gateway = self.gateway_holder.get("gateway")
        if gateway is not None and gateway.applied.get("position_step"):
            self.ekf_variance = 40.0
            self.flags = 5
        else:
            self.ekf_variance = 0.5
            self.flags = 1
        servo = 1000 if (self.dead_servos or not guided) else 1500
        statustext = "guided" if guided else "EKF failsafe check"
        if not self.armed and (self.refuse_arming or self.pre_arm_text):
            # What an autopilot says while its pre-arm checks fail: the text is the
            # evidence that the refusal is the aircraft's decision, not the test's.
            statustext = "PreArm: 3D Accel calibration needed"
        return [
            {
                "mavpackettype": "HEARTBEAT",
                "custom_mode": W.COPTER_MODES and (4 if guided else 5),
                "base_mode": armed_flag | 81,
                "system_status": W.MAV_STATE_STANDBY,
            },
            {
                "mavpackettype": "ATTITUDE",
                "time_boot_ms": boot_ms,
                "roll": 0.01,
                "pitch": -0.02,
                "yaw": 0.3,
            },
            {
                "mavpackettype": "LOCAL_POSITION_NED",
                "time_boot_ms": boot_ms,
                "x": self.position[0],
                "y": self.position[1],
                "z": self.position[2],
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
            },
            {
                "mavpackettype": "SERVO_OUTPUT_RAW",
                **{f"servo{index}_raw": servo for index in range(1, 9)},
            },
            {
                "mavpackettype": "EKF_STATUS_REPORT",
                "flags": self.flags,
                "velocity_variance": self.ekf_variance,
                "pos_horiz_variance": self.ekf_variance,
            },
            {
                "mavpackettype": "AUTOPILOT_VERSION",
                "time_boot_ms": boot_ms,
                "flight_sw_version": 262144,
                "flight_custom_version": bytes([1, 2, 3, 4, 5, 6, 7, 8]),
                "vendor_id": 3,
                "product_id": 1,
            },
            {"mavpackettype": "STATUSTEXT", "text": statustext},
        ]

    def close(self):
        self.closed = True

    # -- scripted behaviour --

    def _advance(self):
        if not self.moves or self.target is None:
            return
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        dz = self.target[2] - self.position[2]
        step = 0.15 * self.direction

        def approach(delta):
            if abs(delta) <= abs(step):
                return delta
            return step if delta > 0 else -step

        self.position = (
            self.position[0] + approach(dx),
            self.position[1] + approach(dy),
            self.position[2] + approach(dz),
        )


def synthetic_pair(*, colourless=False, width=64, height=48):
    """Two frames that contain a saturated red panel, or one channel copied three times."""
    import numpy

    left = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    left[:, :, :] = (90, 100, 110)
    left[8:40, 4:24] = (217, 13, 13) if not colourless else (120, 120, 120)
    right = left.copy()
    right[8:40, 40:60] = left[8:40, 4:24]
    if colourless:
        # Exactly what an averaging bridge produces: one channel, three times.
        left[:, :, 1] = left[:, :, 0]
        left[:, :, 2] = left[:, :, 0]
        right[:, :, 1] = right[:, :, 0]
        right[:, :, 2] = right[:, :, 0]
    return left.tobytes(), right.tobytes()


# ---------------------------------------------------------------------------
# A scene and a configuration that can pass its prerequisite checks
# ---------------------------------------------------------------------------


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        if probe.bind(("127.0.0.1", port)) is not None:
            port = probe.getsockname()[1]
    return port


def write_scene(tmp_path, **overrides):
    """Create a small but complete prerequisite world, and a configuration for it."""
    webots_home = tmp_path / "Webots.app"
    (webots_home / "Contents" / "MacOS").mkdir(parents=True, exist_ok=True)
    binary = webots_home / "Contents" / "MacOS" / "webots"
    binary.write_text("#!/bin/sh\necho webots\n", encoding="utf-8")
    binary.chmod(0o755)
    (webots_home / "Contents" / "Resources").mkdir(parents=True, exist_ok=True)
    (webots_home / "Contents" / "Resources" / "version.txt").write_text("R2025a", encoding="utf-8")

    ardupilot = tmp_path / "ardupilot"
    (ardupilot / "build" / "sitl" / "bin").mkdir(parents=True, exist_ok=True)
    sitl = ardupilot / "build" / "sitl" / "bin" / "arducopter"
    sitl.write_text("#!/bin/sh\necho sitl\n", encoding="utf-8")
    sitl.chmod(0o755)
    if not (ardupilot / ".git").exists():
        subprocess.run(["git", "init", "-q", str(ardupilot)], check=True)
        subprocess.run(["git", "-C", str(ardupilot), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(ardupilot), "-c", "user.email=t@example.com", "-c", "user.name=t",
             "commit", "-q", "-m", "pinned"],
            check=True,
        )
    commit = subprocess.run(
        ["git", "-C", str(ardupilot), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    document = {
        "project": {"stage": "P00", "name": "test-scene"},
        "platform": {
            "ardupilot_root": str(ardupilot),
            "ardupilot_commit": commit,
            "sitl_binary": str(sitl),
            "webots_home": str(webots_home),
            "webots_version": "R2025a",
            "webots_mode": "realtime",
            "sim_model": "webots-python",
            "vehicle": "ArduCopter",
            "sitl_home": "-35.363261,149.165230,584,353",
            "endpoints": {"sitl": "tcp:127.0.0.1:5760", "fdm_port": free_port(), "controller_port": free_port()},
        },
        "scenario": {
            "world": str(SCENE / "worlds" / "compat_stereo.wbt"),
            "params": [
                str(SCENE / "params" / "compat_base.parm"),
                str(SCENE / "params" / "compat_arming.parm"),
            ],
            "estimator_params": [str(SCENE / "params" / "compat_ekf.parm")],
        },
        "sensors": {
            "stereo": {
                "left": "camera left",
                "right": "camera right",
                "width": 64,
                "height": 48,
                "baseline_m": 0.10,
                "sampling_period_ms": 100,
                "encoding": "rgb8",
                "identical_channel_fraction_limit": 0.99,
            },
            "imu": {
                "accelerometer": "accelerometer",
                "gyro": "gyro",
                "inertial_unit": "inertial unit",
                "gps": "gps",
                "sampling_period_ms": 10,
            },
            "calibration": str(SCENE / "calibration.json"),
        },
        "probe": {
            "hover_altitude_m": 1.5,
            "waypoints_local_ned": [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            "hold_per_waypoint_s": 2.0,
            "stream_loss_window_s": 3.0,
            # Short windows: the fake clock advances 0.05 s per wait, so these still
            # exercise the settle and measurement logic without thousands of records.
            "settle_s": 0.1,
            "at_rest_window_s": 0.2,
            "pre_arm_wait_s": 5.0,
            "estimator_fault": {"kind": "position_step", "magnitude_m": 30.0, "hold_s": 2.0},
            "timebase_samples": 5,
            # The scripted clock advances 0.05 s per wait, so a window is a fraction of a
            # second here: the relation only has to span one window to be measurable, and
            # the scripted simulator runs at realtime, inside the declared envelope.
            "realtime_window_s": 0.1,
            "realtime_ratio_envelope": [0.5, 1.5],
            "timebase_poll_s": 0.005,
            "step_timeout_s": {"startup": 30, "ready": 30, "flight": 60},
            "budget_wall_clock_s": 600,
        },
        "output": str(tmp_path / "runs"),
    }
    for section, values in overrides.items():
        document[section].update(values)
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return config_path


def settings_for(config_path, root):
    document = cli.load_config(Path(config_path))
    return W.PlatformSettings.from_config(document, root=root)


def run_probe(
    tmp_path,
    *,
    clock=None,
    gateway_kwargs=None,
    session_kwargs=None,
    runner=None,
    output_name="out",
    config_overrides=None,
    session_class=ScriptedMavlinkSession,
):
    """Run the whole checklist against scripted seams and return the result."""
    clock = clock or FakeClock()
    session_kwargs = dict(session_kwargs or {})
    holder = {}
    sessions = []
    runner = runner or FakeRunner()
    config_path = write_scene(tmp_path, **(config_overrides or {}))
    settings = settings_for(config_path, tmp_path)
    # The scripted vehicle reports the parameters its run layered, so a run whose files
    # were applied is a run whose read-back agrees. Both runs apply the candidate's
    # parameter files, estimator set included, so both report the same values.
    def parameters_for_run():
        if "parameter_values" in session_kwargs:
            return session_kwargs["parameter_values"]
        return W.read_configured_parameters(
            settings.parameter_files(settings.estimator_params)
        )

    def make_gateway():
        gateway = FakeGateway(clock, **(gateway_kwargs or {}))
        holder["gateway"] = gateway
        return gateway

    def make_session():
        values = parameters_for_run()
        overrides = {k: v for k, v in session_kwargs.items() if k != "parameter_values"}
        session = session_class(
            clock, gateway_holder=holder, parameter_values=values, **overrides
        )
        sessions.append(session)
        return session

    probe = W.CompatibilityProbe(
        settings,
        output_dir=tmp_path / output_name,
        runner_factory=lambda: runner,
        session_factory=make_session,
        gateway_factory=make_gateway,
        monotonic_ns=clock.monotonic_ns,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    result = probe.run()
    return result, probe, holder["gateway"], sessions, runner


def check(result, name):
    for entry in result.checks:
        if entry.name.startswith(name):
            return entry
    raise AssertionError(f"{name} was not reported; got {[c.name for c in result.checks]}")


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def test_framing_round_trips_every_message_kind():
    for kind in W.Kind:
        payload = bytes(range(40))
        framed = W.pack_message(kind, sim_time_s=12.5, sequence=7, payload=payload, flags=3)
        message, consumed = W.read_message(framed)
        assert consumed == len(framed)
        assert message.kind is kind
        assert message.flags == 3
        assert message.sim_time_s == 12.5
        assert message.sequence == 7
        assert message.payload == payload


def test_a_truncated_header_or_payload_is_rejected():
    framed = W.pack_message(W.Kind.STATUS, sim_time_s=1.0, sequence=1, payload=b"{}")
    with pytest.raises(W.FramingError):
        W.read_message(framed[:10])
    with pytest.raises(W.FramingError):
        W.read_message(framed[:-1])


def test_a_wrong_magic_version_or_kind_is_rejected():
    framed = bytearray(W.pack_message(W.Kind.STATUS, sim_time_s=1.0, sequence=1, payload=b"{}"))
    framed[0:4] = b"XXXX"
    with pytest.raises(W.FramingError):
        W.read_message(bytes(framed))
    framed = bytearray(W.pack_message(W.Kind.STATUS, sim_time_s=1.0, sequence=1, payload=b"{}"))
    framed[4:6] = struct.pack(">H", W.FORMAT_VERSION + 1)
    with pytest.raises(W.FramingError):
        W.read_message(bytes(framed))
    framed = bytearray(W.pack_message(W.Kind.STATUS, sim_time_s=1.0, sequence=1, payload=b"{}"))
    framed[6] = 99
    with pytest.raises(W.FramingError):
        W.read_message(bytes(framed))


def test_the_stream_reader_waits_for_a_whole_message():
    framed = W.pack_message(W.Kind.PAIR, sim_time_s=2.0, sequence=2, payload=b"0123456789")
    reader = W.FrameReader()
    reader.feed(framed[:12])
    assert reader.next_message() is None
    reader.feed(framed[12:])
    message = reader.next_message()
    assert message is not None and message.payload == b"0123456789"
    assert reader.next_message() is None


def test_pair_payload_round_trip_and_size_check():
    left, right = b"\x01" * (64 * 48 * 3), b"\x02" * (64 * 48 * 3)
    payload = W.encode_pair_payload(
        capture_host_ns=123456789,
        pair_id=4,
        left_frame_id=11,
        right_frame_id=11,
        width=64,
        height=48,
        encoding="rgb8",
        left_bytes=left,
        right_bytes=right,
    )
    decoded = W.decode_pair_payload(payload)
    assert decoded.capture_host_ns == 123456789
    assert (decoded.width, decoded.height) == (64, 48)
    assert decoded.left_bytes == left and decoded.right_bytes == right
    with pytest.raises(W.FramingError):
        W.decode_pair_payload(payload[:-1])
    with pytest.raises(W.FramingError):
        W.decode_pair_payload(payload[: W.PAIR_PAYLOAD_SIZE])


def test_inertial_and_json_payloads_round_trip():
    payload = W.encode_imu_payload(
        capture_host_ns=99,
        accelerometer=(0.1, -0.2, -9.8),
        gyro=(0.01, 0.02, 0.03),
        inertial_unit_rpy=(0.4, 0.5, 0.6),
        device_names=("accelerometer", "gyro", "inertial unit"),
        units="m/s^2; rad/s; rad",
    )
    decoded = W.decode_imu_payload(payload)
    assert decoded.accelerometer == (0.1, -0.2, -9.8)
    assert decoded.device_names == ("accelerometer", "gyro", "inertial unit")
    assert decoded.units == "m/s^2; rad/s; rad"
    status = W.decode_status_payload(W.encode_status_payload({"b": 2, "a": 1}))
    assert status == {"a": 1, "b": 2}
    with pytest.raises(W.FramingError):
        W.decode_status_payload(b"[]")


# ---------------------------------------------------------------------------
# The SITL wire
# ---------------------------------------------------------------------------


def test_the_control_packet_layout_matches_the_pinned_model():
    # SIM_Webots_Python.cpp reads sixteen floats, which is 64 bytes with no padding.
    # SITL fills them with (pulse_width - 1000) / 1000, so the test builds that packet
    # itself rather than asking the module to encode what the module then decodes.
    assert W.CONTROL_SIZE == 64
    packet = struct.pack("f" * 16, 0.0, 0.5, 1.0, *([-1.0] * 13))
    assert W.unpack_controls(packet) == (0.0, 0.5, 1.0, *([-1.0] * 13))
    with pytest.raises(W.FramingError):
        W.unpack_controls(packet[:-4])


def test_the_flight_state_packet_layout_matches_the_pinned_model():
    # struct fdm_packet: one timestamp, then gyro, acceleration, attitude, velocity and
    # position, each three doubles. 16 doubles is 128 bytes with no padding.
    assert W.FDM_SIZE == 128
    state = W.SimFdmState(
        timestamp_s=3.5,
        gyro_rpy=(0.1, 0.2, 0.3),
        accel_xyz=(1.0, 2.0, 3.0),
        attitude_rpy=(0.4, 0.5, 0.6),
        velocity_xyz=(4.0, 5.0, 6.0),
        position_xyz=(7.0, 8.0, 9.0),
    )
    expected = struct.pack(
        "d" * 16,
        3.5,
        0.1, 0.2, 0.3,
        1.0, 2.0, 3.0,
        0.4, 0.5, 0.6,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    )
    assert W.pack_fdm(state) == expected


def test_enu_to_ned_maps_all_six_signed_axes():
    assert W.enu_to_ned((1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)  # north is north
    assert W.enu_to_ned((-1.0, 0.0, 0.0)) == (-1.0, 0.0, 0.0)
    assert W.enu_to_ned((0.0, 1.0, 0.0)) == (0.0, -1.0, 0.0)  # east becomes -y
    assert W.enu_to_ned((0.0, -1.0, 0.0)) == (0.0, 1.0, 0.0)
    assert W.enu_to_ned((0.0, 0.0, 1.0)) == (0.0, 0.0, -1.0)  # up becomes -z
    assert W.enu_to_ned((0.0, 0.0, -1.0)) == (0.0, 0.0, 1.0)
    original = (0.3, -0.4, 0.5)
    assert W.enu_to_ned(W.enu_to_ned(original)) == original
    with pytest.raises(W.FramingError):
        W.enu_to_ned((1.0, 2.0))


def test_propeller_thrust_is_linearized_before_it_reaches_the_motor():
    # A propeller's thrust is quadratic in angular velocity, so a quarter of full
    # throttle is half of the motor's maximum velocity, not a quarter of it.
    assert W.propeller_velocity(1.0, max_velocity=100.0) == pytest.approx(100.0)
    assert W.propeller_velocity(0.25, max_velocity=100.0) == pytest.approx(50.0)
    assert W.propeller_velocity(-0.25, max_velocity=100.0) == pytest.approx(-50.0)
    assert W.propeller_velocity(0.0, max_velocity=100.0) == pytest.approx(0.0)

# ---------------------------------------------------------------------------
# Frame measurement and colour
# ---------------------------------------------------------------------------


def test_colour_measurement_separates_a_colour_frame_from_a_copied_channel():
    colour_left, _ = synthetic_pair()
    quality = W.measure_frame_quality(
        colour_left, 64, 48, identical_channel_fraction_limit=0.99
    )
    assert quality.is_colour
    assert quality.channel_identical_fraction < 0.99

    copied_left, _ = synthetic_pair(colourless=True)
    copied = W.measure_frame_quality(copied_left, 64, 48, identical_channel_fraction_limit=0.99)
    assert not copied.is_colour
    assert copied.channel_identical_fraction == 1.0


def test_bgra_to_rgb8_takes_each_channel_by_name():
    # Webots hands back BGRA; a blue-coded buffer must come out blue in RGB order.
    buffer = bytes([255, 0, 0, 255]) * 4  # B=255, G=0, R=0, A=255
    rgb = W.bgra_to_rgb8(buffer, 2, 2)
    assert rgb[0:3] == bytes([0, 0, 255])
    with pytest.raises(W.FramingError):
        W.bgra_to_rgb8(buffer, 3, 3)


def test_colour_witness_reports_the_dominant_channel_and_its_location():
    left, _ = synthetic_pair()
    witness = W.find_colour_witness(left, 64, 48, (217, 13, 13))
    assert witness["matched_pixels"] > 0
    assert witness["matches"] is True
    assert witness["observed_dominant_channel"] == "red"
    assert witness["centroid_px"][0] < 32  # the panel sits on the left of the frame
    swapped = W.find_colour_witness(left, 64, 48, (13, 13, 217))
    assert swapped["matches"] is False
    assert swapped["expected_dominant_channel"] == "blue"
    empty = W.find_colour_witness(bytes(64 * 48 * 3), 64, 48, (217, 13, 13))
    assert empty["matched_pixels"] == 0 and empty["matches"] is False


# ---------------------------------------------------------------------------
# The sensor gateway over a real loopback socket
# ---------------------------------------------------------------------------


def loopback_stamp():
    """The receipt stamp the gateway is given here, in one named clock domain."""
    return W.ClockStamp(
        host_id="test-host", clock_id="monotonic", monotonic_ns=time.monotonic_ns()
    )


def serve_messages(payloads, port, stop):
    """Send framed messages to one client, then hold the connection open."""

    def run():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)
        connection, _ = server.accept()
        for payload in payloads:
            connection.sendall(payload)
        while not stop.is_set():
            time.sleep(0.01)
        connection.close()
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_gateway_assembles_a_pair_from_the_wire():
    port = free_port()
    left, right = synthetic_pair()
    pair_message = W.pack_message(
        W.Kind.PAIR,
        sim_time_s=4.0,
        sequence=1,
        payload=W.encode_pair_payload(
            capture_host_ns=time.monotonic_ns(),
            pair_id=1,
            left_frame_id=1,
            right_frame_id=1,
            width=64,
            height=48,
            encoding="rgb8",
            left_bytes=left,
            right_bytes=right,
        ),
    )
    status_message = W.pack_message(
        W.Kind.STATUS, sim_time_s=4.0, sequence=0, payload=W.encode_status_payload({"ok": True})
    )
    stop = threading.Event()
    serve_messages([status_message + pair_message], port, stop)
    gateway = W.TcpSensorGateway(stamp=loopback_stamp)
    try:
        gateway.open("127.0.0.1", port, 5.0)
        status = gateway.read_record(5.0)
        assert status is not None and status.status == {"ok": True}
        record = gateway.read_record(5.0)
        assert record is not None and record.pair is not None
        assert record.pair.left_bytes == left
        assert record.pair.right_bytes == right
        assert (record.pair.left_frame_id, record.pair.right_frame_id) == (1, 1)
    finally:
        stop.set()
        gateway.close()


def test_gateway_returns_nothing_when_a_message_stops_mid_frame():
    port = free_port()
    left, right = synthetic_pair()
    pair_message = W.pack_message(
        W.Kind.PAIR,
        sim_time_s=4.0,
        sequence=1,
        payload=W.encode_pair_payload(
            capture_host_ns=time.monotonic_ns(),
            pair_id=1,
            left_frame_id=1,
            right_frame_id=1,
            width=64,
            height=48,
            encoding="rgb8",
            left_bytes=left,
            right_bytes=right,
        ),
    )
    stop = threading.Event()
    serve_messages([pair_message[: len(pair_message) // 2]], port, stop)
    gateway = W.TcpSensorGateway(stamp=loopback_stamp)
    try:
        gateway.open("127.0.0.1", port, 5.0)
        assert gateway.read_record(0.4) is None
    finally:
        stop.set()
        gateway.close()


# ---------------------------------------------------------------------------
# The controller's outgoing queue
# ---------------------------------------------------------------------------


def test_the_outgoing_queue_drops_whole_frames_and_counts_them():
    stream = W.OutboundStream(max_queued_bytes=10)
    assert stream.queue(b"12345") is True
    assert stream.queue(b"67890") is True
    assert stream.queue(b"abcde") is False  # over the bound: dropped, not truncated
    assert stream.dropped_frames == 1
    assert stream.queued_bytes == 10
    with pytest.raises(W.FramingError):
        stream.queue(b"x" * 11)  # a frame that could never fit at all is refused


def test_a_control_frame_is_not_lost_behind_a_full_bulk_queue():
    """The failure this exists to prevent: an acknowledgement dropped by pixels.

    The analysis process waits for the controller's answer to an injection, and that
    answer travels on the same connection as the camera stream. A full queue of frames
    must not be able to discard it, because a lost acknowledgement cannot be told apart
    from a command that was never applied.
    """
    stream = W.OutboundStream(max_queued_bytes=10, max_control_bytes=10)
    assert stream.queue(b"1234567890") is True
    assert stream.queue_control(b"ack") is True
    assert stream.control_queued_bytes == 3
    assert stream.dropped_control_frames == 0
    assert stream.describe()["dropped_control_frames"] == 0


def test_a_full_queue_drops_pixels_to_carry_a_control_frame():
    stream = W.OutboundStream(max_queued_bytes=10, max_control_bytes=10)
    assert stream.queue(b"1234567890") is True
    assert stream.queue(b"more") is False, "the bulk frame does not fit"
    assert stream.dropped_frames == 1
    # Room for the answer is made by evicting pixels, not by refusing the answer.
    assert stream.queue_control(b"ack") is True
    assert stream.dropped_frames == 2
    assert stream.dropped_control_frames == 0
    assert stream.queued_bytes == 3


def test_a_control_frame_never_overtakes_a_frame_that_is_half_sent():
    """A reordered frame is a stream the reader cannot parse.

    A camera frame is larger than the socket buffer, so part of it is regularly left
    queued between flushes. A control frame queued at that moment must wait for the
    rest of the frame in front of it: sending it first would put its bytes inside the
    frame the reader is still assembling, and every frame after that would be read at
    the wrong offset.
    """
    port = free_port()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    client = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    accepted, _ = server.accept()
    accepted.setblocking(False)
    client.settimeout(5.0)
    stream = W.OutboundStream(max_queued_bytes=16 << 20)
    bulk = W.pack_message(W.Kind.PAIR, sim_time_s=1.0, sequence=1, payload=b"p" * (8 << 20))
    control = W.pack_message(W.Kind.FAULT_ACK, sim_time_s=2.0, sequence=2, payload=b"ack")
    try:
        assert stream.queue(bulk) is True
        stream.flush(accepted)
        # The bulk frame is larger than the socket buffer, so it is now half sent.
        assert 0 < stream.queued_bytes < len(bulk)
        assert stream.queue_control(control) is True
        received = bytearray()
        expected = len(bulk) + len(control)
        while len(received) < expected:
            chunk = client.recv(1 << 20)
            assert chunk, "the connection closed before the whole stream arrived"
            received += chunk
            stream.flush(accepted)
        assert bytes(received) == bulk + control, "the frames must leave in order"
        assert stream.queued_bytes == 0
    finally:
        client.close()
        accepted.close()
        server.close()


def test_control_frames_are_counted_when_their_own_bound_is_full():
    stream = W.OutboundStream(max_queued_bytes=100, max_control_bytes=5)
    assert stream.queue_control(b"ack") is True
    assert stream.queue_control(b"ack") is True
    # The oldest control frame is the one evicted, so the reader gets the newest.
    assert stream.dropped_control_frames == 1
    assert stream.control_queued_bytes == 3


def test_a_frame_already_going_out_is_never_discarded_when_the_reader_pauses():
    """The failure this exists to prevent: a large frame on a non-blocking socket.

    A camera frame is about 1.8 MB and does not fit in one socket buffer. Truncating
    the part that has gone out would leave the reader with bytes it cannot
    resynchronise, so flushing keeps what is left and the reader eventually receives
    the frame whole.
    """
    port = free_port()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    client = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    accepted, _ = server.accept()
    accepted.setblocking(False)
    client.settimeout(5.0)
    stream = W.OutboundStream(max_queued_bytes=16 << 20)
    frame = W.pack_message(
        W.Kind.PAIR, sim_time_s=1.0, sequence=1, payload=b"p" * (8 << 20)
    )
    try:
        assert stream.queue(frame) is True
        stream.flush(accepted)
        assert stream.queued_bytes > 0, "the socket took a whole 8 MB frame at once"
        received = bytearray()
        while len(received) < len(frame):
            received += client.recv(1 << 20)
            stream.flush(accepted)
        assert bytes(received) == frame
        assert stream.queued_bytes == 0
    finally:
        client.close()
        accepted.close()
        server.close()


# ---------------------------------------------------------------------------
# Telemetry and clock joining
# ---------------------------------------------------------------------------


def test_the_boot_clock_is_read_from_whichever_message_carries_it():
    # AUTOPILOT_VERSION answers once when asked, so a join that depended on it alone
    # would have almost no samples. The boot clock travels on the streamed messages.
    messages = [
        {
            "mavpackettype": "HEARTBEAT",
            "custom_mode": 4,
            "base_mode": 209,
            "system_status": 4,
        },
        {
            "mavpackettype": "ATTITUDE",
            "time_boot_ms": 4242,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
    ]
    sample = W.decode_telemetry(
        messages, stamp=W.ClockStamp(host_id="h", clock_id="c", monotonic_ns=1)
    )
    assert sample.boot_time_ms == 4242
    assert sample.autopilot_version is None


def test_the_firmware_identity_writes_its_version_blobs_as_hex():
    messages = [
        {
            "mavpackettype": "AUTOPILOT_VERSION",
            "flight_sw_version": 262144,
            "flight_custom_version": [1, 2, 3, 4, 5, 6, 7, 8],
            "uid": [0, 1, 255],
            "vendor_id": 3,
        }
    ]
    sample = W.decode_telemetry(
        messages, stamp=W.ClockStamp(host_id="h", clock_id="c", monotonic_ns=1)
    )
    identity = W.firmware_identity(sample)
    assert identity["flight_custom_version"] == "0102030405060708"
    assert identity["uid"] == "0001ff"
    assert identity["vendor_id"] == 3
    assert W.firmware_identity(None) is None


def test_a_receipt_stamp_is_the_arrival_of_the_message_that_carried_the_clock(tmp_path):
    """A batch's folded time is not the same fact as a message's arrival time.

    The timebase join pairs the autopilot's clock with the host time that message
    arrived. Stamping the sample when the whole batch had been read and logged would
    fold this program's own polling period into the join, and the run would report a
    scatter between two clocks that is really the latency of reading them.
    """
    clock = FakeClock()
    settings = settings_for(write_scene(tmp_path), tmp_path)

    class Session:
        def drain(self):
            return [
                {
                    "mavpackettype": "ATTITUDE",
                    "time_boot_ms": 1000,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    W.RECEIVED_AT_KEY: 5_000_000_000,
                },
                {
                    "mavpackettype": "HEARTBEAT",
                    "custom_mode": 4,
                    "base_mode": 209,
                    "system_status": 4,
                    W.RECEIVED_AT_KEY: 5_090_000_000,
                },
            ]

        def close(self):
            return None

    adapter = W.WebotsArduPilot(
        settings,
        runner=FakeRunner(),
        session=Session(),
        gateway=FakeGateway(clock),
        evidence=W.EvidenceWriter(tmp_path / "out", "run-a"),
        label="run-a",
        monotonic_ns=clock.monotonic_ns,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    sample = adapter.telemetry()
    assert sample.boot_time_ms == 1000
    assert sample.received_stamp.monotonic_ns == 5_000_000_000, (
        "the stamp must be the arrival of the ATTITUDE that carries the clock, not the "
        "later heartbeat in the same batch"
    )


def test_the_timebase_evidence_keeps_its_samples_and_its_resolution():
    samples = [(1000.0, 10.0), (1001.0, 11.0), (1002.5, 12.5)]
    join = W.join_timebase(samples, minimum_span_s=0.5)
    evidence = W.timebase_evidence(
        join, samples, poll_period_s=0.005, host_clock="test-host/monotonic"
    )
    assert evidence["measured"] is True
    assert evidence["poll_period_s"] == 0.005
    assert evidence["samples_host_s_device_s"] == [list(pair) for pair in samples]
    assert evidence["host_seconds_per_device_second"] == pytest.approx(1.0, abs=1e-6)
    assert "poll period" in evidence["resolution_note"]


def test_the_scene_assets_are_read_from_the_world_and_hashed():
    assets = W.scenario_proto_assets(SCENE / "worlds" / "compat_stereo.wbt")
    protos = [asset for asset in assets if asset["role"] == "proto"]
    meshes = [asset for asset in assets if asset["role"] == "mesh"]
    assert [Path(asset["path"]).name for asset in protos] == ["Iris.proto"]
    assert {Path(asset["path"]).name for asset in meshes} == {
        "iris.dae",
        "iris_prop_ccw.dae",
        "iris_prop_cw.dae",
    }
    assert all(asset["sha256"] for asset in assets)
    assert W.scenario_proto_assets(Path("/nonexistent/absent.wbt")) == []


# ---------------------------------------------------------------------------
# Telemetry and clock joining
# ---------------------------------------------------------------------------


def test_telemetry_is_decoded_from_recorded_messages():
    messages = [json.loads(line) for line in (FIXTURES / "telemetry_guided.jsonl").read_text().splitlines() if line]
    stamp = W.ClockStamp(host_id="host", clock_id="monotonic", monotonic_ns=1)
    sample = W.decode_telemetry(messages, stamp=stamp)
    assert sample.mode_name == "GUIDED"
    assert sample.custom_mode == 4
    assert sample.armed is True
    assert sample.local_position_ned == (2.0, 1.0, -1.5)
    assert sample.servo_outputs[:4] == (1500, 1500, 1500, 1500)
    assert sample.ekf_pos_horiz_variance == 0.42
    assert sample.boot_time_ms == 123456
    assert sample.autopilot_version["vendor_id"] == 3
    assert "guided" in sample.statustexts[-1]
    # A second batch keeps the last known values instead of inventing new ones.
    again = W.decode_telemetry([], stamp=stamp, previous=sample)
    assert again.mode_name == "GUIDED"
    assert again.messages_seen == 0


def test_timebase_join_recovers_a_synthetic_offset_and_drift():
    # The device clock runs about 100 ppm slow against the host clock.
    samples = [(1000.0 + index * 0.1, 100.0 + index * 0.09999) for index in range(20)]
    join = W.join_timebase(samples, minimum_span_s=0.5)
    assert join.measured
    # host = offset + scale * boot, so an offset near +900 s is expected here.
    assert join.offset_s == pytest.approx(900.0, abs=0.05)
    assert join.drift_ppm == pytest.approx(100.0, abs=20.0)
    assert join.spread_ms < 1.0


def test_a_wide_spread_is_recorded_rather_than_thresholded():
    """The spread is the error bar on the relation, not a pass/fail of its own.

    Under a simulator the residual spread is the simulator advancing in bursts against
    the host clock, and no threshold on one run separates that from a broken join. What
    the run declares instead is the rate the simulator held, window by window, against
    the envelope the configuration declares; the spread stays in the evidence with the
    samples behind it.
    """
    samples = [
        (1000.0 + index * 0.1, 100.0 + index * 0.1 + (0.5 if index % 2 else 0.0))
        for index in range(10)
    ]
    join = W.join_timebase(samples, minimum_span_s=0.5)
    assert join.measured is True
    assert join.spread_ms > 400.0
    assert join.reason is None


def test_fitting_a_timebase_needs_two_samples_and_a_moving_clock():
    with pytest.raises(W.ProbeFailure):
        W.fit_timebase([(1.0, 1.0)])
    with pytest.raises(W.ProbeFailure):
        W.fit_timebase([(1.0, 5.0), (2.0, 5.0)])


# ---------------------------------------------------------------------------
# The checklist on scripted seams
# ---------------------------------------------------------------------------


def test_the_probe_passes_every_item_with_a_scripted_vehicle(tmp_path):
    result, probe, gateway, sessions, runner = run_probe(tmp_path)
    assert result.gate_status is cli.GateStatus.PASS, result.reasons
    assert result.failed == ()
    names = {entry.name for entry in result.checks}
    for expected in (
        "1_startup_and_transport[run-a]",
        "2_frames_and_timebases",
        "3_guided_local_ned_motion[run-a]",
        "4_actuator_mapping[run-a]",
        "5_stereo_colour_pairs[run-a]",
        "6_imu_stream[run-a]",
        "7_setpoint_stream_loss[run-a]",
        "8_estimator_health_loss[run-a]",
        "1_startup_and_transport[run-b]",
        "8_estimator_health_loss[run-b]",
    ):
        assert expected in names, sorted(names)

    # Every step's raw material is on disk before it is judged.
    for artifact in (
        "prerequisites.json",
        "checks.json",
        "run-a/startup.json",
        "run-a/frames-and-timebases.json",
        "run-a/motion.jsonl",
        "run-a/pairs.jsonl",
        "run-a/imu.json",
        "run-a/stereo.json",
        "run-a/actuators.json",
        "run-a/stream-loss.json",
        "run-a/estimator.json",
        "run-b/estimator.json",
        "run-a/webots.log",
        "run-a/sitl.log",
        "run-a/mavlink.jsonl",
    ):
        assert artifact in result.artifacts, artifact
        assert (tmp_path / "out" / artifact).is_file(), artifact

    # Every guided flight was commanded with the mask the autopilot reads.
    frames = check(result, "2_frames_and_timebases")
    assert frames.evidence["realtime"]["timing_valid"] is True
    assert frames.evidence["realtime"]["windows"]
    assert frames.evidence["host_to_autopilot"]["measured"] is True

    # A stereo pair became a shared record, and its pixels are stored beside it.
    stereo = json.loads((tmp_path / "out" / "run-a" / "stereo.json").read_text())
    assert stereo["observations"]
    observation_document = stereo["observations"][0]
    assert observation_document["encoding"] == "rgb8"
    assert observation_document["calibration_id"] == "compat-stereo-1"
    assert observation_document["capture_stamp"]["monotonic_ns"] > 0
    assert (tmp_path / "out" / observation_document["left_payload"]).is_file()
    assert (tmp_path / "out" / observation_document["right_payload"]).is_file()

    # The manifest states what ran, not what it concluded.
    firmware = result.manifest["run_a"]["autopilot"]["firmware"]
    assert firmware["vendor_id"] == 3
    assert firmware["flight_custom_version"] == "0102030405060708"
    assert result.manifest["run_a"]["webots"]["version"] == "R2025a"
    assert result.manifest["run_a"]["sensors"]["stereo"]["baseline_m"] == 0.1
    assert result.manifest["run_a"]["ports"]["controller_port"] > 0
    assert [
        Path(name).name for name in result.manifest["run_a"]["autopilot"]["parameter_files"]
    ] == ["compat_base.parm", "compat_arming.parm", "compat_ekf.parm"]
    assert Path(result.manifest["run_b"]["autopilot"]["parameter_files"][-1]).name == (
        "compat_ekf.parm"
    )
    assert result.manifest["run_a"]["realtime"]["timing_valid"] is True

    # The scene's own files are named and hashed, so a run can be compared with the
    # revision of the world and the proto that produced it.
    protos = result.manifest["run_a"]["assets"]["protos"]
    assert any(Path(asset["path"]).name == "Iris.proto" for asset in protos)
    assert len([asset for asset in protos if asset["role"] == "mesh"]) == 3
    assert all(asset["sha256"] for asset in protos)

    # The firmware identity answers one request, not a stream.
    assert all(W.MSG_ID_AUTOPILOT_VERSION in session.requested_messages for session in sessions)

    # Both runs stopped both children, and recorded their exit codes.
    assert runner.terminated == ["webots", "sitl", "webots", "sitl"]
    assert result.manifest["run_a"]["shutdown"]["exits"] == {"webots": 0, "sitl": 0}
    assert result.manifest["run_b"]["shutdown"]["exits"] == {"webots": 0, "sitl": 0}


def test_motion_is_published_as_a_guided_setpoint_and_never_as_a_motor_command(tmp_path):
    _, _, _, sessions, _ = run_probe(tmp_path)
    published = [setpoint for session in sessions for setpoint in session.sent]
    assert published, "no setpoint was published"
    for setpoint in published:
        assert setpoint.frame is R.Frame.ODOM
        assert setpoint.type_mask == R.TYPE_MASK_POSITION_VELOCITY
        # No force target is claimed, so its bit stays clear.
        assert not setpoint.type_mask & R.TYPE_MASK_FORCE_SET
        assert setpoint.target.position_ned is not None
        assert setpoint.target.velocity_ned is not None
    commands = [command for session in sessions for command in session.commands]
    assert {command[0] for command in commands} <= {"set_mode", "arm", "takeoff"}
    assert ("set_mode", "GUIDED") in commands


def test_the_gate_fails_when_the_stream_carries_no_colour_but_the_run_completes(tmp_path):
    result, _, _, _, _ = run_probe(tmp_path, gateway_kwargs={"colourless": True})
    assert result.gate_status is cli.GateStatus.FAIL
    stereo = check(result, "5_stereo_colour_pairs")
    assert stereo.status == "fail"
    assert "no colour information" in stereo.reason
    assert stereo.evidence["pairs_captured"] >= 2


def test_unmapped_actuation_is_reported(tmp_path):
    result, _, _, _, _ = run_probe(tmp_path, session_kwargs={"dead_servos": True})
    actuators = check(result, "4_actuator_mapping")
    assert actuators.status == "fail"
    assert "unmapped actuation" in actuators.reason
    assert actuators.evidence["channel_outputs"][:4] == [1000, 1000, 1000, 1000]


def test_a_refused_guided_mode_fails_the_motion_item_and_publishes_nothing(tmp_path):
    result, _, _, sessions, _ = run_probe(tmp_path, session_kwargs={"refuse_guided": True})
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "fail"
    assert "did not enter Guided flight" in motion.reason
    assert motion.evidence["mode_reached"] is False
    assert all(not session.sent for session in sessions)


def test_a_vehicle_that_never_moves_fails_the_motion_item(tmp_path):
    result, _, _, _, _ = run_probe(tmp_path, session_kwargs={"moves": False})
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "fail"
    assert "did not move" in motion.reason


def test_motion_opposite_to_the_command_fails_the_motion_item(tmp_path):
    result, _, _, _, _ = run_probe(tmp_path, session_kwargs={"direction": -1.0})
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "fail"
    assert "target" in motion.reason and "start" in motion.reason and "(moved" in motion.reason


class OvershootThenDescend(ScriptedMavlinkSession):
    """A climb that overshoots the commanded hover, then correct flight.

    Iteration 5 run-b's start: takeoff is commanded to the 1.5 m hover, the climb
    overshoots, and waypoint 1's start sample catches the vehicle 0.7 m above the
    altitude it is about to be commanded to hold. Descending to that target is
    then compliance with the command; the target coordinate's negative sign says
    where the target sits, not which way the command asks the vehicle to move.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.climb_overshoot_m = 0.7
        self.holding_overshoot = False

    def takeoff(self, altitude_m):
        super().takeoff(altitude_m)
        self.position = (0.0, 0.0, -altitude_m - self.climb_overshoot_m)
        self.holding_overshoot = True

    def send_setpoint(self, setpoint):
        super().send_setpoint(setpoint)
        self.holding_overshoot = False

    def _advance(self):
        if self.holding_overshoot or not self.moves or self.target is None:
            return
        # Pinned, not drain-timed: run-b held both absolute targets to 2-5 cm, so
        # the arithmetic under test wants start and end positions it can read
        # exact values from whatever the drain cadence was.
        self.position = self.target


class ClimbingWhenToldToDescend(OvershootThenDescend):
    """The same overshoot, then a vehicle that climbs away from its target.

    The command is a descent (the vehicle sits above the target altitude) and
    the motion is a climb: the rule must still bite when its direction
    reference is the commanded step.
    """

    def _advance(self):
        if self.holding_overshoot or not self.moves or self.target is None:
            return
        self.position = (
            self.target[0],
            self.target[1],
            self.target[2] - 2 * self.climb_overshoot_m,
        )


def test_a_descent_to_the_commanded_altitude_is_not_opposite_motion(tmp_path):
    """Iteration 5 run-b's waypoint record, scripted.

    The climb overshoots, so waypoint 1's start sample sits 0.7 m above the
    target altitude and the correct flight to it is a descent: displacement
    +0.7 m on an axis whose commanded coordinate is -1.5 m. Judged against the
    absolute coordinate that read as opposite motion; judged against the
    commanded step it is exactly the motion that was asked for.
    """
    result, _, _, _, _ = run_probe(tmp_path, session_class=OvershootThenDescend)
    for name in ("3_guided_local_ned_motion[run-a]", "3_guided_local_ned_motion[run-b]"):
        motion = next(entry for entry in result.checks if entry.name == name)
        assert motion.status == "pass", motion.reason
        first = motion.evidence["waypoints"][0]
        assert first["position_before_ned"][2] == pytest.approx(-2.2)
        assert first["displacement_ned"][2] == pytest.approx(0.7)


def test_a_climb_against_a_commanded_descent_still_fails_the_motion_item(tmp_path):
    """The corrected rule still bites: opposite to the commanded step is flagged.

    The command is a descent of 0.7 m and the vehicle climbs 0.7 m instead; the
    finding names the commanded step, which is the reference it judged against.
    """
    result, _, _, _, _ = run_probe(tmp_path, session_class=ClimbingWhenToldToDescend)
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "fail"
    assert "axis 2" in motion.reason
    assert "commanded step +0.70" in motion.reason


def test_a_mid_hold_loss_is_reported_as_the_loss_when_the_motion_followed_the_command(
    tmp_path,
):
    """A lost flight is reported lost, whatever its measured motion said.

    The autopilot leaves Guided mid-hold after two publications; until then the
    vehicle moved toward the commanded target. The record carries the loss
    alone: no axis-direction finding is invented for motion that followed the
    command.
    """
    result, _, _, _, _ = run_probe(
        tmp_path, session_kwargs={"failsafe_after_setpoints": 2}
    )
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "fail"
    assert "Guided flight was lost" in motion.reason
    waypoints = motion.evidence["waypoints"]
    assert waypoints and waypoints[0]["displacement_ned"] is not None
    assert "axis" not in motion.reason


def test_a_run_outside_its_declared_real_time_envelope_is_timing_invalid(tmp_path):
    """The declared envelope is the criterion, and the configuration carries the value.

    Specification section 14.4 asks a run to admit only a declared real-time-ratio
    envelope; a window outside it means the recorded simulator times do not describe the
    environment the scenario declares, and the run reports that rather than being judged
    as if its timing were what it claimed.
    """
    result, _, _, _, _ = run_probe(
        tmp_path, config_overrides={"probe": {"realtime_ratio_envelope": [2.0, 3.0]}}
    )
    frames = check(result, "2_frames_and_timebases")
    assert frames.status == "fail"
    assert "timing-invalid" in frames.reason
    assert frames.evidence["realtime"]["timing_valid"] is False
    assert frames.evidence["realtime"]["windows_outside"]


def test_a_noisy_autopilot_clock_is_recorded_rather_than_failed(tmp_path):
    result, _, _, _, _ = run_probe(tmp_path, session_kwargs={"boot_jitter_s": 0.5})
    frames = check(result, "2_frames_and_timebases")
    assert frames.evidence["host_to_autopilot"]["measured"] is True
    assert frames.evidence["host_to_autopilot"]["spread_ms"] > 60.0


def test_a_join_is_refused_when_the_samples_cannot_resolve_a_window():
    """A fit whose whole span is below the window is not a measurement of the rate.

    Five samples taken inside one brief burst barely move the host clock while the device
    clock jumps up and down: the line absorbs the jumps into a nonsense slope and its
    residuals look small. Reporting that as a relation between two clocks would be a
    measurement claiming a resolution it never had.
    """
    samples = [
        (1000.000, 900.9),
        (1000.010, 899.9),
        (1000.020, 900.9),
        (1000.030, 899.9),
        (1000.040, 900.9),
    ]
    join = W.join_timebase(samples, minimum_span_s=0.5)
    assert join.measured is False
    assert "the host clock advanced only" in join.reason


def test_a_measured_join_reports_the_spread_it_saw():
    samples = [(1000.0 + index, 900.0 + index) for index in range(40)]
    join = W.join_timebase(samples, minimum_span_s=0.5)
    assert join.measured is True
    assert join.spread_ms == pytest.approx(0.0, abs=1e-6)


def test_publication_is_recorded_and_adoption_is_left_to_telemetry(tmp_path):
    result, _, _, sessions, _ = run_probe(tmp_path)
    motion = check(result, "3_guided_local_ned_motion")
    assert "adoption is judged from the telemetry" in motion.evidence["adoption"]
    first_waypoint = motion.evidence["waypoints"][0]
    assert first_waypoint["setpoint_type_mask"] == R.TYPE_MASK_POSITION_VELOCITY
    assert first_waypoint["position_before_ned"] is not None
    assert first_waypoint["displacement_ned"] is not None
    # The publication stamp is in the run's evidence and matches a real publication.
    publication = next(
        setpoint
        for setpoint in sessions[0].sent
        if setpoint.command_sequence == first_waypoint["setpoint_sequence"]
    )
    assert first_waypoint["published_at_monotonic_ns"] == publication.issue_stamp.monotonic_ns


def test_an_injection_is_not_reported_as_applied_until_the_controller_acknowledges(tmp_path):
    result, _, gateway, _, _ = run_probe(tmp_path, gateway_kwargs={"acknowledge": False})
    estimator = check(result, "8_estimator_health_loss")
    assert estimator.status == "fail"
    assert "requested but not applied" in estimator.reason
    assert estimator.evidence["fault"]["requested"].endswith("position_step")
    assert gateway.faults, "the injection was not even sent"

    acknowledged, _, _, _, _ = run_probe(tmp_path, output_name="out-ack")
    applied = check(acknowledged, "8_estimator_health_loss")
    assert applied.status == "pass"
    assert applied.evidence["fault"]["applied"] is True
    assert applied.evidence["observable_signal"] is True


def test_the_estimator_item_is_measured_wherever_the_files_select_an_ekf(tmp_path):
    """The item runs where an estimator is running, decided from the files that were applied.

    A run cannot claim an estimator it is not flying: applicability comes from the
    AHRS_EKF_TYPE the parameter files select, so a configuration that selects the
    simulator's own state records why the item does not apply instead of degrading nothing
    and reporting that as a measurement.
    """
    result, _, _, _, _ = run_probe(tmp_path)
    measured = {entry.name for entry in result.checks}
    assert "8_estimator_health_loss[run-a]" in measured
    assert "8_estimator_health_loss[run-b]" in measured
    assert not (tmp_path / "out" / "run-a" / "estimator-not-applicable.json").exists()

    pinned, _, _, _, _ = run_probe(
        tmp_path,
        output_name="out-pinned",
        config_overrides={
            "scenario": {"estimator_params": [str(SCENE / "params" / "compat_base.parm")]}
        },
    )
    reported = {entry.name for entry in pinned.checks}
    assert not any(name.startswith("8_estimator_health_loss") for name in reported)
    declaration = json.loads(
        (tmp_path / "out-pinned" / "run-a" / "estimator-not-applicable.json").read_text()
    )
    assert declaration["applicable"] is False
    assert declaration["selected_estimator"] == 10.0
    assert any(
        "AHRS_EKF_TYPE" in entry["line"] for entry in declaration["parameter_evidence"]
    )


def test_the_scene_controller_only_uses_shared_names_the_adapter_exposes():
    """The scene's controller reaches the adapter by name, from Webots' own interpreter.

    Nothing here imports the controller — it needs Webots' ``controller`` module — so a
    rename in the adapter is otherwise discovered by starting a whole live run, which is
    how the ``SimFdmState`` rename broke one. The names it uses are read from its source
    instead, so a rename that misses it fails a test rather than a run.
    """
    controller_dir = SCENE / "controllers" / "compat_vehicle_controller"
    used: set[str] = set()
    for path in sorted(controller_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "SHARED"
            ):
                used.add(node.attr)
        for name in re.findall(
            r"from embodied\.platform\.webots_ardupilot import ([A-Za-z_][\w, ]*)", source
        ):
            used.update(part.strip() for part in name.split(",") if part.strip())
    assert used, "the controller no longer reaches the shared module"
    missing = sorted(name for name in used if not hasattr(W, name))
    assert missing == [], f"the adapter does not expose {missing}"

def test_stream_loss_reports_what_the_telemetry_showed(tmp_path):
    result, _, _, _, _ = run_probe(tmp_path)
    loss = check(result, "7_setpoint_stream_loss")
    assert loss.status == "pass"
    assert loss.evidence["samples_in_window"]
    assert loss.evidence["last_publication"]["type_mask"] == R.TYPE_MASK_POSITION_VELOCITY
    assert loss.evidence["control_regained"] is False
    modes = {sample["mode_name"] for sample in loss.evidence["samples_in_window"]}
    assert "LOITER" in modes  # observed in telemetry, not asserted as safe behaviour
    assert "reported from its own telemetry" in loss.evidence["observed_not_designed"]
    # The autopilot left Guided on its own, so publishing after the gap was refused
    # with the mode that caused the refusal rather than sent anyway.
    assert loss.evidence["resume_refused"] is not None
    assert loss.evidence["resume_refused"]["observed_mode"] == "LOITER"


def test_publication_resumes_when_the_autopilot_stays_in_guided(tmp_path):
    result, _, _, sessions, _ = run_probe(tmp_path, session_kwargs={"guided_timeout_s": 600.0})
    loss = check(result, "7_setpoint_stream_loss")
    assert loss.status == "pass"
    assert loss.evidence["resume_refused"] is None
    assert loss.evidence["control_regained"] is True
    # The only recorded changes leave the aircraft under Guided control: taking control
    # at the start of the run, and nothing that took it away again.
    assert all(event["guidance_held"] for event in loss.evidence["control_events"])
    assert any(session.sent for session in sessions)


def test_a_mode_change_stops_the_adapter_assuming_control(tmp_path):
    result, _, _, sessions, _ = run_probe(
        tmp_path, session_kwargs={"failsafe_after_setpoints": 2}
    )
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "fail"
    assert "Guided flight was lost" in motion.reason
    assert motion.evidence["refused_publications"] >= 1
    events = motion.evidence["control_events"]
    assert any(event["to_mode"] == "LOITER" and event["guidance_held"] is False for event in events)
    # Publication stopped at the change: exactly the two setpoints published before the
    # autopilot left Guided, and nothing afterwards.
    assert len(sessions[0].sent) == 2
    loss = check(result, "7_setpoint_stream_loss")
    assert loss.evidence["resume_refused"] is not None


def test_a_step_that_cannot_produce_evidence_still_stops_both_children(tmp_path):
    clock = FakeClock()
    gateway = FakeGateway(clock)
    runner = FakeRunner()

    class Exploding(ScriptedMavlinkSession):
        def drain(self):
            raise W.ProbeFailure("the autopilot stream died mid-step")

    config_path = write_scene(tmp_path)
    settings = settings_for(config_path, tmp_path)
    probe = W.CompatibilityProbe(
        settings,
        output_dir=tmp_path / "out",
        runner_factory=lambda: runner,
        session_factory=lambda: Exploding(clock),
        gateway_factory=lambda: gateway,
        monotonic_ns=clock.monotonic_ns,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    with pytest.raises(W.ProbeFailure):
        probe.run()
    assert runner.terminated == ["webots", "sitl"]
    assert gateway.closed


def test_prerequisites_are_reported_before_any_process_starts(tmp_path):
    config_path = write_scene(
        tmp_path,
        platform={
            "webots_home": str(tmp_path / "absent-Webots.app"),
            "ardupilot_commit": "0" * 40,
        },
    )
    settings = settings_for(config_path, tmp_path)
    checks = W.check_prerequisites(settings, tmp_path / "out")
    unsatisfied = {entry.name for entry in checks if not entry.satisfied}
    assert "webots_application" in unsatisfied
    assert "webots_version" in unsatisfied
    assert "autopilot_commit" in unsatisfied

    runner = FakeRunner()
    probe = W.CompatibilityProbe(
        settings,
        output_dir=tmp_path / "out",
        runner_factory=lambda: runner,
        session_factory=lambda: ScriptedMavlinkSession(FakeClock()),
        gateway_factory=lambda: FakeGateway(FakeClock()),
    )
    with pytest.raises(W.PlatformUnavailable) as failure:
        probe.run()
    assert "webots_application" in str(failure.value)
    assert "autopilot_commit" in str(failure.value)
    assert runner.spawned == []


def test_the_ready_system_states_are_mavlink_s_own():
    """The numbers that decide readiness come from the dialect, not from memory.

    Reading MAV_STATE's order wrong would exclude a landed, ready vehicle from
    readiness and stop every run before it measured anything.
    """
    from pymavlink.dialects.v20 import ardupilotmega as dialect

    assert W.READY_SYSTEM_STATUSES == (
        dialect.MAV_STATE_STANDBY,
        dialect.MAV_STATE_ACTIVE,
    )


def test_the_published_setpoint_is_a_position_and_velocity_target_the_autopilot_accepts(
    tmp_path,
):
    """The mask is MAVLink's own, and the autopilot reads it in field groups.

    Pinned against the dialect, because the numbers are the dialect's, and against the
    way the receiving end reads them: the autopilot treats a group as ignored when *any*
    bit in that group is set, so a mask naming part of a group is not a partial
    instruction — it is a different one. A setpoint that carries a position and a
    velocity therefore has to clear both of those groups, and a mask that threw a
    carried field away would be refused when the record is built.
    """
    from pymavlink.dialects.v20 import ardupilotmega as dialect

    position_group = (
        dialect.POSITION_TARGET_TYPEMASK_X_IGNORE
        | dialect.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | dialect.POSITION_TARGET_TYPEMASK_Z_IGNORE
    )
    velocity_group = (
        dialect.POSITION_TARGET_TYPEMASK_VX_IGNORE
        | dialect.POSITION_TARGET_TYPEMASK_VY_IGNORE
        | dialect.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    )
    acceleration_group = (
        dialect.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | dialect.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | dialect.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    )
    assert R.TYPE_MASK_POSITION_IGNORE == position_group
    assert R.TYPE_MASK_VELOCITY_IGNORE == velocity_group
    assert R.TYPE_MASK_ACCELERATION_IGNORE == acceleration_group
    assert R.TYPE_MASK_FORCE_SET == dialect.POSITION_TARGET_TYPEMASK_FORCE_SET
    assert R.TYPE_MASK_YAW_IGNORE == dialect.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    assert R.TYPE_MASK_YAW_RATE_IGNORE == dialect.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE

    _, _, _, sessions, _ = run_probe(tmp_path)
    published = [setpoint for session in sessions for setpoint in session.sent]
    assert published, "no setpoint was published"
    for setpoint in published:
        mask = setpoint.type_mask
        assert not mask & position_group, "the autopilot would ignore the position target"
        assert not mask & velocity_group, "the autopilot would ignore the velocity target"
        assert mask & acceleration_group == acceleration_group
        assert mask & dialect.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        assert mask & dialect.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        assert not mask & dialect.POSITION_TARGET_TYPEMASK_FORCE_SET


def test_the_probe_waits_for_a_booted_autopilot_not_just_a_heartbeat(tmp_path):
    """A heartbeat arrives while ArduPilot is still initialising.

    Measuring telemetry, arming and setpoints against a vehicle in MAV_STATE_BOOT
    would report a broken transport for a vehicle that had simply not finished
    booting, so readiness has to wait for the vehicle.
    """

    class StillBooting(ScriptedMavlinkSession):
        def drain(self):
            messages = super().drain()
            for message in messages:
                if message["mavpackettype"] == "HEARTBEAT":
                    message["system_status"] = 1  # MAV_STATE_BOOT
            return messages

    clock = FakeClock()
    config_path = write_scene(
        tmp_path, probe={"step_timeout_s": {"startup": 1.0, "ready": 30, "flight": 60}}
    )
    settings = settings_for(config_path, tmp_path)
    probe = W.CompatibilityProbe(
        settings,
        output_dir=tmp_path / "out",
        runner_factory=FakeRunner,
        session_factory=lambda: StillBooting(clock),
        gateway_factory=lambda: FakeGateway(clock),
        monotonic_ns=clock.monotonic_ns,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    with pytest.raises(W.ProbeFailure) as failure:
        probe.run()
    assert "system_status=1" in str(failure.value)
    assert "boot_time_ms" in str(failure.value)


def test_the_at_rest_measurement_waits_for_the_scene_to_settle(tmp_path):
    result, _, _, _, _ = run_probe(
        tmp_path, config_overrides={"probe": {"settle_s": 0.5, "at_rest_window_s": 0.4}}
    )
    imu = check(result, "6_imu_stream")
    assert imu.status == "pass", imu.reason
    assert imu.evidence["settle_s"] == 0.5
    assert imu.evidence["pre_settle_samples_discarded"] > 0
    assert imu.evidence["at_rest_window_start_sim_time_s"] >= 0.5
    assert imu.evidence["at_rest_window_sim_time_s"] >= 0.4
    assert imu.evidence["non_finite_samples"] == 0


def test_an_arming_refusal_ends_the_wait_instead_of_polling_it(tmp_path):
    clock = FakeClock()
    started = clock.monotonic()
    result, _, _, _, _ = run_probe(
        tmp_path,
        clock=clock,
        session_kwargs={"refuse_guided": True},
        config_overrides={
            "probe": {"step_timeout_s": {"startup": 30, "ready": 30, "flight": 600}}
        },
    )
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "fail"
    # The flight timeout is 600 s of the fake clock; a refused check keeps refusing,
    # so the wait has to end long before it.
    assert clock.monotonic() - started < 60.0


def test_the_launch_commands_carry_the_wipe_and_the_parameter_layers(tmp_path):
    """One --defaults argument, comma separated: a repeated flag replaces the earlier one.

    Passing the files as separate flags silently applies only the last of them, and the
    vehicle then runs on firmware defaults while the receipt still names every file.
    """
    settings = settings_for(write_scene(tmp_path), tmp_path)
    argv = settings.sitl_argv(settings.estimator_params)
    assert "--wipe" in argv
    assert argv.count("--defaults") == 1
    layer = argv[argv.index("--defaults") + 1]
    assert [Path(name).name for name in layer.split(",")] == [
        "compat_base.parm",
        "compat_arming.parm",
        "compat_ekf.parm",
    ]


def test_the_flown_yaw_pairing_is_the_firmware_stock_paired_with_the_flown_rate_loop():
    """No control axis flies at a value its own firmware documents as out of range.

    The pin's yaw gains survived every provenance restoration this stage made: the
    roll/pitch rate pairing came from airsim-quadX.parm (which sets no yaw gains at
    all) and the roll/pitch angle P returned to the firmware stock 4.5 in iteration
    7, leaving yaw alone at ATC_ANG_YAW_P 0.5 and ATC_RAT_YAW_P 0.02. Iteration 7's
    own flights then departed yaw-first (run-a's yaw crossed 0.15 rad 0.76 s before
    pitch and 0.95 s before roll, with both runs sustaining 180-280 degrees/s yaw
    spins into the tumble and run-b's EKF3 running an in-flight yaw realignment
    seconds before the crash disarm), so the values below are pinned as literals
    with their sources rather than left to the pin's vestigial context.
    """
    layered = W.read_configured_parameters(
        [SCENE / "params" / name for name in
         ("compat_base.parm", "compat_arming.parm", "compat_ekf.parm")]
    )
    # Angle P, all three axes, the firmware's own stock: AC_ATTITUDE_CONTROL_ANGLE_P
    # 4.5f, "default angle P gain for roll, pitch and yaw", applied to roll, pitch
    # and yaw at work/ardupilot af852591, libraries/AC_AttitudeControl/
    # AC_AttitudeControl.h:15,51-53. The parameter's documented range is
    # 3.000-12.000 (AC_AttitudeControl.cpp:62-85); the pin's 0.5 is below it.
    assert layered["ATC_ANG_RLL_P"] == 4.5
    assert layered["ATC_ANG_PIT_P"] == 4.5
    assert layered["ATC_ANG_YAW_P"] == 4.5
    # Yaw rate P at the firmware stock for this frame class:
    # AC_ATC_MULTI_RATE_YAW_P 0.180f (AC_AttitudeControl_Multi.h:26). The pin's
    # 0.02 is below the parameter's own documented range floor of 0.10
    # (AC_AttitudeControl_Multi.cpp:208-213).
    assert layered["ATC_RAT_YAW_P"] == 0.18
    # The yaw integral is declared by no parameter file: the pin never set it, and
    # the 0.02 iteration 7's autopilot reported is the firmware's own default --
    # within one increment of the stock 0.018 (AC_AttitudeControl_Multi.h:29).
    # Nothing to restore, so nothing is declared.
    assert "ATC_RAT_YAW_I" not in layered


def test_quad_yaw_torque_points_the_same_way_as_the_firmware_mixer():
    """The 3fb6564 yaw gains must be negative feedback, not positive feedback.

    The pinned firmware's quad X yaw factors are (+1,+1,-1,-1) for motors
    1..4 (work/ardupilot/libraries/AP_Motors/AP_MotorsMatrix.cpp:592-600).
    Webots R2025a Propeller applies Q=q1*abs(omega)*omega along +Z; its Motor
    multiplier changes omega's sign. The bridge converts Webots +Z yaw to
    ArduPilot -Z yaw. Flight 8 observed the former opposite sign directly:
    run-a boot 51.83 s, yaw mix -375 us and yaw rate +1.693 rad/s.
    """
    scene = (SCENE / "protos" / "Iris.proto").read_text()
    motors = [int(i) for i in re.findall(r'name "m([1-4])_motor"', scene)]
    torques = [float(q) for q in re.findall(r"torqueConstants\s+([^\s]+)\s+0", scene)]
    multipliers = [int(m) for m in re.findall(r"multiplier\s+(-?1)\b", scene)]
    assert motors == [1, 2, 3, 4]
    assert multipliers == [1, 1, -1, -1]
    assert [-q * m for q, m in zip(torques, multipliers)] == [
        0.0005, 0.0005, -0.0005, -0.0005
    ]


def test_the_probe_requests_control_again_while_the_vehicle_refuses(tmp_path):
    """A vehicle that has just booted is still waiting for its GPS, home and IMU.

    Its refusals are not a list of what is missing — ArduPilot stops repeating a check
    whether it cleared or not — so the probe asks again and reads each answer.
    """
    result, _, _, _, _ = run_probe(
        tmp_path,
        session_kwargs={"pre_arm_failures": 2},
        config_overrides={"probe": {"pre_arm_wait_s": 60.0}},
    )
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "pass", motion.reason
    assert motion.evidence["control_events"]
    flight = json.loads((tmp_path / "out" / "run-a" / "flight-state.json").read_text())
    assert flight["armed"] is True
    # The first request was refused and the probe asked again once the checks cleared.
    assert flight["control_attempts"] > 1
    assert "PreArm: 3D Accel calibration needed" in flight["refusals"]
    assert flight["pre_arm_wait_s"] == 60.0


def test_a_vehicle_whose_pre_arm_checks_never_clear_is_recorded_as_refusing(tmp_path):
    result, _, _, _, _ = run_probe(
        tmp_path,
        session_kwargs={"refuse_arming": True},
        config_overrides={"probe": {"pre_arm_wait_s": 5.0}},
    )
    motion = check(result, "3_guided_local_ned_motion")
    assert motion.status == "fail"
    assert "did not enter Guided flight" in motion.reason
    flight = json.loads((tmp_path / "out" / "run-a" / "flight-state.json").read_text())
    assert flight["armed"] is False
    assert flight["control_attempts"] > 1
    assert "PreArm: 3D Accel calibration needed" in flight["refusals"]



def test_a_vehicle_running_different_parameters_fails_the_startup_item(tmp_path):
    """The read-back is the check that a parameter file was actually applied."""
    result, _, _, _, _ = run_probe(
        tmp_path, session_kwargs={"parameter_values": {"FRAME_CLASS": 0.0}}
    )
    startup = check(result, "1_startup_and_transport")
    assert startup.status == "fail"
    assert "not running the configured parameters" in startup.reason
    assert "FRAME_CLASS" in startup.reason
    assert startup.evidence["parameters"]["reported_by_autopilot"]["FRAME_CLASS"] == 0.0
# ---------------------------------------------------------------------------
# The command: configuration, prerequisites, receipt and exit codes
# ---------------------------------------------------------------------------


def fake_probe_factory(**run_kwargs):
    """Build a probe with scripted seams, for the dispatch tests."""

    def factory(settings, output_dir):
        clock = FakeClock()
        holder = {}
        sessions = []
        session_kwargs = dict(run_kwargs.get("session_kwargs", {}))

        def make_gateway():
            gateway = FakeGateway(clock, **run_kwargs.get("gateway_kwargs", {}))
            holder["gateway"] = gateway
            return gateway

        def make_session():
            # Both runs' vehicles report the candidate's parameters, as the probe checks.
            values = session_kwargs.pop(
                "parameter_values",
                W.read_configured_parameters(
                    settings.parameter_files(settings.estimator_params)
                ),
            )
            session = ScriptedMavlinkSession(
                clock, gateway_holder=holder, parameter_values=values, **session_kwargs
            )
            sessions.append(session)
            return session

        return W.CompatibilityProbe(
            settings,
            output_dir=output_dir,
            runner_factory=lambda: run_kwargs.get("runner", FakeRunner()),
            session_factory=make_session,
            gateway_factory=make_gateway,
            monotonic_ns=clock.monotonic_ns,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    return factory


def test_a_malformed_configuration_is_a_command_error(tmp_path, capsys):
    code = cli.main(
        [
            "compat",
            "--config",
            str(FIXTURES / "malformed_config.yaml"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "unknown keys" in captured.out + captured.err


def test_a_missing_prerequisite_blocks_the_command_before_anything_starts(tmp_path):
    output = tmp_path / "blocked"
    code = cli.main(
        [
            "compat",
            "--config",
            str(FIXTURES / "blocked_config.yaml"),
            "--output",
            str(output),
        ]
    )
    assert code == 2
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["gate_status"] == "not_applicable"
    assert receipt["sensor_mode"] == "simulator-interface"
    reasons = " ".join(receipt["reasons"])
    for missing in ("webots_application", "autopilot_commit", "sitl_binary"):
        assert missing in reasons, missing
    manifest = json.loads((output / "manifest.json").read_text())
    assert any(entry["satisfied"] is False for entry in manifest["prerequisites"])
    assert "configuration" in manifest
    # No wheel was turned: a blocked run reports what it needs instead.
    assert receipt["episode_id"] is None
    assert receipt["trial_group_id"] is None


def test_a_passing_probe_is_exit_zero_with_a_pass_gate(tmp_path, monkeypatch):
    config_path = write_scene(tmp_path)
    monkeypatch.setattr(W, "build_compatibility_probe", fake_probe_factory())
    output = tmp_path / "run"
    code = cli.main(["compat", "--config", str(config_path), "--output", str(output)])
    assert code == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["gate_status"] == "pass"
    assert receipt["stage_id"] == "P00"
    assert receipt["receipt_version"] == cli.RECEIPT_VERSION
    assert receipt["config_hash"] and len(receipt["config_hash"]) == 64
    assert receipt["code_revision"]
    assert receipt["started_at_utc"] and receipt["finished_at_monotonic_s"] >= receipt["started_at_monotonic_s"]
    artifacts = {entry["path"]: entry for entry in receipt["artifacts"]}
    assert artifacts["manifest.json"]["sha256"]
    assert artifacts["checks.json"]["bytes"] > 0
    assert receipt["command"][:3] == ["python", "-m", "embodied"]


def test_a_failed_check_is_exit_zero_with_a_fail_gate(tmp_path, monkeypatch):
    """A completed run whose checks fail the gate is still a valid result."""
    config_path = write_scene(tmp_path)
    monkeypatch.setattr(
        W, "build_compatibility_probe", fake_probe_factory(gateway_kwargs={"colourless": True})
    )
    output = tmp_path / "run"
    code = cli.main(["compat", "--config", str(config_path), "--output", str(output)])
    assert code == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["gate_status"] == "fail"
    assert any("no colour information" in reason for reason in receipt["reasons"])
    assert receipt["limitations"]


def test_a_step_that_cannot_produce_evidence_is_exit_one(tmp_path, monkeypatch):
    config_path = write_scene(tmp_path)

    def factory(settings, output_dir):
        class Exploding(ScriptedMavlinkSession):
            def drain(self):
                raise W.ProbeFailure("the autopilot stream died mid-step")

        clock = FakeClock()
        return W.CompatibilityProbe(
            settings,
            output_dir=output_dir,
            runner_factory=lambda: FakeRunner(),
            session_factory=lambda: Exploding(clock),
            gateway_factory=lambda: FakeGateway(clock),
            monotonic_ns=clock.monotonic_ns,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    monkeypatch.setattr(W, "build_compatibility_probe", factory)
    output = tmp_path / "run"
    code = cli.main(["compat", "--config", str(config_path), "--output", str(output)])
    assert code == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "invalid"
    assert "died mid-step" in " ".join(receipt["reasons"])


def test_an_existing_receipt_is_never_overwritten(tmp_path, monkeypatch):
    config_path = write_scene(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "receipt.json").write_text('{"status": "complete"}', encoding="utf-8")
    monkeypatch.setattr(W, "build_compatibility_probe", fake_probe_factory())
    code = cli.main(["compat", "--config", str(config_path), "--output", str(output)])
    assert code == 1
    assert json.loads((output / "receipt.json").read_text()) == {"status": "complete"}


def test_configuration_problems_name_the_key_they_came_from(tmp_path):
    document = yaml.safe_load(write_scene(tmp_path).read_text(encoding="utf-8"))
    del document["probe"]["budget_wall_clock_s"]
    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(cli.ConfigError) as absent:
        cli.load_config(missing)
    assert "probe.budget_wall_clock_s" in str(absent.value)

    document = yaml.safe_load(write_scene(tmp_path).read_text(encoding="utf-8"))
    document["sensors"]["stereo"]["width"] = "wide"
    wrong_type = tmp_path / "wrong-type.yaml"
    wrong_type.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(cli.ConfigError) as typed:
        cli.load_config(wrong_type)
    assert "sensors.stereo.width" in str(typed.value)


def test_a_settings_value_that_cannot_describe_a_run_is_refused(tmp_path):
    config_path = write_scene(tmp_path)
    document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    document["sensors"]["stereo"]["encoding"] = "gray8"
    path = tmp_path / "gray.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(cli.ConfigError) as refusal:
        settings_for(path, tmp_path)
    assert "rgb8" in str(refusal.value)


def test_the_default_output_directory_is_a_fresh_run_directory():
    first = cli.default_output_directory("p00-compat")
    second = cli.default_output_directory("p00-compat")
    assert first != second
    assert first.parent == cli.repository_root() / "work" / "runs"
    assert re.fullmatch(r"p00-compat-\d{8}T\d{6}Z-[0-9a-f]{4}", first.name)


def test_a_receipt_is_written_for_every_status(tmp_path):
    spec = cli.CommandSpec(
        name="compat",
        help_text="a registered command",
        stage_id="P00",
        run_prefix="p00-compat",
        handler=lambda args, output: None,
        add_arguments=None,
    )
    for status in cli.CommandStatus:
        output = tmp_path / status.value
        cli.write_artifacts(
            output,
            cli.CommandOutcome(
                status=status,
                gate_status=cli.GateStatus.NOT_APPLICABLE,
                sensor_mode=R.SensorMode.SIMULATOR_INTERFACE,
            ),
            spec=spec,
            argv=["python", "-m", "embodied", "compat"],
            config_hash_value=None,
            started_monotonic_s=1.0,
            started_at_utc="2026-01-01T00:00:00+00:00",
        )
        receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == status.value
        assert receipt["gate_status"] == "not_applicable"
        # P00 flies no physical episode and runs no paired trial, whatever the status.
        assert receipt["episode_id"] is None
        assert receipt["trial_group_id"] is None
        assert receipt["sensor_mode"] == "simulator-interface"
        assert [entry["path"] for entry in receipt["artifacts"]] == ["manifest.json"]


def test_usage_problems_and_the_pending_status_map_to_the_declared_codes(capsys):
    assert cli.main([]) == 1
    assert cli.main(["compat", "--nonsense"]) == 1
    assert cli.EXIT_CODES[cli.CommandStatus.PENDING] == 3
    pending = cli.CommandOutcome(
        status=cli.CommandStatus.PENDING, gate_status=cli.GateStatus.NOT_APPLICABLE
    )
    assert pending.exit_code() == 3
    assert cli.CommandOutcome(
        status=cli.CommandStatus.COMPLETE, gate_status=cli.GateStatus.FAIL
    ).exit_code() == 0
    assert cli.CommandOutcome(
        status=cli.CommandStatus.BLOCKED, gate_status=cli.GateStatus.NOT_APPLICABLE
    ).exit_code() == 2
    assert cli.CommandOutcome(
        status=cli.CommandStatus.INVALID, gate_status=cli.GateStatus.NOT_APPLICABLE
    ).exit_code() == 1


def test_registering_the_same_command_twice_is_refused():
    with pytest.raises(cli.CommandError):
        cli.register_command(
            "compat",
            lambda settings, output: None,
            help_text="duplicate",
            stage_id="P00",
            run_prefix="p00-compat",
        )
    assert "compat" in cli.COMMAND_REGISTRY


# ---------------------------------------------------------------------------
# The real subprocess runner: who owns a child's death
# ---------------------------------------------------------------------------


IGNORES_SIGTERM = r"""
import signal, subprocess, sys, time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    ]
)
print("grandchild", grandchild.pid, flush=True)
time.sleep(60)
"""


def _wait_until_gone(pid, timeout_s=5.0):
    """Wait for a pid to stop answering, so a reaped-orphan race is not a flake."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"pid {pid} survived the bounded stop")
        time.sleep(0.05)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are a POSIX facility")
def test_the_real_runner_stops_a_sigterm_ignoring_child_and_the_process_group_it_forked(tmp_path):
    """The stop owns the child's group, stays bounded, and never raises.

    The suite replaces the runner everywhere else, so this is the real stop path:
    the child here ignores SIGTERM exactly as the pinned SITL does, and it forks a
    grandchild that ignores it too, exactly as the simulator forks its vehicle
    controller. A pid-scoped stop would leave both alive and an unguarded wait
    would raise out of ``stop()``'s ``finally`` — abandoning ``shutdown.json``,
    which is how iteration 2's invocation lost its evidence.
    """
    runner = W.SubprocessRunner()
    child = runner.spawn(
        "stubborn",
        [sys.executable, "-c", IGNORES_SIGTERM],
        log_path=tmp_path / "stubborn.log",
    )
    # The child leads its own group, so signalling that group reaches its forks
    # and can never reach the harness's group.
    assert os.getpgid(child.pid) == child.pid

    grandchild_pid = None
    deadline = time.monotonic() + 10.0
    while grandchild_pid is None and time.monotonic() < deadline:
        for line in child.log_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("grandchild "):
                grandchild_pid = int(line.split()[1])
                break
        else:
            time.sleep(0.05)
    assert grandchild_pid is not None, "the child never forked, so the test would prove nothing"

    # SIGTERM is ignored: the stop must escalate inside its own bound and return
    # the observed status instead of raising.
    exit_code = runner.terminate(child, timeout_s=0.5)

    assert exit_code == -signal.SIGKILL
    assert runner.poll(child) is not None
    _wait_until_gone(child.pid)
    _wait_until_gone(grandchild_pid)


# ---------------------------------------------------------------------------
# The reader thread
# ---------------------------------------------------------------------------


class FiniteGateway:
    """A controller stream with a fixed set of records, then silence."""

    def __init__(self, records):
        self.records = list(records)
        self.reads = 0
        self.closed = False

    def open(self, host, port, timeout_s):
        return None

    def read_record(self, timeout_s):
        self.reads += 1
        return self.records.pop(0) if self.records else None

    def send_fault(self, injection):
        raise AssertionError("the reader tests never inject")

    def close(self):
        self.closed = True


class EndlessGateway:
    """A controller whose stream never ends: one fresh inertial record per read.

    ``record_cost_s`` makes production slower than a consumer, so a consumer pass
    has to wait for the reader the way it waits on a real stream.
    """

    def __init__(self, clock, record_cost_s=0.001):
        self.clock = clock
        self.record_cost_s = record_cost_s
        self.produced = 0
        self.closed = False

    def open(self, host, port, timeout_s):
        return None

    def read_record(self, timeout_s):
        time.sleep(self.record_cost_s)
        self.produced += 1
        return W.SensorRecord(
            kind=W.Kind.IMU,
            sim_time_s=self.clock.monotonic() - 1000.0,
            sequence=self.produced,
            flags=0,
            received_stamp=W.ClockStamp(
                host_id="test-host", clock_id="monotonic", monotonic_ns=self.clock.monotonic_ns()
            ),
            imu=W.ImuPayload(
                capture_host_ns=self.clock.monotonic_ns(),
                accelerometer=(0.0, 0.0, -9.81),
                gyro=(0.0, 0.0, 0.0),
                inertial_unit_rpy=(0.0, 0.0, 0.0),
                device_names=("accelerometer", "gyro", "inertial unit"),
                units="m/s^2; rad/s; rad, ENU negated on y and z into NED",
            ),
        )

    def send_fault(self, injection):
        raise AssertionError("the reader tests never inject")

    def close(self):
        self.closed = True


def reading_rig(tmp_path, gateway, *, clock=None, session=None):
    """A real adapter whose reader files through the probe's own filing path."""
    clock = clock or FakeClock()
    settings = settings_for(write_scene(tmp_path), tmp_path)
    probe = W.CompatibilityProbe(
        settings,
        output_dir=tmp_path / "out",
        runner_factory=FakeRunner,
        session_factory=lambda: session or ScriptedMavlinkSession(clock),
        gateway_factory=lambda: gateway,
        monotonic_ns=clock.monotonic_ns,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    writer = W.EvidenceWriter(tmp_path / "out", "run-a")
    adapter = probe._new_adapter(writer, "run-a", ())
    log = W._FlightLog()
    probe._calibration = W.load_calibration_declaration(settings.calibration_path)
    adapter.record_sink = lambda record: probe._file_stream_record(
        record, adapter, writer, log, label="run-a"
    )
    return adapter, probe, writer, log


def test_records_read_while_the_consumer_is_elsewhere_are_all_kept(tmp_path):
    """The stream is read while nobody is consuming it, and nothing is lost.

    Iteration 3's defect: reading shared a thread with the checklist, so a long
    phase stopped the reading and the bounded queues dropped what arrived. Here
    the consumer stays away while the whole stream arrives; every record is filed
    to evidence and then handed over, in order, with the pixels already on disk —
    which is what keeps the handoff itself metadata-only.
    """
    inner = FakeGateway(FakeClock())
    records = [inner._status()]
    for _ in range(12):
        records.append(inner._pair())
        records.append(inner._imu())
    gateway = FiniteGateway(records)
    adapter, _, writer, log = reading_rig(tmp_path, gateway)
    # What the consumer passes would have declared by the time these arrive.
    adapter.reader_phase = "flight"
    adapter.start()
    try:
        # The consumer is deliberately elsewhere while the reader drinks the stream.
        deadline = time.monotonic() + 5.0
        while gateway.reads < len(records) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert gateway.reads >= len(records), "the reader never finished the stream"
        handed = []
        while True:
            record = adapter.sensor_record(0.02)
            if record is None:
                break
            handed.append(record)
    finally:
        adapter.stop()
    assert len(handed) == len(records)
    assert [(record.kind, record.sequence) for record in handed] == [
        (record.kind, record.sequence) for record in records
    ]
    # Pair pixels were filed by the reader, so the handoff carried metadata only.
    assert all(record.pair is None for record in handed if record.kind is W.Kind.PAIR)
    assert all(record.imu is not None for record in handed if record.kind is W.Kind.IMU)
    # And every filed record is in the evidence: 12 pairs, 12 inertial samples.
    assert len(log.pair_metadata) == 12
    assert len(log.imu) == 12
    assert len((writer.directory / "imu.jsonl").read_text().splitlines()) == 12
    assert len((writer.directory / "pairs.jsonl").read_text().splitlines()) == 12
    assert len(list((writer.directory / "pairs").glob("*-left.ppm"))) == 12


def test_the_consumer_pass_reports_when_its_own_valve_stops_it(tmp_path):
    """The drain bounds cap the consumer's pass, not the socket.

    The valve is wall time on the consumer, so this rig runs on the real clock.
    A pass that gives up must leave the stream still being read: the next pass
    gets records the reader gathered while this one had given up, and the pass's
    own accounting says it stopped early.
    """
    settings = settings_for(write_scene(tmp_path), tmp_path)
    probe = W.CompatibilityProbe(
        settings,
        output_dir=tmp_path / "out",
        runner_factory=FakeRunner,
        session_factory=lambda: ScriptedMavlinkSession(FakeClock()),
        gateway_factory=lambda: EndlessGateway(FakeClock()),
        monotonic_ns=time.monotonic_ns,
        monotonic=time.monotonic,
        sleep=time.sleep,
    )
    writer = W.EvidenceWriter(tmp_path / "out", "run-a")
    adapter = probe._new_adapter(writer, "run-a", ())
    log = W._FlightLog()
    adapter.record_sink = lambda record: None  # the valve, not the filing, is under test
    adapter.start()
    try:
        first = probe._read_records(adapter, writer, log, label="run-a")
        stats = probe.reader_stats()
        assert stats["drains"] == 1
        assert stats["records"] == len(first)
        assert stats["drains_stopped_early"] == 1, stats
        assert len(first) < W.MAX_DRAINED_RECORDS, (
            "the time valve fired, so the record bound was never reached"
        )
        # The socket was not left behind: the reader kept reading while the
        # consumer's valve had stopped it, so the next pass is served at once.
        second = probe._read_records(adapter, writer, log, label="run-a")
        assert second
    finally:
        adapter.stop()


def test_an_endless_stream_neither_hangs_nor_grows_the_handoff_unbounded(tmp_path):
    """A stream that never ends cannot hang the run or grow the handoff.

    The reader keeps reading; the handoff stops growing at its declared bound; a
    consumer that comes back is served at once; and the adapter still shuts down
    promptly with the reader mid-stream.
    """
    gateway = EndlessGateway(FakeClock(), record_cost_s=0.0005)
    adapter, _, _, _ = reading_rig(tmp_path, gateway)
    adapter.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if adapter._handoff.qsize() >= W.SENSOR_HANDOFF_RECORDS:
                break
            time.sleep(0.01)
        # One more beat: the stream keeps producing past the bound, so what holds
        # the size down is the bound's shedding, never the stream's end.
        time.sleep(0.05)
        assert adapter._handoff.qsize() <= W.SENSOR_HANDOFF_RECORDS
        assert gateway.produced > W.SENSOR_HANDOFF_RECORDS, (
            "the stream really ran past the bound, so the bound is what stopped the growth"
        )
        started = time.monotonic()
        record = adapter.sensor_record(0.05)
        assert record is not None
        assert time.monotonic() - started < 2.0
    finally:
        stop_started = time.monotonic()
        adapter.stop()
    assert time.monotonic() - stop_started < W.READER_JOIN_TIMEOUT_S + 2.0
    assert gateway.closed


def test_telemetry_folds_the_newest_message_not_the_newest_this_thread_read(tmp_path):
    """The newest state reaches the fold even when this thread never read it.

    Iteration 3's second defect: a waypoint record's start sample described the
    vehicle as it had been tens of seconds earlier, because the messages had
    waited for a busy thread. Here several batches collect while the main thread
    reads none of them; the next fold must carry the newest one.
    """
    clock = FakeClock()

    class AdvancingVehicle(ScriptedMavlinkSession):
        """Every batch the reader collects reports the vehicle half a metre further."""

        def _telemetry_batch(self):
            messages = super()._telemetry_batch()
            self.position = (self.position[0] + 0.5, self.position[1], self.position[2])
            for message in messages:
                if message["mavpackettype"] == "LOCAL_POSITION_NED":
                    message["x"] = self.position[0]
            return messages

    session = AdvancingVehicle(clock)
    gateway = FakeGateway(clock)
    adapter, _, _, _ = reading_rig(tmp_path, gateway, clock=clock, session=session)
    adapter.start()
    try:
        deadline = time.monotonic() + 5.0
        while session.drains < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert session.drains >= 4, "the reader never collected the batches"
        # Freeze the handoff so the assertion is exact: the fold must reach the
        # newest batch the reader collected.
        adapter._reader_stop.set()
        adapter._reader_thread.join(timeout=2.0)
        sample = adapter.telemetry()
        assert sample.local_position_ned[0] == pytest.approx(0.5 * session.drains)
        assert sample.messages_seen >= 2, (
            "the newest batch was folded, whatever the queue shed ahead of it"
        )
        # With nothing new, the fold keeps the last known state.
        again = adapter.telemetry()
        assert again.messages_seen == 0
        assert again.local_position_ned == sample.local_position_ned
    finally:
        adapter.stop()


# ---------------------------------------------------------------------------
# The injection acknowledgement wait
# ---------------------------------------------------------------------------


class SilentFaultGateway:
    """A controller that takes each fault and keeps its sensor stream silent.

    It never answers on its own: the tests place any acknowledgement in the
    stream themselves, so the wait's pacing is what is under test, not the
    controller's willingness to answer.
    """

    def __init__(self, clock):
        self.clock = clock
        self.faults = []
        self.reads = 0
        self.closed = False

    def open(self, host, port, timeout_s):
        return None

    def read_record(self, timeout_s):
        self.reads += 1
        return None

    def send_fault(self, injection):
        self.faults.append(injection)

    def close(self):
        self.closed = True


class LateAckGateway(SilentFaultGateway):
    """A silent stream that lets one acknowledgement surface only after a delay.

    The delay runs on the real clock from the stream's opening, because the
    reader that would deliver the record runs on a real thread.
    """

    def __init__(self, clock, ack, delay_s):
        super().__init__(clock)
        self.ack = ack
        self.delay_s = delay_s
        self.opened_at = None

    def open(self, host, port, timeout_s):
        self.opened_at = time.monotonic()
        return None

    def read_record(self, timeout_s):
        self.reads += 1
        if self.ack is not None and time.monotonic() - self.opened_at >= self.delay_s:
            ack, self.ack = self.ack, None
            return ack
        return None


def acknowledgement_record(clock, injection):
    """The record the controller sends for one fault, as FakeGateway builds it."""
    return W.SensorRecord(
        kind=W.Kind.FAULT_ACK,
        sim_time_s=clock.monotonic() - 1000.0,
        sequence=0,
        flags=0,
        received_stamp=W.ClockStamp(
            host_id="test-host", clock_id="monotonic", monotonic_ns=clock.monotonic_ns()
        ),
        fault_ack={
            "injection_id": injection.injection_id,
            "applied": True,
            "state": {},
            "reason": None,
        },
    )


def test_the_ack_wait_keeps_up_with_a_full_handoff(tmp_path, monkeypatch):
    """An acknowledgement queued behind a full handoff is still found.

    Iteration 5's defect: the wait taxed every returned record with a 50 ms
    sleep — about 20 records a second against a stream that produces about
    510 — so an acknowledgement entering behind a full bounded handoff was
    shed off its head before the poll could reach it. Iteration 4's receipt
    recorded exactly that: ``inj-001-clear`` answered ``"ack": null`` in both
    runs. Here the handoff is full when the clear is sent, the
    acknowledgement sits last in it, and the wait is given half a second: the
    wait must consume what the stream produces and surface it anyway. Under
    the old pacing this cannot pass — 1024 records at 20 a second is far
    beyond half a second.
    """
    clock = FakeClock()
    gateway = SilentFaultGateway(clock)
    adapter, _, writer, _ = reading_rig(tmp_path, gateway, clock=clock)
    fault = W.Injection.clear(1, kind="position_step")
    inner = FakeGateway(clock)
    # The hold loop has just drained and slept, as it always does before a
    # clear is sent, so the acknowledgement enters behind a completely full
    # handoff; enqueueing is the reader's own act, shed rule included.
    for _ in range(W.SENSOR_HANDOFF_RECORDS):
        adapter._enqueue(adapter._handoff, inner._imu())
    adapter._enqueue(adapter._handoff, acknowledgement_record(clock, fault))
    adapter.start()
    try:
        started = clock.now
        monkeypatch.setattr(W, "INJECTION_ACK_TIMEOUT_S", 0.5)
        receipt = adapter.inject(fault)
    finally:
        adapter.stop()
    assert gateway.faults == [fault], "the clear was never even sent"
    assert receipt.applied is True
    assert receipt.requested is fault
    # The wait paid no 50 ms tax at all: a single sleep would have moved this
    # fake clock by a whole step.
    assert clock.now - started < 0.05
    line = json.loads((writer.directory / "injections.jsonl").read_text().splitlines()[-1])
    assert line["injection"] == fault.injection_id
    assert line["ack"]["applied"] is True


def test_the_ack_wait_still_sleeps_on_an_empty_stream_and_finds_a_late_ack(tmp_path, monkeypatch):
    """The empty path keeps its courtesy sleep and still catches the answer.

    The pacing change removes the tax on returned records, not the wait's
    patience: while the stream is silent the poll still sleeps between reads,
    and an acknowledgement that surfaces later — here a third of a second
    after the stream opened — is found inside the deadline.
    """
    clock = FakeClock()
    fault = W.Injection.clear(1, kind="position_step")
    gateway = LateAckGateway(clock, acknowledgement_record(clock, fault), delay_s=0.3)
    adapter, _, writer, _ = reading_rig(tmp_path, gateway, clock=clock)
    adapter.start()
    try:
        started = clock.now
        monkeypatch.setattr(W, "INJECTION_ACK_TIMEOUT_S", 2.0)
        receipt = adapter.inject(fault)
    finally:
        adapter.stop()
    assert gateway.faults == [fault]
    assert receipt.applied is True
    assert receipt.acknowledged_state is not None
    # The stream was empty on the first read, so the wait slept: the fake
    # clock only moves when somebody waits, and it floors every sleep at its
    # own 0.05 s step, so any movement at all proves the empty path slept
    # rather than spinning; the deadline was never reached.
    assert clock.now > started
    assert clock.now - started < 2.0
    line = json.loads((writer.directory / "injections.jsonl").read_text().splitlines()[-1])
    assert line["ack"]["injection_id"] == fault.injection_id


def test_an_acknowledgement_that_never_arrives_is_reported_at_the_deadline(tmp_path, monkeypatch):
    """No acknowledgement means applied: False, with the documented reason.

    A missing acknowledgement is reported as missing information, never
    assumed into a success: at the shortened deadline the receipt says the
    controller did not acknowledge, and the evidence line records the null
    ack — the shape both iteration-4 runs produced for ``inj-001-clear``.
    """
    clock = FakeClock()
    gateway = SilentFaultGateway(clock)
    adapter, _, writer, _ = reading_rig(tmp_path, gateway, clock=clock)
    adapter.start()
    try:
        started = clock.now
        monkeypatch.setattr(W, "INJECTION_ACK_TIMEOUT_S", 0.1)
        receipt = adapter.inject(W.Injection.clear(1, kind="position_step"))
    finally:
        adapter.stop()
    assert gateway.faults, "the injection was not even sent"
    assert receipt.applied is False
    assert receipt.acknowledged_state is None
    assert receipt.reason == "the controller did not acknowledge the injection within 10 s"
    # The wait stopped at its deadline instead of wandering past it.
    assert clock.now - started < 0.5
    line = json.loads((writer.directory / "injections.jsonl").read_text().splitlines()[-1])
    assert line["ack"] is None


def test_a_foreign_acknowledgement_is_parked_without_spinning_the_wait(tmp_path, monkeypatch):
    """An answer to another injection is kept, not re-read in place for ever.

    The mismatch branch hands the record back to ``_pending_records``, which
    ``sensor_record`` reads first: a wait that re-read it without ever waiting
    would spin there for ever without advancing its own deadline. Iteration 5's
    first cut did exactly that — a foreign acknowledgement made this wait
    outlive a 25 s kill while its own deadline was half a second — so the wait
    must pace itself, keep the foreign record for the next reader, and report
    the requested fault as not applied at its deadline. The call runs on its own
    thread so a regression fails here instead of stalling the whole suite.
    """
    clock = FakeClock()
    fault = W.Injection.clear(1, kind="position_step")
    foreign = W.Injection.clear(9, kind="position_step")
    gateway = LateAckGateway(clock, acknowledgement_record(clock, foreign), delay_s=0.0)
    adapter, _, writer, _ = reading_rig(tmp_path, gateway, clock=clock)
    adapter.start()
    try:
        monkeypatch.setattr(W, "INJECTION_ACK_TIMEOUT_S", 0.5)
        started = clock.now
        outcome = {}

        def wait_for_the_answer():
            outcome["receipt"] = adapter.inject(fault)

        caller = threading.Thread(target=wait_for_the_answer, daemon=True)
        caller.start()
        caller.join(timeout=10.0)
        assert not caller.is_alive(), (
            "the wait never returned: it re-read the parked record instead of pacing itself"
        )
        receipt = outcome["receipt"]
    finally:
        adapter.stop()
    assert gateway.faults == [fault]
    assert receipt.applied is False, "another injection's answer is not this fault's answer"
    assert receipt.acknowledged_state is None
    # The foreign record was kept for the next reader, not consumed by this wait.
    parked = [record.fault_ack for record in adapter._pending_records if record.fault_ack]
    assert [entry["injection_id"] for entry in parked] == [foreign.injection_id]
    # And the wait walked its own deadline out instead of spinning in place.
    assert clock.now > started
    assert clock.now - started <= 1.0
    line = json.loads((writer.directory / "injections.jsonl").read_text().splitlines()[-1])
    assert line["injection"] == fault.injection_id
    assert line["ack"] is None
