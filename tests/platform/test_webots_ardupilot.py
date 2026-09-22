"""The platform adapter, its wire formats and the compat command, on substituted seams.

Nothing here starts Webots, SITL, or any network connection beyond loopback. The
three seams the adapter declares (process runner, MAVLink session, sensor gateway)
are replaced with scripted components, so the probe's own sequence and judgement are
under test rather than the simulator's behaviour.
"""

from pathlib import Path
import json
import re
import socket
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
    """A controller that streams status, stereo pairs and inertial samples on demand."""

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
        if self.queue:
            return self.queue.pop(0)
        if not self.status_sent:
            self.status_sent = True
            return self._status()
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
        guided_timeout_s=0.6,
        failsafe_after_setpoints=None,
        boot_jitter_s=0.0,
        gateway_holder=None,
    ):
        self.clock = clock
        self.moves = moves
        self.direction = direction
        self.dead_servos = dead_servos
        self.refuse_guided = refuse_guided
        self.guided_timeout_s = guided_timeout_s
        # A count of publications after which the autopilot leaves Guided on its own,
        # the way a failsafe would: the mode change is the aircraft's, not the test's.
        self.failsafe_after_setpoints = failsafe_after_setpoints
        self.boot_jitter_s = boot_jitter_s
        self.gateway_holder = gateway_holder if gateway_holder is not None else {}
        self.sent = []
        self.commands = []
        self.intervals = []
        self.requested_messages = []
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
        self.armed = True

    def takeoff(self, altitude_m):
        self.commands.append(("takeoff", altitude_m))
        if self.refuse_guided:
            self.mode = "LOITER"
        else:
            self.mode = "GUIDED"
        self.target = (0.0, 0.0, -altitude_m)

    def drain(self):
        self._advance()
        self.drains += 1
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
        return [
            {
                "mavpackettype": "HEARTBEAT",
                "custom_mode": W.COPTER_MODES and (4 if guided else 5),
                "base_mode": armed_flag | 81,
                "system_status": 4,
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
            {"mavpackettype": "STATUSTEXT", "text": "guided" if guided else "EKF failsafe check"},
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
            "params": [str(SCENE / "params" / "compat_base.parm")],
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
            "estimator_fault": {"kind": "position_step", "magnitude_m": 30.0, "hold_s": 2.0},
            "timebase_samples": 5,
            "timebase_spread_limit_ms": 60,
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
):
    """Run the whole checklist against scripted seams and return the result."""
    clock = clock or FakeClock()
    session_kwargs = dict(session_kwargs or {})
    holder = {}
    sessions = []

    def make_gateway():
        gateway = FakeGateway(clock, **(gateway_kwargs or {}))
        holder["gateway"] = gateway
        return gateway

    def make_session():
        session = ScriptedMavlinkSession(clock, gateway_holder=holder, **session_kwargs)
        sessions.append(session)
        return session

    runner = runner or FakeRunner()
    config_path = write_scene(tmp_path)
    settings = settings_for(config_path, tmp_path)
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
    state = W.FlightState(
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
    join = W.join_timebase(samples, spread_limit_ms=60.0)
    assert join.joined
    # host = offset + scale * boot, so an offset near +900 s is expected here.
    assert join.offset_s == pytest.approx(900.0, abs=0.05)
    assert join.drift_ppm == pytest.approx(100.0, abs=20.0)
    assert join.spread_ms < 1.0


def test_an_over_limit_spread_is_reported_as_unjoined_clocks():
    samples = [(1000.0 + index * 0.1, 100.0 + index * 0.1 + (0.5 if index % 2 else 0.0)) for index in range(10)]
    join = W.join_timebase(samples, spread_limit_ms=60.0)
    assert not join.joined
    assert "exceeds the configured" in join.reason


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
        "run-a/estimator-not-applicable.json",
        "run-b/estimator.json",
        "run-a/webots.log",
        "run-a/sitl.log",
        "run-a/mavlink.jsonl",
    ):
        assert artifact in result.artifacts, artifact
        assert (tmp_path / "out" / artifact).is_file(), artifact

    # Run A declares why its estimator item does not apply, with the parameter line.
    declaration = json.loads((tmp_path / "out" / "run-a" / "estimator-not-applicable.json").read_text())
    assert declaration["applicable"] is False
    assert any("AHRS_EKF_TYPE" in entry["line"] for entry in declaration["parameter_evidence"])

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
    assert result.manifest["run_b"]["autopilot"]["parameter_files"][1].endswith("compat_ekf.parm")

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
        assert setpoint.type_mask & R.TYPE_MASK_FORCE  # force is always ignored
        assert setpoint.target.position_ned is not None
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
    assert "opposite to the command" in motion.reason


def test_unjoined_clocks_fail_the_timebase_item(tmp_path):
    result, _, _, _, _ = run_probe(tmp_path, session_kwargs={"boot_jitter_s": 0.5})
    frames = check(result, "2_frames_and_timebases")
    assert frames.status == "fail"
    assert "unjoined clocks" in frames.reason
    assert frames.evidence["host_to_autopilot"]["spread_ms"] > 60.0


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


# ---------------------------------------------------------------------------
# The command: configuration, prerequisites, receipt and exit codes
# ---------------------------------------------------------------------------


def fake_probe_factory(**run_kwargs):
    """Build a probe with scripted seams, for the dispatch tests."""

    def factory(settings, output_dir):
        clock = FakeClock()
        holder = {}

        def make_gateway():
            gateway = FakeGateway(clock, **run_kwargs.get("gateway_kwargs", {}))
            holder["gateway"] = gateway
            return gateway

        def make_session():
            return ScriptedMavlinkSession(
                clock, gateway_holder=holder, **run_kwargs.get("session_kwargs", {})
            )

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
