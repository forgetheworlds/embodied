"""Webots controller for the compat scene: the simulator side of the P00 adapter.

This program runs inside Webots under whatever interpreter Webots was configured
with. It does four things and nothing else:

1. read the named devices each step and send flight state to SITL over the pinned
   UDP layout;
2. apply the motor commands SITL sends back, through the same propeller
   linearization the pinned bridge uses;
3. publish stereo pairs and inertial samples to the analysis process over the shared
   ``EMB1`` framing, with capture stamps, and never with an averaged channel;
4. accept sequence-numbered injection commands on that same connection and
   acknowledge what was actually applied.

The framing, the conversion between the scene's ENU frame and the autopilot's NED
frame, and the camera colour conversion are imported from
``embodied.platform.webots_ardupilot`` rather than copied, so the two ends of the
wire cannot drift apart. If that import fails under Webots' interpreter, the
controller says which interpreter it was and why, then stops: a silent controller
would look like a dead simulator.
"""

import argparse
import json
import os
import select
import socket
import sys
import time
import traceback

SHARED_MODULE_ERROR = None


def _import_shared_module():
    """Import the shared framing, using the documented source-tree handoff if needed."""
    try:
        import embodied.platform.webots_ardupilot as shared

        return shared, None
    except Exception as error:  # noqa: BLE001 - the reason is reported, not swallowed
        first_error = f"{type(error).__name__}: {error}"
    source_root = os.environ.get("EMBODIED_SRC")
    if not source_root:
        return None, first_error + " (EMBODIED_SRC was not set, so there was no fallback path)"
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        import embodied.platform.webots_ardupilot as shared

        return shared, None
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}"


SHARED, SHARED_MODULE_ERROR = _import_shared_module()

if SHARED is None:
    print(
        "Controller: cannot import embodied.platform.webots_ardupilot\n"
        f"  interpreter: {sys.executable}\n"
        f"  version: {sys.version}\n"
        f"  error: {SHARED_MODULE_ERROR}",
        flush=True,
    )
    raise SystemExit(1)

from controller import Robot  # noqa: E402  (Webots puts this on the controller's path)

from sensors import VehicleDevices  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sitl-address", default="127.0.0.1")
    parser.add_argument("--sitl-port", type=int, default=9002)
    parser.add_argument("--controller-port", type=int, default=9010)
    parser.add_argument("--motors", default="m1_motor, m2_motor, m3_motor, m4_motor")
    parser.add_argument("--camera-left", default="camera left")
    parser.add_argument("--camera-right", default="camera right")
    parser.add_argument("--camera-period-ms", type=int, default=100)
    parser.add_argument("--accelerometer", default="accelerometer")
    parser.add_argument("--gyro", default="gyro")
    parser.add_argument("--inertial-unit", default="inertial unit")
    parser.add_argument("--gps", default="gps")
    parser.add_argument("--imu-period-ms", type=int, default=10)
    parser.add_argument("--status-interval", type=int, default=100)
    args = parser.parse_args()
    args.motors = [name.strip() for name in args.motors.split(",") if name.strip()]
    return args


class SitlLink:
    """The pinned UDP exchange with SITL: bind the output port, send to the input port.

    SITL binds ``sim-port-in`` and sends motor commands to ``sim-port-out``; the
    pinned Python bridge binds that second port and sends flight state to the first.
    """

    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", port))
        self.socket.setblocking(False)

    def wait_for_sitl(self, timeout_s=60.0):
        """Wait for SITL's first control packet, and keep it as the first command."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self.socket], [], [], 0.25)
            if readable:
                packet = self.socket.recv(SHARED.CONTROL_SIZE + 64)
                if len(packet) >= SHARED.CONTROL_SIZE:
                    return packet[: SHARED.CONTROL_SIZE]
        return None

    def send_flight_state(self, packet):
        self.socket.sendto(packet, (self.address, self.port + 1))

    def receive_controls(self):
        """The newest control packet SITL has sent, or None when it has not spoken."""
        latest = None
        while True:
            readable, _, _ = select.select([self.socket], [], [], 0)
            if not readable:
                return latest
            packet = self.socket.recv(SHARED.CONTROL_SIZE + 64)
            if len(packet) >= SHARED.CONTROL_SIZE:
                latest = packet[: SHARED.CONTROL_SIZE]

    def close(self):
        self.socket.close()


class ObservationChannel:
    """The connection the analysis process reads records on and injects faults through.

    Records are dropped while nobody is connected rather than queued without bound:
    the platform probe is the only reader, and a queue that grew while it was absent
    would be a memory leak with a flight attached. Once a reader is connected the
    camera frames are large enough that the reader can fall behind, so the queue has
    a bound and a drop count rather than a blocking write: blocking would stop the
    simulation, and an unbounded queue would turn a slow reader into a stale stream.
    """

    def __init__(self, port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", port))
        self.server.listen(1)
        self.server.setblocking(False)
        self.client = None
        self.buffer = bytearray()
        self.outgoing = SHARED.OutboundStream()
        self.sequence = 0
        self.status_sent = False

    def accept(self):
        if self.client is None:
            readable, _, _ = select.select([self.server], [], [], 0)
            if readable:
                client, _ = self.server.accept()
                client.setblocking(False)
                self.client = client
                self.outgoing.reset()
                print("Controller: analysis process connected", flush=True)
        return self.client is not None

    def send(self, kind, sim_time_s, payload):
        """Frame one message and queue it, in the order it was produced.

        Status and injection acknowledgements are queued as control frames: they answer
        the analysis process, so a backlog of camera frames defers them but never
        discards them. Everything else is bulk and is dropped from the newest end when
        the reader falls behind. Both kinds leave in the order they were queued, because
        the reader resynchronises on frame boundaries and a reordered frame is a stream
        it cannot parse.
        """
        if self.client is None:
            return
        self.sequence += 1
        framed = SHARED.pack_message(
            kind, sim_time_s=sim_time_s, sequence=self.sequence, payload=payload
        )
        control = kind in (SHARED.Kind.STATUS, SHARED.Kind.FAULT_ACK)
        queued = (
            self.outgoing.queue_control(framed) if control else self.outgoing.queue(framed)
        )
        if not queued:
            return
        self._flush()

    def describe(self):
        """What this channel has done, for the status the probe records."""
        return {
            "frames_produced": self.sequence,
            "reader_connected": self.client is not None,
            **self.outgoing.describe(),
        }

    def _flush(self):
        """Write what the socket will take, and stop the stream if it has failed."""
        try:
            self.outgoing.flush(self.client)
        except OSError as error:
            print(f"Controller: the analysis connection failed: {error}", flush=True)
            self.outgoing.reset()
            self.client.close()
            self.client = None

    def poll_commands(self):
        """Read the injection commands waiting on the connection."""
        if self.client is None:
            return []
        commands = []
        while True:
            readable, _, _ = select.select([self.client], [], [], 0)
            if not readable:
                return commands
            try:
                chunk = self.client.recv(1 << 16)
            except (BlockingIOError, InterruptedError):
                return commands
            except OSError:
                self.client = None
                return commands
            if not chunk:
                self.client = None
                return commands
            self.buffer.extend(chunk)
            while len(self.buffer) >= SHARED.HEADER_SIZE:
                try:
                    message, consumed = SHARED.read_message(bytes(self.buffer))
                except SHARED.FramingError:
                    self.buffer.clear()
                    break
                del self.buffer[:consumed]
                if message.kind is SHARED.Kind.FAULT:
                    commands.append(json.loads(message.payload.decode("utf-8")))

    def close(self):
        if self.client is not None:
            self.client.close()
        self.server.close()


class Injections:
    """The faults this controller can apply, and what is applied right now.

    Each fault changes what the simulator reports about the body. The acknowledgement
    says whether it was applied, so a request cannot be mistaken for an effect.
    """

    def __init__(self):
        self.hold_flight_state = False
        self.position_step_m = 0.0
        self.applied = {}

    def apply_command(self, command):
        kind = command.get("kind")
        if not bool(command.get("apply")):
            self.hold_flight_state = False
            self.position_step_m = 0.0
            self.applied = {}
            return {"applied": True, "state": {}, "reason": None}
        if kind == "position_step":
            self.position_step_m = float(command.get("magnitude_m") or 0.0)
            self.applied["position_step"] = {"magnitude_m": self.position_step_m}
            return {"applied": True, "state": dict(self.applied), "reason": None}
        if kind == "hold_fdm":
            self.hold_flight_state = True
            self.applied["hold_fdm"] = {"hold_s": command.get("hold_s")}
            return {"applied": True, "state": dict(self.applied), "reason": None}
        return {"applied": False, "state": {}, "reason": f"unknown fault {kind!r}"}


def status_document(args, devices, robot, channel):
    """What the controller reports about itself before any measurement is judged.

    The stream counters are here because a reader that falls behind is a fact about
    the run: a dropped frame is reported rather than silently missing.
    """
    scene = {}
    raw_custom_data = robot.getCustomData()
    if raw_custom_data:
        try:
            scene = json.loads(raw_custom_data)
        except json.JSONDecodeError:
            scene = {"declaration_error": "the robot's customData is not JSON"}
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": sys.platform,
        },
        "shared_module": {
            "file": SHARED.__file__,
            "records_revision": SHARED.RECORDS_REVISION,
            "format_version": SHARED.FORMAT_VERSION,
            "source_root": os.environ.get("EMBODIED_SRC"),
        },
        "host_id": socket.gethostname(),
        "clock_id": "monotonic",
        "devices": {
            "left": args.camera_left,
            "right": args.camera_right,
            "accelerometer": args.accelerometer,
            "gyro": args.gyro,
            "inertial_unit": args.inertial_unit,
            "gps": args.gps,
        },
        "motors": list(args.motors),
        "cameras": devices.camera_periods_ms(),
        "camera_size": list(devices.camera_size()),
        "imu_period_ms": args.imu_period_ms,
        "scene": scene,
        "stream": channel.describe(),
        "injection_state": {},
    }


def send_status(channel, status, sim_time_s):
    """Refresh the stream counters and send the status to the analysis process."""
    status["stream"] = channel.describe()
    channel.send(SHARED.Kind.STATUS, sim_time_s, SHARED.encode_status_payload(status))


def main():
    args = parse_args()
    robot = Robot()
    devices = VehicleDevices(robot, args)
    link = SitlLink(args.sitl_address, args.sitl_port)
    channel = ObservationChannel(args.controller_port)
    injections = Injections()
    status = status_document(args, devices, robot, channel)

    print(f"Controller: interpreter {sys.executable}", flush=True)
    print(f"Controller: shared framing from {SHARED.__file__}", flush=True)
    print(f"Listening for ardupilot SITL at {args.sitl_address}:{args.sitl_port}", flush=True)
    first_controls = link.wait_for_sitl()
    if first_controls is None:
        print("Controller: SITL never sent a control packet; stopping", flush=True)
        devices.stop_motors()
        channel.close()
        link.close()
        return 1
    print("Connected to ardupilot SITL", flush=True)

    controls = first_controls

    try:
        run_loop(devices, link, channel, injections, status, args, controls, first_controls)
    except Exception:
        # A controller that dies silently looks exactly like a simulator that never
        # started, so the reason is printed where the platform probe can read it.
        traceback.print_exc()
        print(
            f"Controller: stopped after an exception (interpreter {sys.executable})",
            flush=True,
        )
        raise
    finally:
        devices.stop_motors()
        channel.close()
        link.close()
        print("Controller: stopped", flush=True)
    return 0


def run_loop(devices, link, channel, injections, status, args, controls, first_controls):
    """The simulation loop: flight state out, motor commands in, records beside."""
    camera_period_ms = max(int(args.camera_period_ms), devices.timestep_ms)
    imu_period_ms = max(int(args.imu_period_ms), devices.timestep_ms)
    next_camera_ms = 0
    next_imu_ms = 0
    pair_counter = 0
    last_flight_state = None
    while True:
        if not devices.step():
            print(
                f"Controller: the simulation closed at {devices.simulator_time_s():.3f}s",
                flush=True,
            )
            return
        elapsed_ms = int(devices.simulator_time_s() * 1000.0)

        if channel.accept() and not channel.status_sent:
            send_status(channel, status, devices.simulator_time_s())
            channel.status_sent = True

        state = devices.read_flight_state(SHARED.enu_to_ned)
        if injections.position_step_m:
            state["position_xyz"] = (
                state["position_xyz"][0] + injections.position_step_m,
                state["position_xyz"][1],
                state["position_xyz"][2],
            )
        if injections.hold_flight_state and last_flight_state is not None:
            state = last_flight_state
        last_flight_state = state

        link.send_flight_state(
            SHARED.pack_fdm(SHARED.FlightState(timestamp_s=devices.simulator_time_s(), **state))
        )

        incoming = link.receive_controls()
        if incoming is not None:
            controls = incoming
        devices.set_motor_commands(SHARED.unpack_controls(controls))

        if elapsed_ms >= next_camera_ms:
            next_camera_ms = elapsed_ms + camera_period_ms
            pair = devices.read_pair(SHARED.bgra_to_rgb8)
            if pair is not None:
                pair_counter += 1
                width, height = devices.camera_size()
                channel.send(
                    SHARED.Kind.PAIR,
                    devices.simulator_time_s(),
                    SHARED.encode_pair_payload(
                        capture_host_ns=time.monotonic_ns(),
                        pair_id=pair_counter,
                        left_frame_id=pair_counter,
                        right_frame_id=pair_counter,
                        width=width,
                        height=height,
                        encoding="rgb8",
                        left_bytes=pair[0],
                        right_bytes=pair[1],
                    ),
                )

        if elapsed_ms >= next_imu_ms:
            next_imu_ms = elapsed_ms + imu_period_ms
            inertial = devices.read_inertial(SHARED.enu_to_ned)
            channel.send(
                SHARED.Kind.IMU,
                devices.simulator_time_s(),
                SHARED.encode_imu_payload(
                    capture_host_ns=time.monotonic_ns(),
                    accelerometer=inertial["accelerometer"],
                    gyro=inertial["gyro"],
                    inertial_unit_rpy=inertial["inertial_unit_rpy"],
                    device_names=(args.accelerometer, args.gyro, args.inertial_unit),
                    units="m/s^2; rad/s; rad, ENU negated on y and z into NED",
                ),
            )

        for command in channel.poll_commands():
            result = injections.apply_command(command)
            status["injection_state"] = injections.applied
            channel.send(
                SHARED.Kind.FAULT_ACK,
                devices.simulator_time_s(),
                json.dumps(
                    {
                        "injection_id": command.get("injection_id"),
                        "applied": result["applied"],
                        "state": result["state"],
                        "reason": result["reason"],
                    },
                    sort_keys=True,
                ).encode("utf-8"),
            )
            print(
                f"Controller: injection {command.get('injection_id')} "
                f"apply={command.get('apply')} -> applied={result['applied']}",
                flush=True,
            )

        if channel.sequence and channel.sequence % args.status_interval == 0:
            send_status(channel, status, devices.simulator_time_s())


if __name__ == "__main__":
    raise SystemExit(main())
