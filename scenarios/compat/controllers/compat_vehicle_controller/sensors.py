"""Device access for the compat controller: the one place that knows the Webots API.

The controller's job is to be a transparent transport: read the named devices, hand
their values on in a declared frame with a capture stamp, and drive the motors with
exactly what SITL asked for. Nothing here decides anything about a mission, and
nothing here averages or reorders a camera channel.

The propeller conversion is imported from the shared module so the simulator side
and the adapter state it once. That import is safe here because the controller
resolves the shared module, including the ``EMBODIED_SRC`` fallback, before it
imports this file.
"""

import time

from embodied.platform.webots_ardupilot import propeller_velocity


class SimWallPacer:
    """Clamp the simulator's advance to at most one simulated second per wall second.

    WHY THIS SHAPE. The world advances only while this synchronized controller is
    inside ``robot.step()`` (Iris.proto ``synchronization TRUE``), so the controller's
    step loop IS the simulator's pacing. Webots' realtime mode recovers lost ground by
    sprinting: after a slow stretch it runs steps back-to-back until simulation time is
    back in phase with the wall, which delivers tens of milliseconds of sensor time
    inside a single wall tick (measured: local sim/wall ratio p95 6.7-7.7 at pair
    arrivals, F2-TRANSPORT-REPORT.md §4). A gate that only slept "when sim leads wall"
    would never fire, because catch-up never leads the wall; the clamp has to hold the
    RATE, not the phase. This pacer therefore schedules each step's release one
    ``timestep`` after the previous one, on the wall clock, and never reschedules
    early: a step that overran its slot may push later slots back, but no later step
    may run early to recover the lost time. The simulation clock is then 1-Lipschitz
    in the wall clock -- over any window, at most that window's duration of simulated
    time can be produced -- which is exactly the owner's ruling that a 10 ms tick must
    mean 10 ms of sensor time. Slower than realtime is still allowed (a loaded host
    cannot be sped up, and a slower simulator only ages the wall-clock side); the
    clamp forbids only the sprint.

    The gate is selected by the run's configuration (the bridge sets
    ``EMBODIED_SIM_WALL_CLAMP=1`` in the simulator process's environment for the
    scored path); with the gate off, Webots' own pacing -- realtime with catch-up, or
    fast mode for iteration -- is untouched.
    """

    def __init__(self, timestep_ms, clock=time.monotonic, sleep=time.sleep):
        self.timestep_ms = int(timestep_ms)
        self._dt = self.timestep_ms / 1000.0
        self._clock = clock
        self._sleep = sleep
        self._next_release = None
        self.steps = 0
        self.gates = 0
        self.slept_s = 0.0

    def after_step(self):
        """Hold the wall until this step's slot has been paid for."""
        self.steps += 1
        now = self._clock()
        if self._next_release is None:
            self._next_release = now + self._dt
            return
        if now < self._next_release:
            self._sleep(self._next_release - now)
            self.gates += 1
            self.slept_s += self._clock() - now
            now = self._clock()
        # Schedule from the later of the planned release and the actual wake-up, so the
        # release times only ever fall behind: a late step cannot be followed by an
        # early one, which is what forbids the catch-up sprint.
        self._next_release = max(self._next_release, now) + self._dt

    def document(self):
        return {
            "enabled": True,
            "rule": (
                "each basic time step holds at least its own duration of wall time; "
                "the schedule never runs early, so no catch-up sprint is possible"
            ),
            "timestep_ms": self.timestep_ms,
            "steps": self.steps,
            "gates": self.gates,
            "slept_s": round(self.slept_s, 3),
        }


class VehicleDevices:
    """The scene's named devices, read and written as one group.

    Device names come from the controller arguments, which come from the world file,
    so a renamed device is a loud load error rather than a silently missing sensor.
    """

    def __init__(self, robot, args, clamp_sim_wall=False):
        self.robot = robot
        self.timestep_ms = int(robot.getBasicTimeStep())
        # The scored path's sim/wall clamp (owner ruling 2026-09-30, APPROVAL-RECORD
        # "F2's denominator"): None leaves the simulator's own pacing untouched, which
        # is what iteration runs keep.
        self.pacer = SimWallPacer(self.timestep_ms) if clamp_sim_wall else None
        self.accelerometer = self._device(robot, args.accelerometer)
        self.gyro = self._device(robot, args.gyro)
        self.inertial_unit = self._device(robot, args.inertial_unit)
        self.gps = self._device(robot, args.gps)
        self.cameras = {
            "left": self._device(robot, args.camera_left),
            "right": self._device(robot, args.camera_right),
        }
        self.motors = [self._device(robot, name) for name in args.motors]

        for device in (self.accelerometer, self.gyro, self.inertial_unit, self.gps):
            device.enable(int(args.imu_period_ms))
        for camera in self.cameras.values():
            camera.enable(int(args.camera_period_ms))
        for motor in self.motors:
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

    def _device(self, robot, name):
        device = robot.getDevice(name)
        if device is None:
            raise RuntimeError(
                f"the scene has no device named {name!r}; the world file and the "
                "controller arguments disagree"
            )
        return device

    def simulator_time_s(self):
        return float(self.robot.getTime())

    def step(self):
        """Advance the simulation by one basic time step. False when Webots closed."""
        stepped = self.robot.step(self.timestep_ms) != -1
        if stepped and self.pacer is not None:
            self.pacer.after_step()
        return stepped

    def pacing_document(self):
        """What this controller enforces on the simulator's pacing, or its absence."""
        if self.pacer is None:
            return {
                "enabled": False,
                "rule": "none: the simulator's own pacing is unmodified",
                "timestep_ms": self.timestep_ms,
            }
        return self.pacer.document()

    def camera_periods_ms(self):
        return {name: int(camera.getSamplingPeriod()) for name, camera in self.cameras.items()}

    def camera_size(self):
        camera = self.cameras["left"]
        return int(camera.getWidth()), int(camera.getHeight())

    def read_pair(self, convert):
        """Read both eyes in the same step and convert them to packed RGB.

        Neither camera is stepped between the two reads, so a pair shares one capture
        instant. That is the property the frame counters travelling beside the pixels
        are there to check.
        """
        frames = {}
        for name, camera in self.cameras.items():
            buffer = camera.getImage()
            if buffer is None:
                return None
            frames[name] = convert(buffer, int(camera.getWidth()), int(camera.getHeight()))
        return frames["left"], frames["right"]

    def read_inertial(self, enu_to_ned):
        """The three inertial devices, converted into the autopilot's NED frame."""
        return {
            "accelerometer": enu_to_ned(self.accelerometer.getValues()),
            "gyro": enu_to_ned(self.gyro.getValues()),
            "inertial_unit_rpy": enu_to_ned(self.inertial_unit.getRollPitchYaw()),
        }

    def read_flight_state(self, enu_to_ned):
        """Everything SITL needs, in the field order the pinned model reads."""
        return {
            "gyro_rpy": enu_to_ned(self.gyro.getValues()),
            "accel_xyz": enu_to_ned(self.accelerometer.getValues()),
            "attitude_rpy": enu_to_ned(self.inertial_unit.getRollPitchYaw()),
            "velocity_xyz": enu_to_ned(self.gps.getSpeedVector()),
            "position_xyz": enu_to_ned(self.gps.getValues()),
        }

    def set_motor_commands(self, fractions):
        """Drive the motors from the fractions SITL sends.

        A fraction below zero means SITL is not using that channel, so the motor is
        left alone. Everything else is a fraction of full throttle, which this scene's
        propellers convert into an angular velocity through the shared conversion.
        """
        for motor, fraction in zip(self.motors, fractions):
            if fraction < 0:
                continue
            motor.setVelocity(
                propeller_velocity(fraction, max_velocity=motor.getMaxVelocity())
            )

    def stop_motors(self):
        for motor in self.motors:
            motor.setVelocity(0.0)
