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

from embodied.platform.webots_ardupilot import propeller_velocity


class VehicleDevices:
    """The scene's named devices, read and written as one group.

    Device names come from the controller arguments, which come from the world file,
    so a renamed device is a loud load error rather than a silently missing sensor.
    """

    def __init__(self, robot, args):
        self.robot = robot
        self.timestep_ms = int(robot.getBasicTimeStep())
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
        return self.robot.step(self.timestep_ms) != -1

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
