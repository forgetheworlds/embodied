"""Control layer: guided motion primitives (Vehicle API).

Owns takeoff, goto, hold, spin, and land. Sim bring-up stays in ``platform``.
"""

from embodied.control.vehicle import (
    ARM_SETTLE_S,
    AutopilotControlEvidence,
    CONTROL_GRANT_GRACE_S,
    CONTROL_RETRY_S,
    FrameError,
    LocalNedTarget,
    REFRESH_S,
    SPIN_RATE_RAD_S,
    SetpointPublication,
    TAKEOFF_ATTEMPTS,
    TAKEOFF_RETRY_SIM_S,
    Vehicle,
    enu_to_ned,
    mask_for_target,
    wrap_angle_rad,
)

__all__ = [
    "ARM_SETTLE_S",
    "AutopilotControlEvidence",
    "CONTROL_GRANT_GRACE_S",
    "CONTROL_RETRY_S",
    "FrameError",
    "LocalNedTarget",
    "REFRESH_S",
    "SPIN_RATE_RAD_S",
    "SetpointPublication",
    "TAKEOFF_ATTEMPTS",
    "TAKEOFF_RETRY_SIM_S",
    "Vehicle",
    "enu_to_ned",
    "mask_for_target",
    "wrap_angle_rad",
]
