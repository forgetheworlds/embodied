"""Execution contract tests (no Webots). Thin — sole writer + replace gates.

Safety.check coverage lives in ``tests/test_safety.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied.contracts.perception_ports import (
    AgeDomain,
    EvidenceClass,
    NavPose,
    NavStatus,
    NavigationState,
    uncompared_disagreement,
)
from embodied.contracts.records import ClockStamp
from embodied.control import Motion, Result, VehicleState, Vec3
from embodied.execution import (
    Execution,
    ExecutionStatusCode,
    GeometryCertificate,
    HoldTrajectory,
    PortBundle,
    SegmentTrajectory,
    StartState,
    StartTolerance,
    TrackingEnvelope,
    TrajectoryCertificate,
    ValidityWindow,
)
from embodied.perception.estimation import StaticEstimationPort


def _stamp(ns: int = 1) -> ClockStamp:
    return ClockStamp(host_id="t", clock_id="host/monotonic", monotonic_ns=ns)


def _nav(*, epoch: str = "e1", status: NavStatus = NavStatus.HEALTHY, valid: bool = True) -> NavigationState:
    pose_valid = valid and status in (NavStatus.HEALTHY, NavStatus.DEGRADED)
    return NavigationState(
        nav_epoch=epoch,
        state_sequence=1,
        controller_alignment_id="a",
        stamp=_stamp(),
        sim_time_s=1.0,
        age_s=0.01,
        age_domain=AgeDomain.SIM_CONTROL,
        monotonic_observed_at_s=1.0,
        pose=NavPose(
            parent_frame="odom",
            child_frame="body",
            stamp=_stamp(),
            position_m=(0.0, 0.0, 1.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            covariance=None,
            nav_epoch=epoch,
            source_ids=("ov",),
            valid=pose_valid,
        ),
        velocity_mps=(0.0, 0.0, 0.0),
        covariance=None,
        status=status,
        valid=pose_valid,
        sigma_pos_m=None,
        visual_source_ids=("c",),
        imu_source_ids=("i",),
        visual_age_s=0.01,
        imu_age_s=0.01,
        feed_stall=False,
        ap_disagreement=uncompared_disagreement(),
        evidence_class=EvidenceClass.SENSOR_DERIVED,
        source_ids=("ov",),
    )


def _vehicle(**kwargs) -> VehicleState:
    base = dict(
        armed=True,
        guided=True,
        position=Vec3(0.0, 0.0, 1.0),
        velocity=Vec3(0.0, 0.0, 0.0),
        yaw=0.0,
        landed=False,
    )
    base.update(kwargs)
    return VehicleState(**base)


@dataclass
class FakeClocks:
    mono: float = 0.0
    sim: float = 0.0

    def now_mono_s(self) -> float:
        return self.mono

    def now_sim_s(self) -> float:
        return self.sim

    def advance(self, dt: float) -> None:
        self.mono += dt
        self.sim += dt


@dataclass
class FakeVehicle:
    state_value: VehicleState
    commands: list[Motion] = field(default_factory=list)
    accept: bool = True

    def state(self) -> VehicleState:
        return self.state_value

    def command(self, motion: Motion) -> Result:
        self.commands.append(motion)
        return Result(accepted=self.accept, reason=None if self.accept else "reject")

    def takeoff(self, altitude_m: float) -> Result:
        del altitude_m
        return Result(accepted=True)

    def land(self) -> Result:
        return Result(accepted=True)


def _cert(*, epoch: str = "e1", duration_s: float = 1.0, geometry: GeometryCertificate | None = None) -> TrajectoryCertificate:
    start = Vec3(0.0, 0.0, 1.0)
    end = Vec3(1.0, 0.0, 1.0)
    return TrajectoryCertificate(
        certificate_id="cert-1",
        primary=SegmentTrajectory(start=start, end=end, duration_s=duration_s),
        terminal=HoldTrajectory(position=end),
        fallbacks={"hold": HoldTrajectory(position=start)},
        nav_epoch=epoch,
        start_state=StartState(position=start, velocity=Vec3(0, 0, 0)),
        start_tolerance=StartTolerance(position_m=0.5, velocity_mps=1.0),
        validity=ValidityWindow(None, None),
        tracking_envelope=TrackingEnvelope(position_m=2.0, velocity_mps=5.0),
        geometry_certificate=geometry,
        safety_evidence_refs=(),
        plant_limits_ref=None,
    )


def test_execution_replace_tick_terminal_keep_publishing():
    clocks = FakeClocks()
    vehicle = FakeVehicle(_vehicle())
    est = StaticEstimationPort(_nav())
    exe = Execution(vehicle=vehicle, clocks=clocks, ports=PortBundle(estimation=est))  # type: ignore[arg-type]
    result = exe.replace(_cert(duration_s=0.2))
    assert isinstance(result, type(result)) and result.certificate_id == "cert-1"  # ReplaceOk
    assert exe.status().code is ExecutionStatusCode.RUNNING

    clocks.advance(0.05)
    exe._tick()
    assert vehicle.commands
    assert exe.status().code is ExecutionStatusCode.RUNNING

    clocks.advance(0.3)
    vehicle.state_value = _vehicle(position=Vec3(1.0, 0.0, 1.0))
    before = len(vehicle.commands)
    exe._tick()
    assert exe.status().primary_completed is True
    assert exe.status().code is ExecutionStatusCode.COMPLETED
    assert len(vehicle.commands) == before + 1


def test_execution_unsupported_stops_publishing():
    clocks = FakeClocks()
    vehicle = FakeVehicle(_vehicle())
    cert = _cert()
    exe = Execution(vehicle=vehicle, clocks=clocks, ports=PortBundle(estimation=StaticEstimationPort(_nav())))  # type: ignore[arg-type]
    exe.replace(cert)
    vehicle.state_value = _vehicle(armed=False, guided=False)
    before = len(vehicle.commands)
    exe._tick()
    assert exe.status().code is ExecutionStatusCode.BLOCKED
    assert exe.status().last_decision == "unsupported"
    assert len(vehicle.commands) == before


def test_execution_replace_err_preserves_prior():
    clocks = FakeClocks()
    vehicle = FakeVehicle(_vehicle())
    exe = Execution(vehicle=vehicle, clocks=clocks, ports=PortBundle(estimation=StaticEstimationPort(_nav())))  # type: ignore[arg-type]
    exe.replace(_cert(epoch="e1"))
    err = exe.replace(_cert(epoch="other"))
    assert err.reason == "nav_epoch_mismatch"
    assert exe.status().certificate_id == "cert-1"
