"""Execution service: sole Vehicle.command writer under Safety leases."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Protocol

from embodied.contracts.perception_ports import EstimationPort, MappingPort, NavigationState
from embodied.control import Motion, Result, Vehicle, VehicleState, Vec3
from embodied.execution.certificate import Trajectory, TrajectoryCertificate
from embodied.execution.plant import PlantLimitsPort, TelemetryHealthPort
from embodied.execution.safety import (
    AllowDecision,
    AuthorityView,
    BackupDecision,
    TrackingState,
    UnsupportedDecision,
    check,
)
from embodied.execution.trajectories import HoldTrajectory


class ExecutionStatusCode(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    BACKUP = "backup"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionStatus:
    code: ExecutionStatusCode
    certificate_id: str | None
    active_trajectory: str | None  # primary | terminal | backup
    backup_name: str | None
    nav_epoch: str | None
    lease_valid_until_mono_s: float | None
    last_decision: str  # allow | backup | unsupported | none
    last_reason: str | None
    publish_sequence: int
    t_traj_s: float | None
    primary_completed: bool


@dataclass(frozen=True)
class ReplaceOk:
    certificate_id: str
    activated_sim_s: float


@dataclass(frozen=True)
class ReplaceErr:
    reason: str


ReplaceResult = ReplaceOk | ReplaceErr


class ClockPorts(Protocol):
    def now_mono_s(self) -> float: ...

    def now_sim_s(self) -> float: ...


@dataclass
class PortBundle:
    estimation: EstimationPort | None = None
    mapping: MappingPort | None = None
    plant_limits: PlantLimitsPort | None = None
    telemetry_health: TelemetryHealthPort | None = None


@dataclass
class _FakeClocks:
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
class Execution:
    vehicle: Vehicle
    clocks: ClockPorts
    ports: PortBundle
    lease_s: float = 0.25
    nav_age_max_s: float = 0.5
    telem_age_max_s: float = 1.0
    tracking_position_slack_m: float = 0.5

    _certificate: TrajectoryCertificate | None = field(default=None, init=False, repr=False)
    _code: ExecutionStatusCode = field(default=ExecutionStatusCode.IDLE, init=False, repr=False)
    _active: str | None = field(default=None, init=False, repr=False)
    _backup_name: str | None = field(default=None, init=False, repr=False)
    _activated_sim_s: float | None = field(default=None, init=False, repr=False)
    _backup_activated_sim_s: float | None = field(default=None, init=False, repr=False)
    _lease_until: float | None = field(default=None, init=False, repr=False)
    _last_decision: str = field(default="none", init=False, repr=False)
    _last_reason: str | None = field(default=None, init=False, repr=False)
    _publish_sequence: int = field(default=0, init=False, repr=False)
    _t_traj_s: float | None = field(default=None, init=False, repr=False)
    _primary_completed: bool = field(default=False, init=False, repr=False)
    _last_published: Motion | None = field(default=None, init=False, repr=False)

    def status(self) -> ExecutionStatus:
        cert = self._certificate
        return ExecutionStatus(
            code=self._code,
            certificate_id=None if cert is None else cert.certificate_id,
            active_trajectory=self._active,
            backup_name=self._backup_name,
            nav_epoch=None if cert is None else cert.nav_epoch,
            lease_valid_until_mono_s=self._lease_until,
            last_decision=self._last_decision,
            last_reason=self._last_reason,
            publish_sequence=self._publish_sequence,
            t_traj_s=self._t_traj_s,
            primary_completed=self._primary_completed,
        )

    def replace(self, certificate: TrajectoryCertificate) -> ReplaceResult:
        now_mono = self.clocks.now_mono_s()
        now_sim = self.clocks.now_sim_s()
        validity = certificate.validity
        if validity.not_before_mono_s is not None and now_mono < validity.not_before_mono_s:
            return ReplaceErr(reason="validity_not_before")
        if validity.not_after_mono_s is not None and now_mono > validity.not_after_mono_s:
            return ReplaceErr(reason="validity_expired")

        nav = self._pull_nav()
        if nav is not None and nav.nav_epoch != certificate.nav_epoch:
            return ReplaceErr(reason="nav_epoch_mismatch")

        state = self.vehicle.state()
        if state.position is not None:
            start = certificate.start_state.position
            err = (
                (state.position.x - start.x) ** 2
                + (state.position.y - start.y) ** 2
                + (state.position.z - start.z) ** 2
            ) ** 0.5
            if err > certificate.start_tolerance.position_m:
                return ReplaceErr(reason="start_state_mismatch")

        self._certificate = certificate
        self._code = ExecutionStatusCode.RUNNING
        self._active = "primary"
        self._backup_name = None
        self._activated_sim_s = now_sim
        self._backup_activated_sim_s = None
        self._primary_completed = False
        self._lease_until = None
        self._last_decision = "none"
        self._last_reason = None
        return ReplaceOk(certificate_id=certificate.certificate_id, activated_sim_s=now_sim)

    def _pull_nav(self) -> NavigationState | None:
        if self.ports.estimation is None:
            return None
        return self.ports.estimation.latest()

    def _active_trajectory(self) -> Trajectory | None:
        cert = self._certificate
        if cert is None or self._active is None:
            return None
        if self._active == "primary":
            return cert.primary
        if self._active == "terminal":
            return cert.terminal
        if self._active == "backup" and self._backup_name is not None:
            return cert.fallbacks[self._backup_name]
        return None

    def _traj_time(self) -> float:
        now_sim = self.clocks.now_sim_s()
        if self._active == "backup" and self._backup_activated_sim_s is not None:
            return max(0.0, now_sim - self._backup_activated_sim_s)
        if self._activated_sim_s is None:
            return 0.0
        return max(0.0, now_sim - self._activated_sim_s)

    def _tick(self) -> None:
        if self._code in (ExecutionStatusCode.IDLE, ExecutionStatusCode.FAILED):
            return
        cert = self._certificate
        if cert is None:
            return

        traj = self._active_trajectory()
        if traj is None:
            self._code = ExecutionStatusCode.FAILED
            self._last_reason = "no_active_trajectory"
            return

        t = self._traj_time()
        # Primary finite end → terminal
        if (
            self._active == "primary"
            and traj.duration_s is not None
            and t >= traj.duration_s
        ):
            self._active = "terminal"
            self._primary_completed = True
            self._code = ExecutionStatusCode.COMPLETED
            self._activated_sim_s = self.clocks.now_sim_s()
            t = 0.0
            traj = cert.terminal

        try:
            if traj.duration_s is not None:
                t = min(t, traj.duration_s)
            candidate = traj.sample(t)
        except ValueError as exc:
            self._code = ExecutionStatusCode.FAILED
            self._last_reason = f"sample_error:{exc}"
            return

        self._t_traj_s = t
        now_mono = self.clocks.now_mono_s()
        now_sim = self.clocks.now_sim_s()
        vehicle_state = self.vehicle.state()
        measured_pos = vehicle_state.position or candidate.position
        measured_vel = vehicle_state.velocity or Vec3(0.0, 0.0, 0.0)

        nav = self._pull_nav()
        occupancy = None if self.ports.mapping is None else self.ports.mapping.occupancy()
        plant = None if self.ports.plant_limits is None else self.ports.plant_limits.latest()
        telem = None if self.ports.telemetry_health is None else self.ports.telemetry_health.latest()

        authority = AuthorityView(
            armed=vehicle_state.armed,
            guided=vehicle_state.guided,
            landed=vehicle_state.landed,
            failsafe_active=False if telem is None else telem.failsafe_active,
            telemetry_age_mono_s=0.0 if telem is None else telem.telemetry_age_s,
            command_rejecting=False,
        )
        tracking = TrackingState(
            candidate=candidate,
            last_published=self._last_published,
            measured_position=measured_pos,
            measured_velocity=measured_vel,
            measured_yaw=vehicle_state.yaw,
            measured_age_mono_s=0.0,
            measured_source="vehicle",
            publish_sequence=self._publish_sequence,
        )
        mode = (
            "backup_active"
            if self._active == "backup"
            else ("terminal" if self._active == "terminal" else "primary")
        )
        decision = check(
            active_prefix=traj,
            stop_continuation=cert.terminal,
            predeclared_fallbacks=cert.fallbacks,
            nav_state=nav,
            vehicle_state=vehicle_state,
            authority=authority,
            tracking_state=tracking,
            tracking_envelope=cert.tracking_envelope,
            occupancy=occupancy,
            plant_limits=plant,
            telemetry_health=telem,
            geometry_certificate=cert.geometry_certificate,
            certificate_nav_epoch=cert.nav_epoch,
            certificate_validity=cert.validity,
            safety_evidence_refs=cert.safety_evidence_refs,
            mode=mode,
            now_mono_s=now_mono,
            now_sim_s=now_sim,
            lease_s=self.lease_s,
            nav_age_max_s=self.nav_age_max_s,
            telem_age_max_s=self.telem_age_max_s,
        )

        if isinstance(decision, AllowDecision):
            self._last_decision = "allow"
            self._last_reason = None
            self._lease_until = decision.valid_until_mono_s
            if self._active == "backup":
                self._code = ExecutionStatusCode.BACKUP
            elif self._primary_completed:
                self._code = ExecutionStatusCode.COMPLETED
            else:
                self._code = ExecutionStatusCode.RUNNING
            result = self.vehicle.command(candidate)
            if not result.accepted:
                self._code = ExecutionStatusCode.FAILED
                self._last_reason = result.reason or "command_rejected"
                return
            self._last_published = candidate
            self._publish_sequence += 1
            return

        if isinstance(decision, BackupDecision):
            self._last_decision = "backup"
            self._last_reason = decision.reason
            self._lease_until = decision.valid_until_mono_s
            if decision.trajectory_name not in cert.fallbacks:
                self._code = ExecutionStatusCode.BLOCKED
                return
            if self._active != "backup" or self._backup_name != decision.trajectory_name:
                self._active = "backup"
                self._backup_name = decision.trajectory_name
                self._backup_activated_sim_s = now_sim
            self._code = ExecutionStatusCode.BACKUP
            backup_traj = cert.fallbacks[decision.trajectory_name]
            bt = max(0.0, now_sim - (self._backup_activated_sim_s or now_sim))
            try:
                backup_motion = backup_traj.sample(bt if backup_traj.duration_s is None else min(bt, backup_traj.duration_s or bt))
            except ValueError:
                backup_motion = HoldTrajectory(position=candidate.position).sample(0.0)
            result = self.vehicle.command(backup_motion)
            if not result.accepted:
                self._code = ExecutionStatusCode.FAILED
                self._last_reason = result.reason or "backup_command_rejected"
                return
            self._last_published = backup_motion
            self._publish_sequence += 1
            return

        # UNSUPPORTED — stop publishing; no invented hold
        assert isinstance(decision, UnsupportedDecision)
        self._last_decision = "unsupported"
        self._last_reason = decision.reason
        self._lease_until = None
        self._code = ExecutionStatusCode.BLOCKED
