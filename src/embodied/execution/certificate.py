"""Trajectory certificate types for Execution / Safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from embodied.control import Motion, Vec3


@runtime_checkable
class Trajectory(Protocol):
    duration_s: float | None

    def sample(self, t_s: float) -> Motion: ...


@dataclass(frozen=True)
class StartState:
    position: Vec3
    velocity: Vec3
    yaw: float | None = None


@dataclass(frozen=True)
class StartTolerance:
    position_m: float
    velocity_mps: float
    yaw_rad: float | None = None


@dataclass(frozen=True)
class ValidityWindow:
    not_before_mono_s: float | None
    not_after_mono_s: float | None
    dependency_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackingEnvelope:
    position_m: float
    velocity_mps: float
    yaw_rad: float | None = None


@dataclass(frozen=True)
class GeometryCertificate:
    nav_epoch: str
    map_revision: str
    volume_refs: tuple[str, ...]
    support_claim: str  # "free" | "unknown"
    checked_at_mono_s: float | None = None

    def __post_init__(self) -> None:
        if self.support_claim not in ("free", "unknown"):
            raise ValueError("support_claim must be 'free' or 'unknown'")


@dataclass(frozen=True)
class TrajectoryCertificate:
    certificate_id: str
    primary: Trajectory
    terminal: Trajectory
    fallbacks: dict[str, Trajectory]
    nav_epoch: str
    start_state: StartState
    start_tolerance: StartTolerance
    validity: ValidityWindow
    tracking_envelope: TrackingEnvelope
    geometry_certificate: GeometryCertificate | None
    safety_evidence_refs: tuple[str, ...]
    plant_limits_ref: str | None
