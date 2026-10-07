"""PlantLimits / TelemetryHealth sibling ports (minimal; no theater)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from embodied.contracts.perception_ports import AgeDomain
from embodied.contracts.records import ClockStamp


@dataclass(frozen=True)
class PlantLimits:
    max_accel_mps2: float
    max_brake_mps2: float
    max_speed_mps: float
    max_yaw_rate_radps: float | None
    max_jerk_mps3: float | None
    assumed_response_delay_s: float
    envelope_radius_m: float
    source: str  # declared | measured | stub
    valid: bool
    stamp: ClockStamp | None
    age_s: float | None
    age_domain: AgeDomain | None


@dataclass(frozen=True)
class TelemetryHealth:
    telemetry_age_s: float
    age_domain: AgeDomain
    link_ok: bool
    failsafe_active: bool
    failsafe_reason: str | None
    mode_guided: bool
    ekf_flags_ok: bool | None
    battery_ok: bool | None
    stamp: ClockStamp | None
    valid: bool


@runtime_checkable
class PlantLimitsPort(Protocol):
    def latest(self) -> PlantLimits | None: ...


@runtime_checkable
class TelemetryHealthPort(Protocol):
    def latest(self) -> TelemetryHealth | None: ...


class StaticPlantLimitsPort:
    def __init__(self, limits: PlantLimits | None) -> None:
        self._limits = limits

    def latest(self) -> PlantLimits | None:
        return self._limits


class StaticTelemetryHealthPort:
    def __init__(self, health: TelemetryHealth | None) -> None:
        self._health = health

    def latest(self) -> TelemetryHealth | None:
        return self._health


def declared_plant(*, envelope_radius_m: float = 0.35, valid: bool = True) -> PlantLimits:
    return PlantLimits(
        max_accel_mps2=2.0,
        max_brake_mps2=2.5,
        max_speed_mps=2.0,
        max_yaw_rate_radps=1.0,
        max_jerk_mps3=None,
        assumed_response_delay_s=0.05,
        envelope_radius_m=envelope_radius_m,
        source="declared",
        valid=valid,
        stamp=None,
        age_s=None,
        age_domain=None,
    )
