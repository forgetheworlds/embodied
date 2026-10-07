"""Concrete Trajectory helpers for proofs and tests."""

from __future__ import annotations

from dataclasses import dataclass

from embodied.control import Motion, Vec3


@dataclass(frozen=True)
class HoldTrajectory:
    """Open-ended station: constant position, zero velocity."""

    position: Vec3
    yaw: float | None = None
    duration_s: float | None = None

    def sample(self, t_s: float) -> Motion:
        if t_s < 0.0:
            raise ValueError("trajectory time must be >= 0")
        if self.duration_s is not None and t_s > self.duration_s:
            raise ValueError("t outside finite hold duration")
        return Motion(position=self.position, velocity=Vec3(0.0, 0.0, 0.0), yaw=self.yaw)


@dataclass(frozen=True)
class SegmentTrajectory:
    """Linear position segment over a finite duration."""

    start: Vec3
    end: Vec3
    duration_s: float
    yaw: float | None = None

    def sample(self, t_s: float) -> Motion:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if t_s < 0.0 or t_s > self.duration_s:
            raise ValueError("t outside segment duration")
        alpha = 0.0 if self.duration_s == 0.0 else t_s / self.duration_s
        position = Vec3(
            self.start.x + (self.end.x - self.start.x) * alpha,
            self.start.y + (self.end.y - self.start.y) * alpha,
            self.start.z + (self.end.z - self.start.z) * alpha,
        )
        velocity = Vec3(
            (self.end.x - self.start.x) / self.duration_s,
            (self.end.y - self.start.y) / self.duration_s,
            (self.end.z - self.start.z) / self.duration_s,
        )
        return Motion(position=position, velocity=velocity, yaw=self.yaw)
