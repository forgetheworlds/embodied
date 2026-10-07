"""MappingPort / OccupancyQuery producers over ``memory.world.MapStore``.

Does not invent a second map. Snapshots are immutable for one ``map_revision``.
Autonomy FREE requires ``evidence_class=sensor_derived``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from embodied.contracts.perception_ports import (
    Aabb,
    AgeDomain,
    CellIndex,
    DynamicEnvelope,
    EvidenceClass,
    MapRevision,
    MappingPort,
    NavEpoch,
    OccupancyQuery,
    OccupancySupport,
    OccupancyVerdict,
    SnapshotId,
    Sphere,
    StopTubeQuery,
    Vec3,
)
from embodied.contracts.records import ClockStamp
from embodied.memory import world as world_module
from embodied.memory.world import MapConfig, MapStore
from embodied.perception.camera import DepthProduct
from embodied.contracts.records import Calibration, PoseEstimate


_SUPPORT_RANK = {
    OccupancySupport.OCCUPIED: 3,
    OccupancySupport.UNKNOWN: 2,
    OccupancySupport.UNSUPPORTED: 1,
    OccupancySupport.FREE: 0,
}


def _cell_ref(cell: CellIndex) -> str:
    return f"cell:{cell[0]},{cell[1]},{cell[2]}"


def _point_in_aabb(point: Vec3, box: Aabb) -> bool:
    return all(box.min_m[i] <= point[i] <= box.max_m[i] for i in range(3))


def _point_in_sphere(point: Vec3, sphere: Sphere) -> bool:
    dx = point[0] - sphere.center_m[0]
    dy = point[1] - sphere.center_m[1]
    dz = point[2] - sphere.center_m[2]
    return (dx * dx + dy * dy + dz * dz) <= sphere.radius_m * sphere.radius_m


def _volume_ok_frame(volume: Aabb | Sphere) -> bool:
    return volume.frame == "odom"


@dataclass(frozen=True)
class SnapshotOccupancyQuery:
    """Immutable occupancy handle for one map revision."""

    nav_epoch: NavEpoch
    map_revision: MapRevision
    snapshot_id: SnapshotId
    submap_id: str
    stamp: ClockStamp
    sim_time_s: float | None
    age_s: float
    age_domain: AgeDomain
    healthy: bool
    evidence_class: EvidenceClass
    data_ages_s: tuple[tuple[str, float], ...]
    voxel_m: float
    bounds_odom_m: tuple[tuple[str, float, float], ...]
    cells: tuple[tuple[CellIndex, str], ...]
    envelopes: tuple[DynamicEnvelope, ...]
    intrusion_cells: tuple[CellIndex, ...]
    intrusion_supported: bool
    limitations: tuple[str, ...] = ()

    def _cell_map(self) -> dict[CellIndex, str]:
        return dict(self.cells)

    def matches_epoch(self, nav_epoch: NavEpoch) -> bool:
        return nav_epoch == self.nav_epoch

    def matches_revision(self, map_revision: MapRevision) -> bool:
        return map_revision == self.map_revision

    def _base_verdict(
        self,
        *,
        support: OccupancySupport,
        reason: str,
        now: ClockStamp,
        capabilities: tuple[str, ...],
        limiting_refs: tuple[str, ...] = (),
        free_fraction: float | None = None,
    ) -> OccupancyVerdict:
        # Hard gate: never emit FREE unless sensor_derived.
        if support is OccupancySupport.FREE and self.evidence_class is not EvidenceClass.SENSOR_DERIVED:
            support = OccupancySupport.UNSUPPORTED
            reason = "evidence_class"
            free_fraction = None
        if not self.healthy and support is not OccupancySupport.UNSUPPORTED:
            support = OccupancySupport.UNSUPPORTED
            reason = "unhealthy_snapshot"
            free_fraction = None
        return OccupancyVerdict(
            support=support,
            reason=reason,
            nav_epoch=self.nav_epoch,
            map_revision=self.map_revision,
            snapshot_id=self.snapshot_id,
            query_stamp=now,
            sim_time_s=self.sim_time_s,
            age_s=self.age_s,
            age_domain=self.age_domain,
            evidence_class=self.evidence_class,
            limiting_refs=limiting_refs,
            free_fraction=free_fraction,
            capabilities=capabilities,
            limitations=self.limitations,
        )

    def _raw_label(self, cell: CellIndex) -> str:
        return self._cell_map().get(cell, world_module.UNKNOWN)

    def _label_to_support(self, label: str) -> tuple[OccupancySupport, str]:
        if label == world_module.FREE:
            if self.evidence_class is not EvidenceClass.SENSOR_DERIVED:
                return OccupancySupport.UNSUPPORTED, "evidence_class"
            return OccupancySupport.FREE, "free"
        if label == world_module.OCCUPIED:
            return OccupancySupport.OCCUPIED, "occupied"
        if label == world_module.UNKNOWN:
            return OccupancySupport.UNKNOWN, "never_observed"
        return OccupancySupport.UNKNOWN, "other"

    def _cell_center(self, cell: CellIndex) -> Vec3:
        bounds = {axis: (low, high) for axis, low, high in self.bounds_odom_m}
        return (
            bounds["x"][0] + (cell[0] + 0.5) * self.voxel_m,
            bounds["y"][0] + (cell[1] + 0.5) * self.voxel_m,
            bounds["z"][0] + (cell[2] + 0.5) * self.voxel_m,
        )

    def _cells_in_volume(self, volume: Aabb | Sphere) -> list[CellIndex]:
        """Sparse: only cells with evidence. Empty ⇒ caller gets never_observed/unsupported."""
        hits: list[CellIndex] = []
        for cell in self._cell_map():
            center = self._cell_center(cell)
            if isinstance(volume, Aabb):
                if _point_in_aabb(center, volume):
                    hits.append(cell)
            elif _point_in_sphere(center, volume):
                hits.append(cell)
        return hits

    def _aggregate(
        self,
        cells: Iterable[CellIndex],
        *,
        now: ClockStamp,
        capabilities: tuple[str, ...],
        empty_reason: str,
    ) -> OccupancyVerdict:
        if not self.healthy:
            return self._base_verdict(
                support=OccupancySupport.UNSUPPORTED,
                reason="unhealthy_snapshot",
                now=now,
                capabilities=capabilities,
            )
        cell_list = list(cells)
        if not cell_list:
            return self._base_verdict(
                support=OccupancySupport.UNSUPPORTED,
                reason=empty_reason,
                now=now,
                capabilities=capabilities,
            )
        best = OccupancySupport.FREE
        reason = "free"
        limiting: list[str] = []
        free_count = 0
        for cell in cell_list:
            support, cell_reason = self._label_to_support(self._raw_label(cell))
            if support is OccupancySupport.FREE:
                free_count += 1
            if _SUPPORT_RANK[support] > _SUPPORT_RANK[best]:
                best = support
                reason = cell_reason
                limiting = [_cell_ref(cell)]
            elif support is best and support is not OccupancySupport.FREE:
                limiting.append(_cell_ref(cell))
        fraction = free_count / len(cell_list)
        return self._base_verdict(
            support=best,
            reason=reason,
            now=now,
            capabilities=capabilities,
            limiting_refs=tuple(limiting[:16]),
            free_fraction=fraction if best is OccupancySupport.FREE else fraction,
        )

    def classify_cell(self, cell: CellIndex, *, now: ClockStamp) -> OccupancyVerdict:
        if not self.healthy:
            return self._base_verdict(
                support=OccupancySupport.UNSUPPORTED,
                reason="unhealthy_snapshot",
                now=now,
                capabilities=("classify_cell",),
            )
        support, reason = self._label_to_support(self._raw_label(cell))
        return self._base_verdict(
            support=support,
            reason=reason,
            now=now,
            capabilities=("classify_cell",),
            limiting_refs=() if support is OccupancySupport.FREE else (_cell_ref(cell),),
            free_fraction=1.0 if support is OccupancySupport.FREE else 0.0,
        )

    def query_cells(self, cells: tuple[CellIndex, ...], *, now: ClockStamp) -> OccupancyVerdict:
        if not cells:
            return self._base_verdict(
                support=OccupancySupport.UNSUPPORTED,
                reason="other",
                now=now,
                capabilities=("cells",),
            )
        return self._aggregate(cells, now=now, capabilities=("cells",), empty_reason="other")

    def query_volume(self, volume: Aabb | Sphere, *, now: ClockStamp) -> OccupancyVerdict:
        if not _volume_ok_frame(volume):
            return self._base_verdict(
                support=OccupancySupport.UNSUPPORTED,
                reason="frame_mismatch",
                now=now,
                capabilities=("volume",),
            )
        return self._aggregate(
            self._cells_in_volume(volume),
            now=now,
            capabilities=("volume",),
            empty_reason="never_observed",
        )

    def query_stop_tube(self, tube: StopTubeQuery, *, now: ClockStamp) -> OccupancyVerdict:
        if not self.healthy:
            return self._base_verdict(
                support=OccupancySupport.UNSUPPORTED,
                reason="unhealthy_snapshot",
                now=now,
                capabilities=("stop_tube",),
            )
        cells: set[CellIndex] = set()
        for sample in tube.samples_odom_m:
            sphere = Sphere(center_m=sample, radius_m=tube.envelope_radius_m, frame="odom")
            cells.update(self._cells_in_volume(sphere))
        if tube.include_brake_region and tube.brake_region is not None:
            if not _volume_ok_frame(tube.brake_region):
                return self._base_verdict(
                    support=OccupancySupport.UNSUPPORTED,
                    reason="frame_mismatch",
                    now=now,
                    capabilities=("stop_tube",),
                )
            cells.update(self._cells_in_volume(tube.brake_region))
        return self._aggregate(
            cells,
            now=now,
            capabilities=("stop_tube",),
            empty_reason="never_observed",
        )

    def query_fov_clearance(
        self,
        origin_m: Vec3,
        direction_unit: Vec3,
        max_range_m: float,
        *,
        now: ClockStamp,
    ) -> OccupancyVerdict:
        del origin_m, direction_unit, max_range_m
        return self._base_verdict(
            support=OccupancySupport.UNSUPPORTED,
            reason="missing_fov_model",
            now=now,
            capabilities=("fov",),
        )

    def query_unknown_intrusion(
        self,
        region: Aabb | Sphere,
        *,
        dynamic_speed_mps: float | None,
        dynamic_reach_s: float | None,
        now: ClockStamp,
    ) -> OccupancyVerdict:
        if not _volume_ok_frame(region):
            return self._base_verdict(
                support=OccupancySupport.UNSUPPORTED,
                reason="frame_mismatch",
                now=now,
                capabilities=("intrusion",),
            )
        bound_ok = (
            self.intrusion_supported
            and dynamic_speed_mps is not None
            and dynamic_reach_s is not None
            and dynamic_speed_mps > 0.0
            and dynamic_reach_s > 0.0
        )
        if not bound_ok:
            return self._base_verdict(
                support=OccupancySupport.UNSUPPORTED,
                reason="no_dynamic_bound",
                now=now,
                capabilities=("intrusion",),
            )
        region_cells = set(self._cells_in_volume(region))
        hits = [cell for cell in self.intrusion_cells if cell in region_cells]
        if hits:
            return self._base_verdict(
                support=OccupancySupport.UNKNOWN,
                reason="dynamic_intrusion",
                now=now,
                capabilities=("intrusion",),
                limiting_refs=tuple(_cell_ref(cell) for cell in hits[:16]),
            )
        return self._base_verdict(
            support=OccupancySupport.FREE if self.evidence_class is EvidenceClass.SENSOR_DERIVED else OccupancySupport.UNSUPPORTED,
            reason="free" if self.evidence_class is EvidenceClass.SENSOR_DERIVED else "evidence_class",
            now=now,
            capabilities=("intrusion",),
        )

    def dynamic_envelopes(self, *, horizon_s: float, now: ClockStamp) -> tuple[DynamicEnvelope, ...]:
        del now
        return tuple(env for env in self.envelopes if env.horizon_s <= horizon_s or horizon_s <= 0.0)


@dataclass
class StubOccupancyQuery:
    """Full method surface; every query is unsupported."""

    nav_epoch: NavEpoch = "stub-epoch"
    map_revision: MapRevision = "stub-revision"
    snapshot_id: SnapshotId = "stub-snapshot"
    submap_id: str = "stub-submap"
    stamp: ClockStamp | None = None
    sim_time_s: float | None = None
    age_s: float = 0.0
    age_domain: AgeDomain = AgeDomain.MONOTONIC
    healthy: bool = False
    evidence_class: EvidenceClass = EvidenceClass.UNAVAILABLE
    data_ages_s: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.stamp is None:
            object.__setattr__(
                self,
                "stamp",
                ClockStamp(host_id="test", clock_id="host/monotonic", monotonic_ns=0),
            )

    def _unsupported(self, now: ClockStamp, reason: str, *capabilities: str) -> OccupancyVerdict:
        return OccupancyVerdict(
            support=OccupancySupport.UNSUPPORTED,
            reason=reason,
            nav_epoch=self.nav_epoch,
            map_revision=self.map_revision,
            snapshot_id=self.snapshot_id,
            query_stamp=now,
            sim_time_s=self.sim_time_s,
            age_s=self.age_s,
            age_domain=self.age_domain,
            evidence_class=self.evidence_class,
            limiting_refs=(),
            free_fraction=None,
            capabilities=capabilities,
            limitations=("stub_producer",),
        )

    def matches_epoch(self, nav_epoch: NavEpoch) -> bool:
        return nav_epoch == self.nav_epoch

    def matches_revision(self, map_revision: MapRevision) -> bool:
        return map_revision == self.map_revision

    def query_volume(self, volume: Aabb | Sphere, *, now: ClockStamp) -> OccupancyVerdict:
        del volume
        return self._unsupported(now, "stub_producer", "volume")

    def query_stop_tube(self, tube: StopTubeQuery, *, now: ClockStamp) -> OccupancyVerdict:
        del tube
        return self._unsupported(now, "stub_producer", "stop_tube")

    def query_cells(self, cells: tuple[CellIndex, ...], *, now: ClockStamp) -> OccupancyVerdict:
        del cells
        return self._unsupported(now, "stub_producer", "cells")

    def classify_cell(self, cell: CellIndex, *, now: ClockStamp) -> OccupancyVerdict:
        del cell
        return self._unsupported(now, "stub_producer", "classify_cell")

    def query_fov_clearance(
        self,
        origin_m: Vec3,
        direction_unit: Vec3,
        max_range_m: float,
        *,
        now: ClockStamp,
    ) -> OccupancyVerdict:
        del origin_m, direction_unit, max_range_m
        return self._unsupported(now, "missing_fov_model", "fov")

    def query_unknown_intrusion(
        self,
        region: Aabb | Sphere,
        *,
        dynamic_speed_mps: float | None,
        dynamic_reach_s: float | None,
        now: ClockStamp,
    ) -> OccupancyVerdict:
        del region, dynamic_speed_mps, dynamic_reach_s
        return self._unsupported(now, "no_dynamic_bound", "intrusion")

    def dynamic_envelopes(self, *, horizon_s: float, now: ClockStamp) -> tuple[DynamicEnvelope, ...]:
        del horizon_s, now
        return ()


class StubMappingPort:
    def __init__(self, query: OccupancyQuery | None = None) -> None:
        self._query = query if query is not None else StubOccupancyQuery()

    def occupancy(self) -> OccupancyQuery | None:
        return self._query

    def current_revision(self) -> MapRevision | None:
        query = self._query
        return None if query is None else query.map_revision


class NullMappingPort:
    """Not wired / writer down."""

    def occupancy(self) -> OccupancyQuery | None:
        return None

    def current_revision(self) -> MapRevision | None:
        return None


class MapStoreMappingPort:
    """Sole map writer adapter: publishes immutable OccupancyQuery snapshots."""

    def __init__(
        self,
        store: MapStore,
        *,
        nav_epoch: NavEpoch,
        evidence_class: EvidenceClass,
        stamp: ClockStamp,
        sim_time_s: float | None = None,
        age_s: float = 0.0,
        age_domain: AgeDomain = AgeDomain.SIM_CONTROL,
        limitations: tuple[str, ...] = ("shared_sensor_stereo_imu",),
    ) -> None:
        self._store = store
        self._nav_epoch = nav_epoch
        self._evidence_class = evidence_class
        self._stamp = stamp
        self._sim_time_s = sim_time_s
        self._age_s = age_s
        self._age_domain = age_domain
        self._limitations = limitations
        self._healthy = True

    def set_nav_epoch(self, nav_epoch: NavEpoch) -> None:
        self._nav_epoch = nav_epoch

    def set_evidence_class(self, evidence_class: EvidenceClass) -> None:
        self._evidence_class = evidence_class

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    def integrate(
        self,
        depth: DepthProduct,
        capture_pose: PoseEstimate,
        calibration: Calibration,
        *,
        stamp_ns: int,
        observation_id: str,
        now_ns: int | None = None,
        nav: object | None = None,
    ) -> MapRevision:
        """Integrate only when autonomy capture pose is sensor-derived and valid.

        ``nav`` may be a stacked ``NavigationState``; when provided, pose-assisted /
        oracle / invalid nav refuses FREE-producing integrate (no-op revision).
        """
        from embodied.contracts.perception_ports import NavigationState, NavStatus

        if nav is not None:
            if not isinstance(nav, NavigationState):
                raise TypeError("nav must be a stacked NavigationState when provided")
            if (
                nav.evidence_class is not EvidenceClass.SENSOR_DERIVED
                or not nav.valid
                or nav.status not in (NavStatus.HEALTHY, NavStatus.DEGRADED)
                or not nav.pose.valid
            ):
                self._store.rejections.append(
                    "capture nav not sensor_derived/valid: nothing integrated, nothing cleared"
                )
                return self._store.revision
            capture_pose = PoseEstimate(
                parent_frame=nav.pose.parent_frame,
                child_frame=nav.pose.child_frame,
                stamp=nav.pose.stamp,
                position_m=nav.pose.position_m,
                quaternion_wxyz=nav.pose.quaternion_wxyz,
                covariance=nav.pose.covariance,
                nav_epoch=nav.pose.nav_epoch,
                source_ids=nav.pose.source_ids,
                valid=True,
            )
            self._nav_epoch = nav.nav_epoch
            self._evidence_class = EvidenceClass.SENSOR_DERIVED
        elif not capture_pose.valid:
            self._store.rejections.append("capture pose invalid: nothing integrated, nothing cleared")
            return self._store.revision

        # Depth pose provenance must also be sensor-derived for autonomy FREE path.
        label = getattr(depth.pose_provenance, "label", "")
        if label and label != "SENSOR_DERIVED" and self._evidence_class is EvidenceClass.SENSOR_DERIVED:
            # Keep integrate for diagnostic maps only when evidence_class downgraded.
            self._evidence_class = EvidenceClass.POSE_ASSISTED

        return self._store.integrate(
            depth,
            capture_pose,
            calibration,
            stamp_ns=stamp_ns,
            observation_id=observation_id,
            now_ns=now_ns,
        )

    def _publish(self) -> SnapshotOccupancyQuery:
        store = self._store
        now_ns = store._newest_stamp_ns()  # noqa: SLF001 — snapshot clock from writer
        cells = tuple(
            (cell, store.classify(cell, now_ns=now_ns)) for cell in store.known_cells()
        )
        envelopes = tuple(
            DynamicEnvelope(
                track_id=track_id,
                center_m=(float(position[0]), float(position[1]), float(position[2])),
                radius_m=float(radius),
                horizon_s=0.0,
                frame="odom",
            )
            for track_id, radius, position in store.dynamic_envelopes(horizon_s=0.0)
        )
        intrusion = tuple(sorted(store.unknown_intrusion()))
        bounds = tuple(
            (axis, float(low), float(high)) for axis, (low, high) in store.config.bounds_odom_m.items()
        )
        ages = tuple(sorted(store.data_ages_s(now_ns=now_ns).items())[:64])
        return SnapshotOccupancyQuery(
            nav_epoch=self._nav_epoch,
            map_revision=store.revision,
            snapshot_id=store.snapshot_id,
            submap_id=store.submap_id,
            stamp=self._stamp,
            sim_time_s=self._sim_time_s,
            age_s=self._age_s,
            age_domain=self._age_domain,
            healthy=self._healthy and bool(cells),
            evidence_class=self._evidence_class,
            data_ages_s=ages,
            voxel_m=float(store.config.voxel_m),
            bounds_odom_m=bounds,
            cells=cells,
            envelopes=envelopes,
            intrusion_cells=intrusion,
            intrusion_supported=store.intrusion_supported(),
            limitations=self._limitations,
        )

    def occupancy(self) -> OccupancyQuery | None:
        if not self._healthy:
            return replace(self._publish(), healthy=False)
        if not self._store.known_cells() and self._evidence_class is EvidenceClass.UNAVAILABLE:
            return StubOccupancyQuery(
                nav_epoch=self._nav_epoch,
                map_revision=self._store.revision,
                snapshot_id=self._store.snapshot_id,
                submap_id=self._store.submap_id,
                stamp=self._stamp,
                sim_time_s=self._sim_time_s,
                age_s=self._age_s,
                age_domain=self._age_domain,
                healthy=False,
                evidence_class=EvidenceClass.UNAVAILABLE,
            )
        return self._publish()

    def current_revision(self) -> MapRevision | None:
        return self._store.revision


def snapshot_from_cell_labels(
    *,
    labels: Mapping[CellIndex, str],
    config: MapConfig,
    nav_epoch: NavEpoch,
    map_revision: MapRevision,
    snapshot_id: SnapshotId,
    stamp: ClockStamp,
    evidence_class: EvidenceClass,
    healthy: bool = True,
    submap_id: str = "test-submap",
    sim_time_s: float | None = 0.0,
    age_s: float = 0.0,
    age_domain: AgeDomain = AgeDomain.SIM_CONTROL,
    intrusion_cells: tuple[CellIndex, ...] = (),
    intrusion_supported: bool = False,
) -> SnapshotOccupancyQuery:
    """Contract-test helper: build a snapshot without Webots."""
    bounds = tuple((axis, float(low), float(high)) for axis, (low, high) in config.bounds_odom_m.items())
    return SnapshotOccupancyQuery(
        nav_epoch=nav_epoch,
        map_revision=map_revision,
        snapshot_id=snapshot_id,
        submap_id=submap_id,
        stamp=stamp,
        sim_time_s=sim_time_s,
        age_s=age_s,
        age_domain=age_domain,
        healthy=healthy,
        evidence_class=evidence_class,
        data_ages_s=(),
        voxel_m=float(config.voxel_m),
        bounds_odom_m=bounds,
        cells=tuple(sorted(labels.items())),
        envelopes=(),
        intrusion_cells=intrusion_cells,
        intrusion_supported=intrusion_supported,
        limitations=(),
    )


_: type[MappingPort] = StubMappingPort
__: type[OccupancyQuery] = StubOccupancyQuery
