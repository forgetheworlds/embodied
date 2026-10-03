"""MapStore: the single writer of local occupancy, and the snapshots readers consume.

Specification section 7 fixes the semantics this module implements, and each one
is a rule rather than a preference:

* Sparse voxel submaps (7.1). Only cells with evidence exist.
* The map writer integrates **valid** depth rays: the interval strictly before a
  valid surface supports free space, the surface band supports possible
  occupancy, and pixels with invalid depth clear nothing.
* Evidence is bounded log-odds, clamped, so repeated correlated returns from one
  frame cannot make a cell certain. Free is published only above the support
  threshold *and* with enough distinct clearing rays *and* while fresh.
  Everything between the thresholds is unknown **with a reason** — never
  observed, stale, or conflicting — and unknown is never free.
* A separate dynamic layer holds moving tracks, with a predicted envelope that
  grows with velocity uncertainty and time; a confident position expires instead
  of burning into the static map.
* Intrusion from unobserved regions is expanded from reachable unknown boundaries
  with the declared dynamic-actor bound, so a blind opening shortens the future
  corridor. If no credible bound is declared, that future-clearance claim is
  marked unsupported rather than assumed safe.
* MapStore publishes immutable snapshots; a reader's view cannot change while it
  computes, and a trajectory names the cell/anchor revisions it depends on so a
  later validation can see whether they moved.

Only MapStore writes. The planner, validator and executor read snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from embodied.contracts import records as R
from embodied.perception import grounding as grounding_module

FREE = "free"
OCCUPIED = "occupied"
UNKNOWN = "unknown"

NEVER_OBSERVED = "never_observed"
STALE = "stale"
CONFLICTING = "conflicting"
SURFACE_BAND = "surface_band"


@dataclass(frozen=True)
class MapConfig:
    """The declared map resolution, bounds and evidence parameters."""

    voxel_m: float
    bounds_odom_m: dict[str, tuple[float, float]]
    surface_band_m: float
    log_odds_hit: float
    log_odds_pass: float
    clamp: float
    free_threshold: float
    occupied_threshold: float
    min_clearing_rays: int
    freshness_s: float
    dynamic_speed_mps: float | None
    dynamic_reach_s: float | None

    @classmethod
    def from_scene(cls, scene: dict) -> "MapConfig":
        """Build the declared configuration from the fixture's own scene document."""
        declared = scene["map"]
        return cls(
            voxel_m=float(declared["voxel_m"]),
            bounds_odom_m={
                axis: (float(low), float(high)) for axis, (low, high) in declared["bounds_odom_m"].items()
            },
            surface_band_m=float(declared["surface_band_m"]),
            log_odds_hit=float(declared["log_odds_hit"]),
            log_odds_pass=float(declared["log_odds_pass"]),
            clamp=float(declared["log_odds_clamp"]),
            free_threshold=float(declared["free_threshold"]),
            occupied_threshold=float(declared["occupied_threshold"]),
            min_clearing_rays=int(declared["min_clearing_rays"]),
            freshness_s=float(declared["freshness_s"]),
            dynamic_speed_mps=float(scene["dynamic"]["speed_mps"]) if scene["dynamic"] else None,
            dynamic_reach_s=float(scene["dynamic"]["reach_s"]) if scene["dynamic"] else None,
        )

    def cell_index(self, point_odom_m: tuple[float, float, float]) -> tuple[int, int, int]:
        """The declared cell rule: index = floor((value - lower_bound) / resolution)."""
        return tuple(
            int(math.floor((value - self.bounds_odom_m[axis][0]) / self.voxel_m))
            for value, axis in zip(point_odom_m, ("x", "y", "z"))
        )

    def cell_center(self, index: tuple[int, int, int]) -> tuple[float, float, float]:
        return tuple(
            self.bounds_odom_m[axis][0] + (index[position] + 0.5) * self.voxel_m
            for position, axis in enumerate(("x", "y", "z"))
        )

    def shape(self) -> tuple[int, int, int]:
        return tuple(
            int(math.floor((self.bounds_odom_m[axis][1] - self.bounds_odom_m[axis][0]) / self.voxel_m))
            for axis in ("x", "y", "z")
        )

    def inside(self, index: tuple[int, int, int]) -> bool:
        return all(
            0 <= index[position] < extent for position, extent in enumerate(self.shape())
        )


@dataclass
class _CellEvidence:
    score: float = 0.0
    clearing_rays: int = 0
    last_pass_ns: int | None = None
    last_hit_ns: int | None = None
    # A cell inside the uncertain surface band supports possible occupancy, never
    # free space: a surface passing through or beside it is not a cleared cell.
    band: bool = False
    frames: set = field(default_factory=set)


@dataclass(frozen=True)
class DynamicTrack:
    """One moving track in the dynamic layer, with its growing predicted envelope."""

    track_id: str
    position_odom_m: tuple[float, float, float]
    position_sigma_m: float
    velocity_mps: tuple[float, float, float]
    motion_model: str
    last_stamp_ns: int | None

    def predicted_envelope_m(self, *, horizon_s: float, speed_uncertainty_mps: float) -> float:
        if horizon_s < 0.0:
            raise R.RecordError("a prediction horizon is not negative")
        return float(self.position_sigma_m + speed_uncertainty_mps * horizon_s)


class MapStore:
    """The single writer of the local occupancy map and of its published snapshots."""

    def __init__(
        self,
        config: MapConfig,
        *,
        submap_id: str,
        nav_epoch: str,
        snapshot_id: str = "snapshot-1",
    ) -> None:
        self.config = config
        self.submap_id = submap_id
        self.nav_epoch = nav_epoch
        self._cells: dict[tuple[int, int, int], _CellEvidence] = {}
        self._revision_number = 0
        self._revision = "rev-0"
        self._snapshot_id = snapshot_id
        self._dynamic: dict[str, DynamicTrack] = {}
        self._data_ages: dict[str, float] = {}
        self.rejections: list[str] = []

    # -- writing ----------------------------------------------------------

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    def integrate(
        self,
        depth,
        capture_pose: R.PoseEstimate,
        calibration: R.Calibration,
        *,
        stamp_ns: int,
        observation_id: str,
        now_ns: int | None = None,
    ) -> str:
        """Integrate one validity-qualified depth product; return the new revision.

        Invalid samples clear nothing: only pixels whose reason is ``valid`` are
        marched. Correlated evidence from one frame is capped by the clamp and by
        the distinct-ray requirement, not trusted as independence.
        """
        if depth is None:
            self.rejections.append("no depth product: nothing integrated, nothing cleared")
            return self._revision
        if not capture_pose.valid:
            self.rejections.append("capture pose invalid: nothing integrated, nothing cleared")
            return self._revision
        if depth.calibration_id != calibration.calibration_id:
            self.rejections.append(
                f"depth names calibration {depth.calibration_id!r}, not "
                f"{calibration.calibration_id!r}: nothing integrated"
            )
            return self._revision
        rows, columns = np.nonzero(depth.valid)
        if rows.size == 0:
            self.rejections.append("no valid depth samples: nothing integrated")
            return self._revision
        pixels = np.stack([columns, rows], axis=1).astype(np.float64)
        depths = np.asarray(depth.depth_m[rows, columns], dtype=np.float64)
        directions = grounding_module.pixel_ray_directions(pixels, calibration)
        directions_odom = directions @ grounding_module.rotation_of(capture_pose).T
        hits = grounding_module.camera_points_to_odom(pixels, depths, calibration, capture_pose)
        camera_center = np.asarray(hits[0], dtype=np.float64) - depths[0] * directions_odom[0]

        resolution = self.config.voxel_m
        band = self.config.surface_band_m
        # Each ray is sampled so that its dominant axis advances by exactly one voxel
        # per step: every cell the ray crosses is visited once, and no cell is skipped
        # because a sample landed on a cell boundary. The ray count is then a genuine
        # count of distinct pixel rays, which is what the three-ray rule needs.
        dominant = np.argmax(np.abs(directions_odom), axis=1)
        axis_lengths = np.abs(directions_odom[np.arange(dominant.size), dominant])
        steps = resolution / axis_lengths
        # Phase the samples onto the *centres* of the dominant-axis cells ahead of the
        # camera. A sample that lands on a cell boundary is decided by floating-point
        # noise, which both double-counts and skips cells; a sample at a cell centre is
        # unambiguous, so every cell the ray crosses is visited exactly once.
        dominant_offsets = np.array(
            [self.config.bounds_odom_m[axis][0] for axis in ("x", "y", "z")], dtype=np.float64
        )
        camera_dominant = camera_center[dominant] - dominant_offsets[dominant]
        camera_cell_along_axis = np.floor(camera_dominant / resolution)
        next_center = (camera_cell_along_axis + 1.5) * resolution
        starts = (next_center - camera_dominant) / axis_lengths
        max_steps = int(np.ceil(float((depths / steps).max()))) + 2
        # One frame's ray march visits the same cell many times over — a 0.1 m
        # voxel a couple of metres out is crossed by dozens of rays — so the
        # visits are packed and collapsed before any cell is written. Walking
        # them one at a time measured 2.27 s per frame against 0.22 s
        # aggregated, which put a perception cycle at 4.86 s against the map's
        # own 5.0 s freshness: cells were refreshed only just ahead of expiring,
        # and the free set decayed in steps. The arithmetic is unchanged — n
        # applications of `max(clamp, score + pass)` are exactly
        # `max(clamp, score + n*pass)` because the clamp is monotone — and the
        # band is idempotent, so deduplicating the hits marks the same cells.
        clearing_batches: list[np.ndarray] = []
        for index in range(max_steps):
            t = starts + index * steps
            alive = t < depths - band / 2.0
            if not bool(alive.any()):
                break
            points = camera_center[None, :] + t[alive, None] * directions_odom[alive]
            clearing_batches.append(self._cell_keys(points))

        for cell, count in self._cell_counts(clearing_batches):
            evidence = self._cells.setdefault(cell, _CellEvidence())
            evidence.frames.add(observation_id)
            evidence.clearing_rays += count
            evidence.score = max(
                -self.config.clamp, evidence.score + count * self.config.log_odds_pass
            )
            evidence.last_pass_ns = stamp_ns
        for cell, count in self._cell_counts([self._cell_keys(hits)]):
            evidence = self._cells.setdefault(cell, _CellEvidence())
            evidence.score = min(
                self.config.clamp, evidence.score + count * self.config.log_odds_hit
            )
            evidence.last_hit_ns = stamp_ns
            for offset in _NEIGHBOUR_OFFSETS:
                neighbour = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
                if self.config.inside(neighbour):
                    self._cells.setdefault(neighbour, _CellEvidence()).band = True
        self._revision_number += 1
        self._revision = f"rev-{self._revision_number}"
        self._drop_expired_dynamic(now_ns if now_ns is not None else stamp_ns)
        return self._revision

    def _cell_keys(self, points: np.ndarray) -> np.ndarray:
        """Packed keys for the in-bounds points, as integers rather than tuples.

        Ray marching only becomes affordable if cell identity stays integer:
        one Python tuple per visit is what made this the dominant cost of a
        perception cycle.
        """
        if points.size == 0:
            return np.empty(0, dtype=np.int64)
        offsets = np.array(
            [self.config.bounds_odom_m[axis][0] for axis in ("x", "y", "z")], dtype=np.float64
        )
        indices = np.floor((points - offsets) / self.config.voxel_m).astype(np.int64)
        extents = np.array(self.config.shape())
        keep = np.all((indices >= 0) & (indices < extents), axis=1)
        if not bool(keep.any()):
            return np.empty(0, dtype=np.int64)
        idx = indices[keep]
        width = int(extents[2])
        plane = int(extents[1]) * width
        return (idx[:, 0] * plane + idx[:, 1] * width + idx[:, 2]).astype(np.int64, copy=False)

    def _cell_counts(
        self, batches: list[np.ndarray]
    ) -> list[tuple[tuple[int, int, int], int]]:
        """Distinct cells across every batch, with the number of visits to each.

        A frame's march is millions of visits over tens of thousands of distinct
        cells, so the visits are counted before the map is written: the update
        loop then runs once per cell rather than once per visit.
        """
        live = [batch for batch in batches if batch.size]
        if not live:
            return []
        uniq, counts = np.unique(np.concatenate(live), return_counts=True)
        extents = self.config.shape()
        width = int(extents[2])
        plane = int(extents[1]) * width
        return [
            (
                (int(key) // plane, (int(key) % plane) // width, int(key) % width),
                int(count),
            )
            for key, count in zip(uniq.tolist(), counts.tolist())
        ]

    def update_dynamic(self, track: DynamicTrack) -> None:
        """Publish or refresh one moving track in the dynamic layer."""
        self._dynamic[track.track_id] = track

    def _drop_expired_dynamic(self, now_ns: int) -> None:
        """A confident moving position expires; it never burns into the static map."""
        for track_id in list(self._dynamic):
            track = self._dynamic[track_id]
            if track.last_stamp_ns is None:
                continue
            if (now_ns - track.last_stamp_ns) / 1e9 > self.config.freshness_s:
                del self._dynamic[track_id]

    def dynamic_envelopes(self, *, horizon_s: float) -> tuple[tuple[str, float, tuple[float, float, float]], ...]:
        """Predicted dynamic envelopes, growing with the declared speed uncertainty."""
        speed_uncertainty = self.config.dynamic_speed_mps or 0.0
        return tuple(
            (track.track_id, track.predicted_envelope_m(
                horizon_s=horizon_s, speed_uncertainty_mps=speed_uncertainty
            ), track.position_odom_m)
            for track in self._dynamic.values()
        )

    # -- reading ----------------------------------------------------------

    def evidence_at(self, cell: tuple[int, int, int]) -> _CellEvidence | None:
        if not self.config.inside(cell):
            return None
        return self._cells.get(cell)

    def classify(self, cell: tuple[int, int, int], *, now_ns: int) -> str:
        """free / occupied / unknown, by the declared thresholds and freshness."""
        evidence = self.evidence_at(cell)
        if evidence is None:
            return UNKNOWN
        age = self._age_s(evidence, now_ns)
        if evidence.score >= self.config.occupied_threshold:
            return OCCUPIED
        if evidence.band:
            # Inside the declared surface band: possible occupancy, never free space.
            return UNKNOWN
        if (
            evidence.score <= -self.config.free_threshold
            and evidence.clearing_rays >= self.config.min_clearing_rays
        ):
            return UNKNOWN if age is None or age > self.config.freshness_s else FREE
        return UNKNOWN

    def unknown_reason(self, cell: tuple[float, float, float] | tuple[int, int, int], *, now_ns: int) -> str:
        """Why a cell is not free: never observed, stale, or conflicting evidence."""
        evidence = self.evidence_at(tuple(int(value) for value in cell))
        if evidence is None:
            return NEVER_OBSERVED
        if evidence.band:
            return SURFACE_BAND
        age = self._age_s(evidence, now_ns)
        if age is not None and age > self.config.freshness_s:
            return STALE
        if evidence.score > -self.config.free_threshold and (
            evidence.last_hit_ns is not None or evidence.clearing_rays > 0
        ):
            return CONFLICTING
        return NEVER_OBSERVED

    def _age_s(self, evidence: _CellEvidence, now_ns: int) -> float | None:
        stamps = [stamp for stamp in (evidence.last_pass_ns, evidence.last_hit_ns) if stamp is not None]
        if not stamps:
            return None
        return (now_ns - max(stamps)) / 1e9

    def unknown_intrusion(self) -> frozenset[tuple[int, int, int]]:
        """Cells a declared dynamic actor could reach from a free-adjacent unknown boundary.

        Section 7.1: expand possible occupied space from *reachable* unknown
        boundaries, so a blind opening beside a route shortens the future free
        corridor before a person is seen. When no credible bound is declared this
        returns no cells and :meth:`intrusion_supported` is False, which is what
        makes a future-clearance claim unsupported rather than assumed safe.
        """
        if not self.intrusion_supported():
            return frozenset()
        reach_m = float(self.config.dynamic_speed_mps) * float(self.config.dynamic_reach_s)
        reach_cells = int(math.ceil(reach_m / self.config.voxel_m))
        newest = self._newest_stamp_ns()
        free = np.zeros(self.config.shape(), dtype=bool)
        boundaries = np.zeros(self.config.shape(), dtype=bool)
        for cell in self._cells:
            if self.classify(cell, now_ns=newest) == FREE:
                free[cell] = True
        shifted = 0
        for offset in _NEIGHBOUR_OFFSETS:
            rolled = np.roll(free, shift=offset, axis=(0, 1, 2))
            boundaries |= free & ~rolled
            shifted += int((free & ~rolled).sum())
        if shifted == 0:
            return frozenset()
        envelope = boundaries.copy()
        for _ in range(reach_cells):
            envelope |= (
                np.roll(envelope, 1, axis=0)
                | np.roll(envelope, -1, axis=0)
                | np.roll(envelope, 1, axis=1)
                | np.roll(envelope, -1, axis=1)
                | np.roll(envelope, 1, axis=2)
                | np.roll(envelope, -1, axis=2)
            )
        reached = np.argwhere(envelope)
        return frozenset(tuple(int(value) for value in cell) for cell in reached)

    def intrusion_supported(self) -> bool:
        """Whether a credible dynamic-actor bound exists for intrusion claims."""
        return (
            self.config.dynamic_speed_mps is not None
            and self.config.dynamic_reach_s is not None
            and self.config.dynamic_speed_mps > 0.0
            and self.config.dynamic_reach_s > 0.0
        )

    def state_of(self, cell: tuple[int, int, int]) -> str:
        """Classify without a caller-supplied clock, using the newest evidence time."""
        newest = self._newest_stamp_ns()
        return self.classify(cell, now_ns=newest)

    def _newest_stamp_ns(self) -> int:
        stamps = [
            stamp
            for evidence in self._cells.values()
            for stamp in (evidence.last_pass_ns, evidence.last_hit_ns)
            if stamp is not None
        ]
        return max(stamps) if stamps else 0

    def _published_free_cells(self) -> set[tuple[int, int, int]]:
        newest = self._newest_stamp_ns()
        return {
            cell
            for cell in self._cells
            if self.classify(cell, now_ns=newest) == FREE
        }

    def free_cells(self, *, now_ns: int | None = None) -> frozenset[tuple[int, int, int]]:
        clock = self._newest_stamp_ns() if now_ns is None else now_ns
        return frozenset(
            cell for cell in self._cells if self.classify(cell, now_ns=clock) == FREE
        )

    def occupied_cells(self, *, now_ns: int | None = None) -> frozenset[tuple[int, int, int]]:
        clock = self._newest_stamp_ns() if now_ns is None else now_ns
        return frozenset(
            cell for cell in self._cells if self.classify(cell, now_ns=clock) == OCCUPIED
        )

    def known_cells(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(sorted(self._cells))

    def data_ages_s(self, *, now_ns: int) -> dict[str, float]:
        ages = {
            f"cell:{cell}": age
            for cell in self._cells
            if (age := self._age_s(self._cells[cell], now_ns)) is not None
        }
        return ages

    def snapshot(
        self, *, navigation_state_ref: str | None, now_ns: int, anchor_transforms: tuple[R.Transform, ...] = ()
    ) -> R.WorldSnapshot:
        """Publish an immutable view from one commit."""
        return R.WorldSnapshot(
            snapshot_id=self._snapshot_id,
            map_revision=self._revision,
            anchor_transforms=anchor_transforms,
            occupancy_ref=f"{self.submap_id}@{self._revision}",
            moving_envelopes=tuple(
                f"{track_id}:{radius:.3f}" for track_id, radius, _position in self.dynamic_envelopes(horizon_s=0.0)
            )
            or None,
            targets=(),
            places=(),
            data_ages=tuple(sorted(self.data_ages_s(now_ns=now_ns).items()))[:64],
            navigation_state_ref=navigation_state_ref,
        )

    def counts(self, *, now_ns: int | None = None) -> dict[str, int]:
        clock = self._newest_stamp_ns() if now_ns is None else now_ns
        counts = {FREE: 0, OCCUPIED: 0, UNKNOWN: 0}
        for cell in self._cells:
            counts[self.classify(cell, now_ns=clock)] += 1
        return counts

    def evidence_breakdown(self, *, now_ns: int) -> dict[str, int]:
        """How the cells holding evidence divide among the rules. Diagnostic only.

        ``map_summary`` reports free cells and search cells, and a map can hold
        many of the first and none of the second. This says which rule withheld
        the rest — the surface band, staleness, a score that never reached the
        free threshold, too few clearing rays — using the same order as
        :meth:`classify` so the tally and the classification cannot disagree.

        It counts and returns; no decision reads it.
        """
        counts = {
            "observed": 0,
            FREE: 0,
            OCCUPIED: 0,
            STALE: 0,
            SURFACE_BAND: 0,
            "score_below_threshold": 0,
            "clearing_rays_short": 0,
            "unstamped": 0,
        }
        for evidence in self._cells.values():
            counts["observed"] += 1
            age = self._age_s(evidence, now_ns)
            if age is None:
                counts["unstamped"] += 1
                continue
            if evidence.score >= self.config.occupied_threshold:
                counts[OCCUPIED] += 1
                continue
            if evidence.band:
                counts[SURFACE_BAND] += 1
                continue
            if age > self.config.freshness_s:
                counts[STALE] += 1
                continue
            if evidence.score > -self.config.free_threshold:
                counts["score_below_threshold"] += 1
                continue
            if evidence.clearing_rays < self.config.min_clearing_rays:
                counts["clearing_rays_short"] += 1
                continue
            counts[FREE] += 1
        return counts


_NEIGHBOUR_OFFSETS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
