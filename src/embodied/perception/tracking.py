"""Gated association across observations, with ambiguity kept visible.

Specification section 8.1 fixes the shape of this module: gate on physically
possible motion and compatible geometry *first*; inside the gate normalize the
spatial residual by its uncertainty; then rank the survivors with the cues that
are actually available — appearance similarity when the detector supplies
descriptors, otherwise image overlap and shape consistency. Do not multiply cue
confidences as independent probabilities, and when no labelled development
associations exist to calibrate the weights, publish an uncalibrated score and
retain the ambiguous alternatives rather than a probability of identity.

That last constraint is the governing one here: no labelled development
associations exist for this stage, so every association this module publishes is
``calibrated=False``, and an alternative within ``AMBIGUITY_MARGIN`` of the best
is retained in ``identity_alternatives``. A track id is not proof of identity
(section 8.3), and this module never claims it is.

Motion is a hypothesis too: a track starts ``static`` and only becomes
``constant_velocity`` when an observation's displacement exceeds the declared
evidence gate, because a mostly stationary object should not acquire a velocity
from measurement noise. Once observations stop, the predicted envelope grows with
time rather than staying a confident point (section 8.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from embodied.contracts import records as R
from embodied.perception import grounding as grounding_module

STATIC = "static"
CONSTANT_VELOCITY = "constant_velocity"

GATE_SIGMA = 3.0
MOTION_EVIDENCE_SIGMA = 2.0
AMBIGUITY_MARGIN = 0.15
NORMAL_AGREEMENT_DEG = 10.0
MIN_SCORE = 0.0


@dataclass(frozen=True)
class GroundedCandidate:
    """A candidate that has already been through the one grounding path."""

    candidate_id: str
    position_odom_m: tuple[float, float, float]
    position_sigma_m: float
    geometry: tuple[float, ...] | None = None
    region: tuple[float, float, float, float] | None = None
    descriptors: tuple[float, ...] | None = None
    selection_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise R.RecordError("a grounded candidate needs a non-empty candidate_id")
        if len(self.position_odom_m) != 3:
            raise R.RecordError("a grounded candidate position is three numbers")
        if not isinstance(self.position_sigma_m, float) or self.position_sigma_m < 0.0:
            raise R.RecordError("a grounded candidate needs a non-negative position sigma")


@dataclass(frozen=True)
class TrackState:
    """What is known about one track, with its motion-model status stated separately."""

    track_id: str
    position_odom_m: tuple[float, float, float]
    position_sigma_m: float
    velocity_mps: tuple[float, float, float]
    motion_model: str
    observation_count: int
    last_stamp_ns: int | None
    last_candidate_id: str
    identity_alternatives: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    region: tuple[float, float, float, float] | None = None
    descriptors: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.motion_model not in (STATIC, CONSTANT_VELOCITY):
            raise R.RecordError(f"{self.motion_model!r} is not a track motion model")


@dataclass(frozen=True)
class Association:
    """One candidate's relation to one track: gated, scored, and never a probability."""

    track_id: str
    candidate_id: str
    gate_passed: bool
    score: float
    calibrated: bool
    residual_m: float | None
    normalized_residual: float | None
    motion_model: str
    identity_alternatives: tuple[str, ...]
    reasons: tuple[str, ...]
    terms: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.calibrated:
            raise R.RecordError(
                "this stage has no labelled development associations, so every published "
                "association score is uncalibrated; claiming otherwise would be false certainty"
            )


def position_sigma_of_geometry(uncertainty: tuple[float, ...] | None) -> float:
    """The in-plane position bound a grounded target's envelope implies.

    For a point target the envelope is (forward, lateral, lateral) and the
    positional bound is the largest of the three. For an aperture the envelope is
    (normal, tangential, edge shrink): the polygon's position on its plane is
    bounded by the tangential term plus the edge uncertainty.
    """
    if uncertainty is None:
        return 0.0
    if len(uncertainty) == 3 and len(uncertainty) >= 3:
        return float(uncertainty[1] + uncertainty[2])
    return float(max(uncertainty))


def _geometry_center(geometry: tuple[float, ...] | None) -> tuple[float, float, float] | None:
    if geometry is None:
        return None
    aperture = grounding_module.parse_aperture(geometry)
    if aperture is not None:
        return aperture.center_odom_m()
    point = grounding_module.parse_point(geometry)
    return point


def candidate_from_target(
    target: R.GroundedTarget, *, candidate_id: str | None = None, region=None, descriptors=None
) -> GroundedCandidate:
    """A grounded candidate built from a GroundedTarget, keeping its envelope."""
    center = _geometry_center(target.geometry)
    if center is None:
        raise R.RecordError("a target with neither point nor aperture geometry cannot be tracked")
    return GroundedCandidate(
        candidate_id=candidate_id or target.target_id,
        position_odom_m=center,
        position_sigma_m=position_sigma_of_geometry(target.uncertainty),
        geometry=tuple(target.geometry),
        region=region,
        descriptors=descriptors,
        selection_id=target.selection_ids[0] if target.selection_ids else None,
    )


def new_track(
    track_id: str, candidate: GroundedCandidate, *, stamp_ns: int
) -> TrackState:
    """A fresh track from one observation: static until motion is supported."""
    return TrackState(
        track_id=track_id,
        position_odom_m=candidate.position_odom_m,
        position_sigma_m=candidate.position_sigma_m,
        velocity_mps=(0.0, 0.0, 0.0),
        motion_model=STATIC,
        observation_count=1,
        last_stamp_ns=stamp_ns,
        last_candidate_id=candidate.candidate_id,
        reasons=("initial observation sets the static model",),
        region=candidate.region,
        descriptors=candidate.descriptors,
    )


def _iou(first, second) -> float | None:
    if first is None or second is None:
        return None
    u0 = max(first[0], second[0])
    v0 = max(first[1], second[1])
    u1 = min(first[2], second[2])
    v1 = min(first[3], second[3])
    if u1 <= u0 or v1 <= v0:
        return 0.0
    intersection = (u1 - u0) * (v1 - v0)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0 else 0.0


def _shape_consistency(
    first: tuple[float, ...] | None, second: tuple[float, ...] | None
) -> float | None:
    """Extent ratio of two grounded geometries, 1.0 when identical."""
    aperture_first = grounding_module.parse_aperture(first) if first else None
    aperture_second = grounding_module.parse_aperture(second) if second else None
    if aperture_first is None or aperture_second is None:
        return None
    ratio = min(aperture_first.width_m, aperture_second.width_m) / max(
        aperture_first.width_m, aperture_second.width_m
    )
    height_ratio = min(aperture_first.height_m, aperture_second.height_m) / max(
        aperture_first.height_m, aperture_second.height_m
    )
    return float(min(ratio, height_ratio))


def _geometry_agrees(first: tuple[float, ...] | None, second: tuple[float, ...] | None) -> bool:
    """Whether two grounded geometries can describe the same surface."""
    if first is None or second is None:
        return True
    aperture_first = grounding_module.parse_aperture(first)
    aperture_second = grounding_module.parse_aperture(second)
    if aperture_first is None or aperture_second is None:
        return True
    first_normal = np.asarray(aperture_first.plane_normal_odom, dtype=np.float64)
    second_normal = np.asarray(aperture_second.plane_normal_odom, dtype=np.float64)
    cosine = float(abs(first_normal @ second_normal))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return angle <= NORMAL_AGREEMENT_DEG


def _descriptor_similarity(first, second) -> float | None:
    if not first or not second or len(first) != len(second):
        return None
    vector_first = np.asarray(first, dtype=np.float64)
    vector_second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(vector_first) * np.linalg.norm(vector_second))
    if denominator == 0.0:
        return None
    return float(0.5 * (1.0 + (vector_first @ vector_second) / denominator))


def associate(
    track: TrackState,
    candidate: GroundedCandidate,
    *,
    stamp_ns: int,
    max_speed_mps: float,
    gate_sigma: float = GATE_SIGMA,
) -> Association:
    """Gate, then score one candidate against one track.

    The gate answers a physical question first: could the target have moved here
    in the elapsed time under the declared bound? Only survivors are ranked, and
    the published score is a mean of the available cues, never a product of
    confidences.
    """
    residual_vector = np.asarray(candidate.position_odom_m, dtype=np.float64) - np.asarray(
        track.position_odom_m, dtype=np.float64
    )
    residual = float(np.linalg.norm(residual_vector))
    combined_sigma = track.position_sigma_m + candidate.position_sigma_m
    dt_s = None
    if track.last_stamp_ns is not None:
        dt_s = (stamp_ns - track.last_stamp_ns) / 1e9
    reasons: list[str] = []
    gate_passed = True
    if dt_s is None:
        reasons.append("no previous stamp, so the motion gate is vacuous")
    elif dt_s < 0.0:
        gate_passed = False
        reasons.append("the observation predates the track's last observation")
    else:
        bound = max_speed_mps * dt_s + gate_sigma * combined_sigma
        if residual > bound:
            gate_passed = False
            reasons.append(
                f"residual {residual:.3f} m exceeds the physically possible bound {bound:.3f} m"
            )
    if not gate_passed:
        return Association(
            track_id=track.track_id,
            candidate_id=candidate.candidate_id,
            gate_passed=False,
            score=MIN_SCORE,
            calibrated=False,
            residual_m=residual,
            normalized_residual=None,
            motion_model=track.motion_model,
            identity_alternatives=(),
            reasons=tuple(reasons),
        )
    normalized = residual / combined_sigma if combined_sigma > 0.0 else None
    spatial = 1.0 if normalized is None else math.exp(-0.5 * normalized * normalized)
    terms: list[tuple[str, float]] = [("spatial_support", spatial)]
    if track.motion_model == STATIC and dt_s is not None:
        terms.append(("static_model_agreement", 1.0 if residual <= MOTION_EVIDENCE_SIGMA * combined_sigma else 0.0))
    return Association(
        track_id=track.track_id,
        candidate_id=candidate.candidate_id,
        gate_passed=True,
        score=float(np.mean([value for _name, value in terms])),
        calibrated=False,
        residual_m=residual,
        normalized_residual=normalized,
        motion_model=track.motion_model,
        identity_alternatives=(),
        reasons=tuple(reasons),
        terms=tuple(terms),
    )


def associate_candidates(
    track: TrackState,
    candidates: tuple[GroundedCandidate, ...],
    *,
    stamp_ns: int,
    max_speed_mps: float,
    geometry: tuple[float, ...] | None = None,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
    gate_sigma: float = GATE_SIGMA,
) -> tuple[Association, ...]:
    """Rank every candidate against one track and record ambiguity among the survivors.

    ``geometry`` is the track's own last grounded geometry; when a candidate and
    the track both carry one, shape consistency is a cue and a normal disagreement
    beyond the declared tolerance fails the gate. When the detector supplies
    descriptors on both sides, appearance similarity is used *instead of* image
    overlap, as section 8.1 requires; otherwise image overlap is used when both
    sides carry a region.
    """
    scored: list[tuple[GroundedCandidate, Association, list[tuple[str, float]]]] = []
    for candidate in candidates:
        association = associate(
            track,
            candidate,
            stamp_ns=stamp_ns,
            max_speed_mps=max_speed_mps,
            gate_sigma=gate_sigma,
        )
        terms = list(association.terms)
        if association.gate_passed and not _geometry_agrees(geometry, candidate.geometry):
            association = Association(
                track_id=track.track_id,
                candidate_id=candidate.candidate_id,
                gate_passed=False,
                score=MIN_SCORE,
                calibrated=False,
                residual_m=association.residual_m,
                normalized_residual=association.normalized_residual,
                motion_model=track.motion_model,
                identity_alternatives=(),
                reasons=association.reasons
                + (f"plane disagreement beyond {NORMAL_AGREEMENT_DEG:.0f} degrees",),
            )
            terms = []
        if association.gate_passed:
            descriptor = _descriptor_similarity(track.descriptors, candidate.descriptors)
            if descriptor is not None:
                terms.append(("appearance_similarity", descriptor))
            else:
                overlap = _iou(track.region, candidate.region)
                if overlap is not None:
                    terms.append(("image_overlap", overlap))
            shape = _shape_consistency(geometry, candidate.geometry)
            if shape is not None:
                terms.append(("shape_consistency", shape))
            association = Association(
                track_id=track.track_id,
                candidate_id=candidate.candidate_id,
                gate_passed=True,
                score=float(np.mean([value for _name, value in terms])) if terms else MIN_SCORE,
                calibrated=False,
                residual_m=association.residual_m,
                normalized_residual=association.normalized_residual,
                motion_model=track.motion_model,
                identity_alternatives=(),
                reasons=association.reasons,
                terms=tuple(terms),
            )
        scored.append((candidate, association, terms))
    ranked = sorted(scored, key=lambda entry: entry[1].score, reverse=True)
    if not ranked:
        return ()
    best_score = ranked[0][1].score
    alternatives = tuple(
        entry[1].candidate_id
        for entry in ranked[1:]
        if entry[1].gate_passed and best_score - entry[1].score < ambiguity_margin
    )
    resolved = []
    for candidate, association, terms in ranked:
        resolved.append(
            Association(
                track_id=association.track_id,
                candidate_id=association.candidate_id,
                gate_passed=association.gate_passed,
                score=association.score,
                calibrated=False,
                residual_m=association.residual_m,
                normalized_residual=association.normalized_residual,
                motion_model=association.motion_model,
                identity_alternatives=alternatives if association.candidate_id == ranked[0][1].candidate_id else (),
                reasons=association.reasons
                + (
                    (f"ambiguous within {ambiguity_margin:.2f} of the best candidate",)
                    if association.candidate_id == ranked[0][1].candidate_id and alternatives
                    else ()
                ),
                terms=association.terms,
            )
        )
    return tuple(resolved)


def observe(
    track: TrackState, association: Association, candidate: GroundedCandidate, *, stamp_ns: int
) -> TrackState:
    """Apply one accepted association, transitioning motion only on evidence."""
    if not association.gate_passed:
        raise R.RecordError("an association that failed its gate is not applied to a track")
    previous = np.asarray(track.position_odom_m, dtype=np.float64)
    current = np.asarray(candidate.position_odom_m, dtype=np.float64)
    displacement = float(np.linalg.norm(current - previous))
    combined_sigma = track.position_sigma_m + candidate.position_sigma_m
    dt_s = None if track.last_stamp_ns is None else (stamp_ns - track.last_stamp_ns) / 1e9
    motion_model = track.motion_model
    velocity = track.velocity_mps
    reasons = list(track.reasons)
    if (
        dt_s is not None
        and dt_s > 0.0
        and displacement > MOTION_EVIDENCE_SIGMA * combined_sigma
    ):
        velocity = tuple(float(value) for value in (current - previous) / dt_s)
        if motion_model != CONSTANT_VELOCITY:
            reasons.append(
                f"displacement {displacement:.3f} m exceeds {MOTION_EVIDENCE_SIGMA:.0f} sigma "
                f"({MOTION_EVIDENCE_SIGMA * combined_sigma:.3f} m): motion transition supported"
            )
        motion_model = CONSTANT_VELOCITY
    return TrackState(
        track_id=track.track_id,
        position_odom_m=candidate.position_odom_m,
        position_sigma_m=candidate.position_sigma_m,
        velocity_mps=velocity,
        motion_model=motion_model,
        observation_count=track.observation_count + 1,
        last_stamp_ns=stamp_ns,
        last_candidate_id=candidate.candidate_id,
        identity_alternatives=association.identity_alternatives,
        reasons=tuple(reasons),
        region=candidate.region,
        descriptors=candidate.descriptors,
    )


def predicted_envelope_m(
    track: TrackState, *, horizon_s: float, speed_uncertainty_mps: float
) -> float:
    """The occupied envelope a moving track will be predicted to need.

    Section 7.1: the dynamic layer's predicted region grows with velocity
    uncertainty and time, so an unobserved moving track becomes more, not less,
    dangerous to the future corridor.
    """
    if horizon_s < 0.0:
        raise R.RecordError("a prediction horizon is not negative")
    return float(track.position_sigma_m + speed_uncertainty_mps * horizon_s)
