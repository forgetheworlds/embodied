"""The pinned goal-conditioned candidate seam: one candidate model for every arm.

Specification section 8.1 makes one decision here: a single pretrained
text-conditioned candidate model supplies image regions, labels and reusable
appearance features for B0/B1/B2 alike, and a mission query conditions candidate
supply without ever supplying hidden target coordinates. This module is that
seam. It is deliberately the only place a candidate can come from, so no second
route into grounding exists.

**The default is an explicit refusal, not a substitute.** The pinned dependency
set (pyproject.toml) contains no offline text-conditioned detector, so
:class:`PinnedDetector` returns :class:`DetectorUnavailable` and grounding turns
that into a ``detector_unavailable`` refusal. Adding a dependency is an owner
decision, not a worker's silent escalation, so this module also provides
:func:`verify_available_detector`, a bounded verification that reports what is
actually importable on this host, whether it is offline, and the license it
carries — the artifact an owner decision needs. It reports; it does not install
and it does not integrate.

**Diagnostic injection is labelled.** :class:`InjectedDetector` lets a test (or
a fixture) supply authored candidates. Every candidate it produces carries
``source="injected-diagnostic"`` and no checkpoint hash, so an injected box can
never be mistaken for model output or used to claim a detector exists. The
grounding path then treats it exactly like a model candidate: the geometry comes
from depth and pose, never from the candidate itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, util
from typing import Sequence

# Packages that could in principle provide a locally loadable, offline,
# text-conditioned detector. Checked by name only: find_spec does not import, so
# this never pays a heavy framework import and never changes the environment.
_CANDIDATE_PACKAGES = (
    "groundingdino",
    "transformers",
    "ultralytics",
    "supervision",
    "torch",
    "torchvision",
    "open_clip",
    "onnxruntime",
)

DIAGNOSTIC_SOURCE = "injected-diagnostic"


@dataclass(frozen=True)
class CandidateProvenance:
    """Where one candidate came from, in the form a manifest can record."""

    model_id: str
    checkpoint_hash: str | None
    license_reference: str | None
    offline: bool
    source: str

    def __post_init__(self) -> None:
        for name in ("model_id", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"candidate provenance {name} must be a non-empty string")
        for name in ("checkpoint_hash", "license_reference"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"candidate provenance {name} must be a non-empty string or None")
        if not isinstance(self.offline, bool):
            raise ValueError("candidate provenance offline must be a bool")


@dataclass(frozen=True)
class Candidate:
    """One candidate region the seam proposes for a query.

    ``region`` is a box in the source image's own coordinate convention, the same
    convention a VisualSelection uses, so a candidate can be grounded by the one
    path without a conversion step. ``descriptors`` are reusable appearance
    features when the model supplies them; without them the tracker ranks by
    image overlap and shape consistency instead.
    """

    candidate_id: str
    region: tuple[float, float, float, float] | None
    score: float | None
    provenance: CandidateProvenance
    descriptors: tuple[float, ...] | None = None
    mask_ref: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("a candidate needs a non-empty candidate_id")
        if self.region is not None:
            if len(self.region) != 4:
                raise ValueError("a candidate region is (u_min, v_min, u_max, v_max)")
            u0, v0, u1, v1 = self.region
            if not (u0 < u1 and v0 < v1):
                raise ValueError("a candidate region must have positive extent")
        if self.region is None and self.mask_ref is None:
            raise ValueError("a candidate carries a region, a mask reference, or both")
        if self.score is not None and not isinstance(self.score, float):
            raise ValueError("a candidate score is an uncalibrated float or None")
        if not isinstance(self.provenance, CandidateProvenance):
            raise ValueError("a candidate needs its provenance")
        if self.descriptors is not None and not self.descriptors:
            raise ValueError("candidate descriptors are absent or non-empty")


@dataclass(frozen=True)
class DetectorUnavailable:
    """The seam has no verified candidate model, so it proposes nothing."""

    reason: str
    detail: str

    def __post_init__(self) -> None:
        for name in ("reason", "detail"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"DetectorUnavailable {name} must be a non-empty string")


class PinnedDetector:
    """The pinned seam: no verified checkpoint, therefore no candidates.

    Refusing is the honest behaviour while no offline, license-compatible,
    text-conditioned model is verified on this host: a fabricated box here would
    silently become geometry downstream.
    """

    model_id = "pinned-goal-conditioned-1"

    def candidates(self, image, query: str) -> tuple[Candidate, ...] | DetectorUnavailable:
        return DetectorUnavailable(
            reason="detector_unavailable",
            detail=(
                f"pinned seam {self.model_id!r} carries no verified checkpoint: the pinned "
                "dependency set contains no offline text-conditioned detector, and adding one "
                "is an owner decision; grounding refuses rather than substituting an estimate"
            ),
        )


class InjectedDetector:
    """A labelled DIAGNOSTIC seam that returns authored candidates.

    Used by tests and the fixture to exercise the phrase wiring end to end. Its
    candidates carry no checkpoint hash and are marked ``injected-diagnostic``,
    so no result reached through it can be reported as model output.
    """

    model_id = "injected-diagnostic-1"

    def __init__(self, *candidates: Candidate) -> None:
        if not candidates:
            raise ValueError("an injected detector needs at least one authored candidate")
        self._candidates = tuple(candidates)

    def candidates(self, image, query: str) -> tuple[Candidate, ...] | DetectorUnavailable:
        return self._candidates


def injected_candidate(
    candidate_id: str,
    region: tuple[float, float, float, float],
    *,
    score: float | None = None,
    label: str | None = None,
    descriptors: tuple[float, ...] | None = None,
) -> Candidate:
    """One authored DIAGNOSTIC candidate, with provenance that says so."""
    return Candidate(
        candidate_id=candidate_id,
        region=region,
        score=score,
        provenance=CandidateProvenance(
            model_id=InjectedDetector.model_id,
            checkpoint_hash=None,
            license_reference=None,
            offline=True,
            source=DIAGNOSTIC_SOURCE,
        ),
        descriptors=descriptors,
        label=label,
    )


def verify_available_detector() -> dict:
    """Bounded verification: what this host could actually load, and under what license.

    Reports only. A package that is importable is not thereby verified as
    offline, text-conditioned, license-compatible or fast enough; each of those
    is a separate check that a real integration would have to pass, and the
    report says so rather than implying a detector exists.
    """
    report: dict = {
        "seam_model_id": PinnedDetector.model_id,
        "verdict": "no verified detector; the seam refuses (detector_unavailable)",
        "packages": {},
        "not_verified_by_this_report": (
            "text-conditioned capability, offline operation, licence compatibility, per-frame "
            "cost and checkpoint hash are separate checks; importability proves none of them"
        ),
    }
    for name in _CANDIDATE_PACKAGES:
        entry: dict = {"importable": util.find_spec(name) is not None}
        if entry["importable"]:
            try:
                package_metadata = metadata.metadata(name)
            except metadata.PackageNotFoundError:
                entry["license"] = None
                entry["version"] = None
            else:
                entry["license"] = package_metadata.get("License")
                entry["version"] = package_metadata.get("Version")
        report["packages"][name] = entry
    return report


def summarise(candidates_or_unavailable) -> str:
    """A one-line rendering for a receipt or a log."""
    if isinstance(candidates_or_unavailable, DetectorUnavailable):
        return f"detector unavailable: {candidates_or_unavailable.reason}"
    return f"{len(candidates_or_unavailable)} candidates"


def candidate_regions(candidates: Sequence[Candidate]) -> tuple[tuple[float, float, float, float], ...]:
    """Every candidate region, in order; a candidate without a region is skipped."""
    return tuple(candidate.region for candidate in candidates if candidate.region is not None)
