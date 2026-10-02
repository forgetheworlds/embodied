"""The first-indoor mission: the B0 executive policy and its evidence machinery.

Three responsibilities, all agent-side:

* :class:`ConventionalColourProposer` — the shared candidate source every
  compared arm uses (owner ruling on the P05 detector question, 2026-10-01).
  It implements the pinned seam's ``candidates(image, query)`` protocol
  deterministically from the aircraft's own frames: HSV colour windows,
  connected components and shape filters, no new dependency, no truth, no
  route encoding. It proposes; it never decides — selection and intent stay
  with the executive. Its provenance label ``conventional-hsv-1`` travels on
  every candidate, and the deviation from specification 8.1 (one pretrained
  text-conditioned model) is recorded beside the parameters: the pinned
  dependency set has no such model and ``perception.detector`` refuses by
  design, so this declared engineering parameter (R2) supplies the candidate
  stream for **all** arms until a verified model lands, at which point it
  replaces the proposer for every arm at once. Its thresholds were chosen
  from the textbook HSV ranges, not tuned on any evaluation scene, and were
  never adjusted after seeing one.
* :func:`interpret_instruction` — the single-clause mission interpretation
  that turns the suite's instruction text into the public MissionContract
  requirements without ever touching hidden facts.
* :func:`build_b0_recipes` and :func:`assemble_mission_claims` — the B0
  conventional policy expressed through the shared :class:`RecipeRunner`:
  bounded frontier exploration, candidate inspection and the return, then a
  final report whose claims are states with evidence, never requested facts
  presented as observed ones.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from embodied.contracts.records import (
    ClaimKind,
    ClockStamp,
    FinalReport,
    MissionContract,
    ReportClaim,
)
from embodied.perception.detector import (
    Candidate,
    CandidateProvenance,
    DetectorUnavailable,
)
from embodied.pilot.recipe_runner import (
    Guard,
    MissionRecipe,
    RecipeStep,
    TargetSelector,
)

# ---------------------------------------------------------------------------
# The shared candidate source (declared engineering parameters, R2)
# ---------------------------------------------------------------------------

# HSV windows in OpenCV's hue scale (0..179 for uint8 images). The textbook
# ranges; "red" wraps the hue circle so it needs two windows. These are the
# proposer's declared parameters, frozen before the first live run and not
# tuned on evaluation scenes (owner condition on the 2026-10-01 ruling).
COLOUR_HUE_WINDOWS: dict[str, tuple[tuple[int, int], ...]] = {
    "red": ((0, 8), (172, 180)),
    "orange": ((9, 19),),
    "yellow": ((20, 33),),
    "green": ((40, 85),),
    "blue": ((95, 130),),
    "purple": ((135, 165),),
}
# Saturation and value floors: a candidate must be a saturated, bright patch,
# not a dim reflection of the room's light. 0.45/0.30 of full scale.
MIN_SATURATION = int(0.45 * 255.0)
MIN_VALUE = int(0.30 * 255.0)
# A candidate needs enough pixels for its centre to carry valid depth samples;
# below this the region is texture noise at the resolution that matters here.
MIN_AREA_PX = 150
# At most this many candidates per frame, largest first, so one noisy wall
# cannot flood the goal table.
MAX_CANDIDATES_PER_FRAME = 4
# Shape filters per shape word: (aspect min, aspect max, bbox fill min). The
# fill fraction is the component's area over its bounding box's area: a solid
# block fills most of its box, a sphere fills pi/4 ~ 0.79, thin or diagonal
# shapes fill less. "block" is the only shape the first-indoor family names.
SHAPE_FILTERS: dict[str, tuple[float, float, float]] = {
    "block": (0.50, 2.00, 0.55),
    "box": (0.50, 2.00, 0.55),
    "cube": (0.50, 2.00, 0.55),
    "sphere": (0.60, 1.67, 0.40),
    "ball": (0.60, 1.67, 0.40),
}

PROPOSER_SOURCE = "conventional-hsv-1"


def _parse_query(query: str) -> tuple[str | None, str | None]:
    """The colour and shape words of one candidate query, lowercased.

    A query with no colour word names no colour window, so no candidate can be
    proposed from it: that is a refusal, not a guess.
    """
    words = re.findall(r"[a-z]+", (query or "").lower())
    colour = next((word for word in words if word in COLOUR_HUE_WINDOWS), None)
    shape = next((word for word in words if word in SHAPE_FILTERS), None)
    return colour, shape


class ConventionalColourProposer:
    """Deterministic colour-and-shape candidate source, shared by every arm.

    Implements the pinned detector seam's protocol (``candidates(image,
    query)``) so it can be handed to the same grounding path unchanged. The
    images are the aircraft's own frames and nothing else enters here.
    """

    model_id = PROPOSER_SOURCE

    def candidates(self, image, query: str) -> tuple[Candidate, ...] | DetectorUnavailable:
        import cv2  # lazy: the same pinned OpenCV the depth worker uses
        import numpy as np

        colour, shape = _parse_query(query)
        if colour is None:
            return DetectorUnavailable(
                reason="query_names_no_colour",
                detail=(
                    f"the query {query!r} names no colour this proposer windows; it refuses "
                    "rather than proposing regions of an unnamed colour"
                ),
            )
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            return DetectorUnavailable(
                reason="no_image",
                detail="no RGB frame was supplied, so no candidate can be proposed",
            )
        hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        colour_window = np.zeros_like(saturation)
        for low, high in COLOUR_HUE_WINDOWS[colour]:
            colour_window |= (hue >= low) & (hue <= high)
        # A candidate pixel satisfies the colour window AND the saturation and
        # value floors together: the hue of a grey pixel is undefined, so hue
        # alone can never admit it.
        mask = (saturation >= MIN_SATURATION) & (value >= MIN_VALUE) & colour_window
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        aspect_low, aspect_high, fill_min = SHAPE_FILTERS.get(
            shape, (0.0, float("inf"), 0.0)
        )
        regions: list[tuple[int, tuple[float, float, float, float]]] = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area < MIN_AREA_PX:
                continue
            aspect = (width / height) if height > 0 else float("inf")
            if not (aspect_low <= aspect <= aspect_high):
                continue
            if float(area) / float(width * height) < fill_min:
                continue
            regions.append(
                (int(area), (float(x), float(y), float(x + width), float(y + height)))
            )
        regions.sort(key=lambda entry: (-entry[0], entry[1]))
        provenance = CandidateProvenance(
            model_id=self.model_id,
            checkpoint_hash=None,
            license_reference="in-repo deterministic routine; no checkpoint",
            offline=True,
            source=PROPOSER_SOURCE,
        )
        return tuple(
            Candidate(
                candidate_id=f"{PROPOSER_SOURCE}-{position}",
                region=region,
                score=None,
                provenance=provenance,
                label=f"{colour} {shape or 'region'}",
            )
            for position, (_area, region) in enumerate(regions[:MAX_CANDIDATES_PER_FRAME])
        )


# ---------------------------------------------------------------------------
# Single-clause mission interpretation
# ---------------------------------------------------------------------------

_FIND = re.compile(r"find\s+(?:the\s+)?(.+?)(?:,|\.|;|$)")
_REQUIREMENT_ACTIONS = ("find", "inspect", "return")


@dataclass(frozen=True)
class InterpretedMission:
    """What the instruction was interpreted to require, from the text alone."""

    target_phrase: str
    colour: str | None
    shape: str | None
    requirements: tuple[str, ...]


def interpret_instruction(instruction: str) -> InterpretedMission:
    """Interpret one single-clause search-inspect-return instruction.

    The interpretation reads the instruction's own words and nothing else: the
    target phrase is the object of "find", the requirements are the three
    clauses the first-indoor family is allowed to carry. An instruction whose
    find-clause is missing names no target, which the caller must refuse
    rather than invent.
    """
    text = (instruction or "").strip().lower().rstrip(".")
    match = _FIND.search(text)
    target_phrase = (match.group(1).strip() if match else "").rstrip(",;")
    colour, shape = _parse_query(target_phrase if target_phrase else text)
    actions = [action for action in _REQUIREMENT_ACTIONS if action in text]
    requirements = tuple(
        {
            "find": f"find the {target_phrase or 'declared target'}",
            "inspect": f"inspect the {target_phrase or 'declared target'}",
            "return": "return to the start position",
        }[action]
        for action in (actions or list(_REQUIREMENT_ACTIONS))
    )
    return InterpretedMission(
        target_phrase=target_phrase,
        colour=colour,
        shape=shape,
        requirements=requirements,
    )


def mission_contract(
    *,
    mission_id: str,
    instruction: str,
    budget: tuple[tuple[str, float], ...],
) -> MissionContract:
    """The public MissionContract: the instruction and its interpretation.

    No hidden fact enters here — the requirements come from the instruction
    text, and the budget from the suite's declared ceilings.
    """
    interpreted = interpret_instruction(instruction)
    if not interpreted.target_phrase:
        raise ValueError(
            f"the instruction {instruction!r} has no find-clause, so it names no target; "
            "refusing to interpret it as a search mission"
        )
    return MissionContract(
        mission_id=mission_id,
        instruction=instruction,
        interpreted_requirements=interpreted.requirements,
        revision=0,
        evidence_obligations=(
            "cite the observation in which each claimed target was seen",
        ),
        return_obligation="return to the start position and land",
        allowed_scope="the scene's indoor rooms, reachable through observed openings",
        budget=budget,
        unresolved_questions=(),
    )


# ---------------------------------------------------------------------------
# The B0 policy: bounded phases through the shared runner
# ---------------------------------------------------------------------------

# The exploration bound. A frontier excursion is one recipe step; four steps
# cover the vestibule, the two rooms and one re-observation, and the bound is
# a search bound, not knowledge of the layout (specification 20.2: B0 is
# tuned on development tasks with declared effort — this is that declaration).
EXPLORE_STEPS = 4
EXPLORE_ATTEMPTS = 2
INSPECT_ATTEMPTS = 2


def build_b0_recipes() -> tuple[MissionRecipe, ...]:
    """The B0 phases as bounded recipes for the shared runner.

    Phase 1 explores up to four frontiers; phase 2 inspects discovered
    candidates; phase 3 returns to the start. Each phase is its own recipe so
    an exhausted frontier set ends exploration without ending the mission:
    the executive, not the runner, decides that a finished search continues
    into inspection and return.
    """
    explore = MissionRecipe(
        steps=tuple(
            RecipeStep(
                action="explore",
                target=TargetSelector("frontier", "next_unvisited"),
                max_attempts=EXPLORE_ATTEMPTS,
                completion="settled at an observed frontier with fresh map evidence",
            )
            for _ in range(EXPLORE_STEPS)
        ),
        max_steps=EXPLORE_STEPS,
        resource_ceiling=float(EXPLORE_STEPS),
        source="B0-local",
    )
    inspect = MissionRecipe(
        steps=(
            RecipeStep(
                action="inspect",
                target=TargetSelector("candidate", "next_uninspected"),
                guard=Guard("candidate_present"),
                max_attempts=INSPECT_ATTEMPTS,
                completion="settled at the inspection standoff with the candidate observed",
            ),
        ),
        max_steps=1,
        resource_ceiling=1.0,
        source="B0-local",
    )
    returns = MissionRecipe(
        steps=(
            RecipeStep(
                action="return",
                target=TargetSelector("place", "start"),
                max_attempts=2,
                completion="settled at the start position",
            ),
        ),
        max_steps=1,
        resource_ceiling=1.0,
        source="B0-local",
    )
    return (explore, inspect, returns)


# ---------------------------------------------------------------------------
# The final report: claims are states with evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimEvidence:
    """One mission predicate's outcome and the observations that support it."""

    achieved: bool
    observation_ids: tuple[str, ...] = ()


def assemble_mission_claims(
    *,
    found: ClaimEvidence,
    inspected: ClaimEvidence,
    returned: ClaimEvidence,
    target_id: str,
    termination_reason: str,
    mission_revision: int,
    now: ClockStamp,
) -> FinalReport:
    """Assemble the mission's final report.

    Every claim states the predicate's outcome as a state name and cites the
    observations it was decided from; an unachieved predicate is reported as
    its negative state with its unmet requirement named, never omitted — a
    report with fewer claims is not a better report (specification 18.3).
    """
    claims = []
    for predicate, negative, evidence in (
        ("found", "not_found", found),
        ("inspected", "not_inspected", inspected),
        ("returned", "not_returned", returned),
    ):
        state = predicate if evidence.achieved else negative
        unmet: tuple[str, ...] = ()
        if not evidence.achieved:
            unmet = (f"requirement_{predicate}_unmet",)
        claims.append(
            ReportClaim(
                predicate=predicate,
                target=target_id,
                observed=state,
                support_refs=evidence.observation_ids,
                kind=ClaimKind.OBSERVATION if evidence.observation_ids else ClaimKind.INFERENCE,
                stamp=now,
                uncertainty=None,
                unmet_requirements=unmet or None,
            )
        )
    unmet_all = sorted(
        {requirement for claim in claims for requirement in (claim.unmet_requirements or ())}
    )
    return FinalReport(
        mission_revision=mission_revision,
        claims=tuple(claims),
        termination_reason=termination_reason,
        unmet_requirements=tuple(unmet_all),
        physical_return_status="returned" if returned.achieved else "not_returned",
        evidence_snapshot_ids=(),
    )
