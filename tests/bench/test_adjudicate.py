"""The adjudicator: support decided by an independent model, blinded, and pinned.

Three properties are pinned here, and each one has already failed somewhere in
this project's history:

* **The blinding is structural.** The brief handed to the model carries no arm,
  no trial-group id and no episode id, and a leak raises rather than shipping.
  Two missions were caught reading the main checkout instead of their worktree
  tonight; the same class of accident on the arm would silently unblind every
  review.
* **The schema is the grader's, not a parallel one.** Every annotation this
  module writes is fed back through ``grader.grade``, which refuses an
  annotation whose citation differs from the report's own.
* **An instrument failure is never a verdict.** An empty or malformed reply
  raises. Writing ``unsupported`` for a call that never answered would be a
  false negative shaped exactly like a finding.

No test here spends money: the transport's opener seam is injected.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from embodied.bench import adjudicate
from embodied.bench.grader import ADJUDICATION_FILENAME, grade
from embodied.bench.referee import truth_store_path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "hand-checkable-episode"

# Hand-written in the fixture itself, and the answer key this module is measured
# against: three claims its frames support, three they do not.
EXPECTED_VERDICTS = {
    0: "supported",
    1: "supported",
    2: "supported",
    3: "unsupported",
    4: "unsupported",
    5: "unsupported",
}


def _episode(tmp_path: Path, *, arm: str | None = None) -> Path:
    """A graded copy of the fixture, optionally carrying an arm on its manifest."""
    episode = tmp_path / "episode"
    shutil.copytree(FIXTURE, episode)
    shutil.copytree(truth_store_path(FIXTURE), truth_store_path(episode))
    if arm is not None:
        manifest_path = episode / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["arm"] = arm
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return episode


def _reply(verdict: str, note: str = "the cited frame shows what the claim asserts") -> dict:
    """One OpenAI-shaped reply, as the transport's opener returns it."""
    return {
        "model": adjudicate.ADJUDICATOR_MODEL_SECTION["id"],
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"verdict": verdict, "note": note}),
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def _opener_for(verdict_for):
    """An opener that answers each call according to the prompt it receives.

    ``verdict_for`` takes the prompt text and returns a verdict, so a test can
    drive a different answer per claim without depending on call order.
    """
    calls: list[str] = []

    def opener(request):
        document = json.loads(request.data.decode("utf-8"))
        prompt = document["messages"][0]["content"][0]["text"]
        calls.append(prompt)
        return _reply(verdict_for(prompt))

    opener.calls = calls  # type: ignore[attr-defined]
    return opener


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------


def test_the_brief_carries_no_arm_and_no_episode_identifier(tmp_path):
    episode = _episode(tmp_path, arm="B2")
    brief = adjudicate.blinded_brief(episode)
    text = json.dumps(brief.as_document(), sort_keys=True)
    assert "B2" not in text, "the arm reached the brief"
    assert "hand-checkable-001" not in text, "the episode id reached the brief"
    assert brief.blinded_identifiers == ("B2", "hand-checkable-001")


def test_a_leak_raises_rather_than_shipping_an_unblinded_review(tmp_path, monkeypatch):
    episode = _episode(tmp_path, arm="B2")
    monkeypatch.setattr(
        adjudicate,
        "_instruction_of",
        lambda surface: "fly the B2 arm through the doorway",
    )
    with pytest.raises(adjudicate.BlindingError) as raised:
        adjudicate.blinded_brief(episode)
    assert "B2" in str(raised.value)


def test_the_adjudicator_cannot_reach_the_bench_side_store(tmp_path):
    """The truth store is outside the projection, so the reader refuses it."""
    from embodied.bench.recorder import AgentSurface, SurfaceViolation

    episode = _episode(tmp_path)
    surface = AgentSurface.open(episode)
    with pytest.raises(SurfaceViolation):
        surface.read_member("truth-events.jsonl")


# ---------------------------------------------------------------------------
# The annotation the grader accepts
# ---------------------------------------------------------------------------


def test_the_annotation_conforms_and_citations_are_verbatim(tmp_path):
    episode = _episode(tmp_path)
    out = tmp_path / "annotation.json"
    run = adjudicate.adjudicate_episode(
        episode, output_path=out, opener=_opener_for(lambda prompt: "supported")
    )
    document = json.loads(out.read_text(encoding="utf-8"))
    assert set(document) == {
        "adjudication_version",
        "rubric_revision",
        "reviewer",
        "entries",
    }
    report = json.loads((episode / "final-report.json").read_text(encoding="utf-8"))
    for entry in document["entries"]:
        assert set(entry) == {"claim_index", "cited_evidence", "verdict", "note"}
        assert entry["cited_evidence"] == report["claims"][entry["claim_index"]]["support_refs"]
    assert document["rubric_revision"] == adjudicate.RUBRIC_REVISION
    assert run.resolved_identities == (adjudicate.ADJUDICATOR_MODEL_SECTION["id"],) * len(
        document["entries"]
    )


def test_the_score_consumes_the_annotation_and_support_is_no_longer_pending(tmp_path):
    """The defect this module exists to fix, demonstrated end to end.

    With the fixture's committed annotation removed — which is the state every
    live episode is in, because nothing produced one — every claim is pending
    and the primary endpoint cannot be reached. Producing an annotation through
    this module is what changes that.
    """
    episode = _episode(tmp_path)
    (episode / ADJUDICATION_FILENAME).unlink()
    before = grade(episode)
    assert before.pending, "with no annotation every claim is pending, which is the defect"
    assert not before.fully_passed
    out = tmp_path / "annotation.json"
    adjudicate.adjudicate_episode(
        episode,
        output_path=out,
        opener=_opener_for(
            lambda prompt: EXPECTED_VERDICTS[_claim_index_of(prompt)]
        ),
    )
    after = grade(episode, out)
    assert not after.pending
    assert [verdict.support for verdict in after.claims] == [
        EXPECTED_VERDICTS[index] for index in range(len(after.claims))
    ]


def test_a_supported_verdict_lets_an_honest_claim_pass(tmp_path):
    """The endpoint must be *reachable*: an honest, supported claim passes."""
    episode = _episode(tmp_path)
    out = tmp_path / "annotation.json"
    adjudicate.adjudicate_episode(
        episode, output_path=out, opener=_opener_for(lambda prompt: "supported")
    )
    score = grade(episode, out)
    # Claims 0-2 are world-correct in the fixture, so support plus correctness
    # is a pass. This is what could not happen before this module existed.
    assert [verdict.passed for verdict in score.claims[:3]] == [True, True, True]


def test_an_unsupported_verdict_fails_a_world_correct_claim(tmp_path):
    """The endpoint must be *refusable*: a true claim the evidence misses fails."""
    episode = _episode(tmp_path)
    out = tmp_path / "annotation.json"
    adjudicate.adjudicate_episode(
        episode, output_path=out, opener=_opener_for(lambda prompt: "unsupported")
    )
    score = grade(episode, out)
    assert score.claims[0].world_correct is True
    assert score.claims[0].support == "unsupported"
    assert score.claims[0].passed is False
    assert not score.fully_passed


# ---------------------------------------------------------------------------
# An instrument failure is not a verdict
# ---------------------------------------------------------------------------


def test_a_truncated_reply_raises_and_writes_nothing(tmp_path):
    """A reasoning model that exhausts its budget returns empty content."""
    episode = _episode(tmp_path)
    out = tmp_path / "annotation.json"

    def opener(request):
        return {
            "model": adjudicate.ADJUDICATOR_MODEL_SECTION["id"],
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "", "reasoning_content": "weighing"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8000},
        }

    with pytest.raises(adjudicate.AdjudicatorError) as raised:
        adjudicate.adjudicate_episode(episode, output_path=out, opener=opener)
    assert "empty reply is an instrument failure" in str(raised.value)
    assert not out.exists(), "a failed adjudication must not leave an annotation"


def test_a_transport_error_raises_rather_than_becoming_unsupported(tmp_path):
    from embodied.pilot.provider import TransportError
    import urllib.error

    episode = _episode(tmp_path)
    out = tmp_path / "annotation.json"

    def opener(request):
        raise urllib.error.URLError("no route to the adjudicator")

    with pytest.raises(adjudicate.AdjudicatorError) as raised:
        adjudicate.adjudicate_episode(episode, output_path=out, opener=opener)
    assert "instrument failure" in str(raised.value)
    assert not out.exists()


def test_a_verdict_outside_the_grader_vocabulary_is_refused(tmp_path):
    episode = _episode(tmp_path)
    out = tmp_path / "annotation.json"
    with pytest.raises(adjudicate.AdjudicatorError) as raised:
        adjudicate.adjudicate_episode(
            episode, output_path=out, opener=_opener_for(lambda prompt: "probably")
        )
    assert "verdict" in str(raised.value)


def test_a_fenced_reply_is_still_read(tmp_path):
    episode = _episode(tmp_path)
    out = tmp_path / "annotation.json"

    def opener(request):
        return {
            "model": adjudicate.ADJUDICATOR_MODEL_SECTION["id"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '```json\n{"verdict": "supported", "note": "the frame shows it"}\n```',
                    },
                }
            ],
            "usage": {},
        }

    adjudicate.adjudicate_episode(episode, output_path=out, opener=opener)
    document = json.loads(out.read_text(encoding="utf-8"))
    assert all(entry["verdict"] == "supported" for entry in document["entries"])


# ---------------------------------------------------------------------------
# What the model is actually asked
# ---------------------------------------------------------------------------


def _claim_index_of(prompt: str) -> int:
    for line in prompt.splitlines():
        if line.startswith("claim "):
            return int(line.split()[1].rstrip(":"))
    raise AssertionError(f"no claim line in the prompt: {prompt[:200]}")


def test_every_claim_is_asked_once_and_carries_its_own_evidence(tmp_path):
    episode = _episode(tmp_path)
    opener = _opener_for(lambda prompt: "supported")
    adjudicate.adjudicate_episode(episode, output_path=tmp_path / "a.json", opener=opener)
    prompts = opener.calls
    assert len(prompts) == len(EXPECTED_VERDICTS)
    assert sorted(_claim_index_of(prompt) for prompt in prompts) == list(EXPECTED_VERDICTS)
    for prompt in prompts:
        assert "the arm" not in prompt
        assert prompt.startswith(adjudicate.RUBRIC[:60])


def test_executions_travel_with_the_observations_they_cite(tmp_path):
    """A return claim is carried by what was executed, not by a frame alone."""
    episode = _episode(tmp_path)
    brief = adjudicate.blinded_brief(episode)
    digest = "\n".join(brief.record_lines)
    assert "executions in recorded order" in digest
    assert "citing observation 2; observation 4" in digest
    assert "inspection_close_up_and_return_frame" in digest


def _arm_bearing_citation(episode: Path, arm: str = "B2") -> tuple[str, ...]:
    """Rewrite an episode so a claim's citation embeds the arm.

    This is not a contrived case: it is what every live episode looks like. The
    transport builds observation ids as `{suite}-{arm}-{run}-obs-NNNNN`, and a
    claim cites those ids, so the arm travels inside the citation by
    construction. The first run of the harness against a real episode failed
    with exactly this, which is why the field exists.
    """
    borrowed = f"first-indoor-{arm}-x-{arm}-obs-00002"
    report_path = episode / "final-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["claims"][0]["support_refs"] = [borrowed]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    events_path = episode / "agent-events.jsonl"
    lines = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("kind") == "observation" and event["payload"].get("record_id") == "obs-2":
            event["payload"]["record_id"] = borrowed
        if event.get("kind") == "report":
            # The report file and its stream event are cross-checked against
            # each other, so a fixture that edits the citation edits both.
            event["payload"]["claims"][0]["support_refs"] = [borrowed]
        lines.append(json.dumps(event, sort_keys=True))
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # An episode records the hash of every agent-side member, and it refuses to
    # be read once one disagrees — which is that check doing its job. A fixture
    # that edits an episode therefore re-hashes what it edited, exactly as
    # tests/bench/test_recorder_replay.py does for the same reason.
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for member in ("final-report.json", "agent-events.jsonl"):
        digest = hashlib.sha256((episode / member).read_bytes()).hexdigest()
        manifest["artifacts"][member] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return (borrowed,)


def test_a_citation_that_embeds_the_arm_is_neutralised_for_the_model(tmp_path):
    """The citation leaks by construction; the model-visible view must not."""
    episode = _episode(tmp_path, arm="B2")
    (borrowed,) = _arm_bearing_citation(episode)
    brief = adjudicate.blinded_brief(episode)
    visible = json.dumps(brief.as_document(), sort_keys=True)
    assert "B2" not in visible, "the arm reached the model-visible brief through a citation"
    assert borrowed not in visible, "the citation itself reached the model-visible brief"
    assert brief.claims[0].cited_labels == ("observation 2",)
    # The annotation still echoes the report's own reference, because the
    # grader refuses an annotation whose citation differs from it.
    assert brief.claims[0].cited_verbatim == (borrowed,)


def test_the_annotation_keeps_the_verbatim_citation_the_grader_requires(tmp_path):
    episode = _episode(tmp_path)
    (borrowed,) = _arm_bearing_citation(episode)
    out = tmp_path / "annotation.json"
    adjudicate.adjudicate_episode(
        episode, output_path=out, opener=_opener_for(lambda prompt: "unsupported")
    )
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["entries"][0]["cited_evidence"] == [borrowed]
    # And the grader accepts it rather than raising a citation mismatch.
    score = grade(episode, out)
    assert score.claims[0].support == "unsupported"
