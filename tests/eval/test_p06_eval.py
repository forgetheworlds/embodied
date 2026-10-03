"""The P06 comparison runner's own instrument tests.

What these cover, and why each matters to a comparison rather than to the code:

* the invalid-versus-outcome classification, because section 20.5 says policy
  crashes and refusals are outcomes while instrument failures are not, and
  pooling them would attribute a host's problem to an executive;
* the freeze refusal, because a sample count chosen after seeing a score is the
  thing section 20.5 exists to prevent, and the refusal is the only mechanism
  enforcing it;
* loud refusal when a scene the protocol names is not on disk, because an
  episode about a scene that has moved is an episode about nothing;
* the pairing, because section 20.2 pairs scenario conditions and a duplicated
  trial group would silently correlate two runs that were meant to be matched;
* the aggregation's deliberate reporting of unmeasured cost rather than zero.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY / "scripts" / "p06_eval.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p06_eval", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # dataclasses resolves `cls.__module__` through sys.modules, so the module
    # has to be registered before exec_module or its Cell dataclass fails to
    # build — a load-from-path detail, not a fact about the runner.
    sys.modules["p06_eval"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _receipt(status: str, reasons=("recorded",)):
    return {"status": status, "gate_status": "pass", "reasons": list(reasons)}


def _run(*argv):
    return subprocess.run(
        [sys.executable, str(RUNNER), *argv],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
    )


# ---------------------------------------------------------------------------
# The distinction the comparison turns on


def test_a_refused_transport_run_is_invalid_not_an_outcome(runner):
    """A host or port gate tells us nothing about the executive."""
    record = {"receipt": _receipt("blocked", ("port 9002 is already in use",))}
    assert runner.classify(record) == "invalid"


def test_a_completed_and_scored_run_is_an_outcome(runner):
    record = {"receipt": _receipt("complete"), "score": {"status": "complete"}}
    assert runner.classify(record) == "outcome"


def test_a_pending_score_is_still_an_outcome(runner):
    """Missing adjudication is pending, not a pass and not an instrument fault."""
    record = {"receipt": _receipt("complete"), "score": {"status": "pending"}}
    assert runner.classify(record) == "outcome"


def test_a_mission_with_no_score_is_unclassified_rather_than_invalid(runner):
    """The mission ran, so it is not an instrument failure — but it is not
    evidence either until something scored it. Guessing would be worse."""
    record = {"receipt": _receipt("complete"), "score": None}
    assert runner.classify(record) == "unclassified"


def test_no_receipt_at_all_is_invalid(runner):
    assert runner.classify({"receipt": None, "score": None}) == "invalid"


# ---------------------------------------------------------------------------
# The freeze


def test_open_questions_lists_every_unsettled_value(runner):
    protocol = {
        "matrix": {"episodes_per_cell": None, "stop_rule": None},
        "owner_questions": {
            "episodes_per_cell": {"value": None},
            "effect_size_or_interval_width": {"value": None},
        },
    }
    outstanding = runner.open_questions(protocol)
    assert "matrix.episodes_per_cell" in outstanding
    assert "owner_questions.effect_size_or_interval_width" in outstanding


def test_open_questions_is_empty_once_every_value_is_set(runner):
    protocol = {
        "matrix": {"episodes_per_cell": 8, "stop_rule": "after 8 per cell"},
        "owner_questions": {
            "episodes_per_cell": {"value": 8},
            "stop_rule": {"value": "after 8 per cell"},
            "effect_size_or_interval_width": {"value": "0.2"},
        },
    }
    assert runner.open_questions(protocol) == []


def _minimal_protocol(tmp_path, *, episodes, stop_rule):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        json.dumps(
            {
                "protocol_revision": "test-1",
                "arms": [{"id": "B0"}],
                "scenes": [
                    {
                        "suite": "holdout-e-wide",
                        "scene_root": "scenarios/missions/holdout/holdout-e-wide",
                        "truth_seed": "scenarios/missions/holdout/holdout-e-wide/truth.yaml",
                        "target": "red_can",
                        "target_present": True,
                    }
                ],
                "pairing": {},
                "endpoints": {},
                "matrix": {"episodes_per_cell": episodes, "stop_rule": stop_rule},
                "owner_questions": {"episodes_per_cell": {"value": episodes}},
            }
        ),
        encoding="utf-8",
    )
    return protocol


def test_the_scored_plan_refuses_while_the_freeze_is_incomplete(tmp_path):
    """The refusal is the enforcement. Without it, a protocol with a null
    sample count would quietly run at whatever count a caller passed."""
    protocol = _minimal_protocol(tmp_path, episodes=None, stop_rule=None)
    completed = _run("plan", "--protocol", str(protocol), "--out", str(tmp_path / "out"))
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "freeze is not complete" in completed.stderr
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_a_dry_run_manifest_can_never_be_executed_as_scored(tmp_path):
    """A dry run's numbers say nothing about an executive, so a manifest that
    is one must not be run in live mode even if a caller asks."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"mode": "dry-run", "open_questions": [], "cells": []}), encoding="utf-8"
    )
    completed = _run(
        "execute", "--manifest", str(manifest), "--out", str(tmp_path / "out"), "--mode", "live"
    )
    assert completed.returncode == 2
    assert "dry run" in completed.stderr


def test_a_scored_manifest_with_open_questions_cannot_spend(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"mode": "scored", "open_questions": ["matrix.episodes_per_cell"], "cells": []}),
        encoding="utf-8",
    )
    completed = _run(
        "execute", "--manifest", str(manifest), "--out", str(tmp_path / "out"), "--mode", "live"
    )
    assert completed.returncode == 2
    assert "freeze is incomplete" in completed.stderr


# ---------------------------------------------------------------------------
# Missing prerequisites and the pairing


def test_a_scene_that_is_not_on_disk_is_refused_loudly(runner):
    protocol = {
        "scenes": [
            {
                "suite": "no-such-suite",
                "scene_root": "scenarios/missions/holdout/absent",
                "truth_seed": "scenarios/missions/holdout/absent/truth.yaml",
            }
        ]
    }
    problems = runner.require_scene_prerequisites(protocol)
    assert problems, "a missing suite and a missing scene must both be reported"
    assert any("no suite declaration" in problem for problem in problems)


def test_every_cell_has_a_unique_episode_id_and_a_shared_trial_group(runner):
    """Section 20.2: a unique episode_id per run, a shared trial_group_id for
    matched conditions. A collided trial group would pair two runs that were
    never matched; a collided episode id would overwrite one."""
    def scene(name: str) -> dict:
        return {
            "suite": name,
            "scene_root": f"scenarios/missions/holdout/{name}",
            "truth_seed": f"scenarios/missions/holdout/{name}/truth.yaml",
            "target": "red_can",
            "target_present": True,
        }

    protocol = {
        "arms": [{"id": "B0"}, {"id": "B1"}],
        "scenes": [scene("holdout-e-wide"), scene("holdout-f-z")],
    }
    cells = runner.build_cells(protocol, 3)
    assert len(cells) == 2 * 2 * 3
    assert len({cell.episode_id for cell in cells}) == len(cells)
    # One trial group per (scene, index) — the same scene and index across two
    # arms is one matched condition, which is the whole point of the pairing.
    groups: dict[str, set[str]] = {}
    for cell in cells:
        groups.setdefault(cell.trial_group, set()).add(cell.arm)
    assert all(arms == {"B0", "B1"} for arms in groups.values()), groups
    assert len(groups) == 2 * 3


# ---------------------------------------------------------------------------
# The aggregation's published numbers


def test_the_aggregation_separates_invalid_from_outcome(runner, tmp_path):
    """Synthetic inputs where both occur, because the distinction is the one
    thing that must not blur: an invalid run excluded from the comparison, an
    outcome carrying it."""
    records = [
        {
            "episode_id": "e1", "trial_group": "g1", "arm": "B0", "suite": "s",
            "episode_index": 1,
            "receipt": _receipt("blocked", ("port 9002 is already in use",)),
            "score": None, "mission": None,
        },
        {
            "episode_id": "e2", "trial_group": "g2", "arm": "B0", "suite": "s",
            "episode_index": 2,
            "receipt": _receipt("complete"),
            "score": {
                "status": "complete",
                "mission": {"safe_task_completion": True, "task_completion": True},
                "missed_present_targets": [],
                "evidence_support": {"pending": 0},
            },
            "mission": {"cloud_calls": []},
        },
        {
            "episode_id": "e3", "trial_group": "g3", "arm": "B0", "suite": "s",
            "episode_index": 3,
            "receipt": _receipt("complete"),
            "score": {
                "status": "pending",
                "mission": {
                    "safe_task_completion": False,
                    "task_completion": False,
                    "takeover": False,
                },
                "missed_present_targets": ["red_can"],
                "evidence_support": {"pending": 3},
            },
            "mission": {"cloud_calls": []},
        },
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    assert runner.main(["aggregate", "--records", str(path), "--out", str(tmp_path)]) == 0

    aggregate = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    cell = aggregate["per_cell"][0]
    assert cell["episodes"] == 3
    assert cell["invalid"] == 1
    assert cell["outcomes"] == 2
    assert cell["safe_task_completions"] == 1
    assert cell["missed_required_targets"] == 1
    assert cell["invalid_reasons"]  # the reason travels with the exclusion
    # Cost and latency are not recorded per call by the runtime yet: unmeasured,
    # never zero, because zero would read as free.
    assert cell["model_calls_cost"] == "unmeasured"


def test_the_report_names_every_cell_before_it_has_any(runner):
    """The skeleton exists so that filling it in is mechanical, and so a cell
    with no episodes is visibly empty rather than quietly absent."""
    empty = {"per_cell": [], "not_yet_measurable": ["latency — not recorded"]}
    report = runner.render_report(empty)
    assert "arm | scene | episodes" in report
    assert "latency — not recorded" in report


def test_the_report_says_when_the_primary_endpoint_cannot_be_read(runner):
    """A column of zeroes means opposite things depending on whether support was
    adjudicated, so the report must say which rather than leave a reader to
    infer it — and the inference a reader would make is the wrong one."""
    unreachable = {
        "per_cell": [],
        "not_yet_measurable": [],
        "primary_endpoint": {
            "name": "safe_task_completion",
            "reachable_from_these_records": False,
            "outcomes": 24,
            "outcomes_with_pending_support": 24,
            "outcomes_adjudicated": 0,
            "reason": "a per-claim support adjudication is required",
        },
    }
    report = runner.render_report(unreachable)
    assert "the primary endpoint is not readable" in report
    assert "not a" in report and "result about the arms" in report
    assert "24 of 24" in report
    assert "per-claim support adjudication is required" in report

    reachable = {
        "per_cell": [],
        "not_yet_measurable": [],
        "primary_endpoint": {
            "name": "safe_task_completion",
            "reachable_from_these_records": True,
            "outcomes": 24,
            "outcomes_with_pending_support": 0,
            "outcomes_adjudicated": 24,
            "reason": "at least one outcome carries adjudicated support",
        },
    }
    healthy = runner.render_report(reachable)
    assert "are meaningful" in healthy
    assert "24 of 24" in healthy
