#!/usr/bin/env python3
"""P06 — the frozen held-out comparison runner.

WHAT THIS IS. The instrument that runs the project's own question: whether a
continuously involved cloud pilot beats a competent conventional executive on
unfamiliar tasks, judged by an independent evaluator whose hidden truth the
pilot never sees. The protocol it executes is ``work/runs/p06/protocol.yaml``,
frozen before any scored run; this file is the mechanism, and it deliberately
contains no criteria of its own.

IT DOES NOT RE-IMPLEMENT THE CHAIN. Recording is the CLI's own
``bench record`` (or, for a dry run, the same ``live_record.record`` the CLI
calls, with a scripted mission driver substituted for the simulator).
Scoring is the CLI's own ``bench score``. If either refuses, this runner
reports the refusal rather than substituting anything: a suite, a scene or a
budget cap that is missing is a blocker to be read, not a gap to fill.

THE FREEZE IS ENFORCED, NOT DOCUMENTED. ``plan`` refuses to build a scored
manifest while any ``owner_questions`` entry in the protocol is null, and
``execute --mode live`` refuses to spend money while the manifest is not a
scored one. Three numbers in that protocol cannot be chosen defensibly yet —
the episode count, the stop rule and the effect target, all of which section
20.5 derives from development pilot variability that does not exist while no
mission moves under command. Inventing them is the error the freeze exists to
prevent, so this runner stops instead.

EXIT CODES: 0 success; 2 refused (a missing prerequisite, or the freeze not
satisfied). A non-zero exit means nothing was scored.

USAGE

    python3 scripts/p06_eval.py plan      --protocol work/runs/p06/protocol.yaml \
                                          --out work/runs/p06 --dry-run --episodes 2
    python3 scripts/p06_eval.py execute   --manifest work/runs/p06/manifest.json \
                                          --out work/runs/p06 --mode scripted
    python3 scripts/p06_eval.py aggregate --records work/runs/p06/records.jsonl \
                                          --out work/runs/p06
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

# Protocol keys whose null value means the comparison cannot be scored yet.
# Kept here rather than inferred, so the runner's refusal logic is readable in
# one place and a new question is added deliberately.
# An owner question may be a scalar null or a mapping with a null `value`.
QUESTION_KEYS = ("episodes_per_cell", "stop_rule", "effect_size_or_interval_width")


# ---------------------------------------------------------------------------
# The protocol


def load_protocol(path: Path) -> dict:
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"protocol {path} is not a mapping")
    for key in ("protocol_revision", "arms", "scenes", "pairing", "endpoints"):
        if key not in document:
            raise SystemExit(f"protocol {path} has no {key!r}")
    return document


def protocol_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_questions(protocol: dict) -> list[str]:
    """Protocol questions whose value is still null.

    ``matrix.episodes_per_cell`` and ``matrix.stop_rule`` are the operational
    copies; ``owner_questions`` is where the reasoning lives. Both are checked,
    because a protocol that filled one and not the other is not frozen.
    """
    outstanding: list[str] = []
    matrix = protocol.get("matrix") or {}
    for key in ("episodes_per_cell", "stop_rule"):
        if matrix.get(key) is None:
            outstanding.append(f"matrix.{key}")
    questions = protocol.get("owner_questions") or {}
    for key in QUESTION_KEYS:
        entry = questions.get(key)
        if entry is None:
            continue
        value = entry.get("value") if isinstance(entry, dict) else entry
        if value is None:
            outstanding.append(f"owner_questions.{key}")
    return outstanding


# ---------------------------------------------------------------------------
# The matrix


@dataclass(frozen=True)
class Cell:
    """One episode: an arm on a scene, at a declared index.

    The pairing is by ``trial_group`` — one per (scene, episode index) — because
    section 20.2 pairs scenario conditions rather than camera images: policies
    choose different flight paths and therefore see different evidence, so the
    shared identity is the scene and the index, never the frames.
    """

    arm: str
    suite: str
    scene_root: str
    truth_seed: str
    target: str
    target_present: bool
    episode_index: int
    trial_group: str
    episode_id: str

    @property
    def label(self) -> str:
        return f"{self.arm}/{self.suite}/{self.episode_index}"


def build_cells(protocol: dict, episodes: int) -> list[Cell]:
    cells: list[Cell] = []
    for arm_entry in protocol["arms"]:
        arm = arm_entry["id"]
        for scene in protocol["scenes"]:
            for index in range(1, episodes + 1):
                cells.append(
                    Cell(
                        arm=arm,
                        suite=scene["suite"],
                        scene_root=scene["scene_root"],
                        truth_seed=scene["truth_seed"],
                        target=scene["target"],
                        target_present=bool(scene["target_present"]),
                        episode_index=index,
                        trial_group=f"{scene['suite']}:{index}",
                        episode_id=f"{scene['suite']}-{arm}-{index}",
                    )
                )
    return cells


def require_scene_prerequisites(
    protocol: dict, repository: Path = REPOSITORY
) -> list[str]:
    """Refuse loudly when a scene the protocol names is not actually there.

    Every pointer is checked on disk, because a protocol that names a scene
    which has since moved would otherwise produce episodes about nothing.
    """
    problems: list[str] = []
    for scene in protocol["scenes"]:
        suite_path = repository / "configs" / "suites" / f"{scene['suite']}.yaml"
        if not suite_path.exists():
            problems.append(
                f"scene {scene['suite']}: no suite declaration at {suite_path}"
            )
        for key in ("scene_root", "truth_seed"):
            path = repository / scene[key]
            if not path.exists():
                problems.append(
                    f"scene {scene['suite']}: {key} {scene[key]} does not exist"
                )
    return problems


# ---------------------------------------------------------------------------
# plan


def cmd_plan(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    outstanding = open_questions(protocol)
    problems = require_scene_prerequisites(protocol)

    if problems:
        print(
            "refused: the protocol names scenes that are not on disk", file=sys.stderr
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    if outstanding and not args.dry_run:
        print(
            "refused: the protocol's freeze is not complete, so a scored manifest "
            "cannot be built. Filling one of these after seeing a score is the "
            "thing the freeze exists to prevent (specification 20.5).",
            file=sys.stderr,
        )
        for key in outstanding:
            print(f"  - {key} is null", file=sys.stderr)
        print(
            "  A DRY RUN is available and is labelled as one: "
            "add --dry-run --episodes N.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        if not args.episodes or args.episodes < 1:
            print("refused: --dry-run needs --episodes N (N >= 1)", file=sys.stderr)
            return 2
        episodes = args.episodes
        mode = "dry-run"
    else:
        episodes = int(protocol["matrix"]["episodes_per_cell"])
        mode = "scored"

    cells = build_cells(protocol, episodes)
    manifest = {
        "manifest_revision": "p06-manifest-1",
        "mode": mode,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_path": str(protocol_path),
        "protocol_revision": protocol["protocol_revision"],
        "protocol_sha256": protocol_digest(protocol_path),
        "protocol_frozen_at_utc": protocol.get("frozen_at_utc"),
        "code_revision": protocol.get("frozen_code_revision"),
        "sensor_mode": protocol["sensor_mode"],
        "arms": [entry["id"] for entry in protocol["arms"]],
        "deferred_arms": [entry["id"] for entry in protocol.get("deferred_arms", [])],
        "episodes_per_cell": episodes,
        "open_questions": outstanding,
        "cell_count": len(cells),
        "cells": [asdict(cell) | {"label": cell.label} for cell in cells],
        "note": (
            "A dry-run manifest is not evidence about any executive: its episodes "
            "are driven by a scripted stand-in rather than a mission."
            if mode == "dry-run"
            else "Scored manifest. Any change to the protocol after the first "
            "recorded episode voids this comparison."
        ),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"mode: {mode}")
    print(
        f"cells: {len(cells)} ({len(protocol['arms'])} arms x {len(protocol['scenes'])} scenes x {episodes})"
    )
    print(f"manifest: {manifest_path}")
    if outstanding:
        print(f"open questions carried in the manifest: {', '.join(outstanding)}")
    return 0


# ---------------------------------------------------------------------------
# execute


def _run_cli(argv: list[str]) -> int:
    """Run the project's own CLI in this checkout and return its exit code."""
    completed = subprocess.run(
        [sys.executable, "-m", "embodied", *argv],
        cwd=REPOSITORY,
        env={"PYTHONPATH": str(REPOSITORY / "src"), **__import__("os").environ},
        capture_output=True,
        text=True,
    )
    return completed.returncode


def record_live(cell: Cell, output: Path) -> tuple[int, str]:
    """One live episode through the CLI, which is what a scored run uses."""
    code = _run_cli(
        [
            "bench",
            "record",
            "--suite",
            cell.suite,
            "--sensor-mode",
            "sensor-derived",
            "--arm",
            cell.arm,
            "--output",
            str(output),
        ]
    )
    return code, _read_json(output / "receipt.json")


def record_scripted(
    cell: Cell, output: Path, *, overclaim: bool = False
) -> tuple[int, dict | None]:
    """One episode through the same transport, with a scripted mission driver.

    This is the dry-run path: identical assembly, identical grading, no
    simulator and no spend. It calls ``live_record.record`` — the function the
    CLI itself calls — so a dry run exercises the real chain rather than a
    parallel implementation of it.

    The one thing it must add is the receipt. ``record()`` returns the
    transport's own outcome; it is the *CLI wrapper* that writes receipt.json
    beside the episode. Calling the transport directly therefore carries the
    same fields here, copied from that outcome and from nothing else — without
    which every dry-run episode looks like an instrument failure with no
    receipt, which is exactly what the first run of this runner did.
    """
    sys.path.insert(0, str(REPOSITORY / "src"))
    from embodied.bench import live_record
    from embodied.cli import RECEIPT_VERSION
    from embodied.contracts.records import SensorMode

    suite_path = REPOSITORY / "configs" / "suites" / f"{cell.suite}.yaml"
    document = live_record.load_suite_document(suite_path)
    outcome = live_record.record(
        suite_name=cell.suite,
        suite_document=document,
        arm=cell.arm,
        sensor_mode=SensorMode.SENSOR_DERIVED,
        output=output,
        root=REPOSITORY,
        platform_config=REPOSITORY / "configs" / "first_indoor.yaml",
        truth_seed=REPOSITORY / cell.truth_seed,
        suite_config=suite_path,
        mission_driver=scripted_driver_for(
            target_present=cell.target_present,
            target_ned=_target_position_from_seed(
                REPOSITORY / cell.truth_seed, cell.target
            ),
            overclaim=overclaim,
        ),
    )
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "stage_id": "P02",
        "status": outcome.status.value,
        "gate_status": outcome.gate_status.value,
        "episode_id": outcome.episode_id,
        "trial_group_id": outcome.trial_group_id,
        "sensor_mode": outcome.sensor_mode.value if outcome.sensor_mode else None,
        "reasons": list(outcome.reasons),
        "limitations": list(outcome.limitations),
        "manifest": dict(outcome.manifest),
        "receipt_source": (
            "carried by scripts/p06_eval.py from live_record.record's own returned "
            "outcome; the CLI wrapper writes this file for a live run"
        ),
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return int(outcome.exit_code()), receipt


def score_episode(cell: Cell, output: Path) -> tuple[int, str]:
    """Score through the CLI's own score path. Exit 3 is pending, not failure."""
    code = _run_cli(
        [
            "bench",
            "score",
            "--episode",
            str(output / "episode"),
            "--output",
            str(output / "score"),
        ]
    )
    return code, ""


def adjudicate_episode(cell: Cell, output: Path) -> tuple[int, str]:
    """Have the independent adjudicator write this episode's support annotation.

    Through the CLI's own path, so the thing the runner exercises is the thing
    an operator runs. Exit 2 is a refused adjudication — an instrument failure,
    which is recorded as one rather than being read as a verdict.
    """
    adjudicate_dir = output / "adjudication"
    code = _run_cli(
        [
            "bench",
            "adjudicate",
            "--episode",
            str(output / "episode"),
            "--output",
            str(adjudicate_dir),
        ]
    )
    return code, ""


def _cell_is_complete(run_dir: Path) -> bool:
    """A cell is complete only when its record step succeeded AND its score
    step produced a file.

    The runner writes receipt.json (:340) — never record.json — so that is the
    file a resumed execute must test. But a receipt alone is not enough: the
    receipt is written for every recorded outcome, including ``blocked`` and
    ``invalid`` ones, and the score step only runs when the record exited 0 and
    writes episode/score.json on success. Requiring both means a resumed run
    skips only cells that genuinely finished, and re-records a cell whose
    receipt says blocked or whose scoring failed, rather than quietly treating
    a half-run cell as evidence.
    """
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    if receipt.get("status") != "complete":
        return False
    return (run_dir / "episode" / "score.json").is_file()


def cmd_execute(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "live" and manifest["mode"] != "scored":
        print(
            "refused: this manifest is a dry run, and a dry run may never be "
            "reported as scored or pooled with one.",
            file=sys.stderr,
        )
        return 2
    if args.mode == "live" and manifest.get("open_questions"):
        print(
            "refused: the freeze is incomplete, so a paid run would spend against "
            "an unsettled protocol.",
            file=sys.stderr,
        )
        for key in manifest["open_questions"]:
            print(f"  - {key} is null", file=sys.stderr)
        return 2

    if args.mode == "live" and args.overclaim:
        print(
            "refused: --overclaim makes the stand-in report a find it never made. "
            "It exists to exercise the grader's negative path in a dry run and has "
            "no place in a scored one.",
            file=sys.stderr,
        )
        return 2

    records_path = out / "records.jsonl"
    written = 0
    with records_path.open("a", encoding="utf-8") as sink:
        for raw in manifest["cells"]:
            cell = Cell(**{k: v for k, v in raw.items() if k != "label"})
            # The transport builds its own episode_id as
            # `{suite}-{arm}-{output.name}` (live_record.py:973), so the run
            # directory is named for the index alone. Naming it for the full
            # episode id doubles that prefix in every recorded episode — which is
            # what the first dry run produced.
            run_dir = out / "runs" / cell.arm / cell.suite / str(cell.episode_index)
            if _cell_is_complete(run_dir):
                continue  # already executed; a re-run is a deliberate act
            run_dir.mkdir(parents=True, exist_ok=True)

            if args.mode == "scripted":
                code, receipt = record_scripted(cell, run_dir, overclaim=args.overclaim)
            else:
                code, receipt = record_live(cell, run_dir)

            # The annotation is produced before the score, because the score
            # reads it: with no annotation every claim is pending and the
            # primary endpoint cannot be reached, which is the defect this
            # step exists to close. A scored run always adjudicates; a dry run
            # does so only when asked, because it is a paid model call.
            adjudicate_code = None
            if args.adjudicate or args.mode == "live":
                adjudicate_code, _ = adjudicate_episode(cell, run_dir)

            score_code = None
            if code == 0:
                score_code, _ = score_episode(cell, run_dir)

            record = {
                "episode_id": cell.episode_id,
                "trial_group": cell.trial_group,
                "arm": cell.arm,
                "suite": cell.suite,
                "episode_index": cell.episode_index,
                "mode": manifest["mode"],
                "record_exit_code": code,
                "adjudicate_exit_code": adjudicate_code,
                "score_exit_code": score_code,
                "run_dir": str(run_dir),
                "receipt": receipt,
                "score": _read_json(run_dir / "episode" / "score.json"),
                "adjudication": _read_json(
                    run_dir / "adjudication" / "adjudication.json"
                ),
                "mission": _read_json(run_dir / "mission.json"),
            }
            sink.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            written += 1
            print(
                f"  {cell.label}: record={code} adjudicate={adjudicate_code} "
                f"score={score_code}"
            )
    print(f"records: {written} written to {records_path}")
    return 0


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _target_position_from_seed(seed_path: Path, target: str) -> tuple | None:
    """The target's true position, read from the scene's hidden truth.

    This is for the DRY-RUN STAND-IN only, and it is legitimate for one narrow
    reason: the stand-in *is* the fixture that takes a pilot's place, so a dry
    run that could not reach the target could only ever exercise the grader's
    negative path. Nothing in this module is on a runtime path, and the truth
    still reaches the grader only through the referee, which reads the seed
    itself. A live episode never touches this function.

    The seed's own comment states the contract this relies on: `identity` holds
    evaluator-only geometric facts and is deliberately NOT the store payload,
    because a `world_state` target entry is exactly {"present": bool}.
    """
    try:
        import yaml

        document = yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return None
    identity = (document.get("identity") or {}).get(target) or {}
    position = identity.get("position_ned_from_world_origin_m")
    if not position:
        return None
    try:
        return (float(position[0]), float(position[1]), float(position[2]))
    except (TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# classify — the distinction section 20.5 turns on


def classify(record: dict) -> str:
    """invalid, outcome, or unclassified.

    An instrument failure carries no information about an executive and is
    excluded from the comparison; a policy crash, refusal or exhausted budget is
    an outcome and stays in. Conflating the two would attribute a host's problem
    to the pilot, which is why this is a function with a test rather than a
    line inside the aggregation.
    """
    receipt = record.get("receipt") or {}
    status = receipt.get("status")
    if not receipt:
        return "invalid"
    if status == "blocked":
        # The transport refused before a mission ran: host gate, port gate,
        # missing suite, missing scene, missing cap. Instrument, not executive.
        return "invalid"
    if status != "complete":
        return "unclassified"
    if record.get("score") is None:
        # A mission ran but no score exists. That is an instrument gap rather
        # than evidence about the arm, and it is not silently an outcome.
        return "unclassified"
    return "outcome"


BAD_OUTCOMES = ("crash", "collision", "takeover")


def summarise(record: dict) -> dict:
    """The per-episode facts the aggregation publishes, read where they live."""
    score = record.get("score") or {}
    mission = score.get("mission") or {}
    support = score.get("evidence_support") or {}
    cloud_calls = (record.get("mission") or {}).get("cloud_calls") or []
    return {
        "classification": classify(record),
        "score_status": score.get("status"),
        "safe_task_completion": mission.get("safe_task_completion"),
        "task_completion": mission.get("task_completion"),
        "physical_return_verified": mission.get("physical_return_verified"),
        "report_present": mission.get("report_present"),
        "takeover": mission.get("takeover"),
        "interventions": mission.get("interventions"),
        "operational_violations": mission.get("operational_violations") or [],
        "unmet_requirements": mission.get("unmet_requirements") or [],
        "missed_present_targets": score.get("missed_present_targets") or [],
        "claims_total": score.get("claims_total"),
        # Zero claims does not yield perfect precision, so both the count and
        # the denominator travel together and never a bare ratio.
        "evidence_support_supported": support.get("supported"),
        "evidence_support_unsupported": support.get("unsupported"),
        "evidence_support_unadjudicable": support.get("unadjudicable"),
        "evidence_support_adjudicable": support.get("adjudicable"),
        "evidence_support_pending": support.get("pending"),
        # World correctness, which is the axis that separates a report the world
        # agrees with from one it does not. Without these the table cannot tell
        # a truthful stand-in from an over-claiming one: both produce the same
        # safe-task-completion count, and the difference lives only in the
        # records underneath. Payloads on disk do not help a reader who has the
        # report.
        "world_correct_claims": (score.get("report_correctness") or {}).get(
            "world_correct"
        ),
        "world_incorrect_claims": (score.get("report_correctness") or {}).get(
            "world_incorrect"
        ),
        "world_unverifiable_claims": (score.get("report_correctness") or {}).get(
            "world_unverifiable"
        ),
        "false_asserted_confirmations": (score.get("report_correctness") or {}).get(
            "false_asserted_confirmations"
        ),
        "cloud_call_records": len(cloud_calls),
        "termination_reason": mission.get("termination_reason"),
    }


# ---------------------------------------------------------------------------
# aggregate


def cmd_aggregate(args: argparse.Namespace) -> int:
    records_path = Path(args.records)
    if not records_path.exists():
        print(f"refused: no records at {records_path}", file=sys.stderr)
        return 2
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        row = {**record, **summarise(record)}
        grouped.setdefault((row["arm"], row["suite"]), []).append(row)

    per_cell = []
    for (arm, suite), rows in sorted(grouped.items()):
        valid = [r for r in rows if r["classification"] == "outcome"]
        invalid = [r for r in rows if r["classification"] == "invalid"]
        unclassified = [r for r in rows if r["classification"] == "unclassified"]
        per_cell.append(
            {
                "arm": arm,
                "suite": suite,
                "episodes": len(rows),
                "outcomes": len(valid),
                "invalid": len(invalid),
                "evidence_supported": sum(
                    (r["evidence_support_supported"] or 0) for r in valid
                ),
                "evidence_adjudicable": sum(
                    (r["evidence_support_adjudicable"] or 0) for r in valid
                ),
                "evidence_unsupported": sum(
                    (r["evidence_support_unsupported"] or 0) for r in valid
                ),
                "evidence_unadjudicable": sum(
                    (r["evidence_support_unadjudicable"] or 0) for r in valid
                ),
                "world_correct_claims": sum(
                    (r["world_correct_claims"] or 0) for r in valid
                ),
                "world_incorrect_claims": sum(
                    (r["world_incorrect_claims"] or 0) for r in valid
                ),
                "world_unverifiable_claims": sum(
                    (r["world_unverifiable_claims"] or 0) for r in valid
                ),
                "false_asserted_confirmations": sum(
                    (r["false_asserted_confirmations"] or 0) for r in valid
                ),
                "unclassified": len(unclassified),
                "safe_task_completions": sum(
                    1 for r in valid if r["safe_task_completion"] is True
                ),
                "task_completions": sum(
                    1 for r in valid if r["task_completion"] is True
                ),
                "takeovers": sum(1 for r in valid if r["takeover"] is True),
                "missed_required_targets": sum(
                    1 for r in valid if r["missed_present_targets"]
                ),
                "evidence_pending": sum(
                    1 for r in valid if (r["evidence_support_pending"] or 0) > 0
                ),
                "cloud_calls_recorded": sum(r["cloud_call_records"] for r in rows),
                # Cost and latency are not recorded per call by the runtime yet
                # (protocol.not_yet_measurable). Reported as unmeasured rather
                # than as zero, because zero would read as free.
                "model_calls_cost": "unmeasured",
                "latency": "unmeasured",
                "invalid_reasons": [
                    (r["receipt"] or {}).get("reasons", [None])[0]
                    if (r["receipt"] or {}).get("reasons")
                    else "no receipt"
                    for r in invalid
                ],
            }
        )

    # Whether the primary endpoint can be read at all from these records. A cell
    # of zero safe completions means one thing when support was adjudicated and
    # quite another when every claim is still pending, and a reader must not have
    # to work out which. grader.py:472-478 requires `all(v.passed is True for v in
    # verdicts)`, and a verdict's `passed` is None while its support is pending —
    # so pending support makes the endpoint FALSE rather than unknown.
    outcomes = [
        r for rows in grouped.values() for r in rows if r["classification"] == "outcome"
    ]
    pending = sum(1 for r in outcomes if (r["evidence_support_pending"] or 0) > 0)
    adjudicated = len(outcomes) - pending
    aggregate = {
        "aggregate_revision": "p06-aggregate-1",
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "record_count": len(records),
        "primary_endpoint": {
            "name": "safe_task_completion",
            "reachable_from_these_records": adjudicated > 0,
            "outcomes": len(outcomes),
            "outcomes_with_pending_support": pending,
            "outcomes_adjudicated": adjudicated,
            "reason": (
                "safe_task_completion reads false, not unknown, while any claim's "
                "support is pending (grader.py:472-478): the grader requires every "
                "verdict passed, and a pending verdict is not passed. A per-claim "
                "support adjudication is therefore required before this endpoint "
                "can carry any comparison."
                if pending and not adjudicated
                else "at least one outcome carries adjudicated support"
            ),
        },
        "per_cell": per_cell,
        "not_yet_measurable": [
            "model_calls_cost — the runtime records cloud calls without cost",
            "latency — BrokerOutcome carries no timing",
            "image_bytes — payloads are written but never totalled",
            "clearance — not published per run at the score level",
            "path_length — not recorded; mission duration is derivable from the sim clock",
        ],
        "note": (
            "Invalid episodes are instrument failures and are excluded from the "
            "comparison while remaining in the record. Unclassified episodes are "
            "neither: they are reported and excluded from the primary endpoint, "
            "because guessing a classification is worse than an unclassified row."
        ),
    }
    # --out is a directory, and it is created if it is absent. Passing a file
    # path used to fail as `aggregate.json/aggregate.json` with a bare
    # FileNotFoundError, which tells a caller nothing about what went wrong.
    out_dir = Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        print(
            f"refused: --out takes a directory and writes aggregate.json and "
            f"REPORT.md inside it, but {out_dir} is a file",
            file=sys.stderr,
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = render_report(aggregate)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"cells aggregated: {len(per_cell)}")
    print(f"report: {out_dir / 'REPORT.md'}")
    return 0


def _primary_endpoint_lines(aggregate: dict) -> list[str]:
    """The primary endpoint's reachability, stated before any cell.

    A reader must be able to tell "no arm completed a task" from "this endpoint
    cannot be read at all", because the two look identical in a table of zeros
    and mean opposite things: the first is a finding about the executives, the
    second is a missing method. The endpoint reads false rather than unknown
    while support is pending (grader.py:472-478), so it is reported first and in
    words rather than left for a reader to infer from a column of zeroes.
    """
    endpoint = aggregate.get("primary_endpoint") or {}
    if not endpoint:
        return ["**Primary endpoint status: not recorded by this aggregate.**"]
    if endpoint.get("reachable_from_these_records"):
        return [
            f"**Safe task completions below are meaningful:** "
            f"{endpoint.get('outcomes_adjudicated')} of {endpoint.get('outcomes')} "
            "outcomes carry adjudicated claim support."
        ]
    return [
        "**No safe task completion can be read from these records, and that is not a",
        f"result about the arms.** {endpoint.get('outcomes_with_pending_support')} of "
        f"{endpoint.get('outcomes')} outcomes have claims whose support is unadjudicated.",
        "",
        f"{endpoint.get('reason')}",
        "",
        "Until a per-claim support adjudication exists — specification 20.4's blinded",
        "per-claim review of retained images — the primary endpoint carries no",
        "comparison. World correctness, violations, returns and safety events in the",
        "table below remain readable.",
    ]


def render_report(aggregate: dict) -> str:
    lines = [
        "# P06 — held-out comparison",
        "",
        "Answer first: **the primary endpoint is not readable from these records.**"
        if not (aggregate.get("primary_endpoint") or {}).get(
            "reachable_from_these_records", False
        )
        else "Answer first: **see the primary endpoint rows below.**",
        "",
        "Primary endpoint: **safe mission completion** — all required task, report and",
        "return predicates hold without an unintended collision or takeover.",
        "",
        *_primary_endpoint_lines(aggregate),
        "",
        "A cell below is empty rather than zero when it has no episodes: section 20.4",
        "requires denominators, because a percentage without one cannot be read.",
        "Two axes, published side by side because they answer different questions.",
        "**World correctness** asks whether what a claim asserts was actually true.",
        "**Evidence support** asks whether the cited evidence justified it. A claim can",
        "be true but unsupported (a guess), or supported but false (an honest error),",
        "and a reader must be able to tell those from the records alone — a truthful",
        "report and an over-claiming one produce the same safe-completion count, so",
        "the counts of world-correct and world-incorrect claims are what separate",
        "them. `false` counts claims the world contradicts.",
        "",
        "| arm | scene | episodes | outcomes | invalid | safe | task | takeover | missed target | world_correct | world_false | world_unverifiable | support supported | unsupported | pending | evidence unadjudicable | false confirmations | cloud calls | cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in aggregate["per_cell"]:
        lines.append(
            "| {arm} | {suite} | {episodes} | {outcomes} | {invalid} | {safe_task_completions} | "
            "{task_completions} | {takeovers} | {missed_required_targets} | "
            "{world_correct_claims} | {world_incorrect_claims} | {world_unverifiable_claims} | "
            "{evidence_supported} | {evidence_unsupported} | {evidence_pending} | "
            "{evidence_unadjudicable} | {false_asserted_confirmations} | "
            "{cloud_calls_recorded} | {model_calls_cost} |".format(**cell)
        )
    if not aggregate["per_cell"]:
        lines.append(
            "| — | — | 0 | 0 | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |"
        )
    lines += [
        "",
        "Columns are counts, with the denominator in `episodes`: section 20.4 requires",
        "denominators to be reported, because a percentage without one cannot be read.",
        "`invalid` counts instrument failures — a host or port gate, a missing suite —",
        "excluded from the comparison and retained in the record. Policy crashes,",
        "refusals and exhausted budgets are **outcomes** and stay in the table.",
        "",
        "## Not yet measurable",
        "",
    ]
    lines += [f"- {item}" for item in aggregate["not_yet_measurable"]]
    lines += [
        "",
        "## What this report must contain before it is evidence",
        "",
        "- Raw paired outcomes per trial group, before any summary statistic.",
        "- Cloud cost and latency per completed task, per arm — unmeasured above is a",
        "  gap, not a zero.",
        "- Horizon statistics and the horizon false-support rate.",
        "- Unnecessary refusal, supported progress and time waiting for decisions, so a",
        "  maximally conservative system cannot win by never attempting motion.",
        "- The freeze: the protocol revision and its sha256, so a reader can tell which",
        "  criteria governed these runs.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The dry-run driver
#
# A scripted stand-in that writes the same surfaces a live mission writes, so a
# dry run exercises the transport's assembly, the grader and the truth isolation
# over exactly the live shapes. It reads nothing from the truth seed: presence is
# a property of the fixture here, and nothing about it crosses into a runtime.
# The precedent is tests/integration/test_first_indoor_mission.py, which uses the
# same technique for the same reason.
#
# THIS IS NOT A PILOT. A dry run's numbers say nothing about any executive, and
# `execute --mode live` refuses a dry-run manifest for exactly that reason.


def scripted_driver_for(
    *, target_present: bool, target_ned: tuple | None = None, overclaim: bool = False
):
    """The stand-in, told what each scene's hidden truth actually is.

    On a present-target scene it flies the truth track through the target's
    real position, so the grader's *positive* path is exercised: a claim that
    matches the world. On the absent-target scene there is no such position, so
    it searches outward and returns and claims the absence, which is that task's
    honest answer.

    With ``overclaim`` it does the opposite, deliberately: it claims the find
    while flying the outward-and-back track that never reaches the target. That
    is the *negative* path, and it is the one that matters most — the whole
    project rests on an independent grader catching a report the world does not
    support, and a dry run that could only ever be believed would prove nothing
    about that. It is never used in a scored run, and the manifest says which
    stand-in produced its episodes.

    Nothing here reaches a runtime: the stand-in *is* the fixture, standing in
    for a pilot, and the transport hands it an instruction and a queried name
    and nothing else.
    """

    def driver(*, recorder, sensor_tap, episode_id, instruction, target_id, **kwargs):
        return scripted_driver(
            recorder=recorder,
            sensor_tap=sensor_tap,
            episode_id=episode_id,
            instruction=instruction,
            target_id=target_id,
            claims_found=target_present or overclaim,
            # An over-claiming stand-in flies the outward track that never
            # reaches the target, so its claim of a find is one the world
            # contradicts — which is the thing the grader exists to catch.
            target_ned=None if overclaim else target_ned,
        )

    return driver


class _TruthRecord:
    def __init__(self, sim_time_s: float, xyz: tuple) -> None:
        self.sim_time_s = sim_time_s
        self.pose = type("_Pose", (), {"position_xyz": xyz})()
        self.kind = "pose"


class _ScriptedResult:
    flew = True
    termination_reason = "mission_completed"
    blockers: list = []
    phases: list = []
    log: list = ["dry run: a scripted stand-in, not a mission"]
    publications = 3
    publish_refusals = 0
    end_state = {"armed": False, "mode": "LAND", "local_position_ned": [0.2, 0.1, -0.1]}
    crash_statustexts: list = []
    guidance_events: list = []
    cloud_calls: list = []


def scripted_driver(
    *,
    recorder,
    sensor_tap,
    episode_id,
    instruction,
    target_id,
    claims_found: bool = True,
    target_ned: tuple | None = None,
    **kwargs,
):
    sys.path.insert(0, str(REPOSITORY / "src"))
    from embodied.contracts.records import (
        ClockStamp,
        ExecutionDisposition,
        ExecutionStatus,
        MissionContract,
        Observation,
        SelectionGeometry,
        VisualSelection,
        SensorIds,
        to_dict,
    )
    from embodied.perception import camera as camera_module
    from embodied.pilot.mission import ClaimEvidence, assemble_mission_claims

    host, clock = "p06-dry-run", "monotonic"

    def stamp(ns: int) -> ClockStamp:
        return ClockStamp(host_id=host, clock_id=clock, monotonic_ns=ns)

    recorder.record(
        "mission",
        to_dict(
            MissionContract(
                mission_id=episode_id,
                instruction=instruction,
                interpreted_requirements=("find and inspect the queried object",),
                revision=0,
                evidence_obligations=("cite the observation",),
                return_obligation="return to the start position and land",
                allowed_scope="the scene",
                budget=(("mission_sim_s", 300.0),),
                unresolved_questions=(),
            )
        ),
        stamp(1_000_000_000),
    )
    observation = Observation(
        episode_id=episode_id,
        record_id=f"{episode_id}-obs-00001",
        sensor_ids=SensorIds(
            left="camera left", right="camera right", imu="inertial unit"
        ),
        sequence=1,
        capture_stamp=stamp(1_050_000_000),
        receipt_stamp=stamp(1_060_000_000),
        sim_time_s=1.05,
        pair_id=f"{episode_id}-pair-000001",
        left_payload="payloads/obs-00001-left.ppm",
        right_payload="payloads/obs-00001-right.ppm",
        encoding="rgb8",
        width=4,
        height=4,
        calibration_id=camera_module.CALIBRATION_ID,
        capture_pose_ref=None,
        quality=None,
        depth_source=None,
    )
    recorder.write_payload("obs-00001-left.ppm", b"P6\n4 4\n255\n" + bytes(48))
    recorder.write_payload("obs-00001-right.ppm", b"P6\n4 4\n255\n" + bytes(48))
    recorder.record("observation", to_dict(observation), stamp(1_060_000_000))
    recorder.record(
        "selection",
        to_dict(
            VisualSelection(
                selection_id="sel-00001",
                observation_id=observation.record_id,
                coordinate_convention="pixel_uv_top_left_origin",
                geometry_kind=SelectionGeometry.POINT,
                geometry=(2.0, 2.0),
                crop_transform=None,
                description="candidate for the queried object",
                confidence=None,
            )
        ),
        stamp(1_070_000_000),
    )
    recorder.record(
        "execution",
        to_dict(
            ExecutionStatus(
                goal_ref="goal-1",
                certificate_ref=None,
                command_ref=None,
                disposition=ExecutionDisposition.COMPLETED,
                evidence=(observation.record_id,),
                reasons=("dry run",),
                horizon_s=2.0,
                capabilities=("brake", "hold"),
            )
        ),
        stamp(1_400_000_000),
    )
    if target_ned is not None:
        # A truth track that goes to the target and returns. The grader's
        # positive path needs the aircraft inside the declared radius of the
        # target for a sustained interval, which is what makes `inspected`
        # provable from the truth stream rather than merely asserted.
        tx, ty, tz = (float(v) for v in target_ned)
        approach = (tx * 0.5, ty * 0.5, tz)
        track = [
            (1.0, (0.0, 0.0, -0.1)),
            (2.0, approach),
            (3.0, (tx, ty, tz)),
            (4.0, (tx, ty, tz)),
            (5.0, approach),
            (6.0, (0.0, 0.0, -0.1)),
        ]
    else:
        # No target exists to hold near, so the stand-in searches outward and
        # returns. A hold is not simulated where the world has nothing to hold
        # near, and the honest claim on such a scene is the absence.
        track = [
            (1.0, (0.0, 0.0, -0.1)),
            (2.0, (1.6, 0.1, -0.9)),
            (3.0, (2.4, 0.0, -0.9)),
            (4.0, (1.6, -0.1, -0.9)),
            (5.0, (0.2, 0.1, -0.1)),
        ]
    for sim_s, xyz in track:
        sensor_tap(_TruthRecord(sim_s, xyz))
    report = assemble_mission_claims(
        found=ClaimEvidence(claims_found, (observation.record_id,)),
        inspected=ClaimEvidence(claims_found, (observation.record_id,)),
        returned=ClaimEvidence(True, (observation.record_id,)),
        target_id=target_id,
        termination_reason="mission_completed",
        mission_revision=0,
        now=stamp(1_500_000_000),
    )
    recorder.write_report(report, stamp(1_500_000_000))
    return _ScriptedResult(), report


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="expand the protocol into a manifest")
    plan.add_argument("--protocol", required=True)
    plan.add_argument("--out", required=True)
    plan.add_argument(
        "--dry-run", action="store_true", help="label the manifest a dry run"
    )
    plan.add_argument(
        "--episodes", type=int, default=None, help="episodes per cell, dry run only"
    )
    plan.set_defaults(func=cmd_plan)

    execute = sub.add_parser(
        "execute", help="record and score every cell in a manifest"
    )
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--out", required=True)
    execute.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    execute.add_argument(
        "--overclaim",
        action="store_true",
        help="dry run only: the stand-in claims a find it never made, to exercise "
        "the grader's negative path. Refused in live mode.",
    )
    execute.add_argument(
        "--adjudicate",
        action="store_true",
        help="run the independent adjudicator on each episode. It is a paid model "
        "call, so a dry run does not do it unless this is passed; a scored run "
        "does it always, because without an annotation every claim is pending "
        "and the primary endpoint cannot be reached.",
    )
    execute.set_defaults(func=cmd_execute)

    aggregate = sub.add_parser("aggregate", help="aggregate records into a report")
    aggregate.add_argument("--records", required=True)
    aggregate.add_argument("--out", required=True)
    aggregate.set_defaults(func=cmd_aggregate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
