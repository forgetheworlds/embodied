"""The ``bench`` command: record, replay and score episode directories.

Handlers live here and register with :func:`embodied.cli.register_command` at
import time; the dispatcher imports this module through ``COMMAND_MODULES``, so
no shared parser is edited. Subcommands accept ``--output`` both before and
after their own name: the shared dispatcher adds it to this command, and the
leaves re-declare it with a suppressed default so a leaf-level flag overrides
without a parent-level default clobbering it back.

This module never names a bench-private file. The score path delegates to the
scorer module, which owns that surface; the replay path can only reach the
agent projection, which refuses anything else.

Exit codes follow the shared receipt (CLI-PLAN): record refuses an
unregistered suite with status blocked (2); replay and a completed score exit
0 even when the episode itself failed — a completed losing episode is valid
data — and a score awaiting support adjudication exits 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied.cli import (
    CommandError,
    CommandOutcome,
    CommandStatus,
    GateStatus,
    register_command,
)
from embodied.contracts.records import SensorMode
from embodied.bench import grader, recorder
from embodied.bench.events import EpisodeError, EventError

_SUBCOMMANDS = ("record", "replay", "score")


def _accept_output(parser: argparse.ArgumentParser) -> None:
    """Let ``--output`` also appear after the subcommand.

    ``default=argparse.SUPPRESS`` keeps the subparser from overwriting a value
    the parent parser already set when the flag appears before the
    subcommand instead.
    """
    parser.add_argument(
        "--output",
        default=argparse.SUPPRESS,
        help="directory for the receipt, manifest and artifacts "
        "(same flag as the one before the subcommand; default: a fresh directory under work/runs)",
    )


def _add_bench_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="bench_command", metavar="{record,replay,score}")

    record = subparsers.add_parser(
        "record", help="record a live episode using a registered suite"
    )
    record.add_argument(
        "--suite",
        required=True,
        help="suite id to record with; first-indoor is registered by P05, and an "
        "unregistered suite is refused rather than substituted",
    )
    record.add_argument(
        "--sensor-mode",
        required=True,
        choices=[mode.value for mode in SensorMode],
        help="declared state-derivation mode for the run",
    )
    record.add_argument(
        "--arm", required=True, help="experiment arm (B0..B3); admission is P06's check"
    )
    _accept_output(record)
    record.set_defaults(_bench_handler=_record)

    replay = subparsers.add_parser(
        "replay", help="reconstruct an episode's timeline and render synchronized evidence"
    )
    replay.add_argument(
        "--episode", type=Path, required=True, help="episode directory to replay"
    )
    _accept_output(replay)
    replay.set_defaults(_bench_handler=_replay)

    score = subparsers.add_parser(
        "score", help="independently score an episode against hidden facts and annotations"
    )
    score.add_argument("--episode", type=Path, required=True, help="episode directory to score")
    score.add_argument(
        "--adjudication",
        type=Path,
        default=None,
        help="versioned support annotation to use instead of the episode's own "
        "adjudication.json; the choice is recorded in the score",
    )
    _accept_output(score)
    score.set_defaults(_bench_handler=_score)


def _record(args: argparse.Namespace, output: Path) -> CommandOutcome:
    sensor_mode = SensorMode(args.sensor_mode)
    manifest = {
        "suite": args.suite,
        "arm": args.arm,
        "sensor_mode": sensor_mode.value,
    }
    limitations = (
        "P02 owns offline record storage; a live episode is recorded by the stage "
        "that runs it, from a registered suite",
    )
    try:
        suite = recorder.resolve_suite(args.suite)
    except recorder.SuiteError as error:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(str(error),),
            limitations=limitations,
            manifest=manifest,
            sensor_mode=sensor_mode,
        )
    if args.sensor_mode != suite.localization_mode:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(
                f"suite {suite.name!r} declares localization mode "
                f"{suite.localization_mode!r}, not {args.sensor_mode!r}",
            ),
            limitations=limitations,
            manifest=manifest,
            sensor_mode=sensor_mode,
        )
    return CommandOutcome(
        status=CommandStatus.BLOCKED,
        gate_status=GateStatus.NOT_APPLICABLE,
        reasons=(
            f"suite {suite.name!r} is registered, but this build has no live recording "
            "transport attached; refusing to substitute a synthetic episode",
        ),
        limitations=limitations,
        manifest=manifest,
        sensor_mode=sensor_mode,
    )


def _replay(args: argparse.Namespace, output: Path) -> CommandOutcome:
    try:
        document = recorder.replay(Path(args.episode))
    except (EpisodeError, EventError) as error:
        raise CommandError(f"replay: {error}") from error
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sensor = document["sensor_mode"]
    return CommandOutcome(
        status=CommandStatus.COMPLETE,
        gate_status=GateStatus.PASS,
        reasons=(
            f"replayed {document['event_count']} events in recorded order over "
            f"{document['span_ns']} ns; no model call and no physical run",
        ),
        manifest={
            "episode": str(args.episode),
            "episode_id": document["episode_id"],
            "events_revision": document["events_revision"],
            "counts": document["counts"],
        },
        artifacts=("replay.json",),
        episode_id=document["episode_id"],
        sensor_mode=SensorMode(sensor) if sensor else None,
    )


def _score(args: argparse.Namespace, output: Path) -> CommandOutcome:
    episode = Path(args.episode)
    try:
        score = grader.grade(episode, args.adjudication)
    except (EpisodeError, EventError, grader.GradeError) as error:
        raise CommandError(f"score: {error}") from error
    output.mkdir(parents=True, exist_ok=True)
    # The episode's score.json is canonical; the receipt hashes an identical
    # copy placed beside it so the receipt names what it describes.
    (output / grader.SCORE_FILENAME).write_bytes(
        (episode / grader.SCORE_FILENAME).read_bytes()
    )
    manifest = recorder.AgentSurface.open(episode).manifest
    limitations = (
        "synthetic grading proves grader behaviour only; it does not prove flight "
        "or benchmark validity",
    )
    outcome_manifest = {
        "episode": str(episode),
        "episode_id": score.episode_id,
        "score_status": score.status,
        "adjudication": score.adjudication["source"] if score.adjudication else None,
        "claims_total": len(score.claims),
    }
    common: dict = {
        "limitations": limitations,
        "manifest": outcome_manifest,
        "artifacts": (grader.SCORE_FILENAME,),
        "episode_id": score.episode_id,
        "sensor_mode": manifest.sensor_mode,
    }
    if score.status == "pending":
        return CommandOutcome(
            status=CommandStatus.PENDING,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(
                "at least one claim has no support annotation; support is pending, "
                "not passed (exit 3: pending mandatory adjudication)",
                f"score: {episode / grader.SCORE_FILENAME}",
            ),
            **common,
        )
    passed = sum(1 for claim in score.claims if claim.passed is True)
    failed = sum(1 for claim in score.claims if claim.passed is False)
    undecided = len(score.claims) - passed - failed
    return CommandOutcome(
        status=CommandStatus.COMPLETE,
        # The gate is the episode's own verdict: a completed failing episode is
        # valid data (exit 0) with gate fail.
        gate_status=GateStatus.PASS if score.fully_passed else GateStatus.FAIL,
        reasons=(
            f"{len(score.claims)} claims: {passed} supported and world-correct, "
            f"{failed} failed, {undecided} unverifiable",
            f"score: {episode / grader.SCORE_FILENAME}",
        ),
        **common,
    )


def _bench(args: argparse.Namespace, output: Path) -> CommandOutcome:
    handler = getattr(args, "_bench_handler", None)
    if handler is None:
        raise CommandError(f"bench needs a subcommand: {', '.join(_SUBCOMMANDS)}")
    return handler(args, output)


register_command(
    "bench",
    _bench,
    help_text="record, replay and score benchmark episodes",
    stage_id="P02",
    run_prefix="p02-bench",
    add_arguments=_add_bench_arguments,
)
