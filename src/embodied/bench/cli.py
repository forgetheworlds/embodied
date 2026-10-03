"""The ``bench`` command: record, replay and score episode directories.

Handlers live here and register with :func:`embodied.cli.register_command` at
import time; the dispatcher imports this module through ``COMMAND_MODULES``, so
no shared parser is edited. Subcommands accept ``--output`` both before and
after their own name: the shared dispatcher adds it to this command, and the
leaves re-declare it with a suppressed default so a leaf-level flag overrides
without a parent-level default clobbering it back.

This module never names a bench-private file or the store that holds hidden
facts. The score path delegates to the scorer module, which owns that
surface; the replay path can only reach the agent projection, which refuses
anything else.

Exit codes follow the shared receipt (CLI-PLAN): record refuses an
unregistered suite with status blocked (2); a score whose bench-side store is
missing is a missing prerequisite, also blocked (2) with the store path
named; replay and a completed score exit 0 even when the episode itself
failed — a completed losing episode is valid data — and a score awaiting
support adjudication exits 3.
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
from embodied.bench import adjudicate, grader, recorder
from embodied.bench.events import EpisodeError, EventError

_SUBCOMMANDS = ("record", "replay", "score", "adjudicate")


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
    subparsers = parser.add_subparsers(
        dest="bench_command", metavar="{" + ",".join(_SUBCOMMANDS) + "}"
    )

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

    adjudicate_command = subparsers.add_parser(
        "adjudicate",
        help="have the independent adjudicator model write an episode's support annotation",
    )
    adjudicate_command.add_argument(
        "--episode",
        type=Path,
        required=True,
        help="episode directory whose claims the adjudicator reviews",
    )
    adjudicate_command.add_argument(
        "--adjudication",
        type=Path,
        default=None,
        help="path for the annotation; default is adjudication.json inside the "
        "episode, which is where bench score looks for it",
    )
    _accept_output(adjudicate_command)
    adjudicate_command.set_defaults(_bench_handler=_adjudicate)


def _record(args: argparse.Namespace, output: Path) -> CommandOutcome:
    """Record one live episode from a registered suite.

    The suite's own declaration registers it (P05 owns first-indoor; P02 owns
    none), and the transport that flies the mission lives in
    :mod:`embodied.bench.live_record`. A suite is selected by the name its own
    declaration carries, and the pointers in that declaration supply the scene
    and the truth seed this run is graded against — the default suite's are
    never substituted for them. An unregistered suite is still refused rather
    than substituted, and so is a sensor mode the suite does not declare: the
    disagreement is the run's blocker, not something to paper over by recording
    with a mode nobody declared.
    """
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
    from embodied.bench import live_record

    try:
        suite_config = live_record.suite_config_path_for(args.suite)
    except live_record.SuiteConfigError as error:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(f"the suite cannot be selected: {error}",),
            limitations=limitations,
            manifest=manifest,
            sensor_mode=sensor_mode,
        )
    try:
        suite_document = live_record.load_suite_document(suite_config)
        registered_name = live_record.register_suite(suite_document)
    except live_record.SuiteConfigError as error:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(f"the suite declaration cannot be used: {error}",),
            limitations=limitations,
            manifest=manifest,
            sensor_mode=sensor_mode,
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
    if registered_name != suite.name or suite_document is None:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(
                f"suite {suite.name!r} is registered without a usable declaration in this "
                "build; refusing to fly a suite whose own file cannot be read",
            ),
            limitations=limitations,
            manifest=manifest,
            sensor_mode=sensor_mode,
        )
    try:
        truth_seed = live_record.declared_path(suite_document, "truth_seed")
    except live_record.SuiteConfigError as error:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(str(error),),
            limitations=limitations,
            manifest=manifest,
            sensor_mode=sensor_mode,
        )
    return live_record.record(
        suite_name=args.suite,
        suite_document=suite_document,
        arm=args.arm,
        sensor_mode=sensor_mode,
        output=output,
        suite_config=suite_config,
        truth_seed=truth_seed,
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
    except grader.StoreMissing as error:
        # The bench-side store is a missing prerequisite, not a command
        # mistake: report blocked with the store path named, and write no
        # score — there are no hidden facts to grade against.
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(str(error),),
            limitations=(
                "an episode without its bench-side store cannot be scored; "
                "no score was written",
            ),
            manifest={"episode": str(episode)},
        )
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

def _adjudicate(args: argparse.Namespace, output: Path) -> CommandOutcome:
    """Produce the per-claim support annotation the score consumes.

    Until this exists the primary endpoint cannot pass: the grader holds every
    claim at ``passed = None`` while support is pending, and nothing produced
    the annotation. The adjudicator is a model from a different family than the
    pilot's, reading the agent projection only and blinded to the arm.

    A transport failure is reported **blocked** rather than written as a
    verdict: an instrument that could not be reached must not look like an
    adjudicator that found the claims unsupported.
    """
    episode = Path(args.episode)
    target = (
        Path(args.adjudication)
        if args.adjudication
        else episode / grader.ADJUDICATION_FILENAME
    )
    try:
        run = adjudicate.adjudicate_episode(episode, output_path=target)
    except adjudicate.AdjudicatorError as error:
        return CommandOutcome(
            status=CommandStatus.BLOCKED,
            gate_status=GateStatus.NOT_APPLICABLE,
            reasons=(f"the adjudicator could not review this episode: {error}",),
            limitations=(
                "an instrument failure is not a verdict; no annotation was written",
            ),
            manifest={"episode": str(episode)},
            artifacts=(),
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / grader.ADJUDICATION_FILENAME).write_bytes(target.read_bytes())
    verdicts: dict[str, int] = {}
    for entry in run.document["entries"]:
        verdicts[entry["verdict"]] = verdicts.get(entry["verdict"], 0) + 1
    manifest = recorder.AgentSurface.open(episode).manifest
    return CommandOutcome(
        status=CommandStatus.COMPLETE,
        gate_status=GateStatus.NOT_APPLICABLE,
        reasons=(
            f"{len(run.document['entries'])} claim(s) reviewed by {run.reviewer}: "
            + ", ".join(f"{name} {count}" for name, count in sorted(verdicts.items())),
            f"annotation: {target}",
        ),
        limitations=(
            "the adjudicator judges evidence support, not world correctness; the "
            "two are reported separately and neither substitutes for the other",
            "model review carries variance; the model identity and the rubric "
            "revision are recorded in every annotation it writes",
        ),
        manifest={
            "episode": str(episode),
            "episode_id": manifest.episode_id,
            "adjudicator": run.reviewer,
            "calls": run.calls,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "prompt_sha256": run.prompt_sha256,
            "resolved_model_identities": list(run.resolved_identities),
        },
        artifacts=(grader.ADJUDICATION_FILENAME,),
        episode_id=manifest.episode_id,
        sensor_mode=manifest.sensor_mode,
    )


def _bench(args: argparse.Namespace, output: Path) -> CommandOutcome:
    handler = getattr(args, "_bench_handler", None)
    if handler is None:
        raise CommandError(f"bench needs a subcommand: {', '.join(_SUBCOMMANDS)}")
    return handler(args, output)


register_command(
    "bench",
    _bench,
    help_text="record, replay, adjudicate and score benchmark episodes",
    stage_id="P02",
    run_prefix="p02-bench",
    add_arguments=_add_bench_arguments,
)
