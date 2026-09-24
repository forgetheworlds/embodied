"""Replay must preserve what was recorded: order, stamps, revisions, results.

The slice's pass condition for replay: "Replay preserves observation, goal
revision, motion target, execution result, timestamps and intervention order.
If … replay changes event order, P02 fails." Every expected value here is
written by hand from the fixture's own files, so a test failure means the
record or the replay changed, not that a fixture-generated expectation moved
with it.

These tests also carry the negative side of the truth-isolation invariant:
the agent projection refuses bench-private members, the agent stream rejects
a foreign kind smuggled into it, and the agent-facing modules name no
bench-private file, no bench-side store and no reader for either.
"""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

import embodied.bench.cli  # registers the bench command with the dispatcher
from embodied.cli import main
from embodied.bench import recorder
from embodied.bench.events import EpisodeError, EventError
from embodied.bench.recorder import AgentSurface, Recorder, RunManifest, SurfaceViolation, replay

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "hand-checkable-episode"

HOST, CLOCK = "synthetic-bench-0", "monotonic"

# (kind, monotonic_ns) for all 23 fixture events, in recorded order.
EXPECTED_TIMELINE = [
    ("mission", 1_000_000_000),
    ("observation", 1_005_000_000),
    ("request", 1_075_000_000),
    ("selection", 1_090_000_000),
    ("goal", 1_100_000_000),
    ("goal_status", 1_150_000_000),
    ("setpoint", 1_200_000_000),
    ("execution", 1_300_000_000),
    ("observation", 2_005_000_000),
    ("request", 2_075_000_000),
    ("selection", 2_090_000_000),
    ("goal", 2_100_000_000),
    ("goal_status", 2_150_000_000),
    ("goal_status", 2_160_000_000),
    ("setpoint", 2_200_000_000),
    ("execution", 2_300_000_000),
    ("observation", 3_005_000_000),
    ("intervention", 3_500_000_000),
    ("observation", 4_005_000_000),
    ("setpoint", 4_400_000_000),
    ("observation", 5_005_000_000),
    ("execution", 5_200_000_000),
    ("report", 5_500_000_000),
]

EXPECTED_CAPTURES = [
    ("obs-0", 1_000_000_000, 1_005_000_000),
    ("obs-1", 2_000_000_000, 2_005_000_000),
    ("obs-2", 3_000_000_000, 3_005_000_000),
    ("obs-3", 4_000_000_000, 4_005_000_000),
    ("obs-4", 5_000_000_000, 5_005_000_000),
]


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "episode"
    shutil.copytree(FIXTURE, destination)
    return destination


def _rehash(episode_dir: Path, member: str) -> None:
    """Keep the manifest honest after a test deliberately edits a member."""
    manifest_path = episode_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][member] = hashlib.sha256(
        (episode_dir / member).read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _timeline(document: dict) -> list[tuple[str, int]]:
    return [(entry["kind"], entry["monotonic_ns"]) for entry in document["timeline"]]


def test_replay_reconstructs_the_recorded_order_and_timestamps(tmp_path):
    episode = _copy_fixture(tmp_path)
    document = replay(episode)
    assert _timeline(document) == EXPECTED_TIMELINE
    assert document["episode_id"] == "hand-checkable-001"
    assert document["event_count"] == 23
    assert document["clock_domain"] == {"host_id": HOST, "clock_id": CLOCK}
    assert document["span_ns"] == 4_500_000_000


def test_replay_preserves_capture_and_receipt_stamps(tmp_path):
    episode = _copy_fixture(tmp_path)
    events = AgentSurface.open(episode).agent_events()
    captures = [
        (
            event.payload["record_id"],
            event.payload["capture_stamp"]["monotonic_ns"],
            event.payload["receipt_stamp"]["monotonic_ns"],
        )
        for event in events
        if event.kind == "observation"
    ]
    assert captures == EXPECTED_CAPTURES


def test_replay_preserves_goal_revisions_motion_targets_and_execution_results(tmp_path):
    episode = _copy_fixture(tmp_path)
    document = replay(episode)

    goals = [e["summary"] for e in document["timeline"] if e["kind"] == "goal"]
    assert [goal["proposal_id"] for goal in goals] == ["g1", "g2"]
    assert [goal["base_goal_revision"] for goal in goals] == [0, 1]

    statuses = [e["summary"] for e in document["timeline"] if e["kind"] == "goal_status"]
    assert [status["disposition"] for status in statuses] == [
        "accepted",
        "superseded",
        "accepted",
    ]
    assert [status["current_disposition"] for status in statuses] == [
        "running",
        "cancelled",
        "running",
    ]

    setpoints = [e["summary"] for e in document["timeline"] if e["kind"] == "setpoint"]
    assert [setpoint["command_sequence"] for setpoint in setpoints] == [0, 1, 2]
    assert [setpoint["goal_revision"] for setpoint in setpoints] == [0, 1, 1]
    assert [setpoint["position_ned"] for setpoint in setpoints] == [
        [1.0, 0.0, -0.8],
        [2.5, 0.4, -0.8],
        [0.1, 0.0, -0.8],
    ]
    assert {setpoint["frame"] for setpoint in setpoints} == {"odom"}
    assert {setpoint["nav_epoch"] for setpoint in setpoints} == {"nav-epoch-0"}

    executions = [e["summary"] for e in document["timeline"] if e["kind"] == "execution"]
    assert [execution["goal_ref"] for execution in executions] == ["g1", "g1", "g2"]
    assert [execution["disposition"] for execution in executions] == [
        "running",
        "cancelled",
        "completed",
    ]


def test_replay_places_the_intervention_between_its_neighbours(tmp_path):
    episode = _copy_fixture(tmp_path)
    timeline = replay(episode)["timeline"]
    assert timeline[16]["kind"] == "observation"
    intervention = timeline[17]
    assert intervention["kind"] == "intervention"
    assert intervention["summary"] == {
        "intervention_id": "iv-0",
        "actor": "operator",
        "category": "operator_hold",
    }
    assert timeline[18]["kind"] == "observation"
    interventions = [entry for entry in timeline if entry["kind"] == "intervention"]
    assert len(interventions) == 1


def test_a_recorder_round_trip_preserves_every_event(tmp_path):
    from embodied.contracts.records import (
        ClockStamp,
        FinalReport,
        Observation,
        SensorIds,
        to_dict,
    )

    episode = tmp_path / "round-trip"
    first = Recorder(episode)
    left = first.write_payload("o0-left.txt", b"left frame\n")
    right = first.write_payload("o0-right.txt", b"right frame\n")
    observation = Observation(
        episode_id="ep-round-trip",
        record_id="o0",
        sensor_ids=SensorIds(left="cam-l", right="cam-r", imu="imu"),
        sequence=0,
        capture_stamp=ClockStamp(host_id="test-host", clock_id="monotonic", monotonic_ns=1_000),
        receipt_stamp=ClockStamp(host_id="test-host", clock_id="monotonic", monotonic_ns=1_500),
        sim_time_s=None,
        pair_id="p0",
        left_payload=left,
        right_payload=right,
        encoding="text/plain",
        width=1,
        height=1,
        calibration_id="cal-0",
        capture_pose_ref=None,
        quality=None,
        depth_source=None,
    )
    first.record(
        "observation", observation, ClockStamp("test-host", "monotonic", 1_500)
    )

    # A second recorder on the same directory continues the stream (an
    # interrupted live recording can still be completed).
    second = Recorder(episode)
    second.record(
        "intervention",
        {
            "intervention_id": "iv-rt",
            "actor": "operator",
            "category": "operator_hold",
            "reason": "round-trip check",
        },
        ClockStamp("test-host", "monotonic", 2_000),
    )
    report = FinalReport(
        mission_revision=0,
        claims=(),
        termination_reason="stopped",
        unmet_requirements=(),
        physical_return_status=None,
        evidence_snapshot_ids=(),
    )
    second.write_report(report, ClockStamp("test-host", "monotonic", 2_500))
    manifest = second.close(
        RunManifest(episode_id="ep-round-trip", episode_kind="synthetic-fixture")
    )

    surface = AgentSurface.open(episode)
    surface.verify_artifacts()
    events = surface.agent_events()
    assert [event.seq for event in events] == [0, 1, 2]
    assert [event.kind for event in events] == ["observation", "intervention", "report"]
    assert events[0].payload == to_dict(observation)
    assert [event.stamp.monotonic_ns for event in events] == [1_500, 2_000, 2_500]
    assert surface.final_report() == report
    assert surface.read_payload(left) == b"left frame\n"

    # The manifest binds every agent-side file it declares.
    assert set(manifest.artifacts) == {
        "agent-events.jsonl",
        "final-report.json",
        "payloads/o0-left.txt",
        "payloads/o0-right.txt",
    }
    document = replay(episode)
    assert _timeline(document) == [
        ("observation", 1_500),
        ("intervention", 2_000),
        ("report", 2_500),
    ]

    # A closed episode is not reopened for recording.
    with pytest.raises(EpisodeError):
        Recorder(episode)


def test_the_agent_surface_refuses_bench_private_members(tmp_path):
    surface = AgentSurface.open(FIXTURE)
    for member in (
        "truth-events.jsonl",
        "adjudication.json",
        "score.json",
        "../truth-events.jsonl",
        "payloads/../truth-events.jsonl",
        "payloads/../../manifest.json",
    ):
        with pytest.raises(SurfaceViolation):
            surface.read_member(member)
    # A member inside the projection is readable, so the refusals above are
    # the projection speaking, not a broken accessor.
    assert b"episode_id" in surface.read_member("manifest.json")
    assert set(surface.manifest.permitted_projection) == {
        "manifest.json",
        "agent-events.jsonl",
        "final-report.json",
        "payloads/",
    }
    # No accessor exists for the private side either.
    assert not any(
        name for name in dir(surface) if "truth" in name or "adjudication" in name
    )


def test_the_agent_stream_rejects_a_foreign_kind_smuggled_into_it(tmp_path):
    episode = _copy_fixture(tmp_path)
    smuggled = {
        "seq": 23,
        "kind": "world_state",
        "stamp": {"host_id": HOST, "clock_id": CLOCK, "monotonic_ns": 5_500_000_001},
        "sim_time_s": None,
        "payload": {"targets": {}, "world_counts": {}},
    }
    with (episode / "agent-events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(smuggled, sort_keys=True) + "\n")
    _rehash(episode, "agent-events.jsonl")  # so the integrity check passes
    with pytest.raises(EventError, match="cannot be of kind 'world_state'"):
        replay(episode)


def test_the_agent_side_modules_name_no_private_file_or_reader():
    recorder_source = Path(recorder.__file__).read_text(encoding="utf-8")
    for token in ("truth-events", "referee", "grader", ".truth", "truth_store"):
        assert token not in recorder_source, f"recorder.py names {token!r}"
    cli_source = Path(embodied.bench.cli.__file__).read_text(encoding="utf-8")
    for token in ("truth-events", "referee", ".truth", "truth_store"):
        assert token not in cli_source, f"bench/cli.py names {token!r}"
    # The referee module is write-only: no function reads its stream.
    from embodied.bench import referee

    reader_names = [
        name
        for name in vars(referee)
        if name.startswith("read")
    ]
    assert reader_names == []


def test_record_refuses_a_suite_that_is_not_registered(tmp_path):
    output = tmp_path / "run"
    code = main(
        [
            "bench",
            "record",
            "--suite",
            "first-indoor",
            "--sensor-mode",
            "sensor-derived",
            "--arm",
            "B2",
            "--output",
            str(output),
        ]
    )
    assert code == 2
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["gate_status"] == "not_applicable"
    assert "first-indoor" in receipt["reasons"][0]
    assert "P05" in receipt["reasons"][0]
    assert receipt["sensor_mode"] == "sensor-derived"
    assert receipt["stage_id"] == "P02"


def test_the_replay_command_renders_the_timeline_without_touching_the_episode(tmp_path):
    output = tmp_path / "run"
    members_before = recorder.episode_members(FIXTURE)
    events_before = hashlib.sha256(
        (FIXTURE / "agent-events.jsonl").read_bytes()
    ).hexdigest()
    code = main(
        ["bench", "replay", "--episode", str(FIXTURE), "--output", str(output)]
    )
    assert code == 0
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["gate_status"] == "pass"
    assert receipt["episode_id"] == "hand-checkable-001"
    document = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    assert _timeline(document) == EXPECTED_TIMELINE
    # Read-only: the episode is byte-identical and no file appeared in it.
    assert recorder.episode_members(FIXTURE) == members_before
    assert (
        hashlib.sha256((FIXTURE / "agent-events.jsonl").read_bytes()).hexdigest()
        == events_before
    )
