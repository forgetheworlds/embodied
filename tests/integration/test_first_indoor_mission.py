"""The first-indoor mission chain, end to end and offline where it can be.

What this file proves, each with the code that actually runs in a live
recording rather than a re-implementation of it:

* the suite registers only from its own declaration, and a build without that
  declaration still refuses (the P02 refusal path survives registration);
* the host gate blocks a loaded host before anything is started, and is wired into
  the run rather than merely present; the tests that record supply their own host
  state so a busy machine cannot fail them;
* ``bench record``'s admission rules (arm, declared sensor mode) are the run's
  blockers, not silently-overridden choices;
* the transport assembles a real episode — agent stream, manifest, bench-side
  store — and the grader scores it, including the pending-support rule and a
  supported pass;
* truth isolation is structural: the agent projection cannot reach the
  bench-side store, the episode directory holds no truth member, and the
  runtime module holds no truth-reader;
* the mission policy's own pieces (instruction interpretation, the claim
  assembler, the shared colour proposer) behave as declared;
* the runtime's offline surface (frontier clustering, target resolution, the
  no-goal publication refusal) works against a real MapStore.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import textwrap

import numpy as np
import pytest
import yaml

from embodied.bench import live_record
from embodied.bench.grader import ADJUDICATION_FILENAME, SCORE_FILENAME, grade
from embodied.bench.recorder import AgentSurface, SurfaceViolation, replay
from embodied.bench.referee import truth_store_path
from embodied.contracts.records import (
    ClaimKind,
    SelectionGeometry,
    ClockStamp,
    ExecutionDisposition,
    ExecutionStatus,
    FinalReport,
    MissionContract,
    Observation,
    ReportClaim,
    SensorMode,
    SpatialGoal,
    VisualSelection,
    to_dict,
)
from embodied.perception import camera as camera_module
from embodied.perception.detector import DetectorUnavailable
from embodied.pilot import mission as mission_module

REPOSITORY = Path(__file__).resolve().parents[2]
REAL_PLATFORM_CONFIG = REPOSITORY / "configs" / "first_indoor.yaml"
FIXTURE_SUITE_NAME = "first-indoor-fixture"
HOST, CLOCK = "p05-integration-0", "monotonic"


@pytest.fixture(autouse=True)
def _healthy_host(monkeypatch):
    """Keep the host gate out of these tests, without losing it.

    The transport reads the real machine through ``uptime`` and ``sysctl`` and
    refuses a loaded one. These tests prove the episode chain, not the machine
    the suite happens to run on — and a full suite run loads the host enough to
    trip the gate inside itself, which made four of them fail while passing in
    isolation. The gate's logic is still exercised with declared states in
    ``test_a_loaded_host_is_blocked_before_anything_starts``, and its wiring
    into the run in ``test_record_refuses_a_loaded_host``.
    """
    monkeypatch.setattr(
        live_record,
        "host_state",
        lambda: {"load_1m": 1.0, "swap_free_mb": 4000.0},
    )


def _stamp(ns: int) -> ClockStamp:
    return ClockStamp(host_id=HOST, clock_id=CLOCK, monotonic_ns=ns)


# ---------------------------------------------------------------------------
# A fixture repository root carrying a suite declaration, a seed and a scene
# ---------------------------------------------------------------------------

WORLD_TEXT = textwrap.dedent(
    """\
    #VRML_SIM R2025a utf8
    WorldInfo { basicTimeStep 2 }
    DEF VEHICLE Iris {
      translation 0.0 0.0 0.09
      rotation 0 0 1 0
    }
    """
)


def _write_fixture_root(
    root: Path,
    *,
    suite_name: str | None = None,
    localization_mode: str = "sensor-derived",
    target_present: bool = True,
) -> dict:
    suite_name = suite_name or f"{FIXTURE_SUITE_NAME}-{abs(hash(str(root))) % 100000}"
    (root / "configs" / "suites").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "scenarios" / "first_indoor").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REAL_PLATFORM_CONFIG, root / "configs" / "first_indoor.yaml")
    world = root / "scenarios" / "first_indoor" / "world.wbt"
    world.write_text(WORLD_TEXT, encoding="utf-8")
    suite = {
        "name": suite_name,
        "registered_by": "P05",
        "localization_mode": localization_mode,
        "provider_budget": {"spend_ceiling_usd": 5.0},
        "scene_root": "scenarios/first_indoor",
        "world": "scenarios/first_indoor/world.wbt",
        "mission_instruction": "Find the red block, inspect it, and return to the start.",
        "truth_seed": "scenarios/first_indoor/truth.yaml",
    }
    suite_path = root / "configs" / "suites" / "first-indoor.yaml"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    seed = {
        "targets": {"red_block": {"present": target_present}},
        "world_counts": {"red_block": 1 if target_present else 0},
        "identity": {
            "red_block": {
                "node": "target_block",
                "position_ned_from_world_origin_m": [4.0, 0.0, -0.9],
                "position_enu_m": [4.0, 0.0, 0.9],
            },
            "decoy_box": {
                "node": "decoy_box",
                "position_ned_from_world_origin_m": [2.0, 1.0, -0.15],
                "in_truth": False,
            },
        },
    }
    (root / "scenarios" / "first_indoor" / "truth.yaml").write_text(
        yaml.safe_dump(seed, sort_keys=False), encoding="utf-8"
    )
    return {"suite": suite, "seed": seed, "world": world, "suite_path": suite_path}


# ---------------------------------------------------------------------------
# The suite declaration and registration
# ---------------------------------------------------------------------------


def test_the_suite_registers_only_from_its_own_declaration(tmp_path):
    """A build without the declaration registers nothing and refuses the suite.

    The registry is process-global, and the real record path legitimately fills
    it: registering the shipped declaration is that path's act, not an import
    side effect. So this asserts the invariant the test is named for — a missing
    declaration loads nothing and a None document registers nothing, leaving the
    registry exactly as it was — rather than the absence of a suite an earlier
    test in the same process may have registered through the real path.
    """
    before = dict(live_record.recorder_module.SUITE_REGISTRY)
    assert live_record.load_suite_document(tmp_path / "missing.yaml") is None
    assert live_record.register_suite(None) is None
    assert dict(live_record.recorder_module.SUITE_REGISTRY) == before


def test_a_loaded_host_is_blocked_before_anything_starts():
    assert live_record.host_blockers({"load_1m": 1.2, "swap_free_mb": 4000.0}) == []
    loaded = live_record.host_blockers({"load_1m": 23.7, "swap_free_mb": 400.0})
    assert any("load" in reason for reason in loaded)
    assert any("swap" in reason for reason in loaded)
    unmeasured = live_record.host_blockers({"load_1m": None, "swap_free_mb": None})
    assert len(unmeasured) == 2


def test_record_refuses_a_loaded_host(tmp_path, monkeypatch):
    """The gate stops a run, so it is wired in and not merely a function."""
    monkeypatch.setattr(
        live_record,
        "host_state",
        lambda: {"load_1m": 23.7, "swap_free_mb": 400.0},
    )
    outcome, output = _record_scripted(tmp_path)
    assert outcome.status.value == "blocked"
    assert any("load" in reason for reason in outcome.reasons)
    assert any("freeze-blocked" in limitation for limitation in outcome.limitations)
    assert not (output / "episode").exists()


def test_registration_is_idempotent_and_a_foreign_claim_is_refused(tmp_path):
    fixture = _write_fixture_root(tmp_path / "repo")
    document = live_record.load_suite_document(fixture["suite_path"])
    name = fixture["suite"]["name"]
    assert name.startswith(FIXTURE_SUITE_NAME)
    assert live_record.register_suite(document) == name
    assert live_record.register_suite(document) == name  # idempotent
    conflicting = dict(document, registered_by="P99")
    with pytest.raises(live_record.SuiteConfigError):
        live_record.register_suite(conflicting)


# ---------------------------------------------------------------------------
# bench record's admission rules
# ---------------------------------------------------------------------------


def _run_record(tmp_path, monkeypatch, suite_path: Path, suite_name: str, *extra, mode="sensor-derived"):
    from embodied.bench import cli as bench_cli  # registers the bench command
    from embodied.cli import main

    monkeypatch.setattr(live_record, "SUITE_CONFIG_RELATIVE", suite_path)
    output = tmp_path / "run"
    code = main(
        [
            "bench", "record",
            "--suite", suite_name,
            "--sensor-mode", mode,
            "--arm", extra[0] if extra else "B0",
            "--output", str(output),
        ]
    )
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    return code, receipt


def test_a_cloud_arm_is_blocked_before_the_host_gate(tmp_path, monkeypatch):
    fixture = _write_fixture_root(tmp_path / "repo")
    code, receipt = _run_record(
        tmp_path, monkeypatch, fixture["suite_path"], fixture["suite"]["name"], "B2"
    )
    assert code == 2
    assert receipt["status"] == "blocked"
    assert "B2" in receipt["reasons"][0]
    assert "P06" in receipt["reasons"][0]


def test_a_sensor_mode_the_suite_does_not_declare_is_blocked(tmp_path, monkeypatch):
    fixture = _write_fixture_root(
        tmp_path / "repo", localization_mode="pose-assisted"
    )
    code, receipt = _run_record(
        tmp_path, monkeypatch, fixture["suite_path"], fixture["suite"]["name"]
    )
    assert code == 2
    assert receipt["status"] == "blocked"
    assert "pose-assisted" in receipt["reasons"][0]


# ---------------------------------------------------------------------------
# The transport's episode assembly, with a scripted mission driver
# ---------------------------------------------------------------------------


class _FakePose:
    def __init__(self, xyz):
        self.position_xyz = xyz


class _FakeTruthRecord:
    """The shape the truth tap sees: a record carrying a pose and a sim time."""

    def __init__(self, sim_time_s, xyz):
        self.sim_time_s = sim_time_s
        self.pose = _FakePose(xyz)
        self.kind = "pose"


def _scripted_driver(*, recorder, sensor_tap, episode_id, instruction, target_id, **kwargs):
    """A mission-shaped episode without a simulator.

    It writes the same surfaces a live run writes — a mission event,
    observations, a goal and its status, a setpoint, an execution and a final
    report — so the transport's assembly, the grader and the truth isolation
    are exercised over exactly the live shapes.
    """
    from embodied.pilot.mission import ClaimEvidence, assemble_mission_claims

    recorder.record(
        "mission",
        to_dict(
            MissionContract(
                mission_id=episode_id,
                instruction=instruction,
                interpreted_requirements=("find the red block",),
                revision=0,
                evidence_obligations=("cite the observation",),
                return_obligation="return to the start position and land",
                allowed_scope="the scene",
                budget=(("mission_sim_s", 300.0),),
                unresolved_questions=(),
            )
        ),
        _stamp(1_000_000_000),
    )
    observation = Observation(
        episode_id=episode_id,
        record_id=f"{episode_id}-obs-00001",
        sensor_ids=_sensor_ids(),
        sequence=1,
        capture_stamp=_stamp(1_050_000_000),
        receipt_stamp=_stamp(1_060_000_000),
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
    recorder.record("observation", to_dict(observation), _stamp(1_060_000_000))
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
                description="red block candidate",
                confidence=None,
            )
        ),
        _stamp(1_070_000_000),
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
                reasons=("settled at the inspection standoff",),
                horizon_s=2.0,
                capabilities=("brake", "hold"),
            )
        ),
        _stamp(1_400_000_000),
    )
    # The truth stream, bench-side only: a hold inside the declared radius of
    # the target and a return to the start.
    for index, (sim_s, xyz) in enumerate(
        [
            (1.0, (0.0, 0.0, -0.1)),
            (2.0, (3.8, 0.1, -0.9)),
            (3.0, (4.0, 0.0, -0.9)),
            (4.0, (4.1, -0.1, -0.9)),
            (5.0, (0.2, 0.1, -0.1)),
        ]
    ):
        sensor_tap(_FakeTruthRecord(sim_s, xyz))
    result = _ScriptedResult()
    report = assemble_mission_claims(
        found=ClaimEvidence(True, (observation.record_id,)),
        inspected=ClaimEvidence(True, (observation.record_id,)),
        returned=ClaimEvidence(True, (observation.record_id,)),
        target_id=target_id,
        termination_reason="mission_completed",
        mission_revision=0,
        now=_stamp(1_500_000_000),
    )
    recorder.write_report(report, _stamp(1_500_000_000))
    return result, report


def _sensor_ids():
    from embodied.contracts.records import SensorIds

    return SensorIds(left="camera left", right="camera right", imu="inertial unit")


class _ScriptedResult:
    flew = True
    termination_reason = "mission_completed"
    blockers: list = []
    phases: list = []
    log: list = []
    publications = 3
    publish_refusals = 0
    end_state = {"armed": False, "mode": "LAND", "local_position_ned": [0.2, 0.1, -0.1]}
    crash_statustexts: list = []
    guidance_events: list = []


def _record_scripted(tmp_path, *, target_present: bool = True, adjudicate: bool = False):
    fixture = _write_fixture_root(tmp_path / "repo", target_present=target_present)
    output = tmp_path / "run"
    outcome = live_record.record(
        suite_name=fixture["suite"]["name"],
        suite_document=fixture["suite"],
        arm="B0",
        sensor_mode=SensorMode.SENSOR_DERIVED,
        output=output,
        root=REPOSITORY,
        platform_config=REAL_PLATFORM_CONFIG,
        truth_seed=tmp_path / "repo" / "scenarios" / "first_indoor" / "truth.yaml",
        suite_config=fixture["suite_path"],
        mission_driver=_scripted_driver,
    )
    return outcome, output


def test_the_transport_closes_a_real_episode_and_the_grader_scores_it(tmp_path):
    outcome, output = _record_scripted(tmp_path)
    assert outcome.status.value == "complete"
    episode = output / "episode"
    assert (episode / "manifest.json").is_file()
    assert (episode / "agent-events.jsonl").is_file()
    assert (episode / "final-report.json").is_file()
    assert (episode / "payloads").is_dir()
    store = truth_store_path(episode)
    assert store.is_dir() and (store / "truth-events.jsonl").is_file()
    # The store is outside the episode and the episode is not inside the store.
    assert episode not in store.resolve().parents
    manifest = json.loads((episode / "manifest.json").read_text())
    assert manifest["episode_kind"] == "physical-run"
    assert manifest["suite"].startswith(FIXTURE_SUITE_NAME)
    assert manifest["arm"] == "B0"
    assert manifest["sensor_mode"] == "sensor-derived"
    assert manifest["model_identity"] is None
    # The bench-side stream carries exactly the enclosed shapes.
    truth_lines = [
        json.loads(line)
        for line in (store / "truth-events.jsonl").read_text().splitlines()
    ]
    assert [line["kind"] for line in truth_lines] == ["world_state", "physical_outcome"]
    assert truth_lines[0]["payload"]["targets"] == {"red_block": {"present": True}}
    assert truth_lines[1]["payload"]["inspected"] == {"red_block": True}
    assert truth_lines[1]["payload"]["return_verified"] is True
    # Grading: every claim is world-correct, and support is pending until a
    # reviewer annotates it (never a pass by default).
    score = grade(episode)
    assert score.status == "pending"
    assert [verdict.world_correct for verdict in score.claims] == [True, True, True]
    assert [verdict.support for verdict in score.claims] == ["pending"] * 3
    assert score.fully_passed is False
    assert score.missed_present_targets == ()
    assert score.mission["physical_return_verified"] is True
    assert (episode / SCORE_FILENAME).is_file()


def test_a_seed_with_no_present_target_is_refused_rather_than_searched(tmp_path):
    """A scenario with nothing to find is not this mission's suite.

    The transport needs the target's true position to measure the inspected
    predicate, and a scenario with no present target would make that
    measurement meaningless; refusing names the disagreement instead of
    recording a run whose outcome cannot be read.
    """
    outcome, _ = _record_scripted(tmp_path, target_present=False)
    assert outcome.status.value == "blocked"
    assert any("present targets" in reason for reason in outcome.reasons)


def test_a_supported_annotation_turns_the_same_episode_into_a_pass(tmp_path):
    outcome, output = _record_scripted(tmp_path)
    episode = output / "episode"
    report = json.loads((episode / "final-report.json").read_text())
    adjudication = {
        "adjudication_version": "p02-review-1",
        "rubric_revision": "p02-rubric-1",
        "reviewer": "integration-test",
        "entries": [
            {
                "claim_index": index,
                "cited_evidence": list(claim["support_refs"]),
                "verdict": "supported",
                "note": "the cited observation was taken at the standoff",
            }
            for index, claim in enumerate(report["claims"])
        ],
    }
    (episode / ADJUDICATION_FILENAME).write_text(
        json.dumps(adjudication, indent=2) + "\n", encoding="utf-8"
    )
    score = grade(episode)
    assert score.status == "complete"
    assert score.fully_passed is True
    assert score.mission["task_completion"] is True
    assert score.mission["safe_task_completion"] is True
    assert score.evidence_support["support_precision"] == 1.0


def test_the_agent_projection_cannot_reach_the_bench_side_store(tmp_path):
    _, output = _record_scripted(tmp_path)
    episode = output / "episode"
    surface = AgentSurface.open(episode)
    # The live episode's own members are what the projection declares.
    from embodied.bench import recorder as recorder_module

    assert surface.manifest.permitted_projection == recorder_module.PROJECTION
    assert surface.final_report() is not None
    # The store's members, by name and by traversal, are refused.
    with pytest.raises(SurfaceViolation):
        surface.read_member("truth-events.jsonl")
    with pytest.raises(SurfaceViolation):
        surface.read_member(f"../{episode.name}.truth/truth-events.jsonl")
    with pytest.raises(SurfaceViolation):
        surface.read_member(str(truth_store_path(episode) / "truth-events.jsonl"))
    # Nothing in the episode directory is a bench-private member.
    assert not (episode / "truth-events.jsonl").exists()
    members = {entry.name for entry in episode.iterdir()}
    assert members == {"manifest.json", "agent-events.jsonl", "final-report.json", "payloads"}
    # And replay reconstructs the agent stream without opening the store.
    document = replay(episode)
    assert document["event_count"] == len(surface.agent_events())


def test_the_runtime_never_reads_a_truth_pose():
    """The runtime's own source: pose records are dispatched out, not read in."""
    source = (REPOSITORY / "src" / "embodied" / "platform" / "mission_runtime.py").read_text()
    # The estimator's input is stereo and inertial only: the client is written
    # in exactly two places, and neither is a truth branch.
    assert source.count("client.send(") == 2
    # Every mention of a truth pose's fields lives in the POSE accounting
    # branch, whose only sinks are the feed statistics the bring-up's declared
    # end-state check reads.
    pose_lines = [line.strip() for line in source.splitlines() if "record.pose" in line]
    assert len(pose_lines) == 3, pose_lines
    assert pose_lines[0].startswith("elif record.kind is Kind.POSE"), pose_lines
    # The two values go into the feed statistics and nowhere else.
    assert source.count("_stats.truth_samples.append") == 1
    assert source.count("_stats.truth_attitudes.append") == 1
    for line in pose_lines:
        for forbidden in ("self.store", "publish", "G.ground", "self._sink", "admit"):
            assert forbidden not in line, f"a truth pose reaches {forbidden}: {line}"
    # And every record still reaches the bench side's tap.
    assert "self._sensor_tap(record)" in source


# ---------------------------------------------------------------------------
# The mission policy's own pieces
# ---------------------------------------------------------------------------


def test_the_instruction_is_interpreted_from_its_own_words():
    interpreted = mission_module.interpret_instruction(
        "Find the red block, inspect it, and return to the start."
    )
    assert interpreted.target_phrase == "red block"
    assert interpreted.colour == "red"
    assert interpreted.shape == "block"
    assert interpreted.requirements == (
        "find the red block",
        "inspect the red block",
        "return to the start position",
    )
    contract = mission_module.mission_contract(
        mission_id="m1",
        instruction="Find the red block, inspect it, and return to the start.",
        budget=(("mission_sim_s", 300.0),),
    )
    assert contract.return_obligation == "return to the start position and land"
    with pytest.raises(ValueError):
        mission_module.mission_contract(
            mission_id="m1", instruction="do something",budget=(("mission_sim_s", 300.0),)
        )


def test_a_claim_states_its_outcome_and_cites_its_evidence():
    report = mission_module.assemble_mission_claims(
        found=mission_module.ClaimEvidence(True, ("obs-1",)),
        inspected=mission_module.ClaimEvidence(False),
        returned=mission_module.ClaimEvidence(True, ("obs-2",)),
        target_id="red_block",
        termination_reason="mission_completed",
        mission_revision=0,
        now=_stamp(1),
    )
    by_predicate = {claim.predicate: claim for claim in report.claims}
    assert by_predicate["found"].observed == "found"
    assert by_predicate["found"].kind is ClaimKind.OBSERVATION
    assert by_predicate["found"].support_refs == ("obs-1",)
    assert by_predicate["inspected"].observed == "not_inspected"
    assert by_predicate["inspected"].kind is ClaimKind.INFERENCE
    assert by_predicate["inspected"].unmet_requirements == ("requirement_inspected_unmet",)
    assert report.unmet_requirements == ("requirement_inspected_unmet",)
    assert report.physical_return_status == "returned"
    assert all(claim.target == "red_block" for claim in report.claims)


def test_the_proposer_finds_the_declared_colour_and_shape_only():
    proposer = mission_module.ConventionalColourProposer()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (120, 120, 120)  # grey room light: saturation zero
    image[40:100, 60:140] = (200, 40, 40)  # the red block
    image[150:220, 200:270] = (40, 180, 60)  # a green object
    found = proposer.candidates(image, "red block")
    assert len(found) == 1
    region = found[0].region
    assert region == pytest.approx((60.0, 40.0, 140.0, 100.0))
    assert found[0].provenance.source == mission_module.PROPOSER_SOURCE
    assert found[0].provenance.checkpoint_hash is None
    # A different colour query finds the other object and not the red one.
    green = proposer.candidates(image, "green sphere")
    assert len(green) == 1
    assert green[0].region == pytest.approx((200.0, 150.0, 270.0, 220.0))
    # A query with no colour word refuses rather than guessing.
    assert isinstance(proposer.candidates(image, "the thing on the table"), DetectorUnavailable)
    # Determinism: the same frame proposes the same candidates.
    assert [c.candidate_id for c in proposer.candidates(image, "red block")] == [
        candidate.candidate_id for candidate in found
    ]


def test_the_proposer_rejects_a_shape_the_query_does_not_name():
    proposer = mission_module.ConventionalColourProposer()
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # A red sliver: aspect ratio far from a block, so the block filter drops it.
    image[90:110, 10:190] = (200, 30, 30)
    assert proposer.candidates(image, "red block") == ()
    assert len(proposer.candidates(image, "red region")) == 1


# ---------------------------------------------------------------------------
# The runtime's offline surface
# ---------------------------------------------------------------------------


def _runtime(tmp_path):
    from embodied.platform.localization_check import _load_localization_config, _platform_settings
    from embodied.platform.mission_runtime import MissionRuntime
    from embodied.bench.recorder import Recorder

    document = _load_localization_config(REAL_PLATFORM_CONFIG)
    settings = _platform_settings(document, REPOSITORY)
    episode_dir = tmp_path / "episode"
    recorder = Recorder(episode_dir)
    runtime = MissionRuntime(
        settings=settings,
        config_document=document,
        episode_dir=episode_dir,
        evidence_dir=tmp_path / "platform",
        recorder=recorder,
        instruction="Find the red block, inspect it, and return to the start.",
        target_id="red_block",
        episode_id="offline-1",
    )
    return runtime, recorder


def test_the_runtime_refuses_to_publish_without_an_admitted_goal(tmp_path):
    runtime, _ = _runtime(tmp_path)
    assert runtime.publish_active() == "no_active_goal"


def test_frontiers_cluster_and_the_start_target_is_grounded(tmp_path):
    from embodied.contracts.records import PoseEstimate
    from embodied.perception.camera import DepthProduct, PoseProvenance

    runtime, _ = _runtime(tmp_path)
    # The mission frame is the aligned frame, so nothing can resolve before the
    # alignment seals from the estimator's first healthy state.
    assert runtime.resolve_targets(
        SpatialGoal(
            proposal_id="p-unsealed", request_id=None, fingerprint="fp-u",
            mission_revision=0, base_goal_revision=0, selection_ids=(),
            target_refs=("start",), intent="return", constraints=(),
            completion_condition="c", lease_bounds=(("step_lease_s", 30.0),),
            local_discretion_bounds=(),
        )
    ) == ()
    runtime.alignment.seal((1.0, 0.0, 0.0, 0.0))
    calibration = runtime.calibration
    height, width = 480, 640
    valid = np.zeros((height, width), dtype=bool)
    valid[200:280, 280:360] = True
    depth_m = np.full((height, width), 2.0, dtype=np.float32)
    product = DepthProduct(
        calibration_id=calibration.calibration_id,
        calibration_version=calibration.version,
        pair_id="pair-1",
        capture_stamp=_stamp(1),
        receipt_stamp=_stamp(2),
        sim_time_s=0.5,
        pose_provenance=PoseProvenance(label="SENSOR_DERIVED", detail="test"),
        frame="camera_optical",
        disparity_px=np.full((height, width), 50.0, dtype=np.float32),
        depth_m=depth_m,
        valid=valid,
        reasons=np.zeros((height, width), dtype=np.uint8),
        uncertainty_m=np.full((height, width), 0.02, dtype=np.float32),
    )
    pose = PoseEstimate(
        parent_frame="odom",
        child_frame="body",
        stamp=_stamp(3),
        position_m=(0.0, 0.0, -1.5),  # NED: 1.5 m above the spawn
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance=None,
        nav_epoch=runtime.nav_epoch,
        source_ids=("ov_stream",),
        valid=True,
    )
    # The map's evidence stamps and its freshness clock are both the controller
    # capture clock, so the test drives them together exactly as a pair would.
    runtime._capture_clock_ns = 500_000_000
    runtime.store.integrate(
        product, pose, calibration, stamp_ns=500_000_000, observation_id="obs-1", now_ns=500_000_000
    )
    assert runtime.store.free_cells(now_ns=500_000_000)
    regions = runtime.frontier_regions()
    assert regions, "an observed free cone beside unknown space must expose a frontier"
    assert all(name.startswith("frontier:") for name in regions)
    # Without a recorded observation there is nothing to cite, so a map-derived
    # target is refused rather than fabricated.
    assert runtime.resolve_targets(
        SpatialGoal(
            proposal_id="p-none", request_id=None, fingerprint="fp0", mission_revision=0,
            base_goal_revision=0, selection_ids=(), target_refs=("start",), intent="return",
            constraints=(), completion_condition="c", lease_bounds=(("step_lease_s", 30.0),),
            local_discretion_bounds=(),
        )
    ) == ()
    runtime._last_observation_id = "obs-1"
    goal = SpatialGoal(
        proposal_id="p-start",
        request_id=None,
        fingerprint="fp",
        mission_revision=0,
        base_goal_revision=0,
        selection_ids=(),
        target_refs=("start",),
        intent="return",
        constraints=(),
        completion_condition="settled at the start position",
        lease_bounds=(("step_lease_s", 30.0),),
        local_discretion_bounds=(),
    )
    targets = runtime.resolve_targets(goal)
    assert len(targets) == 1
    assert targets[0].target_id == "start"
    # The declared stand-off compensation puts the terminal region on the start.
    from embodied.navigation import geometry as GE
    from embodied.platform.mission_runtime import _terminal_region_for

    region = _terminal_region_for(targets)
    # The mission frame is world-anchored local NED, so the spawn sits at the
    # world's own vehicle translation and the hover band above it.
    origin = runtime.alignment.aligned_position_ned((0.0, 0.0, 0.0))
    assert region.contains((origin[0], origin[1], origin[2] - runtime.settings.hover_altitude_m))
    assert region == GE.approach_region(
        targets[0].geometry, GE.Envelope(body_radius_m=0.3, error_allowance_m=0.15),
        direction=(1.0, 0.0, 0.0),
    )
    # A frontier ref resolves to a map-evidenced target at the current revision.
    frontier_ref = sorted(regions)[0]
    frontier_goal = SpatialGoal(
        proposal_id="p-frontier",
        request_id=None,
        fingerprint="fp2",
        mission_revision=0,
        base_goal_revision=0,
        selection_ids=(),
        target_refs=(frontier_ref,),
        intent="explore",
        constraints=(),
        completion_condition="settled at the frontier",
        lease_bounds=(("step_lease_s", 30.0),),
        local_discretion_bounds=(),
    )
    frontier_targets = runtime.resolve_targets(frontier_goal)
    assert len(frontier_targets) == 1
    assert frontier_targets[0].anchor_revision == runtime.store.revision


def test_the_seed_reads_the_suites_world_state_layout(tmp_path):
    """The suite nests its world state under ``world_state``; both shapes work."""
    nested = {
        "schema": "first-indoor-truth-1",
        "world_state": {"targets": {"red_block": {"present": True}}, "world_counts": {"red_block": 1}},
        "identity": {"red_block": {"position_ned_from_world_origin_m": [8.8, 1.6, -0.9]}},
    }
    seed_path = tmp_path / "truth.yaml"
    seed_path.write_text(yaml.safe_dump(nested, sort_keys=False), encoding="utf-8")
    seed = live_record.load_truth_seed(seed_path)
    assert live_record.world_state_payload(seed) == {
        "targets": {"red_block": {"present": True}},
        "world_counts": {"red_block": 1},
    }
    assert live_record.target_position_ned(seed, "red_block") == pytest.approx((8.8, 1.6, -0.9))


def test_the_seed_must_declare_the_target_position(tmp_path):
    seed = {"targets": {"red_block": {"present": True}}, "world_counts": {"red_block": 1}}
    with pytest.raises(live_record.SuiteConfigError):
        live_record.target_position_ned(seed, "red_block")
    good = dict(seed, identity={"red_block": {"position_enu_m": [4.0, 0.0, 0.9]}})
    assert live_record.target_position_ned(good, "red_block") == pytest.approx((4.0, -0.0, -0.9))


def test_the_bench_side_envelope_refuses_a_target_entry_with_extra_keys(tmp_path):
    from embodied.bench.referee import Referee

    episode = tmp_path / "episode"
    episode.mkdir()
    referee = Referee(episode)
    with pytest.raises(Exception):
        referee.record(
            "world_state",
            {
                "targets": {"red_block": {"present": True, "position": [1, 2, 3]}},
                "world_counts": {"red_block": 1},
            },
            _stamp(1),
        )


def test_physical_predicates_need_a_sustained_hold_and_a_return():
    collector = live_record.TruthCollector()
    for sim_s, xyz in [
        (0.0, (0.0, 0.0, -0.1)),
        (0.5, (4.0, 0.0, -0.9)),   # a single pass, not a hold
        (1.0, (9.0, 9.0, -0.9)),
        (2.0, (0.1, 0.0, -0.1)),
    ]:
        collector(_FakeTruthRecord(sim_s, xyz))
    inspected, detail = collector.inspected_within(
        (4.0, 0.0, -0.9), radius_m=1.5, hold_s=1.0
    )
    assert inspected is False and "interval" in detail
    returned, _ = collector.returned_near(radius_m=1.0)
    assert returned is True
    sustained = live_record.TruthCollector()
    for sim_s in (0.0, 0.5, 1.0, 1.5):
        sustained(_FakeTruthRecord(sim_s, (4.0, 0.2, -0.9)))
    inspected, detail = sustained.inspected_within((4.0, 0.0, -0.9), radius_m=1.5, hold_s=1.0)
    assert inspected is True
    assert "held within" in detail
