"""Behaviour of the shared records, one test per rule rather than per field.

The rules being pinned here are the ones a later stage can get wrong quietly:
requirements with no defaults, stamps compared only inside one clock domain, closed
vocabularies, references that a record may not be built without, and a JSON form
stable enough to hash.
"""

import dataclasses
import json
import os
import subprocess
import sys

import pytest

from embodied.contracts import records as R


def stamp(seconds: float, *, host: str = "host-1", clock: str = "monotonic") -> R.ClockStamp:
    return R.ClockStamp(host_id=host, clock_id=clock, monotonic_ns=int(seconds * 1e9))


def calibration() -> R.Calibration:
    intrinsics = R.CameraIntrinsics(focal_length_px=(554.0, 554.0), principal_point_px=(320.0, 240.0))
    distortion = R.Distortion(model="none_declared", coefficients=(0.0,))
    identity = (1.0, 0.0, 0.0, 0.0)
    return R.Calibration(
        calibration_id="cal-1",
        version="1",
        left_intrinsics=intrinsics,
        right_intrinsics=intrinsics,
        left_distortion=distortion,
        right_distortion=distortion,
        T_camera_left_camera_right=R.Transform(
            parent_frame="camera_left",
            child_frame="camera_right",
            translation_m=(0.0, -0.1, 0.0),
            quaternion_wxyz=identity,
        ),
        T_body_camera_left=R.Transform(
            parent_frame="body",
            child_frame="camera_left",
            translation_m=(0.05, 0.05, 0.05),
            quaternion_wxyz=identity,
        ),
        T_body_imu=R.Transform(
            parent_frame="body",
            child_frame="imu",
            translation_m=(0.0, 0.0, 0.0),
            quaternion_wxyz=identity,
        ),
        baseline_m=0.1,
        pixel_convention="top-left origin, [column, row]",
        depth_convention="none in this stage",
        time_offset_s=None,
        time_offset_error_s=None,
        validated_limits=None,
        source="declared for the fixture",
    )


def quality(is_colour: bool = True) -> R.FrameQuality:
    return R.FrameQuality(
        is_colour=is_colour,
        channel_identical_fraction=0.01 if is_colour else 1.0,
        channel_means=(120.0, 90.0, 60.0),
        saturated_pixel_fraction=0.02,
    )


def observation() -> R.Observation:
    return R.Observation(
        episode_id="ep-1",
        record_id="rec-1",
        sensor_ids=R.SensorIds(left="camera left", right="camera right", imu="inertial unit"),
        sequence=3,
        capture_stamp=stamp(10.0),
        receipt_stamp=stamp(10.2),
        sim_time_s=1.25,
        pair_id="pair-3",
        left_payload="pairs/00003-left.ppm",
        right_payload="pairs/00003-right.ppm",
        encoding="rgb8",
        width=640,
        height=480,
        calibration_id="cal-1",
        capture_pose_ref=None,
        quality=quality(),
        depth_source=None,
    )


def pose() -> R.PoseEstimate:
    return R.PoseEstimate(
        parent_frame="odom",
        child_frame="body",
        stamp=stamp(11.0),
        position_m=(1.0, 2.0, -1.5),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance=None,
        nav_epoch="nav-1",
        source_ids=("stereo-inertial",),
        valid=True,
    )


def navigation_state() -> R.NavigationState:
    return R.NavigationState(
        state_sequence=7,
        pose=pose(),
        velocity_mps=(0.1, 0.0, 0.0),
        covariance=None,
        nav_epoch="nav-1",
        visual_source_ids=("camera left", "camera right"),
        imu_source_ids=("inertial unit",),
        status="tracking",
        controller_alignment_id="align-1",
    )


def selection() -> R.VisualSelection:
    return R.VisualSelection(
        selection_id="sel-1",
        observation_id="rec-1",
        coordinate_convention="[column, row] from the top-left of the stored frame",
        geometry_kind=R.SelectionGeometry.POINT,
        geometry=(320.0, 240.0),
        crop_transform=None,
        description="the object the instruction named",
        confidence=None,
    )


def grounded_target() -> R.GroundedTarget:
    return R.GroundedTarget(
        target_id="tgt-1",
        track_id="track-1",
        place_id=None,
        selection_ids=("sel-1",),
        observation_ids=("rec-1",),
        geometry=(3.0, 0.0, -1.0),
        frame=R.Frame.ODOM,
        anchor_id="anchor-1",
        anchor_revision="1",
        uncertainty=(0.02, 0.02, 0.05),
        identity_alternatives=None,
        last_observed_stamp=stamp(11.0),
        valid=True,
    )


def world_snapshot() -> R.WorldSnapshot:
    return R.WorldSnapshot(
        snapshot_id="snap-1",
        map_revision="rev-4",
        anchor_transforms=(calibration().T_body_imu,),
        occupancy_ref=None,
        moving_envelopes=None,
        targets=("tgt-1",),
        places=(),
        data_ages=(("occupancy", 0.4), ("pose", 0.02)),
        navigation_state_ref="nav-state-7",
    )


def mission_contract() -> R.MissionContract:
    return R.MissionContract(
        mission_id="mission-1",
        instruction="find the red object and report where it is",
        interpreted_requirements=("locate the named object", "return to the start"),
        revision=1,
        evidence_obligations=("one observation of the object",),
        return_obligation="return to the start position",
        allowed_scope="the ground floor",
        budget=(("wall_clock_s", 600.0),),
        unresolved_questions=(),
    )


def decision_request() -> R.DecisionRequest:
    return R.DecisionRequest(
        request_id="req-1",
        sequence=0,
        mission_revision=1,
        base_goal_revision=0,
        observation_ids=("rec-1",),
        snapshot_id="snap-1",
        response_deadline_s=4.0,
        model_identity="provider/model@revision",
    )


def spatial_goal() -> R.SpatialGoal:
    return R.SpatialGoal(
        proposal_id="proposal-1",
        request_id="req-1",
        fingerprint="sha256:0a1b",
        mission_revision=1,
        base_goal_revision=0,
        selection_ids=("sel-1",),
        target_refs=("tgt-1",),
        intent="move to the observed object",
        constraints=("stay below 2 m",),
        completion_condition="within 1 m of the target",
        lease_bounds=(("duration_s", 30.0),),
        local_discretion_bounds=(("path_m", 5.0),),
    )


def goal_status() -> R.GoalStatus:
    return R.GoalStatus(
        proposal_id="proposal-1",
        request_id="req-1",
        disposition=R.GoalDisposition.ACCEPTED,
        reason=None,
        admission_ref="goal-1",
        current_disposition=R.ExecutionDisposition.RUNNING,
    )


def motion_setpoint() -> R.MotionSetpoint:
    target = R.MotionTarget(
        position_ned=(2.0, 0.0, -1.5),
        velocity_ned=(0.0, 0.0, 0.0),
        acceleration_ned=None,
        yaw_rad=None,
        yaw_rate_rad_s=None,
    )
    return R.MotionSetpoint(
        command_sequence=5,
        mission_revision=1,
        goal_revision=1,
        nav_epoch="nav-1",
        frame=R.Frame.ODOM,
        type_mask=R.TYPE_MASK_POSITION_VELOCITY,
        target=target,
        issue_stamp=stamp(12.0),
        deadline_s=1.0,
        certificate_ref="cert-1",
        sample_ref=None,
        source=R.SetpointSource.NORMAL,
    )


def execution_status() -> R.ExecutionStatus:
    return R.ExecutionStatus(
        goal_ref="goal-1",
        certificate_ref="cert-1",
        command_ref="5",
        disposition=R.ExecutionDisposition.RUNNING,
        evidence=("LOCAL_POSITION_NED moved 0.4 m toward the target",),
        reasons=(),
        horizon_s=2.0,
        capabilities=("position-hold",),
    )


def report_claim() -> R.ReportClaim:
    return R.ReportClaim(
        predicate="arrived within 1 m of the target",
        target="tgt-1",
        observed=0.6,
        support_refs=("rec-1", "nav-state-7"),
        kind=R.ClaimKind.OBSERVATION,
        stamp=stamp(13.0),
        uncertainty=0.3,
        unmet_requirements=None,
    )


def final_report() -> R.FinalReport:
    return R.FinalReport(
        mission_revision=1,
        claims=(report_claim(),),
        termination_reason="completion claimed: target inspected and start position regained",
        unmet_requirements=(),
        physical_return_status="at the start position within 0.5 m",
        evidence_snapshot_ids=("snap-1",),
    )


COMPLETE_EXAMPLES = {
    "Calibration": calibration,
    "Observation": observation,
    "PoseEstimate": pose,
    "NavigationState": navigation_state,
    "VisualSelection": selection,
    "GroundedTarget": grounded_target,
    "WorldSnapshot": world_snapshot,
    "MissionContract": mission_contract,
    "DecisionRequest": decision_request,
    "SpatialGoal": spatial_goal,
    "GoalStatus": goal_status,
    "MotionSetpoint": motion_setpoint,
    "ExecutionStatus": execution_status,
    "ReportClaim": report_claim,
    "FinalReport": final_report,
}
ALL_RECORD_NAMES = sorted(COMPLETE_EXAMPLES) + ["ClockStamp"]


def test_every_implemented_record_builds_and_round_trips():
    for name, builder in COMPLETE_EXAMPLES.items():
        record = builder()
        document = R.to_dict(record)
        assert document == json.loads(json.dumps(document)), name
        assert R.from_dict(name, document) == record, name
    clock = stamp(1.0)
    assert R.from_dict("ClockStamp", R.to_dict(clock)) == clock


def test_implemented_and_deferred_records_cover_section_22():
    covered = set(R.IMPLEMENTED_RECORDS) | set(R.DEFERRED_RECORDS)
    assert covered == set(R.SPECIFICATION_RECORDS)
    assert len(R.IMPLEMENTED_RECORDS) == len(set(R.IMPLEMENTED_RECORDS))
    for name in R.IMPLEMENTED_RECORDS:
        assert hasattr(R, name), name
    for name in R.DEFERRED_RECORDS:
        assert not hasattr(R, name), f"{name} is listed as deferred but is defined"


def test_omitting_a_required_field_is_a_type_error():
    with pytest.raises(TypeError):
        R.Observation(
            episode_id="ep-1",
            record_id="rec-1",
            sensor_ids=R.SensorIds(left="a", right="b", imu="c"),
            sequence=0,
        )


def test_missing_information_stays_null_through_a_round_trip():
    record = dataclasses.replace(observation(), capture_pose_ref=None, depth_source=None, sim_time_s=None)
    document = R.to_dict(record)
    assert document["capture_pose_ref"] is None
    assert document["depth_source"] is None
    assert document["sim_time_s"] is None
    restored = R.from_dict("Observation", document)
    assert restored.capture_pose_ref is None
    assert restored.depth_source is None
    assert restored.sim_time_s is None


def test_no_field_may_be_omitted_from_any_record():
    """Every field of every record is required: omitting one is refused, so no field
    can fall back to a default value."""
    for name, builder in COMPLETE_EXAMPLES.items():
        record = builder()
        present = {
            field.name: getattr(record, field.name) for field in dataclasses.fields(record)
        }
        for omitted in present:
            partial = {key: value for key, value in present.items() if key != omitted}
            with pytest.raises(TypeError):
                type(record)(**partial)


def test_stamps_are_compared_only_inside_one_clock_domain():
    earlier, later = stamp(10.0), stamp(10.5)
    assert R.elapsed_ns(earlier, later) == 500_000_000
    assert R.same_clock(earlier, later)
    with pytest.raises(R.ClockDomainError):
        R.elapsed_ns(earlier, stamp(10.5, clock="simulation"))
    with pytest.raises(R.ClockDomainError):
        R.elapsed_ns(earlier, stamp(10.5, host="other-host"))
    # A negative monotonic reading is not a stamp at all, and is refused earlier.
    with pytest.raises(R.RecordError):
        R.ClockStamp(host_id="h", clock_id="c", monotonic_ns=-1)


def test_observation_rejects_an_impossible_capture_order():
    with pytest.raises(R.RecordError):
        dataclasses.replace(observation(), receipt_stamp=stamp(9.0))


def test_observation_rejects_a_half_pair_and_zero_dimensions():
    with pytest.raises(R.RecordError):
        dataclasses.replace(observation(), right_payload=None)
    with pytest.raises(R.RecordError):
        dataclasses.replace(observation(), pair_id=None)
    with pytest.raises(R.RecordError):
        dataclasses.replace(observation(), width=0)
    with pytest.raises(R.RecordError):
        dataclasses.replace(observation(), height=-4)


def test_motion_setpoint_rejects_an_empty_mask_an_unknown_frame_and_a_contradiction():
    with pytest.raises(R.RecordError):
        dataclasses.replace(motion_setpoint(), type_mask=0)
    with pytest.raises(R.RecordError):
        dataclasses.replace(motion_setpoint(), frame="world")
    with pytest.raises(R.RecordError):
        dataclasses.replace(motion_setpoint(), frame=R.Frame.MAP)
    # The mask ignores velocity while the target carries one.
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            motion_setpoint(),
            type_mask=R.TYPE_MASK_ALL & ~R.TYPE_MASK_VELOCITY_IGNORE,
        )
    # The mask says heading is used while the target carries none.
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            motion_setpoint(),
            type_mask=R.TYPE_MASK_POSITION_VELOCITY & ~R.TYPE_MASK_YAW_IGNORE,
        )
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            motion_setpoint(),
            target=R.MotionTarget(
                position_ned=(1.0, 0.0, -1.0),
                velocity_ned=None,
                acceleration_ned=None,
                yaw_rad=None,
                yaw_rate_rad_s=None,
            ),
        )


def test_type_mask_names_complete_field_groups_and_never_a_force_target():
    """The mask is read in groups by the autopilot, so only whole groups are accepted.

    A bit that ignores one position axis is read as ignoring the position target
    entirely, which is not what a mask with one axis set appears to say. The force bit
    means the opposite of the others — it claims a force target — and this system never
    sends one.
    """
    # 4 is z_ignore alone: part of the position group, and so not a mask at all.
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            motion_setpoint(), type_mask=(R.TYPE_MASK_POSITION_VELOCITY | 4)
        )
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            motion_setpoint(), type_mask=(R.TYPE_MASK_POSITION_VELOCITY | 8)
        )
    assert R.TYPE_MASK_POSITION_VELOCITY & ~R.TYPE_MASK_ALL == 0
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            motion_setpoint(),
            type_mask=(R.TYPE_MASK_POSITION_VELOCITY | R.TYPE_MASK_FORCE_SET),
        )


def test_calibration_rejects_a_zero_baseline_and_mismatched_transforms():
    with pytest.raises(R.RecordError):
        dataclasses.replace(calibration(), baseline_m=0.0)
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            calibration(),
            T_body_camera_left=R.Transform(
                parent_frame="camera_left",
                child_frame="body",
                translation_m=(0.0, 0.0, 0.0),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        )
    with pytest.raises(R.RecordError):
        dataclasses.replace(calibration(), time_offset_s=0.01)
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            calibration(),
            T_body_imu=R.Transform(
                parent_frame="body",
                child_frame="imu",
                translation_m=(0.0, 0.0, 0.0),
                quaternion_wxyz=(2.0, 0.0, 0.0, 0.0),
            ),
        )


def test_references_a_record_cannot_be_built_without():
    with pytest.raises(R.RecordError):
        dataclasses.replace(pose(), nav_epoch=None)
    with pytest.raises(R.RecordError):
        dataclasses.replace(observation(), calibration_id=None)
    with pytest.raises(R.RecordError):
        dataclasses.replace(grounded_target(), anchor_revision=None)
    with pytest.raises(R.RecordError):
        dataclasses.replace(world_snapshot(), map_revision=None)
    with pytest.raises(R.RecordError):
        dataclasses.replace(motion_setpoint(), nav_epoch=None)
    with pytest.raises(R.RecordError):
        dataclasses.replace(navigation_state(), nav_epoch=None)


def test_closed_vocabularies_reject_unknown_values_and_serialize_by_value():
    with pytest.raises(R.RecordError):
        dataclasses.replace(goal_status(), disposition="maybe")
    with pytest.raises(R.RecordError):
        dataclasses.replace(motion_setpoint(), source="preferred")
    with pytest.raises(R.RecordError):
        dataclasses.replace(report_claim(), kind="assumed")
    document = R.to_dict(goal_status())
    assert document["disposition"] == "accepted"
    assert document["current_disposition"] == "running"
    with pytest.raises(R.RecordError):
        R.from_dict("GoalStatus", {**document, "disposition": "maybe"})


def test_claims_and_completions_need_their_evidence():
    with pytest.raises(R.RecordError):
        dataclasses.replace(report_claim(), support_refs=())
    with pytest.raises(R.RecordError):
        dataclasses.replace(
            execution_status(), disposition=R.ExecutionDisposition.COMPLETED, evidence=()
        )
    inferred = dataclasses.replace(report_claim(), kind=R.ClaimKind.INFERENCE, support_refs=())
    assert R.to_dict(inferred)["kind"] == "inference"


def test_selection_geometry_matches_its_kind():
    with pytest.raises(R.RecordError):
        dataclasses.replace(selection(), geometry_kind=R.SelectionGeometry.BOX)
    with pytest.raises(R.RecordError):
        dataclasses.replace(selection(), confidence=2.0)
    masked = dataclasses.replace(
        selection(), geometry_kind=R.SelectionGeometry.MASK, geometry="masks/00001.png"
    )
    assert R.to_dict(masked)["geometry"] == "masks/00001.png"


def test_goals_are_bounded_and_unbounded_ones_are_rejected():
    with pytest.raises(R.RecordError):
        dataclasses.replace(spatial_goal(), lease_bounds=())
    with pytest.raises(R.RecordError):
        dataclasses.replace(spatial_goal(), lease_bounds=(("duration_s", 0.0),))
    with pytest.raises(R.RecordError):
        dataclasses.replace(spatial_goal(), selection_ids=(), target_refs=())
    with pytest.raises(R.RecordError):
        dataclasses.replace(mission_contract(), budget=())


def test_json_output_is_byte_stable_and_keys_are_sorted():
    # The expected bytes are written out, not derived by encoding twice in one
    # process: the encoder is pure, so re-running it here would certify itself.
    known = json.dumps(R.to_dict(stamp(1.0)), sort_keys=False)
    assert known == (
        '{"clock_id": "monotonic", "host_id": "host-1", "monotonic_ns": 1000000000}'
    )
    document = R.to_dict(observation())
    assert list(document) == sorted(document)
    assert list(document["capture_stamp"]) == sorted(document["capture_stamp"])
    assert list(document["sensor_ids"]) == sorted(document["sensor_ids"])


def test_from_dict_rejects_unknown_and_missing_keys():
    document = R.to_dict(observation())
    with pytest.raises(R.RecordError):
        R.from_dict("Observation", {**document, "surprise": 1})
    without_width = {key: value for key, value in document.items() if key != "width"}
    with pytest.raises(R.RecordError):
        R.from_dict("Observation", without_width)
    with pytest.raises(R.RecordError):
        R.from_dict("NotARecord", document)


def test_records_are_frozen_and_hashable():
    record = observation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.width = 10
    assert hash(record) == hash(observation())


def test_the_module_imports_without_the_runtime_stack(tmp_path):
    """Later stages must be able to load records without numpy, YAML or MAVLink.

    The check runs in a subprocess whose interpreter is the one that ran this test,
    with the three modules blocked on its import path. Nothing is put on PYTHONPATH
    except the blocker itself: the package is the installed one, the same code the
    tests and the command use. The blocker is imported by name rather than shipped
    as ``sitecustomize.py``: a ``sitecustomize`` on PYTHONPATH shadows the
    interpreter's own, and Homebrew's is what adds pip's site-packages (where the
    editable ``embodied`` lives) to ``sys.path``.
    """
    blocker = tmp_path / "runtime_blocker.py"
    blocker.write_text(
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'numpy', 'yaml', 'pymavlink'}:\n"
        "            raise ImportError(name + ' is blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runtime_blocker;"
            "import sys;"
            "from embodied.contracts import records;"
            "print(records.RECORDS_REVISION, len(records.IMPLEMENTED_RECORDS), sys.executable)",
        ],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(tmp_path),
            "PATH": os.environ.get("PATH", ""),
        },
        cwd=str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    assert R.RECORDS_REVISION in completed.stdout
    assert sys.executable in completed.stdout
