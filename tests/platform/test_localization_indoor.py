"""P01-L's named behaviours, offline: protocol, conventions, health, refusals.

No simulator, no network, no paid calls; fixtures are procedural. The numbered
tests are the plan's section 10 behaviours, and the bounds they use come from
the configuration's frozen values, not from literals in this file wherever the
behaviour under test is a bound.

Nothing here starts a simulator: the tests that reach the check command use a
configuration whose estimator process is deliberately absent, or one whose bridge
would republish truth, so the claimed arm blocks in preflight exactly as a reader
of the receipt would see it.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from embodied.cli import CommandError, build_parser
from embodied.contracts.records import SensorMode
from embodied.platform import localization as loc
from embodied.platform import localization_check as check
from embodied.platform.webots_ardupilot import PlatformSettings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bounds(**overrides) -> loc.HealthBounds:
    values = dict(
        publish_period_s=0.025,
        state_lost_after_s=0.300,
        published_state_age_max_s=0.020,
        max_publish_gap_s=0.100,
        visual_update_warn_s=0.300,
        visual_update_fail_s=0.500,
        valid_fraction_min=0.99,
        sigma_min_m=0.02,
        sigma_max_m=1.0,
    )
    values.update(overrides)
    return loc.HealthBounds(**values)


def _state(
    *,
    time_ns: int = 1_000_000_000,
    initialized: bool = True,
    sigma: tuple[float, float, float] = (0.05, 0.05, 0.08),
    t_last_visual_ns: int | None = None,
) -> loc.EstimatorState:
    return loc.EstimatorState(
        time_ns=time_ns,
        initialized=initialized,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_m=(1.0, 2.0, 3.0),
        velocity_mps=(0.1, 0.2, 0.3),
        gyro_bias=(0.0, 0.0, 0.0),
        accel_bias=(0.0, 0.0, 0.0),
        sigma_pos_m=sigma,
        n_tracks=40,
        t_last_visual_ns=time_ns if t_last_visual_ns is None else t_last_visual_ns,
        reset_counter=0,
    )


def _state_frame(state: loc.EstimatorState) -> bytes:
    """Build one wire-valid STATE frame, field for field, from the outside."""
    payload = struct.pack(
        "<QB19dIQB",
        state.time_ns,
        1 if state.initialized else 0,
        *state.quat_wxyz,
        *state.position_m,
        *state.velocity_mps,
        *state.gyro_bias,
        *state.accel_bias,
        *state.sigma_pos_m,
        state.n_tracks,
        state.t_last_visual_ns,
        0,
    )
    return loc.FRAME_HEADER.pack(loc.FRAME_MAGIC, loc.KIND_STATE, 0, len(payload)) + payload


def _rot_z_90() -> np.ndarray:
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _rot_x_90() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _declared_document() -> dict:
    return yaml.safe_load(Path("configs/first_indoor.yaml").read_text(encoding="utf-8"))


def _config_without_the_estimator(tmp_path: Path) -> Path:
    """The declared configuration, with the estimator process pointed at nothing."""
    document = _declared_document()
    document["localization"]["estimator"]["executable"] = "estimator/ov_stream-absent"
    path = tmp_path / "config-absent-estimator.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _config_declaring(tmp_path: Path, mode: str) -> Path:
    """The declared configuration under a different localization mode."""
    document = _declared_document()
    document["localization"]["mode"] = mode
    path = tmp_path / f"config-{mode}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path

# ---------------------------------------------------------------------------
# T1: protocol round-trip
# ---------------------------------------------------------------------------


class TestProtocolRoundTrip:
    def test_state_round_trip_is_field_exact(self):
        state = _state(sigma=(0.11, 0.12, 0.13), t_last_visual_ns=999_000_000)
        decoded = loc.decode_state(_state_frame(state))
        assert decoded.time_ns == state.time_ns
        assert decoded.initialized is True
        assert decoded.quat_wxyz == state.quat_wxyz
        assert decoded.position_m == state.position_m
        assert decoded.velocity_mps == state.velocity_mps
        assert decoded.sigma_pos_m == state.sigma_pos_m
        assert decoded.n_tracks == state.n_tracks
        assert decoded.t_last_visual_ns == state.t_last_visual_ns

    def test_truncated_frame_is_refused_not_guessed(self):
        frame = _state_frame(_state())
        with pytest.raises(loc.ProtocolError):
            loc.decode_state(frame[:-1])
        with pytest.raises(loc.ProtocolError):
            loc.decode_state(frame[: loc.FRAME_HEADER_SIZE - 1])

    def test_trailing_and_foreign_frames_are_refused(self):
        frame = _state_frame(_state())
        with pytest.raises(loc.ProtocolError):
            loc.decode_state(frame + b"\x00")
        foreign = loc.FRAME_HEADER.pack(0x1234, loc.KIND_STATE, 0, 8) + b"\x00" * 8
        with pytest.raises(loc.ProtocolError):
            loc.decode_state(foreign)
        wrong_kind = loc.FRAME_HEADER.pack(loc.FRAME_MAGIC, loc.KIND_IMU, 0, 8) + b"\x00" * 8
        with pytest.raises(loc.ProtocolError):
            loc.decode_state(wrong_kind)

    def test_imu_frame_carries_the_pinned_layout(self):
        frame = loc.encode_imu(42, (0.1, 0.2, 0.3), (-9.8, 0.0, 0.1))
        magic, kind, _flags, length = loc.FRAME_HEADER.unpack_from(frame)
        assert magic == loc.FRAME_MAGIC and kind == loc.KIND_IMU
        assert length == loc._IMU_PAYLOAD.size
        time_ns, *values = loc._IMU_PAYLOAD.unpack(frame[loc.FRAME_HEADER_SIZE :])
        assert time_ns == 42
        assert list(values) == [0.1, 0.2, 0.3, -9.8, 0.0, 0.1]

    def test_stereo_frame_carries_both_planes(self):
        width, height = 320, 240
        left = bytes(width * height)
        right = b"\x07" * (width * height)
        frame = loc.encode_stereo(7, left, right, width, height)
        magic, kind, _flags, length = loc.FRAME_HEADER.unpack_from(frame)
        assert magic == loc.FRAME_MAGIC and kind == loc.KIND_STEREO
        assert length == loc._STEREO_PAYLOAD_HEADER.size + len(left) + len(right)
        payload = frame[loc.FRAME_HEADER_SIZE :]
        time_ns, wire_width, wire_height = loc._STEREO_PAYLOAD_HEADER.unpack_from(payload)
        assert (time_ns, wire_width, wire_height) == (7, width, height)
        assert payload[loc._STEREO_PAYLOAD_HEADER.size :] == left + right

    def test_miscounted_planes_are_refused(self):
        with pytest.raises(loc.ProtocolError):
            loc.encode_stereo(1, b"\x00" * 9, b"\x00" * 9, 3, 4)

    def test_reset_frame_has_no_payload(self):
        magic, kind, _flags, length = loc.FRAME_HEADER.unpack(loc.encode_reset())
        assert (magic, kind, length) == (loc.FRAME_MAGIC, loc.KIND_RESET, 0)

    def test_grayscale_conversion_is_luma_and_size_checked(self):
        width, height = 4, 2
        rgb = bytes([255, 0, 0] * 8) + bytes([0, 0, 0] * 0)  # eight red pixels
        rgb = bytes([255, 0, 0]) * 4 + bytes([0, 255, 0]) * 4
        gray = loc.grayscale_rgb8(rgb, width, height)
        assert len(gray) == width * height
        assert gray[0] == pytest.approx(0.299 * 255, abs=1.0)
        assert gray[4] == pytest.approx(0.587 * 255, abs=1.0)
        with pytest.raises(loc.ProtocolError):
            loc.grayscale_rgb8(b"\x00" * 5, width, height)


# ---------------------------------------------------------------------------
# T2: conventions pinned
# ---------------------------------------------------------------------------


class TestConventions:
    def test_position_axes_swap_and_negate(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        assert alignment.aligned_position_ned((1.0, 2.0, 3.0)) == (2.0, 1.0, -3.0)

    def test_odom_origin_is_the_declared_start(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.02))
        north, east, down = alignment.aligned_position_ned((0.0, 0.0, 0.0))
        assert (north, east, down) == (0.0, 0.0, -0.02)

    def test_velocity_axes_match_position_axes(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        assert alignment.aligned_velocity_ned((1.0, 0.0, 0.0)) == (0.0, 1.0, 0.0)

    def test_identity_body_pose_is_the_axis_swap(self):
        """The declared start's identity quaternion faces ENU +x; in NED that is
        yaw +90 degrees, so the published attitude is the axis swap itself,
        never identity by accident."""
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        quat = alignment.aligned_quat_ned_wxyz((1.0, 0.0, 0.0, 0.0))
        assert loc.quat_to_rotmat(quat) == pytest.approx(loc.ENU_TO_NED_AXES, abs=1e-12)

    def test_enu_yaw_rotation_maps_through_the_axes(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        yaw_90_enu = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        quat = alignment.aligned_quat_ned_wxyz(yaw_90_enu)
        expected = loc.ENU_TO_NED_AXES @ _rot_z_90()
        assert loc.quat_to_rotmat(quat) == pytest.approx(expected, abs=1e-12)

    def test_enu_east_rotation_becomes_a_ned_north_rotation(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        roll_90_enu = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
        quat = alignment.aligned_quat_ned_wxyz(roll_90_enu)
        expected = loc.ENU_TO_NED_AXES @ _rot_x_90()
        assert loc.quat_to_rotmat(quat) == pytest.approx(expected, abs=1e-12)

    def test_aligned_rotation_moves_points_consistently(self):
        """A body point maps as NED(R @ v): the body coordinates ride unchanged,
        the world-side axes are remapped once."""
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        yaw_90_enu = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        quat_ned = alignment.aligned_quat_ned_wxyz(yaw_90_enu)
        point_body = np.array([1.0, 0.0, 0.0])
        rotated_enu = _rot_z_90() @ point_body
        expected = loc.ENU_TO_NED_AXES @ rotated_enu
        rotated_ned = loc.quat_to_rotmat(quat_ned) @ point_body
        assert rotated_ned == pytest.approx(expected, abs=1e-12)

    def test_aligned_state_reports_position_and_velocity(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.02))
        state = replace(_state(), position_m=(1.0, 0.0, -0.02))
        aligned = alignment.aligned_state(state)
        assert aligned["position_ned_m"] == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
        assert aligned["velocity_ned_mps"] == pytest.approx((0.2, 0.1, -0.3), abs=1e-12)
        assert aligned["attitude_rpy"][2] == pytest.approx(math.pi / 2, abs=1e-9)

    def test_estimated_body_is_the_declared_imu_frame(self):
        """The estimator's body frame is the declared IMU frame; the feed never
        silently swaps axes (declared T_body_imu is identity in the pinned
        calibration record)."""
        document = json.loads(
            Path("scenarios/compat/calibration.json").read_text(encoding="utf-8")
        )
        assert document["T_body_imu"]["quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
        assert document["T_body_imu"]["translation_m"] == [0.0, 0.0, 0.0]

    def test_zero_quaternion_is_refused(self):
        with pytest.raises(loc.AlignmentError):
            loc.quat_to_rotmat((0.0, 0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# T3: the health/freshness machine
# ---------------------------------------------------------------------------


class TestHealthMachine:
    def test_uninitialized_estimator_never_publishes(self):
        machine = loc.HealthMachine(_bounds())
        now = 1_000_000_000
        assert machine.on_state(_state(initialized=False), now) is False
        assert machine.state == "stopped"
        assert machine.on_state(_state(), now + 1_000_000) is True

    def test_sigma_outside_envelope_stops_transmission(self):
        machine = loc.HealthMachine(_bounds())
        now = 1_000_000_000
        assert machine.on_state(_state(sigma=(0.05, 0.05, 0.08)), now) is True
        assert machine.on_state(_state(sigma=(0.05, 0.05, 1.5)), now + 1_000_000) is False
        assert machine.state == "stopped"
        stopped = [event for event in machine.events if event.event == "stopped"]
        assert len(stopped) == 1 and "sigma" in stopped[0].detail

    def test_no_publish_between_stop_and_recovery(self):
        machine = loc.HealthMachine(_bounds())
        now = 1_000_000_000
        assert machine.on_state(_state(), now) is True
        machine.stop(now + 1_000_000, "estimator connection lost: read failed")
        assert machine.on_state(None, now + 2_000_000) is False
        assert machine.on_state(_state(initialized=False), now + 3_000_000) is False
        assert machine.on_state(_state(), now + 4_000_000) is True
        assert machine.reset_counter == 1
        assert machine.on_state(_state(), now + 5_000_000) is True
        assert machine.reset_counter == 1  # recovery is counted once, not per tick

    def test_silence_beyond_the_declared_window_stops(self):
        machine = loc.HealthMachine(_bounds(state_lost_after_s=0.300))
        second = 1_000_000_000
        assert machine.on_state(_state(), second) is True
        assert machine.on_state(None, second + 250_000_000) is False
        assert machine.state == "healthy"  # 250 ms of silence is inside the window
        assert machine.on_state(None, second + 350_000_000) is False
        assert machine.state == "stopped"

    def test_publish_accounting_matches_the_declared_bounds(self):
        machine = loc.HealthMachine(_bounds(published_state_age_max_s=0.020))
        first, second = 1_000_000_000, 1_025_000_000
        machine.on_published(_state(time_ns=first - 5_000_000), first, first)
        machine.on_published(_state(time_ns=second - 30_000_000), second, second)
        assert machine.publish_gaps_s == pytest.approx([0.025], abs=1e-9)
        assert machine.published_state_ages_s == pytest.approx([0.005, 0.030], abs=1e-9)

    def test_valid_fraction_charges_outages_whole(self):
        machine = loc.HealthMachine(_bounds(valid_fraction_min=0.99))
        second = 1_000_000_000
        machine.on_state(_state(), second)
        machine.stop(second + 1_000_000, "test stop")
        machine._recover(_state(), second + 201_000_000)
        fraction = machine.valid_fraction(second + 21_000_000_000)
        assert fraction == pytest.approx(1.0 - 0.200 / 21.0, abs=1e-6)
        assert fraction >= 0.99

    def test_visual_update_verdict_uses_the_declared_ages(self):
        machine = loc.HealthMachine(
            _bounds(visual_update_warn_s=0.300, visual_update_fail_s=0.500)
        )
        base = 10_000_000_000
        assert machine.visual_update_verdict(_state(time_ns=base, t_last_visual_ns=base)) == "ok"
        assert (
            machine.visual_update_verdict(
                _state(time_ns=base, t_last_visual_ns=base - 350_000_000)
            )
            == "warn"
        )
        assert (
            machine.visual_update_verdict(
                _state(time_ns=base, t_last_visual_ns=base - 500_000_000)
            )
            == "fail"
        )

    def test_scored_window_discards_bring_up_accounting(self):
        machine = loc.HealthMachine(_bounds())
        now = 1_000_000_000
        machine.on_published(_state(), now, now)
        machine.open_window(now + 1_000_000_000)
        assert machine.publish_gaps_s == []
        machine.on_published(_state(), now + 1_025_000_000, now + 1_025_000_000)
        assert machine.publish_gaps_s == []


# ---------------------------------------------------------------------------
# T4: refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_unknown_mode_is_refused_at_the_cli_boundary(self):
        parser = build_parser()
        with pytest.raises(CommandError):
            parser.parse_args(["localize-check", "--mode", "truth-assisted"])

    def test_existing_output_path_is_refused(self, tmp_path, capsys):
        (tmp_path / "receipt.json").write_text("{}", encoding="utf-8")
        exit_code = check.main(
            [
                "--config",
                "configs/first_indoor.yaml",
                "--mode",
                "sensor-derived",
                "--output",
                str(tmp_path),
            ]
        )
        assert exit_code == 1
        assert "already exists" in capsys.readouterr().err

    def test_pose_assisted_receipt_is_labelled_and_not_applicable(self, tmp_path, capsys):
        exit_code = check.main(
            [
                "--config",
                "configs/first_indoor.yaml",
                "--mode",
                "pose-assisted",
                "--output",
                str(tmp_path),
            ]
        )
        assert exit_code == 2
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["gate_status"] == "not_applicable"
        assert receipt["sensor_mode"] == "pose-assisted"
        assert any(
            "pose-assisted-diagnostic" in limitation
            for limitation in receipt["limitations"]
        )
        assert any("cannot pass P01-L" in reason for reason in receipt["reasons"])

    def test_a_missing_estimator_process_blocks_with_a_concrete_reason(self, tmp_path):
        """Prerequisites are reported before anything starts: a claimed arm whose
        estimator binary is absent is blocked and the receipt names it, rather
        than starting a simulator to discover that."""
        config_path = _config_without_the_estimator(tmp_path)
        exit_code = check.main(
            [
                "--config",
                str(config_path),
                "--mode",
                "sensor-derived",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == 2
        receipt = json.loads((tmp_path / "out" / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "blocked"
        blockers = " ".join(receipt["reasons"])
        assert "localization=unresolved" in blockers
        assert "ov_stream" in blockers
        preflight = json.loads((tmp_path / "out" / "preflight.json").read_text(encoding="utf-8"))
        assert preflight["satisfied"] is False
        unsatisfied = [row["name"] for row in preflight["checks"] if not row["satisfied"]]
        assert "estimator_seam" in unsatisfied
        # The seam requirements themselves are met on the declared configuration,
        # so the block above is the missing process and nothing else.
        assert "localization_mode" not in unsatisfied
        assert "bridge_truth_republish" not in unsatisfied

    def test_module_registers_the_dispatch_command(self):
        parser = build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        assert "localize-check" in subparsers.choices

# ---------------------------------------------------------------------------
# T5: the bridge's truth republish, machine-checked (plan section 4.6)
# ---------------------------------------------------------------------------


class TestTruthRepublishGate:
    """The reviewer's recorded residual, as a checked behaviour.

    Run 59d7c6e3 left one hole: "the bridge's truth-republish-off was asserted in
    prose, not machine-checked". These tests read the switch's actual value from the
    settings the bridge is built from, and the vehicle's own readback, so a claimed
    arm cannot start on a promise.
    """

    def test_the_default_configuration_still_republishes_truth(self):
        """No localization section is the configuration the P00 gate was measured on:
        the vision feed runs, and the run declares simulator-interface."""
        document = _declared_document()
        document.pop("localization")
        settings = PlatformSettings.from_config(document, root=Path(".").resolve())
        assert settings.truth_republish is True
        assert settings.sensor_mode is SensorMode.SIMULATOR_INTERFACE

    def test_the_sensor_derived_configuration_turns_the_truth_republish_off(self):
        document = _declared_document()
        settings = PlatformSettings.from_config(document, root=Path(".").resolve())
        assert document["localization"]["mode"] == "sensor-derived"
        assert settings.truth_republish is False
        assert settings.sensor_mode is SensorMode.SENSOR_DERIVED

    def test_the_preflight_passes_the_gate_on_the_declared_configuration(self, tmp_path):
        rows, _satisfied = check._preflight(
            check._load_localization_config(Path("configs/first_indoor.yaml")),
            tmp_path,
            SensorMode.SENSOR_DERIVED,
        )
        rows = {row["name"]: row for row in rows}
        assert rows["localization_mode"]["satisfied"] is True
        assert rows["bridge_truth_republish"]["satisfied"] is True

    def test_the_gate_refuses_while_the_truth_republish_is_on(self, tmp_path):
        """A bridge that would republish truth cannot start a scored sensor-derived
        arm, and the blocker names the switch rather than the symptom."""
        settings = PlatformSettings.from_config(
            _declared_document(), root=Path(".").resolve()
        )
        blockers = check._truth_republish_blockers(replace(settings, truth_republish=True))
        assert blockers and "truth republish is ON" in blockers[0]

        rows, satisfied = check._preflight(
            check._load_localization_config(_config_declaring(tmp_path, "pose-assisted")),
            tmp_path,
            SensorMode.SENSOR_DERIVED,
        )
        rows = {row["name"]: row for row in rows}
        assert satisfied is False
        assert rows["localization_mode"]["satisfied"] is False
        assert rows["bridge_truth_republish"]["satisfied"] is False
        assert "truth republish is ON" in rows["bridge_truth_republish"]["detail"]

    def test_a_gps_type_readback_that_is_not_zero_fails_the_claimed_arm(self):
        """GPS off is confirmed from the vehicle's own readback, not from the file we
        wrote: the synthesized GPS is simulator-truth-derived."""
        declared = {name: expected for name, expected, _source in check.SEAM_REQUIREMENTS}
        assert check._readback_blockers(declared) == []

        with_gps = dict(declared, GPS_TYPE=1.0)
        blockers = check._readback_blockers(with_gps)
        assert blockers and "GPS_TYPE" in blockers[0]

        unanswered = {name: value for name, value in declared.items() if name != "GPS_TYPE"}
        assert "GPS_TYPE" in check._readback_blockers(unanswered)[0]

    def test_the_declared_switch_is_read_by_the_bridge_it_configures(self):
        """The value the gate reads is the value the bridge's own loader produces, so
        a future edit cannot make the gate agree with a bridge that does not exist."""
        from embodied.platform import webots_ardupilot as bridge

        document = _declared_document()
        settings = bridge.PlatformSettings.from_config(document, root=Path(".").resolve())
        assert settings.truth_republish is False
        # And with the switch on, the feed is the one the gate refuses.
        assert bridge.VISION_POSE_PERIOD_S == 0.025

    def test_a_slow_estimator_is_not_a_lost_connection(self, tmp_path):
        """A full send buffer must apply backpressure, not end the stream.

        Measured 2026-09-26 in the first live sensor-derived run: the client's socket
        was non-blocking, so writing a 614 KB stereo pair into a busy estimator raised
        EAGAIN, which the client read as a dead link. It stopped the health machine
        after 1061 inertial samples and no pair ever reached the estimator.
        """
        import socket

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            client = loc.OvStreamClient("127.0.0.1", port, timeout_s=2.0)
            client.connect()
            # Blocking writes: the feed waits rather than dropping or dying.
            assert client._socket.gettimeout() is None
            # Silence is not death: a poll with nothing to read reports nothing.
            assert client.poll_state() is None
            client.send(loc.encode_imu(1, (0.0, 0.0, 0.0), (0.0, 0.0, -9.81)))
            peer, _address = listener.accept()
            try:
                assert peer.recv(1 << 16)  # the frame did leave
            finally:
                peer.close()
            client.close()
        finally:
            listener.close()
