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
import hashlib
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
from embodied.platform import webots_ardupilot as bridge
from embodied.platform import localization_check as check
from embodied.platform.webots_ardupilot import EvidenceWriter as bridge_EvidenceWriter
from embodied.platform.webots_ardupilot import PlatformSettings


class _RecordedMessage:
    """The minimum pymavlink message surface the bring-up link's send path uses."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def get_type(self) -> str:
        return self.kind


class _RecordingMav:
    """A stand-in for pymavlink's ``mav`` object that records what was encoded.

    The bring-up link's wire discipline is part of what this slice has to be able to
    assert -- one channel, every other field ignored, the release a zero on the same
    field -- and the only honest way to assert it is on the values that reach the
    encoder. A live autopilot is not part of a unit test, so the encoder records.
    """

    def __init__(self) -> None:
        self.encoded: list[tuple[str, tuple]] = []
        self.sent: list[Any] = []

    def _message(self, kind: str, arguments: tuple) -> _RecordedMessage:
        self.encoded.append((kind, arguments))
        return _RecordedMessage(kind)

    def rc_channels_override_encode(self, *arguments: Any) -> _RecordedMessage:
        return self._message("RC_CHANNELS_OVERRIDE", arguments)

    def command_long_encode(self, *arguments: Any) -> _RecordedMessage:
        return self._message("COMMAND_LONG", arguments)

    def param_set_encode(self, *arguments: Any) -> _RecordedMessage:
        return self._message("PARAM_SET", arguments)

    def set_gps_global_origin_encode(self, *arguments: Any) -> _RecordedMessage:
        return self._message("SET_GPS_GLOBAL_ORIGIN", arguments)

    def send(self, message: Any) -> None:
        self.sent.append(message)


class _RecordingConnection:
    """A stand-in for a pymavlink connection whose ``mav`` encoder records."""

    def __init__(self) -> None:
        self.mav = _RecordingMav()

    def close(self) -> None:
        return None


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

def _rot_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


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
    def test_the_world_and_body_maps_are_the_pinned_conversions(self):
        """Plan section 0.3 item 2: the bridge's world->NED map (keep x, negate y
        and z) and the estimator-body FLU -> autopilot-body FRD map (a 180-degree
        roll), both pinned against the bridge the gate proved."""
        assert loc.WORLD_TO_NED_AXES == pytest.approx(
            np.diag([1.0, -1.0, -1.0]), abs=1e-12
        )
        assert loc.FLU_TO_FRD_AXES == pytest.approx(
            np.diag([1.0, -1.0, -1.0]), abs=1e-12
        )

    def test_an_unsealed_alignment_refuses_to_convert(self):
        """The odom frame's yaw is unobservable, so an alignment that has not been
        sealed by an initialized state must refuse rather than guess."""
        alignment = loc.OdomAlignment((-1.0, 0.0, 0.09))
        assert alignment.sealed is False
        with pytest.raises(loc.AlignmentError, match="not sealed"):
            alignment.aligned_position_ned((0.0, 0.0, 0.0))
        with pytest.raises(loc.AlignmentError, match="not sealed"):
            alignment.aligned_attitude_rpy((1.0, 0.0, 0.0, 0.0))

    def test_the_seal_makes_every_initial_yaw_publish_the_declared_start(self):
        """The measured defect that produced this design: the pinned initializer's
        gram_schmidt branch is decided by accelerometer noise at a level start, so
        the odom yaw is arbitrary. The seal derives the epoch rotation from the
        estimator's own first initialized attitude, so whatever yaw the estimator
        reports, the published attitude at the declared start is the declared
        start attitude."""
        for odom_yaw_deg in (0.0, 90.0, 180.0, -90.0):
            half = math.radians(odom_yaw_deg) / 2.0
            q_init = (math.cos(half), 0.0, 0.0, math.sin(half))
            alignment = loc.OdomAlignment((-1.0, 0.0, 0.09), (0.0, 0.0, 0.0))
            alignment.seal(q_init)
            assert alignment.aligned_attitude_rpy(q_init) == pytest.approx(
                (0.0, 0.0, 0.0), abs=1e-9
            )
            # The seal's own yaw is the derivation's, recorded for the receipt:
            # the estimator's arbitrary yaw is folded into it, so no run depends on
            # which gram_schmidt branch the initializer's noise happened to take.
            assert abs(alignment.epoch_yaw_deg()) <= 180.0

    def test_the_declared_start_yaw_is_honoured_by_the_seal(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2))
        alignment.seal((1.0, 0.0, 0.0, 0.0))
        assert alignment.aligned_attitude_rpy((1.0, 0.0, 0.0, 0.0)) == pytest.approx(
            (0.0, 0.0, math.pi / 2), abs=1e-9
        )

    def test_a_pitch_stays_a_pitch_under_every_sealed_initial_yaw(self):
        """The measured departure mechanism, kept as a regression: the pinned
        estimator reports its attitude with the initialization frame's yaw
        composed on the right of the rotation since start, so publishing through
        the epoch rotation conjugates the relative rotation by the initializer's
        gram_schmidt yaw and turns a physical pitch into a published roll
        whenever that yaw is near +/-90 degrees (run
        p01l-bringup-20260926T202549Z: SITL pitch -0.117/-0.414/-1.52 rad
        published as vision roll -0.112/-0.439/-1.557 with pitch near zero;
        204753Z sealed epoch yaw 89.62 degrees and departed on the first
        pitch). The published attitude must be the rotation since the sealed
        start conjugated into FRD: a pitch is a pitch for every branch.

        The estimator's reported quaternion is modelled as the wire carries it
        (measured, run p01l-fix3b-20260927T050042Z): ov_stream conjugates the
        pinned state quaternion's JPL vector part and packs it (w, x, y, z), so
        this module's Hamilton read of the delivered numbers is the odom->body
        rotation -- the transpose of the body->odom rotation the alignment
        needs, which it applies itself."""
        pitch_nose_up = 0.30
        for odom_yaw_deg in (0.0, 90.0, 180.0, -90.0):
            half = math.radians(odom_yaw_deg) / 2.0
            q_init = (math.cos(half), 0.0, 0.0, math.sin(half))
            alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
            alignment.seal(q_init)
            # What the pinned estimator delivers after rotating since start:
            # the body(0)-frame rotation composed on the right of its
            # initialization attitude, transposed into the delivered
            # odom->body direction.
            since_start = loc.rotmat_from_rpy((0.0, -pitch_nose_up, 0.0))
            body_to_odom = loc.quat_to_rotmat(q_init).T @ since_start
            delivered = loc.rotmat_to_quat(body_to_odom.T)
            roll, pitch, yaw = alignment.aligned_attitude_rpy(delivered)
            assert roll == pytest.approx(0.0, abs=1e-9), (
                f"odom yaw {odom_yaw_deg} deg: a physical pitch was published as roll"
            )
            assert pitch == pytest.approx(pitch_nose_up, abs=1e-9)
            assert yaw == pytest.approx(0.0, abs=1e-9)

    def test_the_published_attitude_is_the_true_rotation_not_its_inverse(self):
        """The fix3b regression, measured end to end (run
        p01l-fix3b-20260927T050042Z): through the whole flight the published
        attitude_rpy carried the OPPOSITE SIGN on every axis to the dataflash
        SIM truth (published pitch +12.86 deg against truth -13.99 at the
        takeoff; published roll -13.2 against truth +13.25, published yaw -16.3
        against truth +16.4 during the excursion), and transposing the
        published rotation recovers truth to a transport lag. The composition
        must publish the vehicle's true rotation for every gram_schmidt yaw the
        initializer's noise can pick, not its inverse."""
        cases = {
            "roll": (0.22, 0.0, 0.0),
            "pitch": (0.0, -0.24, 0.0),
            "yaw": (0.0, 0.0, 0.28),
            "combined": (0.17, -0.14, 0.52),
        }
        for odom_yaw_deg in (0.0, 89.62, -89.62, 180.0):
            half = math.radians(odom_yaw_deg) / 2.0
            q_init = (math.cos(half), 0.0, 0.0, math.sin(half))
            alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
            alignment.seal(q_init)
            for name, true_rpy in cases.items():
                # The delivered quaternion for a true NED attitude: the frame
                # chain of the pinned rig, transposed into the delivered
                # odom->body direction (see the pitch regression above).
                body_to_odom = (
                    _rot_z(-odom_yaw_deg)
                    @ loc.WORLD_TO_NED_AXES
                    @ loc.rotmat_from_rpy(true_rpy)
                    @ loc.FLU_TO_FRD_AXES
                )
                delivered = loc.rotmat_to_quat(body_to_odom.T)
                published = alignment.aligned_attitude_rpy(delivered)
                assert published == pytest.approx(true_rpy, abs=1e-9), (
                    f"odom yaw {odom_yaw_deg} deg, {name}: published "
                    f"{published} is not the true rotation {true_rpy}"
                )

    def test_a_true_north_displacement_publishes_north_for_every_odom_yaw(self):
        """The seal's epoch rotation must map odom displacements through the
        measured frame chain for every initializer yaw, not only the yaw the
        good runs happened to seal: at a sealed yaw of +/-90 degrees the
        pre-fix epoch rotated true displacements by the double yaw (unexposed
        only because every route-completing run sealed a yaw near zero)."""
        for odom_yaw_deg in (0.0, 90.0, -90.0, 180.0):
            half = math.radians(odom_yaw_deg) / 2.0
            q_init = (math.cos(half), 0.0, 0.0, math.sin(half))
            alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
            alignment.seal(q_init)
            # One true metre north: the odom displacement the estimator would
            # report for it, through the same frame chain.
            true_delta_ned = np.array([1.0, 0.0, 0.0])
            odom_delta = (
                _rot_z(-odom_yaw_deg) @ loc.WORLD_TO_NED_AXES @ true_delta_ned
            )
            published = alignment.aligned_position_ned(odom_delta)
            assert published == pytest.approx((1.0, 0.0, 0.0), abs=1e-9), (
                f"odom yaw {odom_yaw_deg} deg: a true north displacement "
                f"published {published}"
            )
    def test_the_seal_is_idempotent_once_taken(self):
        """The epoch rotation is fixed: a later call must not move it, or drift
        would be absorbed instead of published."""
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        alignment.seal((0.0, 0.0, 0.0, 1.0))
        first = alignment.epoch_rotation.copy()
        alignment.seal((math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)))
        assert alignment.epoch_rotation == pytest.approx(first, abs=1e-12)

    def test_position_maps_origin_and_odom_displacement_separately(self):
        alignment = loc.OdomAlignment((-1.0, 0.0, 0.09), (0.0, 0.0, 0.0))
        alignment.seal((0.0, 0.0, 0.0, 1.0))  # the geometric branch-b case
        # One metre north of the dev-a-single spawn: the odom displacement reads
        # (-1, 0, 0) in that frame, and a perfect estimator's publication equals
        # the truth's world NED.
        assert alignment.aligned_position_ned((-1.0, 0.0, 0.0)) == pytest.approx(
            (0.0, 0.0, -0.09), abs=1e-12
        )

    def test_odom_origin_is_the_declared_start(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.02))
        alignment.seal((math.cos(math.pi / 2), 0.0, 0.0, math.sin(math.pi / 2)))
        north, east, down = alignment.aligned_position_ned((0.0, 0.0, 0.0))
        assert (north, east, down) == (0.0, 0.0, -0.02)

    def test_velocity_maps_through_the_epoch_rotation(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        alignment.seal((0.0, 0.0, 0.0, 1.0))
        assert alignment.aligned_velocity_ned((1.0, 0.0, 0.0)) == (-1.0, 0.0, 0.0)

    def test_the_first_runs_measured_roll_error_is_reproduced(self):
        """The first textured invocation's refusal, kept as a regression: the
        geometric map without the FLU->FRD body term (and without a seal) reports a
        180-degree roll on a level yaw-0 start -- exactly the delta A1 measured
        (receipt p01l-run5-textured-20260926T104945Zb)."""
        odom_yaw_180 = (0.0, 0.0, 0.0, 1.0)
        rotation = loc.quat_to_rotmat(odom_yaw_180)
        geometric = np.diag([-1.0, 1.0, -1.0])
        without_body_map = loc.rotmat_to_rpy(geometric @ rotation)
        with_body_map = loc.rotmat_to_rpy(geometric @ rotation @ loc.FLU_TO_FRD_AXES)
        assert without_body_map[0] == pytest.approx(math.pi, abs=1e-9)
        assert with_body_map == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)

    def test_rotation_matrix_from_rpy_round_trips(self):
        assert loc.rotmat_to_rpy(loc.rotmat_from_rpy((0.1, -0.2, 0.3))) == pytest.approx(
            (0.1, -0.2, 0.3), abs=1e-12
        )

    def test_a_yaw_in_the_odom_frame_composes_through_the_sealed_epoch(self):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        level_yaw_90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        alignment.seal(level_yaw_90)
        quat = alignment.aligned_quat_ned_wxyz(level_yaw_90)
        assert loc.quat_to_rotmat(quat) == pytest.approx(np.eye(3), abs=1e-12)

    def test_aligned_rotation_moves_points_consistently(self):
        """The published rotation maps body-FRD directions into NED exactly as
        the vehicle's true attitude does, for a delivered quaternion of a real
        rotation since the seal -- the odom frame's arbitrary yaw and the body
        convention each cancel once."""
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0))
        half = math.pi / 8  # the seal's gram_schmidt yaw: 45 degrees
        q_init = (math.cos(half), 0.0, 0.0, math.sin(half))
        alignment.seal(q_init)
        true_rpy = (0.0, 0.0, math.pi / 6)  # a true 30-degree yaw since start
        body_to_odom = (
            _rot_z(-45.0)
            @ loc.WORLD_TO_NED_AXES
            @ loc.rotmat_from_rpy(true_rpy)
            @ loc.FLU_TO_FRD_AXES
        )
        delivered = loc.rotmat_to_quat(body_to_odom.T)
        quat_ned = alignment.aligned_quat_ned_wxyz(delivered)
        point_body = np.array([1.0, 0.0, 0.0])
        rotated_ned = loc.quat_to_rotmat(quat_ned) @ point_body
        assert rotated_ned == pytest.approx(
            loc.rotmat_from_rpy(true_rpy) @ point_body, abs=1e-9
        )

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

    def test_sigma_outside_envelope_stops_transmission_in_flight(self):
        """In the declared window the sigma bound stops transmission exactly as declared."""
        machine = loc.HealthMachine(_bounds())
        now = 1_000_000_000
        machine.open_window(now)
        assert machine.on_state(_state(sigma=(0.05, 0.05, 0.08)), now) is True
        assert machine.on_state(_state(sigma=(0.05, 0.05, 1.5)), now + 1_000_000) is False
        assert machine.state == "stopped"
        stopped = [event for event in machine.events if event.event == "stopped"]
        assert len(stopped) == 1 and "sigma" in stopped[0].detail

    def test_a_parked_sigma_excursion_is_recorded_but_publishes(self):
        """The parked phase is not flight: its ZUPT-walk sigma excursions are benign.

        A stationary launch initializes through the pin's zero-velocity walk, which
        corrupts the covariance until a published sigma reads exactly 0.0 while the
        pose stays accurate (FIXER3: 1 mm p95 parked against truth). Stopping
        publication for it starves the firmware's VISO health window and the arm is
        refused with 'VisOdom: not healthy' (FIXER6: three of four launches dead at
        the arm, valid fraction 0.54/0.70 against the 0.99 of healthy runs). The
        excursion is recorded as an event and publication continues; once the window
        is declared, the same excursion stops transmission.
        """
        machine = loc.HealthMachine(_bounds())
        now = 1_000_000_000
        assert machine.on_state(_state(sigma=(0.05, 0.05, 0.08)), now) is True
        # The excursion while parked: recorded once, publication continues.
        assert machine.on_state(_state(sigma=(0.0, 0.0, 0.0)), now + 1_000_000) is True
        assert machine.on_state(_state(sigma=(0.0, 0.0, 0.0)), now + 2_000_000) is True
        assert machine.state == "healthy"
        excursion = [event for event in machine.events if event.event == "parked sigma excursion"]
        assert len(excursion) == 1 and "benign" in excursion[0].detail
        # Cleared, also once.
        assert machine.on_state(_state(sigma=(0.05, 0.05, 0.08)), now + 3_000_000) is True
        cleared = [event for event in machine.events if event.event == "parked sigma excursion cleared"]
        assert len(cleared) == 1
        # The same excursion after the window is declared stops transmission.
        machine.open_window(now + 4_000_000)
        assert machine.on_state(_state(sigma=(0.05, 0.05, 0.08)), now + 4_000_000) is True
        assert machine.on_state(_state(sigma=(0.0, 0.0, 0.0)), now + 5_000_000) is False
        assert machine.state == "stopped"

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

    def test_publications_after_the_window_close_are_not_scored(self):
        """F2 and F3 score the flight, not the harness teardown after it.

        Measured, run p01l-fix5-20260927T163256Z: after ``close_window`` the drain
        loop has exited, no new offer can arrive, and the publisher re-sent the last
        offered state 28 times at a constant 0.214 s age for 316 ms until its thread
        stopped — while the flight's own 1878 publications had aged at most ~0.019 s.
        That teardown tail was 100 % of the run's F2 failure, the same frozen-tail
        pathology b48b44d removed from E1's accounting; H1's valid_fraction already
        ends at the window's close.
        """
        machine = loc.HealthMachine(_bounds())
        base = 1_000_000_000
        machine.open_window(base)
        machine.on_published(_state(time_ns=base - 5_000_000), base, base)
        machine.close_window(base + 25_000_000)
        machine.on_published(_state(time_ns=base - 200_000_000), base + 300_000_000, base)
        assert machine.published_state_ages_s == pytest.approx([0.005], abs=1e-9)
        assert machine.publish_gaps_s == []

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

    def test_closing_the_scored_window_freezes_the_accounting(self):
        """The window is arm to disarm: the harness shutdown is not part of it.

        Measured, run p01l-zupt5-20260927T042005Z: the adapter republished one frozen
        state 332 times over 10.13 s while Webots and SITL were being killed, 32 % of the
        scored window, and the machine's silence stop in that tail would otherwise be
        charged as an outage of a flight that had already ended.
        """
        machine = loc.HealthMachine(_bounds())
        second = 1_000_000_000
        machine.open_window(second)
        machine.on_state(_state(), second)
        machine.stop(second + 10_000_000_000, "test stop inside the window")
        machine._recover(_state(), second + 10_200_000_000)
        machine.close_window(second + 20_000_000_000)
        # H2/H4 is read at the window's close, so the flight's own end is what is scored
        # and the declared silence stop that follows it is not a fault.
        assert machine.state_at_close == "healthy"
        machine.stop(second + 21_000_000_000, "the flight is over: the tail is not scored")
        assert machine.state_at_close == "healthy"
        fraction = machine.valid_fraction(second + 40_000_000_000)
        # The in-window outage is charged whole over the 20 s window; the post-window stop
        # is charged nothing, and the denominator stops growing at the close.
        assert fraction == pytest.approx(1.0 - 0.200 / 20.0, abs=1e-6)

    def test_a_window_that_closes_stopped_is_recorded_as_stopped(self):
        """A machine stopped when the flight ends must still fail H2/H4."""
        machine = loc.HealthMachine(_bounds())
        second = 1_000_000_000
        machine.open_window(second)
        machine.on_state(_state(), second)
        machine.stop(second + 1_000_000_000, "an estimator fault during the flight")
        machine.close_window(second + 2_000_000_000)
        assert machine.state_at_close == "stopped"

    def test_a_silent_feed_stops_transmission_at_the_declared_bound(self):
        """``state_lost_after_ms: 300`` is a declared bound, not prose.

        The machine's silence watchdog is reachable only when the publisher hands it
        None, and the publisher previously always handed it its last state, so a dead
        feed kept a frozen pose on the wire indefinitely (measured: 332 identical
        publications over 10.13 s, run p01l-zupt5-20260927T042005Z).
        """
        machine = loc.HealthMachine(_bounds(state_lost_after_s=0.300))
        now = [100.0]
        publisher = loc.ExternalNavPublisher(
            "tcp:127.0.0.1:5762",
            loc.OdomAlignment((0.0, 0.0, 0.0)),
            machine,
            clock=lambda: now[0],
        )
        publisher.offer(_state(), 1_000_000_000)
        assert publisher.state_for_publish(now[0] + 0.250) is not None
        assert publisher.state_for_publish(now[0] + 0.301) is None
        # A new offer re-arms it: silence, not the passage of time, is what stops it.
        now[0] += 1.0
        publisher.offer(_state(time_ns=2_000_000_000), 2_000_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is not None

    def test_a_frozen_state_clock_stops_transmission_at_the_declared_bound(self):
        """A state whose own clock stops advancing is a stalled feed, not a pose.

        The silence watchdog covers a feed that stops OFFERING; this bound covers the
        worse half of the same defect, measured on the four mid-climb tumbles of
        2026-09-27 (p01l-fix5c/-5d/-7c/-flightscope): ov_stream kept publishing at
        its 100 Hz tick while its state clock was frozen -- the estimator could not
        consume inertial samples behind a burst of stereo bytes on the one shared
        connection, so its published state carried a healthy sigma and a pose frozen
        at the last accepted image while the vehicle climbed. The adapter republished
        that confident frozen pose until the backlog cleared and the pose stepped
        0.4-0.7 m mid-air, and the yaw excursion and roll-yaw flip followed within
        1.5 s. A clock that has not advanced for the declared state_lost_after_s is
        the same "stale estimate" that bound already forbids, so it stops
        transmission the same way: the state is handed over as None, and the
        machine's silence path takes over.
        """
        machine = loc.HealthMachine(_bounds(state_lost_after_s=0.300))
        machine.open_window(1_000_000_000)
        now = [100.0]
        publisher = loc.ExternalNavPublisher(
            "tcp:127.0.0.1:5762",
            loc.OdomAlignment((0.0, 0.0, 0.0)),
            machine,
            clock=lambda: now[0],
        )
        # A live feed offers states whose clock advances with every sample.
        publisher.offer(_state(time_ns=1_000_000_000), 1_000_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is not None
        # The stall: offers keep arriving fresh on the wall clock -- the silence
        # watchdog alone would never trip -- but the state's own clock is frozen.
        now[0] += 0.100
        publisher.offer(_state(time_ns=1_000_000_000), 1_200_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is not None
        # 0.25 s after the clock first appeared: still within the declared bound.
        now[0] += 0.150
        publisher.offer(_state(time_ns=1_000_000_000), 1_600_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is not None
        # 0.41 s after the clock first appeared: the bound has tripped.
        now[0] += 0.160
        publisher.offer(_state(time_ns=1_000_000_000), 2_000_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is None
        # The recovery: the clock advances again, and transmission resumes.
        now[0] += 1.0
        publisher.offer(_state(time_ns=3_000_000_000), 3_000_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is not None

    def test_a_parked_frozen_clock_does_not_stop_publication(self):
        """Before the flight is declared, a frozen clock does not stop transmission.

        The same reason the parked sigma excursion is benign: the vehicle is parked,
        a stale-but-correct pose is harmless, and a publication stop starves the
        firmware's VISO health window and refuses the arm (FIXER6 runs
        p01l-fixer6-1/-2/-6, refused with 'VisOdom: not healthy' / 'Need Alt
        Estimate'). The flight-side gate is unchanged: the window's declaration turns
        it on (pinned above).
        """
        machine = loc.HealthMachine(_bounds(state_lost_after_s=0.300))
        now = [100.0]
        publisher = loc.ExternalNavPublisher(
            "tcp:127.0.0.1:5762",
            loc.OdomAlignment((0.0, 0.0, 0.0)),
            machine,
            clock=lambda: now[0],
        )
        publisher.offer(_state(time_ns=1_000_000_000), 1_000_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is not None
        now[0] += 0.100
        publisher.offer(_state(time_ns=1_000_000_000), 1_600_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is not None
        now[0] += 1.000
        publisher.offer(_state(time_ns=1_000_000_000), 3_000_000_000)
        assert publisher.state_for_publish(now[0] + 0.001) is not None


class TestRouteYaw:
    def test_the_scored_route_commands_the_declared_spawn_heading(self):
        """The frozen route is a position command over a vehicle that spawns at
        rest yaw 0 facing the doorway (plan section 0.3 item 1), and it declares
        no yaw of its own. A yaw-IGNORED target hands the heading to the
        firmware's default behavior -- WP_YAW_BEHAVIOR 2 aligns yaw with the
        position controller's desired velocity -- and that firmware-invented
        yaw slew, the first yaw maneuver of every flight, is the measured seed
        of the end-of-route tumble (dataflash 00000085, 00000105, 00000106,
        00000107: yaw 0 -> ~51 deg on the second leg, then a growing ~3-4 Hz
        roll-yaw oscillation, motor saturation, tumble from 1.5 m, crash
        disarm inside the scored window). The scored window therefore commands
        the declared heading as an angle, and the wire mask must carry it as a
        yaw command, not an ignored field."""
        assert check.ROUTE_YAW_HOLD_RAD == 0.0
        target = bridge.LocalNedTarget(
            position_ned=(2.0, 0.0, -1.5),
            velocity_ned=(0.0, 0.0, 0.0),
            yaw_rad=check.ROUTE_YAW_HOLD_RAD,
            deadline_s=8.0,
            certificate_ref=None,
        )
        motion = bridge.MotionTarget(
            position_ned=target.position_ned,
            velocity_ned=target.velocity_ned,
            acceleration_ned=None,
            yaw_rad=target.yaw_rad,
            yaw_rate_rad_s=None,
        )
        mask = bridge.mask_for_target(motion)
        assert mask & bridge.TYPE_MASK_YAW_IGNORE == 0
        assert mask & bridge.TYPE_MASK_YAW_RATE_IGNORE != 0

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
        """The pose-assisted arm is now the live E1-DIAG diagnostic; a preflight
        that cannot be satisfied must block BEFORE anything starts, labelled. The
        config here has no estimator process, so the diagnostic blocks in its own
        preflight and the test stays hermetic -- the declared configuration would
        otherwise attempt a real flight on a host that has the simulator."""
        config_path = _config_without_the_estimator(tmp_path)
        exit_code = check.main(
            [
                "--config",
                str(config_path),
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
        # The diagnostic blocks, it never resolves or unresolves the stage claim.
        assert "localization=unresolved" not in receipt["reasons"]

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
        """Re-pointed (plan section 10, T5/T6): the first layer named GPS_TYPE and the
        vehicle has no such parameter (AP_GPS.cpp:364), so the layer now names
        GPS1_TYPE/GPS2_TYPE and the readback gate judges those names."""
        declared = {name: expected for name, expected, _source in check.VEHICLE_REQUIREMENTS}
        assert check._readback_blockers(declared, {}) == []

        with_gps = dict(declared, GPS1_TYPE=1.0)
        blockers = check._readback_blockers(with_gps, {})
        assert blockers and "GPS1_TYPE" in blockers[0]

        unanswered = {name: value for name, value in declared.items() if name != "GPS1_TYPE"}
        blockers = check._readback_blockers(unanswered, {})
        assert any("GPS1_TYPE" in blocker and "silent" in blocker for blocker in blockers)

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


# ---------------------------------------------------------------------------
# The pin, recorded as a measurement rather than a silent pass (plan section 3)
# ---------------------------------------------------------------------------


class TestPinEvidence:
    """The pin leg's evidence, read from the bytes on disk.

    The preflight refused to start an arm on an unproven pin, but a satisfied pin
    left no row at all, so a reader of the receipt could only infer the version
    evidence from the absence of a failure — the reading section 4.6's
    truth-republish gate was re-pointed at. These tests hold the pin's row to a
    measurement: the tarball re-hashed, the build marker read out of the build
    log, and the library and the process hashed as they are.
    """

    def test_the_preflight_records_the_pin_it_re_hashed(self, tmp_path):
        rows, satisfied = check._preflight(
            check._load_localization_config(Path("configs/first_indoor.yaml")),
            tmp_path,
            SensorMode.SENSOR_DERIVED,
        )
        assert satisfied is True
        row = {entry["name"]: entry for entry in rows}["estimator_pin"]
        assert row["satisfied"] is True
        evidence = row["evidence"]
        assert evidence["tarball_matches_configured_pin"] is True
        assert evidence["tarball_sha256_measured"] == evidence["tarball_sha256_configured"]
        assert evidence["build_marker_present"] is True
        assert "sha256" in row["detail"] and "matches the configured pin" in row["detail"]
        # The hashes are of the files themselves, not of their configured names.
        root = check.repository_root()
        for key in ("library", "executable"):
            digest = hashlib.sha256((root / evidence[key]["path"]).read_bytes()).hexdigest()
            assert evidence[key]["sha256"] == digest

    def test_a_tarball_that_is_not_the_pin_fails_the_row_naming_both_digests(self, tmp_path):
        document = _declared_document()
        tarball = tmp_path / "open_vins-2.7.tar.gz"
        tarball.write_bytes(b"not the pinned tarball")
        document["localization"]["estimator"]["tarball_path"] = str(tarball)

        record, blockers = check._pin_evidence(document["localization"], tmp_path)
        assert record["tarball_matches_configured_pin"] is False
        assert blockers and record["tarball_sha256_measured"] in blockers[0]
        assert record["tarball_sha256_configured"] in blockers[0]

# ---------------------------------------------------------------------------
# T6: the re-derived GPS-off gate (plan sections 4.6, 4.7)
# ---------------------------------------------------------------------------


# The vehicle's own recorded answer to the first layer's GPS_TYPE read
# (run-2026-09-26T07-40-00Z/run-a/mavlink.jsonl, recorded as UNKNOWN_345): a
# PARAM_ERROR frame whose payload is param_index -1, target_system 250,
# target_component 190, param_id "GPS_TYPE", error 1 (DOES_NOT_EXIST).
CAPTURED_PARAM_ERROR_FRAME = (
    b"\xfd\x15\x00\x00\x04\x01\x01Y\x01\x00"
    b"\xff\xff\xfa\xbe"
    b"GPS_TYPE\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x01\x0f\x96"
)


class TestGpsOffGate:
    def test_the_captured_param_error_frame_decodes_as_the_vehicle_s_refusal(self):
        decoded = check._decode_param_error(
            {
                "mavpackettype": "UNKNOWN_345",
                "data": str(bytearray(CAPTURED_PARAM_ERROR_FRAME)),
            }
        )
        assert decoded == {
            "param_id": "GPS_TYPE",
            "param_index": -1,
            "error": 1,
            "target_system": 250,
            "target_component": 190,
        }

    def test_a_required_name_the_vehicle_refuses_fails_the_arm_naming_refused(self):
        blockers = check._readback_blockers({}, {"GPS1_TYPE": 1})
        named = [blocker for blocker in blockers if "GPS1_TYPE" in blocker]
        assert len(named) == 1 and "refused" in named[0]

    def test_gps_status_samples_are_judged_per_signal(self, tmp_path):
        log = tmp_path / "mavlink.jsonl"

        def write(*records: dict) -> None:
            log.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )

        write(
            {"mavpackettype": "SYS_STATUS", "onboard_control_sensors_present": 31},
            {"mavpackettype": "GPS_RAW_INT", "fix_type": 0},
        )
        verdict = check._gps_aiding_verdict(log)
        assert verdict["blockers"] == []
        assert verdict["sys_status_samples"] == 1
        assert verdict["gps_raw_int_samples"] == 1

        write({"mavpackettype": "SYS_STATUS", "onboard_control_sensors_present": 32})
        verdict = check._gps_aiding_verdict(log)
        assert any("GPS-present bit" in blocker for blocker in verdict["blockers"])

        write({"mavpackettype": "STATUSTEXT", "text": "GPS 1: detected u-blox"})
        verdict = check._gps_aiding_verdict(log)
        assert any("detected u-blox" in blocker for blocker in verdict["blockers"])

        write({"mavpackettype": "GPS_RAW_INT", "fix_type": 3})
        verdict = check._gps_aiding_verdict(log)
        assert any("fix_type" in blocker for blocker in verdict["blockers"])

        log.write_text("", encoding="utf-8")
        verdict = check._gps_aiding_verdict(log)
        assert any("never sampled" in blocker for blocker in verdict["blockers"])

    def test_the_source_sets_that_cannot_select_gps_are_required(self):
        required = {name: expected for name, expected, _source in check.VEHICLE_REQUIREMENTS}
        assert required["EK3_SRC2_POSXY"] == 0.0
        assert required["EK3_SRC3_YAW"] == 0.0
        gps_selected = dict(required, EK3_SRC2_POSXY=3.0)
        blockers = check._readback_blockers(gps_selected, {})
        assert any("EK3_SRC2_POSXY" in blocker for blocker in blockers)

    def test_params_applied_record_keeps_the_three_outcomes_distinct(self):
        record = check._params_applied_record({"GPS1_TYPE": 0.0}, {"GPS2_TYPE": 1})
        assert record["GPS1_TYPE"]["outcome"] == "answered"
        assert record["GPS2_TYPE"]["outcome"] == "refused"
        assert record["GPS2_TYPE"]["param_error"] == 1
        assert record["EK3_SRC2_POSXY"]["outcome"] == "silent"


# ---------------------------------------------------------------------------
# T7: the scene-admission check (plan sections 3.6, 10, 12.6)
# ---------------------------------------------------------------------------


def _write_left_frame(pairs_dir: Path, name: str, image: np.ndarray) -> None:
    import cv2

    pairs_dir.mkdir(parents=True, exist_ok=True)
    # The recorded captures are 3-channel ppm files; the check reads them back as
    # grayscale, so the fixture writes the same shape the platform produces.
    cv2.imwrite(str(pairs_dir / name), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))


def _write_pair_frames(
    pairs_dir: Path, index: int, left: np.ndarray, right: np.ndarray
) -> None:
    """A capture holds a stereo pair or it holds nothing the stereo gate can use.

    Passing one array twice writes two byte-identical files, which is the duplicate
    second eye the gate must refuse.
    """
    _write_left_frame(pairs_dir, f"{index:05d}-left.ppm", left)
    _write_left_frame(pairs_dir, f"{index:05d}-right.ppm", right)


def _featureless_image(value: int = 0) -> np.ndarray:
    """A flat frame, and its two eyes differ by the one value, so a pair of two
    featureless views is still a pair and is refused for its features alone."""
    return np.full((480, 640), value, np.uint8)


def _textured_image() -> np.ndarray:
    """A textured scene: seeded noise is dense with corners the FAST detector finds,
    which is what the pinned tracker's front end needs (plan section 3.6)."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(480, 640), dtype=np.uint8)


class TestSceneAdmission:
    """T7 and its revision-4 extension: the admission gate is real, both ways,
    and it only measures frames that can prove which world they show."""

    def _world_settings(self, tmp_path: Path, world_text: str = "#VRML_SIM R2025a utf8\n"):
        """Settings whose world exists under tmp_path, so its sha256 is real."""
        world = tmp_path / "scenarios/missions/dev/dev-a-single/world.wbt"
        world.parent.mkdir(parents=True, exist_ok=True)
        world.write_text(world_text, encoding="utf-8")
        settings = PlatformSettings.from_config(_declared_document(), root=tmp_path)
        return replace(settings, world=world), world

    def test_a_stereo_scene_that_clears_the_floor_in_both_eyes_passes(self, tmp_path):
        import cv2

        detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        assert (
            len(detector.detect(_textured_image(), None))
            >= check.INITIALIZER_FEATURE_FLOOR
        )
        settings, world = self._world_settings(tmp_path)
        run_dir = tmp_path / "work/runs/p01-localization/run-test/run-a"
        _write_pair_frames(
            run_dir / "pairs",
            1,
            _textured_image(),
            np.roll(_textured_image(), 4, axis=1),
        )
        (run_dir / "scene-capture.json").write_text(
            json.dumps({"world_sha256": check._sha256(world)}), encoding="utf-8"
        )
        ok, state, detail = check._scene_admission_check(settings, tmp_path)
        assert (ok, state) == (True, "measured_pass"), detail
        assert "left" in detail and "right" in detail

    def test_a_featureless_pair_fails_with_both_eyes_and_the_floor_named(self, tmp_path):
        settings, world = self._world_settings(tmp_path)
        run_dir = tmp_path / "work/runs/p01-localization/run-test/run-a"
        _write_pair_frames(
            run_dir / "pairs", 1, _featureless_image(0), _featureless_image(1)
        )
        (run_dir / "scene-capture.json").write_text(
            json.dumps({"world_sha256": check._sha256(world)}), encoding="utf-8"
        )
        ok, state, detail = check._scene_admission_check(settings, tmp_path)
        assert (ok, state) == (False, "measured_fail")
        assert str(check.INITIALIZER_FEATURE_FLOOR) in detail
        assert "left [0]" in detail and "right [0]" in detail

    def test_a_capture_whose_two_eyes_are_the_same_image_cannot_gate(self, tmp_path):
        """One view duplicated is not a stereo pair: a capture that records the same
        plane twice says nothing about what the estimator's second input carries, so
        it is skipped like an unhashed capture and the arm gate measures the run's
        own pair instead."""
        settings, world = self._world_settings(tmp_path)
        run_dir = tmp_path / "work/runs/p01-localization/run-test/run-a"
        textured = _textured_image()
        _write_pair_frames(run_dir / "pairs", 1, textured, textured)
        (run_dir / "scene-capture.json").write_text(
            json.dumps({"world_sha256": check._sha256(world)}), encoding="utf-8"
        )
        ok, state, detail = check._scene_admission_check(settings, tmp_path)
        assert (ok, state) == (True, "deferred_to_arm_gate"), detail
        assert "no complete, distinct, readable stereo pair" in detail

    def test_a_left_only_capture_cannot_gate_a_stereo_arm(self, tmp_path):
        """Every capture recorded before this revision holds only `-left.ppm` -- the
        last invocation's 24 frames among them -- and the gate passed at 16 left-eye
        keypoints while nothing recorded the second view. A left-only capture is now
        skipped, exactly as an unhashed one is."""
        settings, world = self._world_settings(tmp_path)
        run_dir = tmp_path / "work/runs/p01-localization/run-test/run-a"
        _write_left_frame(run_dir / "pairs", "00001-left.ppm", _textured_image())
        (run_dir / "scene-capture.json").write_text(
            json.dumps({"world_sha256": check._sha256(world)}), encoding="utf-8"
        )
        ok, state, detail = check._scene_admission_check(settings, tmp_path)
        assert (ok, state) == (True, "deferred_to_arm_gate"), detail

    def test_a_capture_of_another_world_is_skipped(self, tmp_path):
        settings, _world = self._world_settings(tmp_path)
        run_dir = tmp_path / "work/runs/p00-compat/accept-other/run-a"
        _write_pair_frames(
            run_dir / "pairs", 1, _textured_image(), np.roll(_textured_image(), 4, axis=1)
        )
        (run_dir / "startup.json").write_text(
            json.dumps({"assets": {"world_sha256": "not-this-world"}}), encoding="utf-8"
        )
        ok, state, detail = check._scene_admission_check(settings, tmp_path)
        assert (ok, state) == (True, "deferred_to_arm_gate")
        assert "no hash-matched recorded stereo capture" in detail

    def test_a_capture_that_records_no_world_hash_is_skipped(self, tmp_path):
        """The accept-5 hole: P00's capture predates the world hash, so its frames
        cannot prove which world they show -- and with a development route
        configured, measuring them would judge one scene against another world's
        gate. Revision 4 skips them instead."""
        settings, _world = self._world_settings(tmp_path)
        run_dir = tmp_path / "work/runs/p00-compat/accept-5/run-a"
        _write_pair_frames(
            run_dir / "pairs", 1, _featureless_image(0), _featureless_image(1)
        )
        ok, state, detail = check._scene_admission_check(settings, tmp_path)
        assert (ok, state) == (True, "deferred_to_arm_gate")
        assert "skipped for recording no world hash" in detail

    def test_the_preflight_no_longer_refuses_the_textured_route(self, tmp_path):
        """The declared configuration names dev-a-single. The preflight admits it
        either way: as `deferred_to_arm_gate` before any capture of that world
        exists, or as `measured_pass` once a run has recorded a stereo capture --
        the first textured invocation's own hash-anchored capture is what turns the
        first state into the second (plan section 0.3 item 4). What must never
        happen is the old refusal, where the compat scene's unprovenanced frames
        were measured against this world's gate."""
        rows, _satisfied = check._preflight(
            check._load_localization_config(Path("configs/first_indoor.yaml")),
            tmp_path,
            SensorMode.SENSOR_DERIVED,
        )
        row = {entry["name"]: entry for entry in rows}["scene_admission"]
        assert row["satisfied"] is True
        assert row["state"] in {"deferred_to_arm_gate", "measured_pass"}
        assert "dev-a-single" in row["detail"]
        assert "compat_stereo" not in row["detail"]

    def test_the_arm_gate_refuses_a_featureless_pair_on_its_own_frames(self, tmp_path):
        """Run 4's refusal, made procedural: the arm gate measures the run's own
        recorded frames and refuses the arm when they do not clear the floor."""
        settings, world = self._world_settings(tmp_path)
        writer = bridge_EvidenceWriter(tmp_path / "run", "run-a")
        pairs_dir = tmp_path / "run/run-a/pairs"
        _write_pair_frames(pairs_dir, 1, _featureless_image(0), _featureless_image(1))
        blocker = check._scene_capture_gate(
            writer, settings, lambda: None, {"count": 1, "last_s": 0.0}, []
        )
        assert blocker and "FAST keypoints" in blocker
        assert "left 0, right 0" in blocker

    def test_the_arm_gate_refuses_a_duplicate_second_eye_without_measuring(self, tmp_path):
        """The gap the last invocation exposed: the capture recorded one eye and the
        gate passed on it. With both eyes recorded, a run whose right plane is the
        left plane duplicated cannot open the arm -- there is no stereo stream to
        feed, and the refusal says which evidence was missing."""
        settings, _world = self._world_settings(tmp_path)
        writer = bridge_EvidenceWriter(tmp_path / "run", "run-a")
        pairs_dir = tmp_path / "run/run-a/pairs"
        textured = _textured_image()
        _write_pair_frames(pairs_dir, 1, textured, textured)
        blocker = check._scene_capture_gate(
            writer, settings, lambda: None, {"count": 1, "last_s": 0.0}, []
        )
        assert blocker and "no complete stereo pair" in blocker
        assert "identical pairs 1" in blocker
        record = json.loads(
            (tmp_path / "run/run-a/scene-capture.json").read_text(encoding="utf-8")
        )
        assert record["state"] == "no_stereo_pair"
        assert record["identical_pairs"] == ["00001-left.ppm"]

    def test_the_arm_gate_passes_a_textured_pair_on_its_own_frames(self, tmp_path):
        settings, _world = self._world_settings(tmp_path)
        writer = bridge_EvidenceWriter(tmp_path / "run", "run-a")
        pairs_dir = tmp_path / "run/run-a/pairs"
        _write_pair_frames(
            pairs_dir, 1, _textured_image(), np.roll(_textured_image(), 4, axis=1)
        )
        blocker = check._scene_capture_gate(
            writer, settings, lambda: None, {"count": 1, "last_s": 0.0}, []
        )
        assert blocker is None
        record = json.loads(
            (tmp_path / "run/run-a/scene-capture.json").read_text(encoding="utf-8")
        )
        assert record["state"] == "measured_pass"
        assert record["world_sha256"] == check._sha256(settings.world)
        assert record["pairs_recorded"] == 1
        assert min(record["left_keypoint_counts"]) >= check.INITIALIZER_FEATURE_FLOOR
        assert min(record["right_keypoint_counts"]) >= check.INITIALIZER_FEATURE_FLOOR


# ---------------------------------------------------------------------------
# T8/T9/T10 (new in revision 4): world selection, origin from the world, A1
# ---------------------------------------------------------------------------


class TestWorldSelection:
    """T8: localization.world selects the development route; scenario.world is
    the compatibility gate's measured vehicle and never moves."""

    def test_the_schema_accepts_the_world_key_and_scenario_world_stands(self):
        from embodied.cli import load_config

        document = load_config(Path("configs/first_indoor.yaml"))
        assert document["localization"]["world"] == (
            "scenarios/missions/dev/dev-a-single/world.wbt"
        )
        assert document["scenario"]["world"] == "scenarios/compat/worlds/compat_stereo.wbt"

    def test_the_check_runs_in_the_declared_development_world(self):
        document = _declared_document()
        settings = check._platform_settings(document, Path.cwd().resolve())
        assert settings.world.name == "world.wbt"
        assert "dev-a-single" in str(settings.world)

    def test_an_absent_key_falls_back_to_scenario_world(self):
        document = _declared_document()
        document["localization"].pop("world")
        settings = check._platform_settings(document, Path.cwd().resolve())
        plain = PlatformSettings.from_config(document, root=Path.cwd().resolve())
        assert settings.world == plain.world
        assert plain.world.name == "compat_stereo.wbt"


class TestOdomOriginFromTheWorld:
    """T9: the odom origin is the configured world's own vehicle translation."""

    def test_the_dev_world_declares_the_vestibule_spawn(self):
        origin = check._declared_start_origin(
            Path("scenarios/missions/dev/dev-a-single/world.wbt")
        )
        assert origin == (-1.0, 0.0, 0.09)

    def test_the_mission_yaml_spawn_agrees_with_the_world(self):
        mission = yaml.safe_load(
            Path("scenarios/missions/dev/dev-a-single/mission.yaml").read_text(
                encoding="utf-8"
            )
        )
        spawn = mission["spawn_pose"]
        origin = check._declared_start_origin(
            Path("scenarios/missions/dev/dev-a-single/world.wbt")
        )
        assert (spawn["x"], spawn["y"], spawn["z"]) == origin

    def test_the_compat_worlds_own_spawn_disagrees_with_the_referee(self):
        """The recorded 7 cm gap (plan section 13): the referee declares
        [0, 0, 0.02] while the compat scene's own Iris spawns at 0.09 -- which is
        why the world file, not the referee, is the odom anchor."""
        origin = check._declared_start_origin(
            Path("scenarios/compat/worlds/compat_stereo.wbt")
        )
        referee = _declared_document()["calibration"]["referee"][
            "body_position_world_m"
        ]
        assert origin == (0.0, 0.0, 0.09)
        assert referee == [0.0, 0.0, 0.02]

    def test_a_world_without_a_vehicle_translation_is_refused(self, tmp_path):
        world = tmp_path / "empty.wbt"
        world.write_text("#VRML_SIM R2025a utf8\n", encoding="utf-8")
        with pytest.raises(Exception, match="Iris"):
            check._declared_start_origin(world)

    def test_the_declared_start_attitude_is_the_identity_in_both_worlds(self):
        """Neither declared world rotates its Iris node, and the seal rests on
        that declaration, so it is asserted rather than assumed."""
        assert check._declared_start_attitude(
            Path("scenarios/missions/dev/dev-a-single/world.wbt")
        ) == (0.0, 0.0, 0.0)
        assert check._declared_start_attitude(
            Path("scenarios/compat/worlds/compat_stereo.wbt")
        ) == (0.0, 0.0, 0.0)

    def test_the_mission_yaw_agrees_with_the_worlds_declared_start(self):
        mission = yaml.safe_load(
            Path("scenarios/missions/dev/dev-a-single/mission.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert mission["spawn_pose"]["yaw_rad"] == check._declared_start_attitude(
            Path("scenarios/missions/dev/dev-a-single/world.wbt")
        )[2]

    def test_a_tilted_start_rotation_is_refused_not_approximated(self, tmp_path):
        world = tmp_path / "tilted.wbt"
        world.write_text(
            'Iris {\n  translation 0 0 0.09\n  rotation 1 0 0 0.5\n'
            '  controller "x"\n}\n',
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="tilted"):
            check._declared_start_attitude(world)

    def test_a_yaw_only_start_rotation_becomes_a_negative_ned_yaw(self, tmp_path):
        world = tmp_path / "yawed.wbt"
        world.write_text(
            'Iris {\n  translation 0 0 0.09\n  rotation 0 0 1 1.5707963267948966\n'
            '  controller "x"\n}\n',
            encoding="utf-8",
        )
        assert check._declared_start_attitude(world) == pytest.approx(
            (0.0, 0.0, -math.pi / 2), abs=1e-9
        )


class TestAttitudeGate:
    """T10's A1 half, second form: the pre-arm gate checks the composition the
    adapter performed and the world's declaration the seal rests on, and refuses
    the arm by name when either is off."""

    @staticmethod
    def _writer(tmp_path: Path):
        return bridge_EvidenceWriter(tmp_path / "run", "run-a")

    @staticmethod
    def _stats(truth_rpy=(0.0, 0.0, 0.0)):
        stats = check._FeedStats()
        stats.truth_attitudes.append((1_000_000_000, tuple(truth_rpy)))
        return stats

    @staticmethod
    def _alignment(declared=(0.0, 0.0, 0.0), initial=(0.0, 0.0, 0.0, 1.0)):
        alignment = loc.OdomAlignment((0.0, 0.0, 0.0), declared)
        alignment.seal(initial)
        return alignment

    def test_the_sealed_start_passes_against_truth(self, tmp_path):
        blocker = check._attitude_gate(
            self._writer(tmp_path),
            {"attitude_rpy": (0.0, 0.0, 0.0)},
            self._alignment(initial=(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))),
            self._stats(),
            [],
        )
        assert blocker is None
        record = json.loads(
            (tmp_path / "run/run-a/attitude-gate.json").read_text(encoding="utf-8")
        )
        assert record["state"] == "measured_pass"
        # The unobservable yaw is recorded, not gated: this run's delivered
        # quaternion reads (as the wire carries it, odom->body) as a +90 degree
        # z rotation, so the odom frame initialized yawed -90 degrees from the
        # declared start attitude, and the seal absorbed exactly that.
        assert record["epoch_yaw_deg"] == pytest.approx(-90.0, abs=1e-6)

    def test_a_composition_that_misses_the_declared_start_refuses_the_arm(self, tmp_path):
        blocker = check._attitude_gate(
            self._writer(tmp_path),
            {"attitude_rpy": (0.0, 0.0, math.pi / 2)},
            self._alignment(),
            self._stats(),
            [],
        )
        assert blocker and "composition deltas" in blocker
        record = json.loads(
            (tmp_path / "run/run-a/attitude-gate.json").read_text(encoding="utf-8")
        )
        assert record["state"] == "measured_fail"

    def test_a_declaration_that_contradicts_the_simulator_refuses_the_arm(self, tmp_path):
        """The world's declared start is checked against the simulator's own
        attitude: a declaration that names a yaw the scene does not have is a
        refusal, not a silent rotation of the published frame."""
        blocker = check._attitude_gate(
            self._writer(tmp_path),
            {"attitude_rpy": (0.0, 0.0, math.pi / 2)},
            self._alignment(declared=(0.0, 0.0, math.pi / 2)),
            self._stats(truth_rpy=(0.0, 0.0, 0.0)),
            [],
        )
        assert blocker and "declaration deltas" in blocker
        record = json.loads(
            (tmp_path / "run/run-a/attitude-gate.json").read_text(encoding="utf-8")
        )
        assert max(abs(value) for value in record["declaration_deltas_deg"]) == pytest.approx(
            90.0, abs=1e-6
        )

    def test_a_tilted_sealed_frame_refuses_the_arm(self, tmp_path):
        """The vertical chain is gated: a sealed rotation that does not carry
        odom-up to NED-down cannot pass on a 90-degree level error."""
        tilted = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
        blocker = check._attitude_gate(
            self._writer(tmp_path),
            {"attitude_rpy": (0.0, 0.0, 0.0)},
            self._alignment(initial=tilted),
            self._stats(),
            [],
        )
        assert blocker and "level error" in blocker
        record = json.loads(
            (tmp_path / "run/run-a/attitude-gate.json").read_text(encoding="utf-8")
        )
        assert record["level_error_deg"] == pytest.approx(90.0, abs=1e-6)

    def test_no_publication_to_compare_is_a_blocker_not_a_pass(self, tmp_path):
        blocker = check._attitude_gate(
            self._writer(tmp_path), None, self._alignment(), self._stats(), []
        )
        assert blocker and "no published state" in blocker


class TestH5Diagnosis:
    """H5's blocker must name the failure the run actually measured (plan section 11).

    The first textured re-run's receipt read "the estimator did not initialize"
    while its own `initializer-diagnostics.json` recorded the pinned initializer's
    success line, and the plan revision that followed was written against the wrong
    cause. These fixtures are the two line shapes that decision rests on, taken
    from that run's recorded `estimator.log`.
    """

    def _log(self, tmp_path, *lines):
        path = tmp_path / "estimator.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_a_successful_initialization_reads_as_the_accessor_not_the_initializer(
        self, tmp_path
    ):
        log = self._log(
            tmp_path,
            "ov_stream ready: openvins v2.7, 2 cameras, gravity 9.81, try_zupt=1",
            "ov_stream: 1 stereo frames, 981 imu samples, initialized=0",
            "\x1b[0m\x1b[32m[init]: successful initialization in 0.0015 seconds",
            "[ZUPT]: passed disparity (0.000 < 1.000, 95 features)",
            "[ZUPT]: accepted |v_IinG| = 0.001 (chi2 0.000 < 84.595)",
            "[ZUPT]: There are no IMU data to check for zero velocity with!!",
            "ov_stream: 550 stereo frames, 27593 imu samples, initialized=0",
        )
        diagnostics = check._initializer_diagnostics(log)
        assert diagnostics["initializer_succeeded"] is True
        assert diagnostics["zupt_accepted_updates"] == 1
        assert diagnostics["zupt_frames_without_imu"] == 1
        blocker = check._initialization_blocker(diagnostics)
        assert "timelastupdate" in blocker
        assert "did not initialize" not in blocker
        assert "no successful initialization" not in blocker
        assert "1 accepted zero-velocity update(s)" in blocker
        assert "owner's disposition" in blocker

    def test_an_initializer_that_never_fired_keeps_the_degenerate_reading(self, tmp_path):
        log = self._log(
            tmp_path,
            "ov_stream: 25 stereo frames, 1470 imu samples, initialized=0",
            "[init]: not enough feats to compute disp: 0,0 < 15",
        )
        diagnostics = check._initializer_diagnostics(log)
        assert diagnostics["initializer_succeeded"] is False
        blocker = check._initialization_blocker(diagnostics)
        assert "did not initialize" in blocker
        assert "no successful initialization" in blocker
        assert "timelastupdate" not in blocker

    def test_a_missing_log_is_not_a_successful_initialization(self, tmp_path):
        diagnostics = check._initializer_diagnostics(tmp_path / "absent.log")
        assert diagnostics["initializer_succeeded"] is False
        assert diagnostics["estimator_log_lines"] == 0


class TestZuftFrameDecisions:
    """T11: the per-frame decision behind H5's blocker (plan section 12 item 14).

    The plan's sufficiency criterion is five frames the zero-velocity updater declines
    inside the bring-up, because only a declined frame reaches
    ``do_feature_propagate_update`` and can make a clone (VioManager.cpp:348-352). The
    line shapes below are the pinned updater's own, taken from a recorded run's
    ``estimator.log``: a disparity line, then accept/reject, or the starvation warning.
    """

    PASSED = "[ZUPT]: passed disparity (0.000 < 1.000, 95 features)"
    ACCEPTED = "[ZUPT]: accepted |v_IinG| = 0.023 (chi2 0.000 < 84.595)"
    FAILED = "[ZUPT]: failed disparity (2.714 > 1.000, 71 features)"
    REJECTED = "[ZUPT]: rejected |v_IinG| = 0.412 (chi2 180.220 > 84.595)"
    STARVED = "[ZUPT]: There are no IMU data to check for zero velocity with!!"

    def _log(self, tmp_path, *lines):
        path = tmp_path / "estimator.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_each_decision_carries_the_numbers_it_was_made_from(self, tmp_path):
        log = self._log(
            tmp_path,
            "\x1b[0m\x1b[36m" + self.PASSED,
            self.ACCEPTED,
            self.FAILED,
            self.REJECTED,
            self.STARVED,
        )
        diagnostics = check._initializer_diagnostics(log)
        frames = diagnostics["zupt_frames"]
        assert [frame["decision"] for frame in frames] == [
            "accepted",
            "declined_motion",
            "declined_no_imu",
        ]
        assert frames[0]["disparity_px"] == 0.0
        assert frames[0]["feature_count"] == 95
        assert frames[0]["velocity_m_s"] == 0.023
        assert frames[1]["disparity_passed"] is False
        assert frames[1]["chi2"] == 180.220
        assert frames[1]["chi2_limit"] == 84.595
        assert diagnostics["zupt_accepted_updates"] == 1
        assert diagnostics["zupt_rejected_updates"] == 1
        assert diagnostics["zupt_frames_without_imu"] == 1
        assert diagnostics["zupt_frames_reaching_visual_path"] == 2

    def test_a_frame_with_no_verdict_is_not_classified_as_a_decision(self, tmp_path):
        log = self._log(tmp_path, self.FAILED)
        diagnostics = check._initializer_diagnostics(log)
        assert diagnostics["zupt_frames"][0]["decision"] == "no_verdict"
        assert diagnostics["zupt_frames_reaching_visual_path"] == 0

    def test_the_blocker_names_the_measured_count_against_the_criterion(self, tmp_path):
        log = self._log(
            tmp_path,
            "\x1b[0m\x1b[32m[init]: successful initialization in 0.0015 seconds",
            self.PASSED,
            self.ACCEPTED,
        )
        blocker = check._initialization_blocker(check._initializer_diagnostics(log))
        assert "0 frame(s) reached the visual path" in blocker
        assert "five the accessor needs" in blocker

    def test_the_summary_agrees_with_the_recorded_run_it_was_built_from(self):
        """The artifact this parses is a real run's log, when that run is on this host.

        ``work/`` is local-only, so a clean clone skips this; on the host that ran it,
        the recorded pre-arm window must classify without leftovers and its frames
        reaching the visual path must be the declined ones, which is the quantity the
        run-6 receipt argued from the log tail.
        """
        log = (
            Path(check.__file__).resolve().parents[3]
            / "work/runs/p01-localization/p01l-run6-both-eyes-20260926T115350Z/run-a/estimator.log"
        )
        if not log.is_file():
            pytest.skip(f"the recorded run log is not on this host: {log}")
        diagnostics = check._initializer_diagnostics(log)
        frames = diagnostics["zupt_frames"]
        assert frames, "a run that consumed 525 stereo frames recorded decisions"
        assert all(frame["decision"] != "no_verdict" for frame in frames)
        assert diagnostics["zupt_accepted_updates"] > 0
        assert diagnostics["zupt_frames_reaching_visual_path"] == (
            diagnostics["zupt_rejected_updates"] + diagnostics["zupt_frames_without_imu"]
        )


# ---------------------------------------------------------------------------
# T12/T13 (E1-DIAG, plan sections 0.7 item 5, 0.8 item 3): the diagnostic's
# labels, and the sensor-derived arm's unchanged behaviour beside it
# ---------------------------------------------------------------------------


class TestPoseAssistedDiagnostic:
    """E1-DIAG's two named behaviours: labelled beyond misreading, and the
    sensor-derived arm untouched by it.

    The diagnostic flies the bounded excitation on the truth-driven arm while the
    pinned estimator observes (plan sections 0.6 item 7, 0.8). Its result can
    never be read as a scored one, and the arm that CAN be scored must come out
    of the build exactly as it went in: truth republish off, nothing published by
    the bridge.
    """

    def test_the_diagnostic_is_labelled_and_refuses_to_be_read_as_a_scored_result(
        self, tmp_path
    ):
        """Every shape the diagnostic's outcome can take carries the labels.

        A completed diagnostic is COMPLETE -- its measurement is a result -- and
        that is exactly the outcome a reader could mistake for a pass, so the
        labels must make the mistake impossible: gate not applicable,
        localization not applicable, the never-pool non-claim, and no
        predeclared bound judged.
        """
        completed = check._pose_assisted_outcome(())
        assert completed.status.value == "complete"
        assert completed.gate_status.value == "not_applicable"
        assert completed.manifest["sensor_mode_label"] == "pose-assisted-diagnostic"
        assert completed.manifest["localization"] == "not_applicable"
        limitations = " ".join(completed.limitations)
        assert "NOT a sensor-derived result" in limitations
        assert "never be pooled" in limitations
        assert "no predeclared E/F/H bound is judged" in limitations
        assert "truth republish" in limitations  # the exemption is stated, not silent
        assert any("cannot pass P01-L" in reason for reason in completed.reasons)

        blocked = check._pose_assisted_outcome(("a blocker",), {"pairs_fed": 3})
        assert blocked.status.value == "blocked"
        assert blocked.gate_status.value == "not_applicable"
        assert blocked.manifest["localization"] == "not_applicable"
        assert blocked.manifest["pairs_fed"] == 3
        assert "localization=unresolved" not in blocked.reasons

        # And the diagnostic's own preflight rows declare, rather than hide, the
        # two things a scored preflight would refuse: the per-run arm override
        # and the truth-republish exemption.
        document = check._load_localization_config(Path("configs/first_indoor.yaml"))
        rows, satisfied = check._diagnostic_preflight(document, tmp_path)
        rows = {row["name"]: row for row in rows}
        assert rows["diagnostic_arm_override"]["satisfied"] is True
        assert "sensor-derived" in rows["diagnostic_arm_override"]["detail"]
        assert rows["bridge_truth_republish"]["state"] == "declared_exemption"
        assert rows["bridge_truth_republish"]["satisfied"] is True
        assert "truth republish is ON" in rows["bridge_truth_republish"]["detail"]
        # The declared configuration's scored preflight is unchanged by all this.
        scored_rows, scored_satisfied = check._preflight(
            document, tmp_path, SensorMode.SENSOR_DERIVED
        )
        scored_rows = {row["name"]: row for row in scored_rows}
        assert scored_rows["localization_mode"]["satisfied"] is True
        assert scored_rows["bridge_truth_republish"]["satisfied"] is True
        assert "exactly one publisher" in scored_rows["bridge_truth_republish"]["detail"]

    def test_the_sensor_derived_arm_is_unchanged_truth_republish_off_published_zero(
        self, tmp_path
    ):
        """The diagnostic's per-run arm override never leaks into the scored arm.

        The scored arm's settings are built with no override, so the bridge's
        truth republish stays off however the diagnostic asks for its own; and
        the recorded scored runs' own artifact answers the same question from
        the vehicle side: the bridge published zero simulator poses.
        """
        root = Path(check.__file__).resolve().parents[3]
        document = check._load_localization_config(Path("configs/first_indoor.yaml"))
        assert document["localization"]["mode"] == "sensor-derived"
        scored = check._platform_settings(document, root)
        assert scored.truth_republish is False
        assert scored.sensor_mode is SensorMode.SENSOR_DERIVED
        # The override exists only when the diagnostic passes it explicitly.
        diagnostic = check._platform_settings(
            document, root, arm=SensorMode.POSE_ASSISTED.value
        )
        assert diagnostic.truth_republish is True
        assert scored.truth_republish is False  # unchanged by the diagnostic's ask

        # The recorded scored arm's own artifact, when this host has one: the
        # bridge's vision-pose feed recorded enabled false and published 0.
        recorded = sorted(
            (
                path
                for path in (root / "work/runs/p01-localization").glob(
                    "*/run-a/vision-pose-feed.json"
                )
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime,
        )
        scored_artifacts = []
        for path in reversed(recorded):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("enabled") is False:
                scored_artifacts.append((path, record))
                break
        if not scored_artifacts:
            pytest.skip("no recorded sensor-derived run artifact is on this host")
        path, record = scored_artifacts[0]
        assert record["published"] == 0, f"{path} records {record['published']}"


# ---------------------------------------------------------------------------
# T14/T15 (the declared ordered bring-up, plan sections 0.6 item 6, 0.8 item 7):
# the exception window is declared and applied, and it is BOUNDED -- the scored
# window refuses to open while it is still in force
# ---------------------------------------------------------------------------


class TestOrderedBringUp:
    """The two named behaviours that make the exception a declaration, not a bypass.

    The bring-up exists because the sensor-derived arm cannot move: the estimator
    latches only under motion (E1-DIAG), and the arm gate forbids motion until the
    vision source is healthy, which is the estimator's own adapter. What makes
    that legitimate rather than a silent disable is that (a) every element of the
    window is declared with the pinned source that requires it, and (b) the window
    cannot still be in force when the scored window opens -- the vehicle's own
    readback is what says so.
    """

    def _settings_and_window(self):
        root = Path(check.__file__).resolve().parents[3]
        document = check._load_localization_config(Path("configs/first_indoor.yaml"))
        settings = check._platform_settings(document, root)
        return settings, check._bring_up_window(settings)

    def test_the_window_applies_the_declared_exception_and_the_origin_datum(self):
        """The window is the declared one: mask, cited checks, and a datum.

        The mask is asserted bit by bit against the pinned firmware's own
        enumeration, because that is the whole content of the exception: a SET bit
        skips exactly that check (AP_Arming.cpp:329-332). The origin is asserted to
        be a DATUM -- a frame definition carrying no pose -- because that is the
        difference between declaring the local frame's anchor and feeding the
        autopilot a position, which no part of this run may do with truth.
        """
        settings, window = self._settings_and_window()

        # The mask: exactly the two declared exceptions, each cited -- and the
        # PINNED name, because the plan's ARMING_CHECK does not resolve at this pin
        # (AP_Arming.cpp:199-205 renames it; the first invocation of the live run
        # measured the old name answering nothing).
        mask = window["arming_skip"]
        assert mask["name"] == check.BRING_UP_ARMING_PARAMETER == "ARMING_SKIPCHK"
        assert mask["window_value"] == (
            check.ARMING_CHECK_BIT_GPS | check.ARMING_CHECK_BIT_VISION
        )
        assert mask["window_value"] == 8 | (1 << 18)
        assert mask["restore_value"] == check.BRING_UP_ARMING_ALL_CHECKS_ENABLED == 0
        assert "SET bit SKIPS" in mask["bit_semantics"]
        assert "ARMING_SKIPCHK" in mask["bit_semantics"]
        excepted = {row["check"]: row for row in window["excepted_arming_checks"]}
        assert set(excepted) == {"Check::VISION", "Check::GPS (the home requirement)"}
        assert excepted["Check::VISION"]["bit"] == 1 << 18
        assert excepted["Check::GPS (the home requirement)"]["bit"] == 1 << 3
        for row in excepted.values():
            assert "AP_Arming" in row["citation"] and row["why"]

        # The datum: the configured home's own coordinate, a frame definition.
        datum = window["origin_datum"]
        assert datum["message_id"] == 48
        assert datum["is_a_pose_feed"] is False
        assert "DATUM" in datum["is"] and "no vehicle" in datum["is"]
        assert (datum["latitude_deg"], datum["longitude_deg"], datum["altitude_msl_m"]) == (
            pytest.approx(-35.363261),
            pytest.approx(149.165230),
            pytest.approx(584.0),
        )
        assert datum["source"] == (
            "platform.sitl_home, the coordinate the autopilot is launched with"
        )
        assert "ExternalNav" in datum["why_it_must_be_declared"]

        # Every window parameter is bounded: a window value, a restore value, a reason.
        rows = {row["name"]: row for row in window["parameter_window"]}
        assert set(rows) == {"ARMING_SKIPCHK", "EK3_SRC1_POSZ", "MOT_IDLE_SEC"}
        assert rows["ARMING_SKIPCHK"]["window_value"] == 8 | (1 << 18)
        assert rows["ARMING_SKIPCHK"]["restore_value"] == 0.0  # nothing skipped
        assert rows["EK3_SRC1_POSZ"]["window_value"] == 1.0  # baro, its own sensor
        assert rows["EK3_SRC1_POSZ"]["restore_value"] == 6.0  # ExternalNav, the seam
        # The third one is the window's thrust path's other half: the airframe's own
        # post-arm idle delay holds the motors in ground idle for longer than this
        # window's whole declared airtime once the spool state is finally asked for.
        assert rows["MOT_IDLE_SEC"]["window_value"] == 0.0  # the firmware's own default
        assert rows["MOT_IDLE_SEC"]["restore_value"] == 4.0  # compat_arming.parm's
        assert "GROUND_IDLE" in rows["MOT_IDLE_SEC"]["why"]
        assert all(row["why"] for row in rows.values())
        # The check no mask can except is named, with what the window does instead.
        assert window["not_exceptable"][0]["check"].startswith("the mandatory altitude")
        assert "mandatory_checks" in window["not_exceptable"][0]["citation"]
        # And the scored-window requirement is the restore column, exactly.
        assert {row["name"]: row["value"] for row in window["scored_window_requires"]} == {
            "ARMING_SKIPCHK": 0.0,
            "EK3_SRC1_POSZ": 6.0,
            "MOT_IDLE_SEC": 4.0,
        }

        # The thrust path is declared as a bounded LOCAL bring-up action, with the
        # frozen E-EXC climb target and the window's own airtime bound.
        thrust = window["thrust_path"]
        assert thrust["kind"] == "bounded_local_bring_up_action"
        assert "bounded local bring-up action" in thrust["statement"]
        assert "setpoint" in thrust["statement"]
        assert thrust["climb_target_m"] == check.EXCITATION_TAKEOFF_ALTITUDE_M == 0.60
        assert thrust["max_airtime_s"] == check.EXCITATION_MAX_AIRTIME_S == 5.0
        assert thrust["sent_by_the_scored_arm"] is False

        # The excitation stays inside the frozen E-EXC envelope.
        excitation = window["window"]
        assert excitation["mode"] == check.BRING_UP_MODE
        assert excitation["takeoff_altitude_m"] == check.EXCITATION_TAKEOFF_ALTITUDE_M
        assert excitation["max_airtime_s"] == check.EXCITATION_MAX_AIRTIME_S
        assert excitation["lateral_setpoint"] is None
        assert "LAND" in excitation["ends_with"]

    def test_the_scored_window_refuses_to_open_while_the_exception_is_in_force(
        self,
    ):
        """Bounded means it cannot still be in force: the readback decides.

        The closure check reads the vehicle's own answers, so a window value
        surviving to the claimed arm -- or a parameter the vehicle never answered,
        which confirms nothing -- is a blocker, and the scored window does not
        open. Beside it, the scored arm's own settings are asserted: the bridge's
        truth republish is off, because the exception is about WHEN the aircraft
        may move and never about what carries truth into the estimate.
        """
        settings, window = self._settings_and_window()
        in_force = {row["name"]: row["window_value"] for row in window["parameter_window"]}
        restored = {row["name"]: row["value"] for row in window["scored_window_requires"]}

        # Still in force: every window parameter is named, with what it should be.
        blockers = check._bring_up_closure_blockers(dict(in_force))
        assert len(blockers) == len(in_force)
        for name, window_value in in_force.items():
            matching = [blocker for blocker in blockers if blocker.startswith(name)]
            assert matching, f"{name} in force must be named: {blockers}"
            assert f"{window_value:g}" in matching[0]
            assert "restored" in matching[0] or "declared" in matching[0]

        # A silent readback is a refusal, not a pass: it cannot show the lift.
        assert check._bring_up_closure_blockers({})

        # Restored exactly: the window is closed and the scored arm may proceed.
        assert check._bring_up_closure_blockers(dict(restored)) == []

        # A third value -- neither the window's nor the declared one -- is a refusal.
        wrong = dict(restored)
        wrong["ARMING_SKIPCHK"] = -1.0  # "skip all", neither the window's nor the declared
        assert check._bring_up_closure_blockers(wrong)

        # The scored arm's own settings: truth republish off, sensor-derived mode.
        assert settings.truth_republish is False
        assert settings.sensor_mode is SensorMode.SENSOR_DERIVED
        # And it is the exception, not the arm, that carries the reason: the
        # declared exception records the truth exemption nowhere, because there is
        # none to record for this arm.
        assert "truth republish is ON" not in window["justification"]
        assert "truth republish is still off" in window["justification"]

    def test_the_window_may_send_the_bounded_throttle_override(self):
        """One channel, derived from the vehicle's own numbers, only in this window.

        The declared quantity is the pilot CLIMB RATE the position-free mode
        consumes -- the physical quantity the window's own airtime bound is about --
        and the channel value that produces it is derived from the vehicle's own
        calibration through the firmware's own arithmetic. The wire discipline is
        asserted beside it: the message carries exactly one channel and leaves every
        other field at MAVLink's own "ignore this field", the release is a zero on
        that same channel, and the scored arm's vocabulary has no RC message type in
        it at all.
        """
        settings, window = self._settings_and_window()
        thrust = window["thrust_path"]

        # Declared, with the citations a reader needs to check it.
        assert thrust["channel"] == check.RC_THROTTLE_CHANNEL == 3
        assert thrust["channel_name"] == "throttle"
        assert thrust["declared_climb_rate_ms"] == check.BRING_UP_THROTTLE_CLIMB_RATE_M_S
        assert thrust["max_climb_rate_ms"] == check.BRING_UP_THROTTLE_MAX_CLIMB_RATE_M_S
        assert thrust["refresh_s"] == check.BRING_UP_OVERRIDE_REFRESH_S > 0.0
        assert "zero" in thrust["release"] and "RC report" in thrust["release"]
        assert "why_a_rate_rather_than_a_stick" in thrust
        assert list(check.BRING_UP_THROTTLE_CALIBRATION) == [
            "RC3_MIN",
            "RC3_MAX",
            "RC3_DZ",
            "THR_DZ",
            "PILOT_SPD_UP",
        ]
        assert sorted(thrust["derived_from_the_vehicle"]) == sorted(
            check.BRING_UP_THROTTLE_CALIBRATION
        )
        assert "mode.cpp" in thrust["why_it_is_needed"]
        assert "RC_CHANNELS" in thrust["confirmed_by_the_vehicle"]

        # The derivation is the firmware's own arithmetic on the vehicle's own
        # numbers: at the pinned firmware's defaults (RC3 1100/1900, RC3_DZ 30,
        # THR_DZ 100 -- ArduCopter/radio.cpp:12-32, config.h:527 -- and
        # PILOT_SPD_UP 2.5) it reproduces the declared rate, above the deadband.
        defaults = {
            "RC3_MIN": 1100.0,
            "RC3_MAX": 1900.0,
            "RC3_DZ": 30.0,
            "THR_DZ": 100.0,
            "PILOT_SPD_UP": 2.5,
        }
        pwm, rate = check._bring_up_throttle_pwm(
            defaults, check.BRING_UP_THROTTLE_CLIMB_RATE_M_S
        )
        # get_control_mid() divides by (radio_max - radio_min - dead_zone), not by the
        # raw channel span: RC_Channel.cpp:329-340.
        mid_stick = int(1000 * ((1100 + 1900) // 2 - (1100 + 30)) / (1900 - 1100 - 30))
        assert mid_stick == 480, "the firmware's own mid stick for this calibration"
        deadband_top_control = mid_stick + int(defaults["THR_DZ"])
        deadband_top_pwm = int(
            (1100 + 30) + (1900 - 1100 - 30) * deadband_top_control / 1000
        )
        assert pwm > deadband_top_pwm, "the value must be above the deadband"
        assert rate == pytest.approx(check.BRING_UP_THROTTLE_CLIMB_RATE_M_S, abs=0.02)
        assert 0.0 < rate <= check.BRING_UP_THROTTLE_MAX_CLIMB_RATE_M_S

        # A calibration that cannot express the rate is refused, never sent: a dead
        # zone that leaves the channel no range at all, a rate the channel's own span
        # cannot reach, and a name the vehicle did not answer are all errors rather
        # than a quieter climb.
        with pytest.raises(check.ConfigError):
            check._bring_up_throttle_pwm(
                {**defaults, "RC3_DZ": 800.0}, check.BRING_UP_THROTTLE_CLIMB_RATE_M_S
            )
        with pytest.raises(check.ConfigError):
            check._bring_up_throttle_pwm(defaults, 5.0)
        with pytest.raises(check.ConfigError):
            check._bring_up_throttle_pwm(
                {name: value for name, value in defaults.items() if name != "THR_DZ"},
                check.BRING_UP_THROTTLE_CLIMB_RATE_M_S,
            )

        # The wire: exactly one channel, every other field left alone, and the release
        # is a zero on the same field.
        link = check.BringUpLink("tcp:127.0.0.1:5763", source_system=255)
        connection = _RecordingConnection()
        link._connection = connection  # no live autopilot serves a unit test
        link.target_system, link.target_component = 1, 1
        assert link.source_system == 255
        record = link.send_rc_channels_override(pwm)
        kind, arguments = connection.mav.encoded[0]
        assert kind == "RC_CHANNELS_OVERRIDE"
        assert list(arguments[2:10]) == [
            0xFFFF,
            0xFFFF,
            pwm,
            0xFFFF,
            0xFFFF,
            0xFFFF,
            0xFFFF,
            0xFFFF,
        ]
        assert all(value == 0 for value in arguments[10:]), "chan9+ are not the throttle"
        assert record["channel"] == 3 and record["pwm"] == pwm
        assert record["release"] is False and link.sent == [record]
        release = link.send_rc_channels_override(check.RC_THROTTLE_RELEASE_PWM)
        assert release["release"] is True and release["pwm"] == 0

        # The scored arm's own vocabulary carries no RC, throttle or pulse command:
        # the override's message type is in the bring-up link's set, which the
        # scored session cannot send from.
        assert "RC_CHANNELS_OVERRIDE" not in bridge.ALLOWED_OUTBOUND_TYPES
        assert "RC_CHANNELS_OVERRIDE" in bridge.BRING_UP_OUTBOUND_TYPES
        assert settings.truth_republish is False

    def test_the_scored_window_refuses_while_the_throttle_override_is_in_force(self):
        """The thrust path is a window element, so the closure gate covers it too.

        The scored window sends no RC override at all, so a window that still has
        one is the whole refusal: an override the window never released refuses, a
        release the vehicle never confirmed refuses (a silent vehicle cannot show
        that a command stopped), and the vehicle's own RC report still reading the
        override's value refuses as well. Only the vehicle's own report of the
        channel back at its radio value clears it, and the parameter rule beside it
        is unchanged.
        """
        settings, window = self._settings_and_window()
        restored = {
            row["name"]: row["value"] for row in window["scored_window_requires"]
        }
        channel = check.RC_THROTTLE_CHANNEL
        pwm = 1644

        def state(**overrides):
            record = {
                "sent": True,
                "channel": channel,
                "sent_pwm": pwm,
                "released": False,
                "observed_during_window": pwm,
                "observed_after_release": None,
            }
            record.update(overrides)
            return record

        # Still in force: the window never released it.
        blockers = check._bring_up_closure_blockers(dict(restored), state())
        assert len(blockers) == 1, blockers
        assert "never released" in blockers[0] and str(pwm) in blockers[0]

        # Released on the wire, but the vehicle never said so: not shown to be lifted.
        blockers = check._bring_up_closure_blockers(dict(restored), state(released=True))
        assert len(blockers) == 1, blockers
        assert "never answered" in blockers[0]

        # The vehicle's own report still reads the override's value: still in force.
        blockers = check._bring_up_closure_blockers(
            dict(restored), state(released=True, observed_after_release=pwm)
        )
        assert len(blockers) == 1, blockers
        assert "still reads" in blockers[0] and str(pwm) in blockers[0]

        # The vehicle's own report of the channel back at its radio value: clear.
        cleared = state(released=True, observed_after_release=1000)
        assert check._bring_up_closure_blockers(dict(restored), cleared) == []
        # A window that never sent one has nothing to close, and is not a refusal.
        assert (
            check._bring_up_closure_blockers(
                dict(restored),
                state(sent=False, sent_pwm=None, observed_during_window=None),
            )
            == []
        )
        # The override is an addition to the closure, never a replacement for it:
        # the parameter rule and the silent-readback rule still refuse beside it.
        assert check._bring_up_closure_blockers({}, cleared)
        wrong = dict(restored)
        wrong["MOT_IDLE_SEC"] = 0.0  # the window's value, not the scored arm's
        assert check._bring_up_closure_blockers(wrong, cleared)
        assert settings.sensor_mode is SensorMode.SENSOR_DERIVED
