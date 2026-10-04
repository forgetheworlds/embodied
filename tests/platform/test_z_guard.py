"""The R24 z guard: a publisher-side bound on |vision z − baro z|.

The FC's altitude channel IS the estimator's z (EK3_SRC1_POSZ 6), so a live but
wrong estimate was indistinguishable from a moving aircraft all the way to the
crash detector (work/runs/night/Z-CLIMB.md, J53-vantage-2). The guard compares
each state about to be published against the vehicle's own barometer, stops the
feed past the declared residual, and names the divergence in the run's record.
These tests prove the declared behaviour, not the flight (R3): the falsifier is
one flight, named in work/runs/night/Z-GUARD.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
REAL_PLATFORM_CONFIG = REPOSITORY / "configs" / "first_indoor.yaml"


class _Clock:
    """The pacing clock: simulated time under the test's hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _machine():
    from embodied.platform import localization as loc

    return loc.HealthMachine(
        loc.HealthBounds(
            publish_period_s=0.01,
            state_lost_after_s=0.3,
            published_state_age_max_s=0.04,
            max_publish_gap_s=0.1,
            visual_update_warn_s=0.3,
            visual_update_fail_s=0.5,
            valid_fraction_min=0.99,
            sigma_min_m=0.02,
            sigma_max_m=1.0,
        )
    )


def _publisher(clock: _Clock, machine=None):
    from embodied.platform import localization as loc

    publisher = loc.ExternalNavPublisher(
        "irrelevant-not-opened",
        loc.OdomAlignment((0.0, 0.0, 0.0)),
        machine or _machine(),
        clock=clock,
    )
    # The publish loop seals the epoch rotation on the first state it accepts.
    # Under the alignment's OV→FRD conjugation this quaternion seals the
    # identity rotation, so the aligned wire z equals the odom z here.
    publisher._alignment.seal((0.0, 1.0, 0.0, 0.0))
    return publisher


def _state(position_m: tuple[float, float, float], time_ns: int):
    from embodied.platform import localization as loc

    return loc.EstimatorState(
        time_ns=time_ns,
        initialized=True,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_m=position_m,
        velocity_mps=(0.0, 0.0, 0.0),
        gyro_bias=(0.0, 0.0, 0.0),
        accel_bias=(0.0, 0.0, 0.0),
        sigma_pos_m=(0.1, 0.1, 0.1),
        n_tracks=40,
        t_last_visual_ns=time_ns - 50_000_000,
        reset_counter=0,
    )


def _spend_offset_window(publisher, clock: _Clock, vision_z: float, baro_alt: float):
    """Publish one state per tick until the declared window has closed.

    The identity alignment sends position z to the wire unchanged, so the
    published vision z is exactly ``vision_z`` and every window sample is
    ``(-vision_z) - baro_alt``.
    """
    publisher.offer_baro(baro_alt)
    ticks = 0
    while not publisher._z_guard_window_closed:
        clock.now += 0.01
        ticks += 1
        assert ticks < 5000, "the window never closed"
        # The live drain hands the guard a fresh reading at ~5 Hz; so does this.
        publisher.offer_baro(baro_alt)
        publisher._z_guard_note_publication(
            _state((0.0, 0.0, vision_z), int(clock.now * 1e9)), clock.now
        )
    assert publisher._z_guard_window_samples > 0


# ---------------------------------------------------------------------------
# The healthy direction: inside-band disagreement publishes unchanged.
# ---------------------------------------------------------------------------


def test_inside_band_disagreement_publishes_unchanged():
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    _spend_offset_window(publisher, clock, vision_z=-5.1, baro_alt=5.0)
    # Offset: (-(-5.1)) - 5.0 = 0.1 m. A further 0.2 m of disagreement is inside
    # the declared 0.6 m bound (the measured healthy worst is 0.2 m, Z-CLIMB.md).
    state = _state((0.0, 0.0, -5.3), int(clock.now * 1e9))
    publisher.offer(state, 0)
    assert publisher.state_for_publish(clock.now) is state
    assert publisher.z_guard_receipt_line() is None


# ---------------------------------------------------------------------------
# The refusal: divergence beyond the band stops the feed and names the cause.
# ---------------------------------------------------------------------------


def test_beyond_band_stops_and_names_the_divergence():
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    _spend_offset_window(publisher, clock, vision_z=-5.1, baro_alt=5.0)
    # J53's shape: the estimate's z jumps 1.2 m in one step — past the bound.
    state = _state((0.0, 0.0, -6.3), int(clock.now * 1e9))
    publisher.offer(state, 0)
    assert publisher.state_for_publish(clock.now) is None
    assert machine.state == "stopped"
    # The stop is named: magnitudes, stamps and the bound are in the record.
    (stop_event,) = [e for e in machine.events if e.event == "stopped"]
    assert "z guard" in stop_event.detail
    assert "1.200 m" in stop_event.detail
    assert "0.600 m" in stop_event.detail
    (trip,) = publisher._z_guard_trips
    assert trip["residual_m"] == pytest.approx(1.2, abs=1e-3)
    assert trip["bound_m"] == pytest.approx(0.6)
    assert trip["vision_z_ned_m"] == pytest.approx(-6.3, abs=1e-3)
    assert trip["baro_alt_m"] == pytest.approx(5.0, abs=1e-3)
    assert trip["offset_m"] == pytest.approx(0.1, abs=1e-3)
    assert trip["vision_time_ns"] == state.time_ns
    # The receipt line is handed over exactly once, and says why the feed stopped.
    line = publisher.z_guard_receipt_line()
    assert line is not None
    assert "|vision z − baro z| = 1.200 m exceeds the declared bound 0.600 m" in line
    assert "R24" in line
    assert publisher.z_guard_receipt_line() is None


def test_a_single_inside_band_spike_does_not_stop_the_feed():
    """J53's first resumption step measured 0.48 m: inside the band, published.

    The populations separate cleanly (healthy <= 0.2 m, divergence >= 1 m in
    4 s), so the bound needs no debounce — a single spike inside it publishes.
    """
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    _spend_offset_window(publisher, clock, vision_z=-5.1, baro_alt=5.0)
    state = _state((0.0, 0.0, -5.58), int(clock.now * 1e9))
    publisher.offer(state, 0)
    assert publisher.state_for_publish(clock.now) is state
    assert publisher._z_guard_trips == []


# ---------------------------------------------------------------------------
# The latch: a breach never silently re-arms.
# ---------------------------------------------------------------------------


def test_a_fresh_state_after_the_stop_does_not_reopen_the_wire():
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    _spend_offset_window(publisher, clock, vision_z=-5.1, baro_alt=5.0)
    publisher.offer(_state((0.0, 0.0, -7.1), int(clock.now * 1e9)), 0)
    assert publisher.state_for_publish(clock.now) is None
    # The publish loop feeds a refusal as on_state(None): the machine stays
    # stopped and records no recovery.
    machine.on_state(None, int(clock.now * 1e9))
    assert machine.state == "stopped"
    assert not [e for e in machine.events if e.event == "recovered"]
    # Even a machine forced back to healthy — the shape a fresh initialized
    # state would drive — cannot re-open the wire: the refusal is the
    # publisher's own memory, so a recovered feed cannot flap.
    healthy = _state((0.0, 0.0, -5.1), int((clock.now + 1.0) * 1e9))
    machine._recover(healthy, int(clock.now * 1e9))
    publisher.offer(healthy, 0)
    assert publisher.state_for_publish(clock.now + 1.0) is None


# ---------------------------------------------------------------------------
# The offset window: before it closes the guard measures and never gates.
# ---------------------------------------------------------------------------


def test_before_the_window_closes_nothing_is_gated():
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    publisher.offer_baro(5.0)
    state = _state((0.0, 0.0, -7.5), int(clock.now * 1e9))
    publisher.offer(state, 0)
    assert publisher.state_for_publish(clock.now) is state


def test_the_window_only_collects_fresh_pairs():
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    publisher.offer_baro(5.0)
    clock.now += 1.0  # the barometer reference is now older than the 0.5 s bound
    publisher._z_guard_note_publication(
        _state((0.0, 0.0, -5.1), int(clock.now * 1e9)), clock.now
    )
    assert publisher._z_guard_window_samples == 0


def test_a_window_with_no_reference_says_so():
    """A guard that never had a reference is named, never silent (R24)."""
    clock = _Clock()
    publisher = _publisher(clock)
    publisher._machine.open_window(int(clock.now * 1e9))
    # The window starts at the first published state and closes after its
    # declared length whether or not a reference ever arrived.
    publisher._z_guard_note_publication(
        _state((0.0, 0.0, -5.1), int(clock.now * 1e9)), clock.now
    )
    clock.now += 11.0
    publisher._z_guard_note_publication(
        _state((0.0, 0.0, -5.1), int(clock.now * 1e9)), clock.now
    )
    assert publisher._z_guard_window_closed
    assert publisher._z_guard_offset_m is None
    line = publisher.z_guard_receipt_line()
    assert line is not None
    assert "no barometric reference" in line


# ---------------------------------------------------------------------------
# Staleness and the parked phase gate nothing.
# ---------------------------------------------------------------------------


def test_a_stale_reference_gates_nothing():
    """A frozen barometer must not false-trip a healthy climb or descent."""
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    _spend_offset_window(publisher, clock, vision_z=-5.1, baro_alt=5.0)
    clock.now += 1.0  # the barometer feed died; the reference is stale
    state = _state((0.0, 0.0, -7.5), int(clock.now * 1e9))
    publisher.offer(state, 0)
    assert publisher.state_for_publish(clock.now) is state


def test_the_parked_phase_gates_nothing():
    """The sigma bound's own shape: the guard protects the flying wire only."""
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    _spend_offset_window(publisher, clock, vision_z=-5.1, baro_alt=5.0)
    assert machine.window_declared is False
    state = _state((0.0, 0.0, -8.5), int(clock.now * 1e9))
    publisher.offer(state, 0)
    assert publisher.state_for_publish(clock.now) is state


# ---------------------------------------------------------------------------
# The baro conversion is ArduPilot's own model, hand-checked.
# ---------------------------------------------------------------------------


def test_baro_relative_altitude_matches_the_pinned_firmware_model():
    from embodied.platform.webots_ardupilot import baro_relative_altitude_m

    # get_altitude_difference_simple (AP_Baro_atmosphere.cpp:54-67) at T0 =
    # 21.5 degC (294.65 K), p0 = 975.00 hPa, p = 974.988 hPa:
    #   153.8462 * 294.65 * (1 - (974.988/975.0)^0.190259) = 0.10615... m
    value = baro_relative_altitude_m(974.988, 2150, 975.0, 2150)
    assert value == pytest.approx(0.10615, rel=1e-3)
    # The datum itself reads zero, and lower pressure reads higher.
    assert baro_relative_altitude_m(975.0, 2150, 975.0, 2150) == pytest.approx(
        0.0, abs=1e-9
    )
    assert baro_relative_altitude_m(974.9, 2150, 975.0, 2150) > 0.0


# ---------------------------------------------------------------------------
# The configuration declares the bound, and the runtime carries it (R2).
# ---------------------------------------------------------------------------


def _runtime(tmp_path, episode_id: str):
    from embodied.bench.recorder import Recorder
    from embodied.platform.localization_check import (
        _load_localization_config,
        _platform_settings,
    )
    from embodied.platform.mission_runtime import MissionRuntime

    document = _load_localization_config(REAL_PLATFORM_CONFIG)
    settings = _platform_settings(document, REPOSITORY)
    episode_dir = tmp_path / "episode"
    return MissionRuntime(
        settings=settings,
        config_document=document,
        episode_dir=episode_dir,
        evidence_dir=tmp_path / "platform",
        recorder=Recorder(episode_dir),
        instruction="Find the red block, inspect it, and return to the start.",
        target_id="red_block",
        episode_id=episode_id,
        arm="B0",
    )


def test_the_config_declares_the_bound_and_the_runtime_carries_it(tmp_path):
    runtime = _runtime(tmp_path, "zguard-config")
    bounds = runtime._machine.bounds
    assert bounds.z_guard_max_residual_m == pytest.approx(0.6)
    assert bounds.z_guard_offset_window_s == pytest.approx(10.0)
    assert bounds.z_guard_baro_stale_after_s == pytest.approx(0.5)


def test_the_runtime_hands_the_barometer_to_the_guard(tmp_path):
    runtime = _runtime(tmp_path, "zguard-baro")
    clock = _Clock()
    publisher = _publisher(clock, runtime._machine)
    runtime._publisher = publisher
    from embodied.platform.webots_ardupilot import TelemetrySample

    from embodied.contracts.records import ClockStamp

    stamp = ClockStamp(host_id="zguard-test-0", clock_id="monotonic", monotonic_ns=0)
    empty = TelemetrySample(
        received_stamp=stamp,
        messages_seen=0,
        heartbeats=0,
        mode_name=None,
        custom_mode=None,
        armed=None,
        system_status=None,
        attitude_rpy=None,
        local_position_ned=None,
        velocity_ned=None,
        servo_outputs=None,
        ekf_flags=None,
        ekf_velocity_variance=None,
        ekf_pos_horiz_variance=None,
        statustexts=(),
        boot_time_ms=None,
        home_position=None,
        autopilot_version=None,
        parameters={},
        press_abs_hpa=None,
        press_temp_cdegc=None,
    )
    runtime._offer_baro(empty)
    assert publisher._baro is None  # no pressure, nothing invented
    sample = TelemetrySample(
        **{
            **empty.__dict__,
            "press_abs_hpa": 975.0,
            "press_temp_cdegc": 2150,
        }
    )
    runtime._offer_baro(sample)
    assert publisher._baro is None  # the datum itself is absorbed, not handed over
    second = TelemetrySample(
        **{**empty.__dict__, "press_abs_hpa": 974.988, "press_temp_cdegc": 2150}
    )
    runtime._offer_baro(second)
    assert publisher._baro is not None
    assert publisher._baro[0] == pytest.approx(0.10615, rel=1e-3)


def test_the_mission_log_says_why_the_feed_stopped(tmp_path):
    """R24: the run's own record names the divergence, exactly once."""
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    _spend_offset_window(publisher, clock, vision_z=-5.1, baro_alt=5.0)
    publisher.offer(_state((0.0, 0.0, -6.5), int(clock.now * 1e9)), 0)
    assert publisher.state_for_publish(clock.now) is None

    runtime = _runtime(tmp_path, "zguard-surface")
    runtime._publisher = publisher
    assert runtime.publish_active() == "no_active_goal"
    (line,) = [entry for entry in runtime.result.log if "vision feed stopped" in entry]
    assert "1.400 m" in line
    assert "0.600 m" in line
    # A second tick must not repeat it: the line is handed over once.
    runtime.publish_active()
    assert len([e for e in runtime.result.log if "vision feed stopped" in e]) == 1


def test_the_end_of_run_record_also_carries_the_line(tmp_path):
    """A trip between setpoint ticks is surfaced by the run's own end."""
    clock = _Clock()
    publisher = _publisher(clock)
    machine = publisher._machine
    machine.open_window(int(clock.now * 1e9))
    _spend_offset_window(publisher, clock, vision_z=-5.1, baro_alt=5.0)
    publisher.offer(_state((0.0, 0.0, -6.5), int(clock.now * 1e9)), 0)
    assert publisher.state_for_publish(clock.now) is None

    runtime = _runtime(tmp_path, "zguard-end")
    runtime._publisher = publisher
    runtime._surface_refusals()
    (line,) = [entry for entry in runtime.result.log if "vision feed stopped" in entry]
    assert "R24" in line
