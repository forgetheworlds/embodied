"""The mission-loop watchdog: silence gets a detector, a hang gets a name.

J50-move-1 hung 23 minutes with no receipt, no mission.json, no shutdown
record, because every declared bound is consulted BETWEEN computations and
the computation that hung was never checked by anything. These tests pin the
supervision that ends that failure mode, each against the runtime the mission
actually flies:

* The watchdog is wired into run(): started inside the try, stopped in the
  finally before the shutdown, and its exception unwinds the loop without
  costing the run its receipt.
* A loop that stops beating is named: fire writes every thread's stack into
  the run's own artifacts, names the termination, and breaks the loop by
  raising MissionLoopStalled on the loop's own thread.
* A beating loop never fires, and a stopped watchdog never fires.
* Fire-then-escalate ordering, with clock and sleep injected: a watchdog
  test must not measure the real machine (LEARNED-FAILURES T18). The only
  real waiting here is a bounded await on another thread's progress flag,
  never on a value of time.
"""

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path
from typing import Callable

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


def _runtime(tmp_path, *, arm: str = "B0"):
    """The runtime as the other platform tests build it: real settings, real
    recorder paths under tmp_path, no simulator, no network."""
    from embodied.bench.recorder import Recorder
    from embodied.platform.localization_check import (
        _load_localization_config,
        _platform_settings,
    )
    from embodied.platform.mission_runtime import MissionRuntime

    document = _load_localization_config(REPOSITORY / "configs" / "first_indoor.yaml")
    settings = _platform_settings(document, REPOSITORY)
    episode_dir = tmp_path / "episode"
    recorder = Recorder(episode_dir)
    return MissionRuntime(
        settings=settings,
        config_document=document,
        episode_dir=episode_dir,
        evidence_dir=tmp_path / "platform",
        recorder=recorder,
        instruction="Find the red block, inspect it, and return to the start.",
        target_id="red_block",
        episode_id="watchdog-0",
        arm=arm,
    )


class _FakeClock:
    """A clock the test drives. ``sleep`` never waits; the test advances time."""

    def __init__(self) -> None:
        self.now_s = 0.0

    def monotonic(self) -> float:
        return self.now_s

    def sleep(self, _seconds: float) -> None:
        time.sleep(0)  # a GIL yield, never a timed wait

    def advance(self, seconds: float) -> None:
        self.now_s += seconds


def _await(predicate: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    """Bounded wait for another thread's progress, not for a value of time."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.001)
    return True


def _watchdog(clock: _FakeClock, fire, escalate, *, stall_s: float = 90.0):
    from embodied.platform.mission_runtime import _MissionLoopWatchdog

    return _MissionLoopWatchdog(
        thread_id=threading.get_ident(),
        fire=fire,
        escalate=escalate,
        stall_s=stall_s,
        poll_s=0.005,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


# ---------------------------------------------------------------------------
# the exception the loop must not be able to absorb
# ---------------------------------------------------------------------------


def test_the_stall_is_not_an_exception_the_loop_can_absorb():
    from embodied.platform import mission_runtime as mr

    assert issubclass(mr.MissionLoopStalled, BaseException)
    assert not issubclass(mr.MissionLoopStalled, Exception)


# ---------------------------------------------------------------------------
# fire-then-escalate, on an injected clock
# ---------------------------------------------------------------------------


def test_the_watchdog_fires_once_then_escalates_one_window_later():
    clock = _FakeClock()
    fired: list[float] = []
    escalated: list[float] = []
    watchdog = _watchdog(clock, fired.append, escalated.append)

    watchdog.start()
    clock.advance(95.0)  # one bound past the last beat, with none coming
    assert _await(lambda: len(fired) == 1), "the stall was never fired"
    assert not escalated, "escalation ran before a full second window"

    clock.advance(95.0)  # a second full window with no beat
    assert _await(lambda: len(escalated) == 1), "the hang was never escalated"
    watchdog.stop()
    watchdog.join(timeout_s=2.0)

    assert len(fired) == 1
    assert len(escalated) == 1
    assert fired[0] == pytest.approx(95.0, abs=1e-9)
    assert escalated[0] == pytest.approx(190.0, abs=1e-9)
    assert watchdog._thread is not None and not watchdog._thread.is_alive()


def test_a_beat_defers_the_fire():
    clock = _FakeClock()
    fired: list[float] = []
    escalated: list[float] = []
    watchdog = _watchdog(clock, fired.append, escalated.append)

    watchdog.start()
    for _ in range(50):  # a turn every 30 s of fake time, well inside the bound
        clock.advance(30.0)
        watchdog.beat()
    assert not fired, "a beating loop fired"
    assert not escalated

    clock.advance(95.0)  # the loop stops turning
    assert _await(lambda: len(fired) == 1)
    watchdog.stop()
    watchdog.join(timeout_s=2.0)
    assert not escalated, "escalation ran without its own full window"


def test_a_failing_fire_still_escalates_and_the_thread_survives():
    clock = _FakeClock()
    escalated: list[float] = []

    def failing_fire(_age_s: float) -> None:
        raise RuntimeError("the artifact write failed")

    watchdog = _watchdog(clock, failing_fire, escalated.append)
    watchdog.start()
    clock.advance(95.0)
    assert _await(lambda: len(escalated) == 1), (
        "a failed fire must hand the ending to escalation"
    )
    watchdog.stop()
    watchdog.join(timeout_s=2.0)
    assert watchdog._thread is not None and not watchdog._thread.is_alive(), (
        "the watchdog thread must never raise out of _run()"
    )


def test_a_stopped_watchdog_neither_fires_nor_escalates():
    clock = _FakeClock()
    fired: list[float] = []
    escalated: list[float] = []
    watchdog = _watchdog(clock, fired.append, escalated.append)

    watchdog.start()
    watchdog.stop()
    watchdog.join(timeout_s=2.0)
    clock.advance(1000.0)  # far past the bound, with the thread already gone
    assert not fired
    assert not escalated


# ---------------------------------------------------------------------------
# the runtime-level wiring, on the runtime the mission actually flies
# ---------------------------------------------------------------------------


def test_a_stalled_loop_is_recorded_named_and_broken(tmp_path):
    from embodied.platform import mission_runtime as mr

    runtime = _runtime(tmp_path)
    clock = _FakeClock()
    runtime._watchdog_stall_s = 0.5
    runtime._watchdog_poll_s = 0.005
    runtime._watchdog_monotonic = clock.monotonic
    runtime._watchdog_sleep = clock.sleep

    started = threading.Event()
    blocked = threading.Event()
    delivered: list[str] = []

    def fake_loop() -> None:
        try:
            runtime._watchdog_start()
            started.set()
            for _ in range(4):
                runtime._beat_watchdog()
                blocked.wait(timeout=0.005)
            # The loop stops beating and blocks, as J50's did.
            while not blocked.is_set():
                blocked.wait(timeout=0.005)
        except mr.MissionLoopStalled:
            delivered.append("stalled")

    thread = threading.Thread(target=fake_loop, daemon=True)
    thread.start()
    assert started.wait(timeout=5.0)

    artifact = runtime.evidence.path("mission-loop-stall.txt")

    def advance_until(predicate, timeout_s: float = 5.0) -> bool:
        # Drive the fake clock while awaiting the watchdog thread; the only
        # real waiting is for another thread's progress, never time itself.
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if time.monotonic() >= deadline:
                return False
            clock.advance(0.05)
            time.sleep(0.001)
        return True

    assert advance_until(artifact.exists), (
        "the stall was never written to the artifacts"
    )
    # The run is ending here: stop the watchdog before its second window.
    runtime._watchdog_stop()
    blocked.set()
    thread.join(timeout=5.0)

    text = artifact.read_text(encoding="utf-8")
    assert "mission loop stalled" in text
    assert "thread 0x" in text.lower(), "the dump must carry the threads' stacks"
    assert runtime.result.termination_reason == "mission_loop_stalled"
    assert delivered == ["stalled"], (
        "MissionLoopStalled must be raised into the loop's own thread"
    )
    assert not thread.is_alive()


def test_a_beating_loop_never_fires(tmp_path):
    runtime = _runtime(tmp_path)
    clock = _FakeClock()
    runtime._watchdog_stall_s = 0.5
    runtime._watchdog_poll_s = 0.005
    runtime._watchdog_monotonic = clock.monotonic
    runtime._watchdog_sleep = clock.sleep

    runtime._watchdog_start()
    for _ in range(300):  # turns every 0.02 s of fake time across many windows
        clock.advance(0.02)
        runtime._beat_watchdog()
    runtime._watchdog_stop()

    assert not runtime.evidence.path("mission-loop-stall.txt").exists()
    assert runtime.result.termination_reason == "not_started"


def test_run_wires_the_watchdog_onto_the_mission_loop():
    """run() needs a simulator to execute, so the wiring itself is asserted
    here: started inside the try, its exception absorbed without losing the
    receipt, stopped in the finally before the shutdown, and beats on every
    long silent path the design names."""
    from embodied.platform import mission_runtime as mr

    run_source = inspect.getsource(mr.MissionRuntime.run)
    assert "self._watchdog_start()" in run_source
    assert "except MissionLoopStalled" in run_source
    finally_source = run_source.split("finally:", 1)[1]
    stop_position = finally_source.index("self._watchdog_stop()")
    shutdown_position = finally_source.index("self._shutdown(")
    assert stop_position < shutdown_position, (
        "the watchdog must be stopped before the shutdown, or a healthy run's "
        "shutdown can be killed mid-receipt by its own escalation"
    )
    drain_body = run_source.split("def drain", 1)[1].split(
        "bring_up_link: BringUpLink", 1
    )[0]
    assert "self._beat_watchdog()" in drain_body, "the bring-up drain must beat"

    for method in ("perceive_if_due", "_fly_the_mission", "observe_in_place"):
        source = inspect.getsource(getattr(mr.MissionRuntime, method))
        assert "self._beat_watchdog()" in source, f"{method} must beat"
    step_source = inspect.getsource(mr._LiveRunnerWorld.step)
    assert "runtime._beat_watchdog()" in step_source, "the lease loop must beat"
