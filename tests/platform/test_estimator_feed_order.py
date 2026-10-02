"""The estimator feed's ordering rule: an image never overtakes its inertial samples.

Earned on `live-14` and `live-15`, where the aircraft armed, took off, and then lost
its estimate: the pinned estimator's own log carries 22 and 27 occurrences of
``Propagator::select_imu_readings(): No IMU measurements to propagate with (0 of 2)``,
its state freezes at 2.90 m of integrated distance and then runs to 47 m inside a 6 m
room, EKF3 follows the published pose, the controller floors the throttle to zero, and
the firmware's own auto-disarm takes the vehicle out of the air while still in GUIDED --
so every mission setpoint was afterwards refused. A gate-passing P01-L flight on the
same stack has zero occurrences.

The feed has two queues. The reader's sink files a stereo pair the moment the reader's
thread sees it, while the inertial sample that follows that pair in the stream is handed
over a moment later on the metadata queue. Draining the pairs blindly therefore lets a
camera frame overtake the samples that must precede it, and the pin's propagator finds
an empty integration window. The pin absorbs a *late* image by interpolating the final
segment of its interval (open_vins-2.7, Propagator.cpp, case 3.4); it does not absorb an
image with no interval behind it. So the oldest pair the inertial feed has not passed
waits in the guard's own slot, and the queue itself is never reordered.
"""

from __future__ import annotations

import queue

from embodied.platform.localization_check import OrderedPairFeed, pair_is_behind_imu


class _Stamp:
    def __init__(self, monotonic_ns: int) -> None:
        self.monotonic_ns = monotonic_ns


class _Record:
    """The two fields the guard reads: the frame's own instant and its receipt."""

    def __init__(self, sim_time_ns: int, received_ns: int = 0) -> None:
        self.sim_time_ns = sim_time_ns
        self.received_stamp = _Stamp(received_ns)


class _Stats:
    """The counters the guard keeps, as ``_FeedStats`` declares them."""

    def __init__(self) -> None:
        self.pairs_held_for_imu = 0
        self.max_pair_hold_s = 0.0


def _guard(pending, fed, stats=None):
    stats = stats or _Stats()
    return (
        OrderedPairFeed(
            pending,
            sim_time_ns_of=lambda record: record.sim_time_ns,
            feed_one=fed.append,
            stats=stats,
        ),
        stats,
    )


def test_an_image_is_behind_the_inertial_feed_only_once_a_later_sample_has_gone() -> None:
    # Strictly past: the pin interpolates its final segment with the first sample at or
    # after the image's instant, so a sample exactly at that instant is not yet enough.
    assert pair_is_behind_imu(1_000, 1_002) is True
    assert pair_is_behind_imu(1_000, 1_000) is False
    assert pair_is_behind_imu(1_002, 1_000) is False


def test_a_pair_the_inertial_feed_has_passed_is_fed() -> None:
    pending = queue.Queue()
    pending.put_nowait(_Record(1_000))
    fed: list = []
    guard, stats = _guard(pending, fed)

    guard.drain(newest_imu_ns=1_002)

    assert [record.sim_time_ns for record in fed] == [1_000]
    assert pending.empty()
    assert stats.pairs_held_for_imu == 0


def test_the_queue_is_never_reordered_by_a_waiting_frame() -> None:
    # The regression this guard must not have: holding a frame by putting it back on the
    # queue reorders the stream. The waiting frame lives in the guard's own slot instead,
    # so the queue keeps its order and the frame that waits goes before anything filed
    # behind it.
    pending = queue.Queue()
    pending.put_nowait(_Record(1_000))
    pending.put_nowait(_Record(2_000))
    pending.put_nowait(_Record(3_000))
    fed: list = []
    guard, _ = _guard(pending, fed)

    guard.drain(newest_imu_ns=1_500)

    assert [record.sim_time_ns for record in fed] == [1_000]
    assert pending.qsize() == 1

    guard.drain(newest_imu_ns=9_000)

    assert [record.sim_time_ns for record in fed] == [1_000, 2_000, 3_000]


def test_a_pair_ahead_of_the_inertial_feed_is_held_and_not_fed() -> None:
    pending = queue.Queue()
    pending.put_nowait(_Record(5_000))
    fed: list = []
    guard, stats = _guard(pending, fed)

    guard.drain(newest_imu_ns=4_000)

    assert fed == []
    assert stats.pairs_held_for_imu == 1




def test_the_waiting_frame_is_fed_before_the_newer_frames_behind_it() -> None:
    pending = queue.Queue()
    pending.put_nowait(_Record(5_000))
    pending.put_nowait(_Record(6_000))
    fed: list = []
    guard, _ = _guard(pending, fed)

    guard.drain(newest_imu_ns=4_000)  # 5_000 waits; 6_000 is never even taken
    assert fed == []

    guard.drain(newest_imu_ns=6_002)  # both are behind now, oldest first

    assert [record.sim_time_ns for record in fed] == [5_000, 6_000]
    assert pending.empty()


def test_a_hold_is_recorded_once_and_measures_the_wait_from_the_receipt() -> None:
    pending = queue.Queue()
    pending.put_nowait(_Record(5_000, received_ns=1_000_000_000))
    fed: list = []
    guard, stats = _guard(pending, fed)

    guard.drain(newest_imu_ns=4_000)
    guard.drain(newest_imu_ns=4_000)
    guard.drain(newest_imu_ns=4_000)

    # Three cycles of waiting are one held frame, not three.
    assert stats.pairs_held_for_imu == 1
    assert stats.max_pair_hold_s > 0.0


def test_a_frame_that_waited_is_still_fed_once_its_samples_arrive() -> None:
    pending = queue.Queue()
    pending.put_nowait(_Record(5_000))
    fed: list = []
    guard, stats = _guard(pending, fed)

    guard.drain(newest_imu_ns=4_000)
    guard.drain(newest_imu_ns=5_002)

    assert [record.sim_time_ns for record in fed] == [5_000]
    assert stats.pairs_held_for_imu == 1
