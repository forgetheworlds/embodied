"""The perception queue's loading discipline.

The queue that feeds depth, candidate proposal and map integration was declared
and drained but never fed on the first flight the mission ever made. Nothing
failed loudly: ``perceive_if_due`` simply found the queue empty and returned, so
no observation, no map, no candidate and no frontier were ever produced, and the
mission reported ``explore: blocked`` against its own empty map.

These tests hold the loading rule that defect was missing: a full queue gives up
its OLDEST frame rather than refusing the newest (specification 3.1, "drop
replaceable old image snapshots rather than allow backlog"), and it never
blocks the thread the frames arrive on.
"""

from __future__ import annotations

import queue

from embodied.platform.mission_runtime import MissionRuntime


class _QueueHolder:
    """The two attributes ``_offer_to_perception`` actually uses.

    The method is called unbound against this stub so the rule can be tested
    without constructing a runtime, whose construction reads the whole platform
    configuration.
    """

    def __init__(self, maxsize: int) -> None:
        self._perception_queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._perception_frames_dropped = 0

    def drain(self) -> list[object]:
        drained: list[object] = []
        while True:
            try:
                drained.append(self._perception_queue.get_nowait())
            except queue.Empty:
                return drained


def test_a_frame_offered_to_an_empty_queue_is_retrievable() -> None:
    """The defect in its simplest form: an offered pair must come back out."""
    holder = _QueueHolder(maxsize=4)
    MissionRuntime._offer_to_perception(holder, "pair-1")
    assert holder.drain() == ["pair-1"]
    assert holder._perception_frames_dropped == 0


def test_a_full_queue_drops_its_oldest_frame_and_keeps_the_newest() -> None:
    """Newest wins. The map should be built from the current frame, not a stale one."""
    holder = _QueueHolder(maxsize=2)
    for frame in ("pair-1", "pair-2", "pair-3"):
        MissionRuntime._offer_to_perception(holder, frame)
    assert holder.drain() == ["pair-2", "pair-3"]
    assert holder._perception_frames_dropped == 1


def test_offering_never_blocks_the_calling_thread() -> None:
    """The feed thread must not stall behind a full perception queue."""
    holder = _QueueHolder(maxsize=1)
    for frame in range(50):
        MissionRuntime._offer_to_perception(holder, f"pair-{frame}")
    assert holder.drain() == ["pair-49"]
    assert holder._perception_frames_dropped == 49
