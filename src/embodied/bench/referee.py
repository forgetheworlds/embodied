"""The bench-side stream: hidden scenario facts and their only writer.

The referee runs outside the agent (specification section 3.1): it holds the
scenario's hidden facts and records physical outcomes. Its stream is a
separate file with its own closed vocabulary (:data:`TRUTH_EVENT_KINDS`),
disjoint from the agent stream's, so the two are separate message surfaces
rather than one log split for tidiness.

This module can only *create and append* that stream — it deliberately
contains no read function. :mod:`embodied.bench.grader` is the only reader of
the stream, which is what "the grader alone reads the truth" means in code.
The stream is written once per episode and refuses to exist up front, so a
completed record cannot be quietly extended after the fact.

Schema revision of the envelope: EVENTS_REVISION (``embodied.bench.events``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied.contracts.records import ClockStamp
from embodied.bench.events import (
    TRUTH_EVENT_KINDS,
    Event,
    EpisodeError,
    append_event_line,
    build_event,
    check_stamp_order,
)

TRUTH_EVENTS_FILENAME = "truth-events.jsonl"

__all__ = [
    "Referee",
    "TRUTH_EVENT_KINDS",
    "TRUTH_EVENTS_FILENAME",
]


class Referee:
    """Writer for the one bench-side stream of an episode.

    Records are held in memory and written in one pass by :meth:`close`, so
    the file appears complete or not at all, and only for a fresh episode.
    """

    def __init__(self, episode_dir: Path) -> None:
        self.episode_dir = Path(episode_dir)
        if (self.episode_dir / TRUTH_EVENTS_FILENAME).exists():
            raise EpisodeError(
                f"{self.episode_dir} already has a bench-side stream; it is written once"
            )
        self._events: list[Event] = []
        self._last_stamp: ClockStamp | None = None
        self._closed = False

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def record(
        self,
        kind: str,
        payload: Any,
        stamp: ClockStamp,
        sim_time_s: float | None = None,
    ) -> Event:
        """Stage one bench-side event; validation happens at construction."""
        if self._closed:
            raise EpisodeError("the bench-side stream is already written")
        if kind not in TRUTH_EVENT_KINDS:
            raise EpisodeError(f"{kind!r} is not a bench-side event kind")
        check_stamp_order(self._last_stamp, stamp)
        event = build_event(len(self._events), kind, payload, stamp, sim_time_s)
        self._events.append(event)
        self._last_stamp = stamp
        return event

    def close(self) -> Path:
        """Write the whole stream once, or refuse."""
        if self._closed:
            raise EpisodeError("the bench-side stream is already written")
        if not self._events:
            raise EpisodeError("the referee recorded nothing; an empty bench-side stream is refused")
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        path = self.episode_dir / TRUTH_EVENTS_FILENAME
        # "x" closes the race the existence check in __init__ only hints at:
        # this process creates the file, never overwrites one.
        with path.open("x", encoding="utf-8"):
            pass
        for event in self._events:
            append_event_line(path, event)
        self._closed = True
        return path
