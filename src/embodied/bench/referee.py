"""The bench-side store: hidden scenario facts, their path and their only writer.

The referee runs outside the agent (specification section 3.1): it holds the
scenario's hidden facts and records physical outcomes. The facts live in a
store directory *beside* the episode directory, never inside it, because the
episode directory is what the runtime side is handed (specification 3.2:
separate storage, not only a separate surface). :func:`truth_store_path` is
the single resolver for that location, and it refuses a store that would
resolve inside the episode — that one check is what makes the isolation
structural rather than a naming convention.

The store's stream has its own closed vocabulary (:data:`TRUTH_EVENT_KINDS`),
disjoint from the agent stream's, so the two are separate message surfaces as
well as separate storage.

This module can only resolve the store's path and append to its stream — it
deliberately contains no read function. :mod:`embodied.bench.grader` is the
only reader of the stream, which is what "the grader alone reads the truth"
means in code. The stream is written once per episode and refuses to exist up
front, so a completed record cannot be quietly extended after the fact.

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

TRUTH_STORE_SUFFIX = ".truth"
TRUTH_EVENTS_FILENAME = "truth-events.jsonl"

__all__ = [
    "Referee",
    "StoreMissing",
    "TRUTH_EVENT_KINDS",
    "TRUTH_EVENTS_FILENAME",
    "TRUTH_STORE_SUFFIX",
    "truth_store_path",
]


class StoreMissing(Exception):
    """The bench-side store does not exist: a missing prerequisite, not a bad score."""


def truth_store_path(episode_dir: Path) -> Path:
    """Where one episode's hidden facts live: a sibling directory named after it.

    The episode directory is handed to the runtime side, so its store must
    never resolve inside it. A store that lands inside — for instance a
    sibling name that is itself a link into the episode — is refused here,
    before any writer or reader can use it (specification 3.2).
    """
    episode = Path(episode_dir).resolve()
    store = episode.parent / (episode.name + TRUTH_STORE_SUFFIX)
    resolved = store.resolve()
    if resolved == episode or episode in resolved.parents:
        raise EpisodeError(
            f"the bench-side store {store} resolves inside the episode directory "
            f"{episode}; hidden truth must live outside the episode"
        )
    return store


class Referee:
    """Writer for one episode's bench-side store.

    Records are held in memory and written in one pass by :meth:`close`, so
    the stream appears complete or not at all, and only for a fresh episode.
    The store directory is created by :meth:`close`; nothing here reads it.
    """

    def __init__(self, episode_dir: Path) -> None:
        self.episode_dir = Path(episode_dir)
        self.store_dir = truth_store_path(self.episode_dir)
        self.stream_path = self.store_dir / TRUTH_EVENTS_FILENAME
        if self.stream_path.exists():
            raise EpisodeError(
                f"{self.stream_path} already exists; the bench-side stream is written once"
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
        """Write the whole stream once into the store, or refuse."""
        if self._closed:
            raise EpisodeError("the bench-side stream is already written")
        if not self._events:
            raise EpisodeError("the referee recorded nothing; an empty bench-side stream is refused")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        # "x" closes the race the existence check in __init__ only hints at:
        # this process creates the file, never overwrites one.
        with self.stream_path.open("x", encoding="utf-8"):
            pass
        for event in self._events:
            append_event_line(self.stream_path, event)
        self._closed = True
        return self.stream_path
