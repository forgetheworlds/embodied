"""Benchmark storage: record, replay and independently score episode directories.

Truth isolation is this package's one hard invariant, enforced as separate
storage plus two separate message surfaces plus negative checks — never by
convention alone:

* The bench-side store lives in a sibling directory *outside* the episode
  directory, which is what the runtime side is handed (specification 3.2).
  Its path is resolved in exactly one place, in
  :mod:`embodied.bench.referee`, and that resolver refuses a store that
  would resolve inside the episode.
* :mod:`embodied.bench.recorder` owns the agent-facing projection. It names no
  bench-private file, and its reader refuses any member outside the declared
  projection, so a runtime that consumes this surface cannot reach hidden
  scenario facts or support annotations.
* :mod:`embodied.bench.referee` is the only writer of the bench-side store and
  has no read path.
* :mod:`embodied.bench.grader` is the only reader of the bench-side store and
  of the adjudication annotation; only it produces a score.

The record payload contract is consumed unchanged from
``embodied.contracts.records`` (RECORDS_REVISION). This package's own schema
revisions are named by the modules that own them: ``p02-events-1`` (event
envelope), ``p02-manifest-1`` (run manifest), ``p02-score-1`` (score).
"""
