"""Benchmark storage: record, replay and independently score episode directories.

Truth isolation is this package's one hard invariant, and it is enforced as two
separate message surfaces plus negative checks, not by convention:

* :mod:`embodied.bench.recorder` owns the agent-facing projection. It names no
  bench-private file, and its reader refuses any member outside the declared
  projection, so a runtime that consumes this surface cannot reach hidden
  scenario facts or support annotations.
* :mod:`embodied.bench.referee` is the only writer of the bench-side stream and
  has no read path.
* :mod:`embodied.bench.grader` is the only reader of the bench-side stream and
  of the adjudication annotation; only it produces a score.

The record payload contract is consumed unchanged from
``embodied.contracts.records`` (RECORDS_REVISION). This package's own schema
revisions are named by the modules that own them: ``p02-events-1`` (event
envelope), ``p02-manifest-1`` (run manifest), ``p02-score-1`` (score).
"""
