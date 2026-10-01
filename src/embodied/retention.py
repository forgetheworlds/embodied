"""Retention: what a scored run keeps on disk.

Mining run artifacts is how this project's real defects were found, and the
runs tree is ignored by git, so a deletion is permanent. The rule here is
retention, not deletion: the receipt-level files (receipt, manifest, checks,
bring-up and the other small JSON records) always stay -- they are kilobytes
and they are the ground truth behind the README's claims -- while the bulky
raw material they were made from (stereo pairs, the MAVLink stream, the
estimator feed) is pruned once a run has left the newest-``WINDOW_RUNS``
window and nothing cites it.

Three things stop a run from being pruned at all:

* a sensor capture under it (``run-*/sensor-capture``) -- a capture exists to
  be replayed, so it is under diagnosis by construction;
* the operator's keep marker, written when a command runs with
  ``--keep-artifacts`` (a run under diagnosis, or a capture that must survive);
* a citation: if any documentation or report names the run, the newest window
  alone does not authorise removing its raw material.

Pruning writes ``prune.json`` beside the receipt, naming every removed path
and its size, because the receipt's artifact hashes describe bytes that may no
longer be on disk and a reader must be able to tell a pruned run from a
tampered one.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

RETENTION_VERSION = "retention-1"
PRUNE_FILENAME = "prune.json"
KEEP_MARKER = "retention-keep"

# How many of a stage's newest runs stay whole. The observed same-day diagnosis
# bursts are six to nine runs (p01l-streak-1..6, p01l-f2tick-1..8,
# p01l-clamp-1..9); ten covers a full session's sweep with one to spare, and
# bounds the retained bulk at roughly half a gigabyte at ~50 MB per run.
WINDOW_RUNS = 10

# The bulky artifact kinds: the stereo pair frames and the wire/feed streams.
BULKY_FILES = frozenset({"mavlink.jsonl", "estimator-feed.jsonl", "imu.jsonl", "pairs.jsonl"})
BULKY_DIRECTORIES = frozenset({"pairs"})

# Files a bare mention inside a document can pin, because reports cite them
# that way ("each run's imu.jsonl", "run-{a,b}/mavlink.jsonl").
MENTIONABLE_FILES = BULKY_FILES | {"motion.jsonl", "checks.json", "startup.json", "shutdown.json"}

ATTEMPT_DIR = re.compile(r"^run-[a-z]+$")
_BRACE = re.compile(r"([^\s`']*)\{([^}]*)\}([^\s`']*)")


def _attempt_directories(run_dir: Path) -> list[Path]:
    try:
        children = sorted(run_dir.iterdir())
    except OSError:
        return []
    return [child for child in children if child.is_dir() and ATTEMPT_DIR.match(child.name)]


def _receipt_artifact_paths(run_dir: Path) -> set[str]:
    try:
        receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {
        entry["path"]
        for entry in receipt.get("artifacts", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }


def prunable_paths(run_dir: Path, *, include_captures: bool = False) -> list[Path]:
    """The bulky artifacts of ``run_dir``: pairs, streams, unscored attempts.

    An attempt directory the receipt never names is a failed launch; its whole
    directory goes. A scored attempt keeps everything except the bulky kinds.
    ``include_captures`` is the one-time apply pass's one-capture rule: the
    runner never sets it, because a run's fresh capture is its replay fixture.
    """
    receipt_paths = _receipt_artifact_paths(run_dir)
    prunable: list[Path] = []
    for attempt in _attempt_directories(run_dir):
        named = any(
            path == attempt.name or path.startswith(attempt.name + "/") for path in receipt_paths
        )
        if not named:
            prunable.append(attempt)
            continue
        for child in sorted(attempt.iterdir()):
            if child.name in BULKY_DIRECTORIES and child.is_dir():
                prunable.append(child)
            elif child.name in BULKY_FILES and child.is_file():
                prunable.append(child)
            elif include_captures and child.name == "sensor-capture" and child.is_dir():
                prunable.append(child)
    return prunable


def _unprotected_paths(
    run_dir: Path,
    protected_paths: Iterable[Path],
    captures_prunable: bool,
    only_paths: Sequence[Path] | None = None,
) -> list[tuple[Path, Path]]:
    """The prunable paths no citation guard covers, with their resolved forms.

    A guard counts only when it is the run directory itself or inside it: a
    citation of ``work/runs/p00-airframe`` names a container, and one
    container-level citation must not shield every run's bulk beneath it.
    """
    run_resolved = run_dir.resolve()
    guards = [
        guard
        for guard in (path.resolve() for path in protected_paths)
        if guard == run_resolved or run_resolved in guard.parents
    ]
    candidates = (
        list(only_paths)
        if only_paths is not None
        else prunable_paths(run_dir, include_captures=captures_prunable)
    )
    unprotected = []
    for path in candidates:
        resolved = path.resolve()
        if any(resolved == guard or guard in resolved.parents for guard in guards):
            continue
        unprotected.append((path, resolved))
    return unprotected


def _run_is_kept_whole(run_dir: Path) -> str | None:
    """Why this run must not be pruned at all, or ``None`` if it may be."""
    if (run_dir / KEEP_MARKER).is_file():
        return f"keep marker {KEEP_MARKER} (operator --keep-artifacts)"
    for attempt in _attempt_directories(run_dir):
        if (attempt / "sensor-capture").is_dir():
            return "a recorded sensor capture lives under this run (replay fixture)"
    return None


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    return sum(
        child.stat().st_size for child in path.rglob("*") if child.is_file()
    )


def prune_run(
    run_dir: Path,
    *,
    protected_paths: Iterable[Path] = (),
    citation_sources: Sequence[Path] = (),
    captures_prunable: bool = False,
    only_paths: Iterable[Path] | None = None,
) -> dict:
    """Prune one run's bulky artifacts, keeping the receipt level.

    ``protected_paths`` are subtrees citations depend on; they survive inside
    an otherwise pruned run. ``citation_sources`` are documents whose mention
    of the run (in any form) keeps the run whole -- the guard the runner uses,
    because a cited run is evidence and the runner cannot know which of its
    files the citation leans on. ``captures_prunable`` lets a one-time apply
    pass take a capture the scoring-time rule would keep (the runner never
    sets it: a fresh capture is always under diagnosis). ``only_paths`` prunes
    exactly those paths (still guard-filtered) instead of the full prunable
    set -- the apply pass uses it to take a capture from a run that otherwise
    stays whole.
    """
    name = run_dir.name
    reason = None if captures_prunable else _run_is_kept_whole(run_dir)
    if reason is not None:
        return {"run": name, "pruned": False, "reason": reason, "bytes_freed": 0}
    if citation_sources and _run_is_cited(run_dir, citation_sources):
        return {
            "run": name,
            "pruned": False,
            "reason": "cited by " + ", ".join(str(source) for source in citation_sources),
            "bytes_freed": 0,
        }

    removed: list[dict[str, object]] = []
    for path, _resolved in _unprotected_paths(
        run_dir, protected_paths, captures_prunable, only_paths
    ):
        bytes_freed = _tree_bytes(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(
            {"path": path.relative_to(run_dir).as_posix(), "bytes": bytes_freed}
        )
    if not removed:
        return {"run": name, "pruned": False, "reason": "nothing prunable", "bytes_freed": 0}

    record = {
        "retention_version": RETENTION_VERSION,
        "run": name,
        "pruned_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": (
            f"receipt-level files kept; bulky artifacts removed "
            f"({', '.join(sorted(BULKY_DIRECTORIES | BULKY_FILES))} and unscored "
            f"attempt directories); newest {WINDOW_RUNS} runs of the stage stay whole"
        ),
        "removed": removed,
        "bytes_freed": sum(entry["bytes"] for entry in removed),
        "receipt_note": (
            "receipt.json hashes describe the bytes at scoring time; the paths "
            "listed here are no longer on disk"
        ),
    }
    try:
        (run_dir / PRUNE_FILENAME).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as error:  # a failed record must not fail a finished run
        print(f"retention: cannot write {run_dir / PRUNE_FILENAME}: {error}", file=sys.stderr)
    return {
        "run": name,
        "pruned": True,
        "reason": "left the retention window",
        "bytes_freed": record["bytes_freed"],
    }


# ---------------------------------------------------------------------------
# Citations: what the documents and reports still point at
# ---------------------------------------------------------------------------

def _expand_braces(token: str) -> list[str]:
    """Expand ``prefix{1..9}suffix`` and ``prefix{a,b}suffix`` once."""
    match = _BRACE.search(token)
    if match is None:
        return [token]
    head, body, tail = match.groups()
    parts: list[str]
    if re.fullmatch(r"\d+\.\.\d+", body):
        low_text, high_text = body.split("..")
        parts = [str(value) for value in range(int(low_text), int(high_text) + 1)]
    else:
        parts = [part for part in body.split(",") if part]
    return [head + part + tail for part in parts]


def _token_names_run(run_name: str, token: str) -> bool:
    """Does a document token (a citation form) reach this run directory?

    Reports cite runs three ways: the full name, a series (``p01l-clamp-1``),
    and a truncated or elided id (``…-20260930T16``, ``p01l-replay-ref-…034502Z``).
    A bare prefix that stops mid-word only counts when the token is itself
    specific enough to be an id rather than prose -- an English ``accept``
    must not protect ``accept-1``.
    """
    if "…" in token or "..." in token:
        mark = "…" if "…" in token else "..."
        prefix, _, suffix = token.partition(mark)
        if len(prefix) < 3 and len(suffix) < 3:
            return False
        return run_name.startswith(prefix) and run_name.endswith(suffix)
    if token == run_name:
        return True
    if not run_name.startswith(token):
        return False
    if run_name[len(token) :][:1] in ("-", "_", "."):
        return True
    return "-" in token and len(token) >= 10


def _mention_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _BRACE.finditer(text):
        for expanded in _expand_braces(match.group(0)):
            tokens.add(expanded)
    tokens.update(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-…]*", text))
    return tokens


def _run_names_in(stage_dir: Path) -> list[str]:
    try:
        return [child.name for child in sorted(stage_dir.iterdir()) if child.is_dir()]
    except OSError:
        return []


def _run_is_cited(run_dir: Path, sources: Sequence[Path]) -> bool:
    return any(
        _token_names_run(run_dir.name, token)
        for source in sources
        for text in _source_text(source)
        for token in _mention_tokens(text)
    )


def _source_text(source: Path) -> list[str]:
    try:
        return [source.read_text(encoding="utf-8", errors="replace")]
    except (OSError, IsADirectoryError):
        return []


def citation_paths(sources: Sequence[Path], stage_dir: Path, runs_root: Path) -> set[Path]:
    """The concrete paths under the runs tree the citations point at.

    Three citation forms resolve: ``work/runs/<rest>`` written in full,
    ``<run-name>/<rest>`` relative to the stage, and attempt-relative or bare
    mentions of a prunable stream name, which pin that file in every run the
    same document names -- the way the p00 reports cite
    ``run-{a,b}/mavlink.jsonl`` beside the run they are discussing.
    """
    texts = [(source, text) for source in sources for text in _source_text(source)]
    names = _run_names_in(stage_dir)
    protected: set[Path] = set()

    for _, text in texts:
        for match in re.finditer(r"work/runs/([^\s`'\)\]]+)", text):
            rest = match.group(1).rstrip(".,;:")
            protected.add((runs_root / rest).resolve())

    for _, text in texts:
        tokens = _mention_tokens(text)
        mentioned = [
            name
            for name in names
            if any(_token_names_run(name, token) for token in tokens)
        ]
        for name in mentioned:
            for match in re.finditer(re.escape(name) + r"/([^\s`'\)\]]+)", text):
                protected.add((stage_dir / name / match.group(1).rstrip(".,;:")).resolve())

    # Bare and attempt-relative stream mentions ("each run's imu.jsonl",
    # "run-{a,b}/mavlink.jsonl") pin that file in the runs the SAME SECTION
    # names: the reports discuss one experiment per section, and a file named
    # in one section must not shield every run the rest of the document cites.
    for _, text in texts:
        for section in re.split(r"(?m)^#{1,6} .*$", text):
            tokens = _mention_tokens(section)
            section_runs = [
                name
                for name in names
                if any(_token_names_run(name, token) for token in tokens)
            ]
            for file_name in MENTIONABLE_FILES:
                if not re.search(r"(^|[/`])" + re.escape(file_name) + r"\b", section):
                    continue
                for name in section_runs:
                    for attempt in ("run-a", "run-b"):
                        protected.add((stage_dir / name / attempt / file_name).resolve())
                    protected.add((stage_dir / name / file_name).resolve())

    return {path for path in protected if path.exists()}


def cited_run_names(sources: Sequence[Path], stage_dir: Path) -> set[str]:
    """The run directories under ``stage_dir`` that any source mentions.

    Mention forms covered: the bare name, a brace series (``p01l-clamp-{1..9}``
    or ``-{C1,C2}-``), and elided ids (``p01l-replay-ref-…034502Z``).
    """
    names = _run_names_in(stage_dir)
    mentioned: set[str] = set()
    for source in sources:
        for text in _source_text(source):
            tokens = _mention_tokens(text)
            for name in names:
                if name in mentioned:
                    continue
                if any(_token_names_run(name, token) for token in tokens):
                    mentioned.add(name)
    return mentioned


def default_citation_sources(root: Path) -> tuple[Path, ...]:
    """The documents a prune must respect: README, the design docs, the reports."""
    sources: list[Path] = [root / "README.md"]
    docs = root / "design" / "docs"
    if docs.is_dir():
        sources.extend(sorted(path for path in docs.rglob("*") if path.is_file()))
    runs = root / "work" / "runs"
    if runs.is_dir():
        sources.extend(sorted(runs.rglob("*.md")))
    return tuple(source for source in sources if source.is_file())


# ---------------------------------------------------------------------------
# The window: the rule the runner applies when a run is scored
# ---------------------------------------------------------------------------

def _ordering_stamp(run_dir: Path) -> float:
    receipt = run_dir / "receipt.json"
    if receipt.is_file():
        return receipt.stat().st_mtime
    return run_dir.stat().st_mtime


def _is_run_directory(path: Path) -> bool:
    if (path / "receipt.json").is_file():
        return True
    return any(ATTEMPT_DIR.match(child.name) for child in _attempt_directories(path))


def enforce_window(stage_dir: Path, *, citation_sources: Sequence[Path] = ()) -> list[dict]:
    """Keep the newest ``WINDOW_RUNS`` runs of one stage whole, prune the rest.

    Runs that carry a capture, a keep marker or a citation never consume a
    window slot: they are simply not candidates. Returns one record per run
    pruned in this pass.
    """
    try:
        candidates = [child for child in sorted(stage_dir.iterdir()) if _is_run_directory(child)]
    except OSError:
        return []
    prunable_candidates = [
        child
        for child in candidates
        if _run_is_kept_whole(child) is None
        and not (citation_sources and _run_is_cited(child, citation_sources))
    ]
    prunable_candidates.sort(key=_ordering_stamp, reverse=True)
    records: list[dict] = []
    for run_dir in prunable_candidates[WINDOW_RUNS:]:
        record = prune_run(run_dir, citation_sources=citation_sources)
        if record["pruned"]:
            records.append(record)
    return records


def apply_on_score(output: Path, root: Path, *, keep_all: bool) -> list[dict]:
    """The runner's hook: called once, right after a run's receipt is written.

    The run just scored is the stage's newest, so it sits inside the window and
    keeps its raw material; what the score does is push the stage's older runs
    out of the window, and those are pruned here. ``keep_all`` is the escape
    hatch: it marks the run so no later pass ever prunes it.
    """
    if keep_all:
        try:
            (output / KEEP_MARKER).write_text(
                json.dumps({"reason": "operator --keep-artifacts"}) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            print(f"retention: cannot write {output / KEEP_MARKER}: {error}", file=sys.stderr)
        return []
    runs_root = (root / "work" / "runs").resolve()
    try:
        stage_dir = output.parent.resolve()
    except OSError:
        return []
    if runs_root not in stage_dir.parents:
        # A run scored outside the runs tree (a scratch or test output) owns
        # only itself; its siblings are not this rule's to touch.
        return []
    try:
        return enforce_window(stage_dir, citation_sources=default_citation_sources(root))
    except OSError as error:
        print(f"retention: window enforcement failed: {error}", file=sys.stderr)
        return []
