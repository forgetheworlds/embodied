#!/usr/bin/env python3
"""Apply the retention rule to the runs already on disk, and check citations.

The runner enforces retention at scoring time (``embodied.retention``); this
script applies the same rule to a tree that accumulated before the rule
existed, with the two guards a one-pass application needs and a scoring-time
pass cannot have:

* hard fences -- the pinned evidence trees and the one recorded reference
  capture the replay gate replays are never touched, whatever the rule says;
* file-level citations -- a report that names ``<run>/run-a/estimator-feed.jsonl``
  pins that exact file inside an otherwise pruned run.

Subcommands:

* ``inventory``  -- print the plan (path, size, decision, reason) and write it
  to ``work/runs/final/RETENTION-INVENTORY.json``; nothing is removed;
* ``apply``      -- execute the plan (dry-run unless ``--yes``);
* ``check-citations`` -- every ``work/...`` path referenced by README.md and
  ``design/docs/**`` must resolve on disk; exit 1 if any does not.

The dataflash logs under ``work/ardupilot/logs`` follow the mission's own rule:
a ``.BIN`` whose number any document or report names is kept, the rest may go.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied import retention  # noqa: E402

# Never touched, whatever the window or the citations say: the pinned estimator
# evidence the preflight re-hashes, and the one recorded reference capture the
# replay gate replays (its pairs and inertial stream ARE the fixture).
FENCED_SUBPATHS = (
    "work/runs/p01-localization/pin-evidence",
    "work/runs/p01-localization/p01l-replay-ref7-20261001T045848Z",
)

DATAFLASH_DIR = "work/ardupilot/logs"


def _fenced(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return True
    return any(relative == fence or relative.startswith(fence + "/") for fence in FENCED_SUBPATHS)


def _stages(runs_root: Path) -> list[Path]:
    """The directories runs score into: the runs root itself and its stages."""
    stages = [runs_root]
    for child in sorted(runs_root.iterdir()):
        if child.is_dir() and any(grandchild.is_dir() for grandchild in child.iterdir()):
            stages.append(child)
    return stages


def _apply_prunable_paths(run_dir: Path) -> list[Path]:
    """The scoring-time prunable set plus the one-capture rule for captures.

    The runner keeps any run carrying a sensor capture whole; this one-time
    pass keeps exactly one capture (the fenced reference run) and prunes the
    rest, which is what "one recorded reference capture" requires.
    """
    return retention.prunable_paths(run_dir, include_captures=True)


def _path_bytes(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _plan_stage(stage: Path, sources, runs_root: Path, window: int) -> list[dict]:
    window = 0 if stage.name in retention.CLOSED_STAGES else window
    cited_names = retention.cited_run_names(sources, stage)
    protected_paths = retention.citation_paths(sources, stage, runs_root)
    candidates = []
    for child in sorted(stage.iterdir()):
        if not child.is_dir() or not retention._is_run_directory(child):
            continue
        candidates.append(child)
    def stamp(run_dir: Path) -> float:
        return retention._ordering_stamp(run_dir)
    candidates.sort(key=stamp, reverse=True)
    plan = []
    unfenced = [child for child in candidates if not _fenced(child)]
    for run_dir in candidates:
        size = _path_bytes(run_dir)
        if _fenced(run_dir):
            plan.append({"path": str(run_dir), "bytes": size, "decision": "keep-whole",
                         "reason": "fenced: pinned evidence or the replay reference capture"})
            continue
        if (run_dir / retention.KEEP_MARKER).is_file():
            plan.append({"path": str(run_dir), "bytes": size, "decision": "keep-whole",
                         "reason": "keep marker present"})
            continue
        if unfenced.index(run_dir) < window:
            captures = [
                attempt / "sensor-capture"
                for attempt in retention._attempt_directories(run_dir)
                if (attempt / "sensor-capture").is_dir()
            ]
            if captures:
                # The window keeps a run whole for diagnosis, but the
                # one-capture rule is absolute: only the fenced reference
                # capture survives, so a window run still loses its capture.
                plan.append({"path": str(run_dir), "bytes": size,
                             "decision": "prune-captures",
                             "reason": f"inside the newest-{window} window, but the "
                                       f"one-capture rule: only the reference capture stays",
                             "removable": [str(c.relative_to(run_dir)) for c in captures],
                             "cited_kept": []})
            else:
                plan.append({"path": str(run_dir), "bytes": size, "decision": "keep-whole",
                             "reason": f"inside the newest-{window} window (diagnosis headroom)"})
            continue
        prunable = _apply_prunable_paths(run_dir)
        removable_paths = retention._unprotected_paths(run_dir, protected_paths, True)
        cited_here = sorted(path for path in prunable if path not in {p for p, _ in removable_paths})
        removable = [path for path, _ in removable_paths]
        freed = sum(_path_bytes(path) for path in removable)
        reason = (
            f"prune to receipt level ({freed / 1e6:.0f} MB prunable"
            + (f", {len(cited_here)} cited file(s) kept" if cited_here else "")
            + (", run name cited in docs/reports" if run_dir.name in cited_names else "")
            + ")"
        )
        plan.append({"path": str(run_dir), "bytes": size, "decision": "prune",
                     "reason": reason,
                     "removable": [str(path.relative_to(run_dir)) for path in removable],
                     "cited_kept": [str(path.relative_to(run_dir)) for path in cited_here]})
    return plan


def _dataflash_plan(sources) -> list[dict]:
    logs_dir = REPO / DATAFLASH_DIR
    if not logs_dir.is_dir():
        return []
    text = "\n".join(
        text_part for source in sources for text_part in retention._source_text(source)
    )
    cited_numbers = set(re.findall(r"\b(00000\d{3})\b", text))
    # Explicit ranges: 00000139-00000145 (.BIN or bare, hyphen or en dash).
    for low, high in re.findall(r"\b(00000\d{3})\s*[-\u2013]\s*(00000\d{3})", text):
        cited_numbers.update(f"{number:08d}" for number in range(int(low), int(high) + 1))
    # Elided ids ("`…69.BIN`" for 00000069.BIN): every existing log whose stem
    # ends with the named digits is kept -- over-keeping a log is cheap,
    # breaking a citation is not.
    elided = re.findall(r"\u2026(\d{1,8})\.BIN", text)
    if elided:
        stems = [log.stem for log in logs_dir.glob("*.BIN")]
        for fragment in elided:
            cited_numbers.update(stem for stem in stems if stem.endswith(fragment))
    plan = []
    for log in sorted(logs_dir.glob("*.BIN")):
        cited = log.stem in cited_numbers
        plan.append({
            "path": str(log),
            "bytes": log.stat().st_size,
            "decision": "keep" if cited else "prune",
            "reason": "cited by a document or report" if cited else "no citation names this log",
        })
    return plan


def _build_plan(window: int) -> dict:
    runs_root = REPO / "work" / "runs"
    sources = retention.default_citation_sources(REPO)
    plan = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "window": window, "runs": [], "dataflash": _dataflash_plan(sources)}
    for stage in _stages(runs_root):
        plan["runs"].extend(_plan_stage(stage, sources, runs_root, window))
    return plan


def _apply(plan: dict) -> dict:
    removed_files = 0
    freed = 0
    for entry in plan["runs"]:
        if entry["decision"] not in ("prune", "prune-captures"):
            continue
        run_dir = Path(entry["path"])
        record = retention.prune_run(
            run_dir,
            protected_paths=[run_dir / relative for relative in entry["cited_kept"]],
            captures_prunable=True,
            only_paths=(
                [run_dir / relative for relative in entry["removable"]]
                if entry["decision"] == "prune-captures"
                else None
            ),
        )
        if record["pruned"]:
            freed += record["bytes_freed"]
    for entry in plan["dataflash"]:
        if entry["decision"] != "prune":
            continue
        path = Path(entry["path"])
        freed += path.stat().st_size
        removed_files += 1
        path.unlink()
    return {"runs_freed": freed, "dataflash_removed": removed_files}





def _check_citations() -> int:
    failures = []
    references = 0
    for source in [REPO / "README.md", *sorted(p for p in (REPO / "design" / "docs").rglob("*") if p.is_file())]:
        text = source.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"\b(work/[A-Za-z0-9_.\-/]+)", text):
            references += 1
            target = (REPO / match.group(1).rstrip(".,;:)")).resolve()
            if not target.exists():
                failures.append((source.relative_to(REPO).as_posix(), match.group(1)))
    for citing, path in sorted(set(failures)):
        print(f"UNRESOLVED: {path}  (referenced by {citing})")
    print(f"{references} work/... references checked, {len(set(failures))} unresolved")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["inventory", "apply", "check-citations"])
    parser.add_argument("--window", type=int, default=retention.WINDOW_RUNS,
                        help="how many of each stage's newest runs stay whole")
    parser.add_argument("--yes", action="store_true", help="apply for real (default: dry run)")
    arguments = parser.parse_args(argv)

    if arguments.command == "check-citations":
        return _check_citations()

    plan = _build_plan(arguments.window)
    out_dir = REPO / "work" / "runs" / "final"
    prunable = [entry for entry in plan["runs"] if entry["decision"] != "keep-whole"]
    keep_whole = [entry for entry in plan["runs"] if entry["decision"] == "keep-whole"]

    for entry in plan["runs"]:
        print(f"{entry['bytes'] / 1e6:9.1f} MB  {entry['decision']:11s}  {entry['path']}"
              f"  -- {entry['reason']}")
    for entry in plan["dataflash"]:
        print(f"{entry['bytes'] / 1e6:9.1f} MB  {entry['decision']:11s}  {entry['path']}"
              f"  -- {entry['reason']}")

    projected_freed = sum(
        _path_bytes(Path(entry["path"]) / relative)
        for entry in prunable
        for relative in entry["removable"]
    )
    print(
        f"\n{len(keep_whole)} runs kept whole, {len(prunable)} runs to prune "
        f"({projected_freed / 1e9:.2f} GB), dataflash: "
        f"{sum(1 for e in plan['dataflash'] if e['decision'] == 'prune')} of "
        f"{len(plan['dataflash'])} logs to prune "
        f"({sum(e['bytes'] for e in plan['dataflash'] if e['decision'] == 'prune') / 1e6:.0f} MB)"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RETENTION-INVENTORY.json").write_text(json.dumps(plan, indent=1) + "\n")

    if arguments.command == "apply":
        if not arguments.yes:
            print("dry run: nothing removed (pass --yes to execute)")
            return 0
        summary = _apply(plan)
        (out_dir / "RETENTION-APPLIED.json").write_text(
            json.dumps({"plan": "RETENTION-INVENTORY.json", **summary,
                        "applied_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                       indent=1) + "\n"
        )
        print(f"applied: {summary['runs_freed'] / 1e9:.2f} GB freed from runs, "
              f"{summary['dataflash_removed']} dataflash logs removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
