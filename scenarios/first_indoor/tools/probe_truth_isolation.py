#!/usr/bin/env python3
"""Truth-isolation probe for the first-indoor scene, against the live bench code.

Proves, with the real modules rather than by reading them, that:

1. the per-episode hidden-truth store is a sibling directory
   (``<episode>.truth/truth-events.jsonl``) and the referee *refuses* a store
   that would resolve inside the episode directory (so isolation is structural,
   not a naming convention);
2. the scene's truth seed declares only ``world_state`` facts and the referee
   writes them into that store — the store is the only place the grader reads
   them from;
3. the agent surface cannot reach the store or the seed: every attempted read
   of a truth member raises ``SurfaceViolation``, the recorded artifact list is
   confined to the agent projection, and ``replay`` — which opens only the
   projection — still works;
4. the grader *does* consume the store: a score's ``missed_present_targets``
   come from the hidden ``world_state`` alone, and removing the store turns the
   score into ``StoreMissing``.

The episode written here is a SYNTHETIC FIXTURE, labelled as such in its own
events: it carries one intervention event and a world_state/physical_outcome
pair whose outcome values are placeholders. It proves the storage and reading
split; it says nothing about flight, and nothing about this scene's outcome.

Run from anywhere; writes a receipt and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

SCENE = Path(__file__).resolve().parents[1]
REPO_ROOT = SCENE.parents[1]


def check(receipt: dict, name: str, ok: bool, detail: str) -> bool:
    receipt["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'pass' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture-root", type=Path, required=True,
                        help="directory the synthetic episode fixture is written under")
    parser.add_argument("--output", type=Path, required=True, help="receipt JSON path")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import yaml

    from embodied.bench.events import EpisodeError, TRUTH_EVENT_KINDS
    from embodied.bench.grader import StoreMissing, grade
    from embodied.bench.recorder import (
        PROJECTION,
        AgentSurface,
        Recorder,
        RunManifest,
        SurfaceViolation,
        replay,
    )
    from embodied.bench.referee import TRUTH_EVENTS_FILENAME, Referee, truth_store_path
    from embodied.contracts.records import ClockStamp

    receipt = {
        "probe": "first-indoor truth isolation",
        "scene": str(SCENE),
        "episode_kind": "synthetic-fixture",
        "note": "synthetic fixture: proves storage/reading split only, no flight claim",
        "checks": [],
        "pass": False,
    }
    ok = True

    stamp_dir = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(2)
    fixture_root = Path(args.fixture_root) / f"truth-probe-{stamp_dir}"
    episode = fixture_root / "episode-001"
    episode.mkdir(parents=True)

    stamp = lambda n: ClockStamp("truth-probe-host", "probe-clock", n)  # noqa: E731

    # --- the episode as the runtime side would see it ------------------------
    recorder = Recorder(episode)
    recorder.record(
        "intervention",
        {
            "intervention_id": "probe-i0",
            "actor": "verifier",
            "category": "synthetic-fixture",
            "reason": "truth-isolation probe placeholder; not a real intervention",
        },
        stamp(1),
        sim_time_s=0.0,
    )
    recorder.close(RunManifest(episode_id="truth-probe-001", episode_kind="synthetic-fixture"))

    # --- the bench-side store, seeded from the scene's truth.yaml ------------
    truth_document = yaml.safe_load((SCENE / "truth.yaml").read_text(encoding="utf-8"))
    world_state = truth_document["world_state"]
    seeded_outcome = {
        "inspected": {"red_block": False},
        "return_verified": False,
        "violations": ["synthetic-fixture: placeholder outcome, not a measured flight"],
        "takeover": False,
    }
    referee = Referee(episode)
    referee.record("world_state", world_state, stamp(2), sim_time_s=0.0)
    referee.record("physical_outcome", seeded_outcome, stamp(3), sim_time_s=0.0)
    store_stream = referee.close()

    # --- 1. the store is a sibling, and inside-episode resolution is refused --
    store = truth_store_path(episode)
    sibling_ok = (
        store == episode.parent / (episode.name + ".truth")
        and episode not in store.parents
        and store.is_dir()
        and (store / TRUTH_EVENTS_FILENAME).is_file()
    )
    ok &= check(
        receipt, "store_is_sibling_directory",
        sibling_ok,
        f"{store.relative_to(fixture_root)} (stream {store_stream.name}, "
        f"truth kinds {sorted(TRUTH_EVENT_KINDS)})",
    )

    inside_case = fixture_root / "episode-inside"
    (inside_case / "sub").mkdir(parents=True)
    link = inside_case.parent / (inside_case.name + ".truth")
    link.symlink_to(inside_case / "sub", target_is_directory=True)
    try:
        truth_store_path(inside_case)
        inside_refused, inside_detail = False, "resolver accepted a store inside the episode"
    except EpisodeError as error:
        inside_refused, inside_detail = True, str(error).split(";")[0]
    ok &= check(receipt, "inside_episode_store_refused", inside_refused, inside_detail)

    # --- 2. the seed declares world_state only -------------------------------
    seed_ok = (
        set(truth_document.get("world_state", {})) == {"targets", "world_counts"}
        and truth_document.get("physical_outcome") is None
        and all(
            set(entry) == {"present"} for entry in truth_document["world_state"]["targets"].values()
        )
    )
    ok &= check(
        receipt, "seed_declares_world_state_only", seed_ok,
        "targets and world_counts only; physical_outcome deliberately null (measured per run)",
    )

    # --- 3. the agent surface cannot reach the truth -------------------------
    surface = AgentSurface.open(episode)
    attempts = [
        TRUTH_EVENTS_FILENAME,
        f"{store.name}/{TRUTH_EVENTS_FILENAME}",
        f"../{store.name}/{TRUTH_EVENTS_FILENAME}",
        "truth.yaml",
        "../truth.yaml",
    ]
    refusals = []
    for member in attempts:
        try:
            surface.read_member(member)
            refusals.append((member, "READ SUCCEEDED - LEAK"))
        except SurfaceViolation:
            refusals.append((member, "SurfaceViolation"))
        except EpisodeError as error:
            refusals.append((member, f"EpisodeError: {error}"))
    all_refused = all(detail == "SurfaceViolation" for _, detail in refusals)
    ok &= check(
        receipt, "agent_surface_refuses_truth_members", all_refused,
        "; ".join(f"{name} -> {detail}" for name, detail in refusals),
    )
    ok &= check(
        receipt, "recorded_artifacts_confined_to_projection",
        set(surface.manifest.artifacts) <= set(PROJECTION)
        and tuple(surface.manifest.permitted_projection) == PROJECTION
        and surface.payload_names() == ()
        and store.name not in json.dumps(surface.manifest.artifacts),
        f"manifest artifacts {sorted(surface.manifest.artifacts)}; projection {list(PROJECTION)}",
    )
    surface.verify_artifacts()
    replayed = replay(episode)
    ok &= check(
        receipt, "replay_reads_projection_only", replayed["event_count"] == 1,
        f"replay returned {replayed['event_count']} agent event(s) with the store beside it",
    )

    # --- 4. the grader consumes the store, and only the store ----------------
    score = grade(episode)
    score_document = json.loads((episode / "score.json").read_text(encoding="utf-8"))
    graded_ok = (
        score_document["missed_present_targets"] == ["red_block"]
        and score_document["mission"]["physical_return_verified"] is False
        and score_document["claims_total"] == 0
    )
    ok &= check(
        receipt, "grader_reads_hidden_world_state", graded_ok,
        f"missed_present_targets={score_document['missed_present_targets']} "
        f"(from the store alone), return_verified="
        f"{score_document['mission']['physical_return_verified']}",
    )

    hidden = fixture_root / "store-hidden"
    store.rename(hidden)
    try:
        grade(episode)
        missing_raised, missing_detail = False, "grade succeeded after the store was removed"
    except StoreMissing as error:
        missing_raised, missing_detail = True, str(error)
    hidden.rename(fixture_root / (episode.name + ".truth"))
    ok &= check(receipt, "score_blocked_without_the_store", missing_raised, missing_detail)

    receipt["fixture"] = str(episode)
    receipt["store"] = str(store)
    receipt["pass"] = bool(ok)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"receipt: {args.output}")
    print("PASS: the store is a sibling the agent surface cannot reach, and the grader reads it"
          if ok else "FAIL: truth isolation was not proven")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
