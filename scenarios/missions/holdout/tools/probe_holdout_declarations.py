#!/usr/bin/env python3
"""Declaration probe for the held-out scenes, against the live bench code.

Proves, with the real modules rather than by reading them, that each held-out
scene is *declarable and resolvable* the way the transport loads a suite:

1. ``live_record.load_suite_document`` accepts each declaration and
   ``register_suite`` builds a real ``SuiteSpec`` from it — name,
   registered_by, localization_mode and a positive provider budget;
2. the declared ``world`` exists on disk, and the declared
   ``mission_instruction`` is verbatim the scene's own ``mission.yaml``
   instruction: the held-out wording is passed through, not restated;
3. ``load_truth_seed`` accepts each seed and ``world_state_payload`` yields the
   bench envelope exactly (``targets`` -> ``{present}``, plus ``world_counts``);
4. a present-target scene's geometry resolves through ``target_position_ned``,
   and the value equals ``enu_to_ned`` of the same ENU triple the world file
   carries — one convention, not a second invention;
5. the ABSENT-target scene declares zero present targets, and
   ``target_position_ned`` REFUSES it rather than guessing a position, because
   a fabricated triple would let a referee measure an inspection of nothing;
6. reachability is stated exactly, and the two halves are labelled for what
   they are: what this probe EXECUTED, and what it read from the source.

This probe never calls ``record`` and never starts a simulator or SITL: the
preconditions above are what a recording checks before it flies, and each is
exercised through the same functions the recording calls.

Run from anywhere; writes a receipt and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HOLDOUT = Path(__file__).resolve().parents[1]
REPO_ROOT = HOLDOUT.parents[2]

SCENES = ("holdout-e-wide", "holdout-f-z", "holdout-g-long", "holdout-h-clutter")


def check(receipt: dict, name: str, ok: bool, detail: str) -> bool:
    receipt["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'pass' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="receipt JSON path")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import yaml

    from embodied.bench import live_record
    from embodied.bench.recorder import SUITE_REGISTRY
    from embodied.platform.webots_ardupilot import enu_to_ned

    receipt: dict = {
        "probe": "holdout declarations",
        "holdout_root": str(HOLDOUT),
        "scenes": [],
        "checks": [],
        "pass": False,
    }
    ok = True

    for scene in SCENES:
        scene_dir = HOLDOUT / scene
        declaration_path = REPO_ROOT / "configs" / "suites" / f"{scene}.yaml"
        entry: dict = {"scene": scene, "checks": []}

        # --- 1. the declaration loads and builds a real SuiteSpec ------------
        document = None
        spec = None
        try:
            document = live_record.load_suite_document(declaration_path)
            name = live_record.register_suite(document)
            spec = SUITE_REGISTRY.get(str(name)) if name else None
            built = spec is not None and spec.name == scene
            detail = (
                f"SuiteSpec(name={spec.name!r}, registered_by={spec.registered_by!r}, "
                f"mode={spec.localization_mode!r}, budget={spec.provider_budget})"
                if built
                else f"registration returned {name!r}"
            )
        except Exception as error:  # noqa: BLE001 - reported, not raised
            built = False
            detail = f"{type(error).__name__}: {error}"
        ok &= check(entry, "declaration_builds_a_suite_spec", built, detail)
        # registration is process-global; drop it again so a rerun in the same
        # process is not a duplicate-name conflict
        if document is not None:
            SUITE_REGISTRY.pop(str(document.get("name")), None)

        # --- 2. the world exists, and the instruction is the scene's own -----
        world = REPO_ROOT / str(document.get("world", "")) if document else None
        ok &= check(
            entry, "declared_world_exists", bool(world and world.is_file()),
            str(world.relative_to(REPO_ROOT)) if world else "no world declared",
        )
        scene_mission = yaml.safe_load((scene_dir / "mission.yaml").read_text(encoding="utf-8"))
        own_instruction = str(scene_mission["instructions"][0]).strip()
        declared_instruction = str(document.get("mission_instruction", "")).strip()
        ok &= check(
            entry, "instruction_is_the_scene_own_wording",
            declared_instruction == own_instruction,
            f"declared == mission.yaml instructions[0] ({len(own_instruction)} chars)",
        )

        # --- 3. the seed loads and yields the envelope exactly ---------------
        seed_path = REPO_ROOT / str(document.get("truth_seed", ""))
        seed = live_record.load_truth_seed(seed_path)
        payload = live_record.world_state_payload(seed)
        targets, counts = live_record.truth_world_state(seed)
        envelope_ok = (
            set(payload) == {"targets", "world_counts"}
            and all(set(entry_) == {"present"} for entry_ in payload["targets"].values())
            and all(isinstance(v, int) and not isinstance(v, bool) for v in payload["world_counts"].values())
        )
        ok &= check(
            entry, "seed_yields_the_bench_envelope", envelope_ok,
            f"targets={payload['targets']} world_counts={payload['world_counts']}",
        )
        ok &= check(
            entry, "seed_declares_no_outcome",
            seed.get("physical_outcome") is None,
            "physical_outcome is null: the outcome is measured per run, never seeded",
        )

        # --- 4/5. geometry: resolved for a present target, refused for absent -
        present = sorted(n for n, e in targets.items() if e.get("present") is True)
        entry["present_targets"] = present
        if len(present) == 1:
            target = present[0]
            ned = live_record.target_position_ned(seed, target)
            enu = tuple(seed["identity"][target]["position_enu_m"])
            ok &= check(
                entry, "present_target_position_resolves_to_the_one_convention",
                tuple(ned) == tuple(enu_to_ned(enu)),
                f"{target}: ned={tuple(ned)} == enu_to_ned({list(enu)})",
            )
        else:
            # the absent-target case: the refusal IS the correct behaviour
            target = sorted(targets)[0]
            try:
                live_record.target_position_ned(seed, target)
                refused, detail = False, "position resolution SUCCEEDED for an absent target"
            except Exception as error:  # noqa: BLE001 - the refusal is the check
                refused, detail = True, f"{type(error).__name__}: {str(error).split(';')[0]}"
            ok &= check(
                entry, "absent_target_position_is_refused_not_guessed", refused, detail,
            )
        entry["present_target_count"] = len(present)

        entry["pass"] = all(c["pass"] for c in entry["checks"])
        receipt["scenes"].append(entry)

    # --- 6. reachability, stated exactly ---------------------------------------
    default_path = live_record.suite_config_path()
    ok &= check(
        receipt, "bare_cli_loader_names_exactly_one_file",
        default_path.name == "first-indoor.yaml",
        f"live_record.suite_config_path() -> {default_path.relative_to(REPO_ROOT)}; "
        "`bench record --suite <name>` therefore resolves first-indoor and cannot "
        "resolve a held-out suite without a one-line change outside this stage's fence",
    )
    import inspect

    record_parameters = set(inspect.signature(live_record.record).parameters)
    ok &= check(
        receipt, "record_entry_point_accepts_a_document_and_a_seed",
        {"suite_document", "truth_seed", "suite_config"} <= record_parameters,
        "live_record.record(..., suite_document=, truth_seed=, suite_config=) is the "
        f"reachable path for a held-out suite; its keyword parameters are "
        f"{sorted(record_parameters)}, and "
        "tests/integration/test_first_indoor_mission.py drives the same entry point "
        "with a fixture suite",
    )
    ok &= check(
        receipt, "transport_requires_exactly_one_present_target",
        True,
        "READ FROM SOURCE, not executed: live_record.py:530-545 refuses a seed whose "
        "present-target count is not exactly one, with the reason 'the truth seed declares "
        "N present targets (...); the first-indoor mission searches for exactly one'. "
        "holdout-g-long truthfully declares zero, so it is refused there. It is not executed "
        "here because record() checks the host gate first (live_record.py:483) and flying "
        "past that gate is outside this probe",
    )

    receipt["pass"] = bool(ok)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"receipt: {args.output}")
    print("PASS: every held-out scene is declarable and its seed resolves"
          if ok else "FAIL: a held-out declaration did not hold up")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
