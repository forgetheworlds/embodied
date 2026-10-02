#!/usr/bin/env python3
"""Suite-resolution probe: does configs/suites/first-indoor.yaml register cleanly?

The CLI refuses a suite that is not in ``embodied.bench.recorder.SUITE_REGISTRY``,
and the registry is filled by the stage that owns live recording when it loads
the suite configuration. Until that loader ships, ``bench record --suite
first-indoor`` refuses on *resolution* alone, and the interesting question for
this stage's data is unanswerable from the CLI:

* is the configuration itself valid for ``SuiteSpec`` (name, registered_by,
  positive provider budget, non-empty localization mode)?
* once registered, does the CLI walk past resolution to the next gate — the
  transport refusal, and the localization-mode refusal for a mismatched mode?

This probe answers both without touching ``src/``: it loads the YAML, builds
the ``SuiteSpec`` exactly as the owning stage's loader will, registers it with
the real ``register_suite``, and then drives the real command dispatch
(``embodied.cli.main``) twice — the matching mode and a mismatched one — so the
receipts show which reason the CLI stops at.

It is a probe of the *data*: it registers nothing persistently (module state
only, inside this process) and it records that the live recording transport is
still J2's work, not something this probe supplies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCENE = Path(__file__).resolve().parents[1]
REPO_ROOT = SCENE.parents[1]
SUITE_YAML = REPO_ROOT / "configs" / "suites" / "first-indoor.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="receipt JSON path for the probe's own record")
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="directory the CLI receipts are written under")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import yaml

    from embodied.bench import recorder
    from embodied.cli import main as bench_main

    document = yaml.safe_load(SUITE_YAML.read_text(encoding="utf-8"))
    receipt: dict = {"probe": "first-indoor suite resolution", "suite_yaml": str(SUITE_YAML)}

    # 1. The configuration must construct a SuiteSpec as the owning stage will.
    spec = recorder.SuiteSpec(
        name=document["name"],
        registered_by=document["registered_by"],
        provider_budget=document["provider_budget"],
        localization_mode=document["localization_mode"],
    )
    receipt["suite_spec"] = {
        "name": spec.name,
        "registered_by": spec.registered_by,
        "provider_budget": spec.provider_budget,
        "localization_mode": spec.localization_mode,
    }
    receipt["suite_yaml_keys"] = sorted(document)
    print(f"[pass] suite YAML builds a SuiteSpec: {spec.name!r} registered_by "
          f"{spec.registered_by!r}, localization {spec.localization_mode!r}, "
          f"budget {spec.provider_budget}")

    # 2. Before registration the CLI must refuse on resolution (the current
    #    state of the checkout, with the loader not yet landed).
    fresh = args.runs_root / "record-unregistered"
    code_before = bench_main(
        ["bench", "record", "--suite", "first-indoor", "--sensor-mode", "sensor-derived",
         "--arm", "B0", "--output", str(fresh)]
    )
    before = json.loads((fresh / "receipt.json").read_text(encoding="utf-8"))
    receipt["cli_unregistered"] = {
        "exit_code": code_before, "status": before["status"], "reasons": before["reasons"],
    }
    print(f"[info] unregistered: exit {code_before} ({before['status']}): {before['reasons'][0]}")

    # 3. Register exactly as the owning stage's loader will, and re-run.
    recorder.register_suite(spec)
    registered = recorder.resolve_suite("first-indoor")
    receipt["resolved_after_registration"] = registered.name == "first-indoor"
    print(f"[pass] resolve_suite('first-indoor') -> {registered.name!r} after registration")

    matching = args.runs_root / "record-registered-mode-match"
    code_match = bench_main(
        ["bench", "record", "--suite", "first-indoor", "--sensor-mode", "sensor-derived",
         "--arm", "B0", "--output", str(matching)]
    )
    match_receipt = json.loads((matching / "receipt.json").read_text(encoding="utf-8"))
    receipt["cli_registered_mode_match"] = {
        "exit_code": code_match, "status": match_receipt["status"],
        "reasons": match_receipt["reasons"],
        "manifest_path": str(matching / "manifest.json"),
    }
    print(f"[info] registered, mode match: exit {code_match} "
          f"({match_receipt['status']}): {match_receipt['reasons'][0]}")

    mismatch = args.runs_root / "record-registered-mode-mismatch"
    code_mismatch = bench_main(
        ["bench", "record", "--suite", "first-indoor", "--sensor-mode", "pose-assisted",
         "--arm", "B0", "--output", str(mismatch)]
    )
    mismatch_receipt = json.loads((mismatch / "receipt.json").read_text(encoding="utf-8"))
    receipt["cli_registered_mode_mismatch"] = {
        "exit_code": code_mismatch, "status": mismatch_receipt["status"],
        "reasons": mismatch_receipt["reasons"],
    }
    print(f"[info] registered, mode mismatch: exit {code_mismatch} "
          f"({mismatch_receipt['status']}): {mismatch_receipt['reasons'][0]}")

    # 4. The verdicts this probe can honestly assert about its own data.
    passed = (
        receipt["resolved_after_registration"]
        and "not registered" in before["reasons"][0]
        and "no live recording transport" in match_receipt["reasons"][0]
        and "declares localization mode" in mismatch_receipt["reasons"][0]
    )
    receipt["checks"] = {
        "suite_yaml_builds_suitespec": True,
        "unregistered_is_refused_on_resolution": "not registered" in before["reasons"][0],
        "registered_mode_match_reaches_transport_refusal":
            "no live recording transport" in match_receipt["reasons"][0],
        "registered_mode_mismatch_is_refused": "declares localization mode" in mismatch_receipt["reasons"][0],
        "pass": bool(passed),
    }
    receipt["note"] = (
        "The probe registers the suite in this process only, standing in for the "
        "stage-owned loader that is not yet in the checkout. The transport refusal "
        "it reaches is J2's remaining work; nothing here claims a live episode."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"receipt: {args.output}")
    print("PASS: the suite data registers cleanly and the CLI stops at the expected gates"
          if passed else "FAIL: unexpected CLI behaviour")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
