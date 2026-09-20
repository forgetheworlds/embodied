# embodied

Building a drone system that can handle unfamiliar objectives in unfamiliar places: gather information, navigate, reason about observations, revise its approach, and report a supported result. The long-term goal is judgment comparable to a capable human pilot, within the aircraft's sensing, energy, and control limits.

**Status: repository foundation and simulation research. The integrated drone system is not implemented.** No supported flight command, drone test suite, or generalization result exists yet.

## The problem

Finding a river, following it downstream, locating a cabin, and searching the surrounding area requires more than waypoints. The system must discover relevant places, choose useful viewpoints, remember observations, handle changed instructions and failures, manage resources, and decide whether its evidence is sufficient. These examples describe the intended judgment, not a fixed task menu.

## Intended system

The working design connects a flight platform, perception and belief, spatial and episodic memory, a mission executive, and an independent evaluator/recorder. Higher-level reasoning cannot bypass control or safety. An action request is not proof of arrival; a model label is not identity proof; and a citation is not proof that its source supports the claim.

The exact control implementation, scored sensors, and first integrated experiment remain design decisions. Reusable skills and conventional components are valid tools; an original control law remains a recorded research requirement under review.

## What has been accomplished

Local research exercised Webots camera/control interfaces and recorded a shipped-controller flight. It exposed issues with motor-command conventions, camera transforms, timing measurements, and earlier implementation plans. These local artifacts are not a reproducible public release or evidence of integrated autonomy.

The repository foundation now separates current agent context from archived planning, preserves design history on a dedicated branch, and provides setup and integrity checks. No drone capability is claimed from these checks.

## Check this checkout

Foundation scripts require Python 3.9+ and Git 2.31+, with no third-party Python packages, model calls, or network access:

```sh
python3 scripts/check_repository.py
```

The public check does not require Atomic, Webots, private design documents, or cloud credentials.

For local agent development, make the local `design` branch available, then run:

```sh
python3 scripts/setup_agent.py
python3 scripts/check_repository.py --agent-context
```

The installer mounts the design branch at `design/` and installs a small ignored `AGENTS.md`. It refuses to overwrite different instructions unless `--replace` is supplied; replacement first saves a backup. It does not fetch branches, install dependencies, start Atomic, or run workflows. A main-only public checkout can skip agent setup.

Start Atomic from the code checkout. Its normal context loading reads `AGENTS.md`, which routes to a bounded startup packet. Dynamic workflows are authored for an approved slice using installed Atomic documentation. Workflow execution and model costs are separate from repository validation.

## Repository boundaries

`main` contains code, related tests/configuration/scenario assets, and this README. The `design` branch contains current docs and the historical archive, mounted locally at `design/`. Runtime state and large experiments stay in ignored `.atomic/` and `work/`. Do not merge the design branch wholesale into main. Publishing main does not require publishing private planning history.

## Reproduction and reporting

Every future released experiment must include pinned dependencies and assets or retrieval instructions, permitted sensors and priors, model identities, scenarios and seeds, exact run/scoring commands, expected artifacts, and measured limitations. The supported no-cloud baseline must run from a clean code checkout without private design documents.

Report mission success, false claims, uncertainty, interventions, safety violations, resource use, and failures. Compare credible baselines on held-out layouts, appearances, and goal compositions. Add images and synchronized video when real artifacts exist. Simulation success alone will not be described as physical flightworthiness or human-level performance.
