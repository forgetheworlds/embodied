# Architecture audit — readability, abstractions, simplicity, quality

Date: 2026-10-05  
Scope: `src/embodied/` (and how it relates to `configs/`, `scenarios/`, `estimator/`, `tests/`)  
Intent: personal assessment only. No refactor planned from this document.

## Verdict

The project's **design ideas are strong**; the **module layout does not match them**.

Evidence discipline (receipts, frozen records, truth isolation, pinned deps) is high quality. Dig-depth is high because ~46% of `src/embodied` Python lives in three files, and higher layers reach into those files for helpers that should be shared primitives.

| Criterion | Score (1–5) | One-line reason |
|---|---|---|
| Readability | 2 | Local prose is careful; finding a helper means opening a 6k-line module |
| Abstractions | 2.5 | Package names look layered; `platform` sits on top and leaks private APIs upward |
| Simplicity | 2 | Same helpers duplicated; adapter / check / mission overlap |
| Quality (evidence) | 4 | Pins, receipts, gates, deferred-record honesty |
| Quality (structure) | 2.5 | Structure fights the principles the README states |

## What already works — keep

These are the real strengths. Any future cleanup should preserve them.

1. **Frozen records, missing stays missing** — `contracts/records.py`  
   One definition per record, no fabricated defaults, stdlib-only import, revision tag (`RECORDS_REVISION`). Deferred names are listed instead of silently reinvented.

2. **Commands write receipts** — `cli.py`  
   Shared outcome shape: revision, config hash, gate status, artifact hashes, fixed exit codes.

3. **Score cannot see the answer key** — `bench/`  
   Recorder / referee / grader split; truth store beside the episode, not inside it.

4. **Named replaceable seams** — in the adapter docstring  
   `ProcessRunner`, `MavlinkSession`, `SensorGateway` are the intended test boundaries without Webots/SITL.

5. **Honest status reporting** — README / receipts  
   Failures and bound re-declarations are kept, not edited away.

## Size and concentration

Approximate at audit time (`src/embodied`, ~46 `.py` files, ~37k lines):

| File | Lines | Role mix |
|---|---|---|
| `platform/webots_ardupilot.py` | ~6685 | Framing, FDM, codecs, settings, subprocess, MAVLink, timebase, probe, evidence, adapter |
| `platform/localization_check.py` | ~6583 | Config blockers, bring-up, live scoring, diagnostics, end-state, CLI registration |
| `platform/mission_runtime.py` | ~3934 | Full mission loop (~2680-line class) plus world/admission helpers |
| `bench/live_record.py` | ~1603 | Live episode recording / predicates |
| `perception/camera.py` | ~1383 | Calibration + depth validation |
| `platform/localization.py` | ~1336 | Estimator seam + health + publish |
| `contracts/records.py` | ~1216 | Shared schemas (appropriate size for its job) |
| `navigation/planner.py` | ~1098 | Planning + certification |

Top three files ≈ **46%** of the package. That is the dig-depth problem.

## Structural findings

### 1. God modules, not deep modules

`webots_ardupilot.py` describes itself as a deep module with a small surface. In practice it is a catalogue: ENU/NED, EMB1 framing, pair/IMU/pose codecs, subprocess runners, pymavlink session, bring-up, TCP gateway, timebase join, compatibility probe, observation building, evidence writing.

`localization_check.py` is orchestration + scoring + many one-off blockers/helpers, including two ~1000-line live/bring-up runners.

`MissionRuntime` is a 50+ method class that wires localization, map, calibration, pilot, navigation, and recording in one place.

### 2. Dependency direction is inverted

Intended mental model: platform at the bottom, pilot/nav/bench above.

Observed package imports (A → B means A imports B):

- `platform` → memory, navigation, perception, pilot, contracts, cli  
- `navigation` → contracts, memory, perception, **platform**  
- `bench` → contracts, pilot, **platform**  
- `perception` → contracts only (cleaner)

Concrete leaks:

- `navigation/executor.py` and `bench/live_record.py` import `enu_to_ned` from `webots_ardupilot`
- `mission_runtime` imports a long list of **private** `_…` symbols from `localization_check` (bring-up, GPS verdict, crash disarm parsing, sim window, estimator start, etc.)

Private helpers crossing module boundaries are the opposite of a stable primitive layer.

### 3. Contracts stop at data

`records.py` freezes schemas and stamps. Runtime primitives are not centralized:

| Concern | Where it lives today | Duplication |
|---|---|---|
| Frame transform (`enu_to_ned`) | `webots_ardupilot` | imported by nav + bench |
| Timebase join | `webots_ardupilot` | probe-local |
| `sha256(path)` | cli, localization_check, camera, bench/recorder | 4 copies |
| `_port_is_free` | webots_ardupilot, localization_check | 2 copies |
| `_percentile` | localization_check, pilot/probe | 2 copies |
| EMB1 / sensor codecs | webots_ardupilot (also used by Webots controller) | correctly shared, wrongly housed |

Deferred records (`TrajectoryCertificate`, `AcceptedGoal`, …) already appear as local types in navigation with comments that P03 is the “first real writer” — schema ownership is drifting before the contract catches up.

### 4. There is no stable primitives layer yet

Reusable pieces are either:

- record types in `contracts/`, or  
- incidental functions inside the megamodule you were trying not to open.

So every new feature pays a “find it in the 6k-line file” tax, or reinvents a helper.

## What is *not* the problem

- Lack of documentation intent — README and module docstrings are unusually explicit.
- Lack of tests — large suite; failures are often treated as evidence, not shame.
- Lack of config discipline — declared bounds and pins are a real asset.
- Need for a framework rewrite — the ideas are fine; the file boundaries are not.

## If a primitives layer were ever added (not doing it now)

A thin base that anything may import without starting a simulator:

```
contracts/     # keep as-is: records, enums, stamps
primitives/    # pure helpers only
  frames.py      # enu_to_ned, quat/rotmat, wrap_angle
  timebase.py    # join / fit timebase
  hashio.py      # sha256, artifact digests
  stats.py       # percentile, shared numeric checks
  framing/       # EMB1 + pair/imu/pose codecs (controller-safe)
platform/      # process + MAVLink + gateway only
gates/         # localization scoring / bring-up (split from check)
mission/       # MissionRuntime orchestration only
```

Rule of thumb: if two packages need it and it does not start a simulator, it belongs in `primitives/` or `contracts/`.

Suggested order if ever tackled: extract pure primitives with re-export shims first; only then carve the three megamodules. Full platform split without a primitives base just moves the dig.

## Approaches considered (deferred)

| Option | What | When it would make sense |
|---|---|---|
| A — Primitives extraction | Lift pure helpers; leave orchestration; shims for imports | First cleanup when dig-depth blocks work |
| B — Full platform carve-up | A + split adapter / gate / mission with public APIs only | After A is stable |
| C — Import lint + map only | Dependency rules, no code moves | Cheap guardrail alongside other work |

None of these are in scope for this audit.

## Bottom line

Keep the evidence and contract culture. The dig cost is almost entirely **three platform megamodules plus missing shared primitives**. When that cost starts blocking other work, start with option A — not a redesign of the pilot, bench, or records.
