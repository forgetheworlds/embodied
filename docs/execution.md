# Execution (sole Motions writer)

**Owner:** Execution agent only. Consumes Safety.check + Perception ports; does not own those layers.

**Rebuild rule:** clean sole-writer on Vehicle + new Safety + Perception ports. No `MissionRuntime` / old navigation dual-writer glue (`prefer-rebuild`).

Frozen surface: project store `docs/execution-safety-frozen.md`. Safety honesty: CLEAR only FREE+`sensor_derived`+epoch match.

## Public API

- `Execution.replace(certificate)` / `status()` / `_tick`
- Sole `Vehicle.command` publisher under Safety leases
- Package: `src/embodied/execution/` (`execution.py`, certificates, trajectories; `safety.py` owned by Safety agent)

## Prove

- Contract: `tests/test_execution.py`
- Joint live: `./configs/layers/execution` — mapping via Perception `compose_mapping_port` if present, else stub (no MapStore in Execution)
  - `require_geometry_clear: false` until Perception live FREE; then flip + re-prove
  - FAIL if CLEAR required without live `free`+`sensor_derived`
