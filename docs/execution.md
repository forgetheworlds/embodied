# Execution (sole Motions writer)

**Owner:** Execution agent only. Consumes Safety.check + Perception ports; does not own those layers.

Frozen surface: project store `docs/execution-safety-frozen.md` (joint contract). Safety honesty: CLEAR only FREE+`sensor_derived`+epoch match (`tests/test_safety.py`).

## Public API

- `Execution.replace(certificate)` / `status()` / `_tick`
- Sole `Vehicle.command` publisher under Safety leases
- Package: `src/embodied/execution/` (`execution.py`, certificates, trajectories; `safety.py` owned by Safety agent)

## Prove

- Contract: `tests/test_execution.py` (replace / terminal / sole-writer; not Safety.check)
- Joint live entry: `./configs/layers/execution`
  - `execution_proof.require_geometry_clear: false` until Perception live FREE
  - When true: FAILS unless live occupancy returns `free`+`sensor_derived` (no invented FREE)
  - Receipt: `ap_ext_nav_mode` vs Perception `evidence_class` + `occupancy_summary`
