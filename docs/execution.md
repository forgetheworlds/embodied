# Safety + Execution

Frozen surface: project store `docs/execution-safety-frozen.md`.

## Public API

- `safety.check(...) → ALLOW | BACKUP | UNSUPPORTED` (pure; never commands)
- `Execution.replace(certificate)` / `status()` — sole `Vehicle.command` writer via `_tick`

Package: `src/embodied/execution/`

## Prove

- Contract: `tests/test_execution.py`
- Live: `./configs/layers/execution` → takeoff → replace → Safety-gated publishes → land

Receipt records `ap_ext_nav_mode` separately from Perception `evidence_class`. Geometry CLEAR is not claimed without sensor_derived FREE.
