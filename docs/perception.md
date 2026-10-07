# Perception / Map (stacked ports)

Frozen surface: project store `docs/perception-ports-frozen.md`.

## Public API

- `EstimationPort.latest() → NavigationState | None`
- `MappingPort.occupancy() → OccupancyQuery | None`
- Types in `embodied.contracts.perception_ports`

## Honesty

- Only `evidence_class=sensor_derived` may yield occupancy `FREE`.
- Stub / empty map → `unsupported`, never free world.
- AP truth VPE is orthogonal (platform); must not set Perception `evidence_class`.

## Prove

- Contract: `tests/test_perception.py` (no sim)
- Live sensor_derived FREE corridor: needs OV capture pose (not claimed until OV wired). Ports FREE honesty covered in pytest + joint execution proof in-process check.
