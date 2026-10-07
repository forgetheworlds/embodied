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

## Live producer (thin path)

Not a `mission_runtime` wrap. Small rebuild under `perception/`:

1. `stereo_imu_nav` — accel+gyro only (no Webots POSE / InertialUnit absolute RPY) → `NavigationState(evidence_class=sensor_derived)`
2. `compute_validated_depth` — stereo SGBM → `DepthProduct` with `PoseProvenance(SENSOR_DERIVED)`
3. `MapStoreMappingPort.integrate(..., nav=…)` — refuses non-sensor_derived capture nav
4. `OccupancyQuery` → `support=free` only when map evidence is sensor_derived

## Prove

- Contract: `tests/test_perception.py`, `tests/test_perception_pipeline.py` (no sim)
- Live: `./configs/layers/perception` → `perception-proof` receipt with `live_sensor_derived_free=true`

OpenVINS/`ov_stream` remains the scored localization pin for Vehicle ext-nav; Perception FREE live proof on this branch uses the thin stereo+IMU nav above when OV is not built.
