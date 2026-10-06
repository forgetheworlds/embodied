# Control layer (Vehicle API v1)

Bottom API: anything that physically moves or rotates the aircraft under
ArduPilot GUIDED. Higher layers call this surface; they do not dig under it.

## Public API

```python
Vehicle.takeoff(altitude_m) -> Result
Vehicle.command(Motion) -> Result
Vehicle.land() -> Result
Vehicle.state() -> VehicleState
```

```python
@dataclass(frozen=True)
class Motion:
    position: Vec3      # odom (ENU: x north, y west, z up)
    velocity: Vec3      # odom
    yaw: float | None = None
    yaw_rate: float | None = None
```

- **Odom only** — no frame field on `Motion`. Vehicle owns odom→NED→MAVLink.
- **Caller-owned refresh** — each `command()` is one shot. If the caller stops
  publishing, Guided times out / fails closed. No last-motion republish and no
  auto-hold-on-cease inside Vehicle.
- Hold = position + zero velocity (+ optional yaw).
- Turns = held position + `yaw_rate` (unwind with opposite rate after a large turn).
- `takeoff` owns arm / GUIDED / EKF-origin bring-up. It prefetches until the
  local estimate is healthy (pose + home, EKF still aiding), skips arm spam
  while PreArm is flapping, keeps pumping sensors between attempts, and
  extends the wait when refusals are still recoverable (vis-odom / gyro rate /
  need-position). Permanent sensor death still fails closed.
- Typed `Result` for accept/refuse. `VehicleState`: armed, guided, position,
  velocity, yaw (+ landed when known).

There is **no** public `goto` / `hold` / `spin` helper surface.

## Live proof contract

`configs/layers/control` runs **one gapless caller loop** while GUIDED:

```text
while flying:
    choose current phase
    produce one Motion
    Vehicle.command(Motion)
    ~50 ms
```

Phase changes (outbound, hold, spin, reface, settle, align, return) only replace
the current Motion. Scoring happens on transition ticks without stopping
publication. Settle/align are explicit hold Motions when used — not silence.

## Layout

| Path | Role |
|---|---|
| `src/embodied/control/` | Layer package |
| `tests/test_control.py` | No-sim contract of the public API |
| `configs/layers/control` | Live sim verification script |
| `configs/layers/control.yaml` | Doorway route |
| `docs/control.md` | This note |

## Hard-won constraints

1. Never stop publishing while armed GUIDED (~50 ms cadence).
2. Refuse `command()` when not armed GUIDED.
3. Keep XY closed-loop during yaw-rate turns (position in `Motion`).
4. Doorway residual **0.10 m**. Load `compat_ekf.parm`.
5. Yaml waypoints are absolute local-NED; proof converts to odom ENU.
6. Prefer native Webots movie (`EMBODIED_WEBOTS_MOVIE`) over desktop
   screen capture — ffmpeg x11grab starves SITL/EKF and tip-strikes.
7. Pin Webots/SITL to reserved cores with `EMBODIED_SIM_CPUS` (default
   `0,1,2` from `./configs/layers/control`) so host load yields to the sim.

## Done bar

1. `pytest tests/test_control.py`
2. `./configs/layers/control` → `gate_status: pass`
3. This doc stays accurate

## How to run

```sh
.venv/bin/python -m pytest tests/test_control.py -q
./configs/layers/control
# movie: work/runs/layers/control/control-doorway.mp4
# skip movie: EMBODIED_WEBOTS_MOVIE= ./configs/layers/control
```
## Does not own

Path planning, mission budgets, perception, estimators, obstacle avoidance, or
replacing ArduPilot’s position controller.
