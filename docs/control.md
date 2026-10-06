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
  auto-hold-on-cease.
- Hold = current/target position + zero velocity (+ optional yaw).
- Turns = hold position + `yaw_rate` (never absolute yaw slam after a large
  heading change; unwind with opposite rate).
- `takeoff` owns arm / GUIDED / EKF-origin bring-up.
- Typed `Result` for accept/refuse (not Guided, bad numbers, yaw+yaw_rate, …).
- `VehicleState`: armed, guided, position, velocity, yaw (+ landed when known).

There is **no** public `goto` / `hold` / `spin` helper surface.

## Layout

| Path | Role |
|---|---|
| `src/embodied/control/` | Layer package |
| `tests/test_control.py` | No-sim contract of the public API |
| `configs/layers/control` | Live sim verification script |
| `configs/layers/control.yaml` | Doorway route |
| `docs/control.md` | This note |

Per-layer convention: `src/embodied/<layer>/`, `tests/test_<layer>.py`,
`configs/layers/<layer>`, `docs/<layer>.md`.

## Hard-won constraints

1. Refresh GUIDED setpoints from the **caller** (~50 ms). Stopping fails closed.
2. Refuse `command()` when not armed GUIDED — never publish blind.
3. Keep XY closed-loop during yaw-rate turns (position held in `Motion`).
4. Doorway residual for the layer proof is **0.10 m**.
5. Always load `compat_ekf.parm`. Waypoints in the yaml are absolute local-NED;
   the proof converts them to odom ENU before building `Motion`.
6. After a ±π yaw-rate pair: brief yaw-ignored settle, then inbound with
   `yaw=0`. Leaving yaw ignored for the whole return lets heading drift and
   tip-strike at the next doorway.

## Done bar

1. `pytest tests/test_control.py`
2. `./configs/layers/control` → receipt `gate_status: pass`
3. This doc stays accurate

## How to run

```sh
.venv/bin/python -m pytest tests/test_control.py -q
./configs/layers/control
```

Evidence: `work/runs/layers/control/` (`receipt.json`, `run-a/motion-proof.json`).

## Does not own

Path planning, mission budgets, perception, estimators, obstacle avoidance, or
replacing ArduPilot’s position controller.
