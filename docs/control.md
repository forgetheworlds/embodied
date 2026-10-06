# Control layer (Vehicle API v1)

Bottom API: anything that physically moves or rotates the aircraft under
ArduPilot GUIDED. Higher layers call this surface; they do not dig under it.

## One doc · one test · one proof

| Path | Role |
|---|---|
| `docs/control.md` | This note |
| `tests/test_control.py` | No-sim contract of the public API |
| `configs/layers/control` | Live Webots + SITL proof (`control.yaml` route) |

```sh
.venv/bin/python -m pytest tests/test_control.py -q
./configs/layers/control          # exit 0 iff receipt gate_status is pass
```

Optional movie (native Webots, not desktop ffmpeg):

```sh
EMBODIED_WEBOTS_MOVIE=work/runs/layers/control/control-doorway.mp4 ./configs/layers/control
```

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

- **Odom only** — Vehicle owns odom→NED→MAVLink.
- **Caller-owned refresh** — each `command()` is one shot; stop publishing and
  Guided fails closed. No last-motion republish inside Vehicle.
- Hold = position + zero velocity (+ optional yaw).
- Turns = held position + `yaw_rate` (unwind with opposite rate after a large turn).
- `takeoff` owns arm / GUIDED / EKF-origin bring-up (prefetch, recoverable PreArm).
- No public `goto` / `hold` / `spin` helpers — compose `Motion` instead.

## Live proof

One gapless caller loop while GUIDED:

```text
while flying:
    choose current phase Motion
    Vehicle.command(Motion)
    ~50 ms
```

Phases (outbound, hold, spin, reface, settle, align, return) only replace the
current Motion. Settle/align are explicit holds — not silence. Doorway residual
gate: **0.10 m**. Yaml waypoints are absolute local-NED; proof converts to odom ENU.

## Constraints that matter

1. Never stop publishing while armed GUIDED (~50 ms).
2. Refuse `command()` when not armed GUIDED.
3. Keep XY closed-loop during yaw-rate turns (position in `Motion`).
4. Load `compat_ekf.parm`.
5. SITL must clamp Webots FDM time jumps
   (`patches/ardupilot-sitl-webots-fdm-time-clamp.patch`) — otherwise
   `time_boot_ms` leaps and VisOdom dies.
6. Prefer `EMBODIED_WEBOTS_MOVIE` over desktop capture. Delay past PreArm
   (`EMBODIED_WEBOTS_MOVIE_DELAY_S`, default 90), finish mid-run
   (`EMBODIED_WEBOTS_MOVIE_DURATION_S`, default 180). No CPU pinning.
   Software-GL movie may look stuttery; that is capture, not the flight.

## Does not own

Path planning, mission budgets, perception, estimators, obstacle avoidance, or
replacing ArduPilot’s position controller.
