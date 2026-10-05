# Control layer

Bottom API: anything that physically moves or rotates the aircraft under
ArduPilot GUIDED. Higher layers call this surface; they do not dig under it.

## Layout

| Path | Role |
|---|---|
| `src/embodied/control/` | Layer package (`Vehicle`, live proof harness) |
| `tests/test_control.py` | Deterministic contract tests — **no** Webots/SITL |
| `configs/layers/control` | Live sim verification script for this layer |
| `configs/layers/control.yaml` | Scene + route the script flies |
| `docs/control.md` | This note |

Per-layer convention (repeat for future layers):

- package: `src/embodied/<layer>/`
- unit: `tests/test_<layer>.py` (flat `tests/`, no live flight)
- sim: `configs/layers/<layer>` (+ matching `.yaml` when needed)
- doc: `docs/<layer>.md`

## API (`Vehicle`)

Owned primitives:

- `takeoff(altitude_m)` — waits for EKF/home before `NAV_TAKEOFF`
- `goto(north, east, down_m=…, hold_s=…, yaw_rad=…)` — local-NED hold
- `hold(duration_s, yaw_rad=…)` — hold current (or last) NED pose
- `spin(angle_rad, rate_rad_s=…)` — **yaw-rate** spin, never absolute yaw slam
- `land()` — `LAND` mode

Sim bring-up (Webots, SITL, MAVLink session, sensors) stays in `platform/`.
Control only publishes setpoints once the adapter reports armed GUIDED.

## Frames and time

- Waypoints for the doorway proof are **absolute EKF local-NED**, not
  spawn-relative offsets. Iris spawn in `first_indoor` is ENU `(-1,0,*)` →
  NED `(-1,0,*)`; EKF origin is world ENU origin.
- Hold / spin / goto deadlines use autopilot **sim time** (`time_boot_ms`),
  not wall clock. Wall clock is only a backstop when realtime drifts.
- Always load `compat_ekf.parm` (via `scenario.estimator_params`). Without EKF,
  GUIDED diverges mid-route.

## Hard-won constraints

1. **Refresh GUIDED setpoints** (~`REFRESH_S = 0.05`). Stop publishing and the
   vehicle stops being guided.
2. **Refuse when not armed GUIDED** — `publish` returns `None` and the proof
   scores `guided_lost` rather than commanding blind.
3. **Spin is yaw-rate only, with position held.** Absolute yaw after a large
   heading change tip-strikes (`Crash: AngErr=…`). Ignoring XY during spin also
   tip-strikes on the next goto when the horizontal controller re-engages.
   After a +π spin, reface with `spin(-π)`, not `hold(yaw_rad=0)`.
4. **Doorway residual** for this layer proof is **0.10 m**; open-field scenes
   historically used 0.15 m.
5. Return legs reverse the outbound chain (skipping the far endpoint) only after
   reface succeeds.

## Done bar for this layer

All three must pass before stacking the next layer:

1. `pytest tests/test_control.py`
2. `configs/layers/control` → receipt `gate_status: pass`
3. This doc stays accurate

## How to run

```sh
# Contract only (fast)
.venv/bin/python -m pytest tests/test_control.py -q

# Live sim (Webots + SITL; minutes)
./configs/layers/control
```

Evidence lands under `work/runs/layers/control/` (`receipt.json`,
`run-a/motion-proof.json`, mavlink/sitl/webots logs).

Optional: `EMBODIED_WEBOTS_VISIBLE=1` keeps the Webots window open for watching.

## What this layer does **not** own

Path planning, mission timing budgets, perception, estimators, obstacle
avoidance, or replacing ArduPilot’s position controller. Those compose
`Vehicle`; they do not reimplement it.
