# Figures

Charts generated from real run logs in this project's local evidence store.
Every number plotted comes from the file named in the caption; nothing is
invented and no channel is smoothed.

| Figure | Source files |
|---|---|
| `failed-run-yaw-motors.svg` | `work/runs/p00-airframe/accept-mission-12-2026-09-25T1714Z/run-a/mavlink.jsonl` |
| `pass-a-run-yaw-motors.svg` | `work/runs/p00-airframe/extnav-it3-2026-09-25T1834Z/run-a/mavlink.jsonl` and `run-a/motion.jsonl` |
| `pass-b-run-yaw-motors.svg` | `work/runs/p00-airframe/extnav-it3-2026-09-25T1834Z/run-b/mavlink.jsonl` and `run-b/motion.jsonl` |
| `waypoint-residuals.svg` | `motion.jsonl` of `extnav-it3-2026-09-25T1834Z` and `extnav-it4-2026-09-25T1838Z`, runs a and b |

The `work/` evidence store is local to the development machine and is not part
of this repository; the figures and these captions are what is published.

**failed-run-yaw-motors.svg** — A failing gate attempt. The aircraft arms at
t≈6 s, climbs to altitude, and from t≈14 s the yaw rate departs, swinging
between −176°/s and +114°/s while the motor outputs saturate near 1950 µs.
The autopilot's crash detector disarms at the red line
(`Crash: Disarming: AngErr=170>30`); guided motion never starts, and the two
waypoint residuals for this run were 255.0 cm and 241.5 cm. Twelve seconds
after the crash the log records `GPS Glitch or Compass error` — the estimator
and the fused compass no longer agreed. Channels: `ATTITUDE.yawspeed`
(converted to °/s) and `SERVO_OUTPUT_RAW` channels 1–4 (µs, autopilot demand,
about 4 Hz), stamped with the host's monotonic receive clock.


**pass-a-run-yaw-motors.svg** — The same probe after the fix, run a. The yaw
trace stays inside ±82°/s; the only transient is the decaying wobble during
the 2 m transit to waypoint 1. Motors peak near 1430 µs at the takeoff climb
and settle to the ~1310 µs hover. The dashed blue lines mark the moments the
bridge published each waypoint; the holds measured 3.18 cm and 2.01 cm from
the commanded local-NED target. Same channels and clock as the failed-run
figure.

**pass-b-run-yaw-motors.svg** — The second run of the same passing
invocation. Holds measured 2.70 cm and 2.00 cm.

**waypoint-residuals.svg** — Tracking residual at the hold deadline (8 s per
waypoint) for the four runs of the two consecutive passing invocations:
1.8–3.2 cm across all eight holds. For contrast, the failed configuration
measured 255.0 cm and 241.5 cm on the same probe.

Two honest limits on what these charts show. The yaw rate is the autopilot's
own attitude estimate, which in the passing configuration is fed by the
simulator's ground-truth pose (see the README's status section), so the plots
are compatibility evidence, not independent sensing. Motor outputs are
commands the autopilot sent, not measured thrust.
