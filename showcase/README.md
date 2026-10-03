# Showcase assets

Every figure here is generated from data already on disk. Nothing is drawn by hand.

Regenerate everything from the repository root:

```
cd /Users/muadhsambul/embodied
PYTHONPATH=src python3 showcase/make_aircraft_view.py     # aircraft-view.png
PYTHONPATH=src python3 showcase/make_video.py             # aircraft-view-J30-move-1.mp4
              python3 showcase/make_trajectory.py         # trajectory-live-14.png
              python3 showcase/make_scene.py              # scene-plan.png
              python3 showcase/make_more.py               # launch-outcomes, perception-and-map,
                                                          # depth-rejections, cloud-latency,
                                                          # integration-cost
```

`make_aircraft_view.py` and `make_video.py` import the project's own depth pipeline
(`embodied.perception.camera.compute_validated_depth`), so they need `PYTHONPATH=src` from a
checkout root that contains `src`. The others only read stored JSON and MAVLink records.

| asset | source |
|---|---|
| `aircraft-view.png` | `work/runs/p05/live-14/episode/payloads/obs-00002-*.ppm` + `configs/first_indoor.yaml` |
| `aircraft-view-J30-move-1.mp4` | `work/runs/p05/J30-move-1/episode/payloads/` (22 pairs) |
| `trajectory-live-14.png` | `work/runs/p05/live-14/platform/run-a/mavlink.jsonl` |
| `scene-plan.png` | `scenarios/first_indoor/world.wbt` + `mission.yaml` + `live-14/.../refused-publications.jsonl` |
| `launch-outcomes.png` | every `work/runs/p05/*/mission.json` |
| `perception-and-map.png` | every `work/runs/p05/*/episode/agent-events.jsonl` and `mission.json` |
| `depth-rejections.png` | `work/runs/p05/J27-breakdown.json` |
| `cloud-latency.png` | `work/runs/p05/J6-arm-{continuous,initial}.json` |
| `integration-cost.png` | `work/runs/p05/J24-sim-REPORT.md` |

The figures are a snapshot: `launch-outcomes.png` and `perception-and-map.png` count whatever runs
existed when they were generated, and later runs will change those counts.
