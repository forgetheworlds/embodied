# first-indoor — the P05 evaluation scene

One vestibule spawn and two rooms; exactly one red block, in one of the rooms.
The mission is `search_inspect_return`: "Find the red block, inspect it, and
return to the start." This scene serves GOAL.md's first test — enter unfamiliar
indoor rooms, find the room with the red object, inspect it, return — under an
evaluator whose hidden task truth the pilot never sees.

## What lives here

| Path | Role |
|---|---|
| `world.wbt` | The whole physical scene. The only place geometry exists. |
| `mission.yaml` | The whole public task surface: instruction, spawn, devices, budgets. Deliberately names no route, no doorway order, no target placement. |
| `truth.yaml` | **Evaluator-only.** The hidden `world_state` facts the referee writes into the bench-side store at record time. |
| `assets/` | Vendored Apache-2.0 appearance protos + textures, sha256-traced in `assets/PROVENANCE.md`. |
| `catalogue.yaml` | Input to the existing headless load validator (`scenarios/missions/tools/validate_scenarios.py`, unchanged). |
| `tools/verify_scene_geometry.py` | Reachability/visibility check from the `.wbt` geometry (verifier for "the red object is reachable and visible from somewhere the aircraft can legally be"). |
| `tools/probe_truth_isolation.py` | Re-proves, against the live bench code, that the agent surface cannot reach this scene's truth or the per-episode store. |
| `tools/capture_stereo_pairs.py` | Captures stereo pairs at declared poses with **no flight**: derives a capture world from `world.wbt` at run time, rewrites only the airframe's controller block, and refuses to run if any node name, any `Solid` or the text before that block differs. Poses: `tools/capture_poses.json`. |
| `tools/measure_depth_on_pairs.py` | Per-pose and pooled depth-rejection breakdown over captured pairs, reading the matcher, border and window out of `configs/first_indoor.yaml` the way the runtime does. |

## The honesty contract of this scene

1. **Nothing here encodes the answer.** `mission.yaml` is the pilot-safe
   metadata by construction: it carries the instruction, spawn pose, device
   list and budgets — no route, no doorway order, no target location. The
   route exists only as geometry in `world.wbt` and is discovered by flying.
   There is deliberately no `layout.yaml`: its one legitimate content here
   (target placement) is exactly what scene metadata must not carry.
2. **Truth is bench-side twice over.** The facts live in `truth.yaml`
   (bench-side seed), and the per-episode store the grader reads is a sibling
   `<episode>.truth/` directory written by `embodied.bench.referee` — outside
   the episode directory, which is all the runtime side is handed. The agent
   surface (`embodied.bench.recorder.PROJECTION`) structurally refuses both.
3. **The decoy is furniture, not a target.** `decoy_box` is near-red with the
   wrong shade. `truth.yaml` does not declare it, so a report that mistakes it
   for the red block grades as unverifiable against no target — and its
   presence can never count as a missed present target.
4. **No fabricated outcomes.** `truth.yaml` seeds only `world_state` (what the
   world contains). `physical_outcome` — inspected, return, violations,
   takeover — is measured per live run by the referee; it is never seeded.

## Verifying the scene

Headless load (no SITL, rendering off), from the repository root:

```bash
python3 scenarios/missions/tools/validate_scenarios.py \
  --catalogue scenarios/first_indoor/catalogue.yaml \
  --output work/runs/p05/validation \
  --webots work/Webots.app/Contents/MacOS/webots
```

Geometry reachability and truth isolation, both pure Python:

```bash
python3 scenarios/first_indoor/tools/verify_scene_geometry.py
python3 scenarios/first_indoor/tools/probe_truth_isolation.py
```

## Measuring the scene at fixed poses, without flying

The depth pipeline's behaviour on this scene is measurable at **fixed** poses, which
a flight cannot give: a flight moves the camera, so any before/after comparison
across flights confounds the scene with whatever route each run happened to fly.

```bash
PYTHONPATH=src python3 scenarios/first_indoor/tools/capture_stereo_pairs.py \
  --out-dir work/runs/p05/scenetexture/before
PYTHONPATH=src python3 scenarios/first_indoor/tools/measure_depth_on_pairs.py \
  --pairs work/runs/p05/scenetexture/before --label before \
  --out work/runs/p05/scenetexture/before-depth.json
```

The controller poses the airframe kinematically — it re-asserts the robot's own
translation and rotation on every step, so the pose at capture is the declared pose
rather than whatever gravity did with it — and reads both cameras in a single step.
It starts no autopilot, arms nothing, and needs no SITL and no ports, so it runs
while siblings are flying.

Two things worth knowing before changing it. Webots does **not** search the world's
own directory for a controller: it searches `scenarios/controllers/` and
`scenarios/compat/controllers/`, and its warning names every path it tried — which is
why this controller lives in the former. And Webots relays a controller's stdout only
when that process exits, so the launcher waits on the pairs appearing on disk rather
than on the console marker.
