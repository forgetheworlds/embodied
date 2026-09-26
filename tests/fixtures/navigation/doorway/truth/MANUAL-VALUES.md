# Doorway fixture: hand-derived values and pre-registered tolerances

Provenance: **DIAGNOSTIC**. Every value is authored or derived in closed form from the
authored scene below. Nothing here was measured on a vehicle, and nothing here is
scored. The capture pose is simulator-scene truth, so no result from this fixture can
enter a sensor-derived gate.

This file was written from the derivation, before any comparison ran. Its tolerances
are pre-registered: a failed tolerance is a finding to report, never a number to
adjust (the P01-C F1 lesson, APPROVAL-RECORD.md:1053-1068). Floor-plane metric depth
is unresolved there, so **no floor surface is declared here** and no expected value
depends on floor depth.

## 1. Authored scene (odom, right-handed, gravity-aligned; x through the doorway,
y left, z up)

| Element | Value |
|---|---|
| wall plane | x = 3.0 m, y in [-2.0, 2.0], z in [-1.0, 2.4] (it continues below the opening, so downward rays return the wall and the space under the corridor is observed) |
| aperture | y in [-0.3, 0.9] (1.2 m wide), z in [0.0, 2.0] |
| far wall | x = 6.0 m, y in [-2.0, 2.0], z in [-1.0, 3.5] (it continues below the band anyone looks through, so downward rays still return a surface) |
| floor | none declared (see above) |
| view 1 body | (0.0, 0.0, 1.0) m, identity quaternion, t = 1.0 s |
| view 2 body | (-4.0, 0.0, 1.0) m, identity quaternion, t = 1.5 s |
| nav_epoch | `p03-epoch-1` |

The wall is declared as three rectangular patches around the aperture (left, right and
header), so `camera.declared_depth_map` (camera.py:678) reproduces the scene exactly.
Rays that pass the aperture continue to the far wall; a ray that leaves the far wall's
span returns nothing.

## 2. Camera and depth conventions (pinned rig, `first-indoor-stereo-1` v2)

* f = 554.256258422 px, principal point = (320.0, 240.0) px, baseline B = 0.1 m.
* Pixel convention (columns along body -y, rows along body -z, optical axis body +x):
  u = c_x - f (P_y - C_y)/(P_x - C_x) and v = c_y - f (P_z - C_z)/(P_x - C_x),
  with C the camera optical centre = body position + (0.05, 0.05, 0.05) m.
* Depth convention: z = f B / d metres along the left rectified optical axis. In the
  fixture's ray parameterisation the direction's x component is 1, so the declared
  depth equals the x-distance from the camera.
* Wall depth (camera at x = 0.05 m) = 3.00 - 0.05 = **2.95 m**;
  far-wall depth = 6.00 - 0.05 = **5.95 m**.
* Uncertainty envelope sigma_z = z^2 sigma_d / (f B) with the authored sigma_d =
  1.0 px: sigma_z(2.95 m) = **0.157012 m**,
  sigma_z(5.95 m) = **0.638739 m**. Transverse sigma = z sigma_px / f
  with sigma_px = 1.0 px: **0.005322 m** at the wall and
  **0.010735 m** at the far wall.
* A grounded target reports the *propagated* envelope: the depth term above plus the
  capture pose's declared 1-sigma bound of 0.02 m, added linearly rather than in
  quadrature because independence between the depth envelope and the shared pose bound is
  not established (section 7.3). So the aperture's normal and tangential terms are
  **0.177012 m** and **0.025322 m**, and the far-wall point's are
  **0.658739 m** and **0.030735 m**.
* Validity: border margin 8 px and every no-return sample are invalid.
  An invalid sample grounds nothing and clears nothing.

## 3. Selections (closed form)

The aperture selection is the outward-rounded projection of the wall rectangle
y in (-0.6, 1.2), z in (-0.05, 2.3): **box (103, 5, 443, 447)** in view 1.
The far-point selection is the single pixel **(297, 245)**, whose declared depth is 5.95 m, so

    p_odom = C + z (1, -(u-c_x)/f, -(v-c_y)/f) = (6.0, 0.296907451, 0.996324467) m

The invalid-depth selection is the box (0, 0, 5, 5): inside the declared border margin,
so every sample is masked and it grounds to nothing at all.

## 4. Aperture polygon (the one grounding path)

A pixel is *through* when its declared depth is finite and more than 0.5 m beyond the wall.
Each through pixel's ray meets the wall plane at

    y = C_y + (3.0 - C_x) (-(u-c_x)/f),  z = C_z + (3.0 - C_x) (-(v-c_y)/f)

The through pixels of view 1 span u in [161, 385],
v in [62, 430], giving wall-plane
y in [-0.295959, 0.896269], z in [0.038735, 1.997396].
Shrinking by the authored edge uncertainty 0.02 m (1 px of ray spread at the wall is
0.005322 m, so 0.02 m covers 3.76 px) gives the
**conservative opening polygon y in [-0.275959, 0.876269],
z in [0.058735, 1.977396]** on the plane x = 3.0, width
1.152228 m.

The polygon plane is the wall. The far wall at x = 6.0 is the depth *behind* the opening:
a ray through the doorway ends there, and that establishes visible space through the
aperture without locating the door frame. Ray direction is `-x` on the far side, so the
traversed direction is `+x`.

The observed opening is bounded by where rays that pass it still return a surface: the
observed z extent (0.058735..1.977396) is narrower than the authored
aperture (0.0..2.0). The polygon is therefore a **lower bound on the usable opening**,
never an outer bounding box, exactly as the specification requires for fit and traversal.

## 5. Fit and route (declared development parameters)

Voxel 0.1 m; inflation = body sphere 0.3 m + error allowance 0.1 m =
**0.4 m**. Inflated corridor width = 1.152228 - 2(0.4) = **0.352228 m > 0: it fits**.
The cell rule is `index = floor((value - lower_bound)/0.1)` in float64 over the authored
bounds (x [-5.0, 7.0], y [-4.0, 4.0], z [-0.5, 3.0]).
Inflation of the authored opening leaves the passable band y in
[0.124041, 0.476269] and the crossing plane x in [2.6, 3.4].

Narrow variant (aperture y in [-0.15, 0.55]): width 0.651918 m, corridor
-0.148082 m < 0, so the constraint named `aperture_clearance` rules the traversal out
with well-supported edges. Blocked variant (a door leaf filling the opening): zero through
pixels, so no free evidence exists through the aperture and no route can be found
(`no_known_supported_route`, execution blocked). Partial block (a leaf over y in
[0.7, 0.9]): width 0.955298 m, corridor 0.155298 m > 0, so the same planner
replans through the remaining observed free part.

## 6. Occupancy probes (closed form)

Log-odds hit +0.7 / pass -0.4, clamped to +/-4.0; free at score >= +1.4 with
at least 3 distinct clearing rays and age <= 5.0 s; occupied at score >= +1.4.
Each probe region below lists closed-form classes derived from the authored surfaces by
the rules in `_derived_probe_class`, before any map code ran:

* `free-before-wall` (free): x (2.35, 2.55), y (-1.15, -0.95), z (0.95, 1.15) - lines of sight from both views cross the whole region before the wall plane; every cell center projects to a valid pixel whose declared depth exceeds the cell's ray distance by more than a voxel
* `occupied-wall-band` (at_least_one_occupied): x (2.9, 3.1), y (-1.15, -0.95), z (0.95, 1.15) - the wall plane x=3.0 lies inside this band; rays from both views end in the cell holding the hit point (+0.7 log-odds each, clamp +4.0, occupied at >= +1.4)
* `free-through-aperture` (free): x (3.4, 4.6), y (0.25, 0.45), z (0.95, 1.15) - rays through the aperture end on the far wall x=6.0, so this region is strictly before the hit on valid rays: four passes at -0.4 = -1.6, and the ray count is far above the three required
* `free-just-before-far-wall` (free): x (5.35, 5.55), y (0.25, 0.45), z (0.95, 1.15) - as free-through-aperture, one half-voxel before the hit on the same rays
* `occupied-far-band` (at_least_one_occupied): x (5.9, 6.1), y (0.25, 0.45), z (0.95, 1.15) - the far wall plane x=6.0 lies inside this band, as occupied-wall-band
* `unknown-beyond-wall` (unknown): x (4.35, 4.75), y (1.75, 1.95), z (0.95, 1.15) - the wall spans y in [-2,2], so a ray toward this region meets the wall and stops; the rays that do pass the aperture reach no higher than y = 1.5 at x = 4.65 from either view, so the region receives no evidence at all: unknown, never free
* `unknown-outside-wall-span` (unknown): x (1.35, 1.75), y (-3.65, -3.45), z (0.95, 1.15) - rays toward this region leave the wall's y span and the far wall's y span, so the declared depth is a no-return and nothing is cleared there

## 7. Pose-degradation injections (coordinator scope addition)

Each injection is a declared construction with a provenance-carrying threshold, applied
by the tests to the nominal fixture. The nominal state carries a declared pose sigma of
0.02 m, which is 3 sigma = 0.06 m <= the 0.1 m allowance; the limit a state must
stay inside is therefore sigma <= 0.033333 m (three-sigma envelope inside the declared allowance).

* `stale`: declared pose validity 5.0 s (P03 plan section 4); the injected 6.0 s is validity plus 1.0 s of clock skew -> expected ['stale_pose', 'stale_state']
* `noise`: declared error allowance 0.10 m inside the 0.40 m inflation (P03 plan section 4); the injected 0.12 m is 20 percent beyond it -> expected ['pose_error_exceeds_allowance']
* `jump`: authored start-state tolerance 0.20 m; the injected 0.50 m is 2.5x it, the size of an estimator realignment that keeps the same nav_epoch -> expected ['start_state_mismatch']
* `epoch`: CONTRACTS.md:7 - a nav_epoch reset invalidates every prior control reference -> expected ['frame_epoch_mismatch']

## 8. Pre-registered tolerances

* `aperture_edge_abs_m` = 0.01
* `point_abs_m` = 0.02
* `uncertainty_abs_m` = 1e-06
* `setpoint_position_abs_m` = 1e-09
* `setpoint_speed_abs_mps` = 1e-09
* `setpoint_accel_abs_mps2` = 1e-09
* `settle_position_abs_m` = 0.1
* `settle_speed_abs_mps` = 0.02
* `probe_cell_rule` = cell index = floor((value - low_bound) / resolution) in float64
* `uncertainty_note` = an absolute tolerance, because the authored uncertainty literals are rounded to six decimal places and a relative bound on a small term would be tighter than that rounding
* `note` = pre-registered before any comparison ran; a failed tolerance is a finding, never a value to adjust
