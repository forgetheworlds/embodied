#!/usr/bin/env python3
"""Reachability and visibility verifier for the first-indoor scene.

Answers, from the scene's own geometry (world.wbt), the verifier question
"the red object is reachable and visible from somewhere the aircraft can
legally be" without launching a simulator:

1. parses every Solid in world.wbt into an axis-aligned box (the scene is
   built entirely of axis-aligned boxes and one floor plane),
2. walks a declared waypoint chain — an explicit verification INPUT passed
   on the command line, never stored in scene metadata — and measures the
   exact distance from every flight leg to every solid,
3. from the final waypoint, casts a sight-line at the target and checks it
   is unobstructed and inside the camera frustum of the Iris stereo pair
   (fieldOfView 1.0472 rad horizontal at 640x480, per Iris.proto),
4. checks every waypoint altitude is inside the mission budget's declared
   hover altitude.

Pass rule: every leg keeps at least CLEARANCE_MIN_M of clearance from every
solid, the sight-line is unobstructed, and the target is inside the frustum.
CLEARANCE_MIN_M is a declared engineering parameter of this check, chosen
consistent with the catalogue family: these worlds' doorways are 0.8-1.2 m
wide for this vehicle, so the bar sits at 0.15 m — a leg that grazes any
surface fails, and the actual measured margins are printed for the reader.

Pure standard library. Writes a JSON receipt and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

CLEARANCE_MIN_M = 0.15
IRIS_HFOV_RAD = 1.047200  # Iris.proto camera fieldOfView (both eyes)
IRIS_IMAGE_W = 640
IRIS_IMAGE_H = 480
# The vehicle node: ``Iris {`` at the top level, its translation on the next line.
IRIS_RE = re.compile(r"\nIris\s*\{[^}]*?translation\s+([^\n]+)", re.DOTALL)


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split())


def box_aabb(translation: tuple[float, ...], size: tuple[float, ...]):
    cx, cy, cz = translation[:3]
    sx, sy, sz = size[:3]
    return (
        (cx - sx / 2, cy - sy / 2, cz - sz / 2),
        (cx + sx / 2, cy + sy / 2, cz + sz / 2),
    )


def iter_solid_blocks(text: str):
    """Every top-level ``Solid { ... }`` block, brace-balanced.

    A regex alone cannot see where a node ends: a Solid whose children hold no
    Box (the floor plane) would otherwise swallow the next Solid's Box. Blocks
    are therefore extracted by counting braces from each ``Solid {``.
    """
    for match in re.finditer(r"\bSolid\s*\{", text):
        start = match.end() - 1
        depth = 0
        for position in range(start, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    yield text[match.start() : position + 1]
                    break


def parse_solids(world_text: str) -> dict[str, tuple[tuple, tuple]]:
    """Named Solids that are axis-aligned boxes, as AABBs.

    A Solid without a geometry Box (the floor, a Plane) is deliberately not an
    obstacle for a flight leg: it is the surface the aircraft flies above.
    """
    solids = {}
    for block in iter_solid_blocks(world_text):
        name_match = re.search(r'name\s+"([^"]+)"', block)
        translation_match = re.search(r"translation\s+([^\n]+)", block)
        size_match = re.search(r"geometry\s+Box\s*\{\s*size\s+([^\n}]+)", block)
        if not (name_match and translation_match and size_match):
            continue
        solids[name_match.group(1)] = box_aabb(
            parse_floats(translation_match.group(1).strip()),
            parse_floats(size_match.group(1).strip()),
        )
    return solids


def distance_point_aabb(point, aabb) -> float:
    (lx, ly, lz), (hx, hy, hz) = aabb
    px, py, pz = point
    dx = max(lx - px, 0.0, px - hx)
    dy = max(ly - py, 0.0, py - hy)
    dz = max(lz - pz, 0.0, pz - hz)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def distance_segment_aabb(a, b, aabb) -> float:
    """Minimum distance from segment ab to the box.

    The distance from a point moving along a segment to a convex set is a
    convex function of the parameter, so a ternary search converges to the
    infimum; a flat zero region (the segment inside the box) is found as zero.
    """
    def distance(t: float) -> float:
        return distance_point_aabb(
            (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t),
            aabb,
        )

    lo, hi = 0.0, 1.0
    for _ in range(200):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if distance(m1) <= distance(m2):
            hi = m2
        else:
            lo = m1
    return min(distance(lo), distance(hi), distance((lo + hi) / 2))


def main(argv=None) -> int:
    scene = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=Path, default=scene / "world.wbt")
    parser.add_argument("--mission", type=Path, default=scene / "mission.yaml")
    parser.add_argument("--target-node", default="target_block")
    parser.add_argument(
        "--chain", nargs="+", type=float, metavar="X Y Z",
        help="declared waypoint chain in world ENU metres; the verification input, "
        "never stored in scene metadata",
    )
    parser.add_argument("--output", type=Path, required=True, help="receipt JSON path")
    args = parser.parse_args(argv)

    import yaml

    world_text = args.world.read_text(encoding="utf-8")
    solids = parse_solids(world_text)
    iris = IRIS_RE.search(world_text)
    if iris is None:
        print("FAIL: no Iris node in the world")
        return 1
    spawn = parse_floats(iris.group(1).strip())[:3]
    if len(args.chain) % 3 != 0 or len(args.chain) < 6:
        parser.error("--chain needs at least two waypoints as X Y Z triplets")
    waypoints = [
        tuple(args.chain[i : i + 3]) for i in range(0, len(args.chain), 3)
    ]
    chain = [spawn] + waypoints

    mission = yaml.safe_load(args.mission.read_text(encoding="utf-8"))
    hover_alt = float(mission["budgets"]["hover_altitude_m"])

    target = solids.get(args.target_node)
    if target is None:
        print(f"FAIL: target node {args.target_node!r} not found in the world")
        return 1
    target_center = tuple((target[0][i] + target[1][i]) / 2 for i in range(3))

    exclusions = {args.target_node}
    leg_reports = []
    ok = True
    for index in range(len(chain) - 1):
        a, b = chain[index], chain[index + 1]
        margins = {}
        for name, aabb in solids.items():
            if name in exclusions:
                continue
            margins[name] = round(distance_segment_aabb(a, b, aabb), 4)
        worst_name, worst = min(margins.items(), key=lambda item: item[1])
        leg_ok = worst > CLEARANCE_MIN_M
        ok &= leg_ok
        leg_reports.append(
            {
                "leg": index,
                "from": a,
                "to": b,
                "min_clearance_m": worst,
                "nearest_solid": worst_name,
                "pass": leg_ok,
                "margins_m": dict(sorted(margins.items(), key=lambda kv: kv[1])[:6]),
            }
        )

    vantage = chain[-1]
    sight = distance_segment_aabb(vantage, target_center, target)  # sanity, ~0
    sight_margins = {
        name: round(distance_segment_aabb(vantage, target_center, aabb), 4)
        for name, aabb in solids.items()
        if name not in exclusions
    }
    sight_worst_name, sight_worst = min(sight_margins.items(), key=lambda item: item[1])
    sight_clear = sight_worst > 0.0

    heading = (
        vantage[0] - chain[-2][0],
        vantage[1] - chain[-2][1],
    )
    bearing = (
        target_center[0] - vantage[0],
        target_center[1] - vantage[1],
    )
    heading_len = math.hypot(*heading) or 1.0
    bearing_len = math.hypot(*bearing) or 1.0
    cos_angle = (heading[0] * bearing[0] + heading[1] * bearing[1]) / (heading_len * bearing_len)
    horizontal_angle = math.acos(max(-1.0, min(1.0, cos_angle)))
    vfov = 2 * math.atan(math.tan(IRIS_HFOV_RAD / 2) * IRIS_IMAGE_H / IRIS_IMAGE_W)
    elevation = math.atan2(
        target_center[2] - vantage[2], bearing_len
    )
    in_frustum = (
        horizontal_angle <= IRIS_HFOV_RAD / 2 and abs(elevation) <= vfov / 2
    )
    altitudes_ok = all(0.0 <= point[2] <= hover_alt for point in chain)
    distance_m = math.sqrt(
        sum((target_center[i] - vantage[i]) ** 2 for i in range(3))
    )

    visible_ok = sight_clear and in_frustum
    ok = ok and visible_ok and altitudes_ok

    receipt = {
        "check": "first-indoor reachability and visibility",
        "world": str(args.world),
        "spawn_enu_m": spawn,
        "declared_chain_enu_m": chain,
        "target_node": args.target_node,
        "target_center_enu_m": target_center,
        "target_distance_from_vantage_m": round(distance_m, 3),
        "clearance_criterion_m": CLEARANCE_MIN_M,
        "camera": {
            "hfov_rad": IRIS_HFOV_RAD,
            "vfov_rad": round(vfov, 4),
            "horizontal_off_axis_rad": round(horizontal_angle, 4),
            "elevation_rad": round(elevation, 4),
        },
        "legs": leg_reports,
        "sight_line": {
            "from_vantage": vantage,
            "unobstructed": sight_clear,
            "nearest_solid": sight_worst_name,
            "nearest_clearance_m": sight_worst,
        },
        "altitudes_within_hover_budget": altitudes_ok,
        "hover_altitude_budget_m": hover_alt,
        "target_visible_from_vantage": visible_ok,
        "pass": ok,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for leg in leg_reports:
        state = "pass" if leg["pass"] else "FAIL"
        print(
            f"leg {leg['leg']}: {leg['from']} -> {leg['to']}  "
            f"min clearance {leg['min_clearance_m']:.3f} m to {leg['nearest_solid']}  [{state}]"
        )
    print(
        f"sight line: {sight_worst:.3f} m to {sight_worst_name}, "
        f"target {distance_m:.2f} m away, off-axis "
        f"{math.degrees(horizontal_angle):.1f} deg, elevation {math.degrees(elevation):.1f} deg "
        f"[{'pass' if visible_ok else 'FAIL'}]"
    )
    print(f"altitudes within hover budget {hover_alt} m: {altitudes_ok}")
    print(f"receipt: {args.output}")
    if not ok:
        print("FAIL: the declared chain does not reach and see the target within the criteria")
        return 1
    print("PASS: the target is reachable along the declared chain and visible from the vantage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
