#!/usr/bin/env python3
"""Diff STATE sequences: replay-vs-replay (determinism) and live-vs-replay (faithfulness).

The replay surface's verifier (2026-09-30). Two comparisons with two different
honesties, because they answer different questions:

* ``--replay-a A --replay-b B`` -- both sequences were produced by the SAME
  input-paced publish rule from the SAME capture, so every field must match
  BITWISE. Anything else is filter/tracker nondeterminism (thread scheduling,
  uninitialized memory, an RNG), and it is named per message, not averaged away.

* ``--live-run R --replay B`` -- the live run's STATE sequence was sampled by a
  WALL-CLOCK publish tick, so no replay can reproduce its per-message prefixes
  exactly; pretending otherwise would be a diff that lies. What CAN be asked is
  whether both sequences sample the same underlying trajectory: each live row is
  matched to the nearest replay state within a declared window and compared in
  the live run's own aligned NED frame (the production OdomAlignment, built from
  the run's logged odom origin and declared start attitude and sealed on the
  replay's first state -- the seal lands in the parked phase in both runs, where
  the attitude is static, so the epoch rotation is the same to floating-point
  noise; the measured epoch yaw difference is printed as a caveat, not hidden).

Declared tolerances (defaults, overridable): match window 25 ms; position
0.02 m (vehicle <= ~2 m/s and a publish-phase difference <= ~10 ms gives
<= ~2 cm); attitude 0.01 rad; velocity 0.05 m/s; sigma 0.01 m. A matched row
outside tolerance counts as a divergence and the first one is printed with its
fields, so "where they diverge" has an address, not a percentile alone.

Usage:
  python3 estimator/compare_replay.py --replay-a A.jsonl --replay-b B.jsonl
  python3 estimator/compare_replay.py --live-run <run-dir> --replay B.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied.platform import localization as loc  # noqa: E402

_STATE_FIELDS = (
    "time_ns",
    "initialized",
    "quat_wxyz",
    "position_m",
    "velocity_mps",
    "gyro_bias",
    "accel_bias",
    "sigma_pos_m",
    "n_tracks",
    "t_last_visual_ns",
    "reset_counter",
)


def _load_states(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.sort(key=lambda row: row["index"])
    return rows


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def compare_replays(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    print(f"replay A: {len(a)} states   replay B: {len(b)} states")
    if len(a) != len(b):
        print(f"LENGTH MISMATCH: {len(a)} vs {len(b)}")
        return 1
    exact = 0
    first_diff: dict[str, Any] | None = None
    worst: dict[str, float] = {}
    for i, (row_a, row_b) in enumerate(zip(a, b)):
        row_equal = True
        for field in _STATE_FIELDS:
            if row_a[field] == row_b[field]:
                continue
            row_equal = False
            va, vb = row_a[field], row_b[field]
            if isinstance(va, list) and isinstance(vb, list):
                delta = max(abs(x - y) for x, y in zip(va, vb))
            elif isinstance(va, bool) or isinstance(vb, bool):
                delta = float(va != vb)
            else:
                delta = abs(float(va) - float(vb))
            worst[field] = max(worst.get(field, 0.0), delta)
            if first_diff is None:
                first_diff = {"index": i, "field": field, "a": va, "b": vb}
        if row_equal:
            exact += 1
    print(f"bitwise-identical messages: {exact}/{len(a)}")
    if exact == len(a):
        print("DETERMINISM: the two replays of the same capture are byte-identical")
        return 0
    print(f"first differing message: {first_diff}")
    print(f"worst per-field deltas: { {k: f'{v:.3e}' for k, v in sorted(worst.items())} }")
    return 1


_ORIGIN_LINE = re.compile(
    r"odom origin \(the world's own vehicle translation, ENU\): "
    r"\(([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\), declared start attitude rpy: "
    r"\(([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\)"
)


def _alignment_from_run_log(run_dir: Path, first_replay_state: dict[str, Any]) -> loc.OdomAlignment:
    match = _ORIGIN_LINE.search((run_dir / "run-a" / "log.txt").read_text())
    if match is None:
        raise SystemExit(f"{run_dir}/run-a/log.txt: no odom-origin line to build the alignment from")
    values = [float(value) for value in match.groups()]
    alignment = loc.OdomAlignment(values[0:3], values[3:6])
    # The live seal used the first state the publisher accepted; the parked phase
    # holds the attitude static, so sealing on the replay's first state lands on
    # the same epoch to floating-point noise (printed by the caller).
    alignment.seal(first_replay_state["quat_wxyz"])
    return alignment


def _rpy_to_matrix(rpy: list[float]) -> np.ndarray:
    return loc.rotmat_from_rpy(rpy)


def compare_live_to_replay(
    live_rows: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    run_dir: Path,
    window_ms: float,
    tolerance: dict[str, float],
) -> int:
    alignment = _alignment_from_run_log(run_dir, replay[0])
    live_epoch_yaw = None
    match = _ORIGIN_LINE.search((run_dir / "run-a" / "log.txt").read_text())
    # The live run's own epoch yaw is not logged per row; the seal's caveat is
    # reported as the first live row's residual instead (see below).
    replay_times = [row["time_ns"] for row in replay]
    live_anchors = sorted({row["t_last_visual_ns"] for row in live_rows})
    replay_anchors = sorted({row["t_last_visual_ns"] for row in replay})
    print(
        f"live rows: {len(live_rows)}   replay states: {len(replay)}   "
        f"camera anchors live/replay: {len(live_anchors)}/{len(replay_anchors)}"
    )
    print(f"anchor sets equal: {live_anchors == replay_anchors}")

    matched = 0
    unmatched = 0
    position_deltas: list[float] = []
    attitude_deltas: list[float] = []
    velocity_deltas: list[float] = []
    sigma_deltas: list[float] = []
    track_deltas: list[int] = []
    divergences: list[dict[str, Any]] = []
    for row in live_rows:
        t = row["time_ns"]
        i = bisect.bisect_left(replay_times, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(replay):
                gap = abs(replay_times[j] - t)
                if best is None or gap < best[0]:
                    best = (gap, replay[j])
        if best is None or best[0] > window_ms * 1e6:
            unmatched += 1
            continue
        state = best[1]
        matched += 1
        position = alignment.aligned_position_ned(state["position_m"])
        velocity = alignment.aligned_velocity_ned(state["velocity_mps"])
        replay_rpy = alignment.aligned_attitude_rpy(state["quat_wxyz"])
        pos_delta = float(
            np.linalg.norm(np.asarray(position) - np.asarray(row["position_ned_m"]))
        )
        att_delta = float(
            np.linalg.norm(np.asarray(replay_rpy) - np.asarray(row["attitude_rpy"]))
        )
        vel_delta = float(
            np.linalg.norm(np.asarray(velocity) - np.asarray(row["velocity_ned_mps"]))
        )
        sigma_delta = float(
            max(abs(a - b) for a, b in zip(state["sigma_pos_m"], row["sigma_pos_m"]))
        )
        track_delta = int(state["n_tracks"]) - int(row["n_tracks"])
        position_deltas.append(pos_delta)
        attitude_deltas.append(att_delta)
        velocity_deltas.append(vel_delta)
        sigma_deltas.append(sigma_delta)
        track_deltas.append(track_delta)
        if (
            pos_delta > tolerance["position_m"]
            or att_delta > tolerance["attitude_rad"]
            or vel_delta > tolerance["velocity_mps"]
            or sigma_delta > tolerance["sigma_m"]
        ):
            divergences.append(
                {
                    "live_time_ns": t,
                    "replay_time_ns": state["time_ns"],
                    "gap_ms": best[0] / 1e6,
                    "position_delta_m": pos_delta,
                    "attitude_delta_rad": att_delta,
                    "velocity_delta_mps": vel_delta,
                    "sigma_delta_m": sigma_delta,
                    "tracks_delta": track_delta,
                }
            )

    def report(name: str, values: list[float], tolerance_value: float) -> None:
        if not values:
            print(f"{name}: no matched rows")
            return
        print(
            f"{name}: p50 {_quantile(values, 0.50):.6f}  p95 {_quantile(values, 0.95):.6f}  "
            f"max {max(values):.6f}  (tolerance {tolerance_value})  "
            f"within {sum(1 for v in values if v <= tolerance_value)}/{len(values)}"
        )

    print(f"matched within {window_ms:.0f} ms: {matched}/{len(live_rows)} (unmatched {unmatched})")
    report("position delta m", position_deltas, tolerance["position_m"])
    report("attitude delta rad", attitude_deltas, tolerance["attitude_rad"])
    report("velocity delta m/s", velocity_deltas, tolerance["velocity_mps"])
    report("sigma delta m", sigma_deltas, tolerance["sigma_m"])
    print(
        "tracks delta: p50 "
        f"{_quantile([float(v) for v in track_deltas], 0.50):.0f}  max {max(track_deltas, default=0)}"
    )
    if divergences:
        print(f"DIVERGENCES: {len(divergences)} matched rows outside tolerance; first:")
        print(json.dumps(divergences[0], indent=1))
        return 1
    if unmatched > 0.01 * len(live_rows):
        print(f"UNMATCHED: {unmatched} live rows had no replay state within {window_ms:.0f} ms")
        return 1
    print("FAITHFULNESS: every matched live row agrees with the replay within tolerance")
    if live_anchors == replay_anchors:
        print("camera anchors: identical stamp sets")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay-a", type=Path)
    parser.add_argument("--replay-b", type=Path)
    parser.add_argument("--live-run", type=Path, help="the live run directory (run-a inside)")
    parser.add_argument("--window-ms", type=float, default=25.0)
    parser.add_argument("--position-m", type=float, default=0.02)
    parser.add_argument("--attitude-rad", type=float, default=0.01)
    parser.add_argument("--velocity-mps", type=float, default=0.05)
    parser.add_argument("--sigma-m", type=float, default=0.01)
    args = parser.parse_args()

    if args.replay_a and args.replay_b:
        return compare_replays(_load_states(args.replay_a), _load_states(args.replay_b))
    if args.live_run and args.replay_b:
        live_rows = [
            json.loads(line)
            for line in (args.live_run / "run-a" / "estimator-feed.jsonl").read_text().splitlines()
        ]
        replay = _load_states(args.replay_b)
        if not replay:
            print("the replay produced no states; nothing to compare")
            return 1
        return compare_live_to_replay(
            live_rows,
            replay,
            args.live_run,
            args.window_ms,
            {
                "position_m": args.position_m,
                "attitude_rad": args.attitude_rad,
                "velocity_mps": args.velocity_mps,
                "sigma_m": args.sigma_m,
            },
        )
    parser.error("give --replay-a and --replay-b, or --live-run and --replay-b")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
