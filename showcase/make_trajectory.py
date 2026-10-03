"""Figure: what the aircraft actually did, from the autopilot's own telemetry.

Source: work/runs/p05/<run>/platform/run-a/mavlink.jsonl
  LOCAL_POSITION_NED (x,y,z,vx,vy,vz,time_boot_ms,received_monotonic_ns)
  ATTITUDE           (roll,pitch,yaw,...,time_boot_ms,received_monotonic_ns)
Both are what the autopilot reported, not what the mission asked for.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/muadhsambul/embodied")
OUT = ROOT / "showcase"

def load(run, pkt):
    rows = []
    for line in (ROOT / "work/runs/p05" / run / "platform" / "run-a" / "mavlink.jsonl").open():
        try: d = json.loads(line)
        except Exception: continue
        if d.get("mavpackettype") == pkt: rows.append(d)
    return rows

def main(run="live-14"):
    pos = load(run, "LOCAL_POSITION_NED")
    att = load(run, "ATTITUDE")
    t0 = min(r["received_monotonic_ns"] for r in pos + att)
    tp = np.array([(r["received_monotonic_ns"] - t0) / 1e9 for r in pos])
    x = np.array([r["x"] for r in pos]); y = np.array([r["y"] for r in pos]); z = np.array([r["z"] for r in pos])
    ta = np.array([(r["received_monotonic_ns"] - t0) / 1e9 for r in att])
    roll = np.degrees([r["roll"] for r in att]); pitch = np.degrees([r["pitch"] for r in att])

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), dpi=130, sharex=True)
    ax[0].plot(tp, x, label="NED x", lw=1.4)
    ax[0].plot(tp, y, label="NED y", lw=1.4)
    ax[0].plot(tp, z, label="NED z (down positive)", lw=1.4)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_ylabel("metres"); ax[0].legend(loc="upper left", fontsize=9)
    ax[0].set_title(f"{run}: the position the autopilot reported (LOCAL_POSITION_NED, n={len(pos)})", fontsize=10.5)
    ax[0].grid(alpha=0.25)
    ax[1].plot(ta, roll, label="roll", lw=1.4)
    ax[1].plot(ta, pitch, label="pitch", lw=1.4)
    ax[1].axhline(0, color="0.6", lw=0.8); ax[1].axhline(90, color="r", ls=":", lw=1.0)
    ax[1].axhline(-90, color="r", ls=":", lw=1.0)
    ax[1].set_ylabel("degrees"); ax[1].set_xlabel("seconds from the first telemetry message")
    ax[1].legend(loc="upper left", fontsize=9); ax[1].grid(alpha=0.25)
    ax[1].set_title(f"{run}: attitude the autopilot reported (ATTITUDE, n={len(att)}); dotted lines at +/-90 deg", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / f"trajectory-{run}.png")
    print("wrote", OUT / f"trajectory-{run}.png")
    print(json.dumps({"run": run, "n_pos": len(pos), "n_att": len(att),
        "duration_s": round(float(max(tp.max(), ta.max())), 1),
        "pos_range_m": {"x": [round(float(x.min()),1), round(float(x.max()),1)],
                        "y": [round(float(y.min()),1), round(float(y.max()),1)],
                        "z": [round(float(z.min()),1), round(float(z.max()),1)]},
        "roll_deg": {"min": round(float(roll.min()),1), "max": round(float(roll.max()),1)},
        "pitch_deg": {"min": round(float(pitch.min()),1), "max": round(float(pitch.max()),1)}}, indent=1))

main("live-14")
