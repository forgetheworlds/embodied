"""Figure: plan view of the mission scene, from scenarios/first_indoor/world.wbt.

Every rectangle is a named Solid in the world file with the translation and size
declared there. Room outlines, the two 1.0 m doorways, the three floor obstacles,
the near-red decoy and the red target block all come from that file. The spawn is
from scenarios/first_indoor/mission.yaml.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/Users/muadhsambul/embodied")
OUT = ROOT / "showcase"

# (name, cx, cy, sx, sy) taken from the world file's translation and size fields.
WALLS = [
    ("w_vest_west",   -2.05, 0.0,   0.10, 2.00),
    ("w_vest_north",  -1.00, 1.05,  2.00, 0.10),
    ("w_vest_south",  -1.00, -1.05, 2.00, 0.10),
    ("w_shared_x0_a",  0.00, 1.50,  0.10, 2.00),
    ("w_shared_x0_b",  0.00, -1.50, 0.10, 2.00),
    ("w_r1_north",     2.50, 2.55,  5.00, 0.10),
    ("w_r1_south",     2.50, -2.55, 5.00, 0.10),
    ("w_shared_x5_a",  5.00, -0.90, 0.10, 3.20),
    ("w_shared_x5_b",  5.00, 2.10,  0.10, 0.80),
    ("w_r2_north",     7.50, 2.55,  5.00, 0.10),
    ("w_r2_south",     7.50, -2.55, 5.00, 0.10),
    ("w_r2_east",     10.05, 0.00,  0.10, 5.00),
]
OBSTACLES = [(1.0, -1.9), (3.9, 1.9), (6.8, -1.9)]
DECOY = (2.2, 1.6, 0.4, 0.4)
TARGET = (8.8, -1.6, 0.6, 0.6)
SPAWN = (-1.0, 0.0)

fig, ax = plt.subplots(figsize=(12, 5.0), dpi=130)
for name, cx, cy, sx, sy in WALLS:
    ax.add_patch(Rectangle((cx - sx/2, cy - sy/2), sx, sy, color="#3a3a3a"))
for ox, oy in OBSTACLES:
    ax.add_patch(Rectangle((ox - 0.25, oy - 0.25), 0.5, 0.5, color="#7a7a7a"))
ax.add_patch(Rectangle((DECOY[0]-DECOY[2]/2, DECOY[1]-DECOY[3]/2), DECOY[2], DECOY[3],
                       color="#e08a3c", ec="k", lw=0.6))
ax.add_patch(Rectangle((TARGET[0]-TARGET[2]/2, TARGET[1]-TARGET[3]/2), TARGET[2], TARGET[3],
                       color="#c02525", ec="k", lw=0.8))
ax.plot(*SPAWN, marker="^", ms=13, color="#1f5fa8", mec="k", zorder=5)
ax.annotate("spawn (-1.0, 0.0)", SPAWN, textcoords="offset points", xytext=(-6, -20), fontsize=9)

# the goal the mission actually asked for, from live-14's refused-publications.jsonl
ax.plot(-0.95, -0.05, marker="o", ms=9, mfc="none", mec="#1f5fa8", ls="none", mew=1.6, zorder=5)
ax.annotate("commanded goal (-0.95, -0.05)\n0.07 m from the spawn",
            (-0.95, -0.05), textcoords="offset points", xytext=(12, 26), fontsize=9, color="#1f5fa8")

ax.annotate("vestibule", (-1.0, -2.25), ha="center", fontsize=9, color="#444")
ax.annotate("room 1", (2.5, -2.25), ha="center", fontsize=9, color="#444")
ax.annotate("room 2", (7.5, -2.25), ha="center", fontsize=9, color="#444")
ax.annotate("doorway 1.0 m\n(centred y=0)", (0, 2.0), ha="center", fontsize=8, color="#444")
ax.annotate("doorway 1.0 m\n(centred y=1.2)", (5, 2.32), ha="center", fontsize=8, color="#444")
ax.annotate("red target block\non a plinth", TARGET[:2], textcoords="offset points",
            xytext=(6, 22), fontsize=9, color="#c02525")
ax.annotate("near-red decoy", DECOY[:2], textcoords="offset points", xytext=(4, 14), fontsize=9, color="#a06020")
ax.annotate("0.5 m obstacles", (3.9, 1.9), textcoords="offset points", xytext=(-4, -22), fontsize=9, color="#555")

ax.set_xlim(-2.6, 10.6); ax.set_ylim(-3.1, 3.4); ax.set_aspect("equal")
ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
ax.set_title("scenarios/first_indoor/world.wbt, plan view: rooms, doorways, obstacles, decoy and target;\n"
             "the spawn from mission.yaml, and the goal live-14 actually commanded (refused)", fontsize=10.5)
ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(OUT / "scene-plan.png")
print("wrote", OUT / "scene-plan.png")
print(json.dumps({"walls": len(WALLS), "obstacles": len(OBSTACLES),
                  "target_xy": TARGET[:2], "decoy_xy": DECOY[:2], "spawn": SPAWN,
                  "commanded_goal": [-0.95, -0.05]}, indent=1))
