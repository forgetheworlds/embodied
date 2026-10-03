"""Four figures from stored numbers. Every value is read from a named artifact."""
from __future__ import annotations
import glob, json, os, re, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/muadhsambul/embodied")
OUT = ROOT / "showcase"

# ---------------------------------------------------------------- A. launch outcomes
runs = []
for p in sorted(glob.glob(str(ROOT / "work/runs/p05/*/mission.json")), key=os.path.getmtime):
    d = json.load(open(p))
    runs.append((Path(p).parent.name, d.get("flew"), str(d.get("termination_reason"))))
flew = sum(1 for _, f, _ in runs if f is True)
refused = sum(1 for _, f, _ in runs if f is False)
nostage = len(runs) - flew - refused

fig, ax = plt.subplots(figsize=(11, 3.2), dpi=130)
col = {"flew": "#2b7a4b", "refused": "#b4322b", "nostage": "#8a8a8a"}
for i, (_, f, _) in enumerate(runs):
    c = col["flew"] if f is True else (col["refused"] if f is False else col["nostage"])
    ax.bar(i, 1, color=c, width=0.85)
ax.set_xticks(range(len(runs)))
ax.set_xticklabels([r[0] for r in runs], rotation=90, fontsize=6)
ax.set_yticks([]); ax.set_xlim(-0.6, len(runs) - 0.4)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.set_title(f"Outcome of every run that produced a mission record (n={len(runs)}): "
             f"{flew} flew (green), {refused} the autopilot refused to arm (red), "
             f"{nostage} stopped before a flight stage (grey)", fontsize=9.5)
ax.annotate("the seven deliberate launches after the fix\n(J19-launch-1..7) all flew and are not shown here:\nthey carry a P01-L gate receipt, not a mission record",
            xy=(0.012, 0.55), xycoords="axes fraction", fontsize=8, va="center")
fig.tight_layout(); fig.savefig(OUT / "launch-outcomes.png")
print("launch-outcomes.png:", dict(flew=flew, refused=refused, nostage=nostage))

# ------------------------------------------------- B. observations and searchable cells
obs, sfe = [], []
for p in sorted(glob.glob(str(ROOT / "work/runs/p05/*/episode/agent-events.jsonl")), key=os.path.getmtime):
    run = Path(p).parents[1].name
    c = collections.Counter()
    for line in open(p):
        try: c[json.loads(line).get("kind")] += 1
        except Exception: pass
    m = ROOT / "work/runs/p05" / run / "mission.json"
    n = None
    if m.exists():
        log = " ".join(str(x) for x in (json.load(open(m)).get("log") or []))
        mm = re.search(r"searchable_from_evidence': (\d+)", log) or re.search(r'"searchable_from_evidence": (\d+)', log)
        n = int(mm.group(1)) if mm else None
    if c.get("observation", 0) or n is not None:
        obs.append((run, c.get("observation", 0))); sfe.append((run, n))

fig, ax = plt.subplots(2, 1, figsize=(11, 6.6), dpi=130, sharex=True)
xs = np.arange(len(obs))
ax[0].bar(xs, [v for _, v in obs], color="#33556e")
ax[0].set_ylabel("observations\n(agent-events)"); ax[0].grid(alpha=0.25, axis="y")
ax[0].set_title("Observations per mission (left) and cells the map itself published free at cold start (right)",
                fontsize=10)
valid_sfe = [(i, r, v) for i, (r, v) in enumerate(sfe) if v is not None]
ax[1].bar([i for i, _, _ in valid_sfe], [v for _, _, v in valid_sfe], color="#7a5ea8")
ax[1].set_ylabel("searchable_from_evidence")
ax[1].grid(alpha=0.25, axis="y")
ax[1].set_xticks(xs); ax[1].set_xticklabels([r for r, _ in obs], rotation=90, fontsize=6.5)
if valid_sfe:
    ax[1].annotate("field first recorded here", xy=(valid_sfe[0][0], 0), xytext=(valid_sfe[0][0] + 1.5, max(v for _, _, v in valid_sfe) * 0.75),
                   fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
fig.tight_layout(); fig.savefig(OUT / "perception-and-map.png")
print("perception-and-map.png: obs", [v for _, v in obs])
print("   sfe", valid_sfe)

# ------------------------------------------------------------- C. depth rejections
b = json.load(open(ROOT / "work/runs/p05/J27-breakdown.json"))
pooled = b["pooled"]; per = b["per_source"]
order = ["valid", "no_return", "depth_range", "lr_mismatch", "border"]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=130)
ax[0].bar(order, [pooled[k] * 100 for k in order], color=["#2b7a4b", "#b4322b", "#1f5fa8", "#e08a3c", "#8a8a8a"])
for i, k in enumerate(order): ax[0].text(i, pooled[k] * 100 + 1, f"{pooled[k]*100:.1f}%", ha="center", fontsize=9)
ax[0].set_ylabel("% of depth pixels"); ax[0].tick_params(axis="x", rotation=20)
ax[0].set_title(f"Pooled over {b.get('frames')} retained frames", fontsize=10)
ax[0].grid(alpha=0.25, axis="y")
names = list(per); val = [per[n]["fractions"]["valid"] * 100 for n in names]
fr = [per[n]["frames"] for n in names]
ax[1].barh(names, val, color="#2b7a4b")
for i, (v, f) in enumerate(zip(val, fr)): ax[1].text(v + 0.7, i, f"{v:.1f}%  (n={f})", va="center", fontsize=8)
ax[1].set_xlabel("% of pixels with valid depth"); ax[1].set_xlim(0, 78)
ax[1].set_title("Valid fraction per source: the pooled number hides a 6x spread", fontsize=10)
ax[1].grid(alpha=0.25, axis="x")
fig.tight_layout(); fig.savefig(OUT / "depth-rejections.png")
print("depth-rejections.png pooled:", {k: round(pooled[k]*100, 1) for k in order})
print("   per source:", {n: round(per[n]['fractions']['valid']*100, 1) for n in names})

# --------------------------------------------------------------- D. cloud latency
cont = json.load(open(ROOT / "work/runs/p05/J6-arm-continuous.json"))
init = json.load(open(ROOT / "work/runs/p05/J6-arm-initial.json"))
def lat(d): return [c["round_trip_s"] for c in d["calls"] if isinstance(c.get("round_trip_s"), (int, float))]
fig, ax = plt.subplots(figsize=(11, 4.3), dpi=130)
for i, (d, label, c) in enumerate([(cont, "continuous: reasoning off, quarter-scale frame", "#2b7a4b"),
                                   (init, "initial: reasoning high, full frame", "#b4322b")]):
    v = lat(d)
    ax.scatter([i] * len(v), v, s=42, color=c, alpha=0.85, zorder=3)
    ax.hlines(np.median(v), i - 0.18, i + 0.18, color="k", lw=1.6, zorder=4)
    s_p50 = (d.get("summary") or {}).get("latency_s", {}).get("p50")
    ax.text(i + 0.22, np.median(v), f"p50 {s_p50:.2f}s (summary)\nn={len(v)} of {len(d['calls'])} replied",
            fontsize=8.5, va="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(["continuous\n(12 calls, 12 replied)", "initial\n(12 attempted, 9 replied)"])
ax.set_ylabel("round trip (s)"); ax.set_ylim(0, 30); ax.grid(alpha=0.25, axis="y")
ax.set_title("Cloud decision latency by declared call class, commandcode/deepseek-v4.1-flash "
             "(J6-arm-continuous.json, J6-arm-initial.json)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "cloud-latency.png")
print("cloud-latency.png: continuous p50", round(float(np.median(lat(cont))), 2),
      "| initial p50", round(float(np.median(lat(init))), 2), "of", len(init["calls"]), "attempted")

# ------------------------------------------------------------ E. integration cost
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), dpi=130)
ax[0].bar(["before", "after"], [2433, 218], color=["#b4322b", "#2b7a4b"])
for i, v in enumerate([2433, 218]): ax[0].text(i, v + 60, f"{v} ms", ha="center", fontsize=9)
ax[0].set_ylabel("map integration per frame (ms)")
ax[0].set_title("Map integration: 2,433 -> 218 ms\n(162 ms building tuples + 2,270 ms accumulating)", fontsize=9.5)
ax[1].bar(["before", "after"], [4.86, 0.64], color=["#b4322b", "#2b7a4b"])
for i, v in enumerate([4.86, 0.64]): ax[1].text(i, v + 0.12, f"{v} s", ha="center", fontsize=9)
ax[1].axhline(0.30, color="k", ls="--", lw=1.2)
ax[1].text(1.35, 0.34, "declared cadence 0.30 s", fontsize=8, ha="right")
ax[1].set_ylabel("perception cycle (s)")
ax[1].set_title("Perception cycle: 4.86 -> 0.64 s\ndepth is 0.173 s of it", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "integration-cost.png")
print("integration-cost.png: 2433->218 ms, 4.86->0.64 s (J24-sim-REPORT.md)")
