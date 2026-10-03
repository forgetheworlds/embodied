"""Figures: launch outcomes over attempts, and perception/map growth across runs.

Sources: every work/runs/p05/*/mission.json and receipt.json on disk.
Nothing is inferred; a run with no mission.json is simply absent.
"""
from __future__ import annotations
import glob, json, os, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/muadhsambul/embodied")
OUT = ROOT / "showcase"

def order_key(p):
    f = ROOT / "work/runs/p05" / Path(p).parent.name / "receipt.json"
    return os.path.getmtime(f) if f.exists() else os.path.getmtime(p)

rows = []
for p in glob.glob(str(ROOT / "work/runs/p05/*/mission.json")):
    d = json.load(open(p))
    run = Path(p).parent.name
    s = d.get("stream") or {}
    log = " ".join(str(x) for x in (d.get("log") or []))
    m = re.search(r"searchable_from_evidence': (\d+)", log) or re.search(r'"searchable_from_evidence": (\d+)', log)
    obs = s.get("pair_records_filed")
    rows.append(dict(run=run, flew=d.get("flew"), term=str(d.get("termination_reason")),
                     obs=obs, sfe=int(m.group(1)) if m else None,
                     mtime=order_key(p)))
rows.sort(key=lambda r: r["mtime"])

print("=== per-run (oldest first) ===")
for r in rows:
    print(f"  {r['run']:16} flew={str(r['flew']):5} term={r['term'][:22]:22} obs={str(r['obs']):6} searchable_from_evidence={r['sfe']}")

# ---- Figure: launch outcomes -------------------------------------------------
counts = {"flew": 0, "refused": 0, "no flight stage": 0}
for r in rows:
    if r["flew"] is True: counts["flew"] += 1
    elif r["flew"] is False: counts["refused"] += 1
    else: counts["no flight stage"] += 1

fix_idx = next((i for i, r in enumerate(rows) if r["run"] == "J19-launch-1"), None)
fig, ax = plt.subplots(figsize=(11, 3.4), dpi=130)
colors = {"flew": "#2b7a4b", "refused": "#b4322b", "no flight stage": "#8a8a8a"}
for i, r in enumerate(rows):
    if r["flew"] is True: c = colors["flew"]
    elif r["flew"] is False: c = colors["refused"]
    else: c = colors["no flight stage"]
    ax.bar(i, 1, color=c, width=0.85)
if fix_idx is not None:
    ax.axvline(fix_idx - 0.5, color="k", ls="--", lw=1.2)
    ax.text(fix_idx + 0.4, 0.55, "launch fix (J19)", fontsize=9, rotation=90, va="center")
ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_yticks([])
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["run"] for r in rows], rotation=90, fontsize=6)
ax.set_title(f"Outcome of every mission attempt on disk (n={len(rows)}): "
             f"{counts['flew']} flew, {counts['refused']} refused to arm, {counts['no flight stage']} stopped before a flight stage",
             fontsize=10)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "launch-outcomes.png"); print("\nwrote launch-outcomes.png", counts)

# ---- Figure: observations and searchable cells -------------------------------
sel = [r for r in rows if r["obs"] is not None]
fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), dpi=130, sharex=True)
xs = np.arange(len(sel))
ax[0].bar(xs, [r["obs"] for r in sel], color="#33556e")
ax[0].set_ylabel("perception frames filed")
ax[0].set_title("Perception frames filed per mission run (oldest to newest)", fontsize=10.5)
ax[0].grid(alpha=0.25, axis="y")
sfe = [np.nan if r["sfe"] is None else r["sfe"] for r in sel]
ax[1].bar(xs, sfe, color="#7a5ea8")
ax[1].set_ylabel("cells the map itself\npublished free")
ax[1].set_title("searchable_from_evidence at cold start: cells the map published free, "
                "excluding the aircraft's own exemption", fontsize=10.5)
ax[1].set_xticks(xs); ax[1].set_xticklabels([r["run"] for r in sel], rotation=90, fontsize=6.5)
ax[1].grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(OUT / "perception-and-map.png"); print("wrote perception-and-map.png")
