"""Figure: the aircraft's own view, the depth it recovered, and what it rejected.

Source: work/runs/p05/live-14/episode/payloads/obs-00002-{left,right}.ppm
Depth is computed by the project's own pipeline:
    embodied.perception.camera.compute_validated_depth
with the settings read from configs/first_indoor.yaml (calibration.matcher +
calibration.bounds). Nothing here changes a declared value.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path("/Users/muadhsambul/embodied")
sys.path.insert(0, str(ROOT / "src"))
from embodied.perception import camera as C  # noqa: E402

OUT = ROOT / "showcase"
RUN = "live-14"
PAY = ROOT / "work/runs/p05" / RUN / "episode" / "payloads"

cfg = yaml.safe_load((ROOT / "configs/first_indoor.yaml").read_text())
cal = cfg["calibration"]
settings = {**cal["matcher"],
            "border_px": cal["bounds"]["border_px"],
            "depth_range_m": cal["bounds"]["depth_range_m"]}

left = C.read_ppm(PAY / "obs-00002-left.ppm")
right = C.read_ppm(PAY / "obs-00002-right.ppm")
product = C.compute_validated_depth(left, right, C.build_calibration(), settings,
                                    pair_id="obs-00002")

names = {C.REASON_VALID: "valid", C.REASON_NO_RETURN: "no_return",
         C.REASON_LR_MISMATCH: "lr_mismatch", C.REASON_DEPTH_RANGE: "depth_range",
         C.REASON_BORDER: "border"}
counts = {names[r]: int((product.reasons == r).sum()) for r in names}
total = int(product.reasons.size)
valid_px = product.valid
depth = product.depth_m.astype(float)

fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), dpi=130)

axes[0, 0].imshow(left); axes[0, 0].set_title(f"Left camera, {RUN} obs-00002 (640x480)", fontsize=10)
axes[0, 1].imshow(right); axes[0, 1].set_title("Right camera, same instant (stereo pair)", fontsize=10)

finite = np.isfinite(depth) & valid_px
im = axes[1, 0].imshow(np.where(finite, depth, np.nan), cmap="viridis", vmin=0.5, vmax=6.0)
axes[1, 0].set_title("Depth from the project's own SGBM pipeline, valid pixels only", fontsize=10)
cb = fig.colorbar(im, ax=axes[1, 0], fraction=0.046); cb.set_label("metres", fontsize=9)

order = [C.REASON_VALID, C.REASON_NO_RETURN, C.REASON_LR_MISMATCH, C.REASON_DEPTH_RANGE, C.REASON_BORDER]
lut = np.zeros((5, 3)); lut[0] = (0.95, 0.95, 0.95); lut[1] = (0.85, 0.16, 0.16)
lut[2] = (0.95, 0.60, 0.12); lut[3] = (0.15, 0.42, 0.78); lut[4] = (0.55, 0.55, 0.55)
idx = np.zeros(product.reasons.shape, dtype=int)
for i, r in enumerate(order): idx[product.reasons == r] = i
axes[1, 1].imshow(idx, cmap=ListedColormap(lut), norm=BoundaryNorm(np.arange(-0.5, 5.5, 1), 5))
share = "  ".join(f"{names[r]} {counts[names[r]]/total*100:.1f}%" for r in order)
axes[1, 1].set_title(f"Which pixels were kept, and why the rest were dropped\n{share}", fontsize=9.5)

for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Retained stereo pair from {RUN}, and what the depth pipeline made of it "
             f"({total-valid_px.sum()} of {total} pixels rejected)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "aircraft-view.png")
print("wrote", OUT / "aircraft-view.png")
print(json.dumps({"counts": counts, "total": total,
                  "valid_fraction": round(int(valid_px.sum())/total, 4),
                  "depth_m": {"min": round(float(depth[finite].min()),2),
                              "p50": round(float(np.median(depth[finite])),2),
                              "max": round(float(depth[finite].max()),2)}}, indent=1))
