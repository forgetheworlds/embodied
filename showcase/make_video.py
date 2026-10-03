"""Video: what the aircraft saw and what the depth pipeline measured, frame by frame.

Source: work/runs/p05/J30-move-1/episode/payloads/ (22 retained stereo pairs).
The aircraft did not translate in this run, so the view barely changes; the caption
says so. Depth is computed by the project's own pipeline, as in make_aircraft_view.py.
"""
from __future__ import annotations
import glob, subprocess, sys, tempfile, json
from pathlib import Path
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image

ROOT = Path("/Users/muadhsambul/embodied")
sys.path.insert(0, str(ROOT / "src"))
from embodied.perception import camera as C  # noqa: E402

RUN = "J30-move-1"
PAY = ROOT / "work/runs/p05" / RUN / "episode" / "payloads"
cfg = yaml.safe_load((ROOT / "configs/first_indoor.yaml").read_text())
cal = cfg["calibration"]
settings = {**cal["matcher"], "border_px": cal["bounds"]["border_px"],
            "depth_range_m": cal["bounds"]["depth_range_m"]}
calibration = C.build_calibration()

order = [C.REASON_VALID, C.REASON_NO_RETURN, C.REASON_LR_MISMATCH, C.REASON_DEPTH_RANGE, C.REASON_BORDER]
lut = np.zeros((5, 3)); lut[0] = (0.95, 0.95, 0.95); lut[1] = (0.85, 0.16, 0.16)
lut[2] = (0.95, 0.60, 0.12); lut[3] = (0.15, 0.42, 0.78); lut[4] = (0.55, 0.55, 0.55)

lefts = sorted(PAY.glob("obs-*-left.ppm"))
tmp = Path(tempfile.mkdtemp(prefix="j36vid-"))
valid_frac = []
for i, lp in enumerate(lefts):
    rp = Path(str(lp).replace("-left.ppm", "-right.ppm"))
    left = C.read_ppm(lp); right = C.read_ppm(rp)
    prod = C.compute_validated_depth(left, right, calibration, settings, pair_id=lp.stem)
    depth = prod.depth_m.astype(float); fin = np.isfinite(depth) & prod.valid
    valid_frac.append(float(prod.valid.mean()))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9), dpi=110)
    ax[0].imshow(left); ax[0].set_title("left camera", fontsize=9)
    im = ax[1].imshow(np.where(fin, depth, np.nan), cmap="viridis", vmin=0.5, vmax=6.0)
    ax[1].set_title("depth, valid pixels only (m)", fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=0.04)
    idx = np.zeros(prod.reasons.shape, dtype=int)
    for k, r in enumerate(order): idx[prod.reasons == r] = k
    ax[2].imshow(idx, cmap=ListedColormap(lut), norm=BoundaryNorm(np.arange(-0.5, 5.5, 1), 5))
    ax[2].set_title(f"kept vs rejected  (valid {prod.valid.mean()*100:.0f}%)", fontsize=9)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{RUN}, retained pair {i+1} of {len(lefts)}   |   "
                 "the aircraft hovered at one position for this run, so the view barely changes",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(tmp / f"f{i:03d}.png"); plt.close(fig)

mp4 = ROOT / "showcase" / f"aircraft-view-{RUN}.mp4"
subprocess.run(["ffmpeg", "-y", "-framerate", "5", "-i", str(tmp / "f%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                str(mp4)], check=True, capture_output=True)
print("wrote", mp4, mp4.stat().st_size // 1024, "kB")
print(json.dumps({"frames": len(lefts), "fps": 5,
                  "duration_s_video": round(len(lefts)/5, 1),
                  "valid_fraction_min": round(min(valid_frac), 3),
                  "valid_fraction_max": round(max(valid_frac), 3),
                  "valid_fraction_mean": round(float(np.mean(valid_frac)), 3)}, indent=1))
