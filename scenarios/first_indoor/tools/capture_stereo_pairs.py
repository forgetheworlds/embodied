#!/usr/bin/env python3
"""Capture stereo pairs from the scene at declared poses, with no flight.

Derives a capture world from ``world.wbt`` at run time -- rewriting only the
airframe's controller block -- and runs it headlessly. Deriving rather than
committing a second world is deliberate: a committed copy drifts from the scene
the mission actually flies, and the whole point of this instrument is that the
before and after frames come from the same geometry.

Usage, from the repository root:

    PYTHONPATH=src python3 scenarios/first_indoor/tools/capture_stereo_pairs.py \\
        --out-dir work/runs/p05/scenetexture/before \\
        --poses scenarios/first_indoor/tools/capture_poses.json

Plain standard library. The Webots argv is the validator's, so the capture runs
the same way a headless load does -- rendering ON, because Camera devices render
on their own pipeline and that is what is being measured.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEBOTS = Path("work/Webots.app/Contents/MacOS/webots")
CONTROLLER_NAME = "capture_poses_controller"
CAPTURE_WORLD_NAME = "world.capture.wbt"
DONE_MARKER = "CAPTURE_DONE"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--poses", type=Path,
                        default=Path("scenarios/first_indoor/tools/capture_poses.json"))
    parser.add_argument("--world", type=Path,
                        default=Path("scenarios/first_indoor/world.wbt"))
    parser.add_argument("--webots", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--settle-steps", type=int, default=30)
    return parser.parse_args(argv)


def derive_world(world_path: Path, out_dir: Path, poses_path: Path, settle_steps: int) -> Path:
    """Write the capture world beside the scene, with only the controller changed.

    The capture world must live in the scene's own directory: the scene's
    EXTERNPROTO references are relative to it, and a generated world in a
    temporary directory would not resolve them. The controller is resolved from
    the scene's own ``controllers/`` directory, which Webots checks first.
    """
    text = world_path.read_text(encoding="utf-8")
    marker = "\nIris {"
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"no Iris block found in {world_path}")
    head = text[: start + 1]
    block = text[start + 1 :].rstrip()
    if not block.endswith("}"):
        raise SystemExit(f"the Iris block in {world_path} does not close at the end")
    if "Iris {" not in block or block.count("\nIris {") > 0:
        raise SystemExit("more than one Iris block; refusing to guess which to rewrite")

    translation = re.search(r"^\s*translation\s+([^\n]+)$", block, re.M)
    custom_data = re.search(r"^\s*customData\s+(\"[^\n]*\")$", block, re.M)
    translation_line = (translation.group(1).strip() if translation else "-1 0 0.09")
    custom_line = (f"  customData {custom_data.group(1)}\n" if custom_data else "")

    rewritten = (
        "Iris {\n"
        f"  translation {translation_line}\n"
        f"  controller \"{CONTROLLER_NAME}\"\n"
        "  controllerArgs [\n"
        f"    \"--out-dir\" \"{out_dir}\"\n"
        f"    \"--poses\" \"{poses_path}\"\n"
        f"    \"--settle-steps\" \"{settle_steps}\"\n"
        "  ]\n"
        "  supervisor TRUE\n"
        f"{custom_line}"
        "}\n"
    )
    capture_world = world_path.parent / CAPTURE_WORLD_NAME
    capture_world.write_text(head + rewritten, encoding="utf-8")

    # The geometry must be untouched: the capture world differs from the scene
    # only inside that block. Verify rather than assert it in prose.
    # The geometry must be untouched, and that is checkable rather than a claim
    # in prose: every node name and every solid in the capture world must be the
    # scene's, and the text before the airframe block must be byte-identical.
    capture_text = capture_world.read_text(encoding="utf-8")
    if not capture_text.startswith(head):
        raise SystemExit("capture world changed the scene before the airframe block")
    scene_names = re.findall(r'^\s*name\s+"([^"]+)"', text, re.M)
    capture_names = re.findall(r'^\s*name\s+"([^"]+)"', capture_text, re.M)
    if scene_names != capture_names:
        raise SystemExit(
            f"capture world changed the scene's nodes: {set(scene_names) ^ set(capture_names)}"
        )
    if text.count("Solid {") != capture_text.count("Solid {"):
        raise SystemExit("capture world changed the scene's solids")
    return capture_world
def controller_environment() -> dict[str, str]:
    source_root = REPO_ROOT / "src"
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(source_root), existing) if p)
    env["EMBODIED_SRC"] = str(source_root)
    return env


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = (REPO_ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    poses_path = (REPO_ROOT / args.poses).resolve() if not args.poses.is_absolute() else args.poses
    world_path = (REPO_ROOT / args.world).resolve() if not args.world.is_absolute() else args.world
    webots = args.webots or (REPO_ROOT / DEFAULT_WEBOTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not poses_path.is_file():
        raise SystemExit(f"no poses file at {poses_path}")
    if not world_path.is_file():
        raise SystemExit(f"no world at {world_path}")
    if not webots.is_file():
        raise SystemExit(f"no Webots binary at {webots}")

    capture_world = derive_world(world_path, out_dir, poses_path, args.settle_steps)
    argv_list = [
        str(webots),
        "--batch",
        "--minimize",
        "--mode=realtime",
        "--stdout",
        "--stderr",
        str(capture_world),
    ]
    print("launching:", " ".join(argv_list), flush=True)
    process = subprocess.Popen(
        argv_list,
        cwd=str(REPO_ROOT),
        env=controller_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []

    def reader():
        for line in process.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            if "CAPTURED" in line or DONE_MARKER in line or "controller" in line.lower():
                print("  |", line, flush=True)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    # Webots forwards a controller's own stdout only when the controller process
    # exits, so the marker can arrive late. Watch the output directory for the
    # expected pairs as well: the capture is finished when its product is on
    # disk, whatever the console has or has not relayed yet.
    expected_pairs = len(json.loads(poses_path.read_text(encoding="utf-8"))["poses"])
    deadline = time.monotonic() + args.timeout_s
    failure = None
    while time.monotonic() < deadline:
        if any(DONE_MARKER in line for line in lines):
            break
        if len(list(out_dir.glob("*-left.ppm"))) >= expected_pairs:
            print(f"all {expected_pairs} pair(s) on disk", flush=True)
            break
        # A controller Webots could not find falls back to its own <generic>
        # controller, which never captures. Fail on that rather than sitting out
        # the whole timeout: Webots' message also names every path it searched,
        # which is the thing worth reading when this breaks.
        for line in lines:
            if "Could not find the controller directory" in line:
                failure = "Webots could not find the capture controller"
                break
        if failure:
            break
        if process.poll() is not None:
            break
        time.sleep(0.5)

    if failure:
        print(f"\nFAILED: {failure}. Paths Webots searched:", flush=True)
        for line in lines:
            if "controllers/" in line:
                print("  |", line)
    elif process.poll() is None:
        print("timeout reached; terminating Webots", flush=True)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
    thread.join(timeout=5)
    if failure:
        return 1

    pairs = sorted(out_dir.glob("*-left.ppm"))
    record = out_dir / "capture-record.json"
    print(f"\nruntime argv: {json.dumps(argv_list)}")
    print(f"pairs written: {len(pairs)}")
    for pair in pairs:
        print("  ", pair.name, pair.stat().st_size, "bytes")
    if not record.is_file():
        print("FAILED: no capture-record.json; last output follows")
        for line in lines[-25:]:
            print("  |", line)
        return 1
    payload = json.loads(record.read_text())
    for entry in payload["poses"]:
        print(
            f"  {entry['id']}: commanded {entry['commanded_translation']} "
            f"-> reported {[round(v, 4) for v in entry['reported_translation']]}"
        )
    print("capture complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
