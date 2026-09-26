#!/usr/bin/env python3
"""Headless load validator for the indoor scenario catalogue.

For every world listed in the catalogue this script:

1. launches it with the Webots binary in batch mode, exactly as the platform adapter
   would,
2. waits for the load to report success or fail,
3. checks the world exposes the declared device set,
4. writes one JSON result object per world into the output directory,

and exits non-zero if any world fails.

What "reports success" means, measured on this installation (R2025a) rather than
assumed: Webots prints its own ``INFO: <controller>: Starting controller: ...`` line
within seconds of a successful load, with the controller arguments the world declares.
The controller's own stdout is forwarded to the Webots console only when the controller
process exits, so the positive device marker — ``Listening for ardupilot SITL``, which
is printed only after every declared device resolved — appears when the controller ends
its 60 s SITL wait. A world that fails to parse never starts a controller; a world that
declares a device it does not have exits immediately with the controller's
``the scene has no device named ...`` error. Both are failures here.

Plain standard library plus PyYAML (already a project dependency). No new dependencies.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

DEFAULT_WEBOTS_RELATIVE = Path("work/Webots.app/Contents/MacOS/webots")
STARTING_MARKER = re.compile(r"INFO: .*: Starting controller: (.+)")
DEVICE_FAILURE_MARKER = "the scene has no device named"
SUCCESS_MARKER = "Listening for ardupilot SITL"
CONTROLLER_EXIT_MARKER = "controller exited with status"
IMPORT_FAILURE_MARKER = "cannot import embodied.platform.webots_ardupilot"
LOG_TAIL_LINES = 40


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalogue", required=True, type=Path,
                        help="path to scenarios/missions/catalogue.yaml")
    parser.add_argument("--output", required=True, type=Path,
                        help="directory to write one result JSON per world into")
    parser.add_argument("--webots", type=Path, default=None,
                        help="Webots binary (default: work/Webots.app/Contents/MacOS/webots "
                             "under the repository root)")
    parser.add_argument("--timeout-s", type=float, default=150.0,
                        help="per-world budget for a load to report success or fail")
    return parser.parse_args(argv)


def load_catalogue(path):
    with path.open(encoding="utf-8") as handle:
        catalogue = yaml.safe_load(handle)
    scenarios = catalogue.get("scenarios") or []
    if not scenarios:
        raise SystemExit(f"{path} lists no scenarios")
    return scenarios


def repo_root_of(catalogue_path):
    """Walk up from the catalogue to the repository root, marked by pyproject.toml."""
    for parent in catalogue_path.resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise SystemExit(f"no repository root above {catalogue_path}")


def controller_environment(repo_root):
    """The environment the Webots controller needs to import the shared framing."""
    source_root = repo_root / "src"
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    merged = os.pathsep.join(part for part in (str(source_root), existing) if part)
    env["PYTHONPATH"] = merged
    env["EMBODIED_SRC"] = str(source_root)
    return env


def launch(webots, world, repo_root):
    """Start Webots and return (process, lines, stop_reader)."""
    argv = [
        str(webots),
        "--batch",
        "--mode=realtime",
        "--stdout",
        "--stderr",
        str(world),
    ]
    process = subprocess.Popen(
        argv,
        cwd=str(repo_root),
        env=controller_environment(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []

    def reader():
        for line in process.stdout:
            lines.append(line.rstrip("\n"))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return process, argv, lines, thread


def stop(process):
    """End the batch run: Webots terminates its controller on SIGTERM."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def devices_present_in_args(declared, controller_args_line):
    """Every declared device name must appear in the world's controller arguments."""
    missing = []
    for device in declared:
        if device not in controller_args_line:
            missing.append(device)
    return missing


def judge(lines, declared, started_at, timeout_s, process):
    """Decide pass/fail from what the load reported so far.

    Returns (state, reason, controller_args) where state is one of
    "running", "passed", "failed".
    """
    joined = "\n".join(lines)
    controller_args = None
    match = STARTING_MARKER.search(joined)
    if match:
        controller_args = match.group(1)

    if DEVICE_FAILURE_MARKER in joined:
        for line in lines:
            if DEVICE_FAILURE_MARKER in line:
                return "failed", line.strip(), controller_args
        return "failed", DEVICE_FAILURE_MARKER, controller_args
    if IMPORT_FAILURE_MARKER in joined:
        return "failed", "controller could not import the shared framing", controller_args

    if SUCCESS_MARKER in joined:
        if controller_args is None:
            return "failed", "device marker without a controller start line", controller_args
        missing = devices_present_in_args(declared, controller_args)
        if missing:
            return "failed", f"controller arguments omit declared devices: {missing}", controller_args
        return "passed", "world loaded and every declared device resolved", controller_args

    if CONTROLLER_EXIT_MARKER in joined:
        return "failed", "controller exited before the declared devices were reported", controller_args

    if process.poll() is not None:
        return "failed", f"Webots exited with status {process.returncode} before reporting", controller_args

    if time.monotonic() - started_at > timeout_s:
        return "failed", f"no load report within {timeout_s:g} s", controller_args

    return "running", None, controller_args


def validate_one(entry, webots, repo_root, timeout_s):
    world = repo_root / entry["world"]
    started_at = time.monotonic()
    result = {
        "id": entry["id"],
        "group": entry["group"],
        "world": entry["world"],
        "declared_devices": entry["devices"],
        "status": "fail",
        "reason": None,
        "controller_args": None,
        "duration_s": 0.0,
        "command": None,
        "log_tail": [],
    }

    if not world.is_file():
        result["reason"] = f"world file does not exist: {world}"
        result["duration_s"] = round(time.monotonic() - started_at, 3)
        return result
    if not webots.is_file():
        result["reason"] = f"Webots binary does not exist: {webots}"
        result["duration_s"] = round(time.monotonic() - started_at, 3)
        return result

    process, argv, lines, _thread = launch(webots, world, repo_root)
    result["command"] = argv
    try:
        while True:
            state, reason, controller_args = judge(
                lines, entry["devices"], started_at, timeout_s, process
            )
            if controller_args:
                result["controller_args"] = controller_args
            if state != "running":
                result["status"] = "pass" if state == "passed" else "fail"
                result["reason"] = reason
                break
            time.sleep(0.25)
    finally:
        stop(process)
        result["duration_s"] = round(time.monotonic() - started_at, 3)
        result["log_tail"] = lines[-LOG_TAIL_LINES:]
    return result


def main(argv=None):
    args = parse_args(argv)
    catalogue_path = args.catalogue.resolve()
    scenarios = load_catalogue(catalogue_path)
    repo_root = repo_root_of(catalogue_path)
    webots = (args.webots or (repo_root / DEFAULT_WEBOTS_RELATIVE)).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    failures = 0
    for entry in scenarios:
        print(f"[{entry['group']}] {entry['id']} ... ", end="", flush=True)
        result = validate_one(entry, webots, repo_root, args.timeout_s)
        destination = output / f"{entry['id']}.json"
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"{result['status']} ({result['duration_s']:.1f} s): {result['reason']}")
        if result["status"] != "pass":
            failures += 1

    print(f"{len(scenarios) - failures}/{len(scenarios)} worlds loaded with their "
          f"declared devices; results in {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
