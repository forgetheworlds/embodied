#!/usr/bin/env bash
# One-time setup for the embodied repository: Python environment, ArduPilot
# SITL at the pinned commit, and a check for Webots.
#
# The layout matches configs/first_indoor.yaml: the ArduPilot clone lives in
# work/ardupilot and the Webots app bundle in work/Webots.app, both relative
# to the repository root. Everything is safe to re-run; finished steps are
# skipped.
#
# Manual step this script cannot do for you: Webots R2025a must be installed
# on this machine and the app bundle placed at work/Webots.app (macOS layout,
# so that work/Webots.app/Contents/MacOS/webots exists).

set -euo pipefail
cd "$(dirname "$0")/.."

ARDUPILOT_COMMIT="af8525911b49a4c2a3bdc83bc5ed57e1d0098134"
ARDUPILOT_DIR="work/ardupilot"
WEBOTS_APP="work/Webots.app"

fail() {
    echo "bootstrap: $*" >&2
    exit 1
}

# 1. Python 3.11+ and the pinned dependencies.
command -v python3 >/dev/null || fail "python3 not found on PATH"
python3 - <<'EOF' || fail "python3 is too old; this project needs 3.11 or newer"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
EOF

if [ ! -x .venv/bin/pip ]; then
    echo "== creating .venv"
    python3 -m venv .venv
fi
echo "== installing the pinned Python dependencies into .venv"
.venv/bin/pip install -q -e '.[dev]'

# 2. ArduPilot SITL at the pinned commit, with submodules, built via waf.
if [ -x "$ARDUPILOT_DIR/build/sitl/bin/arducopter" ]; then
    echo "== $ARDUPILOT_DIR/build/sitl/bin/arducopter already built, skipping"
else
    if [ ! -d "$ARDUPILOT_DIR" ]; then
        echo "== cloning ArduPilot into $ARDUPILOT_DIR (large, with submodules)"
        git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git "$ARDUPILOT_DIR"
    fi
    (
        cd "$ARDUPILOT_DIR"
        current=$(git rev-parse HEAD)
        if [ "$current" != "$ARDUPILOT_COMMIT" ]; then
            echo "== checking out the pinned ArduPilot commit $ARDUPILOT_COMMIT"
            git checkout "$ARDUPILOT_COMMIT"
            git submodule update --init --recursive
        fi
        echo "== building ArduPilot SITL (a few minutes)"
        ./waf configure --board sitl
        ./waf copter
    )
    [ -x "$ARDUPILOT_DIR/build/sitl/bin/arducopter" ] \
        || fail "the waf build finished but $ARDUPILOT_DIR/build/sitl/bin/arducopter is missing"
fi

# 3. Webots. The download is manual (license click-through), so the script
#    only checks for it and says what to do if it is missing.
if [ ! -x "$WEBOTS_APP/Contents/MacOS/webots" ]; then
    fail "Webots was not found at $WEBOTS_APP.
Install Webots R2025a (https://cyberbotics.com/#download), then either move
the app bundle to work/Webots.app or adjust webots_home in
configs/first_indoor.yaml to where you installed it."
fi

echo "== setup complete"
echo "   tests:  .venv/bin/python -m pytest -q"
echo "   run:    .venv/bin/python -m embodied compat --config configs/first_indoor.yaml --output work/runs/<new-dir>"
