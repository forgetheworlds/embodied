#!/usr/bin/env bash
# Build the pinned estimator, then build ov_stream against it.
#
# P01-L plan section 3.5. The pin is not trusted from a previous log: the
# tarball's sha256 is re-derived and compared with the pin recorded in
# configs/first_indoor.yaml, and the OV_MSCKF BUILD OK marker is written only
# after the pinned tree has configured, built and installed in full. A re-run
# that finds a different sha256, or a log whose marker does not match the pin,
# stops rather than building something else.
#
# Layout it works on:
#   work/runs/p01-localization/pin-evidence/open_vins-2.7.tar.gz   the pinned tarball
#   work/runs/p01-localization/pin-evidence/open_vins-2.7/         the unpacked pin
#   work/runs/p01-localization/pin-evidence/open_vins-2.7/ov_msckf/build/install
#   work/runs/p01-localization/pin-evidence/build-openvins.log     the pin's evidence
#   estimator/ov_stream                                            the process this builds
#
# Exit 0 means: the pin on disk matches the configuration, the pinned library is
# installed, and ov_stream was built against it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVIDENCE="${REPO_ROOT}/work/runs/p01-localization/pin-evidence"
TARBALL="${EVIDENCE}/open_vins-2.7.tar.gz"
SOURCE="${EVIDENCE}/open_vins-2.7"
INSTALL="${SOURCE}/ov_msckf/build/install"
LOG="${EVIDENCE}/build-openvins.log"
CONFIG="${REPO_ROOT}/configs/first_indoor.yaml"
BUILD_MARKER="OV_MSCKF BUILD OK"

CERES_PREFIX="${CERES_PREFIX:-${EVIDENCE}/ceres-2.1.0-install}"
EIGEN_PREFIX="${EIGEN_PREFIX:-/opt/homebrew/opt/eigen@3/include/eigen3}"
OPENCV_PREFIX="${OPENCV_PREFIX:-/opt/homebrew/opt/opencv@4}"
# Boost is keg-only on this host, and the pinned library's own load commands
# point at this directory, so the link line names it rather than relying on a
# default search path.
BOOST_PREFIX="${BOOST_PREFIX:-/opt/homebrew/opt/boost}"

log() { printf '%s\n' "$*" | tee -a "${LOG}"; }

# --- 1. The pin, re-derived from the tarball ---------------------------------

configured_sha="$(python3 - "${CONFIG}" <<'PY'
import sys, yaml
document = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(document["localization"]["estimator"]["tarball_sha256"])
PY
)"
if [[ ! -f "${TARBALL}" ]]; then
  log "P01-L build: the pinned tarball is missing at ${TARBALL}"
  exit 1
fi
actual_sha="$(shasum -a 256 "${TARBALL}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${configured_sha}" ]]; then
  log "P01-L build: PIN MISMATCH — the tarball is ${actual_sha}, the configuration pins ${configured_sha}"
  exit 1
fi

# --- 2. Configure, build and install the pinned tree -------------------------

if grep -q "^${BUILD_MARKER}$" "${LOG}" 2>/dev/null && [[ -f "${INSTALL}/lib/libov_msckf_lib.dylib" ]]; then
  log "P01-L build: pin already built and installed at ${INSTALL} (sha256 ${actual_sha} verified)"
else
  if [[ ! -d "${SOURCE}" ]]; then
    log "P01-L build: unpacking the pinned tarball"
    tar -xzf "${TARBALL}" -C "${EVIDENCE}"
  fi
  log "# $(date -u +%Y-%m-%dT%H:%M:%SZ) building the pinned tree"
  log "# cmake version $(cmake --version | head -1)"
  log "# tarball sha256 ${actual_sha} matches the pin"
  # The host-dependency choices are recorded in PIN.md: OpenCV 4 (the pin
  # requires major version 4), Eigen 3 (Ceres 2.1.0 rejects Eigen 5), and Ceres
  # 2.1.0 from source (the pin implements the LocalParameterization API removed
  # in 2.2). ROS and the ArUco tracker are off because this driver is neither.
  #
  # Two stages on purpose: ov_core installs first, and ov_msckf is configured
  # against that install rather than against the source tree.
  for module in ov_core ov_msckf; do
    cmake -S "${SOURCE}/${module}" -B "${SOURCE}/${module}/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
      -DCMAKE_PREFIX_PATH="${OPENCV_PREFIX};${CERES_PREFIX}" \
      -DEIGEN3_INCLUDE_DIR="${EIGEN_PREFIX}" \
      -DENABLE_ROS=OFF -DENABLE_ARUCO_TAGS=OFF \
      -DCMAKE_CXX_FLAGS="-include cassert" \
      >>"${LOG}" 2>&1
    cmake --build "${SOURCE}/${module}/build" --parallel >>"${LOG}" 2>&1
    cmake --install "${SOURCE}/${module}/build" >>"${LOG}" 2>&1
    log "P01-L build: ${module} built and installed"
  done
  # Written only now, after the whole tree installed: this line is the pin's
  # evidence, and a failed build must not leave it.
  log "${BUILD_MARKER}"
fi

if ! grep -q "^${BUILD_MARKER}$" "${LOG}"; then
  log "P01-L build: the build log does not record ${BUILD_MARKER}; stopping"
  exit 1
fi

# --- 3. Build ov_stream against the installed pin ----------------------------

log "# $(date -u +%Y-%m-%dT%H:%M:%SZ) building estimator/ov_stream"
clang++ -std=c++14 -O2 -DROS_AVAILABLE=0 -DENABLE_ARUCO_TAGS=0 -include cassert \
  -I "${INSTALL}/include/open_vins" \
  -I "${EIGEN_PREFIX}" \
  -I "${CERES_PREFIX}/include" \
  -I /opt/homebrew/include \
  -I "${OPENCV_PREFIX}/include/opencv4" \
  "${REPO_ROOT}/estimator/ov_stream.cpp" \
  -o "${REPO_ROOT}/estimator/ov_stream" \
  -L "${INSTALL}/lib" -lov_msckf_lib -Wl,-rpath,"${INSTALL}/lib" \
  -L "${CERES_PREFIX}/lib" -lceres -Wl,-rpath,"${CERES_PREFIX}/lib" \
  -L "${OPENCV_PREFIX}/lib" -lopencv_core -lopencv_imgproc -lopencv_calib3d \
  -lopencv_features2d -lopencv_flann -lopencv_imgcodecs -lopencv_highgui \
  -lopencv_video -lopencv_videoio \
  -L "${BOOST_PREFIX}/lib" -lboost_filesystem -lboost_thread -lboost_date_time \
  >>"${LOG}" 2>&1

log "OV_STREAM BUILD OK: $(cd "${REPO_ROOT}" && ls -l estimator/ov_stream | awk '{print $5}') bytes, pin sha256 ${actual_sha}"
