// ov_stream: the pinned OpenVINS v2.7 process behind one local socket.
//
// P01-L plan section 3.4. This program links libov_msckf_lib built from the
// pinned tarball (tag v2.7, commit 93adc241390d13e99232652cf05cbe18a93c7bea)
// and speaks exactly one protocol on 127.0.0.1. Its frame layout is the one
// already implemented and tested in src/embodied/platform/localization.py
// (FRAME_MAGIC 0x4F56, FRAME_HEADER "<HBBI", KIND_IMU/STEREO/RESET/STATE); this
// side must match it byte for byte, which the Python tests T1 pin.
//
// It receives declared sensor data and nothing else: inertial samples and
// grayscale stereo pairs, with a RESET command. **There is no truth field in
// this protocol by construction**, so no truth can reach the estimator through
// it. It never touches MAVLink, the bridge, or the simulator.
//
// Frames (little-endian throughout, one length-prefixed frame per message).
// The header is localization.py's FRAME_HEADER, struct "<HBBI" -- EIGHT bytes, not
// nine: magic u16, kind u8, flags u8, payload length u32. Getting the flags width
// wrong shifts the length field by one byte, and the first frame is then read as a
// zero-length payload. Measured the hard way on 2026-09-26.
//   [magic u16][kind u8][flags u8][payload length u32][payload]
//   host->estimator: IMU {time_ns:i64, gyro[3]:f64, accel[3]:f64}
//                    STEREO {time_ns:u64, width:u32, height:u32, left, right}
//                    RESET {}
//   estimator->host: STATE {time_ns:u64, initialized:u8, quat wxyz[4]:f64,
//                    pos[3], vel[3], gyro_bias[3], accel_bias[3], sigma_pos[3],
//                    n_tracks:u32, t_last_visual_ns:u64, reset_counter:u8}
//   -> "<QB19dIQB", 174 bytes, matching localization.py's _STATE_PAYLOAD.
//
// CONVENTIONS, each pinned rather than guessed (plan section 3.4's T2):
// 0. Inertial cadence and published-state currency. Every inertial sample is
//    fed to the manager the moment its frame is parsed, never held for the
//    next image: feeding only queues inside the manager (propagator,
//    initializer, ZUPT feeder) and never advances state->_timestamp, so an
//    image arriving after newer inertial samples is still in order -- the
//    pin's out-of-order drop and propagate_and_clone exits key on
//    state->_timestamp, which only camera/ZUPT updates move, and
//    select_imu_readings then interpolates its final segment to the image
//    time instead of extrapolating it (its case 3.4 fallback). The published
//    STATE is the pin's fast_state_propagate projection of the current state
//    to the newest consumed inertial sample -- the between-frames odometry
//    path the pin's own ROS driver publishes -- so the pose the autopilot
//    fuses is current rather than pinned to the last image. The filter's
//    state, its clone structure and the stereo update semantics are
//    untouched by the projection.
//
// 1. Inertial input. The declared transport delivers inertial data in the
//    autopilot's body frame (x forward, y right, z down): the controller
//    converts the Webots devices with its documented "ENU negated on y and z
//    into NED". OpenVINS's global frame is gravity-aligned with +z up
//    (Propagator's _gravity = (0, 0, +gravity_mag)), so a level, stationary
//    vehicle must present its accelerometer as (0, 0, +9.81). The mapping used
//    here is therefore the 180-degree x rotation (x, -y, -z) -- the proper
//    rotation that takes the FRD body frame to its z-up (FLU) counterpart.
//    Feeding FRD-shaped data unchanged is the failure this exists to prevent:
//    the filter would see gravity reversed and never initialize.
//
// 2. Camera frame. OpenVINS wants the OpenCV optical convention (x right,
//    y down, z forward). The declared calibration record gives
//    T_body_camera_left = identity rotation, so the Webots camera's own frame
//    is the body frame (x forward, y left, z up) and the record states the
//    image convention: "the column axis runs along the body's rightward (-y)
//    axis and the row axis runs downward (-z)". The optical frame is therefore
//    (x right, y down, z forward) = (-body_y, -body_z, +body_x), and the
//    extrinsic below is that rotation, derived in the code from the record's
//    declared translations rather than typed in as a matrix.
//
// 3. Reported attitude. The adapter (OdomAlignment) applies the fixed
//    ENU->NED axis swap itself and needs body->odom. OpenVINS's quat() is
//    q_GtoI in its (x, y, z, w) JPL order, whose quat_2_Rot is the JPL form
//    (the negative skew term), so the conjugated (w, x, y, z) numbers packed
//    below read, under the adapter's HAMILTON quat_to_rotmat, as ODOM->BODY:
//    conjugating in JPL and re-reading as Hamilton cancel. Measured, run
//    p01l-fix3b-20260927T050042Z: reading them as body->odom published the
//    inverse rotation on every axis. The adapter therefore transposes its
//    read (localization.py _delivered_body_to_odom); do not "fix" that
//    transpose away without re-measuring against dataflash truth.
//
// 4. Position and velocity need no rotation: the estimator's global frame is
//    z-up and its origin is the initialization point, which is exactly what the
//    adapter's alignment expects (plan section 4.2).
//
// Build: estimator/build-openvins.sh, against the pin.

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/Propagator.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "utils/sensor_data.h"

namespace {

// ---------------------------------------------------------------------------
// The wire protocol (must match localization.py exactly)
// ---------------------------------------------------------------------------

const uint16_t FRAME_MAGIC = 0x4F56;  // "OV"
const uint8_t KIND_IMU = 1;
const uint8_t KIND_STEREO = 2;
const uint8_t KIND_RESET = 3;
const uint8_t KIND_STATE = 4;

// localization.py's FRAME_HEADER is struct "<HBBI": eight bytes, with the length at
// byte 4. These two numbers are the whole contract; every offset below uses them.
const size_t FRAME_HEADER_SIZE = 8;
const size_t FRAME_LENGTH_OFFSET = 4;

const double IMU_PERIOD_NS = 2e6;      // declared 500 Hz (configs/first_indoor.yaml:73)
const double STEREO_PERIOD_NS = 1e8;   // declared 10 Hz (:65)
const size_t STATE_PAYLOAD_SIZE = 8 + 1 + 19 * 8 + 4 + 8 + 1;  // "<QB19dIQB"
const double PUBLISH_PERIOD_S = 0.010; // 100 Hz: F2's 20 ms bound is unreachable on 25 ms ticks (FIXER5)

struct Reader {
  int fd = -1;
  std::vector<uint8_t> buffer;

  // Fills the buffer. Returns false on a closed or failed connection, true
  // otherwise; whether a whole frame has arrived is a separate question.
  bool read_available() {
    uint8_t chunk[1 << 16];
    ssize_t got = ::recv(fd, chunk, sizeof(chunk), 0);
    if (got > 0) {
      buffer.insert(buffer.end(), chunk, chunk + got);
      return true;
    }
    if (got == 0) return false;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return true;
    return false;
  }
};

uint16_t read_u16(const uint8_t *p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

uint32_t read_u32(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

void write_u16(std::vector<uint8_t> &out, uint16_t v) { out.insert(out.end(), (uint8_t *)&v, (uint8_t *)&v + sizeof(v)); }
void write_u32(std::vector<uint8_t> &out, uint32_t v) { out.insert(out.end(), (uint8_t *)&v, (uint8_t *)&v + sizeof(v)); }
void write_u64(std::vector<uint8_t> &out, uint64_t v) { out.insert(out.end(), (uint8_t *)&v, (uint8_t *)&v + sizeof(v)); }
void write_u8(std::vector<uint8_t> &out, uint8_t v) { out.push_back(v); }
void write_f64(std::vector<uint8_t> &out, double v) { out.insert(out.end(), (uint8_t *)&v, (uint8_t *)&v + sizeof(v)); }

void write_header(std::vector<uint8_t> &out, uint8_t kind, uint32_t length) {
  write_u16(out, FRAME_MAGIC);
  write_u8(out, kind);
  write_u8(out, 0);
  write_u32(out, length);
}

bool send_all(int fd, const std::vector<uint8_t> &bytes) {
  size_t sent = 0;
  while (sent < bytes.size()) {
    ssize_t wrote = ::send(fd, bytes.data() + sent, bytes.size() - sent, 0);
    if (wrote > 0) {
      sent += (size_t)wrote;
      continue;
    }
    if (wrote < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// The estimator's configuration, from the declared rig
// ---------------------------------------------------------------------------

// scenarios/compat/calibration.json, rig first-indoor-stereo-1 (P01-C
// re-versioned the same declared geometry as first-indoor-stereo-1 v2):
//   left_intrinsics.focal_length_px   [554.2562584220407, 554.2562584220407]
//   left_intrinsics.principal_point_px [320.0, 240.0]
//   left_distortion model none_declared, coefficients []
//   T_body_camera_left  quaternion_wxyz [1,0,0,0], translation_m [0.05, 0.05, 0.05]
//   T_camera_left_camera_right quaternion_wxyz [1,0,0,0], translation_m [0,-0.1,0]
//   baseline_m 0.1
const double FOCAL_PX = 554.2562584220407;
const double PRINCIPAL_X = 320.0;
const double PRINCIPAL_Y = 240.0;
const int IMAGE_WIDTH = 640;
const int IMAGE_HEIGHT = 480;

const double LEFT_CAMERA_IN_BODY[3] = {0.05, 0.05, 0.05};
const double RIGHT_CAMERA_IN_BODY[3] = {0.05, -0.05, 0.05};  // left + (0, -0.10, 0)

// The optical frame's axes expressed in the body frame (convention 2 above):
// columns are x_O, y_O, z_O.
const double OPTICAL_TO_BODY[3][3] = {
    {0.0, 0.0, 1.0},
    {-1.0, 0.0, 0.0},
    {0.0, -1.0, 0.0},
};

Eigen::Matrix3d optical_to_body() {
  Eigen::Matrix3d m;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) m(r, c) = OPTICAL_TO_BODY[r][c];
  return m;
}

// The extrinsic OpenVINS wants: (q_ItoC in (x,y,z,w), p_IinC).
Eigen::VectorXd camera_extrinsic(const double camera_in_body[3]) {
  Eigen::Matrix3d R_OtoB = optical_to_body();      // optical -> body
  Eigen::Matrix3d R_BtoO = R_OtoB.transpose();     // body -> optical
  Eigen::Vector3d p_camera_in_body(camera_in_body[0], camera_in_body[1], camera_in_body[2]);
  Eigen::Vector3d p_IinC = R_BtoO * (-p_camera_in_body);
  Eigen::Vector4d q_ItoC = ov_core::rot_2_quat(R_BtoO);  // (x, y, z, w)
  Eigen::VectorXd out(7);
  out << q_ItoC, p_IinC;
  return out;
}

// body (FRD) -> the estimator's z-up inertial frame: (x, -y, -z).
Eigen::Vector3d to_estimator_frame(const double xyz[3]) {
  return Eigen::Vector3d(xyz[0], -xyz[1], -xyz[2]);
}

void build_options(ov_msckf::VioManagerOptions &params) {
  params.state_options.num_cameras = 2;
  params.use_stereo = true;
  params.use_klt = true;
  params.use_aruco = false;
  params.gravity_mag = 9.81;
  params.calib_camimu_dt = 0.0;  // declared-by-construction, measured error bound 0.0607 s

  params.camera_intrinsics.clear();
  params.camera_extrinsics.clear();
  for (size_t i = 0; i < 2; ++i) {
    auto camera = std::make_shared<ov_core::CamRadtan>(IMAGE_WIDTH, IMAGE_HEIGHT);
    Eigen::VectorXd calib(8);
    calib << FOCAL_PX, FOCAL_PX, PRINCIPAL_X, PRINCIPAL_Y, 0.0, 0.0, 0.0, 0.0;
    camera->set_value(calib);
    params.camera_intrinsics[i] = camera;
    params.camera_extrinsics[i] =
        camera_extrinsic(i == 0 ? LEFT_CAMERA_IN_BODY : RIGHT_CAMERA_IN_BODY);
  }

  // The no-YAML path leaves five Eigen members uninitialised and then writes
  // them into the state and uses them in propagation unconditionally
  // (VioManagerOptions::print_and_load_state fills them only when a parser is
  // present). The identity values the YAML path would produce are set here
  // explicitly; leaving them untouched is undefined behaviour, not a default.
  params.vec_dw << 1, 0, 0, 1, 0, 1;
  params.vec_da << 1, 0, 0, 1, 0, 1;
  params.vec_tg.setZero();
  params.q_ACCtoIMU << 0, 0, 0, 1;
  params.q_GYROtoIMU << 0, 0, 0, 1;

  // The initializer carries its own copies of the rig, checked independently
  // against num_cameras inside the constructor with a hard exit. Both sides
  // must agree, so they are set from one place.
  params.init_options.num_cameras = params.state_options.num_cameras;
  params.init_options.camera_intrinsics = params.camera_intrinsics;
  params.init_options.camera_extrinsics = params.camera_extrinsics;
  params.init_options.gravity_mag = params.gravity_mag;
  params.init_options.calib_camimu_dt = params.calib_camimu_dt;
  params.init_options.use_stereo = params.use_stereo;

  // The declared route starts stationary and initializes before arm (plan
  // section 6, H5), so the static path must be admissible: without a
  // zero-velocity updater the initializer requires an acceleration jerk, which
  // a stationary launch interval never supplies.
  params.try_zupt = true;
  // The pin's zero-velocity accept gate bypasses chi2 and the velocity bound
  // entirely once the image disparity test passes (UpdaterZeroVelocity.cpp:241
  // keys on !disparity_passed first; the override flag is a hardcoded local at
  // :113, not a parameter), and a vehicle that is physically near-stationary
  // while fighting a wrong pose shows sub-pixel disparity: measured in runs
  // p01l-fix2-20260927T020445Z ("accepted |v_IinG| = 0.065 (chi2 10489.080 <
  // 84.595)", 790 accepts, state frozen ~1.5 m from truth, E1 1.517 m) and
  // p01l-fix2l-20260927T033814Z (749 accepts, the covariance corrupted until a
  // published sigma read 0.0, 291 stop/recover cycles). The pin's own lever is
  // zupt_only_at_beginning (VioManagerOptions.h:95): it gates all three ZUPT
  // feed sites (VioManager.cpp:186/221/294) on !has_moved_since_zupt, which
  // latches at the first completed visual update (VioManager.cpp:360, past the
  // five-clone window). The static start is untouched: the initializer is fed
  // independently (:180-182) and the updater only ever runs once
  // is_initialized_vio, so the excitation's motion both initializes and latches
  // ZUPT off before the scored window opens.
  params.zupt_only_at_beginning = true;
}

// ---------------------------------------------------------------------------
// One STATE frame (must match localization.py's decode_state)
// ---------------------------------------------------------------------------

void log_line(const char *format, ...);

std::vector<uint8_t> encode_state(const std::shared_ptr<ov_msckf::VioManager> &sys,
                                  uint8_t reset_counter, double newest_imu_time) {
  auto state = sys->get_state();
  auto imu = state->_imu;

  // The filter's own timestamp is the last camera/ZUPT update; between updates
  // the newest consumed inertial sample is ahead of it. Publish the projection
  // to that sample when the propagator can cover the interval, and the state
  // as-is otherwise. A refusal is diagnostically interesting -- it is the
  // difference between a current pose and a stale one -- so the first one and
  // every 200th after it are logged rather than swallowed silently.
  static uint64_t refusals = 0;
  Eigen::Vector4d q_GtoI;
  Eigen::Vector3d position, velocity;
  double sigma[3] = {0.0, 0.0, 0.0};
  double publish_time = state->_timestamp;
  Eigen::Matrix<double, 13, 1> state_plus;
  Eigen::Matrix<double, 12, 12> cov_plus;
  if (newest_imu_time > state->_timestamp &&
      sys->get_propagator()->fast_state_propagate(state, newest_imu_time, state_plus, cov_plus)) {
    publish_time = newest_imu_time;
    q_GtoI = state_plus.block(0, 0, 4, 1);
    position = state_plus.block(4, 0, 3, 1);
    // state_plus carries the body-frame velocity (R_GtoI * v_G); the payload's
    // contract is the estimator's global frame, so rotate it back.
    velocity = ov_core::quat_2_Rot(q_GtoI).transpose() * state_plus.block(7, 0, 3, 1);
    for (int i = 0; i < 3; ++i) {
      double variance = cov_plus(3 + i, 3 + i);
      sigma[i] = variance > 0.0 ? std::sqrt(variance) : 0.0;
    }
  } else if (newest_imu_time > state->_timestamp) {
    refusals += 1;
    if (refusals == 1 || refusals % 200 == 0) {
      log_line("ov_stream: projection refused (%llu so far): state_t=%.3f newest_imu=%.3f",
               (unsigned long long)refusals, state->_timestamp, newest_imu_time);
    }
  } else {
    q_GtoI = imu->quat();
    position = imu->pos();
    velocity = imu->vel();
    // Per-axis position sigma: the marginal covariance of the IMU state in
    // [q(3), p(3), v(3), bg(3), ba(3)] order, so position is rows/cols 3..5.
    std::vector<std::shared_ptr<ov_type::Type>> variables{imu};
    Eigen::MatrixXd cov = ov_msckf::StateHelper::get_marginal_covariance(state, variables);
    for (int i = 0; i < 3; ++i) {
      double variance = cov(3 + i, 3 + i);
      sigma[i] = variance > 0.0 ? std::sqrt(variance) : 0.0;
    }
  }

  // Convention 3: body->odom, which is the inverse of OpenVINS's q_GtoI.
  Eigen::Vector4d q_ItoG(-q_GtoI(0), -q_GtoI(1), -q_GtoI(2), q_GtoI(3));

  // The camera-clock time of the last update, taken into the IMU clock by the
  // declared cam-imu offset (zero here, so the two are equal).
  double t_last_visual_s = state->_timestamp + state->_calib_dt_CAMtoIMU->value()(0);

  std::vector<uint8_t> frame;
  write_header(frame, KIND_STATE, (uint32_t)STATE_PAYLOAD_SIZE);
  write_u64(frame, (uint64_t)std::llround(publish_time * 1e9));
  write_u8(frame, (uint8_t)(sys->initialized() ? 1 : 0));
  write_f64(frame, q_ItoG(3));  // w
  write_f64(frame, q_ItoG(0));  // x
  write_f64(frame, q_ItoG(1));  // y
  write_f64(frame, q_ItoG(2));  // z
  for (int i = 0; i < 3; ++i) write_f64(frame, position(i));
  for (int i = 0; i < 3; ++i) write_f64(frame, velocity(i));
  for (int i = 0; i < 3; ++i) write_f64(frame, imu->bias_g()(i));
  for (int i = 0; i < 3; ++i) write_f64(frame, imu->bias_a()(i));
  for (int i = 0; i < 3; ++i) write_f64(frame, sigma[i]);

  // VioManager exposes no live track count while uninitialised and no accessor
  // for its feature database (both recorded absent in the pin's notes), so the
  // honest count is what the public accessor returns and zero before then. It
  // is recorded as a diagnostic, never as a bound: plan section 7 declines to
  // claim a threshold from it.
  double active_time = -1.0;
  std::unordered_map<size_t, Eigen::Vector3d> positions;
  std::unordered_map<size_t, Eigen::Vector3d> uvd;
  sys->get_active_tracks(active_time, positions, uvd);
  write_u32(frame, (uint32_t)uvd.size());
  write_u64(frame, (uint64_t)std::llround(t_last_visual_s * 1e9));
  write_u8(frame, reset_counter);
  return frame;
}

void log_line(const char *format, ...) {
  va_list args;
  va_start(args, format);
  std::vfprintf(stdout, format, args);
  va_end(args);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ov_stream <port>\n");
    return 2;
  }
  int port = std::atoi(argv[1]);
  if (port <= 0) {
    std::fprintf(stderr, "ov_stream: %s is not a port\n", argv[1]);
    return 2;
  }

  ov_msckf::VioManagerOptions params;
  build_options(params);
  auto sys = std::make_shared<ov_msckf::VioManager>(params);
  uint8_t reset_counter = 0;
  log_line(
      "ov_stream ready: openvins v2.7, %zu cameras, gravity %.2f, try_zupt=%d, "
      "imu period %.0f ms, stereo period %.0f ms",
      params.camera_intrinsics.size(), params.gravity_mag, (int)params.try_zupt,
      IMU_PERIOD_NS / 1e6, STEREO_PERIOD_NS / 1e6);

  int listener = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) {
    std::fprintf(stderr, "ov_stream: cannot create the listener: %s\n", std::strerror(errno));
    return 1;
  }
  int reuse = 1;
  ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons((uint16_t)port);
  if (::bind(listener, (sockaddr *)&address, sizeof(address)) != 0) {
    std::fprintf(stderr, "ov_stream: cannot bind 127.0.0.1:%d: %s\n", port, std::strerror(errno));
    return 1;
  }
  if (::listen(listener, 1) != 0) {
    std::fprintf(stderr, "ov_stream: cannot listen: %s\n", std::strerror(errno));
    return 1;
  }

  // The filter persists across connections, and the process serves them in turn.
  // A host that probes the port and closes -- which is how the check command waits
  // for this process to be ready -- must cost nothing: its connection is one of
  // these, and the next one carries the stream. Only a malformed frame or a dead
  // listener ends the process.
  double newest_imu_time = 0.0;
  uint64_t stereo_frames = 0;
  uint64_t imu_samples = 0;
  uint64_t published = 0;
  uint64_t connections = 0;

  while (true) {
    int client = ::accept(listener, nullptr, nullptr);
    if (client < 0) {
      std::fprintf(stderr, "ov_stream: accept failed: %s\n", std::strerror(errno));
      return 1;
    }
    int flags = ::fcntl(client, F_GETFL, 0);
    ::fcntl(client, F_SETFL, flags | O_NONBLOCK);
    connections += 1;
    log_line("ov_stream: host connected (connection %llu)", (unsigned long long)connections);

    Reader reader;
    reader.fd = client;
    auto last_publish = std::chrono::steady_clock::now();
    newest_imu_time = 0.0;

    while (true) {
    fd_set readable;
    FD_ZERO(&readable);
    FD_SET(client, &readable);
    timeval timeout{};
    timeout.tv_usec = 5000;
    int ready = ::select(client + 1, &readable, nullptr, nullptr, &timeout);
    if (ready < 0 && errno != EINTR) break;
    if (ready > 0 && !reader.read_available()) {
      log_line("ov_stream: the host closed the connection");
      break;
    }

    // Drain whole frames. Inertial samples are fed the moment they are parsed
    // (convention 0): the wire carries every sample older than an image before
    // that image, so the ordering OpenVINS requires holds by construction, and
    // samples newer than the pending image are harmless to the filter.
    while (reader.buffer.size() >= FRAME_HEADER_SIZE) {
      uint16_t magic = read_u16(reader.buffer.data());
      uint8_t kind = reader.buffer[2];
      uint32_t length = read_u32(reader.buffer.data() + FRAME_LENGTH_OFFSET);
      if (magic != FRAME_MAGIC) {
        std::fprintf(stderr, "ov_stream: frame magic 0x%04x is not the ov_stream magic\n", magic);
        return 1;
      }
      if (reader.buffer.size() < FRAME_HEADER_SIZE + length) break;
      const uint8_t *payload = reader.buffer.data() + FRAME_HEADER_SIZE;

      if (kind == KIND_IMU) {
        if (length != 8 + 6 * 8) {
          std::fprintf(stderr, "ov_stream: IMU payload is %u bytes, the pinned layout is 56\n", length);
          return 1;
        }
        int64_t time_ns;
        std::memcpy(&time_ns, payload, sizeof(time_ns));
        double values[6];
        std::memcpy(values, payload + 8, sizeof(values));
        ov_core::ImuData message;
        message.timestamp = (double)time_ns / 1e9;
        message.wm = to_estimator_frame(values);
        message.am = to_estimator_frame(values + 3);
        sys->feed_measurement_imu(message);
        newest_imu_time = message.timestamp;
        imu_samples += 1;
      } else if (kind == KIND_STEREO) {
        if (length < 16) {
          std::fprintf(stderr, "ov_stream: STEREO payload is %u bytes, too short for its header\n", length);
          return 1;
        }
        uint64_t time_ns;
        std::memcpy(&time_ns, payload, sizeof(time_ns));
        uint32_t width = read_u32(payload + 8);
        uint32_t height = read_u32(payload + 12);
        size_t plane = (size_t)width * (size_t)height;
        if (length != 16 + 2 * plane) {
          std::fprintf(stderr,
                       "ov_stream: STEREO payload is %u bytes, %ux%u needs %zu\n",
                       length, width, height, 16 + 2 * plane);
          return 1;
        }
        double image_time = (double)time_ns / 1e9;

        ov_core::CameraData message;
        message.timestamp = image_time;
        for (size_t camera = 0; camera < 2; ++camera) {
          const uint8_t *plane_bytes = payload + 16 + camera * plane;
          cv::Mat image((int)height, (int)width, CV_8UC1);
          std::memcpy(image.data, plane_bytes, plane);
          message.sensor_ids.push_back((int)camera);
          message.images.push_back(image);
          // The tracker hard-exits unless masks match images, and its convention
          // is inverted from the word: a pixel is skipped when its mask is
          // greater than 127, so "no mask" is a zero matrix.
          message.masks.push_back(cv::Mat::zeros((int)height, (int)width, CV_8UC1));
        }
        sys->feed_measurement_camera(message);
        stereo_frames += 1;
        if (stereo_frames <= 10 || stereo_frames % 25 == 0) {
          // Diagnostic: the frame's own timestamp, the pixel statistics of what the
          // tracker was handed, and how many features it is carrying. Without these,
          // "the estimator gets no visual updates" cannot be told apart from "the
          // frames arrive stale, repeated, or blank", and all three look alike.
          double active_time = -1.0;
          std::unordered_map<size_t, Eigen::Vector3d> positions;
          std::unordered_map<size_t, Eigen::Vector3d> uvd;
          sys->get_active_tracks(active_time, positions, uvd);
          cv::Scalar mean_l, sd_l, mean_r, sd_r;
          cv::meanStdDev(message.images[0], mean_l, sd_l);
          cv::meanStdDev(message.images[1], mean_r, sd_r);
          log_line("ov_stream: %llu stereo frames, %llu imu samples, initialized=%d, "
                   "frame_t=%.3f, tracks=%zu, left_mean=%.1f, left_sd=%.2f, right_mean=%.1f, right_sd=%.2f",
                   (unsigned long long)stereo_frames, (unsigned long long)imu_samples,
                   (int)sys->initialized(), image_time, uvd.size(),
                   mean_l[0], sd_l[0], mean_r[0], sd_r[0]);
        }
      } else if (kind == KIND_RESET) {
        // A reset is asked for, never improvised: the old state is abandoned and
        // a fresh manager is built. Nothing is published until it initializes
        // again, so the adapter sees silence rather than a jump (plan section
        // 4.4); its own reset counter is what rides the published messages.
        reset_counter += 1;
        newest_imu_time = 0.0;
        sys = std::make_shared<ov_msckf::VioManager>(params);
        log_line("ov_stream: reset %u: fresh filter", (unsigned)reset_counter);
      } else {
        std::fprintf(stderr, "ov_stream: frame kind %u is not in the ov_stream protocol\n", kind);
        return 1;
      }
      reader.buffer.erase(
          reader.buffer.begin(), reader.buffer.begin() + FRAME_HEADER_SIZE + length);
    }

    auto now = std::chrono::steady_clock::now();
    double since_publish = std::chrono::duration<double>(now - last_publish).count();
    if (since_publish >= PUBLISH_PERIOD_S) {
      last_publish = now;
      if (sys->initialized()) {
        std::vector<uint8_t> frame = encode_state(sys, reset_counter, newest_imu_time);
        if (!send_all(client, frame)) {
          log_line("ov_stream: the host is gone while publishing");
          break;
        }
        published += 1;
        if (published == 1 || published % 400 == 0) {
          auto state = sys->get_state();
          Eigen::Vector3d p = state->_imu->pos();
          log_line("ov_stream: published %llu states, p=(%.3f, %.3f, %.3f)",
                   (unsigned long long)published, p(0), p(1), p(2));
        }
      }
    }
    }

    ::close(client);
    log_line(
        "ov_stream: connection %llu closed; imu samples %llu, stereo frames %llu, "
        "published %llu, resets %u so far",
        (unsigned long long)connections, (unsigned long long)imu_samples,
        (unsigned long long)stereo_frames, (unsigned long long)published,
        (unsigned)reset_counter);
  }

  ::close(listener);
  return 0;
}
