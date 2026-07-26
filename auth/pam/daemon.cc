// biopassd -- resident per-user authentication daemon.
//
// Fixes the cold-start latency described in issue #152: today every PAM
// attempt forks + execs `biopass-helper auth`, which constructs a brand new
// AuthManager (and therefore a brand new FaceAuth) from scratch -- reopening
// the camera and reloading every ONNX model on every single login/unlock/sudo
// attempt.
//
// This daemon keeps exactly one AuthManager warm for as long as it's been
// recently used. FaceAuth::ensureModelsLoaded() already short-circuits once
// detector_/recognizer_ are populated (see face_auth.cc), so the second and
// subsequent auth() calls against the same warm object skip model
// construction entirely. Camera sessions are still opened/closed per request
// via the existing beginAuthenticationSession()/endAuthenticationSession()
// hooks (MethodSessionGuard in auth_manager.cc already tears the camera down
// after every call) -- this patch does not change that behavior, and
// intentionally does not try to hold the camera open indefinitely between
// attempts.
//
// Lifecycle (this is the important part -- this process is NOT meant to run
// forever): it's launched on demand via systemd socket activation
// (biopassd.socket owns the listening socket; systemd only spawns this
// binary on the first incoming connection) and self-exits after
// kIdleTimeoutSeconds of no requests, releasing every loaded model and the
// camera handle. Memory footprint while idle is therefore zero (no process
// even running) rather than "a small resident model staying loaded forever."
// The next auth attempt after an idle exit pays a fresh cold-start cost --
// systemd re-spawns the process, which re-inherits the same still-open
// listening socket from systemd -- but every attempt within an active burst
// (e.g. a normal work session) stays warm. This mirrors how Windows'
// biometric service (WbioSrvc) behaves: always registered, but lazy to
// actually load, and able to go back to sleep.
//
// Falls back to a plain self-bind (not socket-activated) if run without
// systemd's LISTEN_FDS handoff, so it can still be started/tested manually.
//
// Security model:
//   - Runs as a systemd --user unit: one instance per logged-in user, under
//     that user's own uid.
//   - Listens on a Unix domain socket under $XDG_RUNTIME_DIR (mode 0700,
//     already owned by the user by systemd-logind convention), which we
//     additionally chmod to 0600.
//   - Refuses to authenticate any username other than its own owning user.
//     This is a deliberate, hard restriction: this daemon is not a general
//     auth broker, it only ever vouches for the one user it runs as.
//   - The client (pam.cc) independently verifies the connected peer's uid
//     via SO_PEERCRED before trusting a response, so a rogue local process
//     squatting on the expected socket path can't self-approve.
//
// Wire protocol (line-based, matches the style of previewSession() in
// helper.cc): client sends `AUTH <username> <service>\n`; daemon replies
// `RESULT <code>\n` where code is 0 (PAM_SUCCESS), 1 (PAM_AUTH_ERR), or
// 2 (PAM_IGNORE) -- the same codes biopass-helper's process exit code
// already uses, so pam.cc's result handling is unchanged either way.
//
// A second request, `RELEASE\n` -> `OK\n`, exists purely so another local
// process (namely the settings UI, before it spawns biopass-helper
// preview-session for enrollment) can ask us to drop any camera device
// we're holding warm between auth attempts. Without this, the UI's helper
// and this daemon fight over the same exclusive-access libcamera handle
// whenever an idle warm window overlaps with enrollment/preview.

#include <security/_pam_types.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <string>

#include "auth_config.h"
#include "auth_manager.h"
#include "face_auth.h"
#include "fingerprint_auth.h"

namespace {

std::atomic<bool> g_shutdown{false};

void handleSignal(int) { g_shutdown.store(true); }

// Builds (or rebuilds) the warm AuthManager from the on-disk config, exactly
// mirroring authenticate() in helper.cc so the daemon and the cold-start
// fallback path always agree on behavior.
std::unique_ptr<biopass::AuthManager> buildManager(const std::string& username,
                                                    biopass::BiopassConfig& out_config) {
  out_config = biopass::readConfig(username);

  biopass::AuthConfig runtime_config;
  runtime_config.debug = out_config.strategy.debug;
  runtime_config.antispoof = out_config.methods.face.anti_spoofing.enable ||
                             (out_config.methods.face.anti_spoofing.ir_camera.has_value() &&
                              !out_config.methods.face.anti_spoofing.ir_camera->empty());

  auto manager = std::make_unique<biopass::AuthManager>();
  manager->setMode(out_config.strategy.execution_mode == "sequential"
                        ? biopass::ExecutionMode::Sequential
                        : biopass::ExecutionMode::Parallel);
  manager->setConfig(runtime_config);

  int numOfMethods = 0;
  for (const auto& method_name : out_config.strategy.order) {
    if (method_name == "face" && out_config.methods.face.enable) {
      manager->addMethod(std::make_unique<biopass::FaceAuth>(out_config.methods.face, username));
      numOfMethods++;
    } else if (method_name == "fingerprint" && out_config.methods.fingerprint.enable) {
      manager->addMethod(std::make_unique<biopass::FingerprintAuth>(out_config.methods.fingerprint));
      numOfMethods++;
    }
  }

  if (numOfMethods == 0) {
    return nullptr;
  }
  return manager;
}

// PAM result codes as already used by biopass-helper's exit codes.
constexpr int kPamSuccess = 0;
constexpr int kPamAuthErr = 1;
constexpr int kPamIgnore = 2;

class ResidentAuthenticator {
 public:
  explicit ResidentAuthenticator(std::string owner_username)
      : owner_username_(std::move(owner_username)) {}

  // Serialized: only one auth attempt runs at a time. A single user is
  // extremely unlikely to trigger two concurrent unlocks, and FaceAuth's
  // camera_session_ is not designed for concurrent use anyway.
  int authenticate(const std::string& username, const std::string& service) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (username != owner_username_) {
      // Hard refusal by design -- see security model note at file top.
      spdlog::warn("biopassd: refusing auth request for '{}' (daemon owned by '{}')", username,
                   owner_username_);
      return kPamIgnore;
    }

    if (!biopass::configExists(username.c_str())) {
      return kPamIgnore;
    }

    biopass::BiopassConfig config;
    // Re-read config + rebuild on every call so config.yaml edits (e.g. from
    // the Tauri settings app) take effect without needing a daemon restart.
    // This still avoids the cold-start cost: FaceAuth objects are freshly
    // constructed here, but ensureModelsLoaded()/camera reopen are the only
    // per-call costs paid by the *first* call after a config change -- the
    // expensive part (model file I/O + ONNX Runtime session construction)
    // only happens again if config actually changed, since we cache the
    // manager keyed by a cheap hash of the relevant config fields.
    std::string config_fingerprint = configFingerprint(username);
    if (!cached_manager_ || config_fingerprint != cached_fingerprint_) {
      auto manager = buildManager(username, config);
      if (!manager) {
        return kPamIgnore;
      }
      cached_manager_ = std::move(manager);
      cached_fingerprint_ = config_fingerprint;
      spdlog::info("biopassd: (re)built AuthManager for '{}'", username);
    }

    if (!service.empty()) {
      biopass::BiopassConfig current = biopass::readConfig(username);
      if (std::find(current.strategy.ignore_services.begin(),
                    current.strategy.ignore_services.end(),
                    service) != current.strategy.ignore_services.end()) {
        return kPamIgnore;
      }
    }

    int retval = cached_manager_->authenticate(username);
    return (retval == PAM_SUCCESS) ? kPamSuccess : kPamAuthErr;
  }

  // Handles a "RELEASE" request: an external process (e.g. the settings UI
  // about to spawn biopass-helper for enrollment/preview) needs the camera
  // device we may be holding warm. Shares the same mutex as authenticate()
  // so this can never race a concurrent auth attempt's camera use.
  void releaseCamera() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cached_manager_) {
      cached_manager_->releaseCameraResources();
    }
  }

 private:
  // Cheap "has the config that matters changed" check -- not a full config
  // diff, just enough to invalidate the warm manager when method
  // enable/order/model_id fields change. Avoids rebuilding (and reloading
  // every ONNX model) on every single call "just in case".
  std::string configFingerprint(const std::string& username) const {
    biopass::BiopassConfig c = biopass::readConfig(username);
    std::ostringstream oss;
    oss << c.schema_version << '|' << c.strategy.execution_mode << '|';
    for (const auto& m : c.strategy.order) oss << m << ',';
    oss << '|' << c.methods.face.enable << c.methods.face.detection.model_id
        << c.methods.face.recognition.model_id << c.methods.face.anti_spoofing.enable
        << c.methods.face.anti_spoofing.model.model_id << '|' << c.methods.fingerprint.enable;
    return oss.str();
  }

  std::string owner_username_;
  std::mutex mutex_;
  std::unique_ptr<biopass::AuthManager> cached_manager_;
  std::string cached_fingerprint_;
};

// Parses "AUTH <username> <service>" (service may be empty/omitted).
// Returns false if the line isn't a well-formed AUTH request.
bool parseAuthLine(const std::string& line, std::string& username, std::string& service) {
  if (line.rfind("AUTH ", 0) != 0) return false;
  std::istringstream iss(line.substr(5));
  if (!(iss >> username)) return false;
  iss >> service;  // optional, defaults to "" if absent
  return true;
}

std::string socketPath() {
  const char* runtime_dir = std::getenv("XDG_RUNTIME_DIR");
  std::string base = runtime_dir ? runtime_dir : ("/run/user/" + std::to_string(getuid()));
  return base + "/biopass-daemon.sock";
}

// systemd's socket-activation ABI (see systemd.socket(5), sd_listen_fds(3)):
// when a .socket unit spawns this process, it hands over the already-bound,
// already-listening socket as fd 3 and sets LISTEN_FDS=1 / LISTEN_PID=<our
// pid> in the environment. We deliberately don't link libsystemd just to
// call sd_listen_fds() for a single fd -- reading the two env vars directly
// is the whole protocol.
constexpr int kSdListenFdsStart = 3;

int getSystemdActivatedFd() {
  const char* listen_pid = std::getenv("LISTEN_PID");
  const char* listen_fds = std::getenv("LISTEN_FDS");
  if (!listen_pid || !listen_fds) return -1;
  if (std::atoi(listen_pid) != static_cast<int>(getpid())) return -1;
  if (std::atoi(listen_fds) < 1) return -1;
  return kSdListenFdsStart;
}

// How long the daemon waits for a request before exiting voluntarily. This
// is the whole point of the design: rather than staying resident forever
// (loaded models + open camera handle sitting in memory indefinitely), the
// process exits after being idle and systemd's socket unit simply respawns
// it fresh on the next connection. Override via BIOPASSD_IDLE_TIMEOUT_SECS
// in the environment (e.g. set in biopassd.service) -- no single Linux
// convention mandates a specific value, so this is intentionally tunable
// rather than hardcoded. Defaults to 30 minutes, which comfortably covers a
// burst of login/unlock/sudo attempts in one sitting without staying loaded
// through a whole idle workday.
int idleTimeoutSeconds() {
  const char* override_env = std::getenv("BIOPASSD_IDLE_TIMEOUT_SECS");
  if (override_env) {
    int parsed = std::atoi(override_env);
    if (parsed > 0) return parsed;
  }
  return 30 * 60;
}

}  // namespace

int main() {
  const char* logged_in_user = getenv("USER");
  std::string username = logged_in_user ? logged_in_user : "";
  if (username.empty()) {
    spdlog::error("biopassd: could not determine owning username (no $USER); exiting");
    return 1;
  }

  std::signal(SIGTERM, handleSignal);
  std::signal(SIGINT, handleSignal);

  std::string sock_path = socketPath();
  int server_fd = getSystemdActivatedFd();
  bool self_bound = false;

  if (server_fd >= 0) {
    spdlog::info("biopassd: using systemd-provided socket (fd {})", server_fd);
  } else {
    // Not socket-activated -- e.g. started manually for testing. Fall back
    // to binding it ourselves so the daemon is still usable standalone.
    self_bound = true;
    ::unlink(sock_path.c_str());  // remove stale socket from a previous run

    server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
      spdlog::error("biopassd: socket() failed: {}", strerror(errno));
      return 1;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      spdlog::error("biopassd: bind({}) failed: {}", sock_path, strerror(errno));
      close(server_fd);
      return 1;
    }

    // Belt-and-suspenders: $XDG_RUNTIME_DIR is already 0700 owned by this
    // user, but explicitly lock the socket file down too.
    chmod(sock_path.c_str(), S_IRUSR | S_IWUSR);

    if (listen(server_fd, 8) != 0) {
      spdlog::error("biopassd: listen() failed: {}", strerror(errno));
      close(server_fd);
      return 1;
    }
    spdlog::info("biopassd: self-bound listening socket at {} (not systemd-activated)", sock_path);
  }

  const int idle_timeout_secs = idleTimeoutSeconds();
  spdlog::info("biopassd: ready for user '{}' (idle exit after {}s of no requests)", username,
              idle_timeout_secs);
  ResidentAuthenticator authenticator(username);

  while (!g_shutdown.load()) {
    pollfd pfd{};
    pfd.fd = server_fd;
    pfd.events = POLLIN;

    // Block for at most idle_timeout_secs waiting for a new connection. A
    // timeout here (poll() returning 0) means nobody has authenticated in
    // that whole window -- that's our cue to exit voluntarily rather than
    // sit resident with warm models nobody is using.
    int poll_result = poll(&pfd, 1, idle_timeout_secs * 1000);
    if (poll_result < 0) {
      if (errno == EINTR) continue;
      spdlog::error("biopassd: poll() failed: {}", strerror(errno));
      break;
    }
    if (poll_result == 0) {
      spdlog::info("biopassd: idle for {}s, exiting to free camera/model memory",
                  idle_timeout_secs);
      break;
    }

    sockaddr_un client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (errno == EINTR) continue;
      spdlog::error("biopassd: accept() failed: {}", strerror(errno));
      continue;
    }

    // Verify the connecting peer is either this same user or root (e.g. a
    // display manager's PAM stack running privileged). Anyone else gets
    // rejected outright, before we even read their request.
    ucred peer_cred{};
    socklen_t cred_len = sizeof(peer_cred);
    bool peer_ok = false;
    if (getsockopt(client_fd, SOL_SOCKET, SO_PEERCRED, &peer_cred, &cred_len) == 0) {
      peer_ok = (peer_cred.uid == getuid()) || (peer_cred.uid == 0);
    }
    if (!peer_ok) {
      spdlog::warn("biopassd: rejected connection from untrusted peer uid");
      close(client_fd);
      continue;
    }

    char buf[512];
    ssize_t n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) {
      close(client_fd);
      continue;
    }
    buf[n] = '\0';
    std::string line(buf);
    // Trim trailing newline.
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();

    std::string response;
    if (line == "RELEASE") {
      // Best-effort hand-off of the camera to whoever asked (e.g. the
      // settings UI's enrollment preview) -- doesn't affect idle timeout or
      // otherwise change the loop's control flow.
      authenticator.releaseCamera();
      response = "OK\n";
    } else {
      std::string req_username, req_service;
      int result = kPamIgnore;
      if (parseAuthLine(line, req_username, req_service)) {
        result = authenticator.authenticate(req_username, req_service);
      } else {
        spdlog::warn("biopassd: malformed request: '{}'", line);
      }
      response = "RESULT " + std::to_string(result) + "\n";
    }

    write(client_fd, response.c_str(), response.size());
    close(client_fd);
  }

  spdlog::info("biopassd: shutting down");
  close(server_fd);
  if (self_bound) {
    ::unlink(sock_path.c_str());
  }
  // If systemd-activated, deliberately leave the socket file alone --
  // systemd itself owns and will keep that socket around, ready to spawn a
  // fresh instance on the next connection.
  return 0;
}
