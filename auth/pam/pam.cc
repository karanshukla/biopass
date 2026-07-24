#include <pwd.h>
#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>

namespace {

// Tries the resident biopassd daemon for `username`'s login session, if one
// is running (see daemon.cc for the full design/security rationale).
// Returns true and fills `out_result` (a PAM_* code) on a clean round trip;
// returns false on any failure whatsoever, in which case the caller must
// fall back to the existing cold-start fork+exec path unchanged. This
// function must never be able to make authentication *more* likely to
// succeed than the fallback path -- any ambiguity resolves to "not handled".
bool tryResidentDaemon(const std::string& username, const char* service, int& out_result) {
  struct passwd* pw = getpwnam(username.c_str());
  if (!pw) return false;

  std::string sock_path = "/run/user/" + std::to_string(pw->pw_uid) + "/biopass-daemon.sock";

  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return false;

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

  // If the daemon isn't running, connect() fails near-instantly
  // (ECONNREFUSED / ENOENT) since nothing is listening on that path -- no
  // risk of hanging the whole PAM stack waiting on it.
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return false;
  }

  // Verify the peer is actually running as the target user (or root) before
  // trusting anything it says. This is the client-side half of the mutual
  // check described in daemon.cc -- protects against a rogue local process
  // squatting on the expected socket path.
  ucred peer_cred{};
  socklen_t cred_len = sizeof(peer_cred);
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer_cred, &cred_len) != 0 ||
      (peer_cred.uid != pw->pw_uid && peer_cred.uid != 0)) {
    close(fd);
    return false;
  }

  std::string request = "AUTH " + username + " " + (service ? service : "") + "\n";
  if (write(fd, request.c_str(), request.size()) < 0) {
    close(fd);
    return false;
  }

  char buf[128];
  ssize_t n = read(fd, buf, sizeof(buf) - 1);
  close(fd);
  if (n <= 0) return false;
  buf[n] = '\0';

  int code = -1;
  if (sscanf(buf, "RESULT %d", &code) != 1) return false;

  switch (code) {
    case 0:
      out_result = PAM_SUCCESS;
      return true;
    case 2:
      out_result = PAM_IGNORE;
      return true;
    case 1:
      out_result = PAM_AUTH_ERR;
      return true;
    default:
      return false;  // unrecognized code -- don't trust it, fall back
  }
}

}  // namespace

// Called by PAM when a user needs to be authenticated
PAM_EXTERN int pam_sm_authenticate(pam_handle_t* pamh, int flags, int argc, const char** argv) {
  (void)flags;
  (void)argc;
  (void)argv;

  int retval;

  const char* service = nullptr;
  retval = pam_get_item(pamh, PAM_SERVICE, (const void**)&service);
  if (retval != PAM_SUCCESS) {
    service = nullptr;
  }

  const char* pUsername;
  retval = pam_get_user(pamh, &pUsername, NULL);
  if (retval != PAM_SUCCESS) {
    return retval;
  }

  // Fast path: a warm resident daemon for this user avoids the cold-start
  // cost entirely (see issue #152). Any failure here -- daemon not running,
  // wrong peer uid, malformed response -- falls straight through to the
  // existing fork+exec path below with zero behavior change.
  int daemon_result;
  if (tryResidentDaemon(pUsername, service, daemon_result)) {
    return daemon_result;
  }

  pid_t pid = fork();
  if (pid < 0) {
    return PAM_AUTH_ERR;
  } else if (pid == 0) {
    // Run "biopass-helper auth --username <username> [--service <name>]"
    if (service != nullptr && service[0] != '\0') {
      execl("/usr/bin/biopass-helper", "biopass-helper", "auth", "--username", pUsername,
            "--service", service, NULL);
    } else {
      execl("/usr/bin/biopass-helper", "biopass-helper", "auth", "--username", pUsername, NULL);
    }

    // If execl returns, it failed. Don't perror() here: this process's
    // stdio is inherited from the PAM caller (e.g. polkit-agent-helper-1),
    // which some callers (GNOME Shell's polkit agent) parse as a strict
    // line protocol -- any unexpected line on it derails the caller's
    // authentication state machine instead of a clean failure.
    exit(1);
  } else {
    int status;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
      int exit_code = WEXITSTATUS(status);
      if (exit_code == 0) {
        return PAM_SUCCESS;
      } else if (exit_code == 2) {
        return PAM_IGNORE;
      } else {
        return PAM_AUTH_ERR;
      }
    } else {
      // Child did not exit normally (e.g., killed by signal)
      return PAM_AUTH_ERR;
    }
  }
}

// The functions below are required by PAM, but not needed in this module
PAM_EXTERN int pam_sm_open_session(pam_handle_t* pamh, int flags, int argc, const char** argv) {
  (void)pamh;
  (void)flags;
  (void)argc;
  (void)argv;
  return PAM_IGNORE;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t* pamh, int flags, int argc, const char** argv) {
  (void)pamh;
  (void)flags;
  (void)argc;
  (void)argv;
  return PAM_IGNORE;
}

PAM_EXTERN int pam_sm_close_session(pam_handle_t* pamh, int flags, int argc, const char** argv) {
  (void)pamh;
  (void)flags;
  (void)argc;
  (void)argv;
  return PAM_IGNORE;
}

PAM_EXTERN int pam_sm_chauthtok(pam_handle_t* pamh, int flags, int argc, const char** argv) {
  (void)pamh;
  (void)flags;
  (void)argc;
  (void)argv;
  return PAM_IGNORE;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t* pamh, int flags, int argc, const char** argv) {
  (void)pamh;
  (void)flags;
  (void)argc;
  (void)argv;
  return PAM_IGNORE;
}
