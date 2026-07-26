#include "face_auth.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <memory>
#include <optional>
#include <vector>

#include "antispoof_check.h"
#include "camera_capture.h"
#include "debug_image_io.h"
#include "image_utils.h"

namespace biopass {

bool FaceAuth::isAvailable() const {
  // If the resident camera_session_ is already acquired (see
  // endAuthenticationSession()), trust it instead of opening a second,
  // throwaway probe session every call -- that probe alone would eat most of
  // the savings from keeping camera_session_ warm in the first place.
  if (camera_session_ && camera_session_->isOpen()) {
    return true;
  }
  return checkCameraAvailability(face_config_.camera);
}

void FaceAuth::ensureIrSession() {
  if (face_config_.anti_spoofing.ir_camera.has_value() &&
      !face_config_.anti_spoofing.ir_camera->empty() &&
      (!ir_camera_session_ || !ir_camera_session_->isOpen())) {
    ir_camera_session_ =
        openCameraSession(*face_config_.anti_spoofing.ir_camera, CameraCaptureFormat::V4L2Grey,
                          kIrCaptureWarmupFrames, kIrCaptureTimeoutMs);
  }
}

bool FaceAuth::ensureModelsLoaded() {
  if (detector_ && recognizer_) {
    return true;
  }

  const std::string detectModelPath =
      model_registry_.resolveModelPath(face_config_.detection.model_id).value_or("");
  const std::string recogModelPath =
      model_registry_.resolveModelPath(face_config_.recognition.model_id).value_or("");
  if (!std::ifstream(recogModelPath).good() || !std::ifstream(detectModelPath).good()) {
    spdlog::error("FaceAuth: Model files not found");
    return false;
  }

  try {
    detector_ =
        std::make_unique<FaceDetection>(detectModelPath, 640, face_config_.detection.threshold);
    spdlog::debug("FaceAuth: Detection model loaded | threshold={:.3f}",
                  face_config_.detection.threshold);
  } catch (const std::exception& e) {
    std::string msg = e.what();
    size_t first_line = msg.find('\n');
    if (first_line != std::string::npos)
      msg = msg.substr(0, first_line);
    spdlog::error("FaceAuth: Failed to load detection model: {}, skipping", msg);
    return false;
  }

  try {
    recognizer_ =
        std::make_unique<FaceRecognition>(recogModelPath, 112, face_config_.recognition.threshold);
    spdlog::debug("FaceAuth: Recognition model loaded | threshold={:.3f}",
                  face_config_.recognition.threshold);
  } catch (const std::exception& e) {
    std::string msg = e.what();
    size_t first_line = msg.find('\n');
    if (first_line != std::string::npos)
      msg = msg.substr(0, first_line);
    spdlog::error("FaceAuth: Failed to load recognition model: {}, skipping", msg);
    detector_.reset();
    return false;
  }

  return true;
}

void FaceAuth::beginAuthenticationSession() {
  if (!camera_session_) {
    camera_session_ = openCameraSession(face_config_.camera);
  }
  ensureIrSession();
  ensureModelsLoaded();
}

void FaceAuth::endAuthenticationSession() {
  // Always torn down: a deliberate anti-spoofing property (see
  // checkAntiSpoof() in authenticate()) so a later call can never reuse a
  // partially-warmed IR session to bypass the presence check.
  ir_camera_session_.reset();

  // The RGB session is deliberately left resident (issue #152 follow-up):
  // LibcameraCaptureSession now only holds the camera acquire()d/
  // configure()d between calls -- no active streaming, no LED power -- and
  // starts/stops the actual stream per capture() (see startStreaming()/
  // stopStreaming() in camera_capture.cc). Keeping it here lets repeat auth
  // attempts on a warm daemon skip acquire/configure/buffer-allocation, not
  // just model reload.
}

void FaceAuth::releaseIdleResources() {
  // libcamera only allows one exclusive acquire() per device: if some other
  // process (e.g. the settings UI's enrollment/preview helper) needs
  // '/dev/video0' right now, our warm-but-idle hold on it must be dropped
  // first or that process's own acquire() fails outright. The next auth
  // attempt just re-opens it, same as any other cold start.
  camera_session_.reset();
  ir_camera_session_.reset();
}

AuthResult FaceAuth::authenticate(const std::string& username, const AuthConfig& config,
                                  std::atomic<bool>* cancel_signal) {
  if (!camera_session_) {
    camera_session_ = openCameraSession(face_config_.camera);
  }
  if (!camera_session_ || !camera_session_->isOpen()) {
    spdlog::error("FaceAuth: Could not open camera");
    if (!checkCameraAvailability(face_config_.camera)) {
      return AuthResult::Unavailable;
    }
    return AuthResult::Retry;
  }

  std::vector<std::string> enrolledFaces = biopass::listFaces(username);
  if (enrolledFaces.empty()) {
    spdlog::error("FaceAuth: No face enrolled for user {}, skipping", username);
    return AuthResult::Unavailable;
  }

  if (!ensureModelsLoaded()) {
    spdlog::error("FaceAuth: Models not available for user {}, skipping", username);
    return AuthResult::Unavailable;
  }

  if (cancel_signal && cancel_signal->load()) {
    return AuthResult::Failure;
  }

  ImageRGB loginFace = camera_session_->capture();
  if (loginFace.empty()) {
    spdlog::error("FaceAuth: Could not read frame");
    camera_session_.reset();
    return AuthResult::Retry;
  }

  std::vector<Detection> detectedImages = detector_->inference(loginFace);
  if (detectedImages.empty()) {
    spdlog::error("FaceAuth: No face detected");
    return AuthResult::Retry;
  }

  ImageRGB face = detectedImages[0].image;

  ensureIrSession();

  if (!checkAntiSpoof(face_config_, username, face, config, model_registry_, detector_.get(),
                      ir_camera_session_.get())) {
    spdlog::warn("FaceAuth: Anti-spoofing failed — returning Failure (no retry allowed)");
    // Always tear down the IR session so a subsequent call cannot reuse a
    // partially-warmed camera to bypass the check.
    ir_camera_session_.reset();
    return AuthResult::Failure;
  }

  // Match against all enrolled faces — succeed if any match.
  spdlog::debug("FaceAuth: Recognition | threshold={:.3f} enrolled_count={}",
                face_config_.recognition.threshold, enrolledFaces.size());
  for (const auto& facePath : enrolledFaces) {
    ImageRGB preparedFace = readImage(facePath);
    if (preparedFace.empty()) {
      spdlog::warn("FaceAuth: Recognition | could not load enrolled image: {}", facePath);
      continue;
    }

    MatchResult match = recognizer_->match(preparedFace, face);
    spdlog::debug("FaceAuth: Recognition | face='{}' score={:.4f} threshold={:.3f} similar={}",
                  facePath, match.dist, face_config_.recognition.threshold, match.similar);
    if (match.similar) {
      spdlog::debug("FaceAuth: Recognition PASSED | matched face='{}' score={:.4f}", facePath,
                    match.dist);
      return AuthResult::Success;
    }
  }

  if (config.debug) {
    saveFailedFace(username, face, "not_similar");
  }

  return AuthResult::Retry;
}

}  // namespace biopass
