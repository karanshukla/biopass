#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#ifdef BIOPASS_HAVE_OPENVINO
#include <openvino/openvino.hpp>
#endif

namespace biopass {

// Owns the inference session plumbing shared by every inference engine in
// this module (detection, recognition, anti-spoofing). Engines compose this
// and only implement their own pre/postprocessing; the public interface
// (constructor + run(), returning Ort::Value) never changes based on backend,
// so callers don't know or care which one actually ran.
//
// Backend: ONNX Runtime CPU always works and is the default. When built with
// BIOPASS_HAVE_OPENVINO, the constructor additionally probes OpenVINO devices
// (NPU, then GPU) and switches to whichever compiles first; if neither does
// (no such device, driver/plugin missing, etc.) it silently falls back to the
// ONNX Runtime CPU path above. Override the probe with the
// BIOPASS_INFERENCE_DEVICE env var (AUTO|NPU|GPU|CPU) -- CPU always forces
// ONNX Runtime, skipping the OpenVINO probe entirely.
//
// !!! BIOPASS_HAVE_OPENVINO is confirmed to crash the resident biopassd
// daemon (heap corruption, `malloc(): invalid size`) the first time any
// OnnxSession here constructs -- even with BIOPASS_INFERENCE_DEVICE=CPU,
// i.e. without ever calling into the OpenVINO API. Not root-caused; see the
// warning in Dependencies.cmake and the writeup linked there before
// re-enabling BIOPASS_USE_OPENVINO for anything other than isolated,
// single-purpose test binaries.
//
// allow_openvino lets a caller opt a model out of the probe altogether
// regardless of BIOPASS_INFERENCE_DEVICE. FaceDetection (YOLO) passes false:
// its ONNX export has a dynamic reshape buried inside the graph (not just
// the input batch dim, which is the only thing the NPU-side fix below
// handles) that crashed the NPU compiler with a hard abort() during
// development -- not a throwable exception, so no amount of try/catch in
// this class can protect against it. Until that's root-caused on the model
// side, detection stays CPU-only unconditionally.
class OnnxSession {
 public:
  OnnxSession(const std::string& model_path, const char* log_name, bool allow_openvino = true);

  std::vector<Ort::Value> run(std::vector<float>& input, const std::vector<int64_t>& shape);

 private:
  std::vector<Ort::Value> runOrt(std::vector<float>& input, const std::vector<int64_t>& shape);

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::vector<std::string> input_names_str_;
  std::vector<std::string> output_names_str_;
  std::vector<const char*> input_names_cstr_;
  std::vector<const char*> output_names_cstr_;

#ifdef BIOPASS_HAVE_OPENVINO
  // Tries each candidate OpenVINO device in order, returns true (and leaves
  // using_openvino_ set) on the first that successfully compiles the model.
  bool initOpenVino(const std::string& model_path, const char* log_name);
  std::vector<Ort::Value> runOpenVino(std::vector<float>& input, const std::vector<int64_t>& shape);

  bool using_openvino_ = false;
  std::string ov_device_;  // "NPU" or "GPU" -- which one this instance landed on, for logging.
  ov::CompiledModel ov_compiled_model_;
  ov::InferRequest ov_infer_request_;
  // Backing store for the most recent inference's output. The Ort::Value
  // returned by runOpenVino() is a non-owning view over this buffer, so it is
  // only valid until the next run() call on this same OnnxSession instance --
  // matches how every existing call site already consumes the tensor
  // immediately after run() returns, before calling it again.
  std::vector<float> ov_output_buffer_;
  std::vector<int64_t> ov_output_shape_;
#endif
};

}  // namespace biopass
