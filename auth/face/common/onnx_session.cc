#include "onnx_session.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

#ifdef BIOPASS_HAVE_OPENVINO
#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#endif

namespace biopass {

namespace {
// AUTO (default) probes NPU then GPU, falling back to ONNX Runtime CPU if
// neither is present. NPU/GPU force that single device (still falling back to
// CPU if it doesn't compile). CPU skips the OpenVINO probe entirely.
std::string preferredDevice() {
  const char* env = std::getenv("BIOPASS_INFERENCE_DEVICE");
  return env ? std::string(env) : std::string("AUTO");
}
}  // namespace

OnnxSession::OnnxSession(const std::string& model_path, const char* log_name, bool allow_openvino)
    : env_(ORT_LOGGING_LEVEL_WARNING, log_name) {
#ifdef BIOPASS_HAVE_OPENVINO
  const std::string want = preferredDevice();
  if (allow_openvino && want != "CPU" && initOpenVino(model_path, log_name)) {
    return;
  }
#else
  (void)allow_openvino;
#endif

  Ort::SessionOptions opts;
  // Intra-op parallelism: model inference (YOLO detection especially) is the
  // dominant CPU cost of a warm authentication, and single-threaded left most
  // cores idle. Cap at 4 so that if two sessions ever run concurrently (e.g.
  // the AI + IR anti-spoofing checks via std::async) they don't badly
  // oversubscribe, and clamp to the machine's actual core count.
  const unsigned hw = std::thread::hardware_concurrency();
  const int intra_op_threads = static_cast<int>(std::max(1u, std::min(4u, hw == 0 ? 1u : hw)));
  opts.SetIntraOpNumThreads(intra_op_threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);

  for (size_t i = 0; i < session_->GetInputCount(); i++) {
    auto name = session_->GetInputNameAllocated(i, allocator_);
    input_names_str_.push_back(name.get());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); i++) {
    auto name = session_->GetOutputNameAllocated(i, allocator_);
    output_names_str_.push_back(name.get());
  }
  for (auto& s : input_names_str_) input_names_cstr_.push_back(s.c_str());
  for (auto& s : output_names_str_) output_names_cstr_.push_back(s.c_str());
}

std::vector<Ort::Value> OnnxSession::run(std::vector<float>& input,
                                         const std::vector<int64_t>& shape) {
#ifdef BIOPASS_HAVE_OPENVINO
  if (using_openvino_) {
    return runOpenVino(input, shape);
  }
#endif
  return runOrt(input, shape);
}

std::vector<Ort::Value> OnnxSession::runOrt(std::vector<float>& input,
                                            const std::vector<int64_t>& shape) {
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(),
                                                            shape.data(), shape.size());

  return session_->Run(Ort::RunOptions{nullptr}, input_names_cstr_.data(), &input_tensor, 1,
                       output_names_cstr_.data(), output_names_cstr_.size());
}

#ifdef BIOPASS_HAVE_OPENVINO
bool OnnxSession::initOpenVino(const std::string& model_path, const char* log_name) {
  // One Core per process: device/plugin enumeration is expensive and every
  // OnnxSession (detection, recognition, anti-spoofing) would otherwise
  // repeat it. Function-local static keeps it off the header.
  static ov::Core core = [] {
    ov::Core c;
    // Anti-spoofing (unlike detection/recognition) rebuilds its OnnxSession
    // on every single auth attempt (see checkAntiSpoofByAIModel() in
    // antispoof_check.cc) rather than caching it for the daemon's lifetime.
    // NPU/GPU model compilation is far more expensive than an ONNX Runtime
    // CPU session load, so without a persistent compile cache that path would
    // pay full recompilation cost on every login -- likely net *slower* than
    // just using CPU. ov::cache_dir makes compile_model() reuse a compiled
    // blob keyed by model+device+config after the first run, even across
    // process restarts.
    const std::string cache_dir = "/var/cache/biopassd/openvino";
    if (::mkdir("/var/cache/biopassd", 0700) == 0 || errno == EEXIST) {
      if (::mkdir(cache_dir.c_str(), 0700) == 0 || errno == EEXIST) {
        c.set_property(ov::cache_dir(cache_dir));
      }
    }
    return c;
  }();

  std::vector<std::string> candidates;
  const std::string want = preferredDevice();
  if (want == "NPU" || want == "GPU") {
    candidates = {want};
  } else {
    candidates = {"NPU", "GPU"};
  }

  const auto available = core.get_available_devices();
  for (const auto& device : candidates) {
    if (std::find(available.begin(), available.end(), device) == available.end()) {
      spdlog::debug("{}: OpenVINO device '{}' not present on this host, skipping", log_name,
                    device);
      continue;
    }
    try {
      auto model = core.read_model(model_path);

      // NPU/GPU compilers require fully static shapes; every model in this
      // codebase is exported with a dynamic (symbolic) batch dimension,
      // which compile_model() otherwise rejects (NPU: "missing upper bound
      // for one or more nodes"). Every call site always runs single-image
      // inference, so pinning batch=1 is exact, not an approximation.
      for (auto& input : model->inputs()) {
        ov::PartialShape shape = input.get_partial_shape();
        shape[0] = 1;
        model->reshape({{input.get_any_name(), shape}});
      }

      ov_compiled_model_ = core.compile_model(model, device);
      ov_infer_request_ = ov_compiled_model_.create_infer_request();
      using_openvino_ = true;
      ov_device_ = device;
      spdlog::info("{}: using OpenVINO device '{}' (compile_model succeeded)", log_name, device);
      return true;
    } catch (const std::exception& e) {
      spdlog::warn("{}: OpenVINO device '{}' failed to compile model ({}), trying next", log_name,
                   device, e.what());
    }
  }

  if (want == "NPU" || want == "GPU") {
    spdlog::warn("{}: requested OpenVINO device '{}' unavailable, falling back to ONNX Runtime CPU",
                 log_name, want);
  }
  return false;
}

std::vector<Ort::Value> OnnxSession::runOpenVino(std::vector<float>& input,
                                                 const std::vector<int64_t>& shape) {
  const auto start = std::chrono::steady_clock::now();

  ov::Shape ov_shape(shape.begin(), shape.end());
  ov::Tensor input_tensor(ov::element::f32, ov_shape, input.data());
  ov_infer_request_.set_input_tensor(input_tensor);
  ov_infer_request_.infer();

  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  // Per-call proof the NPU/GPU is actually doing the work, not just that it
  // compiled at startup -- watch for this line (journalctl -u biopassd -f)
  // while testing an auth attempt.
  spdlog::debug("OnnxSession: OpenVINO '{}' inference took {:.2f}ms", ov_device_, elapsed_ms);

  ov::Tensor output_tensor = ov_infer_request_.get_output_tensor(0);
  const float* data = output_tensor.data<float>();
  ov_output_buffer_.assign(data, data + output_tensor.get_size());

  ov_output_shape_.clear();
  for (auto d : output_tensor.get_shape()) {
    ov_output_shape_.push_back(static_cast<int64_t>(d));
  }

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> result;
  result.push_back(Ort::Value::CreateTensor<float>(memory_info, ov_output_buffer_.data(),
                                                   ov_output_buffer_.size(), ov_output_shape_.data(),
                                                   ov_output_shape_.size()));
  return result;
}
#endif

}  // namespace biopass
