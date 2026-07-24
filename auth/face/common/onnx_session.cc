#include "onnx_session.h"

#include <algorithm>
#include <thread>

namespace biopass {

OnnxSession::OnnxSession(const std::string& model_path, const char* log_name)
    : env_(ORT_LOGGING_LEVEL_WARNING, log_name) {
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
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(),
                                                            shape.data(), shape.size());

  return session_->Run(Ort::RunOptions{nullptr}, input_names_cstr_.data(), &input_tensor, 1,
                       output_names_cstr_.data(), output_names_cstr_.size());
}

}  // namespace biopass
