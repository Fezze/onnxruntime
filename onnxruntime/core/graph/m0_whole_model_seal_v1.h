// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include "core/common/inlined_containers.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
class M0OriginalPreloadOwnerV1;
namespace logging {
class Logger;
}

// The private factory authenticates owned bytes before retaining the parsed model.
class M0WholeModelSealV1 final {
 private:
  friend class M0OriginalPreloadOwnerV1;
  friend struct std::default_delete<M0WholeModelSealV1>;
  static std::unique_ptr<M0WholeModelSealV1> ValidateAndParse(std::string_view bytes,
                                                          const logging::Logger& logger);
  class Snapshot;
  M0WholeModelSealV1();
  ~M0WholeModelSealV1();
  ONNX_NAMESPACE::ModelProto TakeLiveModel();
  bool SameInitializer(const ONNX_NAMESPACE::TensorProto& metadata,
                       const ONNX_NAMESPACE::TensorProto& actual) const;
  M0WholeModelSealV1(const M0WholeModelSealV1&) = delete;
  M0WholeModelSealV1& operator=(const M0WholeModelSealV1&) = delete;
  ONNX_NAMESPACE::ModelProto proto_;
  std::unique_ptr<Snapshot> snapshot_;
  InlinedHashMap<std::string, std::string_view> raw_initializers_;
  bool live_transferred_ = false;
};
}  // namespace onnxruntime
