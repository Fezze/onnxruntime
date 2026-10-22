// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
class M0OriginalPreloadOwnerV1;

// The private factory authenticates owned bytes before retaining the parsed model.
class M0WholeModelSealV1 final {
 private:
  friend class M0OriginalPreloadOwnerV1;
  friend struct std::default_delete<M0WholeModelSealV1>;
  static std::unique_ptr<M0WholeModelSealV1> ValidateAndParse(std::string_view bytes);
  explicit M0WholeModelSealV1(ONNX_NAMESPACE::ModelProto&& proto)
      : proto_(std::move(proto)) {}
  ~M0WholeModelSealV1() = default;
  M0WholeModelSealV1(const M0WholeModelSealV1&) = delete;
  M0WholeModelSealV1& operator=(const M0WholeModelSealV1&) = delete;
  ONNX_NAMESPACE::ModelProto proto_;
};
}  // namespace onnxruntime
