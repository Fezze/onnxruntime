// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <atomic>

namespace onnxruntime {

// Denial only. This value carries no authority to permit model work.
class M0CallbackPolicyV1 final {
 public:
  enum class Channel {
    kConstruction,
    kResolve,
    kRecursiveInference,
    kSchemaInference,
    kInputData,
    kAttribute,
    kOnnxPropagation,
    kOrtPropagation,
    kFunction
  };

  explicit M0CallbackPolicyV1(bool deny_requested = false) noexcept : deny_requested_(deny_requested) {}
  M0CallbackPolicyV1(const M0CallbackPolicyV1&) = delete;
  M0CallbackPolicyV1& operator=(const M0CallbackPolicyV1&) = delete;
  M0CallbackPolicyV1(M0CallbackPolicyV1&&) = delete;
  M0CallbackPolicyV1& operator=(M0CallbackPolicyV1&&) = delete;

  bool Deny(Channel) const noexcept {
    if (!deny_requested_) return false;
    poisoned_.store(true, std::memory_order_relaxed);
    return true;
  }

  bool IsPoisoned() const noexcept { return poisoned_.load(std::memory_order_relaxed); }

 private:
  const bool deny_requested_;
  mutable std::atomic<bool> poisoned_{false};
};
}  // namespace onnxruntime
