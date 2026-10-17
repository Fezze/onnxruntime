// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <type_traits>

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// Private metadata ABI, not an OrtApi entry or an admission decision. Both pointers are borrowed;
// the owning EP module and registry must remain alive while the binding is retained or compared.
struct M0PluginKernelCreatorBindingV1 {
  OrtKernelCreateFunc callback = nullptr;
  void* state = nullptr;
};

static_assert(std::is_standard_layout_v<M0PluginKernelCreatorBindingV1>);
static_assert(std::is_trivially_copyable_v<M0PluginKernelCreatorBindingV1>);

// Resolve only from the independently authenticated, retained WebGPU module handle.
// ABI 1 returns 1 on success, 0 on refusal, with *output unchanged on refusal.
using M0WebGpuLstm14CreatorBindingQueryV1 = int(ORT_API_CALL*)(
    uint32_t abi, M0PluginKernelCreatorBindingV1* output) noexcept;

}  // namespace onnxruntime
