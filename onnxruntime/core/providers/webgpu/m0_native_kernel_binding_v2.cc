// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/session/plugin_ep/m0_native_kernel_binding_v2.h"

#if defined(__ANDROID__) && defined(ORT_USE_EP_API_ADAPTERS) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
namespace onnxruntime::webgpu {
extern "C" ORT_EXPORT int ORT_API_CALL OrtM0WebGpuNativeCreatorBindingV2(
    uint32_t abi, uint32_t record_id, M0NativeProviderKernelBindingV2* output) noexcept {
  if (abi != 2 || !output || !M0NativeKernelRecordByIdV2(record_id)) return 0;
  try {
    // Factory initialization precedes this authenticated query. Registration
    // metadata only: no capability query, GPU allocation or creator invocation.
    const auto registry = GetKernelRegistry(false, true);
    M0NativeProviderKernelBindingV2 binding;
    if (!registry || !registry->M0NativeBindingV2(record_id, binding)) return 0;
    *output = binding;
    return 1;
  } catch (...) {
    return 0;
  }
}
}  // namespace onnxruntime::webgpu
#endif
