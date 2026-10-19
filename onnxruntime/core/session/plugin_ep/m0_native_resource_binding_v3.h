// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <type_traits>
#include "core/session/onnxruntime_ep_c_api.h"
namespace onnxruntime {
// Source-private, payload-free metadata from the authenticated retained EP.
// Context queries use handle=0/required_bytes=0. Resource queries require a live
// allocation in that exact context before the provider may inspect its buffer.
// The actual OrtEp is retained by the private native module, never user supplied.
// This does not authorize an upload, constructor, dispatch, readback or Run.
struct M0NativeResourceBindingV3 {
  uint32_t abi=0,structure_bytes=0;
  uintptr_t ep=0,context=0,device=0,queue=0,handle=0;
  uint64_t generation=0,requested_bytes=0,backing_bytes=0;
  uint64_t max_buffer_bytes=0,max_binding_bytes=0;
  uint64_t session=0,run=0,data_generation=0;
  uintptr_t data_handle=0;
  uint32_t producer_role=0,producer_instance=0;
  uint32_t storage=0,copy_source=0,copy_destination=0;
};
static_assert(std::is_standard_layout_v<M0NativeResourceBindingV3>);
static_assert(std::is_trivially_copyable_v<M0NativeResourceBindingV3>);
// Refusal returns0 and leaves output unchanged. Resolve from the independently
// authenticated SAME retained module handle/factory as its ABI2 registry.
using M0WebGpuNativeResourceQueryV3=int(ORT_API_CALL*)(uint32_t abi,const OrtEp* ep,
    uintptr_t handle,uint64_t required_bytes,M0NativeResourceBindingV3* output) noexcept;
}  // namespace onnxruntime
