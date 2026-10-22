// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <type_traits>
#include "core/session/onnxruntime_c_api.h"
namespace onnxruntime {
// Private metadata from the SAME authenticated retained EP/factory. Cache
// sizing does not acquire memory, grant execution authority, or promise available
// residency. Over-limit backing sizes are observations for the caller to reject.
struct M0NativeDeviceLimitsV1 {
  uint32_t abi=0,structure_bytes=0;
  uintptr_t ep=0,context=0,device=0,queue=0;
  uint64_t max_buffer_bytes=0,max_binding_bytes=0,max_uniform_binding_bytes=0;
  uint32_t max_storage_buffers=0,max_bindings=0;
  uint32_t max_workgroup_x=0,max_workgroup_y=0,max_workgroup_z=0;
  uint32_t max_invocations=0,max_workgroup_storage=0,max_workgroups=0;
  uint64_t storage_requested_bytes=0,storage_backing_bytes=0;
  uint64_t initializer_requested_bytes=0,initializer_backing_bytes=0;
  uint64_t uniform_requested_bytes=0,uniform_backing_bytes=0;
};
static_assert(std::is_standard_layout_v<M0NativeDeviceLimitsV1>);
static_assert(std::is_trivially_copyable_v<M0NativeDeviceLimitsV1>);
// Unknown ABI, overflow or stale/foreign context leaves output unchanged.
using M0WebGpuNativeDeviceLimitsQueryV1=int(ORT_API_CALL*)(uint32_t abi,const OrtEp* actual_ep,
    uint64_t storage_bytes,uint64_t initializer_bytes,uint64_t uniform_bytes,
    M0NativeDeviceLimitsV1* output) noexcept;
}  // namespace onnxruntime
