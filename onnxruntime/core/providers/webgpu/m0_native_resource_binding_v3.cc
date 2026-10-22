// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/plugin_ep/m0_native_resource_binding_v3.h"
#if defined(__ANDROID__) && defined(ORT_USE_EP_API_ADAPTERS) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#include "core/providers/webgpu/ep/ep.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/buffer_manager.h"
#include <limits>
namespace onnxruntime::webgpu {
bool ep::Ep::M0NativeDescribeResourceV3(const OrtEp* candidate,uintptr_t handle,uint64_t required_bytes,
                                      M0NativeResourceBindingV3& output)noexcept{
 try{
  // The core owns candidate through this exact authenticated factory. Check the
  // concrete table before accessing the adapter's implementation/context.
  if(!candidate||candidate->GetName!=&GetNameImpl||candidate->GetKernelRegistry!=&GetKernelRegistryImpl||
     candidate->OnRunStart!=&OnRunStartImpl||candidate->OnRunEnd!=&OnRunEndImpl)return false;
  const auto* ep=static_cast<const Ep*>(candidate);
  const auto* implementation=ep->EpImpl();
  if(!implementation||implementation->Type()!=kWebGpuExecutionProvider)return false;
  auto& context=WebGpuContextFactory::GetContext(implementation->GetDeviceId());
  if(!context.M0NativeResourceStatusV3().IsOK())return false;
  const auto& queue=context.M0NativeResourceQueueV3();
  if(!queue)return false;
  M0NativeResourceBindingV3 pending;
  pending.abi=3;pending.structure_bytes=sizeof(pending);
  pending.ep=reinterpret_cast<uintptr_t>(candidate);pending.context=reinterpret_cast<uintptr_t>(&context);
  pending.device=reinterpret_cast<uintptr_t>(context.Device().Get());pending.queue=reinterpret_cast<uintptr_t>(queue.Get());
  pending.max_buffer_bytes=context.DeviceLimits().maxBufferSize;
  pending.max_binding_bytes=context.DeviceLimits().maxStorageBufferBindingSize;
  if(!pending.device||!pending.max_buffer_bytes||!pending.max_binding_bytes)return false;
  if(!handle){if(required_bytes)return false;}
  else{
   M0FullGpuStorageAllocationV1 storage;
   // The caller holds the actual Tensor allocation until this query finishes.
   // A foreign/untracked handle is never passed to a Dawn buffer operation.
   if(!context.M0DescribeStorageAllocationV1(handle,required_bytes,storage))return false;
   const auto& allocation=storage.identity;
   pending.handle=handle;pending.generation=allocation.generation;
   pending.requested_bytes=allocation.requested_bytes;pending.backing_bytes=storage.backing_bytes;
   pending.session=allocation.session;pending.run=allocation.run;pending.data_handle=allocation.data_handle;
   pending.data_generation=allocation.data_generation;pending.producer_role=allocation.producer_role;
   pending.producer_instance=allocation.producer_instance;
   pending.storage=storage.storage;pending.copy_source=storage.copy_source;
   pending.copy_destination=storage.copy_destination;
  }
  if(!context.M0NativeResourceStatusV3().IsOK())return false;
  output=pending;return true;
 }catch(...){return false;}
}
bool ep::Ep::M0NativeDescribeDeviceLimitsV1(const OrtEp* candidate,uint64_t storage_bytes,
 uint64_t initializer_bytes,uint64_t uniform_bytes,M0NativeDeviceLimitsV1& output) noexcept {
 try {
  if(!candidate||candidate->GetName!=&GetNameImpl||candidate->GetKernelRegistry!=&GetKernelRegistryImpl||
     candidate->OnRunStart!=&OnRunStartImpl||candidate->OnRunEnd!=&OnRunEndImpl)return false;
  const auto* ep=static_cast<const Ep*>(candidate);
  const auto* implementation=ep->EpImpl();
  if(!implementation||implementation->Type()!=kWebGpuExecutionProvider)return false;
  const auto* provider=static_cast<const WebGpuExecutionProvider*>(implementation);
  auto& context=WebGpuContextFactory::GetContext(implementation->GetDeviceId());
  if(!context.M0NativeResourceStatusV3().IsOK())return false;
  const auto& queue=context.M0NativeResourceQueueV3();
  if(!queue)return false;
  for(const auto request:{storage_bytes,initializer_bytes,uniform_bytes})
   if(request>std::numeric_limits<size_t>::max())return false;
  const auto& limits=context.DeviceLimits();
  M0NativeDeviceLimitsV1 pending;
  pending.abi=1;pending.structure_bytes=sizeof(pending);
  pending.ep=reinterpret_cast<uintptr_t>(candidate);pending.context=reinterpret_cast<uintptr_t>(&context);
  pending.device=reinterpret_cast<uintptr_t>(context.Device().Get());pending.queue=reinterpret_cast<uintptr_t>(queue.Get());
  pending.max_buffer_bytes=limits.maxBufferSize;pending.max_binding_bytes=limits.maxStorageBufferBindingSize;
  pending.max_uniform_binding_bytes=limits.maxUniformBufferBindingSize;
  pending.max_storage_buffers=limits.maxStorageBuffersPerShaderStage;pending.max_bindings=limits.maxBindingsPerBindGroup;
  pending.max_workgroup_x=limits.maxComputeWorkgroupSizeX;pending.max_workgroup_y=limits.maxComputeWorkgroupSizeY;
  pending.max_workgroup_z=limits.maxComputeWorkgroupSizeZ;pending.max_invocations=limits.maxComputeInvocationsPerWorkgroup;
  pending.max_workgroup_storage=limits.maxComputeWorkgroupStorageSize;pending.max_workgroups=limits.maxComputeWorkgroupsPerDimension;
  if(!pending.device||!pending.max_buffer_bytes||!pending.max_binding_bytes||!pending.max_uniform_binding_bytes||
     !pending.max_storage_buffers||!pending.max_bindings||!pending.max_workgroup_x||!pending.max_workgroup_y||
     !pending.max_workgroup_z||!pending.max_invocations||!pending.max_workgroup_storage||!pending.max_workgroups)return false;
  pending.storage_requested_bytes=storage_bytes;pending.initializer_requested_bytes=initializer_bytes;
  pending.uniform_requested_bytes=uniform_bytes;
  pending.storage_backing_bytes=provider->BufferManager().M0AllocationBackingSizeV1(size_t(storage_bytes));
  pending.initializer_backing_bytes=provider->InitializerBufferManager().M0AllocationBackingSizeV1(size_t(initializer_bytes));
  pending.uniform_backing_bytes=provider->BufferManager().M0AllocationBackingSizeV1(size_t(uniform_bytes),true);
  if(pending.storage_backing_bytes<storage_bytes||pending.initializer_backing_bytes<initializer_bytes||
     pending.uniform_backing_bytes<uniform_bytes||!context.M0NativeResourceStatusV3().IsOK())return false;
  output=pending;return true;
 }catch(...){return false;}
}
extern "C" ORT_EXPORT int ORT_API_CALL OrtM0WebGpuNativeDeviceLimitsV1(
 uint32_t abi,const OrtEp* actual_ep,uint64_t storage_bytes,uint64_t initializer_bytes,uint64_t uniform_bytes,
 M0NativeDeviceLimitsV1* output) noexcept {
 if(abi!=1||!output)return 0;
 M0NativeDeviceLimitsV1 pending;
 if(!ep::Ep::M0NativeDescribeDeviceLimitsV1(actual_ep,storage_bytes,initializer_bytes,uniform_bytes,pending))return 0;
 *output=pending;return 1;
}
extern "C" ORT_EXPORT int ORT_API_CALL OrtM0WebGpuNativeResourceBindingV3(
 uint32_t abi,const OrtEp* actual_ep,uintptr_t handle,uint64_t required_bytes,M0NativeResourceBindingV3* output)noexcept{
 if(abi!=3||!output)return 0;
 M0NativeResourceBindingV3 pending;
 if(!ep::Ep::M0NativeDescribeResourceV3(actual_ep,handle,required_bytes,pending))return 0;
 *output=pending;return 1;
}
}  // namespace onnxruntime::webgpu
#endif
