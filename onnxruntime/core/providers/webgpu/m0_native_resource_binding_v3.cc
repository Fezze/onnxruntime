// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/plugin_ep/m0_native_resource_binding_v3.h"
#if defined(__ANDROID__) && defined(ORT_USE_EP_API_ADAPTERS) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#include "core/providers/webgpu/ep/ep.h"
#include "core/providers/webgpu/webgpu_context.h"
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
   M0FullGpuAllocationIdentity allocation;
   // The caller holds the actual Tensor allocation until this query finishes.
   // A foreign/untracked handle is never passed to a Dawn buffer operation.
   if(!required_bytes||!context.M0OwnershipV2().Describe(handle,allocation)||!allocation.generation||
      required_bytes>allocation.requested_bytes||allocation.requested_bytes>pending.max_buffer_bytes||
      allocation.requested_bytes>pending.max_binding_bytes)return false;
   const auto buffer=reinterpret_cast<WGPUBuffer>(handle);
   const auto backing=wgpuBufferGetSize(buffer);const auto usage=wgpuBufferGetUsage(buffer);
   if(backing<allocation.requested_bytes||backing>pending.max_buffer_bytes||backing>pending.max_binding_bytes||
      !(usage&WGPUBufferUsage_Storage))return false;
   pending.handle=handle;pending.generation=allocation.generation;
   pending.requested_bytes=allocation.requested_bytes;pending.backing_bytes=backing;
   pending.session=allocation.session;pending.run=allocation.run;pending.data_handle=allocation.data_handle;
   pending.data_generation=allocation.data_generation;pending.producer_role=allocation.producer_role;
   pending.producer_instance=allocation.producer_instance;
   pending.storage=1;pending.copy_source=uint32_t(bool(usage&WGPUBufferUsage_CopySrc));
   pending.copy_destination=uint32_t(bool(usage&WGPUBufferUsage_CopyDst));
   M0FullGpuAllocationIdentity current;
   if(!context.M0OwnershipV2().Describe(handle,current)||current.generation!=allocation.generation||
      current.requested_bytes!=allocation.requested_bytes||current.session!=allocation.session||
      current.run!=allocation.run||current.data_handle!=allocation.data_handle||
      current.data_generation!=allocation.data_generation||current.producer_role!=allocation.producer_role||
      current.producer_instance!=allocation.producer_instance)return false;
  }
  if(!context.M0NativeResourceStatusV3().IsOK())return false;
  output=pending;return true;
 }catch(...){return false;}
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
