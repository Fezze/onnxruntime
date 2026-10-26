// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/program.h"
#include "core/common/m0_observation_options_v1.h"
#if defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/ep/ep.h"
#endif

namespace onnxruntime {
namespace webgpu {

WebGpuKernel::WebGpuKernel(const OpKernelInfo& info)
    : OpKernel(info),
      ep_(*static_cast<const WebGpuExecutionProvider*>(info.GetExecutionProvider())),
      webgpu_context_(WebGpuContextFactory::GetContext(ep_.GetDeviceId())) {
}

Status WebGpuKernel::Compute(OpKernelContext* p_op_kernel_context) const {
  ComputeContext context{webgpu_context_,
                         ep_,
                         *this,
                         *p_op_kernel_context};

  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    webgpu_context_.PushErrorScope();
  }

  webgpu_context_.M0OwnershipV2().RecordWorkload(M0WorkloadEventV1::OperatorStarted);
  Status s = ComputeInternal(context);
#if defined(ORT_USE_EP_API_ADAPTERS)
  if (s.IsOK() && M0GpuNumericTraceV1()) {
    const auto* bridge = static_cast<const ep::Ep*>(Info().GetOrtEp());
    s = bridge->TraceNumericOutputs(context);
  }
#endif

  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    ORT_RETURN_IF_ERROR(webgpu_context_.PopErrorScope());
  }

  if(s.IsOK()){
    const auto ordinal=webgpu_context_.M0OwnershipV2().RecordWorkload(M0WorkloadEventV1::OperatorCompleted);
    if(ordinal%64==0)webgpu_context_.M0OwnershipV2().LogWorkload("operator_completed");
  }
  return s;
}

Status WebGpuKernel::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr /*alloc*/,
                             /*out*/ bool& is_packed, /*out*/ PrePackedWeights* /* prepacked_weights */) {
  ComputeContextBase context{webgpu_context_, ep_, *this};

  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    webgpu_context_.PushErrorScope();
  }

  // Currently, ORT does not allow using prepacked weights in non-CPU EPs.
  // So we do not pass prepacked_weights to PrePackInternal.
  // Kernel implementation that supports prepacking should manage its own storage.
  // Use the EP's prepack allocator which creates unmapped GPU buffers.

  Status s = PrePackInternal(context, tensor, input_idx, ep_.PrepackAllocator(), is_packed);

  if (s.IsOK() && is_packed) {
    // PrePack has no OnRunEnd hook, so finish its deferred pipeline builds and submit any encoded
    // GPU work before returning and allowing ORT to release the original initializer tensor.
    s = webgpu_context_.Flush(ep_.InitializerBufferManager(), ep_.Recording());
  }

  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    ORT_RETURN_IF_ERROR(webgpu_context_.PopErrorScope());
  }

  return s;
}

Status WebGpuKernel::PrePackInternal(ComputeContextBase& /*context*/,
                                     const Tensor& /*tensor*/,
                                     int /*input_idx*/,
                                     AllocatorPtr /*alloc*/,
                                     /*out*/ bool& is_packed) {
  is_packed = false;
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
