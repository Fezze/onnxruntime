// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/compute_context.h"
#include <limits>
#include "core/framework/tensor.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {
namespace webgpu {

ComputeContextBase::ComputeContextBase(WebGpuContext& webgpu_context,
                                       const WebGpuExecutionProvider& ep,
                                       const OpKernel& op_kernel)
    : webgpu_context_{webgpu_context},
      ep_{ep},
      op_kernel_{op_kernel} {
}

const webgpu::BufferManager& ComputeContextBase::BufferManagerAccessor::Get(const ComputeContextBase& context) {
  return context.ep_.BufferManager();
}

CommandRecordingState& ComputeContextBase::BufferManagerAccessor::GetRecording(const ComputeContextBase& context) {
  return context.ep_.Recording();
}

ComputeContext::ComputeContext(WebGpuContext& webgpu_context,
                               const WebGpuExecutionProvider& ep,
                               const OpKernel& op_kernel,
                               OpKernelContext& kernel_context)
    : ComputeContextBase(webgpu_context, ep, op_kernel),
      kernel_context_{kernel_context} {
}

Status ComputeContext::M0CheckCancellationV1() const {
  if(!ep_.Recording().m0_submission_budget_bytes)return Status::OK();
  uint32_t cancelled=0;
  ORT_RETURN_IF_NOT(kernel_context_.M0ReadCancellationV1(cancelled)&&cancelled<=1,
                    "RW07_NATIVE_RUN_CANCELLATION_BRIDGE_UNAVAILABLE");
  ORT_RETURN_IF(cancelled,"RW07_NATIVE_RUN_CANCELLED_BEFORE_DISPATCH");
  return Status::OK();
}

void ComputeContext::M0PrepareAllocationV1(const TensorShape& shape) const {
  auto& recording = ep_.Recording();
  if (!recording.m0_submission_budget_bytes) return;
  const int64_t elements = shape.Size();
  ORT_ENFORCE(elements >= 0 && uint64_t(elements) <= std::numeric_limits<uint64_t>::max() / 8,
              "RW07 native allocation shape overflow");
  // The closed native profile uses elements no wider than eight bytes. This
  // is only a conservative drain hint: the allocator admits the exact dtype
  // and cache-rounded backing afterwards, including streamless Output calls.
  ORT_THROW_IF_ERROR(webgpu_context_.M0DrainOnMemoryPressureV1(
      ep_.BufferManager(), recording, uint64_t(elements) * 8));
}

// Native test targets also include compute_context.h but do not use the EP adapter types.
Tensor ComputeContext::CreateGPUTensor(MLDataType data_type, const TensorShape& shape) {
  M0PrepareAllocationV1(shape);
  AllocatorPtr allocator;
  ORT_THROW_IF_ERROR(kernel_context_.GetTempSpaceAllocator(&allocator));
#if defined(ORT_USE_EP_API_ADAPTERS)
  const size_t bytes = Tensor::CalculateTensorStorageSize(data_type, shape);
  // For performance, use the kernel's stream: plain Alloc during Run immediately flushes cached-buffer
  // clears, causing frequent submissions when scratch allocations reuse cached buffers.
  // A null stream still falls back to plain Alloc's immediate-submission policy.
  auto buffer = IAllocator::MakeUniquePtr<void>(
      allocator, bytes, false, reinterpret_cast<Stream*>(kernel_context_.GetSyncStream()));
  Tensor tensor(data_type, shape, buffer.get(), allocator);
  buffer.release();
  return tensor;
#else
  return {data_type, shape, allocator};
#endif
}

}  // namespace webgpu
}  // namespace onnxruntime
