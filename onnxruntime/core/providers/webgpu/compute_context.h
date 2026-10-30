// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <utility>

#include "core/providers/webgpu/webgpu_external_header.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_provider.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class Tensor;

namespace webgpu {

class WebGpuContext;
class BufferManager;

//
// Class ComputeContextBase is designed to provide basic context information
// for running a compute shader program.
//
// An instance of ComputeContextBase does not depend on OpKernelContext, which needs an execution frame to be created.
//
class ComputeContextBase {
 public:
  // Nested accessor class to provide controlled access to BufferManager
  class BufferManagerAccessor {
    // access to BufferManager is limited to class WebGpuContext.
    // This ensures no access to BufferManager from other classes, avoiding
    // potential misuse.
    friend class WebGpuContext;

   private:
    static const webgpu::BufferManager& Get(const ComputeContextBase& context);
    static CommandRecordingState& GetRecording(const ComputeContextBase& context);
  };

  ComputeContextBase(WebGpuContext& webgpu_context,
                     const WebGpuExecutionProvider& ep,
                     const OpKernel& op_kernel);

  virtual ~ComputeContextBase() = default;

  //
  // Get the node name.
  //
  const OpKernel& Kernel() const noexcept { return op_kernel_; }

  inline decltype(auto) NodeName() const {
    return op_kernel_.Node().Name();
  }

  //
  // Get the operator type.
  //
  inline decltype(auto) OpType() const {
    return op_kernel_.Node().OpType();
  }

  //
  // Get various information from the WebGPU context.
  //

  inline const wgpu::AdapterInfo& AdapterInfo() const {
    return webgpu_context_.AdapterInfo();
  }
  inline const wgpu::Limits& DeviceLimits() const {
    return webgpu_context_.DeviceLimits();
  }
  uint32_t M0ResidualBlockChoiceV1(const std::string& key) const {
    return webgpu_context_.M0ResidualBlockChoiceV1(key);
  }
  void M0RememberResidualBlockV1(const std::string& key,uint32_t block) {
    webgpu_context_.M0RememberResidualBlockV1(key,block);
  }
  inline bool HasFeature(wgpu::FeatureName feature) const {
    return webgpu_context_.DeviceHasFeature(feature);
  }
  inline const wgpu::AdapterPropertiesSubgroupMatrixConfigs& SubgroupMatrixConfigs() const {
    return webgpu_context_.SubgroupMatrixConfigs();
  }

  //
  // Get Split-K configuration.
  //
  inline const SplitKConfig& GetSplitKConfig() const {
    return webgpu_context_.GetSplitKConfig();
  }

  //
  // Get whether graph capture is enabled.
  //
  inline bool IsGraphCaptureEnabled() const {
    return ep_.IsGraphCaptureEnabled();
  }
  // Stable per-Run admission policy; this grants no allocation or enqueue authority.
  inline uint64_t M0SubmissionBudgetBytesV1() const {
    return ep_.Recording().m0_submission_budget_bytes;
  }

  //
  // Get the multi rotary cache concatenation offset (0 = disabled).
  //
  inline uint32_t MultiRotaryCacheConcatOffset() const {
    return ep_.MultiRotaryCacheConcatOffset();
  }

  //
  // Get the KV cache quantization bit width (0 = disabled, 4 = TurboQuant, 8 = symmetric block quantization).
  //
  inline uint32_t KvCacheQuantizationBits() const {
    return ep_.KvCacheQuantizationBits();
  }

  //
  // Get whether KV cache quantization is enabled.
  //
  inline bool KvCacheQuantizationEnabled() const {
    return ep_.KvCacheQuantizationEnabled();
  }

  //
  // Get whether MatMulNBits dot products accumulate in f32 rather than in the output element type.
  //
  inline bool EnableMatmulFp32Accumulation() const {
    return ep_.EnableMatmulFp32Accumulation();
  }

  //
  // Get the logger.
  //
#if defined(ORT_USE_EP_API_ADAPTERS)
  inline const onnxruntime::ep::adapter::Logger& Logger() const {
    return ep_.GetEpLogger();
  }
#else
  inline const logging::Logger& Logger() const {
    return *ep_.GetLogger();
  }
#endif

  //
  // Run a compute shader program.
  //
  inline Status RunProgram(const ProgramBase& program, uint64_t scheduled_products=0) {
    ORT_RETURN_IF_ERROR(M0CheckCancellationV1());
    return webgpu_context_.Run(*this, program, scheduled_products);
  }

  // Complete a related bounded group (for example pack+matrix) with one fence.
  inline Status RunProgramWithQueuedCompletion(const ProgramBase& program,uint64_t scheduled_products=0){
    ORT_RETURN_IF_ERROR(M0CheckCancellationV1());
    return webgpu_context_.RunWithQueuedCompletion(*this,program,scheduled_products);
  }

  // Isolate a costly dispatch from preceding and subsequent submissions.
  inline Status RunProgramWithCheckedCompletion(const ProgramBase& program, uint64_t scheduled_products = 0) {
    ORT_RETURN_IF_ERROR(M0CheckCancellationV1());
    return webgpu_context_.RunWithCheckedCompletion(*this, program, scheduled_products);
  }

 protected:
  virtual Status M0CheckCancellationV1() const { return Status::OK(); }
  WebGpuContext& webgpu_context_;
  const WebGpuExecutionProvider& ep_;
  const OpKernel& op_kernel_;
};

//
// Class ComputeContext provides all information a `ComputeContextBase` provides, and also
// access to `OpKernelContext` for input and output tensors.
//
class ComputeContext final : public ComputeContextBase {
 public:
  ComputeContext(WebGpuContext& webgpu_context,
                 const WebGpuExecutionProvider& ep,
                 const OpKernel& op_kernel,
                 OpKernelContext& kernel_context);

  ~ComputeContext() = default;
  Status M0CheckCancellationV1() const override;

  //
  // Get the kernel context.
  //
  inline OpKernelContext& KernelContext() {
    return kernel_context_;
  }

  uint32_t M0FullGpuFaultModeV2() const { return ep_.M0FullGpuFaultModeV2(); }

  bool M0FullGpuRegisterTerminalFaultV2(const Tensor& packet) const {
    if (M0FullGpuFaultModeV2() != 20) return true;
    M0FullGpuAllocationIdentity id;
    return M0FullGpuDescribeResourceV2(packet, id) &&
      webgpu_context_.M0FaultHooksV2().RegisterTerminal(
        reinterpret_cast<uintptr_t>(packet.DataRaw()), id.generation,
        kernel_context_.M0FullGpuExtentOwnerEpochV1(), kernel_context_.M0FullGpuExtentRunIdV2());
  }

  // Opaque allocation administration only: these methods never map or inspect
  // tensor values. Device guards must validate the corresponding sidecar words.
  bool M0FullGpuDescribeResourceV2(const Tensor& tensor,
                                  M0FullGpuAllocationIdentity& out) const {
    if (tensor.Location().device.Type() != OrtDevice::GPU || !tensor.DataRaw()) return false;
    return webgpu_context_.M0OwnershipV2().Describe(
        reinterpret_cast<uintptr_t>(tensor.DataRaw()), out);
  }

  bool M0FullGpuDescribeStorageV1(const Tensor& tensor,M0FullGpuStorageAllocationV1& out) const {
    if(tensor.Location().device.Type()!=OrtDevice::GPU||!tensor.DataRaw()||tensor.SizeInBytes()==0)return false;
    return webgpu_context_.M0DescribeStorageAllocationV1(
        reinterpret_cast<uintptr_t>(tensor.DataRaw()),tensor.SizeInBytes(),out);
  }

  bool M0FullGpuDescribePairV2(const Tensor& sidecar, const Tensor& carrier,
                              M0FullGpuAllocationIdentity& out) const {
    if (sidecar.Location().device.Type() != OrtDevice::GPU ||
        carrier.Location().device.Type() != OrtDevice::GPU) return false;
    return webgpu_context_.M0OwnershipV2().DescribePair(
        reinterpret_cast<uintptr_t>(sidecar.DataRaw()),
        reinterpret_cast<uintptr_t>(carrier.DataRaw()), out);
  }

  bool M0FullGpuClaimExtentV2(const Tensor& sidecar, const Tensor& carrier,
                             uint32_t role, uint32_t producer_instance,
                             M0FullGpuAllocationIdentity& out) const {
    M0FullGpuAllocationIdentity sidecar_id, carrier_id;
    const uint64_t session = kernel_context_.M0FullGpuExtentOwnerEpochV1();
    const uint64_t run = kernel_context_.M0FullGpuExtentRunIdV2();
    if (!session || !run || !M0FullGpuDescribeResourceV2(sidecar, sidecar_id) ||
        !M0FullGpuDescribeResourceV2(carrier, carrier_id)) return false;
    auto& ownership = webgpu_context_.M0OwnershipV2();
    if (!ownership.Claim(reinterpret_cast<uintptr_t>(sidecar.DataRaw()),
                         sidecar_id.generation, session, run, role,
                         reinterpret_cast<uintptr_t>(carrier.DataRaw()),
                         carrier_id.generation, producer_instance)) return false;
    return M0FullGpuDescribePairV2(sidecar, carrier, out);
  }

  //
  // Get input tensor.
  //
  template <typename T = onnxruntime::Tensor>
  inline const T* Input(int index) const {
    return kernel_context_.Input<T>(index);
  }

  //
  // Get input count.
  //
  inline int InputCount() const {
    return kernel_context_.InputCount();
  }

  //
  // Set output tensor.
  //
  template <typename TensorShapeType>
  inline Tensor* Output(int index, TensorShapeType&& shape) {
    if (ep_.Recording().m0_submission_budget_bytes) M0PrepareAllocationV1(TensorShape(shape));
    return kernel_context_.Output(index, std::forward<TensorShapeType>(shape));
  }

  //
  // Get output count.
  //
  inline int OutputCount() const {
    return kernel_context_.OutputCount();
  }

  //
  // Create CPU tensor.
  //
  // This method creates a tensor of the given data type and shape, using the CPU allocator.
  // The tensor owns the underlying CPU memory buffer.
  //
  template <typename TensorShapeType>
  Tensor CreateCPUTensor(MLDataType data_type, TensorShapeType&& shape) {
    AllocatorPtr allocator;
    ORT_THROW_IF_ERROR(kernel_context_.GetTempSpaceCPUAllocator(&allocator));
    return {data_type, std::forward<TensorShapeType>(shape), allocator};
  }

  //
  // Create GPU tensor.
  //
  // This method creates a tensor of the given data type and shape, using the WebGPU allocator.
  // The tensor owns the underlying WebGPU storage buffer.
  // In the plugin, the temp-space allocator is the existing Session device allocator,
  // not an Env shared allocator or a new allocator created for each tensor.
  //
  Tensor CreateGPUTensor(MLDataType data_type, const TensorShape& shape);

  void M0PrepareAllocationV1(const TensorShape& shape) const;

  //
  // Copy data from a tensor to another tensor.
  //
  // This method assumes that both tensors have the same data size.
  //
  inline Status CopyTensor(const Tensor& src, Tensor& dst) {
    return op_kernel_.Info().GetDataTransferManager().CopyTensor(src, dst);
  }

  //
  // Fill a GPU tensor with zeros.
  //
  inline void FillZero(Tensor& dst) {
    auto& recording = ep_.Recording();
    if(recording.m0_submission_budget_bytes){
      const auto handle=reinterpret_cast<WGPUBuffer>(dst.MutableDataRaw());
      if(M0BatchHeavyProgramsV1()){
        const std::vector<webgpu::M0QueuedBufferBudgetV1::Reference> references{
            {reinterpret_cast<uintptr_t>(handle),wgpuBufferGetSize(handle)}};
        ORT_THROW_IF_ERROR(webgpu_context_.M0BeforeEnqueueUniqueV1(ep_.BufferManager(),recording,references));
      }else{
        ORT_THROW_IF_ERROR(webgpu_context_.M0BeforeEnqueueV1(ep_.BufferManager(),recording,wgpuBufferGetSize(handle)));
      }
    }
    ORT_THROW_IF_ERROR(webgpu_context_.EncodeDeferredDispatches(recording));
    webgpu_context_.EndComputePass(recording);
    auto& command_encoder = webgpu_context_.GetCommandEncoder(recording);
    WGPUBuffer buffer = reinterpret_cast<WGPUBuffer>(dst.MutableDataRaw());
    command_encoder.ClearBuffer(buffer, 0, dst.SizeInBytes());
  }

 private:
  OpKernelContext& kernel_context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
