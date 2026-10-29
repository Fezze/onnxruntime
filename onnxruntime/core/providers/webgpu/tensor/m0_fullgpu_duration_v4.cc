// rw07-shared-extent-ownership-v2; DurationExtentV4 arithmetic unchanged from V3; original v1 is preserved. NOT a full-model admission gate.
// The token descriptor is the original input-time authority. No GPU duration,
// prefix, shape payload or intermediate extent is read by this host code.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "m0_fullgpu_duration_programs_v5.h"
#include <limits>
#include <algorithm>

namespace onnxruntime::webgpu {
namespace {
ProgramOutput Words(Tensor* t, uint32_t words) {
  ProgramOutput p(t, ProgramTensorMetadataDependency::None, TensorShape{words}, 1);
  p.var_type = ProgramVariableDataType::Uint32; return p;
}
bool Device(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
void Trace(ComputeContext& c, const char* role, const Tensor* t, uint64_t epoch) {
  LOGS(c.Logger(), INFO) << "M0_RW07_RESOURCE role=" << role << " epoch=" << epoch
      << " device=" << int(t->Location().device.Type()) << " handle=" << t->DataRaw()
      << " physical_bytes=" << t->SizeInBytes()
      << " allocation_bytes=" << (t->SizeInBytes() ? wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw()))) : 0);
}

using m0_duration_detail::DurationProgram;
using m0_duration_detail::DurationRowsProgram;

class DurationExtent final : public WebGpuKernel {
 public:
  explicit DurationExtent(const OpKernelInfo& i) : WebGpuKernel{i}, capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity", 0)),
      role_(i.GetAttrOrDefault<int64_t>("semantic_role", -1)),
      instance_(i.GetAttrOrDefault<int64_t>("producer_instance", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    ORT_RETURN_IF_NOT(role_ == 1 && instance_ == 1001, "RW07 duration certified producer identity");
    const auto* tokens = c.Input<Tensor>(0); const auto* logits = c.Input<Tensor>(1); const auto* speed = c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(tokens && tokens->IsDataType<int64_t>() && tokens->Shape().NumDimensions() == 2 && tokens->Shape()[0] == 1 &&
                      logits && logits->IsDataType<float>() && logits->Shape().NumDimensions() == 3 && logits->Shape()[0] == 1 && logits->Shape()[2] == 50 &&
                      speed && speed->IsDataType<float>() && speed->Shape() == TensorShape{1}, "RW07 duration exact descriptor contract");
    const int64_t t = tokens->Shape()[1];
    ORT_RETURN_IF_NOT(t >= 0 && t <= 512 && logits->Shape()[1] >= t && logits->Shape()[1] <= 1024 && capacity_ > 0 && capacity_ <= 102400,
                      "RW07 duration physical descriptor domain");
    const uint64_t epoch = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(epoch != 0, "RW07 private owner interface unavailable");
    ORT_RETURN_IF_NOT(Device(tokens) && Device(logits) && Device(speed), "RW07 duration input residency");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 6 && limits.maxBufferSize >= m0_extent_abi2::Words * 4 &&
                      limits.maxStorageBufferBindingSize >= m0_extent_abi2::Words * 4,
                      "RW07 duration actual storage limits");
    for (const auto* input : {logits, speed})
      ORT_RETURN_IF_NOT(input->SizeInBytes() <= limits.maxStorageBufferBindingSize && input->SizeInBytes() <= limits.maxBufferSize,
                        "RW07 duration actual input buffer limit");
    ORT_RETURN_IF_NOT(uint64_t(t) * 8 <= limits.maxStorageBufferBindingSize && uint64_t(t) * 8 <= limits.maxBufferSize,
                      "RW07 duration actual output buffer limit");
    auto* durations = c.Output(0, TensorShape{1,t}); auto* prefix = c.Output(1, TensorShape{t});
    auto* total = c.Output(2, TensorShape{}); auto* extent = c.Output(3, TensorShape{m0_extent_abi2::Words});
    ORT_RETURN_IF_NOT(durations->IsDataType<int64_t>() && prefix->IsDataType<int64_t>() && total->IsDataType<int64_t>() && extent->IsDataType<uint32_t>() &&
                      Device(durations) && Device(prefix) && Device(total) && Device(extent), "RW07 duration output role");
    // Empty original outputs have no binding buffer. Independently owned dummy
    // resources let the error shader run; no empty success/output is published.
    Tensor empty_duration, empty_prefix, empty_logits;
    if (t == 0) {
      empty_duration = c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{2});
      empty_prefix = c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{2});
    }
    if (logits->Shape().Size() == 0)
      empty_logits = c.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape{1});
    LOGS(c.Logger(), INFO) << "M0_RW07_DURATION_BEGIN epoch=" << epoch << " initial_token_descriptor=" << t << " physical_capacity=" << capacity_
        << " adapter=" << std::string_view(c.AdapterInfo().device) << " backend=" << int(c.AdapterInfo().backendType) << " type=" << int(c.AdapterInfo().adapterType)
        << " storage_buffers_limit=" << limits.maxStorageBuffersPerShaderStage << " binding_bytes_limit=" << limits.maxStorageBufferBindingSize;
    Trace(c, "tokens-initial", tokens, epoch); Trace(c, "logits-upload", logits, epoch); Trace(c, "speed-upload", speed, epoch);
    Trace(c, "duration-device-only", durations, epoch); Trace(c, "prefix-device-only", prefix, epoch); Trace(c, "total-device-only", total, epoch); Trace(c, "extent-device-only", extent, epoch);
    M0FullGpuAllocationIdentity output_identity;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*extent, *total, 1, 1001, output_identity),
                      "RW07 duration opaque allocation claim unavailable");
    // Bind the actual prefix allocation to this same producer event and total.
    // Numerical prefix checks cannot establish allocation provenance.
    M0FullGpuAllocationIdentity prefix_identity;
    ORT_RETURN_IF_NOT(t == 0 || c.M0FullGpuClaimExtentV2(*prefix, *total, 1, 1001, prefix_identity),
                      "RW07 duration prefix allocation claim unavailable");
    const uint64_t run = output_identity.run;
    const uint64_t sidecar_generation = output_identity.generation;
    const uint64_t carrier_generation = output_identity.data_generation;
    LOGS(c.Logger(), INFO) << "M0_RW07_ABI2_PRODUCER role=1 instance=1001 run=" << run
      << " sidecar_generation=" << sidecar_generation << " carrier_generation=" << carrier_generation;
    auto rows=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{std::max<int64_t>(t,1),2});
    if(t>0){
      const uint32_t lanes=std::min({uint32_t(t),std::max(1u,c.AdapterInfo().subgroupMaxSize),
          limits.maxComputeWorkgroupSizeX,limits.maxComputeInvocationsPerWorkgroup});
      ORT_RETURN_IF_NOT(lanes>0,"RW07 duration token workgroup support");
      DurationRowsProgram values;
      values.AddInputs({{logits,ProgramTensorMetadataDependency::None},{speed,ProgramTensorMetadataDependency::None}})
        .AddOutput({&rows,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(lanes)
        .SetDispatchGroupSize((uint32_t(t)+lanes-1)/lanes).AddUniformVariables({uint32_t(t)});
      ORT_RETURN_IF_ERROR(c.RunProgram(values));
    }
    DurationProgram p;
    p.AddInput({&rows,ProgramTensorMetadataDependency::None})
        .AddOutputs({Words(t ? durations : &empty_duration, uint32_t(t ? t*2 : 2)), Words(t ? prefix : &empty_prefix, uint32_t(t ? t*2 : 2)), Words(total,2), Words(extent,m0_extent_abi2::Words)})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(t), uint32_t(capacity_), uint32_t(epoch), uint32_t(epoch >> 32),
          uint32_t(run), uint32_t(run >> 32), uint32_t(sidecar_generation), uint32_t(sidecar_generation >> 32),
          uint32_t(carrier_generation), uint32_t(carrier_generation >> 32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(p));
    return Status::OK();
  }
 private: int64_t capacity_, role_, instance_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuDurationExtentV4KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out)->Status {
    out = std::make_unique<DurationExtent>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("DurationExtentV4").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
} // namespace onnxruntime::webgpu
