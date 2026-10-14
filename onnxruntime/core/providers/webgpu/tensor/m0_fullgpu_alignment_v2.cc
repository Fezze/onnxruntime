// Capacity V2: allocation-only 2D dispatch; original numerical/extent semantics retained.
// rw07-fullgpu-v1 representative original alignment MatMul1330/1331.
// Capacity governs allocation/dispatch; original logical D is GPU-owned.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"

namespace onnxruntime::webgpu {
namespace {
ProgramInput InWords(const Tensor* t, uint32_t words) {
  ProgramInput p(t, ProgramTensorMetadataDependency::None, TensorShape{words}, 1);
  p.var_type = ProgramVariableDataType::Uint32; return p;
}
ProgramOutput OutWords(Tensor* t, uint32_t words) {
  ProgramOutput p(t, ProgramTensorMetadataDependency::None, TensorShape{words}, 1);
  p.var_type = ProgramVariableDataType::Uint32; return p;
}
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }

class AlignmentExtentGuard final : public Program<AlignmentExtentGuard> {
 public:
  AlignmentExtentGuard() : Program{"RW07V2AlignmentExtentGuard"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"tokens", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"epoch_low", ProgramUniformVariableDataType::Uint32},
      {"epoch_high", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& source = s.AddInput("source_extent", ShaderUsage::UseValueTypeAlias);
    const auto& prefixes = s.AddInput("prefixes", ShaderUsage::UseValueTypeAlias);
    const auto& extent = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<24u;i++){" << extent.SetByOffset("i", "0u") << "}"
      << extent.SetByOffset("0u", "1u") << extent.SetByOffset("1u", "3u")
      << extent.SetByOffset("22u", "uniforms.epoch_low") << extent.SetByOffset("23u", "uniforms.epoch_high")
      << "if(" << source.GetByOffset("0u") << "!=1u||" << source.GetByOffset("1u") << "!=1u||"
      << source.GetByOffset("22u") << "!=uniforms.epoch_low||" << source.GetByOffset("23u") << "!=uniforms.epoch_high){"
      << extent.SetByOffset("20u", "103u") << "return;}"
      << "let prior_status=" << source.GetByOffset("20u") << ";if(prior_status!=0u){"
      << extent.SetByOffset("20u", "prior_status") << "return;}"
      << "let d=" << source.GetByOffset("2u") << ";if((" << source.GetByOffset("21u") << "&63u)!=63u||"
      << source.GetByOffset("3u") << "!=0u||" << source.GetByOffset("18u") << "!=d||"
      << source.GetByOffset("19u") << "!=0u||d==0u||uniforms.tokens==0u){"
      << extent.SetByOffset("20u", "103u") << "return;}"
      << "if(d>uniforms.capacity){" << extent.SetByOffset("20u", "102u") << "return;}"
      // Validate paired INT64 prefix and positive partition before any model
      // value access. Corruption/ownership failure is never a CPU semantic error.
      << "var previous=0u;for(var t=0u;t<uniforms.tokens;t++){let next=" << prefixes.GetByOffset("2u*t") << ";"
      << "if(" << prefixes.GetByOffset("2u*t+1u") << "!=0u||next<=previous||next>d){"
      << extent.SetByOffset("20u", "103u") << "return;}previous=next;}"
      << "if(previous!=d){" << extent.SetByOffset("20u", "103u") << "return;}"
      << extent.SetByOffset("2u", "1u") << extent.SetByOffset("4u", "uniforms.channels")
      << extent.SetByOffset("6u", "d") << extent.SetByOffset("18u", "uniforms.channels*d")
      << extent.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class AlignmentData final : public Program<AlignmentData> {
 public:
  AlignmentData() : Program{"RW07V2Alignment"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"tokens", ProgramUniformVariableDataType::Uint32},
      {"source_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"epoch_low", ProgramUniformVariableDataType::Uint32},
      {"epoch_high", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& prefixes = s.AddInput("prefixes", ShaderUsage::UseValueTypeAlias);
    const auto& extent = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels*uniforms.capacity){return;}"
      << y.SetByOffset("global_idx", "0.0")
      << "if(" << extent.GetByOffset("20u") << "!=0u||" << extent.GetByOffset("0u") << "!=1u||"
      << extent.GetByOffset("1u") << "!=3u||(" << extent.GetByOffset("21u") << "&63u)!=63u||"
      << extent.GetByOffset("22u") << "!=uniforms.epoch_low||" << extent.GetByOffset("23u") << "!=uniforms.epoch_high){return;}"
      << "let channel=global_idx/uniforms.capacity;let frame=global_idx%uniforms.capacity;let d=" << extent.GetByOffset("6u") << ";"
      << "if(frame>=d){return;}var value=0.0;var previous=0u;"
      // Keep the exact MatMul reduction rather than dropping zero terms. This
      // preserves logical NaN propagation; poisoned physical tails never enter.
      << "for(var t=0u;t<uniforms.tokens;t++){let next=" << prefixes.GetByOffset("2u*t") << ";"
      << "let mask=select(0.0,1.0,frame>=previous&&frame<next);"
      << "value+=" << x.GetByOffset("channel*uniforms.source_capacity+t") << "*mask;previous=next;}"
      << y.SetByOffset("global_idx", "value");
    return Status::OK();
  }
};
class AlignmentExtent final : public WebGpuKernel {
 public:
  explicit AlignmentExtent(const OpKernelInfo& i) : WebGpuKernel{i}, capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity", 0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x = c.Input<Tensor>(0); const auto* prefix = c.Input<Tensor>(1); const auto* source_extent = c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(x && x->IsDataType<float>() && x->Shape().NumDimensions() == 3 && x->Shape()[0] == 1 &&
                      (x->Shape()[1] == 640 || x->Shape()[1] == 512) && x->Shape()[2] >= 0 && x->Shape()[2] <= 1024 &&
                      prefix && prefix->IsDataType<int64_t>() && prefix->Shape().NumDimensions() == 1 &&
                      source_extent && source_extent->IsDataType<uint32_t>() && source_extent->Shape() == TensorShape{rw07::ExtentWords},
                      "RW07 alignment exact descriptors");
    const int64_t t = prefix->Shape()[0], channels = x->Shape()[1];
    ORT_RETURN_IF_NOT(t >= 0 && t <= 512 && x->Shape()[2] >= t && capacity_ > 0 && capacity_ <= 102400,
                      "RW07 alignment physical descriptor domain");
    const uint64_t epoch = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(epoch && Gpu(x) && Gpu(prefix) && Gpu(source_extent), "RW07 alignment owner interface/input residency");
    const auto& limits = c.DeviceLimits();
    const uint64_t output_bytes = uint64_t(channels) * uint64_t(capacity_) * 4;
    const uint64_t invocations = uint64_t(channels) * uint64_t(capacity_);
    const uint64_t groups=(invocations+63)/64, limit=limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 4 && limits.maxBufferSize >= 96 && limits.maxStorageBufferBindingSize >= 96 &&
                      output_bytes <= limits.maxBufferSize && output_bytes <= limits.maxStorageBufferBindingSize &&
                      invocations <= UINT32_MAX && limit > 0 && groups <= limit * limit &&
                      limits.maxComputeWorkgroupSizeX >= 64 && limits.maxComputeInvocationsPerWorkgroup >= 64,
                      "RW07 alignment actual allocation/dispatch limits");
    for (const auto* input : {x, prefix, source_extent})
      ORT_RETURN_IF_NOT(input->SizeInBytes() <= limits.maxBufferSize && input->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 alignment input binding limit");
    auto* y = c.Output(0, TensorShape{1,channels,capacity_}); auto* extent = c.Output(1, TensorShape{rw07::ExtentWords});
    ORT_RETURN_IF_NOT(y->IsDataType<float>() && extent->IsDataType<uint32_t>() && Gpu(y) && Gpu(extent), "RW07 alignment output role");
    Tensor empty_prefix, empty_x;
    if (t == 0) empty_prefix = c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{2});
    if (x->Shape().Size() == 0) empty_x = c.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape{1});
    const auto* actual_prefix = t ? prefix : &empty_prefix;
    AlignmentExtentGuard guard;
    guard.AddInputs({InWords(source_extent,24), InWords(actual_prefix,uint32_t(t ? t*2 : 2))})
        .AddOutput(OutWords(extent,24)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(t),uint32_t(channels),uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const uint64_t gx=std::min(groups,limit), gy=(groups+gx-1)/gx;
    AlignmentData data;
    data.AddInputs({{x->Shape().Size() ? x : &empty_x,ProgramTensorMetadataDependency::None},InWords(actual_prefix,uint32_t(t ? t*2 : 2)),InWords(extent,24)})
        .AddOutput({y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
        .AddUniformVariables({uint32_t(t),uint32_t(x->Shape()[2]),uint32_t(channels),uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(data));
    for (const auto& entry : {std::pair{"alignment-input",x},std::pair{"prefix-device-only",prefix},std::pair{"source-extent-device-only",source_extent},
                              std::pair{"alignment-output",static_cast<const Tensor*>(y)},std::pair{"extent-device-only",static_cast<const Tensor*>(extent)}})
      LOGS(c.Logger(), INFO) << "M0_RW07_RESOURCE role=" << entry.first << " epoch=" << epoch << " handle=" << entry.second->DataRaw()
          << " device=" << int(entry.second->Location().device.Type()) << " physical_bytes=" << entry.second->SizeInBytes();
    return Status::OK();
  }
 private: int64_t capacity_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuAlignmentExtentV2KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status { out=std::make_unique<AlignmentExtent>(i);return Status::OK(); };
  return {KernelDefBuilder().SetName("AlignmentExtentV2").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
} // namespace onnxruntime::webgpu
