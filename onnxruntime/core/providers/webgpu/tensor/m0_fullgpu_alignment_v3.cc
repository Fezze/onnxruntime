// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Exact original alignment cut1318..1331. ABI2 ownership, no host tensor values.
#include <algorithm>

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/m0_fullgpu_resource_ownership.h"

namespace onnxruntime::webgpu {
namespace {
constexpr uint32_t kAlignmentRole = 4;
constexpr uint32_t kAlignment640Instance = 1004;
constexpr uint32_t kAlignment512Instance = 1005;

ProgramInput Words(const Tensor* tensor, uint32_t count) {
  ProgramInput result(tensor, ProgramTensorMetadataDependency::None, TensorShape{count}, 1);
  result.var_type = ProgramVariableDataType::Uint32;
  return result;
}
ProgramOutput Words(Tensor* tensor) {
  ProgramOutput result(tensor, ProgramTensorMetadataDependency::None, TensorShape{m0_extent_abi2::Words}, 1);
  result.var_type = ProgramVariableDataType::Uint32;
  return result;
}
bool Gpu(const Tensor* tensor) {
  return tensor && tensor->Location().device.Type() == OrtDevice::GPU;
}

class AlignmentGuard final : public Program<AlignmentGuard> {
 public:
  AlignmentGuard() : Program{"RW07V3AlignmentOwnedGuard"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"tokens", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"epoch_low", ProgramUniformVariableDataType::Uint32},
      {"epoch_high", ProgramUniformVariableDataType::Uint32},
      {"run_low", ProgramUniformVariableDataType::Uint32},
      {"run_high", ProgramUniformVariableDataType::Uint32},
      {"source_generation_low", ProgramUniformVariableDataType::Uint32},
      {"source_generation_high", ProgramUniformVariableDataType::Uint32},
      {"source_carrier_low", ProgramUniformVariableDataType::Uint32},
      {"source_carrier_high", ProgramUniformVariableDataType::Uint32},
      {"output_generation_low", ProgramUniformVariableDataType::Uint32},
      {"output_generation_high", ProgramUniformVariableDataType::Uint32},
      {"output_carrier_low", ProgramUniformVariableDataType::Uint32},
      {"output_carrier_high", ProgramUniformVariableDataType::Uint32},
      {"instance", ProgramUniformVariableDataType::Uint32});

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& source = shader.AddInput("source", ShaderUsage::UseValueTypeAlias);
    const auto& prefix = shader.AddInput("prefix", ShaderUsage::UseValueTypeAlias);
    const auto& total = shader.AddInput("total", ShaderUsage::UseValueTypeAlias);
    const auto& output = shader.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    auto& f = shader.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << output.SetByOffset("i", "0u") << "}"
      << output.SetByOffset("0u", "2u") << output.SetByOffset("1u", "3u")
      << output.SetByOffset("22u", "uniforms.epoch_low") << output.SetByOffset("23u", "uniforms.epoch_high")
      << output.SetByOffset("24u", "uniforms.run_low") << output.SetByOffset("25u", "uniforms.run_high")
      << output.SetByOffset("26u", "uniforms.output_generation_low")
      << output.SetByOffset("27u", "uniforms.output_generation_high")
      << output.SetByOffset("28u", "uniforms.output_carrier_low")
      << output.SetByOffset("29u", "uniforms.output_carrier_high")
      << output.SetByOffset("30u", "4u") << output.SetByOffset("31u", "uniforms.instance")
      << "if(" << source.GetByOffset("0u") << "!=2u||" << source.GetByOffset("1u") << "!=1u||"
      << source.GetByOffset("22u") << "!=uniforms.epoch_low||"
      << source.GetByOffset("23u") << "!=uniforms.epoch_high||"
      << source.GetByOffset("24u") << "!=uniforms.run_low||"
      << source.GetByOffset("25u") << "!=uniforms.run_high||"
      << source.GetByOffset("26u") << "!=uniforms.source_generation_low||"
      << source.GetByOffset("27u") << "!=uniforms.source_generation_high||"
      << source.GetByOffset("28u") << "!=uniforms.source_carrier_low||"
      << source.GetByOffset("29u") << "!=uniforms.source_carrier_high||"
      << source.GetByOffset("30u") << "!=1u||" << source.GetByOffset("31u") << "!=1001u){"
      << output.SetByOffset("20u", "103u") << "return;}"
      << "let prior_status=" << source.GetByOffset("20u") << ";if(prior_status!=0u){"
      << output.SetByOffset("20u", "prior_status") << "return;}"
      << "for(var i=4u;i<18u;i++){if(" << source.GetByOffset("i") << "!=0u){"
      << output.SetByOffset("20u", "103u") << "return;}}"
      << "let d=" << source.GetByOffset("2u") << ";if("
      << source.GetByOffset("3u") << "!=0u||" << source.GetByOffset("18u") << "!=d||"
      << source.GetByOffset("19u") << "!=0u||(" << source.GetByOffset("21u") << "&63u)!=63u||"
      << "d==0u||uniforms.tokens==0u||d>100u*uniforms.tokens||"
      << total.GetByOffset("0u") << "!=d||" << total.GetByOffset("1u") << "!=0u){"
      << output.SetByOffset("20u", "103u") << "return;}"
      << "if(d>uniforms.capacity){" << output.SetByOffset("20u", "102u") << "return;}"
      << "var previous=0u;for(var t=0u;t<uniforms.tokens;t++){let next=" << prefix.GetByOffset("2u*t")
      << ";if(" << prefix.GetByOffset("2u*t+1u") << "!=0u||next<=previous||next>d){"
      << output.SetByOffset("20u", "103u") << "return;}if(next-previous>100u){"
      << output.SetByOffset("20u", "103u") << "return;}previous=next;}"
      << "if(previous!=d){" << output.SetByOffset("20u", "103u") << "return;}"
      << output.SetByOffset("2u", "1u") << output.SetByOffset("4u", "uniforms.channels")
      << output.SetByOffset("6u", "d") << output.SetByOffset("18u", "uniforms.channels*d")
      << output.SetByOffset("21u", "63u");
    return Status::OK();
  }
};

class AlignmentData final : public Program<AlignmentData> {
 public:
  AlignmentData() : Program{"RW07V3AlignmentOwnedMatMul"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"tokens", ProgramUniformVariableDataType::Uint32},
      {"source_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"epoch_low", ProgramUniformVariableDataType::Uint32},
      {"epoch_high", ProgramUniformVariableDataType::Uint32},
      {"run_low", ProgramUniformVariableDataType::Uint32},
      {"run_high", ProgramUniformVariableDataType::Uint32},
      {"output_generation_low", ProgramUniformVariableDataType::Uint32},
      {"output_generation_high", ProgramUniformVariableDataType::Uint32},
      {"output_carrier_low", ProgramUniformVariableDataType::Uint32},
      {"output_carrier_high", ProgramUniformVariableDataType::Uint32},
      {"instance", ProgramUniformVariableDataType::Uint32});

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& x = shader.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& prefix = shader.AddInput("prefix", ShaderUsage::UseValueTypeAlias);
    const auto& extent = shader.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    auto& f = shader.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels*uniforms.capacity){return;}" << y.SetByOffset("global_idx", "0.0")
      << "if(" << extent.GetByOffset("0u") << "!=2u||" << extent.GetByOffset("1u") << "!=3u||"
      << extent.GetByOffset("20u") << "!=0u||(" << extent.GetByOffset("21u") << "&63u)!=63u||"
      << extent.GetByOffset("22u") << "!=uniforms.epoch_low||"
      << extent.GetByOffset("23u") << "!=uniforms.epoch_high||"
      << extent.GetByOffset("24u") << "!=uniforms.run_low||"
      << extent.GetByOffset("25u") << "!=uniforms.run_high||"
      << extent.GetByOffset("26u") << "!=uniforms.output_generation_low||"
      << extent.GetByOffset("27u") << "!=uniforms.output_generation_high||"
      << extent.GetByOffset("28u") << "!=uniforms.output_carrier_low||"
      << extent.GetByOffset("29u") << "!=uniforms.output_carrier_high||"
      << extent.GetByOffset("30u") << "!=4u||" << extent.GetByOffset("31u") << "!=uniforms.instance||"
      << extent.GetByOffset("2u") << "!=1u||" << extent.GetByOffset("3u") << "!=0u||"
      << extent.GetByOffset("4u") << "!=uniforms.channels||" << extent.GetByOffset("5u") << "!=0u||"
      << extent.GetByOffset("7u") << "!=0u){return;}"
      << "let d=" << extent.GetByOffset("6u") << ";if(d==0u||d>uniforms.capacity||"
      << extent.GetByOffset("18u") << "!=uniforms.channels*d||" << extent.GetByOffset("19u")
      << "!=0u){return;}"
      << "for(var i=8u;i<18u;i++){if(" << extent.GetByOffset("i") << "!=0u){return;}}"
      << "let channel=global_idx/uniforms.capacity;let frame=global_idx%uniforms.capacity;"
      << "if(frame>=d){return;}var value=0.0;var previous=0u;"
      // Retain every original MatMul term, including zero*NaN/Inf.
      << "for(var t=0u;t<uniforms.tokens;t++){let next=" << prefix.GetByOffset("2u*t") << ";"
      << "let mask=select(0.0,1.0,frame>=previous&&frame<next);"
      << "value+=" << x.GetByOffset("channel*uniforms.source_capacity+t") << "*mask;previous=next;}"
      << y.SetByOffset("global_idx", "value");
    return Status::OK();
  }
};

class AlignmentExtent final : public WebGpuKernel {
 public:
  explicit AlignmentExtent(const OpKernelInfo& info)
      : WebGpuKernel{info}, capacity_(info.GetAttrOrDefault<int64_t>("physical_capacity", 0)), source_role_(info.GetAttrOrDefault<int64_t>("source_role", -1)), source_instance_(info.GetAttrOrDefault<int64_t>("source_instance", -1)), semantic_role_(info.GetAttrOrDefault<int64_t>("semantic_role", -1)), instance_(info.GetAttrOrDefault<int64_t>("producer_instance", -1)) {}

  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x = c.Input<Tensor>(0);
    const auto* prefix = c.Input<Tensor>(1);
    const auto* source = c.Input<Tensor>(2);
    const auto* total = c.Input<Tensor>(3);
    ORT_RETURN_IF_NOT(source_role_ == 1 && source_instance_ == 1001 && semantic_role_ == kAlignmentRole,
                      "RW07 alignment V3 source/producer roles");
    ORT_RETURN_IF_NOT(Gpu(x) && x->IsDataType<float>() && x->Shape().NumDimensions() == 3 &&
                          x->Shape()[0] == 1 && x->Shape()[2] >= 0 && x->Shape()[2] <= 1024 &&
                          ((x->Shape()[1] == 640 && instance_ == kAlignment640Instance) ||
                           (x->Shape()[1] == 512 && instance_ == kAlignment512Instance)) &&
                          Gpu(prefix) && prefix->IsDataType<int64_t>() && prefix->Shape().NumDimensions() == 1 &&
                          Gpu(source) && source->IsDataType<uint32_t>() &&
                          source->Shape() == TensorShape{m0_extent_abi2::Words} &&
                          Gpu(total) && total->IsDataType<int64_t>() && total->Shape().NumDimensions() == 0,
                      "RW07 alignment V3 exact input descriptors");
    const int64_t tokens = prefix->Shape()[0], channels = x->Shape()[1];
    ORT_RETURN_IF_NOT(tokens >= 0 && tokens <= 512 && x->Shape()[2] >= tokens &&
                          capacity_ > 0 && capacity_ <= 102400,
                      "RW07 alignment V3 physical domain");
    const uint64_t epoch = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity source_id, total_id, prefix_id;
    ORT_RETURN_IF_NOT(epoch && run && c.M0FullGpuDescribePairV2(*source, *total, source_id) &&
                          c.M0FullGpuDescribeResourceV2(*total, total_id) &&
                          source_id.session == epoch && source_id.run == run && source_id.producer_role == 1 &&
                          source_id.producer_instance == 1001,
                      "RW07 alignment V3 duration/total provenance");
    // The producer separately claims the opaque prefix allocation against total.
    // This is a ledger lookup, not a host read of any prefix element.
    if (tokens > 0) {
      ORT_RETURN_IF_NOT(c.M0FullGpuDescribePairV2(*prefix, *total, prefix_id) && prefix_id.session == epoch &&
                            prefix_id.run == run && prefix_id.producer_role == 1 &&
                            prefix_id.producer_instance == 1001 && prefix_id.data_generation == total_id.generation,
                        "RW07 alignment V3 prefix/total provenance");
    }
    const auto& limits = c.DeviceLimits();
    const uint64_t invocations = uint64_t(channels) * uint64_t(capacity_);
    const uint64_t output_bytes = invocations * sizeof(float);
    const uint64_t groups = (invocations + 63) / 64, limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 4 && limits.maxBufferSize >= m0_extent_abi2::Bytes &&
                          limits.maxStorageBufferBindingSize >= m0_extent_abi2::Bytes &&
                          output_bytes <= limits.maxBufferSize && output_bytes <= limits.maxStorageBufferBindingSize &&
                          invocations <= UINT32_MAX && limit > 0 && groups <= limit * limit &&
                          limits.maxComputeWorkgroupSizeX >= 64 && limits.maxComputeInvocationsPerWorkgroup >= 64,
                      "RW07 alignment V3 allocation/dispatch limits");
    auto binding = [&](const Tensor& tensor) -> Status {
      if (tensor.SizeInBytes() == 0) return Status::OK();
      ORT_RETURN_IF_NOT(tensor.DataRaw(), "RW07 alignment V3 missing physical backing");
      const auto handle = static_cast<WGPUBuffer>(const_cast<void*>(tensor.DataRaw()));
      const uint64_t actual = wgpuBufferGetSize(handle);
      ORT_RETURN_IF_NOT(actual >= tensor.SizeInBytes() && actual <= limits.maxBufferSize &&
                            actual <= limits.maxStorageBufferBindingSize,
                        "RW07 alignment V3 actual whole-buffer binding limit");
      return Status::OK();
    };
    for (const auto* input : {x, prefix, source, total}) ORT_RETURN_IF_ERROR(binding(*input));
    auto* y = c.Output(0, TensorShape{1, channels, capacity_});
    auto* extent = c.Output(1, TensorShape{m0_extent_abi2::Words});
    ORT_RETURN_IF_NOT(Gpu(y) && y->IsDataType<float>() && Gpu(extent) && extent->IsDataType<uint32_t>(),
                      "RW07 alignment V3 output descriptors");
    ORT_RETURN_IF_ERROR(binding(*y));
    ORT_RETURN_IF_ERROR(binding(*extent));
    M0FullGpuAllocationIdentity output_id;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*extent, *y, kAlignmentRole, uint32_t(instance_), output_id),
                      "RW07 alignment V3 output allocation claim");
    Tensor empty_prefix, empty_x;
    if (tokens == 0) empty_prefix = c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{2});
    if (x->Shape().Size() == 0) empty_x = c.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape{1});
    const auto* actual_prefix = tokens ? prefix : &empty_prefix;
    const auto* actual_x = x->Shape().Size() ? x : &empty_x;
    ORT_RETURN_IF_ERROR(binding(*actual_prefix));
    ORT_RETURN_IF_ERROR(binding(*actual_x));
    AlignmentGuard guard;
    guard.AddInputs({Words(source, m0_extent_abi2::Words), Words(actual_prefix, uint32_t(tokens ? tokens * 2 : 2)),
                     Words(total, 2)})
        .AddOutput(Words(extent))
        .SetWorkgroupSize(1)
        .SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(tokens), uint32_t(channels), uint32_t(capacity_),
                              uint32_t(epoch), uint32_t(epoch >> 32), uint32_t(run), uint32_t(run >> 32),
                              uint32_t(source_id.generation), uint32_t(source_id.generation >> 32),
                              uint32_t(total_id.generation), uint32_t(total_id.generation >> 32),
                              uint32_t(output_id.generation), uint32_t(output_id.generation >> 32),
                              uint32_t(output_id.data_generation), uint32_t(output_id.data_generation >> 32),
                              uint32_t(instance_)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const uint64_t gx = std::min(groups, limit), gy = (groups + gx - 1) / gx;
    AlignmentData data;
    data.AddInputs({{actual_x, ProgramTensorMetadataDependency::None},
                    Words(actual_prefix, uint32_t(tokens ? tokens * 2 : 2)),
                    Words(extent, m0_extent_abi2::Words)})
        .AddOutput({y, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64)
        .SetDispatchGroupSize(uint32_t(gx), uint32_t(gy))
        .AddUniformVariables({uint32_t(tokens), uint32_t(x->Shape()[2]), uint32_t(channels), uint32_t(capacity_),
                              uint32_t(epoch), uint32_t(epoch >> 32), uint32_t(run), uint32_t(run >> 32),
                              uint32_t(output_id.generation), uint32_t(output_id.generation >> 32),
                              uint32_t(output_id.data_generation), uint32_t(output_id.data_generation >> 32),
                              uint32_t(instance_)});
    ORT_RETURN_IF_ERROR(c.RunProgram(data));
    LOGS(c.Logger(), INFO) << "M0_RW07_ALIGNMENT_V3 session=" << epoch << " run=" << run
                           << " source_generation=" << source_id.generation << " prefix_generation=" << prefix_id.generation
                           << " total_generation=" << total_id.generation << " output_generation=" << output_id.generation
                           << " output_carrier_generation=" << output_id.data_generation << " role=4 instance=" << instance_;
    return Status::OK();
  }

 private:
  const int64_t capacity_, source_role_, source_instance_, semantic_role_, instance_;
};
}  // namespace

KernelCreateInfo CreateM0FullGpuAlignmentExtentV3KernelInfo() {
  KernelCreatePtrFn factory = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& result) -> Status {
    result = std::make_unique<AlignmentExtent>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("AlignmentExtentV3").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), factory};
}
}  // namespace onnxruntime::webgpu
