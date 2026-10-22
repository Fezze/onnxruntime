// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original1599/1600: exact noise convolutions consuming the owned polar1597 carrier.
#include <algorithm>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"

namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
using m0_residual_stages::ValidOutput;
using m0_residual_stages::PhysicalBuffer;
using m0_residual_stages::StorageBudget;
using m0_residual_stages::StorageBudgetWithWindows;
class Guard final : public Program<Guard> {
 public:
  Guard() : Program{"RW07NoiseConvAfter1597GuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"branch", ProgramUniformVariableDataType::Uint32},
      {"source_pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << o.SetByOffset("i","0u") << "}"
      << o.SetByOffset("0u","2u") << o.SetByOffset("1u","3u")
      << "for(var i=0u;i<10u;i++){" << o.SetByOffset("22u+i",GetElementAt("uniforms.identity","i",10)) << "}"
      << "if(uniforms.source_pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||"
      << e.GetByOffset("1u") << "!=3u){" << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity","i",10) << "){" << o.SetByOffset("20u","103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}}if(" << e.GetByOffset("21u") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}" << o.SetByOffset("20u","status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("4u") << "!=22u||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}}let n=" << e.GetByOffset("6u") << ";"
      << "if(n<121u||n>12288001u||n%120u!=1u||" << e.GetByOffset("18u") << "!=22u*n){"
      << o.SetByOffset("20u","103u") << "return;}"
      << "let count=select((n-1u)/6u,n,uniforms.branch==1u);"
      << "let channels=select(256u,128u,uniforms.branch==1u);"
      << "if(n>uniforms.input_capacity||count>uniforms.output_capacity){"
      << o.SetByOffset("20u","102u") << "return;}"
      << o.SetByOffset("2u","1u") << o.SetByOffset("4u","channels") << o.SetByOffset("6u","count")
      << o.SetByOffset("18u","channels*count") << o.SetByOffset("21u","63u");
    return Status::OK();
  }
};
class NoiseConvExtent final : public WebGpuKernel {
 public:
  explicit NoiseConvExtent(const OpKernelInfo& i) : WebGpuKernel{i}, capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity", 0)), branch_(i.GetAttrOrDefault<int64_t>("branch", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x = c.Input<Tensor>(0); const auto* e = c.Input<Tensor>(1);
    const auto* w = c.Input<Tensor>(2); const auto* b = c.Input<Tensor>(3);
    ORT_RETURN_IF_NOT(branch_ == 0 || branch_ == 1, "RW07 noise conv branch");
    const uint32_t channels = branch_ == 0 ? 256u : 128u, kernel = branch_ == 0 ? 12u : 1u;
    ORT_RETURN_IF_NOT(Gpu(w) && w->IsDataType<float>() && w->Shape() == TensorShape({channels,22,kernel}) &&
                          Gpu(b) && b->IsDataType<float>() && b->Shape() == TensorShape({channels}),
                      "RW07 noise conv exact parameter descriptors");
    ORT_RETURN_IF_NOT(Gpu(x) && x->IsDataType<float>() && x->Shape().NumDimensions() == 3 &&
                          x->Shape()[0] == 1 && x->Shape()[1] == 22 && x->Shape()[2] > 0 && x->Shape()[2] <= 12288001 &&
                          Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          capacity_ > 0 && capacity_ <= 12288001,
                      "RW07 noise conv after1597 exact physical descriptors");
    const auto& limits = c.DeviceLimits();
    const uint64_t bytes = uint64_t(channels) * uint64_t(capacity_) * 4;
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 5 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit > 0 &&
                          bytes <= limits.maxBufferSize,
                      "RW07 noise conv after1597 resource capacity");
    auto binding = [&](const Tensor* t) -> Status { return PhysicalBuffer(c,t); };
    ORT_RETURN_IF_ERROR(binding(x)); ORT_RETURN_IF_ERROR(binding(e));
    ORT_RETURN_IF_ERROR(binding(w)); ORT_RETURN_IF_ERROR(binding(b));
    const uint32_t pc=uint32_t(x->Shape()[2]),oc=uint32_t(capacity_);
    uint32_t input_tile=0;
    ORT_RETURN_IF_ERROR(m0_residual_stages::ResidualTileChannels(c,pc,oc,channels,input_tile));
    input_tile=std::min(input_tile,22u);
    const uint64_t prefix=limits.minStorageBufferOffsetAlignment-sizeof(float);
    const uint32_t output_tile=static_cast<uint32_t>(std::min<uint64_t>(channels,
        (limits.maxStorageBufferBindingSize-prefix)/(uint64_t(std::max(pc,oc))*sizeof(float))));
    const uint64_t groups=(uint64_t(output_tile)*oc+63)/64;
    ORT_RETURN_IF_NOT(output_tile>0&&groups<=limit*limit,"RW07 noise convolution window dispatch limit");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity source, carrier, pair, output;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*e, source) &&
                          c.M0FullGpuDescribeResourceV2(*x, carrier), "RW07 noise conv after1597 actual resources");
    const bool paired = c.M0FullGpuDescribePairV2(*e, *x, pair) && pair.session == session && pair.run == run &&
                        pair.generation == source.generation && pair.data_generation == carrier.generation &&
                        pair.producer_role == 15 && pair.producer_instance == 1016;
    auto* y = c.Output(0, TensorShape{1, channels, capacity_}); auto* out = c.Output(1, TensorShape{32});
    ORT_RETURN_IF_NOT(Gpu(y) && y->IsDataType<float>() && Gpu(out) && out->IsDataType<uint32_t>(),
                      "RW07 noise conv after1597 output residency/types");
    ORT_RETURN_IF_ERROR(binding(y)); ORT_RETURN_IF_ERROR(binding(out));
    ORT_RETURN_IF_NOT(y->DataRaw() != out->DataRaw() && y->DataRaw() != x->DataRaw() && y->DataRaw() != e->DataRaw() &&
                          out->DataRaw() != x->DataRaw() && out->DataRaw() != e->DataRaw(),
                      "RW07 noise conv after1597 distinct outputs/input allocations");
    ORT_RETURN_IF_NOT(y->DataRaw() != w->DataRaw() && y->DataRaw() != b->DataRaw() &&
                          out->DataRaw() != w->DataRaw() && out->DataRaw() != b->DataRaw(),
                      "RW07 noise conv output/parameter allocations distinct");
    ORT_RETURN_IF_ERROR(StorageBudget(c,{e,out}));
    const auto input_window=[&](uint32_t offset,uint32_t count){
      return ProgramInput::FloatBufferWindow(x,ProgramTensorMetadataDependency::None,TensorShape{1,count,pc},
          offset*pc,limits.minStorageBufferOffsetAlignment);
    };
    const auto output_window=[&](uint32_t offset,uint32_t count){
      return ProgramOutput::FloatBufferWindow(y,ProgramTensorMetadataDependency::None,TensorShape{1,count,oc},
          offset*oc,limits.minStorageBufferOffsetAlignment);
    };
    for(uint32_t i=0;i<22;i+=input_tile){for(uint32_t o=0;o<channels;o+=output_tile){
      ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,input_window(i,std::min(input_tile,22u-i)),
          output_window(o,std::min(output_tile,channels-o)),{out,w,b}));
    }}
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, uint32_t(16 + branch_), uint32_t(1017 + branch_), output), "RW07 noise conv after1597 output claim");
    const m0_extent_abi2::OpaqueOwnership src{session, run, source.generation, carrier.generation, 15, 1016};
    const m0_extent_abi2::OpaqueOwnership dst{session, run, output.generation, output.data_generation, uint32_t(16 + branch_), uint32_t(1017 + branch_)};
    ORT_RETURN_IF_NOT(src.Valid() && dst.Valid(), "RW07 noise conv after1597 opaque ownership");
    const auto sw = src.UniformWords(), ow = dst.UniformWords();
    const auto source_identity = gsl::span<const uint32_t>{sw.data(), sw.size()}, identity = gsl::span<const uint32_t>{ow.data(), ow.size()};
    Guard guard;
    guard.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({out, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({pc, oc, uint32_t(branch_), uint32_t(paired), source_identity, identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    for(uint32_t i=0;i<22;i+=input_tile){for(uint32_t o=0;o<channels;o+=output_tile){
      const uint32_t ic=std::min(input_tile,22u-i),co=std::min(output_tile,channels-o);
      const uint64_t current_groups=(uint64_t(co)*oc+63)/64;
      const uint64_t dx=std::min(current_groups,limit),dy=(current_groups+dx-1)/dx;
      m0_residual_stages::GeneratorChannelConv convolution;
      convolution.AddInput(input_window(i,ic)).AddInputs({{out,ProgramTensorMetadataDependency::None},
          {w,ProgramTensorMetadataDependency::None},{b,ProgramTensorMetadataDependency::None}})
          .AddOutput(output_window(o,co)).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(dx),uint32_t(dy))
          .AddUniformVariables({pc,oc,channels,branch_==0?20u:121u,branch_==0?2048000u:12288001u,
              branch_==0?20u:120u,branch_==0?0u:1u,22u,kernel,branch_==0?6u:1u,branch_==0?3u:0u,
              branch_==0?6u:1u,branch_==0?1u:0u,0u,i,ic,o,co,identity});
      ORT_RETURN_IF_ERROR(c.RunProgram(convolution));
    }}
    LOGS(c.Logger(), INFO) << "M0_RW07_NOISE_CONV_AFTER1597_V1 session=" << session << " run=" << run
                          << " source_pair_valid=" << paired << " branch=" << branch_
                          << " GPU_dispatches="<<(1+((22u+input_tile-1)/input_tile)*((channels+output_tile-1)/output_tile))
                          << " input_tile_channels="<<input_tile<<" output_tile_channels="<<output_tile<<" host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_;
  const int64_t branch_;
};
class PacketProgram final : public Program<PacketProgram> {
 public:
  PacketProgram() : Program{"RW07NoiseConvAfter1597TerminalPacketV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"minimum", ProgramUniformVariableDataType::Uint32},
      {"maximum", ProgramUniformVariableDataType::Uint32},
      {"modulus", ProgramUniformVariableDataType::Uint32},
      {"remainder", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& p = s.AddOutput("packet", ShaderUsage::UseValueTypeAlias);
    ValidOutput(s, e);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<18u;i++){" << p.SetByOffset("i", "0u") << "}"
      << p.SetByOffset("0u", "103u")
      << "if(" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=3u){return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.identity", "i", 10) << "){return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){return;}" << p.SetByOffset("0u", "status") << "return;}"
      << "if(!valid_output()){return;}" << p.SetByOffset("0u", "0u") << p.SetByOffset("1u", "3u")
      << "for(var i=0u;i<6u;i++){" << p.SetByOffset("2u+i", e.GetByOffset("2u+i")) << "}";
    return Status::OK();
  }
};
class Packet final : public WebGpuKernel {
 public:
  explicit Packet(const OpKernelInfo& i) : WebGpuKernel{i}, branch_(i.GetAttrOrDefault<int64_t>("branch", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e = c.Input<Tensor>(0); const auto* y = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(branch_ == 0 || branch_ == 1, "RW07 noise packet branch");
    const uint32_t channels = branch_ == 0 ? 256u : 128u;
    ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          Gpu(y) && y->IsDataType<float>() && y->Shape().NumDimensions() == 3 &&
                          y->Shape()[0] == 1 && y->Shape()[1] == channels && y->Shape()[2] > 0 && y->Shape()[2] <= 12288001,
                      "RW07 noise conv after1597 packet physical descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*e, *y, pair) && pair.session == session &&
                          pair.run == run && pair.producer_role == uint32_t(16 + branch_) && pair.producer_instance == uint32_t(1017 + branch_),
                      "RW07 noise conv after1597 terminal actual pair");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 noise conv after1597 packet resources");
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c,y));
    auto* p = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p) && p->IsDataType<uint32_t>() && p->DataRaw() != e->DataRaw() && p->DataRaw() != y->DataRaw(),
                      "RW07 noise conv after1597 distinct packet output");
    ORT_RETURN_IF_ERROR(StorageBudget(c,{e,p}));
    const m0_extent_abi2::OpaqueOwnership owner{session, run, pair.generation, pair.data_generation, uint32_t(16 + branch_), uint32_t(1017 + branch_)};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 noise conv after1597 packet opaque ownership");
    const auto words = owner.UniformWords();
    PacketProgram program;
    program.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({p, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[2]), uint32_t(y->Shape()[2]), channels,
                              branch_ == 0 ? 20u : 121u, branch_ == 0 ? 2048000u : 12288001u,
                              branch_ == 0 ? 20u : 120u, branch_ == 0 ? 0u : 1u, gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p), "RW07 noise conv after1597 packet map fault registration");
    return Status::OK();
  }
 private:
  const int64_t branch_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuNoiseConvAfter1597ExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<NoiseConvExtent>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("NoiseConvAfter1597ExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuNoiseConvAfter1597TerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Packet>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("NoiseConvAfter1597TerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
