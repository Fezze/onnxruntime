// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original1575..1597. The exported phase expansion deliberately is not atan2.
#include <algorithm>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"

namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
void ValidOutput(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_output()->bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=3u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("4u") << "!=22u||" << e.GetByOffset("5u") << "!=0u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let n=" << e.GetByOffset("6u") << ";return n>=121u&&n<=12288001u&&n%120u==1u&&"
    << "n<=uniforms.input_capacity&&n<=uniforms.output_capacity&&" << e.GetByOffset("18u") << "==22u*n;}\n";
}
class Guard final : public Program<Guard> {
 public:
  Guard() : Program{"RW07PolarAfter1573GuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"source_pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("source", ShaderUsage::UseValueTypeAlias);
    const auto& o = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << o.SetByOffset("i", "0u") << "}"
      << o.SetByOffset("0u", "2u") << o.SetByOffset("1u", "3u")
      << "for(var i=0u;i<10u;i++){" << o.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "if(uniforms.source_pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||"
      << e.GetByOffset("1u") << "!=4u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity", "i", 10) << "){" << o.SetByOffset("20u", "103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}if(" << e.GetByOffset("21u") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}" << o.SetByOffset("20u", "status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("5u") << "!=0u||" << e.GetByOffset("6u") << "!=11u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("8u") << "!=2u||"
      << e.GetByOffset("9u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=10u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}let n=" << e.GetByOffset("4u") << ";"
      << "if(n<121u||n>12288001u||n%120u!=1u||" << e.GetByOffset("18u") << "!=22u*n){"
      << o.SetByOffset("20u", "103u") << "return;}"
      << "if(n>uniforms.input_capacity||n>uniforms.output_capacity){" << o.SetByOffset("20u", "102u") << "return;}"
      << o.SetByOffset("2u", "1u") << o.SetByOffset("4u", "22u") << o.SetByOffset("6u", "n")
      << o.SetByOffset("18u", "22u*n") << o.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class Polar final : public Program<Polar> {
 public:
  Polar() : Program{"RW07Original1575Through1597PolarV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("spectrum", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("polar", ShaderUsage::UseValueTypeAlias);
    ValidOutput(s, e);
    s.AdditionalImplementation() << R"WGSL(
fn nan_value()->f32 {return bitcast<f32>(0x7fc00000u);}
fn square(x:f32)->f32 {
 let bits=bitcast<u32>(x)&0x7fffffffu;
 if(bits>0x7f800000u){return nan_value();}
 if(bits==0x7f800000u){return bitcast<f32>(0x7f800000u);}
 if(bits==0u){return 0.0;}return x*x;
}
fn original_div(a:f32,b:f32)->f32 {
 let ua=bitcast<u32>(a);let ub=bitcast<u32>(b);
 let aa=ua&0x7fffffffu;let ab=ub&0x7fffffffu;let sign=(ua^ub)&0x80000000u;
 if(aa>0x7f800000u||ab>0x7f800000u){return nan_value();}
 if((aa==0u&&ab==0u)||(aa==0x7f800000u&&ab==0x7f800000u)){return nan_value();}
 if(ab==0u||aa==0x7f800000u){return bitcast<f32>(sign|0x7f800000u);}
 if(aa==0u||ab==0x7f800000u){return bitcast<f32>(sign);}return a/b;
}
fn original_atan(x:f32)->f32 {
 let bits=bitcast<u32>(x);let a=bits&0x7fffffffu;
 if(a>0x7f800000u){return nan_value();}
 if(a==0x7f800000u){return bitcast<f32>((bits&0x80000000u)|0x3fc90fdbu);}
 if(a==0u){return x;}return atan(x);
}
)WGSL";
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=11u*uniforms.output_capacity){return;}"
      << "let bin=global_idx/uniforms.output_capacity;let frame=global_idx%uniforms.output_capacity;"
      << "let phase_offset=(bin+11u)*uniforms.output_capacity+frame;"
      << y.SetByOffset("global_idx", "0.0") << y.SetByOffset("phase_offset", "0.0")
      << "if(!valid_output()||frame>=" << e.GetByOffset("6u") << "){return;}"
      << "let re=" << x.GetByOffset("22u*frame+2u*bin") << ";"
      << "let im=" << x.GetByOffset("22u*frame+2u*bin+1u") << ";"
      << "let re2=square(re);let im2=square(im);let total:f32=re2+im2;"
      << "let magnitude=sqrt(total);let ratio=original_div(im,re);let angle=original_atan(ratio);"
      << "let pi=bitcast<f32>(0x40490fdbu);let positive:f32=angle+pi;let negative:f32=angle-pi;"
      << "let rb=bitcast<u32>(re);let ib=bitcast<u32>(im);"
      << "let imag_positive=(ib&0x80000000u)==0u&&(ib&0x7fffffffu)>0u&&(ib&0x7fffffffu)<=0x7f800000u;"
      << "let real_negative=(rb&0x80000000u)!=0u&&(rb&0x7fffffffu)>0u&&(rb&0x7fffffffu)<=0x7f800000u;"
      << "let quadrant=select(negative,positive,imag_positive);let phase=select(angle,quadrant,real_negative);"
      << y.SetByOffset("global_idx", "magnitude") << y.SetByOffset("phase_offset", "phase");
    return Status::OK();
  }
};

class PolarExtent final : public WebGpuKernel {
 public:
  explicit PolarExtent(const OpKernelInfo& i) : WebGpuKernel{i}, capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity", 0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x = c.Input<Tensor>(0); const auto* e = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(Gpu(x) && x->IsDataType<float>() && x->Shape().NumDimensions() == 4 &&
                          x->Shape()[0] == 1 && x->Shape()[1] > 0 && x->Shape()[1] <= 12288001 &&
                          x->Shape()[2] == 11 && x->Shape()[3] == 2 &&
                          Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          capacity_ > 0 && capacity_ <= 12288001,
                      "RW07 polar after1573 exact physical descriptors");
    const auto& limits = c.DeviceLimits();
    const uint64_t bytes = 22 * uint64_t(capacity_) * 4, groups = (11 * uint64_t(capacity_) + 63) / 64;
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 3 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit > 0 && groups <= limit * limit &&
                          bytes <= limits.maxBufferSize && bytes <= limits.maxStorageBufferBindingSize,
                      "RW07 polar after1573 resource capacity");
    auto binding = [&](const Tensor* t) -> Status {
      ORT_RETURN_IF_NOT(t->DataRaw() && t->SizeInBytes() > 0 && t->SizeInBytes() <= limits.maxBufferSize &&
                            t->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 polar after1573 binding descriptor");
      const uint64_t actual = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
      ORT_RETURN_IF_NOT(actual >= t->SizeInBytes() && actual <= limits.maxBufferSize && actual <= limits.maxStorageBufferBindingSize,
                        "RW07 polar after1573 actual binding backing");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(binding(x)); ORT_RETURN_IF_ERROR(binding(e));
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity source, carrier, pair, output;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*e, source) &&
                          c.M0FullGpuDescribeResourceV2(*x, carrier), "RW07 polar after1573 actual resources");
    const bool paired = c.M0FullGpuDescribePairV2(*e, *x, pair) && pair.session == session && pair.run == run &&
                        pair.generation == source.generation && pair.data_generation == carrier.generation &&
                        pair.producer_role == 14 && pair.producer_instance == 1015;
    auto* y = c.Output(0, TensorShape{1, 22, capacity_}); auto* out = c.Output(1, TensorShape{32});
    ORT_RETURN_IF_NOT(Gpu(y) && y->IsDataType<float>() && Gpu(out) && out->IsDataType<uint32_t>(),
                      "RW07 polar after1573 output residency/types");
    ORT_RETURN_IF_ERROR(binding(y)); ORT_RETURN_IF_ERROR(binding(out));
    ORT_RETURN_IF_NOT(y->DataRaw() != out->DataRaw() && y->DataRaw() != x->DataRaw() && y->DataRaw() != e->DataRaw() &&
                          out->DataRaw() != x->DataRaw() && out->DataRaw() != e->DataRaw(),
                      "RW07 polar after1573 distinct outputs/input allocations");
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, 15, 1016, output), "RW07 polar after1573 output claim");
    const m0_extent_abi2::OpaqueOwnership src{session, run, source.generation, carrier.generation, 14, 1015};
    const m0_extent_abi2::OpaqueOwnership dst{session, run, output.generation, output.data_generation, 15, 1016};
    ORT_RETURN_IF_NOT(src.Valid() && dst.Valid(), "RW07 polar after1573 opaque ownership");
    const auto sw = src.UniformWords(), ow = dst.UniformWords();
    const auto source_identity = gsl::span<const uint32_t>{sw.data(), sw.size()}, identity = gsl::span<const uint32_t>{ow.data(), ow.size()};
    const uint32_t pc = uint32_t(x->Shape()[1]), oc = uint32_t(capacity_);
    Guard guard;
    guard.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({out, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({pc, oc, uint32_t(paired), source_identity, identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const uint64_t dx = std::min(groups, limit), dy = (groups + dx - 1) / dx;
    Polar polar;
    polar.AddInputs({{x, ProgramTensorMetadataDependency::None}, {out, ProgramTensorMetadataDependency::None}})
        .AddOutput({y, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
        .SetDispatchGroupSize(uint32_t(dx), uint32_t(dy)).AddUniformVariables({pc, oc, identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(polar));
    LOGS(c.Logger(), INFO) << "M0_RW07_POLAR_AFTER1573_V1 session=" << session << " run=" << run
                          << " source_pair_valid=" << paired << " role=15 instance=1016 GPU_dispatches=2 host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_;
};
class PacketProgram final : public Program<PacketProgram> {
 public:
  PacketProgram() : Program{"RW07PolarAfter1573TerminalPacketV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
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
  explicit Packet(const OpKernelInfo& i) : WebGpuKernel{i} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e = c.Input<Tensor>(0); const auto* y = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          Gpu(y) && y->IsDataType<float>() && y->Shape().NumDimensions() == 3 &&
                          y->Shape()[0] == 1 && y->Shape()[1] == 22 && y->Shape()[2] > 0 && y->Shape()[2] <= 12288001,
                      "RW07 polar after1573 packet physical descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*e, *y, pair) && pair.session == session &&
                          pair.run == run && pair.producer_role == 15 && pair.producer_instance == 1016,
                      "RW07 polar after1573 terminal actual pair");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 polar after1573 packet resources");
    auto binding = [&](const Tensor* t) -> Status {
      ORT_RETURN_IF_NOT(t->DataRaw() && t->SizeInBytes() <= limits.maxBufferSize && t->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 polar after1573 packet binding descriptor");
      const uint64_t bytes = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
      ORT_RETURN_IF_NOT(bytes >= t->SizeInBytes() && bytes <= limits.maxBufferSize && bytes <= limits.maxStorageBufferBindingSize,
                        "RW07 polar after1573 packet actual backing");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(binding(e)); ORT_RETURN_IF_ERROR(binding(y));
    auto* p = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p) && p->IsDataType<uint32_t>() && p->DataRaw() != e->DataRaw() && p->DataRaw() != y->DataRaw(),
                      "RW07 polar after1573 distinct packet output");
    ORT_RETURN_IF_ERROR(binding(p));
    const m0_extent_abi2::OpaqueOwnership owner{session, run, pair.generation, pair.data_generation, 15, 1016};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 polar after1573 packet opaque ownership");
    const auto words = owner.UniformWords();
    PacketProgram program;
    program.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({p, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[2]), uint32_t(y->Shape()[2]), gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p), "RW07 polar after1573 packet map fault registration");
    return Status::OK();
  }
};
}  // namespace
KernelCreateInfo CreateM0FullGpuPolarAfter1573ExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<PolarExtent>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("PolarAfter1573ExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuPolarAfter1573TerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Packet>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("PolarAfter1573TerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
