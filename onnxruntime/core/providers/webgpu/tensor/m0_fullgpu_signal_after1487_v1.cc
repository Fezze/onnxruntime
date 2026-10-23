// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original deterministic harmonic source after F0_proj1487 through Reshape1571.
#include <algorithm>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "m0_fullgpu_signal_programs_v1.h"

namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }

using m0_signal_detail::Prefix;
using m0_signal_detail::Signal;

class Guard final : public Program<Guard> {
 public:
  Guard() : Program{"RW07SignalAfter1487GuardV1"} {}
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
      << o.SetByOffset("0u", "2u") << o.SetByOffset("1u", "2u")
      << "for(var i=0u;i<10u;i++){" << o.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "if(uniforms.source_pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||"
      << e.GetByOffset("1u") << "!=3u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity", "i", 10) << "){" << o.SetByOffset("20u", "103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}if(" << e.GetByOffset("21u") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}" << o.SetByOffset("20u", "status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("4u") << "!=1u||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}let len=" << e.GetByOffset("6u") << ";"
      << "if(len<2u||len%2u!=0u||len>204800u||" << e.GetByOffset("18u") << "!=len){"
      << o.SetByOffset("20u", "103u") << "return;}"
      << "let n=300u*len+20u;if(len>uniforms.input_capacity||n>uniforms.output_capacity){"
      << o.SetByOffset("20u", "102u") << "return;}"
      << o.SetByOffset("2u", "1u") << o.SetByOffset("4u", "n")
      << o.SetByOffset("18u", "n") << o.SetByOffset("21u", "63u");
    return Status::OK();
  }
};

class SignalExtent final : public WebGpuKernel {
 public:
  explicit SignalExtent(const OpKernelInfo& i) : WebGpuKernel{i},
      capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity", 0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x = c.Input<Tensor>(0); const auto* e = c.Input<Tensor>(1);
    const auto* w = c.Input<Tensor>(2); const auto* bias = c.Input<Tensor>(3);
    ORT_RETURN_IF_NOT(Gpu(x) && x->IsDataType<float>() && x->Shape().NumDimensions() == 3 &&
                          x->Shape()[0] == 1 && x->Shape()[1] == 1 && x->Shape()[2] > 0 &&
                          x->Shape()[2] <= 204800 &&
                          Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          Gpu(w) && w->IsDataType<float>() && w->Shape() == TensorShape({9, 1}) &&
                          Gpu(bias) && bias->IsDataType<float>() && bias->Shape() == TensorShape{1} &&
                          capacity_ > 0 && capacity_ <= 61440020,
                      "RW07 signal after1487 exact physical descriptors");
    const auto& limits = c.DeviceLimits();
    const uint64_t groups = (uint64_t(capacity_) + 63) / 64, limit = limits.maxComputeWorkgroupsPerDimension;
    const uint64_t scratch_bytes = 9 * uint64_t(x->Shape()[2]) * 4;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 6 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit >= 9 && groups <= limit * limit &&
                          scratch_bytes <= limits.maxBufferSize && scratch_bytes <= limits.maxStorageBufferBindingSize &&
                          uint64_t(capacity_) * 4 <= limits.maxBufferSize &&
                          uint64_t(capacity_) * 4 <= limits.maxStorageBufferBindingSize,
                      "RW07 signal after1487 resource capacity");
    auto binding = [&](const Tensor* t) -> Status {
      ORT_RETURN_IF_NOT(t->DataRaw() && t->SizeInBytes() > 0 && t->SizeInBytes() <= limits.maxBufferSize &&
                            t->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 signal after1487 binding descriptor");
      M0FullGpuStorageAllocationV1 allocation;
      ORT_RETURN_IF_NOT(c.M0FullGpuDescribeStorageV1(*t,allocation),"RW07 live physical storage allocation");
      const uint64_t actual=allocation.backing_bytes;
      ORT_RETURN_IF_NOT(actual >= t->SizeInBytes() && actual <= limits.maxBufferSize &&
                            actual <= limits.maxStorageBufferBindingSize,
                        "RW07 signal after1487 actual binding backing");
      return Status::OK();
    };
    for (const auto* t : {x, e, w, bias}) ORT_RETURN_IF_ERROR(binding(t));
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity source, carrier, pair, output;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*e, source) &&
                          c.M0FullGpuDescribeResourceV2(*x, carrier), "RW07 signal after1487 actual resources");
    const bool paired = c.M0FullGpuDescribePairV2(*e, *x, pair) && pair.session == session && pair.run == run &&
                        pair.generation == source.generation && pair.data_generation == carrier.generation &&
                        pair.producer_role == 11 && pair.producer_instance == 1012;
    auto* y = c.Output(0, TensorShape{1, capacity_}); auto* out = c.Output(1, TensorShape{32});
    auto phase = c.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape{9, x->Shape()[2]});
    ORT_RETURN_IF_NOT(Gpu(y) && y->IsDataType<float>() && Gpu(out) && out->IsDataType<uint32_t>() && Gpu(&phase),
                      "RW07 signal after1487 output residency/types");
    for (const auto* t : {static_cast<const Tensor*>(y), static_cast<const Tensor*>(out), static_cast<const Tensor*>(&phase)}) ORT_RETURN_IF_ERROR(binding(t));
    for (const auto* t : {x, e, w, bias}) {
      ORT_RETURN_IF_NOT(y->DataRaw() != t->DataRaw() && out->DataRaw() != t->DataRaw() && phase.DataRaw() != t->DataRaw(),
                        "RW07 signal after1487 distinct output/input allocations");
    }
    ORT_RETURN_IF_NOT(y->DataRaw() != out->DataRaw() && y->DataRaw() != phase.DataRaw() && out->DataRaw() != phase.DataRaw(),
                      "RW07 signal after1487 distinct outputs/scratch");
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, 13, 1014, output), "RW07 signal after1487 output claim");
    const m0_extent_abi2::OpaqueOwnership src{session, run, source.generation, carrier.generation, 11, 1012};
    const m0_extent_abi2::OpaqueOwnership dst{session, run, output.generation, output.data_generation, 13, 1014};
    ORT_RETURN_IF_NOT(src.Valid() && dst.Valid(), "RW07 signal after1487 opaque ownership");
    const auto sw = src.UniformWords(), ow = dst.UniformWords();
    const auto source_identity = gsl::span<const uint32_t>{sw.data(), sw.size()}, identity = gsl::span<const uint32_t>{ow.data(), ow.size()};
    const uint32_t pc = uint32_t(x->Shape()[2]), oc = uint32_t(capacity_);
    Guard guard;
    guard.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({out, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({pc, oc, uint32_t(paired), source_identity, identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    Prefix prefix;
    prefix.AddInputs({{x, ProgramTensorMetadataDependency::None}, {out, ProgramTensorMetadataDependency::None}})
        .AddOutput({&phase, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(9)
        .AddUniformVariables({pc, oc, identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(prefix));
    const uint64_t dx = std::min(groups, limit), dy = (groups + dx - 1) / dx;
    Signal signal;
    signal.AddInputs({{x, ProgramTensorMetadataDependency::None}, {&phase, ProgramTensorMetadataDependency::None},
                      {out, ProgramTensorMetadataDependency::None}, {w, ProgramTensorMetadataDependency::None}, {bias, ProgramTensorMetadataDependency::None}})
        .AddOutput({y, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
        .SetDispatchGroupSize(uint32_t(dx), uint32_t(dy)).AddUniformVariables({pc, oc, identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(signal));
    LOGS(c.Logger(), INFO) << "M0_RW07_SIGNAL_AFTER1487_V1 session=" << session << " run=" << run
                          << " source_pair_valid=" << paired << " role=13 instance=1014 GPU_dispatches=3 host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_;
};

class PacketProgram final : public Program<PacketProgram> {
 public:
  PacketProgram() : Program{"RW07SignalAfter1487TerminalPacketV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& p = s.AddOutput("packet", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<18u;i++){" << p.SetByOffset("i", "0u") << "}"
      << p.SetByOffset("0u", "103u")
      << "if(" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=2u){return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.identity", "i", 10) << "){return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){return;}" << p.SetByOffset("0u", "status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("5u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){return;}"
      << "for(var i=6u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return;}}"
      << "let n=" << e.GetByOffset("4u") << ";if(n<620u||n>uniforms.capacity||"
      << "(n-20u)%600u!=0u||n>61440020u||" << e.GetByOffset("18u") << "!=n){return;}"
      << p.SetByOffset("0u", "0u") << p.SetByOffset("1u", "2u")
      << p.SetByOffset("2u", "1u") << p.SetByOffset("4u", "n");
    return Status::OK();
  }
};
class Packet final : public WebGpuKernel {
 public:
  explicit Packet(const OpKernelInfo& i) : WebGpuKernel{i} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e = c.Input<Tensor>(0); const auto* y = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          Gpu(y) && y->IsDataType<float>() && y->Shape().NumDimensions() == 2 &&
                          y->Shape()[0] == 1 && y->Shape()[1] > 0 && y->Shape()[1] <= 61440020,
                      "RW07 signal after1487 packet physical descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*e, *y, pair) && pair.session == session &&
                          pair.run == run && pair.producer_role == 13 && pair.producer_instance == 1014,
                      "RW07 signal after1487 terminal actual pair");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 signal after1487 packet resources");
    auto binding = [&](const Tensor* t) -> Status {
      ORT_RETURN_IF_NOT(t->DataRaw() && t->SizeInBytes() <= limits.maxBufferSize &&
                            t->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 signal after1487 packet binding descriptor");
      const uint64_t size = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
      ORT_RETURN_IF_NOT(size >= t->SizeInBytes() && size <= limits.maxBufferSize && size <= limits.maxStorageBufferBindingSize,
                        "RW07 signal after1487 packet actual backing");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(binding(e)); ORT_RETURN_IF_ERROR(binding(y));
    auto* p = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p) && p->IsDataType<uint32_t>() && p->DataRaw() != e->DataRaw() && p->DataRaw() != y->DataRaw(),
                      "RW07 signal after1487 distinct packet output");
    ORT_RETURN_IF_ERROR(binding(p));
    const m0_extent_abi2::OpaqueOwnership owner{session, run, pair.generation, pair.data_generation, 13, 1014};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 signal after1487 packet opaque ownership");
    const auto words = owner.UniformWords();
    PacketProgram program;
    program.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({p, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[1]), gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p), "RW07 signal after1487 packet map fault registration");
    return Status::OK();
  }
};
}  // namespace
KernelCreateInfo CreateM0FullGpuSignalAfter1487ExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& output) -> Status {
    output = std::make_unique<SignalExtent>(info); return Status::OK();
  };
  return {KernelDefBuilder().SetName("SignalAfter1487ExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuSignalAfter1487TerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& output) -> Status {
    output = std::make_unique<Packet>(info); return Status::OK();
  };
  return {KernelDefBuilder().SetName("SignalAfter1487TerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
