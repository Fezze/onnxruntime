// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original1575..1597. The exported phase expansion deliberately is not atan2.
#include <algorithm>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"

#include "core/providers/webgpu/tensor/m0_fullgpu_polar_after1573_v1_programs.h"
namespace onnxruntime::webgpu {
namespace {
using m0_residual_stages::PhysicalBuffer;
using m0_residual_stages::StorageBudget;
using m0_residual_stages::StorageBudgetWithWindows;
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
using m0_polar_detail::ValidOutput;
using m0_polar_detail::Guard;
using m0_polar_detail::Polar;
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
    const uint64_t bytes = 22 * uint64_t(capacity_) * 4;
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 3 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit > 0 &&
                          bytes <= limits.maxBufferSize,
                      "RW07 polar after1573 resource capacity");
    const uint64_t alignment=limits.minStorageBufferOffsetAlignment;
    ORT_RETURN_IF_NOT(alignment>0&&alignment%4==0,"RW07 polar after1573 window alignment");
    const uint64_t prefix=alignment-4;
    ORT_RETURN_IF_NOT(limits.maxStorageBufferBindingSize>=prefix+88,"RW07 polar after1573 minimum frame window");
    const uint64_t budget=std::min<uint64_t>(limits.maxStorageBufferBindingSize,
        std::max<uint64_t>(8ull*1024*1024,prefix+88));
    const uint64_t dispatch_capacity=std::min<uint64_t>(UINT32_MAX/64,limit*limit)*64;
    const uint32_t frames_per_window=uint32_t(std::min({uint64_t(capacity_),(budget-prefix)/88,dispatch_capacity}));
    ORT_RETURN_IF_NOT(frames_per_window>0,"RW07 polar after1573 positive frame window");
    auto binding = [&](const Tensor* t) -> Status { return PhysicalBuffer(c,t); };
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
    ORT_RETURN_IF_ERROR(StorageBudget(c, {e, out}));
    const uint32_t pc=uint32_t(x->Shape()[1]),oc=uint32_t(capacity_);
    const auto input_begin=[&](uint32_t first){return std::min(first,pc-1);};
    const auto input_view=[&](uint32_t first,uint32_t size){
      const uint32_t begin=input_begin(first),source_frames=std::min(size,pc-begin);
      return ProgramInput::FloatBufferWindow(x,ProgramTensorMetadataDependency::None,TensorShape{source_frames,11,2},
          22u*begin,uint32_t(alignment));
    };
    const auto output_view=[&](uint32_t channel,uint32_t first,uint32_t size){
      return ProgramOutput::FloatBufferWindow(y,ProgramTensorMetadataDependency::None,TensorShape{size},
          channel*oc+first,uint32_t(alignment));
    };
    for(uint32_t channel=0;channel<22;++channel){
      for(uint32_t first=0;first<oc;first+=frames_per_window){
        const uint32_t size=std::min(frames_per_window,oc-first);
        ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,input_view(first,size),output_view(channel,first,size),{out}));
      }
    }
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, 15, 1016, output), "RW07 polar after1573 output claim");
    const m0_extent_abi2::OpaqueOwnership src{session, run, source.generation, carrier.generation, 14, 1015};
    const m0_extent_abi2::OpaqueOwnership dst{session, run, output.generation, output.data_generation, 15, 1016};
    ORT_RETURN_IF_NOT(src.Valid() && dst.Valid(), "RW07 polar after1573 opaque ownership");
    const auto sw = src.UniformWords(), ow = dst.UniformWords();
    const auto source_identity = gsl::span<const uint32_t>{sw.data(), sw.size()}, identity = gsl::span<const uint32_t>{ow.data(), ow.size()};
    Guard guard;
    guard.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({out, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({pc, oc, uint32_t(paired), source_identity, identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    for(uint32_t channel=0;channel<22;++channel){
      for(uint32_t first=0;first<oc;first+=frames_per_window){
        const uint32_t size=std::min(frames_per_window,oc-first);
        const uint64_t groups=(uint64_t(size)+63)/64,dx=std::min(groups,limit),dy=(groups+dx-1)/dx;
        Polar polar;
        polar.AddInput(input_view(first,size)).AddInput({out,ProgramTensorMetadataDependency::None})
            .AddOutput(output_view(channel,first,size)).SetWorkgroupSize(64)
            .SetDispatchGroupSize(uint32_t(dx),uint32_t(dy))
            .AddUniformVariables({pc,oc,first,size,input_begin(first),channel,identity});
        ORT_RETURN_IF_ERROR(c.RunProgram(polar));
      }
    }
    LOGS(c.Logger(), INFO) << "M0_RW07_POLAR_AFTER1573_V1 session=" << session << " run=" << run
                          << " source_pair_valid=" << paired << " role=15 instance=1016 GPU_dispatches="
                          <<1+22*((uint64_t(oc)+frames_per_window-1)/frames_per_window)<<" host_payload_reads=0";
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
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c,e)); ORT_RETURN_IF_ERROR(PhysicalBuffer(c,y));
    auto* p = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p) && p->IsDataType<uint32_t>() && p->DataRaw() != e->DataRaw() && p->DataRaw() != y->DataRaw(),
                      "RW07 polar after1573 distinct packet output");
    ORT_RETURN_IF_ERROR(StorageBudget(c, {e, p}));
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
