// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original1573 consumes the genuine owned original1571 signal; no readback of N.
#include <algorithm>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_stft1573_extent_v1_programs.h"
namespace onnxruntime::webgpu {
namespace {
using m0_residual_stages::PhysicalBuffer;
using m0_residual_stages::StorageBudget;
using m0_residual_stages::StorageBudgetWithWindows;
ProgramInput Words(const Tensor* t) {
  ProgramInput p(t, ProgramTensorMetadataDependency::None, TensorShape{32}, 1);
  p.var_type = ProgramVariableDataType::Uint32;
  return p;
}
ProgramOutput Words(Tensor* t) {
  ProgramOutput p(t, ProgramTensorMetadataDependency::None, TensorShape{32}, 1);
  p.var_type = ProgramVariableDataType::Uint32;
  return p;
}
using m0_stft_detail::OriginalWindow;
using m0_stft_detail::Roots;
using m0_stft_detail::OwnedStftGuard;
using m0_stft_detail::ValidSpectrum;
class OwnedStft final : public WebGpuKernel {
 public:
  explicit OwnedStft(const OpKernelInfo& info) : WebGpuKernel{info},
      capacity_(info.GetAttrOrDefault<int64_t>("physical_capacity", 0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x = c.Input<Tensor>(0);
    const auto* e = c.Input<Tensor>(1);
    const auto* hop = c.Input<Tensor>(2);
    const auto* window = c.Input<Tensor>(3);
    const auto* frame = c.Input<Tensor>(4);
    ORT_RETURN_IF_NOT(capacity_ > 0 && capacity_ <= 12288001 && x && x->IsDataType<float>() &&
                          x->Shape().NumDimensions() == 2 && x->Shape()[0] == 1 && x->Shape()[1] > 0 &&
                          x->Shape()[1] <= 61440020 && e && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          hop && hop->IsDataType<int64_t>() && hop->Shape().NumDimensions() == 0 &&
                          frame && frame->IsDataType<int64_t>() && frame->Shape().NumDimensions() == 0 &&
                          window && window->IsDataType<float>() && window->Shape() == TensorShape{20},
                      "RW07 owned STFT original signal/parameter descriptors");
    for (const auto* input : {x, e, hop, window, frame}) ORT_RETURN_IF_ERROR(PhysicalBuffer(c, input));
    const auto& limits = c.DeviceLimits();
    const uint64_t elements = uint64_t(capacity_) * 22;
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 5 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= m0_stft_detail::kOriginalBluesteinFftSamplesV1 &&
                          limits.maxComputeWorkgroupStorageSize>=m0_stft_detail::kOriginalBluesteinWorkgroupBytesV1 && limit > 0 &&
                          elements <= UINT32_MAX && elements * sizeof(float) <= limits.maxBufferSize,
                      "RW07 owned STFT physical output/dispatch limits");
    const uint64_t alignment=limits.minStorageBufferOffsetAlignment;
    ORT_RETURN_IF_NOT(alignment>0&&alignment%sizeof(float)==0,"RW07 STFT window alignment");
    const uint32_t frames_per_window=m0_stft_detail::StftFramesPerWindowV1(limits,uint32_t(capacity_));
    ORT_RETURN_IF_NOT(frames_per_window>0,"RW07 STFT bounded work cohort");
    const uint32_t signal_capacity=uint32_t(x->Shape()[1]),frame_capacity=uint32_t(capacity_);
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity sid, data, pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*e, sid) && c.M0FullGpuDescribeResourceV2(*x, data),
                      "RW07 owned STFT live input resources");
    const bool valid = c.M0FullGpuDescribePairV2(*e, *x, pair) && pair.session == session && pair.run == run &&
        pair.generation == sid.generation && pair.data_generation == data.generation && pair.producer_role == 13 && pair.producer_instance == 1014;
    auto* y = c.Output(0, TensorShape{1, capacity_, 11, 2});
    auto* out = c.Output(1, TensorShape{32});
    ORT_RETURN_IF_NOT(y && y->IsDataType<float>() && out && out->IsDataType<uint32_t>(), "RW07 owned STFT output types");
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c, y));
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c, out));
    ORT_RETURN_IF_NOT(y->DataRaw() != out->DataRaw(), "RW07 owned STFT distinct outputs");
    for (const auto* input : {x, e, hop, window, frame}) {
      ORT_RETURN_IF_NOT(input->DataRaw() != y->DataRaw() && input->DataRaw() != out->DataRaw(), "RW07 owned STFT source/output alias");
    }
    ORT_RETURN_IF_ERROR(StorageBudget(c, {e, hop, window, frame, out}));
    const auto signal_begin=[&](uint32_t first){return std::min<uint64_t>(5ull*first,uint64_t(signal_capacity)-1);};
    const auto signal_view=[&](uint32_t first,uint32_t count){
      const uint64_t begin=signal_begin(first);
      const uint64_t samples=std::min<uint64_t>(5ull*(count-1)+20,uint64_t(signal_capacity)-begin);
      return ProgramInput::FloatBufferWindow(x,ProgramTensorMetadataDependency::None,TensorShape{int64_t(samples)},
          uint32_t(begin),uint32_t(alignment));
    };
    const auto spectrum_view=[&](uint32_t first,uint32_t count){
      return ProgramOutput::FloatBufferWindow(y,ProgramTensorMetadataDependency::None,TensorShape{count,11,2},
          22u*first,uint32_t(alignment));
    };
    for(uint32_t first=0;first<frame_capacity;first+=frames_per_window){
      const uint32_t count=std::min(frames_per_window,frame_capacity-first);
      ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,signal_view(first,count),spectrum_view(first,count),{window,out}));
    }
    M0FullGpuAllocationIdentity output;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, 14, 1015, output), "RW07 owned STFT spectrum pair claim");
    const m0_extent_abi2::OpaqueOwnership source_owner{session, run, sid.generation, data.generation, 13, 1014};
    const m0_extent_abi2::OpaqueOwnership owner{session, run, output.generation, output.data_generation, 14, 1015};
    ORT_RETURN_IF_NOT(source_owner.Valid() && owner.Valid(), "RW07 owned STFT opaque identities");
    const auto source_words = source_owner.UniformWords(), words = owner.UniformWords();
    const auto span = [](const auto& a) { return gsl::span<const uint32_t>{a.data(), a.size()}; };
    auto scalar_words = [](const Tensor* t) {
      ProgramInput value(t, ProgramTensorMetadataDependency::None, TensorShape{2}, 1);
      value.var_type = ProgramVariableDataType::Uint32;
      return value;
    };
    OwnedStftGuard guard;
    guard.AddInputs({Words(e), scalar_words(hop), {window, ProgramTensorMetadataDependency::None}, scalar_words(frame)})
        .AddOutput(Words(out)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({signal_capacity, frame_capacity, uint32_t(valid), span(source_words), span(words)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    ORT_RETURN_IF_ERROR(m0_stft_detail::RunOwnedStftWindowsV1(c,*x,*window,*out,*y,span(words)));
    LOGS(c.Logger(), INFO) << "M0_RW07_STFT1573_EXTENT_V1 session=" << session << " run=" << run
        << " source_pair_valid=" << valid << " role=14 instance=1015 source_capacity=" << signal_capacity
        << " output_capacity=" << frame_capacity << " GPU_dispatches="<<(1+(frame_capacity+frames_per_window-1)/frames_per_window)
        << " frames_per_window="<<frames_per_window<<" host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_;
};
class OwnedStftPacketProgram final : public Program<OwnedStftPacketProgram> {
 public:
  OwnedStftPacketProgram() : Program{"RW07STFT1573OwnedTerminalV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"frame_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& p = s.AddOutput("packet", ShaderUsage::UseValueTypeAlias);
    ValidSpectrum(s, e);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<18u;i++){" << p.SetByOffset("i", "0u") << "}"
      << p.SetByOffset("0u", "103u")
      << "if(" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=4u){return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.identity", "i", 10) << "){return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){return;}" << p.SetByOffset("0u", "status") << "return;}"
      << "if(!valid_spectrum()){return;}" << p.SetByOffset("0u", "0u") << p.SetByOffset("1u", "4u")
      << "for(var i=0u;i<8u;i++){" << p.SetByOffset("2u+i", e.GetByOffset("2u+i")) << "}";
    return Status::OK();
  }
};
class OwnedStftPacket final : public WebGpuKernel {
 public:
  explicit OwnedStftPacket(const OpKernelInfo& info) : WebGpuKernel{info} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e = c.Input<Tensor>(0);
    const auto* y = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(e && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} && y && y->IsDataType<float>() &&
                          y->Shape().NumDimensions() == 4 && y->Shape()[0] == 1 && y->Shape()[1] > 0 &&
                          y->Shape()[1] <= 12288001 && y->Shape()[2] == 11 && y->Shape()[3] == 2,
                      "RW07 owned STFT terminal original descriptors");
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c, e));
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c, y));
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 owned STFT terminal dispatch limits");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*e, *y, pair) && pair.session == session &&
                          pair.run == run && pair.producer_role == 14 && pair.producer_instance == 1015,
                      "RW07 owned STFT terminal actual spectrum pair");
    auto* packet = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(packet && packet->IsDataType<uint32_t>(), "RW07 owned STFT terminal packet type");
    ORT_RETURN_IF_ERROR(StorageBudget(c, {e, packet}));
    ORT_RETURN_IF_NOT(packet->DataRaw() != e->DataRaw() && packet->DataRaw() != y->DataRaw(),
                      "RW07 owned STFT terminal distinct packet");
    const m0_extent_abi2::OpaqueOwnership owner{session, run, pair.generation, pair.data_generation, 14, 1015};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 owned STFT terminal opaque pair");
    const auto words = owner.UniformWords();
    OwnedStftPacketProgram p;
    p.AddInput(Words(e)).AddOutput({packet, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[1]), gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(p));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*packet), "RW07 owned STFT terminal map fault hook");
    return Status::OK();
  }
};
}  // namespace
KernelCreateInfo CreateM0FullGpuSTFT1573ExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<OwnedStft>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("STFT1573ExtentV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuSTFT1573TerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<OwnedStftPacket>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("STFT1573TerminalPacketV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
