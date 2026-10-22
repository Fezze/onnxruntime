// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"
namespace onnxruntime::webgpu {
namespace {
class DecoderResidualPacketProgram final : public Program<DecoderResidualPacketProgram> {
 public:
  DecoderResidualPacketProgram() : Program{"RW07DecoderResidualTerminalPacketV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& extent = shader.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& packet = shader.AddOutput("packet", ShaderUsage::UseValueTypeAlias);
    auto& f = shader.MainFunctionBody();
    f << "for(var i=0u;i<18u;i++){" << packet.SetByOffset("i", "0u") << "}"
      << packet.SetByOffset("0u", "103u")
      << "if(" << extent.GetByOffset("0u") << "!=2u||" << extent.GetByOffset("1u") << "!=3u){return;}"
      << "for(var i=0u;i<10u;i++){if(" << extent.GetByOffset("22u+i")
      << "!=" << GetElementAt("uniforms.identity", "i", 10) << "){return;}}"
      << "let status=" << extent.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}"
      << "for(var i=2u;i<20u;i++){if(" << extent.GetByOffset("i") << "!=0u){return;}}"
      << "if(" << extent.GetByOffset("21u") << "!=0u){return;}"
      << packet.SetByOffset("0u", "status") << "return;}"
      << "if(" << extent.GetByOffset("2u") << "!=1u||" << extent.GetByOffset("3u") << "!=0u||"
      << extent.GetByOffset("4u") << "!=uniforms.channels||" << extent.GetByOffset("5u") << "!=0u||"
      << extent.GetByOffset("7u") << "!=0u||" << extent.GetByOffset("21u") << "!=63u){return;}"
      << "for(var i=8u;i<18u;i++){if(" << extent.GetByOffset("i") << "!=0u){return;}}"
      << "let d=" << extent.GetByOffset("6u") << ";if(d==0u||d>uniforms.capacity||(uniforms.channels==512u&&d%2u!=0u)||"
      << extent.GetByOffset("18u") << "!=uniforms.channels*d||" << extent.GetByOffset("19u") << "!=0u){return;}"
      << packet.SetByOffset("0u", "0u") << packet.SetByOffset("1u", "3u")
      << "for(var i=0u;i<6u;i++){" << packet.SetByOffset("2u+i", extent.GetByOffset("2u+i")) << "}";
    return Status::OK();
  }
};
class DecoderResidualPacket final : public WebGpuKernel {
 public:
  explicit DecoderResidualPacket(const OpKernelInfo& info) : WebGpuKernel{info},
                                                         block_(info.GetAttrOrDefault<int64_t>("block", -1)), role_(20 + block_), instance_(1021 + block_) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* extent = c.Input<Tensor>(0);
    const auto* y = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT((block_ >= 0 && block_ <= 4) && extent && extent->IsDataType<uint32_t>() &&
                          extent->Shape() == TensorShape{m0_extent_abi2::Words} && y && y->IsDataType<float>() &&
                          y->Shape().NumDimensions() == 3 && y->Shape()[0] == 1 &&
                          y->Shape()[1] == (block_ == 4 ? 512 : 1024) &&
                          y->Shape()[2] > 0 && y->Shape()[2] <= (block_ == 4 ? 204800 : 102400) &&
                          extent->Location().device.Type() == OrtDevice::GPU && y->Location().device.Type() == OrtDevice::GPU,
                      "RW07 decoder residual terminal exact cut descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity identity;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*extent, *y, identity) &&
                          identity.session == session && identity.run == run && identity.producer_role == uint32_t(role_) &&
                          identity.producer_instance == uint32_t(instance_),
                      "RW07 decoder residual terminal actual producer pair");
    const auto& limits = c.DeviceLimits();
    for(const auto* input:{extent,y})ORT_RETURN_IF_ERROR(m0_residual_stages::PhysicalBuffer(c,input));
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxBufferSize >= 128 &&
                          limits.maxStorageBufferBindingSize >= 128 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 decoder residual terminal device limits");
    auto* packet = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(packet->IsDataType<uint32_t>() && packet->Location().device.Type() == OrtDevice::GPU &&
                          packet->DataRaw() != extent->DataRaw() && packet->DataRaw() != y->DataRaw(),
                      "RW07 decoder residual terminal packet allocation");
    ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudget(c,{extent,packet}));
    const m0_extent_abi2::OpaqueOwnership ownership{session, run, identity.generation,
                                                    identity.data_generation, uint32_t(role_), uint32_t(instance_)};
    ORT_RETURN_IF_NOT(ownership.Valid(), "RW07 decoder residual terminal identity");
    const auto words = ownership.UniformWords();
    DecoderResidualPacketProgram program;
    program.AddInput({extent, ProgramTensorMetadataDependency::None})
        .AddOutput({packet, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1)
        .SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[1]), uint32_t(y->Shape()[2]), gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*packet), "RW07 decoder residual terminal map fault registration");
    return Status::OK();
  }

 private:
  const int64_t block_, role_, instance_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuDecoderResidualTerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<DecoderResidualPacket>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("DecoderResidualTerminalPacketV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
