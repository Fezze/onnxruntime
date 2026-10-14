// rw07-fullgpu-v1 diagnostic terminal administration only. Its packet can be
// downloaded after the completed graph Run, never used for intermediate work.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"

namespace onnxruntime::webgpu {
namespace {
class TerminalPacketProgram final : public Program<TerminalPacketProgram> {
 public:
  TerminalPacketProgram() : Program{"RW07V1TerminalPacket"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rank",ProgramUniformVariableDataType::Uint32},
      {"epoch_low",ProgramUniformVariableDataType::Uint32},
      {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& extent=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& packet=s.AddOutput("packet",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    // Packet UINT32[18]: status, rank, eight paired INT64 logical dimensions.
    // It does not expose the live sidecar, its owner or its premise bits.
    f<<"for(var i=0u;i<18u;i++){"<<packet.SetByOffset("i","0u")<<"}"
     <<"if("<<extent.GetByOffset("0u")<<"!=1u||"<<extent.GetByOffset("1u")<<"!=uniforms.rank||"
     <<extent.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<extent.GetByOffset("23u")<<"!=uniforms.epoch_high){"
     <<packet.SetByOffset("0u","103u")<<"return;}"
     <<"let status="<<extent.GetByOffset("20u")<<";"
     <<"if(status!=0u){"<<packet.SetByOffset("0u","status")<<"return;}"
     <<"if(("<<extent.GetByOffset("21u")<<"&63u)!=63u){"<<packet.SetByOffset("0u","103u")<<"return;}"
     <<packet.SetByOffset("1u","uniforms.rank")
     <<"for(var i=0u;i<uniforms.rank*2u;i++){"<<packet.SetByOffset("2u+i",extent.GetByOffset("2u+i"))<<"}";
    return Status::OK();
  }
};
class TerminalPacket final : public WebGpuKernel {
 public:
  explicit TerminalPacket(const OpKernelInfo& i):WebGpuKernel{i},rank_(i.GetAttrOrDefault<int64_t>("logical_rank",-1)){}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* extent=c.Input<Tensor>(0);
    const uint64_t epoch=c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(rank_>=0&&rank_<=rw07::MaxRank&&epoch&&extent&&extent->IsDataType<uint32_t>()&&extent->Shape()==TensorShape{rw07::ExtentWords}&&
                      extent->Location().device.Type()==OrtDevice::GPU,"RW07 terminal extent/owner contract");
    const auto& limits=c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=2&&limits.maxBufferSize>=96&&limits.maxStorageBufferBindingSize>=96,
                      "RW07 terminal actual resource limits");
    auto* packet=c.Output(0,TensorShape{18});
    ORT_RETURN_IF_NOT(packet->IsDataType<uint32_t>()&&packet->Location().device.Type()==OrtDevice::GPU,"RW07 terminal administrative GPU producer");
    TerminalPacketProgram p;
    p.AddInput({extent,ProgramTensorMetadataDependency::None}).AddOutput({packet,ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({uint32_t(rank_),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(p));
    LOGS(c.Logger(),INFO)<<"M0_RW07_TERMINAL_PACKET epoch="<<epoch<<" rank="<<rank_<<" source="<<extent->DataRaw()<<" packet="<<packet->DataRaw()
        <<" bytes=72 GPU_producer=1 terminal_only=1";
    return Status::OK();
  }
 private:int64_t rank_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuTerminalPacketKernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status {out=std::make_unique<TerminalPacket>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("ExtentTerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
} // namespace onnxruntime::webgpu
