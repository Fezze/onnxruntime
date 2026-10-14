// rw07-fullgpu-v1 diagnostic terminal administration with GPU downstream status propagation. Its packet can be
// downloaded after the completed graph Run, never used for intermediate work.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"

namespace onnxruntime::webgpu {
namespace {
class TerminalPacketProgramV2 final : public Program<TerminalPacketProgramV2> {
 public:
  TerminalPacketProgramV2() : Program{"RW07V2TerminalPacket"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rank",ProgramUniformVariableDataType::Uint32},
      {"epoch_low",ProgramUniformVariableDataType::Uint32},
      {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& extent=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& completion=s.AddInput("completion",ShaderUsage::UseValueTypeAlias);
    const auto& packet=s.AddOutput("packet",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    // Packet UINT32[18]: status, rank, eight paired INT64 logical dimensions.
    // It does not expose the live sidecar, its owner or its premise bits.
    f<<"for(var i=0u;i<18u;i++){"<<packet.SetByOffset("i","0u")<<"}"
     <<"if("<<extent.GetByOffset("0u")<<"!=1u||"<<extent.GetByOffset("1u")<<"!=uniforms.rank||"
     <<extent.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<extent.GetByOffset("23u")<<"!=uniforms.epoch_high){"
     <<packet.SetByOffset("0u","103u")<<"return;}"
     <<"if("<<completion.GetByOffset("0u")<<"!=1u||"<<completion.GetByOffset("1u")<<"!=3u||"
     <<completion.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<completion.GetByOffset("23u")<<"!=uniforms.epoch_high){"
     <<packet.SetByOffset("0u","103u")<<"return;}"
     <<"let status="<<extent.GetByOffset("20u")<<";let final_status="<<completion.GetByOffset("20u")<<";"
     // A local failure must be propagated into the closed chain's final extent.
     // Contradictions fail closed; no host aggregation or tensor-value work.
     <<"if(status!=0u&&status!=final_status){"<<packet.SetByOffset("0u","103u")<<"return;}"
     <<"if(final_status!=0u){"<<packet.SetByOffset("0u","final_status")<<"return;}"
     <<"if(("<<completion.GetByOffset("21u")<<"&63u)!=63u){"<<packet.SetByOffset("0u","103u")<<"return;}"
     <<"if(status!=0u){"<<packet.SetByOffset("0u","status")<<"return;}"
     <<"if(("<<extent.GetByOffset("21u")<<"&63u)!=63u){"<<packet.SetByOffset("0u","103u")<<"return;}"
     <<packet.SetByOffset("1u","uniforms.rank")
     <<"for(var i=0u;i<uniforms.rank*2u;i++){"<<packet.SetByOffset("2u+i",extent.GetByOffset("2u+i"))<<"}";
    return Status::OK();
  }
};
class TerminalPacketV2 final : public WebGpuKernel {
 public:
  explicit TerminalPacketV2(const OpKernelInfo& i):WebGpuKernel{i},rank_(i.GetAttrOrDefault<int64_t>("logical_rank",-1)){}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* extent=c.Input<Tensor>(0);const auto* completion=c.Input<Tensor>(1);
    const uint64_t epoch=c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(rank_>=0&&rank_<=rw07::MaxRank&&epoch&&extent&&extent->IsDataType<uint32_t>()&&extent->Shape()==TensorShape{rw07::ExtentWords}&&
                      extent->Location().device.Type()==OrtDevice::GPU,"RW07 terminal extent/owner contract");
    ORT_RETURN_IF_NOT(completion&&completion->IsDataType<uint32_t>()&&completion->Shape()==TensorShape{rw07::ExtentWords}&&
                      completion->Location().device.Type()==OrtDevice::GPU,"RW07 terminal completion sidecar contract");
    const auto& limits=c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=3&&limits.maxBufferSize>=96&&limits.maxStorageBufferBindingSize>=96,
                      "RW07 terminal actual resource limits");
    auto* packet=c.Output(0,TensorShape{18});
    ORT_RETURN_IF_NOT(packet->IsDataType<uint32_t>()&&packet->Location().device.Type()==OrtDevice::GPU,"RW07 terminal administrative GPU producer");
    TerminalPacketProgramV2 p;
    p.AddInputs({{extent,ProgramTensorMetadataDependency::None},{completion,ProgramTensorMetadataDependency::None}}).AddOutput({packet,ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({uint32_t(rank_),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(p));
    LOGS(c.Logger(),INFO)<<"M0_RW07_TERMINAL_PACKET epoch="<<epoch<<" rank="<<rank_<<" source="<<extent->DataRaw()<<" packet="<<packet->DataRaw()
        <<" status_source="<<completion->DataRaw()<<" bytes=72 GPU_producer=1 terminal_only=1";
    return Status::OK();
  }
 private:int64_t rank_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuTerminalPacketV2KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status {out=std::make_unique<TerminalPacketV2>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("ExtentTerminalPacketV2").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
} // namespace onnxruntime::webgpu
