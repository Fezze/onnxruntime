// rw07-fullgpu-v1 diagnostic terminal administration with GPU downstream status propagation. Its packet can be
// downloaded after the completed graph Run, never used for intermediate work.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"

namespace onnxruntime::webgpu {
namespace {
class TerminalPacketProgramV4 final : public Program<TerminalPacketProgramV4> {
 public:
  TerminalPacketProgramV4() : Program{"RW07V4TerminalPacket"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rank",ProgramUniformVariableDataType::Uint32},
      {"completion_rank",ProgramUniformVariableDataType::Uint32},
      {"epoch_low",ProgramUniformVariableDataType::Uint32},
      {"epoch_high",ProgramUniformVariableDataType::Uint32},
      {"run_low",ProgramUniformVariableDataType::Uint32},
      {"run_high",ProgramUniformVariableDataType::Uint32},
      {"source_low",ProgramUniformVariableDataType::Uint32},
      {"source_high",ProgramUniformVariableDataType::Uint32},
      {"carrier_low",ProgramUniformVariableDataType::Uint32},
      {"carrier_high",ProgramUniformVariableDataType::Uint32},
      {"source_role",ProgramUniformVariableDataType::Uint32},
      {"source_instance",ProgramUniformVariableDataType::Uint32},
      {"final_low",ProgramUniformVariableDataType::Uint32},
      {"final_high",ProgramUniformVariableDataType::Uint32},
      {"audio_low",ProgramUniformVariableDataType::Uint32},
      {"audio_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& extent=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& completion=s.AddInput("completion",ShaderUsage::UseValueTypeAlias);
    const auto& packet=s.AddOutput("packet",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    auto guard=[&](const auto& e,const char* side,const char* carrier,
                   const char* role,const char* instance){
      auto& body=s.MainFunctionBody();
      body << "if(" << e.GetByOffset("24u") << "!=uniforms.run_low||"
        << e.GetByOffset("25u") << "!=uniforms.run_high||"
        << e.GetByOffset("26u") << "!=uniforms." << side << "_low||"
        << e.GetByOffset("27u") << "!=uniforms." << side << "_high||"
        << e.GetByOffset("28u") << "!=uniforms." << carrier << "_low||"
        << e.GetByOffset("29u") << "!=uniforms." << carrier << "_high||"
        << e.GetByOffset("30u") << "!=" << role << "||"
        << e.GetByOffset("31u") << "!=" << instance << "){"
        << packet.SetByOffset("0u","103u") << "return;}";
    };
    // Packet UINT32[18]: status, rank, eight paired INT64 logical dimensions.
    // It does not expose the live sidecar, its owner or its premise bits.
    f<<"for(var i=0u;i<18u;i++){"<<packet.SetByOffset("i","0u")<<"}"
     <<"if("<<extent.GetByOffset("0u")<<"!=2u||"<<extent.GetByOffset("1u")<<"!=uniforms.rank||"
     <<extent.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<extent.GetByOffset("23u")<<"!=uniforms.epoch_high){"
     <<packet.SetByOffset("0u","103u")<<"return;}"
     <<"if("<<completion.GetByOffset("0u")<<"!=2u||"<<completion.GetByOffset("1u")<<"!=uniforms.completion_rank||"
     <<completion.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<completion.GetByOffset("23u")<<"!=uniforms.epoch_high){"
     <<packet.SetByOffset("0u","103u")<<"return;}"
;
    guard(extent,"source","carrier","uniforms.source_role","uniforms.source_instance");
    guard(completion,"final","audio","3u","1003u");
    f <<"let status="<<extent.GetByOffset("20u")<<";let final_status="<<completion.GetByOffset("20u")<<";"
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
class TerminalPacketV4 final : public WebGpuKernel {
 public:
  explicit TerminalPacketV4(const OpKernelInfo& i):WebGpuKernel{i},rank_(i.GetAttrOrDefault<int64_t>("logical_rank",-1)),completion_rank_(i.GetAttrOrDefault<int64_t>("completion_rank",-1)),
      role_(i.GetAttrOrDefault<int64_t>("source_role",-1)),instance_(i.GetAttrOrDefault<int64_t>("source_instance",-1)),
      final_role_(i.GetAttrOrDefault<int64_t>("completion_role",-1)),final_instance_(i.GetAttrOrDefault<int64_t>("completion_instance",-1)){}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* extent=c.Input<Tensor>(0);const auto* carrier=c.Input<Tensor>(1);
    const auto* completion=c.Input<Tensor>(2);const auto* audio=c.Input<Tensor>(3);
    ORT_RETURN_IF_NOT(m0_extent_abi2::CertifiedProducer(uint32_t(role_),uint32_t(instance_))&&final_role_==3&&final_instance_==1003&&completion_rank_==1, "RW07 terminal certified producer edge");
    const uint64_t run=c.KernelContext().M0FullGpuExtentRunIdV2();
    const uint64_t epoch=c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(rank_>=1&&rank_<=rw07::MaxRank&&(completion_rank_==1||completion_rank_==3)&&epoch&&run&&extent&&extent->IsDataType<uint32_t>()&&extent->Shape()==TensorShape{m0_extent_abi2::Words}&&
                      extent->Location().device.Type()==OrtDevice::GPU,"RW07 terminal extent/owner contract");
    ORT_RETURN_IF_NOT(completion&&completion->IsDataType<uint32_t>()&&completion->Shape()==TensorShape{m0_extent_abi2::Words}&&
                      completion->Location().device.Type()==OrtDevice::GPU,"RW07 terminal completion sidecar contract");
    M0FullGpuAllocationIdentity source_id,carrier_id,final_id,audio_id;
    ORT_RETURN_IF_NOT(carrier&&audio&&c.M0FullGpuDescribeResourceV2(*extent,source_id)&&
      c.M0FullGpuDescribeResourceV2(*carrier,carrier_id)&&c.M0FullGpuDescribeResourceV2(*completion,final_id)&&
      c.M0FullGpuDescribeResourceV2(*audio,audio_id),"RW07 terminal actual allocation descriptors");
    const auto& limits=c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=3&&limits.maxBufferSize>=128&&limits.maxStorageBufferBindingSize>=128,
                      "RW07 terminal actual resource limits");
    auto* packet=c.Output(0,TensorShape{18});
    ORT_RETURN_IF_NOT(packet->IsDataType<uint32_t>()&&packet->Location().device.Type()==OrtDevice::GPU,"RW07 terminal administrative GPU producer");
    TerminalPacketProgramV4 p;
    p.AddInputs({{extent,ProgramTensorMetadataDependency::None},{completion,ProgramTensorMetadataDependency::None}}).AddOutput({packet,ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({uint32_t(rank_),uint32_t(completion_rank_),uint32_t(epoch),uint32_t(epoch>>32),uint32_t(run),uint32_t(run>>32),
          uint32_t(source_id.generation),uint32_t(source_id.generation>>32),uint32_t(carrier_id.generation),uint32_t(carrier_id.generation>>32),
          uint32_t(role_),uint32_t(instance_),uint32_t(final_id.generation),uint32_t(final_id.generation>>32),
          uint32_t(audio_id.generation),uint32_t(audio_id.generation>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(p));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*packet), "RW07 actual terminal map-fault registration");
    LOGS(c.Logger(),INFO)<<"M0_RW07_TERMINAL_PACKET epoch="<<epoch<<" rank="<<rank_<<" source="<<extent->DataRaw()<<" packet="<<packet->DataRaw()
        <<" run="<<run<<" source_generation="<<source_id.generation<<" carrier_generation="<<carrier_id.generation
        <<" final_generation="<<final_id.generation<<" audio_generation="<<audio_id.generation
        <<" status_source="<<completion->DataRaw()<<" bytes=72 GPU_producer=1 terminal_only=1";
    return Status::OK();
  }
 private:int64_t rank_,completion_rank_,role_,instance_,final_role_,final_instance_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuTerminalPacketV4KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status {out=std::make_unique<TerminalPacketV4>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("ExtentTerminalPacketV4").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
} // namespace onnxruntime::webgpu
