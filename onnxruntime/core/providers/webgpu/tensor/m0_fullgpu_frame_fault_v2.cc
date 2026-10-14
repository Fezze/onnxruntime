// Signed bounded diagnostic only: selects an actual current or retained pair.
// Administrative selector does not inspect or manufacture any GPU payload.
#include "m0_fullgpu_frame_projection_v2_common.h"
namespace onnxruntime::webgpu {
namespace {
// Intentional signed diagnostic corruption. Bind the sidecar once as writable
// output; no overlapping read/write bindings and no host payload inspection.
class FrameSidecarMutation final:public Program<FrameSidecarMutation> {
 public:
  FrameSidecarMutation():Program{"RW07V2FrameExtentFaultMutation"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"mode",ProgramUniformVariableDataType::Uint32},
    {"run_low",ProgramUniformVariableDataType::Uint32},
    {"run_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override {
    const auto& e=s.AddOutput("selected_sidecar",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    // Every other word retains its actual prior GPU-produced value.
    f<<"if(uniforms.mode==2u){"<<e.SetByOffset("30u","2u")
     <<e.SetByOffset("31u","1002u")<<"return;}"
     <<"if(uniforms.mode==3u){"<<e.SetByOffset("31u","1002u")<<"return;}"
     <<"if(uniforms.mode==4u){"<<e.SetByOffset("24u","uniforms.run_low")
     <<e.SetByOffset("25u","uniforms.run_high")<<"return;}";
    return Status::OK();
  }
};
class FrameProjectionFault final:public WebGpuKernel {
 public:
  explicit FrameProjectionFault(const OpKernelInfo& i):WebGpuKernel{i},attrs_{i} {}
  Status ComputeInternal(ComputeContext& c)const override {
    const uint32_t mode=c.M0FullGpuFaultModeV2();
    ORT_RETURN_IF_NOT(mode<=4||mode==10||mode==11||mode==12||mode==20,"RW07 closed actual retained-pair fault mode");
    const bool override_pair=mode==1||mode==4;
    const auto* e=c.Input<Tensor>(override_pair?3:0);
    const auto* total=c.Input<Tensor>(override_pair?4:1);
    const auto* x=c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(e&&total&&x,"RW07 explicit current/retained carrier dependencies");
    LOGS(c.Logger(),INFO)<<"M0_RW07_FRAME_FAULT_SELECTOR mode="<<mode
      <<" selected_sidecar="<<e->DataRaw()<<" selected_carrier="<<total->DataRaw();
    if(mode>=2&&mode<=4) {
      ORT_RETURN_IF_NOT(m0_frame_v2::Gpu(e)&&e->IsDataType<uint32_t>()&&
        e->Shape()==TensorShape({m0_extent_abi2::Words}),"RW07 actual mutation sidecar descriptor");
      const auto& limits=c.DeviceLimits();
      ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=1&&limits.maxBufferSize>=m0_extent_abi2::Bytes&&
        limits.maxStorageBufferBindingSize>=m0_extent_abi2::Bytes&&limits.maxComputeWorkgroupsPerDimension>=1&&
        limits.maxComputeWorkgroupSizeX>=1&&limits.maxComputeInvocationsPerWorkgroup>=1,
        "RW07 actual mutation resource/dispatch limits");
      const uint64_t run=c.KernelContext().M0FullGpuExtentRunIdV2();
      M0FullGpuAllocationIdentity identity;
      ORT_RETURN_IF_NOT(run&&c.M0FullGpuDescribeResourceV2(*e,identity),"RW07 actual live mutation allocation");
      FrameSidecarMutation mutation;
      mutation.AddOutput(m0_frame_v2::Words(const_cast<Tensor*>(e))).SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({mode,uint32_t(run),uint32_t(run>>32)});
      ORT_RETURN_IF_ERROR(c.RunProgram(mutation));
      LOGS(c.Logger(),INFO)<<"M0_RW07_FRAME_FAULT_MUTATION mode="<<mode<<" handle="<<e->DataRaw()
        <<" generation="<<identity.generation<<" current_run="<<run
        <<" words="<<(mode==2?"30,31":mode==3?"31":"24,25")
        <<" literal_values="<<(mode==2?"2,1002":mode==3?"1002":"current_run_low,current_run_high")
        <<" GPU_dispatch=RW07V2FrameExtentFaultMutation host_payload_reads=0";
    }
    return m0_frame_v2::Compute(c,e,total,x,attrs_,"FrameProjectionFaultExtentV2");
  }
 private:m0_frame_v2::FrameAttributesV2 attrs_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuFrameProjectionFaultExtentV2KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status {
    out=std::make_unique<FrameProjectionFault>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("FrameProjectionFaultExtentV2").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
    .Provider(kWebGpuExecutionProvider).Build(),fn};
}
} // namespace onnxruntime::webgpu
