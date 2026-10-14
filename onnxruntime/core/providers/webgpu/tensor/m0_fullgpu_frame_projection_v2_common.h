// Shared normal/fault FrameV2 guard and arithmetic. No host tensor values.
#pragma once
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_resource_ownership.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
// New identity: ABI2 is32words; retained schema1 source is untouched.
namespace onnxruntime::webgpu::m0_frame_v2 {
inline ProgramInput Words(const Tensor* t) {
  ProgramInput p(t,ProgramTensorMetadataDependency::None,TensorShape{m0_extent_abi2::Words},1);
  p.var_type=ProgramVariableDataType::Uint32;return p;
}
inline ProgramOutput Words(Tensor* t) {
  ProgramOutput p(t,ProgramTensorMetadataDependency::None,TensorShape{m0_extent_abi2::Words},1);
  p.var_type=ProgramVariableDataType::Uint32;return p;
}
inline bool Gpu(const Tensor* t){return t&&t->Location().device.Type()==OrtDevice::GPU;}
class ProjectionProgram final:public Program<ProjectionProgram>{
 public:
  ProjectionProgram():Program{"RW07V2FrameProjectionExtent"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"multiple",ProgramUniformVariableDataType::Uint32},
    {"offset",ProgramUniformVariableDataType::Uint32},
    {"capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32},
    {"run_low",ProgramUniformVariableDataType::Uint32},
    {"run_high",ProgramUniformVariableDataType::Uint32},
    {"source_generation_low",ProgramUniformVariableDataType::Uint32},
    {"source_generation_high",ProgramUniformVariableDataType::Uint32},
    {"source_carrier_low",ProgramUniformVariableDataType::Uint32},
    {"source_carrier_high",ProgramUniformVariableDataType::Uint32},
    {"output_generation_low",ProgramUniformVariableDataType::Uint32},
    {"output_generation_high",ProgramUniformVariableDataType::Uint32},
    {"output_carrier_low",ProgramUniformVariableDataType::Uint32},
    {"output_carrier_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& e=s.AddInput("root",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    f<<"for(var i=0u;i<32u;i++){"<<o.SetByOffset("i","0u")<<"}"
     <<o.SetByOffset("0u","2u")<<o.SetByOffset("1u","3u")
     <<o.SetByOffset("22u","uniforms.epoch_low")<<o.SetByOffset("23u","uniforms.epoch_high")
     <<o.SetByOffset("24u","uniforms.run_low")<<o.SetByOffset("25u","uniforms.run_high")
     <<o.SetByOffset("26u","uniforms.output_generation_low")<<o.SetByOffset("27u","uniforms.output_generation_high")
     <<o.SetByOffset("28u","uniforms.output_carrier_low")<<o.SetByOffset("29u","uniforms.output_carrier_high")
     <<o.SetByOffset("30u","2u")<<o.SetByOffset("31u","1002u")
     <<"if("
     <<e.GetByOffset("24u")<<"!=uniforms.run_low||"<<e.GetByOffset("25u")<<"!=uniforms.run_high||"
     <<e.GetByOffset("26u")<<"!=uniforms.source_generation_low||"<<e.GetByOffset("27u")<<"!=uniforms.source_generation_high||"
     <<e.GetByOffset("28u")<<"!=uniforms.source_carrier_low||"<<e.GetByOffset("29u")<<"!=uniforms.source_carrier_high||"
     <<e.GetByOffset("30u")<<"!=1u||"<<e.GetByOffset("31u")<<"!=1001u||"
     <<e.GetByOffset("0u")<<"!=2u||"<<e.GetByOffset("1u")<<"!=1u||"
     <<e.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<e.GetByOffset("23u")<<"!=uniforms.epoch_high){"
     <<o.SetByOffset("20u","103u")<<"return;}"
     <<"let status="<<e.GetByOffset("20u")<<";if(status!=0u){"<<o.SetByOffset("20u","status")<<"return;}"
     <<"let d="<<e.GetByOffset("2u")<<";if(d==0u||d>51200u||"<<e.GetByOffset("3u")<<"!=0u||"
     <<e.GetByOffset("18u")<<"!=d||"<<e.GetByOffset("19u")<<"!=0u||("<<e.GetByOffset("21u")<<"&63u)!=63u){"
     <<o.SetByOffset("20u","103u")<<"return;}"
     <<"let len=d*uniforms.multiple+uniforms.offset;if(len>uniforms.capacity){"<<o.SetByOffset("20u","102u")<<"return;}"
     <<o.SetByOffset("2u","1u")<<o.SetByOffset("4u","uniforms.channels")
     <<o.SetByOffset("6u","len")<<o.SetByOffset("18u","uniforms.channels*len")
     <<o.SetByOffset("21u","63u");
    return Status::OK();
  }
};
struct FrameAttributesV2 {
  explicit FrameAttributesV2(const OpKernelInfo& i)
      : role(i.GetAttrOrDefault<int64_t>("role",-1)),
        source_role(i.GetAttrOrDefault<int64_t>("source_role",-1)),
        source_instance(i.GetAttrOrDefault<int64_t>("source_instance",-1)),
        semantic_role(i.GetAttrOrDefault<int64_t>("semantic_role",-1)),
        producer_instance(i.GetAttrOrDefault<int64_t>("producer_instance",-1)) {}
  const int64_t role,source_role,source_instance,semantic_role,producer_instance;
};
inline Status Compute(ComputeContext& c,const Tensor* e,const Tensor* total,const Tensor* x,
                      const FrameAttributesV2& attrs,const char* producer_label) {
    ORT_RETURN_IF_NOT(attrs.source_role==1&&attrs.source_instance==1001&&attrs.semantic_role==2&&attrs.producer_instance==1002,"RW07 certified frame projection producer/role");
    ORT_RETURN_IF_NOT(attrs.role>=0&&attrs.role<=1,"RW07 closed projection role");
    const int64_t channels=22;
    const uint32_t multiple=attrs.role==0?120:1;
    const uint32_t offset=attrs.role==0?1:0;
    ORT_RETURN_IF_NOT(total&&total->IsDataType<int64_t>()&&total->Shape().NumDimensions()==0&&Gpu(total),"RW07 root total scalar explicit GPU carrier");
    ORT_RETURN_IF_NOT(x&&x->IsDataType<float>()&&x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&
      x->Shape()[1]==channels&&x->Shape()[2]>0&&x->Shape()[2]<=6144001&&
      e&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape({m0_extent_abi2::Words}),"RW07 projection physical descriptors");
    const uint64_t epoch=c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    const auto& l=c.DeviceLimits();
    ORT_RETURN_IF_NOT(epoch&&Gpu(e)&&Gpu(x)&&l.maxStorageBuffersPerShaderStage>=2&&
      l.maxBufferSize>=128&&l.maxStorageBufferBindingSize>=128&&l.maxComputeWorkgroupsPerDimension>=1&&
      x->SizeInBytes()<=l.maxBufferSize&&x->SizeInBytes()<=l.maxStorageBufferBindingSize,
      "RW07 projection owner/residency/resources");
    ORT_RETURN_IF_NOT(total->SizeInBytes()<=l.maxBufferSize&&total->SizeInBytes()<=l.maxStorageBufferBindingSize,"RW07 total carrier binding capacity");
    auto* o=c.Output(0,TensorShape({m0_extent_abi2::Words}));
    ORT_RETURN_IF_NOT(Gpu(o)&&o->IsDataType<uint32_t>(),"RW07 projection output residency");
    const uint64_t run=c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity source_identity,carrier_identity,output_identity;
    ORT_RETURN_IF_NOT(run&&c.M0FullGpuDescribeResourceV2(*e,source_identity)&&c.M0FullGpuDescribeResourceV2(*total,carrier_identity)&&c.M0FullGpuClaimExtentV2(*o,*x,2,1002,output_identity),"RW07 actual frame sidecar/carrier allocation provenance");
    // Compare the GPU payload to independently observed live allocations.
    // Stored producer provenance must not cause an earlier host denial.
    source_identity.data_generation=carrier_identity.generation;
    ProjectionProgram p;
    p.AddInput(Words(e)).AddOutput(Words(o)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
      .AddUniformVariables({uint32_t(channels),multiple,offset,uint32_t(x->Shape()[2]),uint32_t(epoch),uint32_t(epoch>>32),uint32_t(run),uint32_t(run>>32),uint32_t(source_identity.generation),uint32_t(source_identity.generation>>32),uint32_t(source_identity.data_generation),uint32_t(source_identity.data_generation>>32),uint32_t(output_identity.generation),uint32_t(output_identity.generation>>32),uint32_t(output_identity.data_generation),uint32_t(output_identity.data_generation>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(p));
    LOGS(c.Logger(),INFO)<<"M0_RW07_EXTENT_OWNERSHIP_V2 producer="<<producer_label<<" session="<<epoch<<" run="<<run
      <<" source_sidecar="<<e->DataRaw()<<" source_generation="<<source_identity.generation
      <<" source_carrier="<<total->DataRaw()<<" source_carrier_generation="<<carrier_identity.generation
      <<" expected_source_role=1 expected_source_instance=1001 output_sidecar="<<o->DataRaw()
      <<" output_generation="<<output_identity.generation<<" output_carrier="<<x->DataRaw()
      <<" output_carrier_generation="<<output_identity.data_generation<<" role=2 instance=1002";
    for(const auto& entry:{std::pair{"projection-source-extent-device-only",e},std::pair{"projection-value-descriptor-only",x},
                          std::pair{"extent-device-only",static_cast<const Tensor*>(o)}})
      LOGS(c.Logger(),INFO)<<"M0_RW07_RESOURCE role="<<entry.first<<" epoch="<<epoch<<" handle="<<entry.second->DataRaw()
        <<" device="<<int(entry.second->Location().device.Type())<<" physical_bytes="<<entry.second->SizeInBytes();
    return Status::OK();
}
} // namespace onnxruntime::webgpu::m0_frame_v2
