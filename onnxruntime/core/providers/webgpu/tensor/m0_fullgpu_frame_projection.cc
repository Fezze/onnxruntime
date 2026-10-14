// rw07-extent-stride-linear-v1. Closed source-proven temporal projections.
// Roles 0/1: original1493 [1,1,2D], original1512 [1,9,600D].
// Roles 2/3: isolated boundary diagnostics [1,1,D]/[1,9,D].
// The value input supplies only an allocation descriptor; no value is read.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"
namespace onnxruntime::webgpu {
namespace {
ProgramInput Words(const Tensor* t) {
  ProgramInput p(t,ProgramTensorMetadataDependency::None,TensorShape{rw07::ExtentWords},1);
  p.var_type=ProgramVariableDataType::Uint32;return p;
}
ProgramOutput Words(Tensor* t) {
  ProgramOutput p(t,ProgramTensorMetadataDependency::None,TensorShape{rw07::ExtentWords},1);
  p.var_type=ProgramVariableDataType::Uint32;return p;
}
bool Gpu(const Tensor* t){return t&&t->Location().device.Type()==OrtDevice::GPU;}
class ProjectionProgram final:public Program<ProjectionProgram>{
 public:
  ProjectionProgram():Program{"RW07V1FrameProjectionExtent"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"multiple",ProgramUniformVariableDataType::Uint32},
    {"offset",ProgramUniformVariableDataType::Uint32},
    {"capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& e=s.AddInput("root",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    f<<"for(var i=0u;i<24u;i++){"<<o.SetByOffset("i","0u")<<"}"
     <<o.SetByOffset("0u","1u")<<o.SetByOffset("1u","3u")
     <<o.SetByOffset("22u","uniforms.epoch_low")<<o.SetByOffset("23u","uniforms.epoch_high")
     <<"if("<<e.GetByOffset("0u")<<"!=1u||"<<e.GetByOffset("1u")<<"!=1u||"
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
class FrameProjection final:public WebGpuKernel{
 public:
  explicit FrameProjection(const OpKernelInfo& i):WebGpuKernel{i},role_(i.GetAttrOrDefault<int64_t>("role",-1)){}
  Status ComputeInternal(ComputeContext& c)const override{
    ORT_RETURN_IF_NOT(role_>=0&&role_<=1,"RW07 closed projection role");
    const int64_t channels=22;
    const uint32_t multiple=role_==0?120:1;
    const uint32_t offset=role_==0?1:0;
    const auto* e=c.Input<Tensor>(0);const auto* x=c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(x&&x->IsDataType<float>()&&x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&
      x->Shape()[1]==channels&&x->Shape()[2]>0&&x->Shape()[2]<=6144001&&
      e&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape({rw07::ExtentWords}),"RW07 projection physical descriptors");
    const uint64_t epoch=c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    const auto& l=c.DeviceLimits();
    ORT_RETURN_IF_NOT(epoch&&Gpu(e)&&Gpu(x)&&l.maxStorageBuffersPerShaderStage>=2&&
      l.maxBufferSize>=96&&l.maxStorageBufferBindingSize>=96&&l.maxComputeWorkgroupsPerDimension>=1&&
      x->SizeInBytes()<=l.maxBufferSize&&x->SizeInBytes()<=l.maxStorageBufferBindingSize,
      "RW07 projection owner/residency/resources");
    auto* o=c.Output(0,TensorShape({rw07::ExtentWords}));
    ORT_RETURN_IF_NOT(Gpu(o)&&o->IsDataType<uint32_t>(),"RW07 projection output residency");
    ProjectionProgram p;
    p.AddInput(Words(e)).AddOutput(Words(o)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
      .AddUniformVariables({uint32_t(channels),multiple,offset,uint32_t(x->Shape()[2]),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(p));
    for(const auto& entry:{std::pair{"projection-source-extent-device-only",e},std::pair{"projection-value-descriptor-only",x},
                          std::pair{"extent-device-only",static_cast<const Tensor*>(o)}})
      LOGS(c.Logger(),INFO)<<"M0_RW07_RESOURCE role="<<entry.first<<" epoch="<<epoch<<" handle="<<entry.second->DataRaw()
        <<" device="<<int(entry.second->Location().device.Type())<<" physical_bytes="<<entry.second->SizeInBytes();
    return Status::OK();
  }
 private:int64_t role_;
};
}
KernelCreateInfo CreateM0FullGpuFrameProjectionExtentKernelInfo(){
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<FrameProjection>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("FrameProjectionExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
    .Provider(kWebGpuExecutionProvider).Build(),fn};
}
}
