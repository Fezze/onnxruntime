// RW07 exact LSTM1340, layout1341-1343, temporal reductions1344-1347.
// Closed original contract: bidirectional H256/I640, batch1, zero H/C,
// no sequence_lens, peepholes, input_forget, activation override or clip.
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
class RecurrenceGuard final:public Program<RecurrenceGuard>{
 public:
  RecurrenceGuard():Program{"RW07V1RecurrenceExtentGuard"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& e=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);auto& f=s.MainFunctionBody();
    f<<"for(var i=0u;i<24u;i++){"<<o.SetByOffset("i","0u")<<"}"
     <<o.SetByOffset("0u","1u")<<o.SetByOffset("1u","3u")
     <<o.SetByOffset("22u","uniforms.epoch_low")<<o.SetByOffset("23u","uniforms.epoch_high")
     <<"if("<<e.GetByOffset("0u")<<"!=1u||"<<e.GetByOffset("1u")<<"!=3u||"
     <<e.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<e.GetByOffset("23u")<<"!=uniforms.epoch_high){"
     <<o.SetByOffset("20u","103u")<<"return;}"
     <<"let status="<<e.GetByOffset("20u")<<";if(status!=0u){"<<o.SetByOffset("20u","status")<<"return;}"
     <<"let len="<<e.GetByOffset("6u")<<";if("<<e.GetByOffset("2u")<<"!=1u||"
     <<e.GetByOffset("3u")<<"!=0u||"<<e.GetByOffset("4u")<<"!=640u||"
     <<e.GetByOffset("5u")<<"!=0u||"<<e.GetByOffset("7u")<<"!=0u||len==0u||"
     <<e.GetByOffset("18u")<<"!=640u*len||"<<e.GetByOffset("19u")<<"!=0u||("
     <<e.GetByOffset("21u")<<"&63u)!=63u){"<<o.SetByOffset("20u","103u")<<"return;}"
     <<"if(len>uniforms.input_capacity||len>uniforms.output_capacity){"<<o.SetByOffset("20u","102u")<<"return;}"
     <<o.SetByOffset("2u","1u")<<o.SetByOffset("4u","512u")<<o.SetByOffset("6u","len")
     <<o.SetByOffset("18u","512u*len")<<o.SetByOffset("21u","63u");return Status::OK();
  }
};
class RecurrenceData final:public Program<RecurrenceData>{
 public:
  RecurrenceData():Program{"RW07V1RecurrenceData"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& x=s.AddInput("x",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("w",ShaderUsage::UseValueTypeAlias);
    const auto& r=s.AddInput("r",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("b",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("y",ShaderUsage::UseValueTypeAlias);
    const auto& h=s.AddOutput("final_h",ShaderUsage::UseValueTypeAlias);
    const auto& c=s.AddOutput("final_c",ShaderUsage::UseValueTypeAlias);
    const auto& m=s.AddOutput("merged",ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation()<<"var<workgroup> previous_h:array<f32,256>;\nvar<workgroup> previous_c:array<f32,256>;\nvar<workgroup> active_length:u32;\n";
    auto& f=s.MainFunctionBody();
    f<<"let unit=local_idx;let direction=workgroup_id.x;let channel=direction*256u+unit;"
     <<"previous_h[unit]=0.0;previous_c[unit]=0.0;"
     <<h.SetByOffset("channel","0.0")<<c.SetByOffset("channel","0.0")
     <<"for(var t=0u;t<uniforms.output_capacity;t++){"
     <<y.SetByOffset("(t*2u+direction)*256u+unit","0.0")
     <<m.SetByOffset("channel*uniforms.output_capacity+t","0.0")<<"}"
     <<"if(unit==0u){active_length=0u;if("<<e.GetByOffset("20u")<<"==0u&&"
     <<e.GetByOffset("22u")<<"==uniforms.epoch_low&&"<<e.GetByOffset("23u")<<"==uniforms.epoch_high){active_length="
     <<e.GetByOffset("6u")<<";}}"
     // This builtin supplies a uniform GPU-derived loop bound without disabling
     // WGSL barrier uniformity validation or obtaining a host length.
     <<"let len=workgroupUniformLoad(&active_length);if(len==0u){return;}"
     <<"for(var step=0u;step<len;step++){let t=select(step,len-1u-step,direction==1u);var wx=vec4<f32>(0.0);var rh=vec4<f32>(0.0);"
     <<"for(var k=0u;k<640u;k++){let v="<<x.GetByOffset("k*uniforms.input_capacity+t")<<";let base=(direction*1024u+unit)*640u+k;"
     <<"wx+=v*vec4<f32>("<<w.GetByOffset("base")<<","<<w.GetByOffset("base+256u*640u")<<","<<w.GetByOffset("base+512u*640u")<<","<<w.GetByOffset("base+768u*640u")<<");}"
     <<"for(var k=0u;k<256u;k++){let v=previous_h[k];let base=(direction*1024u+unit)*256u+k;"
     <<"rh+=v*vec4<f32>("<<r.GetByOffset("base")<<","<<r.GetByOffset("base+256u*256u")<<","<<r.GetByOffset("base+512u*256u")<<","<<r.GetByOffset("base+768u*256u")<<");}"
     <<"let bi=direction*2048u+unit;let wb=vec4<f32>("<<b.GetByOffset("bi")<<","<<b.GetByOffset("bi+256u")<<","<<b.GetByOffset("bi+512u")<<","<<b.GetByOffset("bi+768u")<<");"
     <<"let rb=vec4<f32>("<<b.GetByOffset("bi+1024u")<<","<<b.GetByOffset("bi+1280u")<<","<<b.GetByOffset("bi+1536u")<<","<<b.GetByOffset("bi+1792u")<<");"
     <<"let gates=(wx+rh)+(wb+rb);let i=1.0/(1.0+exp(-gates.x));let o=1.0/(1.0+exp(-gates.y));let forget=1.0/(1.0+exp(-gates.z));let candidate=tanh(gates.w);"
     <<"let next_c=forget*previous_c[unit]+i*candidate;let next_h=o*tanh(next_c);"
     // Every old-H read finishes before any invocation overwrites it. A second
     // barrier publishes the new state before the next active step.
     <<"workgroupBarrier();previous_c[unit]=next_c;previous_h[unit]=next_h;workgroupBarrier();"
     <<y.SetByOffset("(t*2u+direction)*256u+unit","next_h")
     <<m.SetByOffset("channel*uniforms.output_capacity+t","next_h")<<"}"
     <<h.SetByOffset("channel","previous_h[unit]")<<c.SetByOffset("channel","previous_c[unit]");return Status::OK();
  }
};
class TemporalMoments final:public Program<TemporalMoments>{
 public:
  TemporalMoments():Program{"RW07V1TemporalMoments"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& x=s.AddInput("merged",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& mean=s.AddOutput("mean",ShaderUsage::UseValueTypeAlias);
    const auto& variance=s.AddOutput("variance",ShaderUsage::UseValueTypeAlias);auto& f=s.MainFunctionBody();
    f<<"if(global_idx>=512u){return;}"<<mean.SetByOffset("global_idx","0.0")<<variance.SetByOffset("global_idx","0.0")
     <<"if("<<e.GetByOffset("20u")<<"!=0u||"<<e.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<e.GetByOffset("23u")<<"!=uniforms.epoch_high){return;}"
     <<"let len="<<e.GetByOffset("6u")<<";var sum=0.0;for(var t=0u;t<len;t++){sum+="<<x.GetByOffset("global_idx*uniforms.capacity+t")<<";}"
     <<"let mu=sum/f32(len);var squares=0.0;for(var t=0u;t<len;t++){let delta="<<x.GetByOffset("global_idx*uniforms.capacity+t")<<"-mu;squares+=delta*delta;}"
     <<mean.SetByOffset("global_idx","mu")<<variance.SetByOffset("global_idx","squares/f32(len)");return Status::OK();
  }
};
class RecurrenceExtent final:public WebGpuKernel{
 public:
  explicit RecurrenceExtent(const OpKernelInfo& i):WebGpuKernel{i},capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)){}
  Status ComputeInternal(ComputeContext& c)const override{
    const auto* x=c.Input<Tensor>(0);const auto* w=c.Input<Tensor>(1);const auto* r=c.Input<Tensor>(2);const auto* b=c.Input<Tensor>(3);const auto* e=c.Input<Tensor>(4);
    ORT_RETURN_IF_NOT(x&&x->IsDataType<float>()&&x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&x->Shape()[1]==640,"RW07 LSTM1340 input descriptor");
    const int64_t input_capacity=x->Shape()[2];
    ORT_RETURN_IF_NOT(input_capacity>0&&input_capacity<=102400&&capacity_>0&&capacity_<=102400&&
      w&&w->IsDataType<float>()&&w->Shape()==TensorShape({2,1024,640})&&r&&r->IsDataType<float>()&&r->Shape()==TensorShape({2,1024,256})&&
      b&&b->IsDataType<float>()&&b->Shape()==TensorShape({2,2048})&&e&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape({rw07::ExtentWords}),"RW07 exact LSTM1340 physical descriptors");
    const uint64_t epoch=c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(epoch&&Gpu(x)&&Gpu(w)&&Gpu(r)&&Gpu(b)&&Gpu(e),"RW07 recurrence owner/input residency");
    const auto& limits=c.DeviceLimits();const uint64_t output_bytes=512*uint64_t(capacity_)*4;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=9&&output_bytes<=limits.maxBufferSize&&output_bytes<=limits.maxStorageBufferBindingSize&&
      limits.maxBufferSize>=2048&&limits.maxStorageBufferBindingSize>=2048&&limits.maxComputeWorkgroupsPerDimension>=8&&
      limits.maxComputeWorkgroupSizeX>=256&&limits.maxComputeInvocationsPerWorkgroup>=256&&limits.maxComputeWorkgroupStorageSize>=2052,"RW07 recurrence actual resource/dispatch limits");
    for(const auto* input:{x,w,r,b,e})ORT_RETURN_IF_NOT(input->SizeInBytes()<=limits.maxBufferSize&&input->SizeInBytes()<=limits.maxStorageBufferBindingSize,"RW07 recurrence input binding limit");
    auto* y=c.Output(0,TensorShape({capacity_,2,1,256}));auto* h=c.Output(1,TensorShape({2,1,256}));auto* cell=c.Output(2,TensorShape({2,1,256}));
    auto* merged=c.Output(3,TensorShape({1,512,capacity_}));auto* mean=c.Output(4,TensorShape({1,512,1}));auto* variance=c.Output(5,TensorShape({1,512,1}));auto* out=c.Output(6,TensorShape({rw07::ExtentWords}));
    for(const auto* value:{y,h,cell,merged,mean,variance})ORT_RETURN_IF_NOT(Gpu(value)&&value->IsDataType<float>(),"RW07 recurrence output residency/type");
    ORT_RETURN_IF_NOT(Gpu(out)&&out->IsDataType<uint32_t>(),"RW07 recurrence extent residency/type");
    RecurrenceGuard guard;guard.AddInput(Words(e)).AddOutput(Words(out)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
      .AddUniformVariables({uint32_t(input_capacity),uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    RecurrenceData data;data.AddInputs({{x,ProgramTensorMetadataDependency::None},{w,ProgramTensorMetadataDependency::None},{r,ProgramTensorMetadataDependency::None},{b,ProgramTensorMetadataDependency::None},Words(static_cast<const Tensor*>(out))})
      .AddOutputs({{y,ProgramTensorMetadataDependency::None},{h,ProgramTensorMetadataDependency::None},{cell,ProgramTensorMetadataDependency::None},{merged,ProgramTensorMetadataDependency::None}})
      .SetWorkgroupSize(256).SetDispatchGroupSize(2).AddUniformVariables({uint32_t(input_capacity),uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});ORT_RETURN_IF_ERROR(c.RunProgram(data));
    TemporalMoments moments;moments.AddInputs({{merged,ProgramTensorMetadataDependency::None},Words(static_cast<const Tensor*>(out))})
      .AddOutputs({{mean,ProgramTensorMetadataDependency::None},{variance,ProgramTensorMetadataDependency::None}}).SetWorkgroupSize(64).SetDispatchGroupSize(8)
      .AddUniformVariables({uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});ORT_RETURN_IF_ERROR(c.RunProgram(moments));
    for(const auto& entry:{std::pair{"recurrence-input",x},std::pair{"recurrence-W",w},std::pair{"recurrence-R",r},std::pair{"recurrence-B",b},std::pair{"source-extent-device-only",e},
      std::pair{"recurrence-Y",static_cast<const Tensor*>(y)},std::pair{"recurrence-H",static_cast<const Tensor*>(h)},std::pair{"recurrence-C",static_cast<const Tensor*>(cell)},
      std::pair{"recurrence-merged",static_cast<const Tensor*>(merged)},std::pair{"temporal-mean",static_cast<const Tensor*>(mean)},std::pair{"temporal-variance",static_cast<const Tensor*>(variance)},std::pair{"extent-device-only",static_cast<const Tensor*>(out)}})
      LOGS(c.Logger(),INFO)<<"M0_RW07_RESOURCE role="<<entry.first<<" epoch="<<epoch<<" handle="<<entry.second->DataRaw()<<" device="<<int(entry.second->Location().device.Type())<<" physical_bytes="<<entry.second->SizeInBytes();
    return Status::OK();
  }
 private:int64_t capacity_;
};
}
KernelCreateInfo CreateM0FullGpuRecurrenceExtentKernelInfo(){
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{out=std::make_unique<RecurrenceExtent>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("RecurrenceExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
}
