// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Actual original2426 ->2427..2463, with genuine generator ownership.
#include <algorithm>
#include <array>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t){return t&&t->Location().device.Type()==OrtDevice::GPU;}
Status Binding(ComputeContext& c,const Tensor* t){
 const auto& l=c.DeviceLimits();
 ORT_RETURN_IF_NOT(Gpu(t)&&t->DataRaw()&&t->SizeInBytes()>0&&t->SizeInBytes()<=l.maxBufferSize&&t->SizeInBytes()<=l.maxStorageBufferBindingSize,"RW07 reconstruction3 GPU binding");
 const uint64_t actual=wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
 ORT_RETURN_IF_NOT(actual>=t->SizeInBytes()&&actual<=l.maxBufferSize&&actual<=l.maxStorageBufferBindingSize,"RW07 reconstruction3 actual backing");return Status::OK();
}
void ValidOutput(ShaderHelper& s,const ShaderVariableHelper& e){
 auto& f=s.AdditionalImplementation();
 f<<"fn valid_output()->bool{if("<<e.GetByOffset("0u")<<"!=2u||"<<e.GetByOffset("1u")<<"!=1u||"<<e.GetByOffset("20u")<<"!=0u||"<<e.GetByOffset("21u")<<"!=63u){return false;}"
  <<"for(var i=0u;i<10u;i++){if("<<e.GetByOffset("22u+i")<<"!="<<GetElementAt("uniforms.identity","i",10)<<"){return false;}}"
  <<"for(var i=3u;i<18u;i++){if("<<e.GetByOffset("i")<<"!=0u){return false;}}"
  <<"let n="<<e.GetByOffset("2u")<<";return n>=600u&&n<=61440000u&&n%600u==0u&&n<=uniforms.output_capacity&&n/5u+1u<=uniforms.input_capacity&&"
  <<e.GetByOffset("18u")<<"==n&&"<<e.GetByOffset("19u")<<"==0u;}\n";
}
void Arithmetic(ShaderHelper& s){
 s.AdditionalImplementation()<<R"WGSL(
fn original_mul(a:f32,b:f32)->f32{
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0x7f800000u)||(bb==0u&&aa==0x7f800000u)){return bitcast<f32>(0x7fc00000u);}
 if(aa==0x7f800000u||bb==0x7f800000u){return bitcast<f32>(((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)|0x7f800000u);}return a*b;
}
fn original_add(a:f32,b:f32)->f32{
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0x7f800000u&&bb==0x7f800000u&&((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)!=0u)){return bitcast<f32>(0x7fc00000u);}
 if(aa==0x7f800000u){return a;}if(bb==0x7f800000u){return b;}return a+b;
}
fn original_div(a:f32,b:f32)->f32{
 let ua=bitcast<u32>(a);let ub=bitcast<u32>(b);let aa=ua&0x7fffffffu;let bb=ub&0x7fffffffu;let sign=(ua^ub)&0x80000000u;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0u)||(aa==0x7f800000u&&bb==0x7f800000u)){return bitcast<f32>(0x7fc00000u);}
 if(bb==0u||aa==0x7f800000u){return bitcast<f32>(sign|0x7f800000u);}if(aa==0u||bb==0x7f800000u){return bitcast<f32>(sign);}return a/b;
}
)WGSL";
}
class Guard final:public Program<Guard>{
 public:
 Guard():Program{"RW07OriginalReconstruction3Guard"}{}
 WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
  {"input_capacity",ProgramUniformVariableDataType::Uint32},{"output_capacity",ProgramUniformVariableDataType::Uint32},
  {"source_valid",ProgramUniformVariableDataType::Uint32},{"source_identity",ProgramUniformVariableDataType::Uint32},{"identity",ProgramUniformVariableDataType::Uint32});
 Status GenerateShaderCode(ShaderHelper& s)const override{
  const auto& e=s.AddInput("source",ShaderUsage::UseValueTypeAlias);const auto& w=s.AddInput("window",ShaderUsage::UseValueTypeAlias);const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
  s.AdditionalImplementation()<<"const original_window=array<u32,20>(0u,974978362u,1008035403u,1026424773u,1039430992u,1048576000u,1054561486u,1059150100u,1062302038u,1064542128u,1065353216u,1064542128u,1062302038u,1059150100u,1054561486u,1048576000u,1039430992u,1026424773u,1008035403u,974978362u);";
  auto& f=s.MainFunctionBody();
  f<<"for(var i=0u;i<32u;i++){"<<o.SetByOffset("i","0u")<<"}"<<o.SetByOffset("0u","2u")<<o.SetByOffset("1u","1u")
   <<"for(var i=0u;i<10u;i++){"<<o.SetByOffset("22u+i",GetElementAt("uniforms.identity","i",10))<<"}"
   <<"if(uniforms.source_valid==0u||"<<e.GetByOffset("0u")<<"!=2u||"<<e.GetByOffset("1u")<<"!=3u){"<<o.SetByOffset("20u","103u")<<"return;}"
   <<"for(var i=0u;i<10u;i++){if("<<e.GetByOffset("22u+i")<<"!="<<GetElementAt("uniforms.source_identity","i",10)<<"){"<<o.SetByOffset("20u","103u")<<"return;}}"
   <<"let status="<<e.GetByOffset("20u")<<";if(status!=0u){if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"<<o.SetByOffset("20u","103u")<<"return;}"
   <<"for(var i=2u;i<20u;i++){if("<<e.GetByOffset("i")<<"!=0u){"<<o.SetByOffset("20u","103u")<<"return;}}if("<<e.GetByOffset("21u")<<"!=0u){"<<o.SetByOffset("20u","103u")<<"return;}"<<o.SetByOffset("20u","status")<<"return;}"
   <<"if("<<e.GetByOffset("2u")<<"!=1u||"<<e.GetByOffset("3u")<<"!=0u||"<<e.GetByOffset("4u")<<"!=22u||"<<e.GetByOffset("5u")<<"!=0u||"<<e.GetByOffset("7u")<<"!=0u||"<<e.GetByOffset("19u")<<"!=0u||"<<e.GetByOffset("21u")<<"!=63u){"<<o.SetByOffset("20u","103u")<<"return;}"
   <<"for(var i=8u;i<18u;i++){if("<<e.GetByOffset("i")<<"!=0u){"<<o.SetByOffset("20u","103u")<<"return;}}let frames="<<e.GetByOffset("6u")<<";"
   <<"if(frames<121u||frames>12288001u||frames%120u!=1u||"<<e.GetByOffset("18u")<<"!=22u*frames){"<<o.SetByOffset("20u","103u")<<"return;}"
   <<"for(var j=0u;j<20u;j++){if(bitcast<u32>("<<w.GetByOffset("j")<<")!=original_window[j]){"<<o.SetByOffset("20u","103u")<<"return;}let selected="<<w.GetByOffset("j")<<">bitcast<f32>(0x00800000u);if(selected!=(j!=0u)){"<<o.SetByOffset("20u","103u")<<"return;}}"
   <<"let n=5u*(frames-1u);if(frames>uniforms.input_capacity||n>uniforms.output_capacity){"<<o.SetByOffset("20u","102u")<<"return;}"
   <<o.SetByOffset("2u","n")<<o.SetByOffset("18u","n")<<o.SetByOffset("21u","63u");return Status::OK();
 }
};
class Inverse final:public Program<Inverse>{
 public:
 Inverse():Program{"RW07Original2427InverseGemmV3"}{}
 WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_capacity",ProgramUniformVariableDataType::Uint32},{"output_capacity",ProgramUniformVariableDataType::Uint32},{"identity",ProgramUniformVariableDataType::Uint32});
 Status GenerateShaderCode(ShaderHelper& s)const override{
  const auto& x=s.AddInput("spectrum",ShaderUsage::UseValueTypeAlias);const auto& w=s.AddInput("basis",ShaderUsage::UseValueTypeAlias);const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);const auto& y=s.AddOutput("col",ShaderUsage::UseValueTypeAlias);
  ValidOutput(s,e);Arithmetic(s);auto& f=s.MainFunctionBody();
  f<<"if(global_idx>=20u*uniforms.input_capacity){return;}"<<y.SetByOffset("global_idx","0.0")<<"if(!valid_output()){return;}let frame=global_idx%uniforms.input_capacity;let k=global_idx/uniforms.input_capacity;let frames="<<e.GetByOffset("2u")<<"/5u+1u;if(frame>=frames){return;}var sum:f32=0.0;"
   <<"for(var c=0u;c<22u;c++){let product=original_mul("<<x.GetByOffset("c*uniforms.input_capacity+frame")<<","<<w.GetByOffset("c*20u+k")<<");sum=original_add(sum,product);}"<<y.SetByOffset("global_idx","sum");return Status::OK();
 }
};
class Finish final:public Program<Finish>{
 public:
 Finish():Program{"RW07Original2427To2463OverlapScatterCropV3"}{}
 WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_capacity",ProgramUniformVariableDataType::Uint32},{"output_capacity",ProgramUniformVariableDataType::Uint32},{"identity",ProgramUniformVariableDataType::Uint32});
 Status GenerateShaderCode(ShaderHelper& s)const override{
  const auto& col=s.AddInput("col",ShaderUsage::UseValueTypeAlias);const auto& w=s.AddInput("window",ShaderUsage::UseValueTypeAlias);const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);const auto& y=s.AddOutput("pcm",ShaderUsage::UseValueTypeAlias);
  ValidOutput(s,e);Arithmetic(s);auto& f=s.MainFunctionBody();
  f<<"if(global_idx>=uniforms.output_capacity){return;}"<<y.SetByOffset("global_idx","0.0")<<"if(!valid_output()||global_idx>="<<e.GetByOffset("2u")<<"){return;}let frames="<<e.GetByOffset("2u")<<"/5u+1u;let sample=global_idx+10u;var value:f32=0.0;"
   // Original Col2imNd visits kernel taps first; omit noncontributing taps.
   <<"for(var k=0u;k<20u;k++){if(sample>=k){let delta=sample-k;if(delta%5u==0u){let frame=delta/5u;if(frame<frames){value=original_add(value,"<<col.GetByOffset("k*uniforms.input_capacity+frame")<<");}}}}"
   // Original NonZero/ScatterND overwrites only indices1..19, then crop10.
   <<"if(sample<20u){value=original_div(value,"<<w.GetByOffset("sample")<<");}let result=original_mul(value,4.0);"<<y.SetByOffset("global_idx","result");return Status::OK();
 }
};
class Reconstruction final:public WebGpuKernel{
 public:
 explicit Reconstruction(const OpKernelInfo& i):WebGpuKernel{i},capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)){}
 Status ComputeInternal(ComputeContext& c)const override{
  const auto* x=c.Input<Tensor>(0);const auto* e=c.Input<Tensor>(1);const auto* w=c.Input<Tensor>(2);const auto* window=c.Input<Tensor>(3);
  ORT_RETURN_IF_NOT(Gpu(x)&&x->IsDataType<float>()&&x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&x->Shape()[1]==22&&x->Shape()[2]>0&&x->Shape()[2]<=12288001&&
                        Gpu(e)&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape{32}&&Gpu(w)&&w->IsDataType<float>()&&w->Shape()==TensorShape({22,1,20})&&
                        Gpu(window)&&window->IsDataType<float>()&&window->Shape()==TensorShape{20}&&capacity_>0&&capacity_<=61440000,"RW07 reconstruction3 exact physical descriptors");
  const auto& l=c.DeviceLimits();const uint64_t limit=l.maxComputeWorkgroupsPerDimension,pc=uint64_t(x->Shape()[2]);
  ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage>=4&&l.maxComputeWorkgroupSizeX>=64&&l.maxComputeInvocationsPerWorkgroup>=64&&limit>0,"RW07 reconstruction3 dispatch resources");
  for(uint64_t n:{20*pc,uint64_t(capacity_)})ORT_RETURN_IF_NOT(n<=UINT32_MAX&&n*4<=l.maxBufferSize&&n*4<=l.maxStorageBufferBindingSize&&(n+63)/64<=limit*limit,"RW07 reconstruction3 physical resource capacity");
  const std::array<const Tensor*,4> inputs{x,e,w,window};for(const auto* t:inputs)ORT_RETURN_IF_ERROR(Binding(c,t));
  const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();
  M0FullGpuAllocationIdentity sid,data,pair,output;
  ORT_RETURN_IF_NOT(session&&run&&c.M0FullGpuDescribeResourceV2(*e,sid)&&c.M0FullGpuDescribeResourceV2(*x,data),"RW07 reconstruction3 live source allocations");
  const bool paired=c.M0FullGpuDescribePairV2(*e,*x,pair)&&pair.session==session&&pair.run==run&&pair.generation==sid.generation&&pair.data_generation==data.generation&&pair.producer_role==74&&pair.producer_instance==1075;
  auto* y=c.Output(0,TensorShape{capacity_});auto* out=c.Output(1,TensorShape{32});auto col=c.CreateGPUTensor(x->DataType(),TensorShape{20,int64_t(pc)});
  ORT_RETURN_IF_NOT(Gpu(y)&&y->IsDataType<float>()&&Gpu(out)&&out->IsDataType<uint32_t>(),"RW07 reconstruction3 output types");
  const std::array<const Tensor*,3> allocated{y,out,&col};
  for(size_t i=0;i<allocated.size();++i){ORT_RETURN_IF_ERROR(Binding(c,allocated[i]));for(size_t j=0;j<i;++j)ORT_RETURN_IF_NOT(allocated[i]->DataRaw()!=allocated[j]->DataRaw(),"RW07 reconstruction3 distinct scratch/output");for(const auto* t:inputs)ORT_RETURN_IF_NOT(allocated[i]->DataRaw()!=t->DataRaw(),"RW07 reconstruction3 distinct input/output");}
  ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out,*y,75,1076,output),"RW07 reconstruction3 actual PCM ownership");
  const m0_extent_abi2::OpaqueOwnership source{session,run,sid.generation,data.generation,74,1075},owner{session,run,output.generation,output.data_generation,75,1076};
  ORT_RETURN_IF_NOT(source.Valid()&&owner.Valid(),"RW07 reconstruction3 identities");const auto sw=source.UniformWords(),ow=owner.UniformWords();
  const auto span=[](const auto& a){return gsl::span<const uint32_t>{a.data(),a.size()};};
  Guard guard;guard.AddInputs({{e,ProgramTensorMetadataDependency::None},{window,ProgramTensorMetadataDependency::None}}).AddOutput({out,ProgramTensorMetadataDependency::None})
   .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({uint32_t(pc),uint32_t(capacity_),uint32_t(paired),span(sw),span(ow)});ORT_RETURN_IF_ERROR(c.RunProgram(guard));
  const auto dispatch=[&](auto& p,uint64_t count)->Status{const uint64_t groups=(count+63)/64,gx=std::min(groups,limit),gy=(groups+gx-1)/gx;p.SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy));return c.RunProgram(p);};
  Inverse inverse;inverse.AddInputs({{x,ProgramTensorMetadataDependency::None},{w,ProgramTensorMetadataDependency::None},{out,ProgramTensorMetadataDependency::None}}).AddOutput({&col,ProgramTensorMetadataDependency::None})
   .AddUniformVariables({uint32_t(pc),uint32_t(capacity_),span(ow)});ORT_RETURN_IF_ERROR(dispatch(inverse,20*pc));
  Finish finish;finish.AddInputs({{&col,ProgramTensorMetadataDependency::None},{window,ProgramTensorMetadataDependency::None},{out,ProgramTensorMetadataDependency::None}}).AddOutput({y,ProgramTensorMetadataDependency::None})
   .AddUniformVariables({uint32_t(pc),uint32_t(capacity_),span(ow)});ORT_RETURN_IF_ERROR(dispatch(finish,uint64_t(capacity_)));
  LOGS(c.Logger(),INFO)<<"M0_RW07_RECONSTRUCTION_V3 actual_original2426=1 role=75 instance=1076 GPU_dispatches=3 host_payload_reads=0";return Status::OK();
 }
 private:const int64_t capacity_;
};
class PacketProgram final:public Program<PacketProgram>{
 public:
 PacketProgram():Program{"RW07ActualPCM2463TerminalPacketV3"}{}
 WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_capacity",ProgramUniformVariableDataType::Uint32},{"output_capacity",ProgramUniformVariableDataType::Uint32},{"identity",ProgramUniformVariableDataType::Uint32});
 Status GenerateShaderCode(ShaderHelper& s)const override{
  const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);const auto& p=s.AddOutput("packet",ShaderUsage::UseValueTypeAlias);ValidOutput(s,e);auto& f=s.MainFunctionBody();
  f<<"for(var i=0u;i<18u;i++){"<<p.SetByOffset("i","0u")<<"}"<<p.SetByOffset("0u","103u")
   <<"if("<<e.GetByOffset("0u")<<"!=2u||"<<e.GetByOffset("1u")<<"!=1u){return;}for(var i=0u;i<10u;i++){if("<<e.GetByOffset("22u+i")<<"!="<<GetElementAt("uniforms.identity","i",10)<<"){return;}}"
   <<"let status="<<e.GetByOffset("20u")<<";if(status!=0u){if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}for(var i=2u;i<20u;i++){if("<<e.GetByOffset("i")<<"!=0u){return;}}if("<<e.GetByOffset("21u")<<"!=0u){return;}"<<p.SetByOffset("0u","status")<<"return;}if(!valid_output()){return;}"
   <<p.SetByOffset("0u","0u")<<p.SetByOffset("1u","1u")<<p.SetByOffset("2u",e.GetByOffset("2u"));return Status::OK();
 }
};
class Packet final:public WebGpuKernel{
 public:
 explicit Packet(const OpKernelInfo& i):WebGpuKernel{i}{}
 Status ComputeInternal(ComputeContext& c)const override{
  const auto* e=c.Input<Tensor>(0);const auto* y=c.Input<Tensor>(1);
  ORT_RETURN_IF_NOT(Gpu(e)&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape{32}&&Gpu(y)&&y->IsDataType<float>()&&y->Shape().NumDimensions()==1&&y->Shape()[0]>0&&y->Shape()[0]<=61440000,"RW07 PCM3 terminal descriptors");
  const auto& l=c.DeviceLimits();ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage>=2&&l.maxComputeWorkgroupsPerDimension>=1&&l.maxComputeWorkgroupSizeX>=1&&l.maxComputeInvocationsPerWorkgroup>=1,"RW07 PCM3 terminal resources");
  ORT_RETURN_IF_ERROR(Binding(c,e));ORT_RETURN_IF_ERROR(Binding(c,y));
  const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();M0FullGpuAllocationIdentity pair;
  ORT_RETURN_IF_NOT(session&&run&&c.M0FullGpuDescribePairV2(*e,*y,pair)&&pair.session==session&&pair.run==run&&pair.producer_role==75&&pair.producer_instance==1076,"RW07 PCM3 terminal actual pair");
  auto* p=c.Output(0,TensorShape{18});ORT_RETURN_IF_NOT(Gpu(p)&&p->IsDataType<uint32_t>()&&p->DataRaw()!=e->DataRaw()&&p->DataRaw()!=y->DataRaw(),"RW07 PCM3 terminal distinct output");ORT_RETURN_IF_ERROR(Binding(c,p));
  const m0_extent_abi2::OpaqueOwnership owner{session,run,pair.generation,pair.data_generation,75,1076};ORT_RETURN_IF_NOT(owner.Valid(),"RW07 PCM3 terminal identity");const auto words=owner.UniformWords();
  const uint32_t cap=uint32_t(y->Shape()[0]);PacketProgram program;
  program.AddInput({e,ProgramTensorMetadataDependency::None}).AddOutput({p,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(1)
   .AddUniformVariables({12288001u,cap,gsl::span<const uint32_t>{words.data(),words.size()}});ORT_RETURN_IF_ERROR(c.RunProgram(program));
  ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p),"RW07 PCM3 terminal map fault registration");return Status::OK();
 }
};
}  // namespace
KernelCreateInfo CreateM0FullGpuReconstructionExtentV3KernelInfo(){
 KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{out=std::make_unique<Reconstruction>(i);return Status::OK();};
 return {KernelDefBuilder().SetName("ReconstructionExtentV3").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuPCM2463TerminalPacketV3KernelInfo(){
 KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{out=std::make_unique<Packet>(i);return Status::OK();};
 return {KernelDefBuilder().SetName("PCM2463TerminalPacketV3").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
}  // namespace onnxruntime::webgpu
