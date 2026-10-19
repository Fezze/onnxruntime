// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original generator upsample/join/parallel-path average and spectral projection edges.
#include <algorithm>
#include <array>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t){return t&&t->Location().device.Type()==OrtDevice::GPU;}
void ValidOutput(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_output()->bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=3u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("4u") << "!=uniforms.channels||" << e.GetByOffset("5u") << "!=0u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let n=" << e.GetByOffset("6u") << ";return n>=uniforms.minimum&&n<=uniforms.maximum&&"
    << "n%uniforms.modulus==uniforms.remainder&&n<=uniforms.output_capacity&&"
    << e.GetByOffset("18u") << "==uniforms.channels*n;}\n";
}

class EdgeGuard final : public Program<EdgeGuard> {
 public:
  EdgeGuard(uint32_t mode,uint32_t pairs):Program{"RW07GeneratorEdgeGuardV1"},mode_(mode),pairs_(pairs) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"source_capacity",ProgramUniformVariableDataType::Uint32},
    {"source_channels",ProgramUniformVariableDataType::Uint32},
    {"source_minimum",ProgramUniformVariableDataType::Uint32},
    {"source_maximum",ProgramUniformVariableDataType::Uint32},
    {"source_modulus",ProgramUniformVariableDataType::Uint32},
    {"source_remainder",ProgramUniformVariableDataType::Uint32},
    {"pair_valid",ProgramUniformVariableDataType::Uint32},
    {"source_identity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"stage",ProgramUniformVariableDataType::Uint32},
    {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    std::array<const ShaderVariableHelper*,3> extents{};
    for(uint32_t j=0;j<pairs_;j++){extents[j]=&s.AddInput("source"+std::to_string(j),ShaderUsage::UseValueTypeAlias);}
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    for(uint32_t j=0;j<pairs_;j++){
      const auto& e=*extents[j];auto& f=s.AdditionalImplementation();const auto js=std::to_string(j)+"u";
      f<<"fn source_status"<<j<<"()->u32 {if("<<GetElementAt("uniforms.pair_valid",js,3)<<"==0u||"
       <<e.GetByOffset("0u")<<"!=2u||"<<e.GetByOffset("1u")<<"!=3u){return 103u;}"
       <<"for(var i=0u;i<10u;i++){if("<<e.GetByOffset("22u+i")<<"!="
       <<GetElementAt("uniforms.source_identity",std::to_string(j*10)+"u+i",30)<<"){return 103u;}}"
       <<"let status="<<e.GetByOffset("20u")<<";if(status!=0u){"
       <<"if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return 103u;}"
       <<"for(var i=2u;i<20u;i++){if("<<e.GetByOffset("i")<<"!=0u){return 103u;}}"
       <<"if("<<e.GetByOffset("21u")<<"!=0u){return 103u;}return status;}"
       <<"if("<<e.GetByOffset("2u")<<"!=1u||"<<e.GetByOffset("3u")<<"!=0u||"
       <<e.GetByOffset("4u")<<"!=uniforms.source_channels||"<<e.GetByOffset("5u")<<"!=0u||"
       <<e.GetByOffset("7u")<<"!=0u||"<<e.GetByOffset("19u")<<"!=0u||"
       <<e.GetByOffset("21u")<<"!=63u){return 103u;}"
       <<"for(var i=8u;i<18u;i++){if("<<e.GetByOffset("i")<<"!=0u){return 103u;}}let n="<<e.GetByOffset("6u")<<";"
       <<"if(n<uniforms.source_minimum||n>uniforms.source_maximum||n%uniforms.source_modulus!=uniforms.source_remainder||"
       <<e.GetByOffset("18u")<<"!=uniforms.source_channels*n){return 103u;}"
       <<"return 0u;}\n";
    }
    auto& f=s.MainFunctionBody();
    f<<"for(var i=0u;i<32u;i++){"<<o.SetByOffset("i","0u")<<"}"<<o.SetByOffset("0u","2u")<<o.SetByOffset("1u","3u")
     <<"for(var i=0u;i<10u;i++){"<<o.SetByOffset("22u+i",GetElementAt("uniforms.identity","i",10))<<"}";
    for(uint32_t j=0;j<pairs_;j++){f<<"let status"<<j<<"=source_status"<<j<<"();";}
    for(uint32_t j=0;j<pairs_;j++){f<<"if(status"<<j<<"==103u){"<<o.SetByOffset("20u","103u")<<"return;}";}
    for(uint32_t i=0;i<pairs_;i++){for(uint32_t j=i+1;j<pairs_;j++){
      f<<"if(status"<<i<<"==0u&&status"<<j<<"==0u&&"<<extents[i]->GetByOffset("6u")<<"!="<<extents[j]->GetByOffset("6u")
       <<"){"<<o.SetByOffset("20u","103u")<<"return;}";}}
    for(uint32_t j=0;j<pairs_;j++){f<<"if(status"<<j<<"!=0u){"<<o.SetByOffset("20u","status"+std::to_string(j))<<"return;}";}
    f<<"let n="<<extents[0]->GetByOffset("6u")<<";";
    for(uint32_t j=0;j<pairs_;j++){f<<"if(n>"<<GetElementAt("uniforms.source_capacity",std::to_string(j)+"u",3)
      <<"){"<<o.SetByOffset("20u","102u")<<"return;}";}
    if(mode_==0){f<<"let count=select(10u*n,6u*n+1u,uniforms.stage==1u);";}
    else{f<<"let count=n;";}
    f<<"if(count>uniforms.output_capacity){"<<o.SetByOffset("20u","102u")<<"return;}"
     <<o.SetByOffset("2u","1u")<<o.SetByOffset("4u","uniforms.channels")<<o.SetByOffset("6u","count")
     <<o.SetByOffset("18u","uniforms.channels*count")<<o.SetByOffset("21u","63u");
    return Status::OK();
  }
 private:
  const uint32_t mode_,pairs_;
};
class EdgeData final : public Program<EdgeData> {
 public:
  EdgeData(uint32_t mode,uint32_t pairs):Program{"RW07GeneratorEdgeDataV1"},mode_(mode),pairs_(pairs) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"minimum",ProgramUniformVariableDataType::Uint32},
    {"maximum",ProgramUniformVariableDataType::Uint32},
    {"modulus",ProgramUniformVariableDataType::Uint32},
    {"remainder",ProgramUniformVariableDataType::Uint32},
    {"stage",ProgramUniformVariableDataType::Uint32},
    {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    std::array<const ShaderVariableHelper*,3> inputs{};
    for(uint32_t j=0;j<pairs_;j++){inputs[j]=&s.AddInput("carrier"+std::to_string(j),ShaderUsage::UseValueTypeAlias);}
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* w=nullptr;const ShaderVariableHelper* b=nullptr;
    if(mode_==0||mode_==3){w=&s.AddInput("weight",ShaderUsage::UseValueTypeAlias);b=&s.AddInput("bias",ShaderUsage::UseValueTypeAlias);}
    const auto& y=s.AddOutput("output",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e);
    s.AdditionalImplementation() << R"WGSL(
fn original_mul(a:f32,b:f32)->f32 {
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0x7f800000u)||(bb==0u&&aa==0x7f800000u)){
 return bitcast<f32>(0x7fc00000u);}
 if(aa==0x7f800000u||bb==0x7f800000u){return bitcast<f32>(((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)|0x7f800000u);}
 return a*b;
}
fn original_add(a:f32,b:f32)->f32 {
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0x7f800000u&&bb==0x7f800000u&&((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)!=0u)){
 return bitcast<f32>(0x7fc00000u);}
 if(aa==0x7f800000u){return a;}if(bb==0x7f800000u){return b;}return a+b;
}
fn original_sub(a:f32,b:f32)->f32{return original_add(a,-b);}
fn original_sqrt(x:f32)->f32 {
 let bits=bitcast<u32>(x);let magnitude=bits&0x7fffffffu;
 if(magnitude>0x7f800000u||((bits&0x80000000u)!=0u&&magnitude!=0u)){return bitcast<f32>(0x7fc00000u);}
 if(magnitude==0x7f800000u||magnitude==0u){return x;}return sqrt(x);
}
fn original_div(a:f32,b:f32)->f32 {
 let ua=bitcast<u32>(a);let ub=bitcast<u32>(b);let aa=ua&0x7fffffffu;let bb=ub&0x7fffffffu;
 let sign=(ua^ub)&0x80000000u;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0u)||(aa==0x7f800000u&&bb==0x7f800000u)){return bitcast<f32>(0x7fc00000u);}
 if(bb==0u||aa==0x7f800000u){return bitcast<f32>(sign|0x7f800000u);}
 if(aa==0u||bb==0x7f800000u){return bitcast<f32>(sign);}return a/b;
}
)WGSL";
    auto& f=s.MainFunctionBody();
    f<<"if(global_idx>=uniforms.channels*uniforms.output_capacity){return;}"<<y.SetByOffset("global_idx","0.0")
     <<"if(!valid_output()){return;}let t=global_idx%uniforms.output_capacity;let oc=global_idx/uniforms.output_capacity;"
     <<"let count="<<e.GetByOffset("6u")<<";if(t>=count){return;}";
    if(mode_==1||mode_==2){
      for(uint32_t j=0;j<pairs_;j++){
        f<<"let v"<<j<<"="<<inputs[j]->GetByOffset("oc*"+GetElementAt("uniforms.input_capacity",std::to_string(j)+"u",3)+"+t")<<";";
      }
      if(mode_==1){f<<y.SetByOffset("global_idx","original_add(v0,v1)");}
      else{f<<"let first=original_add(v0,v1);let total=original_add(first,v2);"<<y.SetByOffset("global_idx","original_div(total,3.0)");}
    }
    if(mode_==0){
      f<<"let stride=select(10u,6u,uniforms.stage==1u);let kernel=select(20u,12u,uniforms.stage==1u);"
       <<"let pad=select(5u,3u,uniforms.stage==1u);let input_length=select(count/10u,(count-1u)/6u,uniforms.stage==1u);"
       <<"var target=t;";
      f<<"if(uniforms.stage==1u){target=select(t-1u,1u,t==0u);}"
       <<"let input_channels=select(512u,256u,uniforms.stage==1u);var total:f32=0.0;"
       <<"for(var k=0u;k<kernel;k++){let numerator=i32(target+pad)-i32(k);"
       <<"if(numerator>=0&&u32(numerator)%stride==0u){let pos=u32(numerator)/stride;if(pos<input_length){var column:f32=0.0;"
       <<"for(var ic=0u;ic<input_channels;ic++){let value="<<inputs[0]->GetByOffset("ic*"+GetElementAt("uniforms.input_capacity","0u",3)+"+pos")<<";"
       <<"let negative=original_mul(value,bitcast<f32>(0x3dcccccdu));let activated=select(negative,value,value>=0.0);"
       <<"column=original_add(column,original_mul(activated,"<<w->GetByOffset("(ic*uniforms.channels+oc)*kernel+k")<<"));}"
       <<"total=original_add(total,column);}}}"
       <<y.SetByOffset("global_idx","original_add(total,"+b->GetByOffset("oc")+")");
    }
    if(mode_==3){
      f<<"var total:f32=0.0;for(var ic=0u;ic<128u;ic++){for(var k=0u;k<7u;k++){"
       <<"let pos=i32(t)+i32(k)-3;var activated:f32=0.0;if(pos>=0&&pos<i32(count)){let value="
       <<inputs[0]->GetByOffset("ic*"+GetElementAt("uniforms.input_capacity","0u",3)+"+u32(pos)")<<";"
       <<"let negative=original_mul(value,bitcast<f32>(0x3c23d70au));activated=select(negative,value,value>=0.0);}"
       <<"total=original_add(total,original_mul(activated,"<<w->GetByOffset("(oc*128u+ic)*7u+k")<<"));}}"
       <<y.SetByOffset("global_idx","original_add(total,"+b->GetByOffset("oc")+")");
    }
    if(mode_==4){
      f<<"let bin=oc%11u;let magnitude_log="<<inputs[0]->GetByOffset("bin*"+GetElementAt("uniforms.input_capacity","0u",3)+"+t")<<";"
       <<"let phase_value="<<inputs[0]->GetByOffset("(bin+11u)*"+GetElementAt("uniforms.input_capacity","0u",3)+"+t")<<";"
       <<"let mb=bitcast<u32>(magnitude_log);let ma=mb&0x7fffffffu;var amplitude:f32=bitcast<f32>(0x7fc00000u);"
       <<"if(ma<0x7f800000u){if(magnitude_log>=bitcast<f32>(0x42b17218u)){amplitude=bitcast<f32>(0x7f800000u);}else{amplitude=exp(magnitude_log);}}else if(ma==0x7f800000u){amplitude=select(bitcast<f32>(0x7f800000u),0.0,(mb&0x80000000u)!=0u);}"
       <<"let pa=bitcast<u32>(phase_value)&0x7fffffffu;var phase:f32=bitcast<f32>(0x7fc00000u);"
       <<"if(pa<0x7f800000u){phase=sin(phase_value);}var wave:f32=bitcast<f32>(0x7fc00000u);"
       <<"if((bitcast<u32>(phase)&0x7fffffffu)<0x7f800000u){if(oc<11u){wave=cos(phase);}else{wave=sin(phase);}}"
       <<y.SetByOffset("global_idx","original_mul(amplitude,wave)");
    }
    return Status::OK();
  }
 private:
  const uint32_t mode_,pairs_;
};
class GeneratorEdge : public WebGpuKernel {
 public:
  GeneratorEdge(const OpKernelInfo& i,uint32_t mode):WebGpuKernel{i},mode_(mode),
      stage_(i.GetAttrOrDefault<int64_t>("stage",-1)),capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    ORT_RETURN_IF_NOT((stage_==0||stage_==1)&&capacity_>0&&capacity_<=12288001&&(mode_!=3||stage_==1),
                      "RW07 generator edge closed stage/capacity");
    const uint32_t pairs=mode_==1?2u:(mode_==2?3u:1u);
    const uint32_t channels=mode_==3?22u:(stage_==0?256u:128u);
    const uint32_t source_channels=mode_==0?(stage_==0?512u:256u):(mode_==3?128u:channels);
    const uint32_t minimum=stage_==0?20u:121u,maximum=stage_==0?2048000u:12288001u;
    const uint32_t modulus=stage_==0?20u:120u,remainder=stage_==0?0u:1u;
    const uint32_t source_min=mode_==0?(stage_==0?2u:20u):minimum;
    const uint32_t source_max=mode_==0?(stage_==0?204800u:2048000u):maximum;
    const uint32_t source_mod=mode_==0?(stage_==0?2u:20u):modulus,source_rem=mode_==0?0u:remainder;
    const uint32_t role=mode_==0?(stage_==0?50u:62u):(mode_==1?(stage_==0?51u:63u):(mode_==2?(stage_==0?61u:73u):74u));
    std::array<const Tensor*,3> x{},e{};std::array<uint32_t,3> capacities{},paired{};
    std::array<uint32_t,30> source_words{};
    const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();
    ORT_RETURN_IF_NOT(session&&run,"RW07 generator edge current execution identity");
    const auto& limits=c.DeviceLimits();
    auto binding=[&](const Tensor* t)->Status{
      ORT_RETURN_IF_NOT(t&&t->DataRaw()&&t->SizeInBytes()>0&&t->SizeInBytes()<=limits.maxBufferSize&&
                            t->SizeInBytes()<=limits.maxStorageBufferBindingSize,"RW07 generator edge binding descriptor");
      const uint64_t actual=wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
      ORT_RETURN_IF_NOT(actual>=t->SizeInBytes()&&actual<=limits.maxBufferSize&&actual<=limits.maxStorageBufferBindingSize,
                        "RW07 generator edge actual backing");return Status::OK();
    };
    for(uint32_t j=0;j<pairs;j++){
      x[j]=c.Input<Tensor>(2*j);e[j]=c.Input<Tensor>(2*j+1);
      ORT_RETURN_IF_NOT(Gpu(x[j])&&x[j]->IsDataType<float>()&&x[j]->Shape().NumDimensions()==3&&
                            x[j]->Shape()[0]==1&&x[j]->Shape()[1]==source_channels&&x[j]->Shape()[2]>0&&
                            x[j]->Shape()[2]<=12288001&&Gpu(e[j])&&e[j]->IsDataType<uint32_t>()&&e[j]->Shape()==TensorShape{32},
                        "RW07 generator edge source pair descriptors");
      ORT_RETURN_IF_ERROR(binding(x[j]));ORT_RETURN_IF_ERROR(binding(e[j]));capacities[j]=uint32_t(x[j]->Shape()[2]);
      M0FullGpuAllocationIdentity extent,data,pair;
      ORT_RETURN_IF_NOT(c.M0FullGpuDescribeResourceV2(*e[j],extent)&&c.M0FullGpuDescribeResourceV2(*x[j],data),
                        "RW07 generator edge actual source resources");
      const uint32_t source_role=mode_==0?(stage_==0?24u:61u):(mode_==1?(j==0?(stage_==0?50u:62u):(stage_==0?44u:45u)):
          (mode_==2?(stage_==0?54u:66u)+3*j:73u));
      paired[j]=uint32_t(c.M0FullGpuDescribePairV2(*e[j],*x[j],pair)&&pair.session==session&&pair.run==run&&
          pair.generation==extent.generation&&pair.data_generation==data.generation&&pair.producer_role==source_role&&pair.producer_instance==source_role+1001);
      const m0_extent_abi2::OpaqueOwnership owner{session,run,extent.generation,data.generation,source_role,source_role+1001};
      ORT_RETURN_IF_NOT(owner.Valid(),"RW07 generator edge source ownership descriptor");
      const auto words=owner.UniformWords();std::copy(words.begin(),words.end(),source_words.begin()+10*j);
    }
    const Tensor* w=nullptr;const Tensor* b=nullptr;
    if(mode_==0||mode_==3){
      w=c.Input<Tensor>(2);b=c.Input<Tensor>(3);
      const uint32_t kernel=mode_==3?7u:(stage_==0?20u:12u);
      const TensorShape ws=mode_==0?TensorShape({source_channels,channels,kernel}):TensorShape({channels,source_channels,kernel});
      ORT_RETURN_IF_NOT(Gpu(w)&&w->IsDataType<float>()&&w->Shape()==ws&&Gpu(b)&&b->IsDataType<float>()&&b->Shape()==TensorShape({channels}),
                        "RW07 generator edge exact original immutable parameters");
      ORT_RETURN_IF_ERROR(binding(w));ORT_RETURN_IF_ERROR(binding(b));
    }
    const uint64_t groups=(uint64_t(channels)*capacity_+63)/64,limit=limits.maxComputeWorkgroupsPerDimension;
    const uint64_t bytes=uint64_t(channels)*capacity_*4;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=5&&limits.maxComputeWorkgroupSizeX>=64&&
        limits.maxComputeInvocationsPerWorkgroup>=64&&limit>0&&groups<=limit*limit&&bytes<=limits.maxBufferSize&&
        bytes<=limits.maxStorageBufferBindingSize,"RW07 generator edge output resources");
    auto* y=c.Output(0,TensorShape{1,channels,capacity_});auto* out=c.Output(1,TensorShape{32});
    ORT_RETURN_IF_NOT(Gpu(y)&&y->IsDataType<float>()&&Gpu(out)&&out->IsDataType<uint32_t>()&&y->DataRaw()!=out->DataRaw(),
                      "RW07 generator edge distinct output allocations");
    for(const auto* target:{y,out}){
      ORT_RETURN_IF_ERROR(binding(target));for(uint32_t j=0;j<pairs;j++){
        ORT_RETURN_IF_NOT(target->DataRaw()!=x[j]->DataRaw()&&target->DataRaw()!=e[j]->DataRaw(),"RW07 generator edge source/output alias");}
      if(w){ORT_RETURN_IF_NOT(target->DataRaw()!=w->DataRaw()&&target->DataRaw()!=b->DataRaw(),"RW07 generator edge parameter/output alias");}
    }
    M0FullGpuAllocationIdentity output;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out,*y,role,role+1001,output),"RW07 generator edge output claim");
    const m0_extent_abi2::OpaqueOwnership owner{session,run,output.generation,output.data_generation,role,role+1001};
    ORT_RETURN_IF_NOT(owner.Valid(),"RW07 generator edge output ownership");const auto words=owner.UniformWords();
    const auto identity=gsl::span<const uint32_t>{words.data(),words.size()};
    const auto capspan=gsl::span<const uint32_t>{capacities.data(),capacities.size()};
    EdgeGuard guard{mode_,pairs};guard.CacheHint(std::to_string(mode_));
    for(uint32_t j=0;j<pairs;j++){guard.AddInput({e[j],ProgramTensorMetadataDependency::None});}
    guard.AddOutput({out,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(1)
      .AddUniformVariables({capspan,source_channels,source_min,source_max,source_mod,source_rem,
          gsl::span<const uint32_t>{paired.data(),paired.size()},gsl::span<const uint32_t>{source_words.data(),source_words.size()},
          uint32_t(capacity_),channels,uint32_t(stage_),identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    EdgeData data{mode_,pairs};data.CacheHint(std::to_string(mode_));
    for(uint32_t j=0;j<pairs;j++){data.AddInput({x[j],ProgramTensorMetadataDependency::None});}
    data.AddInput({out,ProgramTensorMetadataDependency::None});
    if(w){data.AddInputs({{w,ProgramTensorMetadataDependency::None},{b,ProgramTensorMetadataDependency::None}});}
    const uint64_t gx=std::min(groups,limit),gy=(groups+gx-1)/gx;
    if(mode_==3){
      auto convolved=c.CreateGPUTensor(x[0]->DataType(),TensorShape{1,22,capacity_});
      ORT_RETURN_IF_ERROR(binding(&convolved));
      ORT_RETURN_IF_NOT(convolved.DataRaw()!=y->DataRaw()&&convolved.DataRaw()!=out->DataRaw()&&
          convolved.DataRaw()!=x[0]->DataRaw()&&convolved.DataRaw()!=e[0]->DataRaw()&&
          convolved.DataRaw()!=w->DataRaw()&&convolved.DataRaw()!=b->DataRaw(),"RW07 generator projection private scratch alias");
      data.AddOutput({&convolved,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
        .SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
        .AddUniformVariables({capspan,uint32_t(capacity_),channels,minimum,maximum,modulus,remainder,uint32_t(stage_),identity});
      ORT_RETURN_IF_ERROR(c.RunProgram(data));
      EdgeData spectral{4,1};spectral.CacheHint("spectral");
      const std::array<uint32_t,3> scratch_cap{uint32_t(capacity_),0u,0u};
      spectral.AddInputs({{&convolved,ProgramTensorMetadataDependency::None},{out,ProgramTensorMetadataDependency::None}})
        .AddOutput({y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
        .AddUniformVariables({gsl::span<const uint32_t>{scratch_cap.data(),scratch_cap.size()},uint32_t(capacity_),channels,
            minimum,maximum,modulus,remainder,uint32_t(stage_),identity});
      ORT_RETURN_IF_ERROR(c.RunProgram(spectral));
    }else{
      data.AddOutput({y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
        .AddUniformVariables({capspan,uint32_t(capacity_),channels,minimum,maximum,modulus,remainder,uint32_t(stage_),identity});
      ORT_RETURN_IF_ERROR(c.RunProgram(data));
    }
    LOGS(c.Logger(),INFO)<<"M0_RW07_GENERATOR_EDGE_V1 mode="<<mode_<<" stage="<<stage_<<" role="<<role
      <<" session="<<session<<" run="<<run<<" GPU_dispatches="<<(mode_==3?3:2)<<" host_payload_reads=0";
    return Status::OK();
  }
 private:
  const uint32_t mode_;
  const int64_t stage_,capacity_;
};
class PacketProgram final : public Program<PacketProgram> {
 public:
  PacketProgram() : Program{"RW07GeneratorProjectionTerminalPacketV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"minimum", ProgramUniformVariableDataType::Uint32},
      {"maximum", ProgramUniformVariableDataType::Uint32},
      {"modulus", ProgramUniformVariableDataType::Uint32},
      {"remainder", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& p = s.AddOutput("packet", ShaderUsage::UseValueTypeAlias);
    ValidOutput(s, e);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<18u;i++){" << p.SetByOffset("i", "0u") << "}"
      << p.SetByOffset("0u", "103u")
      << "if(" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=3u){return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.identity", "i", 10) << "){return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){return;}" << p.SetByOffset("0u", "status") << "return;}"
      << "if(!valid_output()){return;}" << p.SetByOffset("0u", "0u") << p.SetByOffset("1u", "3u")
      << "for(var i=0u;i<6u;i++){" << p.SetByOffset("2u+i", e.GetByOffset("2u+i")) << "}";
    return Status::OK();
  }
};
class Packet final : public WebGpuKernel {
 public:
  explicit Packet(const OpKernelInfo& i) : WebGpuKernel{i} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e = c.Input<Tensor>(0); const auto* y = c.Input<Tensor>(1);
    const uint32_t channels = 22u;
    ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          Gpu(y) && y->IsDataType<float>() && y->Shape().NumDimensions() == 3 &&
                          y->Shape()[0] == 1 && y->Shape()[1] == channels && y->Shape()[2] > 0 && y->Shape()[2] <= 12288001,
                      "RW07 generator projection packet physical descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*e, *y, pair) && pair.session == session &&
                          pair.run == run && pair.producer_role == 74u && pair.producer_instance == 1075u,
                      "RW07 generator projection terminal actual pair");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 generator projection packet resources");
    auto binding = [&](const Tensor* t) -> Status {
      ORT_RETURN_IF_NOT(t->DataRaw() && t->SizeInBytes() <= limits.maxBufferSize && t->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 generator projection packet binding descriptor");
      const uint64_t bytes = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
      ORT_RETURN_IF_NOT(bytes >= t->SizeInBytes() && bytes <= limits.maxBufferSize && bytes <= limits.maxStorageBufferBindingSize,
                        "RW07 generator projection packet actual backing");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(binding(e)); ORT_RETURN_IF_ERROR(binding(y));
    auto* p = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p) && p->IsDataType<uint32_t>() && p->DataRaw() != e->DataRaw() && p->DataRaw() != y->DataRaw(),
                      "RW07 generator projection distinct packet output");
    ORT_RETURN_IF_ERROR(binding(p));
    const m0_extent_abi2::OpaqueOwnership owner{session, run, pair.generation, pair.data_generation, 74u, 1075u};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 generator projection packet opaque ownership");
    const auto words = owner.UniformWords();
    PacketProgram program;
    program.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({p, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[2]), uint32_t(y->Shape()[2]), channels,
                              121u, 12288001u,
                              120u, 1u, gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p), "RW07 generator projection packet map fault registration");
    return Status::OK();
  }
};
}  // namespace
KernelCreateInfo CreateM0FullGpuGeneratorUpsampleExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<GeneratorEdge>(i,0);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorUpsampleExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorJoinExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<GeneratorEdge>(i,1);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorJoinExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorMergeExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<GeneratorEdge>(i,2);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorMergeExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorProjectionExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<GeneratorEdge>(i,3);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorProjectionExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorProjectionTerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<Packet>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorProjectionTerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
}  // namespace onnxruntime::webgpu
