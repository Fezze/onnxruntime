// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original generator resblocks.0..5: three exact AdaIN/Snake residual blocks per path.
#include <algorithm>
#include <array>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"

namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
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
class Guard final : public Program<Guard> {
 public:
  Guard() : Program{"RW07GeneratorResidualGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"minimum", ProgramUniformVariableDataType::Uint32},
      {"maximum", ProgramUniformVariableDataType::Uint32},
      {"modulus", ProgramUniformVariableDataType::Uint32},
      {"remainder", ProgramUniformVariableDataType::Uint32},
      {"source_pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << o.SetByOffset("i","0u") << "}"
      << o.SetByOffset("0u","2u") << o.SetByOffset("1u","3u")
      << "for(var i=0u;i<10u;i++){" << o.SetByOffset("22u+i",GetElementAt("uniforms.identity","i",10)) << "}"
      << "if(uniforms.source_pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||"
      << e.GetByOffset("1u") << "!=3u){" << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity","i",10) << "){" << o.SetByOffset("20u","103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}}if(" << e.GetByOffset("21u") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}" << o.SetByOffset("20u","status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("4u") << "!=uniforms.channels||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}}let n=" << e.GetByOffset("6u") << ";"
      << "if(n<uniforms.minimum||n>uniforms.maximum||n%uniforms.modulus!=uniforms.remainder||" << e.GetByOffset("18u") << "!=uniforms.channels*n){"
      << o.SetByOffset("20u","103u") << "return;}"
      << "let count=n;"
      << "let channels=uniforms.channels;"
      << "if(n>uniforms.input_capacity||count>uniforms.output_capacity){"
      << o.SetByOffset("20u","102u") << "return;}"
      << o.SetByOffset("2u","1u") << o.SetByOffset("4u","channels") << o.SetByOffset("6u","count")
      << o.SetByOffset("18u","channels*count") << o.SetByOffset("21u","63u");
    return Status::OK();
  }
};
class GeneratorNormSnake final : public Program<GeneratorNormSnake> {
 public:
  GeneratorNormSnake() : Program{"RW07NoiseResidualNormSnakeV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"minimum", ProgramUniformVariableDataType::Uint32},
      {"maximum", ProgramUniformVariableDataType::Uint32},
      {"modulus", ProgramUniformVariableDataType::Uint32},
      {"remainder", ProgramUniformVariableDataType::Uint32},
      {"kernel", ProgramUniformVariableDataType::Uint32},
      {"dilation", ProgramUniformVariableDataType::Uint32},
      {"residual_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("carrier",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& style=s.AddInput("style",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("fc_weight",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("fc_bias",ShaderUsage::UseValueTypeAlias);
    const auto& alpha=s.AddInput("alpha",ShaderUsage::UseValueTypeAlias);
    const auto& reciprocal=s.AddInput("reciprocal",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("activated",ShaderUsage::UseValueTypeAlias);
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
    f << "if(global_idx>=uniforms.channels){return;}let channel=global_idx;"
      << "for(var t=0u;t<uniforms.output_capacity;t++){"
      << y.SetByOffset("channel*uniforms.output_capacity+t","0.0") << "}if(!valid_output()){return;}"
      << "let count=" << e.GetByOffset("6u") << ";var sum:f32=0.0;"
      << "for(var t=0u;t<count;t++){sum=original_add(sum," << x.GetByOffset("channel*uniforms.input_capacity+t") << ");}"
      << "let mean=original_div(sum,f32(count));var squares:f32=0.0;"
      << "for(var t=0u;t<count;t++){let delta=original_sub(" << x.GetByOffset("channel*uniforms.input_capacity+t") << ",mean);"
      << "squares=original_add(squares,original_mul(delta,delta));}"
      << "let variance=original_div(squares,f32(count));let denominator=original_sqrt(original_add(variance,bitcast<f32>(0x3727c5acu)));"
      << "var gs:f32=0.0;var bs:f32=0.0;for(var k=0u;k<128u;k++){let value=" << style.GetByOffset("k") << ";"
      << "gs=original_add(gs,original_mul(value," << w.GetByOffset("channel*128u+k") << "));"
      << "bs=original_add(bs,original_mul(value," << w.GetByOffset("(channel+uniforms.channels)*128u+k") << "));}"
      << "let gamma=original_add(original_add(gs," << b.GetByOffset("channel") << "),1.0);"
      << "let beta=original_add(bs," << b.GetByOffset("channel+uniforms.channels") << ");"
      << "let a=" << alpha.GetByOffset("channel") << ";let reciprocal_a=" << reciprocal.GetByOffset("channel") << ";"
      << "for(var t=0u;t<count;t++){let centered=original_sub(" << x.GetByOffset("channel*uniforms.input_capacity+t") << ",mean);"
      << "let normalized=original_div(centered,denominator);let affine=original_add(original_mul(gamma,normalized),beta);"
      << "let angle=original_mul(a,affine);let bits=bitcast<u32>(angle)&0x7fffffffu;var sine:f32=bitcast<f32>(0x7fc00000u);"
      << "if(bits<0x7f800000u){sine=sin(angle);}let square=original_mul(sine,sine);"
      << "let result=original_add(affine,original_mul(reciprocal_a,square));"
      << y.SetByOffset("channel*uniforms.output_capacity+t","result") << "}";
    return Status::OK();
  }
};
class GeneratorResidualConv final : public Program<GeneratorResidualConv> {
 public:
  explicit GeneratorResidualConv(bool residual) : Program{"RW07GeneratorResidualConvV1"}, residual_(residual) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"minimum", ProgramUniformVariableDataType::Uint32},
      {"maximum", ProgramUniformVariableDataType::Uint32},
      {"modulus", ProgramUniformVariableDataType::Uint32},
      {"remainder", ProgramUniformVariableDataType::Uint32},
      {"kernel", ProgramUniformVariableDataType::Uint32},
      {"dilation", ProgramUniformVariableDataType::Uint32},
      {"residual_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("carrier",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("bias",ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* prior=residual_? &s.AddInput("prior",ShaderUsage::UseValueTypeAlias):nullptr;
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
    f << "if(global_idx>=uniforms.channels*uniforms.output_capacity){return;}"
      << y.SetByOffset("global_idx","0.0") << "if(!valid_output()){return;}"
      << "let count=" << e.GetByOffset("6u") << ";let t=global_idx%uniforms.output_capacity;"
      << "let oc=global_idx/uniforms.output_capacity;if(t>=count){return;}var total:f32=0.0;"
      << "let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
      << "for(var ic=0u;ic<uniforms.channels;ic++){for(var k=0u;k<uniforms.kernel;k++){"
      << "let pos=i32(t)+i32(k*uniforms.dilation)-pad;var value:f32=0.0;"
      << "if(pos>=0&&pos<i32(count)){value=" << x.GetByOffset("ic*uniforms.input_capacity+u32(pos)") << ";}"
      << "total=original_add(total,original_mul(value," << w.GetByOffset("(oc*uniforms.channels+ic)*uniforms.kernel+k") << "));}}"
      << "let convolved=original_add(total," << b.GetByOffset("oc") << ");";
    if(residual_){f << y.SetByOffset("global_idx","original_add(convolved,"+prior->GetByOffset("oc*uniforms.residual_capacity+t")+")");}
    else {f << y.SetByOffset("global_idx","convolved");}
    return Status::OK();
  }
 private:
  const bool residual_;
};
class GeneratorResidualExtent final : public WebGpuKernel {
 public:
  explicit GeneratorResidualExtent(const OpKernelInfo& i) : WebGpuKernel{i},
      capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)),
      branch_(i.GetAttrOrDefault<int64_t>("branch",-1)), block_(i.GetAttrOrDefault<int64_t>("block",-1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    ORT_RETURN_IF_NOT((branch_>=0&&branch_<6)&&block_>=0&&block_<3&&capacity_>0&&capacity_<=12288001,
                      "RW07 generator residual closed branch/block/capacity");
    const uint32_t channels=branch_<3?256u:128u,kernel=branch_%3==0?3u:(branch_%3==1?7u:11u);
    const auto* x=c.Input<Tensor>(0);const auto* e=c.Input<Tensor>(1);const auto* style=c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(Gpu(x)&&x->IsDataType<float>()&&x->Shape().NumDimensions()==3&&
                          x->Shape()[0]==1&&x->Shape()[1]==channels&&x->Shape()[2]>0&&x->Shape()[2]<=12288001&&
                          Gpu(e)&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape{32}&&
                          Gpu(style)&&style->IsDataType<float>()&&style->Shape()==TensorShape({1,256}),
                      "RW07 generator residual carrier/extent/style descriptors");
    std::array<const Tensor*,12> params{};
    for(size_t i=0;i<params.size();i++){params[i]=c.Input<Tensor>(int(i)+3);}
    for(size_t h=0;h<2;h++){
      const size_t p=6*h;
      const std::array<TensorShape,6> shapes{TensorShape({2*channels,128}),TensorShape({2*channels}),
          TensorShape({1,channels,1}),TensorShape({1,channels,1}),TensorShape({channels,channels,kernel}),TensorShape({channels})};
      for(size_t k=0;k<6;k++){
        ORT_RETURN_IF_NOT(Gpu(params[p+k])&&params[p+k]->IsDataType<float>()&&params[p+k]->Shape()==shapes[k],
                          "RW07 generator residual exact immutable parameters");
      }
    }
    const auto& limits=c.DeviceLimits();const uint64_t groups=(uint64_t(channels)*capacity_+63)/64;
    const uint64_t limit=limits.maxComputeWorkgroupsPerDimension, bytes=uint64_t(channels)*capacity_*4;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=8&&limits.maxComputeWorkgroupSizeX>=64&&
                          limits.maxComputeInvocationsPerWorkgroup>=64&&limit>=4&&groups<=limit*limit&&
                          bytes<=limits.maxBufferSize&&bytes<=limits.maxStorageBufferBindingSize,
                      "RW07 generator residual resource and dispatch capacity");
    auto binding=[&](const Tensor* t)->Status{
      ORT_RETURN_IF_NOT(t&&t->DataRaw()&&t->SizeInBytes()>0&&t->SizeInBytes()<=limits.maxBufferSize&&
                            t->SizeInBytes()<=limits.maxStorageBufferBindingSize,"RW07 generator residual binding descriptor");
      const uint64_t actual=wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
      ORT_RETURN_IF_NOT(actual>=t->SizeInBytes()&&actual<=limits.maxBufferSize&&actual<=limits.maxStorageBufferBindingSize,
                        "RW07 generator residual backing buffer");return Status::OK();
    };
    for(const auto* t:{x,e,style}){ORT_RETURN_IF_ERROR(binding(t));}
    for(const auto* t:params){ORT_RETURN_IF_ERROR(binding(t));}
    const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity source,carrier,pair,output;
    ORT_RETURN_IF_NOT(session&&run&&c.M0FullGpuDescribeResourceV2(*e,source)&&c.M0FullGpuDescribeResourceV2(*x,carrier),
                      "RW07 generator residual actual resources");
    const uint32_t role=uint32_t((branch_<3?52:64)+3*(branch_%3)+block_),instance=role+1001;
    const uint32_t source_role=block_==0?uint32_t(branch_<3?51:63):role-1;
    const uint32_t source_instance=source_role+1001;
    const bool paired=c.M0FullGpuDescribePairV2(*e,*x,pair)&&pair.session==session&&pair.run==run&&
                      pair.generation==source.generation&&pair.data_generation==carrier.generation&&
                      pair.producer_role==source_role&&pair.producer_instance==source_instance;
    auto* y=c.Output(0,TensorShape{1,channels,capacity_});auto* out=c.Output(1,TensorShape{32});
    ORT_RETURN_IF_NOT(Gpu(y)&&y->IsDataType<float>()&&Gpu(out)&&out->IsDataType<uint32_t>(),"RW07 generator residual output types");
    auto activated=c.CreateGPUTensor(x->DataType(),TensorShape{1,channels,capacity_});
    auto conv=c.CreateGPUTensor(x->DataType(),TensorShape{1,channels,capacity_});
    auto second=c.CreateGPUTensor(x->DataType(),TensorShape{1,channels,capacity_});
    std::array<const Tensor*,5> outputs{y,out,&activated,&conv,&second};
    for(size_t i=0;i<outputs.size();i++){
      ORT_RETURN_IF_ERROR(binding(outputs[i]));
      for(size_t j=0;j<i;j++){ORT_RETURN_IF_NOT(outputs[i]->DataRaw()!=outputs[j]->DataRaw(),"RW07 generator residual scratch/output alias");}
      for(const auto* t:{x,e,style}){ORT_RETURN_IF_NOT(outputs[i]->DataRaw()!=t->DataRaw(),"RW07 generator residual source/output alias");}
      for(const auto* t:params){ORT_RETURN_IF_NOT(outputs[i]->DataRaw()!=t->DataRaw(),"RW07 generator residual parameter/output alias");}
    }
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out,*y,role,instance,output),"RW07 generator residual output claim");
    const m0_extent_abi2::OpaqueOwnership src{session,run,source.generation,carrier.generation,source_role,source_instance};
    const m0_extent_abi2::OpaqueOwnership dst{session,run,output.generation,output.data_generation,role,instance};
    ORT_RETURN_IF_NOT(src.Valid()&&dst.Valid(),"RW07 generator residual opaque ownership");
    const auto sw=src.UniformWords(),ow=dst.UniformWords();
    const auto source_identity=gsl::span<const uint32_t>{sw.data(),sw.size()},identity=gsl::span<const uint32_t>{ow.data(),ow.size()};
    const uint32_t pc=uint32_t(x->Shape()[2]),oc=uint32_t(capacity_);
    const uint32_t minimum=branch_<3?20u:121u, maximum=branch_<3?2048000u:12288001u;
    const uint32_t modulus=branch_<3?20u:120u,remainder=branch_<3?0u:1u;
    Guard guard;guard.AddInput({e,ProgramTensorMetadataDependency::None}).AddOutput({out,ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({pc,oc,channels,minimum,maximum,modulus,remainder,uint32_t(paired),source_identity,identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const uint64_t gx=std::min(groups,limit),gy=(groups+gx-1)/gx;
    for(uint32_t h=0;h<2;h++){
      const auto* input=h==0?x:&conv;auto* normout=h==0?&activated:&second;auto* convout=h==0?&conv:y;
      const uint32_t inputcap=h==0?pc:oc, dilation=h==0?uint32_t(1+2*block_):1u;
      GeneratorNormSnake norm;norm.AddInputs({{input,ProgramTensorMetadataDependency::None},{out,ProgramTensorMetadataDependency::None},
          {style,ProgramTensorMetadataDependency::None},{params[6*h],ProgramTensorMetadataDependency::None},
          {params[6*h+1],ProgramTensorMetadataDependency::None},{params[6*h+2],ProgramTensorMetadataDependency::None},
          {params[6*h+3],ProgramTensorMetadataDependency::None}}).AddOutput({normout,ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(64).SetDispatchGroupSize((channels+63)/64)
          .AddUniformVariables({inputcap,oc,channels,minimum,maximum,modulus,remainder,kernel,dilation,pc,identity});
      ORT_RETURN_IF_ERROR(c.RunProgram(norm));
      GeneratorResidualConv convolution{h==1};convolution.CacheHint(h==0?"first":"residual");
      convolution.AddInputs({{normout,ProgramTensorMetadataDependency::None},{out,ProgramTensorMetadataDependency::None},
          {params[6*h+4],ProgramTensorMetadataDependency::None},{params[6*h+5],ProgramTensorMetadataDependency::None}});
      if(h==1){convolution.AddInput({x,ProgramTensorMetadataDependency::None});}
      convolution.AddOutput({convout,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
          .SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
          .AddUniformVariables({oc,oc,channels,minimum,maximum,modulus,remainder,kernel,dilation,pc,identity});
      ORT_RETURN_IF_ERROR(c.RunProgram(convolution));
    }
    LOGS(c.Logger(),INFO)<<"M0_RW07_GENERATOR_RESIDUAL_V1 session="<<session<<" run="<<run<<" branch="<<branch_
      <<" block="<<block_<<" source_pair_valid="<<paired<<" role="<<role<<" instance="<<instance
      <<" GPU_dispatches=5 host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_,branch_,block_;
};
class PacketProgram final : public Program<PacketProgram> {
 public:
  PacketProgram() : Program{"RW07GeneratorResidualTerminalPacketV1"} {}
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
  explicit Packet(const OpKernelInfo& i) : WebGpuKernel{i}, branch_(i.GetAttrOrDefault<int64_t>("branch", -1)), block_(i.GetAttrOrDefault<int64_t>("block", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e = c.Input<Tensor>(0); const auto* y = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT((branch_ >= 0 && branch_ < 6) && block_ >= 0 && block_ < 3, "RW07 generator residual packet branch/block");
    const uint32_t channels = branch_ < 3 ? 256u : 128u;
    ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          Gpu(y) && y->IsDataType<float>() && y->Shape().NumDimensions() == 3 &&
                          y->Shape()[0] == 1 && y->Shape()[1] == channels && y->Shape()[2] > 0 && y->Shape()[2] <= 12288001,
                      "RW07 generator residual after1600 packet physical descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*e, *y, pair) && pair.session == session &&
                          pair.run == run && pair.producer_role == uint32_t((branch_ < 3 ? 52 : 64) + 3 * (branch_ % 3) + block_) && pair.producer_instance == uint32_t((branch_ < 3 ? 1053 : 1065) + 3 * (branch_ % 3) + block_),
                      "RW07 generator residual after1600 terminal actual pair");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 generator residual after1600 packet resources");
    auto binding = [&](const Tensor* t) -> Status {
      ORT_RETURN_IF_NOT(t->DataRaw() && t->SizeInBytes() <= limits.maxBufferSize && t->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 generator residual after1600 packet binding descriptor");
      const uint64_t bytes = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
      ORT_RETURN_IF_NOT(bytes >= t->SizeInBytes() && bytes <= limits.maxBufferSize && bytes <= limits.maxStorageBufferBindingSize,
                        "RW07 generator residual after1600 packet actual backing");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(binding(e)); ORT_RETURN_IF_ERROR(binding(y));
    auto* p = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p) && p->IsDataType<uint32_t>() && p->DataRaw() != e->DataRaw() && p->DataRaw() != y->DataRaw(),
                      "RW07 generator residual after1600 distinct packet output");
    ORT_RETURN_IF_ERROR(binding(p));
    const m0_extent_abi2::OpaqueOwnership owner{session, run, pair.generation, pair.data_generation, uint32_t((branch_ < 3 ? 52 : 64) + 3 * (branch_ % 3) + block_), uint32_t((branch_ < 3 ? 1053 : 1065) + 3 * (branch_ % 3) + block_)};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 generator residual after1600 packet opaque ownership");
    const auto words = owner.UniformWords();
    PacketProgram program;
    program.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({p, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[2]), uint32_t(y->Shape()[2]), channels,
                              branch_ < 3 ? 20u : 121u, branch_ < 3 ? 2048000u : 12288001u,
                              branch_ < 3 ? 20u : 120u, branch_ < 3 ? 0u : 1u, gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p), "RW07 generator residual after1600 packet map fault registration");
    return Status::OK();
  }
 private:
  const int64_t branch_, block_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuGeneratorResidualExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<GeneratorResidualExtent>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("GeneratorResidualExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorResidualTerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Packet>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("GeneratorResidualTerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
