// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::webgpu::m0_reconstruction_detail {
inline void ValidOutput(ShaderHelper& s,const ShaderVariableHelper& e){
 auto& f=s.AdditionalImplementation();
 f<<"fn valid_output()->bool{if("<<e.GetByOffset("0u")<<"!=2u||"<<e.GetByOffset("1u")<<"!=1u||"<<e.GetByOffset("20u")<<"!=0u||"<<e.GetByOffset("21u")<<"!=63u){return false;}"
  <<"for(var i=0u;i<10u;i++){if("<<e.GetByOffset("22u+i")<<"!="<<GetElementAt("uniforms.identity","i",10)<<"){return false;}}"
  <<"for(var i=3u;i<18u;i++){if("<<e.GetByOffset("i")<<"!=0u){return false;}}"
  <<"let n="<<e.GetByOffset("2u")<<";return n>=600u&&n<=61440000u&&n%600u==0u&&n<=uniforms.output_capacity&&n/5u+1u<=uniforms.input_capacity&&"
  <<e.GetByOffset("18u")<<"==n&&"<<e.GetByOffset("19u")<<"==0u;}\n";
}
inline void Arithmetic(ShaderHelper& s){
 s.AdditionalImplementation()<<R"WGSL(
fn original_f32_from_bits(bits:u32)->f32 {return bitcast<f32>(bits);}
fn original_mul(a:f32,b:f32)->f32{
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0x7f800000u)||(bb==0u&&aa==0x7f800000u)){return original_f32_from_bits(0x7fc00000u);}
 if(aa==0x7f800000u||bb==0x7f800000u){return bitcast<f32>(((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)|0x7f800000u);}return a*b;
}
fn original_add(a:f32,b:f32)->f32{
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0x7f800000u&&bb==0x7f800000u&&((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)!=0u)){return original_f32_from_bits(0x7fc00000u);}
 if(aa==0x7f800000u){return a;}if(bb==0x7f800000u){return b;}return a+b;
}
fn original_div(a:f32,b:f32)->f32{
 let ua=bitcast<u32>(a);let ub=bitcast<u32>(b);let aa=ua&0x7fffffffu;let bb=ub&0x7fffffffu;let sign=(ua^ub)&0x80000000u;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0u)||(aa==0x7f800000u&&bb==0x7f800000u)){return original_f32_from_bits(0x7fc00000u);}
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
class GatherSpectrumWindow final:public Program<GatherSpectrumWindow>{
 public:
 GatherSpectrumWindow():Program{"RW07ReconstructionSpectrumWindowGatherV1"}{}
 WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
  {"input_capacity",ProgramUniformVariableDataType::Uint32},{"output_capacity",ProgramUniformVariableDataType::Uint32},
  {"sample_begin",ProgramUniformVariableDataType::Uint32},{"first_frame",ProgramUniformVariableDataType::Uint32},
  {"frame_count",ProgramUniformVariableDataType::Uint32},{"identity",ProgramUniformVariableDataType::Uint32});
 Status GenerateShaderCode(ShaderHelper& s)const override{
  const auto& x=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
  const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
  const auto& y=s.AddOutput("packed",ShaderUsage::UseValueTypeAlias);
  ValidOutput(s,e);auto& f=s.MainFunctionBody();
  f<<"if(global_idx>=uniforms.frame_count){return;}"<<y.SetByOffset("global_idx","0.0")
   <<"if(!valid_output()||uniforms.sample_begin>="<<e.GetByOffset("2u")<<"){return;}"
   <<"if(uniforms.first_frame+global_idx>="<<e.GetByOffset("2u")<<"/5u+1u){return;}"
   <<y.SetByOffset("global_idx",x.GetByOffset("global_idx"));return Status::OK();
 }
};
class Finish final:public Program<Finish>{
 public:
 Finish():Program{"RW07Original2427To2463DirectOverlapScatterCropV4"}{}
 WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
  {"input_capacity",ProgramUniformVariableDataType::Uint32},{"output_capacity",ProgramUniformVariableDataType::Uint32},
  {"sample_begin",ProgramUniformVariableDataType::Uint32},{"sample_count",ProgramUniformVariableDataType::Uint32},
  {"first_frame",ProgramUniformVariableDataType::Uint32},{"packed_capacity",ProgramUniformVariableDataType::Uint32},
  {"identity",ProgramUniformVariableDataType::Uint32});
 Status GenerateShaderCode(ShaderHelper& s)const override{
  const auto& x=s.AddInput("spectrum",ShaderUsage::UseValueTypeAlias);const auto& basis=s.AddInput("basis",ShaderUsage::UseValueTypeAlias);
  const auto& w=s.AddInput("window",ShaderUsage::UseValueTypeAlias);const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);const auto& y=s.AddOutput("pcm",ShaderUsage::UseValueTypeAlias);
  ValidOutput(s,e);Arithmetic(s);auto& f=s.MainFunctionBody();
  f<<"if(global_idx>=uniforms.sample_count){return;}"<<y.SetByOffset("global_idx","0.0")<<"let output_sample=uniforms.sample_begin+global_idx;if(!valid_output()||output_sample>="<<e.GetByOffset("2u")<<"){return;}let frames="<<e.GetByOffset("2u")<<"/5u+1u;let sample=output_sample+10u;var value:f32=0.0;"
   // Each inverse-Gemm col[k,frame] contributes to only sample=k+5*frame.
   // Keep its complete channel sum separate from Col2imNd's ordered tap sum.
   <<"for(var k=0u;k<20u;k++){if(sample>=k){let delta=sample-k;if(delta%5u==0u){let frame=delta/5u;if(frame<frames){var sum:f32=0.0;"
   <<"for(var c=0u;c<22u;c++){let product:f32=original_mul("<<x.GetByOffset("c*uniforms.packed_capacity+frame-uniforms.first_frame")<<","<<basis.GetByOffset("c*20u+k")<<");sum=original_add(sum,product);}"
   <<"value=original_add(value,sum);}}}}"
   // Original NonZero/ScatterND overwrites only indices1..19, then crop10.
   <<"if(sample<20u){value=original_div(value,"<<w.GetByOffset("sample")<<");}let result=original_mul(value,4.0);"<<y.SetByOffset("global_idx","result");return Status::OK();
 }
};
}  // namespace onnxruntime::webgpu::m0_reconstruction_detail
