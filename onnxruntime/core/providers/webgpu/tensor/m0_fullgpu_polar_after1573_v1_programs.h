// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_trig_v1.h"

namespace onnxruntime::webgpu::m0_polar_detail {
inline void ValidOutput(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_output()->bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=3u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("4u") << "!=22u||" << e.GetByOffset("5u") << "!=0u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let n=" << e.GetByOffset("6u") << ";return n>=121u&&n<=12288001u&&n%120u==1u&&"
    << "n<=uniforms.input_capacity&&n<=uniforms.output_capacity&&" << e.GetByOffset("18u") << "==22u*n;}\n";
}
class Guard final : public Program<Guard> {
 public:
  Guard() : Program{"RW07PolarAfter1573GuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"source_pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("source", ShaderUsage::UseValueTypeAlias);
    const auto& o = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << o.SetByOffset("i", "0u") << "}"
      << o.SetByOffset("0u", "2u") << o.SetByOffset("1u", "3u")
      << "for(var i=0u;i<10u;i++){" << o.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "if(uniforms.source_pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||"
      << e.GetByOffset("1u") << "!=4u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity", "i", 10) << "){" << o.SetByOffset("20u", "103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}if(" << e.GetByOffset("21u") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}" << o.SetByOffset("20u", "status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("5u") << "!=0u||" << e.GetByOffset("6u") << "!=11u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("8u") << "!=2u||"
      << e.GetByOffset("9u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=10u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}let n=" << e.GetByOffset("4u") << ";"
      << "if(n<121u||n>12288001u||n%120u!=1u||" << e.GetByOffset("18u") << "!=22u*n){"
      << o.SetByOffset("20u", "103u") << "return;}"
      << "if(n>uniforms.input_capacity||n>uniforms.output_capacity){" << o.SetByOffset("20u", "102u") << "return;}"
      << o.SetByOffset("2u", "1u") << o.SetByOffset("4u", "22u") << o.SetByOffset("6u", "n")
      << o.SetByOffset("18u", "22u*n") << o.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class Polar final : public Program<Polar> {
 public:
  Polar() : Program{"RW07Original1575Through1597PolarTimeWindowV3"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"frame_begin", ProgramUniformVariableDataType::Uint32},
      {"frame_count", ProgramUniformVariableDataType::Uint32},
      {"input_begin", ProgramUniformVariableDataType::Uint32},
      {"channel", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("spectrum", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("polar", ShaderUsage::UseValueTypeAlias);
    ValidOutput(s, e);
    s.AdditionalImplementation() << m0_trig_detail::Atan;
    s.AdditionalImplementation() << R"WGSL(
fn original_f32_from_bits(bits:u32)->f32 {return bitcast<f32>(bits);}
fn square(x:f32)->f32 {
 let bits=bitcast<u32>(x)&0x7fffffffu;
 if(bits>0x7f800000u){return original_f32_from_bits(0x7fc00000u);}
 if(bits==0x7f800000u){return original_f32_from_bits(0x7f800000u);}
 if(bits==0u){return 0.0;}return x*x;
}
fn original_div(a:f32,b:f32)->f32 {
 let ua=bitcast<u32>(a);let ub=bitcast<u32>(b);
 let aa=ua&0x7fffffffu;let ab=ub&0x7fffffffu;let sign=(ua^ub)&0x80000000u;
 if(aa>0x7f800000u||ab>0x7f800000u){return original_f32_from_bits(0x7fc00000u);}
 if((aa==0u&&ab==0u)||(aa==0x7f800000u&&ab==0x7f800000u)){return original_f32_from_bits(0x7fc00000u);}
 if(ab==0u||aa==0x7f800000u){return bitcast<f32>(sign|0x7f800000u);}
 if(aa==0u||ab==0x7f800000u){return bitcast<f32>(sign);}return a/b;
}
fn original_atan(x:f32)->f32 {
 let bits=bitcast<u32>(x);let a=bits&0x7fffffffu;
 if(a>0x7f800000u){return original_f32_from_bits(0x7fc00000u);}
 if(a==0x7f800000u){return bitcast<f32>((bits&0x80000000u)|0x3fc90fdbu);}
 if(a==0u){return x;}return rw07_atan(x);
}
)WGSL";
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.frame_count){return;}"
      << "let bin=uniforms.channel%11u;let frame=uniforms.frame_begin+global_idx;"
      << y.SetByOffset("global_idx", "0.0")
      << "if(!valid_output()||frame>=" << e.GetByOffset("6u") << "){return;}"
      << "let re=" << x.GetByOffset("22u*(frame-uniforms.input_begin)+2u*bin") << ";"
      << "let im=" << x.GetByOffset("22u*(frame-uniforms.input_begin)+2u*bin+1u") << ";"
      << "let re2=square(re);let im2=square(im);let total:f32=re2+im2;"
      << "let magnitude=sqrt(total);let ratio=original_div(im,re);let angle=original_atan(ratio);"
      << "let pi=bitcast<f32>(0x40490fdbu);let positive:f32=angle+pi;let negative:f32=angle-pi;"
      << "let rb=bitcast<u32>(re);let ib=bitcast<u32>(im);"
      << "let imag_positive=(ib&0x80000000u)==0u&&(ib&0x7fffffffu)>0u&&(ib&0x7fffffffu)<=0x7f800000u;"
      << "let real_negative=(rb&0x80000000u)!=0u&&(rb&0x7fffffffu)>0u&&(rb&0x7fffffffu)<=0x7f800000u;"
      << "let quadrant=select(negative,positive,imag_positive);let phase=select(angle,quadrant,real_negative);"
      << y.SetByOffset("global_idx", "select(magnitude,phase,uniforms.channel>=11u)");
    return Status::OK();
  }
};

}  // namespace onnxruntime::webgpu::m0_polar_detail
