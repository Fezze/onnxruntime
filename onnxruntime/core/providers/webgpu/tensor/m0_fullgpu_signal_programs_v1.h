// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/common/m0_observation_options_v1.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_duration_exact_fp32.h"
#include "m0_fullgpu_trig_v1.h"

// Shared by the actual owned signal kernel and numerical program tests.
namespace onnxruntime::webgpu::m0_signal_detail {
inline void ValidOutput(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_output()->bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=2u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("5u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=6u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let n=" << e.GetByOffset("4u") << ";if(n<620u||n>uniforms.output_capacity||"
    << "(n-20u)%600u!=0u||" << e.GetByOffset("18u") << "!=n){return false;}"
    << "return (n-20u)/300u<=uniforms.input_capacity;}\n";
}

class Prefix final : public Program<Prefix> {
 public:
  Prefix() : Program{"RW07SignalOriginal1514Prefix1526ExactV2"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("f0", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& phase = s.AddOutput("phase", ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << ::rw07::DurationExactFp32;
    ValidOutput(s, e);
    auto& a = s.AdditionalImplementation();
    a << "fn rw07_signal_f32(bits:u32)->f32 {return bitcast<f32>(bits);}\n"
      << "fn radians(h:u32,t:u32,len:u32)->f32 {"
      << "let q=min(u32(floor(rw07_div(f32(t),300.0))),len-1u);"
      // Pinned original model sampling rate, in samples per second.
      << "let original_sample_rate_hz=24000.0;let source=" << x.GetByOffset("q") << ";"
      // WGSL division is allowed2.5ULP. A one-bit pitch-normalization error can
      // accumulate into phase jumps before the high-frequency sine. Preserve
      // the original FLOAT Mul/Div/Sub rounding, using existing finite helpers.
      << "if((bitcast<u32>(source)&0x7fffffffu)>=0x7f800000u){return rw07_signal_f32(0x7fc00000u);}"
      << "let harmonic=rw07_mul(source,f32(h+1u));"
      << "if((bitcast<u32>(harmonic)&0x7fffffffu)>=0x7f800000u){return rw07_signal_f32(0x7fc00000u);}"
      << "let normalized=rw07_div(harmonic,original_sample_rate_hz);return rw07_add(normalized,-floor(normalized));}\n";
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=9u){return;}let h=global_idx;"
      << "for(var t=0u;t<uniforms.input_capacity;t++){" << phase.SetByOffset("h*uniforms.input_capacity+t", "0.0") << "}"
      << "if(!valid_output()){return;}let len=(" << e.GetByOffset("4u") << "-20u)/300u;"
      << "let expanded=300u*len;var prefix:f32=0.0;for(var t=0u;t<len;t++){"
      << "let coord=rw07_add(rw07_div(rw07_add(f32(t),0.5),bitcast<f32>(0x3b5a740eu)),-0.5);"
      << "let pos=max(0.0,min(coord,f32(expanded-1u)));let lo=min(u32(pos),expanded-1u);"
      << "let hi=min(lo+1u,expanded-1u);var dx1=abs(pos-f32(lo));var dx2=abs(pos-f32(hi));"
      << "if(lo==hi){dx1=0.5;dx2=0.5;}let nh=min(h+1u,8u);"
      << "let dy1=select(0.0,0.5,h==nh);let dy2=select(1.0,0.5,h==nh);"
      << "let a=radians(h,lo,len);let b=radians(h,hi,len);let c=radians(nh,lo,len);let d=radians(nh,hi,len);"
      << "var value:f32=0.0;"
      << "if((bitcast<u32>(a)&0x7fffffffu)<0x7f800000u&&(bitcast<u32>(b)&0x7fffffffu)<0x7f800000u&&"
      << "(bitcast<u32>(c)&0x7fffffffu)<0x7f800000u&&(bitcast<u32>(d)&0x7fffffffu)<0x7f800000u){"
      << "let p0=rw07_mul(rw07_mul(rw07_mul(dx2,dy2),0.5),a);"
      << "let p1=rw07_mul(rw07_mul(rw07_mul(dx1,dy2),0.5),b);"
      << "let p2=rw07_mul(rw07_mul(rw07_mul(dx2,dy1),0.5),c);"
      << "let p3=rw07_mul(rw07_mul(rw07_mul(dx1,dy1),0.5),d);"
      << "value=rw07_add(rw07_add(rw07_add(p0,p1),p2),p3);"
      << "value=rw07_add(rw07_add(rw07_add(rw07_add(value,p0),p1),p2),p3);"
      << "}else{value=dx2*dy2*0.5*a;value=value+dx1*dy2*0.5*b;"
      << "value=value+dx2*dy1*0.5*c;value=value+dx1*dy1*0.5*d;"
      << "value=value+dx2*dy2*0.5*a;value=value+dx1*dy2*0.5*b;"
      << "value=value+dx2*dy1*0.5*c;value=value+dx1*dy1*0.5*d;}"
      << "if(t==0u){prefix=value;}else if((bitcast<u32>(prefix)&0x7fffffffu)<0x7f800000u&&"
      << "(bitcast<u32>(value)&0x7fffffffu)<0x7f800000u){prefix=rw07_add(prefix,value);}else{prefix=prefix+value;}"
      << "let doubled:f32=prefix*2.0;let angle:f32=doubled*bitcast<f32>(0x40490fdbu);"
      << "var scaled:f32=angle*300.0;"
      // Preserve the three original Mul node rounding boundaries instead of
      // allowing the shader compiler to reassociate their constant factors.
      << "if((bitcast<u32>(prefix)&0x7fffffffu)<0x7f800000u){"
      << "scaled=rw07_mul(rw07_mul(rw07_mul(prefix,2.0),bitcast<f32>(0x40490fdbu)),300.0);}"
      << phase.SetByOffset("h*uniforms.input_capacity+t", "scaled") << "}";
    return Status::OK();
  }
};

class Signal final : public Program<Signal> {
 public:
  explicit Signal(bool reuse_coefficients=M0BatchHeavyProgramsV1())
      : Program{reuse_coefficients?"RW07SignalOriginal1528Through1571CoefficientsV2":"RW07SignalOriginal1528Through1571V1"},
        reuse_coefficients_(reuse_coefficients) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("f0", ShaderUsage::UseValueTypeAlias);
    const auto& phase = s.AddInput("phase", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const auto& bias = s.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& out = s.AddOutput("signal", ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << ::rw07::DurationExactFp32;
    s.AdditionalImplementation() << m0_trig_detail::Trig;
    ValidOutput(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.output_capacity){return;}" << out.SetByOffset("global_idx", "0.0")
      << "if(!valid_output()||global_idx>=" << e.GetByOffset("4u") << "){return;}"
      << "let len=(" << e.GetByOffset("4u") << "-20u)/300u;let expanded=300u*len;"
      << "var sample=0u;if(global_idx<10u){sample=10u-global_idx;}"
      << "else{sample=global_idx-10u;if(sample>=expanded){sample=2u*expanded-2u-sample;}}"
      << "let coord=rw07_add(rw07_div(rw07_add(f32(sample),0.5),300.0),-0.5);"
      << "let pos=max(0.0,min(coord,f32(len-1u)));let lo=min(u32(pos),len-1u);let hi=min(lo+1u,len-1u);"
      << "var dx1=abs(pos-f32(lo));var dx2=abs(pos-f32(hi));if(lo==hi){dx1=0.5;dx2=0.5;}"
      ;
    if(reuse_coefficients_){
      // The same two time weights recur for all nine harmonics. Preserve both
      // original multiplication boundaries, including zero-corner signs.
      f<<"let regular_coefficients=vec4<f32>("
        <<"rw07_mul(rw07_mul(dx2,1.0),0.5),rw07_mul(rw07_mul(dx1,1.0),0.5),"
        <<"rw07_mul(rw07_mul(dx2,0.0),0.5),rw07_mul(rw07_mul(dx1,0.0),0.5));"
        <<"let final_coefficients=vec4<f32>("
        <<"rw07_mul(rw07_mul(dx2,0.5),0.5),rw07_mul(rw07_mul(dx1,0.5),0.5),"
        <<"rw07_mul(rw07_mul(dx2,0.5),0.5),rw07_mul(rw07_mul(dx1,0.5),0.5));";
    }
    f << "let fi=min(u32(floor(rw07_div(f32(sample),300.0))),len-1u);"
      << "let uv=select(0.0,1.0," << x.GetByOffset("fi") << ">10.0);var sum:f32=0.0;"
      << "for(var h=0u;h<9u;h++){let nh=min(h+1u,8u);let dy1=select(0.0,0.5,h==nh);"
      << "let dy2=select(1.0,0.5,h==nh);let a=" << phase.GetByOffset("h*uniforms.input_capacity+lo") << ";"
      << "let b=" << phase.GetByOffset("h*uniforms.input_capacity+hi") << ";"
      << "let c=" << phase.GetByOffset("nh*uniforms.input_capacity+lo") << ";"
      << "let d=" << phase.GetByOffset("nh*uniforms.input_capacity+hi") << ";"
      << "var value:f32=0.0;"
      << "if((bitcast<u32>(a)&0x7fffffffu)<0x7f800000u&&(bitcast<u32>(b)&0x7fffffffu)<0x7f800000u&&"
      << "(bitcast<u32>(c)&0x7fffffffu)<0x7f800000u&&(bitcast<u32>(d)&0x7fffffffu)<0x7f800000u){"
      // Original rank-three Resize visits all eight corners in this order.
      // Reassociation of its large phase values changes the following sine.
      ;
    if(reuse_coefficients_){
      f<<"let coefficients=select(regular_coefficients,final_coefficients,h==8u);"
        <<"let p0=rw07_mul(coefficients.x,a);let p1=rw07_mul(coefficients.y,b);"
        <<"let p2=rw07_mul(coefficients.z,c);let p3=rw07_mul(coefficients.w,d);";
    }else{
      f<<"let p0=rw07_mul(rw07_mul(rw07_mul(dx2,dy2),0.5),a);"
        <<"let p1=rw07_mul(rw07_mul(rw07_mul(dx1,dy2),0.5),b);"
        <<"let p2=rw07_mul(rw07_mul(rw07_mul(dx2,dy1),0.5),c);"
        <<"let p3=rw07_mul(rw07_mul(rw07_mul(dx1,dy1),0.5),d);";
    }
    f
      << "value=rw07_add(rw07_add(rw07_add(p0,p1),p2),p3);"
      << "value=rw07_add(rw07_add(rw07_add(rw07_add(value,p0),p1),p2),p3);"
      << "}else{value=dx2*dy2*0.5*a;value=value+dx1*dy2*0.5*b;"
      << "value=value+dx2*dy1*0.5*c;value=value+dx1*dy1*0.5*d;"
      << "value=value+dx2*dy2*0.5*a;value=value+dx1*dy2*0.5*b;"
      << "value=value+dx2*dy1*0.5*c;value=value+dx1*dy1*0.5*d;}"
      << "let amplitude:f32=rw07_sin(value)*bitcast<f32>(0x3dcccccdu);let voiced:f32=amplitude*uv;"
      << "let product:f32=voiced*" << w.GetByOffset("h") << ";sum=sum+product;}"
      << "let shifted:f32=" << bias.GetByOffset("0u") << "+sum;"
      << out.SetByOffset("global_idx", "tanh(shifted)");
    return Status::OK();
  }
 private:
  const bool reuse_coefficients_;
};

}  // namespace onnxruntime::webgpu::m0_signal_detail
