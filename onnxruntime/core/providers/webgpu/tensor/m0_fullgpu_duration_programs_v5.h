// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_duration_exact_fp32.h"
namespace onnxruntime::webgpu::m0_duration_detail {
// One independent original FP32 row per token; publication/prefix is separate.
class DurationRowsProgram final : public Program<DurationRowsProgram> {
 public:
  DurationRowsProgram():Program{"RW07DurationParallelExactRowsV5"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"tokens",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override {
    const auto& logits=s.AddInput("logits",ShaderUsage::UseValueTypeAlias);
    const auto& speed=s.AddInput("speed",ShaderUsage::UseValueTypeAlias);
    const auto& rows=s.AddOutput("rows",ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << rw07::DurationExactFp32 <<
        "fn rw07_finite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;}\n"
        // Exact pinned ARM64 MLAS coefficients. Clamp is inside its original
        // Sigmoid arithmetic implementation, not an added duration Clip max.
        "fn rw07_sigmoid(raw:f32)->f32{let rb=bitcast<u32>(raw);let mag=rb&0x7fffffffu;let x=bitcast<f32>(select(rb,(rb&0x80000000u)|0x41900000u,mag>0x41900000u));let v=rw07_mul(x,x);"
        "var p=rw07_fma_positive(v,bitcast<f32>(775959889u),bitcast<f32>(871911115u));"
        "p=rw07_fma_positive(p,v,bitcast<f32>(947863867u));p=rw07_fma_positive(p,v,bitcast<f32>(1007385944u));"
        "p=rw07_fma_positive(p,v,bitcast<f32>(1048461106u));p=rw07_mul(p,x);"
        "var q=rw07_fma_positive(v,bitcast<f32>(724288757u),bitcast<f32>(835056251u));"
        "q=rw07_fma_positive(q,v,bitcast<f32>(919803869u));q=rw07_fma_positive(q,v,bitcast<f32>(987698495u));"
        "q=rw07_fma_positive(q,v,bitcast<f32>(1039089176u));q=rw07_fma_positive(q,v,bitcast<f32>(1065238324u));"
        "return clamp(rw07_add(rw07_div(p,q),0.5),0.0,1.0);}\n"
        // Explicit original Round ties-to-even. Conversion of the integral base
        // follows the GPU finite/range check (sum/speed in [0,100]).
        "fn rw07_round_even(v:f32)->f32{let base=floor(v);let fraction=v-base;"
        "if(fraction<0.5){return base;}if(fraction>0.5){return base+1.0;}"
        "return base+f32(u32(base)&1u);}\n";
    auto& f=s.MainFunctionBody();
    f<<"let t=global_idx;if(t>=uniforms.tokens){return;}"
     <<rows.SetByOffset("2u*t","0u")<<rows.SetByOffset("2u*t+1u","0u")
      << "let scale=" << speed.GetByOffset("0u") << ";"
      << "if(!rw07_finite(scale)||scale<0.5||scale>2.0){"
      << rows.SetByOffset("2u*t+1u", "101u") << "return;}\n"
      << "var sigmoid_values:array<f32,50>;"
      << "for(var channel=0u;channel<50u;channel++){let x=" << logits.GetByOffset("t*50u+channel") << ";"
      << "if((bitcast<u32>(x)&0x7fffffffu)>0x7f800000u){" << rows.SetByOffset("2u*t+1u", "100u") << "return;}let sigmoid=rw07_sigmoid(x);"
      << "if(!rw07_finite(sigmoid)||sigmoid<0.0||sigmoid>1.0){"
      << rows.SetByOffset("2u*t+1u", "100u") << "return;}sigmoid_values[channel]=sigmoid;}\n"
      // Eigen ARM64 float Map sum: two 4-lane accumulators, horizontal
      // pairwise reduction, then unaligned prefix and tail. Original logical
      // row stride is 50 floats; canonical CPU allocation is 16-byte aligned.
      << "let start=2u*(t%2u);let end=start+48u;"
      << "var r0=vec4<f32>(sigmoid_values[start],sigmoid_values[start+1u],sigmoid_values[start+2u],sigmoid_values[start+3u]);"
      << "var r1=vec4<f32>(sigmoid_values[start+4u],sigmoid_values[start+5u],sigmoid_values[start+6u],sigmoid_values[start+7u]);"
      << "for(var i=start+8u;i<end;i+=8u){r0=rw07_add4(r0,vec4<f32>(sigmoid_values[i],sigmoid_values[i+1u],sigmoid_values[i+2u],sigmoid_values[i+3u]));"
      << "r1=rw07_add4(r1,vec4<f32>(sigmoid_values[i+4u],sigmoid_values[i+5u],sigmoid_values[i+6u],sigmoid_values[i+7u]));}"
      << "let lanes=rw07_add4(r0,r1);var sum=rw07_add(rw07_add(lanes.x,lanes.y),rw07_add(lanes.z,lanes.w));"
      << "for(var i=0u;i<start;i++){sum=rw07_add(sum,sigmoid_values[i]);}"
      << "for(var i=end;i<50u;i++){sum=rw07_add(sum,sigmoid_values[i]);}\n"
      << "if(!rw07_finite(sum)||sum<0.0||sum>50.0){"
      << rows.SetByOffset("2u*t+1u", "100u") << "return;}\n"
      << "let divided=rw07_div(sum,scale);if(!rw07_finite(divided)||divided<0.0||divided>100.0){"
      << rows.SetByOffset("2u*t+1u", "100u") << "return;}"
      // Original Clip has min1 and absent max. Check, never clamp, its bound.
      << "let clipped=max(rw07_round_even(divided),1.0);"
      << "if(!rw07_finite(clipped)||clipped<1.0||clipped>100.0||clipped!=floor(clipped)){"
      << rows.SetByOffset("2u*t+1u", "100u") << "return;}"
      <<rows.SetByOffset("2u*t","u32(clipped)");
    return Status::OK();
  }
};
class DurationProgram final : public Program<DurationProgram> {
 public:
  DurationProgram():Program{"RW07DurationIntegerPrefixPublicationV5"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"tokens", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"epoch_low", ProgramUniformVariableDataType::Uint32},
      {"epoch_high", ProgramUniformVariableDataType::Uint32},
      {"run_low", ProgramUniformVariableDataType::Uint32},
      {"run_high", ProgramUniformVariableDataType::Uint32},
      {"sidecar_low", ProgramUniformVariableDataType::Uint32},
      {"sidecar_high", ProgramUniformVariableDataType::Uint32},
      {"carrier_low", ProgramUniformVariableDataType::Uint32},
      {"carrier_high", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override {
    const auto& rows=s.AddInput("rows",ShaderUsage::UseValueTypeAlias);
    const auto& durations=s.AddOutput("durations",ShaderUsage::UseValueTypeAlias);
    const auto& prefixes=s.AddOutput("prefixes",ShaderUsage::UseValueTypeAlias);
    const auto& total=s.AddOutput("total",ShaderUsage::UseValueTypeAlias);
    const auto& extent=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << extent.SetByOffset("i", "0u") << "}\n"
      << extent.SetByOffset("0u", "2u") << extent.SetByOffset("1u", "1u")
      << extent.SetByOffset("22u", "uniforms.epoch_low")
      << extent.SetByOffset("23u", "uniforms.epoch_high")
      << extent.SetByOffset("24u", "uniforms.run_low") << extent.SetByOffset("25u", "uniforms.run_high")
      << extent.SetByOffset("26u", "uniforms.sidecar_low") << extent.SetByOffset("27u", "uniforms.sidecar_high")
      << extent.SetByOffset("28u", "uniforms.carrier_low") << extent.SetByOffset("29u", "uniforms.carrier_high")
      << extent.SetByOffset("30u", "1u") << extent.SetByOffset("31u", "1001u")
      << total.SetByOffset("0u", "0u") << total.SetByOffset("1u", "0u")
      << "for(var t=0u;t<uniforms.tokens;t++){"
      << durations.SetByOffset("2u*t", "0u") << durations.SetByOffset("2u*t+1u", "0u")
      << prefixes.SetByOffset("2u*t", "0u") << prefixes.SetByOffset("2u*t+1u", "0u") << "}\n"
      // This is the original Gather(-1) failure, not successful empty audio.
      << "if(uniforms.tokens==0u){" << extent.SetByOffset("20u", "1u") << "return;}\n"
      <<"var prefix=0u;for(var t=0u;t<uniforms.tokens;t++){let status="<<rows.GetByOffset("2u*t+1u")<<";"
      <<"if(status!=0u){"<<extent.SetByOffset("20u","status")<<"return;}"
      <<"let duration="<<rows.GetByOffset("2u*t")<<";if(duration==0u||duration>100u){"<<extent.SetByOffset("20u","100u")<<"return;}if(prefix>uniforms.capacity||duration>uniforms.capacity-prefix){"
      <<extent.SetByOffset("20u","102u")<<"return;}prefix+=duration;"
      <<durations.SetByOffset("2u*t","duration")<<prefixes.SetByOffset("2u*t","prefix")<<"}"
      <<total.SetByOffset("0u","prefix")<<extent.SetByOffset("2u","prefix")
      <<extent.SetByOffset("18u","prefix")<<extent.SetByOffset("21u","63u");
    return Status::OK();
  }
};

} // namespace onnxruntime::webgpu::m0_duration_detail
