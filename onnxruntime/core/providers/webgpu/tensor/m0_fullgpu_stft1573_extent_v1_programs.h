// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::webgpu::m0_stft_detail {
inline const char* OriginalWindow() {
  return "const original_window=array<u32,20>(0u,1019771126u,1036226829u,1045630416u,"
         "1051780163u,1056964608u,1059556830u,1061895308u,1063751134u,1064942648u,"
         "1065353216u,1064942648u,1063751134u,1061895308u,1059556830u,1056964608u,"
         "1051780163u,1045630416u,1036226829u,1019771126u);\n";
}
inline void Roots(ShaderHelper& s) {
    s.AdditionalImplementation()
        << "const cosine=array<f32,20>(1.0,0.9510565162951535,0.8090169943749475,"
           "0.5877852522924731,0.30901699437494745,0.0,-0.30901699437494745,"
           "-0.5877852522924731,-0.8090169943749475,-0.9510565162951535,-1.0,"
           "-0.9510565162951535,-0.8090169943749475,-0.5877852522924731,"
           "-0.30901699437494745,0.0,0.30901699437494745,0.5877852522924731,"
           "0.8090169943749475,0.9510565162951535);\n"
        << "const sine=array<f32,20>(0.0,0.30901699437494745,0.5877852522924731,"
           "0.8090169943749475,0.9510565162951535,1.0,0.9510565162951535,"
           "0.8090169943749475,0.5877852522924731,0.30901699437494745,0.0,"
           "-0.30901699437494745,-0.5877852522924731,-0.8090169943749475,"
           "-0.9510565162951535,-1.0,-0.9510565162951535,-0.8090169943749475,"
           "-0.5877852522924731,-0.30901699437494745);\n";
}
class OwnedStftGuard final : public Program<OwnedStftGuard> {
 public:
  OwnedStftGuard() : Program{"RW07STFT1573OwnedGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"signal_capacity", ProgramUniformVariableDataType::Uint32},
      {"frame_capacity", ProgramUniformVariableDataType::Uint32},
      {"pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("signal_extent", ShaderUsage::UseValueTypeAlias);
    const auto& hop = s.AddInput("hop", ShaderUsage::UseValueTypeAlias);
    const auto& window = s.AddInput("window", ShaderUsage::UseValueTypeAlias);
    const auto& length = s.AddInput("frame_length", ShaderUsage::UseValueTypeAlias);
    const auto& out = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << OriginalWindow();
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << out.SetByOffset("i", "0u") << "}"
      << out.SetByOffset("0u", "2u") << out.SetByOffset("1u", "4u")
      << "for(var i=0u;i<10u;i++){" << out.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "if(uniforms.pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=2u){"
      << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity", "i", 10) << "){" << out.SetByOffset("20u", "103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){" << out.SetByOffset("20u", "103u") << "return;}"
      << out.SetByOffset("20u", "status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("5u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=6u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "let n=" << e.GetByOffset("4u") << ";if(n<620u||n>61440020u||(n-20u)%600u!=0u||"
      << e.GetByOffset("18u") << "!=n){" << out.SetByOffset("20u", "103u") << "return;}"
      << "if(" << hop.GetByOffset("0u") << "!=5u||" << hop.GetByOffset("1u") << "!=0u||"
      << length.GetByOffset("0u") << "!=20u||" << length.GetByOffset("1u") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<20u;i++){if(bitcast<u32>(" << window.GetByOffset("i") << ")!=original_window[i]){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "let frames=(n-20u)/5u+1u;if(n>uniforms.signal_capacity||frames>uniforms.frame_capacity){"
      << out.SetByOffset("20u", "102u") << "return;}"
      << out.SetByOffset("2u", "1u") << out.SetByOffset("4u", "frames") << out.SetByOffset("6u", "11u")
      << out.SetByOffset("8u", "2u") << out.SetByOffset("18u", "22u*frames") << out.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
inline void ValidSpectrum(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_spectrum() -> bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=4u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("5u") << "!=0u||" << e.GetByOffset("6u") << "!=11u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("8u") << "!=2u||"
    << e.GetByOffset("9u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=10u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let frames=" << e.GetByOffset("4u") << ";return frames>=121u&&(frames-1u)%120u==0u&&frames<=uniforms.frame_capacity&&"
    << e.GetByOffset("18u") << "==22u*frames;}\n";
}
class OwnedStftData final : public Program<OwnedStftData> {
 public:
  OwnedStftData() : Program{"RW07STFT1573OwnedTimeWindowDataV2"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"signal_capacity", ProgramUniformVariableDataType::Uint32},
      {"frame_capacity", ProgramUniformVariableDataType::Uint32},
      {"first_frame", ProgramUniformVariableDataType::Uint32},
      {"frame_count", ProgramUniformVariableDataType::Uint32},
      {"signal_begin", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("signal", ShaderUsage::UseValueTypeAlias);
    const auto& window = s.AddInput("window", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("spectrum", ShaderUsage::UseValueTypeAlias);
    Roots(s);
    ValidSpectrum(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=11u*uniforms.frame_count){return;}let at=2u*global_idx;"
      << y.SetByOffset("at", "0.0") << y.SetByOffset("at+1u", "0.0")
      << "if(!valid_spectrum()){return;}let frame=uniforms.first_frame+global_idx/11u;let bin=global_idx%11u;"
      << "if(frame>=" << e.GetByOffset("4u") << "){return;}"
      << "if(5u*frame+19u>=uniforms.signal_capacity){return;}var re:f32=0.0;var im:f32=0.0;"
      << "for(var n=0u;n<20u;n++){let sample:f32=" << x.GetByOffset("5u*frame+n-uniforms.signal_begin") << "*"
      << window.GetByOffset("n") << ";let root=(bin*n)%20u;let rp:f32=sample*cosine[root];"
      << "let ip:f32=sample*(-sine[root]);re=re+rp;im=im+ip;}"
      << y.SetByOffset("at", "re") << y.SetByOffset("at+1u", "im");
    return Status::OK();
  }
};
}  // namespace onnxruntime::webgpu::m0_stft_detail
