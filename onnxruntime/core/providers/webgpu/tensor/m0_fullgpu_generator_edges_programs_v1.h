// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"

namespace onnxruntime::webgpu::m0_generator_edges_detail {
using m0_residual_stages::ValidOutput;
using m0_residual_stages::OriginalArithmetic;
class EdgeData final : public Program<EdgeData> {
 public:
  EdgeData(uint32_t mode,uint32_t pairs):Program{"RW07GeneratorEdgeSpectralWindowDataV3"},mode_(mode),pairs_(pairs) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"minimum",ProgramUniformVariableDataType::Uint32},
    {"maximum",ProgramUniformVariableDataType::Uint32},
    {"modulus",ProgramUniformVariableDataType::Uint32},
    {"remainder",ProgramUniformVariableDataType::Uint32},
    {"stage",ProgramUniformVariableDataType::Uint32},
    {"tile_channels",ProgramUniformVariableDataType::Uint32},
    {"component",ProgramUniformVariableDataType::Uint32},
    {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    std::array<const ShaderVariableHelper*,3> inputs{};
    for(uint32_t j=0;j<pairs_;j++){inputs[j]=&s.AddInput("carrier"+std::to_string(j),ShaderUsage::UseValueTypeAlias);}
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* w=nullptr;const ShaderVariableHelper* b=nullptr;
    if(mode_==0){w=&s.AddInput("weight",ShaderUsage::UseValueTypeAlias);b=&s.AddInput("bias",ShaderUsage::UseValueTypeAlias);}
    const auto& y=s.AddOutput("output",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e);
    OriginalArithmetic(s);
    if(mode_==4){s.AdditionalImplementation() << m0_trig_detail::Trig;}
    auto& f=s.MainFunctionBody();
    f<<"if(global_idx>=uniforms.tile_channels*uniforms.output_capacity){return;}"<<y.SetByOffset("global_idx","0.0")
     <<"if(!valid_output()){return;}let t=global_idx%uniforms.output_capacity;let oc=global_idx/uniforms.output_capacity;"
     <<"let count="<<e.GetByOffset("6u")<<";if(t>=count){return;}";
    if(mode_==1||mode_==2){
      for(uint32_t j=0;j<pairs_;j++){
        f<<"let v"<<j<<"="<<inputs[j]->GetByOffset("oc*"+GetElementAt("uniforms.input_capacity",std::to_string(j)+"u",3)+"+t")<<";";
      }
      if(mode_==1){f<<y.SetByOffset("global_idx","original_add(v0,v1)");}
      else{f<<"let first:f32=original_add(v0,v1);let total:f32=original_add(first,v2);"<<y.SetByOffset("global_idx","original_div(total,3.0)");}
    }
    if(mode_==0){
      f<<"let stride=select(10u,6u,uniforms.stage==1u);let kernel=select(20u,12u,uniforms.stage==1u);"
       <<"let pad=select(5u,3u,uniforms.stage==1u);let input_length=select(count/10u,(count-1u)/6u,uniforms.stage==1u);"
       <<"var output_frame=t;";
      f<<"if(uniforms.stage==1u){output_frame=select(t-1u,1u,t==0u);}"
       <<"let input_channels=select(512u,256u,uniforms.stage==1u);var total:f32=0.0;"
       <<"for(var k=0u;k<kernel;k++){let numerator=i32(output_frame+pad)-i32(k);"
       <<"if(numerator>=0&&u32(numerator)%stride==0u){let pos=u32(numerator)/stride;if(pos<input_length){var column:f32=0.0;"
       <<"for(var ic=0u;ic<input_channels;ic++){let value="<<inputs[0]->GetByOffset("ic*"+GetElementAt("uniforms.input_capacity","0u",3)+"+pos")<<";"
       <<"let negative=original_mul(value,bitcast<f32>(0x3dcccccdu));let activated=select(negative,value,value>=0.0);"
       <<"column=original_add(column,original_mul(activated,"<<w->GetByOffset("(ic*uniforms.channels+oc)*kernel+k")<<"));}"
       <<"total=original_add(total,column);}}}"
       <<y.SetByOffset("global_idx","original_add(total,"+b->GetByOffset("oc")+")");
    }
    if(mode_==4){
      f<<"let bin=oc;let magnitude_log="<<inputs[0]->GetByOffset("bin*"+GetElementAt("uniforms.input_capacity","0u",3)+"+t")<<";"
       <<"let phase_value="<<inputs[1]->GetByOffset("bin*"+GetElementAt("uniforms.input_capacity","1u",3)+"+t")<<";"
       <<"let mb=bitcast<u32>(magnitude_log);let ma=mb&0x7fffffffu;var amplitude:f32=original_f32_from_bits(0x7fc00000u);"
       <<"if(ma<0x7f800000u){if(magnitude_log>=bitcast<f32>(0x42b17218u)){amplitude=original_f32_from_bits(0x7f800000u);}else{amplitude=exp(magnitude_log);}}else if(ma==0x7f800000u){amplitude=select(original_f32_from_bits(0x7f800000u),0.0,(mb&0x80000000u)!=0u);}"
       <<"let pa=bitcast<u32>(phase_value)&0x7fffffffu;var phase:f32=original_f32_from_bits(0x7fc00000u);"
       <<"if(pa<0x7f800000u){phase=rw07_sin(phase_value);}var wave:f32=original_f32_from_bits(0x7fc00000u);"
       <<"if((bitcast<u32>(phase)&0x7fffffffu)<0x7f800000u){if(uniforms.component==0u){wave=rw07_cos(phase);}else{wave=rw07_sin(phase);}}"
       <<y.SetByOffset("global_idx","original_mul(amplitude,wave)");
    }
    return Status::OK();
  }
 private:
  const uint32_t mode_,pairs_;
};
inline void UpsampleTapGeometry(ShaderHelper& s,const ShaderVariableHelper& extent){
  ValidOutput(s,extent);OriginalArithmetic(s);
  s.AdditionalImplementation()<<R"WGSL(
fn upsample_source(frame:u32,count:u32)->i32 {
 let stride=select(10u,6u,uniforms.stage==1u);
 let pad=select(5u,3u,uniforms.stage==1u);
 let length=select(count/10u,(count-1u)/6u,uniforms.stage==1u);
 var output_frame=frame;if(uniforms.stage==1u){output_frame=select(frame-1u,1u,frame==0u);}
 let numerator=i32(output_frame+pad)-i32(uniforms.tap);
 if(numerator<0||u32(numerator)%stride!=0u){return -1;}
 let pos=u32(numerator)/stride;
 if(pos>=length||pos>=uniforms.input_capacity){return -1;}
 return i32(pos);
}
)WGSL";
}
class UpsampleTapColumn final : public Program<UpsampleTapColumn> {
 public:
  UpsampleTapColumn():Program{"RW07GeneratorUpsampleTapColumnWindowsV1"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"minimum",ProgramUniformVariableDataType::Uint32},
    {"maximum",ProgramUniformVariableDataType::Uint32},
    {"modulus",ProgramUniformVariableDataType::Uint32},
    {"remainder",ProgramUniformVariableDataType::Uint32},
    {"stage",ProgramUniformVariableDataType::Uint32},
    {"tap",ProgramUniformVariableDataType::Uint32},
    {"input_offset",ProgramUniformVariableDataType::Uint32},
    {"input_count",ProgramUniformVariableDataType::Uint32},
    {"output_offset",ProgramUniformVariableDataType::Uint32},
    {"output_count",ProgramUniformVariableDataType::Uint32},
    {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& x=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto& column=s.AddOutput("column",ShaderUsage::UseValueTypeAlias);
    UpsampleTapGeometry(s,e);auto& f=s.MainFunctionBody();
    f<<"if(global_idx>=uniforms.output_count*uniforms.output_capacity){return;}"
      <<"if(!valid_output()){ "<<column.SetByOffset("global_idx","0.0")<<"return;}"
      <<"let frame=global_idx%uniforms.output_capacity;let count="<<e.GetByOffset("6u")<<";"
      <<"let source_channels=select(512u,256u,uniforms.stage==1u);"
      <<"if(frame>=count||uniforms.input_offset>=source_channels||uniforms.input_count==0u||"
      <<"uniforms.input_count>source_channels-uniforms.input_offset||uniforms.output_offset>=uniforms.channels||"
      <<"uniforms.output_count>uniforms.channels-uniforms.output_offset){"
      <<column.SetByOffset("global_idx","0.0")<<"return;}"
      <<"let pos=upsample_source(frame,count);if(pos<0){"<<column.SetByOffset("global_idx","0.0")<<"return;}"
      <<"let oc=uniforms.output_offset+global_idx/uniforms.output_capacity;"
      <<"let kernel=select(20u,12u,uniforms.stage==1u);var total:f32=0.0;"
      <<"if(uniforms.input_offset!=0u){total="<<column.GetByOffset("global_idx")<<";}"
      <<"for(var local_ic=0u;local_ic<uniforms.input_count;local_ic++){let ic=uniforms.input_offset+local_ic;"
      <<"let value="<<x.GetByOffset("local_ic*uniforms.input_capacity+u32(pos)")<<";"
      <<"let negative=original_mul(value,bitcast<f32>(0x3dcccccdu));let activated=select(negative,value,value>=0.0);"
      <<"total=original_add(total,original_mul(activated,"
      <<w.GetByOffset("(ic*uniforms.channels+oc)*kernel+uniforms.tap")<<"));}"
      <<column.SetByOffset("global_idx","total");
    return Status::OK();
  }
};
class UpsampleTapAdd final : public Program<UpsampleTapAdd> {
 public:
  UpsampleTapAdd():Program{"RW07GeneratorUpsampleTapOverlapWindowsV1"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"minimum",ProgramUniformVariableDataType::Uint32},
    {"maximum",ProgramUniformVariableDataType::Uint32},
    {"modulus",ProgramUniformVariableDataType::Uint32},
    {"remainder",ProgramUniformVariableDataType::Uint32},
    {"stage",ProgramUniformVariableDataType::Uint32},
    {"tap",ProgramUniformVariableDataType::Uint32},
    {"output_offset",ProgramUniformVariableDataType::Uint32},
    {"output_count",ProgramUniformVariableDataType::Uint32},
    {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& column=s.AddInput("column",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("bias",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("accumulator",ShaderUsage::UseValueTypeAlias);
    UpsampleTapGeometry(s,e);auto& f=s.MainFunctionBody();
    f<<"if(global_idx>=uniforms.output_count*uniforms.output_capacity){return;}"
      <<"if(!valid_output()){ "<<y.SetByOffset("global_idx","0.0")<<"return;}"
      <<"let frame=global_idx%uniforms.output_capacity;let count="<<e.GetByOffset("6u")<<";"
      <<"if(frame>=count||uniforms.output_offset>=uniforms.channels||"
      <<"uniforms.output_count>uniforms.channels-uniforms.output_offset){"
      <<y.SetByOffset("global_idx","0.0")<<"return;}"
      <<"var total:f32=0.0;if(uniforms.tap!=0u){total="<<y.GetByOffset("global_idx")<<";}"
      <<"if(upsample_source(frame,count)>=0){total=original_add(total,"<<column.GetByOffset("global_idx")<<");}"
      <<"if(uniforms.tap+1u==select(20u,12u,uniforms.stage==1u)){"
      <<"let oc=uniforms.output_offset+global_idx/uniforms.output_capacity;"
      <<"total=original_add(total,"<<b.GetByOffset("oc")<<");}"<<y.SetByOffset("global_idx","total");
    return Status::OK();
  }
};
}  // namespace onnxruntime::webgpu::m0_generator_edges_detail
