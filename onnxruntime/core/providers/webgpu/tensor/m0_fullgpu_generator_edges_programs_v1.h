// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"

namespace onnxruntime::webgpu::m0_generator_edges_detail {
using m0_residual_stages::ValidOutput;
using m0_residual_stages::OriginalArithmetic;
class EdgeData final : public Program<EdgeData> {
 public:
  EdgeData(uint32_t mode,uint32_t pairs,bool finite=false,bool phase_taps=true)
      :Program{mode==0&&phase_taps?(finite?"RW07GeneratorEdgePhaseFiniteV6":"RW07GeneratorEdgePhaseScalarV6"):
          finite&&mode==0?"RW07GeneratorEdgeFiniteV5":"RW07GeneratorEdgeElementChunksV4"},
       mode_(mode),pairs_(pairs),finite_(finite&&mode==0),phase_taps_(phase_taps) {}
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
    {"element_begin",ProgramUniformVariableDataType::Uint32},
    {"element_count",ProgramUniformVariableDataType::Uint32},
    {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    std::array<const ShaderVariableHelper*,3> inputs{};
    for(uint32_t j=0;j<pairs_;j++){inputs[j]=&s.AddInput("carrier"+std::to_string(j),ShaderUsage::UseValueTypeAlias);}
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* w=nullptr;const ShaderVariableHelper* b=nullptr;
    if(mode_==0){w=&s.AddInput("weight",ShaderUsage::UseValueTypeAlias);b=&s.AddInput("bias",ShaderUsage::UseValueTypeAlias);}
    const auto& y=s.AddOutput("output",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e,true);
    OriginalArithmetic(s);
    if(mode_==4){s.AdditionalImplementation() << m0_trig_detail::Trig;}
    auto& f=s.MainFunctionBody();
    if(finite_){
      s.AdditionalImplementation()<<"var<workgroup> edge_flags:array<u32,64>;var<workgroup> edge_safe:u32;";
      f<<"let all_elements=uniforms.tile_channels*uniforms.output_capacity;"
        <<"if(uniforms.element_count==0u||uniforms.element_begin>=all_elements||"
        <<"uniforms.element_count>all_elements-uniforms.element_begin||"
        <<"workgroup_idx>=uniforms.element_count/64u+u32(uniforms.element_count%64u!=0u)){return;}"
        <<"let first=uniforms.element_begin+workgroup_idx*64u;"
        <<"let last=first+min(63u,uniforms.element_begin+uniforms.element_count-1u-first);var bad=0u;"
        <<"if(valid_output()){let n="<<e.GetByOffset("6u")<<";"
        <<"let stride=select(10u,6u,uniforms.stage==1u);let ksize=select(20u,12u,uniforms.stage==1u);"
        <<"let pad=select(5u,3u,uniforms.stage==1u);let input_length=select(n/10u,(n-1u)/6u,uniforms.stage==1u);"
        <<"let icount=select(512u,256u,uniforms.stage==1u);let first_oc=first/uniforms.output_capacity;"
        <<"let last_oc=last/uniforms.output_capacity;var lo=first%uniforms.output_capacity;var hi=last%uniforms.output_capacity;"
        <<"if(first_oc!=last_oc){if(lo>=n&&last_oc==first_oc+1u){lo=0u;}else{lo=0u;hi=uniforms.output_capacity-1u;}}"
        <<"hi=min(hi,n-1u);if(lo<=hi){var fmin=lo;var fmax=hi;"
        <<"if(uniforms.stage==1u){if(lo==0u){fmin=1u;fmax=1u;if(hi>=1u){fmin=0u;fmax=max(1u,hi-1u);}}else{fmin=lo-1u;fmax=hi-1u;}}"
        <<"let start=u32(max(0,i32(fmin)+i32(pad)-i32(ksize)+1))/stride;"
        <<"let end=min((fmax+pad)/stride,input_length-1u);"
        <<"if(start<=end){let span=end-start+1u;for(var index=local_idx;index<icount*span;index+=64u){"
        <<"let value="<<inputs[0]->GetByOffset("(index/span)*"+GetElementAt("uniforms.input_capacity","0u",3)+"+start+index%span")<<";"
        <<"if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){bad=1u;}}}"
        <<"for(var index=local_idx;index<(last_oc-first_oc+1u)*icount*ksize;index+=64u){"
        <<"let row=first_oc+index/(icount*ksize);let ic=(index/ksize)%icount;let k=index%ksize;let value="
        <<w->GetByOffset("(ic*uniforms.channels+row)*ksize+k")<<";"
        <<"if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){bad=1u;}}}}"
        <<"edge_flags[local_idx]=bad;workgroupBarrier();if(local_idx==0u){var combined=0u;for(var i=0u;i<64u;i++){combined|=edge_flags[i];}edge_safe=1u-min(combined,1u);}"
        <<"let finite_safe=workgroupUniformLoad(&edge_safe)==1u;";
    }
    if(mode_==0){
    f<<"let elements=uniforms.tile_channels*uniforms.output_capacity;"
      <<"if(uniforms.element_begin>=elements||uniforms.element_count>elements-uniforms.element_begin||"
      <<"global_idx>=uniforms.element_count){return;}let at=uniforms.element_begin+global_idx;";
    }else{f<<"let at=global_idx;";}
    f<<"if(at>=uniforms.tile_channels*uniforms.output_capacity){return;}"<<y.SetByOffset("at","0.0")
     <<"if(!valid_output()){return;}let t=at%uniforms.output_capacity;let oc=at/uniforms.output_capacity;"
     <<"let count="<<e.GetByOffset("6u")<<";if(t>=count){return;}";
    if(mode_==1||mode_==2){
      for(uint32_t j=0;j<pairs_;j++){
        f<<"let v"<<j<<"="<<inputs[j]->GetByOffset("oc*"+GetElementAt("uniforms.input_capacity",std::to_string(j)+"u",3)+"+t")<<";";
      }
      if(mode_==1){f<<y.SetByOffset("at","original_add(v0,v1)");}
      else{f<<"let first:f32=original_add(v0,v1);let total:f32=original_add(first,v2);"<<y.SetByOffset("at","original_div(total,3.0)");}
    }
    if(mode_==0){
      f<<"let stride=select(10u,6u,uniforms.stage==1u);let kernel=select(20u,12u,uniforms.stage==1u);"
       <<"let pad=select(5u,3u,uniforms.stage==1u);let input_length=select(count/10u,(count-1u)/6u,uniforms.stage==1u);"
       <<"var output_frame=t;";
      f<<"if(uniforms.stage==1u){output_frame=select(t-1u,1u,t==0u);}"
       <<"let input_channels=select(512u,256u,uniforms.stage==1u);var total:f32=0.0;";
      // Select once for the complete dot loop. Keep tap/channel accumulation
      // order and the final exceptional-aware bias operation unchanged.
      const auto dot_loop=[&](bool fast){
        // Only these taps satisfy the original stride condition. Their ascending
        // order is unchanged; every lane traverses at most two taps.
        f<<"for(var k="<<(phase_taps_?"(output_frame+pad)%stride":"0u")
         <<";k<kernel;k+="<<(phase_taps_?"stride":"1u")<<"){let numerator=i32(output_frame+pad)-i32(k);"
         <<"if(numerator>=0"<<(phase_taps_?"":"&&u32(numerator)%stride==0u")
         <<"){let pos=u32(numerator)/stride;if(pos<input_length){var column:f32=0.0;"
         <<"for(var ic=0u;ic<input_channels;ic++){let value="
         <<inputs[0]->GetByOffset("ic*"+GetElementAt("uniforms.input_capacity","0u",3)+"+pos")<<";"
         <<"let weight="<<w->GetByOffset("(ic*uniforms.channels+oc)*kernel+k")<<";";
        if(fast){
          f<<"let negative=value*bitcast<f32>(0x3dcccccdu);let activated=select(negative,value,value>=0.0);"
           <<"let product=activated*weight;column=column+product;}total=total+column;";
        }else{
          f<<"let negative=original_mul(value,bitcast<f32>(0x3dcccccdu));let activated=select(negative,value,value>=0.0);"
           <<"column=original_add(column,original_mul(activated,weight));}total=original_add(total,column);";
        }
        f<<"}}}";
      };
      if(finite_){f<<"if(finite_safe){";dot_loop(true);f<<"}else{";dot_loop(false);f<<"}";}
      else{dot_loop(false);}
      f<<y.SetByOffset("at","original_add(total,"+b->GetByOffset("oc")+")");
    }
    if(mode_==4){
      f<<"let bin=oc;let magnitude_log="<<inputs[0]->GetByOffset("bin*"+GetElementAt("uniforms.input_capacity","0u",3)+"+t")<<";"
       <<"let phase_value="<<inputs[1]->GetByOffset("bin*"+GetElementAt("uniforms.input_capacity","1u",3)+"+t")<<";"
       <<"let mb=bitcast<u32>(magnitude_log);let ma=mb&0x7fffffffu;var amplitude:f32=original_f32_from_bits(0x7fc00000u);"
       <<"if(ma<0x7f800000u){if(magnitude_log>=bitcast<f32>(0x42b17218u)){amplitude=original_f32_from_bits(0x7f800000u);}else{amplitude=exp(magnitude_log);}}else if(ma==0x7f800000u){amplitude=select(original_f32_from_bits(0x7f800000u),0.0,(mb&0x80000000u)!=0u);}"
       <<"let pa=bitcast<u32>(phase_value)&0x7fffffffu;var phase:f32=original_f32_from_bits(0x7fc00000u);"
       <<"if(pa<0x7f800000u){phase=rw07_sin(phase_value);}var wave:f32=original_f32_from_bits(0x7fc00000u);"
       <<"if((bitcast<u32>(phase)&0x7fffffffu)<0x7f800000u){if(uniforms.component==0u){wave=rw07_cos(phase);}else{wave=rw07_sin(phase);}}"
       <<y.SetByOffset("at","original_mul(amplitude,wave)");
    }
    return Status::OK();
  }
 private:
  const uint32_t mode_,pairs_;
  const bool finite_,phase_taps_;
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
  UpsampleTapColumn():Program{"RW07GeneratorUpsampleTapColumnElementChunksV2"}{}
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
    {"element_begin",ProgramUniformVariableDataType::Uint32},
    {"element_count",ProgramUniformVariableDataType::Uint32},
    {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& x=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto& column=s.AddOutput("column",ShaderUsage::UseValueTypeAlias);
    UpsampleTapGeometry(s,e);auto& f=s.MainFunctionBody();
    f<<"let elements=uniforms.output_count*uniforms.output_capacity;"
      <<"if(uniforms.element_begin>=elements||uniforms.element_count>elements-uniforms.element_begin||"
      <<"global_idx>=uniforms.element_count){return;}let at=uniforms.element_begin+global_idx;";
    f<<"if(at>=uniforms.output_count*uniforms.output_capacity){return;}"
      <<"if(!valid_output()){ "<<column.SetByOffset("at","0.0")<<"return;}"
      <<"let frame=at%uniforms.output_capacity;let count="<<e.GetByOffset("6u")<<";"
      <<"let source_channels=select(512u,256u,uniforms.stage==1u);"
      <<"if(frame>=count||uniforms.input_offset>=source_channels||uniforms.input_count==0u||"
      <<"uniforms.input_count>source_channels-uniforms.input_offset||uniforms.output_offset>=uniforms.channels||"
      <<"uniforms.output_count>uniforms.channels-uniforms.output_offset){"
      <<column.SetByOffset("at","0.0")<<"return;}"
      <<"let pos=upsample_source(frame,count);if(pos<0){"<<column.SetByOffset("at","0.0")<<"return;}"
      <<"let oc=uniforms.output_offset+at/uniforms.output_capacity;"
      <<"let kernel=select(20u,12u,uniforms.stage==1u);var total:f32=0.0;"
      <<"if(uniforms.input_offset!=0u){total="<<column.GetByOffset("at")<<";}"
      <<"for(var local_ic=0u;local_ic<uniforms.input_count;local_ic++){let ic=uniforms.input_offset+local_ic;"
      <<"let value="<<x.GetByOffset("local_ic*uniforms.input_capacity+u32(pos)")<<";"
      <<"let negative=original_mul(value,bitcast<f32>(0x3dcccccdu));let activated=select(negative,value,value>=0.0);"
      <<"total=original_add(total,original_mul(activated,"
      <<w.GetByOffset("(ic*uniforms.channels+oc)*kernel+uniforms.tap")<<"));}"
      <<column.SetByOffset("at","total");
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
