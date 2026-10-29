// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <bit>
#include <string>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"
namespace onnxruntime::webgpu::m0_recurrence_programs {
// Each payload program checks the complete successful output descriptor itself.
inline void AddValidExtent(ShaderHelper& shader, const ShaderVariableHelper& extent) {
  auto& f = shader.AdditionalImplementation();
  f << "fn valid_extent() -> bool {";
  f << "if(" << extent.GetByOffset("0u") << "!=2u||" << extent.GetByOffset("1u") << "!=3u||"
    << extent.GetByOffset("20u") << "!=0u||" << extent.GetByOffset("21u") << "!=63u){return false;}";
  f << "for(var i=0u;i<10u;i++){if(" << extent.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}";
  f << "if(" << extent.GetByOffset("2u") << "!=1u||" << extent.GetByOffset("3u") << "!=0u||"
    << extent.GetByOffset("4u") << "!=512u||" << extent.GetByOffset("5u") << "!=0u||"
    << extent.GetByOffset("7u") << "!=0u||" << extent.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << extent.GetByOffset("i") << "!=0u){return false;}}"
    << "let d=" << extent.GetByOffset("6u") << ";return d>0u&&d<=uniforms.input_capacity&&"
    << "d<=uniforms.output_capacity&&" << extent.GetByOffset("18u") << "==512u*d;}\n";
}
// Each invocation computes the four independent input gates for one state
// unit at one frame. The recurrence order and hidden-state calculation stay
// unchanged; no CPU tensor computation or readback is involved.
inline constexpr uint32_t kProjectionElementsPerDispatchV1=1600;
inline constexpr uint64_t kProjectionProductsPerElementV1=640u*4u;
inline uint32_t ProjectionElementsPerDispatchV1(uint32_t group_limit,bool batched){
  if(!group_limit)return 0;
  const uint64_t products=uint64_t{1}<<(batched?25:22);
  const uint64_t lanes=products/kProjectionProductsPerElementV1/64*64;
  // Clamp before multiplying so even an invalid oversized device limit cannot overflow.
  const uint64_t dimension=std::min<uint64_t>(group_limit,lanes/64);
  return static_cast<uint32_t>(std::min(lanes,dimension*dimension*64));
}
inline uint64_t ProjectionScheduledProductsV1(uint32_t count){
  return ((uint64_t(count)+63)/64)*64*kProjectionProductsPerElementV1;
}
// Exact permutation only: neighboring state units read neighboring weights.
class RecurrenceWeightTranspose final : public Program<RecurrenceWeightTranspose> {
 public:
  RecurrenceWeightTranspose():Program{"RW07V5RecurrenceWeightTranspose"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"width",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& input=s.AddInput("weights",ShaderUsage::UseValueTypeAlias);
    const auto& output=s.AddOutput("transposed",ShaderUsage::UseValueTypeAlias);
    s.MainFunctionBody()<<"let count=2048u*uniforms.width;if(global_idx>=count){return;}"
      <<"let dir=global_idx/(1024u*uniforms.width);let k=(global_idx/1024u)%uniforms.width;let unit=global_idx%1024u;"
      <<output.SetByOffset("global_idx",input.GetByOffset("(dir*1024u+unit)*uniforms.width+k"));
    return Status::OK();
  }
};
inline Status RunWeightTranspose(ComputeContextBase& context,const Tensor* input,Tensor* output,uint32_t width){
  const uint64_t limit=context.DeviceLimits().maxComputeWorkgroupsPerDimension;
  const uint64_t groups=(uint64_t(width)*2048u+63u)/64u;
  ORT_RETURN_IF_NOT((width==640u||width==256u)&&limit>0&&groups<=limit*limit&&
      input->SizeInBytes()==uint64_t(width)*2048u*sizeof(float)&&output->SizeInBytes()==input->SizeInBytes(),
      "RW07 coalesced recurrence weight descriptor/dispatch bounds");
  const uint64_t gx=std::min(groups,limit),gy=(groups+gx-1u)/gx;
  RecurrenceWeightTranspose program;
  program.AddInput({input,ProgramTensorMetadataDependency::None}).AddOutput({output,ProgramTensorMetadataDependency::None})
    .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy)).AddUniformVariables({width});
  return context.RunProgram(program);
}
inline std::string WeightOffset(const char* base,const std::string& unit,const char* width,bool coalesced){
  return std::string(base)+(coalesced?"+k*1024u+("+unit+")":"+("+unit+")*"+width+"+k");
}
class RecurrenceInputProjection final : public Program<RecurrenceInputProjection> {
 public:
  explicit RecurrenceInputProjection(bool coalesced=false):Program{coalesced?"RW07V5LstmCoalescedInputProjection":"RW07V3LstmInputProjectionFrameChunks"},coalesced_(coalesced){}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity",ProgramUniformVariableDataType::Uint32},
      {"output_capacity",ProgramUniformVariableDataType::Uint32},
      {"identity",ProgramUniformVariableDataType::Uint32},
      {"element_begin",ProgramUniformVariableDataType::Uint32},
      {"element_count",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("x",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("w",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("projection",ShaderUsage::UseValueTypeAlias);
    AddValidExtent(s,e);
    auto& f=s.MainFunctionBody();
    f << "let elements=uniforms.output_capacity*512u;"
      << "if(uniforms.element_begin>=elements||uniforms.element_count>elements-uniforms.element_begin||"
      << "global_idx>=uniforms.element_count){return;}let at=uniforms.element_begin+global_idx;"
      << "let unit=at%256u;let direction=(at/256u)%2u;let t=at/512u;let base=(t*2u+direction)*1024u+unit;"
      << y.SetByOffset("base","0.0") << y.SetByOffset("base+256u","0.0")
      << y.SetByOffset("base+512u","0.0") << y.SetByOffset("base+768u","0.0")
      << "if(!valid_extent()||t>=" << e.GetByOffset("6u") << "){return;}"
      << "var gi=0.0;var go=0.0;var gf=0.0;var gc=0.0;let wb=direction*1024u*640u;"
      << "for(var k=0u;k<640u;k++){let value=" << x.GetByOffset("k*uniforms.input_capacity+t") << ";"
      << "gi+=value*" << w.GetByOffset(WeightOffset("wb","unit","640u",coalesced_)) << ";"
      << "go+=value*" << w.GetByOffset(WeightOffset("wb","256u+unit","640u",coalesced_)) << ";"
      << "gf+=value*" << w.GetByOffset(WeightOffset("wb","512u+unit","640u",coalesced_)) << ";"
      << "gc+=value*" << w.GetByOffset(WeightOffset("wb","768u+unit","640u",coalesced_)) << ";}"
      << y.SetByOffset("base","gi") << y.SetByOffset("base+256u","go")
      << y.SetByOffset("base+512u","gf") << y.SetByOffset("base+768u","gc");
    return Status::OK();
  }
 private:
  const bool coalesced_;
};
// Common production/test caller. Every larger projection charges the actual
// rounded grid work to the unchanged aggregate submission window.
inline Status RunInputProjectionV1(ComputeContextBase& context,const Tensor* x,const Tensor* weights,
    const Tensor* extent,Tensor* projection,uint32_t input_capacity,uint32_t output_capacity,
    gsl::span<const uint32_t> identity,bool coalesced,uint32_t chunk_override=0){
  const uint32_t limit=context.DeviceLimits().maxComputeWorkgroupsPerDimension;
  const uint32_t admitted=ProjectionElementsPerDispatchV1(limit,M0BatchHeavyProgramsV1());
  ORT_RETURN_IF_NOT(admitted&&input_capacity&&input_capacity<=102400&&output_capacity&&output_capacity<=102400&&
      identity.size()==10&&x&&weights&&extent&&projection&&
      x->IsDataType<float>()&&weights->IsDataType<float>()&&extent->IsDataType<uint32_t>()&&projection->IsDataType<float>()&&
      x->Shape()==TensorShape({1,640,input_capacity})&&extent->Shape()==TensorShape{32}&&
      weights->Shape()==(coalesced?TensorShape({2,640,1024}):TensorShape({2,1024,640}))&&
      projection->Shape()==TensorShape({output_capacity,2,1024})&&
      (!chunk_override||chunk_override<=admitted),"RW07 input projection descriptor/work bounds");
  const uint32_t chunk=chunk_override?chunk_override:admitted;
  const uint64_t elements=uint64_t(output_capacity)*512;
  for(uint64_t first=0;first<elements;first+=chunk){
    const uint32_t count=static_cast<uint32_t>(std::min<uint64_t>(chunk,elements-first));
    const uint32_t groups=(count+63)/64,gx=std::min(groups,limit),gy=(groups+gx-1)/gx;
    ORT_RETURN_IF_NOT(gy<=limit,"RW07 input projection dispatch bound");
    RecurrenceInputProjection program{coalesced};
    program.AddInputs({{x,ProgramTensorMetadataDependency::None},{weights,ProgramTensorMetadataDependency::None},
        {extent,ProgramTensorMetadataDependency::None}})
        .AddOutput({projection,ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize(gx,gy)
        .AddUniformVariables({input_capacity,output_capacity,identity,uint32_t(first),count});
    ORT_RETURN_IF_ERROR(context.RunProgram(program,ProjectionScheduledProductsV1(count)));
  }
  return Status::OK();
}
// Matrix geometry follows the device subgroup/lane/storage envelope. The four
// gates and640 input features are original-model constants, not tuning knobs.
class TiledRecurrenceProjectionV1 final : public Program<TiledRecurrenceProjectionV1> {
 public:
  TiledRecurrenceProjectionV1(bool coalesced,uint32_t lanes,uint32_t units,uint32_t inner)
    :Program{"RW07TiledRecurrenceProjectionV6"},coalesced_(coalesced),lanes_(lanes),units_(units),inner_(inner){
    CacheHint(std::to_string(coalesced),std::to_string(lanes),std::to_string(units),std::to_string(inner));}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},{"identity",ProgramUniformVariableDataType::Uint32},
    {"first_time",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader)const override{
    const auto& x=shader.AddInput("x",ShaderUsage::UseValueTypeAlias);
    const auto& w=shader.AddInput("w",ShaderUsage::UseValueTypeAlias);
    const auto& extent=shader.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& out=shader.AddOutput("projection",ShaderUsage::UseValueTypeAlias);
    AddValidExtent(shader,extent);
    const uint32_t times=lanes_/units_;
    shader.AdditionalImplementation()<<"const tile_units="<<units_<<"u;const tile_times="<<times
      <<"u;const tile_k="<<inner_<<"u;const tile_lanes="<<lanes_<<"u;"
      <<"var<workgroup> matrix_x:array<f32,"<<times*inner_<<">;var<workgroup> matrix_w:array<f32,"<<4*units_*inner_
      <<">;var<workgroup> live_count:u32;";
    auto& body=shader.MainFunctionBody();
    body<<"let dir=workgroup_id.y/(256u/tile_units);let first_unit=(workgroup_id.y%(256u/tile_units))*tile_units;"
      <<"let first_time=uniforms.first_time+workgroup_id.x*tile_times;"
      <<"if(local_idx==0u){live_count=0u;if(valid_extent()){live_count="<<extent.GetByOffset("6u")<<";}}workgroupBarrier();"
      <<"let len=workgroupUniformLoad(&live_count);if(first_time>=len){return;}"
      <<"let unit=first_unit+local_idx%tile_units;let time=first_time+local_idx/tile_units;let wb=dir*1024u*640u;"
      <<"var gi=0.0;var go=0.0;var gf=0.0;var gc=0.0;"
      <<"for(var block=0u;block<640u;block+=tile_k){"
      <<"for(var i=local_idx;i<tile_k*tile_times;i+=tile_lanes){let t=first_time+i%tile_times;var v=0.0;if(t<len){v="
      <<x.GetByOffset("(block+i/tile_times)*uniforms.input_capacity+t")<<";}matrix_x[i]=v;}"
      <<"for(var i=local_idx;i<4u*tile_k*tile_units;i+=tile_lanes){let k=block+(i/tile_units)%tile_k;let gate=i/(tile_k*tile_units);"
      <<"let row=gate*256u+first_unit+i%tile_units;matrix_w[i]="
      <<w.GetByOffset(WeightOffset("wb","row","640u",coalesced_))<<";}workgroupBarrier();"
      <<"for(var k=0u;k<tile_k;k++){let value=matrix_x[k*tile_times+local_idx/tile_units];let wi=k*tile_units+local_idx%tile_units;"
      <<"gi=gi+value*matrix_w[wi];go=go+value*matrix_w[tile_k*tile_units+wi];"
      <<"gf=gf+value*matrix_w[2u*tile_k*tile_units+wi];gc=gc+value*matrix_w[3u*tile_k*tile_units+wi];}workgroupBarrier();}"
      <<"if(time<len){let base=(time*2u+dir)*1024u+unit;"
      <<out.SetByOffset("base","gi")<<out.SetByOffset("base+256u","go")
      <<out.SetByOffset("base+512u","gf")<<out.SetByOffset("base+768u","gc")<<"}";
    return Status::OK();
  }
 private:bool coalesced_;uint32_t lanes_,units_,inner_;
};
inline Status RunTiledInputProjectionV1(ComputeContextBase& context,const Tensor* x,const Tensor* weights,
    const Tensor* extent,Tensor* projection,uint32_t input_capacity,uint32_t output_capacity,
    gsl::span<const uint32_t> identity,bool coalesced,bool* used_tiled=nullptr){
  if(used_tiled)*used_tiled=false;
  ORT_RETURN_IF_NOT(input_capacity&&input_capacity<=102400&&output_capacity&&output_capacity<=102400&&identity.size()==10&&
      x&&weights&&extent&&projection&&x->IsDataType<float>()&&weights->IsDataType<float>()&&
      extent->IsDataType<uint32_t>()&&projection->IsDataType<float>()&&
      x->Shape()==TensorShape({1,640,input_capacity})&&extent->Shape()==TensorShape{32}&&
      weights->Shape()==(coalesced?TensorShape({2,640,1024}):TensorShape({2,1024,640}))&&
      projection->Shape()==TensorShape({output_capacity,2,1024}),"RW07 tiled projection descriptor bounds");
  const auto& limits=context.DeviceLimits();
  constexpr uint32_t gates=4,features=640,hidden=256,directions=2,scalar_bytes=sizeof(float),shared_alignment=16;
  // No subgroup operations are used. If the optional subgroup limits are
  // unavailable, derive a cooperative group from one hidden unit per gate
  // instead of silently reducing the group to a single lane.
  const uint32_t preferred_lanes=context.AdapterInfo().subgroupMaxSize?
      context.AdapterInfo().subgroupMaxSize:hidden/gates;
  const uint32_t lanes=std::bit_floor(std::min({limits.maxComputeWorkgroupSizeX,limits.maxComputeInvocationsPerWorkgroup,
      preferred_lanes,hidden}));
  const auto legacy=[&](){return RunInputProjectionV1(context,x,weights,extent,projection,input_capacity,output_capacity,identity,coalesced);};
  if(lanes<gates||limits.maxComputeWorkgroupStorageSize<=shared_alignment)return legacy();
  const uint32_t units=1u<<((std::bit_width(lanes)-1)/2),times=lanes/units;
  // Square output tile from the lane count; at most one K coefficient per
  // lane per gate keeps cooperative loading bounded. Clamp to actual shared RAM.
  const uint32_t inner=std::bit_floor(std::min({features,lanes/gates,
      (limits.maxComputeWorkgroupStorageSize-shared_alignment)/scalar_bytes/(times+gates*units)}));
  const uint32_t unit_groups=directions*(hidden/units);
  if(!inner||features%inner||limits.maxComputeWorkgroupsPerDimension<unit_groups)return legacy();
  const uint64_t products_per_group=uint64_t(lanes)*gates*features;
  const uint32_t blocks=uint32_t(std::min<uint64_t>(limits.maxComputeWorkgroupsPerDimension,
      kM0ScheduledProductsPerWindowV1/(products_per_group*unit_groups)));
  ORT_RETURN_IF_NOT(blocks>0,"RW07 tiled input projection work budget");
  if(used_tiled)*used_tiled=true;
  for(uint32_t first=0;first<output_capacity;){
    const uint32_t groups=std::min(blocks,(output_capacity-first+times-1)/times);
    TiledRecurrenceProjectionV1 program(coalesced,lanes,units,inner);
    program.AddInputs({{x,ProgramTensorMetadataDependency::None},{weights,ProgramTensorMetadataDependency::None},
        {extent,ProgramTensorMetadataDependency::None}}).AddOutput({projection,ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(lanes).SetDispatchGroupSize(groups,unit_groups)
      .AddUniformVariables({input_capacity,output_capacity,identity,first});
    ORT_RETURN_IF_ERROR(context.RunProgram(program,uint64_t(groups)*unit_groups*products_per_group));first+=groups*times;
  }
  return Status::OK();
}
class RecurrenceData final : public Program<RecurrenceData> {
 public:
  explicit RecurrenceData(bool projected=false,bool coalesced=false) : Program{coalesced?(projected?"RW07V5RecurrenceCoalescedProjected":"RW07V5RecurrenceCoalescedData"):(projected?"RW07V3RecurrencePreprojectedData":"RW07V2RecurrenceOwnedData")},projected_(projected),coalesced_(coalesced) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32},
      {"input_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("w", ShaderUsage::UseValueTypeAlias);
    const auto& r = s.AddInput("r", ShaderUsage::UseValueTypeAlias);
    const auto& b = s.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    const auto& h = s.AddOutput("final_h", ShaderUsage::UseValueTypeAlias);
    const auto& c = s.AddOutput("final_c", ShaderUsage::UseValueTypeAlias);
    AddValidExtent(s, e);
    // Same scalar activation/gate boundary as the qualified default LSTM source.
    s.AdditionalImplementation()
        << "fn sigmoid_f(v:f32)->f32 {let bits=bitcast<u32>(v);let magnitude=bits&0x7fffffffu;"
        << "if(magnitude>0x7f800000u){return v;}if(magnitude==0x7f800000u){return select(0.0,1.0,(bits&0x80000000u)==0u);}"
        << "if(v>=0.0){return 1.0/(1.0+exp(-v));}let ex=exp(v);return ex/(1.0+ex);}\n"
        << "fn tanh_f(v:f32)->f32 {let bits=bitcast<u32>(v);let magnitude=bits&0x7fffffffu;"
        << "if(magnitude>0x7f800000u){return v;}if(magnitude==0x7f800000u){return select(-1.0,1.0,(bits&0x80000000u)==0u);}return tanh(v);}\n"
        << "fn lstm_gate_input(v:f32)->f32 {let magnitude=bitcast<u32>(v)&0x7fffffffu;let bound=bitcast<f32>(0x7f7fffffu);"
        << "if(magnitude>0x7f800000u){return bound;}if(v>bound){return bound;}if(v< -bound){return -bound;}return v;}\n"
        << "var<workgroup> previous_h:array<f32,256>;var<workgroup> previous_c:array<f32,256>;var<workgroup> active_length:u32;\n";
    auto& f = s.MainFunctionBody();
    f << "let unit=local_idx;let direction=workgroup_id.x;let channel=direction*256u+unit;"
      << "previous_h[unit]=0.0;previous_c[unit]=0.0;"
      << h.SetByOffset("channel", "0.0") << c.SetByOffset("channel", "0.0")
      << "for(var t=0u;t<uniforms.output_capacity;t++){" << y.SetByOffset("(t*2u+direction)*256u+unit", "0.0") << "}"
      << "if(unit==0u){active_length=0u;if(valid_extent()){active_length=" << e.GetByOffset("6u") << ";}}"
      << "let len=workgroupUniformLoad(&active_length);if(len==0u){return;}"
      << "for(var step=0u;step<len;step++){let t=select(step,len-1u-step,direction==1u);"
      << "var gate_i:f32=0.0;var gate_o:f32=0.0;var gate_f:f32=0.0;var gate_c:f32=0.0;"
      ;
    if(projected_){
      f << "let base=(t*2u+direction)*1024u+unit;"
        << "gate_i=" << x.GetByOffset("base") << ";gate_o=" << x.GetByOffset("base+256u")
        << ";gate_f=" << x.GetByOffset("base+512u") << ";gate_c=" << x.GetByOffset("base+768u") << ";";
    }else{
      f
      << "let w_base=direction*1024u*640u;for(var k=0u;k<uniforms.input_size;k++){let xv=" << x.GetByOffset("k*uniforms.input_capacity+t") << ";"
      << "gate_i+=xv*" << w.GetByOffset(WeightOffset("w_base","unit","640u",coalesced_)) << ";"
      << "gate_o+=xv*" << w.GetByOffset(WeightOffset("w_base","256u+unit","640u",coalesced_)) << ";"
      << "gate_f+=xv*" << w.GetByOffset(WeightOffset("w_base","512u+unit","640u",coalesced_)) << ";"
      << "gate_c+=xv*" << w.GetByOffset(WeightOffset("w_base","768u+unit","640u",coalesced_)) << ";}"
      ;
    }
    f
      << "let r_base=direction*1024u*256u;for(var k=0u;k<uniforms.hidden_size;k++){let hv=previous_h[k];"
      << "gate_i+=hv*" << r.GetByOffset(WeightOffset("r_base","unit","256u",coalesced_)) << ";"
      << "gate_o+=hv*" << r.GetByOffset(WeightOffset("r_base","256u+unit","256u",coalesced_)) << ";"
      << "gate_f+=hv*" << r.GetByOffset(WeightOffset("r_base","512u+unit","256u",coalesced_)) << ";"
      << "gate_c+=hv*" << r.GetByOffset(WeightOffset("r_base","768u+unit","256u",coalesced_)) << ";}"
      << "let bb=direction*2048u;gate_i+=" << b.GetByOffset("bb+unit") << "+" << b.GetByOffset("bb+1024u+unit") << ";"
      << "gate_f+=" << b.GetByOffset("bb+512u+unit") << "+" << b.GetByOffset("bb+1536u+unit") << ";"
      << "gate_c+=" << b.GetByOffset("bb+768u+unit") << "+" << b.GetByOffset("bb+1792u+unit") << ";"
      << "gate_i=sigmoid_f(lstm_gate_input(gate_i));gate_f=sigmoid_f(lstm_gate_input(gate_f));gate_c=tanh_f(lstm_gate_input(gate_c));"
      << "let cu=gate_f*previous_c[unit]+gate_i*gate_c;"
      << "gate_o+=" << b.GetByOffset("bb+256u+unit") << "+" << b.GetByOffset("bb+1280u+unit") << ";"
      << "gate_o=sigmoid_f(lstm_gate_input(gate_o));let hu=gate_o*tanh_f(cu);"
      << "workgroupBarrier();previous_c[unit]=cu;previous_h[unit]=hu;workgroupBarrier();"
      << y.SetByOffset("(t*2u+direction)*256u+unit", "hu") << "}"
      << h.SetByOffset("channel", "previous_h[unit]") << c.SetByOffset("channel", "previous_c[unit]");
    return Status::OK();
  }
 private:
  const bool projected_;
  const bool coalesced_;
};
class RecurrenceMerged final : public Program<RecurrenceMerged> {
 public:
  RecurrenceMerged() : Program{"RW07V2RecurrenceMerged"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& y = s.AddInput("y", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& m = s.AddOutput("merged", ShaderUsage::UseValueTypeAlias);
    AddValidExtent(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=512u*uniforms.output_capacity){return;}" << m.SetByOffset("global_idx", "0.0")
      << "if(!valid_extent()){return;}let channel=global_idx/uniforms.output_capacity;let t=global_idx%uniforms.output_capacity;"
      << "if(t>=" << e.GetByOffset("6u") << "){return;}"
      << m.SetByOffset("global_idx", y.GetByOffset("(t*2u+channel/256u)*256u+channel%256u"));
    return Status::OK();
  }
};
class RecurrenceMoments final : public Program<RecurrenceMoments> {
 public:
  RecurrenceMoments() : Program{"RW07V2RecurrenceMoments"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& m = s.AddInput("merged", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& mean = s.AddOutput("mean", ShaderUsage::UseValueTypeAlias);
    const auto& variance = s.AddOutput("variance", ShaderUsage::UseValueTypeAlias);
    AddValidExtent(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=512u){return;}" << mean.SetByOffset("global_idx", "0.0") << variance.SetByOffset("global_idx", "0.0")
      << "if(!valid_extent()){return;}let len=" << e.GetByOffset("6u") << ";var sum=0.0;"
      << "for(var t=0u;t<len;t++){sum+=" << m.GetByOffset("global_idx*uniforms.output_capacity+t") << ";}"
      << "let mu=sum/f32(len);var squares=0.0;for(var t=0u;t<len;t++){let delta="
      << m.GetByOffset("global_idx*uniforms.output_capacity+t") << "-mu;squares+=delta*delta;}"
      << mean.SetByOffset("global_idx", "mu") << variance.SetByOffset("global_idx", "squares/f32(len)");
    return Status::OK();
  }
};
// A group owns one channel. It writes the layout once and computes moments
// with compensated parallel reductions, avoiding a separate copy program and
// one extra complete read of the merged carrier.
class RecurrenceMergedMoments final : public Program<RecurrenceMergedMoments> {
 public:
  RecurrenceMergedMoments():Program{"RW07V4RecurrenceFusedMergedMoments"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity",ProgramUniformVariableDataType::Uint32},
      {"output_capacity",ProgramUniformVariableDataType::Uint32},
      {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& y=s.AddInput("y",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& merged=s.AddOutput("merged",ShaderUsage::UseValueTypeAlias);
    const auto& mean=s.AddOutput("mean",ShaderUsage::UseValueTypeAlias);
    const auto& variance=s.AddOutput("variance",ShaderUsage::UseValueTypeAlias);
    AddValidExtent(s,e);m0_residual_stages::OriginalArithmetic(s);
    s.AdditionalImplementation()<<R"WGSL(
var<workgroup> moment_partial:array<vec2<f32>,64>;
var<workgroup> moment_mean:f32;
var<workgroup> moment_count:u32;
fn moment_accumulate(state:vec2<f32>,value:f32)->vec2<f32>{
 let adjusted=original_sub(value,state.y);let next=original_add(state.x,adjusted);
 if((bitcast<u32>(state.x)&0x7fffffffu)>=0x7f800000u||
    (bitcast<u32>(value)&0x7fffffffu)>=0x7f800000u||
    (bitcast<u32>(next)&0x7fffffffu)>=0x7f800000u){return vec2<f32>(original_add(state.x,value),0.0);}
 return vec2<f32>(next,original_sub(original_sub(next,state.x),adjusted));
}
)WGSL";
    auto& f=s.MainFunctionBody();
    f << "let channel=workgroup_idx;if(channel>=512u){return;}"
      << "if(local_idx==0u){moment_count=0u;moment_mean=0.0;" << mean.SetByOffset("channel","0.0") << variance.SetByOffset("channel","0.0")
      << "if(valid_extent()){moment_count=" << e.GetByOffset("6u") << ";}}"
      << "let count=workgroupUniformLoad(&moment_count);var sum=vec2<f32>(0.0);"
      << "for(var t=local_idx;t<uniforms.output_capacity;t+=64u){var value=0.0;"
      << "if(t<count){value=" << y.GetByOffset("(t*2u+channel/256u)*256u+channel%256u") << ";sum=moment_accumulate(sum,value);}"
      << merged.SetByOffset("channel*uniforms.output_capacity+t","value") << "}"
      << "moment_partial[local_idx]=sum;workgroupBarrier();"
      << "if(local_idx==0u&&count>0u){var total=vec2<f32>(0.0);for(var lane=0u;lane<64u;lane++){total=moment_accumulate(total,moment_partial[lane].x);}"
      << "moment_mean=original_div(total.x,f32(count));" << mean.SetByOffset("channel","moment_mean") << "}"
      << "workgroupBarrier();var squares=vec2<f32>(0.0);for(var t=local_idx;t<count;t+=64u){"
      << "let delta=original_sub(" << y.GetByOffset("(t*2u+channel/256u)*256u+channel%256u") << ",moment_mean);"
      << "squares=moment_accumulate(squares,original_mul(delta,delta));}moment_partial[local_idx]=squares;workgroupBarrier();"
      << "if(local_idx==0u&&count>0u){var total=vec2<f32>(0.0);for(var lane=0u;lane<64u;lane++){total=moment_accumulate(total,moment_partial[lane].x);}"
      << variance.SetByOffset("channel","original_div(total.x,f32(count))") << "}";
    return Status::OK();
  }
};
inline Status RunFinalization(ComputeContextBase& context,const Tensor* y,const Tensor* extent,
    Tensor* merged,Tensor* mean,Tensor* variance,uint32_t input_capacity,uint32_t output_capacity,
    gsl::span<const uint32_t> identity,bool fused){
  const uint64_t limit=context.DeviceLimits().maxComputeWorkgroupsPerDimension;
  const uint64_t groups=(uint64_t(output_capacity)*512u+63u)/64u;
  ORT_RETURN_IF_NOT(input_capacity>0&&output_capacity>0&&identity.size()==10&&limit>=8&&groups<=limit*limit,
                    "RW07 recurrence finalization descriptor/dispatch bounds");
  const auto words=[&](){
    ProgramInput input(extent,ProgramTensorMetadataDependency::None,TensorShape{32},1);
    input.var_type=ProgramVariableDataType::Uint32;return input;
  };
  if(fused&&limit*limit>=512u){
    const uint32_t gx=uint32_t(std::min<uint64_t>(512u,limit)),gy=(512u+gx-1u)/gx;
    RecurrenceMergedMoments program;
    program.AddInputs({{y,ProgramTensorMetadataDependency::None},words()})
      .AddOutputs({{merged,ProgramTensorMetadataDependency::None},{mean,ProgramTensorMetadataDependency::None},{variance,ProgramTensorMetadataDependency::None}})
      .SetWorkgroupSize(64).SetDispatchGroupSize(gx,gy).AddUniformVariables({input_capacity,output_capacity,identity});
    return context.RunProgram(program);
  }
  const uint64_t gx=std::min(groups,limit),gy=(groups+gx-1u)/gx;
  RecurrenceMerged copy;
  copy.AddInputs({{y,ProgramTensorMetadataDependency::None},words()}).AddOutput({merged,ProgramTensorMetadataDependency::None})
    .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy)).AddUniformVariables({input_capacity,output_capacity,identity});
  ORT_RETURN_IF_ERROR(context.RunProgram(copy));
  RecurrenceMoments moments;
  moments.AddInputs({{merged,ProgramTensorMetadataDependency::None},words()})
    .AddOutputs({{mean,ProgramTensorMetadataDependency::None},{variance,ProgramTensorMetadataDependency::None}})
    .SetWorkgroupSize(64).SetDispatchGroupSize(8).AddUniformVariables({input_capacity,output_capacity,identity});
  return context.RunProgram(moments);
}
}  // namespace onnxruntime::webgpu::m0_recurrence_programs
