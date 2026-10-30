// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"

namespace onnxruntime::webgpu::m0_implicit_gemm {
using m0_residual_stages::ResidualConvSpecV1;
using m0_residual_stages::ValidOutput;
using m0_residual_stages::OriginalArithmetic;
// Existing packed GEMM uses an 8x8 workgroup, four adjacent time samples/lane
// and 32 inner values/barrier. Match the measured register4 kernel's32
// FP32 accumulator scalars/lane: two independent chains x4time samples x4rows.
// Upsampling retains one chain and8rows/lane within that same register envelope.
inline constexpr uint32_t kLanesX=8,kLanesY=8,kTimeTile=4*kLanesX,kInnerTile=32;
inline uint32_t TileRows(const wgpu::Limits& l,uint32_t outputs,bool two_chains=true){
  constexpr uint32_t accumulator_scalars_per_lane=2*4*4; // existing register4 FP32 geometry
  const uint32_t maximum_rows_per_lane=accumulator_scalars_per_lane/(4*(two_chains?2:1));
  for(uint32_t per_lane=maximum_rows_per_lane;per_lane>=1;per_lane/=2){
    const uint32_t rows=kLanesY*per_lane;
    const uint64_t shared=uint64_t(rows+kTimeTile)*kInnerTile*sizeof(float)+66*sizeof(uint32_t);
    if(rows<=std::max(outputs,kLanesY)&&shared<=l.maxComputeWorkgroupStorageSize)return rows;
  }
  return 0;
}
inline bool Supported(const wgpu::Limits& l,uint32_t outputs,bool skip=false){
  return M0GpuExecutionRedesignV1()&&M0BatchHeavyProgramsV1()&&M0OutputChannelTiledConvV1()&&TileRows(l,outputs)&&
    l.maxComputeWorkgroupSizeX>=kLanesX&&l.maxComputeWorkgroupSizeY>=kLanesY&&
    l.maxComputeInvocationsPerWorkgroup>=kLanesX*kLanesY&&
    l.maxStorageBuffersPerShaderStage>=uint32_t(skip?6:5)&&l.maxBindingsPerBindGroup>=uint32_t(skip?7:6);
}
#define M0_IMPLICIT_UNIFORMS \
 {"input_capacity",ProgramUniformVariableDataType::Uint32},{"output_capacity",ProgramUniformVariableDataType::Uint32}, \
 {"channels",ProgramUniformVariableDataType::Uint32},{"minimum",ProgramUniformVariableDataType::Uint32}, \
 {"maximum",ProgramUniformVariableDataType::Uint32},{"modulus",ProgramUniformVariableDataType::Uint32}, \
 {"remainder",ProgramUniformVariableDataType::Uint32},{"kernel",ProgramUniformVariableDataType::Uint32}, \
 {"dilation",ProgramUniformVariableDataType::Uint32},{"residual_capacity",ProgramUniformVariableDataType::Uint32}, \
 {"channel_offset",ProgramUniformVariableDataType::Uint32},{"channel_count",ProgramUniformVariableDataType::Uint32}, \
 {"output_channel_offset",ProgramUniformVariableDataType::Uint32},{"output_count",ProgramUniformVariableDataType::Uint32}, \
 {"group_begin",ProgramUniformVariableDataType::Uint32},{"group_count",ProgramUniformVariableDataType::Uint32}, \
 {"tile_rows",ProgramUniformVariableDataType::Uint32},{"kind",ProgramUniformVariableDataType::Uint32}, \
 {"stage",ProgramUniformVariableDataType::Uint32},{"source_channels",ProgramUniformVariableDataType::Uint32}, \
 {"identity",ProgramUniformVariableDataType::Uint32}

template <typename P> inline void Apply(P& p,const ResidualConvSpecV1& a,uint32_t begin,uint32_t count,
    uint32_t rows,uint32_t kind,uint32_t stage,uint32_t source_channels){
  p.AddUniformVariables({a.input_capacity,a.output_capacity,a.channels,a.minimum,a.maximum,a.modulus,a.remainder,
    a.kernel,a.dilation,a.residual_capacity,a.channel_offset,a.channel_count,a.output_channel_offset,
    a.output_channel_count,begin,count,rows,kind,stage,source_channels,a.identity});
}
class ActiveGrid final:public Program<ActiveGrid>{
 public:
  ActiveGrid():Program{"RW07ImplicitActiveFlatGridV1"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(M0_IMPLICIT_UNIFORMS);
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& a=s.AddOutput("arguments",ShaderUsage::UseValueTypeAlias);ValidOutput(s,e);
    auto& f=s.MainFunctionBody();f<<a.SetByOffset("0u","0u")<<a.SetByOffset("1u","1u")<<a.SetByOffset("2u","1u")
      <<"if(!valid_output()||uniforms.tile_rows==0u||uniforms.group_count==0u){return;}let n="<<e.GetByOffset("6u")<<";"
      <<"var temporal=(n+31u)/32u;if(uniforms.kind==1u){let stride=select(10u,6u,uniforms.stage==1u);"
      <<"let raw=n-select(0u,1u,uniforms.stage==1u);temporal=((raw+stride*32u-1u)/(stride*32u))*stride;}"
      <<"let groups=temporal*((uniforms.output_count+uniforms.tile_rows-1u)/uniforms.tile_rows);"
      <<"if(uniforms.group_begin<groups){"<<a.SetByOffset("0u","min(uniforms.group_count,groups-uniforms.group_begin)")<<"}";
    return Status::OK();
  }
};
class Matrix final:public Program<Matrix>{
 public:
  Matrix(uint32_t kind,uint32_t rows,bool skip,uint32_t kernel,uint32_t dilation):Program{kind?"RW07ImplicitPhaseUpsampleGemmV1":"RW07ImplicitResidualGemmSkipV1"},
      kind_(kind),rows_(rows),kernel_(kernel),dilation_(dilation),skip_(skip){
        CacheHint(std::to_string(rows),std::to_string(skip),std::to_string(kernel),std::to_string(dilation));}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(M0_IMPLICIT_UNIFORMS);
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& x=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("bias",ShaderUsage::UseValueTypeAlias);
    const auto* prior=skip_?&s.AddInput("prior",ShaderUsage::UseValueTypeAlias):nullptr;
    const auto& y=s.AddOutput("output",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e);OriginalArithmetic(s);auto& a=s.AdditionalImplementation();
    a<<"var<workgroup> implicit_count:u32;var<workgroup> implicit_flags:array<u32,64>;var<workgroup> implicit_safe:u32;"
      <<"var<private> implicit_row:u32;var<private> implicit_time:u32;var<private> implicit_phase:u32;"
      <<"var<private> implicit_pass:u32;var<private> implicit_n:u32;"
      <<"fn inner_size()->u32{return uniforms.channel_count"<<(kind_?"":"*"+std::to_string(kernel_)+"u")<<";}"
      <<"fn input_value(k:u32,t:u32)->f32 {if(k>=inner_size()){return 0.0;}";
    if(kind_){
      a<<"let stride=select(10u,6u,uniforms.stage==1u);let pad=stride/2u;"
        <<"let tap=(implicit_phase+pad)%stride+implicit_pass*stride;let numerator=i32(t+pad)-i32(tap);"
        <<"if(numerator<0){return 0.0;}let pos=u32(numerator)/stride;"
        <<"let length=select(implicit_n/10u,(implicit_n-1u)/6u,uniforms.stage==1u);"
        <<"if(pos>=length||pos>=uniforms.input_capacity){return 0.0;}let value="<<x.GetByOffset("k*uniforms.input_capacity+pos")<<";"
        <<"return select(original_mul(value,bitcast<f32>(0x3dcccccdu)),value,value>=0.0);}";
    }else{
      a<<"let ic=k/"<<kernel_<<"u;let tap=k%"<<kernel_<<"u;"
        <<"let pos=i32(t+tap*"<<dilation_<<"u)-"<<((kernel_-1)*dilation_/2)<<";"
        <<"if(pos<0||pos>=i32(implicit_n)||pos>=i32(uniforms.input_capacity)){return 0.0;}return "
        <<x.GetByOffset("ic*uniforms.input_capacity+u32(pos)")<<";}";
    }
    a<<"fn weight_value(r:u32,k:u32)->f32{if(r>=uniforms.output_count||k>=inner_size()){return 0.0;}let row=uniforms.output_channel_offset+r;";
    if(kind_)a<<"let stride=select(10u,6u,uniforms.stage==1u);let tap=(implicit_phase+stride/2u)%stride+implicit_pass*stride;"
      <<"let at=(row*uniforms.kernel+tap)*uniforms.source_channels+uniforms.channel_offset+k;";
    else a<<"let at=(row*uniforms.channels+uniforms.channel_offset)*"<<kernel_<<"u+k;";
    a<<"return "<<w.GetByOffset("at/4u")<<"[at%4u];}"
      <<"fn mm_readA(batch:i32,row:i32,col:i32)->vec4<f32>{let r=implicit_row+u32(row);let k=u32(col)*4u;"
      <<"return vec4<f32>(weight_value(r,k),weight_value(r,k+1u),weight_value(r,k+2u),weight_value(r,k+3u));}"
      <<"fn raw_frame(t:u32)->u32{return "<<(kind_?"implicit_phase+t*select(10u,6u,uniforms.stage==1u)":"t")<<";}"
      <<"fn mm_readB(batch:i32,row:i32,col:i32)->vec4<f32>{let k=u32(row);let t=implicit_time+u32(col)*4u;"
      <<"return vec4<f32>(input_value(k,raw_frame(t)),input_value(k,raw_frame(t+1u)),input_value(k,raw_frame(t+2u)),input_value(k,raw_frame(t+3u)));}"
      <<"fn finish_value(r:u32,t:u32,total_in:f32)->f32{var total=total_in;";
    if(kind_)a<<"if(implicit_pass==1u){total=original_add("<<y.GetByOffset("r*uniforms.output_capacity+t")<<",total);"
      <<"total=original_add(total,"<<b.GetByOffset("uniforms.output_channel_offset+r")<<");}";
    else{
      a<<"if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){total=original_add(total,"<<b.GetByOffset("uniforms.output_channel_offset+r")<<");";
      if(skip_)a<<"total=original_add(total,"<<prior->GetByOffset("r*uniforms.residual_capacity+t")<<");";
      a<<"}";
    }
    a<<"return total;}fn mm_write(batch:i32,row:i32,col:i32,value:vec4<f32>){let r=implicit_row+u32(row);if(r>=uniforms.output_count){return;}"
      <<"for(var lane=0u;lane<4u;lane++){let frame=raw_frame(implicit_time+u32(col)*4u+lane);"
      <<"let t=frame"<<(kind_?"+select(0u,1u,uniforms.stage==1u)":"")<<";if(t<implicit_n){"
      <<y.SetByOffset("r*uniforms.output_capacity+t","finish_value(r,t,value[lane])");
    if(kind_)a<<"if(uniforms.stage==1u&&frame==1u){"<<y.SetByOffset("r*uniforms.output_capacity","finish_value(r,0u,value[lane])")<<"}";
    a<<"}}}";
    // Exceptional tiles use original FP32 operations and original tap/IC order.
    a<<"fn serial_value(r:u32,t:u32)->f32{var total=0.0;";
    if(kind_){
      a<<"for(var tap_pass=0u;tap_pass<2u;tap_pass++){implicit_pass=tap_pass;let stride=select(10u,6u,uniforms.stage==1u);"
        <<"let tap=(implicit_phase+stride/2u)%stride+tap_pass*stride;let numerator=i32(t+stride/2u)-i32(tap);"
        <<"let length=select(implicit_n/10u,(implicit_n-1u)/6u,uniforms.stage==1u);"
        <<"if(numerator>=0&&u32(numerator)/stride<length){var column=0.0;for(var k=0u;k<inner_size();k++){"
        <<"column=original_add(column,original_mul(input_value(k,t),weight_value(r,k)));}total=original_add(total,column);}}"
        <<"return original_add(total,"<<b.GetByOffset("uniforms.output_channel_offset+r")<<");}";
    }else{
      a<<"if(uniforms.channel_offset!=0u){total="<<y.GetByOffset("r*uniforms.output_capacity+t")<<";}"
        <<"for(var k=0u;k<inner_size();k++){total=original_add(total,original_mul(input_value(k,t),weight_value(r,k)));}"
        <<"return finish_value(r,t,total);}";
    }
    auto& f=s.MainFunctionBody();
    f<<"if(local_idx==0u){implicit_count=0u;if(valid_output()){let n="<<e.GetByOffset("6u")<<";"
      <<"let source_n="<<(kind_?"select(n/10u,(n-1u)/6u,uniforms.stage==1u)":"n")<<";"
      <<"if(source_n<=uniforms.input_capacity){implicit_count=n;}}}let n=workgroupUniformLoad(&implicit_count);if(n==0u){return;}implicit_n=n;"
      <<"let group=uniforms.group_begin+workgroup_id.x;var tiles=(n+31u)/32u;";
    if(kind_)f<<"let stride=select(10u,6u,uniforms.stage==1u);let raw=n-select(0u,1u,uniforms.stage==1u);"
      <<"let per_phase=(raw+stride*32u-1u)/(stride*32u);tiles=per_phase*stride;"
      <<"implicit_phase=(group%tiles)/per_phase;implicit_time=((group%tiles)%per_phase)*32u;";
    else f<<"implicit_time=(group%tiles)*32u;implicit_phase=0u;";
    f<<"implicit_row=(group/tiles)*uniforms.tile_rows;var bad=0u;"
      <<"for(var tap_pass=0u;tap_pass<"<<(kind_?2:1)<<"u;tap_pass++){implicit_pass=tap_pass;"
      <<"for(var at=local_idx;at<uniforms.tile_rows*inner_size();at+=64u){let v=weight_value(implicit_row+at/inner_size(),at%inner_size());"
      <<"bad|=select(0u,1u,(bitcast<u32>(v)&0x7fffffffu)>0x58800000u);}"
      <<"for(var at=local_idx;at<32u*inner_size();at+=64u){let v=input_value(at/32u,raw_frame(implicit_time+at%32u));"
      <<"bad|=select(0u,1u,(bitcast<u32>(v)&0x7fffffffu)>0x58800000u);}}"
      ;
    if(!kind_)f<<"if(uniforms.channel_offset!=0u){for(var at=local_idx;at<uniforms.tile_rows*32u;at+=64u){"
      <<"let r=implicit_row+at/32u;let t=implicit_time+at%32u;if(r<uniforms.output_count&&t<n){let v="
      <<y.GetByOffset("r*uniforms.output_capacity+t")<<";bad|=select(0u,1u,(bitcast<u32>(v)&0x7fffffffu)>0x71800000u);}}}";
    f<<"implicit_flags[local_idx]=bad;workgroupBarrier();if(local_idx==0u){var all=0u;for(var i=0u;i<64u;i++){all|=implicit_flags[i];}implicit_safe=select(0u,1u,all==0u);}"
      <<"let safe=workgroupUniformLoad(&implicit_safe);implicit_pass=0u;"
      <<"if(safe==0u){for(var at=local_idx;at<uniforms.tile_rows*32u;at+=64u){let r=implicit_row+at/32u;let frame=raw_frame(implicit_time+at%32u);"
      <<"let t=frame"<<(kind_?"+select(0u,1u,uniforms.stage==1u)":"")<<";if(r<uniforms.output_count&&t<n){let value=serial_value(r,frame);"
      <<y.SetByOffset("r*uniforms.output_capacity+t","value");
    if(kind_)f<<"if(uniforms.stage==1u&&frame==1u){"<<y.SetByOffset("r*uniforms.output_capacity","value")<<"}";
    f<<"}}return;}";
    std::string prefix=std::move(f).str();f=OStringStream(kStringInitialSizeShaderSourceCodeMain);
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(s,{4,rows_/kLanesY,1},kLanesX,kLanesY,"f32",nullptr));
    std::string body=std::move(f).str();f=OStringStream(kStringInitialSizeShaderSourceCodeMain);
    const auto begin=body.find("  let localRow =");ORT_RETURN_IF_NOT(begin!=std::string::npos,"RW07 implicit GEMM body");body=body.substr(begin);
    const std::string needle="uniforms.dim_inner";
    for(size_t i=0;(i=body.find(needle,i))!=std::string::npos;i+=12)body.replace(i,needle.size(),"inner_size()");
    const std::string decl="  var acc: array<vec4<f32>, rowPerThread>;";const auto at=body.find(decl);
    ORT_RETURN_IF_NOT(at!=std::string::npos,"RW07 implicit accumulator");
    // Initialize explicitly inside the pass loop. Residual GEMM uses two
    // independent FMA chains like the measured fast register4 implementation.
    std::string initialization=kind_?"":"var acc_odd: array<vec4<f32>, rowPerThread>;";
    initialization+="for(var row=0u;row<u32(rowPerThread);row++){acc[row]=vec4<f32>(0.0);";
    if(!kind_)initialization+="acc_odd[row]=vec4<f32>(0.0);";
    initialization+="}";
    body.insert(at+decl.size(),initialization);
    if(!kind_){
      for(const auto& pair:std::array<std::pair<std::string,std::string>,2>{{
          {"acc[i] = BCached1 * ACached.y + acc[i];","acc_odd[i] = BCached1 * ACached.y + acc_odd[i];"},
          {"acc[i] = BCached3 * ACached.w + acc[i];","acc_odd[i] = BCached3 * ACached.w + acc_odd[i];"}}}){
        const auto position=body.find(pair.first);ORT_RETURN_IF_NOT(position!=std::string::npos,"RW07 implicit dual FMA chain");
        body.replace(position,pair.first.size(),pair.second);
      }
      const auto write=body.rfind("  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {");
      ORT_RETURN_IF_NOT(write!=std::string::npos,"RW07 implicit final reduction");
      body.insert(write,"for(var row=0u;row<u32(rowPerThread);row++){acc[row]=acc[row]+acc_odd[row];}");
    }
    if(!kind_){
      OStringStream seed(512);
      seed<<"if(uniforms.channel_offset!=0u){for(var r=0u;r<u32(rowPerThread);r++){for(var lane=0u;lane<4u;lane++){"
        <<"let row=implicit_row+u32(globalRow)+r;let t=implicit_time+u32(globalCol)*4u+lane;if(row<uniforms.output_count&&t<n){acc[r][lane]="
        <<y.GetByOffset("row*uniforms.output_capacity+t")<<";}}}}";
      const auto cleared=at+decl.size()+initialization.size();
      body.insert(cleared,std::move(seed).str());
    }
    f<<prefix<<"let logical_workgroup_id=vec3u(0u);let logical_global_id=local_id;for(var tap_pass=0u;tap_pass<"<<(kind_?2:1)<<"u;tap_pass++){implicit_pass=tap_pass;{"<<body<<"}}";
    return Status::OK();
  }
 private:const uint32_t kind_,rows_,kernel_,dilation_;const bool skip_;
};
inline Status Run(ComputeContextBase& c,const ProgramInput& input,const ProgramOutput& output,
    const Tensor* extent,const Tensor* weight,const Tensor* bias,Tensor* arguments,
    const ResidualConvSpecV1& spec,uint64_t& dispatches,const ProgramInput* skip=nullptr,
    uint32_t kind=0,uint32_t stage=0,uint32_t source_channels=0,bool checked=false){
  const auto& l=c.DeviceLimits();const uint32_t rows=TileRows(l,spec.output_channel_count,kind==0);
  ORT_RETURN_IF_NOT(Supported(l,spec.output_channel_count,skip!=nullptr)&&arguments&&arguments->IsDataType<uint32_t>()&&arguments->Shape().Size()==3&&
    extent&&extent->IsDataType<uint32_t>()&&extent->Shape().Size()==32&&weight&&weight->IsDataType<float>()&&weight->Shape().Size()%4==0&&
    bias&&bias->IsDataType<float>()&&spec.output_channel_count&&spec.channel_count&&spec.modulus&&spec.output_capacity&&spec.input_capacity&&
    spec.output_channel_offset<spec.channels&&spec.output_channel_count<=spec.channels-spec.output_channel_offset&&kind<=1&&stage<=1,
    "RW07 implicit convolution descriptors/capability");
  ORT_RETURN_IF_NOT(spec.kernel>0&&spec.kernel<=20&&spec.dilation>0&&spec.dilation<=5&&bias->Shape().Size()>=spec.channels&&
    input.tensor&&input.tensor->IsDataType<float>()&&output.tensor&&output.tensor->IsDataType<float>()&&
    uint64_t(input.tensor->Shape().Size())>=uint64_t(spec.channel_count)*spec.input_capacity&&
    uint64_t(output.tensor->Shape().Size())>=uint64_t(spec.output_channel_count)*spec.output_capacity&&
    (kind?(source_channels&&spec.channel_offset<source_channels&&spec.channel_count<=source_channels-spec.channel_offset):
           (spec.channel_offset<spec.channels&&spec.channel_count<=spec.channels-spec.channel_offset))&&
    uint64_t(weight->Shape().Size())==uint64_t(spec.channels)*(kind?source_channels:spec.channels)*spec.kernel&&
    (!skip||(skip->tensor&&skip->tensor->IsDataType<float>()&&uint64_t(skip->tensor->Shape().Size())>=uint64_t(spec.output_channel_count)*spec.residual_capacity)),
    "RW07 implicit convolution exact operand extents");
  const uint64_t inner=uint64_t(spec.channel_count)*(kind?1:spec.kernel),rounded=(inner+kInnerTile-1)/kInnerTile*kInnerTile;
  const uint64_t products=uint64_t(rows+(kind?1:0))*kTimeTile*rounded*(kind?2:1);
  const uint32_t stride=kind?(stage?6u:10u):1u;
  const uint64_t temporal=((uint64_t(spec.output_capacity)-(kind&&stage?1:0)+stride*kTimeTile-1)/(stride*kTimeTile))*stride;
  const uint64_t physical=temporal*((uint64_t(spec.output_channel_count)+rows-1)/rows);
  const uint64_t window=std::min<uint64_t>(l.maxComputeWorkgroupsPerDimension,kM0ScheduledProductsPerWindowV1/products);
  ORT_RETURN_IF_NOT(window&&physical<=UINT32_MAX,"RW07 implicit submitted work/index bound");
  for(uint64_t first=0;first<physical;first+=window){
    const uint32_t count=uint32_t(std::min(window,physical-first));ActiveGrid grid;
    grid.AddInput({extent,ProgramTensorMetadataDependency::None}).AddOutput({arguments,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(1);
    Apply(grid,spec,uint32_t(first),count,rows,kind,stage,source_channels);ORT_RETURN_IF_ERROR(c.RunProgram(grid));++dispatches;
    Matrix matrix(kind,rows,skip!=nullptr,spec.kernel,spec.dilation);
    matrix.AddInput(ProgramInput(input)).AddInput({extent,ProgramTensorMetadataDependency::None})
      .AddInput({weight,ProgramTensorMetadataDependency::None,ProgramInput::Flatten,4}).AddInput({bias,ProgramTensorMetadataDependency::None});
    if(skip)matrix.AddInput(ProgramInput(*skip));
    matrix.AddOutput(ProgramOutput(output)).SetWorkgroupSize(kLanesX,kLanesY).SetIndirectDispatchTensor(arguments);
    Apply(matrix,spec,uint32_t(first),count,rows,kind,stage,source_channels);
    if(checked&&first+count==physical)ORT_RETURN_IF_ERROR(c.RunProgramWithQueuedCompletion(matrix,count*products));
    else ORT_RETURN_IF_ERROR(c.RunProgram(matrix,count*products));
    ++dispatches;
  }
  return Status::OK();
}
#undef M0_IMPLICIT_UNIFORMS
} // namespace onnxruntime::webgpu::m0_implicit_gemm
