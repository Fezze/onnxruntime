// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <chrono>
#include "core/common/m0_observation_options_v1.h"
#include <cstdint>
#include <initializer_list>
#include <string>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_trig_v1.h"
#include "core/providers/webgpu/math/gemm_utils.h"

// Both original generator and noise residual paths use the same scalar stage
// ordering. Roles, extents, parameters and capacities remain caller supplied.
namespace onnxruntime::webgpu::m0_residual_stages {
// Normalization division/affine product; Snake adds angle, square and scale.
inline constexpr uint32_t kNormAffineWorkPerElementV1=2;
inline constexpr uint32_t kSnakeWorkPerElementV1=m0_trig_detail::kSincosProductsPerElementV1+3;
inline Status RunHeavyProgram(ComputeContextBase& context,const ProgramBase& program,uint64_t active_elements){
  // The native recording still fences each bounded dispatch/resource window and
  // final publication. Keep the legacy per-program isolation available.
  // Pointwise callers charge their actual rounded element range; convolution
  // callers account for the inner products separately.
  return M0BatchHeavyProgramsV1()?context.RunProgram(program,active_elements):context.RunProgramWithCheckedCompletion(program,active_elements);
}

// One immutable sidecar predicate per shader. Insert before any lane-specific
// return; all lanes must participate in this initialization barrier.
inline void ShareOutputValidityV1(ShaderHelper& s){
  s.AdditionalImplementation()<<"var<workgroup> m0_shared_extent_valid:u32;"
    <<"fn valid_output()->bool{return m0_shared_extent_valid!=0u;}\n";
  s.MainFunctionBody()<<"if(local_idx==0u){m0_shared_extent_valid=select(0u,1u,validate_output_descriptor());}workgroupBarrier();";
}
inline void ValidOutput(ShaderHelper& s, const ShaderVariableHelper& e,bool share_workgroup=false) {
  const bool shared=share_workgroup&&M0BatchHeavyProgramsV1();
  auto& f = s.AdditionalImplementation();
  f << "fn " << (shared?"validate_output_descriptor":"valid_output") << "()->bool {if(" << e.GetByOffset("0u") << "!=2u||"
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
  if(shared){
    // All lanes enter before any payload tail return. The read-only sidecar
    // cannot change within a dispatch; validate all words once per workgroup.
    ShareOutputValidityV1(s);
  }
}
// Preserve the original scalar operation order across the staged GPU programs.
inline void OriginalArithmetic(ShaderHelper& s) {
    s.AdditionalImplementation() << R"WGSL(
fn original_f32_from_bits(bits:u32)->f32 {return bitcast<f32>(bits);}
fn original_mul(a:f32,b:f32)->f32 {
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0x7f800000u)||(bb==0u&&aa==0x7f800000u)){
 return original_f32_from_bits(0x7fc00000u);}
 if(aa==0x7f800000u||bb==0x7f800000u){return bitcast<f32>(((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)|0x7f800000u);}
 return a*b;
}
fn original_add(a:f32,b:f32)->f32 {
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0x7f800000u&&bb==0x7f800000u&&((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)!=0u)){
 return original_f32_from_bits(0x7fc00000u);}
 if(aa==0x7f800000u){return a;}if(bb==0x7f800000u){return b;}return a+b;
}
fn original_sub(a:f32,b:f32)->f32{return original_add(a,-b);}
fn original_sqrt(x:f32)->f32 {
 let bits=bitcast<u32>(x);let magnitude=bits&0x7fffffffu;
 if(magnitude>0x7f800000u||((bits&0x80000000u)!=0u&&magnitude!=0u)){return original_f32_from_bits(0x7fc00000u);}
 if(magnitude==0x7f800000u||magnitude==0u){return x;}return sqrt(x);
}
fn original_div(a:f32,b:f32)->f32 {
 let ua=bitcast<u32>(a);let ub=bitcast<u32>(b);let aa=ua&0x7fffffffu;let bb=ub&0x7fffffffu;
 let sign=(ua^ub)&0x80000000u;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0u)||(aa==0x7f800000u&&bb==0x7f800000u)){return original_f32_from_bits(0x7fc00000u);}
 if(bb==0u||aa==0x7f800000u){return bitcast<f32>(sign|0x7f800000u);}
 if(aa==0u||bb==0x7f800000u){return bitcast<f32>(sign);}return a/b;
}
)WGSL";
}
// Authenticate an allocation before asking Dawn about its backing buffer.
inline Status PhysicalBuffer(ComputeContext& c, const Tensor* tensor) {
  M0FullGpuStorageAllocationV1 allocation;
  ORT_RETURN_IF_NOT(tensor&&c.M0FullGpuDescribeStorageV1(*tensor,allocation),
      "RW07 residual stage live physical storage allocation");
  return Status::OK();
}
// Match ProgramManager's actual segment calculation, including each output.
// All these programs use distinct buffers and ordinary Get/SetByOffset access.
inline Status StorageBudget(ComputeContext& c, std::initializer_list<const Tensor*> tensors) {
  const uint64_t binding=c.DeviceLimits().maxStorageBufferBindingSize;
  ORT_RETURN_IF_NOT(binding>0,"RW07 residual stage zero storage binding limit");
  uint64_t segments=0;
  for(const auto* tensor:tensors){
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c,tensor));
    segments+=1+(tensor->SizeInBytes()-1)/binding;
  }
  ORT_RETURN_IF_NOT(segments<=c.DeviceLimits().maxStorageBuffersPerShaderStage,
      "RW07 residual stage segmented storage binding budget");
  return Status::OK();
}
// The activation window is a scheduling policy, not an assumed device size.
// Larger windows remove repeated input-channel passes while each convolution
// dispatch still obeys its independent product and aggregate submission bounds.
inline uint32_t ResidualTileChannelCountV1(const wgpu::Limits& limits,uint32_t input_capacity,
    uint32_t output_capacity,uint32_t channels,bool batched){
  const uint64_t alignment=limits.minStorageBufferOffsetAlignment;
  const uint64_t capacity=std::max(input_capacity,output_capacity);
  if(!alignment||alignment%sizeof(float)||!capacity||!channels||capacity*channels>UINT32_MAX)return 0;
  const uint64_t row_bytes=capacity*sizeof(float),prefix=alignment-sizeof(float);
  const uint64_t physical=std::min(limits.maxStorageBufferBindingSize,limits.maxBufferSize);
  if(row_bytes>physical||prefix>physical-row_bytes)return 0;
  // The admitted working-set plan already reserves a full activation carrier.
  // A queue allowance bounds retained preceding commands, not this allocation:
  // an oversized single command drains and completes immediately. Subtracting
  // its fixed references here collapsed valid full-width tiles to one channel.
  return static_cast<uint32_t>(std::min({uint64_t(channels),uint64_t(batched?256u:32u),
    uint64_t(limits.maxComputeWorkgroupsPerDimension),(physical-prefix)/row_bytes}));
}
inline Status ResidualTileChannels(ComputeContext& c,uint32_t input_capacity,uint32_t output_capacity,
    uint32_t channels,uint32_t& tile_channels,bool wide_residual=false){
  const auto count=ResidualTileChannelCountV1(c.DeviceLimits(),input_capacity,output_capacity,channels,
    wide_residual&&M0BatchHeavyProgramsV1()&&M0OutputChannelTiledConvV1());
  ORT_RETURN_IF_NOT(count>0,"RW07 residual channel window descriptor/device/index limit");
  tile_channels=count;
  return Status::OK();
}
template <typename View>
inline Status ValidatePhysicalWindow(ComputeContext& c, const View& window) {
  ORT_RETURN_IF_ERROR(PhysicalBuffer(c,window.tensor));
  const auto& limits=c.DeviceLimits();
  const uint64_t alignment=limits.minStorageBufferOffsetAlignment,binding=limits.maxStorageBufferBindingSize;
  const uint64_t offset=window.buffer_binding_offset_bytes,bytes=window.buffer_binding_size_bytes;
  const uint64_t prefix=window.buffer_offset_in_elements;
  ORT_RETURN_IF_NOT(window.tensor->template IsDataType<float>()&&window.var_type==ProgramVariableDataType::Float32&&
      window.is_buffer_view&&window.use_override_shape&&window.override_shape.Size()>0&&
      alignment>0&&alignment%sizeof(float)==0&&offset%alignment==0&&prefix<alignment/sizeof(float),
      "RW07 residual invalid physical window metadata");
  const uint64_t count=static_cast<uint64_t>(window.override_shape.Size()),backing=window.tensor->SizeInBytes();
  ORT_RETURN_IF_NOT(count<=UINT32_MAX-prefix&&bytes==(prefix+count)*sizeof(float)&&bytes>0&&
      bytes<=binding&&offset<=backing&&bytes<=backing-offset,"RW07 residual physical window range");
  return Status::OK();
}
inline Status WindowStorageBudget(ComputeContext& c, uint64_t window_bindings,
                                  std::initializer_list<const Tensor*> tensors) {
  const auto& limits=c.DeviceLimits();
  const uint64_t binding=limits.maxStorageBufferBindingSize;
  ORT_RETURN_IF_NOT(binding>0,"RW07 residual zero window binding limit");
  uint64_t segments=window_bindings;
  for(const auto* tensor:tensors){
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c,tensor));
    segments+=1+(tensor->SizeInBytes()-1)/binding;
  }
  ORT_RETURN_IF_NOT(segments<=limits.maxStorageBuffersPerShaderStage,
      "RW07 residual window storage binding budget");
  return Status::OK();
}
// Authenticate complete allocations, but count the actual physical windows.
inline Status StorageBudgetWithWindow(ComputeContext& c, const ProgramInput& window,
                                      std::initializer_list<const Tensor*> tensors) {
  ORT_RETURN_IF_ERROR(ValidatePhysicalWindow(c,window));
  return WindowStorageBudget(c,1,tensors);
}
inline Status StorageBudgetWithWindow(ComputeContext& c, const ProgramOutput& window,
                                      std::initializer_list<const Tensor*> tensors) {
  ORT_RETURN_IF_NOT(!window.is_atomic,"RW07 residual atomic output window");
  ORT_RETURN_IF_ERROR(ValidatePhysicalWindow(c,window));
  for(const auto* tensor:tensors) ORT_RETURN_IF_NOT(tensor&&tensor->DataRaw()!=window.tensor->DataRaw(),
      "RW07 residual output window alias");
  return WindowStorageBudget(c,1,tensors);
}
inline Status StorageBudgetWithWindows(ComputeContext& c, std::initializer_list<ProgramInput> inputs,
                                       const ProgramOutput& output, std::initializer_list<const Tensor*> tensors) {
  ORT_RETURN_IF_NOT(!output.is_atomic,"RW07 residual atomic output window");
  ORT_RETURN_IF_ERROR(ValidatePhysicalWindow(c,output));
  for(const auto& input:inputs){
    ORT_RETURN_IF_ERROR(ValidatePhysicalWindow(c,input));
    ORT_RETURN_IF_NOT(input.tensor->DataRaw()!=output.tensor->DataRaw(),"RW07 residual input/output window alias");
  }
  for(const auto* tensor:tensors) ORT_RETURN_IF_NOT(tensor&&tensor->DataRaw()!=output.tensor->DataRaw(),
      "RW07 residual output window alias");
  return WindowStorageBudget(c,1+inputs.size(),tensors);
}
inline Status StorageBudgetWithWindows(ComputeContext& c, const ProgramInput& input,
                                       const ProgramOutput& output, std::initializer_list<const Tensor*> tensors) {
  return StorageBudgetWithWindows(c,std::initializer_list<ProgramInput>{input},output,tensors);
}
class GeneratorStyleParameters final : public Program<GeneratorStyleParameters> {
 public:
  GeneratorStyleParameters() : Program{"RW07ResidualStyleParametersAndStatsV2"} {}
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
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& style=s.AddInput("style",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("fc_weight",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("fc_bias",ShaderUsage::UseValueTypeAlias);
    const auto& alpha=s.AddInput("alpha",ShaderUsage::UseValueTypeAlias);
    const auto& reciprocal=s.AddInput("reciprocal",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("coefficients",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e);
    OriginalArithmetic(s);
    auto& f=s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels){return;}let channel=global_idx;"
      << "for(var i=0u;i<6u;i++){" << y.SetByOffset("channel*6u+i","0.0") << "}"
      << "if(!valid_output()){return;}"
      << "var gs:f32=0.0;var bs:f32=0.0;for(var k=0u;k<128u;k++){let value=" << style.GetByOffset("k") << ";"
      << "gs=original_add(gs,original_mul(value," << w.GetByOffset("channel*128u+k") << "));"
      << "bs=original_add(bs,original_mul(value," << w.GetByOffset("(channel+uniforms.channels)*128u+k") << "));}"
      << "let gamma=original_add(original_add(gs," << b.GetByOffset("channel") << "),1.0);"
      << "let beta=original_add(bs," << b.GetByOffset("channel+uniforms.channels") << ");"
      << "let a=" << alpha.GetByOffset("channel") << ";let reciprocal_a=" << reciprocal.GetByOffset("channel") << ";";
    f << y.SetByOffset("channel*6u","gamma")
      << y.SetByOffset("channel*6u+1u","beta")
      << y.SetByOffset("channel*6u+2u","a")
      << y.SetByOffset("channel*6u+3u","reciprocal_a");
    return Status::OK();
  }
};
// Compensated FP32 sums avoid linear rounding drift on long channels.
// Mean/variance stay separate original graph stages; no per-sample trig loop.
// The coefficient allocation also holds mean/denominator; no carrier copy.
class GeneratorNormStatistics final : public Program<GeneratorNormStatistics> {
 public:
  explicit GeneratorNormStatistics(bool parallel=M0ParallelNormV1())
    : Program{parallel?"RW07ResidualNormStatisticsParallelCompensatedV3":"RW07ResidualNormStatisticsCompensatedV2"},parallel_(parallel) {}
  uint32_t Workgroups(uint32_t channels) const {return parallel_?channels:(channels+63)/64;}
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
      {"channel_offset", ProgramUniformVariableDataType::Uint32},
      {"channel_count", ProgramUniformVariableDataType::Uint32},
      {"tile_channels", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("carrier",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("coefficients",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e);
    OriginalArithmetic(s);
    s.AdditionalImplementation() << R"WGSL(
fn residual_accumulate(state:vec2<f32>,value:f32)->vec2<f32> {
 let adjusted=original_sub(value,state.y);let next=original_add(state.x,adjusted);
 // Preserve ordinary NaN/Inf/overflow propagation instead of compensating it.
 if((bitcast<u32>(state.x)&0x7fffffffu)>=0x7f800000u||
    (bitcast<u32>(value)&0x7fffffffu)>=0x7f800000u||
    (bitcast<u32>(next)&0x7fffffffu)>=0x7f800000u){
   return vec2<f32>(original_add(state.x,value),0.0);
 }
 return vec2<f32>(next,original_sub(original_sub(next,state.x),adjusted));
}
)WGSL";
    if(parallel_){
      s.AdditionalImplementation()<<"var<workgroup> norm_partial:array<f32,64>;var<workgroup> norm_unsafe:array<u32,64>;var<workgroup> norm_mean:f32;var<workgroup> norm_count:u32;";
      auto& f=s.MainFunctionBody();
      // All lanes execute every barrier, including invalid extents. Each group
      // owns one channel; no global atomics or cross-group coordination.
      f << "let local_channel=workgroup_idx;let channel=uniforms.channel_offset+local_channel;"
        << "let channel_valid=local_channel<uniforms.channel_count&&uniforms.channel_offset<uniforms.channels&&"
        << "uniforms.channel_count<=uniforms.channels-uniforms.channel_offset;"
        << "if(local_idx==0u){norm_count=0u;norm_mean=0.0;if(channel_valid){"
        << y.SetByOffset("channel*6u+4u","0.0") << y.SetByOffset("channel*6u+5u","0.0")
        << "let count=" << e.GetByOffset("6u") << ";if(valid_output()&&count<=uniforms.input_capacity){norm_count=count;}}}"
        << "workgroupBarrier();var sum=vec2<f32>(0.0);var row_requires_serial=0u;"
        // A conservative finite bound excludes overflow in both the sum and
        // squared-deviation reduction. Exceptional rows retain serial order.
        << "let safe_abs=sqrt(bitcast<f32>(0x7f7fffffu)/f32(max(1u,norm_count)))*0.25;"
        << "for(var t=local_idx;t<norm_count;t+=64u){let value=" << x.GetByOffset("local_channel*uniforms.input_capacity+t") << ";"
        << "if((bitcast<u32>(value)&0x7fffffffu)>=0x7f800000u||abs(value)>safe_abs){row_requires_serial=1u;}sum=residual_accumulate(sum,value);}"
        << "norm_partial[local_idx]=sum.x;norm_unsafe[local_idx]=row_requires_serial;workgroupBarrier();"
        << "if(local_idx==0u&&norm_count>0u){var serial=false;for(var lane=0u;lane<64u;lane++){serial=serial||norm_unsafe[lane]!=0u;}"
        << "if(serial){var exact_sum=vec2<f32>(0.0);for(var t=0u;t<norm_count;t++){exact_sum=residual_accumulate(exact_sum,"
        << x.GetByOffset("local_channel*uniforms.input_capacity+t") << ");}let mean=original_div(exact_sum.x,f32(norm_count));"
        << "var exact_squares=vec2<f32>(0.0);for(var t=0u;t<norm_count;t++){let delta=original_sub("
        << x.GetByOffset("local_channel*uniforms.input_capacity+t") << ",mean);exact_squares=residual_accumulate(exact_squares,original_mul(delta,delta));}"
        << "let variance=original_div(exact_squares.x,f32(norm_count));let denominator=original_sqrt(original_add(variance,bitcast<f32>(0x3727c5acu)));"
        << y.SetByOffset("channel*6u+4u","mean") << y.SetByOffset("channel*6u+5u","denominator")
        << "norm_count=0u;}else{var combined=vec2<f32>(0.0);for(var lane=0u;lane<64u;lane++){"
        << "combined=residual_accumulate(combined,norm_partial[lane]);}norm_mean=original_div(combined.x,f32(norm_count));}}"
        << "workgroupBarrier();var squares=vec2<f32>(0.0);"
        << "for(var t=local_idx;t<norm_count;t+=64u){let delta=original_sub(" << x.GetByOffset("local_channel*uniforms.input_capacity+t") << ",norm_mean);"
        << "squares=residual_accumulate(squares,original_mul(delta,delta));}"
        << "norm_partial[local_idx]=squares.x;workgroupBarrier();"
        << "if(local_idx==0u&&norm_count>0u){var combined=vec2<f32>(0.0);for(var lane=0u;lane<64u;lane++){"
        << "combined=residual_accumulate(combined,norm_partial[lane]);}let variance=original_div(combined.x,f32(norm_count));"
        << "let denominator=original_sqrt(original_add(variance,bitcast<f32>(0x3727c5acu)));"
        << y.SetByOffset("channel*6u+4u","norm_mean") << y.SetByOffset("channel*6u+5u","denominator") << "}";
      return Status::OK();
    }
    auto& f=s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channel_count||uniforms.channel_offset>=uniforms.channels||"
      << "uniforms.channel_count>uniforms.channels-uniforms.channel_offset){return;}"
      << "let local_channel=global_idx;let channel=uniforms.channel_offset+local_channel;"
      << y.SetByOffset("channel*6u+4u","0.0") << y.SetByOffset("channel*6u+5u","0.0")
      << "if(!valid_output()){return;}let count=" << e.GetByOffset("6u")
      << ";if(count>uniforms.input_capacity){return;}var sum=vec2<f32>(0.0,0.0);"
      << "for(var t=0u;t<count;t++){sum=residual_accumulate(sum," << x.GetByOffset("local_channel*uniforms.input_capacity+t") << ");}"
      << "let mean=original_div(sum.x,f32(count));var squares=vec2<f32>(0.0,0.0);"
      << "for(var t=0u;t<count;t++){let delta=original_sub(" << x.GetByOffset("local_channel*uniforms.input_capacity+t") << ",mean);"
      << "squares=residual_accumulate(squares,original_mul(delta,delta));}"
      << "let variance=original_div(squares.x,f32(count));let denominator=original_sqrt(original_add(variance,bitcast<f32>(0x3727c5acu)));"
      << y.SetByOffset("channel*6u+4u","mean") << y.SetByOffset("channel*6u+5u","denominator");
    return Status::OK();
  }
 private:
  const bool parallel_;
};
// Each invocation evaluates one sample. Derive chunk size from its worst
// fused norm/Snake product cost within the existing dispatch envelope.
inline uint32_t ResidualSnakeElementsPerDispatchV1(bool batched=M0BatchHeavyProgramsV1()) {
  constexpr uint32_t lanes=64;
  return M0HeavyProductsPerDispatchV1(batched)/(kNormAffineWorkPerElementV1+kSnakeWorkPerElementV1)/lanes*lanes;
}
class GeneratorNormAffine final : public Program<GeneratorNormAffine> {
 public:
  explicit GeneratorNormAffine(bool fused=M0FusedNormSnakeV1())
    : Program{fused?"RW07ResidualNormAffineSnakeFusedV5":"RW07ResidualNormAffineElementChunksV4"},fused_(fused) {}
  bool FusedSnake() const {return fused_;}
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
      {"channel_offset", ProgramUniformVariableDataType::Uint32},
      {"channel_count", ProgramUniformVariableDataType::Uint32},
      {"tile_channels", ProgramUniformVariableDataType::Uint32},
      {"element_begin", ProgramUniformVariableDataType::Uint32},
      {"element_count", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("carrier",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& coefficients=s.AddInput("coefficients",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("activated",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e,true);
    OriginalArithmetic(s);
    if(fused_)s.AdditionalImplementation()<<m0_trig_detail::Trig;
    auto& f=s.MainFunctionBody();
    f << "let elements=uniforms.tile_channels*uniforms.output_capacity;"
      << "if(uniforms.element_begin>=elements||uniforms.element_count>elements-uniforms.element_begin||"
      << "global_idx>=uniforms.element_count){return;}let at=uniforms.element_begin+global_idx;"
      << y.SetByOffset("at","0.0") << "if(!valid_output()){return;}"
      << "let local_channel=at/uniforms.output_capacity;let t=at%uniforms.output_capacity;"
      << "if(local_channel>=uniforms.channel_count||uniforms.channel_offset>=uniforms.channels||"
      << "uniforms.channel_count>uniforms.channels-uniforms.channel_offset){return;}"
      << "let channel=uniforms.channel_offset+local_channel;let count=" << e.GetByOffset("6u")
      << ";if(count>uniforms.input_capacity||t>=count){return;}"
      << "let mean=" << coefficients.GetByOffset("channel*6u+4u") << ";"
      << "let denominator=" << coefficients.GetByOffset("channel*6u+5u") << ";"
      << "let gamma=" << coefficients.GetByOffset("channel*6u") << ";"
      << "let beta=" << coefficients.GetByOffset("channel*6u+1u") << ";"
      << "let centered=original_sub(" << x.GetByOffset("local_channel*uniforms.input_capacity+t") << ",mean);"
      << "let normalized=original_div(centered,denominator);let affine=original_add(original_mul(gamma,normalized),beta);"
      ;
    if(fused_){
      f << "let a=" << coefficients.GetByOffset("channel*6u+2u") << ";"
        << "let reciprocal_a=" << coefficients.GetByOffset("channel*6u+3u") << ";"
        << "let angle=original_mul(a,affine);let bits=bitcast<u32>(angle)&0x7fffffffu;var sine:f32=original_f32_from_bits(0x7fc00000u);"
        << "if(bits<0x7f800000u){sine=rw07_sin(angle);}let square=original_mul(sine,sine);"
        << "let result=original_add(affine,original_mul(reciprocal_a,square));" << y.SetByOffset("at","result");
    }else{f << y.SetByOffset("at","affine");}

    return Status::OK();
  }
 private:
  const bool fused_;
};
// Keep only affine/a/reciprocal live around the trig calculation. Each
// invocation reads and overwrites its own element in the existing tile.
class GeneratorSnakeInPlace final : public Program<GeneratorSnakeInPlace> {
 public:
  GeneratorSnakeInPlace() : Program{"RW07ResidualSnakeInPlaceElementChunksV1"} {}
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
      {"channel_offset", ProgramUniformVariableDataType::Uint32},
      {"channel_count", ProgramUniformVariableDataType::Uint32},
      {"tile_channels", ProgramUniformVariableDataType::Uint32},
      {"element_begin", ProgramUniformVariableDataType::Uint32},
      {"element_count", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& coefficients=s.AddInput("coefficients",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("activated",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e,true);
    OriginalArithmetic(s);
    s.AdditionalImplementation() << m0_trig_detail::Trig;
    auto& f=s.MainFunctionBody();
    f << "let elements=uniforms.tile_channels*uniforms.output_capacity;"
      << "if(uniforms.element_begin>=elements||uniforms.element_count>elements-uniforms.element_begin||"
      << "global_idx>=uniforms.element_count){return;}let at=uniforms.element_begin+global_idx;"
      << "if(!valid_output()){ " << y.SetByOffset("at","0.0") << "return;}"
      << "let local_channel=at/uniforms.output_capacity;let t=at%uniforms.output_capacity;"
      << "let count=" << e.GetByOffset("6u") << ";"
      << "if(local_channel>=uniforms.channel_count||uniforms.channel_offset>=uniforms.channels||"
      << "uniforms.channel_count>uniforms.channels-uniforms.channel_offset||"
      << "count>uniforms.input_capacity||t>=count){" << y.SetByOffset("at","0.0") << "return;}"
      << "let channel=uniforms.channel_offset+local_channel;let affine=" << y.GetByOffset("at") << ";"
      << "let a=" << coefficients.GetByOffset("channel*6u+2u") << ";"
      << "let reciprocal_a=" << coefficients.GetByOffset("channel*6u+3u") << ";"
      << "let angle=original_mul(a,affine);let bits=bitcast<u32>(angle)&0x7fffffffu;var sine:f32=original_f32_from_bits(0x7fc00000u);"
      << "if(bits<0x7f800000u){sine=rw07_sin(angle);}let square=original_mul(sine,sine);"
      << "let result=original_add(affine,original_mul(reciprocal_a,square));"
      << y.SetByOffset("at","result");
    return Status::OK();
  }
};
// Bound scalar products encoded by one residual convolution dispatch. This
// is a scheduling upper bound, not measured GPU instructions, FLOPs or time.
inline constexpr uint32_t kResidualProductsPerDispatchV1 = kM0HeavyProductsPerDispatchV1;
inline uint32_t ResidualProductsPerDispatchV1(bool batched=M0BatchHeavyProgramsV1()) {
  return M0HeavyProductsPerDispatchV1(batched);
}
inline uint32_t ResidualConvElementsPerDispatchV1(uint32_t input_channels, uint32_t kernel,
                                                bool batched=M0BatchHeavyProgramsV1()) {
  const uint64_t products_per_element = uint64_t(input_channels) * kernel;
  const uint32_t budget=ResidualProductsPerDispatchV1(batched);
  if (products_per_element == 0 || products_per_element > budget) return 0;
  const auto elements = static_cast<uint32_t>(budget / products_per_element);
  return elements >= 64 ? elements / 64 * 64 : elements;
}
inline constexpr uint64_t ResidualChannelTileStorageV1(uint32_t block) {
  return uint64_t(block)*680+320;  // includes shared descriptor validity and alignment slack
}
inline uint32_t ResidualChannelBlockV1(uint64_t workgroup_bytes) {
  for(uint32_t block:{8u,4u,2u,1u})if(ResidualChannelTileStorageV1(block)<=workgroup_bytes)return block;
  return 0;
}
inline uint32_t ResidualBestBlockV1(const std::array<double,7>& scores,uint32_t maximum) {
  uint32_t chosen=0;double best=std::numeric_limits<double>::infinity();
  for(size_t i=0;i<scores.size();++i){
    const uint32_t block=1u<<i;
    if(block<=maximum&&scores[i]>0&&(!chosen||scores[i]<best*0.95)){chosen=block;best=scores[i];}
  }
  return chosen?chosen:1;
}
// Only an initial input-channel pass overwrites its output. Trials must never
// repeat a pass that reads a preceding partial sum from that output.
template <typename Run>
inline Status ResidualAutotunedBlockV1(ComputeContextBase& context,const std::string& key,
    uint32_t maximum,bool overwrite_pass,Run&& run,uint32_t& chosen) {
  ORT_RETURN_IF_NOT(maximum==1||maximum==2||maximum==4||maximum==8||maximum==16||maximum==32||maximum==64,"RW07 residual tuning capacity");
  chosen=context.M0ResidualBlockChoiceV1(key);
  if(chosen>maximum)chosen=0;
  if(chosen)return Status::OK();
  chosen=1;
  if(!overwrite_pass||maximum==1)return Status::OK();
  // First compare execution families. Refine channel-block variants only if
  // that family wins the coarse comparison. This is a measured search
  // heuristic, not a proof that the omitted variants could never be faster.
  const uint32_t legacy=std::min(maximum,std::max(1u,ResidualChannelBlockV1(context.DeviceLimits().maxComputeWorkgroupStorageSize)));
  std::array<double,7> scores{};std::array<bool,7> candidates{};
  candidates[std::bit_width(legacy)-1]=true;
  for(uint32_t block=16;block<=maximum;block*=2)candidates[std::bit_width(block)-1]=true;
  const auto measure=[&](const std::array<bool,7>& selected)->Status{
    for(size_t i=0;i<selected.size();++i)if(selected[i]){scores[i]=0;ORT_RETURN_IF_ERROR(run(1u<<i));}
    for(uint32_t round=0;round<2;++round)for(uint32_t i=0;i<7;++i){
      const uint32_t index=round?6u-i:i;if(!selected[index])continue;
      const auto start=std::chrono::steady_clock::now();ORT_RETURN_IF_ERROR(run(1u<<index));
      scores[index]+=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
    }
    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(measure(candidates));chosen=ResidualBestBlockV1(scores,maximum);
  if(chosen<=legacy&&legacy>1){
    candidates.fill(false);for(uint32_t block=1;block<=legacy;block*=2)candidates[std::bit_width(block)-1]=true;
    ORT_RETURN_IF_ERROR(measure(candidates));chosen=ResidualBestBlockV1(scores,maximum);
  }
  context.M0RememberResidualBlockV1(key,chosen);
  LOGS(context.Logger(),INFO)<<"M0_RW07_RESIDUAL_BLOCK_TUNING key="<<key<<" selected="<<chosen
    <<" max_supported="<<maximum<<" samples_ns="<<scores[0]<<","<<scores[1]<<","<<scores[2]<<","<<scores[3]<<","<<scores[4]<<","<<scores[5]<<","<<scores[6];
  return Status::OK();
}
inline constexpr uint64_t kResidualMatrixStorageV1=5760;
inline uint32_t ResidualVariantLimitV1(uint64_t bytes) {
  return bytes>=kResidualMatrixStorageV1?16u:ResidualChannelBlockV1(bytes);
}
inline bool ResidualMatrixSupportedV1(const wgpu::Limits& limits) {
  return limits.maxComputeWorkgroupStorageSize>=kResidualMatrixStorageV1&&
    limits.maxComputeWorkgroupSizeX>=8&&limits.maxComputeWorkgroupSizeY>=8&&
    limits.maxComputeWorkgroupSizeZ>=1&&limits.maxComputeInvocationsPerWorkgroup>=64;
}
// Private uniform ABI shared by packing, matrix compute and baseline slices.
#define M0_RESIDUAL_TILE_UNIFORMS_V1 \
 {"input_capacity",ProgramUniformVariableDataType::Uint32}, \
 {"output_capacity",ProgramUniformVariableDataType::Uint32}, \
 {"channels",ProgramUniformVariableDataType::Uint32}, \
 {"minimum",ProgramUniformVariableDataType::Uint32}, \
 {"maximum",ProgramUniformVariableDataType::Uint32}, \
 {"modulus",ProgramUniformVariableDataType::Uint32}, \
 {"remainder",ProgramUniformVariableDataType::Uint32}, \
 {"kernel",ProgramUniformVariableDataType::Uint32}, \
 {"dilation",ProgramUniformVariableDataType::Uint32}, \
 {"residual_capacity",ProgramUniformVariableDataType::Uint32}, \
 {"channel_offset",ProgramUniformVariableDataType::Uint32}, \
 {"channel_count",ProgramUniformVariableDataType::Uint32}, \
 {"output_channel_offset",ProgramUniformVariableDataType::Uint32}, \
 {"output_channel_count",ProgramUniformVariableDataType::Uint32}, \
 {"element_begin",ProgramUniformVariableDataType::Uint32}, \
 {"element_count",ProgramUniformVariableDataType::Uint32}, \
 {"identity",ProgramUniformVariableDataType::Uint32}, \
 {"time_begin",ProgramUniformVariableDataType::Uint32}, \
 {"time_count",ProgramUniformVariableDataType::Uint32}

struct ResidualConvSpecV1 {
  uint32_t input_capacity,output_capacity,channels,minimum,maximum,modulus,remainder,kernel,dilation,residual_capacity;
  uint32_t channel_offset,channel_count,output_channel_offset,output_channel_count;
  gsl::span<const uint32_t> identity;
  template <typename P> void Apply(P& p,uint32_t t,uint32_t n,uint32_t first,uint32_t count)const {
    p.AddUniformVariables({input_capacity,output_capacity,channels,minimum,maximum,modulus,remainder,kernel,dilation,
      residual_capacity,channel_offset,channel_count,output_channel_offset,output_channel_count,first,count,identity,t,n});
  }
  template <typename P> void ApplyPacked(P& p,uint32_t t,uint32_t n,uint32_t row,uint32_t rows,uint32_t flag_offset=0)const {
    p.AddUniformVariables({input_capacity,output_capacity,channels,minimum,maximum,modulus,remainder,kernel,dilation,
      residual_capacity,channel_offset,channel_count,output_channel_offset,output_channel_count,0u,0u,identity,t,n,row,rows,flag_offset});
  }
};
inline constexpr uint32_t kResidualPackedTimeV1=2048;
inline constexpr uint64_t kResidualPackedScratchBytesV1=kM0ResidualPackedScratchBytesV1;
inline uint32_t ResidualPackedProductsPerDispatchV1(bool batched=M0BatchHeavyProgramsV1()) {
  return batched?(1u<<28):ResidualProductsPerDispatchV1(false);
}
inline uint32_t ResidualPackedInnerV1(uint32_t channels,uint32_t kernel){
  if(channels==0||channels>256||kernel==0||kernel>11)return 0;
  return (channels*kernel+3u)/4u*4u;
}
inline uint64_t ResidualPackedElementsV1(uint32_t channels,uint32_t kernel) {
  const uint64_t inner=ResidualPackedInnerV1(channels,kernel);
  return inner?std::min(inner*kResidualPackedTimeV1,kResidualPackedScratchBytesV1/4):0;
}
// Eight inner rows by32 time positions per flag. Padding at most seven
// inner rows adds at most56 words across the bounded2048-position window.
// The final16 words hold one weight-safety flag per32 output channels.
inline uint64_t ResidualPackedFlagCapacityV1(uint64_t column_elements){
  if(!column_elements||column_elements>kResidualPackedScratchBytesV1/sizeof(float))return 0;
  return (column_elements+255)/256+kResidualPackedTimeV1/32+kM0ResidualPackedWeightFlagWordsV1;
}
inline uint64_t ResidualPackedFlagCountV1(uint32_t channels,uint32_t kernel,uint32_t time_count){
  if(!channels||channels>256||!kernel||kernel>11||!time_count||time_count>kResidualPackedTimeV1||time_count%32)return 0;
  return ((uint64_t(channels)*kernel+7)/8)*(time_count/32);
}
inline constexpr uint64_t kResidualPackedStorageV1=8512;
// Match the existing 8x8 GEMM workgroup and register8 output reuse. Half operands
// keep this 64x32 tile below the already-qualified shared-memory envelope.
inline constexpr uint32_t kResidualMixedRowsPerThreadV1=8;
inline constexpr uint32_t kResidualMixedTileRowsV1=8*kResidualMixedRowsPerThreadV1;

inline uint32_t ResidualPackedTimeV1(uint32_t channels,uint32_t kernel,uint32_t budget=ResidualPackedProductsPerDispatchV1()) {
  const uint64_t inner=ResidualPackedInnerV1(channels,kernel);
  if(!inner)return 0;
  const uint64_t rounded=(inner+31)/32*32;
  const auto count=std::min({uint64_t(kResidualPackedTimeV1),uint64_t(budget)/(rounded*32),kResidualPackedScratchBytesV1/(inner*4)});
  return static_cast<uint32_t>(count/32*32);
}
inline uint64_t ResidualRegisterWeightElementsV1(uint32_t outputs,uint32_t inputs,uint32_t kernel){
  if(!outputs||outputs>512||!inputs||inputs>256||!kernel||kernel>11)return 0;
  const uint64_t rows=(uint64_t(outputs)+63)/64*64,inner=(uint64_t(inputs)*kernel+3)/4;
  const uint64_t elements=rows*inner*4;
  return elements<=kM0ResidualRegisterWeightScratchBytesV1/sizeof(float)?elements:0;
}
inline bool ResidualPackedSupportedV1(const wgpu::Limits& limits) {
  return limits.maxComputeWorkgroupStorageSize>=kResidualPackedStorageV1&&limits.maxComputeWorkgroupSizeX>=8&&
    limits.maxComputeWorkgroupSizeY>=8&&limits.maxComputeInvocationsPerWorkgroup>=64&&
    limits.maxStorageBuffersPerShaderStage>=7&&limits.maxBindingsPerBindGroup>=8;
}

// WGSL permits either adjacent f16 result for an inexact conversion. Choose
// IEEE round-to-nearest/ties-to-even once during packing, independently of the
// driver rounding mode. The final bitcast introduces no additional rounding.
inline void ResidualHalfConversionV1(ShaderHelper& s) {
  s.AdditionalImplementation()<<R"(
const residual_half_max:f32=65504.0;
fn residual_half_bits(value:f32)->u32 {
  let raw=bitcast<u32>(value);let sign=(raw>>16u)&0x8000u;let mag=raw&0x7fffffffu;
  if(mag>0x477fe000u){return 0u;}
  // Binary16 normal exponent floor 2^-14; discard 23-10 significand bits.
  if(mag>=0x38800000u){return sign|(((mag+0xfffu+((mag>>13u)&1u))>>13u)-0x1c000u);}
  // Values below half of the least binary16 subnormal round to signed zero.
  if(mag<0x33000000u){return sign;}
  let shift=126u-(mag>>23u);let mantissa=(mag&0x7fffffu)|0x800000u;
  let kept=mantissa>>shift;let discarded=mantissa&((1u<<shift)-1u);let halfway=1u<<(shift-1u);
  let increment=select(0u,1u,discarded>halfway||(discarded==halfway&&(kept&1u)!=0u));
  return sign|(kept+increment);
}
fn residual_half4(value:vec4<f32>)->vec4<f16> {
  let x=residual_half_bits(value.x);let y=residual_half_bits(value.y);
  let z=residual_half_bits(value.z);let w=residual_half_bits(value.w);
  return bitcast<vec4<f16>>(vec2u(x|(y<<16u),z|(w<<16u)));
}
)";
}

// Physical weights: [row block64][K4][64rows][4K]. Physical columns:
// [K4][time block32][4K][32times]. A group reads compact adjacent K/row/time
// tiles, rather than striding across all padded rows or the whole time window.
// Per invocation/window GPU preparation. No cross-Run cache or CPU weight access.
class ResidualPackedWeightFlagsV1 final : public Program<ResidualPackedWeightFlagsV1> {
 public:
  explicit ResidualPackedWeightFlagsV1(bool pack=false,bool mixed=false)
      :Program{mixed?"RW07ResidualPackHalfWeightsRneV2":pack?"RW07ResidualPackWeightsAndFlagsV5":"RW07ResidualPackedWeightFlagsV1"},pack_weights_(pack),mixed_(mixed){}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(M0_RESIDUAL_TILE_UNIFORMS_V1,
    {"row_begin",ProgramUniformVariableDataType::Uint32},{"row_count",ProgramUniformVariableDataType::Uint32},
    {"weight_flag_offset",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override {
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto* packed=pack_weights_?&s.AddOutput("packed_weight",ShaderUsage::UseValueTypeAlias):nullptr;
    const auto& flags=s.AddOutput("flags",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e,true);
    if(mixed_)ResidualHalfConversionV1(s);
    s.AdditionalImplementation()<<"var<workgroup> weight_unsafe:array<u32,64>;";
    auto& f=s.MainFunctionBody();
    if(pack_weights_){
      f<<"let row_base=workgroup_idx*32u;if(row_base>=uniforms.row_count){return;}"
        <<"let inner=uniforms.channel_count*uniforms.kernel;let vectors=(inner+3u)/4u;"
        <<"var bad=select(1u,0u,valid_output());"
        <<"for(var index=local_idx;index<32u*vectors;index+=64u){let row=row_base+index%32u;let k=index/32u;"
        <<"var value=vec4<f32>(0.0);if(valid_output()&&row<uniforms.output_channel_count){"
        <<"let at=((uniforms.output_channel_offset+row)*uniforms.channels+uniforms.channel_offset)*uniforms.kernel+k*4u;"
        <<"for(var lane=0u;lane<4u;lane++){if(k*4u+lane<inner){value[lane]="
        <<w.GetByOffset("(at+lane)/4u")<<"[(at+lane)%4u];}}}"
        <<packed->SetByOffset("((row/64u)*vectors+k)*64u+row%64u",mixed_?
            "residual_half4(value)":"value")
        <<"if(any((bitcast<vec4<u32>>(value)&vec4u(0x7fffffffu))>vec4u("<<(mixed_?"bitcast<u32>(residual_half_max)":"0x58800000u")<<"))){bad=1u;}}";
    }else{
    f<<"let row_base=workgroup_idx*32u;if(row_base>=uniforms.output_channel_count){return;}"
      <<"let inner=uniforms.channel_count*uniforms.kernel;var bad=select(1u,0u,valid_output());"
      <<"for(var index=local_idx;index<32u*inner;index+=64u){let row=row_base+index/inner;"
      <<"if(row<uniforms.output_channel_count){let at=((uniforms.output_channel_offset+row)*uniforms.channels+"
      <<"uniforms.channel_offset)*uniforms.kernel+index%inner;let value="<<w.GetByOffset("at/4u")<<"[at%4u];"
      <<"bad=bad|select(0u,1u,(bitcast<u32>(value)&0x7fffffffu)>0x58800000u);}}";
    }
    f<<"weight_unsafe[local_idx]=bad;workgroupBarrier();if(local_idx==0u){var combined=0u;"
      <<"for(var lane=0u;lane<64u;lane++){combined|=weight_unsafe[lane];}"
      <<flags.SetByOffset("uniforms.weight_flag_offset+workgroup_idx","combined")<<"}";
    return Status::OK();
  }
 private:
  const bool pack_weights_,mixed_;
};
class ResidualPackColumnsV1 final : public Program<ResidualPackColumnsV1> {
 public:
  explicit ResidualPackColumnsV1(bool skip_inactive=false,bool mixed=false):Program{mixed?"RW07ResidualPackHalfColumnsRneV2":skip_inactive?"RW07ResidualPackActiveColumnsGridV7":"RW07ResidualPackColumnsGridV7"},skip_inactive_(skip_inactive),mixed_(mixed){}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(M0_RESIDUAL_TILE_UNIFORMS_V1,
    {"row_begin",ProgramUniformVariableDataType::Uint32},{"row_count",ProgramUniformVariableDataType::Uint32},
    {"weight_flag_offset",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override {
    const auto& x=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& col=s.AddOutput("columns",ShaderUsage::UseValueTypeAlias);
    const auto& flag=s.AddOutput("flags",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e,true);
    if(mixed_)ResidualHalfConversionV1(s);
    s.AdditionalImplementation()<<"var<workgroup> pack_flags:array<u32,64>;";
    if(skip_inactive_)s.AdditionalImplementation()<<"var<workgroup> skip_tail:u32;";
    auto& f=s.MainFunctionBody();
    f<<"if(uniforms.time_count<32u||uniforms.time_count>2048u||uniforms.time_count%32u!=0u||"
      <<"uniforms.channel_count==0u||uniforms.channel_count>256u||uniforms.kernel==0u||uniforms.kernel>11u){return;}"
      <<"let inner_count=uniforms.channel_count*uniforms.kernel;let time_tiles=uniforms.time_count/32u;"
      <<"let group_count=((inner_count+7u)/8u)*time_tiles;let physical_group="
      <<(IndirectDispatchTensor()?"workgroup_id.y*time_tiles+workgroup_id.x":"workgroup_idx")<<";"
      <<"if(physical_group>=group_count){return;}";
    if(skip_inactive_){
      f<<"if(local_idx==0u){skip_tail=select(0u,1u,valid_output()&&uniforms.time_begin+"
        <<"(physical_group%time_tiles)*32u>="<<e.GetByOffset("6u")<<");}"
        <<"let inactive=workgroupUniformLoad(&skip_tail);if(inactive!=0u){if(local_idx==0u){"
        <<flag.SetByOffset("physical_group","0u")<<"}return;}";
    }
    f<<"var value=vec4<f32>(0.0);"
      <<"let row=(physical_group/time_tiles)*8u+local_idx/8u;"
      <<"let local_t=(physical_group%time_tiles)*32u+(local_idx%8u)*4u;"
      <<"let valid=valid_output()&&uniforms.dilation>=1u&&uniforms.dilation<=5u;"
      <<"if(row<(inner_count+3u)/4u*4u){if(row<inner_count&&valid){let ic=row/uniforms.kernel;let k=row%uniforms.kernel;"
      <<"let n="<<e.GetByOffset("6u")<<";let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
      <<"for(var lane=0u;lane<4u;lane++){let t=uniforms.time_begin+local_t+lane;let pos=i32(t+k*uniforms.dilation)-pad;"
      <<"if(t<n&&pos>=0&&pos<i32(n)&&pos<i32(uniforms.input_capacity)){value[lane]="
      <<x.GetByOffset("ic*uniforms.input_capacity+u32(pos)")<<";}}}"
      <<col.SetByOffset("((row/4u)*time_tiles+local_t/32u)*32u+(row%4u)*8u+(local_t%32u)/4u",mixed_?
          "residual_half4(value)":"value")<<"}"
      <<"pack_flags[local_idx]=select(0u,1u,any((bitcast<vec4<u32>>(value)&vec4u(0x7fffffffu))>vec4u("<<(mixed_?"bitcast<u32>(residual_half_max)":"0x58800000u")<<")));"
      <<"workgroupBarrier();if(local_idx==0u){var bad=0u;for(var i=0u;i<64u;i++){bad|=pack_flags[i];}"
      <<flag.SetByOffset("physical_group","bad")<<"}";
    return Status::OK();
  }
 private:
  const bool skip_inactive_,mixed_;
};
class ResidualPackedMatrixV1 final : public Program<ResidualPackedMatrixV1> {
 public:
  explicit ResidualPackedMatrixV1(bool cached_weights=false,uint32_t register_rows=0,bool mixed=false)
      :Program{mixed?"RW07ResidualHalfOperandsF32AccumV1":register_rows==8?"RW07ResidualRegister8ParallelFmaV7":register_rows==4?"RW07ResidualRegister4ParallelFmaV7":cached_weights?"RW07ResidualPackedMatrixWeightsV3":"RW07ResidualPackedMatrixV2"},
       cached_weights_(cached_weights),register_rows_(register_rows),mixed_(mixed){}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(M0_RESIDUAL_TILE_UNIFORMS_V1,
    {"row_begin",ProgramUniformVariableDataType::Uint32},{"row_count",ProgramUniformVariableDataType::Uint32},
    {"weight_flag_offset",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override {
    ORT_RETURN_IF_NOT((register_rows_==0||register_rows_==4||register_rows_==8)&&(!register_rows_||cached_weights_)&&(!mixed_||(cached_weights_&&register_rows_==0)),"RW07 register/mixed matrix requires prepared safety flags");
    const uint32_t tile_rows=mixed_?kResidualMixedTileRowsV1:register_rows_?8u*register_rows_:32u;
    const uint32_t tile_elements=tile_rows*32u;
    const auto& x=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("bias",ShaderUsage::UseValueTypeAlias);
    const auto& col=s.AddInput("columns",ShaderUsage::UseValueTypeAlias);
    const auto& flags=s.AddInput("flags",ShaderUsage::UseValueTypeAlias);
    const auto* original_weight=mixed_?&s.AddInput("original_weight",ShaderUsage::UseValueTypeAlias):nullptr;
    const auto& y=s.AddOutput("output",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e);OriginalArithmetic(s);auto& a=s.AdditionalImplementation();
    a<<"var<workgroup> matrix_flags:array<u32,64>;var<workgroup> matrix_enabled:u32;var<workgroup> matrix_safe:u32;";
    if(register_rows_||mixed_){
      a<<"fn scalar_weight(at:u32)->f32 {";
      if(mixed_)a<<"return "<<original_weight->GetByOffset("at/4u")<<"[at%4u];}";
      else a<<"let row=at/(uniforms.channels*uniforms.kernel)-uniforms.output_channel_offset;"
        <<"let inner=at%(uniforms.channels*uniforms.kernel)-uniforms.channel_offset*uniforms.kernel;"
        <<"let vectors=(uniforms.channel_count*uniforms.kernel+3u)/4u;return "
        <<w.GetByOffset("((row/64u)*vectors+inner/4u)*64u+row%64u")<<"[inner%4u];}";
      a<<"fn mm_readA(batch:i32,row:i32,c:i32)->vec4<"<<(mixed_?"f16":"f32")<<">{"
        <<"let vectors=(uniforms.channel_count*uniforms.kernel+3u)/4u;let r=uniforms.row_begin+u32(row);"
        <<"if(r>=uniforms.output_channel_count||u32(c)>=vectors){return vec4<"<<(mixed_?"f16":"f32")<<">(0.0);}return "
        <<w.GetByOffset("((r/64u)*vectors+u32(c))*64u+r%64u")<<";}";
    }else{
    a<<"fn scalar_weight(at:u32)->f32{return "<<w.GetByOffset("at/4u")<<"[at%4u];}"
      <<"fn mm_readA(batch:i32,row:i32,c:i32)->vec4<f32>{var result=vec4<f32>(0.0);"
      <<"if(row<0||u32(row)>=uniforms.row_count){return result;}"
      <<"let inner=u32(c)*4u;let count=uniforms.channel_count*uniforms.kernel;if(inner>=count){return result;}"
      <<"let base=((uniforms.output_channel_offset+uniforms.row_begin+u32(row))*uniforms.channels+uniforms.channel_offset)*uniforms.kernel+inner;"
      <<"let shift=base%4u;let first="<<w.GetByOffset("base/4u")<<";"
      <<"if(shift==0u){result=first;}else{var second=vec4<f32>(0.0);"
      <<"if(base/4u+1u<uniforms.channels*uniforms.channels*uniforms.kernel/4u){second="<<w.GetByOffset("base/4u+1u")<<";}"
      <<"for(var lane=0u;lane<4u;lane++){let index=shift+lane;if(index<4u){result[lane]=first[index];}else{result[lane]=second[index-4u];}}}"
      <<"if(inner+4u>count){for(var lane=0u;lane<4u;lane++){if(inner+lane>=count){result[lane]=0.0;}}}return result;}";
    }
    a<<"fn mm_readB(batch:i32,row:i32,c:i32)->vec4<"<<(mixed_?"f16":"f32")<<">{";
    if(!register_rows_)a<<"if(row<0||u32(row)>=uniforms.channel_count*uniforms.kernel){return vec4<"<<(mixed_?"f16":"f32")<<">(0.0);}";
    // Register loads address the physical four-row padding prepared with the
    // columns; shared32-K tiles retain their wider logical-tail guard.
    a<<"return "<<col.GetByOffset("((u32(row)/4u)*(uniforms.time_count/32u)+u32(c)/8u)*32u+(u32(row)%4u)*8u+u32(c)%8u")<<";}"
      <<"fn serial_sample(row:u32,t:u32)->f32 {let n="<<e.GetByOffset("6u")<<";if(t>=n){return 0.0;}"
      <<"var total=0.0;if(uniforms.channel_offset!=0u){total="<<y.GetByOffset("row*uniforms.output_capacity+t")<<";}"
      <<"let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
      <<"for(var ic=0u;ic<uniforms.channel_count;ic++){for(var k=0u;k<uniforms.kernel;k++){let pos=i32(t+k*uniforms.dilation)-pad;"
      <<"var value=0.0;if(pos>=0&&pos<i32(n)){value="<<x.GetByOffset("ic*uniforms.input_capacity+u32(pos)")<<";}"
      <<"let weight=scalar_weight(((uniforms.output_channel_offset+row)*uniforms.channels+uniforms.channel_offset+ic)*uniforms.kernel+k);"
      <<"total=original_add(total,original_mul(value,weight));}}"
      <<"if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){total=original_add(total,"<<b.GetByOffset("uniforms.output_channel_offset+row")<<");}return total;}"
      <<"fn mm_write(batch:i32,row:i32,c:i32,value:vec4<f32>){if(row<0||u32(row)>=uniforms.row_count){return;}"
      <<"let r=uniforms.row_begin+u32(row);let n="<<e.GetByOffset("6u")<<";"
      <<"for(var lane=0u;lane<4u;lane++){let t=uniforms.time_begin+u32(c)*4u+lane;if(t<uniforms.output_capacity){var result=0.0;"
      <<"if(t<n){result=value[lane];if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){result=original_add(result,"
      <<b.GetByOffset("uniforms.output_channel_offset+r")<<");}}"<<y.SetByOffset("r*uniforms.output_capacity+t","result")<<"}}}";
    auto& f=s.MainFunctionBody();
    f<<"let tile_columns=uniforms.time_count/32u;if(tile_columns==0u||workgroup_idx>=tile_columns*((uniforms.row_count+"<<tile_rows-1<<"u)/"<<tile_rows<<"u)){return;}"
      <<"let tile_t=uniforms.time_begin+"<<(IndirectDispatchTensor()?"workgroup_id.x":"(workgroup_idx%tile_columns)")
      <<"*32u;let tile_row=uniforms.row_begin+"<<(IndirectDispatchTensor()?"workgroup_id.y":"(workgroup_idx/tile_columns)")<<"*"<<tile_rows<<"u;"
      <<"if(local_idx==0u){matrix_enabled=0u;let n="<<e.GetByOffset("6u")<<";"
      <<"if(valid_output()&&n<=uniforms.input_capacity&&uniforms.time_count<=2048u&&uniforms.time_count%32u==0u&&"
      <<"uniforms.row_begin<uniforms.output_channel_count&&uniforms.row_count>0u&&uniforms.row_count<=uniforms.output_channel_count-uniforms.row_begin&&"
      <<(register_rows_?"uniforms.row_begin%"+std::to_string(tile_rows)+"u==0u&&":"")
      <<"uniforms.channel_count>0u&&uniforms.channel_count<=256u&&uniforms.channel_offset<uniforms.channels&&"
      <<"uniforms.channel_count<=uniforms.channels-uniforms.channel_offset&&uniforms.output_channel_offset<uniforms.channels&&"
      <<"uniforms.output_channel_count<=uniforms.channels-uniforms.output_channel_offset&&"
      <<"uniforms.kernel>=1u&&uniforms.kernel<=11u&&uniforms.dilation>=1u&&uniforms.dilation<=5u&&tile_t<n){matrix_enabled=1u;}}"
      <<"let enabled=workgroupUniformLoad(&matrix_enabled);"
      <<"if(enabled==0u){for(var index=local_idx;index<"<<tile_elements<<"u;index+=64u){let r=tile_row+index/32u;let t=tile_t+index%32u;"
      <<"if(r<uniforms.row_begin+uniforms.row_count&&t<uniforms.output_capacity){"<<y.SetByOffset("r*uniforms.output_capacity+t","0.0")<<"}}return;}"
      <<"let n="<<e.GetByOffset("6u")<<";var bad=0u;"
      <<"let flag_rows=(uniforms.channel_count*uniforms.kernel+7u)/8u;"
      <<"let flag_time=(tile_t-uniforms.time_begin)/32u;"
      <<"for(var i=local_idx;i<flag_rows;i+=64u){bad|="<<flags.GetByOffset("i*tile_columns+flag_time")<<";}";
    if(cached_weights_){
      f<<"bad|="<<flags.GetByOffset("uniforms.weight_flag_offset+tile_row/32u")<<";";
      if(register_rows_==8||mixed_)f<<"if(tile_row+32u<uniforms.row_begin+uniforms.row_count){bad|="
        <<flags.GetByOffset("uniforms.weight_flag_offset+tile_row/32u+1u")<<";}";
    }
    if(!cached_weights_)f<<"let inner_vectors=(uniforms.channel_count*uniforms.kernel+3u)/4u;"
      <<"for(var index=local_idx;index<32u*inner_vectors;index+=64u){let r=tile_row+index/inner_vectors;"
      <<"if(r<uniforms.row_begin+uniforms.row_count){let values=mm_readA(0,i32(r-uniforms.row_begin),i32(index%inner_vectors));"
      <<"if(any((bitcast<vec4<u32>>(values)&vec4u(0x7fffffffu))>vec4u(0x58800000u))){bad=1u;}}}";
    f<<"if(uniforms.channel_offset!=0u){for(var index=local_idx;index<"<<tile_elements<<"u;index+=64u){let r=tile_row+index/32u;let t=tile_t+index%32u;"
      <<"if(r<uniforms.row_begin+uniforms.row_count&&t<n){let value="<<y.GetByOffset("r*uniforms.output_capacity+t")<<";"
      <<"if((bitcast<u32>(value)&0x7fffffffu)>0x71800000u){bad=1u;}}}}"
      <<"matrix_flags[local_idx]=bad;workgroupBarrier();if(local_idx==0u){var combined=0u;for(var i=0u;i<64u;i++){combined|=matrix_flags[i];}matrix_safe=1u-min(combined,1u);}"
      <<"let safe=workgroupUniformLoad(&matrix_safe);if(safe==0u){for(var index=local_idx;index<"<<tile_elements<<"u;index+=64u){let r=tile_row+index/32u;let t=tile_t+index%32u;"
      <<"if(r<uniforms.row_begin+uniforms.row_count&&t<uniforms.output_capacity){"<<y.SetByOffset("r*uniforms.output_capacity+t","serial_sample(r,t)")<<"}}return;}";
    if(register_rows_){
      // Keep the reduction lane order, but reuse each four-time input vector
      // across four/eight output rows without a shared-memory barrier per K block.
      f<<"let first_row=tile_row-uniforms.row_begin+(local_idx/8u)*"<<register_rows_<<"u;"
        <<"let column=(tile_t-uniforms.time_begin)/4u+local_idx%8u;"
        <<"let first_t=uniforms.time_begin+column*4u;";
      for(uint32_t row=0;row<register_rows_;++row){
        f<<"var partial"<<row<<"=vec4<f32>(0.0);var acc"<<row<<"=vec4<f32>(0.0);if(uniforms.channel_offset!=0u&&first_row+"<<row<<"u<uniforms.row_count){"
          <<"for(var lane=0u;lane<4u;lane++){if(first_t+lane<n){acc"<<row<<"[lane]="
          <<y.GetByOffset("(uniforms.row_begin+first_row+"+std::to_string(row)+"u)*uniforms.output_capacity+first_t+lane")<<";}}}";
      }
      f<<"let inner_vectors=(uniforms.channel_count*uniforms.kernel+3u)/4u;"
        <<"for(var k=0u;k<inner_vectors;k++){"
        <<"let v0=mm_readB(0,i32(k*4u),i32(column));"
        <<"let v1=mm_readB(0,i32(k*4u+1u),i32(column));"
        <<"let v2=mm_readB(0,i32(k*4u+2u),i32(column));"
        <<"let v3=mm_readB(0,i32(k*4u+3u),i32(column));";
      // Interleave two independent rows, using two independent reduction chains. Exceptional
      // tiles still use serial_sample; finite tiles use native FP32 FMA.
      for(uint32_t row=0;row<register_rows_;row+=2){
        f<<"{let weights0=mm_readA(0,i32(first_row+"<<row<<"u),i32(k));"
          <<"let weights1=mm_readA(0,i32(first_row+"<<row+1<<"u),i32(k));";
        for(uint32_t k=0;k<4;++k){
          const char component="xyzw"[k];
          const char* target=k%2?"partial":"acc";
          f<<target<<row<<"=fma(v"<<k<<",vec4<f32>(weights0."<<component<<"),"<<target<<row<<");"
           <<target<<row+1<<"=fma(v"<<k<<",vec4<f32>(weights1."<<component<<"),"<<target<<row+1<<");";
        }
        f<<"}";
      }
      f<<"}";
      for(uint32_t row=0;row<register_rows_;++row)
        f<<"mm_write(0,i32(first_row+"<<row<<"u),i32(column),acc"<<row<<"+partial"<<row<<");";
      return Status::OK();
    }
    std::string prelude=std::move(f).str();f=OStringStream(kStringInitialSizeShaderSourceCodeMain);
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(s,InlinedVector<int64_t>{4,mixed_?kResidualMixedRowsPerThreadV1:4,1},8,8,mixed_?"f16":"f32",nullptr,
        false,false,1.0f,true,4,32,false,32,mixed_,"workgroup_idx",mixed_));
    std::string body=std::move(f).str();f=OStringStream(kStringInitialSizeShaderSourceCodeMain);
    const auto start=body.find("  let localRow =");ORT_RETURN_IF_NOT(start!=std::string::npos,"RW07 packed matrix source layout");body=body.substr(start);
    const std::string decl="  var acc: array<vec4<f32>, rowPerThread>;";const auto at=body.find(decl);
    ORT_RETURN_IF_NOT(at!=std::string::npos,"RW07 packed accumulator layout");OStringStream seed(512);
    seed<<"if(uniforms.channel_offset!=0u){for(var r=0u;r<u32(rowPerThread);r++){let row=uniforms.row_begin+u32(globalRow)+r;"
      <<"for(var lane=0u;lane<4u;lane++){let t=uniforms.time_begin+u32(globalCol)*4u+lane;if(row<uniforms.row_begin+uniforms.row_count&&t<n){acc[r][lane]="
      <<y.GetByOffset("row*uniforms.output_capacity+t")<<";}}}}";body.insert(at+decl.size(),std::move(seed).str());
    const std::string needle="uniforms.dim_inner",replace="(uniforms.channel_count*uniforms.kernel)";
    for(size_t i=0;(i=body.find(needle,i))!=std::string::npos;i+=replace.size())body.replace(i,needle.size(),replace);
    f<<prelude<<"let logical_workgroup_id="<<(IndirectDispatchTensor()?"workgroup_id":"vec3u(workgroup_idx%tile_columns,workgroup_idx/tile_columns,0u)")<<";"
      <<"let logical_global_id=logical_workgroup_id*vec3u(8u,8u,1u)+local_id;"<<body;
    return Status::OK();
  }
 private:
  const bool cached_weights_;
  const uint32_t register_rows_;
  const bool mixed_;
};

class GeneratorResidualConv final : public Program<GeneratorResidualConv> {
 public:
  explicit GeneratorResidualConv(bool vectorized=M0VectorizedConvV1(),bool tiled=M0TiledConvV1(),
                                 bool channel_tiled=M0OutputChannelTiledConvV1(),uint32_t channel_block=1,
                                 bool matrix=false,bool wide_grid=false)
      : Program{matrix?"RW07ResidualImplicitMatMulV10":channel_tiled?(channel_block>1?"RW07ResidualConvOutputChannelBlockV8":"RW07ResidualConvOutputChannelTileV7"):tiled?"RW07ResidualConvSharedInputTileV6":vectorized?"RW07ResidualConvFourTimeSamplesV5":"RW07ResidualConvChannelElementChunksV4"},
        vectorized_(vectorized||tiled||channel_tiled),tiled_(tiled||channel_tiled),channel_tiled_(channel_tiled),channel_block_(channel_block),matrix_(matrix),wide_grid_(matrix||wide_grid) {
    ORT_ENFORCE(!matrix||channel_tiled,"RW07 residual matrix requires channel tiling");
    ORT_ENFORCE(!channel_tiled||(channel_block==1||channel_block==2||channel_block==4||channel_block==8),
                "RW07 residual channel block");
    if(channel_tiled)CacheHint(std::to_string(channel_block));
  }
  uint64_t WorkItems(uint32_t output_channels,uint32_t capacity) const {
    if(!channel_tiled_)return uint64_t(output_channels)*capacity;
    const uint64_t width=8,unit=width*32;
    const uint64_t columns=(uint64_t(output_channels)+width-1)/width,rows=(uint64_t(capacity)+31)/32;
    if(columns&&rows>UINT64_MAX/unit/columns)return 0;
    return columns*rows*unit;
  }
  uint32_t ChunkElements(uint32_t inputs,uint32_t kernel) const {
    const uint64_t inner=uint64_t(inputs)*kernel;
    const uint32_t count=wide_grid_&&inner?static_cast<uint32_t>(ResidualProductsPerDispatchV1()/((inner+31)/32*32)):
      ResidualConvElementsPerDispatchV1(inputs,kernel);
    return channel_tiled_?count/256*256:count;
  }
  uint32_t ValuesPerInvocation() const { return vectorized_?4u:1u; }
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
      {"channel_offset", ProgramUniformVariableDataType::Uint32},
      {"channel_count", ProgramUniformVariableDataType::Uint32},
      {"output_channel_offset", ProgramUniformVariableDataType::Uint32},
      {"output_channel_count", ProgramUniformVariableDataType::Uint32},
      {"element_begin", ProgramUniformVariableDataType::Uint32},
      {"element_count", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32},
      {"time_begin",ProgramUniformVariableDataType::Uint32},
      {"time_count",ProgramUniformVariableDataType::Uint32});
  Status MatrixShader(ShaderHelper& s,const ShaderVariableHelper& x,const ShaderVariableHelper& e,
      const ShaderVariableHelper& w,const ShaderVariableHelper& b,const ShaderVariableHelper& y)const {
    auto& functions=s.AdditionalImplementation();
    functions<<"var<workgroup> matrix_flags:array<u32,64>;var<workgroup> matrix_active:u32;var<workgroup> matrix_safe:u32;"
      <<"fn mm_readA(batch:i32,row:i32,col:i32)->vec4<f32>{var result=vec4<f32>(0.0);"
      <<"if(row<0||u32(row)>=uniforms.output_channel_count){return result;}"
      <<"for(var lane=0u;lane<4u;lane++){let inner=u32(col)*4u+lane;"
      <<"if(inner<uniforms.channel_count*uniforms.kernel){let ic=uniforms.channel_offset+inner/uniforms.kernel;"
      <<"result[lane]="<<w.GetByOffset("((uniforms.output_channel_offset+u32(row))*uniforms.channels+ic)*uniforms.kernel+inner%uniforms.kernel")<<";}}return result;}"
      <<"fn mm_readB(batch:i32,row:i32,col:i32)->vec4<f32>{var result=vec4<f32>(0.0);"
      <<"if(row<0||u32(row)>=uniforms.channel_count*uniforms.kernel){return result;}"
      <<"let count="<<e.GetByOffset("6u")<<";let k=u32(row)%uniforms.kernel;let ic=u32(row)/uniforms.kernel;"
      <<"let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
      <<"for(var lane=0u;lane<4u;lane++){let t=u32(col)*4u+lane;let pos=i32(t+k*uniforms.dilation)-pad;"
      <<"if(t<count&&pos>=0&&pos<i32(count)){result[lane]="<<x.GetByOffset("ic*uniforms.input_capacity+u32(pos)")<<";}}return result;}"
      <<"fn mm_write(batch:i32,row:i32,col:i32,value:vec4<f32>){if(row<0||u32(row)>=uniforms.output_channel_count){return;}"
      <<"let count="<<e.GetByOffset("6u")<<";let oc=uniforms.output_channel_offset+u32(row);"
      <<"for(var lane=0u;lane<4u;lane++){let t=u32(col)*4u+lane;if(t<uniforms.output_capacity){var result=0.0;"
      <<"if(t<count){result=value[lane];"
      <<"if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){result=original_add(result,"<<b.GetByOffset("oc")<<");}}"
      <<y.SetByOffset("u32(row)*uniforms.output_capacity+t","result")<<"}}}";
    auto& f=s.MainFunctionBody();
    f<<"if(workgroup_idx>=uniforms.element_count/256u){return;}"
      <<"let row_tiles=(max(select(uniforms.output_capacity,uniforms.time_count,uniforms.time_count!=0u),1u)+31u)/32u;let tile=uniforms.element_begin/256u+workgroup_idx;"
      <<"let tile_t=(tile%row_tiles)*32u;let tile_oc=(tile/row_tiles)*8u;"
      <<"if(local_idx==0u){matrix_active=0u;let n="<<e.GetByOffset("6u")<<";"
      <<"if(valid_output()&&uniforms.element_begin%256u==0u&&uniforms.element_count%256u==0u&&"
      <<"uniforms.kernel>=1u&&uniforms.kernel<=11u&&uniforms.dilation>=1u&&uniforms.dilation<=5u&&"
      <<"uniforms.channel_offset<uniforms.channels&&uniforms.channel_count>0u&&uniforms.channel_count<=256u&&"
      <<"uniforms.channel_count<=uniforms.channels-uniforms.channel_offset&&"
      <<"uniforms.output_channel_offset<uniforms.channels&&uniforms.output_channel_count>0u&&"
      <<"uniforms.output_channel_count<=uniforms.channels-uniforms.output_channel_offset&&"
      <<"n<=uniforms.input_capacity&&tile_t<n&&tile_oc<uniforms.output_channel_count){matrix_active=1u;}}"
      <<"let matrix_enabled=workgroupUniformLoad(&matrix_active);"
      <<"if(matrix_enabled==0u){for(var index=local_idx;index<256u;index+=64u){let row=tile_oc+index/32u;let t=tile_t+index%32u;"
      <<"if(row<uniforms.output_channel_count&&t<uniforms.output_capacity){"<<y.SetByOffset("row*uniforms.output_capacity+t","0.0")<<"}}return;}"
      <<"let count="<<e.GetByOffset("6u")<<";let span=32u+(uniforms.kernel-1u)*uniforms.dilation;"
      <<"let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);var bad=0u;"
      <<"for(var index=local_idx;index<uniforms.channel_count*span;index+=64u){let pos=i32(tile_t+index%span)-pad;"
      <<"if(pos>=0&&pos<i32(count)){let value="<<x.GetByOffset("(index/span)*uniforms.input_capacity+u32(pos)")<<";"
      <<"if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){bad=1u;}}}"
      <<"for(var index=local_idx;index<8u*uniforms.channel_count*uniforms.kernel;index+=64u){"
      <<"let inner=index%(uniforms.channel_count*uniforms.kernel);let row=tile_oc+index/(uniforms.channel_count*uniforms.kernel);"
      <<"if(row<uniforms.output_channel_count){let value="<<w.GetByOffset("((uniforms.output_channel_offset+row)*uniforms.channels+uniforms.channel_offset+inner/uniforms.kernel)*uniforms.kernel+inner%uniforms.kernel")<<";"
      <<"if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){bad=1u;}}}"
      <<"if(uniforms.channel_offset!=0u){for(var index=local_idx;index<256u;index+=64u){let row=tile_oc+index/32u;let t=tile_t+index%32u;"
      <<"if(row<uniforms.output_channel_count&&t<count){let value="<<y.GetByOffset("row*uniforms.output_capacity+t")<<";"
      <<"if((bitcast<u32>(value)&0x7fffffffu)>0x71800000u){bad=1u;}}}}"
      <<"matrix_flags[local_idx]=bad;workgroupBarrier();if(local_idx==0u){var combined=0u;for(var i=0u;i<64u;i++){combined|=matrix_flags[i];}matrix_safe=1u-min(combined,1u);}"
      <<"let safe=workgroupUniformLoad(&matrix_safe);if(safe==0u){for(var index=local_idx;index<256u;index+=64u){let row=tile_oc+index/32u;let t=tile_t+index%32u;"
      <<"if(row<uniforms.output_channel_count&&t<uniforms.output_capacity){"<<y.SetByOffset("row*uniforms.output_capacity+t","residual_sample(row*uniforms.output_capacity+t)")<<"}}return;}";
    // Share ORT's packed tile arithmetic with convolution-specific read/write
    // functions. No im2col allocation or additional storage binding is needed.
    std::string prelude=std::move(f).str();f=OStringStream(kStringInitialSizeShaderSourceCodeMain);
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(s,InlinedVector<int64_t>{4,1,1},8,8,"f32",nullptr));
    std::string body=std::move(f).str();f=OStringStream(kStringInitialSizeShaderSourceCodeMain);
    const auto start=body.find("  let localRow =");
    ORT_RETURN_IF_NOT(start!=std::string::npos,"RW07 residual packed source layout");
    body=body.substr(start);
    const std::string declaration="  var acc: array<vec4<f32>, rowPerThread>;";
    const auto accumulator=body.find(declaration);
    ORT_RETURN_IF_NOT(accumulator!=std::string::npos,"RW07 residual packed accumulator layout");
    OStringStream seed(512);
    seed<<"if(uniforms.channel_offset!=0u){for(var r=0u;r<u32(rowPerThread);r++){let row=u32(globalRow)+r;"
      <<"for(var lane=0u;lane<4u;lane++){let t=u32(globalCol)*4u+lane;"
      <<"if(row<uniforms.output_channel_count&&t<count){acc[r][lane]="
      <<y.GetByOffset("row*uniforms.output_capacity+t")<<";}}}}";
    body.insert(accumulator+declaration.size(),std::move(seed).str());
    const std::string needle="uniforms.dim_inner",replacement="(uniforms.channel_count*uniforms.kernel)";
    for(size_t at=0;(at=body.find(needle,at))!=std::string::npos;at+=replacement.size())body.replace(at,needle.size(),replacement);
    f<<prelude<<"let logical_workgroup_id=vec3u(tile%row_tiles,tile/row_tiles,0u);"
      <<"let logical_global_id=logical_workgroup_id*vec3u(8u,8u,1u)+local_id;"<<body;
    return Status::OK();
  }
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("carrier",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("bias",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("output",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e,!matrix_);
    OriginalArithmetic(s);
    if(vectorized_){
      auto& functions=s.AdditionalImplementation();
      functions << R"WGSL(
fn original_mul4(a:vec4<f32>,b:f32)->vec4<f32>{
 let aa=bitcast<vec4<u32>>(a)&vec4<u32>(0x7fffffffu);let bb=bitcast<u32>(b)&0x7fffffffu;
 if(all(aa<vec4<u32>(0x7f800000u))&&bb<0x7f800000u){return a*vec4<f32>(b);}
 return vec4<f32>(original_mul(a.x,b),original_mul(a.y,b),original_mul(a.z,b),original_mul(a.w,b));
}
fn original_add4(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{
 let aa=bitcast<vec4<u32>>(a)&vec4<u32>(0x7fffffffu);let bb=bitcast<vec4<u32>>(b)&vec4<u32>(0x7fffffffu);
 if(all(aa<vec4<u32>(0x7f800000u))&&all(bb<vec4<u32>(0x7f800000u))){return a+b;}
 return vec4<f32>(original_add(a.x,b.x),original_add(a.y,b.y),original_add(a.z,b.z),original_add(a.w,b.w));
}
)WGSL";
      functions << "fn residual_sample(at:u32)->f32 {"
        << "if(!valid_output()){return 0.0;}let count=" << e.GetByOffset("6u") << ";"
        << "let t=at%uniforms.output_capacity;let local_oc=at/uniforms.output_capacity;"
        << "if(uniforms.output_channel_offset>=uniforms.channels||uniforms.output_channel_count==0u||"
        << "uniforms.output_channel_count>uniforms.channels-uniforms.output_channel_offset||t>=count||"
        << "uniforms.channel_offset>=uniforms.channels||uniforms.channel_count==0u||"
        << "uniforms.channel_count>uniforms.channels-uniforms.channel_offset){return 0.0;}"
        << "let oc=uniforms.output_channel_offset+local_oc;var total:f32=0.0;"
        << "if(uniforms.channel_offset!=0u){total=" << y.GetByOffset("at") << ";}"
        << "let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
        << "for(var local_ic=0u;local_ic<uniforms.channel_count;local_ic++){let ic=uniforms.channel_offset+local_ic;"
        << "for(var k=0u;k<uniforms.kernel;k++){let pos=i32(t)+i32(k*uniforms.dilation)-pad;var value:f32=0.0;"
        << "if(pos>=0&&pos<i32(count)){value=" << x.GetByOffset("local_ic*uniforms.input_capacity+u32(pos)") << ";}"
        << "total=original_add(total,original_mul(value," << w.GetByOffset("(oc*uniforms.channels+ic)*uniforms.kernel+k") << "));}}"
        << "if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){total=original_add(total," << b.GetByOffset("oc") << ");}"
        << "return total;}";
      if(matrix_)return MatrixShader(s,x,e,w,b,y);
      if(channel_tiled_){
        functions << "var<workgroup> channel_values:array<f32," << 82*channel_block_ << ">;var<workgroup> channel_weights:array<f32,"
          << 88*channel_block_ << ">;var<workgroup> channel_active:u32;var<workgroup> channel_finite:array<u32,64>;var<workgroup> channel_safe:u32;";
        auto& f=s.MainFunctionBody();
        f << "let row_tiles=(max(select(uniforms.output_capacity,uniforms.time_count,uniforms.time_count!=0u),1u)+31u)/32u;"
          << "let tile_group=uniforms.element_begin/256u+workgroup_idx;let tile_t=uniforms.time_begin+(tile_group%row_tiles)*32u;"
          << "let tile_oc=(tile_group/row_tiles)*8u;let local_oc=tile_oc+local_idx/8u;let t=tile_t+(local_idx%8u)*4u;"
          << "let physical=local_oc<uniforms.output_channel_count&&t<uniforms.output_capacity;"
          << "if(workgroup_idx*256u>=uniforms.element_count){return;}"
          << "if(local_idx==0u){channel_active=0u;let n=" << e.GetByOffset("6u") << ";"
          << "if(valid_output()&&uniforms.element_begin%256u==0u&&uniforms.element_count%256u==0u&&"
          << "uniforms.kernel>=1u&&uniforms.kernel<=11u&&uniforms.dilation>=1u&&uniforms.dilation<=5u&&"
          << "uniforms.output_channel_offset<uniforms.channels&&uniforms.output_channel_count>0u&&"
          << "uniforms.output_channel_count<=uniforms.channels-uniforms.output_channel_offset&&"
          << "uniforms.channel_offset<uniforms.channels&&uniforms.channel_count>0u&&"
          << "uniforms.channel_count<=uniforms.channels-uniforms.channel_offset&&uniforms.channel_count<=256u&&n<=uniforms.input_capacity&&tile_t<n&&tile_oc<uniforms.output_channel_count){channel_active=1u;}}"
          << "let active_flag=workgroupUniformLoad(&channel_active);"
          << "if(active_flag==0u){if(physical){for(var lane=0u;lane<4u;lane++){if(t+lane<uniforms.output_capacity){"
          << y.SetByOffset("local_oc*uniforms.output_capacity+t+lane","0.0") << "}}}return;}"
          << "let count=" << e.GetByOffset("6u") << ";let oc=uniforms.output_channel_offset+local_oc;"
          << "var total=vec4<f32>(0.0);if(physical&&uniforms.channel_offset!=0u){"
          << "for(var lane=0u;lane<4u;lane++){if(t+lane<uniforms.output_capacity){total[lane]="
          << y.GetByOffset("local_oc*uniforms.output_capacity+t+lane") << ";}}}"
          << "let span=32u+(uniforms.kernel-1u)*uniforms.dilation;let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
          // Prove each group's magnitudes safe once. The admitted <=256*11
          // products of <=2^50 operands plus <=2^100 initial sums stay <2^112.
          // Exceptional or extreme groups retain the original scalar classes.
          << "var flag_bad=0u;if(physical&&any((bitcast<vec4<u32>>(total)&vec4<u32>(0x7fffffffu))>vec4<u32>(0x71800000u))){flag_bad=1u;}"
          << "for(var index=local_idx;index<uniforms.channel_count*span;index+=64u){let local_ic=index/span;let pos=i32(tile_t+index%span)-pad;"
          << "if(pos>=0&&pos<i32(count)){let value=" << x.GetByOffset("local_ic*uniforms.input_capacity+u32(pos)") << ";"
          << "if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){flag_bad=1u;}}}"
          << "for(var index=local_idx;index<uniforms.channel_count*8u*uniforms.kernel;index+=64u){let local_ic=index/(8u*uniforms.kernel);"
          << "let row=tile_oc+(index/uniforms.kernel)%8u;let k=index%uniforms.kernel;if(row<uniforms.output_channel_count){let value="
          << w.GetByOffset("((uniforms.output_channel_offset+row)*uniforms.channels+uniforms.channel_offset+local_ic)*uniforms.kernel+k") << ";"
          << "if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){flag_bad=1u;}}}"
          << "channel_finite[local_idx]=flag_bad;workgroupBarrier();if(local_idx==0u){var combined=0u;for(var lane=0u;lane<64u;lane++){combined=combined|channel_finite[lane];}channel_safe=1u-min(combined,1u);}"
          << "let finite_safe=workgroupUniformLoad(&channel_safe)==1u;"
          ;
        const auto dot_loop=[&](bool fast){
          f
          << "for(var input_base=0u;input_base<uniforms.channel_count;input_base+=" << channel_block_ << "u){"
          << "let block_count=min(" << channel_block_ << "u,uniforms.channel_count-input_base);"
          << "for(var index=local_idx;index<block_count*span;index+=64u){let input_lane=index/span;let within=index%span;"
          << "let pos=i32(tile_t+within)-pad;var value=0.0;if(pos>=0&&pos<i32(count)){value="
          << x.GetByOffset("(input_base+input_lane)*uniforms.input_capacity+u32(pos)") << ";}channel_values[input_lane*82u+within]=value;}"
          << "for(var index=local_idx;index<block_count*8u*uniforms.kernel;index+=64u){let input_lane=index/(8u*uniforms.kernel);"
          << "let within=index%(8u*uniforms.kernel);let row=tile_oc+within/uniforms.kernel;let k=within%uniforms.kernel;"
          << "var value=0.0;if(row<uniforms.output_channel_count){value="
          << w.GetByOffset("((uniforms.output_channel_offset+row)*uniforms.channels+uniforms.channel_offset+input_base+input_lane)*uniforms.kernel+k")
          << ";}channel_weights[input_lane*88u+within]=value;}"
          << "workgroupBarrier();if(physical){for(var input_lane=0u;input_lane<block_count;input_lane++){"
          << "for(var k=0u;k<uniforms.kernel;k++){let index=input_lane*82u+(local_idx%8u)*4u+k*uniforms.dilation;"
          << "let values=vec4<f32>(channel_values[index],channel_values[index+1u],channel_values[index+2u],channel_values[index+3u]);"
          << "let weight=channel_weights[input_lane*88u+(local_idx/8u)*uniforms.kernel+k];"
          ;
          if(fast)f<<"let product=values*vec4<f32>(weight);total=total+product;";
          else f<<"total=original_add4(total,original_mul4(values,weight));";
          f<<"}}}workgroupBarrier();}";
        };
        f<<"if(finite_safe){";dot_loop(true);f<<"}else{";dot_loop(false);f<<"}";
        f
          << "if(physical){if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){total=original_add4(total,vec4<f32>("
          << b.GetByOffset("oc") << "));}for(var lane=0u;lane<4u;lane++){if(t+lane<uniforms.output_capacity){var value=0.0;"
          << "if(t+lane<count){value=total[lane];}" << y.SetByOffset("local_oc*uniforms.output_capacity+t+lane","value") << "}}}";
        return Status::OK();
      }
      if(tiled_){
        functions << "var<workgroup> conv_values:array<f32,512>;var<workgroup> conv_weights:array<f32,11>;var<workgroup> conv_tile_active:u32;";
      }
      auto& f=s.MainFunctionBody();
      if(tiled_){
        // A full group owns 256 adjacent samples of one output channel. Share
        // overlapping input rows and weights; boundary groups use the exact
        // four-lane path below. All barrier decisions are workgroup-uniform.
        f << "let tile_offset=workgroup_idx*256u;let tile_at=uniforms.element_begin+tile_offset;"
          << "let tile_t=tile_at%uniforms.output_capacity;let tile_count=" << e.GetByOffset("6u") << ";"
          << "if(local_idx==0u){conv_tile_active=0u;let elements=uniforms.output_channel_count*uniforms.output_capacity;"
          << "if(valid_output()&&uniforms.element_begin<elements&&uniforms.element_count<=elements-uniforms.element_begin&&"
          << "tile_offset<uniforms.element_count&&256u<=uniforms.element_count-tile_offset&&"
          << "tile_t+255u<tile_count&&tile_t+255u<uniforms.output_capacity&&tile_count<=uniforms.input_capacity&&"
          << "uniforms.kernel>=1u&&uniforms.kernel<=11u&&uniforms.dilation>=1u&&uniforms.dilation<=5u&&"
          << "uniforms.output_channel_offset<uniforms.channels&&uniforms.output_channel_count>0u&&"
          << "uniforms.output_channel_count<=uniforms.channels-uniforms.output_channel_offset&&"
          << "uniforms.channel_offset<uniforms.channels&&uniforms.channel_count>0u&&"
          << "uniforms.channel_count<=uniforms.channels-uniforms.channel_offset){conv_tile_active=1u;}}"
          << "let tile_active=workgroupUniformLoad(&conv_tile_active);if(tile_active==1u){"
          << "let at=tile_at+local_idx*4u;let oc=uniforms.output_channel_offset+tile_at/uniforms.output_capacity;"
          << "let span=256u+(uniforms.kernel-1u)*uniforms.dilation;let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
          << "var total=vec4<f32>(0.0);if(uniforms.channel_offset!=0u){total=vec4<f32>("
          << y.GetByOffset("at") << "," << y.GetByOffset("at+1u") << "," << y.GetByOffset("at+2u") << "," << y.GetByOffset("at+3u") << ");}"
          << "for(var local_ic=0u;local_ic<uniforms.channel_count;local_ic++){let ic=uniforms.channel_offset+local_ic;"
          << "for(var index=local_idx;index<span;index+=64u){let pos=i32(tile_t+index)-pad;var value=0.0;"
          << "if(pos>=0&&pos<i32(tile_count)){value=" << x.GetByOffset("local_ic*uniforms.input_capacity+u32(pos)") << ";}conv_values[index]=value;}"
          << "if(local_idx<uniforms.kernel){conv_weights[local_idx]=" << w.GetByOffset("(oc*uniforms.channels+ic)*uniforms.kernel+local_idx") << ";}"
          << "workgroupBarrier();for(var k=0u;k<uniforms.kernel;k++){let index=local_idx*4u+k*uniforms.dilation;"
          << "let values=vec4<f32>(conv_values[index],conv_values[index+1u],conv_values[index+2u],conv_values[index+3u]);"
          << "total=original_add4(total,original_mul4(values,conv_weights[k]));}workgroupBarrier();}"
          << "if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){total=original_add4(total,vec4<f32>(" << b.GetByOffset("oc") << "));}"
          << y.SetByOffset("at","total.x") << y.SetByOffset("at+1u","total.y") << y.SetByOffset("at+2u","total.z") << y.SetByOffset("at+3u","total.w")
          << "return;}";
      }
      f << "let elements=uniforms.output_channel_count*uniforms.output_capacity;"
        << "if(uniforms.element_begin>=elements||uniforms.element_count>elements-uniforms.element_begin||"
        << "global_idx>=uniforms.element_count/4u+u32(uniforms.element_count%4u!=0u)){return;}"
        << "let offset=global_idx*4u;let at=uniforms.element_begin+offset;let lanes=min(4u,uniforms.element_count-offset);"
        << "let t=at%uniforms.output_capacity;let count=" << e.GetByOffset("6u") << ";"
        << "if(lanes!=4u||!valid_output()||t+3u>=count||t+3u>=uniforms.output_capacity||"
        << "uniforms.output_channel_offset>=uniforms.channels||uniforms.output_channel_count==0u||"
        << "uniforms.output_channel_count>uniforms.channels-uniforms.output_channel_offset||"
        << "uniforms.channel_offset>=uniforms.channels||uniforms.channel_count==0u||"
        << "uniforms.channel_count>uniforms.channels-uniforms.channel_offset){"
        << "for(var lane=0u;lane<lanes;lane++){let index=at+lane;" << y.SetByOffset("index","residual_sample(index)") << "}return;}"
        << "let oc=uniforms.output_channel_offset+at/uniforms.output_capacity;var total=vec4<f32>(0.0);"
        << "if(uniforms.channel_offset!=0u){total=vec4<f32>(" << y.GetByOffset("at") << "," << y.GetByOffset("at+1u") << ","
        << y.GetByOffset("at+2u") << "," << y.GetByOffset("at+3u") << ");}"
        << "let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
        << "for(var local_ic=0u;local_ic<uniforms.channel_count;local_ic++){let ic=uniforms.channel_offset+local_ic;"
        << "for(var k=0u;k<uniforms.kernel;k++){let pos=i32(t)+i32(k*uniforms.dilation)-pad;var values=vec4<f32>(0.0);"
        << "if(pos>=0&&pos+3<i32(count)){let index=local_ic*uniforms.input_capacity+u32(pos);values=vec4<f32>("
        << x.GetByOffset("index") << "," << x.GetByOffset("index+1u") << "," << x.GetByOffset("index+2u") << "," << x.GetByOffset("index+3u") << ");}"
        << "else{for(var lane=0u;lane<4u;lane++){let p=pos+i32(lane);if(p>=0&&p<i32(count)){values[lane]="
        << x.GetByOffset("local_ic*uniforms.input_capacity+u32(p)") << ";}}}"
        << "let weight=" << w.GetByOffset("(oc*uniforms.channels+ic)*uniforms.kernel+k") << ";total=original_add4(total,original_mul4(values,weight));}}"
        << "if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){total=original_add4(total,vec4<f32>(" << b.GetByOffset("oc") << "));}"
        << y.SetByOffset("at","total.x") << y.SetByOffset("at+1u","total.y") << y.SetByOffset("at+2u","total.z") << y.SetByOffset("at+3u","total.w");
      return Status::OK();
    }
    auto& f=s.MainFunctionBody();
    f << "let elements=uniforms.output_channel_count*uniforms.output_capacity;"
      << "if(uniforms.element_begin>=elements||uniforms.element_count>elements-uniforms.element_begin||"
      << "global_idx>=uniforms.element_count){return;}let at=uniforms.element_begin+global_idx;"
      << "if(!valid_output()){ " << y.SetByOffset("at","0.0") << "return;}"
      << "let count=" << e.GetByOffset("6u") << ";let t=at%uniforms.output_capacity;"
      << "let local_oc=at/uniforms.output_capacity;"
      << "if(uniforms.output_channel_offset>=uniforms.channels||uniforms.output_channel_count==0u||"
      << "uniforms.output_channel_count>uniforms.channels-uniforms.output_channel_offset){"
      << y.SetByOffset("at","0.0") << "return;}"
      << "let oc=uniforms.output_channel_offset+local_oc;"
      << "if(t>=count||uniforms.channel_offset>=uniforms.channels||uniforms.channel_count==0u||"
      << "uniforms.channel_count>uniforms.channels-uniforms.channel_offset){"
      << y.SetByOffset("at","0.0") << "return;}"
      << "var total:f32=0.0;if(uniforms.channel_offset!=0u){total=" << y.GetByOffset("at") << ";}"
      << "let pad=i32((uniforms.kernel-1u)*uniforms.dilation/2u);"
      << "for(var local_ic=0u;local_ic<uniforms.channel_count;local_ic++){let ic=uniforms.channel_offset+local_ic;"
      << "for(var k=0u;k<uniforms.kernel;k++){"
      << "let pos=i32(t)+i32(k*uniforms.dilation)-pad;var value:f32=0.0;"
      << "if(pos>=0&&pos<i32(count)){value=" << x.GetByOffset("local_ic*uniforms.input_capacity+u32(pos)") << ";}"
      << "total=original_add(total,original_mul(value," << w.GetByOffset("(oc*uniforms.channels+ic)*uniforms.kernel+k") << "));}}"
      << "if(uniforms.channel_offset+uniforms.channel_count==uniforms.channels){total=original_add(total,"
      << b.GetByOffset("oc") << ");}";
    f << y.SetByOffset("at","total");
    return Status::OK();
  }
 private:
  const bool vectorized_;
  const bool tiled_;
  const bool channel_tiled_;
  const uint32_t channel_block_;
  const bool matrix_,wide_grid_;
};


// Binary16's finite maximum is defined by IEEE 754. FP32 products/accumulation
// avoid half-precision reduction overflow; original raw operands remain available.
inline bool ResidualMixedSupportedV1(const ComputeContextBase& c) {
  const auto& limits=c.DeviceLimits();
  return M0MixedPrecisionConvolutionV1()&&M0BatchHeavyProgramsV1()&&ResidualPackedSupportedV1(limits)&&
      limits.maxStorageBuffersPerShaderStage>=8&&limits.maxBindingsPerBindGroup>=9&&
      c.HasFeature(wgpu::FeatureName::ShaderF16);
}

inline Status PrepareResidualPackedWeightFlagsV1(ComputeContextBase& c,const Tensor* extent,
    const Tensor* weight,Tensor* flags,const ResidualConvSpecV1& spec,uint64_t& dispatches) {
  const uint32_t count=(spec.output_channel_count+31)/32;
  ORT_RETURN_IF_NOT(extent&&weight&&flags&&weight->IsDataType<float>()&&flags->IsDataType<uint32_t>()&&
      spec.channels>0&&spec.channels<=512&&spec.channel_count>0&&spec.channel_count<=256&&
      spec.kernel>0&&spec.kernel<=11&&spec.channel_offset<spec.channels&&
      spec.channel_count<=spec.channels-spec.channel_offset&&spec.output_channel_count>0&&
      spec.output_channel_offset<spec.channels&&spec.output_channel_count<=spec.channels-spec.output_channel_offset&&
      count<=kM0ResidualPackedWeightFlagWordsV1&&flags->Shape().Size()>kM0ResidualPackedWeightFlagWordsV1&&
      flags->Shape().Size()<=UINT32_MAX&&weight->Shape().Size()==int64_t(spec.channels)*spec.channels*spec.kernel&&
      weight->Shape().Size()%4==0,"RW07 packed weight flag dimensions");
  const uint32_t base=static_cast<uint32_t>(flags->Shape().Size())-kM0ResidualPackedWeightFlagWordsV1;
  ResidualPackedWeightFlagsV1 program;
  program.AddInputs({{extent,ProgramTensorMetadataDependency::None},
                    {weight,ProgramTensorMetadataDependency::None,ProgramInput::Flatten,4}})
      .AddOutput({flags,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(count);
  spec.ApplyPacked(program,0,32,0,spec.output_channel_count,base);
  ORT_RETURN_IF_ERROR(c.RunProgram(program));++dispatches;
  return Status::OK();
}
inline Status PrepareResidualRegisterWeightsV1(ComputeContextBase& c,const Tensor* extent,
    const Tensor* weight,Tensor* packed,Tensor* flags,const ResidualConvSpecV1& spec,uint64_t& dispatches,bool mixed=false){
  const uint64_t elements=ResidualRegisterWeightElementsV1(spec.output_channel_count,spec.channel_count,spec.kernel);
  ORT_RETURN_IF_NOT(elements&&extent&&extent->IsDataType<uint32_t>()&&extent->Shape().Size()==32&&weight&&
    weight->IsDataType<float>()&&packed&&(mixed?packed->IsDataType<MLFloat16>():packed->IsDataType<float>())&&flags&&flags->IsDataType<uint32_t>()&&
    spec.channels>0&&spec.channels<=512&&spec.channel_offset<spec.channels&&
    spec.channel_count<=spec.channels-spec.channel_offset&&spec.output_channel_offset<spec.channels&&
    spec.output_channel_count<=spec.channels-spec.output_channel_offset&&
    weight->Shape().Size()==int64_t(spec.channels)*spec.channels*spec.kernel&&weight->Shape().Size()%4==0&&
    uint64_t(packed->Shape().Size())>=elements&&uint64_t(packed->Shape().Size())<=kM0ResidualRegisterWeightScratchBytesV1/4&&
    flags->Shape().Size()>kM0ResidualPackedWeightFlagWordsV1&&flags->Shape().Size()<=UINT32_MAX,
    "RW07 blocked register weight dimensions/types/bound");
  const uint32_t rows=(spec.output_channel_count+63)/64*64;
  const uint32_t base=static_cast<uint32_t>(flags->Shape().Size())-kM0ResidualPackedWeightFlagWordsV1;
  ResidualPackedWeightFlagsV1 program(true,mixed);
  program.AddInputs({{extent,ProgramTensorMetadataDependency::None},
                    {weight,ProgramTensorMetadataDependency::None,ProgramInput::Flatten,4}})
    .AddOutput({packed,ProgramTensorMetadataDependency::None,ProgramOutput::Flatten,4})
    .AddOutput({flags,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(rows/32);
  spec.ApplyPacked(program,0,32,0,rows,base);
  ORT_RETURN_IF_ERROR(c.RunProgram(program,elements));++dispatches;
  return Status::OK();
}
class ResidualActiveGridV1 final : public Program<ResidualActiveGridV1> {
 public:
  ResidualActiveGridV1():Program{"RW07ResidualActiveGridV2"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(M0_RESIDUAL_TILE_UNIFORMS_V1,
    {"row_begin",ProgramUniformVariableDataType::Uint32},{"row_count",ProgramUniformVariableDataType::Uint32},
    {"weight_flag_offset",ProgramUniformVariableDataType::Uint32},{"grid_rows",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader)const override{
    const auto& extent=shader.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& args=shader.AddOutput("args",ShaderUsage::UseValueTypeAlias);
    ValidOutput(shader,extent);
    shader.MainFunctionBody()<<args.SetByOffset("0u","0u")<<args.SetByOffset("1u","uniforms.grid_rows")
      <<args.SetByOffset("2u","1u")
      <<"if(valid_output()){let n="<<extent.GetByOffset("6u")<<";if(n>uniforms.time_begin){"
      <<"let active_count=min(uniforms.time_count,n-uniforms.time_begin);"
      <<args.SetByOffset("0u","(active_count+31u)/32u")<<"}}";
    return Status::OK();
  }
};
inline Status ResidualSetActiveGridV1(ComputeContextBase& context,ProgramBase& program,const Tensor* extent,
    Tensor* args,const ResidualConvSpecV1& spec,uint32_t begin,uint32_t times,uint32_t row,uint32_t count,
    uint32_t rows,uint64_t& dispatches){
  ORT_RETURN_IF_NOT(args&&args->IsDataType<uint32_t>()&&args->Shape()==TensorShape{3}&&
      args->Location().device.Type()==OrtDevice::GPU&&rows&&rows<=context.DeviceLimits().maxComputeWorkgroupsPerDimension&&
      times/32<=context.DeviceLimits().maxComputeWorkgroupsPerDimension,"RW07 active residual indirect bounds");
  ResidualActiveGridV1 grid;
  grid.AddInput({extent,ProgramTensorMetadataDependency::None}).AddOutput({args,ProgramTensorMetadataDependency::None})
    .SetWorkgroupSize(1).SetDispatchGroupSize(1);
  spec.ApplyPacked(grid,begin,times,row,count);grid.AddUniformVariables({rows});
  ORT_RETURN_IF_ERROR(context.RunProgram(grid));++dispatches;
  program.SetIndirectDispatchTensor(args);
  return Status::OK();
}
inline Status RunResidualPackedTileV1(ComputeContextBase& c,const ProgramInput& input,const ProgramOutput& output,
    const Tensor* extent,const Tensor* weight,const Tensor* bias,Tensor* columns,Tensor* flags,
    const ResidualConvSpecV1& spec,uint32_t time_begin,uint32_t time_count,bool checked,uint64_t& dispatches,
    bool weights_ready=false,uint32_t register_rows=0,Tensor* register_weights=nullptr,Tensor* indirect=nullptr,bool mixed=false) {
  ORT_RETURN_IF_NOT((register_rows==0||register_rows==4||register_rows==8)&&(!register_rows||M0BatchHeavyProgramsV1()),"RW07 register matrix requires bounded batching");
  ORT_RETURN_IF_NOT(ResidualPackedSupportedV1(c.DeviceLimits()),"RW07 packed matrix device requirements");
  ORT_RETURN_IF_NOT(columns&&flags&&(mixed?columns->IsDataType<MLFloat16>():columns->IsDataType<float>())&&flags->IsDataType<uint32_t>()&&
    weight&&weight->IsDataType<float>()&&bias&&bias->IsDataType<float>()&&
    spec.channels>0&&spec.channels<=512&&spec.channel_count>0&&spec.channel_count<=256&&spec.kernel>0&&spec.kernel<=11&&
    spec.dilation>0&&spec.dilation<=5&&spec.output_channel_count>0&&
    spec.output_channel_offset<spec.channels&&spec.output_channel_count<=spec.channels-spec.output_channel_offset&&
    spec.channel_offset<spec.channels&&spec.channel_count<=spec.channels-spec.channel_offset&&
    time_count>0&&time_count<=kResidualPackedTimeV1&&time_count%32==0&&time_begin<spec.output_capacity&&
    weight->Shape().Size()==int64_t(spec.channels)*spec.channels*spec.kernel&&weight->Shape().Size()%4==0,
    "RW07 packed convolution dimensions/types");
  const uint64_t vectors=uint64_t(ResidualPackedInnerV1(spec.channel_count,spec.kernel))*(time_count/4);
  const uint32_t groups=static_cast<uint32_t>(ResidualPackedFlagCountV1(spec.channel_count,spec.kernel,time_count));
  const bool optimized=M0BatchHeavyProgramsV1();
  ORT_RETURN_IF_NOT(uint64_t(columns->Shape().Size())>=vectors*4&&flags->Shape().Size()<=UINT32_MAX&&
    uint64_t(flags->Shape().Size())>=uint64_t(groups)+(optimized?kM0ResidualPackedWeightFlagWordsV1:0),
    "RW07 packed convolution scratch extent");
  const uint32_t flag_base=optimized?static_cast<uint32_t>(flags->Shape().Size())-kM0ResidualPackedWeightFlagWordsV1:0;
  if(register_rows||mixed){
    const uint64_t weight_elements=ResidualRegisterWeightElementsV1(spec.output_channel_count,spec.channel_count,spec.kernel);
    ORT_RETURN_IF_NOT(weight_elements&&register_weights&&(mixed?register_weights->IsDataType<MLFloat16>():register_weights->IsDataType<float>())&&
      uint64_t(register_weights->Shape().Size())>=weight_elements&&
      uint64_t(register_weights->Shape().Size())<=kM0ResidualRegisterWeightScratchBytesV1/4,
      "RW07 register weight storage extent");
  }
  if(optimized&&!weights_ready){
    if(register_rows||mixed)ORT_RETURN_IF_ERROR(PrepareResidualRegisterWeightsV1(c,extent,weight,register_weights,flags,spec,dispatches,mixed));
    else ORT_RETURN_IF_ERROR(PrepareResidualPackedWeightFlagsV1(c,extent,weight,flags,spec,dispatches));
  }
  ResidualPackColumnsV1 pack(optimized,mixed);
  pack.AddInput(ProgramInput(input)).AddInput({extent,ProgramTensorMetadataDependency::None})
    .AddOutput({columns,ProgramTensorMetadataDependency::None,ProgramOutput::Flatten,4})
    .AddOutput({flags,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64);
  const auto limit=c.DeviceLimits().maxComputeWorkgroupsPerDimension;
  ORT_RETURN_IF_NOT(groups&&time_count/32<=limit&&(spec.channel_count*spec.kernel+7)/8<=limit,"RW07 pack workgroup limit");
  spec.ApplyPacked(pack,time_begin,time_count,0,spec.output_channel_count,flag_base);
  if(indirect){
    ORT_RETURN_IF_ERROR(ResidualSetActiveGridV1(c,pack,extent,indirect,spec,time_begin,time_count,0,spec.output_channel_count,
        (spec.channel_count*spec.kernel+7)/8,dispatches));
  }else {const uint32_t dx=std::min(groups,limit);pack.SetDispatchGroupSize(dx,(groups+dx-1)/dx);}
  ORT_RETURN_IF_ERROR(c.RunProgram(pack));++dispatches;
  const uint64_t rounded=(uint64_t(spec.channel_count)*spec.kernel+31)/32*32;
  const uint32_t tile_rows=mixed?kResidualMixedTileRowsV1:register_rows?8u*register_rows:32u;
  const uint32_t row_limit=static_cast<uint32_t>(ResidualPackedProductsPerDispatchV1()/(time_count*rounded)/tile_rows*tile_rows);
  ORT_RETURN_IF_NOT(row_limit>=tile_rows,"RW07 packed matrix work bound");
  for(uint32_t row=0;row<spec.output_channel_count;row+=row_limit){
    const uint32_t count=std::min(row_limit,spec.output_channel_count-row);
    ResidualPackedMatrixV1 matrix(optimized,register_rows,mixed);
    matrix.AddInput(ProgramInput(input)).AddInput({extent,ProgramTensorMetadataDependency::None})
      .AddInput({(register_rows||mixed)?register_weights:weight,ProgramTensorMetadataDependency::None,ProgramInput::Flatten,4})
      .AddInput({bias,ProgramTensorMetadataDependency::None})
      .AddInput({columns,ProgramTensorMetadataDependency::None,ProgramInput::Flatten,4})
      .AddInput({flags,ProgramTensorMetadataDependency::None});
    if(mixed)matrix.AddInput({weight,ProgramTensorMetadataDependency::None,ProgramInput::Flatten,4});
    matrix.AddOutput(ProgramOutput(output))
      .SetWorkgroupSize(8,8);
    spec.ApplyPacked(matrix,time_begin,time_count,row,count,flag_base);
    if(indirect){ORT_RETURN_IF_ERROR(ResidualSetActiveGridV1(c,matrix,extent,indirect,spec,time_begin,time_count,row,count,
        (count+tile_rows-1)/tile_rows,dispatches));}
    else matrix.SetDispatchGroupSize(time_count/32,(count+tile_rows-1)/tile_rows);
    const uint64_t work=uint64_t((count+tile_rows-1)/tile_rows*tile_rows)*time_count*rounded;
    ORT_RETURN_IF_NOT(work<=ResidualPackedProductsPerDispatchV1(),"RW07 packed matrix scheduled work");
    if(!M0BatchHeavyProgramsV1())ORT_RETURN_IF_ERROR(c.RunProgramWithCheckedCompletion(matrix,work));
    else if(checked&&row+count==spec.output_channel_count)ORT_RETURN_IF_ERROR(c.RunProgramWithQueuedCompletion(matrix,work));
    else ORT_RETURN_IF_ERROR(c.RunProgram(matrix,work));
    ++dispatches;
  }
  return Status::OK();
}
inline Status RunResidualConvolutionV1(ComputeContextBase& c,const ProgramInput& input,const ProgramOutput& output,
    const Tensor* extent,const Tensor* weight,const Tensor* bias,Tensor* columns,Tensor* flags,
    const ResidualConvSpecV1& spec,uint64_t& dispatches,bool checked=false,uint32_t forced=0,Tensor* register_weights=nullptr,Tensor* indirect=nullptr) {
  const bool tiled=M0OutputChannelTiledConvV1();
  const bool packed=tiled&&columns&&flags&&ResidualPackedSupportedV1(c.DeviceLimits());
  const bool registers=packed&&M0BatchHeavyProgramsV1()&&register_weights&&
    ResidualRegisterWeightElementsV1(spec.output_channel_count,spec.channel_count,spec.kernel)>0;
  const uint32_t maximum=packed?(registers?64u:16u):std::max(1u,ResidualChannelBlockV1(c.DeviceLimits().maxComputeWorkgroupStorageSize));
  const uint32_t packed_time=ResidualPackedTimeV1(spec.channel_count,spec.kernel);
  ORT_RETURN_IF_NOT(packed_time&&spec.output_channel_count>0,"RW07 residual trial descriptor/work bounds");
  // Time AND every output row belong to each tuning trial. The old32-row
  // denominator let full-width trials grow by the input-channel tile factor.
  const uint64_t trial_rows=(uint64_t(spec.output_channel_count)+63)/64*64;
  const uint64_t trial_inner=(uint64_t(spec.channel_count)*spec.kernel+31)/32*32;
  const uint64_t trial_times=M0HeavyProductsPerDispatchV1()/(trial_rows*trial_inner)/32*32;
  const uint32_t sample_time=uint32_t(std::min<uint64_t>(packed_time,std::max<uint64_t>(32,trial_times)));
  ORT_RETURN_IF_NOT(sample_time>0,"RW07 residual sample work bound");
  const auto original=[&](uint32_t block,uint32_t begin,uint32_t count,bool finish)->Status {
    const GeneratorResidualConv scheduling(M0VectorizedConvV1(),M0TiledConvV1(),tiled,block);
    const uint64_t elements=scheduling.WorkItems(spec.output_channel_count,count?count:spec.output_capacity);
    const uint32_t chunk=scheduling.ChunkElements(spec.channel_count,spec.kernel);
    ORT_RETURN_IF_NOT(elements>0&&elements<=UINT32_MAX&&chunk>0,"RW07 residual convolution virtual range");
    for(uint64_t first=0;first<elements;){
      const uint32_t n=static_cast<uint32_t>(std::min<uint64_t>(chunk,elements-first));
      GeneratorResidualConv program(M0VectorizedConvV1(),M0TiledConvV1(),tiled,block);
      program.AddInput(ProgramInput(input)).AddInputs({{extent,ProgramTensorMetadataDependency::None},
        {weight,ProgramTensorMetadataDependency::None},{bias,ProgramTensorMetadataDependency::None}})
        .AddOutput(ProgramOutput(output));spec.Apply(program,begin,count,uint32_t(first),n);
      const auto groups=(n+64*program.ValuesPerInvocation()-1)/(64*program.ValuesPerInvocation());
      const auto dx=std::min(groups,c.DeviceLimits().maxComputeWorkgroupsPerDimension);
      program.SetWorkgroupSize(64).SetDispatchGroupSize(dx,(groups+dx-1)/dx);
      if(!M0BatchHeavyProgramsV1()||(finish&&first+n==elements))ORT_RETURN_IF_ERROR(c.RunProgramWithCheckedCompletion(program,uint64_t(n)*spec.channel_count*spec.kernel));
      else ORT_RETURN_IF_ERROR(c.RunProgram(program,uint64_t(n)*spec.channel_count*spec.kernel));
      ++dispatches;first+=n;
    }
    return Status::OK();
  };
  bool weights_ready=false,registers_ready=false;
  const auto prepare_registers=[&]()->Status {
    if(!registers_ready){
      ORT_RETURN_IF_ERROR(PrepareResidualRegisterWeightsV1(c,extent,weight,register_weights,flags,spec,dispatches));
      registers_ready=true;weights_ready=true;
    }
    return Status::OK();
  };
  const auto prepare_weights=[&]()->Status {
    if(M0BatchHeavyProgramsV1()&&!weights_ready){
      ORT_RETURN_IF_ERROR(PrepareResidualPackedWeightFlagsV1(c,extent,weight,flags,spec,dispatches));
      weights_ready=true;
    }
    return Status::OK();
  };
  uint32_t chosen=forced?forced:((tiled&&trial_times<32)?maximum:1);
  if(forced){ORT_RETURN_IF_NOT(forced==1||forced==2||forced==4||forced==8||((forced==16||forced==32||forced==64)&&packed&&forced<=maximum)||(forced==128&&registers&&ResidualMixedSupportedV1(c)),"RW07 forced convolution variant");}
  else if(tiled&&trial_times>=32){
    const std::string key="packed-register-active-fma-bounded-trial-v8:"+std::to_string(maximum)+":"+std::to_string(spec.channels)+":"+
      std::to_string(spec.channel_count)+":"+std::to_string(spec.output_channel_count)+":"+
      std::to_string(spec.kernel)+":"+std::to_string(spec.dilation);
    ORT_RETURN_IF_ERROR(ResidualAutotunedBlockV1(c,key,maximum,spec.channel_offset==0,
      [&](uint32_t variant)->Status {
        if(variant==16||variant==32||variant==64){
          if(variant==16)ORT_RETURN_IF_ERROR(prepare_weights());
          else ORT_RETURN_IF_ERROR(prepare_registers());
          return RunResidualPackedTileV1(c,input,output,extent,weight,bias,columns,flags,spec,0,sample_time,true,dispatches,weights_ready,variant==32?4u:variant==64?8u:0u,register_weights);
        }
        return original(variant,0,sample_time,true);
      },chosen));
  }
  const bool mixed_available=registers&&ResidualMixedSupportedV1(c);
  if((!forced&&mixed_available&&trial_times>=32)||forced==128){
    Tensor half_columns(DataTypeImpl::GetType<MLFloat16>(),columns->Shape(),
        columns->MutableDataRaw(),columns->Location());
    Tensor half_weights(DataTypeImpl::GetType<MLFloat16>(),register_weights->Shape(),
        register_weights->MutableDataRaw(),register_weights->Location());
    const uint32_t mixed_time=static_cast<uint32_t>(std::min<uint64_t>(packed_time,
        ResidualPackedProductsPerDispatchV1()/(kResidualMixedTileRowsV1*trial_inner)/32*32));
    ORT_RETURN_IF_NOT(mixed_time>=32,"RW07 mixed matrix scheduled-work envelope");
    bool half_ready=false;
    const auto half_sample=[&](uint32_t begin,uint32_t count,bool finish,bool trial=false)->Status {
      if(!half_ready){
        ORT_RETURN_IF_ERROR(PrepareResidualRegisterWeightsV1(c,extent,weight,&half_weights,flags,spec,dispatches,true));
        half_ready=true;weights_ready=false;registers_ready=false;
      }
      return RunResidualPackedTileV1(c,input,output,extent,weight,bias,&half_columns,flags,spec,begin,count,
          finish,dispatches,true,0,&half_weights,trial?nullptr:indirect,true);
    };
    uint32_t precision=forced==128?2u:0u;
    const std::string precision_key="mixed-half-storage-f32-accum-v2:"+std::to_string(chosen)+":"+
        std::to_string(spec.channels)+":"+std::to_string(spec.channel_count)+":"+
        std::to_string(spec.output_channel_count)+":"+std::to_string(spec.kernel)+":"+std::to_string(spec.dilation);
    if(!precision)precision=c.M0ResidualBlockChoiceV1(precision_key);
    if(precision>2)precision=0;
    if(!precision&&spec.channel_offset==0){
      const auto single_trial=[&](bool half)->Status {
        if(half)return half_sample(0,sample_time,true,true);
        half_ready=false;
        if(chosen!=16&&chosen!=32&&chosen!=64)return original(chosen,0,sample_time,true);
        if(chosen==16)ORT_RETURN_IF_ERROR(prepare_weights());else ORT_RETURN_IF_ERROR(prepare_registers());
        return RunResidualPackedTileV1(c,input,output,extent,weight,bias,columns,flags,spec,0,sample_time,true,
            dispatches,true,chosen==32?4u:chosen==64?8u:0u,register_weights);
      };
      ORT_RETURN_IF_ERROR(single_trial(false));ORT_RETURN_IF_ERROR(single_trial(true));
      std::array<double,2> elapsed{};
      // Both measured variants include packing and one checked completion. Compile
      // outside the timed rounds; reverse their order to reduce ordering bias.
      for(uint32_t round=0;round<2;++round)for(uint32_t i=0;i<2;++i){
        const uint32_t index=round?1u-i:i;
        const auto start=std::chrono::steady_clock::now();ORT_RETURN_IF_ERROR(single_trial(index!=0));
        elapsed[index]+=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
      }
      precision=elapsed[1]<elapsed[0]?2u:1u;
      c.M0RememberResidualBlockV1(precision_key,precision);
      LOGS(c.Logger(),INFO)<<"M0_RW07_RESIDUAL_PRECISION key="<<precision_key<<" selected="<<precision
          <<" fp32_ns="<<elapsed[0]<<" mixed_ns="<<elapsed[1]<<" accumulation=f32 output=f32";
    }
    if(precision==2){
      // The last trial may have overwritten shared scratch with FP32 weights.
      half_ready=false;
      for(uint32_t t=0;t<spec.output_capacity;t+=mixed_time)
        ORT_RETURN_IF_ERROR(half_sample(t,mixed_time,checked&&spec.output_capacity-t<=mixed_time));
      return Status::OK();
    }
    weights_ready=false;registers_ready=false;
  }
  if(chosen!=16&&chosen!=32&&chosen!=64)return original(chosen,0,0,checked);
  if(chosen==16)ORT_RETURN_IF_ERROR(prepare_weights());
  else ORT_RETURN_IF_ERROR(prepare_registers());
  for(uint32_t t=0;t<spec.output_capacity;t+=packed_time){
    ORT_RETURN_IF_ERROR(RunResidualPackedTileV1(c,input,output,extent,weight,bias,columns,flags,spec,t,packed_time,
      checked&&spec.output_capacity-t<=packed_time,dispatches,weights_ready,chosen==32?4u:chosen==64?8u:0u,register_weights,indirect));
  }
  return Status::OK();
}

#undef M0_RESIDUAL_TILE_UNIFORMS_V1
// Original noise and final projection convolutions share ic/k ordering.
class GeneratorChannelConv final : public Program<GeneratorChannelConv> {
 public:
  GeneratorChannelConv() : Program{"RW07GeneratorNoiseProjectionChannelWindowsV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity",ProgramUniformVariableDataType::Uint32},
      {"output_capacity",ProgramUniformVariableDataType::Uint32},
      {"channels",ProgramUniformVariableDataType::Uint32},
      {"minimum",ProgramUniformVariableDataType::Uint32},
      {"maximum",ProgramUniformVariableDataType::Uint32},
      {"modulus",ProgramUniformVariableDataType::Uint32},
      {"remainder",ProgramUniformVariableDataType::Uint32},
      {"input_channels",ProgramUniformVariableDataType::Uint32},
      {"kernel",ProgramUniformVariableDataType::Uint32},
      {"stride",ProgramUniformVariableDataType::Uint32},
      {"padding",ProgramUniformVariableDataType::Uint32},
      {"input_scale",ProgramUniformVariableDataType::Uint32},
      {"input_tail",ProgramUniformVariableDataType::Uint32},
      {"leaky",ProgramUniformVariableDataType::Uint32},
      {"input_channel_offset",ProgramUniformVariableDataType::Uint32},
      {"input_channel_count",ProgramUniformVariableDataType::Uint32},
      {"output_channel_offset",ProgramUniformVariableDataType::Uint32},
      {"output_channel_count",ProgramUniformVariableDataType::Uint32},
      {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("carrier",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("bias",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("accumulator",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e,true);OriginalArithmetic(s);
    auto& f=s.MainFunctionBody();
    f<<"if(global_idx>=uniforms.output_channel_count*uniforms.output_capacity){return;}"
      <<"if(!valid_output()){ "<<y.SetByOffset("global_idx","0.0")<<"return;}"
      <<"let frame=global_idx%uniforms.output_capacity;let count="<<e.GetByOffset("6u")<<";"
      <<"let input_length=uniforms.input_scale*count+uniforms.input_tail;"
      <<"if(frame>=count||input_length>uniforms.input_capacity||"
      <<"uniforms.input_channel_offset>=uniforms.input_channels||uniforms.input_channel_count==0u||"
      <<"uniforms.input_channel_count>uniforms.input_channels-uniforms.input_channel_offset||"
      <<"uniforms.output_channel_offset>=uniforms.channels||uniforms.output_channel_count==0u||"
      <<"uniforms.output_channel_count>uniforms.channels-uniforms.output_channel_offset){"
      <<y.SetByOffset("global_idx","0.0")<<"return;}"
      <<"let oc=uniforms.output_channel_offset+global_idx/uniforms.output_capacity;"
      <<"var total:f32=0.0;if(uniforms.input_channel_offset!=0u){total="<<y.GetByOffset("global_idx")<<";}"
      <<"for(var local_ic=0u;local_ic<uniforms.input_channel_count;local_ic++){"
      <<"let ic=uniforms.input_channel_offset+local_ic;for(var k=0u;k<uniforms.kernel;k++){"
      <<"let pos=i32(frame*uniforms.stride+k)-i32(uniforms.padding);var value:f32=0.0;"
      <<"if(pos>=0&&u32(pos)<input_length){value="<<x.GetByOffset("local_ic*uniforms.input_capacity+u32(pos)")<<";"
      <<"if(uniforms.leaky!=0u){let negative=original_mul(value,bitcast<f32>(0x3c23d70au));"
      <<"value=select(negative,value,value>=0.0);}}"
      <<"let product=original_mul(value,"<<w.GetByOffset("(oc*uniforms.input_channels+ic)*uniforms.kernel+k")<<");"
      <<"total=original_add(total,product);}}"
      <<"if(uniforms.input_channel_offset+uniforms.input_channel_count==uniforms.input_channels){"
      <<"total=original_add(total,"<<b.GetByOffset("oc")<<");}"<<y.SetByOffset("global_idx","total");
    return Status::OK();
  }
};
// The last convolution accumulates directly in this sole read_write output.
class GeneratorResidualAdd final : public Program<GeneratorResidualAdd> {
 public:
  GeneratorResidualAdd() : Program{"RW07ResidualWindowInPlaceAddV3"} {}
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
      {"tile_channels", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& prior=s.AddInput("prior",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("output",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e,true);
    OriginalArithmetic(s);
    auto& f=s.MainFunctionBody();
    f << "if(global_idx>=uniforms.tile_channels*uniforms.output_capacity){return;}"
      << "if(!valid_output()){ " << y.SetByOffset("global_idx","0.0") << "return;}"
      << "let t=global_idx%uniforms.output_capacity;if(t>=" << e.GetByOffset("6u") << "){"
      << y.SetByOffset("global_idx","0.0") << "return;}"
      << "let oc=global_idx/uniforms.output_capacity;"
      << y.SetByOffset("global_idx","original_add("+y.GetByOffset("global_idx")+","+
          prior.GetByOffset("oc*uniforms.residual_capacity+t")+")");
    return Status::OK();
  }
};
}  // namespace onnxruntime::webgpu::m0_residual_stages

// Shared actual decoder payloads used by the native callers and shader tests.
namespace onnxruntime::webgpu::m0_decoder_stages {
using m0_residual_stages::OriginalArithmetic;
inline void PayloadGuard(ShaderHelper& s, const ShaderVariableHelper& e,bool share_workgroup=false) {
  const bool shared=share_workgroup&&M0BatchHeavyProgramsV1();
  auto& f = s.AdditionalImplementation();
  f << "fn " << (shared?"validate_output_descriptor":"valid_output") << "() -> bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=3u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("4u") << "!=select(1024u,512u," << GetElementAt("uniforms.identity", "8u", 10) << "==24u)||" << e.GetByOffset("5u") << "!=0u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let d=" << e.GetByOffset("6u") << ";return d>0u&&d<=select(102400u,204800u," << GetElementAt("uniforms.identity", "8u", 10) << "==24u)&&d<=uniforms.final_capacity&&"
    << e.GetByOffset("18u") << "==select(1024u,512u," << GetElementAt("uniforms.identity", "8u", 10) << "==24u)*d&&(" << GetElementAt("uniforms.identity", "8u", 10) << "!=24u||d%2u==0u);}\n";
  if(shared)m0_residual_stages::ShareOutputValidityV1(s);
}
class DecoderPoolTile final : public Program<DecoderPoolTile> {
 public:
  DecoderPoolTile() : Program{"RW07DecoderResidualDepthwisePoolLocalChannelTileV2"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"channel_offset", ProgramUniformVariableDataType::Uint32},
      {"channel_count", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const auto& b = s.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s, e,true);
    OriginalArithmetic(s);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels*uniforms.output_capacity){return;}"
      << y.SetByOffset("global_idx", "0.0") << "if(!valid_output()){return;}"
      << "let d=" << e.GetByOffset("6u") << ";let t=global_idx%uniforms.output_capacity;"
      << "let local_ch=global_idx/uniforms.output_capacity;"
      << "if(t>=d||local_ch>=uniforms.channel_count||uniforms.channel_offset>=1090u||"
      << "uniforms.channel_count>uniforms.channels||uniforms.channel_count>1090u-uniforms.channel_offset){return;}"
      << "let ch=uniforms.channel_offset+local_ch;var sum:f32=0.0;"
      // Original CPU Col2im visits kernel taps in ascending order.
      << "for(var k=0u;k<3u;k++){"
      << "let numerator=i32(t)+1-i32(k);if(numerator>=0&&numerator%2==0){"
      << "let source_t=u32(numerator/2);if(source_t<d/2u&&source_t<uniforms.input_capacity){"
      << "let product:f32=original_mul(" << x.GetByOffset("local_ch*uniforms.input_capacity+source_t") << ","
      << w.GetByOffset("ch*3u+k") << ");sum=original_add(sum,product);}}}let result:f32=original_add(sum,"
      << b.GetByOffset("ch") << ");" << y.SetByOffset("global_idx", "result");
    return Status::OK();
  }
};
class DecoderConvTile final : public Program<DecoderConvTile> {
 public:
  explicit DecoderConvTile(bool bias,bool finite=M0BatchHeavyProgramsV1(),bool four_time=true,bool vector_loads=true)
      :Program{finite&&four_time?(vector_loads?"RW07DecoderResidualConvFourTimeGatherV6":"RW07DecoderResidualConvFourTimeWindowV5"):finite?"RW07DecoderResidualConvFiniteWindowV4":"RW07DecoderResidualConvOutputWindowV3"},
       bias_(bias),finite_(finite),four_time_(finite&&four_time),vector_loads_(vector_loads) {}
  uint64_t WorkItems(uint32_t rows,uint32_t capacity)const{
    return uint64_t(rows)*(four_time_?capacity/4u+uint32_t(capacity%4u!=0):capacity);
  }
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"input_channels", ProgramUniformVariableDataType::Uint32},
      {"output_channels", ProgramUniformVariableDataType::Uint32},
      {"width", ProgramUniformVariableDataType::Uint32},
      {"channel_offset", ProgramUniformVariableDataType::Uint32},
      {"channel_count", ProgramUniformVariableDataType::Uint32},
      {"output_offset", ProgramUniformVariableDataType::Uint32},
      {"output_count", ProgramUniformVariableDataType::Uint32},
      {"nearest", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("pool_tile",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* b=bias_?&s.AddInput("bias",ShaderUsage::UseValueTypeAlias):nullptr;
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("accumulator",ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s,e,true);OriginalArithmetic(s);
    if(four_time_)return FourTimeShader(s,x,w,b,e,y);
    auto& f=s.MainFunctionBody();
    if(finite_){
      s.AdditionalImplementation()<<"var<workgroup> decoder_flags:array<u32,64>;var<workgroup> decoder_safe:u32;";
      f<<"let elements=uniforms.output_count*uniforms.output_capacity;"
        <<"if(workgroup_idx>=elements/64u+u32(elements%64u!=0u)){return;}"
        <<"var bad=1u;if(valid_output()&&uniforms.channel_count>0u&&uniforms.channel_count<=64u&&"
        <<"uniforms.channel_offset<uniforms.input_channels&&uniforms.channel_count<=uniforms.input_channels-uniforms.channel_offset&&"
        <<"uniforms.output_offset<uniforms.output_channels&&uniforms.output_count<=uniforms.output_channels-uniforms.output_offset&&"
        <<"(uniforms.width==1u||uniforms.width==3u)&&uniforms.input_capacity>0u){"
        <<"bad=0u;let d="<<e.GetByOffset("6u")<<";let first=workgroup_idx*64u;let last=min(first+63u,elements-1u);"
        <<"let first_row=first/uniforms.output_capacity;let last_row=last/uniforms.output_capacity;"
        <<"var lo=first%uniforms.output_capacity;var hi=last%uniforms.output_capacity;"
        <<InputSafetyScan(x)
        <<"let row_elements=uniforms.channel_count*uniforms.width;"
        <<"for(var index=local_idx;index<(last_row-first_row+1u)*row_elements;index+=64u){"
        <<"let row=uniforms.output_offset+first_row+index/row_elements;"
        <<"let ic=uniforms.channel_offset+(index/uniforms.width)%uniforms.channel_count;let k=index%uniforms.width;"
        <<"let value="<<w.GetByOffset("(row*uniforms.input_channels+ic)*uniforms.width+k")<<";"
        <<"if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){bad=1u;}}"
        <<"if(uniforms.channel_offset!=0u){let at=first+local_idx;if(at<=last&&at%uniforms.output_capacity<d){"
        <<"let seed="<<y.GetByOffset("at")<<";if((bitcast<u32>(seed)&0x7fffffffu)>0x7b800000u){bad=1u;}}}}"
        <<"decoder_flags[local_idx]=bad;workgroupBarrier();if(local_idx==0u){var combined=0u;"
        <<"for(var lane=0u;lane<64u;lane++){combined|=decoder_flags[lane];}decoder_safe=1u-min(combined,1u);}"
        <<"let finite_safe=workgroupUniformLoad(&decoder_safe)!=0u;";
    }
    f<<"if(global_idx>=uniforms.output_count*uniforms.output_capacity){return;}"
      <<"if(!valid_output()){ "<<y.SetByOffset("global_idx","0.0")<<"return;}"
      <<"let d="<<e.GetByOffset("6u")<<";let t=global_idx%uniforms.output_capacity;"
      <<"let oc=uniforms.output_offset+global_idx/uniforms.output_capacity;"
      <<"if(t>=d||uniforms.channel_offset>=uniforms.input_channels||uniforms.channel_count==0u||"
      <<"uniforms.channel_count>64u||uniforms.channel_count>uniforms.input_channels-uniforms.channel_offset){"
      <<y.SetByOffset("global_idx","0.0")<<"return;}"
      // Continue the scalar sum; summing independent tile totals would reorder it.
      <<"var sum:f32=0.0;if(uniforms.channel_offset!=0u){sum="<<y.GetByOffset("global_idx")<<";}";
    const auto dot=[&](bool fast){
      f<<"for(var local_ic=0u;local_ic<uniforms.channel_count;local_ic++){"
       <<"let ic=uniforms.channel_offset+local_ic;for(var k=0u;k<uniforms.width;k++){"
       <<"let pos=i32(t)+i32(k)-i32(uniforms.width/2u);var value:f32=0.0;"
       <<"if(pos>=0&&u32(pos)<d){let source_t=select(u32(pos),u32(pos)/2u,uniforms.nearest!=0u);"
       <<"if(source_t<uniforms.input_capacity){value="
       <<x.GetByOffset("local_ic*uniforms.input_capacity+source_t")<<";}}"
       <<"let weight="<<w.GetByOffset("(oc*uniforms.input_channels+ic)*uniforms.width+k")<<";";
      if(fast)f<<"let product=value*weight;sum=sum+product;}}";
      else f<<"let product=original_mul(value,weight);sum=original_add(sum,product);}}";
    };
    // At most192 products <=2^100 plus a seed <=2^120 cannot overflowFP32.
    // Keep the complete original exceptional path and the ic/k reduction order.
    if(finite_){f<<"if(finite_safe){";dot(true);f<<"}else{";dot(false);f<<"}";}
    else dot(false);
    if(bias_) f<<"if(uniforms.channel_offset+uniforms.channel_count==uniforms.input_channels){sum=original_add(sum,"
               <<b->GetByOffset("oc")<<");}";
    f<<y.SetByOffset("global_idx","sum");
    return Status::OK();
  }
 private:
  static std::string InputSafetyScan(const ShaderVariableHelper& x){
    OStringStream code(1024);
    // A group crossing one row boundary owns a suffix and a prefix, not the
    // whole time axis. More than two rows includes a complete interior row.
    code<<"for(var part=0u;part<2u;part++){var low=lo;var high=hi;"
      <<"if(first_row==last_row){if(part!=0u){continue;}}"
      <<"else if(last_row-first_row>1u){if(part!=0u){continue;}low=0u;high=uniforms.output_capacity-1u;}"
      <<"else{if(part==0u){high=uniforms.output_capacity-1u;}else{low=0u;}}"
      <<"if(low<d){high=min(high,d-1u);let radius=uniforms.width/2u;"
      <<"let first_pos=u32(max(0,i32(low)-i32(radius)));let last_pos=min(d-1u,high+radius);"
      <<"let begin=select(first_pos,first_pos/2u,uniforms.nearest!=0u);"
      <<"let end=min(uniforms.input_capacity-1u,select(last_pos,last_pos/2u,uniforms.nearest!=0u));"
      <<"if(begin<=end){let span=end-begin+1u;for(var index=local_idx;index<uniforms.channel_count*span;index+=64u){"
      <<"let value="<<x.GetByOffset("(index/span)*uniforms.input_capacity+begin+index%span")<<";"
      <<"if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){bad=1u;}}}}}";
    return std::move(code).str();
  }
  Status FourTimeShader(ShaderHelper& s,const ShaderVariableHelper& x,const ShaderVariableHelper& w,
      const ShaderVariableHelper* b,const ShaderVariableHelper& e,const ShaderVariableHelper& y)const{
    s.AdditionalImplementation()<<"var<workgroup> decoder_flags:array<u32,64>;var<workgroup> decoder_safe:u32;var<workgroup> decoder_enabled:u32;";
    auto& f=s.MainFunctionBody();
    f<<"let vectors=uniforms.output_capacity/4u+u32(uniforms.output_capacity%4u!=0u);if(vectors==0u){return;}"
      <<"let elements=uniforms.output_count*vectors;if(workgroup_idx>=elements/64u+u32(elements%64u!=0u)){return;}"
      <<"let geometry=valid_output()&&uniforms.channel_count>0u&&uniforms.channel_count<=64u&&"
      <<"uniforms.channel_offset<uniforms.input_channels&&uniforms.channel_count<=uniforms.input_channels-uniforms.channel_offset&&"
      <<"uniforms.output_offset<uniforms.output_channels&&uniforms.output_count<=uniforms.output_channels-uniforms.output_offset&&"
      <<"(uniforms.width==1u||uniforms.width==3u)&&uniforms.input_capacity>0u;"
      <<"if(local_idx==0u){decoder_enabled=select(0u,1u,geometry);}var bad=1u;"
      <<"if(geometry){bad=0u;let d="<<e.GetByOffset("6u")<<";let first=workgroup_idx*64u;let last=min(first+63u,elements-1u);"
      <<"let first_row=first/vectors;let last_row=last/vectors;var lo=(first%vectors)*4u;"
      <<"var hi=min((last%vectors)*4u+3u,uniforms.output_capacity-1u);"
      <<InputSafetyScan(x)
      <<"let row_elements=uniforms.channel_count*uniforms.width;for(var index=local_idx;index<(last_row-first_row+1u)*row_elements;index+=64u){"
      <<"let row=uniforms.output_offset+first_row+index/row_elements;let ic=uniforms.channel_offset+(index/uniforms.width)%uniforms.channel_count;"
      <<"let value="<<w.GetByOffset("(row*uniforms.input_channels+ic)*uniforms.width+index%uniforms.width")<<";"
      <<"if((bitcast<u32>(value)&0x7fffffffu)>0x58800000u){bad=1u;}}"
      <<"if(uniforms.channel_offset!=0u){let at=first+local_idx;if(at<=last){let row=at/vectors;let t=(at%vectors)*4u;"
      <<"for(var lane=0u;lane<4u;lane++){if(t+lane<min(d,uniforms.output_capacity)){let seed="
      <<y.GetByOffset("row*uniforms.output_capacity+t+lane")<<";if((bitcast<u32>(seed)&0x7fffffffu)>0x7b800000u){bad=1u;}}}}}}"
      <<"decoder_flags[local_idx]=bad;workgroupBarrier();if(local_idx==0u){var combined=0u;for(var lane=0u;lane<64u;lane++){combined|=decoder_flags[lane];}"
      <<"decoder_safe=1u-min(combined,1u);}let finite_safe=workgroupUniformLoad(&decoder_safe)!=0u;"
      <<"let enabled=workgroupUniformLoad(&decoder_enabled)!=0u;if(global_idx>=elements){return;}"
      <<"let row=global_idx/vectors;let t=(global_idx%vectors)*4u;let oc=uniforms.output_offset+row;let d="<<e.GetByOffset("6u")<<";"
      <<"if(!enabled||t>=d){for(var lane=0u;lane<4u;lane++){if(t+lane<uniforms.output_capacity){"
      <<y.SetByOffset("row*uniforms.output_capacity+t+lane","0.0")<<"}}return;}"
      <<"var sum=vec4<f32>(0.0);if(uniforms.channel_offset!=0u){for(var lane=0u;lane<4u;lane++){if(t+lane<min(d,uniforms.output_capacity)){"
      <<"sum[lane]="<<y.GetByOffset("row*uniforms.output_capacity+t+lane")<<";}}}";
    const auto dot=[&](bool fast){
      f<<"for(var local_ic=0u;local_ic<uniforms.channel_count;local_ic++){let ic=uniforms.channel_offset+local_ic;"
       <<"for(var k=0u;k<uniforms.width;k++){let weight="<<w.GetByOffset("(oc*uniforms.input_channels+ic)*uniforms.width+k")<<";"
       <<"var value=vec4<f32>(0.0);";
      if(vector_loads_){
        f<<"var gathered=false;let first_pos=i32(t)+i32(k)-i32(uniforms.width/2u);"
          <<"if(t+3u<min(d,uniforms.output_capacity)&&first_pos>=0&&u32(first_pos)+3u<d){"
          <<"let begin=select(u32(first_pos),u32(first_pos)/2u,uniforms.nearest!=0u);"
          <<"let end=select(u32(first_pos)+3u,(u32(first_pos)+3u)/2u,uniforms.nearest!=0u);"
          <<"if(end<uniforms.input_capacity){let base=local_ic*uniforms.input_capacity+begin;"
          <<"let a="<<x.GetByOffset("base")<<";let b="<<x.GetByOffset("base+1u")<<";"
          <<"if(uniforms.nearest==0u){value=vec4<f32>(a,b,"<<x.GetByOffset("base+2u")<<","<<x.GetByOffset("base+3u")<<");}"
          <<"else if(u32(first_pos)%2u==0u){value=vec4<f32>(a,a,b,b);}"
          <<"else{value=vec4<f32>(a,b,b,"<<x.GetByOffset("base+2u")<<");}gathered=true;}}"
          <<"if(!gathered){";
      }
      f<<"for(var lane=0u;lane<4u;lane++){let pos=i32(t+lane)+i32(k)-i32(uniforms.width/2u);"
       <<"if(t+lane<min(d,uniforms.output_capacity)&&pos>=0&&u32(pos)<d){let source_t=select(u32(pos),u32(pos)/2u,uniforms.nearest!=0u);"
       <<"if(source_t<uniforms.input_capacity){value[lane]="<<x.GetByOffset("local_ic*uniforms.input_capacity+source_t")<<";}}}";
      if(vector_loads_)f<<"}";
      if(fast)f<<"let product=value*vec4<f32>(weight);sum=sum+product;}}";
      else f<<"for(var lane=0u;lane<4u;lane++){let product=original_mul(value[lane],weight);sum[lane]=original_add(sum[lane],product);}}}";
    };
    // Four independent sums retain exactly the original ic/k order per sample.
    f<<"if(finite_safe){";dot(true);f<<"}else{";dot(false);f<<"}";
    if(bias_)f<<"if(uniforms.channel_offset+uniforms.channel_count==uniforms.input_channels){let bias="<<b->GetByOffset("oc")
      <<";for(var lane=0u;lane<4u;lane++){sum[lane]=original_add(sum[lane],bias);}}";
    f<<"for(var lane=0u;lane<4u;lane++){if(t+lane<uniforms.output_capacity){var value=0.0;if(t+lane<d){value=sum[lane];}"
      <<y.SetByOffset("row*uniforms.output_capacity+t+lane","value")<<"}}";
    return Status::OK();
  }
  const bool bias_,finite_,four_time_,vector_loads_;
};
class DecoderResidualAdd final : public Program<DecoderResidualAdd> {
 public:
  DecoderResidualAdd() : Program{"RW07DecoderResidualInPlaceResidualAddWindowV3"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& skip = s.AddInput("skip", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s, e,true);
    OriginalArithmetic(s);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels*uniforms.capacity){return;}"
      << "if(!valid_output()){ " << y.SetByOffset("global_idx", "0.0") << "return;}"
      << "if(global_idx%uniforms.capacity>=" << e.GetByOffset("6u") << "){"
      << y.SetByOffset("global_idx", "0.0") << "return;}"
      << "let added:f32=original_add(" << y.GetByOffset("global_idx") << "," << skip.GetByOffset("global_idx") << ");"
      << y.SetByOffset("global_idx", "original_div(added,bitcast<f32>(0x3fb504f3u))");
    return Status::OK();
  }
};
}  // namespace onnxruntime::webgpu::m0_decoder_stages

namespace onnxruntime::webgpu::m0_decoder_concat_stages {
inline void ValidOutput(ShaderHelper& s, const ShaderVariableHelper& e,bool share_workgroup=false) {
  const bool shared=share_workgroup&&M0BatchHeavyProgramsV1();
  auto& f = s.AdditionalImplementation();
  f << "fn " << (shared?"validate_output_descriptor":"valid_output") << "()->bool {if(" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u")
    << "!=3u||" << e.GetByOffset("20u") << "!=0u||" << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("4u") << "!=1090u||" << e.GetByOffset("5u") << "!=0u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let d=" << e.GetByOffset("6u") << ";return d>0u&&d<=102400u&&d<=uniforms.capacity&&"
    << "d<=uniforms.text_capacity&&d<=uniforms.f0_capacity&&d<=uniforms.n_capacity&&"
    << e.GetByOffset("18u") << "==1090u*d;}\n";
  if(shared)m0_residual_stages::ShareOutputValidityV1(s);
}
class Concat final : public Program<Concat> {
 public:
  Concat() : Program{"RW07OriginalDecoderConcatV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"text_capacity", ProgramUniformVariableDataType::Uint32}, {"f0_capacity", ProgramUniformVariableDataType::Uint32},
      {"n_capacity", ProgramUniformVariableDataType::Uint32}, {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("aligned_text", ShaderUsage::UseValueTypeAlias);
    const auto& f0 = s.AddInput("f0_projection", ShaderUsage::UseValueTypeAlias);
    const auto& n = s.AddInput("n_projection", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("decoder_input", ShaderUsage::UseValueTypeAlias);
    ValidOutput(s, e,true);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=1090u*uniforms.capacity){return;}" << y.SetByOffset("global_idx", "0.0")
      << "if(!valid_output()){return;}let t=global_idx%uniforms.capacity;let ch=global_idx/uniforms.capacity;"
      << "if(t>=" << e.GetByOffset("6u") << "){return;}if(ch<1024u){"
      << y.SetByOffset("global_idx",x.GetByOffset("ch*uniforms.text_capacity+t")) << "}else if(ch<1088u){"
      << y.SetByOffset("global_idx",f0.GetByOffset("(ch-1024u)*uniforms.f0_capacity+t")) << "}else{"
      << y.SetByOffset("global_idx",n.GetByOffset("(ch-576u)*uniforms.n_capacity+t")) << "}";
    return Status::OK();
  }
};
}  // namespace onnxruntime::webgpu::m0_decoder_concat_stages
