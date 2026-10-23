// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_trig_v1.h"

// Both original generator and noise residual paths use the same scalar stage
// ordering. Roles, extents, parameters and capacities remain caller supplied.
namespace onnxruntime::webgpu::m0_residual_stages {
inline void ValidOutput(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_output()->bool {if(" << e.GetByOffset("0u") << "!=2u||"
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
inline Status ResidualTileChannels(ComputeContext& c, uint32_t input_capacity, uint32_t output_capacity,
                                   uint32_t channels, uint32_t& tile_channels) {
  const auto& limits=c.DeviceLimits();
  const uint64_t alignment=limits.minStorageBufferOffsetAlignment;
  const uint64_t capacity=std::max(input_capacity,output_capacity);
  ORT_RETURN_IF_NOT(alignment>0&&alignment%sizeof(float)==0&&capacity>0&&channels>0&&
      capacity*channels<=UINT32_MAX,"RW07 residual tile descriptor/index limit");
  const uint64_t row_bytes=capacity*sizeof(float),prefix=alignment-sizeof(float);
  ORT_RETURN_IF_NOT(row_bytes+prefix<=limits.maxStorageBufferBindingSize,
      "RW07 residual channel row exceeds a physical binding window");
  const uint64_t target=std::min<uint64_t>(limits.maxStorageBufferBindingSize,
      std::max<uint64_t>(8ull*1024*1024,row_bytes+prefix));
  const uint64_t count=std::min<uint64_t>(std::min<uint32_t>(32,channels),(target-prefix)/row_bytes);
  ORT_RETURN_IF_NOT(count>0,"RW07 residual empty channel tile");
  tile_channels=static_cast<uint32_t>(count);
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
  GeneratorNormStatistics() : Program{"RW07ResidualNormStatisticsCompensatedV2"} {}
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
};
// Each invocation evaluates at most one sample; cap the number per dispatch.
inline constexpr uint32_t kResidualSnakeElementsPerDispatchV1 = 16384;
class GeneratorNormSnake final : public Program<GeneratorNormSnake> {
 public:
  GeneratorNormSnake() : Program{"RW07ResidualNormSnakeElementChunksV3"} {}
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
    ValidOutput(s,e);
    OriginalArithmetic(s);
    s.AdditionalImplementation() << m0_trig_detail::Trig;
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
      << "let a=" << coefficients.GetByOffset("channel*6u+2u") << ";"
      << "let reciprocal_a=" << coefficients.GetByOffset("channel*6u+3u") << ";"
      << "let centered=original_sub(" << x.GetByOffset("local_channel*uniforms.input_capacity+t") << ",mean);"
      << "let normalized=original_div(centered,denominator);let affine=original_add(original_mul(gamma,normalized),beta);"
      << "let angle=original_mul(a,affine);let bits=bitcast<u32>(angle)&0x7fffffffu;var sine:f32=original_f32_from_bits(0x7fc00000u);"
      << "if(bits<0x7f800000u){sine=rw07_sin(angle);}let square=original_mul(sine,sine);"
      << "let result=original_add(affine,original_mul(reciprocal_a,square));"
      << y.SetByOffset("at","result");
    return Status::OK();
  }
};
// Bound scalar products encoded by one residual convolution dispatch. This
// is a scheduling upper bound, not measured GPU instructions, FLOPs or time.
inline constexpr uint32_t kResidualProductsPerDispatchV1 = 1u << 22;
inline uint32_t ResidualConvElementsPerDispatchV1(uint32_t input_channels, uint32_t kernel) {
  const uint64_t products_per_element = uint64_t(input_channels) * kernel;
  if (products_per_element == 0 || products_per_element > kResidualProductsPerDispatchV1) return 0;
  const auto elements = static_cast<uint32_t>(kResidualProductsPerDispatchV1 / products_per_element);
  return elements >= 64 ? elements / 64 * 64 : elements;
}
class GeneratorResidualConv final : public Program<GeneratorResidualConv> {
 public:
  GeneratorResidualConv() : Program{"RW07ResidualConvChannelElementChunksV4"} {}
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
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("carrier",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("weight",ShaderUsage::UseValueTypeAlias);
    const auto& b=s.AddInput("bias",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("output",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e);
    OriginalArithmetic(s);
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
};
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
    ValidOutput(s,e);OriginalArithmetic(s);
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
    ValidOutput(s,e);
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
