// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original F0.1/F0.2 or N.1/N.2 and scalar projection, preserving GPU-owned D.
#include <algorithm>
#include <array>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"

namespace onnxruntime::webgpu {
namespace {
constexpr uint32_t kTailTileChannels=64;
ProgramInput Words(const Tensor* t) {
  ProgramInput p(t, ProgramTensorMetadataDependency::None, TensorShape{32}, 1);
  p.var_type = ProgramVariableDataType::Uint32;
  return p;
}
ProgramOutput Words(Tensor* t) {
  ProgramOutput p(t, ProgramTensorMetadataDependency::None, TensorShape{32}, 1);
  p.var_type = ProgramVariableDataType::Uint32;
  return p;
}
void PayloadGuard(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_output() -> bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=3u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("4u") << "!=1u||" << e.GetByOffset("5u") << "!=0u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let d=" << e.GetByOffset("6u") << ";return d>0u&&d%2u==0u&&d<=uniforms.final_capacity&&"
    << e.GetByOffset("18u") << "==d;}\n";
}
class TailGuard final : public Program<TailGuard> {
 public:
  TailGuard() : Program{"RW07PredictorTailOwnedGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("source", ShaderUsage::UseValueTypeAlias);
    const auto& o = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << o.SetByOffset("i", "0u") << "}"
      << o.SetByOffset("0u", "2u") << o.SetByOffset("1u", "3u")
      << "for(var i=0u;i<10u;i++){" << o.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "if(uniforms.pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||"
      << e.GetByOffset("1u") << "!=3u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity", "i", 10) << "){" << o.SetByOffset("20u", "103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){" << o.SetByOffset("20u", "103u") << "return;}"
      << o.SetByOffset("20u", "status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("4u") << "!=512u||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}"
      << "let d=" << e.GetByOffset("6u") << ";if(d==0u||d>8388607u||"
      << e.GetByOffset("18u") << "!=512u*d){" << o.SetByOffset("20u", "103u") << "return;}"
      << "if(d>uniforms.input_capacity||d>uniforms.final_capacity/2u){"
      << o.SetByOffset("20u", "102u") << "return;}"
      << o.SetByOffset("2u", "1u") << o.SetByOffset("4u", "1u")
      << o.SetByOffset("6u", "2u*d") << o.SetByOffset("18u", "2u*d") << o.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class TailNormStyle final : public Program<TailNormStyle> {
 public:
  TailNormStyle() : Program{"RW07PredictorTailNormStyleChannelWindowV2"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"features", ProgramUniformVariableDataType::Uint32},
      {"extent_divisor", ProgramUniformVariableDataType::Uint32},
      {"channel_offset", ProgramUniformVariableDataType::Uint32},
      {"channel_count", ProgramUniformVariableDataType::Uint32},
      {"tile_channels", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& style = s.AddInput("style", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const auto& b = s.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.tile_channels){return;}for(var t=0u;t<uniforms.output_capacity;t++){"
      << y.SetByOffset("global_idx*uniforms.output_capacity+t", "0.0") << "}if(!valid_output()){return;}"
      << "if(global_idx>=uniforms.channel_count||uniforms.channel_offset>=uniforms.channels||uniforms.channel_count>uniforms.channels-uniforms.channel_offset){return;}"
      << "let ch=uniforms.channel_offset+global_idx;"
      << "let d=" << e.GetByOffset("6u") << "/uniforms.extent_divisor;"
      << "if(d>uniforms.input_capacity||d>uniforms.output_capacity){return;}var sum:f32=0.0;"
      << "for(var t=0u;t<d;t++){sum=sum+" << x.GetByOffset("global_idx*uniforms.input_capacity+t") << ";}"
      << "let mean:f32=sum/f32(d);var squares:f32=0.0;for(var t=0u;t<d;t++){let centered:f32="
      << x.GetByOffset("global_idx*uniforms.input_capacity+t") << "-mean;let square:f32=centered*centered;"
      << "squares=squares+square;}let variance:f32=squares/f32(d);"
      << "let shifted:f32=variance+bitcast<f32>(0x3727c5acu);let denominator:f32=sqrt(shifted);"
      << "var gamma_sum:f32=0.0;var beta_sum:f32=0.0;for(var k=0u;k<uniforms.features;k++){"
      << "let style_value=" << style.GetByOffset("128u+k") << ";let gp:f32=style_value*"
      << w.GetByOffset("ch*uniforms.features+k") << ";let bp:f32=style_value*"
      << w.GetByOffset("(ch+uniforms.channels)*uniforms.features+k") << ";"
      << "gamma_sum=gamma_sum+gp;beta_sum=beta_sum+bp;}let gamma_fc:f32=gamma_sum+"
      << b.GetByOffset("ch") << ";let gamma:f32=gamma_fc+1.0;let beta:f32=beta_sum+"
      << b.GetByOffset("ch+uniforms.channels") << ";for(var t=0u;t<d;t++){"
      << "let centered:f32=" << x.GetByOffset("global_idx*uniforms.input_capacity+t") << "-mean;"
      << "let normalized:f32=centered/denominator;let product:f32=gamma*normalized;let affine:f32=product+beta;"
      << "let negative:f32=affine*bitcast<f32>(0x3e4ccccdu);"
      << y.SetByOffset("global_idx*uniforms.output_capacity+t", "select(negative,affine,affine>=0.0)") << "}";
    return Status::OK();
  }
};
class TailConv final : public Program<TailConv> {
 public:
  TailConv(bool bias, bool nearest) : Program{"RW07PredictorTailConvChannelWindowsV2"}, bias_(bias), nearest_(nearest) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"input_channels", ProgramUniformVariableDataType::Uint32},
      {"output_channels", ProgramUniformVariableDataType::Uint32},
      {"width", ProgramUniformVariableDataType::Uint32},
      {"input_offset", ProgramUniformVariableDataType::Uint32},
      {"input_count", ProgramUniformVariableDataType::Uint32},
      {"output_offset", ProgramUniformVariableDataType::Uint32},
      {"output_count", ProgramUniformVariableDataType::Uint32},
      {"extent_divisor", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* b = bias_ ? &s.AddInput("bias", ShaderUsage::UseValueTypeAlias) : nullptr;
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.output_count*uniforms.output_capacity){return;}"
      << "if(!valid_output()){"<<y.SetByOffset("global_idx","0.0")<<"return;}"
      << "let d=" << e.GetByOffset("6u") << "/uniforms.extent_divisor;let t=global_idx%uniforms.output_capacity;"
      << "let oc=uniforms.output_offset+global_idx/uniforms.output_capacity;if(t>=d){"<<y.SetByOffset("global_idx","0.0")<<"return;}var sum:f32=0.0;"
      << "if(uniforms.input_offset!=0u){sum="<<y.GetByOffset("global_idx")<<";}"
      << "for(var local_ic=0u;local_ic<uniforms.input_count;local_ic++){let ic=uniforms.input_offset+local_ic;for(var k=0u;k<uniforms.width;k++){"
      << "let pos=i32(t)+i32(k)-i32(uniforms.width/2u);var value:f32=0.0;"
      << "if(pos>=0&&u32(pos)<d){let source_t=" << (nearest_ ? "u32(pos)/2u" : "u32(pos)") << ";"
      << "if(source_t<uniforms.input_capacity){value=" << x.GetByOffset("local_ic*uniforms.input_capacity+source_t") << ";}}"
      << "let product:f32=value*" << w.GetByOffset("(oc*uniforms.input_channels+ic)*uniforms.width+k")
      << ";sum=sum+product;}}";
    if (bias_) f << "if(uniforms.input_offset+uniforms.input_count==uniforms.input_channels){sum=sum+" << b->GetByOffset("oc") << ";}";
    f << y.SetByOffset("global_idx", "sum");
    return Status::OK();
  }
 private:
  const bool bias_, nearest_;
};
class TailPool final : public Program<TailPool> {
 public:
  TailPool() : Program{"RW07PredictorTailDepthwisePoolChannelWindowV2"} {}
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
    PayloadGuard(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels*uniforms.output_capacity){return;}"
      << y.SetByOffset("global_idx", "0.0") << "if(!valid_output()){return;}"
      << "let d=" << e.GetByOffset("6u") << ";let t=global_idx%uniforms.output_capacity;"
      << "let local_ch=global_idx/uniforms.output_capacity;let ch=uniforms.channel_offset+local_ch;if(t>=d||local_ch>=uniforms.channel_count){return;}var sum:f32=0.0;"
      // Original CPU Col2im visits kernel taps in ascending order.
      << "for(var k=0u;k<3u;k++){"
      << "let numerator=i32(t)+1-i32(k);if(numerator>=0&&numerator%2==0){"
      << "let source_t=u32(numerator/2);if(source_t<d/2u&&source_t<uniforms.input_capacity){"
      << "let product:f32=" << x.GetByOffset("local_ch*uniforms.input_capacity+source_t") << "*"
      << w.GetByOffset("ch*3u+k") << ";sum=sum+product;}}}let result:f32=sum+"
      << b.GetByOffset("ch") << ";" << y.SetByOffset("global_idx", "result");
    return Status::OK();
  }
};
class TailResidualAdd final : public Program<TailResidualAdd> {
 public:
  TailResidualAdd() : Program{"RW07PredictorTailResidualAddInPlaceWindowV2"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& skip = s.AddInput("skip", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels*uniforms.capacity){return;}"
      << "if(!valid_output()||global_idx%uniforms.capacity>=" << e.GetByOffset("6u") << "){"<<y.SetByOffset("global_idx","0.0")<<"return;}"
      << "let added:f32=" << y.GetByOffset("global_idx") << "+" << skip.GetByOffset("global_idx") << ";"
      << y.SetByOffset("global_idx", "added/bitcast<f32>(0x3fb504f3u)");
    return Status::OK();
  }
};
Status Binding(ComputeContext& c, const Tensor* t) {
  M0FullGpuStorageAllocationV1 allocation;
  ORT_RETURN_IF_NOT(t&&c.M0FullGpuDescribeStorageV1(*t,allocation),"RW07 live physical storage allocation");
  const auto& limits = c.DeviceLimits();
  const uint64_t bytes=allocation.backing_bytes;
  ORT_RETURN_IF_NOT(bytes >= t->SizeInBytes() && bytes <= limits.maxBufferSize && bytes <= limits.maxStorageBufferBindingSize,
                    "RW07 predictor tail whole-buffer binding limit");
  return Status::OK();
}
class PredictorTail final : public WebGpuKernel {
 public:
  explicit PredictorTail(const OpKernelInfo& info) : WebGpuKernel{info},
      capacity_(info.GetAttrOrDefault<int64_t>("physical_capacity", 0)),
      branch_(info.GetAttrOrDefault<int64_t>("original_branch", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    ORT_RETURN_IF_NOT((branch_ == 0 || branch_ == 1) && capacity_ > 0 && capacity_ <= 204800,
                      "RW07 predictor tail closed attributes");
    const auto* x = c.Input<Tensor>(0);
    const auto* e = c.Input<Tensor>(1);
    const auto* style = c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(x && x->IsDataType<float>() && x->Shape().NumDimensions() == 3 && x->Shape()[0] == 1 &&
                          x->Shape()[1] == 512 && x->Shape()[2] > 0 && x->Shape()[2] <= 102400 &&
                          e && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          style && style->IsDataType<float>() && style->Shape() == TensorShape({1, 256}),
                      "RW07 predictor tail exact source descriptors");
    // Closed input order: block1 norm1/pool/conv1/norm2/conv2/skip,
    // then block2 norm1/conv1/norm2/conv2 and final projection.
    const std::array<TensorShape, 21> shapes{
        TensorShape{1024,128}, TensorShape{1024}, TensorShape{512,1,3}, TensorShape{512},
        TensorShape{256,512,3}, TensorShape{256}, TensorShape{512,128}, TensorShape{512},
        TensorShape{256,256,3}, TensorShape{256}, TensorShape{256,512,1},
        TensorShape{512,128}, TensorShape{512}, TensorShape{256,256,3}, TensorShape{256},
        TensorShape{512,128}, TensorShape{512}, TensorShape{256,256,3}, TensorShape{256},
        TensorShape{1,256,1}, TensorShape{1}};
    for (size_t i = 0; i < shapes.size(); ++i) {
      const auto* p = c.Input<Tensor>(int(i + 3));
      ORT_RETURN_IF_NOT(p && p->IsDataType<float>() && p->Shape() == shapes[i],
                        "RW07 predictor tail exact immutable parameter descriptor at ", i + 3);
    }
    ORT_RETURN_IF_ERROR(m0_residual_stages::PhysicalBuffer(c,x));
    for(int i=1;i<24;++i)ORT_RETURN_IF_ERROR(Binding(c,c.Input<Tensor>(i)));
    const auto& limits = c.DeviceLimits();
    const uint32_t cap=uint32_t(capacity_),in_cap=uint32_t(x->Shape()[2]),norm_cap=std::max(cap,in_cap);
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    const uint64_t alignment=limits.minStorageBufferOffsetAlignment;
    ORT_RETURN_IF_NOT(alignment>0&&alignment%4==0,"RW07 predictor tail window alignment");
    const uint64_t prefix=alignment-4,row_bytes=uint64_t(cap)*4;
    ORT_RETURN_IF_NOT(uint64_t(kTailTileChannels)*norm_cap*4+prefix<=limits.maxStorageBufferBindingSize,
                      "RW07 predictor tail normalization tile binding limit");
    const uint32_t output_tile=uint32_t(std::min<uint64_t>(256,(limits.maxStorageBufferBindingSize-prefix)/row_bytes));
    const uint64_t max_count=std::max<uint64_t>(uint64_t(kTailTileChannels)*norm_cap,uint64_t(output_tile)*cap);
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 6 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit >= 8 &&
                          (max_count + 63) / 64 <= limit * limit && max_count <= UINT32_MAX &&
                          output_tile>0&&256ull*cap*4<=limits.maxBufferSize,
                      "RW07 predictor tail physical scratch/dispatch limits");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity sid, data, pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*e, sid) &&
                          c.M0FullGpuDescribeResourceV2(*x, data), "RW07 predictor tail live source allocations");
    const uint32_t sr = uint32_t(9 + branch_), si = uint32_t(1010 + branch_);
    const uint32_t role = uint32_t(11 + branch_), instance = uint32_t(1012 + branch_);
    const bool valid = c.M0FullGpuDescribePairV2(*e, *x, pair) && pair.session == session && pair.run == run &&
        pair.generation == sid.generation && pair.data_generation == data.generation &&
        pair.producer_role == sr && pair.producer_instance == si;
    auto* y = c.Output(0, TensorShape{1, 1, capacity_});
    auto* out = c.Output(1, TensorShape{32});
    ORT_RETURN_IF_NOT(y && y->IsDataType<float>() && out && out->IsDataType<uint32_t>(),
                      "RW07 predictor tail output types");
    ORT_RETURN_IF_ERROR(m0_residual_stages::PhysicalBuffer(c,y));ORT_RETURN_IF_ERROR(Binding(c,out));
    ORT_RETURN_IF_NOT(y->DataRaw() != out->DataRaw(), "RW07 predictor tail distinct outputs");
    for (int i = 0; i < 24; ++i) {
      const auto* input = c.Input<Tensor>(i);
      ORT_RETURN_IF_NOT(input->DataRaw() != y->DataRaw() && input->DataRaw() != out->DataRaw(),
                        "RW07 predictor tail source/output alias");
    }
    auto main=c.CreateGPUTensor(x->DataType(),TensorShape{1,256,cap});
    auto block1=c.CreateGPUTensor(x->DataType(),TensorShape{1,256,cap});
    auto block2=c.CreateGPUTensor(x->DataType(),TensorShape{1,256,cap});
    auto normalized=c.CreateGPUTensor(x->DataType(),TensorShape{1,kTailTileChannels,norm_cap});
    auto pool=c.CreateGPUTensor(x->DataType(),TensorShape{1,kTailTileChannels,cap});
    const std::array<const Tensor*,5> scratch{&main,&block1,&block2,&normalized,&pool};
    for(size_t i=0;i<scratch.size();++i){
      ORT_RETURN_IF_ERROR(m0_residual_stages::PhysicalBuffer(c,scratch[i]));
      for(int j=0;j<24;++j)ORT_RETURN_IF_NOT(scratch[i]->DataRaw()!=c.Input<Tensor>(j)->DataRaw(),"RW07 predictor tail scratch/input alias");
      ORT_RETURN_IF_NOT(scratch[i]->DataRaw()!=y->DataRaw()&&scratch[i]->DataRaw()!=out->DataRaw(),"RW07 predictor tail scratch/output alias");
      for(size_t j=0;j<i;++j)ORT_RETURN_IF_NOT(scratch[i]->DataRaw()!=scratch[j]->DataRaw(),"RW07 predictor tail live scratch alias");
    }
    const auto input_window=[&](const Tensor* src,uint32_t offset,uint32_t count){
      return ProgramInput::FloatBufferWindow(src,ProgramTensorMetadataDependency::None,TensorShape{1,count,src->Shape()[2]},
          offset*uint32_t(src->Shape()[2]),uint32_t(alignment));
    };
    const auto output_window=[&](Tensor* dst,uint32_t offset,uint32_t count){
      return ProgramOutput::FloatBufferWindow(dst,ProgramTensorMetadataDependency::None,TensorShape{1,count,cap},offset*cap,uint32_t(alignment));
    };
    const auto admit_norm=[&](const Tensor* src,int wi)->Status{
      for(uint32_t offset=0;offset<uint32_t(src->Shape()[1]);offset+=kTailTileChannels){
        const uint32_t count=std::min(kTailTileChannels,uint32_t(src->Shape()[1])-offset);
        ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindow(c,input_window(src,offset,count),
            {out,style,c.Input<Tensor>(wi),c.Input<Tensor>(wi+1),&normalized}));
      }
      return Status::OK();
    };
    const auto admit_conv=[&](const Tensor* src,Tensor* dst,int wi,int bi,uint32_t channels,bool windowed_input)->Status{
      const uint32_t outputs=uint32_t(dst->Shape()[1]);
      for(uint32_t offset=0;offset<channels;offset+=kTailTileChannels){
        const uint32_t count=std::min(kTailTileChannels,channels-offset);
        for(uint32_t o=0;o<outputs;o+=output_tile){
          const auto view=output_window(dst,o,std::min(output_tile,outputs-o));
          if(windowed_input){
            if(bi>=0){ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindows(c,input_window(src,offset,count),view,{c.Input<Tensor>(wi),c.Input<Tensor>(bi),out}));}
            else{ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindows(c,input_window(src,offset,count),view,{c.Input<Tensor>(wi),out}));}
          }else{
            ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindow(c,view,{src,c.Input<Tensor>(wi),c.Input<Tensor>(bi),out}));
          }
        }
      }
      return Status::OK();
    };
    const auto admit_add=[&](const Tensor* skip,Tensor* dst)->Status{
      for(uint32_t o=0;o<256;o+=output_tile){
        const uint32_t count=std::min(output_tile,256u-o);
        ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindows(c,input_window(skip,o,count),output_window(dst,o,count),{out}));
      }
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudget(c,{e,out}));
    ORT_RETURN_IF_ERROR(admit_norm(x,3));
    ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudget(c,{&normalized,c.Input<Tensor>(5),c.Input<Tensor>(6),out,&pool}));
    ORT_RETURN_IF_ERROR(admit_conv(&pool,&main,7,8,512,false));
    ORT_RETURN_IF_ERROR(admit_norm(&main,9));
    ORT_RETURN_IF_ERROR(admit_conv(&normalized,&block1,11,12,256,false));
    ORT_RETURN_IF_ERROR(admit_conv(x,&main,13,-1,512,true));
    ORT_RETURN_IF_ERROR(admit_add(&main,&block1));
    ORT_RETURN_IF_ERROR(admit_norm(&block1,14));
    ORT_RETURN_IF_ERROR(admit_conv(&normalized,&main,16,17,256,false));
    ORT_RETURN_IF_ERROR(admit_norm(&main,18));
    ORT_RETURN_IF_ERROR(admit_conv(&normalized,&block2,20,21,256,false));
    ORT_RETURN_IF_ERROR(admit_add(&block1,&block2));
    ORT_RETURN_IF_ERROR(admit_conv(&block2,y,22,23,256,true));
    M0FullGpuAllocationIdentity output;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, role, instance, output),
                      "RW07 predictor tail output pair claim");
    const m0_extent_abi2::OpaqueOwnership input_owner{session, run, sid.generation, data.generation, sr, si};
    const m0_extent_abi2::OpaqueOwnership owner{session, run, output.generation, output.data_generation, role, instance};
    ORT_RETURN_IF_NOT(input_owner.Valid() && owner.Valid(), "RW07 predictor tail opaque ownership");
    const auto source_words = input_owner.UniformWords(), words = owner.UniformWords();
    const auto span = [](const auto& a) { return gsl::span<const uint32_t>{a.data(), a.size()}; };
    TailGuard guard;
    guard.AddInput(Words(e)).AddOutput(Words(out)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({in_cap, cap, uint32_t(valid), span(source_words), span(words)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    uint64_t dispatches=1;
    const auto dispatch = [&](auto& program, uint64_t count) -> Status {
      const uint64_t groups = (count + 63) / 64, gx = std::min(groups, limit), gy = (groups + gx - 1) / gx;
      program.SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx), uint32_t(gy));
      ORT_RETURN_IF_ERROR(c.RunProgram(program));++dispatches;return Status::OK();
    };
    const auto norm = [&](const Tensor* src,int weight_index,uint32_t divisor,uint32_t offset,uint32_t count) -> Status {
      TailNormStyle p;
      p.AddInput(input_window(src,offset,count));
      p.AddInputs({Words(static_cast<const Tensor*>(out)),
                   {style, ProgramTensorMetadataDependency::None}, {c.Input<Tensor>(weight_index), ProgramTensorMetadataDependency::None},
                   {c.Input<Tensor>(weight_index + 1), ProgramTensorMetadataDependency::None}})
          .AddOutput({&normalized, ProgramTensorMetadataDependency::None})
          .AddUniformVariables({uint32_t(src->Shape()[2]),norm_cap,uint32_t(src->Shape()[1]),128u,divisor,offset,count,kTailTileChannels,cap,span(words)});
      return dispatch(p,kTailTileChannels);
    };
    const auto conv=[&](const Tensor* src,Tensor* dst,int wi,int bi,uint32_t width,bool nearest,
                        uint32_t channels,uint32_t offset,uint32_t count,bool windowed_input)->Status{
      const uint32_t outputs=uint32_t(dst->Shape()[1]);
      for(uint32_t o=0;o<outputs;o+=output_tile){
        const uint32_t output_count=std::min(output_tile,outputs-o);
        TailConv p{bi>=0,nearest};p.CacheHint(bi>=0?"bias":"no-bias",nearest?"nearest":"direct");
        p.AddInput(windowed_input?input_window(src,offset,count):ProgramInput{src,ProgramTensorMetadataDependency::None});
        p.AddInput({c.Input<Tensor>(wi),ProgramTensorMetadataDependency::None});
        if(bi>=0)p.AddInput({c.Input<Tensor>(bi),ProgramTensorMetadataDependency::None});
        p.AddInput(Words(static_cast<const Tensor*>(out))).AddOutput(output_window(dst,o,output_count))
            .AddUniformVariables({uint32_t(src->Shape()[2]),cap,channels,outputs,width,offset,count,o,output_count,1u,cap,span(words)});
        ORT_RETURN_IF_ERROR(dispatch(p,uint64_t(output_count)*cap));
      }
      return Status::OK();
    };
    const auto add=[&](const Tensor* skip,Tensor* dst)->Status{
      for(uint32_t o=0;o<256;o+=output_tile){
        const uint32_t count=std::min(output_tile,256u-o);
        TailResidualAdd p;p.AddInput(input_window(skip,o,count)).AddInput(Words(static_cast<const Tensor*>(out)))
            .AddOutput(output_window(dst,o,count)).AddUniformVariables({cap,count,cap,span(words)});
        ORT_RETURN_IF_ERROR(dispatch(p,uint64_t(count)*cap));
      }
      return Status::OK();
    };
    for(uint32_t offset=0;offset<512;offset+=kTailTileChannels){
      ORT_RETURN_IF_ERROR(norm(x,3,2,offset,kTailTileChannels));
      TailPool p;
      p.AddInputs({{&normalized,ProgramTensorMetadataDependency::None},{c.Input<Tensor>(5),ProgramTensorMetadataDependency::None},
                   {c.Input<Tensor>(6),ProgramTensorMetadataDependency::None},Words(static_cast<const Tensor*>(out))})
          .AddOutput({&pool,ProgramTensorMetadataDependency::None})
          .AddUniformVariables({norm_cap,cap,kTailTileChannels,offset,kTailTileChannels,cap,span(words)});
      ORT_RETURN_IF_ERROR(dispatch(p,uint64_t(kTailTileChannels)*cap));
      ORT_RETURN_IF_ERROR(conv(&pool,&main,7,8,3,false,512,offset,kTailTileChannels,false));
    }
    const auto normalized_conv=[&](const Tensor* src,Tensor* dst,int nw,int cw,int cb)->Status{
      for(uint32_t offset=0;offset<256;offset+=kTailTileChannels){
        ORT_RETURN_IF_ERROR(norm(src,nw,1,offset,kTailTileChannels));
        ORT_RETURN_IF_ERROR(conv(&normalized,dst,cw,cb,3,false,256,offset,kTailTileChannels,false));
      }
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(normalized_conv(&main,&block1,9,11,12));
    // No first-convolution reader remains when main becomes the skip branch.
    for(uint32_t offset=0;offset<512;offset+=kTailTileChannels){
      ORT_RETURN_IF_ERROR(conv(x,&main,13,-1,1,true,512,offset,kTailTileChannels,true));
    }
    ORT_RETURN_IF_ERROR(add(&main,&block1));
    ORT_RETURN_IF_ERROR(normalized_conv(&block1,&main,14,16,17));
    ORT_RETURN_IF_ERROR(normalized_conv(&main,&block2,18,20,21));
    ORT_RETURN_IF_ERROR(add(&block1,&block2));
    for(uint32_t offset=0;offset<256;offset+=kTailTileChannels){
      ORT_RETURN_IF_ERROR(conv(&block2,y,22,23,1,false,256,offset,kTailTileChannels,true));
    }
    LOGS(c.Logger(), INFO) << "M0_RW07_PREDICTOR_TAIL_V1 session=" << session << " run=" << run
      << " branch=" << branch_ << " source_pair_valid=" << valid << " role=" << role << " instance=" << instance
      << " source_capacity=" << in_cap << " output_capacity=" << cap << " GPU_dispatches="<<dispatches
      << " input_tile_channels="<<kTailTileChannels<<" output_tile_channels="<<output_tile<<" host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_, branch_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuPredictorTailExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<PredictorTail>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("PredictorTailExtentV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
