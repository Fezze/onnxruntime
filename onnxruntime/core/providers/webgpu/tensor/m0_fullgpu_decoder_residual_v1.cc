// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original decoder encode/decode residual blocks, including real decode3 upsampling.
#include <algorithm>
#include <array>
#include <optional>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"

namespace onnxruntime::webgpu {
namespace {
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
void OriginalArithmetic(ShaderHelper& s) {
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
using m0_decoder_stages::PayloadGuard;
class DecoderGuard final : public Program<DecoderGuard> {
 public:
  DecoderGuard() : Program{"RW07DecoderResidualOwnedGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"pair_valid", ProgramUniformVariableDataType::Uint32},
      {"input_channels", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Uint32},
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
      << e.GetByOffset("4u") << "!=uniforms.input_channels||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u", "103u") << "return;}}"
      << "let d=" << e.GetByOffset("6u") << ";if(d==0u||d>102400u||"
      << e.GetByOffset("18u") << "!=uniforms.input_channels*d){" << o.SetByOffset("20u", "103u") << "return;}"
      << "if(d>uniforms.input_capacity||d>uniforms.final_capacity/uniforms.scale){"
      << o.SetByOffset("20u", "102u") << "return;}"
      << o.SetByOffset("2u", "1u") << o.SetByOffset("4u", "select(1024u,512u,uniforms.scale==2u)")
      << o.SetByOffset("6u", "uniforms.scale*d") << o.SetByOffset("18u", "select(1024u,512u,uniforms.scale==2u)*uniforms.scale*d") << o.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class DecoderNormStyle final : public Program<DecoderNormStyle> {
 public:
  DecoderNormStyle() : Program{"RW07DecoderResidualNormStyleChannelTileV2"} {}
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
    OriginalArithmetic(s);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.tile_channels){return;}for(var t=0u;t<uniforms.output_capacity;t++){"
      << y.SetByOffset("global_idx*uniforms.output_capacity+t", "0.0") << "}if(!valid_output()){return;}"
      << "if(global_idx>=uniforms.channel_count||uniforms.channel_offset>=uniforms.channels||"
      << "uniforms.channel_count>uniforms.channels-uniforms.channel_offset){return;}"
      << "let ch=uniforms.channel_offset+global_idx;"
      << "let d=" << e.GetByOffset("6u") << "/uniforms.extent_divisor;"
      << "if(d>uniforms.input_capacity||d>uniforms.output_capacity){return;}var sum:f32=0.0;"
      << "for(var t=0u;t<d;t++){sum=original_add(sum," << x.GetByOffset("global_idx*uniforms.input_capacity+t") << ");}"
      << "let mean:f32=original_div(sum,f32(d));var squares:f32=0.0;for(var t=0u;t<d;t++){let centered:f32="
      << "original_sub(" << x.GetByOffset("global_idx*uniforms.input_capacity+t") << ",mean);let square:f32=original_mul(centered,centered);"
      << "squares=original_add(squares,square);}let variance:f32=original_div(squares,f32(d));"
      << "let shifted:f32=original_add(variance,bitcast<f32>(0x3727c5acu));let denominator:f32=original_sqrt(shifted);"
      << "var gamma_sum:f32=0.0;var beta_sum:f32=0.0;for(var k=0u;k<uniforms.features;k++){"
      << "let style_value=" << style.GetByOffset("k") << ";let gp:f32=original_mul(style_value,"
      << w.GetByOffset("ch*uniforms.features+k") << ");let bp:f32=original_mul(style_value,"
      << w.GetByOffset("(ch+uniforms.channels)*uniforms.features+k") << ");"
      << "gamma_sum=original_add(gamma_sum,gp);beta_sum=original_add(beta_sum,bp);}let gamma_fc:f32=original_add(gamma_sum,"
      << b.GetByOffset("ch") << ");let gamma:f32=original_add(gamma_fc,1.0);let beta:f32=original_add(beta_sum,"
      << b.GetByOffset("ch+uniforms.channels") << ");for(var t=0u;t<d;t++){"
      << "let centered:f32=original_sub(" << x.GetByOffset("global_idx*uniforms.input_capacity+t") << ",mean);"
      << "let normalized:f32=original_div(centered,denominator);let product:f32=original_mul(gamma,normalized);let affine:f32=original_add(product,beta);"
      << "let negative:f32=original_mul(affine,bitcast<f32>(0x3e4ccccdu));"
      << y.SetByOffset("global_idx*uniforms.output_capacity+t", "select(negative,affine,affine>=0.0)") << "}";
    return Status::OK();
  }
};
constexpr uint32_t kDecoderPoolTileChannels=64;
using m0_decoder_stages::DecoderPoolTile;
using m0_decoder_stages::DecoderConvTile;
using m0_decoder_stages::DecoderResidualAdd;
Status Binding(ComputeContext& c, const Tensor* t) {
  M0FullGpuStorageAllocationV1 allocation;
  ORT_RETURN_IF_NOT(t&&c.M0FullGpuDescribeStorageV1(*t,allocation),"RW07 live physical storage allocation");
  const auto& limits = c.DeviceLimits();
  const uint64_t bytes=allocation.backing_bytes;
  ORT_RETURN_IF_NOT(bytes >= t->SizeInBytes() && bytes <= limits.maxBufferSize && bytes <= limits.maxStorageBufferBindingSize,
                    "RW07 decoder residual whole-buffer binding limit");
  return Status::OK();
}
class DecoderResidual final : public WebGpuKernel {
 public:
  explicit DecoderResidual(const OpKernelInfo& info) : WebGpuKernel{info},
      capacity_(info.GetAttrOrDefault<int64_t>("physical_capacity", 0)),
      block_(info.GetAttrOrDefault<int64_t>("block", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    ORT_RETURN_IF_NOT(block_ >= 0 && block_ <= 4 && capacity_ > 0 &&
                          capacity_ <= (block_ == 4 ? 204800 : 102400), "RW07 decoder closed attributes");
    const uint32_t ci = block_ == 0 ? 514u : 1090u, co = block_ == 4 ? 512u : 1024u;
    const uint32_t scale = block_ == 4 ? 2u : 1u;
    const int inputs = block_ == 4 ? 14 : 12;
    const auto* x = c.Input<Tensor>(0); const auto* e = c.Input<Tensor>(1); const auto* style = c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(x && x->IsDataType<float>() && x->Shape().NumDimensions() == 3 &&
                          x->Shape()[0] == 1 && x->Shape()[1] == ci && x->Shape()[2] > 0 && x->Shape()[2] <= 102400 &&
                          e && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          style && style->IsDataType<float>() && style->Shape() == TensorShape({1,256}),
                      "RW07 decoder original input descriptors");
    const std::array<TensorShape, 9> shapes{
        TensorShape{2 * ci,128}, TensorShape{2 * ci}, TensorShape{co,ci,3}, TensorShape{co},
        TensorShape{2 * co,128}, TensorShape{2 * co}, TensorShape{co,co,3}, TensorShape{co}, TensorShape{co,ci,1}};
    for (int i = 0; i < 9; ++i) {
      const auto* p = c.Input<Tensor>(i + 3);
      ORT_RETURN_IF_NOT(p && p->IsDataType<float>() && p->Shape() == shapes[i], "RW07 decoder exact parameter ", i + 3);
    }
    if (block_ == 4) {
      const auto* w = c.Input<Tensor>(12); const auto* b = c.Input<Tensor>(13);
      ORT_RETURN_IF_NOT(w && w->IsDataType<float>() && w->Shape() == TensorShape({1090,1,3}) &&
                            b && b->IsDataType<float>() && b->Shape() == TensorShape{1090},
                        "RW07 decode3 original depthwise transpose parameters");
    }
    ORT_RETURN_IF_ERROR(m0_residual_stages::PhysicalBuffer(c,x));
    for (int i = 1; i < inputs; ++i) ORT_RETURN_IF_ERROR(Binding(c, c.Input<Tensor>(i)));
    const auto& limits = c.DeviceLimits();
    const uint32_t cap=uint32_t(capacity_),in_cap=uint32_t(x->Shape()[2]),norm_cap=std::max(in_cap,cap);
    const uint64_t full_count=uint64_t(co)*cap;
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    const uint64_t alignment = limits.minStorageBufferOffsetAlignment;
    ORT_RETURN_IF_NOT(alignment > 0 && alignment % sizeof(float) == 0 &&
                          uint64_t(kDecoderPoolTileChannels)*std::max<uint64_t>(uint64_t(x->Shape()[2]),uint64_t(capacity_))*sizeof(float) + alignment - sizeof(float) <=
                              limits.maxStorageBufferBindingSize, "RW07 decoder input window binding limit");
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 6 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit >= 18 && full_count <= UINT32_MAX &&
                          full_count * sizeof(float) <= limits.maxBufferSize,
                      "RW07 decoder scratch and dispatch resource limits");
    const uint64_t prefix=alignment-sizeof(float),row_bytes=uint64_t(cap)*sizeof(float);
    ORT_RETURN_IF_NOT(row_bytes+prefix<=limits.maxStorageBufferBindingSize,"RW07 decoder output row binding limit");
    const uint32_t output_tile=uint32_t(std::min<uint64_t>(co,(limits.maxStorageBufferBindingSize-prefix)/row_bytes));
    const uint64_t max_count=std::max<uint64_t>(uint64_t(kDecoderPoolTileChannels)*norm_cap,uint64_t(output_tile)*cap);
    ORT_RETURN_IF_NOT(output_tile>0&&(max_count+63)/64<=limit*limit,"RW07 decoder actual window grid limit");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity sid, data, pair, output;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*e, sid) &&
                          c.M0FullGpuDescribeResourceV2(*x, data), "RW07 decoder live source allocations");
    const uint32_t sr = block_ == 0 ? 18u : uint32_t(24 + block_), si = sr + 1001;
    const uint32_t role = uint32_t(20 + block_), instance = role + 1001;
    const bool paired = c.M0FullGpuDescribePairV2(*e, *x, pair) && pair.session == session && pair.run == run &&
                        pair.generation == sid.generation && pair.data_generation == data.generation &&
                        pair.producer_role == sr && pair.producer_instance == si;
    auto* y = c.Output(0, TensorShape{1, co, capacity_}); auto* out = c.Output(1, TensorShape{32});
    ORT_RETURN_IF_NOT(y && y->IsDataType<float>() && out && out->IsDataType<uint32_t>(), "RW07 decoder output types");
    ORT_RETURN_IF_ERROR(m0_residual_stages::PhysicalBuffer(c,y)); ORT_RETURN_IF_ERROR(Binding(c,out));
    ORT_RETURN_IF_NOT(y->DataRaw() != out->DataRaw(), "RW07 decoder distinct outputs");
    for (int i = 0; i < inputs; ++i) {
      const auto* t = c.Input<Tensor>(i);
      ORT_RETURN_IF_NOT(t->DataRaw() != y->DataRaw() && t->DataRaw() != out->DataRaw(), "RW07 decoder input/output alias");
    }
    auto main=c.CreateGPUTensor(x->DataType(),TensorShape{1,co,cap});
    auto normalized=c.CreateGPUTensor(x->DataType(),TensorShape{1,kDecoderPoolTileChannels,norm_cap});
    std::optional<Tensor> pool;
    if(block_==4)pool.emplace(c.CreateGPUTensor(x->DataType(),TensorShape{1,kDecoderPoolTileChannels,cap}));
    const auto distinct=[&](const Tensor& t,gsl::span<const Tensor* const> live)->Status{
      ORT_RETURN_IF_ERROR(m0_residual_stages::PhysicalBuffer(c,&t));
      for(int i=0;i<inputs;++i)ORT_RETURN_IF_NOT(t.DataRaw()!=c.Input<Tensor>(i)->DataRaw(),"RW07 decoder scratch/input alias");
      ORT_RETURN_IF_NOT(t.DataRaw()!=y->DataRaw()&&t.DataRaw()!=out->DataRaw(),"RW07 decoder scratch/output alias");
      for(const auto* p:live)ORT_RETURN_IF_NOT(t.DataRaw()!=p->DataRaw(),"RW07 decoder live scratch alias");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(distinct(main,{}));
    const std::array<const Tensor*,1> live1{&main};ORT_RETURN_IF_ERROR(distinct(normalized,live1));
    if(pool){const std::array<const Tensor*,2> live2{&main,&normalized};ORT_RETURN_IF_ERROR(distinct(*pool,live2));}
    const auto window=[&](const Tensor* src,uint32_t offset,uint32_t count){
      return ProgramInput::FloatBufferWindow(src,ProgramTensorMetadataDependency::None,
          TensorShape{1,count,src->Shape()[2]},offset*uint32_t(src->Shape()[2]),uint32_t(alignment));
    };
    const auto output_window=[&](Tensor* dst,uint32_t offset,uint32_t count){
      return ProgramOutput::FloatBufferWindow(dst,ProgramTensorMetadataDependency::None,
          TensorShape{1,count,cap},offset*cap,uint32_t(alignment));
    };
    ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudget(c,{e,out}));
    const auto admit_norm=[&](const Tensor* src,int wi)->Status{
      for(uint32_t offset=0;offset<uint32_t(src->Shape()[1]);offset+=kDecoderPoolTileChannels){
        const uint32_t count=std::min(kDecoderPoolTileChannels,uint32_t(src->Shape()[1])-offset);
        ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindow(c,window(src,offset,count),
            {out,style,c.Input<Tensor>(wi),c.Input<Tensor>(wi+1),&normalized}));
      }
      return Status::OK();
    };
    const auto admit_conv=[&](const Tensor* src,Tensor* dst,int wi,int bi,uint32_t channels,bool tiled_input)->Status{
      for(uint32_t offset=0;offset<channels;offset+=kDecoderPoolTileChannels){
        const uint32_t count=std::min(kDecoderPoolTileChannels,channels-offset);
        for(uint32_t o=0;o<co;o+=output_tile){
          const auto output_view=output_window(dst,o,std::min(output_tile,co-o));
          if(tiled_input){
            ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindows(c,window(src,offset,count),output_view,
                {c.Input<Tensor>(wi),out}));
          }else{
            ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindow(c,output_view,
                {src,c.Input<Tensor>(wi),c.Input<Tensor>(bi),out}));
          }
        }
      }
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(admit_norm(x,3));
    if(pool)ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudget(c,{&normalized,c.Input<Tensor>(12),c.Input<Tensor>(13),out,&*pool}));
    ORT_RETURN_IF_ERROR(admit_conv(pool?&*pool:&normalized,&main,5,6,ci,false));
    ORT_RETURN_IF_ERROR(admit_norm(&main,7));
    ORT_RETURN_IF_ERROR(admit_conv(&normalized,y,9,10,co,false));
    ORT_RETURN_IF_ERROR(admit_conv(x,&main,11,-1,ci,true));
    for(uint32_t o=0;o<co;o+=output_tile){
      const uint32_t count=std::min(output_tile,co-o);
      ORT_RETURN_IF_ERROR(m0_residual_stages::StorageBudgetWithWindows(c,window(&main,o,count),output_window(y,o,count),{out}));
    }
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, role, instance, output), "RW07 decoder output claim");
    const m0_extent_abi2::OpaqueOwnership source{session,run,sid.generation,data.generation,sr,si};
    const m0_extent_abi2::OpaqueOwnership owner{session,run,output.generation,output.data_generation,role,instance};
    ORT_RETURN_IF_NOT(source.Valid() && owner.Valid(), "RW07 decoder opaque ownership");
    const auto sw = source.UniformWords(), words = owner.UniformWords();
    const auto span = [](const auto& a) { return gsl::span<const uint32_t>{a.data(),a.size()}; };
    DecoderGuard guard;
    guard.AddInput(Words(e)).AddOutput(Words(out)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({in_cap,cap,uint32_t(paired),ci,scale,span(sw),span(words)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const auto dispatch = [&](auto& p, uint64_t count) -> Status {
      const uint64_t groups=(count+63)/64, gx=std::min(groups,limit),gy=(groups+gx-1)/gx;
      p.SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy));return c.RunProgram(p);
    };
    const auto norm = [&](const Tensor* src, Tensor* dst, int wi, uint32_t divisor,
                          uint32_t offset, uint32_t count) -> Status {
      DecoderNormStyle p;
      p.AddInput(window(src,offset,count));
      p.AddInputs({Words(static_cast<const Tensor*>(out)),
                   {style,ProgramTensorMetadataDependency::None},{c.Input<Tensor>(wi),ProgramTensorMetadataDependency::None},
                   {c.Input<Tensor>(wi+1),ProgramTensorMetadataDependency::None}})
          .AddOutput({dst,ProgramTensorMetadataDependency::None})
          .AddUniformVariables({uint32_t(src->Shape()[2]),uint32_t(dst->Shape()[2]),uint32_t(src->Shape()[1]),128u,divisor,
                                offset,count,uint32_t(dst->Shape()[1]),cap,span(words)});
      return dispatch(p,uint64_t(dst->Shape()[1]));
    };
    const auto conv_tile = [&](const Tensor* src, Tensor* dst, int wi, int bi, uint32_t total_channels, uint32_t offset,
                               uint32_t count, uint32_t width, bool nearest, bool tiled_input) -> Status {
      for(uint32_t o=0;o<co;o+=output_tile){
        const uint32_t output_count=std::min(output_tile,co-o);
        DecoderConvTile p{bi>=0};p.CacheHint(bi>=0?"bias":"no-bias");
        p.AddInput(tiled_input?window(src,offset,count):ProgramInput{src,ProgramTensorMetadataDependency::None});
        p.AddInput({c.Input<Tensor>(wi),ProgramTensorMetadataDependency::None});
        if(bi>=0)p.AddInput({c.Input<Tensor>(bi),ProgramTensorMetadataDependency::None});
        p.AddInput(Words(static_cast<const Tensor*>(out))).AddOutput(output_window(dst,o,output_count))
            .AddUniformVariables({uint32_t(src->Shape()[2]),cap,total_channels,co,width,offset,count,o,output_count,
                                  uint32_t(nearest),cap,span(words)});
        ORT_RETURN_IF_ERROR(dispatch(p,uint64_t(output_count)*cap));
      }
      return Status::OK();
    };
    if(block_==4){
        for(uint32_t offset=0;offset<ci;offset+=kDecoderPoolTileChannels){
          const uint32_t count=std::min(kDecoderPoolTileChannels,ci-offset);
          ORT_RETURN_IF_ERROR(norm(x,&normalized,3,scale,offset,count));
          DecoderPoolTile p;
          p.AddInputs({{&normalized,ProgramTensorMetadataDependency::None},{c.Input<Tensor>(12),ProgramTensorMetadataDependency::None},
                       {c.Input<Tensor>(13),ProgramTensorMetadataDependency::None},Words(static_cast<const Tensor*>(out))})
              .AddOutput({&*pool,ProgramTensorMetadataDependency::None})
              .AddUniformVariables({norm_cap,cap,kDecoderPoolTileChannels,offset,count,cap,span(words)});
          ORT_RETURN_IF_ERROR(dispatch(p,uint64_t(pool->Shape().Size())));
          ORT_RETURN_IF_ERROR(conv_tile(&*pool,&main,5,6,ci,offset,count,3,false,false));
        }
    }else{
        for(uint32_t offset=0;offset<ci;offset+=kDecoderPoolTileChannels){
          const uint32_t count=std::min(kDecoderPoolTileChannels,ci-offset);
          ORT_RETURN_IF_ERROR(norm(x,&normalized,3,scale,offset,count));
          ORT_RETURN_IF_ERROR(conv_tile(&normalized,&main,5,6,ci,offset,count,3,false,false));
        }
    }
    // Keep main immutable until every channel has been normalized and consumed.
    for(uint32_t offset=0;offset<co;offset+=kDecoderPoolTileChannels){
      const uint32_t count=std::min(kDecoderPoolTileChannels,co-offset);
      ORT_RETURN_IF_ERROR(norm(&main,&normalized,7,1,offset,count));
      ORT_RETURN_IF_ERROR(conv_tile(&normalized,y,9,10,co,offset,count,3,false,false));
    }
    // All readers of the first convolution precede these writes on the queue.
    for(uint32_t offset=0;offset<ci;offset+=kDecoderPoolTileChannels){
      const uint32_t count=std::min(kDecoderPoolTileChannels,ci-offset);
      ORT_RETURN_IF_ERROR(conv_tile(x,&main,11,-1,ci,offset,count,1,block_==4,true));
    }
    for(uint32_t o=0;o<co;o+=output_tile){
      const uint32_t count=std::min(output_tile,co-o);
      DecoderResidualAdd add;
      add.AddInput(window(&main,o,count)).AddInput(Words(static_cast<const Tensor*>(out)))
          .AddOutput(output_window(y,o,count)).AddUniformVariables({cap,count,cap,span(words)});
      ORT_RETURN_IF_ERROR(dispatch(add,uint64_t(count)*cap));
    }
    const uint64_t output_tiles=(uint64_t(co)+output_tile-1)/output_tile;
    const uint64_t first_tiles=(uint64_t(ci)+kDecoderPoolTileChannels-1)/kDecoderPoolTileChannels;
    const uint64_t second_tiles=(uint64_t(co)+kDecoderPoolTileChannels-1)/kDecoderPoolTileChannels;
    LOGS(c.Logger(),INFO)<<"M0_RW07_DECODER_RESIDUAL_V1 block="<<block_<<" role="<<role<<" source_pair_valid="<<paired
                        <<" GPU_dispatches="<<1+first_tiles*(1+(block_==4?1:0)+2*output_tiles)
                            +second_tiles*(1+output_tiles)+output_tiles
                        <<" input_tile_channels="<<kDecoderPoolTileChannels
                        <<" output_tile_channels="<<output_tile
                        <<" pool_tile_channels="<<(block_==4?kDecoderPoolTileChannels:0)<<" host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_,block_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuDecoderResidualExtentV1KernelInfo(){
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<DecoderResidual>(i);return Status::OK();
  };
  return {KernelDefBuilder().SetName("DecoderResidualExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
              .Provider(kWebGpuExecutionProvider).Build(),fn};
}
}  // namespace onnxruntime::webgpu
