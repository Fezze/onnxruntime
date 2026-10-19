// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original F0.1/F0.2 or N.1/N.2 and scalar projection, preserving GPU-owned D.
#include <algorithm>
#include <array>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"

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
  TailNormStyle() : Program{"RW07PredictorTailNormStyleV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"features", ProgramUniformVariableDataType::Uint32},
      {"extent_divisor", ProgramUniformVariableDataType::Uint32},
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
    f << "if(global_idx>=uniforms.channels){return;}for(var t=0u;t<uniforms.output_capacity;t++){"
      << y.SetByOffset("global_idx*uniforms.output_capacity+t", "0.0") << "}if(!valid_output()){return;}"
      << "let d=" << e.GetByOffset("6u") << "/uniforms.extent_divisor;"
      << "if(d>uniforms.input_capacity||d>uniforms.output_capacity){return;}var sum:f32=0.0;"
      << "for(var t=0u;t<d;t++){sum=sum+" << x.GetByOffset("global_idx*uniforms.input_capacity+t") << ";}"
      << "let mean:f32=sum/f32(d);var squares:f32=0.0;for(var t=0u;t<d;t++){let centered:f32="
      << x.GetByOffset("global_idx*uniforms.input_capacity+t") << "-mean;let square:f32=centered*centered;"
      << "squares=squares+square;}let variance:f32=squares/f32(d);"
      << "let shifted:f32=variance+bitcast<f32>(0x3727c5acu);let denominator:f32=sqrt(shifted);"
      << "var gamma_sum:f32=0.0;var beta_sum:f32=0.0;for(var k=0u;k<uniforms.features;k++){"
      << "let style_value=" << style.GetByOffset("128u+k") << ";let gp:f32=style_value*"
      << w.GetByOffset("global_idx*uniforms.features+k") << ";let bp:f32=style_value*"
      << w.GetByOffset("(global_idx+uniforms.channels)*uniforms.features+k") << ";"
      << "gamma_sum=gamma_sum+gp;beta_sum=beta_sum+bp;}let gamma_fc:f32=gamma_sum+"
      << b.GetByOffset("global_idx") << ";let gamma:f32=gamma_fc+1.0;let beta:f32=beta_sum+"
      << b.GetByOffset("global_idx+uniforms.channels") << ";for(var t=0u;t<d;t++){"
      << "let centered:f32=" << x.GetByOffset("global_idx*uniforms.input_capacity+t") << "-mean;"
      << "let normalized:f32=centered/denominator;let product:f32=gamma*normalized;let affine:f32=product+beta;"
      << "let negative:f32=affine*bitcast<f32>(0x3e4ccccdu);"
      << y.SetByOffset("global_idx*uniforms.output_capacity+t", "select(negative,affine,affine>=0.0)") << "}";
    return Status::OK();
  }
};
class TailConv final : public Program<TailConv> {
 public:
  TailConv(bool bias, bool nearest) : Program{"RW07PredictorTailConvV1"}, bias_(bias), nearest_(nearest) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"input_channels", ProgramUniformVariableDataType::Uint32},
      {"output_channels", ProgramUniformVariableDataType::Uint32},
      {"width", ProgramUniformVariableDataType::Uint32},
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
    f << "if(global_idx>=uniforms.output_channels*uniforms.output_capacity){return;}"
      << y.SetByOffset("global_idx", "0.0") << "if(!valid_output()){return;}"
      << "let d=" << e.GetByOffset("6u") << "/uniforms.extent_divisor;let t=global_idx%uniforms.output_capacity;"
      << "let oc=global_idx/uniforms.output_capacity;if(t>=d){return;}var sum:f32=0.0;"
      << "for(var ic=0u;ic<uniforms.input_channels;ic++){for(var k=0u;k<uniforms.width;k++){"
      << "let pos=i32(t)+i32(k)-i32(uniforms.width/2u);var value:f32=0.0;"
      << "if(pos>=0&&u32(pos)<d){let source_t=" << (nearest_ ? "u32(pos)/2u" : "u32(pos)") << ";"
      << "if(source_t<uniforms.input_capacity){value=" << x.GetByOffset("ic*uniforms.input_capacity+source_t") << ";}}"
      << "let product:f32=value*" << w.GetByOffset("(oc*uniforms.input_channels+ic)*uniforms.width+k")
      << ";sum=sum+product;}}";
    if (bias_) f << "let result:f32=sum+" << b->GetByOffset("oc") << ";";
    else f << "let result:f32=sum;";
    f << y.SetByOffset("global_idx", "result");
    return Status::OK();
  }
 private:
  const bool bias_, nearest_;
};
class TailPool final : public Program<TailPool> {
 public:
  TailPool() : Program{"RW07PredictorTailDepthwisePoolV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
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
      << "let ch=global_idx/uniforms.output_capacity;if(t>=d){return;}var sum:f32=0.0;"
      // Original CPU Col2im visits kernel taps in ascending order.
      << "for(var k=0u;k<3u;k++){"
      << "let numerator=i32(t)+1-i32(k);if(numerator>=0&&numerator%2==0){"
      << "let source_t=u32(numerator/2);if(source_t<d/2u&&source_t<uniforms.input_capacity){"
      << "let product:f32=" << x.GetByOffset("ch*uniforms.input_capacity+source_t") << "*"
      << w.GetByOffset("ch*3u+k") << ";sum=sum+product;}}}let result:f32=sum+"
      << b.GetByOffset("ch") << ";" << y.SetByOffset("global_idx", "result");
    return Status::OK();
  }
};
class TailResidualAdd final : public Program<TailResidualAdd> {
 public:
  TailResidualAdd() : Program{"RW07PredictorTailResidualAddV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"final_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& main = s.AddInput("main", ShaderUsage::UseValueTypeAlias);
    const auto& skip = s.AddInput("skip", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels*uniforms.capacity){return;}"
      << y.SetByOffset("global_idx", "0.0") << "if(!valid_output()){return;}"
      << "if(global_idx%uniforms.capacity>=" << e.GetByOffset("6u") << "){return;}"
      << "let added:f32=" << main.GetByOffset("global_idx") << "+" << skip.GetByOffset("global_idx") << ";"
      << y.SetByOffset("global_idx", "added/bitcast<f32>(0x3fb504f3u)");
    return Status::OK();
  }
};
Status Binding(ComputeContext& c, const Tensor* t) {
  ORT_RETURN_IF_NOT(t && t->Location().device.Type() == OrtDevice::GPU && t->DataRaw() && t->SizeInBytes() > 0,
                    "RW07 predictor tail missing GPU binding");
  const auto& limits = c.DeviceLimits();
  const uint64_t bytes = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
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
    for (int i = 0; i < 24; ++i) ORT_RETURN_IF_ERROR(Binding(c, c.Input<Tensor>(i)));
    const auto& limits = c.DeviceLimits();
    const uint64_t max_count = 512 * std::max<uint64_t>(uint64_t(x->Shape()[2]), uint64_t(capacity_));
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 6 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit >= 8 &&
                          (max_count + 63) / 64 <= limit * limit && max_count <= UINT32_MAX &&
                          max_count * sizeof(float) <= limits.maxBufferSize &&
                          max_count * sizeof(float) <= limits.maxStorageBufferBindingSize,
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
    for (const auto* t : {y, out}) ORT_RETURN_IF_ERROR(Binding(c, t));
    ORT_RETURN_IF_NOT(y->DataRaw() != out->DataRaw(), "RW07 predictor tail distinct outputs");
    for (int i = 0; i < 24; ++i) {
      const auto* input = c.Input<Tensor>(i);
      ORT_RETURN_IF_NOT(input->DataRaw() != y->DataRaw() && input->DataRaw() != out->DataRaw(),
                        "RW07 predictor tail source/output alias");
    }
    M0FullGpuAllocationIdentity output;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, role, instance, output),
                      "RW07 predictor tail output pair claim");
    const m0_extent_abi2::OpaqueOwnership input_owner{session, run, sid.generation, data.generation, sr, si};
    const m0_extent_abi2::OpaqueOwnership owner{session, run, output.generation, output.data_generation, role, instance};
    ORT_RETURN_IF_NOT(input_owner.Valid() && owner.Valid(), "RW07 predictor tail opaque ownership");
    const auto source_words = input_owner.UniformWords(), words = owner.UniformWords();
    const auto span = [](const auto& a) { return gsl::span<const uint32_t>{a.data(), a.size()}; };
    const uint32_t cap = uint32_t(capacity_), in_cap = uint32_t(x->Shape()[2]);
    TailGuard guard;
    guard.AddInput(Words(e)).AddOutput(Words(out)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({in_cap, cap, uint32_t(valid), span(source_words), span(words)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const auto scratch = [&](int64_t channels, int64_t capacity) {
      return c.CreateGPUTensor(x->DataType(), TensorShape{1, channels, capacity});
    };
    const auto check_scratch = [&](const Tensor& t, gsl::span<const Tensor* const> live) -> Status {
      ORT_RETURN_IF_ERROR(Binding(c, &t));
      for (int i = 0; i < 24; ++i) {
        ORT_RETURN_IF_NOT(t.DataRaw() != c.Input<Tensor>(i)->DataRaw(), "RW07 predictor tail scratch/input alias");
      }
      ORT_RETURN_IF_NOT(t.DataRaw() != y->DataRaw() && t.DataRaw() != out->DataRaw(),
                        "RW07 predictor tail scratch/output alias");
      for (const auto* other : live) ORT_RETURN_IF_NOT(t.DataRaw() != other->DataRaw(), "RW07 predictor tail live scratch alias");
      return Status::OK();
    };
    const auto dispatch = [&](auto& program, uint64_t count) -> Status {
      const uint64_t groups = (count + 63) / 64, gx = std::min(groups, limit), gy = (groups + gx - 1) / gx;
      program.SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx), uint32_t(gy));
      return c.RunProgram(program);
    };
    const auto norm = [&](const Tensor* src, Tensor* dst, int weight_index, uint32_t divisor) -> Status {
      TailNormStyle p;
      p.AddInputs({{src, ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out)),
                   {style, ProgramTensorMetadataDependency::None}, {c.Input<Tensor>(weight_index), ProgramTensorMetadataDependency::None},
                   {c.Input<Tensor>(weight_index + 1), ProgramTensorMetadataDependency::None}})
          .AddOutput({dst, ProgramTensorMetadataDependency::None})
          .AddUniformVariables({uint32_t(src->Shape()[2]), uint32_t(dst->Shape()[2]), uint32_t(src->Shape()[1]),
                                uint32_t(128), divisor, cap, span(words)});
      return dispatch(p, uint64_t(src->Shape()[1]));
    };
    const auto conv = [&](const Tensor* src, Tensor* dst, int wi, int bi, uint32_t width, bool nearest) -> Status {
      TailConv p{bi >= 0, nearest};
      p.CacheHint(bi >= 0 ? "bias" : "no-bias", nearest ? "nearest" : "direct");
      p.AddInputs({{src, ProgramTensorMetadataDependency::None}, {c.Input<Tensor>(wi), ProgramTensorMetadataDependency::None}});
      if (bi >= 0) p.AddInput({c.Input<Tensor>(bi), ProgramTensorMetadataDependency::None});
      p.AddInput(Words(static_cast<const Tensor*>(out))).AddOutput({dst, ProgramTensorMetadataDependency::None})
          .AddUniformVariables({uint32_t(src->Shape()[2]), uint32_t(dst->Shape()[2]), uint32_t(src->Shape()[1]),
                                uint32_t(dst->Shape()[1]), width, uint32_t(1), cap, span(words)});
      return dispatch(p, uint64_t(dst->Shape().Size()));
    };
    const auto add = [&](const Tensor* main, const Tensor* skip, Tensor* dst) -> Status {
      TailResidualAdd p;
      p.AddInputs({{main, ProgramTensorMetadataDependency::None}, {skip, ProgramTensorMetadataDependency::None},
                   Words(static_cast<const Tensor*>(out))}).AddOutput({dst, ProgramTensorMetadataDependency::None})
          .AddUniformVariables({cap, uint32_t(256), cap, span(words)});
      return dispatch(p, uint64_t(dst->Shape().Size()));
    };
    auto block1 = scratch(256, cap);
    ORT_RETURN_IF_ERROR(check_scratch(block1, {}));
    {
      auto skip = scratch(256, cap);
      const std::array<const Tensor*, 1> live1{&block1};
      ORT_RETURN_IF_ERROR(check_scratch(skip, live1));
      ORT_RETURN_IF_ERROR(conv(x, &skip, 13, -1, 1, true));
      auto main = scratch(256, cap);
      const std::array<const Tensor*, 2> live2{&block1, &skip};
      ORT_RETURN_IF_ERROR(check_scratch(main, live2));
      {
        auto n1 = scratch(512, in_cap);
        const std::array<const Tensor*, 3> live3{&block1, &skip, &main};
        ORT_RETURN_IF_ERROR(check_scratch(n1, live3));
        ORT_RETURN_IF_ERROR(norm(x, &n1, 3, 2));
        auto pool = scratch(512, cap);
        const std::array<const Tensor*, 4> live4{&block1, &skip, &main, &n1};
        ORT_RETURN_IF_ERROR(check_scratch(pool, live4));
        TailPool p;
        p.AddInputs({{&n1, ProgramTensorMetadataDependency::None}, {c.Input<Tensor>(5), ProgramTensorMetadataDependency::None},
                     {c.Input<Tensor>(6), ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out))})
            .AddOutput({&pool, ProgramTensorMetadataDependency::None})
            .AddUniformVariables({in_cap, cap, uint32_t(512), cap, span(words)});
        ORT_RETURN_IF_ERROR(dispatch(p, uint64_t(pool.Shape().Size())));
        ORT_RETURN_IF_ERROR(conv(&pool, &main, 7, 8, 3, false));
      }
      auto n2 = scratch(256, cap);
      const std::array<const Tensor*, 3> live3{&block1, &skip, &main};
      ORT_RETURN_IF_ERROR(check_scratch(n2, live3));
      ORT_RETURN_IF_ERROR(norm(&main, &n2, 9, 1));
      auto convolved = scratch(256, cap);
      const std::array<const Tensor*, 4> live4{&block1, &skip, &main, &n2};
      ORT_RETURN_IF_ERROR(check_scratch(convolved, live4));
      ORT_RETURN_IF_ERROR(conv(&n2, &convolved, 11, 12, 3, false));
      ORT_RETURN_IF_ERROR(add(&convolved, &skip, &block1));
    }
    auto block2 = scratch(256, cap);
    const std::array<const Tensor*, 1> live1{&block1};
    ORT_RETURN_IF_ERROR(check_scratch(block2, live1));
    {
      auto n1 = scratch(256, cap);
      const std::array<const Tensor*, 2> live2{&block1, &block2};
      ORT_RETURN_IF_ERROR(check_scratch(n1, live2));
      ORT_RETURN_IF_ERROR(norm(&block1, &n1, 14, 1));
      auto main = scratch(256, cap);
      const std::array<const Tensor*, 3> live3{&block1, &block2, &n1};
      ORT_RETURN_IF_ERROR(check_scratch(main, live3));
      ORT_RETURN_IF_ERROR(conv(&n1, &main, 16, 17, 3, false));
      auto n2 = scratch(256, cap);
      const std::array<const Tensor*, 4> live4{&block1, &block2, &n1, &main};
      ORT_RETURN_IF_ERROR(check_scratch(n2, live4));
      ORT_RETURN_IF_ERROR(norm(&main, &n2, 18, 1));
      auto convolved = scratch(256, cap);
      const std::array<const Tensor*, 5> live5{&block1, &block2, &n1, &main, &n2};
      ORT_RETURN_IF_ERROR(check_scratch(convolved, live5));
      ORT_RETURN_IF_ERROR(conv(&n2, &convolved, 20, 21, 3, false));
      ORT_RETURN_IF_ERROR(add(&convolved, &block1, &block2));
    }
    ORT_RETURN_IF_ERROR(conv(&block2, y, 22, 23, 1, false));
    LOGS(c.Logger(), INFO) << "M0_RW07_PREDICTOR_TAIL_V1 session=" << session << " run=" << run
      << " branch=" << branch_ << " source_pair_valid=" << valid << " role=" << role << " instance=" << instance
      << " source_capacity=" << in_cap << " output_capacity=" << cap << " GPU_dispatches=14 host_payload_reads=0";
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
