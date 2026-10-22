// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original F0.0/N.0 Conv1 -> norm2/style -> Conv2 -> residual -> Div.
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
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }

void PayloadGuard(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_output() -> bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=3u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("4u") << "!=512u||" << e.GetByOffset("5u") << "!=0u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let d=" << e.GetByOffset("6u") << ";return d>0u&&d<=uniforms.capacity&&"
    << e.GetByOffset("18u") << "==512u*d;}\n";
}

class ResidualGuard final : public Program<ResidualGuard> {
 public:
  ResidualGuard() : Program{"RW07PredictorResidualGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"source_capacity", ProgramUniformVariableDataType::Uint32},
      {"merged_capacity", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"source_pair_valid", ProgramUniformVariableDataType::Uint32},
      {"merged_pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"merged_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& a = s.AddInput("branch_extent", ShaderUsage::UseValueTypeAlias);
    const auto& b = s.AddInput("merged_extent", ShaderUsage::UseValueTypeAlias);
    const auto& o = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    auto status_function = [&](const char* name, const ShaderVariableHelper& e, const char* valid,
                               const char* identity, const char* capacity) {
      auto& f = s.AdditionalImplementation();
      f << "fn " << name << "() -> u32 {if(uniforms." << valid << "==0u||"
        << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=3u){return 103u;}"
        << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
        << GetElementAt(std::string("uniforms.") + identity, "i", 10) << "){return 103u;}}"
        << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
        << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return 103u;}"
        << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){return 103u;}}"
        << "if(" << e.GetByOffset("21u") << "!=0u){return 103u;}return status;}"
        << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
        << e.GetByOffset("4u") << "!=512u||" << e.GetByOffset("5u") << "!=0u||"
        << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
        << e.GetByOffset("21u") << "!=63u){return 103u;}"
        << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return 103u;}}"
        << "let d=" << e.GetByOffset("6u") << ";if(d==0u||"
        << e.GetByOffset("18u") << "!=512u*d){return 103u;}"
        << "if(d>uniforms." << capacity << "||d>uniforms.capacity){return 102u;}return 0u;}\n";
    };
    status_function("branch_status", a, "source_pair_valid", "source_identity", "source_capacity");
    status_function("merged_status", b, "merged_pair_valid", "merged_identity", "merged_capacity");
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << o.SetByOffset("i", "0u") << "}"
      << o.SetByOffset("0u", "2u") << o.SetByOffset("1u", "3u")
      << "for(var i=0u;i<10u;i++){" << o.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "let a=branch_status();let b=merged_status();"
      << "if(a==103u||b==103u){" << o.SetByOffset("20u", "103u") << "return;}"
      << "if(a!=0u||b!=0u){" << o.SetByOffset("20u", "select(b,a,a!=0u)") << "return;}"
      << "let d=" << a.GetByOffset("6u") << ";if(d!=" << b.GetByOffset("6u") << "){"
      << o.SetByOffset("20u", "103u") << "return;}"
      << o.SetByOffset("2u", "1u") << o.SetByOffset("4u", "512u") << o.SetByOffset("6u", "d")
      << o.SetByOffset("18u", "512u*d") << o.SetByOffset("21u", "63u");
    return Status::OK();
  }
};

class ResidualConv final : public Program<ResidualConv> {
 public:
  explicit ResidualConv(bool final) : Program{"RW07PredictorResidualConvV1"}, final_(final) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"merged_capacity", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const auto& b = s.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* merged = final_ ? &s.AddInput("merged", ShaderUsage::UseValueTypeAlias) : nullptr;
    const auto& y = s.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=512u*uniforms.capacity){return;}" << y.SetByOffset("global_idx", "0.0")
      << "if(!valid_output()){return;}let t=global_idx%uniforms.capacity;let oc=global_idx/uniforms.capacity;"
      << "let d=" << e.GetByOffset("6u") << ";if(t>=d){return;}var sum:f32=0.0;"
      << "for(var ic=0u;ic<512u;ic++){for(var k=0u;k<3u;k++){let p=i32(t)+i32(k)-1;var value:f32=0.0;"
      << "if(p>=0&&u32(p)<d){value=" << x.GetByOffset("ic*uniforms.input_capacity+u32(p)") << ";}"
      << "let product:f32=value*" << w.GetByOffset("(oc*512u+ic)*3u+k") << ";sum=sum+product;}}"
      << "let convolved:f32=sum+" << b.GetByOffset("oc") << ";";
    if (final_) {
      f << "let residual:f32=convolved+" << merged->GetByOffset("oc*uniforms.merged_capacity+t") << ";"
        << y.SetByOffset("global_idx", "residual/bitcast<f32>(0x3fb504f3u)");
    } else {
      f << y.SetByOffset("global_idx", "convolved");
    }
    return Status::OK();
  }
 private:
  const bool final_;
};

class ResidualNormStyle final : public Program<ResidualNormStyle> {
 public:
  ResidualNormStyle() : Program{"RW07PredictorResidualNormStyleV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"features", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("conv1", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& style = s.AddInput("style", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const auto& b = s.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("activated", ShaderUsage::UseValueTypeAlias);
    PayloadGuard(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels){return;}for(var t=0u;t<uniforms.capacity;t++){"
      << y.SetByOffset("global_idx*uniforms.capacity+t", "0.0") << "}if(!valid_output()){return;}"
      << "let d=" << e.GetByOffset("6u") << ";var sum:f32=0.0;for(var t=0u;t<d;t++){sum=sum+"
      << x.GetByOffset("global_idx*uniforms.capacity+t") << ";}let mean:f32=sum/f32(d);var squares:f32=0.0;"
      << "for(var t=0u;t<d;t++){let delta:f32=" << x.GetByOffset("global_idx*uniforms.capacity+t")
      << "-mean;let square:f32=delta*delta;squares=squares+square;}let variance:f32=squares/f32(d);"
      << "let shifted:f32=variance+bitcast<f32>(0x3727c5acu);let denominator:f32=sqrt(shifted);"
      << "var gamma_sum:f32=0.0;var beta_sum:f32=0.0;for(var k=0u;k<uniforms.features;k++){"
      << "let v=" << style.GetByOffset("128u+k") << ";let gp:f32=v*"
      << w.GetByOffset("global_idx*128u+k") << ";let bp:f32=v*"
      << w.GetByOffset("(global_idx+512u)*128u+k") << ";gamma_sum=gamma_sum+gp;beta_sum=beta_sum+bp;}"
      << "let gamma_fc:f32=gamma_sum+" << b.GetByOffset("global_idx") << ";let gamma:f32=gamma_fc+1.0;"
      << "let beta:f32=beta_sum+" << b.GetByOffset("global_idx+512u") << ";for(var t=0u;t<d;t++){"
      << "let centered:f32=" << x.GetByOffset("global_idx*uniforms.capacity+t") << "-mean;"
      << "let normalized:f32=centered/denominator;let product:f32=gamma*normalized;let affine:f32=product+beta;"
      << "let negative:f32=affine*bitcast<f32>(0x3e4ccccdu);"
      << y.SetByOffset("global_idx*uniforms.capacity+t", "select(negative,affine,affine>=0.0)") << "}";
    return Status::OK();
  }
};

Status Binding(ComputeContext& c, const Tensor* t) {
  ORT_RETURN_IF_NOT(Gpu(t) && t->DataRaw() && t->SizeInBytes() > 0,
                    "RW07 predictor residual missing GPU allocation");
  const auto& limits = c.DeviceLimits();
  M0FullGpuStorageAllocationV1 allocation;
  ORT_RETURN_IF_NOT(c.M0FullGpuDescribeStorageV1(*t,allocation),"RW07 live physical storage allocation");
  const uint64_t actual=allocation.backing_bytes;
  ORT_RETURN_IF_NOT(actual >= t->SizeInBytes() && actual <= limits.maxBufferSize &&
                        actual <= limits.maxStorageBufferBindingSize,
                    "RW07 predictor residual whole-buffer binding limit");
  return Status::OK();
}

class PredictorResidual : public WebGpuKernel {
 public:
  explicit PredictorResidual(const OpKernelInfo& info) : WebGpuKernel{info},
      capacity_(info.GetAttrOrDefault<int64_t>("physical_capacity", 0)),
      branch_(info.GetAttrOrDefault<int64_t>("original_branch", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    return ComputeOwned(c, c.Input<Tensor>(0), c.Input<Tensor>(1), c.Input<Tensor>(2), c.Input<Tensor>(3));
  }
 protected:
  Status ComputeOwned(ComputeContext& c, const Tensor* x, const Tensor* extent,
                      const Tensor* merged, const Tensor* merged_extent) const {
    ORT_RETURN_IF_NOT((branch_ == 0 || branch_ == 1) && capacity_ > 0 && capacity_ <= 102400,
                      "RW07 predictor residual closed branch/capacity");
    const auto* style = c.Input<Tensor>(4);
    const auto* w1 = c.Input<Tensor>(5);
    const auto* b1 = c.Input<Tensor>(6);
    const auto* wn = c.Input<Tensor>(7);
    const auto* bn = c.Input<Tensor>(8);
    const auto* w2 = c.Input<Tensor>(9);
    const auto* b2 = c.Input<Tensor>(10);
    for (const auto* t : {x, merged}) {
      ORT_RETURN_IF_NOT(Gpu(t) && t->IsDataType<float>() && t->Shape().NumDimensions() == 3 &&
                            t->Shape()[0] == 1 && t->Shape()[1] == 512 && t->Shape()[2] > 0 && t->Shape()[2] <= 102400,
                        "RW07 predictor residual carrier descriptor");
    }
    for (const auto* e : {extent, merged_extent}) {
      ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32},
                        "RW07 predictor residual sidecar descriptor");
    }
    ORT_RETURN_IF_NOT(Gpu(style) && style->IsDataType<float>() && style->Shape() == TensorShape({1, 256}) &&
                          Gpu(wn) && wn->IsDataType<float>() && wn->Shape() == TensorShape({1024, 128}) &&
                          Gpu(bn) && bn->IsDataType<float>() && bn->Shape() == TensorShape{1024},
                      "RW07 predictor residual exact style descriptors");
    for (const auto* w : {w1, w2}) {
      ORT_RETURN_IF_NOT(Gpu(w) && w->IsDataType<float>() && w->Shape() == TensorShape({512, 512, 3}),
                        "RW07 predictor residual original immutable convolution weights");
    }
    for (const auto* b : {b1, b2}) {
      ORT_RETURN_IF_NOT(Gpu(b) && b->IsDataType<float>() && b->Shape() == TensorShape{512},
                        "RW07 predictor residual original convolution bias");
    }
    const auto& limits = c.DeviceLimits();
    const uint64_t count = uint64_t(capacity_) * 512, groups = (count + 63) / 64;
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 6 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit >= 8 && groups <= limit * limit &&
                          count * sizeof(float) <= limits.maxBufferSize && count * sizeof(float) <= limits.maxStorageBufferBindingSize,
                      "RW07 predictor residual resource/dispatch limits");
    for (const auto* t : {x, extent, merged, merged_extent, style, w1, b1, wn, bn, w2, b2}) {
      ORT_RETURN_IF_ERROR(Binding(c, t));
    }
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity source, source_data, merge, merge_data, pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*extent, source) &&
                          c.M0FullGpuDescribeResourceV2(*x, source_data) &&
                          c.M0FullGpuDescribeResourceV2(*merged_extent, merge) &&
                          c.M0FullGpuDescribeResourceV2(*merged, merge_data),
                      "RW07 predictor residual actual source allocations");
    const uint32_t source_role = uint32_t(7 + branch_), source_instance = uint32_t(1008 + branch_);
    const uint32_t role = uint32_t(9 + branch_), instance = uint32_t(1010 + branch_);
    auto current_pair = [&](const Tensor& e, const Tensor& t, uint32_t r, uint32_t i,
                            const M0FullGpuAllocationIdentity& sid, const M0FullGpuAllocationIdentity& data) {
      return c.M0FullGpuDescribePairV2(e, t, pair) && pair.session == session && pair.run == run &&
             pair.generation == sid.generation && pair.data_generation == data.generation &&
             pair.producer_role == r && pair.producer_instance == i;
    };
    const bool source_valid = current_pair(*extent, *x, source_role, source_instance, source, source_data);
    const bool merge_valid = current_pair(*merged_extent, *merged, 5, 1006, merge, merge_data);
    auto* y = c.Output(0, TensorShape{1, 512, capacity_});
    auto* out = c.Output(1, TensorShape{32});
    ORT_RETURN_IF_NOT(Gpu(y) && y->IsDataType<float>() && Gpu(out) && out->IsDataType<uint32_t>(),
                      "RW07 predictor residual output descriptors");
    auto conv1 = c.CreateGPUTensor(x->DataType(), TensorShape{1, 512, capacity_});
    auto activated = c.CreateGPUTensor(x->DataType(), TensorShape{1, 512, capacity_});
    std::array<const Tensor*, 4> outputs{y, out, &conv1, &activated};
    for (size_t i = 0; i < outputs.size(); ++i) {
      ORT_RETURN_IF_ERROR(Binding(c, outputs[i]));
      for (size_t j = 0; j < i; ++j) {
        ORT_RETURN_IF_NOT(outputs[i]->DataRaw() != outputs[j]->DataRaw(), "RW07 predictor residual output/scratch alias");
      }
      for (const auto* t : {x, extent, merged, merged_extent, style, w1, b1, wn, bn, w2, b2}) {
        ORT_RETURN_IF_NOT(outputs[i]->DataRaw() != t->DataRaw(), "RW07 predictor residual source/output alias");
      }
    }
    M0FullGpuAllocationIdentity output;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, role, instance, output),
                      "RW07 predictor residual output ownership claim");
    const m0_extent_abi2::OpaqueOwnership source_owner{session, run, source.generation, source_data.generation, source_role, source_instance};
    const m0_extent_abi2::OpaqueOwnership merged_owner{session, run, merge.generation, merge_data.generation, 5, 1006};
    const m0_extent_abi2::OpaqueOwnership output_owner{session, run, output.generation, output.data_generation, role, instance};
    ORT_RETURN_IF_NOT(source_owner.Valid() && merged_owner.Valid() && output_owner.Valid(),
                      "RW07 predictor residual opaque identities");
    const auto sw = source_owner.UniformWords(), mw = merged_owner.UniformWords(), ow = output_owner.UniformWords();
    const auto span = [](const auto& a) { return gsl::span<const uint32_t>{a.data(), a.size()}; };
    const auto cap = uint32_t(capacity_), source_cap = uint32_t(x->Shape()[2]), merged_cap = uint32_t(merged->Shape()[2]);
    ResidualGuard guard;
    guard.AddInputs({Words(extent), Words(merged_extent)}).AddOutput(Words(out))
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({source_cap, merged_cap, cap, uint32_t(source_valid), uint32_t(merge_valid), span(sw), span(mw), span(ow)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const uint64_t gx = std::min(groups, limit), gy = (groups + gx - 1) / gx;
    ResidualConv first{false};
    first.CacheHint("first");
    first.AddInputs({{x, ProgramTensorMetadataDependency::None}, {w1, ProgramTensorMetadataDependency::None},
                     {b1, ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out))})
        .AddOutput({&conv1, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx), uint32_t(gy))
        .AddUniformVariables({source_cap, merged_cap, cap, span(ow)});
    ORT_RETURN_IF_ERROR(c.RunProgram(first));
    ResidualNormStyle norm;
    norm.AddInputs({{&conv1, ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out)),
                    {style, ProgramTensorMetadataDependency::None}, {wn, ProgramTensorMetadataDependency::None},
                    {bn, ProgramTensorMetadataDependency::None}})
        .AddOutput({&activated, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize(8)
        .AddUniformVariables({cap, span(ow), uint32_t(512), uint32_t(128)});
    ORT_RETURN_IF_ERROR(c.RunProgram(norm));
    ResidualConv last{true};
    last.CacheHint("final");
    last.AddInputs({{&activated, ProgramTensorMetadataDependency::None}, {w2, ProgramTensorMetadataDependency::None},
                    {b2, ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out)),
                    {merged, ProgramTensorMetadataDependency::None}})
        .AddOutput({y, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx), uint32_t(gy))
        .AddUniformVariables({cap, merged_cap, cap, span(ow)});
    ORT_RETURN_IF_ERROR(c.RunProgram(last));
    LOGS(c.Logger(), INFO) << "M0_RW07_PREDICTOR_RESIDUAL_V1 session=" << session << " run=" << run
      << " branch=" << branch_ << " source_pair_valid=" << source_valid << " merged_pair_valid=" << merge_valid
      << " role=" << role << " instance=" << instance << " source_capacity=" << source_cap
      << " merged_capacity=" << merged_cap << " output_capacity=" << cap
      << " output_generation=" << output.generation << " output_carrier_generation=" << output.data_generation
      << " scratch_logical_bytes=" << count * sizeof(float) * 2 << " GPU_dispatches=4 host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_, branch_;
};
class ResidualMutation final : public Program<ResidualMutation> {
 public:
  ResidualMutation() : Program{"RW07PredictorResidualFaultMutationV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"mode", ProgramUniformVariableDataType::Uint32},
                                          {"fault_case", ProgramUniformVariableDataType::Uint32}, {"run_low", ProgramUniformVariableDataType::Uint32},
                                          {"run_high", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddOutput("selected_sidecar", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "if(uniforms.mode==3u){" << e.SetByOffset("31u", "1005u") << "return;}"
      << "if(uniforms.mode==4u){" << e.SetByOffset("24u", "uniforms.run_low") << e.SetByOffset("25u", "uniforms.run_high") << "return;}"
      << "let fault=uniforms.fault_case;";
    const auto set = [&](int id, const char* index, const char* value) { f << "if(fault==" << id << "u){" << e.SetByOffset(index, value) << "return;}"; };
    f << "if(fault==0u){" << e.SetByOffset("30u", "2u") << e.SetByOffset("31u", "1002u") << "return;}";
    set(1, "0u", "1u");
    set(2, "1u", "2u");
    set(3, "8u", "1u");
    set(4, "3u", "1u");
    f << "if(fault==5u){" << e.SetByOffset("18u", e.GetByOffset("18u") + "+1u") << "return;}";
    set(6, "21u", "127u");
    for (int id = 7; id <= 10; ++id) {
      const auto word = std::to_string(22 + (id - 7) * 2) + "u";
      f << "if(fault==" << id << "u){" << e.SetByOffset(word, e.GetByOffset(word) + "^1u") << "return;}";
    }
    f << "if(fault==11u){" << e.SetByOffset("6u", "0u") << e.SetByOffset("18u", "0u") << "return;}";
    set(12, "20u", "100u");
    set(15, "21u", "62u");
    set(16, "4u", "513u");
    set(17, "19u", "1u");
    f << "for(var i=2u;i<20u;i++){" << e.SetByOffset("i", "0u") << "}" << e.SetByOffset("21u", "0u");
    set(13, "20u", "777u");
    set(14, "20u", "100u");
    set(18, "20u", "1u");
    set(19, "20u", "101u");
    set(20, "20u", "102u");
    set(21, "20u", "103u");
    f << e.SetByOffset("20u", "100u") << e.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class PredictorResidualFault final : public PredictorResidual {
 public:
  explicit PredictorResidualFault(const OpKernelInfo& info) : PredictorResidual{info},
      fault_(info.GetAttrOrDefault<int64_t>("fault_case", 0)),
      source_(info.GetAttrOrDefault<int64_t>("fault_source", -1)) {
    ORT_ENFORCE(fault_ >= 0 && fault_ <= 22 && (source_ == 0 || source_ == 1),
                "RW07 predictor residual closed diagnostic fault recipe");
  }
  Status ComputeInternal(ComputeContext& c) const override {
    const uint32_t mode = c.M0FullGpuFaultModeV2();
    ORT_RETURN_IF_NOT((mode <= 4 || mode == 10 || mode == 11 || mode == 12 || mode == 20 || mode == 21 || mode == 22) &&
                          (fault_ == 0 || mode == 2), "RW07 predictor residual certified fault mode");
    const bool retained = mode == 1 || mode == 4;
    const Tensor* x = c.Input<Tensor>(retained && source_ == 0 ? 11 : 0);
    const Tensor* e = c.Input<Tensor>(retained && source_ == 0 ? 12 : 1);
    const Tensor* merged = c.Input<Tensor>(retained && source_ == 1 ? 13 : 2);
    const Tensor* me = c.Input<Tensor>(retained && source_ == 1 ? 14 : 3);
    if (mode >= 2 && mode <= 4) {
      const auto* selected = source_ == 0 ? e : me;
      ORT_RETURN_IF_NOT(selected && selected->IsDataType<uint32_t>() && selected->Shape() == TensorShape{32},
                        "RW07 predictor residual mutation descriptor");
      ORT_RETURN_IF_ERROR(Binding(c, selected));
      const auto& limits = c.DeviceLimits();
      ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 1 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                            limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                        "RW07 predictor residual mutation resources");
      const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
      M0FullGpuAllocationIdentity live;
      ORT_RETURN_IF_NOT(run && c.M0FullGpuDescribeResourceV2(*selected, live),
                        "RW07 predictor residual actual mutation allocation");
      const auto* selected_carrier = source_ == 0 ? x : merged;
      M0FullGpuAllocationIdentity live_carrier;
      ORT_RETURN_IF_NOT(selected_carrier && c.M0FullGpuDescribeResourceV2(*selected_carrier, live_carrier),
                        "RW07 predictor residual actual mutation carrier allocation");
      LOGS(c.Logger(), INFO) << "M0_RW07_PREDICTOR_RESIDUAL_ACTUAL_SOURCE source=" << source_
          << " mode=" << mode << " sidecar=" << selected->DataRaw() << " generation=" << live.generation
          << " paired_carrier=" << reinterpret_cast<const void*>(live.data_handle)
          << " paired_generation=" << live.data_generation << " carrier=" << selected_carrier->DataRaw()
          << " carrier_generation=" << live_carrier.generation << " producer_run=" << live.run
          << " actual_run=" << run << " host_payload_reads=0";
      ResidualMutation mutation;
      mutation.AddOutput(Words(const_cast<Tensor*>(selected))).SetWorkgroupSize(1).SetDispatchGroupSize(1)
          .AddUniformVariables({mode, uint32_t(fault_), uint32_t(run), uint32_t(run >> 32)});
      ORT_RETURN_IF_ERROR(c.RunProgram(mutation));
      LOGS(c.Logger(), INFO) << "M0_RW07_PREDICTOR_RESIDUAL_FAULT_MUTATION source=" << source_
          << " mode=" << mode << " fault_case=" << fault_ << " handle=" << selected->DataRaw()
          << " generation=" << live.generation << " host_payload_reads=0";
    }
    return ComputeOwned(c, x, e, merged, me);
  }
 private:
  const int64_t fault_, source_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuPredictorResidualExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<PredictorResidual>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("PredictorResidualExtentV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuPredictorResidualFaultExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<PredictorResidualFault>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("PredictorResidualFaultExtentV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
