// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original F0/N0 style producers through LeakyRelu1355/1356; sidecars stay on GPU.
#include <algorithm>
#include <array>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"

namespace onnxruntime::webgpu {
namespace {
ProgramInput Words(const Tensor* tensor) {
  ProgramInput input(tensor, ProgramTensorMetadataDependency::None, TensorShape{32}, 1);
  input.var_type = ProgramVariableDataType::Uint32;
  return input;
}
ProgramOutput Words(Tensor* tensor) {
  ProgramOutput output(tensor, ProgramTensorMetadataDependency::None, TensorShape{32}, 1);
  output.var_type = ProgramVariableDataType::Uint32;
  return output;
}
bool Gpu(const Tensor* tensor) { return tensor && tensor->Location().device.Type() == OrtDevice::GPU; }

// Each payload program checks the complete successful output descriptor itself.
void AddValidExtent(ShaderHelper& shader, const ShaderVariableHelper& extent) {
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
class StyleAffineGuard final : public Program<StyleAffineGuard> {
 public:
  StyleAffineGuard() : Program{"RW07StyleAffineAfter1350OwnedGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"source_pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("source", ShaderUsage::UseValueTypeAlias);
    const auto& out = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << out.SetByOffset("i", "0u") << "}"
      << out.SetByOffset("0u", "2u") << out.SetByOffset("1u", "3u")
      << "for(var i=0u;i<10u;i++){" << out.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "if(uniforms.source_pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||"
      << e.GetByOffset("1u") << "!=3u){" << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity", "i", 10) << "){" << out.SetByOffset("20u", "103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){" << out.SetByOffset("20u", "103u") << "return;}"
      << out.SetByOffset("20u", "status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("4u") << "!=512u||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "let d=" << e.GetByOffset("6u") << ";if(d==0u){" << out.SetByOffset("20u", "103u") << "return;}"
      << "if(d>uniforms.input_capacity||d>uniforms.output_capacity){" << out.SetByOffset("20u", "102u") << "return;}"
      << "if(" << e.GetByOffset("18u") << "!=512u*d){" << out.SetByOffset("20u", "103u") << "return;}"
      << out.SetByOffset("2u", "1u") << out.SetByOffset("4u", "512u") << out.SetByOffset("6u", "d")
      << out.SetByOffset("18u", "512u*d") << out.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class StyleAffineData final : public Program<StyleAffineData> {
 public:
  StyleAffineData() : Program{"RW07StyleAffineAfter1350OwnedDataV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"style_features", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("normalized", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& style = s.AddInput("style", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("weights", ShaderUsage::UseValueTypeAlias);
    const auto& bias = s.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("activated", ShaderUsage::UseValueTypeAlias);
    AddValidExtent(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels){return;}"
      << "for(var t=0u;t<uniforms.output_capacity;t++){"
      << y.SetByOffset("global_idx*uniforms.output_capacity+t", "0.0") << "}"
      << "if(!valid_extent()){return;}let len=" << e.GetByOffset("6u") << ";"
      << "var scale_sum:f32=0.0;var bias_sum:f32=0.0;"
      << "for(var k=0u;k<uniforms.style_features;k++){let v:f32=" << style.GetByOffset("128u+k") << ";"
      << "let scale_product:f32=v*" << w.GetByOffset("global_idx*128u+k") << ";"
      << "let bias_product:f32=v*" << w.GetByOffset("(global_idx+512u)*128u+k") << ";"
      << "scale_sum=scale_sum+scale_product;bias_sum=bias_sum+bias_product;}"
      << "let scale_fc:f32=scale_sum+" << bias.GetByOffset("global_idx") << ";"
      << "let shift:f32=bias_sum+" << bias.GetByOffset("global_idx+512u") << ";"
      << "let scale:f32=scale_fc+1.0;for(var t=0u;t<len;t++){"
      << "let product:f32=scale*" << x.GetByOffset("global_idx*uniforms.input_capacity+t") << ";"
      << "let affine:f32=product+shift;"
      << "let negative:f32=affine*bitcast<f32>(0x3e4ccccdu);"
      << y.SetByOffset("global_idx*uniforms.output_capacity+t", "select(negative,affine,affine>=0.0)") << "}";
    return Status::OK();
  }
};
struct Attributes {
  explicit Attributes(const OpKernelInfo& info)
      : capacity(info.GetAttrOrDefault<int64_t>("physical_capacity", 0)),
        source_role(info.GetAttrOrDefault<int64_t>("source_role", -1)),
        source_instance(info.GetAttrOrDefault<int64_t>("source_instance", -1)),
        role(info.GetAttrOrDefault<int64_t>("semantic_role", -1)),
        instance(info.GetAttrOrDefault<int64_t>("producer_instance", -1)),
        branch(info.GetAttrOrDefault<int64_t>("original_branch", -1)) {}
  int64_t capacity, source_role, source_instance, role, instance, branch;
};
Status ComputeOwnedStyleAffine(ComputeContext& c, const Tensor* x, const Tensor* e, const Attributes& a) {
  const auto* style = c.Input<Tensor>(2);
  const auto* w = c.Input<Tensor>(3);
  const auto* bias = c.Input<Tensor>(4);
  ORT_RETURN_IF_NOT(Gpu(style) && style->IsDataType<float>() && style->Shape() == TensorShape({1, 256}) &&
                        Gpu(w) && w->IsDataType<float>() && w->Shape() == TensorShape({1024, 128}) &&
                        Gpu(bias) && bias->IsDataType<float>() && bias->Shape() == TensorShape({1024}),
                    "RW07 style affine after1350 exact original style/parameter descriptors");
  ORT_RETURN_IF_NOT(Gpu(x) && x->IsDataType<float>() && x->Shape().NumDimensions() == 3 &&
                        x->Shape()[0] == 1 && x->Shape()[1] == 512 && x->Shape()[2] > 0 && x->Shape()[2] <= 102400 &&
                        Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32},
                    "RW07 style affine after1350 exact input descriptors");
  ORT_RETURN_IF_NOT(a.capacity > 0 && a.capacity <= 102400 && a.source_role == 6 && a.source_instance == 1007 &&
                        ((a.branch == 0 && a.role == 7 && a.instance == 1008) ||
                         (a.branch == 1 && a.role == 8 && a.instance == 1009)),
                    "RW07 style affine after1350 closed attributes");
  const auto& limits = c.DeviceLimits();
  const uint64_t bytes = uint64_t(a.capacity) * 512 * sizeof(float);
  ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 6 && limits.maxComputeWorkgroupSizeX >= 64 &&
                        limits.maxComputeInvocationsPerWorkgroup >= 64 && limits.maxComputeWorkgroupsPerDimension >= 8 &&
                        bytes <= limits.maxBufferSize && bytes <= limits.maxStorageBufferBindingSize &&
                        limits.maxBufferSize >= 128 && limits.maxStorageBufferBindingSize >= 128,
                    "RW07 style affine after1350 resource/dispatch limits");
  auto binding = [&](const Tensor* value) -> Status {
    ORT_RETURN_IF_NOT(value->DataRaw() && value->SizeInBytes() > 0 && value->SizeInBytes() <= limits.maxBufferSize &&
                          value->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                      "RW07 style affine after1350 binding descriptor");
    const uint64_t actual = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(value->DataRaw())));
    ORT_RETURN_IF_NOT(actual >= value->SizeInBytes() && actual <= limits.maxBufferSize &&
                          actual <= limits.maxStorageBufferBindingSize,
                      "RW07 style affine after1350 actual whole-buffer binding");
    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(binding(x));
  ORT_RETURN_IF_ERROR(binding(e));
  for (const auto* parameter : {style, w, bias}) ORT_RETURN_IF_ERROR(binding(parameter));
  const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
  const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
  M0FullGpuAllocationIdentity source, carrier, pair;
  ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*e, source) &&
                        c.M0FullGpuDescribeResourceV2(*x, carrier),
                    "RW07 style affine after1350 actual input allocations");
  const bool valid_pair = c.M0FullGpuDescribePairV2(*e, *x, pair) && pair.generation == source.generation &&
                          pair.data_generation == carrier.generation && pair.session == session && pair.run == run &&
                          pair.producer_role == 6 && pair.producer_instance == 1007;
  auto* y = c.Output(0, TensorShape{1, 512, a.capacity});
  auto* out = c.Output(1, TensorShape{32});
  ORT_RETURN_IF_NOT(Gpu(y) && y->IsDataType<float>() && Gpu(out) && out->IsDataType<uint32_t>(),
                    "RW07 style affine after1350 output residency/type");
  ORT_RETURN_IF_ERROR(binding(y));
  ORT_RETURN_IF_ERROR(binding(out));
  ORT_RETURN_IF_NOT(y->DataRaw() != x->DataRaw() && y->DataRaw() != e->DataRaw() &&
                        out->DataRaw() != e->DataRaw() && out->DataRaw() != x->DataRaw() &&
                        y->DataRaw() != out->DataRaw(),
                    "RW07 style affine after1350 distinct outputs");
  for (const auto* parameter : {style, w, bias}) {
    ORT_RETURN_IF_NOT(y->DataRaw() != parameter->DataRaw() && out->DataRaw() != parameter->DataRaw(),
                      "RW07 style affine after1350 distinct parameter/output allocations");
  }
  M0FullGpuAllocationIdentity output;
  ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *y, uint32_t(a.role), uint32_t(a.instance), output),
                    "RW07 style affine after1350 output allocation claim");
  const m0_extent_abi2::OpaqueOwnership input_owner{session, run, source.generation, carrier.generation, 6, 1007};
  const m0_extent_abi2::OpaqueOwnership output_owner{session, run, output.generation, output.data_generation, uint32_t(a.role), uint32_t(a.instance)};
  ORT_RETURN_IF_NOT(input_owner.Valid() && output_owner.Valid(), "RW07 style affine after1350 opaque ownership");
  const auto input_words = input_owner.UniformWords(), output_words = output_owner.UniformWords();
  const auto input_identity = gsl::span<const uint32_t>{input_words.data(), input_words.size()};
  const auto identity = gsl::span<const uint32_t>{output_words.data(), output_words.size()};
  const auto input_capacity = uint32_t(x->Shape()[2]), output_capacity = uint32_t(a.capacity);
  StyleAffineGuard guard;
  guard.AddInput(Words(e)).AddOutput(Words(out)).SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({input_capacity, output_capacity, uint32_t(valid_pair), input_identity, identity});
  ORT_RETURN_IF_ERROR(c.RunProgram(guard));
  StyleAffineData data;
  data.AddInputs({{x, ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out)), {style, ProgramTensorMetadataDependency::None}, {w, ProgramTensorMetadataDependency::None}, {bias, ProgramTensorMetadataDependency::None}})
      .AddOutput({y, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(64)
      .SetDispatchGroupSize(8)
      .AddUniformVariables({input_capacity, output_capacity, identity, uint32_t(x->Shape()[1]), uint32_t(w->Shape()[1])});
  ORT_RETURN_IF_ERROR(c.RunProgram(data));
  LOGS(c.Logger(), INFO) << "M0_RW07_STYLE_AFFINE_AFTER1350_V1 session=" << session << " run=" << run
                         << " source_sidecar=" << e->DataRaw() << " source_generation=" << source.generation
                         << " source_carrier=" << x->DataRaw() << " source_carrier_generation=" << carrier.generation
                         << " source_pair_valid=" << valid_pair << " output_sidecar=" << out->DataRaw()
                         << " output_generation=" << output.generation << " output_carrier=" << y->DataRaw()
                         << " output_carrier_generation=" << output.data_generation << " role=" << a.role << " instance=" << a.instance << " original_branch=" << a.branch
                         << " source_capacity=" << input_capacity << " output_capacity=" << output_capacity
                         << " GPU_dispatches=2 host_payload_reads=0";
  return Status::OK();
}
class StyleAffineExtent final : public WebGpuKernel {
 public:
  explicit StyleAffineExtent(const OpKernelInfo& info) : WebGpuKernel{info}, attrs_(info) {}
  Status ComputeInternal(ComputeContext& c) const override {
    return ComputeOwnedStyleAffine(c, c.Input<Tensor>(0), c.Input<Tensor>(1), attrs_);
  }

 private:
  Attributes attrs_;
};
class StyleAffineMutation final : public Program<StyleAffineMutation> {
 public:
  StyleAffineMutation() : Program{"RW07StyleAffineAfter1350FaultMutationV1"} {}
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
class StyleAffineFault final : public WebGpuKernel {
 public:
  explicit StyleAffineFault(const OpKernelInfo& info) : WebGpuKernel{info}, attrs_(info), fault_(info.GetAttrOrDefault<int64_t>("fault_case", 0)) {
    ORT_ENFORCE(fault_ >= 0 && fault_ <= 22, "RW07 style affine after1350 closed static fault case");
  }
  Status ComputeInternal(ComputeContext& c) const override {
    const auto mode = c.M0FullGpuFaultModeV2();
    ORT_RETURN_IF_NOT((mode <= 4 || mode == 10 || mode == 11 || mode == 12 || mode == 20 || mode == 21 || mode == 22) && (fault_ == 0 || mode == 2),
                      "RW07 style affine after1350 certified fault selection");
    const bool retained = mode == 1 || mode == 4;
    const auto* x = c.Input<Tensor>(retained ? 6 : 0);
    const auto* e = c.Input<Tensor>(retained ? 5 : 1);
    ORT_RETURN_IF_NOT(x && e, "RW07 style affine after1350 explicit current/retained inputs");
    if (mode >= 2 && mode <= 4) {
      ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} && e->DataRaw(),
                        "RW07 style affine after1350 mutation sidecar descriptor");
      const auto& l = c.DeviceLimits();
      const uint64_t actual = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(e->DataRaw())));
      ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage >= 1 && l.maxComputeWorkgroupsPerDimension >= 1 &&
                            l.maxComputeWorkgroupSizeX >= 1 && l.maxComputeInvocationsPerWorkgroup >= 1 && actual >= 128 &&
                            actual <= l.maxBufferSize && actual <= l.maxStorageBufferBindingSize,
                        "RW07 style affine after1350 mutation resource limits");
      const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
      M0FullGpuAllocationIdentity live;
      ORT_RETURN_IF_NOT(run && c.M0FullGpuDescribeResourceV2(*e, live), "RW07 style affine after1350 live mutation descriptor");
      StyleAffineMutation mutation;
      mutation.AddOutput(Words(const_cast<Tensor*>(e))).SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({mode, uint32_t(fault_), uint32_t(run), uint32_t(run >> 32)});
      ORT_RETURN_IF_ERROR(c.RunProgram(mutation));
      LOGS(c.Logger(), INFO) << "M0_RW07_STYLE_AFFINE_AFTER1350_FAULT_MUTATION mode=" << mode << " fault_case=" << fault_
                             << " handle=" << e->DataRaw() << " generation=" << live.generation << " GPU_dispatch=1 host_payload_reads=0";
    }
    LOGS(c.Logger(), INFO) << "M0_RW07_STYLE_AFFINE_AFTER1350_FAULT_SELECTOR mode=" << mode << " selected_sidecar=" << e->DataRaw()
                           << " selected_carrier=" << x->DataRaw();
    return ComputeOwnedStyleAffine(c, x, e, attrs_);
  }

 private:
  Attributes attrs_;
  int64_t fault_;
};
class StyleAffinePacketProgram final : public Program<StyleAffinePacketProgram> {
 public:
  StyleAffinePacketProgram() : Program{"RW07StyleAffineAfter1350TerminalPacketV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& extent = shader.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& packet = shader.AddOutput("packet", ShaderUsage::UseValueTypeAlias);
    auto& f = shader.MainFunctionBody();
    f << "for(var i=0u;i<18u;i++){" << packet.SetByOffset("i", "0u") << "}"
      << packet.SetByOffset("0u", "103u")
      << "if(" << extent.GetByOffset("0u") << "!=2u||" << extent.GetByOffset("1u") << "!=3u){return;}"
      << "for(var i=0u;i<10u;i++){if(" << extent.GetByOffset("22u+i")
      << "!=" << GetElementAt("uniforms.identity", "i", 10) << "){return;}}"
      << "let status=" << extent.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}"
      << "for(var i=2u;i<20u;i++){if(" << extent.GetByOffset("i") << "!=0u){return;}}"
      << "if(" << extent.GetByOffset("21u") << "!=0u){return;}"
      << packet.SetByOffset("0u", "status") << "return;}"
      << "if(" << extent.GetByOffset("2u") << "!=1u||" << extent.GetByOffset("3u") << "!=0u||"
      << extent.GetByOffset("4u") << "!=uniforms.channels||" << extent.GetByOffset("5u") << "!=0u||"
      << extent.GetByOffset("7u") << "!=0u||" << extent.GetByOffset("21u") << "!=63u){return;}"
      << "for(var i=8u;i<18u;i++){if(" << extent.GetByOffset("i") << "!=0u){return;}}"
      << "let d=" << extent.GetByOffset("6u") << ";if(d==0u||d>uniforms.capacity||"
      << extent.GetByOffset("18u") << "!=uniforms.channels*d||" << extent.GetByOffset("19u") << "!=0u){return;}"
      << packet.SetByOffset("0u", "0u") << packet.SetByOffset("1u", "3u")
      << "for(var i=0u;i<6u;i++){" << packet.SetByOffset("2u+i", extent.GetByOffset("2u+i")) << "}";
    return Status::OK();
  }
};
class StyleAffinePacket final : public WebGpuKernel {
 public:
  explicit StyleAffinePacket(const OpKernelInfo& info) : WebGpuKernel{info},
                                                         role_(info.GetAttrOrDefault<int64_t>("source_role", -1)),
                                                         instance_(info.GetAttrOrDefault<int64_t>("source_instance", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* extent = c.Input<Tensor>(0);
    const auto* y = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(((role_ == 7 && instance_ == 1008) || (role_ == 8 && instance_ == 1009)) && extent && extent->IsDataType<uint32_t>() &&
                          extent->Shape() == TensorShape{m0_extent_abi2::Words} && y && y->IsDataType<float>() &&
                          y->Shape().NumDimensions() == 3 && y->Shape()[0] == 1 &&
                          y->Shape()[1] == 512 &&
                          y->Shape()[2] > 0 && y->Shape()[2] <= 102400 &&
                          extent->Location().device.Type() == OrtDevice::GPU && y->Location().device.Type() == OrtDevice::GPU,
                      "RW07 style affine after1350 terminal exact cut descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity identity;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*extent, *y, identity) &&
                          identity.session == session && identity.run == run && identity.producer_role == uint32_t(role_) &&
                          identity.producer_instance == uint32_t(instance_),
                      "RW07 style affine after1350 terminal actual producer pair");
    const auto& limits = c.DeviceLimits();
    for (const auto* input : {extent, y}) {
      ORT_RETURN_IF_NOT(input->DataRaw(), "RW07 style affine after1350 terminal missing backing");
      const auto handle = static_cast<WGPUBuffer>(const_cast<void*>(input->DataRaw()));
      const uint64_t bytes = wgpuBufferGetSize(handle);
      ORT_RETURN_IF_NOT(bytes >= input->SizeInBytes() && bytes <= limits.maxBufferSize &&
                            bytes <= limits.maxStorageBufferBindingSize,
                        "RW07 style affine after1350 terminal actual binding limit");
    }
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxBufferSize >= 128 &&
                          limits.maxStorageBufferBindingSize >= 128,
                      "RW07 style affine after1350 terminal device limits");
    auto* packet = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(packet->IsDataType<uint32_t>() && packet->Location().device.Type() == OrtDevice::GPU,
                      "RW07 style affine after1350 terminal packet allocation");
    const auto packet_handle = static_cast<WGPUBuffer>(const_cast<void*>(packet->DataRaw()));
    ORT_RETURN_IF_NOT(packet_handle && wgpuBufferGetSize(packet_handle) >= packet->SizeInBytes() &&
                          wgpuBufferGetSize(packet_handle) <= limits.maxBufferSize &&
                          wgpuBufferGetSize(packet_handle) <= limits.maxStorageBufferBindingSize,
                      "RW07 style affine after1350 terminal actual packet binding limit");
    const m0_extent_abi2::OpaqueOwnership ownership{session, run, identity.generation,
                                                    identity.data_generation, uint32_t(role_), uint32_t(instance_)};
    ORT_RETURN_IF_NOT(ownership.Valid(), "RW07 style affine after1350 terminal identity");
    const auto words = ownership.UniformWords();
    StyleAffinePacketProgram program;
    program.AddInput({extent, ProgramTensorMetadataDependency::None})
        .AddOutput({packet, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1)
        .SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[1]), uint32_t(y->Shape()[2]), gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*packet), "RW07 style affine after1350 terminal map fault registration");
    return Status::OK();
  }

 private:
  const int64_t role_, instance_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuStyleAffineAfter1350ExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<StyleAffineExtent>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("StyleAffineAfter1350ExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuStyleAffineAfter1350FaultExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<StyleAffineFault>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("StyleAffineAfter1350FaultExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuStyleAffineAfter1350TerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<StyleAffinePacket>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("StyleAffineAfter1350TerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
