// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Exact shared1340 state/recurrence/layout/moments region; sidecars stay on GPU.
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
class RecurrenceGuard final : public Program<RecurrenceGuard> {
 public:
  RecurrenceGuard() : Program{"RW07V2RecurrenceOwnedGuard"} {}
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
      << e.GetByOffset("4u") << "!=640u||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "let d=" << e.GetByOffset("6u") << ";if(d==0u){" << out.SetByOffset("20u", "103u") << "return;}"
      << "if(d>uniforms.input_capacity||d>uniforms.output_capacity){" << out.SetByOffset("20u", "102u") << "return;}"
      << "if(" << e.GetByOffset("18u") << "!=640u*d){" << out.SetByOffset("20u", "103u") << "return;}"
      << out.SetByOffset("2u", "1u") << out.SetByOffset("4u", "512u") << out.SetByOffset("6u", "d")
      << out.SetByOffset("18u", "512u*d") << out.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class RecurrenceData final : public Program<RecurrenceData> {
 public:
  RecurrenceData() : Program{"RW07V2RecurrenceOwnedData"} {}
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
      << "let w_base=direction*1024u*640u;for(var k=0u;k<uniforms.input_size;k++){let xv=" << x.GetByOffset("k*uniforms.input_capacity+t") << ";"
      << "gate_i+=xv*" << w.GetByOffset("w_base+unit*640u+k") << ";"
      << "gate_o+=xv*" << w.GetByOffset("w_base+(256u+unit)*640u+k") << ";"
      << "gate_f+=xv*" << w.GetByOffset("w_base+(512u+unit)*640u+k") << ";"
      << "gate_c+=xv*" << w.GetByOffset("w_base+(768u+unit)*640u+k") << ";}"
      << "let r_base=direction*1024u*256u;for(var k=0u;k<uniforms.hidden_size;k++){let hv=previous_h[k];"
      << "gate_i+=hv*" << r.GetByOffset("r_base+unit*256u+k") << ";"
      << "gate_o+=hv*" << r.GetByOffset("r_base+(256u+unit)*256u+k") << ";"
      << "gate_f+=hv*" << r.GetByOffset("r_base+(512u+unit)*256u+k") << ";"
      << "gate_c+=hv*" << r.GetByOffset("r_base+(768u+unit)*256u+k") << ";}"
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
struct Attributes {
  explicit Attributes(const OpKernelInfo& info)
      : capacity(info.GetAttrOrDefault<int64_t>("physical_capacity", 0)),
        source_role(info.GetAttrOrDefault<int64_t>("source_role", -1)),
        source_instance(info.GetAttrOrDefault<int64_t>("source_instance", -1)),
        role(info.GetAttrOrDefault<int64_t>("semantic_role", -1)),
        instance(info.GetAttrOrDefault<int64_t>("producer_instance", -1)) {}
  int64_t capacity, source_role, source_instance, role, instance;
};
Status ComputeOwnedRecurrence(ComputeContext& c, const Tensor* x, const Tensor* e, const Attributes& attrs) {
  const auto* w = c.Input<Tensor>(1);
  const auto* r = c.Input<Tensor>(2);
  const auto* b = c.Input<Tensor>(3);
  ORT_RETURN_IF_NOT(Gpu(x) && x->IsDataType<float>() && x->Shape().NumDimensions() == 3 &&
                        x->Shape()[0] == 1 && x->Shape()[1] == 640 && x->Shape()[2] > 0 && x->Shape()[2] <= 102400 &&
                        Gpu(w) && w->IsDataType<float>() && w->Shape() == TensorShape({2, 1024, 640}) &&
                        Gpu(r) && r->IsDataType<float>() && r->Shape() == TensorShape({2, 1024, 256}) &&
                        Gpu(b) && b->IsDataType<float>() && b->Shape() == TensorShape({2, 2048}) &&
                        Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32},
                    "RW07 recurrence V2 exact input descriptors");
  ORT_RETURN_IF_NOT(attrs.capacity > 0 && attrs.capacity <= 102400 && attrs.source_role == 4 &&
                        attrs.source_instance == 1004 && attrs.role == 5 && attrs.instance == 1006,
                    "RW07 recurrence V2 closed original1340 attributes");
  const auto& l = c.DeviceLimits();
  const uint64_t bytes = uint64_t(attrs.capacity) * 512 * 4;
  const uint64_t groups = (uint64_t(attrs.capacity) * 512 + 63) / 64;
  ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage >= 8 && l.maxComputeWorkgroupSizeX >= 256 &&
                        l.maxComputeInvocationsPerWorkgroup >= 256 && l.maxComputeWorkgroupStorageSize >= 2052 &&
                        l.maxComputeWorkgroupsPerDimension >= 8 &&
                        groups <= uint64_t(l.maxComputeWorkgroupsPerDimension) * l.maxComputeWorkgroupsPerDimension &&
                        bytes <= l.maxBufferSize && bytes <= l.maxStorageBufferBindingSize &&
                        l.maxBufferSize >= 2048 && l.maxStorageBufferBindingSize >= 2048,
                    "RW07 recurrence V2 actual resource/dispatch limits");
  auto binding = [&](const Tensor* t) -> Status {
    ORT_RETURN_IF_NOT(t->DataRaw() && t->SizeInBytes() > 0 && t->SizeInBytes() <= l.maxBufferSize &&
                          t->SizeInBytes() <= l.maxStorageBufferBindingSize,
                      "RW07 recurrence V2 binding descriptor");
    const uint64_t actual = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
    ORT_RETURN_IF_NOT(actual >= t->SizeInBytes() && actual <= l.maxBufferSize && actual <= l.maxStorageBufferBindingSize,
                      "RW07 recurrence V2 actual whole-buffer binding");
    return Status::OK();
  };
  for (const auto* t : {x, w, r, b, e}) ORT_RETURN_IF_ERROR(binding(t));
  const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
  const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
  M0FullGpuAllocationIdentity source, carrier, pair;
  ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribeResourceV2(*e, source) && c.M0FullGpuDescribeResourceV2(*x, carrier),
                    "RW07 recurrence V2 actual input allocation descriptors");
  const bool valid_pair = c.M0FullGpuDescribePairV2(*e, *x, pair) && pair.generation == source.generation &&
                          pair.data_generation == carrier.generation && pair.session == session && pair.run == run &&
                          pair.producer_role == 4 && pair.producer_instance == 1004;
  auto* y = c.Output(0, TensorShape{attrs.capacity, 2, 1, 256});
  auto* h = c.Output(1, TensorShape{2, 1, 256});
  auto* cell = c.Output(2, TensorShape{2, 1, 256});
  auto* merged = c.Output(3, TensorShape{1, 512, attrs.capacity});
  auto* mean = c.Output(4, TensorShape{1, 512, 1});
  auto* variance = c.Output(5, TensorShape{1, 512, 1});
  auto* out = c.Output(6, TensorShape{32});
  for (const auto* t : {y, h, cell, merged, mean, variance}) {
    ORT_RETURN_IF_NOT(Gpu(t) && t->IsDataType<float>(), "RW07 recurrence V2 output residency/type");
    ORT_RETURN_IF_ERROR(binding(t));
  }
  ORT_RETURN_IF_NOT(Gpu(out) && out->IsDataType<uint32_t>(), "RW07 recurrence V2 extent residency/type");
  ORT_RETURN_IF_ERROR(binding(out));
  M0FullGpuAllocationIdentity output;
  ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out, *merged, 5, 1006, output), "RW07 recurrence V2 output allocation claim");
  const m0_extent_abi2::OpaqueOwnership input_owner{session, run, source.generation, carrier.generation, 4, 1004};
  const m0_extent_abi2::OpaqueOwnership output_owner{session, run, output.generation, output.data_generation, 5, 1006};
  ORT_RETURN_IF_NOT(input_owner.Valid() && output_owner.Valid(), "RW07 recurrence V2 opaque ownership");
  const auto in = input_owner.UniformWords(), identity = output_owner.UniformWords();
  const auto opaque = gsl::span<const uint32_t>{identity.data(), identity.size()};
  const auto input_capacity = uint32_t(x->Shape()[2]), output_capacity = uint32_t(attrs.capacity);
  RecurrenceGuard guard;
  guard.AddInput(Words(e)).AddOutput(Words(out)).SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({input_capacity, output_capacity, uint32_t(valid_pair), gsl::span<const uint32_t>{in.data(), in.size()}, opaque});
  ORT_RETURN_IF_ERROR(c.RunProgram(guard));
  RecurrenceData data;
  data.AddInputs({{x, ProgramTensorMetadataDependency::None}, {w, ProgramTensorMetadataDependency::None}, {r, ProgramTensorMetadataDependency::None}, {b, ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out))})
      .AddOutputs({{y, ProgramTensorMetadataDependency::None}, {h, ProgramTensorMetadataDependency::None}, {cell, ProgramTensorMetadataDependency::None}})
      .SetWorkgroupSize(256)
      .SetDispatchGroupSize(2)
      .AddUniformVariables({input_capacity, output_capacity, opaque, uint32_t(x->Shape()[1]), uint32_t(r->Shape()[2])});
  ORT_RETURN_IF_ERROR(c.RunProgram(data));
  const uint64_t gx = std::min(groups, uint64_t(l.maxComputeWorkgroupsPerDimension));
  const uint64_t gy = (groups + gx - 1) / gx;
  RecurrenceMerged copy;
  copy.AddInputs({{y, ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out))})
      .AddOutput({merged, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(64)
      .SetDispatchGroupSize(uint32_t(gx), uint32_t(gy))
      .AddUniformVariables({input_capacity, output_capacity, opaque});
  ORT_RETURN_IF_ERROR(c.RunProgram(copy));
  RecurrenceMoments moments;
  moments.AddInputs({{merged, ProgramTensorMetadataDependency::None}, Words(static_cast<const Tensor*>(out))})
      .AddOutputs({{mean, ProgramTensorMetadataDependency::None}, {variance, ProgramTensorMetadataDependency::None}})
      .SetWorkgroupSize(64)
      .SetDispatchGroupSize(8)
      .AddUniformVariables({input_capacity, output_capacity, opaque});
  ORT_RETURN_IF_ERROR(c.RunProgram(moments));
  LOGS(c.Logger(), INFO) << "M0_RW07_RECURRENCE_V2 session=" << session << " run=" << run
                         << " source_sidecar=" << e->DataRaw() << " source_generation=" << source.generation
                         << " source_carrier=" << x->DataRaw() << " source_carrier_generation=" << carrier.generation
                         << " source_pair_valid=" << valid_pair << " output_sidecar=" << out->DataRaw()
                         << " output_generation=" << output.generation << " output_carrier=" << merged->DataRaw()
                         << " output_carrier_generation=" << output.data_generation << " role=5 instance=1006";
  for (const auto* t : {y, h, cell, merged, mean, variance, out}) {
    LOGS(c.Logger(), INFO) << "M0_RW07_RECURRENCE_RESOURCE handle=" << t->DataRaw() << " physical_bytes=" << t->SizeInBytes();
  }
  return Status::OK();
}
class RecurrenceExtent final : public WebGpuKernel {
 public:
  explicit RecurrenceExtent(const OpKernelInfo& info) : WebGpuKernel{info}, attrs_(info) {}
  Status ComputeInternal(ComputeContext& c) const override { return ComputeOwnedRecurrence(c, c.Input<Tensor>(0), c.Input<Tensor>(4), attrs_); }

 private:
  Attributes attrs_;
};
class RecurrenceMutation final : public Program<RecurrenceMutation> {
 public:
  RecurrenceMutation() : Program{"RW07V2RecurrenceExtentFaultMutation"} {}
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
    set(16, "4u", "641u");
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
class RecurrenceFault final : public WebGpuKernel {
 public:
  explicit RecurrenceFault(const OpKernelInfo& info) : WebGpuKernel{info}, attrs_(info), fault_(info.GetAttrOrDefault<int64_t>("fault_case", 0)) {
    ORT_ENFORCE(fault_ >= 0 && fault_ <= 22, "RW07 recurrence V2 closed static fault case");
  }
  Status ComputeInternal(ComputeContext& c) const override {
    const auto mode = c.M0FullGpuFaultModeV2();
    ORT_RETURN_IF_NOT((mode <= 4 || mode == 10 || mode == 11 || mode == 12 || mode == 20 || mode == 21 || mode == 22) && (fault_ == 0 || mode == 2),
                      "RW07 recurrence V2 certified fault selection");
    const bool retained = mode == 1 || mode == 4;
    const auto* x = c.Input<Tensor>(retained ? 6 : 0);
    const auto* e = c.Input<Tensor>(retained ? 5 : 4);
    ORT_RETURN_IF_NOT(x && e, "RW07 recurrence V2 explicit current/retained inputs");
    if (mode >= 2 && mode <= 4) {
      ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} && e->DataRaw(),
                        "RW07 recurrence V2 mutation sidecar descriptor");
      const auto& l = c.DeviceLimits();
      const uint64_t actual = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(e->DataRaw())));
      ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage >= 1 && l.maxComputeWorkgroupsPerDimension >= 1 &&
                            l.maxComputeWorkgroupSizeX >= 1 && l.maxComputeInvocationsPerWorkgroup >= 1 && actual >= 128 &&
                            actual <= l.maxBufferSize && actual <= l.maxStorageBufferBindingSize,
                        "RW07 recurrence V2 mutation resource limits");
      const uint64_t run = c.KernelContext().M0FullGpuExtentRunIdV2();
      M0FullGpuAllocationIdentity live;
      ORT_RETURN_IF_NOT(run && c.M0FullGpuDescribeResourceV2(*e, live), "RW07 recurrence V2 live mutation descriptor");
      RecurrenceMutation mutation;
      mutation.AddOutput(Words(const_cast<Tensor*>(e))).SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({mode, uint32_t(fault_), uint32_t(run), uint32_t(run >> 32)});
      ORT_RETURN_IF_ERROR(c.RunProgram(mutation));
      LOGS(c.Logger(), INFO) << "M0_RW07_RECURRENCE_FAULT_MUTATION mode=" << mode << " fault_case=" << fault_
                             << " handle=" << e->DataRaw() << " generation=" << live.generation << " GPU_dispatch=1 host_payload_reads=0";
    }
    LOGS(c.Logger(), INFO) << "M0_RW07_RECURRENCE_FAULT_SELECTOR mode=" << mode << " selected_sidecar=" << e->DataRaw()
                           << " selected_carrier=" << x->DataRaw();
    return ComputeOwnedRecurrence(c, x, e, attrs_);
  }

 private:
  Attributes attrs_;
  int64_t fault_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuRecurrenceExtentV2KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out=std::make_unique<RecurrenceExtent>(info);return Status::OK(); };
  return {KernelDefBuilder().SetName("RecurrenceExtentV2").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuRecurrenceFaultExtentV2KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out=std::make_unique<RecurrenceFault>(info);return Status::OK(); };
  return {KernelDefBuilder().SetName("RecurrenceFaultExtentV2").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
