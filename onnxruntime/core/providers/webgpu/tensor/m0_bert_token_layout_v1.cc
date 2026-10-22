// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Closed original B1/T<=512 BERT embedding, mask and attention layout seams.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
Status Binding(ComputeContext& c, const Tensor* t) {
  const auto& l = c.DeviceLimits();
  ORT_RETURN_IF_NOT(Gpu(t) && t->DataRaw() && t->SizeInBytes() && t->SizeInBytes() <= l.maxBufferSize &&
                        t->SizeInBytes() <= l.maxStorageBufferBindingSize, "RW07 BERT binding descriptor");
  const auto actual = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
  ORT_RETURN_IF_NOT(actual >= t->SizeInBytes() && actual <= l.maxBufferSize && actual <= l.maxStorageBufferBindingSize,
                    "RW07 BERT actual backing buffer limits");
  return Status::OK();
}
ProgramInput Words(const Tensor* tensor, uint32_t words) {
  ProgramInput input(tensor, ProgramTensorMetadataDependency::None, TensorShape{words}, 1);
  input.var_type = ProgramVariableDataType::Uint32;
  return input;
}
ProgramOutput Words(Tensor* tensor, uint32_t words) {
  ProgramOutput output(tensor, ProgramTensorMetadataDependency::None, TensorShape{words}, 1);
  output.var_type = ProgramVariableDataType::Uint32;
  return output;
}
class TokenGuard final : public Program<TokenGuard> {
 public:
  TokenGuard() : Program{"RW07BertTokenGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"tokens", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& tokens = s.AddInput("tokens", ShaderUsage::UseValueTypeAlias);
    const auto& types = s.AddInput("types", ShaderUsage::UseValueTypeAlias);
    const auto& positions = s.AddInput("positions", ShaderUsage::UseValueTypeAlias);
    const auto& status = s.AddOutput("status", ShaderUsage::UseValueTypeAlias);
    const auto& mask = s.AddOutput("mask", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << status.SetByOffset("0u", "0u")
      << "for(var t=0u;t<uniforms.tokens;t++){"
      << "if(" << tokens.GetByOffset("2u*t+1u") << "!=0u||" << tokens.GetByOffset("2u*t") << ">177u||"
      << types.GetByOffset("2u*t+1u") << "!=0u||" << types.GetByOffset("2u*t") << ">1u||"
      << positions.GetByOffset("2u*t+1u") << "!=0u||" << positions.GetByOffset("2u*t") << ">511u){"
      << status.SetByOffset("0u", "103u") << "}"
      // Exact original COS305(+1), broadcast, Sub414 and Where427. Token 0
      // is not a padding mask: every original mask element is positive zero.
      << "let original_cos:f32=1.0;let sub:f32=1.0-original_cos;"
      << "let masked=select(sub,bitcast<f32>(0xff7fffffu),sub!=0.0);"
      << mask.SetByOffset("t", "masked") << "}";
    return Status::OK();
  }
};
class Embedding final : public Program<Embedding> {
 public:
  Embedding() : Program{"RW07BertEmbeddingV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"elements", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& tokens = s.AddInput("tokens", ShaderUsage::UseValueTypeAlias);
    const auto& word = s.AddInput("word", ShaderUsage::UseValueTypeAlias);
    const auto& type_embedding = s.AddInput("type_embedding", ShaderUsage::UseValueTypeAlias);
    const auto& position = s.AddInput("position", ShaderUsage::UseValueTypeAlias);
    const auto& types = s.AddInput("types", ShaderUsage::UseValueTypeAlias);
    const auto& positions = s.AddInput("positions", ShaderUsage::UseValueTypeAlias);
    const auto& status = s.AddInput("status", ShaderUsage::UseValueTypeAlias);
    const auto& out = s.AddOutput("embedding", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.elements){return;}" << out.SetByOffset("global_idx", "0.0")
      << "if(" << status.GetByOffset("0u") << "!=0u){return;}let t=global_idx/128u;let channel=global_idx%128u;"
      << "let token=" << tokens.GetByOffset("2u*t") << ";let type_id=" << types.GetByOffset("2u*t")
      << ";let position_id=" << positions.GetByOffset("2u*t") << ";"
      << "let first=" << word.GetByOffset("token*128u+channel") << "+" << type_embedding.GetByOffset("type_id*128u+channel")
      << ";let second=first+" << position.GetByOffset("position_id*128u+channel") << ";"
      << out.SetByOffset("global_idx", "second");
    return Status::OK();
  }
};
class EmbeddingMask final : public WebGpuKernel {
 public:
  explicit EmbeddingMask(const OpKernelInfo& info) : WebGpuKernel{info} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* tokens = c.Input<Tensor>(0);
    ORT_RETURN_IF_NOT(Gpu(tokens) && tokens->IsDataType<int64_t>() && tokens->Shape().NumDimensions() == 2 &&
                          tokens->Shape()[0] == 1 && tokens->Shape()[1] > 0 && tokens->Shape()[1] <= 512,
                      "RW07 BERT admitted token descriptor [1,T], 1<=T<=512");
    const uint32_t t = uint32_t(tokens->Shape()[1]);
    const auto* word = c.Input<Tensor>(1);
    const auto* type = c.Input<Tensor>(2);
    const auto* position = c.Input<Tensor>(3);
    const auto* types = c.Input<Tensor>(4);
    const auto* positions = c.Input<Tensor>(5);
    ORT_RETURN_IF_NOT(Gpu(word) && word->IsDataType<float>() && word->Shape() == TensorShape({178, 128}) &&
                          Gpu(type) && type->IsDataType<float>() && type->Shape() == TensorShape({2, 128}) &&
                          Gpu(position) && position->IsDataType<float>() && position->Shape() == TensorShape({512, 128}) &&
                          Gpu(types) && types->IsDataType<int64_t>() && types->Shape() == TensorShape({1, 512}) &&
                          Gpu(positions) && positions->IsDataType<int64_t>() && positions->Shape() == TensorShape({1, 512}),
                      "RW07 BERT original embedding initializer descriptors");
    auto* embedding = c.Output(0, TensorShape{1, t, 128});
    auto* mask = c.Output(1, TensorShape{1, 1, 1, t});
    auto* status = c.Output(2, TensorShape{1});
    ORT_RETURN_IF_NOT(embedding && embedding->IsDataType<float>() && mask && mask->IsDataType<float>() &&
                          status && status->IsDataType<uint32_t>(), "RW07 BERT actual output types");
    for (const auto* input : {tokens, word, type, position, types, positions}) {
      ORT_RETURN_IF_ERROR(Binding(c, input));
      ORT_RETURN_IF_NOT(input->DataRaw() != embedding->DataRaw() && input->DataRaw() != mask->DataRaw() &&
                            input->DataRaw() != status->DataRaw(), "RW07 BERT embedding input/output alias");
    }
    for (const auto* output : {embedding, mask, status}) ORT_RETURN_IF_ERROR(Binding(c, output));
    ORT_RETURN_IF_NOT(embedding->DataRaw() != mask->DataRaw() && embedding->DataRaw() != status->DataRaw() &&
                          mask->DataRaw() != status->DataRaw(), "RW07 BERT output alias");
    const auto& l = c.DeviceLimits();
    ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage >= 8 && l.maxComputeWorkgroupSizeX >= 64 &&
                          l.maxComputeInvocationsPerWorkgroup >= 64 && t * 2 <= l.maxComputeWorkgroupsPerDimension,
                      "RW07 BERT embedding actual dispatch limits");
    TokenGuard guard;
    guard.AddInputs({Words(tokens, t * 2), Words(types, 1024), Words(positions, 1024)})
      .AddOutput(Words(status, 1)).AddOutput({mask, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({t});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    Embedding gather;
    gather.AddInput(Words(tokens, t * 2))
      .AddInputs({{word, ProgramTensorMetadataDependency::None}, {type, ProgramTensorMetadataDependency::None},
                  {position, ProgramTensorMetadataDependency::None}})
      .AddInputs({Words(types, 1024), Words(positions, 1024), Words(static_cast<const Tensor*>(status), 1)})
      .AddOutput({embedding, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
      .SetDispatchGroupSize(t * 2).AddUniformVariables({t * 128});
    ORT_RETURN_IF_ERROR(c.RunProgram(gather));
    LOGS(c.Logger(), INFO) << "M0_RW07_BERT_EMBEDDING_V1 GPU_guard=1 token_payload_host_reads=0 status="
                          << status->DataRaw() << " status_requires_final_publication_gate=1";
    return Status::OK();
  }
};
class AttentionLayoutProgram final : public Program<AttentionLayoutProgram> {
 public:
  explicit AttentionLayoutProgram(uint32_t kind) : Program{"RW07BertAttentionLayoutV1"}, kind_(kind) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"tokens", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("input", ShaderUsage::UseValueTypeAlias);
    const auto& out = s.AddOutput("output", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.tokens*768u){return;}";
    if (kind_ == 3) {
      f << "let t=global_idx/768u;let channel=global_idx%768u;let head=channel/64u;let unit=channel%64u;"
        << out.SetByOffset("global_idx", x.GetByOffset("(head*uniforms.tokens+t)*64u+unit"));
    } else {
      if (kind_ == 1) {
        f << "let t=global_idx%uniforms.tokens;let unit=(global_idx/uniforms.tokens)%64u;"
          << "let head=global_idx/(uniforms.tokens*64u);";
      } else {
        f << "let unit=global_idx%64u;let t=(global_idx/64u)%uniforms.tokens;"
          << "let head=global_idx/(uniforms.tokens*64u);";
      }
      f << "let value=" << x.GetByOffset("t*768u+head*64u+unit") << ";";
      if (kind_ == 0 || kind_ == 1) {
        f << "let root=sqrt(64.0);let reciprocal=1.0/root;let scale=sqrt(reciprocal);let scaled=value*scale;"
          << out.SetByOffset("global_idx", "scaled");
      } else {
        f << out.SetByOffset("global_idx", "value");
      }
    }
    return Status::OK();
  }
 private:
  uint32_t kind_;
};
class AttentionLayout final : public WebGpuKernel {
 public:
  explicit AttentionLayout(const OpKernelInfo& info) : WebGpuKernel{info}, kind_(info.GetAttrOrDefault<int64_t>("layout_kind", -1)) {
    ORT_ENFORCE(kind_ >= 0 && kind_ <= 3, "RW07 BERT closed attention layout kind");
  }
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x = c.Input<Tensor>(0);
    ORT_RETURN_IF_NOT(Gpu(x) && x->IsDataType<float>(), "RW07 BERT attention GPU FLOAT input");
    const auto& shape = x->Shape();
    const bool flatten = kind_ == 3;
    ORT_RETURN_IF_NOT((flatten && shape.NumDimensions() == 4 && shape[0] == 1 && shape[1] == 12 &&
                           shape[2] > 0 && shape[2] <= 512 && shape[3] == 64) ||
                          (!flatten && shape.NumDimensions() == 3 && shape[0] == 1 &&
                           shape[1] > 0 && shape[1] <= 512 && shape[2] == 768),
                      "RW07 BERT attention exact 12x64 layout descriptor");
    const uint32_t t = uint32_t(shape[flatten ? 2 : 1]);
    const TensorShape output_shape = flatten ? TensorShape{1, t, 768} :
        kind_ == 1 ? TensorShape{1, 12, 64, t} : TensorShape{1, 12, t, 64};
    auto* out = c.Output(0, output_shape);
    ORT_RETURN_IF_ERROR(Binding(c, x));
    ORT_RETURN_IF_ERROR(Binding(c, out));
    ORT_RETURN_IF_NOT(out->IsDataType<float>() && out->DataRaw() != x->DataRaw(), "RW07 BERT attention output alias/type");
    const auto& l = c.DeviceLimits();
    ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage >= 2 && l.maxComputeWorkgroupSizeX >= 64 &&
                          l.maxComputeInvocationsPerWorkgroup >= 64 && t * 12 <= l.maxComputeWorkgroupsPerDimension,
                      "RW07 BERT attention dispatch limits");
    AttentionLayoutProgram program{uint32_t(kind_)};
    program.CacheHint(std::to_string(kind_));
    program.AddInput({x, ProgramTensorMetadataDependency::None}).AddOutput({out, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(64).SetDispatchGroupSize(t * 12).AddUniformVariables({t});
    return c.RunProgram(program);
  }
 private:
  int64_t kind_;
};
template <typename Kernel>
KernelCreateInfo Create(const char* name) {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Kernel>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName(name).SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace
KernelCreateInfo CreateM0BertEmbeddingMaskV1KernelInfo() { return Create<EmbeddingMask>("BertEmbeddingMaskV1"); }
KernelCreateInfo CreateM0BertAttentionLayoutV1KernelInfo() { return Create<AttentionLayout>("BertAttentionLayoutV1"); }
}  // namespace onnxruntime::webgpu
