// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Fixed token-derived T: original text Gather2/Transpose6 and merged LSTM449.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::webgpu {
namespace {
Status Binding(ComputeContext& c, const Tensor* t) {
  ORT_RETURN_IF_NOT(t && t->Location().device.Type() == OrtDevice::GPU && t->DataRaw() && t->SizeInBytes() > 0,
                    "RW07 text encoder GPU binding descriptor");
  const auto& limits = c.DeviceLimits();
  const uint64_t bytes = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw())));
  ORT_RETURN_IF_NOT(bytes >= t->SizeInBytes() && bytes <= limits.maxBufferSize && bytes <= limits.maxStorageBufferBindingSize,
                    "RW07 text encoder actual binding limit");
  return Status::OK();
}
Status Limits(ComputeContext& c, uint32_t count, uint32_t buffers) {
  const auto& limits = c.DeviceLimits();
  ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= buffers && limits.maxComputeWorkgroupSizeX >= 64 &&
                        limits.maxComputeInvocationsPerWorkgroup >= 64 &&
                        (uint64_t(count) + 63) / 64 <= limits.maxComputeWorkgroupsPerDimension,
                    "RW07 text encoder dispatch limits");
  return Status::OK();
}
class TextTokenGuard final : public Program<TextTokenGuard> {
 public:
  TextTokenGuard() : Program{"RW07TextTokenGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"tokens", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& token = s.AddInput("token_words", ShaderUsage::UseValueTypeAlias);
    const auto& status = s.AddOutput("status", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << status.SetByOffset("0u", "103u")
      << "for(var t=0u;t<uniforms.tokens;t++){if(" << token.GetByOffset("2u*t+1u") << "!=0u||"
      << token.GetByOffset("2u*t") << ">=178u){return;}}" << status.SetByOffset("0u", "0u");
    return Status::OK();
  }
};
class TextEmbeddingProgram final : public Program<TextEmbeddingProgram> {
 public:
  TextEmbeddingProgram() : Program{"RW07TextTokenEmbeddingV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"tokens", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& token = s.AddInput("token_words", ShaderUsage::UseValueTypeAlias);
    const auto& weight = s.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const auto& status = s.AddInput("status", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("embedded", ShaderUsage::UseValueTypeAlias);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=512u*uniforms.tokens){return;}" << y.SetByOffset("global_idx", "0.0")
      << "if(" << status.GetByOffset("0u") << "!=0u){return;}let t=global_idx%uniforms.tokens;"
      << "let channel=global_idx/uniforms.tokens;let id=" << token.GetByOffset("2u*t") << ";"
      << y.SetByOffset("global_idx", weight.GetByOffset("id*512u+channel"));
    return Status::OK();
  }
};
class TextEmbedding final : public WebGpuKernel {
 public:
  explicit TextEmbedding(const OpKernelInfo& info) : WebGpuKernel{info} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* tokens = c.Input<Tensor>(0);
    const auto* weight = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(tokens && tokens->IsDataType<int64_t>() && tokens->Shape().NumDimensions() == 2 &&
                          tokens->Shape()[0] == 1 && tokens->Shape()[1] >= 1 && tokens->Shape()[1] <= 512 &&
                          weight && weight->IsDataType<float>() && weight->Shape() == TensorShape({178,512}),
                      "RW07 text encoder original token/table descriptors");
    ORT_RETURN_IF_ERROR(Binding(c, tokens));
    ORT_RETURN_IF_ERROR(Binding(c, weight));
    const uint32_t count = uint32_t(tokens->Shape()[1]);
    ORT_RETURN_IF_ERROR(Limits(c, count * 512, 4));
    auto* y = c.Output(0, TensorShape{1,512,count});
    auto* status = c.Output(1, TensorShape{1});
    ORT_RETURN_IF_NOT(y && y->IsDataType<float>() && status && status->IsDataType<uint32_t>(),
                      "RW07 text encoder output types");
    ORT_RETURN_IF_ERROR(Binding(c, y));
    ORT_RETURN_IF_ERROR(Binding(c, status));
    ORT_RETURN_IF_NOT(y->DataRaw() != status->DataRaw(), "RW07 text encoder distinct outputs");
    for (const auto* input : {tokens, weight}) {
      ORT_RETURN_IF_NOT(input->DataRaw() != y->DataRaw() && input->DataRaw() != status->DataRaw(),
                        "RW07 text encoder source/output alias");
    }
    auto token_words = [&]() {
      ProgramInput p(tokens, ProgramTensorMetadataDependency::None, TensorShape{2 * count}, 1);
      p.var_type = ProgramVariableDataType::Uint32;
      return p;
    };
    TextTokenGuard guard;
    guard.AddInput(token_words()).AddOutput({status, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({count});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    TextEmbeddingProgram data;
    data.AddInputs({token_words(), {weight, ProgramTensorMetadataDependency::None},
                    {status, ProgramTensorMetadataDependency::None}}).AddOutput({y, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize(count * 8).AddUniformVariables({count});
    ORT_RETURN_IF_ERROR(c.RunProgram(data));
    LOGS(c.Logger(), INFO) << "M0_RW07_TEXT_TOKEN_EMBEDDING_V1 tokens=" << count
                          << " GPU_dispatches=2 host_payload_reads=0";
    return Status::OK();
  }
};
class TextMergeProgram final : public Program<TextMergeProgram> {
 public:
  TextMergeProgram() : Program{"RW07TextMergeDirectionsV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"tokens", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("sequence", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("features", ShaderUsage::UseValueTypeAlias);
    s.MainFunctionBody() << "if(global_idx>=512u*uniforms.tokens){return;}"
      << "let token=global_idx%uniforms.tokens;let channel=global_idx/uniforms.tokens;"
      << y.SetByOffset("global_idx", x.GetByOffset("token*512u+channel"));
    return Status::OK();
  }
};
class TextMerge final : public WebGpuKernel {
 public:
  explicit TextMerge(const OpKernelInfo& info) : WebGpuKernel{info} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* sequence = c.Input<Tensor>(0);
    ORT_RETURN_IF_NOT(sequence && sequence->IsDataType<float>() && sequence->Shape().NumDimensions() == 4 &&
                          sequence->Shape()[0] >= 1 && sequence->Shape()[0] <= 512 && sequence->Shape()[1] == 2 &&
                          sequence->Shape()[2] == 1 && sequence->Shape()[3] == 256,
                      "RW07 text encoder original449 sequence descriptors");
    const uint32_t count = uint32_t(sequence->Shape()[0]);
    auto* y = c.Output(0, TensorShape{1,512,count});
    ORT_RETURN_IF_NOT(y && y->IsDataType<float>(), "RW07 text encoder merged output type");
    ORT_RETURN_IF_ERROR(Binding(c, sequence));
    ORT_RETURN_IF_ERROR(Binding(c, y));
    ORT_RETURN_IF_ERROR(Limits(c, count * 512, 2));
    ORT_RETURN_IF_NOT(y->DataRaw() != sequence->DataRaw(), "RW07 text encoder merged/source alias");
    TextMergeProgram program;
    program.AddInput({sequence, ProgramTensorMetadataDependency::None}).AddOutput({y, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize(count * 8).AddUniformVariables({count});
    return c.RunProgram(program);
  }
};
}  // namespace
KernelCreateInfo CreateM0TextTokenEmbeddingV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<TextEmbedding>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("TextTokenEmbeddingV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0TextMergeDirectionsV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<TextMerge>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("TextMergeDirectionsV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
