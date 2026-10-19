// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Fixed B1 external-BERT diagnostic; no owned logical-extent certification.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::webgpu {
namespace {
bool GpuFloat(const Tensor* tensor) {
  return tensor && tensor->Location().device.Type() == OrtDevice::GPU && tensor->IsDataType<float>();
}
Status Binding(ComputeContext& context, const Tensor* tensor) {
  const auto& limits = context.DeviceLimits();
  ORT_RETURN_IF_NOT(GpuFloat(tensor) && tensor->DataRaw() && tensor->SizeInBytes() > 0 &&
                        tensor->SizeInBytes() <= limits.maxBufferSize &&
                        tensor->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                    "RW07 predictor chain: GPU FLOAT binding descriptor");
  const auto actual = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(tensor->DataRaw())));
  ORT_RETURN_IF_NOT(actual >= tensor->SizeInBytes() && actual <= limits.maxBufferSize &&
                        actual <= limits.maxStorageBufferBindingSize,
                    "RW07 predictor chain: actual backing buffer limits");
  return Status::OK();
}
Status Dispatch(ComputeContext& context, uint32_t elements, uint32_t buffers) {
  const auto& limits = context.DeviceLimits();
  ORT_RETURN_IF_NOT(elements && limits.maxStorageBuffersPerShaderStage >= buffers &&
                        limits.maxComputeWorkgroupSizeX >= 64 && limits.maxComputeInvocationsPerWorkgroup >= 64 &&
                        (uint64_t(elements) + 63) / 64 <= limits.maxComputeWorkgroupsPerDimension,
                    "RW07 predictor chain: dispatch device limits");
  return Status::OK();
}
Status Tokens(const Tensor* input, uint32_t& tokens) {
  ORT_RETURN_IF_NOT(GpuFloat(input) && input->Shape().NumDimensions() == 3 && input->Shape()[0] == 1 &&
                        input->Shape()[1] > 0 && input->Shape()[1] <= 512 && input->Shape()[2] == 512,
                    "RW07 predictor chain: admitted [1,T,512], 1<=T<=512");
  tokens = uint32_t(input->Shape()[1]);
  return Status::OK();
}
class ConcatStyleProgram final : public Program<ConcatStyleProgram> {
 public:
  ConcatStyleProgram() : Program{"RW07PredictorConcatStyleV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"elements", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& x = shader.AddInput("features", ShaderUsage::UseValueTypeAlias);
    const auto& style = shader.AddInput("style", ShaderUsage::UseValueTypeAlias);
    const auto& out = shader.AddOutput("concatenated", ShaderUsage::UseValueTypeAlias);
    shader.MainFunctionBody() << "if(global_idx>=uniforms.elements){return;}"
      << "let token=global_idx/640u;let channel=global_idx%640u;"
      << "if(channel<512u){" << out.SetByOffset("global_idx", x.GetByOffset("token*512u+channel"))
      << "}else{" << out.SetByOffset("global_idx", style.GetByOffset("128u+channel-512u")) << "}";
    return Status::OK();
  }
};
class MergeDirectionsProgram final : public Program<MergeDirectionsProgram> {
 public:
  MergeDirectionsProgram() : Program{"RW07PredictorMergeDirectionsV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"elements", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& y = shader.AddInput("y", ShaderUsage::UseValueTypeAlias);
    const auto& out = shader.AddOutput("merged", ShaderUsage::UseValueTypeAlias);
    // With admitted B1, original Transpose[0,2,1,3] -> Reshape[T,1,512]
    // -> Transpose[1,0,2] retains the same scalar order [token,direction,unit].
    shader.MainFunctionBody() << "if(global_idx>=uniforms.elements){return;}"
                             << out.SetByOffset("global_idx", y.GetByOffset("global_idx"));
    return Status::OK();
  }
};
class StyleProjectionProgram final : public Program<StyleProjectionProgram> {
 public:
  StyleProjectionProgram() : Program{"RW07PredictorStyleProjectionV1"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& style = shader.AddInput("style", ShaderUsage::UseValueTypeAlias);
    const auto& w = shader.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& out = shader.AddOutput("affine", ShaderUsage::UseValueTypeAlias);
    shader.MainFunctionBody() << "if(global_idx>=1024u){return;}var sum:f32=0.0;"
      << "for(var k=0u;k<128u;k++){let product:f32=" << style.GetByOffset("128u+k") << "*"
      << w.GetByOffset("global_idx*128u+k") << ";sum=sum+product;}let value=sum+" << b.GetByOffset("global_idx") << ";"
      << out.SetByOffset("global_idx", "value");
    return Status::OK();
  }
};
class StyleApplyProgram final : public Program<StyleApplyProgram> {
 public:
  StyleApplyProgram() : Program{"RW07PredictorStyleApplyV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"elements", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& x = shader.AddInput("normalized", ShaderUsage::UseValueTypeAlias);
    const auto& affine = shader.AddInput("affine", ShaderUsage::UseValueTypeAlias);
    const auto& out = shader.AddOutput("styled", ShaderUsage::UseValueTypeAlias);
    shader.MainFunctionBody() << "if(global_idx>=uniforms.elements){return;}let channel=global_idx%512u;"
      << "let gamma=" << affine.GetByOffset("channel") << "+1.0;"
      << "let scaled=gamma*" << x.GetByOffset("global_idx") << ";let shifted=scaled+"
      << affine.GetByOffset("512u+channel") << ";" << out.SetByOffset("global_idx", "shifted");
    return Status::OK();
  }
};
class ConcatStyle final : public WebGpuKernel {
 public:
  explicit ConcatStyle(const OpKernelInfo& info) : WebGpuKernel{info} {}
  Status ComputeInternal(ComputeContext& context) const override {
    const auto* x = context.Input<Tensor>(0);
    const auto* style = context.Input<Tensor>(1);
    uint32_t tokens = 0;
    ORT_RETURN_IF_ERROR(Tokens(x, tokens));
    ORT_RETURN_IF_NOT(GpuFloat(style) && style->Shape() == TensorShape({1, 256}),
                      "RW07 predictor chain: original style [1,256]");
    auto* out = context.Output(0, TensorShape{1, tokens, 640});
    for (const auto* input : {x, style}) {
      ORT_RETURN_IF_ERROR(Binding(context, input));
      ORT_RETURN_IF_NOT(out->DataRaw() != input->DataRaw(), "RW07 predictor chain: concat alias");
    }
    ORT_RETURN_IF_ERROR(Binding(context, out));
    ORT_RETURN_IF_ERROR(Dispatch(context, tokens * 640, 3));
    ConcatStyleProgram program;
    program.AddInputs({{x, ProgramTensorMetadataDependency::None}, {style, ProgramTensorMetadataDependency::None}})
      .AddOutput({out, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
      .SetDispatchGroupSize((tokens * 640 + 63) / 64).AddUniformVariables({tokens * 640});
    return context.RunProgram(program);
  }
};
class MergeDirections final : public WebGpuKernel {
 public:
  explicit MergeDirections(const OpKernelInfo& info) : WebGpuKernel{info} {}
  Status ComputeInternal(ComputeContext& context) const override {
    const auto* y = context.Input<Tensor>(0);
    ORT_RETURN_IF_NOT(GpuFloat(y) && y->Shape().NumDimensions() == 4 && y->Shape()[0] > 0 &&
                          y->Shape()[0] <= 512 && y->Shape()[1] == 2 && y->Shape()[2] == 1 && y->Shape()[3] == 256,
                      "RW07 predictor chain: original Y [T,2,1,256]");
    const auto tokens = uint32_t(y->Shape()[0]);
    auto* out = context.Output(0, TensorShape{1, tokens, 512});
    ORT_RETURN_IF_ERROR(Binding(context, y));
    ORT_RETURN_IF_ERROR(Binding(context, out));
    ORT_RETURN_IF_NOT(out->DataRaw() != y->DataRaw(), "RW07 predictor chain: merge alias");
    ORT_RETURN_IF_ERROR(Dispatch(context, tokens * 512, 2));
    MergeDirectionsProgram program;
    program.AddInput({y, ProgramTensorMetadataDependency::None}).AddOutput({out, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(64).SetDispatchGroupSize(tokens * 8).AddUniformVariables({tokens * 512});
    return context.RunProgram(program);
  }
};
class StyleAffine final : public WebGpuKernel {
 public:
  explicit StyleAffine(const OpKernelInfo& info) : WebGpuKernel{info}, stage_(info.GetAttrOrDefault<int64_t>("original_stage", -1)) {
    ORT_ENFORCE(stage_ == 1 || stage_ == 3 || stage_ == 5, "RW07 predictor chain: original affine stage");
  }
  Status ComputeInternal(ComputeContext& context) const override {
    const auto* x = context.Input<Tensor>(0);
    const auto* style = context.Input<Tensor>(1);
    const auto* w = context.Input<Tensor>(2);
    const auto* b = context.Input<Tensor>(3);
    uint32_t tokens = 0;
    ORT_RETURN_IF_ERROR(Tokens(x, tokens));
    ORT_RETURN_IF_NOT(GpuFloat(style) && style->Shape() == TensorShape({1, 256}) &&
                          GpuFloat(w) && w->Shape() == TensorShape({1024, 128}) &&
                          GpuFloat(b) && b->Shape() == TensorShape({1024}),
                      "RW07 predictor chain: original style affine descriptors");
    auto projection = context.CreateGPUTensor(x->DataType(), TensorShape{1, 1024});
    auto* out = context.Output(0, TensorShape{1, tokens, 512});
    for (const auto* input : {x, style, w, b}) {
      ORT_RETURN_IF_ERROR(Binding(context, input));
      ORT_RETURN_IF_NOT(input->DataRaw() != out->DataRaw() && input->DataRaw() != projection.DataRaw(),
                        "RW07 predictor chain: affine input/output alias");
    }
    ORT_RETURN_IF_ERROR(Binding(context, out));
    ORT_RETURN_IF_ERROR(Binding(context, &projection));
    ORT_RETURN_IF_NOT(out->DataRaw() != projection.DataRaw(), "RW07 predictor chain: affine output/scratch alias");
    ORT_RETURN_IF_ERROR(Dispatch(context, 1024, 4));
    ORT_RETURN_IF_ERROR(Dispatch(context, tokens * 512, 3));
    StyleProjectionProgram project;
    project.AddInputs({{style, ProgramTensorMetadataDependency::None}, {w, ProgramTensorMetadataDependency::None},
                       {b, ProgramTensorMetadataDependency::None}})
      .AddOutput({&projection, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(16);
    ORT_RETURN_IF_ERROR(context.RunProgram(project));
    StyleApplyProgram apply;
    apply.AddInputs({{x, ProgramTensorMetadataDependency::None}, {&projection, ProgramTensorMetadataDependency::None}})
      .AddOutput({out, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
      .SetDispatchGroupSize(tokens * 8).AddUniformVariables({tokens * 512});
    return context.RunProgram(apply);
  }
 private:
  int64_t stage_;
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
KernelCreateInfo CreateM0PredictorConcatStyleV1KernelInfo() { return Create<ConcatStyle>("PredictorConcatStyleV1"); }
KernelCreateInfo CreateM0PredictorMergeDirectionsV1KernelInfo() { return Create<MergeDirections>("PredictorMergeDirectionsV1"); }
KernelCreateInfo CreateM0PredictorStyleAffineV1KernelInfo() { return Create<StyleAffine>("PredictorStyleAffineV1"); }
}  // namespace onnxruntime::webgpu
