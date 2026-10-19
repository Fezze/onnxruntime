// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Exact original1573: window20, hop5, forward unnormalized, eleven complex bins.
#include "m0_fullgpu_stft1573_v1.h"

#include <algorithm>
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::webgpu {
namespace {
const char* OriginalWindow() {
  return "const original_window=array<u32,20>(0u,1019771126u,1036226829u,1045630416u,"
         "1051780163u,1056964608u,1059556830u,1061895308u,1063751134u,1064942648u,"
         "1065353216u,1064942648u,1063751134u,1061895308u,1059556830u,1056964608u,"
         "1051780163u,1045630416u,1036226829u,1019771126u);\n";
}
class Guard final : public Program<Guard> {
 public:
  Guard() : Program{"RW07OriginalSTFT1573DiagnosticGuardV1"} {}
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& hop = s.AddInput("hop", ShaderUsage::UseValueTypeAlias);
    const auto& frame = s.AddInput("frame", ShaderUsage::UseValueTypeAlias);
    const auto& window = s.AddInput("window", ShaderUsage::UseValueTypeAlias);
    const auto& status = s.AddOutput("status", ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << OriginalWindow();
    auto& f = s.MainFunctionBody();
    f << status.SetByOffset("0u", "103u")
      << "if(" << hop.GetByOffset("0u") << "!=5u||" << hop.GetByOffset("1u") << "!=0u||"
      << frame.GetByOffset("0u") << "!=20u||" << frame.GetByOffset("1u") << "!=0u){return;}"
      << "for(var n=0u;n<20u;n++){if(bitcast<u32>(" << window.GetByOffset("n")
      << ")!=original_window[n]){return;}}" << status.SetByOffset("0u", "0u");
    return Status::OK();
  }
};
class Transform final : public Program<Transform> {
 public:
  Transform() : Program{"RW07OriginalSTFT1573NumericalV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"signal_capacity", ProgramUniformVariableDataType::Uint32},
      {"frame_capacity", ProgramUniformVariableDataType::Uint32});

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& x = shader.AddInput("signal", ShaderUsage::UseValueTypeAlias);
    const auto& window = shader.AddInput("window", ShaderUsage::UseValueTypeAlias);
    const auto& status = shader.AddInput("status", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("spectrum", ShaderUsage::UseValueTypeAlias);
    // Fixed roots avoid driver trigonometric argument reduction and do not
    // regenerate the original window. Index reduction happens in integers.
    shader.AdditionalImplementation()
        << "const cosine=array<f32,20>(1.0,0.9510565162951535,0.8090169943749475,"
           "0.5877852522924731,0.30901699437494745,0.0,-0.30901699437494745,"
           "-0.5877852522924731,-0.8090169943749475,-0.9510565162951535,-1.0,"
           "-0.9510565162951535,-0.8090169943749475,-0.5877852522924731,"
           "-0.30901699437494745,0.0,0.30901699437494745,0.5877852522924731,"
           "0.8090169943749475,0.9510565162951535);\n"
        << "const sine=array<f32,20>(0.0,0.30901699437494745,0.5877852522924731,"
           "0.8090169943749475,0.9510565162951535,1.0,0.9510565162951535,"
           "0.8090169943749475,0.5877852522924731,0.30901699437494745,0.0,"
           "-0.30901699437494745,-0.5877852522924731,-0.8090169943749475,"
           "-0.9510565162951535,-1.0,-0.9510565162951535,-0.8090169943749475,"
           "-0.5877852522924731,-0.30901699437494745);\n";
    auto& f = shader.MainFunctionBody();
    f << "if(global_idx>=11u*uniforms.frame_capacity){return;}"
      << "let base=2u*global_idx;" << y.SetByOffset("base", "0.0")
      << y.SetByOffset("base+1u", "0.0")
      << "if(" << status.GetByOffset("0u") << "!=0u){return;}"
      << "let frame=global_idx/11u;let bin=global_idx%11u;"
      << "var re:f32=0.0;var im:f32=0.0;for(var n=0u;n<20u;n++){"
      << "let sample:f32=" << x.GetByOffset("5u*frame+n") << "*" << window.GetByOffset("n") << ";"
      << "let root=(bin*n)%20u;let rp:f32=sample*cosine[root];"
         "let ip:f32=sample*(-sine[root]);re=re+rp;im=im+ip;}"
      << y.SetByOffset("base", "re") << y.SetByOffset("base+1u", "im");
    return Status::OK();
  }
};
class Diagnostic final : public WebGpuKernel {
 public:
  explicit Diagnostic(const OpKernelInfo& info) : WebGpuKernel{info} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* signal = c.Input<Tensor>(0);
    const auto* hop = c.Input<Tensor>(1);
    const auto* window = c.Input<Tensor>(2);
    const auto* frame = c.Input<Tensor>(3);
    ORT_RETURN_IF_NOT(signal && signal->IsDataType<float>() && signal->Shape().NumDimensions() == 2 &&
                          signal->Shape()[0] == 1 && signal->Shape()[1] >= 20 && signal->Shape()[1] <= 16777216 &&
                          window && window->IsDataType<float>() && window->Shape() == TensorShape{20} &&
                          hop && hop->IsDataType<int64_t>() && hop->Shape().NumDimensions() == 0 &&
                          frame && frame->IsDataType<int64_t>() && frame->Shape().NumDimensions() == 0,
                      "RW07 original1573 diagnostic external physical descriptors");
    // This external-boundary diagnostic has actual N samples, with no padded
    // logical extent. It cannot substitute for the missing original1571 owner.
    // Bound N so the original CPU frame-count FP32 conversion is exact and its
    // division cannot round a fractional quotient up to the next integer.
    const int64_t frames = 1 + (signal->Shape()[1] - 20) / 5;
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 4 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limits.maxComputeWorkgroupsPerDimension > 0,
                      "RW07 original1573 diagnostic dispatch resources");
    auto binding = [&](const Tensor* tensor) -> Status {
      ORT_RETURN_IF_NOT(tensor->Location().device.Type() == OrtDevice::GPU && tensor->DataRaw() &&
                            tensor->SizeInBytes() > 0 && tensor->SizeInBytes() <= limits.maxBufferSize &&
                            tensor->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 original1573 diagnostic binding descriptor");
      const uint64_t actual = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(tensor->DataRaw())));
      ORT_RETURN_IF_NOT(actual >= tensor->SizeInBytes() && actual <= limits.maxBufferSize &&
                            actual <= limits.maxStorageBufferBindingSize,
                        "RW07 original1573 diagnostic actual whole binding");
      return Status::OK();
    };
    for (const auto* input : {signal, hop, window, frame}) ORT_RETURN_IF_ERROR(binding(input));
    const uint64_t count = 22 * uint64_t(frames), groups = (count / 2 + 63) / 64;
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(count <= UINT32_MAX && count * 4 <= limits.maxBufferSize &&
                          count * 4 <= limits.maxStorageBufferBindingSize && groups <= limit * limit,
                      "RW07 original1573 diagnostic output/dispatch capacity");
    auto* spectrum = c.Output(0, TensorShape{1, frames, 11, 2});
    auto* status = c.Output(1, TensorShape{1});
    ORT_RETURN_IF_NOT(spectrum->IsDataType<float>() && status->IsDataType<uint32_t>(),
                      "RW07 original1573 diagnostic output types");
    ORT_RETURN_IF_ERROR(binding(spectrum));
    ORT_RETURN_IF_ERROR(binding(status));
    ORT_RETURN_IF_NOT(spectrum->DataRaw() != status->DataRaw(), "RW07 original1573 distinct outputs");
    for (const auto* input : {signal, hop, window, frame}) {
      ORT_RETURN_IF_NOT(spectrum->DataRaw() != input->DataRaw() && status->DataRaw() != input->DataRaw(),
                        "RW07 original1573 distinct output/input allocations");
    }
    auto int64_words = [](const Tensor* tensor) {
      ProgramInput words(tensor, ProgramTensorMetadataDependency::None, TensorShape{2}, 1);
      words.var_type = ProgramVariableDataType::Uint32;
      return words;
    };
    Guard guard;
    guard.AddInputs({int64_words(hop), int64_words(frame), {window, ProgramTensorMetadataDependency::None}})
        .AddOutput({status, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1);
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const uint64_t dx = std::min(groups, limit), dy = (groups + dx - 1) / dx;
    Transform program;
    program.AddInputs({{signal, ProgramTensorMetadataDependency::None},
                       {window, ProgramTensorMetadataDependency::None}, {status, ProgramTensorMetadataDependency::None}})
        .AddOutput({spectrum, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(dx), uint32_t(dy))
        .AddUniformVariables({uint32_t(signal->Shape()[1]), uint32_t(frames)});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    LOGS(c.Logger(), INFO) << "M0_RW07_STFT1573_DIAGNOSTIC_V1 external_original_boundary=1571"
                          << " signal=" << signal->DataRaw() << " spectrum=" << spectrum->DataRaw()
                          << " status=" << status->DataRaw() << " physical_samples=" << signal->Shape()[1]
                          << " physical_frames=" << frames << " GPU_dispatches=2 host_payload_reads=0"
                          << " original1571_owner_qualified=0 full_model_admission=0";
    return Status::OK();
  }
};
}  // namespace

KernelCreateInfo CreateM0FullGpuSTFT1573DiagnosticV1KernelInfo() {
  KernelCreatePtrFn creator = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& kernel) -> Status {
    kernel = std::make_unique<Diagnostic>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("STFT1573DiagnosticV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), creator};
}
}  // namespace onnxruntime::webgpu
