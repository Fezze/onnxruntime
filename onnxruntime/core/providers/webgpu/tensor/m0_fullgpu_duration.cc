// rw07-fullgpu-v1 first feasibility producer. NOT a full-model admission gate.
// The token descriptor is the original input-time authority. No GPU duration,
// prefix, shape payload or intermediate extent is read by this host code.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"
#include <limits>

namespace onnxruntime::webgpu {
namespace {
ProgramOutput Words(Tensor* t, uint32_t words) {
  ProgramOutput p(t, ProgramTensorMetadataDependency::None, TensorShape{words}, 1);
  p.var_type = ProgramVariableDataType::Uint32; return p;
}
bool Device(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
void Trace(ComputeContext& c, const char* role, const Tensor* t, uint64_t epoch) {
  LOGS(c.Logger(), INFO) << "M0_RW07_RESOURCE role=" << role << " epoch=" << epoch
      << " device=" << int(t->Location().device.Type()) << " handle=" << t->DataRaw()
      << " physical_bytes=" << t->SizeInBytes()
      << " allocation_bytes=" << (t->SizeInBytes() ? wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw()))) : 0);
}

class DurationProgram final : public Program<DurationProgram> {
 public:
  DurationProgram() : Program{"RW07V1DurationExtent"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"tokens", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"epoch_low", ProgramUniformVariableDataType::Uint32},
      {"epoch_high", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& logits = s.AddInput("logits", ShaderUsage::UseValueTypeAlias);
    const auto& speed = s.AddInput("speed", ShaderUsage::UseValueTypeAlias);
    const auto& durations = s.AddOutput("durations", ShaderUsage::UseValueTypeAlias);
    const auto& prefixes = s.AddOutput("prefixes", ShaderUsage::UseValueTypeAlias);
    const auto& total = s.AddOutput("total", ShaderUsage::UseValueTypeAlias);
    const auto& extent = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() <<
        "fn rw07_finite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;}\n"
        // Explicit original Round ties-to-even. Conversion of the integral base
        // follows the GPU finite/range check (sum/speed in [0,100]).
        "fn rw07_round_even(v:f32)->f32{let base=floor(v);let fraction=v-base;"
        "if(fraction<0.5){return base;}if(fraction>0.5){return base+1.0;}"
        "return base+f32(u32(base)&1u);}\n";
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<24u;i++){" << extent.SetByOffset("i", "0u") << "}\n"
      << extent.SetByOffset("0u", "1u") << extent.SetByOffset("1u", "1u")
      << extent.SetByOffset("22u", "uniforms.epoch_low")
      << extent.SetByOffset("23u", "uniforms.epoch_high")
      << total.SetByOffset("0u", "0u") << total.SetByOffset("1u", "0u")
      << "for(var t=0u;t<uniforms.tokens;t++){"
      << durations.SetByOffset("2u*t", "0u") << durations.SetByOffset("2u*t+1u", "0u")
      << prefixes.SetByOffset("2u*t", "0u") << prefixes.SetByOffset("2u*t+1u", "0u") << "}\n"
      // This is the original Gather(-1) failure, not successful empty audio.
      << "if(uniforms.tokens==0u){" << extent.SetByOffset("20u", "1u") << "return;}\n"
      << "let scale=" << speed.GetByOffset("0u") << ";"
      << "if(!rw07_finite(scale)||scale<0.5||scale>2.0){"
      << extent.SetByOffset("20u", "101u") << "return;}\n"
      << "var prefix=0u;for(var t=0u;t<uniforms.tokens;t++){var sum=0.0;"
      << "for(var channel=0u;channel<50u;channel++){let x=" << logits.GetByOffset("t*50u+channel") << ";"
      << "let sigmoid=1.0/(1.0+exp(-x));"
      << "if(!rw07_finite(sigmoid)||sigmoid<0.0||sigmoid>1.0){"
      << extent.SetByOffset("20u", "100u") << "return;}sum+=sigmoid;}\n"
      << "if(!rw07_finite(sum)||sum<0.0||sum>50.0){"
      << extent.SetByOffset("20u", "100u") << "return;}\n"
      << "let divided=sum/scale;if(!rw07_finite(divided)||divided<0.0||divided>100.0){"
      << extent.SetByOffset("20u", "100u") << "return;}"
      // Original Clip has min1 and absent max. Check, never clamp, its bound.
      << "let clipped=max(rw07_round_even(divided),1.0);"
      << "if(!rw07_finite(clipped)||clipped<1.0||clipped>100.0||clipped!=floor(clipped)){"
      << extent.SetByOffset("20u", "100u") << "return;}"
      << "let duration=u32(clipped);if(prefix>uniforms.capacity||duration>uniforms.capacity-prefix){"
      << extent.SetByOffset("20u", "102u") << "return;}prefix+=duration;"
      << durations.SetByOffset("2u*t", "duration")
      << prefixes.SetByOffset("2u*t", "prefix") << "}\n"
      << total.SetByOffset("0u", "prefix")
      << extent.SetByOffset("2u", "prefix") << extent.SetByOffset("18u", "prefix")
      << extent.SetByOffset("21u", "63u");
    return Status::OK();
  }
};

class DurationExtent final : public WebGpuKernel {
 public:
  explicit DurationExtent(const OpKernelInfo& i) : WebGpuKernel{i}, capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity", 0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* tokens = c.Input<Tensor>(0); const auto* logits = c.Input<Tensor>(1); const auto* speed = c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(tokens && tokens->IsDataType<int64_t>() && tokens->Shape().NumDimensions() == 2 && tokens->Shape()[0] == 1 &&
                      logits && logits->IsDataType<float>() && logits->Shape().NumDimensions() == 3 && logits->Shape()[0] == 1 && logits->Shape()[2] == 50 &&
                      speed && speed->IsDataType<float>() && speed->Shape() == TensorShape{1}, "RW07 duration exact descriptor contract");
    const int64_t t = tokens->Shape()[1];
    ORT_RETURN_IF_NOT(t >= 0 && t <= 512 && logits->Shape()[1] >= t && logits->Shape()[1] <= 1024 && capacity_ > 0 && capacity_ <= 102400,
                      "RW07 duration physical descriptor domain");
    const uint64_t epoch = c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(epoch != 0, "RW07 private owner interface unavailable");
    ORT_RETURN_IF_NOT(Device(tokens) && Device(logits) && Device(speed), "RW07 duration input residency");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 6 && limits.maxBufferSize >= rw07::ExtentWords * 4 &&
                      limits.maxStorageBufferBindingSize >= rw07::ExtentWords * 4,
                      "RW07 duration actual storage limits");
    for (const auto* input : {logits, speed})
      ORT_RETURN_IF_NOT(input->SizeInBytes() <= limits.maxStorageBufferBindingSize && input->SizeInBytes() <= limits.maxBufferSize,
                        "RW07 duration actual input buffer limit");
    ORT_RETURN_IF_NOT(uint64_t(t) * 8 <= limits.maxStorageBufferBindingSize && uint64_t(t) * 8 <= limits.maxBufferSize,
                      "RW07 duration actual output buffer limit");
    auto* durations = c.Output(0, TensorShape{1,t}); auto* prefix = c.Output(1, TensorShape{t});
    auto* total = c.Output(2, TensorShape{}); auto* extent = c.Output(3, TensorShape{rw07::ExtentWords});
    ORT_RETURN_IF_NOT(durations->IsDataType<int64_t>() && prefix->IsDataType<int64_t>() && total->IsDataType<int64_t>() && extent->IsDataType<uint32_t>() &&
                      Device(durations) && Device(prefix) && Device(total) && Device(extent), "RW07 duration output role");
    // Empty original outputs have no binding buffer. Independently owned dummy
    // resources let the error shader run; no empty success/output is published.
    Tensor empty_duration, empty_prefix, empty_logits;
    if (t == 0) {
      empty_duration = c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{2});
      empty_prefix = c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{2});
    }
    if (logits->Shape().Size() == 0)
      empty_logits = c.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape{1});
    LOGS(c.Logger(), INFO) << "M0_RW07_DURATION_BEGIN epoch=" << epoch << " initial_token_descriptor=" << t << " physical_capacity=" << capacity_
        << " adapter=" << std::string_view(c.AdapterInfo().device) << " backend=" << int(c.AdapterInfo().backendType) << " type=" << int(c.AdapterInfo().adapterType)
        << " storage_buffers_limit=" << limits.maxStorageBuffersPerShaderStage << " binding_bytes_limit=" << limits.maxStorageBufferBindingSize;
    Trace(c, "tokens-initial", tokens, epoch); Trace(c, "logits-upload", logits, epoch); Trace(c, "speed-upload", speed, epoch);
    Trace(c, "duration-device-only", durations, epoch); Trace(c, "prefix-device-only", prefix, epoch); Trace(c, "total-device-only", total, epoch); Trace(c, "extent-device-only", extent, epoch);
    DurationProgram p;
    p.AddInputs({{logits->Shape().Size() ? logits : &empty_logits, ProgramTensorMetadataDependency::None}, {speed, ProgramTensorMetadataDependency::None}})
        .AddOutputs({Words(t ? durations : &empty_duration, uint32_t(t ? t*2 : 2)), Words(t ? prefix : &empty_prefix, uint32_t(t ? t*2 : 2)), Words(total,2), Words(extent,rw07::ExtentWords)})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(t), uint32_t(capacity_), uint32_t(epoch), uint32_t(epoch >> 32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(p));
    return Status::OK();
  }
 private: int64_t capacity_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuDurationExtentKernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out)->Status {
    out = std::make_unique<DurationExtent>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("DurationExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(), fn};
}
} // namespace onnxruntime::webgpu
