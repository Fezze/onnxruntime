// rw07-extent-stride-linear-v1: original Resize1514 rank3 trilinear, half_pixel.
// Original FP32 scale bits 0x3b5a740e; no rational 1/300 substitution.
#include <algorithm>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"
#include "m0_duration_exact_fp32.h"

namespace onnxruntime::webgpu {
namespace {
ProgramInput Words(const Tensor* t) {
  ProgramInput p(t, ProgramTensorMetadataDependency::None, TensorShape{rw07::ExtentWords}, 1);
  p.var_type = ProgramVariableDataType::Uint32; return p;
}
ProgramOutput Words(Tensor* t) {
  ProgramOutput p(t, ProgramTensorMetadataDependency::None, TensorShape{rw07::ExtentWords}, 1);
  p.var_type = ProgramVariableDataType::Uint32; return p;
}
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
class ResizeGuard final : public Program<ResizeGuard> {
 public:
  ResizeGuard() : Program{"RW07V1ResizeLinearExtentGuard"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << ::rw07::DurationExactFp32;
    auto& f=s.MainFunctionBody();
    f << "for(var i=0u;i<24u;i++){" << o.SetByOffset("i","0u") << "}"
      << o.SetByOffset("0u","1u") << o.SetByOffset("1u","3u")
      << o.SetByOffset("22u","uniforms.epoch_low") << o.SetByOffset("23u","uniforms.epoch_high")
      << "if(" << e.GetByOffset("0u") << "!=1u||" << e.GetByOffset("1u") << "!=3u||"
      << e.GetByOffset("22u") << "!=uniforms.epoch_low||" << e.GetByOffset("23u") << "!=uniforms.epoch_high){"
      << o.SetByOffset("20u","103u") << "return;}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){" << o.SetByOffset("20u","status") << "return;}"
      << "let len=" << e.GetByOffset("6u") << ";"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("4u") << "!=uniforms.channels||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||len==0u||(" << e.GetByOffset("21u") << "&63u)!=63u||"
      << e.GetByOffset("18u") << "!=uniforms.channels*len||" << e.GetByOffset("19u") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}"
      << "if(len>uniforms.input_capacity){" << o.SetByOffset("20u","102u") << "return;}"
      << "let exact_len=bitcast<f32>(rw07_pack(vec2<u32>(len,0u),60));let outlen=u32(floor(rw07_mul(exact_len,bitcast<f32>(0x3b5a740eu))));if(outlen>uniforms.output_capacity){"
      << o.SetByOffset("20u","102u") << "return;}"
      << o.SetByOffset("2u","1u") << o.SetByOffset("4u","uniforms.channels")
      << o.SetByOffset("6u","outlen") << o.SetByOffset("18u","uniforms.channels*outlen")
      << o.SetByOffset("21u","63u");
    return Status::OK();
  }
};
class ResizeData final : public Program<ResizeData> {
 public:
  ResizeData() : Program{"RW07V1ResizeLinearExtentData"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("x",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("y",ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << ::rw07::DurationExactFp32;
    auto& f=s.MainFunctionBody();
    f << "if(global_idx>=uniforms.channels*uniforms.output_capacity){return;}"
      << y.SetByOffset("global_idx","0.0")
      << "if(" << o.GetByOffset("20u") << "!=0u||" << o.GetByOffset("22u") << "!=uniforms.epoch_low||"
      << o.GetByOffset("23u") << "!=uniforms.epoch_high){return;}"
      << "let oc=global_idx/uniforms.output_capacity;let ot=global_idx%uniforms.output_capacity;"
      << "if(ot>=" << o.GetByOffset("6u") << "){return;}"
      // CPU rank3 path is trilinear: retain zero-weight neighboring-channel
      // terms and duplicated depth terms, including their nonfinite semantics.
      << "let len=" << e.GetByOffset("6u") << ";"
      << "let coord=rw07_add(rw07_div(rw07_add(f32(ot),0.5),bitcast<f32>(0x3b5a740eu)),-0.5);"
      << "let bound=bitcast<f32>(rw07_pack(vec2<u32>(len-1u,0u),60));"
      << "let pos=max(0.0,min(coord,bound));let lo=min(u32(pos),len-1u);let hi=min(lo+1u,len-1u);"
      << "var dx1=abs(pos-f32(lo));var dx2=abs(pos-f32(hi));if(lo==hi){dx1=0.5;dx2=0.5;}"
      << "let next_c=min(oc+1u,8u);let dy1=select(0.0,0.5,oc==next_c);let dy2=select(1.0,0.5,oc==next_c);"
      << "let a=" << x.GetByOffset("oc*uniforms.input_capacity+lo") << ";"
      << "let b=" << x.GetByOffset("oc*uniforms.input_capacity+hi") << ";"
      << "let c=" << x.GetByOffset("next_c*uniforms.input_capacity+lo") << ";"
      << "let d=" << x.GetByOffset("next_c*uniforms.input_capacity+hi") << ";"
      << "var value=dx2*dy2*0.5*a;value=value+dx1*dy2*0.5*b;"
      << "value=value+dx2*dy1*0.5*c;value=value+dx1*dy1*0.5*d;"
      << "value=value+dx2*dy2*0.5*a;value=value+dx1*dy2*0.5*b;"
      << "value=value+dx2*dy1*0.5*c;value=value+dx1*dy1*0.5*d;"
      << y.SetByOffset("global_idx","value");
    return Status::OK();
  }
};
class ResizeLinearExtent final : public WebGpuKernel {
 public:
  explicit ResizeLinearExtent(const OpKernelInfo& i) : WebGpuKernel{i},
    capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x=c.Input<Tensor>(0);const auto* e=c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(x && x->IsDataType<float>() && x->Shape().NumDimensions()==3 && x->Shape()[0]==1 && x->Shape()[1]==9,
                      "RW07 exact Resize1514 input descriptor");
    const int64_t channels=9,input_capacity=x->Shape()[2];
    ORT_RETURN_IF_NOT(input_capacity>0 && input_capacity<=30720000 && capacity_>0 && capacity_<=102400 &&
                      e && e->IsDataType<uint32_t>() && e->Shape()==TensorShape({rw07::ExtentWords}),
                      "RW07 exact Resize1514 physical descriptors");
    const uint64_t epoch=c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(epoch && Gpu(x) && Gpu(e),"RW07 resize owner/input residency");
    const auto& limits=c.DeviceLimits();
    const uint64_t count=uint64_t(channels)*uint64_t(capacity_),groups=(count+63)/64,limit=limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=4 && count*4<=limits.maxBufferSize &&
                      count*4<=limits.maxStorageBufferBindingSize && limits.maxStorageBufferBindingSize>=96 &&
                      limits.maxBufferSize>=96 && count<=UINT32_MAX && limit>0 && groups<=limit*limit &&
                      limits.maxComputeWorkgroupSizeX>=64 && limits.maxComputeInvocationsPerWorkgroup>=64,
                      "RW07 resize actual resource/dispatch limits");
    for(const auto* input:{x,e})
      ORT_RETURN_IF_NOT(input->SizeInBytes()<=limits.maxBufferSize && input->SizeInBytes()<=limits.maxStorageBufferBindingSize,
                        "RW07 resize input binding limit");
    auto* y=c.Output(0,TensorShape({1,channels,capacity_}));auto* out=c.Output(1,TensorShape({rw07::ExtentWords}));
    ORT_RETURN_IF_NOT(Gpu(y)&&Gpu(out)&&y->IsDataType<float>()&&out->IsDataType<uint32_t>(),"RW07 resize output residency");
    ResizeGuard guard;
    guard.AddInput(Words(e)).AddOutput(Words(out)).SetWorkgroupSize(1).SetDispatchGroupSize(1)
      .AddUniformVariables({uint32_t(channels),uint32_t(input_capacity),uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    // Capacity-only 2D dispatch; ShaderHelper flattens actual workgroup IDs.
    // Rounded coverage < 2*(count+63), below UINT32_MAX in this domain.
    const uint64_t gx=std::min(groups,limit),gy=(groups+gx-1)/gx;
    ResizeData data;
    data.AddInputs({{x,ProgramTensorMetadataDependency::None},Words(e),Words(static_cast<const Tensor*>(out))})
      .AddOutput({y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
      .AddUniformVariables({uint32_t(channels),uint32_t(input_capacity),uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(data));
    for(const auto& entry:{std::pair{"resize-input",x},std::pair{"source-extent-device-only",e},
                          std::pair{"resize-output",static_cast<const Tensor*>(y)},std::pair{"extent-device-only",static_cast<const Tensor*>(out)}})
      LOGS(c.Logger(),INFO)<<"M0_RW07_RESOURCE role="<<entry.first<<" epoch="<<epoch<<" handle="<<entry.second->DataRaw()
        <<" device="<<int(entry.second->Location().device.Type())<<" physical_bytes="<<entry.second->SizeInBytes();
    return Status::OK();
  }
 private:int64_t capacity_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuResizeLinearExtentKernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status {out=std::make_unique<ResizeLinearExtent>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("ResizeLinearExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
} // namespace onnxruntime::webgpu
