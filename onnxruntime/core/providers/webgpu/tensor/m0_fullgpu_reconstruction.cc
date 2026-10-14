// rw07-extent-reconstruction-v1: exact original nodes2427..2463, batch1.
// Positive logical frames, immutable inverse basis and original window_sum.
// The col buffer preserves original Gemm then kernel-order Col2im arithmetic.
#include <algorithm>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_extent_layout.h"

namespace onnxruntime::webgpu {
namespace {
ProgramInput Words(const Tensor* t) {
  ProgramInput p(t,ProgramTensorMetadataDependency::None,TensorShape{rw07::ExtentWords},1);
  p.var_type=ProgramVariableDataType::Uint32;return p;
}
ProgramOutput Words(Tensor* t) {
  ProgramOutput p(t,ProgramTensorMetadataDependency::None,TensorShape{rw07::ExtentWords},1);
  p.var_type=ProgramVariableDataType::Uint32;return p;
}
bool Gpu(const Tensor* t) {return t&&t->Location().device.Type()==OrtDevice::GPU;}
class ReconstructionGuard final : public Program<ReconstructionGuard> {
 public:
  ReconstructionGuard():Program{"RW07V1ReconstructionExtentGuard"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("window",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation()<<"const original_window=array<u32,20>(0u,974978362u,1008035403u,1026424773u,1039430992u,1048576000u,1054561486u,1059150100u,1062302038u,1064542128u,1065353216u,1064542128u,1062302038u,1059150100u,1054561486u,1048576000u,1039430992u,1026424773u,1008035403u,974978362u);";
    auto& f=s.MainFunctionBody();
    f<<"for(var i=0u;i<24u;i++){"<<o.SetByOffset("i","0u")<<"}"
     <<o.SetByOffset("0u","1u")<<o.SetByOffset("1u","1u")
     <<o.SetByOffset("22u","uniforms.epoch_low")<<o.SetByOffset("23u","uniforms.epoch_high")
     <<"if("<<e.GetByOffset("0u")<<"!=1u||"<<e.GetByOffset("1u")<<"!=3u||"
     <<e.GetByOffset("22u")<<"!=uniforms.epoch_low||"<<e.GetByOffset("23u")<<"!=uniforms.epoch_high){"
     <<o.SetByOffset("20u","103u")<<"return;}"
     <<"let status="<<e.GetByOffset("20u")<<";if(status!=0u){"<<o.SetByOffset("20u","status")<<"return;}"
     <<"let len="<<e.GetByOffset("6u")<<";if("<<e.GetByOffset("2u")<<"!=1u||"
     <<e.GetByOffset("3u")<<"!=0u||"<<e.GetByOffset("4u")<<"!=22u||"<<e.GetByOffset("5u")<<"!=0u||"
     <<e.GetByOffset("7u")<<"!=0u||len==0u||len>6144001u||("<<e.GetByOffset("21u")<<"&63u)!=63u||"
     <<e.GetByOffset("18u")<<"!=22u*len||"<<e.GetByOffset("19u")<<"!=0u){"
     <<o.SetByOffset("20u","103u")<<"return;}"
     // Runtime proof of immutable window and its exact NonZero selection.
     <<"for(var j=0u;j<20u;j++){if(bitcast<u32>("<<w.GetByOffset("j")<<")!=original_window[j]){"
     <<o.SetByOffset("20u","103u")<<"return;}let selected="<<w.GetByOffset("j")<<">bitcast<f32>(0x00800000u);"
     <<"if(selected!=(j!=0u)){"<<o.SetByOffset("20u","103u")<<"return;}}"
     <<"let audio=5u*(len-1u);if(len>uniforms.input_capacity||audio>uniforms.output_capacity){"
     <<o.SetByOffset("20u","102u")<<"return;}"
     <<o.SetByOffset("2u","audio")<<o.SetByOffset("18u","audio")<<o.SetByOffset("21u","63u");
    return Status::OK();
  }
};
class ReconstructionCol final : public Program<ReconstructionCol> {
 public:
  ReconstructionCol():Program{"RW07V1ReconstructionInverseGemm"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("x",ShaderUsage::UseValueTypeAlias);
    const auto& w=s.AddInput("basis",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& col=s.AddOutput("col",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    f<<"if(global_idx>=20u*uniforms.input_capacity){return;}"<<col.SetByOffset("global_idx","0.0")
     <<"if("<<e.GetByOffset("20u")<<"!=0u||"<<e.GetByOffset("22u")<<"!=uniforms.epoch_low||"
     <<e.GetByOffset("23u")<<"!=uniforms.epoch_high){return;}"
     <<"let frame=global_idx%uniforms.input_capacity;let k=global_idx/uniforms.input_capacity;"
     <<"let len="<<e.GetByOffset("2u")<<"/5u+1u;if(frame>=len){return;}var value=0.0;"
     <<"for(var c=0u;c<22u;c++){let product="<<x.GetByOffset("c*uniforms.input_capacity+frame")
     <<"*"<<w.GetByOffset("c*20u+k")<<";value=value+product;}"
     <<col.SetByOffset("global_idx","value");
    return Status::OK();
  }
};
class ReconstructionAudio final : public Program<ReconstructionAudio> {
 public:
  ReconstructionAudio():Program{"RW07V1ReconstructionOverlapNormalizeCrop"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"input_capacity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"epoch_low",ProgramUniformVariableDataType::Uint32},
    {"epoch_high",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& col=s.AddInput("col",ShaderUsage::UseValueTypeAlias);
    const auto& window=s.AddInput("window",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& audio=s.AddOutput("audio",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    f<<"if(global_idx>=uniforms.output_capacity){return;}"<<audio.SetByOffset("global_idx","0.0")
     <<"if("<<e.GetByOffset("20u")<<"!=0u||"<<e.GetByOffset("22u")<<"!=uniforms.epoch_low||"
     <<e.GetByOffset("23u")<<"!=uniforms.epoch_high||global_idx>="<<e.GetByOffset("2u")<<"){return;}"
     <<"let len="<<e.GetByOffset("2u")<<"/5u+1u;let sample=global_idx+10u;var value=0.0;"
     // CPU Col2imNd iterates kernel index before frames. Retain that order.
     <<"for(var k=0u;k<20u;k++){if(sample>=k){let delta=sample-k;if(delta%5u==0u){let frame=delta/5u;if(frame<len){value=value+"
     <<col.GetByOffset("k*uniforms.input_capacity+frame")<<";}}}}"
     // Original ScatterND replaces only indices1..19. Crop starts at10.
     <<"if(sample<20u){value=value/"<<window.GetByOffset("sample")<<";}value=value*4.0;"
     <<audio.SetByOffset("global_idx","value");
    return Status::OK();
  }
};
class ReconstructionExtent final : public WebGpuKernel {
 public:
  explicit ReconstructionExtent(const OpKernelInfo& i):WebGpuKernel{i},capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* x=c.Input<Tensor>(0);const auto* w=c.Input<Tensor>(1);
    const auto* window=c.Input<Tensor>(2);const auto* e=c.Input<Tensor>(3);
    ORT_RETURN_IF_NOT(x&&x->IsDataType<float>()&&x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&x->Shape()[1]==22,
      "RW07 original reconstruction input descriptor");
    const int64_t pc=x->Shape()[2];
    ORT_RETURN_IF_NOT(pc>0&&pc<=6144001&&capacity_>0&&capacity_<=30720000&&w&&w->IsDataType<float>()&&
      w->Shape()==TensorShape({22,1,20})&&window&&window->IsDataType<float>()&&window->Shape()==TensorShape({20})&&
      e&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape({rw07::ExtentWords}),"RW07 original reconstruction physical descriptors");
    const uint64_t epoch=c.KernelContext().M0FullGpuExtentOwnerEpochV1();
    ORT_RETURN_IF_NOT(epoch&&Gpu(x)&&Gpu(w)&&Gpu(window)&&Gpu(e),"RW07 reconstruction input ownership/residency");
    const auto& limits=c.DeviceLimits();const uint64_t limit=limits.maxComputeWorkgroupsPerDimension;
    const uint64_t col_count=20*uint64_t(pc),audio_count=uint64_t(capacity_);
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=4&&limits.maxComputeWorkgroupSizeX>=64&&
      limits.maxComputeInvocationsPerWorkgroup>=64&&limit>0&&limits.maxBufferSize>=96&&limits.maxStorageBufferBindingSize>=96,
      "RW07 reconstruction actual resource/dispatch limits");
    for(uint64_t count:{col_count,audio_count})ORT_RETURN_IF_NOT(count<=UINT32_MAX&&count*4<=limits.maxBufferSize&&
      count*4<=limits.maxStorageBufferBindingSize&&(count+63)/64<=limit*limit,"RW07 reconstruction physical output/dispatch capacity");
    for(const auto* input:{x,w,window,e})ORT_RETURN_IF_NOT(input->SizeInBytes()<=limits.maxBufferSize&&
      input->SizeInBytes()<=limits.maxStorageBufferBindingSize,"RW07 reconstruction input binding capacity");
    auto* audio=c.Output(0,TensorShape({capacity_}));auto* out=c.Output(1,TensorShape({rw07::ExtentWords}));
    auto col=c.CreateGPUTensor(DataTypeImpl::GetType<float>(),TensorShape({20,pc}));
    ORT_RETURN_IF_NOT(Gpu(audio)&&Gpu(out)&&Gpu(&col),"RW07 reconstruction output/scratch residency");
    ReconstructionGuard guard;guard.AddInputs({Words(e),{window,ProgramTensorMetadataDependency::None}}).AddOutput(Words(out))
      .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({uint32_t(pc),uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const uint64_t cg=(col_count+63)/64,cx=std::min(cg,limit),cy=(cg+cx-1)/cx;
    ReconstructionCol inverse;inverse.AddInputs({{x,ProgramTensorMetadataDependency::None},{w,ProgramTensorMetadataDependency::None},Words(static_cast<const Tensor*>(out))})
      .AddOutput({&col,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(cx),uint32_t(cy))
      .AddUniformVariables({uint32_t(pc),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(inverse));
    const uint64_t ag=(audio_count+63)/64,ax=std::min(ag,limit),ay=(ag+ax-1)/ax;
    ReconstructionAudio finish;finish.AddInputs({{&col,ProgramTensorMetadataDependency::None},{window,ProgramTensorMetadataDependency::None},Words(static_cast<const Tensor*>(out))})
      .AddOutput({audio,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(ax),uint32_t(ay))
      .AddUniformVariables({uint32_t(pc),uint32_t(capacity_),uint32_t(epoch),uint32_t(epoch>>32)});
    ORT_RETURN_IF_ERROR(c.RunProgram(finish));
    for(const auto& entry:{std::pair{"reconstruction-input",x},std::pair{"inverse-basis",w},std::pair{"original-window",window},std::pair{"source-extent-device-only",e},
      std::pair{"reconstruction-col-scratch",static_cast<const Tensor*>(&col)},std::pair{"reconstruction-audio",static_cast<const Tensor*>(audio)},std::pair{"extent-device-only",static_cast<const Tensor*>(out)}})
      LOGS(c.Logger(),INFO)<<"M0_RW07_RESOURCE role="<<entry.first<<" epoch="<<epoch<<" handle="<<entry.second->DataRaw()<<" device="<<int(entry.second->Location().device.Type())<<" physical_bytes="<<entry.second->SizeInBytes();
    return Status::OK();
  }
 private:int64_t capacity_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuReconstructionExtentKernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status {out=std::make_unique<ReconstructionExtent>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("ReconstructionExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
} // namespace onnxruntime::webgpu
