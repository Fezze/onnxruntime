// RW02 isolated prototype v1. Not a general NonZero or dynamic shape implementation.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include <atomic>
namespace onnxruntime::webgpu {
class M0Compact final : public Program<M0Compact> {
 public:
  explicit M0Compact(bool empty) : Program{"M0CompactV1"}, empty_(empty) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"m", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const ShaderVariableHelper* d = empty_ ? nullptr : &s.AddInput("denom", ShaderUsage::UseValueTypeAlias);
    const auto& count=s.AddOutput("count", ShaderUsage::UseValueTypeAlias);
    const auto& idx=s.AddOutput("idx", ShaderUsage::UseValueTypeAlias);
    auto& b=s.MainFunctionBody();
    b << "var k: i32 = 0;\nfor (var j=0u; j<20u; j++) {\n" << idx.SetByOffset("j", "-1") << "}\n";
    if (!empty_) b << "for (var j=0u; j<uniforms.m; j++) { if (" << d->GetByOffset("j") << ">bitcast<f32>(0x00800000u)) {\n" << idx.SetByOffset("u32(k)", "i32(j)") << "k++; }}\n";
    b << count.SetByOffset("0u","k");return Status::OK();
  }
 private: bool empty_;
};
class M0Copy final : public Program<M0Copy> {
 public: M0Copy():Program{"M0CopyV1"}{}
 WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"n",ProgramUniformVariableDataType::Uint32});
 Status GenerateShaderCode(ShaderHelper& s) const override {
 const auto& x=s.AddInput("base",ShaderUsage::UseValueTypeAlias);const auto& y=s.AddOutput("dense",ShaderUsage::UseValueTypeAlias);
 s.MainFunctionBody()<<s.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.n")<<y.SetByOffset("global_idx",x.GetByOffset("global_idx"));return Status::OK();}
};
class M0Normalize final : public Program<M0Normalize> {
 public:M0Normalize():Program{"M0NormalizeV1"}{}
 Status GenerateShaderCode(ShaderHelper& s) const override {
 const auto& x=s.AddInput("base",ShaderUsage::UseValueTypeAlias);const auto& d=s.AddInput("denom",ShaderUsage::UseValueTypeAlias);const auto& k=s.AddInput("count",ShaderUsage::UseValueTypeAlias);const auto& i=s.AddInput("idx",ShaderUsage::UseValueTypeAlias);const auto& y=s.AddOutput("dense",ShaderUsage::UseValueTypeAlias);
 s.MainFunctionBody()<<"if (global_idx >= u32("<<k.GetByOffset("0u")<<")) { return; }\nlet j=u32("<<i.GetByOffset("global_idx")<<");\n"<<y.SetByOffset("j",x.GetByOffset("j")+" / "+d.GetByOffset("j"));return Status::OK();}
};
class M0IndexRegion final : public WebGpuKernel {
 public:explicit M0IndexRegion(const OpKernelInfo& i):WebGpuKernel{i}{}
 Status ComputeInternal(ComputeContext& c) const override {
 const auto* x=c.Input<Tensor>(0);const auto* d=c.Input<Tensor>(1);
 ORT_RETURN_IF_NOT(x&&d&&x->IsDataType<float>()&&d->IsDataType<float>(),"RW02 requires FLOAT inputs");
 const auto& xs=x->Shape();const auto& ds=d->Shape();
 ORT_RETURN_IF_NOT(xs.NumDimensions()==2&&xs[0]==1&&ds.NumDimensions()==1,"RW02 requires [1,N], [M]");
 ORT_RETURN_IF_NOT(xs[1]>=0&&xs[1]<=1048576&&ds[0]>=0&&ds[0]<=20&&ds[0]<=xs[1],"RW02 prototype bounds violated");
 const auto n=static_cast<uint32_t>(xs[1]);const auto m=static_cast<uint32_t>(ds[0]);
 auto* y=c.Output(0,xs);auto* k=c.Output(1,TensorShape{1});auto* idx=c.Output(2,TensorShape{20});
 static std::atomic<uint64_t> next{0};auto id=++next;
 ORT_RETURN_IF_NOT(x->Location().device.Type()==OrtDevice::GPU && d->Location().device.Type()==OrtDevice::GPU && y->Location().device.Type()==OrtDevice::GPU && k->Location().device.Type()==OrtDevice::GPU && idx->Location().device.Type()==OrtDevice::GPU,"RW02 tensor resident-device contract violated");
 const auto logTensor=[&](const char* name,const Tensor* t){LOGS(c.Logger(),INFO)<<"M0_RW02_TENSOR run="<<id<<" name="<<name<<" logical_bytes="<<t->SizeInBytes()<<" device="<<int(t->Location().device.Type())<<" handle="<<t->DataRaw()<<" actual_bytes="<<(t->SizeInBytes()?wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw()))):0);};
 LOGS(c.Logger(),INFO)<<"M0_RW02_BEGIN run="<<id<<" n="<<n<<" m="<<m<<" adapter="<<std::string_view(c.AdapterInfo().device)<<" backend="<<int(c.AdapterInfo().backendType)<<" type="<<int(c.AdapterInfo().adapterType);
 logTensor("base",x);logTensor("denom",d);logTensor("dense",y);logTensor("count",k);logTensor("indices",idx);
 M0Compact compact(m==0);if(m)compact.AddInput({d,ProgramTensorMetadataDependency::TypeAndRank});compact.AddOutputs({{k,ProgramTensorMetadataDependency::TypeAndRank},{idx,ProgramTensorMetadataDependency::TypeAndRank}}).SetWorkgroupSize(1).SetDispatchGroupSize(1).CacheHint(m==0).AddUniformVariable({m});ORT_RETURN_IF_ERROR(c.RunProgram(compact));
 if(n){M0Copy copy;copy.AddInput({x,ProgramTensorMetadataDependency::TypeAndRank,ProgramInput::Flatten}).AddOutput({y,ProgramTensorMetadataDependency::TypeAndRank,ProgramOutput::Flatten}).SetDispatchGroupSize((n+63)/64).SetWorkgroupSize(64).AddUniformVariable({n});ORT_RETURN_IF_ERROR(c.RunProgram(copy));}
 if(m){M0Normalize norm;norm.AddInputs({{x,ProgramTensorMetadataDependency::TypeAndRank,ProgramInput::Flatten},{d,ProgramTensorMetadataDependency::TypeAndRank},{k,ProgramTensorMetadataDependency::TypeAndRank},{idx,ProgramTensorMetadataDependency::TypeAndRank}}).AddOutput({y,ProgramTensorMetadataDependency::TypeAndRank,ProgramOutput::Flatten}).SetWorkgroupSize(32).SetDispatchGroupSize(1);ORT_RETURN_IF_ERROR(c.RunProgram(norm));}
 LOGS(c.Logger(),INFO)<<"M0_RW02_ENCODED run="<<id;return Status::OK();
 }
};
KernelCreateInfo CreateM0IndexRegionKernelInfo(){
 KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& info,std::unique_ptr<OpKernel>& out)->Status{out=std::make_unique<M0IndexRegion>(info);return Status::OK();};
 return {KernelDefBuilder().SetName("IndexRegion").SetDomain("org.localtts.m0").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
}
