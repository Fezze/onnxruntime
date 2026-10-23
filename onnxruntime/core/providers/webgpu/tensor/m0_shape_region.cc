// RW04 isolated v1. Fixed seven-node recipe; no generic shape/control changes.
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#if defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/ep/sync_stream.h"
#endif
#include <atomic>

namespace onnxruntime::webgpu {
namespace {
#if defined(ORT_USE_EP_API_ADAPTERS)
std::atomic<void*> previous_stream{nullptr}; // Diagnostic only; adapter keeps all involved Sessions alive.
#endif
ProgramInput WordsIn(const Tensor* t, uint32_t words) {
  ProgramInput p(t, ProgramTensorMetadataDependency::None, TensorShape{words}, 1);
  p.var_type = ProgramVariableDataType::Uint32;
  return p;
}
ProgramOutput WordsOut(Tensor* t, uint32_t words) {
  ProgramOutput p(t, ProgramTensorMetadataDependency::None, TensorShape{words}, 1);
  p.var_type = ProgramVariableDataType::Uint32;
  return p;
}
void Trace(ComputeContext& c, const char* owner, const char* name, const Tensor* t) {
  LOGS(c.Logger(), INFO) << "M0_RW04_TENSOR owner=" << owner << " name=" << name
    << " logical_bytes=" << t->SizeInBytes() << " device=" << int(t->Location().device.Type())
    << " handle=" << t->DataRaw() << " actual_bytes="
    << (t->SizeInBytes() ? wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(t->DataRaw()))) : 0);
}
bool Device(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }

class ShapePayload final : public Program<ShapePayload> {
 public:
  explicit ShapePayload(bool audit) : Program{"RW04V1ShapePayload"}, audit_(audit) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"rank", ProgramUniformVariableDataType::Uint32}, {"b", ProgramUniformVariableDataType::Uint32},
    {"t", ProgramUniformVariableDataType::Uint32}, {"h", ProgramUniformVariableDataType::Uint32},
    {"fault", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const ShaderVariableHelper* expected = audit_ ? &s.AddInput("expected", ShaderUsage::UseValueTypeAlias) : nullptr;
    const auto& scratch=s.AddOutput("scratch", ShaderUsage::UseValueTypeAlias);
    const auto& dims=s.AddOutput("dims", ShaderUsage::UseValueTypeAlias);
    const auto& status=s.AddOutput("status", ShaderUsage::UseValueTypeAlias);
    const auto& verdict=s.AddOutput("verdict", ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    // Paired u32 INT64 payloads: Shape [3], two scalars, two [1] vectors, Concat [4].
    f << "var v: array<u32,22>; for(var i=0u;i<22u;i++){v[i]=0u;}\n"
         "if(uniforms.rank>0u){v[0]=uniforms.b;} if(uniforms.rank>1u){v[2]=uniforms.t;}"
         "if(uniforms.rank>2u){v[4]=uniforms.h;}\n"
         "var error=0u; if(uniforms.rank==0u){error=1u;} else if(uniforms.rank==1u){error=2u;}\n"
         "if(error==0u){v[6]=v[0];v[8]=v[2];v[10]=v[6];v[12]=v[8];"
         "v[14]=v[10];v[16]=v[12];v[18]=12u;v[20]=64u;"
         "if(v[0]*v[2]*v[4] != v[14]*v[16]*v[18]*v[20]){error=3u;}}\n"
         "var ok=0u;\n";
    if(audit_) f << "if(error==0u){ok=1u;for(var i=0u;i<22u;i++){if(v[i]!=" << expected->GetByOffset("i") << "){ok=0u;}}}\n";
    f << "for(var i=0u;i<22u;i++){" << scratch.SetByOffset("i", "v[i]") << "}\n"
      << "for(var i=0u;i<8u;i++){" << dims.SetByOffset("i", "v[14u+i]") << "}\n"
      // Explicit audit-only corruption after original shape computation. Hash each model variant.
      << "if(uniforms.fault==1u){" << dims.SetByOffset("2u", "v[16]+1u") << "}\n"
      << "if(uniforms.fault==2u){" << dims.SetByOffset("1u", "1u") << "}\n"
      << "if(uniforms.fault==3u){error=99u;}\n"
      << status.SetByOffset("0u","error") << verdict.SetByOffset("0u","ok");
    return Status::OK();
  }
 private: bool audit_;
};
class ValidateShape final : public Program<ValidateShape> {
 public: ValidateShape():Program{"RW04V1ValidateShape"}{}
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& scratch=s.AddInput("scratch",ShaderUsage::UseValueTypeAlias);
    const auto& dims=s.AddInput("dims",ShaderUsage::UseValueTypeAlias);
    const auto& status=s.AddOutput("status",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    // The status buffer is read/write, queue ordered after ShapePayload.
    f << "if("<<status.GetByOffset("0u")<<"!=0u){return;}\nfor(var i=0u;i<8u;i++){if("
      << dims.GetByOffset("i")<<"!="<<scratch.GetByOffset("14u+i")<<"){"
      << status.SetByOffset("0u","4u")<<"return;}}\n";
    return Status::OK();
  }
};
class ShapeData final : public Program<ShapeData> {
 public: explicit ShapeData(bool transpose):Program{transpose?"RW04V1Consumer458":"RW04V1Reshape454"}, transpose_(transpose){}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"capacity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("x",ShaderUsage::UseValueTypeAlias);
    const auto& dims=s.AddInput("dims",ShaderUsage::UseValueTypeAlias);
    const auto& status=s.AddInput("status",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("y",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    f<<"if("<<status.GetByOffset("0u")<<"!=0u){return;}\nlet b="<<dims.GetByOffset("0u")
     <<";let t="<<dims.GetByOffset("2u")<<";let heads="<<dims.GetByOffset("4u")
     <<";let width="<<dims.GetByOffset("6u")<<";let total=b*t*heads*width;\n"
       "if(global_idx>=total || global_idx>=uniforms.capacity){return;}\n";
    if(transpose_) f<<"let cc=global_idx%width;let tt=(global_idx/width)%t;"
                      "let hh=(global_idx/(width*t))%heads;let bb=global_idx/(width*t*heads);"
                      "let src=((bb*t+tt)*heads+hh)*width+cc;\n";
    else f<<"let src=global_idx;\n";
    f<<y.SetByOffset("global_idx",x.GetByOffset("src"));
    return Status::OK();
  }
 private: bool transpose_;
};
class ShapeRegion final : public WebGpuKernel {
 public:
  explicit ShapeRegion(const OpKernelInfo& i):WebGpuKernel{i}, audit_(i.GetAttrOrDefault<int64_t>("audit",0)!=0),fault_(i.GetAttrOrDefault<int64_t>("fault",0)){}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* q=c.Input<Tensor>(0); ORT_RETURN_IF_NOT(q && q->IsDataType<float>(),"RW04 admission: FP32 required");
    const auto& d=q->Shape();const auto rank=d.NumDimensions();
    ORT_RETURN_IF_NOT(rank==0||rank==1||rank==3,"RW04 admission: unsupported rank");
    const uint32_t b=rank?static_cast<uint32_t>(d[0]):0;
    const uint32_t t=rank==3?static_cast<uint32_t>(d[1]):0;
    const uint32_t h=rank==3?static_cast<uint32_t>(d[2]):0;
    if(rank==3) ORT_RETURN_IF_NOT(d[0]>=0&&d[0]<=1&&d[1]>=0&&d[1]<=512&&d[2]>=0&&d[2]<=1024,"RW04 admission: bounded dimensions required");
    if(rank==1) ORT_RETURN_IF_NOT(d[0]>=0&&d[0]<=1024,"RW04 admission: diagnostic rank-one bound");
    ORT_RETURN_IF_NOT(fault_>=0&&fault_<=4,"RW04 admission: invalid diagnostic fault");
#if defined(ORT_USE_EP_API_ADAPTERS)
    auto* current_stream=c.KernelContext().GetSyncStream();
    if(fault_==4){
      auto* foreign=previous_stream.load();
      ORT_RETURN_IF_NOT(foreign && foreign!=current_stream,"RW04 foreign-stream test requires distinct live Sessions");
      AllocatorPtr allocator;ORT_RETURN_IF_ERROR(c.KernelContext().GetTempSpaceAllocator(&allocator));
      auto& current_recording=ep::GetWebGpuStreamCommandState(reinterpret_cast<OrtSyncStream*>(current_stream));
      auto& foreign_recording=ep::GetWebGpuStreamCommandState(reinterpret_cast<OrtSyncStream*>(foreign));
      ORT_RETURN_IF_NOT(&current_recording!=&foreign_recording,"RW04 foreign-stream test requires distinct recording owners");
      void* control=allocator->AllocOnStream(16,reinterpret_cast<Stream*>(current_stream));
      ORT_RETURN_IF_NOT(control,"RW04 foreign-stream test positive control allocation failed");
      LOGS(c.Logger(),INFO)<<"M0_RW04_SAME_STREAM_CONTROL succeeded=1 bytes=16 handle="<<control<<" recording="<<&current_recording;
      allocator->Free(control);
      void* block=nullptr;bool rejected=false;
      try{block=allocator->AllocOnStream(16,reinterpret_cast<Stream*>(foreign));rejected=block==nullptr;}catch(const std::exception&){rejected=true;}
      if(block)allocator->Free(block);
      LOGS(c.Logger(),INFO)<<"M0_RW04_FOREIGN_STREAM rejected="<<rejected<<" current="<<current_stream<<" foreign="<<foreign<<" current_recording="<<&current_recording<<" foreign_recording="<<&foreign_recording;
      return ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,rejected?"RW04 diagnostic: foreign-Session allocation rejected":"RW04 diagnostic FAILURE: foreign-Session allocation unexpectedly accepted");
    }
    previous_stream.store(current_stream);
#else
    ORT_RETURN_IF(fault_==4,"RW04 diagnostic: foreign-Session allocation requires plugin stream API");
#endif
    const auto* expected=audit_?c.Input<Tensor>(1):nullptr;
    if(audit_) ORT_RETURN_IF_NOT(expected&&expected->IsDataType<uint32_t>()&&expected->Shape()==TensorShape{22},"RW04 audit: expected UINT32[22]");
    auto* y=c.Output(0,rank==3?TensorShape{b,t,12,64}:TensorShape{0,0,12,64});
    auto* dims=c.Output(1,TensorShape{4});auto* status=c.Output(2,TensorShape{1});auto* verdict=c.Output(3,TensorShape{1});
    auto scratch=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{22});
    ORT_RETURN_IF_NOT(Device(q)&&Device(y)&&Device(dims)&&Device(status)&&Device(verdict)&&Device(&scratch)&&(!audit_||Device(expected)),"RW04 infrastructure: device memory required");
    static std::atomic<uint64_t> next{0};const auto id=++next;
    LOGS(c.Logger(),INFO)<<"M0_RW04_BEGIN run="<<id<<" rank="<<rank<<" b="<<b<<" t="<<t<<" h="<<h<<" audit="<<audit_<<" fault="<<fault_<<" adapter="<<std::string_view(c.AdapterInfo().device)<<" backend="<<int(c.AdapterInfo().backendType)<<" type="<<int(c.AdapterInfo().adapterType);
    Trace(c,"region","q",q);Trace(c,"region","y",y);Trace(c,"region","shape",dims);Trace(c,"region","status",status);Trace(c,"region","verdict",verdict);Trace(c,"region","scratch",&scratch);if(audit_)Trace(c,"region","expected",expected);
    ShapePayload payload(audit_);if(audit_)payload.AddInput(WordsIn(expected,22));
    payload.AddOutputs({WordsOut(&scratch,22),WordsOut(dims,8),WordsOut(status,1),WordsOut(verdict,1)})
      .SetWorkgroupSize(1).SetDispatchGroupSize(1).CacheHint(audit_).AddUniformVariables({uint32_t(rank),b,t,h,uint32_t(fault_)});
    ORT_RETURN_IF_ERROR(c.RunProgram(payload));
    ValidateShape validate;validate.AddInputs({WordsIn(&scratch,22),WordsIn(dims,8)}).AddOutput(WordsOut(status,1)).SetWorkgroupSize(1).SetDispatchGroupSize(1);
    ORT_RETURN_IF_ERROR(c.RunProgram(validate));
    const auto capacity=static_cast<uint32_t>(y->Shape().Size());
    if(capacity && q->Shape().Size()) {
      ShapeData copy(false);copy.AddInputs({WordsIn(q,static_cast<uint32_t>(q->Shape().Size())),WordsIn(dims,8),WordsIn(status,1)})
        .AddOutput(WordsOut(y,capacity)).SetWorkgroupSize(64).SetDispatchGroupSize((capacity+63)/64).AddUniformVariable({capacity});
      ORT_RETURN_IF_ERROR(c.RunProgram(copy));
    }
    LOGS(c.Logger(),INFO)<<"M0_RW04_SCRATCH_RETIRE run="<<id<<" handle="<<scratch.DataRaw()<<" encoded=1";
    const auto* retired_handle=scratch.DataRaw();scratch=Tensor{};
    LOGS(c.Logger(),INFO)<<"M0_RW04_SCRATCH_RELEASED run="<<id<<" handle="<<retired_handle<<" before_return=1";
    return Status::OK();
  }
 private:bool audit_;int64_t fault_;
};
class ShapeConsumer final : public WebGpuKernel {
 public:explicit ShapeConsumer(const OpKernelInfo& i):WebGpuKernel{i}{}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* y=c.Input<Tensor>(0);const auto* dims=c.Input<Tensor>(1);const auto* status=c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(y&&y->IsDataType<float>()&&y->Shape().NumDimensions()==4&&dims&&dims->IsDataType<int64_t>()&&dims->Shape()==TensorShape{4}&&status&&status->IsDataType<uint32_t>()&&status->Shape()==TensorShape{1},"RW04 consumer admission: recipe descriptors required");
    const auto& d=y->Shape();ORT_RETURN_IF_NOT(d[0]>=0&&d[0]<=1&&d[1]>=0&&d[1]<=512&&d[2]==12&&d[3]==64,"RW04 consumer admission: bounds");
    auto* z=c.Output(0,TensorShape{d[0],12,d[1],64});
    ORT_RETURN_IF_NOT(Device(y)&&Device(dims)&&Device(status)&&Device(z),"RW04 consumer infrastructure: device memory required");
    Trace(c,"consumer458","y",y);Trace(c,"consumer458","shape",dims);Trace(c,"consumer458","status",status);Trace(c,"consumer458","z",z);
    const auto n=static_cast<uint32_t>(z->Shape().Size());
    if(n){ShapeData consumer(true);consumer.AddInputs({WordsIn(y,n),WordsIn(dims,8),WordsIn(status,1)}).AddOutput(WordsOut(z,n)).SetWorkgroupSize(64).SetDispatchGroupSize((n+63)/64).AddUniformVariable({n});ORT_RETURN_IF_ERROR(c.RunProgram(consumer));}
    return Status::OK();
  }
};
}
KernelCreateInfo CreateM0ShapeRegionKernelInfo(const char* name) {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& info,std::unique_ptr<OpKernel>& out)->Status{out=std::make_unique<ShapeRegion>(info);return Status::OK();};
  return {KernelDefBuilder().SetName(name).SetDomain("org.localtts.m0").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0ShapeConsumerKernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& info,std::unique_ptr<OpKernel>& out)->Status{out=std::make_unique<ShapeConsumer>(info);return Status::OK();};
  return {KernelDefBuilder().SetName("ShapeConsumer458V1").SetDomain("org.localtts.m0").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
}
