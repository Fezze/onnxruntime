// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Actual original2426 ->2427..2463, with genuine generator ownership.
#include <algorithm>
#include <array>
#include <optional>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_reconstruction_v3_programs.h"
namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t){return t&&t->Location().device.Type()==OrtDevice::GPU;}
using m0_residual_stages::PhysicalBuffer;
using m0_residual_stages::StorageBudget;
using m0_residual_stages::StorageBudgetWithWindow;
using m0_residual_stages::StorageBudgetWithWindows;
using m0_reconstruction_detail::ValidOutput;
using m0_reconstruction_detail::Arithmetic;
using m0_reconstruction_detail::Guard;
using m0_reconstruction_detail::GatherSpectrumWindow;
using m0_reconstruction_detail::Finish;
class Reconstruction final:public WebGpuKernel{
 public:
 explicit Reconstruction(const OpKernelInfo& i):WebGpuKernel{i},capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)){}
 Status ComputeInternal(ComputeContext& c)const override{
  const auto* x=c.Input<Tensor>(0);const auto* e=c.Input<Tensor>(1);const auto* w=c.Input<Tensor>(2);const auto* window=c.Input<Tensor>(3);
  ORT_RETURN_IF_NOT(Gpu(x)&&x->IsDataType<float>()&&x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&x->Shape()[1]==22&&x->Shape()[2]>0&&x->Shape()[2]<=12288001&&
                        Gpu(e)&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape{32}&&Gpu(w)&&w->IsDataType<float>()&&w->Shape()==TensorShape({22,1,20})&&
                        Gpu(window)&&window->IsDataType<float>()&&window->Shape()==TensorShape{20}&&capacity_>0&&capacity_<=61440000,"RW07 reconstruction3 exact physical descriptors");
  const auto& l=c.DeviceLimits();const uint64_t limit=l.maxComputeWorkgroupsPerDimension,pc=uint64_t(x->Shape()[2]);
  ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage>=5&&l.maxComputeWorkgroupSizeX>=64&&l.maxComputeInvocationsPerWorkgroup>=64&&limit>0,"RW07 reconstruction3 dispatch resources");
  const uint64_t count=uint64_t(capacity_);
  ORT_RETURN_IF_NOT(count<=UINT32_MAX&&count*4<=l.maxBufferSize,"RW07 reconstruction3 physical resource capacity");
  const bool windowed=x->SizeInBytes()>l.maxStorageBufferBindingSize||count*4>l.maxStorageBufferBindingSize;
  const uint64_t alignment=l.minStorageBufferOffsetAlignment;
  uint64_t samples_per_window=count,packed_capacity=pc;
  if(windowed){
   ORT_RETURN_IF_NOT(alignment>0&&alignment%4==0,"RW07 reconstruction3 window alignment");
   const uint64_t prefix=alignment-4,minimum=22ull*6*4;
   ORT_RETURN_IF_NOT(l.maxStorageBufferBindingSize>=prefix+minimum,"RW07 reconstruction3 minimum packed window");
   const uint64_t budget=std::min<uint64_t>(l.maxStorageBufferBindingSize,std::max<uint64_t>(8ull*1024*1024,prefix+minimum));
   const uint64_t frame_budget=(budget-prefix)/(22*4);
   const uint64_t dispatch_capacity=std::min<uint64_t>(UINT32_MAX/64,limit*limit)*64;
   samples_per_window=std::min({count,5*(frame_budget-5),dispatch_capacity});
   ORT_RETURN_IF_NOT(samples_per_window>0,"RW07 reconstruction3 positive PCM window");
   packed_capacity=(samples_per_window+4)/5+5;
  }
  const uint64_t groups=(samples_per_window+63)/64;
  ORT_RETURN_IF_NOT(groups<=limit*limit,"RW07 reconstruction3 window dispatch capacity");
  const std::array<const Tensor*,4> inputs{x,e,w,window};for(const auto* t:inputs)ORT_RETURN_IF_ERROR(PhysicalBuffer(c,t));
  const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();
  M0FullGpuAllocationIdentity sid,data,pair,output;
  ORT_RETURN_IF_NOT(session&&run&&c.M0FullGpuDescribeResourceV2(*e,sid)&&c.M0FullGpuDescribeResourceV2(*x,data),"RW07 reconstruction3 live source allocations");
  const bool paired=c.M0FullGpuDescribePairV2(*e,*x,pair)&&pair.session==session&&pair.run==run&&pair.generation==sid.generation&&pair.data_generation==data.generation&&pair.producer_role==74&&pair.producer_instance==1075;
  auto* y=c.Output(0,TensorShape{capacity_});auto* out=c.Output(1,TensorShape{32});
  ORT_RETURN_IF_NOT(Gpu(y)&&y->IsDataType<float>()&&Gpu(out)&&out->IsDataType<uint32_t>(),"RW07 reconstruction3 output types");
  const std::array<const Tensor*,2> allocated{y,out};
  for(size_t i=0;i<allocated.size();++i){ORT_RETURN_IF_ERROR(PhysicalBuffer(c,allocated[i]));for(size_t j=0;j<i;++j)ORT_RETURN_IF_NOT(allocated[i]->DataRaw()!=allocated[j]->DataRaw(),"RW07 reconstruction3 distinct outputs");for(const auto* t:inputs)ORT_RETURN_IF_NOT(allocated[i]->DataRaw()!=t->DataRaw(),"RW07 reconstruction3 distinct input/output");}
  std::optional<Tensor> packed;
  if(windowed){
   packed.emplace(c.CreateGPUTensor(x->DataType(),TensorShape{22,int64_t(packed_capacity)}));
   ORT_RETURN_IF_ERROR(PhysicalBuffer(c,&*packed));
   for(const auto* t:inputs)ORT_RETURN_IF_NOT(packed->DataRaw()!=t->DataRaw(),"RW07 reconstruction3 distinct packed input");
   for(const auto* t:allocated)ORT_RETURN_IF_NOT(packed->DataRaw()!=t->DataRaw(),"RW07 reconstruction3 distinct packed output");
  }
  // Cover all inverse frames for samples [begin+10,begin+size+9].
  // Extra frames at either edge are harmless; live taps retain their original order.
  const auto frames_for=[&](uint64_t begin,uint64_t size){
   const uint64_t first=std::min<uint64_t>(begin>=9?(begin-9)/5:0,pc-1);
   const uint64_t last=std::min<uint64_t>((begin+size+9)/5,pc-1);
   return std::array<uint64_t,2>{first,last-first+1};
  };
  const auto source_view=[&](uint32_t channel,uint64_t first,uint64_t size){
   return ProgramInput::FloatBufferWindow(x,ProgramTensorMetadataDependency::None,TensorShape{int64_t(size)},
       uint32_t(channel*pc+first),uint32_t(alignment));
  };
  const auto packed_view=[&](uint32_t channel,uint64_t size){
   return ProgramOutput::FloatBufferWindow(&*packed,ProgramTensorMetadataDependency::None,TensorShape{int64_t(size)},
       uint32_t(channel*packed_capacity),uint32_t(alignment));
  };
  const auto pcm_view=[&](uint64_t begin,uint64_t size){
   return ProgramOutput::FloatBufferWindow(y,ProgramTensorMetadataDependency::None,TensorShape{int64_t(size)},
       uint32_t(begin),uint32_t(alignment));
  };
  ORT_RETURN_IF_ERROR(StorageBudget(c,{e,window,out}));
  if(windowed){
   for(uint64_t begin=0;begin<count;begin+=samples_per_window){
    const uint64_t size=std::min(samples_per_window,count-begin);const auto frames=frames_for(begin,size);
    ORT_RETURN_IF_NOT(frames[1]<=packed_capacity,"RW07 reconstruction3 packed frame span");
    for(uint32_t channel=0;channel<22;++channel){
     ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,source_view(channel,frames[0],frames[1]),packed_view(channel,frames[1]),{out}));
    }
    ORT_RETURN_IF_ERROR(StorageBudgetWithWindow(c,pcm_view(begin,size),{&*packed,w,window,out}));
   }
  }else{
   ORT_RETURN_IF_ERROR(StorageBudget(c,{x,w,window,out,y}));
  }
  ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out,*y,75,1076,output),"RW07 reconstruction3 actual PCM ownership");
  const m0_extent_abi2::OpaqueOwnership source{session,run,sid.generation,data.generation,74,1075},owner{session,run,output.generation,output.data_generation,75,1076};
  ORT_RETURN_IF_NOT(source.Valid()&&owner.Valid(),"RW07 reconstruction3 identities");const auto sw=source.UniformWords(),ow=owner.UniformWords();
  const auto span=[](const auto& a){return gsl::span<const uint32_t>{a.data(),a.size()};};
  Guard guard;guard.AddInputs({{e,ProgramTensorMetadataDependency::None},{window,ProgramTensorMetadataDependency::None}}).AddOutput({out,ProgramTensorMetadataDependency::None})
   .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({uint32_t(pc),uint32_t(capacity_),uint32_t(paired),span(sw),span(ow)});ORT_RETURN_IF_ERROR(c.RunProgram(guard));
  for(uint64_t begin=0;begin<count;begin+=samples_per_window){
   const uint64_t size=std::min(samples_per_window,count-begin);
   const auto frames=windowed?frames_for(begin,size):std::array<uint64_t,2>{0,pc};
   if(windowed){
    const uint64_t gather_groups=(frames[1]+63)/64,gx=std::min(gather_groups,limit),gy=(gather_groups+gx-1)/gx;
    for(uint32_t channel=0;channel<22;++channel){
     GatherSpectrumWindow gather;
     gather.AddInput(source_view(channel,frames[0],frames[1])).AddInput({out,ProgramTensorMetadataDependency::None})
      .AddOutput(packed_view(channel,frames[1])).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
      .AddUniformVariables({uint32_t(pc),uint32_t(capacity_),uint32_t(begin),uint32_t(frames[0]),uint32_t(frames[1]),span(ow)});
     ORT_RETURN_IF_ERROR(c.RunProgram(gather));
    }
   }
   const uint64_t finish_groups=(size+63)/64,gx=std::min(finish_groups,limit),gy=(finish_groups+gx-1)/gx;
   Finish finish;
   finish.AddInputs({{windowed?&*packed:x,ProgramTensorMetadataDependency::None},{w,ProgramTensorMetadataDependency::None},
       {window,ProgramTensorMetadataDependency::None},{out,ProgramTensorMetadataDependency::None}})
    .AddOutput(windowed?pcm_view(begin,size):ProgramOutput{y,ProgramTensorMetadataDependency::None})
    .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
    .AddUniformVariables({uint32_t(pc),uint32_t(capacity_),uint32_t(begin),uint32_t(size),uint32_t(frames[0]),uint32_t(packed_capacity),span(ow)});
   ORT_RETURN_IF_ERROR(c.RunProgram(finish));
  }
  LOGS(c.Logger(),INFO)<<"M0_RW07_RECONSTRUCTION_V3 actual_original2426=1 role=75 instance=1076 GPU_dispatches="
   <<1+((count+samples_per_window-1)/samples_per_window)*(windowed?23:1)
   <<" packed_spectrum_bytes="<<(windowed?packed->SizeInBytes():0)<<" inverse_scratch_bytes=0 host_payload_reads=0";return Status::OK();
 }
 private:const int64_t capacity_;
};
class PacketProgram final:public Program<PacketProgram>{
 public:
 PacketProgram():Program{"RW07ActualPCM2463TerminalPacketV3"}{}
 WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_capacity",ProgramUniformVariableDataType::Uint32},{"output_capacity",ProgramUniformVariableDataType::Uint32},{"identity",ProgramUniformVariableDataType::Uint32});
 Status GenerateShaderCode(ShaderHelper& s)const override{
  const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);const auto& p=s.AddOutput("packet",ShaderUsage::UseValueTypeAlias);ValidOutput(s,e);auto& f=s.MainFunctionBody();
  f<<"for(var i=0u;i<18u;i++){"<<p.SetByOffset("i","0u")<<"}"<<p.SetByOffset("0u","103u")
   <<"if("<<e.GetByOffset("0u")<<"!=2u||"<<e.GetByOffset("1u")<<"!=1u){return;}for(var i=0u;i<10u;i++){if("<<e.GetByOffset("22u+i")<<"!="<<GetElementAt("uniforms.identity","i",10)<<"){return;}}"
   <<"let status="<<e.GetByOffset("20u")<<";if(status!=0u){if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}for(var i=2u;i<20u;i++){if("<<e.GetByOffset("i")<<"!=0u){return;}}if("<<e.GetByOffset("21u")<<"!=0u){return;}"<<p.SetByOffset("0u","status")<<"return;}if(!valid_output()){return;}"
   <<p.SetByOffset("0u","0u")<<p.SetByOffset("1u","1u")<<p.SetByOffset("2u",e.GetByOffset("2u"));return Status::OK();
 }
};
class Packet final:public WebGpuKernel{
 public:
 explicit Packet(const OpKernelInfo& i):WebGpuKernel{i}{}
 Status ComputeInternal(ComputeContext& c)const override{
  const auto* e=c.Input<Tensor>(0);const auto* y=c.Input<Tensor>(1);
  ORT_RETURN_IF_NOT(Gpu(e)&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape{32}&&Gpu(y)&&y->IsDataType<float>()&&y->Shape().NumDimensions()==1&&y->Shape()[0]>0&&y->Shape()[0]<=61440000,"RW07 PCM3 terminal descriptors");
  const auto& l=c.DeviceLimits();ORT_RETURN_IF_NOT(l.maxStorageBuffersPerShaderStage>=2&&l.maxComputeWorkgroupsPerDimension>=1&&l.maxComputeWorkgroupSizeX>=1&&l.maxComputeInvocationsPerWorkgroup>=1,"RW07 PCM3 terminal resources");
  ORT_RETURN_IF_ERROR(PhysicalBuffer(c,e));ORT_RETURN_IF_ERROR(PhysicalBuffer(c,y));
  const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();M0FullGpuAllocationIdentity pair;
  ORT_RETURN_IF_NOT(session&&run&&c.M0FullGpuDescribePairV2(*e,*y,pair)&&pair.session==session&&pair.run==run&&pair.producer_role==75&&pair.producer_instance==1076,"RW07 PCM3 terminal actual pair");
  auto* p=c.Output(0,TensorShape{18});ORT_RETURN_IF_NOT(Gpu(p)&&p->IsDataType<uint32_t>()&&p->DataRaw()!=e->DataRaw()&&p->DataRaw()!=y->DataRaw(),"RW07 PCM3 terminal distinct output");
  ORT_RETURN_IF_ERROR(StorageBudget(c,{e,p}));
  const m0_extent_abi2::OpaqueOwnership owner{session,run,pair.generation,pair.data_generation,75,1076};ORT_RETURN_IF_NOT(owner.Valid(),"RW07 PCM3 terminal identity");const auto words=owner.UniformWords();
  const uint32_t cap=uint32_t(y->Shape()[0]);PacketProgram program;
  program.AddInput({e,ProgramTensorMetadataDependency::None}).AddOutput({p,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(1)
   .AddUniformVariables({12288001u,cap,gsl::span<const uint32_t>{words.data(),words.size()}});ORT_RETURN_IF_ERROR(c.RunProgram(program));
  ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p),"RW07 PCM3 terminal map fault registration");return Status::OK();
 }
};
}  // namespace
KernelCreateInfo CreateM0FullGpuReconstructionExtentV3KernelInfo(){
 KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{out=std::make_unique<Reconstruction>(i);return Status::OK();};
 return {KernelDefBuilder().SetName("ReconstructionExtentV3").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuPCM2463TerminalPacketV3KernelInfo(){
 KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{out=std::make_unique<Packet>(i);return Status::OK();};
 return {KernelDefBuilder().SetName("PCM2463TerminalPacketV3").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
}  // namespace onnxruntime::webgpu
