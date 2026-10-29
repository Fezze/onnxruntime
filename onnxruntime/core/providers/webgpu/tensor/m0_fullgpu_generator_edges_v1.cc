// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original generator upsample/join/parallel-path average and spectral projection edges.
#include <algorithm>
#include <array>
#include <optional>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_generator_edges_programs_v1.h"
namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t){return t&&t->Location().device.Type()==OrtDevice::GPU;}
using m0_residual_stages::ValidOutput;
using m0_residual_stages::OriginalArithmetic;
using m0_residual_stages::PhysicalBuffer;
using m0_residual_stages::StorageBudget;
using m0_residual_stages::StorageBudgetWithWindows;
class EdgeGuard final : public Program<EdgeGuard> {
 public:
  EdgeGuard(uint32_t mode,uint32_t pairs):Program{"RW07GeneratorEdgeGuardV1"},mode_(mode),pairs_(pairs) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"source_capacity",ProgramUniformVariableDataType::Uint32},
    {"source_channels",ProgramUniformVariableDataType::Uint32},
    {"source_minimum",ProgramUniformVariableDataType::Uint32},
    {"source_maximum",ProgramUniformVariableDataType::Uint32},
    {"source_modulus",ProgramUniformVariableDataType::Uint32},
    {"source_remainder",ProgramUniformVariableDataType::Uint32},
    {"pair_valid",ProgramUniformVariableDataType::Uint32},
    {"source_identity",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},
    {"channels",ProgramUniformVariableDataType::Uint32},
    {"stage",ProgramUniformVariableDataType::Uint32},
    {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    std::array<const ShaderVariableHelper*,3> extents{};
    for(uint32_t j=0;j<pairs_;j++){extents[j]=&s.AddInput("source"+std::to_string(j),ShaderUsage::UseValueTypeAlias);}
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    for(uint32_t j=0;j<pairs_;j++){
      const auto& e=*extents[j];auto& f=s.AdditionalImplementation();const auto js=std::to_string(j)+"u";
      f<<"fn source_status"<<j<<"()->u32 {if("<<GetElementAt("uniforms.pair_valid",js,3)<<"==0u||"
       <<e.GetByOffset("0u")<<"!=2u||"<<e.GetByOffset("1u")<<"!=3u){return 103u;}"
       <<"for(var i=0u;i<10u;i++){if("<<e.GetByOffset("22u+i")<<"!="
       <<GetElementAt("uniforms.source_identity",std::to_string(j*10)+"u+i",30)<<"){return 103u;}}"
       <<"let status="<<e.GetByOffset("20u")<<";if(status!=0u){"
       <<"if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return 103u;}"
       <<"for(var i=2u;i<20u;i++){if("<<e.GetByOffset("i")<<"!=0u){return 103u;}}"
       <<"if("<<e.GetByOffset("21u")<<"!=0u){return 103u;}return status;}"
       <<"if("<<e.GetByOffset("2u")<<"!=1u||"<<e.GetByOffset("3u")<<"!=0u||"
       <<e.GetByOffset("4u")<<"!=uniforms.source_channels||"<<e.GetByOffset("5u")<<"!=0u||"
       <<e.GetByOffset("7u")<<"!=0u||"<<e.GetByOffset("19u")<<"!=0u||"
       <<e.GetByOffset("21u")<<"!=63u){return 103u;}"
       <<"for(var i=8u;i<18u;i++){if("<<e.GetByOffset("i")<<"!=0u){return 103u;}}let n="<<e.GetByOffset("6u")<<";"
       <<"if(n<uniforms.source_minimum||n>uniforms.source_maximum||n%uniforms.source_modulus!=uniforms.source_remainder||"
       <<e.GetByOffset("18u")<<"!=uniforms.source_channels*n){return 103u;}"
       <<"return 0u;}\n";
    }
    auto& f=s.MainFunctionBody();
    f<<"for(var i=0u;i<32u;i++){"<<o.SetByOffset("i","0u")<<"}"<<o.SetByOffset("0u","2u")<<o.SetByOffset("1u","3u")
     <<"for(var i=0u;i<10u;i++){"<<o.SetByOffset("22u+i",GetElementAt("uniforms.identity","i",10))<<"}";
    for(uint32_t j=0;j<pairs_;j++){f<<"let status"<<j<<"=source_status"<<j<<"();";}
    for(uint32_t j=0;j<pairs_;j++){f<<"if(status"<<j<<"==103u){"<<o.SetByOffset("20u","103u")<<"return;}";}
    for(uint32_t i=0;i<pairs_;i++){for(uint32_t j=i+1;j<pairs_;j++){
      f<<"if(status"<<i<<"==0u&&status"<<j<<"==0u&&"<<extents[i]->GetByOffset("6u")<<"!="<<extents[j]->GetByOffset("6u")
       <<"){"<<o.SetByOffset("20u","103u")<<"return;}";}}
    for(uint32_t j=0;j<pairs_;j++){f<<"if(status"<<j<<"!=0u){"<<o.SetByOffset("20u","status"+std::to_string(j))<<"return;}";}
    f<<"let n="<<extents[0]->GetByOffset("6u")<<";";
    for(uint32_t j=0;j<pairs_;j++){f<<"if(n>"<<GetElementAt("uniforms.source_capacity",std::to_string(j)+"u",3)
      <<"){"<<o.SetByOffset("20u","102u")<<"return;}";}
    if(mode_==0){f<<"let count=select(10u*n,6u*n+1u,uniforms.stage==1u);";}
    else{f<<"let count=n;";}
    f<<"if(count>uniforms.output_capacity){"<<o.SetByOffset("20u","102u")<<"return;}"
     <<o.SetByOffset("2u","1u")<<o.SetByOffset("4u","uniforms.channels")<<o.SetByOffset("6u","count")
     <<o.SetByOffset("18u","uniforms.channels*count")<<o.SetByOffset("21u","63u");
    return Status::OK();
  }
 private:
  const uint32_t mode_,pairs_;
};
using m0_generator_edges_detail::EdgeData;
using m0_generator_edges_detail::UpsampleTapColumn;
using m0_generator_edges_detail::UpsampleTapAdd;
class GeneratorEdge : public WebGpuKernel {
 public:
  GeneratorEdge(const OpKernelInfo& i,uint32_t mode):WebGpuKernel{i},mode_(mode),
      stage_(i.GetAttrOrDefault<int64_t>("stage",-1)),capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    ORT_RETURN_IF_NOT((stage_==0||stage_==1)&&capacity_>0&&capacity_<=12288001&&(mode_!=3||stage_==1),
                      "RW07 generator edge closed stage/capacity");
    const uint32_t pairs=mode_==1?2u:(mode_==2?3u:1u);
    const uint32_t channels=mode_==3?22u:(stage_==0?256u:128u);
    const uint32_t source_channels=mode_==0?(stage_==0?512u:256u):(mode_==3?128u:channels);
    const uint32_t minimum=stage_==0?20u:121u,maximum=stage_==0?2048000u:12288001u;
    const uint32_t modulus=stage_==0?20u:120u,remainder=stage_==0?0u:1u;
    const uint32_t source_min=mode_==0?(stage_==0?2u:20u):minimum;
    const uint32_t source_max=mode_==0?(stage_==0?204800u:2048000u):maximum;
    const uint32_t source_mod=mode_==0?(stage_==0?2u:20u):modulus,source_rem=mode_==0?0u:remainder;
    const uint32_t role=mode_==0?(stage_==0?50u:62u):(mode_==1?(stage_==0?51u:63u):(mode_==2?(stage_==0?61u:73u):74u));
    std::array<const Tensor*,3> x{},e{};std::array<uint32_t,3> capacities{},paired{};
    std::array<uint32_t,30> source_words{};
    const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();
    ORT_RETURN_IF_NOT(session&&run,"RW07 generator edge current execution identity");
    const auto& limits=c.DeviceLimits();
    auto binding=[&](const Tensor* t)->Status{return PhysicalBuffer(c,t);};
    for(uint32_t j=0;j<pairs;j++){
      x[j]=c.Input<Tensor>(2*j);e[j]=c.Input<Tensor>(2*j+1);
      ORT_RETURN_IF_NOT(Gpu(x[j])&&x[j]->IsDataType<float>()&&x[j]->Shape().NumDimensions()==3&&
                            x[j]->Shape()[0]==1&&x[j]->Shape()[1]==source_channels&&x[j]->Shape()[2]>0&&
                            x[j]->Shape()[2]<=12288001&&Gpu(e[j])&&e[j]->IsDataType<uint32_t>()&&e[j]->Shape()==TensorShape{32},
                        "RW07 generator edge source pair descriptors");
      ORT_RETURN_IF_ERROR(binding(x[j]));ORT_RETURN_IF_ERROR(binding(e[j]));capacities[j]=uint32_t(x[j]->Shape()[2]);
      M0FullGpuAllocationIdentity extent,data,pair;
      ORT_RETURN_IF_NOT(c.M0FullGpuDescribeResourceV2(*e[j],extent)&&c.M0FullGpuDescribeResourceV2(*x[j],data),
                        "RW07 generator edge actual source resources");
      const uint32_t source_role=mode_==0?(stage_==0?24u:61u):(mode_==1?(j==0?(stage_==0?50u:62u):(stage_==0?44u:45u)):
          (mode_==2?(stage_==0?54u:66u)+3*j:73u));
      paired[j]=uint32_t(c.M0FullGpuDescribePairV2(*e[j],*x[j],pair)&&pair.session==session&&pair.run==run&&
          pair.generation==extent.generation&&pair.data_generation==data.generation&&pair.producer_role==source_role&&pair.producer_instance==source_role+1001);
      const m0_extent_abi2::OpaqueOwnership owner{session,run,extent.generation,data.generation,source_role,source_role+1001};
      ORT_RETURN_IF_NOT(owner.Valid(),"RW07 generator edge source ownership descriptor");
      const auto words=owner.UniformWords();std::copy(words.begin(),words.end(),source_words.begin()+10*j);
    }
    const Tensor* w=nullptr;const Tensor* b=nullptr;
    if(mode_==0||mode_==3){
      w=c.Input<Tensor>(2);b=c.Input<Tensor>(3);
      const uint32_t kernel=mode_==3?7u:(stage_==0?20u:12u);
      const TensorShape ws=mode_==0?TensorShape({source_channels,channels,kernel}):TensorShape({channels,source_channels,kernel});
      ORT_RETURN_IF_NOT(Gpu(w)&&w->IsDataType<float>()&&w->Shape()==ws&&Gpu(b)&&b->IsDataType<float>()&&b->Shape()==TensorShape({channels}),
                        "RW07 generator edge exact original immutable parameters");
      ORT_RETURN_IF_ERROR(binding(w));ORT_RETURN_IF_ERROR(binding(b));
    }
    const uint64_t groups=(uint64_t(channels)*capacity_+63)/64,limit=limits.maxComputeWorkgroupsPerDimension;
    const uint64_t bytes=uint64_t(channels)*capacity_*4;
    const bool windowed=mode_==1||mode_==2;
    const bool windowed_upsample=mode_==0&&(x[0]->SizeInBytes()>limits.maxStorageBufferBindingSize||bytes>limits.maxStorageBufferBindingSize);
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=5&&limits.maxComputeWorkgroupSizeX>=64&&
        limits.maxComputeInvocationsPerWorkgroup>=64&&limit>0&&(mode_!=0||windowed_upsample||groups<=limit*limit)&&bytes<=limits.maxBufferSize,"RW07 generator edge output resources");
    uint32_t window_channels=channels;
    uint32_t projection_input_channels=0,projection_output_channels=0;
    uint32_t upsample_input_channels=0,upsample_output_channels=0;
    if(windowed_upsample){
      ORT_RETURN_IF_ERROR(m0_residual_stages::ResidualTileChannels(c,capacities[0],uint32_t(capacity_),
          std::max(source_channels,channels),upsample_output_channels));
      upsample_output_channels=std::min(upsample_output_channels,channels);
      const uint64_t prefix=limits.minStorageBufferOffsetAlignment-sizeof(float);
      upsample_input_channels=static_cast<uint32_t>(std::min<uint64_t>(source_channels,
          (limits.maxStorageBufferBindingSize-prefix)/(uint64_t(capacities[0])*sizeof(float))));
      const uint64_t window_groups=(uint64_t(upsample_output_channels)*uint64_t(capacity_)+63)/64;
      ORT_RETURN_IF_NOT(upsample_input_channels>0&&window_groups<=limit*limit,"RW07 upsample window dispatch limit");
    }
    if(windowed){
      const uint32_t source_capacity=*std::max_element(capacities.begin(),capacities.end());
      uint32_t checked_tile=0;
      ORT_RETURN_IF_ERROR(m0_residual_stages::ResidualTileChannels(c,source_capacity,uint32_t(capacity_),channels,checked_tile));
      const uint64_t prefix=limits.minStorageBufferOffsetAlignment-sizeof(float);
      const uint64_t row_bytes=uint64_t(std::max(source_capacity,uint32_t(capacity_)))*sizeof(float);
      window_channels=static_cast<uint32_t>(std::min<uint64_t>(channels,(limits.maxStorageBufferBindingSize-prefix)/row_bytes));
      const uint64_t window_groups=(uint64_t(window_channels)*uint64_t(capacity_)+63)/64;
      ORT_RETURN_IF_NOT(window_channels>0&&window_groups<=limit*limit,"RW07 generator edge window dispatch limit");
    }
    if(mode_==3){
      ORT_RETURN_IF_ERROR(m0_residual_stages::ResidualTileChannels(c,capacities[0],uint32_t(capacity_),128u,projection_input_channels));
      const uint64_t prefix=limits.minStorageBufferOffsetAlignment-sizeof(float);
      const uint64_t row_bytes=uint64_t(std::max(capacities[0],uint32_t(capacity_)))*sizeof(float);
      projection_output_channels=static_cast<uint32_t>(std::min<uint64_t>(channels,(limits.maxStorageBufferBindingSize-prefix)/row_bytes));
      const uint64_t window_groups=(uint64_t(projection_output_channels)*uint64_t(capacity_)+63)/64;
      ORT_RETURN_IF_NOT(projection_output_channels>0&&window_groups<=limit*limit,"RW07 projection window dispatch limit");
    }
    auto* y=c.Output(0,TensorShape{1,channels,capacity_});auto* out=c.Output(1,TensorShape{32});
    ORT_RETURN_IF_NOT(Gpu(y)&&y->IsDataType<float>()&&Gpu(out)&&out->IsDataType<uint32_t>()&&y->DataRaw()!=out->DataRaw(),
                      "RW07 generator edge distinct output allocations");
    for(const auto* target:{y,out}){
      ORT_RETURN_IF_ERROR(binding(target));for(uint32_t j=0;j<pairs;j++){
        ORT_RETURN_IF_NOT(target->DataRaw()!=x[j]->DataRaw()&&target->DataRaw()!=e[j]->DataRaw(),"RW07 generator edge source/output alias");}
      if(w){ORT_RETURN_IF_NOT(target->DataRaw()!=w->DataRaw()&&target->DataRaw()!=b->DataRaw(),"RW07 generator edge parameter/output alias");}
    }
    std::optional<Tensor> scratch;
    if(mode_==3||windowed_upsample){
      scratch.emplace(c.CreateGPUTensor(x[0]->DataType(),TensorShape{1,windowed_upsample?upsample_output_channels:channels,capacity_}));
      ORT_RETURN_IF_ERROR(binding(&*scratch));
      ORT_RETURN_IF_NOT(scratch->DataRaw()!=y->DataRaw()&&scratch->DataRaw()!=out->DataRaw(),
          "RW07 generator edge private scratch/output alias");
      for(uint32_t j=0;j<pairs;j++){
        ORT_RETURN_IF_NOT(scratch->DataRaw()!=x[j]->DataRaw()&&scratch->DataRaw()!=e[j]->DataRaw(),
            "RW07 generator edge private scratch/source alias");
      }
      if(w){ORT_RETURN_IF_NOT(scratch->DataRaw()!=w->DataRaw()&&scratch->DataRaw()!=b->DataRaw(),
          "RW07 generator edge private scratch/parameter alias");}
    }
    // Preflight every actual dispatch, using the same tensor-byte segment counts
    // as ProgramManager, before producing the output sidecar.
    const auto input_window=[&](uint32_t j,uint32_t offset,uint32_t count){
      return ProgramInput::FloatBufferWindow(x[j],ProgramTensorMetadataDependency::None,
          TensorShape{1,count,capacities[j]},offset*capacities[j],limits.minStorageBufferOffsetAlignment);
    };
    const auto output_window=[&](Tensor* target,uint32_t offset,uint32_t count){
      return ProgramOutput::FloatBufferWindow(target,ProgramTensorMetadataDependency::None,
          TensorShape{1,count,capacity_},offset*uint32_t(capacity_),limits.minStorageBufferOffsetAlignment);
    };
    const auto spectral_input_window=[&](uint32_t component,uint32_t offset,uint32_t count){
      return ProgramInput::FloatBufferWindow(&*scratch,ProgramTensorMetadataDependency::None,
          TensorShape{1,count,capacity_},(component*11u+offset)*uint32_t(capacity_),limits.minStorageBufferOffsetAlignment);
    };
    const auto column_input_window=[&](uint32_t count){
      return ProgramInput::FloatBufferWindow(&*scratch,ProgramTensorMetadataDependency::None,
          TensorShape{1,count,capacity_},0u,limits.minStorageBufferOffsetAlignment);
    };
    if(pairs==3){ORT_RETURN_IF_ERROR(StorageBudget(c,{e[0],e[1],e[2],out}));}
    else if(pairs==2){ORT_RETURN_IF_ERROR(StorageBudget(c,{e[0],e[1],out}));}
    else{ORT_RETURN_IF_ERROR(StorageBudget(c,{e[0],out}));}
    if(windowed_upsample){
      for(uint32_t o=0;o<channels;o+=upsample_output_channels){
        const uint32_t count=std::min(upsample_output_channels,channels-o);
        for(uint32_t i=0;i<source_channels;i+=upsample_input_channels){
          ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,input_window(0,i,std::min(upsample_input_channels,source_channels-i)),
              output_window(&*scratch,0u,count),{out,w}));
        }
        ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,column_input_window(count),output_window(y,o,count),{out,b}));
      }
    }else if(windowed){
      for(uint32_t offset=0;offset<channels;offset+=window_channels){
        const uint32_t count=std::min(window_channels,channels-offset);
        if(pairs==3){ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,
            {input_window(0,offset,count),input_window(1,offset,count),input_window(2,offset,count)},output_window(y,offset,count),{out}));}
        else{ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,
            {input_window(0,offset,count),input_window(1,offset,count)},output_window(y,offset,count),{out}));}
      }
    }else if(mode_==3){
      for(uint32_t i=0;i<128;i+=projection_input_channels){
        for(uint32_t o=0;o<channels;o+=projection_output_channels){
          ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,input_window(0,i,std::min(projection_input_channels,128u-i)),
              output_window(&*scratch,o,std::min(projection_output_channels,channels-o)),{out,w,b}));
        }
      }
      const uint32_t spectral_tile=std::min(11u,projection_output_channels);
      for(uint32_t component=0;component<2;component++){
        for(uint32_t offset=0;offset<11;offset+=spectral_tile){
          const uint32_t count=std::min(spectral_tile,11u-offset);
          ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,
              {spectral_input_window(0,offset,count),spectral_input_window(1,offset,count)},
              output_window(y,component*11u+offset,count),{out}));
        }
      }
    }else if(mode_==0){ORT_RETURN_IF_ERROR(StorageBudget(c,{x[0],out,w,b,y}));}
    else{ORT_RETURN_IF_ERROR(StorageBudget(c,{x[0],x[1],out,y}));}
    M0FullGpuAllocationIdentity output;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out,*y,role,role+1001,output),"RW07 generator edge output claim");
    const m0_extent_abi2::OpaqueOwnership owner{session,run,output.generation,output.data_generation,role,role+1001};
    ORT_RETURN_IF_NOT(owner.Valid(),"RW07 generator edge output ownership");const auto words=owner.UniformWords();
    const auto identity=gsl::span<const uint32_t>{words.data(),words.size()};
    const auto capspan=gsl::span<const uint32_t>{capacities.data(),capacities.size()};
    EdgeGuard guard{mode_,pairs};guard.CacheHint(std::to_string(mode_));
    for(uint32_t j=0;j<pairs;j++){guard.AddInput({e[j],ProgramTensorMetadataDependency::None});}
    guard.AddOutput({out,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(1)
      .AddUniformVariables({capspan,source_channels,source_min,source_max,source_mod,source_rem,
          gsl::span<const uint32_t>{paired.data(),paired.size()},gsl::span<const uint32_t>{source_words.data(),source_words.size()},
          uint32_t(capacity_),channels,uint32_t(stage_),identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    if(windowed_upsample){
      uint64_t upsample_dispatches=1;
      const uint32_t kernel=stage_==0?20u:12u;
      for(uint32_t o=0;o<channels;o+=upsample_output_channels){
        const uint32_t co=std::min(upsample_output_channels,channels-o);
        const uint64_t window_groups=(uint64_t(co)*uint64_t(capacity_)+63)/64;
        const uint64_t gx=std::min(window_groups,limit),gy=(window_groups+gx-1)/gx;
        for(uint32_t tap=0;tap<kernel;tap++){
          for(uint32_t i=0;i<source_channels;i+=upsample_input_channels){
            const uint32_t ic=std::min(upsample_input_channels,source_channels-i);
            const uint32_t chunk=m0_residual_stages::ResidualConvElementsPerDispatchV1(ic,1);
            ORT_RETURN_IF_NOT(chunk>0,"RW07 upsample column dispatch work bound");
            const uint64_t elements=uint64_t(co)*capacity_;
            for(uint64_t first=0;first<elements;first+=chunk){
              const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(chunk,elements-first));
              const uint64_t cg=(uint64_t(size)+63)/64,cx=std::min(cg,limit),cy=(cg+cx-1)/cx;
              UpsampleTapColumn column;
              column.AddInput(input_window(0,i,ic)).AddInputs({{out,ProgramTensorMetadataDependency::None},{w,ProgramTensorMetadataDependency::None}})
                  .AddOutput(output_window(&*scratch,0u,co)).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(cx),uint32_t(cy))
                  .AddUniformVariables({capacities[0],uint32_t(capacity_),channels,minimum,maximum,modulus,remainder,
                      uint32_t(stage_),tap,i,ic,o,co,uint32_t(first),size,identity});
              ORT_RETURN_IF_ERROR(m0_residual_stages::RunHeavyProgram(c,column,cg*64*ic));
              ++upsample_dispatches;
            }
          }
          UpsampleTapAdd add;
          add.AddInput(column_input_window(co)).AddInputs({{out,ProgramTensorMetadataDependency::None},{b,ProgramTensorMetadataDependency::None}})
              .AddOutput(output_window(y,o,co)).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
              .AddUniformVariables({capacities[0],uint32_t(capacity_),channels,minimum,maximum,modulus,remainder,
                  uint32_t(stage_),tap,o,co,identity});
          ORT_RETURN_IF_ERROR(c.RunProgram(add));
          ++upsample_dispatches;
        }
      }
      LOGS(c.Logger(),INFO)<<"M0_RW07_GENERATOR_EDGE_V1 mode="<<mode_<<" stage="<<stage_<<" role="<<role
          <<" session="<<session<<" run="<<run<<" GPU_dispatches="<<upsample_dispatches
          <<" max_upsample_weight_products_per_dispatch="<<m0_residual_stages::kResidualProductsPerDispatchV1
          <<" input_tile_channels="<<upsample_input_channels<<" output_tile_channels="<<upsample_output_channels
          <<" column_scratch_bytes="<<scratch->SizeInBytes()<<" host_payload_reads=0";
      return Status::OK();
    }
    if(windowed){
      for(uint32_t offset=0;offset<channels;offset+=window_channels){
        const uint32_t count=std::min(window_channels,channels-offset);
        EdgeData data{mode_,pairs};data.CacheHint(std::to_string(mode_));
        for(uint32_t j=0;j<pairs;j++) data.AddInput(input_window(j,offset,count));
        const uint64_t window_groups=(uint64_t(count)*uint64_t(capacity_)+63)/64;
        const uint64_t gx=std::min(window_groups,limit),gy=(window_groups+gx-1)/gx;
        data.AddInput({out,ProgramTensorMetadataDependency::None}).AddOutput(output_window(y,offset,count))
            .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
            .AddUniformVariables({capspan,uint32_t(capacity_),channels,minimum,maximum,modulus,remainder,uint32_t(stage_),count,0u,0u,0u,identity});
        ORT_RETURN_IF_ERROR(c.RunProgram(data));
      }
      LOGS(c.Logger(),INFO)<<"M0_RW07_GENERATOR_EDGE_V1 mode="<<mode_<<" stage="<<stage_<<" role="<<role
          <<" session="<<session<<" run="<<run<<" GPU_dispatches="<<(1+(channels+window_channels-1)/window_channels)
          <<" window_channels="<<window_channels<<" merge_scratch_bytes=0 host_payload_reads=0";
      return Status::OK();
    }
    if(mode_==3){
      auto& convolved=*scratch;
      for(uint32_t i=0;i<128;i+=projection_input_channels){
        for(uint32_t o=0;o<channels;o+=projection_output_channels){
          const uint32_t ic=std::min(projection_input_channels,128u-i),co=std::min(projection_output_channels,channels-o);
          const uint64_t window_groups=(uint64_t(co)*uint64_t(capacity_)+63)/64;
          const uint64_t gx=std::min(window_groups,limit),gy=(window_groups+gx-1)/gx;
          m0_residual_stages::GeneratorChannelConv convolution;
          convolution.AddInput(input_window(0,i,ic)).AddInputs({{out,ProgramTensorMetadataDependency::None},
              {w,ProgramTensorMetadataDependency::None},{b,ProgramTensorMetadataDependency::None}})
              .AddOutput(output_window(&convolved,o,co)).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
              .AddUniformVariables({capacities[0],uint32_t(capacity_),channels,minimum,maximum,modulus,remainder,
                  128u,7u,1u,3u,1u,0u,1u,i,ic,o,co,identity});
          ORT_RETURN_IF_ERROR(c.RunProgram(convolution));
        }
      }
      const uint32_t spectral_tile=std::min(11u,projection_output_channels);
      const std::array<uint32_t,3> scratch_cap{uint32_t(capacity_),uint32_t(capacity_),0u};
      for(uint32_t component=0;component<2;component++){
        for(uint32_t offset=0;offset<11;offset+=spectral_tile){
          const uint32_t count=std::min(spectral_tile,11u-offset);
          const uint64_t window_groups=(uint64_t(count)*uint64_t(capacity_)+63)/64;
          const uint64_t gx=std::min(window_groups,limit),gy=(window_groups+gx-1)/gx;
          EdgeData spectral{4,2};spectral.CacheHint("spectral");
          spectral.AddInputs({spectral_input_window(0,offset,count),spectral_input_window(1,offset,count),
              {out,ProgramTensorMetadataDependency::None}})
              .AddOutput(output_window(y,component*11u+offset,count)).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
              .AddUniformVariables({gsl::span<const uint32_t>{scratch_cap.data(),scratch_cap.size()},uint32_t(capacity_),channels,
                  minimum,maximum,modulus,remainder,uint32_t(stage_),count,component,0u,0u,identity});
          ORT_RETURN_IF_ERROR(c.RunProgram(spectral));
        }
      }
      LOGS(c.Logger(),INFO)<<"M0_RW07_GENERATOR_EDGE_V1 mode="<<mode_<<" stage="<<stage_<<" role="<<role
          <<" session="<<session<<" run="<<run<<" GPU_dispatches="<<(1+((128u+projection_input_channels-1)/projection_input_channels)*
              ((channels+projection_output_channels-1)/projection_output_channels)+2*((11u+spectral_tile-1)/spectral_tile))
          <<" input_tile_channels="<<projection_input_channels<<" output_tile_channels="<<projection_output_channels<<" host_payload_reads=0";
      return Status::OK();
    }
    // Bound convolution weight products, excluding activation/index operations.
    // At most two transposed-convolution taps contribute to one output.
    const uint32_t chunk=m0_residual_stages::ResidualConvElementsPerDispatchV1(source_channels,2);
    ORT_RETURN_IF_NOT(chunk>0,"RW07 whole upsample dispatch work bound");
    const uint64_t elements=uint64_t(channels)*capacity_;
    uint64_t upsample_dispatches=1;
    for(uint64_t first=0;first<elements;first+=chunk){
      const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(chunk,elements-first));
      EdgeData data{mode_,pairs,M0VectorizedConvV1()};data.CacheHint(std::to_string(mode_));
      for(uint32_t j=0;j<pairs;j++){data.AddInput({x[j],ProgramTensorMetadataDependency::None});}
      data.AddInput({out,ProgramTensorMetadataDependency::None});
      if(w){data.AddInputs({{w,ProgramTensorMetadataDependency::None},{b,ProgramTensorMetadataDependency::None}});}
      const uint64_t cg=(uint64_t(size)+63)/64,gx=std::min(cg,limit),gy=(cg+gx-1)/gx;
      data.AddOutput({y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
          .AddUniformVariables({capspan,uint32_t(capacity_),channels,minimum,maximum,modulus,remainder,uint32_t(stage_),channels,0u,
                                uint32_t(first),size,identity});
      ORT_RETURN_IF_ERROR(m0_residual_stages::RunHeavyProgram(c,data,cg*64*source_channels*2));
      ++upsample_dispatches;
    }
    LOGS(c.Logger(),INFO)<<"M0_RW07_GENERATOR_EDGE_V1 mode="<<mode_<<" stage="<<stage_<<" role="<<role
      <<" session="<<session<<" run="<<run<<" GPU_dispatches="<<upsample_dispatches
      <<" max_upsample_weight_products_per_dispatch="<<m0_residual_stages::kResidualProductsPerDispatchV1<<" host_payload_reads=0";
    return Status::OK();
  }
 private:
  const uint32_t mode_;
  const int64_t stage_,capacity_;
};
class PacketProgram final : public Program<PacketProgram> {
 public:
  PacketProgram() : Program{"RW07GeneratorProjectionTerminalPacketV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"minimum", ProgramUniformVariableDataType::Uint32},
      {"maximum", ProgramUniformVariableDataType::Uint32},
      {"modulus", ProgramUniformVariableDataType::Uint32},
      {"remainder", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& p = s.AddOutput("packet", ShaderUsage::UseValueTypeAlias);
    ValidOutput(s, e);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<18u;i++){" << p.SetByOffset("i", "0u") << "}"
      << p.SetByOffset("0u", "103u")
      << "if(" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=3u){return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.identity", "i", 10) << "){return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){return;}" << p.SetByOffset("0u", "status") << "return;}"
      << "if(!valid_output()){return;}" << p.SetByOffset("0u", "0u") << p.SetByOffset("1u", "3u")
      << "for(var i=0u;i<6u;i++){" << p.SetByOffset("2u+i", e.GetByOffset("2u+i")) << "}";
    return Status::OK();
  }
};
class Packet final : public WebGpuKernel {
 public:
  explicit Packet(const OpKernelInfo& i) : WebGpuKernel{i} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e = c.Input<Tensor>(0); const auto* y = c.Input<Tensor>(1);
    const uint32_t channels = 22u;
    ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          Gpu(y) && y->IsDataType<float>() && y->Shape().NumDimensions() == 3 &&
                          y->Shape()[0] == 1 && y->Shape()[1] == channels && y->Shape()[2] > 0 && y->Shape()[2] <= 12288001,
                      "RW07 generator projection packet physical descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*e, *y, pair) && pair.session == session &&
                          pair.run == run && pair.producer_role == 74u && pair.producer_instance == 1075u,
                      "RW07 generator projection terminal actual pair");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 generator projection packet resources");
    // Carrier ownership is required, but this shader binds only e and p.
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c,y));
    auto* p = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p) && p->IsDataType<uint32_t>() && p->DataRaw() != e->DataRaw() && p->DataRaw() != y->DataRaw(),
                      "RW07 generator projection distinct packet output");
    ORT_RETURN_IF_ERROR(StorageBudget(c,{e,p}));
    const m0_extent_abi2::OpaqueOwnership owner{session, run, pair.generation, pair.data_generation, 74u, 1075u};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 generator projection packet opaque ownership");
    const auto words = owner.UniformWords();
    PacketProgram program;
    program.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({p, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[2]), uint32_t(y->Shape()[2]), channels,
                              121u, 12288001u,
                              120u, 1u, gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p), "RW07 generator projection packet map fault registration");
    return Status::OK();
  }
};
}  // namespace
KernelCreateInfo CreateM0FullGpuGeneratorUpsampleExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<GeneratorEdge>(i,0);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorUpsampleExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorJoinExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<GeneratorEdge>(i,1);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorJoinExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorMergeExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<GeneratorEdge>(i,2);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorMergeExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorProjectionExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<GeneratorEdge>(i,3);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorProjectionExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuGeneratorProjectionTerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<Packet>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("GeneratorProjectionTerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
      .Provider(kWebGpuExecutionProvider).Build(),fn};
}
}  // namespace onnxruntime::webgpu
