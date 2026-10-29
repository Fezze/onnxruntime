// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original noise_res.0/.1 three-block AdaIN/Snake residual paths; one exact block per op.
#include <algorithm>
#include <array>
#include <optional>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"

namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
using m0_residual_stages::ValidOutput;
using m0_residual_stages::PhysicalBuffer;
using m0_residual_stages::StorageBudget;
using m0_residual_stages::StorageBudgetWithWindow;
using m0_residual_stages::StorageBudgetWithWindows;
using m0_residual_stages::GeneratorStyleParameters;
using m0_residual_stages::GeneratorNormAffine;
using m0_residual_stages::GeneratorSnakeInPlace;
using m0_residual_stages::GeneratorNormStatistics;
using m0_residual_stages::GeneratorResidualConv;
using m0_residual_stages::GeneratorResidualAdd;
class Guard final : public Program<Guard> {
 public:
  Guard() : Program{"RW07NoiseResidualAfter1600GuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"input_capacity", ProgramUniformVariableDataType::Uint32},
      {"output_capacity", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"minimum", ProgramUniformVariableDataType::Uint32},
      {"maximum", ProgramUniformVariableDataType::Uint32},
      {"modulus", ProgramUniformVariableDataType::Uint32},
      {"remainder", ProgramUniformVariableDataType::Uint32},
      {"source_pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& o=s.AddOutput("extent",ShaderUsage::UseValueTypeAlias);
    auto& f=s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << o.SetByOffset("i","0u") << "}"
      << o.SetByOffset("0u","2u") << o.SetByOffset("1u","3u")
      << "for(var i=0u;i<10u;i++){" << o.SetByOffset("22u+i",GetElementAt("uniforms.identity","i",10)) << "}"
      << "if(uniforms.source_pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||"
      << e.GetByOffset("1u") << "!=3u){" << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity","i",10) << "){" << o.SetByOffset("20u","103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}}if(" << e.GetByOffset("21u") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}" << o.SetByOffset("20u","status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("4u") << "!=uniforms.channels||" << e.GetByOffset("5u") << "!=0u||"
      << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << o.SetByOffset("20u","103u") << "return;}"
      << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << o.SetByOffset("20u","103u") << "return;}}let n=" << e.GetByOffset("6u") << ";"
      << "if(n<uniforms.minimum||n>uniforms.maximum||n%uniforms.modulus!=uniforms.remainder||" << e.GetByOffset("18u") << "!=uniforms.channels*n){"
      << o.SetByOffset("20u","103u") << "return;}"
      << "let count=n;"
      << "let channels=uniforms.channels;"
      << "if(n>uniforms.input_capacity||count>uniforms.output_capacity){"
      << o.SetByOffset("20u","102u") << "return;}"
      << o.SetByOffset("2u","1u") << o.SetByOffset("4u","channels") << o.SetByOffset("6u","count")
      << o.SetByOffset("18u","channels*count") << o.SetByOffset("21u","63u");
    return Status::OK();
  }
};
class NoiseResidualExtent final : public WebGpuKernel {
 public:
  explicit NoiseResidualExtent(const OpKernelInfo& i) : WebGpuKernel{i},
      capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity",0)),
      branch_(i.GetAttrOrDefault<int64_t>("branch",-1)), block_(i.GetAttrOrDefault<int64_t>("block",-1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    ORT_RETURN_IF_NOT((branch_==0||branch_==1)&&block_>=0&&block_<3&&capacity_>0&&capacity_<=12288001,
                      "RW07 noise residual closed branch/block/capacity");
    const uint32_t channels=branch_==0?256u:128u,kernel=branch_==0?7u:11u;
    const auto* x=c.Input<Tensor>(0);const auto* e=c.Input<Tensor>(1);const auto* style=c.Input<Tensor>(2);
    ORT_RETURN_IF_NOT(Gpu(x)&&x->IsDataType<float>()&&x->Shape().NumDimensions()==3&&
                          x->Shape()[0]==1&&x->Shape()[1]==channels&&x->Shape()[2]>0&&x->Shape()[2]<=12288001&&
                          Gpu(e)&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape{32}&&
                          Gpu(style)&&style->IsDataType<float>()&&style->Shape()==TensorShape({1,256}),
                      "RW07 noise residual carrier/extent/style descriptors");
    std::array<const Tensor*,12> params{};
    for(size_t i=0;i<params.size();i++){params[i]=c.Input<Tensor>(int(i)+3);}
    for(size_t h=0;h<2;h++){
      const size_t p=6*h;
      const std::array<TensorShape,6> shapes{TensorShape({2*channels,128}),TensorShape({2*channels}),
          TensorShape({1,channels,1}),TensorShape({1,channels,1}),TensorShape({channels,channels,kernel}),TensorShape({channels})};
      for(size_t k=0;k<6;k++){
        ORT_RETURN_IF_NOT(Gpu(params[p+k])&&params[p+k]->IsDataType<float>()&&params[p+k]->Shape()==shapes[k],
                          "RW07 noise residual exact immutable parameters");
      }
    }
    const auto& limits=c.DeviceLimits();
    const uint64_t limit=limits.maxComputeWorkgroupsPerDimension, bytes=uint64_t(channels)*capacity_*4;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=7&&limits.maxComputeWorkgroupSizeX>=64&&
                          limits.maxComputeInvocationsPerWorkgroup>=64&&limit>=4&&
                          bytes<=limits.maxBufferSize,
                      "RW07 noise residual resource and dispatch capacity");
    auto binding=[&](const Tensor* t)->Status{return PhysicalBuffer(c,t);};
    for(const auto* t:{x,e,style}){ORT_RETURN_IF_ERROR(binding(t));}
    for(const auto* t:params){ORT_RETURN_IF_ERROR(binding(t));}
    const uint32_t pc=uint32_t(x->Shape()[2]),oc=uint32_t(capacity_);
    uint32_t tile_channels=0;
    ORT_RETURN_IF_ERROR(m0_residual_stages::ResidualTileChannels(c,pc,oc,channels,tile_channels,true));
    // The tile planner validated both complete rows and the worst alignment prefix.
    const uint64_t prefix=limits.minStorageBufferOffsetAlignment-sizeof(float);
    const uint64_t row_bytes=uint64_t(std::max(pc,oc))*sizeof(float);
    const uint32_t output_tile_channels=static_cast<uint32_t>(std::min<uint64_t>(channels,
        (limits.maxStorageBufferBindingSize-prefix)/row_bytes));
    const uint64_t groups=(uint64_t(output_tile_channels)*oc+63)/64;
    ORT_RETURN_IF_NOT(output_tile_channels>0&&groups<=limit*limit,"RW07 residual output window dispatch limit");
    const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity source,carrier,pair,output;
    ORT_RETURN_IF_NOT(session&&run&&c.M0FullGpuDescribeResourceV2(*e,source)&&c.M0FullGpuDescribeResourceV2(*x,carrier),
                      "RW07 noise residual actual resources");
    const uint32_t role=uint32_t(40+2*block_+branch_),instance=role+1001;
    const uint32_t source_role=block_==0?uint32_t(16+branch_):role-2;
    const uint32_t source_instance=source_role+1001;
    const bool paired=c.M0FullGpuDescribePairV2(*e,*x,pair)&&pair.session==session&&pair.run==run&&
                      pair.generation==source.generation&&pair.data_generation==carrier.generation&&
                      pair.producer_role==source_role&&pair.producer_instance==source_instance;
    auto* y=c.Output(0,TensorShape{1,channels,capacity_});auto* out=c.Output(1,TensorShape{32});
    ORT_RETURN_IF_NOT(Gpu(y)&&y->IsDataType<float>()&&Gpu(out)&&out->IsDataType<uint32_t>(),"RW07 noise residual output types");
    auto activated=c.CreateGPUTensor(x->DataType(),TensorShape{1,tile_channels,capacity_});
    auto conv=c.CreateGPUTensor(x->DataType(),TensorShape{1,channels,capacity_});
    auto coefficients=c.CreateGPUTensor(x->DataType(),TensorShape{channels,6});
    std::optional<Tensor> packed_columns,packed_flags,register_weights,active_grid;
    const uint64_t packed_elements=m0_residual_stages::ResidualPackedElementsV1(tile_channels,kernel);
    if(M0OutputChannelTiledConvV1()&&m0_residual_stages::ResidualPackedSupportedV1(limits)&&
        packed_elements*4<=limits.maxStorageBufferBindingSize&&packed_elements*4<=limits.maxBufferSize){
      active_grid.emplace(c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{3}));
      ORT_RETURN_IF_ERROR(binding(&*active_grid));
      packed_columns.emplace(c.CreateGPUTensor(x->DataType(),TensorShape{int64_t(packed_elements)}));
      packed_flags.emplace(c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{int64_t(m0_residual_stages::ResidualPackedFlagCapacityV1(packed_elements))}));
      ORT_RETURN_IF_ERROR(binding(&*packed_columns));ORT_RETURN_IF_ERROR(binding(&*packed_flags));
      ORT_RETURN_IF_NOT(packed_columns->DataRaw()!=packed_flags->DataRaw(),"RW07 packed scratch alias");
      for(const Tensor* t:std::array<const Tensor*,8>{x,e,style,y,out,&activated,&conv,&coefficients}){
        ORT_RETURN_IF_NOT(t->DataRaw()!=packed_columns->DataRaw()&&t->DataRaw()!=packed_flags->DataRaw(),"RW07 packed source/output alias");
      }
      for(const auto* t:params)ORT_RETURN_IF_NOT(t->DataRaw()!=packed_columns->DataRaw()&&t->DataRaw()!=packed_flags->DataRaw(),"RW07 packed parameter alias");
      const uint64_t weight_elements=m0_residual_stages::ResidualRegisterWeightElementsV1(channels,tile_channels,kernel);
      if(M0BatchHeavyProgramsV1()&&weight_elements&&weight_elements*4<=limits.maxStorageBufferBindingSize&&weight_elements*4<=limits.maxBufferSize){
        register_weights.emplace(c.CreateGPUTensor(x->DataType(),TensorShape{int64_t(weight_elements)}));
        ORT_RETURN_IF_ERROR(binding(&*register_weights));
        for(const Tensor* t:std::array<const Tensor*,10>{x,e,style,y,out,&activated,&conv,&coefficients,&*packed_columns,&*packed_flags})
          ORT_RETURN_IF_NOT(t->DataRaw()!=register_weights->DataRaw(),"RW07 register weight scratch/source alias");
        for(const auto* t:params)ORT_RETURN_IF_NOT(t->DataRaw()!=register_weights->DataRaw(),"RW07 register weight parameter alias");
      }
    }
    std::array<const Tensor*,5> outputs{y,out,&activated,&conv,&coefficients};
    for(size_t i=0;i<outputs.size();i++){
      ORT_RETURN_IF_ERROR(binding(outputs[i]));
      for(size_t j=0;j<i;j++){ORT_RETURN_IF_NOT(outputs[i]->DataRaw()!=outputs[j]->DataRaw(),"RW07 noise residual scratch/output alias");}
      for(const auto* t:{x,e,style}){ORT_RETURN_IF_NOT(outputs[i]->DataRaw()!=t->DataRaw(),"RW07 noise residual source/output alias");}
      for(const auto* t:params){ORT_RETURN_IF_NOT(outputs[i]->DataRaw()!=t->DataRaw(),"RW07 noise residual parameter/output alias");}
    }
    const auto window=[&](const Tensor* input,uint32_t offset,uint32_t count){
      const uint32_t capacity=uint32_t(input->Shape()[2]);
      return ProgramInput::FloatBufferWindow(input,ProgramTensorMetadataDependency::None,
          TensorShape{1,count,capacity},offset*capacity,limits.minStorageBufferOffsetAlignment);
    };
    const auto output_window=[&](Tensor* output,uint32_t offset,uint32_t count){
      return ProgramOutput::FloatBufferWindow(output,ProgramTensorMetadataDependency::None,
          TensorShape{1,count,oc},offset*oc,limits.minStorageBufferOffsetAlignment);
    };
    // Admit every actual window and dispatch before publishing or dispatching.
    ORT_RETURN_IF_ERROR(StorageBudget(c,{e,out}));
    for(uint32_t offset=0;offset<channels;offset+=output_tile_channels){
      const uint32_t count=std::min(output_tile_channels,channels-offset);
      ORT_RETURN_IF_ERROR(StorageBudgetWithWindows(c,window(x,offset,count),output_window(y,offset,count),{out}));
    }
    for(size_t h=0;h<2;h++){
      ORT_RETURN_IF_ERROR(StorageBudget(c,{out,style,params[6*h],params[6*h+1],
          params[6*h+2],params[6*h+3],&coefficients}));
      const auto* input=h==0?x:&conv;
      auto* accumulator=h==0?&conv:y;
      for(uint32_t offset=0;offset<channels;offset+=tile_channels){
        const uint32_t count=std::min(tile_channels,channels-offset);
        const auto view=window(input,offset,count);
        ORT_RETURN_IF_ERROR(StorageBudgetWithWindow(c,view,{out,&coefficients,&activated}));
      }
      for(uint32_t offset=0;offset<channels;offset+=output_tile_channels){
        const uint32_t count=std::min(output_tile_channels,channels-offset);
        ORT_RETURN_IF_ERROR(StorageBudgetWithWindow(c,output_window(accumulator,offset,count),
            {&activated,out,params[6*h+4],params[6*h+5]}));
        if(packed_columns)ORT_RETURN_IF_ERROR(StorageBudgetWithWindow(c,output_window(accumulator,offset,count),
            {&activated,out,params[6*h+4],params[6*h+5],&*packed_columns,&*packed_flags}));
        if(register_weights)ORT_RETURN_IF_ERROR(StorageBudgetWithWindow(c,output_window(accumulator,offset,count),
            {&activated,out,&*register_weights,params[6*h+5],&*packed_columns,&*packed_flags}));
      }
    }
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out,*y,role,instance,output),"RW07 noise residual output claim");
    const m0_extent_abi2::OpaqueOwnership src{session,run,source.generation,carrier.generation,source_role,source_instance};
    const m0_extent_abi2::OpaqueOwnership dst{session,run,output.generation,output.data_generation,role,instance};
    ORT_RETURN_IF_NOT(src.Valid()&&dst.Valid(),"RW07 noise residual opaque ownership");
    const auto sw=src.UniformWords(),ow=dst.UniformWords();
    const auto source_identity=gsl::span<const uint32_t>{sw.data(),sw.size()},identity=gsl::span<const uint32_t>{ow.data(),ow.size()};
    const uint32_t minimum=branch_==0?20u:121u, maximum=branch_==0?2048000u:12288001u;
    const uint32_t modulus=branch_==0?20u:120u,remainder=branch_==0?0u:1u;
    Guard guard;guard.AddInput({e,ProgramTensorMetadataDependency::None}).AddOutput({out,ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({pc,oc,channels,minimum,maximum,modulus,remainder,uint32_t(paired),source_identity,identity});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    uint64_t dispatches=1;
    const auto dispatch_window=[&](auto& program,uint32_t count)->Status{
      const uint64_t window_groups=(uint64_t(count)*oc+63)/64;
      const uint64_t gx=std::min(window_groups,limit),gy=(window_groups+gx-1)/gx;
      program.SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy));
      ORT_RETURN_IF_ERROR(c.RunProgram(program));
      ++dispatches;
      return Status::OK();
    };
    for(uint32_t h=0;h<2;h++){
      const auto* input=h==0?x:&conv;auto* normout=&activated;auto* convout=h==0?&conv:y;
      const uint32_t inputcap=h==0?pc:oc, dilation=h==0?uint32_t(1+2*block_):1u;
      GeneratorStyleParameters prepare;
      prepare.AddInputs({{out,ProgramTensorMetadataDependency::None},{style,ProgramTensorMetadataDependency::None},
          {params[6*h],ProgramTensorMetadataDependency::None},{params[6*h+1],ProgramTensorMetadataDependency::None},
          {params[6*h+2],ProgramTensorMetadataDependency::None},{params[6*h+3],ProgramTensorMetadataDependency::None}})
          .AddOutput({&coefficients,ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(64).SetDispatchGroupSize((channels+63)/64)
          .AddUniformVariables({inputcap,oc,channels,minimum,maximum,modulus,remainder,kernel,dilation,pc,identity});
      ORT_RETURN_IF_ERROR(c.RunProgram(prepare));
      ++dispatches;
      for(uint32_t offset=0;offset<channels;offset+=tile_channels){
        const uint32_t count=std::min(tile_channels,channels-offset);
        GeneratorNormStatistics statistics;
        ORT_RETURN_IF_NOT(statistics.Workgroups(count)<=limit,"RW07 normalization workgroup limit");
        statistics.AddInput(window(input,offset,count)).AddInput({out,ProgramTensorMetadataDependency::None})
            .AddOutput({&coefficients,ProgramTensorMetadataDependency::None})
            .SetWorkgroupSize(64).SetDispatchGroupSize(statistics.Workgroups(count))
            .AddUniformVariables({inputcap,oc,channels,minimum,maximum,modulus,remainder,kernel,dilation,pc,
                                  offset,count,tile_channels,identity});
        ORT_RETURN_IF_ERROR(c.RunProgram(statistics));
        ++dispatches;
        const uint64_t norm_elements=uint64_t(tile_channels)*oc;
        const uint32_t norm_chunk=m0_residual_stages::ResidualSnakeElementsPerDispatchV1();
        for(uint64_t first=0;first<norm_elements;first+=norm_chunk){
          const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(norm_chunk,norm_elements-first));
          GeneratorNormAffine norm;
          norm.AddInput(window(input,offset,count)).AddInputs({{out,ProgramTensorMetadataDependency::None},
              {&coefficients,ProgramTensorMetadataDependency::None}}).AddOutput({normout,ProgramTensorMetadataDependency::None})
              .AddUniformVariables({inputcap,oc,channels,minimum,maximum,modulus,remainder,kernel,dilation,pc,
                                    offset,count,tile_channels,uint32_t(first),size,identity});
          const uint64_t norm_groups=(uint64_t(size)+63)/64;
          const uint64_t nx=std::min(norm_groups,limit),ny=(norm_groups+nx-1)/nx;
          norm.SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(nx),uint32_t(ny));
          ORT_RETURN_IF_ERROR(m0_residual_stages::RunHeavyProgram(c,norm,norm_groups*64*(m0_residual_stages::kNormAffineWorkPerElementV1+(norm.FusedSnake()?m0_residual_stages::kSnakeWorkPerElementV1:0))));
          ++dispatches;
          if(!norm.FusedSnake()){
            GeneratorSnakeInPlace snake;
            snake.AddInputs({{out,ProgramTensorMetadataDependency::None},{&coefficients,ProgramTensorMetadataDependency::None}})
                .AddOutput({normout,ProgramTensorMetadataDependency::None})
                .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(nx),uint32_t(ny))
                .AddUniformVariables({inputcap,oc,channels,minimum,maximum,modulus,remainder,kernel,dilation,pc,
                                      offset,count,tile_channels,uint32_t(first),size,identity});
            ORT_RETURN_IF_ERROR(m0_residual_stages::RunHeavyProgram(c,snake,norm_groups*64*m0_residual_stages::kSnakeWorkPerElementV1));
            ++dispatches;
          }
        }
        for(uint32_t output_offset=0;output_offset<channels;output_offset+=output_tile_channels){
          const uint32_t output_count=std::min(output_tile_channels,channels-output_offset);
          const m0_residual_stages::ResidualConvSpecV1 spec{oc,oc,channels,minimum,maximum,modulus,remainder,kernel,dilation,pc,
            offset,count,output_offset,output_count,identity};
          ORT_RETURN_IF_ERROR(m0_residual_stages::RunResidualConvolutionV1(c,
            ProgramInput{normout,ProgramTensorMetadataDependency::None},output_window(convout,output_offset,output_count),
            out,params[6*h+4],params[6*h+5],packed_columns?&*packed_columns:nullptr,packed_flags?&*packed_flags:nullptr,spec,dispatches,false,0,register_weights?&*register_weights:nullptr,active_grid?&*active_grid:nullptr));
        }
      }
    }
    for(uint32_t offset=0;offset<channels;offset+=output_tile_channels){
      const uint32_t count=std::min(output_tile_channels,channels-offset);
      GeneratorResidualAdd add;
      add.AddInput({out,ProgramTensorMetadataDependency::None}).AddInput(window(x,offset,count))
          .AddOutput(output_window(y,offset,count))
          .AddUniformVariables({oc,oc,channels,minimum,maximum,modulus,remainder,kernel,1u,pc,count,identity});
      ORT_RETURN_IF_ERROR(dispatch_window(add,count));
    }
    LOGS(c.Logger(),INFO)<<"M0_RW07_NOISE_RESIDUAL_V1 session="<<session<<" run="<<run<<" branch="<<branch_
      <<" block="<<block_<<" source_pair_valid="<<paired<<" role="<<role<<" instance="<<instance
      <<" GPU_dispatches="<<dispatches
      <<" max_residual_scalar_products_per_dispatch="<<m0_residual_stages::kResidualProductsPerDispatchV1
      <<" max_snake_elements_per_dispatch="<<m0_residual_stages::ResidualSnakeElementsPerDispatchV1()
      <<" activation_tile_channels="<<tile_channels<<" output_tile_channels="<<output_tile_channels<<" host_payload_reads=0";
    return Status::OK();
  }
 private:
  const int64_t capacity_,branch_,block_;
};
class PacketProgram final : public Program<PacketProgram> {
 public:
  PacketProgram() : Program{"RW07NoiseResidualAfter1600TerminalPacketV1"} {}
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
  explicit Packet(const OpKernelInfo& i) : WebGpuKernel{i}, branch_(i.GetAttrOrDefault<int64_t>("branch", -1)), block_(i.GetAttrOrDefault<int64_t>("block", -1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e = c.Input<Tensor>(0); const auto* y = c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT((branch_ == 0 || branch_ == 1) && block_ >= 0 && block_ < 3, "RW07 noise residual packet branch/block");
    const uint32_t channels = branch_ == 0 ? 256u : 128u;
    ORT_RETURN_IF_NOT(Gpu(e) && e->IsDataType<uint32_t>() && e->Shape() == TensorShape{32} &&
                          Gpu(y) && y->IsDataType<float>() && y->Shape().NumDimensions() == 3 &&
                          y->Shape()[0] == 1 && y->Shape()[1] == channels && y->Shape()[2] > 0 && y->Shape()[2] <= 12288001,
                      "RW07 noise residual after1600 packet physical descriptors");
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session && run && c.M0FullGpuDescribePairV2(*e, *y, pair) && pair.session == session &&
                          pair.run == run && pair.producer_role == uint32_t(40 + 2 * block_ + branch_) && pair.producer_instance == uint32_t(1041 + 2 * block_ + branch_),
                      "RW07 noise residual after1600 terminal actual pair");
    const auto& limits = c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 2 && limits.maxComputeWorkgroupsPerDimension >= 1 &&
                          limits.maxComputeWorkgroupSizeX >= 1 && limits.maxComputeInvocationsPerWorkgroup >= 1,
                      "RW07 noise residual after1600 packet resources");
    // The carrier is metadata here; only the sidecar and packet are bound.
    ORT_RETURN_IF_ERROR(PhysicalBuffer(c,y));
    auto* p = c.Output(0, TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p) && p->IsDataType<uint32_t>() && p->DataRaw() != e->DataRaw() && p->DataRaw() != y->DataRaw(),
                      "RW07 noise residual after1600 distinct packet output");
    ORT_RETURN_IF_ERROR(StorageBudget(c,{e,p}));
    const m0_extent_abi2::OpaqueOwnership owner{session, run, pair.generation, pair.data_generation, uint32_t(40 + 2 * block_ + branch_), uint32_t(1041 + 2 * block_ + branch_)};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 noise residual after1600 packet opaque ownership");
    const auto words = owner.UniformWords();
    PacketProgram program;
    program.AddInput({e, ProgramTensorMetadataDependency::None}).AddOutput({p, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({uint32_t(y->Shape()[2]), uint32_t(y->Shape()[2]), channels,
                              branch_ == 0 ? 20u : 121u, branch_ == 0 ? 2048000u : 12288001u,
                              branch_ == 0 ? 20u : 120u, branch_ == 0 ? 0u : 1u, gsl::span<const uint32_t>{words.data(), words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p), "RW07 noise residual after1600 packet map fault registration");
    return Status::OK();
  }
 private:
  const int64_t branch_, block_;
};
}  // namespace
KernelCreateInfo CreateM0FullGpuNoiseResidualAfter1600ExtentV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<NoiseResidualExtent>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("NoiseResidualAfter1600ExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
KernelCreateInfo CreateM0FullGpuNoiseResidualAfter1600TerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn = [](FuncManager&, const OpKernelInfo& i, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Packet>(i); return Status::OK();
  };
  return {KernelDefBuilder().SetName("NoiseResidualAfter1600TerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1, 1)
              .Provider(kWebGpuExecutionProvider).Build(), fn};
}
}  // namespace onnxruntime::webgpu
