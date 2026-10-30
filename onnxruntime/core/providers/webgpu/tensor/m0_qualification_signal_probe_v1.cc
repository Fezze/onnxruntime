// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Isolated production-Program tests, never part of the original model lowering.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
#include <algorithm>
#include <array>
#include "core/common/m0_capacity_ceiling_v1.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_signal_programs_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_implicit_gemm_v1.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_recurrence_programs_v3.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_duration_programs_v5.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_polar_after1573_v1_programs.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_stft1573_extent_v1_programs.h"

namespace onnxruntime::webgpu {
namespace {
// Pinned original signal/STFT geometry, not execution tuning parameters.
constexpr uint32_t kPitchFramesPerDuration=2;
constexpr uint32_t kSamplesPerPitchFrame=kM0PcmSamplesPerDurationFrameV1/kPitchFramesPerDuration;
constexpr uint32_t kOriginalStftWindowSamples=20,kOriginalStftHopSamples=5;
constexpr uint32_t kHarmonicRows=9,kComplexLanes=2;
constexpr uint32_t kSpectralBins=kOriginalStftWindowSamples/2+1,kSpectralChannels=kSpectralBins*kComplexLanes;
constexpr uint32_t kMaximumPitchFrames=kPitchFramesPerDuration*kM0MaximumDurationCapacityV1;
constexpr uint32_t kMaximumSpectralFrames=(kM0PcmSamplesPerDurationFrameV1/kOriginalStftHopSamples)*kM0MaximumDurationCapacityV1+1;
// Match the production Signal/Polar callers' lane geometry for numerical tests.
constexpr uint32_t kProductionWorkgroupLanes=64;
class ProbePoison final : public Program<ProbePoison> {
 public:
  ProbePoison():Program{"M0QualificationOutputCanaryV1"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"count",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& output=s.AddOutput("values",ShaderUsage::UseValueTypeAlias);
    // Existing numerical-fixture canary, exactly representable as FLOAT.
    s.MainFunctionBody()<<"if(global_idx<uniforms.count){"<<output.SetByOffset("global_idx","12345.0")<<"}";
    return Status::OK();
  }
};
// Independent unit-input check: same STFT uniform layout and GPU predicate,
// without shared memory/FFT. It reports only flags/counts for fixed test data.
class ProbeStftInputs final : public Program<ProbeStftInputs> {
 public:
  ProbeStftInputs():Program{"M0QualificationStftInputSummaryV1"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"signal_capacity",ProgramUniformVariableDataType::Uint32},
      {"frame_capacity",ProgramUniformVariableDataType::Uint32},
      {"first_frame",ProgramUniformVariableDataType::Uint32},
      {"frame_count",ProgramUniformVariableDataType::Uint32},
      {"signal_begin",ProgramUniformVariableDataType::Uint32},
      {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("signal",ShaderUsage::UseValueTypeAlias);
    const auto& window=s.AddInput("window",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& out=s.AddOutput("summary",ShaderUsage::UseValueTypeAlias);
    m0_stft_detail::ValidSpectrum(s,e);
    s.AdditionalImplementation()<<m0_stft_detail::OriginalWindow();
    auto& f=s.MainFunctionBody();
    f<<"var nonzero=0u;var finite_data=true;var window_matches=true;"
     <<"for(var i=0u;i<20u;i++){let bits=bitcast<u32>("<<x.GetByOffset("i")<<");"
     <<"nonzero+=select(0u,1u,(bits&0x7fffffffu)>0u);finite_data=finite_data&&(bits&0x7fffffffu)<0x7f800000u;"
     <<"window_matches=window_matches&&bitcast<u32>("<<window.GetByOffset("i")<<")==original_window[i];}"
     <<out.SetByOffset("0u","select(0.0,1.0,valid_spectrum())")
     <<out.SetByOffset("1u","f32(nonzero)")<<out.SetByOffset("2u","select(0.0,1.0,window_matches)")
     <<out.SetByOffset("3u","select(0.0,1.0,finite_data)");
    return Status::OK();
  }
};

// Decode the public integer producer outputs only inside an isolated test.
class ProbeDurationSummaryV1 final : public Program<ProbeDurationSummaryV1> {
 public:
  ProbeDurationSummaryV1():Program{"M0QualificationDurationSummaryV1"}{}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"tokens",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s)const override{
    const auto& duration=s.AddInput("duration",ShaderUsage::UseValueTypeAlias);
    const auto& prefix=s.AddInput("prefix",ShaderUsage::UseValueTypeAlias);
    const auto& total=s.AddInput("total",ShaderUsage::UseValueTypeAlias);
    const auto& extent=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& output=s.AddOutput("summary",ShaderUsage::UseValueTypeAlias);
    s.MainFunctionBody()<<"for(var t=0u;t<uniforms.tokens;t++){"
      <<output.SetByOffset("t","f32("+duration.GetByOffset("2u*t")+")")
      <<output.SetByOffset("uniforms.tokens+t","f32("+prefix.GetByOffset("2u*t")+")")<<"}"
      <<output.SetByOffset("2u*uniforms.tokens","f32("+total.GetByOffset("0u")+")")
      <<output.SetByOffset("2u*uniforms.tokens+1u","f32("+extent.GetByOffset("20u")+")");
    return Status::OK();
  }
};
class SignalProgramProbe final : public WebGpuKernel {
 public:
  explicit SignalProgramProbe(const OpKernelInfo& info)
      : WebGpuKernel{info}, stage_(info.GetAttrOrDefault<int64_t>("stage",0)),
        reuse_(info.GetAttrOrDefault<int64_t>("reuse_coefficients",1)),dilation_(info.GetAttrOrDefault<int64_t>("probe_dilation",1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    // Every genuine Session has an epoch. Ordinary test sessions have no active
    // strict Run; native initialization/execution has a nonzero Run identity.
    // An unavailable core-query bridge returns epoch0 and is denied too.
    ORT_RETURN_IF_NOT(c.KernelContext().M0FullGpuExtentOwnerEpochV1()!=0&&
                          c.KernelContext().M0FullGpuExtentRunIdV2()==0,
                      "M0_PROGRAM_PROBE_REQUIRES_ISOLATED_SESSION");
    ORT_RETURN_IF_NOT((stage_==1||stage_==2||stage_==3||stage_==5||stage_==6||stage_==7||stage_==8||stage_==9||stage_==10)&&(reuse_==0||reuse_==1||(stage_==6&&reuse_>=2&&reuse_<=5)),"M0_PROGRAM_PROBE_STAGE");
    const auto* x=c.Input<Tensor>(0);const auto* e=c.Input<Tensor>(1);
    const auto& limits=c.DeviceLimits();
    const auto storage=[&](const Tensor* tensor){
      if(!tensor||tensor->Location().device.Type()!=OrtDevice::GPU||!tensor->DataRaw()||
          !tensor->SizeInBytes()||tensor->SizeInBytes()>limits.maxBufferSize||
          tensor->SizeInBytes()>limits.maxStorageBufferBindingSize)return false;
      M0FullGpuStorageAllocationV1 allocation;
      return c.M0FullGpuDescribeStorageV1(*tensor,allocation)&&allocation.backing_bytes>=tensor->SizeInBytes()&&
          allocation.backing_bytes<=limits.maxBufferSize&&allocation.backing_bytes<=limits.maxStorageBufferBindingSize;
    };
    ORT_RETURN_IF_NOT(storage(x)&&x->IsDataType<float>()&&storage(e)&&e->IsDataType<uint32_t>()&&
                          e->Shape()==TensorShape{32}&&limits.maxStorageBuffersPerShaderStage>=6&&
                          limits.maxComputeWorkgroupSizeX>=kProductionWorkgroupLanes&&
                          limits.maxComputeInvocationsPerWorkgroup>=kProductionWorkgroupLanes&&
                          limits.maxComputeWorkgroupsPerDimension>=kHarmonicRows&&
                          limits.minStorageBufferOffsetAlignment>=sizeof(float)&&
                          limits.minStorageBufferOffsetAlignment%sizeof(float)==0,"M0_PROGRAM_PROBE_DESCRIPTORS");
    const auto grid=[&](uint64_t elements){
      const uint64_t groups=(elements+kProductionWorkgroupLanes-1)/kProductionWorkgroupLanes;
      const uint64_t dx=std::min<uint64_t>(groups,limits.maxComputeWorkgroupsPerDimension);
      return std::array<uint32_t,2>{uint32_t(dx),uint32_t((groups+dx-1)/dx)};
    };
    const auto poison=[&](Tensor* tensor)->Status{
      ORT_RETURN_IF_NOT(tensor->Shape().Size()>0&&uint64_t(tensor->Shape().Size())<=UINT32_MAX,
                        "M0_PROGRAM_PROBE_CANARY_EXTENT");
      const uint32_t count=uint32_t(tensor->Shape().Size());const auto dispatch=grid(count);
      ORT_RETURN_IF_NOT(dispatch[1]<=limits.maxComputeWorkgroupsPerDimension,"M0_PROGRAM_PROBE_CANARY_GRID");
      ProbePoison program;
      program.AddOutput({tensor,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(kProductionWorkgroupLanes)
          .SetDispatchGroupSize(dispatch[0],dispatch[1]).AddUniformVariables({count});
      return c.RunProgram(program);
    };
    constexpr std::array<uint32_t,10> identity{1,2,3,4,5,6,7,8,9,10};
    const auto words=gsl::span<const uint32_t>{identity.data(),identity.size()};
    if(stage_==8){
      const auto* weight=c.Input<Tensor>(2);
      ORT_RETURN_IF_NOT(x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&x->Shape()[1]==640&&
          x->Shape()[2]>0&&x->Shape()[2]<=102400&&storage(weight)&&weight->Shape()==TensorShape({2,1024,640}),
          "M0_PROJECTION_PROBE_INPUT");
      const uint32_t capacity=uint32_t(x->Shape()[2]);
      auto* output=c.Output(0,TensorShape{capacity,2,1024});c.FillZero(*output);
      Tensor transposed;const Tensor* selected=weight;
      if(reuse_){transposed=c.CreateGPUTensor(weight->DataType(),TensorShape{2,640,1024});
        ORT_RETURN_IF_ERROR(m0_recurrence_programs::RunWeightTranspose(c,weight,&transposed,640));selected=&transposed;}
      bool used_tiled=false;
      ORT_RETURN_IF_ERROR(m0_recurrence_programs::RunTiledInputProjectionV1(c,x,selected,e,output,capacity,capacity,words,reuse_!=0,&used_tiled));
      ORT_RETURN_IF_NOT(used_tiled,"M0_PROJECTION_PROBE_REQUIRES_TILED_PATH");
      return Status::OK();
    }
    if(stage_==6){
      const auto* weight=c.Input<Tensor>(2);const auto* bias=c.Input<Tensor>(3);
      ORT_RETURN_IF_NOT(x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&x->Shape()[1]>0&&x->Shape()[1]<=256&&
          x->Shape()[2]>0&&x->Shape()[2]<=UINT32_MAX&&storage(weight)&&storage(bias)&&
          weight->Shape().NumDimensions()==3,"M0_RESIDUAL_PROBE_INPUT");
      const uint32_t channels=uint32_t(x->Shape()[1]),capacity=uint32_t(x->Shape()[2]),kernel=uint32_t(weight->Shape()[2]);
      auto* output=c.Output(0,x->Shape());c.FillZero(*output);
      uint32_t planned_channels=0;
      ORT_RETURN_IF_ERROR(m0_residual_stages::ResidualTileChannels(c,capacity,capacity,channels,planned_channels,true));
      ORT_RETURN_IF_NOT(planned_channels==channels,"M0_RESIDUAL_PROBE_FULL_ADMITTED_ACTIVATION_TILE");
      if(reuse_>=4){
        ORT_RETURN_IF_NOT(dilation_>=1&&dilation_<=5,"M0_IMPLICIT_PROBE_DILATION");
        auto args=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{3});
        uint64_t dispatches=0;const uint32_t tile=reuse_==5?std::max(1u,channels/2):channels;
        const uint32_t output_tile=reuse_==5?channels-1:channels;
        for(uint32_t offset=0;offset<channels;offset+=tile)for(uint32_t out=0;out<channels;out+=output_tile){
          const uint32_t count=std::min(tile,channels-offset),rows=std::min(output_tile,channels-out);
          const m0_residual_stages::ResidualConvSpecV1 spec{capacity,capacity,channels,1u,capacity,1u,0u,kernel,uint32_t(dilation_),capacity,
              offset,count,out,rows,words};
          const auto input=ProgramInput::FloatBufferWindow(x,ProgramTensorMetadataDependency::None,TensorShape{1,count,capacity},offset*capacity,limits.minStorageBufferOffsetAlignment);
          const auto skip=ProgramInput::FloatBufferWindow(x,ProgramTensorMetadataDependency::None,TensorShape{1,rows,capacity},out*capacity,limits.minStorageBufferOffsetAlignment);
          const auto result=ProgramOutput::FloatBufferWindow(output,ProgramTensorMetadataDependency::None,TensorShape{1,rows,capacity},out*capacity,limits.minStorageBufferOffsetAlignment);
          ORT_RETURN_IF_ERROR(m0_implicit_gemm::Run(c,input,result,e,weight,bias,&args,spec,dispatches,
              reuse_==5?&skip:nullptr,0,0,0,offset+count==channels&&out+rows==channels));
        }
        return Status::OK();
      }
      const uint64_t elements=m0_residual_stages::ResidualPackedElementsV1(planned_channels,kernel);
      auto columns=c.CreateGPUTensor(x->DataType(),TensorShape{int64_t(elements)});
      auto flags=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{int64_t(m0_residual_stages::ResidualPackedFlagCapacityV1(elements))});
      auto weights=c.CreateGPUTensor(x->DataType(),TensorShape{int64_t(m0_residual_stages::ResidualRegisterWeightElementsV1(channels,channels,kernel))});
      auto args=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{3});
      const m0_residual_stages::ResidualConvSpecV1 spec{capacity,capacity,channels,1u,capacity,1u,0u,kernel,1u,capacity,
          0u,channels,0u,channels,words};
      uint64_t dispatches=0;
      return m0_residual_stages::RunResidualConvolutionV1(c,{x,ProgramTensorMetadataDependency::None},
          {output,ProgramTensorMetadataDependency::None},e,weight,bias,&columns,&flags,spec,dispatches,true,reuse_==2?128u:reuse_==3?0u:reuse_?64u:32u,&weights,&args);
    }
    if(stage_==9){
      const auto* coefficients=c.Input<Tensor>(2);
      ORT_RETURN_IF_NOT(x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&storage(coefficients)&&
          coefficients->Shape()==TensorShape({x->Shape()[1],6}),"M0_ACTIVE_NORM_PROBE_INPUT");
      const uint32_t channels=uint32_t(x->Shape()[1]),capacity=uint32_t(x->Shape()[2]);
      auto* y=c.Output(0,x->Shape());c.FillZero(*y);
      auto args=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{3});
      const uint32_t work=m0_residual_stages::kNormAffineWorkPerElementV1+m0_residual_stages::kSnakeWorkPerElementV1;
      const uint64_t physical=uint64_t(channels)*capacity;
      const uint32_t chunk=reuse_?uint32_t((uint64_t(capacity)+63)/64*64):
          uint32_t(std::min<uint64_t>(kM0ScheduledProductsPerWindowV1/work,uint64_t(limits.maxComputeWorkgroupsPerDimension)*64)/64*64);
      for(uint64_t first=0;first<physical;first+=chunk){
        const uint32_t count=uint32_t(std::min<uint64_t>(chunk,physical-first));
        const m0_residual_stages::ResidualConvSpecV1 spec{capacity,capacity,channels,1u,capacity,1u,0u,3u,1u,capacity,0u,channels,0u,channels,words};
        m0_residual_stages::ResidualPointwiseGridV2 grid;
        grid.AddInput({e,ProgramTensorMetadataDependency::None}).AddOutput({&args,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(1);
        spec.Apply(grid,0,0,uint32_t(first),count);ORT_RETURN_IF_ERROR(c.RunProgram(grid));
        m0_residual_stages::GeneratorNormAffine norm(true,true);
        norm.AddInputs({{x,ProgramTensorMetadataDependency::None},{e,ProgramTensorMetadataDependency::None},{coefficients,ProgramTensorMetadataDependency::None}})
            .AddOutput({y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetIndirectDispatchTensor(&args)
            .AddUniformVariables({capacity,capacity,channels,1u,capacity,1u,0u,3u,1u,capacity,0u,channels,channels,uint32_t(first),count,words});
        ORT_RETURN_IF_ERROR(c.RunProgram(norm,uint64_t((count+63)/64)*64*work));
      }
      return Status::OK();
    }
    if(stage_==10){
      const auto* weight=c.Input<Tensor>(2);const auto* bias=c.Input<Tensor>(3);
      ORT_RETURN_IF_NOT(x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&storage(weight)&&storage(bias)&&
          weight->Shape().NumDimensions()==3&&weight->Shape()[0]==x->Shape()[1],"M0_IMPLICIT_UPSAMPLE_PROBE_INPUT");
      const uint32_t stage=uint32_t(reuse_),stride=stage?6u:10u,source=uint32_t(x->Shape()[1]);
      const uint32_t channels=uint32_t(weight->Shape()[1]),pc=uint32_t(x->Shape()[2]),oc=pc*stride+stage;
      auto* y=c.Output(0,TensorShape{1,channels,oc});c.FillZero(*y);
      auto transposed=c.CreateGPUTensor(weight->DataType(),TensorShape{channels,weight->Shape()[2],source});
      const std::array<size_t,3> permutation{1,2,0};ORT_RETURN_IF_ERROR(Transpose::DoTranspose(c,permutation,*weight,transposed));
      auto args=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{3});
      const m0_residual_stages::ResidualConvSpecV1 spec{pc,oc,channels,1u,oc,1u,0u,uint32_t(weight->Shape()[2]),1u,pc,0u,source,0u,channels,words};
      uint64_t dispatches=0;return m0_implicit_gemm::Run(c,{x,ProgramTensorMetadataDependency::None},{y,ProgramTensorMetadataDependency::None},
          e,&transposed,bias,&args,spec,dispatches,nullptr,1u,stage,source,true);
    }
    if(stage_==7){
      const auto* speed=c.Input<Tensor>(2);
      ORT_RETURN_IF_NOT(x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&x->Shape()[1]>0&&x->Shape()[1]<=512&&
          x->Shape()[2]==50&&storage(speed)&&speed->Shape().Size()==1,"M0_DURATION_PROBE_INPUT");
      const uint32_t tokens=uint32_t(x->Shape()[1]),capacity=reuse_?tokens*12-1:tokens*50;
      auto values=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{tokens,2});
      auto durations=c.CreateGPUTensor(DataTypeImpl::GetType<int64_t>(),TensorShape{tokens});
      auto prefixes=c.CreateGPUTensor(DataTypeImpl::GetType<int64_t>(),TensorShape{tokens});
      auto total=c.CreateGPUTensor(DataTypeImpl::GetType<int64_t>(),TensorShape{});
      auto extent=c.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(),TensorShape{32});
      m0_duration_detail::DurationRowsProgram rows;
      rows.AddInputs({{x,ProgramTensorMetadataDependency::None},{speed,ProgramTensorMetadataDependency::None}})
        .AddOutput({&values,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(kProductionWorkgroupLanes)
        .SetDispatchGroupSize((tokens+kProductionWorkgroupLanes-1)/kProductionWorkgroupLanes).AddUniformVariables({tokens});
      ORT_RETURN_IF_ERROR(c.RunProgram(rows));
      const auto in_words=[](const Tensor* tensor){ProgramInput input(tensor,ProgramTensorMetadataDependency::None,
          TensorShape{int64_t(tensor->SizeInBytes()/sizeof(uint32_t))},1);input.var_type=ProgramVariableDataType::Uint32;return input;};
      const auto out_words=[](Tensor* tensor){ProgramOutput output(tensor,ProgramTensorMetadataDependency::None,
          TensorShape{int64_t(tensor->SizeInBytes()/sizeof(uint32_t))},1);output.var_type=ProgramVariableDataType::Uint32;return output;};
      m0_duration_detail::DurationProgram publish;
      publish.AddInput({&values,ProgramTensorMetadataDependency::None})
        .AddOutputs({out_words(&durations),out_words(&prefixes),out_words(&total),out_words(&extent)})
        .SetWorkgroupSize(1).SetDispatchGroupSize(1)
        .AddUniformVariables({tokens,capacity,1u,0u,1u,0u,1u,0u,1u,0u});
      ORT_RETURN_IF_ERROR(c.RunProgram(publish));
      auto* output=c.Output(0,TensorShape{1,2*tokens+2});
      ProbeDurationSummaryV1 summary;
      summary.AddInputs({in_words(&durations),in_words(&prefixes),in_words(&total),in_words(&extent)})
        .AddOutput({output,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({tokens});
      return c.RunProgram(summary);
    }
    if(stage_==1){
      const auto* weight=c.Input<Tensor>(2);const auto* bias=c.Input<Tensor>(3);
      ORT_RETURN_IF_NOT(x->Shape().NumDimensions()==3&&x->Shape()[0]==1&&x->Shape()[1]==1&&
                            x->Shape()[2]>0&&x->Shape()[2]<=kMaximumPitchFrames&&
                            storage(weight)&&weight->IsDataType<float>()&&weight->Shape()==TensorShape({kHarmonicRows,1})&&
                            storage(bias)&&bias->IsDataType<float>()&&bias->Shape()==TensorShape{1},
                        "M0_PROGRAM_PROBE_SIGNAL_INPUTS");
      const uint32_t capacity=uint32_t(x->Shape()[2]);
      const uint32_t output_capacity=kSamplesPerPitchFrame*capacity+kOriginalStftWindowSamples;
      const uint64_t binding_limit=std::min(limits.maxBufferSize,limits.maxStorageBufferBindingSize);
      const auto dispatch=grid(output_capacity);
      ORT_RETURN_IF_NOT(uint64_t(kHarmonicRows)*capacity*sizeof(float)<=binding_limit&&
                            uint64_t(output_capacity)*sizeof(float)<=binding_limit&&
                            dispatch[1]<=limits.maxComputeWorkgroupsPerDimension,
                        "M0_PROGRAM_PROBE_SIGNAL_RESOURCE_LIMIT");
      auto* phase=c.Output(0,TensorShape{kHarmonicRows,capacity});
      auto* signal=c.Output(1,TensorShape{1,output_capacity});
      ORT_RETURN_IF_NOT(storage(phase)&&storage(signal)&&phase->DataRaw()!=signal->DataRaw()&&
                            phase->DataRaw()!=x->DataRaw()&&signal->DataRaw()!=x->DataRaw(),
                        "M0_PROGRAM_PROBE_SIGNAL_OUTPUTS");
      for(const auto* input:{x,e,weight,bias}){
        ORT_RETURN_IF_NOT(phase->DataRaw()!=input->DataRaw()&&signal->DataRaw()!=input->DataRaw(),
                          "M0_PROGRAM_PROBE_SIGNAL_OUTPUT_ALIAS");
      }
      ORT_RETURN_IF_ERROR(poison(phase));ORT_RETURN_IF_ERROR(poison(signal));
      m0_signal_detail::Prefix prefix;
      prefix.AddInputs({{x,ProgramTensorMetadataDependency::None},{e,ProgramTensorMetadataDependency::None}})
          .AddOutput({phase,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(kHarmonicRows)
          .AddUniformVariables({capacity,output_capacity,words});
      ORT_RETURN_IF_ERROR(c.RunProgram(prefix));
      return m0_signal_detail::RunSignalWindowsV3(c,x,phase,e,weight,bias,signal,capacity,output_capacity,words,reuse_!=0);
    }
    if(stage_==3||stage_==5){
      const auto* window=c.Input<Tensor>(2);
      ORT_RETURN_IF_NOT(x->Shape().NumDimensions()==2&&x->Shape()[0]==1&&
          x->Shape()[1]>kOriginalStftWindowSamples&&
          uint64_t(x->Shape()[1])<=uint64_t(kMaximumPitchFrames)*kSamplesPerPitchFrame+kOriginalStftWindowSamples&&
          (x->Shape()[1]-kOriginalStftWindowSamples)%kOriginalStftHopSamples==0&&
          storage(window)&&window->IsDataType<float>()&&window->Shape()==TensorShape{kOriginalStftWindowSamples}&&
          limits.maxComputeWorkgroupStorageSize>=m0_stft_detail::kOriginalBluesteinWorkgroupBytesV1,
          "M0_PROGRAM_PROBE_STFT_INPUTS");
      const auto samples=uint32_t(x->Shape()[1]);
      const auto frames=(samples-kOriginalStftWindowSamples)/kOriginalStftHopSamples+1;
      ORT_RETURN_IF_NOT(frames<=kMaximumSpectralFrames&&uint64_t(frames)*kSpectralChannels*sizeof(float)<=
          std::min(limits.maxBufferSize,limits.maxStorageBufferBindingSize),"M0_PROGRAM_PROBE_STFT_RESOURCES");
      if(stage_==5){
        auto* summary=c.Output(0,TensorShape{4});
        ORT_RETURN_IF_NOT(storage(summary)&&summary->DataRaw()!=x->DataRaw()&&summary->DataRaw()!=e->DataRaw()&&
            summary->DataRaw()!=window->DataRaw(),"M0_PROGRAM_PROBE_STFT_SUMMARY_OUTPUT");
        ORT_RETURN_IF_ERROR(poison(summary));
        ProbeStftInputs program;
        program.AddInputs({{x,ProgramTensorMetadataDependency::None},{window,ProgramTensorMetadataDependency::None},
            {e,ProgramTensorMetadataDependency::None}}).AddOutput({summary,ProgramTensorMetadataDependency::None})
            .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({samples,frames,0u,frames,0u,words});
        return c.RunProgram(program);
      }
      auto* output=c.Output(0,TensorShape{1,frames,kSpectralBins,kComplexLanes});
      ORT_RETURN_IF_NOT(storage(output)&&output->DataRaw()!=x->DataRaw()&&output->DataRaw()!=e->DataRaw()&&
          output->DataRaw()!=window->DataRaw(),"M0_PROGRAM_PROBE_STFT_OUTPUT");
      ORT_RETURN_IF_ERROR(poison(output));
      return m0_stft_detail::RunOwnedStftWindowsV1(c,*x,*window,*e,*output,words);
    }
    ORT_RETURN_IF_NOT(x->Shape().NumDimensions()==4&&x->Shape()[0]==1&&x->Shape()[1]>0&&
                          x->Shape()[1]<=kMaximumSpectralFrames&&x->Shape()[2]==kSpectralBins&&x->Shape()[3]==kComplexLanes,
                      "M0_PROGRAM_PROBE_POLAR_INPUTS");
    const uint32_t capacity=uint32_t(x->Shape()[1]);
    auto* output=c.Output(0,TensorShape{1,kSpectralChannels,capacity});
    ORT_RETURN_IF_NOT(storage(output)&&output->DataRaw()!=x->DataRaw()&&output->DataRaw()!=e->DataRaw(),
                      "M0_PROGRAM_PROBE_POLAR_OUTPUT");
    ORT_RETURN_IF_ERROR(poison(output));
    const auto dispatch=grid(capacity);
    ORT_RETURN_IF_NOT(dispatch[1]<=limits.maxComputeWorkgroupsPerDimension,"M0_PROGRAM_PROBE_POLAR_GRID");
    for(uint32_t channel=0;channel<kSpectralChannels;++channel){
      m0_polar_detail::Polar program;
      program.AddInputs({{x,ProgramTensorMetadataDependency::None},{e,ProgramTensorMetadataDependency::None}})
          .AddOutput(ProgramOutput::FloatBufferWindow(output,ProgramTensorMetadataDependency::None,TensorShape{capacity},
              channel*capacity,limits.minStorageBufferOffsetAlignment))
          .SetWorkgroupSize(kProductionWorkgroupLanes).SetDispatchGroupSize(dispatch[0],dispatch[1])
          .AddUniformVariables({capacity,capacity,0u,capacity,0u,channel,words});
      ORT_RETURN_IF_ERROR(c.RunProgram(program));
    }
    return Status::OK();
  }
 private:
  const int64_t stage_,reuse_,dilation_;
};
}  // namespace
KernelCreateInfo CreateM0QualificationSignalProgramProbeV1KernelInfo(){
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& info,std::unique_ptr<OpKernel>& output)->Status{
    output=std::make_unique<SignalProgramProbe>(info);return Status::OK();
  };
  return {KernelDefBuilder().SetName("SignalProgramProbeV1").SetDomain("org.localtts.m0.qualification")
              .SinceVersion(1,1).Provider(kWebGpuExecutionProvider).Build(),fn};
}
}  // namespace onnxruntime::webgpu
#endif
