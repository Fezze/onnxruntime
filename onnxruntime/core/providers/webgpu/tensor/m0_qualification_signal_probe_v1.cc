// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Isolated production-Program tests, never part of the original model lowering.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
#include <algorithm>
#include <array>
#include "core/common/m0_capacity_ceiling_v1.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_signal_programs_v1.h"
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

class SignalProgramProbe final : public WebGpuKernel {
 public:
  explicit SignalProgramProbe(const OpKernelInfo& info)
      : WebGpuKernel{info}, stage_(info.GetAttrOrDefault<int64_t>("stage",0)),
        reuse_(info.GetAttrOrDefault<int64_t>("reuse_coefficients",1)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    // Every genuine Session has an epoch. Ordinary test sessions have no active
    // strict Run; native initialization/execution has a nonzero Run identity.
    // An unavailable core-query bridge returns epoch0 and is denied too.
    ORT_RETURN_IF_NOT(c.KernelContext().M0FullGpuExtentOwnerEpochV1()!=0&&
                          c.KernelContext().M0FullGpuExtentRunIdV2()==0,
                      "M0_PROGRAM_PROBE_REQUIRES_ISOLATED_SESSION");
    ORT_RETURN_IF_NOT((stage_==1||stage_==2||stage_==3||stage_==5)&&(reuse_==0||reuse_==1),"M0_PROGRAM_PROBE_STAGE");
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
      m0_signal_detail::Signal signal_program(reuse_!=0);
      signal_program.AddInputs({{x,ProgramTensorMetadataDependency::None},{phase,ProgramTensorMetadataDependency::None},
          {e,ProgramTensorMetadataDependency::None},{weight,ProgramTensorMetadataDependency::None},{bias,ProgramTensorMetadataDependency::None}})
          .AddOutput({signal,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(kProductionWorkgroupLanes)
          .SetDispatchGroupSize(dispatch[0],dispatch[1]).AddUniformVariables({capacity,output_capacity,words});
      return c.RunProgram(signal_program);
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
  const int64_t stage_,reuse_;
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
