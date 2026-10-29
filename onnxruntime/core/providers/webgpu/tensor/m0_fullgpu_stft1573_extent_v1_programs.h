// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "m0_fullgpu_stft20_bluestein_constants_v1.h"

namespace onnxruntime::webgpu::m0_stft_detail {
// Bluestein next_pow2(2*20-1), from the original STFT frame length in samples.
inline constexpr uint32_t kOriginalBluesteinFftSamplesV1=64;
inline constexpr uint32_t kStftFrameSamplesV1=20;
inline constexpr uint32_t kStftPositiveFrequencyBinsV1=kStftFrameSamplesV1/2+1;
// A power-of-two lane partition for the11 bins within the retained64-lane
// qualified dispatch envelope; each independent frame owns one partition.
inline constexpr uint32_t kDftLanesPerFrameV1=std::bit_ceil(kStftPositiveFrequencyBinsV1);
inline constexpr uint32_t kDftFramesPerWorkgroupV1=kOriginalBluesteinFftSamplesV1/kDftLanesPerFrameV1;
inline constexpr uint32_t kBluesteinTransformCountV1=2;
inline constexpr uint32_t kBluesteinStagesV1=std::bit_width(kOriginalBluesteinFftSamplesV1)-1;
inline constexpr uint32_t kComplexRealProductsV1=4;
// Each retained bin independently computes the window product and its real
// and imaginary products. Windowing is repeated per bin in this shader.
inline constexpr uint64_t kDftRealProductsPerFrameV1=
    kStftFrameSamplesV1*kStftPositiveFrequencyBinsV1*3;
// Forward error of the window product, rounded root coefficient, complex
// component product and accumulation. gamma(n)=n*u/(1-n*u), u=epsilon/2.
// Divide by the lower bound for the independently accumulated absolute sum.
inline constexpr float kDftUnitRoundoffV1=std::numeric_limits<float>::epsilon()/2;
inline constexpr float DftGammaV1(uint32_t operations){
  return operations*kDftUnitRoundoffV1/(1-operations*kDftUnitRoundoffV1);
}
inline constexpr float kDftPhaseRoundoffBoundV1=
    DftGammaV1(4*kStftFrameSamplesV1)/(1-DftGammaV1(2*kStftFrameSamplesV1));
inline constexpr uint64_t kDftPhaseCheckProductsPerFrameV1=kStftFrameSamplesV1+1;
// Preserve the original native Bluestein arithmetic for exact complex zeros
// and the negative-real phase cut within the derived DFT rounding bound.
// The exported downstream graph
// evaluates atan(im/re), including 0/0, rather than atan2: removing the FFT's
// roundoff can otherwise introduce NaNs into previously finite original audio.
// Both dispatches are admitted at their full worst-case arithmetic cost.
inline constexpr uint64_t kBluesteinRealProductsPerFrameV1=
    kStftFrameSamplesV1*(1+kComplexRealProductsV1)+
    kOriginalBluesteinFftSamplesV1*kComplexRealProductsV1+
    kBluesteinTransformCountV1*kBluesteinStagesV1*kOriginalBluesteinFftSamplesV1*kComplexRealProductsV1+
    kOriginalBluesteinFftSamplesV1*2*kComplexRealProductsV1+
    kStftPositiveFrequencyBinsV1*kComplexRealProductsV1+kDftPhaseCheckProductsPerFrameV1;
inline constexpr uint64_t kStftRealProductsPerFrameV1=
    kDftRealProductsPerFrameV1+kBluesteinRealProductsPerFrameV1;
// Tint's SPIR-V writer rounds each workgroup global to16bytes (std430
// upper-bound accounting, spirv/writer/printer/printer.cc EmitVar).
inline constexpr uint32_t kStftWorkgroupVariableAlignmentBytesV1=16;
inline constexpr uint32_t kOriginalBluesteinWorkgroupBytesV1=
    kOriginalBluesteinFftSamplesV1*2*sizeof(float)+kStftWorkgroupVariableAlignmentBytesV1;
inline const char* OriginalWindow() {
  return "const original_window=array<u32,20>(0u,1019771126u,1036226829u,1045630416u,"
         "1051780163u,1056964608u,1059556830u,1061895308u,1063751134u,1064942648u,"
         "1065353216u,1064942648u,1063751134u,1061895308u,1059556830u,1056964608u,"
         "1051780163u,1045630416u,1036226829u,1019771126u);\n";
}
inline void Roots(ShaderHelper& s) {
    s.AdditionalImplementation()
        << "const cosine=array<f32,20>(1.0,0.9510565162951535,0.8090169943749475,"
           "0.5877852522924731,0.30901699437494745,0.0,-0.30901699437494745,"
           "-0.5877852522924731,-0.8090169943749475,-0.9510565162951535,-1.0,"
           "-0.9510565162951535,-0.8090169943749475,-0.5877852522924731,"
           "-0.30901699437494745,0.0,0.30901699437494745,0.5877852522924731,"
           "0.8090169943749475,0.9510565162951535);\n"
        << "const sine=array<f32,20>(0.0,0.30901699437494745,0.5877852522924731,"
           "0.8090169943749475,0.9510565162951535,1.0,0.9510565162951535,"
           "0.8090169943749475,0.5877852522924731,0.30901699437494745,0.0,"
           "-0.30901699437494745,-0.5877852522924731,-0.8090169943749475,"
           "-0.9510565162951535,-1.0,-0.9510565162951535,-0.8090169943749475,"
           "-0.5877852522924731,-0.30901699437494745);\n";
}
class OwnedStftGuard final : public Program<OwnedStftGuard> {
 public:
  OwnedStftGuard() : Program{"RW07STFT1573OwnedGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"signal_capacity", ProgramUniformVariableDataType::Uint32},
      {"frame_capacity", ProgramUniformVariableDataType::Uint32},
      {"pair_valid", ProgramUniformVariableDataType::Uint32},
      {"source_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e = s.AddInput("signal_extent", ShaderUsage::UseValueTypeAlias);
    const auto& hop = s.AddInput("hop", ShaderUsage::UseValueTypeAlias);
    const auto& window = s.AddInput("window", ShaderUsage::UseValueTypeAlias);
    const auto& length = s.AddInput("frame_length", ShaderUsage::UseValueTypeAlias);
    const auto& out = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << OriginalWindow();
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << out.SetByOffset("i", "0u") << "}"
      << out.SetByOffset("0u", "2u") << out.SetByOffset("1u", "4u")
      << "for(var i=0u;i<10u;i++){" << out.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "if(uniforms.pair_valid==0u||" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=2u){"
      << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
      << GetElementAt("uniforms.source_identity", "i", 10) << "){" << out.SetByOffset("20u", "103u") << "return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
      << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){"
      << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "if(" << e.GetByOffset("21u") << "!=0u){" << out.SetByOffset("20u", "103u") << "return;}"
      << out.SetByOffset("20u", "status") << "return;}"
      << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
      << e.GetByOffset("5u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
      << e.GetByOffset("21u") << "!=63u){" << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=6u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "let n=" << e.GetByOffset("4u") << ";if(n<620u||n>61440020u||(n-20u)%600u!=0u||"
      << e.GetByOffset("18u") << "!=n){" << out.SetByOffset("20u", "103u") << "return;}"
      << "if(" << hop.GetByOffset("0u") << "!=5u||" << hop.GetByOffset("1u") << "!=0u||"
      << length.GetByOffset("0u") << "!=20u||" << length.GetByOffset("1u") << "!=0u){"
      << out.SetByOffset("20u", "103u") << "return;}"
      << "for(var i=0u;i<20u;i++){if(bitcast<u32>(" << window.GetByOffset("i") << ")!=original_window[i]){"
      << out.SetByOffset("20u", "103u") << "return;}}"
      << "let frames=(n-20u)/5u+1u;if(n>uniforms.signal_capacity||frames>uniforms.frame_capacity){"
      << out.SetByOffset("20u", "102u") << "return;}"
      << out.SetByOffset("2u", "1u") << out.SetByOffset("4u", "frames") << out.SetByOffset("6u", "11u")
      << out.SetByOffset("8u", "2u") << out.SetByOffset("18u", "22u*frames") << out.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
inline void ValidSpectrum(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_spectrum() -> bool {if(" << e.GetByOffset("0u") << "!=2u||"
    << e.GetByOffset("1u") << "!=4u||" << e.GetByOffset("20u") << "!=0u||"
    << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("5u") << "!=0u||" << e.GetByOffset("6u") << "!=11u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("8u") << "!=2u||"
    << e.GetByOffset("9u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=10u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let frames=" << e.GetByOffset("4u") << ";return frames>=121u&&(frames-1u)%120u==0u&&frames<=uniforms.frame_capacity&&"
    << e.GetByOffset("18u") << "==22u*frames;}\n";
}
class OwnedStftData final : public Program<OwnedStftData> {
 public:
  OwnedStftData() : Program{"RW07STFT1573RealDft20FourFramesV5"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"signal_capacity", ProgramUniformVariableDataType::Uint32},
      {"frame_capacity", ProgramUniformVariableDataType::Uint32},
      {"first_frame", ProgramUniformVariableDataType::Uint32},
      {"frame_count", ProgramUniformVariableDataType::Uint32},
      {"signal_begin", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("signal", ShaderUsage::UseValueTypeAlias);
    const auto& window = s.AddInput("window", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("spectrum", ShaderUsage::UseValueTypeAlias);
    Roots(s);
    ValidSpectrum(s,e);
    auto& f=s.MainFunctionBody();
    // Four independent real DFT20 frames per64-lane group. Sixteen lanes per
    // frame cover the11 retained bins; no shared storage or FFT barriers.
    f<<"let relative=workgroup_idx*"<<kDftFramesPerWorkgroupV1<<"u+local_idx/"<<kDftLanesPerFrameV1<<"u;let bin=local_idx%"<<kDftLanesPerFrameV1<<"u;"
     <<"if(relative>=uniforms.frame_count||bin>=11u){return;}let frame=uniforms.first_frame+relative;"
     <<y.SetByOffset("22u*relative+2u*bin","0.0")
     <<y.SetByOffset("22u*relative+2u*bin+1u","0.0")
     <<"if(!valid_spectrum()||frame>="<<e.GetByOffset("4u")
     <<"||5u*frame+19u>=uniforms.signal_capacity){return;}"
     <<"var re=0.0;var im=0.0;for(var n=0u;n<20u;n++){"
     <<"let value="<<x.GetByOffset("5u*frame+n-uniforms.signal_begin")<<"*"<<window.GetByOffset("n")<<";"
     <<"let root=(bin*n)%20u;re=re+value*cosine[root];im=im-value*sine[root];}"
     <<y.SetByOffset("22u*relative+2u*bin","re")
     <<y.SetByOffset("22u*relative+2u*bin+1u","im");
    return Status::OK();
  }
};

// Workgroup storage is a per-workgroup limit. Native FP32 windows are bounded
// by actual binding/alignment/grid limits and the existing scheduled-product
// envelope, rather than summing unrelated groups' shared memory into one limit.
class OwnedStftZeroFallbackData final : public Program<OwnedStftZeroFallbackData> {
 public:
  OwnedStftZeroFallbackData() : Program{"RW07STFT1573PhaseBoundaryNativeBluesteinFallbackV7"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"signal_capacity", ProgramUniformVariableDataType::Uint32},
      {"frame_capacity", ProgramUniformVariableDataType::Uint32},
      {"first_frame", ProgramUniformVariableDataType::Uint32},
      {"frame_count", ProgramUniformVariableDataType::Uint32},
      {"signal_begin", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("signal", ShaderUsage::UseValueTypeAlias);
    const auto& window = s.AddInput("window", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("spectrum", ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation()<<OriginalBluesteinCoefficients;
    ValidSpectrum(s,e);
    s.AdditionalImplementation()<<R"WGSL(
var<workgroup> stft_values:array<vec2<f32>,64>;
var<workgroup> stft_active:u32;
fn stft_mul(a:f32,b:f32)->f32{return a*b;}
fn stft_add(a:f32,b:f32)->f32{return a+b;}
fn stft_complex(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
 let rr=stft_mul(a.x,b.x);let ii=stft_mul(a.y,b.y);
 let ri=stft_mul(a.x,b.y);let ir=stft_mul(a.y,b.x);
 return vec2<f32>(stft_add(rr,-ii),stft_add(ri,ir));
}
fn stft_sum(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{return vec2<f32>(stft_add(a.x,b.x),stft_add(a.y,b.y));}
)WGSL";
    auto& f=s.MainFunctionBody();
    // One complete64-point transform per workgroup, one complex lane per
    // invocation. Arithmetic uses the GPU's native FP32 operations.
    f<<"if(workgroup_idx>=uniforms.frame_count){return;}let frame=uniforms.first_frame+workgroup_idx;"
     <<"if(local_idx==0u){stft_active=0u;if(valid_spectrum()&&frame<"<<e.GetByOffset("4u")
     <<"&&5u*frame+19u<uniforms.signal_capacity){var amplitude=0.0;for(var j=0u;j<20u;j++){amplitude+=abs("
     <<x.GetByOffset("5u*frame+j-uniforms.signal_begin")<<"*"<<window.GetByOffset("j")<<");}"
     <<"let phase_error=amplitude*bitcast<f32>("<<std::bit_cast<uint32_t>(kDftPhaseRoundoffBoundV1)<<"u);"
     <<"for(var bin=0u;bin<11u;bin++){let re="
     <<y.GetByOffset("22u*workgroup_idx+2u*bin")<<";let im="<<y.GetByOffset("22u*workgroup_idx+2u*bin+1u")
     <<";if(((bitcast<u32>(re)&0x7fffffffu)==0u&&(bitcast<u32>(im)&0x7fffffffu)==0u)||"
     <<"(re<0.0&&abs(im)<=phase_error)){stft_active=1u;}}}}workgroupBarrier();"
     <<"let frame_enabled=workgroupUniformLoad(&stft_active);if(frame_enabled==0u){return;}"
     <<"let n=reverseBits(local_idx)>>26u;var value=vec2<f32>(0.0);"
     <<"if(n<20u){let windowed=stft_mul("<<x.GetByOffset("5u*frame+n-uniforms.signal_begin")<<","<<window.GetByOffset("n")<<");"
     <<"value=stft_complex(vec2<f32>(windowed,0.0),bitcast<vec2<f32>>(stft_chirp[n]));}"
     <<"stft_values[local_idx]=stft_complex(vec2<f32>(1.0,0.0),value);workgroupBarrier();"
     <<"for(var transform=0u;transform<2u;transform++){"
     <<"for(var size=2u;size<=64u;size*=2u){let fft_midpoint=size/2u;let lane=local_idx%size;"
     <<"let even_index=(local_idx/size)*size+lane%fft_midpoint;"
     <<"let even=stft_values[even_index];let odd=stft_values[even_index+fft_midpoint];"
     <<"let root=bitcast<vec2<f32>>(stft_roots[lane*(64u/size)]);"
     <<"value=stft_sum(even,stft_complex(root,odd));workgroupBarrier();"
     <<"stft_values[local_idx]=value;workgroupBarrier();}"
     <<"if(transform==0u){let index=reverseBits(local_idx)>>26u;"
     <<"value=stft_complex(stft_values[index],bitcast<vec2<f32>>(stft_chirp_spectrum[index]));"
     <<"value=stft_complex(vec2<f32>(1.0,0.0),value);workgroupBarrier();"
     <<"stft_values[local_idx]=value;workgroupBarrier();}}"
     // The inverse uses the forward-root/reversed-bin identity and 1/N scaling.
     <<"if(local_idx<11u){let index=(64u-local_idx)%64u;let source=stft_values[index];"
     <<"let scaled=source/64.0;"
     <<"let result=stft_complex(scaled,bitcast<vec2<f32>>(stft_chirp[local_idx]));"
     <<y.SetByOffset("22u*workgroup_idx+2u*local_idx","result.x")
     <<y.SetByOffset("22u*workgroup_idx+2u*local_idx+1u","result.y")<<"}";
    return Status::OK();
  }
};

inline uint32_t StftFramesPerWindowV1(const wgpu::Limits& limits,uint32_t capacity,
                                    uint32_t requested=std::numeric_limits<uint32_t>::max()) {
  constexpr uint64_t float_bytes=sizeof(float),window_samples=20,hop_samples=5,complex_bins=22;
  const uint64_t alignment=limits.minStorageBufferOffsetAlignment;
  if(!capacity||!requested||!alignment||alignment%float_bytes||
     limits.maxComputeWorkgroupSizeX<kOriginalBluesteinFftSamplesV1||
     limits.maxComputeInvocationsPerWorkgroup<kOriginalBluesteinFftSamplesV1||
     limits.maxComputeWorkgroupStorageSize<kOriginalBluesteinWorkgroupBytesV1||
     !limits.maxComputeWorkgroupsPerDimension)return 0;
  const uint64_t work_frames=kM0ScheduledProductsPerWindowV1/kStftRealProductsPerFrameV1;
  const uint64_t binding=std::min(limits.maxBufferSize,limits.maxStorageBufferBindingSize);
  const uint64_t prefix=alignment-float_bytes;
  if(!work_frames||binding<=prefix||binding-prefix<complex_bins*float_bytes)return 0;
  const uint64_t samples=(binding-prefix)/float_bytes;
  if(samples<window_samples)return 0;
  const uint64_t input_frames=(samples-window_samples)/hop_samples+1;
  const uint64_t output_frames=(binding-prefix)/(complex_bins*float_bytes);
  const uint64_t groups=limits.maxComputeWorkgroupsPerDimension;
  return uint32_t(std::min({uint64_t(capacity),uint64_t(requested),work_frames,input_frames,output_frames,groups*groups}));
}

inline Status RunOwnedStftWindowsV1(ComputeContextBase& context,const Tensor& signal,const Tensor& window,
    const Tensor& extent,Tensor& spectrum,gsl::span<const uint32_t> identity,
    uint32_t requested=std::numeric_limits<uint32_t>::max()) {
  ORT_RETURN_IF_NOT(signal.Location().device.Type()==OrtDevice::GPU&&window.Location().device.Type()==OrtDevice::GPU&&
      extent.Location().device.Type()==OrtDevice::GPU&&spectrum.Location().device.Type()==OrtDevice::GPU&&
      signal.IsDataType<float>()&&signal.Shape().NumDimensions()==2&&signal.Shape()[0]==1&&
      signal.Shape()[1]>=20&&uint64_t(signal.Shape()[1])<=UINT32_MAX&&
      window.IsDataType<float>()&&window.Shape()==TensorShape{20}&&
      extent.IsDataType<uint32_t>()&&extent.Shape()==TensorShape{32}&&
      spectrum.IsDataType<float>()&&spectrum.Shape().NumDimensions()==4&&spectrum.Shape()[0]==1&&
      spectrum.Shape()[1]>0&&uint64_t(spectrum.Shape()[1])<=UINT32_MAX&&
      spectrum.Shape()[2]==11&&spectrum.Shape()[3]==2&&uint64_t(spectrum.Shape().Size())<=UINT32_MAX&&identity.size()==10,
      "RW07_STFT_COHORT_PHYSICAL_SHAPES");
  const auto& limits=context.DeviceLimits();
  const uint32_t signal_capacity=uint32_t(signal.Shape()[1]),frame_capacity=uint32_t(spectrum.Shape()[1]);
  const uint32_t frames=StftFramesPerWindowV1(limits,frame_capacity,requested);
  ORT_RETURN_IF_NOT(frames>0,"RW07_STFT_COHORT_DEVICE_ENVELOPE");
  const uint32_t alignment=limits.minStorageBufferOffsetAlignment;
  for(uint32_t first=0;first<frame_capacity;){
    const uint32_t count=std::min(frames,frame_capacity-first);
    const uint32_t begin=uint32_t(std::min<uint64_t>(5ull*first,uint64_t(signal_capacity)-1));
    const uint32_t samples=uint32_t(std::min<uint64_t>(5ull*(count-1)+20,uint64_t(signal_capacity)-begin));
    const auto configure=[&](auto& program,uint32_t groups){
      const uint32_t dx=std::min(groups,limits.maxComputeWorkgroupsPerDimension),dy=(groups+dx-1)/dx;
      program.AddInput(ProgramInput::FloatBufferWindow(&signal,ProgramTensorMetadataDependency::None,TensorShape{samples},begin,alignment))
          .AddInputs({{&window,ProgramTensorMetadataDependency::None},{&extent,ProgramTensorMetadataDependency::None}})
          .AddOutput(ProgramOutput::FloatBufferWindow(&spectrum,ProgramTensorMetadataDependency::None,TensorShape{count,11,2},22u*first,alignment))
          .SetWorkgroupSize(kOriginalBluesteinFftSamplesV1).SetDispatchGroupSize(dx,dy)
          .AddUniformVariables({signal_capacity,frame_capacity,first,count,begin,identity});
    };
    OwnedStftData program;
    configure(program,(count+kDftFramesPerWorkgroupV1-1)/kDftFramesPerWorkgroupV1);
    ORT_RETURN_IF_ERROR(context.RunProgramWithCheckedCompletion(program,uint64_t(count)*kDftRealProductsPerFrameV1));
    OwnedStftZeroFallbackData fallback;
    configure(fallback,count);
    // The GPU alone decides which frames require the original FFT arithmetic.
    // Completed direct output remains untouched for all other frames.
    ORT_RETURN_IF_ERROR(context.RunProgramWithCheckedCompletion(fallback,uint64_t(count)*kBluesteinRealProductsPerFrameV1));
    first+=count;
  }
  return Status::OK();
}
}  // namespace onnxruntime::webgpu::m0_stft_detail
