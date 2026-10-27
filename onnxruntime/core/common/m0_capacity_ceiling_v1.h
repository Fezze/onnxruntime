// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <string_view>
namespace onnxruntime {
// Existing compiled graph envelope, also M0NativeMemoryPlanV1::kLogicalCapacity.
// This is a model/compiler bound, never a preferred hardware work size.
inline constexpr uint32_t kM0MaximumDurationCapacityV1=102400;
// Pinned Kokoro generator upsamples120positions per duration frame; the final
// overlap reconstruction emits5PCM samples per position. This is an output
// format/model relation, not a device-dependent tuning parameter.
inline constexpr uint32_t kM0PcmSamplesPerDurationFrameV1=120u*5u;
inline constexpr const char* kM0CapacityCeilingConfigV1="m0.fullgpu.duration_capacity_ceiling_v1";
inline bool M0ParseCapacityCeilingV1(std::string_view text,uint32_t& result)noexcept{
  if(text.empty()||text.front()=='0')return false;
  uint32_t value=0;
  for(char c:text){
    if(c<'0'||c>'9')return false;
    const uint32_t digit=uint32_t(c-'0');
    if(value>(kM0MaximumDurationCapacityV1-digit)/10u)return false;
    value=value*10u+digit;
  }
  if(value==0||value>kM0MaximumDurationCapacityV1)return false;
  result=value;return true;
}
// Feasibility is monotonic in duration for the existing memory/device plan.
// Every candidate is checked against the actual retained device and RAM budget.
template<class Fits>
uint32_t M0LargestFittingCapacityV1(uint32_t ceiling,Fits&& fits){
  if(ceiling==0||ceiling>kM0MaximumDurationCapacityV1||!fits(1))return 0;
  uint32_t low=1,high=ceiling;
  while(low<high){const uint32_t middle=low+(high-low+1)/2;if(fits(middle))low=middle;else high=middle-1;}
  return low;
}
inline bool M0ApprovedOriginalConfigEntryV1(std::string_view key,std::string_view value)noexcept{
  if(key==kM0CapacityCeilingConfigV1){uint32_t ignored=0;return M0ParseCapacityCeilingV1(value,ignored);}
  return value=="1"&&(key=="m0.fullgpu.preinit_admission_v1"||key=="m0.fullgpu.preinit_abi_v1"||
    key=="session.disable_prepacking"||key=="session.disable_aot_function_inlining"||key=="session.disable_cpu_ep_fallback");
}
} // namespace onnxruntime
