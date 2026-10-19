// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace onnxruntime {
// Only administrative words downloaded after checked queue completion enter here.
// No duration, alignment, extent sidecar or numerical tensor is inspected.
struct M0NativeTerminalResultV1 {
 uint32_t status=103;
 uint64_t samples=0;
};
inline M0NativeTerminalResultV1 M0ParseNativeTerminalV1(
 const std::array<uint32_t,18>& packet,const std::array<uint32_t,2>& tokens,uint64_t physical_samples){
 M0NativeTerminalResultV1 result;
 for(auto token:tokens)if(token!=0&&token!=103)return result;
 const auto status=packet[0];
 if(status!=0&&status!=1&&status!=100&&status!=101&&status!=102&&status!=103)return result;
 if(status){
  for(size_t i=1;i<packet.size();++i)if(packet[i])return result;
 }else{
  if(packet[1]!=1)return result;
  for(size_t i=4;i<packet.size();++i)if(packet[i])return result;
  const uint64_t samples=uint64_t(packet[2])|(uint64_t(packet[3])<<32);
  if(!samples||samples>61440000||samples>physical_samples||samples%600)return result;
  result.samples=samples;
 }
 result.status=(tokens[0]==103||tokens[1]==103)?103:status;
 if(result.status)result.samples=0;
 return result;
}
}  // namespace onnxruntime
