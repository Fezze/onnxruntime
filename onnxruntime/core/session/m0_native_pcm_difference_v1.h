// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace onnxruntime {
static_assert(sizeof(float)==4,"PCM difference requires float32");
// Diagnostics of already-published public PCM only. No tolerance or acceptance
// policy: callers keep their exact comparison and report the actual difference.
struct M0NativePcmDifferenceV1 {
  size_t expected_bytes=0,actual_bytes=0,samples=0,unequal_bits=0;
  size_t nonfinite_expected=0,nonfinite_actual=0;
  size_t first_unequal=std::numeric_limits<size_t>::max();
  uint32_t first_expected_bits=0,first_actual_bits=0,maximum_ulp=0;
  double maximum_absolute=0,root_mean_square=0;
  bool SameBits()const{return expected_bytes==actual_bytes&&unequal_bits==0;}
};
inline M0NativePcmDifferenceV1 M0MeasureNativePcmDifferenceV1(
    const void* expected,size_t expected_bytes,const void* actual,size_t actual_bytes){
  if(expected_bytes%4||actual_bytes%4||(!expected&&expected_bytes)||(!actual&&actual_bytes))
    throw std::invalid_argument("PCM difference requires valid float32 byte extents");
  M0NativePcmDifferenceV1 result;
  result.expected_bytes=expected_bytes;result.actual_bytes=actual_bytes;
  result.samples=std::min(expected_bytes,actual_bytes)/4;
  const auto* left=static_cast<const unsigned char*>(expected);
  const auto* right=static_cast<const unsigned char*>(actual);
  double squares=0;size_t finite_pairs=0;
  const auto ordered=[](uint32_t value){return value&0x80000000u?~value:value|0x80000000u;};
  for(size_t i=0;i<result.samples;++i){
    uint32_t a,b;float x,y;
    std::memcpy(&a,left+i*4,4);std::memcpy(&b,right+i*4,4);
    std::memcpy(&x,&a,4);std::memcpy(&y,&b,4);
    if(a!=b){
      if(!result.unequal_bits){result.first_unequal=i;result.first_expected_bits=a;result.first_actual_bits=b;}
      ++result.unequal_bits;
    }
    const bool finite_x=std::isfinite(x),finite_y=std::isfinite(y);
    result.nonfinite_expected+=!finite_x;result.nonfinite_actual+=!finite_y;
    if(finite_x&&finite_y){
      const double difference=double(x)-double(y);
      result.maximum_absolute=std::max(result.maximum_absolute,std::abs(difference));
      squares+=difference*difference;++finite_pairs;
      const auto aa=ordered(a),bb=ordered(b);
      result.maximum_ulp=std::max(result.maximum_ulp,aa>bb?aa-bb:bb-aa);
    }
  }
  if(finite_pairs)result.root_mean_square=std::sqrt(squares/double(finite_pairs));
  return result;
}
inline std::ostream& operator<<(std::ostream& out,const M0NativePcmDifferenceV1& d){
  return out<<"bytes="<<d.expected_bytes<<"/"<<d.actual_bytes<<" samples="<<d.samples
    <<" unequal_bits="<<d.unequal_bits<<" nonfinite="<<d.nonfinite_expected<<"/"<<d.nonfinite_actual
    <<" max_absolute="<<d.maximum_absolute<<" rms="<<d.root_mean_square<<" max_ulp="<<d.maximum_ulp
    <<" first_index="<<d.first_unequal<<" first_bits="<<d.first_expected_bits<<"/"<<d.first_actual_bits;
}
} // namespace onnxruntime
