// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstring>
#include <vector>
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
// Shared numerical fixture for the independent CPU oracle and real GPU callers.
// These are public tokens/style/speed, never replacement model parameters.
inline std::vector<OrtValue> M0NativePublicInputFixtureV1(
    const AllocatorPtr& allocator,bool short_input,bool invalid_token=false){
  const std::vector<int64_t> tokens=short_input?std::vector<int64_t>{0,43,0}:
      std::vector<int64_t>{0,43,44,45,46,47,48,0};
  std::vector<OrtValue> values(3);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(),TensorShape{1,int64_t(tokens.size())},allocator,values[0]);
  std::memcpy(values[0].GetMutable<Tensor>()->MutableDataRaw(),tokens.data(),tokens.size()*sizeof(int64_t));
  if(invalid_token)values[0].GetMutable<Tensor>()->MutableData<int64_t>()[1]=999999;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),TensorShape{1,256},allocator,values[1]);
  std::memset(values[1].GetMutable<Tensor>()->MutableDataRaw(),0,256*sizeof(float));
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),TensorShape{1},allocator,values[2]);
  values[2].GetMutable<Tensor>()->MutableData<float>()[0]=1.0f;
  return values;
}
} // namespace onnxruntime
