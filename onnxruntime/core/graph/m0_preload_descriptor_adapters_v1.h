// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime::m0_preload_descriptor_v1 {

// These proposals carry no permission or allocation/dispatch extent. A private
// caller must separately bind the exact schema, node, slots, defaults, literals,
// generation and GPU validation obligations before using a proposal.
namespace detail {

inline bool IsTensorRank(const ONNX_NAMESPACE::TypeProto& type, int element_type, int rank) {
  if (!type.has_tensor_type()) return false;
  const auto& tensor = type.tensor_type();
  if (!tensor.has_elem_type() || tensor.elem_type() != element_type || !tensor.has_shape() ||
      tensor.shape().dim_size() != rank) {
    return false;
  }
  for (const auto& dimension : tensor.shape().dim()) {
    if (dimension.has_dim_value() && dimension.dim_value() < 0) return false;
  }
  return true;
}

inline bool IsTensorLength(const ONNX_NAMESPACE::TypeProto& type, int element_type, int length) {
  if (!IsTensorRank(type, element_type, 1)) return false;
  const auto& dimension = type.tensor_type().shape().dim(0);
  return dimension.has_dim_value() && dimension.dim_value() == length;
}

inline void UnknownDimensions(int element_type, int rank, ONNX_NAMESPACE::TypeProto& output) {
  ONNX_NAMESPACE::TypeProto proposal;
  auto* tensor = proposal.mutable_tensor_type();
  tensor->set_elem_type(element_type);
  auto* shape = tensor->mutable_shape();
  for (int axis = 0; axis < rank; ++axis) shape->add_dim();
  output.Swap(&proposal);
}

}  // namespace detail

// Exact Resize19 FLOAT/rank3 and FLOAT[3] scales descriptors only. The caller
// proves absent roi/sizes and exact scales/attrs without invoking ParseData.
inline bool ProposeResize19Float3(const ONNX_NAMESPACE::TypeProto& input,
                                 const ONNX_NAMESPACE::TypeProto& scales,
                                 ONNX_NAMESPACE::TypeProto& output) {
  if (&output == &input || &output == &scales ||
      !detail::IsTensorRank(input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 3) ||
      !detail::IsTensorLength(scales, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 3)) {
    return false;
  }
  detail::UnknownDimensions(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 3, output);
  return true;
}

// The three present scalar slots are checked independently. The dynamic limit
// is never an absent input, and no carried scalar/list or initializer is read.
inline bool ProposeRange11Int64Scalars(const ONNX_NAMESPACE::TypeProto& start,
                                      const ONNX_NAMESPACE::TypeProto& limit,
                                      const ONNX_NAMESPACE::TypeProto& delta,
                                      ONNX_NAMESPACE::TypeProto& output) {
  if (&output == &start || &output == &limit || &output == &delta ||
      !detail::IsTensorRank(start, ONNX_NAMESPACE::TensorProto_DataType_INT64, 0) ||
      !detail::IsTensorRank(limit, ONNX_NAMESPACE::TensorProto_DataType_INT64, 0) ||
      !detail::IsTensorRank(delta, ONNX_NAMESPACE::TensorProto_DataType_INT64, 0)) {
    return false;
  }
  detail::UnknownDimensions(ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, output);
  return true;
}

// The six original ConstantOfShape20 nodes have present, exact FLOAT[1]
// literal attributes. These helpers neither access nor validate those payloads.
inline bool ProposeConstantOfShape20Float2(const ONNX_NAMESPACE::TypeProto& shape,
                                         ONNX_NAMESPACE::TypeProto& output) {
  if (&output == &shape || !detail::IsTensorLength(shape, ONNX_NAMESPACE::TensorProto_DataType_INT64, 2)) return false;
  detail::UnknownDimensions(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 2, output);
  return true;
}

inline bool ProposeConstantOfShape20Float3(const ONNX_NAMESPACE::TypeProto& shape,
                                         ONNX_NAMESPACE::TypeProto& output) {
  if (&output == &shape || !detail::IsTensorLength(shape, ONNX_NAMESPACE::TensorProto_DataType_INT64, 3)) return false;
  detail::UnknownDimensions(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 3, output);
  return true;
}

}  // namespace onnxruntime::m0_preload_descriptor_v1
