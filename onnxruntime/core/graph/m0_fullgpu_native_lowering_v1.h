// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Source-private construction, not an admission/permission API.
#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "core/common/status.h"
#include "core/graph/onnx_protobuf.h"
namespace onnxruntime {
class M0FullGpuNativeLoweringOwnerV1;
struct M0FullGpuNativeTensorV1 {
  std::string data, extent;
  uint32_t channels=0, role=0, instance=0;
};
struct M0FullGpuNativeEncoderV1 {
  M0FullGpuNativeTensorV1 aligned_text, f0, n0;
  std::array<std::string,2> token_status;
};
struct M0FullGpuNativeOutputV1 {
  M0FullGpuNativeTensorV1 pcm;
  std::string packet;
  std::array<std::string,2> token_status;
};
// The approved recipe is a compiled semantic cut, not a caller-supplied hash.
// Coverage preserves every exact original wire and all actual replacements,
// including shared original producers consumed by multiple fused recipes.
struct M0FullGpuNativeCoverageV1 {
  std::string original_wire;
  bool copied=false;
  std::vector<std::string> recipes, consumers;
};
class M0FullGpuNativeBuilderV1 final {
 public:
  const ONNX_NAMESPACE::ModelProto& Original()const{return original_;}
  const ONNX_NAMESPACE::NodeProto& Node(size_t index)const;
  std::string Input(size_t index,size_t slot)const;
  std::string Output(size_t index,size_t slot=0)const;
  uint32_t Capacity()const{return capacity_;}
  // Only original immutable TensorProto bytes can enter the derived model.
  std::string Initializer(const std::string& name);
  std::string Parameter(size_t original_node,size_t slot);
  void CopyOriginal(size_t index,const std::map<std::string,std::string>& renames={});
  // Boundaries are ORIGINAL edge names. The node's inputs are actual derived
  // references. Dependency closure is recorded, never evaluated on the host.
  void Replace(const std::string& recipe,const std::vector<std::string>& original_outputs,
               const std::vector<std::string>& original_boundaries,ONNX_NAMESPACE::NodeProto node);
  // Administrative nodes have no numerical source coverage and never waive it.
  void Administrative(ONNX_NAMESPACE::NodeProto node);
  void BindDecoderProjectionChannels(const M0FullGpuNativeTensorV1& decoder_input);
  static ONNX_NAMESPACE::NodeProto Custom(const std::string& op,const std::vector<std::string>& inputs,
                                         const std::vector<std::string>& outputs,
                                         const std::vector<std::pair<std::string,int64_t>>& attrs={});
 private:
  friend class M0FullGpuNativeLoweringOwnerV1;
  M0FullGpuNativeBuilderV1(const ONNX_NAMESPACE::ModelProto& original,
                         const ONNX_NAMESPACE::GraphProto& payload_graph,uint32_t capacity);
  M0FullGpuNativeBuilderV1(const M0FullGpuNativeBuilderV1&)=delete;
  M0FullGpuNativeBuilderV1& operator=(const M0FullGpuNativeBuilderV1&)=delete;
  void Emit(ONNX_NAMESPACE::NodeProto node);
  void Finalize(const M0FullGpuNativeOutputV1& outputs);
  const ONNX_NAMESPACE::ModelProto& original_;
  const ONNX_NAMESPACE::GraphProto& payload_graph_;
  const uint32_t capacity_;
  ONNX_NAMESPACE::ModelProto derived_;
  std::array<M0FullGpuNativeCoverageV1,2464> coverage_;
  std::map<std::string,size_t> producers_, initializers_, emitted_values_;
  std::map<std::string,bool> copied_initializers_;
};
common::Status M0AppendNativeEncoderV1(M0FullGpuNativeBuilderV1& builder,M0FullGpuNativeEncoderV1& output);
common::Status M0AppendNativeDecoderV1(M0FullGpuNativeBuilderV1& builder,const M0FullGpuNativeEncoderV1& encoder,M0FullGpuNativeOutputV1& output);
}  // namespace onnxruntime
