// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include "core/common/status.h"
#include <array>
#include <string_view>
#include "core/common/common.h"
#include "core/graph/onnx_protobuf.h"

namespace ONNX_NAMESPACE {
class OpSchema;
}

namespace onnxruntime {
class M0OriginalPreloadAdaptersV1;
class M0OriginalFunctionDescriptorsV1;
class M0FullModelDescriptorRulesV1;
class M0WholeModelSealV1;
class InferenceSession;
namespace logging {
class Logger;
}
class Model;
class Graph;
class Node;
class NodeArg;
class InferenceContextImpl;
class DataPropagationContextImpl;
struct M0OriginalPreloadReceiptV1 {
  size_t resolved = 0, refused = 0, defaults = 0, published = 0, mutated_input_denials = 0, merge_fault_denials = 0;
};
// Fixed original-node cuts only. No Model or authority escapes this call.
M0OriginalPreloadReceiptV1 M0RunFixedOriginalPreloadV1();
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
struct M0WholeDescriptorRulesReceiptV1 {
  size_t profiles = 0, actual_dp_profiles = 0, rank_assertions = 0;
  size_t mixed_dtype_denials = 0, copied_default_denials = 0, lstm_optional_holes = 0, multiple_outputs = 0;
};
M0WholeDescriptorRulesReceiptV1 M0QualifyWholeDescriptorRulesV1(std::string_view bytes);
size_t M0QualifyWholeRetirementV1(std::string_view bytes);
#endif
class M0OriginalPreloadOwnerV1 final {
 private:
  friend class Model;
  friend class Graph;
  friend class InferenceContextImpl;
  friend class DataPropagationContextImpl;
  friend class InferenceSession;
  friend class M0OriginalLstmSelectionV1;
  friend struct std::default_delete<M0OriginalPreloadOwnerV1>;
  friend M0OriginalPreloadReceiptV1 M0RunFixedOriginalPreloadV1();
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
  friend M0WholeDescriptorRulesReceiptV1 M0QualifyWholeDescriptorRulesV1(std::string_view bytes);
  friend size_t M0QualifyWholeRetirementV1(std::string_view bytes);
  static M0WholeDescriptorRulesReceiptV1 QualifyWholeRules(std::string_view bytes);
  static size_t QualifyWholeRetirement(std::string_view bytes);
  void BeginSelectedLstmConstructorsV1(uint64_t generation);
  void FinishSelectedLstmConstructorsV1(uint64_t generation);
  void AbortSelectedLstmConstructorsV1() noexcept;
  uint64_t selection_generation_v1_ = 0;
  bool selection_active_v1_ = false;
#endif
  using Adapters = M0OriginalPreloadAdaptersV1;
  using Functions = M0OriginalFunctionDescriptorsV1;
  using Rules = M0FullModelDescriptorRulesV1;
  class WholeModel;
  struct FrozenRecipe {
    size_t index;
    std::string_view node_wire;
    std::array<std::string_view, 4> types;
    std::array<std::string_view, 4> literals;
    std::string_view output_type;
  };
  template <typename T>
  static FrozenRecipe View(const T& recipe) {
    return {recipe.index, recipe.node_wire, recipe.types, recipe.literals, recipe.output_type};
  }
  explicit M0OriginalPreloadOwnerV1(const FrozenRecipe& recipe);
  explicit M0OriginalPreloadOwnerV1(std::unique_ptr<M0WholeModelSealV1> seal);
#if !defined(ORT_MINIMAL_BUILD)
  ~M0OriginalPreloadOwnerV1();
#else
  ~M0OriginalPreloadOwnerV1() = default;
#endif
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0OriginalPreloadOwnerV1);
  bool Construction(const Model* model) noexcept {
    if (poisoned_ || model_ || stage_ || construction_count_ >= 2 ||
        (construction_model_ && construction_model_ != model)) return !DenyCallback();
    construction_model_ = model;
    ++construction_count_;
    return true;
  }
  bool DenyCallback() noexcept {
    poisoned_ = true;
    return true;
  }
  void CheckGraph(const Graph& graph) const;
  void CheckNode(const Node& node) const;
  void Checked(const Node& node, const ONNX_NAMESPACE::NodeProto& original);
  void Selected(const Node& node);
  void Defaulted(const Node& node, const ONNX_NAMESPACE::AttributeProto& origin);
  void Infer(Node& node, const void* context, std::vector<ONNX_NAMESPACE::TypeProto>& outputs);
  void Propagation(const Node& node, const void* context, const std::vector<ONNX_NAMESPACE::TypeProto>& outputs);
  void Saver(const Node& node, const NodeArg& output, size_t output_index,
             const std::vector<ONNX_NAMESPACE::TypeProto>& outputs);
  void Published(const Node& node);
  common::Status Resolve(Graph& graph);
  static M0OriginalPreloadReceiptV1 RunFixed();
  static std::unique_ptr<Model> Make(const FrozenRecipe& recipe);
  static std::unique_ptr<Model> MakeWhole(std::string_view bytes, const logging::Logger& logger);
  static std::unique_ptr<Model> MakeWholeUnresolved(std::string_view bytes, const logging::Logger& logger);
#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<WholeModel> whole_;
#endif
  const FrozenRecipe recipe_;
  ONNX_NAMESPACE::NodeProto original_;
  std::string model_wire_;
  std::unordered_map<std::string, int> domains_;
  const Model* construction_model_ = nullptr;
  size_t construction_count_ = 0;
  Model* model_ = nullptr;
  Graph* graph_ = nullptr;
  Node* node_ = nullptr;
  const ONNX_NAMESPACE::OpSchema* schema_ = nullptr;
  std::vector<const ONNX_NAMESPACE::AttributeProto*> defaults_;
  ONNX_NAMESPACE::TypeProto proposal_;
  const void* context_ = nullptr;
  const void* propagation_ = nullptr;
  size_t stage_ = 0;
  bool active_ = false, poisoned_ = false, fail_before_merge_ = false;
};
}  // namespace onnxruntime
