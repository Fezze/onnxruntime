// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "core/common/common.h"

namespace ONNX_NAMESPACE {
class GraphProto;
class OpSchema;
class AttributeProto;
class TypeProto;
}  // namespace ONNX_NAMESPACE
namespace onnxruntime {
class Graph;
class Model;
class Node;
class FunctionImpl;
class M0SyntheticOwnerFactoryV1;
class M0GraphRegistrationV1;
class OnnxRuntimeOpSchemaRegistry;
class InferenceContextImpl;
class DataPropagationContextImpl;
class GraphInferencerImpl;

// Values only: neither the candidate nor any capability leaves the fixed factory.
struct M0SyntheticOwnerReceiptV1 {
  uint64_t owner_nonce = 0;
  bool owner_retired = false;
  size_t registered = 0;
  size_t unregistered = 0;
  size_t live_at_capture = 0;
  size_t full_validations = 0;
  size_t generation_acquisitions = 0;
  size_t frame_pushes = 0;
  size_t frame_pops = 0;
  size_t resolve_denials = 0;
  size_t callback_denials = 0;
  bool constructor_failure_unwound = false;
  bool parentless_inherits_normalized_imports = false;
  bool inherited_empty_list_denied = false;
  bool clean_child_denied = false;
  bool foreign_scope_denied = false;
  bool conflict_denied_before_graph_read = false;
  bool same_frame_reentry_denied = false;
  bool poisoned = false;
  bool descriptor_permission = false;
};

M0SyntheticOwnerReceiptV1 M0RunFixedSyntheticOwnerFoundationV1();

// Mechanical live comparisons only; no callback/default/inference authority.
struct M0SyntheticProjectionReceiptV1 {
  size_t checked_phases = 0;
  size_t checked_scopes = 0;
  size_t rejected_mutations = 0;
  size_t retired_owners = 0;
  bool map_body_preserved = false;
  bool failed_registration_preserved = false;
  bool phase_mismatch_poisoned = false;
  bool descriptor_permission = false;
};
M0SyntheticProjectionReceiptV1 M0RunFixedSyntheticProjectionV1();

struct M0SyntheticPreparationReceiptV1 {
  size_t prepared_roots = 0;
  size_t initialized_scopes = 0;
  size_t connected_scopes = 0;
  size_t sorted_scopes = 0;
  size_t redirected_requests = 0;
  size_t rejected_backing_mutations = 0;
  size_t retired_owners = 0;
  bool descriptor_permission = false;
};
M0SyntheticPreparationReceiptV1 M0RunFixedSyntheticPreparationV1();

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
struct M0SyntheticIssuedReceiptV1 {
  size_t source_checks = 0;
  size_t source_selections = 0;
  size_t source_defaults = 0;
  size_t retired_owners = 0;
  size_t callbacks = 0;
  size_t child_joins = 0;
  size_t propagation_no_work = 0;
  size_t saver_no_work = 0;
  size_t live_publications = 0;
  size_t child_skips = 0;
  bool fixed_post_perform_completed = false;
  bool meaningful_nested_failure = false;
  bool callback_conflict_denied = false;
  bool payload_denied = false;
  bool sparse_denied = false;
  bool symbolic_denied = false;
  bool attribute_denied = false;
  bool function_denied = false;
  bool propagation_read_denied = false;
  bool propagation_write_denied = false;
  bool malformed_propagation_denied = false;
  bool engaged_inventory_denied = false;
  bool repeated_body_denied = false;
  bool direct_perform_denied = false;
  bool dispatch_closed = false;
  bool descriptor_permission = false;
};
M0SyntheticIssuedReceiptV1 M0RunFixedSyntheticIssuedV1();
#endif

// CLOSED foundation: registration/generation do not authorize descriptor dispatch.
class M0SyntheticOwnerV1 final {
 private:
  friend class Model;
  friend class Graph;
  friend class M0GraphRegistrationV1;
  friend class M0SyntheticOwnerFactoryV1;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
  friend class InferenceContextImpl;
  friend class DataPropagationContextImpl;
  friend class GraphInferencerImpl;
#endif
  struct Entry {
    const Graph* graph = nullptr;
    const ONNX_NAMESPACE::GraphProto* backing = nullptr;
    std::string original_backing;
    std::string original_canonical;
    std::string name;
    std::string parent_node;
    size_t parent_slot = 0;
    uint64_t scope_id = 0;
    uint64_t owner_nonce = 0;
    const void* schema_collection = nullptr;
    bool constructed = false;
    bool retired = false;
  };
  struct Generation {
    explicit Generation(M0SyntheticOwnerV1& owner) noexcept;
    ~Generation() noexcept;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Generation);
    M0SyntheticOwnerV1& owner;
    uint64_t id = 0;
    bool acquired = false;
  };
  struct Frame {
    Frame(Generation& generation, const Graph* graph);
    ~Frame() noexcept;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Frame);
    Generation& generation;
    const Graph* graph;
    bool pushed = false;
  };

  explicit M0SyntheticOwnerV1(std::shared_ptr<M0SyntheticOwnerReceiptV1> audit);
  ~M0SyntheticOwnerV1() noexcept;
  friend struct std::default_delete<M0SyntheticOwnerV1>;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0SyntheticOwnerV1);
  uint64_t Register(const Graph* graph, const ONNX_NAMESPACE::GraphProto* proto,
                    const Graph* parent, const Node* parent_node);
  void Finish(uint64_t scope_id);
  void Unregister(uint64_t scope_id) noexcept;
  void ValidateAll(const Graph* requested);
  void ValidateBacking(const Model& model);
  void ValidateProjection(const Graph* requested, size_t mechanical_phase);
  common::Status ResolveClosed(const Graph* requested);
  common::Status PrepareClosed(const Graph* requested, Generation& generation);
  void PrepareSorted(const Graph* requested);
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
  enum class IssuedNodeStage { kOriginal,
                               kChecked,
                               kSelected,
                               kDefaulted,
                               kInferred,
                               kPublished,
                               kDone };
  enum class IssuedEffect { kNone,
                            kNestedFailure,
                            kConflict,
                            kPayload,
                            kSparse,
                            kSymbolic,
                            kAttribute,
                            kFunction,
                            kPropagationRead,
                            kPropagationWrite,
                            kMalformedPropagation,
                            kEngagedInventory,
                            kRepeatedBody,
                            kDirectPerform };
  class ContextLayer;
  class ChildJoin;
  class ScopeLayer;
  class NodeLayer;
  class PermitGeneration final {
   private:
    friend class M0SyntheticOwnerFactoryV1;
    friend class ScopeLayer;
    friend class Graph;
    friend class NodeLayer;
    friend class ContextLayer;
    friend class ChildJoin;
    friend class GraphInferencerImpl;
    explicit PermitGeneration(M0SyntheticOwnerV1& owner);
    ~PermitGeneration() noexcept;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PermitGeneration);
    Generation generation_;
    bool root_completed_ = false;
  };
  class ScopeLayer final {
   private:
    friend class M0SyntheticOwnerFactoryV1;
    friend class Graph;
    friend class NodeLayer;
    friend class ContextLayer;
    friend class ChildJoin;
    friend class GraphInferencerImpl;
    ScopeLayer(PermitGeneration& generation, const Graph& graph);
    ~ScopeLayer() noexcept;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopeLayer);
    void Check(const Graph& graph);
    PermitGeneration& permit_;
    Frame frame_;
    bool perform_entered_ = false;
    bool completed_ = false;
    size_t slot_ = 0;
  };
  class NodeLayer final {
   private:
    friend class Graph;
    friend class ContextLayer;
    friend class ChildJoin;
    friend class GraphInferencerImpl;
    NodeLayer(ScopeLayer& scope, const Node& node);
    ~NodeLayer() noexcept;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NodeLayer);
    void Check();
    void Checked();
    void Selected();
    void BeforeDefault(const ONNX_NAMESPACE::AttributeProto& attribute);
    void Defaulted();
    void Published();
    void Completed();
    ScopeLayer& scope_;
    const Node& node_;
    bool completed_ = false;
  };
  class ContextLayer final {
   private:
    friend class Graph;
    friend class InferenceContextImpl;
    friend class DataPropagationContextImpl;
    friend class GraphInferencerImpl;
    friend class ChildJoin;
    ContextLayer(NodeLayer& node);
    ~ContextLayer() noexcept;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ContextLayer);
    void Check();
    void BindInference(const void* context, std::vector<ONNX_NAMESPACE::TypeProto>& outputs);
    void BindPropagation(const void* context, std::vector<ONNX_NAMESPACE::TypeProto>& outputs);
    void EnterCallback(const void* context);
    void ExitCallback(const void* context);
    void DescriptorAccess(const void* context, size_t index);
    void BodyAccess(const void* context, const std::string& name);
    void PropagationNoWork(const void* context);
    void SaverNoWork(const Node& node, const void* output, size_t index,
                     const std::vector<ONNX_NAMESPACE::TypeProto>& outputs);
    void CallbackEffect(InferenceContextImpl& context);
    NodeLayer& node_;
    const void* inference_identity_ = nullptr;
    const void* propagation_identity_ = nullptr;
    std::vector<ONNX_NAMESPACE::TypeProto>* scratch_ = nullptr;
    std::vector<ONNX_NAMESPACE::TypeProto>* propagation_ = nullptr;
    bool callback_active_ = false;
    bool callback_finished_ = false;
    bool propagation_finished_ = false;
    bool saver_finished_ = false;
    bool body_issued_ = false;
  };
  class ChildJoin final {
   private:
    friend class InferenceContextImpl;
    friend class GraphInferencerImpl;
    friend class Graph;
    ChildJoin(ContextLayer& context, Graph& child);
    ~ChildJoin() noexcept;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ChildJoin);
    void Enter(ScopeLayer& scope, const std::vector<const ONNX_NAMESPACE::TypeProto*>& inputs);
    void Joined(const Node& node, const Graph& child);
    void Returned();
    ContextLayer& context_;
    Graph& child_;
    ScopeLayer* scope_ = nullptr;
    bool entered_ = false;
    bool joined_ = false;
    bool returned_ = false;
  };
  PermitGeneration* issued_generation_ = nullptr;
  std::array<IssuedNodeStage, 4> issued_node_stage_{};
  M0SyntheticIssuedReceiptV1* issued_audit_ = nullptr;
  std::array<bool, 4> issued_inferred_child_{};
  IssuedEffect issued_effect_ = IssuedEffect::kNone;
#endif
  bool DenyCallback() noexcept;
  void Poison() noexcept { poisoned_.store(true, std::memory_order_release); }
  bool Poisoned() const noexcept { return poisoned_.load(std::memory_order_acquire); }

  std::shared_ptr<M0SyntheticOwnerReceiptV1> audit_;
  const uint64_t nonce_;
  enum class State { kConstructing,
                     kReady,
                     kResolving,
                     kPoisoned,
                     kRetired };
  State state_ = State::kConstructing;
  std::shared_ptr<OnnxRuntimeOpSchemaRegistry> sealed_registry_;
  std::array<const ONNX_NAMESPACE::OpSchema*, 2> sealed_schemas_{};
  std::array<Entry, 4> entries_;
  std::array<const Graph*, 4> stack_{};
  size_t stack_size_ = 0;
  uint64_t next_scope_id_ = 1;
  uint64_t generation_id_ = 0;
  std::atomic<bool> active_{false};
  std::atomic<bool> poisoned_{false};
  bool parentless_transition_ = false;
  const Model* model_identity_ = nullptr;
  const FunctionImpl* parentless_identity_ = nullptr;
  std::string normalized_model_canonical_;
  std::string original_function_;
  enum class Preparation { kOriginal,
                           kInitialized,
                           kConnected,
                           kSorted };
  std::array<Preparation, 4> preparation_{};
  M0SyntheticPreparationReceiptV1* preparation_audit_ = nullptr;
  size_t fail_after_slot_ = 4;
};

// Declared directly after Graph's owning Model reference; destroyed after graph state.
class M0GraphRegistrationV1 final {
 public:
  M0GraphRegistrationV1(const Model& model, const Graph* graph,
                        const ONNX_NAMESPACE::GraphProto* proto,
                        const Graph* parent, const Node* parent_node);
  ~M0GraphRegistrationV1() noexcept;
  void Finish();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0GraphRegistrationV1);

 private:
  M0SyntheticOwnerV1* owner_ = nullptr;
  uint64_t scope_id_ = 0;
};
}  // namespace onnxruntime
