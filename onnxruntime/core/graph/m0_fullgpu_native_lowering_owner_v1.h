// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include "gsl/span"
#include "core/graph/m0_fullgpu_native_lowering_v1.h"
#include "core/graph/m0_callback_policy_v1.h"
#include <map>
#include <set>
#include "core/framework/run_options.h"
#include "core/framework/session_options.h"
#include "core/framework/ortdevice.h"
#include "core/session/plugin_ep/m0_native_device_limits_v1.h"
struct OrtValue;
namespace ONNX_NAMESPACE { class OpSchema; }
namespace onnxruntime {
class Model;
class InferenceSession;
class SessionState;
class M0NativeGpuModuleV2;

enum class M0NativeInitializerVisitV1;
class M0OriginalPreloadOwnerV1;
class Node;
class NodeArg;
class Graph;
class InferenceContextImpl;
class DataPropagationContextImpl;
namespace logging { class Logger; }
// Source-private owner of immutable construction receipts and actual execution
// state. Construction alone grants no Initialize or Run permission.
class M0FullGpuNativeLoweringOwnerV1 final {
 private:
  friend class M0OriginalPreloadOwnerV1;
  friend class InferenceSession;
  friend class Model;
  friend class Graph;
  friend class SessionState;

  friend class InferenceContextImpl;
  friend class DataPropagationContextImpl;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
  friend class M0NativeLoweringQualificationV1;
#endif
  friend struct std::default_delete<M0FullGpuNativeLoweringOwnerV1>;
  static std::unique_ptr<M0FullGpuNativeLoweringOwnerV1> Prepare(
      const Model& source,const InferenceSession& session,
      const ONNX_NAMESPACE::ModelProto& sealed_original,uint32_t capacity,bool device_capacity=false);
  M0FullGpuNativeLoweringOwnerV1(const Model& source,const InferenceSession& session,
                                uint32_t capacity,bool device_capacity,uint32_t requested_ceiling,
                                uint64_t queued_reference_limit_bytes);
  ~M0FullGpuNativeLoweringOwnerV1();
  void ResolveExecutionModel(const logging::Logger& logger);
  // Lexical proof for one synchronous validation boundary only. It never spans
  // a Run, upload, kernel callback or publication, and cannot be saved/replayed.
  class BoundarySourceValidation final {
    friend class M0FullGpuNativeLoweringOwnerV1;
    explicit BoundarySourceValidation(const M0FullGpuNativeLoweringOwnerV1& owner);
   public:
    ~BoundarySourceValidation();
   private:
    BoundarySourceValidation(const BoundarySourceValidation&)=delete;
    BoundarySourceValidation& operator=(const BoundarySourceValidation&)=delete;
    const M0FullGpuNativeLoweringOwnerV1* const owner_;
    const Model* const source_;
    const Model* const execution_;
    uint64_t serial_=0;
  };
  BoundarySourceValidation ValidateSourceBoundary() const;
  void CheckBoundarySourceValidation(const BoundarySourceValidation& proof) const;
  const ONNX_NAMESPACE::TensorProto& ExpectedInitializer(const ONNX_NAMESPACE::TensorProto& metadata,
      const BoundarySourceValidation* proof=nullptr) const;
  common::Status Callback(M0CallbackPolicyV1::Channel channel,const Model* model);
  void CheckNode(const Node& node) const;
  void Checked(const Node& node,const ONNX_NAMESPACE::NodeProto& wire);
  void Selected(const Node& node);
  void Defaulted(const Node& node,const ONNX_NAMESPACE::AttributeProto& attr);
  void Infer(Node& node,const void* context,std::vector<ONNX_NAMESPACE::TypeProto>& outputs);
  void Propagation(Node& node,const void* context,std::vector<ONNX_NAMESPACE::TypeProto>& outputs);
  void Saver(const Node& node,const NodeArg& output,size_t index,const std::vector<ONNX_NAMESPACE::TypeProto>& values);
  void Published(const Node& node);
  void ValidateExecutionModel(const BoundarySourceValidation* proof=nullptr);
  std::string ExecutionFingerprint() const;
  enum class ExecutionIntent { ObservePlan, Initialize };
  common::Status PrepareExecutionPlan(InferenceSession& session);
  common::Status InitializeExecution(InferenceSession& session);
  common::Status RunExecution(InferenceSession& session,const RunOptions& options,
      gsl::span<const std::string> feed_names,gsl::span<const OrtValue> feeds,
      gsl::span<const std::string> output_names,std::vector<OrtValue>* fetches,
      const std::vector<OrtDevice>* fetch_devices);
  common::Status CreateExecutionPlan(InferenceSession& session,ExecutionIntent intent);
  void BeginInitialization(SessionState& state);
  void FinishInitialization(SessionState& state);
  void AbortInitialization() noexcept;
  void ValidateInitializedExecution(uint64_t active_run=0);
  bool ExecutionInitialized() const noexcept;
  common::Status BeforeInitializerUpload(const SessionState& state,const std::string& name,int index,
      const ONNX_NAMESPACE::TensorProto& proto,M0NativeInitializerVisitV1 visit);
  common::Status AfterInitializerUpload(SessionState& state,int index,const OrtValue& value,bool constant,bool sparse);
  common::Status CaptureExecutionPlan(SessionState& state);
  std::string ExecutionPlanFingerprint(const SessionState& state) const;
  void ValidateInitializerPlan(const SessionState& state,const BoundarySourceValidation* proof=nullptr) const;
  void ValidateResourcePlan(const SessionState& state) const;
  struct ExecutionPlan;
  struct ExecutionPlanDeleter { void operator()(ExecutionPlan*) const; };
  std::unique_ptr<ExecutionPlan,ExecutionPlanDeleter> execution_plan_;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
  struct PlanObservation {
    size_t nodes=0,selected=0,resources=0,initializers=0,kernels=0,allocators=0;
    bool state_bound=false,epoch_bound=false,plan_bound=false,resource_gate_closed=false;
    std::string execution_sha256;
  };
  PlanObservation ObserveExecutionPlan() const;
  SessionState& QualificationExecutionState();
#endif
  void RequireResolveLease();
  enum class Phase { Dormant,Constructing,Resolving,Resolved,Poisoned };
  Phase phase_=Phase::Dormant;
  const Model* constructing_model_=nullptr;
  size_t construction_calls_=0;
  bool descriptor_inference_started_=false;
  struct Descriptor {
    ONNX_NAMESPACE::NodeProto wire;
    const ONNX_NAMESPACE::OpSchema* schema=nullptr;
    const Node* node=nullptr;
    size_t node_index=0;
    std::vector<bool> saved;
    bool checked=false,selected=false,inferred=false,propagated=false,published=false;
    std::set<std::string> defaults;
    const void* inference=nullptr;
    const void* propagation=nullptr;
  };
  std::unique_ptr<Model> execution_model_;
  std::map<std::string,Descriptor> descriptors_;
  std::map<std::string,ONNX_NAMESPACE::TypeProto> value_types_;
  std::map<std::string,const NodeArg*> value_arguments_;
  M0FullGpuNativeLoweringOwnerV1(const M0FullGpuNativeLoweringOwnerV1&)=delete;
  M0FullGpuNativeLoweringOwnerV1& operator=(const M0FullGpuNativeLoweringOwnerV1&)=delete;
  struct PreparedModuleDeleter { void operator()(M0NativeGpuModuleV2* module)const; };
  std::unique_ptr<M0NativeGpuModuleV2,PreparedModuleDeleter> prepared_module_;
  static uint32_t SelectPhysicalCapacity(M0NativeGpuModuleV2& module,const ONNX_NAMESPACE::ModelProto& original,
      const Model& source,ExecutionOrder execution_order,uint32_t requested_ceiling,M0NativeDeviceLimitsV1& limits,uint64_t& initializer_bytes,
      std::vector<std::string>& memory_order,uint64_t& available_bytes,uint64_t& peak_bytes,
      uint64_t& queued_reference_limit_bytes);
  void ValidatePhysicalCapacity(M0NativeGpuModuleV2& module)const;
  const Model* const source_;
  mutable uint64_t source_boundary_serial_=0;
  mutable uint64_t active_source_boundary_=0;
  const InferenceSession* const session_;
  const uint32_t capacity_;
  const bool device_capacity_;
  const uint32_t requested_capacity_ceiling_;
  const uint64_t planned_queued_reference_limit_bytes_;
  M0NativeDeviceLimitsV1 device_limits_;
  uint64_t initializer_capacity_bytes_=0;
  std::vector<std::string> memory_plan_order_;
  uint64_t memory_plan_available_bytes_=0,memory_plan_peak_bytes_=0;
  ONNX_NAMESPACE::ModelProto derived_;
  // Live source payload checked against the independent seal, and bound metadata
  // after one-shot transfer into the execution model.
  std::map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*,
                                const ONNX_NAMESPACE::TensorProto*>> initializer_sources_;
  bool initializer_payloads_transferred_ = false;
  size_t transferred_initializer_raw_bytes_ = 0;
  std::array<M0FullGpuNativeCoverageV1,2464> coverage_;
  M0FullGpuNativeOutputV1 outputs_;
  std::string derived_sha256_,ledger_sha256_,execution_sha256_;
};
}  // namespace onnxruntime
