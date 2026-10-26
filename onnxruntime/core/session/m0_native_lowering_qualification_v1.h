// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include "core/session/m0_native_pcm_difference_v1.h"
#include <cstddef>
#include <string>
#include <vector>
#include "core/common/status.h"

namespace onnxruntime {
class InferenceSession;
enum class M0NativeLoweringCapacityV1 { k1024, k2048, kZero, kAboveMaximum };
enum class M0NativeDescriptorFaultV1 {
  None, InputData, LateResolve, UnusedLstmOutputType, UnusedLstmShapeData,
  UnusedLstmOutputSubstitution, LiveDefaultValue, CopiedDefaultLateCallback, OriginalTokenType
};
struct M0NativeDescriptorReceiptV1 {
  size_t nodes = 0, checked = 0, selected = 0, inferred = 0, propagated = 0, published = 0, defaults = 0;
  size_t saved_outputs = 0, present_outputs = 0;
  bool separate_model = false, no_inferred_values = false;
  bool fault_denied = false, restored_replay_denied = false, retired = false;
};
enum class M0NativeModuleFaultV2 {
  None, QueriedStateSwap, QueriedDefinitionSwap, SelectedInfoSwap, SelectedDefinitionSwap, JointDefinitionMemory
};
enum class M0NativeInitializerFaultV1 {
  None, RawBytes, ValueType, AllocationKind, InitializerOrder, Lifetime
};
enum class M0NativeInitializerHookV1 { Plan, Copy, Endpoint };
struct M0NativeInitializerHookReceiptV1 {
  bool denied = false, replay_denied = false, retired = false;
  bool maps_unchanged = false, session_uninitialized = false;
};
struct M0NativeInitializerReceiptV1 {
  size_t initializers = 0;
  bool resource_denied = false, fault_denied = false, restored_replay_denied = false;
  bool retired = false, session_uninitialized = false;
};
enum class M0NativeResourceFaultV1 { None, AllocationReuse, ForeignRow, ReleaseBeforeLastUse, StreamBinding };
struct M0NativeResourceReceiptV1 {
  size_t resources = 0, releases = 0;
  bool resource_denied = false, fault_denied = false, restored_replay_denied = false;
  bool retired = false, session_uninitialized = false;
};
enum class M0NativeResourceMetadataFaultV3 { None, ContextSwap, QueueSwap };
struct M0NativeResourceMetadataReceiptV3 {
  bool context_present = false, limits_present = false, context_has_no_allocation = false;
  bool stable_requery = false, invalid_handle_denied = false, malformed_context_denied = false;
  bool missing_extent_denied = false, fault_denied = false, restored_requery = false;
  bool session_uninitialized = false;
};
enum class M0NativeInitializedFaultV1 { None, InitializerSubstitution, AllocationReuse, LateCopy };
struct M0NativeInitializedReceiptV1 {
  size_t expected_initializers = 0, initializers = 0, constants = 0, uploaded = 0, planned_sources = 0;
  size_t nodes = 0, kernels = 0, unique_resources = 0;
  bool initialized = false, checked_completion = false, state_bound = false, epoch_bound = false, plan_bound = false;
  bool gpu_parameters = false, original_model_retained = false, original_ports_retained = false;
  bool session_uninitialized = false, replay_retained = false, fault_denied = false, restored_replay_denied = false;
  bool retired = false;
};
struct M0NativeRunReceiptV1 {
  M0NativePcmDifferenceV1 preallocated_difference,empty_wrapper_difference,repeated_difference;
  uint32_t physical_capacity=0;
  bool device_selected_capacity=false;
  size_t long_samples = 0, short_samples = 0, repeated_samples = 0;
  bool valid_logical_pcm = false, checked_idle_after_each_run = false, same_private_state = false;
  bool canonical_token_failure = false, failed_output_unchanged = false;
  bool first_pcm_retained = false, repeated_pcm_equal = false, original_session_retained = false;
  bool public_initialized_replay = false, input_bounds_denied = false, untrusted_option_denied = false;
  bool input_refusal_output_unchanged = false;
  bool sparse_descriptors_checked = false, sparse_input_denied = false, sparse_output_denied = false;
  bool sparse_refusal_values_retained = false;
  bool preallocated_length_denied = false, preallocated_refusal_output_unchanged = false;
  bool strided_descriptors_checked = false, strided_input_denied = false, strided_output_denied = false;
  bool strided_refusal_values_retained = false;
  bool pointer_output_allocated = false, pointer_refusal_unpublished = false;
  bool pointer_empty_wrapper_retained = false, pointer_empty_wrapper_pcm_equal = false;
  bool pointer_preallocated_retained = false, pointer_preallocated_pcm_equal = false;
  bool pointer_preallocated_canaries_retained = false;
};
enum class M0NativeRepeatInterludeV1 { None,Short,Invalid };
struct M0NativeRepeatReceiptV1 {
  M0NativePcmDifferenceV1 difference;
  std::vector<unsigned char> first_pcm,repeated_pcm;
  uint32_t physical_capacity=0;
  bool first_retained=false,checked_idle=false,interlude_valid=false;
};
struct M0NativePlanReceiptV1 {
  size_t nodes = 0, selected = 0, resources = 0, initializers = 0, kernels = 0, allocators = 0;
  bool state_bound = false, epoch_bound = false, plan_bound = false, resource_gate_closed = false;
  bool original_model_retained = false, original_ports_retained = false, session_uninitialized = false;
  bool resource_denied = false, replay_resource_denied = false, retained_plan_holder = false;
  std::string execution_sha256;
};
struct M0NativeModuleReceiptV2 {
  size_t queried_records = 0, selected_records = 0, separate_definition_clones = 0;
  bool stable_requery = false, invalid_records_denied = false, fault_denied = false;
  bool original_session_uninitialized = false;
  bool coherent_definition_drift = false, restored_definition_metadata = false;
};
struct M0NativeLoweringReceiptV1 {
  size_t original_nodes = 0, covered_nodes = 0, derived_nodes = 0;
  bool original_model_retained = false, original_ports_retained = false;
  bool initializer_payloads_transferred = false;
  size_t retained_derived_raw_bytes = 0;
  size_t transferred_initializer_raw_bytes = 0;
  // Only the actual private owner's derived bytes, never a caller's graph.
  std::string derived_model_wire, derived_sha256, ledger_sha256;
};
// Qualification-only access to the same sealed-source implementation. Closed
// initialization/Run fixtures return observations, never caller-supplied graph,
// factory, owner or transferable execution permission.
class M0NativeLoweringQualificationV1 final {
 public:
  static common::Status Prepare(InferenceSession& session, M0NativeLoweringCapacityV1 capacity,
                                M0NativeLoweringReceiptV1* receipt = nullptr);
  static M0NativeDescriptorReceiptV1 Descriptors(InferenceSession& session, M0NativeDescriptorFaultV1 fault);
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
  static M0NativeModuleReceiptV2 ModuleMetadata(InferenceSession& session, M0NativeModuleFaultV2 fault);
  static M0NativePlanReceiptV1 Plan(InferenceSession& session);
  static M0NativeInitializerReceiptV1 InitializerPlan(InferenceSession& session, M0NativeInitializerFaultV1 fault);
  static M0NativeInitializerHookReceiptV1 InitializerHookOutsidePhase(InferenceSession& session, M0NativeInitializerHookV1 hook);
  static M0NativeResourceReceiptV1 ResourcePlan(InferenceSession& session, M0NativeResourceFaultV1 fault);
  static M0NativeResourceMetadataReceiptV3 ResourceMetadata(InferenceSession& session, M0NativeResourceMetadataFaultV3 fault);
  static M0NativeInitializedReceiptV1 InitializedExecution(InferenceSession& session, M0NativeInitializedFaultV1 fault);
  static M0NativeRunReceiptV1 RunSequence(InferenceSession& session);
  static M0NativeRepeatReceiptV1 RepeatPublicPcm(InferenceSession& session,M0NativeRepeatInterludeV1 interlude);
#endif

};
}  // namespace onnxruntime
#endif
