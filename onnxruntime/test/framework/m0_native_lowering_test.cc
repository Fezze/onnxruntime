// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Deferred: fixed original-source qualification and closed real execution fixtures.
#include "core/session/inference_session.h"
#include "core/session/m0_native_lowering_qualification_v1.h"

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <functional>
#include <string>
#include "asserts.h"
#include "core/common/common.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "core/session/m0_native_terminal_result_v1.h"
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#include "test/unittest_util/test_dynamic_plugin_ep.h"
#endif

namespace onnxruntime::test {
namespace {
SessionOptions NativeLoweringOptions() {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  for (const char* key : {"m0.fullgpu.preinit_admission_v1", "m0.fullgpu.preinit_abi_v1",
                           "session.disable_prepacking", "session.disable_aot_function_inlining"}) {
    ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(key, "1"));
  }
  return options;
}

std::string Failure(const std::function<Status()>& action) {
  try {
    const auto status = action();
    return status.IsOK() ? std::string{} : status.ErrorMessage();
  } catch (const std::exception& error) {
    return error.what();
  }
}

class InitializationObservedSession final : public InferenceSession {
 public:
  using InferenceSession::InferenceSession;
  using InferenceSession::IsInitialized;
};

class M0NativeLowering : public testing::Test {
 protected:
  static void SetUpTestSuite() {
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
    // This suite runs in an isolated process with explicit physical startup.
    // Release TestMain's generic factory lease before the private source owner
    // authenticates and exclusively opens the same sibling module.
    dynamic_plugin_ep_infra::Shutdown();
    ASSERT_FALSE(dynamic_plugin_ep_infra::IsInitialized());
#endif
    const char* path = std::getenv("ORT_M0_ORIGINAL_MODEL");
    ASSERT_NE(path, nullptr) << "Qualification requires the fixed original artifact; missing input is not a pass";
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(input.is_open());
    ASSERT_EQ(input.tellg(), std::streampos(325532387));
    original_.resize(325532387);
    input.seekg(0);
    input.read(original_.data(), static_cast<std::streamsize>(original_.size()));
    ASSERT_TRUE(input.good());
  }
  static void TearDownTestSuite() { std::string{}.swap(original_); }
  void Load(InferenceSession& session) {
    ASSERT_STATUS_OK(session.Load(original_.data(), static_cast<int>(original_.size())));
  }
  static std::string original_;
};
std::string M0NativeLowering::original_;
}  // namespace

TEST_F(M0NativeLowering, RejectsUnloadedAndOrdinarySessionsWithoutConstructingAReceipt) {
  InferenceSession strict(NativeLoweringOptions(), GetEnvironment());
  EXPECT_NE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(strict, M0NativeLoweringCapacityV1::k1024); }).find(
                "RW07_NATIVE_LOWERING_SOURCE_STATE"), std::string::npos);
  InferenceSession ordinary(SessionOptions{}, GetEnvironment());
  EXPECT_NE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(ordinary, M0NativeLoweringCapacityV1::k1024); }).find(
                "RW07_NATIVE_LOWERING_SOURCE_STATE"), std::string::npos);
}

TEST_F(M0NativeLowering, ActualSourceRemainsAliveAndUnchangedAndPreparationDoesNotInitializeSession) {
  InitializationObservedSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto* model = &session.GetModel();
  const auto& graph = model->MainGraph();
  const auto* token = graph.GetInputs()[0];
  const auto* audio = graph.GetOutputs()[0];
  const auto token_type = token->TypeAsProto()->SerializeAsString();
  const auto audio_type = audio->TypeAsProto()->SerializeAsString();
  ONNX_NAMESPACE::NodeProto lstm_before;
  graph.GetNode(449)->ToProto(lstm_before);
  M0NativeLoweringReceiptV1 receipt;
  ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024, &receipt));
  EXPECT_EQ(receipt.original_nodes, 2464u);
  EXPECT_EQ(receipt.covered_nodes, 2464u);
  EXPECT_GT(receipt.derived_nodes, 0u);
  EXPECT_TRUE(receipt.original_model_retained);
  EXPECT_TRUE(receipt.original_ports_retained);
  EXPECT_EQ(receipt.derived_sha256.size(), 64u);
  EXPECT_EQ(receipt.ledger_sha256.size(), 64u);
  EXPECT_FALSE(receipt.derived_model_wire.empty());
  // The Python DAG oracle consumes these actual bytes, never an authored
  // replacement graph. A stale capture must not be silently overwritten.
  constexpr const char* capture = "rw07_native_lowered_v1.onnx";
  std::error_code capture_error;
  ASSERT_FALSE(std::filesystem::exists(capture, capture_error));
  ASSERT_FALSE(capture_error);
  std::ofstream captured(capture, std::ios::binary | std::ios::out | std::ios::trunc);
  ASSERT_TRUE(captured.is_open());
  captured.write(receipt.derived_model_wire.data(), static_cast<std::streamsize>(receipt.derived_model_wire.size()));
  captured.flush();
  ASSERT_TRUE(captured.good());
  captured.close();
  ASSERT_FALSE(captured.fail());
  ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
  EXPECT_EQ(&session.GetModel(), model);
  EXPECT_EQ(graph.NumberOfNodes(), 2464);
  EXPECT_EQ(graph.GetAllInitializedTensors().size(), 554u);
  ASSERT_EQ(graph.GetInputs().size(), 3u);
  ASSERT_EQ(graph.GetOutputs().size(), 1u);
  EXPECT_EQ(graph.GetInputs()[0], token);
  EXPECT_EQ(graph.GetOutputs()[0], audio);
  EXPECT_EQ(token->TypeAsProto()->SerializeAsString(), token_type);
  EXPECT_EQ(audio->TypeAsProto()->SerializeAsString(), audio_type);
  ONNX_NAMESPACE::NodeProto lstm_after;
  graph.GetNode(449)->ToProto(lstm_after);
  EXPECT_EQ(lstm_before.SerializeAsString(), lstm_after.SerializeAsString());
  // Preparation alone cannot allocate tensors, construct kernels or publish an
  // initialized state. Public Initialize is covered by the actual G/H fixtures.
  EXPECT_FALSE(session.IsInitialized());
  EXPECT_ANY_THROW(session.GetSessionState());
}

TEST_F(M0NativeLowering, RefusesCachedReceiptCapacitySubstitution) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
  EXPECT_NE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k2048); }).find(
                "RW07_NATIVE_LOWERING_RECEIPT_SUBSTITUTION"), std::string::npos);
  ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
}

TEST_F(M0NativeLowering, RevalidatesActualOriginalBeforePreparationAndOnCacheHits) {
  for (const bool cached : {false, true}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    if (cached) ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
    // A fault against the live source preserves its address and node inventory.
    // Checking only source_/session_ pointers cannot detect this substitution.
    auto& graph = const_cast<Graph&>(session.GetModel().MainGraph());
    ASSERT_NE(graph.GetNode(449), nullptr);
    graph.GetNode(449)->AddAttribute("rw07_unreviewed_native_fault", int64_t{1});
    const auto denied = Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024); });
    EXPECT_FALSE(denied.empty()) << "cached=" << cached;
    EXPECT_NE(denied.find("RW07_WHOLE_"), std::string::npos);
    const auto retry = Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024); });
    EXPECT_FALSE(retry.empty());
    EXPECT_NE(retry.find("RW07_NATIVE_LOWERING_REQUIRES_OWNED_ORIGINAL"), std::string::npos);
  }
}

TEST_F(M0NativeLowering, RechecksCancellationAndClosedOptionsBeforePreparationAndOnCacheHits) {
  for (const bool cached : {false, true}) {
    // Each denial uses a fresh source, so poison/retirement cannot hide a later
    // missing options check.
    for (int fault = 0; fault < 4; ++fault) {
      InferenceSession session(NativeLoweringOptions(), GetEnvironment());
      ASSERT_NO_FATAL_FAILURE(Load(session));
      if (cached) ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
      auto& options = session.GetMutableSessionOptions();
      if (fault == 0) options.load_cancellation_flag->store(true);
      if (fault == 1) options.load_cancellation_flag.reset();
      if (fault == 2) ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.unreviewed_option", "1"));
      if (fault == 3) options.enable_profiling = true;
      const auto denied = Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024); });
      EXPECT_FALSE(denied.empty()) << "cached=" << cached << " fault=" << fault;
      EXPECT_NE(denied.find("RW07_"), std::string::npos);
      // Restore this unrelated side effect before ordinary session teardown.
      if (fault == 3) options.enable_profiling = false;
    }
  }
}

TEST_F(M0NativeLowering, FailedBoundedConstructionRetiresOriginalOwner) {
  for (const auto capacity : {M0NativeLoweringCapacityV1::kZero, M0NativeLoweringCapacityV1::kAboveMaximum}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    EXPECT_FALSE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, capacity); }).empty());
    EXPECT_NE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024); }).find(
                  "RW07_NATIVE_LOWERING_REQUIRES_OWNED_ORIGINAL"), std::string::npos);
  }
}
}  // namespace onnxruntime::test
#endif

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime::test {
TEST_F(M0NativeLowering, ActualSeparateExecutionGraphCompletesDescriptorCallbacksWithoutShapeValues) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::Descriptors(session, M0NativeDescriptorFaultV1::None);
  EXPECT_GT(receipt.nodes, 0u);
  EXPECT_TRUE(receipt.separate_model);
  EXPECT_TRUE(receipt.no_inferred_values);
  EXPECT_EQ(receipt.checked, receipt.nodes);
  EXPECT_EQ(receipt.selected, receipt.nodes);
  EXPECT_EQ(receipt.inferred, receipt.nodes);
  EXPECT_EQ(receipt.propagated, receipt.nodes);
  EXPECT_EQ(receipt.published, receipt.nodes);
  EXPECT_GT(receipt.defaults, 0u);
  EXPECT_GT(receipt.present_outputs, receipt.nodes);
  EXPECT_EQ(receipt.saved_outputs, receipt.present_outputs);
}

TEST_F(M0NativeLowering, ForbiddenDataAndLateResolveRetireActualDescriptorOwner) {
  for (const auto fault : {M0NativeDescriptorFaultV1::InputData, M0NativeDescriptorFaultV1::LateResolve,
                           M0NativeDescriptorFaultV1::CopiedDefaultLateCallback}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::Descriptors(session, fault);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
  }
}

TEST_F(M0NativeLowering, UnconsumedOutputsDefaultsAndSourceTypesDenyAndCannotBeRepairedIntoAReceipt) {
  for (const auto fault : {M0NativeDescriptorFaultV1::UnusedLstmOutputType,
                           M0NativeDescriptorFaultV1::UnusedLstmShapeData,
                           M0NativeDescriptorFaultV1::UnusedLstmOutputSubstitution,
                           M0NativeDescriptorFaultV1::LiveDefaultValue,
                           M0NativeDescriptorFaultV1::OriginalTokenType}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::Descriptors(session, fault);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
  }
}
}  // namespace onnxruntime::test
#endif

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
namespace onnxruntime::test {
TEST_F(M0NativeLowering, AuthenticatedNativeModuleQueriesAllActualCompiledBindingsWithoutInitializingSession) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::ModuleMetadata(session, M0NativeModuleFaultV2::None);
  EXPECT_EQ(receipt.queried_records, 42u);
  EXPECT_EQ(receipt.selected_records, 42u);
  EXPECT_EQ(receipt.separate_definition_clones, 42u);
  EXPECT_TRUE(receipt.stable_requery);
  EXPECT_TRUE(receipt.invalid_records_denied);
  EXPECT_TRUE(receipt.original_session_uninitialized);
}

TEST_F(M0NativeLowering, AuthenticatedNativeModuleRejectsRetainedStateDefinitionAndSelectedEntrySubstitution) {
  for (const auto fault : {M0NativeModuleFaultV2::QueriedStateSwap,
                           M0NativeModuleFaultV2::QueriedDefinitionSwap,
                           M0NativeModuleFaultV2::SelectedInfoSwap,
                           M0NativeModuleFaultV2::SelectedDefinitionSwap}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::ModuleMetadata(session, fault);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.original_session_uninitialized);
  }
}

TEST_F(M0NativeLowering, AuthenticatedNativeModuleRejectsCoherentMutationOfAllThreeActualDefinitions) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::ModuleMetadata(session, M0NativeModuleFaultV2::JointDefinitionMemory);
  EXPECT_EQ(receipt.queried_records, 42u);
  EXPECT_EQ(receipt.separate_definition_clones, 42u);
  // The actual source/provider/core metadata still agrees. The immutable Open
  // snapshot must independently detect the simultaneous memory declaration change.
  EXPECT_TRUE(receipt.coherent_definition_drift);
  EXPECT_TRUE(receipt.fault_denied);
  EXPECT_TRUE(receipt.restored_definition_metadata);
  EXPECT_TRUE(receipt.original_session_uninitialized);
}
}  // namespace onnxruntime::test
#endif

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
namespace onnxruntime::test {
TEST_F(M0NativeLowering, ExplicitObservePlanRetainsAuthenticatedPlanAndRefusesBeforeModelAllocationOrKernelConstruction) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::Plan(session);
  EXPECT_GT(receipt.nodes, 0u);
  EXPECT_EQ(receipt.selected, receipt.nodes);
  EXPECT_GE(receipt.resources, receipt.nodes);
  EXPECT_EQ(receipt.initializers, 0u);
  EXPECT_EQ(receipt.kernels, 0u);
  EXPECT_EQ(receipt.allocators, 0u);
  EXPECT_TRUE(receipt.state_bound);
  EXPECT_TRUE(receipt.epoch_bound);
  EXPECT_TRUE(receipt.plan_bound);
  EXPECT_TRUE(receipt.resource_gate_closed);
  EXPECT_TRUE(receipt.resource_denied);
  EXPECT_TRUE(receipt.replay_resource_denied);
  EXPECT_TRUE(receipt.retained_plan_holder);
  EXPECT_TRUE(receipt.original_model_retained);
  EXPECT_TRUE(receipt.original_ports_retained);
  EXPECT_TRUE(receipt.session_uninitialized);
  EXPECT_EQ(receipt.execution_sha256.size(), 64u);
}

TEST_F(M0NativeLowering, OwnedPlanDoesNotReopenLateResolveOrInputDataChannels) {
  for (const auto fault : {M0NativeDescriptorFaultV1::LateResolve, M0NativeDescriptorFaultV1::InputData}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    ASSERT_TRUE(M0NativeLoweringQualificationV1::Plan(session).resource_denied);
    const auto receipt = M0NativeLoweringQualificationV1::Descriptors(session, fault);
    EXPECT_TRUE(receipt.fault_denied);
    EXPECT_TRUE(receipt.retired);
    EXPECT_TRUE(receipt.restored_replay_denied);
  }
}

TEST_F(M0NativeLowering, ActualInitializerBytesAndSelectedGpuConsumersRemainBehindResourceDenial) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::InitializerPlan(session, M0NativeInitializerFaultV1::None);
  EXPECT_GT(receipt.initializers, 0u);
  EXPECT_TRUE(receipt.resource_denied);
  EXPECT_TRUE(receipt.session_uninitialized);
  EXPECT_FALSE(receipt.retired);
}

TEST_F(M0NativeLowering, ActualInitializerAndLifetimeMutationRetiresOwnerEvenAfterMetadataRestoration) {
  for (const auto fault : {M0NativeInitializerFaultV1::RawBytes,
                           M0NativeInitializerFaultV1::ValueType,
                           M0NativeInitializerFaultV1::AllocationKind,
                           M0NativeInitializerFaultV1::InitializerOrder,
                           M0NativeInitializerFaultV1::Lifetime}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::InitializerPlan(session, fault);
    EXPECT_GT(receipt.initializers, 0u);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, ActualSingleGpuResourcePlanKeepsResourceAdmissionClosed) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::ResourcePlan(session, M0NativeResourceFaultV1::None);
  EXPECT_GT(receipt.resources, 0u);
  EXPECT_GT(receipt.releases, 0u);
  EXPECT_TRUE(receipt.resource_denied);
  EXPECT_TRUE(receipt.session_uninitialized);
  EXPECT_FALSE(receipt.retired);
}

TEST_F(M0NativeLowering, ActualReuseForeignRowPrematureReleaseAndStreamMutationPermanentlyRetireOwner) {
  for (const auto fault : {M0NativeResourceFaultV1::AllocationReuse,
                           M0NativeResourceFaultV1::ForeignRow,
                           M0NativeResourceFaultV1::ReleaseBeforeLastUse,
                           M0NativeResourceFaultV1::StreamBinding}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::ResourcePlan(session, fault);
    EXPECT_GT(receipt.resources, 0u);
    EXPECT_GT(receipt.releases, 0u);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, AuthenticatedResourceMetadataRequeriesActualContextAndRejectsInvalidRequestsWithoutAllocation) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::ResourceMetadata(session, M0NativeResourceMetadataFaultV3::None);
  EXPECT_TRUE(receipt.context_present);
  EXPECT_TRUE(receipt.limits_present);
  EXPECT_TRUE(receipt.context_has_no_allocation);
  EXPECT_TRUE(receipt.stable_requery);
  EXPECT_TRUE(receipt.invalid_handle_denied);
  EXPECT_TRUE(receipt.malformed_context_denied);
  EXPECT_TRUE(receipt.missing_extent_denied);
  EXPECT_TRUE(receipt.restored_requery);
  EXPECT_TRUE(receipt.session_uninitialized);
}

TEST_F(M0NativeLowering, AuthenticatedResourceMetadataRejectsRetainedContextAndQueueSubstitution) {
  for (const auto fault : {M0NativeResourceMetadataFaultV3::ContextSwap, M0NativeResourceMetadataFaultV3::QueueSwap}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::ResourceMetadata(session, fault);
    EXPECT_TRUE(receipt.context_present);
    EXPECT_TRUE(receipt.stable_requery);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_requery);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, ActualInitializerPlanCopyAndEndpointHooksRejectOutsidePrivateInitializationBeforeMapInsert) {
  for (const auto hook : {M0NativeInitializerHookV1::Plan, M0NativeInitializerHookV1::Copy, M0NativeInitializerHookV1::Endpoint}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::InitializerHookOutsidePhase(session, hook);
    EXPECT_TRUE(receipt.denied) << "hook=" << int(hook);
    EXPECT_TRUE(receipt.replay_denied) << "hook=" << int(hook);
    EXPECT_TRUE(receipt.retired) << "hook=" << int(hook);
    EXPECT_TRUE(receipt.maps_unchanged) << "hook=" << int(hook);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, PrivateInitializationUploadsActualParametersConstructsAllKernelsAndRetainsCheckedStateOnReplay) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::InitializedExecution(session, M0NativeInitializedFaultV1::None);
  ASSERT_GT(receipt.expected_initializers, 0u);
  EXPECT_EQ(receipt.initializers, receipt.expected_initializers);
  EXPECT_EQ(receipt.constants, receipt.expected_initializers);
  EXPECT_EQ(receipt.uploaded, receipt.expected_initializers);
  EXPECT_EQ(receipt.planned_sources, receipt.expected_initializers);
  EXPECT_EQ(receipt.unique_resources, receipt.expected_initializers);
  ASSERT_GT(receipt.nodes, 0u);
  EXPECT_EQ(receipt.kernels, receipt.nodes);
  EXPECT_TRUE(receipt.initialized);
  EXPECT_TRUE(receipt.checked_completion);
  EXPECT_TRUE(receipt.state_bound);
  EXPECT_TRUE(receipt.epoch_bound);
  EXPECT_TRUE(receipt.plan_bound);
  EXPECT_TRUE(receipt.gpu_parameters);
  EXPECT_TRUE(receipt.replay_retained);
  EXPECT_TRUE(receipt.original_model_retained);
  EXPECT_TRUE(receipt.original_ports_retained);
  EXPECT_TRUE(receipt.session_uninitialized);
  EXPECT_FALSE(receipt.retired);
}

TEST_F(M0NativeLowering, PrivateInitializedStateRejectsActualParameterPlanAndLateCopyMutationPermanently) {
  for (const auto fault : {M0NativeInitializedFaultV1::InitializerSubstitution,
                           M0NativeInitializedFaultV1::AllocationReuse,
                           M0NativeInitializedFaultV1::LateCopy}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::InitializedExecution(session, fault);
    EXPECT_TRUE(receipt.initialized);
    EXPECT_TRUE(receipt.checked_completion);
    EXPECT_TRUE(receipt.replay_retained);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.original_model_retained);
    EXPECT_TRUE(receipt.original_ports_retained);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, ObservedPlanCannotBeUpgradedIntoPrivateInitialization) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  ASSERT_TRUE(M0NativeLoweringQualificationV1::Plan(session).resource_denied);
  EXPECT_ANY_THROW(M0NativeLoweringQualificationV1::InitializedExecution(session, M0NativeInitializedFaultV1::None));
}

TEST_F(M0NativeLowering, ActualNativeRunsRetainLogicalPcmAcrossLongShortFailureLongSequence) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::RunSequence(session);
  EXPECT_GT(receipt.long_samples, 0u);
  EXPECT_GT(receipt.short_samples, 0u);
  // Distinct token counts alone do not establish that actual GPU D changed.
  EXPECT_NE(receipt.long_samples, receipt.short_samples);
  EXPECT_EQ(receipt.long_samples, receipt.repeated_samples);
  EXPECT_TRUE(receipt.valid_logical_pcm);
  EXPECT_TRUE(receipt.checked_idle_after_each_run);
  EXPECT_TRUE(receipt.same_private_state);
  EXPECT_TRUE(receipt.public_initialized_replay);
  EXPECT_TRUE(receipt.canonical_token_failure);
  EXPECT_TRUE(receipt.failed_output_unchanged);
  EXPECT_TRUE(receipt.input_bounds_denied);
  EXPECT_TRUE(receipt.untrusted_option_denied);
  EXPECT_TRUE(receipt.input_refusal_output_unchanged);
  EXPECT_TRUE(receipt.preallocated_length_denied);
  EXPECT_TRUE(receipt.preallocated_refusal_output_unchanged);
#if !defined(DISABLE_SPARSE_TENSORS)
  EXPECT_TRUE(receipt.sparse_descriptors_checked);
  EXPECT_TRUE(receipt.sparse_input_denied);
  EXPECT_TRUE(receipt.sparse_output_denied);
  EXPECT_TRUE(receipt.sparse_refusal_values_retained);
#else
  EXPECT_FALSE(receipt.sparse_descriptors_checked);
#endif
#ifdef ENABLE_STRIDED_TENSORS
  EXPECT_TRUE(receipt.strided_descriptors_checked);
  EXPECT_TRUE(receipt.strided_input_denied);
  EXPECT_TRUE(receipt.strided_output_denied);
  EXPECT_TRUE(receipt.strided_refusal_values_retained);
#else
  EXPECT_FALSE(receipt.strided_descriptors_checked);
#endif
  EXPECT_TRUE(receipt.pointer_output_allocated);
  EXPECT_TRUE(receipt.pointer_empty_wrapper_retained);
  EXPECT_TRUE(receipt.pointer_empty_wrapper_pcm_equal);
  EXPECT_TRUE(receipt.pointer_refusal_unpublished);
  EXPECT_TRUE(receipt.pointer_preallocated_retained);
  EXPECT_TRUE(receipt.pointer_preallocated_pcm_equal);
  EXPECT_TRUE(receipt.pointer_preallocated_canaries_retained);
  EXPECT_TRUE(receipt.first_pcm_retained);
  EXPECT_TRUE(receipt.repeated_pcm_equal);
  EXPECT_TRUE(receipt.original_session_retained);
}
}  // namespace onnxruntime::test
#endif

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime::test {
namespace {
std::array<uint32_t, 18> NativeSuccessPacket(uint64_t samples) {
  std::array<uint32_t, 18> packet{};
  packet[1] = 1;
  packet[2] = uint32_t(samples);
  packet[3] = uint32_t(samples >> 32);
  return packet;
}
}  // namespace
TEST(M0NativeTerminalParser, AcceptsOnlyBoundedAlignedLogicalPcmAndCanonicalStatuses) {
  for (const uint64_t samples : {uint64_t{600}, uint64_t{1200}, uint64_t{61440000}}) {
    const auto result = M0ParseNativeTerminalV1(NativeSuccessPacket(samples), {0, 0}, samples);
    EXPECT_EQ(result.status, 0u);
    EXPECT_EQ(result.samples, samples);
  }
  for (const uint32_t status : {1u, 100u, 101u, 102u, 103u}) {
    std::array<uint32_t, 18> packet{};
    packet[0] = status;
    const auto result = M0ParseNativeTerminalV1(packet, {0, 0}, 0);
    EXPECT_EQ(result.status, status);
    EXPECT_EQ(result.samples, 0u);
  }
}
TEST(M0NativeTerminalParser, RejectsUnknownReservedMalformedAndOutOfCapacityPacketsAs103) {
  for (const uint64_t samples : {uint64_t{0}, uint64_t{1}, uint64_t{599}, uint64_t{601}, uint64_t{61440600}, uint64_t{1} << 32}) {
    const auto result = M0ParseNativeTerminalV1(NativeSuccessPacket(samples), {0, 0}, UINT64_MAX);
    EXPECT_EQ(result.status, 103u);
    EXPECT_EQ(result.samples, 0u);
  }
  EXPECT_EQ(M0ParseNativeTerminalV1(NativeSuccessPacket(1200), {0, 0}, 600).status, 103u);
  auto malformed = NativeSuccessPacket(600);
  malformed[1] = 2;
  EXPECT_EQ(M0ParseNativeTerminalV1(malformed, {0, 0}, 600).status, 103u);
  for (size_t slot = 4; slot < malformed.size(); ++slot) {
    auto packet = NativeSuccessPacket(600);
    packet[slot] = 1;
    EXPECT_EQ(M0ParseNativeTerminalV1(packet, {0, 0}, 600).status, 103u) << "slot=" << slot;
  }
  for (const uint32_t status : {2u, 99u, 104u, UINT32_MAX}) {
    std::array<uint32_t, 18> packet{};
    packet[0] = status;
    EXPECT_EQ(M0ParseNativeTerminalV1(packet, {0, 0}, 600).status, 103u);
  }
  for (const uint32_t status : {1u, 100u, 101u, 102u}) {
    for (size_t slot = 1; slot < malformed.size(); ++slot) {
      std::array<uint32_t, 18> packet{};
      packet[0] = status;
      packet[slot] = 1;
      EXPECT_EQ(M0ParseNativeTerminalV1(packet, {0, 0}, 600).status, 103u);
    }
  }
}
TEST(M0NativeTerminalParser, OriginalTokenFailureHasCanonical103PrecedenceAndNeverPublishesSamples) {
  const std::array<std::array<uint32_t, 2>, 6> cases{{{103, 0}, {0, 103}, {103, 103}, {1, 0}, {0, 100}, {UINT32_MAX, 0}}};
  for (const auto& tokens : cases) {
    auto result = M0ParseNativeTerminalV1(NativeSuccessPacket(1200), tokens, 1200);
    EXPECT_EQ(result.status, 103u);
    EXPECT_EQ(result.samples, 0u);
    for (const uint32_t status : {1u, 100u, 101u, 102u, 103u}) {
      std::array<uint32_t, 18> packet{};
      packet[0] = status;
      result = M0ParseNativeTerminalV1(packet, tokens, 1200);
      EXPECT_EQ(result.status, 103u);
      EXPECT_EQ(result.samples, 0u);
    }
  }
}
}  // namespace onnxruntime::test
#endif
