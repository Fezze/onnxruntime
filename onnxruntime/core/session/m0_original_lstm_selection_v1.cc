// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/m0_original_lstm_selection_v1.h"

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include <atomic>
#include <array>
#include <limits>
#include "core/common/profiler.h"
#include "core/graph/model.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/session_state.h"
#include "core/framework/m0_fullgpu_preinit_admission.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "core/session/m0_original_lstm_module_v1.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"

namespace onnxruntime {
namespace {
[[maybe_unused]] uint64_t NextSelectionGeneration() {
  static std::atomic<uint64_t> next{1};
  auto value = next.load();
  do {
    ORT_ENFORCE(value && value != std::numeric_limits<uint64_t>::max(), "RW07_LSTM_SELECTION_GENERATIONS_EXHAUSTED");
  } while (!next.compare_exchange_weak(value, value + 1));
  return value;
}
[[maybe_unused]] bool SameBinding(const M0PluginKernelCreatorBindingV1& expected, const KernelCreateInfo& candidate) {
  M0PluginKernelCreatorBindingV1 actual;
  return expected.callback && expected.state && M0InspectPluginKernelCreatorV1(candidate, actual) &&
         actual.callback == expected.callback && actual.state == expected.state;
}
[[maybe_unused]] void CheckDefinition(const KernelDef& definition) {
  ORT_ENFORCE(definition.OpName() == "LSTM" && definition.Domain().empty() &&
                  definition.Provider() == kWebGpuExecutionProvider &&
                  definition.SinceVersion() == std::make_pair(14, std::numeric_limits<int>::max()) &&
                  definition.MayInplace().empty() && definition.Alias().empty() && !definition.VariadicAlias() &&
                  !definition.HasExternalOutputs() && !definition.AllocateInputsContiguously() &&
                  definition.ExecQueueId() == 0 && definition.InputMemoryTypeArgs().empty() &&
                  definition.OutputMemoryTypeArgs().empty() &&
                  definition.DefaultInputMemoryType() == OrtMemTypeDefault &&
                  definition.DefaultOutputMemoryType() == OrtMemTypeDefault,
              "RW07_LSTM_SELECTED_DEFINITION");
  const auto& types = definition.TypeConstraints();
  ORT_ENFORCE(types.size() == 2 && types.count("T") && types.count("T1") &&
                  types.at("T") == std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()} &&
                  types.at("T1") == std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>()},
              "RW07_LSTM_SELECTED_TYPES");
}
}  // namespace

M0OriginalLstmSelectionReceiptV1 M0OriginalLstmSelectionV1::Capture(InferenceSession& session) {
#if defined(__ANDROID__)
  std::lock_guard<std::mutex> lock(session.session_mutex_);
  ORT_ENFORCE(session.m0_fullgpu_strict_requested_v1_ && session.is_model_loaded_ &&
                  session.m0_original_load_attempted_v1_ && !session.is_inited_ && !session.session_state_ &&
                  session.model_ && session.execution_providers_.NumProviders() == 0 &&
                  session.custom_registries_.empty() && session.custom_schema_registries_.empty(),
              "RW07_LSTM_SELECTION_SESSION");
  ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
  const auto cancellation = session.session_options_.load_cancellation_flag;
  const auto environment_lease = OrtEnv::TryGetInstance();
  ORT_ENFORCE(environment_lease && &environment_lease->GetEnvironment() == &session.environment_,
              "RW07_LSTM_SELECTION_ENVIRONMENT_IDENTITY");
  auto retained_model = session.model_;
  auto* owner = retained_model->m0_original_preload_owner_v1_.get();
  ORT_ENFORCE(owner, "RW07_LSTM_SELECTION_WHOLE_OWNER");
  const auto generation = NextSelectionGeneration();
  owner->BeginSelectedLstmConstructorsV1(generation);
  M0OriginalLstmSelectionReceiptV1 receipt;
  try {
    {
      auto module = M0OriginalLstmModuleV1::Open(*session.session_logger_);
      auto registry = module->Registry();
      ORT_ENFORCE(registry && module->Provider().Type() == kWebGpuExecutionProvider,
                  "RW07_LSTM_SELECTION_TRUSTED_PROVIDER");
      const auto expected = module->Binding();
      KernelRegistryManager manager;
      ORT_THROW_IF_ERROR(manager.RegisterKernels(module->Providers()));
      // Avoid all provider allocator callbacks in SessionState's constructor.
      AllocatorMap allocators, initializer_allocators;
      DataTransferManager transfers;
      ExternalDataLoaderManager external_loaders;
      profiling::Profiler profiler;
      SessionOptions options;
      options.enable_mem_pattern = false;
      options.enable_cpu_mem_arena = false;
      options.enable_profiling = false;
      options.use_per_session_threads = false;
      options.graph_optimization_level = TransformerLevel::Default;
      ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("session.disable_prepacking", "1"));
      ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
      SessionState state(retained_model->MainGraph(), module->Providers(), nullptr, nullptr,
                         transfers, external_loaders, *session.session_logger_, profiler, options, nullptr,
                         &allocators, &initializer_allocators);
      const std::array<size_t, 6> indices{449, 1260, 1274, 1288, 1303, 1340};
      for (size_t ordinal = 0; ordinal < indices.size(); ++ordinal) {
        const auto* actual = retained_model->MainGraph().GetNode(indices[ordinal]);
        ORT_ENFORCE(actual && actual->Index() == indices[ordinal], "RW07_LSTM_FIXED_NODE_BINDING");
        const auto& node = *actual;
        const KernelCreateInfo* selected = nullptr;
        ORT_THROW_IF_ERROR(registry->TryFindKernel(node, kWebGpuExecutionProvider,
                                                   manager.GetKernelTypeStrResolver(), *session.session_logger_, &selected));
        ORT_ENFORCE(selected && selected->kernel_def && state.kernel_create_info_map_.size() == ordinal,
                    "RW07_LSTM_SELECTION_NATIVE_LOOKUP");
        ORT_ENFORCE(state.kernel_create_info_map_.emplace(node.Index(), gsl::not_null<const KernelCreateInfo*>(selected)).second,
                    "RW07_LSTM_SELECTION_STORE");
        const auto* definition = selected->kernel_def.get();
        auto revalidate = [&]() {
          ORT_ENFORCE(session.model_.get() == retained_model.get() && owner->selection_active_v1_ &&
                          owner->selection_generation_v1_ == generation && !owner->poisoned_ && !owner->active_ &&
                          &state.graph_ == &retained_model->MainGraph() && state.kernel_create_info_map_.size() == ordinal + 1 &&
                          state.kernel_create_info_map_.at(node.Index()).get() == selected && selected->kernel_def.get() == definition &&
                          state.session_kernels_.empty() && state.initialized_tensors_.empty() &&
                          state.constant_initialized_tensors_.empty() && state.outer_scope_augmented_constant_tensors_.empty() &&
                          state.outer_scope_parent_only_indices_.empty() && !state.outer_scope_augmented_map_built_ &&
                          state.ort_value_name_idx_map_.Size() == 0 && !state.graph_viewer_ && !state.p_seq_exec_plan_ &&
                          allocators.empty() && initializer_allocators.empty() && node.GetExecutionProviderType().empty(),
                      "RW07_LSTM_SELECTION_LIVE_BINDING");
          // Validation only: no Resolve stage/event is reactivated.
          ORT_ENFORCE(owner->stage_ == 7, "RW07_LSTM_CONSTRUCTOR_E_STAGE");
          owner->active_ = true;
          owner->CheckNode(node);
          owner->active_ = false;
          ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
          ORT_ENFORCE(session.session_options_.load_cancellation_flag == cancellation,
                      "RW07_LSTM_SELECTION_CANCELLATION_IDENTITY");
          CheckDefinition(*definition);
          ORT_ENFORCE(SameBinding(expected, *selected), "RW07_LSTM_SELECTION_NATIVE_CREATOR");
        };
        revalidate();
        ++receipt.selected;
        // These candidates pass through the real native plugin-functor inspector.
        // No callback is invoked, even when the source-owned fault candidate has one.
        if (ordinal == 0) {
          size_t foreign_calls = 0;
          const OrtKernelCreateFunc foreign_callback = [](void* value, const OrtKernelInfo*, OrtKernelImpl**) -> OrtStatus* {
            ++*static_cast<size_t*>(value);
            return nullptr;
          };
          for (const auto binding : {M0PluginKernelCreatorBindingV1{nullptr, expected.state},
                                     M0PluginKernelCreatorBindingV1{foreign_callback, &foreign_calls},
                                     M0PluginKernelCreatorBindingV1{expected.callback, &foreign_calls}}) {
            auto candidate = MakePluginEpKernelCreateInfo(definition, binding.callback, binding.state);
            ORT_ENFORCE(!SameBinding(expected, candidate), "RW07_LSTM_SELECTION_ACCEPTED_FOREIGN_BINDING");
            ++receipt.rejected_binding_mutations;
          }
          KernelCreateInfo empty;
          M0PluginKernelCreatorBindingV1 sentinel{expected.callback, &foreign_calls};
          ORT_ENFORCE(!M0InspectPluginKernelCreatorV1(empty, sentinel) && sentinel.callback == expected.callback &&
                          sentinel.state == &foreign_calls,
                      "RW07_LSTM_SELECTION_EMPTY_FUNCTOR");
          ++receipt.rejected_foreign_functors;
          empty.kernel_create_func = [&foreign_calls](FuncManager&, const OpKernelInfo&, std::unique_ptr<OpKernel>&) {
            ++foreign_calls;
            return Status::OK();
          };
          ORT_ENFORCE(!M0InspectPluginKernelCreatorV1(empty, sentinel) && sentinel.callback == expected.callback &&
                          sentinel.state == &foreign_calls && foreign_calls == 0,
                      "RW07_LSTM_SELECTION_FOREIGN_FUNCTOR");
          ++receipt.rejected_foreign_functors;
        }
        revalidate();
        std::unique_ptr<OpKernel> constructed;
        ORT_THROW_IF_ERROR(manager.CreateKernel(node, module->Provider(), state,
                                                state.GetNodeKernelCreateInfo(node.Index()), constructed));
        ORT_ENFORCE(constructed && &constructed->Node() == &node && &constructed->KernelDef() == definition &&
                        constructed->Info().GetExecutionProvider() == &module->Provider(),
                    "RW07_LSTM_CONSTRUCTOR_BINDING");
        ++receipt.constructed;
        revalidate();
        constructed.reset();
        ++receipt.released;
        revalidate();
      }
    }
    // All registry/EP/device/module consumers have retired before checking E again.
    ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
    ORT_ENFORCE(session.session_options_.load_cancellation_flag == cancellation, "RW07_LSTM_SELECTION_CANCELLATION_IDENTITY");
    owner->FinishSelectedLstmConstructorsV1(generation);
    ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
    ORT_ENFORCE(session.session_options_.load_cancellation_flag == cancellation, "RW07_LSTM_SELECTION_CANCELLATION_IDENTITY");
    ++receipt.retired;
  } catch (...) {
    owner->AbortSelectedLstmConstructorsV1();
    throw;
  }
  return receipt;
#else
  ORT_UNUSED_PARAMETER(session);
  ORT_THROW("RW07_LSTM_SELECTION_PLATFORM_UNQUALIFIED");
#endif
}
M0OriginalLstmSelectionReceiptV1 M0QualifyOriginalLstmSelectionV1(InferenceSession& session) {
  auto receipt = M0OriginalLstmSelectionV1::Capture(session);
  bool refused = false;
  try {
    M0OriginalLstmSelectionV1::Capture(session);
  } catch (const std::exception& error) {
    refused = std::string(error.what()).find("RW07_LSTM_SELECTION_OWNER_OR_REUSE") != std::string::npos;
  }
  ORT_ENFORCE(refused, "RW07_LSTM_SELECTION_REPLAY_ACCEPTED");
  return receipt;
}
}  // namespace onnxruntime
#endif
