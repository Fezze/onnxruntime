// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/m0_native_lowering_qualification_v1.h"
#include "core/common/m0_capacity_ceiling_v1.h"

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include "core/session/m0_native_public_input_fixture_v1.h"
#include <algorithm>
#include <cstring>
#include <gsl/gsl>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>
#include "core/common/common.h"
#include "core/graph/model.h"
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"
#include "core/session/inference_session.h"
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#include "core/session/m0_native_gpu_module_v2.h"
#include "core/session/ort_env.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_execution_plan.h"
#if !defined(DISABLE_SPARSE_TENSORS)
#include "core/framework/sparse_tensor.h"
#endif
#include "core/session/m0_fullgpu_native_execution_state_v1.h"
#endif

namespace onnxruntime {
common::Status M0NativeLoweringQualificationV1::Prepare(InferenceSession& session,
    M0NativeLoweringCapacityV1 requested_capacity, M0NativeLoweringReceiptV1* receipt) {
  uint32_t capacity = 0;
  switch (requested_capacity) {
    case M0NativeLoweringCapacityV1::k1024: capacity = 1024; break;
    case M0NativeLoweringCapacityV1::k2048: capacity = 2048; break;
    case M0NativeLoweringCapacityV1::kZero: capacity = 0; break;
    case M0NativeLoweringCapacityV1::kAboveMaximum: capacity = 102401; break;
    default: return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "RW07_NATIVE_QUALIFICATION_CAPACITY_CASE");
  }
  const Model* source = nullptr;
  std::vector<const NodeArg*> inputs, outputs;
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    source = session.model_.get();
    if (source) {
      inputs = source->MainGraph().GetInputs();
      outputs = source->MainGraph().GetOutputs();
    }
  }
  ORT_RETURN_IF_ERROR(session.M0PrepareOriginalLoweringV1(capacity));
  if (!receipt) return common::Status::OK();
  std::lock_guard<std::mutex> lock(session.session_mutex_);
  const auto* native = session.m0_native_lowering_v1_.get();
  ORT_RETURN_IF_NOT(native && native->source_ == source && native->session_ == &session &&
                        native->capacity_ == capacity && session.model_.get() == source &&
                        !session.is_inited_ && !session.session_state_, "RW07_NATIVE_QUALIFICATION_OWNER");
  M0NativeLoweringReceiptV1 pending;
  pending.original_nodes = size_t(source->MainGraph().NumberOfNodes());
  for (const auto& row : native->coverage_) {
    if (!row.original_wire.empty() && !row.recipes.empty() && !row.consumers.empty()) ++pending.covered_nodes;
  }
  pending.derived_nodes = size_t(native->derived_.graph().node_size());
  pending.original_model_retained = session.model_.get() == source;
  pending.original_ports_retained = inputs == source->MainGraph().GetInputs() && outputs == source->MainGraph().GetOutputs();
  pending.initializer_payloads_transferred = native->initializer_payloads_transferred_;
  pending.transferred_initializer_raw_bytes = native->transferred_initializer_raw_bytes_;
  for(const auto& tensor:native->derived_.graph().initializer())
    pending.retained_derived_raw_bytes += tensor.raw_data().size();
  // Full wire materialization is qualification-only and requires an explicit
  // receipt. The application retains metadata plus sealed payload references.
  ONNX_NAMESPACE::ModelProto evidence_proto(native->derived_);
  for(int i=0;i<evidence_proto.graph().initializer_size();++i)
    evidence_proto.mutable_graph()->mutable_initializer(i)->CopyFrom(
        native->ExpectedInitializer(native->derived_.graph().initializer(i)));
  pending.derived_model_wire = evidence_proto.SerializeAsString();
  pending.derived_sha256 = native->derived_sha256_;
  pending.ledger_sha256 = native->ledger_sha256_;
  *receipt = std::move(pending);
  return common::Status::OK();
}
M0NativeDescriptorReceiptV1 M0NativeLoweringQualificationV1::Descriptors(
    InferenceSession& session, M0NativeDescriptorFaultV1 fault) {
  ORT_THROW_IF_ERROR(session.M0PrepareOriginalLoweringV1(1024));
  M0NativeDescriptorReceiptV1 receipt;
  NodeArg* changed_arg = nullptr;
  Node* changed_node = nullptr;
  ONNX_NAMESPACE::TypeProto saved_type;
  ONNX_NAMESPACE::AttributeProto saved_attribute;
  std::unique_ptr<NodeArg> substitute;
  const NodeArg* saved_output = nullptr;
  bool channel_denied = false;
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    auto& native = *session.m0_native_lowering_v1_;
    native.ValidateExecutionModel();
    auto& graph = native.execution_model_->MainGraph();
    receipt.nodes = size_t(graph.NumberOfNodes());
    receipt.separate_model = native.execution_model_.get() != session.model_.get();
    receipt.no_inferred_values = true;
    for (const auto& item : native.descriptors_) {
      const auto& d = item.second;
      receipt.checked += d.checked;
      receipt.selected += d.selected;
      receipt.inferred += d.inferred;
      receipt.propagated += d.propagated;
      receipt.published += d.published;
      receipt.defaults += d.defaults.size();
      for (const bool saved : d.saved) receipt.saved_outputs += saved;
    }
    for (const auto& node : graph.Nodes()) {
      for (const auto* arg : node.OutputDefs()) {
        if (arg->Exists()) ++receipt.present_outputs;
        if (arg->Exists() && (arg->GetInferredShapeValues().has_value() || arg->GetInferredShapeScalarValue().has_value()))
          receipt.no_inferred_values = false;
      }
    }
    if (fault == M0NativeDescriptorFaultV1::None) return receipt;
    if (fault == M0NativeDescriptorFaultV1::InputData) {
      channel_denied = !native.execution_model_->M0CheckCallbackPolicyV1(M0CallbackPolicyV1::Channel::kInputData).IsOK();
    } else if (fault == M0NativeDescriptorFaultV1::LateResolve) {
      channel_denied = !graph.Resolve().IsOK();
    } else if (fault == M0NativeDescriptorFaultV1::OriginalTokenType) {
      changed_arg = const_cast<NodeArg*>(session.model_->MainGraph().GetNodeArg("tokens"));
      saved_type = *changed_arg->TypeAsProto();
      auto changed = saved_type;
      changed.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
      ORT_THROW_IF_ERROR(changed_arg->UpdateTypeAndShape(changed, true, true, graph.GetLogger()));
    } else if (fault == M0NativeDescriptorFaultV1::LiveDefaultValue ||
               fault == M0NativeDescriptorFaultV1::CopiedDefaultLateCallback) {
      for (const auto& item : native.descriptors_) {
        if (!item.second.defaults.empty()) {
          for (auto& node : graph.Nodes()) {
            if (node.Name() == item.first) { changed_node = &node; break; }
          }
          saved_attribute = changed_node->GetAttributes().at(*item.second.defaults.begin());
          break;
        }
      }
      ORT_ENFORCE(changed_node, "RW07_NATIVE_QUALIFICATION_REAL_DEFAULT_REQUIRED");
      if (fault == M0NativeDescriptorFaultV1::CopiedDefaultLateCallback) {
        try {
          native.Defaulted(*changed_node, saved_attribute);
        } catch (const std::exception&) {
          channel_denied = true;
        }
      } else {
        auto changed = saved_attribute;
        if (changed.type() == ONNX_NAMESPACE::AttributeProto::INT) changed.set_i(changed.i() ^ 1);
        else if (changed.type() == ONNX_NAMESPACE::AttributeProto::FLOAT) changed.set_f(changed.f() + 1.0f);
        else ORT_THROW("RW07_NATIVE_QUALIFICATION_SUPPORTED_DEFAULT_REQUIRED");
        changed_node->AddAttributeProto(std::move(changed));
      }
    } else {
      const auto* original_lstm = session.model_->MainGraph().GetNode(449);
      ORT_ENFORCE(original_lstm && original_lstm->OutputDefs().size() == 3, "RW07_NATIVE_QUALIFICATION_REAL_LSTM_REQUIRED");
      for (auto& node : graph.Nodes()) {
        if (node.Name() == original_lstm->Name()) { changed_node = &node; break; }
      }
      ORT_ENFORCE(changed_node, "RW07_NATIVE_QUALIFICATION_RETAINED_LSTM_REQUIRED");
      changed_arg = const_cast<NodeArg*>(changed_node->OutputDefs()[1]);
      saved_type = *changed_arg->TypeAsProto();
      if (fault == M0NativeDescriptorFaultV1::UnusedLstmOutputType) {
        auto changed = saved_type;
        changed.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
        ORT_THROW_IF_ERROR(changed_arg->UpdateTypeAndShape(changed, true, true, graph.GetLogger()));
      } else if (fault == M0NativeDescriptorFaultV1::UnusedLstmShapeData) {
        changed_arg->SetInferredShapeScalarValue(7);
      } else if (fault == M0NativeDescriptorFaultV1::UnusedLstmOutputSubstitution) {
        substitute = std::make_unique<NodeArg>(changed_arg->Name(), &saved_type);
        saved_output = changed_arg;
        changed_node->MutableOutputDefs()[1] = substitute.get();
      } else {
        ORT_THROW("RW07_NATIVE_QUALIFICATION_UNKNOWN_DESCRIPTOR_FAULT");
      }
    }
  }
  const auto refuses = [&]() {
    try {
      return !session.M0PrepareOriginalLoweringV1(1024).IsOK();
    } catch (const std::exception&) {
      return true;
    }
  };
  const bool cache_denied = refuses();
  receipt.fault_denied = cache_denied &&
      ((fault != M0NativeDescriptorFaultV1::InputData && fault != M0NativeDescriptorFaultV1::LateResolve &&
        fault != M0NativeDescriptorFaultV1::CopiedDefaultLateCallback) || channel_denied);
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    auto& native = *session.m0_native_lowering_v1_;
    receipt.retired = native.phase_ == M0FullGpuNativeLoweringOwnerV1::Phase::Poisoned;
    auto& graph = native.execution_model_->MainGraph();
    if (changed_arg) {
      if (fault == M0NativeDescriptorFaultV1::UnusedLstmShapeData) changed_arg->ClearInferredShapeScalarValue();
      else if (fault == M0NativeDescriptorFaultV1::UnusedLstmOutputSubstitution)
        changed_node->MutableOutputDefs()[1] = const_cast<NodeArg*>(saved_output);
      else ORT_THROW_IF_ERROR(changed_arg->UpdateTypeAndShape(saved_type, true, true, graph.GetLogger()));
    }
    if (fault == M0NativeDescriptorFaultV1::LiveDefaultValue) changed_node->AddAttributeProto(saved_attribute);
  }
  receipt.restored_replay_denied = refuses();
  return receipt;
}
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
M0NativeRunReceiptV1 M0NativeLoweringQualificationV1::RunSequence(InferenceSession& session) {
  // Exercise the app's device-selected physical capacity. Explicit1024/2048
  // descriptor and initialization cases remain separate qualification coverage;
  // forcing1024 here can exceed the production device admission's work/memory.
  ORT_THROW_IF_ERROR(session.M0PrepareOriginalDeviceLoweringV1());
  auto& native = *session.m0_native_lowering_v1_;
  ORT_ENFORCE(native.device_capacity_&&native.capacity_>0,"RW07_NATIVE_RUN_QUALIFICATION_DEVICE_CAPACITY");
  ORT_THROW_IF_ERROR(native.InitializeExecution(session));
  const auto* holder = native.execution_plan_.get();
  ORT_THROW_IF_ERROR(session.Initialize());
  const bool public_initialized = session.IsInitialized() && native.execution_plan_.get() == holder;
  const auto* state = holder->state.get();
  const auto* plan = state->GetExecutionPlan();
  const auto* original = session.model_.get();
  const auto original_inputs = original->MainGraph().GetInputs();
  const auto original_outputs = original->MainGraph().GetOutputs();
  const auto allocator = std::make_shared<CPUAllocator>();
  const std::array<std::string, 3> names{"tokens", "style", "speed"};
  const std::array<std::string, 1> outputs{native.outputs_.pcm.data};
  RunOptions options;
  const auto inputs=[&](bool short_input,bool invalid_token){
    return M0NativePublicInputFixtureV1(allocator,short_input,invalid_token);
  };
  M0NativeRunReceiptV1 receipt;
  receipt.device_selected_capacity=native.device_capacity_;
  receipt.physical_capacity=native.capacity_;
  receipt.public_initialized_replay = public_initialized;
  receipt.checked_idle_after_each_run = receipt.same_private_state = true;
  const auto idle = [&]() {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    native.ValidateInitializedExecution();
    receipt.checked_idle_after_each_run = receipt.checked_idle_after_each_run &&
        !state->M0FullGpuExtentRunIdV2() && !state->M0FullGpuRunOwnerV2().Poisoned();
    receipt.same_private_state = receipt.same_private_state && native.execution_plan_.get() == holder &&
        holder->state.get() == state && holder->plan == plan && state->GetExecutionPlan() == plan;
  };
  const auto valid_run = [&](bool short_input, bool pointer_api = false) {
    const auto feeds = inputs(short_input, false);
    std::vector<OrtValue> result;
    if (pointer_api) {
      // This is the pointer/ownership adapter used by the actual public C API.
      const std::array<const char*, 3> input_names{"tokens", "style", "speed"};
      const std::array<const OrtValue*, 3> input_values{&feeds[0], &feeds[1], &feeds[2]};
      const std::array<const char*, 1> output_names{outputs[0].c_str()};
      std::array<OrtValue*, 1> output_values{nullptr};
      const auto status = session.Run(options, input_names, input_values, output_names, output_values);
      std::unique_ptr<OrtValue> published(output_values[0]);
      ORT_THROW_IF_ERROR(status);
      ORT_ENFORCE(published && published->IsTensor(), "RW07_NATIVE_POINTER_OUTPUT_MISSING");
      result.push_back(*published);
      receipt.pointer_output_allocated = true;
    } else {
      ORT_THROW_IF_ERROR(session.Run(options, names, feeds, outputs, &result));
    }
    ORT_ENFORCE(result.size() == 1 && result[0].IsTensor(), "RW07_NATIVE_RUN_QUALIFICATION_PCM_RESULT");
    const auto& tensor = result[0].Get<Tensor>();
    ORT_ENFORCE(tensor.Location().device == OrtDevice() && tensor.IsDataType<float>() &&
                    tensor.Shape().NumDimensions() == 1 && tensor.Shape()[0] > 0 &&
                    uint64_t(tensor.Shape()[0]) <= uint64_t(native.capacity_) * 600 && tensor.Shape()[0] % 600 == 0,
                "RW07_NATIVE_RUN_QUALIFICATION_LOGICAL_PCM");
    idle();
    return result[0];
  };
  const OrtValue first = valid_run(false);
  const auto& first_tensor = first.Get<Tensor>();
  receipt.long_samples = size_t(first_tensor.Shape()[0]);
  const std::string first_bytes(static_cast<const char*>(first_tensor.DataRaw()), first_tensor.SizeInBytes());
  const OrtValue second = valid_run(true);
  receipt.short_samples = size_t(second.Get<Tensor>().Shape()[0]);
  const auto invalid = inputs(true, true);
  std::vector<OrtValue> refused(1);
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{1}, allocator, refused[0]);
  refused[0].GetMutable<Tensor>()->MutableData<float>()[0] = 777.0f;
  const auto* sentinel = refused[0].Get<Tensor>().DataRaw();
  const auto failure = session.Run(options, names, invalid, outputs, &refused);
  receipt.canonical_token_failure = !failure.IsOK() && failure.ErrorMessage() == "RW07_NATIVE_GPU_STATUS_103";
  receipt.failed_output_unchanged = refused.size() == 1 && refused[0].IsTensor() &&
      refused[0].Get<Tensor>().DataRaw() == sentinel && refused[0].Get<Tensor>().Shape() == TensorShape{1} &&
      refused[0].Get<Tensor>().Data<float>()[0] == 777.0f;
  idle();
  auto oversized = inputs(true, false);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape({1, 513}), allocator, oversized[0]);
  std::memset(oversized[0].GetMutable<Tensor>()->MutableDataRaw(), 0, 513 * sizeof(int64_t));
  const auto bad_bound = session.Run(options, names, oversized, outputs, &refused);
  receipt.input_bounds_denied = !bad_bound.IsOK() &&
      bad_bound.ErrorMessage().find("RW07_NATIVE_RUN_TOKEN_DESCRIPTOR_BOUND") != std::string::npos;
  RunOptions untrusted;
  ORT_THROW_IF_ERROR(untrusted.config_options.AddConfigEntry("m0.fullgpu.actual_run_v2", "1"));
  const auto valid_inputs = inputs(true, false);
  const auto bad_option = session.Run(untrusted, names, valid_inputs, outputs, &refused);
  receipt.untrusted_option_denied = !bad_option.IsOK() &&
      bad_option.ErrorMessage().find("RW07_NATIVE_RUN_EXTERNAL_AUTHORITY") != std::string::npos;
  receipt.input_refusal_output_unchanged = refused.size() == 1 && refused[0].IsTensor() &&
      refused[0].Get<Tensor>().DataRaw() == sentinel && refused[0].Get<Tensor>().Shape() == TensorShape{1} &&
      refused[0].Get<Tensor>().Data<float>()[0] == 777.0f;
  idle();
  const auto bad_preallocated = session.Run(options, names, valid_inputs, outputs, &refused);
  receipt.preallocated_length_denied = !bad_preallocated.IsOK() &&
      bad_preallocated.ErrorMessage().find("RW07_NATIVE_RUN_PREALLOCATED_PCM_DESCRIPTOR") != std::string::npos;
  receipt.preallocated_refusal_output_unchanged = refused.size() == 1 && refused[0].IsTensor() &&
      refused[0].Get<Tensor>().DataRaw() == sentinel && refused[0].Get<Tensor>().Shape() == TensorShape{1} &&
      refused[0].Get<Tensor>().Data<float>()[0] == 777.0f;
  idle();
#if !defined(DISABLE_SPARSE_TENSORS)
  // Real public CPU sparse values exercise refusal before the generic validator
  // can access the deliberately absent ordinary SessionState.
  auto sparse_inputs = inputs(true, false);
  SparseTensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape({1, 3}), allocator, sparse_inputs[0]);
  sparse_inputs[0].GetMutable<SparseTensor>()->MakeCooData(0, 0);
  const auto* sparse_input = &sparse_inputs[0].Get<SparseTensor>();
  const auto bad_sparse_input = session.Run(options, names, sparse_inputs, outputs, &refused);
  receipt.sparse_input_denied = !bad_sparse_input.IsOK() &&
      bad_sparse_input.ErrorMessage().find("RW07_NATIVE_RUN_EXTERNAL_INPUT_DESCRIPTOR") != std::string::npos;
  std::vector<OrtValue> sparse_outputs(1);
  SparseTensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{1}, allocator, sparse_outputs[0]);
  sparse_outputs[0].GetMutable<SparseTensor>()->MakeCooData(0, 0);
  const auto* sparse_output = &sparse_outputs[0].Get<SparseTensor>();
  const auto bad_sparse_output = session.Run(options, names, valid_inputs, outputs, &sparse_outputs);
  receipt.sparse_output_denied = !bad_sparse_output.IsOK() &&
      bad_sparse_output.ErrorMessage().find("RW07_NATIVE_RUN_EXTERNAL_OUTPUT_DESCRIPTOR") != std::string::npos;
  receipt.sparse_descriptors_checked = true;
  receipt.sparse_refusal_values_retained = &sparse_inputs[0].Get<SparseTensor>() == sparse_input &&
      sparse_outputs.size() == 1 && sparse_outputs[0].IsSparseTensor() &&
      &sparse_outputs[0].Get<SparseTensor>() == sparse_output &&
      refused[0].Get<Tensor>().DataRaw() == sentinel && refused[0].Get<Tensor>().Data<float>()[0] == 777.0f;
  idle();
#endif
#ifdef ENABLE_STRIDED_TENSORS
  // Genuine CPU views with interleaved storage. The logical token sequence is
  // valid, so this refusal exercises the byte-copy boundary rather than a GPU
  // token/status failure. Both borrowed backing arrays outlive the calls.
  std::array<int64_t, 6> token_storage{0, 999999, 43, 999999, 0, 999999};
  const auto original_token_storage = token_storage;
  const std::array<int64_t, 2> token_strides{6, 2};
  auto strided_inputs = inputs(true, false);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape({1, 3}), token_storage.data(),
                       allocator->Info(), strided_inputs[0], 0, token_strides);
  ORT_ENFORCE(!strided_inputs[0].Get<Tensor>().IsContiguous(), "RW07_NATIVE_STRIDED_INPUT_FIXTURE");
  const auto bad_strided_input = session.Run(options, names, strided_inputs, outputs, &refused);
  receipt.strided_input_denied = !bad_strided_input.IsOK() &&
      bad_strided_input.ErrorMessage().find("RW07_NATIVE_RUN_STRIDED_INPUT") != std::string::npos;
  std::array<float, 4> output_storage{777.0f, 888.0f, 777.0f, 888.0f};
  const auto original_output_storage = output_storage;
  const std::array<int64_t, 1> output_strides{2};
  std::vector<OrtValue> strided_outputs(1);
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{2}, output_storage.data(),
                       allocator->Info(), strided_outputs[0], 0, output_strides);
  ORT_ENFORCE(!strided_outputs[0].Get<Tensor>().IsContiguous(), "RW07_NATIVE_STRIDED_OUTPUT_FIXTURE");
  const auto bad_strided_output = session.Run(options, names, valid_inputs, outputs, &strided_outputs);
  receipt.strided_output_denied = !bad_strided_output.IsOK() &&
      bad_strided_output.ErrorMessage().find("RW07_NATIVE_RUN_STRIDED_OUTPUT") != std::string::npos;
  receipt.strided_descriptors_checked = true;
  receipt.strided_refusal_values_retained = token_storage == original_token_storage &&
      output_storage == original_output_storage &&
      strided_inputs[0].Get<Tensor>().DataRaw() == token_storage.data() &&
      strided_outputs.size() == 1 && strided_outputs[0].IsTensor() &&
      strided_outputs[0].Get<Tensor>().DataRaw() == output_storage.data() &&
      refused[0].Get<Tensor>().DataRaw() == sentinel && refused[0].Get<Tensor>().Data<float>()[0] == 777.0f;
  idle();
#endif
  const std::array<const char*, 3> pointer_names{"tokens", "style", "speed"};
  const std::array<const char*, 1> pointer_outputs{outputs[0].c_str()};
  const std::array<const OrtValue*, 3> pointer_invalid{&invalid[0], &invalid[1], &invalid[2]};
  std::array<OrtValue*, 1> refused_pointer{nullptr};
  const auto pointer_failure = session.Run(options, pointer_names, pointer_invalid, pointer_outputs, refused_pointer);
  // Even unexpected publication is owned and released by this qualification
  // caller; failed Run must leave the caller's null output pointer unchanged.
  std::unique_ptr<OrtValue> unexpected_output(refused_pointer[0]);
  receipt.pointer_refusal_unpublished = !pointer_failure.IsOK() && !refused_pointer[0] &&
      pointer_failure.ErrorMessage() == "RW07_NATIVE_GPU_STATUS_103";
  idle();
  std::vector<float> pcm_storage(receipt.short_samples + 2, 777.0f);
  OrtValue preallocated;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{int64_t(receipt.short_samples)},
                       pcm_storage.data() + 1, allocator->Info(), preallocated);
  const std::array<const OrtValue*, 3> pointer_valid{&valid_inputs[0], &valid_inputs[1], &valid_inputs[2]};
  std::array<OrtValue*, 1> pointer_preallocated{&preallocated};
  ORT_THROW_IF_ERROR(session.Run(options, pointer_names, pointer_valid, pointer_outputs, pointer_preallocated));
  receipt.pointer_preallocated_retained = pointer_preallocated[0] == &preallocated &&
      preallocated.Get<Tensor>().DataRaw() == pcm_storage.data() + 1;
  receipt.pointer_preallocated_pcm_equal = preallocated.Get<Tensor>().SizeInBytes() == second.Get<Tensor>().SizeInBytes() &&
      std::memcmp(preallocated.Get<Tensor>().DataRaw(), second.Get<Tensor>().DataRaw(), second.Get<Tensor>().SizeInBytes()) == 0;
  receipt.preallocated_difference=M0MeasureNativePcmDifferenceV1(second.Get<Tensor>().DataRaw(),second.Get<Tensor>().SizeInBytes(),
      preallocated.Get<Tensor>().DataRaw(),preallocated.Get<Tensor>().SizeInBytes());
  receipt.pointer_preallocated_canaries_retained = pcm_storage.front() == 777.0f && pcm_storage.back() == 777.0f;
  idle();
  OrtValue empty_wrapper;
  std::array<OrtValue*, 1> pointer_empty{&empty_wrapper};
  ORT_THROW_IF_ERROR(session.Run(options, pointer_names, pointer_valid, pointer_outputs, pointer_empty));
  receipt.pointer_empty_wrapper_retained = pointer_empty[0] == &empty_wrapper && empty_wrapper.IsTensor();
  receipt.pointer_empty_wrapper_pcm_equal = empty_wrapper.IsTensor() &&
      empty_wrapper.Get<Tensor>().Shape() == second.Get<Tensor>().Shape() &&
      empty_wrapper.Get<Tensor>().SizeInBytes() == second.Get<Tensor>().SizeInBytes() &&
      std::memcmp(empty_wrapper.Get<Tensor>().DataRaw(), second.Get<Tensor>().DataRaw(), second.Get<Tensor>().SizeInBytes()) == 0;
  if(empty_wrapper.IsTensor())receipt.empty_wrapper_difference=M0MeasureNativePcmDifferenceV1(
      second.Get<Tensor>().DataRaw(),second.Get<Tensor>().SizeInBytes(),empty_wrapper.Get<Tensor>().DataRaw(),empty_wrapper.Get<Tensor>().SizeInBytes());
  idle();
  const OrtValue third = valid_run(false, true);
  const auto& repeated = third.Get<Tensor>();
  receipt.repeated_samples = size_t(repeated.Shape()[0]);
  receipt.valid_logical_pcm = true;
  receipt.first_pcm_retained = first_bytes.size() == first_tensor.SizeInBytes() &&
      std::memcmp(first_bytes.data(), first_tensor.DataRaw(), first_bytes.size()) == 0;
  receipt.repeated_pcm_equal = repeated.SizeInBytes() == first_bytes.size() &&
      std::memcmp(first_bytes.data(), repeated.DataRaw(), first_bytes.size()) == 0;
  receipt.repeated_difference=M0MeasureNativePcmDifferenceV1(first_bytes.data(),first_bytes.size(),repeated.DataRaw(),repeated.SizeInBytes());
  receipt.original_session_retained = session.model_.get() == original && !session.is_inited_ && !session.session_state_ &&
      original_inputs == original->MainGraph().GetInputs() && original_outputs == original->MainGraph().GetOutputs();
  // Only final logical PCM is compared. No internal extent/alignment/activation
  // readback, caller-supplied module/factory or diagnostic authority is used.
  return receipt;
}

M0NativeRepeatReceiptV1 M0NativeLoweringQualificationV1::RepeatPublicPcm(
    InferenceSession& session,M0NativeRepeatInterludeV1 interlude){
  ORT_ENFORCE(interlude==M0NativeRepeatInterludeV1::None||interlude==M0NativeRepeatInterludeV1::Short||
      interlude==M0NativeRepeatInterludeV1::Invalid,"RW07_REPEAT_CLOSED_INTERLUDE");
  ORT_THROW_IF_ERROR(session.M0PrepareOriginalDeviceLoweringV1());
  auto& native=*session.m0_native_lowering_v1_;
  ORT_ENFORCE(native.device_capacity_,"RW07_REPEAT_ACTUAL_DEVICE_CAPACITY");
  ORT_THROW_IF_ERROR(native.InitializeExecution(session));
  ORT_THROW_IF_ERROR(session.Initialize());
  const auto* holder=native.execution_plan_.get();
  const auto allocator=std::make_shared<CPUAllocator>();
  const std::array<std::string,3> names{"tokens","style","speed"};
  const std::array<std::string,1> outputs{native.outputs_.pcm.data};
  RunOptions options;
  M0NativeRepeatReceiptV1 receipt;receipt.physical_capacity=native.capacity_;
  receipt.checked_idle=true;receipt.interlude_valid=true;
  const auto run=[&](bool short_input,bool invalid){
    const auto feeds=M0NativePublicInputFixtureV1(allocator,short_input,invalid);
    std::vector<OrtValue> result;
    const auto status=session.Run(options,names,feeds,outputs,&result);
    if(invalid){
      receipt.interlude_valid=!status.IsOK()&&status.ErrorMessage()=="RW07_NATIVE_GPU_STATUS_103"&&result.empty();
    }else{
      ORT_THROW_IF_ERROR(status);
      ORT_ENFORCE(result.size()==1&&result[0].IsTensor()&&result[0].Get<Tensor>().IsDataType<float>()&&
          result[0].Get<Tensor>().Location().device==OrtDevice()&&result[0].Get<Tensor>().SizeInBytes()>0,
          "RW07_REPEAT_PUBLIC_PCM");
    }
    native.ValidateInitializedExecution();
    receipt.checked_idle=receipt.checked_idle&&native.execution_plan_.get()==holder&&
        !holder->state->M0FullGpuExtentRunIdV2()&&!holder->state->M0FullGpuRunOwnerV2().Poisoned();
    return result;
  };
  const auto first=run(false,false);
  const auto& expected=first[0].Get<Tensor>();
  const auto* bytes=static_cast<const unsigned char*>(expected.DataRaw());
  receipt.first_pcm.assign(bytes,bytes+expected.SizeInBytes());
  if(interlude==M0NativeRepeatInterludeV1::Short){(void)run(true,false);}
  else if(interlude==M0NativeRepeatInterludeV1::Invalid){(void)run(false,true);}
  const auto repeated=run(false,false);
  const auto& actual=repeated[0].Get<Tensor>();
  bytes=static_cast<const unsigned char*>(actual.DataRaw());
  receipt.repeated_pcm.assign(bytes,bytes+actual.SizeInBytes());
  receipt.first_retained=expected.SizeInBytes()==receipt.first_pcm.size()&&
      std::memcmp(expected.DataRaw(),receipt.first_pcm.data(),receipt.first_pcm.size())==0;
  receipt.difference=M0MeasureNativePcmDifferenceV1(receipt.first_pcm.data(),receipt.first_pcm.size(),
      receipt.repeated_pcm.data(),receipt.repeated_pcm.size());
  return receipt;
}

M0NativeInitializedReceiptV1 M0NativeLoweringQualificationV1::InitializedExecution(
    InferenceSession& session, M0NativeInitializedFaultV1 fault) {
  const Model* source = nullptr;
  std::vector<const NodeArg*> inputs, outputs;
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    ORT_ENFORCE(session.m0_fullgpu_strict_requested_v1_ && session.model_ && session.is_model_loaded_ &&
                    !session.is_inited_ && !session.session_state_, "RW07_NATIVE_INIT_QUALIFICATION_SOURCE");
    source = session.model_.get();
    inputs = source->MainGraph().GetInputs();
    outputs = source->MainGraph().GetOutputs();
  }
  ORT_THROW_IF_ERROR(session.M0PrepareOriginalLoweringV1(1024));
  auto& native = *session.m0_native_lowering_v1_;
  ORT_THROW_IF_ERROR(native.InitializeExecution(session));
  M0NativeInitializedReceiptV1 receipt;
  SessionState* state = nullptr;
  const SequentialExecutionPlan* plan = nullptr;
  const void* holder = nullptr;
  std::vector<const Tensor*> tensors;
  std::vector<const OpKernel*> kernels;
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    native.ValidateInitializedExecution();
    const auto& actual = *native.execution_plan_;
    state = actual.state.get();
    plan = state->GetExecutionPlan();
    holder = native.execution_plan_.get();
    receipt.expected_initializers = size_t(native.derived_.graph().initializer_size());
    receipt.initializers = state->GetInitializedTensors().size();
    receipt.constants = state->GetConstantInitializedTensors().size();
    receipt.uploaded = actual.uploaded.size();
    receipt.planned_sources = actual.planned_sources.size();
    receipt.nodes = native.execution_model_->MainGraph().NumberOfNodes();
    receipt.initialized = actual.stage == M0FullGpuNativeLoweringOwnerV1::ExecutionPlan::Stage::Initialized;
    receipt.checked_completion = actual.initialization_attempted && actual.provider_started && actual.provider_end_attempted &&
        !actual.initialization_scope && !state->M0FullGpuExtentRunIdV2() && !state->M0FullGpuRunOwnerV2().Poisoned();
    receipt.state_bound = &state->GetGraphViewer().GetGraph() == &native.execution_model_->MainGraph();
    receipt.epoch_bound = actual.epoch && actual.epoch == state->M0FullGpuExtentOwnerEpochV1();
    receipt.plan_bound = actual.plan == plan && plan;
    std::set<uintptr_t> handles;
    std::set<uint64_t> generations;
    receipt.gpu_parameters = true;
    for (const auto& entry : actual.uploaded) {
      const auto& value = state->GetInitializedTensors().at(entry.first);
      const auto& tensor = value.Get<Tensor>();
      const auto& binding = entry.second.resource;
      receipt.gpu_parameters = receipt.gpu_parameters && tensor.Location().device.Type() == OrtDevice::GPU &&
          !tensor.Location().device.UsesCpuMemory() && tensor.DataRaw() && tensor.SizeInBytes() > 0 &&
          binding.handle == reinterpret_cast<uintptr_t>(tensor.DataRaw()) && binding.generation &&
          binding.requested_bytes >= tensor.SizeInBytes() && binding.storage && binding.copy_destination &&
          handles.insert(binding.handle).second && generations.insert(binding.generation).second;
      tensors.push_back(&tensor);
    }
    receipt.unique_resources = handles.size();
    for (const auto& node : native.execution_model_->MainGraph().Nodes()) {
      const auto* kernel = state->GetKernel(node.Index());
      ORT_ENFORCE(kernel, "RW07_NATIVE_INIT_QUALIFICATION_ACTUAL_KERNEL");
      kernels.push_back(kernel);
    }
    receipt.kernels = kernels.size();
    receipt.original_model_retained = session.model_.get() == source;
    receipt.original_ports_retained = inputs == source->MainGraph().GetInputs() && outputs == source->MainGraph().GetOutputs();
    receipt.session_uninitialized = !session.is_inited_ && !session.session_state_;
  }
  // Revalidation of the same initialized holder, not a second upload/constructor
  // pass. Public Initialize and Run are never called by this qualification API.
  ORT_THROW_IF_ERROR(native.InitializeExecution(session));
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    const auto& repeated = *native.execution_plan_;
    std::vector<const Tensor*> next_tensors;
    std::vector<const OpKernel*> next_kernels;
    for (const auto& entry : repeated.uploaded)
      next_tensors.push_back(&repeated.state->GetInitializedTensors().at(entry.first).Get<Tensor>());
    for (const auto& node : native.execution_model_->MainGraph().Nodes())
      next_kernels.push_back(repeated.state->GetKernel(node.Index()));
    receipt.replay_retained = native.execution_plan_.get() == holder && repeated.state.get() == state &&
        repeated.plan == plan && repeated.state->GetExecutionPlan() == plan && tensors == next_tensors && kernels == next_kernels &&
        repeated.stage == M0FullGpuNativeLoweringOwnerV1::ExecutionPlan::Stage::Initialized &&
        repeated.planned_sources.size() == receipt.planned_sources && repeated.uploaded.size() == receipt.uploaded &&
        !repeated.initialization_scope && !state->M0FullGpuExtentRunIdV2();
  }
  if(fault==M0NativeInitializedFaultV1::None){
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    uint64_t serial=0;
    {
      const auto proof=native.ValidateSourceBoundary();serial=native.active_source_boundary_;
      bool overlap_denied=false;
      try{const auto overlap=native.ValidateSourceBoundary();}catch(const std::exception&){overlap_denied=true;}
      ORT_ENFORCE(overlap_denied&&serial,"RW07_NATIVE_BOUNDARY_QUALIFICATION_OVERLAP");
      native.ValidateInitializerPlan(*state,&proof);native.ValidateExecutionModel(&proof);
    }
    ORT_ENFORCE(!native.active_source_boundary_,"RW07_NATIVE_BOUNDARY_QUALIFICATION_NOT_RETIRED");
    {
      const auto next=native.ValidateSourceBoundary();
      native.ValidateExecutionModel(&next);
      receipt.boundary_source_lease_checked=native.active_source_boundary_>serial;
    }
    ORT_ENFORCE(!native.active_source_boundary_&&receipt.boundary_source_lease_checked,
      "RW07_NATIVE_BOUNDARY_QUALIFICATION_REUSED_SERIAL");
  }
  if (fault != M0NativeInitializedFaultV1::None) {
    constexpr const char* name = "encoder.text_encoder.embedding.weight";
    int index = -1;
    ORT_THROW_IF_ERROR(state->GetOrtValueNameIdxMap().GetIdx(name, index));
    auto& initializers = const_cast<std::unordered_map<int, OrtValue>&>(state->GetInitializedTensors());
    auto& allocations = const_cast<SequentialExecutionPlan*>(plan)->allocation_plan;
    const OrtValue original = initializers.at(index);
    const auto original_allocation = allocations.at(size_t(index));
    auto& source_graph=const_cast<Graph&>(source->MainGraph());
    auto* source_weight=const_cast<ONNX_NAMESPACE::TensorProto*>(source_graph.GetConstantInitializer(name,false));
    ORT_ENFORCE(source_weight&&!source_weight->raw_data().empty(),"RW07_NATIVE_INIT_QUALIFICATION_SOURCE_WEIGHT");
    const char original_byte=source_weight->raw_data()[0];
    const auto original_config=session.session_options_.config_options;
    const auto restore_source=[&](){
      if(fault==M0NativeInitializedFaultV1::SourceRawBytes)(*source_weight->mutable_raw_data())[0]=original_byte;
      if(fault==M0NativeInitializedFaultV1::SourceGraphPorts)source_graph.SetInputs(inputs);
      if(fault==M0NativeInitializedFaultV1::CapacityCeilingMutation)session.session_options_.config_options=original_config;
    };
    auto restore = gsl::finally([&]() noexcept {
      if (fault == M0NativeInitializedFaultV1::InitializerSubstitution) initializers.at(index) = original;
      if (fault == M0NativeInitializedFaultV1::AllocationReuse) allocations[size_t(index)] = original_allocation;
      restore_source();
    });
    {
      std::lock_guard<std::mutex> lock(session.session_mutex_);
      if (fault == M0NativeInitializedFaultV1::InitializerSubstitution) {
        auto other = initializers.begin();
        if (other->first == index) ++other;
        ORT_ENFORCE(other != initializers.end(), "RW07_NATIVE_INIT_QUALIFICATION_OTHER_ACTUAL_PARAMETER");
        initializers.at(index) = other->second;
      } else if (fault == M0NativeInitializedFaultV1::AllocationReuse) {
        allocations[size_t(index)].alloc_kind = AllocKind::kReuse;
      } else if(fault==M0NativeInitializedFaultV1::SourceRawBytes){
        (*source_weight->mutable_raw_data())[0]^=1;
      } else if(fault==M0NativeInitializedFaultV1::SourceGraphPorts){
        source_graph.SetInputs({});
      } else if(fault==M0NativeInitializedFaultV1::CapacityCeilingMutation){
        ORT_THROW_IF_ERROR(session.session_options_.config_options.AddConfigEntry(kM0CapacityCeilingConfigV1,"1"));
      } else ORT_ENFORCE(fault == M0NativeInitializedFaultV1::LateCopy||fault==M0NativeInitializedFaultV1::BoundaryExecutionSubstitution,
                        "RW07_NATIVE_INIT_QUALIFICATION_UNKNOWN_FAULT");
    }
    if (fault == M0NativeInitializedFaultV1::LateCopy) {
      try {
        const auto* proto = native.execution_model_->MainGraph().GetConstantInitializer(name, false);
        receipt.fault_denied = !state->M0BeforeImmutableInitializerV1(name, index, *proto, M0NativeInitializerVisitV1::Copy).IsOK();
      } catch (const std::exception&) { receipt.fault_denied = true; }
    } else {
      if(fault==M0NativeInitializedFaultV1::BoundaryExecutionSubstitution){
       {
        std::lock_guard<std::mutex> lock(session.session_mutex_);
        const auto proof=native.ValidateSourceBoundary();
        // Keep the real model alive while substituting the live owner's pointer.
        // The same proof must fail before any dereference of a foreign/missing
        // execution model, and restoration cannot revive the poisoned owner.
        auto execution=std::move(native.execution_model_);
        auto restore_execution=gsl::finally([&]()noexcept{native.execution_model_=std::move(execution);});
        bool initializer_denied=false,execution_denied=false;
        try{native.ValidateInitializerPlan(*state,&proof);}catch(const std::exception& error){
         initializer_denied=std::string(error.what()).find("RW07_NATIVE_BOUNDARY_SOURCE_SUBSTITUTION")!=std::string::npos;
        }
        try{native.ValidateExecutionModel(&proof);}catch(const std::exception& error){
         execution_denied=std::string(error.what()).find("RW07_NATIVE_BOUNDARY_SOURCE_SUBSTITUTION")!=std::string::npos;
        }
        ORT_ENFORCE(initializer_denied&&execution_denied,"RW07_NATIVE_BOUNDARY_QUALIFICATION_MISSED_SUBSTITUTION");
       }
       ORT_ENFORCE(!native.active_source_boundary_&&native.phase_==M0FullGpuNativeLoweringOwnerV1::Phase::Poisoned,
        "RW07_NATIVE_BOUNDARY_QUALIFICATION_UNRETIRED_OR_REVIVED");
      }
      if(fault==M0NativeInitializedFaultV1::SourceRawBytes||fault==M0NativeInitializedFaultV1::SourceGraphPorts||fault==M0NativeInitializedFaultV1::CapacityCeilingMutation){
        // Exercise the same boundary used by actual Run directly, before the
        // replay entry point's additional source check can hide a regression.
        bool boundary_denied=false;
        {
          std::lock_guard<std::mutex> lock(session.session_mutex_);
          try{native.ValidateInitializedExecution();}catch(const std::exception&){boundary_denied=true;}
        }
        ORT_ENFORCE(boundary_denied,"RW07_NATIVE_INIT_QUALIFICATION_BOUNDARY_MISSED_SOURCE_MUTATION");
      }
      try { receipt.fault_denied = !native.InitializeExecution(session).IsOK(); }
      catch (const std::exception&) { receipt.fault_denied = true; }
    }
    {
      std::lock_guard<std::mutex> lock(session.session_mutex_);
      initializers.at(index) = original;
      allocations[size_t(index)] = original_allocation;
      restore_source();
      receipt.retired = native.phase_ == M0FullGpuNativeLoweringOwnerV1::Phase::Poisoned;
    }
    try { receipt.restored_replay_denied = !native.InitializeExecution(session).IsOK(); }
    catch (const std::exception&) { receipt.restored_replay_denied = true; }
  }
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    receipt.original_model_retained = receipt.original_model_retained && session.model_.get() == source;
    receipt.original_ports_retained = receipt.original_ports_retained && inputs == source->MainGraph().GetInputs() && outputs == source->MainGraph().GetOutputs();
    receipt.session_uninitialized = receipt.session_uninitialized && !session.is_inited_ && !session.session_state_;
  }
  return receipt;
}

M0NativeInitializerHookReceiptV1 M0NativeLoweringQualificationV1::InitializerHookOutsidePhase(
    InferenceSession& session, M0NativeInitializerHookV1 hook) {
  ORT_ENFORCE(Plan(session).resource_denied, "RW07_NATIVE_UPLOAD_HOOK_QUALIFICATION_ACTUAL_PLAN");
  std::lock_guard<std::mutex> lock(session.session_mutex_);
  auto& native = *session.m0_native_lowering_v1_;
  auto& state = native.QualificationExecutionState();
  constexpr const char* name = "encoder.text_encoder.embedding.weight";
  const auto* source = native.execution_model_->MainGraph().GetConstantInitializer(name, false);
  int index = -1;
  ORT_THROW_IF_ERROR(state.GetOrtValueNameIdxMap().GetIdx(name, index));
  ORT_ENFORCE(source && index >= 0 && state.GetInitializedTensors().empty() &&
                  state.GetConstantInitializedTensors().empty(), "RW07_NATIVE_UPLOAD_HOOK_QUALIFICATION_SOURCE");
  // An unallocated endpoint intentionally cannot be inserted. The phase guard
  // must reject BEFORE inspecting it or altering either actual initializer map.
  const OrtValue unallocated;
  const auto refuses = [&]() {
    try {
      common::Status status;
      if (hook == M0NativeInitializerHookV1::Endpoint)
        status = state.AddInitializedTensor(index, unallocated, true, false);
      else {
        ORT_ENFORCE(hook == M0NativeInitializerHookV1::Plan || hook == M0NativeInitializerHookV1::Copy,
                    "RW07_NATIVE_UPLOAD_HOOK_QUALIFICATION_UNKNOWN_HOOK");
        status = state.M0BeforeImmutableInitializerV1(name, index, *source,
            hook == M0NativeInitializerHookV1::Plan ? M0NativeInitializerVisitV1::Plan : M0NativeInitializerVisitV1::Copy);
      }
      return !status.IsOK();
    } catch (const std::exception&) { return true; }
  };
  M0NativeInitializerHookReceiptV1 receipt;
  receipt.denied = refuses();
  receipt.retired = native.phase_ == M0FullGpuNativeLoweringOwnerV1::Phase::Poisoned;
  receipt.replay_denied = refuses();
  receipt.maps_unchanged = state.GetInitializedTensors().empty() && state.GetConstantInitializedTensors().empty();
  receipt.session_uninitialized = !session.is_inited_ && !session.session_state_;
  return receipt;
}

M0NativeResourceMetadataReceiptV3 M0NativeLoweringQualificationV1::ResourceMetadata(
    InferenceSession& session, M0NativeResourceMetadataFaultV3 fault) {
  ORT_THROW_IF_ERROR(session.M0PrepareOriginalLoweringV1(1024));
  std::lock_guard<std::mutex> lock(session.session_mutex_);
  const auto environment = OrtEnv::TryGetInstance();
  ORT_ENFORCE(environment && &environment->GetEnvironment() == &session.environment_ &&
                  session.m0_native_lowering_v1_ && !session.is_inited_ && !session.session_state_,
              "RW07_NATIVE_RESOURCE_METADATA_QUALIFICATION_SESSION");
  auto module = M0NativeGpuModuleV2::Open(*session.session_logger_);
  const auto actual = module->Resource(0, 0);
  M0NativeResourceMetadataReceiptV3 receipt;
  receipt.context_present = actual.abi == 3 && actual.structure_bytes == sizeof(actual) &&
      actual.ep && actual.context && actual.device && actual.queue;
  receipt.limits_present = actual.max_buffer_bytes && actual.max_binding_bytes;
  receipt.context_has_no_allocation = !actual.handle && !actual.generation && !actual.requested_bytes &&
      !actual.backing_bytes && !actual.session && !actual.run && !actual.data_handle && !actual.data_generation &&
      !actual.producer_role && !actual.producer_instance && !actual.storage && !actual.copy_source && !actual.copy_destination;
  const auto same = [&](const M0NativeResourceBindingV3& next) {
    return actual.abi == next.abi && actual.structure_bytes == next.structure_bytes && actual.ep == next.ep &&
        actual.context == next.context && actual.device == next.device && actual.queue == next.queue &&
        actual.max_buffer_bytes == next.max_buffer_bytes && actual.max_binding_bytes == next.max_binding_bytes;
  };
  receipt.stable_requery = same(module->Resource(0, 0));
  const auto refuses = [&](uintptr_t handle, uint64_t bytes) {
    try { (void)module->Resource(handle, bytes); return false; }
    catch (const std::exception&) { return true; }
  };
  // No allocation is created. Handle1 is absent from the actual ownership map,
  // so the provider must refuse before any Dawn BUFFER introspection.
  receipt.invalid_handle_denied = refuses(1, 1);
  receipt.malformed_context_denied = refuses(0, 1);
  receipt.missing_extent_denied = refuses(1, 0);
  const auto saved = module->context_;
  if (fault == M0NativeResourceMetadataFaultV3::ContextSwap) module->context_.context ^= uintptr_t{1};
  else if (fault == M0NativeResourceMetadataFaultV3::QueueSwap) module->context_.queue ^= uintptr_t{1};
  else ORT_ENFORCE(fault == M0NativeResourceMetadataFaultV3::None,
                   "RW07_NATIVE_RESOURCE_METADATA_QUALIFICATION_UNKNOWN_FAULT");
  if (fault != M0NativeResourceMetadataFaultV3::None) receipt.fault_denied = refuses(0, 0);
  module->context_ = saved;
  receipt.restored_requery = same(module->Resource(0, 0));
  receipt.session_uninitialized = !session.is_inited_ && !session.session_state_;
  // This holder is metadata-only and grants no plan/resource/Run permission.
  // Real device-loss testing needs an authenticated device harness, not flags.
  return receipt;
}

M0NativeResourceReceiptV1 M0NativeLoweringQualificationV1::ResourcePlan(
    InferenceSession& session, M0NativeResourceFaultV1 fault) {
  ORT_ENFORCE(Plan(session).resource_denied, "RW07_NATIVE_RESOURCE_QUALIFICATION_PREPARATION");
  std::lock_guard<std::mutex> lock(session.session_mutex_);
  auto& native = *session.m0_native_lowering_v1_;
  auto& state = native.QualificationExecutionState();
  auto& plan = const_cast<SequentialExecutionPlan&>(*state.GetExecutionPlan());
  ORT_ENFORCE(!plan.release_actions.empty(), "RW07_NATIVE_RESOURCE_QUALIFICATION_ACTUAL_RELEASE");
  const auto& graph = native.execution_model_->MainGraph();
  const auto value_index = plan.release_actions.front().value_index;
  ORT_ENFORCE(value_index < plan.allocation_plan.size() &&
                  plan.allocation_plan[value_index].alloc_kind == AllocKind::kAllocate,
              "RW07_NATIVE_RESOURCE_QUALIFICATION_ACTUAL_INTERNAL");
  std::string name;
  ORT_THROW_IF_ERROR(state.GetOrtValueNameIdxMap().GetName(static_cast<int>(value_index), name));
  const Node* producer = nullptr;
  for (const auto& node : graph.Nodes())
    for (const auto* output : node.OutputDefs())
      if (output->Exists() && output->Name() == name) producer = &node;
  ORT_ENFORCE(producer, "RW07_NATIVE_RESOURCE_QUALIFICATION_ACTUAL_PRODUCER");
  M0NativeResourceReceiptV1 receipt;
  receipt.resources = plan.allocation_plan.size();
  receipt.releases = plan.release_actions.size();
  const auto capture = [&]() {
    try {
      const auto status = native.CaptureExecutionPlan(state);
      receipt.resource_denied = !status.IsOK() && status.ErrorMessage() == "RW07_NATIVE_PREINIT_DENY_RESOURCE_CONTRACT";
      return !status.IsOK() && !receipt.resource_denied;
    } catch (const std::exception&) { return true; }
  };
  if (fault == M0NativeResourceFaultV1::None) {
    ORT_ENFORCE(!capture() && receipt.resource_denied, "RW07_NATIVE_RESOURCE_QUALIFICATION_POSITIVE");
  } else {
    const auto saved_allocations = plan.allocation_plan;
    const auto saved_releases = plan.node_release_list;
    const auto saved_streams = plan.node_stream_map_;
    switch (fault) {
      case M0NativeResourceFaultV1::AllocationReuse:
        plan.allocation_plan[value_index].alloc_kind = AllocKind::kReuse;
        plan.allocation_plan[value_index].reused_buffer = 0;
        break;
      case M0NativeResourceFaultV1::ForeignRow:
        plan.allocation_plan.push_back(saved_allocations[value_index]);
        break;
      case M0NativeResourceFaultV1::ReleaseBeforeLastUse: {
        size_t removed = 0;
        for (auto& row : plan.node_release_list) {
          const auto found = std::find(row.begin(), row.end(), size_t{0});
          if (found != row.end()) { row.erase(found); ++removed; }
        }
        ORT_ENFORCE(removed == 1 && producer->Index() < plan.node_release_list.size(),
                    "RW07_NATIVE_RESOURCE_QUALIFICATION_ACTUAL_LAST_USE");
        // Action zero remains the SAME actual action/value; only its release
        // site moves from the last consumer to the real producer.
        plan.node_release_list[producer->Index()].push_back(0);
        break;
      }
      case M0NativeResourceFaultV1::StreamBinding:
        ORT_ENFORCE(producer->Index() < plan.node_stream_map_.size(),
                    "RW07_NATIVE_RESOURCE_QUALIFICATION_ACTUAL_STREAM");
        plan.node_stream_map_[producer->Index()] = plan.execution_plan.size();
        break;
      default: ORT_THROW("RW07_NATIVE_RESOURCE_QUALIFICATION_UNKNOWN_FAULT");
    }
    receipt.fault_denied = capture();
    plan.allocation_plan = saved_allocations;
    plan.node_release_list = saved_releases;
    plan.node_stream_map_ = saved_streams;
    receipt.retired = native.phase_ == M0FullGpuNativeLoweringOwnerV1::Phase::Poisoned;
    receipt.restored_replay_denied = capture();
  }
  receipt.session_uninitialized = !session.is_inited_ && !session.session_state_;
  return receipt;
}

M0NativeInitializerReceiptV1 M0NativeLoweringQualificationV1::InitializerPlan(
    InferenceSession& session, M0NativeInitializerFaultV1 fault) {
  const auto prepared = Plan(session);  // Actual Initialize, authenticated module and retained plan.
  ORT_ENFORCE(prepared.resource_denied && prepared.replay_resource_denied,
              "RW07_NATIVE_INITIALIZER_QUALIFICATION_PREPARATION");
  std::lock_guard<std::mutex> lock(session.session_mutex_);
  auto& native = *session.m0_native_lowering_v1_;
  auto& state = native.QualificationExecutionState();
  auto& plan = const_cast<SequentialExecutionPlan&>(*state.GetExecutionPlan());
  constexpr const char* name = "encoder.text_encoder.embedding.weight";
  auto* tensor = const_cast<ONNX_NAMESPACE::TensorProto*>(
      native.execution_model_->MainGraph().GetConstantInitializer(name, false));
  ORT_ENFORCE(tensor && tensor->data_type() == ONNX_NAMESPACE::TensorProto::FLOAT &&
                  tensor->has_raw_data() && !tensor->raw_data().empty(),
              "RW07_NATIVE_INITIALIZER_QUALIFICATION_ACTUAL_CONSTANT");
  int index = -1;
  ORT_THROW_IF_ERROR(state.GetOrtValueNameIdxMap().GetIdx(name, index));
  ORT_ENFORCE(index >= 0 && size_t(index) < plan.allocation_plan.size(),
              "RW07_NATIVE_INITIALIZER_QUALIFICATION_ACTUAL_INDEX");
  M0NativeInitializerReceiptV1 receipt;
  receipt.initializers = native.execution_model_->MainGraph().GetAllInitializedTensors().size();
  const auto capture = [&]() {
    try {
      const auto status = native.CaptureExecutionPlan(state);
      receipt.resource_denied = !status.IsOK() && status.ErrorMessage() == "RW07_NATIVE_PREINIT_DENY_RESOURCE_CONTRACT";
      return !status.IsOK() && !receipt.resource_denied;
    } catch (const std::exception&) { return true; }
  };
  if (fault == M0NativeInitializerFaultV1::None) {
    ORT_ENFORCE(!capture() && receipt.resource_denied, "RW07_NATIVE_INITIALIZER_QUALIFICATION_POSITIVE");
  } else {
    auto& row = plan.allocation_plan[size_t(index)];
    const auto saved_row = row;
    const auto saved_order = plan.initializer_allocation_order;
    const std::string saved_raw = tensor->raw_data();
    switch (fault) {
      case M0NativeInitializerFaultV1::RawBytes:
        (*tensor->mutable_raw_data())[0] ^= 1;
        break;
      case M0NativeInitializerFaultV1::ValueType:
        row.value_type = DataTypeImpl::TensorTypeFromONNXEnum(ONNX_NAMESPACE::TensorProto::INT32);
        break;
      case M0NativeInitializerFaultV1::AllocationKind:
        row.alloc_kind = AllocKind::kReuse;
        break;
      case M0NativeInitializerFaultV1::InitializerOrder:
        plan.initializer_allocation_order.push_back(index);
        plan.initializer_allocation_order.push_back(index);
        break;
      case M0NativeInitializerFaultV1::Lifetime:
        row.program_counter = AllocPlanPerValue::ProgramCounter{};
        row.program_counter.AddStart(0);
        row.program_counter.AddEnd(0);
        // Guarantee distinct metadata even if the actual interval was [0,0].
        if (saved_row.program_counter.Starts() == row.program_counter.Starts() &&
            saved_row.program_counter.Ends() == row.program_counter.Ends()) {
          row.program_counter.AddStart(1);
          row.program_counter.AddEnd(1);
        }
        break;
      default: ORT_THROW("RW07_NATIVE_INITIALIZER_QUALIFICATION_UNKNOWN_FAULT");
    }
    receipt.fault_denied = capture();
    // Restore the actual retained objects; do not replace the owner/state/plan
    // or open a new resolve lease. The poisoned owner must remain unusable.
    row = saved_row;
    plan.initializer_allocation_order = saved_order;
    tensor->set_raw_data(saved_raw);
    receipt.retired = native.phase_ == M0FullGpuNativeLoweringOwnerV1::Phase::Poisoned;
    receipt.restored_replay_denied = capture();
  }
  receipt.session_uninitialized = !session.is_inited_ && !session.session_state_;
  return receipt;
}

M0NativePlanReceiptV1 M0NativeLoweringQualificationV1::Plan(InferenceSession& session) {
  const Model* source = nullptr;
  std::vector<const NodeArg*> inputs, outputs;
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    ORT_ENFORCE(session.m0_fullgpu_strict_requested_v1_ && session.model_ && session.is_model_loaded_ &&
                    !session.is_inited_ && !session.session_state_, "RW07_NATIVE_PLAN_QUALIFICATION_SOURCE");
    source = session.model_.get();
    inputs = source->MainGraph().GetInputs();
    outputs = source->MainGraph().GetOutputs();
  }
  ORT_THROW_IF_ERROR(session.M0PrepareOriginalLoweringV1(1024));
  // Explicit observation intent; this plan can never be upgraded into an
  // initialization attempt. Public Initialize's eventual policy is independent.
  const auto status = session.m0_native_lowering_v1_->PrepareExecutionPlan(session);
  M0NativePlanReceiptV1 receipt;
  receipt.resource_denied = !status.IsOK() && status.ErrorMessage() == "RW07_NATIVE_PREINIT_DENY_RESOURCE_CONTRACT";
  ORT_ENFORCE(receipt.resource_denied, "RW07_NATIVE_PLAN_QUALIFICATION_EXPECTED_RESOURCE_DENIAL: ", status.ErrorMessage());
  const void* holder = nullptr;
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    const auto& native = *session.m0_native_lowering_v1_;
    const auto actual = native.ObserveExecutionPlan();
    receipt.nodes = actual.nodes;
    receipt.selected = actual.selected;
    receipt.resources = actual.resources;
    receipt.initializers = actual.initializers;
    receipt.kernels = actual.kernels;
    receipt.allocators = actual.allocators;
    receipt.state_bound = actual.state_bound;
    receipt.epoch_bound = actual.epoch_bound;
    receipt.plan_bound = actual.plan_bound;
    receipt.resource_gate_closed = actual.resource_gate_closed;
    receipt.execution_sha256 = actual.execution_sha256;
    receipt.original_model_retained = session.model_.get() == source;
    receipt.original_ports_retained = inputs == source->MainGraph().GetInputs() && outputs == source->MainGraph().GetOutputs();
    receipt.session_uninitialized = !session.is_inited_ && !session.session_state_;
    holder = native.execution_plan_.get();
  }
  const auto replay = session.m0_native_lowering_v1_->PrepareExecutionPlan(session);
  receipt.replay_resource_denied = !replay.IsOK() && replay.ErrorMessage() == "RW07_NATIVE_PREINIT_DENY_RESOURCE_CONTRACT";
  {
    std::lock_guard<std::mutex> lock(session.session_mutex_);
    const auto& native = *session.m0_native_lowering_v1_;
    const auto repeated = native.ObserveExecutionPlan();
    receipt.retained_plan_holder = native.execution_plan_.get() == holder && repeated.state_bound &&
        repeated.epoch_bound && repeated.plan_bound && repeated.execution_sha256 == receipt.execution_sha256;
    receipt.original_model_retained = receipt.original_model_retained && session.model_.get() == source;
    receipt.original_ports_retained = receipt.original_ports_retained && inputs == source->MainGraph().GetInputs() &&
        outputs == source->MainGraph().GetOutputs();
    receipt.session_uninitialized = receipt.session_uninitialized && !session.is_inited_ && !session.session_state_;
  }
  return receipt;
}

M0NativeModuleReceiptV2 M0NativeLoweringQualificationV1::ModuleMetadata(
    InferenceSession& session, M0NativeModuleFaultV2 fault) {
  // The caller cannot supply a module, registry, callback or replacement graph.
  // Preparation proves this is the same fixed sealed original/session path.
  ORT_THROW_IF_ERROR(session.M0PrepareOriginalLoweringV1(1024));
  std::lock_guard<std::mutex> lock(session.session_mutex_);
  const auto environment = OrtEnv::TryGetInstance();
  ORT_ENFORCE(environment && &environment->GetEnvironment() == &session.environment_,
              "RW07_NATIVE_MODULE_QUALIFICATION_ENVIRONMENT");
  ORT_ENFORCE(session.m0_native_lowering_v1_ && !session.is_inited_ && !session.session_state_,
              "RW07_NATIVE_MODULE_QUALIFICATION_SESSION");
  auto module = M0NativeGpuModuleV2::Open(*session.session_logger_);
  module->ValidateCurrent();
  M0NativeModuleReceiptV2 receipt;
  for (const auto& record : kM0NativeKernelRecordsV2) {
    const size_t index = record.id - 1;
    const auto& query = module->queried_[index];
    const auto& selected = module->selected_[index];
    ORT_ENFORCE(query.abi == 2 && query.record_id == record.id && query.type == record.type &&
                    query.creator.callback && query.creator.state && query.registry && query.registration_definition,
                "RW07_NATIVE_MODULE_QUALIFICATION_QUERY");
    ++receipt.queried_records;
    ORT_ENFORCE(selected.selected_registry == module->registry_.get() && selected.selected_create_info &&
                    selected.selected_definition && selected.registered_definition,
                "RW07_NATIVE_MODULE_QUALIFICATION_SELECTED");
    ++receipt.selected_records;
    const auto* registration = reinterpret_cast<const KernelDef*>(query.registration_definition);
    if (selected.registered_definition != selected.selected_definition &&
        selected.registered_definition != registration && selected.selected_definition != registration)
      ++receipt.separate_definition_clones;
  }
  // Exercise borrowed bindings and the real authenticated requery path again.
  (void)module->Binding(1);
  (void)module->Binding(42);
  module->ValidateCurrent();
  receipt.stable_requery = true;
  size_t refused = 0;
  for (const uint32_t record : {0u, 43u}) {
    try { (void)module->Binding(record); } catch (const std::exception&) { ++refused; }
  }
  receipt.invalid_records_denied = refused == 2;
  if (fault == M0NativeModuleFaultV2::JointDefinitionMemory) {
    const auto& selected = module->selected_[0];  // Actual compiled Add record1.
    std::array<KernelDef*, 3> definitions{{
        const_cast<KernelDef*>(reinterpret_cast<const KernelDef*>(module->queried_[0].registration_definition)),
        const_cast<KernelDef*>(selected.registered_definition),
        const_cast<KernelDef*>(selected.selected_definition)}};
    ORT_ENFORCE(definitions[0] != definitions[1] && definitions[0] != definitions[2] && definitions[1] != definitions[2] &&
                    definitions[0]->OpName() == "Add" && module->frozen_definitions_[0],
                "RW07_NATIVE_MODULE_QUALIFICATION_ACTUAL_THREE_DEFINITIONS");
    {
      std::array<KernelDef, 3> originals{{*definitions[0], *definitions[1], *definitions[2]}};
      auto restore = gsl::finally([&]() noexcept {
        // Only memory maps are changed; swap restores their complete exact
        // original contents without allocation, even on an exception path.
        for (size_t i = 0; i < definitions.size(); ++i)
          const_cast<MemTypeMap&>(definitions[i]->InputMemoryTypeArgs()).swap(
              const_cast<MemTypeMap&>(originals[i].InputMemoryTypeArgs()));
      });
      for (auto* definition : definitions)
        const_cast<MemTypeMap&>(definition->InputMemoryTypeArgs())[0] = OrtMemTypeCPUInput;
      M0NativeSelectedKernelBindingV2 coherent;
      receipt.coherent_definition_drift =
          M0SameNativeKernelDefinitionV2(*definitions[0], *definitions[1]) &&
          M0SameNativeKernelDefinitionV2(*definitions[0], *definitions[2]) &&
          !M0SameNativeKernelDefinitionV2(*module->frozen_definitions_[0], *definitions[0]) &&
          M0InspectNativeSelectedKernelV2(*module->registry_, *selected.selected_create_info, module->queried_[0], coherent);
      try { module->ValidateCurrent(); }
      catch (const std::exception& error) {
        receipt.fault_denied = std::string(error.what()).find("RW07_NATIVE_MODULE_CHANGED_FROZEN_DEFINITION") != std::string::npos;
      }
    }  // Restore ALL actual registrations before subsequent queries/destruction.
    module->ValidateCurrent();
    receipt.restored_definition_metadata = true;
  } else if (fault != M0NativeModuleFaultV2::None) {
    if (fault == M0NativeModuleFaultV2::QueriedStateSwap)
      module->queried_[0].creator.state = module->queried_[2].creator.state;
    else if (fault == M0NativeModuleFaultV2::QueriedDefinitionSwap)
      module->queried_[0].registration_definition = module->queried_[1].registration_definition;
    else if (fault == M0NativeModuleFaultV2::SelectedInfoSwap)
      module->selected_[0].selected_create_info = module->selected_[1].selected_create_info;
    else if (fault == M0NativeModuleFaultV2::SelectedDefinitionSwap)
      module->selected_[0].selected_definition = module->selected_[1].selected_definition;
    else ORT_THROW("RW07_NATIVE_MODULE_QUALIFICATION_UNKNOWN_FAULT");
    try { module->ValidateCurrent(); } catch (const std::exception&) { receipt.fault_denied = true; }
  }
  receipt.original_session_uninitialized = !session.is_inited_ && !session.session_state_;
  // The actual module/provider/factory lease remains live until all metadata
  // reads finish. No raw module/registry/creator escapes in the receipt.
  return receipt;
}
#endif
}  // namespace onnxruntime
#endif
