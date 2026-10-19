// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"

#if !defined(ORT_MINIMAL_BUILD)
#include <map>
#include <set>
#include "core/framework/session_state.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/model.h"

namespace onnxruntime {
void M0FullGpuNativeLoweringOwnerV1::ValidateResourcePlan(const SessionState& state) const {
  ORT_ENFORCE(execution_model_ && &state.graph_ == &execution_model_->MainGraph() &&
                  state.m0_native_plan_owner_v1_ == this && state.p_seq_exec_plan_ && state.graph_viewer_ &&
                  !state.sess_options_.enable_mem_reuse && !state.sess_options_.enable_mem_pattern,
              "RW07_NATIVE_RESOURCE_ACTUAL_STATE");
  const auto& graph = state.graph_;
  const auto& plan = *state.p_seq_exec_plan_;
  const auto& names = state.ort_value_name_idx_map_;
  ORT_ENFORCE(graph.GetInputs().size() == 3 && graph.GetOutputs().size() == 4 &&
                  graph.GetOverridableInitializers().empty(), "RW07_NATIVE_RESOURCE_PUBLIC_PORTS");
  // This reviewed profile has one GPU logic stream. Multi-stream synchronization
  // and training schedules require a separate proof; no permission follows here.
  ORT_ENFORCE(plan.execution_plan.size() == 1 && plan.execution_plan[0] &&
                  plan.notification_owner_stream.empty() && plan.downstream_map.empty() && plan.num_barriers == 0,
              "RW07_NATIVE_RESOURCE_SINGLE_STREAM_PROFILE");
  const auto& stream = *plan.execution_plan[0];
  const auto& order = state.graph_viewer_->GetNodesInTopologicalOrder(state.sess_options_.execution_order);
  ORT_ENFORCE(stream.steps_.size() == order.size() && order.size() == graph.NumberOfNodes() &&
                  plan.node_stream_map_.size() == size_t(state.graph_viewer_->MaxNodeIndex()) + 1,
              "RW07_NATIVE_RESOURCE_NODE_ORDER_INVENTORY");
  std::map<NodeIndex, size_t> positions;
  for (size_t pc = 0; pc < order.size(); ++pc) {
    const auto* node = graph.GetNode(order[pc]);
    const auto& step = stream.steps_[pc];
    ORT_ENFORCE(node && step && step->GetNodeIndex() == node->Index() &&
                    step->ToString() == MakeString("LaunchKernel - NodeIndex: ", node->Index(), ", Name: ", node->Name()) &&
                    plan.node_stream_map_[node->Index()] == 0 && positions.emplace(node->Index(), pc).second,
                "RW07_NATIVE_RESOURCE_ACTUAL_LAUNCH_ORDER");
  }
  const auto* provider = state.execution_providers_.Get(kWebGpuExecutionProvider);
  ORT_ENFORCE(provider && state.execution_providers_.NumProviders() == 1,
              "RW07_NATIVE_RESOURCE_PROVIDER");
  const auto device = provider->GetOrtDeviceByMemType(OrtMemTypeDefault);
  ORT_ENFORCE(stream.device_ == device && device.Type() == OrtDevice::GPU && !device.UsesCpuMemory(),
              "RW07_NATIVE_RESOURCE_STREAM_DEVICE");
  struct Value {
    const NodeArg* argument = nullptr;
    AllocKind kind = AllocKind::kNotSet;
    NodeIndex producer = 0;
    bool produced = false;
    std::set<NodeIndex> consumers;
  };
  std::map<std::string, Value> values;
  for (const auto* input : graph.GetInputs()) {
    ORT_ENFORCE(input && values.emplace(input->Name(), Value{input, AllocKind::kPreExisting, 0, false, {}}).second,
                "RW07_NATIVE_RESOURCE_INPUT_IDENTITY");
  }
  for (const auto& initializer : graph.GetAllInitializedTensors()) {
    const auto* arg = graph.GetNodeArg(initializer.first);
    ORT_ENFORCE(arg && values.emplace(initializer.first, Value{arg, AllocKind::kAllocateStatically, 0, false, {}}).second,
                "RW07_NATIVE_RESOURCE_INITIALIZER_IDENTITY");
  }
  std::set<const NodeArg*> outputs(graph.GetOutputs().begin(), graph.GetOutputs().end());
  for (const auto& node : graph.Nodes()) {
    CheckNode(node);
    ORT_ENFORCE(node.ImplicitInputDefs().empty(), "RW07_NATIVE_RESOURCE_IMPLICIT_INPUT");
    const auto selected = state.kernel_create_info_map_.find(node.Index());
    ORT_ENFORCE(selected != state.kernel_create_info_map_.end() && selected->second && selected->second->kernel_def,
                "RW07_NATIVE_RESOURCE_ACTUAL_SELECTED_DEFINITION");
    const auto& def = *selected->second->kernel_def;
    ORT_ENFORCE(def.MayInplace().empty() && def.Alias().empty() && !def.VariadicAlias().has_value() &&
                    !def.HasExternalOutputs() && !def.AllocateInputsContiguously(),
                "RW07_NATIVE_RESOURCE_SELECTED_ALIAS_OR_EXTERNAL");
#ifdef ENABLE_STRIDED_TENSORS
    ORT_ENFORCE(def.MayStridedInput().empty() && def.MayStridedOutput().empty(),
                "RW07_NATIVE_RESOURCE_SELECTED_STRIDED");
#endif
    for (const auto* arg : node.OutputDefs()) if (arg->Exists()) {
      const auto kind = outputs.count(arg) ? AllocKind::kAllocateOutput : AllocKind::kAllocate;
      ORT_ENFORCE(values.emplace(arg->Name(), Value{arg, kind, node.Index(), true, {}}).second,
                  "RW07_NATIVE_RESOURCE_MULTIPLE_PRODUCERS");
    }
  }
  for (const auto& node : graph.Nodes()) {
    for (const auto* arg : node.InputDefs()) if (arg->Exists()) {
      auto found = values.find(arg->Name());
      ORT_ENFORCE(found != values.end() && found->second.argument == arg,
                  "RW07_NATIVE_RESOURCE_FOREIGN_CONSUMER_VALUE");
      auto& value = found->second;
      ORT_ENFORCE(!value.produced || positions.at(value.producer) < positions.at(node.Index()),
                  "RW07_NATIVE_RESOURCE_CONSUMER_BEFORE_PRODUCER");
      value.consumers.insert(node.Index());
    }
  }
  ORT_ENFORCE(values.size() == names.Size() && values.size() == plan.allocation_plan.size() &&
                  size_t(names.MaxIdx() + 1) == values.size(), "RW07_NATIVE_RESOURCE_EXACT_VALUE_INVENTORY");
  std::map<size_t, const Value*> indexed;
  size_t produced_count = 0;
  for (const auto& entry : values) {
    const auto& value = entry.second;
    int index = -1;
    ORT_THROW_IF_ERROR(names.GetIdx(entry.first, index));
    std::string reverse;
    ORT_THROW_IF_ERROR(names.GetName(index, reverse));
    ORT_ENFORCE(index >= 0 && size_t(index) < plan.allocation_plan.size() && reverse == entry.first &&
                    indexed.emplace(size_t(index), &value).second && value.argument == graph.GetNodeArg(entry.first),
                "RW07_NATIVE_RESOURCE_VALUE_INDEX");
    const auto& row = plan.allocation_plan[size_t(index)];
    ORT_ENFORCE(row.alloc_kind == value.kind && row.reused_buffer == index && row.location == device,
                "RW07_NATIVE_RESOURCE_FRESH_ALLOCATION_BINDING");
    const auto* type = value.argument->TypeAsProto();
    ORT_ENFORCE(type && type->has_tensor_type(), "RW07_NATIVE_RESOURCE_TENSOR_TYPE");
    MLDataType expected_type = DataTypeImpl::TensorTypeFromONNXEnum(type->tensor_type().elem_type());
#if !defined(ORT_MEMORY_PROFILE)
    if (value.kind == AllocKind::kAllocateStatically) expected_type = nullptr;
#endif
    ORT_ENFORCE(row.value_type == expected_type, "RW07_NATIVE_RESOURCE_ALLOCATION_TYPE");
#ifdef ENABLE_STRIDED_TENSORS
    ORT_ENFORCE(!row.is_strided_tensor, "RW07_NATIVE_RESOURCE_STRIDED_ALLOCATION");
#endif
    // CalculateProgramCounter is compiled/called only for ENABLE_TRAINING.
    // Do not invent inference intervals when the actual planner leaves none.
#if !defined(ENABLE_TRAINING)
    ORT_ENFORCE(row.program_counter.Starts().empty() && row.program_counter.Ends().empty(),
                "RW07_NATIVE_RESOURCE_UNEXPECTED_PROGRAM_COUNTER");
#else
    ORT_THROW("RW07_NATIVE_RESOURCE_UNREVIEWED_TRAINING_LIFETIME");
#endif
    if (value.produced) {
      ++produced_count;
#ifdef ORT_ENABLE_STREAM
      const auto binding = plan.value_to_stream_map.find(size_t(index));
      ORT_ENFORCE(binding != plan.value_to_stream_map.end() && binding->second == 0,
                  "RW07_NATIVE_RESOURCE_VALUE_STREAM_BINDING");
#endif
    }
  }
#ifdef ORT_ENABLE_STREAM
  ORT_ENFORCE(plan.value_to_stream_map.size() == produced_count, "RW07_NATIVE_RESOURCE_VALUE_STREAM_INVENTORY");
#else
  (void)produced_count;
  ORT_ENFORCE(plan.value_to_stream_map.empty(), "RW07_NATIVE_RESOURCE_UNEXPECTED_VALUE_STREAM_MAP");
#endif
  ORT_ENFORCE(plan.initializer_allocation_order.empty() && plan.activation_allocation_order.empty(),
              "RW07_NATIVE_RESOURCE_UNREQUESTED_CONTIGUOUS_ALLOCATION");
  // GenerateDeallocationPlan attaches one ref_count=1 action to the last
  // consumer on a single stream. The executor decrements it AFTER that kernel.
  ORT_ENFORCE(plan.node_release_list.size() == size_t(state.graph_viewer_->MaxNodeIndex()) + 1,
              "RW07_NATIVE_RESOURCE_RELEASE_NODE_INVENTORY");
  std::map<size_t, size_t> action_for_value;
  std::map<size_t, NodeIndex> last_consumer;
  for (const auto& item : indexed) {
    const auto& value = *item.second;
    if (value.kind != AllocKind::kAllocate || value.consumers.empty()) continue;
    NodeIndex last = *value.consumers.begin();
    for (const auto consumer : value.consumers)
      if (positions.at(consumer) > positions.at(last)) last = consumer;
    last_consumer.emplace(item.first, last);
  }
  ORT_ENFORCE(plan.release_actions.size() == last_consumer.size(), "RW07_NATIVE_RESOURCE_RELEASE_ACTION_INVENTORY");
  for (size_t i = 0; i < plan.release_actions.size(); ++i) {
    const auto& action = plan.release_actions[i];
    ORT_ENFORCE(last_consumer.count(action.value_index) && action.ref_count == 1 &&
                    action_for_value.emplace(action.value_index, i).second,
                "RW07_NATIVE_RESOURCE_RELEASE_ACTION_BINDING");
  }
  std::set<size_t> observed;
  for (size_t node = 0; node < plan.node_release_list.size(); ++node) {
    for (const auto action : plan.node_release_list[node]) {
      ORT_ENFORCE(action < plan.release_actions.size() && observed.insert(action).second &&
                      last_consumer.at(plan.release_actions[action].value_index) == node,
                  "RW07_NATIVE_RESOURCE_RELEASE_BEFORE_LAST_USE");
    }
  }
  ORT_ENFORCE(observed.size() == plan.release_actions.size(), "RW07_NATIVE_RESOURCE_RELEASE_COVERAGE");
  // Descriptor/host-plan liveness only. GPU command completion, actual buffer
  // allocation sizes, upload and resource ownership are not admitted here.
}
}  // namespace onnxruntime
#endif
