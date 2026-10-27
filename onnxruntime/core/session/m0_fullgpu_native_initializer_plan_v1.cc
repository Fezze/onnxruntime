// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"

#if !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include <limits>
#include <set>
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/execution_providers.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/model.h"
#include "core/graph/m0_bounded_backing_v1.h"
#include "core/graph/m0_original_preload_owner_v1.h"
#include "core/session/m0_fullgpu_native_execution_state_v1.h"

namespace onnxruntime {
namespace {
size_t CheckPayload(const ONNX_NAMESPACE::TensorProto& tensor) {
  ORT_ENFORCE(tensor.data_location() == ONNX_NAMESPACE::TensorProto::DEFAULT &&
                  tensor.external_data_size() == 0 && !tensor.has_segment(),
              "RW07_NATIVE_INITIALIZER_EXTERNAL_OR_SEGMENT");
  SafeInt<size_t> elements = 1;
  for (const auto dimension : tensor.dims()) {
    ORT_ENFORCE(dimension >= 0 && static_cast<uint64_t>(dimension) <= std::numeric_limits<size_t>::max(),
                "RW07_NATIVE_INITIALIZER_DIMENSION");
    elements *= static_cast<size_t>(dimension);
  }
  const size_t count = elements;
  size_t width = 0, typed_count = 0;
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      width = 4;
      typed_count = size_t(tensor.float_data_size());
      ORT_ENFORCE(tensor.int32_data_size() == 0 && tensor.int64_data_size() == 0,
                  "RW07_NATIVE_INITIALIZER_FLOAT_ENCODING");
      break;
    case ONNX_NAMESPACE::TensorProto::INT64:
      width = 8;
      typed_count = size_t(tensor.int64_data_size());
      ORT_ENFORCE(tensor.float_data_size() == 0 && tensor.int32_data_size() == 0,
                  "RW07_NATIVE_INITIALIZER_INT64_ENCODING");
      break;
    case ONNX_NAMESPACE::TensorProto::INT32:
      width = 4;
      typed_count = size_t(tensor.int32_data_size());
      ORT_ENFORCE(tensor.float_data_size() == 0 && tensor.int64_data_size() == 0,
                  "RW07_NATIVE_INITIALIZER_INT32_ENCODING");
      break;
    default: ORT_THROW("RW07_NATIVE_INITIALIZER_UNREVIEWED_ELEMENT_TYPE");
  }
  ORT_ENFORCE(tensor.double_data_size() == 0 && tensor.uint64_data_size() == 0 && tensor.string_data_size() == 0,
              "RW07_NATIVE_INITIALIZER_UNREVIEWED_PAYLOAD_FIELD");
  const size_t bytes = SafeInt<size_t>(count) * width;
  if (tensor.has_raw_data()) {
    ORT_ENFORCE(typed_count == 0 && tensor.raw_data().size() == bytes,
                "RW07_NATIVE_INITIALIZER_RAW_BYTE_EXTENT");
  } else {
    ORT_ENFORCE(typed_count == count, "RW07_NATIVE_INITIALIZER_TYPED_ELEMENT_COUNT");
  }
  return bytes;
}
void CheckDescriptor(const NodeArg& argument, const ONNX_NAMESPACE::TensorProto& tensor) {
  const auto* type = argument.TypeAsProto();
  ORT_ENFORCE(argument.Exists() && type && type->has_tensor_type() &&
                  type->tensor_type().elem_type() == tensor.data_type() && type->tensor_type().has_shape() &&
                  type->tensor_type().shape().dim_size() == tensor.dims_size() &&
                  !argument.GetInferredShapeValues().has_value() && !argument.GetInferredShapeScalarValue().has_value(),
              "RW07_NATIVE_INITIALIZER_DESCRIPTOR");
  for (int i = 0; i < tensor.dims_size(); ++i) {
    const auto& dimension = type->tensor_type().shape().dim(i);
    ORT_ENFORCE(dimension.has_dim_value() && dimension.dim_value() == tensor.dims(i) && !dimension.has_dim_param(),
                "RW07_NATIVE_INITIALIZER_FIXED_DIMENSION");
  }
}
}  // namespace

void M0FullGpuNativeLoweringOwnerV1::ValidateInitializerPlan(const SessionState& state,const BoundarySourceValidation* proof) const {
  ORT_ENFORCE(source_ && execution_model_ && &state.graph_ == &execution_model_->MainGraph() &&
                  state.m0_native_plan_owner_v1_ == this && state.p_seq_exec_plan_ && state.graph_viewer_ &&
                  state.m0_fullgpu_preinit_snapshot_v1_, "RW07_NATIVE_INITIALIZER_ACTUAL_STATE");
  if(proof)CheckBoundarySourceValidation(*proof);
  else M0OriginalPreloadOwnerV1::ValidateNativeLoweringSource(*source_);
  ORT_THROW_IF_ERROR(M0FullGpuValidateIngressAuthoritiesV1(state.sess_options_));
  const auto& original_graph = source_->MainGraph();
  const auto& original_proto = source_->model_proto_.graph();
  const auto& execution_proto = execution_model_->model_proto_.graph();
  const auto& graph = state.graph_;
  ORT_ENFORCE(original_proto.sparse_initializer_size() == 0 && derived_.graph().sparse_initializer_size() == 0 &&
                  execution_proto.sparse_initializer_size() == 0, "RW07_NATIVE_INITIALIZER_SPARSE");
  const auto& plan = *state.p_seq_exec_plan_;
  const auto* provider = state.execution_providers_.Get(kWebGpuExecutionProvider);
  ORT_ENFORCE(provider && state.execution_providers_.NumProviders() == 1, "RW07_NATIVE_INITIALIZER_PROVIDER");
  const auto device = provider->GetOrtDeviceByMemType(OrtMemTypeDefault);
  ORT_ENFORCE(!device.UsesCpuMemory(), "RW07_NATIVE_INITIALIZER_DEVICE");
  std::map<std::string, const ONNX_NAMESPACE::TensorProto*> originals, expected, backing;
  for (const auto& tensor : original_proto.initializer())
    ORT_ENFORCE(originals.emplace(tensor.name(), &tensor).second, "RW07_NATIVE_INITIALIZER_ORIGINAL_DUPLICATE");
  for (const auto& tensor : derived_.graph().initializer())
    ORT_ENFORCE(expected.emplace(tensor.name(), &ExpectedInitializer(tensor,proof)).second,
                "RW07_NATIVE_INITIALIZER_DERIVED_DUPLICATE");
  for (const auto& tensor : execution_proto.initializer())
    ORT_ENFORCE(backing.emplace(tensor.name(), &tensor).second, "RW07_NATIVE_INITIALIZER_LIVE_DUPLICATE");
  const auto& live = graph.GetAllInitializedTensors();
  ORT_ENFORCE(!expected.empty() && live.size() == expected.size() && backing.size() == expected.size(),
              "RW07_NATIVE_INITIALIZER_EXACT_INVENTORY");
  std::map<std::string, size_t> uses;
  std::map<int, std::string> indices;
  for (const auto& entry : expected) {
    const auto original = originals.find(entry.first);
    const auto actual = live.find(entry.first);
    const auto stored = backing.find(entry.first);
    ORT_ENFORCE(original != originals.end() && actual != live.end() && actual->second && stored != backing.end() &&
                    actual->second == stored->second && original_graph.GetConstantInitializer(entry.first, false) &&
                    graph.GetConstantInitializer(entry.first, false) == actual->second &&
                    !original_graph.IsSparseInitializer(entry.first) && !graph.IsSparseInitializer(entry.first),
                "RW07_NATIVE_INITIALIZER_ORIGINAL_CONSTANT_IDENTITY");
    const auto original_live = original_graph.GetAllInitializedTensors().find(entry.first);
    ORT_ENFORCE(original_live != original_graph.GetAllInitializedTensors().end() && original_live->second &&
                    original_graph.GetConstantInitializer(entry.first, false) == original_live->second &&
                    m0_internal::M0SameInlineInitializerV1(*original_live->second, *original->second) &&
                    m0_internal::M0SameInlineInitializerV1(*entry.second, *original->second) &&
                    m0_internal::M0SameInlineInitializerV1(*actual->second, *original->second),
                "RW07_NATIVE_INITIALIZER_IMMUTABLE_ORIGINAL_BYTES");
    const size_t bytes = CheckPayload(*actual->second);
    ORT_ENFORCE(execution_plan_ && execution_plan_->module && bytes &&
                    bytes <= execution_plan_->module->context_.max_buffer_bytes &&
                    bytes <= execution_plan_->module->context_.max_binding_bytes,
                "RW07_NATIVE_INITIALIZER_PREUPLOAD_RESOURCE_BOUND");
    const auto* argument = graph.GetNodeArg(entry.first);
    const auto* source_argument = original_graph.GetNodeArg(entry.first);
    ORT_ENFORCE(argument && source_argument && argument == value_arguments_.at(entry.first) &&
                    argument->TypeAsProto() && source_argument->TypeAsProto() &&
                    argument->TypeAsProto()->SerializeAsString() == source_argument->TypeAsProto()->SerializeAsString(),
                "RW07_NATIVE_INITIALIZER_ACTUAL_ARGUMENT");
    CheckDescriptor(*argument, *actual->second);
    // This list contains exactly graph.input proto entries (Graph ctor), not
    // every initializer retained in the graph's constant inventory.
    for (const auto* public_input : graph.GetInputsIncludingInitializers())
      ORT_ENFORCE(public_input->Name() != entry.first, "RW07_NATIVE_INITIALIZER_CALLER_OVERRIDE");
    int index = -1;
    ORT_THROW_IF_ERROR(state.ort_value_name_idx_map_.GetIdx(entry.first, index));
    std::string name;
    ORT_THROW_IF_ERROR(state.ort_value_name_idx_map_.GetName(index, name));
    ORT_ENFORCE(name == entry.first && index >= 0 && size_t(index) < plan.allocation_plan.size() &&
                    indices.emplace(index, name).second, "RW07_NATIVE_INITIALIZER_VALUE_INDEX");
    const auto& allocation = plan.allocation_plan[size_t(index)];
    // GeneratePlanForWeights populates this optional profiling field only in
    // ORT_MEMORY_PROFILE builds. The live NodeArg/payload type is checked above.
#if defined(ORT_MEMORY_PROFILE)
    const auto expected_plan_type = DataTypeImpl::TensorTypeFromONNXEnum(actual->second->data_type());
#else
    const MLDataType expected_plan_type = nullptr;
#endif
    ORT_ENFORCE(allocation.alloc_kind == AllocKind::kAllocateStatically && allocation.location == device &&
                    allocation.value_type == expected_plan_type,
                "RW07_NATIVE_INITIALIZER_ALLOCATION_BINDING");
    uses.emplace(entry.first, 0);
  }
  for (const auto& node : graph.Nodes()) {
    CheckNode(node);
    const auto selected = state.kernel_create_info_map_.find(node.Index());
    ORT_ENFORCE(selected != state.kernel_create_info_map_.end() && selected->second && selected->second->kernel_def &&
                    node.GetExecutionProviderType() == kWebGpuExecutionProvider,
                "RW07_NATIVE_INITIALIZER_SELECTED_CONSUMER");
    const auto& definition = *selected->second->kernel_def;
    for (size_t slot = 0; slot < node.InputDefs().size(); ++slot) {
      const auto* argument = node.InputDefs()[slot];
      if (!argument->Exists() || !expected.count(argument->Name())) continue;
      ORT_ENFORCE(argument == graph.GetNodeArg(argument->Name()) &&
                      definition.InputMemoryType(slot) == OrtMemTypeDefault,
                  "RW07_NATIVE_INITIALIZER_GPU_INPUT_SLOT");
      ++uses.at(argument->Name());
    }
  }
  for (const auto& use : uses) ORT_ENFORCE(use.second > 0, "RW07_NATIVE_INITIALIZER_UNREFERENCED_ORIGINAL");
  std::set<int> ordered;
  for (const auto index : plan.initializer_allocation_order)
    ORT_ENFORCE(indices.count(index) && ordered.insert(index).second, "RW07_NATIVE_INITIALIZER_ORDER_BINDING");
  for (const auto index : plan.activation_allocation_order)
    ORT_ENFORCE(!indices.count(index), "RW07_NATIVE_INITIALIZER_ACTIVATION_ALIAS");
  std::set<std::string> observed;
  for (const auto& item : state.m0_fullgpu_preinit_snapshot_v1_->initializers) {
    const auto actual = live.find(item.name);
    ORT_ENFORCE(actual != live.end() && observed.insert(item.name).second && item.constant &&
                    item.owned_tensor_proto == actual->second && indices.count(item.value_index) &&
                    indices.at(item.value_index) == item.name && item.data_type == actual->second->data_type() &&
                    item.data_location == actual->second->data_location() &&
                    item.raw_data_bytes == actual->second->raw_data().size() && item.encoded_bytes == actual->second->ByteSizeLong() &&
                    item.dimensions.size() == size_t(actual->second->dims_size()) &&
                    std::equal(item.dimensions.begin(), item.dimensions.end(), actual->second->dims().begin()),
                "RW07_NATIVE_INITIALIZER_ACTUAL_SNAPSHOT");
  }
  ORT_ENFORCE(observed.size() == expected.size(), "RW07_NATIVE_INITIALIZER_SNAPSHOT_COVERAGE");
  // This proves immutable source bytes and planned consumers, not upload work,
  // runtime buffer alias/lifetime safety, initializer construction or admission.
}
}  // namespace onnxruntime
#endif
