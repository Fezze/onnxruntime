// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_whole_model_seal_v1.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/m0_fixed_wire_v1.h"
#include "sha256.h"

namespace onnxruntime {
namespace {
void CheckTensorEnvelope(const ONNX_NAMESPACE::TensorProto& tensor) {
  ORT_ENFORCE(tensor.external_data_size() == 0 && !tensor.has_data_location() &&
                  !tensor.has_segment() && tensor.metadata_props_size() == 0 && tensor.data_type() > 0,
              "RW07_ORIGINAL_TENSOR_ENVELOPE");
  for (const auto dimension : tensor.dims())
    ORT_ENFORCE(dimension >= 0, "RW07_ORIGINAL_NEGATIVE_LITERAL_DIM");
}

void CheckValueEnvelope(const ONNX_NAMESPACE::ValueInfoProto& value) {
  ORT_ENFORCE(!value.name().empty() && value.metadata_props_size() == 0 && value.has_type() &&
                  value.type().has_tensor_type() && value.type().tensor_type().elem_type() > 0,
              "RW07_ORIGINAL_VALUE_ENVELOPE");
  for (const auto& dimension : value.type().tensor_type().shape().dim())
    ORT_ENFORCE(!dimension.has_dim_value() || dimension.dim_value() >= 0,
                "RW07_ORIGINAL_NEGATIVE_DESCRIPTOR_DIM");
}
}  // namespace

std::unique_ptr<M0WholeModelSealV1> M0WholeModelSealV1::ValidateAndParse(std::string_view bytes) {
  constexpr size_t kOriginalBytes = 325532387;
  constexpr std::string_view kOriginalSha = "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5";
  ORT_ENFORCE(bytes.size() == kOriginalBytes && bytes.data(), "RW07_ORIGINAL_BYTE_SIZE");
  // Hash and parse the owned copy, never the caller's potentially mutable buffer.
  const std::string owned(bytes);
  ORT_ENFORCE(model_package::Sha256::HashStringHex(owned) == kOriginalSha, "RW07_ORIGINAL_BYTE_IDENTITY");
  ORT_ENFORCE(std::string_view(m0_internal::kRw07KnownWireMessages[5].name) == "onnx.ModelProto",
              "RW07_ORIGINAL_WIRE_TABLE");
  {
    // Release scanner copies before protobuf and live Model construction.
    const auto canonical = m0_internal::FixedCanonicalWire(owned, 5);
    ORT_ENFORCE(canonical.size() <= kOriginalBytes, "RW07_ORIGINAL_WIRE_BOUND");
  }
  ONNX_NAMESPACE::ModelProto proto;
  ORT_ENFORCE(proto.ParseFromArray(owned.data(), static_cast<int>(owned.size())), "RW07_ORIGINAL_PROTO_PARSE");
  ORT_ENFORCE(proto.ir_version() == 9 && proto.has_graph() && proto.opset_import_size() == 1 &&
                  proto.opset_import(0).domain().empty() && proto.opset_import(0).version() == 20 &&
                  proto.metadata_props_size() == 0 && proto.training_info_size() == 0 &&
                  proto.functions_size() == 0 && proto.configuration_size() == 0,
              "RW07_ORIGINAL_MODEL_ENVELOPE");
  const auto& graph = proto.graph();
  ORT_ENFORCE(graph.node_size() == 2464 && graph.initializer_size() == 554 && graph.input_size() == 3 &&
                  graph.output_size() == 1 && graph.sparse_initializer_size() == 0 &&
                  graph.quantization_annotation_size() == 0 && graph.metadata_props_size() == 0,
              "RW07_ORIGINAL_GRAPH_ENVELOPE");
  InlinedHashSet<std::string> initializer_names, operators;
  for (const auto& tensor : graph.initializer()) {
    CheckTensorEnvelope(tensor);
    ORT_ENFORCE(!tensor.name().empty() && initializer_names.insert(tensor.name()).second,
                "RW07_ORIGINAL_INITIALIZER_KEY");
  }
  for (const auto& value : graph.input()) CheckValueEnvelope(value);
  for (const auto& value : graph.output()) CheckValueEnvelope(value);
  for (const auto& value : graph.value_info()) CheckValueEnvelope(value);
  size_t attribute_tensors = 0;
  for (const auto& node : graph.node()) {
    ORT_ENFORCE(node.domain().empty() && node.overload().empty() && node.metadata_props_size() == 0 &&
                    node.device_configurations_size() == 0,
                "RW07_ORIGINAL_NODE_ENVELOPE");
    operators.insert(node.op_type());
    for (const auto& attribute : node.attribute()) {
      ORT_ENFORCE(!attribute.has_g() && attribute.graphs_size() == 0 && !attribute.has_sparse_tensor() &&
                      attribute.sparse_tensors_size() == 0 && !attribute.has_tp() && attribute.type_protos_size() == 0 &&
                      attribute.tensors_size() == 0 && attribute.ref_attr_name().empty(),
                  "RW07_ORIGINAL_ATTRIBUTE_ENVELOPE");
      if (attribute.has_t()) {
        ORT_ENFORCE(node.op_type() == "ConstantOfShape" && attribute.name() == "value",
                    "RW07_ORIGINAL_ATTRIBUTE_TENSOR");
        CheckTensorEnvelope(attribute.t());
        ++attribute_tensors;
      }
    }
  }
  ORT_ENFORCE(operators.size() == 49 && attribute_tensors == 6, "RW07_ORIGINAL_SURFACE_INVENTORY");
  // Protobuf owns its parsed fields; the authenticated serialized copy is no
  // longer needed when the caller constructs the separate live Model.
  return std::unique_ptr<M0WholeModelSealV1>(new M0WholeModelSealV1(std::move(proto)));
}
}  // namespace onnxruntime
#endif
