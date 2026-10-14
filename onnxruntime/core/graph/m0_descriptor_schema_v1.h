// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/graph/schema_registry.h"
#include "onnx/defs/shape_inference.h"

namespace onnxruntime::m0_internal {
inline void FixedNestedDescriptorV1(ONNX_NAMESPACE::InferenceContext& context) {
  auto* inferencer = context.getGraphAttributeInferencer("body");
  ORT_ENFORCE(inferencer != nullptr, "fixed body inferencer missing");
  const auto outputs = inferencer->doInferencing({context.getInputType(0)}, {});
  ORT_ENFORCE(outputs.size() == 1 && outputs[0] != nullptr, "fixed body output missing");
  *context.getOutputType(0) = *outputs[0];
}

// Fresh fixed schemas; stored callbacks grant no owner or generation permission.
inline std::shared_ptr<OnnxRuntimeOpSchemaRegistry> FixedDescriptorRegistryV1() {
  using ONNX_NAMESPACE::AttributeProto;
  using ONNX_NAMESPACE::OpSchema;
  constexpr const char* domain = "rw07.synthetic.descriptor.contract1";
  const auto* donor = ONNX_NAMESPACE::OpSchemaRegistry::Schema("Identity", 19, "");
  ORT_ENFORCE(donor && donor->Name() == "Identity" && donor->domain().empty() &&
                  donor->SinceVersion() == 19 && donor->has_type_and_shape_inference_function() &&
                  !donor->has_data_propagation_function() && !donor->HasFunction() &&
                  !donor->HasContextDependentFunction(),
              "fixed Identity19 donor differs");
  OpSchema identity("DIdentity", __FILE__, __LINE__);
  identity.SetDomain(domain).SinceVersion(1).Input(0, "X", "Fixed FLOAT input", "tensor(float)").Output(0, "Y", "Fixed FLOAT output", "tensor(float)").Attr("marker", "Ignored canonical marker", AttributeProto::INT, int64_t{7}).TypeAndShapeInferenceFunction(donor->GetTypeAndShapeInferenceFunction());
  OpSchema nested("DNested", __FILE__, __LINE__);
  nested.SetDomain(domain).SinceVersion(1).Input(0, "X", "Fixed FLOAT input", "tensor(float)").Output(0, "Y", "Fixed FLOAT output", "tensor(float)").Attr("marker", "Ignored canonical marker", AttributeProto::INT, int64_t{7}).Attr("body", "Fixed explicit-port body", AttributeProto::GRAPH, true).TypeAndShapeInferenceFunction(FixedNestedDescriptorV1);
  ORT_ENFORCE(identity.Name() != nested.Name() && !identity.HasFunction() && !nested.HasFunction() &&
                  !identity.HasContextDependentFunction() && !nested.HasContextDependentFunction(),
              "fixed schema recipe differs");
  auto local = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  ORT_ENFORCE(local->empty(), "local registry must start empty");
  std::vector<OpSchema> schemas{std::move(identity), std::move(nested)};
  ORT_THROW_IF_ERROR(local->RegisterOpSet(schemas, domain, 0, 1));
  SchemaRegistryManager manager;
  manager.RegisterRegistry(local);
  for (const auto* name : {"DIdentity", "DNested"}) {
    const OpSchema* selected = nullptr;
    int earliest = 0;
    manager.GetSchemaAndHistory(name, 1, domain, &selected, &earliest);
    ORT_ENFORCE(selected && selected->Name() == name && selected->domain() == domain &&
                    selected->SinceVersion() == 1 && earliest == 1 &&
                    selected->has_type_and_shape_inference_function() &&
                    !selected->has_data_propagation_function() && !selected->HasFunction() &&
                    !selected->HasContextDependentFunction(),
                "fixed schema readback differs");
  }
  return local;
}
}  // namespace onnxruntime::m0_internal
