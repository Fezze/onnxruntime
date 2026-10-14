// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/model.h"
#include "core/graph/model_editor_api_types.h"
#include "core/graph/model_helpers.h"
#include "core/graph/model_load_utils.h"

#ifdef _MSC_VER
#pragma warning(push)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable : 4800)
#endif
#include <google/protobuf/io/coded_stream.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "core/util/protobuf_parsing_utils.h"

#include <gsl/gsl>

#include "core/platform/env.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/schema_registry.h"
#include "core/graph/function_utils.h"
#include "core/graph/function_impl.h"
#include "core/graph/m0_fixed_wire_v1.h"
#include "core/graph/m0_descriptor_schema_v1.h"
#include "core/graph/runtime_optimization_record_container.h"
#endif

#if defined(__wasm__)
#include <emscripten.h>
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime;
using namespace onnxruntime::common;

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)

void Model::RemoveLocalFunctionsProtos(const InlinedHashSet<std::string>& retained) {
  auto* local_functions = model_proto_.mutable_functions();
  if (retained.empty()) {
    model_local_function_templates_maps_.clear();
    model_local_functions_.clear();
    local_functions->erase(local_functions->begin(), local_functions->end());
  } else {
    const auto retained_end = retained.cend();
    for (auto it = model_local_functions_.begin();
         it != model_local_functions_.end();) {
      if (retained.find(it->first) == retained_end) {
        model_local_function_templates_maps_.erase(it->first);
        it = model_local_functions_.erase(it);
      } else {
        ++it;
      }
    }

    for (auto it = local_functions->begin(); it != local_functions->end();) {
      const auto function_id = function_utils::GetFunctionIdentifier(it->domain(), it->name(), it->overload());
      if (retained.find(function_id) == retained_end) {
        it = local_functions->erase(it);
      } else {
        ++it;
      }
    }
  }
}

static constexpr int DEFAULT_PROTOBUF_BLOCK_SIZE = 4 * 1024 * 1024;

static ModelProto ValidateAndCopyModelProto(const ModelProto& model_proto, const ModelOptions& options) {
  ORT_THROW_IF_ERROR(options.m0_callback_deny_requested_v1
                         ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER")
                         : Status::OK());
  ORT_THROW_IF_ERROR(ValidateModelSubgraphDepth(model_proto));
  return model_proto;
}

// Validate before constructing Model, where function schemas are inferred and errors cannot return a Status.
static Status ValidateFunctionNodeCounts(const ModelProto& model_proto) {
  const auto* schema_registry = OpSchemaRegistry::Instance();
  for (const auto& function : model_proto.functions()) {
    InlinedHashMap<std::string, int> opset_imports;
    for (const auto& opset : function.opset_import()) {
      ORT_RETURN_IF_NOT(opset.version() > 0 && opset.version() <= std::numeric_limits<int>::max(),
                        "Invalid opset version for domain ", opset.domain(), " in function ", function.name());
      opset_imports[opset.domain()] = static_cast<int>(opset.version());
    }

    using NodeRange = google::protobuf::RepeatedPtrField<NodeProto>;
    InlinedVector<const NodeRange*> pending{&function.node()};
    auto add_subgraphs = [&](const auto& attributes) {
      for (const auto& attribute : attributes) {
        if (attribute.has_g()) {
          pending.push_back(&attribute.g().node());
        }
        for (const auto& graph : attribute.graphs()) {
          pending.push_back(&graph.node());
        }
      }
    };
    add_subgraphs(function.attribute_proto());

    while (!pending.empty()) {
      const auto* nodes = pending.back();
      pending.pop_back();
      for (const auto& node : *nodes) {
        const auto opset = opset_imports.find(node.domain());
        ORT_RETURN_IF_NOT(opset != opset_imports.end(),
                          "No opset registered for domain ", node.domain(), " in function ", function.name());
        if (const auto* schema = schema_registry->GetSchema(node.op_type(), opset->second, node.domain())) {
          ORT_RETURN_IF_NOT(node.input_size() >= schema->min_input() && node.input_size() <= schema->max_input(),
                            "Invalid input count for op ", node.op_type(), " in function ", function.name(),
                            ": expected ", schema->min_input(), " to ", schema->max_input(),
                            ", got ", node.input_size());
          ORT_RETURN_IF_NOT(node.output_size() >= schema->min_output() && node.output_size() <= schema->max_output(),
                            "Invalid output count for op ", node.op_type(), " in function ", function.name(),
                            ": expected ", schema->min_output(), " to ", schema->max_output(),
                            ", got ", node.output_size());
        }
        add_subgraphs(node.attribute());
      }
    }
  }

  return Status::OK();
}

Model::Model(const std::string& graph_name,
             bool is_onnx_domain_only,
             const ModelMetaData& model_metadata,
             const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList& local_registries,
             const std::unordered_map<std::string, int>& domain_to_version,
             const std::vector<ONNX_NAMESPACE::FunctionProto>& model_local_functions,
             const logging::Logger& logger,
             const ModelOptions& options)
    : m0_callback_policy_v1_(options.m0_callback_deny_requested_v1),
      model_path_(model_path),
      check_load_cancellation_fn_(options.check_load_cancellation_fn) {
  ORT_THROW_IF_ERROR(M0CheckCallbackPolicyV1(M0CallbackPolicyV1::Channel::kConstruction));
  model_proto_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model_proto_.mutable_graph()->set_name(graph_name);
  model_metadata_ = model_metadata;
  for (auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop{model_proto_.add_metadata_props()};
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  for (const auto& schema_collection : local_registries) {
    schema_registry->RegisterRegistry(schema_collection);
  }

  // IsAllowReleasedONNXOpsetsOnlySet() checks for the appropriate env var in the process (i.e.) process-wide
  // `allow_released_opsets_only` is for this specific Model instance
  // We will only support released opsets iff IsAllowReleasedONNXOpsetsOnlySet() and `allow_released_opsets_only`
  // are both true
  auto allow_released_opsets_only_final =
      options.allow_released_opsets_only && model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

  auto* p_domain_to_version = &domain_to_version;
  DomainToVersionMap domain_to_version_static;
  domain_to_version_static = allow_released_opsets_only_final
                                 ? schema_registry->GetLastReleasedOpsetVersions(is_onnx_domain_only)
                                 : schema_registry->GetLatestOpsetVersions(is_onnx_domain_only);
  if (p_domain_to_version->empty()) {
    p_domain_to_version = &domain_to_version_static;
  }

  for (const auto& [domain, version] : *p_domain_to_version) {
    model_load_utils::ValidateOpsetForDomain(domain_to_version_static, logger, allow_released_opsets_only_final,
                                             domain, version);
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model_proto_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }

  for (const auto& func : model_local_functions) {
    ORT_THROW_IF_ERROR(ValidateFunctionSubgraphDepth(func));
  }

  model_local_functions_.reserve(model_local_functions.size());
  for (auto& func : model_local_functions) {
    auto function_id = function_utils::GetFunctionIdentifier(func.domain(), func.name(), func.overload());
    ORT_ENFORCE(model_local_functions_.find(function_id) == model_local_functions_.end(),
                "Duplicate model-local function identifier: ", function_id);

    auto func_ptr = model_proto_.add_functions();
    func_ptr->CopyFrom(func);
    model_local_functions_.emplace(std::move(function_id), func_ptr);
  }

  ORT_THROW_IF_ERROR(ValidateModelSubgraphDepth(model_proto_));
  ORT_THROW_IF_ERROR(ValidateModelLocalFunctionAcyclic(model_local_functions_));

  model_local_function_templates_maps_.reserve(model_proto_.functions().size());
  for (auto& func : model_proto_.functions()) {
    auto func_schema_ptr = function_utils::CreateSchema(func.domain(),
                                                        func.name(),
                                                        func.overload(),
                                                        model_local_functions_,
                                                        *p_domain_to_version,
                                                        *schema_registry,
                                                        logger,
                                                        allow_released_opsets_only_final, this);
    auto func_template_ptr = std::make_unique<FunctionTemplate>();
    func_template_ptr->op_schema_ = std::move(func_schema_ptr);
    func_template_ptr->onnx_func_proto_ = &func;
    model_local_function_templates_maps_.insert_or_assign(function_utils::GetFunctionIdentifier(func.domain(),
                                                                                                func.name(),
                                                                                                func.overload()),
                                                          std::move(func_template_ptr));
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(*this, model_proto_.mutable_graph(), *p_domain_to_version, IrVersion(), schema_registry,
                         logger, options.strict_shape_type_inference));
}

Model::Model(const ModelProto& model_proto, const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger,
             const ModelOptions& options)
    : Model(ValidateAndCopyModelProto(model_proto, options), model_path, local_registries, logger, options) {
}

Model::Model(ModelProto&& model_proto, const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
             const logging::Logger& logger, const ModelOptions& options)
    : Model(std::move(model_proto), model_path, local_registries, logger, options, nullptr) {}

Model::Model(ModelProto&& model_proto, const PathString& model_path,
             const IOnnxRuntimeOpSchemaRegistryList* local_registries,
             const logging::Logger& logger, const ModelOptions& options,
             std::unique_ptr<M0SyntheticOwnerV1> synthetic_owner)
    : m0_synthetic_owner_v1_(std::move(synthetic_owner)),
      m0_callback_policy_v1_(options.m0_callback_deny_requested_v1),
      model_path_(model_path),
      check_load_cancellation_fn_(options.check_load_cancellation_fn) {
  ORT_THROW_IF_ERROR(M0CheckCallbackPolicyV1(M0CallbackPolicyV1::Channel::kConstruction));
  if (!utils::HasGraph(model_proto)) {
    ORT_THROW("ModelProto does not have a graph.");
  }

  if (model_proto.opset_import_size() == 0) {
    ORT_THROW(
        "Missing opset in the model. All ModelProtos MUST have at least one entry that"
        " specifies which version of the ONNX OperatorSet is being imported.");
  }

  if (!model_proto.has_ir_version()) {
    ORT_THROW("Missing model IR version.");
  }

  if (const auto ir_version = model_proto.ir_version();
      ir_version > ONNX_NAMESPACE::Version::IR_VERSION) {
    ORT_THROW("Unsupported model IR version: ", ir_version,
              ", max supported IR version: ", ONNX_NAMESPACE::Version::IR_VERSION);
  }

  model_proto_ = std::move(model_proto);
  for (auto& prop : model_proto_.metadata_props()) {
    model_metadata_[prop.key()] = prop.value();
  }

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  if (local_registries != nullptr) {
    for (const auto& schema_collection : *local_registries) {
      schema_registry->RegisterRegistry(schema_collection);
    }
  }

  // IsAllowReleasedONNXOpsetsOnlySet() checks for the appropriate env var in the process (i.e.) process-wide
  // `allow_released_opsets_only` is for this specific Model instance
  // We will only support released opsets iff IsAllowReleasedONNXOpsetsOnlySet() and `allow_released_opsets_only`
  // are both true
  auto allow_official_onnx_release_only_final =
      options.allow_released_opsets_only && model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

  const auto onnx_released_versions =
      schema_registry->GetLastReleasedOpsetVersions(false);

  std::unordered_map<std::string, int> domain_to_version;
  for (auto& opSet : model_proto_.opset_import()) {
    const auto& domain = opSet.domain();
    const auto version = gsl::narrow_cast<int>(opSet.version());
    // empty domain and 'ai.onnx' are equivalent
    if ((domain.empty() || domain == kOnnxDomainAlias) && version < 7) {
      // TODO: Check if we can upgrade all the current opset 6 models that are being tested
      // in CI to opset 7 or above
      LOGS(logger, WARNING) << "ONNX Runtime only *guarantees* support for models stamped "
                               "with opset version 7 or above for opset domain 'ai.onnx'. "
                               "Please upgrade your model to opset 7 or higher. "
                               "For now, this opset "
                            << version
                            << " model may run depending upon legacy support "
                               "of some older opset version operators.";
    }

    model_load_utils::ValidateOpsetForDomain(onnx_released_versions, logger,
                                             allow_official_onnx_release_only_final, domain, version);

    // We need to overwrite the domain here with ("") or else the loop below will try to find ("")
    // in the map and if not found (when domain == kOnnxDomainAlias), adds an entry for ("", 11).
    // This effectively ignores the opset version specified by the model for the onnx domain.
    if (domain == kOnnxDomainAlias) {
      domain_to_version[kOnnxDomain] = version;
    } else {
      domain_to_version[domain] = version;
    }
  }

  // special-case the internal NHWC domain as it must match the ONNX opset if not explicitly imported
  if (domain_to_version.find(kMSInternalNHWCDomain) == domain_to_version.end()) {
    auto onnx_version = domain_to_version.find(kOnnxDomain);
    if (onnx_version != domain_to_version.end()) {
      domain_to_version[kMSInternalNHWCDomain] = onnx_version->second;
    }
  }

  auto domain_map = allow_official_onnx_release_only_final
                        ? schema_registry->GetLastReleasedOpsetVersions(false)
                        : schema_registry->GetLatestOpsetVersions(false);
  for (const auto& [domain, version] : domain_map) {
    if (domain_to_version.find(domain) == domain_to_version.end()) {
      domain_to_version[domain] = version;
      const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model_proto_.add_opset_import()};
      opset_id_proto->set_domain(domain);
      opset_id_proto->set_version(version);
    }
  }

  model_local_functions_.reserve(model_proto_.functions().size());
  for (auto& func : model_proto_.functions()) {
    auto function_id = function_utils::GetFunctionIdentifier(func.domain(), func.name(), func.overload());
    const bool inserted = model_local_functions_.emplace(function_id, &func).second;
    ORT_ENFORCE(inserted, "Duplicate model-local function identifier: ", function_id);
  }

  ORT_THROW_IF_ERROR(ValidateModelSubgraphDepth(model_proto_));
  ORT_THROW_IF_ERROR(ValidateModelLocalFunctionAcyclic(model_local_functions_));

  model_local_function_templates_maps_.reserve(model_proto_.functions().size());
  for (auto& func : model_proto_.functions()) {
    auto func_schema_ptr = function_utils::CreateSchema(func.domain(),
                                                        func.name(),
                                                        func.overload(),
                                                        model_local_functions_,
                                                        domain_to_version,
                                                        *schema_registry,
                                                        logger,
                                                        allow_official_onnx_release_only_final, this);
    auto func_template_ptr = std::make_unique<FunctionTemplate>();
    func_template_ptr->op_schema_ = std::move(func_schema_ptr);
    func_template_ptr->onnx_func_proto_ = &func;
    model_local_function_templates_maps_.insert_or_assign(function_utils::GetFunctionIdentifier(func.domain(),
                                                                                                func.name(),
                                                                                                func.overload()),
                                                          std::move(func_template_ptr));
  }

  // create instance. need to call private ctor so can't use make_unique
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(*this, model_proto_.mutable_graph(), domain_to_version, IrVersion(), schema_registry,
                         logger, options.strict_shape_type_inference));
}

const NodeHashMap<std::string, std::unique_ptr<FunctionTemplate>>& Model::GetModelLocalFunctionTemplates() const {
  return model_local_function_templates_maps_;
}

Status Model::ValidateLocalFunctionCallDepth(const Graph& graph) const {
  return ValidateModelLocalFunctionCallDepth(model_local_functions_, graph);
}

Version Model::IrVersion() const {
  if (utils::HasIrVersion(model_proto_)) {
    return model_proto_.ir_version();
  }
  return kNoVersion;
}

const std::string Model::ProducerName() const {
  if (model_proto_.has_producer_name()) {
    return model_proto_.producer_name();
  }
  return std::string();
}

void Model::SetProducerName(const std::string& producer_name) {
  model_proto_.set_producer_name(producer_name);
}

const std::string Model::ProducerVersion() const {
  if (model_proto_.has_producer_version()) {
    return model_proto_.producer_version();
  }
  return std::string();
}

void Model::SetProducerVersion(const std::string& producer_version) {
  model_proto_.set_producer_version(producer_version);
}

const std::string Model::Domain() const {
  if (model_proto_.has_domain()) {
    return model_proto_.domain();
  }
  return std::string();
}

void Model::SetDomain(const std::string& domain) {
  model_proto_.set_domain(domain);
}

Version Model::ModelVersion() const {
  if (utils::HasModelVersion(model_proto_)) {
    return model_proto_.model_version();
  }
  return kNoVersion;
}

void Model::SetModelVersion(onnxruntime::Version version) {
  model_proto_.set_model_version(version);
}

const std::string Model::DocString() const {
  if (model_proto_.has_doc_string()) {
    return model_proto_.doc_string();
  }
  return std::string();
}

void Model::SetDocString(const std::string& doc_string) {
  model_proto_.set_doc_string(doc_string);
}

const std::string Model::GraphDocString() const {
  if (model_proto_.has_graph() && model_proto_.graph().has_doc_string()) {
    return model_proto_.graph().doc_string();
  }
  return std::string();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

const ModelMetaData& Model::MetaData() const noexcept {
  return model_metadata_;
}

ModelMetaData& Model::MetaData() noexcept {
  return model_metadata_;
}

Graph& Model::MainGraph() noexcept {
  return *graph_;
}

const Graph& Model::MainGraph() const noexcept {
  return *graph_;
}

#if !defined(ORT_MINIMAL_BUILD)
ModelProto Model::ToProto() const {
  // We want to return back the original proto
  // To that end invoke const overload of ToGraphProto()
  // that returns by value and, therefore, allows us to filter
  // out dense duplicates of sparse initializers and leave the original
  // proto intact.
  ModelProto result(model_proto_);

  // Sync current model_metadata_ back to protobuf metadata_props
  result.clear_metadata_props();
  for (const auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop{result.add_metadata_props()};
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  const auto& graph = *graph_;
  *(result.mutable_graph()) = graph.ToGraphProto();
  return result;
}

ModelProto Model::ToGraphProtoWithExternalInitializers(const std::filesystem::path& external_file_name,
                                                       const std::filesystem::path& file_path,
                                                       const ModelSavingOptions& model_saving_options) const {
  ModelProto result(model_proto_);

  // Sync current model_metadata_ back to protobuf metadata_props
  result.clear_metadata_props();
  for (const auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop{result.add_metadata_props()};
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  const auto& graph = *graph_;
  *(result.mutable_graph()) = graph.ToGraphProtoWithExternalInitializers(external_file_name,
                                                                         file_path,
                                                                         model_saving_options);
  return result;
}

common::Status Model::ToGraphProtoWithExternalInitializers(
    const std::filesystem::path& external_file_name,
    const ModelSavingOptions& model_saving_options,
    std::ostream& external_stream,
    ONNX_NAMESPACE::ModelProto& model_proto) const {
  model_proto = model_proto_;
  model_proto.clear_metadata_props();
  for (const auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> property{model_proto.add_metadata_props()};
    property->set_key(metadata.first);
    property->set_value(metadata.second);
  }

  return graph_->ToGraphProtoWithExternalInitializers(external_file_name, model_saving_options,
                                                      external_stream, *model_proto.mutable_graph());
}

common::Status Model::ToGraphProtoWithCustomInitializerHandling(OrtGetInitializerLocationFunc handle_initializer_func,
                                                                void* state,
                                                                /*out*/ ONNX_NAMESPACE::ModelProto& model_proto) const {
  model_proto = model_proto_;

  // Sync current model_metadata_ back to protobuf metadata_props
  model_proto.clear_metadata_props();
  for (const auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop{model_proto.add_metadata_props()};
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  const auto& graph = *graph_;
  ORT_RETURN_IF_ERROR(graph.ToGraphProtoWithCustomInitializerHandling(handle_initializer_func,
                                                                      state, *model_proto.mutable_graph()));
  return Status::OK();
}

Status Model::Load(std::istream& model_istream, ModelProto* p_model_proto) {
  if (!model_istream.good()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid istream object.");
  }
  if (!p_model_proto) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Null model_proto ptr.");
  }

  google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
  const bool result = p_model_proto->ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Failed to load model because protobuf parsing failed.");
  }
  return Status::OK();
}

Status Model::Load(const ModelProto& model_proto,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger,
                   const ModelOptions& options) {
  return Model::Load(model_proto, PathString{}, model, local_registries, logger, options);
}

Status Model::Load(const ModelProto& model_proto,
                   const PathString& model_path,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger,
                   const ModelOptions& options) {
  ORT_RETURN_IF_ERROR(options.m0_callback_deny_requested_v1
                          ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER")
                          : Status::OK());
  // we expect a graph to be present
  if (!utils::HasGraph(model_proto)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  ORT_RETURN_IF_ERROR(ValidateFunctionNodeCounts(model_proto));

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)

  auto status = Status::OK();
  ORT_TRY {
    model = std::make_unique<Model>(model_proto, model_path, local_registries, logger, options);
  }
  ORT_CATCH(const OnnxRuntimeException& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ex.Category(), ex.Code(), ex.what());
    });
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to load model with error: " + std::string(ex.what()));
    });
  }
  ORT_RETURN_IF_ERROR(status);

  Graph::ResolveOptions resolve_options;
  resolve_options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve(resolve_options));

  return status;
}

Status Model::Load(ModelProto&& model_proto,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger,
                   const ModelOptions& options) {
  return Model::Load(std::move(model_proto), PathString{}, model, local_registries, logger, options);
}

Status Model::Load(ModelProto&& model_proto,
                   const PathString& model_path,
                   std::shared_ptr<Model>& model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger,
                   const ModelOptions& options) {
  ORT_RETURN_IF_ERROR(options.m0_callback_deny_requested_v1
                          ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER")
                          : Status::OK());
  // we expect a graph to be present
  if (!utils::HasGraph(model_proto)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  ORT_RETURN_IF_ERROR(ValidateFunctionNodeCounts(model_proto));

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  auto status = Status::OK();
  ORT_TRY {
    model = std::make_unique<Model>(std::move(model_proto), model_path, local_registries, logger, options);
  }
  ORT_CATCH(const OnnxRuntimeException& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ex.Category(), ex.Code(), ex.what());
    });
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to load model with error: " + std::string(ex.what()));
    });
  }
  ORT_RETURN_IF_ERROR(status);

  Graph::ResolveOptions resolve_options;
  resolve_options.no_proto_sync_required = true;
  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve(resolve_options));

  return status;
}

template <typename T, typename Loader>
static Status LoadModelHelper(const T& file_path, Loader loader) {
  int fd;
  Status status = Env::Default().FileOpenRd(file_path, fd);
  if (!status.IsOK()) {
    if (status.Category() == common::SYSTEM) {
      switch (status.Code()) {
        case ENOENT:
          return ORT_MAKE_STATUS(ONNXRUNTIME, NO_SUCHFILE, "Load model ", ToUTF8String(file_path),
                                 " failed. File doesn't exist");
        case EINVAL:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Load model ", ToUTF8String(file_path), " failed");
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "system error number ", status.Code());
      }
    }
  }

  ORT_TRY {
    status = loader(fd);
  }
  ORT_CATCH(const OnnxRuntimeException& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ex.Category(), ex.Code(), ex.what());
    });
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, FAIL, ex.what());
    });
  }

  if (!status.IsOK()) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return status;
  }
  return Env::Default().FileClose(fd);
}

template <typename T>
static Status LoadModel(const T& file_path, ONNX_NAMESPACE::ModelProto& model_proto) {
  const auto loader = [&model_proto](int fd) {
    return Model::Load(fd, model_proto);
  };

  return LoadModelHelper(file_path, loader);
}

template <typename T>
static Status LoadModel(const T& file_path, std::shared_ptr<Model>& p_model,
                        const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                        const logging::Logger& logger, const ModelOptions& options) {
  ORT_RETURN_IF_ERROR(options.m0_callback_deny_requested_v1
                          ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER")
                          : Status::OK());
  const auto loader = [&file_path, &p_model, local_registries, &logger, &options](int fd) {
    return Model::Load(fd, ToPathString(file_path), p_model, local_registries, logger, options);
  };

  return LoadModelHelper(file_path, loader);
}

template <typename T>
static Status SaveModel(Model& model, const T& file_path) {
#if defined(__wasm__) && defined(ORT_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL)
  ORT_RETURN_IF_ERROR(model.MainGraph().Resolve());
  auto model_proto = model.ToProto();
  auto buffer_size = model_proto.ByteSizeLong();
  void* buffer = malloc(buffer_size);
  model_proto.SerializeToArray(buffer, buffer_size);

  EM_ASM(({
           const buffer = Number($0);
           const buffer_size = Number($1);
           const file_path = UTF8ToString($2);
           const bytes = new Uint8Array(buffer_size);
           bytes.set(HEAPU8.subarray(buffer, buffer + buffer_size));
           if (typeof process == 'object' && typeof process.versions == 'object' &&
               typeof process.versions.node == 'string') {
             // Node.js
             require('fs').writeFileSync(file_path, bytes);
           } else {
             // Browser
             const file = new File([bytes], file_path, {type: "application/octet-stream" });
             const url = URL.createObjectURL(file);
             window.open(url, '_blank');
           }
         }),
         buffer,
         buffer_size,
         file_path.c_str());

  free(buffer);
  return Status::OK();

#else
  int fd;
  Status status = Env::Default().FileOpenWr(file_path, fd);
  ORT_RETURN_IF_ERROR(status);

  ORT_TRY {
    status = Model::Save(model, fd);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, FAIL, ex.what());
    });
  }
  if (!status.IsOK()) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return status;
  }
  return Env::Default().FileClose(fd);
#endif
}

Status Model::Save(Model& model, const PathString& file_path) {
  return SaveModel(model, file_path);
}

template <typename T>
static Status SaveModelWithExternalInitializers(Model& model,
                                                const T& file_path,
                                                const std::filesystem::path& external_file_name,
                                                const ModelSavingOptions& save_options) {
  int fd = 0;
  Status status = Env::Default().FileOpenWr(file_path, fd);
  ORT_RETURN_IF_ERROR(status);

  ORT_TRY {
    status = Model::SaveWithExternalInitializers(model, fd, file_path, external_file_name, save_options);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, FAIL, ex.what());
    });
  }
  if (!status.IsOK()) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return status;
  }
  return Env::Default().FileClose(fd);
}

Status Model::Load(const PathString& file_path,
                   ONNX_NAMESPACE::ModelProto& model_proto) {
  return LoadModel(file_path, model_proto);
}

GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const PathString& file_path, std::shared_ptr<Model>& p_model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger, const ModelOptions& options) {
  return LoadModel(file_path, p_model, local_registries, logger, options);
}

GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const PathString& file_path, const PathString& graph_model_path,
                   std::shared_ptr<Model>& p_model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger, const ModelOptions& options) {
  ORT_RETURN_IF_ERROR(options.m0_callback_deny_requested_v1
                          ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER")
                          : Status::OK());
  const auto loader = [&graph_model_path, &p_model, local_registries, &logger, &options](int fd) {
    return Model::Load(fd, graph_model_path, p_model, local_registries, logger, options);
  };

  return LoadModelHelper(file_path, loader);
}

Status Model::SaveWithExternalInitializers(Model& model, const std::filesystem::path& file_path,
                                           const std::filesystem::path& external_file_name,
                                           const ModelSavingOptions& save_options) {
  return SaveModelWithExternalInitializers(model, file_path, external_file_name, save_options);
}

Status Model::LoadFromBytes(int count, const void* p_bytes, /*out*/ ONNX_NAMESPACE::ModelProto& model_proto) {
  const bool result = model_proto.ParseFromArray(p_bytes, count);
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  return Status::OK();
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ std::shared_ptr<Model>& p_model,
                            const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger,
                            const ModelOptions& options) {
  return LoadFromBytes(count, p_bytes, PathString{}, p_model, local_registries, logger, options);
}

Status Model::LoadFromBytes(int count, void* p_bytes, const PathString& model_path,
                            std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                            const logging::Logger& logger, const ModelOptions& options) {
  ORT_RETURN_IF_ERROR(options.m0_callback_deny_requested_v1
                          ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER")
                          : Status::OK());
  ModelProto model_proto;

  auto status = LoadFromBytes(count, p_bytes, model_proto);
  if (!status.IsOK()) {
    return status;
  }

  return Load(std::move(model_proto), model_path, p_model, local_registries, logger, options);
}

using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::ZeroCopyInputStream;

Status Model::Load(int fd, ONNX_NAMESPACE::ModelProto& model_proto) {
  if (fd < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "<p_fd> less than 0.");
  }

#if GOOGLE_PROTOBUF_VERSION >= 3002000
  size_t file_size = 0;
  int block_size = -1;
  Status st = Env::Default().GetFileLength(fd, file_size);
  if (st.IsOK()) {
    block_size = std::min(DEFAULT_PROTOBUF_BLOCK_SIZE, static_cast<int>(file_size));
  }
  FileInputStream input(fd, block_size);
  const bool result = model_proto.ParseFromZeroCopyStream(&input) && input.GetErrno() == 0;
  if (!result) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }
#else
  // CNTK uses ORT as a submodule in order to use its GraphIR code.
  // CNTK needs to be built with protobuf 3.1.0 for its version specific features.
  // This code block is needed to support CNTK and any other
  // GraphIR client that will be built with protobuf at a version older than 3.2.0.
  FileInputStream fs(fd);
  CodedInputStream cis(&fs);

  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  cis.SetTotalBytesLimit(INT_MAX);
  if (!model_proto->ParseFromCodedStream(&cis)) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }
#endif
  return Status::OK();
}

Status Model::Load(int fd, std::shared_ptr<Model>& p_model, const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                   const logging::Logger& logger, const ModelOptions& options) {
  return Load(fd, PathString{}, p_model, local_registries, logger, options);
}

Status Model::Load(int fd, const PathString& model_path, std::shared_ptr<Model>& p_model,
                   const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger,
                   const ModelOptions& options) {
  ORT_RETURN_IF_ERROR(options.m0_callback_deny_requested_v1
                          ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER")
                          : Status::OK());
  ModelProto model_proto;

  ORT_RETURN_IF_ERROR(Load(fd, model_proto));

  return Load(std::move(model_proto), model_path, p_model, local_registries, logger, options);
}

// static
common::Status Model::LoadFromModelEditorApiModel(const OrtModel& model_editor_api_model,
                                                  const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                                                  const ModelOptions& options,
                                                  const logging::Logger& logger,
                                                  std::unique_ptr<Model>& model) {
  ORT_RETURN_IF_ERROR(options.m0_callback_deny_requested_v1
                          ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER")
                          : Status::OK());
  model = std::make_unique<Model>();
  model->model_proto_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  // The optimizer Initializer class requires a path if external data is used, however in Model Editor API usage the
  // external data is pointing to pre-allocated memory and does not require a path. Set a dummy value to make it happy.
  model->model_path_ = std::filesystem::path("_MODEL_EDITOR_API_MODEL_");

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  if (local_registries != nullptr) {
    for (const auto& schema_collection : *local_registries) {
      schema_registry->RegisterRegistry(schema_collection);
    }
  }

  // add the other domains that might be used by ORT internally.
  auto domain_to_version = model_editor_api_model.domain_to_version;  // copy
  auto allow_official_onnx_release_only_final = options.allow_released_opsets_only &&
                                                model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

  for (const auto& [domain, version] : domain_to_version) {
    model_load_utils::ValidateOpsetForDomain(schema_registry->GetLastReleasedOpsetVersions(false), logger,
                                             allow_official_onnx_release_only_final, domain, version);
  }

  // convert kOnnxDomainAlias to kOnnxDomain if needed and remove the alias entry
  if (auto onnx_alias_iter = domain_to_version.find(kOnnxDomainAlias); onnx_alias_iter != domain_to_version.end()) {
    if (domain_to_version.find(kOnnxDomain) == domain_to_version.end()) {
      domain_to_version[kOnnxDomain] = onnx_alias_iter->second;
    }

    domain_to_version.erase(onnx_alias_iter);
  }

  if (auto onnx_iter = domain_to_version.find(kOnnxDomain); onnx_iter == domain_to_version.end()) {
    // ONNX domain must be explicitly specified as we can't simply default to the latest opset ORT knows about.
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "The opset for the ONNX domain must be explicitly specified.");
  }

  // special-case the internal NHWC domain as it must match the ONNX opset if not explicitly imported
  if (domain_to_version.find(kMSInternalNHWCDomain) == domain_to_version.end()) {
    auto onnx_version = domain_to_version.find(kOnnxDomain);
    if (onnx_version != domain_to_version.end()) {
      domain_to_version[kMSInternalNHWCDomain] = onnx_version->second;
    }
  }

  auto domain_map = allow_official_onnx_release_only_final ? schema_registry->GetLastReleasedOpsetVersions(false)
                                                           : schema_registry->GetLatestOpsetVersions(false);

  // Merge in the other domains ORT may use internally, without overriding opsets the caller supplied
  // explicitly (e.g. the ONNX domain). Only missing domains get the registry version from domain_map.
  for (const auto& [domain, version] : domain_map) {
    if (domain_to_version.find(domain) == domain_to_version.end()) {
      domain_to_version[domain] = version;
    }
  }

  // Populate the model proto's opset imports from the merged map (not domain_map) so they stay consistent
  // with the graph's domain-to-version map and preserve the caller's explicit opsets.
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model->model_proto_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }

  ORT_RETURN_IF_ERROR(Graph::LoadFromModelEditorApiModel(*model_editor_api_model.graph,
                                                         *model,
                                                         domain_to_version,
                                                         schema_registry,
                                                         options.strict_shape_type_inference,
                                                         logger,
                                                         model->graph_));

  return Status::OK();
}

Status Model::Save(Model& model, int p_fd) {
  if (p_fd < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "<p_fd> is less than 0.");
  }

  ORT_RETURN_IF_ERROR(model.MainGraph().Resolve());

  auto model_proto = model.ToProto();
  google::protobuf::io::FileOutputStream output(p_fd);
  const bool result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
  if (result) {
    return Status::OK();
  }
  return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf serialization failed.");
}

Status Model::SaveWithExternalInitializers(Model& model,
                                           int fd,
                                           const std::filesystem::path& file_path,
                                           const std::filesystem::path& external_file_name,
                                           const ModelSavingOptions& model_saving_options) {
  if (fd < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "<fd> is less than 0.");
  }

  ORT_RETURN_IF_ERROR(model.MainGraph().Resolve());

  auto model_proto = model.ToGraphProtoWithExternalInitializers(external_file_name, file_path,
                                                                model_saving_options);
  google::protobuf::io::FileOutputStream output(fd);
  const bool result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
  if (result) {
    return Status::OK();
  }
  return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Protobuf serialization failed.");
}

common::Status Model::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                      flatbuffers::Offset<fbs::Model>& fbs_model) const {
  auto producer_name = fbs::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_producer_name(), model_proto_.producer_name());
  auto producer_version = fbs::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_producer_version(), model_proto_.producer_version());
  auto domain = builder.CreateSharedString(model_proto_.domain());
  auto doc_string = fbs::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_doc_string(), model_proto_.doc_string());
  auto graph_doc_string = fbs::utils::SaveStringToOrtFormat(
      builder, model_proto_.has_graph() && model_proto_.graph().has_doc_string(), model_proto_.graph().doc_string());

  std::vector<flatbuffers::Offset<fbs::OperatorSetId>> op_set_ids_vec;
  op_set_ids_vec.reserve(model_proto_.opset_import().size());
  for (const auto& entry : model_proto_.opset_import()) {
    auto op_set_domain = builder.CreateSharedString(entry.domain());
    fbs::OperatorSetIdBuilder ob(builder);
    ob.add_domain(op_set_domain);
    ob.add_version(entry.version());
    op_set_ids_vec.push_back(ob.Finish());
  }
  auto op_set_ids = builder.CreateVector(op_set_ids_vec);

  flatbuffers::Offset<flatbuffers::Vector<
      flatbuffers::Offset<onnxruntime::fbs::StringStringEntry>>>
      metadata_props{0};

  // We will not serialize an empty metadata_props
  if (!model_metadata_.empty()) {
    std::vector<flatbuffers::Offset<onnxruntime::fbs::StringStringEntry>> metadata_props_vec;
    metadata_props_vec.reserve(model_metadata_.size());
    for (const auto& prop : model_metadata_) {
      metadata_props_vec.push_back(
          fbs::CreateStringStringEntryDirect(builder, prop.first.c_str(), prop.second.c_str()));
    }
    metadata_props = builder.CreateVector(metadata_props_vec);
  }

  flatbuffers::Offset<fbs::Graph> fbs_graph;
  ORT_RETURN_IF_ERROR(graph_->SaveToOrtFormat(builder, fbs_graph));

  fbs::ModelBuilder mb(builder);
  mb.add_ir_version(IrVersion());
  mb.add_opset_import(op_set_ids);
  mb.add_producer_name(producer_name);
  mb.add_producer_version(producer_version);
  mb.add_domain(domain);
  mb.add_model_version(ModelVersion());
  mb.add_doc_string(doc_string);
  mb.add_graph_doc_string(graph_doc_string);
  mb.add_metadata_props(metadata_props);
  mb.add_graph(fbs_graph);

  // add graph
  fbs_model = mb.Finish();

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

Model::Model() : model_path_{} {
}

common::Status Model::LoadFromOrtFormat(const fbs::Model& fbs_model,
#if !defined(ORT_MINIMAL_BUILD)
                                        const IOnnxRuntimeOpSchemaRegistryList* local_registries,
#endif
                                        const OrtFormatLoadOptions& load_options,
                                        const logging::Logger& logger,
                                        std::unique_ptr<Model>& model,
                                        bool callback_deny_requested) {
  ORT_RETURN_IF(callback_deny_requested, "RW07_CALLBACK_DENY_UNREVIEWED_OWNER");
  model = std::make_unique<Model>();

  // Load the model metadata
  if (const auto* fbs_metadata_props = fbs_model.metadata_props()) {
    model->model_metadata_.reserve(fbs_metadata_props->size());
    for (const auto* prop : *fbs_metadata_props) {
      ORT_RETURN_IF(nullptr == prop, "Null entry in metadata_props. Invalid ORT format model.");
      std::string key, value;
      fbs::utils::LoadStringFromOrtFormat(key, prop->key());
      fbs::utils::LoadStringFromOrtFormat(value, prop->value());
      model->model_metadata_.insert({key, value});
    }
  }

#if !defined(ORT_MINIMAL_BUILD)
  LOAD_STR_FROM_ORT_FORMAT(model->model_proto_, producer_name, fbs_model.producer_name());
  LOAD_STR_FROM_ORT_FORMAT(model->model_proto_, producer_version, fbs_model.producer_version());
  LOAD_STR_FROM_ORT_FORMAT(model->model_proto_, domain, fbs_model.domain());
  LOAD_STR_FROM_ORT_FORMAT(model->model_proto_, doc_string, fbs_model.doc_string());
  if (fbs_model.graph_doc_string()) {
    model->model_proto_.mutable_graph()->set_doc_string(fbs_model.graph_doc_string()->c_str());
  }
  model->model_proto_.set_model_version(fbs_model.model_version());
  model->model_proto_.set_ir_version(fbs_model.ir_version());

  auto schema_registry = std::make_shared<SchemaRegistryManager>();
  if (local_registries != nullptr) {
    for (const auto& schema_collection : *local_registries) {
      schema_registry->RegisterRegistry(schema_collection);
    }
  }

  // Populate the metadata to model_proto
  for (auto& metadata : model->model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop{model->model_proto_.add_metadata_props()};
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }
#else
  fbs::utils::LoadStringFromOrtFormat(model->producer_name_, fbs_model.producer_name());
  fbs::utils::LoadStringFromOrtFormat(model->producer_version_, fbs_model.producer_version());
  fbs::utils::LoadStringFromOrtFormat(model->domain_, fbs_model.domain());
  fbs::utils::LoadStringFromOrtFormat(model->doc_string_, fbs_model.doc_string());
  fbs::utils::LoadStringFromOrtFormat(model->graph_doc_string_, fbs_model.graph_doc_string());
  model->model_version_ = fbs_model.model_version();
  model->ir_version_ = fbs_model.ir_version();
#endif

  std::unordered_map<std::string, int> domain_to_version;
  ORT_RETURN_IF_ERROR(fbs::utils::LoadOpsetImportOrtFormat(fbs_model.opset_import(), domain_to_version));

  auto fbs_graph = fbs_model.graph();
  ORT_RETURN_IF(nullptr == fbs_graph, "Graph is null. Invalid ORT format model.");

#if !defined(ORT_MINIMAL_BUILD)
  // add the opset imports to the model_proto in case we're updating an ORT format model and need those to be
  // included when SaveToOrtFormat is called later
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto{model->model_proto_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }

  ORT_RETURN_IF_ERROR(Graph::LoadFromOrtFormat(*fbs_graph, *model, domain_to_version, schema_registry,
                                               load_options, logger, model->graph_));
#else
  ORT_RETURN_IF_ERROR(Graph::LoadFromOrtFormat(*fbs_graph, *model, domain_to_version,
                                               load_options, logger, model->graph_));
#endif
  return Status::OK();
}
}  // namespace onnxruntime

namespace onnxruntime {
// CLOSED source-private lifecycle/generation implementation; no descriptor callback permission.
M0GraphRegistrationV1::M0GraphRegistrationV1(const Model& model, const Graph* graph,
                                             const ONNX_NAMESPACE::GraphProto* proto,
                                             const Graph* parent, const Node* parent_node)
    : owner_(model.m0_synthetic_owner_v1_.get()) {
  if (owner_) scope_id_ = owner_->Register(graph, proto, parent, parent_node);
}
M0GraphRegistrationV1::~M0GraphRegistrationV1() noexcept {
  if (owner_) owner_->Unregister(scope_id_);
}
M0SyntheticOwnerV1::~M0SyntheticOwnerV1() noexcept {
  if (active_.load(std::memory_order_acquire) || stack_size_ != 0) Poison();
  for (const auto& entry : entries_) {
    if (entry.graph || (entry.scope_id != 0 && !entry.retired)) Poison();
  }
  audit_->owner_retired = !active_.load(std::memory_order_acquire) && stack_size_ == 0 &&
                          std::all_of(entries_.begin(), entries_.end(), [](const auto& entry) { return !entry.graph; });
  state_ = State::kRetired;
  audit_->poisoned = Poisoned();
}
void M0GraphRegistrationV1::Finish() {
  if (owner_) owner_->Finish(scope_id_);
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_NO_EXCEPTIONS)
namespace {
uint64_t M0FreshOwnerNonce() {
  static std::atomic<uint64_t> next{1};
  auto nonce = next.load(std::memory_order_relaxed);
  for (;;) {
    ORT_ENFORCE(nonce != std::numeric_limits<uint64_t>::max(), "RW07_OWNER_NONCE_EXHAUSTED");
    if (next.compare_exchange_weak(nonce, nonce + 1, std::memory_order_relaxed)) return nonce;
  }
}
std::string M0GraphCanonical(const ONNX_NAMESPACE::GraphProto& proto) {
  int id = -1;
  for (size_t index = 0; index < std::size(m0_internal::kRw07KnownWireMessages); ++index) {
    if (std::string_view(m0_internal::kRw07KnownWireMessages[index].name) == "onnx.GraphProto") {
      id = static_cast<int>(index);
    }
  }
  ORT_ENFORCE(id >= 0, "RW07_GRAPH_BINDING_MISSING");
  return m0_internal::FixedCanonicalWire(proto.SerializeAsString(), id);
}
std::string M0ModelCanonical(const ONNX_NAMESPACE::ModelProto& proto) {
  for (size_t index = 0; index < std::size(m0_internal::kRw07KnownWireMessages); ++index) {
    if (std::string_view(m0_internal::kRw07KnownWireMessages[index].name) == "onnx.ModelProto") {
      return m0_internal::FixedCanonicalWire(proto.SerializeAsString(), static_cast<int>(index));
    }
  }
  ORT_THROW("RW07_MODEL_BINDING_MISSING");
}
constexpr std::pair<std::string_view, int> kM0NormalizedImports[] = {
    {"", 27}, {"ai.onnx.ml", 5}, {"ai.onnx.preview", 1}, {"ai.onnx.preview.training", 1}, {"ai.onnx.training", 1}, {"com.microsoft", 1}, {"com.microsoft.experimental", 1}, {"com.microsoft.nchwc", 1}, {"com.ms.internal.nhwc", 26}, {"org.localtts.m0.rw07", 1}, {"org.pytorch.aten", 1}, {"rw07.synthetic.descriptor.contract1", 1}};
void M0RequireNormalizedImports(const Graph& graph) {
  const auto& imports = graph.DomainToVersionMap();
  ORT_ENFORCE(imports.size() == std::size(kM0NormalizedImports), "RW07_IMPORT_COUNT");
  for (const auto& [domain, version] : kM0NormalizedImports) {
    auto it = imports.find(std::string(domain));
    ORT_ENFORCE(it != imports.end() && it->second == version, "RW07_IMPORT_TUPLE");
  }
}
}  // namespace

M0SyntheticOwnerV1::M0SyntheticOwnerV1(std::shared_ptr<M0SyntheticOwnerReceiptV1> audit)
    : audit_(std::move(audit)), nonce_(M0FreshOwnerNonce()) {
  audit_->owner_nonce = nonce_;
  ONNX_NAMESPACE::ModelProto recipe;
  ORT_ENFORCE(recipe.ParseFromString(m0_internal::CheckedFixedBytes("nested-depth2.model-original")),
              "RW07_FIXED_MODEL_PARSE");
  auto normalized = recipe;
  normalized.clear_opset_import();
  for (const auto& [domain, version] : kM0NormalizedImports) {
    auto* import = normalized.add_opset_import();
    import->set_domain(std::string(domain));
    import->set_version(version);
  }
  normalized_model_canonical_ = M0ModelCanonical(normalized);
  original_function_ = m0_internal::CheckedFixedBytes("parentless.function-original");
  const auto& root = recipe.graph();
  const auto& body1 = root.node(0).attribute(0).g();
  const auto& body2 = body1.node(0).attribute(0).g();
  const std::array<const ONNX_NAMESPACE::GraphProto*, 3> scopes{&root, &body1, &body2};
  for (size_t index = 0; index < scopes.size(); ++index) {
    entries_[index].name = scopes[index]->name();
    entries_[index].original_canonical = M0GraphCanonical(*scopes[index]);
  }
  entries_[1].parent_slot = 0;
  entries_[1].parent_node = "outer";
  entries_[2].parent_slot = 1;
  entries_[2].parent_node = "inner";
  entries_[3].name = "parentless";
}

uint64_t M0SyntheticOwnerV1::Register(const Graph* graph, const ONNX_NAMESPACE::GraphProto* proto,
                                      const Graph* parent, const Node* parent_node) {
  ORT_ENFORCE(!active_.load(std::memory_order_acquire), "RW07_OWNER_REGISTER_ACTIVE");
  auto fail = [&]() { Poison(); ORT_THROW("RW07_OWNER_UNLISTED_CONSTRUCTION"); };
  if (Poisoned() || state_ != State::kConstructing || !proto || !graph) fail();
  size_t slot = entries_.size();
  for (size_t i = 0; i < 3; ++i) {
    if (proto->name() == entries_[i].name) slot = i;
  }
  if (proto->SerializeAsString().empty() && parentless_transition_) slot = 3;
  if (slot == entries_.size()) fail();
  auto& entry = entries_[slot];
  if (entry.graph || entry.retired) fail();
  if (slot == 1 || slot == 2) {
    if (parent != entries_[entry.parent_slot].graph || !parent || !parent_node ||
        parent_node->Name() != entry.parent_node) fail();
  } else if (parent || parent_node) {
    fail();
  }
  if (slot != 3 && M0GraphCanonical(*proto) != entry.original_canonical) fail();
  if (next_scope_id_ == std::numeric_limits<uint64_t>::max()) fail();
  entry.scope_id = next_scope_id_++;
  entry.owner_nonce = nonce_;
  entry.graph = graph;
  entry.backing = proto;
  entry.original_backing = proto->SerializeAsString();
  ++audit_->registered;
  ++audit_->live_at_capture;
  if (slot == 3) parentless_transition_ = false;
  return entry.scope_id;
}

void M0SyntheticOwnerV1::Finish(uint64_t scope_id) {
  for (size_t slot = 0; slot < entries_.size(); ++slot) {
    auto& entry = entries_[slot];
    if (entry.scope_id != scope_id) continue;
    ORT_ENFORCE(entry.graph && !entry.constructed, "RW07_OWNER_FINISH_STATE");
    if (slot == fail_after_slot_) {
      Poison();
      ORT_THROW("RW07_FIXED_CONSTRUCTOR_FAILURE");
    }
    entry.schema_collection = entry.graph->schema_registry_.get();
    entry.constructed = true;
    return;
  }
  Poison();
  ORT_THROW("RW07_OWNER_FINISH_UNLISTED");
}

void M0SyntheticOwnerV1::Unregister(uint64_t scope_id) noexcept {
  if (active_.load(std::memory_order_acquire)) Poison();
  for (auto& entry : entries_) {
    if (entry.scope_id != scope_id || !entry.graph) continue;
    entry.graph = nullptr;
    entry.constructed = false;
    entry.retired = true;
    ++audit_->unregistered;
    --audit_->live_at_capture;
    audit_->poisoned = Poisoned();
    return;
  }
  Poison();
  audit_->poisoned = true;
}

M0SyntheticOwnerV1::Generation::Generation(M0SyntheticOwnerV1& owner_value) noexcept : owner(owner_value) {
  bool expected = false;
  if (!owner.active_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    owner.Poison();
    return;
  }
  acquired = true;
  if (owner.generation_id_ == std::numeric_limits<uint64_t>::max()) {
    owner.Poison();
    return;
  }
  id = ++owner.generation_id_;
  ++owner.audit_->generation_acquisitions;
  if (owner.Poisoned()) return;
  if (owner.state_ != State::kReady) {
    owner.Poison();
    return;
  }
  owner.state_ = State::kResolving;
}
M0SyntheticOwnerV1::Generation::~Generation() noexcept {
  if (!acquired) return;
  if (owner.stack_size_ != 0) owner.Poison();
  owner.state_ = owner.Poisoned() ? State::kPoisoned : State::kReady;
  owner.audit_->poisoned = owner.Poisoned();
  owner.active_.store(false, std::memory_order_release);
}
M0SyntheticOwnerV1::Frame::Frame(Generation& generation_value, const Graph* graph_value)
    : generation(generation_value), graph(graph_value) {
  auto& owner = generation.owner;
  ORT_ENFORCE(generation.acquired && generation.id == owner.generation_id_ && !owner.Poisoned(),
              "RW07_OWNER_FRAME_GENERATION");
  bool registered = false;
  for (const auto& entry : owner.entries_) registered |= entry.graph == graph && entry.constructed;
  bool duplicate = false;
  for (size_t index = 0; index < owner.stack_size_; ++index) duplicate |= owner.stack_[index] == graph;
  if (!graph || !registered || duplicate || owner.stack_size_ == owner.stack_.size()) {
    owner.Poison();
    ORT_THROW("RW07_OWNER_FRAME_REENTRY_OR_FOREIGN");
  }
  owner.stack_[owner.stack_size_++] = graph;
  ++owner.audit_->frame_pushes;
  pushed = true;
}
M0SyntheticOwnerV1::Frame::~Frame() noexcept {
  if (!pushed) return;
  auto& owner = generation.owner;
  if (!generation.acquired || generation.id != owner.generation_id_ || owner.stack_size_ == 0 ||
      owner.stack_[owner.stack_size_ - 1] != graph) {
    owner.Poison();
    return;
  }
  owner.stack_[--owner.stack_size_] = nullptr;
  ++owner.audit_->frame_pops;
}

void M0SyntheticOwnerV1::ValidateAll(const Graph* requested) {
  ValidateProjection(requested, 0);
}

void M0SyntheticOwnerV1::ValidateBacking(const Model& model) {
  ORT_ENFORCE(model_identity_ == &model && model.graph_.get() == entries_[0].graph &&
                  model.model_path_.empty() && !model.check_load_cancellation_fn_ &&
                  M0ModelCanonical(model.model_proto_) == normalized_model_canonical_,
              "RW07_OWNER_MODEL_BACKING");
  ORT_ENFORCE(parentless_identity_ && &parentless_identity_->function_body_graph_ == entries_[3].graph &&
                  &parentless_identity_->function_storage_proto_ == entries_[3].backing &&
                  original_function_ == m0_internal::CheckedFixedBytes("parentless.function-original"),
              "RW07_OWNER_FUNCTION_LINEAGE");
  for (size_t slot = 0; slot < entries_.size(); ++slot) {
    const auto& entry = entries_[slot];
    const auto& graph = *entry.graph;
    ORT_ENFORCE(graph.nodes_.size() == 1 && graph.nodes_[0] &&
                    graph.nodes_[0]->InputDefs().size() == 1 && graph.nodes_[0]->InputDefs()[0] &&
                    graph.nodes_[0]->OutputDefs().size() == 1 && graph.nodes_[0]->OutputDefs()[0],
                "RW07_OWNER_BACKING_NODE_PORTS");
    if (slot == 0) {
      ORT_ENFORCE(entry.backing == &model.model_proto_.graph(), "RW07_OWNER_ROOT_BACKING_LINK");
    } else if (slot < 3) {
      const auto* parent = entries_[entry.parent_slot].graph;
      const auto& attributes = parent->nodes_[0]->GetAttributes();
      const auto body = attributes.find("body");
      ORT_ENFORCE(body != attributes.end() && body->second.has_g() && &body->second.g() == entry.backing,
                  "RW07_OWNER_CHILD_BACKING_LINK");
    }
    // Establish the owning protobuf relationship before following a recorded
    // body pointer: removing/replacing the parent attribute can destroy it.
    ORT_ENFORCE(entry.backing && graph.graph_proto_ == entry.backing &&
                    entry.backing->SerializeAsString() == entry.original_backing,
                "RW07_OWNER_GRAPH_BACKING");
    ORT_ENFORCE(graph.deserialized_proto_data_.SerializeAsString().empty() &&
                    !graph.prepacked_key_to_blobs_ && !graph.prepacked_weights_for_graph_ &&
                    graph.runtime_optimizations_ptr_ &&
                    &graph.runtime_optimizations_ == graph.runtime_optimizations_ptr_.get() &&
                    graph.runtime_optimizations_.IsEmpty() &&
                    !graph.node_replacement_callback_ && !graph.node_removal_callback_ &&
                    graph.name_generator_ == 0 && graph.generated_node_names_.empty() &&
                    graph.generated_node_arg_names_.empty() && graph.num_resolves_ == 0 &&
                    !graph.strict_shape_type_inference_,
                "RW07_OWNER_PROTECTED_CHANNELS");
    ORT_ENFORCE(std::all_of(std::begin(graph.weight_data_type_freq_), std::end(graph.weight_data_type_freq_),
                            [](int32_t count) { return count == 0; }),
                "RW07_OWNER_WEIGHT_COUNTS");
    const auto& context = graph.resolve_context_;
    bool inferred_child_expected = false;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
    inferred_child_expected = issued_generation_ && issued_inferred_child_[slot];
#endif
    ORT_ENFORCE(inferred_child_expected
                    ? slot < 2 && context.inferred_subgraphs.size() == 1 &&
                          context.inferred_subgraphs.contains(entries_[slot + 1].graph)
                    : context.inferred_subgraphs.empty(),
                "RW07_OWNER_INFERRED_CHILD_SET");
    const auto* node = graph.nodes_[0].get();
    const auto* expected_original = slot < 3 ? &entry.backing->node(0) : nullptr;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
    if (issued_generation_ && issued_node_stage_[slot] >= IssuedNodeStage::kChecked) expected_original = nullptr;
#endif
    ORT_ENFORCE(node->GetOriginalNodeProto() == expected_original,
                "RW07_OWNER_ORIGINAL_NODE_LINEAGE");
    if (preparation_[slot] == Preparation::kOriginal) {
      ORT_ENFORCE(context.output_args.empty() && context.inputs_and_initializers.empty() &&
                      context.node_name_to_index.empty() && context.nodes_with_subgraphs.empty(),
                  "RW07_OWNER_INITIAL_RESOLVE_CONTEXT");
    } else {
      const auto output = context.output_args.find(node->OutputDefs()[0]->Name());
      const auto name = context.node_name_to_index.find(node->Name());
      ORT_ENFORCE(context.output_args.size() == 1 && output != context.output_args.end() &&
                      output->second.first == node && output->second.second == 0 &&
                      context.inputs_and_initializers.size() == 1 &&
                      context.inputs_and_initializers.contains(node->InputDefs()[0]->Name()) &&
                      context.node_name_to_index.size() == 1 && name != context.node_name_to_index.end() &&
                      name->second == 0 && context.nodes_with_subgraphs.size() == (slot < 2 ? 1u : 0u) &&
                      (slot >= 2 || context.nodes_with_subgraphs.contains(const_cast<Node*>(node))),
                  "RW07_OWNER_PREPARED_RESOLVE_CONTEXT");
    }
    if (preparation_[slot] < Preparation::kConnected) {
      ORT_ENFORCE(graph.node_arg_to_producer_node_.empty() && graph.node_arg_to_consumer_nodes_.empty(),
                  "RW07_OWNER_PREMATURE_LOOKUPS");
    } else {
      const auto producer = graph.node_arg_to_producer_node_.find(node->OutputDefs()[0]->Name());
      const auto consumers = graph.node_arg_to_consumer_nodes_.find(node->InputDefs()[0]->Name());
      ORT_ENFORCE(graph.node_arg_to_producer_node_.size() == 1 &&
                      producer != graph.node_arg_to_producer_node_.end() && producer->second == 0 &&
                      graph.node_arg_to_consumer_nodes_.size() == 1 &&
                      consumers != graph.node_arg_to_consumer_nodes_.end() &&
                      consumers->second == std::unordered_set<NodeIndex>{0},
                  "RW07_OWNER_PREPARED_LOOKUPS");
    }
  }
}

void M0SyntheticOwnerV1::ValidateProjection(const Graph* requested, size_t mechanical_phase) {
  try {
    // The phase argument is private mechanical fixture state, never callback authority.
    ORT_ENFORCE(mechanical_phase <= 7 && state_ == State::kResolving, "RW07_OWNER_PROJECTION_PHASE");
    ORT_ENFORCE(active_.load(std::memory_order_acquire) && !Poisoned(), "RW07_OWNER_VALIDATION_GENERATION");
    bool found = false;
    for (const auto& entry : entries_) found |= requested && requested == entry.graph && entry.constructed;
    if (!found) {
      Poison();
      ORT_THROW("RW07_OWNER_REQUESTED_SCOPE");
    }
    ++audit_->full_validations;
    const auto& model = requested->GetModel();
    ORT_ENFORCE(model.model_local_functions_.empty() && model.model_local_function_templates_maps_.empty() &&
                    model.model_metadata_.empty() && model.model_proto_.functions_size() == 0 &&
                    model.model_proto_.training_info_size() == 0 && model.model_proto_.metadata_props_size() == 0,
                "RW07_OWNER_MODEL_CHANNELS");
    // All entries must be live before the independent backing traversal.
    for (const auto& entry : entries_) {
      ORT_ENFORCE(entry.graph && entry.constructed && !entry.retired, "RW07_OWNER_SCOPE_SET");
    }
    ValidateBacking(model);
    ONNX_NAMESPACE::GraphProto phased_root;
    if (mechanical_phase != 0) {
      constexpr const char* phases[] = {
          "nested-depth2.expected-sequence.outer-default", "nested-depth2.expected-sequence.inner-default",
          "nested-depth2.expected-sequence.leaf-default", "nested-depth2.expected-sequence.leaf-output",
          "nested-depth2.expected-sequence.inner-output", "nested-depth2.expected-sequence.outer-output",
          "nested-depth2.expected-live-post"};
      ORT_ENFORCE(phased_root.ParseFromString(m0_internal::CheckedFixedBytes(phases[mechanical_phase - 1])),
                  "RW07_OWNER_PHASE_PARSE");
    }
    size_t slot = 0;
    for (const auto& entry : entries_) {
      if (!entry.graph || !entry.constructed || entry.retired) {
        Poison();
        ORT_THROW("RW07_OWNER_SCOPE_SET");
      }
      const auto& graph = *entry.graph;
      ORT_ENFORCE(graph.GetModel().m0_synthetic_owner_v1_.get() == this, "RW07_OWNER_FOREIGN_MODEL");
      ORT_ENFORCE(entry.owner_nonce == nonce_ && entry.scope_id != 0 &&
                      graph.schema_registry_.get() == entry.schema_collection && sealed_registry_,
                  "RW07_OWNER_SCOPE_BINDING");
      M0RequireNormalizedImports(graph);
      const auto latest = graph.schema_registry_->GetLatestOpsetVersions(false);
      ORT_ENFORCE(latest.size() == std::size(kM0NormalizedImports), "RW07_OWNER_REGISTRY_DOMAINS");
      for (const auto& [domain, version] : kM0NormalizedImports) {
        const auto found_domain = latest.find(std::string(domain));
        ORT_ENFORCE(found_domain != latest.end() && found_domain->second == version, "RW07_OWNER_REGISTRY_IMPORT");
      }
      for (size_t schema_index = 0; schema_index < sealed_schemas_.size(); ++schema_index) {
        const auto* name = schema_index == 0 ? "DIdentity" : "DNested";
        const ONNX_NAMESPACE::OpSchema* selected = nullptr;
        int earliest = 0;
        graph.schema_registry_->GetSchemaAndHistory(name, 1, "rw07.synthetic.descriptor.contract1", &selected, &earliest);
        ORT_ENFORCE(selected && selected == sealed_schemas_[schema_index] && earliest == 1 &&
                        selected->Name() == name && selected->domain() == "rw07.synthetic.descriptor.contract1" &&
                        selected->SinceVersion() == 1 && selected->inputs().size() == 1 && selected->outputs().size() == 1 &&
                        selected->has_type_and_shape_inference_function() && !selected->has_data_propagation_function() &&
                        !selected->HasFunction() && !selected->HasContextDependentFunction() &&
                        selected->attributes().size() == (schema_index == 0 ? 1u : 2u),
                    "RW07_OWNER_SEALED_SCHEMA");
        auto check_formal = [](const auto& formal, const char* name) {
          ORT_ENFORCE(formal.GetName() == name && formal.GetTypeStr() == "tensor(float)" &&
                          formal.GetDescription().empty() && static_cast<int>(formal.GetOption()) == 0 &&
                          formal.GetMinArity() == 1 && formal.GetIsHomogeneous() &&
                          static_cast<int>(formal.GetDifferentiationCategory()) == 0 &&
                          formal.GetTypes().size() == 1 && *formal.GetTypes().begin() &&
                          **formal.GetTypes().begin() == "tensor(float)",
                      "RW07_OWNER_SCHEMA_FORMAL");
        };
        check_formal(selected->inputs()[0], "X");
        check_formal(selected->outputs()[0], "Y");
        ORT_ENFORCE(selected->typeConstraintParams().empty() && selected->typeConstraintMap().empty() &&
                        selected->min_input() == 1 && selected->max_input() == 1 &&
                        selected->min_output() == 1 && selected->max_output() == 1 && !selected->deprecated() &&
                        static_cast<int>(selected->support_level()) == 0 &&
                        selected->function_opset_versions().empty() &&
                        selected->context_dependent_function_opset_versions().empty(),
                    "RW07_OWNER_SCHEMA_CONSTRAINTS");
        const auto marker = selected->attributes().find("marker");
        ORT_ENFORCE(marker != selected->attributes().end() && !marker->second.required &&
                        marker->second.type == ONNX_NAMESPACE::AttributeProto::INT &&
                        marker->second.default_value.SerializeAsString() == m0_internal::CheckedFixedBytes("marker7"),
                    "RW07_OWNER_SCHEMA_DEFAULT");
        if (schema_index == 1) {
          const auto body = selected->attributes().find("body");
          ORT_ENFORCE(body != selected->attributes().end() && body->second.required &&
                          body->second.type == ONNX_NAMESPACE::AttributeProto::GRAPH &&
                          body->second.default_value.SerializeAsString().empty(),
                      "RW07_OWNER_SCHEMA_BODY");
        }
      }
      ONNX_NAMESPACE::GraphProto expected;
      ORT_ENFORCE(expected.ParseFromString(entry.original_canonical), "RW07_OWNER_EXPECTED_PARSE");
      const auto original = expected;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
      if (issued_generation_ && issued_node_stage_[slot] >= IssuedNodeStage::kPublished) {
        expected.mutable_output(0)->mutable_type()->CopyFrom(expected.input(0).type());
      }
      if (issued_generation_ && issued_node_stage_[slot] >= IssuedNodeStage::kDefaulted) {
        auto* marker = expected.mutable_node(0)->add_attribute();
        ORT_ENFORCE(marker->ParseFromString(m0_internal::CheckedFixedBytes("marker7")), "RW07_ISSUED_MARKER_PARSE");
      }
#endif
      if (mechanical_phase != 0 && slot < 3) {
        const ONNX_NAMESPACE::GraphProto* phase_graph = &phased_root;
        for (size_t depth = 0; depth < slot; ++depth) {
          const ONNX_NAMESPACE::GraphProto* child = nullptr;
          for (const auto& attr : phase_graph->node(0).attribute()) {
            if (attr.name() == "body") child = &attr.g();
          }
          ORT_ENFORCE(child, "RW07_OWNER_PHASE_BODY");
          phase_graph = child;
        }
        expected = *phase_graph;
      }
      ORT_ENFORCE(graph.Name() == expected.name(), "RW07_OWNER_NAME");
      ORT_ENFORCE(graph.nodes_.size() == static_cast<size_t>(expected.node_size()), "RW07_OWNER_NODE_COUNT");
      ORT_ENFORCE(graph.ir_version_ == 8 && graph.num_of_nodes_ == expected.node_size() &&
                      graph.nodes_in_topological_order_ ==
                          (preparation_[slot] == Preparation::kSorted ? std::vector<NodeIndex>{0} : std::vector<NodeIndex>{}) &&
                      graph.name_to_initial_tensor_.empty() && graph.ortvalue_initializers_.empty() &&
                      graph.sparse_tensor_names_.empty() && graph.graph_overridable_initializers_.empty() &&
                      graph.outer_scope_node_arg_names_.empty() && graph.fused_schemas_containers_.empty() &&
                      graph.reusable_fused_schema_map_.empty() && graph.generated_shared_prepack_ids_.empty(),
                  "RW07_OWNER_GRAPH_CHANNELS");
      ORT_ENFORCE(graph.is_loaded_from_model_file_ == (slot < 3) &&
                      graph.graph_inputs_manually_set_ == (slot == 3) && graph.graph_outputs_manually_set_ == (slot == 3),
                  "RW07_OWNER_GRAPH_CONSTRUCTION");
      const auto* parent = slot == 1 || slot == 2 ? entries_[entry.parent_slot].graph : nullptr;
      ORT_ENFORCE(graph.parent_graph_ == parent &&
                      graph.parent_node_ == (parent ? parent->nodes_[0].get() : nullptr),
                  "RW07_OWNER_PARENT_LINK");
      auto check_ports = [&](const auto& ports, const auto& recipe_ports) {
        ORT_ENFORCE(ports.size() == static_cast<size_t>(recipe_ports.size()), "RW07_OWNER_GRAPH_PORT_COUNT");
        for (int port = 0; port < recipe_ports.size(); ++port) {
          const auto it = graph.node_args_.find(recipe_ports.Get(port).name());
          ORT_ENFORCE(it != graph.node_args_.end() && it->second && ports[port] &&
                          ports[port] == it->second.get() && ports[port]->Exists(),
                      "RW07_OWNER_GRAPH_PORT_IDENTITY");
        }
      };
      check_ports(graph.graph_inputs_including_initializers_, expected.input());
      check_ports(graph.graph_outputs_, expected.output());
      if (slot < 3 || preparation_[slot] != Preparation::kOriginal)
        check_ports(graph.graph_inputs_excluding_initializers_, expected.input());
      else
        ORT_ENFORCE(graph.graph_inputs_excluding_initializers_.empty(), "RW07_OWNER_PARENTLESS_INPUT_PHASE");
      ORT_ENFORCE(graph.value_info_.empty() && expected.value_info_size() == 0, "RW07_OWNER_VALUE_INFO");
      InlinedHashMap<std::string, std::string> types;
      for (const auto& value : expected.input()) types.emplace(value.name(), value.SerializeAsString());
      for (const auto& value : expected.output()) types.emplace(value.name(), value.SerializeAsString());
      for (const auto& value : expected.value_info()) types.emplace(value.name(), value.SerializeAsString());
      ORT_ENFORCE(graph.node_args_.size() == types.size(), "RW07_OWNER_NODEARG_INVENTORY");
      for (const auto& [name, arg] : graph.node_args_) {
        const auto it = types.find(name);
        ORT_ENFORCE(it != types.end() && arg && arg->TypeAsProto() &&
                        arg->ToProto().SerializeAsString() == it->second &&
                        !arg->GetInferredShapeScalarValue().has_value() &&
                        !arg->GetInferredShapeValues().has_value(),
                    "RW07_OWNER_NODEARG_CURRENT");
      }
      for (int index = 0; index < expected.node_size(); ++index) {
        const auto& node = graph.nodes_[static_cast<size_t>(index)];
        const auto& recipe = expected.node(index);
        const ONNX_NAMESPACE::OpSchema* expected_op = nullptr;
        int expected_version = -1;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
        if (issued_generation_ && issued_node_stage_[slot] >= IssuedNodeStage::kSelected) {
          expected_op = sealed_schemas_[slot < 2 ? 1 : 0];
          expected_version = 1;
        }
#endif
        ORT_ENFORCE(node && node->Index() == static_cast<NodeIndex>(index) && node->GetContainingGraph() == &graph &&
                        node->InputArgCount() == std::vector<int>{1} && node->SinceVersion() == expected_version && node->Overload() == recipe.overload() &&
                        node->Description() == recipe.doc_string() && node->GetFunctionBody() == nullptr &&
                        node->GetExecutionProviderType().empty() && node->ControlInputs().empty() &&
                        node->GetInputEdgesCount() == 0 && node->GetOutputEdgesCount() == 0 &&
                        node->Name() == recipe.name() && node->OpType() == recipe.op_type() &&
                        node->Domain() == recipe.domain() && node->Op() == expected_op && node->ImplicitInputDefs().empty() &&
                        node->InputDefs().size() == static_cast<size_t>(recipe.input_size()) &&
                        node->OutputDefs().size() == static_cast<size_t>(recipe.output_size()) &&
                        node->GetAttributes().size() == static_cast<size_t>(recipe.attribute_size()),
                    "RW07_OWNER_NODE_CURRENT");
        for (int port = 0; port < recipe.input_size(); ++port) {
          ORT_ENFORCE(node->InputDefs()[port] == graph.GetNodeArg(recipe.input(port)) && node->InputDefs()[port]->Exists(), "RW07_OWNER_INPUT_SLOT");
        }
        for (int port = 0; port < recipe.output_size(); ++port) {
          ORT_ENFORCE(node->OutputDefs()[port] == graph.GetNodeArg(recipe.output(port)) && node->OutputDefs()[port]->Exists(), "RW07_OWNER_OUTPUT_SLOT");
        }
        const auto children = node->GetAttributeNameToSubgraphMap();
        ORT_ENFORCE(children.size() == (slot < 2 ? 1u : 0u), "RW07_OWNER_CHILD_COUNT");
        if (slot < 2) {
          const auto child = children.find("body");
          ORT_ENFORCE(child != children.end() && child->second.get() == entries_[slot + 1].graph,
                      "RW07_OWNER_CHILD_LINK");
        }
        for (const auto& attr : recipe.attribute()) {
          const auto it = node->GetAttributes().find(attr.name());
          ORT_ENFORCE(it != node->GetAttributes().end() && it->first == it->second.name() && it->second.SerializeAsString() == [&]() {
                          if (attr.name() != "body") return attr.SerializeAsString();
                          for (const auto& original_attr : original.node(index).attribute()) {
                            if (original_attr.name() == "body") return original_attr.SerializeAsString();
                          }
                          return std::string{}; }(), "RW07_OWNER_ATTRIBUTE_CURRENT");
        }
      }
      ++slot;
    }
  } catch (...) {
    Poison();
    throw;
  }
}

common::Status M0SyntheticOwnerV1::ResolveClosed(const Graph* requested) {
  Generation generation(*this);
  if (!generation.acquired || Poisoned()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_SYNTHETIC_OWNER_CONFLICT_OR_POISON");
  }
  try {
    ValidateAll(requested);
    if (preparation_audit_) return PrepareClosed(requested, generation);
    Frame frame(generation, requested);
    ++audit_->resolve_denials;
    Poison();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_SYNTHETIC_DESCRIPTOR_GATE_CLOSED");
  } catch (const std::exception& error) {
    Poison();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error.what());
  }
}
bool M0SyntheticOwnerV1::DenyCallback() noexcept {
  Poison();
  ++audit_->callback_denials;
  return true;
}

common::Status M0SyntheticOwnerV1::PrepareClosed(const Graph* requested, Generation& generation) {
  const auto* root = requested == entries_[3].graph ? entries_[3].graph : entries_[0].graph;
  Frame frame(generation, root);
  if (requested != root) ++preparation_audit_->redirected_requests;
  PrepareSorted(requested);
  ++audit_->resolve_denials;
  Poison();
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_SYNTHETIC_CALLBACK_LAYER_GATE_CLOSED");
}

void M0SyntheticOwnerV1::PrepareSorted(const Graph* requested) {
  const size_t first = requested == entries_[3].graph ? 3 : 0;
  const size_t end = first == 3 ? 4 : 3;
  auto* root = const_cast<Graph*>(entries_[first].graph);
  for (size_t slot = first; slot < end; ++slot) {
    ORT_ENFORCE(preparation_[slot] == Preparation::kOriginal && !Poisoned(),
                "RW07_OWNER_PREPARATION_ENTRY");
    auto* graph = const_cast<Graph*>(entries_[slot].graph);
    const auto status = graph->InitInputsInitializersOutputs();
    ORT_THROW_IF_ERROR(status);
    preparation_[slot] = Preparation::kInitialized;
    ValidateAll(requested);
    if (preparation_audit_) ++preparation_audit_->initialized_scopes;
  }
  std::unordered_set<std::string> outer_scope_consumed;
  bool removed_node = false;
  ORT_ENFORCE(!Poisoned(), "RW07_OWNER_PREPARATION_POISON");
  ORT_THROW_IF_ERROR(root->BuildConnections(outer_scope_consumed, removed_node));
  ORT_ENFORCE(!removed_node && outer_scope_consumed.empty(), "RW07_OWNER_PREPARATION_CONNECTIONS");
  for (size_t slot = first; slot < end; ++slot) {
    preparation_[slot] = Preparation::kConnected;
    if (preparation_audit_) ++preparation_audit_->connected_scopes;
  }
  ValidateAll(requested);
  for (size_t slot = first; slot < end; ++slot) {
    ORT_ENFORCE(!Poisoned(), "RW07_OWNER_PREPARATION_POISON");
    auto* graph = const_cast<Graph*>(entries_[slot].graph);
    ORT_THROW_IF_ERROR(graph->PerformTopologicalSortAndCheckIsAcyclic());
    preparation_[slot] = Preparation::kSorted;
    ValidateAll(requested);
    if (preparation_audit_) ++preparation_audit_->sorted_scopes;
  }
  if (preparation_audit_) ++preparation_audit_->prepared_roots;
}

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
M0SyntheticOwnerV1::PermitGeneration::PermitGeneration(M0SyntheticOwnerV1& owner) : generation_(owner) {
  ORT_ENFORCE(generation_.acquired && !owner.Poisoned() && !owner.issued_generation_, "RW07_ISSUED_GENERATION_ENTRY");
  owner.ValidateAll(owner.entries_[0].graph);
  owner.issued_generation_ = this;
}
M0SyntheticOwnerV1::PermitGeneration::~PermitGeneration() noexcept {
  auto& owner = generation_.owner;
  if (owner.issued_generation_ != this || owner.stack_size_ != 0 || (!root_completed_ && !owner.Poisoned())) owner.Poison();
  owner.issued_generation_ = nullptr;
}
M0SyntheticOwnerV1::ScopeLayer::ScopeLayer(PermitGeneration& permit, const Graph& graph)
    : permit_(permit), frame_(permit.generation_, &graph) {
  auto& owner = permit_.generation_.owner;
  while (slot_ < owner.entries_.size() && owner.entries_[slot_].graph != &graph) ++slot_;
  Check(graph);
}
M0SyntheticOwnerV1::ScopeLayer::~ScopeLayer() noexcept {
  if (!completed_) permit_.generation_.owner.Poison();
}
void M0SyntheticOwnerV1::ScopeLayer::Check(const Graph& graph) {
  auto& owner = permit_.generation_.owner;
  ORT_ENFORCE(owner.issued_generation_ == &permit_ && permit_.generation_.acquired &&
                  permit_.generation_.id == owner.generation_id_ && frame_.pushed &&
                  frame_.graph == &graph && slot_ < 3 && owner.stack_size_ == slot_ + 1 &&
                  owner.stack_[slot_] == &graph && &graph == owner.entries_[slot_].graph && !owner.Poisoned(),
              "RW07_ISSUED_SCOPE_JOIN");
  owner.ValidateAll(&graph);
}
M0SyntheticOwnerV1::NodeLayer::NodeLayer(ScopeLayer& scope, const Node& node) : scope_(scope), node_(node) {
  Check();
  ORT_ENFORCE(scope_.perform_entered_ &&
                  scope_.permit_.generation_.owner.issued_node_stage_[scope_.slot_] == IssuedNodeStage::kOriginal,
              "RW07_ISSUED_NODE_ENTRY");
}
M0SyntheticOwnerV1::NodeLayer::~NodeLayer() noexcept {
  if (!completed_) scope_.permit_.generation_.owner.Poison();
}
void M0SyntheticOwnerV1::NodeLayer::Check() {
  scope_.Check(*node_.GetContainingGraph());
  auto& owner = scope_.permit_.generation_.owner;
  ORT_ENFORCE(owner.entries_[scope_.slot_].graph->GetNode(0) == &node_ && owner.issued_audit_, "RW07_ISSUED_NODE_JOIN");
}
void M0SyntheticOwnerV1::NodeLayer::Checked() {
  auto& owner = scope_.permit_.generation_.owner;
  ORT_ENFORCE(!owner.Poisoned() && owner.issued_node_stage_[scope_.slot_] == IssuedNodeStage::kOriginal &&
                  !node_.GetOriginalNodeProto() && !node_.Op() && node_.SinceVersion() == -1,
              "RW07_ISSUED_CHECKER_TRANSITION");
  owner.issued_node_stage_[scope_.slot_] = IssuedNodeStage::kChecked;
  Check();
  ++owner.issued_audit_->source_checks;
}
void M0SyntheticOwnerV1::NodeLayer::Selected() {
  auto& owner = scope_.permit_.generation_.owner;
  ORT_ENFORCE(!owner.Poisoned() && owner.issued_node_stage_[scope_.slot_] == IssuedNodeStage::kChecked &&
                  node_.Op() == owner.sealed_schemas_[scope_.slot_ < 2 ? 1 : 0] && node_.SinceVersion() == 1 &&
                  node_.InputArgCount() == std::vector<int>{1},
              "RW07_ISSUED_SELECTION_TRANSITION");
  owner.issued_node_stage_[scope_.slot_] = IssuedNodeStage::kSelected;
  Check();
  ++owner.issued_audit_->source_selections;
}
void M0SyntheticOwnerV1::NodeLayer::BeforeDefault(const ONNX_NAMESPACE::AttributeProto& attribute) {
  Check();
  auto& owner = scope_.permit_.generation_.owner;
  ORT_ENFORCE(owner.issued_node_stage_[scope_.slot_] == IssuedNodeStage::kSelected &&
                  attribute.SerializeAsString() == m0_internal::CheckedFixedBytes("marker7") &&
                  !node_.GetAttributes().contains("marker") && !owner.Poisoned(),
              "RW07_ISSUED_DEFAULT_PROVENANCE");
}
void M0SyntheticOwnerV1::NodeLayer::Defaulted() {
  auto& owner = scope_.permit_.generation_.owner;
  ORT_ENFORCE(!owner.Poisoned() && owner.issued_node_stage_[scope_.slot_] == IssuedNodeStage::kSelected &&
                  node_.GetAttributes().contains("marker") &&
                  node_.GetAttributes().at("marker").SerializeAsString() == m0_internal::CheckedFixedBytes("marker7"),
              "RW07_ISSUED_DEFAULT_TRANSITION");
  owner.issued_node_stage_[scope_.slot_] = IssuedNodeStage::kDefaulted;
  Check();
  ++owner.issued_audit_->source_defaults;
}
void M0SyntheticOwnerV1::NodeLayer::Published() {
  auto& owner = scope_.permit_.generation_.owner;
  ORT_ENFORCE(!owner.Poisoned() && owner.issued_node_stage_[scope_.slot_] == IssuedNodeStage::kInferred &&
                  node_.OutputDefs()[0]->TypeAsProto()->SerializeAsString() ==
                      node_.InputDefs()[0]->TypeAsProto()->SerializeAsString(),
              "RW07_ISSUED_PUBLICATION_TRANSITION");
  owner.issued_node_stage_[scope_.slot_] = IssuedNodeStage::kPublished;
  Check();
  ++owner.issued_audit_->live_publications;
}
void M0SyntheticOwnerV1::NodeLayer::Completed() {
  Check();
  auto& owner = scope_.permit_.generation_.owner;
  ORT_ENFORCE(owner.issued_node_stage_[scope_.slot_] == IssuedNodeStage::kPublished && !completed_,
              "RW07_ISSUED_NODE_EXIT");
  owner.issued_node_stage_[scope_.slot_] = IssuedNodeStage::kDone;
  completed_ = true;
}

#endif

class M0SyntheticOwnerFactoryV1 final {
 public:
  static M0SyntheticOwnerReceiptV1 Run();
  static M0SyntheticProjectionReceiptV1 Projection();
  static M0SyntheticPreparationReceiptV1 Preparation();
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
  static M0SyntheticIssuedReceiptV1 Issued();
#endif

 private:
  static std::unique_ptr<Model> Make(const std::shared_ptr<M0SyntheticOwnerReceiptV1>& audit,
                                     size_t failure_slot = 4);
  static std::unique_ptr<FunctionImpl> AddParentless(Model& model);
};
std::unique_ptr<Model> M0SyntheticOwnerFactoryV1::Make(
    const std::shared_ptr<M0SyntheticOwnerReceiptV1>& audit, size_t failure_slot) {
  ONNX_NAMESPACE::ModelProto proto;
  ORT_ENFORCE(proto.ParseFromString(m0_internal::CheckedFixedBytes("nested-depth2.model-original")),
              "RW07_FIXED_MODEL_PARSE");
  auto owner = std::unique_ptr<M0SyntheticOwnerV1>(new M0SyntheticOwnerV1(audit));
  owner->fail_after_slot_ = failure_slot;
  auto local = m0_internal::FixedDescriptorRegistryV1();
  owner->sealed_registry_ = local;
  owner->sealed_schemas_[0] = local->GetSchema("DIdentity", 1, "rw07.synthetic.descriptor.contract1");
  owner->sealed_schemas_[1] = local->GetSchema("DNested", 1, "rw07.synthetic.descriptor.contract1");
  const IOnnxRuntimeOpSchemaRegistryList registries{local};
  ModelOptions options;
  options.allow_released_opsets_only = false;
  return std::unique_ptr<Model>(new Model(std::move(proto), PathString(), &registries,
                                          logging::LoggingManager::DefaultLogger(), options, std::move(owner)));
}
std::unique_ptr<FunctionImpl> M0SyntheticOwnerFactoryV1::AddParentless(Model& model) {
  ONNX_NAMESPACE::FunctionProto proto;
  ORT_ENFORCE(proto.ParseFromString(m0_internal::CheckedFixedBytes("parentless.function-original")),
              "RW07_FIXED_FUNCTION_PARSE");
  auto& owner = *model.m0_synthetic_owner_v1_;
  owner.model_identity_ = &model;
  ORT_ENFORCE(proto.SerializeAsString() == owner.original_function_, "RW07_OWNER_RAW_FUNCTION");
  owner.parentless_transition_ = true;
  auto function = std::make_unique<FunctionImpl>(model.MainGraph(), proto);
  owner.parentless_identity_ = function.get();
  auto& body = function->MutableBody();
  body.SetName("parentless");
  ONNX_NAMESPACE::GraphProto storage_recipe;
  storage_recipe.set_name("parentless");
  owner.entries_[3].original_backing = storage_recipe.SerializeAsString();
  ONNX_NAMESPACE::GraphProto recipe;
  ORT_ENFORCE(recipe.ParseFromString(m0_internal::CheckedFixedBytes("parentless.expected-assembled")),
              "RW07_FIXED_PARENTLESS_PARSE");
  auto& input = body.GetOrCreateNodeArg("X", &recipe.input(0).type());
  auto& output = body.GetOrCreateNodeArg("Y", &recipe.output(0).type());
  body.SetInputs({&input});
  body.SetOutputs({&output});
  body.AddNode("leaf", "DIdentity", "", {&input}, {&output}, nullptr, "rw07.synthetic.descriptor.contract1");
  owner.entries_[3].original_canonical = M0GraphCanonical(recipe);
  M0RequireNormalizedImports(body);
  owner.audit_->parentless_inherits_normalized_imports = true;
  ORT_ENFORCE(owner.state_ == M0SyntheticOwnerV1::State::kConstructing && !owner.Poisoned(), "RW07_OWNER_SEAL_STATE");
  for (const auto& entry : owner.entries_) {
    ORT_ENFORCE(entry.graph && entry.constructed && !entry.retired && entry.owner_nonce == owner.nonce_,
                "RW07_OWNER_SEAL_INVENTORY");
  }
  owner.state_ = M0SyntheticOwnerV1::State::kReady;
  return function;
}

M0SyntheticOwnerReceiptV1 M0SyntheticOwnerFactoryV1::Run() {
  auto audit = std::make_shared<M0SyntheticOwnerReceiptV1>();
  {
    auto model = Make(audit);
    auto function = AddParentless(*model);
    auto& owner = *model->m0_synthetic_owner_v1_;
    {
      M0SyntheticOwnerV1::Generation generation(owner);
      owner.ValidateAll(&function->Body());
      M0SyntheticOwnerV1::Frame root(generation, &model->MainGraph());
      M0SyntheticOwnerV1::Frame child(generation, owner.entries_[1].graph);
    }
    const auto status = function->MutableBody().Resolve();
    ORT_ENFORCE(!status.IsOK() && status.ErrorMessage().find("RW07_SYNTHETIC_DESCRIPTOR_GATE_CLOSED") !=
                                      std::string::npos,
                "RW07_CLOSED_RESOLVE_MUST_DENY");
    ORT_ENFORCE(!model->M0CheckCallbackPolicyV1(M0CallbackPolicyV1::Channel::kSchemaInference).IsOK(),
                "RW07_CLOSED_CALLBACK_MUST_DENY");
  }
  ORT_ENFORCE(audit->registered == 4 && audit->unregistered == 4 && audit->live_at_capture == 0 && audit->owner_retired,
              "RW07_OWNER_LIFETIME_RECEIPT");
  {
    auto failure = std::make_shared<M0SyntheticOwnerReceiptV1>();
    try {
      auto model = Make(failure, 1);
      ORT_THROW("RW07_MISSING_CONSTRUCTOR_FAILURE");
    } catch (const std::exception& error) {
      ORT_ENFORCE(std::string(error.what()).find("RW07_FIXED_CONSTRUCTOR_FAILURE") != std::string::npos,
                  "RW07_UNEXPECTED_CONSTRUCTOR_FAILURE");
    }
    audit->constructor_failure_unwound = failure->registered == 3 && failure->unregistered == 3 &&
                                         failure->live_at_capture == 0 && failure->poisoned;
  }
  {
    auto extra = std::make_shared<M0SyntheticOwnerReceiptV1>();
    auto model = Make(extra);
    auto function = AddParentless(*model);
    model->MainGraph().GraphResolveNeeded(false);
    auto* child = const_cast<Graph*>(model->m0_synthetic_owner_v1_->entries_[2].graph);
    child->GraphResolveNeeded(false);
    child->GetOrCreateNodeArg("X", nullptr).GetMutableInferredShapeValues().emplace();
    auto status = child->Resolve();
    audit->inherited_empty_list_denied = !status.IsOK() &&
                                         status.ErrorMessage().find("RW07_OWNER_NODEARG_CURRENT") != std::string::npos;
    audit->clean_child_denied = audit->inherited_empty_list_denied && extra->full_validations == 1;
  }
  {
    auto extra = std::make_shared<M0SyntheticOwnerReceiptV1>();
    auto model = Make(extra);
    auto function = AddParentless(*model);
    auto& owner = *model->m0_synthetic_owner_v1_;
    M0SyntheticOwnerV1::Generation generation(owner);
    Model foreign("foreign", false, logging::LoggingManager::DefaultLogger());
    try {
      owner.ValidateAll(&foreign.MainGraph());
    } catch (const std::exception&) {
      audit->foreign_scope_denied = owner.Poisoned();
    }
  }
  {
    auto extra = std::make_shared<M0SyntheticOwnerReceiptV1>();
    auto model = Make(extra);
    auto function = AddParentless(*model);
    auto& owner = *model->m0_synthetic_owner_v1_;
    M0SyntheticOwnerV1::Generation generation(owner);
    owner.ValidateAll(&model->MainGraph());
    M0SyntheticOwnerV1::Frame frame(generation, &model->MainGraph());
    try {
      M0SyntheticOwnerV1::Frame duplicate(generation, &model->MainGraph());
    } catch (const std::exception&) {
      audit->same_frame_reentry_denied = owner.Poisoned();
    }
  }
  {
    auto extra = std::make_shared<M0SyntheticOwnerReceiptV1>();
    auto model = Make(extra);
    auto function = AddParentless(*model);
    auto& owner = *model->m0_synthetic_owner_v1_;
    M0SyntheticOwnerV1::Generation generation(owner);
    owner.ValidateAll(&model->MainGraph());
    M0SyntheticOwnerV1::Frame frame(generation, &model->MainGraph());
    const auto before = extra->full_validations;
    bool denied = false;
    std::thread worker([&]() {
      M0SyntheticOwnerV1::Generation conflict(owner);
      denied = !conflict.acquired;
    });
    worker.join();
    audit->conflict_denied_before_graph_read = denied && owner.Poisoned() && extra->full_validations == before;
  }
  return *audit;
}
M0SyntheticProjectionReceiptV1 M0SyntheticOwnerFactoryV1::Projection() {
  M0SyntheticProjectionReceiptV1 receipt;
  {
    auto audit = std::make_shared<M0SyntheticOwnerReceiptV1>();
    {
      auto model = Make(audit);
      auto function = AddParentless(*model);
      auto& owner = *model->m0_synthetic_owner_v1_;
      M0SyntheticOwnerV1::Generation generation(owner);
      owner.ValidateProjection(&model->MainGraph(), 0);
      ++receipt.checked_phases;
      receipt.checked_scopes += 4;
      const auto root_body = model->MainGraph().GetNode(0)->GetAttributes().at("body").SerializeAsString();
      const auto inner_body = owner.entries_[1].graph->GetNode(0)->GetAttributes().at("body").SerializeAsString();
      ONNX_NAMESPACE::AttributeProto marker;
      ORT_ENFORCE(marker.ParseFromString(m0_internal::CheckedFixedBytes("marker7")),
                  "RW07_PROJECTION_MARKER_PARSE");
      ONNX_NAMESPACE::TypeProto inferred;
      ORT_ENFORCE(inferred.ParseFromString(m0_internal::CheckedFixedBytes("type.float-shape-2-3")),
                  "RW07_PROJECTION_TYPE_PARSE");
      for (size_t phase = 1; phase <= 7; ++phase) {
        if (phase <= 3) {
          auto& graph = *const_cast<Graph*>(owner.entries_[phase - 1].graph);
          graph.GetNode(0)->AddAttributeProto(marker);
        } else if (phase <= 6) {
          auto& graph = *const_cast<Graph*>(owner.entries_[6 - phase].graph);
          ORT_THROW_IF_ERROR(graph.GetNodeArg(graph.GetOutputs()[0]->Name())->UpdateTypeAndShape(inferred, true, false, graph.GetLogger()));
        }
        owner.ValidateProjection(&model->MainGraph(), phase);
        ++receipt.checked_phases;
        receipt.checked_scopes += 4;
      }
      receipt.map_body_preserved = root_body == model->MainGraph().GetNode(0)->GetAttributes().at("body").SerializeAsString() &&
                                   inner_body == owner.entries_[1].graph->GetNode(0)->GetAttributes().at("body").SerializeAsString();
      try {
        owner.ValidateAll(&model->MainGraph());
      } catch (const std::exception&) {
        receipt.phase_mismatch_poisoned = owner.Poisoned();
      }
    }
    ORT_ENFORCE(audit->registered == 4 && audit->unregistered == 4 && audit->live_at_capture == 0 && audit->owner_retired,
                "RW07_PROJECTION_LIFETIME");
    ++receipt.retired_owners;
  }
  // Every mismatch owns a fresh candidate; no poisoned candidate is reset/rebound.
  for (size_t variant = 0; variant < 20; ++variant) {
    auto audit = std::make_shared<M0SyntheticOwnerReceiptV1>();
    {
      auto model = Make(audit);
      auto function = AddParentless(*model);
      auto& owner = *model->m0_synthetic_owner_v1_;
      auto& root = model->MainGraph();
      auto& child = *const_cast<Graph*>(owner.entries_[2].graph);
      NodeArg foreign("X", child.GetNodeArg("X")->TypeAsProto());
      root.GraphResolveNeeded(false);
      child.GraphResolveNeeded(false);
      switch (variant) {
        case 0:
          child.graph_inputs_including_initializers_[0] = &foreign;
          break;
        case 1:
          child.graph_outputs_[0] = child.GetNodeArg("X");
          break;
        case 2:
          child.GetNode(0)->MutableInputDefs()[0] = &foreign;
          break;
        case 3:
          child.GetNode(0)->MutableOutputDefs()[0] = child.GetNodeArg("X");
          break;
        case 4:
          child.parent_graph_ = &root;
          break;
        case 5:
          child.parent_node_ = root.GetNode(0);
          break;
        case 6:
          child.nodes_in_topological_order_.push_back(0);
          break;
        case 7:
          child.num_of_nodes_ = 0;
          break;
        case 8:
          child.outer_scope_node_arg_names_.insert("X");
          break;
        case 9:
          child.value_info_.insert(child.GetNodeArg("X"));
          break;
        case 10:
          child.GetNode(0)->SetOverload("unapproved");
          break;
        case 11:
          child.GetNode(0)->SetSinceVersion(1);
          break;
        case 12:
          child.schema_registry_ = std::make_shared<SchemaRegistryManager>();
          break;
        case 13:
          child.GetOrCreateNodeArg("unused", nullptr).SetInferredShapeScalarValue(7);
          break;
        case 14:
          child.GetOrCreateNodeArg("unused", nullptr).GetMutableInferredShapeValues().emplace();
          break;
        case 15:
          child.SetOpSchemaFromRegistryForNode(*child.GetNode(0));
          break;
        case 16:
          child.graph_inputs_including_initializers_[0] = nullptr;
          child.node_args_["X"].reset();
          break;
        case 17:
          const_cast<std::vector<int>&>(child.GetNode(0)->InputArgCount()) = {0, 1};
          break;
        case 18:
          child.ir_version_ = 9;
          break;
        case 19:
          const_cast<ONNX_NAMESPACE::OpSchema*>(owner.sealed_schemas_[0])->TypeConstraint("T", {"tensor(int64)"}, "unapproved");
          break;
      }
      const auto status = child.Resolve();
      ORT_ENFORCE(!status.IsOK() && owner.Poisoned() && audit->full_validations == 1 &&
                      audit->resolve_denials == 0 && audit->callback_denials == 0,
                  "RW07_PROJECTION_MISMATCH_NOT_DENIED");
      ++receipt.rejected_mutations;
      child.parent_graph_ = const_cast<Graph*>(owner.entries_[1].graph);
      child.parent_node_ = owner.entries_[1].graph->GetNode(0);
    }
    ORT_ENFORCE(audit->registered == 4 && audit->unregistered == 4 && audit->live_at_capture == 0 && audit->owner_retired,
                "RW07_PROJECTION_NEGATIVE_LIFETIME");
    ++receipt.retired_owners;
  }
  {
    auto audit = std::make_shared<M0SyntheticOwnerReceiptV1>();
    {
      auto model = Make(audit);
      auto function = AddParentless(*model);
      auto& owner = *model->m0_synthetic_owner_v1_;
      ONNX_NAMESPACE::GraphProto proto;
      ORT_ENFORCE(proto.ParseFromString(m0_internal::CheckedFixedBytes("nested-depth2.root.original-graph")),
                  "RW07_FAILED_REGISTRATION_RECIPE");
      const auto before = audit->registered;
      try {
        Graph duplicate(*model, &proto, model->MainGraph().DomainToVersionMap(), model->IrVersion(),
                        model->MainGraph().GetSchemaRegistry(), model->MainGraph().GetLogger(), false);
      } catch (const std::exception&) {
        receipt.failed_registration_preserved = owner.Poisoned() && audit->registered == before &&
                                                audit->live_at_capture == 4;
      }
    }
    ORT_ENFORCE(audit->registered == 4 && audit->unregistered == 4 && audit->live_at_capture == 0 && audit->owner_retired,
                "RW07_FAILED_REGISTRATION_LIFETIME");
    ++receipt.retired_owners;
  }
  ORT_ENFORCE(receipt.checked_phases == 8 && receipt.checked_scopes == 32 && receipt.rejected_mutations == 20 &&
                  receipt.retired_owners == 22 && receipt.map_body_preserved && receipt.phase_mismatch_poisoned &&
                  receipt.failed_registration_preserved,
              "RW07_PROJECTION_RECEIPT_INCOMPLETE");
  return receipt;
}
M0SyntheticPreparationReceiptV1 M0SyntheticOwnerFactoryV1::Preparation() {
  M0SyntheticPreparationReceiptV1 receipt;
  for (size_t requested_slot = 0; requested_slot < 4; ++requested_slot) {
    auto audit = std::make_shared<M0SyntheticOwnerReceiptV1>();
    {
      auto model = Make(audit);
      auto function = AddParentless(*model);
      auto& owner = *model->m0_synthetic_owner_v1_;
      owner.preparation_audit_ = &receipt;
      auto* requested = const_cast<Graph*>(owner.entries_[requested_slot].graph);
      // Clean flags cannot bypass whole-owner validation or the internal route.
      requested->GraphResolveNeeded(false);
      const auto status = requested->Resolve();
      ORT_ENFORCE(!status.IsOK() &&
                      status.ErrorMessage().find("RW07_SYNTHETIC_CALLBACK_LAYER_GATE_CLOSED") != std::string::npos &&
                      owner.Poisoned() && audit->frame_pushes == 1 && audit->frame_pops == 1 &&
                      audit->generation_acquisitions == 1 && audit->callback_denials == 0,
                  "RW07_PREPARATION_CLOSED_EXIT");
    }
    ORT_ENFORCE(audit->owner_retired && audit->live_at_capture == 0 && audit->unregistered == 4,
                "RW07_PREPARATION_RETIREMENT");
    ++receipt.retired_owners;
  }
  for (size_t variant = 0; variant < 20; ++variant) {
    auto audit = std::make_shared<M0SyntheticOwnerReceiptV1>();
    {
      auto model = Make(audit);
      auto function = AddParentless(*model);
      auto& owner = *model->m0_synthetic_owner_v1_;
      auto& child = *const_cast<Graph*>(owner.entries_[2].graph);
      owner.preparation_audit_ = &receipt;
      ONNX_NAMESPACE::GraphProto foreign_backing = *child.graph_proto_;
      switch (variant) {
        case 0:
          model->model_proto_.set_producer_name("foreign");
          break;
        case 1:
          model->model_proto_.set_doc_string("foreign");
          break;
        case 2:
          model->model_proto_.add_opset_import()->CopyFrom(model->model_proto_.opset_import(0));
          break;
        case 3:
          model->model_path_ = "foreign";
          break;
        case 4:
          model->check_load_cancellation_fn_ = [] { return false; };
          break;
        case 5:
          child.graph_proto_ = &foreign_backing;
          break;
        case 6:
          child.graph_proto_->set_doc_string("foreign");
          break;
        case 7:
          function->function_storage_proto_.set_doc_string("foreign");
          break;
        case 8:
          child.deserialized_proto_data_.set_name("foreign");
          break;
        case 9:
          child.prepacked_key_to_blobs_.emplace();
          break;
        case 10:
          child.weight_data_type_freq_[ONNX_NAMESPACE::TensorProto::FLOAT] = 1;
          break;
        case 11:
          child.generated_node_names_.insert("foreign");
          break;
        case 12:
          child.node_arg_to_producer_node_["Y"] = 0;
          break;
        case 13:
          child.resolve_context_.inputs_and_initializers.insert("X");
          break;
        case 14:
          child.resolve_context_.inferred_subgraphs.insert(&child);
          break;
        case 15:
          child.name_generator_ = 1;
          break;
        case 16:
          child.GetNode(0)->SetOriginalNodeProto(&foreign_backing.node(0));
          break;
        case 17:
          child.runtime_optimizations_.AddRecord("foreign", RuntimeOptimizationRecord{});
          break;
        case 18:
          model->MainGraph().GetNode(0)->ClearAttribute("body");
          break;
        case 19:
          const_cast<NodeAttributes&>(model->MainGraph().GetNode(0)->GetAttributes()).at("body").clear_g();
          break;
      }
      const auto before = receipt.prepared_roots;
      const auto status = child.Resolve();
      ORT_ENFORCE(!status.IsOK() && owner.Poisoned() && audit->full_validations == 1 &&
                      audit->frame_pushes == 0 && receipt.prepared_roots == before &&
                      std::all_of(owner.preparation_.begin(), owner.preparation_.end(),
                                  [](auto stage) { return stage == M0SyntheticOwnerV1::Preparation::kOriginal; }),
                  "RW07_BACKING_REFUSAL_BEFORE_PREPARATION");
      ++receipt.rejected_backing_mutations;
      // Borrowed negative pointer remains valid through refusal; restore only
      // lifetime plumbing for disposal, never retry or clear poison.
      if (variant == 5) child.graph_proto_ = const_cast<ONNX_NAMESPACE::GraphProto*>(owner.entries_[2].backing);
    }
    ORT_ENFORCE(audit->owner_retired && audit->live_at_capture == 0 && audit->unregistered == 4,
                "RW07_BACKING_REFUSAL_RETIREMENT");
    ++receipt.retired_owners;
  }
  return receipt;
}
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
M0SyntheticIssuedReceiptV1 M0SyntheticOwnerFactoryV1::Issued() {
  M0SyntheticIssuedReceiptV1 receipt;
  for (const auto effect : {M0SyntheticOwnerV1::IssuedEffect::kNone, M0SyntheticOwnerV1::IssuedEffect::kNestedFailure,
                            M0SyntheticOwnerV1::IssuedEffect::kConflict, M0SyntheticOwnerV1::IssuedEffect::kPayload,
                            M0SyntheticOwnerV1::IssuedEffect::kSparse, M0SyntheticOwnerV1::IssuedEffect::kSymbolic,
                            M0SyntheticOwnerV1::IssuedEffect::kAttribute, M0SyntheticOwnerV1::IssuedEffect::kFunction,
                            M0SyntheticOwnerV1::IssuedEffect::kPropagationRead, M0SyntheticOwnerV1::IssuedEffect::kPropagationWrite,
                            M0SyntheticOwnerV1::IssuedEffect::kMalformedPropagation, M0SyntheticOwnerV1::IssuedEffect::kEngagedInventory,
                            M0SyntheticOwnerV1::IssuedEffect::kRepeatedBody, M0SyntheticOwnerV1::IssuedEffect::kDirectPerform}) {
    auto audit = std::make_shared<M0SyntheticOwnerReceiptV1>();
    {
      auto model = Make(audit);
      auto function = AddParentless(*model);
      auto& owner = *model->m0_synthetic_owner_v1_;
      M0SyntheticIssuedReceiptV1 local;
      owner.issued_audit_ = &local;
      owner.issued_effect_ = effect;
      {
        M0SyntheticOwnerV1::PermitGeneration permit(owner);
        M0SyntheticOwnerV1::ScopeLayer root(permit, model->MainGraph());
        owner.PrepareSorted(&model->MainGraph());
        const auto status = model->MainGraph().PerformTypeAndShapeInferencing({}, root);
        if (effect == M0SyntheticOwnerV1::IssuedEffect::kNone) {
          ORT_ENFORCE(status.IsOK() && !owner.Poisoned() && local.source_checks == 3 &&
                          local.source_selections == 3 && local.source_defaults == 3 && local.callbacks == 3 &&
                          local.child_joins == 2 && local.propagation_no_work == 3 && local.saver_no_work == 3 &&
                          local.live_publications == 3 && local.child_skips == 2 && audit->frame_pushes == 3 &&
                          audit->callback_denials == 0 && owner.stack_size_ == 1,
                      "RW07_ISSUED_POST_PERFORM");
          owner.ValidateAll(&model->MainGraph());
          receipt = local;
          receipt.fixed_post_perform_completed = true;
        } else {
          ORT_ENFORCE(!status.IsOK() && owner.Poisoned(), "RW07_ISSUED_EFFECT_MUST_FAIL");
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kNestedFailure) {
            ORT_ENFORCE(local.live_publications == 1 && local.source_defaults == 3 &&
                            owner.entries_[2].graph->GetNode(0)->OutputDefs()[0]->Shape() &&
                            !owner.entries_[1].graph->GetNode(0)->OutputDefs()[0]->Shape(),
                        "RW07_ISSUED_MEANINGFUL_NESTED_FAILURE");
            ORT_ENFORCE(owner.entries_[2].graph->GetNode(0)->OutputDefs()[0]->TypeAsProto()->SerializeAsString() ==
                                owner.entries_[2].graph->GetNode(0)->InputDefs()[0]->TypeAsProto()->SerializeAsString() &&
                            !model->MainGraph().GetNode(0)->OutputDefs()[0]->Shape() && local.child_joins == 1 &&
                            local.callbacks == 3 && local.propagation_no_work == 1 && local.saver_no_work == 1,
                        "RW07_ISSUED_LATE_FAILURE_RECEIPT");
            receipt.meaningful_nested_failure = true;
          } else if (effect != M0SyntheticOwnerV1::IssuedEffect::kMalformedPropagation &&
                     effect != M0SyntheticOwnerV1::IssuedEffect::kEngagedInventory) {
            ORT_ENFORCE(local.callbacks == 1 && local.child_joins == 0 && local.propagation_no_work == 0 &&
                            local.saver_no_work == 0 && local.live_publications == 0 && audit->frame_pushes == 1 &&
                            owner.stack_size_ == 1,
                        "RW07_ISSUED_EARLY_DENIAL_RECEIPT");
          }
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kConflict) receipt.callback_conflict_denied = true;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kPayload) receipt.payload_denied = audit->callback_denials == 1;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kSparse) receipt.sparse_denied = audit->callback_denials == 1;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kSymbolic) receipt.symbolic_denied = audit->callback_denials == 1;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kAttribute) receipt.attribute_denied = audit->callback_denials == 1;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kFunction) receipt.function_denied = audit->callback_denials == 1;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kPropagationRead) receipt.propagation_read_denied = audit->callback_denials == 1;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kPropagationWrite) receipt.propagation_write_denied = audit->callback_denials == 1;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kMalformedPropagation) {
            ORT_ENFORCE(local.callbacks == 3 && local.propagation_no_work == 0 && local.live_publications == 0 &&
                            status.ErrorMessage().find("RW07_ISSUED_DP_ACTUAL_EMPTY_TYPE") != std::string::npos,
                        "RW07_ISSUED_MALFORMED_DP_RECEIPT");
            receipt.malformed_propagation_denied = true;
          }
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kEngagedInventory) {
            ORT_ENFORCE(local.callbacks == 3 && local.propagation_no_work == 1 && local.saver_no_work == 0 &&
                            local.live_publications == 0 && owner.entries_[2].graph->GetNode(0)->OutputDefs()[0]->GetInferredShapeValues().has_value(),
                        "RW07_ISSUED_SAVER_INVENTORY_RECEIPT");
            receipt.engaged_inventory_denied = true;
          }
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kRepeatedBody) receipt.repeated_body_denied = true;
          if (effect == M0SyntheticOwnerV1::IssuedEffect::kDirectPerform) receipt.direct_perform_denied = audit->callback_denials == 1;
        }
      }
      owner.issued_audit_ = nullptr;
    }
    ORT_ENFORCE(audit->owner_retired && audit->unregistered == 4 && audit->live_at_capture == 0 &&
                    audit->frame_pushes == audit->frame_pops,
                "RW07_ISSUED_RETIREMENT");
    ++receipt.retired_owners;
  }
  return receipt;
}
M0SyntheticIssuedReceiptV1 M0RunFixedSyntheticIssuedV1() { return M0SyntheticOwnerFactoryV1::Issued(); }
#endif

M0SyntheticPreparationReceiptV1 M0RunFixedSyntheticPreparationV1() {
  return M0SyntheticOwnerFactoryV1::Preparation();
}
M0SyntheticProjectionReceiptV1 M0RunFixedSyntheticProjectionV1() {
  return M0SyntheticOwnerFactoryV1::Projection();
}
M0SyntheticOwnerReceiptV1 M0RunFixedSyntheticOwnerFoundationV1() {
  return M0SyntheticOwnerFactoryV1::Run();
}
#else
// Minimal/editor paths cannot acquire a synthetic owner.
uint64_t M0SyntheticOwnerV1::Register(const Graph*, const ONNX_NAMESPACE::GraphProto*, const Graph*, const Node*) {
  Poison();
  ORT_THROW("RW07_SYNTHETIC_MINIMAL_CLOSED");
}
void M0SyntheticOwnerV1::Finish(uint64_t) {
  Poison();
  ORT_THROW("RW07_SYNTHETIC_MINIMAL_CLOSED");
}
void M0SyntheticOwnerV1::Unregister(uint64_t) noexcept { Poison(); }
bool M0SyntheticOwnerV1::DenyCallback() noexcept {
  Poison();
  return true;
}
common::Status M0SyntheticOwnerV1::ResolveClosed(const Graph*) {
  Poison();
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_SYNTHETIC_CONFIGURATION_CLOSED");
}
#endif

}  // namespace onnxruntime
