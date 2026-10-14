// Source-private RW07 final-plan admission ordering tests. These are not a
// complete-model certificate or proof about earlier InferenceSession callbacks.
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <string>

#include "asserts.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/execution_providers.h"
#include "core/framework/customregistry.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "core/graph/m0_fixed_wire_v1.h"
#include "core/graph/m0_descriptor_schema_v1.h"
#include "core/graph/data_propagation/custom_data_propagation.h"
#include "core/graph/function_impl.h"
#include "core/graph/function_utils.h"
#include "core/graph/schema_registry.h"
#include "onnx/defs/shape_inference.h"
#include "core/graph/model_editor_api_types.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/session/utils.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"

namespace onnxruntime::test {
ONNX_OPERATOR_SCHEMA(M0PreInitSentinel)
    .SetDoc("Source-private before-initialization denial sentinel.")
    .Input(0, "weight", "Initializer input.", "tensor(float)")
    .Input(1, "optional", "Optional input.", "tensor(float)", ONNX_NAMESPACE::OpSchema::Optional)
    .Output(0, "output", "Sentinel output.", "tensor(float)");

namespace {
struct SentinelCounts {
  int constructors = 0;
  int prepacks = 0;
};

class PreInitSentinel final : public OpKernel {
 public:
  PreInitSentinel(const OpKernelInfo& info, SentinelCounts& counts)
      : OpKernel(info), counts_(counts) {
    ++counts_.constructors;
  }
  Status Compute(OpKernelContext*) const override { return Status::OK(); }
  Status PrePack(const Tensor&, int, AllocatorPtr, bool& is_packed, PrePackedWeights*) override {
    ++counts_.prepacks;
    is_packed = false;
    return Status::OK();
  }
 private:
  SentinelCounts& counts_;
};

void CheckFinalization(bool strict, const char* abi = "1", bool disable_prepack = true, bool rich = false, bool absent = false) {
  SentinelCounts counts;
  Model model("rw07_preinit_sentinel", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto type;
  type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& input = graph.GetOrCreateNodeArg("weight", &type);
  auto& output = graph.GetOrCreateNodeArg("output", &type);
  auto& empty = graph.GetOrCreateNodeArg("", nullptr);
  std::vector<NodeArg*> inputs{&input};
  if (absent) inputs.push_back(&empty);
  auto& node = graph.AddNode("sentinel", "M0PreInitSentinel", "", inputs, {&output});
  const auto node_index = node.Index();
  graph.SetOutputs({&output});
  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name("weight");
  initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  initializer.add_dims(1);
  initializer.add_float_data(1.0f);
  graph.AddInitializedTensor(initializer);
  ASSERT_STATUS_OK(graph.Resolve());
  node.SetExecutionProviderType(kCpuExecutionProvider);

  ExecutionProviders providers;
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider,
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false))));
  KernelRegistryManager registries;
  ASSERT_STATUS_OK(registries.RegisterKernels(providers));
  auto sentinel_registry = std::make_shared<KernelRegistry>();
  KernelDefBuilder builder;
  builder.SetName("M0PreInitSentinel").SetDomain(kOnnxDomain)
      .Provider(kCpuExecutionProvider).SinceVersion(1).AllocateInputsContiguously();
  if (rich) {
    builder.SinceVersion(1, 3).Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 0).OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 7); // Preserve an unused declaration too.
  }
  auto definition = builder.Build();
  KernelCreateInfo sentinel_info(std::move(definition),
      [&counts](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& kernel) {
        kernel = std::make_unique<PreInitSentinel>(info, counts);
        return Status::OK();
      });
  if (rich) sentinel_info.m0_implementation_id_v1 = "test-only-unreviewed-selected-v2";
  ASSERT_STATUS_OK(sentinel_registry->Register(std::move(sentinel_info)));
  registries.RegisterKernelRegistry(sentinel_registry);
  const KernelCreateInfo* resolved = nullptr;
  ASSERT_STATUS_OK(registries.SearchKernelRegistry(node, DefaultLoggingManager().DefaultLogger(), &resolved));
  ASSERT_EQ(resolved, &sentinel_registry->GetKernelCreateMap().begin()->second);

  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  if (strict) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.preinit_admission_v1", "1"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.preinit_abi_v1", abi));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking,
                                                          disable_prepack ? "1" : "0"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableAheadOfTimeFunctionInlining, "1"));
  }
  DataTransferManager transfers;
  ExternalDataLoaderManager loaders;
  profiling::Profiler profiler;
  SessionState state(graph, providers, nullptr, nullptr, transfers, loaders,
      DefaultLoggingManager().DefaultLogger(), profiler, options);
  const auto status = state.FinalizeSessionState(ORT_TSTR(""), registries);
  if (strict) {
    ASSERT_FALSE(status.IsOK());
    // The owned plan establishes that this direct SessionState test reached
    // CreatePlan, independent of any InferenceSession preload rejection.
    ASSERT_NE(state.GetExecutionPlan(), nullptr);
    if (std::string(abi) == "1" && disable_prepack) {
      EXPECT_NE(status.ErrorMessage().find("RW07_PREINIT_DENY_UNREVIEWED_CONTRACT"), std::string::npos);
      const auto* snapshot = state.M0FullGpuPreinitSnapshot();
      ASSERT_NE(snapshot, nullptr);
      EXPECT_EQ(snapshot->owned_plan, state.GetExecutionPlan());
      EXPECT_EQ(snapshot->session_epoch, state.M0FullGpuExtentOwnerEpochV1());
      ASSERT_EQ(snapshot->nodes.size(), 1U);
      EXPECT_EQ(snapshot->nodes.front().index, node_index);
      EXPECT_EQ(snapshot->observation_schema, 2U);
      const auto& row = snapshot->nodes.front();
      EXPECT_EQ(row.implementation_id, rich ? "test-only-unreviewed-selected-v2" : "");
      EXPECT_EQ(row.since_version, node.SinceVersion());
      EXPECT_EQ(row.input_arg_counts, node.InputArgCount());
      EXPECT_TRUE(row.implicit_inputs.empty());
      EXPECT_EQ(row.kernel.selected_create_info, resolved);
      EXPECT_EQ(row.kernel.selected_definition, resolved->kernel_def.get());
      const auto& def = *resolved->kernel_def;
      EXPECT_EQ(row.kernel.op_name, def.OpName());
      EXPECT_EQ(row.kernel.domain, def.Domain());
      EXPECT_EQ(row.kernel.provider, def.Provider());
      EXPECT_EQ(row.kernel.version_start, def.SinceVersion().first);
      EXPECT_EQ(row.kernel.version_end, def.SinceVersion().second);
      EXPECT_EQ(row.kernel.may_inplace, def.MayInplace());
      EXPECT_EQ(row.kernel.aliases, def.Alias());
      EXPECT_EQ(row.kernel.has_variadic_alias, def.VariadicAlias().has_value());
      EXPECT_EQ(row.kernel.contiguous_inputs, def.AllocateInputsContiguously());
      EXPECT_EQ(row.kernel.external_outputs, def.HasExternalOutputs());
      EXPECT_EQ(row.kernel.execution_queue, def.ExecQueueId());
      EXPECT_EQ(row.kernel.default_input_memory, def.DefaultInputMemoryType());
      EXPECT_EQ(row.kernel.default_output_memory, def.DefaultOutputMemoryType());
      EXPECT_EQ(row.kernel.input_memory.size(), def.InputMemoryTypeArgs().size());
      for (const auto& entry : row.kernel.input_memory)
        EXPECT_EQ(entry.second, def.InputMemoryTypeArgs().at(entry.first));
      EXPECT_EQ(row.kernel.output_memory.size(), def.OutputMemoryTypeArgs().size());
      for (const auto& entry : row.kernel.output_memory)
        EXPECT_EQ(entry.second, def.OutputMemoryTypeArgs().at(entry.first));
      ASSERT_EQ(row.inputs.size(), absent ? 2U : 1U);
      if (absent) {
        EXPECT_FALSE(row.inputs[1].present);
        EXPECT_TRUE(row.inputs[1].name.empty());
        EXPECT_EQ(row.inputs[1].value_index, -1);
        EXPECT_EQ(row.inputs[1].effective_memory_type, def.InputMemoryType(1));
      }
      if (rich) {
        ASSERT_EQ(row.kernel.aliases.size(), 1U);
        EXPECT_EQ(row.kernel.input_memory.back().first, 7);
      }
      EXPECT_TRUE(row.inputs[0].effective_memory_available);
      EXPECT_TRUE(row.outputs[0].effective_memory_available);
      EXPECT_EQ(row.inputs[0].effective_memory_type, def.InputMemoryType(0));
      EXPECT_EQ(row.outputs[0].effective_memory_type, def.OutputMemoryType(0));
      EXPECT_TRUE(snapshot->nodes.front().inputs.front().present);
      EXPECT_EQ(snapshot->nodes.front().inputs.front().name, "weight");
      EXPECT_EQ(snapshot->initializer_order, state.GetExecutionPlan()->initializer_allocation_order);
      EXPECT_EQ(snapshot->resources.size(), state.GetExecutionPlan()->allocation_plan.size());
      int initializer_index = -1;
      ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetIdx("weight", initializer_index));
      ASSERT_GE(initializer_index, 0);
      ASSERT_LT(static_cast<size_t>(initializer_index), snapshot->resources.size());
      EXPECT_EQ(snapshot->resources[initializer_index].value_index, initializer_index);
      EXPECT_EQ(snapshot->resources[initializer_index].allocation_kind,
                static_cast<int>(state.GetExecutionPlan()->allocation_plan[initializer_index].alloc_kind));
      EXPECT_EQ(row.inputs[0].value_index, initializer_index);
      ASSERT_EQ(snapshot->initializers.size(), 1U);
      const auto& init = snapshot->initializers.front();
      const auto* tensor_proto = graph.GetAllInitializedTensors().at("weight");
      EXPECT_EQ(init.owned_tensor_proto, tensor_proto);
      EXPECT_EQ(init.name, "weight");
      EXPECT_EQ(init.value_index, initializer_index);
      EXPECT_EQ(init.encoded_bytes, tensor_proto->ByteSizeLong());
      EXPECT_EQ(init.raw_data_bytes, tensor_proto->raw_data().size());
      EXPECT_EQ(init.data_type, tensor_proto->data_type());
      EXPECT_EQ(init.data_location, tensor_proto->data_location());
      EXPECT_EQ(init.dimensions, std::vector<int64_t>({1}));
      EXPECT_TRUE(init.constant);
      const auto& plan = *state.GetExecutionPlan();
      EXPECT_EQ(snapshot->activation_order, plan.activation_allocation_order);
      EXPECT_EQ(snapshot->node_release_list, plan.node_release_list);
      ASSERT_EQ(snapshot->release_actions.size(), plan.release_actions.size());
      for (size_t i = 0; i < plan.release_actions.size(); ++i) {
        EXPECT_EQ(snapshot->release_actions[i].first, plan.release_actions[i].value_index);
        EXPECT_EQ(snapshot->release_actions[i].second, plan.release_actions[i].ref_count);
      }
      EXPECT_EQ(snapshot->value_streams.size(), plan.value_to_stream_map.size());
      for (const auto& entry : snapshot->value_streams)
        EXPECT_EQ(entry.second, plan.value_to_stream_map.at(entry.first));
      size_t cursor = 0;
      for (size_t stream = 0; stream < plan.execution_plan.size(); ++stream) {
        if (!plan.execution_plan[stream]) continue;
        for (const auto& step : plan.execution_plan[stream]->steps_) {
          ASSERT_LT(cursor, snapshot->execution_steps.size());
          EXPECT_EQ(snapshot->execution_steps[cursor].stream_index, stream);
          EXPECT_EQ(snapshot->execution_steps[cursor].node_index, step->GetNodeIndex());
          EXPECT_EQ(snapshot->execution_steps[cursor++].description, step->ToString());
        }
      }
      EXPECT_GT(cursor, 0U);
      EXPECT_EQ(cursor, snapshot->execution_steps.size());
      ASSERT_EQ(snapshot->graph_outputs.size(), 1U);
      EXPECT_EQ(snapshot->graph_outputs[0].name, "output");
      for (size_t i = 0; i < snapshot->resources.size(); ++i) {
        const auto& resource = snapshot->resources[i];
        std::string name;
        ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetName(static_cast<int>(i), name));
        EXPECT_EQ(resource.value_name, name);
        EXPECT_EQ(resource.lifetime_starts, plan.allocation_plan[i].program_counter.Starts());
        EXPECT_EQ(resource.lifetime_ends, plan.allocation_plan[i].program_counter.Ends());
        EXPECT_EQ(resource.reused_buffer, plan.allocation_plan[i].reused_buffer);
      }
      EXPECT_EQ(graph.GetAllInitializedTensors().size(), 1U);
    } else {
      EXPECT_NE(status.ErrorMessage().find(std::string(abi) == "1" ? "RW07_PREINIT_DENY_OPTIONS"
                                                                  : "RW07_PREINIT_DENY_ABI"),
                std::string::npos);
      EXPECT_EQ(state.M0FullGpuPreinitSnapshot(), nullptr);
    }
    EXPECT_EQ(counts.constructors, 0);
    EXPECT_EQ(counts.prepacks, 0);
    EXPECT_TRUE(state.GetInitializedTensors().empty());
    EXPECT_EQ(state.GetKernel(node_index), nullptr);
  } else {
    ASSERT_STATUS_OK(status);
    EXPECT_EQ(state.M0FullGpuPreinitSnapshot(), nullptr);
    EXPECT_EQ(counts.constructors, 1);
    EXPECT_EQ(counts.prepacks, 1);
    EXPECT_FALSE(state.GetInitializedTensors().empty());
    EXPECT_NE(state.GetKernel(node_index), nullptr);
  }
}
}  // namespace

TEST(M0FullGpuPreInitAdmissionTest, MissingReviewedContractDeniesBeforeInitializerConstructorAndPrePack) {
  CheckFinalization(true);
}
TEST(M0FullGpuPreInitAdmissionTest, NamedSelectedDefinitionStillDeniesWithExactAliasAndMemoryDeclarations) {
  CheckFinalization(true, "1", true, true);
}
TEST(M0FullGpuPreInitAdmissionTest, AbsentOptionalSlotRetainsPresenceIndexAndMemory) {
  CheckFinalization(true, "1", true, false, true);
}
TEST(M0FullGpuPreInitAdmissionTest, UnknownPrivateAbiDeniesBeforeConstructorAndPrePack) {
  CheckFinalization(true, "99");
}
TEST(M0FullGpuPreInitAdmissionTest, EnabledUnreviewedPrePackDeniesBeforeAllocationAndConstructor) {
  CheckFinalization(true, "1", false);
}
TEST(M0FullGpuPreInitAdmissionTest, OptOutControlActuallyConstructsAndPrePacksSentinel) {
  CheckFinalization(false);
}
// Load byte inputs directly so malformed protobuf bytes act as a parse-boundary
// sentinel: the trusted preload contract denial must take precedence.
TEST(M0FullGpuPreInitAdmissionTest, PreloadRejectsValidAndMalformedBytesBeforeParsing) {
  ONNX_NAMESPACE::ModelProto proto;
  proto.set_ir_version(7);
  proto.add_opset_import()->set_version(13);
  auto* graph = proto.mutable_graph();
  graph->set_name("rw07_preload_identity");
  auto* node = graph->add_node();
  node->set_op_type("Identity");
  node->add_input("X");
  node->add_output("Y");
  const auto add_value = [](ONNX_NAMESPACE::ValueInfoProto* value, const char* name) {
    value->set_name(name);
    auto* tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor->mutable_shape()->add_dim()->set_dim_value(1);
  };
  add_value(graph->add_input(), "X");
  add_value(graph->add_output(), "Y");
  std::string valid;
  ASSERT_TRUE(proto.SerializeToString(&valid));
  const std::string malformed("not a protobuf model");

  SessionOptions strict;
  strict.graph_optimization_level = TransformerLevel::Default;
  strict.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  ASSERT_STATUS_OK(strict.config_options.AddConfigEntry("m0.fullgpu.preinit_admission_v1", "1"));
  ASSERT_STATUS_OK(strict.config_options.AddConfigEntry("m0.fullgpu.preinit_abi_v1", "1"));
  ASSERT_STATUS_OK(strict.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, "1"));
  ASSERT_STATUS_OK(strict.config_options.AddConfigEntry(kOrtSessionOptionsDisableAheadOfTimeFunctionInlining, "1"));
  for (const auto* bytes : std::array<const std::string*, 2>{&valid, &malformed}) {
    InferenceSession session(strict, GetEnvironment());
    const auto status = session.Load(bytes->data(), static_cast<int>(bytes->size()));
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find("RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT"), std::string::npos);
    try {
      InferenceSession model_constructor(strict, GetEnvironment(), bytes->data(), static_cast<int>(bytes->size()));
      FAIL() << "Strict model-taking constructor must deny before parsing";
    } catch (const OnnxRuntimeException& error) {
      EXPECT_NE(std::string(error.what()).find("RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT"), std::string::npos);
    }
  }
  // A distinct opt-out Session proves these serialized bytes describe a loadable
  // graph; the malformed control proves a parse error was actually possible.
  SessionOptions normal;
  InferenceSession valid_control(normal, GetEnvironment());
  ASSERT_STATUS_OK(valid_control.Load(valid.data(), static_cast<int>(valid.size())));
  InferenceSession malformed_control(normal, GetEnvironment());
  const auto malformed_status = malformed_control.Load(malformed.data(), static_cast<int>(malformed.size()));
  ASSERT_FALSE(malformed_status.IsOK());
  EXPECT_NE(malformed_status.ErrorMessage().find("protobuf parsing failed"), std::string::npos);
}

TEST(M0FullGpuPreInitAdmissionTest, ReviewedImplementationMetadataSurvivesMovesAndRegistryInsertion) {
  auto definition = KernelDefBuilder().SetName("Identity").SetDomain(kOnnxDomain)
      .Provider(kCpuExecutionProvider).SinceVersion(1)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).Build();
  KernelCreateInfo info(std::move(definition),
      [](FuncManager&, const OpKernelInfo&, std::unique_ptr<OpKernel>&) { return Status::OK(); });
  info.m0_implementation_id_v1 = "test-only-rw07-sentinel-metadata-v1";
  KernelCreateInfo moved(std::move(info));
  EXPECT_EQ(moved.m0_implementation_id_v1, "test-only-rw07-sentinel-metadata-v1");
  KernelRegistry registry;
  ASSERT_STATUS_OK(registry.Register(std::move(moved)));
  ASSERT_EQ(registry.GetKernelCreateMap().size(), 1U);
  EXPECT_EQ(registry.GetKernelCreateMap().begin()->second.m0_implementation_id_v1,
      "test-only-rw07-sentinel-metadata-v1");
}

namespace {
SessionOptions StrictIngressOptions() {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("m0.fullgpu.preinit_admission_v1", "1"));
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("m0.fullgpu.preinit_abi_v1", "1"));
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, "1"));
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableAheadOfTimeFunctionInlining, "1"));
  return options;
}

// No get area: even one parser read must reach underflow/uflow/xsgetn.
class IngressReadSentinel final : public std::streambuf {
 public:
  int reads = 0;
 protected:
  int_type underflow() override { ++reads; return traits_type::eof(); }
  int_type uflow() override { ++reads; return traits_type::eof(); }
  std::streamsize xsgetn(char*, std::streamsize) override { ++reads; return 0; }
};

void ExpectPreloadDenial(const Status& status) {
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT"), std::string::npos);
}
}  // namespace

TEST(M0FullGpuPreInitAdmissionTest, StreamIngressDeniesWithoutAnyParserRead) {
  const auto strict = StrictIngressOptions();
  IngressReadSentinel buffer;
  std::istream stream(&buffer);
  InferenceSession session(strict, GetEnvironment());
  ExpectPreloadDenial(session.Load(stream));
  EXPECT_EQ(buffer.reads, 0);
  EXPECT_TRUE(stream.good());
  try {
    InferenceSession constructor(strict, GetEnvironment(), stream);
    FAIL() << "Stream constructor must deny before reading";
  } catch (const OnnxRuntimeException& error) {
    EXPECT_NE(std::string(error.what()).find("RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT"), std::string::npos);
  }
  EXPECT_EQ(buffer.reads, 0);
  EXPECT_TRUE(stream.good());
  InferenceSession control(SessionOptions{}, GetEnvironment());
  EXPECT_FALSE(control.Load(stream).IsOK());
  EXPECT_GT(buffer.reads, 0);
}

TEST(M0FullGpuPreInitAdmissionTest, MissingPathIngressDeniesBeforeFileError) {
  const auto strict = StrictIngressOptions();
  const PathString missing = ORT_TSTR("/rw07-ingress-nonexistent-directory/model.onnx");
  InferenceSession session(strict, GetEnvironment());
  ExpectPreloadDenial(session.Load(missing));
  try {
    InferenceSession constructor(strict, GetEnvironment(), missing);
    FAIL() << "Path constructor must deny before loading";
  } catch (const OnnxRuntimeException& error) {
    EXPECT_NE(std::string(error.what()).find("RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT"), std::string::npos);
  }
  InferenceSession control(SessionOptions{}, GetEnvironment());
  const auto status = control.Load(missing);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.ErrorMessage().find("RW07_PRELOAD_DENY"), std::string::npos);
}

TEST(M0FullGpuPreInitAdmissionTest, ExplicitOrtFormatIngressStillDeniesBeforeFormatValidation) {
  auto strict = StrictIngressOptions();
  ASSERT_STATUS_OK(strict.config_options.AddConfigEntry(kOrtSessionOptionsConfigLoadModelFormat, "ORT"));
  const std::string invalid("invalid ort flatbuffer");
  InferenceSession session(strict, GetEnvironment());
  ExpectPreloadDenial(session.Load(invalid.data(), static_cast<int>(invalid.size())));
  ExpectPreloadDenial(session.Load(PathString(ORT_TSTR("/rw07-ingress-nonexistent-directory/model.ort"))));
  SessionOptions normal;
  ASSERT_STATUS_OK(normal.config_options.AddConfigEntry(kOrtSessionOptionsConfigLoadModelFormat, "ORT"));
  InferenceSession control(normal, GetEnvironment());
  const auto status = control.Load(invalid.data(), static_cast<int>(invalid.size()));
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.ErrorMessage().find("RW07_PRELOAD_DENY"), std::string::npos);
}

TEST(M0FullGpuPreInitAdmissionTest, InitializeIngressDeniesBeforeMissingModelAndInvalidOptions) {
  auto strict = StrictIngressOptions();
  InferenceSession session(strict, GetEnvironment());
  ExpectPreloadDenial(session.Initialize());
  ASSERT_STATUS_OK(strict.config_options.AddConfigEntry("m0.fullgpu.preinit_abi_v1", "99"));
  InferenceSession bad_abi(strict, GetEnvironment());
  const auto abi_status = bad_abi.Initialize();
  ASSERT_FALSE(abi_status.IsOK());
  EXPECT_NE(abi_status.ErrorMessage().find("RW07_PREINIT_DENY_ABI"), std::string::npos);
  ASSERT_STATUS_OK(strict.config_options.AddConfigEntry("m0.fullgpu.preinit_abi_v1", "1"));
  ASSERT_STATUS_OK(strict.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, "0"));
  InferenceSession bad_options(strict, GetEnvironment());
  const auto option_status = bad_options.Initialize();
  ASSERT_FALSE(option_status.IsOK());
  EXPECT_NE(option_status.ErrorMessage().find("RW07_PREINIT_DENY_OPTIONS"), std::string::npos);
  InferenceSession control(SessionOptions{}, GetEnvironment());
  const auto missing_model = control.Initialize();
  ASSERT_FALSE(missing_model.IsOK());
  EXPECT_EQ(missing_model.ErrorMessage().find("RW07_PREINIT_DENY"), std::string::npos);
}
namespace {
void ExpectIngressDiagnostic(const Status& status, const char* diagnostic) {
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find(diagnostic), std::string::npos) << status.ErrorMessage();
}

void ExpectConstructorAuthorityDenial(const SessionOptions& options) {
  try {
    InferenceSession session(options, GetEnvironment());
    FAIL() << "Strict constructor must reject replacement or callback authority";
  } catch (const OnnxRuntimeException& error) {
    EXPECT_NE(std::string(error.what()).find("RW07_PREINIT_DENY_AUTHORITY"), std::string::npos);
  }
}

class CountingIngressFactory final : public IExecutionProviderFactory {
 public:
  explicit CountingIngressFactory(int& calls) : calls_(calls) {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ++calls_;
    return std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  }
 private:
  int& calls_;
};
}  // namespace

TEST(M0FullGpuPreInitAdmissionTest, StrictIntentSurvivesDisableAndWholeOptionsReplacement) {
  for (bool replace : {false, true}) {
    InferenceSession session(StrictIngressOptions(), GetEnvironment());
    if (replace) {
      session.GetMutableSessionOptions() = SessionOptions{};
    } else {
      ASSERT_STATUS_OK(session.GetMutableSessionOptions().config_options.AddConfigEntry(
          "m0.fullgpu.preinit_admission_v1", "0"));
    }
    IngressReadSentinel buffer;
    std::istream stream(&buffer);
    ExpectIngressDiagnostic(session.Load(stream), "RW07_PREINIT_DENY_INTENT_MUTATION");
    EXPECT_EQ(buffer.reads, 0);
    EXPECT_TRUE(stream.good());
    ExpectIngressDiagnostic(session.Load(PathString(ORT_TSTR("/rw07-immutable-missing/model.onnx"))),
                            "RW07_PREINIT_DENY_INTENT_MUTATION");
    ExpectIngressDiagnostic(session.Initialize(), "RW07_PREINIT_DENY_INTENT_MUTATION");
    const OrtModel editor_model{};
    ExpectIngressDiagnostic(session.Load(editor_model), "RW07_PREINIT_DENY_INTENT_MUTATION");
    ExpectIngressDiagnostic(session.ApplyUpdates(editor_model), "RW07_PREINIT_DENY_INTENT_MUTATION");
  }
}

TEST(M0FullGpuPreInitAdmissionTest, MutableAbiAndOptionsDenyBeforeReadAndInitialize) {
  for (bool abi : {false, true}) {
    InferenceSession session(StrictIngressOptions(), GetEnvironment());
    ASSERT_STATUS_OK(session.GetMutableSessionOptions().config_options.AddConfigEntry(
        abi ? "m0.fullgpu.preinit_abi_v1" : kOrtSessionOptionsConfigDisablePrepacking, abi ? "99" : "0"));
    const char* diagnostic = abi ? "RW07_PREINIT_DENY_ABI" : "RW07_PREINIT_DENY_OPTIONS";
    IngressReadSentinel buffer;
    std::istream stream(&buffer);
    ExpectIngressDiagnostic(session.Load(stream), diagnostic);
    EXPECT_EQ(buffer.reads, 0);
    ExpectIngressDiagnostic(session.Initialize(), diagnostic);
  }
}

TEST(M0FullGpuPreInitAdmissionTest, OrdinaryLoadedSessionRejectsLateOptInAndCanRestoreOrdinaryOptions) {
  Model model("rw07_late_optin", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto type;
  type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& input = graph.GetOrCreateNodeArg("X", &type);
  auto& output = graph.GetOrCreateNodeArg("Y", &type);
  graph.AddNode("identity", "Identity", "", {&input}, {&output});
  graph.SetInputs({&input});
  graph.SetOutputs({&output});
  ASSERT_STATUS_OK(graph.Resolve());
  auto proto = model.ToProto();
  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));
  InferenceSession session(SessionOptions{}, GetEnvironment());
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  session.GetMutableSessionOptions() = StrictIngressOptions();
  ExpectIngressDiagnostic(session.Initialize(), "RW07_PREINIT_DENY_INTENT_MUTATION");
  session.GetMutableSessionOptions() = SessionOptions{};
  ASSERT_STATUS_OK(session.Initialize());
}

TEST(M0FullGpuPreInitAdmissionTest, StrictConstructorRejectsThreadAndLoggingAuthorityWithoutCallbacks) {
  int thread_calls = 0;
  int logging_calls = 0;
  auto thread_options = StrictIngressOptions();
  thread_options.custom_create_thread_fn = [](void* state, OrtThreadWorkerFn, void*) -> OrtCustomThreadHandle {
    ++*static_cast<int*>(state);
    return nullptr;
  };
  thread_options.custom_thread_creation_options = &thread_calls;
  ExpectConstructorAuthorityDenial(thread_options);
  auto logging_options = StrictIngressOptions();
  logging_options.user_logging_function = [](void* state, OrtLoggingLevel, const char*, const char*,
                                              const char*, const char*) {
    ++*static_cast<int*>(state);
  };
  logging_options.user_logging_param = &logging_calls;
  ExpectConstructorAuthorityDenial(logging_options);
  EXPECT_EQ(thread_calls, 0);
  EXPECT_EQ(logging_calls, 0);
}

TEST(M0FullGpuPreInitAdmissionTest, StrictConstructorAndMutableIngressRejectInitializerAuthority) {
  OrtValue replacement;
  auto shared = StrictIngressOptions();
  shared.initializers_to_share_map.emplace("weight", &replacement);
  ExpectConstructorAuthorityDenial(shared);
  InferenceSession session(StrictIngressOptions(), GetEnvironment());
  session.GetMutableSessionOptions().initializers_to_share_map.emplace("weight", &replacement);
  ExpectIngressDiagnostic(session.Initialize(), "RW07_PREINIT_DENY_AUTHORITY");
#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
  char backing = 0;
  auto mapped = StrictIngressOptions();
  mapped.external_initializer_files_mmap.emplace(ORT_TSTR("weight.bin"), std::make_pair(&backing, size_t{1}));
  ExpectConstructorAuthorityDenial(mapped);
  auto external = StrictIngressOptions();
  external.external_initializers.emplace("weight", replacement);
  ExpectConstructorAuthorityDenial(external);
#endif
}

TEST(M0FullGpuPreInitAdmissionTest, DirectRegistrationAndPrepackDenyBeforeSessionInstallation) {
  auto provider = std::make_shared<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  auto registry = std::make_shared<CustomRegistry>();
  PrepackedWeightsContainer weights;
  InferenceSession strict(StrictIngressOptions(), GetEnvironment());
  ExpectPreloadDenial(strict.RegisterExecutionProvider(provider));
  EXPECT_EQ(strict.GetExecutionProviders().Get(kCpuExecutionProvider), nullptr);
  ExpectPreloadDenial(strict.RegisterCustomRegistry(registry));
  ExpectPreloadDenial(strict.AddPrePackedWeightsContainer(&weights));
  ExpectPreloadDenial(strict.AddCustomOpDomains({}));
  EXPECT_EQ(weights.GetNumberOfElements(), 0U);
  InferenceSession control(SessionOptions{}, GetEnvironment());
  ASSERT_STATUS_OK(control.RegisterExecutionProvider(provider));
  EXPECT_EQ(control.GetExecutionProviders().Get(kCpuExecutionProvider), provider.get());
  ASSERT_STATUS_OK(control.RegisterCustomRegistry(registry));
  ASSERT_STATUS_OK(control.AddPrePackedWeightsContainer(&weights));
}

TEST(M0FullGpuPreInitAdmissionTest, CHelperRejectsStrictAndMutatedIntentBeforeProviderFactory) {
  int calls = 0;
  OrtSessionOptions options;
  options.value = StrictIngressOptions();
  options.provider_factories.push_back(std::make_shared<CountingIngressFactory>(calls));
  OrtEnv* raw_env = nullptr;
  ASSERT_EQ(OrtApis::CreateEnv(ORT_LOGGING_LEVEL_WARNING, "rw07_strict_creation", &raw_env), nullptr);
  std::unique_ptr<OrtEnv, decltype(&OrtApis::ReleaseEnv)> env(raw_env, OrtApis::ReleaseEnv);
  std::unique_ptr<InferenceSession> created;
  OrtStatus* creation_status = CreateSessionAndLoadModel(&options, env.get(), ORT_TSTR("/"), nullptr, 0, created);
  ASSERT_NE(creation_status, nullptr);
  EXPECT_NE(std::string(OrtApis::GetErrorMessage(creation_status)).find("RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT"),
            std::string::npos);
  OrtApis::ReleaseStatus(creation_status);
  EXPECT_EQ(created, nullptr);
  EXPECT_EQ(calls, 0);
  InferenceSession session(options.value, GetEnvironment());
  for (bool disable : {false, true}) {
    if (disable) {
      ASSERT_STATUS_OK(session.GetMutableSessionOptions().config_options.AddConfigEntry(
          "m0.fullgpu.preinit_admission_v1", "0"));
      ASSERT_STATUS_OK(options.value.config_options.AddConfigEntry("m0.fullgpu.preinit_admission_v1", "0"));
    }
    OrtStatus* status = InitializeSession(&options, session);
    ASSERT_NE(status, nullptr);
    EXPECT_NE(std::string(OrtApis::GetErrorMessage(status)).find(
                  disable ? "RW07_PREINIT_DENY_INTENT_MUTATION" : "RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT"),
              std::string::npos);
    OrtApis::ReleaseStatus(status);
    EXPECT_EQ(calls, 0);
  }
  InferenceSession control(SessionOptions{}, GetEnvironment());
  OrtStatus* status = InitializeSession(&options, control);
  ASSERT_NE(status, nullptr);  // No model: ordinary registration still happens.
  EXPECT_EQ(std::string(OrtApis::GetErrorMessage(status)).find("RW07_PREINIT_DENY"), std::string::npos);
  OrtApis::ReleaseStatus(status);
  EXPECT_EQ(calls, 1);
}

#if !defined(_WIN32)
TEST(M0FullGpuPreInitAdmissionTest, EnvironmentAuthorityDeniesBeforeConstructorAndMutableIngress) {
  const char* key = "ORT_LOAD_CONFIG_FROM_MODEL";
  const char* prior = std::getenv(key);
  const bool had_prior = prior != nullptr;
  const std::string saved = prior ? prior : "";
  struct RestoreEnvironment {
    const char* key;
    bool present;
    std::string value;
    ~RestoreEnvironment() {
      if (present) setenv(key, value.c_str(), 1);
      else unsetenv(key);
    }
  } restore{key, had_prior, saved};
  ASSERT_EQ(setenv(key, "0", 1), 0);
  InferenceSession session(StrictIngressOptions(), GetEnvironment());
  ASSERT_EQ(setenv(key, "1", 1), 0);
  IngressReadSentinel buffer;
  std::istream stream(&buffer);
  ExpectIngressDiagnostic(session.Load(stream), "RW07_PREINIT_DENY_ENVIRONMENT");
  EXPECT_EQ(buffer.reads, 0);
  ExpectIngressDiagnostic(session.Initialize(), "RW07_PREINIT_DENY_ENVIRONMENT");
  try {
    InferenceSession constructor(StrictIngressOptions(), GetEnvironment());
    FAIL() << "Strict constructor must reject model-derived option authority";
  } catch (const OnnxRuntimeException& error) {
    EXPECT_NE(std::string(error.what()).find("RW07_PREINIT_DENY_ENVIRONMENT"), std::string::npos);
  }
}
#endif

#if !defined(ORT_MINIMAL_BUILD)
namespace {
ModelOptions CallbackDenyOptions() {
  ModelOptions options;
  options.m0_callback_deny_requested_v1 = true;
  return options;
}

void ExpectOwnerDenial(const Status& status) {
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("RW07_CALLBACK_DENY_UNREVIEWED_OWNER"), std::string::npos);
}

template <typename F>
void ExpectOwnerConstructorDenial(F&& construct) {
  try {
    construct();
    FAIL() << "Strict Model construction must refuse before graph work";
  } catch (const OnnxRuntimeException& error) {
    EXPECT_NE(std::string(error.what()).find("RW07_CALLBACK_DENY_UNREVIEWED_OWNER"), std::string::npos);
  }
}
}  // namespace

TEST(M0CallbackOwnerIntegrationTest, ConstructorsRefuseBeforeInvalidProtoValidationOrConsumption) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  const auto options = CallbackDenyOptions();
  ExpectOwnerConstructorDenial([&]() { Model model("deny", false, logger, options); });
  ONNX_NAMESPACE::ModelProto invalid;
  invalid.set_producer_name("caller-owned-no-graph");
  const std::string before = invalid.SerializeAsString();
  ExpectOwnerConstructorDenial([&]() { Model model(invalid, nullptr, logger, options); });
  ExpectOwnerConstructorDenial([&]() { Model model(std::move(invalid), nullptr, logger, options); });
  EXPECT_EQ(invalid.SerializeAsString(), before);
}

TEST(M0CallbackOwnerIntegrationTest, ProtoLoadersRefuseWithoutReplacingOutputOrConsumingInput) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  const auto options = CallbackDenyOptions();
  ONNX_NAMESPACE::ModelProto invalid;
  invalid.set_producer_name("no-graph");
  const auto before = invalid.SerializeAsString();
  for (bool populated : {false, true}) {
    std::shared_ptr<Model> output = populated ? std::make_shared<Model>() : nullptr;
    const auto* original = output.get();
    ExpectOwnerDenial(Model::Load(invalid, output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
    ExpectOwnerDenial(Model::Load(invalid, ORT_TSTR("synthetic"), output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
    ExpectOwnerDenial(Model::Load(std::move(invalid), output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
    ExpectOwnerDenial(Model::Load(std::move(invalid), ORT_TSTR("synthetic"), output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
    EXPECT_EQ(invalid.SerializeAsString(), before);
  }
  std::shared_ptr<Model> ordinary;
  const auto status = Model::Load(invalid, ordinary, nullptr, logger);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.ErrorMessage().find("RW07_CALLBACK_DENY"), std::string::npos);
  EXPECT_EQ(ordinary, nullptr);
}

TEST(M0CallbackOwnerIntegrationTest, ByteLoadersDenyBeforeMalformedParsingAndOutputAssignment) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  const auto options = CallbackDenyOptions();
  std::array<unsigned char, 1> malformed{0xff};
  for (bool populated : {false, true}) {
    std::shared_ptr<Model> output = populated ? std::make_shared<Model>() : nullptr;
    const auto* original = output.get();
    ExpectOwnerDenial(Model::LoadFromBytes(1, malformed.data(), output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
    ExpectOwnerDenial(Model::LoadFromBytes(1, malformed.data(), ORT_TSTR("synthetic"), output,
                                           nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
  }
  std::shared_ptr<Model> ordinary;
  const auto status = Model::LoadFromBytes(1, malformed.data(), ordinary, nullptr, logger);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.ErrorMessage().find("RW07_CALLBACK_DENY"), std::string::npos);
}

TEST(M0CallbackOwnerIntegrationTest, FileAndDescriptorLoadersDenyBeforeIoErrorsAndOutputAssignment) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  const auto options = CallbackDenyOptions();
  const PathString missing = ORT_TSTR("/rw07-nonexistent-directory/owner-test.onnx");
  for (bool populated : {false, true}) {
    std::shared_ptr<Model> output = populated ? std::make_shared<Model>() : nullptr;
    const auto* original = output.get();
    ExpectOwnerDenial(Model::Load(missing, output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
    ExpectOwnerDenial(Model::Load(missing, missing, output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
    ExpectOwnerDenial(Model::Load(-1, output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
    ExpectOwnerDenial(Model::Load(-1, missing, output, nullptr, logger, options));
    EXPECT_EQ(output.get(), original);
  }
  std::shared_ptr<Model> ordinary;
  for (const auto& status : {Model::Load(missing, ordinary, nullptr, logger),
                             Model::Load(-1, ordinary, nullptr, logger)}) {
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.ErrorMessage().find("RW07_CALLBACK_DENY"), std::string::npos);
  }
}

TEST(M0CallbackOwnerIntegrationTest, EditorLoaderDeniesWithoutPublishingOrReplacingModel) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  const auto options = CallbackDenyOptions();
  OrtModel editor;
  for (bool populated : {false, true}) {
    std::unique_ptr<Model> output = populated ? std::make_unique<Model>() : nullptr;
    const auto* original = output.get();
    ExpectOwnerDenial(Model::LoadFromModelEditorApiModel(editor, nullptr, options, logger, output));
    EXPECT_EQ(output.get(), original);
  }
}

TEST(M0CallbackOwnerIntegrationTest, OrtFormatLoaderDeniesBeforeMalformedMetadataOrOutputAssignment) {
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(fbs::CreateModel(builder));
  const auto* serialized = flatbuffers::GetRoot<fbs::Model>(builder.GetBufferPointer());
  ASSERT_NE(serialized, nullptr);
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  OrtFormatLoadOptions options;
  for (bool populated : {false, true}) {
    std::unique_ptr<Model> output = populated ? std::make_unique<Model>() : nullptr;
    const auto* original = output.get();
    ExpectOwnerDenial(Model::LoadFromOrtFormat(*serialized, nullptr, options, logger, output, true));
    EXPECT_EQ(output.get(), original);
  }
  std::unique_ptr<Model> ordinary;
  const auto status = Model::LoadFromOrtFormat(*serialized, nullptr, options, logger, ordinary);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.ErrorMessage().find("RW07_CALLBACK_DENY"), std::string::npos);
  // Ordinary loaders can publish before a later failure; no rollback claim.
}

TEST(M0CallbackOwnerIntegrationTest, OrdinaryOwnerDoesNotRetroactivelyAttachMutableDenialRequest) {
  ModelOptions options;
  Model model("ordinary", false, DefaultLoggingManager().DefaultLogger(), options);
  options.m0_callback_deny_requested_v1 = true;
  for (auto channel : {M0CallbackPolicyV1::Channel::kConstruction,
                       M0CallbackPolicyV1::Channel::kResolve}) {
    ASSERT_STATUS_OK(model.M0CheckCallbackPolicyV1(channel));
  }
  Model default_model;
  ASSERT_STATUS_OK(default_model.M0CheckCallbackPolicyV1(M0CallbackPolicyV1::Channel::kConstruction));
}
// Integrate inside onnxruntime::test; needs core/graph/schema_registry.h and
// onnx/defs/shape_inference.h. These model-local wrappers count actual selected
// callbacks and forwarded context operations, not arithmetic instructions.
namespace {
struct M0OrdinaryRouteCounts {
  int dispatch = 0;
  std::array<int, 4> data_requests{};
  std::array<int, 4> data_present{};
  int value_attribute_present = 0;
  int output_requests = 0;
};

class M0CountingInferenceContext final : public ONNX_NAMESPACE::InferenceContext {
 public:
  M0CountingInferenceContext(ONNX_NAMESPACE::InferenceContext& inner, M0OrdinaryRouteCounts& counts)
      : inner_(inner), counts_(counts) {}
  const ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const override {
    const auto* result = inner_.getAttribute(name);
    if (name == "value" && result != nullptr) ++counts_.value_attribute_present;
    return result;
  }
  size_t getNumInputs() const override { return inner_.getNumInputs(); }
  const ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const override { return inner_.getInputType(index); }
  bool hasInput(size_t index) const override { return inner_.hasInput(index); }
  const ONNX_NAMESPACE::TensorProto* getInputData(size_t index) const override {
    const auto* result = inner_.getInputData(index);
    if (index < counts_.data_requests.size()) {
      ++counts_.data_requests[index];
      if (result != nullptr) ++counts_.data_present[index];
    }
    return result;
  }
  size_t getNumOutputs() const override { return inner_.getNumOutputs(); }
  ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) override {
    ++counts_.output_requests;
    return inner_.getOutputType(index);
  }
  bool hasOutput(size_t index) override { return inner_.hasOutput(index); }
  ONNX_NAMESPACE::GraphInferencer* getGraphAttributeInferencer(const std::string& name) override {
    return inner_.getGraphAttributeInferencer(name);
  }
  const ONNX_NAMESPACE::SparseTensorProto* getInputSparseData(size_t index) const override {
    return inner_.getInputSparseData(index);
  }
  const ONNX_NAMESPACE::TensorShapeProto* getSymbolicInput(size_t index) const override {
    return inner_.getSymbolicInput(index);
  }
  std::string getDisplayName() const override { return inner_.getDisplayName(); }

 private:
  ONNX_NAMESPACE::InferenceContext& inner_;
  M0OrdinaryRouteCounts& counts_;
};

void M0ControlInput(ONNX_NAMESPACE::GraphProto& graph, const char* name, int dtype,
                    std::initializer_list<int64_t> dims) {
  auto* input = graph.add_input();
  input->set_name(name);
  auto* tensor = input->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(dtype);
  auto* shape = tensor->mutable_shape();
  for (int64_t dim : dims) shape->add_dim()->set_dim_value(dim);
}

void M0ControlIntLiteral(ONNX_NAMESPACE::GraphProto& graph, const char* name,
                         std::initializer_list<int64_t> dims,
                         std::initializer_list<int64_t> values) {
  auto* literal = graph.add_initializer();
  literal->set_name(name);
  literal->set_data_type(ONNX_NAMESPACE::TensorProto::INT64);
  for (int64_t dim : dims) literal->add_dims(dim);
  for (int64_t value : values) literal->add_int64_data(value);
}

void M0RunOrdinaryControl(const char* op_name, int original_version, bool missing_range_limit = false) {
  using namespace ONNX_NAMESPACE;
  const auto* selected = OpSchemaRegistry::Schema(op_name, original_version, "");
  ASSERT_NE(selected, nullptr);
  ASSERT_EQ(selected->SinceVersion(), original_version);
  auto callback = selected->GetTypeAndShapeInferenceFunction();
  ASSERT_TRUE(static_cast<bool>(callback));
  M0OrdinaryRouteCounts counts;
  OpSchema wrapper = *selected;
  wrapper.SetDomain("rw07.synthetic.callback").SinceVersion(1).TypeAndShapeInferenceFunction([callback, &counts](InferenceContext& context) {
    ++counts.dispatch;
    M0CountingInferenceContext forwarding(context, counts);
    callback(forwarding);
  });
  auto registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<OpSchema> schemas{std::move(wrapper)};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, "rw07.synthetic.callback", 0, 1));
  IOnnxRuntimeOpSchemaRegistryList registries{registry};
  ModelProto proto;
  proto.set_ir_version(8);
  auto* opset = proto.add_opset_import();
  opset->set_domain("rw07.synthetic.callback");
  opset->set_version(1);
  auto* graph = proto.mutable_graph();
  graph->set_name("rw07_safe_ordinary_callback");
  auto* node = graph->add_node();
  node->set_name("observed_selected_callback");
  node->set_op_type(op_name);
  node->set_domain("rw07.synthetic.callback");
  node->add_output("Y");
  graph->add_output()->set_name("Y");  // Deliberately no prefilled output shape.
  std::vector<int64_t> expected;
  if (std::string(op_name) == "Resize") {
    M0ControlInput(*graph, "X", TensorProto::FLOAT, {2, 3});
    auto* scales = graph->add_initializer();
    scales->set_name("scales");
    scales->set_data_type(TensorProto::FLOAT);
    scales->add_dims(2);
    scales->add_float_data(1.5f);
    scales->add_float_data(2.0f);
    node->add_input("X");
    node->add_input("");
    node->add_input("scales");
    expected = {3, 6};
  } else if (std::string(op_name) == "Range") {
    M0ControlIntLiteral(*graph, "start", {}, {0});
    if (missing_range_limit)
      M0ControlInput(*graph, "limit", TensorProto::INT64, {});
    else
      M0ControlIntLiteral(*graph, "limit", {}, {5});
    M0ControlIntLiteral(*graph, "delta", {}, {2});
    for (const char* name : {"start", "limit", "delta"}) node->add_input(name);
    expected = {3};
  } else if (std::string(op_name) == "ConstantOfShape") {
    M0ControlIntLiteral(*graph, "shape", {2}, {2, 3});
    node->add_input("shape");
    auto* value = node->add_attribute();
    value->set_name("value");
    value->set_type(AttributeProto::TENSOR);
    value->mutable_t()->set_data_type(TensorProto::FLOAT);
    value->mutable_t()->add_dims(1);  // One-element rank1 is valid.
    value->mutable_t()->add_float_data(0.25f);
    expected = {2, 3};
  } else {
    ASSERT_EQ(std::string(op_name), "STFT");
    M0ControlInput(*graph, "signal", TensorProto::FLOAT, {1, 16, 1});
    M0ControlIntLiteral(*graph, "step", {}, {4});
    M0ControlIntLiteral(*graph, "length", {}, {8});
    for (const char* name : {"signal", "step", "", "length"}) node->add_input(name);
    auto* onesided = node->add_attribute();
    onesided->set_name("onesided");
    onesided->set_type(AttributeProto::INT);
    onesided->set_i(1);
    expected = {1, 3, 5, 2};
  }
  Model model(std::move(proto), &registries, DefaultLoggingManager().DefaultLogger());
  ASSERT_EQ(counts.dispatch, 0);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_GT(counts.dispatch, 0);
  EXPECT_GT(counts.output_requests, 0);
  const auto* output_type = model.MainGraph().GetNodeArg("Y")->TypeAsProto();
  ASSERT_NE(output_type, nullptr);
  ASSERT_TRUE(output_type->tensor_type().has_shape());
  const auto& shape = output_type->tensor_type().shape();
  ASSERT_EQ(shape.dim_size(), static_cast<int>(expected.size()));
  if (missing_range_limit) {
    EXPECT_FALSE(shape.dim(0).has_dim_value());
    EXPECT_GT(counts.data_requests[1], 0);
    EXPECT_EQ(counts.data_present[1], 0);
  } else {
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(shape.dim(static_cast<int>(i)).dim_value(), expected[i]);
  }
  if (std::string(op_name) == "Resize") EXPECT_GT(counts.data_present[2], 0);
  if (std::string(op_name) == "Range") {
    EXPECT_GT(counts.data_present[0], 0);
    EXPECT_GT(counts.data_present[2], 0);
    if (!missing_range_limit) EXPECT_GT(counts.data_present[1], 0);
  }
  if (std::string(op_name) == "ConstantOfShape") {
    EXPECT_GT(counts.data_present[0], 0);
    EXPECT_GT(counts.value_attribute_present, 0);
  }
  if (std::string(op_name) == "STFT") {
    EXPECT_GT(counts.data_present[1], 0);
    EXPECT_GT(counts.data_present[3], 0);
  }
  ASSERT_FALSE(model.MainGraph().GraphResolveNeeded());
  const int before_clean_repeat = counts.dispatch;
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_EQ(counts.dispatch, before_clean_repeat);
}
}  // namespace

TEST(M0CallbackOwnerIntegrationTest, OrdinaryResize19PayloadRoute) { M0RunOrdinaryControl("Resize", 19); }
TEST(M0CallbackOwnerIntegrationTest, OrdinaryRange11PayloadRoute) { M0RunOrdinaryControl("Range", 11); }
TEST(M0CallbackOwnerIntegrationTest, OrdinaryRange11MissingLimitUnknown) { M0RunOrdinaryControl("Range", 11, true); }
TEST(M0CallbackOwnerIntegrationTest, OrdinaryConstantOfShape20AttributeAndShapeRoute) { M0RunOrdinaryControl("ConstantOfShape", 20); }
TEST(M0CallbackOwnerIntegrationTest, OrdinarySTFT17PayloadRoute) { M0RunOrdinaryControl("STFT", 17); }

namespace {
void M0CheckControlShape(const ONNX_NAMESPACE::TypeProto* type, std::initializer_list<int64_t> expected) {
  ASSERT_NE(type, nullptr);
  ASSERT_TRUE(type->has_tensor_type());
  ASSERT_TRUE(type->tensor_type().has_shape());
  const auto& shape = type->tensor_type().shape();
  ASSERT_EQ(shape.dim_size(), static_cast<int>(expected.size()));
  int i = 0;
  for (auto dim : expected) {
    EXPECT_TRUE(shape.dim(i).has_dim_value());
    EXPECT_EQ(shape.dim(i++).dim_value(), dim);
  }
}

void M0NestedGraph(ONNX_NAMESPACE::GraphProto& graph, int depth) {
  using namespace ONNX_NAMESPACE;
  graph.set_name("finite_nested_control");
  M0ControlInput(graph, "X", TensorProto::FLOAT, {2, 3});
  auto* output = graph.add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  auto* node = graph.add_node();
  node->set_op_type(depth == 0 ? "Identity" : "Nested");
  node->set_domain("rw07.synthetic.routes");
  node->add_input("X");
  node->add_output("Y");
  if (depth > 0) {
    auto* body = node->add_attribute();
    body->set_name("body");
    body->set_type(AttributeProto::GRAPH);
    M0NestedGraph(*body->mutable_g(), depth - 1);
  }
}
}  // namespace

TEST(M0OrdinaryRoutesTest, NestedDirectGraphInferencerAndParentlessFunctionBody) {
  using namespace ONNX_NAMESPACE;
  int recursive_entries = 0;
  int leaf_entries = 0;
  auto identity = *OpSchemaRegistry::Schema("Identity", 19, "");
  auto identity_callback = identity.GetTypeAndShapeInferenceFunction();
  identity.SetDomain("rw07.synthetic.routes").SinceVersion(1).TypeAndShapeInferenceFunction([&leaf_entries, identity_callback](InferenceContext& context) {
    ++leaf_entries;
    identity_callback(context);
  });
  OpSchema nested;
  nested.SetName("Nested").SetDomain("rw07.synthetic.routes").SinceVersion(1).Input(0, "X", "Finite input", "tensor(float)").Output(0, "Y", "Inferred output", "tensor(float)").Attr("body", "Finite nested body", AttributeProto::GRAPH, true).TypeAndShapeInferenceFunction([&recursive_entries](InferenceContext& context) {
    auto* inferencer = context.getGraphAttributeInferencer("body");
    ASSERT_NE(inferencer, nullptr);
    ++recursive_entries;
    const auto outputs = inferencer->doInferencing({context.getInputType(0)}, {});
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_NE(outputs[0], nullptr);
    *context.getOutputType(0) = *outputs[0];
  });
  auto registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<OpSchema> schemas{std::move(identity), std::move(nested)};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, "rw07.synthetic.routes", 0, 1));
  IOnnxRuntimeOpSchemaRegistryList registries{registry};
  ModelProto proto;
  proto.set_ir_version(8);
  auto* opset = proto.add_opset_import();
  opset->set_domain("rw07.synthetic.routes");
  opset->set_version(1);
  M0NestedGraph(*proto.mutable_graph(), 2);
  Model model(std::move(proto), &registries, DefaultLoggingManager().DefaultLogger());
  ASSERT_EQ(recursive_entries, 0);
  ASSERT_EQ(leaf_entries, 0);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_GE(recursive_entries, 2);
  EXPECT_GT(leaf_entries, 0);
  M0CheckControlShape(model.MainGraph().GetNodeArg("Y")->TypeAsProto(), {2, 3});
  const int before_recursive = recursive_entries;
  const int before_leaf = leaf_entries;
  ASSERT_FALSE(model.MainGraph().GraphResolveNeeded());
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_EQ(recursive_entries, before_recursive);
  EXPECT_EQ(leaf_entries, before_leaf);

  FunctionProto function;
  auto* function_opset = function.add_opset_import();
  function_opset->set_domain("rw07.synthetic.routes");
  function_opset->set_version(1);
  FunctionImpl derived(model.MainGraph(), function);
  auto& body = derived.MutableBody();
  EXPECT_EQ(body.ParentGraph(), nullptr);
  EXPECT_EQ(&body.GetModel(), &model);
  TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  for (int64_t dim : {2, 3}) input_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  auto& x = body.GetOrCreateNodeArg("body_x", &input_type);
  auto& y = body.GetOrCreateNodeArg("body_y", nullptr);
  body.AddNode("body_leaf", "Identity", "", {&x}, {&y}, nullptr, "rw07.synthetic.routes");
  body.SetInputs({&x});
  body.SetOutputs({&y});
  ASSERT_STATUS_OK(body.Resolve());
  EXPECT_GT(leaf_entries, before_leaf);
  M0CheckControlShape(y.TypeAsProto(), {2, 3});
}

TEST(M0OrdinaryRoutesTest, FunctionAvailabilityDispatchAndSelectedSoftmaxBody18) {
  using namespace ONNX_NAMESPACE;
  int builder_entries = 0;
  OpSchema schema;
  schema.SetName("Available").SetDomain("rw07.synthetic.function").SinceVersion(1).Input(0, "X", "Input", "tensor(float)").Output(0, "Y", "Output", "tensor(float)").TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput).SetContextDependentFunctionBodyBuilder([&builder_entries](const FunctionBodyBuildContext& context, const OpSchema& selected, FunctionProto& body) {
    ++builder_entries;
    EXPECT_TRUE(context.hasInput(0));
    EXPECT_NE(context.getInputType(0), nullptr);
    auto* node = body.add_node();
    node->set_op_type("Identity");
    node->add_input("X");
    node->add_output("Y");
    selected.BuildFunction(body);
    return true;
  });
  auto registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<OpSchema> schemas{std::move(schema)};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, "rw07.synthetic.function", 0, 1));
  IOnnxRuntimeOpSchemaRegistryList registries{registry};
  ModelProto proto;
  proto.set_ir_version(8);
  auto* opset = proto.add_opset_import();
  opset->set_domain("rw07.synthetic.function");
  opset->set_version(1);
  auto* graph = proto.mutable_graph();
  graph->set_name("function_availability");
  M0ControlInput(*graph, "X", TensorProto::FLOAT, {2, 3});
  graph->add_output()->set_name("Y");
  auto* calling = graph->add_node();
  calling->set_op_type("Available");
  calling->set_domain("rw07.synthetic.function");
  calling->add_input("X");
  calling->add_output("Y");
  Model model(std::move(proto), &registries, DefaultLoggingManager().DefaultLogger());
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  auto& node = *model.MainGraph().Nodes().begin();
  builder_entries = 0;
  EXPECT_TRUE(node.CanBeInlined());
  EXPECT_EQ(builder_entries, 1);
  FunctionProto body;
  EXPECT_TRUE(node.TryGetFunctionProto(body));
  EXPECT_EQ(builder_entries, 2);
  ASSERT_EQ(body.node_size(), 1);
  EXPECT_EQ(body.node(0).op_type(), "Identity");

  const auto* softmax = OpSchemaRegistry::Schema("Softmax", 20, "");
  ASSERT_NE(softmax, nullptr);
  ASSERT_EQ(softmax->SinceVersion(), 13);
  ASSERT_TRUE(softmax->HasContextDependentFunctionWithOpsetVersion(18));
  NodeProto softmax_node;
  softmax_node.add_input("X");
  softmax_node.add_output("Y");
  auto* axis = softmax_node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(1);
  std::vector<TypeProto> types{*model.MainGraph().GetNodeArg("X")->TypeAsProto()};
  FunctionBodyBuildContextImpl context(softmax_node, types);
  FunctionProto selected_body;
  ASSERT_TRUE(softmax->BuildContextDependentFunction(context, selected_body, 20));
  ASSERT_GT(selected_body.node_size(), 0);
  bool imported20 = false;
  bool reduce_max_axes_input = false;
  bool axis_literal = false;
  for (const auto& imported : selected_body.opset_import()) {
    if (imported.domain().empty()) imported20 = imported.version() == 20;
  }
  for (const auto& expanded : selected_body.node()) {
    if (expanded.op_type() == "Constant" && expanded.output_size() == 1 && expanded.output(0) == "axes") {
      for (const auto& attribute : expanded.attribute()) {
        if (attribute.name() == "value") {
          const auto& literal = attribute.t();
          axis_literal = literal.data_type() == TensorProto::INT64 && literal.dims_size() == 1 &&
                         literal.dims(0) == 1 && literal.int64_data_size() == 1 && literal.int64_data(0) == 1;
        }
      }
    }
    if (expanded.op_type() == "ReduceMax") {
      reduce_max_axes_input = expanded.input_size() == 2 && expanded.input(1) == "axes";
    }
  }
  EXPECT_TRUE(imported20);
  EXPECT_TRUE(reduce_max_axes_input);
  EXPECT_TRUE(axis_literal);
}

TEST(M0OrdinaryRoutesTest, ModelOwnedLocalFunctionInferenceCallback) {
  using namespace ONNX_NAMESPACE;
  Model owner("local_function_owner", false, DefaultLoggingManager().DefaultLogger());
  FunctionProto function;
  function.set_name("LocalIdentity");
  function.set_domain("rw07.synthetic.local");
  function.add_input("X");
  function.add_output("Y");
  auto* dependency = function.add_opset_import();
  dependency->set_domain("");
  dependency->set_version(20);
  auto* body = function.add_node();
  body->set_op_type("Identity");
  body->add_input("X");
  body->add_output("Y");
  const std::unordered_map<std::string, const FunctionProto*> functions{
      {function_utils::GetFunctionIdentifier(function.domain(), function.name()), &function}};
  const std::unordered_map<std::string, int> versions{{"", 20}, {function.domain(), 1}};
  SchemaRegistryManager function_registry;
  auto schema = function_utils::CreateSchema(function.domain(), function.name(), "", functions, versions,
                                             function_registry,
                                             DefaultLoggingManager().DefaultLogger(), true, &owner);
  ASSERT_NE(schema, nullptr);
  auto local_callback = schema->GetTypeAndShapeInferenceFunction();
  ASSERT_TRUE(static_cast<bool>(local_callback));
  M0OrdinaryRouteCounts counts;
  schema->TypeAndShapeInferenceFunction([local_callback, &counts](InferenceContext& context) {
    ++counts.dispatch;
    M0CountingInferenceContext forwarding(context, counts);
    local_callback(forwarding);
  });
  auto registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<OpSchema> schemas{*schema};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, function.domain(), 0, 1));
  IOnnxRuntimeOpSchemaRegistryList registries{registry};
  ModelProto proto;
  proto.set_ir_version(8);
  for (const auto& version : versions) {
    auto* imported = proto.add_opset_import();
    imported->set_domain(version.first);
    imported->set_version(version.second);
  }
  auto* graph = proto.mutable_graph();
  graph->set_name("direct_local_inference");
  M0ControlInput(*graph, "X", TensorProto::FLOAT, {2, 3});
  graph->add_output()->set_name("Y");
  auto* node = graph->add_node();
  node->set_op_type(function.name());
  node->set_domain(function.domain());
  node->add_input("X");
  node->add_output("Y");
  Model model(std::move(proto), &registries, DefaultLoggingManager().DefaultLogger());
  EXPECT_EQ(counts.dispatch, 0);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_GT(counts.dispatch, 0);
  EXPECT_GT(counts.output_requests, 0);
  M0CheckControlShape(model.MainGraph().GetNodeArg("Y")->TypeAsProto(), {2, 3});
}

namespace {
struct M0PropagationCounts {
  int dispatch = 0;
  std::array<int, 2> reads{};
  std::array<int, 2> present{};
  int writes = 0;
  int input_type_reads = 0;
};

class M0ForwardingDataPropagationContext final : public ONNX_NAMESPACE::DataPropagationContext {
 public:
  M0ForwardingDataPropagationContext(ONNX_NAMESPACE::DataPropagationContext& inner, M0PropagationCounts& counts)
      : inner_(inner), counts_(counts) {}
  const ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const override {
    return inner_.getAttribute(name);
  }
  size_t getNumInputs() const override { return inner_.getNumInputs(); }
  const ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const override {
    ++counts_.input_type_reads;
    return inner_.getInputType(index);
  }
  size_t getNumOutputs() const override { return inner_.getNumOutputs(); }
  const ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) const override { return inner_.getOutputType(index); }
  const ONNX_NAMESPACE::TensorShapeProto* getInputData(size_t index) override {
    const auto* result = inner_.getInputData(index);
    if (index < counts_.reads.size()) {
      ++counts_.reads[index];
      if (result != nullptr) ++counts_.present[index];
    }
    return result;
  }
  void addOutputData(size_t index, ONNX_NAMESPACE::TensorShapeProto&& value) override {
    ++counts_.writes;
    inner_.addOutputData(index, std::move(value));
  }

 private:
  ONNX_NAMESPACE::DataPropagationContext& inner_;
  M0PropagationCounts& counts_;
};

void M0RunOrdinaryPropagationChain(bool custom_arithmetic) {
  using namespace ONNX_NAMESPACE;
  constexpr const char* domain = "rw07.synthetic.propagation";
  std::array<M0PropagationCounts, 4> counts;
  M0OrdinaryRouteCounts consumer_counts;
  int payload_checks = 0;
  std::vector<OpSchema> schemas;
  const std::array<const char*, 4> names{"Shape", "Add", "Mul", "Sub"};
  const std::array<int, 4> versions{19, 14, 14, 14};
  for (size_t i = 0; i < names.size(); ++i) {
    const auto* selected = OpSchemaRegistry::Schema(names[i], versions[i], "");
    ASSERT_NE(selected, nullptr);
    ASSERT_EQ(selected->SinceVersion(), versions[i]);
    auto callback = selected->GetDataPropagationFunction();
    ASSERT_TRUE(static_cast<bool>(callback));
    OpSchema wrapper = *selected;
    wrapper.SetDomain(domain).SinceVersion(1).PartialDataPropagationFunction(
        [callback, &counts, i, custom_arithmetic](DataPropagationContext& context) {
          ++counts[i].dispatch;
          // Suppress only arithmetic ONNX writes in the independently observed custom route.
          if (custom_arithmetic && i != 0) return;
          M0ForwardingDataPropagationContext forwarding(context, counts[i]);
          callback(forwarding);
        });
    schemas.push_back(std::move(wrapper));
  }
  const auto* selected_consumer = OpSchemaRegistry::Schema("ConstantOfShape", 20, "");
  ASSERT_NE(selected_consumer, nullptr);
  ASSERT_EQ(selected_consumer->SinceVersion(), 20);
  auto consumer_callback = selected_consumer->GetTypeAndShapeInferenceFunction();
  ASSERT_TRUE(static_cast<bool>(consumer_callback));
  OpSchema consumer = *selected_consumer;
  consumer.SetDomain(domain).SinceVersion(1).TypeAndShapeInferenceFunction(
      [consumer_callback, &consumer_counts, &payload_checks](InferenceContext& context) {
        ++consumer_counts.dispatch;
        M0CountingInferenceContext forwarding(context, consumer_counts);
        const auto* payload = forwarding.getInputData(0);
        ASSERT_NE(payload, nullptr);
        EXPECT_EQ(payload->data_type(), TensorProto::INT64);
        ASSERT_EQ(payload->dims_size(), 1);
        EXPECT_EQ(payload->dims(0), 1);
        ASSERT_EQ(payload->int64_data_size(), 1);
        EXPECT_EQ(payload->int64_data(0), 12);
        ++payload_checks;
        consumer_callback(forwarding);
      });
  schemas.push_back(std::move(consumer));
  auto registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, domain, 0, 1));
  IOnnxRuntimeOpSchemaRegistryList registries{registry};
  ModelProto proto;
  proto.set_ir_version(8);
  auto* opset = proto.add_opset_import();
  opset->set_domain(domain);
  opset->set_version(1);
  auto* graph = proto.mutable_graph();
  graph->set_name(custom_arithmetic ? "rw07_custom_propagation" : "rw07_onnx_propagation");
  M0ControlInput(*graph, "X", TensorProto::FLOAT, {2});
  M0ControlInput(*graph, "Z", TensorProto::FLOAT, {3});
  auto add = [&](const char* name, const char* op, std::initializer_list<const char*> inputs, const char* output) {
    auto* node = graph->add_node();
    node->set_name(name);
    node->set_op_type(op);
    node->set_domain(domain);
    for (const char* input : inputs) node->add_input(input);
    node->add_output(output);
  };
  add("shape_x", "Shape", {"X"}, "S2");
  add("shape_z", "Shape", {"Z"}, "S3");
  add("add_values", "Add", {"S2", "S3"}, "S5");
  add("multiply_values", "Mul", {"S5", "S3"}, "S15");
  add("subtract_values", "Sub", {"S15", "S3"}, "S12");
  add("observe_propagated_value", "ConstantOfShape", {"S12"}, "Y");
  graph->add_output()->set_name("Y");
  ASSERT_EQ(graph->initializer_size(), 0);
  Model model(std::move(proto), &registries, DefaultLoggingManager().DefaultLogger());
  for (const auto& count : counts) ASSERT_EQ(count.dispatch, 0);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_GT(counts[0].dispatch, 0);
  EXPECT_GT(counts[0].writes, 0);
  EXPECT_GT(counts[0].input_type_reads, 0);
  for (size_t i = 1; i < counts.size(); ++i) {
    EXPECT_GT(counts[i].dispatch, 0);
    if (custom_arithmetic) {
      EXPECT_EQ(counts[i].reads[0], 0);
      EXPECT_EQ(counts[i].reads[1], 0);
      EXPECT_EQ(counts[i].writes, 0);
    } else {
      EXPECT_GT(counts[i].reads[0], 0);
      EXPECT_GT(counts[i].reads[1], 0);
      EXPECT_GT(counts[i].present[0], 0);
      EXPECT_GT(counts[i].present[1], 0);
      EXPECT_GT(counts[i].writes, 0);
    }
  }
  // Resolve clears transient carried values; observe them in the downstream callback.
  EXPECT_GT(payload_checks, 0);
  EXPECT_GT(consumer_counts.data_present[0], 0);
  EXPECT_GT(consumer_counts.output_requests, 0);
  const auto* output = model.MainGraph().GetNodeArg("Y")->TypeAsProto();
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(output->tensor_type().has_shape());
  ASSERT_EQ(output->tensor_type().shape().dim_size(), 1);
  EXPECT_EQ(output->tensor_type().shape().dim(0).dim_value(), 12);
  ASSERT_FALSE(model.MainGraph().GraphResolveNeeded());
  const auto before = counts;
  const int before_consumer = consumer_counts.dispatch;
  const int before_payload = payload_checks;
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  for (size_t i = 0; i < counts.size(); ++i) {
    EXPECT_EQ(counts[i].dispatch, before[i].dispatch);
    EXPECT_EQ(counts[i].reads, before[i].reads);
    EXPECT_EQ(counts[i].present, before[i].present);
    EXPECT_EQ(counts[i].writes, before[i].writes);
    EXPECT_EQ(counts[i].input_type_reads, before[i].input_type_reads);
  }
  EXPECT_EQ(consumer_counts.dispatch, before_consumer);
  EXPECT_EQ(payload_checks, before_payload);
}
}  // namespace

TEST(M0OrdinaryRoutesTest, OrdinaryOnnxPartialPropagationReadDispatchWriteChain) {
  M0RunOrdinaryPropagationChain(false);
}

TEST(M0OrdinaryRoutesTest, OrdinaryOrtCustomPropagationWithoutOnnxArithmeticWrites) {
  M0RunOrdinaryPropagationChain(true);
}

namespace {
struct M0CarriedRouteCounts {
  int dispatch = 0;
  std::array<int, 5> reads{};
  std::array<int, 5> present{};
  int writes = 0;
};

class M0CarriedRouteContext final : public ONNX_NAMESPACE::DataPropagationContext {
 public:
  M0CarriedRouteContext(ONNX_NAMESPACE::DataPropagationContext& inner,
                        M0CarriedRouteCounts& counts, const std::vector<int64_t>& expected)
      : inner_(inner), counts_(counts), expected_(expected) {}
  const ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const override {
    return inner_.getAttribute(name);
  }
  size_t getNumInputs() const override { return inner_.getNumInputs(); }
  const ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const override { return inner_.getInputType(index); }
  size_t getNumOutputs() const override { return inner_.getNumOutputs(); }
  const ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) const override { return inner_.getOutputType(index); }
  const ONNX_NAMESPACE::TensorShapeProto* getInputData(size_t index) override {
    const auto* value = inner_.getInputData(index);
    if (index < counts_.reads.size()) {
      ++counts_.reads[index];
      if (value != nullptr) ++counts_.present[index];
    }
    return value;
  }
  void addOutputData(size_t index, ONNX_NAMESPACE::TensorShapeProto&& value) override {
    EXPECT_EQ(index, 0u);
    if (!expected_.empty()) {
      ASSERT_EQ(value.dim_size(), static_cast<int>(expected_.size()));
      for (size_t i = 0; i < expected_.size(); ++i) {
        ASSERT_TRUE(value.dim(static_cast<int>(i)).has_dim_value());
        EXPECT_EQ(value.dim(static_cast<int>(i)).dim_value(), expected_[i]);
      }
    }
    ++counts_.writes;
    inner_.addOutputData(index, std::move(value));
  }

 private:
  ONNX_NAMESPACE::DataPropagationContext& inner_;
  M0CarriedRouteCounts& counts_;
  const std::vector<int64_t>& expected_;
};

void M0RunCarriedIndexChain(bool explicit_slice_inputs) {
  using namespace ONNX_NAMESPACE;
  constexpr const char* domain = "rw07.synthetic.carried.index";
  const std::array<const char*, 5> names{"Shape", "Concat", "Gather", "Slice", "Cast"};
  const std::array<int, 5> versions{19, 13, 13, 13, 19};
  const std::array<std::vector<int64_t>, 5> expected{{{}, {2, 3, 4}, {4, 2, 3}, {2, 3}, {2, 3}}};
  std::array<M0CarriedRouteCounts, 5> counts;
  std::vector<OpSchema> schemas;
  for (size_t i = 0; i < names.size(); ++i) {
    const auto* selected = OpSchemaRegistry::Schema(names[i], versions[i], "");
    ASSERT_NE(selected, nullptr);
    ASSERT_EQ(selected->SinceVersion(), versions[i]);
    const auto callback = selected->GetDataPropagationFunction();
    ASSERT_TRUE(static_cast<bool>(callback));
    OpSchema wrapper = *selected;
    wrapper.SetDomain(domain).SinceVersion(1).PartialDataPropagationFunction(
        [&, callback, i](DataPropagationContext& context) {
          ++counts[i].dispatch;
          M0CarriedRouteContext forwarding(context, counts[i], expected[i]);
          callback(forwarding);
        });
    schemas.push_back(std::move(wrapper));
  }
  int payload_checks = 0;
  M0OrdinaryRouteCounts consumer_counts;
  const auto* selected = OpSchemaRegistry::Schema("ConstantOfShape", 20, "");
  ASSERT_NE(selected, nullptr);
  ASSERT_EQ(selected->SinceVersion(), 20);
  const auto callback = selected->GetTypeAndShapeInferenceFunction();
  ASSERT_TRUE(static_cast<bool>(callback));
  OpSchema consumer = *selected;
  consumer.SetDomain(domain).SinceVersion(1).TypeAndShapeInferenceFunction(
      [&, callback](InferenceContext& context) {
        ++consumer_counts.dispatch;
        M0CountingInferenceContext forwarding(context, consumer_counts);
        const auto* value = forwarding.getInputData(0);
        ASSERT_NE(value, nullptr);
        EXPECT_EQ(value->data_type(), TensorProto::INT64);
        ASSERT_EQ(value->dims_size(), 1);
        EXPECT_EQ(value->dims(0), 2);
        ASSERT_EQ(value->int64_data_size(), 2);
        EXPECT_EQ(value->int64_data(0), 2);
        EXPECT_EQ(value->int64_data(1), 3);
        ++payload_checks;
        callback(forwarding);
      });
  schemas.push_back(std::move(consumer));
  auto registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, domain, 0, 1));
  IOnnxRuntimeOpSchemaRegistryList registries{registry};
  ModelProto proto;
  proto.set_ir_version(8);
  auto* imported = proto.add_opset_import();
  imported->set_domain(domain);
  imported->set_version(1);
  auto* graph = proto.mutable_graph();
  graph->set_name("carried_index_chain");
  auto add = [&](const char* op, std::initializer_list<const char*> inputs, const char* output) -> NodeProto* {
    auto* node = graph->add_node();
    node->set_op_type(op);
    node->set_domain(domain);
    for (const char* input : inputs) node->add_input(input);
    node->add_output(output);
    return node;
  };
  auto attr = [](NodeProto* node, const char* name, int64_t value) {
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto::INT);
    attribute->set_i(value);
  };
  // Descriptor-derived Shape values reach every data callback without seeded carried fields.
  M0ControlInput(*graph, "X", TensorProto::FLOAT, {2, 3});
  M0ControlInput(*graph, "Z", TensorProto::FLOAT, {4});
  M0ControlInput(*graph, "I", TensorProto::FLOAT, {2, 0, 1});
  M0ControlInput(*graph, "B", TensorProto::FLOAT, {1});
  M0ControlInput(*graph, "E", TensorProto::FLOAT, {3});
  add("Shape", {"X"}, "SX");
  add("Shape", {"Z"}, "SZ");
  add("Shape", {"I"}, "SI");
  add("Shape", {"B"}, "SB");
  add("Shape", {"E"}, "SE");
  attr(add("Concat", {"SX", "SZ"}, "SC"), "axis", 0);
  attr(add("Gather", {"SC", "SI"}, "SG"), "axis", 0);
  if (explicit_slice_inputs) {
    M0ControlInput(*graph, "A", TensorProto::FLOAT, {0});
    M0ControlInput(*graph, "P", TensorProto::FLOAT, {1});
    add("Shape", {"A"}, "SA");
    add("Shape", {"P"}, "SP");
    add("Slice", {"SG", "SB", "SE", "SA", "SP"}, "SS");
  } else {
    add("Slice", {"SG", "SB", "SE"}, "SS");
  }
  attr(add("Cast", {"SS"}, "SK"), "to", TensorProto::INT64);
  add("ConstantOfShape", {"SK"}, "Y");
  graph->add_output()->set_name("Y");
  ASSERT_EQ(graph->initializer_size(), 0);
  Model model(std::move(proto), &registries, DefaultLoggingManager().DefaultLogger());
  for (const auto& count : counts) ASSERT_EQ(count.dispatch, 0);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_GT(counts[0].dispatch, 0);
  EXPECT_GT(counts[0].writes, 0);
  const std::array<size_t, 5> required_reads{0, 2, 2, explicit_slice_inputs ? 5u : 3u, 1};
  for (size_t i = 1; i < counts.size(); ++i) {
    EXPECT_GT(counts[i].dispatch, 0);
    EXPECT_GT(counts[i].writes, 0);
    for (size_t input = 0; input < required_reads[i]; ++input) {
      EXPECT_GT(counts[i].reads[input], 0);
      EXPECT_GT(counts[i].present[input], 0);
    }
  }
  if (!explicit_slice_inputs) {
    EXPECT_EQ(counts[3].reads[3], 0);
    EXPECT_EQ(counts[3].reads[4], 0);
  }
  EXPECT_GT(payload_checks, 0);
  EXPECT_GT(consumer_counts.data_present[0], 0);
  EXPECT_GT(consumer_counts.output_requests, 0);
  M0CheckControlShape(model.MainGraph().GetNodeArg("Y")->TypeAsProto(), {2, 3});
  ASSERT_FALSE(model.MainGraph().GraphResolveNeeded());
  const auto before = counts;
  const int before_payload = payload_checks;
  const int before_consumer = consumer_counts.dispatch;
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  for (size_t i = 0; i < counts.size(); ++i) {
    EXPECT_EQ(counts[i].dispatch, before[i].dispatch);
    EXPECT_EQ(counts[i].reads, before[i].reads);
    EXPECT_EQ(counts[i].present, before[i].present);
    EXPECT_EQ(counts[i].writes, before[i].writes);
  }
  EXPECT_EQ(payload_checks, before_payload);
  EXPECT_EQ(consumer_counts.dispatch, before_consumer);
}
}  // namespace

TEST(M0OrdinaryCarriedRoutesTest, ConcatGatherSliceCastExplicitAxesAndSteps) {
  M0RunCarriedIndexChain(true);
}

TEST(M0OrdinaryCarriedRoutesTest, ConcatGatherSliceCastDefaultAxesAndSteps) {
  M0RunCarriedIndexChain(false);
}

namespace {
void M0CustomSeedValues(NodeArg& arg, std::initializer_list<int64_t> values) {
  auto& carried = arg.GetMutableInferredShapeValues();
  carried.emplace();
  for (int64_t value : values) carried->add_dim()->set_dim_value(value);
}

void M0CustomCheckValues(const NodeArg& arg, std::initializer_list<int64_t> expected) {
  ASSERT_TRUE(arg.GetInferredShapeValues().has_value());
  EXPECT_FALSE(arg.GetInferredShapeScalarValue().has_value());
  const auto& carried = *arg.GetInferredShapeValues();
  ASSERT_EQ(carried.dim_size(), static_cast<int>(expected.size()));
  int i = 0;
  for (int64_t value : expected) {
    ASSERT_TRUE(carried.dim(i).has_dim_value());
    EXPECT_EQ(carried.dim(i++).dim_value(), value);
  }
}

void M0CustomCheckScalar(const NodeArg& arg, int64_t expected) {
  ASSERT_TRUE(arg.GetInferredShapeScalarValue().has_value());
  EXPECT_EQ(*arg.GetInferredShapeScalarValue(), expected);
  EXPECT_FALSE(arg.GetInferredShapeValues().has_value());
}

struct M0CustomDirectCounts {
  int factory_requests = 0;
  int infer_requests = 0;
  int initializer_reads = 0;
};

void M0CustomRunFactory(const Node& node, NodeArg& output, Graph& graph,
                        const ONNX_NAMESPACE::TypeProto& onnx_result,
                        M0CustomDirectCounts& counts) {
  GetInitializedInputValuesFunc read = [&graph, &counts](const std::string& name,
                                                         TensorShapeVector& values,
                                                         int& rank) {
    ++counts.initializer_reads;
    const auto* literal = graph.GetConstantInitializer(name, true);
    EXPECT_NE(literal, nullptr);
    if (!literal) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expected finite initializer");
    EXPECT_EQ(literal->data_type(), ONNX_NAMESPACE::TensorProto::INT64);
    EXPECT_LE(literal->dims_size(), 1);
    EXPECT_EQ(literal->int64_data_size(), 1);
    if (literal->dims_size() == 1) EXPECT_EQ(literal->dims(0), 1);
    rank = literal->dims_size();
    values.assign(literal->int64_data().begin(), literal->int64_data().end());
    return Status::OK();
  };
  ++counts.factory_requests;
  auto propagator = CreateCustomDataPropagation(node, output, read, onnx_result,
                                                DefaultLoggingManager().DefaultLogger());
  ASSERT_NE(propagator, nullptr);
  ++counts.infer_requests;
  ASSERT_STATUS_OK(propagator->infer());
}

ONNX_NAMESPACE::TypeProto M0CustomIntType(std::initializer_list<int64_t> dims) {
  ONNX_NAMESPACE::TypeProto type;
  auto* tensor = type.mutable_tensor_type();
  tensor->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
  auto* shape = tensor->mutable_shape();
  for (int64_t dim : dims) shape->add_dim()->set_dim_value(dim);
  return type;
}

void M0CustomAddLiteral(Graph& graph, const char* name, int64_t value, bool scalar = false) {
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(ONNX_NAMESPACE::TensorProto::INT64);
  if (!scalar) tensor.add_dims(1);
  tensor.add_int64_data(value);
  graph.AddInitializedTensor(tensor);
}
}  // namespace

TEST(M0OrdinaryCustomRoutesTest, DirectCustomScalarAndDivPublication) {
  const std::array<const char*, 4> ops{{"Add", "Sub", "Mul", "Div"}};
  const std::array<int64_t, 4> expected{{15, 9, 36, 4}};
  for (size_t op = 0; op < ops.size(); ++op) {
    for (int rank_case = 0; rank_case < 4; ++rank_case) {
      SCOPED_TRACE(::testing::Message() << ops[op] << " rank case " << rank_case);
      const bool left_rank1 = (rank_case & 1) != 0;
      const bool right_rank1 = (rank_case & 2) != 0;
      Model model("rw07_custom_arithmetic", false, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();
      auto scalar_type = M0CustomIntType({});
      auto vector_type = M0CustomIntType({1});
      auto& left = graph.GetOrCreateNodeArg("left", left_rank1 ? &vector_type : &scalar_type);
      auto& right = graph.GetOrCreateNodeArg("right", right_rank1 ? &vector_type : &scalar_type);
      auto& output = graph.GetOrCreateNodeArg("output", (left_rank1 || right_rank1) ? &vector_type : &scalar_type);
      if (left_rank1)
        M0CustomSeedValues(left, {12});
      else
        left.SetInferredShapeScalarValue(12);
      if (right_rank1)
        M0CustomSeedValues(right, {3});
      else
        right.SetInferredShapeScalarValue(3);
      // Distinct stale channels make output publication and opposite-channel clearing observable.
      output.SetInferredShapeScalarValue(901);
      M0CustomSeedValues(output, {902});
      auto& node = graph.AddNode("direct", ops[op], "", {&left, &right}, {&output});
      ONNX_NAMESPACE::TypeProto empty_onnx_result;
      M0CustomDirectCounts counts;
      M0CustomRunFactory(node, output, graph, empty_onnx_result, counts);
      EXPECT_EQ(counts.factory_requests, 1);
      EXPECT_EQ(counts.infer_requests, 1);
      EXPECT_EQ(counts.initializer_reads, 0);
      if (left_rank1 || right_rank1)
        M0CustomCheckValues(output, {expected[op]});
      else
        M0CustomCheckScalar(output, expected[op]);
      auto populated_onnx_result = M0CustomIntType({expected[op]});
      EXPECT_EQ(CreateCustomDataPropagation(node, output, {}, populated_onnx_result,
                                            DefaultLoggingManager().DefaultLogger()),
                nullptr);
    }
  }
}

TEST(M0OrdinaryCustomRoutesTest, DirectCustomGatherInitializerGetterAndPublication) {
  for (int index_case = 0; index_case < 4; ++index_case) {
    SCOPED_TRACE(index_case);
    Model model("rw07_custom_gather", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    auto data_type = M0CustomIntType({3});
    auto index_type = M0CustomIntType({1});
    auto& data = graph.GetOrCreateNodeArg("data", &data_type);
    auto scalar_type = M0CustomIntType({});
    auto& index = graph.GetOrCreateNodeArg("index", index_case == 3 ? &scalar_type : &index_type);
    auto& output = graph.GetOrCreateNodeArg("output", index_case == 3 ? &scalar_type : &index_type);
    M0CustomSeedValues(data, {2, 7, 11});
    M0CustomAddLiteral(graph, "index", index_case == 0 || index_case == 3 ? 1 : (index_case == 1 ? -1 : 3),
                       index_case == 3);
    auto& node = graph.AddNode("direct", "Gather", "", {&data, &index}, {&output});
    ONNX_NAMESPACE::TypeProto empty_onnx_result;
    M0CustomDirectCounts counts;
    M0CustomRunFactory(node, output, graph, empty_onnx_result, counts);
    EXPECT_EQ(counts.factory_requests, 1);
    EXPECT_EQ(counts.infer_requests, 1);
    EXPECT_EQ(counts.initializer_reads, 1);
    if (index_case < 2) {
      M0CustomCheckValues(output, {index_case == 0 ? 7 : 11});
    } else if (index_case == 3) {
      M0CustomCheckScalar(output, 7);
    } else {
      // HandleNegativeAxis throws before dim access; Gather catches/logs and returns OK unset.
      EXPECT_FALSE(output.GetInferredShapeScalarValue().has_value());
      EXPECT_FALSE(output.GetInferredShapeValues().has_value());
    }
  }
}

TEST(M0OrdinaryCustomRoutesTest, DirectCustomSqueezeUnsqueezeScalarPublication) {
  Model model("rw07_custom_rank", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto vector_type = M0CustomIntType({1});
  auto scalar_type = M0CustomIntType({});
  auto& data = graph.GetOrCreateNodeArg("data", &vector_type);
  auto& axis = graph.GetOrCreateNodeArg("axis", &vector_type);
  auto& scalar = graph.GetOrCreateNodeArg("scalar", &scalar_type);
  auto& vector = graph.GetOrCreateNodeArg("vector", &vector_type);
  M0CustomSeedValues(data, {17});
  M0CustomAddLiteral(graph, "axis", 0);
  auto& squeeze = graph.AddNode("squeeze", "Squeeze", "", {&data, &axis}, {&scalar});
  auto& unsqueeze = graph.AddNode("unsqueeze", "Unsqueeze", "", {&scalar, &axis}, {&vector});
  ONNX_NAMESPACE::TypeProto empty_onnx_result;
  M0CustomDirectCounts scalar_counts;
  M0CustomRunFactory(squeeze, scalar, graph, empty_onnx_result, scalar_counts);
  M0CustomCheckScalar(scalar, 17);
  M0CustomRunFactory(unsqueeze, vector, graph, empty_onnx_result, scalar_counts);
  M0CustomCheckValues(vector, {17});
  EXPECT_EQ(scalar_counts.factory_requests, 2);
  EXPECT_EQ(scalar_counts.infer_requests, 2);
  EXPECT_EQ(scalar_counts.initializer_reads, 0);

  auto matrix_type = M0CustomIntType({1, 1});
  auto& declined = graph.GetOrCreateNodeArg("declined", &matrix_type);
  auto& lift = graph.AddNode("lift", "Unsqueeze", "", {&vector, &axis}, {&declined});
  M0CustomDirectCounts decline_counts;
  M0CustomRunFactory(lift, declined, graph, empty_onnx_result, decline_counts);
  EXPECT_EQ(decline_counts.factory_requests, 1);
  EXPECT_EQ(decline_counts.infer_requests, 1);
  EXPECT_EQ(decline_counts.initializer_reads, 0);
  EXPECT_FALSE(declined.GetInferredShapeValues().has_value());
  EXPECT_FALSE(declined.GetInferredShapeScalarValue().has_value());
}

namespace {
struct M0FunctionBuildReads {
  std::array<int, 3> input_presence{};
  std::array<int, 3> output_presence{};
  std::array<int, 3> type_reads{};
  int attribute_reads = 0;
};

class M0ForwardFunctionBuildContext final : public ONNX_NAMESPACE::FunctionBodyBuildContext {
 public:
  M0ForwardFunctionBuildContext(const ONNX_NAMESPACE::FunctionBodyBuildContext& inner,
                                M0FunctionBuildReads& counts) : inner_(inner), counts_(counts) {}
  const ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const override {
    ++counts_.attribute_reads;
    return inner_.getAttribute(name);
  }
  bool hasInput(int index) const override {
    if (index >= 0 && index < 3) ++counts_.input_presence[index];
    return inner_.hasInput(index);
  }
  bool hasOutput(int index) const override {
    if (index >= 0 && index < 3) ++counts_.output_presence[index];
    return inner_.hasOutput(index);
  }
  const ONNX_NAMESPACE::TypeProto* getInputType(int index) const override {
    if (index >= 0 && index < 3) ++counts_.type_reads[index];
    return inner_.getInputType(index);
  }

 private:
  const ONNX_NAMESPACE::FunctionBodyBuildContext& inner_;
  M0FunctionBuildReads& counts_;
};

ONNX_NAMESPACE::TypeProto M0FunctionFloatType(std::initializer_list<int64_t> dims) {
  ONNX_NAMESPACE::TypeProto type;
  auto* tensor = type.mutable_tensor_type();
  tensor->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  auto* shape = tensor->mutable_shape();
  for (int64_t dim : dims) shape->add_dim()->set_dim_value(dim);
  return type;
}

const ONNX_NAMESPACE::NodeProto* M0FunctionFindOutput(const ONNX_NAMESPACE::FunctionProto& body,
                                                      const char* output) {
  for (const auto& node : body.node()) {
    if (node.output_size() == 1 && node.output(0) == output) return &node;
  }
  return nullptr;
}

void M0FunctionCheckIntLiteral(const ONNX_NAMESPACE::FunctionProto& body,
                               const char* output, int64_t value) {
  const auto* node = M0FunctionFindOutput(body, output);
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->op_type(), "Constant");
  ASSERT_EQ(node->attribute_size(), 1);
  const auto& attr = node->attribute(0);
  ASSERT_EQ(attr.name(), "value");
  ASSERT_EQ(attr.type(), ONNX_NAMESPACE::AttributeProto::TENSOR);
  const auto& tensor = attr.t();
  EXPECT_EQ(tensor.data_type(), ONNX_NAMESPACE::TensorProto::INT64);
  ASSERT_EQ(tensor.dims_size(), 1);
  EXPECT_EQ(tensor.dims(0), 1);
  ASSERT_EQ(tensor.int64_data_size(), 1);
  EXPECT_EQ(tensor.int64_data(0), value);
}
void M0FunctionLayerNormInference(const ONNX_NAMESPACE::OpSchema& selected, bool bias) {
  using namespace ONNX_NAMESPACE;
  auto copied = selected;
  auto original = selected.GetTypeAndShapeInferenceFunction();
  ASSERT_TRUE(static_cast<bool>(original));
  M0OrdinaryRouteCounts counts;
  copied.SetDomain("rw07.synthetic.function.inference").SinceVersion(1).TypeAndShapeInferenceFunction([original, &counts](InferenceContext& context) {
    ++counts.dispatch;
    M0CountingInferenceContext forwarding(context, counts);
    original(forwarding);
  });
  auto registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<OpSchema> schemas{std::move(copied)};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, "rw07.synthetic.function.inference", 0, 1));
  IOnnxRuntimeOpSchemaRegistryList registries{registry};
  ModelProto proto;
  proto.set_ir_version(8);
  proto.add_opset_import()->set_version(20);
  auto* imported = proto.add_opset_import();
  imported->set_domain("rw07.synthetic.function.inference");
  imported->set_version(1);
  auto* graph = proto.mutable_graph();
  graph->set_name("layernorm_callback");
  M0ControlInput(*graph, "X", TensorProto::FLOAT, {2, 3});
  M0ControlInput(*graph, "Scale", TensorProto::FLOAT, {3});
  if (bias) M0ControlInput(*graph, "B", TensorProto::FLOAT, {3});
  for (const char* name : {"Y", "Mean", "InvStdDev"}) graph->add_output()->set_name(name);
  auto* node = graph->add_node();
  node->set_op_type("LayerNormalization");
  node->set_domain("rw07.synthetic.function.inference");
  for (const char* name : {"X", "Scale"}) node->add_input(name);
  node->add_input(bias ? "B" : "");
  for (const char* name : {"Y", "Mean", "InvStdDev"}) node->add_output(name);
  auto* axis = node->add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(1);
  Model model(std::move(proto), &registries, DefaultLoggingManager().DefaultLogger());
  EXPECT_EQ(counts.dispatch, 0);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_GT(counts.dispatch, 0);
  EXPECT_GE(counts.output_requests, 3);
  M0CheckControlShape(model.MainGraph().GetNodeArg("Y")->TypeAsProto(), {2, 3});
  M0CheckControlShape(model.MainGraph().GetNodeArg("Mean")->TypeAsProto(), {2, 1});
  M0CheckControlShape(model.MainGraph().GetNodeArg("InvStdDev")->TypeAsProto(), {2, 1});
  for (const char* name : {"Y", "Mean", "InvStdDev"}) {
    EXPECT_EQ(model.MainGraph().GetNodeArg(name)->TypeAsProto()->tensor_type().elem_type(), TensorProto::FLOAT);
  }
}
}  // namespace

TEST(M0OrdinaryFunctionRoutesTest, LayerNormalization17SelectsHistorical18BuilderAt20) {
  using namespace ONNX_NAMESPACE;
  const auto* schema = OpSchemaRegistry::Schema("LayerNormalization", 20, "");
  ASSERT_NE(schema, nullptr);
  ASSERT_EQ(schema->SinceVersion(), 17);
  ASSERT_TRUE(schema->HasContextDependentFunctionWithOpsetVersion(17));
  ASSERT_TRUE(schema->HasContextDependentFunctionWithOpsetVersion(18));
  EXPECT_FALSE(schema->HasContextDependentFunctionWithOpsetVersion(20));
  for (bool bias : {false, true}) {
    M0FunctionLayerNormInference(*schema, bias);
    NodeProto calling;
    calling.set_op_type("LayerNormalization");
    calling.add_input("X");
    calling.add_input("Scale");
    calling.add_input(bias ? "B" : "");
    calling.add_output("Y");
    calling.add_output("Mean");
    calling.add_output("InvStdDev");
    auto* axis = calling.add_attribute();
    axis->set_name("axis");
    axis->set_type(AttributeProto::INT);
    axis->set_i(1);
    std::vector<TypeProto> types{M0FunctionFloatType({2, 3}), M0FunctionFloatType({3}),
                                 bias ? M0FunctionFloatType({3}) : TypeProto{}};
    for (int requested : {17, 20}) {
      SCOPED_TRACE(::testing::Message() << "bias " << bias << " requested " << requested);
      FunctionBodyBuildContextImpl actual(calling, types);
      M0FunctionBuildReads counts;
      M0ForwardFunctionBuildContext forwarding(actual, counts);
      FunctionProto body;
      ASSERT_TRUE(schema->BuildContextDependentFunction(forwarding, body, requested));
      EXPECT_EQ(counts.type_reads[0], 1);
      EXPECT_EQ(counts.attribute_reads, 3);
      EXPECT_EQ(counts.input_presence[2], 1);
      EXPECT_EQ(counts.output_presence[1], 1);
      EXPECT_EQ(counts.output_presence[2], 1);
      bool imported_requested = false;
      for (const auto& imported : body.opset_import()) {
        if (imported.domain().empty()) imported_requested = imported.version() == requested;
      }
      EXPECT_TRUE(imported_requested);
      M0FunctionCheckIntLiteral(body, "Axis1D", 1);
      const auto* mean = M0FunctionFindOutput(body, "Mean2D");
      ASSERT_NE(mean, nullptr);
      ASSERT_EQ(mean->op_type(), "ReduceMean");
      if (requested == 20) {
        M0FunctionCheckIntLiteral(body, "Axes_1", 1);
        ASSERT_EQ(mean->input_size(), 2);
        EXPECT_EQ(mean->input(1), "Axes_1");
        EXPECT_EQ(mean->attribute_size(), 0);
      } else {
        EXPECT_EQ(M0FunctionFindOutput(body, "Axes_1"), nullptr);
        ASSERT_EQ(mean->input_size(), 1);
        ASSERT_EQ(mean->attribute_size(), 1);
        EXPECT_EQ(mean->attribute(0).name(), "axes");
        ASSERT_EQ(mean->attribute(0).ints_size(), 1);
        EXPECT_EQ(mean->attribute(0).ints(0), 1);
      }
      const auto* biased = M0FunctionFindOutput(body, "Biased");
      ASSERT_NE(biased, nullptr);
      EXPECT_EQ(biased->op_type(), bias ? "Add" : "Identity");
      EXPECT_NE(M0FunctionFindOutput(body, "Mean"), nullptr);
      EXPECT_NE(M0FunctionFindOutput(body, "InvStdDev"), nullptr);
    }
  }
}

TEST(M0OrdinaryFunctionRoutesTest, Clip13OptionalPresenceSelectsExactFunctionBranches) {
  using namespace ONNX_NAMESPACE;
  const auto* schema = OpSchemaRegistry::Schema("Clip", 20, "");
  ASSERT_NE(schema, nullptr);
  ASSERT_EQ(schema->SinceVersion(), 13);
  ASSERT_TRUE(schema->HasContextDependentFunctionWithOpsetVersion(13));
  const std::array<std::vector<std::string>, 4> expected{{{"Identity"}, {"Less", "Where"}, {"Less", "Where"}, {"Less", "Where", "Less", "Where"}}};
  for (int mask = 0; mask < 4; ++mask) {
    SCOPED_TRACE(mask);
    NodeProto calling;
    calling.set_op_type("Clip");
    calling.add_input("input");
    calling.add_input((mask & 1) ? "min" : "");
    calling.add_input((mask & 2) ? "max" : "");
    calling.add_output("output");
    std::vector<TypeProto> types{M0FunctionFloatType({2, 3}),
                                 (mask & 1) ? M0FunctionFloatType({}) : TypeProto{},
                                 (mask & 2) ? M0FunctionFloatType({}) : TypeProto{}};
    FunctionBodyBuildContextImpl actual(calling, types);
    M0FunctionBuildReads counts;
    M0ForwardFunctionBuildContext forwarding(actual, counts);
    FunctionProto body;
    ASSERT_TRUE(schema->BuildContextDependentFunction(forwarding, body, 20));
    EXPECT_EQ(counts.input_presence[1], 1);
    EXPECT_EQ(counts.input_presence[2], 1);
    ASSERT_EQ(body.node_size(), static_cast<int>(expected[mask].size()));
    for (int i = 0; i < body.node_size(); ++i) EXPECT_EQ(body.node(i).op_type(), expected[mask][i]);
    if (mask == 1 || mask == 3) {
      ASSERT_EQ(body.node(0).input_size(), 2);
      EXPECT_EQ(body.node(0).input(0), "input");
      EXPECT_EQ(body.node(0).input(1), "min");
      ASSERT_EQ(body.node(1).input_size(), 3);
      EXPECT_EQ(body.node(1).input(1), "min");
      EXPECT_EQ(body.node(1).input(2), "input");
    }
    if (mask == 2 || mask == 3) {
      const int offset = mask == 3 ? 2 : 0;
      ASSERT_EQ(body.node(offset).input_size(), 2);
      EXPECT_EQ(body.node(offset).input(0), "max");
      EXPECT_EQ(body.node(offset).input(1), mask == 3 ? "tmp" : "input");
      ASSERT_EQ(body.node(offset + 1).input_size(), 3);
      EXPECT_EQ(body.node(offset + 1).input(1), "max");
      EXPECT_EQ(body.node(offset + 1).input(2), mask == 3 ? "tmp" : "input");
    }
    const std::array<std::vector<std::vector<std::string>>, 4> inputs{{{{"input"}},
                                                                       {{"input", "min"}, {"input_less_than_min", "min", "input"}},
                                                                       {{"max", "input"}, {"input_large_than_max", "max", "input"}},
                                                                       {{"input", "min"}, {"input_less_than_min", "min", "input"}, {"max", "tmp"}, {"output_large_than_max", "max", "tmp"}}}};
    const std::array<std::vector<std::string>, 4> outputs{{{"output"}, {"input_less_than_min", "output"}, {"input_large_than_max", "output"}, {"input_less_than_min", "tmp", "output_large_than_max", "output"}}};
    for (int i = 0; i < body.node_size(); ++i) {
      const auto& expanded = body.node(i);
      ASSERT_EQ(expanded.input_size(), static_cast<int>(inputs[mask][i].size()));
      for (int j = 0; j < expanded.input_size(); ++j) EXPECT_EQ(expanded.input(j), inputs[mask][i][j]);
      ASSERT_EQ(expanded.output_size(), 1);
      EXPECT_EQ(expanded.output(0), outputs[mask][i]);
      EXPECT_EQ(expanded.attribute_size(), 0);
      EXPECT_TRUE(expanded.domain().empty());
    }
    ASSERT_EQ(body.node(body.node_size() - 1).output_size(), 1);
    EXPECT_EQ(body.node(body.node_size() - 1).output(0), "output");
  }
}

TEST(M0OrdinaryFunctionRoutesTest, StaticRelu14LiteralAndVersionCheckedNodeAvailability) {
  using namespace ONNX_NAMESPACE;
  const auto* schema = OpSchemaRegistry::Schema("Relu", 20, "");
  ASSERT_NE(schema, nullptr);
  ASSERT_EQ(schema->SinceVersion(), 14);
  ASSERT_TRUE(schema->HasFunction());
  EXPECT_FALSE(schema->HasContextDependentFunction());
  const auto* body = schema->GetFunction(18, true);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->node_size(), 3);
  EXPECT_EQ(body->node(0).op_type(), "Constant");
  EXPECT_EQ(body->node(1).op_type(), "CastLike");
  EXPECT_EQ(body->node(2).op_type(), "Max");
  const auto* zero = M0FunctionFindOutput(*body, "Zero");
  ASSERT_NE(zero, nullptr);
  ASSERT_EQ(zero->attribute_size(), 1);
  const auto& literal = zero->attribute(0).t();
  EXPECT_EQ(literal.data_type(), TensorProto::FLOAT);
  EXPECT_EQ(literal.dims_size(), 0);
  ASSERT_EQ(literal.float_data_size(), 1);
  const float zero_value = literal.float_data(0);
  uint32_t zero_bits = 1;
  static_assert(sizeof(zero_value) == sizeof(zero_bits));
  std::memcpy(&zero_bits, &zero_value, sizeof(zero_bits));
  EXPECT_EQ(zero_bits, 0u);

  EXPECT_EQ(schema->GetFunction(20, true), nullptr);
  EXPECT_EQ(schema->GetFunction(20, false), body);
  EXPECT_NE(OpSchemaRegistry::Schema("CastLike", 18, ""), OpSchemaRegistry::Schema("CastLike", 20, ""));
  for (int requested : {18, 20}) {
    SCOPED_TRACE(requested);
    ModelProto proto;
    proto.set_ir_version(8);
    proto.add_opset_import()->set_version(requested);
    auto* graph = proto.mutable_graph();
    graph->set_name("static_relu");
    M0ControlInput(*graph, "X", TensorProto::FLOAT, {2, 3});
    graph->add_output()->set_name("Y");
    auto* calling = graph->add_node();
    calling->set_op_type("Relu");
    calling->add_input("X");
    calling->add_output("Y");
    Model model(std::move(proto), nullptr, DefaultLoggingManager().DefaultLogger());
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    auto& node = *model.MainGraph().Nodes().begin();
    EXPECT_EQ(node.Op(), schema);
    EXPECT_EQ(node.CanBeInlined(), requested == 18);
    FunctionProto selected;
    EXPECT_EQ(node.TryGetFunctionProto(selected), requested == 18);
    if (requested == 18)
      EXPECT_EQ(selected.SerializeAsString(), body->SerializeAsString());
    else
      EXPECT_EQ(selected.node_size(), 0);
    M0CheckControlShape(model.MainGraph().GetNodeArg("Y")->TypeAsProto(), {2, 3});
  }
}

TEST(M0OrdinaryFunctionRoutesTest, AutomaticModelLocalTemplateAvailabilityAndActualInference) {
  using namespace ONNX_NAMESPACE;
  M0OrdinaryRouteCounts counts;
  ModelProto proto;
  proto.set_ir_version(8);
  proto.add_opset_import()->set_version(20);
  auto* local_opset = proto.add_opset_import();
  local_opset->set_domain("rw07.synthetic.auto.local");
  local_opset->set_version(1);
  auto* function = proto.add_functions();
  function->set_name("TransposeLocal");
  function->set_domain("rw07.synthetic.auto.local");
  function->add_input("X");
  function->add_output("Y");
  function->add_opset_import()->set_version(20);
  auto* leaf = function->add_node();
  leaf->set_op_type("Transpose");
  leaf->add_input("X");
  leaf->add_output("Y");
  auto* perm = leaf->add_attribute();
  perm->set_name("perm");
  perm->set_type(AttributeProto::INTS);
  perm->add_ints(1);
  perm->add_ints(0);
  const auto function_id = function_utils::GetFunctionIdentifier(function->domain(), function->name());
  const auto original_proto = function->SerializeAsString();
  auto* graph = proto.mutable_graph();
  graph->set_name("automatic_local_template");
  M0ControlInput(*graph, "X", TensorProto::FLOAT, {2, 3});
  graph->add_output()->set_name("Y");
  auto* calling = graph->add_node();
  calling->set_op_type(function->name());
  calling->set_domain(function->domain());
  calling->add_input("X");
  calling->add_output("Y");
  Model model(std::move(proto), nullptr, DefaultLoggingManager().DefaultLogger());
  const auto& templates = model.GetModelLocalFunctionTemplates();
  ASSERT_EQ(templates.size(), 1u);
  const auto found = templates.find(function_id);
  ASSERT_NE(found, templates.end());
  ASSERT_NE(found->second, nullptr);
  auto* schema = found->second->op_schema_.get();
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(found->second->onnx_func_proto_, nullptr);
  EXPECT_EQ(found->second->onnx_func_proto_->SerializeAsString(), original_proto);
  auto original_callback = schema->GetTypeAndShapeInferenceFunction();
  ASSERT_TRUE(static_cast<bool>(original_callback));
  schema->TypeAndShapeInferenceFunction([original_callback, &counts](InferenceContext& context) {
    ++counts.dispatch;
    M0CountingInferenceContext forwarding(context, counts);
    original_callback(forwarding);
  });
  EXPECT_EQ(counts.dispatch, 0);
  const auto* before_type = model.MainGraph().GetNodeArg("Y")->TypeAsProto();
  EXPECT_TRUE(before_type == nullptr || !before_type->tensor_type().has_shape());
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_GT(counts.dispatch, 0);
  EXPECT_GT(counts.output_requests, 0);
  auto& node = *model.MainGraph().Nodes().begin();
  EXPECT_EQ(node.Op(), schema);
  ASSERT_NE(node.GetContainingGraph(), nullptr);
  EXPECT_EQ(&node.GetContainingGraph()->GetModel(), &model);
  EXPECT_TRUE(node.CanBeInlined());
  FunctionProto body;
  ASSERT_TRUE(node.TryGetFunctionProto(body));
  EXPECT_EQ(body.SerializeAsString(), original_proto);
  M0CheckControlShape(model.MainGraph().GetNodeArg("Y")->TypeAsProto(), {3, 2});
  const int before = counts.dispatch;
  ASSERT_FALSE(model.MainGraph().GraphResolveNeeded());
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  EXPECT_EQ(counts.dispatch, before);
}

#endif

}  // namespace onnxruntime::test

// Metadata-only qualification candidate; no owner or callback invocation.
// Coordinator appends this complete guarded block outside existing namespaces.
#if !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <system_error>
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>
#include "core/common/inlined_containers.h"
#include "core/graph/schema_registry.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "onnx/defs/shape_inference.h"

namespace onnxruntime::test {
namespace {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;

bool MetadataUnsignedLess(const std::string& a, const std::string& b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                      [](char x, char y) { return static_cast<unsigned char>(x) < static_cast<unsigned char>(y); });
}

std::string MetadataNativeBytes(const google::protobuf::MessageLite& message) {
  std::string bytes;
  {
    google::protobuf::io::StringOutputStream raw(&bytes);
    google::protobuf::io::CodedOutputStream coded(&raw);
    coded.SetSerializationDeterministic(true);
    ORT_ENFORCE(message.SerializeToCodedStream(&coded), "metadata protobuf serialization failed");
    ORT_ENFORCE(!coded.HadError(), "metadata protobuf stream failure");
  }
  return bytes;
}

struct MetadataFormal {
  std::string name, type, description;
  int option, min_arity, differentiation;
  bool homogeneous;
  InlinedVector<std::string> allowed_types;
};

MetadataFormal MetadataReadFormal(const OpSchema::FormalParameter& f) {
  MetadataFormal r{f.GetName(), f.GetTypeStr(), f.GetDescription(), static_cast<int>(f.GetOption()), f.GetMinArity(), static_cast<int>(f.GetDifferentiationCategory()), f.GetIsHomogeneous(), {}};
  for (const auto* type : f.GetTypes()) {
    ORT_ENFORCE(type != nullptr, "null formal type");
    r.allowed_types.push_back(*type);
  }
  std::sort(r.allowed_types.begin(), r.allowed_types.end(), MetadataUnsignedLess);
  return r;
}

struct MetadataAttribute {
  std::string key, name, description, default_wire;
  int type;
  bool required;
};
struct MetadataConstraint {
  std::string parameter, description;
  InlinedVector<std::string> allowed_types;
};
struct MetadataSchema {
  std::string name, domain, file;
  int since, line, support, min_input, max_input, min_output, max_output;
  bool deprecated, descriptor, data, function, context_function;
  InlinedVector<int> function_versions, context_versions;
  InlinedVector<MetadataFormal> inputs, outputs;
  InlinedVector<MetadataAttribute> attributes;
  InlinedVector<MetadataConstraint> declared_constraints, finalized_constraints;
};

MetadataSchema MetadataReadSchema(const OpSchema& schema) {
  MetadataSchema r{};
  r.name = schema.Name();
  r.domain = schema.domain();
  r.file = schema.file();
  r.since = schema.since_version();
  r.line = schema.line();
  ORT_ENFORCE(r.since == schema.SinceVersion(), "schema version getters differ");
  r.support = static_cast<int>(schema.support_level());
  r.min_input = schema.min_input();
  r.max_input = schema.max_input();
  r.min_output = schema.min_output();
  r.max_output = schema.max_output();
  r.deprecated = schema.deprecated();
  r.descriptor = schema.has_type_and_shape_inference_function();
  r.data = schema.has_data_propagation_function();
  r.function = schema.HasFunction();
  r.context_function = schema.HasContextDependentFunction();
  const auto static_versions = schema.function_opset_versions();
  const auto context_versions = schema.context_dependent_function_opset_versions();
  r.function_versions.assign(static_versions.begin(), static_versions.end());
  r.context_versions.assign(context_versions.begin(), context_versions.end());
  std::sort(r.function_versions.begin(), r.function_versions.end());
  std::sort(r.context_versions.begin(), r.context_versions.end());
  for (const auto& f : schema.inputs()) r.inputs.push_back(MetadataReadFormal(f));
  for (const auto& f : schema.outputs()) r.outputs.push_back(MetadataReadFormal(f));
  for (const auto& [key, a] : schema.attributes()) {
    ORT_ENFORCE(key == a.name, "attribute key/name differ");
    r.attributes.push_back({key, a.name, a.description, MetadataNativeBytes(a.default_value),
                            static_cast<int>(a.type), a.required});
  }
  std::sort(r.attributes.begin(), r.attributes.end(), [](const auto& a, const auto& b) {
    return MetadataUnsignedLess(a.key, b.key);
  });
  for (const auto& c : schema.typeConstraintParams()) {
    MetadataConstraint row{c.type_param_str, c.description, {}};
    row.allowed_types.assign(c.allowed_type_strs.begin(), c.allowed_type_strs.end());
    std::sort(row.allowed_types.begin(), row.allowed_types.end(), MetadataUnsignedLess);
    ORT_ENFORCE(std::adjacent_find(row.allowed_types.begin(), row.allowed_types.end()) == row.allowed_types.end(),
                "duplicate declared constraint type");
    r.declared_constraints.push_back(std::move(row));
  }
  for (const auto& [key, value] : schema.typeConstraintMap()) {
    MetadataConstraint row{key, value.second, {}};
    for (const auto* type : value.first) {
      ORT_ENFORCE(type != nullptr, "null constraint type");
      row.allowed_types.push_back(*type);
    }
    std::sort(row.allowed_types.begin(), row.allowed_types.end(), MetadataUnsignedLess);
    r.finalized_constraints.push_back(std::move(row));
  }
  auto by_parameter = [](const auto& a, const auto& b) { return MetadataUnsignedLess(a.parameter, b.parameter); };
  std::sort(r.declared_constraints.begin(), r.declared_constraints.end(), by_parameter);
  std::sort(r.finalized_constraints.begin(), r.finalized_constraints.end(), by_parameter);
  for (size_t i = 1; i < r.declared_constraints.size(); ++i) {
    ORT_ENFORCE(r.declared_constraints[i - 1].parameter != r.declared_constraints[i].parameter,
                "duplicate declared type parameter");
  }
  return r;
}

struct MetadataCapture {
  MetadataSchema donor, identity, nested;
  InlinedVector<std::pair<std::string, int>> local_latest, merged_latest;
  int identity_earliest, nested_earliest, query_max;
  bool selection_stable;
  std::string empty_type_wire;
  int empty_type_value_case;
  bool empty_type_tensor_present, empty_type_denotation_present;
};

// No arguments, Model/Graph construction, owner token, callback call or registry return.
MetadataCapture MetadataFixedCapture() {
  constexpr const char* domain = "rw07.synthetic.descriptor.contract1";
  const auto* donor = ONNX_NAMESPACE::OpSchemaRegistry::Schema("Identity", 19, "");
  ORT_ENFORCE(donor != nullptr && donor->Name() == "Identity" && donor->domain().empty() &&
                  donor->SinceVersion() == 19 && donor->has_type_and_shape_inference_function() &&
                  !donor->has_data_propagation_function() && !donor->HasFunction() &&
                  !donor->HasContextDependentFunction(),
              "fixed Identity19 donor differs");
  MetadataCapture r{};
  r.donor = MetadataReadSchema(*donor);
  auto local = m0_internal::FixedDescriptorRegistryV1();
  SchemaRegistryManager manager;
  manager.RegisterRegistry(local);
  const OpSchema* selected_identity = nullptr;
  const OpSchema* selected_nested = nullptr;
  r.query_max = 1;
  manager.GetSchemaAndHistory("DIdentity", 1, domain, &selected_identity, &r.identity_earliest);
  manager.GetSchemaAndHistory("DNested", 1, domain, &selected_nested, &r.nested_earliest);
  ORT_ENFORCE(selected_identity && selected_nested, "private schemas missing");
  r.identity = MetadataReadSchema(*selected_identity);
  r.nested = MetadataReadSchema(*selected_nested);
  auto copy_latest = [](const DomainToVersionMap& map) {
    InlinedVector<std::pair<std::string, int>> result(map.begin(), map.end());
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
      return MetadataUnsignedLess(a.first, b.first);
    });
    return result;
  };
  r.local_latest = copy_latest(local->GetLatestOpsetVersions(false));
  r.merged_latest = copy_latest(manager.GetLatestOpsetVersions(false));
  r.selection_stable = manager.GetSchema("DIdentity", 1, domain) == selected_identity &&
                       manager.GetSchema("DNested", 1, domain) == selected_nested;
  ONNX_NAMESPACE::TypeProto empty_type;
  r.empty_type_wire = MetadataNativeBytes(empty_type);
  r.empty_type_value_case = static_cast<int>(empty_type.value_case());
  r.empty_type_tensor_present = empty_type.has_tensor_type();
  r.empty_type_denotation_present = empty_type.has_denotation();
  ORT_ENFORCE(empty_type.value_case() == ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET &&
                  !r.empty_type_tensor_present && !r.empty_type_denotation_present,
              "default TypeProto fields populated");
  ORT_ENFORCE(r.selection_stable && r.empty_type_wire.empty(), "metadata capture inconsistency");
  return r;
}

void MetadataRequireUtf8(const std::string& text) {
  size_t index = 0;
  while (index < text.size()) {
    const auto byte = static_cast<unsigned char>(text[index]);
    if (byte < 0x80) {
      ++index;
      continue;
    }
    const size_t count = byte >= 0xc2 && byte <= 0xdf ? 2 : byte >= 0xe0 && byte <= 0xef ? 3
                                                        : byte >= 0xf0 && byte <= 0xf4   ? 4
                                                                                         : 0;
    ORT_ENFORCE(count != 0 && count <= text.size() - index, "metadata text is not UTF8");
    const auto second = static_cast<unsigned char>(text[index + 1]);
    ORT_ENFORCE(!(byte == 0xe0 && second < 0xa0) && !(byte == 0xed && second >= 0xa0) &&
                    !(byte == 0xf0 && second < 0x90) && !(byte == 0xf4 && second >= 0x90),
                "invalid UTF8 scalar");
    for (size_t offset = 1; offset < count; ++offset) {
      const auto continuation = static_cast<unsigned char>(text[index + offset]);
      ORT_ENFORCE(continuation >= 0x80 && continuation <= 0xbf, "invalid UTF8 continuation");
    }
    index += count;
  }
}

std::string MetadataHex(const std::string& bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char byte : bytes) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}

std::string MetadataJsonString(const std::string& text) {
  MetadataRequireUtf8(text);
  constexpr char digits[] = "0123456789abcdef";
  std::string result{"\""};
  for (unsigned char byte : text) {
    if (byte == '"' || byte == '\\') {
      result.push_back('\\');
      result.push_back(static_cast<char>(byte));
    } else if (byte < 0x20) {
      result += "\\u00";
      result.push_back(digits[byte >> 4]);
      result.push_back(digits[byte & 15]);
    } else {
      result.push_back(static_cast<char>(byte));
    }
  }
  result.push_back('"');
  return result;
}

class MetadataJsonObject final {
 public:
  MetadataJsonObject() = default;
  void String(const char* key, const std::string& value) { Raw(key, MetadataJsonString(value)); }
  void Int(const char* key, int value) { Raw(key, std::to_string(value)); }
  void Bool(const char* key, bool value) { Raw(key, value ? "true" : "false"); }
  void Raw(const char* key, const std::string& encoded) {
    if (!fields_.empty()) fields_ += ',';
    fields_ += MetadataJsonString(key) + ':' + encoded;
  }
  std::string Finish() const { return '{' + fields_ + '}'; }

 private:
  std::string fields_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MetadataJsonObject);
};

template <typename Range, typename Encode>
std::string MetadataJsonArray(const Range& values, Encode encode) {
  std::string result{"["};
  bool first = true;
  for (const auto& value : values) {
    if (!first) result += ',';
    first = false;
    result += encode(value);
  }
  result += ']';
  return result;
}

std::string MetadataStrings(const InlinedVector<std::string>& values) {
  return MetadataJsonArray(values, MetadataJsonString);
}

std::string MetadataFormalJson(const MetadataFormal& row) {
  MetadataJsonObject result;
  result.String("name", row.name);
  result.String("type", row.type);
  result.String("description", row.description);
  result.Int("option", row.option);
  result.Int("min_arity", row.min_arity);
  result.Int("differentiation", row.differentiation);
  result.Bool("homogeneous", row.homogeneous);
  result.Raw("allowed_types", MetadataStrings(row.allowed_types));
  return result.Finish();
}

std::string MetadataConstraintJson(const MetadataConstraint& row) {
  MetadataJsonObject result;
  result.String("parameter", row.parameter);
  result.String("description", row.description);
  result.Raw("allowed_types", MetadataStrings(row.allowed_types));
  return result.Finish();
}

std::string MetadataSchemaJson(const MetadataSchema& row) {
  MetadataJsonObject result;
  result.String("name", row.name);
  result.String("domain", row.domain);
  result.String("file", row.file);
  result.Int("since", row.since);
  result.Int("line", row.line);
  result.Int("support", row.support);
  result.Int("min_input", row.min_input);
  result.Int("max_input", row.max_input);
  result.Int("min_output", row.min_output);
  result.Int("max_output", row.max_output);
  result.Bool("deprecated", row.deprecated);
  result.Bool("type_shape_inference_present", row.descriptor);
  result.Bool("data_propagation_present", row.data);
  result.Bool("static_function_present", row.function);
  result.Bool("context_function_present", row.context_function);
  auto integer = [](int value) { return std::to_string(value); };
  result.Raw("static_function_versions", MetadataJsonArray(row.function_versions, integer));
  result.Raw("context_function_versions", MetadataJsonArray(row.context_versions, integer));
  result.Raw("inputs", MetadataJsonArray(row.inputs, MetadataFormalJson));
  result.Raw("outputs", MetadataJsonArray(row.outputs, MetadataFormalJson));
  result.Raw("attributes", MetadataJsonArray(row.attributes, [](const MetadataAttribute& attr) {
               MetadataJsonObject encoded;
               encoded.String("key", attr.key);
               encoded.String("name", attr.name);
               encoded.String("description", attr.description);
               encoded.Int("type", attr.type);
               encoded.Bool("required", attr.required);
               encoded.String("default_wire_hex", MetadataHex(attr.default_wire));
               ORT_ENFORCE(attr.default_wire.size() <= 1048576, "metadata default exceeds fixed export bound");
               encoded.Int("default_wire_length", static_cast<int>(attr.default_wire.size()));
               return encoded.Finish();
             }));
  result.Raw("declared_constraints", MetadataJsonArray(row.declared_constraints, MetadataConstraintJson));
  result.Raw("finalized_constraints", MetadataJsonArray(row.finalized_constraints, MetadataConstraintJson));
  return result.Finish();
}

std::string MetadataLatestJson(const InlinedVector<std::pair<std::string, int>>& rows) {
  return MetadataJsonArray(rows, [](const auto& row) {
    MetadataJsonObject result;
    result.String("domain", row.first);
    result.Int("version", row.second);
    return result.Finish();
  });
}

#define M0_METADATA_STRINGIFY_INNER(value) #value
#define M0_METADATA_STRINGIFY(value) M0_METADATA_STRINGIFY_INNER(value)
std::string MetadataConfigJson() {
  MetadataJsonObject result;
  result.String("onnx_namespace", M0_METADATA_STRINGIFY(ONNX_NAMESPACE));
#ifdef ONNX_ML
  result.Bool("onnx_ml", true);
#else
  result.Bool("onnx_ml", false);
#endif
#ifdef ONNX_USE_LITE_PROTO
  result.Bool("onnx_lite_proto", true);
#else
  result.Bool("onnx_lite_proto", false);
#endif
#ifdef __ONNX_NO_DOC_STRINGS
  result.Bool("onnx_no_doc_strings", true);
#else
  result.Bool("onnx_no_doc_strings", false);
#endif
#ifdef USE_WEBGPU
  result.Bool("webgpu", true);
#else
  result.Bool("webgpu", false);
#endif
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  result.Bool("exceptions", true);
#else
  result.Bool("exceptions", false);
#endif
  result.String("cplusplus", std::to_string(__cplusplus));
#ifdef GOOGLE_PROTOBUF_VERSION
  result.Int("protobuf_header_version", GOOGLE_PROTOBUF_VERSION);
#endif
  result.String("source_file", __FILE__);
  result.String("archive_source_config_binding", "EXTERNAL_COORDINATOR_FREEZE_REQUIRED");
  return result.Finish();
}
#undef M0_METADATA_STRINGIFY
#undef M0_METADATA_STRINGIFY_INNER

std::string MetadataExportJson(const MetadataCapture& capture) {
  MetadataJsonObject result;
  result.Int("format_version", 1);
  result.String("artifact_status", "METADATA_ONLY_NO_OWNER_PERMISSION");
  result.Bool("admission", false);
  result.Bool("owner_permission", false);
  result.String("model_parse_graph_resolve_inference_numerics", "NOT_PERFORMED_BY_CAPTURE");
  result.String("global_registry", "BUILTIN_METADATA_INITIALIZATION_ONLY_NO_REGISTRATION_REPLACEMENT");
  result.Raw("donor", MetadataSchemaJson(capture.donor));
  result.Raw("identity", MetadataSchemaJson(capture.identity));
  result.Raw("nested", MetadataSchemaJson(capture.nested));
  result.Raw("local_latest", MetadataLatestJson(capture.local_latest));
  result.Raw("merged_latest", MetadataLatestJson(capture.merged_latest));
  MetadataJsonObject selection;
  selection.Int("query_max_inclusive", capture.query_max);
  selection.Int("identity_earliest_unchanged", capture.identity_earliest);
  selection.Int("nested_earliest_unchanged", capture.nested_earliest);
  selection.Bool("same_process_pointer_selection_stable", capture.selection_stable);
  selection.String("schema_end_version", "NO_PUBLIC_GETTER_NO_VALUE_INFERRED");
  selection.String("pointer_scope", "SAME_CAPTURE_ONLY_NOT_PORTABLE_CALLBACK_IDENTITY");
  result.Raw("selection", selection.Finish());
  MetadataJsonObject empty;
  empty.String("wire_hex", MetadataHex(capture.empty_type_wire));
  empty.Int("wire_length", static_cast<int>(capture.empty_type_wire.size()));
  empty.Int("value_case", capture.empty_type_value_case);
  empty.Bool("tensor_type_present", capture.empty_type_tensor_present);
  empty.Bool("denotation_present", capture.empty_type_denotation_present);
  empty.String("origin", "DEFAULT_CONSTRUCTED_NATIVE_TYPEPROTO_NOT_RUNTIME_DP_OBSERVATION");
  empty.Int("fixed_recipe_output_count_expected", 1);
  result.Raw("empty_type", empty.Finish());
  result.Raw("compile_flags", MetadataConfigJson());
  result.String("host_fixture_compatibility", "NOT_CHECKED_BY_CAPTURE_REQUIRES_INDEPENDENT_COMPARISON");
  result.String("live_graph_phase_receipts", "NOT_CAPTURED");
  result.String("normalized_model_imports", "SOURCE_DERIVED_EXPECTATION_FROM_MERGED_LATEST_NOT_MODEL_OBSERVATION");
  return result.Finish() + '\n';
}
}  // namespace

TEST(M0DescriptorMetadataTest, CapturePinnedSchemaAndImportMetadata) {
  ASSERT_EQ(MetadataJsonString(std::string("\0\1\b\t\n\f\r\x1f\"\\", 10)),
            "\"\\u0000\\u0001\\u0008\\u0009\\u000a\\u000c\\u000d\\u001f\\\"\\\\\"");
  ASSERT_EQ(MetadataJsonString(std::string("\xc3\xa9", 2)), std::string("\"\xc3\xa9\"", 4));
  EXPECT_THROW(MetadataJsonString(std::string("\xc0\x80", 2)), std::exception);
  const auto capture = MetadataFixedCapture();
  ASSERT_TRUE(capture.selection_stable);
  ASSERT_EQ(capture.query_max, 1);
  ASSERT_EQ(capture.identity.name, "DIdentity");
  ASSERT_EQ(capture.nested.name, "DNested");
  ASSERT_EQ(capture.local_latest.size(), 1u);
  ASSERT_EQ(capture.local_latest[0].first, "rw07.synthetic.descriptor.contract1");
  ASSERT_EQ(capture.local_latest[0].second, 1);
  ASSERT_FALSE(capture.merged_latest.empty());
  ASSERT_TRUE(capture.empty_type_wire.empty());
  ASSERT_EQ(capture.identity.inputs.size(), 1u);
  ASSERT_EQ(capture.identity.outputs.size(), 1u);
  ASSERT_EQ(capture.nested.inputs.size(), 1u);
  ASSERT_EQ(capture.nested.outputs.size(), 1u);
  const std::string bytes = MetadataExportJson(capture);
  ASSERT_LE(bytes.size(), 1048576u);
  constexpr const char* output = "rw07_descriptor_metadata_v1.json";
  std::error_code filesystem_error;
  const bool exists = std::filesystem::exists(output, filesystem_error);
  ASSERT_FALSE(filesystem_error) << filesystem_error.message();
  ASSERT_FALSE(exists) << "metadata output must use a new coordinator-owned stage directory";
  std::ofstream stream(output, std::ios::binary | std::ios::out | std::ios::trunc);
  ASSERT_TRUE(stream.is_open());
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  stream.flush();
  ASSERT_TRUE(stream.good());
  stream.close();
  ASSERT_FALSE(stream.fail());
}

namespace {
template <typename Proto>
void CheckRw07NativeRoundtrip(const m0_internal::Rw07FixedFixtureBinding& binding) {
  const auto original = m0_internal::CheckedFixedBytes(binding.id);
  Proto proto;
  ASSERT_TRUE(proto.ParseFromString(original)) << binding.id;
  ASSERT_EQ(proto.SerializeAsString(), original) << binding.id;
  const auto canonical = m0_internal::FixedCanonicalWire(proto.SerializeAsString(), binding.message_id);
  ASSERT_EQ(canonical, m0_internal::FixedBytes(binding.canonical, binding.canonical_size)) << binding.id;
  Proto reparsed;
  ASSERT_TRUE(reparsed.ParseFromString(canonical)) << binding.id;
  ASSERT_EQ(reparsed.SerializeAsString(), canonical) << binding.id;
}
}  // namespace

TEST(M0FixedNativeBindingTest, All36OriginalAndCanonicalMessagesMatchPinnedNativeProtobuf) {
  size_t count = 0;
  for (const auto& binding : m0_internal::kRw07FixedFixtureBindings) {
    const std::string_view name = m0_internal::kRw07KnownWireMessages[binding.message_id].name;
    if (name == "onnx.ModelProto")
      CheckRw07NativeRoundtrip<ONNX_NAMESPACE::ModelProto>(binding);
    else if (name == "onnx.GraphProto")
      CheckRw07NativeRoundtrip<ONNX_NAMESPACE::GraphProto>(binding);
    else if (name == "onnx.NodeProto")
      CheckRw07NativeRoundtrip<ONNX_NAMESPACE::NodeProto>(binding);
    else if (name == "onnx.FunctionProto")
      CheckRw07NativeRoundtrip<ONNX_NAMESPACE::FunctionProto>(binding);
    else if (name == "onnx.AttributeProto")
      CheckRw07NativeRoundtrip<ONNX_NAMESPACE::AttributeProto>(binding);
    else if (name == "onnx.TypeProto")
      CheckRw07NativeRoundtrip<ONNX_NAMESPACE::TypeProto>(binding);
    else
      FAIL() << "unbound fixture type " << name;
    ++count;
  }
  ASSERT_EQ(count, 36u);
  ONNX_NAMESPACE::TypeProto empty;
  ASSERT_TRUE(empty.ParseFromString(m0_internal::CheckedFixedBytes("type.dp-default-empty")));
  ASSERT_EQ(empty.value_case(), ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET);
  ASSERT_FALSE(empty.has_denotation());
  ASSERT_TRUE(empty.SerializeAsString().empty());
}

TEST(M0FixedNativeBindingTest, RejectsCollapsedWireAndKeyAmbiguityBeforeNativeParse) {
  using namespace m0_internal;
  const auto& marker = FixedBinding("marker7");
  auto unknown = CheckedFixedBytes(marker.id);
  AppendFixedVarint(unknown, uint64_t{999} << 3);
  unknown.push_back('\0');
  EXPECT_THROW(FixedCanonicalWire(unknown, marker.message_id), std::exception);
  const auto marker_bytes = CheckedFixedBytes(marker.id);
  EXPECT_THROW(FixedCanonicalWire(marker_bytes + marker_bytes, marker.message_id), std::exception);
  EXPECT_THROW(FixedCanonicalWire(std::string("\x80\x00", 2), marker.message_id), std::exception);
  EXPECT_THROW(FixedCanonicalWire(std::string("\x0a\xff", 2), marker.message_id), std::exception);
  EXPECT_THROW(FixedCanonicalWire(std::string("\x0b", 1), marker.message_id), std::exception);
  const auto& type = FixedBinding("type.dp-default-empty");
  EXPECT_THROW(FixedCanonicalWire(std::string("\x0a\x00\x22\x00", 4), type.message_id), std::exception);
  ONNX_NAMESPACE::ModelProto model;
  const auto& model_binding = FixedBinding("nested-depth2.model-original");
  ASSERT_TRUE(model.ParseFromString(CheckedFixedBytes(model_binding.id)));
  auto* attrs = model.mutable_graph()->mutable_node(0)->mutable_attribute();
  auto duplicate = attrs->Get(0);
  attrs->Add()->CopyFrom(duplicate);
  EXPECT_THROW(FixedCanonicalWire(model.SerializeAsString(), model_binding.message_id), std::exception);
  ASSERT_TRUE(model.ParseFromString(CheckedFixedBytes(model_binding.id)));
  auto import = model.opset_import(0);
  model.add_opset_import()->CopyFrom(import);
  EXPECT_THROW(FixedCanonicalWire(model.SerializeAsString(), model_binding.message_id), std::exception);
  ASSERT_TRUE(model.ParseFromString(CheckedFixedBytes(model_binding.id)));
  model.add_opset_import()->set_domain("");
  model.mutable_opset_import(1)->set_version(1);
  model.add_opset_import()->set_domain("ai.onnx");
  model.mutable_opset_import(2)->set_version(1);
  EXPECT_THROW(FixedCanonicalWire(model.SerializeAsString(), model_binding.message_id), std::exception);
}

#if !defined(ORT_NO_EXCEPTIONS)
TEST(M0SyntheticOwnerFoundationTest, FixedPrivateFactoryTracksScopesUnwindAndClosedGenerationRoutes) {
  const auto receipt = M0RunFixedSyntheticOwnerFoundationV1();
  EXPECT_EQ(receipt.registered, 4u);
  EXPECT_EQ(receipt.unregistered, 4u);
  EXPECT_EQ(receipt.live_at_capture, 0u);
  EXPECT_EQ(receipt.full_validations, 2u);
  EXPECT_EQ(receipt.generation_acquisitions, 2u);
  EXPECT_EQ(receipt.frame_pushes, 3u);
  EXPECT_EQ(receipt.frame_pops, 3u);
  EXPECT_EQ(receipt.resolve_denials, 1u);
  EXPECT_EQ(receipt.callback_denials, 1u);
  EXPECT_TRUE(receipt.constructor_failure_unwound);
  EXPECT_TRUE(receipt.parentless_inherits_normalized_imports);
  EXPECT_TRUE(receipt.inherited_empty_list_denied);
  EXPECT_TRUE(receipt.clean_child_denied);
  EXPECT_TRUE(receipt.foreign_scope_denied);
  EXPECT_TRUE(receipt.conflict_denied_before_graph_read);
  EXPECT_TRUE(receipt.same_frame_reentry_denied);
  EXPECT_TRUE(receipt.poisoned);
  EXPECT_FALSE(receipt.descriptor_permission);
}
TEST(M0SyntheticProjectionTest, PrivateMechanicalPhasesPreserveBodyMapAndDenyLiveMutations) {
  const auto receipt = M0RunFixedSyntheticProjectionV1();
  EXPECT_EQ(receipt.checked_phases, 8u);
  EXPECT_EQ(receipt.checked_scopes, 32u);
  EXPECT_EQ(receipt.rejected_mutations, 20u);
  EXPECT_EQ(receipt.retired_owners, 22u);
  EXPECT_TRUE(receipt.map_body_preserved);
  EXPECT_TRUE(receipt.failed_registration_preserved);
  EXPECT_TRUE(receipt.phase_mismatch_poisoned);
  EXPECT_FALSE(receipt.descriptor_permission);
}
TEST(M0SyntheticPreparationTest, OriginalBackingLineageAndActualSourceTopologyStopBeforeCallbacks) {
  const auto receipt = M0RunFixedSyntheticPreparationV1();
  EXPECT_EQ(receipt.prepared_roots, 4u);
  EXPECT_EQ(receipt.initialized_scopes, 10u);
  EXPECT_EQ(receipt.connected_scopes, 10u);
  EXPECT_EQ(receipt.sorted_scopes, 10u);
  EXPECT_EQ(receipt.redirected_requests, 2u);
  EXPECT_EQ(receipt.rejected_backing_mutations, 20u);
  EXPECT_EQ(receipt.retired_owners, 24u);
  EXPECT_FALSE(receipt.descriptor_permission);
}
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
TEST(M0SyntheticFinalizedTest, FixedFinalizationCleanCachedEqual7AndParentlessPreserveExactState) {
  const auto receipt = M0RunFixedSyntheticFinalizedV1();
  EXPECT_EQ(receipt.native_bindings, 2u);
  EXPECT_EQ(receipt.finalized_scopes, 8u);
  EXPECT_EQ(receipt.finalization_steps, 40u);
  EXPECT_EQ(receipt.clean_repeats, 3u);
  EXPECT_EQ(receipt.cached_callbacks, 3u);
  EXPECT_EQ(receipt.present_defaults, 4u);
  EXPECT_EQ(receipt.parentless_callbacks, 1u);
  EXPECT_EQ(receipt.equal7_callbacks, 1u);
  EXPECT_EQ(receipt.rejected_mutations, 10u);
  EXPECT_EQ(receipt.retired_owners, 13u);
  EXPECT_FALSE(receipt.descriptor_permission);
}
TEST(M0SyntheticIssuedTest, PrivateActualNestedCallbacksJoinDefaultsScratchAndNoWorkBeforePublication) {
  const auto receipt = M0RunFixedSyntheticIssuedV1();
  EXPECT_EQ(receipt.source_checks, 3u);
  EXPECT_EQ(receipt.source_selections, 3u);
  EXPECT_EQ(receipt.source_defaults, 3u);
  EXPECT_EQ(receipt.callbacks, 3u);
  EXPECT_EQ(receipt.child_joins, 2u);
  EXPECT_EQ(receipt.propagation_no_work, 3u);
  EXPECT_EQ(receipt.saver_no_work, 3u);
  EXPECT_EQ(receipt.live_publications, 3u);
  EXPECT_EQ(receipt.child_skips, 2u);
  EXPECT_EQ(receipt.retired_owners, 14u);
  EXPECT_TRUE(receipt.fixed_post_perform_completed);
  EXPECT_TRUE(receipt.meaningful_nested_failure);
  EXPECT_TRUE(receipt.callback_conflict_denied);
  EXPECT_TRUE(receipt.payload_denied);
  EXPECT_TRUE(receipt.sparse_denied);
  EXPECT_TRUE(receipt.symbolic_denied);
  EXPECT_TRUE(receipt.attribute_denied);
  EXPECT_TRUE(receipt.function_denied);
  EXPECT_TRUE(receipt.propagation_read_denied);
  EXPECT_TRUE(receipt.propagation_write_denied);
  EXPECT_TRUE(receipt.malformed_propagation_denied);
  EXPECT_TRUE(receipt.engaged_inventory_denied);
  EXPECT_TRUE(receipt.repeated_body_denied);
  EXPECT_TRUE(receipt.direct_perform_denied);
  EXPECT_FALSE(receipt.descriptor_permission);
}
#endif
#endif  // !ORT_NO_EXCEPTIONS
}  // namespace onnxruntime::test
#endif  // !ORT_MINIMAL_BUILD
