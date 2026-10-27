// RW07 source-private admission diagnostics. No public OrtApi extension.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/session_options.h"
#include "core/common/m0_capacity_ceiling_v1.h"

namespace onnxruntime {

inline bool M0FullGpuPreinitRequestedV1(const SessionOptions& options) {
  return options.config_options.GetConfigOrDefault("m0.fullgpu.preinit_admission_v1", "0") != "0";
}

inline Status M0FullGpuReadCapacityCeilingV1(const SessionOptions& options,uint32_t& ceiling){
  const auto& entries=options.config_options.GetConfigOptionsMap();
  const auto found=entries.find(kM0CapacityCeilingConfigV1);
  ceiling=kM0MaximumDurationCapacityV1;
  ORT_RETURN_IF_NOT(found==entries.end()||M0ParseCapacityCeilingV1(found->second,ceiling),"RW07_NATIVE_CAPACITY_CEILING_CONFIG");
  return Status::OK();
}

inline Status M0FullGpuValidatePreinitOptionsV1(const SessionOptions& options) {
  uint32_t ceiling=0;ORT_RETURN_IF_ERROR(M0FullGpuReadCapacityCeilingV1(options,ceiling));
  ORT_RETURN_IF_NOT(options.config_options.GetConfigOrDefault("m0.fullgpu.preinit_admission_v1", "0") == "1" &&
                        options.config_options.GetConfigOrDefault("m0.fullgpu.preinit_abi_v1", "0") == "1",
                    "RW07_PREINIT_DENY_ABI");
  ORT_RETURN_IF_NOT(options.graph_optimization_level == TransformerLevel::Default &&
                        options.execution_mode == ExecutionMode::ORT_SEQUENTIAL &&
                        options.config_options.GetConfigOrDefault("session.disable_prepacking", "0") == "1" &&
                        options.config_options.GetConfigOrDefault("session.disable_aot_function_inlining", "0") == "1",
                    "RW07_PREINIT_DENY_OPTIONS");
  return Status::OK();
}

// These authorities are outside the closed embedded-model candidate. Checking
// them before ConstructorCommon avoids invoking session-supplied callbacks.
inline Status M0FullGpuValidateIngressAuthoritiesV1(const SessionOptions& options) {
  const auto has_thread_authority = [](const OrtThreadPoolParams& params) {
    bool present = params.custom_create_thread_fn || params.custom_join_thread_fn ||
                   params.custom_thread_creation_options;
#ifdef ORT_ENABLE_SESSION_THREADPOOL_CALLBACKS
    present = present || params.work_callbacks.on_enqueue || params.work_callbacks.on_start_work ||
              params.work_callbacks.on_stop_work || params.work_callbacks.on_abandon;
#endif
    return present;
  };
  ORT_RETURN_IF_NOT(!options.custom_create_thread_fn && !options.custom_join_thread_fn &&
                        !options.custom_thread_creation_options && !options.user_logging_function &&
                        !options.user_logging_param && !has_thread_authority(options.intra_op_param) &&
                        !has_thread_authority(options.inter_op_param) && !options.ep_selection_policy.enable &&
                        !options.ep_selection_policy.delegate && !options.ep_context_data_read_func &&
                        !options.ep_context_data_read_state && !options.has_explicit_ep_context_gen_options,
                    "RW07_PREINIT_DENY_AUTHORITY");
  ORT_RETURN_IF_NOT(options.initializers_to_share_map.empty(), "RW07_PREINIT_DENY_AUTHORITY");
#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
  ORT_RETURN_IF_NOT(options.external_initializers.empty() && options.external_initializer_files_mmap.empty(),
                    "RW07_PREINIT_DENY_AUTHORITY");
#endif
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  ORT_RETURN_IF_NOT(!options.custom_op_libs, "RW07_PREINIT_DENY_AUTHORITY");
#endif
  return Status::OK();
}

inline Status M0FullGpuCheckPreloadV1(const SessionOptions& options) {
  if (!M0FullGpuPreinitRequestedV1(options)) return Status::OK();
  ORT_RETURN_IF_ERROR(M0FullGpuValidatePreinitOptionsV1(options));
  ORT_RETURN_IF_ERROR(M0FullGpuValidateIngressAuthoritiesV1(options));
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT");
}

struct M0FullGpuPreinitSlotV1 {
  std::string name;
  std::string type_proto;
  bool present = false;
  int value_index = -1;
  int effective_memory_type = 0;
  bool effective_memory_available = false;
};

// Observed registry definition, never constructor/Compute/error qualification.
struct M0FullGpuPreinitKernelV1 {
  const void* selected_create_info = nullptr;
  const void* selected_definition = nullptr;
  std::string op_name, domain, provider;
  int version_start = 0, version_end = 0, execution_queue = 0;
  int default_input_memory = 0, default_output_memory = 0;
  bool contiguous_inputs = false, external_outputs = false;
  std::vector<std::pair<std::string, std::vector<std::string>>> type_constraints;
  std::vector<std::pair<int, int>> may_inplace, aliases;
  bool has_variadic_alias = false;
  std::pair<int, int> variadic_alias{0, 0};
  std::vector<std::pair<size_t, int>> input_memory, output_memory;
#ifdef ENABLE_STRIDED_TENSORS
  std::vector<int> may_strided_inputs;
  std::vector<std::pair<int, int>> may_strided_outputs;
#endif
};

struct M0FullGpuPreinitNodeV1 {
  size_t index = 0;
  int since_version = 0;
  std::vector<int> input_arg_counts;
  std::string node_proto;
  std::string provider;
  std::string implementation_id;
  M0FullGpuPreinitKernelV1 kernel;
  std::vector<M0FullGpuPreinitSlotV1> implicit_inputs;
  std::vector<M0FullGpuPreinitSlotV1> inputs;
  std::vector<M0FullGpuPreinitSlotV1> outputs;
};

struct M0FullGpuPreinitResourceV1 {
  int value_index = 0;
  int allocation_kind = 0;
  int reused_buffer = 0;
  int device_type = 0;
  int memory_type = 0;
  int device_id = 0;
  uint32_t vendor_id = 0;
  std::string value_name, value_type;
  std::vector<size_t> lifetime_starts, lifetime_ends;
#ifdef ENABLE_STRIDED_TENSORS
  bool is_strided_tensor = false;
#endif
};

// Initializer identity is observed without copying the large tensor payload.
// The pointer is tied to this SessionState lifetime. Metadata is NOT a byte
// digest, upload/init-path certificate or an exemption from role enforcement.
struct M0FullGpuPreinitInitializerV1 {
  std::string name;
  const void* owned_tensor_proto = nullptr;
  int value_index = -1, data_type = 0, data_location = 0;
  uint64_t encoded_bytes = 0, raw_data_bytes = 0;
  std::vector<int64_t> dimensions;
  bool constant = false;
};
struct M0FullGpuPreinitStepV1 {
  size_t stream_index = 0, node_index = 0;
  std::string description;
};

// An observed owned plan, never a certificate. Serialization alone does not
// prove lineage, host-work safety, roles, dynamic extent or initialization.
struct M0FullGpuPreinitSnapshotV1 {
  uint32_t observation_schema = 2;
  uint64_t session_epoch = 0;
  const void* owned_plan = nullptr;
  std::vector<M0FullGpuPreinitNodeV1> nodes;
  std::vector<M0FullGpuPreinitResourceV1> resources;
  std::vector<int> initializer_order, activation_order;
  std::vector<M0FullGpuPreinitInitializerV1> initializers;
  std::vector<M0FullGpuPreinitSlotV1> graph_inputs, graph_outputs;
  std::vector<M0FullGpuPreinitStepV1> execution_steps;
  std::vector<std::pair<size_t, size_t>> value_streams, release_actions;
  std::vector<std::vector<size_t>> node_release_list;
};

}  // namespace onnxruntime
