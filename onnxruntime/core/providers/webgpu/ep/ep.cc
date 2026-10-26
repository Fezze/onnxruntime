// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"
#include <sstream>
#include <fstream>
#include <functional>
#include "core/common/m0_observation_options_v1.h"
#include "core/providers/webgpu/m0_numeric_trace_v1.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/allocator.h"
#include "core/common/m0_ep_profiler_bridge_v1.h"
#include "core/common/m0_first_run_profile_v1.h"

#include "allocator.h"
#include "factory.h"
#include "sync_stream.h"

#include "core/framework/run_options.h"
#include "core/framework/kernel_registry.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

#include "ep/get_capability_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

using onnxruntime::ep::Api;

// Each EP owns its diagnostic buffer; no cross-session tensor or address cache.
// Only unpaired FP32 outputs are covered in V1. Extent-bearing physical carriers
// are explicitly skipped rather than treating inactive padding as audio data.
class M0NumericTraceV1 final {
 public:
  explicit M0NumericTraceV1(WebGpuExecutionProvider& ep)
      : ep_(ep), buffer_(nullptr, [&ep](WGPUBuffer value) {
          if (value) {
            try { ep.BufferManager().Release(value, &ep.Recording()); }
            catch (...) { ep.BufferManager().M0OwnershipV2().Poison(); }
          }
        }) {
    const char* path = std::getenv("M0_GPU_NUMERIC_TRACE_FILE");
    ORT_ENFORCE(path && *path, "M0 numeric trace requires an explicit output file");
    file_.open(path, std::ios::out | std::ios::app);
    ORT_ENFORCE(file_.is_open(), "M0 numeric trace output cannot be opened");
  }
  void Begin(const std::string& session, const std::string& owner_run) {
    session_ = session; owner_run_ = owner_run;
    for (const auto* id : {&session_, &owner_run_}) {
      ORT_ENFORCE(!id->empty() && id->size() <= 20 && id->find_first_not_of("0123456789") == std::string::npos,
                  "M0 numeric trace session/run identity");
    }
    active_ = run_ < 3 && file_.tellp() < std::streampos(32ull * 1024 * 1024);
    started_ = false;
    if (!active_) return;
    buffer_.reset(); rows_.clear(); hash_rows_ = 0; copy_rows_ = 0; omitted_ = 0; ordinal_ = 0;
  }
  Status Observe(ComputeContext& context) {
    if (!active_) return Status::OK();
    if (!started_) {
      // Initializer-transfer scopes use the same callbacks but execute no
      // kernel; they must not consume a numerical Run or produce empty traces.
      started_ = true; ++run_;
      file_ << "{\"event\":\"begin\",\"run\":" << run_
            << ",\"session\":\"" << session_ << "\",\"owner_run\":\"" << owner_run_
            << "\",\"name_prefix_limit_bytes\":256,\"diagnostic_overhead\":true,\"scope\":\"unpaired_fp32_physical_outputs\"}\n";
      file_.flush();
      ORT_RETURN_IF_NOT(file_.good(), "M0 numeric trace begin write failed");
    }
    ++ordinal_;
    const auto outputs = context.KernelContext().AllocatedOutputs();
    for (size_t port = 0; port < outputs.size(); ++port) {
      const auto& tensor = outputs[port];
      if (rows_.size() >= 8192) { ++omitted_; continue; }
      Row row;
      row.ordinal = ordinal_; row.port = port; row.node = Hex(context.NodeName()); row.op = Hex(context.OpType());
      if (!tensor.DataRaw()) { row.skip = "unallocated"; rows_.push_back(std::move(row)); continue; }
      row.shape.assign(tensor.Shape().GetDims().begin(), tensor.Shape().GetDims().end());
      bool paired = false;
      for (const auto& other : outputs) {
        if (!other.DataRaw() || !other.IsDataType<uint32_t>() || other.Shape().Size() != 32) continue;
        M0FullGpuAllocationIdentity identity;
        if (ep_.BufferManager().M0OwnershipV2().DescribePair(
              reinterpret_cast<uintptr_t>(other.DataRaw()), reinterpret_cast<uintptr_t>(tensor.DataRaw()), identity)) {
          paired = true; break;
        }
      }
      const char* skip = M0NumericTraceSkipV1(tensor.Location().device.Type() == OrtDevice::GPU,
                                             tensor.IsDataType<float>(), paired, tensor.Shape().Size(), hash_rows_);
      if (skip) { row.skip = skip; rows_.push_back(std::move(row)); continue; }
      if (!buffer_) {
        // Actual BufferManager admission/ownership/queue accounting remains active.
        buffer_.reset(ep_.BufferManager().Create(ep_.Recording(), kM0NumericTraceBytesV1,
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
      }
      const OrtMemoryInfo memory(WEBGPU_BUFFER, OrtDeviceAllocator, WebGpuDevice(ep_.GetDeviceId()), OrtMemTypeDefault);
      Tensor digest(DataTypeImpl::GetType<uint32_t>(), {kM0NumericTraceRowsV1 * kM0NumericTraceWordsV1}, buffer_.get(), memory);
      M0NumericTraceProgramV1 program;
      program.AddInput({&tensor, ProgramTensorMetadataDependency::None})
          .AddOutput({&digest, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(1)
          .AddUniformVariables({static_cast<uint32_t>(tensor.Shape().Size()), hash_rows_});
      // Uses the actual context so cancellation and native dispatch admission are retained.
      ORT_RETURN_IF_ERROR(context.RunProgram(program));
      row.hash_row = hash_rows_++; rows_.push_back(std::move(row));
      if (context.OpType() == "Transpose" && port == 0 && hash_rows_ < kM0NumericTraceRowsV1 && rows_.size() < 8192) {
        const auto* source = context.KernelContext().Input<Tensor>(0);
        ORT_RETURN_IF_NOT(source && source->IsDataType<float>() && source->Shape().Size() == tensor.Shape().Size(),
                          "M0 transpose diagnostic source extent");
        const auto rank = source->Shape().NumDimensions();
        if (rank == 0 || rank > 8) continue;
        TensorShapeVector permutation;
        if (!context.Kernel().Info().GetAttrs("perm", permutation).IsOK()) {
          for (size_t axis = rank; axis > 0; --axis) permutation.push_back(static_cast<int64_t>(axis-1));
        }
        ORT_RETURN_IF_NOT(permutation.size() == rank, "M0 transpose diagnostic permutation rank");
        std::array<uint32_t,8> shape{}, axes{};
        std::array<bool,8> seen{};
        for (size_t axis = 0; axis < rank; ++axis) {
          ORT_RETURN_IF_NOT(source->Shape()[axis] > 0 && source->Shape()[axis] <= kM0NumericTraceMaxElementsV1 &&
                            permutation[axis] >= 0 && static_cast<uint64_t>(permutation[axis]) < rank,
                            "M0 transpose diagnostic dimensions/permutation");
          shape[axis] = static_cast<uint32_t>(source->Shape()[axis]);
          axes[axis] = static_cast<uint32_t>(permutation[axis]);
          ORT_RETURN_IF_NOT(!seen[axes[axis]], "M0 transpose diagnostic duplicate permutation");
          seen[axes[axis]] = true;
        }
        M0TransposeCopyCheckProgramV1 check;
        check.AddInputs({{source,ProgramTensorMetadataDependency::None}, {&tensor,ProgramTensorMetadataDependency::None}})
            .AddOutput({&digest,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(1)
            .AddUniformVariables({static_cast<uint32_t>(tensor.Shape().Size()), hash_rows_, static_cast<uint32_t>(rank),
                                  gsl::span<const uint32_t>(shape), gsl::span<const uint32_t>(axes)});
        ORT_RETURN_IF_ERROR(context.RunProgram(check));
        Row copy = rows_.back(); copy.copy_check = true; copy.hash_row = hash_rows_++; ++copy_rows_;
        copy.copy_alias = source->DataRaw() == tensor.DataRaw();
        rows_.push_back(std::move(copy));
      }
    }
    return Status::OK();
  }
  void End(bool checked) {
    if (!active_) return;
    active_ = false;
    if (!started_) return;
    std::vector<uint32_t> digests;
    if (checked && hash_rows_) {
      digests.resize(hash_rows_ * kM0NumericTraceWordsV1);
      // Run queue completion has already succeeded. Read only GPU-generated
      // integer summaries; no model tensor payload crosses the CPU boundary.
      ep_.BufferManager().Download(ep_.Recording(), buffer_.get(), digests.data(), digests.size() * sizeof(uint32_t));
    }
    for (const auto& row : rows_) {
      file_ << "{\"event\":\"output\",\"run\":" << run_ << ",\"ordinal\":" << row.ordinal
            << ",\"node_utf8_hex\":\"" << row.node << "\",\"op_utf8_hex\":\"" << row.op
            << "\",\"port\":" << row.port << ",\"shape\":[";
      for (size_t i = 0; i < row.shape.size(); ++i) { if (i) file_ << ','; file_ << row.shape[i]; }
      file_ << "],\"queue_checked\":" << (checked ? "true" : "false");
      if (!row.skip.empty()) file_ << ",\"skip\":\"" << row.skip << '"';
      else if (!checked) file_ << ",\"skip\":\"unchecked_queue\"";
      else if (row.copy_check) {
        const auto offset = row.hash_row * kM0NumericTraceWordsV1;
        ORT_ENFORCE(digests[offset+3] == 0x54525031u, "M0 transpose diagnostic marker");
        file_ << ",\"kind\":\"transpose_copy_check\",\"copy_mismatches\":" << digests[offset]
              << ",\"first_copy_mismatch\":" << digests[offset+1] << ",\"elements\":" << digests[offset+2]
              << ",\"input_output_alias\":" << (row.copy_alias ? "true" : "false");
      }
      else {
        const auto offset = row.hash_row * kM0NumericTraceWordsV1;
        file_ << ",\"xor\":" << digests[offset] << ",\"sum\":" << digests[offset + 1]
              << ",\"nonfinite\":" << digests[offset + 2] << ",\"elements\":" << digests[offset + 3];
      }
      file_ << "}\n";
    }
    file_ << "{\"event\":\"end\",\"run\":" << run_ << ",\"queue_checked\":" << (checked ? "true" : "false")
          << ",\"hashed_outputs\":" << (hash_rows_-copy_rows_) << ",\"copy_checks\":" << copy_rows_ << ",\"reported_outputs\":" << rows_.size()
          << ",\"omitted_outputs\":" << omitted_ << ",\"diagnostic_overhead\":true}\n";
    file_.flush(); buffer_.reset();
    ORT_ENFORCE(file_.good(), "M0 numeric trace end write failed");
  }
 private:
  static std::string Hex(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    value = value.substr(0, 256);
    std::string result; result.reserve(value.size() * 2);
    for (unsigned char c : value) { result += digits[c >> 4]; result += digits[c & 15]; }
    return result;
  }
  struct Row {
    uint64_t ordinal = 0; size_t port = 0; uint32_t hash_row = 0; bool copy_check = false, copy_alias = false;
    std::string node, op, skip; std::vector<int64_t> shape;
  };
  WebGpuExecutionProvider& ep_;
  std::unique_ptr<WGPUBufferImpl, std::function<void(WGPUBuffer)>> buffer_;
  std::ofstream file_;
  std::string session_, owner_run_;
  std::vector<Row> rows_;
  bool active_ = false, started_ = false;
  uint32_t hash_rows_ = 0, copy_rows_ = 0;
  uint64_t run_ = 0, ordinal_ = 0, omitted_ = 0;
};

Ep::~Ep() = default;
Status Ep::TraceNumericOutputs(ComputeContext& context) const {
  ORT_RETURN_IF_NOT(numeric_trace_, "M0 numeric trace observer missing");
  return numeric_trace_->Observe(context);
}


// Constructor
Ep::Ep(std::unique_ptr<IExecutionProvider> impl, Factory& factory, const OrtLogger& logger, const Config& config)
    : onnxruntime::ep::adapter::Ep{std::move(impl), config.cpu_allocator, config.device_allocator},
      factory_{factory},
      logger_{logger},
      config_{config} {
  if (M0GpuNumericTraceV1()) {
    numeric_trace_ = std::make_unique<M0NumericTraceV1>(*static_cast<WebGpuExecutionProvider*>(EpImpl()));
  }
  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  CreateProfiler = CreateProfilerImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = nullptr;  // Per-kernel EP does not use Compile
  ReleaseNodeComputeInfos = nullptr;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  SetDynamicOptions = nullptr;  // Not implemented
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;
  CreateAllocator = CreateAllocatorImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  GetDefaultMemoryDevice = GetDefaultMemoryDeviceImpl;
  GetCompiledModelCompatibilityInfo = nullptr;  // Not a compiled EP
  IsConcurrentRunSupported = IsConcurrentRunSupportedImpl;
  IsGraphCaptureEnabled = IsGraphCaptureEnabledImpl;
  IsGraphCaptured = IsGraphCapturedImpl;
  ReplayGraph = ReplayGraphImpl;
  ReleaseCapturedGraph = ReleaseCapturedGraphImpl;
  GetGraphCaptureNodeAssignmentPolicy = GetGraphCaptureNodeAssignmentPolicyImpl;
}

OrtStatus* ORT_API_CALL Ep::CreateProfilerImpl(OrtEp* self,OrtEpProfilerImpl** profiler) noexcept{
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  if(!profiler)return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"Profiler output absent");
  *profiler=nullptr;
  auto* retained=static_cast<Ep*>(self)->RetainedProfiler();
  if(retained)*profiler=new M0EpProfilerBridgeV1(*retained);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

// OrtEp interface implementations
const char* ORT_API_CALL Ep::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->factory_.GetName(&ep->factory_);
}

OrtStatus* ORT_API_CALL Ep::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                              OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN

  auto& ep = *static_cast<WebGpuExecutionProvider*>(static_cast<Ep*>(this_ptr)->EpImpl());
  Ort::ConstGraph ort_graph{graph};

  // Get all nodes in the graph
  std::vector<Ort::ConstNode> all_nodes = ort_graph.GetNodes();

  if (all_nodes.empty()) {
    return nullptr;  // No nodes to process
  }

  std::vector<const OrtNode*> candidate_nodes;
  std::vector<const OrtNode*> tentative_candidate_nodes;

  // For each node, check if we have a registered kernel for it
  for (const auto& node : all_nodes) {
    std::string ep_name = node.GetEpName();

    if (ep_name == kWebGpuExecutionProvider) {
      candidate_nodes.push_back(node);
      continue;
    }

    // Reject nodes already assigned to a different (non-CPU) EP
    if (!ep_name.empty() && ep_name != kCpuExecutionProvider) {
      continue;
    }

    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(Api().ep.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def));

    LOGS(ep.GetEpLogger(), INFO) << "M0_RW01_LOOKUP index=" << node.GetId()
                                 << " result=" << (kernel_def ? "found" : "missing");
    if (kernel_def != nullptr) {
      Ort::ConstKernelDef definition{kernel_def};
      std::ostringstream memory;
      memory << "M0_RW01_MEMORY index=" << node.GetId() << " inputs=";
      const auto inputs = node.GetInputs();
      const auto outputs = node.GetOutputs();
      for (size_t i = 0; i < inputs.size(); ++i) memory << int(definition.GetInputMemType(i)) << ",";
      memory << " outputs=";
      for (size_t i = 0; i < outputs.size(); ++i) memory << int(definition.GetOutputMemType(i)) << ",";
      LOGS(ep.GetEpLogger(), INFO) << memory.str();
    }
    if (kernel_def == nullptr) {
      LOGS(ep.GetEpLogger(), INFO) << "webgpu kernel not found in registries for Op type: "
                                   << node.GetOperatorType() << " node name: " << node.GetName();
      continue;
    }

    auto cpu_node_names = ep.GetForceCpuNodeNames();
    if (std::find(cpu_node_names.begin(),
                  cpu_node_names.end(),
                  node.GetName()) != cpu_node_names.end()) {
      LOGS(ep.GetEpLogger(), INFO) << "Force CPU execution for node: " << node.GetName();
      continue;
    }

    //
    // The following code checks if the node is really supported by webgpu EP
    //

#define FALLBACK_TO_CPU_IF_EXIST_INPUT(idx)            \
  if (inputs.size() > idx && inputs[idx] != nullptr) { \
    continue;                                          \
  }

#define FALLBACK_TO_CPU_IF_EXIST_OUTPUT(idx)             \
  if (outputs.size() > idx && outputs[idx] != nullptr) { \
    continue;                                            \
  }

    // Check for Attention
    if (node.GetOperatorType() == "Attention" && node.GetDomain() == kMSDomain) {
      const auto& inputs = node.GetInputs();
      const auto& outputs = node.GetOutputs();

      // Current implementation does not support mask_index(input[3]), past(input[4]) and past_seq_len(input[6])
      FALLBACK_TO_CPU_IF_EXIST_INPUT(3);
      FALLBACK_TO_CPU_IF_EXIST_INPUT(4);
      FALLBACK_TO_CPU_IF_EXIST_INPUT(6);

      // Current implementation does not support present(output[1])
      FALLBACK_TO_CPU_IF_EXIST_OUTPUT(1);

      // If attribute past_present_share_buffer is set, fallback to CPU
      bool has_past_present_share_buffer = false;
      for (const auto& attr : node.GetAttributes()) {
        if (attr.GetName() == "past_present_share_buffer") {
          int64_t val = 0;
          RETURN_IF_ERROR(attr.GetValue(val));
          if (val != 0) {
            has_past_present_share_buffer = true;
          }
          break;
        }
      }
      if (has_past_present_share_buffer) {
        continue;
      }
    }

    candidate_nodes.push_back(node);
    tentative_candidate_nodes.push_back(node);
  }

  std::unordered_set<const OrtNode*> cpu_preferred_nodes;
  RETURN_IF_ERROR(onnxruntime::ep::GetCpuPreferredNodes(*ort_graph,
                                                        *graph_support_info,
                                                        static_cast<Ep*>(this_ptr)->GetOrtLogger(),
                                                        tentative_candidate_nodes,
                                                        cpu_preferred_nodes));

  for (const auto& node : candidate_nodes) {
    LOGS(ep.GetEpLogger(), INFO) << "M0_RW01_CAPABILITY name=" << Ort::ConstNode{node}.GetName()
                                 << " decision=" << (cpu_preferred_nodes.count(node) ? "cpu_preferred" : "candidate_only");
    if (cpu_preferred_nodes.count(node) == 0) {
      RETURN_IF_ERROR(Api().ep.EpGraphSupportInfo_AddSingleNode(graph_support_info, node));
    }
  }

  return nullptr;

  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr,
    _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN

  *kernel_registry = nullptr;

  // For the WebGPU EP, delegate to the CreateKernelRegistry function
  // which properly constructs a registry using only public APIs
  auto* ep = static_cast<Ep*>(this_ptr);

  auto& webgpu_ep = *static_cast<WebGpuExecutionProvider*>(ep->EpImpl());

  *kernel_registry = *webgpu_ep.GetKernelRegistryImpl();
  return nullptr;

  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                       _Out_ OrtEpDataLayout* preferred_data_layout) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // Delegate to the underlying WebGPU EP's GetPreferredLayout()
  // DataLayout enum values map 1:1 to OrtEpDataLayout (NCHW=0, NHWC=1)
  auto* ep = static_cast<Ep*>(this_ptr);
  *preferred_data_layout = static_cast<OrtEpDataLayout>(ep->EpImpl()->GetPreferredLayout());
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                             _In_z_ const char* domain,
                                                             _In_z_ const char* op_type,
                                                             _In_ OrtEpDataLayout target_data_layout,
                                                             _Outptr_ int* should_convert) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // DataLayout enum values map 1:1 to OrtEpDataLayout (NCHW=0, NHWC=1)
  auto* ep = static_cast<Ep*>(this_ptr);
  auto result = ep->EpImpl()->ShouldConvertDataLayoutForOp(domain, op_type,
                                                           static_cast<DataLayout>(target_data_layout));
  if (result.has_value()) {
    *should_convert = result.value() ? 1 : 0;
  } else {
    *should_convert = -1;
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::OnRunStartImpl(_In_ OrtEp* this_ptr,
                                           _In_ const OrtRunOptions* run_options) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  onnxruntime::RunOptions options{};
  // Forward the closed native execution profile as well as graph annotation.
  auto graph_annotation_str = Api().ort.GetRunConfigEntry(run_options, kOrtRunOptionsConfigCudaGraphAnnotation);
  if (graph_annotation_str != nullptr) {
    auto status = options.config_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, graph_annotation_str);
    if (!status.IsOK()) {
      return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                    status.ErrorMessage().c_str());
    }
  }
  for (const char* key : kM0NativeRunCallbackKeysV1) {
    const char* value = Api().ort.GetRunConfigEntry(run_options, key);
    if (value) {
      auto forwarded = options.config_options.AddConfigEntry(key, value);
      if (!forwarded.IsOK()) return Api().ort.CreateStatus(static_cast<OrtErrorCode>(forwarded.Code()), forwarded.ErrorMessage().c_str());
    }
  }
  auto* ep = static_cast<Ep*>(this_ptr);
  if (ep->numeric_trace_) {
    if (options.config_options.GetConfigOrDefault("m0.fullgpu.bounded_submissions_v1", "0") != "1") {
      return Api().ort.CreateStatus(ORT_INVALID_ARGUMENT, "M0 numeric trace requires bounded native Run");
    }
    ep->numeric_trace_->Begin(options.config_options.GetConfigOrDefault("m0.fullgpu.actual_session_v2", ""),
                              options.config_options.GetConfigOrDefault("m0.fullgpu.actual_run_v2", ""));
  }
  auto status = ep->EpImpl()->OnRunStart(options);
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::OnRunEndImpl(_In_ OrtEp* this_ptr,
                                         _In_ const OrtRunOptions* run_options,
                                         _In_ bool sync_stream) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  onnxruntime::RunOptions options{};
  for (const char* key : kM0NativeRunCallbackKeysV1) {
    const char* value = Api().ort.GetRunConfigEntry(run_options, key);
    if (value) {
      auto forwarded = options.config_options.AddConfigEntry(key, value);
      if (!forwarded.IsOK()) return Api().ort.CreateStatus(static_cast<OrtErrorCode>(forwarded.Code()), forwarded.ErrorMessage().c_str());
    }
  }
  auto status = ep->EpImpl()->OnRunEnd(sync_stream, options);
  if (ep->numeric_trace_) ep->numeric_trace_->End(status.IsOK() && sync_stream);
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::IsConcurrentRunSupportedImpl(_In_ OrtEp* /*this_ptr*/, _Out_ bool* is_concurrent_run_supported) noexcept {
  *is_concurrent_run_supported = false;
  return nullptr;
}

bool ORT_API_CALL Ep::IsGraphCaptureEnabledImpl(_In_ const OrtEp* this_ptr) noexcept {
  auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->EpImpl()->IsGraphCaptureEnabled();
}

bool ORT_API_CALL Ep::IsGraphCapturedImpl(_In_ const OrtEp* this_ptr, _In_ int graph_annotation_id) noexcept {
  auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->EpImpl()->IsGraphCaptured(graph_annotation_id);
}

OrtStatus* ORT_API_CALL Ep::ReplayGraphImpl(_In_ OrtEp* this_ptr, _In_ int graph_annotation_id) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->ReplayGraph(graph_annotation_id);
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::ReleaseCapturedGraphImpl(_In_ OrtEp* this_ptr, _In_ int graph_annotation_id) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->ReleaseCapturedGraph(graph_annotation_id);
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtGraphCaptureNodeAssignmentPolicy ORT_API_CALL Ep::GetGraphCaptureNodeAssignmentPolicyImpl(
    _In_ const OrtEp* this_ptr) noexcept {
  auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->EpImpl()->GetGraphCaptureNodeAssignmentPolicy();
}

OrtStatus* ORT_API_CALL Ep::CreateSyncStreamForDeviceImpl(
    OrtEp* this_ptr, const OrtMemoryDevice* memory_device, OrtSyncStreamImpl** stream) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep_adapter = static_cast<Ep*>(this_ptr);
  auto& ep = *static_cast<WebGpuExecutionProvider*>(ep_adapter->EpImpl());
  ORT_ENFORCE(Api().ep.MemoryDevice_GetDeviceType(memory_device) == OrtMemoryInfoDeviceType_GPU &&
                  static_cast<int64_t>(Api().ep.MemoryDevice_GetDeviceId(memory_device)) == ep.GetDeviceId(),
              "Unsupported memory device for WebGPU Session stream.");
  *stream = CreateWebGpuSyncStream(ep);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::GetDefaultMemoryDeviceImpl(
    const OrtEp* this_ptr, const OrtMemoryDevice** memory_device) noexcept {
  const auto* ep = static_cast<const Ep*>(this_ptr);
  *memory_device = Api().ep.MemoryInfo_GetMemoryDevice(&ep->config_.device_allocator->Info());
  return nullptr;
}

OrtStatus* ORT_API_CALL Ep::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                _In_ const OrtMemoryInfo* memory_info,
                                                _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  Ort::ConstMemoryInfo ort_memory_info{memory_info};
  // Wrap the existing Session allocator, unlike Factory::CreateAllocatorImpl's Env path.
  // The wrapper retains the allocator implementation, not the EP borrowed by its getters.
  if (ort_memory_info.GetAllocatorType() == OrtReadOnlyAllocator) {
    *allocator = CreateWebGpuSessionAllocator(ep->config_.initializer_allocator);
  } else {
    *allocator = CreateWebGpuSessionAllocator(ep->config_.device_allocator);
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
