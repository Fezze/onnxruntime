// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/webgpu/m0_fullgpu_resource_ownership.h"
#include "core/providers/webgpu/m0_queued_buffer_budget_v1.h"
#include "core/providers/webgpu/m0_dawn_artifact_cache_v1.h"
#include "core/providers/webgpu/m0_host_costs_v1.h"
#include "core/providers/webgpu/m0_fullgpu_fault_hooks.h"

#include <atomic>
#include "core/common/inlined_containers.h"
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/providers/webgpu/webgpu_external_header.h"

#include "core/common/common.h"
#include "core/common/m0_memory_reserve_v1.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/webgpu_utils.h"

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
#include "core/providers/webgpu/webgpu_pix_frame_generator.h"
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
#include <d3d12.h>
#include <wrl/client.h>
#endif

namespace onnxruntime {
class Tensor;

namespace webgpu {

class WebGpuContext;
class ComputeContextBase;
class ComputeContext;
class ProgramBase;
// GPU counter origins need not stay comparable across submissions. Durations
// use a single query pair; placement uses that submission's CPU invocation.
inline bool M0DecodeGpuTimestampIntervalV1(uint64_t begin, uint64_t end, uint64_t origin,
    int64_t cpu_offset_us, int64_t& timestamp_us, int64_t& duration_us) {
  if(end<begin || begin<origin || cpu_offset_us<0)return false;
  const auto micros=[](uint64_t ns){return ns/1000u+uint64_t(ns%1000u>=500u);};
  const uint64_t offset=micros(begin-origin),duration=micros(end-begin);
  if(offset>uint64_t(INT64_MAX-cpu_offset_us)||duration>uint64_t(INT64_MAX))return false;
  timestamp_us=cpu_offset_us+int64_t(offset);duration_us=int64_t(duration);
  return true;
}
// PendingKernelInfo stores profiling information for a kernel execution
struct PendingKernelInfo {
  PendingKernelInfo(std::string_view kernel_name,
                    std::string_view kernel_type,
                    std::string_view program_name,
                    std::string_view cache_key,
                    const std::vector<ProgramInput>& inputs,
                    const std::vector<ProgramOutput>& outputs)
      : name{absl::StrJoin({kernel_name, kernel_type, program_name}, "&")}, cache_key{cache_key} {
    // Store shape information instead of tensor pointers to avoid accessing released tensors
    input_shapes.reserve(inputs.size());
    for (const auto& input : inputs) {
      input_shapes.emplace_back(input.use_override_shape ? input.override_shape : input.tensor->Shape());
    }
    output_shapes.reserve(outputs.size());
    for (const auto& output : outputs) {
      output_shapes.emplace_back(output.use_override_shape ? output.override_shape : output.tensor->Shape());
    }
  }

  PendingKernelInfo(const PendingKernelInfo&) = default;
  PendingKernelInfo& operator=(const PendingKernelInfo&) = default;
  PendingKernelInfo(PendingKernelInfo&&) = default;
  PendingKernelInfo& operator=(PendingKernelInfo&&) = default;

  std::string name;
  std::string cache_key;
  std::vector<TensorShape> input_shapes;
  std::vector<TensorShape> output_shapes;
};

// State for one in-flight pipeline build. The compiled pipeline is written into
// `callback_context->pipeline` by the async callback, which shares ownership of the result state
// independently of this build's lifetime.
struct PendingPipelineBuild {
  std::string name;
  std::vector<int> shape_uniform_ranks;
  wgpu::BindGroupLayout bind_group_layout;
  std::shared_ptr<PipelineCallbackContext> callback_context;
  wgpu::Future future;
};

// Resources for one recorded dispatch. The pipeline fields represent these lifecycle states:
//
//                                 pending_build    compute_pipeline
// 1. Program cache hit                 empty              set
//    The pipeline was already available when the dispatch was recorded.
//
// 2. First cache miss for a key         set              empty
//    This command owns the asynchronous build for its program key.
//
// 3. Later cache miss for the same key  empty            empty
//    An earlier command in deferred_dispatches_ owns the build. This command keeps program_key so
//    pipeline resolution can obtain the completed pipeline from the program cache.
//
// 4. Deferred pipeline resolved         empty             set
//    WaitForDeferredPipelineBuilds() has waited for the owning build as needed, populated the
//    program cache, and assigned a ready pipeline to every command before encoding starts.
//
// 5. Captured command / graph replay    empty             set
//    The command was moved into captured graph storage after encoding and retains the ready
//    pipeline and bind group for later replays. Replay never depends on pending_build.
//
// A command with both fields empty is therefore valid only in state 3 before deferred pipeline
// resolution. DispatchCommand() requires compute_pipeline to be set.
struct CommandRecordingState;
static_assert(M0UniformPackingV1::kMaximumBytes==kM0UniformArenaBytesV1);
struct M0UniformUploadBatchV1 {
  M0UniformUploadBatchV1(const BufferManager& owner,CommandRecordingState& timeline,WGPUBuffer value,size_t capacity)
      :manager(owner),recording(timeline),buffer(value),packing(capacity){}
  ~M0UniformUploadBatchV1(){
    try{manager.Release(buffer,&recording);}catch(...){manager.M0OwnershipV2().Poison();}
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0UniformUploadBatchV1);
  const BufferManager& manager;
  CommandRecordingState& recording;
  WGPUBuffer buffer;
  M0UniformPackingV1 packing;
};

struct CapturedCommandInfo {
  std::string program_key;
  std::optional<PendingPipelineBuild> pending_build;
  std::optional<wgpu::ComputePipeline> compute_pipeline;
  wgpu::BindGroup bind_group;
  std::array<uint32_t, 3> dispatch_group{1, 1, 1};
  // WGPUBuffer for indirect dispatch, nullptr if not using indirect dispatch
  WGPUBuffer indirect_buffer = nullptr;
  std::shared_ptr<M0UniformUploadBatchV1> m0_uniform_upload;
  // Optional profiling data
  std::optional<PendingKernelInfo> pending_kernel_info;
};

// Conservative scheduling bound, not measured GPU instructions or FLOPs.
inline constexpr uint64_t kM0ScheduledProductsPerWindowV1=uint64_t{1}<<28;
// A command recording timeline has one caller at a time. ORT serializes Run on a Session;
// plugin streamless operations use independent encoders. BufferManager synchronizes buffer
// releases from concurrent allocator callers separately from command recording.
struct CommandRecordingState {
  CommandRecordingState() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CommandRecordingState);

  wgpu::CommandEncoder command_encoder;
  wgpu::ComputePassEncoder compute_pass_encoder;
  uint32_t num_pending_dispatches = 0;
  // Concurrent allocator frees inspect only this flag, never the encoders or deferred dispatches.
  // BufferManager clears it under its cache lock after submission or abandonment.
  std::atomic<bool> has_unsubmitted_work{false};
  // Legacy mode charges repeated buffer references; batched mode charges unique
  // retained backing buffers. Both bound the recording, not physical residency.
  uint64_t m0_submission_budget_bytes=0;
  uint64_t m0_queued_reference_bytes=0;
  uint64_t m0_queued_scheduled_products=0;
  webgpu::M0QueuedBufferBudgetV1 m0_unique_queued_buffers;
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  std::mutex pending_release_callbacks_mutex;
  std::vector<std::function<void()>> pending_release_callbacks;
#endif
  std::vector<CapturedCommandInfo> deferred_dispatches;
  std::vector<PendingKernelInfo> pending_kernels;
  GraphCaptureState graph_capture_state{GraphCaptureState::Default};
  std::vector<CapturedCommandInfo>* external_captured_commands = nullptr;
  // Keep uploaded blocks alive through Submit, including already encoded commands.
  std::shared_ptr<M0UniformUploadBatchV1> m0_uniform_current;
  std::shared_ptr<M0UniformUploadBatchV1> m0_uniform_retired;
  std::vector<std::shared_ptr<M0UniformUploadBatchV1>> m0_uniform_batches;
  uint64_t m0_uniform_arena_uploads=0,m0_uniform_arena_bytes=0;
  uint64_t m0_uniform_arena_allocations=0,m0_uniform_arena_recycled=0;
  void M0RetireUniformUploadsV1(){
    if(!m0_uniform_batches.empty()){
      auto candidate=m0_uniform_batches.back();
      m0_uniform_current.reset();
      for(auto& command:deferred_dispatches)command.m0_uniform_upload.reset();
      m0_uniform_batches.clear();
      m0_uniform_retired.reset();
      if(m0_submission_budget_bytes&&graph_capture_state==GraphCaptureState::Default&&candidate.use_count()==1)
        m0_uniform_retired=std::move(candidate);
    }
    for(auto& command:deferred_dispatches)command.m0_uniform_upload.reset();
    m0_uniform_batches.clear();
  }
  void M0RecycleUniformUploadsAfterCheckedCompletionV1(){
    // Admission can drain while Run still owns a prospective command's block.
    // Its uniforms are already packed, but its command has not been submitted.
    // Sole ownership is therefore essential even after a successful queue wait.
    if(m0_submission_budget_bytes&&graph_capture_state==GraphCaptureState::Default&&
       m0_uniform_retired&&m0_uniform_retired.use_count()==1&&
       m0_uniform_retired->packing.Sealed()){
      m0_uniform_retired->packing.ResetAfterCheckedCompletion();
      m0_uniform_current=std::move(m0_uniform_retired);
      ++m0_uniform_arena_recycled;
    }
    m0_uniform_retired.reset();
  }
  void M0ReleaseUniformUploadsV1(){
    m0_uniform_current.reset();
    m0_uniform_retired.reset();
    for(auto& command:deferred_dispatches)command.m0_uniform_upload.reset();
    m0_uniform_batches.clear();
  }
};

struct WebGpuBufferCacheConfig {
  struct ConfigEntry {
    BufferCacheMode mode;
    std::string config_string;  // preserved for customized configuration, eg. bucket sizes
  };
  ConfigEntry storage{BufferCacheMode::Bucket, {}};
  ConfigEntry uniform{BufferCacheMode::Simple, {}};
  ConfigEntry query_resolve{BufferCacheMode::Disabled, {}};
  ConfigEntry default_entry{BufferCacheMode::Disabled, {}};
};

/// <summary>
/// Represents the configuration options for creating a WebGpuContext.
/// </summary>
struct WebGpuContextConfig {
  int context_id{0};
  WGPUInstance instance{nullptr};
  WGPUDevice device{nullptr};
  const void* dawn_proc_table{nullptr};
  ValidationMode validation_mode{
#ifndef NDEBUG
      webgpu::ValidationMode::Full  // for debug build, enable full validation by default
#else
      webgpu::ValidationMode::Basic  // for release build, enable basic validation by default
#endif  // !NDEBUG
  };
  bool validation_mode_explicitly_set{false};
  bool enable_robustness{
#ifndef NDEBUG
      true  // for debug builds, enable robust buffer access by default
#else
      false  // for release builds, disable robust buffer access for performance by default
#endif
  };
  bool enable_robustness_explicitly_set{false};
  bool preserve_device{false};
  WeightLoadAccelerationMode weight_load_acceleration_mode{
      WeightLoadAccelerationMode::Off};
  // When true, skip Dawn adapter/device creation and all device-dependent initialization; the context
  // can only be used for graph transformation, not execution. Derived from kOrtSessionOptionCompileOnly.
  bool compile_only{false};
  uint32_t max_num_pending_dispatches{16};
  uint64_t max_storage_buffer_binding_size{0};
  // Internal test hook. Provider-option parsing never populates this field.
  uint64_t test_only_max_storage_buffer_binding_size{0};
  uint32_t max_storage_buffers_per_shader_stage{0};
  WebGpuBufferCacheConfig buffer_cache_config{};
  std::optional<uint32_t> adapter_index;
  int power_preference{static_cast<int>(WGPUPowerPreference_HighPerformance)};
  int backend_type{
#ifdef _WIN32
  // Setup Windows default backend type based on the build configuration
#if defined(DAWN_ENABLE_D3D12)
      static_cast<int>(WGPUBackendType_D3D12)
#elif defined(DAWN_ENABLE_VULKAN)
      static_cast<int>(WGPUBackendType_Vulkan)
#else
      0
#endif
#else
      0
#endif
  };
};

class WebGpuContextFactory {
 public:
  struct WebGpuContextInfo {
    std::unique_ptr<WebGpuContext> context;
    int ref_count;
  };

  /// <summary>
  /// Create a new WebGPU context for the specified context ID if not present, or return the existing one. (ref-count based)
  /// </summary>
  static WebGpuContext& CreateContext(const WebGpuContextConfig& config);

  /// <summary>
  /// Get the WebGPU context for the specified context ID. Throw if not present.
  /// </summary>
  static WebGpuContext& GetContext(int context_id);

  /// <summary>
  /// Retain an existing WebGPU context. (ref-count based)
  /// </summary>
  static void RetainContext(int context_id);

  /// <summary>
  /// Release the WebGPU context. (ref-count based)
  /// </summary>
  static void ReleaseContext(int context_id);

  static void Cleanup();

  /// <summary>
  /// Return the default context. Create if not present.
  /// </summary>
  static WebGpuContext& DefaultContext();

 private:
  WebGpuContextFactory() {}

  static std::mutex mutex_;
  static std::once_flag init_default_flag_;

  // Use pointers to heap-allocated objects so that their destructors do NOT run
  // during static destruction at process exit. This avoids crashes when dependent
  // DLLs (e.g. dxcompiler.dll) have already been unloaded by the OS.
  // Cleanup() explicitly deletes them during normal unload. In the shared-library
  // build this is reached via ReleaseEpFactory, and in the WebGPU static-lib build
  // it is reached from OrtEnv::~OrtEnv via CleanupWebGpuContexts().
  // On abnormal/process termination they simply leak, which is safe.
  static std::unordered_map<int32_t, WebGpuContextInfo>* contexts_;
  static WGPUInstance default_instance_;
};

// Class WebGpuContext includes all necessary resources for the context.
class WebGpuContext final {
 public:
  static constexpr uint64_t kMinConfigurableStorageBufferBindingSize = 256;
  // WebGPU's guaranteed maxStorageBufferBindingSize limit is 128 MiB.
  // See https://gpuweb.github.io/gpuweb/?utm_source=openai
  static constexpr uint64_t kWebGpuGuaranteedMaxStorageBufferBindingSize =
      128ULL * 1024 * 1024;

  uint32_t M0ResidualBlockChoiceV1(const std::string& key) const {
    std::lock_guard<std::mutex> lock(m0_residual_block_mutex_);
    const auto found=m0_residual_blocks_.find(key);
    return found==m0_residual_blocks_.end()?0u:found->second;
  }
  void M0RememberResidualBlockV1(const std::string& key,uint32_t block) {
    ORT_ENFORCE(block==1||block==2||block==4||block==8||block==16||block==32||block==64,"RW07 residual tuning block");
    std::lock_guard<std::mutex> lock(m0_residual_block_mutex_);
    if(m0_residual_blocks_.size()>=128)m0_residual_blocks_.clear();
    m0_residual_blocks_[key]=block;
  }

  ~WebGpuContext();

  Status Wait(wgpu::Future f);

  const wgpu::Instance& Instance() const { return instance_; }
  const wgpu::Device& Device() const { return device_; }
  // Payload-free admission metadata; never flushes or submits GPU work.
  Status M0NativeResourceStatusV3() const;
  // Caller retains the allocation throughout this payload-free query. Refusal
  // leaves output unchanged; foreign handles never reach a Dawn buffer query.
  bool M0DescribeStorageAllocationV1(uintptr_t handle, uint64_t required_bytes,
                                     webgpu::M0FullGpuStorageAllocationV1& output) const;
  const wgpu::Queue& M0NativeResourceQueueV3() const { return device_queue_; }

  const wgpu::AdapterInfo& AdapterInfo() const { return adapter_info_; }
  const wgpu::Limits& DeviceLimits() const { return device_limits_; }
  const M0DawnArtifactCacheV1* M0ArtifactCacheV1()const{return m0_dawn_artifact_cache_;}
  M0HostCostsV1& M0HostCosts()const{return m0_host_costs_;}
  bool DeviceHasFeature(wgpu::FeatureName feature) const { return device_features_.contains(feature); }
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& SubgroupMatrixConfigs() const { return subgroup_matrix_configs_; }

  const wgpu::CommandEncoder& GetCommandEncoder(CommandRecordingState& recording) {
    if (!recording.command_encoder) {
      recording.command_encoder = device_.CreateCommandEncoder();
      recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);
    }
    return recording.command_encoder;
  }

  const wgpu::ComputePassEncoder& GetComputePassEncoder(CommandRecordingState& recording) {
    if (!recording.compute_pass_encoder) {
      auto& command_encoder = GetCommandEncoder(recording);

      wgpu::ComputePassDescriptor compute_pass_desc{};

      if (is_profiling_ && query_type_ == TimestampQueryType::AtPasses &&
          recording.graph_capture_state != GraphCaptureState::Capturing) {
        wgpu::PassTimestampWrites timestampWrites = {
            nullptr,
            query_set_,
            recording.num_pending_dispatches * 2,
            recording.num_pending_dispatches * 2 + 1};
        compute_pass_desc.timestampWrites = &timestampWrites;
      }

      recording.compute_pass_encoder = command_encoder.BeginComputePass(&compute_pass_desc);
    }
    return recording.compute_pass_encoder;
  }

  void EndComputePass(CommandRecordingState& recording) {
    if (recording.compute_pass_encoder) {
      recording.compute_pass_encoder.End();
      recording.compute_pass_encoder = nullptr;
    }
  }
  void CaptureBegin(std::vector<webgpu::CapturedCommandInfo>* captured_commands,
                    const webgpu::BufferManager& buffer_manager,
                    CommandRecordingState& recording);
  void CaptureEnd(CommandRecordingState& recording);
  void Replay(const std::vector<webgpu::CapturedCommandInfo>& captured_commands,
              const webgpu::BufferManager& buffer_manager,
              CommandRecordingState& recording);
  void ReleaseGraphResources(std::vector<webgpu::CapturedCommandInfo>& captured_commands);

  Status Flush(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording);

  // Unlike Flush, waits for submitted queue work and checks scoped errors and device loss.
  // Serialize with other operations on this recording. Scopes cover flushing, not earlier encoding.
  Status FlushAndWaitChecked(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording,
                            bool diagnostic_zero_deadline = false);
  Status M0BeforeEnqueueV1(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording,
                         uint64_t reference_bytes);
  Status M0BeforeEnqueueUniqueV1(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording,
                                  const std::vector<webgpu::M0QueuedBufferBudgetV1::Reference>& references);
  Status M0DrainOnMemoryPressureV1(const webgpu::BufferManager& buffer_mgr,
                                  CommandRecordingState& recording, uint64_t allocation_bytes);
  Status M0FlushIfBudgetReachedV1(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording);

  // Context-level managers are shared by sessions and synchronize their buffer caches internally.
  webgpu::BufferManager& BufferManager() const { return *buffer_mgr_; }
  webgpu::M0FullGpuResourceOwnership& M0OwnershipV2() const { return m0_ownership_v2_; }
  webgpu::M0FullGpuFaultHooksV2& M0FaultHooksV2() const { return m0_fault_hooks_v2_; }
  bool M0ProfilingActiveV2() const { return is_profiling_; }
  uint32_t M0DispatchWindowV1() const { return max_num_pending_dispatches_; }
  bool M0BoundedSubmissionsActiveV1() const { return m0_bounded_submissions_active_.load(std::memory_order_acquire); }
  bool M0BeginBoundedSubmissionsV1(){
    bool expected=false;
    const bool acquired=m0_bounded_submissions_active_.compare_exchange_strong(expected,true,std::memory_order_acq_rel);
    if(acquired)m0_memory_probe_window_.Reset();
    return acquired;
  }
  bool M0EndBoundedSubmissionsV1(){
    bool expected=true;
    return m0_bounded_submissions_active_.compare_exchange_strong(expected,false,std::memory_order_acq_rel);
  }
  webgpu::BufferManager& InitializerBufferManager() const { return *initializer_buffer_mgr_; }

  inline webgpu::ValidationMode ValidationMode() const {
    return validation_mode_;
  }

  // False for a device-free ("virtual device") context, which has no Dawn device and can only run graph
  // transformation. Used to hand out a no-op allocator instead of a real GpuBufferAllocator.
  inline bool HasDevice() const { return device_ != nullptr; }
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  ID3D12Device* WeightLoadingD3D12Device() const;
  bool D3D12SharedResourceFeaturesAvailable() const {
    return d3d12_shared_resource_features_available_;
  }
  wgpu::BackendType SelectedBackendType() const {
    return selected_backend_type_;
  }
#endif

  //
  // Get Split-K configuration.
  //
  const SplitKConfig& GetSplitKConfig() const {
    return *split_k_config_;
  }

  // Set the CPU time base (ORT profiler's profiling_start_time) used to align GPU
  // timestamps with ORT CPU events. Pushed in by WebGpuProfiler::StartProfiling, which
  // is the only place that receives the framework's profiling start time for both
  // session-level and run-level profiling.
  void SetProfilingStartTime(TimePoint profiling_start_time) { profiling_start_time_ = profiling_start_time; }
  void StartProfiling();
  // Collect pending GPU profiling data into the given events vector.
  void CollectProfilingData(profiling::Events& events);
  // Collect pending GPU profiling data into the shared events_ vector (run-level).
  void CollectProfilingData();
  void EndProfiling(TimePoint, profiling::Events& events);

  //
  // Push error scope.
  //
  // This is useful only when "skip_validation" is not set.
  //
  void PushErrorScope(wgpu::ErrorFilter filter = wgpu::ErrorFilter::Validation);

  //
  // Pop error scope.
  //
  // This is useful only when "skip_validation" is not set.
  //
  Status PopErrorScope();

  Status Run(ComputeContextBase& context, const ProgramBase& program, uint64_t scheduled_products=0);
  Status RunWithCheckedCompletion(ComputeContextBase& context, const ProgramBase& program);

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
  std::unique_ptr<WebGpuPIXFrameGenerator> CreatePIXFrameGenerator() {
    return std::make_unique<WebGpuPIXFrameGenerator>(instance_,
                                                     Device());
  }
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

 private:
  mutable std::mutex m0_residual_block_mutex_;
  InlinedHashMap<std::string,uint32_t> m0_residual_blocks_;
  enum class TimestampQueryType {
    None = 0,
    InsidePasses,
    AtPasses
  };

  static uint64_t ResolveMaxStorageBufferBindingSize(
      uint64_t max_storage_buffer_binding_size,
      uint64_t test_only_max_storage_buffer_binding_size) {
    ORT_ENFORCE(max_storage_buffer_binding_size == 0 ||
                    max_storage_buffer_binding_size >= kWebGpuGuaranteedMaxStorageBufferBindingSize,
                "max_storage_buffer_binding_size must be 0 or at least 128 MiB");
    ORT_ENFORCE(test_only_max_storage_buffer_binding_size == 0 ||
                    test_only_max_storage_buffer_binding_size >= kMinConfigurableStorageBufferBindingSize,
                "test_only_max_storage_buffer_binding_size must be 0 or at least 256 bytes");
    return test_only_max_storage_buffer_binding_size != 0
               ? test_only_max_storage_buffer_binding_size
               : max_storage_buffer_binding_size;
  }

  WebGpuContext(WGPUInstance instance,
                WGPUDevice device,
                webgpu::ValidationMode validation_mode,
                bool validation_mode_explicitly_set,
                bool preserve_device,
                uint64_t max_storage_buffer_binding_size,
                uint64_t test_only_max_storage_buffer_binding_size,
                uint32_t max_storage_buffers_per_shader_stage)
      : instance_{instance},
        device_{device},
        validation_mode_{validation_mode},
        validation_mode_explicitly_set_{validation_mode_explicitly_set},
        query_type_{TimestampQueryType::None},
        preserve_device_{preserve_device},
        max_storage_buffer_binding_size_{ResolveMaxStorageBufferBindingSize(
            max_storage_buffer_binding_size, test_only_max_storage_buffer_binding_size)},
        max_storage_buffers_per_shader_stage_{max_storage_buffers_per_shader_stage} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuContext);

  void Initialize(const WebGpuContextConfig& config);

  wgpu::BindGroup CreateBindGroup(const std::vector<WGPUBuffer>& bind_buffers,
                                  const std::vector<uint32_t>& bind_buffers_segments,
                                  const std::vector<uint64_t>& bind_buffer_offsets,
                                  const std::vector<uint64_t>& bind_buffer_sizes,
                                  uint64_t uniform_buffer_bytes,
                                  const wgpu::BindGroupLayout& bind_group_layout,
                                  std::string_view label) const;
  void DispatchCommand(const webgpu::CapturedCommandInfo& command, CommandRecordingState& recording);
  Status EncodeDeferredDispatches(CommandRecordingState& recording);

  std::vector<const char*> GetEnabledAdapterToggles() const;
  std::vector<const char*> GetEnabledDeviceToggles() const;
  std::vector<const char*> GetDisabledDeviceToggles() const;
  std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const;
  wgpu::Limits GetRequiredLimits(const wgpu::Adapter& adapter) const;
  void WriteTimestamp(CommandRecordingState& recording, uint32_t query_index);

  struct PendingQueryInfo {
    PendingQueryInfo(std::vector<PendingKernelInfo>&& kernels, wgpu::Buffer query_buffer, int64_t cpu_offset_us)
        : kernels{std::move(kernels)}, query_buffer{query_buffer}, cpu_offset_us{cpu_offset_us} {}

    PendingQueryInfo(PendingQueryInfo&&) = default;
    PendingQueryInfo& operator=(PendingQueryInfo&&) = default;
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(PendingQueryInfo);

    std::vector<PendingKernelInfo> kernels;
    wgpu::Buffer query_buffer;
    int64_t cpu_offset_us;
  };

  // Find the build owner for a cache key in the current deferred window.
  PendingPipelineBuild* FindPendingPipelineBuild(CommandRecordingState& recording, std::string_view key);
  Status WaitForDeferredPipelineBuilds(CommandRecordingState& recording);
  Status CheckDeviceStatus() const;

  friend class BufferManager;
  friend class ComputeContext;
  friend class WebGpuContextFactory;

  std::once_flag init_flag_;

  struct DeviceErrorState {
    std::mutex mutex;
    Status status;
  };
  std::shared_ptr<DeviceErrorState> device_error_state_{std::make_shared<DeviceErrorState>()};
  wgpu::Instance instance_;
  // Non-owning process service: asynchronous Dawn callbacks can outlive a context.
  M0DawnArtifactCacheV1* m0_dawn_artifact_cache_=nullptr;
  mutable M0HostCostsV1 m0_host_costs_;
  wgpu::Device device_;

  webgpu::ValidationMode validation_mode_;
  bool validation_mode_explicitly_set_;
  bool enable_robustness_ = false;
  std::optional<uint32_t> adapter_index_;
  int adapter_power_preference_ = static_cast<int>(WGPUPowerPreference_HighPerformance);
  int adapter_backend_type_ = 0;

  wgpu::Queue device_queue_;
  wgpu::AdapterInfo adapter_info_;
  wgpu::Limits device_limits_;
  std::unordered_set<wgpu::FeatureName> device_features_;
  wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs_;

  mutable webgpu::M0FullGpuResourceOwnership m0_ownership_v2_;
  std::atomic<bool> m0_bounded_submissions_active_{false};
  mutable webgpu::M0FullGpuFaultHooksV2 m0_fault_hooks_v2_;
  std::unique_ptr<webgpu::BufferManager> buffer_mgr_;
  std::unique_ptr<webgpu::BufferManager> initializer_buffer_mgr_;
  std::unique_ptr<ProgramManager> program_mgr_;

  uint32_t max_num_pending_dispatches_ = 16;
  M0MemoryProbeWindowV1 m0_memory_probe_window_;

  std::unique_ptr<SplitKConfig> split_k_config_;

  // profiling
  TimestampQueryType query_type_;
  wgpu::QuerySet query_set_;
  wgpu::Buffer query_resolve_buffer_;

  // info of queries pending appending to profiling events
  std::vector<PendingQueryInfo> pending_queries_;

  // ORT profiler's CPU time base, set via SetProfilingStartTime; GPU timestamps are
  // aligned to it. See SetProfilingStartTime.
  TimePoint profiling_start_time_;
  bool is_profiling_ = false;
  // Shared GPU profiling events for run-level profiling.
  profiling::Events events_;
  bool preserve_device_;
  WeightLoadAccelerationMode weight_load_acceleration_mode_{
      WeightLoadAccelerationMode::Off};
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  Microsoft::WRL::ComPtr<ID3D12Device> weight_loading_d3d12_device_;
  bool d3d12_shared_resource_features_available_ = false;
  wgpu::BackendType selected_backend_type_ = wgpu::BackendType::Undefined;
#endif
  uint64_t max_storage_buffer_binding_size_;
  uint32_t max_storage_buffers_per_shader_stage_;
  GraphCaptureState graph_capture_state_{GraphCaptureState::Default};

  // External vector to store captured commands, owned by EP
  std::vector<webgpu::CapturedCommandInfo>* external_captured_commands_ = nullptr;
};

#if !defined(__wasm__)
namespace detail {

// Returns true when the largest host visible device local heap is at least as
// large as the largest device local heap. Without Resizable BAR a discrete GPU
// exposes only a small PCIe window of such memory where larger initializers
// fail to allocate so mapped initializer upload is not used there. It is kept
// separate from adapter feature discovery for heap layout testing.
bool CanMapDeviceLocalMemory(gsl::span<const wgpu::MemoryHeapInfo> heaps);

}  // namespace detail
#endif  // !defined(__wasm__)

}  // namespace webgpu
}  // namespace onnxruntime
