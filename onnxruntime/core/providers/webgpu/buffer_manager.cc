// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/common/m0_memory_reserve_v1.h"
#include "core/common/m0_observation_options_v1.h"
#include <gsl/gsl>

namespace onnxruntime {
namespace webgpu {

namespace {

constexpr size_t kM0TransferChunkBytes = 8 * 1024 * 1024;

void EnforceM0MemoryReserve(WebGpuContext& context, const BufferManager& manager,
                            CommandRecordingState& recording, uint64_t additional_bytes) {
  if (context.M0BoundedSubmissionsActiveV1()) {
    ORT_THROW_IF_ERROR(context.M0DrainOnMemoryPressureV1(manager, recording, additional_bytes));
  }
}

constexpr size_t NormalizeBufferSize(size_t size) {
  return (size + 15) / 16 * 16;
}

// WebGPU requires that the copy size in CopyBufferToBuffer must be a multiple of 4 bytes.
constexpr size_t NormalizeCopySize(size_t size) {
  return (size + 3) / 4 * 4;
}

void EnforceBufferUnmapped(WebGpuContext& context, WGPUBuffer buffer) {
  if (context.ValidationMode() > ValidationMode::Basic) {
    ORT_ENFORCE(wgpuBufferGetMapState(buffer) == WGPUBufferMapState_Unmapped, "Buffer is still mapped.");
  }
}

}  // namespace

class DisabledCacheManager : public IBufferCacheManager {
  bool SupportsBufferReuse() const override { return false; }

  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/) override {
    // always return empty buffer
    return nullptr;
  }
  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }
  void ReleaseBuffer(WGPUBuffer buffer) override {
    wgpuBufferRelease(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    // no-op
  }
};

class LazyReleaseCacheManager : public IBufferCacheManager {
  bool SupportsBufferReuse() const override { return false; }

  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/) override {
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    Release();
    pending_buffers_.clear();
  }

 public:
  ~LazyReleaseCacheManager() {
    Release();
  }

 protected:
  void Release() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
  }

  std::vector<WGPUBuffer> pending_buffers_;
};

class SimpleCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buffers_.find(buffer_size);
    if (it != buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }

    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    for (auto& buffer : pending_buffers_) {
      buffers_[static_cast<size_t>(wgpuBufferGetSize(buffer))].emplace_back(buffer);
    }
    pending_buffers_.clear();
  }

 public:
  ~SimpleCacheManager() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
  }

 protected:
  std::map<size_t, std::vector<WGPUBuffer>> buffers_;
  std::vector<WGPUBuffer> pending_buffers_;
};

// TODO: maybe use different bucket size for storage and uniform buffers?
constexpr std::initializer_list<std::pair<const size_t, size_t>> BUCKET_DEFAULT_LIMIT_TABLE = {
    {64, 250},
    {128, 200},
    {256, 200},
    {512, 200},
    {2048, 230},
    {4096, 200},
    {8192, 50},
    {16384, 50},
    {32768, 50},
    {65536, 50},
    {131072, 50},
    {262144, 50},
    {524288, 50},
    {1048576, 50},
    {2097152, 30},
    {4194304, 20},
    {8388608, 10},
    {12582912, 10},
    {16777216, 10},
    {26214400, 15},
    {33554432, 22},
    {44236800, 2},
    {58982400, 6},
    // we don't want to cache the bucket sizes below but not caching them
    // results in some major performance hits for models like sd-turbo.
    {67108864, 6},
    {134217728, 6},
    {167772160, 6},
};

class BucketCacheManager : public IBufferCacheManager {
 public:
  BucketCacheManager() : buckets_limit_{BUCKET_DEFAULT_LIMIT_TABLE} {
    Initialize();
  }
  BucketCacheManager(std::unordered_map<size_t, size_t>&& buckets_limit) : buckets_limit_{buckets_limit} {
    Initialize();
  }

  size_t CalculateBufferSize(size_t request_size) override {
    // binary serch size
    auto it = std::lower_bound(buckets_keys_.begin(), buckets_keys_.end(), request_size);
    if (it == buckets_keys_.end()) {
      return NormalizeBufferSize(request_size);
    } else {
      return *it;
    }
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    auto buffer_size = static_cast<size_t>(wgpuBufferGetSize(buffer));

    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && it->second.size() < buckets_limit_[buffer_size]) {
      it->second.emplace_back(buffer);
    } else {
      pending_buffers_.emplace_back(buffer, buffer_size);
    }
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    for (auto& [buffer, buffer_size] : pending_buffers_) {
      auto it = buckets_.find(buffer_size);
      if (it != buckets_.end() && it->second.size() < buckets_limit_[buffer_size]) {
        it->second.emplace_back(buffer);
      } else {
        wgpuBufferRelease(buffer);
      }
    }
    pending_buffers_.clear();
  }

  ~BucketCacheManager() {
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
    for (auto& buffer_info : pending_buffers_) {
      wgpuBufferRelease(buffer_info.first);
    }
  }

 protected:
  void Initialize() {
    buckets_keys_.reserve(buckets_limit_.size());
    buckets_.reserve(buckets_limit_.size());
    for (const auto& pair : buckets_limit_) {
      buckets_keys_.push_back(pair.first);
      buckets_.emplace(pair.first, std::vector<WGPUBuffer>());
    }
    std::sort(buckets_keys_.begin(), buckets_keys_.end());

#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(std::all_of(buckets_keys_.begin(), buckets_keys_.end(), [](size_t size) { return size % 16 == 0; }),
                "Bucket sizes must be multiples of 16.");

    for (size_t i = 1; i < buckets_keys_.size(); ++i) {
      ORT_ENFORCE(buckets_keys_[i] > buckets_keys_[i - 1], "Bucket sizes must be in increasing order.");
    }
#endif
  }
  std::unordered_map<size_t, size_t> buckets_limit_;
  std::unordered_map<size_t, std::vector<WGPUBuffer>> buckets_;
  std::vector<std::pair<WGPUBuffer, size_t>> pending_buffers_;
  std::vector<size_t> buckets_keys_;
};

class GraphCacheManager : public IBufferCacheManager {
 public:
  GraphCacheManager() : buckets_limit_{BUCKET_DEFAULT_LIMIT_TABLE} {
    Initialize();
  }
  GraphCacheManager(std::unordered_map<size_t, size_t>&& buckets_limit) : buckets_limit_{buckets_limit} {
    Initialize();
  }

  size_t CalculateBufferSize(size_t request_size) override {
    // binary serch size
    auto it = std::lower_bound(buckets_keys_.begin(), buckets_keys_.end(), request_size);
    if (it == buckets_keys_.end()) {
      return NormalizeBufferSize(request_size);
    } else {
      return *it;
    }
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    auto buffer_size = static_cast<size_t>(wgpuBufferGetSize(buffer));
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end()) {
      it->second.emplace_back(buffer);
    } else {
      // Insert a new bucket for non-standard buffer sizes
      buckets_[buffer_size] = std::vector<WGPUBuffer>{buffer};
    }
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    // no-op - buffers are already in buckets_
  }

  std::vector<std::pair<size_t, WGPUBuffer>> ExtractCachedBuffers() override {
    std::vector<std::pair<size_t, WGPUBuffer>> result;
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        result.emplace_back(pair.first, buffer);
      }
      pair.second.clear();
    }
    return result;
  }

  void AbsorbCachedBuffers(std::vector<std::pair<size_t, WGPUBuffer>>&& buffers) override {
    for (auto& entry : buffers) {
      if (entry.second) {
        ReleaseBuffer(entry.second);
      }
    }
  }

  ~GraphCacheManager() {
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
  }

 protected:
  void Initialize() {
    buckets_keys_.reserve(buckets_limit_.size());
    buckets_.reserve(buckets_limit_.size());
    for (const auto& pair : buckets_limit_) {
      buckets_keys_.push_back(pair.first);
      buckets_.emplace(pair.first, std::vector<WGPUBuffer>());
    }
    std::sort(buckets_keys_.begin(), buckets_keys_.end());

#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(std::all_of(buckets_keys_.begin(), buckets_keys_.end(), [](size_t size) { return size % 16 == 0; }),
                "Bucket sizes must be multiples of 16.");

    for (size_t i = 1; i < buckets_keys_.size(); ++i) {
      ORT_ENFORCE(buckets_keys_[i] > buckets_keys_[i - 1], "Bucket sizes must be in increasing order.");
    }
#endif
  }
  std::unordered_map<size_t, size_t> buckets_limit_;
  std::unordered_map<size_t, std::vector<WGPUBuffer>> buckets_;
  std::vector<size_t> buckets_keys_;
};

class GraphSimpleCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buffers_.find(buffer_size);
    if (it != buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }

    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState graph_capture_state) override {
    for (auto& buffer : pending_buffers_) {
      if (graph_capture_state == GraphCaptureState::Default) {
        buffers_[static_cast<size_t>(wgpuBufferGetSize(buffer))].emplace_back(buffer);
      } else {
        captured_buffers_.emplace_back(buffer);
      }
    }
    pending_buffers_.clear();
  }

 public:
  ~GraphSimpleCacheManager() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
    for (auto& buffer : captured_buffers_) {
      wgpuBufferRelease(buffer);
    }
  }

  std::vector<std::pair<size_t, WGPUBuffer>> ExtractCachedBuffers() override {
    // Donation is expected after captured commands have been released and any
    // in-flight work has completed; all three containers therefore hold buffers
    // no longer referenced by the device.
    std::vector<std::pair<size_t, WGPUBuffer>> result;
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        result.emplace_back(pair.first, buffer);
      }
      pair.second.clear();
    }
    buffers_.clear();
    for (auto& buffer : pending_buffers_) {
      result.emplace_back(static_cast<size_t>(wgpuBufferGetSize(buffer)), buffer);
    }
    pending_buffers_.clear();
    for (auto& buffer : captured_buffers_) {
      result.emplace_back(static_cast<size_t>(wgpuBufferGetSize(buffer)), buffer);
    }
    captured_buffers_.clear();
    return result;
  }

  void AbsorbCachedBuffers(std::vector<std::pair<size_t, WGPUBuffer>>&& buffers) override {
    for (auto& entry : buffers) {
      if (entry.second) {
        buffers_[entry.first].emplace_back(entry.second);
      }
    }
  }

 protected:
  std::map<size_t, std::vector<WGPUBuffer>> buffers_;
  std::vector<WGPUBuffer> pending_buffers_;
  std::vector<WGPUBuffer> captured_buffers_;
};

std::unique_ptr<IBufferCacheManager> CreateBufferCacheManager(BufferCacheMode cache_mode) {
  switch (cache_mode) {
    case BufferCacheMode::Disabled:
      return std::make_unique<DisabledCacheManager>();
    case BufferCacheMode::LazyRelease:
      return std::make_unique<LazyReleaseCacheManager>();
    case BufferCacheMode::Simple:
      return std::make_unique<SimpleCacheManager>();
    case BufferCacheMode::Bucket:
      return std::make_unique<BucketCacheManager>();
    case BufferCacheMode::Graph:
      return std::make_unique<GraphCacheManager>();
    case BufferCacheMode::GraphSimple:
      return std::make_unique<GraphSimpleCacheManager>();
    default:
      ORT_NOT_IMPLEMENTED("Unsupported buffer cache mode");
  }
}

std::ostream& operator<<(std::ostream& os, BufferCacheMode mode) {
  switch (mode) {
    case BufferCacheMode::Disabled:
      os << "Disabled";
      break;
    case BufferCacheMode::LazyRelease:
      os << "LazyRelease";
      break;
    case BufferCacheMode::Simple:
      os << "Simple";
      break;
    case BufferCacheMode::Bucket:
      os << "Bucket";
      break;
    case BufferCacheMode::Graph:
      os << "Graph";
      break;
    case BufferCacheMode::GraphSimple:
      os << "GraphSimple";
      break;
    default:
      os << "Unknown(" << static_cast<int>(mode) << ")";
  }
  return os;
}

BufferManager::BufferManager(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode, BufferCacheMode default_buffer_cache_mode)
    : context_{context},
      storage_cache_{CreateBufferCacheManager(storage_buffer_cache_mode)},
      uniform_cache_{CreateBufferCacheManager(uniform_buffer_cache_mode)},
      query_resolve_cache_{CreateBufferCacheManager(query_resolve_buffer_cache_mode)},
      default_cache_{CreateBufferCacheManager(default_buffer_cache_mode)} {
}

M0FullGpuResourceOwnership& BufferManager::M0OwnershipV2() const {
  return context_.M0OwnershipV2();
}

bool BufferManager::M0NonRetainingCachesV1() const {
  std::lock_guard<std::mutex> lock{mutex_};
  return !storage_cache_->SupportsBufferReuse() && !uniform_cache_->SupportsBufferReuse() &&
         !query_resolve_cache_->SupportsBufferReuse() && !default_cache_->SupportsBufferReuse();
}

void BufferManager::Upload(CommandRecordingState& recording, void* src, WGPUBuffer dst, size_t size) const {
  M0HostCostTimerV1 host_cost(context_.M0HostCosts(),M0HostCostV1::Upload);
  LOGS_DEFAULT(INFO) << "M0_RW02_UPLOAD timeline=" << &recording << " bytes=" << size << " dst=" << dst;
  if(size==0)return;
  ORT_ENFORCE(src&&dst&&size<=std::numeric_limits<size_t>::max()-3,"Invalid GPU upload size or pointer.");
  const uint64_t dst_bytes=wgpuBufferGetSize(dst);
  ORT_ENFORCE(NormalizeCopySize(size)<=dst_bytes,"GPU upload exceeds destination backing.");
  // If the buffer is mapped, we can directly write to it.
  void* mapped_data = wgpuBufferGetMappedRange(dst, 0, WGPU_WHOLE_MAP_SIZE);  // ensure the buffer is mapped
  if (mapped_data) {
    memcpy(mapped_data, src, size);
    wgpuBufferUnmap(dst);
    M0OwnershipV2().RecordWorkload(M0WorkloadEventV1::Upload,size,0);
    M0OwnershipV2().LogWorkload("upload_mapped");
    return;
  }

  const bool chunked=recording.m0_submission_budget_bytes&&size>kM0TransferChunkBytes;
  const size_t chunk_limit=chunked?kM0TransferChunkBytes:size;
  for(size_t offset=0;offset<size;){
    const size_t payload=std::min(chunk_limit,size-offset),copy_size=NormalizeCopySize(payload);
    ORT_ENFORCE(offset%4==0&&offset<=dst_bytes&&copy_size<=dst_bytes-offset,
                "GPU upload range exceeds destination backing.");
    wgpu::BufferDescriptor desc{};
    desc.size=copy_size;
    desc.usage=wgpu::BufferUsage::CopySrc|wgpu::BufferUsage::MapWrite;
    desc.mappedAtCreation=true;
    EnforceM0MemoryReserve(context_, *this, recording, desc.size);
    if(recording.m0_submission_budget_bytes){
      ORT_ENFORCE(dst_bytes<=std::numeric_limits<uint64_t>::max()-copy_size,"RW07 upload reference size overflow");
      ORT_THROW_IF_ERROR(context_.M0BeforeEnqueueV1(*this,recording,dst_bytes+copy_size));
    }
    auto staging_buffer=context_.Device().CreateBuffer(&desc);
    mapped_data=staging_buffer.GetMappedRange();
    ORT_ENFORCE(mapped_data,"Failed to map GPU upload staging buffer.");
    memcpy(mapped_data,static_cast<const uint8_t*>(src)+offset,payload);
    if(copy_size>payload)memset(static_cast<uint8_t*>(mapped_data)+payload,0,copy_size-payload);
    staging_buffer.Unmap();
    ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording));
    auto& command_encoder=context_.GetCommandEncoder(recording);
    context_.EndComputePass(recording);
    command_encoder.CopyBufferToBuffer(staging_buffer,0,dst,offset,copy_size);
    M0OwnershipV2().RecordWorkload(M0WorkloadEventV1::Upload,payload,copy_size);
    M0OwnershipV2().LogWorkload("upload_encoded");
    // Wait before allocating another staging chunk, not merely at the queue budget.
    if(chunked){
      ORT_THROW_IF_ERROR(context_.FlushAndWaitChecked(*this,recording));
    }else{
      // The encoder owns the immutable staging copy after this function returns.
      // Native initialization/Run fences the complete bounded window; submitting
      // every small initializer/feed separately only adds queue/driver overhead.
      if(!recording.m0_submission_budget_bytes||!M0BatchHeavyProgramsV1())
        ORT_THROW_IF_ERROR(context_.Flush(*this,recording));
      ORT_THROW_IF_ERROR(context_.M0FlushIfBudgetReachedV1(*this,recording));
    }
    offset+=payload;
  }
}

void BufferManager::MemCpy(CommandRecordingState& recording, WGPUBuffer src, WGPUBuffer dst, size_t size) const {
  ORT_ENFORCE(size<=std::numeric_limits<size_t>::max()-3,"GPU copy size alignment overflow.");
  MemCpyRange(recording,src,0,dst,0,NormalizeCopySize(size));
}

void BufferManager::MemCpyRange(CommandRecordingState& recording,WGPUBuffer src,uint64_t source_offset,
                                WGPUBuffer dst,uint64_t destination_offset,uint64_t size) const {
  ORT_ENFORCE(src&&dst,"Source and destination GPU buffers must exist.");
  ORT_ENFORCE(src != dst, "Source and destination buffers must be different.");
  ORT_ENFORCE(source_offset%4==0&&destination_offset%4==0&&size%4==0,"GPU copy range must be 4-byte aligned.");
  const uint64_t src_size=wgpuBufferGetSize(src),dst_size=wgpuBufferGetSize(dst);
  ORT_ENFORCE(source_offset<=src_size&&size<=src_size-source_offset&&
              destination_offset<=dst_size&&size<=dst_size-destination_offset,
              "Source and destination GPU ranges exceed their physical backing.");
  if(size==0)return;
  EnforceBufferUnmapped(context_, src);
  EnforceBufferUnmapped(context_, dst);

  if(recording.m0_submission_budget_bytes){
    ORT_ENFORCE(dst_size<=std::numeric_limits<uint64_t>::max()-src_size,
                "RW07 copy reference size overflow");
    ORT_THROW_IF_ERROR(context_.M0BeforeEnqueueV1(*this,recording,src_size+dst_size));
  }
  ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording));
  auto& command_encoder = context_.GetCommandEncoder(recording);
  context_.EndComputePass(recording);
  command_encoder.CopyBufferToBuffer(src,source_offset,dst,destination_offset,size);
  M0OwnershipV2().RecordWorkload(M0WorkloadEventV1::DeviceCopy,size);
  ORT_THROW_IF_ERROR(context_.M0FlushIfBudgetReachedV1(*this,recording));
}

WGPUBuffer BufferManager::Create(CommandRecordingState& recording, size_t size, wgpu::BufferUsage usage,
                                 bool initialize_to_zero,
                                 bool submit_zero_initialize) const {
  M0HostCostTimerV1 host_cost(context_.M0HostCosts(),M0HostCostV1::Allocation);
  size_t buffer_size;
  WGPUBuffer buffer;
  {
    std::lock_guard<std::mutex> lock{mutex_};
    auto& cache = GetCacheManager(usage);
    // Native planning is expressed in exact aligned tensor bytes. Bucket
    // padding can otherwise turn three legal carrier views into a per-command
    // queue-budget overflow. Non-bucket sizes retain the existing release policy.
    buffer_size = M0GpuExecutionRedesignV1()&&bool(usage&wgpu::BufferUsage::Storage)?
        NormalizeBufferSize(size):cache.CalculateBufferSize(size);
    buffer = cache.TryAcquireCachedBuffer(buffer_size);
  }
  if (buffer) {
    auto buffer_guard = wgpu::Buffer::Acquire(buffer);
    EnforceM0MemoryReserve(context_, *this, recording, 0);
    const uint64_t generation = M0OwnershipV2().Acquire(reinterpret_cast<uintptr_t>(buffer), size);
    bool ownership_committed = false;
    auto ownership_rollback = gsl::finally([&]() {
      if (!ownership_committed) M0OwnershipV2().Retire(reinterpret_cast<uintptr_t>(buffer), generation);
    });
    LOGS_DEFAULT(INFO) << "M0_RW02_ALLOC cached=1 requested=" << size << " actual=" << buffer_size << " handle=" << buffer;
    LOGS_DEFAULT(INFO) << "M0_RW07_ALLOCATION_ACQUIRE handle=" << buffer << " generation=" << generation << " requested=" << size << " cached=1";
    if (initialize_to_zero) {
      // Plugin plain allocations pass call-local recording and submit only their own clear.
      // AllocOnStream defers the clear on the Session timeline. Built-in allocations use the
      // Session recording and retain their caller-supplied submission policy.
      ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording));
      context_.EndComputePass(recording);
      context_.GetCommandEncoder(recording).ClearBuffer(buffer, 0, buffer_size);
      if (submit_zero_initialize) {
        ORT_THROW_IF_ERROR(context_.Flush(*this, recording));
      }
    }
    ownership_committed = true;
    return buffer_guard.MoveToCHandle();
  }

  // cache miss, create a new buffer
  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = usage;
  if (usage & wgpu::BufferUsage::MapWrite) {
    desc.mappedAtCreation = true;  // ensure the buffer is mapped for writing at creation
  }
  EnforceM0MemoryReserve(context_, *this, recording, buffer_size);
  buffer = context_.Device().CreateBuffer(&desc).MoveToCHandle();
  LOGS_DEFAULT(INFO) << "M0_RW02_ALLOC cached=0 requested=" << size << " actual=" << buffer_size << " handle=" << buffer;

  ORT_ENFORCE(buffer, "Failed to create GPU buffer: size=", buffer_size, ", usage=", uint64_t(usage), ".");
  auto buffer_guard = wgpu::Buffer::Acquire(buffer);
  const uint64_t generation = M0OwnershipV2().Acquire(reinterpret_cast<uintptr_t>(buffer), size);
  bool ownership_committed = false;
  auto ownership_rollback = gsl::finally([&]() {
    if (!ownership_committed) M0OwnershipV2().Retire(reinterpret_cast<uintptr_t>(buffer), generation);
  });

  {
    std::lock_guard<std::mutex> lock{mutex_};
    GetCacheManager(usage).RegisterBuffer(buffer, size);
  }
  LOGS_DEFAULT(INFO) << "M0_RW07_ALLOCATION_ACQUIRE handle=" << buffer << " generation=" << generation << " requested=" << size << " cached=0";
  ownership_committed = true;
  return buffer_guard.MoveToCHandle();
}

size_t BufferManager::M0AllocationBackingSizeV1(size_t size, bool uniform) const {
  if (size == 0) return 0;
  // All current caches normalize to 16 bytes on their non-bucket path.
  // Reject before invoking the exact cache calculation, not after wraparound.
  ORT_ENFORCE(size <= std::numeric_limits<size_t>::max() - 15, "RW07_NATIVE_DEVICE_LIMITS_SIZE_OVERFLOW");
  std::lock_guard<std::mutex> lock{mutex_};
  auto& cache = GetCacheManager(uniform ? wgpu::BufferUsage::Uniform : wgpu::BufferUsage::Storage);
  const size_t backing = M0GpuExecutionRedesignV1()&&!uniform?NormalizeBufferSize(size):cache.CalculateBufferSize(size);
  ORT_ENFORCE(backing >= size, "RW07_NATIVE_DEVICE_LIMITS_BACKING_OVERFLOW");
  return backing;
}

bool BufferManager::SupportsUMA() const {
#if !defined(__wasm__)
  // Check if the device supports the BufferMapExtendedUsages feature
  return context_.DeviceHasFeature(wgpu::FeatureName::BufferMapExtendedUsages);
#else
  return false;
#endif  // !defined(__wasm__)
}

void BufferManager::Release(WGPUBuffer buffer, const CommandRecordingState* recording) const {
  EnforceBufferUnmapped(context_, buffer);
  std::lock_guard<std::mutex> lock{mutex_};
  M0FullGpuAllocationIdentity identity;
  ORT_ENFORCE(M0OwnershipV2().Describe(reinterpret_cast<uintptr_t>(buffer), identity) &&
              M0OwnershipV2().Retire(reinterpret_cast<uintptr_t>(buffer), identity.generation),
              "RW07 GPU allocation release identity missing");
  context_.M0FaultHooksV2().RetireTerminal(reinterpret_cast<uintptr_t>(buffer), identity.generation);
  LOGS_DEFAULT(INFO) << "M0_RW07_ALLOCATION_RETIRE handle=" << buffer << " generation=" << identity.generation
                    << " requested=" << identity.requested_bytes;
  auto& cache = GetCacheManager(buffer);
  // Keep reusable buffers out of the cache while this recording has unsubmitted work.
  // Graph caches also need this protection from streamless allocations.
  // Relaxed ordering suffices: recording a buffer's use (and setting the flag) must happen-before
  // its release, on the same thread or through a synchronized ownership handoff. Only independent
  // buffer releases may race with recording. The flag publishes no command data, and mutex_ orders
  // releases against clearing it after submission or abandonment.
  if (cache.SupportsBufferReuse() && recording != nullptr &&
      recording->has_unsubmitted_work.load(std::memory_order_relaxed)) {
    pending_buffers_[recording].emplace_back(wgpu::Buffer::Acquire(buffer));
    return;
  }

  cache.ReleaseBuffer(buffer);
}

void BufferManager::Download(CommandRecordingState& recording, WGPUBuffer src, void* dst, size_t size) const {
  M0HostCostTimerV1 host_cost(context_.M0HostCosts(),M0HostCostV1::Download);
  LOGS_DEFAULT(INFO) << "M0_RW02_DOWNLOAD_BEGIN timeline=" << &recording << " bytes=" << size << " src=" << src;
  if(size>0){
    ORT_ENFORCE(src&&dst&&size<=std::numeric_limits<size_t>::max()-15,"Invalid GPU download size or pointer.");
    ORT_ENFORCE(NormalizeBufferSize(size)<=wgpuBufferGetSize(src),"GPU download exceeds source backing.");
    const bool chunked=recording.m0_submission_budget_bytes&&size>kM0TransferChunkBytes;
    const size_t chunk_limit=chunked?kM0TransferChunkBytes:size;
    for(size_t offset=0;offset<size;){
      const size_t payload=std::min(chunk_limit,size-offset);
      DownloadRange(recording,src,offset,static_cast<uint8_t*>(dst)+offset,payload);
      if(chunked)ORT_THROW_IF_ERROR(context_.FlushAndWaitChecked(*this,recording));
      offset+=payload;
    }
  }
  LOGS_DEFAULT(INFO) << "M0_RW02_DOWNLOAD_COMPLETE timeline=" << &recording << " src=" << src << " status=success";
}

void BufferManager::DownloadRange(CommandRecordingState& recording,WGPUBuffer src,size_t source_offset,
                                  void* dst,size_t size) const {
  ORT_ENFORCE(size>0&&size<=std::numeric_limits<size_t>::max()-15&&source_offset%4==0,
              "Invalid GPU download range.");
  const size_t buffer_size=NormalizeBufferSize(size);
  const uint64_t source_bytes=wgpuBufferGetSize(src);
  ORT_ENFORCE(source_offset<=source_bytes&&buffer_size<=source_bytes-source_offset,
              "GPU download range exceeds source backing.");
  // Encode pending deferred dispatches before recording the readback; the flush below submits both
  // in order.
  ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording));

  EnforceBufferUnmapped(context_, src);

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

  EnforceM0MemoryReserve(context_, *this, recording, desc.size);
  if(recording.m0_submission_budget_bytes){
    ORT_ENFORCE(source_bytes<=std::numeric_limits<uint64_t>::max()-buffer_size,"RW07 download reference size overflow");
    ORT_THROW_IF_ERROR(context_.M0BeforeEnqueueV1(*this,recording,source_bytes+buffer_size));
  }
  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  auto& command_encoder = context_.GetCommandEncoder(recording);
  context_.EndComputePass(recording);
  command_encoder.CopyBufferToBuffer(src, source_offset, staging_buffer, 0, buffer_size);
  M0OwnershipV2().RecordWorkload(M0WorkloadEventV1::Download,size,buffer_size);
  M0OwnershipV2().LogWorkload("download_encoded");
  ORT_THROW_IF_ERROR(context_.Flush(*this, recording));
  ORT_THROW_IF_ERROR(context_.M0FlushIfBudgetReachedV1(*this,recording));

  // TODO: revise wait in whole project

  struct MapAsyncResult {
    wgpu::MapAsyncStatus status{};
    std::string message{};
    bool callback_exception = false;
    wgpu::Buffer staging;
    wgpu::Buffer source;
  };
  auto map_async_result = std::make_shared<MapAsyncResult>();
  map_async_result->staging = staging_buffer;
  wgpuBufferAddRef(src);
  map_async_result->source = wgpu::Buffer::Acquire(src);

  M0FullGpuAllocationIdentity source_identity;
  const bool injected_invalid_map = M0OwnershipV2().Describe(reinterpret_cast<uintptr_t>(src), source_identity) &&
    context_.M0FaultHooksV2().ConsumeInvalidTerminalMap(reinterpret_cast<uintptr_t>(src), source_identity.generation);
  if (injected_invalid_map) { LOGS_DEFAULT(INFO) << "M0_RW07_ACTUAL_INVALID_MAP_BEGIN src=" << src
      << " generation=" << source_identity.generation << " allocation_bytes=" << buffer_size
      << " requested_map_bytes=" << (buffer_size + 8); }
  auto pop_injected_scope = [&]() -> Status {
    struct ScopeResult {
      wgpu::PopErrorScopeStatus pop{};
      wgpu::ErrorType error{};
      bool received = false, callback_exception = false;
      std::string message;
    };
    auto scope = std::make_shared<ScopeResult>();
    const auto waited = context_.Wait(context_.Device().PopErrorScope(wgpu::CallbackMode::WaitAnyOnly,
      [scope](wgpu::PopErrorScopeStatus pop, wgpu::ErrorType error, wgpu::StringView message) noexcept {
        scope->pop = pop; scope->error = error; scope->received = true;
        try { scope->message = std::string{std::string_view(message)}; }
        catch (...) { scope->callback_exception = true; }
      }));
    if (!waited.IsOK() || !scope->received || scope->callback_exception ||
        scope->pop != wgpu::PopErrorScopeStatus::Success ||
        (scope->error != wgpu::ErrorType::Validation && scope->error != wgpu::ErrorType::NoError)) {
      context_.M0OwnershipV2().Poison();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07 unknown injected map scope completion");
    }
    LOGS_DEFAULT(INFO) << "M0_RW07_ACTUAL_INVALID_MAP_SCOPE pop=" << static_cast<int>(scope->pop)
      << " error=" << static_cast<int>(scope->error) << " message=" << scope->message;
    return Status::OK();
  };
  bool injected_scope_pending = false;
  auto injected_scope_cleanup = gsl::finally([&]() noexcept {
    if (!injected_scope_pending) return;
    try { if (!pop_injected_scope().IsOK()) context_.M0OwnershipV2().Poison(); }
    catch (...) { context_.M0OwnershipV2().Poison(); }
  });
  if (injected_invalid_map) { context_.PushErrorScope(wgpu::ErrorFilter::Validation); injected_scope_pending = true; }
  const auto map_wait_status = context_.Wait(
      staging_buffer.MapAsync(
          wgpu::MapMode::Read, 0, injected_invalid_map ? buffer_size + 8 : buffer_size, wgpu::CallbackMode::WaitAnyOnly,
          // Note: Don't throw from a Dawn callback.
          [map_async_result](wgpu::MapAsyncStatus status, wgpu::StringView message) noexcept {
            map_async_result->status = status;
            try {
              if (auto message_sv = static_cast<std::string_view>(message); !message_sv.empty())
                map_async_result->message = std::string{message_sv};
            } catch (...) { map_async_result->callback_exception = true; }
          }));
  if (!map_wait_status.IsOK()) context_.M0OwnershipV2().Poison();
  ORT_THROW_IF_ERROR(map_wait_status);
  if (injected_scope_pending) {
    injected_scope_pending = false;
    try { ORT_THROW_IF_ERROR(pop_injected_scope()); }
    catch (...) { context_.M0OwnershipV2().Poison(); throw; }
  }
  ORT_ENFORCE(!map_async_result->callback_exception, "RW07 map callback metadata allocation failed");

  if (injected_invalid_map) { LOGS_DEFAULT(INFO) << "M0_RW07_ACTUAL_INVALID_MAP_CALLBACK src=" << src
      << " generation=" << source_identity.generation << " status=" << static_cast<int>(map_async_result->status)
      << " message=" << map_async_result->message << " wait_completed=1 source_staging_retained=1"; }
  ORT_ENFORCE(map_async_result->status == wgpu::MapAsyncStatus::Success,
              "Failed to download data from buffer. wgpu::MapAsyncStatus value: ",
              static_cast<int>(map_async_result->status), ", message: ", map_async_result->message);

  auto mapped_data = staging_buffer.GetConstMappedRange();
  ORT_ENFORCE(mapped_data,"Failed to access GPU download staging buffer.");
  memcpy(dst, mapped_data, size);
  staging_buffer.Unmap();
}

void BufferManager::RefreshPendingBuffers(CommandRecordingState& recording, GraphCaptureState graph_capture_state) const {
  std::lock_guard<std::mutex> lock{mutex_};
  // Serialize the transition to idle with Free, so a concurrent release cannot miss this refresh.
  recording.has_unsubmitted_work.store(false, std::memory_order_relaxed);
  if (auto it = pending_buffers_.find(&recording); it != pending_buffers_.end()) {
    for (auto& buffer : it->second) {
      GetCacheManager(buffer.Get()).ReleaseBuffer(buffer.MoveToCHandle());
    }
    pending_buffers_.erase(it);
  }

  if (graph_capture_state == GraphCaptureState::Replaying) {
    return;
  }

  storage_cache_->OnRefresh(graph_capture_state);
  uniform_cache_->OnRefresh(graph_capture_state);
  query_resolve_cache_->OnRefresh(graph_capture_state);
  default_cache_->OnRefresh(graph_capture_state);
}

void BufferManager::DiscardPendingBuffers(CommandRecordingState& recording) const {
  std::lock_guard<std::mutex> lock{mutex_};
  recording.has_unsubmitted_work.store(false, std::memory_order_relaxed);
  pending_buffers_.erase(&recording);
}

std::vector<std::pair<size_t, WGPUBuffer>> BufferManager::ExtractCachedBuffers(wgpu::BufferUsage usage) {
  std::lock_guard<std::mutex> lock{mutex_};
  return GetCacheManager(usage).ExtractCachedBuffers();
}

void BufferManager::AbsorbCachedBuffers(
    wgpu::BufferUsage usage,
    std::vector<std::pair<size_t, WGPUBuffer>>&& buffers) {
  std::lock_guard<std::mutex> lock{mutex_};
  GetCacheManager(usage).AbsorbCachedBuffers(std::move(buffers));
}

IBufferCacheManager& BufferManager::GetCacheManager(wgpu::BufferUsage usage) const {
  if (usage & wgpu::BufferUsage::Storage) {
    return *storage_cache_;
  } else if (usage & wgpu::BufferUsage::Uniform) {
    return *uniform_cache_;
  } else if (usage & wgpu::BufferUsage::QueryResolve) {
    return *query_resolve_cache_;
  } else {
    return *default_cache_;
  }
}

IBufferCacheManager& BufferManager::GetCacheManager(WGPUBuffer buffer) const {
  auto usage = static_cast<wgpu::BufferUsage>(wgpuBufferGetUsage(buffer));
  return GetCacheManager(usage);
}

std::unique_ptr<BufferManager> BufferManagerFactory::Create(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode, BufferCacheMode default_buffer_cache_mode) {
  return std::make_unique<BufferManager>(context, storage_buffer_cache_mode, uniform_buffer_cache_mode, query_resolve_buffer_cache_mode, default_buffer_cache_mode);
}

}  // namespace webgpu
}  // namespace onnxruntime
