// Source-private RW07 opaque resource administration; never reads GPU payloads.
#pragma once
#include <cstdint>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace onnxruntime::webgpu {
struct M0FullGpuAllocationIdentity {
  uint64_t generation = 0;
  uint64_t requested_bytes = 0;
  uint64_t session = 0;
  uint64_t run = 0;
  uint32_t producer_role = 0;
  uint32_t producer_instance = 0;
  uintptr_t data_handle = 0;
  uint64_t data_generation = 0;
};

// Integration must call Acquire on EVERY successful physical allocation/cache
// acquisition and Retire at the client's Release boundary, before deferred
// cache recycling. Cached WGPUBuffer pointers are not allocation identities.
// GPU guards must compare these opaque IDs against device sidecar fields.
class M0FullGpuResourceOwnership final {
 public:
  uint64_t Acquire(uintptr_t handle, uint64_t requested_bytes);
  bool Retire(uintptr_t handle, uint64_t generation);
  bool Describe(uintptr_t handle, M0FullGpuAllocationIdentity& out) const;
  // Atomically snapshots actual live bound allocations, without comparing
  // producer provenance. GPU guards compare these descriptors with payloads.
  bool DescribeBindings(uintptr_t sidecar, uintptr_t data_handle,
                        M0FullGpuAllocationIdentity& sidecar_out,
                        M0FullGpuAllocationIdentity& data_out) const;
  // Additionally validates the immutable recorded carrier pairing is live.
  bool DescribePair(uintptr_t sidecar, uintptr_t data_handle,
                    M0FullGpuAllocationIdentity& out) const;
  // A live sidecar allocation may be assigned one producer in one Run only.
  // Repeating the identical claim is allowed for multi-dispatch producers.
  bool Claim(uintptr_t sidecar, uint64_t generation, uint64_t session,
             uint64_t run, uint32_t role, uintptr_t data_handle,
             uint64_t data_generation);
  bool Claim(uintptr_t sidecar, uint64_t generation, uint64_t session,
             uint64_t run, uint32_t role, uintptr_t data_handle,
             uint64_t data_generation, uint32_t producer_instance);
  bool ValidatePair(uintptr_t sidecar, uint64_t generation, uint64_t session,
                    uint64_t run, uint32_t role, uint32_t producer_instance,
                    uintptr_t data_handle, uint64_t data_generation,
                    M0FullGpuAllocationIdentity& out) const;
  void Poison() { poisoned_.store(true, std::memory_order_release); }
  bool Poisoned() const { return poisoned_.load(std::memory_order_acquire); }

 private:
  bool ClaimImpl(uintptr_t sidecar, uint64_t generation, uint64_t session,
                 uint64_t run, uint32_t role, uintptr_t data_handle,
                 uint64_t data_generation, uint32_t producer_instance);
  mutable std::mutex mutex_;
  std::unordered_map<uintptr_t, M0FullGpuAllocationIdentity> live_;
  std::atomic<bool> poisoned_{false};
};
}  // namespace onnxruntime::webgpu
