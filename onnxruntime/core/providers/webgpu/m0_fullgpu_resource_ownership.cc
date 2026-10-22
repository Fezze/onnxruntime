#include "core/providers/webgpu/m0_fullgpu_resource_ownership.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include <atomic>
#include <limits>
#include <stdexcept>

namespace onnxruntime::webgpu {
namespace {
std::atomic<uint64_t> next_allocation{1};
uint64_t NextAllocation() {
  uint64_t value = next_allocation.load(std::memory_order_relaxed);
  for (;;) {
    if (value == std::numeric_limits<uint64_t>::max()) {
      throw std::overflow_error("RW07 allocation identity exhausted");
    }
    if (next_allocation.compare_exchange_weak(value, value+1, std::memory_order_relaxed)) return value;
  }
}
}  // namespace

uint64_t M0FullGpuResourceOwnership::Acquire(uintptr_t handle, uint64_t requested_bytes) {
  if (!handle || !requested_bytes) throw std::invalid_argument("RW07 invalid resource descriptor");
  std::lock_guard<std::mutex> lock{mutex_};
  if (live_.count(handle)) throw std::logic_error("RW07 allocation aliases a live resource");
  if (statistics_.acquisitions == std::numeric_limits<uint64_t>::max() ||
      requested_bytes > std::numeric_limits<uint64_t>::max() - statistics_.live_requested_bytes) {
    throw std::overflow_error("RW07 resource statistics exhausted");
  }
  const uint64_t id = NextAllocation();
  M0FullGpuAllocationIdentity identity{};
  identity.generation = id;
  identity.requested_bytes = requested_bytes;
  live_.emplace(handle, identity);
  ++statistics_.acquisitions;
  ++statistics_.live_allocations;
  statistics_.live_requested_bytes += requested_bytes;
  if (statistics_.live_requested_bytes > statistics_.peak_live_requested_bytes)
    statistics_.peak_live_requested_bytes = statistics_.live_requested_bytes;
  return id;
}

bool M0FullGpuResourceOwnership::Retire(uintptr_t handle, uint64_t generation) {
  std::lock_guard<std::mutex> lock{mutex_};
  auto it = live_.find(handle);
  if (it == live_.end() || !generation || it->second.generation != generation) return false;
  statistics_.live_requested_bytes -= it->second.requested_bytes;
  --statistics_.live_allocations;
  ++statistics_.retirements;
  live_.erase(it);
  return true;
}

M0FullGpuResourceStatisticsV1 M0FullGpuResourceOwnership::Statistics() const {
  std::lock_guard<std::mutex> lock{mutex_};
  return statistics_;
}

bool M0FullGpuResourceOwnership::Describe(uintptr_t handle, M0FullGpuAllocationIdentity& out) const {
  out = {};
  std::lock_guard<std::mutex> lock{mutex_};
  auto it = live_.find(handle);
  if (it == live_.end()) return false;
  out = it->second;
  return true;
}

bool M0FullGpuResourceOwnership::Claim(uintptr_t sidecar, uint64_t generation,
                                     uint64_t session, uint64_t run, uint32_t role,
                                     uintptr_t data_handle, uint64_t data_generation) {
  return ClaimImpl(sidecar, generation, session, run, role, data_handle, data_generation, 0);
}

bool M0FullGpuResourceOwnership::Claim(uintptr_t sidecar, uint64_t generation,
                                     uint64_t session, uint64_t run, uint32_t role,
                                     uintptr_t data_handle, uint64_t data_generation,
                                     uint32_t producer_instance) {
  if (!m0_extent_abi2::CertifiedProducer(role, producer_instance)) return false;
  return ClaimImpl(sidecar, generation, session, run, role, data_handle, data_generation, producer_instance);
}

bool M0FullGpuResourceOwnership::ClaimImpl(uintptr_t sidecar, uint64_t generation,
                                         uint64_t session, uint64_t run, uint32_t role,
                                         uintptr_t data_handle, uint64_t data_generation,
                                         uint32_t producer_instance) {
  if (Poisoned() || !session || !run || !role || sidecar == data_handle) return false;
  std::lock_guard<std::mutex> lock{mutex_};
  auto s = live_.find(sidecar), d = live_.find(data_handle);
  if (s == live_.end() || d == live_.end() || !generation || !data_generation ||
      s->second.generation != generation || d->second.generation != data_generation) return false;
  auto& identity = s->second;
  if (identity.run) {
    return identity.session == session && identity.run == run && identity.producer_role == role &&
           identity.producer_instance == producer_instance &&
           identity.data_handle == data_handle && identity.data_generation == data_generation;
  }
  identity.session = session;
  identity.run = run;
  identity.producer_role = role;
  identity.producer_instance = producer_instance;
  identity.data_handle = data_handle;
  identity.data_generation = data_generation;
  return true;
}

bool M0FullGpuResourceOwnership::DescribeBindings(uintptr_t sidecar, uintptr_t data_handle,
                                                 M0FullGpuAllocationIdentity& sidecar_out,
                                                 M0FullGpuAllocationIdentity& data_out) const {
  sidecar_out = {}; data_out = {};
  if (!sidecar || !data_handle || sidecar == data_handle) return false;
  std::lock_guard<std::mutex> lock{mutex_};
  auto s = live_.find(sidecar), d = live_.find(data_handle);
  if (s == live_.end() || d == live_.end()) return false;
  sidecar_out = s->second; data_out = d->second;
  return true;
}

bool M0FullGpuResourceOwnership::DescribePair(uintptr_t sidecar, uintptr_t data_handle,
                                             M0FullGpuAllocationIdentity& out) const {
  out = {};
  if (!sidecar || !data_handle || sidecar == data_handle) return false;
  std::lock_guard<std::mutex> lock{mutex_};
  auto s = live_.find(sidecar), d = live_.find(data_handle);
  if (s == live_.end() || d == live_.end() || !s->second.run ||
      s->second.data_handle != data_handle || s->second.data_generation != d->second.generation) return false;
  out = s->second;
  return true;
}

bool M0FullGpuResourceOwnership::ValidatePair(uintptr_t sidecar, uint64_t generation,
                                             uint64_t session, uint64_t run, uint32_t role,
                                             uint32_t producer_instance, uintptr_t data_handle,
                                             uint64_t data_generation,
                                             M0FullGpuAllocationIdentity& out) const {
  out = {};
  M0FullGpuAllocationIdentity paired;
  if (Poisoned() || !session || !run || !generation || !data_generation ||
      !m0_extent_abi2::CertifiedProducer(role, producer_instance) ||
      !DescribePair(sidecar, data_handle, paired) || paired.generation != generation ||
      paired.session != session || paired.run != run || paired.producer_role != role ||
      paired.producer_instance != producer_instance || paired.data_generation != data_generation) return false;
  out = paired;
  return true;
}
}  // namespace onnxruntime::webgpu
