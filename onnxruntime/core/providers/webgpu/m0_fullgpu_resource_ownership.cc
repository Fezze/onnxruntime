#include "core/providers/webgpu/m0_fullgpu_resource_ownership.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include <atomic>
#include <limits>
#include <stdexcept>
#include "core/common/logging/logging.h"

namespace onnxruntime::webgpu {
namespace {
std::atomic<uint64_t> next_allocation{1};
void AddObserved(uint64_t& total, uint64_t amount, bool& saturated) {
  if(amount>std::numeric_limits<uint64_t>::max()-total){total=std::numeric_limits<uint64_t>::max();saturated=true;}
  else total+=amount;
}
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
  AddObserved(statistics_.total_requested_bytes,requested_bytes,statistics_.counters_saturated);
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
  AddObserved(statistics_.total_retired_bytes,it->second.requested_bytes,statistics_.counters_saturated);
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

uint64_t M0FullGpuResourceOwnership::RecordWorkload(M0WorkloadEventV1 event,uint64_t value,uint64_t auxiliary) {
  std::lock_guard<std::mutex> lock{mutex_};
  auto& s=statistics_;
  auto add=[&](uint64_t& field,uint64_t amount){AddObserved(field,amount,s.counters_saturated);};
  switch(event){
    case M0WorkloadEventV1::Upload:add(s.upload_payload_bytes,value);add(s.upload_copy_bytes,auxiliary);return s.upload_payload_bytes;
    case M0WorkloadEventV1::Download:add(s.download_payload_bytes,value);add(s.download_copy_bytes,auxiliary);return s.download_payload_bytes;
    case M0WorkloadEventV1::DeviceCopy:add(s.device_copy_bytes,value);return s.device_copy_bytes;
    case M0WorkloadEventV1::OperatorStarted:add(s.operators_started,1);return s.operators_started;
    case M0WorkloadEventV1::OperatorCompleted:add(s.operators_completed,1);return s.operators_completed;
    case M0WorkloadEventV1::Dispatch:add(s.dispatches_encoded,1);add(s.workgroups_encoded,value);s.counters_saturated=s.counters_saturated||auxiliary!=0;return s.dispatches_encoded;
    case M0WorkloadEventV1::IndirectDispatch:add(s.dispatches_encoded,1);add(s.indirect_dispatches_encoded,1);return s.dispatches_encoded;
  }
  return 0;
}
void M0FullGpuResourceOwnership::LogWorkload(const char* phase) const {
  const auto s=Statistics();
  LOGS_DEFAULT(INFO)<<"M0_RW07_WORKLOAD phase="<<phase
    <<" live_requested_bytes="<<s.live_requested_bytes<<" peak_live_requested_bytes="<<s.peak_live_requested_bytes
    <<" total_requested_bytes="<<s.total_requested_bytes<<" total_retired_bytes="<<s.total_retired_bytes
    <<" upload_payload_bytes="<<s.upload_payload_bytes<<" upload_copy_bytes="<<s.upload_copy_bytes
    <<" download_payload_bytes="<<s.download_payload_bytes<<" download_copy_bytes="<<s.download_copy_bytes
    <<" device_copy_bytes="<<s.device_copy_bytes<<" operators_started="<<s.operators_started
    <<" operators_completed="<<s.operators_completed<<" dispatches_encoded="<<s.dispatches_encoded
    <<" workgroups_encoded="<<s.workgroups_encoded<<" indirect_dispatches_encoded="<<s.indirect_dispatches_encoded
    <<" counters_saturated="<<s.counters_saturated
    <<" physical_RAM_traffic_measured=0 shader_memory_accesses_measured=0 FLOPs_measured=0";
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
